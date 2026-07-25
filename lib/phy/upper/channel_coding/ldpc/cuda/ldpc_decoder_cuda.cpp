// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ldpc_decoder_cuda.h"
#include "cuda_rt_utils.h"
#include "ocudu/support/ocudu_assert.h"
#include "fmt/format.h"
#include <algorithm>
#include <cmath>

using namespace ocudu;

namespace {

/// Convert logical bit index to the physical bit position within a uint32_t word,
/// using MSB-first-per-byte format expected by CUDA GPU kernels.
inline unsigned cuda_bit_pos(unsigned bit_idx)
{
  unsigned bit_in_word  = bit_idx % 32;
  unsigned byte_in_word = bit_in_word / 8;
  unsigned bit_in_byte  = 7 - (bit_in_word % 8);
  return byte_in_word * 8 + bit_in_byte;
}

} // namespace

ldpc_decoder_cuda::ldpc_decoder_cuda()
{
  // Initialize CUDA library.
  nr_ldpc_status_t status = ocudu_phy_cuda_init();
  ocudu_assert(status == NR_LDPC_SUCCESS, "Failed to initialize CUDA library: {}", nr_ldpc_get_error_string(status));

  // Create CUDA stream.
  cudaError_t cuda_status = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  ocudu_assert(cuda_status == cudaSuccess, "Failed to create CUDA stream: {}", cudaGetErrorString(cuda_status));

  // Create CUDA decoder.
  status = ldpc_decoder_create(&decoder_handle_);
  ocudu_assert(status == NR_LDPC_SUCCESS, "Failed to create CUDA decoder: {}", nr_ldpc_get_error_string(status));

  // Create CUDA rate dematcher.
  status = rate_matcher_create(&rate_dematcher_handle_);
  ocudu_assert(
      status == NR_LDPC_SUCCESS, "Failed to create CUDA rate dematcher: {}", nr_ldpc_get_error_string(status));

  // Allocate device memory for LLR input.
  cuda_status = cudaMalloc(&d_llr_input_, MAX_LLR_SIZE * sizeof(float));
  ocudu_assert(cuda_status == cudaSuccess,
               "Failed to allocate device memory for LLR input: {}",
               cudaGetErrorString(cuda_status));

  // Allocate device memory for de-rate matched LLRs.
  cuda_status = cudaMalloc(&d_llr_dematched_, MAX_LLR_SIZE * sizeof(float));
  ocudu_assert(cuda_status == cudaSuccess,
               "Failed to allocate device memory for de-rate matched LLRs: {}",
               cudaGetErrorString(cuda_status));

  // Allocate device memory for output bits.
  cuda_status = cudaMalloc(&d_output_bits_, MAX_OUTPUT_WORDS * sizeof(uint32_t));
  ocudu_assert(cuda_status == cudaSuccess,
               "Failed to allocate device memory for output bits: {}",
               cudaGetErrorString(cuda_status));

  // Allocate host memory.
  h_output_bits_.resize(MAX_OUTPUT_WORDS);
  h_llr_float_.resize(MAX_LLR_SIZE);
}

ldpc_decoder_cuda::~ldpc_decoder_cuda()
{
  // Free device memory.
  if (d_llr_input_) {
    cudaFree(d_llr_input_);
  }
  if (d_llr_dematched_) {
    cudaFree(d_llr_dematched_);
  }
  if (d_output_bits_) {
    cudaFree(d_output_bits_);
  }

  // Destroy CUDA handles.
  if (rate_dematcher_handle_) {
    rate_matcher_destroy(rate_dematcher_handle_);
  }
  if (decoder_handle_) {
    ldpc_decoder_destroy(decoder_handle_);
  }

  // Destroy CUDA stream.
  if (stream_) {
    cudaStreamDestroy(stream_);
  }

  // Note: We intentionally do NOT call ocudu_phy_cuda_cleanup() here.
  // The library should remain initialized until program exit to avoid
  // destroying the CUDA context while other CUDA instances are active.
}

void ldpc_decoder_cuda::configure_decoder(const configuration& cfg)
{
  // Check if reconfiguration is needed.
  unsigned lifting_size = static_cast<unsigned>(cfg.lifting_size);
  unsigned filler_bits  = cfg.nof_filler_bits;
  if (cached_base_graph_ == cfg.base_graph && cached_lifting_size_ == lifting_size &&
      cached_filler_bits_ == filler_bits) {
    // No reconfiguration needed - same configuration.
    return;
  }

  // Determine base graph number (1 or 2).
  int bg = (cfg.base_graph == ldpc_base_graph_type::BG1) ? 1 : 2;

  // Compute LDPC configuration.
  nr_ldpc_config_t ldpc_cfg  = {};
  ldpc_cfg.base_graph        = bg;
  ldpc_cfg.lifting_size      = static_cast<int>(lifting_size);
  ldpc_cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(ldpc_cfg.lifting_size);

  // Compute information and parity bits based on base graph.
  // CUDA convention: num_info_bits = Kd = K - F (excluding filler bits).
  int F = static_cast<int>(cfg.nof_filler_bits);
  if (bg == 1) {
    ldpc_cfg.num_info_bits    = 22 * ldpc_cfg.lifting_size - F; // Kd = K - F
    ldpc_cfg.num_parity_bits  = 46 * ldpc_cfg.lifting_size;
    ldpc_cfg.max_parity_nodes = 46;
  } else {
    ldpc_cfg.num_info_bits    = 10 * ldpc_cfg.lifting_size - F; // Kd = K - F
    ldpc_cfg.num_parity_bits  = 42 * ldpc_cfg.lifting_size;
    ldpc_cfg.max_parity_nodes = 42;
  }
  int K                       = ldpc_cfg.num_info_bits + F;
  ldpc_cfg.num_codeword_bits  = K + ldpc_cfg.num_parity_bits;
  ldpc_cfg.num_filler_bits    = F;
  ldpc_cfg.puncture           = true;
  ldpc_cfg.redundancy_version = 0; // RV handled elsewhere in ocudu-foss

  // Configure decoder parameters.
  ldpc_decoder_params_t dec_params;
  ldpc_decoder_params_init(&dec_params);
  dec_params.max_iterations = static_cast<int>(cfg.max_iterations);
  // Release default for CUDA min-sum decoding. Keep this fixed unless a public config knob is added.
  dec_params.min_sum_scale            = 0.8f;
  dec_params.early_termination        = true;
  dec_params.auto_scale               = true; // Use CUDA's rate-adaptive scaling tables.
  dec_params.deferred_iteration_stats = true; // Enable async iteration tracking

  // Configure the CUDA decoder.
  nr_ldpc_status_t status = ldpc_decoder_configure(decoder_handle_, &ldpc_cfg, &dec_params);
  ocudu_assert(status == NR_LDPC_SUCCESS, "Failed to configure CUDA decoder: {}", nr_ldpc_get_error_string(status));

  // Cache configuration.
  cached_base_graph_    = cfg.base_graph;
  cached_lifting_size_  = lifting_size;
  cached_filler_bits_   = filler_bits;
  cached_nof_info_bits_ = static_cast<unsigned>(ldpc_cfg.num_info_bits);
}

void ldpc_decoder_cuda::convert_llrs_to_float(span<const log_likelihood_ratio> llrs, unsigned count)
{
  // Convert int8 LLRs to float.
  // CUDA expects float LLRs in range roughly [-32, 32].
  for (unsigned i = 0; i < count; ++i) {
    h_llr_float_[i] = static_cast<float>(llrs[i].to_value_type());
  }

  // Zero-pad remaining LLRs if needed.
  for (unsigned i = count; i < h_llr_float_.size() && i < MAX_LLR_SIZE; ++i) {
    h_llr_float_[i] = 0.0f;
  }
}

void ldpc_decoder_cuda::extract_decoded_bits(bit_buffer& output, unsigned nof_bits)
{
  // Copy output bits from device to host.
  unsigned    nof_words = (nof_bits + 31) / 32;
  cudaError_t status    = cudaMemcpyAsync(
      h_output_bits_.data(), d_output_bits_, nof_words * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_);
  ocudu_assert(status == cudaSuccess, "Failed to copy decoded bits from device: {}", cudaGetErrorString(status));

  // Synchronize to ensure transfer is complete.
  cudaStreamSynchronizeYielding(stream_);

  // Extract bits into the output bit_buffer. CUDA writes hard decisions in MSB-first-per-byte order.
  unsigned nof_full_bytes = nof_bits / 8;
  for (unsigned byte_idx = 0; byte_idx < nof_full_bytes; ++byte_idx) {
    uint8_t  byte_val = 0;
    unsigned base_bit = byte_idx * 8;
    for (unsigned j = 0; j < 8; ++j) {
      unsigned src_bit_idx = base_bit + j;
      unsigned src_word    = src_bit_idx / 32;
      unsigned src_bit     = cuda_bit_pos(src_bit_idx);
      uint8_t  bit         = (h_output_bits_[src_word] >> src_bit) & 1;
      byte_val |= (bit << (7 - j));
    }
    output.set_byte(byte_val, byte_idx);
  }

  // Handle remaining bits.
  unsigned remaining_bits = nof_bits % 8;
  if (remaining_bits > 0) {
    unsigned base_bit = nof_full_bytes * 8;
    for (unsigned j = 0; j < remaining_bits; ++j) {
      unsigned src_bit_idx = base_bit + j;
      unsigned src_word    = src_bit_idx / 32;
      unsigned src_bit     = cuda_bit_pos(src_bit_idx);
      unsigned bit         = (h_output_bits_[src_word] >> src_bit) & 1;
      output.insert(bit, base_bit + j, 1);
    }
  }
}

std::optional<unsigned> ldpc_decoder_cuda::decode(bit_buffer&                      output,
                                                     span<const log_likelihood_ratio> input,
                                                     crc_calculator*                  crc,
                                                     const configuration&             cfg)
{
  unsigned lifting_size   = static_cast<unsigned>(cfg.lifting_size);
  unsigned bg_K           = (cfg.base_graph == ldpc_base_graph_type::BG1) ? 22 : 10;
  unsigned bg_N           = (cfg.base_graph == ldpc_base_graph_type::BG1) ? 68 : 52;
  unsigned message_length = bg_K * lifting_size;
  unsigned puncture_size  = 2 * lifting_size;

  // Full codeblock size for CUDA (including punctured positions).
  unsigned N_full = bg_N * lifting_size;

  ocudu_assert(output.size() == message_length,
               "Output size {} does not match message length {}.",
               output.size(),
               message_length);

  // Find the last non-zero LLR to determine actual input size.
  auto     last = std::find_if(input.rbegin(), input.rend(), [](const log_likelihood_ratio& llr) { return llr != 0; });
  unsigned input_size = std::distance(input.begin(), last.base());

  // Check if there are enough LLRs.
  // The minimum required is (K-2)*Z + some parity.
  unsigned min_input = (bg_K - 2) * lifting_size + 2 * lifting_size;
  if (input_size < min_input) {
    if (crc == nullptr) {
      output.one();
    }
    return std::nullopt;
  }

  // Configure the CUDA decoder if needed.
  configure_decoder(cfg);

  // srsRAN provides shortened LLRs (N - 2*Z) without the punctured systematic bits.
  // CUDA expects full codeblock LLRs (N) with the first 2*Z positions being punctured.
  // We need to pad the input with 2*Z zero LLRs at the beginning.
  //
  // LLR conversion: srsRAN uses int8 LLRs in range [-120, 120].
  // CUDA's decoder clamps LLRs internally (default clamp = 32.0).
  // We pass the raw LLR values and let CUDA clamp them.

  // Clear the LLR buffer first.
  std::fill(h_llr_float_.begin(), h_llr_float_.begin() + N_full, 0.0f);

  // Pad the first 2*Z positions with zeros (punctured systematic bits have unknown LLRs = 0).
  // Then copy the shortened input LLRs starting at position 2*Z, without scaling.
  // srsRAN uses positive LLR = bit 0 likely, negative LLR = bit 1 likely.
  // CUDA uses the same convention based on simple_decoder_test.
  for (size_t i = 0; i < input.size(); ++i) {
    h_llr_float_[puncture_size + i] = static_cast<float>(input[i].to_value_type());
  }

  // Copy full LLRs (N values) to device.
  cudaError_t cuda_status =
      cudaMemcpyAsync(d_llr_input_, h_llr_float_.data(), N_full * sizeof(float), cudaMemcpyHostToDevice, stream_);
  ocudu_assert(cuda_status == cudaSuccess, "Failed to copy LLRs to device: {}", cudaGetErrorString(cuda_status));

  // Note: CUDA decoder already clears the output buffer before decoding.

  // Run LDPC decoding on GPU.
  // CUDA decoder expects full codeblock LLRs (N values).
  nr_ldpc_status_t status = ldpc_decoder_decode(decoder_handle_, d_llr_input_, d_output_bits_, stream_);
  ocudu_assert(status == NR_LDPC_SUCCESS, "CUDA decode failed: {}", nr_ldpc_get_error_string(status));

  // Synchronize stream.
  cudaStreamSynchronizeYielding(stream_);

  // Extract decoded bits.
  // CUDA outputs K*Z information bits (including the punctured 2*Z systematic bits).
  // We need all K*Z bits for the output.
  extract_decoded_bits(output, message_length);

  // Compute number of significant bits (excluding filler bits).
  unsigned nof_significant_bits = message_length - cfg.nof_filler_bits;

  // Check CRC if provided.
  if (crc != nullptr) {
    if (crc->calculate(output.first(nof_significant_bits)) == 0) {
      // CRC passed - return iteration count.
      // Note: CUDA doesn't expose iteration count per codeblock,
      // so we return a nominal value. This could be enhanced.
      return 1;
    }
    // CRC failed.
    return std::nullopt;
  }

  // No CRC check - return without iteration count.
  return std::nullopt;
}
