// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "pusch_codeblock_decoder_cuda_batch.h"
#include "../ldpc_graph_impl.h"
#include "../ldpc_luts_impl.h"
#include "cuda_rt_utils.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/support/ocudu_assert.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>

// CUDA headers for GPU CRC batch check
namespace {

bool is_env_enabled(const char* name)
{
  const char* value = std::getenv(name);
  if (!value) {
    return false;
  }
  return (std::strcmp(value, "1") == 0) || (std::strcmp(value, "true") == 0) || (std::strcmp(value, "TRUE") == 0) ||
         (std::strcmp(value, "on") == 0) || (std::strcmp(value, "ON") == 0);
}

bool env_value_is_disabled(const char* value)
{
  return value != nullptr &&
         ((std::strcmp(value, "0") == 0) || (std::strcmp(value, "false") == 0) || (std::strcmp(value, "FALSE") == 0) ||
          (std::strcmp(value, "off") == 0) || (std::strcmp(value, "OFF") == 0));
}

bool env_value_is_enabled(const char* value)
{
  return value != nullptr &&
         ((std::strcmp(value, "1") == 0) || (std::strcmp(value, "true") == 0) || (std::strcmp(value, "TRUE") == 0) ||
          (std::strcmp(value, "on") == 0) || (std::strcmp(value, "ON") == 0));
}

unsigned get_env_unsigned_or_default(const char* name, unsigned default_value)
{
  const char* value = std::getenv(name);
  if ((value == nullptr) || (*value == '\0')) {
    return default_value;
  }

  char*         end_ptr = nullptr;
  unsigned long parsed  = std::strtoul(value, &end_ptr, 10);
  if (end_ptr == value) {
    return default_value;
  }
  return static_cast<unsigned>(parsed);
}

bool select_ldpc_boxplus_algorithm(const std::string& algorithm, unsigned nof_cbs)
{
  bool use_boxplus = true;
  if (algorithm == "min_sum") {
    use_boxplus = false;
  } else if ((algorithm == "auto") || algorithm.empty()) {
    // Profiling on GB10 shows box-plus is faster for smaller batches, while
    // the half2 min-sum path wins once the TB has enough codeblocks.
    static const unsigned auto_boxplus_max_cbs = get_env_unsigned_or_default("OCUDU_LDPC_BOXPLUS_MAX_CBS", 192);
    use_boxplus                                = nof_cbs < auto_boxplus_max_cbs;
  }

  // Env override for test binaries without YAML config.
  const char* boxplus_env = std::getenv("OCUDU_LDPC_BOXPLUS");
  if (env_value_is_disabled(boxplus_env)) {
    use_boxplus = false;
  } else if (env_value_is_enabled(boxplus_env)) {
    use_boxplus = true;
  }

  return use_boxplus;
}

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

extern "C" {
#include "transport_block.h"
}

using namespace ocudu;

/// Helper to get modulation order Q_m from modulation scheme.
static unsigned get_mod_order(modulation_scheme mod)
{
  switch (mod) {
    case modulation_scheme::BPSK:
    case modulation_scheme::PI_2_BPSK:
      return 1;
    case modulation_scheme::QPSK:
      return 2;
    case modulation_scheme::QAM16:
      return 4;
    case modulation_scheme::QAM64:
      return 6;
    case modulation_scheme::QAM256:
      return 8;
    default:
      return 2;
  }
}

/// Helper to convert srsRAN CRC polynomial to CUDA CRC type.
static crc_type_t to_crc_type(crc_generator_poly poly)
{
  switch (poly) {
    case crc_generator_poly::CRC24A:
      return CRC_TYPE_24A;
    case crc_generator_poly::CRC24B:
      return CRC_TYPE_24B;
    case crc_generator_poly::CRC16:
      return CRC_TYPE_16;
    default:
      return CRC_TYPE_24B; // Default to CRC24B for codeblocks
  }
}

pusch_codeblock_decoder_cuda_batch::pusch_codeblock_decoder_cuda_batch(
    std::unique_ptr<ldpc_rate_dematcher> rate_dematcher,
    sch_crc                              crcs) :
  dematcher_(std::move(rate_dematcher)),
  crc_set_({std::move(crcs.crc16), std::move(crcs.crc24A), std::move(crcs.crc24B)})
{
  ocudu_assert(dematcher_, "Invalid rate dematcher.");
  ocudu_assert(crc_set_.crc16, "Invalid CRC16 calculator.");
  ocudu_assert(crc_set_.crc24A, "Invalid CRC24A calculator.");
  ocudu_assert(crc_set_.crc24B, "Invalid CRC24B calculator.");

  // Initialize CUDA library.
  nr_ldpc_status_t status = ocudu_phy_cuda_init();
  ocudu_assert(
      status == NR_LDPC_SUCCESS, "Failed to initialize CUDA library: {}", nr_ldpc_get_error_string(status));

  // Configure CUDA to yield CPU when waiting for GPU operations.
  // This is critical for compatibility with real-time (SCHED_FIFO) threads.
  cudaSetDeviceFlags(cudaDeviceScheduleYield);

  cudaError_t cuda_status = ocudu::cudaStreamCreateUpperPhy(&stream_);
  ocudu_assert(cuda_status == cudaSuccess, "Failed to create CUDA stream: {}", cudaGetErrorString(cuda_status));

  // Create CUDA decoder.
  status = ldpc_decoder_create(&decoder_handle_);
  ocudu_assert(status == NR_LDPC_SUCCESS, "Failed to create CUDA decoder: {}", nr_ldpc_get_error_string(status));

  // Create CUDA rate matcher for GPU dematching.
  status = rate_matcher_create(&rate_matcher_handle_);
  ocudu_assert(
      status == NR_LDPC_SUCCESS, "Failed to create CUDA rate matcher: {}", nr_ldpc_get_error_string(status));

  // Create CUDA scrambler for GPU descrambling.
  status = scrambler_create(&scrambler_handle_);
  ocudu_assert(
      status == NR_LDPC_SUCCESS, "Failed to create CUDA scrambler: {}", nr_ldpc_get_error_string(status));

  // Allocate device memory for received LLRs (rate-matched, before dematching).
  size_t received_batch_size = MAX_BATCH_SIZE * MAX_RM_LLRS * sizeof(float);
  cuda_status                = cudaMalloc(&d_received_llr_batch_, received_batch_size);
  ocudu_assert(
      cuda_status == cudaSuccess, "Failed to allocate device received LLR memory: {}", cudaGetErrorString(cuda_status));

  // Allocate device memory for deinterleaved LLRs (after GPU deinterleaving, before dematching).
  cuda_status = cudaMalloc(&d_deinterleaved_llr_batch_, received_batch_size);
  ocudu_assert(cuda_status == cudaSuccess,
               "Failed to allocate device deinterleaved LLR memory: {}",
               cudaGetErrorString(cuda_status));

  // Allocate device memory for dematched LLRs (full codeblock size, fp32).
  size_t llr_batch_size = MAX_BATCH_SIZE * MAX_CB_LLRS * sizeof(float);
  cuda_status           = cudaMalloc(&d_llr_batch_, llr_batch_size);
  ocudu_assert(
      cuda_status == cudaSuccess, "Failed to allocate device LLR batch memory: {}", cudaGetErrorString(cuda_status));

  // Allocate device memory for dematched LLRs (fp16, triple-buffered for GPU-resident pipeline).
  size_t llr_batch_half_size = MAX_BATCH_SIZE * MAX_CB_LLRS * sizeof(uint16_t);
  for (int i = 0; i < NUM_DECODE_BUFFERS; i++) {
    cuda_status = cudaMalloc(&d_llr_batch_half_[i], llr_batch_half_size);
    ocudu_assert(cuda_status == cudaSuccess,
                 "Failed to allocate device fp16 LLR batch memory [{}]: {}",
                 i,
                 cudaGetErrorString(cuda_status));
  }

  // Allocate device memory for batched output bits.
  size_t output_batch_size = MAX_BATCH_SIZE * MAX_CB_OUTPUT_WORDS * sizeof(uint32_t);
  cuda_status              = cudaMalloc(&d_output_batch_, output_batch_size);
  ocudu_assert(
      cuda_status == cudaSuccess, "Failed to allocate device output batch memory: {}", cudaGetErrorString(cuda_status));

  // Allocate device memory for CRC check results.
  cuda_status = cudaMalloc(&d_crc_results_, MAX_BATCH_SIZE * sizeof(int));
  ocudu_assert(
      cuda_status == cudaSuccess, "Failed to allocate device CRC results memory: {}", cudaGetErrorString(cuda_status));

  // Allocate device memory for per-CB bit counts.
  cuda_status = cudaMalloc(&d_bits_per_cb_, MAX_BATCH_SIZE * sizeof(int));
  ocudu_assert(
      cuda_status == cudaSuccess, "Failed to allocate device bits per CB memory: {}", cudaGetErrorString(cuda_status));

  // Allocate device memory for reassembled TB output.
  cuda_status = cudaMalloc(&d_tb_output_, MAX_TB_BYTES);
  ocudu_assert(
      cuda_status == cudaSuccess, "Failed to allocate device TB output memory: {}", cudaGetErrorString(cuda_status));

  // Allocate device memory for TB CRC result.
  cuda_status = cudaMalloc(&d_tb_crc_result_, sizeof(int));
  ocudu_assert(cuda_status == cudaSuccess,
               "Failed to allocate device TB CRC result memory: {}",
               cudaGetErrorString(cuda_status));

  // Allocate device memory for gather kernel source offsets (non-uniform CB path).
  cuda_status = cudaMalloc(&d_gather_offsets_, MAX_BATCH_SIZE * sizeof(unsigned int));
  ocudu_assert(
      cuda_status == cudaSuccess, "Failed to allocate device gather offsets: {}", cudaGetErrorString(cuda_status));

  // Allocate pinned host memory for gather kernel source offsets.
  cuda_status = cudaMallocHost(&h_gather_offsets_, MAX_BATCH_SIZE * sizeof(unsigned int));
  ocudu_assert(
      cuda_status == cudaSuccess, "Failed to allocate pinned gather offsets: {}", cudaGetErrorString(cuda_status));

  // Allocate host memory.
  h_received_llr_batch_.resize(MAX_BATCH_SIZE * MAX_RM_LLRS);
  h_llr_batch_.resize(MAX_BATCH_SIZE * MAX_CB_LLRS);
  h_output_batch_.resize(MAX_BATCH_SIZE * MAX_CB_OUTPUT_WORDS);
  h_bits_per_cb_.resize(MAX_BATCH_SIZE);

  // Create CUDA event for async completion tracking (no timing for minimum overhead).
  cuda_status = cudaEventCreateWithFlags(&completion_event_, cudaEventDisableTiming);
  ocudu_assert(cuda_status == cudaSuccess, "Failed to create completion event: {}", cudaGetErrorString(cuda_status));

  // Allocate pinned host memory for TB CRC result (required for async D2H).
  cuda_status = cudaMallocHost(&h_tb_crc_result_pinned_, sizeof(int));
  ocudu_assert(
      cuda_status == cudaSuccess, "Failed to allocate pinned TB CRC result: {}", cudaGetErrorString(cuda_status));

  // Allocate pinned host memory for per-CB CRC results.
  cuda_status = cudaMallocHost(&h_crc_results_pinned_, MAX_BATCH_SIZE * sizeof(int));
  ocudu_assert(
      cuda_status == cudaSuccess, "Failed to allocate pinned CB CRC results: {}", cudaGetErrorString(cuda_status));

  // Allocate pinned host memory for TB output staging (required for true async D2H).
  // Without this, cudaMemcpyAsync to unpinned memory blocks internally, defeating async.
  cuda_status = cudaMallocHost(&h_tb_output_pinned_, MAX_TB_BYTES);
  ocudu_assert(cuda_status == cudaSuccess, "Failed to allocate pinned TB output: {}", cudaGetErrorString(cuda_status));

  // Keep detailed CUDA timing instrumentation opt-in for the production latency path.
  // When enabled, each decode records multiple CUDA events and reads elapsed times
  // after completion.
  timing_enabled_ = is_env_enabled("OCUDU_PUSCH_ACCELERATION_TIMING");

  // Pre-allocate CUDA buffers for worst-case capacity to eliminate dynamic allocations during processing.
  // This prevents the 1.5 ms cudaMalloc overhead that dominates GPU latency.

  // Configure decoder with worst-case parameters (BG1, Z=384) to enable pre-allocation
  // for maximum possible buffer sizes. This will be reconfigured on first actual use.
  constexpr unsigned WORST_CASE_Z = 384;               // Maximum lifting size
  constexpr int      WORST_CASE_K = 22 * WORST_CASE_Z; // BG1 info bits

  nr_ldpc_config_t worst_case_cfg   = {};
  worst_case_cfg.base_graph         = 1; // BG1 (larger than BG2)
  worst_case_cfg.lifting_size       = WORST_CASE_Z;
  worst_case_cfg.lifting_set_index  = nr_ldpc_get_lifting_set_index(WORST_CASE_Z);
  worst_case_cfg.num_info_bits      = WORST_CASE_K;      // Maximum info bits (no filler)
  worst_case_cfg.num_parity_bits    = 46 * WORST_CASE_Z; // BG1 parity
  worst_case_cfg.max_parity_nodes   = 46;                // BG1
  worst_case_cfg.num_codeword_bits  = WORST_CASE_K + worst_case_cfg.num_parity_bits;
  worst_case_cfg.num_filler_bits    = 0; // No filler for worst-case sizing
  worst_case_cfg.puncture           = true;
  worst_case_cfg.redundancy_version = 0;

  ldpc_decoder_params_t worst_case_params;
  ldpc_decoder_params_init(&worst_case_params);
  // Worst-case warmup defaults used only to allocate and initialize the CUDA decoder path.
  worst_case_params.max_iterations        = 6;
  worst_case_params.early_termination     = true;
  worst_case_params.auto_scale            = true;
  worst_case_params.crc_early_termination = true;
  worst_case_params.crc_type              = LDPC_CRC_24B;
  // Conservative confidence threshold for the warmup configuration.
  worst_case_params.confidence_threshold = 0.15f;
  worst_case_params.use_boxplus          = false; // Default to min-sum, can be overridden via env var

  // Configure with worst-case parameters (will be reconfigured on first use with actual parameters)
  status = ldpc_decoder_configure(decoder_handle_, &worst_case_cfg, &worst_case_params);
  if (status != NR_LDPC_SUCCESS) {
    ocudulog::fetch_basic_logger("PHY").warning("Failed to configure LDPC decoder with worst-case parameters: {}",
                                                nr_ldpc_get_error_string(status));
  } else {
    ocudulog::fetch_basic_logger("PHY").debug("Configured LDPC decoder with worst-case parameters (BG1, Z={})",
                                              WORST_CASE_Z);
  }

  // Pre-allocate LDPC decoder workspace for max batch size with worst-case configuration.
  status = ldpc_decoder_preallocate_workspace(decoder_handle_, MAX_BATCH_SIZE);
  if (status != NR_LDPC_SUCCESS) {
    ocudulog::fetch_basic_logger("PHY").warning("Failed to pre-allocate LDPC decoder workspace for {} CBs: {}",
                                                MAX_BATCH_SIZE,
                                                nr_ldpc_get_error_string(status));
  } else {
    ocudulog::fetch_basic_logger("PHY").debug(
        "Pre-allocated LDPC decoder workspace for {} CBs (BG1, Z={})", MAX_BATCH_SIZE, WORST_CASE_Z);
  }

  // Configure rate matcher with worst-case parameters for pre-allocation
  nr_rate_match_config_t worst_case_rm_cfg = {};
  // Set worst-case rate matching parameters (maximum E for max capacity)
  worst_case_rm_cfg.E              = 15000; // Max rate matched output size
  worst_case_rm_cfg.Q_m            = 8;     // 256QAM modulation order
  worst_case_rm_cfg.rv             = 0;     // Redundancy version
  worst_case_rm_cfg.N_cb           = 25344; // BG1 circular buffer (N=66*Z=66*384)
  worst_case_rm_cfg.k0             = 0;     // Starting position in circular buffer
  worst_case_rm_cfg.limited_buffer = false;

  status = rate_matcher_configure_rx(rate_matcher_handle_, &worst_case_cfg, &worst_case_rm_cfg);
  if (status != NR_LDPC_SUCCESS) {
    ocudulog::fetch_basic_logger("PHY").warning("Failed to configure rate matcher with worst-case parameters: {}",
                                                nr_ldpc_get_error_string(status));
  } else {
    ocudulog::fetch_basic_logger("PHY").debug("Configured rate matcher with worst-case parameters (BG1, Z={})",
                                              WORST_CASE_Z);
  }

  // Pre-allocate scrambler sequence buffer for max LLR count.
  status = scrambler_preallocate_sequence(scrambler_handle_, MAX_BATCH_SIZE * MAX_RM_LLRS);
  if (status != NR_LDPC_SUCCESS) {
    ocudulog::fetch_basic_logger("PHY").warning("Failed to pre-allocate scrambler sequence for {} LLRs: {}",
                                                MAX_BATCH_SIZE * MAX_RM_LLRS,
                                                nr_ldpc_get_error_string(status));
  } else {
    ocudulog::fetch_basic_logger("PHY").debug("Pre-allocated scrambler sequence for {} LLRs",
                                              MAX_BATCH_SIZE * MAX_RM_LLRS);
  }

  // Initialize CRC tables (one-time constant memory write).
  crc_init_tables();

  // Reset cached configuration to force reconfiguration on first actual use.
  // This ensures the decoder gets properly configured with real parameters.
  cached_lifting_size_ = 0;
}

void pusch_codeblock_decoder_cuda_batch::ensure_timing_events()
{
  if (timing_events_[0] == nullptr) {
    for (unsigned i = 0; i < NUM_TIMING_EVENTS; ++i) {
      cudaEventCreate(&timing_events_[i]);
    }
  }
}

void pusch_codeblock_decoder_cuda_batch::destroy_timing_events()
{
  for (unsigned i = 0; i < NUM_TIMING_EVENTS; ++i) {
    if (timing_events_[i] != nullptr) {
      cudaEventDestroy(timing_events_[i]);
      timing_events_[i] = nullptr;
    }
  }
}

pusch_codeblock_decoder_cuda_batch::~pusch_codeblock_decoder_cuda_batch()
{
  // Free timing events.
  destroy_timing_events();

  // Free device memory.
  if (d_received_llr_batch_) {
    cudaFree(d_received_llr_batch_);
  }
  if (d_deinterleaved_llr_batch_) {
    cudaFree(d_deinterleaved_llr_batch_);
  }
  if (d_llr_batch_) {
    cudaFree(d_llr_batch_);
  }
  for (int i = 0; i < NUM_DECODE_BUFFERS; i++) {
    if (d_llr_batch_half_[i]) {
      cudaFree(d_llr_batch_half_[i]);
    }
  }
  if (d_output_batch_) {
    cudaFree(d_output_batch_);
  }
  if (d_crc_results_) {
    cudaFree(d_crc_results_);
  }
  if (d_bits_per_cb_) {
    cudaFree(d_bits_per_cb_);
  }
  if (d_tb_output_) {
    cudaFree(d_tb_output_);
  }
  if (d_tb_crc_result_) {
    cudaFree(d_tb_crc_result_);
  }
  if (d_gather_offsets_) {
    cudaFree(d_gather_offsets_);
  }
  if (h_gather_offsets_) {
    cudaFreeHost(h_gather_offsets_);
  }

  // Destroy CUDA scrambler.
  if (scrambler_handle_) {
    scrambler_destroy(scrambler_handle_);
  }

  // Destroy CUDA rate matcher.
  if (rate_matcher_handle_) {
    rate_matcher_destroy(rate_matcher_handle_);
  }

  // Destroy CUDA decoder.
  if (decoder_handle_) {
    ldpc_decoder_destroy(decoder_handle_);
  }

  // Destroy CUDA stream.
  if (stream_) {
    cudaStreamDestroy(stream_);
  }

  // Destroy completion event.
  if (completion_event_) {
    cudaEventDestroy(completion_event_);
  }

  // Free pinned TB CRC memory.
  if (h_tb_crc_result_pinned_) {
    cudaFreeHost(h_tb_crc_result_pinned_);
  }
  if (h_crc_results_pinned_) {
    cudaFreeHost(h_crc_results_pinned_);
  }

  // Free pinned TB output staging buffer.
  if (h_tb_output_pinned_) {
    cudaFreeHost(h_tb_output_pinned_);
  }
}

bool pusch_codeblock_decoder_cuda_batch::can_copy_payload_output_direct(span<uint8_t> payload_output)
{
  if (payload_output.empty() || payload_output.data() == nullptr) {
    return false;
  }

  if (cached_payload_output_ptr_ == payload_output.data() && cached_payload_output_bytes_ >= payload_output.size()) {
    return cached_payload_output_async_safe_;
  }

  cached_payload_output_ptr_        = payload_output.data();
  cached_payload_output_bytes_      = payload_output.size();
  cached_payload_output_async_safe_ = false;

  cudaPointerAttributes attrs  = {};
  cudaError_t           status = cudaPointerGetAttributes(&attrs, payload_output.data());
  if (status != cudaSuccess) {
    // Pageable host memory is reported as an invalid CUDA pointer. Clear the sticky error and use pinned staging.
    cudaGetLastError();
    return false;
  }

#if CUDART_VERSION >= 10000
  cached_payload_output_async_safe_ = (attrs.type == cudaMemoryTypeHost) || (attrs.type == cudaMemoryTypeManaged);
#else
  cached_payload_output_async_safe_ = (attrs.memoryType == cudaMemoryTypeHost) || (attrs.isManaged != 0);
#endif
  return cached_payload_output_async_safe_;
}

crc_calculator* pusch_codeblock_decoder_cuda_batch::select_crc(crc_generator_poly poly)
{
  switch (poly) {
    case crc_generator_poly::CRC16:
      return crc_set_.crc16.get();
    case crc_generator_poly::CRC24A:
      return crc_set_.crc24A.get();
    case crc_generator_poly::CRC24B:
      return crc_set_.crc24B.get();
    default:
      return nullptr;
  }
}

void pusch_codeblock_decoder_cuda_batch::configure_decoder(ldpc_base_graph_type base_graph,
                                                                 unsigned             lifting_size,
                                                                 unsigned             nof_ldpc_iterations,
                                                                 unsigned             nof_filler_bits,
                                                                 unsigned             nof_cbs,
                                                                 crc_generator_poly   crc_poly)
{
  const bool use_boxplus = select_ldpc_boxplus_algorithm(ldpc_decoder_algorithm_, nof_cbs);

  // Check if reconfiguration is needed.
  bool need_reconfig =
      (cached_base_graph_ != base_graph || cached_lifting_size_ != lifting_size ||
       cached_max_iters_ != nof_ldpc_iterations || cached_filler_bits_ != nof_filler_bits ||
       cached_nof_cbs_ != nof_cbs || cached_crc_poly_ != crc_poly || cached_use_boxplus_ != use_boxplus);

  if (!need_reconfig) {
    return;
  }

  // Compute LDPC configuration.
  int              bg        = (base_graph == ldpc_base_graph_type::BG1) ? 1 : 2;
  nr_ldpc_config_t ldpc_cfg  = {};
  ldpc_cfg.base_graph        = bg;
  ldpc_cfg.lifting_size      = static_cast<int>(lifting_size);
  ldpc_cfg.lifting_set_index = nr_ldpc_get_lifting_set_index(ldpc_cfg.lifting_size);

  // Compute dimensions based on base graph.
  // CUDA convention: num_info_bits = Kd = K - F (excluding filler bits).
  // The library internally computes K = num_info_bits + num_filler_bits.
  int K;
  if (bg == 1) {
    K                         = 22 * ldpc_cfg.lifting_size;
    ldpc_cfg.num_info_bits    = K - static_cast<int>(nof_filler_bits); // Kd
    ldpc_cfg.num_parity_bits  = 46 * ldpc_cfg.lifting_size;
    ldpc_cfg.max_parity_nodes = 46;
    llr_stride_               = 68 * lifting_size;
    output_stride_            = (K + 31) / 32; // Output stride based on K (full systematic)
  } else {
    K                         = 10 * ldpc_cfg.lifting_size;
    ldpc_cfg.num_info_bits    = K - static_cast<int>(nof_filler_bits); // Kd
    ldpc_cfg.num_parity_bits  = 42 * ldpc_cfg.lifting_size;
    ldpc_cfg.max_parity_nodes = 42;
    llr_stride_               = 52 * lifting_size;
    output_stride_            = (K + 31) / 32; // Output stride based on K (full systematic)
  }
  ldpc_cfg.num_codeword_bits  = K + ldpc_cfg.num_parity_bits;
  ldpc_cfg.num_filler_bits    = static_cast<int>(nof_filler_bits);
  ldpc_cfg.puncture           = true;
  ldpc_cfg.redundancy_version = 0; // Not used for decode

  // Configure decoder parameters.
  ldpc_decoder_params_t dec_params;
  ldpc_decoder_params_init(&dec_params);
  dec_params.max_iterations    = static_cast<int>(nof_ldpc_iterations);
  dec_params.early_termination = true;
  // Fixed NMS scale=0.75 tracks the CPU BLER waterfall more closely than 0.80
  // across the 1L1P PRB/MCS sweep, while keeping the same iteration budget.
  dec_params.auto_scale    = false;
  dec_params.min_sum_scale = 0.75f;
  // A small offset compensates the fully resident FP16 LLR handoff without adding work.
  dec_params.min_sum_offset           = 0.10f;
  dec_params.deferred_iteration_stats = true; // Enable async iteration tracking
  // LLR clamp for the min-sum decoder.
  dec_params.llr_clamp = 127.0f;
  // Syndrome tolerance for early termination (default: 0 = strict parity check).
  // With the layered FP16 kernel (min1/min2 algorithm), exact parity should be achieved.
  dec_params.syndrome_tolerance = 0;

  // CRC-based early termination for FP16 decoder.
  // Uses hybrid confidence + CRC check: fast confidence filter followed by CRC verification.
  // This matches CPU decoder behavior and achieves Iter=1-2 at high SNR.
  // Reduced threshold to 0.15 for more aggressive early exit (optimization).
  dec_params.crc_early_termination = true;
  // Release default for CUDA CRC-assisted early termination.
  dec_params.confidence_threshold = 0.15f;

  // The OCUDU_LDPC_* env var overrides below keep standalone benchmarks and integration tests configurable without
  // threading YAML-only options through every test factory.
  // Box-plus algorithm: binary tree exclusion product via min.xorsign.abs.f16x2 PTX.
  // In auto mode, use it below OCUDU_LDPC_BOXPLUS_MAX_CBS and switch to min-sum
  // for larger half2 batches where profiling shows lower latency.
  dec_params.use_boxplus = use_boxplus;

  // Optional iteration override for boxplus kernel testing.
  if (dec_params.use_boxplus) {
    const char* iters_env = std::getenv("OCUDU_LDPC_BOXPLUS_ITERS");
    if (iters_env) {
      dec_params.max_iterations = std::atoi(iters_env);
    }
    // Optional NMS scale override for boxplus tuning.
    // When set, uses multiplicative-only NMS (offset=0) with the given scale.
    const char* scale_env = std::getenv("OCUDU_LDPC_BOXPLUS_SCALE");
    if (scale_env) {
      dec_params.min_sum_scale  = std::atof(scale_env);
      dec_params.min_sum_offset = 0.0f;
    }
  }

  // General NMS scale override (works for both min-sum and boxplus).
  // When set, disables rate-adaptive auto_scale and uses the given value.
  // Usage: OCUDU_LDPC_SCALE=0.8 to force a fixed scale for comparisons.
  const char* ldpc_scale_env = std::getenv("OCUDU_LDPC_SCALE");
  if (ldpc_scale_env) {
    dec_params.auto_scale     = false;
    dec_params.min_sum_scale  = std::atof(ldpc_scale_env);
    dec_params.min_sum_offset = 0.0f;
  }

  // Optional decoder tuning overrides for controlled sensitivity experiments.
  // These are intentionally applied after scale overrides so they can be swept independently.
  const char* ldpc_offset_env = std::getenv("OCUDU_LDPC_OFFSET");
  if (ldpc_offset_env) {
    dec_params.min_sum_offset = std::atof(ldpc_offset_env);
  }
  const char* ldpc_clamp_env = std::getenv("OCUDU_LDPC_LLR_CLAMP");
  if (ldpc_clamp_env) {
    dec_params.llr_clamp = std::atof(ldpc_clamp_env);
  }

  // Determine CRC type based on number of code blocks:
  // - Single-CB: Use TB CRC (CRC16 for small TB, CRC24A for large TB)
  // - Multi-CB: Use CB CRC (CRC24B)
  if (nof_cbs == 1) {
    // Single-CB: check TB CRC type
    if (crc_poly == crc_generator_poly::CRC16) {
      dec_params.crc_type = LDPC_CRC_16;
    } else {
      // CRC24A for single-CB large TB
      dec_params.crc_type = LDPC_CRC_24A;
    }
  } else {
    // Multi-CB: always use CRC24B (per-codeblock CRC)
    dec_params.crc_type = LDPC_CRC_24B;
  }

  // Configure the CUDA decoder.
  nr_ldpc_status_t status = ldpc_decoder_configure(decoder_handle_, &ldpc_cfg, &dec_params);
  ocudu_assert(
      status == NR_LDPC_SUCCESS, "Failed to configure CUDA decoder: {}", nr_ldpc_get_error_string(status));

  // Cache configuration.
  cached_base_graph_   = base_graph;
  cached_lifting_size_ = lifting_size;
  cached_max_iters_    = nof_ldpc_iterations;
  cached_filler_bits_  = nof_filler_bits;
  cached_nof_cbs_      = nof_cbs;
  cached_crc_poly_     = crc_poly;
  cached_use_boxplus_  = use_boxplus;
}

void pusch_codeblock_decoder_cuda_batch::configure_rate_dematcher(const codeblock_metadata& cfg)
{
  const auto& tb_common   = cfg.tb_common;
  const auto& cb_specific = cfg.cb_specific;

  unsigned lifting_size = static_cast<unsigned>(tb_common.lifting_size);
  unsigned rv           = tb_common.rv;
  unsigned rm_length    = cb_specific.rm_length;
  unsigned nof_filler   = cb_specific.nof_filler_bits;

  // Get modulation order from scheme.
  unsigned Q_m;
  switch (tb_common.mod) {
    case modulation_scheme::BPSK:
    case modulation_scheme::PI_2_BPSK:
      Q_m = 1;
      break;
    case modulation_scheme::QPSK:
      Q_m = 2;
      break;
    case modulation_scheme::QAM16:
      Q_m = 4;
      break;
    case modulation_scheme::QAM64:
      Q_m = 6;
      break;
    case modulation_scheme::QAM256:
      Q_m = 8;
      break;
    default:
      Q_m = 2;
  }

  // Check if reconfiguration is needed. Use rate-matcher-specific cached BG/Z to avoid
  // stale-match bugs when configure_decoder updates the shared cached_base_graph_/lifting_size_.
  if (cached_rm_bg_ == tb_common.base_graph && cached_rm_Z_ == lifting_size && cached_rm_E_ == rm_length &&
      cached_rm_rv_ == rv && cached_rm_filler_ == nof_filler && cached_rm_Qm_ == Q_m) {
    rm_stride_ = rm_length;
    return;
  }

  // Determine base graph number.
  int bg = (tb_common.base_graph == ldpc_base_graph_type::BG1) ? 1 : 2;

  // Compute LDPC configuration.
  nr_ldpc_config_t ldpc_cfg   = {};
  ldpc_cfg.base_graph         = bg;
  ldpc_cfg.lifting_size       = static_cast<int>(lifting_size);
  ldpc_cfg.lifting_set_index  = nr_ldpc_get_lifting_set_index(ldpc_cfg.lifting_size);
  ldpc_cfg.redundancy_version = static_cast<int>(rv);

  // Configure LDPC dimensions based on base graph.
  // num_info_bits = Kd = K - F (actual info bits, excluding filler).
  // This tells the rate dematcher where the filler "hole" starts in the codeword.
  // The filler kernel will place filler LLRs at positions [puncture_offset + Kd, puncture_offset + Kd + F).
  if (bg == 1) {
    int K                     = 22 * ldpc_cfg.lifting_size;
    ldpc_cfg.num_info_bits    = K - static_cast<int>(nof_filler); // Kd = K - F
    ldpc_cfg.num_parity_bits  = 46 * ldpc_cfg.lifting_size;
    ldpc_cfg.max_parity_nodes = 46;
  } else {
    int K                     = 10 * ldpc_cfg.lifting_size;
    ldpc_cfg.num_info_bits    = K - static_cast<int>(nof_filler); // Kd = K - F
    ldpc_cfg.num_parity_bits  = 42 * ldpc_cfg.lifting_size;
    ldpc_cfg.max_parity_nodes = 42;
  }
  // num_codeword_bits = K + parity (full codeword including filler positions)
  int K_total                = (bg == 1) ? 22 * ldpc_cfg.lifting_size : 10 * ldpc_cfg.lifting_size;
  ldpc_cfg.num_codeword_bits = K_total + ldpc_cfg.num_parity_bits;
  ldpc_cfg.num_filler_bits   = static_cast<int>(nof_filler);
  ldpc_cfg.puncture          = true;

  // Compute rate matching configuration.
  nr_rate_match_config_t rm_cfg = {};
  rm_cfg.E                      = static_cast<int>(rm_length);
  rm_cfg.Q_m                    = static_cast<int>(Q_m);
  rm_cfg.rv                     = static_cast<int>(rv);

  // Circular buffer size.
  int N_short           = ldpc_cfg.num_codeword_bits - 2 * ldpc_cfg.lifting_size;
  rm_cfg.N_cb           = (tb_common.Nref > 0) ? std::min(static_cast<int>(tb_common.Nref), N_short) : N_short;
  rm_cfg.k0             = rate_matcher_compute_k0(bg, ldpc_cfg.lifting_size, static_cast<int>(rv), rm_cfg.N_cb);
  rm_cfg.limited_buffer = (tb_common.Nref > 0);

  // Configure rate dematcher for RX direction.
  nr_ldpc_status_t status = rate_matcher_configure_rx(rate_matcher_handle_, &ldpc_cfg, &rm_cfg);
  ocudu_assert(
      status == NR_LDPC_SUCCESS, "Failed to configure CUDA rate dematcher: {}", nr_ldpc_get_error_string(status));

  // Cache configuration (rate-matcher-specific BG/Z to avoid shared-state bugs).
  cached_rm_bg_     = tb_common.base_graph;
  cached_rm_Z_      = lifting_size;
  cached_rm_E_      = rm_length;
  cached_rm_rv_     = rv;
  cached_rm_filler_ = nof_filler;
  cached_rm_Qm_     = rm_cfg.Q_m;
  rm_stride_        = rm_length;
}

void pusch_codeblock_decoder_cuda_batch::rate_dematch_only(span<const log_likelihood_ratio> cb_llrs,
                                                                 span<log_likelihood_ratio>       rm_buffer,
                                                                 bool                             new_data,
                                                                 const codeblock_metadata&        metadata)
{
  dematcher_->rate_dematch(rm_buffer, cb_llrs, new_data, metadata);
}

void pusch_codeblock_decoder_cuda_batch::extract_decoded_bits(bit_buffer& output,
                                                                    unsigned    cb_idx,
                                                                    unsigned    nof_bits)
{
  // Get pointer to this CB's output in the batch buffer.
  // LDPC kernel writes hard decisions in MSB-first-per-byte order directly.
  // Just extract bytes — no bit reversal needed.
  const uint32_t* cb_output = h_output_batch_.data() + cb_idx * output_stride_;

  unsigned nof_full_bytes = nof_bits / 8;
  unsigned bit_idx        = 0;

  for (unsigned byte_idx = 0; byte_idx < nof_full_bytes; ++byte_idx) {
    unsigned src_word = bit_idx / 32;
    unsigned src_bit  = bit_idx % 32;

    uint8_t byte_val;
    if (src_bit + 8 <= 32) {
      // Fast path: entire byte fits in one uint32_t word
      byte_val = (cb_output[src_word] >> src_bit) & 0xFF;
    } else {
      // Slow path: byte spans two uint32_t words (at word boundary)
      unsigned lo_bits = 32 - src_bit;
      byte_val         = (cb_output[src_word] >> src_bit) & ((1u << lo_bits) - 1);
      byte_val |= (cb_output[src_word + 1] & ((1u << (8 - lo_bits)) - 1)) << lo_bits;
    }
    output.set_byte(byte_val, byte_idx);
    bit_idx += 8;
  }

  // Handle remaining bits.
  unsigned remaining_bits = nof_bits % 8;
  if (remaining_bits > 0) {
    for (unsigned j = 0; j < remaining_bits; ++j) {
      unsigned bit_pos = bit_idx + j;
      unsigned bit     = (cb_output[bit_pos / 32] >> cuda_bit_pos(bit_pos)) & 1;
      output.insert(bit, bit_pos, 1);
    }
  }
}

std::vector<pusch_codeblock_decoder_cuda_batch::codeblock_result>
pusch_codeblock_decoder_cuda_batch::decode_batch(span<const described_rx_codeblock> codeblock_llrs,
                                                       span<bit_buffer>                   cb_data_buffers,
                                                       span<span<log_likelihood_ratio>>   rm_buffers,
                                                       bool                               new_data,
                                                       crc_generator_poly                 crc_poly,
                                                       bool                               use_early_stop,
                                                       unsigned                           nof_ldpc_iterations)
{
  unsigned nof_cbs = codeblock_llrs.size();

  ocudu_assert(nof_cbs > 0 && nof_cbs <= MAX_BATCH_SIZE, "Invalid number of codeblocks: {}", nof_cbs);
  ocudu_assert(cb_data_buffers.size() == nof_cbs, "Mismatched cb_data_buffers size");
  ocudu_assert(rm_buffers.size() == nof_cbs, "Mismatched rm_buffers size");

  std::vector<codeblock_result> results(nof_cbs);

  // Get common parameters from first codeblock (all CBs share tb_common).
  const auto& tb_common    = codeblock_llrs[0].second.tb_common;
  unsigned    lifting_size = static_cast<unsigned>(tb_common.lifting_size);
  unsigned    bg_K         = (tb_common.base_graph == ldpc_base_graph_type::BG1) ? 22 : 10;
  unsigned    msg_length   = bg_K * lifting_size;

  // Get CRC calculator.
  crc_calculator* crc = select_crc(crc_poly);
  ocudu_assert(crc != nullptr, "Invalid CRC calculator.");

  // Check if all CBs have the same E and filler bits (required for GPU batch dematching).
  // Also check if this is a new transmission (HARQ combining not yet supported on GPU).
  const auto& first_cb_specific = codeblock_llrs[0].second.cb_specific;
  unsigned    common_E          = first_cb_specific.rm_length;
  unsigned    common_F          = first_cb_specific.nof_filler_bits;

  // Configure decoder for this batch (need filler bits for correct LDPC config).
  configure_decoder(tb_common.base_graph, lifting_size, nof_ldpc_iterations, common_F, nof_cbs, crc_poly);

  // GPU rate dematching is enabled for new transmissions with uniform CB configs.
  // HARQ retransmissions require LLR combining with previous soft buffers,
  // which is currently done on CPU.
  bool can_use_gpu_dematch = new_data;

  for (unsigned cb_idx = 1; cb_idx < nof_cbs && can_use_gpu_dematch; ++cb_idx) {
    const auto& cb_specific = codeblock_llrs[cb_idx].second.cb_specific;
    if (cb_specific.rm_length != common_E || cb_specific.nof_filler_bits != common_F) {
      can_use_gpu_dematch = false;
    }
  }

  cudaError_t      cuda_status;
  nr_ldpc_status_t status;

  if (can_use_gpu_dematch) {
    // GPU E2E path: Deinterleave + Rate dematch + LDPC decode all on GPU.
    configure_rate_dematcher(codeblock_llrs[0].second);

    if (timing_enabled_) {
      ensure_timing_events();
      last_timing_stats_                = pipeline_timing_stats{};
      last_timing_stats_.nof_cbs        = nof_cbs;
      last_timing_stats_.timing_enabled = true;
    }

    // Phase 1: Convert int8 LLRs to FP32 and copy to host staging buffer.
    for (unsigned cb_idx = 0; cb_idx < nof_cbs; ++cb_idx) {
      const auto&    cb_llrs     = codeblock_llrs[cb_idx].first;
      float*         cb_recv_ptr = h_received_llr_batch_.data() + cb_idx * rm_stride_;
      const unsigned E           = cb_llrs.size();

      for (size_t i = 0; i < E; ++i) {
        cb_recv_ptr[i] = static_cast<float>(cb_llrs[i].to_value_type());
      }
    }

    // Phase 2: H2D transfer of interleaved LLRs.
    if (timing_enabled_) {
      cudaEventRecord(timing_events_[0], stream_);
    }
    cuda_status = cudaMemcpyAsync(d_received_llr_batch_,
                                  h_received_llr_batch_.data(),
                                  nof_cbs * rm_stride_ * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  stream_);
    ocudu_assert(
        cuda_status == cudaSuccess, "Failed to copy received LLRs to device: {}", cudaGetErrorString(cuda_status));
    if (timing_enabled_) {
      cudaEventRecord(timing_events_[1], stream_);
    }

    // Phase 3: GPU deinterleaving (TS 38.212 Section 5.4.2.2).
    const unsigned Q_m = cached_rm_Qm_;
    if (Q_m >= 2) {
      status = rate_matcher_deinterleave_llr_batch(rate_matcher_handle_,
                                                   d_received_llr_batch_,
                                                   d_deinterleaved_llr_batch_,
                                                   static_cast<int>(cached_rm_E_),
                                                   static_cast<int>(Q_m),
                                                   static_cast<int>(nof_cbs),
                                                   stream_);
      ocudu_assert(
          status == NR_LDPC_SUCCESS, "CUDA batch deinterleave failed: {}", nr_ldpc_get_error_string(status));
    } else {
      cuda_status = cudaMemcpyAsync(d_deinterleaved_llr_batch_,
                                    d_received_llr_batch_,
                                    nof_cbs * rm_stride_ * sizeof(float),
                                    cudaMemcpyDeviceToDevice,
                                    stream_);
      ocudu_assert(cuda_status == cudaSuccess, "Failed to copy LLRs for BPSK: {}", cudaGetErrorString(cuda_status));
    }
    if (timing_enabled_) {
      cudaEventRecord(timing_events_[2], stream_);
    }

    // Phase 4: GPU FP32 rate dematching.
    status = rate_matcher_dematch_batch(
        rate_matcher_handle_, d_deinterleaved_llr_batch_, d_llr_batch_, static_cast<int>(nof_cbs), stream_);
    ocudu_assert(
        status == NR_LDPC_SUCCESS, "CUDA batch rate dematch failed: {}", nr_ldpc_get_error_string(status));

    if (timing_enabled_) {
      cudaEventRecord(timing_events_[3], stream_);
    }

    // Phase 5: FP32→FP16 quantize and FP16 LDPC decode.
    // The FP16 decoder matches the gnb E2E GPU pipeline.
    status =
        ldpc_decoder_decode_batch(decoder_handle_, d_llr_batch_, d_output_batch_, static_cast<int>(nof_cbs), stream_);
    ocudu_assert(status == NR_LDPC_SUCCESS, "CUDA batch decode failed: {}", nr_ldpc_get_error_string(status));

    if (timing_enabled_) {
      cudaEventRecord(timing_events_[4], stream_);
    }

    // Phase 6: D2H transfer of decoded output bits.
    cuda_status = cudaMemcpyAsync(h_output_batch_.data(),
                                  d_output_batch_,
                                  nof_cbs * output_stride_ * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream_);
    ocudu_assert(cuda_status == cudaSuccess, "Failed to copy outputs from device: {}", cudaGetErrorString(cuda_status));

    if (timing_enabled_) {
      cudaEventRecord(timing_events_[5], stream_);
    }

    // Phase 7: Synchronize.
    cudaStreamSynchronizeYielding(stream_);

    if (timing_enabled_) {
      float ms;
      cudaEventElapsedTime(&ms, timing_events_[0], timing_events_[1]);
      last_timing_stats_.h2d_transfer_us = ms * 1000.0f;
      cudaEventElapsedTime(&ms, timing_events_[1], timing_events_[2]);
      last_timing_stats_.deinterleave_us = ms * 1000.0f;
      cudaEventElapsedTime(&ms, timing_events_[2], timing_events_[3]);
      last_timing_stats_.rate_dematch_us = ms * 1000.0f;
      cudaEventElapsedTime(&ms, timing_events_[3], timing_events_[4]);
      last_timing_stats_.ldpc_decode_us = ms * 1000.0f;
      cudaEventElapsedTime(&ms, timing_events_[4], timing_events_[5]);
      last_timing_stats_.d2h_transfer_us = ms * 1000.0f;
      cudaEventElapsedTime(&ms, timing_events_[0], timing_events_[5]);
      last_timing_stats_.total_e2e_us = ms * 1000.0f;
      last_timing_stats_.total_backend_us =
          last_timing_stats_.deinterleave_us + last_timing_stats_.rate_dematch_us + last_timing_stats_.ldpc_decode_us;
    }

  } else {
    // CPU fallback path: Rate dematch on CPU, LDPC decode on GPU.
    // Used for HARQ retransmissions or when CBs have different E/F.
    std::fill(h_llr_batch_.begin(), h_llr_batch_.begin() + nof_cbs * llr_stride_, 0.0f);

    for (unsigned cb_idx = 0; cb_idx < nof_cbs; ++cb_idx) {
      const auto& cb_llrs = codeblock_llrs[cb_idx].first;
      const auto& cb_meta = codeblock_llrs[cb_idx].second;

      dematcher_->rate_dematch(rm_buffers[cb_idx], cb_llrs, new_data, cb_meta);

      float*      cb_batch_ptr    = h_llr_batch_.data() + cb_idx * llr_stride_;
      const auto& rm_buf          = rm_buffers[cb_idx];
      unsigned    puncture_offset = 2 * lifting_size;

      size_t copy_size = std::min(rm_buf.size(), static_cast<size_t>(llr_stride_ - puncture_offset));
      for (size_t i = 0; i < copy_size; ++i) {
        cb_batch_ptr[puncture_offset + i] = static_cast<float>(rm_buf[i].to_value_type());
      }
    }

    cuda_status = cudaMemcpyAsync(
        d_llr_batch_, h_llr_batch_.data(), nof_cbs * llr_stride_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    ocudu_assert(cuda_status == cudaSuccess, "Failed to copy LLRs to device: {}", cudaGetErrorString(cuda_status));

    status =
        ldpc_decoder_decode_batch(decoder_handle_, d_llr_batch_, d_output_batch_, static_cast<int>(nof_cbs), stream_);
    ocudu_assert(
        status == NR_LDPC_SUCCESS, "CUDA FP32 batch decode failed: {}", nr_ldpc_get_error_string(status));

    cuda_status = cudaMemcpyAsync(h_output_batch_.data(),
                                  d_output_batch_,
                                  nof_cbs * output_stride_ * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream_);
    ocudu_assert(cuda_status == cudaSuccess, "Failed to copy outputs from device: {}", cudaGetErrorString(cuda_status));

    cudaStreamSynchronizeYielding(stream_);
  }

  // Get actual average iterations from the decoder (tracks early termination).
  float    avg_iters    = ldpc_decoder_get_avg_iterations(decoder_handle_);
  unsigned actual_iters = static_cast<unsigned>(avg_iters + 0.5f); // Round to nearest
  if (actual_iters == 0) {
    actual_iters = nof_ldpc_iterations; // Fallback if not tracked
  }

  // Phase 7: Extract decoded bits and check CRCs (common to both paths).
  for (unsigned cb_idx = 0; cb_idx < nof_cbs; ++cb_idx) {
    const auto& cb_meta = codeblock_llrs[cb_idx].second;

    // Extract decoded bits.
    extract_decoded_bits(cb_data_buffers[cb_idx], cb_idx, msg_length);

    // Check CRC.
    unsigned nof_significant_bits = msg_length - cb_meta.cb_specific.nof_filler_bits;
    unsigned host_crc_val         = crc->calculate(cb_data_buffers[cb_idx].first(nof_significant_bits));
    if (host_crc_val == 0) {
      results[cb_idx].crc_ok         = true;
      results[cb_idx].nof_iterations = actual_iters;
    } else {
      results[cb_idx].crc_ok         = false;
      results[cb_idx].nof_iterations = nof_ldpc_iterations;
    }
  }

  return results;
}

// ============================================================================
// GPU TB Reassembly Implementation
// ============================================================================

pusch_codeblock_decoder_cuda_batch::tb_decode_result
pusch_codeblock_decoder_cuda_batch::decode_resident_softbits(
    void*                                                d_llrs_half,
    size_t                                               num_llrs,
    const scrambling_config*                             scrambling_cfg,
    const codeblock_metadata::tb_common_metadata&        tb_common,
    span<const codeblock_metadata::cb_specific_metadata> cb_specific,
    span<uint8_t>                                        tb_output,
    crc_generator_poly                                   cb_crc_poly,
    unsigned                                             nof_ldpc_iterations,
    void*                                                execution_context,
    int                                                  buffer_index)
{
  unsigned nof_cbs = cb_specific.size();
  ocudu_assert(nof_cbs > 0 && nof_cbs <= MAX_BATCH_SIZE, "Invalid number of codeblocks: {}", nof_cbs);

  // Select buffer index for triple-buffering.
  int buf_idx;
  if (buffer_index >= 0 && buffer_index < NUM_DECODE_BUFFERS) {
    buf_idx = buffer_index;
  } else {
    buf_idx             = current_decode_buf_;
    current_decode_buf_ = (current_decode_buf_ + 1) % NUM_DECODE_BUFFERS;
  }

  tb_decode_result result;
  result.nof_iterations = nof_ldpc_iterations;

  // Get common parameters.
  unsigned       lifting_size      = static_cast<unsigned>(tb_common.lifting_size);
  unsigned       bg_K              = (tb_common.base_graph == ldpc_base_graph_type::BG1) ? 22 : 10;
  unsigned       msg_length        = bg_K * lifting_size;
  const size_t   payload_bytes     = tb_output.size();
  const size_t   payload_bits      = payload_bytes * 8;               // TB size in bits (excluding CRC).
  const unsigned tb_crc_bits       = (payload_bits > 3824) ? 24 : 16; // CRC-24A for large TBs, CRC-16 for small.
  const unsigned cb_crc_bits       = (nof_cbs > 1) ? 24 : 0;          // CB CRC only for multi-CB.
  const unsigned cb_info_bits      = msg_length - cb_specific[0].nof_filler_bits - cb_crc_bits;
  const size_t   tb_with_crc_bytes = (payload_bits + tb_crc_bits + 7) / 8;

  // Check if all CBs have the same E and filler bits (required for GPU batch dematching).
  unsigned common_E       = cb_specific[0].rm_length;
  unsigned common_F       = cb_specific[0].nof_filler_bits;
  bool     uniform_config = true;

  for (unsigned cb_idx = 1; cb_idx < nof_cbs && uniform_config; ++cb_idx) {
    if (cb_specific[cb_idx].rm_length != common_E || cb_specific[cb_idx].nof_filler_bits != common_F) {
      uniform_config = false;
    }
  }

  // Initialize timing before any host-side stages so optional config timings are not lost.
  if (timing_enabled_) {
    ensure_timing_events();
    last_timing_stats_                = pipeline_timing_stats{};
    last_timing_stats_.nof_cbs        = nof_cbs;
    last_timing_stats_.timing_enabled = true;
  }

  // Use provided execution context or our own CUDA stream.
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(execution_context);
  cudaStream_t stream      = (cuda_stream != nullptr) ? cuda_stream : stream_;

  nr_ldpc_status_t status;
  const int*       d_decoder_crc_results       = nullptr;
  auto             capture_decoder_crc_results = [&]() {
    d_decoder_crc_results = nullptr;
    if (ldpc_decoder_has_crc_results(decoder_handle_)) {
      d_decoder_crc_results = ldpc_decoder_get_crc_results_ptr(decoder_handle_);
    }
  };

  // Configure decoder for this batch (need filler bits for correct LDPC config).
  auto decoder_cfg_start = timing_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  configure_decoder(tb_common.base_graph, lifting_size, nof_ldpc_iterations, common_F, nof_cbs, cb_crc_poly);
  if (timing_enabled_) {
    last_timing_stats_.decoder_config_us =
        std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - decoder_cfg_start).count();
    cudaEventRecord(timing_events_[0], stream);
  }

  // Phase 1: Generate scrambling sequence (if needed) for fused descramble+rate dematch.
  const unsigned int* scramble_seq_ptr    = nullptr;
  int                 scramble_bit_offset = 0;
  if (scrambling_cfg != nullptr) {
    nr_scrambling_config_t scr_cfg = {};
    scr_cfg.n_RNTI                 = scrambling_cfg->n_rnti;
    scr_cfg.n_ID                   = scrambling_cfg->n_id;
    scr_cfg.q                      = scrambling_cfg->q;
    scr_cfg.n_s                    = scrambling_cfg->n_s;

    status = scrambler_configure(scrambler_handle_, &scr_cfg);
    ocudu_assert(status == NR_LDPC_SUCCESS, "Failed to configure scrambler: {}", nr_ldpc_get_error_string(status));

    if (scrambling_cfg->bit_offset > 0) {
      status = scrambler_set_offset(scrambler_handle_, static_cast<int>(scrambling_cfg->bit_offset));
      ocudu_assert(status == NR_LDPC_SUCCESS, "Failed to set scrambler offset");
    }

    status = scrambler_generate_sequence(scrambler_handle_, static_cast<int>(num_llrs), stream);
    ocudu_assert(
        status == NR_LDPC_SUCCESS, "Failed to generate scrambling sequence: {}", nr_ldpc_get_error_string(status));

    // Get sequence pointer for fused descramble+rate dematch (no separate descramble kernel).
    scramble_seq_ptr    = scrambler_get_sequence_ptr(scrambler_handle_);
    scramble_bit_offset = 0; // Sequence is already generated at the correct offset.
  }

  if (timing_enabled_) {
    cudaEventRecord(timing_events_[1], stream); // End scramble seq gen
  }

  const bool     direct_rm_ldpc_enabled = !env_value_is_disabled(std::getenv("OCUDU_LDPC_DIRECT_RM"));
  const unsigned direct_rm_min_cbs      = get_env_unsigned_or_default("OCUDU_LDPC_DIRECT_RM_MIN_CBS", 64);
  bool           direct_rm_ldpc_path    = false;
  int            direct_N_cb            = 0;
  int            direct_k0              = 0;

  if (direct_rm_ldpc_enabled && nof_cbs >= direct_rm_min_cbs && cached_use_boxplus_ &&
      tb_common.base_graph == ldpc_base_graph_type::BG1 && lifting_size == 384 &&
      common_F == cb_specific[0].nof_filler_bits) {
    const int K_total           = 22 * static_cast<int>(lifting_size);
    const int num_codeword_bits = K_total + 46 * static_cast<int>(lifting_size);
    const int N_short           = num_codeword_bits - 2 * static_cast<int>(lifting_size);
    direct_N_cb                 = (tb_common.Nref > 0) ? std::min(static_cast<int>(tb_common.Nref), N_short) : N_short;
    direct_k0 = rate_matcher_compute_k0(1, static_cast<int>(lifting_size), static_cast<int>(tb_common.rv), direct_N_cb);

    direct_rm_ldpc_path         = true;
    const int effective_N_cb    = direct_N_cb - static_cast<int>(common_F);
    unsigned  cumulative_offset = 0;
    for (unsigned cb_idx = 0; cb_idx != nof_cbs; ++cb_idx) {
      if (cb_specific[cb_idx].nof_filler_bits != common_F ||
          static_cast<int>(cb_specific[cb_idx].rm_length) > effective_N_cb) {
        direct_rm_ldpc_path = false;
      }
      h_bits_per_cb_[cb_idx]    = static_cast<int>(cb_specific[cb_idx].rm_length);
      h_gather_offsets_[cb_idx] = cumulative_offset;
      cumulative_offset += cb_specific[cb_idx].rm_length;
    }
    if (cumulative_offset != num_llrs) {
      direct_rm_ldpc_path = false;
    }
  }

  // Phase 2: GPU batch rate dematching + LDPC decode.
  if (direct_rm_ldpc_path) {
    bool direct_metadata_on_device =
        cached_direct_rm_metadata_valid_ && (cached_direct_rm_nof_cbs_ == nof_cbs) &&
        (cached_direct_rm_num_llrs_ == num_llrs) && (cached_direct_rm_n_cb_ == direct_N_cb) &&
        (cached_direct_rm_k0_ == direct_k0) && (cached_direct_rm_filler_bits_ == common_F) &&
        (std::memcmp(cached_direct_rm_bits_per_cb_.data(), h_bits_per_cb_.data(), nof_cbs * sizeof(int)) == 0) &&
        (std::memcmp(cached_direct_rm_gather_offsets_.data(), h_gather_offsets_, nof_cbs * sizeof(unsigned int)) == 0);

    if (!direct_metadata_on_device) {
      cudaError_t cuda_status =
          cudaMemcpyAsync(d_bits_per_cb_, h_bits_per_cb_.data(), nof_cbs * sizeof(int), cudaMemcpyHostToDevice, stream);
      ocudu_assert(cuda_status == cudaSuccess,
                   "Failed to copy direct LDPC rate-match lengths: {}",
                   cudaGetErrorString(cuda_status));
      cuda_status = cudaMemcpyAsync(
          d_gather_offsets_, h_gather_offsets_, nof_cbs * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
      ocudu_assert(cuda_status == cudaSuccess,
                   "Failed to copy direct LDPC rate-match offsets: {}",
                   cudaGetErrorString(cuda_status));

      cached_direct_rm_metadata_valid_ = true;
      cached_direct_rm_nof_cbs_        = nof_cbs;
      cached_direct_rm_num_llrs_       = num_llrs;
      cached_direct_rm_n_cb_           = direct_N_cb;
      cached_direct_rm_k0_             = direct_k0;
      cached_direct_rm_filler_bits_    = common_F;
      std::copy_n(h_bits_per_cb_.data(), nof_cbs, cached_direct_rm_bits_per_cb_.data());
      std::copy_n(h_gather_offsets_, nof_cbs, cached_direct_rm_gather_offsets_.data());
    }

    if (timing_enabled_) {
      cudaEventRecord(timing_events_[2], stream); // End direct metadata setup.
    }

    unsigned Q_m = get_mod_order(tb_common.mod);
    status       = ldpc_decoder_decode_batch_half_from_rate_matched(decoder_handle_,
                                                              d_llrs_half,
                                                              d_output_batch_,
                                                              static_cast<int>(nof_cbs),
                                                              static_cast<int>(Q_m),
                                                              direct_N_cb,
                                                              direct_k0,
                                                              d_bits_per_cb_,
                                                              d_gather_offsets_,
                                                              scramble_seq_ptr,
                                                              scramble_bit_offset,
                                                              stream,
                                                              buf_idx);

    if (status == NR_LDPC_SUCCESS) {
      capture_decoder_crc_results();

      if (timing_enabled_) {
        cudaEventRecord(timing_events_[3], stream); // End LDPC decode
      }
    } else if (status == NR_LDPC_ERROR_INVALID_CONFIG || status == NR_LDPC_ERROR_CUDA_FAILED) {
      direct_rm_ldpc_path = false;
    } else {
      ocudu_assert(false, "CUDA direct rate-match LDPC decode failed: {}", nr_ldpc_get_error_string(status));
    }
  }

  if (!direct_rm_ldpc_path && uniform_config) {
    // Configure rate matcher for batch operation.
    codeblock_metadata cfg = {};
    cfg.tb_common          = tb_common;
    cfg.cb_specific        = cb_specific[0];
    auto rate_cfg_start = timing_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    configure_rate_dematcher(cfg);
    if (timing_enabled_) {
      last_timing_stats_.rate_match_config_us =
          std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - rate_cfg_start).count();
    }

    // Fused descramble + deinterleave + rate dematch (FP16).
    unsigned Q_m = get_mod_order(tb_common.mod);
    status       = rate_matcher_deinterleave_and_dematch_batch_half(rate_matcher_handle_,
                                                              d_llrs_half,
                                                              d_llr_batch_half_[buf_idx],
                                                              static_cast<int>(Q_m),
                                                              static_cast<int>(nof_cbs),
                                                              stream,
                                                              scramble_seq_ptr,
                                                              scramble_bit_offset);
    ocudu_assert(status == NR_LDPC_SUCCESS,
                 "CUDA fused deinterleave+dematch failed: {}",
                 nr_ldpc_get_error_string(status));

    if (timing_enabled_) {
      cudaEventRecord(timing_events_[2], stream); // End rate dematch
    }

    // Batch LDPC decode (FP16 - stable with erasures).
    status = ldpc_decoder_decode_batch_half(
        decoder_handle_, d_llr_batch_half_[buf_idx], d_output_batch_, static_cast<int>(nof_cbs), stream, buf_idx);
    ocudu_assert(
        status == NR_LDPC_SUCCESS, "CUDA FP16 batch decode failed: {}", nr_ldpc_get_error_string(status));
    capture_decoder_crc_results();

    if (timing_enabled_) {
      cudaEventRecord(timing_events_[3], stream); // End LDPC decode
    }

  } else if (!direct_rm_ldpc_path) {
    // Non-uniform E: process each consecutive (E,F) run directly from the resident LLR buffer. If this API receives
    // scrambled input, each run uses its absolute codeword offset into the generated scrambling sequence; this is
    // equivalent to the old full-buffer in-place descramble but avoids an extra device memory pass.
    unsigned Q_m = get_mod_order(tb_common.mod);

    struct cb_run {
      unsigned first_cb;
      unsigned nof_cbs;
      unsigned rm_length;
      unsigned nof_filler_bits;
      unsigned cw_offset;
    };

    std::array<cb_run, MAX_BATCH_SIZE> cb_runs           = {};
    unsigned                           nof_runs          = 0;
    unsigned                           cumulative_offset = 0;
    for (unsigned cb_idx = 0; cb_idx != nof_cbs;) {
      const unsigned first_cb      = cb_idx;
      const unsigned run_E         = cb_specific[cb_idx].rm_length;
      const unsigned run_F         = cb_specific[cb_idx].nof_filler_bits;
      const unsigned run_cw_offset = cumulative_offset;
      unsigned       run_nof_cbs   = 0;

      do {
        cumulative_offset += cb_specific[cb_idx].rm_length;
        ++cb_idx;
        ++run_nof_cbs;
      } while ((cb_idx != nof_cbs) && (cb_specific[cb_idx].rm_length == run_E) &&
               (cb_specific[cb_idx].nof_filler_bits == run_F));

      cb_runs[nof_runs++] = cb_run{first_cb, run_nof_cbs, run_E, run_F, run_cw_offset};
    }
    ocudu_assert(cumulative_offset == num_llrs,
                 "PUSCH resident decode LLR count mismatch: metadata describes {} LLRs but demodulator provided {}.",
                 cumulative_offset,
                 num_llrs);

    // Rate-dematch each run separately, writing to the matching codeblock
    // range in d_llr_batch_half_. LDPC decode happens once after all runs; the
    // decoder dimensions do not depend on E.
    for (unsigned run_idx = 0; run_idx != nof_runs; ++run_idx) {
      const cb_run& run = cb_runs[run_idx];

      codeblock_metadata cfg = {};
      cfg.tb_common          = tb_common;
      cfg.cb_specific        = cb_specific[run.first_cb];
      auto rate_cfg_start =
          timing_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      configure_rate_dematcher(cfg);
      if (timing_enabled_) {
        last_timing_stats_.rate_match_config_us +=
            std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - rate_cfg_start).count();
      }

      const void* d_group_in =
          static_cast<const char*>(d_llrs_half) + static_cast<size_t>(run.cw_offset) * sizeof(uint16_t);
      void* d_group_llr_half = static_cast<char*>(d_llr_batch_half_[buf_idx]) +
                               static_cast<size_t>(run.first_cb) * llr_stride_ * sizeof(uint16_t);
      status = rate_matcher_deinterleave_and_dematch_batch_half(rate_matcher_handle_,
                                                                d_group_in,
                                                                d_group_llr_half,
                                                                static_cast<int>(Q_m),
                                                                static_cast<int>(run.nof_cbs),
                                                                stream,
                                                                scramble_seq_ptr,
                                                                scramble_bit_offset + static_cast<int>(run.cw_offset));
      ocudu_assert(status == NR_LDPC_SUCCESS,
                   "CUDA fused deinterleave+dematch failed: {}",
                   nr_ldpc_get_error_string(status));
    }

    if (timing_enabled_) {
      cudaEventRecord(timing_events_[2], stream); // End rate dematch
    }

    status = ldpc_decoder_decode_batch_half(
        decoder_handle_, d_llr_batch_half_[buf_idx], d_output_batch_, static_cast<int>(nof_cbs), stream, buf_idx);
    ocudu_assert(
        status == NR_LDPC_SUCCESS, "CUDA FP16 batch decode failed: {}", nr_ldpc_get_error_string(status));
    capture_decoder_crc_results();

    if (timing_enabled_) {
      cudaEventRecord(timing_events_[3], stream); // End LDPC decode
    }
  }

  // Phase 4: GPU TB desegmentation (no mid-pipeline sync — always deseg,
  // check CRCs on CPU after single sync at end).

  // Validate F is uniform across all CBs (5G NR invariant required for deseg).
  for (unsigned cb_idx = 1; cb_idx < nof_cbs; ++cb_idx) {
    ocudu_assert(cb_specific[cb_idx].nof_filler_bits == cb_specific[0].nof_filler_bits,
                 "Non-uniform filler bits: CB[0]={} CB[{}]={} — deseg requires uniform F",
                 cb_specific[0].nof_filler_bits,
                 cb_idx,
                 cb_specific[cb_idx].nof_filler_bits);
  }

  ocudu_assert(payload_bytes <= MAX_TB_BYTES,
               "TB output span has {} bytes, exceeding staging capacity {}.",
               payload_bytes,
               MAX_TB_BYTES);
  ocudu_assert(tb_with_crc_bytes <= MAX_TB_BYTES,
               "TB output plus CRC has {} bytes, exceeding staging capacity {}.",
               tb_with_crc_bytes,
               MAX_TB_BYTES);

  tb_desegment_config_t deseg_cfg = {};
  deseg_cfg.num_code_blocks       = static_cast<int>(nof_cbs);
  deseg_cfg.tb_size_bits          = static_cast<int>(payload_bits);
  deseg_cfg.cb_info_bits          = static_cast<int>(cb_info_bits);
  deseg_cfg.cb_stride_words       = static_cast<int>(output_stride_);
  deseg_cfg.tb_crc_bits           = static_cast<int>(tb_crc_bits);
  deseg_cfg.cb_crc_bits           = static_cast<int>(cb_crc_bits);
  deseg_cfg.nof_filler_bits       = static_cast<int>(cb_specific[0].nof_filler_bits);

  // GPU desegmentation + GPU TB CRC check (all on device, no mid-pipeline sync).
  status = tb_desegment_and_check_crc_async(d_output_batch_, d_tb_output_, &deseg_cfg, d_tb_crc_result_, stream);
  ocudu_assert(status == NR_LDPC_SUCCESS, "GPU TB desegmentation + CRC failed");

  if (timing_enabled_) {
    cudaEventRecord(timing_events_[4], stream); // End deseg + TB CRC
  }

  // Phase 5: D2H transfer of payload bytes, CB CRCs, and TB CRC result (single sync).
  //
  // ABI boundary: tb_output owns payload bytes only. The device TB buffer also contains appended
  // TB CRC bytes used by the GPU CRC kernel. Direct D2H may target tb_output only for payload_bytes;
  // the CRC tail must never be written into the caller span.
  const bool direct_payload_copy = can_copy_payload_output_direct(tb_output);
  if (direct_payload_copy) {
    cudaMemcpyAsync(tb_output.data(), d_tb_output_, payload_bytes, cudaMemcpyDeviceToHost, stream);
  } else {
    cudaMemcpyAsync(h_tb_output_pinned_, d_tb_output_, tb_with_crc_bytes, cudaMemcpyDeviceToHost, stream);
  }

  if (d_decoder_crc_results != nullptr) {
    cudaError_t cuda_status = cudaMemcpyAsync(
        h_crc_results_pinned_, d_decoder_crc_results, nof_cbs * sizeof(int), cudaMemcpyDeviceToHost, stream);
    ocudu_assert(cuda_status == cudaSuccess,
                 "Failed to copy decoder CB CRC results to host: {}",
                 cudaGetErrorString(cuda_status));
  } else {
    // Missing decoder CRC status must never look like success. Fill fallback scratch with non-zero failure statuses.
    cudaError_t cuda_status = cudaMemsetAsync(d_crc_results_, 1, nof_cbs * sizeof(int), stream);
    ocudu_assert(cuda_status == cudaSuccess,
                 "Failed to initialize fail-closed CB CRC fallback: {}",
                 cudaGetErrorString(cuda_status));
    cuda_status =
        cudaMemcpyAsync(h_crc_results_pinned_, d_crc_results_, nof_cbs * sizeof(int), cudaMemcpyDeviceToHost, stream);
    ocudu_assert(cuda_status == cudaSuccess,
                 "Failed to copy fail-closed CB CRC fallback to host: {}",
                 cudaGetErrorString(cuda_status));
  }

  cudaMemcpyAsync(h_tb_crc_result_pinned_, d_tb_crc_result_, sizeof(int), cudaMemcpyDeviceToHost, stream);

  if (timing_enabled_) {
    cudaEventRecord(timing_events_[5], stream); // End D2H
  }

  if (completion_event_) {
    auto wait_start = timing_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    cudaEventRecord(completion_event_, stream);
    cudaEventSynchronizeYielding(completion_event_);
    if (timing_enabled_) {
      last_timing_stats_.completion_wait_us =
          std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - wait_start).count();
    }
  } else {
    auto wait_start = timing_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    cudaStreamSynchronizeYielding(stream);
    if (timing_enabled_) {
      last_timing_stats_.completion_wait_us =
          std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - wait_start).count();
    }
  }

  // Collect timing statistics.
  if (timing_enabled_) {
    float ms;
    cudaEventElapsedTime(&ms, timing_events_[0], timing_events_[1]);
    last_timing_stats_.deinterleave_us = ms * 1000.0f; // Scramble seq gen
    cudaEventElapsedTime(&ms, timing_events_[1], timing_events_[2]);
    last_timing_stats_.rate_dematch_us = ms * 1000.0f;
    cudaEventElapsedTime(&ms, timing_events_[2], timing_events_[3]);
    last_timing_stats_.ldpc_decode_us = ms * 1000.0f;
    cudaEventElapsedTime(&ms, timing_events_[3], timing_events_[4]);
    last_timing_stats_.crc_check_us = ms * 1000.0f; // Deseg + TB CRC
    cudaEventElapsedTime(&ms, timing_events_[4], timing_events_[5]);
    last_timing_stats_.d2h_transfer_us = ms * 1000.0f;
    cudaEventElapsedTime(&ms, timing_events_[0], timing_events_[5]);
    last_timing_stats_.total_e2e_us = ms * 1000.0f;
    last_timing_stats_.total_backend_us =
        last_timing_stats_.rate_dematch_us + last_timing_stats_.ldpc_decode_us + last_timing_stats_.crc_check_us;
  }

  // Copy only payload bytes into the caller span when we used pinned staging.
  if (!direct_payload_copy) {
    auto tb_copy_start = timing_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::memcpy(tb_output.data(), h_tb_output_pinned_, payload_bytes);
    if (timing_enabled_) {
      last_timing_stats_.tb_output_copy_us =
          std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - tb_copy_start).count();
    }
  }

  // Read GPU-computed CB CRC results (kernel convention: 0 = pass).
  bool all_cb_crcs_ok = true;
  for (unsigned cb_idx = 0; cb_idx < nof_cbs; ++cb_idx) {
    if (h_crc_results_pinned_[cb_idx] != 0) {
      all_cb_crcs_ok = false;
      break;
    }
  }

  // Read GPU-computed TB CRC result (kernel convention: 1 = pass, 0 = fail).
  int gpu_tb_crc_ok     = *h_tb_crc_result_pinned_;
  result.tb_crc_ok      = all_cb_crcs_ok && (gpu_tb_crc_ok == 1);
  result.all_cb_crcs_ok = all_cb_crcs_ok;

  // Get actual average iterations from the decoder.
  auto  avg_iters_start = timing_enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  float avg_iters       = ldpc_decoder_get_avg_iterations(decoder_handle_);
  if (timing_enabled_) {
    last_timing_stats_.avg_iters_query_us =
        std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - avg_iters_start).count();
  }
  unsigned actual_iters = static_cast<unsigned>(avg_iters + 0.5f);
  if (actual_iters == 0) {
    actual_iters = nof_ldpc_iterations;
  }
  result.nof_iterations = actual_iters;

  return result;
}
