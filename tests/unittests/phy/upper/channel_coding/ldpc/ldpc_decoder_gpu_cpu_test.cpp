// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief Validates GPU LDPC decoder (CUDA) against CPU LDPC encoder (generic).
///
/// For each (base graph, lifting size, filler percentage) combination:
/// 1. Generate random information bits and encode with the CPU encoder.
/// 2. Build ideal LLRs from the encoded codeblock (bit 1 -> -127, bit 0 -> +127).
/// 3. Upload LLRs to GPU, decode with CUDA, and compare decoded bits to original input.
///
/// This validates that the GPU decoder produces correct output for clean (noiseless)
/// codewords across all 51 lifting sizes and both base graphs.

#include "ocudu/adt/bit_buffer.h"
#include "ocudu/phy/upper/channel_coding/channel_coding_factories.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_encoder.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_encoder_buffer.h"
#include <ocudu_phy_cuda.h>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace ocudu;
using namespace ocudu::ldpc;

namespace {

/// Converts a linear decoded-bit index to CUDA's MSB-first-per-byte bit position inside a uint32_t word.
inline unsigned cuda_bit_pos(unsigned bit_idx)
{
  return (bit_idx % 32) ^ 7U;
}

} // namespace

/// All 51 valid 5G NR lifting sizes (TS 38.212 Table 5.3.2-1).
static constexpr unsigned ALL_LIFTING_SIZES[] = {
    2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  18,  20,
    22,  24,  26,  28,  30,  32,  36,  40,  44,  48,  52,  56,  60,  64,  72,  80,  88,
    96,  104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384};

struct decoder_test_params {
  ldpc_base_graph_type bg;
  unsigned             Z;
  unsigned             filler_percent; // 0, 25, 50
};

std::ostream& operator<<(std::ostream& os, const decoder_test_params& p)
{
  return os << "BG" << ((p.bg == ldpc_base_graph_type::BG1) ? 1 : 2) << "_Z" << p.Z << "_F" << p.filler_percent;
}

class LDPCDecoderGpuCpuTest : public ::testing::TestWithParam<decoder_test_params>
{
public:
  static void SetUpTestSuite()
  {
    // Create CPU encoder factory.
    if (!cpu_encoder_factory) {
      cpu_encoder_factory = create_ldpc_encoder_factory_sw("generic");
      ASSERT_NE(cpu_encoder_factory, nullptr) << "Failed to create CPU encoder factory";
    }

    // Initialize CUDA library.
    nr_ldpc_status_t status = ocudu_phy_cuda_init();
    ASSERT_EQ(status, NR_LDPC_SUCCESS) << "Failed to initialize CUDA: " << nr_ldpc_get_error_string(status);

    // Create CUDA decoder.
    status = ldpc_decoder_create(&decoder_handle);
    ASSERT_EQ(status, NR_LDPC_SUCCESS) << "Failed to create CUDA decoder: " << nr_ldpc_get_error_string(status);

    // Create CUDA stream.
    cudaError_t cuda_status = cudaStreamCreate(&stream);
    ASSERT_EQ(cuda_status, cudaSuccess) << "Failed to create CUDA stream: " << cudaGetErrorString(cuda_status);

    // Allocate device memory for the largest possible configuration.
    // BG1: N_full = 68 * 384 = 26112 floats for LLRs.
    // Output: K_max = 22 * 384 = 8448 bits -> 264 uint32_t words.
    unsigned max_llr_count = 68 * 384;
    unsigned max_output_words = (22 * 384 + 31) / 32;

    cuda_status = cudaMalloc(&d_llrs, max_llr_count * sizeof(float));
    ASSERT_EQ(cuda_status, cudaSuccess) << "Failed to allocate device LLR memory: " << cudaGetErrorString(cuda_status);

    cuda_status = cudaMalloc(&d_output, max_output_words * sizeof(uint32_t));
    ASSERT_EQ(cuda_status, cudaSuccess) << "Failed to allocate device output memory: " << cudaGetErrorString(cuda_status);
  }

  static void TearDownTestSuite()
  {
    if (d_llrs) {
      cudaFree(d_llrs);
      d_llrs = nullptr;
    }
    if (d_output) {
      cudaFree(d_output);
      d_output = nullptr;
    }
    if (decoder_handle) {
      ldpc_decoder_destroy(decoder_handle);
      decoder_handle = nullptr;
    }
    if (stream) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }

protected:
  /// Encode with CPU, decode with GPU, compare decoded bits to original input.
  /// Returns the number of bit mismatches (0 = pass).
  unsigned encode_decode_compare(const dynamic_bit_buffer& input,
                                 ldpc_base_graph_type      bg,
                                 unsigned                  Z,
                                 unsigned                  filler_bits)
  {
    int bg_num         = (bg == ldpc_base_graph_type::BG1) ? 1 : 2;
    unsigned Kb        = (bg == ldpc_base_graph_type::BG1) ? 22 : 10;
    unsigned N_cols    = (bg == ldpc_base_graph_type::BG1) ? 68 : 52;
    unsigned parity_nodes = (bg == ldpc_base_graph_type::BG1) ? 46 : 42;
    unsigned K         = Kb * Z;
    unsigned N_full    = N_cols * Z;
    unsigned F         = filler_bits;
    unsigned Kd        = K - F;

    // --- Step 1: Encode with CPU encoder ---
    auto cpu_enc = cpu_encoder_factory->create();
    EXPECT_NE(cpu_enc, nullptr);
    if (!cpu_enc) {
      return 999999;
    }

    ldpc_encoder::configuration enc_cfg;
    enc_cfg.base_graph   = bg;
    enc_cfg.lifting_size = static_cast<lifting_size_t>(Z);

    const ldpc_encoder_buffer& enc_buf = cpu_enc->encode(input, enc_cfg);

    // CPU encoder outputs bits starting at column 2 (after 2 punctured columns).
    // Total output length = (N_cols - 2) * Z bits.
    unsigned enc_len = enc_buf.get_codeblock_length();
    std::vector<uint8_t> encoded_bits(enc_len);
    enc_buf.write_codeblock(encoded_bits, 0);

    // --- Step 2: Build full codeblock LLRs ---
    std::vector<float> h_llrs(N_full, 0.0f);

    // Positions 0..2Z-1: 0.0f (punctured, unknown).
    // Already zero from initialization.

    // Positions 2Z..N_full-1: from CPU encoded bits.
    // CPU encoded bit index i corresponds to codeword position (2*Z + i).
    // encoded bit 1 -> LLR = -127.0f, encoded bit 0 -> LLR = +127.0f.
    for (unsigned i = 0; i < enc_len; ++i) {
      h_llrs[2 * Z + i] = encoded_bits[i] ? -127.0f : 127.0f;
    }

    // Filler bit positions: Kd..K-1 in the codeword.
    // These are known to be zero, so LLR = +127.0f.
    for (unsigned i = Kd; i < K; ++i) {
      h_llrs[i] = 127.0f;
    }

    // --- Step 3: Upload LLRs to GPU ---
    cudaError_t cuda_status =
        cudaMemcpy(d_llrs, h_llrs.data(), N_full * sizeof(float), cudaMemcpyHostToDevice);
    EXPECT_EQ(cuda_status, cudaSuccess) << "Failed to copy LLRs to device";
    if (cuda_status != cudaSuccess) {
      return 999999;
    }

    // --- Step 4: Configure CUDA decoder ---
    nr_ldpc_config_t ldpc_cfg   = {};
    ldpc_cfg.base_graph         = bg_num;
    ldpc_cfg.lifting_size       = static_cast<int>(Z);
    ldpc_cfg.lifting_set_index  = nr_ldpc_get_lifting_set_index(static_cast<int>(Z));
    ldpc_cfg.num_info_bits      = static_cast<int>(Kd);
    ldpc_cfg.num_parity_bits    = static_cast<int>(parity_nodes * Z);
    ldpc_cfg.max_parity_nodes   = static_cast<int>(parity_nodes);
    ldpc_cfg.num_codeword_bits  = static_cast<int>(K + parity_nodes * Z);
    ldpc_cfg.num_filler_bits    = static_cast<int>(F);
    ldpc_cfg.puncture           = true;
    ldpc_cfg.redundancy_version = 0;

    ldpc_decoder_params_t dec_params;
    ldpc_decoder_params_init(&dec_params);
    dec_params.max_iterations        = 10;
    dec_params.early_termination     = true;
    dec_params.auto_scale            = true;
    dec_params.llr_clamp             = 127.0f;
    dec_params.crc_early_termination = false;

    nr_ldpc_status_t status = ldpc_decoder_configure(decoder_handle, &ldpc_cfg, &dec_params);
    EXPECT_EQ(status, NR_LDPC_SUCCESS) << "Failed to configure decoder: " << nr_ldpc_get_error_string(status);
    if (status != NR_LDPC_SUCCESS) {
      return 999999;
    }

    // --- Step 5: Decode ---
    status = ldpc_decoder_decode_batch(decoder_handle, d_llrs, d_output, 1, stream);
    EXPECT_EQ(status, NR_LDPC_SUCCESS) << "Decode failed: " << nr_ldpc_get_error_string(status);
    if (status != NR_LDPC_SUCCESS) {
      return 999999;
    }

    cudaStreamSynchronize(stream);

    // --- Step 6: Copy output back to host ---
    unsigned nof_output_words = (K + 31) / 32;
    std::vector<uint32_t> h_output(nof_output_words, 0);
    cuda_status = cudaMemcpy(h_output.data(), d_output, nof_output_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(cuda_status, cudaSuccess) << "Failed to copy output from device";
    if (cuda_status != cudaSuccess) {
      return 999999;
    }

    // --- Step 7: Extract decoded bits (MSB-first-per-byte format) ---
    unsigned nof_bytes = (Kd + 7) / 8;
    std::vector<uint8_t> decoded_bytes(nof_bytes, 0);

    for (unsigned byte_idx = 0; byte_idx < nof_bytes; ++byte_idx) {
      uint8_t  byte_val = 0;
      unsigned base_bit = byte_idx * 8;
      for (unsigned j = 0; j < 8 && (base_bit + j) < Kd; ++j) {
        unsigned src_word = (base_bit + j) / 32;
        unsigned src_bit  = cuda_bit_pos(base_bit + j);
        byte_val |= (((h_output[src_word] >> src_bit) & 1) << (7 - j));
      }
      decoded_bytes[byte_idx] = byte_val;
    }

    // --- Step 8: Extract original input bits for comparison ---
    // The input dynamic_bit_buffer stores bits MSB-first in bytes (big-endian bit order).
    // We compare the first Kd bits.
    unsigned nof_input_bytes = (Kd + 7) / 8;
    std::vector<uint8_t> input_bytes(nof_input_bytes, 0);
    for (unsigned byte_idx = 0; byte_idx < nof_input_bytes; ++byte_idx) {
      unsigned base_bit = byte_idx * 8;
      uint8_t  byte_val = 0;
      for (unsigned j = 0; j < 8 && (base_bit + j) < Kd; ++j) {
        unsigned bit_val = input.extract(base_bit + j, 1);
        byte_val |= (bit_val << (7 - j));
      }
      input_bytes[byte_idx] = byte_val;
    }

    // --- Step 9: Compare ---
    unsigned mismatches     = 0;
    int      first_mismatch = -1;
    for (unsigned bit_idx = 0; bit_idx < Kd; ++bit_idx) {
      unsigned byte_idx = bit_idx / 8;
      unsigned bit_pos  = 7 - (bit_idx % 8);
      unsigned dec_bit  = (decoded_bytes[byte_idx] >> bit_pos) & 1;
      unsigned ref_bit  = (input_bytes[byte_idx] >> bit_pos) & 1;
      if (dec_bit != ref_bit) {
        if (first_mismatch < 0) {
          first_mismatch = static_cast<int>(bit_idx);
        }
        ++mismatches;
      }
    }

    if (mismatches > 0) {
      unsigned mismatch_col = static_cast<unsigned>(first_mismatch) / Z;
      unsigned mismatch_z   = static_cast<unsigned>(first_mismatch) % Z;
      ADD_FAILURE() << "BG" << bg_num << " Z=" << Z << " F=" << F << ": " << mismatches
                    << " bit mismatches out of " << Kd << " info bits (first at bit " << first_mismatch << " = col "
                    << mismatch_col << " z=" << mismatch_z << ")";
    }

    // --- Step 10: Check iterations ---
    float avg_iterations = ldpc_decoder_get_avg_iterations(decoder_handle);
    EXPECT_LE(avg_iterations, static_cast<float>(dec_params.max_iterations))
        << "Average iterations exceeded max for BG" << bg_num << " Z=" << Z;

    return mismatches;
  }

  static std::shared_ptr<ldpc_encoder_factory> cpu_encoder_factory;
  static ldpc_decoder_handle_t                 decoder_handle;
  static cudaStream_t                          stream;
  static float*                                d_llrs;
  static uint32_t*                             d_output;
};

std::shared_ptr<ldpc_encoder_factory> LDPCDecoderGpuCpuTest::cpu_encoder_factory = nullptr;
ldpc_decoder_handle_t                 LDPCDecoderGpuCpuTest::decoder_handle      = nullptr;
cudaStream_t                          LDPCDecoderGpuCpuTest::stream              = nullptr;
float*                                LDPCDecoderGpuCpuTest::d_llrs              = nullptr;
uint32_t*                             LDPCDecoderGpuCpuTest::d_output            = nullptr;

/// Test with random data (multiple seeds).
TEST_P(LDPCDecoderGpuCpuTest, RandomData)
{
  auto [bg, Z, filler_percent] = GetParam();
  unsigned Kb = (bg == ldpc_base_graph_type::BG1) ? 22 : 10;
  unsigned K  = Kb * Z;

  // Compute filler bits rounded to Z boundary.
  unsigned F_nodes = (filler_percent * Kb) / 100;
  unsigned F       = F_nodes * Z;
  unsigned Kd      = K - F;

  for (unsigned seed = 0; seed < 3; ++seed) {
    std::mt19937                           rgen(seed * 1000 + Z);
    std::uniform_int_distribution<uint8_t> byte_gen(0, 255);

    dynamic_bit_buffer input(K);
    // Fill first Kd bits with random data.
    for (unsigned i = 0; i < Kd; i += 8) {
      unsigned nbits = std::min(8U, Kd - i);
      unsigned val   = byte_gen(rgen);
      if (nbits < 8) {
        val &= (1U << nbits) - 1;
      }
      input.insert(val, i, nbits);
    }
    // Filler bits (positions Kd..K-1) remain zero.

    ASSERT_EQ(encode_decode_compare(input, bg, Z, F), 0U) << "Seed=" << seed;
  }
}

/// Test with all-zero input.
TEST_P(LDPCDecoderGpuCpuTest, AllZeros)
{
  auto [bg, Z, filler_percent] = GetParam();
  if (filler_percent != 0) {
    GTEST_SKIP() << "AllZeros only runs with filler_percent=0";
  }

  unsigned Kb = (bg == ldpc_base_graph_type::BG1) ? 22 : 10;
  unsigned K  = Kb * Z;

  dynamic_bit_buffer input(K);
  // dynamic_bit_buffer initializes to all zeros.

  ASSERT_EQ(encode_decode_compare(input, bg, Z, 0), 0U);
}

/// Test with all-one input.
TEST_P(LDPCDecoderGpuCpuTest, AllOnes)
{
  auto [bg, Z, filler_percent] = GetParam();
  if (filler_percent != 0) {
    GTEST_SKIP() << "AllOnes only runs with filler_percent=0";
  }

  unsigned Kb = (bg == ldpc_base_graph_type::BG1) ? 22 : 10;
  unsigned K  = Kb * Z;

  dynamic_bit_buffer input(K);
  for (unsigned i = 0; i < K; i += 8) {
    unsigned nbits = std::min(8U, K - i);
    input.insert(static_cast<unsigned>(0xFF >> (8 - nbits)), i, nbits);
  }

  ASSERT_EQ(encode_decode_compare(input, bg, Z, 0), 0U);
}

/// Test with filler bits (25% and 50%).
TEST_P(LDPCDecoderGpuCpuTest, WithFillerBits)
{
  auto [bg, Z, filler_percent] = GetParam();
  if (filler_percent == 0) {
    GTEST_SKIP() << "WithFillerBits only runs with filler_percent > 0";
  }

  unsigned Kb = (bg == ldpc_base_graph_type::BG1) ? 22 : 10;
  unsigned K  = Kb * Z;

  // Compute filler bits rounded to Z boundary.
  unsigned F_nodes = (filler_percent * Kb) / 100;
  unsigned F       = F_nodes * Z;
  unsigned Kd      = K - F;

  std::mt19937                           rgen(42 + Z + filler_percent);
  std::uniform_int_distribution<uint8_t> byte_gen(0, 255);

  dynamic_bit_buffer input(K);
  // Fill info bits (first Kd) with random data.
  for (unsigned i = 0; i < Kd; i += 8) {
    unsigned nbits = std::min(8U, Kd - i);
    input.insert(byte_gen(rgen), i, nbits);
  }
  // Filler bits (positions Kd..K-1) remain zero.

  ASSERT_EQ(encode_decode_compare(input, bg, Z, F), 0U);
}

/// Generate test cases: all valid Z for both BG1 and BG2, with filler percentages 0, 25, 50.
static std::vector<decoder_test_params> generate_decoder_test_cases()
{
  std::vector<decoder_test_params> cases;
  for (unsigned Z : ALL_LIFTING_SIZES) {
    for (unsigned filler_pct : {0U, 25U, 50U}) {
      cases.push_back({ldpc_base_graph_type::BG1, Z, filler_pct});
      cases.push_back({ldpc_base_graph_type::BG2, Z, filler_pct});
    }
  }
  return cases;
}

static std::string decoder_test_name_generator(const ::testing::TestParamInfo<decoder_test_params>& param_info)
{
  return fmt::format("BG{}_Z{}_F{}",
                     (param_info.param.bg == ldpc_base_graph_type::BG1) ? 1 : 2,
                     param_info.param.Z,
                     param_info.param.filler_percent);
}

INSTANTIATE_TEST_SUITE_P(LDPCDecoderComparison,
                         LDPCDecoderGpuCpuTest,
                         ::testing::ValuesIn(generate_decoder_test_cases()),
                         decoder_test_name_generator);
