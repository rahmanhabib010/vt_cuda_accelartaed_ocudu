// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief Validates GPU LDPC encoder (CUDA) against CPU LDPC encoder (generic).
///
/// Uses the same factory path as the gNB: create_ldpc_encoder_factory_sw for CPU,
/// create_ldpc_encoder_factory_cuda for GPU. Tests all valid lifting sizes for
/// both BG1 and BG2, comparing encoded codeblocks bit-by-bit.

#include "ocudu/adt/bit_buffer.h"
#include "ocudu/phy/upper/channel_coding/channel_coding_factories.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_encoder.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_encoder_buffer.h"
#include "ldpc_cuda_factories.h"
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace ocudu;
using namespace ocudu::ldpc;

/// All 51 valid 5G NR lifting sizes (TS 38.212 Table 5.3.2-1).
static constexpr unsigned ALL_LIFTING_SIZES[] = {
    2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  18,  20,
    22,  24,  26,  28,  30,  32,  36,  40,  44,  48,  52,  56,  60,  64,  72,  80,  88,
    96,  104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384};

struct encoder_test_params {
  ldpc_base_graph_type bg;
  unsigned             Z;
};

std::ostream& operator<<(std::ostream& os, const encoder_test_params& p)
{
  return os << "BG" << ((p.bg == ldpc_base_graph_type::BG1) ? 1 : 2) << "_Z" << p.Z;
}

class LDPCEncoderGpuCpuTest : public ::testing::TestWithParam<encoder_test_params>
{
public:
  static void SetUpTestSuite()
  {
    if (!cpu_factory) {
      cpu_factory = create_ldpc_encoder_factory_sw("generic");
      ASSERT_NE(cpu_factory, nullptr) << "Failed to create CPU encoder factory";
    }
    if (!gpu_factory) {
      gpu_factory = create_ldpc_encoder_factory_cuda(cpu_factory);
      ASSERT_NE(gpu_factory, nullptr) << "Failed to create GPU encoder factory";
    }
  }

protected:
  /// Compare CPU and GPU encoded outputs for a given input.
  /// Returns the number of bit mismatches (0 = pass).
  unsigned compare_encode(const bit_buffer& input, ldpc_base_graph_type bg, unsigned Z)
  {
    auto cpu_enc = cpu_factory->create();
    auto gpu_enc = gpu_factory->create();
    EXPECT_NE(cpu_enc, nullptr);
    EXPECT_NE(gpu_enc, nullptr);
    if (!cpu_enc || !gpu_enc) {
      return 999999;
    }

    ldpc_encoder::configuration cfg;
    cfg.base_graph   = bg;
    cfg.lifting_size = static_cast<lifting_size_t>(Z);

    const ldpc_encoder_buffer& cpu_buf = cpu_enc->encode(input, cfg);
    const ldpc_encoder_buffer& gpu_buf = gpu_enc->encode(input, cfg);

    unsigned cpu_len = cpu_buf.get_codeblock_length();
    unsigned gpu_len = gpu_buf.get_codeblock_length();

    // CPU returns bg_N_short * Z bits (nodes 2 through bg_N_short+1).
    // GPU wrapper may return N_full * Z (including punctured cols 0-1) when using GPU,
    // or bg_N_short * Z when falling back to CPU.
    std::vector<uint8_t> cpu_out(cpu_len);
    std::vector<uint8_t> gpu_out(cpu_len);

    cpu_buf.write_codeblock(cpu_out, 0);

    if (gpu_len > cpu_len) {
      // GPU returned full codeblock: skip first 2*Z bits (punctured columns 0-1).
      gpu_buf.write_codeblock(gpu_out, 2 * Z);
    } else {
      // Same length: GPU fell back to CPU (trivially matching) or same format.
      gpu_buf.write_codeblock(gpu_out, 0);
    }

    unsigned mismatches    = 0;
    int      first_mismatch = -1;
    for (unsigned i = 0; i < cpu_len; i++) {
      if (cpu_out[i] != gpu_out[i]) {
        if (first_mismatch < 0) {
          first_mismatch = static_cast<int>(i);
        }
        mismatches++;
      }
    }

    if (mismatches > 0) {
      // Identify which node/column the first mismatch is in.
      unsigned mismatch_col = 2 + static_cast<unsigned>(first_mismatch) / Z;
      unsigned mismatch_z   = static_cast<unsigned>(first_mismatch) % Z;
      ADD_FAILURE() << "BG" << ((bg == ldpc_base_graph_type::BG1) ? 1 : 2) << " Z=" << Z << ": " << mismatches
                    << " bit mismatches out of " << cpu_len << " (first at position " << first_mismatch << " = col "
                    << mismatch_col << " z=" << mismatch_z << ", cpu=" << (int)cpu_out[first_mismatch]
                    << " gpu=" << (int)gpu_out[first_mismatch] << ")"
                    << " gpu_len=" << gpu_len << " cpu_len=" << cpu_len
                    << (gpu_len == cpu_len ? " [GPU fallback to CPU]" : " [GPU kernel]");
    }

    return mismatches;
  }

  static std::shared_ptr<ldpc_encoder_factory> cpu_factory;
  static std::shared_ptr<ldpc_encoder_factory> gpu_factory;
};

std::shared_ptr<ldpc_encoder_factory> LDPCEncoderGpuCpuTest::cpu_factory = nullptr;
std::shared_ptr<ldpc_encoder_factory> LDPCEncoderGpuCpuTest::gpu_factory = nullptr;

/// Test with random data (multiple seeds).
TEST_P(LDPCEncoderGpuCpuTest, RandomData)
{
  auto [bg, Z] = GetParam();
  unsigned Kb = (bg == ldpc_base_graph_type::BG1) ? 22 : 10;
  unsigned K  = Kb * Z;

  // Test with 3 different random seeds.
  for (unsigned seed = 0; seed < 3; seed++) {
    std::mt19937                           rgen(seed * 1000 + Z);
    std::uniform_int_distribution<uint8_t> byte_gen(0, 255);

    dynamic_bit_buffer input(K);
    for (unsigned i = 0; i < K; i += 8) {
      unsigned nbits = std::min(8U, K - i);
      unsigned val = byte_gen(rgen);
      if (nbits < 8) {
        val &= (1U << nbits) - 1;
      }
      input.insert(val, i, nbits);
    }

    ASSERT_EQ(compare_encode(input, bg, Z), 0U) << "Seed=" << seed;
  }
}

/// Test with all-zero input.
TEST_P(LDPCEncoderGpuCpuTest, AllZeros)
{
  auto [bg, Z] = GetParam();
  unsigned Kb = (bg == ldpc_base_graph_type::BG1) ? 22 : 10;
  unsigned K  = Kb * Z;

  dynamic_bit_buffer input(K);
  // dynamic_bit_buffer initializes to all zeros.

  ASSERT_EQ(compare_encode(input, bg, Z), 0U);
}

/// Test with all-one input.
TEST_P(LDPCEncoderGpuCpuTest, AllOnes)
{
  auto [bg, Z] = GetParam();
  unsigned Kb = (bg == ldpc_base_graph_type::BG1) ? 22 : 10;
  unsigned K  = Kb * Z;

  dynamic_bit_buffer input(K);
  for (unsigned i = 0; i < K; i += 8) {
    unsigned nbits = std::min(8U, K - i);
    input.insert(static_cast<unsigned>(0xFF >> (8 - nbits)), i, nbits);
  }

  ASSERT_EQ(compare_encode(input, bg, Z), 0U);
}

/// Test with filler bits (trailing zeros in the message, simulating real-world usage).
TEST_P(LDPCEncoderGpuCpuTest, WithFillerBits)
{
  auto [bg, Z] = GetParam();
  unsigned Kb = (bg == ldpc_base_graph_type::BG1) ? 22 : 10;
  unsigned K  = Kb * Z;

  // Filler = half the message length (rounded to Z boundary).
  unsigned filler = (Kb / 2) * Z;
  unsigned info   = K - filler;

  std::mt19937                           rgen(42 + Z);
  std::uniform_int_distribution<uint8_t> byte_gen(0, 255);

  dynamic_bit_buffer input(K);
  // Fill info bits with random data.
  for (unsigned i = 0; i < info; i += 8) {
    unsigned nbits = std::min(8U, info - i);
    input.insert(byte_gen(rgen), i, nbits);
  }
  // Filler bits (positions info..K-1) are already zero.

  ASSERT_EQ(compare_encode(input, bg, Z), 0U);
}

/// Generate test cases: all valid Z for both BG1 and BG2.
static std::vector<encoder_test_params> generate_encoder_test_cases()
{
  std::vector<encoder_test_params> cases;
  for (unsigned Z : ALL_LIFTING_SIZES) {
    cases.push_back({ldpc_base_graph_type::BG1, Z});
    cases.push_back({ldpc_base_graph_type::BG2, Z});
  }
  return cases;
}

static std::string encoder_test_name_generator(const ::testing::TestParamInfo<encoder_test_params>& param_info)
{
  return fmt::format(
      "BG{}_Z{}", (param_info.param.bg == ldpc_base_graph_type::BG1) ? 1 : 2, param_info.param.Z);
}

INSTANTIATE_TEST_SUITE_P(LDPCEncoderComparison,
                         LDPCEncoderGpuCpuTest,
                         ::testing::ValuesIn(generate_encoder_test_cases()),
                         encoder_test_name_generator);
