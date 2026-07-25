// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "cuda/pusch_sch_llr_compactor.h"
#include "ocudu/ocuduvec/bit.h"
#include "ocudu/phy/upper/channel_coding/short/short_block_encoder.h"
#include "ocudu/ran/uci/uci_info.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace ocudu;

namespace {

class CudaUciShortBlockDetectorFixture : public ::testing::Test
{
protected:
  void SetUp() override
  {
    encoder = create_short_block_encoder();
    ASSERT_NE(encoder, nullptr);
  }

  std::unique_ptr<short_block_encoder> encoder;
};

TEST_F(CudaUciShortBlockDetectorFixture, MatchesShortBlockEncoder)
{
  static constexpr modulation_scheme modulation      = modulation_scheme::QPSK;
  static constexpr float             llr_magnitude   = 16.0F;
  static constexpr unsigned          bits_per_symbol = 2;

  for (unsigned nof_payload_bits = 1; nof_payload_bits <= 11; ++nof_payload_bits) {
    unsigned nof_encoded_bits = calculate_uci_min_encoded_bits(nof_payload_bits);
    for (unsigned message = 0, message_end = 1U << nof_payload_bits; message != message_end; ++message) {
      static_vector<uint8_t, 32> unpacked_message(nof_payload_bits);
      ocuduvec::bit_unpack(unpacked_message, message, nof_payload_bits);

      static_vector<uint8_t, 32> codeword(nof_encoded_bits);
      encoder->encode(codeword, unpacked_message, modulation);

      std::vector<__half> llrs(nof_encoded_bits);
      for (unsigned i = 0; i != nof_encoded_bits; ++i) {
        llrs[i] = __float2half(codeword[i] ? -llr_magnitude : llr_magnitude);
      }

      __half*                        d_llrs   = nullptr;
      pusch_uci_short_decode_result* d_result = nullptr;
      ASSERT_EQ(cudaSuccess, cudaMalloc(&d_llrs, llrs.size() * sizeof(__half)));
      ASSERT_EQ(cudaSuccess, cudaMalloc(&d_result, sizeof(pusch_uci_short_decode_result)));
      ASSERT_EQ(cudaSuccess, cudaMemcpy(d_llrs, llrs.data(), llrs.size() * sizeof(__half), cudaMemcpyHostToDevice));

      pusch_decode_uci_short_block_half(d_llrs, nof_encoded_bits, nof_payload_bits, bits_per_symbol, d_result, nullptr);
      ASSERT_EQ(cudaSuccess, cudaGetLastError());

      pusch_uci_short_decode_result result = {};
      ASSERT_EQ(cudaSuccess, cudaMemcpy(&result, d_result, sizeof(result), cudaMemcpyDeviceToHost));
      ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

      ASSERT_EQ(1, result.decoded) << "payload_bits=" << nof_payload_bits << " message=" << message;
      ASSERT_EQ(1, result.status) << "payload_bits=" << nof_payload_bits << " message=" << message;
      ASSERT_EQ(nof_payload_bits, result.nof_bits) << "payload_bits=" << nof_payload_bits << " message=" << message;
      for (unsigned i = 0; i != nof_payload_bits; ++i) {
        ASSERT_EQ(unpacked_message[i], result.payload[i])
            << "payload_bits=" << nof_payload_bits << " message=" << message << " bit=" << i;
      }

      cudaFree(d_result);
      cudaFree(d_llrs);
    }
  }
}

TEST(PuschSchLlrCompactor, ErasesHarqPlaceholdersInSchCompaction)
{
  static constexpr unsigned nof_bits_per_re = 2;

  std::vector<__half> full_llrs(12);
  for (unsigned i = 0; i != full_llrs.size(); ++i) {
    full_llrs[i] = __float2half(static_cast<float>(i + 1));
  }

  std::vector<int>    sch_re_indices   = {0, 1, 2, 3, 4, 5};
  std::vector<int>    erase_re_indices = {1, 4};
  std::vector<__half> compacted_llrs(full_llrs.size());

  __half* d_full_llrs = nullptr;
  __half* d_sch_llrs  = nullptr;
  int*    d_sch_re    = nullptr;
  int*    d_erase_re  = nullptr;

  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_full_llrs, full_llrs.size() * sizeof(__half)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_sch_llrs, compacted_llrs.size() * sizeof(__half)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_sch_re, sch_re_indices.size() * sizeof(int)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_erase_re, erase_re_indices.size() * sizeof(int)));
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(d_full_llrs, full_llrs.data(), full_llrs.size() * sizeof(__half), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(d_sch_re, sch_re_indices.data(), sch_re_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
  ASSERT_EQ(
      cudaSuccess,
      cudaMemcpy(d_erase_re, erase_re_indices.data(), erase_re_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

  pusch_compact_sch_llrs_half(d_full_llrs,
                              d_sch_llrs,
                              d_sch_re,
                              static_cast<unsigned>(sch_re_indices.size()),
                              nof_bits_per_re,
                              nullptr,
                              d_erase_re,
                              static_cast<unsigned>(erase_re_indices.size()));
  ASSERT_EQ(cudaSuccess, cudaGetLastError());
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  ASSERT_EQ(
      cudaSuccess,
      cudaMemcpy(compacted_llrs.data(), d_sch_llrs, compacted_llrs.size() * sizeof(__half), cudaMemcpyDeviceToHost));

  for (unsigned re = 0; re != sch_re_indices.size(); ++re) {
    bool erased = (sch_re_indices[re] == 1) || (sch_re_indices[re] == 4);
    for (unsigned bit = 0; bit != nof_bits_per_re; ++bit) {
      unsigned llr_index = re * nof_bits_per_re + bit;
      float    expected  = erased ? 0.0F : static_cast<float>(llr_index + 1);
      ASSERT_FLOAT_EQ(expected, __half2float(compacted_llrs[llr_index])) << "re=" << re << " bit=" << bit;
    }
  }

  cudaFree(d_erase_re);
  cudaFree(d_sch_re);
  cudaFree(d_sch_llrs);
  cudaFree(d_full_llrs);
}

TEST(PuschSchLlrCompactor, FusedShortUciCompactionErasesHarqPlaceholders)
{
  static constexpr unsigned nof_bits_per_re = 2;

  std::vector<__half> full_llrs(12);
  for (unsigned i = 0; i != full_llrs.size(); ++i) {
    full_llrs[i] = __float2half(static_cast<float>(i + 1));
  }

  std::vector<int>    sch_re_indices  = {0, 1, 2, 3, 4, 5};
  std::vector<int>    harq_re_indices = {1, 4};
  std::vector<__half> compacted_llrs(full_llrs.size());

  __half*                        d_full_llrs   = nullptr;
  __half*                        d_sch_llrs    = nullptr;
  int*                           d_sch_re      = nullptr;
  int*                           d_harq_re     = nullptr;
  pusch_uci_short_decode_result* d_harq_result = nullptr;

  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_full_llrs, full_llrs.size() * sizeof(__half)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_sch_llrs, compacted_llrs.size() * sizeof(__half)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_sch_re, sch_re_indices.size() * sizeof(int)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_harq_re, harq_re_indices.size() * sizeof(int)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_harq_result, sizeof(pusch_uci_short_decode_result)));
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(d_full_llrs, full_llrs.data(), full_llrs.size() * sizeof(__half), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess,
            cudaMemcpy(d_sch_re, sch_re_indices.data(), sch_re_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
  ASSERT_EQ(
      cudaSuccess,
      cudaMemcpy(d_harq_re, harq_re_indices.data(), harq_re_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

  pusch_compact_sch_and_decode_uci_short_blocks_half(d_full_llrs,
                                                     d_sch_llrs,
                                                     d_sch_re,
                                                     static_cast<unsigned>(sch_re_indices.size()),
                                                     d_harq_re,
                                                     static_cast<unsigned>(harq_re_indices.size()),
                                                     1,
                                                     d_harq_result,
                                                     nullptr,
                                                     0,
                                                     0,
                                                     nullptr,
                                                     nof_bits_per_re,
                                                     nof_bits_per_re,
                                                     nullptr,
                                                     d_harq_re,
                                                     static_cast<unsigned>(harq_re_indices.size()));
  ASSERT_EQ(cudaSuccess, cudaGetLastError());
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  ASSERT_EQ(
      cudaSuccess,
      cudaMemcpy(compacted_llrs.data(), d_sch_llrs, compacted_llrs.size() * sizeof(__half), cudaMemcpyDeviceToHost));

  for (unsigned re = 0; re != sch_re_indices.size(); ++re) {
    bool erased = (sch_re_indices[re] == 1) || (sch_re_indices[re] == 4);
    for (unsigned bit = 0; bit != nof_bits_per_re; ++bit) {
      unsigned llr_index = re * nof_bits_per_re + bit;
      float    expected  = erased ? 0.0F : static_cast<float>(llr_index + 1);
      ASSERT_FLOAT_EQ(expected, __half2float(compacted_llrs[llr_index])) << "re=" << re << " bit=" << bit;
    }
  }

  cudaFree(d_harq_result);
  cudaFree(d_harq_re);
  cudaFree(d_sch_re);
  cudaFree(d_sch_llrs);
  cudaFree(d_full_llrs);
}

} // namespace
