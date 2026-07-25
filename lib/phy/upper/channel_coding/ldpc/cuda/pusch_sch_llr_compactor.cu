// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "pusch_sch_llr_compactor.h"
#include <cuda_fp16.h>
#include <math_constants.h>

namespace {

constexpr unsigned MAX_UCI_SHORT_BITS      = 11;
constexpr unsigned MAX_UCI_SHORT_CODE_BITS = 32;

__device__ __constant__ uint8_t UCI_SHORT_BASIS[MAX_UCI_SHORT_BITS][MAX_UCI_SHORT_CODE_BITS] = {
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0},
    {0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0},
    {0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0},
    {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0},
    {0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0},
    {0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0},
    {0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0},
    {0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0}};

__device__ unsigned calculate_uci_min_encoded_bits_device(unsigned nof_payload_bits)
{
  constexpr unsigned min_small_block_rm_size[MAX_UCI_SHORT_BITS] = {2, 3, 9, 10, 11, 12, 13, 14, 14, 15, 17};
  if (nof_payload_bits == 0 || nof_payload_bits > MAX_UCI_SHORT_BITS) {
    return 0;
  }
  return min_small_block_rm_size[nof_payload_bits - 1];
}

__device__ void rate_dematch_short_half(float* output, unsigned output_size, const __half* input, unsigned input_size)
{
  for (unsigned i = 0; i != output_size; ++i) {
    output[i] = 0.0F;
  }

  unsigned nof_copy = min(input_size, output_size);
  for (unsigned i = 0; i != nof_copy; ++i) {
    output[i] = __half2float(input[i]);
  }

  if (input_size <= output_size) {
    return;
  }

  unsigned offset = output_size;
  while (offset != input_size) {
    unsigned block_size = min(output_size, input_size - offset);
    for (unsigned i = 0; i != block_size; ++i) {
      output[i] += __half2float(input[offset + i]);
    }
    offset += block_size;
  }
}

__device__ float
read_indexed_uci_llr(const __half* full_llrs, const int* re_indices, unsigned llr_index, unsigned nof_bits_per_re)
{
  unsigned re_index  = llr_index / nof_bits_per_re;
  unsigned bit_index = llr_index - re_index * nof_bits_per_re;
  return __half2float(full_llrs[static_cast<unsigned>(re_indices[re_index]) * nof_bits_per_re + bit_index]);
}

__device__ bool contains_sorted_re_index(const int* re_indices, unsigned nof_re, unsigned re_index)
{
  unsigned low  = 0;
  unsigned high = nof_re;
  while (low != high) {
    unsigned mid       = low + (high - low) / 2;
    unsigned candidate = static_cast<unsigned>(re_indices[mid]);
    if (candidate < re_index) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return (low != nof_re) && (static_cast<unsigned>(re_indices[low]) == re_index);
}

__device__ void rate_dematch_short_half_indexed(float*        output,
                                                unsigned      output_size,
                                                const __half* full_llrs,
                                                const int*    re_indices,
                                                unsigned      input_size,
                                                unsigned      nof_bits_per_re)
{
  for (unsigned i = 0; i != output_size; ++i) {
    output[i] = 0.0F;
  }

  unsigned nof_copy = min(input_size, output_size);
  for (unsigned i = 0; i != nof_copy; ++i) {
    output[i] = read_indexed_uci_llr(full_llrs, re_indices, i, nof_bits_per_re);
  }

  if (input_size <= output_size) {
    return;
  }

  unsigned offset = output_size;
  while (offset != input_size) {
    unsigned block_size = min(output_size, input_size - offset);
    for (unsigned i = 0; i != block_size; ++i) {
      output[i] += read_indexed_uci_llr(full_llrs, re_indices, offset + i, nof_bits_per_re);
    }
    offset += block_size;
  }
}

__device__ float detect_2_device(uint8_t* output, const float* input, unsigned input_size)
{
  float llr[3] = {};

  if (input_size == 3) {
    llr[0] = input[0];
    llr[1] = input[1];
    llr[2] = input[2];
  } else {
    unsigned step = input_size / 3 - 2;
    llr[0]        = input[0] + input[step + 3];
    llr[1]        = input[1] + input[2 * step + 4];
    llr[2]        = input[step + 2] + input[2 * step + 5];
  }

  constexpr int table[4][3] = {{1, 1, 1}, {-1, 1, -1}, {1, -1, -1}, {-1, -1, 1}};

  unsigned max_idx    = 0;
  float    max_metric = -CUDART_INF_F;
  for (unsigned cdwd_idx = 0; cdwd_idx != 4; ++cdwd_idx) {
    float metric = 0.0F;
    for (unsigned i = 0; i != 3; ++i) {
      metric += llr[i] * static_cast<float>(table[cdwd_idx][i]);
    }
    if (metric > max_metric) {
      max_metric = metric;
      max_idx    = cdwd_idx;
    }
  }

  output[0] = static_cast<uint8_t>(max_idx & 1U);
  output[1] = static_cast<uint8_t>((max_idx >> 1U) & 1U);

  float norm_sqr = 0.0F;
  for (unsigned i = 0; i != 3; ++i) {
    norm_sqr += llr[i] * llr[i];
  }
  float metric_sqr = max_metric * max_metric;
  float denom      = 3.0F * norm_sqr - metric_sqr;
  return (denom > 0.0F) ? (2.0F * metric_sqr / denom) : CUDART_INF_F;
}

__device__ float detect_3_11_device(uint8_t* output, const float* input, unsigned nof_payload_bits)
{
  unsigned nof_codewords = 1U << (nof_payload_bits - 1U);

  unsigned max_idx    = 0;
  float    max_metric = -CUDART_INF_F;
  uint8_t  bit0       = 0;
  for (unsigned cdwd_idx = 0; cdwd_idx != nof_codewords; ++cdwd_idx) {
    unsigned value  = 2U * cdwd_idx;
    float    metric = 0.0F;

    for (unsigned bit = 0; bit != MAX_UCI_SHORT_CODE_BITS; ++bit) {
      uint8_t code_bit = 0;
      for (unsigned msg_bit = 0; msg_bit != MAX_UCI_SHORT_BITS; ++msg_bit) {
        if (((value >> msg_bit) & 1U) != 0U) {
          code_bit ^= UCI_SHORT_BASIS[msg_bit][bit];
        }
      }
      metric += input[bit] * (code_bit ? -1.0F : 1.0F);
    }

    float metric_abs = fabsf(metric);
    if (metric_abs > max_metric) {
      max_metric = metric_abs;
      max_idx    = cdwd_idx;
      bit0       = static_cast<uint8_t>(metric < 0.0F);
    }
  }

  unsigned detected = 2U * max_idx + bit0;
  for (unsigned i = 0; i != nof_payload_bits; ++i) {
    output[i] = static_cast<uint8_t>((detected >> i) & 1U);
  }

  float norm_sqr = 0.0F;
  for (unsigned i = 0; i != MAX_UCI_SHORT_CODE_BITS; ++i) {
    norm_sqr += input[i] * input[i];
  }
  float metric_sqr = max_metric * max_metric;
  float denom      = static_cast<float>(MAX_UCI_SHORT_CODE_BITS) * norm_sqr - metric_sqr;
  return (denom > 0.0F) ? ((static_cast<float>(MAX_UCI_SHORT_CODE_BITS - 1) * metric_sqr) / denom) : CUDART_INF_F;
}

__global__ void pusch_decode_uci_short_block_half_kernel(const __half* __restrict__ llrs,
                                                         unsigned                       nof_llrs,
                                                         unsigned                       nof_payload_bits,
                                                         unsigned                       bits_per_symbol,
                                                         pusch_uci_short_decode_result* result)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  result->nof_bits = static_cast<uint8_t>(min(nof_payload_bits, MAX_UCI_SHORT_BITS));
  result->status   = 0;
  result->decoded  = 0;
  for (unsigned i = 0; i != MAX_UCI_SHORT_BITS; ++i) {
    result->payload[i] = 1;
  }

  if (nof_payload_bits == 0 || nof_payload_bits > MAX_UCI_SHORT_BITS || bits_per_symbol == 0) {
    return;
  }

  unsigned non_zero_count = 0;
  for (unsigned i = 0; i != nof_llrs; ++i) {
    non_zero_count += (__half2float(llrs[i]) != 0.0F) ? 1U : 0U;
  }
  if (non_zero_count < calculate_uci_min_encoded_bits_device(nof_payload_bits)) {
    result->decoded = 1;
    return;
  }
  if (nof_payload_bits <= 2 && nof_llrs < bits_per_symbol) {
    result->decoded = 1;
    return;
  }

  float   tmp[MAX_UCI_SHORT_CODE_BITS] = {};
  float   metric                       = 0.0F;
  uint8_t payload[MAX_UCI_SHORT_BITS]  = {};
  if (nof_payload_bits == 1) {
    rate_dematch_short_half(tmp, bits_per_symbol, llrs, nof_llrs);
    payload[0] = (tmp[0] > 0.0F) ? 0 : 1;
    metric     = 1.0F;
  } else if (nof_payload_bits == 2) {
    unsigned tmp_size = 3U * bits_per_symbol;
    if (tmp_size > MAX_UCI_SHORT_CODE_BITS) {
      return;
    }
    rate_dematch_short_half(tmp, tmp_size, llrs, nof_llrs);
    metric = detect_2_device(payload, tmp, tmp_size);
  } else {
    rate_dematch_short_half(tmp, MAX_UCI_SHORT_CODE_BITS, llrs, nof_llrs);
    metric = detect_3_11_device(payload, tmp, nof_payload_bits);
  }

  constexpr float thresholds[MAX_UCI_SHORT_BITS] = {
      0.0F, 0.0F, 12.0F, 14.0F, 16.0F, 18.0F, 20.0F, 22.0F, 24.0F, 26.0F, 29.0F};
  for (unsigned i = 0; i != nof_payload_bits; ++i) {
    result->payload[i] = payload[i];
  }
  result->status  = (metric > thresholds[nof_payload_bits - 1]) ? 1 : 0;
  result->decoded = 1;
}

__device__ void decode_short_block_indexed(const __half*                  full_llrs,
                                           const int*                     re_indices,
                                           unsigned                       nof_re,
                                           unsigned                       nof_payload_bits,
                                           unsigned                       nof_bits_per_re,
                                           unsigned                       bits_per_symbol,
                                           pusch_uci_short_decode_result* result)
{
  result->nof_bits = static_cast<uint8_t>(min(nof_payload_bits, MAX_UCI_SHORT_BITS));
  result->status   = 0;
  result->decoded  = 0;
  for (unsigned i = 0; i != MAX_UCI_SHORT_BITS; ++i) {
    result->payload[i] = 1;
  }

  if (!full_llrs || !re_indices || nof_re == 0 || nof_payload_bits == 0 || nof_payload_bits > MAX_UCI_SHORT_BITS ||
      nof_bits_per_re == 0 || bits_per_symbol == 0) {
    return;
  }

  unsigned nof_llrs       = nof_re * nof_bits_per_re;
  unsigned non_zero_count = 0;
  for (unsigned i = 0; i != nof_llrs; ++i) {
    non_zero_count += (read_indexed_uci_llr(full_llrs, re_indices, i, nof_bits_per_re) != 0.0F) ? 1U : 0U;
  }
  if (non_zero_count < calculate_uci_min_encoded_bits_device(nof_payload_bits)) {
    result->decoded = 1;
    return;
  }
  if (nof_payload_bits <= 2 && nof_llrs < bits_per_symbol) {
    result->decoded = 1;
    return;
  }

  float   tmp[MAX_UCI_SHORT_CODE_BITS] = {};
  float   metric                       = 0.0F;
  uint8_t payload[MAX_UCI_SHORT_BITS]  = {};
  if (nof_payload_bits == 1) {
    rate_dematch_short_half_indexed(tmp, bits_per_symbol, full_llrs, re_indices, nof_llrs, nof_bits_per_re);
    payload[0] = (tmp[0] > 0.0F) ? 0 : 1;
    metric     = 1.0F;
  } else if (nof_payload_bits == 2) {
    unsigned tmp_size = 3U * bits_per_symbol;
    if (tmp_size > MAX_UCI_SHORT_CODE_BITS) {
      return;
    }
    rate_dematch_short_half_indexed(tmp, tmp_size, full_llrs, re_indices, nof_llrs, nof_bits_per_re);
    metric = detect_2_device(payload, tmp, tmp_size);
  } else {
    rate_dematch_short_half_indexed(tmp, MAX_UCI_SHORT_CODE_BITS, full_llrs, re_indices, nof_llrs, nof_bits_per_re);
    metric = detect_3_11_device(payload, tmp, nof_payload_bits);
  }

  constexpr float thresholds[MAX_UCI_SHORT_BITS] = {
      0.0F, 0.0F, 12.0F, 14.0F, 16.0F, 18.0F, 20.0F, 22.0F, 24.0F, 26.0F, 29.0F};
  for (unsigned i = 0; i != nof_payload_bits; ++i) {
    result->payload[i] = payload[i];
  }
  result->status  = (metric > thresholds[nof_payload_bits - 1]) ? 1 : 0;
  result->decoded = 1;
}

__global__ void pusch_decode_uci_short_blocks_from_full_half_kernel(const __half* __restrict__ full_llrs,
                                                                    const int* __restrict__ harq_ack_re_indices,
                                                                    unsigned                       nof_harq_ack_re,
                                                                    unsigned                       nof_harq_ack_bits,
                                                                    pusch_uci_short_decode_result* harq_ack_result,
                                                                    const int* __restrict__ csi_part1_re_indices,
                                                                    unsigned                       nof_csi_part1_re,
                                                                    unsigned                       nof_csi_part1_bits,
                                                                    pusch_uci_short_decode_result* csi_part1_result,
                                                                    unsigned                       nof_bits_per_re,
                                                                    unsigned                       bits_per_symbol)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  if (harq_ack_result) {
    decode_short_block_indexed(full_llrs,
                               harq_ack_re_indices,
                               nof_harq_ack_re,
                               nof_harq_ack_bits,
                               nof_bits_per_re,
                               bits_per_symbol,
                               harq_ack_result);
  }
  if (csi_part1_result) {
    decode_short_block_indexed(full_llrs,
                               csi_part1_re_indices,
                               nof_csi_part1_re,
                               nof_csi_part1_bits,
                               nof_bits_per_re,
                               bits_per_symbol,
                               csi_part1_result);
  }
}

__global__ void
pusch_compact_sch_and_decode_uci_short_blocks_half_kernel(const __half* __restrict__ full_llrs,
                                                          __half* __restrict__ sch_llrs,
                                                          const int* __restrict__ sch_re_indices,
                                                          unsigned nof_sch_llrs,
                                                          const int* __restrict__ harq_ack_re_indices,
                                                          unsigned                       nof_harq_ack_re,
                                                          unsigned                       nof_harq_ack_bits,
                                                          pusch_uci_short_decode_result* harq_ack_result,
                                                          const int* __restrict__ csi_part1_re_indices,
                                                          unsigned                       nof_csi_part1_re,
                                                          unsigned                       nof_csi_part1_bits,
                                                          pusch_uci_short_decode_result* csi_part1_result,
                                                          unsigned                       nof_bits_per_re,
                                                          unsigned                       bits_per_symbol,
                                                          const int* __restrict__ sch_erasure_re_indices,
                                                          unsigned nof_sch_erasure_re)
{
  unsigned dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx < nof_sch_llrs) {
    unsigned sch_re_idx = dst_idx / nof_bits_per_re;
    unsigned bit_idx    = dst_idx - sch_re_idx * nof_bits_per_re;
    unsigned src_re_idx = static_cast<unsigned>(sch_re_indices[sch_re_idx]);
    bool     erase      = (sch_erasure_re_indices != nullptr) &&
                 contains_sorted_re_index(sch_erasure_re_indices, nof_sch_erasure_re, src_re_idx);
    sch_llrs[dst_idx] = erase ? __float2half(0.0F) : full_llrs[src_re_idx * nof_bits_per_re + bit_idx];
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (harq_ack_result) {
      decode_short_block_indexed(full_llrs,
                                 harq_ack_re_indices,
                                 nof_harq_ack_re,
                                 nof_harq_ack_bits,
                                 nof_bits_per_re,
                                 bits_per_symbol,
                                 harq_ack_result);
    }
    if (csi_part1_result) {
      decode_short_block_indexed(full_llrs,
                                 csi_part1_re_indices,
                                 nof_csi_part1_re,
                                 nof_csi_part1_bits,
                                 nof_bits_per_re,
                                 bits_per_symbol,
                                 csi_part1_result);
    }
  }
}

} // namespace

__global__ void pusch_compact_sch_llrs_half_kernel(const __half* __restrict__ full_llrs,
                                                   __half* __restrict__ sch_llrs,
                                                   const int* __restrict__ sch_re_indices,
                                                   unsigned nof_sch_llrs,
                                                   unsigned nof_bits_per_re,
                                                   const int* __restrict__ sch_erasure_re_indices,
                                                   unsigned nof_sch_erasure_re)
{
  unsigned dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx >= nof_sch_llrs) {
    return;
  }

  unsigned sch_re_idx = dst_idx / nof_bits_per_re;
  unsigned bit_idx    = dst_idx - sch_re_idx * nof_bits_per_re;
  unsigned src_re_idx = static_cast<unsigned>(sch_re_indices[sch_re_idx]);
  bool     erase      = (sch_erasure_re_indices != nullptr) &&
               contains_sorted_re_index(sch_erasure_re_indices, nof_sch_erasure_re, src_re_idx);
  sch_llrs[dst_idx] = erase ? __float2half(0.0F) : full_llrs[src_re_idx * nof_bits_per_re + bit_idx];
}

void pusch_compact_sch_llrs_half(const void*  d_full_llrs_half,
                                 void*        d_sch_llrs_half,
                                 const int*   d_sch_re_indices,
                                 unsigned     nof_sch_re,
                                 unsigned     nof_bits_per_re,
                                 cudaStream_t stream,
                                 const int*   d_sch_erasure_re_indices,
                                 unsigned     nof_sch_erasure_re)
{
  if (nof_sch_re == 0 || nof_bits_per_re == 0) {
    return;
  }

  unsigned           nof_sch_llrs = nof_sch_re * nof_bits_per_re;
  constexpr unsigned block_size   = 256;
  unsigned           grid_size    = (nof_sch_llrs + block_size - 1) / block_size;

  pusch_compact_sch_llrs_half_kernel<<<grid_size, block_size, 0, stream>>>(static_cast<const __half*>(d_full_llrs_half),
                                                                           static_cast<__half*>(d_sch_llrs_half),
                                                                           d_sch_re_indices,
                                                                           nof_sch_llrs,
                                                                           nof_bits_per_re,
                                                                           d_sch_erasure_re_indices,
                                                                           nof_sch_erasure_re);
}

void pusch_decode_uci_short_block_half(const void*  d_llrs_half,
                                       unsigned     nof_llrs,
                                       unsigned     nof_payload_bits,
                                       unsigned     bits_per_symbol,
                                       void*        d_result,
                                       cudaStream_t stream)
{
  if (!d_llrs_half || !d_result || nof_llrs == 0 || nof_payload_bits == 0) {
    return;
  }

  pusch_decode_uci_short_block_half_kernel<<<1, 1, 0, stream>>>(static_cast<const __half*>(d_llrs_half),
                                                                nof_llrs,
                                                                nof_payload_bits,
                                                                bits_per_symbol,
                                                                static_cast<pusch_uci_short_decode_result*>(d_result));
}

void pusch_decode_uci_short_blocks_from_full_half(const void*  d_full_llrs_half,
                                                  const int*   d_harq_ack_re_indices,
                                                  unsigned     nof_harq_ack_re,
                                                  unsigned     nof_harq_ack_bits,
                                                  void*        d_harq_ack_result,
                                                  const int*   d_csi_part1_re_indices,
                                                  unsigned     nof_csi_part1_re,
                                                  unsigned     nof_csi_part1_bits,
                                                  void*        d_csi_part1_result,
                                                  unsigned     nof_bits_per_re,
                                                  unsigned     bits_per_symbol,
                                                  cudaStream_t stream)
{
  if (!d_full_llrs_half || nof_bits_per_re == 0 || bits_per_symbol == 0) {
    return;
  }

  pusch_decode_uci_short_blocks_from_full_half_kernel<<<1, 1, 0, stream>>>(
      static_cast<const __half*>(d_full_llrs_half),
      d_harq_ack_re_indices,
      nof_harq_ack_re,
      nof_harq_ack_bits,
      static_cast<pusch_uci_short_decode_result*>(d_harq_ack_result),
      d_csi_part1_re_indices,
      nof_csi_part1_re,
      nof_csi_part1_bits,
      static_cast<pusch_uci_short_decode_result*>(d_csi_part1_result),
      nof_bits_per_re,
      bits_per_symbol);
}

void pusch_compact_sch_and_decode_uci_short_blocks_half(const void*  d_full_llrs_half,
                                                        void*        d_sch_llrs_half,
                                                        const int*   d_sch_re_indices,
                                                        unsigned     nof_sch_re,
                                                        const int*   d_harq_ack_re_indices,
                                                        unsigned     nof_harq_ack_re,
                                                        unsigned     nof_harq_ack_bits,
                                                        void*        d_harq_ack_result,
                                                        const int*   d_csi_part1_re_indices,
                                                        unsigned     nof_csi_part1_re,
                                                        unsigned     nof_csi_part1_bits,
                                                        void*        d_csi_part1_result,
                                                        unsigned     nof_bits_per_re,
                                                        unsigned     bits_per_symbol,
                                                        cudaStream_t stream,
                                                        const int*   d_sch_erasure_re_indices,
                                                        unsigned     nof_sch_erasure_re)
{
  if (!d_full_llrs_half || !d_sch_llrs_half || !d_sch_re_indices || nof_sch_re == 0 || nof_bits_per_re == 0 ||
      bits_per_symbol == 0) {
    return;
  }

  unsigned           nof_sch_llrs = nof_sch_re * nof_bits_per_re;
  constexpr unsigned block_size   = 256;
  unsigned           grid_size    = (nof_sch_llrs + block_size - 1) / block_size;

  pusch_compact_sch_and_decode_uci_short_blocks_half_kernel<<<grid_size, block_size, 0, stream>>>(
      static_cast<const __half*>(d_full_llrs_half),
      static_cast<__half*>(d_sch_llrs_half),
      d_sch_re_indices,
      nof_sch_llrs,
      d_harq_ack_re_indices,
      nof_harq_ack_re,
      nof_harq_ack_bits,
      static_cast<pusch_uci_short_decode_result*>(d_harq_ack_result),
      d_csi_part1_re_indices,
      nof_csi_part1_re,
      nof_csi_part1_bits,
      static_cast<pusch_uci_short_decode_result*>(d_csi_part1_result),
      nof_bits_per_re,
      bits_per_symbol,
      d_sch_erasure_re_indices,
      nof_sch_erasure_re);
}
