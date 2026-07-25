// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <polar.h>

/// Compacts interleaved PUSCH LLRs into SCH-only order on device.
void pusch_compact_sch_llrs_half(const void*  d_full_llrs_half,
                                 void*        d_sch_llrs_half,
                                 const int*   d_sch_re_indices,
                                 unsigned     nof_sch_re,
                                 unsigned     nof_bits_per_re,
                                 cudaStream_t stream,
                                 const int*   d_sch_erasure_re_indices = nullptr,
                                 unsigned     nof_sch_erasure_re       = 0);

/// Result of a device-side UCI decode. Short-block decoders use only the first 11 payload entries.
using pusch_uci_short_decode_result = polar_uci_decode_result_t;

/// Decodes a short-block UCI payload from compact FP16 LLRs on device.
void pusch_decode_uci_short_block_half(const void*  d_llrs_half,
                                       unsigned     nof_llrs,
                                       unsigned     nof_payload_bits,
                                       unsigned     bits_per_symbol,
                                       void*        d_result,
                                       cudaStream_t stream);

/// Decodes HARQ-ACK and CSI Part 1 short-block UCI directly from the full FP16 PUSCH LLR stream.
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
                                                  cudaStream_t stream);

/// Compacts SCH LLRs and decodes short-block HARQ-ACK/CSI Part 1 directly from the full FP16 PUSCH LLR stream.
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
                                                        const int*   d_sch_erasure_re_indices = nullptr,
                                                        unsigned     nof_sch_erasure_re       = 0);
