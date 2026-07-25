// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/phy/upper/channel_coding/channel_coding_factories.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_encoder.h"
#include <cuda_runtime.h>
#include <memory>

namespace ocudu {

/// Maximum batch size for GPU LDPC encoding.
static constexpr unsigned MAX_LDPC_BATCH_SIZE = 64;

/// Maximum input words per codeblock (22 * 384 / 32 for BG1).
static constexpr unsigned MAX_CB_INPUT_WORDS = 264;

/// Maximum output words per codeblock (68 * 384 / 32 for BG1).
static constexpr unsigned MAX_CB_OUTPUT_WORDS = 816;

/// GPU LDPC encoder buffer info for GPU-resident mode.
struct gpu_ldpc_buffer_info {
  uint32_t*    d_encoded_bits;  ///< Device pointer to encoded bits.
  unsigned     num_cbs;         ///< Number of codeblocks.
  unsigned     output_stride;   ///< Words per codeblock output.
  cudaStream_t stream;          ///< CUDA stream used for encoding.
  bool         valid;           ///< True if GPU data is valid.
};

/// Batched LDPC encoder interface for GPU acceleration.
class ldpc_encoder_batch
{
public:
  virtual ~ldpc_encoder_batch() = default;

  /// Encode a batch of codeblocks on GPU.
  ///
  /// \param[in]  inputs  Span of input bit buffers (one per codeblock).
  /// \param[out] outputs Span of output buffer pointers.
  /// \param[in]  cfg     LDPC encoder configuration (same for all CBs).
  virtual void encode_batch(span<const bit_buffer>      inputs,
                            span<ldpc_encoder_buffer*>  outputs,
                            const ldpc_encoder::configuration& cfg) = 0;

  /// Encode a batch with GPU-resident input/output (device-to-device).
  ///
  /// \param[in]  d_inputs      Device pointer to packed input bits (all CBs contiguous).
  /// \param[out] d_outputs     Device pointer to packed output bits (all CBs contiguous).
  /// \param[in]  num_cbs       Number of codeblocks in the batch.
  /// \param[in]  input_stride  Words between consecutive CB inputs.
  /// \param[in]  output_stride Words between consecutive CB outputs.
  /// \param[in]  cfg           LDPC encoder configuration.
  /// \param[in]  stream        CUDA stream for async execution.
  virtual void encode_batch_gpu_resident(const uint32_t*                    d_inputs,
                                         uint32_t*                          d_outputs,
                                         unsigned                           num_cbs,
                                         unsigned                           input_stride,
                                         unsigned                           output_stride,
                                         const ldpc_encoder::configuration& cfg,
                                         cudaStream_t                       stream) = 0;

  /// Get GPU buffer info after encode_batch() for GPU-resident downstream processing.
  virtual gpu_ldpc_buffer_info get_gpu_buffer_info() const = 0;

  /// Check if GPU is available for encoding.
  virtual bool is_gpu_available() const = 0;

  /// Get the maximum supported batch size.
  virtual unsigned get_max_batch_size() const = 0;
};

/// Create a batched GPU LDPC encoder with CPU fallback.
std::unique_ptr<ldpc_encoder_batch> create_ldpc_encoder_batch_cuda(
    std::shared_ptr<ldpc_encoder_factory> cpu_encoder_factory);

/// Check if batched GPU LDPC encoding is available.
bool is_ldpc_encoder_batch_gpu_available();

} // namespace ocudu
