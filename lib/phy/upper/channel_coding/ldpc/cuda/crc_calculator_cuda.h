// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief CUDA GPU-accelerated CRC calculator declaration.

#pragma once

#include "ocudu/phy/upper/channel_coding/crc_calculator.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace ocudu {

/// \brief GPU-accelerated CRC calculator using CUDA.
///
/// This class wraps the CUDA CRC kernels (CRC24A, CRC24B, CRC16) to provide
/// GPU-accelerated CRC computation. It falls back to CPU for small data sizes
/// where GPU overhead exceeds the computation benefit.
class crc_calculator_cuda : public crc_calculator
{
public:
  /// Minimum number of bytes to use GPU acceleration.
  /// Below this threshold, CPU fallback is faster due to GPU overhead.
  static constexpr unsigned MIN_GPU_BYTES = 256;

  /// \brief Constructor.
  /// \param[in] poly             CRC polynomial type.
  /// \param[in] fallback_calc    Fallback CPU CRC calculator.
  crc_calculator_cuda(crc_generator_poly                   poly,
                         std::unique_ptr<crc_calculator>      fallback_calc);

  /// Destructor.
  ~crc_calculator_cuda() override;

  // See interface for documentation.
  crc_calculator_checksum_t calculate_byte(span<const uint8_t> data) const override;

  // See interface for documentation.
  crc_calculator_checksum_t calculate_bit(span<const uint8_t> data) const override;

  // See interface for documentation.
  crc_calculator_checksum_t calculate(const bit_buffer& data) const override;

  // See interface for documentation.
  crc_generator_poly get_generator_poly() const override;

private:
  /// CRC polynomial type.
  crc_generator_poly poly_;

  /// Fallback CPU calculator.
  std::unique_ptr<crc_calculator> fallback_;

  /// GPU availability flag.
  bool gpu_available_ = false;

  /// CUDA stream for async operations.
  cudaStream_t stream_ = nullptr;

  /// Device memory for input data.
  mutable uint8_t* d_data_ = nullptr;

  /// Device memory for CRC result.
  mutable uint32_t* d_crc_ = nullptr;

  /// Maximum allocated device buffer size.
  mutable size_t d_data_capacity_ = 0;

  /// \brief Compute CRC on GPU.
  /// \param[in] data  Input byte data.
  /// \return CRC checksum.
  crc_calculator_checksum_t compute_gpu(span<const uint8_t> data) const;

  /// \brief Ensure device buffer has sufficient capacity.
  /// \param[in] size  Required size in bytes.
  void ensure_capacity(size_t size) const;
};

} // namespace ocudu
