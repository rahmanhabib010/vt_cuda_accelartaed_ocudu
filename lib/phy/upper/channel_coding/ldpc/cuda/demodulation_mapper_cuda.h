// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief CUDA CUDA-accelerated soft demodulation mapper.

#pragma once

#include "ocudu/phy/upper/channel_modulation/demodulation_mapper.h"
#include <cuda_runtime.h>
#include <memory>

// Forward declarations for CUDA types
struct modulator_ctx;
typedef struct modulator_ctx* modulator_handle_t;

namespace ocudu {

/// \brief CUDA CUDA-accelerated demodulation mapper.
///
/// This implementation uses the CUDA library for GPU-accelerated soft
/// demodulation. For small symbol counts where GPU overhead exceeds the benefit,
/// it falls back to a CPU implementation.
///
/// Key features:
/// - Per-symbol noise variance support for accurate LLR computation after MMSE equalization
/// - Automatic CPU fallback for small data or when GPU is unavailable
/// - LLR quantization to int8_t with configurable range limit
class demodulation_mapper_cuda : public demodulation_mapper
{
public:
  /// Minimum number of symbols to use GPU (below this, CPU is faster due to overhead).
  static constexpr unsigned MIN_GPU_SYMBOLS = 128;

  /// Range limit for LLR quantization (matches srsRAN CPU implementations).
  static constexpr float LLR_RANGE_LIMIT = 20.0f;

  /// \brief Constructor.
  /// \param[in] fallback_mapper CPU-based fallback demodulation mapper.
  explicit demodulation_mapper_cuda(std::unique_ptr<demodulation_mapper> fallback_mapper);

  /// Destructor - releases CUDA resources.
  ~demodulation_mapper_cuda() override;

  // Deleted copy/move operations due to CUDA resource management.
  demodulation_mapper_cuda(const demodulation_mapper_cuda&)            = delete;
  demodulation_mapper_cuda& operator=(const demodulation_mapper_cuda&) = delete;
  demodulation_mapper_cuda(demodulation_mapper_cuda&&)                 = delete;
  demodulation_mapper_cuda& operator=(demodulation_mapper_cuda&&)      = delete;

  /// See interface for documentation.
  void demodulate_soft(span<log_likelihood_ratio> llrs,
                       span<const cf_t>           symbols,
                       span<const float>          noise_vars,
                       modulation_scheme          mod) override;

  /// \brief GPU-resident soft demodulation (keeps LLRs on GPU).
  ///
  /// Performs soft demodulation on GPU but does NOT copy LLRs back to host.
  /// Use get_device_llrs() to obtain the device pointer for downstream GPU operations.
  /// This avoids GPU→CPU→GPU round-trips when the decoder is also on GPU.
  ///
  /// \param[in] symbols    Complex symbols to demodulate (host memory).
  /// \param[in] noise_vars Noise variances after equalization (host memory).
  /// \param[in] mod        Modulation scheme.
  /// \return true if demodulation was performed on GPU, false if fell back to CPU.
  /// \note If this returns false, use demodulate_soft() for the CPU path instead.
  bool demodulate_soft_gpu_resident(span<const cf_t>  symbols,
                                    span<const float> noise_vars,
                                    modulation_scheme mod);

  /// \brief Get device pointer to GPU-resident float LLRs.
  ///
  /// Returns pointer to float LLRs on GPU from the last demodulate_soft_gpu_resident() call.
  /// The LLRs are in float format (not quantized to int8).
  ///
  /// \return Device pointer to float LLRs, or nullptr if last operation used CPU.
  float* get_device_llrs() const { return gpu_llrs_valid_ ? static_cast<float*>(d_llrs_float_) : nullptr; }

  /// \brief Get device pointer to GPU-resident fp16 LLRs.
  ///
  /// Returns pointer to __half LLRs on GPU from the last demodulate_soft_gpu_resident_half() call.
  ///
  /// \return Device pointer to fp16 LLRs, or nullptr if last operation used CPU.
  void* get_device_llrs_half() const { return gpu_llrs_half_valid_ ? d_llrs_half_ : nullptr; }

  /// \brief Get number of LLRs from last demodulation.
  size_t get_last_num_llrs() const { return last_num_llrs_; }

  /// \brief GPU-resident fp16 soft demodulation (keeps LLRs on GPU as half-precision).
  ///
  /// Performs soft demodulation on GPU and outputs fp16 (__half) LLRs directly.
  /// This avoids fp32→fp16 conversion overhead in the downstream LDPC decoder.
  /// Use get_device_llrs_half() to obtain the device pointer for downstream GPU operations.
  ///
  /// \param[in] symbols    Complex symbols to demodulate (host memory).
  /// \param[in] noise_vars Noise variances after equalization (host memory).
  /// \param[in] mod        Modulation scheme.
  /// \return true if demodulation was performed on GPU, false if fell back to CPU.
  bool demodulate_soft_gpu_resident_half(span<const cf_t>  symbols,
                                          span<const float> noise_vars,
                                          modulation_scheme mod);

  /// \brief Check if GPU demodulation is available.
  bool is_gpu_available() const { return gpu_available_; }

  /// \brief Get CUDA stream used by this demodulator.
  cudaStream_t get_cuda_stream() const { return stream_; }

private:
  /// Fallback CPU demodulation mapper.
  std::unique_ptr<demodulation_mapper> fallback_;

  /// CUDA modulator handle for GPU operations.
  modulator_handle_t handle_ = nullptr;

  /// CUDA stream for asynchronous operations.
  cudaStream_t stream_ = nullptr;

  /// Whether GPU is available and initialized.
  bool gpu_available_ = false;

  /// Device memory for symbols.
  mutable void* d_symbols_ = nullptr;
  /// Device memory for noise variances.
  mutable void* d_noise_vars_ = nullptr;
  /// Device memory for float LLRs (before quantization).
  mutable void* d_llrs_float_ = nullptr;
  /// Device memory for fp16 LLRs.
  mutable void* d_llrs_half_ = nullptr;
  /// Host memory for float LLRs (for quantization).
  mutable std::vector<float> h_llrs_float_;
  /// Current capacity of device buffers.
  mutable size_t d_symbols_capacity_    = 0;
  mutable size_t d_noise_vars_capacity_ = 0;
  mutable size_t d_llrs_capacity_       = 0;
  mutable size_t d_llrs_half_capacity_  = 0;

  /// Track whether GPU-resident LLRs are valid (from demodulate_soft_gpu_resident).
  mutable bool gpu_llrs_valid_ = false;
  /// Track whether GPU-resident fp16 LLRs are valid.
  mutable bool gpu_llrs_half_valid_ = false;
  /// Number of LLRs from last demodulation.
  mutable size_t last_num_llrs_ = 0;

  /// \brief Ensure device buffers have sufficient capacity.
  void ensure_capacity(size_t num_symbols, size_t num_llrs) const;

  /// \brief Perform GPU demodulation.
  void demodulate_gpu(span<log_likelihood_ratio> llrs,
                      span<const cf_t>           symbols,
                      span<const float>          noise_vars,
                      modulation_scheme          mod) const;

  /// \brief Convert modulation_scheme to CUDA mod_order.
  static int get_mod_order(modulation_scheme mod);
};

} // namespace ocudu
