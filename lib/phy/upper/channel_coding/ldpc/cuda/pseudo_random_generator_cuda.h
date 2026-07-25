// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief CUDA GPU-accelerated pseudo-random generator declaration.

#pragma once

#include "ocudu/phy/upper/sequence_generators/pseudo_random_generator.h"
#include <cstdint>
#include <memory>

// Forward declaration
struct scrambler_ctx;
typedef struct scrambler_ctx* scrambler_handle_t;

namespace ocudu {

/// \brief CUDA GPU-accelerated implementation of the pseudo-random generator.
///
/// This class provides GPU-accelerated scrambling/descrambling operations using
/// the CUDA library. It implements the same interface as the CPU-based
/// pseudo_random_generator_impl but offloads LLR descrambling to the GPU.
///
/// \note For small data sizes, the overhead of GPU memory transfers may outweigh
/// the computational benefits. A minimum threshold is used to decide whether to
/// use GPU or fall back to CPU processing.
class pseudo_random_generator_cuda : public pseudo_random_generator
{
public:
  /// Minimum number of LLRs to use GPU acceleration (below this, CPU is faster).
  static constexpr unsigned MIN_GPU_LLRS = 256;

  /// \brief Constructor.
  /// \param[in] fallback_prng Fallback CPU-based generator for small blocks and non-LLR operations.
  explicit pseudo_random_generator_cuda(std::unique_ptr<pseudo_random_generator> fallback_prng);

  /// Destructor.
  ~pseudo_random_generator_cuda() override;

  // Disable copy
  pseudo_random_generator_cuda(const pseudo_random_generator_cuda&)            = delete;
  pseudo_random_generator_cuda& operator=(const pseudo_random_generator_cuda&) = delete;

  // Enable move
  pseudo_random_generator_cuda(pseudo_random_generator_cuda&&) noexcept;
  pseudo_random_generator_cuda& operator=(pseudo_random_generator_cuda&&) noexcept;

  /// \brief Check if the GPU is available and CUDA is properly initialized.
  /// \return true if GPU acceleration is available.
  bool is_gpu_available() const;

  // See interface for the documentation.
  void init(unsigned c_init) override;

  // See interface for the documentation.
  void init(const state_s& state) override;

  // See interface for the documentation.
  state_s get_state() const override;

  // See interface for the documentation.
  void advance(unsigned count) override;

  // See interface for the documentation.
  void apply_xor(bit_buffer& out, const bit_buffer& in) override;

  // See interface for the documentation.
  void apply_xor(span<uint8_t> out, span<const uint8_t> in) override;

  /// \brief GPU-accelerated LLR descrambling.
  ///
  /// For blocks larger than MIN_GPU_LLRS, this uses the GPU to descramble LLRs.
  /// For smaller blocks, it falls back to the CPU implementation.
  void apply_xor(span<log_likelihood_ratio> out, span<const log_likelihood_ratio> in) override;

  // See interface for the documentation.
  void generate(bit_buffer& data) override;

  // See interface for the documentation.
  void generate(span<float> buffer, float value) override;

private:
  /// CUDA scrambler handle.
  scrambler_handle_t scrambler_handle = nullptr;
  /// Fallback CPU-based generator.
  std::unique_ptr<pseudo_random_generator> fallback;
  /// Current c_init value.
  unsigned current_c_init = 0;
  /// Current bit offset (cumulative from advance() calls).
  unsigned current_offset = 0;
  /// Flag indicating if GPU is available.
  bool gpu_available = false;
  /// Device memory for input LLRs.
  float* d_input_llrs = nullptr;
  /// Device memory for output LLRs.
  float* d_output_llrs = nullptr;
  /// Current allocated size for device buffers.
  unsigned allocated_size = 0;

  /// Ensure device buffers are large enough.
  void ensure_device_buffer_size(unsigned size);
};

/// \brief Create a CUDA GPU-accelerated pseudo-random generator.
/// \param[in] fallback Fallback CPU-based generator.
/// \return Unique pointer to the GPU-accelerated generator.
std::unique_ptr<pseudo_random_generator>
create_pseudo_random_generator_cuda(std::unique_ptr<pseudo_random_generator> fallback);

} // namespace ocudu
