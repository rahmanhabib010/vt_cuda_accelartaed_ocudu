// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "modulation_mapper_cuda.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <ocudu_phy_cuda.h>
#include <modulation.h>
#include <vector>

using namespace ocudu;

namespace {

/// Minimum number of symbols to use GPU.
static constexpr unsigned MIN_SYMBOLS_FOR_GPU = 100;

/// Convert srsRAN modulation scheme to CUDA mod order.
static int get_mod_order(modulation_scheme scheme)
{
  switch (scheme) {
    case modulation_scheme::BPSK:
      return 1;
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
      return 0;
  }
}

/// Modulation mapper using CUDA GPU acceleration.
class modulation_mapper_cuda_impl : public modulation_mapper
{
public:
  modulation_mapper_cuda_impl(std::unique_ptr<modulation_mapper> fallback) : fallback_(std::move(fallback))
  {
    // Initialize CUDA if not already done
    if (ocudu_phy_cuda_init() != NR_LDPC_SUCCESS) {
      gpu_available_ = false;
      return;
    }

    // Create CUDA stream
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      gpu_available_ = false;
      return;
    }

    // Create modulator handle
    if (modulator_create(&modulator_) != 0) {
      cudaStreamDestroy(stream_);
      gpu_available_ = false;
      return;
    }

    // Allocate device memory (max codeword size: ~1M bits, ~125K symbols for QPSK)
    constexpr size_t max_bits    = 1024 * 1024;
    constexpr size_t max_words   = max_bits / 32;
    constexpr size_t max_symbols = max_bits / 2; // QPSK has most symbols per bit

    if (cudaMalloc(&d_bits_, max_words * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(&d_symbols_, max_symbols * sizeof(cuFloatComplex)) != cudaSuccess) {
      modulator_destroy(modulator_);
      cudaStreamDestroy(stream_);
      if (d_bits_)
        cudaFree(d_bits_);
      if (d_symbols_)
        cudaFree(d_symbols_);
      gpu_available_ = false;
      return;
    }

    h_bits_.resize(max_words);
    h_symbols_.resize(max_symbols);

    gpu_available_ = true;
  }

  ~modulation_mapper_cuda_impl()
  {
    if (gpu_available_) {
      modulator_destroy(modulator_);
      cudaFree(d_bits_);
      cudaFree(d_symbols_);
      cudaStreamDestroy(stream_);
    }
  }

  void modulate(span<cf_t> symbols, const bit_buffer& input, modulation_scheme scheme) override
  {
    int mod_order = get_mod_order(scheme);
    if (mod_order == 0 || !gpu_available_ || symbols.size() < MIN_SYMBOLS_FOR_GPU) {
      fallback_->modulate(symbols, input, scheme);
      return;
    }

    unsigned num_bits    = input.size();
    unsigned num_symbols = num_bits / mod_order;
    unsigned num_words   = (num_bits + 31) / 32;

    // Pack bits into words
    std::memset(h_bits_.data(), 0, num_words * sizeof(uint32_t));
    for (unsigned i = 0; i < num_bits; ++i) {
      if (input.extract(i, 1)) {
        h_bits_[i / 32] |= (1u << (i % 32));
      }
    }

    // Upload to GPU
    cudaMemcpyAsync(d_bits_, h_bits_.data(), num_words * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_);

    // Modulate on GPU
    if (modulator_modulate(
            modulator_, d_bits_, reinterpret_cast<cuFloatComplex*>(d_symbols_), num_bits, mod_order, stream_) != 0) {
      fallback_->modulate(symbols, input, scheme);
      return;
    }

    // Download result
    cudaMemcpyAsync(
        h_symbols_.data(), d_symbols_, num_symbols * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Copy to output
    for (unsigned i = 0; i < num_symbols; ++i) {
      symbols[i] = cf_t(h_symbols_[i].x, h_symbols_[i].y);
    }
  }

  float modulate(span<ci8_t> symbols, const bit_buffer& input, modulation_scheme scheme) override
  {
    int mod_order = get_mod_order(scheme);
    if (mod_order == 0 || !gpu_available_ || symbols.size() < MIN_SYMBOLS_FOR_GPU) {
      return fallback_->modulate(symbols, input, scheme);
    }

    unsigned num_bits    = input.size();
    unsigned num_symbols = num_bits / mod_order;
    unsigned num_words   = (num_bits + 31) / 32;

    // Pack bits into words
    std::memset(h_bits_.data(), 0, num_words * sizeof(uint32_t));
    for (unsigned i = 0; i < num_bits; ++i) {
      if (input.extract(i, 1)) {
        h_bits_[i / 32] |= (1u << (i % 32));
      }
    }

    // Upload to GPU
    cudaMemcpyAsync(d_bits_, h_bits_.data(), num_words * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_);

    // Modulate on GPU
    if (modulator_modulate(
            modulator_, d_bits_, reinterpret_cast<cuFloatComplex*>(d_symbols_), num_bits, mod_order, stream_) != 0) {
      return fallback_->modulate(symbols, input, scheme);
    }

    // Download result
    cudaMemcpyAsync(
        h_symbols_.data(), d_symbols_, num_symbols * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Get scaling factor and convert to ci8_t
    float scaling     = modulation_mapper::get_modulation_scaling(scheme);
    float inv_scaling = 1.0f / scaling;

    for (unsigned i = 0; i < num_symbols; ++i) {
      // GPU symbols are already normalized, scale for ci8_t output
      float re   = h_symbols_[i].x * inv_scaling * 127.0f;
      float im   = h_symbols_[i].y * inv_scaling * 127.0f;
      symbols[i] = ci8_t(static_cast<int8_t>(std::round(re)), static_cast<int8_t>(std::round(im)));
    }

    return scaling;
  }

private:
  std::unique_ptr<modulation_mapper> fallback_;
  bool                               gpu_available_ = false;

  // CUDA resources
  modulator_handle_t modulator_ = nullptr;
  cudaStream_t       stream_    = nullptr;

  // Device memory
  uint32_t*       d_bits_    = nullptr;
  cuFloatComplex* d_symbols_ = nullptr;

  // Host memory
  std::vector<uint32_t>       h_bits_;
  std::vector<cuFloatComplex> h_symbols_;
};

} // namespace

std::unique_ptr<modulation_mapper> ocudu::create_modulation_mapper_cuda(std::unique_ptr<modulation_mapper> fallback)
{
  return std::make_unique<modulation_mapper_cuda_impl>(std::move(fallback));
}
