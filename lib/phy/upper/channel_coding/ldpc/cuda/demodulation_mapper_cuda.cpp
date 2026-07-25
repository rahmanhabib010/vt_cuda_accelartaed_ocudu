// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "demodulation_mapper_cuda.h"
#include "cuda_rt_utils.h"
#include "ocudu/phy/upper/log_likelihood_ratio.h"
#include "ocudu/support/ocudu_assert.h"
#include <cuComplex.h>
#include <modulation.h>

using namespace ocudu;

demodulation_mapper_cuda::demodulation_mapper_cuda(std::unique_ptr<demodulation_mapper> fallback_mapper) :
  fallback_(std::move(fallback_mapper))
{
  ocudu_assert(fallback_, "Fallback demodulation mapper cannot be null.");

  // Check if CUDA is available.
  int         device_count = 0;
  cudaError_t err          = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    gpu_available_ = false;
    return;
  }

  // Create CUDA modulator handle.
  if (modulator_create(&handle_) != 0) {
    gpu_available_ = false;
    return;
  }

  // Create CUDA stream for async operations.
  err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    modulator_destroy(handle_);
    handle_        = nullptr;
    gpu_available_ = false;
    return;
  }

  gpu_available_ = true;
}

demodulation_mapper_cuda::~demodulation_mapper_cuda()
{
  // Free device memory.
  if (d_symbols_) {
    cudaFree(d_symbols_);
  }
  if (d_noise_vars_) {
    cudaFree(d_noise_vars_);
  }
  if (d_llrs_float_) {
    cudaFree(d_llrs_float_);
  }
  if (d_llrs_half_) {
    cudaFree(d_llrs_half_);
  }

  // Destroy CUDA stream.
  if (stream_) {
    cudaStreamDestroy(stream_);
  }

  // Destroy CUDA handle.
  if (handle_) {
    modulator_destroy(handle_);
  }
}

int demodulation_mapper_cuda::get_mod_order(modulation_scheme mod)
{
  switch (mod) {
    case modulation_scheme::BPSK:
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
      return -1;
  }
}

void demodulation_mapper_cuda::ensure_capacity(size_t num_symbols, size_t num_llrs) const
{
  cudaError_t err;

  // Allocate/reallocate symbols buffer if needed.
  size_t symbols_bytes = num_symbols * sizeof(cuFloatComplex);
  if (symbols_bytes > d_symbols_capacity_) {
    if (d_symbols_) {
      cudaFree(d_symbols_);
    }
    err = cudaMalloc(&d_symbols_, symbols_bytes);
    if (err != cudaSuccess) {
      d_symbols_          = nullptr;
      d_symbols_capacity_ = 0;
      return;
    }
    d_symbols_capacity_ = symbols_bytes;
  }

  // Allocate/reallocate noise variance buffer if needed.
  size_t noise_bytes = num_symbols * sizeof(float);
  if (noise_bytes > d_noise_vars_capacity_) {
    if (d_noise_vars_) {
      cudaFree(d_noise_vars_);
    }
    err = cudaMalloc(&d_noise_vars_, noise_bytes);
    if (err != cudaSuccess) {
      d_noise_vars_          = nullptr;
      d_noise_vars_capacity_ = 0;
      return;
    }
    d_noise_vars_capacity_ = noise_bytes;
  }

  // Allocate/reallocate LLR buffer if needed.
  size_t llrs_bytes = num_llrs * sizeof(float);
  if (llrs_bytes > d_llrs_capacity_) {
    if (d_llrs_float_) {
      cudaFree(d_llrs_float_);
    }
    err = cudaMalloc(&d_llrs_float_, llrs_bytes);
    if (err != cudaSuccess) {
      d_llrs_float_    = nullptr;
      d_llrs_capacity_ = 0;
      return;
    }
    d_llrs_capacity_ = llrs_bytes;
  }

  // Allocate/reallocate fp16 LLR buffer if needed (2 bytes per LLR).
  size_t llrs_half_bytes = num_llrs * sizeof(uint16_t);
  if (llrs_half_bytes > d_llrs_half_capacity_) {
    if (d_llrs_half_) {
      cudaFree(d_llrs_half_);
    }
    err = cudaMalloc(&d_llrs_half_, llrs_half_bytes);
    if (err != cudaSuccess) {
      d_llrs_half_          = nullptr;
      d_llrs_half_capacity_ = 0;
      return;
    }
    d_llrs_half_capacity_ = llrs_half_bytes;
  }

  // Resize host LLR buffer.
  if (h_llrs_float_.size() < num_llrs) {
    h_llrs_float_.resize(num_llrs);
  }
}

void demodulation_mapper_cuda::demodulate_gpu(span<log_likelihood_ratio> llrs,
                                                 span<const cf_t>           symbols,
                                                 span<const float>          noise_vars,
                                                 modulation_scheme          mod) const
{
  size_t num_symbols = symbols.size();
  size_t num_llrs    = llrs.size();
  int    mod_order   = get_mod_order(mod);

  // Ensure device buffers are large enough.
  ensure_capacity(num_symbols, num_llrs);

  if (!d_symbols_ || !d_noise_vars_ || !d_llrs_float_) {
    // Memory allocation failed, fall back to CPU.
    fallback_->demodulate_soft(llrs, symbols, noise_vars, mod);
    return;
  }

  // Copy symbols to device (cf_t is std::complex<float> which is layout-compatible with cuFloatComplex).
  cudaMemcpyAsync(d_symbols_, symbols.data(), num_symbols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice, stream_);

  // Copy noise variances to device.
  cudaMemcpyAsync(d_noise_vars_, noise_vars.data(), num_symbols * sizeof(float), cudaMemcpyHostToDevice, stream_);

  // Perform GPU soft demodulation with per-symbol noise variance.
  int result = modulator_soft_demod_per_symbol(handle_,
                                               static_cast<cuFloatComplex*>(d_symbols_),
                                               static_cast<float*>(d_noise_vars_),
                                               static_cast<float*>(d_llrs_float_),
                                               static_cast<int>(num_symbols),
                                               mod_order,
                                               stream_);

  if (result != 0) {
    // GPU operation failed, fall back to CPU.
    cudaStreamSynchronizeYielding(stream_);
    fallback_->demodulate_soft(llrs, symbols, noise_vars, mod);
    return;
  }

  // Copy float LLRs back to host.
  cudaMemcpyAsync(h_llrs_float_.data(), d_llrs_float_, num_llrs * sizeof(float), cudaMemcpyDeviceToHost, stream_);

  // Wait for all operations to complete.
  cudaStreamSynchronizeYielding(stream_);

  // Quantize float LLRs to int8_t log_likelihood_ratio.
  for (size_t i = 0; i < num_llrs; ++i) {
    llrs[i] = log_likelihood_ratio::quantize(h_llrs_float_[i], LLR_RANGE_LIMIT);
  }
}

void demodulation_mapper_cuda::demodulate_soft(span<log_likelihood_ratio> llrs,
                                                  span<const cf_t>           symbols,
                                                  span<const float>          noise_vars,
                                                  modulation_scheme          mod)
{
  ocudu_assert(symbols.size() == noise_vars.size(), "Symbols and noise_vars must have the same length.");
  ocudu_assert(symbols.size() * get_bits_per_symbol(mod) == llrs.size(), "Input and output lengths are incompatible.");

  // Use CPU for unsupported modulations or small symbol counts.
  int mod_order = get_mod_order(mod);
  if (!gpu_available_ || mod_order < 0 || symbols.size() < MIN_GPU_SYMBOLS) {
    fallback_->demodulate_soft(llrs, symbols, noise_vars, mod);
    return;
  }

  // Use GPU for larger symbol counts.
  demodulate_gpu(llrs, symbols, noise_vars, mod);

  // Mark GPU LLRs as invalid since we copied to host.
  gpu_llrs_valid_ = false;
}

bool demodulation_mapper_cuda::demodulate_soft_gpu_resident(span<const cf_t>  symbols,
                                                               span<const float> noise_vars,
                                                               modulation_scheme mod)
{
  ocudu_assert(symbols.size() == noise_vars.size(), "Symbols and noise_vars must have the same length.");

  // Mark GPU LLRs as invalid initially.
  gpu_llrs_valid_ = false;
  last_num_llrs_  = 0;

  // Check if we should use GPU.
  int mod_order = get_mod_order(mod);
  if (!gpu_available_ || mod_order < 0 || symbols.size() < MIN_GPU_SYMBOLS) {
    // Cannot use GPU - caller should fall back to CPU path.
    return false;
  }

  size_t num_symbols = symbols.size();
  size_t num_llrs    = num_symbols * static_cast<size_t>(mod_order);

  // Ensure device buffers are large enough.
  ensure_capacity(num_symbols, num_llrs);

  if (!d_symbols_ || !d_noise_vars_ || !d_llrs_float_) {
    // Memory allocation failed.
    return false;
  }

  // Copy symbols to device.
  cudaMemcpyAsync(d_symbols_, symbols.data(), num_symbols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice, stream_);

  // Copy noise variances to device.
  cudaMemcpyAsync(d_noise_vars_, noise_vars.data(), num_symbols * sizeof(float), cudaMemcpyHostToDevice, stream_);

  // Perform GPU soft demodulation with per-symbol noise variance.
  int result = modulator_soft_demod_per_symbol(handle_,
                                               static_cast<cuFloatComplex*>(d_symbols_),
                                               static_cast<float*>(d_noise_vars_),
                                               static_cast<float*>(d_llrs_float_),
                                               static_cast<int>(num_symbols),
                                               mod_order,
                                               stream_);

  if (result != 0) {
    // GPU operation failed.
    cudaStreamSynchronizeYielding(stream_);
    return false;
  }

  // DO NOT copy LLRs back to host - keep them on GPU for downstream operations.
  // DO NOT synchronize - let downstream operations chain on the same stream.

  // Mark GPU LLRs as valid.
  gpu_llrs_valid_ = true;
  last_num_llrs_  = num_llrs;

  return true;
}

bool demodulation_mapper_cuda::demodulate_soft_gpu_resident_half(span<const cf_t>  symbols,
                                                                    span<const float> noise_vars,
                                                                    modulation_scheme mod)
{
  ocudu_assert(symbols.size() == noise_vars.size(), "Symbols and noise_vars must have the same length.");

  // Mark GPU LLRs as invalid initially.
  gpu_llrs_valid_      = false;
  gpu_llrs_half_valid_ = false;
  last_num_llrs_       = 0;

  // Check if we should use GPU.
  int mod_order = get_mod_order(mod);
  if (!gpu_available_ || mod_order < 0 || symbols.size() < MIN_GPU_SYMBOLS) {
    // Cannot use GPU - caller should fall back to CPU path.
    return false;
  }

  size_t num_symbols = symbols.size();
  size_t num_llrs    = num_symbols * static_cast<size_t>(mod_order);

  // Ensure device buffers are large enough.
  ensure_capacity(num_symbols, num_llrs);

  if (!d_symbols_ || !d_noise_vars_ || !d_llrs_half_) {
    // Memory allocation failed.
    return false;
  }

  // Copy symbols to device.
  cudaMemcpyAsync(d_symbols_, symbols.data(), num_symbols * sizeof(cuFloatComplex), cudaMemcpyHostToDevice, stream_);

  // Copy noise variances to device.
  cudaMemcpyAsync(d_noise_vars_, noise_vars.data(), num_symbols * sizeof(float), cudaMemcpyHostToDevice, stream_);

  // Perform GPU soft demodulation with per-symbol noise variance, outputting fp16 LLRs.
  int result = modulator_soft_demod_per_symbol_half(handle_,
                                                    static_cast<cuFloatComplex*>(d_symbols_),
                                                    static_cast<float*>(d_noise_vars_),
                                                    d_llrs_half_,
                                                    static_cast<int>(num_symbols),
                                                    mod_order,
                                                    stream_);

  if (result != 0) {
    // GPU operation failed.
    cudaStreamSynchronizeYielding(stream_);
    return false;
  }

  // DO NOT copy LLRs back to host - keep them on GPU for downstream operations.
  // DO NOT synchronize - let downstream operations chain on the same stream.

  // Mark GPU fp16 LLRs as valid.
  gpu_llrs_half_valid_ = true;
  last_num_llrs_       = num_llrs;

  return true;
}
