// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief CUDA GPU-accelerated pseudo-random generator implementation.

#include "pseudo_random_generator_cuda.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/support/ocudu_assert.h"

#include <cuda_runtime.h>

// CUDA headers
extern "C" {
#include "scrambling.h"
}

using namespace ocudu;

/// Helper to check CUDA errors.
static bool check_cuda_available()
{
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess) && (device_count > 0);
}

pseudo_random_generator_cuda::pseudo_random_generator_cuda(
    std::unique_ptr<pseudo_random_generator> fallback_prng) :
  fallback(std::move(fallback_prng))
{
  ocudu_assert(fallback != nullptr, "Fallback PRNG cannot be null");

  // Check if GPU is available
  gpu_available = check_cuda_available();
  if (!gpu_available) {
    ocudulog::fetch_basic_logger("PHY").warning("CUDA: No CUDA GPU available, using CPU fallback for scrambling");
    return;
  }

  // Create CUDA scrambler
  nr_ldpc_status_t status = scrambler_create(&scrambler_handle);
  if (status != NR_LDPC_SUCCESS) {
    ocudulog::fetch_basic_logger("PHY").warning("CUDA: Failed to create scrambler, using CPU fallback");
    gpu_available = false;
    return;
  }

  ocudulog::fetch_basic_logger("PHY").info("CUDA: GPU-accelerated scrambler initialized");
}

pseudo_random_generator_cuda::~pseudo_random_generator_cuda()
{
  if (d_input_llrs) {
    cudaFree(d_input_llrs);
  }
  if (d_output_llrs) {
    cudaFree(d_output_llrs);
  }
  if (scrambler_handle) {
    scrambler_destroy(scrambler_handle);
  }
}

pseudo_random_generator_cuda::pseudo_random_generator_cuda(pseudo_random_generator_cuda&& other) noexcept :
  scrambler_handle(other.scrambler_handle),
  fallback(std::move(other.fallback)),
  current_c_init(other.current_c_init),
  current_offset(other.current_offset),
  gpu_available(other.gpu_available),
  d_input_llrs(other.d_input_llrs),
  d_output_llrs(other.d_output_llrs),
  allocated_size(other.allocated_size)
{
  other.scrambler_handle = nullptr;
  other.d_input_llrs     = nullptr;
  other.d_output_llrs    = nullptr;
  other.allocated_size   = 0;
  other.current_offset   = 0;
}

pseudo_random_generator_cuda&
pseudo_random_generator_cuda::operator=(pseudo_random_generator_cuda&& other) noexcept
{
  if (this != &other) {
    // Clean up existing resources
    if (d_input_llrs) {
      cudaFree(d_input_llrs);
    }
    if (d_output_llrs) {
      cudaFree(d_output_llrs);
    }
    if (scrambler_handle) {
      scrambler_destroy(scrambler_handle);
    }

    // Move from other
    scrambler_handle = other.scrambler_handle;
    fallback         = std::move(other.fallback);
    current_c_init   = other.current_c_init;
    current_offset   = other.current_offset;
    gpu_available    = other.gpu_available;
    d_input_llrs     = other.d_input_llrs;
    d_output_llrs    = other.d_output_llrs;
    allocated_size   = other.allocated_size;

    other.scrambler_handle = nullptr;
    other.d_input_llrs     = nullptr;
    other.d_output_llrs    = nullptr;
    other.allocated_size   = 0;
    other.current_offset   = 0;
  }
  return *this;
}

bool pseudo_random_generator_cuda::is_gpu_available() const
{
  return gpu_available;
}

void pseudo_random_generator_cuda::init(unsigned c_init)
{
  current_c_init  = c_init;
  current_offset = 0;  // Reset offset on init
  fallback->init(c_init);

  if (gpu_available && scrambler_handle) {
    // Use scrambler_configure_c_init() to set c_init directly.
    // This avoids issues with decomposing/recomposing n_RNTI/q/n_ID fields
    // which can lose bits due to masking in the CUDA scrambler_configure().
    scrambler_configure_c_init(scrambler_handle, c_init);
  }
}

void pseudo_random_generator_cuda::init(const state_s& state)
{
  // For state-based init, delegate to fallback
  fallback->init(state);
}

pseudo_random_generator::state_s pseudo_random_generator_cuda::get_state() const
{
  return fallback->get_state();
}

void pseudo_random_generator_cuda::advance(unsigned count)
{
  current_offset += count;
  fallback->advance(count);

  // Update CUDA scrambler offset
  if (gpu_available && scrambler_handle) {
    scrambler_advance(scrambler_handle, static_cast<int>(count));
  }
}

void pseudo_random_generator_cuda::apply_xor(bit_buffer& out, const bit_buffer& in)
{
  // Bit-level XOR uses CPU fallback
  fallback->apply_xor(out, in);
}

void pseudo_random_generator_cuda::apply_xor(span<uint8_t> out, span<const uint8_t> in)
{
  // Byte-level XOR uses CPU fallback
  fallback->apply_xor(out, in);
}

void pseudo_random_generator_cuda::ensure_device_buffer_size(unsigned size)
{
  if (size > allocated_size) {
    // Free existing buffers
    if (d_input_llrs) {
      cudaFree(d_input_llrs);
    }
    if (d_output_llrs) {
      cudaFree(d_output_llrs);
    }

    // Allocate new buffers with some headroom
    unsigned new_size = size + (size / 4); // 25% extra
    cudaMalloc(&d_input_llrs, new_size * sizeof(float));
    cudaMalloc(&d_output_llrs, new_size * sizeof(float));
    allocated_size = new_size;
  }
}

void pseudo_random_generator_cuda::apply_xor(span<log_likelihood_ratio> out, span<const log_likelihood_ratio> in)
{
  ocudu_assert(in.size() == out.size(), "Input and output spans must have the same size");

  unsigned num_llrs = in.size();

  // Use CPU fallback for small blocks or if GPU is not available
  if (!gpu_available || !scrambler_handle || num_llrs < MIN_GPU_LLRS) {
    fallback->apply_xor(out, in);
    // Track the offset change
    current_offset += num_llrs;
    return;
  }

  // Ensure device buffers are large enough
  ensure_device_buffer_size(num_llrs);

  // Generate scrambling sequence on GPU (uses current offset internally)
  nr_ldpc_status_t status = scrambler_generate_sequence(scrambler_handle, num_llrs, 0);
  if (status != NR_LDPC_SUCCESS) {
    // Fall back to CPU on error
    fallback->apply_xor(out, in);
    current_offset += num_llrs;
    return;
  }

  // Copy input LLRs to GPU
  // Note: log_likelihood_ratio is int8_t internally, we need to convert to float
  std::vector<float> float_llrs(num_llrs);
  for (unsigned i = 0; i < num_llrs; ++i) {
    float_llrs[i] = static_cast<float>(in[i].to_int());
  }
  cudaMemcpy(d_input_llrs, float_llrs.data(), num_llrs * sizeof(float), cudaMemcpyHostToDevice);

  // Descramble on GPU
  status = scrambler_descramble_llr(scrambler_handle, d_input_llrs, d_output_llrs, num_llrs, 0);
  if (status != NR_LDPC_SUCCESS) {
    // Fall back to CPU on error
    fallback->apply_xor(out, in);
    current_offset += num_llrs;
    return;
  }

  // Copy results back and convert to log_likelihood_ratio
  if (cudaMemcpy(float_llrs.data(), d_output_llrs, num_llrs * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
    fallback->apply_xor(out, in);
    current_offset += num_llrs;
    return;
  }

  for (unsigned i = 0; i < num_llrs; ++i) {
    // Clamp to int8_t range
    int val = static_cast<int>(std::round(float_llrs[i]));
    val     = std::max(-128, std::min(127, val));
    out[i]  = log_likelihood_ratio(static_cast<log_likelihood_ratio::value_type>(val));
  }

  // Update offset tracking
  current_offset += num_llrs;

  // Advance both the fallback and the CUDA scrambler to stay in sync
  fallback->advance(num_llrs);
  scrambler_advance(scrambler_handle, static_cast<int>(num_llrs));
}

void pseudo_random_generator_cuda::generate(bit_buffer& data)
{
  // Bit generation uses CPU fallback
  fallback->generate(data);
}

void pseudo_random_generator_cuda::generate(span<float> buffer, float value)
{
  // Float generation uses CPU fallback
  fallback->generate(buffer, value);
}

std::unique_ptr<pseudo_random_generator>
ocudu::create_pseudo_random_generator_cuda(std::unique_ptr<pseudo_random_generator> fallback)
{
  return std::make_unique<pseudo_random_generator_cuda>(std::move(fallback));
}
