// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "crc_calculator_cuda.h"
#include "cuda_rt_utils.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/support/ocudu_assert.h"

// CUDA headers
extern "C" {
#include "transport_block.h"
}

using namespace ocudu;

/// Helper to check CUDA availability.
static bool check_cuda_available()
{
  int         device_count = 0;
  cudaError_t err          = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess) && (device_count > 0);
}

crc_calculator_cuda::crc_calculator_cuda(crc_generator_poly poly, std::unique_ptr<crc_calculator> fallback_calc) :
  poly_(poly), fallback_(std::move(fallback_calc))
{
  ocudu_assert(fallback_ != nullptr, "Fallback CRC calculator cannot be null");
  ocudu_assert(fallback_->get_generator_poly() == poly_, "Fallback CRC calculator polynomial mismatch");

  // Check if CUDA is available.
  if (!check_cuda_available()) {
    ocudulog::fetch_basic_logger("PHY").info("CUDA CRC: No CUDA GPU available, using CPU fallback");
    return;
  }

  // Create CUDA stream.
  cudaError_t cuda_status = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  if (cuda_status != cudaSuccess) {
    ocudulog::fetch_basic_logger("PHY").warning("CUDA CRC: Failed to create CUDA stream, using CPU fallback");
    return;
  }

  // Allocate device memory for CRC result.
  cuda_status = cudaMalloc(&d_crc_, sizeof(uint32_t));
  if (cuda_status != cudaSuccess) {
    ocudulog::fetch_basic_logger("PHY").warning("CUDA CRC: Failed to allocate CRC buffer, using CPU fallback");
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
    return;
  }

  gpu_available_ = true;
  ocudulog::fetch_basic_logger("PHY").info("CUDA: GPU-accelerated CRC calculator initialized ({})",
                                           (poly_ == crc_generator_poly::CRC24A)   ? "CRC24A"
                                           : (poly_ == crc_generator_poly::CRC24B) ? "CRC24B"
                                                                                   : "CRC16");
}

crc_calculator_cuda::~crc_calculator_cuda()
{
  if (d_data_) {
    cudaFree(d_data_);
  }
  if (d_crc_) {
    cudaFree(d_crc_);
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

void crc_calculator_cuda::ensure_capacity(size_t size) const
{
  if (size <= d_data_capacity_) {
    return;
  }

  // Free existing buffer if any.
  if (d_data_) {
    cudaFree(d_data_);
    d_data_ = nullptr;
  }

  // Allocate new buffer with some extra capacity.
  size_t new_capacity = std::max(size, d_data_capacity_ * 2);
  new_capacity        = std::max(new_capacity, static_cast<size_t>(4096)); // Minimum 4KB

  cudaError_t status = cudaMalloc(&d_data_, new_capacity);
  if (status != cudaSuccess) {
    ocudulog::fetch_basic_logger("PHY").error("CUDA CRC: Failed to allocate {} bytes", new_capacity);
    d_data_capacity_ = 0;
    return;
  }

  d_data_capacity_ = new_capacity;
}

crc_calculator_checksum_t crc_calculator_cuda::compute_gpu(span<const uint8_t> data) const
{
  // Ensure device buffer has enough capacity.
  ensure_capacity(data.size());

  if (!d_data_ || d_data_capacity_ < data.size()) {
    // Allocation failed, fall back to CPU.
    return fallback_->calculate_byte(data);
  }

  // Copy data to device.
  cudaError_t status = cudaMemcpyAsync(d_data_, data.data(), data.size(), cudaMemcpyHostToDevice, stream_);
  if (status != cudaSuccess) {
    return fallback_->calculate_byte(data);
  }

  // Compute CRC on GPU.
  int              num_bits = static_cast<int>(data.size() * 8);
  nr_ldpc_status_t crc_status;

  switch (poly_) {
    case crc_generator_poly::CRC24A:
      crc_status = crc24a_compute(d_data_, num_bits, d_crc_, stream_);
      break;
    case crc_generator_poly::CRC24B:
      crc_status = crc24b_compute(d_data_, num_bits, d_crc_, stream_);
      break;
    case crc_generator_poly::CRC16:
      crc_status = crc16_compute(d_data_, num_bits, reinterpret_cast<uint16_t*>(d_crc_), stream_);
      break;
    default:
      // Unsupported polynomial, use fallback.
      return fallback_->calculate_byte(data);
  }

  if (crc_status != NR_LDPC_SUCCESS) {
    return fallback_->calculate_byte(data);
  }

  // Copy result back to host.
  uint32_t crc_result = 0;
  status              = cudaMemcpyAsync(&crc_result, d_crc_, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_);
  if (status != cudaSuccess) {
    return fallback_->calculate_byte(data);
  }

  // Synchronize to get result.
  cudaStreamSynchronizeYielding(stream_);

  // Mask to appropriate size.
  if (poly_ == crc_generator_poly::CRC16) {
    return static_cast<crc_calculator_checksum_t>(crc_result & 0xFFFF);
  }
  return static_cast<crc_calculator_checksum_t>(crc_result & 0xFFFFFF);
}

crc_calculator_checksum_t crc_calculator_cuda::calculate_byte(span<const uint8_t> data) const
{
  // Use GPU only for larger data sizes where it's beneficial.
  if (!gpu_available_ || data.size() < MIN_GPU_BYTES) {
    return fallback_->calculate_byte(data);
  }

  return compute_gpu(data);
}

crc_calculator_checksum_t crc_calculator_cuda::calculate_bit(span<const uint8_t> data) const
{
  // Bit-wise calculation is less common and typically small.
  // Use fallback for simplicity.
  return fallback_->calculate_bit(data);
}

crc_calculator_checksum_t crc_calculator_cuda::calculate(const bit_buffer& data) const
{
  // For bit_buffer, we need to handle non-byte-aligned data.
  // If byte-aligned and large enough, use GPU.
  if (gpu_available_ && (data.size() % 8 == 0) && (data.size() / 8 >= MIN_GPU_BYTES)) {
    // Extract bytes from bit_buffer.
    unsigned             num_bytes = data.size() / 8;
    std::vector<uint8_t> bytes(num_bytes);
    for (unsigned i = 0; i < num_bytes; ++i) {
      bytes[i] = data.get_byte(i);
    }
    return compute_gpu(bytes);
  }

  return fallback_->calculate(data);
}

crc_generator_poly crc_calculator_cuda::get_generator_poly() const
{
  return poly_;
}
