// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "pusch_device_grid_reader_cuda.h"
#include "cuda_rt_utils.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

using namespace ocudu;

pusch_device_grid_reader_cuda::pusch_device_grid_reader_cuda(const resource_grid_reader& host_reader_,
                                                                   grid_allocation_mode        allocation_mode_) :
  host_reader(host_reader_), allocation_mode(allocation_mode_)
{
  cudaEvent_t event = nullptr;
  if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess) {
    grid_ready_event = event;
  }
  resize(host_reader.get_nof_ports(), host_reader.get_nof_symbols(), host_reader.get_nof_subc());
}

pusch_device_grid_reader_cuda::~pusch_device_grid_reader_cuda()
{
  if (grid_ready_event != nullptr) {
    cudaEventDestroy(static_cast<cudaEvent_t>(grid_ready_event));
    grid_ready_event = nullptr;
  }
  if (d_grid_bf16 != nullptr) {
    cudaFree(d_grid_bf16);
    d_grid_bf16 = nullptr;
  }
}

bool pusch_device_grid_reader_cuda::resize(unsigned nof_ports_, unsigned nof_symbols_, unsigned nof_subc_)
{
  size_t required_bytes = static_cast<size_t>(nof_ports_) * nof_symbols_ * nof_subc_ * sizeof(cbf16_t);
  if (required_bytes == 0) {
    return false;
  }

  if (required_bytes > device_grid_capacity) {
    if (d_grid_bf16 != nullptr) {
      cudaFree(d_grid_bf16);
      d_grid_bf16 = nullptr;
    }

    cudaError_t status = cudaSuccess;
    if (allocation_mode == grid_allocation_mode::managed) {
      status = cudaMallocManaged(&d_grid_bf16, required_bytes);
    } else {
      status = cudaMalloc(&d_grid_bf16, required_bytes);
    }
    if (status != cudaSuccess) {
      d_grid_bf16          = nullptr;
      device_grid_capacity = 0;
      return false;
    }
    device_grid_capacity = required_bytes;
  }

  device_grid_bytes       = required_bytes;
  device_grid_nof_ports   = nof_ports_;
  device_grid_nof_symbols = nof_symbols_;
  device_grid_nof_subc    = nof_subc_;
  grid_ready_recorded     = false;
  return true;
}

bool pusch_device_grid_reader_cuda::stage_host_grid_async(void* stream)
{
  if (!supports_device_grid_reading() ||
      !resize(host_reader.get_nof_ports(), host_reader.get_nof_symbols(), host_reader.get_nof_subc())) {
    return false;
  }

  cudaStream_t cuda_stream  = static_cast<cudaStream_t>(stream);
  size_t       grid_stride  = static_cast<size_t>(device_grid_nof_symbols) * device_grid_nof_subc;
  char*        device_bytes = static_cast<char*>(d_grid_bf16);

  if (allocation_mode == grid_allocation_mode::managed) {
    for (unsigned port = 0; port != device_grid_nof_ports; ++port) {
      for (unsigned symbol = 0; symbol != device_grid_nof_symbols; ++symbol) {
        span<const cbf16_t> src = host_reader.get_view(port, symbol);
        auto*    dst    = static_cast<cbf16_t*>(d_grid_bf16) + port * grid_stride + symbol * device_grid_nof_subc;
        unsigned nof_re = std::min<unsigned>(device_grid_nof_subc, src.size());
        std::memcpy(dst, src.data(), nof_re * sizeof(cbf16_t));
      }
    }

  } else {
    for (unsigned port = 0; port != device_grid_nof_ports; ++port) {
      span<const cbf16_t> first_view    = host_reader.get_view(port, 0);
      bool                is_contiguous = first_view.size() >= device_grid_nof_subc;
      for (unsigned symbol = 1; symbol != device_grid_nof_symbols && is_contiguous; ++symbol) {
        span<const cbf16_t> view = host_reader.get_view(port, symbol);
        is_contiguous            = (view.data() == first_view.data() + symbol * device_grid_nof_subc);
      }

      if (is_contiguous) {
        cudaMemcpyAsync(device_bytes + port * grid_stride * sizeof(cbf16_t),
                        first_view.data(),
                        grid_stride * sizeof(cbf16_t),
                        cudaMemcpyHostToDevice,
                        cuda_stream);
      } else {
        for (unsigned symbol = 0; symbol != device_grid_nof_symbols; ++symbol) {
          span<const cbf16_t> src    = host_reader.get_view(port, symbol);
          size_t              offset = (port * grid_stride + symbol * device_grid_nof_subc) * sizeof(cbf16_t);
          cudaMemcpyAsync(device_bytes + offset,
                          src.data(),
                          std::min<unsigned>(device_grid_nof_subc, src.size()) * sizeof(cbf16_t),
                          cudaMemcpyHostToDevice,
                          cuda_stream);
        }
      }
    }
  }

  if (cudaGetLastError() != cudaSuccess) {
    grid_ready_recorded = false;
    return false;
  }

  if (grid_ready_event == nullptr ||
      cudaEventRecord(static_cast<cudaEvent_t>(grid_ready_event), cuda_stream) != cudaSuccess) {
    grid_ready_recorded = false;
    return false;
  }

  grid_ready_recorded = true;
  return true;
}

bool pusch_device_grid_reader_cuda::synchronize_device_grid_ready() const
{
  if (!grid_ready_recorded || (grid_ready_event == nullptr)) {
    return true;
  }
  return cudaEventSynchronizeYielding(static_cast<cudaEvent_t>(grid_ready_event)) == cudaSuccess;
}

bool pusch_device_grid_reader_cuda::supports_device_grid_reading() const
{
  return (d_grid_bf16 != nullptr) && (grid_ready_event != nullptr);
}

bool pusch_device_grid_reader_cuda::prepare_device_grid_reading(void* stream) const
{
  if (!supports_device_grid_reading()) {
    return false;
  }
  if (!grid_ready_recorded || (grid_ready_event == nullptr)) {
    return true;
  }
  return cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(grid_ready_event), 0) ==
         cudaSuccess;
}
