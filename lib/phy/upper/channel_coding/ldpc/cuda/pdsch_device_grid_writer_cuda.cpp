// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "pdsch_device_grid_writer_cuda.h"
#include "cuda_rt_utils.h"
#include <cstring>
#include <cuda_runtime.h>

using namespace ocudu;

pdsch_device_grid_writer_cuda::pdsch_device_grid_writer_cuda(resource_grid_writer& host_writer_,
                                                                   grid_allocation_mode  allocation_mode_) :
  host_writer(host_writer_), allocation_mode(allocation_mode_)
{
  cudaEvent_t event = nullptr;
  if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess) {
    clear_event = event;
  }
  resize(host_writer.get_nof_ports(), host_writer.get_nof_symbols(), host_writer.get_nof_subc());
}

pdsch_device_grid_writer_cuda::~pdsch_device_grid_writer_cuda()
{
  if (clear_event != nullptr) {
    cudaEventDestroy(static_cast<cudaEvent_t>(clear_event));
    clear_event = nullptr;
  }
  for (void* event : grid_ready_events) {
    cudaEventDestroy(static_cast<cudaEvent_t>(event));
  }
  grid_ready_event = nullptr;
  grid_ready_events.clear();
  pending_grid_ready_events.clear();
  if (d_grid_bf16 != nullptr) {
    cudaFree(d_grid_bf16);
    d_grid_bf16 = nullptr;
  }
}

bool pdsch_device_grid_writer_cuda::resize(unsigned nof_ports_, unsigned nof_symbols_, unsigned nof_subc_)
{
  size_t required_bytes = static_cast<size_t>(nof_ports_) * nof_symbols_ * nof_subc_ * 2U * sizeof(uint16_t);
  if (required_bytes > device_grid_capacity) {
    if (d_grid_bf16 != nullptr) {
      (void)synchronize_device_grid_ready();
      cudaFree(d_grid_bf16);
      d_grid_bf16 = nullptr;
    }
    cudaError_t alloc_status = (allocation_mode == grid_allocation_mode::managed)
                                   ? cudaMallocManaged(&d_grid_bf16, required_bytes, cudaMemAttachGlobal)
                                   : cudaMalloc(&d_grid_bf16, required_bytes);
    if (alloc_status != cudaSuccess) {
      device_grid_capacity    = 0;
      device_grid_bytes       = 0;
      device_grid_nof_ports   = 0;
      device_grid_nof_symbols = 0;
      device_grid_nof_subc    = 0;
      return false;
    }
    device_grid_capacity = required_bytes;
  }

  device_grid_bytes       = required_bytes;
  device_grid_nof_ports   = nof_ports_;
  device_grid_nof_symbols = nof_symbols_;
  device_grid_nof_subc    = nof_subc_;
  host_grid_words.resize(device_grid_bytes / sizeof(uint16_t));
  {
    std::lock_guard<std::mutex> lock(grid_ready_mutex);
    pending_grid_ready_events.clear();
    grid_ready_event        = nullptr;
    clear_event_recorded    = false;
    device_grid_cleared     = false;
    grid_ready_recorded     = false;
    device_grid_has_mapping = false;
  }
  return true;
}

bool pdsch_device_grid_writer_cuda::clear_device_grid_async(void* stream)
{
  std::lock_guard<std::mutex> lock(grid_ready_mutex);
  if ((d_grid_bf16 == nullptr) || (clear_event == nullptr)) {
    return false;
  }
  pending_grid_ready_events.clear();
  grid_ready_event        = nullptr;
  grid_ready_recorded     = false;
  device_grid_has_mapping = false;

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (cudaMemsetAsync(d_grid_bf16, 0, device_grid_bytes, cuda_stream) != cudaSuccess) {
    clear_event_recorded = false;
    device_grid_cleared  = false;
    return false;
  }
  if (cudaEventRecord(static_cast<cudaEvent_t>(clear_event), cuda_stream) != cudaSuccess) {
    clear_event_recorded = false;
    device_grid_cleared  = false;
    return false;
  }
  clear_event_recorded = true;
  device_grid_cleared  = true;
  return true;
}

bool pdsch_device_grid_writer_cuda::copy_device_grid_to_host(span<uint16_t> output) const
{
  if ((d_grid_bf16 == nullptr) || (output.size() * sizeof(uint16_t) < device_grid_bytes)) {
    return false;
  }
  if (!synchronize_device_grid_ready()) {
    return false;
  }
  if (allocation_mode == grid_allocation_mode::managed) {
    std::memcpy(output.data(), d_grid_bf16, device_grid_bytes);
    return true;
  }
  return cudaMemcpy(output.data(), d_grid_bf16, device_grid_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool pdsch_device_grid_writer_cuda::prepare_device_grid_mapping(void* stream)
{
  std::lock_guard<std::mutex> lock(grid_ready_mutex);
  if (!supports_device_grid_mapping()) {
    return false;
  }

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (!device_grid_cleared) {
    pending_grid_ready_events.clear();
    grid_ready_event        = nullptr;
    grid_ready_recorded     = false;
    device_grid_has_mapping = false;

    if (cudaMemsetAsync(d_grid_bf16, 0, device_grid_bytes, cuda_stream) != cudaSuccess) {
      clear_event_recorded = false;
      device_grid_cleared  = false;
      return false;
    }
    if (cudaEventRecord(static_cast<cudaEvent_t>(clear_event), cuda_stream) != cudaSuccess) {
      clear_event_recorded = false;
      device_grid_cleared  = false;
      return false;
    }
    clear_event_recorded = true;
    device_grid_cleared  = true;
    return true;
  }

  if (clear_event_recorded &&
      (cudaStreamWaitEvent(cuda_stream, static_cast<cudaEvent_t>(clear_event), 0) != cudaSuccess)) {
    return false;
  }
  return true;
}

bool pdsch_device_grid_writer_cuda::on_device_grid_mapping_enqueued(void* stream)
{
  std::lock_guard<std::mutex> lock(grid_ready_mutex);
  void*                       event = get_next_grid_ready_event_locked();
  if (event == nullptr) {
    return false;
  }
  if (cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)) != cudaSuccess) {
    grid_ready_recorded = false;
    return false;
  }
  pending_grid_ready_events.push_back(event);
  grid_ready_event        = event;
  grid_ready_recorded     = true;
  device_grid_has_mapping = true;
  return true;
}

bool pdsch_device_grid_writer_cuda::synchronize_device_grid_ready() const
{
  std::lock_guard<std::mutex> lock(grid_ready_mutex);
  if (clear_event_recorded && (clear_event != nullptr) &&
      (cudaEventSynchronizeYielding(static_cast<cudaEvent_t>(clear_event)) != cudaSuccess)) {
    return false;
  }
  if (!grid_ready_recorded || pending_grid_ready_events.empty()) {
    return true;
  }
  for (void* event : pending_grid_ready_events) {
    if (cudaEventSynchronizeYielding(static_cast<cudaEvent_t>(event)) != cudaSuccess) {
      return false;
    }
  }
  return true;
}

bool pdsch_device_grid_writer_cuda::materialize_nonzero_device_grid_to_host()
{
  {
    std::lock_guard<std::mutex> lock(grid_ready_mutex);
    if (!device_grid_has_mapping) {
      return true;
    }
  }
  if (!synchronize_device_grid_ready()) {
    return false;
  }

  if (host_grid_words.size() * sizeof(uint16_t) < device_grid_bytes) {
    host_grid_words.resize(device_grid_bytes / sizeof(uint16_t));
  }
  if (allocation_mode == grid_allocation_mode::managed) {
    std::memcpy(host_grid_words.data(), d_grid_bf16, device_grid_bytes);
  } else if (cudaMemcpy(host_grid_words.data(), d_grid_bf16, device_grid_bytes, cudaMemcpyDeviceToHost) !=
             cudaSuccess) {
    return false;
  }

  static_assert(sizeof(cbf16_t) == 2U * sizeof(uint16_t), "Unexpected cbf16_t layout.");
  const unsigned nof_re_per_port = device_grid_nof_symbols * device_grid_nof_subc;
  for (unsigned port = 0; port != device_grid_nof_ports; ++port) {
    for (unsigned symbol = 0; symbol != device_grid_nof_symbols; ++symbol) {
      span<cbf16_t> host_symbol = host_writer.get_view(port, symbol);
      unsigned      symbol_base = (port * nof_re_per_port + symbol * device_grid_nof_subc) * 2U;
      for (unsigned subc = 0; subc != device_grid_nof_subc; ++subc) {
        unsigned word_index = symbol_base + subc * 2U;
        if ((host_grid_words[word_index] != 0U) || (host_grid_words[word_index + 1] != 0U)) {
          host_symbol[subc].real = bf16_t(host_grid_words[word_index]);
          host_symbol[subc].imag = bf16_t(host_grid_words[word_index + 1]);
        }
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(grid_ready_mutex);
    pending_grid_ready_events.clear();
    grid_ready_event        = nullptr;
    clear_event_recorded    = false;
    device_grid_cleared     = false;
    grid_ready_recorded     = false;
    device_grid_has_mapping = false;
  }
  return true;
}

void* pdsch_device_grid_writer_cuda::get_next_grid_ready_event_locked()
{
  if (pending_grid_ready_events.size() < grid_ready_events.size()) {
    return grid_ready_events[pending_grid_ready_events.size()];
  }

  cudaEvent_t event = nullptr;
  if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
    return nullptr;
  }
  grid_ready_events.push_back(event);
  return event;
}

span<const cf_t> pdsch_device_grid_writer_cuda::put(unsigned                                   port,
                                                       unsigned                                   l,
                                                       unsigned                                   k_init,
                                                       const bounded_bitset<MAX_NOF_SUBCARRIERS>& mask,
                                                       span<const cf_t>                           symbols)
{
  return host_writer.put(port, l, k_init, mask, symbols);
}

span<const cbf16_t> pdsch_device_grid_writer_cuda::put(unsigned                                   port,
                                                          unsigned                                   l,
                                                          unsigned                                   k_init,
                                                          const bounded_bitset<MAX_NOF_SUBCARRIERS>& mask,
                                                          span<const cbf16_t>                        symbols)
{
  return host_writer.put(port, l, k_init, mask, symbols);
}

void pdsch_device_grid_writer_cuda::put(unsigned port, unsigned l, unsigned k_init, span<const cf_t> symbols)
{
  host_writer.put(port, l, k_init, symbols);
}

void pdsch_device_grid_writer_cuda::put(unsigned            port,
                                           unsigned            l,
                                           unsigned            k_init,
                                           unsigned            stride,
                                           span<const cbf16_t> symbols)
{
  host_writer.put(port, l, k_init, stride, symbols);
}

span<cbf16_t> pdsch_device_grid_writer_cuda::get_view(unsigned port, unsigned l)
{
  return host_writer.get_view(port, l);
}
