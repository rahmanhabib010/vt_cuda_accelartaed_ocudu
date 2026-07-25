// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief PRACH buffer with CUDA-visible managed backing memory.

#pragma once

#include "phy_acceleration_runtime_options.h"
#include "ocudu/adt/tensor.h"
#include "ocudu/phy/support/prach_buffer.h"
#include "ocudu/support/error_handling.h"
#include <atomic>
#include <cstdlib>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <sched.h>
#include <type_traits>
#include <vector>

namespace ocudu {

namespace detail {

template <unsigned NDIMS, typename Type, typename IndexType = unsigned>
class prach_external_tensor : public tensor<NDIMS, Type, IndexType>
{
public:
  using dimensions_size_type = typename tensor<NDIMS, Type, IndexType>::dimensions_size_type;

  prach_external_tensor(Type* data_, const dimensions_size_type& dimensions_) : data(data_) { resize(dimensions_); }

  void resize(const dimensions_size_type& dimensions_) override { dimensions = dimensions_; }

  const dimensions_size_type& get_dimensions_size() const override { return dimensions; }

protected:
  span<Type> get_data() override { return span<Type>(data, size()); }

  span<const Type> get_data() const override { return span<const Type>(data, size()); }

private:
  unsigned size() const { return std::accumulate(dimensions.begin(), dimensions.end(), 1U, std::multiplies<>()); }

  Type*                data = nullptr;
  dimensions_size_type dimensions;
};

} // namespace detail

/// PRACH buffer whose host API and device API share a CUDA managed allocation.
class cuda_visible_prach_buffer : public prach_buffer
{
public:
  cuda_visible_prach_buffer(unsigned max_nof_ports_,
                            unsigned max_nof_td_occasions_,
                            unsigned max_nof_fd_occasions_,
                            unsigned max_nof_symbols_,
                            unsigned sequence_length_) :
    max_nof_ports(max_nof_ports_),
    max_nof_td_occasions(max_nof_td_occasions_),
    max_nof_fd_occasions(max_nof_fd_occasions_),
    max_nof_symbols(max_nof_symbols_),
    sequence_length(sequence_length_)
  {
    if (cudaGetDevice(&device_id) != cudaSuccess) {
      device_id = 0;
    }

    static constexpr unsigned initial_ready_events = 16;
    buffer_ready_events.reserve(initial_ready_events);
    pending_buffer_ready_events.reserve(initial_ready_events);
    for (unsigned i_event = 0; i_event != initial_ready_events; ++i_event) {
      cudaEvent_t event = nullptr;
      if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
        break;
      }
      if (buffer_ready_event == nullptr) {
        buffer_ready_event = event;
      }
      buffer_ready_events.push_back(event);
    }

    managed_bytes = static_cast<size_t>(max_nof_ports) * max_nof_td_occasions * max_nof_fd_occasions * max_nof_symbols *
                    sequence_length * sizeof(cbf16_t);
    if ((buffer_ready_event == nullptr) || (managed_bytes == 0) ||
        (cudaMallocManaged(&managed_data, managed_bytes) != cudaSuccess)) {
      managed_data = nullptr;
      return;
    }

    advise_managed_memory();
    data = std::make_unique<managed_tensor>(
        static_cast<cbf16_t*>(managed_data),
        managed_tensor::dimensions_size_type{
            sequence_length, max_nof_symbols, max_nof_fd_occasions, max_nof_td_occasions, max_nof_ports});
  }

  ~cuda_visible_prach_buffer() override
  {
    for (void* event : buffer_ready_events) {
      cudaEventDestroy(static_cast<cudaEvent_t>(event));
    }
    buffer_ready_events.clear();
    pending_buffer_ready_events.clear();
    buffer_ready_event = nullptr;
    if (managed_data != nullptr) {
      cudaFree(managed_data);
      managed_data = nullptr;
    }
  }

  bool is_valid() const { return managed_data != nullptr; }

  unsigned get_max_nof_ports() const override { return max_nof_ports; }

  unsigned get_max_nof_td_occasions() const override { return max_nof_td_occasions; }

  unsigned get_max_nof_fd_occasions() const override { return max_nof_fd_occasions; }

  unsigned get_max_nof_symbols() const override { return max_nof_symbols; }

  unsigned get_sequence_length() const override { return sequence_length; }

  span<cbf16_t> get_symbol(unsigned i_port, unsigned i_td_occasion, unsigned i_fd_occasion, unsigned i_symbol) override
  {
    report_fatal_error_if_not(prepare_host_access(), "Failed to synchronize CUDA-visible PRACH buffer.");
    return data->get_view({i_symbol, i_fd_occasion, i_td_occasion, i_port});
  }

  span<const cbf16_t>
  get_symbol(unsigned i_port, unsigned i_td_occasion, unsigned i_fd_occasion, unsigned i_symbol) const override
  {
    report_fatal_error_if_not(prepare_host_access(), "Failed to synchronize CUDA-visible PRACH buffer.");
    return data->get_view({i_symbol, i_fd_occasion, i_td_occasion, i_port});
  }

  bool supports_device_prach_buffer_reading() const override { return managed_data != nullptr; }

  bool supports_device_prach_buffer_mapping() const override { return managed_data != nullptr; }

  const void* get_device_prach_buffer_cbf16() const override { return managed_data; }

  void* get_device_prach_buffer_cbf16() override { return managed_data; }

  unsigned get_device_prach_symbol_offset(unsigned i_port,
                                          unsigned i_td_occasion,
                                          unsigned i_fd_occasion,
                                          unsigned i_symbol) const override
  {
    ocudu_assert(i_port < max_nof_ports,
                 "The port index (i.e., {}) exceeds the maximum number of ports (i.e., {}).",
                 i_port,
                 max_nof_ports);
    ocudu_assert(i_td_occasion < max_nof_td_occasions,
                 "The time-domain occasion (i.e., {}) exceeds the maximum number of time-domain occasions (i.e., {}).",
                 i_td_occasion,
                 max_nof_td_occasions);
    ocudu_assert(
        i_fd_occasion < max_nof_fd_occasions,
        "The frequency-domain occasion (i.e., {}) exceeds the maximum number of frequency-domain occasions (i.e., {}).",
        i_fd_occasion,
        max_nof_fd_occasions);
    ocudu_assert(i_symbol < max_nof_symbols,
                 "The symbol index (i.e., {}) exceeds the maximum number of symbols (i.e., {}).",
                 i_symbol,
                 max_nof_symbols);

    return sequence_length *
           (i_symbol +
            max_nof_symbols * (i_fd_occasion + max_nof_fd_occasions * (i_td_occasion + max_nof_td_occasions * i_port)));
  }

  bool prepare_device_prach_buffer_reading(void* stream) const override { return prepare_device_access(stream); }

  bool prepare_device_prach_buffer_mapping(void* stream) override { return prepare_device_mapping(stream); }

  bool on_device_prach_buffer_mapping_enqueued(void* stream) override { return record_device_mapping(stream); }

  bool cancel_device_prach_buffer_mapping() override { return cancel_device_mapping(); }

  bool synchronize_device_prach_buffer_mapping() const override { return prepare_host_access(); }

private:
  enum class dims : unsigned {
    re          = 0,
    symbol      = 1,
    fd_occasion = 2,
    td_occasion = 3,
    port        = 4,
    count       = 5,
  };

  using managed_tensor =
      detail::prach_external_tensor<static_cast<std::underlying_type_t<dims>>(dims::count), cbf16_t, dims>;

  bool record_device_mapping(void* stream)
  {
    std::lock_guard<std::mutex> lock(buffer_ready_mutex);
    void*                       event = get_next_buffer_ready_event();
    if (event == nullptr) {
      device_mapping_active = false;
      buffer_ready_recorded = false;
      return false;
    }
    if (cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)) != cudaSuccess) {
      device_mapping_active = false;
      buffer_ready_recorded = false;
      return false;
    }
    pending_buffer_ready_events.push_back(event);
    buffer_ready_event    = event;
    buffer_ready_recorded = !pending_buffer_ready_events.empty();
    device_mapping_active = false;
    return true;
  }

  bool prepare_device_mapping(void* stream)
  {
    std::unique_lock<std::mutex> lock(buffer_ready_mutex);
    if (managed_data == nullptr) {
      return false;
    }
    wait_for_active_device_mapping_locked(lock);
    if (!wait_for_device_mapping_locked(stream)) {
      return false;
    }
    if (!prefetch_managed_memory(device_location(), stream)) {
      return false;
    }
    device_mapping_active = true;
    return true;
  }

  bool cancel_device_mapping()
  {
    std::lock_guard<std::mutex> lock(buffer_ready_mutex);
    device_mapping_active = false;
    return true;
  }

  bool prepare_device_access(void* stream) const
  {
    std::unique_lock<std::mutex> lock(buffer_ready_mutex);
    if (managed_data == nullptr) {
      return false;
    }
    wait_for_active_device_mapping_locked(lock);
    if (!wait_for_device_mapping_locked(stream)) {
      return false;
    }
    return prefetch_managed_memory(device_location(), stream);
  }

  bool prepare_host_access() const
  {
    std::unique_lock<std::mutex> lock(buffer_ready_mutex);
    wait_for_active_device_mapping_locked(lock);
    if (!buffer_ready_recorded || pending_buffer_ready_events.empty()) {
      return true;
    }
    for (void* event : pending_buffer_ready_events) {
      if (cuda_event_synchronize_yielding(static_cast<cudaEvent_t>(event)) != cudaSuccess) {
        return false;
      }
    }
    if (!prefetch_managed_memory(host_location(), nullptr)) {
      return false;
    }
    pending_buffer_ready_events.clear();
    buffer_ready_recorded = false;
    return true;
  }

  bool wait_for_device_mapping_locked(void* stream) const
  {
    if (!buffer_ready_recorded || pending_buffer_ready_events.empty()) {
      return managed_data != nullptr;
    }
    for (void* event : pending_buffer_ready_events) {
      if (cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(event), 0) != cudaSuccess) {
        return false;
      }
    }
    return true;
  }

  void wait_for_active_device_mapping_locked(std::unique_lock<std::mutex>& lock) const
  {
    while (device_mapping_active) {
      lock.unlock();
      sched_yield();
      lock.lock();
    }
  }

  void* get_next_buffer_ready_event()
  {
    if (pending_buffer_ready_events.size() < buffer_ready_events.size()) {
      return buffer_ready_events[pending_buffer_ready_events.size()];
    }

    cudaEvent_t event = nullptr;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
      return nullptr;
    }
    buffer_ready_events.push_back(event);
    return event;
  }

  static cudaError_t cuda_event_synchronize_yielding(cudaEvent_t event)
  {
    cudaError_t status;
    while ((status = cudaEventQuery(event)) == cudaErrorNotReady) {
      sched_yield();
    }
    return status;
  }

  void advise_managed_memory() const
  {
    if (managed_data == nullptr) {
      return;
    }
    (void)cudaMemAdvise(managed_data, managed_bytes, cudaMemAdviseSetAccessedBy, device_location());
    if (phy_acceleration_env_flag_enabled("OCUDU_CUDA_VISIBLE_PRACH_BUFFER_PREFER_DEVICE", false)) {
      (void)cudaMemAdvise(managed_data, managed_bytes, cudaMemAdviseSetPreferredLocation, device_location());
    }
  }

#if CUDART_VERSION >= 13000
  cudaMemLocation device_location() const { return cudaMemLocation{cudaMemLocationTypeDevice, device_id}; }

  static cudaMemLocation host_location() { return cudaMemLocation{cudaMemLocationTypeHost, 0}; }

  static bool is_host_location(cudaMemLocation destination)
  {
    return (destination.type == cudaMemLocationTypeHost) || (destination.type == cudaMemLocationTypeHostNuma) ||
           (destination.type == cudaMemLocationTypeHostNumaCurrent);
  }
#else
  int device_location() const { return device_id; }

  static int host_location() { return cudaCpuDeviceId; }

  static bool is_host_location(int destination) { return destination == cudaCpuDeviceId; }
#endif

  bool prefetch_managed_memory(decltype(host_location()) destination, void* stream) const
  {
    if (!prefetch_enabled || (managed_data == nullptr)) {
      return true;
    }
#if CUDART_VERSION >= 13000
    cudaError_t status =
        cudaMemPrefetchAsync(managed_data, managed_bytes, destination, 0, static_cast<cudaStream_t>(stream));
#else
    cudaError_t status =
        cudaMemPrefetchAsync(managed_data, managed_bytes, destination, static_cast<cudaStream_t>(stream));
#endif
    if (status == cudaErrorNotSupported) {
      return true;
    }
    if (status != cudaSuccess) {
      return false;
    }
    if (is_host_location(destination)) {
      return cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) == cudaSuccess;
    }
    return true;
  }

  void*                      managed_data       = nullptr;
  size_t                     managed_bytes      = 0;
  void*                      buffer_ready_event = nullptr;
  std::vector<void*>         buffer_ready_events;
  mutable std::vector<void*> pending_buffer_ready_events;
  mutable std::mutex         buffer_ready_mutex;
  mutable bool               buffer_ready_recorded = false;
  mutable bool               device_mapping_active = false;
  // PRACH is timing-critical and the GPU-resident path should not pay managed-memory page migration on first access.
  bool     prefetch_enabled     = phy_acceleration_env_flag_enabled("OCUDU_CUDA_VISIBLE_PRACH_BUFFER_PREFETCH", true);
  int      device_id            = 0;
  unsigned max_nof_ports        = 0;
  unsigned max_nof_td_occasions = 0;
  unsigned max_nof_fd_occasions = 0;
  unsigned max_nof_symbols      = 0;
  unsigned sequence_length      = 0;
  std::unique_ptr<managed_tensor> data;
};

} // namespace ocudu
