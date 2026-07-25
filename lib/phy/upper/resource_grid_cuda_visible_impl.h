// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief Resource grid with CUDA managed backing memory for GPU-visible grids.

#pragma once

#include "../support/resource_grid_reader_impl.h"
#include "../support/resource_grid_writer_impl.h"
#include "phy_acceleration_runtime_options.h"
#include "resource_grid_pinned_impl.h"
#include "ocudu/adt/tensor.h"
#include "ocudu/ocuduvec/zero.h"
#include "ocudu/phy/support/resource_grid.h"
#include "ocudu/phy/support/resource_grid_dimensions.h"
#include "ocudu/phy/support/support_factories.h"
#include "ocudu/support/error_handling.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <sched.h>
#include <vector>

namespace ocudu {

namespace detail {

template <unsigned NDIMS, typename Type, typename IndexType = unsigned>
class external_tensor : public tensor<NDIMS, Type, IndexType>
{
public:
  using dimensions_size_type = typename tensor<NDIMS, Type, IndexType>::dimensions_size_type;

  external_tensor(Type* data_, const dimensions_size_type& dimensions_) : data(data_) { resize(dimensions_); }

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

/// \brief Resource grid whose host writer and GPU reader/writer share CUDA managed backing memory.
class cuda_visible_resource_grid : public resource_grid
{
public:
  cuda_visible_resource_grid(unsigned nof_ports_, unsigned nof_symbols_, unsigned nof_subc_) :
    nof_ports(nof_ports_), nof_symbols(nof_symbols_), nof_subc(nof_subc_)
  {
    if (cudaGetDevice(&device_id) != cudaSuccess) {
      device_id = 0;
    }

    static constexpr unsigned initial_ready_events = 32;
    grid_ready_events.reserve(initial_ready_events);
    pending_grid_ready_events.reserve(initial_ready_events);
    for (unsigned i_event = 0; i_event != initial_ready_events; ++i_event) {
      cudaEvent_t event = nullptr;
      if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
        break;
      }
      if (grid_ready_event == nullptr) {
        grid_ready_event = event;
      }
      grid_ready_events.push_back(event);
    }
    cudaStream_t stream = nullptr;
    if (create_realtime_stream(&stream) == cudaSuccess) {
      maintenance_stream = stream;
    }

    managed_bytes = static_cast<size_t>(nof_ports) * nof_symbols * nof_subc * sizeof(cbf16_t);
    if ((grid_ready_event == nullptr) || managed_bytes == 0 ||
        cudaMallocManaged(&managed_data, managed_bytes) != cudaSuccess) {
      managed_data = nullptr;
      return;
    }
    advise_managed_memory();

    rg_buffer = std::make_unique<managed_tensor>(
        static_cast<cbf16_t*>(managed_data), managed_tensor::dimensions_size_type{nof_subc, nof_symbols, nof_ports});
    writer = std::make_unique<writer_impl>(*this, *rg_buffer, empty);
    reader = std::make_unique<reader_impl>(*this, *rg_buffer, empty);

    set_all_zero();
  }

  ~cuda_visible_resource_grid() override
  {
    if (maintenance_stream != nullptr) {
      cudaStreamSynchronize(static_cast<cudaStream_t>(maintenance_stream));
    }
    for (void* event : grid_ready_events) {
      cudaEventDestroy(static_cast<cudaEvent_t>(event));
    }
    grid_ready_events.clear();
    pending_grid_ready_events.clear();
    grid_ready_event = nullptr;
    if (maintenance_stream != nullptr) {
      cudaStreamDestroy(static_cast<cudaStream_t>(maintenance_stream));
      maintenance_stream = nullptr;
    }
    if (managed_data != nullptr) {
      cudaFree(managed_data);
      managed_data = nullptr;
    }
  }

  bool is_valid() const { return managed_data != nullptr; }

  void set_all_zero() override
  {
    if (!reader) {
      return;
    }

    unsigned all_empty = all_ports_empty_mask();
    if (empty.load(std::memory_order_acquire) == all_empty) {
      return;
    }

    if (clear_on_device()) {
      empty.store(all_empty, std::memory_order_release);
      return;
    }

    for (unsigned port = 0; port != nof_ports; ++port) {
      if (!reader->is_port_empty(port)) {
        ocuduvec::zero(rg_buffer->template get_view<static_cast<unsigned>(resource_grid_dimensions::port)>({port}));
      }
    }
    empty.store(all_empty, std::memory_order_release);
    std::lock_guard<std::mutex> lock(grid_ready_mutex);
    pending_grid_ready_events.clear();
    grid_ready_recorded = false;
  }

  resource_grid_writer& get_writer() override { return *writer; }

  const resource_grid_reader& get_reader() const override { return *reader; }

private:
  using managed_tensor =
      detail::external_tensor<static_cast<unsigned>(resource_grid_dimensions::all), cbf16_t, resource_grid_dimensions>;

  class reader_impl : public resource_grid_reader_impl
  {
  public:
    reader_impl(const cuda_visible_resource_grid& parent_,
                const storage_type&               data_,
                const std::atomic<unsigned>&      empty_) :
      resource_grid_reader_impl(data_, empty_), parent(parent_)
    {
    }

    bool supports_device_grid_reading() const override { return parent.managed_data != nullptr; }

    const void* get_device_grid_cbf16() const override { return parent.managed_data; }

    void* get_device_grid_ready_event() const override { return parent.grid_ready_event; }

    bool prepare_device_grid_reading(void* stream) const override { return parent.prepare_device_access(stream); }

    bool on_device_grid_reading_enqueued(void* stream) const override { return parent.record_device_access(stream); }

    bool synchronize_device_grid_reading() const override { return parent.synchronize_device_mapping(); }

    bool is_empty(unsigned port) const override
    {
      return parent.prepare_host_access() && resource_grid_reader_impl::is_empty(port);
    }

    bool is_empty() const override { return parent.prepare_host_access() && resource_grid_reader_impl::is_empty(); }

    span<cf_t> get(span<cf_t>                                 symbols,
                   unsigned                                   port,
                   unsigned                                   l,
                   unsigned                                   k_init,
                   const bounded_bitset<MAX_NOF_SUBCARRIERS>& mask) const override
    {
      report_fatal_error_if_not(parent.prepare_host_access(), "Failed to synchronize CUDA-visible resource grid.");
      return resource_grid_reader_impl::get(symbols, port, l, k_init, mask);
    }

    span<cbf16_t> get(span<cbf16_t>                              symbols,
                      unsigned                                   port,
                      unsigned                                   l,
                      unsigned                                   k_init,
                      const bounded_bitset<MAX_NOF_SUBCARRIERS>& mask) const override
    {
      report_fatal_error_if_not(parent.prepare_host_access(), "Failed to synchronize CUDA-visible resource grid.");
      return resource_grid_reader_impl::get(symbols, port, l, k_init, mask);
    }

    void get(span<cf_t> symbols, unsigned port, unsigned l, unsigned k_init, unsigned stride = 1) const override
    {
      report_fatal_error_if_not(parent.prepare_host_access(), "Failed to synchronize CUDA-visible resource grid.");
      resource_grid_reader_impl::get(symbols, port, l, k_init, stride);
    }

    void get(span<cbf16_t> symbols, unsigned port, unsigned l, unsigned k_init) const override
    {
      report_fatal_error_if_not(parent.prepare_host_access(), "Failed to synchronize CUDA-visible resource grid.");
      resource_grid_reader_impl::get(symbols, port, l, k_init);
    }

    span<const cbf16_t> get_view(unsigned port, unsigned l) const override
    {
      report_fatal_error_if_not(parent.prepare_host_access(), "Failed to synchronize CUDA-visible resource grid.");
      return resource_grid_reader_impl::get_view(port, l);
    }

  private:
    const cuda_visible_resource_grid& parent;
  };

  class writer_impl : public resource_grid_writer_impl
  {
  public:
    writer_impl(cuda_visible_resource_grid& parent_, storage_type& data_, std::atomic<unsigned>& empty_) :
      resource_grid_writer_impl(data_, empty_), parent(parent_)
    {
    }

    bool supports_device_grid_mapping() const override { return parent.managed_data != nullptr; }

    bool device_grid_mapping_aliases_host_grid() const override { return parent.managed_data != nullptr; }

    void* get_device_grid_bf16() override { return parent.managed_data; }

    span<const cf_t> put(unsigned                                   port,
                         unsigned                                   l,
                         unsigned                                   k_init,
                         const bounded_bitset<MAX_NOF_SUBCARRIERS>& mask,
                         span<const cf_t>                           symbols) override
    {
      report_fatal_error_if_not(parent.prepare_host_access(), "Failed to synchronize CUDA-visible resource grid.");
      return resource_grid_writer_impl::put(port, l, k_init, mask, symbols);
    }

    span<const cbf16_t> put(unsigned                                   port,
                            unsigned                                   l,
                            unsigned                                   k_init,
                            const bounded_bitset<MAX_NOF_SUBCARRIERS>& mask,
                            span<const cbf16_t>                        symbols) override
    {
      report_fatal_error_if_not(parent.prepare_host_access(), "Failed to synchronize CUDA-visible resource grid.");
      return resource_grid_writer_impl::put(port, l, k_init, mask, symbols);
    }

    void put(unsigned port, unsigned l, unsigned k_init, span<const cf_t> symbols) override
    {
      report_fatal_error_if_not(parent.prepare_host_access(), "Failed to synchronize CUDA-visible resource grid.");
      resource_grid_writer_impl::put(port, l, k_init, symbols);
    }

    void put(unsigned port, unsigned l, unsigned k_init, unsigned stride, span<const cbf16_t> symbols) override
    {
      report_fatal_error_if_not(parent.prepare_host_access(), "Failed to synchronize CUDA-visible resource grid.");
      resource_grid_writer_impl::put(port, l, k_init, stride, symbols);
    }

    span<cbf16_t> get_view(unsigned port, unsigned l) override
    {
      report_fatal_error_if_not(parent.prepare_host_access(), "Failed to synchronize CUDA-visible resource grid.");
      return resource_grid_writer_impl::get_view(port, l);
    }

    bool prepare_device_grid_mapping(void* stream) override { return parent.prepare_device_mapping(stream); }

    bool on_device_grid_mapping_enqueued(void* stream) override
    {
      if (!parent.record_device_mapping(stream)) {
        return false;
      }
      for (unsigned port = 0; port != parent.nof_ports; ++port) {
        clear_empty(port);
      }
      return true;
    }

    bool cancel_device_grid_mapping() override { return parent.cancel_device_mapping(); }

    bool synchronize_device_grid_mapping() override { return parent.synchronize_device_mapping(); }

  private:
    cuda_visible_resource_grid& parent;
  };

  bool record_device_mapping(void* stream)
  {
    std::lock_guard<std::mutex> lock(grid_ready_mutex);
    void*                       event = get_next_grid_ready_event();
    if (event == nullptr) {
      device_mapping_active           = false;
      grid_ready_recorded             = false;
      device_mapping_replaces_pending = false;
      return false;
    }
    if (cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)) != cudaSuccess) {
      device_mapping_active           = false;
      grid_ready_recorded             = false;
      device_mapping_replaces_pending = false;
      return false;
    }
    if (device_mapping_replaces_pending) {
      pending_grid_ready_events.clear();
    }
    pending_grid_ready_events.push_back(event);
    grid_ready_event                = event;
    grid_ready_recorded             = !pending_grid_ready_events.empty();
    device_mapping_active           = false;
    device_mapping_replaces_pending = false;
    return true;
  }

  bool wait_for_device_mapping(void* stream) const
  {
    std::lock_guard<std::mutex> lock(grid_ready_mutex);
    return wait_for_device_mapping_locked(stream);
  }

  bool record_device_access(void* stream) const
  {
    std::lock_guard<std::mutex> lock(grid_ready_mutex);
    void*                       event = get_next_grid_ready_event();
    if (event == nullptr) {
      grid_ready_recorded = !pending_grid_ready_events.empty();
      return false;
    }
    if (cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)) != cudaSuccess) {
      grid_ready_recorded = !pending_grid_ready_events.empty();
      return false;
    }
    pending_grid_ready_events.push_back(event);
    grid_ready_event    = event;
    grid_ready_recorded = !pending_grid_ready_events.empty();
    return true;
  }

  bool wait_for_device_mapping_locked(void* stream) const
  {
    if (!grid_ready_recorded || pending_grid_ready_events.empty()) {
      return managed_data != nullptr;
    }
    for (void* event : pending_grid_ready_events) {
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

  bool prepare_device_mapping(void* stream)
  {
    std::unique_lock<std::mutex> lock(grid_ready_mutex);
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
    device_mapping_active           = true;
    device_mapping_replaces_pending = true;
    return true;
  }

  bool cancel_device_mapping()
  {
    std::lock_guard<std::mutex> lock(grid_ready_mutex);
    device_mapping_active           = false;
    device_mapping_replaces_pending = false;
    return true;
  }

  bool prepare_device_access(void* stream) const
  {
    std::unique_lock<std::mutex> lock(grid_ready_mutex);
    if (managed_data == nullptr) {
      return false;
    }
    wait_for_active_device_mapping_locked(lock);
    if (!wait_for_device_mapping_locked(stream)) {
      return false;
    }
    return prefetch_managed_memory(device_location(), stream);
  }

  bool synchronize_device_mapping() { return prepare_host_access(); }

  bool synchronize_device_mapping() const { return prepare_host_access(); }

  bool prepare_host_access() const
  {
    std::unique_lock<std::mutex> lock(grid_ready_mutex);
    wait_for_active_device_mapping_locked(lock);
    if (!grid_ready_recorded || pending_grid_ready_events.empty()) {
      return true;
    }
    for (void* event : pending_grid_ready_events) {
      if (cuda_event_synchronize_yielding(static_cast<cudaEvent_t>(event)) != cudaSuccess) {
        return false;
      }
    }
    if (!prefetch_managed_memory(host_location(), nullptr)) {
      return false;
    }
    pending_grid_ready_events.clear();
    grid_ready_recorded = false;
    return true;
  }

  void* get_next_grid_ready_event() const
  {
    for (void* event : grid_ready_events) {
      if (std::find(pending_grid_ready_events.begin(), pending_grid_ready_events.end(), event) ==
          pending_grid_ready_events.end()) {
        return event;
      }
    }

    cudaEvent_t event = nullptr;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
      return nullptr;
    }
    grid_ready_events.push_back(event);
    return event;
  }

  unsigned all_ports_empty_mask() const
  {
    return (nof_ports >= 8U * sizeof(unsigned)) ? ~0U : ((1U << nof_ports) - 1U);
  }

  bool clear_on_device()
  {
    if ((managed_data == nullptr) || (managed_bytes == 0) || (maintenance_stream == nullptr)) {
      return false;
    }

    std::unique_lock<std::mutex> lock(grid_ready_mutex);
    wait_for_active_device_mapping_locked(lock);
    cudaStream_t stream = static_cast<cudaStream_t>(maintenance_stream);
    if (!wait_for_device_mapping_locked(stream)) {
      return false;
    }
    void* event = get_next_grid_ready_event();
    if (event == nullptr) {
      return false;
    }
    if (cudaMemsetAsync(managed_data, 0, managed_bytes, stream) != cudaSuccess) {
      return false;
    }
    if (cudaEventRecord(static_cast<cudaEvent_t>(event), stream) != cudaSuccess) {
      return false;
    }
    pending_grid_ready_events.clear();
    pending_grid_ready_events.push_back(event);
    grid_ready_event                = event;
    grid_ready_recorded             = true;
    device_mapping_active           = false;
    device_mapping_replaces_pending = false;
    return true;
  }

  static cudaError_t cuda_event_synchronize_yielding(cudaEvent_t event)
  {
    cudaError_t status;
    while ((status = cudaEventQuery(event)) == cudaErrorNotReady) {
      sched_yield();
    }
    return status;
  }

  static cudaError_t create_realtime_stream(cudaStream_t* stream)
  {
    int         least_priority    = 0;
    int         greatest_priority = 0;
    cudaError_t status            = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    if (status == cudaSuccess) {
      status = cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking, greatest_priority);
      if (status == cudaSuccess) {
        return cudaSuccess;
      }
      (void)cudaGetLastError();
    }

    return cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
  }

  void advise_managed_memory() const
  {
    if (managed_data == nullptr) {
      return;
    }
    // Best-effort hints only. The public resource-grid contract must not depend on managed-memory advice support.
    (void)cudaMemAdvise(managed_data, managed_bytes, cudaMemAdviseSetAccessedBy, device_location());
    if (phy_acceleration_env_flag_enabled("OCUDU_CUDA_VISIBLE_GRID_PREFER_DEVICE", false)) {
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
  mutable void*              grid_ready_event   = nullptr;
  void*                      maintenance_stream = nullptr;
  mutable std::vector<void*> grid_ready_events;
  mutable std::vector<void*> pending_grid_ready_events;
  std::atomic<unsigned>      empty = {};
  mutable std::mutex         grid_ready_mutex;
  mutable bool               grid_ready_recorded             = false;
  mutable bool               device_mapping_active           = false;
  mutable bool               device_mapping_replaces_pending = false;
  bool     prefetch_enabled = phy_acceleration_env_flag_enabled("OCUDU_CUDA_VISIBLE_GRID_PREFETCH", false);
  int      device_id        = 0;
  unsigned nof_ports        = 0;
  unsigned nof_symbols      = 0;
  unsigned nof_subc         = 0;
  std::unique_ptr<managed_tensor>            rg_buffer;
  std::unique_ptr<resource_grid_writer_impl> writer;
  std::unique_ptr<reader_impl>               reader;
};

/// \brief Resource-grid factory that uses CUDA managed memory only on integrated/unified CUDA devices.
class resource_grid_cuda_visible_factory : public resource_grid_factory
{
public:
  enum class direction { generic, downlink, uplink };

  explicit resource_grid_cuda_visible_factory(direction grid_direction_ = direction::generic) :
    grid_direction(grid_direction_), use_managed_grid(should_use_managed_grid(grid_direction))
  {
  }

  std::unique_ptr<resource_grid> create(unsigned nof_ports, unsigned nof_symbols, unsigned nof_subc) override
  {
    if (use_managed_grid) {
      auto grid = std::make_unique<cuda_visible_resource_grid>(nof_ports, nof_symbols, nof_subc);
      if (grid->is_valid()) {
        return grid;
      }
    }
    return fallback_factory->create(nof_ports, nof_symbols, nof_subc);
  }

private:
  static const char* get_direction_env_name(direction grid_direction)
  {
    switch (grid_direction) {
      case direction::downlink:
        return "OCUDU_DL_CUDA_VISIBLE_GRID";
      case direction::uplink:
        return "OCUDU_UL_CUDA_VISIBLE_GRID";
      case direction::generic:
      default:
        return nullptr;
    }
  }

  static bool mode_uses_managed(const char* mode)
  {
    if (phy_acceleration_env_mode_is_managed(mode)) {
      return true;
    }
    if ((mode != nullptr) && ((std::strcmp(mode, "pinned") == 0) || (std::strcmp(mode, "off") == 0) ||
                              (std::strcmp(mode, "disabled") == 0))) {
      return false;
    }
    // Unknown values, including "auto", keep the automatic platform policy.
    return should_use_managed_grid_auto();
  }

  static bool should_use_managed_grid(direction grid_direction)
  {
    const char* mode = phy_acceleration_cuda_visible_grid_mode(get_direction_env_name(grid_direction));
    if ((mode == nullptr) && (grid_direction == direction::generic)) {
      mode = std::getenv("OCUDU_UL_CUDA_VISIBLE_GRID");
    }
    if (mode != nullptr) {
      return mode_uses_managed(mode);
    }

    return should_use_managed_grid_auto();
  }

  static bool should_use_managed_grid_auto()
  {
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
      return false;
    }

    int integrated = 0;
    if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device_id) != cudaSuccess) {
      return false;
    }
    return integrated != 0;
  }

  direction                              grid_direction;
  bool                                   use_managed_grid;
  std::shared_ptr<resource_grid_factory> fallback_factory = std::make_shared<resource_grid_pinned_factory>();
};

} // namespace ocudu
