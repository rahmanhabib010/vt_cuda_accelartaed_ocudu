// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/adt/span.h"
#include "ocudu/phy/support/resource_grid_writer.h"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace ocudu {

/// Resource-grid writer adaptor that exposes a CUDA BF16 grid destination for PDSCH data mapping.
class pdsch_device_grid_writer_cuda : public resource_grid_writer
{
public:
  enum class grid_allocation_mode { device, managed };

  /// Creates an adaptor over an existing host writer and allocates a device grid with matching dimensions.
  explicit pdsch_device_grid_writer_cuda(resource_grid_writer& host_writer_,
                                            grid_allocation_mode  allocation_mode_ = grid_allocation_mode::device);

  /// Frees the owned CUDA device grid.
  ~pdsch_device_grid_writer_cuda() override;

  pdsch_device_grid_writer_cuda(const pdsch_device_grid_writer_cuda&)            = delete;
  pdsch_device_grid_writer_cuda& operator=(const pdsch_device_grid_writer_cuda&) = delete;

  /// Reallocates the device grid if the requested dimensions exceed the current capacity.
  bool resize(unsigned nof_ports_, unsigned nof_symbols_, unsigned nof_subc_);

  /// Clears the active device grid asynchronously on \c stream.
  bool clear_device_grid_async(void* stream = nullptr);

  /// Copies the active device grid into a host BF16-word buffer.
  bool copy_device_grid_to_host(span<uint16_t> output) const;

  /// Returns the number of bytes in the active device grid.
  size_t get_device_grid_size_bytes() const { return device_grid_bytes; }

  // See interface for documentation.
  unsigned get_nof_ports() const override { return host_writer.get_nof_ports(); }

  // See interface for documentation.
  unsigned get_nof_subc() const override { return host_writer.get_nof_subc(); }

  // See interface for documentation.
  unsigned get_nof_symbols() const override { return host_writer.get_nof_symbols(); }

  // See interface for documentation.
  span<const cf_t> put(unsigned                                   port,
                       unsigned                                   l,
                       unsigned                                   k_init,
                       const bounded_bitset<MAX_NOF_SUBCARRIERS>& mask,
                       span<const cf_t>                           symbols) override;

  // See interface for documentation.
  span<const cbf16_t> put(unsigned                                   port,
                          unsigned                                   l,
                          unsigned                                   k_init,
                          const bounded_bitset<MAX_NOF_SUBCARRIERS>& mask,
                          span<const cbf16_t>                        symbols) override;

  // See interface for documentation.
  void put(unsigned port, unsigned l, unsigned k_init, span<const cf_t> symbols) override;

  // See interface for documentation.
  void put(unsigned port, unsigned l, unsigned k_init, unsigned stride, span<const cbf16_t> symbols) override;

  // See interface for documentation.
  span<cbf16_t> get_view(unsigned port, unsigned l) override;

  // See interface for documentation.
  bool supports_device_grid_mapping() const override { return (d_grid_bf16 != nullptr) && (clear_event != nullptr); }

  // See interface for documentation.
  void* get_device_grid_bf16() override { return d_grid_bf16; }

  // See interface for documentation.
  bool prepare_device_grid_mapping(void* stream) override;

  // See interface for documentation.
  bool on_device_grid_mapping_enqueued(void* stream) override;

  /// Synchronizes with the most recently enqueued device-grid mapping.
  bool synchronize_device_grid_ready() const;

  // See interface for documentation.
  bool synchronize_device_grid_mapping() override { return synchronize_device_grid_ready(); }

  /// Copies nonzero device-grid REs into the delegated host writer.
  bool materialize_nonzero_device_grid_to_host();

private:
  void* get_next_grid_ready_event_locked();

  resource_grid_writer& host_writer;
  grid_allocation_mode  allocation_mode         = grid_allocation_mode::device;
  void*                 d_grid_bf16             = nullptr;
  void*                 clear_event             = nullptr;
  void*                 grid_ready_event        = nullptr;
  bool                  clear_event_recorded    = false;
  bool                  device_grid_cleared     = false;
  bool                  grid_ready_recorded     = false;
  bool                  device_grid_has_mapping = false;
  size_t                device_grid_capacity    = 0;
  size_t                device_grid_bytes       = 0;
  unsigned              device_grid_nof_ports   = 0;
  unsigned              device_grid_nof_symbols = 0;
  unsigned              device_grid_nof_subc    = 0;
  std::vector<uint16_t> host_grid_words;
  std::vector<void*>    grid_ready_events;
  std::vector<void*>    pending_grid_ready_events;
  mutable std::mutex    grid_ready_mutex;
};

} // namespace ocudu
