// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/phy/support/resource_grid_reader.h"
#include <cstddef>

namespace ocudu {

/// Resource-grid reader adaptor that exposes a CUDA-visible BF16 grid source for PUSCH GPU processing.
class pusch_device_grid_reader_cuda : public resource_grid_reader
{
public:
  enum class grid_allocation_mode { device, managed };

  explicit pusch_device_grid_reader_cuda(const resource_grid_reader& host_reader_,
                                            grid_allocation_mode allocation_mode_ = grid_allocation_mode::device);

  ~pusch_device_grid_reader_cuda() override;

  pusch_device_grid_reader_cuda(const pusch_device_grid_reader_cuda&)            = delete;
  pusch_device_grid_reader_cuda& operator=(const pusch_device_grid_reader_cuda&) = delete;

  /// Reallocates the device-visible grid if the requested dimensions exceed the current capacity.
  bool resize(unsigned nof_ports_, unsigned nof_symbols_, unsigned nof_subc_);

  /// Stages the delegated host reader into the CUDA-visible grid and records a readiness event.
  bool stage_host_grid_async(void* stream = nullptr);

  /// Synchronizes with the most recently staged device-visible grid.
  bool synchronize_device_grid_ready() const;

  // See interface for documentation.
  unsigned get_nof_ports() const override { return host_reader.get_nof_ports(); }

  // See interface for documentation.
  unsigned get_nof_subc() const override { return host_reader.get_nof_subc(); }

  // See interface for documentation.
  unsigned get_nof_symbols() const override { return host_reader.get_nof_symbols(); }

  // See interface for documentation.
  bool is_empty(unsigned port) const override { return host_reader.is_empty(port); }

  // See interface for documentation.
  bool is_empty() const override { return host_reader.is_empty(); }

  // See interface for documentation.
  span<cf_t> get(span<cf_t>                                 symbols,
                 unsigned                                   port,
                 unsigned                                   l,
                 unsigned                                   k_init,
                 const bounded_bitset<MAX_NOF_SUBCARRIERS>& mask) const override
  {
    return host_reader.get(symbols, port, l, k_init, mask);
  }

  // See interface for documentation.
  span<cbf16_t> get(span<cbf16_t>                              symbols,
                    unsigned                                   port,
                    unsigned                                   l,
                    unsigned                                   k_init,
                    const bounded_bitset<MAX_NOF_SUBCARRIERS>& mask) const override
  {
    return host_reader.get(symbols, port, l, k_init, mask);
  }

  // See interface for documentation.
  void get(span<cf_t> symbols, unsigned port, unsigned l, unsigned k_init, unsigned stride = 1) const override
  {
    host_reader.get(symbols, port, l, k_init, stride);
  }

  // See interface for documentation.
  void get(span<cbf16_t> symbols, unsigned port, unsigned l, unsigned k_init) const override
  {
    host_reader.get(symbols, port, l, k_init);
  }

  // See interface for documentation.
  span<const cbf16_t> get_view(unsigned port, unsigned l) const override { return host_reader.get_view(port, l); }

  // See interface for documentation.
  bool supports_device_grid_reading() const override;

  // See interface for documentation.
  const void* get_device_grid_cbf16() const override { return d_grid_bf16; }

  // See interface for documentation.
  void* get_device_grid_ready_event() const override { return grid_ready_event; }

  // See interface for documentation.
  bool prepare_device_grid_reading(void* stream) const override;

private:
  const resource_grid_reader& host_reader;
  grid_allocation_mode        allocation_mode         = grid_allocation_mode::device;
  void*                       d_grid_bf16             = nullptr;
  void*                       grid_ready_event        = nullptr;
  bool                        grid_ready_recorded     = false;
  size_t                      device_grid_capacity    = 0;
  size_t                      device_grid_bytes       = 0;
  unsigned                    device_grid_nof_ports   = 0;
  unsigned                    device_grid_nof_symbols = 0;
  unsigned                    device_grid_nof_subc    = 0;
};

} // namespace ocudu
