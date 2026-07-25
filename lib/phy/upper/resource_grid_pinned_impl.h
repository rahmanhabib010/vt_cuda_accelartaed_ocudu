// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief Resource grid with CUDA-pinned backing memory for zero-copy GPU DMA.

#pragma once

#include "ocudu/phy/support/resource_grid.h"
#include "ocudu/phy/support/support_factories.h"
#include <cuda_runtime.h>
#include <memory>

namespace ocudu {

/// \brief Resource grid factory that pins the grid's backing memory via cudaHostRegister.
///
/// On GH200, this allows the GPU to DMA directly from the grid tensor without an intermediate
/// staging memcpy. The grid is created normally via the base factory, then the backing memory
/// is page-locked for GPU access. The one-time registration cost (~0.5ms) is negligible since
/// grids are created once at startup.
class resource_grid_pinned_factory : public resource_grid_factory
{
public:
  std::unique_ptr<resource_grid> create(unsigned nof_ports, unsigned nof_symbols, unsigned nof_subc) override
  {
    // Create a standard resource grid via the base factory.
    auto grid = base_factory_->create(nof_ports, nof_symbols, nof_subc);
    if (!grid) {
      return nullptr;
    }

    // Get a pointer to the start of the contiguous tensor data via the reader interface.
    // Port 0, symbol 0 is the base of the tensor's backing store.
    const auto&              reader    = grid->get_reader();
    span<const cbf16_t>      base_view = reader.get_view(0, 0);
    const void*              base_ptr  = base_view.data();
    size_t                   total_bytes = nof_ports * nof_symbols * nof_subc * sizeof(cbf16_t);
    if (base_ptr == nullptr || total_bytes == 0) {
      return grid;
    }

    // The direct-DMA fast path assumes the base factory stores the full grid as
    // port-contiguous [subcarrier, symbol, port] memory. Validate that once at
    // startup before registering the whole backing range.
    for (unsigned port = 0; port != nof_ports; ++port) {
      for (unsigned symbol = 0; symbol != nof_symbols; ++symbol) {
        span<const cbf16_t> view = reader.get_view(port, symbol);
        if (view.data() != static_cast<const cbf16_t*>(base_ptr) +
                               (static_cast<size_t>(port) * nof_symbols + symbol) * nof_subc) {
          return grid;
        }
      }
    }

    // Pin the memory for GPU DMA access. This page-locks the existing heap allocation
    // so cudaMemcpyAsync can DMA directly from it without internal staging.
    cudaError_t err = cudaHostRegister(const_cast<void*>(base_ptr), total_bytes, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
      // Registration failed (e.g., no CUDA context). Grid still works normally — just won't
      // benefit from direct DMA. This is not fatal.
      return grid;
    }

    // Wrap the grid to ensure we unregister on destruction.
    return std::make_unique<pinned_resource_grid>(std::move(grid), const_cast<void*>(base_ptr));
  }

private:
  std::shared_ptr<resource_grid_factory> base_factory_ = create_resource_grid_factory();

  /// Wrapper that unregisters pinned memory on destruction.
  class pinned_resource_grid : public resource_grid
  {
  public:
    pinned_resource_grid(std::unique_ptr<resource_grid> inner, void* pinned_ptr)
      : inner_(std::move(inner)), pinned_ptr_(pinned_ptr) {}

    ~pinned_resource_grid() override
    {
      if (pinned_ptr_) {
        cudaHostUnregister(pinned_ptr_);
      }
    }

    // Forward all interface methods to the inner grid.
    void                       set_all_zero() override { inner_->set_all_zero(); }
    resource_grid_writer&      get_writer() override { return inner_->get_writer(); }
    const resource_grid_reader& get_reader() const override { return inner_->get_reader(); }

  private:
    std::unique_ptr<resource_grid> inner_;
    void*                          pinned_ptr_;
  };
};

} // namespace ocudu
