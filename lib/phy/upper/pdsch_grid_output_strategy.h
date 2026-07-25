// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief PDSCH resource-grid output strategy abstraction.

#pragma once

#include "ocudu/ocudulog/logger.h"
#include "ocudu/phy/support/resource_grid_context.h"
#include "ocudu/phy/support/resource_grid_writer.h"
#include "ocudu/phy/support/shared_resource_grid.h"
#include <memory>

namespace ocudu {

/// Selects the resource-grid writer used by PDSCH and performs any required completion work before grid transmission.
class pdsch_grid_output_strategy
{
public:
  virtual ~pdsch_grid_output_strategy() = default;

  /// Configures the strategy for a new slot resource grid.
  virtual void configure(const resource_grid_context& context, shared_resource_grid& grid) = 0;

  /// Returns the writer that PDSCH should use for the current slot.
  virtual resource_grid_writer& get_pdsch_writer(shared_resource_grid& grid) = 0;

  /// Completes any pending grid materialization before the grid is sent to lower PHY.
  virtual void before_send_grid() = 0;
};

/// Creates the implementation-specific PDSCH grid output strategy.
std::unique_ptr<pdsch_grid_output_strategy> create_pdsch_grid_output_strategy(ocudulog::basic_logger& logger,
                                                                              bool use_device_grid_writer);

} // namespace ocudu
