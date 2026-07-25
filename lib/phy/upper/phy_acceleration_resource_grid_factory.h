// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief Resource-grid factory selection for implementation-specific PHY acceleration paths.

#pragma once

#include "ocudu/phy/support/support_factories.h"
#include <memory>

namespace ocudu {

/// Direction-specific resource-grid usage.
enum class phy_acceleration_resource_grid_direction { downlink, uplink };

/// Returns an implementation-specific resource-grid factory when acceleration is enabled and available.
///
/// When acceleration is disabled, or when the build has no accelerated implementation, the fallback factory is
/// returned.
std::shared_ptr<resource_grid_factory>
create_phy_acceleration_resource_grid_factory(std::shared_ptr<resource_grid_factory>   fallback_factory,
                                              phy_acceleration_resource_grid_direction direction,
                                              bool                                     enable_accelerated_grid);

/// Returns true when the active CUDA device is known to be discrete. Unknown devices return false.
bool phy_acceleration_cuda_current_device_is_discrete();

} // namespace ocudu
