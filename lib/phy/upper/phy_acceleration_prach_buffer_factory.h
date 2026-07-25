// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief PRACH-buffer factory selection for implementation-specific PHY acceleration paths.

#pragma once

#include "ocudu/phy/support/prach_buffer.h"
#include <memory>

namespace ocudu {

/// Returns a PRACH buffer suitable for accelerated uplink processing when requested and available.
std::unique_ptr<prach_buffer> create_phy_acceleration_prach_buffer_long(unsigned max_nof_ports,
                                                                        unsigned max_nof_fd_occasions,
                                                                        bool     enable_accelerated_buffer);

/// Returns a PRACH buffer suitable for accelerated uplink processing when requested and available.
std::unique_ptr<prach_buffer> create_phy_acceleration_prach_buffer_short(unsigned max_nof_ports,
                                                                         unsigned max_nof_td_occasions,
                                                                         unsigned max_nof_fd_occasions,
                                                                         bool     enable_accelerated_buffer);

} // namespace ocudu
