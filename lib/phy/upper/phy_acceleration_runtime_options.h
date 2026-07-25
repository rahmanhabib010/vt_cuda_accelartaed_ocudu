// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief Runtime overrides for implementation-specific PHY acceleration paths.

#pragma once

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace ocudu {

/// Returns true if \c value is one of the accepted false-like environment flag values.
inline bool phy_acceleration_env_flag_value_is_false(const char* value)
{
  return (std::strcmp(value, "0") == 0) || (std::strcmp(value, "false") == 0) || (std::strcmp(value, "False") == 0) ||
         (std::strcmp(value, "FALSE") == 0) || (std::strcmp(value, "off") == 0) || (std::strcmp(value, "OFF") == 0) ||
         (std::strcmp(value, "no") == 0) || (std::strcmp(value, "NO") == 0);
}

/// Returns true if \c value is one of the accepted true-like environment flag values.
inline bool phy_acceleration_env_flag_value_is_true(const char* value)
{
  return (std::strcmp(value, "1") == 0) || (std::strcmp(value, "true") == 0) || (std::strcmp(value, "True") == 0) ||
         (std::strcmp(value, "TRUE") == 0) || (std::strcmp(value, "on") == 0) || (std::strcmp(value, "ON") == 0) ||
         (std::strcmp(value, "yes") == 0) || (std::strcmp(value, "YES") == 0);
}

/// Reads a boolean environment flag with the accepted true/false string set.
inline bool phy_acceleration_env_flag_enabled(const char* name, bool default_value = false)
{
  const char* value = std::getenv(name);
  if ((value == nullptr) || (*value == '\0')) {
    return default_value;
  }

  if (phy_acceleration_env_flag_value_is_true(value)) {
    return true;
  }
  if (phy_acceleration_env_flag_value_is_false(value)) {
    return false;
  }

  return default_value;
}

/// Reads an unsigned integer environment override, returning \c default_value for unset or malformed values.
inline unsigned phy_acceleration_env_unsigned(const char* name, unsigned default_value)
{
  const char* value = std::getenv(name);
  if ((value == nullptr) || (*value == '\0')) {
    return default_value;
  }
  if (*value == '-') {
    return default_value;
  }

  char* end_ptr        = nullptr;
  errno                = 0;
  unsigned long parsed = std::strtoul(value, &end_ptr, 10);
  if ((end_ptr == value) || (*end_ptr != '\0') || (errno == ERANGE) ||
      (parsed > std::numeric_limits<unsigned>::max())) {
    return default_value;
  }

  return static_cast<unsigned>(parsed);
}

/// Returns true when a grid allocation mode string requests managed memory.
inline bool phy_acceleration_env_mode_is_managed(const char* value)
{
  return (value != nullptr) && (std::strcmp(value, "managed") == 0);
}

/// Returns the CUDA-visible grid mode, preferring an optional direction-specific override.
inline const char* phy_acceleration_cuda_visible_grid_mode(const char* direction_env_name = nullptr)
{
  const char* mode = nullptr;
  if (direction_env_name != nullptr) {
    mode = std::getenv(direction_env_name);
  }
  if (mode == nullptr) {
    mode = std::getenv("OCUDU_CUDA_VISIBLE_GRID");
  }
  return mode;
}

/// Returns true when CUDA-visible resource grids should use managed memory.
inline bool phy_acceleration_cuda_visible_grid_managed_requested(const char* direction_env_name = nullptr)
{
  return phy_acceleration_env_mode_is_managed(phy_acceleration_cuda_visible_grid_mode(direction_env_name));
}

/// Returns true when the PDSCH auto-enable override has been configured.
inline bool phy_acceleration_pdsch_auto_enable_discrete_configured()
{
  const char* value = std::getenv("OCUDU_PDSCH_AUTO_ENABLE_DISCRETE");
  return (value != nullptr) && (*value != '\0');
}

/// Returns true when PDSCH auto-enable for discrete GPUs is requested.
inline bool phy_acceleration_pdsch_auto_enable_discrete_requested()
{
  return phy_acceleration_env_flag_enabled("OCUDU_PDSCH_AUTO_ENABLE_DISCRETE");
}

} // namespace ocudu
