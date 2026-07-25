// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief Runtime decision helpers for implementation-specific PUSCH acceleration paths.

#pragma once

#include "../../phy_acceleration_runtime_options.h"

namespace ocudu {

struct pusch_acceleration_thresholds {
  unsigned min_prb        = 0;
  unsigned min_ulsch_bits = 0;
};

inline unsigned pusch_acceleration_read_min_rb()
{
  return phy_acceleration_env_unsigned("OCUDU_PUSCH_ACCELERATION_MIN_RB", 0);
}

inline pusch_acceleration_thresholds pusch_acceleration_read_thresholds()
{
  return {.min_prb        = pusch_acceleration_read_min_rb(),
          .min_ulsch_bits = phy_acceleration_env_unsigned("OCUDU_PUSCH_ACCELERATION_MIN_ULSCH_BITS", 0)};
}

inline bool pusch_acceleration_prefers_cpu_for_sch_grant(bool                                 has_sch,
                                                         unsigned                             nof_rb,
                                                         unsigned                             nof_ulsch_bits,
                                                         const pusch_acceleration_thresholds& thresholds)
{
  if (!has_sch) {
    return false;
  }

  const bool below_prb_threshold  = (thresholds.min_prb != 0) && (nof_rb < thresholds.min_prb);
  const bool below_bits_threshold = (thresholds.min_ulsch_bits != 0) && (nof_ulsch_bits < thresholds.min_ulsch_bits);
  return below_prb_threshold || below_bits_threshold;
}

inline bool pusch_acceleration_prefers_cpu_for_demodulator_grant(unsigned nof_rb, unsigned min_gpu_prb)
{
  return (min_gpu_prb != 0) && (nof_rb < min_gpu_prb);
}

inline bool pusch_acceleration_device_uci_enabled()
{
  return phy_acceleration_env_flag_enabled("OCUDU_PUSCH_ENABLE_ACCELERATED_UCI");
}

} // namespace ocudu
