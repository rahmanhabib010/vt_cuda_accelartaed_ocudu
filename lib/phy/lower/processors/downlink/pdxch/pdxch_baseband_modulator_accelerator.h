// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "ocudu/adt/span.h"
#include "ocudu/gateways/baseband/buffer/baseband_gateway_buffer_dynamic.h"
#include "ocudu/phy/lower/amplitude_controller/amplitude_controller_factories.h"
#include "ocudu/phy/lower/lower_phy_baseband_metrics.h"
#include "ocudu/phy/lower/sampling_rate.h"
#include "ocudu/phy/support/resource_grid_reader.h"
#include "ocudu/ran/cyclic_prefix.h"
#include "ocudu/ran/subcarrier_spacing.h"
#include <memory>

namespace ocudu {

/// Optional accelerator hook for PDxCH slot-to-baseband modulation.
///
/// Implementations own backend-specific streams, events and memory. The generic
/// lower-PHY path only observes whether an accelerated submission succeeded and
/// waits on an opaque completion through this interface.
class pdxch_baseband_modulator_accelerator
{
public:
  virtual ~pdxch_baseband_modulator_accelerator() = default;

  /// Enqueues accelerated slot baseband modulation.
  ///
  /// \param[out] output            Baseband output buffer for one slot or subslot.
  /// \param[in]  grid              Source downlink resource grid.
  /// \param[in]  i_symbol_sf_begin First OFDM symbol in the subframe represented by \c output.
  /// \param[in]  symbol_sizes_sf   Number of baseband samples per OFDM symbol in the subframe.
  /// \return True if the accelerated work was accepted, otherwise false.
  virtual bool enqueue(baseband_gateway_buffer_dynamic& output,
                       const resource_grid_reader&      grid,
                       unsigned                         i_symbol_sf_begin,
                       span<const unsigned>             symbol_sizes_sf) = 0;

  /// Waits for the last accepted accelerated submission to complete.
  virtual bool wait() = 0;

  /// Optional startup hook for preparing stable output buffers before realtime slots begin.
  virtual bool prepare_output_buffer(baseband_gateway_buffer_dynamic&) { return true; }

  /// Updates the carrier center frequency used by phase compensation.
  virtual void set_center_frequency(double) {}

  /// Collects lower-PHY baseband modulation metrics from the most recent submission.
  virtual lower_phy_baseband_metrics collect_metrics() const { return {}; }
};

/// Creates a CUDA-backed PDxCH baseband modulator accelerator.
std::unique_ptr<pdxch_baseband_modulator_accelerator>
create_pdxch_baseband_modulator_accelerator_cuda(subcarrier_spacing                          scs,
                                                       cyclic_prefix                               cp,
                                                       sampling_rate                               srate,
                                                       unsigned                                    bandwidth_rb,
                                                       double                                      center_freq_Hz,
                                                       unsigned                                    nof_ports,
                                                       const amplitude_controller_clipping_config& amplitude_config);

} // namespace ocudu
