// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "ocudu/phy/lower/amplitude_controller/amplitude_controller_factories.h"
#include "ocudu/phy/lower/processors/downlink/pdxch/pdxch_processor_factories.h"

namespace ocudu {

/// Returns true when the CUDA lower-PHY TX accelerator can be instantiated.
bool is_pdxch_processor_factory_cuda_available();

/// Creates a PDxCH processor factory with an optional CUDA full-slot TX accelerator.
///
/// This is intentionally private to the lower-PHY implementation. Public factory
/// headers remain backend-neutral; CUDA-specific details are selected by the
/// split-8 SDR factory when CUDA is available.
std::shared_ptr<pdxch_processor_factory>
create_pdxch_processor_factory_cuda(std::shared_ptr<ofdm_modulator_factory>       ofdm_mod_factory,
                                          std::shared_ptr<amplitude_controller_factory> amplitude_control_factory,
                                          const amplitude_controller_clipping_config&   amplitude_config);

} // namespace ocudu
