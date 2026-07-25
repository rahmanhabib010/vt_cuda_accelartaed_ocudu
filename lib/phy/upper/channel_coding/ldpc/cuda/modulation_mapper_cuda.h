// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief Modulation mapper using CUDA CUDA acceleration (TX side).

#pragma once

#include "ocudu/phy/upper/channel_modulation/modulation_mapper.h"
#include <memory>

namespace ocudu {

/// \brief Creates a modulation mapper using CUDA CUDA acceleration.
///
/// The returned mapper uses NVIDIA CUDA for hardware-accelerated modulation.
/// It automatically falls back to CPU when GPU is unavailable or for small blocks.
///
/// \param[in] fallback Fallback CPU-based modulation mapper.
/// \return Unique pointer to the CUDA modulation mapper.
std::unique_ptr<modulation_mapper> create_modulation_mapper_cuda(std::unique_ptr<modulation_mapper> fallback);

} // namespace ocudu
