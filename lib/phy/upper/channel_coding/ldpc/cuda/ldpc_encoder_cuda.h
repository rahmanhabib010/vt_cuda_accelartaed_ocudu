// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief LDPC encoder using CUDA CUDA acceleration.

#pragma once

#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_encoder.h"
#include <memory>

namespace ocudu {

/// \brief Creates an LDPC encoder using CUDA CUDA acceleration.
///
/// The returned encoder uses NVIDIA CUDA for hardware-accelerated LDPC encoding.
/// It automatically falls back to CPU when GPU is unavailable or for configurations
/// that are not well-suited for GPU processing.
///
/// \param[in] fallback Fallback CPU-based LDPC encoder for unsupported cases.
/// \return Unique pointer to the CUDA LDPC encoder.
std::unique_ptr<ldpc_encoder> create_ldpc_encoder_cuda(std::unique_ptr<ldpc_encoder> fallback);

/// \brief Check if GPU LDPC encoder is available.
/// \return True if GPU encoding is available, false otherwise.
bool is_ldpc_encoder_gpu_available();

} // namespace ocudu
