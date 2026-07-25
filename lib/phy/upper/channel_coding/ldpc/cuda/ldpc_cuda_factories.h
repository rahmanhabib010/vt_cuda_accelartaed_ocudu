// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief Factory functions for CUDA accelerated LDPC components.

#pragma once

#include "ocudu/phy/upper/channel_coding/channel_coding_factories.h"
#include "ocudu/phy/upper/channel_modulation/channel_modulation_factories.h"
#include "ocudu/phy/upper/sequence_generators/sequence_generator_factories.h"
#include <memory>

namespace ocudu {

/// \brief Creates an LDPC decoder factory using CUDA acceleration.
///
/// The returned factory produces ldpc_decoder instances that use NVIDIA CUDA
/// for hardware-accelerated LDPC decoding.
///
/// \return Shared pointer to the CUDA LDPC decoder factory.
/// \note Requires CUDA-capable GPU and CUDA library.
std::shared_ptr<ldpc_decoder_factory> create_ldpc_decoder_factory_cuda();

/// \brief Checks if CUDA acceleration is available.
///
/// Verifies that:
/// - CUDA runtime is available.
/// - At least one CUDA-capable GPU is present.
/// - CUDA library can be initialized.
///
/// \return True if CUDA acceleration is available, false otherwise.
bool is_cuda_available();

/// \brief Creates a pseudo-random generator factory using CUDA acceleration.
///
/// The returned factory produces pseudo_random_generator instances that use NVIDIA CUDA
/// for hardware-accelerated LLR descrambling. Operations on small data sizes or non-LLR
/// operations fall back to a CPU-based implementation.
///
/// \param[in] fallback_factory Factory for creating fallback CPU-based generators.
/// \return Shared pointer to the CUDA pseudo-random generator factory.
/// \note Requires CUDA-capable GPU and CUDA library.
std::shared_ptr<pseudo_random_generator_factory>
create_pseudo_random_generator_factory_cuda(std::shared_ptr<pseudo_random_generator_factory> fallback_factory);

/// \brief Creates a CRC calculator factory using CUDA acceleration.
///
/// The returned factory produces crc_calculator instances that use NVIDIA CUDA
/// for hardware-accelerated CRC computation. For small data sizes where GPU overhead
/// exceeds the benefit, the calculator falls back to a CPU-based implementation.
///
/// \param[in] fallback_factory Factory for creating fallback CPU-based CRC calculators.
/// \return Shared pointer to the CUDA CRC calculator factory.
/// \note Requires CUDA-capable GPU and CUDA library.
std::shared_ptr<crc_calculator_factory>
create_crc_calculator_factory_cuda(std::shared_ptr<crc_calculator_factory> fallback_factory);

/// \brief Creates a demodulation mapper factory using CUDA acceleration.
///
/// The returned factory produces demodulation_mapper instances that use NVIDIA CUDA
/// for hardware-accelerated soft demodulation. The GPU implementation supports:
/// - Per-symbol noise variance for accurate LLR computation after MMSE equalization
/// - All 5G NR modulation schemes (BPSK, QPSK, 16QAM, 64QAM, 256QAM)
/// - Automatic fallback to CPU for small data or when GPU is unavailable
///
/// \param[in] fallback_factory Factory for creating fallback CPU-based demodulators.
/// \return Shared pointer to the CUDA demodulation mapper factory.
/// \note Requires CUDA-capable GPU and CUDA library.
std::shared_ptr<demodulation_mapper_factory>
create_demodulation_mapper_factory_cuda(std::shared_ptr<demodulation_mapper_factory> fallback_factory);

/// \brief Creates an LDPC encoder factory using CUDA acceleration.
///
/// The returned factory produces ldpc_encoder instances that use NVIDIA CUDA
/// for hardware-accelerated LDPC encoding (TX side).
///
/// \param[in] fallback_factory Factory for creating fallback CPU-based encoders.
/// \return Shared pointer to the CUDA LDPC encoder factory.
/// \note Requires CUDA-capable GPU and CUDA library.
std::shared_ptr<ldpc_encoder_factory>
create_ldpc_encoder_factory_cuda(std::shared_ptr<ldpc_encoder_factory> fallback_factory);

/// \brief Creates a modulation mapper factory using CUDA acceleration.
///
/// The returned factory produces modulation_mapper instances that use NVIDIA CUDA
/// for hardware-accelerated modulation (TX side).
///
/// \param[in] fallback_factory Factory for creating fallback CPU-based modulators.
/// \return Shared pointer to the CUDA modulation mapper factory.
/// \note Requires CUDA-capable GPU and CUDA library.
std::shared_ptr<modulation_mapper_factory>
create_modulation_mapper_factory_cuda(std::shared_ptr<modulation_mapper_factory> fallback_factory);

} // namespace ocudu
