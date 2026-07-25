// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include <optional>

namespace ocudu {

/// PUSCH demodulator interface to notify the demodulation statistics of a codeword.
class pusch_demodulator_notifier
{
public:
  /// Collects PUSCH demodulation statistics.
  struct demodulation_stats {
    /// Estimated Signal-to-Interference-plus-Noise Ratio (SINR) at the output of the equalizer.
    std::optional<float> sinr_dB;
    /// Measured EVM.
    std::optional<float> evm;
    /// Energy Per Resource Element (EPRE) in dB.
    std::optional<float> epre_dB;
    /// Reference Signal Received Power (RSRP) in dB.
    std::optional<float> rsrp_dB;
    /// Time Alignment in seconds.
    std::optional<float> time_alignment_s;
    /// Carrier Frequency Offset in Hz.
    std::optional<float> cfo_Hz;
  };

  /// Default destructor.
  virtual ~pusch_demodulator_notifier() = default;

  /// \brief Notifies intermediate PUSCH demodulator statistics.
  /// \param i_symbol OFDM symbol index within the slot.
  /// \param stats    OFDM symbol statistics.
  virtual void on_provisional_stats(unsigned i_symbol, const demodulation_stats& stats) = 0;

  /// Notifies the end of PUSCH processing and the final demodulator statistics.
  virtual void on_end_stats(const demodulation_stats& stats) = 0;
};

} // namespace ocudu
