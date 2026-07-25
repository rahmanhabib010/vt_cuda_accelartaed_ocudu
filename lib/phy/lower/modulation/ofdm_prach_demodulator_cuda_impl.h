// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/phy/lower/modulation/ofdm_prach_demodulator.h"
#include "ocudu/phy/lower/sampling_rate.h"
#include <memory>
#include <mutex>
#include <low_phy_prach_rx.h>

namespace ocudu {

/// CUDA-backed PRACH demodulator wrapper with CPU fallback.
class ofdm_prach_demodulator_cuda_impl : public ofdm_prach_demodulator
{
public:
  /// Creates a PRACH demodulator that tries the CUDA backend before using \c fallback_.
  ofdm_prach_demodulator_cuda_impl(std::unique_ptr<ofdm_prach_demodulator> fallback_,
                                         sampling_rate                           srate_,
                                         bool                                    force_gpu_path_);

  /// Releases the CUDA PRACH demodulator handle, when one was created.
  ~ofdm_prach_demodulator_cuda_impl() override;

  /// Demodulates floating-point PRACH samples, falling back to the wrapped demodulator when needed.
  void demodulate(prach_buffer& buffer, span<const cf_t> input, const configuration& config) override;

  /// Demodulates packed 16-bit complex PRACH samples directly on the CUDA path when supported.
  bool demodulate_ci16(prach_buffer&        buffer,
                       span<const ci16_t>   input,
                       float                input_scale,
                       const configuration& config) override;

private:
  /// Time-domain geometry needed by the CUDA PRACH kernel for one PRACH occasion.
  struct td_occasion_geometry;

  /// Runs CUDA PRACH demodulation for floating-point samples.
  bool demodulate_gpu(prach_buffer& buffer, span<const cf_t> input, const configuration& config);

  /// Runs CUDA PRACH demodulation for packed 16-bit complex samples.
  bool demodulate_gpu_ci16(prach_buffer&        buffer,
                           span<const ci16_t>   input,
                           float                input_scale,
                           const configuration& config);

  /// Builds CUDA PRACH kernel geometry for a single time-domain occasion.
  bool build_td_occasion_geometry(td_occasion_geometry& geometry,
                                  const configuration&  config,
                                  unsigned              i_td_occasion,
                                  subcarrier_spacing    pusch_scs) const;

  /// Creates or reconfigures the backend handle for \c cfg.
  bool ensure_handle(const ocudu_lowphy_prach_rx_config_t& cfg);

  /// CPU demodulator used when acceleration is unavailable or unsuitable.
  std::unique_ptr<ofdm_prach_demodulator> fallback;
  /// Lower-PHY sampling rate associated with the input stream.
  sampling_rate                           srate;
  /// True when failure to use the CUDA path is fatal.
  bool                                    force_gpu_path = false;
  /// True when a CUDA device was detected during construction.
  bool                                    gpu_available  = false;
  /// CUDA PRACH backend handle.
  ocudu_lowphy_prach_rx_handle_t*         handle         = nullptr;
  /// Serializes handle reconfiguration and destruction.
  std::mutex                              demodulator_mutex;
};

} // namespace ocudu
