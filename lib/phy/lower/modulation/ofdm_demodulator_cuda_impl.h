// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ofdm_demodulator_impl.h"
#include "phase_compensation_lut.h"
#include "ocudu/phy/lower/modulation/ofdm_demodulator.h"
#include <atomic>
#include <low_phy_puxch_rx.h>
#include <memory>
#include <mutex>

namespace ocudu {

class ofdm_symbol_demodulator_cuda_impl : public ofdm_symbol_demodulator
{
public:
  ofdm_symbol_demodulator_cuda_impl(std::unique_ptr<ofdm_symbol_demodulator> fallback_,
                                          const ofdm_demodulator_configuration&    ofdm_config,
                                          bool                                     force_gpu_path_);

  ~ofdm_symbol_demodulator_cuda_impl() override;

  unsigned get_symbol_size(unsigned symbol_index) const override
  {
    return cp.get_length(symbol_index, scs).to_samples(sampling_rate_Hz) + dft_size;
  }

  void set_center_frequency(double center_frequency_Hz) override
  {
    next_center_freq_Hz.store(center_frequency_Hz, std::memory_order_relaxed);
    fallback->set_center_frequency(center_frequency_Hz);
  }

  void
  demodulate(resource_grid_writer& grid, span<const cf_t> input, unsigned port_index, unsigned symbol_index) override
  {
    fallback->demodulate(grid, input, port_index, symbol_index);
  }

  bool demodulate_ci16(resource_grid_writer& grid,
                       span<const ci16_t>    input,
                       float                 input_scale,
                       unsigned              port_index,
                       unsigned              symbol_index) override;

  bool demodulate_ci16_ports(resource_grid_writer&     grid,
                             span<const ci16_t* const> inputs,
                             unsigned                  nof_samples,
                             float                     input_scale,
                             span<const unsigned>      port_indices,
                             unsigned                  symbol_index) override;

private:
  bool demodulate_gpu_ci16_ports(resource_grid_writer&     grid,
                                 span<const ci16_t* const> inputs,
                                 unsigned                  nof_samples,
                                 float                     input_scale,
                                 span<const unsigned>      port_indices,
                                 unsigned                  symbol_index);

  bool ensure_handle(const ocudu_lowphy_puxch_rx_config_t& cfg);
  void warmup_handle();

  std::unique_ptr<ofdm_symbol_demodulator> fallback;
  unsigned                                 dft_size;
  unsigned                                 rg_size;
  cyclic_prefix                            cp;
  unsigned                                 nof_samples_window_offset;
  subcarrier_spacing                       scs;
  unsigned                                 sampling_rate_Hz;
  float                                    scale;
  phase_compensation_lut                   phase_compensation_table;
  std::atomic<double>                      next_center_freq_Hz;
  double                                   current_center_freq_Hz;
  bool                                     force_gpu_path = false;
  bool                                     gpu_available  = false;
  ocudu_lowphy_puxch_rx_handle_t*          handle         = nullptr;
  std::mutex                               demodulator_mutex;
};

} // namespace ocudu
