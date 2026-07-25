// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ofdm_prach_demodulator_cuda_impl.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/ran/prach/prach_frequency_mapping.h"
#include "ocudu/ran/prach/prach_preamble_information.h"
#include "ocudu/ran/resource_block.h"
#include "ocudu/support/error_handling.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

using namespace ocudu;

namespace {

bool env_flag_disabled(const char* name)
{
  const char* value = std::getenv(name);
  return value &&
         ((std::strcmp(value, "0") == 0) || (std::strcmp(value, "false") == 0) || (std::strcmp(value, "off") == 0) ||
          (std::strcmp(value, "no") == 0) || (std::strcmp(value, "disabled") == 0));
}

bool lowphy_prach_demodulation_disabled_by_env()
{
  return env_flag_disabled("OCUDU_LOWPHY_PRACH_DEMODULATION_ACCELERATION") ||
         env_flag_disabled("OCUDU_LOWPHY_PRACH_ACCELERATION");
}

std::atomic<bool> logged_gpu_path_selected{false};

} // namespace

/// Time-domain geometry needed by the CUDA PRACH kernel for one PRACH occasion.
struct ofdm_prach_demodulator_cuda_impl::td_occasion_geometry {
  /// Preamble timing and transform parameters for the configured PRACH format.
  prach_preamble_information                                   preamble_info;
  /// Start time of this PRACH time-domain occasion.
  phy_time_unit                                                t_occasion_start;
  /// Sample offset of this occasion within the input buffer.
  unsigned                                                     sample_offset        = 0;
  /// Number of input samples consumed by this occasion.
  unsigned                                                     nof_samples          = 0;
  /// DFT size used by this PRACH occasion.
  unsigned                                                     dft_size             = 0;
  /// Cyclic prefix length in samples.
  unsigned                                                     cyclic_prefix_length = 0;
  /// Frequency-domain PRACH grid size in subcarriers.
  unsigned                                                     prach_grid_size      = 0;
  /// Starting PRACH subcarrier for each frequency-domain occasion.
  std::array<unsigned, OCUDU_LOWPHY_PRACH_RX_MAX_FD_OCCASIONS> k_start              = {};
};

ofdm_prach_demodulator_cuda_impl::ofdm_prach_demodulator_cuda_impl(
    std::unique_ptr<ofdm_prach_demodulator> fallback_,
    sampling_rate                           srate_,
    bool                                    force_gpu_path_) :
  fallback(std::move(fallback_)), srate(srate_), force_gpu_path(force_gpu_path_)
{
  ocudu_assert(fallback, "Invalid fallback PRACH demodulator.");

  int device_count = 0;
  if ((cudaGetDeviceCount(&device_count) != cudaSuccess) || (device_count == 0)) {
    cudaGetLastError();
    return;
  }
  gpu_available = true;
}

ofdm_prach_demodulator_cuda_impl::~ofdm_prach_demodulator_cuda_impl()
{
  std::lock_guard<std::mutex> lock(demodulator_mutex);
  if (handle != nullptr) {
    ocudu_lowphy_prach_rx_destroy(handle);
    handle = nullptr;
  }
}

void ofdm_prach_demodulator_cuda_impl::demodulate(prach_buffer&        buffer,
                                                        span<const cf_t>     input,
                                                        const configuration& config)
{
  if (!gpu_available || lowphy_prach_demodulation_disabled_by_env() || !buffer.supports_device_prach_buffer_mapping()) {
    report_fatal_error_if_not(!force_gpu_path, "Accelerated lower-PHY PRACH demodulation requested but unavailable.");
    fallback->demodulate(buffer, input, config);
    return;
  }

  if (demodulate_gpu(buffer, input, config)) {
    return;
  }

  report_fatal_error_if_not(!force_gpu_path, "Accelerated lower-PHY PRACH demodulation failed.");
  fallback->demodulate(buffer, input, config);
}

bool ofdm_prach_demodulator_cuda_impl::demodulate_ci16(prach_buffer&        buffer,
                                                             span<const ci16_t>   input,
                                                             float                input_scale,
                                                             const configuration& config)
{
  if (!gpu_available || lowphy_prach_demodulation_disabled_by_env() || !buffer.supports_device_prach_buffer_mapping()) {
    report_fatal_error_if_not(!force_gpu_path, "Accelerated lower-PHY PRACH demodulation requested but unavailable.");
    return false;
  }

  if (demodulate_gpu_ci16(buffer, input, input_scale, config)) {
    return true;
  }

  report_fatal_error_if_not(!force_gpu_path, "Accelerated lower-PHY PRACH demodulation failed.");
  return false;
}

bool ofdm_prach_demodulator_cuda_impl::demodulate_gpu(prach_buffer&        buffer,
                                                            span<const cf_t>     input,
                                                            const configuration& config)
{
  std::lock_guard<std::mutex> lock(demodulator_mutex);

  subcarrier_spacing pusch_scs = to_subcarrier_spacing(config.slot.numerology());

  td_occasion_geometry first_geometry;
  if (!build_td_occasion_geometry(first_geometry, config, 0, pusch_scs) ||
      input.size() < first_geometry.sample_offset + first_geometry.nof_samples) {
    return false;
  }

  ocudu_lowphy_prach_rx_config_t first_cfg = {};
  first_cfg.dft_size                       = static_cast<int>(first_geometry.dft_size);
  first_cfg.sequence_length                = static_cast<int>(first_geometry.preamble_info.sequence_length);
  first_cfg.nof_symbols                    = static_cast<int>(first_geometry.preamble_info.nof_symbols);
  first_cfg.nof_fd_occasions               = static_cast<int>(config.nof_fd_occasions);
  first_cfg.input_nof_samples              = static_cast<int>(first_geometry.nof_samples);
  first_cfg.cyclic_prefix_length           = static_cast<int>(first_geometry.cyclic_prefix_length);
  first_cfg.prach_grid_size                = static_cast<int>(first_geometry.prach_grid_size);
  first_cfg.output_symbol_stride           = (first_geometry.preamble_info.nof_symbols > 1)
                                                 ? static_cast<int>(buffer.get_device_prach_symbol_offset(config.port, 0, 0, 1) -
                                                          buffer.get_device_prach_symbol_offset(config.port, 0, 0, 0))
                                                 : static_cast<int>(first_geometry.preamble_info.sequence_length);
  first_cfg.output_fd_stride               = (config.nof_fd_occasions > 1)
                                                 ? static_cast<int>(buffer.get_device_prach_symbol_offset(config.port, 0, 1, 0) -
                                                      buffer.get_device_prach_symbol_offset(config.port, 0, 0, 0))
                                                 : static_cast<int>(buffer.get_max_nof_symbols() * buffer.get_sequence_length());
  first_cfg.input_is_device                = 0;
  first_cfg.dft_scale                      = 1.0F / std::sqrt(static_cast<float>(first_geometry.dft_size));
  for (unsigned fd = 0; fd != config.nof_fd_occasions; ++fd) {
    first_cfg.k_start[fd] = static_cast<int>(first_geometry.k_start[fd]);
  }

  if (!ensure_handle(first_cfg)) {
    return false;
  }

  void* stream = ocudu_lowphy_prach_rx_get_stream(handle);
  if (!buffer.prepare_device_prach_buffer_mapping(stream)) {
    return false;
  }

  auto cancel_gpu_mapping = [&buffer]() { (void)buffer.cancel_device_prach_buffer_mapping(); };

  for (unsigned i_td_occasion = 0; i_td_occasion != config.nof_td_occasions; ++i_td_occasion) {
    td_occasion_geometry geometry;
    if (!build_td_occasion_geometry(geometry, config, i_td_occasion, pusch_scs) ||
        input.size() < geometry.sample_offset + geometry.nof_samples) {
      cancel_gpu_mapping();
      return false;
    }

    ocudu_lowphy_prach_rx_config_t cuda_cfg = first_cfg;
    cuda_cfg.dft_size                       = static_cast<int>(geometry.dft_size);
    cuda_cfg.sequence_length                = static_cast<int>(geometry.preamble_info.sequence_length);
    cuda_cfg.nof_symbols                    = static_cast<int>(geometry.preamble_info.nof_symbols);
    cuda_cfg.input_nof_samples              = static_cast<int>(geometry.nof_samples);
    cuda_cfg.cyclic_prefix_length           = static_cast<int>(geometry.cyclic_prefix_length);
    cuda_cfg.prach_grid_size                = static_cast<int>(geometry.prach_grid_size);
    cuda_cfg.output_symbol_stride =
        (geometry.preamble_info.nof_symbols > 1)
            ? static_cast<int>(buffer.get_device_prach_symbol_offset(config.port, i_td_occasion, 0, 1) -
                               buffer.get_device_prach_symbol_offset(config.port, i_td_occasion, 0, 0))
            : static_cast<int>(geometry.preamble_info.sequence_length);
    cuda_cfg.output_fd_stride =
        (config.nof_fd_occasions > 1)
            ? static_cast<int>(buffer.get_device_prach_symbol_offset(config.port, i_td_occasion, 1, 0) -
                               buffer.get_device_prach_symbol_offset(config.port, i_td_occasion, 0, 0))
            : static_cast<int>(buffer.get_max_nof_symbols() * buffer.get_sequence_length());
    cuda_cfg.dft_scale = 1.0F / std::sqrt(static_cast<float>(geometry.dft_size));
    for (unsigned fd = 0; fd != config.nof_fd_occasions; ++fd) {
      cuda_cfg.k_start[fd] = static_cast<int>(geometry.k_start[fd]);
    }

    if (!ensure_handle(cuda_cfg)) {
      cancel_gpu_mapping();
      return false;
    }

    const cf_t* input_occasion = input.subspan(geometry.sample_offset, geometry.nof_samples).data();
    auto*       output_base    = static_cast<uint32_t*>(buffer.get_device_prach_buffer_cbf16()) +
                        buffer.get_device_prach_symbol_offset(config.port, i_td_occasion, 0, 0);
    if (ocudu_lowphy_prach_rx_process(handle, input_occasion, output_base, stream) == 0) {
      (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
      cancel_gpu_mapping();
      return false;
    }
  }

  if (!buffer.on_device_prach_buffer_mapping_enqueued(stream)) {
    (void)ocudu_lowphy_prach_rx_synchronize(handle);
    cancel_gpu_mapping();
    return false;
  }

  if (!logged_gpu_path_selected.exchange(true)) {
    ocudulog::fetch_basic_logger("PHY").info(
        "Lower-PHY PRACH demodulation GPU path selected: direct CUDA-visible PRACH buffer writer.");
  }

  return true;
}

bool ofdm_prach_demodulator_cuda_impl::demodulate_gpu_ci16(prach_buffer&        buffer,
                                                                 span<const ci16_t>   input,
                                                                 float                input_scale,
                                                                 const configuration& config)
{
  std::lock_guard<std::mutex> lock(demodulator_mutex);

  subcarrier_spacing pusch_scs = to_subcarrier_spacing(config.slot.numerology());

  td_occasion_geometry first_geometry;
  if (!build_td_occasion_geometry(first_geometry, config, 0, pusch_scs) ||
      input.size() < first_geometry.sample_offset + first_geometry.nof_samples) {
    return false;
  }

  ocudu_lowphy_prach_rx_config_t first_cfg = {};
  first_cfg.dft_size                       = static_cast<int>(first_geometry.dft_size);
  first_cfg.sequence_length                = static_cast<int>(first_geometry.preamble_info.sequence_length);
  first_cfg.nof_symbols                    = static_cast<int>(first_geometry.preamble_info.nof_symbols);
  first_cfg.nof_fd_occasions               = static_cast<int>(config.nof_fd_occasions);
  first_cfg.input_nof_samples              = static_cast<int>(first_geometry.nof_samples);
  first_cfg.cyclic_prefix_length           = static_cast<int>(first_geometry.cyclic_prefix_length);
  first_cfg.prach_grid_size                = static_cast<int>(first_geometry.prach_grid_size);
  first_cfg.output_symbol_stride           = (first_geometry.preamble_info.nof_symbols > 1)
                                                 ? static_cast<int>(buffer.get_device_prach_symbol_offset(config.port, 0, 0, 1) -
                                                          buffer.get_device_prach_symbol_offset(config.port, 0, 0, 0))
                                                 : static_cast<int>(first_geometry.preamble_info.sequence_length);
  first_cfg.output_fd_stride               = (config.nof_fd_occasions > 1)
                                                 ? static_cast<int>(buffer.get_device_prach_symbol_offset(config.port, 0, 1, 0) -
                                                      buffer.get_device_prach_symbol_offset(config.port, 0, 0, 0))
                                                 : static_cast<int>(buffer.get_max_nof_symbols() * buffer.get_sequence_length());
  first_cfg.input_is_device                = 0;
  first_cfg.dft_scale                      = 1.0F / std::sqrt(static_cast<float>(first_geometry.dft_size));
  for (unsigned fd = 0; fd != config.nof_fd_occasions; ++fd) {
    first_cfg.k_start[fd] = static_cast<int>(first_geometry.k_start[fd]);
  }

  if (!ensure_handle(first_cfg)) {
    return false;
  }

  void* stream = ocudu_lowphy_prach_rx_get_stream(handle);
  if (!buffer.prepare_device_prach_buffer_mapping(stream)) {
    return false;
  }

  auto cancel_gpu_mapping = [&buffer]() { (void)buffer.cancel_device_prach_buffer_mapping(); };

  for (unsigned i_td_occasion = 0; i_td_occasion != config.nof_td_occasions; ++i_td_occasion) {
    td_occasion_geometry geometry;
    if (!build_td_occasion_geometry(geometry, config, i_td_occasion, pusch_scs) ||
        input.size() < geometry.sample_offset + geometry.nof_samples) {
      cancel_gpu_mapping();
      return false;
    }

    ocudu_lowphy_prach_rx_config_t cuda_cfg = first_cfg;
    cuda_cfg.dft_size                       = static_cast<int>(geometry.dft_size);
    cuda_cfg.sequence_length                = static_cast<int>(geometry.preamble_info.sequence_length);
    cuda_cfg.nof_symbols                    = static_cast<int>(geometry.preamble_info.nof_symbols);
    cuda_cfg.input_nof_samples              = static_cast<int>(geometry.nof_samples);
    cuda_cfg.cyclic_prefix_length           = static_cast<int>(geometry.cyclic_prefix_length);
    cuda_cfg.prach_grid_size                = static_cast<int>(geometry.prach_grid_size);
    cuda_cfg.output_symbol_stride =
        (geometry.preamble_info.nof_symbols > 1)
            ? static_cast<int>(buffer.get_device_prach_symbol_offset(config.port, i_td_occasion, 0, 1) -
                               buffer.get_device_prach_symbol_offset(config.port, i_td_occasion, 0, 0))
            : static_cast<int>(geometry.preamble_info.sequence_length);
    cuda_cfg.output_fd_stride =
        (config.nof_fd_occasions > 1)
            ? static_cast<int>(buffer.get_device_prach_symbol_offset(config.port, i_td_occasion, 1, 0) -
                               buffer.get_device_prach_symbol_offset(config.port, i_td_occasion, 0, 0))
            : static_cast<int>(buffer.get_max_nof_symbols() * buffer.get_sequence_length());
    cuda_cfg.dft_scale = 1.0F / std::sqrt(static_cast<float>(geometry.dft_size));
    for (unsigned fd = 0; fd != config.nof_fd_occasions; ++fd) {
      cuda_cfg.k_start[fd] = static_cast<int>(geometry.k_start[fd]);
    }

    if (!ensure_handle(cuda_cfg)) {
      cancel_gpu_mapping();
      return false;
    }

    const ci16_t* input_occasion = input.subspan(geometry.sample_offset, geometry.nof_samples).data();
    auto*         output_base    = static_cast<uint32_t*>(buffer.get_device_prach_buffer_cbf16()) +
                        buffer.get_device_prach_symbol_offset(config.port, i_td_occasion, 0, 0);
    if (ocudu_lowphy_prach_rx_process_ci16(handle, input_occasion, input_scale, output_base, stream) == 0) {
      (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
      cancel_gpu_mapping();
      return false;
    }
  }

  if (!buffer.on_device_prach_buffer_mapping_enqueued(stream)) {
    (void)ocudu_lowphy_prach_rx_synchronize(handle);
    cancel_gpu_mapping();
    return false;
  }

  if (!logged_gpu_path_selected.exchange(true)) {
    ocudulog::fetch_basic_logger("PHY").info(
        "Lower-PHY PRACH demodulation GPU path selected: direct CUDA-visible PRACH buffer writer.");
  }

  return true;
}

bool ofdm_prach_demodulator_cuda_impl::build_td_occasion_geometry(td_occasion_geometry& geometry,
                                                                        const configuration&  config,
                                                                        unsigned              i_td_occasion,
                                                                        subcarrier_spacing    pusch_scs) const
{
  static constexpr phy_time_unit sixteen_kappa         = phy_time_unit::from_units_of_kappa(16);
  static constexpr unsigned      nof_ofdm_symbols_slot = 14;

  unsigned      pusch_scs_Hz          = scs_to_khz(pusch_scs) * 1000;
  phy_time_unit pusch_symbol_duration = phy_time_unit::from_units_of_kappa((144U + 2048U) >> config.slot.numerology());

  bool is_last_occasion = (i_td_occasion == (config.nof_td_occasions - 1));
  geometry.preamble_info =
      is_long_preamble(config.format)
          ? get_prach_preamble_long_info(config.format)
          : get_prach_preamble_short_info(config.format, to_ra_subcarrier_spacing(pusch_scs), is_last_occasion);

  unsigned      t_occasion_start_symbol = config.start_symbol + get_preamble_duration(config.format) * i_td_occasion;
  phy_time_unit t_occasion_start        = pusch_symbol_duration * t_occasion_start_symbol;
  phy_time_unit t_slot_start = pusch_symbol_duration * config.slot.subframe_slot_index() * nof_ofdm_symbols_slot;
  phy_time_unit t_ra_start   = t_occasion_start + t_slot_start;

  if ((geometry.preamble_info.scs == prach_subcarrier_spacing::kHz1_25) ||
      (geometry.preamble_info.scs == prach_subcarrier_spacing::kHz5) ||
      (geometry.preamble_info.scs == prach_subcarrier_spacing::kHz15) ||
      (geometry.preamble_info.scs == prach_subcarrier_spacing::kHz30)) {
    if (t_occasion_start > phy_time_unit::from_seconds(0.0)) {
      t_occasion_start += sixteen_kappa;
    }
    if (t_occasion_start > phy_time_unit::from_seconds(0.5e-3)) {
      t_occasion_start += sixteen_kappa;
    }
  }

  phy_time_unit t_ra_end = t_ra_start + geometry.preamble_info.cp_length + geometry.preamble_info.symbol_length();
  if (is_short_preamble(geometry.preamble_info.scs)) {
    if ((t_ra_start <= phy_time_unit::from_seconds(0.0)) && (t_ra_end >= phy_time_unit::from_seconds(0.0))) {
      geometry.preamble_info.cp_length += sixteen_kappa;
    }
    if ((t_ra_start <= phy_time_unit::from_seconds(0.5e-3)) && (t_ra_end >= phy_time_unit::from_seconds(0.5e-3))) {
      geometry.preamble_info.cp_length += sixteen_kappa;
    }
  }

  phy_time_unit occasion_duration = geometry.preamble_info.cp_length + geometry.preamble_info.symbol_length();
  geometry.sample_offset          = t_occasion_start.to_samples(srate.to_Hz());
  geometry.nof_samples            = occasion_duration.to_samples(srate.to_Hz());
  geometry.dft_size               = srate.get_dft_size(ra_scs_to_Hz(geometry.preamble_info.scs));
  geometry.cyclic_prefix_length   = geometry.preamble_info.cp_length.to_samples(srate.to_Hz());

  prach_frequency_mapping_information freq_mapping_info =
      prach_frequency_mapping_get(geometry.preamble_info.scs, pusch_scs);
  if ((freq_mapping_info.nof_rb_ra == PRACH_FREQUENCY_MAPPING_INFORMATION_RESERVED.nof_rb_ra) ||
      (freq_mapping_info.k_bar == PRACH_FREQUENCY_MAPPING_INFORMATION_RESERVED.k_bar)) {
    return false;
  }

  unsigned prach_scs_Hz    = ra_scs_to_Hz(geometry.preamble_info.scs);
  unsigned K               = pusch_scs_Hz / prach_scs_Hz;
  geometry.prach_grid_size = config.nof_prb_ul_grid * K * NOF_SUBCARRIERS_PER_RB;
  if ((geometry.dft_size <= geometry.prach_grid_size) ||
      (config.nof_fd_occasions > OCUDU_LOWPHY_PRACH_RX_MAX_FD_OCCASIONS)) {
    return false;
  }

  for (unsigned fd = 0; fd != config.nof_fd_occasions; ++fd) {
    unsigned k_start =
        K * NOF_SUBCARRIERS_PER_RB * (config.rb_offset + freq_mapping_info.nof_rb_ra * fd) + freq_mapping_info.k_bar;
    if (k_start + geometry.preamble_info.sequence_length >= geometry.prach_grid_size) {
      return false;
    }
    geometry.k_start[fd] = k_start;
  }

  return true;
}

bool ofdm_prach_demodulator_cuda_impl::ensure_handle(const ocudu_lowphy_prach_rx_config_t& cfg)
{
  if (handle == nullptr) {
    return ocudu_lowphy_prach_rx_create(&cfg, &handle) != 0;
  }
  return ocudu_lowphy_prach_rx_update_config(handle, &cfg) != 0;
}
