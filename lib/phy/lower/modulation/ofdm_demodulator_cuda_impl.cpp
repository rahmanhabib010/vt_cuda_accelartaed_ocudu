// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ofdm_demodulator_cuda_impl.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/phy/support/resource_grid_writer.h"
#include "ocudu/ran/subcarrier_spacing.h"
#include "ocudu/support/error_handling.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <mutex>

using namespace ocudu;

namespace {

bool env_flag_disabled(const char* name)
{
  const char* value = std::getenv(name);
  return value &&
         ((std::strcmp(value, "0") == 0) || (std::strcmp(value, "false") == 0) || (std::strcmp(value, "off") == 0) ||
          (std::strcmp(value, "no") == 0) || (std::strcmp(value, "disabled") == 0));
}

bool lowphy_puxch_demodulation_disabled_by_env()
{
  return env_flag_disabled("OCUDU_LOWPHY_RX_ACCELERATION") ||
         env_flag_disabled("OCUDU_LOWPHY_PUXCH_DEMODULATION_ACCELERATION");
}

unsigned parse_unsigned_env(const char* name, unsigned default_value, unsigned max_value)
{
  const char* value = std::getenv(name);
  if ((value == nullptr) || (*value == '\0')) {
    return default_value;
  }

  char*         end    = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value) {
    return default_value;
  }

  return static_cast<unsigned>(std::min<unsigned long>(parsed, max_value));
}

unsigned get_rx_stage_timing_warn_threshold_us()
{
  static const unsigned threshold_us = parse_unsigned_env("OCUDU_LOWPHY_RX_STAGE_TIMING_WARN_US", 0, 1000000);
  return threshold_us;
}

unsigned get_rx_stage_timing_warn_max_logs()
{
  static const unsigned max_logs = parse_unsigned_env("OCUDU_LOWPHY_RX_STAGE_TIMING_WARN_MAX_LOGS", 128, 1000000);
  return max_logs;
}

void log_rx_stage_timing_if_slow(const char*                         phase,
                                 std::chrono::steady_clock::duration duration,
                                 unsigned                            nof_ports,
                                 unsigned                            nof_samples,
                                 unsigned                            symbol_index)
{
  const unsigned threshold_us = get_rx_stage_timing_warn_threshold_us();
  if (threshold_us == 0) {
    return;
  }

  const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
  if (duration_us.count() <= static_cast<std::chrono::microseconds::rep>(threshold_us)) {
    return;
  }

  static std::atomic<unsigned> nof_logs{0};
  const unsigned               log_index = nof_logs.fetch_add(1, std::memory_order_relaxed);
  if (log_index >= get_rx_stage_timing_warn_max_logs()) {
    return;
  }

  ocudulog::fetch_basic_logger("PHY").warning(
      "Lower-PHY RX GPU stage timing: {} took {} us for {} port(s), {} samples, symbol {}.",
      phase,
      duration_us.count(),
      nof_ports,
      nof_samples,
      symbol_index);
}

std::atomic<bool>      logged_gpu_path_selected{false};
static constexpr float ci16_input_scale = std::numeric_limits<int16_t>::max();

} // namespace

ofdm_symbol_demodulator_cuda_impl::ofdm_symbol_demodulator_cuda_impl(
    std::unique_ptr<ofdm_symbol_demodulator> fallback_,
    const ofdm_demodulator_configuration&    ofdm_config,
    bool                                     force_gpu_path_) :
  fallback(std::move(fallback_)),
  dft_size(ofdm_config.dft_size),
  rg_size(ofdm_config.bw_rb * NOF_SUBCARRIERS_PER_RB),
  cp(ofdm_config.cp),
  nof_samples_window_offset(ofdm_config.nof_samples_window_offset),
  scs(to_subcarrier_spacing(ofdm_config.numerology)),
  sampling_rate_Hz(to_sampling_rate_Hz(scs, dft_size)),
  scale(ofdm_config.scale),
  phase_compensation_table(to_subcarrier_spacing(ofdm_config.numerology),
                           ofdm_config.cp,
                           ofdm_config.dft_size,
                           ofdm_config.center_freq_Hz,
                           false),
  next_center_freq_Hz(ofdm_config.center_freq_Hz),
  current_center_freq_Hz(ofdm_config.center_freq_Hz),
  force_gpu_path(force_gpu_path_)
{
  ocudu_assert(fallback, "Invalid fallback OFDM demodulator.");
  report_fatal_error_if_not(std::isnormal(scale), "Invalid scaling factor {}.", scale);
  report_fatal_error_if_not(
      dft_size > rg_size, "The DFT size ({}) must be greater than the resource grid size ({}).", dft_size, rg_size);

  int device_count = 0;
  if ((cudaGetDeviceCount(&device_count) != cudaSuccess) || (device_count == 0)) {
    cudaGetLastError();
    return;
  }
  gpu_available = true;
  warmup_handle();
}

ofdm_symbol_demodulator_cuda_impl::~ofdm_symbol_demodulator_cuda_impl()
{
  std::lock_guard<std::mutex> lock(demodulator_mutex);
  if (handle != nullptr) {
    ocudu_lowphy_puxch_rx_destroy(handle);
    handle = nullptr;
  }
}

bool ofdm_symbol_demodulator_cuda_impl::demodulate_ci16(resource_grid_writer& grid,
                                                              span<const ci16_t>    input,
                                                              float                 input_scale,
                                                              unsigned              port_index,
                                                              unsigned              symbol_index)
{
  if (!gpu_available || lowphy_puxch_demodulation_disabled_by_env() || !grid.supports_device_grid_mapping()) {
    report_fatal_error_if_not(!force_gpu_path, "Accelerated lower-PHY RX demodulation requested but unavailable.");
    return false;
  }

  const ci16_t* input_ptr = input.data();
  if (demodulate_gpu_ci16_ports(grid,
                                span<const ci16_t* const>(&input_ptr, 1),
                                input.size(),
                                input_scale,
                                span<const unsigned>(&port_index, 1),
                                symbol_index)) {
    return true;
  }

  report_fatal_error_if_not(!force_gpu_path, "Accelerated lower-PHY RX demodulation failed.");
  return false;
}

bool ofdm_symbol_demodulator_cuda_impl::demodulate_ci16_ports(resource_grid_writer&     grid,
                                                                    span<const ci16_t* const> inputs,
                                                                    unsigned                  nof_samples,
                                                                    float                     input_scale,
                                                                    span<const unsigned>      port_indices,
                                                                    unsigned                  symbol_index)
{
  if (!gpu_available || lowphy_puxch_demodulation_disabled_by_env() || !grid.supports_device_grid_mapping()) {
    report_fatal_error_if_not(!force_gpu_path, "Accelerated lower-PHY RX demodulation requested but unavailable.");
    return false;
  }

  if (demodulate_gpu_ci16_ports(grid, inputs, nof_samples, input_scale, port_indices, symbol_index)) {
    return true;
  }

  report_fatal_error_if_not(!force_gpu_path, "Accelerated lower-PHY RX demodulation failed.");
  return false;
}

bool ofdm_symbol_demodulator_cuda_impl::demodulate_gpu_ci16_ports(resource_grid_writer&     grid,
                                                                        span<const ci16_t* const> inputs,
                                                                        unsigned                  nof_samples,
                                                                        float                     input_scale,
                                                                        span<const unsigned>      port_indices,
                                                                        unsigned                  symbol_index)
{
  const bool timing_enabled = get_rx_stage_timing_warn_threshold_us() != 0;
  auto       total_start    = std::chrono::steady_clock::time_point{};
  auto       stage_start    = std::chrono::steady_clock::time_point{};
  if (timing_enabled) {
    total_start = std::chrono::steady_clock::now();
  }

  if (timing_enabled) {
    stage_start = std::chrono::steady_clock::now();
  }
  std::unique_lock<std::mutex> lock(demodulator_mutex);
  if (timing_enabled) {
    log_rx_stage_timing_if_slow(
        "lock_wait", std::chrono::steady_clock::now() - stage_start, inputs.size(), nof_samples, symbol_index);
  }

  if (inputs.empty() || (inputs.size() != port_indices.size()) || (inputs.size() > OCUDU_LOWPHY_PUXCH_RX_MAX_PORTS)) {
    return false;
  }

  double center_freq_Hz = next_center_freq_Hz.load(std::memory_order_relaxed);
  if (center_freq_Hz != current_center_freq_Hz) {
    phase_compensation_table = phase_compensation_lut(scs, cp, dft_size, center_freq_Hz, false);
    current_center_freq_Hz   = center_freq_Hz;
  }

  unsigned nsymb  = get_nsymb_per_slot(cp);
  unsigned cp_len = cp.get_length(symbol_index, scs).to_samples(sampling_rate_Hz);
  if ((nof_samples != cp_len + dft_size) || (cp_len < nof_samples_window_offset) || (grid.get_nof_subc() < rg_size) ||
      (grid.get_nof_symbols() < nsymb)) {
    return false;
  }
  for (unsigned port_index : port_indices) {
    if (port_index >= grid.get_nof_ports()) {
      return false;
    }
  }

  cf_t phase_compensation = phase_compensation_table.get_coefficient(symbol_index);

  ocudu_lowphy_puxch_rx_config_t cuda_cfg = {};
  cuda_cfg.dft_size                       = static_cast<int>(dft_size);
  cuda_cfg.rg_size                        = static_cast<int>(rg_size);
  cuda_cfg.grid_nof_subc                  = static_cast<int>(grid.get_nof_subc());
  cuda_cfg.grid_nof_symbols               = static_cast<int>(grid.get_nof_symbols());
  cuda_cfg.nof_ports                      = static_cast<int>(inputs.size());
  cuda_cfg.input_nof_samples              = static_cast<int>(nof_samples);
  cuda_cfg.cyclic_prefix_length           = static_cast<int>(cp_len);
  cuda_cfg.window_offset                  = static_cast<int>(nof_samples_window_offset);
  cuda_cfg.symbol_index                   = static_cast<int>(symbol_index % nsymb);
  cuda_cfg.input_is_device                = 0;
  cuda_cfg.dft_scale                      = scale;
  cuda_cfg.phase_re                       = phase_compensation.real();
  cuda_cfg.phase_im                       = phase_compensation.imag();
  for (unsigned i_port = 0; i_port != port_indices.size(); ++i_port) {
    cuda_cfg.port_indices[i_port] = static_cast<int>(port_indices[i_port]);
  }

  if (timing_enabled) {
    stage_start = std::chrono::steady_clock::now();
  }
  if (!ensure_handle(cuda_cfg)) {
    return false;
  }
  if (timing_enabled) {
    log_rx_stage_timing_if_slow(
        "ensure_handle", std::chrono::steady_clock::now() - stage_start, inputs.size(), nof_samples, symbol_index);
  }

  void* stream = ocudu_lowphy_puxch_rx_get_stream(handle);
  if (timing_enabled) {
    stage_start = std::chrono::steady_clock::now();
  }
  if (!grid.prepare_device_grid_mapping(stream)) {
    return false;
  }
  if (timing_enabled) {
    log_rx_stage_timing_if_slow("prepare_grid_mapping",
                                std::chrono::steady_clock::now() - stage_start,
                                inputs.size(),
                                nof_samples,
                                symbol_index);
  }

  auto cancel_gpu_mapping = [&grid]() { (void)grid.cancel_device_grid_mapping(); };

  std::array<const void*, OCUDU_LOWPHY_PUXCH_RX_MAX_PORTS> input_ptrs = {};
  for (unsigned i_port = 0; i_port != inputs.size(); ++i_port) {
    input_ptrs[i_port] = inputs[i_port];
  }

  if (timing_enabled) {
    stage_start = std::chrono::steady_clock::now();
  }
  const int enqueue_result =
      (inputs.size() == 1)
          ? ocudu_lowphy_puxch_rx_process_ci16(handle, input_ptrs[0], input_scale, grid.get_device_grid_bf16(), stream)
          : ocudu_lowphy_puxch_rx_process_ci16_ports(
                handle, input_ptrs.data(), input_scale, grid.get_device_grid_bf16(), stream);
  if (enqueue_result == 0) {
    (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    cancel_gpu_mapping();
    return false;
  }
  if (timing_enabled) {
    log_rx_stage_timing_if_slow(
        "enqueue_demod", std::chrono::steady_clock::now() - stage_start, inputs.size(), nof_samples, symbol_index);
  }

  if (timing_enabled) {
    stage_start = std::chrono::steady_clock::now();
  }
  if (!grid.on_device_grid_mapping_enqueued(stream)) {
    (void)ocudu_lowphy_puxch_rx_synchronize(handle);
    cancel_gpu_mapping();
    return false;
  }
  if (timing_enabled) {
    log_rx_stage_timing_if_slow("record_grid_mapping",
                                std::chrono::steady_clock::now() - stage_start,
                                inputs.size(),
                                nof_samples,
                                symbol_index);
    log_rx_stage_timing_if_slow(
        "total", std::chrono::steady_clock::now() - total_start, inputs.size(), nof_samples, symbol_index);
  }

  if (!logged_gpu_path_selected.exchange(true)) {
    ocudulog::fetch_basic_logger("PHY").info(
        "Lower-PHY RX GPU path selected: direct CUDA-visible uplink resource-grid writer.");
  }

  return true;
}

bool ofdm_symbol_demodulator_cuda_impl::ensure_handle(const ocudu_lowphy_puxch_rx_config_t& cfg)
{
  if (handle == nullptr) {
    return ocudu_lowphy_puxch_rx_create(&cfg, &handle) != 0;
  }
  return ocudu_lowphy_puxch_rx_update_config(handle, &cfg) != 0;
}

void ofdm_symbol_demodulator_cuda_impl::warmup_handle()
{
  if (!gpu_available || lowphy_puxch_demodulation_disabled_by_env()) {
    return;
  }

  std::lock_guard<std::mutex> lock(demodulator_mutex);

  const unsigned nsymb                    = get_nsymb_per_slot(cp);
  const unsigned nof_symbols_per_subframe = nsymb * get_nof_slots_per_subframe(scs);

  ocudu_lowphy_puxch_rx_config_t cuda_cfg = {};
  cuda_cfg.dft_size                       = static_cast<int>(dft_size);
  cuda_cfg.rg_size                        = static_cast<int>(rg_size);
  cuda_cfg.grid_nof_subc                  = static_cast<int>(rg_size);
  cuda_cfg.grid_nof_symbols               = static_cast<int>(nsymb);
  cuda_cfg.nof_ports                      = 1;
  cuda_cfg.input_is_device                = 0;
  cuda_cfg.dft_scale                      = scale;
  cuda_cfg.port_indices[0]                = 0;

  for (unsigned symbol = 0; symbol != nof_symbols_per_subframe; ++symbol) {
    unsigned cp_len = cp.get_length(symbol, scs).to_samples(sampling_rate_Hz);
    if (cp_len < nof_samples_window_offset) {
      continue;
    }

    cuda_cfg.input_nof_samples    = static_cast<int>(dft_size + cp_len);
    cuda_cfg.cyclic_prefix_length = static_cast<int>(cp_len);
    cuda_cfg.window_offset        = static_cast<int>(nof_samples_window_offset);
    cuda_cfg.symbol_index         = static_cast<int>(symbol % nsymb);
    cf_t phase_compensation       = phase_compensation_table.get_coefficient(symbol);
    cuda_cfg.phase_re             = phase_compensation.real();
    cuda_cfg.phase_im             = phase_compensation.imag();

    if (!ensure_handle(cuda_cfg)) {
      return;
    }
    if (ocudu_lowphy_puxch_rx_prime_config(handle, &cuda_cfg, ci16_input_scale, nullptr) == 0) {
      return;
    }
  }
}
