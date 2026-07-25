// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "prach_detector_cuda_impl.h"
#include "../../channel_coding/ldpc/cuda/cuda_rt_utils.h"
#include "prach_detector_generic_thresholds.h"
#include "ocudu/adt/interval.h"
#include "ocudu/phy/support/prach_buffer.h"
#include "ocudu/ran/prach/prach_cyclic_shifts.h"
#include "ocudu/ran/prach/prach_preamble_information.h"
#include "ocudu/support/error_handling.h"
#include "ocudu/support/math/math_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <string>

using namespace ocudu;

namespace {

bool env_flag_enabled(const char* name, bool default_value = false)
{
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  std::string mode(value);
  return (mode == "1") || (mode == "true") || (mode == "on") || (mode == "yes") || (mode == "enabled");
}

unsigned env_unsigned(const char* name, unsigned default_value)
{
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  char*         end    = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if ((end == value) || (parsed > std::numeric_limits<unsigned>::max())) {
    return default_value;
  }
  return static_cast<unsigned>(parsed);
}

bool is_success(nr_ldpc_status_t status)
{
  return status == NR_LDPC_SUCCESS;
}

} // namespace

struct prach_detector_cuda_impl::detector_geometry {
  prach_preamble_information preamble_info;
  unsigned                   n_cs;
  unsigned                   nof_shifts;
  unsigned                   nof_sequences;
  unsigned                   active_sequence_start;
  unsigned                   active_nof_sequences;
  unsigned                   sequence_length;
  unsigned                   dft_size;
  unsigned                   win_width;
  unsigned                   win_margin;
  unsigned                   max_delay_samples;
  unsigned                   nof_symbols;
  float                      threshold;
  bool                       combine_symbols;
  double                     sampling_rate_hz;
  unsigned                   cp_prach;
};

prach_detector_cuda_impl::prach_detector_cuda_impl(std::unique_ptr<prach_generator> generator_,
                                                               std::unique_ptr<prach_detector>  fallback_,
                                                               bool                             force_gpu_path_) :
  generator(std::move(generator_)), fallback(std::move(fallback_)), force_gpu_path(force_gpu_path_)
{
  ocudu_assert(generator, "Invalid PRACH generator.");
  ocudu_assert(fallback, "Invalid fallback PRACH detector.");

  int device_count = 0;
  if ((cudaGetDeviceCount(&device_count) != cudaSuccess) || (device_count == 0)) {
    cudaGetLastError();
    return;
  }

  device_id = 0;
  if (cudaSetDevice(device_id) != cudaSuccess) {
    return;
  }

  cudaError_t status = cudaSetDeviceFlags(cudaDeviceScheduleYield);
  if (status == cudaErrorSetOnActiveProcess) {
    cudaGetLastError();
  } else if (status != cudaSuccess) {
    return;
  }

  cudaStream_t cuda_stream = nullptr;
  status                   = ocudu::cudaStreamCreateUpperPhy(&cuda_stream);
  if (status != cudaSuccess) {
    return;
  }
  stream = cuda_stream;

  if (!is_success(prach_detector_create(&handle))) {
    cudaStreamDestroy(cuda_stream);
    stream = nullptr;
    handle = nullptr;
    return;
  }

  gpu_available = true;
}

prach_detector_cuda_impl::~prach_detector_cuda_impl()
{
  std::lock_guard<std::mutex> lock(detector_mutex);
  if (gpu_available) {
    (void)cudaSetDevice(device_id);
  }
  if (handle != nullptr) {
    prach_detector_destroy(handle);
    handle = nullptr;
  }
  if (stream != nullptr) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    stream = nullptr;
  }
}

bool ocudu::is_prach_detector_cuda_available()
{
  int device_count = 0;
  if ((cudaGetDeviceCount(&device_count) != cudaSuccess) || (device_count == 0)) {
    cudaGetLastError();
    return false;
  }
  return true;
}

prach_detection_result prach_detector_cuda_impl::detect(const prach_buffer&                  input,
                                                              const prach_detector::configuration& config)
{
  detector_geometry geometry;
  geometry.preamble_info = is_long_preamble(config.format)
                               ? get_prach_preamble_long_info(config.format)
                               : get_prach_preamble_short_info(config.format, config.ra_scs, false);
  geometry.n_cs          = prach_cyclic_shifts_get(config.ra_scs, config.restricted_set, config.zero_correlation_zone);
  report_fatal_error_if_not(geometry.n_cs != PRACH_CYCLIC_SHIFTS_RESERVED, "Reserved PRACH cyclic shift.");

  geometry.sequence_length =
      is_short_preamble(config.format) ? prach_constants::SHORT_SEQUENCE_LENGTH : prach_constants::LONG_SEQUENCE_LENGTH;
  geometry.nof_shifts    = 1;
  geometry.nof_sequences = 64;
  if (geometry.n_cs != 0) {
    geometry.nof_shifts    = std::min(prach_constants::MAX_NUM_PREAMBLES, geometry.sequence_length / geometry.n_cs);
    geometry.nof_sequences = divide_ceil(64U, geometry.nof_shifts);
  }
  unsigned first_requested_sequence =
      std::min(config.start_preamble_index / geometry.nof_shifts, geometry.nof_sequences - 1U);
  unsigned requested_preamble_end =
      std::min(prach_constants::MAX_NUM_PREAMBLES, config.start_preamble_index + config.nof_preamble_indices);
  unsigned last_requested_sequence =
      std::min(geometry.nof_sequences, divide_ceil(requested_preamble_end, geometry.nof_shifts));
  if (last_requested_sequence <= first_requested_sequence) {
    last_requested_sequence = first_requested_sequence + 1U;
  }
  geometry.active_sequence_start = first_requested_sequence;
  geometry.active_nof_sequences  = last_requested_sequence - first_requested_sequence;

  geometry.dft_size         = is_long_preamble(config.format) ? 1024U : 256U;
  geometry.sampling_rate_hz = geometry.dft_size * ra_scs_to_Hz(geometry.preamble_info.scs);

  double cp_duration = geometry.preamble_info.cp_length.to_seconds();
  geometry.cp_prach  = static_cast<unsigned>(
      std::floor(cp_duration * geometry.sequence_length * ra_scs_to_Hz(geometry.preamble_info.scs)));
  geometry.win_width = std::min(geometry.n_cs, geometry.cp_prach);
  if (geometry.n_cs == 0) {
    geometry.win_width = geometry.cp_prach;
  }
  if (geometry.win_width == geometry.sequence_length) {
    geometry.win_width -= 20;
  }
  geometry.win_width = (geometry.win_width * geometry.dft_size) / geometry.sequence_length;

  detail::threshold_params th_params;
  th_params.nof_rx_ports                        = config.nof_rx_ports;
  th_params.scs                                 = config.ra_scs;
  th_params.format                              = config.format;
  th_params.zero_correlation_zone               = config.zero_correlation_zone;
  auto [threshold, combine_symbols, win_margin] = detail::get_threshold_and_margin(th_params);
  geometry.threshold                            = threshold;
  geometry.combine_symbols                      = combine_symbols;
  geometry.win_margin                           = win_margin;
  geometry.max_delay_samples =
      (geometry.n_cs == 0) ? geometry.cp_prach : std::min(std::max(geometry.n_cs, 1U) - 1U, geometry.cp_prach);
  geometry.max_delay_samples = (geometry.max_delay_samples * geometry.dft_size) / geometry.sequence_length;
  geometry.nof_symbols       = geometry.preamble_info.nof_symbols;

  if (!should_use_gpu(input, config, geometry)) {
    return fallback->detect(input, config);
  }

  std::lock_guard<std::mutex> lock(detector_mutex);
  return detect_gpu(input, config, geometry);
}

bool prach_detector_cuda_impl::should_use_gpu(const prach_buffer&      input,
                                                    const configuration&     config,
                                                    const detector_geometry& geometry) const
{
  if (!gpu_available || (handle == nullptr) || (stream == nullptr)) {
    report_fatal_error_if_not(!force_gpu_path, "Accelerated PRACH detector requested but CUDA is not available.");
    return false;
  }

  if (force_gpu_path || env_flag_enabled("OCUDU_PRACH_ACCELERATION_FORCE")) {
    return true;
  }

  if (is_short_preamble(config.format)) {
    unsigned min_short_work             = env_unsigned("OCUDU_PRACH_ACCELERATION_MIN_SHORT_WORK", 128);
    unsigned min_device_short_preambles = env_unsigned("OCUDU_PRACH_ACCELERATION_MIN_DEVICE_SHORT_PREAMBLES", 128);
    unsigned short_work                 = config.nof_rx_ports * config.nof_preamble_indices;
    return (short_work >= min_short_work) ||
           (input.supports_device_prach_buffer_reading() && config.nof_preamble_indices >= min_device_short_preambles);
  }

  unsigned min_long_sequence_work = env_unsigned("OCUDU_PRACH_ACCELERATION_MIN_LONG_SEQUENCE_WORK", 4);
  unsigned long_sequence_work     = config.nof_rx_ports * geometry.active_nof_sequences;
  return long_sequence_work >= min_long_sequence_work;
}

prach_detection_result prach_detector_cuda_impl::detect_gpu(const prach_buffer&      input,
                                                                  const configuration&     config,
                                                                  const detector_geometry& geometry)
{
  prepare_roots(config, geometry);

  prach_detector_config_t cuda_config = {};
  cuda_config.sequence_length         = static_cast<int>(geometry.sequence_length);
  cuda_config.dft_size                = static_cast<int>(geometry.dft_size);
  cuda_config.nof_rx_ports            = static_cast<int>(config.nof_rx_ports);
  cuda_config.nof_symbols             = static_cast<int>(geometry.nof_symbols);
  cuda_config.nof_sequences           = static_cast<int>(geometry.active_nof_sequences);
  cuda_config.sequence_start          = static_cast<int>(geometry.active_sequence_start);
  cuda_config.nof_shifts              = static_cast<int>(geometry.nof_shifts);
  cuda_config.n_cs                    = static_cast<int>(geometry.n_cs);
  cuda_config.win_width               = static_cast<int>(geometry.win_width);
  cuda_config.win_margin              = static_cast<int>(geometry.win_margin);
  cuda_config.max_delay_samples       = static_cast<int>(geometry.max_delay_samples);
  cuda_config.start_preamble_index    = static_cast<int>(config.start_preamble_index);
  cuda_config.nof_preamble_indices    = static_cast<int>(config.nof_preamble_indices);
  cuda_config.combine_symbols         = geometry.combine_symbols ? 1 : 0;
  cuda_config.threshold               = geometry.threshold;
  uint64_t active_root_key =
      cached_root_key ^ (0x9e3779b97f4a7c15ULL + (static_cast<uint64_t>(geometry.active_sequence_start) << 6) +
                         (static_cast<uint64_t>(geometry.active_sequence_start) >> 2));
  cuda_config.root_cache_key = active_root_key;

  const void* prach_input = nullptr;
  if (input.supports_device_prach_buffer_reading() && input.prepare_device_prach_buffer_reading(stream)) {
    const auto* device_base         = static_cast<const uint32_t*>(input.get_device_prach_buffer_cbf16());
    unsigned    offset0             = input.get_device_prach_symbol_offset(0, 0, 0, 0);
    prach_input                     = device_base + offset0;
    cuda_config.input_is_device     = 1;
    cuda_config.input_symbol_stride = (input.get_max_nof_symbols() > 1)
                                          ? static_cast<int>(input.get_device_prach_symbol_offset(0, 0, 0, 1) - offset0)
                                          : static_cast<int>(geometry.sequence_length);
    cuda_config.input_port_stride   = (config.nof_rx_ports > 1)
                                          ? static_cast<int>(input.get_device_prach_symbol_offset(1, 0, 0, 0) - offset0)
                                          : static_cast<int>(geometry.nof_symbols * geometry.sequence_length);
  } else {
    host_prach_packed.resize(static_cast<size_t>(config.nof_rx_ports) * geometry.nof_symbols *
                             geometry.sequence_length);
    for (unsigned i_port = 0; i_port != config.nof_rx_ports; ++i_port) {
      for (unsigned i_symbol = 0; i_symbol != geometry.nof_symbols; ++i_symbol) {
        span<const cbf16_t> symbol = input.get_symbol(i_port, 0, 0, i_symbol);
        std::memcpy(host_prach_packed.data() + (i_port * geometry.nof_symbols + i_symbol) * geometry.sequence_length,
                    symbol.data(),
                    geometry.sequence_length * sizeof(cbf16_t));
      }
    }
    prach_input                     = host_prach_packed.data();
    cuda_config.input_is_device     = 0;
    cuda_config.input_symbol_stride = static_cast<int>(geometry.sequence_length);
    cuda_config.input_port_stride   = static_cast<int>(geometry.nof_symbols * geometry.sequence_length);
  }

  prach_detector_result_t cuda_result = {};
  const cf_t*             active_roots =
      host_roots.data() + static_cast<size_t>(geometry.active_sequence_start) * geometry.sequence_length;
  nr_ldpc_status_t status =
      prach_detector_detect(handle, prach_input, active_roots, &cuda_config, &cuda_result, stream);
  if (!is_success(status)) {
    report_fatal_error_if_not(!force_gpu_path, "Accelerated PRACH detection failed.");
    return fallback->detect(input, config);
  }

  prach_detection_result result;
  result.rssi_dB         = convert_power_to_dB(cuda_result.rssi);
  result.time_resolution = phy_time_unit::from_seconds(1.0 / geometry.sampling_rate_hz);
  result.time_advance_max =
      phy_time_unit::from_seconds(static_cast<double>(geometry.max_delay_samples) * 0.8 / geometry.sampling_rate_hz);
  result.preambles.clear();
  for (int i = 0; i != cuda_result.nof_candidates; ++i) {
    const prach_detector_candidate_t&            candidate = cuda_result.candidates[i];
    prach_detection_result::preamble_indication& info      = result.preambles.emplace_back();
    info.preamble_index                                    = static_cast<unsigned>(candidate.preamble_index);
    info.time_advance =
        phy_time_unit::from_seconds(static_cast<double>(candidate.delay_samples) / geometry.sampling_rate_hz);
    info.detection_metric  = candidate.detection_metric;
    info.preamble_power_dB = convert_power_to_dB(candidate.preamble_power);
  }

  return result;
}

void prach_detector_cuda_impl::prepare_roots(const configuration& config, const detector_geometry& geometry)
{
  uint64_t root_key = make_root_cache_key(config, geometry);
  if (root_key == cached_root_key && host_roots.size() == geometry.nof_sequences * geometry.sequence_length) {
    return;
  }

  host_roots.resize(static_cast<size_t>(geometry.nof_sequences) * geometry.sequence_length);
  for (unsigned i_sequence = 0; i_sequence != geometry.nof_sequences; ++i_sequence) {
    prach_generator::configuration generator_config;
    generator_config.format                = config.format;
    generator_config.root_sequence_index   = config.root_sequence_index;
    generator_config.preamble_index        = i_sequence * geometry.nof_shifts;
    generator_config.restricted_set        = config.restricted_set;
    generator_config.zero_correlation_zone = config.zero_correlation_zone;

    span<const cf_t> root = generator->generate(generator_config);
    std::copy(root.begin(), root.end(), host_roots.begin() + i_sequence * geometry.sequence_length);
  }
  cached_root_key = root_key;
}

uint64_t prach_detector_cuda_impl::make_root_cache_key(const configuration&     config,
                                                             const detector_geometry& geometry) const
{
  uint64_t key = 0;
  key ^= static_cast<uint64_t>(config.root_sequence_index & 0xffffU);
  key ^= static_cast<uint64_t>(config.zero_correlation_zone & 0xffU) << 16;
  key ^= static_cast<uint64_t>(static_cast<unsigned>(config.format) & 0xffU) << 24;
  key ^= static_cast<uint64_t>(static_cast<unsigned>(config.ra_scs) & 0xffU) << 32;
  key ^= static_cast<uint64_t>(static_cast<unsigned>(config.restricted_set) & 0x3U) << 40;
  key ^= static_cast<uint64_t>(geometry.nof_sequences & 0xffU) << 42;
  key ^= static_cast<uint64_t>(geometry.nof_shifts & 0x3fffU) << 50;
  return key;
}
