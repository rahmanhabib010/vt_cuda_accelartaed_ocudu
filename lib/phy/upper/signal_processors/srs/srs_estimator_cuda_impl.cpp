// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "srs_estimator_cuda_impl.h"
#include "../../channel_coding/ldpc/cuda/cuda_rt_utils.h"
#include "cuda/pusch_device_grid_reader_cuda.h"
#include "srs_validator_generic_impl.h"
#include "ocudu/adt/expected.h"
#include "ocudu/adt/static_vector.h"
#include "ocudu/phy/support/resource_grid_reader.h"
#include "ocudu/phy/support/time_alignment_estimator/time_alignment_measurement.h"
#include "ocudu/phy/upper/signal_processors/srs/srs_estimator_configuration.h"
#include "ocudu/phy/upper/signal_processors/srs/srs_estimator_result.h"
#include "ocudu/ran/cyclic_prefix.h"
#include "ocudu/ran/phy_time_unit.h"
#include "ocudu/ran/resource_block.h"
#include "ocudu/ran/srs/srs_channel_matrix.h"
#include "ocudu/ran/srs/srs_constants.h"
#include "ocudu/ran/srs/srs_information.h"
#include "ocudu/support/math/math_utils.h"
#include "ocudu/support/ocudu_assert.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <string>
#include <utility>

using namespace ocudu;

namespace {

/// \brief Looks at the output of the validator and, if unsuccessful, fills \c msg with the error message.
[[maybe_unused]] bool handle_validation(std::string& msg, const error_type<std::string>& err)
{
  bool is_success = err.has_value();
  if (!is_success) {
    msg = err.error();
  }
  return is_success;
}

bool is_env_enabled(const char* name)
{
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }

  return (std::strcmp(value, "1") == 0) || (std::strcmp(value, "true") == 0) || (std::strcmp(value, "TRUE") == 0) ||
         (std::strcmp(value, "on") == 0) || (std::strcmp(value, "ON") == 0);
}

bool use_windowed_correlation()
{
  if (is_env_enabled("OCUDU_SRS_ACCELERATION_FULL_CORRELATION")) {
    return false;
  }

  const char* value = std::getenv("OCUDU_SRS_ACCELERATION_WINDOWED_CORRELATION");
  if (value == nullptr) {
    return true;
  }

  return is_env_enabled("OCUDU_SRS_ACCELERATION_WINDOWED_CORRELATION");
}

bool force_srs_gpu()
{
  return is_env_enabled("OCUDU_SRS_ACCELERATION_FORCE");
}

unsigned get_srs_gpu_min_pilot_re()
{
  const char* value = std::getenv("OCUDU_SRS_ACCELERATION_MIN_PILOT_RE");
  if (value == nullptr) {
    return 4096;
  }

  return static_cast<unsigned>(std::strtoul(value, nullptr, 10));
}

bool is_srs_gpu_workload_large_enough(unsigned nof_rx_ports, unsigned nof_symbols, unsigned sequence_length)
{
  if (force_srs_gpu()) {
    return true;
  }

  unsigned min_pilot_re = get_srs_gpu_min_pilot_re();
  if (min_pilot_re == 0) {
    return true;
  }

  size_t nof_observed_pilots =
      static_cast<size_t>(nof_rx_ports) * static_cast<size_t>(nof_symbols) * static_cast<size_t>(sequence_length);
  return nof_observed_pilots >= min_pilot_re;
}

unsigned get_gpu_ta_dft_size(unsigned nof_required_re)
{
  constexpr unsigned max_dft_size = pow2(log2_ceil(static_cast<unsigned>(MAX_NOF_SUBCARRIERS)));
  const unsigned     min_dft_size = pow2(log2_ceil(static_cast<unsigned>(
      1.0F / (15000 * phy_time_unit::from_timing_advance(1, subcarrier_spacing::kHz15).to_seconds()))));

  ocudu_assert(nof_required_re <= MAX_NOF_SUBCARRIERS,
               "The number of required RE (i.e., {}) is larger than the maximum allowed number of RE (i.e., {}).",
               nof_required_re,
               MAX_NOF_SUBCARRIERS);

  // Match the DFT size selection of time_alignment_estimator_dft_impl::get_idft().
  nof_required_re = (nof_required_re * max_dft_size) / MAX_NOF_SUBCARRIERS;
  return std::max(min_dft_size, pow2(log2_ceil(nof_required_re)));
}

unsigned get_ta_max_samples(unsigned correlation_size, unsigned stride, subcarrier_spacing scs, double max_ta)
{
  phy_time_unit half_cyclic_prefix_duration =
      phy_time_unit::from_units_of_kappa(144) / pow2(to_numerology_value(scs) + 1);
  double   sampling_rate_Hz = correlation_size * scs_to_khz(scs) * 1000 * stride;
  unsigned max_ta_samples   = std::floor(half_cyclic_prefix_duration.to_seconds() * sampling_rate_Hz);
  if (std::isnormal(max_ta)) {
    max_ta_samples = std::min(max_ta_samples, static_cast<unsigned>(std::floor(max_ta * sampling_rate_Hz)));
  }

  return max_ta_samples;
}

unsigned
get_ta_correlation_window_size(unsigned correlation_size, unsigned stride, subcarrier_spacing scs, double max_ta)
{
  unsigned max_ta_samples = get_ta_max_samples(correlation_size, stride, scs, max_ta);

  // The TA estimator only searches the early and late windows. Two guard samples preserve fractional peak fitting.
  return std::min(correlation_size / 2, max_ta_samples + 2U);
}

bool is_success(nr_ldpc_status_t status)
{
  return status == NR_LDPC_SUCCESS;
}

} // namespace

srs_estimator_cuda_impl::srs_estimator_cuda_impl(
    std::unique_ptr<low_papr_sequence_generator> sequence_generator_,
    std::unique_ptr<srs_estimator>               fallback_,
    unsigned                                     max_nof_prb_,
    bool                                         force_gpu_path_) :
  sequence_generator(std::move(sequence_generator_)),
  fallback(std::move(fallback_)),
  max_nof_prb(max_nof_prb_),
  force_gpu_path(force_gpu_path_)
{
  ocudu_assert(sequence_generator, "Invalid sequence generator.");
  ocudu_assert(fallback, "Invalid fallback SRS estimator.");
  ocudu_assert(max_nof_prb != 0, "Maximum number of PRB cannot be zero.");

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
  stream = static_cast<void*>(cuda_stream);

  if (!is_success(srs_estimator_create(&handle))) {
    cudaStreamDestroy(cuda_stream);
    stream = nullptr;
    handle = nullptr;
    return;
  }

  gpu_available = true;
}

srs_estimator_cuda_impl::~srs_estimator_cuda_impl()
{
  std::lock_guard<std::mutex> lock(estimator_mutex);
  if (gpu_available) {
    (void)cudaSetDevice(device_id);
  }
  if (handle != nullptr) {
    srs_estimator_destroy(handle);
    handle = nullptr;
  }
  if (stream != nullptr) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    stream = nullptr;
  }
}

bool ocudu::is_srs_estimator_cuda_available()
{
  int device_count = 0;
  if ((cudaGetDeviceCount(&device_count) != cudaSuccess) || (device_count == 0)) {
    cudaGetLastError();
    return false;
  }
  return true;
}

srs_estimator_result srs_estimator_cuda_impl::estimate(const resource_grid_reader&        grid,
                                                             const srs_estimator_configuration& config)
{
  std::lock_guard<std::mutex> lock(estimator_mutex);

  // Makes sure the PDU is valid, matching the software estimator contract.
  [[maybe_unused]] std::string msg;
  ocudu_assert(handle_validation(msg, srs_validator_generic_impl(max_nof_prb).is_valid(config)), "{}", msg);

  if (!gpu_available || (handle == nullptr) || (stream == nullptr)) {
    return fallback->estimate(grid, config);
  }
  if (cudaSetDevice(device_id) != cudaSuccess) {
    return fallback->estimate(grid, config);
  }

  unsigned nof_rx_ports         = config.ports.size();
  auto     nof_antenna_ports    = static_cast<unsigned>(config.resource.nof_antenna_ports);
  auto     nof_symbols          = static_cast<unsigned>(config.resource.nof_symbols);
  unsigned nof_symbols_per_slot = get_nsymb_per_slot(cyclic_prefix::NORMAL);
  ocudu_assert(config.resource.start_symbol.value() + nof_symbols <= nof_symbols_per_slot,
               "The start symbol index (i.e., {}) plus the number of symbols (i.e., {}) exceeds the number of symbols "
               "per slot (i.e., {})",
               config.resource.start_symbol,
               nof_symbols,
               nof_symbols_per_slot);

  subcarrier_spacing scs             = to_subcarrier_spacing(config.slot.numerology());
  auto               comb_size       = static_cast<unsigned>(config.resource.comb_size);
  srs_information    common_info     = get_srs_information(config.resource, 0);
  unsigned           sequence_length = common_info.sequence_length;
  double             max_ta = 1.0 / static_cast<double>(common_info.n_cs_max * scs_to_khz(scs) * 1000 * comb_size);

  if ((sequence_length == 0) || (sequence_length > SRS_ESTIMATOR_MAX_SEQUENCE_LENGTH) ||
      (nof_rx_ports > SRS_ESTIMATOR_MAX_RX_PORTS) || (nof_antenna_ports > SRS_ESTIMATOR_MAX_TX_PORTS)) {
    return fallback->estimate(grid, config);
  }

  unsigned dft_size = get_gpu_ta_dft_size(sequence_length);
  if (dft_size > SRS_ESTIMATOR_MAX_DFT_SIZE) {
    return fallback->estimate(grid, config);
  }

  if (!force_gpu_path && !is_srs_gpu_workload_large_enough(nof_rx_ports, nof_symbols, sequence_length)) {
    return fallback->estimate(grid, config);
  }

  srs_information info_port0         = get_srs_information(config.resource, /*i_antenna_port=*/0);
  bool            interleaved_pilots = (nof_antenna_ports == 4) && (info_port0.n_cs >= info_port0.n_cs_max / 2);

  srs_estimator_config_t gpu_cfg = {};
  gpu_cfg.nof_rx_ports           = static_cast<int>(nof_rx_ports);
  gpu_cfg.nof_tx_ports           = static_cast<int>(nof_antenna_ports);
  gpu_cfg.nof_symbols            = static_cast<int>(nof_symbols);
  gpu_cfg.start_symbol           = static_cast<int>(config.resource.start_symbol.value());
  gpu_cfg.sequence_length        = static_cast<int>(sequence_length);
  gpu_cfg.comb_size              = static_cast<int>(comb_size);
  gpu_cfg.grid_nof_ports         = static_cast<int>(grid.get_nof_ports());
  gpu_cfg.grid_nof_symbols       = static_cast<int>(grid.get_nof_symbols());
  gpu_cfg.grid_nof_subcarriers   = static_cast<int>(grid.get_nof_subc());
  gpu_cfg.scs_khz                = static_cast<int>(scs_to_khz(scs));
  gpu_cfg.dft_size               = static_cast<int>(dft_size);
  gpu_cfg.correlation_window_size =
      use_windowed_correlation() ? static_cast<int>(get_ta_correlation_window_size(dft_size, comb_size, scs, max_ta))
                                 : 0;
  gpu_cfg.ta_max_samples     = static_cast<int>(get_ta_max_samples(dft_size, comb_size, scs, max_ta));
  gpu_cfg.interleaved_pilots = interleaved_pilots ? 1 : 0;

  for (unsigned i_rx_port_index = 0; i_rx_port_index != nof_rx_ports; ++i_rx_port_index) {
    gpu_cfg.rx_ports[i_rx_port_index] = config.ports[i_rx_port_index];
  }

  std::array<srs_information, SRS_ESTIMATOR_MAX_TX_PORTS> info_per_tx;
  std::vector<int>                                        config_signature;
  config_signature.reserve(32);
  config_signature.insert(config_signature.end(),
                          {gpu_cfg.nof_rx_ports,
                           gpu_cfg.nof_tx_ports,
                           gpu_cfg.nof_symbols,
                           gpu_cfg.start_symbol,
                           gpu_cfg.sequence_length,
                           gpu_cfg.comb_size,
                           gpu_cfg.grid_nof_ports,
                           gpu_cfg.grid_nof_symbols,
                           gpu_cfg.grid_nof_subcarriers,
                           gpu_cfg.scs_khz,
                           gpu_cfg.dft_size,
                           gpu_cfg.correlation_window_size,
                           gpu_cfg.ta_max_samples,
                           gpu_cfg.interleaved_pilots});
  for (unsigned i_rx_port_index = 0; i_rx_port_index != nof_rx_ports; ++i_rx_port_index) {
    config_signature.push_back(gpu_cfg.rx_ports[i_rx_port_index]);
  }

  for (unsigned i_antenna_port = 0; i_antenna_port != nof_antenna_ports; ++i_antenna_port) {
    srs_information info        = get_srs_information(config.resource, i_antenna_port);
    info_per_tx[i_antenna_port] = info;

    gpu_cfg.mapping_initial_subcarrier[i_antenna_port] = static_cast<int>(info.mapping_initial_subcarrier);
    config_signature.insert(config_signature.end(),
                            {static_cast<int>(info.sequence_length),
                             static_cast<int>(info.sequence_group),
                             static_cast<int>(info.sequence_number),
                             static_cast<int>(info.n_cs),
                             static_cast<int>(info.n_cs_max),
                             static_cast<int>(info.mapping_initial_subcarrier),
                             static_cast<int>(info.comb_size)});
  }

  bool needs_configure = !gpu_configured || (config_signature != last_config_signature);
  if (needs_configure) {
    sequences.resize(nof_antenna_ports * sequence_length);
    for (unsigned i_antenna_port = 0; i_antenna_port != nof_antenna_ports; ++i_antenna_port) {
      srs_information info = info_per_tx[i_antenna_port];

      static_vector<cf_t, SRS_ESTIMATOR_MAX_SEQUENCE_LENGTH> sequence(sequence_length);
      sequence_generator->generate(sequence, info.sequence_group, info.sequence_number, info.n_cs, info.n_cs_max);
      for (unsigned i_re = 0; i_re != sequence_length; ++i_re) {
        srs_estimator_cf_t& dst = sequences[i_antenna_port * sequence_length + i_re];
        dst.real                = sequence[i_re].real();
        dst.imag                = sequence[i_re].imag();
      }
    }

    if (!is_success(srs_estimator_configure(handle, &gpu_cfg, sequences.data(), stream))) {
      gpu_configured = false;
      return fallback->estimate(grid, config);
    }
    last_config_signature = std::move(config_signature);
    gpu_configured        = true;
  }

  const void* d_grid_cbf16       = nullptr;
  bool        direct_device_grid = false;
  if (grid.supports_device_grid_reading() && (grid.get_device_grid_cbf16() != nullptr) &&
      grid.prepare_device_grid_reading(stream)) {
    d_grid_cbf16       = grid.get_device_grid_cbf16();
    direct_device_grid = true;
  } else {
    if ((staged_grid_source != &grid) || !staged_grid_reader) {
      staged_grid_reader = std::make_unique<pusch_device_grid_reader_cuda>(grid);
      staged_grid_source = &grid;
    }

    if (!staged_grid_reader->stage_host_grid_async(stream)) {
      return fallback->estimate(grid, config);
    }
    d_grid_cbf16 = staged_grid_reader->get_device_grid_cbf16();
  }

  srs_estimator_metrics_t        metrics            = {};
  srs_estimator_time_alignment_t gpu_time_alignment = {};
  if ((d_grid_cbf16 == nullptr) ||
      !is_success(srs_estimator_estimate_auto_ta(
          handle, d_grid_cbf16, static_cast<float>(max_ta), &gpu_time_alignment, &metrics, stream))) {
    return fallback->estimate(grid, config);
  }
  if (direct_device_grid && !grid.on_device_grid_reading_enqueued(stream)) {
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
  }

  srs_estimator_result result;
  result.time_alignment = time_alignment_measurement{.time_alignment = gpu_time_alignment.time_alignment_s,
                                                     .resolution     = gpu_time_alignment.resolution_s,
                                                     .min            = gpu_time_alignment.min_s,
                                                     .max            = gpu_time_alignment.max_s};
  result.channel_matrix = srs_channel_matrix(nof_rx_ports, nof_antenna_ports);
  result.noise_variance = metrics.noise_variance;
  result.epre_dB        = metrics.epre_dB;
  result.rsrp_dB        = metrics.rsrp_dB;

  for (unsigned i_rx_port = 0; i_rx_port != nof_rx_ports; ++i_rx_port) {
    for (unsigned i_antenna_port = 0; i_antenna_port != nof_antenna_ports; ++i_antenna_port) {
      const srs_estimator_cf_t& coefficient = metrics.coeff[i_rx_port * nof_antenna_ports + i_antenna_port];
      result.channel_matrix.set_coefficient({coefficient.real, coefficient.imag}, i_rx_port, i_antenna_port);
    }
  }

  return result;
}
