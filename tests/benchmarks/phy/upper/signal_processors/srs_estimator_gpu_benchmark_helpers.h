// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#pragma once

#include "ocudu/adt/complex.h"
#include "ocudu/adt/span.h"
#include "ocudu/adt/static_vector.h"
#include "ocudu/phy/generic_functions/generic_functions_factories.h"
#include "ocudu/phy/support/resource_grid.h"
#include "ocudu/phy/support/resource_grid_reader.h"
#include "ocudu/phy/support/resource_grid_writer.h"
#include "ocudu/phy/support/support_factories.h"
#include "ocudu/phy/support/time_alignment_estimator/time_alignment_estimator_factories.h"
#include "ocudu/phy/upper/sequence_generators/sequence_generator_factories.h"
#include "ocudu/phy/upper/signal_processors/srs/srs_estimator.h"
#include "ocudu/phy/upper/signal_processors/srs/srs_estimator_configuration.h"
#include "ocudu/phy/upper/signal_processors/srs/srs_estimator_factory.h"
#include "ocudu/phy/upper/signal_processors/srs/srs_estimator_result.h"
#include "ocudu/ran/cyclic_prefix.h"
#include "ocudu/ran/srs/srs_channel_matrix.h"
#include "ocudu/ran/srs/srs_information.h"
#include "ocudu/ran/srs/srs_resource_configuration.h"
#include "ocudu/support/error_handling.h"
#include "ocudu/support/math/math_utils.h"
#include "ocudu/support/ocudu_test.h"
#include "fmt/format.h"
#ifdef ENABLE_CUDA
#include "resource_grid_cuda_visible_impl.h"
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace ocudu::srs_gpu_bench {

enum class grid_mode { host, visible };

struct estimator_pair {
  std::unique_ptr<srs_estimator> cpu;
  std::unique_ptr<srs_estimator> gpu;
};

struct latency_stats {
  double mean_us = 0;
  double p50_us  = 0;
  double p95_us  = 0;
  double min_us  = 0;
  double max_us  = 0;
};

struct srs_benchmark_profile {
  const char*             name                = "default";
  unsigned                nof_rx_ports        = 4;
  unsigned                nof_tx_ports        = 4;
  std::array<unsigned, 4> rx_ports            = {0, 1, 2, 3};
  subcarrier_spacing      scs                 = subcarrier_spacing::kHz30;
  unsigned                nof_symbols         = 4;
  unsigned                start_symbol        = 0;
  tx_comb_size            comb_size           = tx_comb_size::n4;
  unsigned                comb_offset         = 0;
  unsigned                cyclic_shift        = 0;
  unsigned                configuration_index = 63;
  unsigned                bandwidth_index     = 0;
  unsigned                freq_position       = 0;
  unsigned                freq_shift          = 0;
  unsigned                sequence_id         = 0;
};

inline srs_resource_configuration::one_two_four_enum to_srs_ports(unsigned nof_tx_ports)
{
  switch (nof_tx_ports) {
    case 1:
      return srs_resource_configuration::one_two_four_enum::one;
    case 2:
      return srs_resource_configuration::one_two_four_enum::two;
    case 4:
      return srs_resource_configuration::one_two_four_enum::four;
    default:
      report_fatal_error("SRS benchmark only supports 1, 2 or 4 TX ports.");
  }
}

inline srs_estimator_configuration make_srs_config(unsigned           nof_rx_ports,
                                                   unsigned           nof_tx_ports,
                                                   unsigned           nof_symbols,
                                                   unsigned           start_symbol,
                                                   subcarrier_spacing scs,
                                                   tx_comb_size       comb_size           = tx_comb_size::n4,
                                                   unsigned           comb_offset         = 0,
                                                   unsigned           cyclic_shift        = 0,
                                                   unsigned           configuration_index = 63,
                                                   unsigned           bandwidth_index     = 0,
                                                   unsigned           freq_position       = 0,
                                                   unsigned           freq_shift          = 0,
                                                   unsigned           sequence_id         = 0)
{
  srs_resource_configuration srs_resource;
  srs_resource.nof_antenna_ports   = to_srs_ports(nof_tx_ports);
  srs_resource.nof_symbols         = static_cast<srs_nof_symbols>(nof_symbols);
  srs_resource.start_symbol        = start_symbol;
  srs_resource.configuration_index = configuration_index;
  srs_resource.sequence_id         = sequence_id;
  srs_resource.bandwidth_index     = bandwidth_index;
  srs_resource.comb_size           = comb_size;
  srs_resource.comb_offset         = comb_offset;
  srs_resource.cyclic_shift        = cyclic_shift;
  srs_resource.freq_position       = freq_position;
  srs_resource.freq_shift          = freq_shift;
  srs_resource.freq_hopping        = bandwidth_index;
  srs_resource.hopping             = srs_group_or_sequence_hopping::neither;
  srs_resource.periodicity         = srs_resource_configuration::periodicity_and_offset{0, 0};

  srs_estimator_configuration config;
  config.slot     = slot_point(to_numerology_value(scs), 0);
  config.resource = srs_resource;
  config.ports.resize(nof_rx_ports);
  std::iota(config.ports.begin(), config.ports.end(), 0);
  return config;
}

inline srs_estimator_configuration make_srs_config(const srs_benchmark_profile& profile)
{
  srs_estimator_configuration config = make_srs_config(profile.nof_rx_ports,
                                                       profile.nof_tx_ports,
                                                       profile.nof_symbols,
                                                       profile.start_symbol,
                                                       profile.scs,
                                                       profile.comb_size,
                                                       profile.comb_offset,
                                                       profile.cyclic_shift,
                                                       profile.configuration_index,
                                                       profile.bandwidth_index,
                                                       profile.freq_position,
                                                       profile.freq_shift,
                                                       profile.sequence_id);
  for (unsigned i_rx_port = 0; i_rx_port != profile.nof_rx_ports; ++i_rx_port) {
    config.ports[i_rx_port] = profile.rx_ports[i_rx_port];
  }
  return config;
}

inline unsigned get_required_grid_ports(const srs_benchmark_profile& profile)
{
  unsigned nof_ports = 0;
  for (unsigned i_rx_port = 0; i_rx_port != profile.nof_rx_ports; ++i_rx_port) {
    nof_ports = std::max(nof_ports, profile.rx_ports[i_rx_port] + 1);
  }
  return nof_ports;
}

inline const char* comb_to_string(tx_comb_size value)
{
  return value == tx_comb_size::n2 ? "n2" : "n4";
}

inline std::string describe_profile(const srs_benchmark_profile& profile)
{
  std::string rx_ports;
  for (unsigned i_rx_port = 0; i_rx_port != profile.nof_rx_ports; ++i_rx_port) {
    rx_ports += (i_rx_port == 0) ? "" : ",";
    rx_ports += std::to_string(profile.rx_ports[i_rx_port]);
  }

  return fmt::format("{} rx={} [{}] tx={} scs={} sym={} start={} comb={} off={} cs={} cSRS={} bSRS={} fpos={} "
                     "fshift={} seq={}",
                     profile.name,
                     profile.nof_rx_ports,
                     rx_ports,
                     profile.nof_tx_ports,
                     to_string(profile.scs),
                     profile.nof_symbols,
                     profile.start_symbol,
                     comb_to_string(profile.comb_size),
                     profile.comb_offset,
                     profile.cyclic_shift,
                     profile.configuration_index,
                     profile.bandwidth_index,
                     profile.freq_position,
                     profile.freq_shift,
                     profile.sequence_id);
}

inline std::vector<srs_benchmark_profile> make_supported_srs_profiles(bool full)
{
  std::vector<srs_benchmark_profile> profiles = {
      {"baseline_4x4_n4", 4, 4, {0, 1, 2, 3}, subcarrier_spacing::kHz30, 4, 0, tx_comb_size::n4, 0, 0, 63, 0, 0, 0, 0},
      {"n2_1x1_phys3", 1, 1, {3, 0, 0, 0}, subcarrier_spacing::kHz15, 1, 13, tx_comb_size::n2, 1, 3, 0, 0, 0, 0, 11},
      {"n2_2x2_sparse_rx",
       2,
       2,
       {1, 3, 0, 0},
       subcarrier_spacing::kHz60,
       2,
       12,
       tx_comb_size::n2,
       1,
       7,
       14,
       2,
       3,
       0,
       48},
      {"n4_4x2_shifted",
       4,
       2,
       {0, 1, 2, 3},
       subcarrier_spacing::kHz120,
       4,
       10,
       tx_comb_size::n4,
       2,
       6,
       31,
       2,
       5,
       1,
       65},
      {"n4_2x4_interleaved",
       2,
       4,
       {1, 3, 0, 0},
       subcarrier_spacing::kHz30,
       4,
       10,
       tx_comb_size::n4,
       3,
       11,
       63,
       3,
       1,
       0,
       114},
  };

  if (!full) {
    return profiles;
  }

  profiles.insert(
      profiles.end(),
      {{"n2_4x1_wide", 4, 1, {0, 1, 2, 3}, subcarrier_spacing::kHz30, 4, 10, tx_comb_size::n2, 0, 4, 53, 1, 9, 0, 70},
       {"n4_1x4_phys2", 1, 4, {2, 0, 0, 0}, subcarrier_spacing::kHz15, 2, 12, tx_comb_size::n4, 1, 5, 7, 1, 1, 0, 24},
       {"n2_4x4_120khz", 4, 4, {0, 1, 2, 3}, subcarrier_spacing::kHz120, 4, 0, tx_comb_size::n2, 0, 0, 36, 3, 7, 0, 87},
       {"n4_2x1_sparse_rx",
        2,
        1,
        {1, 3, 0, 0},
        subcarrier_spacing::kHz60,
        1,
        13,
        tx_comb_size::n4,
        0,
        0,
        0,
        0,
        0,
        0,
        3}});

  return profiles;
}

inline std::optional<srs_benchmark_profile> find_supported_srs_profile(std::string_view name)
{
  for (const srs_benchmark_profile& profile : make_supported_srs_profiles(true)) {
    if (name == profile.name) {
      return profile;
    }
  }

  return std::nullopt;
}

inline std::string supported_srs_profile_names()
{
  std::string result;
  for (const srs_benchmark_profile& profile : make_supported_srs_profiles(true)) {
    result += result.empty() ? "" : ", ";
    result += profile.name;
  }
  return result;
}

inline bool srs_benchmark_env_enabled(const char* name)
{
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }

  return (std::strcmp(value, "1") == 0) || (std::strcmp(value, "true") == 0) || (std::strcmp(value, "TRUE") == 0) ||
         (std::strcmp(value, "on") == 0) || (std::strcmp(value, "ON") == 0);
}

inline unsigned get_benchmark_gpu_min_pilot_re()
{
  const char* value = std::getenv("OCUDU_SRS_ACCELERATION_MIN_PILOT_RE");
  if (value == nullptr) {
    return 4096;
  }

  return static_cast<unsigned>(std::strtoul(value, nullptr, 10));
}

inline unsigned get_observed_pilot_re(const srs_estimator_configuration& config)
{
  unsigned sequence_length = get_srs_information(config.resource, 0).sequence_length;
  unsigned nof_symbols     = static_cast<unsigned>(config.resource.nof_symbols);
  return static_cast<unsigned>(config.ports.size()) * nof_symbols * sequence_length;
}

inline bool predicts_gpu_path(const srs_estimator_configuration& config)
{
  if (srs_benchmark_env_enabled("OCUDU_SRS_ACCELERATION_FORCE")) {
    return true;
  }

  unsigned min_pilot_re = get_benchmark_gpu_min_pilot_re();
  return (min_pilot_re == 0) || (get_observed_pilot_re(config) >= min_pilot_re);
}

inline std::unique_ptr<resource_grid>
create_grid(unsigned nof_ports, unsigned nof_symbols, unsigned nof_subc, grid_mode mode)
{
#ifdef ENABLE_CUDA
  if (mode == grid_mode::visible) {
    resource_grid_cuda_visible_factory factory(resource_grid_cuda_visible_factory::direction::uplink);
    return factory.create(nof_ports, nof_symbols, nof_subc);
  }
#else
  (void)mode;
#endif

  std::shared_ptr<resource_grid_factory> rg_factory = create_resource_grid_factory();
  TESTASSERT(rg_factory != nullptr, "Invalid resource grid factory.");
  return rg_factory->create(nof_ports, nof_symbols, nof_subc);
}

inline std::shared_ptr<srs_estimator_factory> make_srs_estimator_factory(const std::string& gpu_mode)
{
  std::shared_ptr<low_papr_sequence_generator_factory> low_papr_seq_gen_factory =
      create_low_papr_sequence_generator_sw_factory();
  TESTASSERT(low_papr_seq_gen_factory);

  std::shared_ptr<dft_processor_factory> dft_factory = create_dft_processor_factory_fftw_slow();
  if (!dft_factory) {
    dft_factory = create_dft_processor_factory_generic();
  }
  TESTASSERT(dft_factory);

  std::shared_ptr<time_alignment_estimator_factory> ta_est_factory =
      create_time_alignment_estimator_dft_factory(dft_factory);
  TESTASSERT(ta_est_factory);

  return create_srs_estimator_generic_factory(low_papr_seq_gen_factory, ta_est_factory, MAX_NOF_PRBS, gpu_mode);
}

inline estimator_pair make_estimators()
{
  std::shared_ptr<srs_estimator_factory> cpu_srs_est_factory = make_srs_estimator_factory("disabled");
  TESTASSERT(cpu_srs_est_factory);
  std::shared_ptr<srs_estimator_factory> gpu_srs_est_factory = make_srs_estimator_factory("auto");
  TESTASSERT(gpu_srs_est_factory);

  estimator_pair result;
  result.cpu = cpu_srs_est_factory->create();
  result.gpu = gpu_srs_est_factory->create();

  TESTASSERT(result.cpu);
  TESTASSERT(result.gpu);
  return result;
}

inline cf_t make_channel_coefficient(unsigned i_rx_port, unsigned i_tx_port)
{
  float amplitude = 0.45F + 0.07F * static_cast<float>(i_rx_port + 1) + 0.05F * static_cast<float>(i_tx_port + 1);
  float phase     = 0.37F * static_cast<float>(i_rx_port + 1) - 0.29F * static_cast<float>(i_tx_port + 1);
  return std::polar(amplitude, phase);
}

inline void
populate_srs_grid(resource_grid& grid, const srs_estimator_configuration& config, float snr_db, unsigned seed)
{
  grid.set_all_zero();

  unsigned nof_rx_ports       = config.ports.size();
  unsigned nof_tx_ports       = static_cast<unsigned>(config.resource.nof_antenna_ports);
  unsigned nof_symbols        = static_cast<unsigned>(config.resource.nof_symbols);
  unsigned start_symbol       = config.resource.start_symbol.value();
  unsigned grid_nof_symbols   = grid.get_reader().get_nof_symbols();
  unsigned grid_nof_subc      = grid.get_reader().get_nof_subc();
  unsigned sequence_length    = get_srs_information(config.resource, 0).sequence_length;
  auto     sequence_generator = create_low_papr_sequence_generator_sw_factory()->create();
  TESTASSERT(sequence_generator);

  std::vector<std::vector<cf_t>> sequences(nof_tx_ports, std::vector<cf_t>(sequence_length));
  std::vector<srs_information>   info_per_tx;
  info_per_tx.reserve(nof_tx_ports);

  for (unsigned i_tx_port = 0; i_tx_port != nof_tx_ports; ++i_tx_port) {
    srs_information info = get_srs_information(config.resource, i_tx_port);
    info_per_tx.push_back(info);
    sequence_generator->generate(
        span<cf_t>(sequences[i_tx_port]), info.sequence_group, info.sequence_number, info.n_cs, info.n_cs_max);
  }

  std::vector<std::vector<cf_t>>    symbols(nof_rx_ports * grid_nof_symbols, std::vector<cf_t>(grid_nof_subc, cf_t()));
  std::vector<std::vector<uint8_t>> active(nof_rx_ports * grid_nof_symbols, std::vector<uint8_t>(grid_nof_subc, 0));

  double   signal_power_sum = 0.0;
  unsigned active_count     = 0;

  for (unsigned i_rx_port = 0; i_rx_port != nof_rx_ports; ++i_rx_port) {
    for (unsigned i_symbol = start_symbol; i_symbol != start_symbol + nof_symbols; ++i_symbol) {
      std::vector<cf_t>&    symbol = symbols[i_rx_port * grid_nof_symbols + i_symbol];
      std::vector<uint8_t>& mask   = active[i_rx_port * grid_nof_symbols + i_symbol];

      for (unsigned i_tx_port = 0; i_tx_port != nof_tx_ports; ++i_tx_port) {
        const srs_information& info    = info_per_tx[i_tx_port];
        cf_t                   channel = make_channel_coefficient(i_rx_port, i_tx_port);
        for (unsigned i_re = 0; i_re != sequence_length; ++i_re) {
          unsigned k = info.mapping_initial_subcarrier + i_re * info.comb_size;
          symbol[k] += channel * sequences[i_tx_port][i_re];
          mask[k] = 1;
        }
      }

      for (unsigned k = 0; k != grid_nof_subc; ++k) {
        if (mask[k] != 0) {
          signal_power_sum += std::norm(symbol[k]);
          ++active_count;
        }
      }
    }
  }

  float signal_power = active_count != 0 ? static_cast<float>(signal_power_sum / active_count) : 1.0F;
  float noise_var    = signal_power / convert_dB_to_power(snr_db);
  float noise_sigma  = std::sqrt(noise_var / 2.0F);

  std::mt19937                    rgen(seed);
  std::normal_distribution<float> normal_dist(0.0F, noise_sigma);
  resource_grid_writer&           writer = grid.get_writer();
  for (unsigned i_rx_port = 0; i_rx_port != nof_rx_ports; ++i_rx_port) {
    unsigned physical_rx_port = config.ports[i_rx_port];
    for (unsigned i_symbol = 0; i_symbol != grid_nof_symbols; ++i_symbol) {
      span<cbf16_t>               view   = writer.get_view(physical_rx_port, i_symbol);
      const std::vector<cf_t>&    symbol = symbols[i_rx_port * grid_nof_symbols + i_symbol];
      const std::vector<uint8_t>& mask   = active[i_rx_port * grid_nof_symbols + i_symbol];
      for (unsigned k = 0; k != grid_nof_subc; ++k) {
        cf_t value = symbol[k];
        if (mask[k] != 0) {
          value += cf_t(normal_dist(rgen), normal_dist(rgen));
        }
        view[k] = to_cbf16(value);
      }
    }
  }
}

inline latency_stats summarize_latencies(std::vector<double> samples_us)
{
  TESTASSERT(!samples_us.empty(), "No latency samples were collected.");
  std::sort(samples_us.begin(), samples_us.end());

  auto percentile = [&samples_us](double pct) {
    double index = pct * static_cast<double>(samples_us.size() - 1);
    return samples_us[static_cast<size_t>(std::round(index))];
  };

  latency_stats stats;
  stats.mean_us = std::accumulate(samples_us.begin(), samples_us.end(), 0.0) / static_cast<double>(samples_us.size());
  stats.p50_us  = percentile(0.50);
  stats.p95_us  = percentile(0.95);
  stats.min_us  = samples_us.front();
  stats.max_us  = samples_us.back();
  return stats;
}

template <typename Func>
inline std::vector<double> measure_us(unsigned repetitions, Func&& func)
{
  std::vector<double> samples;
  samples.reserve(repetitions);
  for (unsigned i = 0; i != repetitions; ++i) {
    auto start = std::chrono::steady_clock::now();
    func();
    auto stop = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
  }
  return samples;
}

inline float channel_matrix_relative_error(const srs_channel_matrix& reference, const srs_channel_matrix& candidate)
{
  TESTASSERT(reference.get_nof_rx_ports() == candidate.get_nof_rx_ports());
  TESTASSERT(reference.get_nof_tx_ports() == candidate.get_nof_tx_ports());

  double error_power = 0.0;
  double ref_power   = 0.0;
  for (unsigned i_rx_port = 0; i_rx_port != reference.get_nof_rx_ports(); ++i_rx_port) {
    for (unsigned i_tx_port = 0; i_tx_port != reference.get_nof_tx_ports(); ++i_tx_port) {
      cf_t ref  = reference.get_coefficient(i_rx_port, i_tx_port);
      cf_t cand = candidate.get_coefficient(i_rx_port, i_tx_port);
      error_power += std::norm(ref - cand);
      ref_power += std::norm(ref);
    }
  }

  return (ref_power > 0.0) ? static_cast<float>(std::sqrt(error_power / ref_power)) : 0.0F;
}

inline float optional_delta(std::optional<float> reference, std::optional<float> candidate)
{
  if (!reference.has_value() || !candidate.has_value()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return candidate.value() - reference.value();
}

inline const char* to_string(grid_mode mode)
{
  return mode == grid_mode::visible ? "visible" : "host";
}

inline grid_mode parse_grid_mode(const char* value)
{
  std::string mode(value);
  if (mode == "visible") {
    return grid_mode::visible;
  }
  if (mode == "host") {
    return grid_mode::host;
  }
  report_fatal_error("Unsupported grid mode '{}'. Use 'visible' or 'host'.", mode);
}

} // namespace ocudu::srs_gpu_bench
