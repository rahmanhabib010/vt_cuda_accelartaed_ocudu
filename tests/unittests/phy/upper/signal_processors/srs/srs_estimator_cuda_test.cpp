// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "phy_acceleration_resource_grid_factory.h"
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
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ocudu;

namespace {

enum class grid_mode { host, cuda_visible };

struct srs_test_profile {
  unsigned                nof_rx_ports        = 1;
  unsigned                nof_tx_ports        = 1;
  std::array<unsigned, 4> rx_ports            = {0, 1, 2, 3};
  subcarrier_spacing      scs                 = subcarrier_spacing::kHz30;
  unsigned                nof_symbols         = 1;
  unsigned                start_symbol        = 13;
  tx_comb_size            comb_size           = tx_comb_size::n4;
  unsigned                comb_offset         = 0;
  unsigned                cyclic_shift        = 0;
  unsigned                configuration_index = 63;
  unsigned                bandwidth_index     = 0;
  unsigned                freq_position       = 0;
  unsigned                freq_shift          = 0;
  unsigned                sequence_id         = 0;
};

srs_resource_configuration::one_two_four_enum to_srs_ports(unsigned nof_tx_ports)
{
  switch (nof_tx_ports) {
    case 1:
      return srs_resource_configuration::one_two_four_enum::one;
    case 2:
      return srs_resource_configuration::one_two_four_enum::two;
    case 4:
      return srs_resource_configuration::one_two_four_enum::four;
    default:
      report_fatal_error("Unsupported SRS port count {}.", nof_tx_ports);
  }
}

srs_estimator_configuration make_srs_config(const srs_test_profile& profile)
{
  srs_resource_configuration resource;
  resource.nof_antenna_ports   = to_srs_ports(profile.nof_tx_ports);
  resource.nof_symbols         = static_cast<srs_nof_symbols>(profile.nof_symbols);
  resource.start_symbol        = profile.start_symbol;
  resource.configuration_index = profile.configuration_index;
  resource.sequence_id         = profile.sequence_id;
  resource.bandwidth_index     = profile.bandwidth_index;
  resource.comb_size           = profile.comb_size;
  resource.comb_offset         = profile.comb_offset;
  resource.cyclic_shift        = profile.cyclic_shift;
  resource.freq_position       = profile.freq_position;
  resource.freq_shift          = profile.freq_shift;
  resource.freq_hopping        = profile.bandwidth_index;
  resource.hopping             = srs_group_or_sequence_hopping::neither;
  resource.periodicity         = srs_resource_configuration::periodicity_and_offset{0, 0};

  srs_estimator_configuration config;
  config.slot     = slot_point(to_numerology_value(profile.scs), 0);
  config.resource = resource;
  config.ports.resize(profile.nof_rx_ports);
  for (unsigned i_rx_port = 0; i_rx_port != profile.nof_rx_ports; ++i_rx_port) {
    config.ports[i_rx_port] = profile.rx_ports[i_rx_port];
  }
  return config;
}

std::shared_ptr<srs_estimator_factory> create_test_srs_factory(const std::string& acceleration_mode)
{
  std::shared_ptr<low_papr_sequence_generator_factory> sequence_factory =
      create_low_papr_sequence_generator_sw_factory();
  EXPECT_NE(sequence_factory, nullptr);

  std::shared_ptr<dft_processor_factory> dft_factory = create_dft_processor_factory_fftw_slow();
  if (!dft_factory) {
    dft_factory = create_dft_processor_factory_generic();
  }
  EXPECT_NE(dft_factory, nullptr);

  std::shared_ptr<time_alignment_estimator_factory> ta_factory =
      create_time_alignment_estimator_dft_factory(dft_factory);
  EXPECT_NE(ta_factory, nullptr);

  return create_srs_estimator_generic_factory(sequence_factory, ta_factory, MAX_NOF_PRBS, acceleration_mode);
}

unsigned get_required_grid_ports(const srs_estimator_configuration& config)
{
  unsigned nof_ports = 0;
  for (unsigned port : config.ports) {
    nof_ports = std::max(nof_ports, port + 1U);
  }
  return nof_ports;
}

std::unique_ptr<resource_grid> create_test_grid(const srs_estimator_configuration& config, grid_mode mode)
{
  std::shared_ptr<resource_grid_factory> fallback_factory = create_resource_grid_factory();
  EXPECT_NE(fallback_factory, nullptr);

  if (mode == grid_mode::cuda_visible) {
    std::shared_ptr<resource_grid_factory> accelerated_factory = create_phy_acceleration_resource_grid_factory(
        fallback_factory, phy_acceleration_resource_grid_direction::uplink, true);
    EXPECT_NE(accelerated_factory, nullptr);
    return accelerated_factory->create(
        get_required_grid_ports(config), get_nsymb_per_slot(cyclic_prefix::NORMAL), MAX_NOF_SUBCARRIERS);
  }

  return fallback_factory->create(
      get_required_grid_ports(config), get_nsymb_per_slot(cyclic_prefix::NORMAL), MAX_NOF_SUBCARRIERS);
}

cf_t make_channel_coefficient(unsigned i_rx_port, unsigned i_tx_port)
{
  float amplitude = 0.45F + 0.07F * static_cast<float>(i_rx_port + 1) + 0.05F * static_cast<float>(i_tx_port + 1);
  float phase     = 0.37F * static_cast<float>(i_rx_port + 1) - 0.29F * static_cast<float>(i_tx_port + 1);
  return std::polar(amplitude, phase);
}

srs_channel_matrix make_expected_channel_matrix(unsigned nof_rx_ports, unsigned nof_tx_ports)
{
  srs_channel_matrix matrix(nof_rx_ports, nof_tx_ports);
  for (unsigned i_rx_port = 0; i_rx_port != nof_rx_ports; ++i_rx_port) {
    for (unsigned i_tx_port = 0; i_tx_port != nof_tx_ports; ++i_tx_port) {
      matrix.set_coefficient(make_channel_coefficient(i_rx_port, i_tx_port), i_rx_port, i_tx_port);
    }
  }
  return matrix;
}

void populate_srs_grid(resource_grid& grid, const srs_estimator_configuration& config, float snr_db, unsigned seed)
{
  grid.set_all_zero();

  unsigned nof_rx_ports     = config.ports.size();
  unsigned nof_tx_ports     = static_cast<unsigned>(config.resource.nof_antenna_ports);
  unsigned nof_symbols      = static_cast<unsigned>(config.resource.nof_symbols);
  unsigned start_symbol     = config.resource.start_symbol.value();
  unsigned grid_nof_symbols = grid.get_reader().get_nof_symbols();
  unsigned grid_nof_subc    = grid.get_reader().get_nof_subc();
  unsigned sequence_length  = get_srs_information(config.resource, 0).sequence_length;

  std::unique_ptr<low_papr_sequence_generator> sequence_generator =
      create_low_papr_sequence_generator_sw_factory()->create();
  ASSERT_NE(sequence_generator, nullptr);

  std::vector<std::vector<cf_t>> sequences(nof_tx_ports, std::vector<cf_t>(sequence_length));
  std::vector<srs_information>   info_per_tx;
  info_per_tx.reserve(nof_tx_ports);

  for (unsigned i_tx_port = 0; i_tx_port != nof_tx_ports; ++i_tx_port) {
    srs_information info = get_srs_information(config.resource, i_tx_port);
    ASSERT_LE(info.mapping_initial_subcarrier + (info.sequence_length - 1U) * info.comb_size, grid_nof_subc - 1U);
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

  float signal_power = (active_count != 0) ? static_cast<float>(signal_power_sum / active_count) : 1.0F;
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

float channel_matrix_relative_error(const srs_channel_matrix& reference, const srs_channel_matrix& candidate)
{
  EXPECT_EQ(reference.get_nof_rx_ports(), candidate.get_nof_rx_ports());
  EXPECT_EQ(reference.get_nof_tx_ports(), candidate.get_nof_tx_ports());

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

float relative_delta(float reference, float candidate)
{
  float denominator = std::max(std::abs(reference), 1e-12F);
  return std::abs(candidate - reference) / denominator;
}

void assert_srs_results_match(const srs_estimator_result& cpu_result,
                              const srs_estimator_result& gpu_result,
                              const srs_channel_matrix&   expected_channel)
{
  EXPECT_LT(channel_matrix_relative_error(cpu_result.channel_matrix, gpu_result.channel_matrix), 6.0e-2F);
  EXPECT_TRUE(cpu_result.channel_matrix.is_near(expected_channel, 8.0e-2F));
  EXPECT_TRUE(gpu_result.channel_matrix.is_near(expected_channel, 8.0e-2F));

  ASSERT_TRUE(cpu_result.epre_dB.has_value());
  ASSERT_TRUE(gpu_result.epre_dB.has_value());
  EXPECT_NEAR(cpu_result.epre_dB.value(), gpu_result.epre_dB.value(), 2.0e-2F);

  ASSERT_TRUE(cpu_result.rsrp_dB.has_value());
  ASSERT_TRUE(gpu_result.rsrp_dB.has_value());
  EXPECT_NEAR(cpu_result.rsrp_dB.value(), gpu_result.rsrp_dB.value(), 2.0e-2F);

  ASSERT_TRUE(cpu_result.noise_variance.has_value());
  ASSERT_TRUE(gpu_result.noise_variance.has_value());
  EXPECT_LT(relative_delta(cpu_result.noise_variance.value(), gpu_result.noise_variance.value()), 1.5e-1F);

  EXPECT_NEAR(cpu_result.time_alignment.time_alignment, gpu_result.time_alignment.time_alignment, 5.0e-9);
}

void run_srs_gpu_correctness_case(const srs_test_profile& profile, grid_mode mode)
{
  if (!is_srs_estimator_acceleration_available()) {
    GTEST_SKIP() << "CUDA SRS estimator is not available.";
  }

  srs_estimator_configuration    config = make_srs_config(profile);
  std::unique_ptr<resource_grid> grid   = create_test_grid(config, mode);
  ASSERT_NE(grid, nullptr);
  if (mode == grid_mode::cuda_visible) {
    ASSERT_TRUE(grid->get_reader().supports_device_grid_reading());
  }

  populate_srs_grid(*grid, config, 45.0F, 0x5eed1234U + profile.nof_rx_ports * 17U + profile.nof_tx_ports * 101U);

  std::shared_ptr<srs_estimator_factory> cpu_factory = create_test_srs_factory("disabled");
  std::shared_ptr<srs_estimator_factory> gpu_factory = create_test_srs_factory("enabled");
  ASSERT_NE(cpu_factory, nullptr);
  ASSERT_NE(gpu_factory, nullptr);

  std::unique_ptr<srs_estimator> cpu_estimator = cpu_factory->create();
  std::unique_ptr<srs_estimator> gpu_estimator = gpu_factory->create();
  ASSERT_NE(cpu_estimator, nullptr);
  ASSERT_NE(gpu_estimator, nullptr);

  srs_estimator_result cpu_result = cpu_estimator->estimate(grid->get_reader(), config);
  srs_estimator_result gpu_result = gpu_estimator->estimate(grid->get_reader(), config);
  assert_srs_results_match(
      cpu_result, gpu_result, make_expected_channel_matrix(profile.nof_rx_ports, profile.nof_tx_ports));
}

} // namespace

TEST(srs_estimator_cuda_test, forced_gpu_host_grid_matches_cpu_and_known_channel)
{
  srs_test_profile profile;
  profile.nof_rx_ports        = 1;
  profile.nof_tx_ports        = 1;
  profile.rx_ports            = {0, 1, 2, 3};
  profile.scs                 = subcarrier_spacing::kHz30;
  profile.nof_symbols         = 4;
  profile.start_symbol        = 10;
  profile.comb_size           = tx_comb_size::n2;
  profile.comb_offset         = 1;
  profile.cyclic_shift        = 3;
  profile.configuration_index = 0;
  profile.bandwidth_index     = 0;
  profile.freq_position       = 0;
  profile.freq_shift          = 0;
  profile.sequence_id         = 0;

  run_srs_gpu_correctness_case(profile, grid_mode::host);
}

TEST(srs_estimator_cuda_test, forced_gpu_cuda_visible_grid_matches_cpu_for_sparse_rx_ports)
{
  srs_test_profile profile;
  profile.nof_rx_ports        = 1;
  profile.nof_tx_ports        = 1;
  profile.rx_ports            = {3, 0, 0, 0};
  profile.scs                 = subcarrier_spacing::kHz15;
  profile.nof_symbols         = 1;
  profile.start_symbol        = 13;
  profile.comb_size           = tx_comb_size::n2;
  profile.comb_offset         = 1;
  profile.cyclic_shift        = 3;
  profile.configuration_index = 0;
  profile.bandwidth_index     = 0;
  profile.freq_position       = 0;
  profile.freq_shift          = 0;
  profile.sequence_id         = 11;

  run_srs_gpu_correctness_case(profile, grid_mode::cuda_visible);
}
