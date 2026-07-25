// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "../../../../support/resource_grid_test_doubles.h"
#include "pdxch_baseband_modulator_accelerator.h"
#include "ocudu/gateways/baseband/buffer/baseband_gateway_buffer_dynamic.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/phy/lower/amplitude_controller/amplitude_controller_factories.h"
#include "ocudu/phy/lower/lower_phy_baseband_metrics.h"
#include "ocudu/phy/lower/modulation/modulation_factories.h"
#include "ocudu/phy/lower/sampling_rate.h"
#include "ocudu/phy/support/support_factories.h"
#include "ocudu/ran/cyclic_prefix.h"
#include "ocudu/ran/resource_block.h"
#include "ocudu/ran/subcarrier_spacing.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace ocudu;

namespace {

std::shared_ptr<dft_processor_factory> make_dft_factory()
{
  std::shared_ptr<dft_processor_factory> dft_factory = create_dft_processor_factory_fftw_fast();
  if (dft_factory == nullptr) {
    dft_factory = create_dft_processor_factory_generic();
  }
  return dft_factory;
}

std::vector<unsigned> make_symbol_sizes(sampling_rate srate, subcarrier_spacing scs, cyclic_prefix cp)
{
  const unsigned        nof_symbols = get_nsymb_per_slot(cp);
  std::vector<unsigned> symbol_sizes(nof_symbols);
  for (unsigned symbol = 0; symbol != nof_symbols; ++symbol) {
    symbol_sizes[symbol] = srate.get_dft_size(scs) + cp.get_length(symbol, scs).to_samples(srate.to_Hz());
  }
  return symbol_sizes;
}

lower_phy_baseband_metrics compute_output_metrics(span<const ci16_t> samples, span<const unsigned> symbol_sizes)
{
  static constexpr float ci16_to_cf_scale   = 1.0F / static_cast<float>(std::numeric_limits<int16_t>::max());
  static constexpr float clipping_threshold = 0.95F;

  lower_phy_baseband_metrics metrics = {};
  metrics.clipping                   = clipping_counters{.nof_clipped_samples = 0, .nof_processed_samples = 0};
  double   avg_power_acc             = 0.0;
  unsigned nof_avg_terms             = 0;

  unsigned sample_offset = 0;
  for (unsigned symbol_size : symbol_sizes) {
    double symbol_power = 0.0;
    for (unsigned i_sample = 0; i_sample != symbol_size; ++i_sample) {
      const ci16_t sample = samples[sample_offset + i_sample];
      const float  re     = static_cast<float>(sample.real()) * ci16_to_cf_scale;
      const float  im     = static_cast<float>(sample.imag()) * ci16_to_cf_scale;
      const float  power  = re * re + im * im;
      symbol_power += power;
      metrics.peak_power = std::max(metrics.peak_power, power);
      metrics.clipping->nof_clipped_samples +=
          (std::abs(re) > clipping_threshold || std::abs(im) > clipping_threshold) ? 1U : 0U;
    }
    avg_power_acc += symbol_power / static_cast<double>(symbol_size);
    ++nof_avg_terms;
    metrics.clipping->nof_processed_samples += symbol_size;
    sample_offset += symbol_size;
  }

  metrics.avg_power = static_cast<float>(avg_power_acc / static_cast<double>(nof_avg_terms));
  return metrics;
}

void fill_deterministic_grid(resource_grid_reader_spy& grid_reader,
                             unsigned                  nof_ports,
                             unsigned                  nof_symbols,
                             unsigned                  rg_size)
{
  for (unsigned port = 0; port != nof_ports; ++port) {
    for (unsigned symbol = 0; symbol != nof_symbols; ++symbol) {
      for (unsigned subcarrier = 0; subcarrier != rg_size; ++subcarrier) {
        resource_grid_reader_spy::expected_entry_t entry = {};
        entry.port                                       = port;
        entry.symbol                                     = symbol;
        entry.subcarrier                                 = subcarrier;

        if ((subcarrier % 7) == ((port + symbol) % 7)) {
          const int re_seed = static_cast<int>((port + 3) * (symbol + 5) * (subcarrier + 11) % 29) - 14;
          const int im_seed = static_cast<int>((port + 7) * (symbol + 2) * (subcarrier + 17) % 31) - 15;
          entry.value       = cf_t(static_cast<float>(re_seed) * 4e-4F, static_cast<float>(im_seed) * 4e-4F);
        } else {
          entry.value = cf_t();
        }
        grid_reader.write(entry);
      }
    }
  }
}

} // namespace

TEST(pdxch_baseband_modulator_cuda, reports_metrics_from_completed_output_buffer)
{
  ocudulog::init();

  static constexpr unsigned nof_ports    = 1;
  static constexpr unsigned bandwidth_rb = 6;

  const sampling_rate      srate       = sampling_rate::from_MHz(3.84);
  const subcarrier_spacing scs         = subcarrier_spacing::kHz30;
  const cyclic_prefix      cp          = cyclic_prefix::NORMAL;
  const unsigned           nof_symbols = get_nsymb_per_slot(cp);
  const unsigned           rg_size     = bandwidth_rb * NOF_SUBCARRIERS_PER_RB;

  amplitude_controller_clipping_config amplitude_config = {};
  amplitude_config.enable_clipping                      = false;
  amplitude_config.input_gain_dB                        = 0.0F;
  amplitude_config.full_scale_lin                       = 1.0F;
  amplitude_config.ceiling_dBFS                         = 0.0F;

  auto accelerator = create_pdxch_baseband_modulator_accelerator_cuda(
      scs, cp, srate, bandwidth_rb, 3.5e9, nof_ports, amplitude_config);
  ASSERT_NE(accelerator, nullptr);

  resource_grid_reader_spy grid_reader(nof_ports, nof_symbols, bandwidth_rb);
  for (unsigned symbol = 0; symbol != nof_symbols; ++symbol) {
    for (unsigned subcarrier = 0; subcarrier != rg_size; ++subcarrier) {
      resource_grid_reader_spy::expected_entry_t entry = {};
      entry.port                                       = 0;
      entry.symbol                                     = symbol;
      entry.subcarrier                                 = subcarrier;
      entry.value = (subcarrier == rg_size / 2) ? cf_t(0.25F, 0.0F) : cf_t(0.0F, 0.0F);
      grid_reader.write(entry);
    }
  }

  std::vector<unsigned>           symbol_sizes = make_symbol_sizes(srate, scs, cp);
  const unsigned                  nof_samples  = std::accumulate(symbol_sizes.begin(), symbol_sizes.end(), 0U);
  baseband_gateway_buffer_dynamic output(nof_ports, nof_samples);
  output.resize(nof_samples);

  ASSERT_TRUE(accelerator->enqueue(output, grid_reader, 0, symbol_sizes));
  ASSERT_TRUE(accelerator->wait());

  const lower_phy_baseband_metrics metrics  = accelerator->collect_metrics();
  const lower_phy_baseband_metrics expected = compute_output_metrics(output[0], symbol_sizes);

  EXPECT_GT(metrics.avg_power, 0.0F);
  EXPECT_GT(metrics.peak_power, 0.0F);
  EXPECT_NEAR(metrics.avg_power, expected.avg_power, 1e-7F);
  EXPECT_NEAR(metrics.peak_power, expected.peak_power, 1e-7F);
  ASSERT_TRUE(metrics.clipping.has_value());
  ASSERT_TRUE(expected.clipping.has_value());
  EXPECT_EQ(metrics.clipping->nof_clipped_samples, expected.clipping->nof_clipped_samples);
  EXPECT_EQ(metrics.clipping->nof_processed_samples, expected.clipping->nof_processed_samples);
}

TEST(pdxch_baseband_modulator_cuda, live_20mhz_four_port_output_matches_cpu_modulator)
{
  ocudulog::init();

  static constexpr unsigned nof_ports      = 4;
  static constexpr unsigned bandwidth_rb   = 51;
  static constexpr unsigned slot_index_sf  = 1;
  static constexpr double   center_freq_Hz = 3.4101e9;

  const sampling_rate      srate             = sampling_rate::from_MHz(23.04);
  const subcarrier_spacing scs               = subcarrier_spacing::kHz30;
  const cyclic_prefix      cp                = cyclic_prefix::NORMAL;
  const unsigned           nof_symbols       = get_nsymb_per_slot(cp);
  const unsigned           rg_size           = bandwidth_rb * NOF_SUBCARRIERS_PER_RB;
  const unsigned           dft_size          = srate.get_dft_size(scs);
  const unsigned           i_symbol_sf_begin = slot_index_sf * nof_symbols;

  amplitude_controller_clipping_config amplitude_config = {};
  amplitude_config.enable_clipping                      = false;
  amplitude_config.input_gain_dB                        = 0.0F;
  amplitude_config.full_scale_lin                       = 1.0F;
  amplitude_config.ceiling_dBFS                         = 0.0F;

  auto accelerator = create_pdxch_baseband_modulator_accelerator_cuda(
      scs, cp, srate, bandwidth_rb, center_freq_Hz, nof_ports, amplitude_config);
  ASSERT_NE(accelerator, nullptr);

  resource_grid_reader_spy grid_reader(nof_ports, nof_symbols, bandwidth_rb);
  fill_deterministic_grid(grid_reader, nof_ports, nof_symbols, rg_size);

  std::vector<unsigned> symbol_sizes = make_symbol_sizes(srate, scs, cp);
  const unsigned        nof_samples  = std::accumulate(symbol_sizes.begin(), symbol_sizes.end(), 0U);

  baseband_gateway_buffer_dynamic gpu_output(nof_ports, nof_samples);
  gpu_output.resize(nof_samples);

  ASSERT_TRUE(accelerator->enqueue(gpu_output, grid_reader, i_symbol_sf_begin, symbol_sizes));
  ASSERT_TRUE(accelerator->wait());

  ofdm_factory_generic_configuration ofdm_config = {.dft_factory = make_dft_factory()};
  ASSERT_NE(ofdm_config.dft_factory, nullptr);
  std::shared_ptr<ofdm_modulator_factory> ofdm_factory = create_ofdm_modulator_factory_generic(ofdm_config);
  ASSERT_NE(ofdm_factory, nullptr);

  std::unique_ptr<ofdm_symbol_modulator> cpu_modulator =
      ofdm_factory->create_ofdm_symbol_modulator({.numerology     = to_numerology_value(scs),
                                                  .bw_rb          = bandwidth_rb,
                                                  .dft_size       = dft_size,
                                                  .cp             = cp,
                                                  .scale          = 1.0F,
                                                  .center_freq_Hz = center_freq_Hz});
  ASSERT_NE(cpu_modulator, nullptr);

  static constexpr float ci16_to_cf_scale = 1.0F / static_cast<float>(std::numeric_limits<int16_t>::max());
  unsigned               sample_offset    = 0;
  for (unsigned symbol = 0; symbol != nof_symbols; ++symbol) {
    std::vector<cf_t> cpu_symbol(symbol_sizes[symbol]);

    for (unsigned port = 0; port != nof_ports; ++port) {
      cpu_modulator->modulate(cpu_symbol, grid_reader, port, i_symbol_sf_begin + symbol);

      span<const ci16_t> gpu_symbol = gpu_output[port].subspan(sample_offset, symbol_sizes[symbol]);
      ASSERT_EQ(gpu_symbol.size(), cpu_symbol.size());
      for (unsigned sample = 0; sample != cpu_symbol.size(); ++sample) {
        const cf_t gpu_sample(static_cast<float>(gpu_symbol[sample].real()) * ci16_to_cf_scale,
                              static_cast<float>(gpu_symbol[sample].imag()) * ci16_to_cf_scale);
        EXPECT_NEAR(cpu_symbol[sample].real(), gpu_sample.real(), 1.5e-3F);
        EXPECT_NEAR(cpu_symbol[sample].imag(), gpu_sample.imag(), 1.5e-3F);
      }
    }
    sample_offset += symbol_sizes[symbol];
  }
}
