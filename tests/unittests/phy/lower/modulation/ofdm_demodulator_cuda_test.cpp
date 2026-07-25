// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "resource_grid_cuda_visible_impl.h"
#include "ocudu/phy/lower/modulation/modulation_factories.h"
#include "ocudu/phy/support/support_factories.h"
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>

using namespace ocudu;

namespace {

bool cuda_available()
{
  int device_count = 0;
  if ((cudaGetDeviceCount(&device_count) != cudaSuccess) || (device_count == 0)) {
    cudaGetLastError();
    return false;
  }
  return true;
}

std::shared_ptr<dft_processor_factory> make_dft_factory()
{
  std::shared_ptr<dft_processor_factory> dft_factory = create_dft_processor_factory_fftw_fast();
  if (dft_factory == nullptr) {
    dft_factory = create_dft_processor_factory_generic();
  }
  return dft_factory;
}

void run_ci16_fast_path_match(unsigned window_offset)
{
  ASSERT_EQ(setenv("OCUDU_UL_CUDA_VISIBLE_GRID", "managed", 1), 0);
  ASSERT_EQ(setenv("OCUDU_LOWPHY_RX_ACCELERATION", "enabled", 1), 0);
  ASSERT_EQ(setenv("OCUDU_LOWPHY_RX_CUDA_GRAPHS", "1", 1), 0);

  static constexpr float input_scale = std::numeric_limits<int16_t>::max();

  ofdm_demodulator_configuration config = {};
  config.numerology                     = 1;
  config.bw_rb                          = 52;
  config.dft_size                       = 1024;
  config.cp                             = cyclic_prefix::NORMAL;
  config.nof_samples_window_offset      = window_offset;
  config.scale                          = 1.0F / std::sqrt(static_cast<float>(config.bw_rb * NOF_SUBCARRIERS_PER_RB));
  config.center_freq_Hz                 = 0.0;

  unsigned           symbol_index = 1;
  subcarrier_spacing scs          = to_subcarrier_spacing(config.numerology);
  unsigned           srate_Hz     = to_sampling_rate_Hz(scs, config.dft_size);
  unsigned           cp_len       = config.cp.get_length(symbol_index, scs).to_samples(srate_Hz);
  ASSERT_GE(cp_len, window_offset);

  std::vector<ci16_t>                   input_ci16(cp_len + config.dft_size);
  std::vector<cf_t>                     input_quantized(input_ci16.size());
  std::mt19937                          rgen(0);
  std::uniform_real_distribution<float> dist(-0.05F, 0.05F);
  for (unsigned i = 0; i != input_ci16.size(); ++i) {
    int16_t re         = static_cast<int16_t>(std::round(dist(rgen) * input_scale));
    int16_t im         = static_cast<int16_t>(std::round(dist(rgen) * input_scale));
    input_ci16[i]      = ci16_t(re, im);
    input_quantized[i] = cf_t(static_cast<float>(re) / input_scale, static_cast<float>(im) / input_scale);
  }

  ofdm_factory_generic_configuration common_config = {.dft_factory = make_dft_factory()};
  ASSERT_NE(common_config.dft_factory, nullptr);

  auto cpu_factory = create_ofdm_demodulator_factory_generic(common_config);
  auto gpu_factory = create_ofdm_demodulator_factory_accelerated(common_config, "enabled");
  ASSERT_NE(cpu_factory, nullptr);
  ASSERT_NE(gpu_factory, nullptr);

  std::unique_ptr<ofdm_symbol_demodulator> cpu_demodulator = cpu_factory->create_ofdm_symbol_demodulator(config);
  std::unique_ptr<ofdm_symbol_demodulator> gpu_demodulator = gpu_factory->create_ofdm_symbol_demodulator(config);
  ASSERT_NE(cpu_demodulator, nullptr);
  ASSERT_NE(gpu_demodulator, nullptr);

  std::shared_ptr<resource_grid_factory> cpu_grid_factory = create_resource_grid_factory();
  ASSERT_NE(cpu_grid_factory, nullptr);
  std::unique_ptr<resource_grid> cpu_grid =
      cpu_grid_factory->create(1, get_nsymb_per_slot(config.cp), config.bw_rb * NOF_SUBCARRIERS_PER_RB);
  ASSERT_NE(cpu_grid, nullptr);

  resource_grid_cuda_visible_factory gpu_grid_factory(resource_grid_cuda_visible_factory::direction::uplink);
  std::unique_ptr<resource_grid>     gpu_grid =
      gpu_grid_factory.create(1, get_nsymb_per_slot(config.cp), config.bw_rb * NOF_SUBCARRIERS_PER_RB);
  ASSERT_NE(gpu_grid, nullptr);
  ASSERT_TRUE(gpu_grid->get_writer().supports_device_grid_mapping());

  cpu_demodulator->demodulate(cpu_grid->get_writer(), input_quantized, 0, symbol_index);
  ASSERT_TRUE(gpu_demodulator->demodulate_ci16(gpu_grid->get_writer(), input_ci16, input_scale, 0, symbol_index));
  ASSERT_TRUE(gpu_grid->get_writer().synchronize_device_grid_mapping());

  unsigned            l          = symbol_index % get_nsymb_per_slot(config.cp);
  span<const cbf16_t> cpu_symbol = cpu_grid->get_reader().get_view(0, l);
  span<const cbf16_t> gpu_symbol = gpu_grid->get_reader().get_view(0, l);
  ASSERT_EQ(cpu_symbol.size(), gpu_symbol.size());
  for (unsigned i_re = 0; i_re != cpu_symbol.size(); ++i_re) {
    cf_t cpu = to_cf(cpu_symbol[i_re]);
    cf_t gpu = to_cf(gpu_symbol[i_re]);
    EXPECT_NEAR(cpu.real(), gpu.real(), 0.03F);
    EXPECT_NEAR(cpu.imag(), gpu.imag(), 0.03F);
  }
}

void run_ci16_multiport_all_symbols_match(unsigned window_offset)
{
  ASSERT_EQ(setenv("OCUDU_UL_CUDA_VISIBLE_GRID", "managed", 1), 0);
  ASSERT_EQ(setenv("OCUDU_LOWPHY_RX_ACCELERATION", "enabled", 1), 0);
  ASSERT_EQ(setenv("OCUDU_LOWPHY_RX_CUDA_GRAPHS", "1", 1), 0);

  static constexpr float    input_scale = std::numeric_limits<int16_t>::max();
  static constexpr unsigned nof_ports   = 2;

  ofdm_demodulator_configuration config = {};
  config.numerology                     = 1;
  config.bw_rb                          = 52;
  config.dft_size                       = 1024;
  config.cp                             = cyclic_prefix::NORMAL;
  config.nof_samples_window_offset      = window_offset;
  config.scale                          = 1.0F / std::sqrt(static_cast<float>(config.bw_rb * NOF_SUBCARRIERS_PER_RB));
  config.center_freq_Hz                 = 3410.1e6;

  subcarrier_spacing scs                      = to_subcarrier_spacing(config.numerology);
  unsigned           srate_Hz                 = to_sampling_rate_Hz(scs, config.dft_size);
  unsigned           nsymb                    = get_nsymb_per_slot(config.cp);
  unsigned           nof_symbols_per_subframe = nsymb * get_nof_slots_per_subframe(scs);
  unsigned           nof_subc                 = config.bw_rb * NOF_SUBCARRIERS_PER_RB;

  ofdm_factory_generic_configuration common_config = {.dft_factory = make_dft_factory()};
  ASSERT_NE(common_config.dft_factory, nullptr);

  auto cpu_factory = create_ofdm_demodulator_factory_generic(common_config);
  auto gpu_factory = create_ofdm_demodulator_factory_accelerated(common_config, "enabled");
  ASSERT_NE(cpu_factory, nullptr);
  ASSERT_NE(gpu_factory, nullptr);

  std::unique_ptr<ofdm_symbol_demodulator> cpu_demodulator = cpu_factory->create_ofdm_symbol_demodulator(config);
  std::unique_ptr<ofdm_symbol_demodulator> gpu_demodulator = gpu_factory->create_ofdm_symbol_demodulator(config);
  ASSERT_NE(cpu_demodulator, nullptr);
  ASSERT_NE(gpu_demodulator, nullptr);

  std::shared_ptr<resource_grid_factory> cpu_grid_factory = create_resource_grid_factory();
  ASSERT_NE(cpu_grid_factory, nullptr);
  std::unique_ptr<resource_grid> cpu_grid = cpu_grid_factory->create(nof_ports, nsymb, nof_subc);
  ASSERT_NE(cpu_grid, nullptr);

  resource_grid_cuda_visible_factory gpu_grid_factory(resource_grid_cuda_visible_factory::direction::uplink);
  std::unique_ptr<resource_grid>     gpu_grid = gpu_grid_factory.create(nof_ports, nsymb, nof_subc);
  ASSERT_NE(gpu_grid, nullptr);
  ASSERT_TRUE(gpu_grid->get_writer().supports_device_grid_mapping());

  std::mt19937                          rgen(100 + window_offset);
  std::uniform_real_distribution<float> dist(-0.05F, 0.05F);
  std::array<unsigned, nof_ports>       port_indices = {0, 1};

  for (unsigned symbol_index = 0; symbol_index != nof_symbols_per_subframe; ++symbol_index) {
    unsigned cp_len = config.cp.get_length(symbol_index, scs).to_samples(srate_Hz);
    ASSERT_GE(cp_len, window_offset);
    unsigned nof_samples = cp_len + config.dft_size;

    std::array<std::vector<ci16_t>, nof_ports> input_ci16;
    std::array<std::vector<cf_t>, nof_ports>   input_quantized;
    std::array<const ci16_t*, nof_ports>       input_ptrs = {};
    for (unsigned port = 0; port != nof_ports; ++port) {
      input_ci16[port].resize(nof_samples);
      input_quantized[port].resize(nof_samples);
      for (unsigned i = 0; i != nof_samples; ++i) {
        int16_t re               = static_cast<int16_t>(std::round(dist(rgen) * input_scale));
        int16_t im               = static_cast<int16_t>(std::round(dist(rgen) * input_scale));
        input_ci16[port][i]      = ci16_t(re, im);
        input_quantized[port][i] = cf_t(static_cast<float>(re) / input_scale, static_cast<float>(im) / input_scale);
      }
      input_ptrs[port] = input_ci16[port].data();
      cpu_demodulator->demodulate(cpu_grid->get_writer(), input_quantized[port], port, symbol_index);
    }

    ASSERT_TRUE(gpu_demodulator->demodulate_ci16_ports(gpu_grid->get_writer(),
                                                       span<const ci16_t* const>(input_ptrs.data(), nof_ports),
                                                       nof_samples,
                                                       input_scale,
                                                       span<const unsigned>(port_indices.data(), nof_ports),
                                                       symbol_index));
    ASSERT_TRUE(gpu_grid->get_writer().synchronize_device_grid_mapping());

    unsigned l = symbol_index % nsymb;
    for (unsigned port = 0; port != nof_ports; ++port) {
      span<const cbf16_t> cpu_symbol = cpu_grid->get_reader().get_view(port, l);
      span<const cbf16_t> gpu_symbol = gpu_grid->get_reader().get_view(port, l);
      ASSERT_EQ(cpu_symbol.size(), gpu_symbol.size());
      for (unsigned i_re = 0; i_re != cpu_symbol.size(); ++i_re) {
        cf_t cpu = to_cf(cpu_symbol[i_re]);
        cf_t gpu = to_cf(gpu_symbol[i_re]);
        EXPECT_NEAR(cpu.real(), gpu.real(), 0.03F);
        EXPECT_NEAR(cpu.imag(), gpu.imag(), 0.03F);
      }
    }
  }
}

} // namespace

TEST(ofdm_demodulator_cuda, gpu_ci16_fast_path_matches_cpu)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA is not available.";
  }
  run_ci16_fast_path_match(0);
}

TEST(ofdm_demodulator_cuda, gpu_ci16_fast_path_matches_cpu_with_window_offset)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA is not available.";
  }
  run_ci16_fast_path_match(8);
}

TEST(ofdm_demodulator_cuda, gpu_ci16_multiport_all_symbols_matches_cpu_with_phase_compensation)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA is not available.";
  }
  run_ci16_multiport_all_symbols_match(8);
}
