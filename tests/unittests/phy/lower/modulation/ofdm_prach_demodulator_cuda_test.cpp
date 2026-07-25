// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "phy_acceleration_prach_buffer_factory.h"
#include "ocudu/phy/lower/modulation/modulation_factories.h"
#include "ocudu/phy/support/support_factories.h"
#include "ocudu/phy/upper/channel_processors/prach/factories.h"
#include "ocudu/phy/upper/channel_processors/prach/prach_detector.h"
#include "ocudu/phy/upper/channel_processors/prach/prach_generator.h"
#include "ocudu/ran/prach/prach_frequency_mapping.h"
#include "ocudu/ran/prach/prach_preamble_information.h"
#include "ocudu/ran/resource_block.h"
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

ofdm_prach_demodulator::configuration make_config()
{
  return {.slot             = slot_point(1, 0, 0),
          .format           = prach_format_type::B4,
          .nof_td_occasions = 1,
          .nof_fd_occasions = 1,
          .start_symbol     = 0,
          .rb_offset        = 0,
          .nof_prb_ul_grid  = 52,
          .port             = 0};
}

ofdm_prach_demodulator::configuration make_live_long_config()
{
  return {.slot             = slot_point(1, 0, 18),
          .format           = prach_format_type::zero,
          .nof_td_occasions = 1,
          .nof_fd_occasions = 1,
          .start_symbol     = 0,
          .rb_offset        = 12,
          .nof_prb_ul_grid  = 51,
          .port             = 0};
}

std::shared_ptr<dft_processor_factory> make_dft_factory()
{
  std::shared_ptr<dft_processor_factory> dft_factory = create_dft_processor_factory_fftw_fast();
  if (dft_factory == nullptr) {
    dft_factory = create_dft_processor_factory_generic();
  }
  return dft_factory;
}

void assert_symbols_near(const prach_buffer&                          cpu_buffer,
                         const prach_buffer&                          gpu_buffer,
                         const ofdm_prach_demodulator::configuration& config,
                         float                                        tolerance)
{
  const subcarrier_spacing pusch_scs = to_subcarrier_spacing(config.slot.numerology());
  const auto               preamble_info =
      is_long_preamble(config.format)
                        ? get_prach_preamble_long_info(config.format)
                        : get_prach_preamble_short_info(config.format, to_ra_subcarrier_spacing(pusch_scs), true);

  for (unsigned i_symbol = 0; i_symbol != preamble_info.nof_symbols; ++i_symbol) {
    span<const cbf16_t> cpu_symbol = cpu_buffer.get_symbol(0, 0, 0, i_symbol);
    span<const cbf16_t> gpu_symbol = gpu_buffer.get_symbol(0, 0, 0, i_symbol);
    ASSERT_EQ(cpu_symbol.size(), gpu_symbol.size());
    for (unsigned i_re = 0; i_re != cpu_symbol.size(); ++i_re) {
      cf_t cpu = to_cf(cpu_symbol[i_re]);
      cf_t gpu = to_cf(gpu_symbol[i_re]);
      EXPECT_NEAR(cpu.real(), gpu.real(), tolerance);
      EXPECT_NEAR(cpu.imag(), gpu.imag(), tolerance);
    }
  }
}

unsigned get_prach_window_samples(sampling_rate srate, const ofdm_prach_demodulator::configuration& config)
{
  return get_prach_window_duration(config.format,
                                   to_subcarrier_spacing(config.slot.numerology()),
                                   config.start_symbol,
                                   config.nof_td_occasions)
      .to_samples(srate.to_Hz());
}

std::vector<cf_t> build_time_domain_prach_preamble(sampling_rate                                srate,
                                                   const ofdm_prach_demodulator::configuration& config,
                                                   unsigned                                     root_sequence_index,
                                                   unsigned                                     preamble_index,
                                                   std::shared_ptr<dft_processor_factory>       dft_factory,
                                                   unsigned timing_offset_samples = 0)
{
  const subcarrier_spacing pusch_scs     = to_subcarrier_spacing(config.slot.numerology());
  const unsigned           pusch_scs_Hz  = scs_to_khz(pusch_scs) * 1000;
  const auto               preamble_info = get_prach_preamble_long_info(config.format);
  const unsigned           prach_scs_Hz  = ra_scs_to_Hz(preamble_info.scs);
  const unsigned           dft_size      = srate.get_dft_size(prach_scs_Hz);
  const unsigned           cp_len        = preamble_info.cp_length.to_samples(srate.to_Hz());
  const unsigned           nof_samples   = get_prach_window_samples(srate, config);

  prach_frequency_mapping_information freq_mapping_info = prach_frequency_mapping_get(preamble_info.scs, pusch_scs);
  EXPECT_NE(freq_mapping_info.nof_rb_ra, PRACH_FREQUENCY_MAPPING_INFORMATION_RESERVED.nof_rb_ra);
  EXPECT_NE(freq_mapping_info.k_bar, PRACH_FREQUENCY_MAPPING_INFORMATION_RESERVED.k_bar);

  const unsigned K               = pusch_scs_Hz / prach_scs_Hz;
  const unsigned prach_grid_size = config.nof_prb_ul_grid * K * NOF_SUBCARRIERS_PER_RB;
  EXPECT_GT(dft_size, prach_grid_size);

  std::unique_ptr<prach_generator> generator = create_prach_generator_factory_sw()->create();
  prach_generator::configuration   generator_config;
  generator_config.format                = config.format;
  generator_config.root_sequence_index   = root_sequence_index;
  generator_config.preamble_index        = preamble_index;
  generator_config.restricted_set        = restricted_set_config::UNRESTRICTED;
  generator_config.zero_correlation_zone = 0;
  span<const cf_t> sequence              = generator->generate(generator_config);
  EXPECT_EQ(sequence.size(), preamble_info.sequence_length);

  std::unique_ptr<dft_processor> idft =
      dft_factory->create({.size = dft_size, .dir = dft_processor::direction::INVERSE});
  EXPECT_NE(idft, nullptr);

  span<cf_t> idft_input = idft->get_input();
  std::fill(idft_input.begin(), idft_input.end(), cf_t());

  const unsigned k_start   = K * NOF_SUBCARRIERS_PER_RB * config.rb_offset + freq_mapping_info.k_bar;
  const unsigned half_grid = prach_grid_size / 2;
  const float    dft_scale = std::sqrt(static_cast<float>(dft_size));
  for (unsigned i_re = 0; i_re != sequence.size(); ++i_re) {
    unsigned k      = k_start + i_re;
    unsigned src    = (k < half_grid) ? (dft_size - half_grid + k) : (k - half_grid);
    idft_input[src] = sequence[i_re] * dft_scale;
  }

  span<const cf_t>  time_symbol = idft->run();
  std::vector<cf_t> input(nof_samples, cf_t());
  for (unsigned i = 0; i != cp_len; ++i) {
    unsigned dst = timing_offset_samples + i;
    if (dst < input.size()) {
      input[dst] = time_symbol[dft_size - cp_len + i] / static_cast<float>(dft_size);
    }
  }
  for (unsigned i = 0; i != dft_size; ++i) {
    unsigned dst = timing_offset_samples + cp_len + i;
    if (dst < input.size()) {
      input[dst] = time_symbol[i] / static_cast<float>(dft_size);
    }
  }
  return input;
}

const prach_detection_result::preamble_indication* find_preamble(const prach_detection_result& result,
                                                                 unsigned                      preamble_index)
{
  for (const auto& preamble : result.preambles) {
    if (preamble.preamble_index == preamble_index) {
      return &preamble;
    }
  }
  return nullptr;
}

} // namespace

TEST(ofdm_prach_demodulator_cuda, gpu_matches_cpu_for_short_prach_window)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA is not available.";
  }

  ASSERT_EQ(setenv("OCUDU_UL_CUDA_VISIBLE_PRACH_BUFFER", "managed", 1), 0);

  sampling_rate srate     = sampling_rate::from_MHz(61.44);
  auto          config    = make_config();
  auto          pusch_scs = to_subcarrier_spacing(config.slot.numerology());
  unsigned      nof_samples =
      get_prach_window_duration(config.format, pusch_scs, config.start_symbol, config.nof_td_occasions)
          .to_samples(srate.to_Hz());

  std::vector<cf_t>                     input(nof_samples);
  std::mt19937                          rgen(0);
  std::uniform_real_distribution<float> dist(-0.02F, 0.02F);
  for (cf_t& sample : input) {
    sample = cf_t(dist(rgen), dist(rgen));
  }

  std::shared_ptr<dft_processor_factory> dft_factory = make_dft_factory();
  ASSERT_NE(dft_factory, nullptr);

  auto cpu_factory = create_ofdm_prach_demodulator_factory_sw(dft_factory, srate, frequency_range::FR1);
  auto gpu_factory =
      create_ofdm_prach_demodulator_factory_accelerated(dft_factory, srate, frequency_range::FR1, "enabled");
  ASSERT_NE(cpu_factory, nullptr);
  ASSERT_NE(gpu_factory, nullptr);

  std::unique_ptr<ofdm_prach_demodulator> cpu_demodulator = cpu_factory->create();
  std::unique_ptr<ofdm_prach_demodulator> gpu_demodulator = gpu_factory->create();
  ASSERT_NE(cpu_demodulator, nullptr);
  ASSERT_NE(gpu_demodulator, nullptr);

  std::unique_ptr<prach_buffer> cpu_buffer = create_prach_buffer_short(1, 1, 1);
  std::unique_ptr<prach_buffer> gpu_buffer = create_phy_acceleration_prach_buffer_short(1, 1, 1, true);
  ASSERT_NE(cpu_buffer, nullptr);
  ASSERT_NE(gpu_buffer, nullptr);
  ASSERT_TRUE(gpu_buffer->supports_device_prach_buffer_mapping());

  cpu_demodulator->demodulate(*cpu_buffer, input, config);
  gpu_demodulator->demodulate(*gpu_buffer, input, config);
  ASSERT_TRUE(gpu_buffer->synchronize_device_prach_buffer_mapping());

  assert_symbols_near(*cpu_buffer, *gpu_buffer, config, 0.03F);
}

TEST(ofdm_prach_demodulator_cuda, gpu_ci16_fast_path_matches_cpu_for_short_prach_window)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA is not available.";
  }

  ASSERT_EQ(setenv("OCUDU_UL_CUDA_VISIBLE_PRACH_BUFFER", "managed", 1), 0);

  sampling_rate srate     = sampling_rate::from_MHz(61.44);
  auto          config    = make_config();
  auto          pusch_scs = to_subcarrier_spacing(config.slot.numerology());
  unsigned      nof_samples =
      get_prach_window_duration(config.format, pusch_scs, config.start_symbol, config.nof_td_occasions)
          .to_samples(srate.to_Hz());

  static constexpr float                input_scale = std::numeric_limits<int16_t>::max();
  std::vector<ci16_t>                   input_ci16(nof_samples);
  std::vector<cf_t>                     input_quantized(nof_samples);
  std::mt19937                          rgen(0);
  std::uniform_real_distribution<float> dist(-0.02F, 0.02F);
  for (unsigned i = 0; i != nof_samples; ++i) {
    int16_t re         = static_cast<int16_t>(std::round(dist(rgen) * input_scale));
    int16_t im         = static_cast<int16_t>(std::round(dist(rgen) * input_scale));
    input_ci16[i]      = ci16_t(re, im);
    input_quantized[i] = cf_t(static_cast<float>(re) / input_scale, static_cast<float>(im) / input_scale);
  }

  std::shared_ptr<dft_processor_factory> dft_factory = make_dft_factory();
  ASSERT_NE(dft_factory, nullptr);

  auto cpu_factory = create_ofdm_prach_demodulator_factory_sw(dft_factory, srate, frequency_range::FR1);
  auto gpu_factory =
      create_ofdm_prach_demodulator_factory_accelerated(dft_factory, srate, frequency_range::FR1, "enabled");
  ASSERT_NE(cpu_factory, nullptr);
  ASSERT_NE(gpu_factory, nullptr);

  std::unique_ptr<ofdm_prach_demodulator> cpu_demodulator = cpu_factory->create();
  std::unique_ptr<ofdm_prach_demodulator> gpu_demodulator = gpu_factory->create();
  ASSERT_NE(cpu_demodulator, nullptr);
  ASSERT_NE(gpu_demodulator, nullptr);

  std::unique_ptr<prach_buffer> cpu_buffer = create_prach_buffer_short(1, 1, 1);
  std::unique_ptr<prach_buffer> gpu_buffer = create_phy_acceleration_prach_buffer_short(1, 1, 1, true);
  ASSERT_NE(cpu_buffer, nullptr);
  ASSERT_NE(gpu_buffer, nullptr);
  ASSERT_TRUE(gpu_buffer->supports_device_prach_buffer_mapping());

  cpu_demodulator->demodulate(*cpu_buffer, input_quantized, config);
  ASSERT_TRUE(gpu_demodulator->demodulate_ci16(*gpu_buffer, input_ci16, input_scale, config));
  ASSERT_TRUE(gpu_buffer->synchronize_device_prach_buffer_mapping());

  assert_symbols_near(*cpu_buffer, *gpu_buffer, config, 0.03F);
}

TEST(ofdm_prach_demodulator_cuda, gpu_ci16_fast_path_matches_cpu_for_live_long_prach_window)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA is not available.";
  }

  ASSERT_EQ(setenv("OCUDU_UL_CUDA_VISIBLE_PRACH_BUFFER", "managed", 1), 0);

  sampling_rate srate       = sampling_rate::from_MHz(23.04);
  auto          config      = make_live_long_config();
  unsigned      nof_samples = get_prach_window_samples(srate, config);

  static constexpr float                input_scale = std::numeric_limits<int16_t>::max();
  std::vector<ci16_t>                   input_ci16(nof_samples);
  std::vector<cf_t>                     input_quantized(nof_samples);
  std::mt19937                          rgen(0);
  std::uniform_real_distribution<float> dist(-0.02F, 0.02F);
  for (unsigned i = 0; i != nof_samples; ++i) {
    int16_t re         = static_cast<int16_t>(std::round(dist(rgen) * input_scale));
    int16_t im         = static_cast<int16_t>(std::round(dist(rgen) * input_scale));
    input_ci16[i]      = ci16_t(re, im);
    input_quantized[i] = cf_t(static_cast<float>(re) / input_scale, static_cast<float>(im) / input_scale);
  }

  std::shared_ptr<dft_processor_factory> dft_factory = make_dft_factory();
  ASSERT_NE(dft_factory, nullptr);

  auto cpu_factory = create_ofdm_prach_demodulator_factory_sw(dft_factory, srate, frequency_range::FR1);
  auto gpu_factory =
      create_ofdm_prach_demodulator_factory_accelerated(dft_factory, srate, frequency_range::FR1, "enabled");
  ASSERT_NE(cpu_factory, nullptr);
  ASSERT_NE(gpu_factory, nullptr);

  std::unique_ptr<ofdm_prach_demodulator> cpu_demodulator = cpu_factory->create();
  std::unique_ptr<ofdm_prach_demodulator> gpu_demodulator = gpu_factory->create();
  ASSERT_NE(cpu_demodulator, nullptr);
  ASSERT_NE(gpu_demodulator, nullptr);

  std::unique_ptr<prach_buffer> cpu_buffer = create_prach_buffer_long(1, 1);
  std::unique_ptr<prach_buffer> gpu_buffer = create_phy_acceleration_prach_buffer_long(1, 1, true);
  ASSERT_NE(cpu_buffer, nullptr);
  ASSERT_NE(gpu_buffer, nullptr);
  ASSERT_TRUE(gpu_buffer->supports_device_prach_buffer_mapping());

  cpu_demodulator->demodulate(*cpu_buffer, input_quantized, config);
  ASSERT_TRUE(gpu_demodulator->demodulate_ci16(*gpu_buffer, input_ci16, input_scale, config));
  ASSERT_TRUE(gpu_buffer->synchronize_device_prach_buffer_mapping());

  assert_symbols_near(*cpu_buffer, *gpu_buffer, config, 0.03F);
}

TEST(ofdm_prach_demodulator_cuda, live_long_prach_e2e_detects_in_device_buffer)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA is not available.";
  }

  ASSERT_EQ(setenv("OCUDU_UL_CUDA_VISIBLE_PRACH_BUFFER", "managed", 1), 0);

  static constexpr unsigned root_sequence_index = 1;
  static constexpr unsigned target_preamble     = 7;

  sampling_rate srate  = sampling_rate::from_MHz(23.04);
  auto          config = make_live_long_config();

  std::shared_ptr<dft_processor_factory> dft_factory = make_dft_factory();
  ASSERT_NE(dft_factory, nullptr);

  std::vector<cf_t> input =
      build_time_domain_prach_preamble(srate, config, root_sequence_index, target_preamble, dft_factory);

  auto cpu_demod_factory = create_ofdm_prach_demodulator_factory_sw(dft_factory, srate, frequency_range::FR1);
  auto gpu_demod_factory =
      create_ofdm_prach_demodulator_factory_accelerated(dft_factory, srate, frequency_range::FR1, "enabled");
  ASSERT_NE(cpu_demod_factory, nullptr);
  ASSERT_NE(gpu_demod_factory, nullptr);

  std::unique_ptr<ofdm_prach_demodulator> cpu_demodulator = cpu_demod_factory->create();
  std::unique_ptr<ofdm_prach_demodulator> gpu_demodulator = gpu_demod_factory->create();
  ASSERT_NE(cpu_demodulator, nullptr);
  ASSERT_NE(gpu_demodulator, nullptr);

  std::unique_ptr<prach_buffer> cpu_buffer = create_prach_buffer_long(1, 1);
  std::unique_ptr<prach_buffer> gpu_buffer = create_phy_acceleration_prach_buffer_long(1, 1, true);
  ASSERT_NE(cpu_buffer, nullptr);
  ASSERT_NE(gpu_buffer, nullptr);
  ASSERT_TRUE(gpu_buffer->supports_device_prach_buffer_mapping());

  cpu_demodulator->demodulate(*cpu_buffer, input, config);
  gpu_demodulator->demodulate(*gpu_buffer, input, config);
  ASSERT_TRUE(gpu_buffer->synchronize_device_prach_buffer_mapping());
  assert_symbols_near(*cpu_buffer, *gpu_buffer, config, 0.03F);

  std::shared_ptr<prach_generator_factory> generator_factory = create_prach_generator_factory_sw();
  ASSERT_NE(generator_factory, nullptr);
  std::unique_ptr<prach_detector> cpu_detector =
      create_prach_detector_factory_sw(dft_factory, generator_factory)->create();
  std::unique_ptr<prach_detector> gpu_detector =
      create_prach_detector_factory_accelerated(dft_factory, generator_factory, {}, "enabled")->create();

  prach_detector::configuration detector_config = {.root_sequence_index   = root_sequence_index,
                                                   .format                = config.format,
                                                   .restricted_set        = restricted_set_config::UNRESTRICTED,
                                                   .zero_correlation_zone = 0,
                                                   .start_preamble_index  = 0,
                                                   .nof_preamble_indices  = 64,
                                                   .ra_scs                = prach_subcarrier_spacing::kHz1_25,
                                                   .nof_rx_ports          = 1,
                                                   .slot                  = config.slot};

  prach_detection_result cpu_result = cpu_detector->detect(*cpu_buffer, detector_config);
  prach_detection_result gpu_result = gpu_detector->detect(*gpu_buffer, detector_config);

  const auto* cpu_preamble = find_preamble(cpu_result, target_preamble);
  const auto* gpu_preamble = find_preamble(gpu_result, target_preamble);
  ASSERT_NE(cpu_preamble, nullptr);
  ASSERT_NE(gpu_preamble, nullptr);
  EXPECT_EQ(cpu_preamble->preamble_index, gpu_preamble->preamble_index);
  EXPECT_NEAR(cpu_preamble->time_advance.to_seconds(), gpu_preamble->time_advance.to_seconds(), 1e-9);
  EXPECT_NEAR(cpu_preamble->detection_metric, gpu_preamble->detection_metric, 0.2F);
  EXPECT_NEAR(cpu_preamble->preamble_power_dB, gpu_preamble->preamble_power_dB, 0.3F);
}

TEST(ofdm_prach_demodulator_cuda, live_long_prach_e2e_matches_cpu_for_preamble_46_with_timing_offset)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA is not available.";
  }

  ASSERT_EQ(setenv("OCUDU_UL_CUDA_VISIBLE_PRACH_BUFFER", "managed", 1), 0);

  static constexpr unsigned root_sequence_index    = 1;
  static constexpr unsigned target_preamble        = 46;
  static constexpr unsigned timing_offset_samples  = 64;
  static constexpr double   time_advance_tolerance = 1e-9;

  sampling_rate srate  = sampling_rate::from_MHz(23.04);
  auto          config = make_live_long_config();

  std::shared_ptr<dft_processor_factory> dft_factory = make_dft_factory();
  ASSERT_NE(dft_factory, nullptr);

  std::vector<cf_t> input = build_time_domain_prach_preamble(
      srate, config, root_sequence_index, target_preamble, dft_factory, timing_offset_samples);

  auto cpu_demod_factory = create_ofdm_prach_demodulator_factory_sw(dft_factory, srate, frequency_range::FR1);
  auto gpu_demod_factory =
      create_ofdm_prach_demodulator_factory_accelerated(dft_factory, srate, frequency_range::FR1, "enabled");
  ASSERT_NE(cpu_demod_factory, nullptr);
  ASSERT_NE(gpu_demod_factory, nullptr);

  std::unique_ptr<ofdm_prach_demodulator> cpu_demodulator = cpu_demod_factory->create();
  std::unique_ptr<ofdm_prach_demodulator> gpu_demodulator = gpu_demod_factory->create();
  ASSERT_NE(cpu_demodulator, nullptr);
  ASSERT_NE(gpu_demodulator, nullptr);

  std::unique_ptr<prach_buffer> cpu_buffer = create_prach_buffer_long(1, 1);
  std::unique_ptr<prach_buffer> gpu_buffer = create_phy_acceleration_prach_buffer_long(1, 1, true);
  ASSERT_NE(cpu_buffer, nullptr);
  ASSERT_NE(gpu_buffer, nullptr);
  ASSERT_TRUE(gpu_buffer->supports_device_prach_buffer_mapping());

  cpu_demodulator->demodulate(*cpu_buffer, input, config);
  gpu_demodulator->demodulate(*gpu_buffer, input, config);
  ASSERT_TRUE(gpu_buffer->synchronize_device_prach_buffer_mapping());
  assert_symbols_near(*cpu_buffer, *gpu_buffer, config, 0.03F);

  std::shared_ptr<prach_generator_factory> generator_factory = create_prach_generator_factory_sw();
  ASSERT_NE(generator_factory, nullptr);
  std::unique_ptr<prach_detector> cpu_detector =
      create_prach_detector_factory_sw(dft_factory, generator_factory)->create();
  std::unique_ptr<prach_detector> gpu_detector =
      create_prach_detector_factory_accelerated(dft_factory, generator_factory, {}, "enabled")->create();

  prach_detector::configuration detector_config = {.root_sequence_index   = root_sequence_index,
                                                   .format                = config.format,
                                                   .restricted_set        = restricted_set_config::UNRESTRICTED,
                                                   .zero_correlation_zone = 0,
                                                   .start_preamble_index  = 0,
                                                   .nof_preamble_indices  = 64,
                                                   .ra_scs                = prach_subcarrier_spacing::kHz1_25,
                                                   .nof_rx_ports          = 1,
                                                   .slot                  = config.slot};

  prach_detection_result cpu_result = cpu_detector->detect(*cpu_buffer, detector_config);
  prach_detection_result gpu_result = gpu_detector->detect(*gpu_buffer, detector_config);

  const auto* cpu_preamble = find_preamble(cpu_result, target_preamble);
  const auto* gpu_preamble = find_preamble(gpu_result, target_preamble);
  ASSERT_NE(cpu_preamble, nullptr);
  ASSERT_NE(gpu_preamble, nullptr);
  EXPECT_EQ(cpu_preamble->preamble_index, gpu_preamble->preamble_index);
  EXPECT_NEAR(cpu_preamble->time_advance.to_seconds(), gpu_preamble->time_advance.to_seconds(), time_advance_tolerance);
  EXPECT_NEAR(cpu_preamble->detection_metric, gpu_preamble->detection_metric, 0.75F);
  EXPECT_NEAR(cpu_preamble->preamble_power_dB, gpu_preamble->preamble_power_dB, 0.3F);
}
