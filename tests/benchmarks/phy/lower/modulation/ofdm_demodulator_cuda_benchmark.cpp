// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "ocudu/ocuduvec/conversion.h"
#include "ocudu/phy/lower/modulation/modulation_factories.h"
#include "ocudu/phy/support/resource_grid.h"
#include "ocudu/phy/support/support_factories.h"
#include "ocudu/support/benchmark_utils.h"
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <getopt.h>
#include <limits>
#include <low_phy_puxch_rx.h>
#include <random>

using namespace ocudu;

namespace {

struct bench_case {
  const char* name;
  unsigned    nof_prb;
  unsigned    dft_size;
};

static unsigned nof_repetitions = 200;
static bool     silent          = false;

static void usage(const char* prog)
{
  fmt::print("Usage: {} [-R repetitions] [-s]\n", prog);
  fmt::print("\t-R Repetitions [Default {}]\n", nof_repetitions);
  fmt::print("\t-s Toggle silent operation [Default {}]\n", silent);
  fmt::print("\t-h Show this message\n");
}

static void parse_args(int argc, char** argv)
{
  int opt = 0;
  while ((opt = getopt(argc, argv, "R:sh")) != -1) {
    switch (opt) {
      case 'R':
        nof_repetitions = std::strtol(optarg, nullptr, 10);
        break;
      case 's':
        silent = !silent;
        break;
      case 'h':
      default:
        usage(argv[0]);
        std::exit(0);
    }
  }
}

std::shared_ptr<dft_processor_factory> make_dft_factory()
{
  std::shared_ptr<dft_processor_factory> dft_factory = create_dft_processor_factory_fftw_fast();
  if (dft_factory == nullptr) {
    dft_factory = create_dft_processor_factory_generic();
  }
  return dft_factory;
}

void fill_input(span<ci16_t> input)
{
  std::mt19937                          rgen(0);
  std::uniform_real_distribution<float> dist(-0.05F, 0.05F);
  static constexpr float                input_scale = std::numeric_limits<int16_t>::max();
  for (ci16_t& sample : input) {
    int16_t re = static_cast<int16_t>(std::round(dist(rgen) * input_scale));
    int16_t im = static_cast<int16_t>(std::round(dist(rgen) * input_scale));
    sample     = ci16_t(re, im);
  }
}

void benchmark_case(benchmarker& perf, const bench_case& bc, unsigned nof_ports)
{
  static constexpr float input_scale = std::numeric_limits<int16_t>::max();

  ofdm_demodulator_configuration config = {};
  config.numerology                     = 1;
  config.bw_rb                          = bc.nof_prb;
  config.dft_size                       = bc.dft_size;
  config.cp                             = cyclic_prefix::NORMAL;
  config.scale                          = 1.0F / std::sqrt(static_cast<float>(bc.nof_prb * NOF_SUBCARRIERS_PER_RB));
  config.center_freq_Hz                 = 0.0;

  subcarrier_spacing scs      = to_subcarrier_spacing(config.numerology);
  unsigned           srate_Hz = to_sampling_rate_Hz(scs, config.dft_size);
  if (!config.cp.is_valid(scs, config.dft_size)) {
    return;
  }

  unsigned symbol_index            = 1;
  unsigned nof_symbols             = get_nsymb_per_slot(config.cp);
  unsigned cp_len                  = config.cp.get_length(symbol_index, scs).to_samples(srate_Hz);
  config.nof_samples_window_offset = cp_len / 2;

  std::vector<std::vector<ci16_t>> input_ci16(nof_ports, std::vector<ci16_t>(cp_len + config.dft_size));
  std::vector<cf_t>                input_cf(cp_len + config.dft_size);
  for (std::vector<ci16_t>& port_input : input_ci16) {
    fill_input(port_input);
  }

  ofdm_factory_generic_configuration       common_config = {.dft_factory = make_dft_factory()};
  std::unique_ptr<ofdm_symbol_demodulator> cpu_demodulator =
      create_ofdm_demodulator_factory_generic(common_config)->create_ofdm_symbol_demodulator(config);

  std::shared_ptr<resource_grid_factory> rg_factory = create_resource_grid_factory();
  std::unique_ptr<resource_grid>         cpu_grid =
      rg_factory->create(nof_ports, nof_symbols, config.bw_rb * NOF_SUBCARRIERS_PER_RB);

  perf.new_measure(
      fmt::format("{} {}p CPU ci16-convert+FFTW", bc.name, nof_ports), input_ci16.front().size() * nof_ports, [&]() {
        for (unsigned port = 0; port != nof_ports; ++port) {
          ocuduvec::convert(input_cf, input_ci16[port], input_scale);
          cpu_demodulator->demodulate(cpu_grid->get_writer(), input_cf, port, symbol_index);
        }
      });

  ocudu_lowphy_puxch_rx_config_t gpu_config = {};
  gpu_config.dft_size                       = static_cast<int>(config.dft_size);
  gpu_config.rg_size                        = static_cast<int>(config.bw_rb * NOF_SUBCARRIERS_PER_RB);
  gpu_config.grid_nof_subc                  = gpu_config.rg_size;
  gpu_config.grid_nof_symbols               = static_cast<int>(nof_symbols);
  gpu_config.nof_ports                      = static_cast<int>(nof_ports);
  gpu_config.input_nof_samples              = static_cast<int>(input_ci16.front().size());
  gpu_config.cyclic_prefix_length           = static_cast<int>(cp_len);
  gpu_config.window_offset                  = static_cast<int>(config.nof_samples_window_offset);
  gpu_config.symbol_index                   = static_cast<int>(symbol_index % nof_symbols);
  gpu_config.input_is_device                = 0;
  gpu_config.dft_scale                      = config.scale;
  gpu_config.phase_re                       = 1.0F;
  gpu_config.phase_im                       = 0.0F;
  for (unsigned port = 0; port != nof_ports; ++port) {
    gpu_config.port_indices[port] = static_cast<int>(port);
  }

  ocudu_lowphy_puxch_rx_handle_t* handle = nullptr;
  if (ocudu_lowphy_puxch_rx_create(&gpu_config, &handle) == 0) {
    fmt::print("Skipping {} GPU path: failed to create CUDA lower-PHY RX handle.\n", bc.name);
    return;
  }

  uint32_t* d_grid = nullptr;
  size_t    grid_words =
      static_cast<size_t>(nof_ports) * gpu_config.grid_nof_subc * gpu_config.grid_nof_symbols * sizeof(uint32_t);
  if (cudaMalloc(&d_grid, grid_words) != cudaSuccess) {
    ocudu_lowphy_puxch_rx_destroy(handle);
    return;
  }

  std::array<const void*, OCUDU_LOWPHY_PUXCH_RX_MAX_PORTS> input_ptrs = {};
  for (unsigned port = 0; port != nof_ports; ++port) {
    input_ptrs[port] = input_ci16[port].data();
  }
  auto process_gpu_once = [&](void* output_grid) {
    if (nof_ports == 1) {
      return ocudu_lowphy_puxch_rx_process_ci16(handle, input_ptrs.front(), input_scale, output_grid, nullptr) != 0;
    }
    return ocudu_lowphy_puxch_rx_process_ci16_ports(handle, input_ptrs.data(), input_scale, output_grid, nullptr) != 0;
  };

  (void)process_gpu_once(d_grid);
  (void)ocudu_lowphy_puxch_rx_synchronize(handle);

  perf.new_measure(
      fmt::format("{} {}p GPU ci16+VkFFT", bc.name, nof_ports), input_ci16.front().size() * nof_ports, [&]() {
        (void)process_gpu_once(d_grid);
        (void)ocudu_lowphy_puxch_rx_synchronize(handle);
      });

  static constexpr unsigned nof_rotating_grids = 512;
  std::vector<uint32_t*>    rotating_grids(nof_rotating_grids, nullptr);
  bool                      rotating_grids_ready = true;
  for (uint32_t*& rotating_grid : rotating_grids) {
    if (cudaMalloc(&rotating_grid, grid_words) != cudaSuccess) {
      rotating_grids_ready = false;
      break;
    }
  }

  if (rotating_grids_ready) {
    unsigned rotate_index = 0;
    perf.new_measure(fmt::format("{} {}p GPU ci16+VkFFT rotating-grid", bc.name, nof_ports),
                     input_ci16.front().size() * nof_ports,
                     [&]() {
                       uint32_t* output_grid = rotating_grids[rotate_index++ % rotating_grids.size()];
                       (void)process_gpu_once(output_grid);
                       (void)ocudu_lowphy_puxch_rx_synchronize(handle);
                     });
  } else {
    fmt::print("Skipping {} {}p GPU rotating-grid path: failed to allocate output grids.\n", bc.name, nof_ports);
  }

  for (uint32_t* rotating_grid : rotating_grids) {
    if (rotating_grid != nullptr) {
      cudaFree(rotating_grid);
    }
  }
  cudaFree(d_grid);
  ocudu_lowphy_puxch_rx_destroy(handle);
}

} // namespace

int main(int argc, char** argv)
{
  parse_args(argc, argv);

  int device_count = 0;
  if ((cudaGetDeviceCount(&device_count) != cudaSuccess) || (device_count == 0)) {
    cudaGetLastError();
    fmt::print("CUDA is not available.\n");
    return 0;
  }

  benchmarker perf("Lower-PHY OFDM demodulator", nof_repetitions);
  for (const bench_case& bc : {bench_case{"5MHz", 11, 256},
                               bench_case{"10MHz", 24, 512},
                               bench_case{"20MHz", 51, 1024},
                               bench_case{"100MHz", 273, 4096}}) {
    for (unsigned nof_ports : {1U, 2U, 4U}) {
      benchmark_case(perf, bc, nof_ports);
    }
  }

  if (!silent) {
    perf.print_percentiles_time("microseconds", 1e-3);
  }
  return 0;
}
