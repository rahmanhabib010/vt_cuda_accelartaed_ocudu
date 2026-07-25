// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "srs_estimator_gpu_benchmark_helpers.h"
#include "fmt/format.h"
#include <getopt.h>

using namespace ocudu;
using namespace ocudu::srs_gpu_bench;

static unsigned    nof_repetitions = 1000;
static unsigned    nof_warmups     = 50;
static unsigned    nof_rx_ports    = 4;
static unsigned    nof_tx_ports    = 4;
static unsigned    nof_symbols     = 4;
static float       snr_db          = 30.0F;
static grid_mode   mode            = grid_mode::visible;
static std::string profile_set     = "single";

static void usage(const char* prog)
{
  fmt::print("Usage: {} [-R repetitions] [-W warmups] [-r rx_ports] [-t tx_ports] [-s symbols] [-S snr_db] [-G "
             "host|visible] [-P single|quick|full|profile_name]\n",
             prog);
  fmt::print("\t-R Repetitions [Default {}]\n", nof_repetitions);
  fmt::print("\t-W Warm-up repetitions [Default {}]\n", nof_warmups);
  fmt::print("\t-r Number of RX ports [Default {}]\n", nof_rx_ports);
  fmt::print("\t-t Number of TX ports, one of 1, 2, 4 [Default {}]\n", nof_tx_ports);
  fmt::print("\t-s Number of SRS symbols [Default {}]\n", nof_symbols);
  fmt::print("\t-S SNR in dB [Default {}]\n", snr_db);
  fmt::print("\t-G Resource-grid mode: visible uses CUDA-visible grid, host uses staged host grid [Default {}]\n",
             to_string(mode));
  fmt::print("\t-P Profile set/name: single uses -r/-t/-s, quick/full cover supported SRS modes [Default {}]\n",
             profile_set);
  fmt::print("\t   Named profiles: {}\n", supported_srs_profile_names());
  fmt::print("\t   Set OCUDU_SRS_ACCELERATION_FORCE=1 to bypass the production small-workload CPU fallback.\n");
  fmt::print("\t-h Show this message\n");
}

static void parse_args(int argc, char** argv)
{
  int opt = 0;
  while ((opt = getopt(argc, argv, "R:W:r:t:s:S:G:P:h")) != -1) {
    switch (opt) {
      case 'R':
        nof_repetitions = std::strtoul(optarg, nullptr, 10);
        break;
      case 'W':
        nof_warmups = std::strtoul(optarg, nullptr, 10);
        break;
      case 'r':
        nof_rx_ports = std::strtoul(optarg, nullptr, 10);
        break;
      case 't':
        nof_tx_ports = std::strtoul(optarg, nullptr, 10);
        break;
      case 's':
        nof_symbols = std::strtoul(optarg, nullptr, 10);
        break;
      case 'S':
        snr_db = std::strtof(optarg, nullptr);
        break;
      case 'G':
        mode = parse_grid_mode(optarg);
        break;
      case 'P':
        profile_set = optarg;
        break;
      case 'h':
      default:
        usage(argv[0]);
        std::exit(0);
    }
  }
}

static void print_stats(const char* label, const latency_stats& stats)
{
  fmt::print("{} latency: mean={:.3f}us p50={:.3f}us p95={:.3f}us min={:.3f}us max={:.3f}us\n",
             label,
             stats.mean_us,
             stats.p50_us,
             stats.p95_us,
             stats.min_us,
             stats.max_us);
}

static void run_profile(estimator_pair&              estimators,
                        const srs_benchmark_profile& profile,
                        unsigned                     grid_nof_symbols,
                        unsigned                     grid_nof_subc)
{
  auto config = make_srs_config(profile);
  auto grid   = create_grid(get_required_grid_ports(profile), grid_nof_symbols, grid_nof_subc, mode);
  TESTASSERT(grid);
  populate_srs_grid(*grid, config, snr_db, 0);

  srs_estimator_result cpu_result;
  srs_estimator_result gpu_result;
  for (unsigned i = 0; i != nof_warmups; ++i) {
    cpu_result = estimators.cpu->estimate(grid->get_reader(), config);
    gpu_result = estimators.gpu->estimate(grid->get_reader(), config);
  }

  std::vector<double> cpu_samples =
      measure_us(nof_repetitions, [&]() { cpu_result = estimators.cpu->estimate(grid->get_reader(), config); });
  std::vector<double> gpu_samples =
      measure_us(nof_repetitions, [&]() { gpu_result = estimators.gpu->estimate(grid->get_reader(), config); });

  latency_stats cpu_stats = summarize_latencies(cpu_samples);
  latency_stats gpu_stats = summarize_latencies(gpu_samples);

  float  matrix_rel_error = channel_matrix_relative_error(cpu_result.channel_matrix, gpu_result.channel_matrix);
  double ta_delta_ns =
      std::abs(cpu_result.time_alignment.time_alignment - gpu_result.time_alignment.time_alignment) * 1e9;
  float epre_delta_db = optional_delta(cpu_result.epre_dB, gpu_result.epre_dB);
  float rsrp_delta_db = optional_delta(cpu_result.rsrp_dB, gpu_result.rsrp_dB);
  float noise_delta   = optional_delta(cpu_result.noise_variance, gpu_result.noise_variance);
  float mean_speedup  = static_cast<float>(cpu_stats.mean_us / gpu_stats.mean_us);
  float p50_speedup   = static_cast<float>(cpu_stats.p50_us / gpu_stats.p50_us);

  fmt::print("SRS GPU latency benchmark: grid={} profile=\"{}\" pilot_re={} expected_path={} snr={:.1f}dB "
             "repetitions={}\n",
             to_string(mode),
             describe_profile(profile),
             get_observed_pilot_re(config),
             predicts_gpu_path(config) ? "gpu" : "cpu-fallback",
             snr_db,
             nof_repetitions);
  print_stats("CPU", cpu_stats);
  print_stats("GPU", gpu_stats);
  fmt::print("Speedup: mean={:.3f}x p50={:.3f}x\n", mean_speedup, p50_speedup);
  fmt::print("Accuracy: matrix_rel_error={:.6e} ta_delta_ns={:.3f} epre_delta_db={:.6f} rsrp_delta_db={:.6f} "
             "noise_variance_delta={:.6e}\n",
             matrix_rel_error,
             ta_delta_ns,
             epre_delta_db,
             rsrp_delta_db,
             noise_delta);
}

int main(int argc, char** argv)
{
  parse_args(argc, argv);

#ifndef ENABLE_CUDA
  fmt::print("SRS GPU latency benchmark requires a CUDA-enabled build.\n");
  return 0;
#else
  TESTASSERT(nof_repetitions != 0, "The number of repetitions cannot be zero.");
  TESTASSERT(nof_rx_ports >= 1 && nof_rx_ports <= srs_constants::max_nof_rx_ports);
  TESTASSERT(nof_symbols >= 1 && nof_symbols <= 4);
  (void)to_srs_ports(nof_tx_ports);

  unsigned       grid_nof_symbols = get_nsymb_per_slot(cyclic_prefix::NORMAL);
  unsigned       grid_nof_subc    = MAX_NOF_SUBCARRIERS;
  estimator_pair estimators       = make_estimators();

  if (profile_set == "single") {
    srs_benchmark_profile profile = {
        "single", nof_rx_ports, nof_tx_ports, {0, 1, 2, 3}, subcarrier_spacing::kHz30, nof_symbols};
    run_profile(estimators, profile, grid_nof_symbols, grid_nof_subc);
  } else if ((profile_set == "quick") || (profile_set == "full")) {
    bool                               full     = profile_set == "full";
    std::vector<srs_benchmark_profile> profiles = make_supported_srs_profiles(full);
    fmt::print("SRS GPU latency profile sweep: grid={} profiles={} snr={:.1f}dB repetitions={} warmups={}\n",
               to_string(mode),
               profiles.size(),
               snr_db,
               nof_repetitions,
               nof_warmups);
    for (const srs_benchmark_profile& profile : profiles) {
      run_profile(estimators, profile, grid_nof_symbols, grid_nof_subc);
    }
  } else {
    std::optional<srs_benchmark_profile> profile = find_supported_srs_profile(profile_set);
    TESTASSERT(profile.has_value(),
               "Unsupported profile '{}'. Use single, quick, full or one of: {}.",
               profile_set,
               supported_srs_profile_names());
    run_profile(estimators, profile.value(), grid_nof_symbols, grid_nof_subc);
  }

  return 0;
#endif
}
