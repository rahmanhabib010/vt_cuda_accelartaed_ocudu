// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "srs_estimator_gpu_benchmark_helpers.h"
#include "fmt/format.h"
#include <getopt.h>

using namespace ocudu;
using namespace ocudu::srs_gpu_bench;

static unsigned    nof_frames   = 50;
static unsigned    nof_rx_ports = 4;
static unsigned    nof_tx_ports = 4;
static unsigned    nof_symbols  = 4;
static float       snr_start_db = 0.0F;
static float       snr_stop_db  = 30.0F;
static float       snr_step_db  = 5.0F;
static grid_mode   mode         = grid_mode::visible;
static std::string profile_set  = "single";

struct sweep_accumulator {
  double   matrix_rel_sum   = 0.0;
  double   matrix_rel_max   = 0.0;
  double   ta_abs_ns_sum    = 0.0;
  double   ta_abs_ns_max    = 0.0;
  double   epre_abs_db_sum  = 0.0;
  double   rsrp_abs_db_sum  = 0.0;
  double   noise_rel_sum    = 0.0;
  double   noise_rel_max    = 0.0;
  unsigned nof_observations = 0;

  void push(const srs_estimator_result& cpu_result, const srs_estimator_result& gpu_result)
  {
    double matrix_rel = channel_matrix_relative_error(cpu_result.channel_matrix, gpu_result.channel_matrix);
    double ta_abs_ns =
        std::abs(cpu_result.time_alignment.time_alignment - gpu_result.time_alignment.time_alignment) * 1e9;
    double epre_abs_db = std::abs(optional_delta(cpu_result.epre_dB, gpu_result.epre_dB));
    double rsrp_abs_db = std::abs(optional_delta(cpu_result.rsrp_dB, gpu_result.rsrp_dB));

    double noise_rel = 0.0;
    if (cpu_result.noise_variance.has_value() && gpu_result.noise_variance.has_value()) {
      double denom = std::max(std::abs(static_cast<double>(cpu_result.noise_variance.value())), 1e-12);
      noise_rel =
          std::abs(static_cast<double>(gpu_result.noise_variance.value() - cpu_result.noise_variance.value())) / denom;
    }

    matrix_rel_sum += matrix_rel;
    matrix_rel_max = std::max(matrix_rel_max, matrix_rel);
    ta_abs_ns_sum += ta_abs_ns;
    ta_abs_ns_max = std::max(ta_abs_ns_max, ta_abs_ns);
    epre_abs_db_sum += epre_abs_db;
    rsrp_abs_db_sum += rsrp_abs_db;
    noise_rel_sum += noise_rel;
    noise_rel_max = std::max(noise_rel_max, noise_rel);
    ++nof_observations;
  }

  double mean(double value) const { return value / static_cast<double>(nof_observations); }
};

static void usage(const char* prog)
{
  fmt::print("Usage: {} [-F frames] [-r rx_ports] [-t tx_ports] [-s symbols] [-A snr_start] [-B snr_stop] "
             "[-D snr_step] [-G host|visible] [-P single|quick|full|profile_name]\n",
             prog);
  fmt::print("\t-F Frames per SNR point [Default {}]\n", nof_frames);
  fmt::print("\t-r Number of RX ports [Default {}]\n", nof_rx_ports);
  fmt::print("\t-t Number of TX ports, one of 1, 2, 4 [Default {}]\n", nof_tx_ports);
  fmt::print("\t-s Number of SRS symbols [Default {}]\n", nof_symbols);
  fmt::print("\t-A SNR sweep start in dB [Default {}]\n", snr_start_db);
  fmt::print("\t-B SNR sweep stop in dB [Default {}]\n", snr_stop_db);
  fmt::print("\t-D SNR sweep step in dB [Default {}]\n", snr_step_db);
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
  while ((opt = getopt(argc, argv, "F:r:t:s:A:B:D:G:P:h")) != -1) {
    switch (opt) {
      case 'F':
        nof_frames = std::strtoul(optarg, nullptr, 10);
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
      case 'A':
        snr_start_db = std::strtof(optarg, nullptr);
        break;
      case 'B':
        snr_stop_db = std::strtof(optarg, nullptr);
        break;
      case 'D':
        snr_step_db = std::strtof(optarg, nullptr);
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

static void run_profile_snr(estimator_pair&              estimators,
                            const srs_benchmark_profile& profile,
                            unsigned                     grid_nof_symbols,
                            unsigned                     grid_nof_subc,
                            float                        snr_db,
                            sweep_accumulator&           acc,
                            unsigned                     seed_offset)
{
  auto config = make_srs_config(profile);
  auto grid   = create_grid(get_required_grid_ports(profile), grid_nof_symbols, grid_nof_subc, mode);
  TESTASSERT(grid);

  for (unsigned i_frame = 0; i_frame != nof_frames; ++i_frame) {
    populate_srs_grid(*grid, config, snr_db, seed_offset + i_frame);
    srs_estimator_result cpu_result = estimators.cpu->estimate(grid->get_reader(), config);
    srs_estimator_result gpu_result = estimators.gpu->estimate(grid->get_reader(), config);
    acc.push(cpu_result, gpu_result);
  }
}

int main(int argc, char** argv)
{
  parse_args(argc, argv);

#ifndef ENABLE_CUDA
  fmt::print("SRS GPU sensitivity sweep requires a CUDA-enabled build.\n");
  return 0;
#else
  TESTASSERT(nof_frames != 0, "The number of frames cannot be zero.");
  TESTASSERT(snr_step_db > 0.0F, "The SNR step must be positive.");
  TESTASSERT(snr_stop_db >= snr_start_db, "The SNR stop must be greater than or equal to the start.");
  TESTASSERT(nof_rx_ports >= 1 && nof_rx_ports <= srs_constants::max_nof_rx_ports);
  TESTASSERT(nof_symbols >= 1 && nof_symbols <= 4);
  (void)to_srs_ports(nof_tx_ports);

  unsigned                           grid_nof_symbols = get_nsymb_per_slot(cyclic_prefix::NORMAL);
  unsigned                           grid_nof_subc    = MAX_NOF_SUBCARRIERS;
  std::vector<srs_benchmark_profile> profiles;
  if (profile_set == "single") {
    profiles.push_back({"single", nof_rx_ports, nof_tx_ports, {0, 1, 2, 3}, subcarrier_spacing::kHz30, nof_symbols});
  } else if ((profile_set == "quick") || (profile_set == "full")) {
    bool full = profile_set == "full";
    profiles  = make_supported_srs_profiles(full);
  } else {
    std::optional<srs_benchmark_profile> profile = find_supported_srs_profile(profile_set);
    TESTASSERT(profile.has_value(),
               "Unsupported profile '{}'. Use single, quick, full or one of: {}.",
               profile_set,
               supported_srs_profile_names());
    profiles.push_back(profile.value());
  }

  estimator_pair estimators = make_estimators();

  fmt::print("SRS GPU sensitivity sweep: grid={} profile_set={} profiles={} frames_per_snr={}\n",
             to_string(mode),
             profile_set,
             profiles.size(),
             nof_frames);
  if (profile_set == "single") {
    fmt::print("profile=\"{}\"\n", describe_profile(profiles.front()));
  }
  fmt::print("snr_db\tprofiles\tobservations\tmatrix_rel_mean\tmatrix_rel_max\tta_abs_ns_mean\tta_abs_ns_max\tepre_abs_"
             "db_mean\t"
             "rsrp_abs_db_mean\tnoise_rel_mean\tnoise_rel_max\n");

  for (float snr_db = snr_start_db; snr_db <= snr_stop_db + 0.5F * snr_step_db; snr_db += snr_step_db) {
    sweep_accumulator acc;
    for (unsigned i_profile = 0; i_profile != profiles.size(); ++i_profile) {
      run_profile_snr(
          estimators, profiles[i_profile], grid_nof_symbols, grid_nof_subc, snr_db, acc, 1000U + i_profile * 10000U);
    }

    fmt::print("{:.2f}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6e}\t{:.6e}\n",
               snr_db,
               profiles.size(),
               acc.nof_observations,
               acc.mean(acc.matrix_rel_sum),
               acc.matrix_rel_max,
               acc.mean(acc.ta_abs_ns_sum),
               acc.ta_abs_ns_max,
               acc.mean(acc.epre_abs_db_sum),
               acc.mean(acc.rsrp_abs_db_sum),
               acc.mean(acc.noise_rel_sum),
               acc.noise_rel_max);
  }

  return 0;
#endif
}
