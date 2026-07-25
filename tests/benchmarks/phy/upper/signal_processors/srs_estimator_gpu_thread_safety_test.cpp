// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "srs_estimator_gpu_benchmark_helpers.h"
#include "fmt/format.h"
#include <atomic>
#include <getopt.h>
#include <thread>

using namespace ocudu;
using namespace ocudu::srs_gpu_bench;

static unsigned    nof_threads    = 4;
static unsigned    nof_iterations = 32;
static float       snr_db         = 30.0F;
static grid_mode   mode           = grid_mode::visible;
static std::string profile_name   = "baseline_4x4_n4";

static void usage(const char* prog)
{
  fmt::print("Usage: {} [-T threads] [-I iterations] [-S snr_db] [-G host|visible] [-P profile_name]\n", prog);
  fmt::print("\t-T Concurrent SRS estimator calls [Default {}]\n", nof_threads);
  fmt::print("\t-I Iterations per thread [Default {}]\n", nof_iterations);
  fmt::print("\t-S SNR in dB [Default {}]\n", snr_db);
  fmt::print("\t-G Resource-grid mode: visible uses CUDA-visible grid, host uses staged host grid [Default {}]\n",
             to_string(mode));
  fmt::print("\t-P Named profile [Default {}]\n", profile_name);
  fmt::print("\t   Named profiles: {}\n", supported_srs_profile_names());
  fmt::print("\t-h Show this message\n");
}

static void parse_args(int argc, char** argv)
{
  int opt = 0;
  while ((opt = getopt(argc, argv, "T:I:S:G:P:h")) != -1) {
    switch (opt) {
      case 'T':
        nof_threads = std::strtoul(optarg, nullptr, 10);
        break;
      case 'I':
        nof_iterations = std::strtoul(optarg, nullptr, 10);
        break;
      case 'S':
        snr_db = std::strtof(optarg, nullptr);
        break;
      case 'G':
        mode = parse_grid_mode(optarg);
        break;
      case 'P':
        profile_name = optarg;
        break;
      case 'h':
      default:
        usage(argv[0]);
        std::exit(0);
    }
  }
}

int main(int argc, char** argv)
{
  parse_args(argc, argv);

#ifndef ENABLE_CUDA
  fmt::print("SRS GPU thread-safety test requires a CUDA-enabled build.\n");
  return 0;
#else
  TESTASSERT(nof_threads != 0, "The number of threads cannot be zero.");
  TESTASSERT(nof_iterations != 0, "The number of iterations cannot be zero.");

  std::optional<srs_benchmark_profile> profile = find_supported_srs_profile(profile_name);
  TESTASSERT(
      profile.has_value(), "Unsupported profile '{}'. Use one of: {}.", profile_name, supported_srs_profile_names());

  srs_estimator_configuration config = make_srs_config(profile.value());
  auto                        grid   = create_grid(
      get_required_grid_ports(profile.value()), get_nsymb_per_slot(cyclic_prefix::NORMAL), MAX_NOF_SUBCARRIERS, mode);
  TESTASSERT(grid);
  populate_srs_grid(*grid, config, snr_db, 0);

  std::shared_ptr<srs_estimator_factory> cpu_factory = make_srs_estimator_factory("disabled");
  TESTASSERT(cpu_factory);
  std::unique_ptr<srs_estimator> cpu_estimator = cpu_factory->create();
  TESTASSERT(cpu_estimator);
  srs_estimator_result reference = cpu_estimator->estimate(grid->get_reader(), config);

  std::shared_ptr<srs_estimator_factory> gpu_factory = make_srs_estimator_factory("auto");
  TESTASSERT(gpu_factory);
  std::shared_ptr<srs_estimator_factory> pooled_factory = create_srs_estimator_pool(gpu_factory, nof_threads);
  TESTASSERT(pooled_factory);
  std::unique_ptr<srs_estimator> pooled_estimator = pooled_factory->create();
  TESTASSERT(pooled_estimator);

  std::atomic<bool>        start{false};
  std::atomic<unsigned>    failures{0};
  std::vector<std::thread> workers;
  workers.reserve(nof_threads);

  for (unsigned i_thread = 0; i_thread != nof_threads; ++i_thread) {
    workers.emplace_back([&, i_thread]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      for (unsigned i_iteration = 0; i_iteration != nof_iterations; ++i_iteration) {
        srs_estimator_result result = pooled_estimator->estimate(grid->get_reader(), config);
        float  matrix_rel_error     = channel_matrix_relative_error(reference.channel_matrix, result.channel_matrix);
        double ta_delta_ns =
            std::abs(reference.time_alignment.time_alignment - result.time_alignment.time_alignment) * 1e9;
        float epre_delta_db = std::abs(optional_delta(reference.epre_dB, result.epre_dB));
        float rsrp_delta_db = std::abs(optional_delta(reference.rsrp_dB, result.rsrp_dB));
        float noise_delta   = std::abs(optional_delta(reference.noise_variance, result.noise_variance));

        bool ok = (matrix_rel_error <= 6.0e-2F) && (ta_delta_ns <= 5.0) && (epre_delta_db <= 1.0e-2F) &&
                  (rsrp_delta_db <= 1.0e-2F) && (noise_delta <= 5.0e-2F);
        if (!ok) {
          fmt::print("thread={} iteration={} matrix_rel={:.3e} ta_ns={:.3e} epre_db={:.3e} rsrp_db={:.3e} "
                     "noise_delta={:.3e}\n",
                     i_thread,
                     i_iteration,
                     matrix_rel_error,
                     ta_delta_ns,
                     epre_delta_db,
                     rsrp_delta_db,
                     noise_delta);
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (std::thread& worker : workers) {
    worker.join();
  }

  unsigned nof_failures = failures.load(std::memory_order_relaxed);
  fmt::print("SRS GPU thread-safety test: grid={} profile=\"{}\" threads={} iterations={} failures={}\n",
             to_string(mode),
             describe_profile(profile.value()),
             nof_threads,
             nof_iterations,
             nof_failures);

  return nof_failures == 0 ? 0 : 1;
#endif
}
