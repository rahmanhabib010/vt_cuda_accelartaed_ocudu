// SPDX-FileCopyrightText: Copyright (C) 2021-2026 DeepSig Inc
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

#include "srs_estimator_gpu_benchmark_helpers.h"
#include "ocudu/ran/srs/srs_information.h"
#include "fmt/format.h"
#include <getopt.h>

using namespace ocudu;
using namespace ocudu::srs_gpu_bench;

static grid_mode mode          = grid_mode::visible;
static float     snr_db        = 30.0F;
static bool      verbose       = false;
static bool      quick         = false;
static bool      exact_mode    = true;
static unsigned  max_failures  = 16;
static unsigned  max_nof_cases = 0;

namespace {

struct env_guard {
  explicit env_guard(const char* name_, const char* value) : name(name_)
  {
    const char* current = std::getenv(name);
    if (current != nullptr) {
      had_value = true;
      old_value = current;
    }
    setenv(name, value, 1);
  }

  ~env_guard()
  {
    if (had_value) {
      setenv(name, old_value.c_str(), 1);
    } else {
      unsetenv(name);
    }
  }

  const char* name;
  bool        had_value = false;
  std::string old_value;
};

struct srs_case {
  unsigned           rx_ports;
  unsigned           tx_ports;
  unsigned           rx_port_pattern;
  subcarrier_spacing scs;
  unsigned           nof_symbols;
  unsigned           start_symbol;
  tx_comb_size       comb_size;
  unsigned           comb_offset;
  unsigned           cyclic_shift;
  unsigned           configuration_index;
  unsigned           bandwidth_index;
  unsigned           freq_position;
  unsigned           freq_shift;
  unsigned           sequence_id;
};

const char* comb_to_string(tx_comb_size value)
{
  return value == tx_comb_size::n2 ? "n2" : "n4";
}

float relative_delta(float reference, float candidate)
{
  float denom = std::max(std::abs(reference), 1.0e-12F);
  return std::abs(candidate - reference) / denom;
}

bool resource_fits_grid(const srs_estimator_configuration& config, unsigned nof_subcarriers)
{
  for (unsigned i_tx_port = 0, nof_tx_ports = static_cast<unsigned>(config.resource.nof_antenna_ports);
       i_tx_port != nof_tx_ports;
       ++i_tx_port) {
    srs_information info = get_srs_information(config.resource, i_tx_port);
    if (info.mapping_initial_subcarrier + (info.sequence_length - 1) * info.comb_size >= nof_subcarriers) {
      return false;
    }
  }
  return true;
}

std::vector<srs_case> make_cases()
{
  struct port_case {
    unsigned rx_ports;
    unsigned tx_ports;
    unsigned rx_port_pattern;
  };
  struct scs_case {
    subcarrier_spacing scs;
  };
  struct time_case {
    unsigned nof_symbols;
    unsigned start_symbol;
  };
  struct comb_case {
    tx_comb_size comb_size;
    unsigned     comb_offset;
    unsigned     cyclic_shift;
  };
  struct bw_case {
    unsigned configuration_index;
    unsigned bandwidth_index;
    unsigned freq_position;
    unsigned freq_shift;
  };

  const std::vector<port_case> port_cases =
      quick ? std::vector<port_case>{{1, 1, 0}, {2, 2, 1}, {4, 4, 0}}
            : std::vector<port_case>{
                  {1, 1, 0}, {1, 1, 1}, {2, 1, 0}, {2, 1, 1}, {4, 1, 0}, {2, 2, 0}, {2, 2, 1}, {4, 2, 0}, {4, 4, 0}};
  const std::vector<scs_case>  scs_cases = quick ? std::vector<scs_case>{{subcarrier_spacing::kHz30}}
                                                 : std::vector<scs_case>{{subcarrier_spacing::kHz15},
                                                                         {subcarrier_spacing::kHz30},
                                                                         {subcarrier_spacing::kHz60},
                                                                         {subcarrier_spacing::kHz120}};
  const std::vector<time_case> time_cases =
      quick ? std::vector<time_case>{{1, 13}, {4, 10}} : std::vector<time_case>{{1, 13}, {2, 12}, {4, 10}, {4, 0}};
  const std::vector<comb_case> comb_cases = {
      {tx_comb_size::n2, 0, 0},
      {tx_comb_size::n2, 1, 3},
      {tx_comb_size::n2, 0, 4},
      {tx_comb_size::n2, 1, 7},
      {tx_comb_size::n4, 0, 0},
      {tx_comb_size::n4, 1, 5},
      {tx_comb_size::n4, 2, 6},
      {tx_comb_size::n4, 3, 11},
  };
  const std::vector<bw_case> bw_cases = quick ? std::vector<bw_case>{{0, 0, 0, 0}, {31, 2, 1, 0}, {63, 0, 0, 0}}
                                              : std::vector<bw_case>{{0, 0, 0, 0},
                                                                     {7, 1, 1, 0},
                                                                     {14, 2, 3, 0},
                                                                     {31, 2, 5, 1},
                                                                     {36, 3, 7, 0},
                                                                     {53, 1, 9, 0},
                                                                     {63, 0, 0, 0},
                                                                     {63, 3, 1, 0}};

  std::vector<srs_case> cases;
  for (const bw_case& bw : bw_cases) {
    for (const comb_case& comb : comb_cases) {
      for (const scs_case& scs : scs_cases) {
        for (const port_case& ports : port_cases) {
          for (const time_case& time : time_cases) {
            cases.push_back({ports.rx_ports,
                             ports.tx_ports,
                             ports.rx_port_pattern,
                             scs.scs,
                             time.nof_symbols,
                             time.start_symbol,
                             comb.comb_size,
                             comb.comb_offset,
                             comb.cyclic_shift,
                             bw.configuration_index,
                             bw.bandwidth_index,
                             bw.freq_position,
                             bw.freq_shift,
                             bw.configuration_index + 17U * bw.bandwidth_index});
          }
        }
      }
    }
  }

  if ((max_nof_cases != 0) && (cases.size() > max_nof_cases)) {
    cases.resize(max_nof_cases);
  }
  return cases;
}

void print_case(const srs_case& test_case)
{
  fmt::print("rx={} tx={} rxpat={} scs={} sym={} start={} comb={} off={} cs={} cSRS={} bSRS={} fpos={} fshift={} "
             "seq={}",
             test_case.rx_ports,
             test_case.tx_ports,
             test_case.rx_port_pattern,
             to_string(test_case.scs),
             test_case.nof_symbols,
             test_case.start_symbol,
             comb_to_string(test_case.comb_size),
             test_case.comb_offset,
             test_case.cyclic_shift,
             test_case.configuration_index,
             test_case.bandwidth_index,
             test_case.freq_position,
             test_case.freq_shift,
             test_case.sequence_id);
}

void apply_rx_port_pattern(srs_estimator_configuration& config, unsigned pattern)
{
  if (pattern == 0) {
    return;
  }
  if (config.ports.size() == 1) {
    config.ports[0] = 3;
    return;
  }
  if (config.ports.size() == 2) {
    config.ports[0] = 1;
    config.ports[1] = 3;
  }
}

} // namespace

static void usage(const char* prog)
{
  fmt::print("Usage: {} [-G host|visible] [-S snr_db] [-q] [-N max_cases] [-w] [-v]\n", prog);
  fmt::print("\t-G Resource-grid mode [Default {}]\n", to_string(mode));
  fmt::print("\t-S SNR in dB [Default {}]\n", snr_db);
  fmt::print("\t-q Quick sweep\n");
  fmt::print("\t-N Maximum number of generated cases, 0 means all selected cases [Default {}]\n", max_nof_cases);
  fmt::print("\t-w Use the default windowed-correlation low-latency path instead of exact full correlation\n");
  fmt::print("\t-v Verbose per-case output\n");
  fmt::print("\t-h Show this message\n");
}

static void parse_args(int argc, char** argv)
{
  int opt = 0;
  while ((opt = getopt(argc, argv, "G:S:N:qwvh")) != -1) {
    switch (opt) {
      case 'G':
        mode = parse_grid_mode(optarg);
        break;
      case 'S':
        snr_db = std::strtof(optarg, nullptr);
        break;
      case 'N':
        max_nof_cases = std::strtoul(optarg, nullptr, 10);
        break;
      case 'q':
        quick = true;
        break;
      case 'w':
        exact_mode = false;
        break;
      case 'v':
        verbose = true;
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
  fmt::print("SRS GPU configuration sweep requires a CUDA-enabled build.\n");
  return 0;
#else
  env_guard full_correlation("OCUDU_SRS_ACCELERATION_FULL_CORRELATION", exact_mode ? "1" : "0");
  env_guard force_gpu("OCUDU_SRS_ACCELERATION_FORCE", "1");

  unsigned grid_nof_symbols = get_nsymb_per_slot(cyclic_prefix::NORMAL);
  unsigned grid_nof_subc    = MAX_NOF_SUBCARRIERS;
  auto     grid             = create_grid(srs_constants::max_nof_rx_ports, grid_nof_symbols, grid_nof_subc, mode);
  TESTASSERT(grid);

  estimator_pair estimators = make_estimators();

  unsigned nof_run      = 0;
  unsigned nof_skipped  = 0;
  unsigned nof_failures = 0;

  for (const srs_case& test_case : make_cases()) {
    auto config = make_srs_config(test_case.rx_ports,
                                  test_case.tx_ports,
                                  test_case.nof_symbols,
                                  test_case.start_symbol,
                                  test_case.scs,
                                  test_case.comb_size,
                                  test_case.comb_offset,
                                  test_case.cyclic_shift,
                                  test_case.configuration_index,
                                  test_case.bandwidth_index,
                                  test_case.freq_position,
                                  test_case.freq_shift,
                                  test_case.sequence_id);
    apply_rx_port_pattern(config, test_case.rx_port_pattern);

    if (!resource_fits_grid(config, grid_nof_subc)) {
      ++nof_skipped;
      continue;
    }

    populate_srs_grid(*grid, config, snr_db, nof_run + 1U);
    srs_estimator_result cpu_result = estimators.cpu->estimate(grid->get_reader(), config);
    srs_estimator_result gpu_result = estimators.gpu->estimate(grid->get_reader(), config);

    float  matrix_rel_error = channel_matrix_relative_error(cpu_result.channel_matrix, gpu_result.channel_matrix);
    double ta_delta_ns =
        std::abs(cpu_result.time_alignment.time_alignment - gpu_result.time_alignment.time_alignment) * 1e9;
    float epre_delta_db = std::abs(optional_delta(cpu_result.epre_dB, gpu_result.epre_dB));
    float rsrp_delta_db = std::abs(optional_delta(cpu_result.rsrp_dB, gpu_result.rsrp_dB));
    float noise_rel =
        relative_delta(cpu_result.noise_variance.value_or(0.0F), gpu_result.noise_variance.value_or(0.0F));

    float  matrix_threshold = exact_mode ? 2.0e-2F : 6.0e-2F;
    double ta_threshold_ns  = exact_mode ? 1.0e-3 : 5.0;
    float  metric_threshold = exact_mode ? 1.0e-4F : 1.0e-2F;
    float  noise_threshold  = exact_mode ? 3.0e-2F : 1.2e-1F;

    bool case_ok = (matrix_rel_error <= matrix_threshold) && (ta_delta_ns <= ta_threshold_ns) &&
                   (epre_delta_db <= metric_threshold) && (rsrp_delta_db <= metric_threshold) &&
                   (noise_rel <= noise_threshold);

    if (verbose || !case_ok) {
      print_case(test_case);
      fmt::print(" -> matrix_rel={:.3e} ta_ns={:.3e} epre_db={:.3e} rsrp_db={:.3e} noise_rel={:.3e} {}\n",
                 matrix_rel_error,
                 ta_delta_ns,
                 epre_delta_db,
                 rsrp_delta_db,
                 noise_rel,
                 case_ok ? "PASS" : "FAIL");
    }

    ++nof_run;
    if (!case_ok) {
      ++nof_failures;
      if (nof_failures >= max_failures) {
        break;
      }
    }
  }

  fmt::print("SRS GPU configuration sweep: grid={} cases_run={} skipped={} failures={} correlation={}\n",
             to_string(mode),
             nof_run,
             nof_skipped,
             nof_failures,
             exact_mode ? "full" : "windowed");

  return nof_failures == 0 ? 0 : 1;
#endif
}
