// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief Compares CPU PUSCH MIMO equalizer output against CUDA MIMO E2E equalizer output.

#include "pxsch_bler_test_channel_emulator.h"
#include "pxsch_bler_test_factories.h"
#include "ocudu/phy/support/resource_grid.h"
#include "ocudu/phy/support/support_factories.h"
#include "ocudu/phy/upper/channel_processors/pdsch/pdsch_processor.h"
#include "ocudu/phy/upper/equalization/dynamic_ch_est_list.h"
#include "ocudu/phy/upper/equalization/equalization_factories.h"
#include "ocudu/phy/upper/dmrs_mapping.h"
#include "ocudu/phy/generic_functions/generic_functions_factories.h"
#include "ocudu/phy/upper/sequence_generators/sequence_generator_factories.h"
#include "ocudu/phy/support/time_alignment_estimator/time_alignment_estimator_factories.h"
#include "ocudu/phy/upper/signal_processors/channel_estimator/factories.h"
#include "ocudu/phy/upper/signal_processors/pusch/dmrs_pusch_estimator.h"
#include "ocudu/phy/upper/signal_processors/pusch/factories.h"
#include "ocudu/ran/precoding/precoding_codebooks.h"
#include "ocudu/ran/pusch/pusch_constants.h"
#include "ocudu/ran/pusch/pusch_mcs.h"
#include "ocudu/ran/resource_allocation/rb_interval.h"
#include "ocudu/ran/sch/sch_dmrs_power.h"
#include "ocudu/ran/sch/sch_mcs.h"
#include "ocudu/ran/sch/sch_segmentation.h"
#include "ocudu/ran/sch/tbs_calculator.h"
#include "ocudu/support/executors/task_worker_pool.h"

#ifdef ENABLE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pusch_e2e.h>
#endif

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>

using namespace ocudu;

static constexpr subcarrier_spacing scs                         = subcarrier_spacing::kHz30;
static constexpr cyclic_prefix      cyclic_prefix_value         = cyclic_prefix::NORMAL;
static constexpr unsigned           nof_ofdm_symbols             = 14;
static constexpr unsigned           bwp_start_rb                 = 0;
static constexpr uint16_t           rnti                         = 0x1234;
static constexpr unsigned           n_id                         = 0;
static constexpr unsigned           scrambling_id                = 0;
static constexpr bool               n_scid                       = false;
static constexpr dmrs_config_type          dmrs                         = dmrs_config_type::type1;
static constexpr unsigned           nof_cdm_groups_without_data  = 2;
static const symbol_slot_mask       dmrs_symbols_mask             = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
static constexpr unsigned           nof_prb                      = 24;
static unsigned                     nof_layers                   = 2;
static unsigned                     nof_rx_ports                 = 2;
static channel_equalizer_algorithm_type equalizer_algorithm      = channel_equalizer_algorithm_type::mmse;
[[maybe_unused]] static bool        disable_cpu_fd_smoothing     = false;
[[maybe_unused]] static bool        disable_cpu_cfo_compensation = false;
static constexpr sch_mcs_index      mcs_index                    = 10;
static constexpr pusch_mcs_table    mcs_table                    = pusch_mcs_table::qam64;
static constexpr float              channel_sinr_dB              = 45.0F;

namespace {

class pdsch_notifier_adaptor : public pdsch_processor_notifier
{
public:
  void on_finish_processing() override { completed = true; }

  void wait()
  {
    while (!completed.load()) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

private:
  std::atomic<bool> completed = false;
};

class estimator_notifier_adaptor : public dmrs_pusch_estimator_notifier
{
public:
  void on_estimation_complete(const dmrs_pusch_estimator_results& results_) override
  {
    std::unique_lock<std::mutex> lock(mutex);
    results = &results_;
    cvar.notify_all();
  }

  const dmrs_pusch_estimator_results& wait()
  {
    std::unique_lock<std::mutex> lock(mutex);
    while (results == nullptr) {
      cvar.wait(lock);
    }
    return *results;
  }

private:
  const dmrs_pusch_estimator_results* results = nullptr;
  std::mutex                          mutex;
  std::condition_variable             cvar;
};

[[maybe_unused]] static std::shared_ptr<resource_grid_factory> make_grid_factory()
{
  auto precoder_factory = create_channel_precoder_factory("auto");
  report_fatal_error_if_not(precoder_factory, "Failed to create channel precoder factory.");
  return create_resource_grid_factory();
}

[[maybe_unused]] static std::vector<int> build_re_indices(const crb_bitmap& rb_mask)
{
  std::vector<int> indices;
  re_prb_mask      active_re_per_prb      = ~re_prb_mask();
  re_prb_mask      active_re_per_prb_dmrs = ~get_dmrs_prb_mask(dmrs, nof_cdm_groups_without_data);
  auto             re_mask                = rb_mask.kronecker_product<NOF_SUBCARRIERS_PER_RB>(active_re_per_prb);
  auto             re_mask_dmrs           = rb_mask.kronecker_product<NOF_SUBCARRIERS_PER_RB>(active_re_per_prb_dmrs);

  for (unsigned i_symbol = 0; i_symbol != nof_ofdm_symbols; ++i_symbol) {
    const auto& symbol_re_mask = dmrs_symbols_mask.test(i_symbol) ? re_mask_dmrs : re_mask;
    unsigned    base           = i_symbol * MAX_NOF_SUBCARRIERS;
    symbol_re_mask.for_each(0, symbol_re_mask.size(), [&](unsigned k) { indices.push_back(base + k); });
  }
  return indices;
}

[[maybe_unused]] static std::vector<cf_t> compute_cpu_eq(const resource_grid_reader&         grid,
                                                         const dmrs_pusch_estimator_results& est_results,
                                                         const crb_bitmap&                   rb_mask,
                                                         channel_equalizer&                  equalizer,
                                                         std::vector<float>* noise_out = nullptr)
{
  std::vector<cf_t>  eq_all;
  std::vector<float> noise_all;
  dynamic_re_buffer<cbf16_t> ch_re(pusch_constants::MAX_NOF_RX_PORTS, MAX_NOF_SUBCARRIERS);
  dynamic_ch_est_list        ch_estimates;
  std::array<float, pusch_constants::MAX_NOF_RX_PORTS> noise_var_estimates = {};
  static_vector<uint8_t, MAX_PORTS> rx_ports(nof_rx_ports);
  std::iota(rx_ports.begin(), rx_ports.end(), 0U);

  re_prb_mask active_re_per_prb      = ~re_prb_mask();
  re_prb_mask active_re_per_prb_dmrs = ~get_dmrs_prb_mask(dmrs, nof_cdm_groups_without_data);
  auto        re_mask                = rb_mask.kronecker_product<NOF_SUBCARRIERS_PER_RB>(active_re_per_prb);
  auto        re_mask_dmrs           = rb_mask.kronecker_product<NOF_SUBCARRIERS_PER_RB>(active_re_per_prb_dmrs);

  for (unsigned i_port = 0; i_port != nof_rx_ports; ++i_port) {
    noise_var_estimates[i_port] = est_results.get_noise_variance(i_port);
  }
  fmt::print("CPU pre-EQ noise:");
  for (unsigned i_port = 0; i_port != nof_rx_ports; ++i_port) {
    fmt::print(" p{}={:.6e}", i_port, noise_var_estimates[i_port]);
  }
  fmt::print("\n");

  for (unsigned i_symbol = 0; i_symbol != nof_ofdm_symbols; ++i_symbol) {
    const auto& symbol_re_mask = dmrs_symbols_mask.test(i_symbol) ? re_mask_dmrs : re_mask;
    unsigned    nof_re_symbol  = symbol_re_mask.count();
    if (nof_re_symbol == 0) {
      continue;
    }

    interval<unsigned> re_interval(rb_mask.find_lowest() * NOF_SUBCARRIERS_PER_RB,
                                   (rb_mask.find_highest() + 1) * NOF_SUBCARRIERS_PER_RB);
    auto               symbol_re_mask_local = symbol_re_mask.slice(re_interval.start(), re_interval.stop());

    ch_re.resize(nof_rx_ports, nof_re_symbol);
    for (unsigned i_port = 0; i_port != nof_rx_ports; ++i_port) {
      span<cbf16_t> tail = grid.get(ch_re.get_slice(i_port), rx_ports[i_port], i_symbol, 0, symbol_re_mask);
      report_fatal_error_if_not(tail.empty(), "Failed to read all grid RE.");
    }

    ch_estimates.resize(nof_re_symbol, nof_rx_ports, nof_layers);
    for (unsigned i_port = 0; i_port != nof_rx_ports; ++i_port) {
      for (unsigned i_layer = 0; i_layer != nof_layers; ++i_layer) {
        est_results.get_symbol_ch_estimate(
            ch_estimates.get_channel(i_port, i_layer), i_symbol, i_port, i_layer, symbol_re_mask_local);
      }
    }

    std::vector<cf_t>  eq_symbol(nof_re_symbol * nof_layers);
    std::vector<float> noise_symbol(nof_re_symbol * nof_layers);
    equalizer.equalize(eq_symbol,
                       noise_symbol,
                       ch_re,
                       ch_estimates,
                       span<float>(noise_var_estimates).first(nof_rx_ports),
                       1.0F);
    eq_all.insert(eq_all.end(), eq_symbol.begin(), eq_symbol.end());
    noise_all.insert(noise_all.end(), noise_symbol.begin(), noise_symbol.end());
  }

  if (noise_out != nullptr) {
    *noise_out = std::move(noise_all);
  }

  return eq_all;
}

[[maybe_unused]] static std::vector<cf_t> compute_cpu_ce(const dmrs_pusch_estimator_results& est_results,
                                                         const crb_bitmap&                   rb_mask)
{
  std::vector<cf_t> ce_all;

  re_prb_mask active_re_per_prb      = ~re_prb_mask();
  re_prb_mask active_re_per_prb_dmrs = ~get_dmrs_prb_mask(dmrs, nof_cdm_groups_without_data);
  auto        re_mask                = rb_mask.kronecker_product<NOF_SUBCARRIERS_PER_RB>(active_re_per_prb);
  auto        re_mask_dmrs           = rb_mask.kronecker_product<NOF_SUBCARRIERS_PER_RB>(active_re_per_prb_dmrs);

  for (unsigned i_symbol = 0; i_symbol != nof_ofdm_symbols; ++i_symbol) {
    const auto& symbol_re_mask = dmrs_symbols_mask.test(i_symbol) ? re_mask_dmrs : re_mask;
    unsigned    nof_re_symbol  = symbol_re_mask.count();
    if (nof_re_symbol == 0) {
      continue;
    }

    interval<unsigned> re_interval(rb_mask.find_lowest() * NOF_SUBCARRIERS_PER_RB,
                                   (rb_mask.find_highest() + 1) * NOF_SUBCARRIERS_PER_RB);
    auto               symbol_re_mask_local = symbol_re_mask.slice(re_interval.start(), re_interval.stop());

    dynamic_ch_est_list ch_estimates;
    ch_estimates.resize(nof_re_symbol, nof_rx_ports, nof_layers);
    for (unsigned i_port = 0; i_port != nof_rx_ports; ++i_port) {
      for (unsigned i_layer = 0; i_layer != nof_layers; ++i_layer) {
        est_results.get_symbol_ch_estimate(
            ch_estimates.get_channel(i_port, i_layer), i_symbol, i_port, i_layer, symbol_re_mask_local);
      }
    }

    for (unsigned i_re = 0; i_re != nof_re_symbol; ++i_re) {
      for (unsigned i_port = 0; i_port != nof_rx_ports; ++i_port) {
        for (unsigned i_layer = 0; i_layer != nof_layers; ++i_layer) {
          ce_all.emplace_back(to_cf(ch_estimates.get_channel(i_port, i_layer)[i_re]));
        }
      }
    }
  }

  return ce_all;
}

[[maybe_unused]] static std::vector<cf_t> compute_cpu_eq_from_flat_ce(const resource_grid_reader& grid,
                                                                      const crb_bitmap&           rb_mask,
                                                                      span<const cf_t>            ce_values,
                                                                      span<const float> noise_var_estimates,
                                                                      channel_equalizer& equalizer,
                                                                      std::vector<float>* noise_out = nullptr)
{
  std::vector<cf_t>  eq_all;
  std::vector<float> noise_all;
  dynamic_re_buffer<cbf16_t> ch_re(pusch_constants::MAX_NOF_RX_PORTS, MAX_NOF_SUBCARRIERS);
  dynamic_ch_est_list        ch_estimates;
  static_vector<uint8_t, MAX_PORTS> rx_ports(nof_rx_ports);
  std::iota(rx_ports.begin(), rx_ports.end(), 0U);

  re_prb_mask active_re_per_prb      = ~re_prb_mask();
  re_prb_mask active_re_per_prb_dmrs = ~get_dmrs_prb_mask(dmrs, nof_cdm_groups_without_data);
  auto        re_mask                = rb_mask.kronecker_product<NOF_SUBCARRIERS_PER_RB>(active_re_per_prb);
  auto        re_mask_dmrs           = rb_mask.kronecker_product<NOF_SUBCARRIERS_PER_RB>(active_re_per_prb_dmrs);

  unsigned ce_re_offset = 0;
  for (unsigned i_symbol = 0; i_symbol != nof_ofdm_symbols; ++i_symbol) {
    const auto& symbol_re_mask = dmrs_symbols_mask.test(i_symbol) ? re_mask_dmrs : re_mask;
    unsigned    nof_re_symbol  = symbol_re_mask.count();
    if (nof_re_symbol == 0) {
      continue;
    }

    ch_re.resize(nof_rx_ports, nof_re_symbol);
    for (unsigned i_port = 0; i_port != nof_rx_ports; ++i_port) {
      span<cbf16_t> tail = grid.get(ch_re.get_slice(i_port), rx_ports[i_port], i_symbol, 0, symbol_re_mask);
      report_fatal_error_if_not(tail.empty(), "Failed to read all grid RE.");
    }

    ch_estimates.resize(nof_re_symbol, nof_rx_ports, nof_layers);
    for (unsigned i_port = 0; i_port != nof_rx_ports; ++i_port) {
      for (unsigned i_layer = 0; i_layer != nof_layers; ++i_layer) {
        span<cbf16_t> channel = ch_estimates.get_channel(i_port, i_layer);
        for (unsigned i_re = 0; i_re != nof_re_symbol; ++i_re) {
          size_t ce_idx = (static_cast<size_t>(ce_re_offset + i_re) * nof_rx_ports + i_port) * nof_layers + i_layer;
          channel[i_re] = to_cbf16(ce_values[ce_idx]);
        }
      }
    }

    std::vector<cf_t>  eq_symbol(nof_re_symbol * nof_layers);
    std::vector<float> noise_symbol(nof_re_symbol * nof_layers);
    equalizer.equalize(eq_symbol, noise_symbol, ch_re, ch_estimates, noise_var_estimates, 1.0F);
    eq_all.insert(eq_all.end(), eq_symbol.begin(), eq_symbol.end());
    noise_all.insert(noise_all.end(), noise_symbol.begin(), noise_symbol.end());
    ce_re_offset += nof_re_symbol;
  }

  if (noise_out != nullptr) {
    *noise_out = std::move(noise_all);
  }

  return eq_all;
}

static bool invert_matrix(std::vector<cf_t>& inv, std::vector<cf_t> mat, unsigned n)
{
  inv.assign(n * n, {});
  for (unsigned i = 0; i != n; ++i) {
    inv[i * n + i] = {1.0F, 0.0F};
  }

  for (unsigned i = 0; i != n; ++i) {
    cf_t  pivot     = mat[i * n + i];
    float pivot_pow = std::norm(pivot);
    if (!std::isfinite(pivot_pow) || pivot_pow == 0.0F) {
      return false;
    }
    cf_t pivot_inv = std::conj(pivot) / pivot_pow;

    for (unsigned j = i + 1; j != n; ++j) {
      mat[i * n + j] *= pivot_inv;
    }
    for (unsigned j = 0; j <= i; ++j) {
      inv[i * n + j] *= pivot_inv;
    }

    for (unsigned k = 0; k != n; ++k) {
      if (k == i) {
        continue;
      }
      cf_t factor = mat[k * n + i];
      for (unsigned j = i; j != n; ++j) {
        mat[k * n + j] -= factor * mat[i * n + j];
      }
      for (unsigned j = 0; j != i; ++j) {
        inv[k * n + j] -= factor * inv[i * n + j];
      }
      inv[k * n + i] = -factor * inv[i * n + i];
    }
  }

  return true;
}

[[maybe_unused]] static std::vector<cf_t> compute_scalar_eq_from_flat_ce(const resource_grid_reader& grid,
                                                                         span<const int>             re_indices,
                                                                         span<const cf_t>            ce_values,
                                                                         span<const float> noise_var_estimates)
{
  std::vector<cf_t> eq_all(re_indices.size() * nof_layers);
  float             noise_var = *std::max_element(noise_var_estimates.begin(), noise_var_estimates.end());

  for (size_t i_re = 0; i_re != re_indices.size(); ++i_re) {
    unsigned src_re = static_cast<unsigned>(re_indices[i_re]);
    unsigned symbol = src_re / MAX_NOF_SUBCARRIERS;
    unsigned subc   = src_re % MAX_NOF_SUBCARRIERS;

    std::vector<cf_t> y(nof_rx_ports);
    for (unsigned port = 0; port != nof_rx_ports; ++port) {
      y[port] = to_cf(grid.get_view(port, symbol)[subc]);
    }

    std::vector<cf_t> gram(nof_layers * nof_layers, cf_t{});
    for (unsigned i = 0; i != nof_layers; ++i) {
      for (unsigned j = 0; j != nof_layers; ++j) {
        cf_t sum = {};
        for (unsigned port = 0; port != nof_rx_ports; ++port) {
          size_t ce_i = (i_re * nof_rx_ports + port) * nof_layers + i;
          size_t ce_j = (i_re * nof_rx_ports + port) * nof_layers + j;
          sum += std::conj(ce_values[ce_i]) * ce_values[ce_j];
        }
        if ((i == j) && (equalizer_algorithm == channel_equalizer_algorithm_type::mmse)) {
          sum += cf_t(noise_var, 0.0F);
        }
        gram[i * nof_layers + j] = sum;
      }
    }

    std::vector<cf_t> gram_inv;
    bool              valid = invert_matrix(gram_inv, gram, nof_layers);
    if (!valid) {
      continue;
    }

    std::vector<cf_t> mf(nof_layers, cf_t{});
    for (unsigned layer = 0; layer != nof_layers; ++layer) {
      for (unsigned port = 0; port != nof_rx_ports; ++port) {
        size_t ce_idx = (i_re * nof_rx_ports + port) * nof_layers + layer;
        mf[layer] += std::conj(ce_values[ce_idx]) * y[port];
      }
    }

    for (unsigned layer = 0; layer != nof_layers; ++layer) {
      cf_t eq = {};
      for (unsigned k = 0; k != nof_layers; ++k) {
        eq += gram_inv[layer * nof_layers + k] * mf[k];
      }
      if (equalizer_algorithm == channel_equalizer_algorithm_type::mmse) {
        float c = 1.0F - noise_var * gram_inv[layer * nof_layers + layer].real();
        eq *= (c > 1e-10F) ? (1.0F / c) : 0.0F;
      }
      eq_all[i_re * nof_layers + layer] = eq;
    }
  }

  return eq_all;
}

[[maybe_unused]] static unsigned count_dmrs_symbols()
{
  return dmrs_symbols_mask.count();
}

} // namespace

int main(int argc, char** argv)
{
#ifndef ENABLE_CUDA
  fmt::print("CUDA is not enabled, skipping.\n");
  return 0;
#else
  for (int i_arg = 1; i_arg != argc; ++i_arg) {
    std::string_view arg(argv[i_arg]);
    auto read_uint = [&](unsigned& value) {
      report_fatal_error_if_not(i_arg + 1 < argc, "Missing value for {}.", arg);
      value = static_cast<unsigned>(std::stoul(argv[++i_arg]));
    };
    if (arg == "--layers") {
      read_uint(nof_layers);
    } else if (arg == "--ports") {
      read_uint(nof_rx_ports);
    } else if (arg == "--equalizer") {
      report_fatal_error_if_not(i_arg + 1 < argc, "Missing value for --equalizer.");
      std::string value(argv[++i_arg]);
      if (value == "zf" || value == "ZF") {
        equalizer_algorithm = channel_equalizer_algorithm_type::zf;
      } else if (value == "mmse" || value == "MMSE") {
        equalizer_algorithm = channel_equalizer_algorithm_type::mmse;
      } else {
        report_fatal_error("Invalid --equalizer value '{}'. Expected zf or mmse.", value);
      }
    } else if (arg == "--disable-cpu-fd-smoothing") {
      disable_cpu_fd_smoothing = true;
    } else if (arg == "--disable-cpu-cfo") {
      disable_cpu_cfo_compensation = true;
    } else {
      report_fatal_error("Unknown argument {}.", arg);
    }
  }
  report_fatal_error_if_not((nof_layers >= 2 && nof_layers <= 4) && nof_layers <= nof_rx_ports &&
                                nof_rx_ports <= pusch_constants::MAX_NOF_RX_PORTS,
                            "Invalid MIMO dimensions: layers={} ports={}.",
                            nof_layers,
                            nof_rx_ports);

  ocudulog::init();
  ocudulog::fetch_basic_logger("ALL").set_level(ocudulog::basic_levels::warning);

  unsigned max_nof_threads = std::min(8U, std::thread::hardware_concurrency());
  task_worker_pool<concurrent_queue_policy::locking_mpmc> worker_pool("thread", max_nof_threads, 1024);
  task_worker_pool_executor<concurrent_queue_policy::locking_mpmc> executor(worker_pool);

  auto grid_factory = make_grid_factory();
  auto tx_grid      = grid_factory->create(nof_layers, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);
  auto rx_grid      = grid_factory->create(nof_rx_ports, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);

  auto pdsch_factory = create_sw_pdsch_processor_factory(executor, max_nof_threads + 1, "", "auto");
  report_fatal_error_if_not(pdsch_factory, "Failed to create PDSCH processor factory.");
  auto pdsch = pdsch_factory->create();

  sch_mcs_description mcs_descr = pusch_mcs_get_config(mcs_table, mcs_index, false, false);
  rb_allocation       freq_alloc = rb_allocation::make_type1(bwp_start_rb, nof_prb, std::nullopt);
  crb_bitmap          rb_mask    = freq_alloc.get_crb_mask(bwp_start_rb, nof_prb);

  tbs_calculator_configuration tbs_config = {};
  tbs_config.mcs_descr                    = mcs_descr;
  tbs_config.n_prb                        = nof_prb;
  tbs_config.nof_layers                   = nof_layers;
  tbs_config.nof_symb_sh                  = nof_ofdm_symbols;
  tbs_config.nof_dmrs_prb = get_nof_re_per_prb(dmrs) * dmrs_symbols_mask.count() * nof_cdm_groups_without_data;
  unsigned tbs            = tbs_calculator_calculate(tbs_config).to_bits().value();
  ldpc_base_graph_type ldpc_base_graph =
      get_ldpc_base_graph(mcs_descr.get_normalised_target_code_rate(), units::bits(tbs));

  std::vector<uint8_t> tx_data(tbs / 8);
  std::mt19937         rgen(0x5eed);
  for (uint8_t& byte : tx_data) {
    byte = static_cast<uint8_t>(rgen() & 0xff);
  }

  pdsch_processor::pdu_t pdsch_pdu;
  pdsch_pdu.slot                        = slot_point(to_numerology_value(scs), 1);
  pdsch_pdu.rnti                        = rnti;
  pdsch_pdu.bwp_size_rb                 = nof_prb;
  pdsch_pdu.bwp_start_rb                = bwp_start_rb;
  pdsch_pdu.cp                          = cyclic_prefix_value;
  pdsch_pdu.n_id                        = n_id;
  pdsch_pdu.ref_point                   = pdsch_processor::pdu_t::PRB0;
  pdsch_pdu.dmrs_symbol_mask            = dmrs_symbols_mask;
  pdsch_pdu.dmrs                        = dmrs;
  pdsch_pdu.scrambling_id               = scrambling_id;
  pdsch_pdu.n_scid                      = n_scid;
  pdsch_pdu.nof_cdm_groups_without_data = nof_cdm_groups_without_data;
  pdsch_pdu.freq_alloc                  = freq_alloc;
  pdsch_pdu.start_symbol_index          = 0;
  pdsch_pdu.nof_symbols                 = nof_ofdm_symbols;
  pdsch_pdu.ldpc_base_graph             = ldpc_base_graph;
  pdsch_pdu.tbs_lbrm                    = tbs_lbrm_default;
  pdsch_pdu.ratio_pdsch_data_to_sss_dB  = 0.0F;
  pdsch_pdu.ratio_pdsch_dmrs_to_sss_dB  = get_sch_to_dmrs_ratio_dB(nof_cdm_groups_without_data);
  pdsch_pdu.precoding                   = precoding_configuration::make_wideband(make_identity(nof_layers));
  pdsch_pdu.codewords.emplace_back(pdsch_processor::codeword_description{mcs_descr.modulation, 0});

  pdsch_notifier_adaptor pdsch_notifier;
  pdsch->process(tx_grid->get_writer(), pdsch_notifier, {shared_transport_block(tx_data)}, pdsch_pdu);
  pdsch_notifier.wait();

  channel_emulator emulator("single-tap",
                            "uniform-phase",
                            channel_sinr_dB,
                            0.0F,
                            0,
                            nof_layers,
                            nof_rx_ports,
                            nof_prb * NOF_SUBCARRIERS_PER_RB,
                            nof_ofdm_symbols,
                            max_nof_threads,
                            scs,
                            executor);
  emulator.run(rx_grid->get_writer(), tx_grid->get_reader());

  auto prg_factory          = create_pseudo_random_generator_sw_factory();
  auto low_papr_factory     = create_low_papr_sequence_generator_sw_factory();
  auto dft_factory          = create_dft_processor_factory_generic();
  auto ta_est_factory       = create_time_alignment_estimator_dft_factory(dft_factory);
  auto port_est_factory     = create_port_channel_estimator_factory_sw(ta_est_factory);
  auto pusch_est_factory    = create_dmrs_pusch_estimator_factory_sw(prg_factory,
                                                                  low_papr_factory,
                                                                  port_est_factory,
                                                                  executor,
                                                                  pusch_constants::MAX_NOF_RX_PORTS,
                                                                  disable_cpu_fd_smoothing ?
                                                                      port_channel_estimator_fd_smoothing_strategy::none :
                                                                      port_channel_estimator_fd_smoothing_strategy::filter,
                                                                  port_channel_estimator_td_interpolation_strategy::average,
                                                                  !disable_cpu_cfo_compensation);
  auto equalizer_factory    = create_channel_equalizer_generic_factory(equalizer_algorithm);
  auto estimator            = pusch_est_factory->create();
  auto equalizer            = equalizer_factory->create();

  dmrs_pusch_estimator::configuration est_config;
  est_config.slot            = slot_point(to_numerology_value(scs), 1);
  est_config.sequence_config = dmrs_pusch_estimator::pseudo_random_sequence_configuration{
      .type = dmrs, .nof_tx_layers = nof_layers, .scrambling_id = scrambling_id, .n_scid = n_scid};
  est_config.scaling      = convert_dB_to_amplitude(-get_sch_to_dmrs_ratio_dB(nof_cdm_groups_without_data));
  est_config.c_prefix     = cyclic_prefix_value;
  est_config.symbols_mask = dmrs_symbols_mask;
  est_config.rb_mask      = rb_mask;
  est_config.first_symbol = 0;
  est_config.nof_symbols  = nof_ofdm_symbols;
  for (unsigned port = 0; port != nof_rx_ports; ++port) {
    est_config.rx_ports.push_back(port);
  }

  estimator_notifier_adaptor estimator_notifier;
  estimator->estimate(estimator_notifier, rx_grid->get_reader(), est_config);
  const dmrs_pusch_estimator_results& est_results = estimator_notifier.wait();
  std::vector<float> cpu_pre_eq_noise(nof_rx_ports);
  for (unsigned i_port = 0; i_port != nof_rx_ports; ++i_port) {
    cpu_pre_eq_noise[i_port] = est_results.get_noise_variance(i_port);
  }

  std::vector<float> cpu_eq_noise;
  std::vector<cf_t>  cpu_eq = compute_cpu_eq(rx_grid->get_reader(), est_results, rb_mask, *equalizer, &cpu_eq_noise);
  std::vector<cf_t> cpu_ce = compute_cpu_ce(est_results, rb_mask);
  std::vector<int>  re_indices = build_re_indices(rb_mask);
  report_fatal_error_if_not(cpu_eq.size() == re_indices.size() * nof_layers,
                            "CPU EQ size {} does not match RE/layer count {}.",
                            cpu_eq.size(),
                            re_indices.size() * nof_layers);
  report_fatal_error_if_not(cpu_ce.size() == re_indices.size() * nof_rx_ports * nof_layers,
                            "CPU CE size {} does not match RE/port/layer count {}.",
                            cpu_ce.size(),
                            re_indices.size() * nof_rx_ports * nof_layers);

  std::vector<uint32_t> h_grid(nof_rx_ports * MAX_NSYMB_PER_SLOT * MAX_NOF_SUBCARRIERS);
  for (unsigned port = 0; port != nof_rx_ports; ++port) {
    for (unsigned sym = 0; sym != MAX_NSYMB_PER_SLOT; ++sym) {
      span<const cbf16_t> view = rx_grid->get_reader().get_view(port, sym);
      std::memcpy(&h_grid[(port * MAX_NSYMB_PER_SLOT + sym) * MAX_NOF_SUBCARRIERS],
                  view.data(),
                  MAX_NOF_SUBCARRIERS * sizeof(cbf16_t));
    }
  }

  void*  d_grid      = nullptr;
  int*   d_indices   = nullptr;
  void* d_llrs       = nullptr;
  float2* d_eq       = nullptr;
  float* d_eq_noise  = nullptr;
  cudaMalloc(&d_grid, h_grid.size() * sizeof(uint32_t));
  cudaMalloc(&d_indices, re_indices.size() * sizeof(int));
  cudaMalloc(&d_llrs, re_indices.size() * nof_layers * get_bits_per_symbol(mcs_descr.modulation) * sizeof(__half));
  cudaMalloc(&d_eq, cpu_eq.size() * sizeof(float2));
  cudaMalloc(&d_eq_noise, cpu_eq.size() * sizeof(float));
  cudaMemcpy(d_grid, h_grid.data(), h_grid.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, re_indices.data(), re_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_eq, 0, cpu_eq.size() * sizeof(float2));
  cudaMemset(d_eq_noise, 0, cpu_eq.size() * sizeof(float));

  pusch_e2e_handle_t handle = nullptr;
  report_fatal_error_if_not(pusch_e2e_create(&handle) == NR_LDPC_SUCCESS, "Failed to create PUSCH E2E handle.");
  pusch_e2e_config_t cfg          = {};
  cfg.nof_prb                    = nof_prb;
  cfg.nof_symbols                = nof_ofdm_symbols;
  cfg.nof_rx_ports               = nof_rx_ports;
  cfg.nof_tx_layers              = nof_layers;
  cfg.grid_nof_subcarriers       = MAX_NOF_SUBCARRIERS;
  cfg.grid_nof_symbols           = MAX_NSYMB_PER_SLOT;
  cfg.dmrs_type                  = DMRS_TYPE_1;
  cfg.dmrs_symbol_mask           = static_cast<int>(dmrs_symbols_mask.to_uint64());
  cfg.nof_cdm_groups_without_data = nof_cdm_groups_without_data;
  cfg.scrambling_id              = scrambling_id;
  cfg.n_scid                     = n_scid ? 1 : 0;
  cfg.slot_idx                   = 1;
  cfg.dmrs_scaling               = est_config.scaling;
  cfg.mod_order                  = get_bits_per_symbol(mcs_descr.modulation);
  cfg.rnti                       = rnti;
  cfg.n_id                       = n_id;
  cfg.start_prb                  = bwp_start_rb;
  cfg.start_symbol               = 0;
  cfg.tx_scaling                 = 1.0F;
  cfg.equalizer_algorithm        = (equalizer_algorithm == channel_equalizer_algorithm_type::mmse) ? EQUALIZER_MMSE :
                                                                                                    EQUALIZER_ZF;
  cfg.scs_khz                    = 30;
  cfg.compensate_cfo             = 0;
  cfg.noise_mode                 = 1;
  cfg.time_interp_mode           = 0;
  report_fatal_error_if_not(pusch_e2e_configure(handle, &cfg) == NR_LDPC_SUCCESS, "Failed to configure PUSCH E2E.");
  pusch_e2e_set_eq_output(handle, d_eq, d_eq_noise);
  nr_ldpc_status_t status = pusch_e2e_process_full_gpu_optimized_mimo_half(
      handle, d_grid, d_llrs, d_indices, static_cast<int>(re_indices.size()), 0);
  cudaDeviceSynchronize();
  report_fatal_error_if_not(status == NR_LDPC_SUCCESS, "PUSCH E2E processing failed.");

  unsigned            nof_dmrs_symbols = count_dmrs_symbols();
  size_t              gpu_est_count     = nof_dmrs_symbols * nof_prb * nof_rx_ports * nof_layers * 12;
  std::vector<__half2> gpu_estimates_fp16(gpu_est_count);
  const void*          d_gpu_estimates = pusch_e2e_get_estimates_fp16(handle);
  report_fatal_error_if_not(d_gpu_estimates != nullptr, "PUSCH E2E FP16 estimates buffer is null.");
  cudaMemcpy(gpu_estimates_fp16.data(), d_gpu_estimates, gpu_estimates_fp16.size() * sizeof(__half2), cudaMemcpyDeviceToHost);

  double ce_sum_sq = 0.0;
  double ce_max_abs = 0.0;
  size_t ce_max_idx = 0;
  cf_t   ce_gpu_max = {};
  std::array<double, NOF_SUBCARRIERS_PER_RB> ce_sc_sum_sq = {};
  std::array<unsigned, NOF_SUBCARRIERS_PER_RB> ce_sc_count = {};
  std::vector<double> ce_layer_sum_sq(nof_layers, 0.0);
  std::vector<unsigned> ce_layer_count(nof_layers, 0);
  std::vector<cf_t> gpu_ce(cpu_ce.size());
  unsigned re_per_layer = 12;
  unsigned re_per_port = nof_layers * re_per_layer;
  unsigned re_per_prb = nof_rx_ports * re_per_port;
  unsigned re_per_dmrs_symbol = nof_prb * re_per_prb;
  for (size_t i_re = 0; i_re != re_indices.size(); ++i_re) {
    unsigned src_re = static_cast<unsigned>(re_indices[i_re]);
    unsigned subc = src_re % MAX_NOF_SUBCARRIERS;
    unsigned prb_idx = (subc - bwp_start_rb * NOF_SUBCARRIERS_PER_RB) / NOF_SUBCARRIERS_PER_RB;
    unsigned sc_in_prb = (subc - bwp_start_rb * NOF_SUBCARRIERS_PER_RB) % NOF_SUBCARRIERS_PER_RB;

    for (unsigned i_port = 0; i_port != nof_rx_ports; ++i_port) {
      for (unsigned i_layer = 0; i_layer != nof_layers; ++i_layer) {
        float gpu_re = 0.0F;
        float gpu_im = 0.0F;
        for (unsigned i_dmrs = 0; i_dmrs != nof_dmrs_symbols; ++i_dmrs) {
          size_t est_idx = i_dmrs * re_per_dmrs_symbol + prb_idx * re_per_prb + i_port * re_per_port +
                           i_layer * re_per_layer + sc_in_prb;
          __half2 h = gpu_estimates_fp16[est_idx];
          gpu_re += __half2float(h.x);
          gpu_im += __half2float(h.y);
        }
        gpu_re /= static_cast<float>(nof_dmrs_symbols);
        gpu_im /= static_cast<float>(nof_dmrs_symbols);

        size_t ce_idx = (i_re * nof_rx_ports + i_port) * nof_layers + i_layer;
        double dr = static_cast<double>(gpu_re) - static_cast<double>(cpu_ce[ce_idx].real());
        double di = static_cast<double>(gpu_im) - static_cast<double>(cpu_ce[ce_idx].imag());
        double err = std::sqrt(dr * dr + di * di);
        gpu_ce[ce_idx] = {gpu_re, gpu_im};
        ce_sum_sq += err * err;
        ce_sc_sum_sq[sc_in_prb] += err * err;
        ++ce_sc_count[sc_in_prb];
        ce_layer_sum_sq[i_layer] += err * err;
        ++ce_layer_count[i_layer];
        if (err > ce_max_abs) {
          ce_max_abs = err;
          ce_max_idx = ce_idx;
          ce_gpu_max = {gpu_re, gpu_im};
        }
      }
    }
  }
  double ce_rms = std::sqrt(ce_sum_sq / static_cast<double>(cpu_ce.size()));
  std::vector<cf_t> cpu_ce_fp16(cpu_ce.size());
  for (size_t i = 0; i != cpu_ce.size(); ++i) {
    cpu_ce_fp16[i] = {__half2float(__float2half(cpu_ce[i].real())),
                      __half2float(__float2half(cpu_ce[i].imag()))};
  }

  std::vector<float2> gpu_eq(cpu_eq.size());
  cudaMemcpy(gpu_eq.data(), d_eq, gpu_eq.size() * sizeof(float2), cudaMemcpyDeviceToHost);
  std::vector<float> gpu_eq_noise(cpu_eq_noise.size());
  cudaMemcpy(gpu_eq_noise.data(), d_eq_noise, gpu_eq_noise.size() * sizeof(float), cudaMemcpyDeviceToHost);
  std::vector<float> gpu_noise_vars(nof_rx_ports);
  cudaMemcpy(gpu_noise_vars.data(), pusch_e2e_get_noise_vars(handle), gpu_noise_vars.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  fmt::print("GPU pre-EQ noise:");
  for (unsigned i_port = 0; i_port != nof_rx_ports; ++i_port) {
    fmt::print(" p{}={:.6e}", i_port, gpu_noise_vars[i_port]);
  }
  fmt::print("\n");

  std::vector<float> hybrid_eq_noise;
  std::vector<cf_t>  hybrid_eq = compute_cpu_eq_from_flat_ce(
      rx_grid->get_reader(), rb_mask, span<const cf_t>(gpu_ce), span<const float>(gpu_noise_vars), *equalizer, &hybrid_eq_noise);
  std::vector<cf_t> scalar_cpu_ce_eq =
      compute_scalar_eq_from_flat_ce(rx_grid->get_reader(), span<const int>(re_indices), span<const cf_t>(cpu_ce),
                                     span<const float>(cpu_pre_eq_noise));
  std::vector<cf_t> scalar_cpu_ce_fp16_eq =
      compute_scalar_eq_from_flat_ce(rx_grid->get_reader(), span<const int>(re_indices), span<const cf_t>(cpu_ce_fp16),
                                     span<const float>(cpu_pre_eq_noise));
  std::vector<cf_t> scalar_gpu_ce_eq =
      compute_scalar_eq_from_flat_ce(rx_grid->get_reader(), span<const int>(re_indices), span<const cf_t>(gpu_ce),
                                     span<const float>(gpu_noise_vars));

  double hybrid_cpu_sum_sq = 0.0;
  double hybrid_gpu_sum_sq = 0.0;
  double scalar_cpu_sum_sq = 0.0;
  double scalar_cpu_fp16_sum_sq = 0.0;
  double scalar_gpu_sum_sq = 0.0;
  double scalar_gpu_vs_device_sum_sq = 0.0;
  double hybrid_cpu_max_abs = 0.0;
  double hybrid_gpu_max_abs = 0.0;
  double scalar_cpu_max_abs = 0.0;
  double scalar_cpu_fp16_max_abs = 0.0;
  double scalar_gpu_max_abs = 0.0;
  double scalar_gpu_vs_device_max_abs = 0.0;
  size_t hybrid_cpu_max_idx = 0;
  size_t hybrid_gpu_max_idx = 0;
  size_t scalar_cpu_max_idx = 0;
  size_t scalar_cpu_fp16_max_idx = 0;
  size_t scalar_gpu_max_idx = 0;
  size_t scalar_gpu_vs_device_max_idx = 0;
  for (size_t i = 0; i != cpu_eq.size(); ++i) {
    double cpu_dr  = static_cast<double>(hybrid_eq[i].real()) - static_cast<double>(cpu_eq[i].real());
    double cpu_di  = static_cast<double>(hybrid_eq[i].imag()) - static_cast<double>(cpu_eq[i].imag());
    double cpu_err = std::sqrt(cpu_dr * cpu_dr + cpu_di * cpu_di);
    hybrid_cpu_sum_sq += cpu_err * cpu_err;
    if (cpu_err > hybrid_cpu_max_abs) {
      hybrid_cpu_max_abs = cpu_err;
      hybrid_cpu_max_idx = i;
    }

    double gpu_dr  = static_cast<double>(hybrid_eq[i].real()) - static_cast<double>(gpu_eq[i].x);
    double gpu_di  = static_cast<double>(hybrid_eq[i].imag()) - static_cast<double>(gpu_eq[i].y);
    double gpu_err = std::sqrt(gpu_dr * gpu_dr + gpu_di * gpu_di);
    hybrid_gpu_sum_sq += gpu_err * gpu_err;
    if (gpu_err > hybrid_gpu_max_abs) {
      hybrid_gpu_max_abs = gpu_err;
      hybrid_gpu_max_idx = i;
    }

    double scalar_cpu_dr = static_cast<double>(scalar_cpu_ce_eq[i].real()) - static_cast<double>(cpu_eq[i].real());
    double scalar_cpu_di = static_cast<double>(scalar_cpu_ce_eq[i].imag()) - static_cast<double>(cpu_eq[i].imag());
    double scalar_cpu_err = std::sqrt(scalar_cpu_dr * scalar_cpu_dr + scalar_cpu_di * scalar_cpu_di);
    scalar_cpu_sum_sq += scalar_cpu_err * scalar_cpu_err;
    if (scalar_cpu_err > scalar_cpu_max_abs) {
      scalar_cpu_max_abs = scalar_cpu_err;
      scalar_cpu_max_idx = i;
    }

    double scalar_cpu_fp16_dr =
        static_cast<double>(scalar_cpu_ce_fp16_eq[i].real()) - static_cast<double>(cpu_eq[i].real());
    double scalar_cpu_fp16_di =
        static_cast<double>(scalar_cpu_ce_fp16_eq[i].imag()) - static_cast<double>(cpu_eq[i].imag());
    double scalar_cpu_fp16_err =
        std::sqrt(scalar_cpu_fp16_dr * scalar_cpu_fp16_dr + scalar_cpu_fp16_di * scalar_cpu_fp16_di);
    scalar_cpu_fp16_sum_sq += scalar_cpu_fp16_err * scalar_cpu_fp16_err;
    if (scalar_cpu_fp16_err > scalar_cpu_fp16_max_abs) {
      scalar_cpu_fp16_max_abs = scalar_cpu_fp16_err;
      scalar_cpu_fp16_max_idx = i;
    }

    double scalar_gpu_dr = static_cast<double>(scalar_gpu_ce_eq[i].real()) - static_cast<double>(cpu_eq[i].real());
    double scalar_gpu_di = static_cast<double>(scalar_gpu_ce_eq[i].imag()) - static_cast<double>(cpu_eq[i].imag());
    double scalar_gpu_err = std::sqrt(scalar_gpu_dr * scalar_gpu_dr + scalar_gpu_di * scalar_gpu_di);
    scalar_gpu_sum_sq += scalar_gpu_err * scalar_gpu_err;
    if (scalar_gpu_err > scalar_gpu_max_abs) {
      scalar_gpu_max_abs = scalar_gpu_err;
      scalar_gpu_max_idx = i;
    }

    double scalar_device_dr = static_cast<double>(scalar_gpu_ce_eq[i].real()) - static_cast<double>(gpu_eq[i].x);
    double scalar_device_di = static_cast<double>(scalar_gpu_ce_eq[i].imag()) - static_cast<double>(gpu_eq[i].y);
    double scalar_device_err = std::sqrt(scalar_device_dr * scalar_device_dr + scalar_device_di * scalar_device_di);
    scalar_gpu_vs_device_sum_sq += scalar_device_err * scalar_device_err;
    if (scalar_device_err > scalar_gpu_vs_device_max_abs) {
      scalar_gpu_vs_device_max_abs = scalar_device_err;
      scalar_gpu_vs_device_max_idx = i;
    }
  }
  double hybrid_cpu_rms = std::sqrt(hybrid_cpu_sum_sq / static_cast<double>(cpu_eq.size()));
  double hybrid_gpu_rms = std::sqrt(hybrid_gpu_sum_sq / static_cast<double>(cpu_eq.size()));
  double scalar_cpu_rms = std::sqrt(scalar_cpu_sum_sq / static_cast<double>(cpu_eq.size()));
  double scalar_cpu_fp16_rms = std::sqrt(scalar_cpu_fp16_sum_sq / static_cast<double>(cpu_eq.size()));
  double scalar_gpu_rms = std::sqrt(scalar_gpu_sum_sq / static_cast<double>(cpu_eq.size()));
  double scalar_gpu_vs_device_rms = std::sqrt(scalar_gpu_vs_device_sum_sq / static_cast<double>(cpu_eq.size()));

  double sum_sq = 0.0;
  double center_sum_sq = 0.0;
  size_t center_count = 0;
  double max_abs = 0.0;
  size_t max_idx = 0;
  std::vector<double> layer_sum_sq(nof_layers, 0.0);
  std::vector<double> layer_max_abs(nof_layers, 0.0);
  for (size_t i = 0; i != cpu_eq.size(); ++i) {
    double dr  = static_cast<double>(gpu_eq[i].x) - static_cast<double>(cpu_eq[i].real());
    double di  = static_cast<double>(gpu_eq[i].y) - static_cast<double>(cpu_eq[i].imag());
    double err = std::sqrt(dr * dr + di * di);
    unsigned layer = i % nof_layers;
    size_t i_re = i / nof_layers;
    unsigned src_re = static_cast<unsigned>(re_indices[i_re]);
    unsigned subc = src_re % MAX_NOF_SUBCARRIERS;
    unsigned prb_idx = (subc - bwp_start_rb * NOF_SUBCARRIERS_PER_RB) / NOF_SUBCARRIERS_PER_RB;
    bool is_center_prb = (prb_idx >= 2) && (prb_idx + 2 < nof_prb);
    layer_sum_sq[layer] += err * err;
    layer_max_abs[layer] = std::max(layer_max_abs[layer], err);
    sum_sq += err * err;
    if (is_center_prb) {
      center_sum_sq += err * err;
      ++center_count;
    }
    if (err > max_abs) {
      max_abs = err;
      max_idx = i;
    }
  }
  double rms = std::sqrt(sum_sq / static_cast<double>(cpu_eq.size()));
  double center_rms = (center_count != 0) ? std::sqrt(center_sum_sq / static_cast<double>(center_count)) : 0.0;

  double noise_sum_sq = 0.0;
  double noise_max_abs = 0.0;
  double cpu_noise_sum = 0.0;
  double gpu_noise_sum = 0.0;
  size_t noise_max_idx = 0;
  for (size_t i = 0; i != cpu_eq_noise.size(); ++i) {
    double err = static_cast<double>(gpu_eq_noise[i]) - static_cast<double>(cpu_eq_noise[i]);
    noise_sum_sq += err * err;
    cpu_noise_sum += cpu_eq_noise[i];
    gpu_noise_sum += gpu_eq_noise[i];
    if (std::abs(err) > noise_max_abs) {
      noise_max_abs = std::abs(err);
      noise_max_idx = i;
    }
  }
  double noise_rms = std::sqrt(noise_sum_sq / static_cast<double>(cpu_eq_noise.size()));
  double cpu_noise_mean = cpu_noise_sum / static_cast<double>(cpu_eq_noise.size());
  double gpu_noise_mean = gpu_noise_sum / static_cast<double>(gpu_eq_noise.size());
  fmt::print("PUSCH MIMO EQ parity: layers={} ports={} equalizer={} PRB={} RE={} symbols={}\n",
             nof_layers,
             nof_rx_ports,
             to_string(equalizer_algorithm),
             nof_prb,
             re_indices.size(),
             cpu_eq.size());
  fmt::print("CE RMS error: {:.6f}, max error: {:.6f} at ce_idx={} (re={}, port={}, layer={})\n",
             ce_rms,
             ce_max_abs,
             ce_max_idx,
             ce_max_idx / (nof_rx_ports * nof_layers),
             (ce_max_idx / nof_layers) % nof_rx_ports,
             ce_max_idx % nof_layers);
  fmt::print("CPU_CE[{0}]=({1:.6f},{2:.6f}) GPU_CE[{0}]=({3:.6f},{4:.6f})\n",
             ce_max_idx,
             cpu_ce[ce_max_idx].real(),
             cpu_ce[ce_max_idx].imag(),
             ce_gpu_max.real(),
             ce_gpu_max.imag());
  fmt::print("CE RMS by layer:");
  for (unsigned layer = 0; layer != nof_layers; ++layer) {
    fmt::print(" l{}={:.6f}", layer, std::sqrt(ce_layer_sum_sq[layer] / static_cast<double>(ce_layer_count[layer])));
  }
  fmt::print("\nCE RMS by sc:");
  for (unsigned sc = 0; sc != NOF_SUBCARRIERS_PER_RB; ++sc) {
    fmt::print(" {}:{:.6f}", sc, std::sqrt(ce_sc_sum_sq[sc] / static_cast<double>(ce_sc_count[sc])));
  }
  fmt::print("\n");
  for (unsigned i_port = 0; i_port != nof_rx_ports; ++i_port) {
    for (unsigned i_layer = 0; i_layer != nof_layers; ++i_layer) {
      cf_t   cross = {};
      double gpu_power = 0.0;
      double pair_sum_sq = 0.0;
      for (size_t i_re = 0; i_re != re_indices.size(); ++i_re) {
        size_t idx = (i_re * nof_rx_ports + i_port) * nof_layers + i_layer;
        cross += cpu_ce[idx] * std::conj(gpu_ce[idx]);
        gpu_power += std::norm(gpu_ce[idx]);
      }
      cf_t alpha = (gpu_power > 0.0) ? cross / static_cast<float>(gpu_power) : cf_t{};
      double corrected_sum_sq = 0.0;
      for (size_t i_re = 0; i_re != re_indices.size(); ++i_re) {
        size_t idx = (i_re * nof_rx_ports + i_port) * nof_layers + i_layer;
        cf_t raw_err = gpu_ce[idx] - cpu_ce[idx];
        cf_t corrected_err = alpha * gpu_ce[idx] - cpu_ce[idx];
        pair_sum_sq += std::norm(raw_err);
        corrected_sum_sq += std::norm(corrected_err);
      }
      fmt::print("CE pair p{} l{}: rms={:.6f}, alpha=({:.6f},{:.6f}), alpha_rms={:.6f}\n",
                 i_port,
                 i_layer,
                 std::sqrt(pair_sum_sq / static_cast<double>(re_indices.size())),
                 alpha.real(),
                 alpha.imag(),
                 std::sqrt(corrected_sum_sq / static_cast<double>(re_indices.size())));
    }
  }
  fmt::print("RMS error: {:.6f}, max error: {:.6f} at eq_idx={} (re={}, layer={})\n",
             rms,
             max_abs,
             max_idx,
             max_idx / nof_layers,
             max_idx % nof_layers);
  fmt::print("Center-PRB EQ RMS error excluding first/last two PRBs: {:.6f} over {} symbols\n", center_rms, center_count);
  fmt::print("CPU equalizer with GPU CE/noise: vs CPU RMS={:.6f}, max={:.6f} at eq_idx={} (re={}, layer={}); "
             "vs GPU RMS={:.6f}, max={:.6f} at eq_idx={} (re={}, layer={})\n",
             hybrid_cpu_rms,
             hybrid_cpu_max_abs,
             hybrid_cpu_max_idx,
             hybrid_cpu_max_idx / nof_layers,
             hybrid_cpu_max_idx % nof_layers,
             hybrid_gpu_rms,
             hybrid_gpu_max_abs,
             hybrid_gpu_max_idx,
             hybrid_gpu_max_idx / nof_layers,
             hybrid_gpu_max_idx % nof_layers);
  fmt::print("Scalar float EQ: CPU_CE vs CPU RMS={:.6f}, max={:.6f} at eq_idx={} (re={}, layer={}); "
             "CPU_CE rounded FP16 vs CPU RMS={:.6f}, max={:.6f} at eq_idx={} (re={}, layer={}); "
             "GPU_CE vs CPU RMS={:.6f}, max={:.6f} at eq_idx={} (re={}, layer={}); "
             "GPU_CE vs device GPU RMS={:.6f}, max={:.6f} at eq_idx={} (re={}, layer={})\n",
             scalar_cpu_rms,
             scalar_cpu_max_abs,
             scalar_cpu_max_idx,
             scalar_cpu_max_idx / nof_layers,
             scalar_cpu_max_idx % nof_layers,
             scalar_cpu_fp16_rms,
             scalar_cpu_fp16_max_abs,
             scalar_cpu_fp16_max_idx,
             scalar_cpu_fp16_max_idx / nof_layers,
             scalar_cpu_fp16_max_idx % nof_layers,
             scalar_gpu_rms,
             scalar_gpu_max_abs,
             scalar_gpu_max_idx,
             scalar_gpu_max_idx / nof_layers,
             scalar_gpu_max_idx % nof_layers,
             scalar_gpu_vs_device_rms,
             scalar_gpu_vs_device_max_abs,
             scalar_gpu_vs_device_max_idx,
             scalar_gpu_vs_device_max_idx / nof_layers,
             scalar_gpu_vs_device_max_idx % nof_layers);
  fmt::print("EQ RMS by layer:");
  for (unsigned layer = 0; layer != nof_layers; ++layer) {
    double layer_count = static_cast<double>(cpu_eq.size() / nof_layers);
    fmt::print(" l{}={:.6f}(max={:.6f})", layer, std::sqrt(layer_sum_sq[layer] / layer_count), layer_max_abs[layer]);
  }
  fmt::print("\n");
  fmt::print("CPU[{0}]=({1:.6f},{2:.6f}) GPU[{0}]=({3:.6f},{4:.6f})\n",
             max_idx,
             cpu_eq[max_idx].real(),
             cpu_eq[max_idx].imag(),
             gpu_eq[max_idx].x,
             gpu_eq[max_idx].y);
  fmt::print("EQ noise: CPU mean={:.6e}, GPU mean={:.6e}, ratio={:.3f}, rms_err={:.6e}, max_err={:.6e} at idx={} (re={}, layer={})\n",
             cpu_noise_mean,
             gpu_noise_mean,
             cpu_noise_mean > 0.0 ? gpu_noise_mean / cpu_noise_mean : 0.0,
             noise_rms,
             noise_max_abs,
             noise_max_idx,
             noise_max_idx / nof_layers,
             noise_max_idx % nof_layers);

  pusch_e2e_destroy(handle);
  cudaFree(d_grid);
  cudaFree(d_indices);
  cudaFree(d_llrs);
  cudaFree(d_eq);
  cudaFree(d_eq_noise);
  worker_pool.stop();

  return (ce_rms < 0.15 && ce_max_abs < 1.0 && rms < 0.15 && max_abs < 1.0) ? 0 : 1;
#endif
}
