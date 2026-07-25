// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief 64QAM Stress Test: Multiple iterations to catch intermittent failures
///
/// This test runs many iterations of 64QAM processing through both CPU and GPU
/// to detect any intermittent decode failures or metric discrepancies.

#include "pxsch_bler_test_channel_emulator.h"
#include "pxsch_bler_test_factories.h"
#include "ocudu/phy/support/resource_grid.h"
#include "ocudu/phy/support/support_factories.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_decoder_result.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_processor_result_notifier.h"
#include "ocudu/phy/upper/rx_buffer_pool.h"
#include "ocudu/phy/upper/unique_rx_buffer.h"
#include "ocudu/phy/upper/trx_buffer_identifier.h"
#include "ocudu/ran/precoding/precoding_codebooks.h"
#include "ocudu/ran/pusch/pusch_mcs.h"
#include "ocudu/ran/resource_allocation/rb_interval.h"
#include "ocudu/ran/sch/sch_dmrs_power.h"
#include "ocudu/ran/sch/sch_mcs.h"
#include "ocudu/ran/sch/sch_segmentation.h"
#include "ocudu/ran/sch/tbs_calculator.h"
#include "ocudu/support/executors/task_worker_pool.h"
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>

using namespace ocudu;

// Test parameters
static constexpr subcarrier_spacing scs                         = subcarrier_spacing::kHz30;
static uint16_t                     rnti                        = 0x1234;
static constexpr unsigned           bwp_start_rb                = 0;
static constexpr unsigned           nof_ofdm_symbols            = 14;
static symbol_slot_mask             dmrs_symbols_mask            = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
static constexpr unsigned           nof_ldpc_iterations         = 6;
static constexpr dmrs_config_type          dmrs                        = dmrs_config_type::type1;
static constexpr unsigned           nof_cdm_groups_without_data = 2;
static constexpr cyclic_prefix      cy_prefix                   = cyclic_prefix::NORMAL;
static constexpr unsigned           rv                          = 0;
static unsigned                     n_id                        = 0;
static unsigned                     scrambling_id               = 0;
static bool                         n_scid                      = false;
static constexpr bool               use_early_stop              = true;
static constexpr unsigned           nof_layers                  = 1;
static pusch_mcs_table              mcs_table                   = pusch_mcs_table::qam64;

// Test sweep parameters
static unsigned    nof_iterations   = 100;
static float       max_ta_offset_us = 0.0f;
static float       cfo_std_hz       = 0.0f;
static std::string channel_profile  = "single-tap";
static std::string fading_dist      = "uniform-phase";
static bool        csi_sweep_mode   = false;
static unsigned    csi_sweep_iters  = 50;
static float       rx_gain_range_dB = 0.0f;

namespace {

class pusch_processor_notifier_adaptor : public pusch_processor_result_notifier
{
public:
  void on_uci(const pusch_processor_result_control& uci_) override {}

  void on_sch(const pusch_processor_result_data& sch_) override
  {
    std::unique_lock<std::mutex> lock(mutex);
    completed = true;
    sch       = sch_;
    cvar.notify_all();
  }

  const pusch_processor_result_data& wait_for_completion()
  {
    std::unique_lock<std::mutex> lock(mutex);
    while (!completed) {
      cvar.wait(lock);
    }
    return sch;
  }

  void reset()
  {
    std::unique_lock<std::mutex> lock(mutex);
    completed = false;
  }

  bool                           completed = false;
  pusch_processor_result_data    sch;
  std::mutex                     mutex;
  std::condition_variable        cvar;
};

class pdsch_processor_notifier_adaptor : public pdsch_processor_notifier
{
public:
  void on_finish_processing() override { completed = true; }

  void wait_for_completion()
  {
    while (!completed.load()) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  void reset() { completed = false; }

private:
  std::atomic<bool> completed = {false};
};

struct latency_stats {
  double min_us = 1e9;
  double max_us = 0;
  double sum_us = 0;
  unsigned count = 0;

  void add(double us) {
    min_us = std::min(min_us, us);
    max_us = std::max(max_us, us);
    sum_us += us;
    count++;
  }

  double mean() const { return count > 0 ? sum_us / count : 0; }
};

/// Accumulates optional float CSI metrics across iterations.
struct csi_metric_stats {
  double   sum       = 0.0;
  double   sum_sq    = 0.0;
  double   min_val   = 1e30;
  double   max_val   = -1e30;
  unsigned count     = 0;
  unsigned populated = 0; ///< Number of iterations where value was present.

  void add(std::optional<float> val)
  {
    ++count;
    if (val.has_value()) {
      double v = static_cast<double>(val.value());
      sum += v;
      sum_sq += v * v;
      min_val = std::min(min_val, v);
      max_val = std::max(max_val, v);
      ++populated;
    }
  }

  bool   has_data() const { return populated > 0; }
  double mean() const { return populated > 0 ? sum / populated : 0.0; }
  double stddev() const
  {
    if (populated < 2) return 0.0;
    double m = mean();
    double var = sum_sq / populated - m * m;
    return var > 0.0 ? std::sqrt(var) : 0.0;
  }
  double population_rate() const { return count > 0 ? 100.0 * populated / count : 0.0; }
};

/// Tracks GPU−CPU delta for a CSI metric across iterations.
struct csi_delta_stats {
  double   sum         = 0.0;
  double   sum_sq      = 0.0;
  double   min_val     = 1e30;
  double   max_val     = -1e30;
  double   max_abs     = 0.0;
  unsigned count       = 0;
  unsigned total       = 0;
  unsigned cpu_present = 0;
  unsigned gpu_present = 0;
  unsigned missing_gpu = 0;
  unsigned missing_cpu = 0;

  void add(std::optional<float> cpu_val, std::optional<float> gpu_val)
  {
    ++total;
    if (cpu_val.has_value()) {
      ++cpu_present;
    }
    if (gpu_val.has_value()) {
      ++gpu_present;
    }
    if (cpu_val.has_value() && gpu_val.has_value()) {
      double delta = static_cast<double>(gpu_val.value()) - static_cast<double>(cpu_val.value());
      sum += delta;
      sum_sq += delta * delta;
      min_val = std::min(min_val, delta);
      max_val = std::max(max_val, delta);
      max_abs = std::max(max_abs, std::abs(delta));
      ++count;
    } else if (cpu_val.has_value()) {
      ++missing_gpu;
    } else if (gpu_val.has_value()) {
      ++missing_cpu;
    }
  }

  bool   has_data() const { return count > 0; }
  double mean() const { return count > 0 ? sum / count : 0.0; }
  double stddev() const
  {
    if (count < 2) return 0.0;
    double m = mean();
    double var = sum_sq / count - m * m;
    return var > 0.0 ? std::sqrt(var) : 0.0;
  }
};

/// Tolerance thresholds for CPU/GPU CSI metric parity.
static constexpr double kTolSinrMean = 1.0;  // dB
static constexpr double kTolSinrMax  = 5.0;  // dB
static constexpr double kTolEpreMean = 0.25; // dB
static constexpr double kTolEpreMax  = 0.5;  // dB
static constexpr double kTolRsrpMean = 0.5;  // dB
static constexpr double kTolRsrpMax  = 1.0;  // dB
static constexpr double kTolTaMean   = 0.05; // us
static constexpr double kTolTaMax    = 0.10; // us
static constexpr double kTolCfoMean  = 1.0;  // Hz
static constexpr double kTolCfoMax   = 5.0;  // Hz
static constexpr double kTolEvmMean  = 10.0; // percentage points
static constexpr double kTolEvmMax   = 50.0; // percentage points, reported but not hard-gated in broad sweeps

struct test_results {
  unsigned cpu_pass = 0;
  unsigned cpu_fail = 0;
  unsigned gpu_pass = 0;
  unsigned gpu_fail = 0;
  unsigned disagreements = 0;
  latency_stats cpu_latency;
  latency_stats gpu_latency;
  double cpu_iters_sum = 0;
  double gpu_iters_sum = 0;
  unsigned iters_count = 0;

  // CSI metric parity tracking.
  csi_metric_stats cpu_sinr, gpu_sinr;
  csi_metric_stats cpu_evm, gpu_evm;
  csi_metric_stats cpu_epre, gpu_epre;
  csi_metric_stats cpu_ta, gpu_ta;
  csi_metric_stats cpu_rsrp, gpu_rsrp;
  csi_metric_stats cpu_cfo, gpu_cfo;
  csi_metric_stats cpu_ta_error, gpu_ta_error;
  csi_metric_stats cpu_cfo_error, gpu_cfo_error;

  // Per-iteration GPU−CPU delta tracking.
  csi_delta_stats delta_sinr, delta_evm, delta_epre, delta_ta, delta_rsrp, delta_cfo;

#ifdef ENABLE_CUDA
  // GPU decoder breakdown timing
  latency_stats gpu_deinterleave;
  latency_stats gpu_rate_dematch;
  latency_stats gpu_ldpc_decode;
  latency_stats gpu_crc_check;
  latency_stats gpu_d2h_transfer;
  // Gap profiling: host-side overhead
  latency_stats gpu_extract_bits;
  latency_stats gpu_grid_staging;
  latency_stats gpu_demod_sync;
  // Detailed gap profiling
  latency_stats gpu_decode_call;
  latency_stats gpu_sinr_readback;
#endif
};

void run_stress_test(unsigned nof_prb, float sinr_dB, sch_mcs_index mcs_idx)
{
  fmt::print("\n=== PUSCH Stress Test ===\n");
  fmt::print("Config: {} PRB, {:.1f} dB SINR, MCS {} (table={}), {} iterations, channel={}, fading={}{}{}\n\n",
             nof_prb, sinr_dB, mcs_idx, pusch_mcs_table_to_string(mcs_table), nof_iterations,
             channel_profile, fading_dist,
             max_ta_offset_us > 0.0f ? fmt::format(", TA offset: +/-{:.1f}us", max_ta_offset_us) : "",
             cfo_std_hz > 0.0f ? fmt::format(", CFO std: {:.1f}Hz", cfo_std_hz) : "",
             rx_gain_range_dB > 0.0f ? fmt::format(", RX gain: +/-{:.1f}dB", rx_gain_range_dB) : "");

  test_results results;

  // Setup
  unsigned max_nof_threads = std::min(8U, std::thread::hardware_concurrency());
  auto worker_pool =
      std::make_unique<task_worker_pool<concurrent_queue_policy::locking_mpmc>>("thread", max_nof_threads, 1024);
  auto executor = std::make_unique<task_worker_pool_executor<concurrent_queue_policy::locking_mpmc>>(*worker_pool);

  sch_mcs_description mcs_descr = pusch_mcs_get_config(mcs_table, mcs_idx, false, false);
  prb_interval freq_allocation = {bwp_start_rb, bwp_start_rb + nof_prb};

  tbs_calculator_configuration tbs_config = {};
  tbs_config.mcs_descr                    = mcs_descr;
  tbs_config.n_prb                        = freq_allocation.length();
  tbs_config.nof_layers                   = nof_layers;
  tbs_config.nof_symb_sh                  = nof_ofdm_symbols;
  tbs_config.nof_dmrs_prb = get_nof_re_per_prb(dmrs) * dmrs_symbols_mask.count() * nof_cdm_groups_without_data;
  unsigned tbs            = tbs_calculator_calculate(tbs_config).to_bits().value();

  ldpc_base_graph_type ldpc_base_graph =
      get_ldpc_base_graph(mcs_descr.get_normalised_target_code_rate(), units::bits(tbs));

  rb_allocation freq_alloc = rb_allocation::make_type1(freq_allocation.start(), freq_allocation.length(), std::nullopt);

  // Create factories
  std::shared_ptr<channel_precoder_factory> precod_factory = create_channel_precoder_factory("auto");
  std::shared_ptr<resource_grid_factory> grid_factory = create_resource_grid_factory();

  std::shared_ptr<pdsch_processor_factory> pdsch_factory =
      create_sw_pdsch_processor_factory(*executor, max_nof_threads + 1, "", "auto");

  std::shared_ptr<pusch_processor_factory> cpu_factory =
      create_sw_pusch_processor_factory(*executor, max_nof_threads + 1,
                                        nof_ldpc_iterations, use_early_stop, "auto",
                                        port_channel_estimator_td_interpolation_strategy::average,
                                        channel_equalizer_algorithm_type::zf);

  std::shared_ptr<pusch_processor_factory> gpu_factory =
      create_sw_pusch_processor_factory(*executor, max_nof_threads + 1,
                                        nof_ldpc_iterations, use_early_stop, "gpu",
                                        port_channel_estimator_td_interpolation_strategy::average,
                                        channel_equalizer_algorithm_type::zf);

  if (!gpu_factory) {
    fmt::print("GPU PUSCH not available, skipping test\n");
    worker_pool->stop();
    return;
  }

  // Create processors
  auto pdsch_proc = pdsch_factory->create();
  auto cpu_proc = cpu_factory->create();
  auto gpu_proc = gpu_factory->create();

  // Create grids
  auto tx_grid = grid_factory->create(nof_layers, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);
  auto rx_grid = grid_factory->create(1, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);

  unsigned nof_codeblocks = compute_nof_codeblocks(units::bits(tbs), ldpc_base_graph);

  rx_buffer_pool_config pool_config;
  pool_config.max_codeblock_size = ldpc::MAX_CODEBLOCK_SIZE;
  pool_config.nof_buffers        = 2;
  pool_config.nof_codeblocks     = nof_codeblocks;
  pool_config.expire_timeout_slots = 10;
  pool_config.external_soft_bits = false;

  auto cpu_pool = create_rx_buffer_pool(pool_config);
  auto gpu_pool = create_rx_buffer_pool(pool_config);

  // Channel emulator
  channel_emulator emulator(channel_profile, fading_dist, sinr_dB, 0.0f, 0,
                           nof_layers, 1, nof_prb * NOF_SUBCARRIERS_PER_RB,
                           nof_ofdm_symbols, max_nof_threads, scs, *executor);

  std::random_device rd;
  std::mt19937 rgen(rd());

  // Buffer for applying TA phase slope and CFO (reused across iterations).
  std::vector<cf_t> ta_symbol_buffer(nof_prb * NOF_SUBCARRIERS_PER_RB);
  std::uniform_real_distribution<float> ta_dist(-max_ta_offset_us, max_ta_offset_us);
  std::normal_distribution<float> cfo_dist(0.0f, std::max(cfo_std_hz, 1e-9f));
  std::uniform_real_distribution<float> gain_dist(-rx_gain_range_dB, rx_gain_range_dB);
  float scs_hz = 15000.0f * static_cast<float>(1 << to_numerology_value(scs));

  // Precompute OFDM symbol start times for CFO injection (matches CPU epoch calculation).
  std::array<double, MAX_NSYMB_PER_SLOT> symbol_start_times_s{};
  {
    double symbol_duration_s = 1.0 / static_cast<double>(scs_hz);
    symbol_start_times_s[0]  = cy_prefix.get_length(0, scs).to_seconds();
    for (unsigned i = 1; i < MAX_NSYMB_PER_SLOT; ++i) {
      symbol_start_times_s[i] =
          symbol_start_times_s[i - 1] + cy_prefix.get_length(i, scs).to_seconds() + symbol_duration_s;
    }
  }

  // Run iterations
  for (unsigned iter = 0; iter < nof_iterations; ++iter) {
    // Generate random data
    std::vector<uint8_t> tx_data(tbs / 8);
    for (auto& byte : tx_data) {
      byte = static_cast<uint8_t>(rgen() & 0xff);
    }

    std::vector<uint8_t> cpu_rx_data(tbs / 8);
    std::vector<uint8_t> gpu_rx_data(tbs / 8);

    // PDSCH PDU
    pdsch_processor::pdu_t pdsch_pdu;
    pdsch_pdu.context                     = std::nullopt;
    pdsch_pdu.slot                        = slot_point(to_numerology_value(scs), iter);  // Use iter as slot index
    pdsch_pdu.rnti                        = rnti;
    pdsch_pdu.bwp_size_rb                 = nof_prb;
    pdsch_pdu.bwp_start_rb                = bwp_start_rb;
    pdsch_pdu.cp                          = cy_prefix;
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
    pdsch_pdu.reserved                    = {};
    pdsch_pdu.ratio_pdsch_data_to_sss_dB  = 0.0F;
    pdsch_pdu.ratio_pdsch_dmrs_to_sss_dB  = get_sch_to_dmrs_ratio_dB(nof_cdm_groups_without_data);
    pdsch_pdu.precoding                   = precoding_configuration::make_wideband(make_identity(nof_layers));
    pdsch_pdu.codewords.emplace_back(pdsch_processor::codeword_description{mcs_descr.modulation, rv});

    // Transmit
    pdsch_processor_notifier_adaptor tx_notifier;
    pdsch_proc->process(tx_grid->get_writer(), tx_notifier, {shared_transport_block(tx_data)}, pdsch_pdu);
    tx_notifier.wait_for_completion();

    // Apply channel
    emulator.run(rx_grid->get_writer(), tx_grid->get_reader());

    // Apply random timing offset as frequency-domain phase slope.
    float injected_ta_us = 0.0f;
    if (max_ta_offset_us > 0.0f) {
      injected_ta_us = ta_dist(rgen);
      float    ta_s          = injected_ta_us * 1e-6f;
      float    phase_per_sc  = -2.0f * static_cast<float>(M_PI) * scs_hz * ta_s;
      unsigned nof_subc      = nof_prb * NOF_SUBCARRIERS_PER_RB;
      for (unsigned sym = 0; sym < nof_ofdm_symbols; ++sym) {
        rx_grid->get_reader().get(ta_symbol_buffer, 0, sym, 0);
        for (unsigned k = 0; k < nof_subc; ++k) {
          float phase = phase_per_sc * static_cast<float>(k);
          ta_symbol_buffer[k] *= cf_t(std::cos(phase), std::sin(phase));
        }
        rx_grid->get_writer().put(0, sym, 0, ta_symbol_buffer);
      }
    }

    // Apply random CFO as per-symbol uniform phase rotation.
    float injected_cfo_hz = 0.0f;
    if (cfo_std_hz > 0.0f) {
      injected_cfo_hz = cfo_dist(rgen);
      unsigned nof_subc = nof_prb * NOF_SUBCARRIERS_PER_RB;
      for (unsigned sym = 0; sym < nof_ofdm_symbols; ++sym) {
        float phase = static_cast<float>(2.0 * M_PI * symbol_start_times_s[sym] * injected_cfo_hz);
        cf_t  coeff(std::cos(phase), std::sin(phase));
        rx_grid->get_reader().get(ta_symbol_buffer, 0, sym, 0);
        for (unsigned k = 0; k < nof_subc; ++k) {
          ta_symbol_buffer[k] *= coeff;
        }
        rx_grid->get_writer().put(0, sym, 0, ta_symbol_buffer);
      }
    }

    // Apply random RX gain scaling (simulates varying received power / path loss).
    float injected_gain_dB = 0.0f;
    if (rx_gain_range_dB > 0.0f) {
      injected_gain_dB = gain_dist(rgen);
      float    linear_gain = std::pow(10.0f, injected_gain_dB / 20.0f);
      unsigned nof_subc    = nof_prb * NOF_SUBCARRIERS_PER_RB;
      for (unsigned sym = 0; sym < nof_ofdm_symbols; ++sym) {
        rx_grid->get_reader().get(ta_symbol_buffer, 0, sym, 0);
        for (unsigned k = 0; k < nof_subc; ++k) {
          ta_symbol_buffer[k] *= linear_gain;
        }
        rx_grid->get_writer().put(0, sym, 0, ta_symbol_buffer);
      }
    }

    // PUSCH PDU
    static_vector<uint8_t, MAX_PORTS> rx_ports(1);
    std::iota(rx_ports.begin(), rx_ports.end(), 0U);

    pusch_processor::pdu_t pusch_pdu;
    pusch_pdu.context            = std::nullopt;
    pusch_pdu.slot               = slot_point(to_numerology_value(scs), iter);  // Use iter as slot index
    pusch_pdu.rnti               = rnti;
    pusch_pdu.bwp_size_rb        = nof_prb;
    pusch_pdu.bwp_start_rb       = bwp_start_rb;
    pusch_pdu.cp                 = cy_prefix;
    pusch_pdu.mcs_descr          = mcs_descr;
    pusch_pdu.codeword           = {rv, ldpc_base_graph, true};
    pusch_pdu.uci                = {};
    pusch_pdu.n_id               = n_id;
    pusch_pdu.nof_tx_layers      = nof_layers;
    pusch_pdu.rx_ports           = rx_ports;
    pusch_pdu.dmrs_symbol_mask   = dmrs_symbols_mask;
    pusch_pdu.dmrs               = pusch_processor::dmrs_configuration{
        .dmrs                        = dmrs,
        .scrambling_id               = scrambling_id,
        .n_scid                      = n_scid,
        .nof_cdm_groups_without_data = nof_cdm_groups_without_data};
    pusch_pdu.tbs_lbrm           = tbs_lbrm_default;
    pusch_pdu.freq_alloc         = freq_alloc;
    pusch_pdu.start_symbol_index = 0;
    pusch_pdu.nof_symbols        = nof_ofdm_symbols;
    pusch_pdu.dc_position        = std::nullopt;

    // CPU decode
    unique_rx_buffer cpu_buffer =
        cpu_pool->get_pool().reserve(pusch_pdu.slot, trx_buffer_identifier(rnti, 0), nof_codeblocks, true);

    pusch_processor_notifier_adaptor cpu_notifier;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_proc->process(cpu_rx_data, std::move(cpu_buffer), cpu_notifier, rx_grid->get_reader(), pusch_pdu);
    const auto& cpu_result = cpu_notifier.wait_for_completion();
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    results.cpu_latency.add(cpu_us);

    // GPU decode
    unique_rx_buffer gpu_buffer =
        gpu_pool->get_pool().reserve(pusch_pdu.slot, trx_buffer_identifier(rnti, 1), nof_codeblocks, true);

    pusch_processor_notifier_adaptor gpu_notifier;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu_proc->process(gpu_rx_data, std::move(gpu_buffer), gpu_notifier, rx_grid->get_reader(), pusch_pdu);
    const auto& gpu_result = gpu_notifier.wait_for_completion();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    results.gpu_latency.add(gpu_us);

#ifdef ENABLE_CUDA
    if (gpu_result.data.acceleration_pipeline_timing.valid) {
      results.gpu_deinterleave.add(gpu_result.data.acceleration_pipeline_timing.deinterleave_us);
      results.gpu_rate_dematch.add(gpu_result.data.acceleration_pipeline_timing.rate_dematch_us);
      results.gpu_ldpc_decode.add(gpu_result.data.acceleration_pipeline_timing.ldpc_decode_us);
      results.gpu_crc_check.add(gpu_result.data.acceleration_pipeline_timing.crc_check_us);
      results.gpu_d2h_transfer.add(gpu_result.data.acceleration_pipeline_timing.d2h_transfer_us);
      results.gpu_extract_bits.add(gpu_result.data.acceleration_pipeline_timing.extract_bits_us);
      results.gpu_grid_staging.add(gpu_result.data.acceleration_pipeline_timing.grid_staging_us);
      results.gpu_demod_sync.add(gpu_result.data.acceleration_pipeline_timing.demod_sync_us);
      results.gpu_decode_call.add(gpu_result.data.acceleration_pipeline_timing.decode_call_us);
      results.gpu_sinr_readback.add(gpu_result.data.acceleration_pipeline_timing.sinr_readback_us);
    }
#endif

    // Collect results
    if (cpu_result.data.tb_crc_ok) {
      results.cpu_pass++;
    } else {
      results.cpu_fail++;
    }

    if (gpu_result.data.tb_crc_ok) {
      results.gpu_pass++;
    } else {
      results.gpu_fail++;
    }

    // Track average LDPC iterations per decode.
    if (cpu_result.data.ldpc_decoder_stats.get_nof_observations() > 0) {
      results.cpu_iters_sum += cpu_result.data.ldpc_decoder_stats.get_mean();
    }
    if (gpu_result.data.ldpc_decoder_stats.get_nof_observations() > 0) {
      results.gpu_iters_sum += gpu_result.data.ldpc_decoder_stats.get_mean();
    }
    results.iters_count++;

    if (cpu_result.data.tb_crc_ok != gpu_result.data.tb_crc_ok) {
      results.disagreements++;
      fmt::print("DISAGREEMENT at iteration {}: CPU={} GPU={}\n",
                 iter, cpu_result.data.tb_crc_ok ? "PASS" : "FAIL",
                 gpu_result.data.tb_crc_ok ? "PASS" : "FAIL");
    }

    // CSI quality comparison: all fields between CPU and GPU paths.
    {
      const auto& cpu_csi = cpu_result.csi;
      const auto& gpu_csi = gpu_result.csi;

      // Extract all CSI fields.
      auto cpu_sinr_val = cpu_csi.get_sinr_dB();
      auto gpu_sinr_val = gpu_csi.get_sinr_dB();
      auto cpu_evm_val  = cpu_csi.get_total_evm();
      auto gpu_evm_val  = gpu_csi.get_total_evm();
      auto cpu_epre_val = cpu_csi.get_epre_dB();
      auto gpu_epre_val = gpu_csi.get_epre_dB();
      auto cpu_rsrp_val = cpu_csi.get_rsrp_dB();
      auto gpu_rsrp_val = gpu_csi.get_rsrp_dB();
      auto cpu_cfo_val  = cpu_csi.get_cfo_Hz();
      auto gpu_cfo_val  = gpu_csi.get_cfo_Hz();

      // Time alignment: convert to microseconds.
      auto                 cpu_ta_raw = cpu_csi.get_time_alignment();
      auto                 gpu_ta_raw = gpu_csi.get_time_alignment();
      std::optional<float> cpu_ta_us =
          cpu_ta_raw.has_value()
              ? std::optional<float>(static_cast<float>(cpu_ta_raw.value().to_seconds() * 1e6))
              : std::nullopt;
      std::optional<float> gpu_ta_us =
          gpu_ta_raw.has_value()
              ? std::optional<float>(static_cast<float>(gpu_ta_raw.value().to_seconds() * 1e6))
              : std::nullopt;

      // Accumulate statistics.
      results.cpu_sinr.add(cpu_sinr_val);
      results.gpu_sinr.add(gpu_sinr_val);
      std::optional<float> cpu_evm_pct =
          cpu_evm_val.has_value() ? std::optional<float>(cpu_evm_val.value() * 100.0f) : std::nullopt;
      std::optional<float> gpu_evm_pct =
          gpu_evm_val.has_value() ? std::optional<float>(gpu_evm_val.value() * 100.0f) : std::nullopt;
      results.cpu_evm.add(cpu_evm_pct);
      results.gpu_evm.add(gpu_evm_pct);
      results.cpu_epre.add(cpu_epre_val);
      results.gpu_epre.add(gpu_epre_val);
      results.cpu_ta.add(cpu_ta_us);
      results.gpu_ta.add(gpu_ta_us);
      results.cpu_rsrp.add(cpu_rsrp_val);
      results.gpu_rsrp.add(gpu_rsrp_val);
      results.cpu_cfo.add(cpu_cfo_val);
      results.gpu_cfo.add(gpu_cfo_val);

      // Accumulate per-iteration GPU−CPU deltas.
      results.delta_sinr.add(cpu_sinr_val, gpu_sinr_val);
      results.delta_evm.add(cpu_evm_pct, gpu_evm_pct);
      results.delta_epre.add(cpu_epre_val, gpu_epre_val);
      results.delta_ta.add(cpu_ta_us, gpu_ta_us);
      results.delta_rsrp.add(cpu_rsrp_val, gpu_rsrp_val);
      results.delta_cfo.add(cpu_cfo_val, gpu_cfo_val);

      // Track TA estimation error when injection is active.
      if (max_ta_offset_us > 0.0f) {
        results.cpu_ta_error.add(cpu_ta_us.has_value()
                                     ? std::optional<float>(cpu_ta_us.value() - injected_ta_us)
                                     : std::nullopt);
        results.gpu_ta_error.add(gpu_ta_us.has_value()
                                     ? std::optional<float>(gpu_ta_us.value() - injected_ta_us)
                                     : std::nullopt);
      }

      // Track CFO estimation error when injection is active.
      if (cfo_std_hz > 0.0f) {
        results.cpu_cfo_error.add(cpu_cfo_val.has_value()
                                      ? std::optional<float>(cpu_cfo_val.value() - injected_cfo_hz)
                                      : std::nullopt);
        results.gpu_cfo_error.add(gpu_cfo_val.has_value()
                                      ? std::optional<float>(gpu_cfo_val.value() - injected_cfo_hz)
                                      : std::nullopt);
      }

      // Helper to format an optional float or "N/A".
      auto fmt_opt = [](std::optional<float> v, const char* fmt_str) -> std::string {
        if (v.has_value()) {
          return fmt::format(fmt::runtime(fmt_str), v.value());
        }
        return "N/A";
      };

      // Per-iteration delta printing.
      fmt::print("[iter {:3d}] CPU: CRC={} SINR={:>6s}dB EVM={:>6s}% EPRE={:>6s}dB TA={:>7s}us RSRP={:>6s}dB "
                 "CFO={:>7s}Hz | "
                 "GPU: CRC={} SINR={:>6s}dB EVM={:>6s}% EPRE={:>6s}dB TA={:>7s}us RSRP={:>6s}dB CFO={:>7s}Hz\n",
                 iter,
                 cpu_result.data.tb_crc_ok ? "PASS" : "FAIL",
                 fmt_opt(cpu_sinr_val, "{:.2f}"),
                 fmt_opt(cpu_evm_val.has_value()
                             ? std::optional<float>(cpu_evm_val.value() * 100.0f)
                             : std::nullopt,
                         "{:.1f}"),
                 fmt_opt(cpu_epre_val, "{:.2f}"),
                 fmt_opt(cpu_ta_us, "{:.2f}"),
                 fmt_opt(cpu_rsrp_val, "{:.1f}"),
                 fmt_opt(cpu_cfo_val, "{:.1f}"),
                 gpu_result.data.tb_crc_ok ? "PASS" : "FAIL",
                 fmt_opt(gpu_sinr_val, "{:.2f}"),
                 fmt_opt(gpu_evm_val.has_value()
                             ? std::optional<float>(gpu_evm_val.value() * 100.0f)
                             : std::nullopt,
                         "{:.1f}"),
                 fmt_opt(gpu_epre_val, "{:.2f}"),
                 fmt_opt(gpu_ta_us, "{:.2f}"),
                 fmt_opt(gpu_rsrp_val, "{:.1f}"),
                 fmt_opt(gpu_cfo_val, "{:.1f}"));

      if (max_ta_offset_us > 0.0f) {
        auto ta_err = [](std::optional<float> est, float inj) -> std::string {
          return est.has_value() ? fmt::format("{:+.3f}", est.value() - inj) : std::string("N/A");
        };
        fmt::print("          TA inject: {:+.2f}us | CPU err: {:>7s}us | GPU err: {:>7s}us\n",
                   injected_ta_us, ta_err(cpu_ta_us, injected_ta_us), ta_err(gpu_ta_us, injected_ta_us));
      }
      if (cfo_std_hz > 0.0f) {
        auto cfo_err = [](std::optional<float> est, float inj) -> std::string {
          return est.has_value() ? fmt::format("{:+.2f}", est.value() - inj) : std::string("N/A");
        };
        fmt::print("          CFO inject: {:+.1f}Hz | CPU err: {:>7s}Hz | GPU err: {:>7s}Hz\n",
                   injected_cfo_hz, cfo_err(cpu_cfo_val, injected_cfo_hz), cfo_err(gpu_cfo_val, injected_cfo_hz));
      }
      if (rx_gain_range_dB > 0.0f) {
        fmt::print("          RX gain: {:+.1f}dB | EPRE delta: {:+.3f}dB | RSRP delta: {:+.3f}dB\n",
                   injected_gain_dB,
                   (cpu_epre_val.has_value() && gpu_epre_val.has_value())
                       ? static_cast<double>(gpu_epre_val.value() - cpu_epre_val.value()) : 0.0,
                   (cpu_rsrp_val.has_value() && gpu_rsrp_val.has_value())
                       ? static_cast<double>(gpu_rsrp_val.value() - cpu_rsrp_val.value()) : 0.0);
      }
    }

    if ((iter + 1) % 10 == 0) {
      fmt::print(".");
      std::cout.flush();
    }
  }

  fmt::print("\n\n=== Results ===\n");
  fmt::print("CPU: {}/{} pass ({:.1f}% BLER)\n",
             results.cpu_pass, nof_iterations,
             100.0f * results.cpu_fail / nof_iterations);
  fmt::print("GPU: {}/{} pass ({:.1f}% BLER)\n",
             results.gpu_pass, nof_iterations,
             100.0f * results.gpu_fail / nof_iterations);
  fmt::print("Disagreements: {} ({:.1f}%)\n",
             results.disagreements,
             100.0f * results.disagreements / nof_iterations);

  if (results.iters_count > 0) {
    fmt::print("\n=== LDPC Iterations (max={}) ===\n", nof_ldpc_iterations);
    fmt::print("CPU avg: {:.2f}\n", results.cpu_iters_sum / results.iters_count);
    fmt::print("GPU avg: {:.2f}\n", results.gpu_iters_sum / results.iters_count);
  }

  fmt::print("\n=== Latency (microseconds) ===\n");
  fmt::print("CPU Latency:\n");
  fmt::print("  Min:  {:8.1f} us\n", results.cpu_latency.min_us);
  fmt::print("  Mean: {:8.1f} us\n", results.cpu_latency.mean());
  fmt::print("  Max:  {:8.1f} us\n", results.cpu_latency.max_us);

  fmt::print("\nGPU Latency:\n");
  fmt::print("  Min:  {:8.1f} us\n", results.gpu_latency.min_us);
  fmt::print("  Mean: {:8.1f} us\n", results.gpu_latency.mean());
  fmt::print("  Max:  {:8.1f} us\n", results.gpu_latency.max_us);

  double speedup = results.cpu_latency.mean() / results.gpu_latency.mean();
  fmt::print("\nSpeedup: {:.2f}x (GPU is {:.1f}% {})\n",
             speedup,
             std::abs(speedup - 1.0) * 100.0,
             speedup > 1.0 ? "faster" : "slower");

#ifdef ENABLE_CUDA
  if (results.gpu_deinterleave.count > 0) {
    fmt::print("\n=== GPU Decoder Breakdown (microseconds) ===\n");
    fmt::print("{:<16s} {:>8s} {:>8s} {:>8s}\n", "", "Min", "Mean", "Max");
    fmt::print("{:<16s} {:8.1f} {:8.1f} {:8.1f}\n", "Deinterleave:",
               results.gpu_deinterleave.min_us, results.gpu_deinterleave.mean(), results.gpu_deinterleave.max_us);
    fmt::print("{:<16s} {:8.1f} {:8.1f} {:8.1f}\n", "Rate Dematch:",
               results.gpu_rate_dematch.min_us, results.gpu_rate_dematch.mean(), results.gpu_rate_dematch.max_us);
    fmt::print("{:<16s} {:8.1f} {:8.1f} {:8.1f}\n", "LDPC Decode:",
               results.gpu_ldpc_decode.min_us, results.gpu_ldpc_decode.mean(), results.gpu_ldpc_decode.max_us);
    fmt::print("{:<16s} {:8.1f} {:8.1f} {:8.1f}\n", "CRC Check:",
               results.gpu_crc_check.min_us, results.gpu_crc_check.mean(), results.gpu_crc_check.max_us);
    fmt::print("{:<16s} {:8.1f} {:8.1f} {:8.1f}\n", "D2H Transfer:",
               results.gpu_d2h_transfer.min_us, results.gpu_d2h_transfer.mean(), results.gpu_d2h_transfer.max_us);

    double total_mean = results.gpu_deinterleave.mean() + results.gpu_rate_dematch.mean() +
                        results.gpu_ldpc_decode.mean() + results.gpu_crc_check.mean() +
                        results.gpu_d2h_transfer.mean();
    fmt::print("{:<16s} {:8s} {:8.1f}\n", "Total Pipeline:", "", total_mean);

    if (total_mean > 0) {
      fmt::print("\nPercentage Breakdown:\n");
      fmt::print("  Deinterleave: {:5.1f}%\n", 100.0 * results.gpu_deinterleave.mean() / total_mean);
      fmt::print("  Rate Dematch: {:5.1f}%\n", 100.0 * results.gpu_rate_dematch.mean() / total_mean);
      fmt::print("  LDPC Decode:  {:5.1f}%\n", 100.0 * results.gpu_ldpc_decode.mean() / total_mean);
      fmt::print("  CRC Check:    {:5.1f}%\n", 100.0 * results.gpu_crc_check.mean() / total_mean);
      fmt::print("  D2H Transfer: {:5.1f}%\n", 100.0 * results.gpu_d2h_transfer.mean() / total_mean);
    }

    // Gap profiling: host-side overhead between pipeline and total latency
    double gap_mean = results.gpu_latency.mean() - total_mean;
    fmt::print("\n=== Gap Profiling (host overhead, microseconds) ===\n");
    fmt::print("{:<16s} {:>8s} {:>8s} {:>8s}\n", "", "Min", "Mean", "Max");
    fmt::print("{:<16s} {:8.1f} {:8.1f} {:8.1f}\n", "Grid Staging:",
               results.gpu_grid_staging.min_us, results.gpu_grid_staging.mean(), results.gpu_grid_staging.max_us);
    fmt::print("{:<16s} {:8.1f} {:8.1f} {:8.1f}\n", "Demod Sync:",
               results.gpu_demod_sync.min_us, results.gpu_demod_sync.mean(), results.gpu_demod_sync.max_us);
    fmt::print("{:<16s} {:8.1f} {:8.1f} {:8.1f}\n", "Extract Bits:",
               results.gpu_extract_bits.min_us, results.gpu_extract_bits.mean(), results.gpu_extract_bits.max_us);
    fmt::print("{:<16s} {:8.1f} {:8.1f} {:8.1f}\n", "Decode Call:",
               results.gpu_decode_call.min_us, results.gpu_decode_call.mean(), results.gpu_decode_call.max_us);
    fmt::print("{:<16s} {:8.1f} {:8.1f} {:8.1f}\n", "SINR Readback:",
               results.gpu_sinr_readback.min_us, results.gpu_sinr_readback.mean(), results.gpu_sinr_readback.max_us);
    double profiled_gap = results.gpu_grid_staging.mean() + results.gpu_demod_sync.mean() +
                          results.gpu_extract_bits.mean() + results.gpu_sinr_readback.mean();
    fmt::print("{:<16s} {:8s} {:8.1f}\n", "Profiled Gap:", "", profiled_gap);
    fmt::print("{:<16s} {:8s} {:8.1f}\n", "Total Gap:", "", gap_mean);
    fmt::print("{:<16s} {:8s} {:8.1f}\n", "Unaccounted:", "", gap_mean - profiled_gap);
  }
#endif

  if (results.disagreements > 0) {
    fmt::print("\n*** WARNING: CPU and GPU disagree on decode results! ***\n");
  } else if (results.cpu_fail == 0 && results.gpu_fail == 0) {
    fmt::print("\nPerfect: Both paths decoded all {} iterations\n", nof_iterations);
  } else {
    fmt::print("\nAgreement: Both paths have same BLER\n");
  }

  // === CSI Metric Parity Report ===
  {
    // Helper to format mean or "N/A".
    auto mean_str = [](const csi_metric_stats& s, const char* fmt_str) -> std::string {
      if (s.has_data()) {
        return fmt::format(fmt::runtime(fmt_str), s.mean());
      }
      return "N/A";
    };

    // Helper to format delta between two stats or "N/A".
    auto delta_str = [](const csi_metric_stats& cpu_s, const csi_metric_stats& gpu_s,
                        const char* fmt_str) -> std::string {
      if (cpu_s.has_data() && gpu_s.has_data()) {
        return fmt::format(fmt::runtime(fmt_str), gpu_s.mean() - cpu_s.mean());
      }
      return "N/A";
    };

    // Helper to format "GPU Populated?" column.
    auto pop_str = [](const csi_metric_stats& gpu_s) -> std::string {
      if (gpu_s.populated == gpu_s.count) {
        return "Yes";
      }
      if (gpu_s.populated == 0) {
        return "NO";
      }
      return fmt::format("Partial ({:.0f}%)", gpu_s.population_rate());
    };

    fmt::print("\n=== CSI Metric Parity Report ===\n");
    fmt::print("{:<16s} {:>10s} {:>10s} {:>10s}   {}\n",
               "", "CPU Mean", "GPU Mean", "Delta", "GPU Populated?");
    fmt::print("{:<16s} {:>10s} {:>10s} {:>10s}   {}\n",
               "SINR (dB):",
               mean_str(results.cpu_sinr, "{:.2f}"),
               mean_str(results.gpu_sinr, "{:.2f}"),
               delta_str(results.cpu_sinr, results.gpu_sinr, "{:+.2f}"),
               pop_str(results.gpu_sinr));
    fmt::print("{:<16s} {:>10s} {:>10s} {:>10s}   {}\n",
               "EVM (%):",
               mean_str(results.cpu_evm, "{:.2f}"),
               mean_str(results.gpu_evm, "{:.2f}"),
               delta_str(results.cpu_evm, results.gpu_evm, "{:+.2f}"),
               pop_str(results.gpu_evm));
    fmt::print("{:<16s} {:>10s} {:>10s} {:>10s}   {}\n",
               "EPRE (dB):",
               mean_str(results.cpu_epre, "{:.2f}"),
               mean_str(results.gpu_epre, "{:.2f}"),
               delta_str(results.cpu_epre, results.gpu_epre, "{:+.2f}"),
               pop_str(results.gpu_epre));
    fmt::print("{:<16s} {:>10s} {:>10s} {:>10s}   {}\n",
               "TA (us):",
               mean_str(results.cpu_ta, "{:.2f}"),
               mean_str(results.gpu_ta, "{:.2f}"),
               delta_str(results.cpu_ta, results.gpu_ta, "{:+.2f}"),
               pop_str(results.gpu_ta));
    fmt::print("{:<16s} {:>10s} {:>10s} {:>10s}   {}\n",
               "RSRP (dB):",
               mean_str(results.cpu_rsrp, "{:.2f}"),
               mean_str(results.gpu_rsrp, "{:.2f}"),
               delta_str(results.cpu_rsrp, results.gpu_rsrp, "{:+.2f}"),
               pop_str(results.gpu_rsrp));
    fmt::print("{:<16s} {:>10s} {:>10s} {:>10s}   {}\n",
               "CFO (Hz):",
               mean_str(results.cpu_cfo, "{:.2f}"),
               mean_str(results.gpu_cfo, "{:.2f}"),
               delta_str(results.cpu_cfo, results.gpu_cfo, "{:+.2f}"),
               pop_str(results.gpu_cfo));

    if (max_ta_offset_us > 0.0f) {
      fmt::print("\n=== TA Estimation Accuracy (injected range: +/-{:.1f}us) ===\n", max_ta_offset_us);
      fmt::print("  CPU mean error: {}us\n", mean_str(results.cpu_ta_error, "{:+.4f}"));
      fmt::print("  GPU mean error: {}us\n", mean_str(results.gpu_ta_error, "{:+.4f}"));
    }

    if (cfo_std_hz > 0.0f) {
      fmt::print("\n=== CFO Estimation Accuracy (injected std: {:.1f}Hz) ===\n", cfo_std_hz);
      fmt::print("  CPU mean error: {}Hz\n", mean_str(results.cpu_cfo_error, "{:+.2f}"));
      fmt::print("  GPU mean error: {}Hz\n", mean_str(results.gpu_cfo_error, "{:+.2f}"));
    }

    // === GPU Metric Gaps ===
    bool any_gap = false;
    auto check_gap = [&any_gap](const csi_metric_stats& gpu_s, const csi_metric_stats& cpu_s,
                                const char* field_name, const char* detail) {
      bool cpu_has  = cpu_s.has_data();
      bool gpu_has  = gpu_s.has_data();
      bool gpu_zero = gpu_has && (std::abs(gpu_s.mean()) < 1e-9);

      if (cpu_has && !gpu_has) {
        if (!any_gap) {
          fmt::print("\n=== GPU Metric Gaps ===\n");
          any_gap = true;
        }
        fmt::print("  WARNING {}: GPU does not populate ({})\n", field_name, detail);
      } else if (cpu_has && gpu_has && gpu_zero && cpu_s.mean() != 0.0) {
        if (!any_gap) {
          fmt::print("\n=== GPU Metric Gaps ===\n");
          any_gap = true;
        }
        fmt::print("  WARNING {}: GPU always reports 0.0 ({})\n", field_name, detail);
      }
    };

    check_gap(results.gpu_ta, results.cpu_ta, "Time Alignment",
              "hardcoded in gpu_stub_estimator_results");
    check_gap(results.gpu_rsrp, results.cpu_rsrp, "RSRP",
              "not computed in E2E kernel");
    check_gap(results.gpu_cfo, results.cpu_cfo, "CFO",
              "not computed in E2E kernel");

    if (!any_gap) {
      fmt::print("\n  No GPU metric gaps detected.\n");
    }
  }

  // === Detailed Parity Analysis ===
  {
    struct parity_row {
      const char*            name;
      const csi_delta_stats& delta;
      double                 mean_tol;
      double                 max_tol;
      const char*            unit;
      bool                   gate_max;
    };
    parity_row rows[] = {
        {"SINR", results.delta_sinr, kTolSinrMean, kTolSinrMax, "dB", true},
        {"EPRE", results.delta_epre, kTolEpreMean, kTolEpreMax, "dB", true},
        {"RSRP", results.delta_rsrp, kTolRsrpMean, kTolRsrpMax, "dB", true},
        {"TA",   results.delta_ta,   kTolTaMean,   kTolTaMax,   "us", true},
        {"CFO",  results.delta_cfo,  kTolCfoMean,  kTolCfoMax,  "Hz", true},
        {"EVM",  results.delta_evm,  kTolEvmMean,  kTolEvmMax,  "%",  false},
    };

    fmt::print("\n=== Detailed Parity Analysis (GPU - CPU) ===\n");
    fmt::print("{:<8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>9s} {:>9s} {:>6s}\n",
               "Metric", "MeanD", "StdDev", "MinD", "MaxD", "MaxAbs", "MeanTol", "MaxTol", "Result");

    unsigned pass_count = 0;
    unsigned total_count = 0;
    for (const auto& r : rows) {
      if (!r.delta.has_data()) {
        fmt::print("{:<8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>9.2f}{} {:>9s} {:>6s}\n",
                   r.name, "N/A", "N/A", "N/A", "N/A", "N/A", r.mean_tol, r.unit, "report", "N/A");
        continue;
      }
      bool pass = (r.delta.missing_gpu == 0) && (std::abs(r.delta.mean()) <= r.mean_tol) &&
                  (!r.gate_max || (r.delta.max_abs <= r.max_tol));
      if (pass) ++pass_count;
      ++total_count;
      std::string max_tol = r.gate_max ? fmt::format("{:.2f}{}", r.max_tol, r.unit) : std::string("report");
      fmt::print("{:<8s} {:>+8.3f} {:>8.3f} {:>+8.3f} {:>+8.3f} {:>8.3f} {:>8.2f}{} {:>9s} {:>6s}\n",
                 r.name,
                 r.delta.mean(),
                 r.delta.stddev(),
                 r.delta.min_val,
                 r.delta.max_val,
                 r.delta.max_abs,
                 r.mean_tol,
                 r.unit,
                 max_tol,
                 pass ? "PASS" : "FAIL");
    }

    if (total_count > 0) {
      bool all_pass = (pass_count == total_count);
      fmt::print("\nOverall: {}/{} metrics within tolerance - {}\n",
                 pass_count, total_count, all_pass ? "ALL PASS" : "SOME FAIL");
    }
  }

  worker_pool->stop();
}

bool run_csi_sweep()
{
  struct sweep_config {
    std::string    channel;
    std::string    fading;
    unsigned       nof_prb;
    sch_mcs_index  mcs_idx;
    pusch_mcs_table mcs_tbl;
    float          snr;
  };

  // Build sweep matrix.
  struct chan_entry { std::string profile; std::string fading; };
  std::vector<chan_entry> channels = {{"single-tap", "uniform-phase"}, {"TDLA", "rayleigh"}};
  std::vector<unsigned>  prbs     = {25, 52, 106, 273};
  struct mcs_entry { unsigned idx; pusch_mcs_table tbl; };
  std::vector<mcs_entry> mcs_list = {
      {2,  pusch_mcs_table::qam64},
      {14, pusch_mcs_table::qam64},
      {20, pusch_mcs_table::qam64},
      {27, pusch_mcs_table::qam256},
  };
  std::vector<float> snrs = {10.0f, 15.0f, 20.0f, 25.0f, 28.0f};

  unsigned total_points = channels.size() * prbs.size() * mcs_list.size() * snrs.size();
  fmt::print("\n=== CSI Parity Sweep ===\n");
  fmt::print("Channels: {}, PRBs: {}, MCS: {}, SNRs: {}, Iters/point: {}\n",
             channels.size(), prbs.size(), mcs_list.size(), snrs.size(), csi_sweep_iters);
  fmt::print("Total: {} sweep points\n", total_points);
  fmt::print("Result gates GPU metric population plus SINR/EPRE/RSRP/TA/CFO/EVM parity and BLER agreement.\n");
  fmt::print("EVM is reported for every point and mean-gated when at least one path decodes a TB successfully.\n\n");

  // Create shared infrastructure once.
  unsigned max_nof_threads = std::min(8U, std::thread::hardware_concurrency());
  auto worker_pool =
      std::make_unique<task_worker_pool<concurrent_queue_policy::locking_mpmc>>("thread", max_nof_threads, 1024);
  auto executor = std::make_unique<task_worker_pool_executor<concurrent_queue_policy::locking_mpmc>>(*worker_pool);

  auto precod_factory = create_channel_precoder_factory("auto");
  auto grid_factory   = create_resource_grid_factory();
  auto pdsch_factory  = create_sw_pdsch_processor_factory(*executor, max_nof_threads + 1, "", "auto");
  auto cpu_factory    = create_sw_pusch_processor_factory(*executor, max_nof_threads + 1,
                                                          nof_ldpc_iterations, use_early_stop, "auto",
                                                          port_channel_estimator_td_interpolation_strategy::average,
                                                          channel_equalizer_algorithm_type::zf);
  auto gpu_factory    = create_sw_pusch_processor_factory(*executor, max_nof_threads + 1,
                                                          nof_ldpc_iterations, use_early_stop, "gpu",
                                                          port_channel_estimator_td_interpolation_strategy::average,
                                                          channel_equalizer_algorithm_type::zf);
  if (!gpu_factory) {
    fmt::print("GPU PUSCH not available, skipping CSI sweep\n");
    worker_pool->stop();
    return false;
  }

  auto pdsch_proc = pdsch_factory->create();
  auto cpu_proc   = cpu_factory->create();
  auto gpu_proc   = gpu_factory->create();
  auto tx_grid    = grid_factory->create(nof_layers, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);
  auto rx_grid    = grid_factory->create(1, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);

  std::random_device rd;
  std::mt19937 rgen(rd());

  // Accumulate results for summary table.
  struct sweep_result {
    std::string channel;
    unsigned    nof_prb;
    unsigned    mcs;
    float       snr;
    csi_delta_stats delta_sinr, delta_epre, delta_rsrp, delta_ta, delta_cfo, delta_evm;
    unsigned    cpu_crc_pass      = 0;
    unsigned    gpu_crc_pass      = 0;
    unsigned    crc_disagreements = 0;
    unsigned    byte_mismatches   = 0;
    bool        pass              = false;
  };
  std::vector<sweep_result> all_results;
  unsigned pass_total = 0;

  unsigned slot_counter = 0;

  for (const auto& ch : channels) {
    for (unsigned nof_prb : prbs) {
      for (const auto& mcs_e : mcs_list) {
        // Precompute config for this PRB/MCS combo.
        sch_mcs_index       mcs_idx  = static_cast<sch_mcs_index>(mcs_e.idx);
        sch_mcs_description mcs_desc = pusch_mcs_get_config(mcs_e.tbl, mcs_idx, false, false);
        prb_interval freq_interval   = {bwp_start_rb, bwp_start_rb + nof_prb};

        tbs_calculator_configuration tbs_config = {};
        tbs_config.mcs_descr    = mcs_desc;
        tbs_config.n_prb        = freq_interval.length();
        tbs_config.nof_layers   = nof_layers;
        tbs_config.nof_symb_sh  = nof_ofdm_symbols;
        tbs_config.nof_dmrs_prb = get_nof_re_per_prb(dmrs) * dmrs_symbols_mask.count() * nof_cdm_groups_without_data;
        unsigned tbs = tbs_calculator_calculate(tbs_config).to_bits().value();

        ldpc_base_graph_type ldpc_bg = get_ldpc_base_graph(mcs_desc.get_normalised_target_code_rate(), units::bits(tbs));
        rb_allocation freq_alloc     = rb_allocation::make_type1(freq_interval.start(), freq_interval.length(), std::nullopt);
        unsigned nof_codeblocks      = compute_nof_codeblocks(units::bits(tbs), ldpc_bg);

        rx_buffer_pool_config pool_config;
        pool_config.max_codeblock_size   = ldpc::MAX_CODEBLOCK_SIZE;
        pool_config.nof_buffers          = 2;
        pool_config.nof_codeblocks       = nof_codeblocks;
        pool_config.expire_timeout_slots = 10;
        pool_config.external_soft_bits   = false;
        auto cpu_pool = create_rx_buffer_pool(pool_config);
        auto gpu_pool = create_rx_buffer_pool(pool_config);

        for (float snr : snrs) {
          sweep_result sr;
          sr.channel = ch.profile;
          sr.nof_prb = nof_prb;
          sr.mcs     = mcs_e.idx;
          sr.snr     = snr;

          channel_emulator emulator(ch.profile, ch.fading, snr, 0.0f, 0,
                                    nof_layers, 1, nof_prb * NOF_SUBCARRIERS_PER_RB,
                                    nof_ofdm_symbols, max_nof_threads, scs, *executor);

          std::vector<uint8_t> tx_data(tbs / 8);
          std::vector<uint8_t> cpu_rx_data(tbs / 8);
          std::vector<uint8_t> gpu_rx_data(tbs / 8);

          for (unsigned iter = 0; iter < csi_sweep_iters; ++iter) {
            for (auto& byte : tx_data) {
              byte = static_cast<uint8_t>(rgen() & 0xff);
            }

            slot_point slot(to_numerology_value(scs), slot_counter++);

            // Build PDSCH PDU.
            pdsch_processor::pdu_t pdsch_pdu;
            pdsch_pdu.context                     = std::nullopt;
            pdsch_pdu.slot                        = slot;
            pdsch_pdu.rnti                        = rnti;
            pdsch_pdu.bwp_size_rb                 = nof_prb;
            pdsch_pdu.bwp_start_rb                = bwp_start_rb;
            pdsch_pdu.cp                          = cy_prefix;
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
            pdsch_pdu.ldpc_base_graph             = ldpc_bg;
            pdsch_pdu.tbs_lbrm                    = tbs_lbrm_default;
            pdsch_pdu.reserved                    = {};
            pdsch_pdu.ratio_pdsch_data_to_sss_dB  = 0.0F;
            pdsch_pdu.ratio_pdsch_dmrs_to_sss_dB  = get_sch_to_dmrs_ratio_dB(nof_cdm_groups_without_data);
            pdsch_pdu.precoding                   = precoding_configuration::make_wideband(make_identity(nof_layers));
            pdsch_pdu.codewords.emplace_back(pdsch_processor::codeword_description{mcs_desc.modulation, rv});

            // Transmit.
            pdsch_processor_notifier_adaptor tx_notifier;
            pdsch_proc->process(tx_grid->get_writer(), tx_notifier, {shared_transport_block(tx_data)}, pdsch_pdu);
            tx_notifier.wait_for_completion();

            // Channel.
            emulator.run(rx_grid->get_writer(), tx_grid->get_reader());

            // PUSCH PDU.
            static_vector<uint8_t, MAX_PORTS> rx_ports(1);
            std::iota(rx_ports.begin(), rx_ports.end(), 0U);

            pusch_processor::pdu_t pusch_pdu;
            pusch_pdu.context            = std::nullopt;
            pusch_pdu.slot               = slot;
            pusch_pdu.rnti               = rnti;
            pusch_pdu.bwp_size_rb        = nof_prb;
            pusch_pdu.bwp_start_rb       = bwp_start_rb;
            pusch_pdu.cp                 = cy_prefix;
            pusch_pdu.mcs_descr          = mcs_desc;
            pusch_pdu.codeword           = {rv, ldpc_bg, true};
            pusch_pdu.uci                = {};
            pusch_pdu.n_id               = n_id;
            pusch_pdu.nof_tx_layers      = nof_layers;
            pusch_pdu.rx_ports           = rx_ports;
            pusch_pdu.dmrs_symbol_mask   = dmrs_symbols_mask;
            pusch_pdu.dmrs               = pusch_processor::dmrs_configuration{
                .dmrs                        = dmrs,
                .scrambling_id               = scrambling_id,
                .n_scid                      = n_scid,
                .nof_cdm_groups_without_data = nof_cdm_groups_without_data};
            pusch_pdu.tbs_lbrm           = tbs_lbrm_default;
            pusch_pdu.freq_alloc         = freq_alloc;
            pusch_pdu.start_symbol_index = 0;
            pusch_pdu.nof_symbols        = nof_ofdm_symbols;
            pusch_pdu.dc_position        = std::nullopt;

            // CPU decode.
            unique_rx_buffer cpu_buffer =
                cpu_pool->get_pool().reserve(slot, trx_buffer_identifier(rnti, 0), nof_codeblocks, true);
            pusch_processor_notifier_adaptor cpu_notifier;
            cpu_proc->process(cpu_rx_data, std::move(cpu_buffer), cpu_notifier, rx_grid->get_reader(), pusch_pdu);
            const auto& cpu_result = cpu_notifier.wait_for_completion();

            // GPU decode.
            unique_rx_buffer gpu_buffer =
                gpu_pool->get_pool().reserve(slot, trx_buffer_identifier(rnti, 1), nof_codeblocks, true);
            pusch_processor_notifier_adaptor gpu_notifier;
            gpu_proc->process(gpu_rx_data, std::move(gpu_buffer), gpu_notifier, rx_grid->get_reader(), pusch_pdu);
            const auto& gpu_result = gpu_notifier.wait_for_completion();

            bool cpu_crc_ok = cpu_result.data.tb_crc_ok;
            bool gpu_crc_ok = gpu_result.data.tb_crc_ok;
            if (cpu_crc_ok) {
              ++sr.cpu_crc_pass;
            }
            if (gpu_crc_ok) {
              ++sr.gpu_crc_pass;
            }
            if (cpu_crc_ok != gpu_crc_ok) {
              ++sr.crc_disagreements;
            }
            if (cpu_crc_ok && gpu_crc_ok && (cpu_rx_data != gpu_rx_data)) {
              ++sr.byte_mismatches;
            }

            // Extract CSI deltas.
            const auto& cpu_csi = cpu_result.csi;
            const auto& gpu_csi = gpu_result.csi;

            auto cpu_ta_raw = cpu_csi.get_time_alignment();
            auto gpu_ta_raw = gpu_csi.get_time_alignment();
            std::optional<float> cpu_ta_us =
                cpu_ta_raw.has_value()
                    ? std::optional<float>(static_cast<float>(cpu_ta_raw.value().to_seconds() * 1e6))
                    : std::nullopt;
            std::optional<float> gpu_ta_us =
                gpu_ta_raw.has_value()
                    ? std::optional<float>(static_cast<float>(gpu_ta_raw.value().to_seconds() * 1e6))
                    : std::nullopt;

            auto cpu_evm_val = cpu_csi.get_total_evm();
            auto gpu_evm_val = gpu_csi.get_total_evm();
            std::optional<float> cpu_evm_pct =
                cpu_evm_val.has_value() ? std::optional<float>(cpu_evm_val.value() * 100.0f) : std::nullopt;
            std::optional<float> gpu_evm_pct =
                gpu_evm_val.has_value() ? std::optional<float>(gpu_evm_val.value() * 100.0f) : std::nullopt;

            sr.delta_sinr.add(cpu_csi.get_sinr_dB(), gpu_csi.get_sinr_dB());
            sr.delta_epre.add(cpu_csi.get_epre_dB(), gpu_csi.get_epre_dB());
            sr.delta_rsrp.add(cpu_csi.get_rsrp_dB(), gpu_csi.get_rsrp_dB());
            sr.delta_ta.add(cpu_ta_us, gpu_ta_us);
            sr.delta_cfo.add(cpu_csi.get_cfo_Hz(), gpu_csi.get_cfo_Hz());
            sr.delta_evm.add(cpu_evm_pct, gpu_evm_pct);
          }

          // Check pass/fail for this point. If CPU reported a CSI metric, GPU must report it too.
          auto metric_present = [](const csi_delta_stats& d) { return d.missing_gpu == 0; };
          auto metric_close = [](const csi_delta_stats& d, double mean_tol, double max_tol, bool gate_max) {
            return !d.has_data() ||
                   ((std::abs(d.mean()) <= mean_tol) && (!gate_max || (d.max_abs <= max_tol)));
          };
          auto metric_pass = [&metric_present, &metric_close](const csi_delta_stats& d,
                                                              double                 mean_tol,
                                                              double                 max_tol,
                                                              bool                   gate_max) {
            return metric_present(d) && metric_close(d, mean_tol, max_tol, gate_max);
          };
          unsigned crc_delta = (sr.cpu_crc_pass > sr.gpu_crc_pass) ? (sr.cpu_crc_pass - sr.gpu_crc_pass)
                                                                   : (sr.gpu_crc_pass - sr.cpu_crc_pass);
          unsigned allowed_crc_delta = std::max(1U, (csi_sweep_iters + 9U) / 10U);
          bool gate_evm_tolerance = (sr.cpu_crc_pass != 0) || (sr.gpu_crc_pass != 0);
          sr.pass = metric_pass(sr.delta_sinr, kTolSinrMean, kTolSinrMax, true) &&
                    metric_pass(sr.delta_epre, kTolEpreMean, kTolEpreMax, true) &&
                    metric_pass(sr.delta_rsrp, kTolRsrpMean, kTolRsrpMax, true) &&
                    metric_pass(sr.delta_ta, kTolTaMean, kTolTaMax, true) &&
                    metric_pass(sr.delta_cfo, kTolCfoMean, kTolCfoMax, true) &&
                    metric_present(sr.delta_evm) &&
                    (!gate_evm_tolerance || metric_close(sr.delta_evm, kTolEvmMean, kTolEvmMax, false)) &&
                    (crc_delta <= allowed_crc_delta) && (sr.crc_disagreements <= allowed_crc_delta) &&
                    (sr.byte_mismatches == 0);

          if (sr.pass) ++pass_total;
          all_results.push_back(std::move(sr));

          fmt::print(".");
          std::cout.flush();
        }
      }
    }
  }

  // Print summary table.
  fmt::print("\n\n{:<12s} {:>4s} {:>4s} {:>5s} {:>8s} {:>8s} {:>5s} {:>5s}  {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}  {:>6s}\n",
             "Channel", "PRB", "MCS", "SNR", "CPU_BLER", "GPU_BLER", "CRC_D", "Bytes",
             "SINR_Δ", "EPRE_Δ", "RSRP_Δ", "TA_Δ", "CFO_Δ", "EVM_Δ", "Result");

  for (const auto& sr : all_results) {
    auto delta_mean = [](const csi_delta_stats& d) -> std::string {
      return d.has_data() ? fmt::format("{:+.2f}", d.mean()) : std::string("N/A");
    };

    auto bler_str = [](unsigned crc_pass, unsigned total) -> std::string {
      return fmt::format("{:.1f}%", 100.0 * static_cast<double>(total - crc_pass) / static_cast<double>(total));
    };
    unsigned crc_delta = (sr.cpu_crc_pass > sr.gpu_crc_pass) ? (sr.cpu_crc_pass - sr.gpu_crc_pass)
                                                             : (sr.gpu_crc_pass - sr.cpu_crc_pass);

    fmt::print("{:<12s} {:>4d} {:>4d} {:>5.1f} {:>8s} {:>8s} {:>5d} {:>5d}  {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}  {:>6s}\n",
               sr.channel, sr.nof_prb, sr.mcs, sr.snr,
               bler_str(sr.cpu_crc_pass, csi_sweep_iters),
               bler_str(sr.gpu_crc_pass, csi_sweep_iters),
               crc_delta,
               sr.byte_mismatches,
               delta_mean(sr.delta_sinr),
               delta_mean(sr.delta_epre),
               delta_mean(sr.delta_rsrp),
               delta_mean(sr.delta_ta),
               delta_mean(sr.delta_cfo),
               delta_mean(sr.delta_evm),
               sr.pass ? "PASS" : "FAIL");
  }

  fmt::print("\nOverall: {}/{} PASS\n", pass_total, total_points);

  // Print failing details.
  bool has_failures = false;
  for (const auto& sr : all_results) {
    if (sr.pass) continue;
    if (!has_failures) {
      fmt::print("Failing:\n");
      has_failures = true;
    }
    std::string reasons;
    auto append_reason = [&reasons](const std::string& reason) {
      if (!reasons.empty()) reasons += ", ";
      reasons += reason;
    };
    auto check_metric = [&append_reason](const csi_delta_stats& d,
                                         const char*            name,
                                         double                 mean_tol,
                                         double                 max_tol,
                                         const char*            unit,
                                         bool                   gate_max,
                                         bool                   gate_tolerance = true) {
      if (d.missing_gpu != 0) {
        append_reason(fmt::format("{} missing_gpu={}/{}", name, d.missing_gpu, d.cpu_present));
      }
      if (!gate_tolerance) {
        return;
      }
      if (d.has_data() && std::abs(d.mean()) > mean_tol) {
        append_reason(fmt::format("{} mean={:+.3f}{} (tol={:.2f})", name, d.mean(), unit, mean_tol));
      }
      if (gate_max && d.has_data() && d.max_abs > max_tol) {
        append_reason(fmt::format("{} max_abs={:.3f}{} (tol={:.2f})", name, d.max_abs, unit, max_tol));
      }
    };
    unsigned crc_delta = (sr.cpu_crc_pass > sr.gpu_crc_pass) ? (sr.cpu_crc_pass - sr.gpu_crc_pass)
                                                             : (sr.gpu_crc_pass - sr.cpu_crc_pass);
    unsigned allowed_crc_delta = std::max(1U, (csi_sweep_iters + 9U) / 10U);
    if (crc_delta > allowed_crc_delta) {
      append_reason(fmt::format("BLER delta count={} (tol={})", crc_delta, allowed_crc_delta));
    }
    if (sr.crc_disagreements > allowed_crc_delta) {
      append_reason(fmt::format("CRC disagreements={} (tol={})", sr.crc_disagreements, allowed_crc_delta));
    }
    if (sr.byte_mismatches != 0) {
      append_reason(fmt::format("byte_mismatches={}", sr.byte_mismatches));
    }
    bool gate_evm_tolerance = (sr.cpu_crc_pass != 0) || (sr.gpu_crc_pass != 0);
    check_metric(sr.delta_sinr, "SINR", kTolSinrMean, kTolSinrMax, "dB", true);
    check_metric(sr.delta_epre, "EPRE", kTolEpreMean, kTolEpreMax, "dB", true);
    check_metric(sr.delta_rsrp, "RSRP", kTolRsrpMean, kTolRsrpMax, "dB", true);
    check_metric(sr.delta_ta, "TA", kTolTaMean, kTolTaMax, "us", true);
    check_metric(sr.delta_cfo, "CFO", kTolCfoMean, kTolCfoMax, "Hz", true);
    check_metric(sr.delta_evm, "EVM", kTolEvmMean, kTolEvmMax, "%", false, gate_evm_tolerance);
    fmt::print("  PRB={} MCS={} SNR={:.1f} {}: {}\n",
               sr.nof_prb, sr.mcs, sr.snr, sr.channel, reasons);
  }

  worker_pool->stop();
  return pass_total == total_points;
}

} // namespace

int main(int argc, char** argv)
{
  // Parse args
  unsigned nof_prb = 5;
  float sinr_dB = 15.0f;
  sch_mcs_index mcs_idx = static_cast<sch_mcs_index>(20);

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--prb" && i + 1 < argc) {
      nof_prb = std::stoi(argv[++i]);
    } else if (arg == "--sinr" && i + 1 < argc) {
      sinr_dB = std::stof(argv[++i]);
    } else if (arg == "--mcs" && i + 1 < argc) {
      mcs_idx = static_cast<sch_mcs_index>(std::stoi(argv[++i]));
    } else if (arg == "--iterations" && i + 1 < argc) {
      nof_iterations = std::stoi(argv[++i]);
    } else if (arg == "--mcs-table" && i + 1 < argc) {
      std::string table_str = argv[++i];
      if (table_str == "qam64") {
        mcs_table = pusch_mcs_table::qam64;
      } else if (table_str == "qam256") {
        mcs_table = pusch_mcs_table::qam256;
      } else if (table_str == "qam64LowSe") {
        mcs_table = pusch_mcs_table::qam64LowSe;
      } else {
        fmt::print("Unknown MCS table: {}. Use qam64, qam256, or qam64LowSe\n", table_str);
        return 1;
      }
    } else if (arg == "--ta-offset" && i + 1 < argc) {
      max_ta_offset_us = std::stof(argv[++i]);
    } else if (arg == "--cfo-std" && i + 1 < argc) {
      cfo_std_hz = std::stof(argv[++i]);
    } else if (arg == "--rnti" && i + 1 < argc) {
      rnti = static_cast<uint16_t>(std::stoi(argv[++i]));
    } else if (arg == "--n-id" && i + 1 < argc) {
      n_id = std::stoi(argv[++i]);
    } else if (arg == "--scrambling-id" && i + 1 < argc) {
      scrambling_id = std::stoi(argv[++i]);
    } else if (arg == "--n-scid" && i + 1 < argc) {
      n_scid = (std::stoi(argv[++i]) != 0);
    } else if (arg == "--dmrs-mask" && i + 1 < argc) {
      unsigned mask = static_cast<unsigned>(std::stoi(argv[++i], nullptr, 0));
      dmrs_symbols_mask = {};
      for (unsigned s = 0; s < 14; s++) {
        if (mask & (1U << s)) {
          dmrs_symbols_mask.set(s);
        }
      }
    } else if (arg == "--channel" && i + 1 < argc) {
      channel_profile = argv[++i];
    } else if (arg == "--fading" && i + 1 < argc) {
      fading_dist = argv[++i];
    } else if (arg == "--csi-sweep") {
      csi_sweep_mode = true;
    } else if (arg == "--csi-sweep-iters" && i + 1 < argc) {
      csi_sweep_iters = std::stoi(argv[++i]);
    } else if (arg == "--rx-gain-range" && i + 1 < argc) {
      rx_gain_range_dB = std::stof(argv[++i]);
    } else if (arg == "--help") {
      fmt::print("Usage: {} [options]\n", argv[0]);
      fmt::print("Options:\n");
      fmt::print("  --prb N             Number of PRBs (default: 5)\n");
      fmt::print("  --sinr N            SINR in dB (default: 15.0)\n");
      fmt::print("  --mcs N             MCS index (default: 20)\n");
      fmt::print("  --mcs-table T       MCS table: qam64, qam256, qam64LowSe (default: qam64)\n");
      fmt::print("  --iterations N      Number of iterations (default: 100)\n");
      fmt::print("  --ta-offset N       Max random TA offset in us (default: 0, disabled)\n");
      fmt::print("  --cfo-std N         CFO standard deviation in Hz (default: 0, disabled)\n");
      fmt::print("  --channel STR       Channel model: single-tap, TDLA, TDLB, TDLC (default: single-tap)\n");
      fmt::print("  --fading STR        Fading distribution: uniform-phase, rayleigh (default: uniform-phase)\n");
      fmt::print("  --csi-sweep         Run CSI parity sweep across channel/PRB/MCS/SNR matrix\n");
      fmt::print("  --csi-sweep-iters N Iterations per sweep point (default: 50)\n");
      fmt::print("  --rx-gain-range N   Random RX gain in +/-N dB per iteration (default: 0, disabled)\n");
      fmt::print("  --rnti N            RNTI value (default: 0x1234)\n");
      fmt::print("  --n-id N            Data scrambling n_ID (default: 0)\n");
      fmt::print("  --scrambling-id N   DMRS scrambling ID (default: 0)\n");
      fmt::print("  --n-scid N          DMRS n_SCID 0 or 1 (default: 0)\n");
      fmt::print("  --dmrs-mask N       DMRS symbol bitmask (hex/dec, default: 0x0804 = syms 2,11)\n");
      return 0;
    }
  }

  ocudulog::init();
  ocudulog::fetch_basic_logger("ALL").set_level(ocudulog::basic_levels::warning);
  ocudulog::fetch_basic_logger("PHY").set_level(ocudulog::basic_levels::warning);

  if (csi_sweep_mode) {
    return run_csi_sweep() ? 0 : 1;
  }

  run_stress_test(nof_prb, sinr_dB, mcs_idx);
  return 0;
}
