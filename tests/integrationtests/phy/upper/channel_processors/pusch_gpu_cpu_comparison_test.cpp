// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief GPU vs CPU PUSCH Metrics Comparison Test
///
/// This test performs realistic end-to-end PUSCH processing through both CPU and GPU paths
/// and compares the reported metrics (SINR, BLER, EVM, decoder iterations).
///
/// Signal flow:
/// [Random TB Data] -> [PDSCH Encoder] -> [TX Grid with DMRS]
///                                              |
///                                    [Channel Emulator (AWGN)]
///                                              |
///                                       [RX Grid (noisy)]
///                                        |             |
///                                 [CPU PUSCH]   [GPU PUSCH]
///                                        |             |
///                                [CPU Metrics] [GPU Metrics]
///                                        |             |
///                                      [COMPARE & REPORT]

#include "pxsch_bler_test_channel_emulator.h"
#include "pxsch_bler_test_factories.h"
#include "ocudu/phy/support/resource_grid.h"
#include "ocudu/phy/support/support_factories.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_decoder_result.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_processor_result_notifier.h"
#include "ocudu/phy/upper/rx_buffer_pool.h"
#include "ocudu/phy/upper/unique_rx_buffer.h"
#include "ocudu/ran/precoding/precoding_codebooks.h"
#include "ocudu/ran/pusch/pusch_mcs.h"
#include "ocudu/ran/resource_allocation/rb_interval.h"
#include "ocudu/ran/sch/sch_dmrs_power.h"
#include "ocudu/ran/sch/sch_mcs.h"
#include "ocudu/ran/sch/sch_segmentation.h"
#include "ocudu/ran/sch/tbs_calculator.h"
#include "ocudu/support/executors/task_worker_pool.h"
#include "ocudu/support/math/stats.h"
#include <condition_variable>
#include <getopt.h>
#include <mutex>
#include <random>
#include <thread>

using namespace ocudu;

// Default test parameters
static constexpr subcarrier_spacing scs                         = subcarrier_spacing::kHz30;
static constexpr uint16_t           rnti                        = 0x1234;
static constexpr unsigned           bwp_start_rb                = 0;
static constexpr unsigned           nof_ofdm_symbols            = 14;
static const symbol_slot_mask       dmrs_symbols_mask            = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
static constexpr unsigned           nof_ldpc_iterations         = 10;
static constexpr dmrs_config_type          dmrs                        = dmrs_config_type::type1;
static constexpr unsigned           nof_cdm_groups_without_data = 2;
static constexpr cyclic_prefix      cy_prefix                   = cyclic_prefix::NORMAL;
static constexpr unsigned           rv                          = 0;
static constexpr unsigned           n_id                        = 0;
static constexpr unsigned           scrambling_id               = 0;
static constexpr bool               n_scid                      = false;
static constexpr bool               use_early_stop              = true;

// Configurable test parameters
static unsigned              max_nof_threads = std::min(8U, std::thread::hardware_concurrency());
static unsigned              nof_repetitions = 100;
static std::vector<unsigned> prb_allocations = {3, 4, 5, 6};
static std::vector<float>    sinr_values_dB  = {5.0F, 10.0F, 15.0F, 20.0F, 25.0F, 30.0F};
static std::vector<unsigned> rx_port_counts  = {1, 2};
static unsigned              nof_layers      = 1;
static pusch_mcs_table       mcs_table       = pusch_mcs_table::qam64;
static sch_mcs_index         mcs_index       = 0; // QPSK for reliable decoding

namespace {

/// Structure to hold comparison results for a single configuration.
struct comparison_results {
  unsigned nof_prb;
  float    target_sinr_dB;
  unsigned nof_rx_ports;

  sample_statistics<float> cpu_sinr_stats;
  sample_statistics<float> gpu_sinr_stats;
  sample_statistics<float> cpu_evm_stats;
  sample_statistics<float> gpu_evm_stats;
  sample_statistics<float> cpu_ta_stats_us;
  sample_statistics<float> gpu_ta_stats_us;

  unsigned cpu_crc_errors      = 0;
  unsigned gpu_crc_errors      = 0;
  unsigned cpu_data_errors     = 0;
  unsigned gpu_data_errors     = 0;
  unsigned total_transmissions = 0;

  sample_statistics<float> cpu_iterations_stats;
  sample_statistics<float> gpu_iterations_stats;

  sample_statistics<float> cpu_epre_stats;
  sample_statistics<float> gpu_epre_stats;

  // Track decode agreement
  unsigned decode_agreement = 0; // Both paths agree on CRC result
};

class pusch_gpu_cpu_comparison_test
{
public:
  pusch_gpu_cpu_comparison_test() { ocudulog::init(); }

  void run()
  {
    print_header();

    // Run tests for each configuration
    for (unsigned nof_rx_ports : rx_port_counts) {
      for (unsigned nof_prb : prb_allocations) {
        fmt::print("\n--- PRB={}, Rx Ports={} ---\n", nof_prb, nof_rx_ports);

        for (float sinr_dB : sinr_values_dB) {
          comparison_results results = run_comparison(nof_prb, sinr_dB, nof_rx_ports);
          all_results.push_back(results);
          print_results(results);
        }
      }
    }

    print_summary();
  }

private:
  std::shared_ptr<resource_grid_factory> create_grid_factory()
  {
    std::shared_ptr<channel_precoder_factory> precod_factory = create_channel_precoder_factory("auto");
    report_fatal_error_if_not(precod_factory, "Failed to create channel precoding factory.");
    return create_resource_grid_factory();
  }

  /// Implements a PDSCH processor notifier adaptor for synchronization.
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

  private:
    std::atomic<bool> completed = {false};
  };

  /// Implements a PUSCH processor notifier adaptor for synchronizing decoder results.
  class pusch_processor_notifier_adaptor : public pusch_processor_result_notifier
  {
  public:
    void on_uci(const pusch_processor_result_control& uci_) override
    {
      std::unique_lock<std::mutex> lock(mutex);
      uci = uci_;
    }

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
    pusch_processor_result_control uci;
    pusch_processor_result_data    sch;
    std::mutex                     mutex;
    std::condition_variable        cvar;
  };

  comparison_results run_comparison(unsigned nof_prb, float sinr_dB, unsigned nof_rx_ports)
  {
    comparison_results results;
    results.nof_prb        = nof_prb;
    results.target_sinr_dB = sinr_dB;
    results.nof_rx_ports   = nof_rx_ports;

    // Prepare executors
    auto worker_pool =
        std::make_unique<task_worker_pool<concurrent_queue_policy::locking_mpmc>>("thread", max_nof_threads, 1024);
    auto executor = std::make_unique<task_worker_pool_executor<concurrent_queue_policy::locking_mpmc>>(*worker_pool);

    // Prepare logging
    ocudulog::fetch_basic_logger("ALL").set_level(ocudulog::basic_levels::warning);

    // Compute modulation and code scheme (QPSK for reliable decoding)
    sch_mcs_description mcs_descr = pusch_mcs_get_config(mcs_table, mcs_index, false, false);

    // Frequency allocation
    prb_interval freq_allocation = {bwp_start_rb, bwp_start_rb + nof_prb};

    // Calculate transport block size
    tbs_calculator_configuration tbs_config = {};
    tbs_config.mcs_descr                    = mcs_descr;
    tbs_config.n_prb                        = freq_allocation.length();
    tbs_config.nof_layers                   = nof_layers;
    tbs_config.nof_symb_sh                  = nof_ofdm_symbols;
    tbs_config.nof_dmrs_prb = get_nof_re_per_prb(dmrs) * dmrs_symbols_mask.count() * nof_cdm_groups_without_data;
    unsigned tbs            = tbs_calculator_calculate(tbs_config).to_bits().value();

    // Select LDPC base graph
    ldpc_base_graph_type ldpc_base_graph =
        get_ldpc_base_graph(mcs_descr.get_normalised_target_code_rate(), units::bits(tbs));

    // Generate frequency allocation
    rb_allocation freq_alloc = rb_allocation::make_type1(freq_allocation.start(), freq_allocation.length(), std::nullopt);

    // Create PDSCH processor factory (for TX)
    std::shared_ptr<pdsch_processor_factory> pdsch_proc_factory =
        create_sw_pdsch_processor_factory(*executor, max_nof_threads + 1, "", "auto");
    report_fatal_error_if_not(pdsch_proc_factory, "Failed to create PDSCH processor factory.");

    // Create CPU PUSCH processor factory
    std::shared_ptr<pusch_processor_factory> cpu_pusch_proc_factory =
        create_sw_pusch_processor_factory(*executor,
                                          max_nof_threads + 1,
                                          nof_ldpc_iterations,
                                          use_early_stop,
                                          "auto",
                                          port_channel_estimator_td_interpolation_strategy::average,
                                          channel_equalizer_algorithm_type::zf);
    report_fatal_error_if_not(cpu_pusch_proc_factory, "Failed to create CPU PUSCH processor factory.");

    // Create GPU PUSCH processor factory
    std::shared_ptr<pusch_processor_factory> gpu_pusch_proc_factory =
        create_sw_pusch_processor_factory(*executor,
                                          max_nof_threads + 1,
                                          nof_ldpc_iterations,
                                          use_early_stop,
                                          "gpu",
                                          port_channel_estimator_td_interpolation_strategy::average,
                                          channel_equalizer_algorithm_type::zf);
    if (!gpu_pusch_proc_factory) {
      fmt::print("  [SKIP] GPU PUSCH processor not available\n");
      worker_pool->stop();
      return results;
    }

    // Create resource grid factory
    std::shared_ptr<resource_grid_factory> grid_factory = create_grid_factory();
    report_fatal_error_if_not(grid_factory, "Failed to create resource grid factory.");

    // Create processors
    std::unique_ptr<pdsch_processor> transmitter = pdsch_proc_factory->create();
    report_fatal_error_if_not(transmitter, "Failed to create PDSCH processor.");

    std::unique_ptr<pusch_processor> cpu_receiver = cpu_pusch_proc_factory->create();
    report_fatal_error_if_not(cpu_receiver, "Failed to create CPU PUSCH processor.");

    std::unique_ptr<pusch_processor> gpu_receiver = gpu_pusch_proc_factory->create();
    report_fatal_error_if_not(gpu_receiver, "Failed to create GPU PUSCH processor.");

    // Create resource grids
    std::unique_ptr<resource_grid> tx_grid = grid_factory->create(nof_layers, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);
    std::unique_ptr<resource_grid> rx_grid =
        grid_factory->create(nof_rx_ports, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);

    // Calculate number of codeblocks
    unsigned nof_codeblocks = compute_nof_codeblocks(units::bits(tbs), ldpc_base_graph);

    // Prepare receive soft buffer pools (separate for CPU and GPU)
    rx_buffer_pool_config buffer_pool_config;
    buffer_pool_config.max_codeblock_size   = ldpc::MAX_CODEBLOCK_SIZE;
    buffer_pool_config.nof_buffers          = 2;
    buffer_pool_config.nof_codeblocks       = nof_codeblocks;
    buffer_pool_config.expire_timeout_slots = 10;
    buffer_pool_config.external_soft_bits   = false;

    std::unique_ptr<rx_buffer_pool_controller> cpu_buffer_pool = create_rx_buffer_pool(buffer_pool_config);
    report_error_if_not(cpu_buffer_pool, "Failed to create CPU buffer pool.");

    std::unique_ptr<rx_buffer_pool_controller> gpu_buffer_pool = create_rx_buffer_pool(buffer_pool_config);
    report_error_if_not(gpu_buffer_pool, "Failed to create GPU buffer pool.");

    // Prepare PDSCH processor configuration
    pdsch_processor::pdu_t pdsch_config;
    pdsch_config.context                     = std::nullopt;
    pdsch_config.slot                        = slot_point(to_numerology_value(scs), 0);
    pdsch_config.rnti                        = rnti;
    pdsch_config.bwp_size_rb                 = nof_prb;
    pdsch_config.bwp_start_rb                = bwp_start_rb;
    pdsch_config.cp                          = cy_prefix;
    pdsch_config.n_id                        = n_id;
    pdsch_config.ref_point                   = pdsch_processor::pdu_t::PRB0;
    pdsch_config.dmrs_symbol_mask            = dmrs_symbols_mask;
    pdsch_config.dmrs                        = dmrs;
    pdsch_config.scrambling_id               = scrambling_id;
    pdsch_config.n_scid                      = n_scid;
    pdsch_config.nof_cdm_groups_without_data = nof_cdm_groups_without_data;
    pdsch_config.freq_alloc                  = freq_alloc;
    pdsch_config.start_symbol_index          = 0;
    pdsch_config.nof_symbols                 = nof_ofdm_symbols;
    pdsch_config.ldpc_base_graph             = ldpc_base_graph;
    pdsch_config.tbs_lbrm                    = tbs_lbrm_default;
    pdsch_config.reserved                    = {};
    pdsch_config.ratio_pdsch_data_to_sss_dB  = 0.0F;
    pdsch_config.ratio_pdsch_dmrs_to_sss_dB  = get_sch_to_dmrs_ratio_dB(nof_cdm_groups_without_data);
    pdsch_config.precoding                   = precoding_configuration::make_wideband(make_identity(nof_layers));
    pdsch_config.codewords.emplace_back(pdsch_processor::codeword_description{mcs_descr.modulation, rv});

    static_vector<uint8_t, MAX_PORTS> rx_ports(nof_rx_ports);
    std::iota(rx_ports.begin(), rx_ports.end(), 0U);

    // Prepare PUSCH processor configuration
    pusch_processor::pdu_t pusch_config;
    pusch_config.context            = std::nullopt;
    pusch_config.slot               = slot_point(to_numerology_value(scs), 0);
    pusch_config.rnti               = rnti;
    pusch_config.bwp_size_rb        = nof_prb;
    pusch_config.bwp_start_rb       = bwp_start_rb;
    pusch_config.cp                 = cy_prefix;
    pusch_config.mcs_descr          = mcs_descr;
    pusch_config.codeword           = {rv, ldpc_base_graph, true};
    pusch_config.uci                = {};
    pusch_config.n_id               = n_id;
    pusch_config.nof_tx_layers      = nof_layers;
    pusch_config.rx_ports           = rx_ports;
    pusch_config.dmrs_symbol_mask   = dmrs_symbols_mask;
    pusch_config.dmrs               = pusch_processor::dmrs_configuration{.dmrs                        = dmrs,
                                                                          .scrambling_id               = scrambling_id,
                                                                          .n_scid                      = n_scid,
                                                                          .nof_cdm_groups_without_data = nof_cdm_groups_without_data};
    pusch_config.freq_alloc         = freq_alloc;
    pusch_config.start_symbol_index = 0;
    pusch_config.nof_symbols        = nof_ofdm_symbols;
    pusch_config.tbs_lbrm           = tbs_lbrm_default;
    pusch_config.dc_position        = {};

    // Resize data buffers
    std::vector<uint8_t> tx_data(tbs / 8);
    std::vector<uint8_t> cpu_rx_data(tbs / 8);
    std::vector<uint8_t> gpu_rx_data(tbs / 8);

    // Create channel emulator
    auto emulator = std::make_unique<channel_emulator>("single-tap",
                                                        "uniform-phase",
                                                        sinr_dB,
                                                        0.0F, // CFO
                                                        0,    // corrupted RE
                                                        nof_layers,
                                                        nof_rx_ports,
                                                        MAX_NOF_SUBCARRIERS,
                                                        nof_ofdm_symbols,
                                                        max_nof_threads,
                                                        scs,
                                                        *executor);

    // Random generator
    std::mt19937 rgen(0);

    // Run test iterations
    for (unsigned n = 0; n != nof_repetitions; ++n) {
      // Generate random data
      std::generate(tx_data.begin(), tx_data.end(), [&rgen]() { return static_cast<uint8_t>(rgen() & 0xff); });

      // Process PDSCH (transmit)
      pdsch_processor_notifier_adaptor tx_notifier;
      transmitter->process(tx_grid->get_writer(), tx_notifier, {shared_transport_block(tx_data)}, pdsch_config);
      tx_notifier.wait_for_completion();

      // Apply channel
      emulator->run(rx_grid->get_writer(), tx_grid->get_reader());

      // Get CPU receive buffer
      unique_rx_buffer cpu_buffer =
          cpu_buffer_pool->get_pool().reserve(pusch_config.slot, trx_buffer_identifier(rnti, 0), nof_codeblocks, true);
      report_error_if_not(cpu_buffer, "Invalid CPU buffer.");

      // Get GPU receive buffer
      unique_rx_buffer gpu_buffer =
          gpu_buffer_pool->get_pool().reserve(pusch_config.slot, trx_buffer_identifier(rnti, 1), nof_codeblocks, true);
      report_error_if_not(gpu_buffer, "Invalid GPU buffer.");

      // Process CPU PUSCH
      pusch_processor_notifier_adaptor cpu_notifier;
      cpu_receiver->process(cpu_rx_data, std::move(cpu_buffer), cpu_notifier, rx_grid->get_reader(), pusch_config);
      const pusch_processor_result_data& cpu_result = cpu_notifier.wait_for_completion();

      // Process GPU PUSCH (using same rx_grid)
      pusch_processor_notifier_adaptor gpu_notifier;
      gpu_receiver->process(gpu_rx_data, std::move(gpu_buffer), gpu_notifier, rx_grid->get_reader(), pusch_config);
      const pusch_processor_result_data& gpu_result = gpu_notifier.wait_for_completion();

      // Collect CPU metrics
      if (cpu_result.csi.get_sinr_dB().has_value()) {
        results.cpu_sinr_stats.update(*cpu_result.csi.get_sinr_dB());
      }
      if (cpu_result.csi.get_total_evm().has_value()) {
        results.cpu_evm_stats.update(*cpu_result.csi.get_total_evm() * 100.0F); // Convert to percentage
      }
      if (cpu_result.csi.get_time_alignment().has_value()) {
        results.cpu_ta_stats_us.update(cpu_result.csi.get_time_alignment()->to_seconds() * 1e6F);
      }
      if (cpu_result.csi.get_epre_dB().has_value()) {
        results.cpu_epre_stats.update(*cpu_result.csi.get_epre_dB());
      }
      results.cpu_iterations_stats.update(cpu_result.data.ldpc_decoder_stats.get_mean());
      if (!cpu_result.data.tb_crc_ok) {
        ++results.cpu_crc_errors;
      }
      if (tx_data != cpu_rx_data) {
        ++results.cpu_data_errors;
      }

      // Collect GPU metrics
      if (gpu_result.csi.get_sinr_dB().has_value()) {
        results.gpu_sinr_stats.update(*gpu_result.csi.get_sinr_dB());
      }
      if (gpu_result.csi.get_total_evm().has_value()) {
        results.gpu_evm_stats.update(*gpu_result.csi.get_total_evm() * 100.0F); // Convert to percentage
      }
      if (gpu_result.csi.get_time_alignment().has_value()) {
        results.gpu_ta_stats_us.update(gpu_result.csi.get_time_alignment()->to_seconds() * 1e6F);
      }
      if (gpu_result.csi.get_epre_dB().has_value()) {
        results.gpu_epre_stats.update(*gpu_result.csi.get_epre_dB());
      }
      results.gpu_iterations_stats.update(gpu_result.data.ldpc_decoder_stats.get_mean());
      if (!gpu_result.data.tb_crc_ok) {
        ++results.gpu_crc_errors;
      }
      if (tx_data != gpu_rx_data) {
        ++results.gpu_data_errors;
      }

      // Check decode agreement
      if (cpu_result.data.tb_crc_ok == gpu_result.data.tb_crc_ok) {
        ++results.decode_agreement;
      }

      ++results.total_transmissions;

      // Increment slots
      ++pdsch_config.slot;
      ++pusch_config.slot;
    }

    worker_pool->stop();
    return results;
  }

  void print_header()
  {
    fmt::print("============================================================\n");
    fmt::print("PUSCH GPU vs CPU Metrics Comparison Test\n");
    fmt::print("============================================================\n");
    fmt::print("Configuration: {} layer(s), QPSK, {} repetitions per config\n", nof_layers, nof_repetitions);
    fmt::print("PRB allocations: ");
    for (unsigned prb : prb_allocations) {
      fmt::print("{} ", prb);
    }
    fmt::print("\nSINR range: ");
    for (float sinr : sinr_values_dB) {
      fmt::print("{:.0f} ", sinr);
    }
    fmt::print("dB\n");
    fmt::print("Rx ports: ");
    for (unsigned ports : rx_port_counts) {
      fmt::print("{} ", ports);
    }
    fmt::print("\n");
  }

  void print_results(const comparison_results& r)
  {
    if (r.total_transmissions == 0) {
      return;
    }

    float cpu_bler = 100.0F * static_cast<float>(r.cpu_crc_errors) / static_cast<float>(r.total_transmissions);
    float gpu_bler = 100.0F * static_cast<float>(r.gpu_crc_errors) / static_cast<float>(r.total_transmissions);

    float sinr_delta = r.gpu_sinr_stats.get_mean() - r.cpu_sinr_stats.get_mean();
    float bler_delta = gpu_bler - cpu_bler;
    float evm_delta  = r.gpu_evm_stats.get_mean() - r.cpu_evm_stats.get_mean();

    // Determine if within tolerance
    bool sinr_ok = std::abs(sinr_delta) < 1.0F;
    bool bler_ok = std::abs(bler_delta) < 5.0F; // Allow 5% BLER difference

    std::string status = (sinr_ok && bler_ok) ? "[OK]" : "[WARN]";

    fmt::print("  SINR={:>2.0f} dB:\n", r.target_sinr_dB);
    fmt::print("    CPU: SINR={:>5.1f}+/-{:.1f} dB, BLER={:>5.1f}%, EVM={:>5.1f}%, EPRE={:>5.1f} dB, Iter={:.1f}\n",
               r.cpu_sinr_stats.get_mean(),
               r.cpu_sinr_stats.get_std(),
               cpu_bler,
               r.cpu_evm_stats.get_mean(),
               r.cpu_epre_stats.get_nof_observations() > 0 ? r.cpu_epre_stats.get_mean() : -99.0F,
               r.cpu_iterations_stats.get_mean());
    fmt::print("    GPU: SINR={:>5.1f}+/-{:.1f} dB, BLER={:>5.1f}%, EVM={:>5.1f}%, EPRE={:>5.1f} dB, Iter={:.1f}\n",
               r.gpu_sinr_stats.get_mean(),
               r.gpu_sinr_stats.get_std(),
               gpu_bler,
               r.gpu_evm_stats.get_mean(),
               r.gpu_epre_stats.get_nof_observations() > 0 ? r.gpu_epre_stats.get_mean() : -99.0F,
               r.gpu_iterations_stats.get_mean());
    float epre_delta = (r.gpu_epre_stats.get_nof_observations() > 0 && r.cpu_epre_stats.get_nof_observations() > 0)
                         ? r.gpu_epre_stats.get_mean() - r.cpu_epre_stats.get_mean()
                         : 0.0F;
    fmt::print("    Delta: SINR={:+.1f} dB, BLER={:+.1f}%, EVM={:+.1f}%, EPRE={:+.1f} dB {}\n",
               sinr_delta, bler_delta, evm_delta, epre_delta, status);
  }

  void print_summary()
  {
    fmt::print("\n============================================================\n");
    fmt::print("Summary\n");
    fmt::print("============================================================\n");

    if (all_results.empty()) {
      fmt::print("No results collected.\n");
      return;
    }

    float max_sinr_delta  = 0.0F;
    float sum_sinr_delta  = 0.0F;
    float max_bler_delta  = 0.0F;
    unsigned total_decode_agreement = 0;
    unsigned total_transmissions    = 0;

    for (const auto& r : all_results) {
      if (r.total_transmissions == 0) {
        continue;
      }

      float sinr_delta = std::abs(r.gpu_sinr_stats.get_mean() - r.cpu_sinr_stats.get_mean());
      max_sinr_delta   = std::max(max_sinr_delta, sinr_delta);
      sum_sinr_delta += sinr_delta;

      float cpu_bler = 100.0F * static_cast<float>(r.cpu_crc_errors) / static_cast<float>(r.total_transmissions);
      float gpu_bler = 100.0F * static_cast<float>(r.gpu_crc_errors) / static_cast<float>(r.total_transmissions);
      max_bler_delta = std::max(max_bler_delta, std::abs(gpu_bler - cpu_bler));

      total_decode_agreement += r.decode_agreement;
      total_transmissions += r.total_transmissions;
    }

    float mean_sinr_delta = sum_sinr_delta / static_cast<float>(all_results.size());
    float decode_agreement_pct =
        100.0F * static_cast<float>(total_decode_agreement) / static_cast<float>(total_transmissions);

    fmt::print("  Max SINR delta: {:.1f} dB\n", max_sinr_delta);
    fmt::print("  Mean SINR delta: {:.1f} dB\n", mean_sinr_delta);
    fmt::print("  Max BLER delta: {:.1f}%\n", max_bler_delta);
    fmt::print("  Decode agreement: {:.1f}% (both paths agree on decode success/fail)\n", decode_agreement_pct);

    bool all_ok = (max_sinr_delta < 1.0F) && (max_bler_delta < 5.0F);
    fmt::print("  All metrics within tolerance: {}\n", all_ok ? "YES" : "NO");
  }

  std::vector<comparison_results> all_results;
};

} // namespace

static void usage(std::string_view prog)
{
  fmt::print("Usage: {} [-R X] [-P X,X,...] [-S X,X,...] [-N X,X,...] [-h]\n", prog);
  fmt::print("\t-R       Number of repetitions per configuration. [Default {}]\n", nof_repetitions);
  fmt::print("\t-P       Comma-separated list of PRB allocations. [Default 3,4,5,6]\n");
  fmt::print("\t-S       Comma-separated list of SINR values in dB. [Default 5,10,15,20,25,30]\n");
  fmt::print("\t-N       Comma-separated list of Rx port counts. [Default 1,2]\n");
  fmt::print("\t-h       Print this message.\n");
}

static std::vector<unsigned> parse_unsigned_list(const char* str)
{
  std::vector<unsigned> result;
  std::string           input(str);
  std::stringstream     ss(input);
  std::string           item;
  while (std::getline(ss, item, ',')) {
    result.push_back(std::stoul(item));
  }
  return result;
}

static std::vector<float> parse_float_list(const char* str)
{
  std::vector<float> result;
  std::string        input(str);
  std::stringstream  ss(input);
  std::string        item;
  while (std::getline(ss, item, ',')) {
    result.push_back(std::stof(item));
  }
  return result;
}

static void parse_args(int argc, char** argv)
{
  int opt = 0;
  while ((opt = getopt(argc, argv, "R:P:S:N:h")) != -1) {
    switch (opt) {
      case 'R':
        nof_repetitions = std::strtol(optarg, nullptr, 10);
        break;
      case 'P':
        prb_allocations = parse_unsigned_list(optarg);
        break;
      case 'S':
        sinr_values_dB = parse_float_list(optarg);
        break;
      case 'N':
        rx_port_counts = parse_unsigned_list(optarg);
        break;
      case 'h':
      default:
        usage(argv[0]);
        std::exit(-1);
    }
  }
}

int main(int argc, char** argv)
{
  parse_args(argc, argv);

  pusch_gpu_cpu_comparison_test test;
  test.run();

  return 0;
}
