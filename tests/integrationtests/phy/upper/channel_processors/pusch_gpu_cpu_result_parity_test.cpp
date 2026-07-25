// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief Result-level GPU versus CPU PUSCH comparison.
///
/// This test processes the same received signal through both CPU and GPU PUSCH processors,
/// comparing decoded results and metrics for a 64QAM allocation.

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
#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>

using namespace ocudu;

// Fixed test parameters for the default 64QAM parity scenario.
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
static constexpr unsigned           nof_layers                  = 1;
static constexpr pusch_mcs_table    mcs_table                   = pusch_mcs_table::qam64;

// Default scenario; command-line options can adjust the PRB count, SINR and MCS.
static unsigned nof_prb       = 5;
static float    target_sinr_dB = 15.0f;
static unsigned nof_rx_ports  = 1;
static sch_mcs_index mcs_index = 20;

namespace {

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

class result_parity_test
{
public:
  result_parity_test() { ocudulog::init(); }

  void run()
  {
    fmt::print("=== PUSCH GPU/CPU Result Parity Test ===\n");
    fmt::print("Configuration: 64QAM (MCS {}), {} PRB, {:.1f} dB SINR, {} Rx port(s)\n\n",
               mcs_index, nof_prb, target_sinr_dB, nof_rx_ports);

    run_single_test();
  }

private:
  std::shared_ptr<resource_grid_factory> create_grid_factory()
  {
    std::shared_ptr<channel_precoder_factory> precod_factory = create_channel_precoder_factory("auto");
    report_fatal_error_if_not(precod_factory, "Failed to create channel precoding factory.");
    return create_resource_grid_factory();
  }

  void run_single_test()
  {
    // Prepare executors
    unsigned max_nof_threads = std::min(8U, std::thread::hardware_concurrency());
    auto worker_pool =
        std::make_unique<task_worker_pool<concurrent_queue_policy::locking_mpmc>>("thread", max_nof_threads, 1024);
    auto executor = std::make_unique<task_worker_pool_executor<concurrent_queue_policy::locking_mpmc>>(*worker_pool);

    // Logging
    ocudulog::fetch_basic_logger("ALL").set_level(ocudulog::basic_levels::info);

    // Compute modulation and code scheme
    sch_mcs_description mcs_descr = pusch_mcs_get_config(mcs_table, mcs_index, false, false);

    fmt::print("MCS Description:\n");
    fmt::print("  Modulation: {}\n", to_string(mcs_descr.modulation));
    fmt::print("  Target code rate: {:.3f}\n", mcs_descr.get_normalised_target_code_rate());

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

    fmt::print("  Transport block size: {} bits ({} bytes)\n\n", tbs, tbs / 8);

    // Select LDPC base graph
    ldpc_base_graph_type ldpc_base_graph =
        get_ldpc_base_graph(mcs_descr.get_normalised_target_code_rate(), units::bits(tbs));

    // Generate frequency allocation
    rb_allocation freq_alloc = rb_allocation::make_type1(freq_allocation.start(), freq_allocation.length(), std::nullopt);

    // Create PDSCH processor factory (for TX)
    std::shared_ptr<pdsch_processor_factory> pdsch_proc_factory =
        create_sw_pdsch_processor_factory(*executor, max_nof_threads + 1, "", "auto");
    report_fatal_error_if_not(pdsch_proc_factory, "Failed to create PDSCH processor factory.");

    // Create CPU PUSCH processor
    std::shared_ptr<pusch_processor_factory> cpu_pusch_proc_factory =
        create_sw_pusch_processor_factory(*executor,
                                          max_nof_threads + 1,
                                          nof_ldpc_iterations,
                                          use_early_stop,
                                          "auto",
                                          port_channel_estimator_td_interpolation_strategy::average,
                                          channel_equalizer_algorithm_type::zf);
    report_fatal_error_if_not(cpu_pusch_proc_factory, "Failed to create CPU PUSCH processor factory.");

    // Create GPU PUSCH processor
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
      return;
    }

    // Create processors
    std::unique_ptr<pdsch_processor> pdsch_proc = pdsch_proc_factory->create();
    std::unique_ptr<pusch_processor> cpu_pusch_proc = cpu_pusch_proc_factory->create();
    std::unique_ptr<pusch_processor> gpu_pusch_proc = gpu_pusch_proc_factory->create();

    // Create resource grids
    std::shared_ptr<resource_grid_factory> grid_factory = create_grid_factory();
    std::unique_ptr<resource_grid> tx_grid = grid_factory->create(nof_layers, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);
    std::unique_ptr<resource_grid> rx_grid = grid_factory->create(nof_rx_ports, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);

    // Calculate number of codeblocks
    unsigned nof_codeblocks = compute_nof_codeblocks(units::bits(tbs), ldpc_base_graph);

    // Create RX buffer pools
    rx_buffer_pool_config rx_pool_config;
    rx_pool_config.max_codeblock_size = ldpc::MAX_CODEBLOCK_SIZE;
    rx_pool_config.nof_buffers        = 2;
    rx_pool_config.nof_codeblocks     = nof_codeblocks;
    rx_pool_config.expire_timeout_slots = 10;
    rx_pool_config.external_soft_bits = false;

    std::unique_ptr<rx_buffer_pool_controller> cpu_rx_buffer_pool = create_rx_buffer_pool(rx_pool_config);
    std::unique_ptr<rx_buffer_pool_controller> gpu_rx_buffer_pool = create_rx_buffer_pool(rx_pool_config);

    // Generate random transport block
    std::vector<uint8_t> tx_data(tbs / 8);
    std::random_device   rd;
    std::mt19937         rgen(rd());
    for (auto& byte : tx_data) {
      byte = static_cast<uint8_t>(rgen() & 0xff);
    }

    // RX data buffers
    std::vector<uint8_t> cpu_rx_data(tbs / 8);
    std::vector<uint8_t> gpu_rx_data(tbs / 8);

    // Build PDSCH PDU for transmission
    pdsch_processor::pdu_t pdsch_pdu;
    pdsch_pdu.context                     = std::nullopt;
    pdsch_pdu.slot                        = slot_point(to_numerology_value(scs), 0);
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
    fmt::print("Transmitting {} bytes via PDSCH...\n", tx_data.size());
    pdsch_processor_notifier_adaptor pdsch_notifier;
    pdsch_proc->process(tx_grid->get_writer(), pdsch_notifier, {shared_transport_block(tx_data)}, pdsch_pdu);
    pdsch_notifier.wait_for_completion();

    // Apply channel with AWGN
    fmt::print("Applying channel (AWGN, {:.1f} dB SINR)...\n", target_sinr_dB);
    channel_emulator emulator("single-tap",
                              "uniform-phase",
                              target_sinr_dB,
                              0.0f,  // No CFO
                              0,     // No corrupted REs
                              nof_layers,
                              nof_rx_ports,
                              nof_prb * NOF_SUBCARRIERS_PER_RB,
                              nof_ofdm_symbols,
                              max_nof_threads,
                              scs,
                              *executor);

    emulator.run(rx_grid->get_writer(), tx_grid->get_reader());

    // RX ports
    static_vector<uint8_t, MAX_PORTS> rx_ports(nof_rx_ports);
    std::iota(rx_ports.begin(), rx_ports.end(), 0U);

    // Build PUSCH PDU for reception
    pusch_processor::pdu_t pusch_pdu;
    pusch_pdu.context            = std::nullopt;
    pusch_pdu.slot               = slot_point(to_numerology_value(scs), 0);
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

    // Get RX buffers
    unique_rx_buffer cpu_buffer =
        cpu_rx_buffer_pool->get_pool().reserve(pusch_pdu.slot, trx_buffer_identifier(rnti, 0), nof_codeblocks, true);
    unique_rx_buffer gpu_buffer =
        gpu_rx_buffer_pool->get_pool().reserve(pusch_pdu.slot, trx_buffer_identifier(rnti, 1), nof_codeblocks, true);

    // Process with CPU
    fmt::print("\n--- CPU PUSCH Processing ---\n");
    pusch_processor_notifier_adaptor cpu_notifier;
    cpu_pusch_proc->process(cpu_rx_data, std::move(cpu_buffer), cpu_notifier, rx_grid->get_reader(), pusch_pdu);
    const pusch_processor_result_data& cpu_result = cpu_notifier.wait_for_completion();

    fmt::print("CPU Results:\n");
    fmt::print("  CRC: {}\n", cpu_result.data.tb_crc_ok ? "PASS" : "FAIL");
    if (cpu_result.csi.get_sinr_dB().has_value()) {
      fmt::print("  SINR: {:.2f} dB\n", cpu_result.csi.get_sinr_dB().value());
    }
    if (cpu_result.csi.get_total_evm().has_value()) {
      fmt::print("  EVM: {:.3f}%%\n", cpu_result.csi.get_total_evm().value() * 100.0F);
    }
    fmt::print("  LDPC iterations: {:.1f}\n", cpu_result.data.ldpc_decoder_stats.get_mean());

    // Process with GPU
    fmt::print("\n--- GPU PUSCH Processing ---\n");
    pusch_processor_notifier_adaptor gpu_notifier;
    gpu_pusch_proc->process(gpu_rx_data, std::move(gpu_buffer), gpu_notifier, rx_grid->get_reader(), pusch_pdu);
    const pusch_processor_result_data& gpu_result = gpu_notifier.wait_for_completion();

    fmt::print("GPU Results:\n");
    fmt::print("  CRC: {}\n", gpu_result.data.tb_crc_ok ? "PASS" : "FAIL");
    if (gpu_result.csi.get_sinr_dB().has_value()) {
      fmt::print("  SINR: {:.2f} dB\n", gpu_result.csi.get_sinr_dB().value());
    }
    if (gpu_result.csi.get_total_evm().has_value()) {
      fmt::print("  EVM: {:.3f}%%\n", gpu_result.csi.get_total_evm().value() * 100.0F);
    }
    fmt::print("  LDPC iterations: {:.1f}\n", gpu_result.data.ldpc_decoder_stats.get_mean());

    // Compare results
    fmt::print("\n=== Comparison ===\n");
    fmt::print("CRC Agreement: {}\n", cpu_result.data.tb_crc_ok == gpu_result.data.tb_crc_ok ? "YES" : "NO");
    if (cpu_result.csi.get_sinr_dB().has_value() && gpu_result.csi.get_sinr_dB().has_value()) {
      fmt::print("SINR Delta: {:.2f} dB\n", gpu_result.csi.get_sinr_dB().value() - cpu_result.csi.get_sinr_dB().value());
    }
    if (cpu_result.csi.get_total_evm().has_value() && gpu_result.csi.get_total_evm().has_value()) {
      fmt::print("EVM Delta: {:.3f}%%\n",
                 (gpu_result.csi.get_total_evm().value() - cpu_result.csi.get_total_evm().value()) * 100.0F);
    }
    fmt::print("LDPC Iterations Delta: {:.1f}\n",
               gpu_result.data.ldpc_decoder_stats.get_mean() - cpu_result.data.ldpc_decoder_stats.get_mean());

    if (cpu_result.data.tb_crc_ok != gpu_result.data.tb_crc_ok) {
      fmt::print("\n*** DIVERGENCE DETECTED ***\n");
      fmt::print("CPU and GPU produce different decode results for 64QAM!\n");
      fmt::print("Investigate equalized-symbol and LLR parity before release qualification.\n");
    } else {
      fmt::print("\nBoth paths decoded successfully (CRC OK).\n");
    }

    // Stop worker pool
    worker_pool->stop();
  }
};

} // namespace

int main(int argc, char** argv)
{
  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--prb" && i + 1 < argc) {
      nof_prb = std::stoi(argv[++i]);
    } else if (arg == "--sinr" && i + 1 < argc) {
      target_sinr_dB = std::stof(argv[++i]);
    } else if (arg == "--mcs" && i + 1 < argc) {
      mcs_index = static_cast<sch_mcs_index>(std::stoi(argv[++i]));
    } else if (arg == "--help") {
      fmt::print("Usage: {} [options]\n", argv[0]);
      fmt::print("Options:\n");
      fmt::print("  --prb N     Number of PRBs (default: 5)\n");
      fmt::print("  --sinr N    SINR in dB (default: 15.0)\n");
      fmt::print("  --mcs N     MCS index (default: 20 for 64QAM)\n");
      fmt::print("  --help      Show this help\n");
      return 0;
    }
  }

  result_parity_test test;
  test.run();
  return 0;
}
