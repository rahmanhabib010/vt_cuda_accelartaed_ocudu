// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief E2E PUSCH GPU Pipeline Correctness Sweep
///
/// This test validates the full GPU PUSCH pipeline (Rx Symbols → TB Bits)
/// across the complete parameter space:
/// - All MCS indices (0-27): QPSK, 16QAM, 64QAM, 256QAM
/// - TB sizes: tiny (5 PRB) to large (273 PRB)
/// - Base graphs: BG1 and BG2 (automatically selected)
/// - Lifting sizes: determined by TB size
/// - Filler bits: various configurations

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
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef ENABLE_CUDA
#include "lib/phy/upper/channel_processors/pusch/pusch_decoder_impl.h"
#include "lib/phy/upper/channel_processors/pusch/pusch_processor_impl.h"
#endif

using namespace ocudu;

static symbol_slot_mask make_dmrs_symbol_mask(const std::vector<unsigned>& symbols)
{
  symbol_slot_mask mask(MAX_NSYMB_PER_SLOT);
  for (unsigned symbol : symbols) {
    if (symbol >= MAX_NSYMB_PER_SLOT) {
      throw std::runtime_error(fmt::format("Invalid DMRS symbol index {}.", symbol));
    }
    mask.set(symbol);
  }
  if (mask.none() || mask.count() > 4) {
    throw std::runtime_error(fmt::format("Expected 1-4 DMRS symbols, got {}.", mask.count()));
  }
  return mask;
}

static symbol_slot_mask parse_dmrs_symbols(const std::string& value)
{
  std::vector<unsigned> symbols;
  std::stringstream     ss(value);
  std::string           token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      symbols.push_back(static_cast<unsigned>(std::stoul(token)));
    }
  }
  return make_dmrs_symbol_mask(symbols);
}

static std::string dmrs_symbols_to_string(const symbol_slot_mask& mask)
{
  std::string result;
  for (unsigned i = 0; i != mask.size(); ++i) {
    if (!mask.test(i)) {
      continue;
    }
    if (!result.empty()) {
      result += ",";
    }
    result += std::to_string(i);
  }
  return result;
}

static dmrs_config_type parse_dmrs_type(const std::string& value)
{
  if (value == "type1" || value == "TYPE1" || value == "1") {
    return dmrs_config_type::type1;
  }
  if (value == "type2" || value == "TYPE2" || value == "2") {
    return dmrs_config_type::type2;
  }
  throw std::runtime_error(fmt::format("Invalid DMRS type '{}'. Expected type1 or type2.", value));
}

// Test parameters
static constexpr subcarrier_spacing scs                         = subcarrier_spacing::kHz30;
static uint16_t                     rnti                        = 0x1234;
static constexpr unsigned           bwp_start_rb                = 0;
static constexpr unsigned           nof_ofdm_symbols            = 14;
static symbol_slot_mask             dmrs_symbols_mask            = make_dmrs_symbol_mask({2, 11});
static constexpr unsigned           nof_ldpc_iterations         = 10;
static dmrs_config_type                    dmrs                        = dmrs_config_type::type1;
static constexpr unsigned           nof_cdm_groups_without_data = 2;
static constexpr cyclic_prefix      cy_prefix                   = cyclic_prefix::NORMAL;
static constexpr unsigned           rv                          = 0;
static unsigned                     n_id                        = 0;
static unsigned                     scrambling_id               = 0;
static bool                         n_scid                      = false;
static constexpr bool               use_early_stop              = true;
static unsigned                     nof_layers                  = 1;
static unsigned                     nof_rx_ports                = 1;
static bool                         dmrs_matrix                 = false;
static channel_equalizer_algorithm_type equalizer_algorithm     = channel_equalizer_algorithm_type::mmse;

// Test sweep parameters - fewer iterations per config since we test many configs
static unsigned nof_iterations = 10;
static float    sinr_offset_dB = 0.0f;

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

struct test_results {
  unsigned cpu_pass = 0;
  unsigned cpu_fail = 0;
  unsigned gpu_pass = 0;
  unsigned gpu_fail = 0;
  unsigned disagreements = 0;
  latency_stats cpu_latency;
  latency_stats gpu_latency;

#ifdef ENABLE_CUDA
  // GPU decoder breakdown timing (CUDA event-based)
  latency_stats gpu_scramble;
  latency_stats gpu_rate_dematch;
  latency_stats gpu_ldpc_decode;
  latency_stats gpu_crc_check;
  latency_stats gpu_d2h_transfer;
  latency_stats gpu_extract_bits;
  latency_stats gpu_decoder_e2e;     // total decoder pipeline (CUDA events)
  latency_stats gpu_grid_staging;    // CPU grid staging memcpy
  latency_stats gpu_demod_sync;      // cudaEventSync for E2E demod kernel
#endif
};

test_results run_single_config(unsigned          nof_prb,
                               float             sinr_dB,
                               sch_mcs_index     mcs_idx,
                               pusch_mcs_table   mcs_table,
                               pdsch_processor&  pdsch_proc,
                               pusch_processor&  cpu_proc,
                               pusch_processor&  gpu_proc,
                               resource_grid&    tx_grid,
                               resource_grid&    rx_grid,
                               task_executor&    executor)
{
  test_results results;

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
  unsigned max_nof_threads = std::min(8U, std::thread::hardware_concurrency());
  channel_emulator emulator("single-tap", "uniform-phase", sinr_dB, 0.0f, 0,
                           nof_layers, nof_rx_ports, nof_prb * NOF_SUBCARRIERS_PER_RB,
                           nof_ofdm_symbols, max_nof_threads, scs, executor);

  std::random_device rd;
  std::mt19937 rgen(rd());

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
    pdsch_pdu.slot                        = slot_point(to_numerology_value(scs), iter);
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
    pdsch_proc.process(tx_grid.get_writer(), tx_notifier, {shared_transport_block(tx_data)}, pdsch_pdu);
    tx_notifier.wait_for_completion();

    // Apply channel
    emulator.run(rx_grid.get_writer(), tx_grid.get_reader());

    // PUSCH PDU
    static_vector<uint8_t, MAX_PORTS> rx_ports(nof_rx_ports);
    std::iota(rx_ports.begin(), rx_ports.end(), 0U);

    pusch_processor::pdu_t pusch_pdu;
    pusch_pdu.context            = std::nullopt;
    pusch_pdu.slot               = slot_point(to_numerology_value(scs), iter);
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
    cpu_proc.process(cpu_rx_data, std::move(cpu_buffer), cpu_notifier, rx_grid.get_reader(), pusch_pdu);
    const auto& cpu_result = cpu_notifier.wait_for_completion();
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    results.cpu_latency.add(cpu_us);

    // GPU decode
    unique_rx_buffer gpu_buffer =
        gpu_pool->get_pool().reserve(pusch_pdu.slot, trx_buffer_identifier(rnti, 1), nof_codeblocks, true);

    pusch_processor_notifier_adaptor gpu_notifier;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu_proc.process(gpu_rx_data, std::move(gpu_buffer), gpu_notifier, rx_grid.get_reader(), pusch_pdu);
    const auto& gpu_result = gpu_notifier.wait_for_completion();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    results.gpu_latency.add(gpu_us);

#ifdef ENABLE_CUDA
    // Capture GPU decoder breakdown timing (static_cast: RTTI disabled, we know concrete types).
    {
      auto* gpu_proc_impl = static_cast<pusch_processor_impl*>(&gpu_proc);
      auto* gpu_decoder = gpu_proc_impl->get_decoder();
      if (gpu_decoder) {
        auto* gpu_decoder_impl = static_cast<pusch_decoder_impl*>(gpu_decoder);
        auto* batch_decoder = gpu_decoder_impl->get_batch_gpu_decoder();
        if (batch_decoder) {
          const auto& timing = batch_decoder->get_last_timing_stats();
          if (timing.timing_enabled) {
            results.gpu_scramble.add(timing.deinterleave_us);
            results.gpu_rate_dematch.add(timing.rate_dematch_us);
            results.gpu_ldpc_decode.add(timing.ldpc_decode_us);
            results.gpu_crc_check.add(timing.crc_check_us);
            results.gpu_d2h_transfer.add(timing.d2h_transfer_us);
            results.gpu_extract_bits.add(timing.extract_bits_us);
            results.gpu_decoder_e2e.add(timing.total_e2e_us);
            results.gpu_grid_staging.add(timing.grid_staging_us);
            results.gpu_demod_sync.add(timing.demod_sync_us);
          }
        }
      }
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

    if (cpu_result.data.tb_crc_ok != gpu_result.data.tb_crc_ok) {
      results.disagreements++;
      fmt::print("DISAGREEMENT at iteration {}: CPU={} GPU={}\n",
                 iter, cpu_result.data.tb_crc_ok ? "PASS" : "FAIL",
                 gpu_result.data.tb_crc_ok ? "PASS" : "FAIL");
    }

    if ((iter + 1) % 10 == 0) {
      fmt::print(".");
      std::cout.flush();
    }
  }

  return results;
}

} // namespace

int main(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--rnti" && i + 1 < argc) {
      rnti = static_cast<uint16_t>(std::stoi(argv[++i]));
    } else if (arg == "--n-id" && i + 1 < argc) {
      n_id = std::stoi(argv[++i]);
    } else if (arg == "--scrambling-id" && i + 1 < argc) {
      scrambling_id = std::stoi(argv[++i]);
    } else if (arg == "--n-scid" && i + 1 < argc) {
      n_scid = (std::stoi(argv[++i]) != 0);
    } else if (arg == "--layers" && i + 1 < argc) {
      nof_layers = std::stoi(argv[++i]);
    } else if (arg == "--ports" && i + 1 < argc) {
      nof_rx_ports = std::stoi(argv[++i]);
    } else if (arg == "--dmrs-symbols" && i + 1 < argc) {
      dmrs_symbols_mask = parse_dmrs_symbols(argv[++i]);
    } else if (arg == "--dmrs-type" && i + 1 < argc) {
      dmrs = parse_dmrs_type(argv[++i]);
    } else if (arg == "--dmrs-matrix") {
      dmrs_matrix = true;
    } else if (arg == "--iterations" && i + 1 < argc) {
      nof_iterations = std::stoi(argv[++i]);
    } else if (arg == "--sinr-offset" && i + 1 < argc) {
      sinr_offset_dB = std::stof(argv[++i]);
    } else if (arg == "--equalizer" && i + 1 < argc) {
      std::string equalizer = argv[++i];
      if (equalizer == "zf" || equalizer == "ZF") {
        equalizer_algorithm = channel_equalizer_algorithm_type::zf;
      } else if (equalizer == "mmse" || equalizer == "MMSE") {
        equalizer_algorithm = channel_equalizer_algorithm_type::mmse;
      } else {
        fmt::print("Invalid --equalizer '{}'. Expected zf or mmse.\n", equalizer);
        return 1;
      }
    } else if (arg == "--help") {
      fmt::print("Usage: {} [options]\n", argv[0]);
      fmt::print("Options:\n");
      fmt::print("  --rnti N           RNTI value (default: 0x1234)\n");
      fmt::print("  --n-id N           Data scrambling n_ID (default: 0)\n");
      fmt::print("  --scrambling-id N  DMRS scrambling ID (default: 0)\n");
      fmt::print("  --n-scid N         DMRS n_SCID 0 or 1 (default: 0)\n");
      fmt::print("  --layers N         Number of TX layers (default: 1; 3-layer tests use at least 4 RX ports)\n");
      fmt::print("  --ports N          Number of RX ports (default: 1)\n");
      fmt::print("  --dmrs-symbols CSV DMRS OFDM symbols, comma-separated, 1-4 entries (default: 2,11)\n");
      fmt::print("  --dmrs-type STR    DMRS type1 or type2 (default: type1)\n");
      fmt::print("  --dmrs-matrix      Sweep representative 1-, 2-, 3-, and 4-symbol DMRS masks\n");
      fmt::print("  --iterations N     Iterations per config (default: 10)\n");
      fmt::print("  --sinr-offset F    Add F dB to each built-in correctness SNR point (default: 0)\n");
      fmt::print("  --equalizer STR    zf or mmse (default: mmse)\n");
      return 0;
    }
  }

  // Default ports to match layers if not explicitly set. The GPU-resident 3-layer path is validated with 4 or 8 ports.
  if ((nof_layers == 3) && (nof_rx_ports < 4)) {
    nof_rx_ports = 4;
  } else if (nof_rx_ports < nof_layers) {
    nof_rx_ports = nof_layers;
  }

  ocudulog::init();
  ocudulog::fetch_basic_logger("ALL").set_level(ocudulog::basic_levels::warning);
  ocudulog::fetch_basic_logger("PHY").set_level(ocudulog::basic_levels::warning);

  fmt::print("\n================================================================\n");
  fmt::print("  E2E GPU PUSCH Pipeline Correctness Sweep\n");
  fmt::print("  Testing: All MCS, TB sizes, base graphs, lifting sizes\n");
  fmt::print("  {}\n",
             fmt::format("Layers: {}  RX Ports: {}  DMRS: {} [{}]",
                         nof_layers,
                         nof_rx_ports,
                         to_string(dmrs),
                         dmrs_symbols_to_string(dmrs_symbols_mask)));
  fmt::print("  {}\n", fmt::format("Equalizer: {}", to_string(equalizer_algorithm)));
  fmt::print("================================================================\n\n");

  // Create shared resources once — processors pre-allocate to max size internally,
  // so the same instances handle any config without reallocation.
  unsigned max_nof_threads = std::min(8U, std::thread::hardware_concurrency());
  auto     worker_pool =
      std::make_unique<task_worker_pool<concurrent_queue_policy::locking_mpmc>>("thread", max_nof_threads, 1024);
  auto executor = std::make_unique<task_worker_pool_executor<concurrent_queue_policy::locking_mpmc>>(*worker_pool);

  auto grid_factory = create_resource_grid_factory();

  auto pdsch_factory = create_sw_pdsch_processor_factory(*executor, max_nof_threads + 1, "", "auto");

  auto cpu_factory = create_sw_pusch_processor_factory(*executor, max_nof_threads + 1,
                                                       nof_ldpc_iterations, use_early_stop, "auto",
                                                       port_channel_estimator_td_interpolation_strategy::average,
                                                       equalizer_algorithm);

  auto gpu_factory = create_sw_pusch_processor_factory(*executor, max_nof_threads + 1,
                                                       nof_ldpc_iterations, use_early_stop, "gpu",
                                                       port_channel_estimator_td_interpolation_strategy::average,
                                                       equalizer_algorithm);

  if (!gpu_factory) {
    fmt::print("GPU PUSCH not available, skipping test\n");
    worker_pool->stop();
    return 0;
  }

  auto pdsch_proc = pdsch_factory->create();
  auto cpu_proc   = cpu_factory->create();
  auto gpu_proc   = gpu_factory->create();

  auto tx_grid = grid_factory->create(nof_layers, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);
  auto rx_grid = grid_factory->create(nof_rx_ports, MAX_NSYMB_PER_SLOT, MAX_NOF_SUBCARRIERS);

  // Test configurations: (PRB, MCS table, SNR, description)
  // 64QAM configs use 20dB (comfortably above waterfall).
  // 256QAM configs use 28dB (waterfall is ~26dB for high-rate MCS).
  struct test_config {
    unsigned nof_prb;
    pusch_mcs_table mcs_table;
    float sinr_dB;
    std::vector<sch_mcs_index> mcs_indices;
    std::string description;
  };

  std::vector<test_config> configs = {
    // QAM64 table tests (20dB — comfortably above waterfall)
    {5,   pusch_mcs_table::qam64, 20.0f, {
      static_cast<sch_mcs_index>(0),   // QPSK low
      static_cast<sch_mcs_index>(4),   // QPSK mid
      static_cast<sch_mcs_index>(11),  // 16QAM low
      static_cast<sch_mcs_index>(16),  // 16QAM/64QAM boundary
      static_cast<sch_mcs_index>(22),  // 64QAM mid
      static_cast<sch_mcs_index>(27),  // 64QAM max
    }, "Tiny (BG2)"},

    {25,  pusch_mcs_table::qam64, 20.0f, {
      static_cast<sch_mcs_index>(0),
      static_cast<sch_mcs_index>(11),
      static_cast<sch_mcs_index>(19),
      static_cast<sch_mcs_index>(27),
    }, "Medium (BG1/BG2)"},

    {52,  pusch_mcs_table::qam64, 20.0f, {
      static_cast<sch_mcs_index>(4),
      static_cast<sch_mcs_index>(14),
      static_cast<sch_mcs_index>(22),
      static_cast<sch_mcs_index>(27),
    }, "Large (BG1)"},

    {106, pusch_mcs_table::qam64, 20.0f, {
      static_cast<sch_mcs_index>(8),
      static_cast<sch_mcs_index>(16),
      static_cast<sch_mcs_index>(25),
    }, "VeryLarge (multi-CB)"},

    {273, pusch_mcs_table::qam64, 20.0f, {
      static_cast<sch_mcs_index>(16),
      static_cast<sch_mcs_index>(25),
      static_cast<sch_mcs_index>(27),
    }, "FullBW (100MHz)"},

    // QAM256 table tests (28dB — above 256QAM waterfall ~26dB)
    {106, pusch_mcs_table::qam256, 28.0f, {
      static_cast<sch_mcs_index>(16),
      static_cast<sch_mcs_index>(23),
      static_cast<sch_mcs_index>(27),
    }, "VeryLarge-256QAM"},

    {273, pusch_mcs_table::qam256, 28.0f, {
      static_cast<sch_mcs_index>(16),
      static_cast<sch_mcs_index>(23),
      static_cast<sch_mcs_index>(27),
    }, "FullBW-256QAM (100MHz)"},
  };

  struct dmrs_profile {
    std::string      label;
    symbol_slot_mask mask;
  };

  std::vector<dmrs_profile> dmrs_profiles = {{"configured", dmrs_symbols_mask}};
  if (dmrs_matrix) {
    dmrs_profiles = {
        {"1dmrs", make_dmrs_symbol_mask({2})},
        {"2dmrs", make_dmrs_symbol_mask({2, 11})},
        {"3dmrs", make_dmrs_symbol_mask({2, 7, 11})},
        {"4dmrs", make_dmrs_symbol_mask({2, 5, 8, 11})},
    };
  }

  unsigned total_tests = 0;
  unsigned passed_tests = 0;
  unsigned failed_tests = 0;

  // Collect per-config latency for summary table
  struct latency_row {
    std::string label;
    double cpu_mean_us;
    double gpu_mean_us;
    double speedup;
#ifdef ENABLE_CUDA
    double scramble_us;
    double rate_dematch_us;
    double ldpc_decode_us;
    double crc_check_us;
    double d2h_us;
    double extract_us;
    double decoder_e2e_us;
    double grid_staging_us;
    double demod_sync_us;
#endif
  };
  std::vector<latency_row> latency_table;
  latency_stats overall_cpu, overall_gpu;

  for (const auto& profile : dmrs_profiles) {
    dmrs_symbols_mask = profile.mask;
    fmt::print("\n=== DMRS profile {}: symbols [{}] ===\n", profile.label, dmrs_symbols_to_string(dmrs_symbols_mask));

    for (const auto& config : configs) {
      float test_sinr_dB = config.sinr_dB + sinr_offset_dB;
      fmt::print("\n--- {} PRB: {} ({:.0f}dB) ---\n", config.nof_prb, config.description, test_sinr_dB);

      for (auto mcs_idx : config.mcs_indices) {
        total_tests++;
        fmt::print("  MCS{:2d}... ", mcs_idx.value());
        std::cout.flush();

        auto result = run_single_config(config.nof_prb, test_sinr_dB, mcs_idx, config.mcs_table,
                                        *pdsch_proc, *cpu_proc, *gpu_proc, *tx_grid, *rx_grid, *executor);

        bool test_passed = (result.cpu_pass == nof_iterations &&
                           result.gpu_pass == nof_iterations &&
                           result.disagreements == 0);

        if (test_passed) {
          passed_tests++;
          double speedup = (result.gpu_latency.mean() > 0) ? result.cpu_latency.mean() / result.gpu_latency.mean() : 0;
          fmt::print("PASS  CPU:{:.0f}us  GPU:{:.0f}us  {:.2f}x\n",
                    result.cpu_latency.mean(), result.gpu_latency.mean(), speedup);
          latency_row row;
          row.label = fmt::format("{} {}PRB MCS{}", profile.label, config.nof_prb, mcs_idx.value());
          row.cpu_mean_us = result.cpu_latency.mean();
          row.gpu_mean_us = result.gpu_latency.mean();
          row.speedup = speedup;
#ifdef ENABLE_CUDA
          row.scramble_us = result.gpu_scramble.mean();
          row.rate_dematch_us = result.gpu_rate_dematch.mean();
          row.ldpc_decode_us = result.gpu_ldpc_decode.mean();
          row.crc_check_us = result.gpu_crc_check.mean();
          row.d2h_us = result.gpu_d2h_transfer.mean();
          row.extract_us = result.gpu_extract_bits.mean();
          row.decoder_e2e_us = result.gpu_decoder_e2e.mean();
          row.grid_staging_us = result.gpu_grid_staging.mean();
          row.demod_sync_us = result.gpu_demod_sync.mean();
#endif
          latency_table.push_back(row);
        } else {
          failed_tests++;
          fmt::print("FAIL (CPU:{}/{} GPU:{}/{} Disagree:{})\n",
                    result.cpu_pass, nof_iterations,
                    result.gpu_pass, nof_iterations,
                    result.disagreements);
        }

        // Accumulate overall latency stats
        for (unsigned i = 0; i < result.cpu_latency.count; i++) {
          overall_cpu.add(result.cpu_latency.mean());
        }
        for (unsigned i = 0; i < result.gpu_latency.count; i++) {
          overall_gpu.add(result.gpu_latency.mean());
        }
      }
    }
  }

  fmt::print("\n\n=== FINAL RESULTS ===\n\n");

  fmt::print("Total Tests:  {}\n", total_tests);
  fmt::print("Passed:       {} ({:.1f}%)\n", passed_tests, 100.0 * passed_tests / total_tests);
  fmt::print("Failed:       {} ({:.1f}%)\n", failed_tests, 100.0 * failed_tests / total_tests);

  // Latency summary table
  if (!latency_table.empty()) {
    fmt::print("\n=== Latency Summary ===\n");
    fmt::print("{:<22s}  {:>10s}  {:>10s}  {:>8s}\n", "Config", "CPU (us)", "GPU (us)", "Speedup");
    fmt::print("{:-<22s}  {:-<10s}  {:-<10s}  {:-<8s}\n", "", "", "", "");
    for (const auto& row : latency_table) {
      fmt::print("{:<22s}  {:>10.0f}  {:>10.0f}  {:>7.2f}x\n",
                row.label, row.cpu_mean_us, row.gpu_mean_us, row.speedup);
    }
  }

#ifdef ENABLE_CUDA
  // GPU pipeline breakdown table
  if (!latency_table.empty() && latency_table[0].decoder_e2e_us > 0) {
    fmt::print("\n=== GPU Pipeline Breakdown (us, mean over {} iters) ===\n", nof_iterations);
    fmt::print("{:<18s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}\n",
              "Config", "Total", "DecE2E", "Frnt%", "Scramb", "Dmatch", "LDPC", "CRC", "D2H", "Extrc", "Front");
    fmt::print("{:-<18s}  {:-<7s}  {:-<7s}  {:-<7s}  {:-<7s}  {:-<7s}  {:-<7s}  {:-<7s}  {:-<7s}  {:-<7s}  {:-<7s}\n",
              "", "", "", "", "", "", "", "", "", "", "");
    for (const auto& row : latency_table) {
      double frontend_us = row.gpu_mean_us - row.decoder_e2e_us - row.extract_us;
      if (frontend_us < 0) frontend_us = 0;
      double frontend_pct = (row.gpu_mean_us > 0) ? 100.0 * frontend_us / row.gpu_mean_us : 0;
      fmt::print("{:<18s}  {:>7.0f}  {:>7.0f}  {:>6.1f}%  {:>7.1f}  {:>7.1f}  {:>7.1f}  {:>7.1f}  {:>7.1f}  {:>7.1f}  {:>7.0f}\n",
                row.label, row.gpu_mean_us, row.decoder_e2e_us, frontend_pct,
                row.scramble_us, row.rate_dematch_us, row.ldpc_decode_us,
                row.crc_check_us, row.d2h_us, row.extract_us, frontend_us);
    }
    fmt::print("\nFront = Total - DecE2E - Extract (chanest + eq + demod + grid staging + sync)\n");
  }
#endif

  worker_pool->stop();

  if (failed_tests == 0) {
    fmt::print("\nALL TESTS PASSED\n");
    return 0;
  } else {
    fmt::print("\nCORRECTNESS ISSUES DETECTED\n");
    return 1;
  }
}
