// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI

/// \file
/// \brief E2E PUSCH Sensitivity Sweep — finds target BLER threshold SNR for CPU vs GPU pipelines.
///
/// Two-phase adaptive sweep:
///   Phase 1 (coarse): Few frames per SNR point to quickly locate the waterfall region.
///   Phase 2 (refine): Top-up with full frames only at SNR points near the BLER crossing.
/// All config-specific state (TBS, pools, etc.) is precomputed once per PRB/MCS — no
/// reallocation between SNR points or between phases.

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
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

using namespace ocudu;

/// Parse a comma-separated list of unsigned integers, e.g. "25,52,106,273".
static std::vector<unsigned> parse_csv_unsigned(const std::string& s)
{
  std::vector<unsigned> result;
  std::istringstream    ss(s);
  std::string           token;
  while (std::getline(ss, token, ',')) {
    result.push_back(static_cast<unsigned>(std::stoi(token)));
  }
  return result;
}

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
  return make_dmrs_symbol_mask(parse_csv_unsigned(value));
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

// Scrambling / MIMO parameters (overridable via CLI).
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
static channel_equalizer_algorithm_type equalizer_algorithm     = channel_equalizer_algorithm_type::mmse;

// Sweep parameters (overridable via CLI).
static float    snr_start   = -5.0f;
static float    snr_stop    = 25.0f;
static float    snr_step    = 0.1f;
static unsigned nof_frames  = 100;
static float    target_bler = 0.1f;
static unsigned rng_seed    = 12345;
static std::string gpu_pxsch_type = "gpu";

// Channel parameters (overridable via CLI).
static std::string channel_profile = "single-tap";
static std::string fading_dist     = "uniform-phase";

// CLI config lists: comma-separated PRB and MCS values.
static std::vector<unsigned> cli_prb_list;
static std::vector<unsigned> cli_mcs_list;
static pusch_mcs_table       cli_mcs_table = pusch_mcs_table::qam64;

namespace {

// ---------------------------------------------------------------------------
// Notifier adaptors (identical to correctness sweep).
// ---------------------------------------------------------------------------
class pusch_processor_notifier_adaptor : public pusch_processor_result_notifier
{
public:
  void on_uci(const pusch_processor_result_control& /*uci_*/) override {}

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

  bool                        completed = false;
  pusch_processor_result_data sch;
  std::mutex                  mutex;
  std::condition_variable     cvar;
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

// ---------------------------------------------------------------------------
// Per-config precomputed state — created once, reused across all SNR points.
// ---------------------------------------------------------------------------
struct config_state {
  sch_mcs_description  mcs_descr;
  unsigned             tbs;
  ldpc_base_graph_type ldpc_base_graph;
  rb_allocation        freq_alloc;
  unsigned             nof_codeblocks;
  unsigned             nof_prb;

  std::unique_ptr<rx_buffer_pool_controller> cpu_pool;
  std::unique_ptr<rx_buffer_pool_controller> gpu_pool;

  static config_state create(unsigned nof_prb, sch_mcs_index mcs_idx, pusch_mcs_table mcs_table)
  {
    config_state ctx;
    ctx.nof_prb   = nof_prb;
    ctx.mcs_descr = pusch_mcs_get_config(mcs_table, mcs_idx, false, false);

    prb_interval freq_allocation = {bwp_start_rb, bwp_start_rb + nof_prb};

    tbs_calculator_configuration tbs_config = {};
    tbs_config.mcs_descr                    = ctx.mcs_descr;
    tbs_config.n_prb                        = freq_allocation.length();
    tbs_config.nof_layers                   = nof_layers;
    tbs_config.nof_symb_sh                  = nof_ofdm_symbols;
    tbs_config.nof_dmrs_prb = get_nof_re_per_prb(dmrs) * dmrs_symbols_mask.count() * nof_cdm_groups_without_data;
    ctx.tbs                 = tbs_calculator_calculate(tbs_config).to_bits().value();

    ctx.ldpc_base_graph =
        get_ldpc_base_graph(ctx.mcs_descr.get_normalised_target_code_rate(), units::bits(ctx.tbs));

    ctx.freq_alloc =
        rb_allocation::make_type1(freq_allocation.start(), freq_allocation.length(), std::nullopt);

    ctx.nof_codeblocks = compute_nof_codeblocks(units::bits(ctx.tbs), ctx.ldpc_base_graph);

    rx_buffer_pool_config pool_config;
    pool_config.max_codeblock_size   = ldpc::MAX_CODEBLOCK_SIZE;
    pool_config.nof_buffers          = 2;
    pool_config.nof_codeblocks       = ctx.nof_codeblocks;
    pool_config.expire_timeout_slots = 10;
    pool_config.external_soft_bits   = false;

    ctx.cpu_pool = create_rx_buffer_pool(pool_config);
    ctx.gpu_pool = create_rx_buffer_pool(pool_config);

    return ctx;
  }
};

// ---------------------------------------------------------------------------
// Per-SNR BLER accumulator — supports incremental frame addition.
// ---------------------------------------------------------------------------
struct snr_bler_point {
  float    snr_dB   = 0;
  unsigned cpu_fails = 0;
  unsigned gpu_fails = 0;
  unsigned frames    = 0;

  float cpu_bler() const { return frames > 0 ? static_cast<float>(cpu_fails) / frames : 1.0f; }
  float gpu_bler() const { return frames > 0 ? static_cast<float>(gpu_fails) / frames : 1.0f; }
};

/// Run additional frames at a single SNR point, accumulating into \c point.
/// No allocation — uses pre-created config_state pools.
/// CPU and GPU decode run concurrently on separate threads.
void run_frames(snr_bler_point&   point,
                unsigned          num_frames,
                unsigned&         slot_counter,
                config_state&     ctx,
                pdsch_processor&  pdsch_proc,
                pusch_processor&  cpu_proc,
                pusch_processor&  gpu_proc,
                resource_grid&    tx_grid,
                resource_grid&    rx_grid,
                task_executor&    executor,
                std::mt19937&     rgen)
{
  unsigned max_nof_threads = std::min(8U, std::thread::hardware_concurrency());
  channel_emulator emulator(channel_profile, fading_dist, point.snr_dB, 0.0f, 0, nof_layers, nof_rx_ports,
                            ctx.nof_prb * NOF_SUBCARRIERS_PER_RB, nof_ofdm_symbols, max_nof_threads, scs, executor);

  std::vector<uint8_t> tx_data(ctx.tbs / 8);
  std::vector<uint8_t> cpu_rx_data(ctx.tbs / 8);
  std::vector<uint8_t> gpu_rx_data(ctx.tbs / 8);

  // Build PDU templates once — only slot changes per frame.
  pdsch_processor::pdu_t pdsch_pdu;
  pdsch_pdu.context                     = std::nullopt;
  pdsch_pdu.rnti                        = rnti;
  pdsch_pdu.bwp_size_rb                 = ctx.nof_prb;
  pdsch_pdu.bwp_start_rb                = bwp_start_rb;
  pdsch_pdu.cp                          = cy_prefix;
  pdsch_pdu.n_id                        = n_id;
  pdsch_pdu.ref_point                   = pdsch_processor::pdu_t::PRB0;
  pdsch_pdu.dmrs_symbol_mask            = dmrs_symbols_mask;
  pdsch_pdu.dmrs                        = dmrs;
  pdsch_pdu.scrambling_id               = scrambling_id;
  pdsch_pdu.n_scid                      = n_scid;
  pdsch_pdu.nof_cdm_groups_without_data = nof_cdm_groups_without_data;
  pdsch_pdu.freq_alloc                  = ctx.freq_alloc;
  pdsch_pdu.start_symbol_index          = 0;
  pdsch_pdu.nof_symbols                 = nof_ofdm_symbols;
  pdsch_pdu.ldpc_base_graph             = ctx.ldpc_base_graph;
  pdsch_pdu.tbs_lbrm                    = tbs_lbrm_default;
  pdsch_pdu.reserved                    = {};
  pdsch_pdu.ratio_pdsch_data_to_sss_dB  = 0.0F;
  pdsch_pdu.ratio_pdsch_dmrs_to_sss_dB  = get_sch_to_dmrs_ratio_dB(nof_cdm_groups_without_data);
  pdsch_pdu.precoding                   = precoding_configuration::make_wideband(make_identity(nof_layers));
  pdsch_pdu.codewords.emplace_back(pdsch_processor::codeword_description{ctx.mcs_descr.modulation, rv});

  static_vector<uint8_t, MAX_PORTS> rx_ports(nof_rx_ports);
  std::iota(rx_ports.begin(), rx_ports.end(), 0U);

  pusch_processor::pdu_t pusch_pdu;
  pusch_pdu.context            = std::nullopt;
  pusch_pdu.rnti               = rnti;
  pusch_pdu.bwp_size_rb        = ctx.nof_prb;
  pusch_pdu.bwp_start_rb       = bwp_start_rb;
  pusch_pdu.cp                 = cy_prefix;
  pusch_pdu.mcs_descr          = ctx.mcs_descr;
  pusch_pdu.codeword           = {rv, ctx.ldpc_base_graph, true};
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
  pusch_pdu.freq_alloc         = ctx.freq_alloc;
  pusch_pdu.start_symbol_index = 0;
  pusch_pdu.nof_symbols        = nof_ofdm_symbols;
  pusch_pdu.dc_position        = std::nullopt;

  for (unsigned f = 0; f < num_frames; ++f) {
    unsigned slot_idx = slot_counter++;

    // Random payload.
    for (auto& byte : tx_data) {
      byte = static_cast<uint8_t>(rgen() & 0xff);
    }

    // Update slot in PDU templates.
    slot_point slot(to_numerology_value(scs), slot_idx);
    pdsch_pdu.slot = slot;
    pusch_pdu.slot = slot;

    // --- TX (PDSCH) ---
    pdsch_processor_notifier_adaptor tx_notifier;
    pdsch_proc.process(tx_grid.get_writer(), tx_notifier, {shared_transport_block(tx_data)}, pdsch_pdu);
    tx_notifier.wait_for_completion();

    // --- Channel ---
    emulator.run(rx_grid.get_writer(), tx_grid.get_reader());

    // --- RX: CPU and GPU concurrently ---
    unique_rx_buffer cpu_buffer =
        ctx.cpu_pool->get_pool().reserve(slot, trx_buffer_identifier(rnti, 0), ctx.nof_codeblocks, true);
    unique_rx_buffer gpu_buffer =
        ctx.gpu_pool->get_pool().reserve(slot, trx_buffer_identifier(rnti, 1), ctx.nof_codeblocks, true);

    pusch_processor_notifier_adaptor cpu_notifier;
    pusch_processor_notifier_adaptor gpu_notifier;

    // Launch CPU on a separate thread so it overlaps with GPU.
    bool cpu_crc_ok = false;
    std::thread cpu_thread([&]() {
      cpu_proc.process(cpu_rx_data, std::move(cpu_buffer), cpu_notifier, rx_grid.get_reader(), pusch_pdu);
      const auto& result = cpu_notifier.wait_for_completion();
      cpu_crc_ok = result.data.tb_crc_ok;
    });

    // GPU on main thread.
    gpu_proc.process(gpu_rx_data, std::move(gpu_buffer), gpu_notifier, rx_grid.get_reader(), pusch_pdu);
    const auto& gpu_result = gpu_notifier.wait_for_completion();

    cpu_thread.join();

    if (!cpu_crc_ok) {
      point.cpu_fails++;
    }
    if (!gpu_result.data.tb_crc_ok) {
      point.gpu_fails++;
    }

    point.frames++;
  }
}

/// Interpolate the SNR at which BLER crosses \c target in the log domain.
/// Returns NaN if the crossing is not found.
float interpolate_bler_crossing(const std::vector<snr_bler_point>& curve,
                                float                              target,
                                bool                               use_cpu)
{
  for (unsigned i = 0; i + 1 < curve.size(); ++i) {
    float bler_a = use_cpu ? curve[i].cpu_bler() : curve[i].gpu_bler();
    float bler_b = use_cpu ? curve[i + 1].cpu_bler() : curve[i + 1].gpu_bler();

    if (bler_a >= target && bler_b < target) {
      float la = std::log10(std::max(bler_a, 1e-6f));
      float lb = std::log10(std::max(bler_b, 1e-6f));
      float lt = std::log10(target);
      float frac = (lt - la) / (lb - la);
      return curve[i].snr_dB + frac * (curve[i + 1].snr_dB - curve[i].snr_dB);
    }
  }
  return std::nanf("");
}

const char* modulation_name(modulation_scheme mod)
{
  switch (mod) {
    case modulation_scheme::QPSK:
      return "QPSK";
    case modulation_scheme::QAM16:
      return "16QAM";
    case modulation_scheme::QAM64:
      return "64QAM";
    case modulation_scheme::QAM256:
      return "256QAM";
    default:
      return "?";
  }
}

struct sweep_config {
  unsigned        nof_prb;
  sch_mcs_index   mcs_index;
  pusch_mcs_table mcs_table;
};


} // namespace

int main(int argc, char** argv)
{
  // --- CLI parsing ---
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
    } else if (arg == "--nof_prb" && i + 1 < argc) {
      cli_prb_list = parse_csv_unsigned(argv[++i]);
    } else if (arg == "--mcs_index" && i + 1 < argc) {
      cli_mcs_list = parse_csv_unsigned(argv[++i]);
    } else if (arg == "--mcs_table" && i + 1 < argc) {
      int tbl = std::stoi(argv[++i]);
      if (tbl == 2) {
        cli_mcs_table = pusch_mcs_table::qam256;
      } else if (tbl == 3) {
        cli_mcs_table = pusch_mcs_table::qam64LowSe;
      } else {
        cli_mcs_table = pusch_mcs_table::qam64;
      }
    } else if (arg == "--channel" && i + 1 < argc) {
      channel_profile = argv[++i];
    } else if (arg == "--fading" && i + 1 < argc) {
      fading_dist = argv[++i];
    } else if (arg == "--snr_start" && i + 1 < argc) {
      snr_start = std::stof(argv[++i]);
    } else if (arg == "--snr_stop" && i + 1 < argc) {
      snr_stop = std::stof(argv[++i]);
    } else if (arg == "--snr_step" && i + 1 < argc) {
      snr_step = std::stof(argv[++i]);
    } else if (arg == "--nof_frames" && i + 1 < argc) {
      nof_frames = std::stoi(argv[++i]);
    } else if (arg == "--target_bler" && i + 1 < argc) {
      target_bler = std::stof(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      rng_seed = std::stoi(argv[++i]);
    } else if (arg == "--gpu-type" && i + 1 < argc) {
      gpu_pxsch_type = argv[++i];
    } else if (arg == "--help") {
      fmt::print("Usage: {} [options]\n\n", argv[0]);
      fmt::print("Sensitivity sweep options:\n");
      fmt::print("  --nof_prb N[,N...] PRB counts, comma-separated (e.g. 25,106,273)\n");
      fmt::print("  --mcs_index N[,N.] MCS indices, comma-separated (e.g. 11,16,27)\n");
      fmt::print("  --mcs_table N      MCS table: 1=qam64, 2=qam256, 3=qam64LowSE (default: 1)\n");
      fmt::print("  --channel STR      Delay profile: single-tap, TDLA, TDLB, TDLC (default: single-tap)\n");
      fmt::print("  --fading STR       Fading distribution (default: uniform-phase)\n");
      fmt::print("  --snr_start F      Start SNR in dB (default: -5.0)\n");
      fmt::print("  --snr_stop F       Stop SNR in dB (default: 25.0)\n");
      fmt::print("  --snr_step F       SNR step in dB (default: 1.0)\n");
      fmt::print("  --nof_frames N     Frames per SNR point in waterfall region (default: 100)\n");
      fmt::print("  --target_bler F    Target BLER threshold (default: 0.1)\n");
      fmt::print("  --seed N           RNG seed (default: 12345)\n");
      fmt::print("  --gpu-type STR     GPU comparison path: gpu, gpu-demod, gpu-decoder (default: gpu)\n");
      fmt::print("\nScrambling / MIMO:\n");
      fmt::print("  --rnti N           RNTI value (default: 0x1234)\n");
      fmt::print("  --n-id N           Data scrambling n_ID (default: 0)\n");
      fmt::print("  --scrambling-id N  DMRS scrambling ID (default: 0)\n");
      fmt::print("  --n-scid N         DMRS n_SCID 0 or 1 (default: 0)\n");
      fmt::print("  --layers N         Number of TX layers (default: 1; 3-layer tests use at least 4 RX ports)\n");
      fmt::print("  --ports N          Number of RX ports (default: 1)\n");
      fmt::print("  --dmrs-symbols CSV DMRS OFDM symbols, comma-separated, 1-4 entries (default: 2,11)\n");
      fmt::print("  --dmrs-type STR    DMRS type1 or type2 (default: type1)\n");
      fmt::print("  --equalizer STR    zf or mmse (default: mmse)\n");
      fmt::print("\nThe sweep uses a two-phase approach:\n");
      fmt::print("  Phase 1: coarse scan (10 frames/point) to locate the waterfall region.\n");
      fmt::print("  Phase 2: top-up to --nof_frames only at SNR points near the BLER crossing.\n");
      return 0;
    }
  }

  if ((nof_layers == 3) && (nof_rx_ports < 4)) {
    nof_rx_ports = 4;
  } else if (nof_rx_ports < nof_layers) {
    nof_rx_ports = nof_layers;
  }

  // Build config list: cross-product of CLI PRB x MCS lists, or built-in defaults.
  std::vector<sweep_config> configs;
  if (!cli_prb_list.empty() && !cli_mcs_list.empty()) {
    for (unsigned prb : cli_prb_list) {
      for (unsigned mcs : cli_mcs_list) {
        configs.push_back({prb, static_cast<sch_mcs_index>(mcs), cli_mcs_table});
      }
    }
  } else if (!cli_prb_list.empty()) {
    // PRBs specified, use default MCS set.
    for (unsigned prb : cli_prb_list) {
      for (unsigned mcs : {11U, 16U, 27U}) {
        configs.push_back({prb, static_cast<sch_mcs_index>(mcs), cli_mcs_table});
      }
    }
  } else if (!cli_mcs_list.empty()) {
    // MCS specified, use default PRB set.
    for (unsigned prb : {25U, 52U, 106U, 273U}) {
      for (unsigned mcs : cli_mcs_list) {
        configs.push_back({prb, static_cast<sch_mcs_index>(mcs), cli_mcs_table});
      }
    }
  } else {
    // Built-in defaults.
    configs = {
        {25,  static_cast<sch_mcs_index>(11), pusch_mcs_table::qam64},
        {25,  static_cast<sch_mcs_index>(27), pusch_mcs_table::qam64},
        {52,  static_cast<sch_mcs_index>(14), pusch_mcs_table::qam64},
        {52,  static_cast<sch_mcs_index>(27), pusch_mcs_table::qam64},
        {106, static_cast<sch_mcs_index>(16), pusch_mcs_table::qam64},
        {106, static_cast<sch_mcs_index>(27), pusch_mcs_table::qam64},
        {273, static_cast<sch_mcs_index>(16), pusch_mcs_table::qam64},
        {273, static_cast<sch_mcs_index>(27), pusch_mcs_table::qam64},
    };
  }

  // Coarse phase: wide steps (1dB min) with few frames to locate waterfall quickly.
  // Fine phase: fill in at snr_step resolution with full frames in the waterfall region.
  unsigned coarse_frames = std::min(nof_frames, 10U);
  float    coarse_step   = std::max(snr_step, 1.0f);

  ocudulog::init();
  ocudulog::fetch_basic_logger("ALL").set_level(ocudulog::basic_levels::warning);
  ocudulog::fetch_basic_logger("PHY").set_level(ocudulog::basic_levels::warning);

  fmt::print("\n");
  fmt::print("================================================================\n");
  fmt::print("  PUSCH E2E Sensitivity Sweep -- CPU vs GPU\n");
  fmt::print("  Channel: {}  Fading: {}\n", channel_profile, fading_dist);
  fmt::print("  SNR: {:.1f} to {:.1f} dB, fine step {:.1f} dB\n", snr_start, snr_stop, snr_step);
  fmt::print("  Coarse: {} frames @ {:.1f}dB step  Fine: {} frames @ {:.1f}dB step\n",
             coarse_frames, coarse_step, nof_frames, snr_step);
  fmt::print("  Target BLER: {:.0f}%  Seed: {}\n", target_bler * 100, rng_seed);
  fmt::print("  Layers: {}  RX Ports: {}  DMRS: {} [{}]\n",
             nof_layers,
             nof_rx_ports,
             to_string(dmrs),
             dmrs_symbols_to_string(dmrs_symbols_mask));
  fmt::print("  Equalizer: {}\n", to_string(equalizer_algorithm));
  fmt::print("  GPU comparison path: {}\n", gpu_pxsch_type);
  fmt::print("================================================================\n\n");

  // --- Create shared resources once (reused across all configs). ---
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

  auto gpu_factory = create_sw_pusch_processor_factory(*executor,
                                                       max_nof_threads + 1,
                                                       nof_ldpc_iterations,
                                                       use_early_stop,
                                                       gpu_pxsch_type,
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

  // Summary table rows.
  struct summary_row {
    std::string config_label;
    float       cpu_threshold;
    float       gpu_threshold;
  };
  std::vector<summary_row> summary;

  std::mt19937 master_rng(rng_seed);

  // --- Sweep each config ---
  for (const auto& cfg : configs) {
    // Precompute config state (TBS, pools, etc.) — one allocation per config.
    auto ctx = config_state::create(cfg.nof_prb, cfg.mcs_index, cfg.mcs_table);

    std::string config_label =
        fmt::format("{}PRB MCS{} {}", cfg.nof_prb, cfg.mcs_index.value(), modulation_name(ctx.mcs_descr.modulation));

    fmt::print("\n=== {} ===\n", config_label);

    // Per-config RNG for reproducibility.
    std::mt19937 config_rng(master_rng());

    // Monotonically increasing slot counter (shared across all SNR points + phases).
    unsigned slot_counter = 0;

    // ---------------------------------------------------------------
    // Phase 1: Coarse scan — wide steps, few frames, to locate waterfall.
    // ---------------------------------------------------------------
    std::vector<snr_bler_point> coarse_curve;
    unsigned                    consecutive_zero = 0;

    // Probe the starting SNR — if it's not 100% BLER, step backwards to find it.
    float effective_start = snr_start;
    {
      snr_bler_point probe{};
      probe.snr_dB = effective_start;
      run_frames(probe, coarse_frames, slot_counter, ctx, *pdsch_proc, *cpu_proc, *gpu_proc, *tx_grid, *rx_grid,
                 *executor, config_rng);

      if (probe.cpu_bler() < 1.0f || probe.gpu_bler() < 1.0f) {
        fmt::print("  Start SNR {:.1f}dB not at 100%% BLER, searching lower...\n", effective_start);
        std::vector<snr_bler_point> backtrack;
        backtrack.push_back(probe);

        while (effective_start > -30.0f) {
          effective_start -= 1.0f;
          snr_bler_point bp{};
          bp.snr_dB = effective_start;
          run_frames(bp, coarse_frames, slot_counter, ctx, *pdsch_proc, *cpu_proc, *gpu_proc, *tx_grid, *rx_grid,
                     *executor, config_rng);
          backtrack.push_back(bp);
          if (bp.cpu_bler() >= 1.0f && bp.gpu_bler() >= 1.0f) {
            break;
          }
        }

        for (auto it = backtrack.rbegin(); it != backtrack.rend(); ++it) {
          coarse_curve.push_back(*it);
        }
        fmt::print("  Found 100%% BLER floor at {:.1f}dB\n", effective_start);
      } else {
        coarse_curve.push_back(probe);
      }
    }

    float forward_start = coarse_curve.back().snr_dB + coarse_step;

    fmt::print("  Phase 1: coarse scan ({} frames @ {:.1f}dB step) from {:.1f}dB...\n",
               coarse_frames, coarse_step, forward_start);

    for (float snr = forward_start; snr <= snr_stop + 1e-6f; snr += coarse_step) {
      snr_bler_point point{};
      point.snr_dB = snr;

      run_frames(point, coarse_frames, slot_counter, ctx, *pdsch_proc, *cpu_proc, *gpu_proc, *tx_grid, *rx_grid,
                 *executor, config_rng);
      coarse_curve.push_back(point);

      if (point.cpu_bler() == 0.0f && point.gpu_bler() == 0.0f) {
        consecutive_zero++;
        if (consecutive_zero >= 2) {
          break;
        }
      } else {
        consecutive_zero = 0;
      }
    }

    // ---------------------------------------------------------------
    // Phase 2: Fine sweep in waterfall region with full frames.
    // ---------------------------------------------------------------
    // Identify waterfall bracket from coarse data.
    float wf_low  = coarse_curve.front().snr_dB;
    float wf_high = coarse_curve.back().snr_dB;

    for (unsigned i = 0; i < coarse_curve.size(); ++i) {
      if (coarse_curve[i].cpu_bler() < 1.0f || coarse_curve[i].gpu_bler() < 1.0f) {
        wf_low = (i > 0) ? coarse_curve[i - 1].snr_dB : coarse_curve[i].snr_dB;
        break;
      }
    }
    for (unsigned i = coarse_curve.size(); i > 0; --i) {
      if (coarse_curve[i - 1].cpu_bler() > 0.0f || coarse_curve[i - 1].gpu_bler() > 0.0f) {
        wf_high = (i < coarse_curve.size()) ? coarse_curve[i].snr_dB : coarse_curve[i - 1].snr_dB;
        break;
      }
    }

    // Add margin of one coarse step on each side.
    wf_low  -= coarse_step;
    wf_high += coarse_step;

    // Build the fine curve in the waterfall region.
    std::vector<snr_bler_point> curve;

    unsigned fine_count = static_cast<unsigned>(std::floor((wf_high - wf_low) / snr_step)) + 1;
    fmt::print("  Phase 2: fine sweep ({} frames @ {:.1f}dB step) over [{:.1f}, {:.1f}]dB ({} points)...\n",
               nof_frames, snr_step, wf_low, wf_high, fine_count);

    for (float snr = wf_low; snr <= wf_high + 1e-6f; snr += snr_step) {
      snr_bler_point point{};
      point.snr_dB = snr;

      run_frames(point, nof_frames, slot_counter, ctx, *pdsch_proc, *cpu_proc, *gpu_proc, *tx_grid, *rx_grid,
                 *executor, config_rng);
      curve.push_back(point);
    }

    // ---------------------------------------------------------------
    // Print combined results.
    // ---------------------------------------------------------------
    fmt::print("\n{:>7s}  {:>8s}  {:>8s}  {:>6s}\n", "SNR(dB)", "CPU_BLER", "GPU_BLER", "Frames");
    fmt::print("{:->7s}  {:->8s}  {:->8s}  {:->6s}\n", "", "", "", "");

    // Find first non-100% point; print only the last 100% point before it.
    unsigned first_interesting = 0;
    for (unsigned i = 0; i < curve.size(); ++i) {
      if (curve[i].cpu_bler() < 1.0f || curve[i].gpu_bler() < 1.0f) {
        first_interesting = i;
        break;
      }
      first_interesting = i; // will be last all-fail if none transitions
    }
    unsigned print_start = (first_interesting > 0) ? first_interesting - 1 : 0;

    for (unsigned i = print_start; i < curve.size(); ++i) {
      const auto& point = curve[i];
      fmt::print("{:>7.1f}  {:>8.3f}  {:>8.3f}  {:>6d}\n", point.snr_dB, point.cpu_bler(), point.gpu_bler(),
                 point.frames);
    }
    if (print_start > 0) {
      fmt::print("  ({} all-fail points below {:.1f}dB omitted)\n", print_start, curve[print_start].snr_dB);
    }

    // Interpolate thresholds.
    float cpu_thr = interpolate_bler_crossing(curve, target_bler, true);
    float gpu_thr = interpolate_bler_crossing(curve, target_bler, false);

    fmt::print("\n");
    if (!std::isnan(cpu_thr) && !std::isnan(gpu_thr)) {
      fmt::print("{:.0f}% BLER: CPU={:.1f}dB  GPU={:.1f}dB  delta={:+.1f}dB\n", target_bler * 100, cpu_thr, gpu_thr,
                 gpu_thr - cpu_thr);
    } else if (!std::isnan(cpu_thr)) {
      fmt::print("{:.0f}% BLER: CPU={:.1f}dB  GPU=N/A\n", target_bler * 100, cpu_thr);
    } else if (!std::isnan(gpu_thr)) {
      fmt::print("{:.0f}% BLER: CPU=N/A  GPU={:.1f}dB\n", target_bler * 100, gpu_thr);
    } else {
      fmt::print("{:.0f}% BLER: crossing not found in [{:.1f}, {:.1f}] dB range\n", target_bler * 100, snr_start,
                 snr_stop);
    }

    summary.push_back({config_label, cpu_thr, gpu_thr});
  }

  // --- Summary table ---
  fmt::print("\n\n=== Summary ===\n");
  fmt::print("{:<28s}  {:>7s}  {:>7s}  {:>7s}\n", "Config", "CPU_10%", "GPU_10%", "Delta");
  fmt::print("{:-<28s}  {:-<7s}  {:-<7s}  {:-<7s}\n", "", "", "", "");

  for (const auto& row : summary) {
    std::string cpu_str = std::isnan(row.cpu_threshold) ? "N/A" : fmt::format("{:.1f}", row.cpu_threshold);
    std::string gpu_str = std::isnan(row.gpu_threshold) ? "N/A" : fmt::format("{:.1f}", row.gpu_threshold);
    std::string delta_str;
    if (!std::isnan(row.cpu_threshold) && !std::isnan(row.gpu_threshold)) {
      delta_str = fmt::format("{:+.1f}", row.gpu_threshold - row.cpu_threshold);
    } else {
      delta_str = "N/A";
    }
    fmt::print("{:<28s}  {:>7s}  {:>7s}  {:>7s}\n", row.config_label, cpu_str, gpu_str, delta_str);
  }

  fmt::print("\n");

  worker_pool->stop();
  return 0;
}
