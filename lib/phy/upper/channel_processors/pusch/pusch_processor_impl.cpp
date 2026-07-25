// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "pusch_processor_impl.h"
#include "../../phy_acceleration_runtime_options.h"
#include "pusch_acceleration_runtime_options.h"
#include "pusch_decoder_buffer_dummy.h"
#include "pusch_processor_notifier_adaptor.h"
#include "pusch_processor_validator_impl.h"
#include "ocudu/instrumentation/traces/du_traces.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc.h"
#include "ocudu/phy/upper/channel_processors/pusch/formatters.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_codeword_buffer.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_decoder_buffer.h"
#include "ocudu/phy/upper/unique_rx_buffer.h"
#include "ocudu/ran/pusch/ulsch_info.h"
#include "ocudu/ran/sch/sch_dmrs_power.h"
#include "ocudu/ran/uci/uci_formatters.h"
#include "ocudu/ran/uci/uci_part2_size_calculator.h"
#include "ocudu/support/error_handling.h"
#include "ocudu/support/tracing/scoped_trace.h"
#include <chrono>

using namespace ocudu;

/// \brief Looks at the output of the validator and, if unsuccessful, fills \c msg with the error message.
///
/// This is used to call the validator inside the process methods only if asserts are active.
[[maybe_unused]] static bool handle_validation(std::string& msg, const error_type<std::string>& err)
{
  bool is_success = err.has_value();
  if (!is_success) {
    msg = err.error();
  }
  return is_success;
}

namespace {
bool supports_resident_accelerated_pusch_demodulation(const pusch_processor::pdu_t& pdu)
{
  unsigned nof_rx_ports = static_cast<unsigned>(pdu.rx_ports.size());
  bool     supported_layers =
      (pdu.nof_tx_layers == 1) || (pdu.nof_tx_layers == 2) || (pdu.nof_tx_layers == 3) || (pdu.nof_tx_layers == 4);
  bool supported_ports     = (nof_rx_ports == 1) || (nof_rx_ports == 2) || (nof_rx_ports == 4) || (nof_rx_ports == 8);
  bool transform_precoding = !std::holds_alternative<pusch_processor::dmrs_configuration>(pdu.dmrs);

  return supported_layers && supported_ports && (pdu.nof_tx_layers <= nof_rx_ports) &&
         (!transform_precoding || (pdu.nof_tx_layers == 1));
}

bool prefer_cpu_for_small_pusch_grants(const pusch_processor::pdu_t& pdu, const ulsch_information& info)
{
  static const pusch_acceleration_thresholds thresholds = pusch_acceleration_read_thresholds();
  return pusch_acceleration_prefers_cpu_for_sch_grant(
      info.sch.has_value(), pdu.freq_alloc.get_nof_rb(), info.nof_ul_sch_bits.value(), thresholds);
}

bool resident_sch_device_uci_enabled()
{
  // Device-side UCI demux remains opt-in. OTA testing showed silent HARQ/CSI publication misses in this path, which can
  // drive scheduler HARQ timeouts and RLF. Keep the policy decision here instead of spreading the environment knob
  // through the generic PUSCH data path.
  return pusch_acceleration_device_uci_enabled();
}

struct resident_sch_plan {
  bool has_sch_data             = false;
  bool has_ul_sch_softbits      = false;
  bool no_uci                   = false;
  bool no_csi_part2             = false;
  bool supports_resident_demod  = false;
  bool supports_resident_decode = false;
  bool supports_pdu             = false;
  bool small_grant_prefers_cpu  = false;
  bool device_uci_enabled       = false;

  bool can_use_accelerated_pusch() const
  {
    return has_sch_data && supports_resident_demod && supports_resident_decode && supports_pdu &&
           (no_uci || no_csi_part2);
  }

  bool can_bypass_channel_estimator() const
  {
    return can_use_accelerated_pusch() && has_ul_sch_softbits && !small_grant_prefers_cpu;
  }

  bool use_resident_decode() const { return use_resident_sch_only() || use_resident_sch_with_device_uci(); }

  bool use_resident_sch_only() const
  {
    return has_ul_sch_softbits && no_uci && supports_resident_demod && supports_resident_decode && supports_pdu &&
           !small_grant_prefers_cpu;
  }

  bool use_resident_sch_with_device_uci() const
  {
    return device_uci_enabled && has_ul_sch_softbits && !no_uci && no_csi_part2 && supports_resident_demod &&
           supports_resident_decode && supports_pdu && !small_grant_prefers_cpu;
  }
};

resident_sch_plan make_resident_sch_plan(const pusch_processor::pdu_t& pdu,
                                         const ulsch_information&      info,
                                         pusch_demodulator&            demodulator,
                                         pusch_decoder&                decoder)
{
  resident_sch_plan plan;
  plan.has_sch_data             = pdu.codeword.has_value();
  plan.has_ul_sch_softbits      = plan.has_sch_data && (info.nof_ul_sch_bits.value() != 0);
  plan.no_uci                   = (pdu.uci.nof_harq_ack == 0) && (pdu.uci.nof_csi_part1 == 0);
  plan.no_csi_part2             = pdu.uci.csi_part2_size.entries.empty();
  plan.supports_resident_demod  = demodulator.supports_resident_mode();
  plan.supports_resident_decode = decoder.supports_resident_decode();
  plan.supports_pdu             = supports_resident_accelerated_pusch_demodulation(pdu);
  plan.small_grant_prefers_cpu  = plan.has_ul_sch_softbits && prefer_cpu_for_small_pusch_grants(pdu, info);
  plan.device_uci_enabled       = resident_sch_device_uci_enabled();
  return plan;
}

class scoped_resident_sch_mode
{
public:
  scoped_resident_sch_mode(pusch_demodulator& demodulator_, pusch_decoder& decoder_, bool active_) :
    demodulator(active_ ? &demodulator_ : nullptr), decoder(active_ ? &decoder_ : nullptr)
  {
    if (demodulator != nullptr) {
      demodulator->enable_resident_mode();
      decoder->enable_resident_decode();
    }
  }

  scoped_resident_sch_mode(const scoped_resident_sch_mode&)            = delete;
  scoped_resident_sch_mode& operator=(const scoped_resident_sch_mode&) = delete;
  scoped_resident_sch_mode(scoped_resident_sch_mode&&)                 = delete;
  scoped_resident_sch_mode& operator=(scoped_resident_sch_mode&&)      = delete;

  ~scoped_resident_sch_mode()
  {
    if (demodulator != nullptr) {
      demodulator->disable_resident_mode();
      decoder->disable_resident_decode();
    }
  }

private:
  pusch_demodulator* demodulator = nullptr;
  pusch_decoder*     decoder     = nullptr;
};

using resident_codeword_info = resident_softbit_buffer;
using resident_uci_info      = resident_uci_buffer;

resident_codeword_info get_last_resident_codeword(const pusch_demodulator& demodulator)
{
  return demodulator.get_resident_softbits();
}

resident_uci_info get_last_resident_uci(const pusch_demodulator& demodulator)
{
  return demodulator.get_resident_uci();
}

bool try_start_resident_decode(pusch_decoder& decoder, const resident_codeword_info& codeword)
{
  return decoder.try_decode_resident(codeword);
}

class pusch_processor_csi_part1_feedback_impl : public pusch_processor_csi_part1_feedback
{
public:
  pusch_processor_csi_part1_feedback_impl(pusch_uci_decoder_wrapper& csi_part2_decoder_,
                                          pusch_decoder&             ulsch_decoder_,
                                          ulsch_demultiplex&         demultiplex_,
                                          modulation_scheme          modulation_,
                                          uci_part2_size_description csi_part2_size_,
                                          ulsch_configuration        ulsch_config_) :
    csi_part2_decoder(csi_part2_decoder_),
    ulsch_decoder(ulsch_decoder_),
    demultiplex(demultiplex_),
    modulation(modulation_),
    csi_part2_size(std::move(csi_part2_size_)),
    ulsch_config(std::move(ulsch_config_))
  {
  }

  void connect_notifier(pusch_processor_notifier_adaptor& notifier_) { notifier = &notifier_; }

  void on_csi_part1(const uci_payload_type& part1) override
  {
    ocudu_assert(notifier != nullptr, "Notifier not connected.");

    unsigned nof_csi_part_2_bits = uci_part2_get_size(part1, csi_part2_size);

    // Skip if the number of CSI Part 2 bits is zero.
    if (nof_csi_part_2_bits == 0) {
      return;
    }

    // Update the number of CSI Part 2 bits.
    ulsch_config.nof_csi_part2_bits = units::bits(nof_csi_part_2_bits);

    // Recalculate the UL-SCH information.
    ulsch_information info = get_ulsch_information(ulsch_config);

    // Get CSI Part 2 notifier.
    pusch_uci_decoder_notifier& csi_part2_notifier = notifier->get_csi_part2_notifier();

    // Configure CSI Part 2 decoder.
    pusch_decoder_buffer& csi_part2_buffer =
        csi_part2_decoder.new_transmission(nof_csi_part_2_bits, modulation, csi_part2_notifier);

    // Configure UL-SCH demultiplex.
    demultiplex.set_csi_part2(csi_part2_buffer, nof_csi_part_2_bits, info.nof_csi_part2_bits.value());

    // Set the number of UL-SCH softbits in the PUSCH decoder.
    ulsch_decoder.set_nof_softbits(info.nof_ul_sch_bits);
  }

private:
  pusch_processor_notifier_adaptor* notifier;
  pusch_uci_decoder_wrapper&        csi_part2_decoder;
  pusch_decoder&                    ulsch_decoder;
  ulsch_demultiplex&                demultiplex;
  modulation_scheme                 modulation;
  uci_part2_size_description        csi_part2_size;
  ulsch_configuration               ulsch_config;
};

/// Stub estimator results for resident acceleration mode (skips CPU channel estimation).
/// CSI (SINR/EVM) is populated later via deferred accelerator readback.
class resident_stub_estimator_results : public dmrs_pusch_estimator_results
{
public:
  float                           get_noise_variance(unsigned /*rx_port*/) const override { return 1.0f; }
  float                           get_rsrp(unsigned /*rx_port*/, unsigned /*tx_layer*/) const override { return 1.0f; }
  static_vector<float, MAX_PORTS> get_rsrp_all_ports(unsigned /*tx_layer*/) const override { return {1.0f}; }
  float                           get_epre(unsigned /*rx_port*/) const override { return 1.0f; }
  float                           get_snr(unsigned /*rx_port*/) const override { return 100.0f; }
  float                           get_layer_average_snr(unsigned /*tx_layer*/) const override { return 100.0f; }
  phy_time_unit get_time_alignment(unsigned /*rx_port*/) const override { return phy_time_unit::from_seconds(0.0); }
  std::optional<float> get_cfo_Hz(unsigned /*rx_port*/) const override { return std::nullopt; }
  void                 get_symbol_ch_estimate(span<cbf16_t> estimates, unsigned, unsigned, unsigned) const override
  {
    std::fill(estimates.begin(), estimates.end(), cbf16_t{});
  }
  void get_symbol_ch_estimate(span<cbf16_t> estimates,
                              unsigned,
                              unsigned,
                              unsigned,
                              const bounded_bitset<MAX_NOF_SUBCARRIERS>&) const override
  {
    std::fill(estimates.begin(), estimates.end(), cbf16_t{});
  }
  void get_channel_state_information(channel_state_information& /*csi*/) const override
  {
    // Leave CSI empty. SINR is set via deferred resident-path readback.
  }
};

} // namespace

// Dummy PUSCH decoder buffer. Used for PUSCH transmissions without SCH data.
static pusch_decoder_buffer_dummy decoder_buffer_dummy;

pusch_processor_impl::pusch_processor_impl(configuration& config) :
  estimator_notifier_configurator(*this),
  logger(ocudulog::fetch_basic_logger("PHY")),
  dependencies_pool(std::move(config.dependencies_pool)),
  decoder(std::move(config.decoder)),
  dec_nof_iterations(config.dec_nof_iterations),
  dec_enable_early_stop(config.dec_enable_early_stop),
  ce_dims(config.ce_dims),
  csi_sinr_calc_method(config.csi_sinr_calc_method)
{
  ocudu_assert(dependencies_pool, "Invalid dependency pool.");
  ocudu_assert(decoder, "Invalid decoder.");
  ocudu_assert(dec_nof_iterations != 0, "The decoder number of iterations must be non-zero.");
}

void pusch_processor_impl::process(span<uint8_t>                    data,
                                   unique_rx_buffer                 rm_buffer,
                                   pusch_processor_result_notifier& notifier,
                                   const resource_grid_reader&      grid,
                                   const pusch_processor::pdu_t&    pdu)
{
  auto scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.process");

  bool collect_accelerator_timing = phy_acceleration_env_flag_enabled("OCUDU_PUSCH_ACCELERATION_TIMING");

  // Get dependencies.
  concurrent_dependencies_pool_type::ptr dependencies = [&]() {
    auto pool_scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.dependencies");
    return dependencies_pool->get();
  }();

  if (!dependencies) {
    logger.error("Failed to retrieve PUSCH processor dependencies.");

    // Notify
    if (pdu.uci.nof_harq_ack != 0) {
      notifier.on_uci({.harq_ack  = {.payload = uci_payload_type(pdu.uci.nof_harq_ack), .status = uci_status::invalid},
                       .csi_part1 = {},
                       .csi_part2 = {},
                       .csi       = {}});
    }

    // Notify the completion of the data processing as the CRC check is KO.
    if (pdu.codeword.has_value()) {
      notifier.on_sch({});
    }

    return;
  }

  // Assert PDU.
  [[maybe_unused]] std::string msg;
  ocudu_assert(handle_validation(msg, pusch_processor_validator_impl(ce_dims).is_valid(pdu)), "{}", msg);

  // Get RB mask relative to Point A. It assumes PUSCH is never interleaved.
  crb_bitmap rb_mask = pdu.freq_alloc.get_crb_mask(pdu.bwp_start_rb, pdu.bwp_size_rb);

  bool             enable_transform_precoding  = false;
  unsigned         scrambling_id               = 0;
  unsigned         n_rs_id                     = 0;
  bool             n_scid                      = false;
  unsigned         nof_cdm_groups_without_data = 2;
  dmrs_config_type dmrs_type                   = dmrs_config_type::type1;
  if (std::holds_alternative<ocudu::pusch_processor::dmrs_configuration>(pdu.dmrs)) {
    const auto& dmrs_config     = std::get<ocudu::pusch_processor::dmrs_configuration>(pdu.dmrs);
    scrambling_id               = dmrs_config.scrambling_id;
    n_scid                      = dmrs_config.n_scid;
    nof_cdm_groups_without_data = dmrs_config.nof_cdm_groups_without_data;
    dmrs_type                   = dmrs_config.dmrs;
  } else {
    const auto& dmrs_config    = std::get<ocudu::pusch_processor::dmrs_transform_precoding_configuration>(pdu.dmrs);
    enable_transform_precoding = true;
    n_rs_id                    = dmrs_config.n_rs_id;
  }

  // Check if resident SCH processing is possible BEFORE running CPU channel estimation.
  // If so, bypass the CPU estimator entirely; the accelerator computes channel, noise and scheduler-visible metrics.
  {
    auto policy_scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.resident_policy");
    if (pdu.codeword.has_value() && dependencies->get_demodulator().supports_resident_mode() &&
        decoder->supports_resident_decode() && supports_resident_accelerated_pusch_demodulation(pdu) &&
        (((pdu.uci.nof_harq_ack == 0) && (pdu.uci.nof_csi_part1 == 0)) || pdu.uci.csi_part2_size.entries.empty())) {
      bool overlap_dc = false;
      if (pdu.dc_position.has_value()) {
        unsigned dc_position_prb = *pdu.dc_position / NOF_SUBCARRIERS_PER_RB;
        overlap_dc               = rb_mask.test(dc_position_prb);
      }

      ulsch_configuration ulsch_config;
      ulsch_config.tbs                   = units::bytes(data.size()).to_bits();
      ulsch_config.mcs_descr             = pdu.mcs_descr;
      ulsch_config.nof_harq_ack_bits     = units::bits(pdu.uci.nof_harq_ack);
      ulsch_config.nof_csi_part1_bits    = units::bits(pdu.uci.nof_csi_part1);
      ulsch_config.nof_csi_part2_bits    = units::bits(0);
      ulsch_config.alpha_scaling         = pdu.uci.alpha_scaling;
      ulsch_config.beta_offset_harq_ack  = pdu.uci.beta_offset_harq_ack;
      ulsch_config.beta_offset_csi_part1 = pdu.uci.beta_offset_csi_part1;
      ulsch_config.beta_offset_csi_part2 = pdu.uci.beta_offset_csi_part2;
      ulsch_config.nof_rb                = pdu.freq_alloc.get_nof_rb();
      ulsch_config.start_symbol_index    = pdu.start_symbol_index;
      ulsch_config.nof_symbols           = pdu.nof_symbols;
      ulsch_config.dmrs_type                   = dmrs_type;
      ulsch_config.dmrs_symbol_mask            = pdu.dmrs_symbol_mask;
      ulsch_config.nof_cdm_groups_without_data = nof_cdm_groups_without_data;
      ulsch_config.nof_layers                  = pdu.nof_tx_layers;
      ulsch_config.contains_dc                 = overlap_dc;

      ulsch_information info = [&]() {
        auto ulsch_scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.ulsch_info");
        return get_ulsch_information(ulsch_config);
      }();
      resident_sch_plan resident_plan = make_resident_sch_plan(pdu, info, dependencies->get_demodulator(), *decoder);
      if (resident_plan.can_bypass_channel_estimator()) {
        resident_stub_estimator_results stub_results;
        process_data(data,
                     std::move(rm_buffer),
                     std::move(dependencies),
                     notifier,
                     stub_results,
                     grid,
                     pdu,
                     dmrs_type,
                     nof_cdm_groups_without_data,
                     0.0f);
        return;
      }
    }
  }

  // Configure the channel estimator.
  dmrs_pusch_estimator::configuration ch_est_config;
  ch_est_config.slot = pdu.slot;
  if (enable_transform_precoding) {
    ch_est_config.sequence_config = dmrs_pusch_estimator::low_papr_sequence_configuration{.n_rs_id = n_rs_id};
  } else {
    ch_est_config.sequence_config = dmrs_pusch_estimator::pseudo_random_sequence_configuration{
        .type = dmrs_type, .nof_tx_layers = pdu.nof_tx_layers, .scrambling_id = scrambling_id, .n_scid = n_scid};
  }
  ch_est_config.scaling      = convert_dB_to_amplitude(-get_sch_to_dmrs_ratio_dB(nof_cdm_groups_without_data));
  ch_est_config.c_prefix     = pdu.cp;
  ch_est_config.symbols_mask = pdu.dmrs_symbol_mask;
  ch_est_config.rb_mask      = rb_mask;
  ch_est_config.first_symbol = pdu.start_symbol_index;
  ch_est_config.nof_symbols  = pdu.nof_symbols;
  ch_est_config.rx_ports.assign(pdu.rx_ports.begin(), pdu.rx_ports.end());

  // Configure and get the estimator notifier.
  dmrs_pusch_estimator&          estimator = dependencies->get_estimator();
  dmrs_pusch_estimator_notifier& estimator_notifier =
      estimator_notifier_configurator.configure(data,
                                                std::move(rm_buffer),
                                                std::move(dependencies),
                                                notifier,
                                                grid,
                                                pdu,
                                                dmrs_type,
                                                nof_cdm_groups_without_data,
                                                collect_accelerator_timing);

  // Run the channel estimator. When done, the notifier will trigger the remaining steps for recovering the PUSCH data.
  {
    auto estimate_scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.channel_estimate");
    estimator.estimate(estimator_notifier, grid, ch_est_config);
  }
}

void pusch_processor_impl::process_data(span<uint8_t>                          data,
                                        unique_rx_buffer                       rm_buffer,
                                        concurrent_dependencies_pool_type::ptr dependencies,
                                        pusch_processor_result_notifier&       notifier,
                                        const dmrs_pusch_estimator_results&    est_results,
                                        const resource_grid_reader&            grid,
                                        const pdu_t&                           pdu,
                                        dmrs_config_type                       dmrs_type,
                                        unsigned                               nof_cdm_groups_without_data,
                                        float                                  ch_estimate_us)
{
  using namespace units::literals;

  auto scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.process_data");

  const bool collect_accelerator_timing = phy_acceleration_env_flag_enabled("OCUDU_PUSCH_ACCELERATION_TIMING");
  auto       process_data_start         = std::chrono::steady_clock::time_point{};
  if (collect_accelerator_timing) {
    process_data_start = std::chrono::steady_clock::now();
  }

  // Get RB mask relative to Point A. According to TS38.211 Section 6.3.1.7, the VRB-to-PRB mapping for PUSCH is never
  // interleaved.
  crb_bitmap rb_mask = pdu.freq_alloc.get_crb_mask(pdu.bwp_start_rb, pdu.bwp_size_rb);

  // Extract channel state information.
  channel_state_information csi(csi_sinr_calc_method);
  est_results.get_channel_state_information(csi);

  // Number of RB used by this transmission.
  unsigned nof_rb = pdu.freq_alloc.get_nof_rb();

  // Determine if the PUSCH allocation overlaps with the position of the DC.
  bool overlap_dc = false;
  if (pdu.dc_position.has_value()) {
    unsigned dc_position_prb = *pdu.dc_position / NOF_SUBCARRIERS_PER_RB;
    overlap_dc               = rb_mask.test(dc_position_prb);
  }

  // Configure the UL SCH transmission.
  ulsch_configuration ulsch_config;
  ulsch_config.tbs                         = units::bytes(data.size()).to_bits();
  ulsch_config.mcs_descr                   = pdu.mcs_descr;
  ulsch_config.nof_harq_ack_bits           = units::bits(pdu.uci.nof_harq_ack);
  ulsch_config.nof_csi_part1_bits          = units::bits(pdu.uci.nof_csi_part1);
  ulsch_config.nof_csi_part2_bits          = 0_bits;
  ulsch_config.alpha_scaling               = pdu.uci.alpha_scaling;
  ulsch_config.beta_offset_harq_ack        = pdu.uci.beta_offset_harq_ack;
  ulsch_config.beta_offset_csi_part1       = pdu.uci.beta_offset_csi_part1;
  ulsch_config.beta_offset_csi_part2       = pdu.uci.beta_offset_csi_part2;
  ulsch_config.nof_rb                      = nof_rb;
  ulsch_config.start_symbol_index          = pdu.start_symbol_index;
  ulsch_config.nof_symbols                 = pdu.nof_symbols;
  ulsch_config.dmrs_type                   = dmrs_type;
  ulsch_config.dmrs_symbol_mask            = pdu.dmrs_symbol_mask;
  ulsch_config.nof_cdm_groups_without_data = nof_cdm_groups_without_data;
  ulsch_config.nof_layers                  = pdu.nof_tx_layers;
  ulsch_config.contains_dc                 = overlap_dc;

  // Prepare demultiplex configuration.
  ulsch_information info = [&]() {
    auto ulsch_scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.process_data.ulsch_info");
    return get_ulsch_information(ulsch_config);
  }();
  ulsch_demultiplex::configuration demux_config;
  demux_config.modulation                  = pdu.mcs_descr.modulation;
  demux_config.nof_layers                  = pdu.nof_tx_layers;
  demux_config.nof_prb                     = ulsch_config.nof_rb;
  demux_config.start_symbol_index          = pdu.start_symbol_index;
  demux_config.nof_symbols                 = pdu.nof_symbols;
  demux_config.nof_harq_ack_rvd            = info.nof_harq_ack_rvd.value();
  demux_config.dmrs                        = dmrs_type;
  demux_config.dmrs_symbol_mask            = ulsch_config.dmrs_symbol_mask;
  demux_config.nof_cdm_groups_without_data = ulsch_config.nof_cdm_groups_without_data;
  demux_config.nof_harq_ack_bits           = ulsch_config.nof_harq_ack_bits.value();
  demux_config.nof_enc_harq_ack_bits       = info.nof_harq_ack_bits.value();
  demux_config.nof_csi_part1_bits          = ulsch_config.nof_csi_part1_bits.value();
  demux_config.nof_enc_csi_part1_bits      = info.nof_csi_part1_bits.value();

  pusch_demodulator& demodulator   = dependencies->get_demodulator();
  resident_sch_plan  resident_plan = [&]() {
    auto policy_scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.process_data.resident_policy");
    return make_resident_sch_plan(pdu, info, demodulator, *decoder);
  }();
  bool use_resident_sch_with_uci = resident_plan.use_resident_sch_with_device_uci();

  // Prepare decoder buffers with dummy instances.
  std::reference_wrapper<pusch_decoder_buffer> decoder_buffer(decoder_buffer_dummy);
  std::reference_wrapper<pusch_decoder_buffer> harq_ack_buffer(decoder_buffer_dummy);
  std::reference_wrapper<pusch_decoder_buffer> csi_part1_buffer(decoder_buffer_dummy);
  pusch_uci_decoder_notifier*                  harq_ack_notifier  = nullptr;
  pusch_uci_decoder_notifier*                  csi_part1_notifier = nullptr;

  // Prepare CSI Part 1 feedback.
  pusch_processor_csi_part1_feedback_impl csi_part1_feedback(dependencies->get_csi_part2_decoder(),
                                                             *decoder,
                                                             dependencies->get_demultiplex(),
                                                             pdu.mcs_descr.modulation,
                                                             pdu.uci.csi_part2_size,
                                                             ulsch_config);

  // Prepare notifiers.
  notifier_adaptor.new_transmission(notifier, csi_part1_feedback, csi);
  csi_part1_feedback.connect_notifier(notifier_adaptor);

  if (resident_plan.has_sch_data) {
    units::bits tbs            = units::bytes(data.size()).to_bits();
    unsigned    nof_codeblocks = compute_nof_codeblocks(tbs, pdu.codeword->ldpc_base_graph);
    units::bits Nref           = ldpc::compute_N_ref(pdu.tbs_lbrm, nof_codeblocks);

    // Prepare decoder configuration.
    pusch_decoder::configuration decoder_config;
    decoder_config.base_graph          = pdu.codeword->ldpc_base_graph;
    decoder_config.rv                  = pdu.codeword->rv;
    decoder_config.mod                 = pdu.mcs_descr.modulation;
    decoder_config.Nref                = Nref.value();
    decoder_config.nof_layers          = pdu.nof_tx_layers;
    decoder_config.nof_ldpc_iterations = dec_nof_iterations;
    decoder_config.use_early_stop      = dec_enable_early_stop;
    decoder_config.new_data            = pdu.codeword->new_data;

    // Setup decoder.
    {
      auto decoder_setup_scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.decoder_new_data");
      decoder_buffer =
          decoder->new_data(data, std::move(rm_buffer), notifier_adaptor.get_sch_data_notifier(), decoder_config);
    }

    // If there is no expected CSI Part 2 payload, the number of UL-SCH LLRs is known without the need to decode the
    // CSI Part 1 payload.
    if (pdu.uci.csi_part2_size.entries.empty()) {
      decoder->set_nof_softbits(info.nof_ul_sch_bits);
    }
  }

  // Prepares HARQ-ACK notifier and buffer.
  if (pdu.uci.nof_harq_ack != 0) {
    harq_ack_notifier = &notifier_adaptor.get_harq_ack_notifier();
    harq_ack_buffer   = dependencies->get_harq_ack_decoder().new_transmission(
        pdu.uci.nof_harq_ack, pdu.mcs_descr.modulation, *harq_ack_notifier);
  }

  // Prepares CSI Part 1 notifier and buffer.
  if (pdu.uci.nof_csi_part1 != 0) {
    csi_part1_notifier = &notifier_adaptor.get_csi_part1_notifier();
    csi_part1_buffer   = dependencies->get_csi_part1_decoder().new_transmission(
        pdu.uci.nof_csi_part1, pdu.mcs_descr.modulation, *csi_part1_notifier);
  }

  // Demultiplex SCH data, HARQ-ACK and CSI Part 1. In hybrid resident/UCI mode, SCH and UCI are compacted on device.
  // The host demux buffer is still prepared so a demodulator fallback can recover through the normal path.
  pusch_decoder_buffer& sch_demux_buffer =
      use_resident_sch_with_uci ? static_cast<pusch_decoder_buffer&>(decoder_buffer_dummy) : decoder_buffer.get();
  pusch_codeword_buffer& demodulator_buffer = [&]() -> pusch_codeword_buffer& {
    auto demux_scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.demux_setup");
    return dependencies->get_demultiplex().demultiplex(
        sch_demux_buffer, harq_ack_buffer, csi_part1_buffer, demux_config);
  }();

  // Demodulate.
  bool enable_transform_precoding = !std::holds_alternative<ocudu::pusch_processor::dmrs_configuration>(pdu.dmrs);

  pusch_demodulator::configuration demod_config;
  demod_config.rnti                        = pdu.rnti;
  demod_config.rb_mask                     = pdu.freq_alloc.get_crb_mask(pdu.bwp_start_rb, pdu.bwp_size_rb);
  demod_config.modulation                  = pdu.mcs_descr.modulation;
  demod_config.start_symbol_index          = pdu.start_symbol_index;
  demod_config.nof_symbols                 = pdu.nof_symbols;
  demod_config.dmrs_symb_pos               = pdu.dmrs_symbol_mask;
  demod_config.dmrs_type                   = demux_config.dmrs;
  demod_config.nof_cdm_groups_without_data = ulsch_config.nof_cdm_groups_without_data;
  demod_config.n_id                        = pdu.n_id;
  // Extract DMRS scrambling parameters from the PDU's DMRS configuration variant.
  if (std::holds_alternative<ocudu::pusch_processor::dmrs_configuration>(pdu.dmrs)) {
    const auto& dmrs_cfg            = std::get<ocudu::pusch_processor::dmrs_configuration>(pdu.dmrs);
    demod_config.dmrs_scrambling_id = dmrs_cfg.scrambling_id;
    demod_config.n_scid             = dmrs_cfg.n_scid;
  } else {
    demod_config.dmrs_scrambling_id = pdu.n_id;
    demod_config.n_scid             = false;
  }
  demod_config.nof_tx_layers              = pdu.nof_tx_layers;
  demod_config.dc_position                = pdu.dc_position;
  demod_config.enable_transform_precoding = enable_transform_precoding;
  if (enable_transform_precoding) {
    const auto& dmrs_cfg = std::get<ocudu::pusch_processor::dmrs_transform_precoding_configuration>(pdu.dmrs);
    demod_config.n_rs_id = dmrs_cfg.n_rs_id;
  }
  demod_config.slot     = pdu.slot;
  demod_config.rx_ports = pdu.rx_ports;
  demod_config.n_rapid  = pdu.n_rapid;
  if (use_resident_sch_with_uci) {
    demod_config.resident.device_uci_demux     = true;
    auto& resident_compaction                  = demod_config.resident.sch_compaction;
    resident_compaction.enabled                = true;
    resident_compaction.nof_ul_sch_bits        = info.nof_ul_sch_bits.value();
    resident_compaction.nof_harq_ack_rvd       = demux_config.nof_harq_ack_rvd;
    resident_compaction.nof_harq_ack_bits      = demux_config.nof_harq_ack_bits;
    resident_compaction.nof_enc_harq_ack_bits  = demux_config.nof_enc_harq_ack_bits;
    resident_compaction.nof_csi_part1_bits     = demux_config.nof_csi_part1_bits;
    resident_compaction.nof_enc_csi_part1_bits = demux_config.nof_enc_csi_part1_bits;
  }

  // Check if the resident accelerator pipeline is possible:
  // - Demodulator supports resident mode.
  // - Decoder supports resident decode.
  // - PDU has SCH softbits, with either no UCI or only HARQ/CSI Part 1 UCI.
  // - Transform precoding is supported by the resident demodulator implementation.
  // Resident decode keeps LLRs in accelerator memory from demodulation through decoding, eliminating
  // both D2H (demod->host) and H2D (host->decoder) transfers.
  // The FP16 path uses optimized fused rate dematching and LDPC decode.
  bool use_resident_decode = resident_plan.use_resident_decode();

  ocudulog::fetch_basic_logger("PHY").debug(
      "PUSCH resident acceleration: has_sch={} no_uci={} demod={} decode={} supported_pdu={} hybrid_uci={} -> {}",
      resident_plan.has_sch_data,
      resident_plan.no_uci,
      resident_plan.supports_resident_demod,
      resident_plan.supports_resident_decode,
      resident_plan.supports_pdu,
      use_resident_sch_with_uci,
      use_resident_decode);

  scoped_resident_sch_mode resident_mode(demodulator, *decoder, use_resident_decode);

  float process_data_setup_us = 0.0f;
  auto  demod_call_start      = std::chrono::steady_clock::time_point{};
  if (collect_accelerator_timing) {
    auto setup_end        = std::chrono::steady_clock::now();
    process_data_setup_us = std::chrono::duration<float, std::micro>(setup_end - process_data_start).count();
    demod_call_start      = setup_end;
  }

  // Run demodulation
  {
    auto demod_scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.demodulate");
    demodulator.demodulate(
        demodulator_buffer, notifier_adaptor.get_demodulator_notifier(), grid, est_results, demod_config);
  }

  float demod_call_us = 0.0f;
  if (collect_accelerator_timing) {
    auto demod_call_end = std::chrono::steady_clock::now();
    demod_call_us       = std::chrono::duration<float, std::micro>(demod_call_end - demod_call_start).count();
  }

  // If resident mode is active, route accelerator-resident LLRs directly to decoder.
  if (use_resident_decode) {
    auto resident_handoff_scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.resident_handoff");

    auto notify_resident_uci = [&]() {
      resident_uci_info uci_info = get_last_resident_uci(demodulator);
      if (uci_info.valid) {
        if (uci_info.harq_ack_decoded) {
          ocudu_assert(harq_ack_notifier != nullptr, "Invalid HARQ-ACK notifier.");
          harq_ack_notifier->on_uci_decoded(uci_info.harq_ack_payload, uci_info.harq_ack_status);
        } else if (!uci_info.harq_ack_llrs.empty()) {
          harq_ack_buffer.get().on_new_softbits(uci_info.harq_ack_llrs);
          harq_ack_buffer.get().on_end_softbits();
        }
        if (uci_info.csi_part1_decoded) {
          ocudu_assert(csi_part1_notifier != nullptr, "Invalid CSI Part 1 notifier.");
          csi_part1_notifier->on_uci_decoded(uci_info.csi_part1_payload, uci_info.csi_part1_status);
        } else if (!uci_info.csi_part1_llrs.empty()) {
          csi_part1_buffer.get().on_new_softbits(uci_info.csi_part1_llrs);
          csi_part1_buffer.get().on_end_softbits();
        }
      } else {
        ocudulog::fetch_basic_logger("PHY").warning(
            "PUSCH resident UCI demux did not publish compact UCI LLRs; UCI may have used fallback demux.");
      }
    };

    bool resident_uci_notified = false;
    if (use_resident_sch_with_uci) {
      resident_uci_info uci_info = get_last_resident_uci(demodulator);
      if (uci_info.valid) {
        notify_resident_uci();
        resident_uci_notified = true;
      }
    }

    resident_codeword_info resident_codeword = get_last_resident_codeword(demodulator);

    if (resident_codeword.valid) {
      if (collect_accelerator_timing) {
        // Pass demodulator gap timing to decoder for inclusion in results.
        decoder->set_demod_gap_timing(demodulator.get_last_grid_staging_us(), demodulator.get_last_demod_sync_us());
        decoder->set_processor_stage_timing(ch_estimate_us, process_data_setup_us, demod_call_us);
      }

      const bool resident_stats_deferred = demodulator.get_last_demod_sync_us() == 0;
      float      sinr_readback_us        = 0.0f;
      decoder->set_pre_join_callback([&demodulator,
                                      this,
                                      use_resident_sch_with_uci,
                                      notify_resident_uci,
                                      &resident_uci_notified,
                                      resident_stats_deferred,
                                      collect_accelerator_timing,
                                      decoder_ptr = decoder.get(),
                                      &sinr_readback_us]() {
        if (resident_stats_deferred) {
          if (collect_accelerator_timing) {
            auto sinr_start = std::chrono::steady_clock::now();
            demodulator.report_deferred_sinr(notifier_adaptor.get_demodulator_notifier());
            sinr_readback_us =
                std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - sinr_start).count();
            decoder_ptr->set_detailed_gap_timing(0.0f, sinr_readback_us);
          } else {
            demodulator.report_deferred_sinr(notifier_adaptor.get_demodulator_notifier());
          }
        }
        if (use_resident_sch_with_uci && !resident_uci_notified) {
          demodulator.finalize_resident_uci();
          notify_resident_uci();
          resident_uci_notified = true;
        }
      });

      // Resident decode: LLRs stay on the accelerator side, avoiding demodulator D2H and decoder H2D transfers.
      bool resident_decode_started = [&]() {
        auto decode_scope = make_scoped_trace(l1_ul_tracer, "L1.UL.PUSCH.resident_decode_start");
        return try_start_resident_decode(*decoder, resident_codeword);
      }();
      if (!resident_decode_started) {
        report_fatal_error("PUSCH resident decode failed after resident demodulation; no safe host fallback exists.");
      }

    } else if (demodulator.used_host_codeword_fallback()) {
      ocudulog::fetch_basic_logger("PHY").warning(
          "PUSCH resident demodulation fell back to host softbits; continuing with normal decode.");
    } else {
      report_fatal_error("PUSCH resident demodulation did not publish resident LLRs; no safe host fallback exists.");
    }
  }
}
