// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "scheduler.h"
#include "helpers.h"
#include "json_generators/generator_helpers.h"
#include "ocudu/scheduler/scheduler_metrics.h"
#include <complex>

using namespace ocudu;
using namespace app_helpers;
using namespace json_generators;

static const char* event_to_string(scheduler_cell_event::event_type ev)
{
  switch (ev) {
    case scheduler_cell_event::event_type::ue_add:
      return "ue_create";
    case scheduler_cell_event::event_type::ue_reconf:
      return "ue_reconf";
    case scheduler_cell_event::event_type::ue_rem:
      return "ue_rem";
    default:
      break;
  }
  return "invalid";
}

namespace ocudu {

void to_json(nlohmann::json& json, const scheduler_cell_event& metrics)
{
  json["slot"]       = metrics.slot;
  json["rnti"]       = metrics.rnti;
  json["event_type"] = event_to_string(metrics.type);
// habib added
}

static const char* crc_status_to_string(scheduler_pusch_crc_status status)
{
  switch (status) {
    case scheduler_pusch_crc_status::pending:
      return "pending";
    case scheduler_pusch_crc_status::pass:
      return "pass";
    case scheduler_pusch_crc_status::fail:
      return "fail";
  }
  return "pending";
}

static const char* ul_tx_type_to_string(scheduler_ul_tx_type type)
{
  return type == scheduler_ul_tx_type::retx ? "retx" : "newtx";
}

void to_json(nlohmann::json& json, const scheduler_ul_retx_candidate& candidate)
{
  json["rnti"]    = candidate.rnti;
  json["harq_id"] = candidate.harq_id;
}

void to_json(nlohmann::json& json, const scheduler_ul_newtx_candidate& candidate)
{
  json["rnti"]                      = candidate.rnti;
  json["pending_bytes_at_decision"] = candidate.pending_bytes_at_decision;
  json["priority"]                  = candidate.priority;
  json["rank"]                      = candidate.rank;

// habib added
  if (candidate.grant_context.has_value()) {
    const auto& ctxt = candidate.grant_context.value();

    json["vrb_lims"] = {
        {"rb_start", ctxt.vrb_lims.rb_start},
        {"rb_stop", ctxt.vrb_lims.rb_start + ctxt.vrb_lims.rb_length}};

    json["nof_rb_lims"] = {
        {"min_prbs", ctxt.min_nof_rbs},
        {"max_prbs", ctxt.max_nof_rbs}};

    json["recommended_mcs"]  = ctxt.recommended_mcs;
    json["expected_nof_rbs"] = ctxt.expected_nof_rbs;

    json["pusch_cfg"] = {
        {"time_domain_resource_index", ctxt.pusch_cfg.time_domain_resource_index},
        {"start_symbol", ctxt.pusch_cfg.start_symbol},
        {"nof_symbols", ctxt.pusch_cfg.nof_symbols},
        {"nof_layers", ctxt.pusch_cfg.nof_layers},
        {"mcs_table", ctxt.pusch_cfg.mcs_table},
        {"transform_precoding", ctxt.pusch_cfg.transform_precoding}};
  }
// habib added
}

void to_json(nlohmann::json& json, const scheduler_ul_selected_grant& grant)
{
  json["rnti"]    = grant.rnti;
  json["tx_type"] = ul_tx_type_to_string(grant.tx_type);
}

void to_json(nlohmann::json& json, const scheduler_ul_scheduler_decision& decision)
{
  json["decision_id"] = decision.decision_id;
  json["decision_slot"] = {
      {"hyper_sfn", decision.decision_slot.hyper_sfn()},
      {"sfn", decision.decision_slot.sfn()},
      {"slot_index", decision.decision_slot.slot_index()}};
  json["target_pusch_slot"] = {
      {"hyper_sfn", decision.target_pusch_slot.hyper_sfn()},
      {"sfn", decision.target_pusch_slot.sfn()},
      {"slot_index", decision.target_pusch_slot.slot_index()}};
  json["k2"]               = decision.k2;
// habib added
  if (decision.state_snapshot.has_value()) {
    const auto& snapshot = decision.state_snapshot.value();
    auto& state_json = json["state_snapshot"];

    state_json["snapshot_stage"] = "after_retx_before_newtx_allocation";
    state_json["bwp_size_prbs"]  = snapshot.bwp_size_prbs;
    state_json["remaining_rbs"]  = snapshot.remaining_rbs;
// habib added
    state_json["final_usable_prbs"] = snapshot.final_usable_prbs;
// habib added
    state_json["occupied_prb_ranges"] = nlohmann::json::array();

    for (const auto& range : snapshot.occupied_prb_ranges) {
      state_json["occupied_prb_ranges"].push_back({
          {"rb_start", range.rb_start},
          {"rb_stop", range.rb_start + range.rb_length}});
    }
  }
// habib added
  json["retx_candidates"]  = decision.retx_candidates;
  json["newtx_candidates"] = decision.newtx_candidates;
// habib added
  json["feasible_action_mask"] = decision.feasible_action_mask;
// habib added

  json["selected_grants"]  = decision.selected_grants;
// habib added
}

//habib added
void to_json(
    nlohmann::json& json,
    const scheduler_prb_range& range)
{
  json["rb_start"] =
      range.rb_start;
/*
  json["rb_length"] =
      range.rb_length;
*/

  // The stop index is exclusive.
  json["rb_stop"] =
      range.rb_start + range.rb_length;
}

/*
void to_json(
    nlohmann::json& json,
    const scheduler_pusch_allocation& allocation)
{
  json["hyper_sfn"] =
      allocation.slot.hyper_sfn();

  json["sfn"] =
      allocation.slot.sfn();

  json["slot_index"] =
      allocation.slot.slot_index();

  json["slot_count"] =
      allocation.slot.count();

  json["allocation_type"] =
      allocation.allocation_type;

  json["bwp_start_crb"] =
      allocation.bwp_start_crb;

  json["bwp_size_prbs"] =
      allocation.bwp_size_prbs;

  json["nof_prbs"] =
      allocation.nof_prbs;

  json["prb_ranges"] =
      allocation.prb_ranges;

  json["intra_slot_freq_hopping"] =
      allocation.intra_slot_freq_hopping;

  if (allocation.allocation_type == 0) {
    json["rbg_indices"] =
        allocation.rbg_indices;
  }

  if (allocation.second_hop_rb_start.has_value()) {
    json["second_hop_rb_start"] =
        allocation.second_hop_rb_start.value();
  }

  if (allocation.allocation_type == 1 &&
      allocation.prb_ranges.size() == 1) {
    json["rb_start"] =
        allocation.prb_ranges.front().rb_start;

    json["rb_length"] =
        allocation.prb_ranges.front().rb_length;

    json["rb_stop"] =
        allocation.prb_ranges.front().rb_start +
        allocation.prb_ranges.front().rb_length;
  }
}

*/


void to_json(
    nlohmann::json& json,
    const scheduler_pusch_allocation& allocation)
{
// habib added
  if (allocation.decision_id.has_value()) {
    json["decision_id"] =
        allocation.decision_id.value();
  }

// habib added
  json["hyper_sfn"] =
      allocation.slot.hyper_sfn();

  json["sfn"] =
      allocation.slot.sfn();

  json["slot_index"] =
      allocation.slot.slot_index();

  json["allocation_type"] =
      allocation.allocation_type;

  json["bwp_size_prbs"] =
      allocation.bwp_size_prbs;

  json["nof_prbs"] =
      allocation.nof_prbs;

  json["prb_ranges"] =
      allocation.prb_ranges;
// habib added

  json["mcs"] =
      allocation.mcs;
  json["tbs_bytes"] =
      allocation.tbs_bytes;
  json["harq_id"] =
      allocation.harq_id;
  json["crc_status"] =
      crc_status_to_string(allocation.crc_status);
// habib added
}
//habib added

//habib added
void to_json(
    nlohmann::json& json,
    const scheduler_srs_report& report)
{
  // -------------------------------------------------------
  // Slot information.
  // -------------------------------------------------------

  json["srs_slot"] = {
      {"sfn", report.srs_slot.sfn()},
      {"slot_index", report.srs_slot.slot_index()},
      {"slot_count", report.srs_slot.count()}
  };

  // -------------------------------------------------------
  // Wideband SRS channel matrix.
  //
  // Do not expose rx_port / tx_port in your requested JSON.
  // -------------------------------------------------------

  json["channel_matrix"] =
      nlohmann::json::array();

  const unsigned nof_rx_ports =
      report.channel_matrix.get_nof_rx_ports();

  const unsigned nof_tx_ports =
      report.channel_matrix.get_nof_tx_ports();

  for (unsigned rx = 0;
       rx != nof_rx_ports;
       ++rx) {

    for (unsigned tx = 0;
         tx != nof_tx_ports;
         ++tx) {

      const cf_t h =
          report.channel_matrix.get_coefficient(
              rx, tx);

      json["channel_matrix"].push_back({
          {"real", std::real(h)},
          {"imag", std::imag(h)},
          {"magnitude", std::abs(h)},
          {"phase_rad", std::arg(h)}
      });
    }
  }

  // -------------------------------------------------------
  // Raw/derived SRS measurements.
  // -------------------------------------------------------

  if (report.srs_epre_db.has_value()) {
    json["srs_epre_db"] =
        report.srs_epre_db.value();
  } else {
    json["srs_epre_db"] = nullptr;
  }

  if (report.srs_rsrp_db.has_value()) {
    json["srs_rsrp_db"] =
        report.srs_rsrp_db.value();
  } else {
    json["srs_rsrp_db"] = nullptr;
  }

  if (report.srs_noise_variance.has_value()) {
    json["srs_noise_variance"] =
        report.srs_noise_variance.value();
  } else {
    json["srs_noise_variance"] = nullptr;
  }

  if (report.srs_snr_db.has_value()) {
    json["srs_snr_db"] =
        report.srs_snr_db.value();
  } else {
    json["srs_snr_db"] = nullptr;
  }

  if (report.srs_ta_ns.has_value()) {
    json["srs_ta_ns"] =
        report.srs_ta_ns.value();
  } else {
    json["srs_ta_ns"] = nullptr;
  }
}
//habib added


/*
void to_json(nlohmann::json& json, const scheduler_ue_metrics& metrics)
{
  json["ue"]   = metrics.ue_index;
  json["pci"]  = metrics.pci;
  json["rnti"] = metrics.rnti;
  json["cqi"]  = (metrics.cqi_stats.get_nof_observations() > 0)
                     ? static_cast<uint8_t>(std::round(metrics.cqi_stats.get_mean()))
                     : -1;

  json["dl_ri"] = metrics.dl_ri_stats.get_nof_observations() > 0 ? metrics.dl_ri_stats.get_mean() : 1;
  json["ul_ri"] = metrics.ul_ri_stats.get_nof_observations() > 0 ? metrics.ul_ri_stats.get_mean() : 1;

  json["dl_mcs"]     = metrics.dl_mcs.value();
  json["dl_brate"]   = metrics.dl_brate_kbps * 1e3;
  json["dl_nof_ok"]  = metrics.dl_nof_ok;
  json["dl_nof_nok"] = metrics.dl_nof_nok;
  json["dl_bs"]      = metrics.dl_bs;
  if (!std::isnan(metrics.pusch_snr_db) && !iszero(metrics.pusch_snr_db)) {
    json["pusch_snr_db"] = std::clamp(metrics.pusch_snr_db, -99.9f, 99.9f);
  }
  if (!std::isnan(metrics.pusch_rsrp_db) && !iszero(metrics.pusch_rsrp_db)) {
    json["pusch_rsrp_db"] = std::clamp(metrics.pusch_rsrp_db, -99.9f, 0.0f);
  }
  if (!std::isnan(metrics.pucch_snr_db) && !iszero(metrics.pucch_snr_db)) {
    json["pucch_snr_db"] = std::clamp(metrics.pucch_snr_db, -99.9f, 99.9f);
  }

  json["ta_ns"] =
      (metrics.ta_stats.get_nof_observations() > 0) ? std::optional{metrics.ta_stats.get_mean() * 1e9} : 0.0f;
  json["pusch_ta_ns"] = (metrics.pusch_ta_stats.get_nof_observations() > 0)
                            ? std::optional{metrics.pusch_ta_stats.get_mean() * 1e9}
                            : 0.0f;
  json["pucch_ta_ns"] = (metrics.pucch_ta_stats.get_nof_observations() > 0)
                            ? std::optional{metrics.pucch_ta_stats.get_mean() * 1e9}
                            : 0.0f;
  json["srs_ta_ns"] =
      (metrics.srs_ta_stats.get_nof_observations() > 0) ? std::optional{metrics.srs_ta_stats.get_mean() * 1e9} : 0.0f;
  json["ul_mcs"]                       = metrics.ul_mcs.value();
  
  //habib added
  json["tot_pusch_prbs_used"]          = metrics.tot_pusch_prbs_used;

  json["pusch_allocations"]            = metrics.pusch_allocations;
  //habib added

  json["ul_brate"]                     = metrics.ul_brate_kbps * 1e3;
  json["ul_nof_ok"]                    = metrics.ul_nof_ok;
  json["ul_nof_nok"]                   = metrics.ul_nof_nok;
  json["last_phr"]                     = metrics.last_phr.has_value() ? metrics.last_phr : 0;
  json["max_pusch_distance"]           = metrics.max_pusch_distance_ms.has_value() ? metrics.max_pusch_distance_ms : 0;
  json["max_pdsch_distance"]           = metrics.max_pdsch_distance_ms.has_value() ? metrics.max_pdsch_distance_ms : 0;
  json["bsr"]                          = metrics.bsr;
  json["nof_pucch_f0f1_invalid_harqs"] = metrics.nof_pucch_f0f1_invalid_harqs;
  json["nof_pucch_f2f3f4_invalid_harqs"] = metrics.nof_pucch_f2f3f4_invalid_harqs;
  json["nof_pucch_f2f3f4_invalid_csis"]  = metrics.nof_pucch_f2f3f4_invalid_csis;
  json["nof_pusch_invalid_harqs"]        = metrics.nof_pusch_invalid_harqs;
  json["nof_pusch_invalid_csis"]         = metrics.nof_pusch_invalid_csis;
  json["avg_ce_delay"]                   = metrics.avg_ce_delay_ms.has_value() ? metrics.avg_ce_delay_ms : 0.0f;
  json["max_ce_delay"]                   = metrics.max_ce_delay_ms.has_value() ? metrics.max_ce_delay_ms : 0.0f;
  json["avg_crc_delay"]                  = metrics.avg_crc_delay_ms.has_value() ? metrics.avg_crc_delay_ms : 0.0f;
  json["max_crc_delay"]                  = metrics.max_crc_delay_ms.has_value() ? metrics.max_crc_delay_ms : 0.0f;
  json["avg_pusch_harq_delay"] = metrics.avg_pusch_harq_delay_ms.has_value() ? metrics.avg_pusch_harq_delay_ms : 0.0f;
  json["max_pusch_harq_delay"] = metrics.max_pusch_harq_delay_ms.has_value() ? metrics.max_pusch_harq_delay_ms : 0.0f;
  json["avg_pucch_harq_delay"] = metrics.avg_pucch_harq_delay_ms.has_value() ? metrics.avg_pucch_harq_delay_ms : 0.0f;
  json["max_pucch_harq_delay"] = metrics.max_pucch_harq_delay_ms.has_value() ? metrics.max_pucch_harq_delay_ms : 0.0f;
  json["avg_sr_to_pusch_delay"] =
      metrics.avg_sr_to_pusch_delay_ms.has_value() ? metrics.avg_sr_to_pusch_delay_ms : 0.0f;
  json["max_sr_to_pusch_delay"] =
      metrics.max_sr_to_pusch_delay_ms.has_value() ? metrics.max_sr_to_pusch_delay_ms : 0.0f;
}
*/
void to_json(nlohmann::json& json, const scheduler_ue_metrics& metrics)
{
  json["ue"]   = metrics.ue_index;
  json["rnti"] = metrics.rnti;

  json["cqi"] =
      (metrics.cqi_stats.get_nof_observations() > 0)
          ? static_cast<uint8_t>(
                std::round(metrics.cqi_stats.get_mean()))
          : -1;

  json["ul_ri"] =
      metrics.ul_ri_stats.get_nof_observations() > 0
          ? metrics.ul_ri_stats.get_mean()
          : 1;

  json["dl_mcs"]   = metrics.dl_mcs.value();
  json["dl_brate"] = metrics.dl_brate_kbps * 1e3;
  json["dl_bs"]    = metrics.dl_bs;

  if (!std::isnan(metrics.pusch_snr_db) &&
      !iszero(metrics.pusch_snr_db)) {
    json["pusch_snr_db"] =
        std::clamp(metrics.pusch_snr_db, -99.9f, 99.9f);
  }

  if (!std::isnan(metrics.pusch_rsrp_db) &&
      !iszero(metrics.pusch_rsrp_db)) {
    json["pusch_rsrp_db"] =
        std::clamp(metrics.pusch_rsrp_db, -99.9f, 0.0f);
  }

  json["ta_ns"] =
      (metrics.ta_stats.get_nof_observations() > 0)
          ? std::optional{
                metrics.ta_stats.get_mean() * 1e9}
          : 0.0f;

  json["pusch_ta_ns"] =
      (metrics.pusch_ta_stats.get_nof_observations() > 0)
          ? std::optional{
                metrics.pusch_ta_stats.get_mean() * 1e9}
          : 0.0f;

  json["srs_ta_ns"] =
      (metrics.srs_ta_stats.get_nof_observations() > 0)
          ? std::optional{
                metrics.srs_ta_stats.get_mean() * 1e9}
          : 0.0f;

  json["bsr"] = metrics.bsr;

  json["avg_ce_delay"] =
      metrics.avg_ce_delay_ms.has_value()
          ? metrics.avg_ce_delay_ms
          : 0.0f;

  json["avg_crc_delay"] =
      metrics.avg_crc_delay_ms.has_value()
          ? metrics.avg_crc_delay_ms
          : 0.0f;

  json["tot_pusch_prbs_used"] =
      metrics.tot_pusch_prbs_used;

  //habib added
  json["pusch_allocations"] =
      metrics.pusch_allocations;

  json["srs_reports"] =
    metrics.srs_reports;
  //habib added


  json["ul_mcs"]     = metrics.ul_mcs.value();
  json["ul_brate"]   = metrics.ul_brate_kbps * 1e3;
  json["ul_nof_ok"]  = metrics.ul_nof_ok;
  json["ul_nof_nok"] = metrics.ul_nof_nok;
}

/*
void to_json(nlohmann::json& json, const scheduler_cell_metrics& metrics)
{
  // Cell metrics.
  auto& cell_json                      = json["cell_metrics"];
  cell_json["pci"]                     = metrics.pci;
  cell_json["error_indication_count"]  = metrics.nof_error_indications;
  cell_json["average_latency"]         = metrics.average_decision_latency.count();
  cell_json["max_latency"]             = metrics.max_decision_latency.count();
  cell_json["nof_failed_pdcch_allocs"] = metrics.nof_failed_pdcch_allocs;
  cell_json["nof_failed_uci_allocs"]   = metrics.nof_failed_uci_allocs;
  cell_json["latency_histogram"]       = metrics.latency_histogram;
  cell_json["msg3_nof_ok"]             = metrics.nof_msg3_ok;
  cell_json["msg3_nof_nok"]            = metrics.nof_msg3_nok;
  cell_json["avg_prach_delay"] = metrics.avg_prach_delay_slots.has_value() ? metrics.avg_prach_delay_slots : 0.0f;
  cell_json["late_dl_harqs"]   = metrics.nof_failed_pdsch_allocs_late_harqs;
  cell_json["late_ul_harqs"]   = metrics.nof_failed_pusch_allocs_late_harqs;
  cell_json["pucch_tot_rb_usage_avg"] = metrics.pucch_tot_rb_usage_avg;
  if (metrics.pusch_prbs_used_per_tdd_slot_idx.size()) {
    cell_json["pusch_prbs_used_per_tdd_slot_idx"] = metrics.pusch_prbs_used_per_tdd_slot_idx;
  }
  if (metrics.pdsch_prbs_used_per_tdd_slot_idx.size()) {
    cell_json["pdsch_prbs_used_per_tdd_slot_idx"] = metrics.pdsch_prbs_used_per_tdd_slot_idx;
  }

  if (!metrics.ue_metrics.empty()) {
    json["ue_list"] = metrics.ue_metrics;
  }
  if (!metrics.events.empty()) {
    json["event_list"] = metrics.events;
  }
}
*/

void to_json(
    nlohmann::json& json,
    const scheduler_cell_metrics& metrics)
{
  auto& cell_json = json["cell_metrics"];

  // Keep PCI here so the Python collector can identify the cell.
  cell_json["pci"] = metrics.pci;

  cell_json["average_latency"] =
      metrics.average_decision_latency.count();

// habib added
  json["ul_scheduler_decisions"] =
      metrics.ul_scheduler_decisions;

// habib added
  if (!metrics.ue_metrics.empty()) {
    json["ue_list"] = metrics.ue_metrics;
  }
}

} // namespace ocudu

nlohmann::json ocudu::app_helpers::json_generators::generate(const scheduler_metrics_report& metrics)
{
  nlohmann::json json;

  json["timestamp"] = get_time_stamp();
  json["cells"]     = metrics.cells;

  return json;
}

std::string ocudu::app_helpers::json_generators::generate_string(const scheduler_metrics_report& metrics, int indent)
{
  return generate(metrics).dump(indent);
}
