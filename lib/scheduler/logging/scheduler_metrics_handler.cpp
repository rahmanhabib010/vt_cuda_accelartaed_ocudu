// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "scheduler_metrics_handler.h"
#include "../config/cell_configuration.h"
#include "../uci_scheduling/uci_indication_selector.h"
#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/ran/resource_allocation/rb_bitmap.h"
#include "ocudu/ran/slot_point.h"
#include "ocudu/scheduler/result/sched_result.h"
#include "ocudu/scheduler/scheduler_rach_handler.h"

// habib added
#include "ocudu/ran/resource_allocation/rb_bitmap.h"
#include "ocudu/scheduler/result/resource_block_group.h"
// habib added
#include <algorithm>
// habib added
#include <utility>

#include "ocudu/support/math/math_utils.h"
#include <cmath>
#include <utility>
// habib added

using namespace ocudu;

namespace {

class null_metrics_notifier final : public scheduler_cell_metrics_notifier
{
private:
  scheduler_cell_metrics& get_next() override { return null_report; }

  void commit(scheduler_cell_metrics& report) override
  {
    // do nothing
    null_report.ue_metrics.clear();
    null_report.events.clear();
// habib added
    null_report.ul_scheduler_decisions.clear();
// habib added
// habib added
    null_report.late_crc_updates.clear();
// habib added
  }
  bool is_sched_report_required(slot_point_extended sl_tx) const override { return false; }

  scheduler_cell_metrics null_report{};
};

} // namespace

static null_metrics_notifier null_notifier;

cell_metrics_handler::cell_metrics_handler(
    const cell_configuration&                                                      cell_cfg_,
    const std::optional<sched_cell_configuration_request_message::metrics_config>& metrics_cfg) :
  notifier(metrics_cfg.has_value() and metrics_cfg->notifier != nullptr ? *metrics_cfg->notifier : null_notifier),
  cell_cfg(cell_cfg_),
  nof_slots_per_sf(get_nof_slots_per_subframe(cell_cfg.scs_common()))
{
  if (not enabled()) {
    return;
  }

  // Pre-reserve space.
  ues.reserve(MAX_NOF_DU_UES);
  rnti_to_ue_index_lookup.reserve(MAX_NOF_DU_UES);
  const unsigned pre_reserved_event_capacity = std::min(3U * MAX_NOF_DU_UES, metrics_cfg->max_ue_events_per_report);
  pending_events.reserve(pre_reserved_event_capacity);
  unsigned tdd_period_slots = cell_cfg.is_tdd() ? nof_slots_per_tdd_period(*cell_cfg.params.tdd_cfg) : 0U;
  ul_prbs_used_per_tdd_slot_idx.resize(tdd_period_slots);
  dl_prbs_used_per_tdd_slot_idx.resize(tdd_period_slots);
}

cell_metrics_handler::~cell_metrics_handler() {}

bool cell_metrics_handler::enabled() const
{
  return &notifier != &null_notifier;
// habib added
}

slot_point_extended cell_metrics_handler::extend_slot(slot_point slot) const
{
  if (not last_slot_tx.valid()) {
    return slot_point_extended{slot};
  }

  slot_point_extended extended = last_slot_tx;
  extended += slot - last_slot_tx.without_hyper_sfn();
  return extended;
}

scheduler_ul_scheduler_decision*
cell_metrics_handler::find_ul_scheduler_decision(uint64_t decision_id)
{
  auto it = std::find_if(
      data.ul_scheduler_decisions.begin(),
      data.ul_scheduler_decisions.end(),
      [decision_id](const scheduler_ul_scheduler_decision& decision) {
        return decision.decision_id == decision_id;
      });

  return it != data.ul_scheduler_decisions.end() ? &(*it) : nullptr;
}

uint64_t cell_metrics_handler::start_ul_scheduler_decision(
    slot_point decision_slot,
    slot_point target_pusch_slot)
{
  if (not enabled()) {
    return 0;
  }

  scheduler_ul_scheduler_decision decision{};
  decision.decision_id   = next_ul_scheduler_decision_id++;
  decision.decision_slot = extend_slot(decision_slot);

  const int k2 = target_pusch_slot - decision_slot;
  ocudu_assert(k2 >= 0, "Invalid negative PUSCH k2={}", k2);
  decision.k2 = static_cast<unsigned>(k2);

  decision.target_pusch_slot = decision.decision_slot;
  decision.target_pusch_slot += k2;

  data.ul_scheduler_decisions.push_back(std::move(decision));
  return data.ul_scheduler_decisions.back().decision_id;
}

void cell_metrics_handler::add_ul_retx_candidate(
    uint64_t  decision_id,
    rnti_t    rnti,
    harq_id_t harq_id)
{
  if (decision_id == 0) {
    return;
  }

  if (auto* decision = find_ul_scheduler_decision(decision_id)) {
    decision->retx_candidates.push_back(
        scheduler_ul_retx_candidate{rnti, static_cast<unsigned>(harq_id)});
  }
}

void cell_metrics_handler::add_ul_newtx_candidate(
    uint64_t     decision_id,
    rnti_t       rnti,
    units::bytes pending_bytes_at_decision,
    double       priority,
    unsigned     rank)
{
  if (decision_id == 0) {
    return;
  }

  if (auto* decision = find_ul_scheduler_decision(decision_id)) {
    decision->newtx_candidates.push_back(
        scheduler_ul_newtx_candidate{
            rnti,
            pending_bytes_at_decision.value(),
            priority,
            rank});
// habib added
    decision->feasible_action_mask.push_back(0);
// habib added
  }
}

// habib added
// habib added
void cell_metrics_handler::mark_ul_newtx_candidate_feasible(uint64_t decision_id, rnti_t rnti)
{
  if (decision_id == 0) {
    return;
  }

  if (auto* decision = find_ul_scheduler_decision(decision_id)) {
    auto candidate_it = std::find_if(
        decision->newtx_candidates.begin(),
        decision->newtx_candidates.end(),
        [rnti](const scheduler_ul_newtx_candidate& candidate) { return candidate.rnti == rnti; });

    if (candidate_it == decision->newtx_candidates.end()) {
      return;
    }

    if (decision->feasible_action_mask.size() < decision->newtx_candidates.size()) {
      decision->feasible_action_mask.resize(decision->newtx_candidates.size(), 0);
    }

    const size_t candidate_index =
        static_cast<size_t>(candidate_it - decision->newtx_candidates.begin());
    decision->feasible_action_mask[candidate_index] = 1;
  }
}
// habib added

void cell_metrics_handler::set_ul_newtx_candidate_grant_context(
    uint64_t                                      decision_id,
    rnti_t                                        rnti,
    const scheduler_ul_newtx_grant_context& grant_context)
{
  if (decision_id == 0) {
    return;
  }

  if (auto* decision = find_ul_scheduler_decision(decision_id)) {
    auto candidate_it = std::find_if(
        decision->newtx_candidates.begin(),
        decision->newtx_candidates.end(),
        [rnti](const scheduler_ul_newtx_candidate& candidate) { return candidate.rnti == rnti; });

    if (candidate_it != decision->newtx_candidates.end()) {
      candidate_it->grant_context = grant_context;
    }
  }
}

void cell_metrics_handler::set_ul_scheduler_state_snapshot(
    uint64_t                    decision_id,
    scheduler_ul_state_snapshot snapshot)
{
  if (decision_id == 0) {
    return;
  }

  if (auto* decision = find_ul_scheduler_decision(decision_id)) {
    decision->state_snapshot = std::move(snapshot);
  }
}
// habib added

void cell_metrics_handler::add_ul_selected_grant(
    uint64_t             decision_id,
    rnti_t               rnti,
    scheduler_ul_tx_type tx_type,
    slot_point           target_pusch_slot)
{
  if (decision_id == 0) {
    return;
  }

  if (auto* decision = find_ul_scheduler_decision(decision_id)) {
    decision->selected_grants.push_back(
        scheduler_ul_selected_grant{rnti, tx_type});

    pending_ul_grant_correlations.push_back(
        pending_ul_grant_correlation{decision_id, rnti, target_pusch_slot});
  }
}

void cell_metrics_handler::finish_ul_scheduler_decision(uint64_t decision_id)
{
  if (decision_id == 0) {
    return;
  }

  auto it = std::find_if(
      data.ul_scheduler_decisions.begin(),
      data.ul_scheduler_decisions.end(),
      [decision_id](const scheduler_ul_scheduler_decision& decision) {
        return decision.decision_id == decision_id;
      });

  if (it == data.ul_scheduler_decisions.end()) {
    return;
  }

  if (it->retx_candidates.empty() &&
      it->newtx_candidates.empty() &&
      it->selected_grants.empty()) {
    data.ul_scheduler_decisions.erase(it);
  }
// habib added
}

void cell_metrics_handler::handle_ue_creation(du_ue_index_t ue_index, rnti_t rnti, pci_t pcell_pci)
{
  if (not enabled()) {
    return;
  }

  ues.emplace(ue_index);
  ues[ue_index].rnti     = rnti;
  ues[ue_index].ue_index = ue_index;
  ues[ue_index].pci      = pcell_pci;
  rnti_to_ue_index_lookup.emplace(rnti, ue_index);

  if (pending_events.size() < pending_events.capacity()) {
    pending_events.push_back(
        scheduler_cell_event{last_slot_tx.without_hyper_sfn(), rnti, scheduler_cell_event::event_type::ue_add});
  } else {
    data.filtered_events_counter++;
  }
}

void cell_metrics_handler::handle_ue_reconfiguration(du_ue_index_t ue_index)
{
  if (not enabled()) {
    return;
  }
  if (pending_events.size() < pending_events.capacity()) {
    pending_events.push_back(scheduler_cell_event{
        last_slot_tx.without_hyper_sfn(), ues[ue_index].rnti, scheduler_cell_event::event_type::ue_reconf});
  } else {
    data.filtered_events_counter++;
  }
}

void cell_metrics_handler::handle_ue_deletion(du_ue_index_t ue_index)
{
  if (not enabled()) {
    return;
  }
  if (ues.contains(ue_index)) {
    rnti_t rnti = ues[ue_index].rnti;

    if (pending_events.size() < pending_events.capacity()) {
      pending_events.push_back(
          scheduler_cell_event{last_slot_tx.without_hyper_sfn(), rnti, scheduler_cell_event::event_type::ue_rem});
    } else {
      data.filtered_events_counter++;
    }

    rnti_to_ue_index_lookup.erase(rnti);
    ues.erase(ue_index);
  }
}

/*
void cell_metrics_handler::handle_rach_indication(const rach_indication_message& msg, slot_point sl_tx)
{
  if (not enabled()) {
    return;
  }
  unsigned slot_diff = sl_tx - msg.slot_rx;
  for (const auto& occ : msg.occasions) {
    data.nof_prach_preambles += occ.preambles.size();
    data.sum_prach_delay_slots += slot_diff * occ.preambles.size();
  }
}

void cell_metrics_handler::handle_msg3_crc_indication(const ul_crc_pdu_indication& crc_pdu)
{
  if (not enabled()) {
    return;
  }

  if (crc_pdu.tb_crc_success) {
    data.nof_msg3_ok++;
  } else {
    data.nof_msg3_nok++;
  }
}
*/
// habib added
void cell_metrics_handler::handle_rach_indication(
    const rach_indication_message& msg,
    slot_point                     sl_tx)
{
  // Reduced metrics mode: PRACH metrics are not collected.
  (void)msg;
  (void)sl_tx;
}

void cell_metrics_handler::handle_msg3_crc_indication(
    const ul_crc_pdu_indication& crc_pdu)
{
  // Reduced metrics mode: Msg3 metrics are not collected.
  (void)crc_pdu;
}
// habib added

void cell_metrics_handler::handle_crc_indication(slot_point                   sl_rx,
                                                 const ul_crc_pdu_indication& crc_pdu,
                                                 units::bytes                 tbs)
{
  if (not enabled()) {
    return;
  }
  if (ues.contains(crc_pdu.ue_index)) {
    auto& u = ues[crc_pdu.ue_index];
// habib added

    const scheduler_pusch_crc_status final_crc_status =
        crc_pdu.tb_crc_success ? scheduler_pusch_crc_status::pass
                               : scheduler_pusch_crc_status::fail;

    bool updated_current_report_allocation = false;

    for (auto it = u.data.pusch_allocations.rbegin();
         it != u.data.pusch_allocations.rend();
         ++it) {
      if (it->slot.without_hyper_sfn() == sl_rx &&
          it->harq_id == static_cast<unsigned>(crc_pdu.harq_id)) {
        it->crc_status = final_crc_status;
// habib added
        it->pusch_sinr_db = crc_pdu.ul_sinr_dB;
        it->pusch_rsrp_db = crc_pdu.ul_rsrp_dBFS;
// habib added
        updated_current_report_allocation = true;
        break;
      }
    }

    if (not updated_current_report_allocation) {
      auto pending_it = std::find_if(
          pending_pusch_outcomes.begin(),
          pending_pusch_outcomes.end(),
          [&](const pending_pusch_outcome& pending) {
            return pending.rnti == u.rnti &&
                   pending.target_pusch_slot.without_hyper_sfn() == sl_rx &&
                   pending.harq_id == static_cast<unsigned>(crc_pdu.harq_id);
          });

      if (pending_it != pending_pusch_outcomes.end()) {
        scheduler_late_crc_update update{};
        update.decision_id       = pending_it->decision_id;
        update.rnti              = pending_it->rnti;
        update.target_pusch_slot = pending_it->target_pusch_slot;
        update.harq_id           = pending_it->harq_id;
        update.crc_status        = final_crc_status;
// habib added
        update.pusch_sinr_db     = crc_pdu.ul_sinr_dB;
        update.pusch_rsrp_db     = crc_pdu.ul_rsrp_dBFS;
// habib added

        late_crc_updates_for_next_report.push_back(std::move(update));
        pending_pusch_outcomes.erase(pending_it);
      }
    }

// habib added
    u.data.count_crc_acks += crc_pdu.tb_crc_success ? 1 : 0;
    ++u.data.count_crc_pdus;
    if (crc_pdu.ul_sinr_dB.has_value()) {
      ++u.data.nof_pusch_snr_reports;
      u.data.sum_pusch_snrs += crc_pdu.ul_sinr_dB.value();
    }
    if (crc_pdu.ul_rsrp_dBFS.has_value()) {
      ++u.data.nof_pusch_rsrp_reports;
      u.data.sum_pusch_rsrp += crc_pdu.ul_rsrp_dBFS.value();
    }
    if (crc_pdu.tb_crc_success) {
      u.data.sum_ul_tb_bytes += tbs.value();
    }
    if (crc_pdu.time_advance_offset.has_value()) {
      u.data.ta.update(crc_pdu.time_advance_offset.value().to_seconds());
      u.data.pusch_ta.update(crc_pdu.time_advance_offset.value().to_seconds());
    }
    u.data.sum_crc_delay_slots += last_slot_tx.without_hyper_sfn() - sl_rx;
    //u.data.max_crc_delay_slots =
    //    std::max(static_cast<unsigned>(last_slot_tx.without_hyper_sfn() - sl_rx), u.data.max_crc_delay_slots);
  }
}

/*
void cell_metrics_handler::handle_srs_indication(const srs_indication::srs_indication_pdu& srs_pdu, unsigned ri)
{
  if (not enabled()) {
    return;
  }
  if (ues.contains(srs_pdu.ue_index)) {
    auto& u = ues[srs_pdu.ue_index];
    if (srs_pdu.time_advance_offset.has_value()) {
      u.data.ta.update(srs_pdu.time_advance_offset.value().to_seconds());
      u.data.srs_ta.update(srs_pdu.time_advance_offset.value().to_seconds());
      u.data.ul_ri.update(ri);
    }
  }
}
*/
// habib added
void cell_metrics_handler::handle_srs_indication(
    slot_point srs_slot,
    const srs_indication::srs_indication_pdu& srs_pdu,
    unsigned ri)
{
  if (not enabled()) {
    return;
  }

  if (not ues.contains(srs_pdu.ue_index)) {
    return;
  }

  auto& u = ues[srs_pdu.ue_index];

  scheduler_srs_report report{};

  // -------------------------------------------------------
  // SRS slot.
  // -------------------------------------------------------

  report.srs_slot = srs_slot;

  // -------------------------------------------------------
  // Wideband normalized SRS channel matrix.
  // -------------------------------------------------------

  report.channel_matrix = srs_pdu.channel_matrix;

  // -------------------------------------------------------
  // Raw PHY SRS estimator measurements.
  // -------------------------------------------------------

  report.srs_epre_db =
      srs_pdu.epre_dB;

  report.srs_rsrp_db =
      srs_pdu.rsrp_dB;

  report.srs_noise_variance =
      srs_pdu.noise_variance;

  // -------------------------------------------------------
  // Derive qualitative SRS SNR from normalized H matrix.
  //
  // H is already noise normalized.
  //
  // Therefore:
  //
  // SNR_linear ~= ||H||_F^2
  // -------------------------------------------------------

  const float frobenius_norm =
      srs_pdu.channel_matrix.frobenius_norm();

  const float snr_linear =
      frobenius_norm * frobenius_norm;

  if (std::isfinite(snr_linear) &&
      snr_linear > 0.0F) {

    report.srs_snr_db =
        convert_power_to_dB(snr_linear);
  }

  // -------------------------------------------------------
  // SRS timing advance.
  // -------------------------------------------------------

  if (srs_pdu.time_advance_offset.has_value()) {

    const float ta_seconds =
        srs_pdu.time_advance_offset.value().to_seconds();

    // Preserve existing aggregate TA metrics.
    u.data.ta.update(ta_seconds);
    u.data.srs_ta.update(ta_seconds);

    // Event-level value requested by you.
    report.srs_ta_ns =
        ta_seconds * 1e9F;
  }

  // -------------------------------------------------------
  // Preserve existing UL RI metric.
  //
  // This is NOT included in srs_reports[].
  // -------------------------------------------------------

  u.data.ul_ri.update(ri);

  // -------------------------------------------------------
  // Store this individual SRS event.
  // -------------------------------------------------------

  u.data.srs_reports.push_back(
      std::move(report));
}
// habib added

/*
void cell_metrics_handler::handle_pucch_sinr(ue_metric_context& u, float sinr)
{
  ++u.data.nof_pucch_snr_reports;
  u.data.sum_pucch_snrs += sinr;
}
*/

// habib added
void cell_metrics_handler::handle_pucch_sinr(
    ue_metric_context& u,
    float              sinr)
{
  // Reduced metrics mode: PUCCH SINR is not collected.
  (void)u;
  (void)sinr;
}
// habib added


void cell_metrics_handler::handle_csi_report(ue_metric_context& u, const csi_report_data& csi)
{
  // Add new CQI and RI observations if they are available in the CSI report.
  if (csi.first_tb_wideband_cqi.has_value()) {
    u.data.cqi.update(csi.first_tb_wideband_cqi->value());
  }
  //if (csi.ri.has_value()) {
  //  u.data.dl_ri.update(csi.ri->value());
  //}
}

void cell_metrics_handler::handle_uci_with_harq_ack(du_ue_index_t ue_index, slot_point sl_rx, bool pucch)
{
  if (ues.contains(ue_index)) {
    auto& u    = ues[ue_index];
    auto  diff = last_slot_tx.without_hyper_sfn() - sl_rx;
    if (pucch) {
      u.data.sum_pucch_harq_delay_slots += diff;
      u.data.max_pucch_harq_delay_slots = std::max(static_cast<unsigned>(diff), u.data.max_pucch_harq_delay_slots);
      ++u.data.count_pucch_harq_pdus;
    } else {
      u.data.sum_pusch_harq_delay_slots += diff;
      u.data.max_pusch_harq_delay_slots = std::max(static_cast<unsigned>(diff), u.data.max_pusch_harq_delay_slots);
      ++u.data.count_pusch_harq_pdus;
    }
  }
}

void cell_metrics_handler::handle_dl_harq_ack(du_ue_index_t ue_index, bool ack, units::bytes tbs)
{
  if (ues.contains(ue_index)) {
    auto& u = ues[ue_index];
    u.data.count_uci_harq_acks += ack ? 1 : 0;
    ++u.data.count_uci_harqs;
    if (ack) {
      u.data.sum_dl_tb_bytes += tbs.value();
    }
  }
}

void cell_metrics_handler::handle_harq_timeout(du_ue_index_t ue_index, bool is_dl)
{
  if (ues.contains(ue_index)) {
    auto& u = ues[ue_index];
    if (is_dl) {
      ++u.data.count_uci_harqs;
    } else {
      ++u.data.count_crc_pdus;
    }
  }
}

void cell_metrics_handler::handle_uci_pdu_indication(du_ue_index_t ue_index, const uci_action& action)
{
  if (not enabled()) {
    return;
  }
  if (ues.contains(ue_index)) {
    auto& u = ues[ue_index];

    if (action.ul_sinr_dB.has_value()) {
      handle_pucch_sinr(u, *action.ul_sinr_dB);
    }

    if (action.time_advance_offset.has_value()) {
      u.data.ta.update(action.time_advance_offset->to_seconds());
      u.data.pucch_ta.update(action.time_advance_offset->to_seconds());
    }

    if (not action.uci_valid and not action.harq_ack_bits.empty()) {
      switch (action.type) {
        case uci_action::pdu_type::pucch_f0f1:
          ++u.data.nof_pucch_f0f1_invalid_harqs;
          break;
        case uci_action::pdu_type::pucch_f2f3f4:
          ++u.data.nof_pucch_f2f3f4_invalid_harqs;
          break;
        default:
          ++u.data.nof_pusch_invalid_harqs;
      }
    }

    if (action.csi.has_value()) {
      if (action.csi->valid) {
        handle_csi_report(u, action.csi.value());
      } else {
        if (action.type == uci_action::pdu_type::pucch_f2f3f4) {
          ++u.data.nof_pucch_f2f3f4_invalid_csis;
        } else {
          ++u.data.nof_pusch_invalid_csis;
        }
      }
    }
  }
}

void cell_metrics_handler::handle_sr_indication(du_ue_index_t ue_index, slot_point sr_slot)
{
  if (ues.contains(ue_index)) {
    auto& u = ues[ue_index];
    if (not u.data.last_sr_slot.valid()) {
      u.data.last_sr_slot = sr_slot;
    }
    ++u.data.count_sr;
  }
}

void cell_metrics_handler::handle_ul_bsr_indication(const ul_bsr_indication_message& bsr)
{
  if (not enabled()) {
    return;
  }
  if (ues.contains(bsr.ue_index)) {
    auto& u = ues[bsr.ue_index];

    // Store last BSR.
    u.last_bsr = 0;
    // TODO: Handle different BSR formats.
    for (unsigned i = 0; i != bsr.reported_lcgs.size(); ++i) {
      u.last_bsr += bsr.reported_lcgs[i].nof_bytes;
    }
  }
}

void cell_metrics_handler::handle_ul_phr_indication(const ul_phr_indication_message& phr_ind)
{
  if (not enabled()) {
    return;
  }
  if (ues.contains(phr_ind.ue_index)) {
    auto& u = ues[phr_ind.ue_index];

    // Store last PHR.
    if (not phr_ind.phr.get_phr().empty()) {
      // Log the floor of the average of the PH interval.
      interval<int> rg = phr_ind.phr.get_phr().front().ph;
      u.last_phr       = (rg.start() + rg.stop()) / 2;
      auto diff        = last_slot_tx.without_hyper_sfn() - phr_ind.slot_rx;
      u.data.sum_ul_ce_delay_slots += diff;
      u.data.max_ul_ce_delay_slots = std::max(static_cast<unsigned>(diff), u.data.max_ul_ce_delay_slots);
      ++u.data.nof_ul_ces;
    }
  }
}

void cell_metrics_handler::handle_dl_buffer_state_indication(const dl_buffer_state_indication_message& dl_bs)
{
  if (not enabled()) {
    return;
  }
  if (ues.contains(dl_bs.ue_index)) {
    auto& u = ues[dl_bs.ue_index];

    // Store last DL buffer state.
    u.last_dl_bs[dl_bs.lcid] = dl_bs.bs;
  }
}

void cell_metrics_handler::handle_error_indication()
{
  ++data.error_indication_counter;
}

void cell_metrics_handler::handle_late_dl_harqs()
{
  ++data.nof_failed_pdsch_allocs_late_harqs;
}

void cell_metrics_handler::handle_late_ul_harqs()
{
  ++data.nof_failed_pusch_allocs_late_harqs;
}

void cell_metrics_handler::report_metrics()
{
  auto next_report = notifier.get_builder();

// habib added
  // Preserve unresolved PUSCHs before compute_report() moves/reset them.
  for (const ue_metric_context& ue : ues) {
    for (const scheduler_pusch_allocation& allocation : ue.data.pusch_allocations) {
      if (allocation.crc_status != scheduler_pusch_crc_status::pending ||
          not allocation.decision_id.has_value()) {
        continue;
      }

      const uint64_t decision_id = allocation.decision_id.value();

      const bool already_pending = std::any_of(
          pending_pusch_outcomes.begin(),
          pending_pusch_outcomes.end(),
          [&](const pending_pusch_outcome& pending) {
            return pending.decision_id == decision_id &&
                   pending.rnti == ue.rnti &&
                   pending.target_pusch_slot.count() == allocation.slot.count() &&
                   pending.harq_id == allocation.harq_id;
          });

      if (not already_pending) {
        pending_pusch_outcomes.push_back(
            pending_pusch_outcome{
                decision_id,
                ue.rnti,
                allocation.slot,
                allocation.harq_id});
      }
    }
  }

  next_report->late_crc_updates = std::move(late_crc_updates_for_next_report);
  late_crc_updates_for_next_report.clear();
// habib added


  const std::chrono::milliseconds report_period{data.nof_slots / last_slot_tx.nof_slots_per_subframe()};
  for (ue_metric_context& ue : ues) {
    // Compute statistics of the UE metrics and push the result to the report.
    next_report->ue_metrics.push_back(ue.compute_report(report_period, nof_slots_per_sf));
  }
// habib added
  next_report->ul_scheduler_decisions = std::move(data.ul_scheduler_decisions);
// habib added
  next_report->events.swap(pending_events);

  next_report->pci = cell_cfg.params.pci;
  // The window of slots for a report should be [start, stop) = [last_slot_tx + 1 - period, last_slot_tx + 1).
  // e.g. if the report period is 10, and we are at slot 0.9 (the last slot of the report), then the start slot is
  // 0.9 + 0.1 - 1.0 == 0.
  next_report->slot                  = last_slot_tx.without_hyper_sfn() + 1 - data.nof_slots;
  next_report->nof_slots             = data.nof_slots;
  //next_report->nof_error_indications = data.error_indication_counter;
  next_report->average_decision_latency =
      next_report->nof_slots > 0 ? data.decision_latency_sum / next_report->nof_slots : std::chrono::microseconds{0};
  //next_report->max_decision_latency      = data.max_decision_latency;
  //next_report->max_decision_latency_slot = data.max_decision_latency_slot;
  //next_report->latency_histogram         = data.decision_latency_hist;
  next_report->nof_prbs                  = cell_cfg.nof_dl_prbs; // TODO: to be removed from the report.
  //next_report->nof_dl_slots              = data.nof_dl_slots;
  //next_report->nof_ul_slots              = data.nof_ul_slots;
  next_report->nof_prach_preambles       = data.nof_prach_preambles;
  next_report->dl_grants_count           = data.nof_ue_pdsch_grants;
  next_report->ul_grants_count           = data.nof_ue_pusch_grants;
  next_report->nof_failed_pdcch_allocs   = data.nof_failed_pdcch_allocs;
  next_report->nof_failed_uci_allocs     = data.nof_failed_uci_allocs;
  next_report->nof_msg3_ok               = data.nof_msg3_ok;
  next_report->nof_msg3_nok              = data.nof_msg3_nok;
  next_report->avg_prach_delay_slots =
      data.nof_prach_preambles > 0
          ? std::optional{static_cast<float>(data.sum_prach_delay_slots) / static_cast<float>(data.nof_prach_preambles)}
          : std::nullopt;
  next_report->nof_failed_pdsch_allocs_late_harqs = data.nof_failed_pdsch_allocs_late_harqs;
  next_report->nof_failed_pusch_allocs_late_harqs = data.nof_failed_pusch_allocs_late_harqs;
  next_report->nof_filtered_events                = data.filtered_events_counter;
  // Note: PUCCH is only allocated on full UL slots.
  next_report->pucch_tot_rb_usage_avg =
      data.nof_ul_slots > 0 ? static_cast<float>(data.pucch_rbs_used) / data.nof_ul_slots : 0;
  if (cell_cfg.is_tdd()) {
    const float nof_tdd_periods_per_metric_report =
        static_cast<float>(next_report->nof_slots) /
        static_cast<float>(nof_slots_per_tdd_period(*cell_cfg.params.tdd_cfg));

    for (unsigned rb_count : ul_prbs_used_per_tdd_slot_idx) {
      const auto avg_nof_rbs =
          static_cast<unsigned>(std::round(static_cast<float>(rb_count) / nof_tdd_periods_per_metric_report));
      next_report->pusch_prbs_used_per_tdd_slot_idx.push_back(avg_nof_rbs);
    }

    for (unsigned rb_count : dl_prbs_used_per_tdd_slot_idx) {
      const auto avg_nof_rbs =
          static_cast<unsigned>(std::round(static_cast<float>(rb_count) / nof_tdd_periods_per_metric_report));
      next_report->pdsch_prbs_used_per_tdd_slot_idx.push_back(avg_nof_rbs);
    }
  }
  // Reset cell-wide metric counters.
  data = {};

  // Clear the PRB vectors for the next report.
  for (unsigned& rb_count : ul_prbs_used_per_tdd_slot_idx) {
    rb_count = 0;
  }
  for (unsigned& rb_count : dl_prbs_used_per_tdd_slot_idx) {
    rb_count = 0;
  }

  // Report all UE metrics in a batch.
  // Note: next_report will be reset afterwards. However, we prefer to first commit before fetching a new report.
  next_report.reset();
}

void cell_metrics_handler::handle_slot_result(slot_point_extended       sl_tx,
                                              const sched_result&       slot_result,
                                              std::chrono::microseconds slot_decision_latency)
{
  if (OCUDU_UNLIKELY(not last_slot_tx.valid())) {
    data.nof_slots = 1;
  } else {
    data.nof_slots += sl_tx - last_slot_tx;
  }
  last_slot_tx = sl_tx;

  data.nof_ue_pdsch_grants += slot_result.dl.ue_grants.size();
  for (const dl_msg_alloc& dl_grant : slot_result.dl.ue_grants) {
    auto it = rnti_to_ue_index_lookup.find(dl_grant.pdsch_cfg.rnti);
    if (it == rnti_to_ue_index_lookup.end()) {
      // UE not found.
      continue;
    }
    ue_metric_context& u = ues[it->second];
    for (const auto& cw : dl_grant.pdsch_cfg.codewords) {
      u.data.dl_mcs += cw.mcs_index.value();
      ++u.data.nof_dl_cws;
    }

    unsigned grant_prbs;
    if (dl_grant.pdsch_cfg.rbs.is_type0()) {
      grant_prbs = convert_rbgs_to_prbs(dl_grant.pdsch_cfg.rbs.type0(),
                                        {0, cell_cfg.nof_dl_prbs},
                                        get_nominal_rbg_size(cell_cfg.nof_dl_prbs, true))
                       .count();
    } else {
      grant_prbs = (dl_grant.pdsch_cfg.rbs.type1().length());
    }
    u.data.tot_dl_prbs_used += grant_prbs;
    if (not dl_prbs_used_per_tdd_slot_idx.empty()) {
      dl_prbs_used_per_tdd_slot_idx[last_slot_tx.count() % dl_prbs_used_per_tdd_slot_idx.size()] += grant_prbs;
    }
    u.last_dl_olla = dl_grant.context.olla_offset;
    if (u.data.last_pdsch_slot.valid()) {
      u.data.max_pdsch_distance_slots =
          std::max(static_cast<unsigned>(last_slot_tx.without_hyper_sfn() - u.data.last_pdsch_slot),
                   u.data.max_pdsch_distance_slots);
    }
    u.data.last_pdsch_slot = last_slot_tx.without_hyper_sfn();
  }
/*
  data.nof_ue_pusch_grants += slot_result.ul.puschs.size();
  for (const ul_sched_info& ul_grant : slot_result.ul.puschs) {
    auto it = rnti_to_ue_index_lookup.find(ul_grant.pusch_cfg.rnti);
    if (it == rnti_to_ue_index_lookup.end()) {
      // UE not found.
      continue;
    }
    unsigned grant_prbs;
    if (ul_grant.pusch_cfg.rbs.is_type0()) {
      grant_prbs = convert_rbgs_to_prbs(ul_grant.pusch_cfg.rbs.type0(),
                                        {0, cell_cfg.nof_dl_prbs},
                                        get_nominal_rbg_size(cell_cfg.nof_dl_prbs, true))
                       .count();
    } else {
      grant_prbs = (ul_grant.pusch_cfg.rbs.type1().length());
    }
    ues[it->second].data.tot_ul_prbs_used += grant_prbs;
    if (not ul_prbs_used_per_tdd_slot_idx.empty()) {
      ul_prbs_used_per_tdd_slot_idx[last_slot_tx.count() % ul_prbs_used_per_tdd_slot_idx.size()] += grant_prbs;
    }
    ue_metric_context& u = ues[it->second];
    u.data.ul_mcs += ul_grant.pusch_cfg.mcs_index.value();
    u.last_ul_olla = ul_grant.context.olla_offset;
    if (u.data.last_sr_slot.valid()) {
      unsigned sr_to_pusch_delay = last_slot_tx.without_hyper_sfn() - u.data.last_sr_slot;
      u.data.sum_sr_to_pusch_delay_slots += sr_to_pusch_delay;
      u.data.max_sr_to_pusch_delay_slots = std::max(sr_to_pusch_delay, u.data.max_sr_to_pusch_delay_slots);
      u.data.last_sr_slot.clear();
      u.data.count_handled_sr++;
    }
    ++u.data.nof_puschs;
    if (u.data.last_pusch_slot.valid()) {
      u.data.max_pusch_distance_slots =
          std::max(static_cast<unsigned>(last_slot_tx.without_hyper_sfn() - u.data.last_pusch_slot),
                   u.data.max_pusch_distance_slots);
    }
    u.data.last_pusch_slot = last_slot_tx.without_hyper_sfn();
  }
*/

/*
 //habib added - v1
  data.nof_ue_pusch_grants += slot_result.ul.puschs.size();

  for (const ul_sched_info& ul_grant : slot_result.ul.puschs) {
    auto it = rnti_to_ue_index_lookup.find(ul_grant.pusch_cfg.rnti);

    if (it == rnti_to_ue_index_lookup.end()) {
      // The allocation does not correspond to a currently tracked UE.
      continue;
    }

    ue_metric_context& u = ues[it->second];

    scheduler_pusch_allocation allocation{};

    // Store the radio-slot identity.
    allocation.slot = sl_tx;

    // Store PUSCH BWP information.
    const crb_interval bwp_crbs =
        ul_grant.pusch_cfg.bwp_cfg->crbs;

    allocation.bwp_start_crb =
        bwp_crbs.start();

    allocation.bwp_size_prbs =
        bwp_crbs.length();

    // Store frequency-hopping information.
    allocation.intra_slot_freq_hopping =
        ul_grant.pusch_cfg.intra_slot_freq_hopping;

    if (allocation.intra_slot_freq_hopping) {
      allocation.second_hop_rb_start =
          ul_grant.pusch_cfg.pusch_second_hop_prb;
    }

    if (ul_grant.pusch_cfg.rbs.is_type1()) {
      
     // * Resource-allocation Type 1:
     // *
     // * One contiguous VRB range. For uplink Type 1, OCUDU
     // * represents contiguous non-interleaved VRBs, so these
     // * indices correspond directly to PRB positions within
     // * the PUSCH BWP.
      
      allocation.allocation_type = 1;

      const vrb_interval& vrbs =
          ul_grant.pusch_cfg.rbs.type1();

      allocation.prb_ranges.push_back(
          scheduler_prb_range{
              static_cast<unsigned>(vrbs.start()),
              static_cast<unsigned>(vrbs.length())
          });

      allocation.nof_prbs =
          vrbs.length();

    } else {
      
    //  * Resource-allocation Type 0:
    //  *
    //  * The scheduler stores an RBG bitmap. Convert the selected
    //  * RBGs to the corresponding PRB bitmap, then store the
    //  * resulting contiguous PRB ranges.
    
      allocation.allocation_type = 0;

      const rbg_bitmap& rbgs =
          ul_grant.pusch_cfg.rbs.type0();

      // Preserve the selected RBG indices.
      for (size_t rbg_index : rbgs.get_bit_positions()) {
        allocation.rbg_indices.push_back(
            static_cast<unsigned>(rbg_index));
      }

      const nominal_rbg_size nominal_rbg =
          get_nominal_rbg_size(
              allocation.bwp_size_prbs,
              true);

      const prb_bitmap allocated_prbs =
          convert_rbgs_to_prbs(
              rbgs,
              bwp_crbs,
              nominal_rbg);

      // Convert the PRB bitmap into one or more contiguous ranges.
      for_each_interval(
          allocated_prbs,
          [&allocation](size_t start, size_t stop) {
            allocation.prb_ranges.push_back(
                scheduler_prb_range{
                    static_cast<unsigned>(start),
                    static_cast<unsigned>(stop - start)
                });
          });

      allocation.nof_prbs =
          allocated_prbs.count();
    }

    // Existing aggregate UE-level PRB counter.
    u.data.tot_ul_prbs_used +=
        allocation.nof_prbs;

    // Existing aggregate cell-level TDD slot-index counter.
    if (not ul_prbs_used_per_tdd_slot_idx.empty()) {
      const unsigned tdd_slot_index =
          last_slot_tx.count() %
          ul_prbs_used_per_tdd_slot_idx.size();

      ul_prbs_used_per_tdd_slot_idx[tdd_slot_index] +=
          allocation.nof_prbs;
    }

    // Store the detailed grant for this UE.
    u.data.pusch_allocations.push_back(
        std::move(allocation));

    // Keep the remaining existing UE metric updates.
    u.data.ul_mcs +=
        ul_grant.pusch_cfg.mcs_index.value();

    u.last_ul_olla =
        ul_grant.context.olla_offset;

    if (u.data.last_sr_slot.valid()) {
      unsigned sr_to_pusch_delay =
          last_slot_tx.without_hyper_sfn() -
          u.data.last_sr_slot;

      u.data.sum_sr_to_pusch_delay_slots +=
          sr_to_pusch_delay;

      u.data.max_sr_to_pusch_delay_slots =
          std::max(
              sr_to_pusch_delay,
              u.data.max_sr_to_pusch_delay_slots);

      u.data.last_sr_slot.clear();
      u.data.count_handled_sr++;
    }

    ++u.data.nof_puschs;

    if (u.data.last_pusch_slot.valid()) {
      u.data.max_pusch_distance_slots =
          std::max(
              static_cast<unsigned>(
                  last_slot_tx.without_hyper_sfn() -
                  u.data.last_pusch_slot),
              u.data.max_pusch_distance_slots);
    }

    u.data.last_pusch_slot =
        last_slot_tx.without_hyper_sfn();
  }
  //habib added - v1
*/
 
// habib added - v2

// habib added - reduced PUSCH metrics collection.
//
// Only collect the PUSCH information required by the reduced
// scheduler metrics output:
//   - slot identity
//   - allocation type
//   - BWP size
//   - allocated PRB ranges
//   - number of allocated PRBs
//   - aggregate UE PUSCH PRB usage
//   - UL MCS

for (const ul_sched_info& ul_grant : slot_result.ul.puschs) {
  auto it = rnti_to_ue_index_lookup.find(ul_grant.pusch_cfg.rnti);

  if (it == rnti_to_ue_index_lookup.end()) {
    // The allocation does not correspond to a currently tracked UE.
    continue;
  }

  ue_metric_context& u = ues[it->second];

  scheduler_pusch_allocation allocation{};

  // ------------------------------------------------------------------
  // Slot identity.
  // ------------------------------------------------------------------
  allocation.slot = sl_tx;

// habib added

  // Grant metadata already present in the finalized ul_sched_info.
  allocation.mcs        = ul_grant.pusch_cfg.mcs_index.value();
  allocation.tbs_bytes  = ul_grant.pusch_cfg.tb_size_bytes.value();
  allocation.harq_id    = static_cast<unsigned>(ul_grant.pusch_cfg.harq_id);
  allocation.crc_status = scheduler_pusch_crc_status::pending;

  // Correlate this PUSCH with the decision that selected the same RNTI
  // for this target slot. This vector persists across reporting boundaries.
  auto correlation_it = std::find_if(
      pending_ul_grant_correlations.begin(),
      pending_ul_grant_correlations.end(),
      [&ul_grant, sl_tx](const pending_ul_grant_correlation& correlation) {
        return correlation.rnti == ul_grant.pusch_cfg.rnti &&
               correlation.target_pusch_slot == sl_tx.without_hyper_sfn();
      });

  if (correlation_it != pending_ul_grant_correlations.end()) {
    allocation.decision_id = correlation_it->decision_id;
    pending_ul_grant_correlations.erase(correlation_it);
  }

// habib added
  // ------------------------------------------------------------------
  // PUSCH BWP information.
  //
  // We only expose the BWP size. bwp_crbs is still needed internally
  // for converting Type-0 RBG allocations into PRB positions.
  // ------------------------------------------------------------------
  const crb_interval bwp_crbs =
      ul_grant.pusch_cfg.bwp_cfg->crbs;

  allocation.bwp_size_prbs =
      bwp_crbs.length();

  // ------------------------------------------------------------------
  // Frequency-domain resource allocation.
  // ------------------------------------------------------------------
  if (ul_grant.pusch_cfg.rbs.is_type1()) {
    /*
     * Resource-allocation Type 1.
     *
     * Type 1 contains one contiguous VRB interval.
     * For this PUSCH allocation the VRB positions correspond to
     * PRB positions relative to the active PUSCH BWP.
     */
    allocation.allocation_type = 1;

    const vrb_interval& vrbs =
        ul_grant.pusch_cfg.rbs.type1();

    allocation.prb_ranges.push_back(
        scheduler_prb_range{
            static_cast<unsigned>(vrbs.start()),
            static_cast<unsigned>(vrbs.length())
        });

    allocation.nof_prbs =
        static_cast<unsigned>(vrbs.length());

  } else {
    /*
     * Resource-allocation Type 0.
     *
     * The scheduler represents the allocation as an RBG bitmap.
     * Convert the selected RBGs into a PRB bitmap and then convert
     * that bitmap into one or more contiguous PRB ranges.
     */
    allocation.allocation_type = 0;

    const rbg_bitmap& rbgs =
        ul_grant.pusch_cfg.rbs.type0();

    const nominal_rbg_size nominal_rbg =
        get_nominal_rbg_size(
            allocation.bwp_size_prbs,
            true);

    const prb_bitmap allocated_prbs =
        convert_rbgs_to_prbs(
            rbgs,
            bwp_crbs,
            nominal_rbg);

    // Convert the allocated PRB bitmap into contiguous ranges.
    for_each_interval(
        allocated_prbs,
        [&allocation](size_t start, size_t stop) {
          allocation.prb_ranges.push_back(
              scheduler_prb_range{
                  static_cast<unsigned>(start),
                  static_cast<unsigned>(stop - start)
              });
        });

    allocation.nof_prbs =
        static_cast<unsigned>(allocated_prbs.count());
  }

  // ------------------------------------------------------------------
  // Aggregate UE-level PUSCH PRB usage.
  //
  // Required for:
  //   tot_pusch_prbs_used
  // ------------------------------------------------------------------
  u.data.tot_ul_prbs_used +=
      allocation.nof_prbs;

  // ------------------------------------------------------------------
  // Store this detailed PUSCH grant.
  //
  // Required for:
  //   pusch_allocations[]
  // ------------------------------------------------------------------
  u.data.pusch_allocations.push_back(
      std::move(allocation));

  // ------------------------------------------------------------------
  // UL MCS accumulation.
  //
  // compute_report() uses this together with nof_puschs to calculate
  // the reported UL MCS.
  // ------------------------------------------------------------------
  u.data.ul_mcs +=
      ul_grant.pusch_cfg.mcs_index.value();

  ++u.data.nof_puschs;
}

// habib added - v2
/*
    // PUCCH resource usage.
    prb_bitmap pucch_prbs(cell_cfg.nof_ul_prbs);
    for (const auto& pucch : slot_result.ul.pucchs) {
      // Mark the PRBs used by this PUCCH.
      const prb_interval prbs = pucch.grant_prbs();
      pucch_prbs.fill(prbs.start(), prbs.stop());
      if (pucch.res->second_hop_prb.has_value()) {
        pucch_prbs.fill(*pucch.res->second_hop_prb, *pucch.res->second_hop_prb + prbs.length());
      }
    }
  data.pucch_rbs_used += pucch_prbs.count();
*/


  // Count DL and UL slots.
  //data.nof_dl_slots += slot_result.dl.nof_dl_symbols > 0;
  //data.nof_ul_slots += slot_result.ul.nof_ul_symbols > 0;

  // Process latency.
  data.decision_latency_sum += slot_decision_latency;
/*
  if (data.max_decision_latency < slot_decision_latency) {
    data.max_decision_latency      = slot_decision_latency;
    data.max_decision_latency_slot = last_slot_tx.without_hyper_sfn();
  }
  unsigned bin_idx = slot_decision_latency.count() / scheduler_cell_metrics::nof_usec_per_bin;
  bin_idx          = std::min(bin_idx, scheduler_cell_metrics::latency_hist_bins - 1);
  ++data.decision_latency_hist[bin_idx];

*/
  
  // Failed allocation attempts.
  data.nof_failed_pdcch_allocs += slot_result.failed_attempts.pdcch;
  data.nof_failed_uci_allocs += slot_result.failed_attempts.uci;
}

void cell_metrics_handler::push_result(slot_point_extended       sl_tx,
                                       const sched_result&       slot_result,
                                       std::chrono::microseconds slot_decision_latency)
{
  if (not enabled()) {
    return;
  }

  handle_slot_result(sl_tx, slot_result, slot_decision_latency);

  if (notifier.is_sched_report_required(sl_tx)) {
    // Prepare report and forward it to the notifier.
    report_metrics();
  }
}

void cell_metrics_handler::handle_cell_deactivation()
{
  // Commit whatever is pending for the report.
  report_metrics();
  last_slot_tx = {};
// habib added
  pending_ul_grant_correlations.clear();
// habib added
// habib added
  pending_pusch_outcomes.clear();
  late_crc_updates_for_next_report.clear();
// habib added
}

scheduler_ue_metrics
cell_metrics_handler::ue_metric_context::compute_report(std::chrono::milliseconds metric_report_period,
                                                        unsigned                  slots_per_sf)
{
  auto convert_slots_to_ms = [slots_per_sf](unsigned slots) {
    return static_cast<float>(slots) / static_cast<float>(slots_per_sf);
  };
  scheduler_ue_metrics ret{};
  ret.ue_index            = ue_index;
  ret.pci                 = pci;
  ret.rnti                = rnti;
  ret.cqi_stats           = data.cqi;
  ret.dl_ri_stats         = data.dl_ri;
  uint8_t mcs             = data.nof_dl_cws > 0 ? std::round(static_cast<float>(data.dl_mcs) / data.nof_dl_cws) : 0;
  ret.dl_mcs              = sch_mcs_index{mcs};
  mcs                     = data.nof_puschs > 0 ? std::round(static_cast<float>(data.ul_mcs) / data.nof_puschs) : 0;
  ret.ul_mcs              = sch_mcs_index{mcs};
  ret.tot_pdsch_prbs_used = data.tot_dl_prbs_used;
  ret.tot_pusch_prbs_used = data.tot_ul_prbs_used;
  
  //habib added
  // Transfer all individual PUSCH allocation records into the report.
  ret.pusch_allocations = std::move(data.pusch_allocations);
  //habib added

   // habib added
  ret.srs_reports = std::move(data.srs_reports);
  //habib added

  ret.dl_brate_kbps       = static_cast<double>(data.sum_dl_tb_bytes * 8U) / metric_report_period.count();
  ret.ul_brate_kbps       = static_cast<double>(data.sum_ul_tb_bytes * 8U) / metric_report_period.count();
  ret.dl_nof_ok           = data.count_uci_harq_acks;
  ret.dl_nof_nok          = data.count_uci_harqs - data.count_uci_harq_acks;
  ret.ul_nof_ok           = data.count_crc_acks;
  ret.ul_nof_nok          = data.count_crc_pdus - data.count_crc_acks;
  ret.pusch_snr_db        = data.nof_pusch_snr_reports > 0 ? data.sum_pusch_snrs / data.nof_pusch_snr_reports : 0;
  ret.pusch_rsrp_db       = data.nof_pusch_rsrp_reports > 0 ? data.sum_pusch_rsrp / data.nof_pusch_rsrp_reports
                                                            : -std::numeric_limits<float>::infinity();
  ret.ul_ri_stats         = data.ul_ri;
  ret.pucch_snr_db        = data.nof_pucch_snr_reports > 0 ? data.sum_pucch_snrs / data.nof_pucch_snr_reports : 0;
  ret.last_dl_olla        = last_dl_olla;
  ret.last_ul_olla        = last_ul_olla;
  ret.bsr                 = last_bsr;
  ret.sr_count            = data.count_sr;
  ret.dl_bs               = 0;
  for (const unsigned value : last_dl_bs) {
    ret.dl_bs += value;
  }
  ret.ta_stats                       = data.ta;
  ret.pusch_ta_stats                 = data.pusch_ta;
  ret.pucch_ta_stats                 = data.pucch_ta;
  ret.srs_ta_stats                   = data.srs_ta;
 


  ret.last_phr                       = last_phr;
  ret.max_pdsch_distance_ms          = convert_slots_to_ms(data.max_pdsch_distance_slots);
  ret.max_pusch_distance_ms          = convert_slots_to_ms(data.max_pusch_distance_slots);
  ret.nof_pucch_f0f1_invalid_harqs   = data.nof_pucch_f0f1_invalid_harqs;
  ret.nof_pucch_f2f3f4_invalid_harqs = data.nof_pucch_f2f3f4_invalid_harqs;
  ret.nof_pucch_f2f3f4_invalid_harqs = data.nof_pucch_f2f3f4_invalid_harqs;
  ret.nof_pucch_f2f3f4_invalid_csis  = data.nof_pucch_f2f3f4_invalid_csis;
  ret.nof_pusch_invalid_harqs        = data.nof_pusch_invalid_harqs;
  ret.nof_pusch_invalid_csis         = data.nof_pusch_invalid_csis;
  if (data.nof_ul_ces > 0) {
    ret.avg_ce_delay_ms = convert_slots_to_ms(data.sum_ul_ce_delay_slots) / static_cast<float>(data.nof_ul_ces);
    ret.max_ce_delay_ms = convert_slots_to_ms(data.max_ul_ce_delay_slots);
  }
  if (data.count_crc_pdus > 0) {
    ret.avg_crc_delay_ms = convert_slots_to_ms(data.sum_crc_delay_slots) / static_cast<float>(data.count_crc_pdus);
    ret.max_crc_delay_ms = convert_slots_to_ms(data.max_crc_delay_slots);
  }
  if (data.count_pusch_harq_pdus > 0) {
    ret.avg_pusch_harq_delay_ms =
        convert_slots_to_ms(data.sum_pusch_harq_delay_slots) / static_cast<float>(data.count_pusch_harq_pdus);
    ret.max_pusch_harq_delay_ms = convert_slots_to_ms(data.max_pusch_harq_delay_slots);
  }
  if (data.count_pucch_harq_pdus > 0) {
    ret.avg_pucch_harq_delay_ms =
        convert_slots_to_ms(data.sum_pucch_harq_delay_slots) / static_cast<float>(data.count_pucch_harq_pdus);
    ret.max_pucch_harq_delay_ms = convert_slots_to_ms(data.max_pucch_harq_delay_slots);
  }
  if (data.count_handled_sr > 0) {
    ret.avg_sr_to_pusch_delay_ms =
        convert_slots_to_ms(data.sum_sr_to_pusch_delay_slots) / static_cast<float>(data.count_handled_sr);
    ret.max_sr_to_pusch_delay_ms = convert_slots_to_ms(data.max_sr_to_pusch_delay_slots);
  }

  // Reset UE stats metrics on every report.
  reset();

  return ret;
}

void cell_metrics_handler::ue_metric_context::reset()
{
  // Note: for BSR and CQI we just keep the last without resetting the value at every slot.
  data = {};
}

cell_metrics_handler* scheduler_metrics_handler::add_cell(
    const cell_configuration&                                                      cell_cfg,
    const std::optional<sched_cell_configuration_request_message::metrics_config>& metrics_cfg)
{
  if (cells.contains(cell_cfg.cell_index)) {
    ocudulog::fetch_basic_logger("SCHED").warning("Cell={} already exists", fmt::underlying(cell_cfg.cell_index));
    return nullptr;
  }

  cells.emplace(cell_cfg.cell_index, std::make_unique<cell_metrics_handler>(cell_cfg, metrics_cfg));

  return cells[cell_cfg.cell_index].get();
}

void scheduler_metrics_handler::rem_cell(du_cell_index_t cell_index)
{
  cells.erase(cell_index);
}
