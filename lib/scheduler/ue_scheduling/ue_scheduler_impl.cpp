// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "ue_scheduler_impl.h"
#include "../logging/scheduler_metrics_handler.h"

// habib added
// habib added

using namespace ocudu;

ue_scheduler_impl::ue_scheduler_impl(const scheduler_ue_expert_config& expert_cfg_) :
  expert_cfg(expert_cfg_), logger(ocudulog::fetch_basic_logger("SCHED")), ue_db(expert_cfg), event_mng(ue_db)
{
}

ue_cell_scheduler* ue_scheduler_impl::do_add_cell(const ue_cell_scheduler_creation_request& params)
{
  cells.emplace(params.cell_index, *this, params);
  auto& cell = cells[params.cell_index];

  // Create a cell-specific UE event manager.
  cell.ev_mng = event_mng.add_cell(cell_creation_event{*params.cell_res_alloc,
                                                       cell.ue_cell_db,
                                                       cell.fallback_sched,
                                                       cell.uci_sched,
                                                       cell.slice_sched,
                                                       cell.srs_sched,
                                                       cell.uci_selector,
                                                       *params.cell_metrics,
                                                       *params.ev_logger});

  return &cell;
}

void ue_scheduler_impl::do_start_cell(du_cell_index_t cell_index)
{
  ocudu_assert(cells.contains(cell_index), "Cell reference not found in the scheduler");

  // Signal event manager that new events can be processed for this cell.
  cells[cell_index].ev_mng->start();
}

void ue_scheduler_impl::do_stop_cell(du_cell_index_t cell_index)
{
  ocudu_assert(cells.contains(cell_index), "Cell reference not found in the scheduler");
  auto& c = cells[cell_index];

  // Halt any pending events associated with this cell.
  cells[cell_index].ev_mng->stop();

  // Stop sub-schedulers.
  c.fallback_sched.stop();
  c.srs_sched.stop();
  c.uci_sched.stop();

  // Remove UEs from the UE repository associated with this cell.
  ue_db.handle_cell_deactivation(cell_index);
}

void ue_scheduler_impl::do_rem_cell(du_cell_index_t cell_index)
{
  ocudu_assert(cells.contains(cell_index), "Cell reference not found in the scheduler");

  do_stop_cell(cell_index);

  // Remove cell from UE scheduler.
  cells.erase(cell_index);
}

void ue_scheduler_impl::run_dl_sched_strategy(du_cell_index_t cell_index)
{
  auto& cell = cells[cell_index];

  // Schedule DL first.
  // Note: DL should be scheduled first so that the right DAI value is picked in DCI format 0_1.
  while (auto dl_slice_candidate = cell.slice_sched.get_next_dl_candidate()) {
    scheduler_policy& policy = cell.slice_sched.get_policy(dl_slice_candidate->id());
    cell.intra_slice_sched.dl_sched(dl_slice_candidate.value(), policy);
  }
}

bool ue_scheduler_impl::collect_next_ul_sched_batch(du_cell_index_t cell_index)
{
  auto& cell = cells[cell_index];

  while (auto ul_slice_candidate = cell.slice_sched.get_next_ul_candidate()) {
    scheduler_policy& policy = cell.slice_sched.get_policy(ul_slice_candidate->id());
    if (cell.intra_slice_sched.collect_ul_sched(ul_slice_candidate.value(), policy)) {
      return true;
    }
  }
  return false;
}

// habib added
bool ue_scheduler_impl::collect_next_ul_sched_batch(du_cell_index_t cell_index, uint64_t sync_id)
{
  auto& cell = cells[cell_index];

  while (auto ul_slice_candidate = cell.slice_sched.get_next_ul_candidate()) {
    scheduler_policy& policy = cell.slice_sched.get_policy(ul_slice_candidate->id());

    if (cell.intra_slice_sched.collect_ul_sched(ul_slice_candidate.value(), policy, sync_id)) {
      return true;
    }
  }

  return false;
}
// habib added - PART21_LOCAL_COLLECT_COMPLETE
// habib added

// habib added

[[maybe_unused]] static bool puxch_grant_sanitizer(cell_resource_allocator& cell_alloc, ocudulog::basic_logger& logger)
{
  const unsigned HARQ_SLOT_DELAY = 0;
  const auto&    slot_alloc      = cell_alloc[HARQ_SLOT_DELAY];

  if (not cell_alloc.cfg.is_ul_enabled(slot_alloc.slot)) {
    return true;
  }

  // Spans through the PUCCH grant list and check if there is any PUCCH grant scheduled for a UE that has a PUSCH.
  for (const auto& pucch : slot_alloc.result.ul.pucchs) {
    const auto* pusch_grant =
        std::find_if(slot_alloc.result.ul.puschs.begin(),
                     slot_alloc.result.ul.puschs.end(),
                     [&pucch](const ul_sched_info& pusch) { return pusch.pusch_cfg.rnti == pucch.crnti; });

    if (pusch_grant != slot_alloc.result.ul.puschs.end()) {
      logger.error("rnti={}: has both PUCCH and PUSCH grants scheduled at slot {}, PUCCH  format={} with uci_bits={}",
                   pucch.crnti,
                   slot_alloc.slot,
                   static_cast<unsigned>(pucch.format()),
                   pucch.uci_bits);
      return false;
    }
  }
  return true;
}

void ue_scheduler_impl::run_slot_impl(slot_point sl_tx)
{
  std::unique_lock<std::mutex> lock(cell_group_mutex, std::defer_lock);
  if (cells.size() > 1) {
    // Only mutex if the cell group has more than one cell (Carrier Aggregation case).
    lock.lock();
  }
  if (last_sl_ind == sl_tx) {
    // This slot has already been processed by a cell of the same cell group.
    return;
  }
  last_sl_ind = sl_tx;

  // Prepare every cell before running the scheduling strategies. In particular, periodic UCI and SRS resources must
  // be present before feasible UL newTx candidates are collected.
  for (auto& group_cell : cells) {
    // Process any pending events that are directed at UEs.
    group_cell.ev_mng->run_slot(sl_tx);

    // Update all UEs state.
    ue_db.slot_indication(sl_tx);

    // Schedule periodic UCI (SR and CSI) before any UL grants.
    group_cell.uci_sched.run_slot(*group_cell.cell_res_alloc);

    // Schedule periodic SRS before any UE grants.
    group_cell.srs_sched.run_slot(*group_cell.cell_res_alloc);

    // Run cell-specific SRB0 scheduler.
    group_cell.fallback_sched.run_slot(*group_cell.cell_res_alloc);

    // Update slice context and compute slice priorities.
    group_cell.slice_sched.slot_indication(sl_tx, *group_cell.cell_res_alloc);

    // Update intra-slice scheduler context.
    group_cell.intra_slice_sched.slot_indication(sl_tx);

    // Inject synthetic BSR for triggered UL grants due this slot.
    group_cell.trig_ul_sched.run_slot(sl_tx);
  }

  // Schedule DL first for every cell. DL must precede UL so that DCI format 0_1 observes the correct DAI.
  for (auto& group_cell : cells) {
    run_dl_sched_strategy(group_cell.cell_res_alloc->cfg.cell_index);
  }

  // Advance UL scheduling in synchronized rounds. Each active cell collects at most one feasible newTx builder batch
  // and then yields. Only after all cells have either yielded or exhausted their UL work can any cell derive its VRBs.
  // habib added - PART21_SAFE_SHARED_UL_ROUND_LOOP
  // Do not rendezvous during one-cell startup. Both cell groups first have to
  // be observed on the same slot; synchronization starts from a following slot.
  bool slot_sync_enabled =
      multicell_sync != nullptr &&
      multicell_sync->observe_slot(participant_tag, static_cast<uint64_t>(sl_tx.count()));

  uint32_t multicell_round_index = 1;

  while (true) {
    // Always reserve a valid local fallback ID. If the cross-group rendezvous
    // is unavailable or times out, the existing independent scheduler uses it.
    uint64_t sync_id = next_multicell_sync_id++;
    bool     round_synchronized = false;

    const uint64_t round_key =
        (static_cast<uint64_t>(sl_tx.count()) << 32U) |
        static_cast<uint64_t>(multicell_round_index);

    if (slot_sync_enabled) {
      const multicell_ul_rendezvous::begin_result begin =
          multicell_sync->begin_round(round_key);

      if (begin.synchronized) {
        sync_id            = begin.sync_id;
        round_synchronized = true;
      } else {
        slot_sync_enabled = false;
        logger.warning(
            "Multi-cell UL begin rendezvous timed out at slot={}; using independent scheduling for this slot",
            sl_tx);
      }
    }

    bool local_pending_ul_batch = false;

    for (auto& group_cell : cells) {
      if (collect_next_ul_sched_batch(group_cell.cell_res_alloc->cfg.cell_index, sync_id)) {
        local_pending_ul_batch = true;
      }
    }

    bool any_pending_ul_batch = local_pending_ul_batch;

    if (round_synchronized) {
      const multicell_ul_rendezvous::collection_result collected =
          multicell_sync->finish_collection(round_key, local_pending_ul_batch);

      if (collected.synchronized) {
        any_pending_ul_batch = collected.any_pending;
      } else {
        round_synchronized   = false;
        slot_sync_enabled    = false;
        any_pending_ul_batch = local_pending_ul_batch;

        logger.warning(
            "Multi-cell UL collect rendezvous timed out at slot={}; using independent scheduling for this slot",
            sl_tx);
      }
    }

    if (not any_pending_ul_batch) {
      break;
    }

    // Multi-cell AI input synchronization point.
    //
    // round_synchronized=true means:
    //   * all registered cell groups entered the same {slot, round},
    //   * all groups completed collect_ul_sched(),
    //   * no group has entered finalize_ul_sched() yet.
    //
    // This is the intended location for future cross-cell MARL inference.

    // Until MARL is connected, retain the existing OCUDU final VRB logic.
    for (auto& group_cell : cells) {
      if (group_cell.intra_slice_sched.has_pending_ul_sched()) {
        group_cell.intra_slice_sched.finalize_ul_sched();
      }
    }

    if (round_synchronized) {
      if (not multicell_sync->finish_finalization(round_key)) {
        slot_sync_enabled = false;

        logger.warning(
            "Multi-cell UL finalize rendezvous timed out at slot={}; synchronization will re-arm later",
            sl_tx);
      }
    }

    ++multicell_round_index;
  }
  // habib added - PART21_SAFE_SHARED_UL_ROUND_LOOP

  for (auto& group_cell : cells) {
    // The post processing is done for DL and UL slots.
    group_cell.intra_slice_sched.post_process_results();

    // Record UEs needing triggered UL grants based on the finalized DL grant list.
    group_cell.trig_ul_sched.process_dl_results(sl_tx, (*group_cell.cell_res_alloc)[0].result);

    // Update the UCI indication handler after the slot scheduling.
    group_cell.uci_selector.handle_result(sl_tx, (*group_cell.cell_res_alloc)[0].result);

    ocudu_sanity_check(puxch_grant_sanitizer(*group_cell.cell_res_alloc, logger),
                       "PUCCH and PUSCH found for the same UE in the same slot");
  }
}

ue_scheduler_impl::cell_context::cell_context(ue_scheduler_impl&                        parent_,
                                              const ue_cell_scheduler_creation_request& params) :
  parent(parent_),
  cell_res_alloc(params.cell_res_alloc),
  ue_cell_db(parent.ue_db.add_cell(params.cell_res_alloc->cfg, params.cell_metrics)),
  uci_sched(params.cell_res_alloc->cfg, *params.uci_alloc, parent.ue_db),
  fallback_sched(parent.expert_cfg,
                 params.cell_res_alloc->cfg,
                 *params.pdcch_sched,
                 *params.pucch_alloc,
                 *params.uci_alloc,
                 parent.ue_db),
  slice_sched(params.cell_res_alloc->cfg, parent.ue_db),
  intra_slice_sched(parent.expert_cfg,
                    parent.ue_db,
                    *params.pdcch_sched,
                    *params.uci_alloc,
                    *params.srs_alloc,
                    *params.cell_res_alloc,
                    *params.cell_metrics,
                    ocudulog::fetch_basic_logger("SCHED")),
  srs_sched(params.cell_res_alloc->cfg, parent.ue_db),
  trig_ul_sched(parent.ue_db, params.cell_res_alloc->cfg.cell_index),
  uci_selector(*this)
{
}

ue_scheduler_impl::cell_context::~cell_context()
{
  parent.ue_db.rem_cell(ue_cell_db.cell_index());
}
