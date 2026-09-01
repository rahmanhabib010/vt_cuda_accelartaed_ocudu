// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "../slicing/inter_slice_scheduler.h"
#include "../srs/srs_scheduler_impl.h"
#include "../uci_scheduling/uci_indication_selector.h"
#include "../uci_scheduling/uci_scheduler_impl.h"
#include "../ue_context/ue_repository.h"
#include "intra_slice_scheduler.h"
#include "triggered_ul_grant_scheduler.h"
#include "ue_cell_grid_allocator.h"
#include "ue_event_manager.h"
#include "ue_fallback_scheduler.h"
#include "ue_scheduler.h"
// habib added - PART21 safe multi-cell UL rendezvous
#include "multicell_ul_rendezvous.h"
// habib added - PART21 safe multi-cell UL rendezvous
#include "ocudu/scheduler/config/scheduler_expert_config.h"
#include <mutex>

namespace ocudu {

/// \brief Interface of data scheduler that is used to allocate UE DL and UL grants in a given slot.
/// The data_scheduler object will be common to all cells and slots.
class ue_scheduler_impl final : public ue_scheduler
{
public:
  explicit ue_scheduler_impl(const scheduler_ue_expert_config& expert_cfg_);

// habib added - PART21 safe multi-cell UL rendezvous
  /// Connect this DU cell-group scheduler to the scheduler-wide rendezvous.
  void set_multicell_rendezvous(multicell_ul_rendezvous& sync, unsigned participant_tag_)
  {
    multicell_sync   = &sync;
    participant_tag = participant_tag_;

    // Fallback IDs occupy a group-specific high range and cannot accidentally
    // look like a successfully shared sync_id.
    next_multicell_sync_id =
        (static_cast<uint64_t>(participant_tag + 1U) * 1000000000ULL) + 100001ULL;

    multicell_sync->register_participant(participant_tag);
  }
// habib added - PART21 safe multi-cell UL rendezvous

private:
  ue_cell_scheduler* do_add_cell(const ue_cell_scheduler_creation_request& params) override;

  void do_start_cell(du_cell_index_t cell_index);
  void do_stop_cell(du_cell_index_t cell_index);

  void do_rem_cell(du_cell_index_t cell_index) override;

  void run_slot_impl(slot_point sl_tx);

  void run_dl_sched_strategy(du_cell_index_t cell_index);

  /// Advances UL scheduling for one cell until it either reaches the newTx synchronization point or runs out of work.
  /// Returns true if the cell has a pending batch that must be finalized.
  bool collect_next_ul_sched_batch(du_cell_index_t cell_index);
// habib added
  bool collect_next_ul_sched_batch(du_cell_index_t cell_index, uint64_t sync_id);
// habib added

  struct cell_context final : public ue_cell_scheduler, public uci_indication_timeout_notifier {
    ue_scheduler_impl& parent;

    cell_resource_allocator* cell_res_alloc;

    /// Repository of UEs for this cell.
    ue_cell_repository& ue_cell_db;

    /// PUCCH scheduler.
    uci_scheduler_impl uci_sched;

    /// Fallback scheduler.
    ue_fallback_scheduler fallback_sched;

    /// Slice scheduler.
    inter_slice_scheduler slice_sched;

    /// Intra-slice scheduler.
    intra_slice_scheduler intra_slice_sched;

    /// SRS scheduler
    srs_scheduler_impl srs_sched;

    /// Triggered UL grant sub-scheduler.
    triggered_ul_grant_scheduler trig_ul_sched;

    /// Handler of UCI indications.
    uci_indication_selector uci_selector;

    /// Cell-specific event manager.
    std::unique_ptr<ue_cell_event_manager> ev_mng;

    cell_context(ue_scheduler_impl& parent, const ue_cell_scheduler_creation_request& params);
    ~cell_context() override;

    void run_slot(slot_point sl_tx) override { parent.run_slot_impl(sl_tx); }

    void handle_error_indication(slot_point sl_tx, scheduler_slot_handler::error_outcome event) override
    {
      ev_mng->handle_error_indication(sl_tx, event);
    }

    void handle_slice_reconfiguration_request(const du_cell_slice_reconfig_request& slice_reconf_req) override
    {
      ev_mng->handle_slice_reconfiguration_request(slice_reconf_req);
    }

    scheduler_feedback_handler&                   get_feedback_handler() override { return *ev_mng; }
    scheduler_cell_positioning_handler&           get_positioning_handler() override { return *ev_mng; }
    scheduler_dl_buffer_state_indication_handler& get_dl_buffer_state_indication_handler() override { return *ev_mng; }
    sched_ue_configuration_handler&               get_ue_configurator() override { return *ev_mng; }

    void start() override { parent.do_start_cell(cell_res_alloc->cfg.cell_index); }

    void stop() override { parent.do_stop_cell(cell_res_alloc->cfg.cell_index); }

    void on_timeout(slot_point sl_rx, rnti_t crnti, const uci_action& action) override
    {
      ev_mng->handle_uci_indication_timeout(sl_rx, crnti, action);
    }
  };

  const scheduler_ue_expert_config& expert_cfg;
  ocudulog::basic_logger&           logger;

  // List of cells of the UE scheduler.
  slotted_array<cell_context, MAX_NOF_DU_CELLS> cells;

  /// Repository of created UEs.
  ue_repository ue_db;

  /// Processor of UE input events.
  ue_event_manager event_mng;

  // Mutex to lock cells of the same cell group (when CA enabled) for joint carrier scheduling
  std::mutex cell_group_mutex;

// habib added
  uint64_t next_multicell_sync_id = 100001;
// habib added

// habib added - PART21 safe multi-cell UL rendezvous
  multicell_ul_rendezvous* multicell_sync = nullptr;
  unsigned                 participant_tag = 0;
// habib added - PART21 safe multi-cell UL rendezvous

  // Last slot run.
  slot_point last_sl_ind;
};

} // namespace ocudu
