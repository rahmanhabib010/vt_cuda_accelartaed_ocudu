// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "ocudu/ran/du_types.h"
#include "ocudu/ran/pci.h"
#include "ocudu/ran/rnti.h"
#include "ocudu/ran/sch/sch_mcs.h"
#include "ocudu/ran/slot_point.h"
#include "ocudu/ran/slot_point_extended.h"
#include "ocudu/support/math/stats.h"
#include "ocudu/support/zero_copy_notifier.h"
#include <array>
#include <chrono>
#include <optional>
#include <vector>

//habib added
#include "ocudu/ran/srs/srs_channel_matrix.h"
//habib added

namespace ocudu {

// habib added
/// One contiguous PRB range inside the PUSCH BWP.
///
/// The represented interval is:
///   [rb_start, rb_start + rb_length)
struct scheduler_prb_range {
  /// Starting PRB index relative to the PUSCH BWP.
  unsigned rb_start = 0;

  /// Number of contiguous PRBs in this range.
  unsigned rb_length = 0;
};

// habib added
/// CRC state associated with a reported PUSCH transmission.
enum class scheduler_pusch_crc_status { pending, pass, fail };

// habib added
/// Detailed information about one PUSCH allocation for one UE.
struct scheduler_pusch_allocation {
  /// Transmission slot containing this PUSCH grant.
  slot_point_extended slot;
// habib added

  /// Scheduler decision that produced this grant.
  /// Empty for PUSCHs created outside the regular intra-slice UE scheduler.
  std::optional<uint64_t> decision_id;
// habib added

  /// Resource-allocation type:
  ///   0 = RBG bitmap, potentially non-contiguous.
  ///   1 = contiguous PRB/VRB interval.
  unsigned allocation_type = 1;

  /// Starting CRB of the active PUSCH BWP.
  unsigned bwp_start_crb = 0;

  /// Size of the active PUSCH BWP in PRBs.
  unsigned bwp_size_prbs = 0;

  /// Total number of PRBs allocated by this grant.
  unsigned nof_prbs = 0;

  /// Frequency-domain PRB ranges relative to the PUSCH BWP.
  ///
  /// Type 1 normally contains exactly one range.
  /// Type 0 can contain one or more ranges.
  std::vector<scheduler_prb_range> prb_ranges;

  /// RBG indices selected by a Type-0 allocation.
  ///
  /// Empty for Type 1.
  std::vector<unsigned> rbg_indices;

  /// Whether intra-slot PUSCH frequency hopping is enabled.
  bool intra_slot_freq_hopping = false;

  /// First PRB after the intra-slot frequency hop.
  ///
  /// Present only when intra_slot_freq_hopping is true.
  std::optional<unsigned> second_hop_rb_start;
// habib added

  /// Exact MCS used by this PUSCH.
  unsigned mcs = 0;

  /// Transport block size in bytes.
  uint64_t tbs_bytes = 0;

  /// UL HARQ process identifier.
  unsigned harq_id = 0;

  /// CRC state for this PUSCH.
  scheduler_pusch_crc_status crc_status = scheduler_pusch_crc_status::pending;
// habib added
};
// habib added

//habib added
/// Detailed information associated with one received UE SRS.
struct scheduler_srs_report {
  /// Slot where this SRS was received.
  slot_point srs_slot;

  /// Wideband SRS channel matrix.
  srs_channel_matrix channel_matrix;

  /// Wideband SRS EPRE in dB.
  std::optional<float> srs_epre_db;

  /// Wideband SRS RSRP in dB.
  std::optional<float> srs_rsrp_db;

  /// Wideband SRS noise variance, linear quantity.
  std::optional<float> srs_noise_variance;

  /// Qualitative SRS SNR derived from the normalized channel matrix.
  std::optional<float> srs_snr_db;

  /// SRS-derived timing advance in ns.
  std::optional<float> srs_ta_ns;
};
//habib added

/// \brief Snapshot of the metrics for a UE.
// habib added
/// Whether an actually selected UL grant is a new transmission or HARQ retransmission.
enum class scheduler_ul_tx_type { newtx, retx };

struct scheduler_ul_retx_candidate {
  rnti_t   rnti = rnti_t::INVALID_RNTI;
  unsigned harq_id = 0;
};

struct scheduler_ul_newtx_candidate {
  rnti_t   rnti = rnti_t::INVALID_RNTI;
  uint64_t pending_bytes_at_decision = 0;
  double   priority = 0.0;
  unsigned rank = 0;
};

struct scheduler_ul_selected_grant {
  rnti_t               rnti = rnti_t::INVALID_RNTI;
  scheduler_ul_tx_type tx_type = scheduler_ul_tx_type::newtx;
};

struct scheduler_ul_scheduler_decision {
  uint64_t            decision_id = 0;
  slot_point_extended decision_slot;
  slot_point_extended target_pusch_slot;
  unsigned            k2 = 0;

  std::vector<scheduler_ul_retx_candidate>  retx_candidates;
  std::vector<scheduler_ul_newtx_candidate> newtx_candidates;
  std::vector<scheduler_ul_selected_grant>  selected_grants;
};

// habib added
struct scheduler_ue_metrics {
  /// UE index in the DU for this UE.
  du_ue_index_t ue_index;
  /// PCI of the UE's PCell.
  pci_t pci;
  /// Currently used C-RNTI for this UE.
  rnti_t rnti;
  /// Average MCS index used for DL grants.
  sch_mcs_index dl_mcs;
  /// Number of RBs used for PDSCH.
  unsigned tot_pdsch_prbs_used;
  /// \brief Experienced MAC DL bit rate in kbps, considering the size of the allocated MAC DL PDUs for which a positive
  /// HARQ-ACK was received.
  double dl_brate_kbps;
  /// Number of positive HARQ-ACKs received.
  unsigned dl_nof_ok;
  /// Number of detected HARQ NACKs or HARQ-ACK misdetections.
  unsigned dl_nof_nok;
  /// SNR in dB estimated for the PUSCH.
  float pusch_snr_db;
  /// RSRP in dB estimated for the PUSCH.
  float pusch_rsrp_db;
  /// SNR in dB estimated for the PUCCH.
  float pucch_snr_db;
  /// Average MCS index used for UL grants.
  sch_mcs_index ul_mcs;
  /// Number of RBs used for PUSCH.
  unsigned tot_pusch_prbs_used;

  // habib added
  /// Individual PUSCH allocations during this metrics report window.
  std::vector<scheduler_pusch_allocation> pusch_allocations;
  // habib added

  //habib added
  /// Detailed SRS reports collected in this metrics reporting interval.
  std::vector<scheduler_srs_report> srs_reports;
  //habib added


  /// \brief Experienced MAC UL bit rate in kbps, considering the size of the allocated MAC UL PDUs for which the
  /// respective CRC was decoded.
  double ul_brate_kbps;
  /// Number of positive CRC PDU indications received.
  unsigned ul_nof_ok;
  /// Number of negative CRC PDU indications received.
  unsigned ul_nof_nok;
  /// Sum of the last UL buffer status reports (BSRs) of all logical channel groups.
  unsigned bsr;
  /// Number of scheduling requests detected.
  unsigned sr_count;
  /// Sum of the last DL buffer occupancy reports of all logical channels.
  unsigned dl_bs;
  /// Invalid UCI reception metrics.
  /// @{
  unsigned nof_pucch_f0f1_invalid_harqs;
  unsigned nof_pucch_f2f3f4_invalid_harqs;
  unsigned nof_pucch_f2f3f4_invalid_csis;
  unsigned nof_pusch_invalid_harqs;
  unsigned nof_pusch_invalid_csis;
  /// @}
  /// Delay metrics.
  /// @{
  std::optional<float> avg_ce_delay_ms;
  std::optional<float> max_ce_delay_ms;
  std::optional<float> avg_crc_delay_ms;
  std::optional<float> max_crc_delay_ms;
  std::optional<float> avg_pusch_harq_delay_ms;
  std::optional<float> max_pusch_harq_delay_ms;
  std::optional<float> avg_pucch_harq_delay_ms;
  std::optional<float> max_pucch_harq_delay_ms;
  std::optional<float> avg_sr_to_pusch_delay_ms;
  std::optional<float> max_sr_to_pusch_delay_ms;
  /// @}
  std::optional<float>    last_dl_olla;
  std::optional<float>    last_ul_olla;
  std::optional<int>      last_phr;
  std::optional<unsigned> max_pdsch_distance_ms;
  std::optional<unsigned> max_pusch_distance_ms;
  /// Time advance statistics in seconds.
  sample_statistics<float> ta_stats;
  sample_statistics<float> pusch_ta_stats;
  sample_statistics<float> pucch_ta_stats;
  sample_statistics<float> srs_ta_stats;
  /// CQI statistics over the metrics report interval.
  sample_statistics<unsigned> cqi_stats;
  /// DL RI statistics over the metrics report interval.
  sample_statistics<unsigned> dl_ri_stats;
  /// UL RI statistics over the metrics report interval.
  sample_statistics<unsigned> ul_ri_stats;
};

/// \brief Event that occurred in the cell of the scheduler.
struct scheduler_cell_event {
  enum class event_type { ue_add, ue_reconf, ue_rem };

  slot_point slot;
  rnti_t     rnti = rnti_t::INVALID_RNTI;
  event_type type;
};

inline const char* sched_event_to_string(scheduler_cell_event::event_type ev)
{
  static constexpr std::array<const char*, 3> names = {"ue_add", "ue_reconf", "ue_rem"};
  return names[std::min(static_cast<size_t>(ev), names.size() - 1)];
}




/// \brief Snapshot of the metrics for a cell and its UEs.
struct scheduler_cell_metrics {
  /// Latency histogram number of bins.
  static constexpr unsigned latency_hist_bins = 10;
  /// Distance between histogram bins.
  static constexpr unsigned nof_usec_per_bin = 50;

  /// Cell PCI for which the metrics are reported.
  pci_t pci;
  /// Slot at which the metrics started being tracked for this report.
  slot_point slot;
  /// Number of slots accounted for in this report.
  unsigned nof_slots = 0;
  /// Number of cell PRBs.
  unsigned nof_prbs = 0;
  /// Number of downlink slots.
  unsigned nof_dl_slots = 0;
  /// Number of uplink slots (only full uplink slots counted for now).
  unsigned nof_ul_slots = 0;
  /// Number of PRACH preambles detected.
  unsigned nof_prach_preambles = 0;
  /// Counter of UE PDSCH grants (RARs, SIBs and Paging are not considered).
  unsigned dl_grants_count = 0;
  /// Counter of UE PUSCH grants.
  unsigned ul_grants_count = 0;
  /// Number of failed PDCCH allocation attempts.
  unsigned nof_failed_pdcch_allocs = 0;
  /// Number of failed UCI allocation attempts.
  unsigned nof_failed_uci_allocs = 0;
  /// Number of MSG3s.
  unsigned nof_msg3_ok = 0;
  /// Number of MSG3 KOs.
  unsigned nof_msg3_nok = 0;
  /// Average PRACH delay in slots.
  std::optional<float> avg_prach_delay_slots;
  /// Number of failed PDSCH allocations due to late HARQs.
  unsigned nof_failed_pdsch_allocs_late_harqs = 0;
  /// Number of failed PUSCH allocations due to late HARQs.
  unsigned nof_failed_pusch_allocs_late_harqs = 0;
  /// Number of UE events not reported because the maximum number of events was reached.
  unsigned nof_filtered_events = 0;
  /// Average number of RBs used for PUCCH per UL slot.
  float pucch_tot_rb_usage_avg = 0.0f;

  unsigned                                nof_error_indications = 0;
  std::chrono::microseconds               average_decision_latency{0};
  std::chrono::microseconds               max_decision_latency{0};
  slot_point                              max_decision_latency_slot;
  std::array<unsigned, latency_hist_bins> latency_histogram{0};
  /// Average number of RBs used for PUSCH per slot index in the TDD pattern.
  std::vector<unsigned>             pusch_prbs_used_per_tdd_slot_idx;
  std::vector<unsigned>             pdsch_prbs_used_per_tdd_slot_idx;
// habib added
  std::vector<scheduler_ul_scheduler_decision> ul_scheduler_decisions;
// habib added
  std::vector<scheduler_cell_event> events;
  std::vector<scheduler_ue_metrics> ue_metrics;
};

/// Scheduler metrics report for all active cells of the DU.
struct scheduler_metrics_report {
  std::vector<scheduler_cell_metrics> cells;
};

/// \brief Notifier interface used by scheduler to report metrics.
class scheduler_metrics_notifier
{
public:
  virtual ~scheduler_metrics_notifier() = default;

  /// \brief This method will be called periodically by the scheduler to report the latest UE metrics statistics.
  virtual void report_metrics(const scheduler_cell_metrics& report) = 0;
};

/// Interface used by the scheduler to determine whether a new metric report is required.
class scheduler_cell_metrics_notifier : public zero_copy_notifier<scheduler_cell_metrics>
{
public:
  /// Check whether a new metric report is required given the current slot.
  virtual bool is_sched_report_required(slot_point_extended sl_tx) const = 0;
};

} // namespace ocudu
