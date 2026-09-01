#!/usr/bin/env python3
"""
Reduced OCUDU multi-cell scheduler metrics collector.

Collects the synchronized UL metrics enabled by the current code changes:
  * common sync_id and top-level decision/target timing
  * per-cell immutable UL state snapshot
  * reTx/newTx candidates and feasible_action_mask
  * selected grants
  * PUSCH allocation/PRB ranges with sync_id
  * late CRC updates with sync_id
  * UE PUSCH RSRP/SINR and existing reduced UE metrics
  * SRS reports

Default output:
    ~/ocudu_metrics/metrics_by_cell.jsonl
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import websocket


WS_URL = os.environ.get("WS_URL", "127.0.0.1:8001").strip()
WS_ENDPOINT = WS_URL if WS_URL.startswith(("ws://", "wss://")) else f"ws://{WS_URL}"

OUTPUT_FILE = Path(
    os.environ.get(
        "OUTPUT_FILE",
        "~/ocudu_metrics/metrics_by_cell.jsonl",
    )
).expanduser()

# Keep disabled for high-rate collection (e.g. 10 ms reporting).
# Set PRINT_RECORDS=1 only when you want to inspect every grouped report.
PRINT_RECORDS = os.environ.get("PRINT_RECORDS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics to keep in the final JSONL file.
# ---------------------------------------------------------------------------

CELL_METRICS_KEEP = (
    "average_latency",
)

UE_METRICS_KEEP = (
    "avg_ce_delay",
    "avg_crc_delay",
    "bsr",
    "cqi",
    "dl_brate",
    "dl_bs",
    "dl_mcs",
    "pusch_rsrp_db",
    "pusch_snr_db",
    "pusch_ta_ns",
    "rnti",
    "srs_ta_ns",
    "ta_ns",
    "tot_pusch_prbs_used",
    "ue",
    "ul_brate",
    "ul_mcs",
    "ul_nof_nok",
    "ul_nof_ok",
    "ue_ul_nok",
    "ue_ul_ok",
    "ul_ri",
)

# habib added
# Preserve the scheduler-decision linkage and finalized PUSCH grant metadata
# added by the UL scheduler instrumentation. PRB information remains in the
# existing fields below and is not duplicated anywhere else.
PUSCH_ALLOCATION_KEEP = (
    "sync_id",
    "allocation_type",
    "bwp_size_prbs",
    "hyper_sfn",
    "nof_prbs",
    "sfn",
    "slot_index",
    "mcs",
    "tbs_bytes",
    "harq_id",
    "crc_status",
    # habib added
    "pusch_sinr_db",
    "pusch_rsrp_db",
    # habib added
)
# habib added

PRB_RANGE_KEEP = (
    "rb_start",
    "rb_stop",
)

# SRS report fields to preserve in the reduced JSONL output.
SRS_REPORT_KEEP = (
    "srs_epre_db",
    "srs_rsrp_db",
    "srs_noise_variance",
    "srs_snr_db",
    "srs_ta_ns",
)

SRS_SLOT_KEEP = (
    "sfn",
    "slot_index",
    "slot_count",
)

SRS_CHANNEL_COEFFICIENT_KEEP = (
    "real",
    "imag",
    "magnitude",
    "phase_rad",
)


# Maps the position in the scheduler "cells" list to its PCI.
# This is only a fallback. The reduced scheduler report now carries
# PCI directly inside cell_metrics.
cell_index_to_pci: dict[int, int] = {}


def parse_static_pci_map() -> None:
    """
    Optional manual fallback mapping.

    Example:
        CELL_PCI_MAP="0:1,1:2"

    means:
        scheduler cells[0] -> PCI 1
        scheduler cells[1] -> PCI 2
    """
    mapping = os.environ.get("CELL_PCI_MAP", "").strip()

    if not mapping:
        return

    for item in mapping.split(","):
        index_text, pci_text = item.split(":", maxsplit=1)
        cell_index_to_pci[int(index_text)] = int(pci_text)


def pci_key(pci: Any) -> str:
    """Format PCI 1 as PCI=01."""
    try:
        return f"PCI={int(pci):02d}"
    except (TypeError, ValueError):
        return f"PCI={pci}"


def rnti_key(rnti: Any) -> str:
    """
    OCUDU reports RNTI as an integer in JSON.
    Represent it in hexadecimal, matching the gNB console.
    """
    try:
        return f"RNTI={int(rnti):04X}"
    except (TypeError, ValueError):
        return f"RNTI={rnti}"


# habib added
def rnti_hex(rnti: Any) -> str:
    """Format scheduler-decision RNTIs as bare hexadecimal strings, e.g. 4604."""
    try:
        return f"{int(rnti):04X}"
    except (TypeError, ValueError):
        return str(rnti)


def extract_sync_id(record: dict[str, Any]) -> Any:
    """
    Return the common multi-cell synchronization ID.

    Final OCUDU JSON exposes ``sync_id`` directly.  The decision_id fallback is
    retained so the collector also tolerates intermediate builds where the
    internal correlation ID is encoded as:

        (sync_id << 16) | local_decision_id
    """
    sync_id = record.get("sync_id")
    if sync_id is not None:
        return sync_id

    decision_id = record.get("decision_id")
    if decision_id is None:
        return None

    try:
        decision_id_int = int(decision_id)
    except (TypeError, ValueError):
        return decision_id

    encoded_sync_id = decision_id_int >> 16
    return encoded_sync_id if encoded_sync_id != 0 else decision_id_int


def filter_ul_scheduler_decision(
    decision: dict[str, Any],
) -> dict[str, Any]:
    """
    Keep UL scheduler decision information, including the immutable
    MARL state snapshot and decision-time newTx grant context.
    """
    output: dict[str, Any] = {}

    # habib added
    # Per-cell records keep only the common multi-cell synchronization key.
    sync_id = extract_sync_id(decision)
    if sync_id is not None:
        output["sync_id"] = sync_id
    # habib added

    # habib added
    # Immutable cell-level state captured after reTx scheduling and before
    # newTx resource allocation.
    state_snapshot = decision.get("state_snapshot")
    if isinstance(state_snapshot, dict):
        filtered_state: dict[str, Any] = {
            key: state_snapshot[key]
            for key in (
                "snapshot_stage",
                "bwp_size_prbs",
                "remaining_rbs",
                "final_usable_prbs",
            )
            if key in state_snapshot
        }

        occupied_ranges = state_snapshot.get("occupied_prb_ranges", [])
        if isinstance(occupied_ranges, list):
            filtered_state["occupied_prb_ranges"] = [
                {
                    key: prb_range[key]
                    for key in ("rb_start", "rb_stop")
                    if key in prb_range
                }
                for prb_range in occupied_ranges
                if isinstance(prb_range, dict)
            ]
        else:
            filtered_state["occupied_prb_ranges"] = []

        output["state_snapshot"] = filtered_state
    # habib added

    retx_candidates = decision.get("retx_candidates", [])
    output["retx_candidates"] = [
        {
            "rnti": rnti_hex(candidate.get("rnti")),
            **(
                {"harq_id": candidate["harq_id"]}
                if "harq_id" in candidate
                else {}
            ),
        }
        for candidate in retx_candidates
        if isinstance(candidate, dict)
    ]

    newtx_candidates = decision.get("newtx_candidates", [])
    output["newtx_candidates"] = []

    for candidate in newtx_candidates:
        if not isinstance(candidate, dict):
            continue

        filtered_candidate: dict[str, Any] = {
            "rnti": rnti_hex(candidate.get("rnti")),
        }

        for key in (
            "pending_bytes_at_decision",
            "priority",
            "rank",
            "recommended_mcs",
            "expected_nof_rbs",
        ):
            if key in candidate:
                filtered_candidate[key] = candidate[key]

        # habib added
        # Decision-time frequency and grant-size constraints.
        vrb_lims = candidate.get("vrb_lims")
        if isinstance(vrb_lims, dict):
            filtered_candidate["vrb_lims"] = {
                key: vrb_lims[key]
                for key in ("rb_start", "rb_stop")
                if key in vrb_lims
            }

        nof_rb_lims = candidate.get("nof_rb_lims")
        if isinstance(nof_rb_lims, dict):
            filtered_candidate["nof_rb_lims"] = {
                key: nof_rb_lims[key]
                for key in ("min_prbs", "max_prbs")
                if key in nof_rb_lims
            }

        # Decision-time PUSCH configuration used by the grant builder.
        pusch_cfg = candidate.get("pusch_cfg")
        if isinstance(pusch_cfg, dict):
            filtered_candidate["pusch_cfg"] = {
                key: pusch_cfg[key]
                for key in (
                    "time_domain_resource_index",
                    "start_symbol",
                    "nof_symbols",
                    "nof_layers",
                    "mcs_table",
                    "transform_precoding",
                )
                if key in pusch_cfg
            }
        # habib added

        output["newtx_candidates"].append(filtered_candidate)

    # habib added
    feasible_action_mask = decision.get("feasible_action_mask", [])
    if isinstance(feasible_action_mask, list):
        output["feasible_action_mask"] = [
            1 if bool(value) else 0
            for value in feasible_action_mask
        ]
    else:
        output["feasible_action_mask"] = []
    # habib added

    selected_grants = decision.get("selected_grants", [])
    output["selected_grants"] = [
        {
            "rnti": rnti_hex(grant.get("rnti")),
            **(
                {"tx_type": grant["tx_type"]}
                if "tx_type" in grant
                else {}
            ),
        }
        for grant in selected_grants
        if isinstance(grant, dict)
    ]

    return output
# habib added



def learn_pci_mapping_from_mac(metric: dict[str, Any]) -> None:
    """
    Optional fallback for configurations where MAC metrics are enabled.

    Expected MAC structure:
        du -> du_high -> mac -> dl -> [
            {"pci": 1, ...},
            {"pci": 2, ...}
        ]

    If MAC metrics are disabled, this function simply returns.
    """
    try:
        mac_cells = metric["du"]["du_high"]["mac"]["dl"]
    except (KeyError, TypeError):
        return

    if not isinstance(mac_cells, list):
        return

    for index, mac_cell in enumerate(mac_cells):
        if not isinstance(mac_cell, dict):
            continue

        pci = mac_cell.get("pci")
        if pci is not None:
            cell_index_to_pci[index] = int(pci)


def determine_cell_pci(
    cell: dict[str, Any],
    cell_index: int,
) -> int | None:
    """
    Determine the PCI using this priority:

      1. scheduler cell_metrics["pci"]  <-- preferred for reduced metrics
      2. direct cell["pci"], if present
      3. UE-level PCI, if present
      4. learned/static cell-index mapping
    """

    # Preferred source: scheduler cell_metrics.
    cell_metrics = cell.get("cell_metrics", {})

    if isinstance(cell_metrics, dict):
        pci = cell_metrics.get("pci")
        if pci is not None:
            return int(pci)

    # Fallback: direct PCI field.
    direct_pci = cell.get("pci")
    if direct_pci is not None:
        return int(direct_pci)

    # Fallback: UE PCI.
    ue_list = cell.get("ue_list", [])

    if isinstance(ue_list, list):
        for ue in ue_list:
            if isinstance(ue, dict) and ue.get("pci") is not None:
                return int(ue["pci"])

    # Final fallback: learned/static cell-index mapping.
    return cell_index_to_pci.get(cell_index)


# habib added
def filter_late_crc_update(
    update: dict[str, Any],
) -> dict[str, Any]:
    """Keep one finalized CRC received after its original report boundary."""

    output: dict[str, Any] = {}

    # habib added
    sync_id = extract_sync_id(update)
    if sync_id is not None:
        output["sync_id"] = sync_id
    # habib added

    if "rnti" in update:
        output["rnti"] = rnti_hex(update["rnti"])

    for key in (
        "harq_id",
        "crc_status",
        # habib added
        "pusch_sinr_db",
        "pusch_rsrp_db",
        # habib added
    ):
        if key in update:
            output[key] = update[key]

    return output
# habib added


def filter_pusch_allocation(
    allocation: dict[str, Any],
) -> dict[str, Any]:
    """
    Keep only the selected per-grant PUSCH allocation fields.

    PRB ranges are reduced to:
        rb_start
        rb_stop
    """

    output = {
        key: allocation[key]
        for key in PUSCH_ALLOCATION_KEEP
        if key in allocation
    }

    sync_id = extract_sync_id(allocation)
    if sync_id is not None:
        output["sync_id"] = sync_id

    ranges = allocation.get("prb_ranges")

    if isinstance(ranges, list):
        filtered_ranges: list[dict[str, Any]] = []

        for prb_range in ranges:
            if not isinstance(prb_range, dict):
                continue

            filtered_range = {
                key: prb_range[key]
                for key in PRB_RANGE_KEEP
                if key in prb_range
            }

            if filtered_range:
                filtered_ranges.append(filtered_range)

        output["prb_ranges"] = filtered_ranges

    return output


def filter_srs_report(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Keep only the requested fields from one SRS report."""

    output = {
        key: report[key]
        for key in SRS_REPORT_KEEP
        if key in report
    }

    srs_slot = report.get("srs_slot")
    if isinstance(srs_slot, dict):
        output["srs_slot"] = {
            key: srs_slot[key]
            for key in SRS_SLOT_KEEP
            if key in srs_slot
        }

    channel_matrix = report.get("channel_matrix")
    if isinstance(channel_matrix, list):
        output["channel_matrix"] = [
            {
                key: coefficient[key]
                for key in SRS_CHANNEL_COEFFICIENT_KEEP
                if key in coefficient
            }
            for coefficient in channel_matrix
            if isinstance(coefficient, dict)
        ]

    return output


def filter_ue_metrics(
    ue: dict[str, Any],
) -> dict[str, Any]:
    """Keep only the selected UE scheduler metrics."""

    output = {
        key: ue[key]
        for key in UE_METRICS_KEEP
        if key in ue
    }

    allocations = ue.get("pusch_allocations")

    if isinstance(allocations, list):
        output["pusch_allocations"] = [
            filter_pusch_allocation(allocation)
            for allocation in allocations
            if isinstance(allocation, dict)
        ]

    # Always expose the SRS report vector.
    # When there was no SRS in this reporting interval, write [].
    srs_reports = ue.get("srs_reports", [])

    if isinstance(srs_reports, list):
        output["srs_reports"] = [
            filter_srs_report(report)
            for report in srs_reports
            if isinstance(report, dict)
        ]
    else:
        output["srs_reports"] = []

    return output


# habib added
def build_multicell_decisions(cells: list[Any]) -> list[dict[str, Any]]:
    """
    Build one common timing record per sync_id.

    The raw OCUDU report still carries decision_slot, target_pusch_slot and k2
    inside each cell so the collector can merge them. These timing fields are
    removed from the per-cell ul_scheduler_decisions output.
    """
    by_sync_id: dict[int, dict[str, Any]] = {}

    for cell in cells:
        if not isinstance(cell, dict):
            continue

        decisions = cell.get("ul_scheduler_decisions", [])
        if not isinstance(decisions, list):
            continue

        for decision in decisions:
            if not isinstance(decision, dict):
                continue

            sync_id = extract_sync_id(decision)
            if sync_id is None:
                continue

            try:
                sync_key = int(sync_id)
            except (TypeError, ValueError):
                continue

            record = by_sync_id.setdefault(
                sync_key,
                {
                    "sync_id": sync_id,
                },
            )

            for slot_name in ("decision_slot", "target_pusch_slot"):
                slot = decision.get(slot_name)
                if isinstance(slot, dict) and slot_name not in record:
                    record[slot_name] = {
                        key: slot[key]
                        for key in ("hyper_sfn", "sfn", "slot_index")
                        if key in slot
                    }

            if "k2" in decision and "k2" not in record:
                record["k2"] = decision["k2"]

    return [by_sync_id[key] for key in sorted(by_sync_id)]
# habib added


def transform_scheduler_report(
    metric: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Convert OCUDU scheduler metrics into:

        {
          "timestamp": ...,
          "multicell_decisions": [...],
          "cells": {
            "PCI=01": {
              "cell_metrics": {...},
              # habib added
              "ul_scheduler_decisions": [...],
              # habib added
              "ues": {...}
            }
          }
        }

    Only the selected reduced metric set is written.
    """

    cells = metric.get("cells")

    if not isinstance(cells, list):
        return None

    output: dict[str, Any] = {
        "timestamp": metric.get("timestamp"),
        # habib added
        "multicell_decisions": build_multicell_decisions(cells),
        # habib added
        "cells": {},
    }

    for cell_index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue

        pci = determine_cell_pci(cell, cell_index)

        if pci is None:
            # This should normally not happen now because the reduced
            # scheduler JSON includes cell_metrics["pci"].
            cell_name = f"CELL_INDEX={cell_index}"
        else:
            cell_name = pci_key(pci)

        raw_cell_metrics = cell.get("cell_metrics", {})

        if not isinstance(raw_cell_metrics, dict):
            raw_cell_metrics = {}

        cell_output: dict[str, Any] = {
            "cell_metrics": {
                key: raw_cell_metrics[key]
                for key in CELL_METRICS_KEEP
                if key in raw_cell_metrics
            },
            # habib added
            # Always expose UL scheduler decisions in the reduced JSON.
            "ul_scheduler_decisions": [],
            # habib added
            # habib added
            "late_crc_updates": [],
            # habib added
            "ues": {},
        }

        # habib added
        raw_ul_scheduler_decisions = cell.get(
            "ul_scheduler_decisions",
            [],
        )

        if isinstance(raw_ul_scheduler_decisions, list):
            cell_output["ul_scheduler_decisions"] = [
                filter_ul_scheduler_decision(decision)
                for decision in raw_ul_scheduler_decisions
                if isinstance(decision, dict)
            ]
        # habib added

        # habib added
        raw_late_crc_updates = cell.get("late_crc_updates", [])
        if isinstance(raw_late_crc_updates, list):
            cell_output["late_crc_updates"] = [
                filter_late_crc_update(update)
                for update in raw_late_crc_updates
                if isinstance(update, dict)
            ]
        # habib added

        ue_list = cell.get("ue_list", [])

        if isinstance(ue_list, list):
            for ue_index, ue in enumerate(ue_list):
                if not isinstance(ue, dict):
                    continue

                rnti = ue.get("rnti")

                if rnti is not None:
                    ue_name = rnti_key(rnti)
                else:
                    ue_name = f"UE_INDEX={ue.get('ue', ue_index)}"

                cell_output["ues"][ue_name] = filter_ue_metrics(ue)

        output["cells"][cell_name] = cell_output

    return output


def write_record(record: dict[str, Any]) -> None:
    """
    Append one compact JSON object per line.

    Pretty-printing is disabled by default because at a 10 ms reporting
    period terminal I/O can add unnecessary collector-side workload.
    """

    compact_json = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    with OUTPUT_FILE.open("a", encoding="utf-8") as output:
        output.write(compact_json + "\n")

    if PRINT_RECORDS:
        print(json.dumps(record, indent=2), flush=True)


def on_open(ws: websocket.WebSocketApp) -> None:
    print(f"Connected to {WS_ENDPOINT}", file=sys.stderr)
    print(f"Writing metrics to {OUTPUT_FILE}", file=sys.stderr)

    if not PRINT_RECORDS:
        print(
            "Per-report stdout printing is disabled "
            "(set PRINT_RECORDS=1 to enable).",
            file=sys.stderr,
        )

    ws.send(json.dumps({"cmd": "metrics_subscribe"}))


def on_message(
    _ws: websocket.WebSocketApp,
    message: str,
) -> None:
    try:
        metric = json.loads(message)
    except json.JSONDecodeError:
        return

    if not isinstance(metric, dict):
        return

    # Ignore command acknowledgements.
    if "cmd" in metric:
        return

    # Optional fallback only. With the reduced scheduler serializer,
    # PCI is read directly from cell_metrics["pci"].
    learn_pci_mapping_from_mac(metric)

    # Scheduler reports contain the top-level "cells" list.
    grouped_report = transform_scheduler_report(metric)

    if grouped_report is not None:
        write_record(grouped_report)


def on_error(
    _ws: websocket.WebSocketApp,
    error: Any,
) -> None:
    print(f"WebSocket error: {error}", file=sys.stderr)


def on_close(
    _ws: websocket.WebSocketApp,
    status_code: Any,
    message: Any,
) -> None:
    print(
        f"WebSocket closed: status={status_code}, message={message}",
        file=sys.stderr,
    )


def main() -> None:
    parse_static_pci_map()

    print(
        f"Initial fallback PCI mapping: {cell_index_to_pci}",
        file=sys.stderr,
    )

    while True:
        app = websocket.WebSocketApp(
            WS_ENDPOINT,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        try:
            app.run_forever()
        except KeyboardInterrupt:
            print(
                "\nMetrics collection stopped.",
                file=sys.stderr,
            )
            break
        except Exception as error:
            print(
                f"Collector error: {error}",
                file=sys.stderr,
            )

        print(
            "Reconnecting in one second...",
            file=sys.stderr,
        )
        time.sleep(1)


if __name__ == "__main__":
    main()
