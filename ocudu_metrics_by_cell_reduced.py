#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import websocket


WS_URL = os.environ.get("WS_URL", "127.0.0.1:8001")

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
    "ul_ri",
)

PUSCH_ALLOCATION_KEEP = (
    "allocation_type",
    "bwp_size_prbs",
    "hyper_sfn",
    "nof_prbs",
    "sfn",
    "slot_index",
)

PRB_RANGE_KEEP = (
    "rb_start",
    "rb_stop",
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

    return output


def transform_scheduler_report(
    metric: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Convert OCUDU scheduler metrics into:

        {
          "timestamp": ...,
          "cells": {
            "PCI=01": {
              "cell_metrics": {...},
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
            "ues": {},
        }

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
    print(f"Connected to ws://{WS_URL}", file=sys.stderr)
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
            f"ws://{WS_URL}",
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
