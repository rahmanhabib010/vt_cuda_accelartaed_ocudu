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

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Maps the position in the scheduler "cells" list to its PCI.
# This is learned from MAC reports, whose "dl" list includes one PCI
# entry for every configured cell.
cell_index_to_pci: dict[int, int] = {}


def parse_static_pci_map() -> None:
    """
    Optional manual mapping:

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
    A MAC metrics message has:

    du -> du_high -> mac -> dl -> [
        {"pci": 1, ...},
        {"pci": 2, ...}
    ]
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
    Scheduler cell_metrics normally does not contain PCI directly.
    Determine it from:

    1. A direct cell PCI field, if available.
    2. Any UE connected to that cell.
    3. The index-to-PCI mapping learned from MAC metrics.
    """

    direct_pci = cell.get("pci")
    if direct_pci is not None:
        return int(direct_pci)

    ue_list = cell.get("ue_list", [])

    if isinstance(ue_list, list):
        for ue in ue_list:
            if isinstance(ue, dict) and ue.get("pci") is not None:
                return int(ue["pci"])

    return cell_index_to_pci.get(cell_index)


def transform_scheduler_report(
    metric: dict[str, Any],
) -> dict[str, Any] | None:
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
            # This may happen briefly before the first MAC metrics report.
            cell_name = f"CELL_INDEX={cell_index}"
        else:
            cell_name = pci_key(pci)

        cell_output: dict[str, Any] = {
            "cell_metrics": cell.get("cell_metrics", {}),
            "event_list": cell.get("event_list", []),
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

                # Preserve every UE metric exactly as received.
                cell_output["ues"][ue_name] = ue

        output["cells"][cell_name] = cell_output

    return output


def write_record(record: dict[str, Any]) -> None:
    compact_json = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    with OUTPUT_FILE.open("a", encoding="utf-8") as output:
        output.write(compact_json + "\n")

    print(json.dumps(record, indent=2), flush=True)


def on_open(ws: websocket.WebSocketApp) -> None:
    print(f"Connected to ws://{WS_URL}", file=sys.stderr)
    print(f"Writing metrics to {OUTPUT_FILE}", file=sys.stderr)

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

    # MAC reports provide the PCI order for configured cells.
    learn_pci_mapping_from_mac(metric)

    # Only scheduler reports contain the top-level cells list.
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

    print(f"Initial PCI mapping: {cell_index_to_pci}", file=sys.stderr)

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
            print("\nMetrics collection stopped.", file=sys.stderr)
            break
        except Exception as error:
            print(f"Collector error: {error}", file=sys.stderr)

        print("Reconnecting in one second...", file=sys.stderr)
        time.sleep(1)


if __name__ == "__main__":
    main()
