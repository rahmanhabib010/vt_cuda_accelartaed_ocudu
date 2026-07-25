#!/bin/bash
# Multi-port E2E PUSCH GPU correctness sweep
# Runs the full MCS/PRB correctness sweep for 1, 2, 4, and 8 RX ports.

set -e

BUILD_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-dir) BUILD_DIR="$2"; shift 2;;
        *) echo "Unknown option: $1"; echo "Usage: $0 --build-dir <path>"; exit 1;;
    esac
done
if [ -z "$BUILD_DIR" ]; then echo "Usage: $0 --build-dir <path>"; exit 1; fi
if [ ! -d "$BUILD_DIR" ]; then echo "Error: build directory not found: $BUILD_DIR"; exit 1; fi
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
BIN="${BUILD_DIR}/tests/integrationtests/phy/upper/channel_processors/pusch_e2e_correctness_sweep"

if [ ! -x "$BIN" ]; then
  echo "Binary not found: $BIN"
  echo "Build with: cmake --build build --target pusch_e2e_correctness_sweep -j\$(nproc)"
  exit 1
fi

FAILED=0

# Each run uses different RNTI, n_id, scrambling_id, n_scid to exercise
# all scrambling seed paths and verify the DMRS precompute cache.
PORTS_LIST=(1        2        4        8)
RNTI_LIST=(0x1234   0xBEEF   0x4321   0x0001)
NID_LIST=(0         500      1023     42)
SCID_LIST=(0        1000     65535    12345)
NSCID_LIST=(0       1        0        1)

for i in "${!PORTS_LIST[@]}"; do
  PORTS=${PORTS_LIST[$i]}
  RNTI=${RNTI_LIST[$i]}
  NID=${NID_LIST[$i]}
  SCID=${SCID_LIST[$i]}
  NSCID=${NSCID_LIST[$i]}

  echo ""
  echo "================================================================"
  echo "  RX Ports: $PORTS  RNTI: $RNTI  n_id: $NID  scrambling_id: $SCID  n_scid: $NSCID"
  echo "================================================================"
  if ! "$BIN" --ports "$PORTS" --rnti "$RNTI" --n-id "$NID" --scrambling-id "$SCID" --n-scid "$NSCID" "$@"; then
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "================================================================"
if [ "$FAILED" -eq 0 ]; then
  echo "  ALL PORT CONFIGURATIONS PASSED"
else
  echo "  $FAILED PORT CONFIGURATION(S) FAILED"
fi
echo "================================================================"

exit $FAILED
