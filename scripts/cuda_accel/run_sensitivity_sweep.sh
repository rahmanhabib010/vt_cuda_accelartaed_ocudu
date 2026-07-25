#!/bin/bash
# PUSCH E2E Sensitivity Sweep - CPU vs GPU
# Finds 10% BLER threshold SNR for each PRB/MCS config.
#
# Usage:
#   bash run_sensitivity_sweep.sh           # default (100 frames, 1dB step)
#   bash run_sensitivity_sweep.sh --quick   # fast smoke test (30 frames, 2dB step)
#   bash run_sensitivity_sweep.sh --full    # high-accuracy (500 frames, 0.5dB step)

set -e

BUILD_DIR=""
MODE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-dir) BUILD_DIR="$2"; shift 2;;
        --quick) MODE="quick"; shift;;
        --full) MODE="full"; shift;;
        *) echo "Unknown option: $1"; echo "Usage: $0 --build-dir <path> [--quick|--full]"; exit 1;;
    esac
done
if [ -z "$BUILD_DIR" ]; then echo "Usage: $0 --build-dir <path> [--quick|--full]"; exit 1; fi
if [ ! -d "$BUILD_DIR" ]; then echo "Error: build directory not found: $BUILD_DIR"; exit 1; fi
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
cd "$BUILD_DIR"

BINARY="./tests/integrationtests/phy/upper/channel_processors/pusch_e2e_sensitivity_sweep"

if [ ! -f "$BINARY" ]; then
    echo "Binary not found: $BINARY"
    exit 1
fi

# Defaults
FRAMES=100
STEP=0.1
START=-5.0
STOP=25.0

case "$MODE" in
    quick)
        FRAMES=30
        STEP=2.0
        START=-2.0
        STOP=22.0
        ;;
    full)
        FRAMES=500
        STEP=0.5
        START=-5.0
        STOP=25.0
        ;;
esac

echo ""
echo "================================================================"
echo "  PUSCH E2E Sensitivity Sweep"
echo "  Frames/point: $FRAMES  Step: ${STEP}dB  Range: [${START}, ${STOP}]dB"
echo "================================================================"
echo ""

exec "$BINARY" \
    --nof_frames "$FRAMES" \
    --snr_step "$STEP" \
    --snr_start "$START" \
    --snr_stop "$STOP" \
    "$@"
