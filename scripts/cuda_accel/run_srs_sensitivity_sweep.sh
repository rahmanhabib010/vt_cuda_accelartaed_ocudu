#!/bin/bash
# CPU vs GPU SRS channel-estimator accuracy sweep across SNR.

set -e

BUILD_DIR=""
FRAMES=50
SNR_START=0
SNR_STOP=30
SNR_STEP=5
GRID_MODE="visible"
RX_PORTS=4
TX_PORTS=4
SYMBOLS=4
PROFILES="single"
FORCE_GPU=0

usage() {
  echo "Usage: $0 --build-dir <path> [--quick|--full] [--frames N] [--snr-start DB] [--snr-stop DB]"
  echo "          [--snr-step DB] [--grid host|visible] [--rx-ports N] [--tx-ports 1|2|4] [--symbols N]"
  echo "          [--profiles single|quick|full] [--force-gpu]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --quick) FRAMES=10; SNR_STEP=10; shift ;;
    --full) FRAMES=200; SNR_STEP=2.5; shift ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --snr-start) SNR_START="$2"; shift 2 ;;
    --snr-stop) SNR_STOP="$2"; shift 2 ;;
    --snr-step) SNR_STEP="$2"; shift 2 ;;
    --grid) GRID_MODE="$2"; shift 2 ;;
    --rx-ports) RX_PORTS="$2"; shift 2 ;;
    --tx-ports) TX_PORTS="$2"; shift 2 ;;
    --symbols) SYMBOLS="$2"; shift 2 ;;
    --profiles) PROFILES="$2"; shift 2 ;;
    --force-gpu) FORCE_GPU=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$BUILD_DIR" ]]; then
  usage
  exit 1
fi
if [[ ! -d "$BUILD_DIR" ]]; then
  echo "Error: build directory not found: $BUILD_DIR"
  exit 1
fi

BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
BINARY="${BUILD_DIR}/tests/benchmarks/phy/upper/signal_processors/srs_estimator_gpu_sensitivity_sweep"

if [[ ! -f "$BINARY" ]]; then
  echo "Binary not found: $BINARY"
  echo "Build with: cmake --build ${BUILD_DIR} -j\$(nproc) --target srs_estimator_gpu_sensitivity_sweep"
  exit 1
fi

echo "================================================================"
echo "  SRS GPU Sensitivity Sweep"
echo "  Grid: ${GRID_MODE}  Frames/SNR: ${FRAMES}  SNR: ${SNR_START}:${SNR_STEP}:${SNR_STOP} dB"
echo "  Profiles: ${PROFILES}  RX ports: ${RX_PORTS}  TX ports: ${TX_PORTS}  Symbols: ${SYMBOLS}"
echo "  Force GPU: ${FORCE_GPU}"
echo "================================================================"

if [[ "$FORCE_GPU" -eq 1 ]]; then
  export OCUDU_SRS_ACCELERATION_FORCE=1
fi

exec "$BINARY" \
  -F "$FRAMES" \
  -A "$SNR_START" \
  -B "$SNR_STOP" \
  -D "$SNR_STEP" \
  -G "$GRID_MODE" \
  -P "$PROFILES" \
  -r "$RX_PORTS" \
  -t "$TX_PORTS" \
  -s "$SYMBOLS"
