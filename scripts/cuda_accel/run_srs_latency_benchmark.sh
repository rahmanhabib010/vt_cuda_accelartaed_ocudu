#!/bin/bash
# CPU vs GPU SRS channel-estimator latency benchmark.

set -e

BUILD_DIR=""
REPETITIONS=1000
WARMUPS=50
GRID_MODE="visible"
SNR=30
RX_PORTS=4
TX_PORTS=4
SYMBOLS=4
PROFILES="single"
FORCE_GPU=0

usage() {
  echo "Usage: $0 --build-dir <path> [--quick|--full] [--repetitions N] [--warmups N] [--grid host|visible]"
  echo "          [--snr DB] [--rx-ports N] [--tx-ports 1|2|4] [--symbols N] [--profiles single|quick|full]"
  echo "          [--force-gpu]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --quick) REPETITIONS=100; WARMUPS=10; shift ;;
    --full) REPETITIONS=5000; WARMUPS=200; shift ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --warmups) WARMUPS="$2"; shift 2 ;;
    --grid) GRID_MODE="$2"; shift 2 ;;
    --snr) SNR="$2"; shift 2 ;;
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
BINARY="${BUILD_DIR}/tests/benchmarks/phy/upper/signal_processors/srs_estimator_gpu_latency_benchmark"

if [[ ! -f "$BINARY" ]]; then
  echo "Binary not found: $BINARY"
  echo "Build with: cmake --build ${BUILD_DIR} -j\$(nproc) --target srs_estimator_gpu_latency_benchmark"
  exit 1
fi

echo "================================================================"
echo "  SRS GPU Latency Benchmark"
echo "  Grid: ${GRID_MODE}  Repetitions: ${REPETITIONS}  Warmups: ${WARMUPS}"
echo "  Profiles: ${PROFILES}  RX ports: ${RX_PORTS}  TX ports: ${TX_PORTS}  Symbols: ${SYMBOLS}  SNR: ${SNR} dB"
echo "  Force GPU: ${FORCE_GPU}"
echo "================================================================"

if [[ "$FORCE_GPU" -eq 1 ]]; then
  export OCUDU_SRS_ACCELERATION_FORCE=1
fi

exec "$BINARY" \
  -R "$REPETITIONS" \
  -W "$WARMUPS" \
  -G "$GRID_MODE" \
  -P "$PROFILES" \
  -S "$SNR" \
  -r "$RX_PORTS" \
  -t "$TX_PORTS" \
  -s "$SYMBOLS"
