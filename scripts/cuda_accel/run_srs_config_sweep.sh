#!/bin/bash
# CPU vs GPU SRS channel-estimator configuration coverage sweep.

set -e

BUILD_DIR=""
GRID_MODE="visible"
SNR=30
QUICK_FLAG=""
MAX_CASES=0
VERBOSE_FLAG=""
CORRELATION_FLAG=""

usage() {
  echo "Usage: $0 --build-dir <path> [--quick|--full] [--max-cases N] [--grid host|visible] [--snr DB]"
  echo "          [--windowed|--full-correlation] [--verbose]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --quick) QUICK_FLAG="-q"; MAX_CASES=96; shift ;;
    --full) QUICK_FLAG=""; MAX_CASES=0; shift ;;
    --max-cases) MAX_CASES="$2"; shift 2 ;;
    --grid) GRID_MODE="$2"; shift 2 ;;
    --snr) SNR="$2"; shift 2 ;;
    --windowed) CORRELATION_FLAG="-w"; shift ;;
    --full-correlation) CORRELATION_FLAG=""; shift ;;
    --verbose) VERBOSE_FLAG="-v"; shift ;;
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
BINARY="${BUILD_DIR}/tests/benchmarks/phy/upper/signal_processors/srs_estimator_gpu_config_sweep"

if [[ ! -f "$BINARY" ]]; then
  echo "Binary not found: $BINARY"
  echo "Build with: cmake --build ${BUILD_DIR} -j\$(nproc) --target srs_estimator_gpu_config_sweep"
  exit 1
fi

echo "================================================================"
echo "  SRS GPU Configuration Sweep"
echo "  Grid: ${GRID_MODE}  SNR: ${SNR} dB  Max cases: ${MAX_CASES}"
echo "  Correlation: $([[ -n "${CORRELATION_FLAG}" ]] && echo windowed || echo full)"
echo "================================================================"

exec "$BINARY" \
  -G "$GRID_MODE" \
  -S "$SNR" \
  -N "$MAX_CASES" \
  ${QUICK_FLAG} \
  ${CORRELATION_FLAG} \
  ${VERBOSE_FLAG}
