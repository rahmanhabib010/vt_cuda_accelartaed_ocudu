#!/bin/bash
# CPU vs GPU PUSCH Latency Profiling Sweep
#
# Runs the stress test across multiple PRB/MCS configurations to produce
# a comprehensive latency comparison with GPU phase breakdown.
#
# Usage: bash run_latency_profile.sh [iterations]

set -e

BUILD_DIR=""
ITERATIONS=50
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-dir) BUILD_DIR="$2"; shift 2;;
        --iterations) ITERATIONS="$2"; shift 2;;
        *) echo "Unknown option: $1"; echo "Usage: $0 --build-dir <path> [--iterations N]"; exit 1;;
    esac
done
if [ -z "$BUILD_DIR" ]; then echo "Usage: $0 --build-dir <path> [--iterations N]"; exit 1; fi
if [ ! -d "$BUILD_DIR" ]; then echo "Error: build directory not found: $BUILD_DIR"; exit 1; fi
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
TEST_BIN="${BUILD_DIR}/tests/integrationtests/phy/upper/channel_processors/pusch_64qam_stress_test"
SINR=25

if [ ! -f "${TEST_BIN}" ]; then
  echo "Test binary not found at ${TEST_BIN}"
  echo "Build with: cmake --build ${BUILD_DIR} -j\$(nproc) --target pusch_64qam_stress_test"
  exit 1
fi

echo "========================================"
echo "  CPU vs GPU PUSCH Latency Profile"
echo "  SINR: ${SINR} dB, Iterations: ${ITERATIONS}"
echo "========================================"

RESULTS_FILE=$(mktemp)

# Test configurations: PRB MCS MCS_TABLE LABEL
CONFIGS=(
  "5   0  qam64  QPSK"
  "5  11  qam64  16QAM"
  "5  19  qam64  64QAM"
  "5  27  qam64  64QAM-HR"
  "5  25  qam256 256QAM"
  "25  0  qam64  QPSK"
  "25 11  qam64  16QAM"
  "25 19  qam64  64QAM"
  "25 27  qam64  64QAM-HR"
  "25 25  qam256 256QAM"
  "52  0  qam64  QPSK"
  "52 11  qam64  16QAM"
  "52 19  qam64  64QAM"
  "52 27  qam64  64QAM-HR"
  "52 25  qam256 256QAM"
  "106  0  qam64  QPSK"
  "106 11  qam64  16QAM"
  "106 19  qam64  64QAM"
  "106 27  qam64  64QAM-HR"
  "106 25  qam256 256QAM"
  "273  0  qam64  QPSK"
  "273 11  qam64  16QAM"
  "273 19  qam64  64QAM"
  "273 27  qam64  64QAM-HR"
  "273 25  qam256 256QAM"
)

for cfg in "${CONFIGS[@]}"; do
  read -r PRB MCS TABLE LABEL <<< "$cfg"
  echo ""
  echo "----------------------------------------"
  echo "  PRB=${PRB}, MCS=${MCS} (${LABEL}, table=${TABLE})"
  echo "----------------------------------------"

  OUTPUT=$("${TEST_BIN}" --prb "${PRB}" --sinr "${SINR}" --mcs "${MCS}" --mcs-table "${TABLE}" --iterations "${ITERATIONS}" 2>&1)
  echo "$OUTPUT"

  # Parse results
  CPU_MEAN=$(echo "$OUTPUT" | grep -A2 "^CPU Latency:" | grep "Mean:" | grep -oP '[\d.]+' | head -1)
  GPU_MEAN=$(echo "$OUTPUT" | grep -A2 "^GPU Latency:" | grep "Mean:" | grep -oP '[\d.]+' | head -1)
  CPU_BLER=$(echo "$OUTPUT" | grep "^CPU:" | grep -oP '[\d.]+(?=% BLER)' | head -1)
  GPU_BLER=$(echo "$OUTPUT" | grep "^GPU:" | grep -oP '[\d.]+(?=% BLER)' | head -1)
  CPU_ITERS=$(echo "$OUTPUT" | grep "^CPU avg:" | grep -oP '[\d.]+' | head -1)
  GPU_ITERS=$(echo "$OUTPUT" | grep "^GPU avg:" | grep -oP '[\d.]+' | head -1)
  SPEEDUP=$(echo "$OUTPUT" | grep "^Speedup:" | grep -oP '[\d.]+' | head -1)

  # Default to "-" if not found
  CPU_MEAN=${CPU_MEAN:-"-"}
  GPU_MEAN=${GPU_MEAN:-"-"}
  CPU_BLER=${CPU_BLER:-"-"}
  GPU_BLER=${GPU_BLER:-"-"}
  CPU_ITERS=${CPU_ITERS:-"-"}
  GPU_ITERS=${GPU_ITERS:-"-"}
  SPEEDUP=${SPEEDUP:-"-"}

  echo "${PRB} ${MCS} ${LABEL} ${TABLE} ${CPU_MEAN} ${GPU_MEAN} ${SPEEDUP} ${CPU_BLER} ${GPU_BLER} ${CPU_ITERS} ${GPU_ITERS}" >> "${RESULTS_FILE}"
done

echo ""
echo ""
echo "=============================================================================="
echo "  SUMMARY TABLE"
echo "  SINR: ${SINR} dB, Iterations: ${ITERATIONS}"
echo "=============================================================================="
printf "%-5s %-4s %-9s %8s %8s %7s %6s %6s %6s %6s\n" \
       "PRB" "MCS" "Mod" "CPU(us)" "GPU(us)" "Speedup" "C-BLER" "G-BLER" "C-Iter" "G-Iter"
echo "------------------------------------------------------------------------------"

PREV_PRB=""
while IFS=' ' read -r PRB MCS LABEL TABLE CPU_MEAN GPU_MEAN SPEEDUP CPU_BLER GPU_BLER CPU_ITERS GPU_ITERS; do
  # Print separator between PRB groups
  if [ -n "$PREV_PRB" ] && [ "$PRB" != "$PREV_PRB" ]; then
    echo "------------------------------------------------------------------------------"
  fi
  PREV_PRB="$PRB"

  printf "%-5s %-4s %-9s %8s %8s %6sx %5s%% %5s%% %6s %6s\n" \
         "$PRB" "$MCS" "$LABEL" "$CPU_MEAN" "$GPU_MEAN" "$SPEEDUP" "$CPU_BLER" "$GPU_BLER" "$CPU_ITERS" "$GPU_ITERS"
done < "${RESULTS_FILE}"

echo "=============================================================================="
echo ""

rm -f "${RESULTS_FILE}"
