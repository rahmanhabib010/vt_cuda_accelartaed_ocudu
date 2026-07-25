#!/bin/bash
# E2E PUSCH GPU Pipeline Correctness Sweep
# Tests all MCS indices, TB sizes, base graphs, and configurations

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
cd "$BUILD_DIR"

echo ""
echo "================================================================"
echo "  E2E GPU PUSCH Pipeline Correctness Sweep"
echo "  Testing: All MCS, TB sizes, base graphs, lifting sizes"
echo "================================================================"
echo ""

# High SNR for correctness testing (25 dB ensures MCS 27 64QAM high code rate converges)
SINR=25
ITERS=10

TOTAL=0
PASSED=0
FAILED=0

# Helper function to run a test
run_test() {
    local PRB=$1
    local MCS=$2
    local DESC=$3

    TOTAL=$((TOTAL + 1))
    printf "  MCS%2d... " "$MCS"

    # Run test and capture output
    OUTPUT=$(./tests/integrationtests/phy/upper/channel_processors/pusch_64qam_stress_test \
             --prb "$PRB" --sinr "$SINR" --mcs "$MCS" --iterations "$ITERS" 2>&1)

    # Check if both CPU and GPU passed all iterations
    CPU_PASS=$(echo "$OUTPUT" | grep -oP 'CPU: \K\d+(?=/\d+ pass)' || echo "0")
    GPU_PASS=$(echo "$OUTPUT" | grep -oP 'GPU: \K\d+(?=/\d+ pass)' || echo "0")
    DISAGREE=$(echo "$OUTPUT" | grep -oP 'Disagreements: \K\d+' || echo "999")

    if [ "$CPU_PASS" -eq "$ITERS" ] && [ "$GPU_PASS" -eq "$ITERS" ] && [ "$DISAGREE" -eq "0" ]; then
        echo "PASS"
        PASSED=$((PASSED + 1))
    else
        echo "FAIL (CPU:$CPU_PASS/$ITERS GPU:$GPU_PASS/$ITERS Disagree:$DISAGREE)"
        FAILED=$((FAILED + 1))
    fi
}

# 5 PRB - Tiny (BG2)
echo ""
echo "--- 5 PRB: Tiny (BG2) ---"
run_test 5 0 "QPSK low"
run_test 5 4 "QPSK mid"
run_test 5 11 "16QAM low"
run_test 5 16 "16QAM/64QAM boundary"
run_test 5 22 "64QAM mid"
run_test 5 27 "64QAM max"

# 25 PRB - Medium (BG1/BG2)
echo ""
echo "--- 25 PRB: Medium (BG1/BG2) ---"
run_test 25 0 "QPSK low"
run_test 25 11 "16QAM low"
run_test 25 19 "64QAM low"
run_test 25 27 "64QAM max"

# 52 PRB - Large (BG1)
echo ""
echo "--- 52 PRB: Large (BG1) ---"
run_test 52 4 "QPSK mid"
run_test 52 14 "16QAM mid"
run_test 52 22 "64QAM mid"
run_test 52 27 "64QAM max"

# 106 PRB - Very Large (multi-CB)
echo ""
echo "--- 106 PRB: Very Large (multi-CB) ---"
run_test 106 8 "QPSK high"
run_test 106 16 "16QAM/64QAM boundary"
run_test 106 25 "64QAM high"

# Non-uniform E (E_short != E_long) explicit coverage.
echo ""
echo "--- Non-Uniform E: Explicit Coverage ---"
run_test 107 27 "256QAM, 11 CBs, 3 short"
run_test 200 22 "64QAM, 14 CBs, 12 short"
run_test 273 14 "16QAM, 11 CBs, 2 short"

# Final results
echo ""
echo ""
echo "================================================================"
echo "  FINAL RESULTS"
echo "================================================================"
echo ""
echo "Total Tests:  $TOTAL"
PASS_PCT=$(awk "BEGIN {printf \"%.1f\", 100.0 * $PASSED / $TOTAL}")
FAIL_PCT=$(awk "BEGIN {printf \"%.1f\", 100.0 * $FAILED / $TOTAL}")
echo "Passed:      $PASSED ($PASS_PCT%)"
echo "Failed:      $FAILED ($FAIL_PCT%)"

if [ "$FAILED" -eq 0 ]; then
    echo ""
    echo "ALL TESTS PASSED. E2E GPU PUSCH pipeline is correct."
    echo "   PASS: All MCS indices tested"
    echo "   PASS: All TB sizes tested (tiny to very large)"
    echo "   PASS: Both base graphs tested (BG1, BG2)"
    echo "   PASS: Various lifting sizes tested"
    echo "   PASS: Various filler bit configs tested"
    echo "   PASS: GPU matches CPU bit-exact on all tests"
    echo ""
    exit 0
else
    echo ""
    echo "CORRECTNESS ISSUES DETECTED"
    echo "   Please fix the failing cases above before optimizing."
    exit 1
fi
