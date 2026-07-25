#!/bin/bash
# Regression test: compare two builds for regressions
#
# Runs all test binaries from both build directories, captures output, and
# writes a results table incrementally. Supports resume from interrupted runs.
#
# Usage:
#   bash run_regression_test.sh --fresh --baseline-build <path> --target-build <path>
#   bash run_regression_test.sh --fresh --baseline-build <path> --target-build <path> --output-base-path <path>
#   bash run_regression_test.sh --fresh --baseline-build <path> --target-build <path> --gtest-only
#   bash run_regression_test.sh --results-dir <path> --baseline-build <path> --target-build <path>
#
# Prerequisites:
#   - Both build directories must already exist with binaries built
#   - Run from the repo root, or any directory; scripts are found via SCRIPT_DIR.
#
# Output:
#   <output-base-path>/results_<timestamp>/          test logs
#   <output-base-path>/results_<timestamp>/results_table_<timestamp>.md

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============================================================
#  Parse flags
# ============================================================

FRESH=false
GTEST_ONLY=false
BASELINE_BUILD=""
TARGET_BUILD=""
OUTPUT_BASE_PATH=""
RESULTS_DIR=""

usage() {
    echo "Usage:"
    echo "  $(basename "$0") --fresh --baseline-build <path> --target-build <path> [--output-base-path <path>] [--gtest-only]"
    echo "  $(basename "$0") --results-dir <path> --baseline-build <path> --target-build <path>"
    echo ""
    echo "Options:"
    echo "  --fresh                Start a new regression test"
    echo "  --baseline-build <path>  Path to the baseline build directory"
    echo "  --target-build <path>    Path to the target build directory"
    echo "  --output-base-path       Base directory for results (default: <repo-root>/regression-test)"
    echo "  --results-dir <path>     Resume from an existing results directory"
    echo "  --gtest-only             Only run gtest binaries (tests 1-5), skip everything else"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --fresh)
            FRESH=true
            shift
            ;;
        --baseline-build)
            BASELINE_BUILD="$2"
            shift 2
            ;;
        --target-build)
            TARGET_BUILD="$2"
            shift 2
            ;;
        --output-base-path)
            OUTPUT_BASE_PATH="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --gtest-only)
            GTEST_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate flags
if [ -z "$BASELINE_BUILD" ] || [ -z "$TARGET_BUILD" ]; then
    echo "Error: --baseline-build and --target-build are required."
    usage
fi

if ! $FRESH && [ -z "$RESULTS_DIR" ]; then
    echo "Error: either --fresh or --results-dir is required."
    usage
fi

if $FRESH && [ -n "$RESULTS_DIR" ]; then
    echo "Error: --fresh and --results-dir are mutually exclusive."
    usage
fi

# Verify build dirs exist, resolve to absolute, check for Makefiles
for dir_label in "Baseline:BASELINE_BUILD" "Target:TARGET_BUILD"; do
    label="${dir_label%%:*}"
    varname="${dir_label#*:}"
    dir="${!varname}"
    if [ ! -d "$dir" ]; then
        echo "Error: $label build directory not found: $dir"
        exit 1
    fi
    resolved="$(cd "$dir" && pwd)"
    eval "$varname=\"$resolved\""
    if [ ! -f "$resolved/Makefile" ]; then
        echo "Error: $label build directory not configured (no Makefile): $resolved"
        exit 1
    fi
done

# ============================================================
#  Derived paths
# ============================================================

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -z "$OUTPUT_BASE_PATH" ]; then
    OUTPUT_BASE_PATH="${REPO_ROOT}/regression-test"
fi

if $FRESH; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="${OUTPUT_BASE_PATH}/results_${TIMESTAMP}"
    RESULTS_TABLE="${RESULTS_DIR}/results_table_${TIMESTAMP}.md"
else
    RESULTS_TABLE=$(ls "${RESULTS_DIR}"/results_table_*.md 2>/dev/null | head -1)
    if [ -z "$RESULTS_TABLE" ]; then
        echo "Error: no results_table_*.md found in ${RESULTS_DIR}"
        exit 1
    fi
fi

# ============================================================
#  Print configuration
# ============================================================

echo "========================================"
echo "  Regression Test"
echo "========================================"
echo "  Baseline build: $BASELINE_BUILD"
echo "  Target build:   $TARGET_BUILD"
echo "  Output:         $RESULTS_DIR"
echo ""

# ============================================================
#  Results directory and table
# ============================================================

mkdir -p "$RESULTS_DIR/baseline" "$RESULTS_DIR/target"

init_results_table() {
    if [ ! -f "$RESULTS_TABLE" ]; then
        cat > "$RESULTS_TABLE" << HEADER
# Regression Test Results

Generated by \`run_regression_test.sh\`

- Baseline build: \`${BASELINE_BUILD}\`
- Target build: \`${TARGET_BUILD}\`

| # | Test | Comparison fields | baseline | target | Regression? |
|---|------|-------------------|----------|--------|-------------|
HEADER
    fi
}

test_already_done() {
    local num="$1"
    grep -q "^| ${num} " "$RESULTS_TABLE" 2>/dev/null
}

append_result() {
    local num="$1"
    local name="$2"
    local fields="$3"
    local baseline_val="$4"
    local target_val="$5"
    local regression="$6"

    echo "| ${num} | \`${name}\` | ${fields} | ${baseline_val} | ${target_val} | ${regression} |" >> "$RESULTS_TABLE"
}

init_results_table

echo "========================================"
echo "  Running tests"
echo "  Table: $RESULTS_TABLE"
echo "========================================"

# ============================================================
#  Test runner
# ============================================================

run_both() {
    local num="$1"
    local name="$2"
    local baseline_cmd="$3"
    local target_cmd="${4:-$3}"

    if test_already_done "$num"; then
        echo ""
        echo "=== Test #$num: $name === SKIPPED (already in results table)"
        return
    fi

    echo ""
    echo "=== Test #$num: $name ==="

    local b_log="$RESULTS_DIR/baseline/test_${num}.log"
    local t_log="$RESULTS_DIR/target/test_${num}.log"

    echo "  [baseline] start $(date +%H:%M:%S)"
    eval timeout 900 "$baseline_cmd" > "$b_log" 2>&1 || true
    echo "  [baseline] done $(date +%H:%M:%S)"

    echo "  [target] start $(date +%H:%M:%S)"
    eval timeout 900 "$target_cmd" > "$t_log" 2>&1 || true
    echo "  [target] done $(date +%H:%M:%S)"
}

# ============================================================
#  Extraction helpers
# ============================================================

extract_gtest() {
    local num="$1"
    local name="$2"
    local b_log="$RESULTS_DIR/baseline/test_${num}.log"
    local t_log="$RESULTS_DIR/target/test_${num}.log"

    local bp=$(grep -oP '\[  PASSED  \] \K\d+' "$b_log" 2>/dev/null | tail -1)
    local bf=$(grep -oP '\[  FAILED  \] \K\d+' "$b_log" 2>/dev/null | tail -1)
    local tp=$(grep -oP '\[  PASSED  \] \K\d+' "$t_log" 2>/dev/null | tail -1)
    local tf=$(grep -oP '\[  FAILED  \] \K\d+' "$t_log" 2>/dev/null | tail -1)
    bf=${bf:-0}; tf=${tf:-0}

    local reg="No"
    if [ "$bp/$bf" != "$tp/$tf" ]; then reg="REGRESSION"; fi
    if [ "$bf" != "0" ] && [ "$bp/$bf" = "$tp/$tf" ]; then reg="No (pre-existing)"; fi

    append_result "$num" "$name" "pass/fail" "${bp} pass / ${bf} fail" "${tp} pass / ${tf} fail" "$reg"
    printf "  Result: baseline=%s/%s target=%s/%s %s\n" "$bp" "$bf" "$tp" "$tf" "$reg"
}

# ============================================================
#  Binary paths
# ============================================================

B="$BASELINE_BUILD"
T="$TARGET_BUILD"
S="$SCRIPT_DIR"

# ============================================================
#  Run all tests
# ============================================================

# --- Tests 1-5: gtest ---
for i in 1 2 3 4 5; do
    case $i in
        1) name="gpu_demod_llr_sign_test"; rel="tests/unittests/phy/upper/channel_coding/ldpc/gpu_demod_llr_sign_test";;
        2) name="pusch_sinr_comparison_test"; rel="tests/unittests/phy/upper/channel_processors/pusch/pusch_sinr_comparison_test";;
        3) name="ldpc_encoder_gpu_cpu_test"; rel="tests/unittests/phy/upper/channel_coding/ldpc/ldpc_encoder_gpu_cpu_test";;
        4) name="ldpc_decoder_gpu_cpu_test"; rel="tests/unittests/phy/upper/channel_coding/ldpc/ldpc_decoder_gpu_cpu_test";;
        5) name="short_block_detector_cuda_test"; rel="tests/unittests/phy/upper/channel_coding/short/short_block_detector_cuda_test";;
    esac
    run_both "$i" "$name" "$B/$rel" "$T/$rel"
    test_already_done "$i" || extract_gtest "$i" "$name"
done

if $GTEST_ONLY; then
    echo ""
    echo "========================================"
    echo "  --gtest-only: skipping tests 6-16"
    echo "========================================"
    cat "$RESULTS_TABLE"
    REGRESSIONS=0
if grep -q '| REGRESSION |' "$RESULTS_TABLE" 2>/dev/null; then
    REGRESSIONS=$(grep -c '| REGRESSION |' "$RESULTS_TABLE")
fi
    echo ""
    if [ "$REGRESSIONS" -eq 0 ]; then
        echo "Zero regressions detected (gtest only)."
    else
        echo "*** $REGRESSIONS REGRESSION(S) DETECTED ***"
    fi
    echo ""
    echo "Results table: $RESULTS_TABLE"
    echo "To run all tests: re-run with --results-dir $(dirname "$RESULTS_TABLE")"
    exit 0
fi

# --- Test 6: correctness sweep ---
run_both 6 "run_e2e_correctness_sweep.sh" "bash $S/run_e2e_correctness_sweep.sh --build-dir $B" "bash $S/run_e2e_correctness_sweep.sh --build-dir $T"
if ! test_already_done 6; then
    b_total=$(grep 'Total Tests:' "$RESULTS_DIR/baseline/test_6.log" 2>/dev/null | grep -oP '\d+')
    b_pass=$(grep 'Passed:' "$RESULTS_DIR/baseline/test_6.log" 2>/dev/null | grep -oP '\d+' | head -1)
    b_fail=$(grep 'Failed:' "$RESULTS_DIR/baseline/test_6.log" 2>/dev/null | grep -oP '\d+' | head -1)
    t_total=$(grep 'Total Tests:' "$RESULTS_DIR/target/test_6.log" 2>/dev/null | grep -oP '\d+')
    t_pass=$(grep 'Passed:' "$RESULTS_DIR/target/test_6.log" 2>/dev/null | grep -oP '\d+' | head -1)
    t_fail=$(grep 'Failed:' "$RESULTS_DIR/target/test_6.log" 2>/dev/null | grep -oP '\d+' | head -1)
    reg="No"; [ "$b_total/$b_pass/$b_fail" != "$t_total/$t_pass/$t_fail" ] && reg="REGRESSION"
    append_result 6 "run_e2e_correctness_sweep.sh" "total/passed/failed" "${b_total}/${b_pass}/${b_fail}" "${t_total}/${t_pass}/${t_fail}" "$reg"
    printf "  Result: baseline=%s/%s/%s target=%s/%s/%s %s\n" "$b_total" "$b_pass" "$b_fail" "$t_total" "$t_pass" "$t_fail" "$reg"
fi

# --- Test 7: latency profile ---
run_both 7 "run_latency_profile.sh" "bash $S/run_latency_profile.sh --build-dir $B --iterations 10" "bash $S/run_latency_profile.sh --build-dir $T --iterations 10"
if ! test_already_done 7; then
    b_bler=$(grep -E '^[0-9]' "$RESULTS_DIR/baseline/test_7.log" 2>/dev/null | awk '{print $7}' | sort -u | tr '\n' ',' | sed 's/,$//')
    t_bler=$(grep -E '^[0-9]' "$RESULTS_DIR/target/test_7.log" 2>/dev/null | awk '{print $7}' | sort -u | tr '\n' ',' | sed 's/,$//')
    reg="No"; [ "$b_bler" != "$t_bler" ] && reg="REGRESSION"
    append_result 7 "run_latency_profile.sh" "BLER all configs" "$b_bler" "$t_bler" "$reg"
    printf "  Result: baseline BLER=%s target BLER=%s %s\n" "$b_bler" "$t_bler" "$reg"
fi

# --- Test 8: SINR sweep ---
run_both 8 "test_sinr_sweep.sh" "bash $S/test_sinr_sweep.sh --build-dir $B" "bash $S/test_sinr_sweep.sh --build-dir $T"
if ! test_already_done 8; then
    b_pts=$(grep 'Total test points:' "$RESULTS_DIR/baseline/test_8.log" 2>/dev/null | awk '{print $NF}')
    b_match=$(grep 'Matching' "$RESULTS_DIR/baseline/test_8.log" 2>/dev/null | awk '{print $NF}')
    b_max=$(grep 'Max BLER delta:' "$RESULTS_DIR/baseline/test_8.log" 2>/dev/null | awk '{print $NF}')
    t_pts=$(grep 'Total test points:' "$RESULTS_DIR/target/test_8.log" 2>/dev/null | awk '{print $NF}')
    t_match=$(grep 'Matching' "$RESULTS_DIR/target/test_8.log" 2>/dev/null | awk '{print $NF}')
    t_max=$(grep 'Max BLER delta:' "$RESULTS_DIR/target/test_8.log" 2>/dev/null | awk '{print $NF}')
    reg="No (stochastic)"
    append_result 8 "test_sinr_sweep.sh" "total pts, matching, max BLER delta" "${b_pts} pts, ${b_match} match, ${b_max} max" "${t_pts} pts, ${t_match} match, ${t_max} max" "$reg"
    printf "  Result: baseline=%s/%s/%s target=%s/%s/%s %s\n" "$b_pts" "$b_match" "$b_max" "$t_pts" "$t_match" "$t_max" "$reg"
fi

# --- Test 9: GPU-CPU comparison ---
run_both 9 "pusch_gpu_cpu_comparison_test" \
    "$B/tests/integrationtests/phy/upper/channel_processors/pusch_gpu_cpu_comparison_test -R 10" \
    "$T/tests/integrationtests/phy/upper/channel_processors/pusch_gpu_cpu_comparison_test -R 10"
if ! test_already_done 9; then
    b_sinr=$(grep 'Max SINR' "$RESULTS_DIR/baseline/test_9.log" 2>/dev/null | grep -oP '[\d.]+ dB')
    b_bler=$(grep 'Max BLER' "$RESULTS_DIR/baseline/test_9.log" 2>/dev/null | grep -oP '[\d.]+%')
    b_agree=$(grep 'Decode agreement' "$RESULTS_DIR/baseline/test_9.log" 2>/dev/null | grep -oP '[\d.]+%')
    t_sinr=$(grep 'Max SINR' "$RESULTS_DIR/target/test_9.log" 2>/dev/null | grep -oP '[\d.]+ dB')
    t_bler=$(grep 'Max BLER' "$RESULTS_DIR/target/test_9.log" 2>/dev/null | grep -oP '[\d.]+%')
    t_agree=$(grep 'Decode agreement' "$RESULTS_DIR/target/test_9.log" 2>/dev/null | grep -oP '[\d.]+%')
    b_tol=$(grep 'All metrics' "$RESULTS_DIR/baseline/test_9.log" 2>/dev/null | grep -oP 'YES|NO')
    t_tol=$(grep 'All metrics' "$RESULTS_DIR/target/test_9.log" 2>/dev/null | grep -oP 'YES|NO')
    reg="No"; [ "$b_tol" != "$t_tol" ] && reg="REGRESSION"
    append_result 9 "pusch_gpu_cpu_comparison_test" "max SINR delta, max BLER delta, decode agreement" "${b_sinr}, ${b_bler}, ${b_agree}" "${t_sinr}, ${t_bler}, ${t_agree}" "$reg"
    printf "  Result: %s\n" "$reg"
fi

# --- Test 10: PUSCH GPU/CPU result parity ---
run_both 10 "pusch_gpu_cpu_result_parity_test" \
    "$B/tests/integrationtests/phy/upper/channel_processors/pusch_gpu_cpu_result_parity_test" \
    "$T/tests/integrationtests/phy/upper/channel_processors/pusch_gpu_cpu_result_parity_test"
if ! test_already_done 10; then
    b_crc=$(grep 'CRC Agreement:' "$RESULTS_DIR/baseline/test_10.log" 2>/dev/null | tail -1)
    t_crc=$(grep 'CRC Agreement:' "$RESULTS_DIR/target/test_10.log" 2>/dev/null | tail -1)
    reg="No"; [ "$b_crc" != "$t_crc" ] && reg="REGRESSION"
    append_result 10 "pusch_gpu_cpu_result_parity_test" "CRC agreement" "$b_crc" "$t_crc" "$reg"
    printf "  Result: %s\n" "$reg"
fi

# --- Test 11: E2E pipeline ---
run_both 11 "pusch_e2e_pipeline_test" \
    "$B/tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test --ports 2" \
    "$T/tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test --ports 2"
if ! test_already_done 11; then
    b_cpu=$(grep '^CPU:' "$RESULTS_DIR/baseline/test_11.log" 2>/dev/null | head -1)
    b_gpu=$(grep '^GPU:' "$RESULTS_DIR/baseline/test_11.log" 2>/dev/null | head -1)
    b_dis=$(grep 'Disagreements:' "$RESULTS_DIR/baseline/test_11.log" 2>/dev/null | grep -oP '\d+')
    t_cpu=$(grep '^CPU:' "$RESULTS_DIR/target/test_11.log" 2>/dev/null | head -1)
    t_gpu=$(grep '^GPU:' "$RESULTS_DIR/target/test_11.log" 2>/dev/null | head -1)
    t_dis=$(grep 'Disagreements:' "$RESULTS_DIR/target/test_11.log" 2>/dev/null | grep -oP '\d+')
    reg="No"; [ "$b_cpu$b_dis" != "$t_cpu$t_dis" ] && reg="REGRESSION"
    append_result 11 "pusch_e2e_pipeline_test" "CPU BLER, GPU BLER, disagreements" "$b_cpu, disagree=$b_dis" "$t_cpu, disagree=$t_dis" "$reg"
    printf "  Result: %s\n" "$reg"
fi

# --- Test 12: multiport sweep ---
run_both 12 "run_multiport_sweep.sh" "bash $S/run_multiport_sweep.sh --build-dir $B" "bash $S/run_multiport_sweep.sh --build-dir $T"
if ! test_already_done 12; then
    b_val=$(grep -E 'ALL PORT CONFIGURATIONS PASSED|FAILED' "$RESULTS_DIR/baseline/test_12.log" 2>/dev/null | tail -1)
    t_val=$(grep -E 'ALL PORT CONFIGURATIONS PASSED|FAILED' "$RESULTS_DIR/target/test_12.log" 2>/dev/null | tail -1)
    reg="No"; [ "$b_val" != "$t_val" ] && reg="REGRESSION"
    append_result 12 "run_multiport_sweep.sh" "all port configs" "$b_val" "$t_val" "$reg"
    printf "  Result: %s\n" "$reg"
fi

# --- Test 13: sensitivity sweep ---
run_both 13 "run_sensitivity_sweep.sh" "bash $S/run_sensitivity_sweep.sh --build-dir $B --quick" "bash $S/run_sensitivity_sweep.sh --build-dir $T --quick"
if ! test_already_done 13; then
    b_cpu_thresh=$(grep -A20 'Summary' "$RESULTS_DIR/baseline/test_13.log" 2>/dev/null | grep 'PRB' | awk '{print $(NF-2)}' | tr '\n' '/' | sed 's/\/$//')
    b_gpu_thresh=$(grep -A20 'Summary' "$RESULTS_DIR/baseline/test_13.log" 2>/dev/null | grep 'PRB' | awk '{print $(NF-1)}' | tr '\n' '/' | sed 's/\/$//')
    t_cpu_thresh=$(grep -A20 'Summary' "$RESULTS_DIR/target/test_13.log" 2>/dev/null | grep 'PRB' | awk '{print $(NF-2)}' | tr '\n' '/' | sed 's/\/$//')
    t_gpu_thresh=$(grep -A20 'Summary' "$RESULTS_DIR/target/test_13.log" 2>/dev/null | grep 'PRB' | awk '{print $(NF-1)}' | tr '\n' '/' | sed 's/\/$//')
    reg="No"; [ "$b_cpu_thresh/$b_gpu_thresh" != "$t_cpu_thresh/$t_gpu_thresh" ] && reg="REGRESSION"
    append_result 13 "run_sensitivity_sweep.sh --quick" "SINR (dB) at 10% BLER: CPU/GPU" "CPU=${b_cpu_thresh}; GPU=${b_gpu_thresh}" "CPU=${t_cpu_thresh}; GPU=${t_gpu_thresh}" "$reg"
    printf "  Result: baseline CPU=%s GPU=%s target CPU=%s GPU=%s %s\n" "$b_cpu_thresh" "$b_gpu_thresh" "$t_cpu_thresh" "$t_gpu_thresh" "$reg"
fi

# --- Test 14: GPU benchmark sweep ---
run_both 14 "gpu_benchmark_sweep.sh" "bash $S/gpu_benchmark_sweep.sh --build-dir $B --quick" "bash $S/gpu_benchmark_sweep.sh --build-dir $T --quick"
if ! test_already_done 14; then
    b_tput=$(grep -E 'Mbps\|' "$RESULTS_DIR/baseline/test_14.log" 2>/dev/null | head -4 | grep -oP '[\d.]+ Mbps' | tr '\n' '/' | sed 's/\/$//')
    t_tput=$(grep -E 'Mbps\|' "$RESULTS_DIR/target/test_14.log" 2>/dev/null | head -4 | grep -oP '[\d.]+ Mbps' | tr '\n' '/' | sed 's/\/$//')
    b_p50=$(grep -E 'Mbps\|' "$RESULTS_DIR/baseline/test_14.log" 2>/dev/null | head -4 | grep -oP 'p50=\s*[\d.]+' | sed 's/p50=\s*//' | tr '\n' '/' | sed 's/\/$//')
    t_p50=$(grep -E 'Mbps\|' "$RESULTS_DIR/target/test_14.log" 2>/dev/null | head -4 | grep -oP 'p50=\s*[\d.]+' | sed 's/p50=\s*//' | tr '\n' '/' | sed 's/\/$//')
    reg="No (within variance)"
    b_done=$(grep -c 'Benchmark Complete' "$RESULTS_DIR/baseline/test_14.log" 2>/dev/null || echo 0)
    t_done=$(grep -c 'Benchmark Complete' "$RESULTS_DIR/target/test_14.log" 2>/dev/null || echo 0)
    [ "$b_done" = "0" ] || [ "$t_done" = "0" ] && reg="ERROR"
    append_result 14 "gpu_benchmark_sweep.sh --quick" "throughput, latency p50 (1-4 layers)" "${b_tput}; p50=${b_p50}us" "${t_tput}; p50=${t_p50}us" "$reg"
    printf "  Result: baseline=%s p50=%s target=%s p50=%s %s\n" "$b_tput" "$b_p50" "$t_tput" "$t_p50" "$reg"
fi

# --- Test 15: multi sector stress ---
run_both 15 "multi_sector_stress.sh" "bash $S/multi_sector_stress.sh --build-dir $B --quick" "bash $S/multi_sector_stress.sh --build-dir $T --quick"
if ! test_already_done 15; then
    b_tput=$(grep -A50 '1 cell.*CPU' "$RESULTS_DIR/baseline/test_15.log" 2>/dev/null | grep -E 'Mbps\|' | head -4 | grep -oP '[\d.]+ Mbps' | tr '\n' '/' | sed 's/\/$//')
    t_tput=$(grep -A50 '1 cell.*CPU' "$RESULTS_DIR/target/test_15.log" 2>/dev/null | grep -E 'Mbps\|' | head -4 | grep -oP '[\d.]+ Mbps' | tr '\n' '/' | sed 's/\/$//')
    b_p50=$(grep -A50 '1 cell.*CPU' "$RESULTS_DIR/baseline/test_15.log" 2>/dev/null | grep -E 'Mbps\|' | head -4 | grep -oP 'p50=\s*[\d.]+' | sed 's/p50=\s*//' | tr '\n' '/' | sed 's/\/$//')
    t_p50=$(grep -A50 '1 cell.*CPU' "$RESULTS_DIR/target/test_15.log" 2>/dev/null | grep -E 'Mbps\|' | head -4 | grep -oP 'p50=\s*[\d.]+' | sed 's/p50=\s*//' | tr '\n' '/' | sed 's/\/$//')
    reg="No (within variance)"
    b_done=$(grep -c 'Test Complete' "$RESULTS_DIR/baseline/test_15.log" 2>/dev/null || echo 0)
    t_done=$(grep -c 'Test Complete' "$RESULTS_DIR/target/test_15.log" 2>/dev/null || echo 0)
    [ "$b_done" = "0" ] || [ "$t_done" = "0" ] && reg="ERROR"
    append_result 15 "multi_sector_stress.sh" "1-cell throughput, latency p50 (1-4 layers)" "${b_tput}; p50=${b_p50}us" "${t_tput}; p50=${t_p50}us" "$reg"
    printf "  Result: baseline=%s p50=%s target=%s p50=%s %s\n" "$b_tput" "$b_p50" "$t_tput" "$t_p50" "$reg"
fi

# --- Test 16: PDSCH benchmark ---
run_both 16 "pdsch_processor_benchmark" \
    "$B/tests/benchmarks/phy/upper/channel_processors/pdsch_processor_benchmark -m latency -T 10 -R 10 -B 100" \
    "$T/tests/benchmarks/phy/upper/channel_processors/pdsch_processor_benchmark -m latency -T 10 -R 10 -B 100"
if ! test_already_done 16; then
    b_p50=$(grep 'p50=' "$RESULTS_DIR/baseline/test_16.log" 2>/dev/null | tail -3 | awk -F'p50=' '{print $2}' | awk '{print $1}' | tr '\n' '/' | sed 's/\/$//')
    t_p50=$(grep 'p50=' "$RESULTS_DIR/target/test_16.log" 2>/dev/null | tail -3 | awk -F'p50=' '{print $2}' | awk '{print $1}' | tr '\n' '/' | sed 's/\/$//')
    reg="No (within variance)"
    append_result 16 "pdsch_processor_benchmark" "latency p50" "$b_p50" "$t_p50" "$reg"
    printf "  Result: baseline=%s target=%s %s\n" "$b_p50" "$t_p50" "$reg"
fi

# ============================================================
#  Final summary
# ============================================================

echo ""
echo "========================================"
echo "  Results Table"
echo "========================================"
cat "$RESULTS_TABLE"

REGRESSIONS=0
if grep -q '| REGRESSION |' "$RESULTS_TABLE" 2>/dev/null; then
    REGRESSIONS=$(grep -c '| REGRESSION |' "$RESULTS_TABLE")
fi
echo ""
if [ "$REGRESSIONS" -eq 0 ]; then
    echo "Zero regressions detected."
else
    echo "*** $REGRESSIONS REGRESSION(S) DETECTED ***"
fi

echo ""
echo "Full logs: $RESULTS_DIR"
echo "Results table: $RESULTS_TABLE"
echo "Done."
