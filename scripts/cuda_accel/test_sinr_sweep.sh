#!/bin/bash
# E2E GPU vs CPU BLER Sweep
# Compares GPU and CPU BLER across MCS indices, PRB widths, and SINR values

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
TEST_BIN="$BUILD_DIR/tests/integrationtests/phy/upper/channel_processors/pusch_64qam_stress_test"

if [ ! -x "$TEST_BIN" ]; then
    echo "ERROR: Test binary not found: $TEST_BIN"
    echo "Build with: cmake --build build --target pusch_64qam_stress_test"
    exit 1
fi

ITERS=50

# MCS configs: index, mcs_table, modulation name, SINR sweep range
# One representative MCS per modulation order (QPSK, 16QAM, 64QAM, 256QAM)
MCS_CONFIGS=(
    "4   qam64  QPSK          -2 0 2 4 6 8"
    "14  qam64  16QAM          4 6 8 10 12 14"
    "22  qam64  64QAM         10 12 14 16 18 20"
    "27  qam64  64QAM_HR      14 16 18 20 22 24"
    "25  qam256 256QAM        16 18 20 22 24 26"
)

PRB_SIZES=(5 25 52 106 273)

echo ""
echo "=================================================================="
echo "  E2E GPU vs CPU BLER Sweep"
echo "  Iterations per point: $ITERS"
echo "  PRB sizes: ${PRB_SIZES[*]}"
echo "=================================================================="

TOTAL_TESTS=0
TOTAL_MATCH=0
TOTAL_GPU_BETTER=0
TOTAL_CPU_BETTER=0
MAX_BLER_DELTA=0
RESULTS_FILE=$(mktemp)

for PRB in "${PRB_SIZES[@]}"; do
    echo ""
    echo "=================================================================="
    echo "  PRB = $PRB"
    echo "=================================================================="
    printf "  %-12s" "MCS"
    # Print SINR header (will be filled per MCS)
    echo ""

    for mcs_line in "${MCS_CONFIGS[@]}"; do
        read -r MCS MCS_TABLE MOD_NAME SINR_LIST <<< "$mcs_line"
        read -ra SINRS <<< "$SINR_LIST"

        printf "  MCS%-2d %-10s |" "$MCS" "($MOD_NAME)"

        for SINR in "${SINRS[@]}"; do
            OUTPUT=$("$TEST_BIN" --prb "$PRB" --sinr "$SINR" --mcs "$MCS" --mcs-table "$MCS_TABLE" --iterations "$ITERS" 2>&1)

            CPU_BLER=$(echo "$OUTPUT" | grep "^CPU:" | grep -oP '\(\K[0-9.]+(?=% BLER)' || echo "?")
            GPU_BLER=$(echo "$OUTPUT" | grep "^GPU:" | grep -oP '\(\K[0-9.]+(?=% BLER)' || echo "?")

            TOTAL_TESTS=$((TOTAL_TESTS + 1))

            if [ "$CPU_BLER" = "?" ] || [ "$GPU_BLER" = "?" ]; then
                printf " %3sdB:ERR |" "$SINR"
                continue
            fi

            # Compare
            DELTA=$(awk "BEGIN {printf \"%.1f\", $GPU_BLER - $CPU_BLER}")
            ABS_DELTA=$(awk "BEGIN {d=$GPU_BLER - $CPU_BLER; printf \"%.1f\", (d<0?-d:d)}")

            # Track max delta
            IS_BIGGER=$(awk "BEGIN {print ($ABS_DELTA > $MAX_BLER_DELTA) ? 1 : 0}")
            if [ "$IS_BIGGER" -eq 1 ]; then
                MAX_BLER_DELTA="$ABS_DELTA"
            fi

            if [ "$CPU_BLER" = "0.0" ] && [ "$GPU_BLER" = "0.0" ]; then
                printf " %3sdB: 0/0  |" "$SINR"
                TOTAL_MATCH=$((TOTAL_MATCH + 1))
            elif [ "$CPU_BLER" = "100.0" ] && [ "$GPU_BLER" = "100.0" ]; then
                printf " %3sdB:##/## |" "$SINR"
                TOTAL_MATCH=$((TOTAL_MATCH + 1))
            else
                CPU_INT=$(printf "%.0f" "$CPU_BLER")
                GPU_INT=$(printf "%.0f" "$GPU_BLER")
                printf " %3sdB:%2d/%2d |" "$SINR" "$CPU_INT" "$GPU_INT"

                IS_MATCH=$(awk "BEGIN {d=$GPU_BLER - $CPU_BLER; print ((d<0?-d:d) < 5.0) ? 1 : 0}")
                if [ "$IS_MATCH" -eq 1 ]; then
                    TOTAL_MATCH=$((TOTAL_MATCH + 1))
                fi

                IS_GPU_BETTER=$(awk "BEGIN {print ($GPU_BLER < $CPU_BLER - 2.0) ? 1 : 0}")
                IS_CPU_BETTER=$(awk "BEGIN {print ($CPU_BLER < $GPU_BLER - 2.0) ? 1 : 0}")
                if [ "$IS_GPU_BETTER" -eq 1 ]; then
                    TOTAL_GPU_BETTER=$((TOTAL_GPU_BETTER + 1))
                fi
                if [ "$IS_CPU_BETTER" -eq 1 ]; then
                    TOTAL_CPU_BETTER=$((TOTAL_CPU_BETTER + 1))
                fi
            fi

            # Save to results file for summary table
            echo "$PRB $MCS $MOD_NAME $SINR $CPU_BLER $GPU_BLER $DELTA" >> "$RESULTS_FILE"
        done
        echo ""
    done
done

echo ""
echo "=================================================================="
echo "  Legend: SNR:CPU/GPU (BLER %)    0/0 = both 0%    ##/## = both 100%"
echo "=================================================================="
echo ""
echo "=================================================================="
echo "  SUMMARY"
echo "=================================================================="
echo "  Total test points:     $TOTAL_TESTS"
echo "  Matching (<5% delta):  $TOTAL_MATCH"
echo "  GPU better (>2% gap):  $TOTAL_GPU_BETTER"
echo "  CPU better (>2% gap):  $TOTAL_CPU_BETTER"
echo "  Max BLER delta:        ${MAX_BLER_DELTA}%"
echo ""

# Print waterfall summary: for each MCS+PRB, show the SINR where BLER drops to 0
echo "=================================================================="
echo "  BLER Waterfall: SINR (dB) for 0% BLER"
echo "=================================================================="
printf "  %-15s" "MCS"
for PRB in "${PRB_SIZES[@]}"; do
    printf " %6s PRB    " "$PRB"
done
echo ""
printf "  %-15s" ""
for PRB in "${PRB_SIZES[@]}"; do
    printf " CPU  GPU     "
done
echo ""
echo "  -------------------------------------------------------------------"

for mcs_line in "${MCS_CONFIGS[@]}"; do
    read -r MCS _ MOD_NAME _ <<< "$mcs_line"
    printf "  MCS%-2d %-7s " "$MCS" "$MOD_NAME"

    for PRB in "${PRB_SIZES[@]}"; do
        CPU_THRESH="-"
        GPU_THRESH="-"
        # Find first SINR where BLER = 0 for this MCS+PRB
        while IFS=' ' read -r r_prb r_mcs r_mod r_sinr r_cpu r_gpu r_delta; do
            if [ "$r_prb" = "$PRB" ] && [ "$r_mcs" = "$MCS" ]; then
                if [ "$CPU_THRESH" = "-" ] && [ "$r_cpu" = "0.0" ]; then
                    CPU_THRESH="$r_sinr"
                fi
                if [ "$GPU_THRESH" = "-" ] && [ "$r_gpu" = "0.0" ]; then
                    GPU_THRESH="$r_sinr"
                fi
            fi
        done < "$RESULTS_FILE"
        printf " %3s  %3s     " "$CPU_THRESH" "$GPU_THRESH"
    done
    echo ""
done

rm -f "$RESULTS_FILE"

echo ""
echo "  (Lower SINR threshold = better performance)"
echo ""
