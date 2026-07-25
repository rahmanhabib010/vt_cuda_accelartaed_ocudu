#!/bin/bash
#
# Multi-Sector Stress Test Script
#
# This script tests multi-cell GPU scaling using the benchmark's native -C flag.
# Multiple cells share GPU/factory resources (like a real multi-sector gNB).
#
# Usage: ./multi_sector_stress.sh [OPTIONS] [benchmark_binary_path]
#
# Timing Constraints for 100MHz (30kHz SCS):
#   - Slot duration: 500 us
#   - Symbols/slot: 14
#   - Symbol duration: ~35.7 us
#   - Max processing delay: 5 slots = 2500 us (default)
#
# Environment Variables:
#   OCUDU_PUSCH_ACCELERATION_TRACE=1  - Enable detailed GPU path tracing logs
#   MAX_SECTORS=4            - Maximum number of sectors to test (default: 4)
#

set -e

# Configuration
BENCHMARK_BIN=""
PROFILE="scs30_100MHz_256qam_rv0_4port_nlayer"
REPETITIONS=20
THREADS_PER_CELL=4
BATCH_SIZE=100
MAX_SECTORS="${MAX_SECTORS:-4}"

OUTPUT_DIR="./benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/multi_sector_stress_${TIMESTAMP}.txt"

# Timing parameters for 100MHz 30kHz SCS (microseconds)
SLOT_DURATION_US=500
SYMBOL_DURATION_US=35.7
MAX_PROCESSING_SLOTS=5
MAX_PROCESSING_TIME_US=$((SLOT_DURATION_US * MAX_PROCESSING_SLOTS))

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN} $1${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
}

# Check if benchmark binary exists
check_benchmark() {
    if [[ ! -x "$BENCHMARK_BIN" ]]; then
        log_error "Benchmark binary not found or not executable: $BENCHMARK_BIN"
        echo ""
        echo "Please build the benchmark first:"
        echo "  cd build && cmake .. -DENABLE_CUDA=ON && make pusch_processor_benchmark"
        exit 1
    fi
    log_success "Found benchmark binary: $BENCHMARK_BIN"
}

# Check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU Status:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    else
        log_warning "nvidia-smi not found"
    fi
}

# Create output directory
setup_output() {
    mkdir -p "$OUTPUT_DIR"
    log_info "Results will be saved to: $OUTPUT_FILE"
}

# Run multi-cell benchmark using native -C flag
# This correctly models shared GPU resources across cells
run_multi_cell_test() {
    local num_cells="$1"
    local gpu_flag="$2"
    local mode_name="CPU"
    local total_threads=$((num_cells * THREADS_PER_CELL))

    if [[ -n "$gpu_flag" ]]; then
        mode_name="GPU"
    fi

    echo ""
    echo ">>> $num_cells cell(s), $total_threads threads ($mode_name)"
    echo "---"

    local cmd="$BENCHMARK_BIN $gpu_flag -C $num_cells -T $total_threads -B $BATCH_SIZE -R $REPETITIONS -m latency -P $PROFILE"
    echo "Command: $cmd"
    echo ""

    # Run with timeout and capture output
    local output
    if output=$(timeout 300 $cmd 2>&1); then
        echo "$output"
        echo ""
    else
        log_warning "Benchmark failed or timed out"
        echo ""
    fi
}

# Parse latency from benchmark output (50th percentile)
parse_latency_from_output() {
    local output="$1"
    # The benchmark prints percentiles like: "50.00%   152.2"
    echo "$output" | grep -E "^\s*50\.00%" | awk '{print $2}' | head -1
}

# Calculate theoretical max sectors
calculate_max_sectors() {
    local per_sector_latency="$1"

    if (( $(echo "$per_sector_latency > 0" | bc -l) )); then
        local max=$(echo "scale=2; $MAX_PROCESSING_TIME_US / $per_sector_latency" | bc)
        echo "$max"
    else
        echo "N/A"
    fi
}

# Run stress test with cell scaling
run_stress_test() {
    local gpu_flag="$1"
    local mode_name="CPU"

    if [[ -n "$gpu_flag" ]]; then
        mode_name="GPU"
    fi

    log_section "$mode_name MULTI-CELL STRESS TEST"

    echo "Configuration:"
    echo "  - Mode:               $mode_name"
    echo "  - Profile:            $PROFILE"
    echo "  - Threads per cell:   $THREADS_PER_CELL"
    echo "  - Batch size:         $BATCH_SIZE"
    echo "  - Repetitions:        $REPETITIONS"
    echo "  - Max cells:          $MAX_SECTORS"
    echo ""
    echo "Timing Constraints (100MHz, 30kHz SCS):"
    echo "  - Slot duration:      ${SLOT_DURATION_US} us"
    echo "  - Symbol duration:    ${SYMBOL_DURATION_US} us"
    echo "  - Max processing:     ${MAX_PROCESSING_TIME_US} us (${MAX_PROCESSING_SLOTS} slots)"
    echo ""

    # Test increasing cell counts
    echo "=========================================="
    echo " Cell Scaling Test"
    echo "=========================================="

    for N in $(seq 1 $MAX_SECTORS); do
        run_multi_cell_test $N "$gpu_flag"
    done
}

# Run CPU vs GPU comparison
run_comparison() {
    log_section "CPU vs GPU MULTI-CELL COMPARISON"

    echo "Configuration:"
    echo "  - Profile:            $PROFILE"
    echo "  - Threads per cell:   $THREADS_PER_CELL"
    echo "  - Batch size:         $BATCH_SIZE"
    echo "  - Repetitions:        $REPETITIONS"
    echo "  - Max cells:          $MAX_SECTORS"
    echo ""
    echo "Timing Budget: ${SLOT_DURATION_US} us/slot, ${MAX_PROCESSING_TIME_US} us max"
    echo ""

    # Results storage
    declare -a cpu_results
    declare -a gpu_results

    # Test each cell count
    for N in $(seq 1 $MAX_SECTORS); do
        local total_threads=$((N * THREADS_PER_CELL))

        echo ""
        echo "=========================================="
        echo " Testing $N cell(s) ($total_threads threads)"
        echo "=========================================="

        echo ""
        echo "--- CPU ---"
        run_multi_cell_test $N ""

        echo ""
        echo "--- GPU ---"
        run_multi_cell_test $N "-G"
    done

    echo ""
    echo "=========================================="
    echo " Summary"
    echo "=========================================="
    echo ""
    echo "Cells | Threads | CPU vs GPU"
    echo "------|---------|------------"
    for N in $(seq 1 $MAX_SECTORS); do
        local total_threads=$((N * THREADS_PER_CELL))
        printf "%-5s | %-7s | See results above\n" "$N" "$total_threads"
    done
    echo ""
    echo "To determine optimal configuration, look for:"
    echo "  1. CPU latency exceeding slot duration (${SLOT_DURATION_US} us)"
    echo "  2. GPU latency remaining under budget"
    echo "  3. GPU speedup factor at each cell count"
}

# Quick single-cell comparison
run_quick_comparison() {
    log_section "QUICK CPU vs GPU COMPARISON (Single Cell)"

    echo "--- CPU Baseline ---"
    run_multi_cell_test 1 ""

    echo "--- GPU Accelerated ---"
    run_multi_cell_test 1 "-G"
}

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS] [benchmark_binary_path]"
    echo ""
    echo "Options:"
    echo "  --cpu         Run CPU-only tests"
    echo "  --gpu         Run GPU-accelerated tests"
    echo "  --compare     Run both CPU and GPU and compare (default)"
    echo "  --quick       Quick single-cell comparison"
    echo "  --cells N     Set maximum number of cells to test (default: 4)"
    echo "  --threads N   Set threads per cell (default: 4)"
    echo "  --help        Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  OCUDU_PUSCH_ACCELERATION_TRACE=1  Enable GPU path tracing"
    echo "  MAX_SECTORS=N            Maximum cells to test"
    echo ""
    echo "This script uses the benchmark's native -C flag for multi-cell testing."
    echo "Multiple cells share GPU/factory resources, accurately modeling real"
    echo "multi-sector gNB behavior."
}

# Main
main() {
    local run_mode="compare"  # Default to comparison

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cpu)
                run_mode="cpu"
                shift
                ;;
            --gpu)
                run_mode="gpu"
                shift
                ;;
            --compare)
                run_mode="compare"
                shift
                ;;
            --quick)
                run_mode="quick"
                shift
                ;;
            --cells)
                MAX_SECTORS="$2"
                shift 2
                ;;
            --threads)
                THREADS_PER_CELL="$2"
                shift 2
                ;;
            --build-dir)
                if [[ ! -d "$2" ]]; then
                    echo "Error: build directory not found: $2"
                    exit 1
                fi
                local resolved_dir
                resolved_dir="$(cd "$2" && pwd)"
                BENCHMARK_BIN="${resolved_dir}/tests/benchmarks/phy/upper/channel_processors/pusch/pusch_processor_benchmark"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                if [[ -x "$1" ]]; then
                    BENCHMARK_BIN="$1"
                fi
                shift
                ;;
        esac
    done

    if [[ -z "$BENCHMARK_BIN" ]]; then
        echo "Error: no build dir or benchmark binary specified. Use --build-dir <path>."
        exit 1
    fi

    echo "========================================"
    echo " OCUDU Multi-Cell Stress Test"
    echo " (Native multi-cell via -C flag)"
    echo " Timestamp: $TIMESTAMP"
    echo "========================================"
    echo ""

    check_benchmark
    check_gpu
    setup_output

    # Redirect output to file
    exec > >(tee -a "$OUTPUT_FILE") 2>&1

    echo "Start time: $(date)"
    echo ""

    case $run_mode in
        cpu)
            run_stress_test ""
            ;;
        gpu)
            run_stress_test "-G"
            ;;
        compare)
            run_comparison
            ;;
        quick)
            run_quick_comparison
            ;;
    esac

    echo ""
    echo "========================================"
    echo " Test Complete"
    echo " End time: $(date)"
    echo " Results saved to: $OUTPUT_FILE"
    echo "========================================"
}

main "$@"
