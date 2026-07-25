#!/bin/bash
#
# GPU vs CPU Benchmark Sweep Script
#
# This script runs comprehensive benchmarks to compare GPU and CPU PUSCH processing
# performance across different configurations. It sweeps thread counts, batch sizes,
# and measurement modes to find optimal configurations.
#
# Usage: ./gpu_benchmark_sweep.sh [benchmark_binary_path]
#
# Environment Variables:
#   OCUDU_PUSCH_ACCELERATION_TRACE=1  - Enable detailed GPU path tracing logs
#   CUDA_VISIBLE_DEVICES=0   - Select GPU device
#
# Output: Results are printed to stdout and optionally saved to a file.

set -e

# Configuration
BENCHMARK_BIN=""
PROFILE="scs30_100MHz_256qam_rv0_4port_nlayer"
REPETITIONS=20
OUTPUT_DIR="./benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/gpu_benchmark_sweep_${TIMESTAMP}.txt"

# Thread counts to test
THREAD_COUNTS="1 2 4 8 16 32"

# Batch sizes to test
BATCH_SIZES="10 50 100 200 500"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo "============================================================"
    echo " $1"
    echo "============================================================"
    echo ""
}

# Check if benchmark binary exists
check_benchmark() {
    if [[ ! -x "$BENCHMARK_BIN" ]]; then
        log_error "Benchmark binary not found or not executable: $BENCHMARK_BIN"
        echo ""
        echo "Please build the benchmark first:"
        echo "  cd build && cmake .. -DENABLE_CUDA=ON && make pusch_processor_benchmark"
        echo ""
        echo "Or specify the path to the benchmark binary:"
        echo "  $0 /path/to/pusch_processor_benchmark"
        exit 1
    fi
    log_success "Found benchmark binary: $BENCHMARK_BIN"
}

# Check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        log_warning "nvidia-smi not found - GPU benchmarks may not work"
    fi
}

# Create output directory
setup_output() {
    mkdir -p "$OUTPUT_DIR"
    log_info "Results will be saved to: $OUTPUT_FILE"
}

# Run single benchmark
run_benchmark() {
    local gpu_flag="$1"
    local threads="$2"
    local batch="$3"
    local mode="$4"
    local description="$5"

    local cmd="$BENCHMARK_BIN $gpu_flag -m $mode -R $REPETITIONS -T $threads -B $batch -P $PROFILE"

    echo ">>> $description"
    echo "Command: $cmd"
    echo "---"

    # Run with timeout to prevent hangs
    if timeout 300 $cmd 2>&1; then
        echo ""
    else
        log_warning "Benchmark failed or timed out"
        echo ""
    fi
}

# Section 1: Thread Count Sweep
run_thread_sweep() {
    log_section "THREAD COUNT SWEEP (Latency Mode)"

    echo "Testing CPU performance across thread counts..."
    echo ""

    for T in $THREAD_COUNTS; do
        run_benchmark "" "$T" "100" "latency" "CPU - $T threads"
    done

    echo ""
    echo "Testing GPU performance across thread counts..."
    echo ""

    for T in $THREAD_COUNTS; do
        run_benchmark "-G" "$T" "100" "latency" "GPU - $T threads"
    done
}

# Section 2: Throughput Mode Comparison
run_throughput_comparison() {
    log_section "THROUGHPUT MODE COMPARISON"

    local THREADS=16

    echo "Running throughput benchmarks with $THREADS threads..."
    echo ""

    # Total throughput - CPU
    run_benchmark "" "$THREADS" "100" "throughput_total" "CPU - Total Throughput"

    # Total throughput - GPU
    run_benchmark "-G" "$THREADS" "100" "throughput_total" "GPU - Total Throughput"

    # Per-thread throughput - CPU
    run_benchmark "" "$THREADS" "100" "throughput_thread" "CPU - Per-Thread Throughput"

    # Per-thread throughput - GPU
    run_benchmark "-G" "$THREADS" "100" "throughput_thread" "GPU - Per-Thread Throughput"

    # All metrics - CPU
    run_benchmark "" "$THREADS" "100" "all" "CPU - All Metrics"

    # All metrics - GPU
    run_benchmark "-G" "$THREADS" "100" "all" "GPU - All Metrics"
}

# Section 3: Batch Size Variations
run_batch_sweep() {
    log_section "BATCH SIZE SWEEP"

    local THREADS=16

    echo "Testing GPU amortization effects with different batch sizes..."
    echo ""

    for B in $BATCH_SIZES; do
        echo "=== Batch Size: $B ==="
        run_benchmark "" "$THREADS" "$B" "latency" "CPU - Batch $B"
        run_benchmark "-G" "$THREADS" "$B" "latency" "GPU - Batch $B"
    done
}

# Section 4: Path Tracing Verification
run_path_verification() {
    log_section "GPU PATH VERIFICATION"

    echo "Running with GPU path tracing enabled to verify code paths..."
    echo "Set OCUDU_PUSCH_ACCELERATION_TRACE=1 to see detailed path selection logs."
    echo ""

    # Run a single benchmark with tracing enabled
    export OCUDU_PUSCH_ACCELERATION_TRACE=1
    run_benchmark "-G" "4" "10" "latency" "GPU with Path Tracing (verify E2E path)"
    unset OCUDU_PUSCH_ACCELERATION_TRACE
}

# Section 5: Quick Comparison
run_quick_comparison() {
    log_section "QUICK CPU vs GPU COMPARISON"

    local THREADS=8
    local BATCH=100

    echo "Direct comparison with optimal settings..."
    echo ""

    echo "=== CPU Baseline ==="
    run_benchmark "" "$THREADS" "$BATCH" "all" "CPU Baseline"

    echo "=== GPU Accelerated ==="
    run_benchmark "-G" "$THREADS" "$BATCH" "all" "GPU Accelerated"
}

# Section 6: Multi-Cell GPU Scaling
run_multi_cell_sweep() {
    log_section "MULTI-CELL GPU SCALING"

    local THREADS_PER_CELL=4
    local BATCH=100
    local MAX_CELLS=4

    echo "Testing how GPU scales with multiple cells sharing resources."
    echo "This models real multi-sector gNB behavior."
    echo ""
    echo "Configuration:"
    echo "  - Threads per cell: $THREADS_PER_CELL"
    echo "  - Batch size: $BATCH"
    echo "  - Max cells: $MAX_CELLS"
    echo ""

    for CELLS in $(seq 1 $MAX_CELLS); do
        local TOTAL_THREADS=$((CELLS * THREADS_PER_CELL))

        echo ""
        echo "=========================================="
        echo " $CELLS cell(s), $TOTAL_THREADS total threads"
        echo "=========================================="
        echo ""

        # CPU baseline
        echo ">>> CPU - $CELLS cell(s)"
        local cpu_cmd="$BENCHMARK_BIN -C $CELLS -T $TOTAL_THREADS -B $BATCH -R $REPETITIONS -m latency -P $PROFILE"
        echo "Command: $cpu_cmd"
        echo "---"
        if timeout 300 $cpu_cmd 2>&1; then
            echo ""
        else
            log_warning "CPU benchmark failed or timed out"
            echo ""
        fi

        # GPU accelerated
        echo ">>> GPU - $CELLS cell(s)"
        local gpu_cmd="$BENCHMARK_BIN -G -C $CELLS -T $TOTAL_THREADS -B $BATCH -R $REPETITIONS -m latency -P $PROFILE"
        echo "Command: $gpu_cmd"
        echo "---"
        if timeout 300 $gpu_cmd 2>&1; then
            echo ""
        else
            log_warning "GPU benchmark failed or timed out"
            echo ""
        fi
    done

    echo ""
    echo "=========================================="
    echo " Multi-Cell Summary"
    echo "=========================================="
    echo ""
    echo "Cells | Threads | Notes"
    echo "------|---------|------"
    for CELLS in $(seq 1 $MAX_CELLS); do
        local TOTAL_THREADS=$((CELLS * THREADS_PER_CELL))
        printf "%-5s | %-7s | Compare CPU vs GPU latency above\n" "$CELLS" "$TOTAL_THREADS"
    done
    echo ""
    echo "Look for:"
    echo "  1. Point where CPU exceeds 500 us (slot duration)"
    echo "  2. GPU latency scaling with cell count"
    echo "  3. Maximum cells before GPU exceeds budget"
}

# Main execution
main() {
    # Parse command line options first
    local run_all=true
    local run_quick=false
    local run_mode=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                run_all=false
                run_quick=true
                shift
                ;;
            --threads-only)
                run_all=false
                run_mode="threads"
                shift
                ;;
            --throughput-only)
                run_all=false
                run_mode="throughput"
                shift
                ;;
            --batch-only)
                run_all=false
                run_mode="batch"
                shift
                ;;
            --verify-only)
                run_all=false
                run_mode="verify"
                shift
                ;;
            --multi-cell-only)
                run_all=false
                run_mode="multi-cell"
                shift
                ;;
            --build-dir)
                if [[ ! -d "$2" ]]; then
                    log_error "Build directory not found: $2"
                    exit 1
                fi
                local resolved_dir
                resolved_dir="$(cd "$2" && pwd)"
                BENCHMARK_BIN="${resolved_dir}/tests/benchmarks/phy/upper/channel_processors/pusch/pusch_processor_benchmark"
                shift 2
                ;;
            -*)
                # Skip unknown options
                shift
                ;;
            *)
                # Assume it's a benchmark binary path
                if [[ -x "$1" ]]; then
                    BENCHMARK_BIN="$1"
                fi
                shift
                ;;
        esac
    done

    if [[ -z "$BENCHMARK_BIN" ]]; then
        log_error "No build dir or benchmark binary specified. Use --build-dir <path> or pass binary path as argument."
        exit 1
    fi

    echo "========================================"
    echo " OCUDU GPU vs CPU Benchmark Sweep"
    echo " Profile: $PROFILE"
    echo " Timestamp: $TIMESTAMP"
    echo "========================================"
    echo ""

    check_benchmark
    check_gpu
    setup_output

    # Redirect all output to file while also displaying
    exec > >(tee -a "$OUTPUT_FILE") 2>&1

    echo "Start time: $(date)"
    echo ""

    # Run the selected mode
    case $run_mode in
        threads)
            run_thread_sweep
            ;;
        throughput)
            run_throughput_comparison
            ;;
        batch)
            run_batch_sweep
            ;;
        verify)
            run_path_verification
            ;;
        multi-cell)
            run_multi_cell_sweep
            ;;
        *)
            # Default behavior - run quick or all
            if $run_quick; then
                run_quick_comparison
            elif $run_all; then
                run_quick_comparison
                run_thread_sweep
                run_throughput_comparison
                run_batch_sweep
                run_multi_cell_sweep
                run_path_verification
            fi
            ;;
    esac

    echo ""
    echo "========================================"
    echo " Benchmark Complete"
    echo " End time: $(date)"
    echo " Results saved to: $OUTPUT_FILE"
    echo "========================================"
}

# Run main with all arguments
main "$@"
