#!/usr/bin/env bash
# Run normal CP-OFDM Type-1 DMRS CPU-vs-GPU sweeps for UL and DL.
#
# Coverage:
#   - PUSCH sensitivity: CPU vs GPU comparison from pusch_e2e_sensitivity_sweep.
#   - PUSCH latency: CPU vs GPU comparison from pusch_e2e_pipeline_test.
#   - PDSCH latency: CPU and GPU runs from pdsch_gpu_latency_benchmark.
#
# Defaults target 30 kHz SCS bandwidth equivalents:
#   20 MHz -> 51 PRB, 40 MHz -> 106 PRB, 100 MHz -> 273 PRB.
#
# Default topologies:
#   - UL: 1L/1P, 2L/4P, 4L/8P.
#   - DL: same layer counts, with ports=layers because the PDSCH latency bench
#         only supports identity precoding when layers=ports for multi-layer.
#
# Example:
#   scripts/cuda_accel/run_type1_dmrs_ul_dl_gpu_cpu_sweeps.sh \
#     --build-dir build-cuda-default-check --quick

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUILD_DIR="${REPO_ROOT}/build-cuda-default-check"
OUT_DIR=""
PRBS="51,106,273"
TOPOLOGIES="1x1,2x4,4x8"
MCS=20
MCS_TABLE="qam64"
PUSCH_SNR_START="-5.0"
PUSCH_SNR_STOP="25.0"
PUSCH_SNR_STEP="1.0"
PUSCH_FRAMES=100
PUSCH_LATENCY_SINR="25"
PUSCH_LATENCY_ITERATIONS=100
PUSCH_LATENCY_WARMUP=5
PDSCH_LATENCY_ITERATIONS=100
PDSCH_LATENCY_WARMUP=10
PDSCH_LATENCY_RUNS=1
RX_DEVICE_GRID="auto"
RESOURCE_GRID_MEMORY="auto"
DEVICE_GRID_MEMORY="managed"
BUILD_TARGETS=1

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --build-dir DIR             Build directory (default: ${BUILD_DIR})
  --out-dir DIR               Output directory (default: artifacts/type1_dmrs_gpu_cpu_sweeps/<timestamp>)
  --quick                     Short smoke run: fewer frames/iterations and wider SNR step
  --full                      Longer sensitivity run: more frames and finer SNR step
  --prbs CSV                  PRB list (default: ${PRBS}; 30 kHz 20/40/100 MHz)
  --topologies CSV            UL layer/port list as LxP entries (default: ${TOPOLOGIES})
                              DL uses the same layer counts with ports=layers.
  --mcs N                     MCS index for latency and sensitivity (default: ${MCS})
  --mcs-table NAME            qam64, qam256, qam64LowSe (default: ${MCS_TABLE})
  --pusch-snr-start DB        PUSCH sensitivity start SNR (default: ${PUSCH_SNR_START})
  --pusch-snr-stop DB         PUSCH sensitivity stop SNR (default: ${PUSCH_SNR_STOP})
  --pusch-snr-step DB         PUSCH sensitivity SNR step (default: ${PUSCH_SNR_STEP})
  --pusch-frames N            PUSCH sensitivity frames per waterfall point (default: ${PUSCH_FRAMES})
  --pusch-latency-sinr DB     PUSCH latency SINR (default: ${PUSCH_LATENCY_SINR})
  --pusch-latency-iterations N
                              PUSCH latency measured iterations (default: ${PUSCH_LATENCY_ITERATIONS})
  --pdsch-latency-iterations N
                              PDSCH latency measured iterations (default: ${PDSCH_LATENCY_ITERATIONS})
  --rx-device-grid MODE       PUSCH RX device grid mode: off, device, managed, direct-managed, auto
                              (default: ${RX_DEVICE_GRID})
  --resource-grid-memory MODE PDSCH resource grid memory: host, managed, auto
                              (default: ${RESOURCE_GRID_MEMORY})
  --device-grid-memory MODE   PDSCH sidecar device grid memory: device, managed
                              (default: ${DEVICE_GRID_MEMORY})
  --no-build                  Do not build benchmark targets before running
  -h, --help                  Show this help

All runs are Type-1 DMRS, non-transform, double-DMRS symbols 2,11.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --quick)
      PUSCH_FRAMES=30
      PUSCH_SNR_STEP="2.0"
      PUSCH_LATENCY_ITERATIONS=20
      PDSCH_LATENCY_ITERATIONS=20
      shift
      ;;
    --full)
      PUSCH_FRAMES=500
      PUSCH_SNR_STEP="0.5"
      PUSCH_LATENCY_ITERATIONS=200
      PDSCH_LATENCY_ITERATIONS=200
      PDSCH_LATENCY_RUNS=3
      shift
      ;;
    --prbs) PRBS="$2"; shift 2 ;;
    --topologies) TOPOLOGIES="$2"; shift 2 ;;
    --mcs) MCS="$2"; shift 2 ;;
    --mcs-table) MCS_TABLE="$2"; shift 2 ;;
    --pusch-snr-start) PUSCH_SNR_START="$2"; shift 2 ;;
    --pusch-snr-stop) PUSCH_SNR_STOP="$2"; shift 2 ;;
    --pusch-snr-step) PUSCH_SNR_STEP="$2"; shift 2 ;;
    --pusch-frames) PUSCH_FRAMES="$2"; shift 2 ;;
    --pusch-latency-sinr) PUSCH_LATENCY_SINR="$2"; shift 2 ;;
    --pusch-latency-iterations) PUSCH_LATENCY_ITERATIONS="$2"; shift 2 ;;
    --pdsch-latency-iterations) PDSCH_LATENCY_ITERATIONS="$2"; shift 2 ;;
    --rx-device-grid) RX_DEVICE_GRID="$2"; shift 2 ;;
    --resource-grid-memory) RESOURCE_GRID_MEMORY="$2"; shift 2 ;;
    --device-grid-memory) DEVICE_GRID_MEMORY="$2"; shift 2 ;;
    --no-build) BUILD_TARGETS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${REPO_ROOT}/artifacts/type1_dmrs_gpu_cpu_sweeps/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUT_DIR}"

RUN_LOG="${OUT_DIR}/run.log"
if [[ -z "${TYPE1_DMRS_SWEEP_LOG_ACTIVE:-}" ]]; then
  export TYPE1_DMRS_SWEEP_LOG_ACTIVE=1
  exec > >(tee "${RUN_LOG}") 2>&1
fi

PUSCH_SENS_BIN="${BUILD_DIR}/tests/integrationtests/phy/upper/channel_processors/pusch_e2e_sensitivity_sweep"
PUSCH_LAT_BIN="${BUILD_DIR}/tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
PDSCH_LAT_BIN="${BUILD_DIR}/tests/integrationtests/phy/upper/channel_processors/pdsch_gpu_latency_benchmark"

if [[ "${BUILD_TARGETS}" -ne 0 ]]; then
  cmake --build "${BUILD_DIR}" --target \
    pusch_e2e_sensitivity_sweep \
    pusch_e2e_pipeline_test \
    pdsch_gpu_latency_benchmark \
    -j"$(nproc)"
fi

for bin in "${PUSCH_SENS_BIN}" "${PUSCH_LAT_BIN}" "${PDSCH_LAT_BIN}"; do
  if [[ ! -x "${bin}" ]]; then
    echo "Required benchmark not found or not executable: ${bin}" >&2
    exit 1
  fi
done

IFS=',' read -r -a PRB_LIST <<< "${PRBS}"
IFS=',' read -r -a TOPOLOGY_LIST <<< "${TOPOLOGIES}"

mcs_table_id() {
  case "$1" in
    qam64) echo 1 ;;
    qam256) echo 2 ;;
    qam64LowSe|qam64LowSE|qam64lowse) echo 3 ;;
    *) echo "$1" ;;
  esac
}

bw_label() {
  case "$1" in
    51) echo "20MHz" ;;
    106) echo "40MHz" ;;
    273) echo "100MHz" ;;
    *) echo "${1}PRB" ;;
  esac
}

parse_topology() {
  local topo="$1"
  topo="$(echo "${topo}" | tr -d '[:space:]')"
  if [[ ! "${topo}" =~ ^([0-9]+)[xX]([0-9]+)$ ]]; then
    echo "Invalid topology '${topo}'. Use LxP, for example 2x4." >&2
    exit 1
  fi
  echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[1]}L${BASH_REMATCH[2]}P"
}

MANIFEST="${OUT_DIR}/manifest.txt"
cat > "${MANIFEST}" <<EOF
type1_dmrs_ul_dl_gpu_cpu_sweeps
date=$(date --iso-8601=seconds)
repo=${REPO_ROOT}
build_dir=${BUILD_DIR}
run_log=${RUN_LOG}
prbs=${PRBS}
topologies=${TOPOLOGIES}
dl_ports_policy=ports_equal_layers
mcs=${MCS}
mcs_table=${MCS_TABLE}
dmrs_type=type1
dmrs_symbols=2,11
transform_precoding=off
pusch_sensitivity_frames=${PUSCH_FRAMES}
pusch_sensitivity_snr=${PUSCH_SNR_START}:${PUSCH_SNR_STEP}:${PUSCH_SNR_STOP}
pusch_latency_sinr=${PUSCH_LATENCY_SINR}
pusch_latency_iterations=${PUSCH_LATENCY_ITERATIONS}
pdsch_latency_iterations=${PDSCH_LATENCY_ITERATIONS}
rx_device_grid=${RX_DEVICE_GRID}
resource_grid_memory=${RESOURCE_GRID_MEMORY}
device_grid_memory=${DEVICE_GRID_MEMORY}
EOF

run_and_log() {
  local name="$1"
  shift
  local log="${OUT_DIR}/${name}.log"

  echo
  echo "RUN ${name}"
  printf '  CMD'
  printf ' %q' "$@"
  echo
  echo "  RAW_LOG ${log}"

  if ! {
    printf 'CMD'
    printf ' %q' "$@"
    echo
    "$@"
  } > "${log}" 2>&1; then
    echo "  FAILED. Last 80 log lines:" >&2
    tail -n 80 "${log}" >&2
    exit 1
  fi
}

parse_pusch_sensitivity() {
  local log="$1"
  sed -n 's/.*CPU=\([^d]*\)dB  GPU=\([^d]*\)dB  delta=\([^d]*\)dB.*/\1\t\2\t\3/p' "${log}" | tail -n 1
}

parse_pusch_latency() {
  local log="$1"
  awk '
    /^CPU:/ {cpu_pass=$2}
    /^GPU:/ {gpu_pass=$2}
    /^Byte mismatches:/ {mismatch=$3}
    /^CPU total:/ {cpu_mean=$4}
    /^GPU total:/ {gpu_mean=$4}
    /^Speedup:/ {speed=$2; gsub(/x/, "", speed)}
    /^SINR \(dB\):/ {sinr_delta=$5}
    END {printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", cpu_pass, gpu_pass, mismatch, cpu_mean, gpu_mean, speed, sinr_delta}
  ' "${log}"
}

parse_pdsch_latency() {
  local log="$1"
  awk '
    /^mcs_table=/ {
      for (i = 1; i <= NF; ++i) {
        split($i, kv, "=")
        if (kv[1] == "tbs_bits") tbs = kv[2]
        if (kv[1] == "avg_us") avg = kv[2]
        if (kv[1] == "p50_us") p50 = kv[2]
        if (kv[1] == "p90_us") p90 = kv[2]
        if (kv[1] == "p99_us") p99 = kv[2]
        if (kv[1] == "resource_grid_path") path = kv[2]
      }
    }
    END {printf "%s\t%s\t%s\t%s\t%s\t%s\n", tbs, avg, p50, p90, p99, path}
  ' "${log}"
}

summary_file="${OUT_DIR}/summary.tsv"
sens_results="${OUT_DIR}/pusch_sensitivity_results.tsv"
pusch_lat_results="${OUT_DIR}/pusch_latency_results.tsv"
pdsch_lat_results="${OUT_DIR}/pdsch_latency_results.tsv"

printf "direction\ttest\tprb\tbw\ttopology\tlayers\tports\tbackend\tlog\n" > "${summary_file}"
printf "bw\tprb\ttopology\tlayers\tul_ports\tcpu_10pct_db\tgpu_10pct_db\tdelta_db\tlog\n" > "${sens_results}"
printf "bw\tprb\ttopology\tlayers\tul_ports\tcpu_mean_us\tgpu_mean_us\tspeedup\tcpu_pass\tgpu_pass\tmismatches\tsinr_delta_db\tlog\n" > "${pusch_lat_results}"
printf "bw\tprb\ttopology\tlayers\tdl_ports\tbackend\tavg_us\tp50_us\tp90_us\tp99_us\ttbs_bits\tgrid_path\tlog\n" > "${pdsch_lat_results}"

echo
echo "Writing logs to ${OUT_DIR}"
echo "Combined script log: ${RUN_LOG}"
echo "Manifest: ${MANIFEST}"

for topo in "${TOPOLOGY_LIST[@]}"; do
  read -r layers ul_ports topo_label <<< "$(parse_topology "${topo}")"
  dl_ports="${layers}"
  dl_topo_label="${layers}L${dl_ports}P"

  for prb in "${PRB_LIST[@]}"; do
    prb="$(echo "${prb}" | tr -d '[:space:]')"
    [[ -n "${prb}" ]] || continue
    bw="$(bw_label "${prb}")"
    log_suffix="prb${prb}_${topo_label}"

    run_and_log "pusch_sensitivity_${log_suffix}" \
      "${PUSCH_SENS_BIN}" \
        --nof_prb "${prb}" \
        --mcs_index "${MCS}" \
        --mcs_table "$(mcs_table_id "${MCS_TABLE}")" \
        --snr_start "${PUSCH_SNR_START}" \
        --snr_stop "${PUSCH_SNR_STOP}" \
        --snr_step "${PUSCH_SNR_STEP}" \
        --nof_frames "${PUSCH_FRAMES}" \
        --gpu-type gpu \
        --layers "${layers}" \
        --ports "${ul_ports}" \
        --dmrs-type type1 \
        --dmrs-symbols 2,11 \
        --equalizer mmse
    sens_log="${OUT_DIR}/pusch_sensitivity_${log_suffix}.log"
    read -r cpu_10 gpu_10 delta_10 <<< "$(parse_pusch_sensitivity "${sens_log}")"
    printf "UL\tsensitivity\t%s\t%s\t%s\t%s\t%s\tcpu_vs_gpu\t%s\n" \
      "${prb}" "${bw}" "${topo_label}" "${layers}" "${ul_ports}" "${sens_log}" >> "${summary_file}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${bw}" "${prb}" "${topo_label}" "${layers}" "${ul_ports}" "${cpu_10}" "${gpu_10}" "${delta_10}" "${sens_log}" >> "${sens_results}"

    run_and_log "pusch_latency_${log_suffix}" \
      "${PUSCH_LAT_BIN}" \
        --prb "${prb}" \
        --sinr "${PUSCH_LATENCY_SINR}" \
        --mcs "${MCS}" \
        --mcs-table "${MCS_TABLE}" \
        --iterations "${PUSCH_LATENCY_ITERATIONS}" \
        --warmup "${PUSCH_LATENCY_WARMUP}" \
        --layers "${layers}" \
        --ports "${ul_ports}" \
        --dmrs-type type1 \
        --dmrs-symbols 2,11 \
        --rx-device-grid "${RX_DEVICE_GRID}" \
        --quiet
    pusch_lat_log="${OUT_DIR}/pusch_latency_${log_suffix}.log"
    read -r cpu_pass gpu_pass mismatches cpu_mean gpu_mean speedup sinr_delta <<< "$(parse_pusch_latency "${pusch_lat_log}")"
    printf "UL\tlatency\t%s\t%s\t%s\t%s\t%s\tcpu_vs_gpu\t%s\n" \
      "${prb}" "${bw}" "${topo_label}" "${layers}" "${ul_ports}" "${pusch_lat_log}" >> "${summary_file}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${bw}" "${prb}" "${topo_label}" "${layers}" "${ul_ports}" "${cpu_mean}" "${gpu_mean}" "${speedup}" \
      "${cpu_pass}" "${gpu_pass}" "${mismatches}" "${sinr_delta}" "${pusch_lat_log}" >> "${pusch_lat_results}"

    for backend in cpu gpu; do
      pdsch_log_name="pdsch_latency_${backend}_prb${prb}_${dl_topo_label}"
      pdsch_args=(
        "${PDSCH_LAT_BIN}"
        "--backend=${backend}"
        "--prb=${prb}"
        "--mcs=${MCS}"
        "--mcs-table=${MCS_TABLE}"
        "--layers=${layers}"
        "--ports=${dl_ports}"
        "--dmrs=double"
        "--dmrs-type=type1"
        "--cdm=2"
        "--iterations=${PDSCH_LATENCY_ITERATIONS}"
        "--warmup=${PDSCH_LATENCY_WARMUP}"
        "--runs=${PDSCH_LATENCY_RUNS}"
        "--sync-after-process=1"
      )
      if [[ "${backend}" == "gpu" ]]; then
        pdsch_args+=(
          "--device-grid=1"
          "--device-grid-memory=${DEVICE_GRID_MEMORY}"
          "--resource-grid-memory=${RESOURCE_GRID_MEMORY}"
        )
      fi

      run_and_log "${pdsch_log_name}" "${pdsch_args[@]}"
      pdsch_log="${OUT_DIR}/${pdsch_log_name}.log"
      read -r tbs_bits avg_us p50_us p90_us p99_us grid_path <<< "$(parse_pdsch_latency "${pdsch_log}")"
      printf "DL\tlatency\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${prb}" "${bw}" "${dl_topo_label}" "${layers}" "${dl_ports}" "${backend}" "${pdsch_log}" >> "${summary_file}"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${bw}" "${prb}" "${dl_topo_label}" "${layers}" "${dl_ports}" "${backend}" "${avg_us}" "${p50_us}" \
        "${p90_us}" "${p99_us}" "${tbs_bits}" "${grid_path}" "${pdsch_log}" >> "${pdsch_lat_results}"
    done
  done
done

echo
echo "================================================================"
echo "PUSCH sensitivity summary"
echo "================================================================"
printf "%-7s %-5s %-7s %5s %8s %10s %10s %9s\n" "BW" "PRB" "ULTopo" "L" "ULPorts" "CPU10dB" "GPU10dB" "Delta"
tail -n +2 "${sens_results}" | while IFS=$'\t' read -r bw prb topo layers ul_ports cpu_10 gpu_10 delta_10 log; do
  printf "%-7s %-5s %-7s %5s %8s %10s %10s %9s\n" "${bw}" "${prb}" "${topo}" "${layers}" "${ul_ports}" "${cpu_10}" "${gpu_10}" "${delta_10}"
done

echo
echo "================================================================"
echo "PUSCH latency summary"
echo "================================================================"
printf "%-7s %-5s %-7s %5s %8s %10s %10s %8s %10s %10s %10s\n" \
  "BW" "PRB" "ULTopo" "L" "ULPorts" "CPU(us)" "GPU(us)" "Speedup" "CPU" "GPU" "Mismatches"
tail -n +2 "${pusch_lat_results}" | while IFS=$'\t' read -r bw prb topo layers ul_ports cpu_mean gpu_mean speedup cpu_pass gpu_pass mismatches sinr_delta log; do
  printf "%-7s %-5s %-7s %5s %8s %10s %10s %8sx %10s %10s %10s\n" \
    "${bw}" "${prb}" "${topo}" "${layers}" "${ul_ports}" "${cpu_mean}" "${gpu_mean}" "${speedup}" "${cpu_pass}" "${gpu_pass}" "${mismatches}"
done

echo
echo "================================================================"
echo "PDSCH latency summary"
echo "================================================================"
printf "%-7s %-5s %-7s %5s %8s %-7s %10s %10s %10s %10s %10s\n" \
  "BW" "PRB" "DLTopo" "L" "DLPorts" "Backend" "Avg(us)" "P50(us)" "P90(us)" "P99(us)" "Grid"
tail -n +2 "${pdsch_lat_results}" | while IFS=$'\t' read -r bw prb topo layers dl_ports backend avg_us p50_us p90_us p99_us tbs_bits grid_path log; do
  printf "%-7s %-5s %-7s %5s %8s %-7s %10s %10s %10s %10s %10s\n" \
    "${bw}" "${prb}" "${topo}" "${layers}" "${dl_ports}" "${backend}" "${avg_us}" "${p50_us}" "${p90_us}" "${p99_us}" "${grid_path}"
done

echo
echo "================================================================"
echo "Completed Type-1 DMRS non-transform GPU-vs-CPU sweep."
echo "Output directory: ${OUT_DIR}"
echo "Combined script log: ${RUN_LOG}"
echo "Raw benchmark logs: ${OUT_DIR}/*.log"
echo "Summary index: ${summary_file}"
echo "PUSCH sensitivity TSV: ${sens_results}"
echo "PUSCH latency TSV: ${pusch_lat_results}"
echo "PDSCH latency TSV: ${pdsch_lat_results}"
echo "================================================================"
