#!/usr/bin/env bash
# Concurrent GPU stream/load smoke test for CUDA UL/DL/SRS paths.
#
# This is not an OTA slot-clock replacement. It is a repeatable stress gate that
# runs the existing correctness/thread-safety binaries concurrently so CUDA stream
# ordering, event bookkeeping, and CUDA-visible grid paths get exercised under
# sustained GPU load.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUILD_DIR="${REPO_ROOT}/build-cuda-default-check"
OUT_DIR=""
ROUNDS=3
BUILD_TARGETS=1

PUSCH_JOBS=2
PUSCH_PRB=273
PUSCH_PORTS=4
PUSCH_LAYERS=2
PUSCH_ITERATIONS=50
PUSCH_WARMUP=2
PUSCH_SINR=20
PUSCH_MCS=20
PUSCH_RX_GRID="direct-managed"

PDSCH_JOBS=1
PDSCH_PRB=273
PDSCH_PORTS=2
PDSCH_LAYERS=2
PDSCH_ITERATIONS=100
PDSCH_WARMUP=10
PDSCH_MCS=20
PDSCH_MCS_TABLE="qam64"
PDSCH_RESOURCE_GRID="host"
PDSCH_DEVICE_GRID_MEMORY="managed"
PDSCH_CORRECTNESS_JOBS=1
PDSCH_CORRECTNESS_REPEAT=1
PDSCH_DIRECT_VISIBLE=0
PDSCH_GRID_MODE="managed"

SRS_JOBS=2
SRS_THREADS=4
SRS_ITERATIONS=64
SRS_SNR=30
SRS_GRID="visible"
SRS_PROFILES="baseline_4x4_n4,n4_4x2_shifted,n2_4x4_120khz"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --build-dir DIR             Build directory (default: ${BUILD_DIR})
  --out-dir DIR               Output directory (default: artifacts/gpu_stream_load_test/<timestamp>)
  --quick                     Short smoke run
  --full                      Longer load run
  --rounds N                  Repetitions per worker (default: ${ROUNDS})
  --no-build                  Do not build targets before running

  --pusch-jobs N              Concurrent PUSCH workers (default: ${PUSCH_JOBS})
  --pusch-prb N               PUSCH PRBs (default: ${PUSCH_PRB})
  --pusch-ports N             PUSCH RX ports (default: ${PUSCH_PORTS})
  --pusch-layers N            PUSCH TX layers (default: ${PUSCH_LAYERS})
  --pusch-iterations N        PUSCH measured iterations per worker round (default: ${PUSCH_ITERATIONS})
  --pusch-rx-grid MODE        off, device, managed, direct-managed, auto (default: ${PUSCH_RX_GRID})

  --pdsch-jobs N              Concurrent high-rate PDSCH load workers (default: ${PDSCH_JOBS})
  --pdsch-prb N               PDSCH PRBs (default: ${PDSCH_PRB})
  --pdsch-ports N             PDSCH ports (default: ${PDSCH_PORTS})
  --pdsch-layers N            PDSCH layers (default: ${PDSCH_LAYERS})
  --pdsch-iterations N        PDSCH load iterations per worker round (default: ${PDSCH_ITERATIONS})
  --pdsch-resource-grid MODE  host, managed, auto for PDSCH load worker (default: ${PDSCH_RESOURCE_GRID})
  --pdsch-device-grid-memory MODE
                              device or managed sidecar memory (default: ${PDSCH_DEVICE_GRID_MEMORY})
  --pdsch-correctness-jobs N  Concurrent PDSCH CPU/GPU correctness workers (default: ${PDSCH_CORRECTNESS_JOBS})
  --pdsch-correctness-repeat N
                              gtest repeat count per correctness worker round (default: ${PDSCH_CORRECTNESS_REPEAT})
  --pdsch-direct-visible      Opt into direct CUDA-visible PDSCH writer
  --pdsch-grid MODE           OCUDU_DL_CUDA_VISIBLE_GRID mode (default: ${PDSCH_GRID_MODE})

  --srs-jobs N                Concurrent SRS workers (default: ${SRS_JOBS})
  --srs-threads N             SRS threads inside each worker (default: ${SRS_THREADS})
  --srs-iterations N          SRS iterations per SRS thread (default: ${SRS_ITERATIONS})
  --srs-profiles CSV          SRS profile rotation (default: ${SRS_PROFILES})

Examples:
  $0 --quick --build-dir build-cuda-default-check
  $0 --full --pusch-jobs 4 --pdsch-jobs 2 --srs-jobs 4
  $0 --quick --pdsch-direct-visible
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --quick)
      ROUNDS=1
      PUSCH_JOBS=1
      PUSCH_PRB=106
      PUSCH_ITERATIONS=8
      PDSCH_JOBS=1
      PDSCH_PRB=106
      PDSCH_ITERATIONS=10
      PDSCH_WARMUP=2
      PDSCH_CORRECTNESS_JOBS=1
      PDSCH_CORRECTNESS_REPEAT=1
      SRS_JOBS=1
      SRS_THREADS=4
      SRS_ITERATIONS=32
      shift
      ;;
    --full)
      ROUNDS=10
      PUSCH_JOBS=4
      PUSCH_PRB=273
      PUSCH_ITERATIONS=50
      PDSCH_JOBS=2
      PDSCH_PRB=273
      PDSCH_ITERATIONS=100
      PDSCH_CORRECTNESS_JOBS=1
      PDSCH_CORRECTNESS_REPEAT=2
      SRS_JOBS=4
      SRS_THREADS=4
      SRS_ITERATIONS=128
      shift
      ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --no-build) BUILD_TARGETS=0; shift ;;
    --pusch-jobs) PUSCH_JOBS="$2"; shift 2 ;;
    --pusch-prb) PUSCH_PRB="$2"; shift 2 ;;
    --pusch-ports) PUSCH_PORTS="$2"; shift 2 ;;
    --pusch-layers) PUSCH_LAYERS="$2"; shift 2 ;;
    --pusch-iterations) PUSCH_ITERATIONS="$2"; shift 2 ;;
    --pusch-rx-grid) PUSCH_RX_GRID="$2"; shift 2 ;;
    --pdsch-jobs) PDSCH_JOBS="$2"; shift 2 ;;
    --pdsch-prb) PDSCH_PRB="$2"; shift 2 ;;
    --pdsch-ports) PDSCH_PORTS="$2"; shift 2 ;;
    --pdsch-layers) PDSCH_LAYERS="$2"; shift 2 ;;
    --pdsch-iterations) PDSCH_ITERATIONS="$2"; shift 2 ;;
    --pdsch-resource-grid) PDSCH_RESOURCE_GRID="$2"; shift 2 ;;
    --pdsch-device-grid-memory) PDSCH_DEVICE_GRID_MEMORY="$2"; shift 2 ;;
    --pdsch-correctness-jobs) PDSCH_CORRECTNESS_JOBS="$2"; shift 2 ;;
    --pdsch-correctness-repeat) PDSCH_CORRECTNESS_REPEAT="$2"; shift 2 ;;
    --pdsch-direct-visible) PDSCH_DIRECT_VISIBLE=1; shift ;;
    --pdsch-grid) PDSCH_GRID_MODE="$2"; shift 2 ;;
    --srs-jobs) SRS_JOBS="$2"; shift 2 ;;
    --srs-threads) SRS_THREADS="$2"; shift 2 ;;
    --srs-iterations) SRS_ITERATIONS="$2"; shift 2 ;;
    --srs-profiles) SRS_PROFILES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${REPO_ROOT}/artifacts/gpu_stream_load_test/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUT_DIR}"

RUN_LOG="${OUT_DIR}/run.log"
exec > >(tee "${RUN_LOG}") 2>&1

PUSCH_BIN="${BUILD_DIR}/tests/integrationtests/phy/upper/channel_processors/pusch_e2e_pipeline_test"
PDSCH_LOAD_BIN="${BUILD_DIR}/tests/integrationtests/phy/upper/channel_processors/pdsch_gpu_latency_benchmark"
PDSCH_CORRECTNESS_BIN="${BUILD_DIR}/tests/integrationtests/phy/upper/channel_processors/pdsch_gpu_e2e_test"
SRS_BIN="${BUILD_DIR}/tests/benchmarks/phy/upper/signal_processors/srs_estimator_gpu_thread_safety_test"

if [[ "${BUILD_TARGETS}" -ne 0 ]]; then
  cmake --build "${BUILD_DIR}" --target \
    pusch_e2e_pipeline_test \
    pdsch_gpu_latency_benchmark \
    pdsch_gpu_e2e_test \
    srs_estimator_gpu_thread_safety_test \
    -j"$(nproc)"
fi

for bin in "${PUSCH_BIN}" "${PDSCH_LOAD_BIN}" "${PDSCH_CORRECTNESS_BIN}" "${SRS_BIN}"; do
  if [[ ! -x "${bin}" ]]; then
    echo "Required binary not found or not executable: ${bin}" >&2
    exit 1
  fi
done

IFS=',' read -r -a SRS_PROFILE_LIST <<< "${SRS_PROFILES}"
if [[ "${#SRS_PROFILE_LIST[@]}" -eq 0 ]]; then
  echo "No SRS profiles configured." >&2
  exit 1
fi

echo "================================================================"
echo "  CUDA GPU Stream Load Test"
echo "  Build dir: ${BUILD_DIR}"
echo "  Output:    ${OUT_DIR}"
echo "  Rounds:    ${ROUNDS}"
echo "  PUSCH:     jobs=${PUSCH_JOBS} prb=${PUSCH_PRB} ports=${PUSCH_PORTS} layers=${PUSCH_LAYERS} grid=${PUSCH_RX_GRID}"
echo "  PDSCH:     load_jobs=${PDSCH_JOBS} prb=${PDSCH_PRB} ports=${PDSCH_PORTS} layers=${PDSCH_LAYERS}"
echo "             correctness_jobs=${PDSCH_CORRECTNESS_JOBS} correctness_repeat=${PDSCH_CORRECTNESS_REPEAT}"
echo "             resource_grid=${PDSCH_RESOURCE_GRID} device_grid_memory=${PDSCH_DEVICE_GRID_MEMORY}"
echo "             dl_grid_env=${PDSCH_GRID_MODE} direct_visible=${PDSCH_DIRECT_VISIBLE}"
echo "  SRS:       jobs=${SRS_JOBS} threads=${SRS_THREADS} iterations=${SRS_ITERATIONS} profiles=${SRS_PROFILES}"
echo "================================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
    --format=csv > "${OUT_DIR}/gpu_before.csv" || true
fi

pids=()
names=()
logs=()

run_worker() {
  local name="$1"
  local log_file="$2"
  shift 2

  (
    set +e
    for round in $(seq 1 "${ROUNDS}"); do
      echo "[$(date -Is)] ${name} round ${round}/${ROUNDS}"
      echo "command: $*"
      "$@"
      rc=$?
      if [[ "${rc}" -ne 0 ]]; then
        echo "[$(date -Is)] ${name} failed in round ${round} with rc=${rc}"
        exit "${rc}"
      fi
    done
    echo "[$(date -Is)] ${name} completed"
  ) > "${log_file}" 2>&1 &

  pids+=("$!")
  names+=("${name}")
  logs+=("${log_file}")
}

for idx in $(seq 1 "${PUSCH_JOBS}"); do
  run_worker "pusch_${idx}" "${OUT_DIR}/pusch_${idx}.log" \
    env OCUDU_UL_CUDA_VISIBLE_GRID=managed \
    "${PUSCH_BIN}" \
      --rx-device-grid "${PUSCH_RX_GRID}" \
      --prb "${PUSCH_PRB}" \
      --ports "${PUSCH_PORTS}" \
      --layers "${PUSCH_LAYERS}" \
      --sinr "${PUSCH_SINR}" \
      --mcs "${PUSCH_MCS}" \
      --iterations "${PUSCH_ITERATIONS}" \
      --warmup "${PUSCH_WARMUP}" \
      --quiet
done

for idx in $(seq 1 "${PDSCH_JOBS}"); do
  pdsch_precoding="identity"
  if [[ "${PDSCH_LAYERS}" -eq 1 ]]; then
    pdsch_precoding="all-ports"
  fi
  if [[ "${PDSCH_DIRECT_VISIBLE}" -eq 1 ]]; then
    run_worker "pdsch_load_${idx}" "${OUT_DIR}/pdsch_load_${idx}.log" \
      env OCUDU_DL_CUDA_VISIBLE_GRID="${PDSCH_GRID_MODE}" OCUDU_PDSCH_DIRECT_DEVICE_GRID=1 \
      "${PDSCH_LOAD_BIN}" \
        --backend=gpu \
        --prb="${PDSCH_PRB}" \
        --ports="${PDSCH_PORTS}" \
        --layers="${PDSCH_LAYERS}" \
        --mcs="${PDSCH_MCS}" \
        --mcs-table="${PDSCH_MCS_TABLE}" \
        --dmrs=double \
        --dmrs-type=type1 \
        --cdm=2 \
        --precoding="${pdsch_precoding}" \
        --cb-batch=all \
        --device-grid=1 \
        --device-grid-memory="${PDSCH_DEVICE_GRID_MEMORY}" \
        --resource-grid-memory="${PDSCH_RESOURCE_GRID}" \
        --sync-after-process=1 \
        --iterations="${PDSCH_ITERATIONS}" \
        --warmup="${PDSCH_WARMUP}" \
        --runs=1
  else
    run_worker "pdsch_load_${idx}" "${OUT_DIR}/pdsch_load_${idx}.log" \
      env OCUDU_DL_CUDA_VISIBLE_GRID="${PDSCH_GRID_MODE}" \
      "${PDSCH_LOAD_BIN}" \
        --backend=gpu \
        --prb="${PDSCH_PRB}" \
        --ports="${PDSCH_PORTS}" \
        --layers="${PDSCH_LAYERS}" \
        --mcs="${PDSCH_MCS}" \
        --mcs-table="${PDSCH_MCS_TABLE}" \
        --dmrs=double \
        --dmrs-type=type1 \
        --cdm=2 \
        --precoding="${pdsch_precoding}" \
        --cb-batch=all \
        --device-grid=1 \
        --device-grid-memory="${PDSCH_DEVICE_GRID_MEMORY}" \
        --resource-grid-memory="${PDSCH_RESOURCE_GRID}" \
        --sync-after-process=1 \
        --iterations="${PDSCH_ITERATIONS}" \
        --warmup="${PDSCH_WARMUP}" \
        --runs=1
  fi
done

for idx in $(seq 1 "${PDSCH_CORRECTNESS_JOBS}"); do
  if [[ "${PDSCH_DIRECT_VISIBLE}" -eq 1 ]]; then
    run_worker "pdsch_correctness_${idx}" "${OUT_DIR}/pdsch_correctness_${idx}.log" \
      env OCUDU_DL_CUDA_VISIBLE_GRID="${PDSCH_GRID_MODE}" OCUDU_PDSCH_DIRECT_DEVICE_GRID=1 \
      "${PDSCH_CORRECTNESS_BIN}" --gtest_brief=1 --gtest_repeat="${PDSCH_CORRECTNESS_REPEAT}"
  else
    run_worker "pdsch_correctness_${idx}" "${OUT_DIR}/pdsch_correctness_${idx}.log" \
      env OCUDU_DL_CUDA_VISIBLE_GRID="${PDSCH_GRID_MODE}" \
      "${PDSCH_CORRECTNESS_BIN}" --gtest_brief=1 --gtest_repeat="${PDSCH_CORRECTNESS_REPEAT}"
  fi
done

for idx in $(seq 1 "${SRS_JOBS}"); do
  profile_index=$(( (idx - 1) % ${#SRS_PROFILE_LIST[@]} ))
  profile="${SRS_PROFILE_LIST[${profile_index}]}"
  run_worker "srs_${idx}_${profile}" "${OUT_DIR}/srs_${idx}_${profile}.log" \
    "${SRS_BIN}" \
      -T "${SRS_THREADS}" \
      -I "${SRS_ITERATIONS}" \
      -S "${SRS_SNR}" \
      -G "${SRS_GRID}" \
      -P "${profile}"
done

failures=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "PASS ${names[$idx]}"
  else
    rc=$?
    echo "FAIL ${names[$idx]} rc=${rc} log=${logs[$idx]}"
    failures=$((failures + 1))
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
    --format=csv > "${OUT_DIR}/gpu_after.csv" || true
fi

echo
echo "Summary snippets"
for log in "${logs[@]}"; do
  echo "--- $(basename "${log}") ---"
  grep -E "PASS:|PASSED|FAILED|failures=|CPU:|GPU:|Disagreements:|Byte mismatches:|Speedup:|median_run_|resource_grid_path=|p99_us=" "${log}" | tail -30 || true
done

if [[ "${failures}" -ne 0 ]]; then
  echo
  echo "GPU stream load test failed: ${failures} worker(s) failed. Logs are in ${OUT_DIR}."
  exit 1
fi

echo
echo "GPU stream load test passed. Logs are in ${OUT_DIR}."
