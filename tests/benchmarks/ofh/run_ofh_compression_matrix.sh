#!/usr/bin/env bash
#
# Runs the Open Fronthaul IQ compression benchmark matrix and reports CPU/GPU
# TX/RX p50 latency and speedups.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

build_dir="${repo_root}/build-cuda-bench-v2"
benchmark_bin=""
repetitions=500
symbols=14
types="bfp none"
bandwidths="5 10 20 100"
ports="1 2 4"
cpu_impl="auto"
cpu_grid="host"
gpu_impl="cuda"
gpu_grid="device-device-symbol-batch"
output_dir=""

usage() {
  cat <<EOF
Usage: $0 [options]

Runs CPU vs GPU Open Fronthaul IQ TX/RX compression benchmarks.
TX is compression, RX is decompression. The benchmark sweeps all bit widths
supported by ofh_compression_benchmark: 8, 9, 10, 12, 14, and 16.

Options:
  --build-dir DIR       Build directory containing tests/benchmarks/ofh/ofh_compression_benchmark.
                        Default: ${build_dir}
  --benchmark-bin PATH  Benchmark binary path. Overrides --build-dir.
  --repetitions N      Repetitions per measurement. Default: ${repetitions}
  --symbols N          Consecutive OFDM symbols per run. Default: ${symbols}
  --types LIST         Compression types, space-separated. Default: "${types}"
  --bandwidths LIST    Bandwidths in MHz, space-separated. Default: "${bandwidths}"
  --ports LIST         Port counts, space-separated. Default: "${ports}"
  --gpu-grid MODE      GPU grid mode. Default: ${gpu_grid}
  --output-dir DIR     Directory for CSV results. Default: BUILD_DIR/benchmark-results
  -h, --help           Show this help.

Examples:
  $0 --repetitions 1000
  $0 --bandwidths "20 100" --ports "1 4" --types "bfp"
  $0 --build-dir build-cuda-bench-v2 --output-dir /tmp/ofh-bench
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      build_dir="$2"
      shift 2
      ;;
    --benchmark-bin)
      benchmark_bin="$2"
      shift 2
      ;;
    --repetitions)
      repetitions="$2"
      shift 2
      ;;
    --symbols)
      symbols="$2"
      shift 2
      ;;
    --types)
      types="$2"
      shift 2
      ;;
    --bandwidths)
      bandwidths="$2"
      shift 2
      ;;
    --ports)
      ports="$2"
      shift 2
      ;;
    --gpu-grid)
      gpu_grid="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${benchmark_bin}" ]]; then
  benchmark_bin="${build_dir}/tests/benchmarks/ofh/ofh_compression_benchmark"
fi
if [[ -z "${output_dir}" ]]; then
  output_dir="${build_dir}/benchmark-results"
fi

if [[ ! -x "${benchmark_bin}" ]]; then
  echo "Benchmark binary not found or not executable: ${benchmark_bin}" >&2
  echo "Build it first, for example: cmake --build ${build_dir} --target ofh_compression_benchmark" >&2
  exit 1
fi

mkdir -p "${output_dir}"
run_id="$(date +%Y%m%d_%H%M%S)"
raw_csv="${output_dir}/ofh_compression_matrix_${run_id}_raw.csv"
speedup_csv="${output_dir}/ofh_compression_matrix_${run_id}_speedups.csv"

echo "type,bw_mhz,ports,engine,op,width_bits,p50_us,repetitions,symbols,impl,grid" > "${raw_csv}"

run_case() {
  local type="$1"
  local bw="$2"
  local port_count="$3"
  local engine="$4"
  local impl="$5"
  local grid="$6"

  printf 'Running type=%s bw=%sMHz ports=%s engine=%s impl=%s grid=%s\n' \
    "${type}" "${bw}" "${port_count}" "${engine}" "${impl}" "${grid}" >&2

  "${benchmark_bin}" \
    -R "${repetitions}" \
    -T "${type}" \
    -F "${impl}" \
    -G "${grid}" \
    -B "${bw}" \
    -N "${port_count}" \
    -Y "${symbols}" |
    awk -v type="${type}" \
        -v bw="${bw}" \
        -v ports="${port_count}" \
        -v engine="${engine}" \
        -v repetitions="${repetitions}" \
        -v symbols="${symbols}" \
        -v impl="${impl}" \
        -v grid="${grid}" '
      /All values are in microseconds/ { in_time = 1; next }
      /All values are in megasamples/ { in_time = 0 }
      in_time && /^[[:space:]]*(BFP|none)-[0-9]+b (compression|decompression)/ {
        split($0, cols, "|");
        name = cols[1];
        p50 = cols[2];
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", name);
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", p50);
        split(name, parts, " ");
        width = parts[1];
        op = parts[2];
        gsub(/[^0-9]/, "", width);
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
               type, bw, ports, engine, op, width, p50, repetitions, symbols, impl, grid;
      }' >> "${raw_csv}"
}

for type in ${types}; do
  for bw in ${bandwidths}; do
    for port_count in ${ports}; do
      run_case "${type}" "${bw}" "${port_count}" CPU "${cpu_impl}" "${cpu_grid}"
      run_case "${type}" "${bw}" "${port_count}" GPU "${gpu_impl}" "${gpu_grid}"
    done
  done
done

tmp_speedups="$(mktemp)"
awk -F, '
  NR == 1 { next }
  {
    key = $1 "," $2 "," $3 "," $5 "," $6;
    if ($4 == "CPU") {
      cpu[key] = $7;
    } else if ($4 == "GPU") {
      gpu[key] = $7;
    }
  }
  END {
    print "type,bw_mhz,ports,path,width_bits,cpu_p50_us,gpu_p50_us,cpu_over_gpu_speedup";
    for (key in cpu) {
      if (!(key in gpu)) {
        continue;
      }
      split(key, parts, ",");
      path = (parts[4] == "compression") ? "tx_compression" : "rx_decompression";
      printf "%s,%s,%s,%s,%s,%.3f,%.3f,%.3f\n",
             parts[1], parts[2], parts[3], path, parts[5], cpu[key], gpu[key], cpu[key] / gpu[key];
    }
  }' "${raw_csv}" > "${tmp_speedups}"

{
  head -n 1 "${tmp_speedups}"
  tail -n +2 "${tmp_speedups}" | sort -t, -k1,1 -k2,2n -k3,3n -k4,4 -k5,5n
} > "${speedup_csv}"
rm -f "${tmp_speedups}"

echo
echo "Raw p50 CSV: ${raw_csv}"
echo "Speedup CSV: ${speedup_csv}"
echo
echo "CPU/GPU p50 speedups (>1 means GPU faster):"
awk -F, '
  NR == 1 {
    printf "%-5s %-6s %-5s %-18s %-5s %10s %10s %9s\n",
           "type", "bw", "port", "path", "bits", "CPU us", "GPU us", "speedup";
    next;
  }
  {
    printf "%-5s %-6s %-5s %-18s %-5s %10.1f %10.1f %9.2f\n",
           $1, $2 "M", $3, $4, $5, $6, $7, $8;
  }' "${speedup_csv}"
