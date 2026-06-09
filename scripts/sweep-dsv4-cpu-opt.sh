#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

build=${DSV4_SWEEP_BUILD:-"$root/build-cpu-opt"}
model=${DSV4_MODEL:-"/home/mheiss/.cache/huggingface/hub/models--teamblobfish--DeepSeek-V4-Flash-GGUF/snapshots/b281094221a72c210a2b986709510b4c4b51b67e/Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00001-of-00004.gguf"}
results_dir=${DSV4_SWEEP_RESULTS:-"$root/dsv4-cascade-lake-results/numa-sweep"}
ctx="${1:-${DSV4_CTX_SIZE:-1024}}"
n_predict="${2:-${DSV4_N_PREDICT:-192}}"
filler_lines="${3:-${DSV4_FILLER_LINES:-0}}"
flash_modes="${DSV4_SWEEP_FLASH_MODES:-on}"

baseline_threads=${DSV4_BASELINE_THREADS:-32}
baseline_threads_batch=${DSV4_BASELINE_THREADS_BATCH:-52}
baseline_numa=${DSV4_BASELINE_NUMA:-distribute}

mkdir -p "$results_dir"

usage() {
    cat <<EOF
usage: $0 [ctx] [n_predict] [filler_lines]

Run a same-binary DeepSeek4 CPU scheduling sweep against the canonical
build-cpu-opt baseline policy. Flash attention is the default gate.

Environment:
  DSV4_SWEEP_BUILD           build dir containing bin/llama-server, default build-cpu-opt
  DSV4_MODEL                 GGUF model path
  DSV4_SWEEP_RESULTS         result dir, default dsv4-cascade-lake-results/numa-sweep
  DSV4_SWEEP_FLASH_MODES     comma-separated on/off list, default on
  DSV4_BASELINE_THREADS      baseline generation threads, default 32
  DSV4_BASELINE_THREADS_BATCH baseline prompt threads, default 52
  DSV4_BASELINE_NUMA         baseline --numa policy, default distribute
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -x "$build/bin/llama-server" ]]; then
    echo "missing llama-server at $build/bin/llama-server" >&2
    exit 1
fi

run_case() {
    local name="$1"
    local flash="$2"
    local mode_args=()
    shift 2

    case "$flash" in
        on)  mode_args=(--flash-attn) ;;
        off) mode_args=() ;;
        *)
            echo "invalid flash mode: $flash" >&2
            exit 2
            ;;
    esac

    local log="$results_dir/${name}-fa-${flash}.log"
    local summary="$results_dir/${name}-fa-${flash}.summary"

    echo
    echo "===== ${name} fa-${flash} ====="
    python3 "$root/scripts/engine_test_harness.py" \
        "${mode_args[@]}" \
        --model "$model" \
        --server-bin "$build/bin/llama-server" \
        --ik-server-bin "$build/bin/llama-server" \
        --ctx-size "$ctx" \
        --n-predict "$n_predict" \
        --filler-lines "$filler_lines" \
        --threads "$baseline_threads" \
        --threads-batch "$baseline_threads_batch" \
        --numa "$baseline_numa" \
        "$@" \
        2>&1 | tee "$log"

    rg "comparison_status|max_abs_logprob_diff|mean_abs_logprob_diff|baseline_ik perf|opt_ik perf|root:" "$log" \
        >"$summary" || true
}

run_all_modes() {
    local name="$1"
    shift
    IFS=',' read -r -a modes <<<"$flash_modes"
    for mode in "${modes[@]}"; do
        run_case "$name" "$mode" "$@"
    done
}

run_all_modes "control_distribute_t32_tb52" \
    --ik-numa distribute \
    --ik-threads 32 \
    --ik-threads-batch 52

run_all_modes "distribute_t52_tb52" \
    --ik-numa distribute \
    --ik-threads 52 \
    --ik-threads-batch 52

run_all_modes "distribute_t104_tb104" \
    --ik-numa distribute \
    --ik-threads 104 \
    --ik-threads-batch 104

run_all_modes "isolate_t52_tb52" \
    --ik-numa isolate \
    --ik-threads 52 \
    --ik-threads-batch 52

if command -v numactl >/dev/null 2>&1; then
    run_all_modes "numactl_interleave_t52_tb52" \
        --ik-server-prefix "numactl --interleave=all" \
        --ik-numa numactl \
        --ik-threads 52 \
        --ik-threads-batch 52

    run_all_modes "numactl_phys_all_t52_tb52" \
        --ik-server-prefix "numactl --physcpubind=0-51 --interleave=all" \
        --ik-numa numactl \
        --ik-threads 52 \
        --ik-threads-batch 52

    run_all_modes "numactl_node0_phys_t26_tb26" \
        --ik-server-prefix "numactl --physcpubind=0-25 --membind=0" \
        --ik-numa numactl \
        --ik-threads 26 \
        --ik-threads-batch 26

    run_all_modes "numactl_node0_smt_t52_tb52" \
        --ik-server-prefix "numactl --cpunodebind=0 --membind=0" \
        --ik-numa numactl \
        --ik-threads 52 \
        --ik-threads-batch 52
else
    echo "numactl not found; skipping external NUMA placement cases"
fi

echo
echo "summaries:"
for summary in "$results_dir"/*.summary; do
    [[ -e "$summary" ]] || continue
    echo "----- $summary -----"
    cat "$summary"
done
