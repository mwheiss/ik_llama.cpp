#!/usr/bin/env bash
set -euo pipefail

ctx="${1:-1024}"
n_predict="${2:-192}"
filler_lines="${3:-0}"
flash_modes="${DSV4_SWEEP_FLASH_MODES:-on,off}"

run_case() {
    local name="$1"
    local flash="$2"
    local mode_args=()
    shift
    shift
    if [[ "$flash" == "on" ]]; then
        mode_args=(--flash-attn)
    elif [[ "$flash" != "off" ]]; then
        echo "invalid flash mode: $flash" >&2
        exit 2
    fi
    echo
    echo "===== ${name} fa-${flash} ====="
    python3 scripts/engine_test_harness.py \
        "${mode_args[@]}" \
        --ctx-size "${ctx}" \
        --n-predict "${n_predict}" \
        --filler-lines "${filler_lines}" \
        "$@" \
        | tee "sweep-dsv4-cpu-opt-${name}-fa-${flash}.log" \
        | rg "comparison_status|max_abs_logprob_diff|mean_abs_logprob_diff|decode_tps=|prefill_tps=|root:"
}

run_all_modes() {
    local name="$1"
    shift
    IFS=',' read -r -a modes <<<"$flash_modes"
    for mode in "${modes[@]}"; do
        run_case "$name" "$mode" "$@"
    done
}

run_all_modes "t24_tb52" --ik-threads 24 --ik-threads-batch 52
run_all_modes "t32_tb52" --ik-threads 32 --ik-threads-batch 52
run_all_modes "t40_tb52" --ik-threads 40 --ik-threads-batch 52
run_all_modes "t52_tb52" --ik-threads 52 --ik-threads-batch 52
run_all_modes "t104_tb104" --ik-threads 104 --ik-threads-batch 104
run_all_modes "ik_numa_distribute_t52_tb52" --ik-numa distribute --ik-threads 52 --ik-threads-batch 52
run_all_modes "ik_numa_isolate_t52_tb52" --ik-numa isolate --ik-threads 52 --ik-threads-batch 52
run_all_modes "ik_numa_distribute_t32_tb52_repack" --ik-numa distribute --ik-threads 32 --ik-threads-batch 52 --ik-run-time-repack

if command -v numactl >/dev/null 2>&1; then
    run_all_modes "numa0_phys_t26_tb26" \
        --ik-server-prefix "numactl --physcpubind=0-25 --membind=0" \
        --ik-numa numactl \
        --ik-threads 26 --ik-threads-batch 26
    run_all_modes "numa0_smt_t52_tb52" \
        --ik-server-prefix "numactl --cpunodebind=0 --membind=0" \
        --ik-numa numactl \
        --ik-threads 52 --ik-threads-batch 52
    run_all_modes "numa_interleave_t52_tb52" \
        --ik-server-prefix "numactl --interleave=all" \
        --ik-numa numactl \
        --ik-threads 52 --ik-threads-batch 52
fi
