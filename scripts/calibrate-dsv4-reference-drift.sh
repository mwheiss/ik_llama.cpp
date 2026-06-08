#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build-cpu-clx}"
ctx="${2:-1024}"
n_predict="${3:-192}"
filler_lines="${4:-0}"
flash_mode="${5:-off}"
server_bin="${build_dir}/bin/llama-server"
flash_args=()
if [[ "${flash_mode}" == "on" || "${flash_mode}" == "flash-on" || "${flash_mode}" == "fa-on" ]]; then
    flash_args=(--flash-attn)
    flash_mode="on"
elif [[ "${flash_mode}" != "off" && "${flash_mode}" != "flash-off" && "${flash_mode}" != "fa-off" ]]; then
    echo "unknown flash mode: ${flash_mode}" >&2
    exit 2
else
    flash_mode="off"
fi

common_args=(
    --server-bin "${server_bin}"
    --ik-server-bin "${server_bin}"
    --ctx-size "${ctx}"
    --n-predict "${n_predict}"
    --filler-lines "${filler_lines}"
    "${flash_args[@]}"
    --threads 1
    --threads-batch 1
    --max-logprob-diff 1
    --mean-logprob-diff 1
    --warn-max-logprob-diff 1
    --warn-row-logprob-diff 1
    --warn-max-rows-above-logprob-diff 999999
)

summary="reference-drift-calibration-fa-${flash_mode}-${ctx}-${n_predict}-${filler_lines}.tsv"
printf "case\tstatus\tmax_abs_logprob_diff\tmean_abs_logprob_diff\tprefill_tps\tdecode_tps\troot\n" > "${summary}"

run_case() {
    local name="$1"
    shift
    local log="reference-drift-${name}.log"
    local refresh=()
    if [[ "${name}" == "ref_t52_tb52" ]]; then
        refresh=(--refresh-baseline-cache)
    fi

    echo
    echo "===== ${name} ====="
    set +e
    python3 scripts/engine_test_harness.py "${common_args[@]}" "${refresh[@]}" "$@" \
        | tee "${log}" \
        | rg "comparison_status|max_abs_logprob_diff:|mean_abs_logprob_diff:|opt_ik perf:|root:"
    local rc=${PIPESTATUS[0]}
    set -e

    local status maxdiff meandiff prefill decode root
    status=$(rg "^comparison_status:" "${log}" | tail -1 | sed 's/^comparison_status: //')
    maxdiff=$(rg "^max_abs_logprob_diff:" "${log}" | tail -1 | sed 's/^max_abs_logprob_diff: //')
    meandiff=$(rg "^mean_abs_logprob_diff:" "${log}" | tail -1 | sed 's/^mean_abs_logprob_diff: //')
    root=$(rg "^root:" "${log}" | tail -1 | sed 's/^root: //')
    prefill=$(rg "^opt_ik perf:" "${log}" | tail -1 | sed -E 's/.*prefill_tps=([^ ]+).*/\1/')
    decode=$(rg "^opt_ik perf:" "${log}" | tail -1 | sed -E 's/.*decode_tps=([^ ]+).*/\1/')
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${name}" "${status:-rc_${rc}}" "${maxdiff:-}" "${meandiff:-}" "${prefill:-}" "${decode:-}" "${root:-}" | tee -a "${summary}"
}

run_case "ref_t52_tb52" --ik-threads 52 --ik-threads-batch 52
run_case "ref_t32_tb52" --ik-threads 32 --ik-threads-batch 52
run_case "ref_t104_tb104" --ik-threads 104 --ik-threads-batch 104
run_case "ref_numa_distribute_t32_tb52" --ik-numa distribute --ik-threads 32 --ik-threads-batch 52
run_case "ref_numa_distribute_t52_tb52" --ik-numa distribute --ik-threads 52 --ik-threads-batch 52

if command -v numactl >/dev/null 2>&1; then
    run_case "ref_numa0_phys_t26_tb26" \
        --ik-server-prefix "numactl --physcpubind=0-25 --membind=0" \
        --ik-numa numactl \
        --ik-threads 26 \
        --ik-threads-batch 26
fi

echo
echo "summary: ${summary}"
