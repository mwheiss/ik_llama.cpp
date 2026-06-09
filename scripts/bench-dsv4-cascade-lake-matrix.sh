#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
results_dir=${DSV4_CPU_MATRIX_RESULTS:-"$root/dsv4-cascade-lake-results"}
model=${DSV4_MODEL:-"/home/mheiss/.cache/huggingface/hub/models--teamblobfish--DeepSeek-V4-Flash-GGUF/snapshots/b281094221a72c210a2b986709510b4c4b51b67e/Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00001-of-00004.gguf"}
baseline_build=${DSV4_BASELINE_BUILD:-"$root/build-cpu-opt"}
ctx_size=${DSV4_CTX_SIZE:-1024}
n_predict=${DSV4_N_PREDICT:-192}
filler_lines=${DSV4_FILLER_LINES:-0}
threads=${DSV4_THREADS:-32}
threads_batch=${DSV4_THREADS_BATCH:-52}
numa_policy=${DSV4_NUMA_POLICY:-distribute}
blas_thread_list=${DSV4_BLAS_THREADS:-1}
flash_modes=${DSV4_FLASH_MODES:-on,off}
run_llama_bench=${DSV4_RUN_LLAMA_BENCH:-0}
no_mmap=${DSV4_NO_MMAP:-0}
ik_no_mmap=${DSV4_IK_NO_MMAP:-0}

default_cases=(
    icx-native
    icx-mkl
    gcc-mkl
    clang-native
    clang-mkl
    gcc-flexiblas-openblas
    gcc-blis
    gcc-lto
    icx-mkl-ipo
)

usage() {
    cat <<EOF
usage: $0 [build-dir-or-case ...]

Compare Cascade Lake build candidates against the current optimized baseline
using scripts/engine_test_harness.py. Case names map to build-cpu-<case>.

Environment:
  DSV4_BASELINE_BUILD    baseline build dir, default build-cpu-opt
  DSV4_MODEL             GGUF model path
  DSV4_CPU_MATRIX_RESULTS result dir, default dsv4-cascade-lake-results
  DSV4_CTX_SIZE          harness context, default 1024
  DSV4_N_PREDICT         harness generation length, default 192
  DSV4_THREADS           generation threads, default 32
  DSV4_THREADS_BATCH     batch threads, default 52
  DSV4_NUMA_POLICY       ik --numa policy, default distribute
  DSV4_BLAS_THREADS      comma-separated BLAS thread counts, default 1
  DSV4_FLASH_MODES       comma-separated modes, default on,off
  DSV4_RUN_LLAMA_BENCH   set to 1 for optional llama-bench shapes
  DSV4_NO_MMAP           set to 1 to pass --no-mmap to both engines
  DSV4_IK_NO_MMAP        set to 1 to pass --no-mmap only to the candidate
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

mkdir -p "$results_dir"

if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    # Make MKL-built candidates runnable even when invoked from a plain shell.
    set +u
    # shellcheck source=/dev/null
    source /opt/intel/oneapi/setvars.sh >/dev/null
    set -u
fi

export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}
export KMP_AFFINITY=${KMP_AFFINITY:-granularity=fine,compact,1,0}
export KMP_BLOCKTIME=${KMP_BLOCKTIME:-0}
export MKL_DYNAMIC=${MKL_DYNAMIC:-FALSE}
export OMP_DYNAMIC=${OMP_DYNAMIC:-FALSE}

resolve_build() {
    local value=$1
    if [[ -d "$value" ]]; then
        printf '%s\n' "$value"
    elif [[ -d "$root/$value" ]]; then
        printf '%s\n' "$root/$value"
    else
        printf '%s\n' "$root/build-cpu-$value"
    fi
}

run_harness() {
    local build=$1
    local case_id=$2
    local flash=$3
    local blas_threads=$4

    local mode_name=fa-off
    local flash_arg=()
    if [[ "$flash" == "on" ]]; then
        mode_name=fa-on
        flash_arg=(--flash-attn)
    fi

    local case_results="$results_dir/$case_id"
    mkdir -p "$case_results"
    local log="$case_results/harness-${mode_name}-blas${blas_threads}.log"
    local mmap_args=()
    if [[ "$no_mmap" == "1" ]]; then
        mmap_args+=(--no-mmap)
    fi
    if [[ "$ik_no_mmap" == "1" ]]; then
        mmap_args+=(--ik-no-mmap)
    fi

    export MKL_NUM_THREADS=$blas_threads
    export OPENBLAS_NUM_THREADS=$blas_threads
    export BLIS_NUM_THREADS=$blas_threads
    export OMP_NUM_THREADS=$threads_batch
    if [[ "$case_id" == *flexiblas* ]]; then
        export FLEXIBLAS=${FLEXIBLAS:-OPENBLAS-OPENMP}
    fi

    echo
    echo "===== harness $case_id $mode_name blas_threads=$blas_threads ====="
    python3 "$root/scripts/engine_test_harness.py" \
        "${flash_arg[@]}" \
        --model "$model" \
        --server-bin "$baseline_build/bin/llama-server" \
        --ik-server-bin "$build/bin/llama-server" \
        --ctx-size "$ctx_size" \
        --n-predict "$n_predict" \
        --filler-lines "$filler_lines" \
        --threads "$threads" \
        --threads-batch "$threads_batch" \
        --numa "$numa_policy" \
        --ik-threads "$threads" \
        --ik-threads-batch "$threads_batch" \
        --ik-numa "$numa_policy" \
        "${mmap_args[@]}" \
        2>&1 | tee "$log"

    rg "comparison_status|max_abs_logprob_diff|mean_abs_logprob_diff|baseline_ik perf|opt_ik perf|root:" "$log" \
        >"$case_results/harness-${mode_name}-blas${blas_threads}.summary" || true
}

run_llama_bench_shapes() {
    local build=$1
    local case_id=$2
    local blas_threads=$3

    [[ "$run_llama_bench" == "1" ]] || return 0
    [[ -x "$build/bin/llama-bench" ]] || return 0

    local case_results="$results_dir/$case_id"
    local log="$case_results/llama-bench-blas${blas_threads}.log"

    export MKL_NUM_THREADS=$blas_threads
    export OPENBLAS_NUM_THREADS=$blas_threads
    export BLIS_NUM_THREADS=$blas_threads
    export OMP_NUM_THREADS=$threads_batch

    : >"$log"
    for threads_case in 26 52; do
        for shape in mixed prefill decode; do
            local args=()
            case "$shape" in
                mixed)   args=(-p 512  -n 128) ;;
                prefill) args=(-p 2048 -n 16) ;;
                decode)  args=(-p 64   -n 256) ;;
            esac
            echo "=== $case_id llama-bench threads=$threads_case shape=$shape blas_threads=$blas_threads ===" | tee -a "$log"
            numactl --interleave=all "$build/bin/llama-bench" \
                -m "$model" -t "$threads_case" -tb "$threads_case" "${args[@]}" \
                2>&1 | tee -a "$log"
        done
    done
}

cases=("$@")
if [[ ${#cases[@]} -eq 0 ]]; then
    cases=("${default_cases[@]}")
fi

IFS=',' read -r -a blas_threads_cases <<<"$blas_thread_list"
IFS=',' read -r -a flash_mode_cases <<<"$flash_modes"

failed=()
for case_id in "${cases[@]}"; do
    build=$(resolve_build "$case_id")
    if [[ ! -x "$build/bin/llama-server" ]]; then
        echo "missing llama-server for $case_id at $build" >&2
        failed+=("$case_id")
        continue
    fi
    for bt in "${blas_threads_cases[@]}"; do
        for flash in "${flash_mode_cases[@]}"; do
            case "$flash" in
                on|off) ;;
                *)
                    echo "invalid DSV4_FLASH_MODES entry: $flash" >&2
                    failed+=("$case_id:invalid-flash:$flash")
                    continue
                    ;;
            esac
            run_harness "$build" "$case_id" "$flash" "$bt" || failed+=("$case_id:fa-$flash:blas$bt")
        done
        run_llama_bench_shapes "$build" "$case_id" "$bt" || failed+=("$case_id:llama-bench:blas$bt")
    done
done

if [[ ${#failed[@]} -ne 0 ]]; then
    printf 'failed benchmarks:' >&2
    printf ' %s' "${failed[@]}" >&2
    printf '\n' >&2
    exit 1
fi

echo
echo "all requested benchmarks passed"
