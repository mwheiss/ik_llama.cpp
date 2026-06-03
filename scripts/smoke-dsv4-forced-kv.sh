#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "usage: $0 <build-dir> [q4-gguf] [q8-gguf]" >&2
    exit 2
fi

build_dir=$1
q4_gguf=${2:-/home/mheiss/.cache/huggingface/hub/models--teamblobfish--DeepSeek-V4-Flash-GGUF/snapshots/b281094221a72c210a2b986709510b4c4b51b67e/Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00001-of-00004.gguf}
q8_gguf=${3:-/home/mheiss/.cache/huggingface/hub/models--teamblobfish--DeepSeek-V4-Flash-GGUF/snapshots/b281094221a72c210a2b986709510b4c4b51b67e/Q8_0/DeepSeek-V4-Flash-Q8_0-00001-of-00007.gguf}

llama_cli="$build_dir/bin/llama-cli"
if [[ ! -x "$llama_cli" ]]; then
    echo "missing llama-cli: $llama_cli" >&2
    exit 2
fi

run_one() {
    local label=$1
    local model=$2
    local log
    log=$(mktemp "${TMPDIR:-/tmp}/dsv4-forced-kv-${label}.XXXXXX.log")

    echo "== $label forced q8_0 KV smoke =="
    timeout 900 "$llama_cli" \
        -m "$model" \
        -ngl 0 \
        -t 52 \
        -tb 52 \
        -c 128 \
        -b 128 \
        -ub 128 \
        -n 1 \
        -p Hello \
        --flash-attn off \
        --no-warmup \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        >"$log" 2>&1

    if ! grep -q "DeepSeek4: forcing fp16 KV cache; requested K=q8_0, V=q8_0" "$log"; then
        echo "$label: missing DeepSeek4 forced-F16 warning" >&2
        tail -80 "$log" >&2
        exit 1
    fi

    if ! grep -q "K (f16):" "$log" || ! grep -q "V (f16):" "$log"; then
        echo "$label: cache summary did not report F16 K/V" >&2
        tail -80 "$log" >&2
        exit 1
    fi

    if grep -Eiq "GGML_ASSERT|assert failed|found (nan|inf)|fatal|exception|runtime_error|segmentation fault" "$log"; then
        echo "$label: failure marker found in log" >&2
        tail -120 "$log" >&2
        exit 1
    fi

    grep -E "DeepSeek4: forcing fp16 KV cache|KV self size|llama_print_timings:.*eval time|Log end" "$log"
    rm -f "$log"
}

run_one q4 "$q4_gguf"
run_one q8 "$q8_gguf"
