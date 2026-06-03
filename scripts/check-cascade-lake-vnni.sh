#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <build-dir>" >&2
    exit 2
fi

build_dir=$1
artifacts=(
    "$build_dir/bin/llama-cli"
    "$build_dir/bin/llama-server"
    "$build_dir/src/libllama.so"
    "$build_dir/ggml/src/libggml.so"
)

existing=()
for artifact in "${artifacts[@]}"; do
    if [[ -f "$artifact" ]]; then
        existing+=("$artifact")
    fi
done

if [[ ${#existing[@]} -eq 0 ]]; then
    echo "no build artifacts found under $build_dir" >&2
    exit 1
fi

if ! command -v objdump >/dev/null 2>&1; then
    echo "objdump is required" >&2
    exit 1
fi

dump=$(mktemp "${TMPDIR:-/tmp}/cascade-lake-vnni.XXXXXX.asm")
trap 'rm -f "$dump"' EXIT
objdump -d "${existing[@]}" >"$dump"

if ! grep -Eq '\bvpdp(busd|wssd)\b' "$dump"; then
    echo "missing AVX-512 VNNI instructions (expected vpdpbusd or vpdpwssd)" >&2
    exit 1
fi

if grep -Eiq '\b(vdpbf16ps|tdpbf16ps|tdpbusd|tdpbssd|tdpbuud|vcvtph2psx|vpermb)\b' "$dump"; then
    echo "found an instruction outside the Cascade Lake target envelope" >&2
    grep -Ein '\b(vdpbf16ps|tdpbf16ps|tdpbusd|tdpbssd|tdpbuud|vcvtph2psx|vpermb)\b' "$dump" | head -20 >&2
    exit 1
fi

echo "Cascade Lake VNNI check passed: found AVX-512 VNNI and no AMX/BF16/FP16-native/VBMI markers"

