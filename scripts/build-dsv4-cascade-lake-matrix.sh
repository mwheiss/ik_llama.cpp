#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
results_dir=${DSV4_CPU_MATRIX_RESULTS:-"$root/dsv4-cascade-lake-results"}
jobs=${DSV4_CPU_MATRIX_JOBS:-$(nproc)}
targets=${DSV4_CPU_MATRIX_TARGETS:-"llama-server llama-bench test-dsv4-primitives"}

default_cases=(
    gcc-native
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
usage: $0 [case ...]

Build DeepSeek V4 CPU optimization candidates for Cascade Lake.

Default cases:
  ${default_cases[*]}

Environment:
  DSV4_CPU_MATRIX_RESULTS  result directory, default dsv4-cascade-lake-results
  DSV4_CPU_MATRIX_JOBS     build jobs, default nproc
  DSV4_CPU_MATRIX_TARGETS  targets, default "llama-server llama-bench test-dsv4-primitives"
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

mkdir -p "$results_dir"

common_cmake_flags=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_NATIVE=OFF
    -DGGML_AVX=ON
    -DGGML_AVX2=ON
    -DGGML_FMA=ON
    -DGGML_F16C=ON
    -DGGML_AVX512=ON
    -DGGML_AVX512_VNNI=ON
    -DGGML_AVX512_VBMI=OFF
    -DGGML_AVX512_BF16=OFF
    -DGGML_CUDA=OFF
    -DGGML_METAL=OFF
    -DGGML_VULKAN=OFF
    -DGGML_SYCL=OFF
    "-DCMAKE_C_FLAGS=-O3 -march=cascadelake"
    "-DCMAKE_CXX_FLAGS=-O3 -march=cascadelake"
)

require_cmd() {
    local cmd=$1
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "missing required command: $cmd" >&2
        return 1
    fi
}

targets_has() {
    local target=$1
    [[ " $targets " == *" $target "* ]]
}

source_oneapi() {
    if [[ ! -f /opt/intel/oneapi/setvars.sh ]]; then
        echo "missing /opt/intel/oneapi/setvars.sh" >&2
        return 1
    fi
    set +u
    # shellcheck source=/dev/null
    source /opt/intel/oneapi/setvars.sh >/dev/null
    set -u
}

configure_case() {
    local case_id=$1

    build_dir="$root/build-cpu-${case_id}"
    compiler_c=gcc
    compiler_cxx=g++
    need_oneapi=0
    blas=OFF
    blas_vendor=Generic
    lto=OFF
    extra_flags=()

    case "$case_id" in
        gcc-native)
            ;;
        icx-native)
            need_oneapi=1
            compiler_c=icx
            compiler_cxx=icpx
            ;;
        icx-mkl)
            need_oneapi=1
            compiler_c=icx
            compiler_cxx=icpx
            blas=ON
            blas_vendor=Intel10_64lp
            ;;
        gcc-mkl)
            need_oneapi=1
            blas=ON
            blas_vendor=Intel10_64lp
            ;;
        clang-native)
            compiler_c=clang
            compiler_cxx=clang++
            ;;
        clang-mkl)
            need_oneapi=1
            compiler_c=clang
            compiler_cxx=clang++
            blas=ON
            blas_vendor=Intel10_64lp
            ;;
        gcc-flexiblas-openblas)
            blas=ON
            blas_vendor=FlexiBLAS
            ;;
        gcc-blis)
            blas=ON
            blas_vendor=FLAME
            extra_flags+=(
                -DBLAS_INCLUDE_DIRS=/usr/include/blis
                -DBLAS_LIBRARIES=/usr/lib64/libblis.so
            )
            ;;
        gcc-lto)
            lto=ON
            ;;
        icx-mkl-ipo)
            need_oneapi=1
            compiler_c=icx
            compiler_cxx=icpx
            blas=ON
            blas_vendor=Intel10_64lp
            lto=ON
            ;;
        *)
            echo "unknown case: $case_id" >&2
            return 2
            ;;
    esac
}

run_case() {
    local case_id=$1
    configure_case "$case_id"

    local case_results="$results_dir/$case_id"
    mkdir -p "$case_results"
    echo "RUNNING" >"$case_results/status.txt"
    {
        echo "case=$case_id"
        echo "build_dir=$build_dir"
        echo "cc=$compiler_c"
        echo "cxx=$compiler_cxx"
        echo "blas=$blas"
        echo "blas_vendor=$blas_vendor"
        echo "lto=$lto"
        echo "targets=$targets"
        echo "jobs=$jobs"
        git -C "$root" rev-parse HEAD
    } >"$case_results/config.txt"

    if [[ $need_oneapi -eq 1 ]]; then
        source_oneapi
    fi

    require_cmd cmake
    require_cmd ninja
    require_cmd "$compiler_c"
    require_cmd "$compiler_cxx"
    require_cmd objdump

    echo
    echo "===== configure $case_id ====="
    if ! (
        cd "$root"
        cmake -S . -B "$build_dir" -G Ninja \
            -DCMAKE_C_COMPILER="$compiler_c" \
            -DCMAKE_CXX_COMPILER="$compiler_cxx" \
            -DGGML_BLAS="$blas" \
            -DGGML_BLAS_VENDOR="$blas_vendor" \
            -DGGML_LTO="$lto" \
            "${common_cmake_flags[@]}" \
            "${extra_flags[@]}"
    ) 2>&1 | tee "$case_results/configure.log"; then
        echo "configure failed for $case_id" >&2
        return 1
    fi

    echo
    echo "===== build $case_id ====="
    # shellcheck disable=SC2086
    if ! cmake --build "$build_dir" --config Release -j"$jobs" --target $targets \
        2>&1 | tee "$case_results/build.log"; then
        echo "build failed for $case_id" >&2
        return 1
    fi

    cp "$build_dir/CMakeCache.txt" "$case_results/CMakeCache.txt"
    if [[ -x "$build_dir/bin/llama-server" ]]; then
        ldd "$build_dir/bin/llama-server" >"$case_results/ldd.llama-server.txt" || true
        "$build_dir/bin/llama-server" --version >"$case_results/version.llama-server.txt" 2>&1 || true
    fi
    if [[ -x "$build_dir/bin/llama-bench" ]]; then
        ldd "$build_dir/bin/llama-bench" >"$case_results/ldd.llama-bench.txt" || true
    fi

    if targets_has test-dsv4-primitives; then
        if [[ ! -x "$build_dir/bin/test-dsv4-primitives" ]]; then
            echo "missing expected test binary: $build_dir/bin/test-dsv4-primitives" >&2
            return 1
        fi

        echo
        echo "===== primitive test $case_id ====="
        if ! "$build_dir/bin/test-dsv4-primitives" 2>&1 | tee "$case_results/test-dsv4-primitives.log"; then
            echo "primitive test failed for $case_id" >&2
            return 1
        fi
    fi

    if targets_has llama-server; then
        if [[ ! -x "$build_dir/bin/llama-server" ]]; then
            echo "missing expected server binary: $build_dir/bin/llama-server" >&2
            return 1
        fi
        if [[ ! -f "$build_dir/src/libllama.so" || ! -f "$build_dir/ggml/src/libggml.so" ]]; then
            echo "missing expected shared libraries for VNNI check in $build_dir" >&2
            return 1
        fi

        echo
        echo "===== VNNI artifact check $case_id ====="
        if ! "$root/scripts/check-cascade-lake-vnni.sh" "$build_dir" 2>&1 | tee "$case_results/vnni.log"; then
            echo "VNNI artifact check failed for $case_id" >&2
            return 1
        fi

        objdump -d "$build_dir/bin/llama-server" "$build_dir/src/libllama.so" "$build_dir/ggml/src/libggml.so" \
            >"$case_results/objdump-selected.asm" 2>/dev/null || true
        if [[ -s "$case_results/objdump-selected.asm" ]]; then
            rg -c '\bvpdp(busd|wssd)\b' "$case_results/objdump-selected.asm" \
                >"$case_results/vnni-instruction-count.txt" || true
        fi
    fi

    echo "PASS" >"$case_results/status.txt"
}

cases=("$@")
if [[ ${#cases[@]} -eq 0 ]]; then
    cases=("${default_cases[@]}")
fi

failed=()
for case_id in "${cases[@]}"; do
    if ! run_case "$case_id"; then
        echo "FAIL" >"$results_dir/$case_id/status.txt" 2>/dev/null || true
        failed+=("$case_id")
    fi
done

if [[ ${#failed[@]} -ne 0 ]]; then
    printf 'failed cases:' >&2
    printf ' %s' "${failed[@]}" >&2
    printf '\n' >&2
    exit 1
fi

echo
echo "all requested cases passed"
