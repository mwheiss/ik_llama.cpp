#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <ik-build-dir> <cchuter-build-dir>" >&2
    exit 2
fi

IK_BUILD=$1
CCHUTER_BUILD=$2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IK_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
IK_BUILD=$(cd "$IK_BUILD" && pwd)
CCHUTER_BUILD=$(cd "$CCHUTER_BUILD" && pwd)
CCHUTER_ROOT=$(cd "$CCHUTER_BUILD/.." && pwd)

if ! git -C "$CCHUTER_ROOT" diff --quiet; then
    echo "error: cchuter reference has uncommitted changes: $CCHUTER_ROOT" >&2
    git -C "$CCHUTER_ROOT" status --short >&2
    exit 1
fi

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/dsv4-primitive-parity.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT

PROBE_SRC="$TMP_DIR/dsv4_primitive_probe.cpp"
IK_PROBE="$TMP_DIR/ik_probe"
CCHUTER_PROBE="$TMP_DIR/cchuter_probe"
IK_OUT="$TMP_DIR/ik.out"
CCHUTER_OUT="$TMP_DIR/cchuter.out"

cat > "$PROBE_SRC" <<'CPP'
#include "ggml.h"
#if defined(CCHUTER_ENGINE)
#include "ggml-cpu.h"
#else
#include "build_deepseek4_helpers.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void compute(ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

#if defined(CCHUTER_ENGINE)
    ggml_cplan plan = ggml_graph_plan(gf, 4, nullptr);
#else
    ggml_cplan plan = ggml_graph_plan(gf, 4);
#endif
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();

    const ggml_status status = ggml_graph_compute(gf, &plan);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_graph_compute failed: %d\n", status);
        std::abort();
    }
}

static void emit(const char * name, ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    printf("BEGIN %s %lld\n", name, (long long) n);
    for (int64_t i = 0; i < n; ++i) {
        printf("%.9g\n", ggml_get_f32_1d(t, i));
    }
    printf("END %s\n", name);
}

static void run_hc_split_sinkhorn() {
    ggml_init_params params = { 8 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_hc = 4;
    constexpr int64_t n_tokens = 3;
    constexpr int64_t mix_hc = (2 + n_hc) * n_hc;

    ggml_tensor * mixes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mix_hc, n_tokens);
    ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor * base  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mix_hc);

    ggml_set_f32_1d(scale, 0,  0.50f);
    ggml_set_f32_1d(scale, 1, -0.25f);
    ggml_set_f32_1d(scale, 2,  0.75f);

    for (int64_t i = 0; i < mix_hc; ++i) {
        ggml_set_f32_1d(base, i, 0.01f * (float) ((i % 9) - 4));
        for (int64_t t = 0; t < n_tokens; ++t) {
            ggml_set_f32_nd(mixes, i, t, 0, 0, 0.02f * (float) (i - 7) - 0.03f * (float) t);
        }
    }

    ggml_tensor * out = ggml_dsv4_hc_split_sinkhorn(ctx, mixes, scale, base, n_hc, 5, 1.0e-6f);
    compute(ctx, out);
    emit("hc_split_sinkhorn", out);
    ggml_free(ctx);
}

static void run_fp8_kv_quantize() {
    ggml_init_params params = { 8 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t head_dim = 80;
    constexpr int64_t n_rows = 2;
    constexpr int n_rot = 16;

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, n_rows);
    for (int64_t row = 0; row < n_rows; ++row) {
        for (int64_t i = 0; i < head_dim; ++i) {
            float v = 0.0f;
            if (i < 64) {
                const float sign = (i % 2) == 0 ? 1.0f : -1.0f;
                v = sign * (0.00003f * (float) (i + 1) + 0.125f * (float) (i % 9));
            } else {
                v = 1000.0f + (float) (row * 100 + i);
            }
            if (i == 0) { v = 0.0f; }
            if (i == 1) { v = 1.0e-5f; }
            if (i == 2) { v = -1.0e-5f; }
            if (i == 3) { v = 448.0f; }
            if (i == 4) { v = -448.0f; }
            if (i == 5) { v = 900.0f; }
            if (i == 6) { v = -900.0f; }
            ggml_set_f32_nd(x, i, row, 0, 0, v);
        }
    }

    ggml_tensor * out = ggml_dsv4_fp8_kv_quantize(ctx, x, n_rot);
    compute(ctx, out);
    emit("fp8_kv_quantize", out);
    ggml_free(ctx);
}

static void run_hc_weighted_sum() {
    ggml_init_params params = { 8 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd = 5;
    constexpr int64_t n_hc = 4;
    constexpr int64_t n_tokens = 3;

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_hc, n_tokens);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, n_tokens);

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t h = 0; h < n_hc; ++h) {
            ggml_set_f32_nd(weights, h, t, 0, 0, 0.1f * (float) (h + 1) - 0.03f * (float) t);
            for (int64_t e = 0; e < n_embd; ++e) {
                ggml_set_f32_nd(x, e, h, t, 0, 0.01f * (float) (11*e + 5*h - 7*t));
            }
        }
    }

#if defined(CCHUTER_ENGINE)
    ggml_tensor * out = ggml_dsv4_hc_weighted_sum(ctx, x, weights);
#else
    ggml_tensor * out = llm_build_deepseek4_hc_weighted_sum(ctx, x, weights);
#endif
    compute(ctx, out);
    emit("hc_weighted_sum", out);
    ggml_free(ctx);
}

static void run_hc_expand() {
    ggml_init_params params = { 8 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd = 4;
    constexpr int64_t n_hc = 3;
    constexpr int64_t n_tokens = 2;

    ggml_tensor * block_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * residual  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_hc, n_tokens);
    ggml_tensor * post      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, n_tokens);
    ggml_tensor * comb      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_hc, n_hc, n_tokens);

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_embd; ++e) {
            ggml_set_f32_nd(block_out, e, t, 0, 0, 0.2f * (float) (e + 1) - 0.05f * (float) t);
        }
        for (int64_t h = 0; h < n_hc; ++h) {
            ggml_set_f32_nd(post, h, t, 0, 0, 0.25f + 0.04f * (float) h - 0.02f * (float) t);
            for (int64_t e = 0; e < n_embd; ++e) {
                ggml_set_f32_nd(residual, e, h, t, 0, -0.1f + 0.03f * (float) e + 0.07f * (float) h + 0.02f * (float) t);
            }
            for (int64_t src = 0; src < n_hc; ++src) {
                ggml_set_f32_nd(comb, src, h, t, 0, 0.05f * (float) (src + 1) - 0.04f * (float) h + 0.01f * (float) t);
            }
        }
    }

#if defined(CCHUTER_ENGINE)
    ggml_tensor * out = ggml_dsv4_hc_expand(ctx, block_out, residual, post, comb);
#else
    ggml_tensor * out = llm_build_deepseek4_hc_expand(ctx, block_out, residual, post, comb);
#endif
    compute(ctx, out);
    emit("hc_expand", out);
    ggml_free(ctx);
}

static void run_rope_tail() {
    ggml_init_params params = { 8 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd_head = 8;
    constexpr int64_t n_heads = 2;
    constexpr int64_t n_tokens = 3;
    constexpr int64_t n_rot = 4;

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_heads, n_tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);

    const int32_t positions[n_tokens] = { 1, 3, 7 };
    for (int64_t t = 0; t < n_tokens; ++t) {
        ggml_set_i32_1d(pos, (int) t, positions[t]);
        for (int64_t h = 0; h < n_heads; ++h) {
            for (int64_t e = 0; e < n_embd_head; ++e) {
                ggml_set_f32_nd(x, e, h, t, 0, 0.1f * (float) e + 0.03f * (float) h - 0.02f * (float) t);
            }
        }
    }

#if defined(CCHUTER_ENGINE)
    ggml_tensor * out = ggml_dsv4_rope_tail(ctx, x, pos, nullptr, n_rot, 0, 0,
            10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, false);
#else
    ggml_tensor * out = llm_build_deepseek4_rope_tail(ctx, x, pos, nullptr, n_rot, 0, 0,
            10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, false);
#endif
    compute(ctx, out);
    emit("rope_tail", out);
    ggml_free(ctx);
}

int main() {
    run_hc_split_sinkhorn();
    run_fp8_kv_quantize();
    run_hc_weighted_sum();
    run_hc_expand();
    run_rope_tail();
    return 0;
}
CPP

IK_LIB_DIRS=(
    "$IK_BUILD/src"
    "$IK_BUILD/ggml/src"
    "$IK_BUILD/ggml/src/ggml-cpu"
    "$IK_BUILD/ggml/src/ggml-base"
)
CCHUTER_LIB_DIRS=(
    "$CCHUTER_BUILD/bin"
)

IK_LDFLAGS=()
for dir in "${IK_LIB_DIRS[@]}"; do
    [[ -d "$dir" ]] || continue
    IK_LDFLAGS+=("-L$dir" "-Wl,-rpath,$dir")
done

CCHUTER_LDFLAGS=()
for dir in "${CCHUTER_LIB_DIRS[@]}"; do
    CCHUTER_LDFLAGS+=("-L$dir" "-Wl,-rpath,$dir")
done

IK_LIBS=(
    "$IK_BUILD/src/libllama.so"
    "$IK_BUILD/ggml/src/libggml.so"
)
if [[ -f "$IK_BUILD/ggml/src/ggml-cpu/libggml-cpu.so" ]]; then
    IK_LIBS+=("$IK_BUILD/ggml/src/ggml-cpu/libggml-cpu.so")
fi
if [[ -f "$IK_BUILD/ggml/src/ggml-base/libggml-base.so" ]]; then
    IK_LIBS+=("$IK_BUILD/ggml/src/ggml-base/libggml-base.so")
fi

c++ -std=c++17 -O2 -DIK_ENGINE \
    -I"$IK_ROOT/ggml/include" \
    -I"$IK_ROOT/src/graphs" \
    "$PROBE_SRC" \
    "${IK_LDFLAGS[@]}" \
    "${IK_LIBS[@]}" -lm -pthread \
    -o "$IK_PROBE"

c++ -std=c++17 -O2 -DCCHUTER_ENGINE \
    -I"$CCHUTER_ROOT/ggml/include" \
    "$PROBE_SRC" \
    "${CCHUTER_LDFLAGS[@]}" \
    -lggml -lggml-cpu -lggml-base -lm -pthread \
    -o "$CCHUTER_PROBE"

"$IK_PROBE" > "$IK_OUT"
"$CCHUTER_PROBE" > "$CCHUTER_OUT"

python3 - "$IK_OUT" "$CCHUTER_OUT" <<'PY'
import math
import sys

def read_blocks(path):
    blocks = {}
    current = None
    expected = None
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "BEGIN":
                current = parts[1]
                expected = int(parts[2])
                values = []
            elif parts[0] == "END":
                if current != parts[1]:
                    raise SystemExit(f"{path}: block end mismatch: {line.strip()}")
                if len(values) != expected:
                    raise SystemExit(f"{path}: block {current} has {len(values)} values, expected {expected}")
                blocks[current] = values
                current = None
            else:
                if current is None:
                    raise SystemExit(f"{path}: value outside block: {line.strip()}")
                values.append(float(parts[0]))
    if current is not None:
        raise SystemExit(f"{path}: unterminated block {current}")
    return blocks

ik = read_blocks(sys.argv[1])
cc = read_blocks(sys.argv[2])

expected = [
    "hc_split_sinkhorn",
    "fp8_kv_quantize",
    "hc_weighted_sum",
    "hc_expand",
    "rope_tail",
]

tolerances = {
    "hc_split_sinkhorn": 2.0e-6,
    "fp8_kv_quantize": 1.0e-6,
    "hc_weighted_sum": 2.0e-6,
    "hc_expand": 2.0e-6,
    "rope_tail": 2.0e-6,
}

print("op                         count      max_abs     mean_abs      max_rel  status")
for name in expected:
    if name not in ik or name not in cc:
        raise SystemExit(f"missing block {name}")
    a = ik[name]
    b = cc[name]
    if len(a) != len(b):
        raise SystemExit(f"{name}: length mismatch {len(a)} != {len(b)}")

    diffs = []
    rels = []
    for i, (x, y) in enumerate(zip(a, b)):
        if not math.isfinite(x) or not math.isfinite(y):
            raise SystemExit(f"{name}: non-finite at {i}: ik={x} cchuter={y}")
        diff = abs(x - y)
        diffs.append(diff)
        rels.append(diff / max(1.0, abs(y)))

    max_abs = max(diffs) if diffs else 0.0
    mean_abs = sum(diffs) / len(diffs) if diffs else 0.0
    max_rel = max(rels) if rels else 0.0
    tol = tolerances[name]
    status = "ok" if max_abs <= tol else "FAIL"
    print(f"{name:<25} {len(a):>6} {max_abs:12.6g} {mean_abs:12.6g} {max_rel:12.6g}  {status}")
    if status != "ok":
        worst = max(range(len(diffs)), key=lambda i: diffs[i])
        raise SystemExit(
            f"{name}: max_abs {max_abs:.9g} exceeds {tol:.9g} at {worst}: "
            f"ik={a[worst]:.9g} cchuter={b[worst]:.9g}"
        )
PY
