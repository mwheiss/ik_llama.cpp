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
#include <string>
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

static ggml_tensor * new_filled_2d(ggml_context * ctx, int64_t n0, int64_t n1, float value) {
    return ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1), value);
}

static ggml_tensor * probe_arange_i32(ggml_context * ctx, int64_t begin, int64_t end) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, end - begin);
    for (int64_t i = begin; i < end; ++i) {
        ggml_set_i32_1d(t, i - begin, (int32_t) i);
    }
    return t;
}

static ggml_tensor * probe_view_cols(
        ggml_context * ctx,
        ggml_tensor  * x,
        int64_t        n0,
        int64_t        n1,
        int64_t        off0,
        int64_t        off1) {
    return ggml_view_2d(ctx, x, n0, n1, x->nb[1], off1*x->nb[1] + off0*x->nb[0]);
}

static ggml_tensor * probe_softmax_pool_ratio(ggml_context * ctx, ggml_tensor * kv, ggml_tensor * score) {
    score = ggml_soft_max(ctx, score);
    ggml_tensor * pooled = ggml_mul(ctx, kv, score);
    pooled = ggml_sum_rows(ctx, pooled);
    return ggml_reshape_2d(ctx, pooled, kv->ne[1], kv->ne[2]);
}

static ggml_tensor * probe_pool_decode_state(
        ggml_context * ctx,
        ggml_tensor  * kv,
        ggml_tensor  * score,
        ggml_tensor  * norm,
        ggml_tensor  * pos,
        int64_t        head_dim,
        int64_t        n_rot) {
    const int64_t n_rows = kv->ne[1];
    kv    = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, kv)),    n_rows, head_dim, 1);
    score = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, score)), n_rows, head_dim, 1);
    ggml_tensor * pooled = probe_softmax_pool_ratio(ctx, kv, score);
    pooled = ggml_rms_norm(ctx, pooled, 1.0e-6f);
    pooled = ggml_mul(ctx, pooled, norm);
    pooled = ggml_reshape_3d(ctx, pooled, head_dim, 1, 1);
#if defined(CCHUTER_ENGINE)
    return ggml_dsv4_rope_tail(ctx, pooled, pos, nullptr, n_rot, 0, 0,
            10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, false);
#else
    return llm_build_deepseek4_rope_tail(ctx, pooled, pos, nullptr, n_rot, 0, 0,
            10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, false);
#endif
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

static void run_hc_pre() {
    ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd = 4;
    constexpr int64_t n_hc = 3;
    constexpr int64_t n_tokens = 2;
    constexpr int64_t hc_dim = n_embd * n_hc;
    constexpr int64_t hc_mix = (2 + n_hc) * n_hc;

    ggml_tensor * x        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_hc, n_tokens);
    ggml_tensor * hc_fn    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, hc_mix);
    ggml_tensor * hc_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor * hc_base  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_mix);

    ggml_set_f32_1d(hc_scale, 0,  0.40f);
    ggml_set_f32_1d(hc_scale, 1, -0.15f);
    ggml_set_f32_1d(hc_scale, 2,  0.65f);

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t h = 0; h < n_hc; ++h) {
            for (int64_t e = 0; e < n_embd; ++e) {
                ggml_set_f32_nd(x, e, h, t, 0, 0.03f * (float) e - 0.07f * (float) h + 0.02f * (float) t);
            }
        }
    }
    for (int64_t i = 0; i < hc_mix; ++i) {
        ggml_set_f32_1d(hc_base, i, 0.005f * (float) ((i % 7) - 3));
        for (int64_t d = 0; d < hc_dim; ++d) {
            ggml_set_f32_nd(hc_fn, d, i, 0, 0, 0.001f * (float) ((13*d + 5*i) % 17 - 8));
        }
    }

    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, 1.0e-6f);
#if defined(CCHUTER_ENGINE)
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat);
    ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(ctx, mixes, hc_scale, hc_base, n_hc, 5, 1.0e-6f);
    ggml_tensor * pre  = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], 0);
    ggml_tensor * post = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], n_hc * split->nb[0]);
    ggml_tensor * comb = ggml_view_2d(ctx, split, n_hc * n_hc, n_tokens, split->nb[1], 2 * n_hc * split->nb[0]);
    pre  = ggml_cont(ctx, pre);
    post = ggml_cont(ctx, post);
    comb = ggml_cont(ctx, comb);
    comb = ggml_reshape_3d(ctx, comb, n_hc, n_hc, n_tokens);
    ggml_tensor * y = ggml_dsv4_hc_weighted_sum(ctx, x, pre);
#else
    llm_deepseek4_hc_mix mix = llm_build_deepseek4_hc_pre(ctx,
            x, hc_fn, hc_scale, hc_base, n_embd, n_hc, n_tokens, 1.0e-6f, 5, 1.0e-6f);
    ggml_tensor * y      = mix.x;
    ggml_tensor * mixes  = mix.mixes;
    ggml_tensor * pre    = mix.pre;
    ggml_tensor * post   = mix.post;
    ggml_tensor * comb   = mix.comb;
#endif

    compute(ctx, flat);
    emit("hc_pre_flat", flat);
    compute(ctx, y);
    emit("hc_pre_y", y);
    compute(ctx, mixes);
    emit("hc_pre_mixes", mixes);
    compute(ctx, pre);
    emit("hc_pre_pre", pre);
    compute(ctx, post);
    emit("hc_pre_post", post);
    compute(ctx, comb);
    emit("hc_pre_comb", comb);

    ggml_free(ctx);
}

static void run_hc_head() {
    ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd = 4;
    constexpr int64_t n_hc = 3;
    constexpr int64_t n_tokens = 2;
    constexpr int64_t hc_dim = n_embd * n_hc;

    ggml_tensor * x        = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_hc, n_tokens);
    ggml_tensor * hc_fn    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, n_hc);
    ggml_tensor * hc_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * hc_base  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_hc);

    ggml_set_f32_1d(hc_scale, 0, 0.75f);
    for (int64_t h = 0; h < n_hc; ++h) {
        ggml_set_f32_1d(hc_base, h, 0.02f * (float) (h - 1));
    }
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t h = 0; h < n_hc; ++h) {
            for (int64_t e = 0; e < n_embd; ++e) {
                ggml_set_f32_nd(x, e, h, t, 0, 0.025f * (float) (3*e - 2*h + t));
            }
        }
    }
    for (int64_t h = 0; h < n_hc; ++h) {
        for (int64_t d = 0; d < hc_dim; ++d) {
            ggml_set_f32_nd(hc_fn, d, h, 0, 0, 0.0015f * (float) ((7*d + 3*h) % 11 - 5));
        }
    }

#if defined(CCHUTER_ENGINE)
    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, 1.0e-6f);
    ggml_tensor * pre = ggml_mul_mat(ctx, hc_fn, flat);
    pre = ggml_mul(ctx, pre, ggml_repeat(ctx, ggml_view_2d(ctx, hc_scale, 1, 1, hc_scale->nb[1], 0), pre));
    pre = ggml_add(ctx, pre, ggml_repeat(ctx, ggml_view_2d(ctx, hc_base, n_hc, 1, hc_base->nb[1], 0), pre));
    pre = ggml_scale_bias(ctx, ggml_cont(ctx, ggml_sigmoid(ctx, pre)), 1.0f, 1.0e-6f);
    ggml_tensor * out = ggml_dsv4_hc_weighted_sum(ctx, x, pre);
#else
    ggml_tensor * out = llm_build_deepseek4_hc_head(ctx,
            x, hc_fn, hc_scale, hc_base, n_embd, n_hc, n_tokens, 1.0e-6f, 1.0e-6f);
#endif
    compute(ctx, out);
    emit("hc_head", out);
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

static void run_grouped_out() {
    ggml_init_params params = { 8 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd_head = 3;
    constexpr int64_t n_head = 4;
    constexpr int64_t n_groups = 2;
    constexpr int64_t group_heads = n_head / n_groups;
    constexpr int64_t group_dim = n_embd_head * group_heads;
    constexpr int64_t o_lora_rank = 2;
    constexpr int64_t n_tokens = 2;
    constexpr int64_t n_out = 5;

    ggml_tensor * o    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, n_tokens);
    ggml_tensor * wo_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, group_dim, o_lora_rank * n_groups);
    ggml_tensor * wo_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, o_lora_rank * n_groups, n_out);

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t h = 0; h < n_head; ++h) {
            for (int64_t e = 0; e < n_embd_head; ++e) {
                ggml_set_f32_nd(o, e, h, t, 0, 0.03f * (float) e - 0.04f * (float) h + 0.02f * (float) t);
            }
        }
    }
    for (int64_t col = 0; col < o_lora_rank * n_groups; ++col) {
        for (int64_t row = 0; row < group_dim; ++row) {
            ggml_set_f32_nd(wo_a, row, col, 0, 0, 0.01f * (float) ((7*row + 3*col) % 11 - 5));
        }
    }
    for (int64_t col = 0; col < n_out; ++col) {
        for (int64_t row = 0; row < o_lora_rank * n_groups; ++row) {
            ggml_set_f32_nd(wo_b, row, col, 0, 0, 0.02f * (float) ((5*row - 2*col) % 13 - 6));
        }
    }

#if defined(CCHUTER_ENGINE)
    ggml_tensor * x = ggml_cont(ctx, o);
    x = ggml_reshape_3d(ctx, x, group_dim, n_groups, n_tokens);
    ggml_tensor * wo_a_g = ggml_reshape_3d(ctx, wo_a, group_dim, o_lora_rank, n_groups);
    ggml_tensor * ids = ggml_arange(ctx, 0.0f, (float) n_groups, 1.0f);
    ids = ggml_cast(ctx, ids, GGML_TYPE_I32);
    ids = ggml_repeat_4d(ctx, ids, n_groups, n_tokens, 1, 1);
    ggml_tensor * low = ggml_mul_mat_id(ctx, wo_a_g, x, ids);
    low = ggml_reshape_2d(ctx, low, o_lora_rank * n_groups, n_tokens);
    ggml_tensor * out = ggml_mul_mat(ctx, wo_b, low);
#else
    ggml_tensor * out = llm_build_deepseek4_grouped_out(ctx, o, wo_a, wo_b,
            n_embd_head, n_head, n_groups, o_lora_rank, n_tokens);
#endif
    compute(ctx, out);
    emit("grouped_out", out);
    ggml_free(ctx);
}

static void run_compressor_prefill_state() {
    ggml_init_params params = { 8 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd = 2;
    constexpr int64_t head_dim = 1;
    constexpr int64_t ratio = 4;
    constexpr int64_t width = 2 * head_dim;
    constexpr int64_t n_tokens = 6;

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * wkv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, width);
    ggml_tensor * wgate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, width);
    ggml_tensor * ape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, ratio);

    for (int64_t t = 0; t < n_tokens; ++t) {
        ggml_set_f32_nd(x, 0, t, 0, 0, 10.0f + (float) t);
        ggml_set_f32_nd(x, 1, t, 0, 0, 20.0f + (float) t);
    }
    for (int64_t i = 0; i < n_embd; ++i) {
        for (int64_t o = 0; o < width; ++o) {
            ggml_set_f32_nd(wkv, i, o, 0, 0, 0.0f);
            ggml_set_f32_nd(wgate, i, o, 0, 0, 0.0f);
        }
    }
    ggml_set_f32_nd(wkv, 0, 0, 0, 0, 1.0f);
    ggml_set_f32_nd(wkv, 1, 1, 0, 0, 1.0f);
    ggml_set_f32_nd(wgate, 0, 0, 0, 0, 2.0f);
    ggml_set_f32_nd(wgate, 1, 1, 0, 0, 3.0f);

    for (int64_t c = 0; c < ratio; ++c) {
        ggml_set_f32_nd(ape, 0, c, 0, 0, 0.1f * (float) c);
        ggml_set_f32_nd(ape, 1, c, 0, 0, 1.0f + 0.1f * (float) c);
    }

#if defined(CCHUTER_ENGINE)
    const int64_t cutoff = (n_tokens / ratio) * ratio;
    const int64_t remainder = n_tokens - cutoff;
    ggml_tensor * kv = ggml_mul_mat(ctx, wkv, x);
    ggml_tensor * score = ggml_mul_mat(ctx, wgate, x);

    ggml_tensor * kv_prev = new_filled_2d(ctx, width, ratio, 0.0f);
    ggml_tensor * score_prev = new_filled_2d(ctx, width, ratio, -INFINITY);
    if (cutoff >= ratio) {
        kv_prev = ggml_view_2d(ctx, kv, width, ratio, kv->nb[1], (cutoff - ratio) * kv->nb[1]);
        score_prev = ggml_view_2d(ctx, score, width, ratio, score->nb[1], (cutoff - ratio) * score->nb[1]);
        score_prev = ggml_add(ctx, score_prev, ape);
    }

    ggml_tensor * kv_curr = new_filled_2d(ctx, width, ratio, 0.0f);
    ggml_tensor * score_curr = new_filled_2d(ctx, width, ratio, -INFINITY);
    if (remainder > 0) {
        ggml_tensor * kv_rem = ggml_view_2d(ctx, kv, width, remainder, kv->nb[1], cutoff * kv->nb[1]);
        ggml_tensor * sc_rem = ggml_view_2d(ctx, score, width, remainder, score->nb[1], cutoff * score->nb[1]);
        sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape, width, remainder, ape->nb[1], 0));
        kv_curr = ggml_concat(ctx, kv_rem, new_filled_2d(ctx, width, ratio - remainder, 0.0f), 1);
        score_curr = ggml_concat(ctx, sc_rem, new_filled_2d(ctx, width, ratio - remainder, -INFINITY), 1);
    }

    ggml_tensor * kv_state = ggml_concat(ctx, kv_prev, kv_curr, 1);
    ggml_tensor * score_state = ggml_concat(ctx, score_prev, score_curr, 1);
#else
    llm_deepseek4_state_pair state = llm_build_deepseek4_compressor_prefill_state(
            ctx, x, wkv, wgate, ape, head_dim, ratio);
    ggml_tensor * kv_state = state.kv;
    ggml_tensor * score_state = state.score;
#endif

    compute(ctx, kv_state);
    emit("compressor_prefill_state_kv", kv_state);
    compute(ctx, score_state);
    emit("compressor_prefill_state_score", score_state);

    ggml_free(ctx);
}

static void init_decode_case(
        ggml_tensor * x,
        ggml_tensor * prev_kv_state,
        ggml_tensor * prev_score_state,
        ggml_tensor * wkv,
        ggml_tensor * wgate,
        ggml_tensor * ape,
        ggml_tensor * norm) {
    constexpr int64_t n_embd = 2;
    constexpr int64_t head_dim = 2;
    constexpr int64_t ratio = 4;
    constexpr int64_t width = 2 * head_dim;
    constexpr int64_t rows = 2 * ratio;

    ggml_set_f32_nd(x, 0, 0, 0, 0, 1.25f);
    ggml_set_f32_nd(x, 1, 0, 0, 0, -0.5f);

    for (int64_t i = 0; i < n_embd; ++i) {
        for (int64_t o = 0; o < width; ++o) {
            ggml_set_f32_nd(wkv, i, o, 0, 0, 0.0f);
            ggml_set_f32_nd(wgate, i, o, 0, 0, 0.0f);
        }
    }
    ggml_set_f32_nd(wkv, 0, 0, 0, 0, 1.0f);
    ggml_set_f32_nd(wkv, 1, 1, 0, 0, 1.0f);
    ggml_set_f32_nd(wkv, 0, 2, 0, 0, 1.0f);
    ggml_set_f32_nd(wkv, 1, 2, 0, 0, 1.0f);
    ggml_set_f32_nd(wkv, 0, 3, 0, 0, 2.0f);
    ggml_set_f32_nd(wkv, 1, 3, 0, 0, -1.0f);

    ggml_set_f32_nd(wgate, 0, 0, 0, 0, 2.0f);
    ggml_set_f32_nd(wgate, 1, 1, 0, 0, 3.0f);
    ggml_set_f32_nd(wgate, 0, 2, 0, 0, 1.0f);
    ggml_set_f32_nd(wgate, 1, 2, 0, 0, -1.0f);
    ggml_set_f32_nd(wgate, 0, 3, 0, 0, -1.0f);
    ggml_set_f32_nd(wgate, 1, 3, 0, 0, 0.5f);

    for (int64_t c = 0; c < ratio; ++c) {
        for (int64_t r = 0; r < width; ++r) {
            ggml_set_f32_nd(ape, r, c, 0, 0, 0.01f * (float) r + 0.1f * (float) c);
        }
    }
    for (int64_t r = 0; r < head_dim; ++r) {
        ggml_set_f32_1d(norm, r, 1.0f);
    }
    for (int64_t c = 0; c < rows; ++c) {
        for (int64_t r = 0; r < width; ++r) {
            ggml_set_f32_nd(prev_kv_state, r, c, 0, 0, 100.0f + 10.0f * (float) c + (float) r);
            ggml_set_f32_nd(prev_score_state, r, c, 0, 0, -100.0f - 10.0f * (float) c - (float) r);
        }
    }
}

static void run_compressor_decode_one(int64_t pos, const char * prefix) {
    ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd = 2;
    constexpr int64_t head_dim = 2;
    constexpr int64_t ratio = 4;
    constexpr int64_t width = 2 * head_dim;
    constexpr int64_t rows = 2 * ratio;

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_tensor * prev_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    ggml_tensor * prev_score = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    ggml_tensor * wkv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, width);
    ggml_tensor * wgate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, width);
    ggml_tensor * ape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, ratio);
    ggml_tensor * norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
    init_decode_case(x, prev_kv, prev_score, wkv, wgate, ape, norm);

#if defined(CCHUTER_ENGINE)
    const int64_t pos_mod = pos % ratio;
    const int64_t row = ratio + pos_mod;
    ggml_tensor * kv_cur = ggml_mul_mat(ctx, wkv, x);
    ggml_tensor * sc_cur = ggml_mul_mat(ctx, wgate, x);
    sc_cur = ggml_add(ctx, sc_cur, ggml_view_2d(ctx, ape, width, 1, ape->nb[1], pos_mod*ape->nb[1]));
    ggml_tensor * kv_state = ggml_set_2d_inplace(ctx, prev_kv, kv_cur, prev_kv->nb[1], row*prev_kv->nb[1]);
    ggml_tensor * score_state = ggml_set_2d_inplace(ctx, prev_score, sc_cur, prev_score->nb[1], row*prev_score->nb[1]);
    ggml_tensor * kv_comp = nullptr;
    if ((pos + 1) % ratio == 0) {
        ggml_tensor * kv_prev = probe_view_cols(ctx, kv_state, head_dim, ratio, 0, 0);
        ggml_tensor * kv_curr = probe_view_cols(ctx, kv_state, head_dim, ratio, head_dim, ratio);
        ggml_tensor * sc_prev = probe_view_cols(ctx, score_state, head_dim, ratio, 0, 0);
        ggml_tensor * sc_curr = probe_view_cols(ctx, score_state, head_dim, ratio, head_dim, ratio);
        ggml_tensor * kv_pool = ggml_concat(ctx, kv_prev, kv_curr, 1);
        ggml_tensor * score_pool = ggml_concat(ctx, sc_prev, sc_curr, 1);
        ggml_tensor * shifted_kv = probe_view_cols(ctx, kv_state, width, ratio, 0, ratio);
        ggml_tensor * shifted_score = probe_view_cols(ctx, score_state, width, ratio, 0, ratio);
        kv_state = ggml_concat(ctx, shifted_kv, shifted_kv, 1);
        score_state = ggml_concat(ctx, shifted_score, shifted_score, 1);
        kv_comp = probe_pool_decode_state(ctx, kv_pool, score_pool, norm,
                probe_arange_i32(ctx, pos + 1 - ratio, pos + 2 - ratio), head_dim, 2);
    }
#else
    llm_deepseek4_decode_compressor dec = llm_build_deepseek4_compressor_decode(
            ctx, x, prev_kv, prev_score, wkv, wgate, ape, norm,
            head_dim, 2, pos, ratio, 0, 0,
            10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0e-6f);
    ggml_tensor * kv_state = dec.kv_state;
    ggml_tensor * score_state = dec.score_state;
    ggml_tensor * kv_comp = dec.kv_comp;
#endif

    compute(ctx, ggml_sum(ctx, kv_state));
    emit((std::string(prefix) + "_kv_state").c_str(), kv_state);
    compute(ctx, ggml_sum(ctx, score_state));
    emit((std::string(prefix) + "_score_state").c_str(), score_state);
    if (kv_comp != nullptr) {
        compute(ctx, kv_comp);
        emit((std::string(prefix) + "_kv_comp").c_str(), kv_comp);
    }
    ggml_free(ctx);
}

static void run_compressor_decode() {
    run_compressor_decode_one(5, "compressor_decode_pos5");
    run_compressor_decode_one(7, "compressor_decode_pos7");
}

int main() {
    run_hc_split_sinkhorn();
    run_fp8_kv_quantize();
    run_hc_weighted_sum();
    run_hc_expand();
    run_hc_pre();
    run_hc_head();
    run_grouped_out();
    run_compressor_prefill_state();
    run_compressor_decode();
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
import struct
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
    "hc_pre_flat",
    "hc_pre_mixes",
    "hc_pre_pre",
    "hc_pre_post",
    "hc_pre_comb",
    "hc_pre_y",
    "hc_head",
    "grouped_out",
    "compressor_prefill_state_kv",
    "compressor_prefill_state_score",
    "compressor_decode_pos5_kv_state",
    "compressor_decode_pos5_score_state",
    "compressor_decode_pos7_kv_state",
    "compressor_decode_pos7_score_state",
    "compressor_decode_pos7_kv_comp",
    "rope_tail",
]

tolerances = {
    "hc_split_sinkhorn": 2.0e-6,
    "fp8_kv_quantize": 1.0e-6,
    "hc_weighted_sum": 2.0e-6,
    "hc_expand": 2.0e-6,
    "hc_pre_flat": 2.0e-6,
    "hc_pre_y": 2.0e-6,
    "hc_pre_mixes": 2.0e-6,
    "hc_pre_pre": 2.0e-6,
    "hc_pre_post": 2.0e-6,
    "hc_pre_comb": 2.0e-6,
    "hc_head": 2.0e-6,
    "grouped_out": 2.0e-6,
    "compressor_prefill_state_kv": 2.0e-6,
    "compressor_prefill_state_score": 2.0e-6,
    "compressor_decode_pos5_kv_state": 2.0e-6,
    "compressor_decode_pos5_score_state": 2.0e-6,
    "compressor_decode_pos7_kv_state": 2.0e-6,
    "compressor_decode_pos7_score_state": 2.0e-6,
    "compressor_decode_pos7_kv_comp": 2.0e-6,
    "rope_tail": 2.0e-6,
}

def f32_ordered_bits(x):
    bits = struct.unpack("!I", struct.pack("!f", float(x)))[0]
    return (~bits & 0xffffffff) if (bits & 0x80000000) else (bits | 0x80000000)

def f32_ulp_distance(x, y):
    return abs(f32_ordered_bits(x) - f32_ordered_bits(y))

print("op                         count      max_abs     mean_abs      max_rel  worst  ulp  status")
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
            if math.isinf(x) and math.isinf(y) and math.copysign(1.0, x) == math.copysign(1.0, y):
                diff = 0.0
                rel = 0.0
            else:
                raise SystemExit(f"{name}: non-finite mismatch at {i}: ik={x} cchuter={y}")
        else:
            diff = abs(x - y)
            rel = diff / max(1.0, abs(y))
        diffs.append(diff)
        rels.append(rel)

    max_abs = max(diffs) if diffs else 0.0
    mean_abs = sum(diffs) / len(diffs) if diffs else 0.0
    max_rel = max(rels) if rels else 0.0
    worst = max(range(len(diffs)), key=lambda i: diffs[i]) if diffs else 0
    worst_ulp = f32_ulp_distance(a[worst], b[worst]) if diffs else 0
    tol = tolerances[name]
    status = "ok" if max_abs <= tol else "FAIL"
    print(
        f"{name:<25} {len(a):>6} {max_abs:12.6g} {mean_abs:12.6g} {max_rel:12.6g} "
        f"{worst:6d} {worst_ulp:4d}  {status}"
    )
    if max_abs != 0.0:
        print(f"  worst {name}[{worst}]: ik={a[worst]:.9g} cchuter={b[worst]:.9g}")
    if status != "ok":
        raise SystemExit(
            f"{name}: max_abs {max_abs:.9g} exceeds {tol:.9g} at {worst}: "
            f"ik={a[worst]:.9g} cchuter={b[worst]:.9g}, ulp={worst_ulp}"
        )
PY
