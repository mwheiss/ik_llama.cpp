#include "build_deepseek4_helpers.h"

static struct ggml_tensor * llm_build_deepseek4_hc_mix_project(
        struct ggml_context * ctx,
        struct ggml_tensor  * hc_fn,
        struct ggml_tensor  * flat) {
    GGML_ASSERT(hc_fn->type == GGML_TYPE_F32);
    GGML_ASSERT(flat->type  == GGML_TYPE_F32);
    GGML_ASSERT(hc_fn->ne[0] == flat->ne[0]);
    GGML_ASSERT(hc_fn->ne[2] == 1);
    GGML_ASSERT(hc_fn->ne[3] == 1);
    GGML_ASSERT(flat->ne[2] == 1);
    GGML_ASSERT(flat->ne[3] == 1);

    const int64_t hc_dim   = flat->ne[0];
    const int64_t hc_mix   = hc_fn->ne[1];

    struct ggml_tensor * out = nullptr;
    for (int64_t im = 0; im < hc_mix; ++im) {
        struct ggml_tensor * w = ggml_view_2d(ctx, hc_fn,
                hc_dim, 1,
                hc_fn->nb[1], im*hc_fn->nb[1]);
        w = ggml_repeat(ctx, w, flat);

        struct ggml_tensor * row = ggml_sum_rows(ctx, ggml_mul(ctx, flat, w));
        out = out ? ggml_concat(ctx, out, row, 0) : row;
    }

    return out;
}

struct ggml_tensor * llm_build_deepseek4_rope_tail(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * inp_pos,
        struct ggml_tensor  * rope_factors,
        int64_t               n_rot,
        int                   rope_type,
        int32_t               n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  inverse) {
    GGML_ASSERT(x->ne[2] == inp_pos->ne[0]);
    GGML_ASSERT(n_rot > 0);
    GGML_ASSERT(n_rot <= x->ne[0]);
    GGML_ASSERT(n_rot % 2 == 0);

    if (n_rot == x->ne[0]) {
        return inverse
            ? ggml_rope_back(ctx, x, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow)
            : ggml_rope_ext(ctx, x, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
    }

    const int64_t n_nope = x->ne[0] - n_rot;
    GGML_ASSERT(n_nope > 0);

    struct ggml_tensor * nope = ggml_view_3d(ctx, x,
            n_nope, x->ne[1], x->ne[2],
            x->nb[1], x->nb[2], 0);
    struct ggml_tensor * tail = ggml_view_3d(ctx, x,
            n_rot, x->ne[1], x->ne[2],
            x->nb[1], x->nb[2], n_nope*x->nb[0]);

    tail = inverse
        ? ggml_rope_back(ctx, tail, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow)
        : ggml_rope_ext(ctx, tail, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

    return ggml_concat(ctx, nope, tail, 0);
}

struct ggml_tensor * llm_build_deepseek4_hc_weighted_sum(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * weights) {
    GGML_ASSERT(x->type       == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[1] == weights->ne[0]);
    GGML_ASSERT(x->ne[2] == weights->ne[1]);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_hc     = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    struct ggml_tensor * acc = nullptr;

    for (int64_t ih = 0; ih < n_hc; ++ih) {
        struct ggml_tensor * xh = ggml_view_2d(ctx, x,
                n_embd, n_tokens,
                x->nb[2], ih*x->nb[1]);
        struct ggml_tensor * wh = ggml_view_2d(ctx, weights,
                1, n_tokens,
                weights->nb[1], ih*weights->nb[0]);
        wh = ggml_repeat(ctx, wh, xh);

        struct ggml_tensor * term = ggml_mul(ctx, xh, wh);
        acc = acc ? ggml_add(ctx, acc, term) : term;
    }

    return acc;
}

struct llm_deepseek4_hc_mix llm_build_deepseek4_hc_pre(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * hc_fn,
        struct ggml_tensor  * hc_scale,
        struct ggml_tensor  * hc_base,
        int64_t               n_embd,
        int64_t               n_hc,
        int64_t               n_tokens,
        float                 norm_eps,
        int                   sinkhorn_iters,
        float                 hc_eps) {
    GGML_ASSERT(x->type        == GGML_TYPE_F32);
    GGML_ASSERT(hc_fn->type    == GGML_TYPE_F32);
    GGML_ASSERT(hc_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(hc_base->type  == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == n_hc);
    GGML_ASSERT(x->ne[2] == n_tokens);
    GGML_ASSERT(hc_fn->ne[0] == n_embd * n_hc);
    GGML_ASSERT(hc_fn->ne[1] == (2 + n_hc) * n_hc);
    GGML_ASSERT(hc_scale->ne[0] == 3);
    GGML_ASSERT(hc_base->ne[0] == (2 + n_hc) * n_hc);

    const int64_t hc_dim = n_embd * n_hc;

    struct ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);

    // Keep this staged DSV4 reference path independent of backend GEMM fast
    // paths. The output is scalar-equivalent to ggml_mul_mat(hc_fn, flat).
    struct ggml_tensor * mixes = llm_build_deepseek4_hc_mix_project(ctx, hc_fn, flat);
    struct ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(ctx, mixes, hc_scale, hc_base, n_hc, sinkhorn_iters, hc_eps);

    struct ggml_tensor * pre  = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], 0);
    struct ggml_tensor * post = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], n_hc * split->nb[0]);
    struct ggml_tensor * comb = ggml_view_2d(ctx, split, n_hc * n_hc, n_tokens, split->nb[1], 2 * n_hc * split->nb[0]);
    if (n_tokens != 1) {
        pre  = ggml_cont(ctx, pre);
        post = ggml_cont(ctx, post);
        comb = ggml_cont(ctx, comb);
    }
    comb = ggml_reshape_3d(ctx, comb, n_hc, n_hc, n_tokens);

    struct ggml_tensor * y = llm_build_deepseek4_hc_weighted_sum(ctx, x, pre);
    return { y, mixes, pre, post, comb };
}

struct ggml_tensor * llm_build_deepseek4_hc_expand(
        struct ggml_context * ctx,
        struct ggml_tensor  * block_out,
        struct ggml_tensor  * residual,
        struct ggml_tensor  * post,
        struct ggml_tensor  * comb) {
    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(block_out->ne[0] == residual->ne[0]);
    GGML_ASSERT(block_out->ne[1] == residual->ne[2]);
    GGML_ASSERT(post->ne[0] == residual->ne[1]);
    GGML_ASSERT(post->ne[1] == residual->ne[2]);
    GGML_ASSERT(comb->ne[0] == residual->ne[1]);
    GGML_ASSERT(comb->ne[1] == residual->ne[1]);
    GGML_ASSERT(comb->ne[2] == residual->ne[2]);

    const int64_t n_embd   = residual->ne[0];
    const int64_t n_hc     = residual->ne[1];
    const int64_t n_tokens = residual->ne[2];

    struct ggml_tensor * out = nullptr;

    for (int64_t dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
        struct ggml_tensor * post_h = ggml_view_2d(ctx, post,
                1, n_tokens,
                post->nb[1], dst_hc*post->nb[0]);
        struct ggml_tensor * acc = ggml_mul(ctx, block_out, ggml_repeat(ctx, post_h, block_out));

        for (int64_t src_hc = 0; src_hc < n_hc; ++src_hc) {
            struct ggml_tensor * res_h = ggml_view_2d(ctx, residual,
                    n_embd, n_tokens,
                    residual->nb[2], src_hc*residual->nb[1]);
            // cchuter's expand contract deliberately reads comb transposed:
            // comb[dst_hc, src_hc, t] for a source layout [src_hc, dst_hc, t].
            struct ggml_tensor * comb_h = ggml_view_2d(ctx, comb,
                    1, n_tokens,
                    comb->nb[2], dst_hc*comb->nb[0] + src_hc*comb->nb[1]);
            acc = ggml_add(ctx, acc, ggml_mul(ctx, res_h, ggml_repeat(ctx, comb_h, res_h)));
        }

        struct ggml_tensor * acc_3d = ggml_reshape_3d(ctx, acc, n_embd, 1, n_tokens);
        out = out ? ggml_concat(ctx, out, acc_3d, 1) : acc_3d;
    }

    return out;
}
