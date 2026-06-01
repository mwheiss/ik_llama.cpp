#include "build_deepseek4_helpers.h"

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
