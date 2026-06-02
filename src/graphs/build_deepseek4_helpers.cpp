#include "build_deepseek4_helpers.h"

#include <cmath>

static struct ggml_tensor * llm_build_deepseek4_f32_project(
        struct ggml_context * ctx,
        struct ggml_tensor  * w,
        struct ggml_tensor  * flat) {
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(flat->type  == GGML_TYPE_F32);
    GGML_ASSERT(w->ne[0] == flat->ne[0]);
    GGML_ASSERT(w->ne[2] == 1);
    GGML_ASSERT(w->ne[3] == 1);
    GGML_ASSERT(flat->ne[2] == 1);
    GGML_ASSERT(flat->ne[3] == 1);

    const int64_t hc_dim   = flat->ne[0];
    const int64_t hc_mix   = w->ne[1];

    struct ggml_tensor * out = nullptr;
    for (int64_t im = 0; im < hc_mix; ++im) {
        struct ggml_tensor * wi = ggml_view_2d(ctx, w,
                hc_dim, 1,
                w->nb[1], im*w->nb[1]);
        wi = ggml_repeat(ctx, wi, flat);

        struct ggml_tensor * row = ggml_sum_rows(ctx, ggml_mul(ctx, flat, wi));
        out = out ? ggml_concat(ctx, out, row, 0) : row;
    }

    return out;
}

static struct ggml_tensor * llm_build_deepseek4_softmax_pool_ratio(
        struct ggml_context * ctx,
        struct ggml_tensor  * kv,
        struct ggml_tensor  * score) {
    score = ggml_soft_max(ctx, score);
    struct ggml_tensor * pooled = ggml_mul(ctx, kv, score);
    pooled = ggml_sum_rows(ctx, pooled);
    return ggml_reshape_2d(ctx, pooled, kv->ne[1], kv->ne[2]);
}

static struct ggml_tensor * llm_build_deepseek4_shift_overlap_state(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        float                 pad_value) {
    const int64_t n_embd = x->ne[0];
    const int64_t ratio  = x->ne[1];
    const int64_t n_comp = x->ne[2];

    struct ggml_tensor * first = ggml_view_3d(ctx, x, n_embd, ratio, 1,
            x->nb[1], x->nb[2], 0);
    struct ggml_tensor * pad = ggml_fill(ctx, ggml_cont(ctx, first), pad_value);

    if (n_comp == 1) {
        return pad;
    }

    struct ggml_tensor * prev = ggml_view_3d(ctx, x, n_embd, ratio, n_comp - 1,
            x->nb[1], x->nb[2], 0);
    return ggml_concat(ctx, pad, prev, 2);
}

static struct ggml_tensor * llm_build_deepseek4_add_scalar(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        float                 value) {
    struct ggml_tensor * shape = x;
    x = ggml_cont(ctx, x);
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
    x = ggml_scale_bias(ctx, x, 1.0f, value);
    return ggml_reshape(ctx, x, shape);
}

static struct ggml_tensor * llm_build_deepseek4_mul_scalar(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        float                 value) {
    struct ggml_tensor * shape = x;
    x = ggml_cont(ctx, x);
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
    x = ggml_scale(ctx, x, value);
    return ggml_reshape(ctx, x, shape);
}

static struct ggml_tensor * llm_build_deepseek4_new_filled_3d(
        struct ggml_context * ctx,
        int64_t               n0,
        int64_t               n1,
        int64_t               n2,
        float                 value) {
    return ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2), value);
}

static struct ggml_tensor * llm_build_deepseek4_new_filled_2d(
        struct ggml_context * ctx,
        int64_t               n0,
        int64_t               n1,
        float                 value) {
    return ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1), value);
}

static struct ggml_tensor * llm_build_deepseek4_arange_i32(
        struct ggml_context * ctx,
        int64_t               begin,
        int64_t               end) {
    GGML_ASSERT(end >= begin);
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, end - begin);
    for (int64_t i = begin; i < end; ++i) {
        ggml_set_i32_1d(t, i - begin, (int32_t) i);
    }
    return t;
}

static struct ggml_tensor * llm_build_deepseek4_view_cols(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        int64_t               n0,
        int64_t               n1,
        int64_t               off0,
        int64_t               off1) {
    return ggml_view_2d(ctx, x, n0, n1, x->nb[1], off1*x->nb[1] + off0*x->nb[0]);
}

static struct ggml_tensor * llm_build_deepseek4_pool_decode_state(
        struct ggml_context * ctx,
        struct ggml_tensor  * kv,
        struct ggml_tensor  * score,
        struct ggml_tensor  * norm,
        struct ggml_tensor  * pos,
        int64_t               n_embd_head,
        int64_t               n_rot,
        int                   rope_type,
        int32_t               n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 norm_eps) {
    const int64_t n_rows = kv->ne[1];
    kv    = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, kv)),    n_rows, n_embd_head, 1);
    score = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, score)), n_rows, n_embd_head, 1);

    struct ggml_tensor * pooled = llm_build_deepseek4_softmax_pool_ratio(ctx, kv, score);
    pooled = ggml_rms_norm(ctx, pooled, norm_eps);
    pooled = ggml_mul(ctx, pooled, norm);
    pooled = ggml_reshape_3d(ctx, pooled, n_embd_head, 1, 1);

    return llm_build_deepseek4_rope_tail(ctx, pooled, pos, nullptr, n_rot, rope_type,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, false);
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

    // Keep the F32 staged reference path independent of backend GEMM fast
    // paths. Quantized model tensors still use GGML's normal mul_mat path.
    struct ggml_tensor * mixes = hc_fn->type == GGML_TYPE_F32
        ? llm_build_deepseek4_f32_project(ctx, hc_fn, flat)
        : ggml_mul_mat(ctx, hc_fn, flat);
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

struct ggml_tensor * llm_build_deepseek4_hc_head(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * hc_fn,
        struct ggml_tensor  * hc_scale,
        struct ggml_tensor  * hc_base,
        int64_t               n_embd,
        int64_t               n_hc,
        int64_t               n_tokens,
        float                 norm_eps,
        float                 hc_eps) {
    GGML_ASSERT(x->type        == GGML_TYPE_F32);
    GGML_ASSERT(hc_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(hc_base->type  == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == n_hc);
    GGML_ASSERT(x->ne[2] == n_tokens);
    GGML_ASSERT(hc_fn->ne[0] == n_embd * n_hc);
    GGML_ASSERT(hc_fn->ne[1] == n_hc);
    GGML_ASSERT(hc_scale->ne[0] == 1);
    GGML_ASSERT(hc_base->ne[0] == n_hc);

    const int64_t hc_dim = n_embd * n_hc;

    struct ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);

    struct ggml_tensor * pre = hc_fn->type == GGML_TYPE_F32
        ? llm_build_deepseek4_f32_project(ctx, hc_fn, flat)
        : ggml_mul_mat(ctx, hc_fn, flat);

    struct ggml_tensor * scale = ggml_view_2d(ctx, hc_scale, 1, 1, hc_scale->nb[1], 0);
    scale = ggml_repeat(ctx, scale, pre);
    struct ggml_tensor * base = ggml_view_2d(ctx, hc_base, n_hc, 1, hc_base->nb[1], 0);
    base = ggml_repeat(ctx, base, pre);

    pre = ggml_add(ctx, ggml_mul(ctx, pre, scale), base);
    pre = llm_build_deepseek4_add_scalar(ctx, ggml_sigmoid(ctx, pre), hc_eps);

    return llm_build_deepseek4_hc_weighted_sum(ctx, x, pre);
}

struct ggml_tensor * llm_build_deepseek4_grouped_out(
        struct ggml_context * ctx,
        struct ggml_tensor  * o,
        struct ggml_tensor  * wo_a,
        struct ggml_tensor  * wo_b,
        int64_t               n_embd_head,
        int64_t               n_head,
        int64_t               n_groups,
        int64_t               o_lora_rank,
        int64_t               n_tokens) {
    GGML_ASSERT(o->type == GGML_TYPE_F32);
    GGML_ASSERT(n_head % n_groups == 0);
    GGML_ASSERT(o->ne[0] == n_embd_head);
    GGML_ASSERT(o->ne[1] == n_head);
    GGML_ASSERT(o->ne[2] == n_tokens);

    const int64_t group_heads = n_head / n_groups;
    const int64_t group_dim   = n_embd_head * group_heads;

    o = ggml_cont(ctx, o);
    o = ggml_reshape_3d(ctx, o, group_dim, n_groups, n_tokens);

    GGML_ASSERT(wo_a->ne[0] == group_dim);
    GGML_ASSERT(wo_a->ne[1] == o_lora_rank * n_groups);
    GGML_ASSERT(wo_b->ne[0] == o_lora_rank * n_groups);

    struct ggml_tensor * low = nullptr;
    for (int64_t ig = 0; ig < n_groups; ++ig) {
        struct ggml_tensor * o_g = ggml_view_2d(ctx, o,
                group_dim, n_tokens,
                o->nb[2], ig*o->nb[1]);
        struct ggml_tensor * wo_a_g = ggml_view_2d(ctx, wo_a,
                group_dim, o_lora_rank,
                wo_a->nb[1], ig*o_lora_rank*wo_a->nb[1]);
        struct ggml_tensor * low_g = wo_a_g->type == GGML_TYPE_F32 && o_g->type == GGML_TYPE_F32
            ? llm_build_deepseek4_f32_project(ctx, wo_a_g, o_g)
            : ggml_mul_mat(ctx, wo_a_g, o_g);
        low = low ? ggml_concat(ctx, low, low_g, 0) : low_g;
    }

    return wo_b->type == GGML_TYPE_F32 && low->type == GGML_TYPE_F32
        ? llm_build_deepseek4_f32_project(ctx, wo_b, low)
        : ggml_mul_mat(ctx, wo_b, low);
}

struct ggml_tensor * llm_build_deepseek4_compressor_prefill(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * wkv,
        struct ggml_tensor  * wgate,
        struct ggml_tensor  * ape,
        struct ggml_tensor  * norm,
        struct ggml_tensor  * pos,
        int64_t               n_embd_head,
        int64_t               n_rot,
        int64_t               compress_ratio,
        int                   rope_type,
        int32_t               n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 norm_eps) {
    GGML_ASSERT(compress_ratio > 0);
    const int64_t n_tokens = x->ne[1];
    const int64_t n_comp = n_tokens / compress_ratio;
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp * compress_ratio <= n_tokens);

    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t n_kv = coff * n_embd_head;

    GGML_ASSERT(wkv->ne[0] == x->ne[0]);
    GGML_ASSERT(wkv->ne[1] == n_kv);
    GGML_ASSERT(wgate->ne[0] == x->ne[0]);
    GGML_ASSERT(wgate->ne[1] == n_kv);
    GGML_ASSERT(ape->ne[0] == n_kv);
    GGML_ASSERT(ape->ne[1] == compress_ratio);
    GGML_ASSERT(norm->ne[0] == n_embd_head);
    GGML_ASSERT(pos->ne[0] == n_comp);

    struct ggml_tensor * kv    = ggml_mul_mat(ctx, wkv,   x);
    struct ggml_tensor * score = ggml_mul_mat(ctx, wgate, x);

    kv = ggml_view_3d(ctx, kv, n_kv, compress_ratio, n_comp,
            kv->nb[1], kv->nb[1]*compress_ratio, 0);
    score = ggml_view_3d(ctx, score, n_kv, compress_ratio, n_comp,
            score->nb[1], score->nb[1]*compress_ratio, 0);

    struct ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
    score = ggml_add(ctx, score, ggml_repeat(ctx, ape_f, score));

    if (coff == 1) {
        kv    = ggml_cont(ctx, ggml_permute(ctx, kv,    1, 0, 2, 3));
        score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));
        kv = llm_build_deepseek4_softmax_pool_ratio(ctx, kv, score);
    } else {
        struct ggml_tensor * kv_prev = ggml_view_3d(ctx, kv, n_embd_head, compress_ratio, n_comp,
                kv->nb[1], kv->nb[2], 0);
        struct ggml_tensor * kv_curr = ggml_view_3d(ctx, kv, n_embd_head, compress_ratio, n_comp,
                kv->nb[1], kv->nb[2], n_embd_head*kv->nb[0]);
        struct ggml_tensor * score_prev = ggml_view_3d(ctx, score, n_embd_head, compress_ratio, n_comp,
                score->nb[1], score->nb[2], 0);
        struct ggml_tensor * score_curr = ggml_view_3d(ctx, score, n_embd_head, compress_ratio, n_comp,
                score->nb[1], score->nb[2], n_embd_head*score->nb[0]);

        kv_prev    = llm_build_deepseek4_shift_overlap_state(ctx, kv_prev,    0.0f);
        score_prev = llm_build_deepseek4_shift_overlap_state(ctx, score_prev, -INFINITY);

        kv_prev    = ggml_cont(ctx, ggml_permute(ctx, kv_prev,    1, 0, 2, 3));
        kv_curr    = ggml_cont(ctx, ggml_permute(ctx, kv_curr,    1, 0, 2, 3));
        score_prev = ggml_cont(ctx, ggml_permute(ctx, score_prev, 1, 0, 2, 3));
        score_curr = ggml_cont(ctx, ggml_permute(ctx, score_curr, 1, 0, 2, 3));

        kv    = ggml_concat(ctx, kv_prev,    kv_curr,    0);
        score = ggml_concat(ctx, score_prev, score_curr, 0);
        kv = llm_build_deepseek4_softmax_pool_ratio(ctx, kv, score);
    }

    kv = ggml_rms_norm(ctx, kv, norm_eps);
    kv = ggml_mul(ctx, kv, norm);
    kv = ggml_reshape_3d(ctx, kv, n_embd_head, 1, n_comp);

    return llm_build_deepseek4_rope_tail(ctx, kv, pos, nullptr, n_rot, rope_type,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, false);
}

struct llm_deepseek4_state_pair llm_build_deepseek4_compressor_prefill_state(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * wkv,
        struct ggml_tensor  * wgate,
        struct ggml_tensor  * ape,
        int64_t               n_embd_head,
        int64_t               compress_ratio) {
    GGML_ASSERT(compress_ratio > 0);
    const int64_t n_tokens = x->ne[1];
    const int64_t cutoff = (n_tokens / compress_ratio) * compress_ratio;
    const int64_t remainder = n_tokens - cutoff;
    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t width = coff * n_embd_head;

    GGML_ASSERT(wkv->ne[0] == x->ne[0]);
    GGML_ASSERT(wkv->ne[1] == width);
    GGML_ASSERT(wgate->ne[0] == x->ne[0]);
    GGML_ASSERT(wgate->ne[1] == width);
    GGML_ASSERT(ape->ne[0] == width);
    GGML_ASSERT(ape->ne[1] == compress_ratio);

    struct ggml_tensor * kv    = ggml_mul_mat(ctx, wkv,   x);
    struct ggml_tensor * score = ggml_mul_mat(ctx, wgate, x);
    struct ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);

    if (compress_ratio == 4) {
        struct ggml_tensor * kv_prev    = llm_build_deepseek4_new_filled_2d(ctx, width, compress_ratio, 0.0f);
        struct ggml_tensor * score_prev = llm_build_deepseek4_new_filled_2d(ctx, width, compress_ratio, -INFINITY);

        if (cutoff >= compress_ratio) {
            kv_prev = ggml_view_2d(ctx, kv, width, compress_ratio, kv->nb[1], (cutoff - compress_ratio) * kv->nb[1]);
            score_prev = ggml_view_2d(ctx, score, width, compress_ratio, score->nb[1], (cutoff - compress_ratio) * score->nb[1]);
            score_prev = ggml_add(ctx, score_prev, ape_f);
        }

        struct ggml_tensor * kv_curr    = llm_build_deepseek4_new_filled_2d(ctx, width, compress_ratio, 0.0f);
        struct ggml_tensor * score_curr = llm_build_deepseek4_new_filled_2d(ctx, width, compress_ratio, -INFINITY);

        if (remainder > 0) {
            struct ggml_tensor * kv_rem = ggml_view_2d(ctx, kv, width, remainder, kv->nb[1], cutoff * kv->nb[1]);
            struct ggml_tensor * sc_rem = ggml_view_2d(ctx, score, width, remainder, score->nb[1], cutoff * score->nb[1]);
            sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape_f, width, remainder, ape_f->nb[1], 0));

            if (remainder == compress_ratio) {
                kv_curr = kv_rem;
                score_curr = sc_rem;
            } else {
                kv_curr = ggml_concat(ctx, kv_rem,
                        llm_build_deepseek4_new_filled_2d(ctx, width, compress_ratio - remainder, 0.0f), 1);
                score_curr = ggml_concat(ctx, sc_rem,
                        llm_build_deepseek4_new_filled_2d(ctx, width, compress_ratio - remainder, -INFINITY), 1);
            }
        }

        return {
            ggml_concat(ctx, kv_prev,    kv_curr,    1),
            ggml_concat(ctx, score_prev, score_curr, 1),
        };
    }

    struct ggml_tensor * kv_state    = llm_build_deepseek4_new_filled_2d(ctx, width, compress_ratio, 0.0f);
    struct ggml_tensor * score_state = llm_build_deepseek4_new_filled_2d(ctx, width, compress_ratio, -INFINITY);

    if (remainder > 0) {
        struct ggml_tensor * kv_rem = ggml_view_2d(ctx, kv, width, remainder, kv->nb[1], cutoff * kv->nb[1]);
        struct ggml_tensor * sc_rem = ggml_view_2d(ctx, score, width, remainder, score->nb[1], cutoff * score->nb[1]);
        sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape_f, width, remainder, ape_f->nb[1], 0));

        if (remainder == compress_ratio) {
            kv_state = kv_rem;
            score_state = sc_rem;
        } else {
            kv_state = ggml_concat(ctx, kv_rem,
                    llm_build_deepseek4_new_filled_2d(ctx, width, compress_ratio - remainder, 0.0f), 1);
            score_state = ggml_concat(ctx, sc_rem,
                    llm_build_deepseek4_new_filled_2d(ctx, width, compress_ratio - remainder, -INFINITY), 1);
        }
    }

    return { kv_state, score_state };
}

struct llm_deepseek4_decode_compressor llm_build_deepseek4_compressor_decode(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * prev_kv_state,
        struct ggml_tensor  * prev_score_state,
        struct ggml_tensor  * wkv,
        struct ggml_tensor  * wgate,
        struct ggml_tensor  * ape,
        struct ggml_tensor  * norm,
        int64_t               n_embd_head,
        int64_t               n_rot,
        int64_t               pos,
        int64_t               compress_ratio,
        int                   rope_type,
        int32_t               n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 norm_eps,
        struct ggml_tensor  * comp_pos) {
    GGML_ASSERT(compress_ratio > 0);
    GGML_ASSERT(pos >= 0);

    const int64_t pos_mod = pos % compress_ratio;
    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t width = coff * n_embd_head;
    const int64_t rows = coff * compress_ratio;
    const int64_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = (pos + 1) % compress_ratio == 0;

    GGML_ASSERT(x->ne[1] == 1);
    GGML_ASSERT(wkv->ne[0] == x->ne[0]);
    GGML_ASSERT(wkv->ne[1] == width);
    GGML_ASSERT(wgate->ne[0] == x->ne[0]);
    GGML_ASSERT(wgate->ne[1] == width);
    GGML_ASSERT(ape->ne[0] == width);
    GGML_ASSERT(ape->ne[1] == compress_ratio);
    GGML_ASSERT(norm->ne[0] == n_embd_head);
    GGML_ASSERT(prev_kv_state->type == GGML_TYPE_F32);
    GGML_ASSERT(prev_score_state->type == GGML_TYPE_F32);
    GGML_ASSERT(prev_kv_state->ne[0] == width);
    GGML_ASSERT(prev_kv_state->ne[1] == rows);
    GGML_ASSERT(prev_score_state->ne[0] == width);
    GGML_ASSERT(prev_score_state->ne[1] == rows);
    GGML_ASSERT(row < rows);

    struct ggml_tensor * kv_cur = ggml_mul_mat(ctx, wkv, x);
    struct ggml_tensor * sc_cur = ggml_mul_mat(ctx, wgate, x);
    struct ggml_tensor * ape_f  = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
    sc_cur = ggml_add(ctx, sc_cur, ggml_view_2d(ctx, ape_f, width, 1, ape_f->nb[1], pos_mod*ape_f->nb[1]));

    auto set_row = [&](struct ggml_tensor * dst, struct ggml_tensor * row_src) -> struct ggml_tensor * {
        // cchuter uses a cpy-into-view dependency workaround for multi-GPU
        // scheduling. ik's CPU bring-up can use the structured SET primitive,
        // which preserves the same row-update semantics without relying on a
        // bare destination-view CPY as an intermediate graph output.
        return ggml_set_2d_inplace(ctx, dst, row_src, dst->nb[1], row * dst->nb[1]);
    };

    struct ggml_tensor * kv_state    = set_row(prev_kv_state,    kv_cur);
    struct ggml_tensor * score_state = set_row(prev_score_state, sc_cur);
    struct ggml_tensor * kv_comp = nullptr;

    if (should_compress) {
        struct ggml_tensor * kv_pool = nullptr;
        struct ggml_tensor * score_pool = nullptr;

        if (compress_ratio == 4) {
            struct ggml_tensor * kv_prev = llm_build_deepseek4_view_cols(ctx, kv_state,    n_embd_head, compress_ratio, 0,           0);
            struct ggml_tensor * kv_curr = llm_build_deepseek4_view_cols(ctx, kv_state,    n_embd_head, compress_ratio, n_embd_head, compress_ratio);
            struct ggml_tensor * sc_prev = llm_build_deepseek4_view_cols(ctx, score_state, n_embd_head, compress_ratio, 0,           0);
            struct ggml_tensor * sc_curr = llm_build_deepseek4_view_cols(ctx, score_state, n_embd_head, compress_ratio, n_embd_head, compress_ratio);

            kv_pool    = ggml_concat(ctx, kv_prev, kv_curr, 1);
            score_pool = ggml_concat(ctx, sc_prev, sc_curr, 1);

            struct ggml_tensor * shifted_kv    = llm_build_deepseek4_view_cols(ctx, kv_state,    width, compress_ratio, 0, compress_ratio);
            struct ggml_tensor * shifted_score = llm_build_deepseek4_view_cols(ctx, score_state, width, compress_ratio, 0, compress_ratio);
            kv_state    = ggml_concat(ctx, shifted_kv,    shifted_kv,    1);
            score_state = ggml_concat(ctx, shifted_score, shifted_score, 1);
        } else {
            kv_pool = kv_state;
            score_pool = score_state;
        }

        if (comp_pos == nullptr) {
            comp_pos = llm_build_deepseek4_arange_i32(ctx,
                    pos + 1 - compress_ratio, pos + 2 - compress_ratio);
        }
        GGML_ASSERT(comp_pos->type == GGML_TYPE_I32);
        GGML_ASSERT(comp_pos->ne[0] == 1);
        kv_comp = llm_build_deepseek4_pool_decode_state(ctx, kv_pool, score_pool, norm, comp_pos,
                n_embd_head, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow, norm_eps);
    }

    return { kv_state, score_state, kv_comp };
}

struct ggml_tensor * llm_build_deepseek4_indexer_scores_prefill(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * qr,
        struct ggml_tensor  * index_kv,
        struct ggml_tensor  * wq_b,
        struct ggml_tensor  * wproj,
        struct ggml_tensor  * pos,
        struct ggml_tensor  * causal_mask,
        int64_t               n_index_head,
        int64_t               n_index_head_size,
        int64_t               n_rot,
        int                   rope_type,
        int32_t               n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    const int64_t n_tokens = x->ne[1];
    const int64_t n_comp = index_kv->ne[2];

    GGML_ASSERT(qr->ne[1] == n_tokens);
    GGML_ASSERT(index_kv->ne[0] == n_index_head_size);
    GGML_ASSERT(index_kv->ne[1] == 1);
    GGML_ASSERT(wq_b->ne[1] == n_index_head*n_index_head_size);
    GGML_ASSERT(wproj->ne[1] == n_index_head);
    GGML_ASSERT(pos->ne[0] == n_tokens);
    GGML_ASSERT(causal_mask->ne[0] == n_comp);
    GGML_ASSERT(causal_mask->ne[1] == n_tokens);

    struct ggml_tensor * q = ggml_mul_mat(ctx, wq_b, qr);
    q = ggml_reshape_3d(ctx, q, n_index_head_size, n_index_head, n_tokens);
    q = llm_build_deepseek4_rope_tail(ctx, q, pos, nullptr, n_rot, rope_type,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, false);

    struct ggml_tensor * k = ggml_permute(ctx, index_kv, 0, 2, 1, 3);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);

    struct ggml_tensor * score = ggml_mul_mat(ctx, k, q);
    score = ggml_relu(ctx, score);

    struct ggml_tensor * weights = ggml_mul_mat(ctx, wproj, x);
    const float scale = 1.0f/std::sqrt((float) n_index_head_size*(float) n_index_head);
    weights = llm_build_deepseek4_mul_scalar(ctx, weights, scale);
    weights = ggml_reshape_3d(ctx, weights, 1, n_index_head, n_tokens);
    weights = ggml_permute(ctx, weights, 0, 2, 1, 3);

    score = ggml_mul(ctx, score, weights);
    score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 2, 0, 3));
    score = ggml_sum_rows(ctx, score);
    score = ggml_reshape_2d(ctx, score, n_comp, n_tokens);

    return ggml_add(ctx, score, causal_mask);
}

struct ggml_tensor * llm_build_deepseek4_compressed_mask_from_topk(
        struct ggml_context * ctx,
        struct ggml_tensor  * scores,
        struct ggml_tensor  * topk) {
    const int64_t n_comp   = scores->ne[0];
    const int64_t n_tokens = scores->ne[1];

    GGML_ASSERT(topk->type == GGML_TYPE_I32);
    GGML_ASSERT(topk->ne[1] == n_tokens);

    struct ggml_tensor * scores_rows = ggml_reshape_3d(ctx, scores, 1, n_comp, n_tokens);
    struct ggml_tensor * selected_scores = ggml_get_rows(ctx, scores_rows, topk);
    struct ggml_tensor * valid = ggml_step(ctx, llm_build_deepseek4_add_scalar(ctx, selected_scores, 1.0e30f));
    struct ggml_tensor * values = llm_build_deepseek4_mul_scalar(ctx,
            llm_build_deepseek4_add_scalar(ctx, valid, -1.0f), 1.0e9f);

    struct ggml_tensor * mask = llm_build_deepseek4_new_filled_3d(ctx, 1, n_comp, n_tokens, -INFINITY);
    mask = ggml_set_rows(ctx, mask, values, topk);
    return ggml_reshape_2d(ctx, mask, n_comp, n_tokens);
}
