#pragma once

#include "ggml.h"

struct llm_deepseek4_hc_mix {
    struct ggml_tensor * x;
    struct ggml_tensor * mixes;
    struct ggml_tensor * pre;
    struct ggml_tensor * post;
    struct ggml_tensor * comb;
};

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
        bool                  inverse);

struct ggml_tensor * llm_build_deepseek4_hc_weighted_sum(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * weights);

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
        float                 hc_eps);

struct ggml_tensor * llm_build_deepseek4_hc_expand(
        struct ggml_context * ctx,
        struct ggml_tensor  * block_out,
        struct ggml_tensor  * residual,
        struct ggml_tensor  * post,
        struct ggml_tensor  * comb);

struct ggml_tensor * llm_build_deepseek4_grouped_out(
        struct ggml_context * ctx,
        struct ggml_tensor  * o,
        struct ggml_tensor  * wo_a,
        struct ggml_tensor  * wo_b,
        int64_t               n_embd_head,
        int64_t               n_head,
        int64_t               n_groups,
        int64_t               o_lora_rank,
        int64_t               n_tokens);
