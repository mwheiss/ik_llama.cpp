#pragma once

#include "ggml.h"

struct llm_deepseek4_hc_mix {
    struct ggml_tensor * x;
    struct ggml_tensor * mixes;
    struct ggml_tensor * pre;
    struct ggml_tensor * post;
    struct ggml_tensor * comb;
};

struct llm_deepseek4_state_pair {
    struct ggml_tensor * kv;
    struct ggml_tensor * score;
};

struct llm_deepseek4_decode_compressor {
    struct ggml_tensor * kv_state;
    struct ggml_tensor * score_state;
    struct ggml_tensor * kv_comp;
};

struct llm_deepseek4_indexer_decode_trace {
    struct ggml_tensor * q_projected;
    struct ggml_tensor * q_rope;
    struct ggml_tensor * k_cache;
    struct ggml_tensor * score_raw;
    struct ggml_tensor * score_relu;
    struct ggml_tensor * weights_projected;
    struct ggml_tensor * weights_scaled;
    struct ggml_tensor * score_weighted;
    struct ggml_tensor * score_sum;
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
        float                 hc_eps);

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
        float                 norm_eps);

struct llm_deepseek4_state_pair llm_build_deepseek4_compressor_prefill_state(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * wkv,
        struct ggml_tensor  * wgate,
        struct ggml_tensor  * ape,
        int64_t               n_embd_head,
        int64_t               compress_ratio);

struct llm_deepseek4_decode_compressor llm_build_deepseek4_compressor_decode_projected(
        struct ggml_context * ctx,
        struct ggml_tensor  * kv_cur,
        struct ggml_tensor  * sc_cur,
        struct ggml_tensor  * prev_kv_state,
        struct ggml_tensor  * prev_score_state,
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
        struct ggml_tensor  * comp_pos = nullptr);

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
        struct ggml_tensor  * comp_pos = nullptr);

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
        float                 beta_slow);

struct ggml_tensor * llm_build_deepseek4_indexer_scores_decode(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * qr,
        struct ggml_tensor  * index_kv,
        struct ggml_tensor  * wq_b,
        struct ggml_tensor  * wproj,
        struct ggml_tensor  * pos,
        int64_t               n_index_head,
        int64_t               n_index_head_size,
        int64_t               n_comp,
        int64_t               n_rot,
        int                   rope_type,
        int32_t               n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        struct llm_deepseek4_indexer_decode_trace * trace = nullptr);

struct ggml_tensor * llm_build_deepseek4_compressed_mask_from_topk(
        struct ggml_context * ctx,
        struct ggml_tensor  * scores,
        struct ggml_tensor  * topk);
