#pragma once

#include "ggml.h"

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

struct ggml_tensor * llm_build_deepseek4_hc_expand(
        struct ggml_context * ctx,
        struct ggml_tensor  * block_out,
        struct ggml_tensor  * residual,
        struct ggml_tensor  * post,
        struct ggml_tensor  * comb);
