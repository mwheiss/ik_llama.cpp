#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include "build_deepseek4_helpers.h"

#include <cmath>
#include <stdexcept>

struct dsv4_rope_cfg {
    int32_t n_ctx_orig;
    float   freq_base;
    float   freq_scale;
    float   ext_factor;
    float   attn_factor;
    float   beta_fast;
    float   beta_slow;
};

static dsv4_rope_cfg dsv4_make_rope_cfg(
        const llama_hparams & hparams,
        const llama_cparams  & cparams,
        uint32_t              compress_ratio) {
    if (compress_ratio == 0) {
        return {
            0,
            hparams.rope_freq_base_train,
            1.0f,
            0.0f,
            1.0f,
            cparams.yarn_beta_fast,
            cparams.yarn_beta_slow,
        };
    }

    float attn_factor = 1.0f;
    if (cparams.yarn_ext_factor != 0.0f && cparams.rope_freq_scale > 0.0f) {
        // DeepSeek V4 compressed RoPE follows cchuter's YaRN interpolation rule
        // without applying YaRN's magnitude scaling to attention.
        attn_factor /= 1.0f + 0.1f * std::log(1.0f / cparams.rope_freq_scale);
    }

    return {
        (int32_t) cparams.n_ctx_orig_yarn,
        hparams.compress_rope_freq_base > 0.0f ? hparams.compress_rope_freq_base : cparams.rope_freq_base,
        cparams.rope_freq_scale,
        cparams.yarn_ext_factor,
        attn_factor,
        cparams.yarn_beta_fast,
        cparams.yarn_beta_slow,
    };
}

static void dsv4_log_tensor_shape(const char * name, const ggml_tensor * t) {
    LLAMA_LOG_INFO("%s: %-24s = [%5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %s\n",
            __func__, name, t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_type_name(t->type));
}

ggml_cgraph * llm_build_context::build_deepseek4() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_hc = hparams.n_hc;
    const int64_t n_lora_q = hparams.n_lora_q;
    GGML_ASSERT(n_hc > 0);
    GGML_ASSERT(n_lora_q > 0);
    GGML_ASSERT(n_layer > 0);

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    cb(inpL, "inp_embd", -1);
    dsv4_log_tensor_shape("inp_embd", inpL);

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);
    cb(inpL, "hc_residual_init", -1);
    dsv4_log_tensor_shape("hc_residual_init", inpL);

    const int il = 0;
    const auto & layer = model.layers[il];
    GGML_ASSERT(layer.hc_attn_fn != nullptr);
    GGML_ASSERT(layer.hc_attn_scale != nullptr);
    GGML_ASSERT(layer.hc_attn_base != nullptr);
    GGML_ASSERT(layer.attn_norm != nullptr);
    GGML_ASSERT(layer.attn_q_a_norm != nullptr);
    GGML_ASSERT(layer.attn_kv_a_norm != nullptr);
    GGML_ASSERT(layer.wq_a != nullptr);
    GGML_ASSERT(layer.wq_b != nullptr);
    GGML_ASSERT(layer.attn_kv != nullptr);

    const uint32_t compress_ratio = hparams.attn_compress_ratio[il];
    const dsv4_rope_cfg rope_cfg = dsv4_make_rope_cfg(hparams, cparams, compress_ratio);

    LLAMA_LOG_INFO("%s: DeepSeek4 graph slice: layer=%d hc_pre n_embd=%" PRId64
            " n_hc=%" PRId64 " n_tokens=%d sinkhorn_iters=%u hc_eps=%.9g\n",
            __func__, il, n_embd, n_hc, n_tokens, hparams.hc_sinkhorn_iters, hparams.hc_eps);

    llm_deepseek4_hc_mix mix = llm_build_deepseek4_hc_pre(ctx0, inpL,
            layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
            n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);

    cb(mix.x,      "hc_attn_pre",              il);
    cb(mix.mixes,  "hc_attn_pre_mixes",        il);
    cb(mix.pre,    "hc_attn_pre_weights",      il);
    cb(mix.post,   "hc_attn_pre_post_weights", il);
    cb(mix.comb,   "hc_attn_pre_comb",         il);

    dsv4_log_tensor_shape("hc_attn_pre",              mix.x);
    dsv4_log_tensor_shape("hc_attn_pre_mixes",        mix.mixes);
    dsv4_log_tensor_shape("hc_attn_pre_weights",      mix.pre);
    dsv4_log_tensor_shape("hc_attn_pre_post_weights", mix.post);
    dsv4_log_tensor_shape("hc_attn_pre_comb",         mix.comb);

    ggml_tensor * cur = llm_build_norm(ctx0, mix.x, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "attn_norm", il);
    dsv4_log_tensor_shape("attn_norm", cur);

    ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    cb(qr, "q_lora", il);
    dsv4_log_tensor_shape("q_lora", qr);
    qr = llm_build_norm(ctx0, qr, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(qr, "q_lora_norm", il);
    dsv4_log_tensor_shape("q_lora_norm", qr);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
    q = ggml_rms_norm(ctx0, q, norm_rms_eps);
    cb(q, "Qnorm", il);
    dsv4_log_tensor_shape("Qnorm", q);
    q = llm_build_deepseek4_rope_tail(ctx0, q, inp_pos, nullptr, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
    cb(q, "Qcur", il);
    dsv4_log_tensor_shape("Qcur", q);

    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.attn_kv, cur);
    kv = llm_build_norm(ctx0, kv, hparams, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head_k, 1, n_tokens);
    cb(kv, "KVnorm", il);
    dsv4_log_tensor_shape("KVnorm", kv);
    kv = llm_build_deepseek4_rope_tail(ctx0, kv, inp_pos, nullptr, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
    cb(kv, "KVrope", il);
    dsv4_log_tensor_shape("KVrope", kv);
    kv = ggml_dsv4_fp8_kv_quantize(ctx0, kv, n_rot);
    cb(kv, "KVcur", il);
    dsv4_log_tensor_shape("KVcur", kv);

    (void) n_lora_q;

    throw std::runtime_error("DeepSeek V4 KV cache write graph segment not implemented yet");

    return gf;
}
