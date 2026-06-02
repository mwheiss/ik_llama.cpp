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
    const int64_t n_lora_o = hparams.n_lora_o;
    const int64_t n_out_group = hparams.n_attn_out_groups;
    GGML_ASSERT(n_hc > 0);
    GGML_ASSERT(n_lora_q > 0);
    GGML_ASSERT(n_lora_o > 0);
    GGML_ASSERT(n_out_group > 0);
    GGML_ASSERT(n_layer > 0);

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();
    cb(inpL, "inp_embd", -1);
    dsv4_log_tensor_shape("inp_embd", inpL);
    dsv4_log_tensor_shape("KQ_mask_swa", KQ_mask_swa);

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);
    cb(inpL, "hc_residual_init", -1);
    dsv4_log_tensor_shape("hc_residual_init", inpL);

    auto build_local_layer = [&](ggml_tensor * layer_inp, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];
        GGML_ASSERT(layer.hc_attn_fn != nullptr);
        GGML_ASSERT(layer.hc_attn_scale != nullptr);
        GGML_ASSERT(layer.hc_attn_base != nullptr);
        GGML_ASSERT(layer.hc_ffn_fn != nullptr);
        GGML_ASSERT(layer.hc_ffn_scale != nullptr);
        GGML_ASSERT(layer.hc_ffn_base != nullptr);
        GGML_ASSERT(layer.attn_norm != nullptr);
        GGML_ASSERT(layer.attn_q_a_norm != nullptr);
        GGML_ASSERT(layer.attn_kv_a_norm != nullptr);
        GGML_ASSERT(layer.ffn_norm != nullptr);
        GGML_ASSERT(layer.wq_a != nullptr);
        GGML_ASSERT(layer.wq_b != nullptr);
        GGML_ASSERT(layer.attn_kv != nullptr);
        GGML_ASSERT(layer.attn_wo_a != nullptr);
        GGML_ASSERT(layer.attn_wo_b != nullptr);
        GGML_ASSERT(layer.ffn_gate_inp != nullptr);
        GGML_ASSERT(layer.ffn_up_exps != nullptr);
        GGML_ASSERT(layer.ffn_gate_exps != nullptr);
        GGML_ASSERT(layer.ffn_down_exps != nullptr);
        GGML_ASSERT(layer.ffn_up_shexp != nullptr);
        GGML_ASSERT(layer.ffn_gate_shexp != nullptr);
        GGML_ASSERT(layer.ffn_down_shexp != nullptr);

        const uint32_t compress_ratio = hparams.attn_compress_ratio[il];
        GGML_ASSERT(compress_ratio == 0);
        const dsv4_rope_cfg rope_cfg = dsv4_make_rope_cfg(hparams, cparams, compress_ratio);
        const float kq_scale = 1.0f / std::sqrt(float(n_embd_head_k));

        LLAMA_LOG_INFO("%s: DeepSeek4 local graph slice: layer=%d n_embd=%" PRId64
                " n_hc=%" PRId64 " n_tokens=%d sinkhorn_iters=%u hc_eps=%.9g\n",
                __func__, il, n_embd, n_hc, n_tokens, hparams.hc_sinkhorn_iters, hparams.hc_eps);

        ggml_tensor * residual = layer_inp;
        llm_deepseek4_hc_mix mix = llm_build_deepseek4_hc_pre(ctx0, layer_inp,
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

        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, kv);

        // Uncompressed/local layers use the FP8-simulated KV tensor as both
        // the local K and V source before attention, matching cchuter V4.
        dsv4_log_tensor_shape("kv_cache_k_local", kv_self.k_l[il]);
        dsv4_log_tensor_shape("kv_cache_v_local", kv_self.v_l[il]);
        llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, kv, kv, n_tokens, kv_head, cb, il);

        ggml_tensor * attn_out = llm_build_kv(ctx0, lctx, kv_self, gf,
                nullptr, nullptr,
                nullptr, nullptr,
                q, KQ_mask_swa,
                n_tokens, kv_head, n_kv, kq_scale, cb, il, layer.attn_sinks, hparams.n_swa);
        cb(attn_out, "dsv4_local_attn_out", il);
        dsv4_log_tensor_shape("dsv4_local_attn_out", attn_out);

        cur = ggml_reshape_3d(ctx0, attn_out, n_embd_head_v, n_head, n_tokens);
        cur = llm_build_deepseek4_rope_tail(ctx0, cur, inp_pos, nullptr, n_rot, rope_type,
                rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, true);
        cb(cur, "attn_out_unrope", il);
        dsv4_log_tensor_shape("attn_out_unrope", cur);

        cur = llm_build_deepseek4_grouped_out(ctx0, cur, layer.attn_wo_a, layer.attn_wo_b,
                n_embd_head_v, n_head, n_out_group, n_lora_o, n_tokens);
        cb(cur, "attn_out", il);
        dsv4_log_tensor_shape("attn_out", cur);

        layer_inp = llm_build_deepseek4_hc_expand(ctx0, cur, residual, mix.post, mix.comb);
        cb(layer_inp, "hc_attn_post", il);
        dsv4_log_tensor_shape("hc_attn_post", layer_inp);

        residual = layer_inp;
        mix = llm_build_deepseek4_hc_pre(ctx0, layer_inp,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        cur = mix.x;
        cb(cur, "hc_ffn_pre", il);
        cb(mix.mixes, "hc_ffn_pre_mixes", il);
        cb(mix.pre, "hc_ffn_pre_weights", il);
        cb(mix.post, "hc_ffn_pre_post_weights", il);
        cb(mix.comb, "hc_ffn_pre_comb", il);
        dsv4_log_tensor_shape("hc_ffn_pre", cur);
        dsv4_log_tensor_shape("hc_ffn_pre_mixes", mix.mixes);
        dsv4_log_tensor_shape("hc_ffn_pre_weights", mix.pre);
        dsv4_log_tensor_shape("hc_ffn_pre_post_weights", mix.post);
        dsv4_log_tensor_shape("hc_ffn_pre_comb", mix.comb);

        cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);
        dsv4_log_tensor_shape("ffn_norm", cur);

        ggml_tensor * selected = nullptr;
        if ((uint32_t) il < hparams.n_hash_layers && !warmup) {
            GGML_ASSERT(lctx.inp_tokens != nullptr);
            GGML_ASSERT(layer.ffn_gate_tid2eid != nullptr);
            selected = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, lctx.inp_tokens);
            cb(selected, "ffn_moe_hash_topk", il);
            dsv4_log_tensor_shape("ffn_moe_hash_topk", selected);
        }

        ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                true, hparams.expert_weights_scale,
                (llm_expert_gating_func_type) hparams.expert_gating_func,
                cb, il, gf, false, nullptr, nullptr, nullptr, nullptr, selected);
        cb(moe_out, "ffn_moe_out", il);
        dsv4_log_tensor_shape("ffn_moe_out", moe_out);

        ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                layer.ffn_up_shexp,   nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(ffn_shexp, "ffn_shexp", il);
        dsv4_log_tensor_shape("ffn_shexp", ffn_shexp);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
        dsv4_log_tensor_shape("ffn_out", cur);

        layer_inp = llm_build_deepseek4_hc_expand(ctx0, cur, residual, mix.post, mix.comb);
        cb(layer_inp, "hc_ffn_post", il);
        dsv4_log_tensor_shape("hc_ffn_post", layer_inp);

        return layer_inp;
    };

    auto build_compressed_prefix = [&](ggml_tensor * layer_inp, int il) {
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
        GGML_ASSERT(layer.attn_compressor_kv != nullptr);
        GGML_ASSERT(layer.attn_compressor_gate != nullptr);
        GGML_ASSERT(layer.attn_compressor_ape != nullptr);
        GGML_ASSERT(layer.attn_compressor_norm != nullptr);

        const uint32_t compress_ratio = hparams.attn_compress_ratio[il];
        GGML_ASSERT(compress_ratio == 4 || compress_ratio == 128);
        const dsv4_rope_cfg rope_cfg = dsv4_make_rope_cfg(hparams, cparams, compress_ratio);

        LLAMA_LOG_INFO("%s: DeepSeek4 compressed graph prefix: layer=%d ratio=%u n_embd=%" PRId64
                " n_hc=%" PRId64 " n_tokens=%d\n",
                __func__, il, compress_ratio, n_embd, n_hc, n_tokens);

        llm_deepseek4_hc_mix mix = llm_build_deepseek4_hc_pre(ctx0, layer_inp,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);

        ggml_tensor * cur = mix.x;
        cb(cur, "hc_attn_pre", il);
        cb(mix.mixes, "hc_attn_pre_mixes", il);
        cb(mix.pre, "hc_attn_pre_weights", il);
        cb(mix.post, "hc_attn_pre_post_weights", il);
        cb(mix.comb, "hc_attn_pre_comb", il);
        dsv4_log_tensor_shape("hc_attn_pre", cur);
        dsv4_log_tensor_shape("hc_attn_pre_mixes", mix.mixes);
        dsv4_log_tensor_shape("hc_attn_pre_weights", mix.pre);
        dsv4_log_tensor_shape("hc_attn_pre_post_weights", mix.post);
        dsv4_log_tensor_shape("hc_attn_pre_comb", mix.comb);

        cur = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
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

        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, kv);

        const int64_t n_comp = n_tokens / compress_ratio;
        if (n_comp > 0) {
            ggml_tensor * comp_pos = ggml_arange(ctx0, 0.0f, float(n_comp * compress_ratio), float(compress_ratio));
            comp_pos = ggml_cast(ctx0, comp_pos, GGML_TYPE_I32);
            cb(comp_pos, "comp_pos", il);
            dsv4_log_tensor_shape("comp_pos", comp_pos);

            ggml_tensor * kv_comp = llm_build_deepseek4_compressor_prefill(ctx0,
                    cur,
                    layer.attn_compressor_kv,
                    layer.attn_compressor_gate,
                    layer.attn_compressor_ape,
                    layer.attn_compressor_norm,
                    comp_pos,
                    n_embd_head_k,
                    n_rot,
                    compress_ratio,
                    rope_type,
                    rope_cfg.n_ctx_orig,
                    rope_cfg.freq_base,
                    rope_cfg.freq_scale,
                    rope_cfg.ext_factor,
                    rope_cfg.attn_factor,
                    rope_cfg.beta_fast,
                    rope_cfg.beta_slow,
                    norm_rms_eps);
            kv_comp = ggml_dsv4_fp8_kv_quantize(ctx0, kv_comp, n_rot);
            cb(kv_comp, "KVcompress", il);
            dsv4_log_tensor_shape("KVcompress", kv_comp);
            ggml_build_forward_expand(gf, kv_comp);
        }
    };

    for (int il = 0; il < n_layer; ++il) {
        const uint32_t compress_ratio = hparams.attn_compress_ratio[il];
        if (compress_ratio != 0) {
            LLAMA_LOG_INFO("%s: DeepSeek4 graph slice: reached compressed layer %d, ratio=%u\n",
                    __func__, il, compress_ratio);
            build_compressed_prefix(inpL, il);
            throw std::runtime_error("DeepSeek V4 compressed KV cache/indexer graph segment not implemented yet");
        }

        inpL = build_local_layer(inpL, il);
    }

    (void) n_lora_q;

    throw std::runtime_error("DeepSeek V4 next layer graph segment not implemented yet");

    return gf;
}
