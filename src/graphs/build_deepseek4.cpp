#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include "build_deepseek4_helpers.h"

#include <algorithm>
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
    if (t == nullptr) {
        LLAMA_LOG_INFO("%s: %-24s = <null>\n", __func__, name);
        return;
    }
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

    auto build_dsv4_mask_input = [&](llama_dsv4_mask_kind kind,
                                     int64_t n0,
                                     int64_t n1,
                                     int64_t n_raw,
                                     int64_t n_comp,
                                     int64_t window,
                                     int64_t ratio,
                                     const char * name,
                                     int il) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n0, n1);
        ggml_set_input(t);
        cb(t, name, il);
        lctx.inp_dsv4_masks.push_back({ t, kind, n_raw, n_comp, window, ratio });
        return t;
    };

    auto build_dsv4_i32_input = [&](int32_t value, const char * name, int il) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_input(t);
        cb(t, name, il);
        lctx.inp_dsv4_i32.push_back({ t, { value } });
        return t;
    };

    auto store_dsv4_cache_rows = [&](ggml_tensor * cache, ggml_tensor * src, int64_t row_start, int64_t n_rows) {
        if (cache == nullptr || src == nullptr || n_rows <= 0) {
            return;
        }
        GGML_ASSERT(row_start >= 0);
        GGML_ASSERT(row_start + n_rows <= cache->ne[1]);
        GGML_ASSERT(src->ne[0] == cache->ne[0]);
        GGML_ASSERT(src->ne[2] >= n_rows);

        ggml_tensor * src_rows = ggml_reshape_2d(ctx0, src, src->ne[0], n_rows);
        ggml_tensor * dst_rows = ggml_view_2d(ctx0, cache,
                cache->ne[0], n_rows,
                cache->nb[1],
                row_start * cache->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, src_rows, dst_rows));
    };

    auto dsv4_state_layout = [](uint32_t ratio, int64_t head_dim) {
        struct layout {
            int64_t width;
            int64_t rows;
            int64_t elems;
        };

        const int64_t coff = ratio == 4 ? 2 : 1;
        const int64_t width = coff * head_dim;
        const int64_t rows  = coff * int64_t(ratio);
        return layout{ width, rows, width * rows };
    };

    auto store_dsv4_state_segment = [&](ggml_tensor * cache, ggml_tensor * src, int64_t elem_offset) {
        if (cache == nullptr || src == nullptr) {
            return;
        }
        GGML_ASSERT(cache->type == GGML_TYPE_F32);
        GGML_ASSERT(src->type == GGML_TYPE_F32);
        GGML_ASSERT(cache->ne[1] == 1);
        const int64_t n_elem = ggml_nelements(src);
        GGML_ASSERT(elem_offset >= 0);
        GGML_ASSERT(elem_offset + n_elem <= cache->ne[0]);

        ggml_tensor * src_flat = ggml_reshape_1d(ctx0, src, n_elem);
        ggml_tensor * dst_flat = ggml_view_1d(ctx0, cache, n_elem, elem_offset * ggml_element_size(cache));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, src_flat, dst_flat));
    };

    auto view_dsv4_state_segment = [&](ggml_tensor * cache, int64_t elem_offset, int64_t width, int64_t rows) {
        GGML_ASSERT(cache != nullptr);
        GGML_ASSERT(cache->type == GGML_TYPE_F32);
        GGML_ASSERT(elem_offset >= 0);
        GGML_ASSERT(elem_offset + width * rows <= cache->ne[0]);
        return ggml_view_2d(ctx0, cache,
                width, rows,
                width * ggml_element_size(cache),
                elem_offset * ggml_element_size(cache));
    };

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);
    cb(inpL, "hc_residual_init", -1);
    dsv4_log_tensor_shape("hc_residual_init", inpL);

    auto build_ffn_update = [&](ggml_tensor * layer_inp, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];
        GGML_ASSERT(layer.hc_ffn_fn != nullptr);
        GGML_ASSERT(layer.hc_ffn_scale != nullptr);
        GGML_ASSERT(layer.hc_ffn_base != nullptr);
        GGML_ASSERT(layer.ffn_norm != nullptr);
        GGML_ASSERT(layer.ffn_gate_inp != nullptr);
        GGML_ASSERT(layer.ffn_up_exps != nullptr);
        GGML_ASSERT(layer.ffn_gate_exps != nullptr);
        GGML_ASSERT(layer.ffn_down_exps != nullptr);
        GGML_ASSERT(layer.ffn_up_shexp != nullptr);
        GGML_ASSERT(layer.ffn_gate_shexp != nullptr);
        GGML_ASSERT(layer.ffn_down_shexp != nullptr);

        ggml_tensor * residual = layer_inp;
        llm_deepseek4_hc_mix mix = llm_build_deepseek4_hc_pre(ctx0, layer_inp,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        ggml_tensor * cur = mix.x;
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

    auto build_compressed_attn_update = [&](
            ggml_tensor             * layer_inp,
            ggml_tensor             * q,
            ggml_tensor             * k_all,
            ggml_tensor             * v_all,
            ggml_tensor             * attn_mask,
            const llm_deepseek4_hc_mix & mix,
            const dsv4_rope_cfg     & rope_cfg,
            int                       il) -> ggml_tensor * {
        const auto & layer = model.layers[il];
        const float kq_scale = 1.0f / std::sqrt(float(n_embd_head_k));

        ggml_tensor * q_attn = ggml_permute(ctx0, q, 0, 2, 1, 3);
        ggml_tensor * k_attn = ggml_permute(ctx0, k_all, 0, 2, 1, 3);
        ggml_tensor * v_attn = ggml_permute(ctx0, v_all, 0, 2, 1, 3);
        if (cparams.flash_attn && k_attn->type == GGML_TYPE_F32) {
            k_attn = ggml_cast(ctx0, k_attn, GGML_TYPE_F16);
        }
        if (cparams.flash_attn && v_attn->type == GGML_TYPE_F32) {
            v_attn = ggml_cast(ctx0, v_attn, GGML_TYPE_F16);
        }
        cb(q_attn, "dsv4_attn_q", il);
        cb(k_attn, "dsv4_attn_k", il);
        cb(v_attn, "dsv4_attn_v", il);
        dsv4_log_tensor_shape("dsv4_attn_q", q_attn);
        dsv4_log_tensor_shape("dsv4_attn_k", k_attn);
        dsv4_log_tensor_shape("dsv4_attn_v", v_attn);

        ggml_tensor * attn_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, attn_mask, GGML_TYPE_F16) : attn_mask;
        ggml_tensor * attn_out = ggml_flash_attn_ext(ctx0, q_attn, k_attn, v_attn,
                attn_mask_cnv, kq_scale, hparams.f_max_alibi_bias, 0.0f);
        ggml_flash_attn_ext_add_sinks(attn_out, layer.attn_sinks);
        if (hparams.n_swa > 0) {
            ((int32_t *) attn_out->op_params)[4] = hparams.n_swa;
        }
        ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
        cb(attn_out, "dsv4_compressed_attn_out", il);
        dsv4_log_tensor_shape("dsv4_compressed_attn_out", attn_out);
        ggml_build_forward_expand(gf, attn_out);

        ggml_tensor * out = ggml_reshape_3d(ctx0, attn_out, n_embd_head_v, n_head, n_tokens);
        out = llm_build_deepseek4_rope_tail(ctx0, out, inp_pos, nullptr, n_rot, rope_type,
                rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, true);
        cb(out, "attn_out_unrope", il);
        dsv4_log_tensor_shape("attn_out_unrope", out);

        out = llm_build_deepseek4_grouped_out(ctx0, out, layer.attn_wo_a, layer.attn_wo_b,
                n_embd_head_v, n_head, n_out_group, n_lora_o, n_tokens);
        cb(out, "attn_out", il);
        dsv4_log_tensor_shape("attn_out", out);

        out = llm_build_deepseek4_hc_expand(ctx0, out, layer_inp, mix.post, mix.comb);
        cb(out, "hc_attn_post", il);
        dsv4_log_tensor_shape("hc_attn_post", out);
        ggml_build_forward_expand(gf, out);
        return out;
    };

    auto build_local_layer = [&](ggml_tensor * layer_inp, int il) -> ggml_tensor * {
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
        GGML_ASSERT(layer.attn_wo_a != nullptr);
        GGML_ASSERT(layer.attn_wo_b != nullptr);

        const uint32_t compress_ratio = hparams.attn_compress_ratio[il];
        GGML_ASSERT(compress_ratio == 0);
        const dsv4_rope_cfg rope_cfg = dsv4_make_rope_cfg(hparams, cparams, compress_ratio);

        LLAMA_LOG_INFO("%s: DeepSeek4 local graph slice: layer=%d n_embd=%" PRId64
                " n_hc=%" PRId64 " n_tokens=%d sinkhorn_iters=%u hc_eps=%.9g\n",
                __func__, il, n_embd, n_hc, n_tokens, hparams.hc_sinkhorn_iters, hparams.hc_eps);

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

        ggml_tensor * k_cache = ggml_view_3d(ctx0, kv_self.k_l[il],
                n_embd_head_k, 1, n_kv,
                kv_self.k_l[il]->nb[1],
                kv_self.k_l[il]->nb[1],
                0);
        cb(k_cache, "dsv4_local_k_cache", il);
        dsv4_log_tensor_shape("dsv4_local_k_cache", k_cache);

        layer_inp = build_compressed_attn_update(layer_inp, q, k_cache, k_cache, KQ_mask_swa, mix, rope_cfg, il);
        return build_ffn_update(layer_inp, il);
    };

    auto build_compressed_prefix = [&](ggml_tensor * layer_inp, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];
        const uint32_t compress_ratio = hparams.attn_compress_ratio[il];
        GGML_ASSERT(compress_ratio == 4 || compress_ratio == 128);
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
        if (compress_ratio == 4) {
            GGML_ASSERT(layer.indexer_attn_q_b != nullptr);
            GGML_ASSERT(layer.indexer_proj != nullptr);
            GGML_ASSERT(layer.indexer_compressor_kv != nullptr);
            GGML_ASSERT(layer.indexer_compressor_gate != nullptr);
            GGML_ASSERT(layer.indexer_compressor_ape != nullptr);
            GGML_ASSERT(layer.indexer_compressor_norm != nullptr);
        }

        const dsv4_rope_cfg rope_cfg = dsv4_make_rope_cfg(hparams, cparams, compress_ratio);
        const float kq_scale = 1.0f / std::sqrt(float(n_embd_head_k));
        const llama_pos first_pos = batch.pos ? batch.pos[0] : batch.all_pos_0;
        const bool is_prefill = batch.pos == nullptr || first_pos == 0;

        LLAMA_LOG_INFO("%s: DeepSeek4 compressed graph prefix: layer=%d ratio=%u n_embd=%" PRId64
                " n_hc=%" PRId64 " n_tokens=%d first_pos=%d is_prefill=%d\n",
                __func__, il, compress_ratio, n_embd, n_hc, n_tokens, first_pos, int(is_prefill));

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

        if (kv_self.k_l[il] != nullptr && !kv_self.v_l.empty() && kv_self.v_l[il] != nullptr) {
            // Compressed layers still attend over the local/SWA KV stream during
            // decode. Store the post-FP8 KV activation now, matching cchuter's
            // raw/local cache update before compressed-cache composition.
            dsv4_log_tensor_shape("kv_cache_k_compressed_local", kv_self.k_l[il]);
            dsv4_log_tensor_shape("kv_cache_v_compressed_local", kv_self.v_l[il]);
            llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, kv, kv, n_tokens, kv_head, cb, il);
        }

        if (is_prefill && il < (int) kv_self.dsv4_layers.size()) {
            auto & dsv4_cache = kv_self.dsv4_layers[il];
            if (dsv4_cache.kv_state != nullptr && dsv4_cache.score_state != nullptr) {
                const auto attn_layout = dsv4_state_layout(compress_ratio, n_embd_head_k);
                GGML_ASSERT(attn_layout.elems <= hparams.dsv4_state_size);

                llm_deepseek4_state_pair attn_state = llm_build_deepseek4_compressor_prefill_state(ctx0,
                        cur,
                        layer.attn_compressor_kv,
                        layer.attn_compressor_gate,
                        layer.attn_compressor_ape,
                        n_embd_head_k,
                        compress_ratio);
                cb(attn_state.kv,    "dsv4_attn_kv_state_prefill",    il);
                cb(attn_state.score, "dsv4_attn_score_state_prefill", il);
                dsv4_log_tensor_shape("dsv4_attn_kv_state_prefill",    attn_state.kv);
                dsv4_log_tensor_shape("dsv4_attn_score_state_prefill", attn_state.score);
                store_dsv4_state_segment(dsv4_cache.kv_state,    attn_state.kv,    0);
                store_dsv4_state_segment(dsv4_cache.score_state, attn_state.score, 0);

                if (compress_ratio == 4) {
                    const auto index_layout = dsv4_state_layout(compress_ratio, hparams.indexer_head_size);
                    GGML_ASSERT(attn_layout.elems + index_layout.elems <= hparams.dsv4_state_size);

                    llm_deepseek4_state_pair index_state = llm_build_deepseek4_compressor_prefill_state(ctx0,
                            cur,
                            layer.indexer_compressor_kv,
                            layer.indexer_compressor_gate,
                            layer.indexer_compressor_ape,
                            hparams.indexer_head_size,
                            compress_ratio);
                    cb(index_state.kv,    "dsv4_index_kv_state_prefill",    il);
                    cb(index_state.score, "dsv4_index_score_state_prefill", il);
                    dsv4_log_tensor_shape("dsv4_index_kv_state_prefill",    index_state.kv);
                    dsv4_log_tensor_shape("dsv4_index_score_state_prefill", index_state.score);
                    store_dsv4_state_segment(dsv4_cache.kv_state,    index_state.kv,    attn_layout.elems);
                    store_dsv4_state_segment(dsv4_cache.score_state, index_state.score, attn_layout.elems);
                }
            }
        }

        if (!is_prefill && il < (int) kv_self.dsv4_layers.size()) {
            auto & dsv4_cache = kv_self.dsv4_layers[il];
            if (dsv4_cache.kv_state != nullptr && dsv4_cache.score_state != nullptr) {
                const int64_t n_comp_before  = first_pos / compress_ratio;
                const int64_t n_comp_visible = (first_pos + 1) / compress_ratio;
                GGML_ASSERT(n_comp_visible <= dsv4_cache.n_comp);

                const auto attn_layout = dsv4_state_layout(compress_ratio, n_embd_head_k);
                ggml_tensor * comp_pos = nullptr;
                if ((first_pos + 1) % compress_ratio == 0) {
                    comp_pos = build_dsv4_i32_input(int32_t(first_pos + 1 - compress_ratio), "dsv4_decode_comp_pos", il);
                    dsv4_log_tensor_shape("dsv4_decode_comp_pos", comp_pos);
                }
                ggml_tensor * prev_attn_kv_state = view_dsv4_state_segment(
                        dsv4_cache.kv_state, 0, attn_layout.width, attn_layout.rows);
                ggml_tensor * prev_attn_score_state = view_dsv4_state_segment(
                        dsv4_cache.score_state, 0, attn_layout.width, attn_layout.rows);

                llm_deepseek4_decode_compressor dec = llm_build_deepseek4_compressor_decode(ctx0,
                        cur,
                        prev_attn_kv_state,
                        prev_attn_score_state,
                        layer.attn_compressor_kv,
                        layer.attn_compressor_gate,
                        layer.attn_compressor_ape,
                        layer.attn_compressor_norm,
                        n_embd_head_k,
                        n_rot,
                        first_pos,
                        compress_ratio,
                        rope_type,
                        rope_cfg.n_ctx_orig,
                        rope_cfg.freq_base,
                        rope_cfg.freq_scale,
                        rope_cfg.ext_factor,
                        rope_cfg.attn_factor,
                        rope_cfg.beta_fast,
                        rope_cfg.beta_slow,
                        norm_rms_eps,
                        comp_pos);
                cb(dec.kv_state,    "dsv4_attn_kv_state_decode",    il);
                cb(dec.score_state, "dsv4_attn_score_state_decode", il);
                dsv4_log_tensor_shape("dsv4_attn_kv_state_decode",    dec.kv_state);
                dsv4_log_tensor_shape("dsv4_attn_score_state_decode", dec.score_state);
                store_dsv4_state_segment(dsv4_cache.kv_state,    dec.kv_state,    0);
                store_dsv4_state_segment(dsv4_cache.score_state, dec.score_state, 0);

                if (dec.kv_comp != nullptr) {
                    GGML_ASSERT(n_comp_visible > n_comp_before);
                    ggml_tensor * kv_comp = ggml_dsv4_fp8_kv_quantize(ctx0, dec.kv_comp, n_rot);
                    cb(kv_comp, "dsv4_attn_kv_comp_decode", il);
                    dsv4_log_tensor_shape("dsv4_attn_kv_comp_decode", kv_comp);
                    store_dsv4_cache_rows(dsv4_cache.attn_k, kv_comp, n_comp_before, n_comp_visible - n_comp_before);
                }

                if (compress_ratio == 4) {
                    const auto index_layout = dsv4_state_layout(compress_ratio, hparams.indexer_head_size);
                    GGML_ASSERT(attn_layout.elems + index_layout.elems <= hparams.dsv4_state_size);

                    ggml_tensor * prev_index_kv_state = view_dsv4_state_segment(
                            dsv4_cache.kv_state, attn_layout.elems, index_layout.width, index_layout.rows);
                    ggml_tensor * prev_index_score_state = view_dsv4_state_segment(
                            dsv4_cache.score_state, attn_layout.elems, index_layout.width, index_layout.rows);

                    llm_deepseek4_decode_compressor index_dec = llm_build_deepseek4_compressor_decode(ctx0,
                            cur,
                            prev_index_kv_state,
                            prev_index_score_state,
                            layer.indexer_compressor_kv,
                            layer.indexer_compressor_gate,
                            layer.indexer_compressor_ape,
                            layer.indexer_compressor_norm,
                            hparams.indexer_head_size,
                            n_rot,
                            first_pos,
                            compress_ratio,
                            rope_type,
                            rope_cfg.n_ctx_orig,
                            rope_cfg.freq_base,
                            rope_cfg.freq_scale,
                            rope_cfg.ext_factor,
                            rope_cfg.attn_factor,
                            rope_cfg.beta_fast,
                            rope_cfg.beta_slow,
                            norm_rms_eps,
                            comp_pos);
                    cb(index_dec.kv_state,    "dsv4_index_kv_state_decode",    il);
                    cb(index_dec.score_state, "dsv4_index_score_state_decode", il);
                    dsv4_log_tensor_shape("dsv4_index_kv_state_decode",    index_dec.kv_state);
                    dsv4_log_tensor_shape("dsv4_index_score_state_decode", index_dec.score_state);
                    store_dsv4_state_segment(dsv4_cache.kv_state,    index_dec.kv_state,    attn_layout.elems);
                    store_dsv4_state_segment(dsv4_cache.score_state, index_dec.score_state, attn_layout.elems);

                    if (index_dec.kv_comp != nullptr) {
                        GGML_ASSERT(n_comp_visible > n_comp_before);
                        cb(index_dec.kv_comp, "dsv4_index_kv_comp_decode", il);
                        dsv4_log_tensor_shape("dsv4_index_kv_comp_decode", index_dec.kv_comp);
                        store_dsv4_cache_rows(dsv4_cache.index_k, index_dec.kv_comp, n_comp_before, n_comp_visible - n_comp_before);
                    }
                }

                LLAMA_LOG_INFO("%s: DeepSeek4 decode compressor state update: layer=%d pos=%d ratio=%u n_comp_before=%" PRId64
                        " n_comp_visible=%" PRId64 " n_comp_cache=%u\n",
                        __func__, il, first_pos, compress_ratio, n_comp_before, n_comp_visible, dsv4_cache.n_comp);

                if (n_comp_visible == 0) {
                    ggml_tensor * attn_out = llm_build_kv(ctx0, lctx, kv_self, gf,
                            nullptr, nullptr,
                            nullptr, nullptr,
                            q, KQ_mask_swa,
                            n_tokens, kv_head, n_kv, kq_scale, cb, il, layer.attn_sinks, hparams.n_swa);
                    cb(attn_out, "dsv4_decode_raw_attn_out", il);
                    dsv4_log_tensor_shape("dsv4_decode_raw_attn_out", attn_out);

                    ggml_tensor * out = ggml_reshape_3d(ctx0, attn_out, n_embd_head_v, n_head, n_tokens);
                    out = llm_build_deepseek4_rope_tail(ctx0, out, inp_pos, nullptr, n_rot, rope_type,
                            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, true);
                    cb(out, "attn_out_unrope", il);
                    dsv4_log_tensor_shape("attn_out_unrope", out);

                    out = llm_build_deepseek4_grouped_out(ctx0, out, layer.attn_wo_a, layer.attn_wo_b,
                            n_embd_head_v, n_head, n_out_group, n_lora_o, n_tokens);
                    cb(out, "attn_out", il);
                    dsv4_log_tensor_shape("attn_out", out);

                    out = llm_build_deepseek4_hc_expand(ctx0, out, layer_inp, mix.post, mix.comb);
                    cb(out, "hc_attn_post", il);
                    dsv4_log_tensor_shape("hc_attn_post", out);
                    ggml_build_forward_expand(gf, out);
                    return out;
                }

                GGML_ASSERT(dsv4_cache.attn_k != nullptr);
                GGML_ASSERT(kv_self.k_l[il] != nullptr);
                GGML_ASSERT(hparams.n_head_kv(il) == 1);

                ggml_tensor * k_raw = ggml_view_3d(ctx0, kv_self.k_l[il],
                        n_embd_head_k, 1, n_kv,
                        kv_self.k_l[il]->nb[1],
                        kv_self.k_l[il]->nb[1],
                        0);
                cb(k_raw, "dsv4_decode_k_raw", il);
                dsv4_log_tensor_shape("dsv4_decode_k_raw", k_raw);

                ggml_tensor * kv_comp_cache = ggml_view_3d(ctx0, dsv4_cache.attn_k,
                        n_embd_head_k, 1, n_comp_visible,
                        dsv4_cache.attn_k->nb[1],
                        dsv4_cache.attn_k->nb[1],
                        0);
                cb(kv_comp_cache, "dsv4_decode_kv_comp_cache", il);
                dsv4_log_tensor_shape("dsv4_decode_kv_comp_cache", kv_comp_cache);

                // GGML only executes non-F32 concat along dim 0. Promote the F16
                // cache views for the dim-2 append; flash attention casts the
                // combined K/V back to F16 below, so cached values are preserved.
                ggml_tensor * k_raw_cat = k_raw->type == GGML_TYPE_F32 ? k_raw : ggml_cast(ctx0, k_raw, GGML_TYPE_F32);
                ggml_tensor * kv_comp_cat = kv_comp_cache->type == GGML_TYPE_F32 ? kv_comp_cache : ggml_cast(ctx0, kv_comp_cache, GGML_TYPE_F32);
                ggml_tensor * k_all = ggml_concat(ctx0, k_raw_cat, kv_comp_cat, 2);
                ggml_tensor * v_all = k_all;
                cb(k_all, "dsv4_decode_k_all", il);
                dsv4_log_tensor_shape("dsv4_decode_k_all", k_all);

                const int64_t n_tokens_attn = cparams.flash_attn ? GGML_PAD(n_tokens, GGML_KQ_MASK_PAD) : n_tokens;
                ggml_tensor * raw_mask = ggml_view_2d(ctx0, KQ_mask_swa,
                        n_kv, n_tokens_attn,
                        KQ_mask_swa->nb[1],
                        0);
                raw_mask = raw_mask->type == GGML_TYPE_F32 ? raw_mask : ggml_cast(ctx0, raw_mask, GGML_TYPE_F32);
                cb(raw_mask, "dsv4_decode_raw_mask", il);
                dsv4_log_tensor_shape("dsv4_decode_raw_mask", raw_mask);

                ggml_tensor * comp_mask = nullptr;
                if (compress_ratio == 4) {
                    if (n_comp_visible <= hparams.indexer_top_k) {
                        comp_mask = build_dsv4_mask_input(
                                llama_dsv4_mask_kind::COMPRESS_CAUSAL,
                                n_comp_visible, n_tokens_attn,
                                0, n_comp_visible, 0, compress_ratio,
                                "dsv4_decode_attn_compress_mask", il);
                    } else {
                        GGML_ASSERT(dsv4_cache.index_k != nullptr);
                        if (n_tokens != 1) {
                            throw std::runtime_error("DeepSeek V4 sparse multi-token decode indexer mask not implemented yet");
                        }
                        LLAMA_LOG_INFO("%s: DeepSeek4 decode sparse indexer mask: layer=%d ratio=%u n_comp_visible=%" PRId64
                                " indexer_top_k=%u\n",
                                __func__, il, compress_ratio, n_comp_visible, hparams.indexer_top_k);

                        ggml_tensor * index_cache = ggml_view_3d(ctx0, dsv4_cache.index_k,
                                hparams.indexer_head_size, 1, n_comp_visible,
                                dsv4_cache.index_k->nb[1],
                                dsv4_cache.index_k->nb[1],
                                0);
                        cb(index_cache, "dsv4_decode_index_cache", il);
                        dsv4_log_tensor_shape("dsv4_decode_index_cache", index_cache);

                        index_cache = ggml_reshape_2d(ctx0, index_cache, hparams.indexer_head_size, n_comp_visible);
                        cb(index_cache, "dsv4_decode_index_cache_2d", il);
                        dsv4_log_tensor_shape("dsv4_decode_index_cache_2d", index_cache);

                        ggml_tensor * index_scores = llm_build_deepseek4_indexer_scores_decode(ctx0,
                                cur,
                                qr,
                                index_cache,
                                layer.indexer_attn_q_b,
                                layer.indexer_proj,
                                inp_pos,
                                hparams.indexer_n_head,
                                hparams.indexer_head_size,
                                n_comp_visible,
                                n_rot,
                                rope_type,
                                rope_cfg.n_ctx_orig,
                                rope_cfg.freq_base,
                                rope_cfg.freq_scale,
                                rope_cfg.ext_factor,
                                rope_cfg.attn_factor,
                                rope_cfg.beta_fast,
                                rope_cfg.beta_slow);
                        cb(index_scores, "indexer_scores", il);
                        dsv4_log_tensor_shape("indexer_scores", index_scores);

                        const int top_k = std::min<int64_t>(hparams.indexer_top_k, n_comp_visible);
                        ggml_tensor * sorted = ggml_argsort(ctx0, index_scores, GGML_SORT_ORDER_DESC);
                        cb(sorted, "indexer_argsort", il);
                        dsv4_log_tensor_shape("indexer_argsort", sorted);
                        ggml_tensor * topk = ggml_view_2d(ctx0, sorted,
                                top_k, n_tokens,
                                sorted->nb[1], 0);
                        cb(topk, "indexer_topk", il);
                        dsv4_log_tensor_shape("indexer_topk", topk);
                        ggml_build_forward_expand(gf, topk);

                        comp_mask = llm_build_deepseek4_compressed_mask_from_topk(ctx0, index_scores, topk);
                    }
                } else {
                    comp_mask = build_dsv4_mask_input(
                            llama_dsv4_mask_kind::COMPRESS_CAUSAL,
                            n_comp_visible, n_tokens_attn,
                            0, n_comp_visible, 0, compress_ratio,
                            "dsv4_decode_attn_compress_mask", il);
                }
                cb(comp_mask, "dsv4_decode_attn_compress_mask", il);
                dsv4_log_tensor_shape("dsv4_decode_attn_compress_mask", comp_mask);

                ggml_tensor * attn_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
                cb(attn_mask, "dsv4_decode_attn_mask", il);
                dsv4_log_tensor_shape("dsv4_decode_attn_mask", attn_mask);

                return build_compressed_attn_update(layer_inp, q, k_all, v_all, attn_mask, mix, rope_cfg, il);
            } else {
                LLAMA_LOG_INFO("%s: DeepSeek4 decode cache state unavailable: layer=%d pos=%d ratio=%u nextn_tail=%d\n",
                        __func__, il, first_pos, compress_ratio, int(il >= int(hparams.n_layer - hparams.nextn_predict_layers)));
                throw std::runtime_error("DeepSeek V4 decode NextN/tail cache path not implemented yet");
            }

            throw std::runtime_error("DeepSeek V4 decode compressed cache replay not implemented yet");
        }

        const int64_t n_comp = n_tokens / compress_ratio;
        if (is_prefill && n_comp > 0) {
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
            if (il < (int) kv_self.dsv4_layers.size()) {
                store_dsv4_cache_rows(kv_self.dsv4_layers[il].attn_k, kv_comp, 0, n_comp);
            }

            ggml_tensor * k_all = ggml_concat(ctx0, kv, kv_comp, 2);
            ggml_tensor * v_all = k_all;
            cb(k_all, "dsv4_attn_k_all", il);
            dsv4_log_tensor_shape("dsv4_attn_k_all", k_all);

            if (compress_ratio == 4) {
                ggml_tensor * index_kv = llm_build_deepseek4_compressor_prefill(ctx0,
                        cur,
                        layer.indexer_compressor_kv,
                        layer.indexer_compressor_gate,
                        layer.indexer_compressor_ape,
                        layer.indexer_compressor_norm,
                        comp_pos,
                        hparams.indexer_head_size,
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
                cb(index_kv, "indexer_KVcompress", il);
                dsv4_log_tensor_shape("indexer_KVcompress", index_kv);
                ggml_build_forward_expand(gf, index_kv);
                if (il < (int) kv_self.dsv4_layers.size()) {
                    store_dsv4_cache_rows(kv_self.dsv4_layers[il].index_k, index_kv, 0, n_comp);
                }

                ggml_tensor * index_mask = build_dsv4_mask_input(
                        llama_dsv4_mask_kind::COMPRESS_CAUSAL,
                        n_comp, n_tokens, 0, n_comp, 0, compress_ratio,
                        "dsv4_indexer_causal_mask", il);
                dsv4_log_tensor_shape("dsv4_indexer_causal_mask", index_mask);

                ggml_tensor * index_scores = llm_build_deepseek4_indexer_scores_prefill(ctx0,
                        cur,
                        qr,
                        index_kv,
                        layer.indexer_attn_q_b,
                        layer.indexer_proj,
                        inp_pos,
                        index_mask,
                        hparams.indexer_n_head,
                        hparams.indexer_head_size,
                        n_rot,
                        rope_type,
                        rope_cfg.n_ctx_orig,
                        rope_cfg.freq_base,
                        rope_cfg.freq_scale,
                        rope_cfg.ext_factor,
                        rope_cfg.attn_factor,
                        rope_cfg.beta_fast,
                        rope_cfg.beta_slow);
                cb(index_scores, "indexer_scores", il);
                dsv4_log_tensor_shape("indexer_scores", index_scores);

                const int top_k = std::min<int64_t>(hparams.indexer_top_k, n_comp);
                ggml_tensor * sorted = ggml_argsort(ctx0, index_scores, GGML_SORT_ORDER_DESC);
                cb(sorted, "indexer_argsort", il);
                dsv4_log_tensor_shape("indexer_argsort", sorted);
                ggml_tensor * topk = ggml_view_2d(ctx0, sorted,
                        top_k, n_tokens,
                        sorted->nb[1], 0);
                cb(topk, "indexer_topk", il);
                dsv4_log_tensor_shape("indexer_topk", topk);
                ggml_build_forward_expand(gf, topk);

                ggml_tensor * comp_mask = llm_build_deepseek4_compressed_mask_from_topk(ctx0, index_scores, topk);
                cb(comp_mask, "dsv4_attn_compress_mask", il);
                dsv4_log_tensor_shape("dsv4_attn_compress_mask", comp_mask);
                ggml_build_forward_expand(gf, comp_mask);

                ggml_tensor * raw_mask = build_dsv4_mask_input(
                        llama_dsv4_mask_kind::RAW_WINDOW,
                        n_tokens, n_tokens, n_tokens, n_comp, hparams.n_swa, compress_ratio,
                        "dsv4_attn_raw_window_mask", il);
                dsv4_log_tensor_shape("dsv4_attn_raw_window_mask", raw_mask);
                ggml_tensor * attn_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
                cb(attn_mask, "dsv4_attn_mask", il);
                dsv4_log_tensor_shape("dsv4_attn_mask", attn_mask);

                return build_compressed_attn_update(layer_inp, q, k_all, v_all, attn_mask, mix, rope_cfg, il);
            } else {
                const int64_t n_tokens_attn = cparams.flash_attn ? GGML_PAD(n_tokens, GGML_KQ_MASK_PAD) : n_tokens;
                ggml_tensor * attn_mask = build_dsv4_mask_input(
                        llama_dsv4_mask_kind::ATTN_STATIC,
                        n_tokens + n_comp, n_tokens_attn, n_tokens, n_comp, hparams.n_swa, compress_ratio,
                        "dsv4_attn_static_mask", il);
                dsv4_log_tensor_shape("dsv4_attn_static_mask", attn_mask);

                return build_compressed_attn_update(layer_inp, q, k_all, v_all, attn_mask, mix, rope_cfg, il);
            }
        }

        if (is_prefill) {
            const int64_t n_tokens_attn = cparams.flash_attn ? GGML_PAD(n_tokens, GGML_KQ_MASK_PAD) : n_tokens;
            ggml_tensor * raw_mask = build_dsv4_mask_input(
                    llama_dsv4_mask_kind::RAW_WINDOW,
                    n_tokens, n_tokens_attn, n_tokens, 0, hparams.n_swa, compress_ratio,
                    "dsv4_attn_raw_window_mask", il);
            dsv4_log_tensor_shape("dsv4_attn_raw_window_mask", raw_mask);

            return build_compressed_attn_update(layer_inp, q, kv, kv, raw_mask, mix, rope_cfg, il);
        }

        throw std::runtime_error("DeepSeek V4 compressed attention path produced no output");
    };

    for (int il = 0; il < n_layer; ++il) {
        const uint32_t compress_ratio = hparams.attn_compress_ratio[il];
        if (compress_ratio != 0) {
            LLAMA_LOG_INFO("%s: DeepSeek4 graph slice: reached compressed layer %d, ratio=%u\n",
                    __func__, il, compress_ratio);
            inpL = build_compressed_prefix(inpL, il);
            inpL = build_ffn_update(inpL, il);
        } else {
            inpL = build_local_layer(inpL, il);
        }
    }

    (void) n_lora_q;

    if (n_tokens > 1) {
        ggml_tensor * inp_out_ids = build_inp_out_ids();
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd * n_hc, n_tokens);
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_outputs);
        cb(inpL, "result_gather_hc", -1);
        dsv4_log_tensor_shape("result_gather_hc", inpL);
    }

    ggml_tensor * cur = llm_build_deepseek4_hc_head(ctx0, inpL,
            model.output_hc_fn, model.output_hc_scale, model.output_hc_base,
            n_embd, n_hc, n_tokens > 1 ? n_outputs : n_tokens,
            norm_rms_eps, hparams.hc_eps);
    cb(cur, "result_hc", -1);
    dsv4_log_tensor_shape("result_hc", cur);

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, nullptr, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);
    dsv4_log_tensor_shape("result_norm", cur);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    dsv4_log_tensor_shape("result_output", cur);
    ggml_build_forward_expand(gf, cur);

    return gf;
}
