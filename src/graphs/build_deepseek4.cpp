#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include "build_deepseek4_helpers.h"

#include <stdexcept>

static void dsv4_log_tensor_shape(const char * name, const ggml_tensor * t) {
    LLAMA_LOG_INFO("%s: %-24s = [%5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %s\n",
            __func__, name, t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_type_name(t->type));
}

ggml_cgraph * llm_build_context::build_deepseek4() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_hc = hparams.n_hc;
    GGML_ASSERT(n_hc > 0);
    GGML_ASSERT(n_layer > 0);

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
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

    throw std::runtime_error("DeepSeek V4 attention projection graph segment not implemented yet");

    return gf;
}
