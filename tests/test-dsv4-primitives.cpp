#include "ggml.h"
#include "../src/graphs/build_deepseek4_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void graph_compute(ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_cplan plan = ggml_graph_plan(gf, 4);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();

    const ggml_status status = ggml_graph_compute(gf, &plan);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_graph_compute failed: %d\n", status);
        std::abort();
    }
}

static void assert_close(float actual, float expected, const char * what, float eps = 1.0e-5f) {
    const float diff = std::fabs(actual - expected);
    if (diff > eps) {
        fprintf(stderr, "%s: got %.9g, expected %.9g, diff %.9g\n", what, actual, expected, diff);
        std::abort();
    }
}

static float tensor_get_3d(const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2) {
    return ggml_get_f32_nd(t, i0, i1, i2, 0);
}

static float ref_e4m3fn_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0 ? std::ldexp((float) mant, -9) : std::ldexp(1.0f + (float) mant / 8.0f, exp - 7);
}

static float ref_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = std::min(std::fabs(x), 448.0f);

    int best = 0;
    float best_diff = ax;
    for (int i = 1; i < 127; ++i) {
        const float diff = std::fabs(ax - ref_e4m3fn_value(i));
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }

    return sign * ref_e4m3fn_value(best);
}

static std::vector<float> ref_fp8_kv_row(const std::vector<float> & row, int64_t n_rot) {
    const int64_t head_dim = (int64_t) row.size();
    const int64_t n_nope = head_dim - n_rot;
    std::vector<float> out(row);

    for (int64_t off = 0; off < n_nope; off += 64) {
        float amax = 0.0f;
        for (int64_t i = 0; i < 64; ++i) {
            amax = std::max(amax, std::fabs(row[off + i]));
        }

        amax = std::max(amax, 1.0e-4f);
        const float scale = std::ldexp(1.0f, (int) std::ceil(std::log2(amax / 448.0f)));
        for (int64_t i = 0; i < 64; ++i) {
            const float q = std::max(-448.0f, std::min(448.0f, row[off + i] / scale));
            out[off + i] = ref_e4m3fn_dequant(q) * scale;
        }
    }

    return out;
}

static void ref_hc_split_sinkhorn_row(
        const std::vector<float> & mixes,
        const std::vector<float> & scale,
        const std::vector<float> & base,
        int64_t n_hc,
        int sinkhorn_iters,
        float eps,
        std::vector<float> & out) {
    const int64_t mix_hc = (2 + n_hc) * n_hc;
    out.assign(mix_hc, 0.0f);

    for (int64_t i = 0; i < n_hc; ++i) {
        const float z = mixes[i] * scale[0] + base[i];
        out[i] = 1.0f / (1.0f + std::exp(-z)) + eps;
    }

    for (int64_t i = 0; i < n_hc; ++i) {
        const int64_t off = n_hc + i;
        const float z = mixes[off] * scale[1] + base[off];
        out[off] = 2.0f / (1.0f + std::exp(-z));
    }

    std::vector<float> c(n_hc * n_hc);
    for (int64_t dst = 0; dst < n_hc; ++dst) {
        float row_max = -INFINITY;
        for (int64_t src = 0; src < n_hc; ++src) {
            const int64_t idx = src + dst * n_hc;
            const int64_t off = 2 * n_hc + idx;
            const float v = mixes[off] * scale[2] + base[off];
            c[idx] = v;
            row_max = std::max(row_max, v);
        }

        float row_sum = 0.0f;
        for (int64_t src = 0; src < n_hc; ++src) {
            const int64_t idx = src + dst * n_hc;
            c[idx] = std::exp(c[idx] - row_max);
            row_sum += c[idx];
        }

        const float inv_sum = 1.0f / row_sum;
        for (int64_t src = 0; src < n_hc; ++src) {
            const int64_t idx = src + dst * n_hc;
            c[idx] = c[idx] * inv_sum + eps;
        }
    }

    for (int64_t src = 0; src < n_hc; ++src) {
        float sum = 0.0f;
        for (int64_t dst = 0; dst < n_hc; ++dst) {
            sum += c[src + dst * n_hc];
        }
        const float inv_denom = 1.0f / (sum + eps);
        for (int64_t dst = 0; dst < n_hc; ++dst) {
            c[src + dst * n_hc] *= inv_denom;
        }
    }

    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        for (int64_t dst = 0; dst < n_hc; ++dst) {
            float sum = 0.0f;
            for (int64_t src = 0; src < n_hc; ++src) {
                sum += c[src + dst * n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int64_t src = 0; src < n_hc; ++src) {
                c[src + dst * n_hc] *= inv_denom;
            }
        }

        for (int64_t src = 0; src < n_hc; ++src) {
            float sum = 0.0f;
            for (int64_t dst = 0; dst < n_hc; ++dst) {
                sum += c[src + dst * n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int64_t dst = 0; dst < n_hc; ++dst) {
                c[src + dst * n_hc] *= inv_denom;
            }
        }
    }

    for (int64_t i = 0; i < n_hc * n_hc; ++i) {
        out[2 * n_hc + i] = c[i];
    }
}

static void test_hc_split_sinkhorn() {
    ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_hc = 4;
    constexpr int64_t n_tokens = 3;
    constexpr int64_t mix_hc = (2 + n_hc) * n_hc;
    constexpr int sinkhorn_iters = 5;
    constexpr float eps = 1.0e-6f;

    ggml_tensor * mixes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mix_hc, n_tokens);
    ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor * base  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mix_hc);

    ggml_set_f32_1d(scale, 0, 0.50f);
    ggml_set_f32_1d(scale, 1, -0.25f);
    ggml_set_f32_1d(scale, 2, 0.75f);

    for (int64_t i = 0; i < mix_hc; ++i) {
        ggml_set_f32_1d(base, i, 0.01f * (float) ((i % 9) - 4));
        for (int64_t t = 0; t < n_tokens; ++t) {
            ggml_set_f32_nd(mixes, i, t, 0, 0, 0.02f * (float) (i - 7) - 0.03f * (float) t);
        }
    }

    ggml_tensor * out = ggml_dsv4_hc_split_sinkhorn(ctx, mixes, scale, base, n_hc, sinkhorn_iters, eps);
    graph_compute(ctx, out);

    std::vector<float> scale_ref = { 0.50f, -0.25f, 0.75f };
    std::vector<float> base_ref(mix_hc);
    for (int64_t i = 0; i < mix_hc; ++i) {
        base_ref[i] = ggml_get_f32_1d(base, i);
    }

    for (int64_t t = 0; t < n_tokens; ++t) {
        std::vector<float> mixes_ref(mix_hc);
        for (int64_t i = 0; i < mix_hc; ++i) {
            mixes_ref[i] = ggml_get_f32_nd(mixes, i, t, 0, 0);
        }

        std::vector<float> ref;
        ref_hc_split_sinkhorn_row(mixes_ref, scale_ref, base_ref, n_hc, sinkhorn_iters, eps, ref);
        for (int64_t i = 0; i < mix_hc; ++i) {
            assert_close(ggml_get_f32_nd(out, i, t, 0, 0), ref[i], "hc_split_sinkhorn", 2.0e-6f);
        }
    }

    ggml_free(ctx);
}

static void test_fp8_kv_quantize() {
    ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t head_dim = 80;
    constexpr int64_t n_rows = 2;
    constexpr int n_rot = 16;

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, n_rows);
    for (int64_t row = 0; row < n_rows; ++row) {
        for (int64_t i = 0; i < head_dim; ++i) {
            float v = 0.0f;
            if (i < 64) {
                const float sign = (i % 2) == 0 ? 1.0f : -1.0f;
                v = sign * (0.00003f * (float) (i + 1) + 0.125f * (float) (i % 9));
            } else {
                v = 1000.0f + (float) (row * 100 + i);
            }
            if (i == 0)  { v = 0.0f; }
            if (i == 1)  { v = 1.0e-5f; }
            if (i == 2)  { v = -1.0e-5f; }
            if (i == 3)  { v = 448.0f; }
            if (i == 4)  { v = -448.0f; }
            if (i == 5)  { v = 900.0f; }
            if (i == 6)  { v = -900.0f; }
            ggml_set_f32_nd(x, i, row, 0, 0, v);
        }
    }

    ggml_tensor * out = ggml_dsv4_fp8_kv_quantize(ctx, x, n_rot);
    graph_compute(ctx, out);

    for (int64_t row = 0; row < n_rows; ++row) {
        std::vector<float> src(head_dim);
        for (int64_t i = 0; i < head_dim; ++i) {
            src[i] = ggml_get_f32_nd(x, i, row, 0, 0);
        }

        std::vector<float> ref = ref_fp8_kv_row(src, n_rot);
        for (int64_t i = 0; i < head_dim; ++i) {
            assert_close(ggml_get_f32_nd(out, i, row, 0, 0), ref[i], "fp8_kv_quantize", 1.0e-6f);
        }
    }

    ggml_free(ctx);
}

static void test_hc_weighted_sum_helper() {
    ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd = 5;
    constexpr int64_t n_hc = 4;
    constexpr int64_t n_tokens = 3;

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_hc, n_tokens);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, n_tokens);

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t h = 0; h < n_hc; ++h) {
            ggml_set_f32_nd(weights, h, t, 0, 0, 0.1f * (float) (h + 1) - 0.03f * (float) t);
            for (int64_t e = 0; e < n_embd; ++e) {
                const float v = 0.01f * (float) (11*e + 5*h - 7*t);
                ggml_set_f32_nd(x, e, h, t, 0, v);
            }
        }
    }

    ggml_tensor * out = llm_build_deepseek4_hc_weighted_sum(ctx, x, weights);
    graph_compute(ctx, out);

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_embd; ++e) {
            float ref = 0.0f;
            for (int64_t h = 0; h < n_hc; ++h) {
                ref += tensor_get_3d(x, e, h, t) * ggml_get_f32_nd(weights, h, t, 0, 0);
            }
            assert_close(ggml_get_f32_nd(out, e, t, 0, 0), ref, "hc_weighted_sum_helper", 1.0e-6f);
        }
    }

    ggml_free(ctx);
}

static void test_hc_expand_helper() {
    ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd = 4;
    constexpr int64_t n_hc = 3;
    constexpr int64_t n_tokens = 2;

    ggml_tensor * block_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * residual  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_hc, n_tokens);
    ggml_tensor * post      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, n_tokens);
    ggml_tensor * comb      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_hc, n_hc, n_tokens);

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_embd; ++e) {
            ggml_set_f32_nd(block_out, e, t, 0, 0, 0.2f * (float) (e + 1) - 0.05f * (float) t);
        }
        for (int64_t h = 0; h < n_hc; ++h) {
            ggml_set_f32_nd(post, h, t, 0, 0, 0.25f + 0.04f * (float) h - 0.02f * (float) t);
            for (int64_t e = 0; e < n_embd; ++e) {
                ggml_set_f32_nd(residual, e, h, t, 0, -0.1f + 0.03f * (float) e + 0.07f * (float) h + 0.02f * (float) t);
            }
            for (int64_t src = 0; src < n_hc; ++src) {
                // Store source-major values; the helper intentionally reads this
                // as comb[dst, src, t], matching cchuter's transposed contract.
                ggml_set_f32_nd(comb, src, h, t, 0, 0.05f * (float) (src + 1) - 0.04f * (float) h + 0.01f * (float) t);
            }
        }
    }

    ggml_tensor * out = llm_build_deepseek4_hc_expand(ctx, block_out, residual, post, comb);
    graph_compute(ctx, out);

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t dst = 0; dst < n_hc; ++dst) {
            for (int64_t e = 0; e < n_embd; ++e) {
                float ref = ggml_get_f32_nd(block_out, e, t, 0, 0) * ggml_get_f32_nd(post, dst, t, 0, 0);
                for (int64_t src = 0; src < n_hc; ++src) {
                    ref += tensor_get_3d(residual, e, src, t) * tensor_get_3d(comb, dst, src, t);
                }
                assert_close(tensor_get_3d(out, e, dst, t), ref, "hc_expand_helper", 1.0e-6f);
            }
        }
    }

    ggml_free(ctx);
}

static void test_grouped_out_helper() {
    ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd_head = 3;
    constexpr int64_t n_head = 4;
    constexpr int64_t n_groups = 2;
    constexpr int64_t group_heads = n_head / n_groups;
    constexpr int64_t group_dim = n_embd_head * group_heads;
    constexpr int64_t o_lora_rank = 2;
    constexpr int64_t n_tokens = 2;
    constexpr int64_t n_out = 5;

    ggml_tensor * o    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, n_tokens);
    ggml_tensor * wo_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, group_dim, o_lora_rank * n_groups);
    ggml_tensor * wo_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, o_lora_rank * n_groups, n_out);

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t h = 0; h < n_head; ++h) {
            for (int64_t e = 0; e < n_embd_head; ++e) {
                ggml_set_f32_nd(o, e, h, t, 0, 0.03f * (float) e - 0.04f * (float) h + 0.02f * (float) t);
            }
        }
    }
    for (int64_t col = 0; col < o_lora_rank * n_groups; ++col) {
        for (int64_t row = 0; row < group_dim; ++row) {
            ggml_set_f32_nd(wo_a, row, col, 0, 0, 0.01f * (float) ((7*row + 3*col) % 11 - 5));
        }
    }
    for (int64_t col = 0; col < n_out; ++col) {
        for (int64_t row = 0; row < o_lora_rank * n_groups; ++row) {
            ggml_set_f32_nd(wo_b, row, col, 0, 0, 0.02f * (float) ((5*row - 2*col) % 13 - 6));
        }
    }

    ggml_tensor * out = llm_build_deepseek4_grouped_out(ctx, o, wo_a, wo_b,
            n_embd_head, n_head, n_groups, o_lora_rank, n_tokens);
    graph_compute(ctx, out);

    for (int64_t t = 0; t < n_tokens; ++t) {
        std::vector<float> low(o_lora_rank * n_groups, 0.0f);
        for (int64_t g = 0; g < n_groups; ++g) {
            for (int64_t r = 0; r < o_lora_rank; ++r) {
                float acc = 0.0f;
                for (int64_t gh = 0; gh < group_heads; ++gh) {
                    const int64_t h = g * group_heads + gh;
                    for (int64_t e = 0; e < n_embd_head; ++e) {
                        const int64_t d = e + n_embd_head * gh;
                        acc += tensor_get_3d(o, e, h, t) * ggml_get_f32_nd(wo_a, d, r + o_lora_rank * g, 0, 0);
                    }
                }
                low[r + o_lora_rank * g] = acc;
            }
        }
        for (int64_t col = 0; col < n_out; ++col) {
            float ref = 0.0f;
            for (int64_t i = 0; i < o_lora_rank * n_groups; ++i) {
                ref += low[i] * ggml_get_f32_nd(wo_b, i, col, 0, 0);
            }
            assert_close(ggml_get_f32_nd(out, col, t, 0, 0), ref, "grouped_out_helper", 1.0e-6f);
        }
    }

    ggml_free(ctx);
}

static void test_compressor_prefill_ratio4_helper() {
    ggml_init_params params = {
        /* .mem_size   = */ 8 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd = 3;
    constexpr int64_t n_embd_head = 2;
    constexpr int64_t n_tokens = 4;
    constexpr int64_t ratio = 4;
    constexpr int64_t coff = 2;
    constexpr int64_t n_kv = coff * n_embd_head;
    constexpr int64_t n_comp = 1;
    constexpr int64_t n_rot = 2;
    constexpr float norm_eps = 1.0e-6f;

    ggml_tensor * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * wkv   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_kv);
    ggml_tensor * wgate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_kv);
    ggml_tensor * ape   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, ratio);
    ggml_tensor * norm  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd_head);
    ggml_tensor * pos   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_comp);

    ggml_set_i32_1d(pos, 0, 0);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_embd; ++e) {
            ggml_set_f32_nd(x, e, t, 0, 0, 0.10f * (float) (e + 1) - 0.03f * (float) t);
        }
    }
    for (int64_t r = 0; r < n_kv; ++r) {
        for (int64_t e = 0; e < n_embd; ++e) {
            ggml_set_f32_nd(wkv, e, r, 0, 0, 0.04f * (float) (r + 1) - 0.02f * (float) e);
            ggml_set_f32_nd(wgate, e, r, 0, 0, -0.03f * (float) (r + 1) + 0.015f * (float) e);
        }
        for (int64_t t = 0; t < ratio; ++t) {
            ggml_set_f32_nd(ape, r, t, 0, 0, 0.01f * (float) r - 0.02f * (float) t);
        }
    }
    ggml_set_f32_1d(norm, 0, 1.25f);
    ggml_set_f32_1d(norm, 1, 0.75f);

    ggml_tensor * out = llm_build_deepseek4_compressor_prefill(ctx,
            x, wkv, wgate, ape, norm, pos,
            n_embd_head, n_rot, ratio,
            0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, norm_eps);
    graph_compute(ctx, out);

    float pooled[n_embd_head] = {};
    for (int64_t h = 0; h < n_embd_head; ++h) {
        float vals[ratio];
        float scores[ratio];
        float max_score = -INFINITY;
        const int64_t row = n_embd_head + h;
        for (int64_t t = 0; t < ratio; ++t) {
            vals[t] = 0.0f;
            scores[t] = ggml_get_f32_nd(ape, row, t, 0, 0);
            for (int64_t e = 0; e < n_embd; ++e) {
                const float xe = ggml_get_f32_nd(x, e, t, 0, 0);
                vals[t] += xe * ggml_get_f32_nd(wkv, e, row, 0, 0);
                scores[t] += xe * ggml_get_f32_nd(wgate, e, row, 0, 0);
            }
            max_score = std::max(max_score, scores[t]);
        }

        float denom = 0.0f;
        for (float score : scores) {
            denom += std::exp(score - max_score);
        }
        for (int64_t t = 0; t < ratio; ++t) {
            pooled[h] += vals[t] * std::exp(scores[t] - max_score) / denom;
        }
    }

    float rms = 0.0f;
    for (float v : pooled) {
        rms += v * v;
    }
    rms = std::sqrt(rms / (float) n_embd_head + norm_eps);

    for (int64_t h = 0; h < n_embd_head; ++h) {
        const float expected = pooled[h] / rms * ggml_get_f32_1d(norm, h);
        assert_close(tensor_get_3d(out, h, 0, 0), expected, "compressor_prefill_ratio4", 2.0e-6f);
    }

    ggml_free(ctx);
}

static void test_indexer_scores_prefill_helper() {
    ggml_init_params params = {
        /* .mem_size   = */ 8 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd = 3;
    constexpr int64_t q_rank = 2;
    constexpr int64_t n_tokens = 2;
    constexpr int64_t n_comp = 2;
    constexpr int64_t n_index_head = 1;
    constexpr int64_t head_size = 2;
    constexpr int64_t n_rot = 2;

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * qr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, q_rank, n_tokens);
    ggml_tensor * index_kv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_size, 1, n_comp);
    ggml_tensor * wq_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, q_rank, n_index_head * head_size);
    ggml_tensor * wproj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_index_head);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_comp, n_tokens);

    for (int64_t t = 0; t < n_tokens; ++t) {
        ggml_set_i32_1d(pos, (int) t, 0);
        for (int64_t e = 0; e < n_embd; ++e) {
            ggml_set_f32_nd(x, e, t, 0, 0, 0.20f + 0.05f * (float) e - 0.03f * (float) t);
        }
        for (int64_t e = 0; e < q_rank; ++e) {
            ggml_set_f32_nd(qr, e, t, 0, 0, -0.10f + 0.07f * (float) e + 0.02f * (float) t);
        }
        for (int64_t c = 0; c < n_comp; ++c) {
            ggml_set_f32_nd(mask, c, t, 0, 0, c > t ? -INFINITY : 0.0f);
        }
    }
    for (int64_t c = 0; c < n_comp; ++c) {
        for (int64_t h = 0; h < head_size; ++h) {
            ggml_set_f32_nd(index_kv, h, 0, c, 0, 0.15f * (float) (h + 1) - 0.04f * (float) c);
        }
    }
    for (int64_t o = 0; o < n_index_head * head_size; ++o) {
        for (int64_t e = 0; e < q_rank; ++e) {
            ggml_set_f32_nd(wq_b, e, o, 0, 0, 0.03f * (float) (o + 1) + 0.02f * (float) e);
        }
    }
    for (int64_t h = 0; h < n_index_head; ++h) {
        for (int64_t e = 0; e < n_embd; ++e) {
            ggml_set_f32_nd(wproj, e, h, 0, 0, 0.06f * (float) (e + 1));
        }
    }

    ggml_tensor * scores = llm_build_deepseek4_indexer_scores_prefill(ctx,
            x, qr, index_kv, wq_b, wproj, pos, mask,
            n_index_head, head_size, n_rot,
            0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    graph_compute(ctx, scores);

    for (int64_t t = 0; t < n_tokens; ++t) {
        float q[head_size] = {};
        for (int64_t h = 0; h < head_size; ++h) {
            for (int64_t e = 0; e < q_rank; ++e) {
                q[h] += ggml_get_f32_nd(qr, e, t, 0, 0) * ggml_get_f32_nd(wq_b, e, h, 0, 0);
            }
        }

        float weight = 0.0f;
        for (int64_t e = 0; e < n_embd; ++e) {
            weight += ggml_get_f32_nd(x, e, t, 0, 0) * ggml_get_f32_nd(wproj, e, 0, 0, 0);
        }
        weight *= 1.0f / std::sqrt((float) (head_size * n_index_head));

        for (int64_t c = 0; c < n_comp; ++c) {
            float dot = 0.0f;
            for (int64_t h = 0; h < head_size; ++h) {
                dot += ggml_get_f32_nd(index_kv, h, 0, c, 0) * q[h];
            }
            const float base = std::max(0.0f, dot) * weight;
            const float m = ggml_get_f32_nd(mask, c, t, 0, 0);
            const float expected = std::isinf(m) ? -INFINITY : base + m;
            const float actual = ggml_get_f32_nd(scores, c, t, 0, 0);
            if (std::isinf(expected)) {
                if (!std::isinf(actual) || actual > 0.0f) {
                    fprintf(stderr, "indexer_scores_mask: got %.9g, expected -inf\n", actual);
                    std::abort();
                }
            } else {
                assert_close(actual, expected, "indexer_scores_prefill", 1.0e-6f);
            }
        }
    }

    ggml_free(ctx);
}

static void test_compressed_mask_from_topk_helper() {
    ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_comp = 4;
    constexpr int64_t n_tokens = 2;
    constexpr int64_t top_k = 2;

    ggml_tensor * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_comp, n_tokens);
    ggml_tensor * topk = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, n_tokens);

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t c = 0; c < n_comp; ++c) {
            ggml_set_f32_nd(scores, c, t, 0, 0, 0.1f * (float) c);
        }
    }
    ggml_set_i32_nd(topk, 0, 0, 0, 0, 2);
    ggml_set_i32_nd(topk, 1, 0, 0, 0, 0);
    ggml_set_i32_nd(topk, 0, 1, 0, 0, 3);
    ggml_set_i32_nd(topk, 1, 1, 0, 0, 1);
    ggml_set_f32_nd(scores, 1, 1, 0, 0, -INFINITY);

    ggml_tensor * mask = llm_build_deepseek4_compressed_mask_from_topk(ctx, scores, topk);
    graph_compute(ctx, mask);

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t c = 0; c < n_comp; ++c) {
            const float v = ggml_get_f32_nd(mask, c, t, 0, 0);
            const bool selected =
                (t == 0 && (c == 2 || c == 0)) ||
                (t == 1 && (c == 3 || c == 1));
            const bool invalid_selected = t == 1 && c == 1;

            if (invalid_selected) {
                assert_close(v, -1.0e9f, "compressed_mask_invalid_selected", 1.0f);
            } else if (selected) {
                assert_close(v, 0.0f, "compressed_mask_selected", 1.0e-6f);
            } else if (!std::isinf(v) || v > 0.0f) {
                fprintf(stderr, "compressed_mask_unselected: got %.9g, expected -inf\n", v);
                std::abort();
            }
        }
    }

    ggml_free(ctx);
}

static float ref_rope_standard(float x0, float x1, int32_t pos, int64_t pair, int64_t n_rot, bool second) {
    const float theta_scale = std::pow(10000.0f, -2.0f / (float) n_rot);
    const float theta = (float) pos * std::pow(theta_scale, (float) pair);
    const float c = std::cos(theta);
    const float s = std::sin(theta);
    return second ? x0*s + x1*c : x0*c - x1*s;
}

static void test_rope_tail_helper() {
    ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);

    constexpr int64_t n_embd_head = 8;
    constexpr int64_t n_heads = 2;
    constexpr int64_t n_tokens = 3;
    constexpr int64_t n_rot = 4;
    constexpr int64_t n_nope = n_embd_head - n_rot;

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_heads, n_tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);

    const int32_t positions[n_tokens] = { 1, 3, 7 };
    for (int64_t t = 0; t < n_tokens; ++t) {
        ggml_set_i32_1d(pos, (int) t, positions[t]);
        for (int64_t h = 0; h < n_heads; ++h) {
            for (int64_t e = 0; e < n_embd_head; ++e) {
                ggml_set_f32_nd(x, e, h, t, 0, 0.1f * (float) e + 0.03f * (float) h - 0.02f * (float) t);
            }
        }
    }

    ggml_tensor * out = llm_build_deepseek4_rope_tail(ctx, x, pos, nullptr, n_rot,
            0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, false);
    graph_compute(ctx, out);

    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t h = 0; h < n_heads; ++h) {
            for (int64_t e = 0; e < n_nope; ++e) {
                assert_close(tensor_get_3d(out, e, h, t), tensor_get_3d(x, e, h, t), "rope_tail_nope", 1.0e-6f);
            }
            for (int64_t pair = 0; pair < n_rot/2; ++pair) {
                const int64_t e0 = n_nope + 2*pair + 0;
                const int64_t e1 = n_nope + 2*pair + 1;
                const float x0 = tensor_get_3d(x, e0, h, t);
                const float x1 = tensor_get_3d(x, e1, h, t);
                assert_close(tensor_get_3d(out, e0, h, t),
                        ref_rope_standard(x0, x1, positions[t], pair, n_rot, false), "rope_tail_even", 1.0e-6f);
                assert_close(tensor_get_3d(out, e1, h, t),
                        ref_rope_standard(x0, x1, positions[t], pair, n_rot, true), "rope_tail_odd", 1.0e-6f);
            }
        }
    }

    ggml_free(ctx);
}

int main() {
    test_hc_split_sinkhorn();
    test_fp8_kv_quantize();
    test_hc_weighted_sum_helper();
    test_hc_expand_helper();
    test_grouped_out_helper();
    test_compressor_prefill_ratio4_helper();
    test_indexer_scores_prefill_helper();
    test_compressed_mask_from_topk_helper();
    test_rope_tail_helper();
    return 0;
}
