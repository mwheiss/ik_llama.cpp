#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// Scalar softmax attention is the oracle, including for dimensions without an
// IQK kernel and worker counts that leave some query groups empty.
static bool check(int d, int heads, int queries, int keys, int threads, bool masked) {
    auto * ctx = ggml_init({16*1024*1024, nullptr, false});
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, queries, heads, 1);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, keys, heads, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, keys, heads, 1);
    auto * mask = masked ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, keys, GGML_PAD(queries, GGML_KQ_MASK_PAD)) : nullptr;
    auto * qp = static_cast<float *>(q->data);
    auto * kp = static_cast<ggml_fp16_t *>(k->data);
    auto * vp = static_cast<ggml_fp16_t *>(v->data);
    for (int i=0; i<ggml_nelements(q); ++i) qp[i] = 2.0f*std::sin(0.13f*i);
    for (int i=0; i<ggml_nelements(k); ++i) {
        kp[i] = ggml_fp32_to_fp16(1.3f*std::cos(0.17f*i));
        vp[i] = ggml_fp32_to_fp16(3.0f*std::sin(0.19f*i));
    }
    if (mask) {
        auto * mp=static_cast<ggml_fp16_t *>(mask->data);
        for (int i=0; i<ggml_nelements(mask); ++i) mp[i]=ggml_fp32_to_fp16(-INFINITY);
        for (int t=0;t<queries;++t) for(int j=0;j<std::min(keys,6+t);++j) mp[t*keys+j]=ggml_fp32_to_fp16(0);
    }
    auto * out = ggml_flash_attn_ext(ctx,q,k,v,mask,1.0f/std::sqrt(float(d)),0,0);
    auto * gf=ggml_new_graph(ctx);ggml_build_forward_expand(gf,out);
    bool ok=ggml_graph_compute_with_ctx(ctx,gf,threads)==GGML_STATUS_SUCCESS;
    const auto * actual=static_cast<const float *>(out->data);
    double error=0;
    for(int h=0;h<heads;++h) for(int t=0;t<queries;++t) {
        int count=masked ? std::min(keys,6+t) : keys;std::vector<double> scores(count);double maximum=-INFINITY;
        for(int j=0;j<count;++j) {
            double dot=0;
            for(int i=0;i<d;++i) dot+=double(qp[i+d*(t+queries*h)])*ggml_fp16_to_fp32(kp[i+d*(j+keys*h)]);
            scores[j]=dot/std::sqrt(double(d));maximum=std::max(maximum,scores[j]);
        }
        double total=0;for(auto & score:scores){score=std::exp(score-maximum);total+=score;}
        for(int i=0;i<d;++i) {
            double expected=0;for(int j=0;j<count;++j) expected+=scores[j]/total*ggml_fp16_to_fp32(vp[i+d*(j+keys*h)]);
            double value=actual[i+d*(h+heads*t)];
            error=std::isfinite(value) ? std::max(error,std::abs(expected-value)) : INFINITY;
        }
    }
    // Covers F16 query conversion in the generic kernel, while accumulation is F32.
    ok=ok && std::isfinite(error) && error<0.003;
    if(!ok)std::fprintf(stderr,"d=%d heads=%d q=%d k=%d threads=%d mask=%d error=%g\n",d,heads,queries,keys,threads,masked,error);
    ggml_free(ctx);return ok;
}
int main() {
    int failed=0,total=0;
    for(int d:{8,48,64})for(int heads:{6,8})for(int queries:{1,5})for(int threads:{1,4,16})for(bool masked:{false,true}) {
        ++total;if(!check(d,heads,queries,masked?256:31,threads,masked))++failed;
    }
    std::printf("scalar attention oracle: %d/%d passed\n",total-failed,total);return failed?1:0;
}
