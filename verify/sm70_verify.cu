// ============================================================================
// sm70 d256 attention — standalone cross-check harness
//
// Feeds the production kernel (fattn-sm70-d256-kernel.cuh, byte-identical to
// the one in the sm70-attn build) the EXACT same layout/strides/params the
// llama.cpp launcher (fattn-sm70-d256.cu) uses, and compares output
// element-wise against a CPU reference: softmax(QK^T/sqrt(256))·V, causal,
// GQA 6:1, head_dim 256.
//
// Row sampling: small cases compare ALL rows/heads; big cases compare a
// boundary-focused subset (tile edges 64/128/256, diagonal rows, tail rows)
// across a few heads — tile-boundary and causal-diagonal bugs show up there.
//
// OOB-GUARD (arg "o" / SM70_VERIFY_OOB=1): K/V allocated at EXACT
// [nb][hkv][kvlen][256] (no headroom). Any read past kvlen rows is an
// illegal access — visible here (crash / NaN leak) and under
// compute-sanitizer --tool memcheck.
//
// Build: see build.sh
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FLASH_NAMESPACE flash_sm70
#include "fattn-sm70-d256-kernel.cuh"

using El = cutlass::half_t;
using Traits = FLASH_NAMESPACE::Sm70D256SplitDTraits;
static_assert(Traits::kHeadDim == 256 && Traits::kBlockM == 64 && Traits::kBlockN == 32);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    cudaGetLastError(); \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)

static uint32_t s_rng = 0x9E3779B9u;
static float frand() {
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17;
    return (float)(s_rng >> 8) / 8388608.0f - 1.0f;
}

// CPU f32 reference for ONE (batch, q-head, q-row): dot with each kv col,
// online-softmax-safe (scale in raw space, exp2 parity with kernel).
static std::vector<float> ref_row(const float* q_row, const float* k, const float* v,
                                  int kvlen, int kv_offset, int r, float out[256]) {
    const float scale = 1.0f / 16.0f;          // 1/sqrt(256)
    const int limit = std::min(kvlen, r + 1 + kv_offset);
    float mx = -1e30f;
    for (int c = 0; c < limit; ++c) {
        float s = 0.0f;
        for (int d = 0; d < 256; ++d) s += q_row[d] * k[(size_t)c * 256 + d];
        mx = std::max(mx, s);
    }
    float sum = 0.0f;
    std::vector<float> pbuf(limit);
    for (int c = 0; c < limit; ++c) {
        float s = 0.0f;
        for (int d = 0; d < 256; ++d) s += q_row[d] * k[(size_t)c * 256 + d];
        pbuf[c] = exp2f((s - mx) * scale * 1.4426950408889634f);
        sum += pbuf[c];
    }
    const float inv = 1.0f / sum;
    for (int d = 0; d < 256; ++d) {
        float o = 0.0f;
        for (int c = 0; c < limit; ++c) o += pbuf[c] * v[(size_t)c * 256 + d];
        out[d] = o * inv;
    }
    return pbuf;
}

struct Case { const char* name; int nb, kvlen, q_len; bool sample; };

static int run_case(const Case& tc, bool oob) {
    const int nb = tc.nb, kvlen = tc.kvlen, q_len = tc.q_len;
    const int hkv = 4, heads_q = 24, gqa = 6, D = 256;
    const int q_pad = ((q_len + 63) / 64) * 64;
    const int kv_offset = kvlen - q_len;
    const float scale_log2 = (1.0f / 16.0f) * 1.4426950408889634f;

    // host f32 source (Q rows beyond q_len = pad = zero, per launcher memset)
    std::vector<float> Qf((size_t) nb * heads_q * q_pad * D, 0.0f),
                       Kf((size_t) nb * hkv * kvlen * D),
                       Vf((size_t) nb * hkv * kvlen * D);
    for (int b = 0; b < nb; ++b) for (int h = 0; h < heads_q; ++h)
        for (int r = 0; r < q_len; ++r) for (int d = 0; d < D; ++d)
            Qf[((size_t)b * heads_q + h) * q_pad * D + r * D + d] = frand();
    for (auto& x : Kf) x = frand();
    for (auto& x : Vf) x = frand();

    const int kv_alloc_rows = oob ? kvlen : kvlen + 3;
    std::vector<__half> Qh(Qf.size()), Kh((size_t) nb * hkv * kv_alloc_rows * D), Vh((size_t) nb * hkv * kv_alloc_rows * D);
    for (size_t i = 0; i < Qf.size(); ++i) Qh[i] = __float2half(Qf[i]);
    for (size_t i = 0; i < Kf.size(); ++i) Kh[i] = __float2half(Kf[i]);
    for (size_t i = 0; i < Vf.size(); ++i) Vh[i] = __float2half(Vf[i]);

    void *dQ, *dK, *dV, *dO;
    CK(cudaMalloc(&dQ, Qh.size() * sizeof(El)));
    CK(cudaMalloc(&dK, Kh.size() * sizeof(El)));
    CK(cudaMalloc(&dV, Vh.size() * sizeof(El)));
    CK(cudaMalloc(&dO, (size_t) nb * heads_q * q_len * D * sizeof(El)));
    CK(cudaMemcpy(dQ, Qh.data(), Qh.size() * sizeof(El), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, Kh.data(), Kh.size() * sizeof(El), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, Vh.data(), Vh.size() * sizeof(El), cudaMemcpyHostToDevice));

    auto kernel = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false>;
    static bool smem_raised = false;
    if (!smem_raised) {
        CK(cudaFuncSetAttribute((const void*) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemBytes));
        smem_raised = true;
    }
    const dim3 block(Traits::kNThreads);
    const dim3 grid(q_pad / 64, nb, hkv * gqa);
    const int64_t head_s = (int64_t) kv_alloc_rows * D;  // contiguous per batch (== launcher dequant path)
    const int k_outer = (nb == 1) ? 0 : (int) head_s;   // nb=1: exact production value (0); nb=2: correct value
    const int v_outer = (nb == 1) ? 0 : (int) head_s;

    kernel<<<grid, block, Traits::kSmemBytes, 0>>>(
        (const El*) dQ, (const El*) dK, (const El*) dV, (El*) dO,
        (int64_t) heads_q * q_pad * D, D, (int64_t) q_pad * D,
        k_outer, D, (int) head_s,
        v_outer, D, (int) head_s,
        q_pad, kvlen, heads_q, hkv, kv_offset, scale_log2, nullptr, 0, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__half> Ohost((size_t) nb * heads_q * q_len * D);
    CK(cudaMemcpy(Ohost.data(), dO, Ohost.size() * sizeof(El), cudaMemcpyDeviceToHost));

    // which rows/heads to check
    std::vector<int> rows, heads;
    if (!tc.sample) {
        for (int r = 0; r < q_len; ++r) rows.push_back(r);
        for (int h = 0; h < heads_q; ++h) heads.push_back(h);
    } else {
        std::vector<int> cand = {0, 1, 2, 3, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};
        for (int r = q_len - 3; r < q_len; ++r) cand.push_back(r);
        for (int r = 32; r < q_len; r += 1024) for (int e = -1; e <= 1; ++e) cand.push_back(r + e);
        std::sort(cand.begin(), cand.end()); cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
        for (int r : cand) if (r >= 0 && r < q_len) rows.push_back(r);
        heads = {0, 1, 5, 6, 11, 12, 17, 23};  // across kv-head boundaries (0,5|6,11|12,17|23)
    }

    double max_abs = 0.0; int bad = 0, nan_out = 0, ncmp = 0;
    int first_r = -1, first_h = -1, first_b = -1, first_d = -1;
    // top-5 errors (even in PASS cases: tells noise floor apart from near-miss)
    struct ErrRec { double e; int b, h, r, d; };
    std::vector<ErrRec> top;
    auto consider = [&](double e, int b, int h, int r, int d) {
        if (top.size() < 5) top.push_back({e, b, h, r, d});
        else if (e > top.back().e) {
            top.back() = {e, b, h, r, d};
            std::sort(top.begin(), top.end(), [](const ErrRec& a, const ErrRec& b){ return a.e > b.e; });
        }
    };
    for (int b = 0; b < nb; ++b) {
      for (int h : heads) {
        const int hkv_idx = h / gqa;
        const float* qbase = &Qf[((size_t) b * heads_q + h) * q_pad * D];
        const float* kbase = &Kf[((size_t) b * hkv + hkv_idx) * kvlen * D];
        const float* vbase = &Vf[((size_t) b * hkv + hkv_idx) * kvlen * D];
        const __half* obase = &Ohost[((size_t) b * heads_q + h) * q_len * D];
        for (int r : rows) {
            float ref[256];
            ref_row(qbase + (size_t) r * D, kbase, vbase, kvlen, kv_offset, r, ref);
            for (int d = 0; d < D; ++d) {
                const float got = __half2float(obase[(size_t) r * D + d]);
                ncmp++;
                if (std::isnan(got) || std::isinf(got)) {
                    nan_out++;
                    if (nan_out <= 5) printf("  [%s] NaN/Inf out b=%d h=%d r=%d d=%d got=%g\n", tc.name, b, h, r, d, got);
                    continue;
                }
                const double err = std::abs((double) got - ref[d]);
                if (err > max_abs) max_abs = err;
                consider(err, b, h, r, d);
                if (err > 1e-2) {
                    bad++;
                    if (bad <= 5) printf("  [%s] big err=%.3e b=%d h=%d r=%d d=%d got=%.5f ref=%.5f\n", tc.name, err, b, h, r, d, got, ref[d]);
                    if (first_r < 0) { first_r = r; first_h = h; first_b = b; first_d = d; }
                }
            }
        }
      }
    }
    printf("case %-13s nb=%d kv=%6d q=%5d pad=%5d | elems=%-7d max|err|=%.3e  >1e-2: %6d  nan/inf: %d%s\n",
           tc.name, nb, kvlen, q_len, q_pad, ncmp, max_abs, bad, nan_out,
           (bad || nan_out) ? "   <-- FAIL" : "");
    for (auto& t : top) printf("   top: err=%.3e b=%d h=%d r=%d d=%d\n", t.e, t.b, t.h, t.r, t.d);
    if (bad > 0) printf("   first big: b=%d head=%d row=%d d=%d\n", first_b, first_h, first_r, first_d);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return (bad || nan_out) ? 1 : 0;
}

int main(int argc, char** argv) {
    const bool oob = (getenv("SM70_VERIFY_OOB") != nullptr) || (argc > 1 && argv[1][0] == 'o');
    const bool full = (getenv("SM70_VERIFY_FULL") != nullptr) || (argc > 1 && argv[1][0] == 'f');

    static const Case small[] = {
        { "diag-64",     1,   64,  64, false },  // single N-block + M-block
        { "diag-279",    1,  279, 279, false },  // 8/20 production shape (q_len=279)
        { "cap512-q279", 1,  512, 279, false },  // production chunk: kv_offset=233 > 0
        { "tail-31",     1,   31,  31, false },  // kv_len < kBlockN: partial first/only block
        { "tail-95",     1,   95,  95, false },  // 3 N-blocks, last partial
        { "long-3000",   1, 3000, 3000, true  },  // 94 N-blocks, sampled
        { "nb2-279",     2,  279, 279, true  },  // batch=2, sampled
        { "nb2-long",    2, 3000, 3000, true  },
    };
    static const Case big[] = {
        { "full-32k",    1, 32768, 32768, true },  // 1024 N-blocks, sampled
    };

    int fails = 0;
    printf("=== sm70 d256 cross-check (%s) ===\n", oob ? "OOB-GUARD: exact-size K/V" : "normal: 3-row K/V headroom");
    for (auto& tc : small) fails += run_case(tc, oob);
    if (full) for (auto& tc : big) fails += run_case(tc, oob);
    printf("=== %s: %d case(s) failing ===\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
