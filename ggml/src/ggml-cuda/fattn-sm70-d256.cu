// SPDX-FileCopyrightText: Copyright 2026 the llama.cpp authors
// SPDX-License-Identifier: MIT
//
// sm70-attn plugin — commit B (path A): real SM70 D256 Split-D kernel.
//
// The device kernel is the 1CatAI-verified Split-D N32 flash attention
// (provenance in fattn-sm70-d256-kernel.cuh; 1Cat-vLLM v1.3.0). The
// verified core (smem layouts / HMMA.884 atoms / K-V pipeline / online
// softmax / causal mask) is byte-identical to upstream. Kernel edits:
// one extra `kv_offset` parameter (causal boundary for padded Q) + the
// Mask construction using it. All adaptation lives in this file.
//
// Design record: p1b-design-final.md. Key points:
//   * Scale: stock pre-multiplies Q by `scale` (natural-log domain); the
//     1Cat kernel keeps Q unscaled and folds the scale into exp2 via
//     softmax_scale_log2 = scale*log2(e). Mathematically identical.
//   * Q: staged f32->f16 into a 64-row-padded scratch (the kernel's tiled
//     Q copy is unguarded, so the last partial Q tile MUST be padded; pad
//     rows are zero and their outputs are never written back by the scatter).
//   * K/V: read directly from the stock f16 buffers — native f16 cache, or
//     the stock f16 dequant extra (this launcher runs the same to_fp16
//     dequant the stock launch_fattn does). No K/V staging: the kernel's
//     causal n_block_max bound + causal mask guarantee no read beyond
//     kv_len (only the causal diagonal block is read unguarded, and all
//     its columns are < kv_len), and the f16 buffers are physically
//     larger than kv_len*256.
//   * Causal: derived from positions (col > kv_offset + row); kv_offset =
//     kv_len - q_len passed explicitly (padded Q would break the kernel's
//     own derivation). No mask tensor consumed.
//   * GQA: kernel grid.z = hkv*gqa (head_q = j*gqa + c); the kernel maps
//     head_q -> head_kv. Q staging/scatter map head_q -> (b, j, c):
//     c = Q head within the KV group (Q data depends only on c),
//     j = KV head (K/V select), b = sequence.
//   * Scratch layout: [hkv][gqa][nb][rows][256] f16, slice(j,c) at
//     (j*gqa + c) * nb * rows * 256; seq b at b * rows * 256 inside.
//   * Rollback: env LLAMA_SM70_D256=0 forces the stock path.

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-sm70-d256-kernel.cuh"
#include <cstdio>

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634f
#endif

// NOTE: no anonymous namespace in this file. The CuTe vendor headers
// (cute/atom/mma_traits_sm70.hpp) open their own anonymous namespaces;
// a second anonymous namespace in this TU makes cudafe's
// _GLOBAL__N__<hash> symbol mangling ambiguous ("reference to
// '_GLOBAL__N__...' is ambiguous"). Helpers below are therefore plain
// statics / file-scope constants.

constexpr int SM70_D256_BLOCK_M = 64;
constexpr int SM70_D256_D = 256;

// Q f32 -> f16 staging.
// grid = (q_pad, nb*hkv, gqa); block = 128 (float2 grain over the 256 row).
// dst (Qs) layout: [hkv][gqa][nb][q_pad][256] f16.
__global__ void sm70_d256_stage_q_kernel(
        const float2 * __restrict__ src, half2 * __restrict__ dst,
        const int q_len, const int hkv, const int gqa, const int nb,
        const int64_t src_row, const int64_t src_head, const int64_t src_seq) {
    const int r  = blockIdx.x;
    const int bj = blockIdx.y;          // b*hkv + j
    const int c  = blockIdx.z;          // Q head within the KV group
    if (r >= q_len) {
        return;                         // pad rows: pre-zeroed scratch
    }
    const int j = bj % hkv;
    const int b = bj / hkv;
    const float2 v = src[threadIdx.x
                   + (int64_t) r * src_row
                   + (int64_t) c * src_head
                   + (int64_t) b * src_seq];
    dst[threadIdx.x
      + (int64_t) ((int64_t) j * gqa + c) * (int64_t) nb * (gridDim.x * 128)
      + (int64_t) b * (gridDim.x * 128)
      + (int64_t) r * 128] = __float22half2_rn(v);
}

// Output scatter: staged [hkv][gqa][nb][rows][256] f16 -> stock f32 dst.
// grid = (q_len, nb*hkv, gqa); block = 128.
__global__ void sm70_d256_scatter_kernel(
        const half2 * __restrict__ src, float2 * __restrict__ dst,
        const int hkv, const int gqa, const int nb,
        const int64_t dst_row, const int64_t dst_head, const int64_t dst_seq) {
    const int r  = blockIdx.x;
    const int bj = blockIdx.y;
    const int c  = blockIdx.z;
    const int j  = bj % hkv;
    const int b  = bj / hkv;
    const half2 v = src[threadIdx.x
        + (int64_t) ((int64_t) j * gqa + c) * (int64_t) nb * (gridDim.x * 128)
        + (int64_t) b * (gridDim.x * 128)
        + (int64_t) r * 128];
    float2 o;
    o.x = __low2float(v);
    o.y = __high2float(v);
    dst[threadIdx.x
      + (int64_t) r * dst_row
      + (int64_t) c * dst_head
      + (int64_t) b * dst_seq] = o;
}

// dequant K/V (q4_0 / f32) into the stock f16 extra buffers — same code
// path (to_fp16 / to_fp16_nc) as the stock launch_fattn.
static void sm70_d256_dequant_kv(
        ggml_tensor * K, ggml_tensor * V,
        const ggml_cuda_flash_attn_ext_f16_extra_data & f16_extra,
        const bool V_is_K_view, cudaStream_t stream) {
    if (K->type != GGML_TYPE_F16) {
        const char * K_data = (const char *) K->data;
        half * K_f16 = (half *) f16_extra.K;
        GGML_ASSERT(f16_extra.K != 0);
        if (ggml_is_contiguously_allocated(K)) {
            const size_t bs = ggml_blck_size(K->type);
            const size_t ts = ggml_type_size(K->type);
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16, ggml_nelements(K), stream);
        } else {
            const size_t bs = ggml_blck_size(K->type);
            const size_t ts = ggml_type_size(K->type);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            to_fp16(K_data, K_f16, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                    K->nb[1] / ts, K->nb[2] / ts, K->nb[3] / ts, stream);
        }
    }
    if (!V_is_K_view && V->type != GGML_TYPE_F16) {
        const char * V_data = (const char *) V->data;
        half * V_f16 = (half *) f16_extra.V;
        GGML_ASSERT(f16_extra.V != 0);
        if (ggml_is_contiguously_allocated(V)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
            to_fp16(V_data, V_f16, ggml_nelements(V), stream);
        } else {
            const size_t ts = ggml_type_size(V->type);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
            to_fp16(V_data, V_f16, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                    V->nb[1] / ts, V->nb[2] / ts, V->nb[3] / ts, stream);
        }
    }
}

static bool sm70_env_disabled() {
    static const bool disabled = [] {
        const char * e = getenv("LLAMA_SM70_D256");
        return e && e[0] == '0';
    }();
    return disabled;
}

// Routing probe: prints why a decision was made. Default = first decision
// only; set LLAMA_SM70_D256_DEBUG=1 for every call.
static void sm70_d256_probe(const char * reason, int cc,
                            const ggml_tensor * Q, const ggml_tensor * K,
                            const ggml_tensor * V, const ggml_tensor * mask) {
    static const bool verbose = getenv("LLAMA_SM70_D256_DEBUG") != nullptr;
    static int printed = 0;
    if (!verbose && printed >= 20) {
        return;
    }
    fprintf(stderr, "[sm70-d256] #%d %s | cc=%d Q=(%lld,%lld,%lld,%lld) Qtype=%d "
            "K=(%lld,%lld,%lld,%lld) Ktype=%d Knb0=%llu Knb1=%llu Knb2=%llu rowK=%llu "
            "Vtype=%d Vnb0=%llu Vnb1=%llu Vnb2=%llu rowV=%llu mask=%p\n",
            ++printed, reason, cc,
            (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3], (int) Q->type,
            (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3], (int) K->type,
            (unsigned long long) K->nb[0], (unsigned long long) K->nb[1], (unsigned long long) K->nb[2], (unsigned long long) ggml_row_size(K->type, K->ne[0]),
            (int) V->type, (unsigned long long) V->nb[0], (unsigned long long) V->nb[1], (unsigned long long) V->nb[2], (unsigned long long) ggml_row_size(V->type, V->ne[0]),
            (const void *) mask);
}

// ------------------------------------------------------------------- public
bool ggml_cuda_sm70_d256_supported(int cc, const ggml_tensor * dst) {
    if (cc != GGML_CUDA_CC_VOLTA || sm70_env_disabled()) {
        // probe only when cc is volta (the interesting case for the probe)
        if (cc == GGML_CUDA_CC_VOLTA) {
            const ggml_tensor * Qp = dst->src[0];
            const ggml_tensor * Kp = dst->src[1];
            const ggml_tensor * Vp = dst->src[2];
            const ggml_tensor * Mp = dst->src[3];
            sm70_d256_probe(cc != GGML_CUDA_CC_VOLTA ? "REJECT: cc!=volta" : "REJECT: env disabled", cc, Qp, Kp, Vp, Mp);
        }
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (Q->ne[0] != SM70_D256_D || K->ne[0] != SM70_D256_D || V->ne[0] != SM70_D256_D) {
        sm70_d256_probe("REJECT: head_dim != 256", cc, Q, K, V, mask);
        return false;
    }
    if (!mask || Q->ne[1] < 256) { // prefill only; decode/MTP/small batches -> stock
        sm70_d256_probe("REJECT: no mask or q_len < 256", cc, Q, K, V, mask);
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0) {
        sm70_d256_probe("REJECT: gqa ratio", cc, Q, K, V, mask);
        return false;
    }
    const bool kv_ok = (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_F32 || K->type == GGML_TYPE_Q4_0)
                    && (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_F32 || V->type == GGML_TYPE_Q4_0);
    if (!kv_ok) {
        sm70_d256_probe("REJECT: kv type", cc, Q, K, V, mask);
        return false;
    }
    // NB (route A post-mortem, 8/19): ggml nb[0] = bytes per ELEMENT (2 for F16);
    // the KV cache is laid out [ctx][head][dim] (dim fastest, nb[0] contiguous),
    // so rows are NOT contiguous - nb[1] (ctx stride) >> row size.
    // All the kernel needs for the f16-direct path is per-row contiguity
    // (nb[0] == elem size); head/ctx access goes through explicit strides.
    // The dequant path (to_fp16_nc) handles ANY source strides.
    if (K->type == GGML_TYPE_F16 && K->nb[0] != sizeof(half)) {
        sm70_d256_probe("REJECT: K rows not contiguous", cc, Q, K, V, mask);
        return false;
    }
    if (V->type == GGML_TYPE_F16 && V->nb[0] != sizeof(half)) {
        sm70_d256_probe("REJECT: V rows not contiguous", cc, Q, K, V, mask);
        return false;
    }
    sm70_d256_probe("ACCEPT: sm70 d256 kernel selected", cc, Q, K, V, mask);
    return true;
}

// scratch (all carved from the get_alloc_size extra region after dst->data,
// the stock f16_extra model): [K dequant][V dequant] (f16_extra layout) +
// Qs (padded f16 Q) + Os (f16 output staging).
size_t ggml_cuda_sm70_d256_alloc_size(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
    const bool need_f16_K = K->type != GGML_TYPE_F16;
    const bool need_f16_V = !V_is_K_view && V->type != GGML_TYPE_F16;

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);
    size_t dequant = (size_t) (f16_extra.end - (uintptr_t) dst->data);

    const int q_pad = (((int) Q->ne[1] + SM70_D256_BLOCK_M - 1) / SM70_D256_BLOCK_M) * SM70_D256_BLOCK_M;
    const int64_t nQ = (int64_t) Q->ne[2] * q_pad * SM70_D256_D * (int) Q->ne[3]; // f16 elems
    dequant = GGML_PAD(dequant, 128);
    dequant += (size_t) (2 * nQ) * sizeof(half);   // Qs + Os
    return dequant;
}

void ggml_cuda_flash_attn_ext_sm70_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    GGML_ASSERT(cc == GGML_CUDA_CC_VOLTA);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int hkv    = (int) K->ne[2];
    const int gqa    = (int) (Q->ne[2] / K->ne[2]);
    const int q_len  = (int) Q->ne[1];
    const int kv_len = (int) K->ne[1];
    const int nb     = (int) Q->ne[3];
    GGML_ASSERT(Q->ne[1] == K->ne[1]);
    GGML_ASSERT(K->type == V->type);

    const int q_pad = ((q_len + SM70_D256_BLOCK_M - 1) / SM70_D256_BLOCK_M) * SM70_D256_BLOCK_M;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    GGML_ASSERT(logit_softcap == 0.0f);
    const float softmax_scale_log2 = scale * M_LOG2E;
    const int kv_offset = kv_len - q_len;

    const int64_t nQ = (int64_t) hkv * gqa * nb * q_pad * SM70_D256_D; // f16 elems

    // ------------------------------------- scratch layout (extra region)
    // base = start of the get_alloc_size extra region (right after dst out)
    const char * base = (const char *) dst->data + ggml_nbytes(dst);

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst,
            K->type != GGML_TYPE_F16,
            !(V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs))
              && K->type == GGML_TYPE_F16) && V->type != GGML_TYPE_F16);

    size_t dequant_bytes = (size_t) (f16_extra.end - (uintptr_t) dst->data);
    dequant_bytes = GGML_PAD(dequant_bytes, 128);
    char * Qs_bytes = (char *) base + dequant_bytes;
    half * Qs = (half *) Qs_bytes;
    half * Os = (half *) (Qs_bytes + (size_t) nQ * sizeof(half));

    // zero pad rows of Qs (their outputs are never scattered back) and all of Os
    CUDA_CHECK(cudaMemsetAsync((void *) Qs_bytes, 0, (size_t) 2 * nQ * sizeof(half), ctx.stream()));

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    cudaStream_t stream = ctx.stream();
    sm70_d256_dequant_kv((ggml_tensor *) K, (ggml_tensor *) V, f16_extra, V_is_K_view, stream);

    const half * K_h2;
    const half * V_h2;
    int64_t k_row_stride, k_head_stride;
    int64_t v_row_stride, v_head_stride;
    if (K->type == GGML_TYPE_F16) {
        K_h2 = (const half *) K->data;
        k_row_stride   = K->nb[1] / sizeof(half);
        k_head_stride  = K->nb[2] / sizeof(half);
    } else {
        K_h2 = (const half *) f16_extra.K;
        k_row_stride   = K->ne[0];          // contiguous dequant buffer
        k_head_stride  = (int64_t) K->ne[1] * K->ne[0];
    }
    if (V_is_K_view) {
        V_h2 = K_h2;
        v_row_stride  = k_row_stride;
        v_head_stride = k_head_stride;
    } else if (V->type == GGML_TYPE_F16) {
        V_h2 = (const half *) V->data;
        v_row_stride   = V->nb[1] / sizeof(half);
        v_head_stride  = V->nb[2] / sizeof(half);
    } else {
        V_h2 = (const half *) f16_extra.V;
        v_row_stride   = V->ne[0];
        v_head_stride  = (int64_t) V->ne[1] * V->ne[0];
    }

    // ------------------------------------------------------ stage Q (f32->f16)
    {
        const dim3 grid(q_pad, nb * hkv, gqa);
        sm70_d256_stage_q_kernel<<<grid, 128, 0, stream>>>(
            (const float2 *) Q->data, (half2 *) Qs, q_len, hkv, gqa, nb,
            Q->nb[1] / 8, Q->nb[2] / 8, Q->nb[3] / 8);
        CUDA_CHECK(cudaGetLastError());
    }

    // ------------------------------------------------------------- attention
    using Traits = FLASH_NAMESPACE::Sm70D256SplitDTraits;
    using El = cutlass::half_t;
    auto kernel = FLASH_NAMESPACE::sm70_d256_splitd_dense_kernel<El, false>;

    static bool smem_raised = false;
    if (!smem_raised) {
        CUDA_CHECK(cudaFuncSetAttribute((const void *) kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemBytes));
        smem_raised = true;
    }

    const dim3 block(Traits::kNThreads);
    const dim3 grid(q_pad / SM70_D256_BLOCK_M, nb, hkv * gqa);

    kernel<<<grid, block, Traits::kSmemBytes, stream>>>(
            (const El *) Qs,
            (const El *) K_h2,
            (const El *) V_h2,
            (El *) Os,
            /*q_batch_stride*/ (int64_t) (hkv * gqa) * nb * q_pad * SM70_D256_D,
            /*q_row_stride  */ SM70_D256_D,
            /*q_head_stride */ (int64_t) nb * q_pad * SM70_D256_D,
            /*k_outer_stride*/ 0,
            /*k_row_stride  */ (int) k_row_stride,
            /*k_head_stride */ (int) k_head_stride,
            /*v_outer_stride*/ 0,
            /*v_row_stride  */ (int) v_row_stride,
            /*v_head_stride */ (int) v_head_stride,
            q_pad,
            kv_len,
            hkv * gqa,   // heads_q
            hkv,         // heads_kv
            kv_offset,
            softmax_scale_log2,
            nullptr, 0, 0);
    CUDA_CHECK(cudaGetLastError());

    // ------------------------------------------------------------- scatter
    {
        const dim3 grid(q_len, nb * hkv, gqa);
        sm70_d256_scatter_kernel<<<grid, 128, 0, stream>>>(
            (const half2 *) Os, (float2 *) dst->data, hkv, gqa, nb,
            Q->nb[1] / 8, Q->nb[2] / 8, Q->nb[3] / 8);
        CUDA_CHECK(cudaGetLastError());
    }
}
