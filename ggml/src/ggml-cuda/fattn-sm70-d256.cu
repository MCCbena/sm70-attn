// SPDX-FileCopyrightText: Copyright 2026-2026 the llama.cpp authors
// SPDX-License-Identifier: MIT
//
// sm70-attn: SM70 (Volta) D256 FlashAttention kernel for Qwen3.8-27B.
//
// v1.0 (commit A): thin hook that routes (SM70 + D256 + prefill) to the exact
//   stock kernel shape the prefill path runs today:
//     get_best_fattn_kernel -> volta branch -> MMA_F16
//     mma_f16_switch_ncols2<256,256> (Volta, GQA6: gqa_ratio%2==0) -> ncols2=2
//     mma_f16_switch_ncols1<256,256,2>: ne[1]>16 -> case <256,256,32,2>
//   (ncols = 32*2 = 64 Q rows/tile, 128 threads, nbatch_fa=32, Q_in_reg=true).
//   This validates the full plugin pipeline (enum / alloc / dequant /
//   launcher / hook condition) with zero kernel change, so G1 (numeric) is
//   guaranteed to pass — it runs the very same kernel the stock build runs.
//
// Why stock is slow here (design target for v1.1): the config (256,256,ncols>=32)
// uses Q_in_reg=true with nbatch_fa=32 on a 128-thread block. The full-DV f16
// PV accumulators (T_C_VKQ x DV/16 fragments) + KQ_C + Q_B fragments push the
// per-thread register count past the 255 limit, so the kernel spills. This is
// the structural cause of the measured 15.8 TFLOPS (13% of peak) and matches
// the upstream "TODO tune specifically for Volta" (fattn-mma-f16.cuh L123).
//
// v1.1 (commit B, next): bespoke Split-D kernel replacing this extern call:
//   - 4 warps in 2 pairs; each pair shares the QK KQ accumulator (f32)
//   - PV split across D: each warp owns 128 of the 256 output dims
//     -> per-thread PV accumulator 128-dim instead of 256-dim, kills the spill
//   - d-chunk (4x64) K/V double-buffered pipeline (LDG->smem; no cp.async on Volta)
//
// Rollback: LLAMA_SM70_D256=0 env var disables the hook (recompile-free).
// Target shape: (SM70, head_dim=256, f16 KV, causal mask, prefill ne01>=256).

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"   // provides the mma_f16_case<> template + explicit instantiations

void ggml_cuda_flash_attn_ext_sm70_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // v1.0: exact stock shape for D256/GQA6/Volta prefill — validates the pipeline.
    // v1.1: this body becomes the bespoke Split-D SM70 D256 kernel launch.
    ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 32, 2>(ctx, dst);
}
