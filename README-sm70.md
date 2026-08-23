# sm70-attn

Private fork of llama.cpp carrying a SM70 (V100) D256 flash-attention plugin for
Qwen3.5/3.6/3.8-27B (head_dim=256, GQA 6:1, 16 full-attention layers).

## What is in here

- Upstream master baseline (tag `baseline-2026-08-18` = ggml-org/llama.cpp `25ae3a9b3`).
- `bench/prompt_46k.txt` - deterministic synthetic 46300-token prompt
  (Qwen tokenizer), the standard A/B workload. Do not edit.
- `bench/prompt_176k.txt` - 176292-token standard workload (the ROI
  shape; attention is 62% of prefill time here). Do not edit.
- `ggml/src/ggml-cuda/fattn.cu` - hook (guarded by cc==700 + head_dim==256
  + causal mask + prefill ne[1]>=256; F16/Q4_0 K/V only). All other shapes
  keep the stock path. Also carries the SM70_DUMP debug hook (see Tooling).
- `ggml/src/ggml-cuda/fattn-sm70-d256.cu` - launcher: Q f32->f16 padded
  staging, K/V dequant + direct read (**pos-major strides**, see the 8/23 fix),
  attention launch, f32 output scatter. Scratch is carved from the
  get_alloc_size extra region (includes SplitKV3 partials).
- `ggml/src/ggml-cuda/fattn-sm70-d256-kernel.cuh` - the 1CatAI-verified
  Split-D N32 D256 kernel (provenance in its header). Documented deviations
  vs upstream:
  - `kv_offset` parameter + Mask construction using it (8/18);
  - `ElementOut` template parameter — production launcher instantiates
    `float` so the attention output stays f32 end-to-end (8/23);
  - `SplitKV3` template branch + `sm70_d256_splitkv3_merge_kernel` —
    3-way KV split for long-prefix prefill (8/23, upstream splitkv3 patch),
    with an empty-segment gmem guard and a finite row_max init (-1e30) that
    upstream does not have.
- `ggml/src/ggml-cuda/sm70-vendor/` - header-only third-party closure:
  cute/ + cutlass/ (NVIDIA cutlass @ 62750a2b, Apache-2.0) and flash/
  (FA2 base layer from zhinianqin/flash-attention-v100 @ c2eda5e6,
  BSD-3). See sm70-vendor/README.md for provenance.
- `sm70-hook.patch` - the fattn.cu hook as a standalone patch for rebase
  re-apply.
- `sync.sh` - upstream rebase script (run on the VM side).

## Rollback / tuning

- `LLAMA_SM70_D256=0` in the environment disables the hook without rebuilding.
- `LLAMA_SM70_SPLITKV3_MIN_KV=<n>` - SplitKV3 activation threshold on kv_len
  (default 2048; `0` disables SplitKV3 entirely).

## The 8/23 stride fix (root cause of the "context contamination")

The launcher assumed `to_fp16_nc` writes its dequant dst as `[ne2][ne1][ne0]`
(head-major) but it actually linearizes as `[ne1][ne2][ne0]` = position-major
`[ctx][hkv][D]`. The old head-major strides (row=D, head=ctx*D) scrambled K on
every q4_0-K prefill: every full-attn layer output was off by ~0.69 relative
error, which compounded into O(1) nat logit drift, greedy argmax flips, and
the 23 "amnesia" signatures of the 585eeb session. Fixed in `284b251` (9
lines); pinned by the harness `posK/posKV` regression cases and verified by
three-way dump diffing (see Tooling). Full write-up: the case archive on the
physical box (sm70-attn上下文污染案-根因与修复档案-20260823.md).

## SplitKV3 (8/23 port of the upstream splitkv3 patch)

For long-prefix prefill chunks (kv_len >= threshold), the KV sweep is split 3
ways across `gridDim.y`, tripling CTA count so late chunks of a long prefill
stop serializing their KV scan on a saturated SM grid; a small merge kernel
combines the partial (max, sum, numerator) triples. 176k net gain is +3.7%
(420 -> 435.6 tok/s) — bounded by the HBM bandwidth wall, not SM occupancy.
Edge case upstream misses: a row's causal window can be fully masked within
one segment; our port initializes row_max to -1e30 (not -inf) in the SplitKV3
instantiation to keep `(-inf)-(-inf)=NaN` out of the partial chain.

## Validation gates

- G1: build clean; greedy decode 64 tokens top-1 identical vs stock,
  logit max diff < 1e-2.
- G1b: 176k full-prefill last-256 logits vs stock path: max abs diff
  < 1e-2, cosine > 0.9999 (f16 working precision).
  **Status 8/23: never executed on the real kernel (COMMIT B) before 8/23 —
  that gap is how the stride bug survived five days. The equivalent evidence
  now is the three-way dump verdict below.**
- G3: 46k prompt prefill A/B (see `bench/prompt_46k.txt`): cumulative
  tokens/s 46k >= 620, 30k >= 640, 4k within 5% of stock.
  **8/23 rerun: ON 613.25 tok/s.**
- G3b: 176k prefill A/B (see `bench/prompt_176k.txt`): cumulative
  tokens/s >= 380 (stock baseline 281.5).
  **8/23 rerun: ON 435.62 / stock 304.71 (+42.9%), post-fix and with
  SplitKV3; pre-splitkv3 fix-only rerun was 420.00 / 305.17 (+37.6%).**
- Rollback check: `LLAMA_SM70_D256=0` reproduces stock bit-for-bit.
- NEW G4 (8/23): harness `verify/sm70_verify` — 19 cases, 0 failing
  (dense / f32-out / spiky / pos-major K,V / SplitKV3 incl. empty-segment
  edge).
- NEW G5 (8/23): three-way dump verdict — first-layer ON-vs-OFF must be at
  the f16 rounding floor (~5e-4 dense, ~3e-3 splitkv3) and layer-avg
  |A-C| == |B-C| (equal distance to the CPU f32 reference). See Tooling.

## Tooling (8/23, the debugging kit that closed the case)

All under `verify/`; every script is self-contained and runs on the physical
box (`bash verify/<script>.sh`).

- `sm70_verify.cu` / `run.sh` — kernel harness, CPU f32 reference, md5
  fingerprint guard on the kernel source.
- `sm70_layer_diff.py` — layerwise ON/OFF diff from SM70_DUMP captures
  (`--c` adds the CPU f32 reference phase for three-way blame assignment).
- `sm70_variant_scan.py` — layout variant scan (this is what pinned the
  stride bug: K-transposed variant dropped |v-C| from 0.69 to 0.08).
- `sm70_repro.py` — minimal reproduction from a captured kernel-input dump.
- `dump_ab.sh` / `dump_cpu.sh` / `dump_kv.sh` — capture phases (env:
  `PROMPT`, `CT_V`, `ON_DUMP`, `OFF_DUMP`).
- `dump_ab.sh` + `SM70_DUMP`/`SM70_DUMP_MAX` env on the server — per-call
  attention output capture (hooks both sm70 and stock; CUDA-graph-capture
  aware; CPU twin hook in ggml-cpu/ops.cpp behind the same env).
- `logit_ab_fixed.sh` — three-phase logit fork verdict (A/A2/B + compare).
- `perf_ab.sh` / `perf_ab_176k.sh` — post-change throughput acceptance.

## Branch layout

- `main` - working branch (upstream master + plugin).
- remote `upstream` on the VM - ggml-org/llama.cpp (rebase source).
- Physical build box does `git pull && cmake --build build` only.

## Commit chain

- `fe7a3e7ca` - v1.0 hook (pipeline validation, stock kernel) [COMMIT A]
- `bfc1e99` - v1.1 real SM70 D256 Split-D kernel + vendor closure [COMMIT B]
- `afa6e46` - logit_ab v3 final (the 8/22 verdict tooling)
- `284b251` - **8/23 stride fix** (the 0.69 root cause; K/V dequant
  strides are position-major)
- `daca614`..`0b76746` - 8/23 debugging kit (dump hooks, three-way
  analysis, variant scan, repro)
- `db66e31` - A-group hardening (F32 gating REJECT, posK/posKV harness
  regression cases)
- `6e2c53d` + `1ce3b3d` - **SplitKV3 port** (kernel branch + merge + gate +
  NaN edge fix)
