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
  + causal mask + prefill ne[1]>=256). All other shapes keep the stock path.
- `ggml/src/ggml-cuda/fattn-sm70-d256.cu` - launcher: Q f32->f16 padded
  staging, K/V dequant + direct read, attention launch, f16->f32 output
  scatter. Scratch is carved from the get_alloc_size extra region.
- `ggml/src/ggml-cuda/fattn-sm70-d256-kernel.cuh` - the 1CatAI-verified
  Split-D N32 D256 kernel (provenance in its header). The only kernel
  edits vs upstream: one extra `kv_offset` parameter + Mask construction
  using it.
- `ggml/src/ggml-cuda/sm70-vendor/` - header-only third-party closure:
  cute/ + cutlass/ (NVIDIA cutlass @ 62750a2b, Apache-2.0) and flash/
  (FA2 base layer from zhinianqin/flash-attention-v100 @ c2eda5e6,
  BSD-3). See sm70-vendor/README.md for provenance.
- `sm70-hook.patch` - the fattn.cu hook as a standalone patch for rebase
  re-apply.
- `sync.sh` - upstream rebase script (run on the VM side).

## Rollback

`LLAMA_SM70_D256=0` in the environment disables the hook without rebuilding.

## Validation gates

- G1: build clean; greedy decode 64 tokens top-1 identical vs stock,
  logit max diff < 1e-2.
- G1b: 176k full-prefill last-256 logits vs stock path: max abs diff
  < 1e-2, cosine > 0.9999 (f16 working precision).
- G3: 46k prompt prefill A/B (see `bench/prompt_46k.txt`): cumulative
  tokens/s 46k >= 620, 30k >= 640, 4k within 5% of stock.
- G3b: 176k prefill A/B (see `bench/prompt_176k.txt`): cumulative
  tokens/s >= 380 (stock baseline 281.5).
- Rollback check: `LLAMA_SM70_D256=0` reproduces stock bit-for-bit.

## Branch layout

- `main` - working branch (upstream master + plugin).
- remote `upstream` on the VM - ggml-org/llama.cpp (rebase source).
- Physical build box does `git pull && cmake --build build` only.

## Commit chain

- `fe7a3e7ca` - v1.0 hook (pipeline validation, stock kernel) [COMMIT A]
- (next) - v1.1 real SM70 D256 Split-D kernel + vendor closure [COMMIT B]
