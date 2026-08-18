# sm70-attn

Private fork of llama.cpp carrying a SM70 (V100) D256 flash-attention plugin for
Qwen3.5/3.6/3.8-27B (head_dim=256, GQA 6:1, 16 full-attention layers).

## What is in here

- Upstream master baseline (tag `baseline-2026-08-18` = ggml-org/llama.cpp `25ae3a9b3`).
- `bench/prompt_46k.txt` - deterministic synthetic 46300-token prompt
  (Qwen tokenizer), the standard A/B workload (document-class prefill). Do not edit.
- `bench/prompt_176k.txt` - deterministic synthetic 176292-token prompt, standard
  long-context workload (matches the 2026-08-18 production run length 1:1). Do not edit.
- `ggml/src/ggml-cuda/fattn.cu` - ~15-line hook (guarded by cc==700 + head_dim==256).
  All other shapes keep the stock path.
- `ggml/src/ggml-cuda/fattn-sm70-d256.cu` - the new kernel (self-contained).
- `sm70-hook.patch` - the hook as a standalone patch for rebase re-apply.
- `sync.sh` - upstream rebase script (run on the VM side).

## Rollback

`LLAMA_SM70_D256=0` in the environment disables the hook without rebuilding.

## Validation gates

- G1: greedy decode 64 tokens, top-1 identical vs stock kernel, logit max diff < 1e-2.
- G2: ncu bank conflicts < 1%, no register spills.
- G3: 46k prompt prefill (see `bench/prompt_46k.txt`), cumulative tokens/s
  46k >= 620, 30k >= 640, 4k unchanged within 5%.
- G3b: 176k standard workload (see `bench/prompt_176k.txt`), cumulative
  tokens/s >= 380 (stock baseline 281.5).

## Branch layout

- `main` - working branch (upstream master + plugin).
- remote `upstream` on the VM - ggml-org/llama.cpp (rebase source).
- Physical build box does `git pull && cmake --build build` only.
