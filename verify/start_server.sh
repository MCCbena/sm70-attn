#!/bin/bash
# long-lived production server for interactive verification
cd ~/sm70-attn
BIN=build/bin/llama-server
MODEL=$HOME/models/qwen3.8-27b-ud-q4kxl/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMPROJ=$HOME/models/qwen3.8-27b-ud-q6k/mmproj-BF16.gguf
DRAFT=$HOME/models/qwen3.8-dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
exec env CUDA_VISIBLE_DEVICES=1 "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --main-gpu 0 \
  -c 8192 -b 4096 -ub 512 -ngl 99 -fa on -ctk q4_0 -ctv f16 \
  --spec-type draft-dflash --spec-draft-model "$DRAFT" --spec-draft-n-max 2 \
  -t 12 -tb 16 --temp 0 --top-p 1 --top-k 1 -n 256 -np 1 \
  --host 127.0.0.1 --port 8080
