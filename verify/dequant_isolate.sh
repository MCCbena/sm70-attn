#!/bin/bash
# Isolate the whole-cache K dequant cost on the STOCK path: q4_0 K (dequant
# per call) vs f16 K (no dequant, but 2x KV read volume).
BIN=~/sm70-attn/build/bin/llama-bench
MODEL=~/models/qwen3.8-27b-ud-q4kxl/Qwen3.8-27B-UD-Q4_K_XL.gguf
P=${1:-32768}
for CTK in q4_0 f16; do
    R=$(CUDA_VISIBLE_DEVICES=1 LLAMA_SM70_D256=0 $BIN -m "$MODEL" -fa 1 \
        -ctk $CTK -ctv f16 -p "$P" -n 0 -ngl 99 -r 1 -t 12 2>&1 \
        | grep -E "pp$P" | grep -oE '[0-9]+\.[0-9]+ ±' | grep -oE '[0-9]+\.[0-9]+')
    echo "stock ctk=$CTK: $R tok/s"
done
# and sm70 staged with both KV quantized (V dequant too)
R=$(CUDA_VISIBLE_DEVICES=1 LLAMA_SM70_D256_Q4_DIRECT=0 $BIN -m "$MODEL" -fa 1 \
    -ctk q4_0 -ctv q4_0 -p "$P" -n 0 -ngl 99 -r 1 -t 12 2>&1 \
    | grep -E "pp$P" | grep -oE '[0-9]+\.[0-9]+ ±' | grep -oE '[0-9]+\.[0-9]+')
echo "sm70-staged ctk=q4_0 ctv=q4_0: $R tok/s"
R=$(CUDA_VISIBLE_DEVICES=1 $BIN -m "$MODEL" -fa 1 \
    -ctk q4_0 -ctv q4_0 -p "$P" -n 0 -ngl 99 -r 1 -t 12 2>&1 \
    | grep -E "pp$P" | grep -oE '[0-9]+\.[0-9]+ ±' | grep -oE '[0-9]+\.[0-9]+')
echo "sm70-direct ctk=q4_0 ctv=q4_0: $R tok/s"
