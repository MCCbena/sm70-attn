#!/bin/bash
# q4-direct A/B/C: direct vs staged vs stock-off, long-context prefill.
BIN=~/sm70-attn/build/bin/llama-bench
MODEL=~/models/qwen3.8-27b-ud-q4kxl/Qwen3.8-27B-UD-Q4_K_XL.gguf
P=${1:-32768}
for MODE in direct staged off; do
    case $MODE in
        direct) unset LLAMA_SM70_D256_Q4_DIRECT; unset LLAMA_SM70_D256;;
        staged) export LLAMA_SM70_D256_Q4_DIRECT=0;  unset LLAMA_SM70_D256;;
        off)    export LLAMA_SM70_D256=0;;
    esac
    R=$(CUDA_VISIBLE_DEVICES=1 $BIN -m "$MODEL" -fa 1 -ctk q4_0 -ctv f16 \
        -p "$P" -n 0 -ngl 99 -r 1 -t 12 2>&1 \
        | grep -E "pp$P" | grep -oE '[0-9]+\.[0-9]+ ±' | grep -oE '[0-9]+\.[0-9]+')
    echo "$MODE: $R tok/s"
done
