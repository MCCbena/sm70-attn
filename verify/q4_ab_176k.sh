#!/bin/bash
# 176k single-shot A/B: q4-direct vs staged. The decisive long-context
# number (attention's share of prefill is largest here).
BIN=~/sm70-attn/build/bin/llama-bench
MODEL=~/models/qwen3.8-27b-ud-q4kxl/Qwen3.8-27B-UD-Q4_K_XL.gguf
for MODE in ${@:-direct staged}; do
    case $MODE in
        direct) unset LLAMA_SM70_D256_Q4_DIRECT;;
        staged) export LLAMA_SM70_D256_Q4_DIRECT=0;;
    esac
    T0=$(date +%s)
    R=$(CUDA_VISIBLE_DEVICES=1 $BIN -m "$MODEL" -fa 1 -ctk q4_0 -ctv f16 \
        -p 176000 -n 0 -ngl 99 -r 1 -t 12 2>&1 \
        | grep -E "pp176000" | grep -oE '[0-9]+\.[0-9]+ ±' | grep -oE '[0-9]+\.[0-9]+')
    T1=$(date +%s)
    echo "$MODE: $R tok/s  (wall $((T1-T0))s)"
done
