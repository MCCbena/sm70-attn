#!/usr/bin/env bash
# logit_ab_fixed.sh — 修复后的 logit_ab 三相判决 (A/A2/B + compare)
# 用法: bash verify/logit_ab_fixed.sh   (GPU 空闲时; 全程约 50 分钟)
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/bin/llama-server
MODEL=$HOME/models/qwen3.8-27b-ud-q4kxl/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMPROJ=$HOME/models/qwen3.8-27b-ud-q6k/mmproj-BF16.gguf
DRAFT=$HOME/models/qwen3.8-dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf

SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

start_server() { # $1 = sm70 (1|0), $2 = log
  echo "--- starting server (LLAMA_SM70_D256=$1) ..."
  CUDA_VISIBLE_DEVICES=1 LLAMA_SM70_D256=$1 \
  "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --main-gpu 0 -c 229376 -b 4096 -ub 512 \
    -ngl 99 -fa on -ctk q4_0 -ctv f16 \
    --spec-type draft-dflash --spec-draft-model "$DRAFT" --spec-draft-n-max 2 \
    -t 12 -tb 16 --temp 0 --top-p 1 --top-k 1 -n 20480 -np 1 \
    --host 127.0.0.1 --port 8080 > "$2" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 200); do
    curl -sf http://127.0.0.1:8080/health > /dev/null && { echo "    ready (pid $SRV_PID)"; return 0; }
    kill -0 "$SRV_PID" 2>/dev/null || { tail -15 "$2"; exit 1; }
    sleep 2
  done
  echo "ERROR: server not ready"; tail -15 "$2"; exit 1
}

echo "=== phase A (sm70 ON, fixed) ==="
start_server 1 /tmp/logitab_on_server.log
python3 logit_ab.py --phase A
echo
echo "=== phase A2 (ON control) ==="
python3 logit_ab.py --phase A2
kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; SRV_PID=""
sleep 5

echo
echo "=== phase B (stock OFF) ==="
start_server 0 /tmp/logitab_off_server.log
python3 logit_ab.py --phase B
kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; SRV_PID=""
sleep 5

echo
echo "=== verdict ==="
python3 logit_ab.py --compare 2>&1 | tee /tmp/logitab_verdict.txt
