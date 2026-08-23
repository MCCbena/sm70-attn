#!/usr/bin/env bash
# mtmd + dflash repro: feed an image via /v1/chat/completions to the
# production stack (mmproj + --spec-type draft-dflash). Upstream issue
# ggml-org#27408: draft-cache position holes / M-RoPE rows ->
# llama_decode(ctx_dft) rc=-1 -> HTTP 500.
# Usage: mtmd_repro.sh [image] [n_requests]
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=build/bin/llama-server
MODEL=$HOME/models/qwen3.8-27b-ud-q4kxl/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMPROJ=$HOME/models/qwen3.8-27b-ud-q6k/mmproj-BF16.gguf
DRAFT=$HOME/models/qwen3.8-dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
IMG=${1:-$HOME/ik_llama.cpp/media/llama1-logo.png}
NREQ=${2:-1}

SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=1 "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --main-gpu 0 \
  -c 8192 -b 4096 -ub 512 -ngl 99 -fa on -ctk q4_0 -ctv f16 \
  --spec-type draft-dflash --spec-draft-model "$DRAFT" --spec-draft-n-max 2 \
  -t 12 -tb 16 --temp 0 --top-p 1 --top-k 1 -n 256 -np 1 \
  --host 127.0.0.1 --port 8080 > /tmp/mtmd_repro_server.log 2>&1 &
SRV_PID=$!
for _ in $(seq 1 200); do
  curl -sf http://127.0.0.1:8080/health > /dev/null && break
  kill -0 "$SRV_PID" 2>/dev/null || { tail -15 /tmp/mtmd_repro_server.log; exit 1; }
  sleep 2
done
echo "server ready; image=$IMG n=$NREQ"
python3 verify/mtmd_repro2.py "$IMG" "$NREQ"
echo "=== draft/mtmd errors in server log ==="
grep -iE 'rc=-1|pos_max|did not run|failed to (decode|encode)|draft' /tmp/mtmd_repro_server.log | grep -viE 'loading|load_|fitting|measure' | tail -8
