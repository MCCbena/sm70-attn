#!/usr/bin/env bash
# perf_ab.sh — 修复后 ON/OFF prefill 性能对照 (46k prompt)
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/bin/llama-server
MODEL=$HOME/models/qwen3.8-27b-ud-q4kxl/Qwen3.8-27B-UD-Q4_K_XL.gguf
DRAFT=$HOME/models/qwen3.8-dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
PROMPT=bench/prompt_46k.txt

SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

run_phase() { # $1 = sm70 (1|0), $2 = log
  echo "--- phase sm70=$1 ..."
  CUDA_VISIBLE_DEVICES=1 LLAMA_SM70_D256=$1 \
  "$BIN" -m "$MODEL" --main-gpu 0 -c 65536 -b 4096 -ub 512 \
    -ngl 99 -fa on -ctk q4_0 -ctv f16 \
    --spec-type draft-dflash --spec-draft-model "$DRAFT" --spec-draft-n-max 2 \
    -t 12 -tb 16 --temp 0 --top-p 1 --top-k 1 -n 8 -np 1 \
    --host 127.0.0.1 --port 8090 > "$2" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 150); do
    curl -sf http://127.0.0.1:8090/health > /dev/null && break
    kill -0 "$SRV_PID" 2>/dev/null || { tail -15 "$2"; exit 1; }
    sleep 2
  done
  python3 - "$PROMPT" <<'EOF'
import json, sys, urllib.request
prompt = open(sys.argv[1], encoding='utf-8').read()
body = json.dumps({'prompt': prompt, 'temperature': 0.0, 'top_k': 1,
                   'n_predict': 8, 'stream': False}).encode()
req = urllib.request.Request('http://127.0.0.1:8090/completion', data=body,
                             headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=3600) as r:
    json.loads(r.read())
print('prompt done')
EOF
  kill "$SRV_PID" 2>/dev/null || true
  wait "$SRV_PID" 2>/dev/null || true
  SRV_PID=""
  sleep 3
}

run_phase 1 /tmp/perf_on.log
run_phase 0 /tmp/perf_off.log

echo "=== ON  (sm70, fixed) ==="
grep -a "prompt eval" /tmp/perf_on.log | tail -1
echo "=== OFF (stock) ==="
grep -a "prompt eval" /tmp/perf_off.log | tail -1
