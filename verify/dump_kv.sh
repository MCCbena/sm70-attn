#!/usr/bin/env bash
# dump_kv.sh — 抓第一次 sm70 调用的 kernel 输入 (K dequant 后 / V 原生 / Qs)
# 用法: bash verify/dump_kv.sh   (GPU 空闲时; 约 3 分钟)
# 产物: /tmp/sm70_kv.bin
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/bin/llama-server
MODEL=$HOME/models/qwen3.8-27b-ud-q4kxl/Qwen3.8-27B-UD-Q4_K_XL.gguf
DRAFT=$HOME/models/qwen3.8-dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
PROMPT=verify/dump_prompt.txt
OUT=/tmp/sm70_kv.bin

[ -x "$BIN" ] || { echo "ERROR: $BIN 不存在"; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
  echo "!! GPU 被占用, 先关 server"; exit 1
fi

rm -f "$OUT"
SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

echo "--- starting ON server (SM70_DUMP_KV=$OUT) ..."
CUDA_VISIBLE_DEVICES=1 LLAMA_SM70_D256=1 SM70_DUMP_KV=$OUT \
"$BIN" -m "$MODEL" --main-gpu 0 -c 8192 -b 4096 -ub 512 \
  -ngl 99 -fa on -ctk q4_0 -ctv f16 \
  --spec-type draft-dflash --spec-draft-model "$DRAFT" --spec-draft-n-max 2 \
  -t 12 -tb 16 --temp 0 --top-p 1 --top-k 1 -n 8 -np 1 \
  --host 127.0.0.1 --port 8090 > /tmp/sm70_dump_kv_server.log 2>&1 &
SRV_PID=$!
for _ in $(seq 1 150); do
  curl -sf http://127.0.0.1:8090/health > /dev/null && { echo "    ready"; break; }
  kill -0 "$SRV_PID" 2>/dev/null || { tail -20 /tmp/sm70_dump_kv_server.log; exit 1; }
  sleep 2
done

python3 - "$PROMPT" <<'EOF'
import json, sys, urllib.request
prompt = open(sys.argv[1], encoding='utf-8').read()
body = json.dumps({'prompt': prompt, 'temperature': 0.0, 'top_k': 1,
                   'n_predict': 8, 'stream': False}).encode()
req = urllib.request.Request('http://127.0.0.1:8090/completion', data=body,
                             headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=1800) as r:
    json.loads(r.read())
print('prompt done')
EOF

kill "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
sleep 2
ls -la "$OUT"
echo "done. -> $OUT"
