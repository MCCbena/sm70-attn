#!/usr/bin/env bash
# dump_cpu.sh — C 相: 纯 CPU + f32 V 的真值参照 (三相定责的裁判相).
#
# 用法 (WSL, ~/sm70-attn 下):   bash verify/dump_cpu.sh
# 前置: GPU 上没跑 dump_ab.sh (两者共用 8090 端口)
# 产物: /tmp/sm70_dump_cpu.bin
# 耗时: CPU 27B prefill ~10-20 分钟 (一次性成本, 换 0.69 的定责判决)

set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/bin/llama-server
MODEL=$HOME/models/qwen3.8-27b-ud-q4kxl/Qwen3.8-27B-UD-Q4_K_XL.gguf
PROMPT="${PROMPT:-verify/dump_prompt.txt}"
OUT=/tmp/sm70_dump_cpu.bin
LOG=/tmp/sm70_dump_cpu_server.log

[ -x "$BIN" ] || { echo "ERROR: $BIN 不存在"; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: model 不存在: $MODEL"; exit 1; }

if curl -sf http://127.0.0.1:8090/health > /dev/null 2>&1; then
  echo "!! 8090 端口已有 server 在跑 (dump_ab.sh?), 先等它结束"; exit 1
fi

rm -f "$OUT"
echo "--- starting CPU llama-server (-ngl 0 -ctv f32, SM70_DUMP=$OUT) ..."
# CPU 相不带 draft/mmproj: 它们的 fattn 记录被形状过滤, 不参与 chunk1 对齐,
# 省下 ~15GB 内存和加载时间.
CUDA_VISIBLE_DEVICES='' LLAMA_SM70_D256=1 SM70_DUMP=$OUT SM70_DUMP_MAX=160 \
"$BIN" -m "$MODEL" --main-gpu 0 -c 8192 -b 4096 -ub 512 -ngl 0 -fa on \
  -ctk q4_0 -ctv f32 -t 12 --temp 0 --top-p 1 --top-k 1 -n 8 -np 1 \
  --host 127.0.0.1 --port 8090 > "$LOG" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true; wait $SRV_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
  if curl -sf http://127.0.0.1:8090/health > /dev/null; then
    echo "    server ready (pid $SRV_PID)"
    break
  fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "ERROR: server died, log tail:"; tail -30 "$LOG"; exit 1
  fi
  sleep 2
done

echo "--- sending fixed prompt (CPU prefill, be patient ~10-20 min) ..."
python3 - "$PROMPT" <<'EOF'
import json, sys, urllib.request
prompt = open(sys.argv[1], encoding='utf-8').read()
body = json.dumps({"prompt": prompt, "temperature": 0.0, "top_k": 1,
                   "n_predict": 8, "stream": False}).encode()
req = urllib.request.Request("http://127.0.0.1:8090/completion", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=7200) as r:
    d = json.loads(r.read())
print("generated:", repr(str(d.get("content", ""))[:120]))
EOF

echo "--- stopping server ..."
kill "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
sleep 2
ls -la "$OUT"
echo "done. dump -> $OUT   (三相分析: python3 verify/sm70_layer_diff.py --a /tmp/sm70_dump_on.bin --b /tmp/sm70_dump_off.bin --c $OUT)"
