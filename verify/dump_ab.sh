#!/usr/bin/env bash
# dump_ab.sh — 成因观测: ON/OFF 两个 server 各发同一条固定 prompt, 抓每层
# attention 输出 (SM70_DUMP), 逐层对比. 8/23 复查报告第五节 E1 的自动化版.
#
# 用法 (WSL, ~/sm70-attn 下):   bash verify/dump_ab.sh
#   CT_V=q4_0 bash verify/dump_ab.sh   # V 走 dequant 路径的变体 (A4 验证)
# 前置:
#   1. GPU 空闲 — 先关掉生产 llama-server (脚本会检查并拒绝运行)
#   2. build/bin/llama-server 已编译到本 commit (cmake --build build)
#   3. python3 + numpy
# 产物: /tmp/sm70_dump_on.bin /tmp/sm70_dump_off.bin /tmp/sm70_layer_diff.csv
#       + 终端的逐层判决输出
#
# 全程约 6 分钟 (两次 server 启动各 ~2 分钟 + 两次 prompt + 分析).

set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/bin/llama-server
MODEL=$HOME/models/qwen3.8-27b-ud-q4kxl/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMPROJ=$HOME/models/qwen3.8-27b-ud-q6k/mmproj-BF16.gguf
DRAFT=$HOME/models/qwen3.8-dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
PROMPT=verify/dump_prompt.txt
PORT=8090
ON=/tmp/sm70_dump_on.bin
OFF=/tmp/sm70_dump_off.bin

[ -x "$BIN" ] || { echo "ERROR: $BIN 不存在, 先: cmake --build build -j"; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: model 不存在: $MODEL"; exit 1; }
[ -f "$PROMPT" ] || { echo "ERROR: $PROMPT 不存在"; exit 1; }

# python 选择: 优先 conda (带 numpy), 退回 python3
PY=python3
for cand in "$HOME/miniconda3/bin/python" "$HOME/anaconda3/bin/python"; do
    if [ -x "$cand" ] && "$cand" -c "import numpy" 2>/dev/null; then
        PY="$cand"
        break
    fi
done
"$PY" -c "import numpy" 2>/dev/null || { echo "ERROR: numpy 不可用 ($PY)"; exit 1; }
echo "python: $PY"

# GPU 必须空闲
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
  echo "!! GPU 上有进程在跑 (生产 llama-server?), 先关掉再跑本脚本:"
  nvidia-smi --query-compute-apps=pid,process_name --format=csv
  exit 1
fi

SRV_PID=""
cleanup() {
  if [ -n "$SRV_PID" ]; then
    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

start_server() { # $1 = LLAMA_SM70_D256 (1|0), $2 = dump path
  rm -f "$2"
  echo "--- starting llama-server (LLAMA_SM70_D256=$1, SM70_DUMP=$2) ..."
  CUDA_VISIBLE_DEVICES=1 LLAMA_SM70_D256=$1 SM70_DUMP=$2 SM70_DUMP_MAX=256 \
  "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --main-gpu 0 -c 8192 -b 4096 -ub 512 \
    -ngl 99 -fa on -ctk q4_0 -ctv "${CT_V:-f16}" \
    --spec-type draft-dflash --spec-draft-model "$DRAFT" --spec-draft-n-max 2 \
    -t 12 -tb 16 --temp 0 --top-p 1 --top-k 1 -n 8 -np 1 \
    --host 127.0.0.1 --port $PORT > /tmp/sm70_dump_server.log 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 150); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
      echo "    server ready (pid $SRV_PID)"
      return 0
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "ERROR: server died, log tail:"
      tail -30 /tmp/sm70_dump_server.log
      exit 1
    fi
    sleep 2
  done
  echo "ERROR: server not ready in 300s"
  tail -30 /tmp/sm70_dump_server.log
  exit 1
}

stop_server() {
  echo "--- stopping server ..."
  kill "$SRV_PID" 2>/dev/null || true
  wait "$SRV_PID" 2>/dev/null || true
  SRV_PID=""
  sleep 3
}

send_prompt() {
  echo "--- sending fixed prompt (greedy, 8 tokens) ..."
  python3 - "$PROMPT" <<'EOF'
import json, sys, urllib.request
prompt = open(sys.argv[1], encoding='utf-8').read()
body = json.dumps({"prompt": prompt, "temperature": 0.0, "top_k": 1,
                   "n_predict": 8, "stream": False}).encode()
req = urllib.request.Request("http://127.0.0.1:8090/completion", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    d = json.loads(r.read())
text = d.get("content", "")
if isinstance(text, list):
    text = "".join(t.get("text", "") for t in text if isinstance(t, dict))
print("    generated:", repr(text[:120]))
EOF
}

echo "=== phase A: sm70 ON ==="
start_server 1 "$ON"
send_prompt
stop_server

echo
echo "=== phase B: sm70 OFF ==="
start_server 0 "$OFF"
send_prompt
stop_server

echo
echo "=== layerwise diff analysis ==="
ls -la "$ON" "$OFF" || true
"$PY" verify/sm70_layer_diff.py --a "$ON" --b "$OFF" --csv /tmp/sm70_layer_diff.csv
echo
echo "done. csv -> /tmp/sm70_layer_diff.csv   dumps -> $ON $OFF"
echo "(server log: /tmp/sm70_dump_server.log)"
