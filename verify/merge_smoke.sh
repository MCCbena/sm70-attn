#!/usr/bin/env bash
# merge smoke: full production stack (qwen3.8 + mmproj + dflash spec + sm70)
# after the upstream merge. One greedy completion; assert 200 + non-empty.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=build/bin/llama-server
MODEL=$HOME/models/qwen3.8-27b-ud-q4kxl/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMPROJ=$HOME/models/qwen3.8-27b-ud-q6k/mmproj-BF16.gguf
DRAFT=$HOME/models/qwen3.8-dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=1 "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --main-gpu 0 \
  -c 8192 -b 4096 -ub 512 -ngl 99 -fa on -ctk q4_0 -ctv f16 \
  --spec-type draft-dflash --spec-draft-model "$DRAFT" --spec-draft-n-max 2 \
  -t 12 -tb 16 --temp 0 --top-p 1 --top-k 1 -n 256 -np 1 \
  --host 127.0.0.1 --port 8080 > /tmp/merge_smoke_server.log 2>&1 &
SRV_PID=$!
for _ in $(seq 1 200); do
  curl -sf http://127.0.0.1:8080/health > /dev/null && break
  kill -0 "$SRV_PID" 2>/dev/null || { tail -15 /tmp/merge_smoke_server.log; exit 1; }
  sleep 2
done
python3 - <<'EOF'
import json, urllib.request
prompt = open("verify/dump_prompt3k.txt", encoding="utf-8").read()[:8000]
body = json.dumps({"prompt": prompt, "n_predict": 48, "temperature": 0,
                   "top_p": 1, "top_k": 1}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/completion", data=body,
                             headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=600).read())
txt = d.get("content", "")
assert len(txt) > 20, f"empty/malformed completion: {d}"
print(f"SMOKE OK: {len(txt)} chars generated; head: {txt[:80]!r}")
EOF
grep -c 'ACCEPT: sm70 d256' /tmp/merge_smoke_server.log || true
