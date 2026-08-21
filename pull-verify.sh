#!/usr/bin/env bash
# pull-verify.sh — 物理机 WSL2 用: git pull → 构建 → 全量 verify (含 32k 大 case)
# 用法: cd ~/sm70-attn && bash pull-verify.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== [1/3] git pull =="
if ! git pull --ff-only; then
    echo "直连失败, 用代理 ${GIT_PROXY:-http://127.0.0.1:7890} 重试..."
    GIT_HTTP_PROXY="${GIT_PROXY:-http://127.0.0.1:7890}" git pull --ff-only
fi
git log -1 --format='HEAD: %h  %s'

echo
echo "== [2/3] 构建 + [3/3] 全量 verify (full: 含 full-32k / posV-32k) =="
bash verify/run.sh full
