#!/usr/bin/env python3
# logit_ab.py — sm70 vs stock: greedy 首分叉 + 边际厚度测量
#
# 目的: 12/12 verify 测的是 sm70<->CPU 参考(算术地板)。
#       生产里真正对抗的是 stock 路径, 两者在 greedy 下的真实 logit 差
#       决定"每多少 token 翻一次 argmax"。 本脚本测这个。
#
# 用法(物理机 WSL2, ~/sm70-attn/ 下):
#   1) 用常规启动命令开 server, 确保 LLAMA_SM70_D256=1   (draft 模型可保留)
#      python3 logit_ab.py --phase A --server http://127.0.0.1:8080
#   2) 杀掉 server, 其它参数逐字不变, 只改 LLAMA_SM70_D256=0, 重开
#      python3 logit_ab.py --phase B --server http://127.0.0.1:8080
#   3) python3 logit_ab.py --compare
#
# 判读:
#   - no fork in N  -> 翻转率 < 1/N (Poisson 95% 上界 ~3/N)
#   - fork at i     -> [0,i) 区间的 top1-top2 边际分布说明薄到什么程度
#
# 注意:
#   - 每次 phase 必须对着**刚重启的** server (无残留 slot/KV)
#   - 请求内强制 temperature=0 + top_k=1, 与启动命令的 --temp 无关
#   - dflash draft 保留即可: 输出 token 仍是 target 的 greedy, draft 只影响速度
#   - 若 /completion 响应里没有 "tokens" 数组, 脚本会报错并打印响应片段,
#     把片段发回给天机改字段名

import json, os, random, sys, time, urllib.request
import argparse

TOP = 8          # 每个 token 要 top-8 logprob
SEED = 20260822  # 两次 phase 必须同种子 -> 同语料

WORDS = ("kernel attention softmax rescale tile vector cache decode prefill token "
         "margin argmax logit probe stride offset buffer layout head batch quant "
         "floor rounding residual flash paged kv f16 f32 ulp reduce accumulate "
         "greedy divergence drift context memory write read patch diff byte").split()


def build_corpus(turns):
    """确定性伪 agentic 语料: 任务简报 + N 轮'工具回显'。同种子同文本。"""
    rng = random.Random(SEED)
    def para(n):
        return " ".join(rng.choice(WORDS) for _ in range(n))
    brief = "task brief (seed %d):\n%s\n" % (SEED, para(900))
    chunks = ["turn %d tool echo:\n%s\n" % (t + 1, para(700)) for t in range(turns)]
    return brief, chunks


def post_completion(server, prompt, max_tokens):
    body = json.dumps({
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,               # 硬 greedy, 与 server 端 --temp 无关
        "max_tokens": max_tokens,
        "logprobs": TOP,
        "stream": False,
    }).encode()
    req = urllib.request.Request(server + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        data = json.loads(r.read())
    return data, time.time() - t0


def run_phase(server, phase, turns, decode, outdir):
    brief, chunks = build_corpus(turns)
    prompt = brief
    out = {"phase": phase, "turns": []}
    for t, chunk in enumerate(chunks):
        data, secs = post_completion(server, prompt, decode)
        toks = data["choices"][0].get("tokens")
        if not toks:
            # 字段名对不上时的自救: 打印片段供回传
            snippet = json.dumps(data)[:1500]
            raise SystemExit("ERROR: /completion 响应里没有 choices[0].tokens 数组。\n"
                             "响应片段(回传给天机改字段名):\n" + snippet)
        out["turns"].append({"turn": t + 1, "n": len(toks),
                             "secs": round(secs, 1), "tokens": toks})
        prompt += chunk
        print("  turn %2d: %5d tok in %7.1fs  (%.1f tok/s)"
              % (t + 1, len(toks), secs, len(toks) / max(secs, 1e-9)), flush=True)
    path = os.path.join(outdir, "logit_ab_%s.json" % phase)
    with open(path, "w") as f:
        json.dump(out, f)
    print("saved ->", path)


def flat_tokens(phase_data):
    out = []
    for turn in phase_data["turns"]:
        out.extend(turn["tokens"])
    return out


def top1(tok):
    if "text" in tok and tok["text"]:
        return tok["text"]
    tl = tok.get("top_logprobs") or []
    return tl[0]["text"] if tl else "?"


def margin_of(tok):
    tl = tok.get("top_logprobs") or []
    if len(tl) >= 2:
        return tl[0]["logprob"] - tl[1]["logprob"]
    return None


def compare(outdir):
    with open(os.path.join(outdir, "logit_ab_A.json")) as f:
        A = json.load(f)
    with open(os.path.join(outdir, "logit_ab_B.json")) as f:
        B = json.load(f)
    a, b = flat_tokens(A), flat_tokens(B)
    n = min(len(a), len(b))
    print("A tokens: %d   B tokens: %d   comparable: %d" % (len(a), len(b), n))

    fork, margins, marg_at_thin = None, [], []
    for i in range(n):
        ta, tb = a[i], b[i]
        if top1(ta) != top1(tb):
            fork = i
            break
        m = margin_of(ta)
        if m is not None:
            margins.append(m)

    print()
    if fork is None:
        print("== NO FORK in %d tokens ==" % n)
        print("   greedy 翻转率 < 1/%d  (Poisson 95%% 上界 ~ 3/%d = %.1e /token)"
              % (n, n, 3.0 / n))
        lo = ">= %d" % n
    else:
        print("== FIRST FORK at token %d / %d ==" % (fork, n))
        lo = str(fork)
    ms = sorted(margins)
    if ms:
        q = lambda p: ms[min(len(ms) - 1, int(p * len(ms)))]
        thin05 = sum(1 for m in ms if m < 0.05)
        thin01 = sum(1 for m in ms if m < 0.01)
        print("   首分叉前的 top1-top2 边际 (n=%d, nats):" % len(ms))
        print("     min=%.4f  p1=%.4f  p5=%.4f  median=%.4f  p95=%.4f"
              % (ms[0], q(0.01), q(0.05), q(0.5), q(0.95)))
        print("     边际 < 0.05: %d (%.2f%%)   < 0.01: %d (%.2f%%)"
              % (thin05, 100.0 * thin05 / len(ms), thin01, 100.0 * thin01 / len(ms)))
    print()
    print("== 判决参考 ==")
    print("   首分叉位置下界: token %s" % lo)
    if fork is None and n >= 20000:
        print("   2万+ token 不翻 -> 长 agentic 象限基本可用, 作品全救回")
    elif fork is not None and fork > 10000:
        print("   >1万 token 才翻 -> 介于两者之间, 可考虑路三(bit 对齐)或接受短会话用 ON")
    elif fork is not None:
        print("   <=1万 token 就翻 -> 长 agentic 象限死刑, 路一(分场景 env)就是终局")
    # 附: 提示 server 侧确认 sm70 真的接管了 prefill
    print()
    print("   [附] grep 'sm70' server log 确认 A 侧有 ACCEPT 行 / B 侧有 REJECT-env-disabled 行")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--turns", type=int, default=8, help="多轮 prefill 次数 (默认 8)")
    ap.add_argument("--decode", type=int, default=2560, help="每轮解码 token (默认 2560, 共 ~20k)")
    ap.add_argument("--outdir", default=".", help="结果目录")
    args = ap.parse_args()

    if args.compare:
        compare(args.outdir)
        return
    if not args.phase:
        ap.error("需要 --phase A|B 或 --compare")
    print("phase %s  server=%s  turns=%d decode=%d (total ~%d output tok)"
          % (args.phase, args.server, args.turns, args.decode, args.turns * args.decode))
    run_phase(args.server, args.phase, args.turns, args.decode, args.outdir)


if __name__ == "__main__":
    main()
