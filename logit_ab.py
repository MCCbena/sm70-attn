#!/usr/bin/env python3
# logit_ab.py v3 — sm70 vs stock: 逐轮前缀对齐找首个 greedy 分叉
#
# v3 为什么重写 (v2 的判决无效):
#   1) v2 每相开跑前发 4-token 探针, 探针的输出 token 已进 slot 上下文,
#      污染了 turn1 的起点 -> 平铺比较在 token0 必假分叉
#   2) v2 把 8 轮 token 平铺成一条流比较, 但 A/B 的 turn 长度不同,
#      平铺对齐从第一个长度差起就错位 -> 所有 fork 位置不可信
#   v3: 每轮自包含 (turn t 的 prompt 只依赖固定种子, 不依赖上一轮输出),
#       逐轮在各自上下文内对齐找第一个分叉, 位置 = 该轮上下文内的 token 序
#
# 流程 (物理机 WSL2, ~/sm70-attn/):
#   1) git pull
#   2) server 1: LLAMA_SM70_D256=1 (其余参数不变)
#        python3 logit_ab.py --phase A
#        python3 logit_ab.py --phase A2     <- 确定性自检 (同一 server, 必跑)
#   3) 关 server, 只改 env LLAMA_SM70_D256=0, 重起
#        python3 logit_ab.py --phase B
#   4) python3 logit_ab.py --compare
#
# 判读顺序:
#   先 A vs A2 (控制): 全部 NO FORK -> server greedy 确定性成立, A/B 结论有效
#   有 FORK  -> server 本身不确定 (kernel/CPU 路径), A/B 无意义, 直接报告
#   再 A vs B: fork 出现得越早, sm70 非恒等的影响越大

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

# server 是本机/局域网地址, 永远不走系统代理 (Clash 等 env 代理会把它劫持成 502)
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost,26.99.171.98,0.0.0.0"

TOP = 8
SEED = 20260822
DEFAULT_SERVER = "http://127.0.0.1:8080"
WORDS = ("kernel attention softmax rescale tile vector cache decode prefill token "
         "margin argmax logit probe stride offset buffer layout head batch quant "
         "floor rounding residual flash paged kv f16 f32 ulp reduce accumulate "
         "greedy divergence drift context memory write read patch diff byte").split()


def build_corpus(turns):
    rng = random.Random(SEED)
    def para(n):
        return " ".join(rng.choice(WORDS) for _ in range(n))
    brief = "task brief (seed %d):\n%s\n" % (SEED, para(900))
    chunks = ["turn %d tool echo:\n%s\n" % (t + 1, para(700)) for t in range(turns)]
    return brief, chunks


def post(server, path, body_bytes, timeout=7200):
    req = urllib.request.Request(server + path, data=body_bytes,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), time.time() - t0
    except urllib.error.URLError as e:
        print("ERROR: 连不上 server (%s)" % e.reason)
        print("       确认 llama-server 已启动并监听 8080 后重跑本 phase")
        sys.exit(9)
    except Exception as e:
        print("ERROR: 请求失败 %s %s -> %r" % (server, path, e))
        sys.exit(9)


def completion(server, prompt, max_tokens, logprobs=TOP):
    body = json.dumps({
        "prompt": prompt, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
        "max_tokens": max_tokens, "logprobs": logprobs, "stream": False,
    }).encode()
    return post(server, "/completion", body)


def extract_tokens(data):
    """返回 [{'text','logprob','top':[('text',lp),...]}, ...] 或 None。
    兼容三种已知响应形状, 未知形状返回 None 由调用方留档。"""
    # 形状1: OpenAI 兼容 choices[0].tokens
    ch = data.get("choices")
    if isinstance(ch, list) and ch and isinstance(ch[0].get("tokens"), list):
        out = []
        for tk in ch[0]["tokens"]:
            tops = [(t.get("text"), t.get("logprob")) for t in (tk.get("top_logprobs") or [])][:TOP]
            out.append({"text": tk.get("text", "?"),
                        "logprob": (tops[0][1] if tops else tk.get("logprob")),
                        "top": tops})
        return out
    # 形状2: 原生 llama.cpp /completion: completion_probabilities[]
    cpl = data.get("completion_probabilities")
    if isinstance(cpl, list) and cpl:
        out = []
        for e in cpl:
            tops = [(t.get("token"), t.get("logprob")) for t in (e.get("top_logprobs") or [])][:TOP]
            out.append({"text": e.get("token", "?"),
                        "logprob": e.get("logprob"),
                        "top": tops})
        return out
    # 形状3: 原生 llama.cpp: data["tokens"] + data["logprobs"]
    if isinstance(data.get("tokens"), list) and isinstance(data.get("logprobs"), list):
        toks, lps = data["tokens"], data["logprobs"]
        def _t(x):
            if isinstance(x, dict):
                v = x.get("token", x.get("text"))
                return v if isinstance(v, str) else "?"
            return x if isinstance(x, str) else "?"
        out = []
        for i, tk in enumerate(toks):
            entry = lps[i] if i < len(lps) else {}
            inner = entry.get("tokens") or entry.get("top_logprobs") or []
            tops = [(_t(x), x.get("logprob") if isinstance(x, dict) else None) for x in inner[:TOP]]
            out.append({"text": _t(tk), "logprob": entry.get("logprob"), "top": tops})
        return out
    return None


def run_phase(server, phase, turns, decode, outdir):
    print("=" * 64)
    if phase == "A2":
        print("PHASE %s   (同一 server 上直接跑, 不要重启 — 这是确定性控制组)" % phase)
    else:
        print("PHASE %s   (开跑前确认: server 刚启动, 且本轮启动期间没发过其他请求)" % phase)
    print("=" * 64, flush=True)
    if not (server.startswith("http://") or server.startswith("https://")):
        print("ERROR: server 地址不合法: %r (不要加 --server, 默认就是 %s)"
              % (server, DEFAULT_SERVER))
        sys.exit(4)
    brief, chunks = build_corpus(turns)
    prompt = brief
    out = {"phase": phase, "server": server, "seed": SEED,
           "start_wall": time.time(), "turns": []}
    for t, chunk in enumerate(chunks):
        data, secs = completion(server, prompt, decode)
        toks = extract_tokens(data)
        if not toks:
            print("ERROR: turn %d 响应无法解析, 留档 logit_ab_%s_badturn%d.json:" % (t + 1, phase, t + 1))
            with open(os.path.join(outdir, "logit_ab_%s_badturn%d.json" % (phase, t + 1)), "w") as f:
                json.dump(data, f, indent=1)
            print("响应顶层 keys:", list(data.keys()))
            print("片段:", json.dumps(data)[:800])
            sys.exit(5)
        out["turns"].append({"turn": t + 1, "n": len(toks),
                             "secs": round(secs, 1), "tokens": toks})
        prompt += chunk
        print("  turn %2d: %5d tok in %7.1fs  (%.1f tok/s)"
              % (t + 1, len(toks), secs, len(toks) / max(secs, 1e-9)), flush=True)
        # 每轮落盘: 中途崩溃不丢已完成的轮 (上版只存结尾, turn8 后崩 = 白跑 15 分钟)
        out["end_wall"] = time.time()
        with open(os.path.join(outdir, "logit_ab_%s.json" % phase), "w") as f:
            json.dump(out, f)
    out["end_wall"] = time.time()
    path = os.path.join(outdir, "logit_ab_%s.json" % phase)
    with open(path, "w") as f:
        json.dump(out, f)
    print("saved ->", path)
    # 探针放在 phase 结束: 只为确认连接/形状 (输出会进 slot, 绝不能放在 phase 前)
    try:
        data, secs = completion(server, "probe line one two three", 4, TOP)
        with open(os.path.join(outdir, "logit_ab_%s_probe.json" % phase), "w") as f:
            json.dump(data, f, indent=1)
        pt = extract_tokens(data)
        if not pt or not pt[0]["top"]:
            print("!! 事后探针: token 或 top_logprobs 取不到 (本 phase 数据若正常可忽略), 已留档")
        else:
            print("  事后探针 OK: %d tok, top1=%r (%.1fs)"
                  % (len(pt), pt[0]["top"][0][0], secs))
    except SystemExit:
        print("  事后探针连不上 server (phase 数据已保存, 可忽略)")
    nxt = {"A": "A2", "A2": "B", "B": None}[phase]
    if nxt == "A2":
        print("下一步: 同一 server 上直接再跑  python3 logit_ab.py --phase A2  (确定性自检)")
    elif nxt:
        print("下一步: 关掉 server, 只把 env 改成 LLAMA_SM70_D256=0, 重起后跑 --phase %s" % nxt)
    else:
        print("最后: python3 logit_ab.py --compare")


def load(outdir, phase):
    path = os.path.join(outdir, "logit_ab_%s.json" % phase)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def top1(tok):
    return tok.get("text") or "?"


def margin_of(tok):
    tops = tok.get("top") or []
    if len(tops) >= 2 and tops[0][1] is not None and tops[1][1] is not None:
        return tops[0][1] - tops[1][1]
    return None


def compare_turns(ta, tb):
    """ta/tb: 同一轮上下文两侧的输出 token 列表。
    返回 (fork_idx, margins) — fork 前(含全部) token 的边际。"""
    n = min(len(ta), len(tb))
    margins, fork = [], None
    for i in range(n):
        if top1(ta[i]) != top1(tb[i]):
            fork = i
            break
        m = margin_of(ta[i])
        if m is not None:
            margins.append(m)
    return fork, margins


def compare_pair(outdir, p1, p2, label):
    A, B = load(outdir, p1), load(outdir, p2)
    if not A or not B:
        print("[%s] 缺文件 (logit_ab_%s.json / logit_ab_%s.json), 跳过" % (label, p1, p2))
        return None
    print("=" * 64)
    print("对比 %s  (A: %d tok, B: %d tok)"
          % (label, sum(t["n"] for t in A["turns"]), sum(t["n"] for t in B["turns"])))
    print("=" * 64)
    forks, all_margins = [], []
    for ta, tb in zip(A["turns"], B["turns"]):
        f, ms = compare_turns(ta["tokens"], tb["tokens"])
        all_margins.extend(ms)
        if f is None:
            print("  turn %d: NO FORK  (%d/%d tok 完全一致)" % (ta["turn"], len(ta["tokens"]), len(tb["tokens"])))
        else:
            forks.append((ta["turn"], f, ta["tokens"][f], tb["tokens"][f]))
            print("  turn %d: FORK at token %d / %d" % (ta["turn"], f, len(ta["tokens"])))
            for side, tk in (("p1", ta["tokens"][f]), ("p2", tb["tokens"][f])):
                tops = tk.get("top") or []
                if tops and tops[0][1] is not None:
                    m = margin_of(tk)
                    print("    %s: top1=%r lp=%.5f  top2=%r lp=%.5f  margin=%s"
                          % (side, tops[0][0], tops[0][1],
                             tops[1][0] if len(tops) > 1 else None,
                             tops[1][1] if len(tops) > 1 else None,
                             ("%.5f" % m) if m is not None else None))
                else:
                    print("    %s: top1=%r (top_logprobs 空)" % (side, tk.get("text")))
    ms = sorted(m for m in all_margins if m is not None)
    if ms:
        q = lambda p: ms[min(len(ms) - 1, int(p * len(ms)))]
        thin05 = sum(1 for m in ms if m < 0.05)
        thin01 = sum(1 for m in ms if m < 0.01)
        print("  分叉前 top1-top2 边际 (n=%d, nats):" % len(ms))
        print("    min=%.4f  p1=%.4f  p5=%.4f  median=%.4f  p95=%.4f"
              % (ms[0], q(0.01), q(0.05), q(0.5), q(0.95)))
        print("    边际 < 0.05: %d (%.2f%%)   < 0.01: %d (%.2f%%)"
              % (thin05, 100.0 * thin05 / len(ms), thin01, 100.0 * thin01 / len(ms)))
    return forks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "A2", "B"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--server", default=DEFAULT_SERVER,
                    help="默认 http://127.0.0.1:8080, 一般不要改")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--decode", type=int, default=2560)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    if args.compare:
        A = load(args.outdir, "A")
        if not A or not load(args.outdir, "B"):
            print("ERROR: 需要 logit_ab_A.json 和 logit_ab_A2.json 和 logit_ab_B.json (各 phase 成功后生成)")
            sys.exit(6)
        # 种子一致性
        phases = [load(args.outdir, p) for p in ("A", "A2", "B")]
        if len({p.get("seed") for p in phases}) != 1:
            print("!! 警告: 各相种子不同, 语料不一致, 对比无效。")
            sys.exit(8)
        # 重启守卫: 用与 B 相邻的那一相 (A2, 缺则 A) 的 end_wall
        a_end = (phases[1] or phases[0]).get("end_wall")
        b_start = phases[2].get("start_wall")
        if a_end and b_start:
            gap = b_start - a_end
            if abs(gap) < 120:
                print("!! A2 结束到 B 开始仅相隔 %.0f 秒 (每相实际 ~15 分钟)。" % gap)
                print("!! 说明 B 相 server 没有重启 —— KV cache 跨相残留, 对比无效。")
                print("!! 重启 server 后重跑 --phase B。")
                sys.exit(7)
            print("(A2 结束→B 开始: %.1f 秒, 通过重启守卫)" % gap)
        print()
        ctrl = compare_pair(args.outdir, "A", "A2", "A vs A2  [确定性控制: 同一 server, 必须全 NO FORK]")
        print()
        main_f = compare_pair(args.outdir, "A", "B", "A vs B  [sm70 ON vs OFF]")
        print()
        print("=" * 64)
        print("判决")
        print("=" * 64)
        if ctrl is None:
            print("[控制] 缺 A2 — A/B 结论未排除 server 自身不确定性, 可信度打折")
        elif ctrl:
            first = ctrl[0]
            print("[控制] !! A vs A2 (同一 server) 出现 FORK: turn %d token %d" % (first[0], first[1]))
            print("[控制] !! server greedy 本身不确定 -> A/B 对比无意义, 把本段发回给天机")
        else:
            print("[控制] A vs A2 全 NO FORK -> server greedy 确定性成立, A/B 结论有效")
        print()
        if main_f is None:
            print("[A/B] 缺文件")
        elif not main_f:
            ntok = sum(t["n"] for t in A["turns"])
            print("[A/B] %d token 零分叉" % ntok)
            print("[A/B]   -> 该语料下 sm70 ON 与 OFF 的 greedy 序列逐 token 相同")
            print("[A/B]   -> 污染不是'每次 prefill 必翻'; 触发需要特定上下文 (更窄 gap / 更长自引用)")
            print("[A/B]   -> 可加大 --turns/--decode 或换真实任务语料再压")
        else:
            first = main_f[0]
            n_fork = len(main_f)
            print("[A/B] %d 轮中 %d 轮分叉, 最早 turn %d token %d"
                  % (len(A["turns"]), n_fork, first[0], first[1]))
            if first[1] < 1000:
                print("[A/B]   -> 短上下文前段就翻: 非长度累积, 与 585eeb '第6条消息就漂' 一致")
            else:
                print("[A/B]   -> 分叉出现在较长前缀: 偏累积型")
            print("[A/B]   -> 分叉轮之后各轮上下文已分岔, 其 fork 位置无独立意义, 只看最早的")
        print()
        print("   [附] grep -c sm70 两侧 server log: ON 侧应有 ACCEPT 行, OFF 侧 REJECT-env-disabled 行")
        if ctrl:
            sys.exit(10)
        return
    if not args.phase:
        ap.error("需要 --phase A|A2|B 或 --compare")
    print("phase %s  turns=%d decode=%d (total ~%d output tok)"
          % (args.phase, args.turns, args.decode, args.turns * args.decode))
    print("(预计 ~15 分钟)", flush=True)
    run_phase(args.server, args.phase, args.turns, args.decode, args.outdir)


if __name__ == "__main__":
    main()
