#!/usr/bin/env python3
# logit_ab.py v2 — sm70 vs stock: greedy 首分叉 + 边际厚度测量
#
# 流程(物理机 WSL2, ~/sm70-attn/):
#   1) git pull
#   2) 用常规启动命令开 server (LLAMA_SM70_D256=1, 其余参数不变)
#      python3 logit_ab.py --phase A        (不要加任何 --server 参数)
#   3) 杀掉 server, 只把 env 改成 LLAMA_SM70_D256=0, 重新启动
#      python3 logit_ab.py --phase B
#   4) python3 logit_ab.py --compare
#
# 判读: no fork in N -> 翻转率 < 1/N (95% 上界 ~3/N)
#
# 铁律: 每个 phase 必须对着刚重启的 server (KV cache 不能跨相残留)
# 请求内强制 temperature=0 + top_k=1, 与 server 端 --temp 无关;
# dflash draft 保留即可 (输出 token 仍是 target 的 greedy)

import json, os, random, sys, time, urllib.request, urllib.error
import argparse

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
        print("ERROR: 连不上 server %s (%s)" % (server, e.reason))
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
    兼容两种已知响应形状 + 未知形状留档。"""
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
    # 形状3: 原生 llama.cpp /completion: completion_probabilities[] (2026-08-22 实测生产 server)
    cpl = data.get("completion_probabilities")
    if isinstance(cpl, list) and cpl:
        out = []
        for e in cpl:
            tops = [(t.get("token"), t.get("logprob")) for t in (e.get("top_logprobs") or [])][:TOP]
            out.append({"text": e.get("token", "?"),
                        "logprob": e.get("logprob"),
                        "top": tops})
        return out
    # 形状4: 原生 llama.cpp: data["tokens"] + data["logprobs"]
    if isinstance(data.get("tokens"), list) and isinstance(data.get("logprobs"), list):
        toks, lps = data["tokens"], data["logprobs"]
        out = []
        for i, tk in enumerate(toks):
            entry = lps[i] if i < len(lps) else {}
            def _t(x):
                if isinstance(x, dict):
                    v = x.get("token", x.get("text"))
                    return v if isinstance(v, str) else "?"
                return x if isinstance(x, str) else "?"
            inner = entry.get("tokens") or entry.get("top_logprobs") or []
            tops = [(_t(x), x.get("logprob") if isinstance(x, dict) else None) for x in inner[:TOP]]
            out.append({"text": _t(tk),
                        "logprob": entry.get("logprob"),
                        "top": tops})
        return out
    return None


def probe_and_validate(server, phase):
    """小探针: 确认 server 活着 + 响应形状可解析, 留档。"""
    print("  预检探针 (max_tokens=4)...", flush=True)
    data, secs = completion(server, "probe line one two three", 4, TOP)
    with open("logit_ab_%s_probe.json" % phase, "w") as f:
        json.dump(data, f, indent=1)
    toks = extract_tokens(data)
    if not toks:
        print("ERROR: 响应形状无法解析。已留档 logit_ab_%s_probe.json," % phase)
        print("       把该文件内容 + server 版本 (启动 log 第一行) 发回给天机。")
        print("响应顶层 keys:", list(data.keys()))
        print("片段:", json.dumps(data)[:600])
        sys.exit(2)
    sample = toks[0]
    if not sample["top"]:
        print("ERROR: 能取到 token 但取不到 top_logprobs (logprobs 未生效?)。")
        print("响应顶层 keys:", list(data.keys()))
        print("片段:", json.dumps(data)[:600])
        sys.exit(3)
    print("  探针 OK: %d tok, top1=%r top2=%r (%.1fs)"
          % (len(toks), sample["top"][0][0] if sample["top"] else "?",
             sample["top"][1][0] if len(sample["top"]) > 1 else "?", secs))


def run_phase(server, phase, turns, decode, outdir):
    print("=" * 64)
    print("PHASE %s  (server 必须刚重启: %s)" % (phase, server))
    print("=" * 64, flush=True)
    if not (server.startswith("http://") or server.startswith("https://")):
        print("ERROR: server 地址不合法: %r (不要加 --server, 默认就是 %s)"
              % (server, DEFAULT_SERVER))
        sys.exit(4)
    probe_and_validate(server, phase)
    brief, chunks = build_corpus(turns)
    prompt = brief
    out = {"phase": phase, "server": server, "seed": SEED,
           "start_wall": time.time(), "turns": []}
    for t, chunk in enumerate(chunks):
        data, secs = completion(server, prompt, decode)
        toks = extract_tokens(data)
        if not toks:
            print("ERROR: turn %d 响应无法解析, 留档:" % (t + 1))
            with open("logit_ab_%s_badturn%d.json" % (phase, t + 1), "w") as f:
                json.dump(data, f, indent=1)
            print(json.dumps(data)[:800])
            sys.exit(5)
        out["turns"].append({"turn": t + 1, "n": len(toks),
                             "secs": round(secs, 1), "tokens": toks})
        prompt += chunk
        print("  turn %2d: %5d tok in %7.1fs  (%.1f tok/s)"
              % (t + 1, len(toks), secs, len(toks) / max(secs, 1e-9)), flush=True)
    path = os.path.join(outdir, "logit_ab_%s.json" % phase)
    out["end_wall"] = time.time()
    with open(path, "w") as f:
        json.dump(out, f)
    print("saved ->", path)
    print("下一步: 另开一相时先重启 server (换 env 值), 再跑 --phase %s"
          % ("B" if phase == "A" else "A"))


def flat(phase_data):
    return [tk for turn in phase_data["turns"] for tk in turn["tokens"]]


def top1(tok):
    return tok.get("text") or "?"


def margin_of(tok):
    tops = tok.get("top") or []
    if len(tops) >= 2 and tops[0][1] is not None and tops[1][1] is not None:
        return tops[0][1] - tops[1][1]
    return None


def compare(outdir):
    pa, pb = os.path.join(outdir, "logit_ab_A.json"), os.path.join(outdir, "logit_ab_B.json")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print("ERROR: 缺文件。需要 logit_ab_A.json 和 logit_ab_B.json (各自 phase 成功后生成)")
        sys.exit(6)
    with open(pa) as f:
        A = json.load(f)
    with open(pb) as f:
        B = json.load(f)
    # 重启守卫: 用文件内记录的实际 wall clock (比 mtime 可靠, 不受拷贝/时间戳干扰)
    a_end, b_start = A.get("end_wall"), B.get("start_wall")
    if a_end and b_start:
        gap = b_start - a_end
        if abs(gap) < 120:
            print("!! A 结束到 B 开始仅相隔 %.0f 秒 (每 phase 实际 ~15-20 分钟)。" % gap)
            print("!! 说明两相之间 server 没有重启 —— KV cache 跨相残留,")
            print("!! 本次对比无效。重启 server 后重跑较晚的那个 phase。")
            sys.exit(7)
        print("   (A 结束→B 开始: %.1f 秒, 通过重启守卫)" % gap)
    if A.get("seed") != B.get("seed"):
        print("!! 警告: A/B 种子不同, 语料不一致, 对比无效。")
        sys.exit(8)
    a, b = flat(A), flat(B)
    n = min(len(a), len(b))
    print("A: %d tok   B: %d tok   可对齐: %d" % (len(a), len(b), n))

    fork, margins = None, []
    for i in range(n):
        if top1(a[i]) != top1(b[i]):
            fork = i
            break
        m = margin_of(a[i])
        if m is not None:
            margins.append(m)

    print()
    if fork is None:
        print("== NO FORK in %d tokens ==" % n)
        print("   greedy 翻转率 < 1/%d  (Poisson 95%% 上界 ~ 3/%d = %.1e/token)"
              % (n, n, 3.0 / n))
    else:
        print("== FIRST FORK at token %d / %d ==" % (fork, n))
    ms = sorted(m for m in margins if m is not None)
    if ms:
        q = lambda p: ms[min(len(ms) - 1, int(p * len(ms)))]
        thin05 = sum(1 for m in ms if m < 0.05)
        thin01 = sum(1 for m in ms if m < 0.01)
        print("   首分叉前 top1-top2 边际 (n=%d, nats):" % len(ms))
        print("     min=%.4f  p1=%.4f  p5=%.4f  median=%.4f  p95=%.4f"
              % (ms[0], q(0.01), q(0.05), q(0.5), q(0.95)))
        print("     边际 < 0.05: %d (%.2f%%)   < 0.01: %d (%.2f%%)"
              % (thin05, 100.0 * thin05 / len(ms), thin01, 100.0 * thin01 / len(ms)))
    print()
    print("== 判决参考 ==")
    if fork is None and n >= 20000:
        print("   2万+ token 不翻 -> 长 agentic 象限基本可用, 作品全救回")
    elif fork is None:
        print("   本样本内不翻, 但样本 < 2万, 可加大 --turns/--decode 再跑")
    elif fork > 10000:
        print("   >1万 token 才翻 -> 介于两者之间: 可考虑 bit 对齐, 或接受该象限用 OFF")
    else:
        print("   <=1万 token 就翻 -> 长 agentic 象限死刑, 分场景 env 就是终局")
    print()
    print("   [附] grep -c sm70 两侧 server log: A 侧应有 ACCEPT 行, B 侧应有 REJECT-env-disabled 行")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--server", default=DEFAULT_SERVER,
                    help="默认 http://127.0.0.1:8080, 一般不要改")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--decode", type=int, default=2560)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    if args.compare:
        compare(args.outdir)
        return
    if not args.phase:
        ap.error("需要 --phase A|B 或 --compare")
    print("phase %s  turns=%d decode=%d (total ~%d output tok)"
          % (args.phase, args.turns, args.decode, args.turns * args.decode))
    print("(预计每 phase 15-20 分钟)")
    run_phase(args.server, args.phase, args.turns, args.decode, args.outdir)


if __name__ == "__main__":
    main()
