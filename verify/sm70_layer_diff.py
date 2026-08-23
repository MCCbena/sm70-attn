#!/usr/bin/env python3
# sm70_layer_diff.py — layerwise ON/OFF attention-output diff from SM70_DUMP captures.
#
# 输入: 两个 dump 文件 (verify/dump_ab.sh 产生; 或手动: SM70_DUMP=<path> 起 server
# 发一条 prompt 后关闭). 记录格式见 ggml/src/ggml-cuda/fattn.cu 的
# sm70_dump_attn_output 注释: u32 magic, u32 ver, u32 call, u32 q_len,
# u32 ne0..ne3, u64 nbytes, payload f32 (little-endian).
#
# 用法: python3 sm70_layer_diff.py --a dump_on.bin --b dump_off.bin [--csv out.csv]
#
# 判决什么 (8/23 复查提出的三个成因假说):
#   H-A 相干复合: rel_err 逐层指数增长 (gamma>1), 且误差向量方向层间相干 (cos>0)
#   H-B 拼接注入: decode 段 (两边同为 stock kernel) 误差相对 prefill 终态的变化
#   H-C 结构化输入放大: 第 1 层 rel_err vs harness 随机输入地板 (~3.9e-4)

import argparse
import csv
import math
import struct
import sys

HDR = struct.Struct('<8IQ')
MAGIC = 0x53444D37


def read_dump(path):
    recs = []
    with open(path, 'rb') as f:
        while True:
            h = f.read(HDR.size)
            if len(h) < HDR.size:
                break
            magic, ver, call, q_len, ne0, ne1, ne2, ne3, nbytes = HDR.unpack(h)
            if magic != MAGIC:
                print("ERROR: %s: bad magic at rec %d (corrupt or mixed versions)" % (path, len(recs)))
                sys.exit(2)
            payload = f.read(nbytes)
            if len(payload) < nbytes:
                print("WARN: %s: truncated payload at rec %d" % (path, len(recs)))
                break
            recs.append({'call': call, 'q_len': q_len,
                         'ne': (ne0, ne1, ne2, ne3), 'data': payload})
    return recs


def fit_gamma(pts):
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return math.exp(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='ON phase dump')
    ap.add_argument('--b', required=True, help='OFF phase dump')
    ap.add_argument('--csv', default=None, help='write per-layer CSV')
    args = ap.parse_args()
    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy required (pip3 install numpy)")
        sys.exit(3)

    A = read_dump(args.a)
    B = read_dump(args.b)
    print("A (%s): %d records" % (args.a, len(A)))
    print("B (%s): %d records" % (args.b, len(B)))
    if len(A) != len(B):
        print("!! record count mismatch — 调用序列不对齐 (prompt 不同?), 只在公共前缀上分析")
    n = min(len(A), len(B))
    for i in range(n):
        if A[i]['q_len'] != B[i]['q_len'] or A[i]['ne'] != B[i]['ne']:
            print("!! shape mismatch at record %d: A q_len=%d ne=%s vs B q_len=%d ne=%s"
                  % (i, A[i]['q_len'], A[i]['ne'], B[i]['q_len'], B[i]['ne']))
            print("   ON/OFF dispatch 在该调用分岔, 对齐破坏, 终止")
            sys.exit(4)

    pre = [i for i in range(n) if A[i]['q_len'] >= 256]
    dec = [i for i in range(n) if A[i]['q_len'] == 1]
    print("prefill records: %d   decode records: %d   other: %d"
          % (len(pre), len(dec), n - len(pre) - len(dec)))

    # ---- 第一个 prefill chunk = 开头连续的 prefill 段
    first_chunk = []
    if pre:
        last = pre[0] - 1
        for i in pre:
            if i == last + 1:
                first_chunk.append(i)
                last = i
            else:
                break
    L = len(first_chunk)
    if not L:
        print("ERROR: no prefill records found (prompt < 256 tokens? see dump_ab.sh)")
        sys.exit(5)
    print("first prefill chunk: %d layers (q_len=%d)" % (L, A[first_chunk[0]]['q_len']))

    # ---- 逐层误差
    rows = []
    deltas = []
    for k, i in enumerate(first_chunk):
        a = np.frombuffer(A[i]['data'], dtype='<f4')
        b = np.frombuffer(B[i]['data'], dtype='<f4')
        d = a - b
        nb = float(np.linalg.norm(b))
        rel = float(np.linalg.norm(d) / max(nb, 1e-30))
        rows.append({'layer': k + 1, 'call': A[i]['call'], 'q_len': A[i]['q_len'],
                     'rel_err': rel, 'norm_diff': float(np.linalg.norm(d)), 'norm_b': nb})
        deltas.append(d)

    pts = [(r['layer'], math.log(r['rel_err'])) for r in rows if r['rel_err'] > 0]
    gam_full = fit_gamma(pts)
    gam_early = fit_gamma([p for p in pts if p[0] <= 16])

    coss = []
    for k in range(len(deltas) - 1):
        x, y = deltas[k], deltas[k + 1]
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        if nx > 0 and ny > 0:
            coss.append(float(np.dot(x, y) / (nx * ny)))
    cos_mean = sum(coss) / len(coss) if coss else float('nan')

    print()
    print("=== per-layer attention output diff (first prefill chunk, ON vs OFF) ===")
    mx = max((r['rel_err'] for r in rows), default=1e-30)
    print("%5s %10s  curve" % ("layer", "rel_err"))
    for r in rows:
        print("%5d %10.3e  %s" % (r['layer'], r['rel_err'], '#' * int(60 * r['rel_err'] / mx)))
    print()
    print("first-layer rel_err : %.3e   (harness random-input floor: ~3.9e-4)" % rows[0]['rel_err'])
    print("last-layer  rel_err : %.3e" % rows[-1]['rel_err'])
    print("growth ratio        : %.1fx over %d layers" % (rows[-1]['rel_err'] / max(rows[0]['rel_err'], 1e-30), L))
    if gam_full:
        print("gamma fit (all)     : %.4f" % gam_full)
    if gam_early:
        print("gamma fit (<=16)    : %.4f" % gam_early)
    print("dir coherence cos   : mean=%+.4f  (random walk ~0; coherent >0)" % cos_mean)

    print()
    print("=== 判决 ===")
    if gam_full is not None:
        if gam_full > 1.02:
            print("[H-A 相干复合] gamma=%.3f > 1: 误差逐层指数增长, 复合机制坐实" % gam_full)
        elif gam_full < 0.98:
            print("[!] gamma=%.3f < 1: 误差饱和/收缩, 0.97 nat 需另找来源" % gam_full)
        else:
            print("[!] gamma=%.3f ~ 1: 随机游走式复合, 单靠它到不了 0.97 nat — 看首层" % gam_full)
    print("[H-C 首层实测] 第 1 层 rel_err=%.3e vs harness 随机输入 ~3.9e-4:" % rows[0]['rel_err'])
    if rows[0]['rel_err'] < 2e-3:
        print("    接近 -> 真实输入没有显著放大单层误差 (原'4~5 个数量级'表述可撤)")
    else:
        print("    大得多 -> 结构化输入确实放大单层误差 (H-C 成立)")

    if dec:
        print()
        print("=== decode segment (q_len==1; 两边同走 stock decode kernel) ===")
        print("    (误差反映 prefill 写入的 KV 内容差异 — H-B 的观测点)")
        Ld = L
        for s in range(min(len(dec) // Ld, 8)):
            block = dec[s * Ld:(s + 1) * Ld]
            rels = []
            for i in block:
                a = np.frombuffer(A[i]['data'], dtype='<f4')
                b = np.frombuffer(B[i]['data'], dtype='<f4')
                nb = float(np.linalg.norm(b))
                if nb > 0:
                    rels.append(float(np.linalg.norm(a - b) / nb))
            if rels:
                print("  decode step %d: layer-avg rel_err=%.3e  last-layer=%.3e"
                      % (s + 1, sum(rels) / len(rels), rels[-1]))

    if args.csv and rows:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print()
        print("csv -> %s" % args.csv)


if __name__ == '__main__':
    main()
