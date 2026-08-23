#!/usr/bin/env python3
# sm70_layer_diff.py — layerwise ON/OFF attention-output diff from SM70_DUMP captures.
#
# 输入: dump 文件 (verify/dump_ab.sh 或手动: SM70_DUMP=<path> 起 server 发一条 prompt).
# 记录格式见 ggml/src/ggml-cuda/fattn.cu 的 sm70_dump_attn_output 注释:
#   u32 magic, u32 ver, u32 call, u32 q_len, u32 ne0..ne3, u64 nbytes, payload f32.
#
# 用法:
#   两相:  python3 sm70_layer_diff.py --a dump_on.bin --b dump_off.bin [--csv out.csv]
#   三相:  加 --c dump_cpu.bin  (CPU f32 参照: llama-server -ngl 0 -ctv f32 同 prompt)
#          -> 逐层 |A-C| vs |B-C| 定责: sm70 和 stock 谁离 f32 真值更近
#
# 判决什么 (8/23 复查 + 当晚实验链):
#   实测: 第一个 full-attn 层 ON vs OFF 输出差 ~0.69 (远超舍入级);
#   合成尖 softmax 下 sm70 vs CPU 参照仅 1.3e-3 -> 0.69 的责任归属未知.
#   三相对比给 0.69 定责: sm70 更准 / stock 更准 / 各偏各.

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


def first_prefill_chunk(recs, label):
    pre = [i for i, r in enumerate(recs) if r['q_len'] >= 256]
    chunk = []
    if pre:
        last = pre[0] - 1
        for i in pre:
            if i == last + 1:
                chunk.append(i)
                last = i
            else:
                break
    if not chunk:
        print("ERROR: %s: no prefill records (prompt < 256 tokens?)" % label)
        sys.exit(5)
    return chunk


def rel(a_bytes, b_bytes):
    import numpy as np
    a = np.frombuffer(a_bytes, dtype='<f4')
    b = np.frombuffer(b_bytes, dtype='<f4')
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / max(nb, 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='ON phase dump (sm70)')
    ap.add_argument('--b', required=True, help='OFF phase dump (stock)')
    ap.add_argument('--c', default=None, help='CPU f32 reference dump (-ngl 0 -ctv f32)')
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
    C = read_dump(args.c) if args.c else None
    if C is not None:
        print("C (%s): %d records" % (args.c, len(C)))

    n = min(len(A), len(B))
    for i in range(n):
        if A[i]['q_len'] != B[i]['q_len'] or A[i]['ne'] != B[i]['ne']:
            print("!! shape mismatch at record %d: A q_len=%d ne=%s vs B q_len=%d ne=%s"
                  % (i, A[i]['q_len'], A[i]['ne'], B[i]['q_len'], B[i]['ne']))
            print("   ON/OFF dispatch 在该调用分岔, 对齐破坏, 终止")
            sys.exit(4)

    ca = first_prefill_chunk(A, 'A')
    cb = first_prefill_chunk(B, 'B')
    L = len(ca)
    if [A[i]['q_len'] for i in ca] != [B[i]['q_len'] for i in cb] or L != len(cb):
        print("ERROR: A/B first prefill chunk mismatch")
        sys.exit(6)
    print("first prefill chunk: %d layers (q_len=%d)" % (L, A[ca[0]]['q_len']))

    cc = None
    if C is not None:
        cc = first_prefill_chunk(C, 'C')
        if len(cc) != L or any(C[cc[k]]['q_len'] != A[ca[k]]['q_len'] or C[cc[k]]['ne'] != A[ca[k]]['ne']
                               for k in range(L)):
            print("ERROR: C first prefill chunk mismatch (q_len/ne differ from A)")
            sys.exit(6)

    # ---- 逐层误差
    rows = []
    for k in range(L):
        row = {'layer': k + 1, 'q_len': A[ca[k]]['q_len'],
               'rel_ab': rel(A[ca[k]]['data'], B[cb[k]]['data'])}
        if cc is not None:
            row['rel_ac'] = rel(A[ca[k]]['data'], C[cc[k]]['data'])
            row['rel_bc'] = rel(B[cb[k]]['data'], C[cc[k]]['data'])
        rows.append(row)

    def fit_gamma(pts):
        if len(pts) < 3:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in pts)
        den = sum((x - mx) ** 2 for x in xs)
        return math.exp(num / den) if den else None

    pts = [(r['layer'], math.log(r['rel_ab'])) for r in rows if r['rel_ab'] > 0]
    gam = fit_gamma(pts)

    import numpy as np
    deltas = [np.frombuffer(A[ca[k]]['data'], dtype='<f4') - np.frombuffer(B[cb[k]]['data'], dtype='<f4')
              for k in range(L)]
    coss = []
    for k in range(len(deltas) - 1):
        x, y = deltas[k], deltas[k + 1]
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        if nx > 0 and ny > 0:
            coss.append(float(np.dot(x, y) / (nx * ny)))
    cos_mean = sum(coss) / len(coss) if coss else float('nan')

    # ---- 打印
    print()
    if cc is None:
        print("=== per-layer attention output diff (first prefill chunk, ON vs OFF) ===")
        mx = max(r['rel_ab'] for r in rows)
        print("%5s %10s  curve" % ("layer", "rel_err"))
        for r in rows:
            print("%5d %10.3e  %s" % (r['layer'], r['rel_ab'], '#' * int(60 * r['rel_ab'] / mx)))
    else:
        print("=== per-layer three-way diff (C = CPU f32 reference) ===")
        print("%5s %10s %10s %10s  judgement" % ("layer", "A-B", "A-C", "B-C"))
        for r in rows:
            if r['rel_ac'] < r['rel_bc']:
                j = "sm70 closer" if r['rel_ac'] < 0.7 * r['rel_bc'] else "sm70 ~"
            elif r['rel_bc'] < r['rel_ac']:
                j = "stock closer" if r['rel_bc'] < 0.7 * r['rel_ac'] else "stock ~"
            else:
                j = "equal"
            print("%5d %10.3e %10.3e %10.3e  %s" % (r['layer'], r['rel_ab'], r['rel_ac'], r['rel_bc'], j))

    print()
    print("first-layer A-B : %.3e" % rows[0]['rel_ab'])
    print("last-layer  A-B : %.3e" % rows[-1]['rel_ab'])
    if gam:
        print("gamma fit (A-B) : %.4f" % gam)
    print("dir coherence   : mean=%+.4f  (random walk ~0; coherent >0)" % cos_mean)

    if cc is not None:
        mac = sum(r['rel_ac'] for r in rows) / L
        mbc = sum(r['rel_bc'] for r in rows) / L
        lac, lbc = rows[-1]['rel_ac'], rows[-1]['rel_bc']
        print()
        print("=== 定责判决 (C = f32 真值参照) ===")
        print("layer-avg |A-C| (sm70  vs 真值): %.3e" % mac)
        print("layer-avg |B-C| (stock vs 真值): %.3e" % mbc)
        print("last-layer |A-C|              : %.3e" % lac)
        print("last-layer |B-C|              : %.3e" % lbc)
        print()
        if mac < 0.7 * mbc:
            print("[判决] sm70 更接近 f32 真值 (|A-C| < 0.7|B-C|).")
            print("       -> 0.69 的大头是 stock 的 f16 PV 累加链误差;")
            print("       -> '污染'的重定性: 不是 sm70 算错, 是两套精度世界切换;")
            print("       -> 修复方向: 统一体系 (decode 也 f32 或 prefill 复刻 f16), 而非'修 sm70'.")
        elif mbc < 0.7 * mac:
            print("[判决] stock 更接近 f32 真值 (|B-C| < 0.7|A-C|).")
            print("       -> sm70 在真实输入下有 harness 未覆盖的行为问题;")
            print("       -> 沿 C 相逐层差定位具体层/模式, 那是可修的 bug.")
        else:
            print("[判决] 两者距真值同量级, 各偏各的.")
            print("       -> 0.69 是两套舍入体系的合法分歧; 混合链路的体系切换是主嫌疑.")

    if args.csv and rows:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print()
        print("csv -> %s" % args.csv)


if __name__ == '__main__':
    main()
