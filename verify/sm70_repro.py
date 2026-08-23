#!/usr/bin/env python3
# sm70_repro.py — minimal reproduction of the 0.69 real-input divergence.
#
# 输入:
#   --kv  : SM70_DUMP_KV 捕获的 kernel 实际输入 (K dequant 后 / V 原生 cache / Qs staging)
#   --a   : ON 相 dump (SM70_DUMP), 取第一个 prefill 记录 = 第一层输出
#   --c   : CPU f32 真值相 dump, 同样取第一个 prefill 记录
#
# 做法: 用 dump 的 K/V/Q 在 numpy f32 里精确计算 attention 输出 (repro),
# 与 A (server sm70 实际输出) 和 C (CPU 真值) 三方对比.
#
# 判定矩阵:
#   |repro-C| ~ 0     -> dump 的输入数据正确; 错误在 kernel 对数据的处理
#                        (stride 传参或 kernel 内部) -> 走 kernel 二分
#   |repro-C| ~ 0.69  -> dump 的输入数据本身已错 (dequant/Q staging bug)
#                        -> 走预处理二分
#   |repro-A| ~ 0     -> kernel 与'拿这些输入算'的行为一致, 错在输入
#   |repro-A| ~ 0.69  -> kernel 在 server 几何下算出了与这些输入不一致的结果
#                        -> launcher 传参 (stride/布局) 与 dump 数据不匹配

import argparse
import struct
import sys

import numpy as np


def read_kv_dump(path):
    with open(path, 'rb') as f:
        data = f.read()
    off = 0
    hdr = struct.unpack_from('<7I', data, off); off += 28
    magic, ver, kv_len, q_len, q_pad, hkv, heads_q = hdr
    assert magic == 0x51444B53, 'bad magic'
    strides = struct.unpack_from('<4q', data, off); off += 32
    k_row_stride, k_head_stride, v_row_stride, v_head_stride = strides
    counts = struct.unpack_from('<3Q', data, off); off += 24
    k_count, v_count, q_count = counts
    K = np.frombuffer(data, dtype='<f2', count=k_count, offset=off).astype(np.float32); off += 2 * k_count
    V = np.frombuffer(data, dtype='<f2', count=v_count, offset=off).astype(np.float32); off += 2 * v_count
    Q = np.frombuffer(data, dtype='<f2', count=q_count, offset=off).astype(np.float32)
    meta = dict(kv_len=kv_len, q_len=q_len, q_pad=q_pad, hkv=hkv, heads_q=heads_q,
                k_row_stride=k_row_stride, k_head_stride=k_head_stride,
                v_row_stride=v_row_stride, v_head_stride=v_head_stride)
    return meta, K, V, Q


def read_first_prefill(path):
    HDR = struct.Struct('<8IQ')
    with open(path, 'rb') as f:
        while True:
            h = f.read(HDR.size)
            if len(h) < HDR.size:
                return None
            magic, ver, call, q_len, ne0, ne1, ne2, ne3, nbytes = HDR.unpack(h)
            payload = f.read(nbytes)
            if len(payload) < nbytes:
                return None
            if q_len >= 256:
                return np.frombuffer(payload, dtype='<f4')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kv', required=True)
    ap.add_argument('--a', required=True)
    ap.add_argument('--c', required=True)
    args = ap.parse_args()

    meta, K, V, Q = read_kv_dump(args.kv)
    m = {k: meta[k] for k in ('kv_len', 'q_len', 'q_pad', 'hkv', 'heads_q',
                               'k_row_stride', 'k_head_stride', 'v_row_stride', 'v_head_stride')}
    print('meta:', m)

    kv_len, q_len, hkv, heads_q = meta['kv_len'], meta['q_len'], meta['hkv'], meta['heads_q']
    gqa = heads_q // hkv
    D = 256
    scale = 1.0 / np.sqrt(D)

    # Qs layout: [head_q][row][d] (single batch)
    Qr = Q.reshape(heads_q, meta['q_pad'], D)[:, :q_len, :]
    # K layout: contiguous [head_kv][ctx][d]
    Kr = K.reshape(hkv, kv_len, D)
    # V layout: pos-major [ctx][head_kv][d]
    Vr = V.reshape(kv_len, hkv, D).transpose(1, 0, 2)

    kv_offset = kv_len - q_len

    # attention per kv-head group
    O = np.zeros((heads_q, q_len, D), dtype=np.float64)
    for h in range(heads_q):
        kh = h // gqa
        S = (Qr[h].astype(np.float64) @ Kr[kh].astype(np.float64).T) * scale  # (q_len, kv_len)
        # causal mask: col j valid iff j <= i + kv_offset
        rows = np.arange(q_len)[:, None] + kv_offset
        cols = np.arange(kv_len)[None, :]
        S = np.where(cols <= rows, S, -np.inf)
        S -= S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        O[h] = P @ Vr[kh].astype(np.float64)

    # output layouts: dump dst is [q_len][heads][D]
    repro = O.transpose(1, 0, 2).reshape(-1)
    A = read_first_prefill(args.a)
    C = read_first_prefill(args.c)
    if A is None or C is None:
        print('ERROR: no prefill record found in A/C dumps')
        sys.exit(1)

    def rel(x, y):
        return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))

    print()
    print('|repro - C (CPU 真值)| : %.4f' % rel(repro, C))
    print('|repro - A (sm70 输出)| : %.4f' % rel(repro, A))
    print('|A    - C             | : %.4f' % rel(A, C))
    print()
    rc, ra = rel(repro, C), rel(repro, A)
    if rc < 0.05:
        print('[判定] dump 输入数据正确 (repro≈C). 错误在 kernel 侧:')
        if ra > 0.3:
            print('       且 repro≠A -> server 几何下 kernel 算得与输入不一致: 查 launcher 传参/stride/kernel')
        else:
            print('       但 repro≈A -> kernel 与这些输入自洽, 与 C 的差来自更早的输入差异 (不可能, 需复查)')
    elif rc > 0.3:
        print('[判定] dump 输入数据本身已错 (repro 偏离 C 0.3+). 错误在预处理:')
        if ra < 0.1:
            print('       且 repro≈A -> kernel 忠实计算了错的数据: dequant (q4_0 K) 或 Q staging 有 bug')
        else:
            print('       且 repro≠A -> 输入错 + kernel 也错: 两处都要查')
    else:
        print('[判定] 中间地带, 需进一步二分')


if __name__ == '__main__':
    main()
