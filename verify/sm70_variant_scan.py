#!/usr/bin/env python3
# sm70_variant_scan.py — 布局变体扫描: 找出让 repro≈C 的正确解读,
# 直接锁定 launcher 适配层错位的 stride/映射参数.
#
# 已知: kernel 忠实计算它读到的数据 (repro≈A 0.0001), 但数据语义错 0.69.
# 嫌疑变体:
#   base : K[hkv][ctx][d], V[ctx][hkv][d], GQA h->h//6   (launcher 当前语义)
#   Kt   : K dequant dst 实为 [ctx][hkv][d] (to_fp16 输出布局理解反)
#   Vt   : V 实为 head-major [hkv][ctx][d]
#   KVt  : K 和 V 都转置
#   GQAi : GQA 交错映射 h->h%4
#   GQA_Kt / GQA_Vt / 全变体组合

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
    strides = struct.unpack_from('<4q', data, off); off += 32
    k_row_stride, k_head_stride, v_row_stride, v_head_stride = strides
    counts = struct.unpack_from('<3Q', data, off); off += 24
    k_count, v_count, q_count = counts
    K = np.frombuffer(data, dtype='<f2', count=k_count, offset=off).astype(np.float32); off += 2 * k_count
    V = np.frombuffer(data, dtype='<f2', count=v_count, offset=off).astype(np.float32); off += 2 * v_count
    Q = np.frombuffer(data, dtype='<f2', count=q_count, offset=off).astype(np.float32)
    return dict(kv_len=kv_len, q_len=q_len, q_pad=q_pad, hkv=hkv, heads_q=heads_q), K, V, Q


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


def attention(Qr, Kr, Vr, head_map, kv_offset, scale):
    heads_q, q_len, D = Qr.shape
    kv_len = Kr.shape[1]
    O = np.zeros((heads_q, q_len, D), dtype=np.float64)
    for h in range(heads_q):
        kh = head_map(h)
        S = (Qr[h].astype(np.float64) @ Kr[kh].astype(np.float64).T) * scale
        rows = np.arange(q_len)[:, None] + kv_offset
        cols = np.arange(kv_len)[None, :]
        S = np.where(cols <= rows, S, -np.inf)
        S -= S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        O[h] = P @ Vr[kh].astype(np.float64)
    return O.transpose(1, 0, 2).reshape(-1)


def rel(x, y):
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kv', required=True)
    ap.add_argument('--a', required=True)
    ap.add_argument('--c', required=True)
    args = ap.parse_args()

    meta, K, V, Q = read_kv_dump(args.kv)
    kv_len, q_len, hkv, heads_q = meta['kv_len'], meta['q_len'], meta['hkv'], meta['heads_q']
    q_pad = meta['q_pad']
    gqa = heads_q // hkv
    D = 256
    scale = 1.0 / np.sqrt(D)
    kv_offset = kv_len - q_len
    print('meta:', meta, 'gqa:', gqa)

    A = read_first_prefill(args.a)
    C = read_first_prefill(args.c)

    Qr = Q.reshape(heads_q, q_pad, D)[:, :q_len, :]

    # 布局变体
    K_base = K.reshape(hkv, kv_len, D)                       # [hkv][ctx][d]
    K_t = K.reshape(kv_len, hkv, D).transpose(1, 0, 2)       # 若实为 [ctx][hkv][d]
    V_base = V.reshape(kv_len, hkv, D).transpose(1, 0, 2)    # [ctx][hkv][d] -> [hkv][ctx][d]
    V_t = V.reshape(hkv, kv_len, D)                          # 若实为 [hkv][ctx][d]

    def h_group(h):
        return h // gqa

    def h_inter(h):
        return h % hkv

    variants = [
        ('base   K[h][c] V[c][h] h//g', K_base, V_base, h_group),
        ('Kt     K[c][h] V[c][h] h//g', K_t,   V_base, h_group),
        ('Vt     K[h][c] V[h][c] h//g', K_base, V_t,   h_group),
        ('KVt    K[c][h] V[h][c] h//g', K_t,   V_t,   h_group),
        ('GQAi   K[h][c] V[c][h] h%hkv', K_base, V_base, h_inter),
        ('GQAiKt K[c][h] V[c][h] h%hkv', K_t,   V_base, h_inter),
        ('GQAiVt K[h][c] V[h][c] h%hkv', K_base, V_t,   h_inter),
        ('ALL    K[c][h] V[h][c] h%hkv', K_t,   V_t,   h_inter),
    ]

    print()
    print('%-32s %10s %10s' % ('variant', '|v-C|', '|v-A|'))
    best = None
    for name, Kr, Vr, hm in variants:
        try:
            O = attention(Qr, Kr, Vr, hm, kv_offset, scale)
            rc, ra = rel(O, C), rel(O, A)
            print('%-32s %10.4f %10.4f' % (name, rc, ra))
            if best is None or rc < best[1]:
                best = (name, rc, ra)
        except Exception as e:
            print('%-32s ERROR %s' % (name, e))
    print()
    print('best: %s  |v-C|=%.4f  |v-A|=%.4f' % best)
    if best[1] < 0.05:
        print('[锁定] 该变体是正确解读 -> launcher 的对应 stride/映射参数错位, 按此修复')
    else:
        print('[!] 没有变体接近 C (best %.4f) -> 错误不在这些布局维度, 需扩大扫描' % best[1])


if __name__ == '__main__':
    main()
