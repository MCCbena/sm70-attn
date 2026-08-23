# 1Cat-vLLM 可移植性研究（2026-08-23）

上游：https://github.com/1CatAI/1Cat-vLLM （main @ `c4c9b84`）
研究对象：`flash-attention-v100/kernel/` 全部 14 个文件 + 最近一周的
kernel 提交（#259、#268、#270）。逐文件读源码 + 本机实测决策。

## 一、决策级测量：decode 移植价值 ≈ 0

上游最大的资产是 XQA decode kernel（`flash_decode_paged.cu`，3490 行）。
为它做了一次本机测量再决定要不要移植：

```
llama-bench -fa 1 -ctk q4_0 -ctv f16 -ngl 99   (Qwen3.8-27B Q4_K_XL, V100)
  tg64 @ ctx≈0    : 28.99 tok/s
  tg64 @ ctx 8192 : 29.06 tok/s
```

**decode 速度与上下文长度完全无关**。定量：8k 上下文每步 KV 读取量
4 heads × 8192 × 256 × (0.5625B q4_0 + 2B f16) ≈ 21.5MB ≈ 24µs，
而单 token 步长 34ms（16.7GB 权重 / ≈490GB/s 有效带宽）。
**注意力占 decode 时间 < 0.1%**；即使在 176k 上下文也只有 ≈1.5%。

结论：移植 3490 行 paged-KV XQA kernel（含 block table、E5M2/FP8 KV、
CUDA Graph 变体、batch/context 路由）到 llama.cpp 的连续 KV 布局上，
工作量巨大、收益 <1%。**不移植，留档决策依据。**

（上游 XQA 的存在意义是 vLLM 侧 batch>1 的吞吐服务场景 + CUDA graph
重放开销优化；我们是单卡交互式 llama.cpp，问题域不同。）

## 二、逐文件清单与判定

| 上游文件 | 行数 | 内容 | 判定 |
|---|---|---|---|
| `fused_mha_forward_paged*.cu` | 850+604 | Split-D prefill（已 vendor + 修复 + splitkv3）；`_full` 是无 split 的简化变体（D 模板 16-256，D=256 时 M32/N64） | **已吃干**，`_full` 无新东西 |
| `flash_decode_paged.cu` | 3490 | XQA decode：scalar partition kernel + TC wide kernel（WMMA M=8, D=256）+ split-reduce 双阶段 + batch/context 路由 | **不移植**（见第一节） |
| `flash_decode_turboquant.cu` | 631 | PQ 式 KV（MSE bits + 质心表 + VQ4/3bit）decode | 不移植：私有量化格式，需整套校准基础设施 |
| `fp8_kv_bridge.cu` / `fp8_kv_utils.cuh` | - | FP8(E5M2/E4M3) KV 桥接 | 不移植：ggml 无 FP8 KV 类型，是 ggml 层特性不是 kernel 层 |
| `contiguous_to_paged*.cu` / `paged_to_contiguous*.cu` | 4 个 | vLLM paged KV 布局转换 | 不适用：llama.cpp 无 block table |
| `fused_mha_backward.cu` | - | 训练反向 | 不适用 |
| `flash_v100_traits.cuh` / `paged_kv_utils.cuh` | - | 布局/页表工具 | 参考价值 |
| `flash_qla/` | - | GDN linear attention（Qwen hybrid 另一半） | 维持原判：先测 GDN 层耗时占比再议 |
| `flashinfer-sm70/` | - | flashinfer 移植层 | 不适用 |

## 三、真正值得带走的（按 ROI 排序）

### 1. 核内 q4_0 直读（省掉整缓存去量化 staging）— 最高优先级

现状（`fattn-sm70-d256.cu` 的 `sm70_d256_dequant_kv`，stock 路径同罪）：
每次 flash_attn 调用把**整个 K/V 缓存**去量化成 f16 extra buffer：

- llama-bench 单次大 prompt：每层一次性 dequant 176k 缓存 = 101MB 读 +
  361MB 写 × 62 层 ≈ 28.6GB 额外流量 ≈ 32s（约占 405s prefill 的 8%）
- **服务端 chunked prefill（ub=512）更糟**：每个 chunk 每层都重新 dequant
  整个已增长的缓存，长上下文下趋于 O(n²)——这是 stock llama.cpp 量化 KV
  长上下文 prefill 的经典痛点，我们的 kernel 继承了它

上游给的法门：XQA kernel 的 `load_xqa_tc_kv_vector<KV_DTYPE>` 把
"按 dtype 加载 KV 进 smem panel"做成模板特化（FP16 直接 uint4 加载、
E5M2 转 half8）。**给我们的 Split-D kernel 加一个 Q4_0 特化**：
每线程读 18 字节 block → 寄存器 dequant → 写 f16 smem panel。
kernel 只需读 101MB 而不是 462MB（读+写+再读），且 chunked prefill
的 O(n²) 直接消失。

工作量：kernel 加载路径改造（一个模板分支 + q4_0 解码函数，
模式在 stock `fattn-tile` 的核内 dequant 里现成）+ launcher 删 staging
分支。收益：长上下文 prefill 8%（单发）到数倍（服务端 chunked）。

### 2. smem bank 冲突 padding 常量 — 顺手带走

```cpp
kXQATC256WidePaddedQStride   = 264 / 272;   // D=256 Q panel
kXQATC256WidePaddedKVStride  = 136 / 144;   // D=256 KV panel (N=128)
```

Volta WMMA 的 A/B fragment 加载在 256/128-half 稠密 stride 上会撞
smem bank。如果做第 1 项时重写加载路径，直接用这组经验值。

### 3. "#268 宽加载"思想（非代码）

跨 256-token 分区摊销索引计算 + 对齐 128-bit 成对加载。我们没有
page table，但"把 per-token 的 block 偏移计算摊到 per-128-token panel"
这个思想适用于第 1 项的 q4_0 block 地址计算。

### 4. 不带走但记录

- **#259 batch/context 路由 + CUDA Graph 变体隔离**：vLLM 专属
  （llama.cpp 无 CUDA graph），思想层面对应我们的
  `LLAMA_SM70_SPLITKV3_MIN_KV` 环境变量 gating，已有等价物。
- **#270 NVFP4 MoE / MTP 冷启动**：模型量化路径，与 attention 无关。

## 四、结论

1Cat-vLLM 的 prefill 资产我们已经吃干（Split-D + splitkv3 + 修复链）；
decode 资产（XQA）对本机负转载无意义（实测 <0.1% 占比）。下一个
真正的大项是**核内 q4_0 直读消灭 staging**——这严格说不是"移植某文件"，
而是把上游"quantized KV 核内加载"的架构思想装进我们自己的 kernel。

前置条件：无（kernel 和 harness 都在我们手里，回归 case 19/19 现成）。
