<div align="center">

# sm70-attn

**为 Tesla V100 重新点亮的 FlashAttention —— llama.cpp 深度定制分叉**

针对 Qwen3.5 / 3.6 / 3.8-27B（head_dim=256、GQA 6:1、16 层全注意力）
优化的 SM 7.0 CUDA kernel 插件，附 DFlash2 投机解码与多模态修复。

![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)
![SM](https://img.shields.io/badge/SM_7.0-Volta-76B900?logo=nvidia&logoColor=white)
![FA](https://img.shields.io/badge/FlashAttention-D256%20Split--D-3b82f6)
![SplitKV3](https://img.shields.io/badge/KV%20Split-3--way-22c55e)
![KV](https://img.shields.io/badge/KV%20Cache-q4_0%20%2B%20f16-f97316)
![Perf](https://img.shields.io/badge/176k%20prefill-%2B39.9%25-16a34a)
![Gates](https://img.shields.io/badge/harness-23%2F23%20PASS-6366f1)

[English](README-sm70.md)

</div>

---

<img src="sm70-attn.png" alt="sm70-attn vs stock：176k prefill 吞吐对比（8/26 同场 A/B）">

## 为什么存在这个分叉

llama.cpp 的官方 flash-attention 路径对 SM 7.0（Volta / Tesla V100）没有
tensor-core 实现。本仓库把一颗经过 1CatAI 验证的 Split-D N32 D256 kernel
移植进 llama.cpp 主线，并在此之上完成：

- **8/23 根因修复**：五天悬案"上下文污染"——launcher 对 q4_0 去量化输出
  布局的头优先假设是错的（实际为位置优先 `[ctx][head][D]`），9 行修复
  （`284b251`），每层 0.69 的相对误差归零；
- **SplitKV3**：长前缀 prefill 的三路 KV 切分，176k 再 +3.7%；
- **f32 输出全链路**：注意力输出不再经过 f16 暂存（单层舍入减半）；
- **q4_0 核内直读**（opt-in）：XQA 风格的按 dtype 模板加载，省 470MB 显存；
- **mtmd + dflash 修复**：多模态图片 + 投机解码在 M-RoPE 位置刻度上的
  崩溃（上游 ggml-org#27408），端到端修通。

所有行为都可以用环境变量一键回退到 stock 路径，逐位对齐。

## 性能

V100 单卡 · Qwen3.8-27B Q4_K_XL · `-fa on -ctk q4_0 -ctv f16`：

| 负载 | stock | sm70-attn | 提升 |
|:---|---:|---:|---:|
| 176k prefill（8/26 同场 A/B，唯一对比） | 372.94 tok/s | 521.93 tok/s | **+39.9%** |
| 8k 上下文 decode（tg64） | 28.99 tok/s | 29.06 tok/s | 持平（模型前向瓶颈） |

> decode 实测与上下文长度无关（注意力占比 <0.1%）——单卡 decode 的瓶颈
> 是 17GB 权重的显存带宽，不是注意力。这正是我们**不**移植 1Cat XQA
> decode kernel 的决策依据（详见下文「技术深度 · 上游状态」）。

## 快速开始

```bash
# 编译（需要 CUDA >= 12.x，目标 GPU 为 Volta）
cmake -B build -DGGML_CUDA=ON
cmake --build build --target llama-server -j

# 生产形态启动（27B 主模型 + mmproj + DFlash2 投机解码）
./build/bin/llama-server \
    -m Qwen3.8-27B-UD-Q4_K_XL.gguf \
    --mmproj mmproj-BF16.gguf \
    --spec-type draft-dflash \
    --spec-draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
    --spec-draft-n-max 2 \
    -fa on -ctk q4_0 -ctv f16 -c 229376 -b 4096 -ub 512 -ngl 99 \
    --host 127.0.0.1 --port 8080
```

非 Volta / 非 D256 / 非 causal / decode / 小 batch（ne[1]<256）会自动走
llama.cpp 原生路径，无需配置。

## 环境变量

| 变量 | 默认 | 作用 |
|:---|:---:|:---|
| `LLAMA_SM70_D256` | `1` | 总开关；`0` 禁用插件（与 stock 逐位一致） |
| `LLAMA_SM70_SPLITKV3_MIN_KV` | `2048` | SplitKV3 激活阈值（kv_len）；`0` 彻底关闭 |
| `LLAMA_SM70_D256_Q4_DIRECT` | `0` | 核内 q4_0 直读：省 ~470MB 显存，代价 ~4% prefill 速度 |
| `LLAMA_SM70_D256_DEBUG` | `0` | 打印 kernel 选路 probe（ACCEPT/REJECT 及原因） |
| `SM70_DUMP` / `SM70_DUMP_KV` | off | 捕获 kernel 输入/输出用于离线复现与三相定责 |

## 验证体系（概览）

每一处改动都被这些门拦着，全部自包含可复跑（`bash verify/<脚本>`）：

| 门 | 内容 | 状态 |
|:---|:---|:---:|
| G4 harness | 23 个 case：dense / f32-out / spiky / 位置优先 K,V / SplitKV3 边界 / q4 直读 | **23/23** |
| G5 三相定责 | 第一层 ON-OFF 误差必须落在 f16 舍入地板（~5e-4） | ✅ |
| G3 / G3b | 46k / 176k prefill 吞吐 A/B | ✅ |
| logit A/B | 贪心解码 128 token，ON vs stock 零分叉 | ✅ |
| 多模态回归 | 图片请求 200 + 视觉描述正确 + 纯文本接受率 0.88 | ✅ |

完整门定义（G1–G5 判定标准与历史注记）见下文「技术深度 · 验证门全量」。

## 修复档案（完整因果链）

### 上下文污染（8/23，9 行修复，`284b251`）

**症状**：585eeb 会话出现 23 个"失忆"签名——模型能续写最近内容，但记不住
更早的上下文。

**根因**：launcher 假设 `to_fp16_nc` 把去量化结果写成 `[ne2][ne1][ne0]`
（头优先），实际是 `[ne1][ne2][ne0]` = 位置优先 `[ctx][hkv][D]`。旧的头优先
stride（row=D、head=ctx\*D）把 K 在每次 q4_0-K prefill 上都打乱了：每层全
注意力输出偏差 ~0.69 相对误差，复合成 O(1) nat 的 logit 漂移，贪心 argmax
翻转，于是表现为"失忆"。

**定位**：`sm70_variant_scan.py` 布局变体扫描 12 分钟定案——K 转置变体把
|v-C| 从 0.69 压到 0.08，钉死布局错误。

**修复**：9 行（`284b251`），`posK/posKV` 回归 case 永久钉入 harness。

**防御**：G5 三相定责（ON-OFF-CPU f32 参考三份 dump 比对），且第一层 ON-OFF
必须落在 f16 舍入地板（~5e-4）——8/23 之前 G1b 从未在真实 kernel（COMMIT B）
上跑过，这个空窗就是 stride bug 存活五天的原因。

### 多模态 + 投机解码崩溃（8/24，上游 #27408）

**症状**：`--spec-type draft-dflash` 下喂图片直接 `llama_decode(ctx_dft)
rc=-1` -> HTTP 500 "failed to process mtmd chunk"。

**根因**（继承自 PR #27342 的 dflash2 移植，上游 ggml-org#27408）：多模态
batch 每一行位置是**常数**（752 个图片行全在 pos 53，空间结构只存在于
target 的 M-RoPE 机制里），后续文本从 `image_pos + grid_height`（53+47）接续
而非 `image_pos + n_rows`。draft 的一维 KV cache 两种形状都存不了——ubatch
循环的 chunk 2 连续检查失败。

**修复**（`common/speculative.cpp` 三处外科手术）：

1. `process()` 完全跳过 embedding batch；空洞用零特征 encoder 行填充
   （target 仍校验每个 drafted token——输出分布精确不变，只有图片区间的
   接受率下降）；
2. `draft()` 的噪声块基点改用 draft cache 自身的 `pos_max + 1` 而非
   `dp.n_past`（token 数与位置刻度在图片后分叉；服务端接受后的
   `seq_rm(pos_next, -1)` 让 `pos_max` 恰好停在已接受上下文末尾）；
3. `begin()` 的 pos_max-vs-N 警告降级为调试启发式（图片后该比较无意义）。

**验证**：图片请求 HTTP 200 + 视觉描述正确（噪声网格 -> "pixelated
checkerboard pattern"）、零 decode 失败、零填充恰好每图触发一次（logo 47
行）、纯文本接受率 0.88（生产正常水平；图片区间降至 ~0.57 后恢复）。
复现：`verify/mtmd_repro.sh [image] [n]`；视觉检查：`verify/mtmd_vision_check.py`。

## 技术深度

### 验证门全量（G1–G5）

- **G1**：构建干净；贪心解码 64 token top-1 与 stock 完全一致，logit 最大
  差 < 1e-2。
- **G1b**：176k 全量 prefill 末 256 logits 与 stock 路径比：最大绝对差
  < 1e-2，余弦 > 0.9999（f16 工作精度）。**8/23 注记：8/23 之前从未在真实
  kernel（COMMIT B）上执行过——这个空窗就是 stride bug 存活五天的地方；
  如今等效证据是 G5 三相定责裁决。**
- **G3**：46k prompt prefill A/B（`bench/prompt_46k.txt`）：累计 tok/s
  46k >= 620、30k >= 640、4k 与 stock 相差 5% 以内。
- **G3b**：176k prefill A/B（`bench/prompt_176k.txt`）：累计 tok/s >= 380
  （stock 基线 281.5）。**8/26 最终同场 A/B——本仓库保留的唯一 benchmark：
  ON 521.93 / stock 372.94 tok/s（+39.9%）。同一二进制、同一 prompt
  （176340 token），`LLAMA_SM70_D256=0` vs 默认；prefill 墙钟 337.86s vs
  472.83s（-28.5%）。**
- **回退检查**：`LLAMA_SM70_D256=0` 逐位复现 stock。
- **G4**（8/23）：harness `verify/sm70_verify`——19 cases 0 失败（dense /
  f32-out / spiky / 位置优先 K,V / SplitKV3 含空段边界）。**8/24 扩到 23
  cases：`q4K-pV`（生产全几何：直接 q4_0 K + paged f16 V）、`q4KV-279/3000`、
  `q4K-s3`；全部落在 f16 噪声地板（~1.7e-4），确认核内去量化位级一致。**
- **G5**（8/23）：三相 dump 裁决——第一层 ON-vs-OFF 必须在 f16 舍入地板
  （dense ~5e-4、splitkv3 ~3e-3），且层平均 |A-C| == |B-C|（到 CPU f32 参考
  等距）。工具见下文「调试工具链」。

### q4_0 核内直读：为什么默认 OFF（8/24，来自 1Cat XQA 架构）

kernel 新增 `Kq4/Vq4` 模板分支，直接读原始 q4_0 block cache
（`[ctx][head][block]`，字节 stride），在寄存器片段 / smem panel 内完成去量化
（4/8 元素组、分摊 block 寻址、对齐 u16 nibble 加载）。舍入与 staged
`to_fp16` 路径位级一致（精确 f32 乘积 + 一次 RN），harness（`q4K-pV`/
`q4KV`/`q4K-s3`，23/23）与 logit A/B（`verify/logit_q4direct.sh`）确认。

**为什么默认关**（8/24 实测，V100，q4_0 K + f16 V）：当初推动移植的那份
研究存在 1000 倍单位错误——整缓存去量化 staging 在 770s 的 176k prefill 里
只占 ~0.1s（28.7GB 流量），不是"数分钟"。与此同时窄加载让注意力 kernel
本身在 176k 慢 3.6%（457 vs 474 tok/s）。净账：opt-in 用 ~4% prefill 速度
换 ~470MB 显存（`-c 229k` 时 f16 镜像的体积）——只在显存受限时值得开。

### SplitKV3：三路 KV 切分（8/23 移植上游 splitkv3 patch）

kv_len >= 阈值时，KV 扫描沿 `gridDim.y` 切成 3 路，CTA 数 ×3，让长 prefill
后期不再因为饱和 SM 网格而串行化 KV 扫描；小型 merge kernel 合并
(max, sum, numerator) 三元组部分和。176k 净收益 +3.7%（420 -> 435.6
tok/s）——受 HBM 带宽墙限制，而非 SM 占用率。

上游遗漏的边界：一行因果窗口可能完全被单段掩盖；我们的移植把 SplitKV3
实例化的 row_max 初始化为 -1e30（而非 -inf），防止 `(-inf)-(-inf)=NaN`
混进部分和链。

### 上游（1Cat-vLLM）状态：已榨干（8/24）

关闭移植问题的一等测量（完整 8/23 研究报告 8/24 删除——没有可行动的残留）：

- **Prefill kernels**：Split-D 已 vendor + 修复 + splitkv3，干净吃完。
  `_full` 变体是无切分的简化，无新东西。
- **XQA decode kernel（3490 行）**：**未移植**，实测无关紧要——decode 在这
  张卡上是模型前向瓶颈：tg64 @ ctx 0 = 28.99 tok/s、@ ctx 8192 = 29.06
  tok/s。KV 读取在 34ms 的 step 里占 24us（<0.1%）；即便 176k 上下文也才
  ~1.5%。为它移植 paged-KV/CUDA-graph 是亏本买卖。
- **TurboQuant / FP8 KV / paged 布局转换器**：私有量化格式 + vLLM 专用
  基础设施，kernel 层面不可移植。
- **flash_qla（GDN 线性注意力）**：唯一开放线程——先测 GDN 层时间占比
  再定（profiling 工作流，不是移植）。
- **smem bank-padding 知识**（264/272 风格常量）：我们的 vendor kernel
  已经带 pitch-68 + Swizzle<3,3,3> + TT-swizzled V 布局。

## 调试工具链（8/23，破案工具包）

全部在 `verify/` 下，每个脚本自包含，在物理机运行（`bash verify/<脚本>.sh`）。

- `sm70_verify.cu` / `run.sh` — kernel harness，CPU f32 参考，kernel 源码
  md5 指纹防护。
- `sm70_layer_diff.py` — SM70_DUMP 捕获的逐层 ON/OFF diff（`--c` 加 CPU
  f32 参考相位，用于三相定责）。
- `sm70_variant_scan.py` — 布局变体扫描（钉死 stride bug 的那个工具：K
  转置变体把 |v-C| 从 0.69 压到 0.08）。
- `sm70_repro.py` — 从捕获的 kernel 输入 dump 最小复现。
- `dump_ab.sh` / `dump_cpu.sh` / `dump_kv.sh` — 捕获相位（env：
  `PROMPT`、`CT_V`、`ON_DUMP`、`OFF_DUMP`）。
- `dump_ab.sh` + 服务端 `SM70_DUMP`/`SM70_DUMP_MAX` — 每次调用的注意力输出
  捕获（同时钩 sm70 与 stock；CUDA-graph-capture 感知；ggml-cpu/ops.cpp
  里同一 env 后有 CPU 孪生钩子）。
- `logit_ab_fixed.sh` — 三相位 logit 分叉裁决（A/A2/B + 对比）。
- `perf_ab.sh` / `perf_ab_176k.sh` — 变更后吞吐验收。

## 仓库结构

```
ggml/src/ggml-cuda/
├── fattn.cu                     # 选路钩子（cc==700 + D256 + causal + prefill）
├── fattn-sm70-d256.cu           # launcher：Q 暂存 / K,V 去量化 / 8 路分发
├── fattn-sm70-d256-kernel.cuh   # Split-D kernel + SplitKV3 + q4 直读分支
└── sm70-vendor/                 # cute/cutlass (Apache-2.0) + flash (BSD-3)
verify/                          # 23-case harness + 三相 dump + 定责工具链
bench/prompt_46k.txt             # 标准负载（勿改动）
bench/prompt_176k.txt            # ROI 负载（勿改动）
```

关键文件明细：

- `fattn.cu` — 选路钩子（cc==700 + head_dim==256 + causal mask + prefill
  ne[1]>=256；F16/Q4_0 K/V）。其余形状全走 stock。同时携带 SM70_DUMP 调试
  钩子。
- `fattn-sm70-d256.cu` — launcher：Q f32->f16 补齐暂存、K/V 去量化 + 直读
  （**位置优先 stride**，见 8/23 修复）、注意力 launch、f32 输出回写。scratch
  从 get_alloc_size 额外区域切（含 SplitKV3 partials）。
- `fattn-sm70-d256-kernel.cuh` — 1CatAI 验证的 Split-D N32 D256 kernel
  （header 内有出处）。对上游的有文档偏差：
  - `kv_offset` 参数 + 用它构建 Mask（8/18）；
  - `ElementOut` 模板参数——生产 launcher 实例化 `float`，注意力输出全链路
    f32（8/23）；
  - `SplitKV3` 模板分支 + `sm70_d256_splitkv3_merge_kernel`——长前缀 prefill
    三路 KV 切分（8/23，上游 splitkv3 patch），带空段 gmem 防护与有限
    row_max 初始化（-1e30），上游没有。
- `sm70-vendor/` — 纯头文件三方闭包：cute/ + cutlass/（NVIDIA cutlass @
  62750a2b，Apache-2.0）+ flash/（zhinianqin/flash-attention-v100 @
  c2eda5e6 的 FA2 base layer，BSD-3）。出处见 sm70-vendor/README.md。
- `sm70-hook.patch` — fattn.cu 钩子独立补丁，rebase 重放用。

## 分支与同步

- `main`：工作分支 = 上游 master + 插件（基线 `baseline-2026-08-18` =
  ggml-org/llama.cpp `25ae3a9b3`）
- 远端 `upstream`（在 VM 上）→ ggml-org/llama.cpp；已验证可干净合并
  （8/24 合入 103 提交，冲突仅 workflow 删除项，全链路冒烟通过）
- 物理构建机只做 `git pull && cmake --build build`
- 保留分支 `pre-merge-backup-0824` 可整体回滚

## 提交链

- `fe7a3e7ca` - v1.0 钩子（pipeline 验证，stock kernel）[COMMIT A]
- `bfc1e99` - v1.1 真实 SM70 D256 Split-D kernel + vendor 闭包 [COMMIT B]
- `afa6e46` - logit_ab v3 终版（8/22 裁决工具链）
- `284b251` - **8/23 stride 修复**（0.69 根因；K/V 去量化 stride 是位置优先）
- `daca614`..`0b76746` - 8/23 调试工具包（dump 钩子、三相分析、变体扫描、
  复现）
- `db66e31` - A 组加固（F32 门 REJECT、posK/posKV harness 回归 case）
- `6e2c53d` + `1ce3b3d` - **SplitKV3 移植**（kernel 分支 + merge + 门 +
  NaN 边界修复）

## 致谢与许可

- [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) —— Split-D D256
  kernel 与 SplitKV3 patch 的上游来源
- [zhinianqin/flash-attention-v100](https://github.com/zhinianqin/flash-attention-v100)
  —— FA2 base layer（BSD-3）
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) —— CuTe/CUTLASS 头文件
  闭包（Apache-2.0）
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) —— 宿主（MIT）

许可随上游：MIT（llama.cpp 部分）+ BSD-3（flash vendor）+ Apache-2.0
（cutlass vendor）。
