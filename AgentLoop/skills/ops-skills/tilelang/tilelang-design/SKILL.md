---
name: tilelang-design
description: >-
  This skill should be used when the user asks to analyze a TileLang operator
  requirement and produce a detailed design (grid planning, on-chip memory plan,
  T.gemm tiling, pipelining stages, autotune strategy, test strategy) for Ascend
  NPU before any code is written. It covers two Ascend backends: tilelang-mlir-ascend
  on Atlas A2/A3 (arch22) and tilelang built with USE_ASCEND on Ascend 950 (arch35).
  Typical triggers include "用 TileLang 设计这个算子", "这个算子怎么用 TileLang 实现",
  "评估 TileLang 迁移可行性", "规划 T.gemm tiling 和 pipeline", "设计 TileLang kernel 方案".
version: 1.0.0
domain: ops
function: tilelang
tags:
  - tilelang
  - design
  - tiling
  - planning
  - npu
author: mindspeed-agent
compatibility: "tilelang >= 0.1.15, numpy == 1.26.4, ml-dtypes < 0.5, Python 3.12, CANN >= 9.0.0 (Ascend 950) / CANN 8.3+ (A2/A3 tilelang-mlir-ascend)"
targets-repo:
  - tilelang
  - tilelang-mlir-ascend
---

# TileLang 算子需求分析与详细设计

## 功能概述

本 Skill 覆盖算子开发三阶段流水线中的**阶段一**：从原始需求出发，完成可行性评估 → TileLang 详细设计 → 任务拆解与风险登记，产出下游 `tilelang-op-dev` 可直接执行的完整设计文档。

```text
┌────────────────────────────┐   ┌───────────────────────────┐   ┌──────────────────────┐
│ 阶段一：需求分析与详设        │ → │ 阶段二：代码生成与优化      │ → │ 阶段三：精度性能验收    │
│ tilelang-design（本 Skill）│   │ tilelang-op-dev          │   │ ops-test-gen          │
└────────────────────────────┘   └───────────────────────────┘   └──────────────────────┘
```

本 Skill 是 `ops-design-agent` 在 `dev_lang = tilelang` 路径下的核心能力。

## 适用场景

- 收到算子需求，需从分析到设计到规划给出完整 TileLang 方案
- GPU Triton / CUDA / PyTorch 算子迁到昇腾 NPU，评估 TileLang 可行性与适配方案
- GEMM / Attention / element-wise / reduction 等算子需设计 compile-time tiling 与 pipeline
- 不确定 TileLang 是否适配目标算子，需可行性评估与备选路线（回退 Triton / Ascend C）

## 前置条件

| 输入项 | 来源 | 必要性 |
| ------ | ------ | :---: |
| 算子需求描述（GPU 代码 / 自然语言 / 论文 / PyTorch） | 用户 | 必须 |
| 目标芯片（A2 / A3 / 950） | 用户或环境 | 必须 |
| 性能目标（延迟 / speedup） | 用户 | 推荐 |

缺失时主动向用户确认后再开始。

## TileLang 昇腾后端速览（设计前必读）

TileLang 在昇腾上有两条后端路线，**按目标芯片分流**：

| 目标芯片 | 后端 | 说明 |
| ------ | ------ | ------ |
| Atlas A2/A3（arch22，910B/910A） | `tilelang-mlir-ascend` | 源码构建的 MLIR 适配器，两条编译路线：AscendNPU IR（MLIR）与 Ascend C & PTO |
| Ascend 950（arch35 / DAV_3510） | `tilelang`（`USE_ASCEND=ON`） | 主仓原生 Ascend 后端，SIMD/SIMT 向量编程 |

设计文档必须**标注目标芯片与对应后端**，否则落地阶段无法选择编译路线。安装/构建细节见 `tilelang-op-dev`（环境构建章节）。

---

## 第零部分：TileLang 设计期硬约束

跨算子类别的设计期硬约束，草图阶段即须满足：

| 编号 | 约束 |
| ---- | ---- |
| TL-1 | 片上内存显式分配：全局数据经 `T.alloc_shared`（映射到 NPU 的 UB）搬入，累加/中间结果用 `T.alloc_fragment`（寄存器）；禁止裸指针逐元素访问 |
| TL-2 | `T.gemm` 只接受 `shared` 输入 + `fragment` 累加器；输入 tiling（BM/BK、BK/BN）须满足后端 Cube 粒度（A2/A3 走 tilelang-mlir-ascend 的 PTO 约束，950 走主仓 Cube 约束） |
| TL-3 | 循环分块用 `T.Pipelined(N, num_stages=k)` 显式声明；K 维度循环必须可静态展开（compile-time 常量），动态 K 需在 host 侧 pad 后传入 |
| TL-4 | `T.Kernel` 是**一维 AI-core grid（无 `threads` 参数）**：单维算子 grid = `ceil(N/block_N)`；二维算子（GEMM 等）把 tile grid 展平到一维（`by = bx // grid_n; bxx = bx % grid_n`）；核内线程并行用 `T.SimtVF(threads=...)` |
| TL-5 | epilogue 用 `T.Parallel` 表达，且**必须包在 `T.SimtVF()` / `T.SimdVF()` VF 块内**；fp32 累加器 → fp16 输出的 cast 在 VF 块内完成（`T.copy` DMA 不允许隐式 cast）；`T.Parallel` 内避免跨线程通信/动态索引 |
| TL-6 | dtype 约定：forward 计算 fp16/bf16，累加器 fp32；`T.gemm` 累加器 dtype 显式指定 |
| TL-7 | `T.copy` 的源/目标形状与 stride 必须 compile-time 确定，禁止运行时可变 shape 直接作为 copy 长度 |

---

## 第一部分：需求分析与可行性评估

### 步骤一：需求信息收集

确认：算子来源、功能（element-wise / reduction / matmul / attention / conv / 自定义）、输入输出（shape/dtype/layout）、精度要求、性能目标、目标芯片、交付时间。

### 步骤二：TileLang 可行性评估

| 算子类型 | TileLang 支持度 | 注意事项 |
| --------- | :---: | --------- |
| Element-wise / vector | ★★★★★ | `T.Kernel` + `T.copy` + `T.Parallel` 直接表达 |
| GEMM / Matmul | ★★★★★ | `T.gemm` + `T.Pipelined`，参考实现近手写 AscendC（~0.95–0.98x） |
| Softmax / LayerNorm | ★★★★☆ | reduction 需在 `T.Parallel` 内显式归约，注意 epilogue 分段 |
| Attention（Flash） | ★★★★☆ | tilelang-mlir-ascend 支持 FA 前反向；多级 tiling 需仔细规划 |
| Conv | ★★★☆☆ | im2col / 直接卷积取决于后端支持面 |
| 复杂原子操作 / 递归 | ★★☆☆☆ | TileLang 表达力有限，评估回退 Ascend C |

**不可行时**明确给出替代路线：回退 Triton（`ops-design` / `triton-op-dev`）或 Ascend C（`ascendc-design` / `ascendc-op-dev`）。

### 步骤三：确定开发方式与后端

- 用户显式指定 `dev_lang = tilelang` 时遵循
- 未指定时：需要编译期多级 tiling / 接近手写算子的极致性能、且目标算子落在 GEMM/向量/Attention 支持面内 → 选 TileLang；否则优先 Triton（开发效率最高）
- 目标芯片为 A2/A3 → 后端 `tilelang-mlir-ascend`；950 → 后端 `tilelang`（`USE_ASCEND=ON`）

---

## 第二部分：TileLang 详细设计

### 步骤四：片上内存规划（TL-1 / TL-2）

| Buffer | 分配方式 | 映射 | 用途 |
| ------ | ------ | ------ | ------ |
| A_shared / B_shared | `T.alloc_shared` | NPU UB | `T.gemm` 输入 tile |
| C_local / acc | `T.alloc_fragment` | 寄存器 | `T.gemm` 累加器 |
| 中间临时量 | `T.alloc_shared` 或 `T.alloc_fragment` | UB / 寄存器 | epilogue 中间结果 |

预算计算（A2/A3 经 tilelang-mlir-ascend 时 UB 语义与 Triton/Ascend C 一致）：

```python
# A2/A3 UB ≈ 192KB（double buffer 后有效 ≈ 96KB）；950 UB ≈ 256KB
UB_EFFECTIVE = 96 * 1024      # A2/A3，bytes；950 用 128 * 1024
BYTES_PER_ELEM = 2            # fp16 = 2B；fp32 = 4B
# GEMM: (BM*BK + BK*BN) * dtype_bytes 为 shared 输入占用；BM*BN * 4 为 fp32 累加器
ub_usage = (BM*BK + BK*BN) * BYTES_PER_ELEM + BM*BN*4
assert ub_usage <= UB_EFFECTIVE
```

### 步骤五：Grid 与 Tiling 设计（TL-4）

```python
# 昇腾 T.Kernel 是一维 AI-core grid，二维 tile grid 展平到一维
grid_m, grid_n = T.ceildiv(M, BM), T.ceildiv(N, BN)
num_tiles = grid_m * grid_n
# with T.Kernel(num_tiles) as bx:
#     by, bxx = bx // grid_n, bx % grid_n
```

GEMM tiling 决策速查：

| 场景 | 推荐 tiling |
| ------ | ------ |
| 中等方阵 | BM=BN=128, BK=32（A2/A3 经 PTO）/ 按 950 Cube 粒度 |
| M 大 N 小 | BM=128, BN=32/64 不对称 |
| Attention（Flash） | BM/BN 按 head dim，BK 分 K 循环 |

### 步骤六：Pipeline 与调度设计（TL-3）

- K 循环用 `T.Pipelined(K_tiles, num_stages=k)`，`num_stages` 常取 2–3；UB 紧张时降为 2 或关 double buffer
- epilogue（激活 / 归一化 / 量化）放 `T.Parallel` 并包在 `T.SimtVF()` / `T.SimdVF()` 内，与主循环解耦
- 目标为「搬运（`T.copy`）与计算（`T.gemm`）重叠」，设计时标注预期流水级数

### 步骤七：Kernel 架构与接口规格

```python
import tilelang
from tilelang.ascend import language as T   # 昇腾专用 language

@tilelang.jit(out_idx=[-1], target="ascend")
def my_kernel(M, N, K, block_M, block_N, block_K, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        grid_m, grid_n = T.ceildiv(M, block_M), T.ceildiv(N, block_N)
        with T.Kernel(grid_m * grid_n) as bx:      # 一维 AI-core grid
            by, bxx = bx // grid_n, bx % grid_n
            # ... T.alloc_shared / T.alloc_fragment / T.Pipelined(T.gemm)
            # ... epilogue 在 T.SimtVF()/T.SimdVF() 内完成 cast 后 T.copy 落盘
    return main

# host 侧：kernel = my_kernel(M, N, K, ...); C = kernel(A, B)  # out_idx=[-1] → 返回输出
# 生产 GEMM 优先用内置 tilelang.ascend.op.gemm（处理 fractal layout + epilogue）
```

### 步骤八：测试策略设计

复用 `ops-test-gen` 规范，shape×dtype 覆盖矩阵 + 边界（空张量/非对齐尾轴/奇数 K）+ 性能基线（vs `torch.nn.functional.*` 标杆）。

---

## 第三部分：任务规划与风险管理

按 T1（需求分析）→ T2（代码生成）→ T3（性能优化）→ T4（检视）→ T5/T6（精度/性能验收）拆解，估算工时，登记风险（后端编译失败 / UB 溢出 / fp16 累积误差 / 性能不达目标）。

## 输出模板

完成以上步骤后，按以下模板输出设计文档（`dev_lang = tilelang`，并标注目标芯片与后端）：

```markdown
# <算子名称> TileLang 开发方案

## 1. 需求概要（算子类型 / I/O / 计算逻辑 / 目标芯片 / 精度 / 性能目标）
## 2. 可行性结论（✅ 可行 / ⚠️ 有限制 / ❌ 不可行 + 后端路线 tilelang-mlir-ascend|tilelang）
## 3. 硬件约束与片上预算（UB 大小 / BM/BK/BN / ub_usage 占比）
## 4. Grid 与 Tiling 方案（grid 策略 / BM,BN,BK / 核数固定 or 大 grid）
## 5. 片上内存规划（alloc_shared / alloc_fragment 分配表）
## 6. Pipeline 与调度（num_stages / epilogue 布局）
## 7. Kernel 架构与接口签名（@tilelang.jit + launch 函数）
## 8. 测试策略（shape×dtype 矩阵 / 边界 / 性能基线 / 验收门禁）
## 9. 任务清单（T1–T6）
## 10. 风险登记
## 11. 交付清单
```

## 相关 Skill

| Skill | 关系 |
| ------ | ------ |
| `tilelang-op-dev` | 下游 — 按本 Skill 设计生成代码并优化 |
| `ops-design` / `ascendc-design` | 同阶段 — 回退路线（Triton / Ascend C） |
| `ops-test-gen` / `ops-profiling` / `ops-boundary-survey` | 下游 — 测试/性能/边界（三条语言路径共用） |

## 参考信息

- [TileLang 主仓库](https://github.com/tile-ai/tilelang)
- [tilelang-mlir-ascend（A2/A3）](https://github.com/tile-ai/tilelang-mlir-ascend)
- [TileKernels（DeepSeek TileLang 算子库）](https://github.com/deepseek-ai/TileKernels)
