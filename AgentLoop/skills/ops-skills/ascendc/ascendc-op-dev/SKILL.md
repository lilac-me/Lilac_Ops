---
name: ascendc-op-dev
description: >-
  This skill should be used when the user asks to generate, build, debug, or optimize
  an Ascend C operator kernel for Ascend NPU (Atlas A2/A3 arch22 or Ascend 950 arch35),
  from a detailed design document or existing CUDA/PyTorch reference. It orchestrates
  kernel direct-invoke / AscendDSL / aclNN / CATLASS(Blaze) code generation, remote
  compile-and-run on NPU, hypothesis-driven debugging (ascendc-op-debug, precision-debug,
  runtime-debug, crash-debug), performance tuning (ascendc-perf-optimize, ops-profiling),
  and code review (ascendc-code-review). Typical triggers include "用 Ascend C 写个算子",
  "生成 aclnn 算子工程", "这个 kernel 跑出 NaN 了", "优化 Ascend C 算子性能",
  "950 上用 Blaze 写 Matmul".
version: 1.0.0
domain: ops
function: ascendc
tags:
  - ascendc
  - code-generation
  - optimization
  - debugging
  - cann
author: mindspeed-agent
---

# Ascend C 算子开发与性能优化

## 功能概述

本 Skill 覆盖算子开发三阶段流水线中**阶段二 + 阶段三（Ascend C 路径）**的全部工作：按上游详细设计文档
生成 NPU 可运行的 Ascend C 算子代码，上板编译与测试，进行系统性优化，并完成代码检视与精度/性能验收。

> 与 `triton-op-dev`（Triton 路径）平行。仅当 `dev_lang = ascendc` 时被 `ops-implement-agent` /
> `ops-optimize-agent` 调用。

## 定位：三阶段流水线（Ascend C 路径）

```text
┌──────────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────┐
│  阶段一：需求分析与详设         │ →  │  阶段二+三：开发与优化      │ →  │  阶段四：精度性能验收    │
│  ascendc-design       │    │  ascendc-op-dev（本 Skill）│    │  ops-test-gen           │
└──────────────────────────────┘    └──────────────────────────┘    └──────────────────────┘
```

本 Skill 是 `ops-implement-agent` / `ops-optimize-agent` 在 Ascend C 路径下的核心能力，接收详设文档，
产出可运行的 Ascend C 算子代码 + 测试/性能报告。

## 适用场景

- 从详细设计文档生成 Ascend C 算子（直调 / DSL / aclNN / CATLASS）
- CUDA kernel → Ascend C 迁移适配
- 自然语言 / PyTorch / NumPy → Ascend C 实现
- 性能不达标（UB 溢出、多核负载不均、流水断裂、Cube/Vector 协同差）
- 运行异常：精度错误（NaN/随机值）、崩溃（Segmentation Fault / Abort）、卡死（Kernel Hang）
- 950 上 Matmul/融合算子走 Blaze / CATLASS

## 支持的输入形式

| 输入形式 | 支持程度 | 说明 |
| --------- | :---: | ------ |
| 详细设计文档（来自阶段一） | ✅ 完整 | 按设计方案编码，效率最高 |
| CUDA 实现 | ✅ 完整 | 算法参考 → Ascend C 改造 |
| 自然语言描述 / PyTorch | ✅ 完整 | 从零生成 |
| Triton 算子 | ⚠️ 反向 | 使用对应 Triton Skill |
| 950 Matmul 参考 | ✅ 完整 | 走 CATLASS（catlass-op-design/develop） |

---

## 第一部分：环境约束速查

### 一、目标架构差异（务必先确认）

调用 `npu-arch` 和 `ascendc-env-check` 确认：

| 项 | A2 / A3（arch22） | Ascend 950（arch35 / DAV_3510） |
| ---- | ------ | ------ |
| UB | 192 KB | 256 KB |
| 编程模型 | SIMD（Vector）+ Cube | SIMD / **SIMT**（950 引入，`ascendc-simt-best-practices`） |
| Matmul | 自写 Cube kernel / aclNN | **CATLASS / Blaze**（`catlass-op-design`+`catlass-op-develop`） |
| 多核并行 | `block_dim` 固定核数 | 同左，核数更多 |
| 特殊 API | 标准 Ascend C | RegBase（`ascendc-regbase-best-practice`，950 Reg API） |

### 二、UB / 对齐硬约束

- **UB 超限**：`UB size exceeded` → 减小 TILE / 二次切分 / 关 double buffer
- **对齐**：`DataCopy` 全量搬运时尾轴需满足数据类型对齐（fp16 32B 等），参考 `ascendc-api-best-practices`
- **多核尾块**：`process` + mask 控制，避免越界写 GM

### 三、反作弊约束

- **禁止** `torch.nn.functional.*` / `F.xxx` / Triton 包装冒充 Ascend C
- 必须是真实 Ascend C kernel（`.cpp` + 直调 / DSL / aclNN），计算逻辑可审计

### 四、向量化开发基础

采用 Ascend 架构独有的向量化开发模式，避免大规模标量操作。
编码前调用 `ascendc-vector-dev` 并按以下要求检查：

| 要求 | 说明 |
| ------ | ------ |
| V1 数据搬运 | 批量数据一律 `DataCopy/DataCopyPad`（DMA），避免 `GetValue`/`__gm__` 标量逐元素 GM 读 |
| V2 数学函数 | `sin/cos/atan/sqrt` 用向量硬件指令（标量入 UB tensor + `SetFlag/WaitFlag<HardEvent::V_S>`），避免手写多项式 |
| V3 顺序算法 | 拆成「并行矩阵（每对独立）+ 向量归约（如 NMS 的 `maskTemp &= maskRow`）」 |
| V4 流水 | `TQue<..., BUFFER_NUM=2>` 双缓冲 CopyIn/Compute/CopyOut |
| V5 同步 | 向量写→标量读用 `SetFlag/WaitFlag<HardEvent::V_S>`；DataCopy 后消费用 EnQue/DeQue 或 `V_MTE2` 事件 |

**常见易错点（编码后逐条自检）**：

- R1 多核标量写共享缓存行 → 写丢/错乱：记录按 64B 对齐（stride=16 float）或改 DMA；同输入连跑 3 次确定性验证
- R2 `DataCopy` 异步未同步 → 读旧数据：必须 EnQue/DeQue 或 `V_MTE2` 事件
- R3 向量写 UB → 标量读：`PipeBarrier<PIPE_V>` 可能不够，用 `SetFlag/WaitFlag<HardEvent::V_S>` 或 `PIPE_ALL`
- R4 算术掩码 `0/0→NaN` 复活已抑制项：分母加极小量（如 `1e-20f`）
- R5 golden 转写符号/象限易错：先做中间量交叉验证（dump 中间 buffer vs Python golden），再跑最终精度
- R6 稀疏/上三角输出：host 预清零（`at::zeros`）或对跳过区做向量 ones 写

> 详情见 `ascendc-vector-dev`（向量化模式 + 同步正确性 + 成熟算子参考）。

---

## 第二部分：算子代码生成

### 步骤一：选择开发形态与工程模板

| 形态 | 触发 | 工程模板 Skill |
| ------ | ------ | ------ |
| **Kernel 直调** | POC / 单算子验证 | `ascendc-direct-invoke-template`（add_custom 样例，`<<<>>>` 调用） |
| **AscendDSL** | 类 Python 快速开发 | `dsl-baseline-generation` → `dsl-lowering` |
| **自定义算子(aclNN)** | 生产入库 | `ascendc-registry-invoke-template` |
| **CATLASS(Blaze)** | 950 Matmul/融合 | `catlass-op-design` + `catlass-op-develop` |
| **Torch 扩展（便于 pytest）** | 复用 ops-test-gen | `torch-ascendc-op-extension`（直调 kernel → `torch.ops.npu.xxx`） |

> **推荐路径**：直调 kernel 开发 → 通过 `torch-ascendc-op-extension` 注册为 `torch.ops.npu.xxx`，
> 即可复用 `ops-test-gen`（pytest + `torch_npu.profiler`）完成 L1–L5 验收，与 Triton 路径测试层一致。
> **适配层选型（`ascendc-adapter-layer`）**：编码前确定调用方式——单算子（`<<<>>>` + torch 扩展，
> 快速验证/POC）还是 aclnn 注册（生产/入图）。交付 README 必须说明用户如何调用算子。

### 步骤二：生成 Ascend C 算子代码（以 Kernel 直调为例）

参考 `ascendc-direct-invoke-template` 工程结构：

```cpp
// kernel_src/my_op.cpp
#include "kernel_operator.h"
using namespace AscendC;

class MyOp {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t total) { ... }
    __aicore__ inline void Process() {
        // CopyIn -> Compute -> CopyOut 三段流水
    }
};

extern "C" __global__ __aicore__ void my_op(GM_ADDR x, GM_ADDR y, uint32_t total) {
    MyOp op;
    op.Init(x, y, total);
    op.Process();
}
// 调用：my_op<<<blockDim, nullptr, stream>>>(x, y, total);
```

#### 编码原则

- 多核切分按设计文档 `block_dim`，尾块用 `process` + mask 防越界
- UB 通过 `TPipe::AllocTensor` 分配，double buffer 用 `SetBuffer` + `EnQue/DeQue`
- 优先 fp16/bf16，关键累加用 fp32
- 同步用 `PipeBarrier` / `SetFlag`+`WaitFlag`

### 步骤三：昇腾 NPU 适配与上板

1. **阶段 0 环境确认**（必做）：`ascendc-env-check` 确认 CANN / 设备 / 架构
2. **编译**：直调工程用 CMake 或模板自带编译脚本；远端 `docker exec` 编译
3. **运行**：SSH + `docker exec` 运行测试程序或 pytest（torch 扩展路径）

---

## 第三部分：调试（hypothesis-driven）

**任何运行异常必须先调用 `ascendc-op-debug` / `ascendc-precision-debug` 做假设驱动定位，再修复。**

### 性能/精度/崩溃路由

| 现象 | 首选 Skill |
| ------ | ------ |
| 输出异常（全 0 / 随机 / 未初始化 / NaN） | `ascendc-precision-debug` |
| 精度验证 rtol/atol 不达标 | `ops-precision-standard` + `ascendc-precision-debug` |
| 程序卡死 / 挂起 / Timeout | `ascendc-crash-debug`（plog 解析、死锁、buffer 冲突） |
| 崩溃（SegFault / Abort / aic error） | `ascendc-crash-debug` |
| aclnn 错误码（161xxx/361xxx/561xxx） | `ascendc-runtime-debug`（Tiling / Kernel 查找） |
| 偶发崩溃 / 内存越界踩踏 | `ascendc-crash-debug`（msSanitizer） |

---

## 第四部分：性能优化

> 前置：算子已通过正确性验证。优化是迭代过程——每次只改一个维度，测量对比后再下一轮。

### 步骤四：瓶颈定位

- 首选 `ops-profiling`（msprof / `torch_npu.profiler` → `kernel_details.csv`，与 Triton 路径共用）
- 950 专用：参考 `ascendc-perf-optimize`、`code-performance-advisor`、`ascendc-blaze-best-practice`（Blaze）

### 步骤五：优化手法（按瓶颈索引）

| 瓶颈 | 手法 | 参考 |
| ------ | ------ | ------ |
| UB 溢出 | 减小 TILE / 二次切分 / 关 double buffer | `ascendc-api-best-practices` |
| 多核负载不均 | 重新切分 block_dim / 尾块均衡 | `ascendc-tiling-design` |
| 流水断裂 | CopyIn/Compute/CopyOut 重排，double buffer | `ascendc-perf-optimize` |
| Cube/Vector 协同差（CV） | CATLASS tile 调整（`catlass-op-perf-tune`） | `catlass-op-design` |
| 950 SIMT 线程排布差 | 参考 `ascendc-simt-tiling-design` | `ascendc-simt-best-practices` |
| 950 Matmul | Blaze / CATLASS tile shape 调优 | `ascendc-blaze-best-practice` |
| 搬运占比高（MTE2） | 连续访存 / buffer 复用 / 对齐 | `ascendc-perf-optimize` |

### 步骤六：每轮精度复验

优化后重跑精度（torch 扩展 → pytest；或直调测试程序），确认无回归。

---

## 第五部分：代码检视

调用 `ascendc-code-review` 做假设检验驱动的逐条检视：

- 硬件约束合规（UB / 对齐 / 多核越界）
- 正确性（dtype 一致 / broadcast / reduction 语义 / NaN 路径）
- 性能反模式（流水断裂 / buffer 未复用）
- 反作弊（无 torch/triton 包装）

---

## 第六部分：与状态机集成

`ops-implement-agent` 完成后输出 `→ handoff: IMPLEMENT_DONE`；`ops-optimize-agent` 每轮输出
`→ handoff: OPTIMIZE_CONTINUE` / `OPTIMIZE_DONE`，由 `state_machine.py` 驱动（与 Triton 路径完全一致）。

---

## 优化模式速查表

| 症状 | 优化手段 | 预期收益 |
| ------ | --------- | :---: |
| UB overflow | TILE 二次切分 | — |
| 多核尾块不均 | block_dim 重新切分 | 1.2–2× |
| 流水断裂 | double buffer / 三段重排 | 1.2–2× |
| 搬运瓶颈 | 连续访存 + 对齐 | 1.5–3× |
| CV 协同差 | CATLASS tile 调优 | 1.2–2× |
| 950 Matmul | Blaze tile shape | 1.5–3× |

---

## 注意事项

- **一次只改一个维度**：同时改 TILE 和 block_dim 无法判断哪个有效
- **每次改后重测精度**：优化可能改变浮点运算顺序
- **架构代际差异**：A2/A3 → 950 迁移重新核算 UB 与核数
- **先调试后修复**：运行异常必须先走 ascendc-op-debug 系列，禁止无依据盲改

## 参考信息

- [CANN 算子开发指南](https://www.hiascend.com/document)

### 相关 Skill（可复用，来自 cannbot / 本仓库）

| Skill | 关系 |
| ------ | ------ |
| `ascendc-direct-invoke-template` | 核心 — Kernel 直调工程模板 |
| `ascendc-vector-dev` | **核心** — 向量化开发模式/同步正确性/常见易错点（开发必读） |
| `ascendc-api-best-practices` | **核心** — API 正确用法与限制（DataCopy/对齐/buffer），开发时对照 |
| `ascendc-adapter-layer` | **核心** — 适配层设计：单算子(`<<<>>>`+torch) vs aclnn 调用选型、适配代码、交付 README |
| `dsl-baseline-generation` / `dsl-lowering` | 核心 — AscendDSL 生成与下沉 |
| `ascendc-registry-invoke-template` | 核心 — aclNN 自定义算子工程 |
| `torch-ascendc-op-extension` | 核心 — 直调 kernel → torch.ops.npu.xxx（复用 pytest） |
| `catlass-op-design` / `catlass-op-develop` / `catlass-op-perf-tune` | 核心（950 Matmul）— CATLASS / Blaze |
| `ascendc-tiling-design` | 核心 — Tiling 方法论 |
| `npu-arch` / `ascendc-env-check` | 核心 — 架构参数与环境的确认 |
| `ascendc-api-best-practices` / `ascendc-docs-search` | 核心 — API 用法与文档检索 |
| `ascendc-op-debug` / `ascendc-precision-debug` / `ascendc-runtime-debug` / `ascendc-crash-debug` | 调试 — 假设驱动定位 |
| `ascendc-perf-optimize` / `code-performance-advisor` / `ascendc-blaze-best-practice` | 优化 — 性能调优 |
| `ascendc-code-review` / `cake-code-review` | 检视 — 代码质量与合规 |
| `ops-test-gen` / `ops-boundary-survey` / `ops-profiling` / `ops-precision-standard` | 验收 — 与 Triton 路径共用 |
