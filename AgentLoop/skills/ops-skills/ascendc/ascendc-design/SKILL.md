---
name: ascendc-design
description: >-
  This skill should be used when the user asks to analyze an operator requirement and
  produce a detailed Ascend C operator design (hardware constraints, Tiling strategy,
  memory plan, kernel architecture, interface specs, test strategy) for Atlas A2/A3
  (arch22) or Ascend 950 (arch35 / DAV_3510), before any code is written. It reuses
  ascendc-tiling-design, npu-arch, ascendc-docs-search and ascendc-api-best-practices.
  Typical triggers include "帮我设计 Ascend C 算子方案", "写个 aclnn 算子设计文档",
  "这个算子用 Ascend C 怎么实现", "规划 Tiling 和 UB 方案", "评估 Arch22→Arch35 迁移".
version: 1.0.0
domain: ops
function: ascendc
tags:
  - ascendc
  - design
  - tiling
  - planning
  - npu
author: mindspeed-agent
---

# Ascend C 算子需求分析与详细设计

## 功能概述

本 Skill 覆盖算子开发三阶段流水线中**阶段一**的 Ascend C 路径：从原始需求出发，完成可行性评估 →
详细技术设计（硬件约束 / Tiling / 内存 / 调度 / 架构）→ 任务拆解与风险登记，产出下游
`ops-implement-agent` 可直接执行的完整设计文档。

> 与 `ops-design`（Triton 路径）平行：选择哪种开发方式由 `ops-design-agent` 的「开发方式决策」决定。
> 本 Skill 仅在 `dev_lang = ascendc` 时被调用。

## 定位：三阶段流水线（Ascend C 路径）

```text
┌──────────────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  阶段一：需求分析与详设         │ →  │  阶段二：算子开发与优化  │ →  │  阶段三：精度性能验收    │
│  ascendc-design（本 Skill）│    │  ascendc-op-dev       │    │  ops-test-gen           │
└──────────────────────────────┘    └──────────────────────┘    └──────────────────────┘
```

## 适用场景

- 算子需要精细控制内存 / 同步 / 流水线（Triton 表达力不足）
- 自定义复合算子、复杂原子操作、动态线程通信、递归控制流
- Matmul / Attention 类 Cube 算子，需要在 Ascend 950 上使用 Blaze / CATLASS 或 RegBase
- GPU CUDA kernel → 昇腾 NPU 迁移（行对行改造为 Ascend C）
- 同一算子需要分别生成 A2/A3 与 Ascend 950 两套实现

## 前置条件

| 输入项 | 来源 | 必要性 |
| ------- | ------ | :---: |
| 算子需求描述（CUDA 代码 / 自然语言 / 论文 / PyTorch 实现） | 用户 | 必须 |
| 目标架构（Arch22 = A2/A3，或 Arch35 = Ascend 950） | 用户或环境 | 必须 |
| 目标开发形态 | 用户 | 推荐：Kernel 直调 / AscendDSL / 自定义算子(aclNN) / CATLASS(Blaze) |

如关键输入缺失，应主动向用户确认后再开始。

---

## 第一部分：需求分析与可行性评估

### 步骤一：需求信息收集

向用户确认以下信息，缺失项标注为"待确认"：

1. **算子来源**：CUDA 代码 / GPU Triton / 论文算法 / 自然语言 / PyTorch 实现
2. **算子功能**：核心计算逻辑（element-wise / reduction / matmul / attention / conv / 自定义）
3. **输入输出**：shape（含维度语义和范围）、dtype、layout、是否含 workspace
4. **精度要求**：目标 dtype、允许误差（rtol / atol）
5. **性能目标**：延迟上限 / 吞吐目标 / 对标 CUDA 或 torch 的 speedup
6. **目标架构**：A2 / A3（arch22）或 Ascend 950（arch35 / DAV_3510）
7. **开发形态**：Kernel 直调（`<<<>>>`） / AscendDSL / 自定义算子工程（aclNN）/ CATLASS(Blaze) / RegBase（950）

#### 输入类型处理指南

| 输入形式 | 处理方式 |
| --------- | --------- |
| **CUDA 源码**（`.cu`） | 提取计算语义 → Ascend C kernel 改造；参考 `ascendc-docs-search` 查 API 映射 |
| **PyTorch 实现** | 提取 `forward` 计算逻辑，标注与 Ascend C 的语义差异（broadcast / reduction 语义） |
| **自然语言 / 论文** | 拆解为可独立实现的算子模块，按依赖排序 |

### 步骤二：NPU 迁移可行性评估

调用 `npu-arch` 确认目标架构能力，按以下维度评估：

| 评估维度 | 检查项 | 结论 |
| --------- | ------ | ------ |
| **算子类型** | element-wise / reduction / matmul / attention / conv / 自定义 | 标注 Ascend C 支持度（★1–5） |
| **GPU 特性依赖** | warp shuffle / shared memory / CUDA 特定库 | 标注 Ascend C 替代方案（TPipe / 多核并行） |
| **硬件约束** | UB 容量 / L1/L0 层级 / 对齐 / 多核数 | 标注是否可满足（`npu-arch` 提供代际参数） |
| **精度风险** | fp16 累积误差 / 特殊数学函数 | 标注风险等级 |
| **性能风险** | 是否 Cube/Vector 协同瓶颈 | 标注评估 |

#### 不可行场景速查

| 场景 | 原因 | 替代方案 |
| ------ | ------ | --------- |
| 信息严重缺失 | 无法设计 Tiling | 请用户补充 |
| Ascend C 仍不支持的操作 | 编程模型限制 | 拆 Host 侧 / 多算子组合 |
| A2/A3 → 950 语义差异过大 | API 不兼容 | 走 `ascendc-cross-gen-port`（跨代迁移） |

> **核心原则**：发现不可行时，明确说明具体原因和替代方案，不强行设计。

---

## 第二部分：算子详细设计

### 步骤三：硬件约束分析（按架构分支）

调用 `npu-arch` 与 `ascendc-tiling-design` 获取架构参数。

#### 3.1 芯片参数速查

| 参数 | A2 / A3（arch22） | Ascend 950（arch35 / DAV_3510） |
| ------ | --------- | ----- |
| UB (Unified Buffer) | 192 KB | 256 KB |
| L1 / L0 层级 | 标准 AIC | 扩展层级（参考 `ascendc-blaze-best-practice` 若走 Blaze） |
| AI Core 数 | 视产品（如 910B 20/24） | 视产品（如 950 系列） |
| 编程模型 | SIMD（Vector） + Cube | SIMD / SIMT（950 引入 SIMT，`ascendc-simt-best-practices`） |
| 多核并行 | 固定 block_dim（核数） | 同左，核数可能更多 |
| 直调工程 | `ascendc-direct-invoke-template`（add_custom 样例） | 同左；Blaze 用 `ascendc-blaze-best-practice` |
| Matmul 算子 | 自写 Cube kernel / aclNN | CATLASS(Blaze) `catlass-op-design` + `catlass-op-develop` |

> **跨架构注意**：950 的 UB 更大、引入 SIMT 线程级并行；同一算子从 A2/A3 迁到 950 需重新核算 Tiling 与核数。反之 950→A2/A3 可能 UB 溢出。

#### 3.2 内存预算计算（UB 视角）

```python
UB_TOTAL = 192 * 1024      # A2/A3；950 用 256 * 1024
UB_EXT = 8 * 1024 # 预留给中间接口的UB，不用于计算和搬运
UB_ABLE = UB_TOTAL - UB_EXT # 计算与搬运过程的可用UB空间
UB_DOUBLEBUFFER = UB_ABLE // 2   # 开启 double buffer 后的有效 UB
BYTES_PER_ELEM = 2         # fp16
# 估算单核可容纳元素数，结合 block_dim（核数）得到整体 Tiling
```

#### 3.3 对齐分析

| 算子类型 | 对齐要求 | 说明 |
| --------- | --------- | ------ |
| 纯 Vector | 尾轴按数据类型对齐（fp16 32B / fp32 64B 等，参考 `ascendc-api-best-practices`） | 用 `DataCopy` 全量搬运时对 last axis 有对齐约束 |
| Cube (Matmul) | M/N/K 分块对齐 Cube 粒度 | 参考 `catlass-op-design` TileShape |

---

### 步骤四：Tiling 方案设计

调用 `ascendc-tiling-design` 完成：多核切分（`block_dim`）、核内 UB 切分（`TILE` / `BLOCK`）、
double buffer 规划、分支覆盖。

#### 4.1 Vector 算子（element-wise / reduction）

```cpp
// 伪代码：一维切分
uint32_t total = ...;
uint32_t blockDim = ...;          // 核数，参考 npu-arch
uint32_t tile = (total + blockDim - 1) / blockDim;
// 每核处理 [pid*tile, min((pid+1)*tile, total))
// 尾块用 mask / process 控制，避免越界
```

#### 4.2 Cube 算子（Matmul / Attention）

- M/N/K 分块满足 Cube 粒度，UB 内放得下 A/B/C 分块 + 累加器
- 参考 `catlass-op-design`（950）或自写 Cube kernel（A2/A3）
- 多核按 M 维切分，K 维做流水

#### 4.3 跨代 Tiling 决策

| 场景 | 推荐 |
| ------ | ------ |
| A2/A3 纯 Vector | SIMD + `block_dim` 固定核数 + UB Tiling |
| Ascend 950 纯 Vector | 可选 SIMT（`ascendc-simt-tiling-design`），线程级并行 |
| Ascend 950 Matmul | CATLASS / Blaze（`catlass-op-design` + `catlass-op-develop`） |
| A2/A3 Matmul | 自写 Cube kernel 或 aclNN |

---

### 步骤五：内存规划

#### 5.1 UB 分配表

| Buffer (pipe) | 用途 | 大小 (bytes) | 生命周期 | 可复用 |
| -------- | ------ | :---: | ------ | :---: |
| que_in | 输入队列（TPipe AllocTensor） | TILE×dtype | EnQue→计算 | — |
| que_out | 输出队列 | TILE×dtype | 计算→DeQue | 可复用 in |
| tmp | 中间结果 | ... | 计算内部 | — |

> Ascend C 使用 `TPipe::AllocTensor` 管理 UB，double buffer 通过 `SetBuffer` + `EnQue/DeQue` 流水实现。
> 估算用有效 UB（开 double buffer 时减半）。

#### 5.2 Buffer 复用与 double buffer

- 生命周期不重叠的 queue 复用同一块 UB
- 计算/搬运瓶颈开 double buffer 隐藏延迟

---

### 步骤六：计算调度设计

#### 6.1 多核分配

```cpp
// Ascend C 由框架按 block_dim 分发核号
// 核内用 pipe（VECIN/VECOUT/CONV）组织流水
```

#### 6.2 流水（Pipeline）

- Vector：CopyIn → Compute → CopyOut 三段流水（`PipeBarrier` 同步）
- 用 `SetFlag` / `WaitFlag` 或 `EnQue`/`DeQue` 管理阶段同步

---

### 步骤七：Kernel 架构与接口规格

#### 7.1 开发形态选择

| 形态 | 适用 | 工程模板 |
| ------ | ------ | ------ |
| **Kernel 直调** | 快速验证 / 单算子 POC | `ascendc-direct-invoke-template`（`<<<>>>` 调用） |
| **AscendDSL** | 类 Python 快速开发 | `dsl-baseline-generation` + `dsl-lowering` |
| **自定义算子(aclNN)** | 生产入库 / 接入图 | `ascendc-registry-invoke-template` |
| **CATLASS(Blaze)** | 950 Matmul/融合 | `catlass-op-design` + `catlass-op-develop` |
| **RegBase** | 950 Reg API 算子 | `ascendc-regbase-best-practice` |

#### 7.2 接口签名（直调示例）

```cpp
extern "C" __global__ __aicore__ void my_op(GM_ADDR x, GM_ADDR y, uint32_t total) {
    // 使用 kernel 直调模板的 Kernel 类
}
// 调用：my_op<<<blockDim, nullptr, stream>>>(x, y, total);
```

#### 7.3 反作弊约束（下游门禁）

- **禁止**用 `torch.nn.functional.*` / `F.xxx` / Triton 包装冒充 Ascend C 实现
- 必须是真实 Ascend C kernel（`.cpp` + 直调 / DSL / aclNN），`DataCopy`/`Compute` 逻辑可审计

#### 7.4 交付件结构（设计文档必须明确 dev_form 对应的交付形态）

> 生产入库（aclnn 注册）的交付件中，`op_host/<op>.cpp` 是 **OpDef 注册 + Tiling 函数 + InferShape**，
> **没有 main**；`<<<>>>` 直调工程只是 POC/验证形态，不是生产交付件。

**① 生产交付件（dev_form = aclnn，推荐入库/接入图）**：

```text
kernels/<op>/
├── op_host/
│   ├── <op>.cpp            # OpDef 注册（class <Op> : public OpDef，Input/Output/Attr、
│   │                       #   SetInferShape/SetInferDataType、AICore().SetTiling、OP_ADD）
│   │                       #   + Tiling 函数（TilingContext → SaveToBuffer → SetBlockDim → workspace）
│   └── <op>_tiling.h       # Tiling 数据结构（BEGIN_TILING_DATA_DEF + REGISTER_TILING_DATA_CLASS）
└── op_kernel/
    └── <op>.cpp            # device kernel（Kernel 类 + extern "C" 入口 + GET_TILING_DATA）

适配层（用户 Python 调用入口）：
├── ops/<op>.py             # torch.autograd.Function 包装（forward 调 csrc）
├── csrc/<Op>.cpp           # aclnn 调用（EXEC_NPU_CMD(aclnn<Op>, ...)）
├── csrc/function.h         # C++ ABI 函数声明
└── csrc/pybind.cpp         # pybind 绑定（m.def("npu_<op>", &npu_<op>)）
```

**② POC / 快速验证工程（dev_form = direct-invoke，`<<<>>>`，非生产交付件）**：

```text
op_kernel/<op>.cpp          # device kernel（ASC 语言）
op_host/<op>.cpp            # host + main（`<<<>>>` 调用）+ data_utils.h
op_extension/               # torch 扩展（可选，注册 torch.ops.npu.<op> 复用 pytest）
CMakeLists.txt              # 双 target：可执行文件 + torch 扩展 .so
```

> 设计文档「交付清单」节必须写明采用哪种 dev_form 及对应文件清单；生产入库选 ①，POC/验证选 ②。

---

### 步骤七·五：向量化友好设计（设计阶段必做）

> 向量化是基础，**设计阶段**就要把算法形态设计成向量友好，不能等编码后再补。配套
> `ascendc-vector-dev` 落地。设计文档必须回答以下三点：

1. **算法形态是否可并行化**：把「串行依赖」拆成「并行矩阵 + 向量归约」。
   - 例：NMS 输出 int16 **mask 矩阵** `mask[i][j]`（每对独立计算，按核×列块并行），
     顺序贪心挪到独立 gather kernel 用向量 `&`（`maskTemp &= maskRow`）归约。
   - 设计文档需标注：哪个 kernel 全并行、哪个 kernel 含顺序依赖、顺序部分如何向量化。
2. **数据移动与布局**：批量数据按 16/64B 对齐用 DMA（`DataCopy/DataCopyPad`）搬运；
   每记录 stride 按 64B（16 float）对齐避免多核标量写踩缓存行；上三角/稀疏输出 host 预清零
   或向量 ones 写。
3. **数学函数与同步**：`sin/cos/atan` 用向量硬件指令（标量入 UB tensor + `SetFlag/WaitFlag<V_S>`）；
   数据流标注 MTE→V、V→S 同步点（TQue EnQue/DeQue 或 HardEvent）。

**验收项**：设计文档的 Tiling/内存/调度章节必须包含上述三点的明确结论，否则回退设计。

---

### 步骤八：测试策略设计

#### 8.1 Shape × Dtype 覆盖矩阵

| 编号 | Shape | Dtype | 测试目的 |
| :---: | ------ | ------ | ------ |
| 1 | 典型场景 shape | float16 | 基础精度 |
| 2 | 大 shape | float16 | UB / 多核合规 |
| 3 | 非对齐尾轴 | float16 | 对齐鲁棒性 |
| 4 | 典型 shape | bfloat16 | bf16 精度 |
| 5 | 典型 shape | float32 | fp32 精度 |

#### 8.2 边界场景

| 编号 | 场景 | 预期行为 |
| :---: | ------ | ------ |
| E1 | 空张量 `N=0` | 不崩溃 |
| E2 | 单元素 `N=1` | 精度通过 |
| E3 | 全零输入 | 无 NaN/Inf |
| E4 | 极大值 | 无 NaN/Inf |
| E5 | 多核尾块不均 | 每核边界正确 |

#### 8.3 性能基线

| Shape | 目标延迟 | 对标 |
| ------ | :---: | :---: |
| 典型 shape | ≤ Xms | torch / CUDA 参考 |
| 大 shape | ≤ Yms | ≥ 1.5× torch |

#### 8.4 验收门禁

- [ ] 编译通过（CMake / 直调模板编译无错）
- [ ] UB 占用 ≤ 目标架构上限
- [ ] 多核切分正确，无越界
- [ ] 所有 shape × dtype 精度通过（参考 `ops-precision-standard`）
- [ ] 边界场景无异常
- [ ] 性能达标
- [ ] 真实 Ascend C kernel，无 torch/triton 包装

---

## 第三部分：任务规划与风险管理

### 步骤九：任务拆解（T1–T6）

```markdown
## 任务清单：<算子名称>（Ascend C / <arch22|arch35>）

### T1: 需求分析与可行性评估
- 产出：需求规格 + 可行性报告（含目标架构）
- 执行 Agent：ops-design-agent（ascendc-design）

### T2: 算子代码生成
- 产出：Ascend C kernel + 直调/DSL 工程 + 测试 harness
- 执行 Agent：ops-implement-agent（ascendc-op-dev）
- 门禁：编译通过 + 基础精度达标

### T3: 性能优化
- 产出：优化后 kernel + 性能报告（msprof / ops-profiling）
- 执行 Agent：ops-implement-agent / ops-optimize-agent（ascendc-op-dev）
- 门禁：精度复验通过 + 性能达标

### T4: 代码检视
- 产出：检视报告（ascendc-code-review）
- 执行 Agent：ops-implement-agent

### T5: 精度验收
- 产出：精度测试报告（ops-test-gen + ops-precision-standard）
- 执行 Agent：ops-implement-agent

### T6: 性能验收
- 产出：性能测试报告 + 基线（ops-profiling）
- 执行 Agent：ops-implement-agent
```

### 步骤十：风险登记

```markdown
| # | 风险 | 等级 | 缓解措施 |
|---|------|:---:|---------|
| 1 | UB 溢出（大 shape） | 🔴 | 减小 TILE / 二次切分 / 关 double buffer |
| 2 | 多核尾块越界 | 🔴 | mask + process 控制 |
| 3 | 950 SIMT 线程排布错误 | 🟡 | 参考 ascendc-simt-tiling-design |
| 4 | 精度累积误差 | 🟡 | 关键累加 fp32 |
```

---

## 输出模板

完成以上步骤后，按 `ops-design` 的 12 节模板输出完整设计文档，并在第 2 节「开发方式」明确标注：
`dev_lang = ascendc`，`target_arch = arch22 | arch35`，`dev_form = direct-invoke | dsl | aclnn | catlass`。

---

## 注意事项

- **先算后画**：先完成 UB / 多核 / 对齐数值计算，再套模板
- **架构代际差异**：A2/A3 与 950 的 UB、核数、SIMT 支持不同，Tiling 不可直接套用
- **开发形态决定下游路径**：直调 → pytest + torch 扩展（`torch-ascendc-op-extension`）；aclNN → 完整 UT/ST
- **必要时回退**：步骤二发现不可行时明确标记并说明原因
- **信息不足**：列出待确认项，不强行推测

## 参考信息

- [CANN 算子开发指南](https://www.hiascend.com/document)

### 相关 Skill（可复用）

| Skill | 关系 |
| ------ | ------ |
| `ascendc-tiling-design` | 核心 — Tiling 方法论（多核切分 / UB 切分 / buffer 规划） |
| `ascendc-vector-dev` | 核心 — 向量化友好设计约束（并行矩阵+向量归约/DMA 布局/同步） |
| `npu-arch` | 核心 — 目标架构参数（A2/A3 vs 950）与条件编译 |
| `ascendc-api-best-practices` | 核心 — API 正确用法与限制（DataCopy / 对齐 / buffer） |
| `ascendc-docs-search` | 辅助 — 查 API 文档与示例 |
| `ascendc-op-dev` | 下游 — 按本设计生成并优化 Ascend C kernel |
| `catlass-op-design` / `catlass-op-develop` | 下游（950 Matmul）— CATLASS / Blaze 组件选择 |
| `ops-test-gen` | 下游 — 精度/性能测试框架（与 Triton 路径共用） |
| `ops-profiling` | 下游 — msprof 性能采集（与 Triton 路径共用） |
