---
name: ops-design
description: >-
  This skill should be used when the user asks to analyze operator requirements,
  assess NPU migration feasibility, produce a detailed operator design document
  (tiling strategy, memory plan, kernel architecture, interface specs, test
  strategy), break down operator development into executable tasks with effort
  estimates, or create the complete planning-to-design deliverable before coding.
  Typical triggers include "帮我设计这个算子的方案", "写个详细设计文档",
  "这个算子该怎么实现", "梳理开发任务", "评估迁移工作量和风险",
  or "帮我规划 tiling 策略和 grid 方案".
version: 1.0.0
domain: ops
function: triton
tags:
  - design
  - tiling
  - planning
  - feasibility
  - task-management
  - ascend
author: mindspeed-agent
compatibility: MindSpeed >= 0.1.0, CANN >= 9.1.0, Triton >= 3.2.2
targets-repo:
  - triton-ascend
---

# 需求分析与算子开发详设

## 功能概述

本 Skill 覆盖算子开发三阶段流水线中的**阶段一**全部工作：从原始需求出发，完成可行性评估 → 详细技术设计 → 任务拆解与风险登记，产出下游 Agent 可直接执行的完整设计文档。

## 定位：三阶段流水线

```text
┌──────────────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  阶段一：需求分析与详设         │ →  │  阶段二：算子开发与优化  │ →  │  阶段三：精度性能验收    │
│  ops-design（本 Skill）│    │  triton-op-dev         │    │  ops-test-gen           │
└──────────────────────────────┘    └──────────────────────┘    └──────────────────────┘
```

本 Skill 是 `ops-design-agent` 的核心能力，产出交付给阶段二的 `ops-implement-agent`。

## 适用场景

- 收到算子开发需求后，需要从分析到设计到规划的完整方案
- GPU 算子迁移到 NPU 前，需要评估可行性和设计适配方案
- 复杂算子（Attention / Matmul / Conv）需要系统性设计 Tiling 和内存方案
- 多个算子并行开发，需要统一规划优先级和工时
- 用户不确定从哪一步开始，需要完整的开发蓝图

## 前置条件

| 输入项 | 来源 | 必要性 |
| ------- | ------ | :---: |
| 算子需求描述（GPU 代码 / 自然语言 / 论文） | 用户 | 必须 |
| 目标芯片（A2 / A3 / A5） | 用户或环境 | 必须 |
| 性能目标 | 用户 | 推荐 |

如关键输入缺失，应主动向用户确认后再开始。

---

## 第一部分：需求分析与可行性评估

### 步骤一：需求信息收集

向用户确认以下信息，缺失项标注为"待确认"：

1. **算子来源**：GPU Triton / CUDA 代码 / 论文算法 / 自然语言描述 / PyTorch 实现
2. **算子功能**：核心计算逻辑（element-wise / reduction / matmul / attention / conv / 自定义）
3. **输入输出**：shape（含维度语义和范围）、dtype、layout
4. **精度要求**：目标 dtype、允许误差（rtol / atol）
5. **性能目标**：延迟上限 / 吞吐目标 / speedup 期望
6. **目标芯片**：A2 / A3 / A5
7. **交付时间**：紧急程度

#### 输入类型处理指南

| 输入形式 | 处理方式 |
| --------- | --------- |
| **GPU Triton 代码** | 提取计算逻辑、标注 GPU 特定假设（thread/block 模型、共享内存大小、warp shuffle 等），列出 NPU 差异点 |
| **CUDA 源码**（`.cu` / `.cuh`） | 作为算法参考，提取计算语义；不做行对行翻译（CUDA SIMT 与 Triton 编程范式差异大） |
| **自然语言描述** | 主动澄清不明确信息，规格化为结构化描述 |
| **论文 / 算法** | 拆解为可独立实现的算子模块，按依赖关系排序 |
| **PyTorch 实现** | 提取 `forward` 计算逻辑，标注与 Triton kernel 的语义差异 |

### 步骤二：NPU 迁移可行性评估

按以下维度逐一评估：

| 评估维度 | 检查项 | 结论 |
| --------- | ------ | ------ |
| **算子类型** | element-wise / reduction / matmul / attention / conv / 自定义 | 标注 Triton 支持度（★1–5） |
| **GPU 特性依赖** | warp shuffle / shared memory 大小 / CUDA 特定库 / 动态线程通信 | 标注替代方案或不可行原因 |
| **NPU 硬件约束** | Grid ≤ 65535 / UB 容量 / 对齐要求 / 指针别名 | 标注是否可满足 |
| **精度风险** | fp16 累积误差 / 特殊数学函数 / 动态范围 | 标注风险等级（高/中/低） |
| **性能风险** | 预期能否达到性能目标 | 标注评估 |
| **Triton 语言支持** | 所需操作是否在 Triton 表达能力内 | 不支持的标注替代方案 |

#### 不可行场景速查

| 场景 | 原因 | 替代方案 |
| ------ | ------ | --------- |
| 信息严重缺失（不知道算什么、shape/dtype） | 无法设计 tiling | 请用户补充 |
| Grid + UB 夹逼无解（N 极大，增大 BLOCK_SIZE 到 UB 极限仍无法让 grid ≤ 65535） | 物理上限冲突 | 拆分为多 kernel 串行 |
| Triton 不支持的核心操作（动态线程间通信、复杂原子操作、递归） | 编程模型限制 | 改用 Ascend C |
| 外部依赖/状态（文件 I/O、网络通信） | kernel 是无状态纯计算单元 | Host 侧管理状态 |

> **核心原则**：发现不可行时，明确说明具体原因和替代方案，不强行设计。

---

## 第二部分：算子详细设计

可行性评估通过后，按以下步骤逐项完成技术设计。

### 步骤三：硬件约束分析

根据目标芯片，计算关键数值边界。

#### 3.1 芯片参数速查

| 参数 | A2 / A3 | A5 |
| ------ | --------- | ----- |
| UB 总大小 | 192KB | 256KB |
| UB 有效大小（doublebuffer 默认开启） | ~96KB | ~128KB |
| Grid 上限 | 65535 | 65535 |
| VV 对齐 | 32B | 32B |
| CV 对齐 | 512B | 512B |

#### 3.2 内存预算计算

```python
UB_EFFECTIVE = 96 * 1024  # A2/A3，bytes；A5 用 128 * 1024
BYTES_PER_ELEM = 2        # float16 = 2B, float32 = 4B
MAX_ELEMS_PER_BLOCK = UB_EFFECTIVE // BYTES_PER_ELEM

# 假设 kernel 加载 X（输入）+ Y（输出）+ 1 个中间 buffer：
BUFFER_COUNT = 3
MAX_ELEMS_PER_BUFFER = MAX_ELEMS_PER_BLOCK // BUFFER_COUNT
```

#### 3.3 Grid 约束计算

```python
import triton
N = <total_elements>
MIN_BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(N, 65535))
MAX_BLOCK_SIZE_BY_UB = MAX_ELEMS_PER_BUFFER
# 若 MIN_BLOCK_SIZE > MAX_BLOCK_SIZE_BY_UB → 需要 BLOCK_SIZE_SUB
```

#### 3.4 对齐分析

| 算子类型 | 对齐要求 | 本算子尾轴 | 状态 |
| --------- | --------- | ----------- | :---: |
| VV / CV | 32B / 512B | ... | ✅ / ⚠️ |

---

### 步骤四：Tiling 方案设计

#### 4.1 一维算子

```python
N = <total_elements>
BLOCK_SIZE = 1024  # 在 [MIN_BLOCK_SIZE, MAX_BLOCK_SIZE_BY_UB] 内选 2 的幂
grid = (triton.cdiv(N, BLOCK_SIZE),)
# grid > 65535 → 增大 BLOCK_SIZE → UB 溢出 → BLOCK_SIZE_SUB
```

#### 4.2 二维算子（Matmul / Attention）

```python
M, N, K = <dims>
BM, BN, BK = 64, 64, 32  # 首选值
ub_usage = (BM * BK + BK * BN + BM * BN) * 2  # fp16, bytes
assert ub_usage <= UB_EFFECTIVE
grid_size = triton.cdiv(M, BM) * triton.cdiv(N, BN)
assert grid_size <= 65535
```

#### 4.3 BLOCK_SIZE_SUB 二次切分

当 BLOCK_SIZE 增大到满足 grid 约束后触发 UB 溢出时：

```python
MAIN_BLOCK_SIZE = 16384  # 满足 grid ≤ 65535
SUB_BLOCK_SIZE = 1024    # 控制单次 UB 占用
NUM_SUB_BLOCKS = MAIN_BLOCK_SIZE // SUB_BLOCK_SIZE

@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    for sub in range(NUM_SUB_BLOCKS):
        offs = base + sub * BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
        ...
```

#### 4.4 Tiling 决策速查

| 场景 | 推荐 BLOCK_SIZE | Grid 策略 |
| ------ | :---: | ------ |
| 小 shape（N < 4096） | 128–512 | 固定核心数 |
| 中 shape（4096 ≤ N < 10⁶） | 1024–2048 | `ceil(N / BLOCK_SIZE)` |
| 大 shape（N ≥ 10⁶） | 2048+ | 优先压 grid ≤ 65535 |
| 2D matmul（M,N 均中等） | BM=64, BN=64 | `ceil(M/BM), ceil(N/BN)` |
| 2D matmul（M 大 N 小） | BM=128, BN=32 | 不对称 tiling |
| Attention | BM=64, BN=64, BK=32 | 控制 K 循环次数 |

---

### 步骤五：内存规划

#### 5.1 UB 分配表

| Buffer | 用途 | 大小 (bytes) | 生命周期 | 可复用 |
| -------- | ------ | :---: | ------ | :---: |
| x_tile | 输入加载 | BLOCK_SIZE×2 | load → compute → store | — |
| y_tile | 输出暂存 | BLOCK_SIZE×2 | compute → store | — |
| tmp | 中间结果 | BLOCK_SIZE×2 | compute 内部 | — |
| **总计** | | **`<total> / <UB_EFFECTIVE>`** | | **`<ratio>%`** |

#### 5.2 Buffer 复用

生命周期不重叠的 buffer 复用同一块 UB：

```python
# 复用条件：前驱最后使用 < 后继首次使用，dtype 相同，大小取 max
```

#### 5.3 Doublebuffer 策略

| 场景 | 建议 | 理由 |
| ------ | :---: | ------ |
| 计算/内存瓶颈 | 开启 | 隐藏延迟 |
| UB 紧张且计算简单 | 可关闭 | 换取更大 BLOCK_SIZE |

---

### 步骤六：计算调度设计

#### 6.1 循环顺序

- 累加维度放最内层 → 累加器保留在寄存器
- 复用维度放外层 → 最大化数据复用

#### 6.2 核心分配

| 算子特征 | 使用核心 | 获取方式 |
| --------- | --------- | --------- |
| 纯向量算子（无 tl.dot） | Vector Core | `props["num_vectorcore"]` |
| 含 tl.dot | AI Core（= Cube Core） | `props["num_aicore"]` |

```python
NUM_CORE = props["num_vectorcore"]  # 或 num_aicore
grid = (NUM_CORE,)
# kernel 内用步长分发：for block_idx in range(pid, NUM_BLOCKS, NUM_CORE)
```

---

### 步骤七：Kernel 架构设计

#### 7.1 单 Kernel vs 多 Kernel

```text
算子是否需要 Cube Core 和 Vector Core 同时大量使用？
├── 是 → 考虑拆分为 2 个 kernel，减少 core 切换
└── 否 → 单 kernel UB 是否溢出？
         ├── 是 → 尝试减少中间变量、buffer 复用，仍不行则拆分
         └── 否 → 单 kernel
```

#### 7.2 Kernel Fusion 分析

| 融合对象 | 融合方式 | 收益 | 风险 |
| --------- | --------- | :---: | ------ |
| 上游 Norm | Norm+Matmul 融合 | 省一次 GM 读写 | UB 压力增大 |
| 下游激活 | kernel 末尾直接计算 | 省一次 kernel 启动 | 低风险 |
| 下游 Reshape | 调整 store stride | 省内存拷贝 | 低风险 |

#### 7.3 接口规格

```python
@triton.jit
def my_kernel(
    x_ptr, w_ptr, out_ptr,        # 数据指针
    B, N, C,                      # shape 维度
    stride_xb, stride_xn, stride_xc,  # stride
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
):
    ...

def launch(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Args: x (B,N,C) fp16, w (C,) fp16. Returns: out (B,N,C) fp16."""
    ...
```

---

### 步骤八：测试策略设计

#### 8.1 Shape × Dtype 覆盖矩阵

| 编号 | Shape | Dtype | 测试目的 |
| :---: | ------ | ------ | ------ |
| 1 | 典型场景 shape | float16 | 基础精度 |
| 2 | 大 shape | float16 | grid 合规 |
| 3 | 非对齐尾轴 | float16 | 对齐鲁棒性 |
| 4 | 典型 shape | bfloat16 | bf16 精度 |
| 5 | 典型 shape | float32 | fp32 精度 |

#### 8.2 边界场景

| 编号 | 场景 | 预期行为 |
| :---: | ------ | ------ |
| E1 | 空张量 `N=0` | 不崩溃 |
| E2 | 单元素 `N=1` | 精度通过 |
| E3 | 全零输入 | 无 NaN/Inf |
| E4 | 极大值（±65504 for fp16） | 无 NaN/Inf |
| E5 | 负值输入（如有 sqrt/log） | 按算子语义 |

#### 8.3 性能基线

| Shape | 目标延迟 | speedup vs PyTorch |
| ------ | :---: | :---: |
| 典型 shape | ≤ Xms | ≥ 2× |
| 大 shape | ≤ Yms | ≥ 3× |

#### 8.4 验收门禁

- [ ] 编译通过，无 MLIR/SCCL 错误
- [ ] UB 占用 ≤ 目标芯片上限
- [ ] Grid 总量 ≤ 65535
- [ ] 所有 shape × dtype 组合精度通过
- [ ] 边界场景无异常
- [ ] 性能达标
- [ ] `import torch_npu`、设备为 `npu`、无指针别名、`BLOCK_SIZE` 为 2 的幂

---

## 第三部分：任务规划与风险管理

### 步骤九：任务拆解（T1–T6）

将整个开发流程拆解为标准任务项：

```markdown
## 任务清单：<算子名称>

### T1: 需求分析与可行性评估
- **输入**：用户原始需求
- **产出**：需求规格文档、可行性评估报告
- **预估工时**：xx h
- **依赖**：无
- **执行 Agent**：ops-design-agent

### T2: 算子代码生成
- **输入**：T1 需求规格 + 详细设计文档
- **产出**：可编译的 Triton 算子代码 + 单元测试文件
- **预估工时**：xx h
- **依赖**：T1
- **执行 Agent**：ops-design-agent（阶段一）
- **门禁**：编译通过 + 基础精度达标

### T3: 性能优化
- **输入**：T2 算子代码
- **产出**：优化后代码 + 性能报告（speedup 对比）
- **预估工时**：xx h
- **依赖**：T2
- **执行 Agent**：ops-implement-agent（阶段二）
- **门禁**：精度复验通过 + 性能达标

### T4: 代码检视
- **输入**：T3 优化后代码 + 性能报告
- **产出**：检视报告 + 修正后代码
- **预估工时**：xx h
- **依赖**：T3
- **执行 Agent**：ops-implement-agent
- **门禁**：阻塞项清零

### T5: 精度验收测试
- **输入**：T4 修正后代码
- **产出**：精度测试报告
- **预估工时**：xx h
- **依赖**：T4
- **执行 Agent**：ops-implement-agent（精度部分）

### T6: 性能验收测试
- **输入**：T4 修正后代码
- **产出**：性能测试报告 + 性能基线
- **预估工时**：xx h
- **依赖**：T4（可与 T5 并行）
- **执行 Agent**：ops-implement-agent（性能部分）
```

### 步骤十：工作量评估

| 迁移难度 | 工作量（人天） | 典型场景 |
| :---: | :---: | ------ |
| 简单 | 0.5–1 | element-wise 算子，GPU 可直接映射 |
| 中等 | 1–3 | reduction / 小 matmul，需调整 tiling |
| 复杂 | 3–7 | attention / 大 matmul / conv，多级 tiling + autotune |
| 极复杂 | 7–15+ | Flash Attention / 自定义复合算子，深度优化 |

> 工作量为纯开发时间，不含环境搭建和跨团队沟通。建议保守估算（上取整），预留调优和 debug 时间。

### 步骤十一：风险登记

```markdown
## 风险登记表：<算子名称>

| # | 风险描述 | 严重程度 | 概率 | 缓解措施 | 负责人 |
|---|---------|:---:|:---:|---------|:---:|
| 1 | UB 溢出（大 shape 下 BLOCK_SIZE 难以平衡 grid 和 UB） | 🔴 高 | 中 | 准备 BLOCK_SIZE_SUB 二次切分方案 | T3 |
| 2 | fp16 累积误差超出精度阈值 | 🟡 中 | 低 | 关键累加使用 fp32，放宽阈值 | T5 |
| 3 | 性能不达目标 | 🟡 中 | 中 | max_autotune 兜底，必要时拆分为多 kernel | T3 |
```

---

## 输出模板

完成以上步骤后，按以下模板输出完整设计文档：

```markdown
# <算子名称> 开发方案

## 1. 需求概要
- 算子名称：...
- 算子类型：element-wise / reduction / matmul / attention / ...
- 输入输出：<IO spec>
- 计算逻辑：<formula>
- 目标芯片：A2 / A3 / A5
- 精度目标：fp16 / bf16 / fp32，atol=<>, rtol=<>
- 性能目标：≤ <latency>ms @ shape=<>

## 2. 可行性结论
- 总体评估：✅ 可行 / ⚠️ 可行但有限制 / ❌ 不可行
- 关键技术风险：[...]
- 待确认项：[...]

## 3. 硬件约束分析
- UB 有效大小：<size>KB
- MIN_BLOCK_SIZE（grid 约束）：<value>
- MAX_BLOCK_SIZE（UB 约束）：<value>
- 对齐状态：✅ / ⚠️ <问题>

## 4. Tiling 方案
| 参数 | 值 | 决策理由 |
|------|-----|---------|
| BLOCK_SIZE / (BM, BN, BK) | ... | ... |
| BLOCK_SIZE_SUB | ...（如需要） | ... |
| Grid | ... | ... |

## 5. 内存规划
| Buffer | 大小 | 可复用 | 备注 |
|--------|------|:---:|------|
| ... | ... | ... | ... |
| **UB 总占用** | **<total> / <max>** | — | **<ratio>%** |

## 6. 计算调度
- 循环顺序：...
- Core 类型：Vector / AI Core
- Grid 策略：固定核心数 / 动态 grid
- Doublebuffer：开启 / 关闭（理由）

## 7. Kernel 架构
- 架构：单 kernel / <N> 个 kernel
- Fusion 建议：<list>
- 接口签名：<launch 函数>

## 8. 测试策略
- Shape × Dtype 矩阵：<N> 个组合
- 边界场景：<N> 个
- 性能基线：<table>
- 验收门禁：<checklist>

## 9. 任务清单
| # | 任务 | 产出 | 工时 | 依赖 | Agent | 门禁 |
|---|------|------|:---:|------|-------|------|
| T1 | 需求分析与可行性评估 | ... | xx | - | analysis | ... |
| T2 | 算子代码生成 | ... | xx | T1 | gen-optimize | 编译+精度 |
| T3 | 性能优化 | ... | xx | T2 | gen-optimize | 精度+性能 |
| T4 | 代码检视 | ... | xx | T3 | review | 阻塞清零 |
| T5 | 精度验收 | ... | xx | T4 | test | 全覆盖 |
| T6 | 性能验收 | ... | xx | T4 | test | 性能基线 |

## 10. 时间线
T1(T+0) → T2(T+xx) → T3(T+xx) → T4(T+xx) → T5/T6(T+xx，可并行)

## 11. 风险登记
| # | 风险 | 等级 | 缓解措施 |
|---|------|:---:|---------|
| 1 | ... | ... | ... |

## 12. 交付清单
- [ ] 需求规格文档
- [ ] 可行性评估报告
- [ ] 详细设计文档（Tiling + 内存 + 调度 + 架构）（本文档第 3–7 节）
- [ ] 测试策略与验收门禁
- [ ] 任务规划表
- [ ] 风险登记表
- [ ] 接口规格（launch 函数签名）
```

---

## 注意事项

- **先算后画**：先完成数值计算和决策，再套用模板；跳过计算直接填模板是本末倒置
- **设计即合约**：详设文档是下游 Agent 的编码依据，参数缺失或决策模糊会导致代码偏离预期
- **芯片代际差异**：A5（UB 256KB）→ A2/A3（UB 192KB）迁移时需重新核算步骤三的 UB 预算
- **可变 shape 处理**：runtime shape 变化时，标注各维度的范围，Tiling 参数覆盖最大和最小 shape
- **必要时回退**：步骤二发现不可行时明确标记并说明原因，不强行设计
- **工作量保守估算**：上取整，预留调优和 debug 时间
- **风险高且概率高**：建议先做 POC 验证再全量开发
- **信息不足**：列出待确认项，不强行推测

## 参考信息

- [Triton-Ascend 开发者文档](https://triton-ascend.readthedocs.io/zh-cn/latest/index.html)
- [GPU→NPU 迁移指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/migration_guide/migrate_from_gpu.md)
- [AscendNPU-IR 最佳实践](https://gitcode.com/Ascend/AscendNPU-IR/blob/master/docs/source/zh_cn/user_guide/best_practice.md)
- [CANN 算子开发指南](https://www.hiascend.com/document)

### 相关 Skill

| Skill | 关系 |
| ------ | ------ |
| `triton-op-dev` | 下游 — 按本 Skill 的设计方案生成代码并优化 |
| `ops-test-gen` | 下游 — 按本 Skill 的测试策略构造精度和性能用例 |
