---
name: ops-boundary-survey
description: >-
  This skill should be used when the user asks to survey, analyze, or test
  the boundary conditions of operators — from extracting constraints and
  classification frameworks, to generating boundary test cases, executing
  validation, and producing coverage reports. Typical triggers include
  "摸排 `<算子>` 边界", "分析 `<算子>` 边界", "边界测试 `<算子>`",
  or "生成边界测试用例". The test direction (forward-only / backward-only /
  both) is confirmed with the user at the start of the workflow.
version: 0.0.1
domain: ops
function: common
tags:
  - boundary-testing
  - operator-validation
  - test-generation
  - ascend
author: mindspeed-agent
compatibility: CANN >= 9.1.0
targets-repo:
  - triton-ascend
---

# 算子边界摸排

## 功能概述

分析算子源码与现有 UT，提取边界条件，生成并执行边界测试用例，产出**边界摸排报告**与**覆盖率报告**。**测试方向（仅正向 / 仅反向 / 双向）在流程开始时由用户确定**，双向采用「先正向后反向」两阶段策略（见[双向测试策略](#双向测试策略先正向后反向)），本 Skill 按确定的方向生成对应测试。

## 定位：算子开发支撑

```text
┌──────────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────┐
│  阶段一：算子开发               │ →  │  阶段二：边界摸排          │ →  │  阶段三：精度性能验收   │
│  triton-op-dev / AscendC      │    │  op-boundary-survey      │    │  op-test-gen          │
└──────────────────────────────┘    │  （本 Skill）             │    └──────────────────────┘
                                     └──────────────────────────┘
```

本 Skill 是 `ops-test-agent` 的核心能力，接收算子实现代码，产出边界测试报告与覆盖率分析。

## 适用场景

- 新算子开发完成后摸排边界条件
- 统计边界测试的代码覆盖率
- 验证约束外异常的正确拦截

## 前置条件

| 输入项 | 来源 | 必要性 |
| ------- | ------ | :---: |
| 算子实现代码（适配层 / AscendC / API 文档路径） | 用户 | 必须 |
| 目标平台（如 Atlas A2 训练系列产品） | 用户或环境 | 必须 |
| **测试方向（仅正向 / 仅反向 / 双向）** | 用户 | 必须（未指定则询问） |
| 现有 UT（如有） | 仓库 `tests/` | 推荐 |

---

## 算子分类框架

根据算子**输入参数复杂度**分类，边界维度不同：

| 层级 | 输入 | 数据类型 | 实现 | 示例 |
| ------ | ------ | ---------- | ------ | ------ |
| **Layer 1: 简单算子** | 1-2 个 Tensor + 少量简单参数（int/float/bool） | 仅支持 1-2 种 dtype | 单一 tiling key，适配层无复杂 shape 转换 | add_relu（x, y） |
| **Layer 2: 普通算子** | 多个 Tensor + 多个参数（可能含 bool/int） | 支持 1-2 种 dtype，可能含不同 dtype 混合组合 | 1-2 个 tiling key 分支，适配层少量 shape 转换 | roi_align_rotated（feature_map, rois, spatial_scale, sampling_ratio, aligned, clockwise...） |
| **Layer 3: 复杂算子** | 多个 Tensor + 复杂参数 | 可能含多种 dtype（float32/half/int32/int16），不同 dtype 可能混合组合 | 多 tiling key 分支，适配层可能含大量转换和前后置处理 | deformable_conv2d（x, offset, weight, stride, padding, dilation, groups, deformable_groups） |

---

## 工作流程

> **时限**：整体摸排流程需在 **60 分钟**内完成。

```text
Step 0: 明确测试方向（正向 / 反向 / 双向）
Step 1: 分析现有 UT
Step 2: 提取约束
Step 3: 生成边界摸排报告
Step 4: 生成 UT（按方向）
Step 5: 执行测试与错误分析
Step 6: 覆盖率统计
```

---

### Step 0: 明确测试方向

> **任务启动后需确认测试方向，用户未指定时主动询问，并记录到报告「测试方向」字段。**

| 方向 | 含义 | 用例形态 |
| ------ | ------ | ---------- |
| 仅正向（forward-only） | 只验证前向 output 精度 | `test_*_forward` |
| 仅反向（backward-only） | 只验证反向梯度精度 | `test_*_backward` |
| 双向（both） | 两阶段：先仅正向，通过后再仅反向 | 阶段1 `test_<op>_boundary.py` + 阶段2 `test_<op>_backward_boundary.py` |

**判定规则**：

- 明确「只测正向」→ 仅正向；明确「只测反向」→ 仅反向；明确「都测」→ 双向（见[双向测试策略](#双向测试策略先正向后反向)）；无法判断 → 主动询问。
- 反向约束：需先跑一次前向构建 autograd 图，但**只断言梯度，不断言 output**；`grad_output` 由 `gen_input_data` 一并生成（也可独立生成）；算子无显式 backward 时反向不可用，需向用户说明。

---

## 双向测试策略（先正向后反向）

当测试方向为「双向」时，**按两阶段顺序执行，先摸排正向，正向通过后再摸排反向**，不做单文件混排的 forward+backward 用例：

| 阶段 | 方向 | 执行步骤 | 产物 |
| ------ | ------ | ---------- | ------ |
| 阶段 1 | 仅正向 | Step 0（仅正向）→ Step 1-6 | `boundary_report_<op>.md` + `test_<op>_boundary.py` + 覆盖率报告 |
| 阶段 2 | 仅反向 | Step 0（仅反向）→ Step 1-6 | `boundary_report_<op>_backward.md` + `test_<op>_backward_boundary.py` + 覆盖率报告 |

**原则**：

1. **顺序强约束**：阶段 1 全量通过前，禁止进入阶段 2（反向以正向成立为前提；正向可先行发现版本分发、dtype、shape 等公共边界，供反向复用）。
2. **约束复用**：阶段 2 复用阶段 1 的公共约束（dtype / shape / attr / Tiling 分支 / 实测修正），仅补提取反向专属约束（grad_output 形状、反向容差、grad 内核硬编码）。
3. **质量标准独立**：每阶段独立满足「约束内 ≥24、异常拦截 ≥8、总 ≥32」；仅统计本阶段方向的用例。
4. **异常按阶段覆盖**：正向异常用例验证 forward 拦截；反向异常用例经「前向+反向」全链路验证（约束类异常在前向阶段即被拦截）。
5. **产物独立命名**：反向产物加 `_backward` 后缀，避免覆盖；报告「测试方向」字段记录当前阶段。
6. **覆盖率分别统计**：两阶段独立统计 python 覆盖率；反向阶段经前向构建 autograd 图，可补齐正向未覆盖的 backward 分支。

---

### Step 1: 分析现有 UT（必须）

```bash
# 在开始边界分析前，先读取现有 UT
tests/torch/test_<op>.py
```

从中提取：

1. **调用方式**：`mx_driving.<op_call_name>(...)` 的参数顺序
2. **Golden 来源**：
   - PyTorch 函数
   - 自定义 CPU 实现（如现有 UT 中的 cpu_xxx 函数），通常为 numpy 函数
   - mmcv.ops 实现，若是此类 Golden 需引导用户编译安装 mmcv：

   ```bash
   git clone https://github.com/open-mmlab/mmcv.git -b 1.x
   cd mmcv
   MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py build_ext
   MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
   ```

3. **测试场景**：已有的 shape/dtype 覆盖
4. **精度阈值**：atol/rtol 经验值（正、反向通常不同，分别记录）
5. **反向支撑**：现有 UT 是否有 forward / backward（或只测了其中一种）——决定本算子反向测试的 Golden 可行性与参考容差

---

### Step 2: 提取约束

阅读边界摸排报告模板（见 Step 3），从适配层代码（2.1）、AscendC 源码（2.2）和 API 文档（2.3）中**提取所有约束条件**，整理边界维度（2.4），确认边界用例（2.5），并生成边界摸排报告。

> **此步骤需先确认输入、输出的 Shape，再设计边界用例。**

#### 2.1 阅读 Python 适配层和 C++ 适配层

```bash
mx_driving/ops/<op>.py
mx_driving/csrc/<OpName>.cpp
```

提取：

- 调用条件（如 numel 阈值）
- 调用的 ACLNN API
- **显式 backward 函数**（如 `class XxxFunction(Function)`）——存在则反向可测，不存在则反向不可用
- **shape 约束**（如 `c_in % 64 == 0`）
- **attr 约束**（如 deformable_groups == 1）
- format 转换（如 NCHW → NHWC）

#### 2.2 阅读 AscendC 实现

```bash
kernels/<op>/op_host/<op>.cpp         # Atlas A2 训练系列产品 / Atlas A3 训练系列产品 Host 侧
kernels/<op>/op_host/<op>_tiling.h    # Atlas A2 训练系列产品 / Atlas A3 训练系列产品 Tiling 参数
kernels/<op>/op_kernel/<op>.cpp       # Atlas A2 训练系列产品 / Atlas A3 训练系列产品 Kernel 侧

kernels/<op>/op_host/arch35/<op>.cpp  # Ascend 950 产品 Host 侧
kernels/<op>/op_host/arch35/<op>_tiling.h    # Ascend 950 产品 Tiling 参数
kernels/<op>/op_kernel/arch35/<op>.cpp       # Ascend 950 产品 Kernel 侧
```

**必须提取**：

- 支持的 **dtype**（DT_FLOAT, DT_FLOAT16, DT_BF16 等）
- **Tiling 分支**（每个分支均需设计对应测试用例）
- **硬件平台支持**（Atlas A2 训练系列产品 / Atlas A3 训练系列产品 / Ascend 950 产品）
- 版本/分支选择逻辑（如 V1/V2 依据 channel 与 groups 动态选择），**防止把未摸排版本当成摸排对象**

#### 2.3 阅读 API 文档

```bash
docs/zh/api/context/<op>.md
```

补充：

- 参数说明
- dtype 支持
- **shape 约束**（如 h_out 公式）
- **attr 约束**（如 groups 整除要求）
- 返回值格式

#### 2.4 生成边界摸排报告

根据 Step 0（测试方向）、Step 1（现有 UT）和 Step 2（约束）提取的信息，用 Step 3 的模板生成报告。约束内 UT ≥24 组，异常拦截 UT ≥8 组。

> **实测校准**：源码/文档声明的支持范围可能与实测不一致（如内核硬编码只支持标准参数），**以实测为准**并在报告中记录「实测修正」。

---

### Step 3: 生成边界摸排报告

#### 边界摸排报告模板

```markdown
# <算子名称> 边界摸排报告

## 基本信息

| 项目 | 内容 |
|------|------|
| 算子名称 | <op_name> |
| 算子分类 | 简单算子 / 普通算子 / 复杂算子 |
| 支持平台 | Atlas A2 训练系列产品 / Atlas A3 训练系列产品 / Ascend 950 产品 |
| **测试方向** | 仅正向 / 仅反向 / 双向 |

## 算子分类说明

<简单/普通/复杂>算子，算子功能：<简单描述>，算子的 IR 信息及约束边界如下：

### 算子IR
| IR类型 | 变量名称 | 变量类型 | Shape | Dtype |
| 输入 | input1 | Tensor | [B, C, N, W] | float32 / float16 |
| 输入 | input2 | Tensor | [B, N, 3] | float32 |
| 输出 | output1 | Tensor | [B, W, C] | float32 |
| 属性 | attr1 | Value | - | int32 |

### Dtype 约束

| dtype | 约束条件 | 约束内测试 | 约束外测试 |
|-------|----------|------------|------------|
| <dtype1> | 支持 | <合法值> | <非法值> |
| <dtype2> | 不支持 | - | <非法值> |

### Shape 约束

| 维度 | 约束条件 | 约束内测试 | 约束外测试 |
|------|----------|------------|------------|
| <dim1> | <约束> | <合法值> | <非法值> |
| <dim2> | <约束> | <合法值> | <非法值> |

### Attr 约束

| 参数 | 约束条件 | 约束内测试 | 约束外测试 |
|------|----------|------------|------------|
| <attr1> | <约束> | <合法值> | <非法值> |

### Tiling 分支

| 分支条件 | 触发条件 | 约束内测试 |
|----------|----------|------------|
| <branch1> | <条件> | <参数组合> |

### 取值范围

| 值 | 约束条件 | 约束内测试 | 约束外测试 |
|------|----------|------------|------------|
| <bool_attr> | True/False | 两种都测 | - |
| <numeric_attr> | <约束> | <正常值> | <边界值> |

## 边界用例表

> 每个用例需有**唯一「用例 ID」**（对应 UT 方法名）与**「维度」标注**（dtype / shape / attr / 版本分支 / Tiling 分支 / 数值边界 / 异常拦截），供 UT 用例覆盖率脚本自动匹配。

### 约束内测试（按 Step 0 确定的测试方向）

| # | 用例 ID | 维度 | 输入 1 Shape | 输入 2 Shape | attr1 | attr2 | dtype | TilingKey | 方向(正向/反向) | 用例说明 |
|---|---------|------|-------------|-------------|-------|-------|-------|---------|----------------|----------|
| 1 | <case_id> | <维度> | <shape1> | <shape2> | <val1> | <val2> | <dtype> | <tiling> | <fwd/bwd> | <desc> |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 约束外测试 (异常拦截)

| # | 用例 ID | 维度 | 输入 1 Shape | 输入 2 Shape | attr1 | attr2 | dtype | TilingKey | 方向(正向/反向) | 用例说明 | 预期异常 |
|---|---------|------|-------------|-------------|-------|-------|-------|---------|----------------|----------|----------|
| 1 | <case_id> | <维度> | <shape1> | <shape2> | <val1> | <val2> | <dtype> | <tiling> | <fwd/bwd> | <desc> | <error_msg> |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
## 质量统计

| 指标 | 目标 | 实际 |
|------|------|------|
| 约束内用例（正向） | ≥24 组 | <actual> |
| 约束内用例（反向） | ≥24 组 | <actual> |
| 异常拦截用例 | ≥8 组 | <actual> |
| 总用例数 | ≥32 组 | <actual> |
```

> 质量统计按实际测试方向填写：仅正向时只统计正向列，仅反向时只统计反向列，双向时两列都统计。

---

### Step 4: 生成 UT（分步创建）

基于 Step 3 的报告，参考模板 `scripts/test_boundary_template.py`，**按以下顺序**分步创建 UT（每步用 write 工具一次性写入完整内容，不用 bash echo 等命令）：

1. **创建框架**：import + class + helper 函数（pass 占位）
2. **创建输入构造**：`gen_input_data` 返回所有输入，每个测试用例复用
3. **创建标杆函数**：`golden_forward` / `golden_backward`（来源：NumPy、PyTorch 或 mmcv.ops）
4. **填入约束内测试**：按 Step 0 确定的测试方向生成 `test_*_forward` 和/或 `test_*_backward`
5. **填入约束外测试**：异常拦截用例，同样按方向生成

**方向生成规则（必须遵守）**：

| 测试方向 | 生成的测试方法 |
| ---------- | ---------------- |
| 仅正向 | 只生成 `test_*_forward`（不生成任何 backward 方法） |
| 仅反向 | 只生成 `test_*_backward` |
| 双向 | 分两阶段分别生成：阶段 1 仅 forward，阶段 2 仅 backward（见[双向测试策略](#双向测试策略先正向后反向)） |

**注意事项**：

- 输出必须是**纯 Python 代码**，不使用 Jinja2 语法
- 使用 `@golden_data_cache` 装饰器缓存输入
- **通用 gen_input_data**：一次性返回所有输入（x, offset, weight, ...）
- **forward/backward 错误拦截分别测**：约束外值可能在 forward 或 backward 阶段被拦截，异常用例按测试方向覆盖对应阶段
- **方法命名必须与报告用例 ID 对应**：`test_<case_id>_forward` / `test_<case_id>_backward` / `test_<case_id>_error`，供 Step 6 覆盖率脚本按 ID 自动匹配
- **写完即结束**：UT 写入完成后全面检查代码，确认无误可运行后，执行 Step 5 与用户协同测试

---

#### UT 测试模板说明

模板文件：`scripts/test_boundary_template.py`（含 `gen_input_data` / `golden_*` / `npu_*` / `test_<case_id>_*` 骨架，按需替换 `<OpName>`、`<op_name>`、`<case_id>` 等占位符）。

**要点**：

1. **方向由用户决定，正反向分开**：
   - 仅正向：只保留 `test_*_forward`，不生成 `test_*_backward`
   - 仅反向：只保留 `test_*_backward`
   - 双向：分两阶段分别生成（见[双向测试策略](#双向测试策略先正向后反向)）
   - forward 用例只对比 output；backward 用例只对比各输入梯度
2. **`golden_forward` / `npu_forward`**：前向 golden（CPU 或 mmcv 等）与 NPU 实现，返回 output
3. **`golden_backward` / `npu_backward`**：需先跑一次前向构建 autograd 图再 backward，返回各输入梯度
4. **异常 case 检验**（`test_*_error`）：
   - `with self.assertRaises(Exception) as ctx:` 捕获异常
   - `self.assertEqual(str(ctx.exception), "<预期错误消息>\n")` 验证错误消息
5. **`@golden_data_cache`**：缓存输入数据，避免重复生成

---

### Step 5: 执行测试与错误分析

#### 5.1 由用户协同测试

> ⚠️ **重要**：必须在 `DrivingSDK` 的**上一层**目录运行，否则会导致循环死锁。

**用户执行（必须重定向日志）**：

```bash
cd DrivingSDK的上级目录
python DrivingSDK/tests/torch/test_<op>_boundary.py > DrivingSDK/tests/torch/test_<op>_boundary.log 2>&1
cat DrivingSDK/tests/torch/test_<op>_boundary.log
```

**必须查看日志**：重定向日志才能看到 pass/fail/skip 统计。

#### 5.2 分析日志

查找统计信息：

- `Ran N tests in X.XXXs` - 获取测试数量
- `FAILED (failures=X, errors=Y)` - 确认失败数量

识别失败模式：

| 错误类型 | 原因 | 修复 |
| ---------- | ------ | ------ |
| `result error` | 精度超限 | **不允许**调整 atol/rtol 或删除反向测试用例，需分析用例设计是否存在错误 |
| `shape error` | shape 不匹配 | 检查用例配置 |
| `AssertionError` | 异常拦截失败 | 检查 expected_msg |

**如失败**：修改测试脚本 → 删除缓存 `data_cache/test_<op>_boundary/` → 重新执行 5.1，直到通过。**整体摸排流程需在 60 分钟内完成。**

> **Golden 反向可信度**：若反向结果与 golden 系统性不一致，需判别是算子反向缺陷还是 golden 不可靠。**避免使用 `torch.autograd.gradcheck`（NPU 上极慢）**，改用更高效方法：在 CPU 侧用更小 shape 的 float64 golden 对比验证反向正确性（与现有 grad UT 相同方法），再决定容差或更换 golden。

---

### Step 6: 覆盖率统计（UT 用例覆盖率）

**覆盖目标**：边界场景覆盖度，即「已实现 UT 用例数 ÷ 边界报告中全部边界场景数」。

**由 Agent 自动生成覆盖率报告**：运行本 Skill 自带脚本 `scripts/boundary_coverage.py`。脚本解析边界报告「边界用例表」（按用例 ID + 维度）与 UT 文件（按 `test_<case_id>_*` 方法名）自动匹配，产出 `coverage_report_<op>.md`（覆盖矩阵 + 总覆盖率 + 未覆盖场景清单）：

```bash
python scripts/boundary_coverage.py --report boundary_report_<op>.md --ut test_<op>_boundary.py \
    --output coverage_report_<op>.md
```

**与用户协同**：覆盖率报告生成后，与用户核对未覆盖场景是否合理（如约束外不构造的场景需在报告中说明），确认后归档。

> 报告与 UT 的匹配依赖用例 ID 约定（见 Step 3 边界用例表与 Step 4 方法命名）。

#### 覆盖率报告模板

`coverage_report_<op>.md` 结构如下：

```markdown
# <算子名称> 边界 UT 用例覆盖率报告

## 覆盖率统计

| 指标 | 数值 |
|------|------|
| 边界场景总数（约束内 + 约束外） | <total> |
| 已覆盖场景数（存在对应 UT 方法） | <covered> |
| **UT 用例覆盖率** | **<xx.x>%** |

## 覆盖矩阵

### <维度 1>（如 dtype）

| # | 用例 ID | 所属表格 | UT 方法 | 覆盖状态 |
|---|---------|----------|---------|:---:|
| 1 | <case_id> | 约束内测试 | `test_<case_id>_forward` | OK |

### <维度 2>（如 shape）

...

## 未覆盖场景

| # | 用例 ID | 维度 | 所属表格 |
|---|---------|------|----------|
| 1 | <case_id> | <dim> | 约束外测试 |

## 结论

OK UT 用例覆盖率 <xx.x>% >= 95%，达标。 / FAIL 未达标，请补充以下场景...
```

## 质量标准

| 指标 | 目标 |
| ------ | ------ |
| 约束内用例（正向，仅当测试方向含正向） | ≥24 组 |
| 约束内用例（反向，仅当测试方向含反向） | ≥24 组 |
| 异常拦截用例 | ≥8 组 |
| 总用例数 | ≥32 组 |
| 正向 pass 率 | 100% |
| 反向 pass 率 | 100% |
| 异常测试通过率 | 100% |
| UT 用例覆盖率 | ≥95% |

---

## 输出产物

| 阶段 | 产物 | 命名 |
| ------ | ------ | ------ |
| 边界分析 | boundary_report.md | `boundary_report_<op>.md`（反向阶段为 `boundary_report_<op>_backward.md`） |
| 测试框架 | boundary.py | `test_<op>_boundary.py`（反向阶段为 `test_<op>_backward_boundary.py`） |
| 执行日志 | .log | `test_<op>_boundary.log`（反向阶段为 `test_<op>_backward_boundary.log`） |
| 覆盖率报告 | coverage_report.md | `coverage_report_<op>.md`（反向阶段为 `coverage_report_<op>_backward.md`），由 `scripts/boundary_coverage.py` 自动生成 |

> 双向策略下，阶段 1（仅正向）与阶段 2（仅反向）各自独立产出上述 4 类产物，反向产物加 `_backward` 后缀。

## 参考信息

- [PyTorch testing 文档](https://pytorch.org/docs/stable/testing.html)
- [PyTorch autograd 文档](https://pytorch.org/docs/stable/autograd.html)
- [pytest 文档](https://docs.pytest.org/)
- [AscendC 算子开发](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/guide/)

### 相关 Skill

| Skill | 关系 |
| ------ | ------ |
| `triton-op-dev` | 上游 — 提供待测试的算子代码 |
| `op-test-gen` | 下游 — 精度和性能测试用例构造 |
