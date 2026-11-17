---
name: ascend-msprof-analyze-cli
description: >-
  This skill should be used when the user asks to "分析 profiling 数据",
  "分析性能数据", "msprof-analyze", "集群分析", "专家建议", "慢卡分析",
  "advisor 分析", "看下 profiling 结果", or wants to run MindStudio Profiler
  Analyze for performance bottleneck diagnosis on collected Ascend profiling data.
  Supports both cluster analysis (multi-rank summary, cross-rank comparison,
  communication bottleneck) and advisor mode (automatic bottleneck identification
  with optimization suggestions).
version: 1.0.0
domain: inference-model
function: performance-optimization
tags:
  - vllm-ascend
  - profiling
  - analysis
  - performance
author: Liccol
hooks:
  PreToolUse:
    - matcher: "Bash|PowerShell"
      prompt: |
        STOP. Before executing any command, you MUST:
        1. Present the analysis configuration in a table to the user (including: analysis
           capability choice — Cluster vs Advisor — with the reasoning based on data format,
           card count, and user scenario; data path; output path; analysis mode/subcommand).
        2. Wait for the user to explicitly confirm the configuration.
        If you have not done both of these, DO NOT execute — present the config table first.

    - tools: ["Bash"]
      matcher: "ssh.*\\s/(home|workspace|tmp|usr|vllm)"
      prompt: |
        ⚠️ Git Bash 会转换 Linux 路径。请用 **PowerShell** 执行 SSH 命令。

    - tools: ["Bash", "PowerShell"]
      matcher: "(compilation_config|additional_config|profiler.config)"
      prompt: |
        ⚠️ JSON 双引号必须转义：PowerShell 用 `\"`，Git Bash 用 `\\"`。
targets-repo:
  - cann
  - msprobe
  - mstt
---

# ascend-msprof-analyze-cli

## 功能概述

使用 `msprof-analyze` 对已采集的 Ascend profiling 数据进行分析，提供计算、通信、调度多维度的瓶颈诊断和优化建议。

本 skill 提供两种分析能力，根据用户需求自动选择对应工具：

| 能力 | 工具命令 | 侧重点 |
| ------ | ---------- | -------- |
| 集群分析 (Cluster) | `msprof-analyze -m <mode>` | 集群多维信息汇总、拆解对比、通信瓶颈定位和下发问题分析 |
| 专家建议 (Advisor) | `msprof-analyze advisor <subcommand>` | 基于性能数据自动识别计算、调度、通信等潜在问题并输出优化建议 |

## 使用方式

- 直接调用：`/mindspeed-agent:ascend-msprof-analyze-cli`
- 描述分析需求（如"分析 profiling 数据""看下慢卡""帮我诊断瓶颈"），由 Claude 自动匹配调用

> Windows 下用 **PowerShell** + `py -3`，不要用 Git Bash（MSYS 会转换远程路径，破坏 SSH 命令）。

## 操作步骤

### 步骤一：选择分析能力

#### 选择规则

| 条件 | 优先使用的分析能力 | 原因 |
| ------ | ------------------- | ------ |
| 卡数 ≥ 64 或数据量很大（db 大于 1G） | 集群分析 | 专门优化了大数据量场景，能快速汇总 |
| 数据格式为 **db** | 集群分析 | 集群分析仅支持 db 格式 |
| 数据格式为 **text**（json/csv） | 专家建议 | 专家建议支持 text 和 db 格式 |
| 需要分析慢卡/慢链路原因 | 集群分析 → 专家建议 | 先定位慢卡，再用专家建议深入诊断 |
| 需要算子优化建议 | 专家建议 | 专家建议输出详细的调优建议 |
| 单卡数据以 `_ascend_pt` 或 `_ascend_ms` 结尾 | 专家建议 | 直接使用专家建议诊断 |

#### 典型决策流程

```text
用户提出分析需求
        │
        ▼
┌─────────────────────────────┐
│ 数据量评估：卡数、数据规模    │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ 数据格式判断：db 还是 text？  │
└─────────────────────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
   【db 格式且为多卡数据】          【text 格式或单卡数据】
        │                              │
        ▼                              ▼
  优先集群分析                    只能专家建议
        │                        （跳过步骤二的集群分析）
        ▼
┌─────────────────────────────┐
│ 集群分析是否找到慢卡？        │
└─────────────────────────────┘
        │
        ├─────────────┬─────────────┐
        ▼             ▼             ▼
     是慢卡        不是慢卡        无瓶颈
        │             │             │
        ▼             │             ▼
  对该慢卡用          分析其他问题     结论输出
  专家建议深入
```

#### 场景与能力匹配表

| 用户需求场景 | 推荐能力 | 理由 |
| ------------- | ---------- | ------ |
| "分析这个 16 卡集群的慢卡问题" | 集群分析 → 专家建议 | 先定位慢卡，再深入诊断 |
| "这个单卡的数据有算子执行慢" | 专家建议 | 单卡场景专家建议更全面 |
| "帮我看看这个 profiling 数据" | 先判断数据格式 | 格式决定能力选择 |
| "通信带宽低是什么原因" | 集群分析 | 通信矩阵分析是集群分析强项 |
| "这个算子为什么慢" | 专家建议 | 算子问题诊断是专家建议强项 |
| "数据格式是 csv 的" | 专家建议 | 集群分析仅支持 db |

### 步骤二：执行分析

#### 能力一：集群分析 (Cluster)

**分析模式选择（`-m` 参数）**：

| `-m` 参数 | 说明 |
| ----------- | ------ |
| `cluster_time_summary` | 集群训练迭代耗时拆解，定位性能瓶颈 |
| `compute_op_sum` | Device 侧计算类算子汇总 |
| `freq_analysis` | 识别 AICore 空闲（800MHz）或异常频率 |
| `ep_load_balance` | MoE 负载信息汇总分析 |
| `communication_time_sum` | 集群通信时间和带宽汇总 |
| `hccl_sum` | 通信类算子信息汇总 |
| `slow_rank` | 快慢卡统计，识别慢卡原因 |
| `cann_api_sum` | CANN 层 API 汇总 |
| `free_analysis` | Device 侧大块空闲时间自动分析 |
| `all`（默认） | 同时解析通信矩阵和通信耗时数据 |

**集群分析命令模板**：

```bash
msprof-analyze -m <mode> -d <cluster_data_path> [-o <output_path>] [--force] --agent
```

命令示例：

```bash
# 默认完整分析
msprof-analyze -d ./cluster_data -o ./output --agent

# 指定分析模式
msprof-analyze -m cluster_time_summary -d ./cluster_data -o ./output --agent
msprof-analyze -m free_analysis -d ./cluster_data -o ./output --agent
```

**执行要点**：

1. 根据用户需求确定 `-m` 参数值。若用户要求"完整分析"或"全部"或没有明确要求，默认使用 `all`。
2. 默认不需要 `-o` 参数，除非用户明确指定输出路径。
3. 如果命令返回 error 或 JSON 中 message 为空，不要继续。提示让用户自己执行命令（去掉 `--agent` 参数），查看日志定位原因。
4. 后续分析可结合返回 JSON 中的 `suggestion` 字段。

**数据路径要求**：

- 单卡：指定到单个 `*_ascend_pt` 目录
- 多卡/集群：指定到所有 `*_ascend_pt` 的父目录

#### 能力二：专家建议 (Advisor)

**分析子命令**：

| 子命令 | 说明 | 包含功能 |
| -------- | ------ | ---------- |
| `all`（默认） | 总体性能瓶颈 | 计算 + 通信 + 调度 |
| `computation` | 计算瓶颈 | computation + Kernel compare |
| `schedule` | 调度瓶颈 | schedule + API compare |

**功能详情**：

| 维度 | 功能 | 说明 |
| ------ | ------ | ------ |
| overall | Overall Summary | 计算、通信、空闲等维度拆解 |
| overall | Environment Variable Issues | 环境变量设置推荐 |
| overall | slow rank | 慢卡识别 |
| overall | slow link | 慢链路识别 |
| computation | AICPU Issues | AI CPU 调优 |
| computation | Operator Dynamic Shape Issues | 识别动态 Shape 算子 |
| computation | AI Core Performance Analysis | MatMul、FlashAttentionScore 等算子分析 |
| computation | Block Dim Issues | Block Dim 算子调优 |
| computation | Operator No Bound Issues | 算子瓶颈分析 |
| computation | Fusion Issues | 融合算子图调优 |
| computation | AI Core Frequency Issues | AI Core 算子降频分析 |
| communication | Packet Analysis | 通信小包检测 |
| communication | Bandwidth Contention Analysis | 通信计算带宽抢占检测 |
| communication | Communication Retransmission Analysis | 通信重传检测 |
| communication | Byte Alignment Analysis | 通信算子字节对齐检测 |
| schedule | Affinity API Issues | 亲和 API 替换调优 |
| schedule | Operator Dispatch Issues | 识别算子下发问题 |
| schedule | SyncBatchNorm Issues | BatchNorm 同步检测 |
| schedule | Synchronize Stream Issues | 流同步检测 |
| schedule | GC Analysis | 垃圾回收事件检测 |
| schedule | Fusible Operator Analysis | 检测 Host/MTE 瓶颈算子序列 |
| dataloader | Slow Dataloader Issues | 异常 dataloader 检测 |
| memory | Memory Operator Issues | 识别异常的内存申请释放操作 |
| comparison | Kernel compare | 标杆性能数据 Kernel 对比 |
| comparison | API compare | 标杆性能数据 API 对比 |

**Advisor 命令模板**：

```bash
msprof-analyze advisor <subcommand> -d <profiling_path> [-o <output_path>] [-bp <benchmark_path>] [--force] --agent
```

命令示例：

```bash
# 单卡
msprof-analyze advisor all -d ./profiling_data -o ./output --force --agent

# 集群（多卡 `_ascend_pt` 的父目录）
msprof-analyze advisor all -d ./cluster_profiling -o ./output --force --agent
```

**数据路径要求**：

| 场景 | 路径指向 |
| ------ | ---------- |
| 单卡 | 指定到 `*_ascend_pt` 或 `*_ascend_ms` 目录 |
| 多卡/集群 | 指定到所有 `*_ascend_pt` / `*_ascend_ms` 的父目录 |

### 步骤三：解读结果

#### 集群分析专用诊断规则

> ⚠️ 禁止仅凭单项指标字面意思下结论，必须严格遵守以下诊断逻辑。

- **【Host 下发瓶颈（伪快卡）】**：某卡 `Free Time` 极长（占比 > 10% 或远超均值），且 `Compute` 和 `Communication` 时间异常偏短 → 该卡是导致集群阻塞的"慢卡"。CPU 下发慢导致 NPU 饿死（巨大 Free Time），当它终于发起通信时其他卡已等待多时，故其通信瞬间完成。验证：使用 `free_analysis` 分析空闲原因，`freq_analysis` 检查频率。
- **【纯计算快慢卡】**：各卡 `Free Time` 普遍较短且均匀，但某卡 `Compute Time` 显著大于均值 → 计算型慢卡。
- **【慢链路定位】**：某条链路 `Bandwidth(GB/s)` 显著低于同类型链路均值。链路类型速度：LOCAL（片内拷贝）> HCCS/PCIE（节点内片间）> RDMA（节点间）。验证：使用 communication_matrix 查看各链路带宽。
- **【频率异常】**：NPU 频率为 800MHz（空闲状态）或非 1800MHz/800MHz → 可能表示算子等待或调度问题。验证：使用 `freq_analysis`。
- **【通信瓶颈】**：通信耗时占比高，快慢卡明显。使用 `communication_bottleneck` 分析是 Host 侧下发慢还是 Device 侧计算慢导致通信等待。

#### 输出报告结构

分析完成后按以下结构输出：

1. **分析概要**：使用的分析模式、执行的完整命令、输出结果路径
2. **详细数据**：表格形式展示分析结果
3. **瓶颈定位**：瓶颈类型、关键证据（引用具体数值和占比）
4. **优化建议**：针对性建议

> 报告中所有时间值从 μs 换算为 ms 后再呈现，并明确标注单位。

### 步骤四：生成本地报告

分析完成后，必须将分析结果整理为报告并保存在本地 `.workspace-local/ascend-profiling-collection/runs/` 对应采集 run 目录下（与采集 report.md 同级）。

**报告文件命名**：`analysis_report.md`

**报告内容**：

1. **分析概要**：分析时间、使用的分析模式、执行的完整命令、数据路径
2. **关键发现**：瓶颈类型、关键证据（引用具体数值和占比）
3. **详细数据**：表格形式展示分析结果
4. **优化建议**：按优先级排列的针对性建议
5. **下一步**：是否建议进一步采集或验证

**报告示例结构**：

```markdown
# Profiling 分析报告

> **分析时间**: 2026-07-14 09:00:00 UTC
> **分析模式**: 集群分析 (all) + 专家建议 (computation)
> **数据路径**: /home/workspace/my-tag
> **命令**:
>   msprof-analyze -d /home/workspace/my-tag -o ./output --agent
>   msprof-analyze advisor computation -d /home/workspace/my-tag --agent

## 分析概要

| 维度 | 结果 |
|------|------|
| 卡数 | 4 |
| 数据格式 | db |
| 瓶颈类型 | 计算瓶颈 |

## 关键发现

### 1. MatMul 算子耗时占比过高
- **证据**: MatMul 类算子占总计算时间 68.3%
- **影响**: 单步迭代延迟 245.6 ms
- **建议**: 检查是否有动态 Shape 触发重编译

### 2. 通信带宽偏低
- **证据**: RDMA 链路平均带宽 1.2 GB/s（预期 > 3 GB/s）

## 优化建议

| 优先级 | 建议 | 预期收益 |
|--------|------|----------|
| P0 | 消除动态 Shape 算子 | 延迟降低 15-20% |
| P1 | 检查通信链路配置 | 带宽提升 2x |
```

## 注意事项

- `msprof-analyze` 必须在远程 Ascend 容器内运行，不在本地执行
- 集群场景下严禁仅分析单个 Rank 数据而不进行多卡对比，最好只基于最后输出的文件进行分析
- 命令行失败后不能继续读取各个 rank 的数据
- 如果无法确定使用哪个分析能力，请让用户明确选择，不能自己尝试
- 如 `--agent` 参数不被当前版本支持，去掉后重试
- 安装（如容器中未预装）：

  ```bash
  pip install msprof-analyze -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
  ```

### 集群分析常见故障

| 故障 | 现象 | 规避 |
| ------ | ------ | ------ |
| 时间单位混淆 | 输出的时间字段单位不明确 | 统一使用毫秒（ms），在报告中明确说明 |
| profiler_level 过低 | 无法获取通信带宽和通信矩阵信息 | profiler_level 设置为 Level1 或更高 |
| db 类型数据缺失 | Recipe 分析报错缺少 db 文件 | 确认使用 db 类型数据；Ascend PyTorch Profiler 需指定 `export_type=["db"]` |
| Rank/Step ID 无效 | 指定 `--rank_list` 或 `--step_id` 后无输出或报错 | 确认配置的 ID 在实际数据中存在 |

## 参考信息

- [recipe_output_format_introduct.md](references/recipe_output_format_introduct.md) — Cluster Analysis 输出格式说明（cluster_step_trace_time.csv、cluster_communication_matrix.json 等表头字段含义和数据解读方法）
- 依赖 skill：`ascend-profiling-collection`（提供 profiling 数据）
