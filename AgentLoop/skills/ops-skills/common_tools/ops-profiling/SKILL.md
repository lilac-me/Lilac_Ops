---
name: ops-profiling
description: >-
  This skill should be used when the user asks to profile Triton operator
  performance on Ascend NPU, collect kernel_details.csv, analyze kernel pipeline
  utilization (scalar/vector/cube/mte pipeline breakdown), identify performance
  bottlenecks via kernel_details.csv. Typical triggers include "profile this kernel",
  "collect profiling data", "analyze kernel details", "why is this kernel slow",
  or "check pipeline utilization".
version: 1.0.0
domain: ops
function: common
tags:
  - profiling
  - kernel-details
  - performance-analysis
  - ascend
  - triton
author: mindspeed-agent
compatibility: MindSpeed >= 0.1.0, CANN >= 9.1.0
targets-repo:
  - triton-ascend
---

# 算子 Profiling 性能分析

使用 `torch_npu.profiler.profile(Level1)` 采集，`kernel_details.csv` 完成全部分析。

## 适用场景

- 算子性能不达标，需要定位瓶颈（compute bound vs memory bound）
- 需要采集 vector / scalar / mte 各 pipeline 利用率
- 优化后需要对比 profiling 数据验证效果
- 怀疑标量降级（i64/i32 比较无法启用 vector）

---

## 第一步：采集性能数据

### 1.1 采集脚本

```python
import torch
import torch_npu

with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    ),
    record_shapes=True,
    record_input_shapes=True,
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof"),
) as prof:
    for _ in range(4):  # warmup(1) + active(3), 对齐 schedule
        launch_kernel(x, y)
        prof.step()
```

落盘：`prof/*.pt.trace.json` + `ASCEND_PROFILER_OUTPUT/kernel_details.csv`

### 1.2 多 shape × 多 dtype 采集（必须）

**最少 3 shape × 2 dtype**，每个组合单独采集。

```python
SHAPES = [(65536,), (1024, 65536), (1073741824,)]       # small/medium/large
DTYPES = [torch.bfloat16, torch.float32]

for dtype in DTYPES:
    for shape in SHAPES:
        x = torch.randn(*shape, device="npu", dtype=dtype)
        tag = f"{'x'.join(map(str,shape))}_{str(dtype).split('.')[-1]}"
        with torch_npu.profiler.profile(
            activities=[...],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
            record_shapes=True, record_input_shapes=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"./prof/{tag}"),
        ) as prof:
            for _ in range(4):
                your_launch_fn(x, y)
                prof.step()
```

> **只关注 NPU Device 侧耗时**：所有延迟、带宽、占比指标取自 kernel_details.csv（NPU device 侧），不关注 CPU 侧。

---

## 第二步：分析 kernel_details.csv

路径 `ASCEND_PROFILER_OUTPUT/kernel_details.csv`，每行一个 NPU 算子。

### 2.1 关键列

**基础信息**：

| 列 | 用途 |
| ------ | ------ |
| `Name` | 算子名（Triton kernel / Cast / Free） |
| `Type` | 算子类型 |
| `Accelerator Core` | AI Core 或 Vector Core |
| `Duration(us)` | 设备侧总耗时，**排序找最慢算子** |
| `Block Num` | >1 = 计算 kernel，=1 = Cast |
| `Input Data Types` / `Input Shapes` | dtype 和 shape |

**Pipeline 耗时与占比**（`aic_` = AI Core, `aiv_` = Vector Core）：

| 耗时列 | 占比列 | 管线 | 判断 |
| ------ | ------ | ------ | ------ |
| `aiv_vec_time(us)` | `aiv_vec_ratio` | Vector 计算 | VV 算子主指标, < 0.1 → 空闲 |
| `aiv_scalar_time(us)` | `aiv_scalar_ratio` | 标量 | > 0.5 → 标量降级 |
| `aiv_mte2_time(us)` | `aiv_mte2_ratio` | HBM↔UB 搬运 | > 0.7 → Memory Bound |
| `aiv_mte3_time(us)` | `aiv_mte3_ratio` | UB↔L1 中转 | 中转瓶颈 |
| `aic_mac_time(us)` | `aic_mac_ratio` | Cube MAC | CV 算子核心 |
| `aic_mte2_time(us)` | `aic_mte2_ratio` | AI Core 搬运 | CV 算子 Memory Bound |
| `aic_fixpipe_time(us)` | `aic_fixpipe_ratio` | 固定功能管线 | 激活函数等 |
| `cube_utilization(%)` | — | Cube Core 利用率 | 仅 CV 算子 |

### 2.2 分析流程

1. 按 `Duration(us)` 降序，剔除 `Name` 含 `Free`/`Cast`/`Computing` 的行
2. VV 算子看 `aiv_*` 列，CV 算子看 `aic_*` 列
3. 直接看 ratio 定位瓶颈，看 Duration 找最慢算子

---

## 第三步：端到端优化决策引擎

输入 `kernel_details.csv` 的 ratio 列，按决策树输出优先级排序的优化建议。

### 3.1 决策树

```text
输入: aiv_scalar_ratio / aiv_mte2_ratio / aiv_vec_ratio / aic_mac_ratio
  │
  ├─ UB Overflow (编译期)?
  │   → P0: BLOCK_SIZE_SUB / buffer 复用 / 拆分 kernel
  │
  ├─ aiv_scalar_ratio > 0.5 && aiv_vec_ratio < 0.1?
  │   → P0: 标量降级 → int→fp32 向量化
  │
  ├─ aiv_mte2_ratio > 0.7?
  │   ├─ Memory Bound
  │   ├─ HBM BW < 50% Peak → P0: block_ptr 连续访存, 尾轴对齐
  │   ├─ HBM BW > 90% Peak → 硬件极限, P2: buffer 复用消除 Free
  │   └─ HBM BW 50-90% → P1: 增大 BLOCK_SIZE, num_stages 调优
  │
  ├─ aiv_vec_ratio > 0.5 || aic_mac_ratio > 0.5?
  │   ├─ Compute Bound
  │   ├─ VV 算子 → P1: autotune BLOCK_SIZE, P2: fp32→fp16
  │   └─ CV 算子 → P1: max_autotune, BK 调优
  │
  ├─ 小 shape (N < 100K) && Duration < 10us?
  │   → Launch Bound → P1: 固定核心 grid + 步长分发, P2: buffer 预分配
  │
  └─ 中大 shape && HBM BW > 90% Peak?
      → 已收敛, 无进一步优化空间
```

### 3.2 决策输出格式

profiling_report.md 末尾**必须**包含结构化优化建议表：

```markdown
## 优化建议

| 优先级 | 瓶颈 | 当前值 | 优化手法 | 预期收益 | 风险 |
|:---:|------|:---:|------|:---:|------|
| P0 | xxx | xx% | xxx | 1.5-2x | 低 |
| P1 | xxx | xx | xxx | 1.2x | 中 |
```

每条建议包含：**优先级**(P0/P1/P2/P3) | **瓶颈**(具体指标) | **当前值**(量化) | **优化手法**(代码/参数变更) | **预期收益**(量化) | **风险**

---

## profiling_report.md 产出要求

1. **输入规格表**：shape、dtype、芯片
2. **逐组合算子全景表**：只列 NPU 计算 kernel（剔除 Free/Cast/Computing），Duration + ratio
3. **HBM 带宽对比表**：所有组合的 BW 和利用率
4. **跨组合延迟矩阵**：shape × dtype 交叉对比
5. **优化建议表**：按决策树输出，带优先级

---

## 注意事项

- `ProfilerLevel.Level1` 自动生成 `kernel_details.csv`，无需 msprof 等其他工具
- 多 MTE pipeline 并行时共享带宽

## 参考

- [Triton-Ascend Profiling 指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/debug_guide/profiling.md)
- [AscendNPU-IR 最佳实践](https://gitcode.com/Ascend/AscendNPU-IR/blob/master/docs/source/zh_cn/user_guide/best_practice.md)
