# CSV 列口径与判定规则（S2 前端）

本文是 `parse_profile.py` 处理口径的**权威说明**，与设计文稿 §6 一致。

## 一、真实 CSV 列（以实测 Ascend profiler 产物为准）

`kernel_details.csv` 相关列（按 header 名取列，不写死列号）：

| 列名 | 用途 |
| --- | --- |
| `Step Id` | 所属 step；值 `N/A`/非数字的行无法归 step，跳过 |
| `Name` | kernel 名（op 类型），聚合维度 |
| `Duration(us)` | 单次执行耗时；数字转换失败或 ≤0 的行跳过 |
| `Accelerator Core` | AI_CORE / AI_VECTOR_CORE / AI_CPU / AIC / COMMUNICATION 等 |
| `Input Shapes` | 命中该 candidate 的输入 shape 样本（用于 S4 的 profiled shape 测试点） |
| `Input Data Types` | 对应 dtype 样本 |
| `aicore_time(us)` / `aiv_time(us)` / `cube_utilization(%)` | 留存备用（当前不参与占比计算） |

注意：个别数值字段可能内嵌 tab（如 `Start Time(us)`），用 `csv.DictReader` 按 header 取列并
对数值 strip/coerce 即可，Duration 字段本身干净。

`step_trace_time.csv`：用 `Step` 列作为稳态 step 集。其 `Computing` 等耗时字段当前不作为占比分母
（分母统一用 kernel_details 内该 step 的 Duration 之和，口径自洽）。

## 二、稳态窗口与占比口径（§6.2 / §6.3）

- 稳态 step 集 = `step_trace_time.csv` 的 `Step`；缺失 → 回落为 kernel_details 去重后的 `Step Id`。
- 每 step：`step_total = Σ Duration(us)`；`op_total[name] = Σ Duration(name)`；
- `op占比 = op_total / step_total`；
- **跨 step 平均**：对每个 candidate，mean = 所有稳态 step 的 per-step 占比的平均（缺场均按 0）；
  同时给 `coverage = 出现该 name 的 step 数 / 稳态 step 数`，供编排参考。

## 三、两档 Triton 判定（§6.4，config 全可配）

默认规则：

```jsonc
{
  "builtin_prefixes": ["aclnn", "hcom", "Hccl", "aicpu", "Atc", "acl",
                        "aclnnGroupedMatmul", "ForeachCopy", "RmsNorm"],
  "strong_patterns": ["_kernel$", "kernel", "triton"],
  "semi_builtin_prefixes": ["ascend", "npu", "aco_", "mep"]
}
```

| 判定 | 条件 | 去向 |
| --- | --- | --- |
| 排除（None） | `Name` 以任一 `builtin_prefixes` 开头 | 不进候选 |
| strong | 非内建前缀 且 命中任一 `strong_patterns` | 可进 S4 |
| weak | 非内建前缀、无强特征 或 命中 `semi_builtin_prefixes` | 「待人工核定」，不改判、不自动进 S4 |

语义主观性提醒：`Name → 是否 Triton` 是本工作流唯一主观点；同名多实现、aclnn 邻接名一律保守归 weak。

## 四、`candidates.json` 结构

见 `SKILL.md`「产出 candidates.json」一节。字段：

- `tier`：`strong` / `weak`
- `mean_op_ratio`：跨 step 平均 op 占比（排序键，降序）
- `coverage`：稳态 step 出现率
- `total_duration_us`：稳态 step 内该 op 的累计耗时
- `shapes`：去重后按出现顺序取前 N（`max_shapes`）个 `{shape, dtype, core}`

## 五、验证

- 单测：`python -m pytest skills/ops-skills/triton/profile-bottleneck-analyzer/tests/`
- 真实冒烟（存在数据才跑）：tests 内 `test_real_zips_smoke`，扫环境变量
  `PROFILE_SMOKE_DIR` 指向目录下的 `*.zip`，断言能识别 `_kernel` strong 候选。
  未设置该变量（或目录下无 zip）时自动跳过，仓库内不写死任何本机路径：
  `PROFILE_SMOKE_DIR=/path/to/profiles python -m pytest .../tests/`。
