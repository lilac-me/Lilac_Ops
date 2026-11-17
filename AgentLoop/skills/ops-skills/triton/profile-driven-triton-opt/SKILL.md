---
name: profile-driven-triton-opt
description: >-
  This skill should be used when the user wants to identify hot Triton kernels
  from Ascend profiling and tune their existing sources with cannbot. It supports
  the operator-level S0-S6 workflow and a single model pass: collect profiling
  once, freeze the candidate batch, tune each operator, integrate verified
  patches, and measure whole-model regression once without a model-level MFU
  target. It validates correctness and performance against the user's original
  kernels and produces patches and measured reports; it does not generate new operators.
version: 1.1.0
domain: ops
function: triton
tags:
  - triton
  - tuning
  - profiling
  - workflow
  - acceleration
author: mindspeed-agent
compatibility: MindSpeed >= 0.1.0, CANN >= 9.1.0, Triton >= 3.2.2
targets-repo:
  - triton-ascend
---

# Profile 驱动 Triton 算子自动调优（顶层 S0–S6 工作流）

## 功能概述

把「profile 解析 → 瓶颈识别 → 源装配 → 单算子性能调优 → 汇总报告」串成一条**算子级**闭环。
用户要求从模型训练采集热点并返回整网优化结果时，使用新增的**整网单批模式**：
先读 [整网接入与配置](docs/model-hotspot-pass.md)，运行
`plugins/ops/engine/model_hotspot.py` 的 `run`，或按 `prepare → 算子调优 → finish` 执行。
仅采集一次热点并冻结本轮清单；算子内部可迭代，整网只做一次优化后回归，不设置整网 MFU 目标。

下述 S0–S6 仍是复用的算子内层；算子报告中的「未做整网验收」只描述内层产物，
整网模式的实际验收结果以 `model_report.json` / `model_report.md` 为准。

入口命令：

```text
@profile-driven-triton-opt <profile.zip> [--config <config.json>] [--source-map <map.yaml>] \
    [--metric-mode speedup|mfu]
```

**算子内层定位与边界（整网外层见上述接入文档）：**

- 只做**算子级归因**，不做端到端验收。
- **只做性能调优，不重新生成算子**——用户源即已产出的真实 triton kernel。
- 加速比**分母唯一 = 用户提供的源本身**。
- 替换回图验证不做，是用户责任。
- 源装配是**全链 #1 前置依赖**；无源 → `SKIPPED_NO_SOURCE`，不硬造。

## 依赖 Skill

| Skill | 阶段 | 用途 |
| --- | --- | --- |
| `profile-bottleneck-analyzer` | S1/S2 | 解析 zip、稳态聚合、两档判定、产出 candidates.json |
| `triton-op-source-assembler` | S3 | 源装配 + 校验 + SKIPPED_NO_SOURCE |
| `vendored-triton-op/*` 调优模块 | S4 内层 | verifier 收敛门 + latency/simulator optimizer（vendor 副本）；**调优必须经 `triton-latency-optimizer` 唯一执行通道**（`Index.md` 按编号有序扫描，点数 N 当场读取**不硬编码**，命中即加载参考文档 + checklist，cursor 驱动，`scan_complete` 编排器推导），禁止自写 tiling 扫描顶替。该 skill 现为「无状态单步函数 + 每次调用前强制 KB 检索（preflight + 盒装摘要）」，编排器须把摘要记进 `optimizer.kb_preflight` 证物 |
| `vendored-triton-op/triton-mfu-analyzer` | S4（`metric_mode=mfu`） | MFU 测定通道：极限 case 搜索 + `flops.py` 自检 + 峰值标定 + `mfu_decompose`（`mfu_achievable_pct`）。speedup 模式下不调用 |
| `restructure_engine/` + `scripts/restructure_guard.py` | S4.R（可选） | 调用侧 kernel-body 重构（读放大/gather 复用瓶颈时，见 `docs/restructure-stage.md`）。**当瓶颈是结构性问题、vendor 33 点全不含重构时**，经 4 闸守卫（g1 I/O 冻结+g2 位级矩阵+g3 IR 同构；g4=官方 verify/benchmark 溯源）后作为新 code_version 基线；vendor **字节不动**。`config.restructure.enabled=false` 默认关 |

## S0 run 目录（强制）——每次运行隔离，默认不看之前的结果

入口第一条**必须**先创建**本次唯一运行目录** `RUN_DIR`，贯穿 S1–S6 所有阶段：

```text
RUN_DIR=$(python3 scripts/init_run.py <out_root>)   # 默认 fresh：总是新建，绝不复用旧目录
```

- `init_run.py` 返回 `<out_root>/profile_opt_<ts>_<rid>/`（ts=UTC 时间戳，rid 随机）；写成 `run.json` 溯源凭据。
- **fresh-run 纪律**：起跑**默认只看 `RUN_DIR` 自己的产物**——S2 candidates 必须由本次
  `parse_profile -o $RUN_DIR/s2_analyze` 产出；**禁止读取/复用其他 run 目录或平铺旧产物的
  candidates/summary 作为本轮判定依据**。曾因平铺旧产物被静默继承、把上一轮 BELOW 当本轮结果。
- 续跑必须**显式** `--reuse-last`；不传则一律 fresh。

## S0 指标模式（metric_mode）与设备隔离

`metric_mode` 是本工作流自己的 S4 判定开关：优化目标、采纳判定与退出判据都由它决定，
**两种模式只有这一处差异，S0–S3 流程完全一致**。

| metric_mode | 优化目标 | 判定量 | 测定通道 |
| --- | --- | --- | --- |
| `speedup`（默认） | 相对用户源的加速比 | `speedup_vs_user = 新 triton / 用户源` | `verify.py` + `benchmark.py`（取 `implementation.avg_latency_ms`） |
| `mfu` | 极限 case 的算力利用率 | `best_mfu` / `baseline_mfu` vs `target_mfu`（默认 60%） | `triton-mfu-analyzer`（`mfu_decompose.json` 的 `mfu_achievable_pct`），**不以 benchmark.py 作判据** |

判定优先级（禁止静默臆造）：显式 `--metric-mode` > `config.metric_mode` > 入口提示词中的指标词
（「加速比 / speedup」→ speedup；「MFU / 算力利用率」→ mfu）。都没有时按文档默认 `speedup`，
并必须在报告里注明「metric_mode 取自默认值」。`metric_mode` 写入 `RUN_DIR/metric_mode.txt`
供 S4/S5 引用；`mfu` 模式另需在 RUN_DIR 冻结 `mfu_grid.json` / `extreme_case.jsonl` /
`flops.py` / `peak.json`（口径冻结后整轮复用，分子或峰值变了 MFU 不可比）。

**设备隔离（强制）**：进入 S1 前按 `config.npu_device` 设定进程级
`ASCEND_RT_VISIBLE_DEVICES`（torch_npu 只认该变量；只设 `ASCEND_VISIBLE_DEVICES` 会让所有
进程落到物理 0 卡 → 507014/507034）。`null` = 沿用既有环境；设置后逻辑卡号从 0 开始，
S4 传给各 skill 的 `npu` 参数一律用逻辑号。串行单卡运行全流程固定同一张卡，禁止中途换卡。

## 编排约定（详见 AGENTS.md）

- S4 逐 candidate **串行**遍历（避免并行致 NPU 环境污染/flaky）。
- 三态结果：`REACHED`（`metric >= target_metric` **须 `optimizer_rounds>0` 且有非空 `best_code`**）/
  `BELOW`（成功但未达标，或 REACHED 前置未满足自动降级）/ `FAILED`。
  其中 `metric` / `target_metric` 按 `metric_mode` 取义：speedup 模式为 `speedup_vs_user` /
  `target_speedup`；mfu 模式为 `best_mfu` / `target_mfu`。
- **REACHED 硬前置**：未跑调优轮（`optimizer_rounds<=0`）或无 `best_code`，一律降 `BELOW`，不得判 `REACHED`。
- `target_speedup` 默认 **2.0x、`target_mfu` 默认 60%**，均可配置；报告只给「是否达标」，**不吹实际收益**；另给
  「建议收不收 best_code」次要列（仅兜底可执行建议，不改判不夸大）。
- `tier=weak` 候选：只进 batch_report「待人工核定」栏，给建议不改判、不进 S4。

### 产物契约（防 torch 参考顶包性能分母）

每个 candidate 的 `summary.json` 必须显式区分两个角色字段，S4 出口过
`scripts/check_summary_contract.py` 强制校验：

| 字段 | 角色 | 允许 |
| --- | --- | --- |
| `gate_ref` | 收敛门真值（torch 参考） | **仅数值校验**，禁止进性能分母 |
| `perf_baseline` | 性能分母 | **必须指向用户源 triton 文件**，不得等于 `gate_ref` |

**硬约束**：`perf_baseline` ≠ `gate_ref`；把 torch 参考当 `speedup_vs_user` 分母即契约 FAIL。

| 字段 | 角色 | 允许 |
| --- | --- | --- |
| `restructure` | S4.R 重构证物（可选/正交） | 仅当本轮 best_code 由 restructure 而来；缺省不影响既有契约。含 `trigger`/`applied_transform`/`engine`/`gates`/`io_freeze_basis`；`optimizer` 块不受影响 |

**测量溯源闸（防手写计时器顶包官方数据）**：总结的 `bench` 块必须可溯源官方产物。

- `bench.tool` 必须是唯一合法计时器标识 `scripts/benchmark.py`（内部用 torch_npu.profiler）；
  手写计时器（`torch.npu Event` / `time.perf_counter`）即契约 FAIL。
- `bench.baseline_result_json` / `bench.optimized_result_json` 必须存在并含
  `implementation.avg_latency_ms`，且 `bench.baseline_avg_ms` / `bench.optimized_avg_ms` 与官方产物一致。
- 显式 `bench.waiver = {reason}` 可将溯源放宽为「公示的偏离」：gate 打印 waiver 并标记
  `CONTRACT PASS (WITH WAIVER)`（deviation 被可见化，而非静默放行）。

因此基线/优化的最终性能数字一律回灌官方 `verify.py` + `benchmark.py`（Mode-A 适配器）产出。

### S4 出口自检表

落 summary 前逐项自检（任一项不满足即不认定为 REACHED）：

- [ ] `perf_baseline` 已显式指向**用户源 triton**（分母唯一），且 ≠ `gate_ref`
- [ ] 本轮**确实跑了调优轮**（`optimizer_rounds > 0`），非基线试跑
- [ ] `REACHED` 时存在**非空 `best_code`**（优化后产物）
- [ ] `bench.tool = scripts/benchmark.py`，且基线/优化数字溯源官方 result JSON（`verify.py`+`benchmark.py` 回灌）
- [ ] `optimizer_rounds>0` 时带 `optimizer.tool = triton-latency-optimizer` 及 `points_hit` / `loaded_docs` / `points_considered` / `scan_complete` / `scan_trace` 证物（调优执行者闸 presence + scan-trace 穷尽闸）；**应用轮的 `record.mandatory` 覆盖强制点集**（`--index` 读 Index.md `MANDATORY_PER_ROUND`，如点 31 IR 分析每轮必须执行，禁止并进 not_hit 静默跳过；强制点闸）
- [ ] `optimizer.kb_preflight` 已记录本轮调优的强制知识检索（`called` / 主证据卡 / 指向优化点；KB 不可用记 `available=false` 并说明原因）。latency-optimizer 现为「每轮调用前强制 preflight」的单步函数，无检索摘要即视为未按纪律执行（缺省仅告警，不阻断）
- [ ] `best_code` 非空时带 `patch` 块（`mode` / `path` / `diff_from`=冻结用户源副本 / `apply_target`=perf_baseline / `resverified`）；补丁已字节派生并经 verify 重验（patch 闸）
- [ ] （`restructure.enabled` 且本轮 best_code 由 restructure 而来时）`summary.restructure` 块存在且 `gates.{g1_io_frozen,g2_matrix,g3_ir_homology}` 全 true，且 `optimizer_rounds>0`（防「重构+直接判 REACHED」）；报告必含「重构过程」「调优过程」两节（R 阶段 / §8）

## 默认 config.json

```jsonc
{
  "bottleneck_threshold": 0.03,
  "metric_mode": "speedup",      // speedup | mfu（S4 判定口径；--metric-mode 优先）
  "target_speedup": 2.0,         // speedup 模式目标，可配置，用户按算子实际调
  "target_mfu": 60,              // mfu 模式目标（%，dtype 修正实测峰值口径）
  "arch": "auto",
  "npu_device": null,            // 写入 ASCEND_RT_VISIBLE_DEVICES 的物理卡号；null=沿用环境
  "npu_preflight": { "min_free_mb": 49152, "smi_timeout_seconds": 5 },
  "run_mode": "serial_single_card"
}
```

## 产物（强制写入 RUN_DIR，禁止平铺到 out_root）

每次运行必须有**独立 `RUN_DIR`**（由 `scripts/init_run.py` 新建，见「S0 run 目录」），
**所有阶段产物一律写进 `RUN_DIR/`**。`run.json` 为本轮溯源凭据；禁止把 candidates/summary
平铺到 `<out_root>` 根或复用到其他 run 目录（违反即判定失效，需重跑）。

```text
RUN_DIR=out/profile_opt_<ts>_<rid>/   # init_run.py 每次新建；run.json 溯源
  metric_mode.txt      # S0：本轮判定口径（speedup | mfu）
  mfu/                 # S4（仅 mfu 模式）：mfu_grid.json / extreme_case.jsonl / flops.py / peak.json
  candidates.json      # S2（parse_profile -o $RUN_DIR/s2_analyze）
  assembly/            # S3：每 candidate 输入包
  op_<name>/           # S4：每 candidate 独立工作目录（summary.json / report.md / best_code）
    patch/             # S4.5：{op}.patch + apply.sh + README（采纳更优 best_code 时强制产出）
  batch_report.md      # S5：汇总表（target/actual/是否达标/失败原因/待人工核定）
  session.jsonl        # S6：当前 Claude Code 会话原始记录（后台自动存储的 jsonl 副本）
  session.md           # S6：会话 Markdown 渲染（经 render_session.py；渲染失败可能缺失）
```

- **S6**：`session.jsonl` = 后台 `project/*.jsonl` 的副本（用工作目录绝对路径精确过滤定位）；
  `session.md` 由 `skills/ops-skills/vendored-triton-op/render_session.py` 渲染。定位/渲染失败**仅告警不阻塞**。
