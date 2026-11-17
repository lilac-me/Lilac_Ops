# Profiling 定位规则

命令示例以本 Skill 根目录（与 `SKILL.md` 同级）为工作目录。

当 Phase 1/2 batch invariance 失败，并且已经采集到昇腾 profiling 结果时，使用本参考文档定位可疑算子和下一步验证方向。

## 输入

- 失败用例对应的 `summary.csv` 行和 attempt 日志。
- 失败场景的模型路径、完整并行拓扑与各维度 size、设备、prompt 长度、输出长度、请求并行数、temperature、seed 和 prefix caching 配置。
- 在同一失败配置下采集到的 `ASCEND_PROFILER_OUTPUT/op_statistic.csv`。

## 分析方法

执行：

```bash
python3 scripts/analyze_op_statistic.py \
  <path>/ASCEND_PROFILER_OUTPUT/op_statistic.csv \
  --json \
  > <evidence-dir>/phase1_operator_analysis.json
```

白名单外算子默认返回非零退出码，不需要增加 `--strict`。A2/A3 和 A5 的退出码与分析器决策一致：全部算子在对应白名单内为 `PASS_TRIAGE/0`；白名单外且黑名单外为 `REVIEW_REQUIRED/2`；命中公共黑名单为 `FAIL_BLACKLIST/3`。退出码 `3` 是代码检查与修复重验的入口，不代表可以忽略原始失败后直接进入 Phase 2。

结果解释：

- 算子判定顺序是“先白名单、后黑名单”：先将算子拆分为白名单内和白名单外；仅对白名单外算子检查 `SoftmaxV2`、`MatMulV2`、`FusedInferAttentionScore`、`AddRmsNorm`、`ReduceSum` 五个基础名称。白名单外命中黑名单时，输出并保留 `FAIL_BLACKLIST`、返回非零退出码，并跳过基础算子的单算子验证。随后检查目标模型和当前 vLLM-Ascend 的实际调用代码、注册入口、参数语义与芯片实现；禁止仅拼接 `BatchInvariant` 推断替换。恢复状态允许实施证据最强的最小修改并重新 profiling、重跑 10,000 条验证，五轮不能闭环后才通知用户。
- `allreduceAicpuKernel`、`allgatherAicpuKernel`、`hcom_allReduce` 和 `hcom_allGather` 属于通信相关算子。它们单独出现不等价于失败，需要结合 TP、阶段、耗时和 mismatch step 一起判断。
- 如果出现白名单外算子，`batch_invariance_satisfied` 为 `null`，不能声明满足或进入 Phase 2。先把它们作为第一优先级嫌疑点；逐项完成关键路径分析，关键路径算子再结合 profiling 中的 shape、dtype、rank、device 和输入分布构造单算子 batch invariance 复现用例。
- 如果 profiling 中出现的算子都在 `batch_invariant_ops.md` 对应芯片分区的白名单内，继续排查调度、随机性、prefix caching、seed、TP/device 映射和模型级二分。
- 当前正式支持的并行证据只覆盖 TP。PP、EP、DP 的 profiling 只能支撑“理论支持场景下的实验性分析”，其他并行方式只能支撑“可能支持场景下的探索”；不能用算子出现情况替代端到端验证。
- temperature 是唯一支持泛化的采样参数，可为 `0`、`0.6` 等有限非负值。其他采样参数不在当前分析矩阵内。
- A5 使用独立白名单分区，并与 A2/A3 复用相同强制黑名单。A5 算子包由环境阶段使用 `ascend950` 自动构建；环境缺失不改变 profiling 分析规则。

## 必须报告

报告需要包含以下字段：

- profiling 根目录和 `op_statistic.csv` 路径
- 白名单内算子
- 白名单外算子
- 通信相关和 MoE 相关算子
- 下一步动作：代码替换检查、等待用户确定、单算子复现、模型级二分或环境修复
- 并行能力状态和 temperature；非 TP-only 场景必须显式标为未验证

最终回复按 `final-report.md` 的 Phase 1 表格逐项关联 profiling 根目录、CSV 和 `phase1_operator_analysis.json`。多个 CSV 不得共用一个无法区分来源的分析结果文件。
