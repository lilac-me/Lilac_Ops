# 本轮优化决策记录

在调用 CANNBot 前填写 A–D，整网测试后填写 E；未知写 unknown，不填虚构值。

## A. 固定基线

- Run / 当前保留版本 / workload ID：
- 整网主指标、单位、方向及实际 baseline：
- 实际工作量、采样窗口、环境与配置快照：
- 基线波动、最小正收益门槛、精度及保护指标容差：
- 实测来源（无 profiler）：

## B. 关键路径与候选比较

| 候选 | rank/phase/step | kernel 累计时间（仅线索） | 关键路径分类及 trace 证据 | p 估计/范围 | s 估计/范围 | 整网预告/置信度 | 成本/风险 | 决策 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待分析 | unknown | unknown | UNKNOWN | unknown | unknown | unknown | unknown | DEFERRED |

证据请写真实文件、事件/时间区间、依赖与同步关系；如果不能确认关键路径，不填写伪造比例。记录预测假设及上界，区分实测与估计。

## C. 唯一优化单元

- 选定算子/不可拆分融合子图：
- 为什么选它（实际关键路径、收益潜力及成本）：
- 真实源码/dispatch、形态、reference、语言及目标插件：
- 当前 AgentLoop task / request 路径：
- 其它候选：DEFERRED。

## D. CANNBot 内层结果

- 原始实测 latency / 优化实测 latency / 加速比：
- 迭代轮数、精度验证、最佳版本及补丁证据：
- 整网新实现实际执行的确认方式：

## E. 整网独立验收

| 指标 | Baseline 实测 | 优化后实测 | 改善率/是否过门槛 | 证据 |
| --- | --- | --- | --- | --- |
| 主指标 | 待测 | 待测 | - | - |
| 精度 | 待测 | 待测 | - | - |
| 保护指标 | 待测 | 待测 | - | - |

- 预测与实测差异及诊断：
- 最终状态：NOT_VERIFIED / E2E_GAIN / NO_E2E_GAIN / ACCURACY_FAILED / PERFORMANCE_REGRESSION
- 保留版本或回滚证据：
- 后续候选应使用的最新基线：
