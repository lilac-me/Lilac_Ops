# Batch Invariance 最终报告

## 1. 报告原则

从环境准入开始维护本报告的运行副本，并保存到 `<RUN_RECORD_DIR>/batch_invariance_final_report.md`；每完成一个阶段立即更新，不得只在全部流程结束后汇总。报告中的证据使用相对于 `<RUN_RECORD_DIR>` 的位置；未生成的证据写“未生成”并说明原因，不使用预期位置冒充实际产物。最终回复只给出运行记录目录和最终报告，不展示 skill 安装目录、内部脚本路径或 reference 路径。

每个被执行的命令都必须落盘保存可复现记录：完整命令或参数化脚本、执行环境/输入配置、开始与结束时间、stdout、stderr 和退出码。若阶段内构造、修改或临时生成了测试脚本（尤其是单算子批次不变性用例），必须将实际执行版本复制到运行记录目录，或保存可还原该版本的 Git commit 加补丁；仅记录源目录路径不算交付。敏感凭据必须脱敏。

阶段状态：

| 状态 | 含义 |
| ------ | ------ |
| `PASS` | 该阶段全部必需检查和场景通过，证据完整 |
| `FAIL` | 已获得明确不满足证据 |
| `BLOCKED` | 因环境、OOM、接口、数据或证据缺失无法完成判定 |
| `NOT_RUN` | 用户未要求或前置阶段未通过，因此未执行 |

整体状态：

- `PASS`：环境准入、Phase 1、Phase 2、Phase 3 全部 `PASS`。
- `FAIL`：任一已执行的必需阶段为 `FAIL`。
- `BLOCKED`：没有明确 `FAIL`，但任一必需阶段为 `BLOCKED`。
- `PARTIAL`：只要求或只完成部分阶段，其余阶段为 `NOT_RUN`。

A5 与 A2/A3 使用相同三阶段状态规则。缺少算子包、SO 或 Torch 注册时，环境准入从匹配的 vLLM Ascend 源码分别使用 `ascend910b`、`ascend910_93` 或 `ascend950` 自动构建并复检。

## 2. 最终报告模板

```markdown
# Batch Invariance 验证报告

## 结论

- 整体状态：<PASS|FAIL|BLOCKED|PARTIAL>
- 当前结论：<一句话说明通过范围或首个阻塞/失败阶段>
- 运行记录：<RUN_RECORD_DIR>
- 最终报告：batch_invariance_final_report.md
- 模型：<model path>
- vLLM / vLLM Ascend 来源：<version / git commit / branch / commit time / collected UTC；不可取得的字段写 unavailable>
- 芯片与设备：<SOC_VERSION / ASCEND_RT_VISIBLE_DEVICES>
- 并行能力：TP=<sizes，按实际证据判定>；PP/EP/DP=理论支持、尚未验证；其他并行=可能支持、尚未分析/验证
- 采样参数：Phase 2 temperature=<value，默认 0.6>；Phase 3 temperature=0

## 阶段汇总

| 阶段 | 状态 | 核心结论 | 主证据 |
|------|------|----------|--------|
| 环境准入 | <status> | <TP confirmation + package provenance + package/SO/torch op> | <preflight output/log> |
| Phase 1 | <status> | <对应芯片 allowlist/公共 blacklist/单算子结论> | <analysis result> |
| Phase 2.1 长输出 | <status> | <TP/BS/length/temperature> | <summary.csv> |
| Phase 2.1 长输入 | <status> | <TP/BS/length/temperature> | <summary.csv> |
| Phase 2.2 稳定性矩阵 | <status> | <矩阵与轮数> | <summary.csv> |
| Phase 2 汇总 | <status> | <三个场景是否齐全> | <summary files> |
| Phase 3 | <status> | <配置、accuracy、评分与原始文本一致性> | <raw_result + phase3_acceptance.json> |

## 交付件清单

只登记已落盘且核验存在的文件。`SHA256` 用于证明报告引用的脚本、配置和关键结果与实际执行版本一致；不适用时写 `N/A` 并说明原因。

| 阶段 | 类型 | 相对路径 | SHA256 | 说明 |
|------|------|----------|--------|------|
| 环境准入 | 配置/命令/日志/退出码 | <environment/...> | <hash> | <版本、TP 门禁及五项预检> |
| Phase 1 | profiling/分析结果 | <phase1/...> | <hash> | <rank、op_statistic、白/黑名单结论> |
| Phase 1 | 单算子脚本/补丁 | <phase1/operator_cases/...> | <hash> | <实际执行版本；未触发则说明> |
| Phase 1 | 单算子命令/输入/日志/退出码 | <phase1/operator_runs/...> | <hash> | <API、shape、dtype、有效 NPU 用例数> |
| Phase 1 | 替换 diff 与重验 | <phase1/revalidation/...> | <hash> | <未触发则说明> |
| Phase 2 | 场景配置/脚本版本/日志/汇总 | <phase2/...> | <hash> | <按场景和 attempt 分项登记> |
| Phase 3 | 评测配置/命令/日志/原始预测/验收结果 | <phase3/...> | <hash> | <10 轮逐项登记或索引> |

脚本交付件至少记录：来源、实际执行文件、Git commit（如适用）、补丁（如有）、完整命令、输入配置、stdout、stderr、退出码、开始/结束时间。报告结论不得引用未出现在本清单中的关键证据。

## Phase 1 算子 Profiling

- 是否执行：<yes/no>
- 芯片支持状态：<A2/A3/A5；安装平台参数；自动构建与复检结果>
- 分析参数与退出码：<chip/rank/options/exit code；不展示内部入口路径>

| Profiling 根目录 | Rank/范围 | op_statistic.csv | 分析输出 | 退出码 | 决策 |
|------------------|-----------|------------------|----------|--------|------|
| <phase1-relative-root> | <rank/all> | <relative csv path> | <relative JSON/text path> | <0/2/3/...> | <PASS_TRIAGE/REVIEW_REQUIRED/FAIL_BLACKLIST> |

- 白名单内算子：<list or count + analysis output reference>
- 白名单外算子：<list>
- 黑名单命中：<list>
- 黑名单修复：<未触发，或 initial FAIL_BLACKLIST / 实际代码位置 / 修改 diff / 用户决策 / 重采与重验结果>
- 替换恢复：<none，或每轮根因假设、最小修改、diff、验证和最终恢复耗尽状态>
- 白名单外评审：<无，或逐项关键路径排除/10,000 条验证结论；REVIEW_REQUIRED 不得直接写 PASS>
- 通信/MoE 辅助分类：<list>
- 单算子 10,000 条验证：<未触发，或列出 API、有效用例数、结果文件和首个失败>
- Phase 1 结论：<PASS|FAIL|BLOCKED>

## Phase 2 功能测试

| 场景 | TP | BS | prompt:max_tokens | temperature | KV capacity/required | Scheduler | 退出码 | 状态 | summary.csv | summary.md | scheduler JSON | failure manifest | attempt 日志 |
|------|----|----|-------------------|-------------|----------------------|-----------|--------|------|-------------|------------|----------------|------------------|--------------|
| Phase 2.1 长输出 | <...> | <...> | <...> | <...> | <...>/<...> | <status> | <...> | <...> | <path> | <path> | <paths> | <paths or 未触发> | <paths> |
| Phase 2.1 长输入 | <...> | <...> | <...> | <...> | <...>/<...> | <status> | <...> | <...> | <path> | <path> | <paths> | <paths or 未触发> | <paths> |
| Phase 2.2 稳定性矩阵 | <...> | <...> | <...> | <...> | <...>/<...> | <status> | <...> | <...> | <path> | <path> | <paths> | <paths or 未触发> | <paths> |

- 请求驱逐支持状态：不支持
- 最大 KV cache usage：<percent>
- 显式 preemptions：<count>
- Running/Waiting 驱逐状态转移：<none or evidence line>
- 初始排队：<是否出现 Running<BS/Waiting>0；仅记录，不作为阻塞>

上述 Running/Waiting 状态转移字段仅用于 Phase 2 的静态 BS=N 场景。Phase 3 Math500 不输出或使用该字段作为门禁；`Waiting` 仅作为持续请求流的调度观测，最终报告应聚焦明确的 preemption/eviction、请求/服务错误、每轮 500 条成功样本和跨轮完整原始预测一致性。
- chunked prefill / block size：<False / 128；与 profiling 和 Math500 是否一致>
- 失败 marker：<none，或 BATCH INVARIANCE FAILURES 摘要>
- step mismatch：<count/details>
- token mismatch：<count/details>
- logprob mismatch：<count/details>
- Phase 2 结论：<PASS|FAIL|BLOCKED|NOT_RUN>

`EVICTION_UNSUPPORTED` 或 `INCOMPLETE` 任一 Scheduler 状态都使对应场景为 `BLOCKED`。初始排队不构成阻塞；只有满批运行后回退为运行数减少且等待数增加才属于驱逐状态转移。报告下一步参数调整，不把调度阻塞写成模型数值不一致。

## Phase 3 Math500

- 公共 accuracy Markdown 报告：<phase3-relative-path>
- 公共 `raw_result`：<phase3-relative-path>
- 公共 accuracy skill 退出码：<code>
- 模型与服务配置：<validator scenario_config 逐项结果；model/devices/CANN/TP/DP/PP/EP/dataset/batch size/seed/VLLM_BATCH_INVARIANT/prefix caching/serve args>
- temperature：0；seed：1234
- 每轮 accuracy：<round 1..10>
- 唯一分数集合：<values>
- accuracy 阈值及结果：<threshold / pass/fail>
- 评分一致性：<pass/fail>
- 原始预测文件：<10 phase3-relative JSONL paths>
- 原始文本一致性报告：phase3/phase3_acceptance.json
- `validate_math500_acceptance.py` 退出码：<code>
- 预测轮数/每轮样本数：<10 / 500>
- 原始文本一致性：<pass/fail and mismatch count>
- Phase 3 结论：<PASS|FAIL|BLOCKED|NOT_RUN>

## 失败、阻塞与后续动作

| 阶段 | 类型 | 首个证据 | 下一步动作 |
|------|------|----------|------------|
| <phase> | <FAIL/BLOCKED> | <path + concise evidence> | <action> |
```

## 3. Phase 1 证据要求

按 [Profiling 分诊规则](../../batch-invariance-operator-analysis/references/profiling-triage.md) 为每个 `op_statistic.csv` 保存独立 JSON 分析结果，并在模板表格中逐项关联。采集报告、`kernel_details.csv` 和 `trace_view.json` 只证明 profiling 完整性，不能替代 `op_statistic.csv`；缺失时标记 `BLOCKED_MISSING_OP_STATISTIC`。

## 4. Phase 2 证据要求

三个正式场景分别产生独立结果目录。报告从各自 `summary.csv` 读取状态、KV capacity/required、SchedulerStatus 和失败计数，并引用 scheduler JSON、`summary.md` 与 attempt 日志。发生 token 或 logprob 不一致时还必须引用当前 attempt 的 failure manifest；dump 方法及理由记录在“后续动作”中，并以 `model-debugging.md` 的证据规则为准。pytest footer 不能替代 scheduler 检查、`summary.csv` 和 `BATCH INVARIANCE FAILURES:` marker 检查。

只执行部分正式场景时，整体报告为 `PARTIAL` 或 `BLOCKED`，不能写完整 Phase 2 通过。任一场景命中 `EVICTION_UNSUPPORTED` 时，该场景和 Phase 2 汇总均不能为 `PASS`。详细判定以 `phase2-acceptance.md` 和 `scheduler-eviction.md` 为准。

## 5. Phase 3 证据要求

同时引用公共 Markdown 报告、`raw_result`、10 份预测 JSONL 和 `phase3_acceptance.json`。缺少任一项或不满足 `math500-acceptance.md` 的轮数、样本数与一致性要求时，Phase 3 不能为 `PASS`。
