# MindSpeed-Agent 能力 Benchmark

> 目标：给这个代码仓里每个 Agent 建立**能力基线**，让「不同模型 × 不同 harness」在**同一张表**上可比，并精确回答「同样都完成的时候，哪个完成得更好」。

## 这个 benchmark 要回答的两个问题

### 问题一：每个 agent 的能力基线是什么？

把每个领域 Agent 的能力抽象成**一组确定性的、自包含的任务（case）**，在同一套评分公式下打分。因为 case 与公式对所有人都一样，分数就能横向比较：

- 同一个 case，用不同 **model**（sonnet / opus / GPT / DeepSeek-V4 …）跑，看谁得分高；
- 同一个 case，用不同 **harness**（Claude Code / Codex / DSH）跑，看谁的编排/工具链更有效；
- 同一个 agent，跑**不同难度**的 case（easy / medium / hard / complex），就是它的能力边界。

### 问题二：同样都完成，怎么算「完成得更好」？

关键原则：**效果与成本必须分开量，不能揉成一个数**。

| 维度 | 含义 | 越大越好？ |
| ------ | ------ | :---: |
| **效果分（effect）** | 完成度 + 性能 + 工程完整性 + 收敛质量，四项加权 | ✅ 越高越好 |
| **成本（cost）** | token 总数、墙钟耗时、迭代/派生次数 | ⬇️ 越低越好 |
| **效费比（effect / token）** | 单位 token 换来的效果 | ✅ 越高越好 |

「完成得更好」的判定顺序（也是报告的排序规则）：

1. **先比完成**：没把任务做对（完成度 < 100）的 run，永远排在做对的后面；
2. **再比效果**：都完成时，效果分更高者更优（精度更准、性能更好、工程更扎实、收敛更诚实）；
3. **最后比成本**：效果相当时，token / 耗时更少者更优（效费比更高）。

> 结论：**token 更少**与**效果更好**不是二选一，而是两级排序——效果是主序，token 是同效果下的打破平局项。真正要警惕的是「为了省 token 牺牲效果」，所以算式里永远先保效果、再看成本。

## 五维评分模型

每个 run 产出 5 个 0–100 的分。前 4 维构成**效果分**，第 5 维与原始成本一起构成**成本面**。

| # | 维度 | 测什么 | 证据来源 |
| :---: | ------ | -------- | ---------- |
| D1 | completion 完成度 | case 声明的形状 × dtype + 强制掩码的尾块用例，通过率 | 真机精度测试结果 |
| D2 | performance 性能 | `100 × clamp(deciding_speedup) × 覆盖率闸门`（读 `bench_result.json`，缺失才退回解析 `speedup.md`）。deciding = **最大 required shape 内最差的 dtype**（保守取值）；闸门 = 实际覆盖的 required shape 占比 | 真机 profiler 实测（重复 3 次取中位数 + std） |
| D3 | engineering 工程 | 10 项交付物齐备（占 50%）+ 反作弊通过（25%）+ 设计文档节数（25%） | 落盘文件 + `kernel.py` + `design.md` |
| D4 | convergence 收敛 | 结构化 `bottleneck.json` 逐项核对：分类（**与实测真值比对**）、带宽自报、建议分级、诚实收敛理由；缺失则降级为关键词匹配并标注 | `bottleneck.json` + `bottleneck.md` + `state_transitions.jsonl` |
| D5 | efficiency 成本 | 是否落在该难度的时间/token/迭代预算内；**无任何成本证据时记 `null`**（"没测到"≠"满分"） | `run_meta.json` + `state_transitions.jsonl` |

- **效果分 = 0.35·D1 + 0.30·D2 + 0.20·D3 + 0.15·D4**
- **成本** = { total_tokens, wall_ms, iterations }（原始值，不做归一化，越小越好）
- **效费比** = 效果分 / 总 token × 10⁶（仅在采集到 token 时给出）

各维度的精确算法规格见 `agents/ops/scoring.py` 与 `agents/ops/metrics.py` 的 docstring；分数为何这样来，每个维度都带 `note` 说明。

> **证据路径只有一条是契约，其余是兼容**：D2 认 `bench_result.json`、D4 认 `bottleneck.json`，
> 两者都已写进 `plugins/ops/ops.yaml` 的产物契约与出口门禁。**新 run 必须产出这两份**；
> `speedup.md` 的 Markdown 解析与 `bottleneck.md` 的关键词匹配只是**兼容历史 run 的降级路径**
> （脆弱、且拿不到 `speedup_std`/`runs` 这类噪声证据），新 run 不应依赖，评分结果里也会显式标注来源。

### 怎么在 NPU 上测出一条基线

测量实现只有一份（`agents/ops/bench_runs.py`），它**在 NPU 容器里、以算子工作区为 cwd** 运行，
`import kernel` 就解析到该 case 的 kernel；脚本会自己把 cwd 放进 `sys.path`，
所以**不需要把它拷贝到工作区**（早期那种"每个 job 目录一份副本"的做法已经漂移过一次）：

```bash
# 进容器后，cd 到算子工作区（那里有 kernel.py），用仓库里的脚本原样跑
cd outputs/gelu
ASCEND_RT_VISIBLE_DEVICES=0 python /path/to/repo/benchmark/agents/ops/bench_runs.py \
    --case gelu --runs 3
# -> 写出 bench_result.json：每 shape × dtype 的 speedup 中位数 + speedup_std + runs
```

两条硬性前提：

1. **先确认卡空闲**（`npu-smi info`：AICore≈0%、HBM 占用低）。被别的任务占用的卡实测会让
   同一算子的重复测量 std 飙到 32%，那种数字不是基线；
2. **必须重复 ≥3 次**（`--runs 3`）。单次采样不可信——fp32 曾出现 13% 的离散度，
   `gelu` 的最大 shape 在不同轮次间漂移约 7%。

### 为什么是这些 case（对标调研）

36 个 case 的分类不是拍的，它跟随业界算子 benchmark 的**分层共识**：按算子类型分层，
核心维度是**计算单元（Vector/SIMD vs Cube/矩阵）**与**融合程度**——而不是只有纯 element-wise。

| 参考 benchmark | 分类方式 |
| ------ | ------ |
| KernelBench（*Can LLMs Write GPU Kernels?*） | 4 级，按算子类型：element-wise/归一化 → 双张量矩阵 → 融合/注意力 → 卷积等 |
| tritonbench（pytorch-labs） | 按算子族：elementwise / reduction / **matrix（GEMM）** / attention |
| DeepBench（baidu-research） | **GEMM / convolution / recurrent / all-reduce** |
| TritonForge | Kernelbook + KernelBench，同样按算子难度分级 |
| 昇腾 AI Core 编程模型 | **Vector（SIMD，向量/element-wise/归约）** vs **Cube（矩阵/GEMM）** |

Triton-Ascend 里对应「VV（纯向量）」与「CV（含 `tl.dot` 走 Cube）」——这就是 manifest
`kernel_type: VV/CV` 字段的来源。映射到本 benchmark：

| 共识类别 | 计算单元 | manifest 覆盖 | 已实跑 |
| ------ | :---: | :---: | :---: |
| element-wise（激活/算术） | Vector | 15 | ✅ relu、gelu |
| reduction / normalization | Vector（归约） | 10 | ✅ rmsnorm |
| **matrix / GEMM** | **Cube** | 4 | ✅ matmul |
| **attention（融合）** | Cube+Vector | 3 | ✅ sdpa |
| **fusion（多算子）** | 混合 | 3 | ✅ swiglu |
| 其它（conv） | Cube | 1 | ⏳ |

共 36 例（vec 25 + cube/attention/fusion 11）。**已实跑的 6 例覆盖全部 5 个类别**，
含两个关键的非 vec 类别（Cube 的 matmul、attention 的 sdpa），所以「只测纯 vec 算子」
这个常见缺陷在本 benchmark 里是被显式排除的——这也是「不只测 vec」的落地证据。

## 目录结构

```text
benchmark/
├── README.md                     # 本文件：唯一的说明源（方法论 + 现状 + 复现 + TODO）
├── agents/
│   └── ops/                      # 算子开发（ops-dev 流水线）
│       ├── README.md             # ops 领域用法与现状
│       ├── testcases/manifest.yaml   # 36 个算子用例（单源真相，被 ops.yaml 引用）
│       ├── metrics.py            # 证据提取（结构化优先 + 陈旧证据作废）
│       ├── scoring.py            # 五维评分引擎
│       ├── bench_runs.py         # 唯一的精度/性能测量实现
│       ├── runner.py             # list-cases / record-meta / score
│       ├── report.py             # 跑法无关汇总（读 results/**/result.json）
│       ├── sensitivity.py        # 权重敏感性验证
│       ├── console.py            # 各 CLI 入口共用的 UTF-8 stdout 处理
│       ├── harness_adapters/     # dsh_tokens.py —— 真实 token 记账
│       └── tests/                # 50 项单测
└── results/
    ├── _index.md                                    # 汇总对比表
    └── ops/<case>/<harness>__<model>/result.json    # 一个 {case × harness × model} 一行
```

**统一 result schema**：每个领域用自己的 `scoring.py` 产出相同顶层结构
`{domain, case, difficulty, harness, model, completed, dimensions{...}, effect_score, cost}`；
`report.py` 只认这个 schema，因此任意 harness/model 落到同一棵结果树、同一张表。
「五维评分」是 **ops 域**的维度定义；将来接入新领域时，「效果分 vs 成本」的拆分口径保持一致即可。

> **TODO（跨 harness / 跨模型补测：目前所有基线都只有 DSH）**：本 PR 的 6 条结果**全部由
> DSH + `deepseek-v4` 产出**（`relu` 那条是 `reference` 回灌），Claude Code / Codex 以及
> 其它模型**一条数据都没有**——这不是"没差别"，而是"没测"。补测**就是再跑一遍 benchmark**，
> 框架已经支持，不需要另一套跑法或另一套工具链：
>
> 1. 用目标环境的 harness 原生产物触发同一 case（或直接跑 ops 流水线），产出仍落盘 `outputs/<case>/`；
> 2. 按上面的适配器契约写 `run_meta.json`（`harness` / `model` 决定它在表里的身份）；
> 3. `python -m benchmark.agents.ops.runner score <workspace> --case <id>`；
> 4. `python -m benchmark.agents.ops.report` 汇总——新环境的结果**自动进同一张表**，
>    靠 `harness` / `model` 两列横向对比。
>
> 曾尝试过一条旁路：让模型在**没有硬件访问权**的条件下写 kernel，另配一套容器侧评测器、
> 会话级 token 归属和远端脚本（代号 V2，两轮：冷启动 / 带证据优化）。它跑出过一些观察
> （平台 API 契约导致的失败、以及"找到思路≠能实现"），但**跨 harness 数据一条都没有**
> （Claude Code 凭据 403、Codex 未安装），整套装置无法被评审复现，因此已全部移除。
> 相关**结论也一并删除**——证据不留、结论就不留。
> 若将来要做，正确做法是复用上面的四步 + 适配器契约，而不是再造一套旁路。
>
> **TODO（Ascend C 路径尚未纳入评分）**：ops 流水线现在支持双语言
> （`dev_lang = triton | ascendc`，Ascend C 侧还区分 `target_arch = arch22 | arch35`），
> 但本 benchmark 的 6 条基线、36 个 case 的 `requirement` 与 `scoring.py` 的交付物清单
> **全部是 Triton 口径**：D3 的 `deliverable_files` 要求 `kernel.py`，
> 而 ascendc 路径产出的是 `op_kernel/*.cpp` · `op_host/*` · `csrc/` 或 `op_extension/`。
> 因此：
>
> - **现有基线只代表 Triton 实现**，不要当成 Ascend C 的能力基线；
> - 若要评测 ascendc 路径，需要把交付物清单按 `dev_lang` 参数化（`scoring.score(ws, case)`
>   增加语言维度），并给 manifest 的 case 补 ascendc 的 `requirement` 文本；
> - 出口门禁**故意不检查 kernel 源码文件**正是为此：它的路径由 `dev_lang` 决定，
>   写死任一个都会让另一条路径永远过不了门禁（详见 `plugins/ops/ops.yaml` implement 阶段的注释）。
>
> **TODO（未接入的领域）**：`doc-agent`（资料工作流）与 `inference-model-optimize`（推理优化）
> 的 benchmark **尚未落地**，本仓先不提供这两个域的 `scoring.py` / manifest / 结果，
> 以免留下未经实测验证的脚手架。接入时的要点：
>
> - **doc-agent**：包装现有确定性质量引擎（`plugins/doc-agent/engine/state_machine.py score`）
>   的 `quality_report.json`（4 质量维度 + verdict）即可，不必自造评分；但要注意
>   ① 审查类任务的 `verdict=fail` 是「被审文档有问题」这一**发现**，与生成/修复类语义相反；
>   ② L0 只跑确定性 lint、**无 agent 变异性**，谈不上测 agent 能力，能力差异要到 L1/L2 语义审查才出现；
>   ③ 成本必须由 harness 在**同一次** run 内按轮采集——我们曾把 L0 阶段的记账挂到 L1 结果上，
>   使 D5 虚高到 100，这类张冠李戴比没有数据更糟；
> - **inference-model-optimize**：包装 `plugins/inference-model-optimize/engine/gates.py` 校验过的
>   tester 结果（吞吐/时延/拓扑）。注意该域依赖运行环境（vLLM 服务 + NPU + 模型），
>   首跑很可能先暴露**工具自身**的缺陷（我们首次实跑就撞上 `kvcache_calculator.py` 的
>   `norm(cfg, *keys)` 把尾参默认值当 key 用、导致 `num_shared_experts` 取到 `None` 后报
>   `TypeError`）——那种结果不是 agent 能力基线，不要当基线入库。
>
> 两者都**没有单测覆盖**，接入时请一并补上（本域的单测在 `agents/ops/tests/`，可作为模板）。

## 跨 harness / 跨模型对比协议

基准的核心理念是**接管「跑」是 harness 的事，接管「打分」是 benchmark 的事**：

1. **跑**：每个 harness 用自己原生的 agent 机制去跑 ops-dev 流水线（Claude Code 的 subagent、
   Codex 的 task、DSH 的 subagent），产出仍落盘到 `outputs/<case_id>/`；
2. **记账**：每个 harness 写一个标准 `run_meta.json` 到该 workspace，记录
   harness / model / provider / token / 耗时（适配器契约，见下）；
3. **打分**：`scoring` 只读 workspace、不关心是谁跑的，于是不同 harness/model 的产物进入同一评分口径；
4. **汇总**：`report` 把一批 `result.json` 排进**同一张表**。

**换 harness、换模型、换环境都是同一件事：再跑一遍，换掉 `run_meta.json` 里的身份字段。**
结果自动进同一张表，`harness` 与 `model` 两列就是对比维度——不需要为它另造跑法、另造工具链
或另开一张榜。

### `run_meta.json` 适配器契约

```json
{
  "harness": "claude | codex | dsh | manual",
  "model": "sonnet | opus | gpt-5 | deepseek-v4 | ...",
  "provider": "anthropic | openai | deepseek | ...",
  "case": "relu",
  "input_tokens": 12345,
  "output_tokens": 6789,
  "total_tokens": 19134,
  "wall_ms": 2139052,
  "tool_calls": 41,
  "spawns": 3,
  "note": "可选的自由说明"
}
```

`total_tokens` 缺省时记为「未采集」，评分仍会用 wall / iterations 计算成本效率，并显式标注
`tokens_captured: false`。各 harness 从自己的会话元数据里填这个文件即可
（DSH 用 `harness_adapters/dsh_tokens.py`，Claude Code 从 usage API 取，Codex 从 subtask 统计取）。

**完全没有成本证据时（既无 token 也无 wall）D5 记为 `null` 而不是 100**——
「没测到」不等于「满分」。迭代分量永远可算（无 state log 即 0 次浪费迭代），
所以没有这道闸门时，零证据的 run 会在重归一化后拿到满分。

### 换一个环境跑的具体步骤

```text
1) 跑 case：用目标环境/harness 原生的 agent 机制触发同一 requirement
   （ops 看 plugins/ops/README.md 的触发语；产出仍落盘到 outputs/<case>/）
2) 记账：写 run_meta.json 到该 workspace（字段见上面的适配器契约）
   —— DSH 可用 benchmark/agents/ops/harness_adapters/dsh_tokens.py；
      Claude Code 从 usage API 取；Codex 从 subtask 统计取
3) 打分：python -m benchmark.agents.ops.runner score <workspace> --case <id>
   （默认写到 results/ops/<case>/<harness>__<model>/result.json，即 report 能收到的地方）
4) 汇总：python -m benchmark.agents.ops.report \
     --results benchmark/results --out benchmark/results/_index.md
```

跑完同一 `gelu`（easy）→ 表上直接比「claude/sonnet vs codex/gpt vs dsh/deepseek-v4
谁用更少 token 拿到 100 分」。

### D4 的契约必须写在流水线里，而不是只写在评分器里

D4 读结构化的 `bottleneck.json`。这个契约原先**只存在于 `scoring.py`**——
`plugins/ops/` 的产物清单里没有它，于是一个独立的 agent/harness 不可能知道要产出它，
结构化满分路径实际不可达，只能退化成关键词匹配。

现已把它写进 `plugins/ops/ops.yaml` 的产物契约（implement 的出口门禁 + 输出清单 +
prompt 里的字段枚举，optimize 负责写回 `convergence`）与
`agents/ops-implement-agent/agent.md` 的强制产物表。同时 D4 不再只验枚举成员资格：

- `classification` 要与该 case **实测确认过的**瓶颈集合相符（manifest 的
  `bottleneck_classes`，同义类都列入以免误伤措辞），取错扣 30 分；
- `convergence.reason = HARDWARE_LIMIT` 要与自报的带宽利用率自洽（`ops.yaml` 自己把
  硬件极限定义为 ≥95%），矛盾时扣 15 分。

这两条都来自实测反例：一个 run 在自报带宽利用率只有 34% 的情况下声称"已到硬件极限"，
另一个 run 一边写 `bandwidth_bound`、一边在 judgement 里说瓶颈是慢速 `tl.sum`
（即 `reduction_bound`）。只验格式的话它们都拿满分——那和关键词匹配是同一类错误。

## 当前状态

**当前只落地了 `ops` 一个领域**（有完整用例、评分、单测与真机基线）：

| 领域 | 用例 | 评分来源 | 实跑基线 |
| ------ | ------ | ---------- | :---: |
| `ops`（算子开发） | 36 | 自建五维评分（完成/性能/工程/收敛/成本） | ✅ 6 条 |
| `doc-agent`（资料工作流） | — | 待接入（见上文 TODO） | ❌ 未落地 |
| `inference-model-optimize`（推理优化） | — | 待接入（见上文 TODO） | ❌ 未落地 |

> 后两个域**故意不先摆脚手架**：能力基线的价值在于可信，一个没有单测、没有验证过的
> `scoring.py` 加上一条经不起追问的"基线"，比明确标 TODO 更糟。接入要点见上文 TODO。

数字以 `results/_index.md` 为准（该文件由 `report.py` 从 `result.json` 生成，不是手写的；
它同时会打印自己的**覆盖范围**，就是下表上方那段引用）。

> **覆盖范围：6 条基线全部来自同一个被测组合——DSH + `deepseek-v4`。**
> 其中 `relu` 是 `reference` 黄金回灌基线，**不是**"第二个 harness 跑了一遍"。
> 所以这张表只能读作**该组合的能力基线**，不能当成"哪个模型 / 哪个 harness 更强"的结论：
> Claude Code、Codex 以及其它模型的排列组合**一条数据都没有**，是后续 TODO
> （机制已就绪，按上文「换一个环境跑的具体步骤」补跑即可）。

6 条基线（全部 `ops`，全部真机实测）：

- `gelu` 100.00、`relu` 99.12（reference 回灌）、`sdpa` 95.68、`swiglu` 95.59、
  `matmul` 93.60、`rmsnorm` 75.05；
- **能力分水岭在归约**：easy（element-wise）≈ 99–100，hard（Cube/attention）93–96，
  medium（归约）75.05 —— `rmsnorm` 的失分几乎全在 D2 性能（`tl.sum` 归约 5–9× 慢于厂商核）。

⏳ 36 个 ops case 只实跑 6 个；**跨 harness / 跨模型补测尚未进行**（Claude Code 本机凭据
返回 403、Codex 未安装，见上文 TODO）；`doc-agent` / `inference-model-optimize` 两个域尚未接入；
跨 session 复现性扫描待补。

**已能实证回答两个问题**（均限定在上面那一个被测组合内）：

1. **能力基线随难度下降**：easy ≈ 99–100 → medium（归约）≈ 75；
2. **效果与成本分离可比**：`gelu` 效果最高但 286,674 token（效费比 348.8/百万），
   `rmsnorm` 效果更低但只用 118,506 token（效费比 633.3/百万）——单看效果分会得出"gelu 更好"，
   把成本并列出来才能看到这是两种不同的取舍。

### 已验证 / 未验证

说到哪一步就写到哪一步，避免把"跑过一次"说成"可靠"：

**已验证**

- 给定同一份证据，打分是**确定性**的：反复重跑分数一致；改 D4/D5 后 6 个 case
  重新打分，分数**一分未变**；
- 同一 run 内 3 次重复测量的 speedup std ≤ 0.014；
- 文档与本 README 里断言的每个数字都与 `result.json` 逐项比对过（0 处不符）；
- 50 项单测通过；`benchmark/` 目录 `ruff` 零告警（口径见下方复现命令：`E/W/N/F/B/I`
  全开、行长 88）——本仓其余目录的 Python 不在此次改动范围内，见「未验证」。
- 仓根文档门禁 `npm run lint:md` 全仓 **0 issues**（250 个 Markdown 文件）；
  表格分隔行统一写成 `| --- | --- |`——紧凑的 `|---|---|` 会被 MD060 判为违规，
  生成表格的 `report.py` / `sensitivity.py` 也按同一口径输出。

**未验证**

- **跨 session 的测量复现性只对 `relu` 做过**，其余 case 的 speedup 跨会话是否稳定尚未系统扫描；
- **换 harness / 换环境完全没有数据**——机制就绪（适配器契约 + 同一张表），
  但"换个环境分数会怎样"目前只是设计意图，不是实测结论；
- 36 个 case 只实跑 6 个，难度谱只覆盖到 3 档各若干；
- 权重敏感性只验证了**名次**稳定（6 种方案下 0 次翻转），未验证分数值本身的含义；
- **PEP 8 只在 `benchmark/` 目录内自查过**：本仓没有 Python lint 配置，CI（`ci/guard.py`）
  也不跑 lint，所以标准是这里写下的命令而不是一份可被工具自动读取的配置；`skills/` /
  `plugins/` / `ci/` 下的既有 Python 仍有一批行长超限（全仓 `E501` 约 1.9k 处），
  属于历史代码，未在本次改动范围内。

**复现命令**

```bash
# 打分（默认即写入 report.py 能收到的地方）+ 汇总
python -m benchmark.agents.ops.runner score outputs/gelu --case gelu
python -m benchmark.agents.ops.report --results benchmark/results --out benchmark/results/_index.md

# 自检
python -m benchmark.agents.ops.sensitivity          # 权重敏感性（输出即自检产物，不落库）
python -m pytest benchmark/agents/ops/tests -q      # 50 passed
python -m ruff check benchmark --select E,W,N,F,B,I --line-length 88   # 0 告警
npm run lint:md                                     # 仓根文档门禁，规则见 .doctools
```

各域的详细用法见 `agents/ops/README.md`。

## 语义学备注

- 「五维评分」的名称沿用 `plugins/ops/README.md` 里的既有说法，但一个关键修正：**成本维度（token/耗时）不混进效果分**，而是并列呈现——这是对「是 token 更少还是效果更好」的直接回答。
- 所有打分都从**落盘的证据**（`result.json` / `bench_result.json` / `kernel.py`）推导，不采信对话里的自述；证据缺失即扣分，从机制上鼓励诚实收敛（见 D4）和反作弊（见 D3）。
- **"没测到"不等于"满分"**：没有任何成本证据时 D5 记 `null`，不是 100。
- **记账必须与被计分的那一跳对齐**：本仓只收录能被 harness 留痕的成本。曾出现过把前一阶段的
  token 记账挂到后一阶段结果上的情况（doc 域 L0 → L1），使 D5 虚高到满分——这类张冠李戴
  比没有数据更糟，因为它在报告里看不出来。
- **只收真实可复现的数据，不留未经实测的脚手架**：宁可把某个域标成 TODO，也不先摆一个
  没有单测、没有验证过的 `scoring.py` 和"基线"。
