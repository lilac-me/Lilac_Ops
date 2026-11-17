# Doc-Agent 设计规格（DESIGN）

本文件是 doc-agent 重构的唯一真相源。所有实现（engine、linters、agents、skills、校验器）必须与本文件一致；冲突时以本文件为准。

## 1. 设计原则

1. **编排确定性**：Phase-Gate、循环、评分由引擎（状态机 + 门控谓词 + 账本）机器求值；Agent 只做单步"输入契约 → 输出契约"转换，不承担流程控制。
2. **契约 Schema 强制**：产物结构由 `contracts/*.json`（JSON Schema draft-07 子集）定义；Agent/Skill 只引用 Schema 名，不复述字段。
3. **规则目录**：审查规则集中在 `rules/rules.yaml`（rule_id + 维度 + 类型等级 + 扣分权重 + 主责维度 + 判定方式）；issue 必须绑定 rule_id。
4. **两层审查**：确定性 lint 层（脚本，永远先行）+ 语义审查层（LLM，按深度调度）；两层产出同一 issue 模型。
5. **深度分层**：L0 = 仅 lint；L1 = lint + 2 reviewer（correctness-code + completeness）；L2 = lint + 5 reviewer 全对抗。
6. **单一来源**：模板层只约束**仓库级必备资料清单**（`templates/repo_manifest.yaml`：必备 + 按仓库能力的条件项 + 用户诉求驱动的可选项），不约束每份资料的内部目录结构。清单与检测规则为**正式定义**（变更流程见 `templates/README.md`）。
7. **账本驱动**：语义 issue 在 `issues.jsonl` 中只追加，生命周期 open → fixed → verified → closed/disputed；确定性 issue 以当前 lint_report 为准。轮次由阶段账本查询得出，不靠 prompt。
8. **docs/en 全局出域**：英文资料树由专门翻译团队负责，整个 doc-agent 执行过程（上下文收集、完整性判定、全部 lint、语义审查）都不考虑 `docs/en/**` 下的内容。

## 2. 目录结构

```text
plugins/doc-agent/
├── DESIGN.md                    # 本文件
├── .claude-plugin/plugin.json   # Claude 适配层：注册 5 个 skill 路径
├── .codex-plugin/plugin.json    # Codex 适配层：注册插件内 skills 聚合目录
├── AGENTS.md                    # Codex 运行时根目录解析与角色调度约定
├── skills/                      # 安装脚本生成的 5 个 skill 链接/目录联接
├── engine/
│   ├── state_machine.py         # CLI: next / current / verify / score / doc-hash
│   ├── doc_pipeline.yaml        # 状态机定义（单一真相源）
│   ├── gates.py                 # 门控谓词：产物存在 + Schema 校验 + 评分阈值
│   ├── ledger.py                # issues.jsonl / ledger.jsonl 追加式读写与查询
│   └── yamlite.py               # 最小 YAML 子集加载器（无第三方依赖）
├── contracts/                   # JSON Schema（详见 §4）
│   ├── issue.json
│   ├── context_report.json
│   ├── design_plan.json
│   ├── figure_requests.json
│   ├── quality_report.json
│   ├── lint_report.json
│   ├── review_report.json
│   ├── code_verify.json
│   └── memory_event.json
├── rules/
│   ├── rules.yaml               # 规则目录（详见 §5）
│   └── README.md                # 规则治理：新增/废弃/版本化流程
├── linters/
│   ├── doc_lint.py              # 统一 CLI（详见 §6）
│   ├── link_lint.py             # 由旧 scan_links.py 改造（保留 xlsx 报告能力）
│   ├── manifest_lint.py         # 仓库必备资料 vs repo_manifest.yaml
│   ├── code_lint.py             # 代码块语法/引用静态检查
│   ├── placeholder_lint.py      # TODO/<...>/占位符
│   ├── term_lint.py             # 术语/错别字/简繁启发式
│   ├── image_lint.py            # 图片路径存在性 + 替代文本
│   ├── nav_lint.py              # README 导航可达性（尽力而为）
│   ├── placement_lint.py        # 自研仓资料存放位置 vs repo_manifest.yaml placement
│   ├── style_lint.py            # 段落长度/标题密度
│   ├── xref_lint.py             # 文档内交叉引用目标存在性
│   └── version_lint.py          # 版本标注启发式（尽力而为）
├── templates/
│   ├── repo_manifest.yaml       # 仓库级必备资料清单（正式定义 1.0.0）
│   └── README.md                # 待用户提供正式粗粒度标题集合的说明
├── agents/                      # 精简 agent.md（详见 §7）
└── .gitignore                   # 忽略 .doc-work/
```

工作产物目录：`.doc-work/<repo>/<run-id>/`（不污染目标仓；每份文档或审查分组一个 run）。参考实现在 `plugins/inference-model-optimize/engine/`。

## 3. 状态机

`engine/doc_pipeline.yaml` 定义（每条状态的 `transitions` 自上而下匹配，每条状态必有 `when: always` 兜底；带 `gates` 的转移门控失败且无 `fallback` 时继续尝试下一条转移）：

- 场景 A（新文档）：`init → context → design → confirm(人工确认) → write → review ⇄ fix(≤3) → finalize → done`
- 场景 B（审视已有文档）：`init → plan → review ⇄ fix(≤3) → finalize → done`

Handoff token（Agent 输出末尾，引擎 `extract_handoff` 取最后一条 `handoff: XXX` 行）：

| 状态 | 出边 | token |
| ------ | ------ | ------- |
| init | → context | `handoff: CONTEXT_READY` |
| init | → plan | `handoff: PLAN_READY` |
| context | → design | `handoff: DESIGN_READY` |
| design | → confirm | `handoff: DESIGN_DONE` |
| confirm | → write | `handoff: USER_APPROVED` |
| confirm | → design | `handoff: USER_REJECT` |
| write | → review | `handoff: REVIEW_READY` |
| plan | → review | `handoff: REVIEW_READY` |
| review | → fix | `handoff: NEED_FIX` |
| review | → finalize | `handoff: PASSED` |
| fix | → review | `handoff: FIXED` |
| fix | → finalize | `handoff: MANUAL_INTERVENTION` |
| finalize | → done | `handoff: DONE` |

**confirm 人工确认环节**（仅场景 A）：design 产出 design_plan.json（含 H2 章节清单）后进入 confirm；doc-orchestrator 向用户展示 H2 清单并等待确认——用户确认后写 `design_approval.md`（确认后的章节清单 + 确认说明 + 时间戳，写入 workspace）并推进 `USER_APPROVED`；用户要求修改则按意见修订 design_plan.json 后重新确认，或 `USER_REJECT` 回 design 重设计。`USER_APPROVED` 的门控强制要求 `design_approval.md` 存在，防止未确认流转。

门控谓词（`gates.py` 求值，YAML 中引用）：

- `product_exists`：产物文件存在且非空
- `schema_valid`：按 contracts 对应 Schema 校验通过
- `score_pass`：quality_report.json 契约合法，绑定当前文档 SHA-256 与 lint/review/issues/code 输入指纹，且 `verdict == "pass"`
- `lint_complete`：lint_report.json 为 completed，depth/目标文档/SHA-256 与质量报告一致
- `review_complete`：review_report.json 为 completed，且 depth 要求的 reviewer 集合全部完成
- `code_verify_present`：L1/L2 深度要求 code_verify.json 存在、契约合法且所有代码块状态为 `pass`（L0 豁免，按 quality_report.depth 判定）
- `iter_below_cap`：fix 计数 < 3

CLI（工作流先解析插件根绝对路径并记为 `<DOC_AGENT_ROOT>`：Claude Code 使用 `CLAUDE_PLUGIN_ROOT`；Codex 先取已加载的 `skills/doc-orchestrator/SKILL.md` 所在目录，再向上两级得到插件根目录；源码调试使用 `plugins/doc-agent`。解析结果必须包含 engine/rules/contracts 关键文件。先用 `new` 创建隔离 run，再使用 `next/current/verify/score/doc-hash/issue-status`；`score` 的 depth 缺省 L2）：

```bash
python "<DOC_AGENT_ROOT>/engine/state_machine.py" new \
  --workspace .doc-work/<repo> --doc <doc>
python "<DOC_AGENT_ROOT>/engine/state_machine.py" next \
  --workspace <run-workspace> --yaml "<DOC_AGENT_ROOT>/engine/doc_pipeline.yaml" \
  --handoff "handoff: CONTEXT_READY"
python "<DOC_AGENT_ROOT>/engine/state_machine.py" current --workspace <run-workspace>
python "<DOC_AGENT_ROOT>/engine/state_machine.py" verify --workspace <run-workspace>
python "<DOC_AGENT_ROOT>/engine/state_machine.py" doc-hash --workspace <run-workspace> --doc <doc>
python "<DOC_AGENT_ROOT>/engine/state_machine.py" score --workspace <run-workspace> \
  --rules "<DOC_AGENT_ROOT>/rules/rules.yaml" --depth L2 --doc <doc>
```

- 转移日志 `ledger.jsonl` 只追加；`verify` 重放校验每条转移都在 YAML 中声明。
- 迭代计数从 `ledger.jsonl` 推导，不做带外计数。
- 所有深度评分都必须能从 `--doc` 或 `new --doc` 得到目标文档；lint/review 报告必须绑定该文档当前 SHA-256。L1/L2 的 code_verify 还必须逐块绑定 fenced code block 的真实文件与起始行。

## 4. 契约（contracts/*.json）

统一约定：`"$schema"` 省略；required 数组强制关键字段；additionalProperties: false。

- `issue.json`：审查问题记录
  - `id: string`、`rule_id: string`（必填，必须存在于 rules.yaml）、`dimension: correctness|completeness|clarity|accessibility`、`severity: critical|major|minor`、`location: {file: string, line: integer?, anchor: string?}`、`evidence: string?`、`suggestion: string?`、`source: lint:<tool>|reviewer:<agent>`、`status: open|fixed|verified|closed|disputed`
- `context_report.json`：`repo: string`、`profile: string?`、`documents: string[]`、`version_baseline: string?`、`topics: string?`、`dependencies: string[]`、`capabilities: string?`、`missing: string[]`
- `design_plan.json`：`repo: string`、`doc: string`、`type_id: string`（repo_manifest.yaml 已登记条目 D-XXX）、`contribution_mode: ascend_dir|inline|none`、`required_sections: string[]?`（设计者自拟章节清单，不强制）、`optional_sections: string[]?`、`figure_baseline: [{no, section, purpose, elements, form, alt_text, status}]`、`rationale: string`
- `figure_requests.json`：`doc: string`、`items: [{no, section, purpose, elements, form, alt_text, status: pending|delivered}]`
- `lint_report.json`：`doc`、`doc_sha256`、`depth`、`status: completed|failed`、`issues`、`errors`；评分只接受 completed 且未过期的报告
- `review_report.json`：`doc`、`doc_sha256`、`depth`、`status`、`required_reviewers`、`completed_reviewers`、`errors`；集合必须精确匹配 depth
- `quality_report.json`：`repo`、`doc`、`doc_sha256`、`inputs_sha256`、`round`、`depth`、四维 scores、`issues`、`verdict`、`manual_reasons`；任一审查输入变化后旧报告失效，L1/L2 的 code_verify `manual`/`fail` 不得生成 `pass`
- `code_verify.json`：`doc`、`doc_sha256`、`blocks: [{location: {file, line}, tier, status, evidence, env?}]`；file/line 必须逐一对应目标文档 fenced code block 起始位置，evidence 非空且不得复用过期文档结果
- `memory_event.json`：`ts: string`、`type: phase_enter|gate_pass|gate_fail|issue_found|issue_fixed|issue_verified|checkpoint`、`phase: string?`、`issue_id: string?`、`payload: object?`

运行时主产物：`run.json`、`context_report.json`、`design_plan.json`、`draft_{doc}.md`、`figure_requests.json`、`lint_report.json`、`review_report.json`、`code_verify.json`、`quality_report.json`；`issues.jsonl` 只保存语义 issue 的追加式生命周期，确定性 issue 直接来自当前 lint_report；`ledger.jsonl` 为阶段账本。

## 5. 规则目录（rules/rules.yaml）

```yaml
scoring:
  weights: { correctness: 40, completeness: 25, clarity: 20, accessibility: 15 }
  floors:   { correctness: 90, completeness: 80, clarity: 60, accessibility: 60 }
  pass_total: 80
  max_rounds: 3
rules:
  - id: R-COR-001
    dimension: correctness
    title: 链接可达性
    kind: deterministic      # deterministic | semantic
    linter: link             # deterministic 时必填（对应 doc_lint 子命令）
    severity: critical       # 类型等级（投票取高基准）
    weight: 5                # 扣分权重
    primary: true            # 该对象主责维度（跨维度去重：非主责维度命中不重复扣分）
    judge: HTTP 2xx/3xx 视为有效；跨仓链接分支与目标仓由 R-COR-004 复核
    evidence: 请求状态码与目标 URL
```

规则清单（33 条，从早期实现统一后的检查项迁移，维度/权重/主责一一对应）：

- correctness：R-COR-001 链接有效(link,-5,primary)、R-COR-002 代码示例可运行静态(code,-15,primary)、R-COR-003 参数与实现一致(semantic,-10)、R-COR-004 版本与配套一致(version,-10)、R-COR-005 代码注释准确(semantic,-8)、R-COR-006 插图真实无伪造(semantic,-8,primary)、R-COR-007 语义与实现一致(semantic,-20,primary)、R-COR-008 数据口径准确(semantic,-20)、R-COR-009 代码示例验证闭环分级实测(semantic,-10,primary)
- completeness：R-COM-001 仓库必备资料齐全(manifest,-10,primary)、R-COM-002 前置条件完整(semantic,-10)、R-COM-003 成功标志明确(semantic,-5)、R-COM-004 错误处理覆盖(semantic,-8)、R-COM-005 交叉引用存在(xref,-5)、R-COM-006 无占位符(placeholder,-5,primary)、R-COM-007 版本信息标注(version,-5)、R-COM-008 插图诉求齐备(image,-5,primary)
- clarity：R-CLA-001 逻辑连贯(semantic,-5)、R-CLA-002 术语定义(semantic,-3)、R-CLA-003 段落长度(style,-1)、R-CLA-004 示例质量(semantic,-8)、R-CLA-005 标题密度(style,-3)、R-CLA-006 主动语态(semantic,-1)、R-CLA-007 受众匹配(semantic,-5)、R-CLA-008 错别字简繁(term,-2)
- accessibility：R-ACC-001 README 导航(nav,-10)、R-ACC-002 关键资料入口(semantic,-5)、R-ACC-003 跳转步数(nav,-5)、R-ACC-004 搜索友好(semantic,-3)、R-ACC-005 跨仓导航可达(crossrepo,-8；可达性主责在 R-COR-001，本规则只查导航路径不重复扣链接失效)、R-ACC-006 导航一致性(nav,-5)、R-ACC-007 图片路径有效(image,-3)、R-ACC-008 资料存放位置合规(placement,-5；自研仓强制 docs/zh/，三方仓不适用)

**同一问题判同**：`(rule_id, location.file, location.line)` 三元组相同即同一问题（line 缺失时用 anchor）。
**类型等级取高**：severity 为 critical 的规则命中即 critical，不因投票人数少降级；semantic 规则由 reviewer 自评等级，投票裁决：≥3 位独立发现 → critical、2 位 → major、1 位 → minor（专长维度可升级）。
**扣分**：dimension 得分 = 100 − Σ(该维度已确认 issue 的 weight)，下限 0；非主责维度命中同一对象只记录不扣分。
**语义发现必须映射 rule_id**；无匹配规则的发现登记为规则提案（进 `rules/README.md` 提案流程），不计分。

## 6. Linters（doc_lint.py）

统一 CLI（输出 lint_report.json；退出码 0 无 issue / 1 有 issue，二者均为 completed；2 表示执行失败或目标出域且不得评分；无第三方依赖）：

```bash
python "<DOC_AGENT_ROOT>/linters/doc_lint.py" lint \
  --repo <目标仓> --doc <文档路径> [--manifest "<DOC_AGENT_ROOT>/templates/repo_manifest.yaml"] \
  [--capabilities models,toolchain] [--include D-EVAL] [--exclude D-ACKS]
  [--rules "<DOC_AGENT_ROOT>/rules/rules.yaml"] [--depth L0] \
  --out <run-workspace>/lint_report.json
python "<DOC_AGENT_ROOT>/linters/doc_lint.py" rules
python "<DOC_AGENT_ROOT>/linters/doc_lint.py" link
```

`lint` 输出包含文档哈希、depth、completed/failed 状态、issues 与 errors。`docs/en/**` 目标在任何 linter 执行前直接生成无 issue 的 failed 报告并返回 2，交由翻译团队人工审核；lint 前后文档哈希不一致时同样 fail closed。任一声明的 linter 缺失或异常均产生 failed 报告并返回 2，不允许降级为空结果。lint 不做扣分与投票。

子模块分工：

- link：迁移旧 `scan_links.py`（保留 7 种 URL 提取、四分类、HEAD/GET 回退、xlsx 输出）；emit R-COR-001
- manifest：对照 repo_manifest.yaml，按仓库能力（`--capabilities models,toolchain`）与用户增删（`--include/--exclude`）计算必备清单，files glob（递归）或 H1/H2 标题匹配定位，缺失 emit R-COM-001
- code：提取代码块做 py_compile/shell -n 语法检查与目标仓符号引用静态检查；emit R-COR-002
- placeholder：TODO/`<...>`/未交付插图占位；emit R-COM-006
- term：错别字/简繁/术语启发式（基础词典即可）；emit R-CLA-008
- image：相对图片路径存在性 + 替代文本缺失；emit R-ACC-007、R-COM-008
- nav：README 入口 → 目标文档的可达步数（简单链接图 BFS）；emit R-ACC-001/003/006
- placement：自研仓（`--self-developed`，即 design_plan.contribution_mode == none）中文资料必须位于 docs/zh/，三档语义（docs/en 出域 / examples-tests-docker 零星 minor / 其余 major）与根文件豁免见 repo_manifest.yaml `placement`；emit R-ACC-008；三方仓不启用
- version/xref/crossrepo/style：best-effort；无法可靠实现时输出 `{"unsupported": true, "reason": ...}`，由语义层兜底，不得编造结果

## 7. Agents（10 个，精简定义）

agent.md 只含：角色定位（1 句）、输入/输出契约（引用 Schema 名）、操作摘要（≤10 步）、纪律（引用 skill/规则，不复述细则）。frontmatter 规范沿用仓库约定（name/description/version/domain/skills/tools/user-invocable/author）。

| Agent | skills | tools | 说明 |
| ------- | -------- | ------- | ------ |
| doc-orchestrator | doc-orchestrator | Read,Grep,Glob,Write,Edit,Bash,Agent | 场景识别、状态机驱动（调 engine CLI）、--depth 决策、修复循环节奏 |
| doc-designer | doc-design | Read,Grep,Glob,Write | 类型/变体/协作模式 → design_plan.json |
| doc-writer | doc-generate | Read,Write,Edit,Grep,Glob | 按骨架标题成文 + figure_requests.json；插图只登记不编造 |
| doc-quality-check | doc-quality-check | Read,Grep,Glob,Write,Bash,Agent | 跑 lint → 按 depth 调度 reviewer → rule_id 聚合投票 → quality_report.json |
| doc-updater | doc-update | Read,Edit,Write,Grep | 按账本 open issue 修复，遵守协作模式修改边界；不执行 lint 脚本 |
| correctness-code-reviewer | doc-quality-check | Read,Grep,Glob,Bash,WebFetch | L1/L2；复核 lint 客观项 + 代码块分级验证（T1 直接执行/T2 最小脚本/T3 静态核对，无环境 manual）回传 code_verify 结果块 + 版本/链接语义补充 |
| correctness-semantic-reviewer | doc-quality-check | Read,Grep,Glob,WebFetch | 语义/数据维度，发现映射 rule_id |
| completeness-reviewer | doc-quality-check | Read,Grep,Glob | 结构完整性（骨架章节 + 前置/成功标志/错误处理） |
| clarity-reviewer | doc-quality-check | Read,Grep | 易理解维度 |
| accessibility-reviewer | doc-quality-check | Read,Grep,Glob | 易获取维度 |

reviewer 通用纪律（写在各自 agent.md，统一措辞）：只输出 issue 清单（JSON，绑定 rule_id）作为最终消息回传 doc-quality-check，由其统一写账本；不写任何文件；不读取作者元注释；等级自评仅参考，最终等级由投票 + 类型等级取高决定。

> **角色/知识层边界（首版本规划，强制）**：Agent 定义只存在于 `agents/*/agent.md`，Skill 定义只存在于 `skills/doc-skills/**/SKILL.md` 及其 `references/*.md`——两个目录只允许 Markdown 提示词文件，**不得引入 .py/.yaml/.json 等任何代码形态**。角色身份、调度关系、知识内容永远不进代码；代码仅限 engine/linters/contracts/rules 等确定性基础设施（Agent 通过 Bash 调用，Skill 教 Agent 如何调用）。`tools/check_doc_agent_structure.py` 强制校验此边界。

## 8. Skills（5 个精简 knowledge pack）

`skills/doc-skills/` 下只保留：

- `common/doc-orchestrator`：引擎使用方法（CLI 命令、handoff、depth 决策、workspace 布局）
- `generation/doc-design`：类型/变体/协作模式匹配方法（迁移旧 template_matching 决策树与设计决策速查，去掉对旧模板资产的依赖）
- `generation/doc-generate`：写作规范（迁移旧 writing_guidelines：代码块纪律、术语、链接规范、插图纪律）
- `generation/doc-update`：修复策略（迁移旧 fix_strategy：等级处理、修改边界、验证）
- `review/doc-quality-check`：审查方法（迁移旧 scoring_rubric/quality_checklists 的语义判定部分与四维判例，引用 contracts/rules；references/ 下保留判例文件）

不再保留：doc-workflow（并入 engine 与 orchestrator skill）、doc-check-* 四个维度 skill（并入 doc-quality-check references）、`_tools/` 七个工具 skill（确定性能力并入 engine/linters/rules；doc-template-loader 的清单职责由 repo_manifest.yaml + doc-design 承担）。

Doc-Agent 支持 Claude Code 与 Codex：两个 manifest 都只注册安装脚本生成的插件内 `skills/` 聚合目录；仅 `doc-orchestrator` 为公开入口。Claude Code 使用 Agent 工具调度 `agents/*/agent.md`，Codex 由 doc-orchestrator 读取同一角色文件并使用宿主原生子任务能力；宿主无子任务能力时按相同契约串行执行。两个平台共享 engine/linters/contracts/rules/templates，不复制确定性实现。

## 9. 结构校验器与运行时自检

结构一致性校验器：`tools/check_doc_agent_structure.py`；运行时黑盒回归：`tools/check_doc_agent_runtime.py`。

校验项（无第三方依赖）：

1. 5 个 SKILL.md frontmatter（name/description/version/domain: doc/function 合法/author/user-invocable）+ 正文非空；只有 doc-orchestrator 可公开调用
2. 10 个 agent.md frontmatter + agent skills ⊆ plugin.json skills + tools 非空；只有 doc-orchestrator 可公开调用
3. Claude Code 与 Codex doc-agent manifest 的 skills 路径均限定在插件根内；安装脚本生成聚合链接后可解析
4. contracts/*.json 为合法 JSON 对象且含 `type/properties/required` 关键结构
5. rules.yaml 可解析、rule id 唯一、字段齐全、kind 为 deterministic 时有 linter 字段、scoring 阈值存在
6. repo_manifest.yaml 可解析、条目 id 唯一、when 条件合法、必备/可选分区存在
7. engine/*.py 与 linters/*.py `py_compile` 通过
8. 旧实现残留检查（skills/doc-skills/_tools、doc-check-* 不得存在）

## 10. 模板说明（重要）

`templates/repo_manifest.yaml` 为**正式定义**：仓库级必备资料清单（必备 14 条——其中 D-MODEL-SUPPORT/D-MODEL-DEV 始终必备、标题随 `repo_has_models` 自适应；D-TUNING/D-TOOLCHAIN 为能力条件项；可选 6 条由用户诉求动态增删）与检测规则（files glob 递归匹配 / H1-H2 标题匹配，heading 可为列表）。变更流程见 `templates/README.md`：更新清单后需同步 rules 中 R-COM-001 的引用与校验器。清单不约束每份资料的内部目录结构（由 doc-generate skill 的风格规范 + 目标仓实测取证原则覆盖）。
