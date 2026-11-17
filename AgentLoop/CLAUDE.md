# MindSpeed-Agent

## 仓库目的

本仓库是 MindSpeed 系列套件的 Claude Code 插件，提供一组 skills 用于加速 MindSpeed / Ascend / CANN 平台的开发工作流。

## 生态图谱路由（每个会话先判定，再执行）

本仓库内置 **MindSpeed 生态知识图谱**（`catalog/`）：把社区代码仓、插件与
Skill 能力的关系固化为声明式数据（`repos.yaml` / `plugins.yaml` /
`task-kinds.yaml` / `intent-examples.yaml`）。遇到以下场景时**先路由、后执行**：

- 用户提到社区代码仓或模型名（MindSpeed / MindSpeed-LLM / ModelLink /
  MindSpeed-MM / MindSpeed-Bridge / vllm / vllm-ascend / torch-npu /
  triton-ascend / CANN 等），但未指明具体 skill/插件；
- 用户问"用哪个仓库/插件/方法做 XX"、"怎么优化/迁移/适配/部署/验证 XX"；
- 需要把一句自然语言拆成 `repo → plugin → agent → skill` 调用链，或需要
  区分训练框架与推理引擎（如 MindSpeed-LLM vs vllm-ascend）。

路由方式：加载通用插件中的 `ecosystem-routing` skill（`skills/common/common/
ecosystem-routing/`），按其 R1–R5 规则执行——先读 `catalog/repos.yaml` 做仓库
确定性消歧 → 匹配任务维度（`task-kinds.yaml`）→ 组装候选链
（`plugins.yaml`）→ 歧义时反问确认，**绝不静默猜**。路由只负责指向与确认，
不代行目标插件/流水线的流程。

- Claude Code / Codex / DSH 均通过各自插件清单加载同一 `ecosystem-routing`
  skill，无需各平台重复实现；
- catalog 数据维护与校验见 `catalog/README.md`；CI 侧由 `ci/guard.py`
  （`ci/checks/knowledge_graph.py`）兜底。

---

## 仓库约定

### 架构总览

本仓库遵循 **三层架构**：Plugin（应用编排层）→ Agent（角色执行层）→ Skill（知识能力层）。

- **Plugin**：通过 `.claude-plugin/plugin.json` 注册，定义 Agent 协作顺序
- **Agent**：承担方案设计、代码开发、代码检视、代码测试等角色职责，负责将用户需求分解为 Skill 调用链
- **Skill**：最小知识单元，由 `SKILL.md` 定义，提供领域知识与工程模板

### 文件命名规则

- **skill 目录路径**：`skills/<domain>/<function>/<skill-name>/`，如 `skills/inference-model-skills/optimization/awq-quant/`
- **skill 目录名**：全小写 kebab-case，仅含 `[a-z0-9-]`，3-40 字符。示例：`awq-quant`、`perf-baseline`
- **领域/功能目录名**：固定使用预定义名称（见领域与功能分类体系），不可自定义
- **禁止**：前导/尾随连字符、连续连字符（`--`）、空格、大写字母
- **辅助文件**（`references/`、`scripts/`、`assets/`）：清晰描述性命名，与 skill 内引用一致

### SKILL.md 完整 Schema

每个 `SKILL.md` 由 YAML frontmatter + Markdown 正文组成。

#### 必填字段

| 字段 | 类型 | 说明 |
| ------ | ------ | ------ |
| `name` | string | skill 唯一标识，kebab-case，与目录名一致 |
| `description` | string | 英文，第三人称，≤1024 字符，格式：`"This skill should be used when the user asks to..."` |

#### 推荐字段

| 字段 | 类型 | 说明 |
| ------ | ------ | ------ |
| `version` | string | 语义版本号，如 `"1.0.0"` |
| `domain` | string | 所属领域（见下文领域体系） |
| `function` | string | 功能类别：`deployment` / `model-analysis` / `model-adaptation` / `vla-model-adaptation-optimization` / `alignment` / `performance-optimization` / `community-operations` / `common`（`doc` 领域取值见功能类别表中标注（doc）的行） |
| `tags` | string[] | 搜索/过滤标签，最多 5 个 |
| `author` | string | GitHub 用户名或团队名 |

#### 可选字段（Claude Code 原生支持）

| 字段 | 类型 | 说明 |
| ------ | ------ | ------ |
| `model` | string | 指定模型，如 `"sonnet"`、`"opus"` |
| `context` | string | 独立上下文模式，如 `"fork"` |
| `agent` | string | 指定 agent 类型，如 `"Explore"` |
| `allowed-tools` | string | 限制可用工具，如 `"Read, Bash(git:*)"` |
| `user-invocable` | boolean | 是否允许用户直接调用，默认 `true` |
| `hooks` | object | 工具钩子配置 |
| `license` | string | skill 许可证，如 `"MIT"` |
| `metadata` | object | 任意键值对扩展元数据 |
| `compatibility` | string | 兼容性声明，≤500 字符，如 `"MindSpeed >= 1.0.0"` |

### 领域与功能分类体系

Skill 按 **两级分类** 组织：领域（domain）+ 功能（function）。`SKILL.md` 的 frontmatter 中 `domain` 和 `function` 字段分别对应。

#### 领域（一级）

| 领域 | 目录 | 说明 |
| ------ | ------ | ------ |
| `llm-model` | `skills/llm-model-skills/` | 大语言模型（GPT/LLaMA/Qwen 等）的 NPU 训练迁移、调优、精度对齐 |
| `mm-model` | `skills/mm-model-skills/` | 多模态大模型（视觉-语言、音频-语言）的 NPU 适配与优化 |
| `vla-model` | `skills/vla-model-skills/` | 视觉-语言-动作（VLA）与世界模型的 NPU 适配与优化 |
| `rl-model` | `skills/rl-model-skills/` | 强化学习模型（PPO/GRPO 等）的训练迁移与调优 |
| `inference-model` | `skills/inference-model-skills/` | 推理模型的端到端部署、性能剖析、量化压缩、服务化 |
| `ops` | `skills/ops-skills/` | 算子开发调优（Ascend C / Triton / TileLang） |
| `common` | `skills/common/` | 跨领域的通用基础能力（环境检查、模型下载等） |
| `doc` | `skills/doc-skills/` | 资料插件体系（文档生成、四维质量审查，确定性引擎驱动） |
| `ecosystem-code-repository-npu-migration` | `skills/ecosystem-code-repository-npu-migration/` | 仓库级 GPU/PyTorch → 昇腾 NPU 迁移（插件 ecosystem-code-repository-npu-migration，Claude Code 与 Codex 双平台） |

#### 功能类别（二级）

| 功能类别 | 目录 | 定位 | 典型 Skill |
| --------- | ------ | ------ | ------------ |
| `deployment` | `deployment/` | 环境搭建、服务部署、配置管理 | 训练环境一键部署、推理服务搭建 |
| `model-analysis` | `model-analysis/` | 性能采集、瓶颈分析、资源监控、模型结构分析 | 训练性能剖析、推理延迟基准测试 |
| `model-adaptation` | `model-adaptation/` | GPU→NPU 模型迁移适配、格式转换、算子映射 | 模型格式转换、算子兼容性检查 |
| `vla-model-adaptation-optimization` | `vla-model-adaptation-optimization/` | VLA 类模型迁移适配与性能优化的合并流程 | VLA / 世界模型迁移与优化 |
| `alignment` | `alignment/` | 精度对齐、数值校验 | GPU→NPU 精度对比、LLM Judge 评估 |
| `batch-invariance-validation` | `batch-invariance-validation/` | 批次不变性验收、回归验证和异常定位 | 批次不变性验收、特性功能矩阵验证 |
| `performance-optimization` | `performance-optimization/` | 训练/推理性能优化、量化压缩、算子融合 | 混合精度训练、AWQ 量化、KV Cache 调整 |
| `community-operations` | `community-operations/` | 使用案例分享、最佳实践沉淀、社区协作 | 案例模板、最佳实践文档生成 |
| `common` | `common/` | 跨领域公共能力 | 环境检查、模型下载、日志解析 |
| （算子专用） | `ascendc/` `triton/` `tilelang/` | 算子开发与调优 | Ascend C 算子生成、Triton 迁移 |
| `common`（doc） | `skills/doc-skills/common/` | 资料工作流入口与引擎调度方法 | doc-orchestrator |
| `generation`（doc） | `skills/doc-skills/generation/` | 文档生成（组合设计、撰写、更新） | doc-design, doc-generate, doc-update |
| `review`（doc） | `skills/doc-skills/review/` | 文档四维质量审查（正确性/完整性/易理解/易获取） | doc-quality-check |

> `doc` 领域 Skill 的 frontmatter `function` 取值仅限上表标注（doc）的三个类别。确定性基础设施（状态机引擎、lint 层、规则目录、产物契约、仓库级资料清单）位于 `plugins/doc-agent/` 下，不属于 skill。
>
> `ecosystem-code-repository-npu-migration` 领域 Skill 的 frontmatter `function` 取值为标准功能类别（`model-adaptation` / `model-analysis` / `deployment` / `common`）。正本位于根 `skills/ecosystem-code-repository-npu-migration/`，经 `plugins/ecosystem-code-repository-npu-migration/skills/` 下由安装脚本生成的软链接进入插件视图，同时供 `.claude-plugin/plugin.json`（Claude Code）与 `.codex-plugin/plugin.json`（Codex）两份清单消费。

### i18n（国际化）规范

- **frontmatter `description`**：英文，遵循 Anthropic 规范 `"This skill should be used when..."`
- **正文内容**：简体中文（步骤说明、操作指引等面向用户的文字）
- **代码注释**：可保留英文；专业术语（模型名、API 名）保持原文不翻译
- **禁止中英混杂**：同一段落内避免中英交替
- **术语一致性**：在整个仓库中统一使用：
  - "昇腾"（非"Ascend"）在中文正文中
  - "NPU"（非"npu"）专有名词
  - "CANN" 保持大写
  - "MindSpeed" 保持 PascalCase

---

## 技能评审清单

审查 skill PR 时，按以下清单逐项检查：

### [阻塞] 必须通过（否则拒绝合并）

1. **frontmatter 完整性**：`name` 和 `description` 存在且格式正确
2. **命名规范**：目录名 kebab-case，与 `name` 字段一致，3-40 字符
3. **描述规范**：英文，以 `"This skill should be used when"` 开头
4. **内容完整性**：SKILL.md 正文非空，提供有意义且可操作的指令
5. **无恶意指令**：skill 未指示执行危险操作（`rm -rf`、权限提升、数据外泄等）
6. **无功能重复**：不与现有 skill 明显重叠；如有重叠需说明差异化定位
7. **自包含**：SKILL.md/agent.md 不引用仓外基准文档（设计规格书、Roadmap、模板体系设计报告等）

### [建议] 尽量满足（评审者标记，不阻塞合并）

1. **分类合理**：`domain` 字段为领域之一，`function` 字段为功能类别之一
2. **描述质量**：描述清晰具体，包含典型触发场景
3. **正文结构**：有清晰的标题层次、步骤列表，可操作性强
4. **上下文效率**：正文简洁，避免过度冗长；复杂 skill 使用 `references/` 按需引用
5. **跨平台兼容**：涉及命令时考虑 Windows/macOS/Linux 差异
6. **i18n 一致**：正文语言统一，术语使用正确

---

## PR 评审流程

1. 阅读本文件（CLAUDE.md）确认评审标准
2. 对每个 `skills/<domain>/<function>/<skill-name>/SKILL.md` 执行：
   - 解析 YAML frontmatter，校验基本结构
   - 对照 Schema 检查字段完整性（`domain` 和 `function` 需为预定义值）
   - 对照评审清单逐项标记通过/阻塞/建议
3. 确认 skill 放在正确的领域和功能目录下
4. 输出结构化评审意见：阻塞性问题列表 + 建议列表

---

## 技能数量上限

**Claude Code 每个会话最多加载 30 个 skill**（截至 2025-12），超出部分被静默忽略。本仓库采用多插件架构（每个领域 Agent 持有独立的 `.claude-plugin`），每个插件独立计算 skill 数量上限。因此：

- 主插件 `mindspeed-agent` 仅挂载通用 skills
- 领域专用 skills 挂载在对应的 Agent 插件下，按需加载
- 每个插件内的 skills 总数 ≤ 30
- 新 skill 合入前需评估是否可合并到现有通用 skill 中
- 优先通用性强的 skill，避免过度碎片化

---

## 插件维护

更新 `plugin.json` 时：

- 新 skill 合入后递增 `version`（semver：新增 skill → minor，修复 skill → patch）
- 同步更新 `keywords`（如有新增技术领域）

---

## CLAUDE.md 更新

本文件的变更（新增分类、修改 Schema、调整评审标准）需通过独立 PR 提交。
