# 贡献指南

欢迎向 MindSpeed-Agent 贡献 skill！本文档将指导你完成从创建到合入的全流程。

## 快速开始

1. **Fork 本仓库** 到你的 GitCode 账号
2. **安装 [Claude Code](https://code.claude.com/docs/en/setup)**
3. **克隆你的 fork** 到本地
4. **本地安装插件**：

   ```bash
   claude plugin marketplace add ./MindSpeed-Agent
   claude plugin install mindspeed-agent@MindSpeed-Agent
   ```

5. **验证安装**：运行 `claude plugin list` 确认插件已安装，并在 Claude Code 中发起一个实际任务验证可用。

## Skill 开发流程

### 第一步：确定 skill 想法

在动手前，先回答以下问题：

- 这个 skill 解决什么问题？
- 用户会在什么场景下触发它？（越具体越好）
- 是否已有类似 skill？（查看 `skills/` 目录和 [README 中的技能列表](README.md)）
- 它属于哪个领域和功能类别？（参考下方分类指南）

### 第二步：创建 skill 目录

在 `skills/<domain>/<function>/` 下创建新目录，目录名就是 skill 名称。

**命名规则**：kebab-case，全小写字母 + 数字 + 连字符，3-40 字符。

**目录路径格式**：`skills/<domain>/<function>/<skill-name>/`

```text
skills/
└── <domain>/              # 领域（llm-model-skills / ops-skills / common 等）
    └── <function>/        # 功能类别（deployment / profiling / optimization / alignment）
        └── your-skill-name/
            └── SKILL.md   # 唯一必需文件
```

### 第三步：编写 SKILL.md

复制 `docs/skill-template/SKILL.md` 作为起点，或按以下模板：

```markdown
---
name: your-skill-name
description: "This skill should be used when the user asks to...（英文，第三人称）"
version: 1.0.0
domain: llm-model           # 所属领域
function: optimization      # 功能类别
tags:
  - mindspeed
  - ascend
author: your-github-username
---

# 技能名称

技能正文（简体中文），包含清晰的操作步骤和说明。
```

#### frontmatter 字段说明

| 字段 | 必填 | 说明 |
| ------ | :--: | ------ |
| `name` | ✅ | 唯一标识，kebab-case，与目录名一致 |
| `description` | ✅ | 英文，≤1024 字符，以 `"This skill should be used when..."` 开头 |
| `version` | 推荐 | 语义版本号，如 `"1.0.0"` |
| `domain` | 推荐 | 所属领域（见下方分类指南） |
| `function` | 推荐 | 功能类别（见下方分类指南） |
| `tags` | 推荐 | 搜索标签列表，最多 5 个 |
| `author` | 推荐 | GitHub 用户名或团队名 |
| `model` | 可选 | 指定模型，如 `"claude-sonnet-4-20250514"` |
| `allowed-tools` | 可选 | 限制可用工具，如 `"Read, Bash(git:*)"` |
| `metadata` | 可选 | 扩展元数据，如 `zh-description: "中文补充描述"` |

#### 正文编写建议

- **开头用 H1 标题**：与 skill 显示名一致
- **结构清晰**：使用 `## 功能概述`、`## 使用方式`、`## 操作步骤`、`## 注意事项` 等章节
- **步骤可操作**：给 Claude 明确的指令，而非模糊的建议
- **补充 why**：不仅说"做什么"，也说明"为什么这样做"
- **控制长度**：正文超过约 300 行时，考虑将详细参考资料外移到 `references/` 子目录

### 第四步：本地测试

1. 确保插件已本地安装
2. 在 Claude Code 中触发新 skill：
   - 显式调用：`/mindspeed-agent:<skill-name>`
   - 或构造一个匹配该 skill 描述的对话场景，看 Claude 是否自动匹配
3. 至少跑通一次完整的 skill 流程，确认输出符合预期

### 第五步：提交 PR

1. **分支命名**：`feat/<skill-name>` 或 `fix/<skill-name>`
2. **提交信息**：`feat(skills): add <skill-name> skill`
3. **PR 标题**：简洁描述，如 `feat(skills): add infer-awq-quant skill`
4. **PR 内容**：按 PR 模板填写 checklist
5. **Markdown 检查**：提交前运行 `npm run lint:md`（首次需先 `npm install`），确认 0 issues，详见下文「Markdown 语法检查」。

---

## 分类指南

Skill 按 **两级分类** 组织：领域（domain）+ 功能（function）。

### 领域（一级 — 决定 skill 放在哪个目录）

| 领域 | 目录 | 说明 | 什么 skill 放这里 |
| ------ | ------ | ------ | ------------------- |
| `llm-model` | `skills/llm-model-skills/` | LLM 模型迁移调优 | 大语言模型（GPT/LLaMA/Qwen 等）的训练迁移、混合精度、分布式训练 |
| `mm-model` | `skills/mm-model-skills/` | 多模态模型迁移调优 | 视觉-语言、音频-语言等多模态模型的 NPU 适配与优化 |
| `vla-model` | `skills/vla-model-skills/` | VLA / 世界模型适配优化 | 视觉-语言-动作模型与世界模型（含训练 pipeline）的 NPU 迁移、性能优化与交付 |
| `rl-model` | `skills/rl-model-skills/` | 强化学习模型迁移调优 | RL 模型（PPO/GRPO 等）的训练迁移与调优 |
| `inference-model` | `skills/inference-model-skills/` | 推理模型部署优化 | 推理管线搭建、量化压缩、服务化部署、性能剖析 |
| `ops` | `skills/ops-skills/` | 算子开发调优 | Ascend C / Triton / TileLang 算子开发、优化、精度调试 |
| `common` | `skills/common/` | 通用基础工具 | 环境检查、模型下载、日志解析等跨领域公共能力 |

### 功能类别（二级 — 决定 skill 放在领域的哪个子目录）

| 功能类别 | 目录 | 定位 | 典型 Skill |
| --------- | ------ | ------ | ------------ |
| `deployment` | `deployment/` | 环境搭建、服务部署 | 训练环境一键部署、推理服务搭建 |
| `model-analysis` | `model-analysis/` | 性能采集、瓶颈分析、模型结构分析 | 训练性能剖析、推理延迟基准测试 |
| `model-adaptation` | `model-adaptation/` | GPU→NPU 适配、格式转换、算子映射 | 模型格式转换、算子兼容性检查 |
| `vla-model-adaptation-optimization` | `vla-model-adaptation-optimization/` | VLA 类模型迁移适配与性能优化的合并流程 | VLA / 世界模型迁移与优化 |
| `alignment` | `alignment/` | 精度对齐、数值校验 | GPU→NPU 精度对比、LLM Judge 评估 |
| `batch-invariance-validation` | `batch-invariance-validation/` | 批次不变性验收、回归验证和异常定位 | 批次不变性验收、特性功能矩阵验证 |
| `performance-optimization` | `performance-optimization/` | 性能优化（核心） | 混合精度训练、AWQ 量化、KV Cache 调整 |
| `community-operations` | `community-operations/` | 案例分享、最佳实践沉淀、社区协作 | 案例模板、最佳实践文档生成 |
| `common` | `common/` | 跨领域公共能力 | 环境检查、模型下载、日志解析 |

---

## 提交规范

### 分支命名

```text
feat/<skill-name>      # 新 skill
fix/<skill-name>       # 修复现有 skill
docs/<topic>           # 文档变更
refactor/<scope>       # 重构
```

### 提交信息格式

```text
feat(skills): add <skill-name> skill
fix(skills): <skill-name> - <简要描述修复内容>
docs: update CONTRIBUTING.md
refactor(skills): <skill-name> - <重构内容>
```

提交信息使用英文，遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范。

---

## Markdown 语法检查（markdownlint）

仓库引入 [markdownlint-cli2](https://github.com/DavidAnson/markdownlint) 对全仓 Markdown（`SKILL.md`、`agent.md`、`references/*.md`、`README.md` 等）做语法检查。

```bash
npm install          # 首次：安装依赖（node_modules/ 已 gitignore）
npm run lint:md      # 检查全部 md，无输出即通过
npm run lint:md:fix  # 自动修复可安全自动修的问题（空行/尾随空格/末尾换行等）
```

- 配置在 `.markdownlint-cli2.jsonc`，已关闭与中文正文、`<domain>`/`<N>` 等占位符、ASCII 架构图、多阶段标题、模板冲突的排版/结构规则，只保留真语法检查（表格列数、链接正确性、代码块成对、空行包裹标题/列表/代码块等）。
- `--fix` 只处理可安全自动修的问题；表格列数不齐、链接失效、标题结构等问题需手工修复。
- 新增或改动 md 后，提交 PR 前先跑一次 `npm run lint:md`。

---

## 质量标准

### 优秀 skill 的特征

1. **触发场景明确**：description 清晰描述了用户会说/做什么来触发此 skill
2. **指令具体可操作**：步骤精确，命令可直接执行
3. **包含原理说明**：解释 why，帮助 Claude 理解上下文
4. **术语准确**：正确使用 MindSpeed/Ascend/CANN 生态的专有名词
5. **可测试**：贡献者至少手动跑过一次完整流程
6. **上下文高效**：内容精简，不占用不必要的 token

### 编写技巧

- **description 写得越具体越好**。对比：
  - ❌ `"This skill should be used when the user asks about MindSpeed."`
  - ✅ `"This skill should be used when the user asks to set up a CANN development environment, check CANN installation, or configure MindSpeed dependencies."`
- **给 Claude 决策依据**：不只说"执行 X"，而是说"当条件 A 满足时执行 X，否则执行 Y"
- **考虑边界情况**：常见错误如何处理？权限不足怎么办？

---

## 评审流程

1. **自动化检查（待 CI 上线）**：YAML 格式校验、分类合法性、命名规范
2. **人工评审**：
   - 至少 1 位领域专家（熟悉 skill 主题的人）
   - 至少 1 位平台专家（熟悉 Claude Code 插件系统的人）
3. **合入门槛**：至少 1 位评审者批准，且无未解决的阻塞性问题

评审者对照 [CLAUDE.md](CLAUDE.md) 中的评审清单进行检查。

---

## 许可

所有贡献遵循仓库的 MIT 许可证。提交 PR 即表示你同意将你的贡献以 MIT 许可发布。
