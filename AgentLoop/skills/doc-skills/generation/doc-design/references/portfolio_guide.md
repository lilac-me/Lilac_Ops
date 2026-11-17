# 资料组合判定指引（portfolio_guide）

> 约束模型：仓库级必备资料清单（`<DOC_AGENT_ROOT>/templates/repo_manifest.yaml`）。只约束"仓库需要哪些资料"，不约束每份资料的内部目录结构；根目录由 doc-orchestrator 解析并传递。

## 判定顺序

1. 仓库形态：模型仓 / 加速库 / 适配器仓。
2. 仓库能力（决定条件资料与标题自适应）：
   - `repo_has_models`：仓库承载模型内容（模型仓必开；加速库/适配器仓按实际判定）；
   - `repo_has_toolchain`：仓库提供工具链相关能力。判定依据有二，两者互补：① 分析代码仓资料/代码（如 tools 目录、工具脚本、工具使用文档）；② 参考用户提供的输入 prompt 信息。
3. 目标仓性质（引导用户输入确认，不默认臆测）：昇腾自研开源仓，或参与业界主流开源社区的三方仓。
4. 三方仓协作模式：**仅目标仓为参与开源三方社区的三方仓时**判定（`ascend_dir` / `inline`）；昇腾自研开源仓不涉及协作模式，`contribution_mode` 记 `none`。
5. 组合计算：
   - 必备：manifest `required` 中无 `when` 条目全部适用；`when` 条目按仓库能力启用；
   - 可选：manifest `optional` 默认不强制，按用户输入诉求增删（`--include/--exclude` 语义）；
   - 用户诉求与 manifest 冲突时以用户诉求为准并记录理由。

## 标题自适应

两条必备资料的标题随仓库能力变化，条目本身始终必备：

| id | repo_has_models 为真时 | repo_has_models 为假时 |
| ---- | ---------------------- | ---------------------- |
| D-MODEL-SUPPORT | 支持模型/特性 | 支持特性 |
| D-MODEL-DEV | 模型/特性开发指导 | 特性开发指导 |

manifest 中以 `name`（无模型标题）+ `name_when_models`（有模型标题）登记；检测（heading 匹配）与问题报告使用对应能力的标题。

## 必备资料速览（14 条）

| id | 资料 | 条件 |
| ---- | ------ | ------ |
| D-INTRO | 简介 | — |
| D-NEWS | 最新消息 | — |
| D-RELEASE | 版本说明 | — |
| D-INSTALL | 安装指导 | — |
| D-QUICKSTART | 快速上手 | — |
| D-MODEL-SUPPORT | 支持特性 / 支持模型/特性（标题随能力，见上表） | 始终必备 |
| D-MODEL-DEV | 特性开发指导 / 模型/特性开发指导（标题随能力，见上表） | 始终必备 |
| D-TUNING | 模型性能调优指导 | repo_has_models |
| D-SECURITY | 安全声明 | — |
| D-DISCLAIMER | 免责声明 | — |
| D-LICENSE | License 声明 | — |
| D-CONTRIB | 贡献声明 | — |
| D-FAQ | FAQ | — |
| D-ACKS | 致谢 | — |

## 可选资料速览（默认清单，可按诉求动态增减）

| id | 资料 | 条件 |
| ---- | ------ | ------ |
| D-STRUCTURE | 目录结构 | — |
| D-TOOLCHAIN | 工具链使用指导 | repo_has_toolchain |
| D-ROADMAP | 未来规划 | — |
| D-JOIN | 加入我们 | — |
| D-INFER | 模型推理 | — |
| D-EVAL | 模型评测 | — |

## 章节清单（设计者自拟）

- 每份资料的内部章节由设计者按目标仓场景、目标受众与同仓惯例自拟，写入 design_plan.json 的 `required_sections` / `optional_sections`；无骨架约束。
- 章节自拟参考只取两样：目标仓已有同类资料惯例、目标受众的实际任务。**不提供统一的读者任务路径模板**——不同资料内部行文结构差异较大，统一路径参考会把行文逻辑带偏。
- 术语与链接规范见 doc-generate skill；完整性检查不校验章节级约束（R-COM-001 只查仓库必备资料齐备）。

## 三方仓协作模式（仅参与开源三方社区的三方仓适用）

- 前置：目标仓已确认参与业界主流开源社区（引导用户输入确认）；昇腾自研开源仓跳过本节，`contribution_mode` 记 `none`。
- `ascend_dir`：内容多、适配深、中文为主 → 昇腾目录（`ascend/`）内按现有资料形态补充。
- `inline`：内容少、纯说明、跟随社区语言 → 以章节单元为判定对象，遵守最小改动。

## 自研仓资料存放规则

- 昇腾自研开源仓（`contribution_mode: none`）的中文资料**强制存放在仓库根目录下的 `docs/zh/` 内**；检测由 lint `--self-developed` 开关驱动（R-ACC-008）。
- 三档语义：`docs/en/**` 出域（由翻译团队负责，不纳入 doc-agent 检查）；`examples/tests/docker` 零星文件合理（minor 提示）；根目录约定文件（README*/CONTRIBUTING/LICENSE/CHANGELOG 等）豁免（清单见 repo_manifest.yaml `placement` 段）。
