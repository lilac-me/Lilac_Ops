# 模板层说明（templates/）

## 约束模型：仓库级必备资料清单

`templates/repo_manifest.yaml` 是模板层的**唯一约束**：只约束"一个代码仓需要哪些资料"，**不约束每份资料的内部目录结构**——章节清单由 doc-designer 按目标仓场景自拟，内容由 doc-writer 按目标仓实测灵活撰写。

清单分三类：

| 类别 | 含义 | 启用方式 |
| ------ | ------ | --------- |
| `required` | 仓库必备资料（14 条） | 全部适用；其中 D-MODEL-SUPPORT/D-MODEL-DEV 标题随 `repo_has_models` 自适应 |
| `when` 条件项 | 按仓库能力决定是否必备 | `repo_has_models`（D-TUNING 模型性能调优指导）、`repo_has_toolchain`（D-TOOLCHAIN 工具链使用指导） |
| `optional` | 默认可选资料（6 条） | 由 doc-agent 使用者通过输入诉求动态增删（`--include/--exclude` 语义） |

## 当前状态：正式（1.1.0）

- **清单条目**：用户正式定义的仓库必备/可选资料盘点（必备 14 条：简介、最新消息、版本说明、安装指导、快速上手、支持模型/支持特性†、模型/特性开发指导†、模型性能调优指导*、安全声明、免责声明、License 声明、贡献声明、FAQ、致谢；可选 6 条：目录结构、工具链使用指导*、未来规划、加入我们、模型推理、模型评测。† = 始终必备、标题随 `repo_has_models` 自适应（无模型时称"支持特性"/"特性开发指导"）；* = 能力条件项：D-TUNING 仅 `repo_has_models` 时必备，D-TOOLCHAIN 仅 `repo_has_toolchain` 时必备）。
- **检测规则**：`files` 文件名 glob（相对仓库根、递归匹配任意深度，覆盖 docs/zh/** 深层布局）+ `heading` H1/H2 标题匹配（资料以 README 章节形态存在时）；两者满足其一即视为存在。已用 MindSpeed-LLM 仓实测校准（14 条必备全部命中）。
- **存放规则（`placement`）**：自研代码仓的中文资料强制存放在仓库根目录下 `docs/zh/` 内，三档语义：`out_of_scope`（`docs/en/**` 由翻译团队负责，不纳入检查）、`soft_scope`（examples/tests/docker 零星文件合理，minor 提示）、`exceptions`（README*/CONTRIBUTING/LICENSE 等约定根文件豁免）；由 lint `--self-developed` 开关驱动，对应规则 R-ACC-008，三方仓不适用。

## 变更流程

清单或检测规则需要调整时：

1. **更新 `repo_manifest.yaml`**：保持 `required/optional` 分区与条目 `id/name/files/heading/when/name_when_models` 结构；id 唯一；`when` 仅取 `repo_has_models` / `repo_has_toolchain`；`heading` 可为字符串或列表；文件头注释登记 `changed: <日期> <变更摘要>` 并递增版本号。
2. **同步 rules R-COM-001 引用**：核对 `rules/rules.yaml` 中 R-COM-001（仓库必备资料齐全）的判定说明与清单一致。
3. **同步结构校验器**：更新 `tools/check_doc_agent_structure.py` 中对 repo_manifest.yaml 的解析与数量断言。
4. **回归**：源码仓运行 `python tools/check_doc_agent_structure.py` 与 `python tools/check_doc_agent_runtime.py`。
