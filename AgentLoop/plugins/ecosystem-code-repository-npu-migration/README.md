# ecosystem-code-repository-npu-migration

`ecosystem-code-repository-npu-migration` 是一个以提示词为先、支持 Claude Code 与 Codex 双平台的昇腾 NPU 迁移插件。

## 来源与版本

- 上游仓库：`https://github.com/addsubmuldiv/ecosystem-code-repository-npu-migration.git`
- 导入基线：`codex/npu-codex-skill-only@4c3a7159e279fb3faf6c9d5e613053abc463d8a6`
- 上游快照版本为 `0.15.0`；MindSpeed-Agent 归档版本为 `0.15.1`。本目录保存源码快照，不包含上游 Git 历史。
- `0.16.0`：skill 正本迁移至仓库根 `skills/ecosystem-code-repository-npu-migration/` 并升级为仓库规范富 frontmatter，插件目录改为软链接视图（详见 [架构文档](../../docs/ecosystem-code-repository-npu-migration.md)）。

## 目录结构

- `.claude-plugin/plugin.json`：Claude Code 插件清单（skills 显式数组）。
- `.codex-plugin/plugin.json`：Codex 插件清单。
- `skills/`：8 个 skill 的扁平软链接视图（由 `install.sh` / `install.bat` / `ensure_skill_links()` 生成，不入 git）。正本位于仓库根 `skills/ecosystem-code-repository-npu-migration/<function>/<skill-name>/`；`goal` 是仓库级入口，其余是可独立调用的专项 skill。
- `references/`：专项 skill 路由、共享 NPU 迁移知识和可选语义产物边界。skill 内引用这些资源使用 `<plugin-root>/...` 锚点（软链接下 `../../` 会解析到 canonical 路径的父级，不可用）。
- `scripts/`：插件打包、版本同步、旧版安装迁移和 skill 设计校验脚本。
- `bundle.lock.json`：插件清单、skill 集合和发布文件摘要。
- `tests/`：提示词、路径、bundle 和校验脚本契约；`conftest.py` 会在导入期自动重建软链接视图。

主 Agent 决定工作顺序、整合源码并判断最终支持范围；仓库扫描、官方资料研究、单目标 repair，以及证据稳定后的写作，可按需委派给职责边界明确的 subagent。共享环境、源码和真实 NPU 命令默认由单一写入者负责；主 Agent 仍须依据当前 checkout 和直接证据复核所有返回结果。

初步适配和具体失败修复会按需查询 `references/migration_knowledge/` 与当前官方文档，但必须用目标仓库源码和真实命令验证。插件不包含 JavaScript 工作流、Python 阶段控制器、执行队列或自动重试状态机。

## 使用

### 前置条件

- 已安装 Claude Code CLI（`claude` 命令）或 Codex CLI（`codex` 命令），任选其一或两者皆有
- 目标仓库已完成 clone，当前工作目录在目标仓库内（或明确给出目标仓库路径）

### 安装

```bash
# 方式一：一键脚本（自动创建软链接/junction 并安装插件）
bash install.sh claude-code    # Unix，Claude Code
install.bat claude-code        # Windows，Claude Code
bash install.sh codex          # Unix，Codex
install.bat codex              # Windows，Codex

# 方式二：手动安装
claude plugin marketplace add /path/to/MindSpeed-Agent
claude plugin install ecosystem-code-repository-npu-migration@MindSpeed-Agent

codex plugin marketplace add /path/to/MindSpeed-Agent
codex plugin add ecosystem-code-repository-npu-migration@MindSpeed-Agent
```

### 触发方式

在 Claude Code 或 Codex 会话中任选其一：

```text
$ecosystem-code-repository-npu-migration:goal 把 <目标仓库> 迁移到昇腾 NPU
```

或直接用自然语言描述：

```text
把这个仓库迁移到昇腾 NPU 上跑通
```

`goal` 是仓库级入口，会自适应编排其余 specialist；也可以按需直接调用专项 skill：

| 命令 | 用途 |
| ------ | ------ |
| `$ecosystem-code-repository-npu-migration:feature-model` | 建立仓库功能与可执行入口图，枚举范围内测试/示例 |
| `$ecosystem-code-repository-npu-migration:local-assets` | 发现/校验本地模型或数据集资产（不下载新内容） |
| `$ecosystem-code-repository-npu-migration:env-setup` | 研究 CANN/torch/torch_npu/Python 兼容性并搭建环境 |
| `$ecosystem-code-repository-npu-migration:repair` | 诊断并修复单个失败的 NPU 单元/组件/命令 |
| `$ecosystem-code-repository-npu-migration:vllm-ascend-env` | 在健康昇腾环境内安装/修复/验证 vLLM + vLLM-Ascend |
| `$ecosystem-code-repository-npu-migration:report-pr` | 把命令、日志和证据转为保守工程报告或 PR 文本 |
| `$ecosystem-code-repository-npu-migration:docs-support` | 从已接受证据创建/更新仓库 NPU 支持文档 |

### 典型流程走查

一次完整的仓库级迁移通常按以下次序推进（由 `goal` 协调，无需手工逐步调用）：

1. **盘点**：`feature-model` 建功能地图 → `local-assets` 确认可用模型/数据集
2. **环境**：`env-setup` 研究版本兼容性并搭建 CANN/torch_npu 环境
3. **适配**：主 Agent 修改源码，按需查询迁移知识与官方文档；失败时 `repair` 做诊断与最小修复
4. **验证**：跑通范围内测试与代表性示例；需要服务化时 `vllm-ascend-env` 验证 vLLM
5. **收口**：`report-pr` 汇总证据出报告/PR，`docs-support` 更新仓库文档

任务可在所有当前可行工作完成后带明确 blocker 结束；只有全部请求内 capability checks 和 representative goals 通过时才声明完整支持。

### 能力边界

- 不做自动重试队列、断点恢复或流程状态机；`.npu-agent/` 下只有轻量工作笔记
- 不静默下载大型资产；不修改目标仓库中未要求修改的版本路径
- 结构/提示词/路径校验不等于迁移成功——支持结论只来自真实 NPU 执行证据

## 外部知识根目录

`goal` 和 `repair` 可在单次任务中接入一个外部 Markdown 知识库目录：

```text
external_knowledge_root: /absolute/path/to/kb
```

也可以使用命令式写法：

```text
--knowledge-root /absolute/path/to/kb
```

目录必须是可读的绝对路径；插件会递归发现 `.md` 和 `.markdown`，并将其
作为内置迁移知识的补充。可选的根目录 `index.md` 只用于路由提示。发现
结果只输出到当前任务，不复制外部文件、不建立向量索引，也不写入仓库级
配置；无效或空目录会报告 `BLOCKED_EXTERNAL_KNOWLEDGE`。外部资料仍须用
当前源码、安装版本、官方文档和真实 NPU 命令验证。

`.npu-agent` 还可以保存仓库级 goal 使用的
`migration_progress.md` 和 `compatibility.json` 两份轻量工作记录。它们只
用于在 context compaction 后帮助主 Agent 回看进展，不是断点恢复文件、工作
流状态机或支持结论来源；默认不提交业务仓库，也不能代替真实 NPU 执行证据。

`migration_progress.md` 简要记录目标、已完成工作、当前重点、剩余工作、阻塞
项和关键决定。`compatibility.json` 按已有 entry 或 group 记录命令、日志、结论
和风险。主 `goal` Agent 在主要里程碑后更新它们，`repair` 和子代理只返回事实，
不直接写入。文件缺失时可以重新创建，不需要额外的 schema、校验器、历史记录
或恢复控制逻辑。最小写法可参考 [migration_progress.md 示例](../../skills/ecosystem-code-repository-npu-migration/model-adaptation/goal/examples/migration_progress.md)
和 [compatibility.json 示例](../../skills/ecosystem-code-repository-npu-migration/model-adaptation/goal/examples/compatibility.json)，示例中的路径、状态和结果都必须替换为当前任务事实。

知识查询结果默认只留在当前任务中，不生成独立知识快照或查询产物。

最终报告分别给出任务处置与迁移结果。任务可在所有当前可行工作完成后带明确 blocker 结束，但只有全部请求内 capability checks 和 representative goals 通过时才能声明完整支持。

## 发布校验

以下命令从 MindSpeed-Agent 仓库根目录执行：

```bash
python plugins/ecosystem-code-repository-npu-migration/scripts/build_bundle_lock.py --check
python plugins/ecosystem-code-repository-npu-migration/scripts/validate_plugin_package.py --json
python plugins/ecosystem-code-repository-npu-migration/scripts/validate_bundle.py --json
python plugins/ecosystem-code-repository-npu-migration/scripts/validate_skill_design.py --json
```

校验器会自动重建软链接视图，fresh clone 上可直接运行（Windows 无符号链接权限时请先运行 `install.bat`）。

`plugins/ecosystem-code-repository-npu-migration/scripts/set_plugin_version.py <semver>` 统一更新 Claude/Codex 两份插件清单、两个 marketplace 和 bundle lock；执行后需重跑上述全部校验命令。
