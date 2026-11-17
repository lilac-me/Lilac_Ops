# ecosystem-code-repository-npu-migration 插件架构说明

> 适用版本：ecosystem-code-repository-npu-migration 0.16.0（bundle lock schema v3）。本文档描述插件的架构定位、目录布局、skill 协作、知识分层与发布校验；使用指引见[插件 README](../plugins/ecosystem-code-repository-npu-migration/README.md)。

## 1. 定位与三层架构

`ecosystem-code-repository-npu-migration` 是 MindSpeed-Agent 仓库中支持 **Claude Code 与 Codex 双平台** 的子插件，提供仓库级 GPU/PyTorch → 昇腾 NPU 迁移能力。它以提示词为先：不包含 JavaScript workflow、Python 阶段控制器、执行队列或自动重试状态机，与 `inference-model-optimize`（YAML + JS 状态机驱动）是两种不同的编排范式。

按仓库三层架构的对应关系：

| 层 | ecosystem-code-repository-npu-migration 的实现 |
| ---- | -------------------- |
| Plugin 层（应用编排） | `plugins/ecosystem-code-repository-npu-migration/`：`.claude-plugin/plugin.json` + `.codex-plugin/plugin.json` 双清单、`bundle.lock.json` 完整性锁、`references/` 共享知识、`scripts/` 工程脚本 |
| Agent 层（角色执行） | 主 Agent（`$ecosystem-code-repository-npu-migration:goal`）自适应编排，按需有界委派 subagent；无固定 Agent 角色文件 |
| Skill 层（知识能力） | 8 个 skill，正本位于仓库根 `skills/ecosystem-code-repository-npu-migration/`，经插件内软链接进入双平台插件视图 |

主 Agent 负责理解目标仓库、维护迁移计划、整合源码、执行真实命令并判断最终支持范围；扫描、官方资料研究、单目标修复、证据稳定后的写作等可委派给职责边界明确的 subagent。所有返回结果仍须由主 Agent 依据当前 checkout 和直接证据复核。

## 2. 目录与软链接布局

### 2.1 正本（canonical，入 git）

```text
skills/ecosystem-code-repository-npu-migration/                  # ecosystem-code-repository-npu-migration 领域
├── model-adaptation/                         # GPU→NPU 适配与修复
│   ├── goal/                                 # 仓库级迁移协调入口
│   └── repair/                               # 单目标失败诊断与修复
├── model-analysis/
│   └── feature-model/                        # 仓库功能地图与覆盖计划
├── deployment/
│   ├── env-setup/                            # CANN/torch/torch_npu 环境
│   └── vllm-ascend-env/                      # vLLM + vLLM-Ascend 安装验证
└── common/
    ├── local-assets/                         # 本地模型/数据集资产发现
    ├── report-pr/                            # 证据 → 报告/PR 文本
    └── docs-support/                         # 仓库 NPU 支持文档
```

### 2.2 插件视图（软链接，不入 git）

```text
plugins/ecosystem-code-repository-npu-migration/
├── .claude-plugin/plugin.json   # Claude 清单：skills 为显式数组
├── .codex-plugin/plugin.json    # Codex 清单："skills": "./skills/"，扫描整个视图目录
├── skills/                      # 恰好 8 个软链接，名字保持扁平
│   ├── goal -> ../../../skills/ecosystem-code-repository-npu-migration/model-adaptation/goal
│   └── ...
├── references/                  # 插件级共享知识（不随 skill 移动）
├── scripts/                     # 打包、校验、版本、安装迁移脚本
├── tests/                       # 10+ 个契约测试（conftest 自动建链接）
├── bundle.lock.json             # 发布文件哈希锁（schema v3）
├── AGENTS.md / README.md
```

链接由三处生成/重建，语义一致：

1. `install.sh`（Unix，`ln -s`）/ `install.bat`（Windows，`mklink /J` junction）
2. `plugins/ecosystem-code-repository-npu-migration/scripts/plugin_bundle.py` 的 `ensure_skill_links()`——校验器与测试自动调用的幂等兜底
3. `plugin_bundle.SKILL_FUNCTION` 常量是 Python 侧链接表的单一来源；install.sh / install.bat 各持有一份镜像列表

Windows 安装时，`install.bat` 先把 junction 物化成真实目录拷贝（stage），`codex plugin add` 完成后再恢复 junction（restore）——因为 Codex 缓存复制不跟随 junction。doc-agent 已有同构实现。

### 2.3 路径解析规则（重要）

skill 正本位于仓库根 `skills/` 下，但 skill 引用插件级 `references/`、`scripts/` 资源时**必须使用 `<plugin-root>/...` 锚点**，不得使用 `../../` 相对路径：

- 符号链接路径下的 `../../` 按链接目标（canonical 路径）解析父级，而不是链接所在位置。例如 `plugins/ecosystem-code-repository-npu-migration/skills/goal/../../references/` 实际解析到 `skills/ecosystem-code-repository-npu-migration/references/`（不存在），而非 `plugins/ecosystem-code-repository-npu-migration/references/`。
- `<plugin-root>` 由运行时的 Codex Agent 按安装后的插件根解析，与目录布局无关。插件内已有先例：`skills/goal/references/validation_guide.md`、`references/external_knowledge_policy.md`。

`validate_skill_design.py` 会识别 `<plugin-root>/` 锚点并校验目标存在；漏改的 `../../` 引用会直接导致校验 BLOCKED。

### 2.4 消费边界

`skills/ecosystem-code-repository-npu-migration/` 是源码真相（canonical）。**运行态消费一律经插件视图**：Claude 清单显式列出 `plugins/ecosystem-code-repository-npu-migration/skills/<name>`，Codex 清单扫描整个 `plugins/ecosystem-code-repository-npu-migration/skills/`；skill 里的 `<plugin-root>` 锚点也只在插件上下文内有定义。直接从根目录引用这些 skill 属于未定义行为。

## 3. Skill 职责与协作

`goal` 是唯一的仓库级入口，其余 7 个是可独立调用的 specialist。路由规则见 `plugins/ecosystem-code-repository-npu-migration/references/skill_routing.md`。

典型迁移次序：

```text
$ecosystem-code-repository-npu-migration:goal
  ├─ feature-model     建仓库功能/入口图，枚举范围内测试与示例
  ├─ local-assets      盘点本地模型/数据集（不下载新内容）
  ├─ env-setup         研究 CANN/torch/torch_npu 兼容性，建环境装 NPU 栈
  ├─ (初始适配)         主 Agent 修改源码，按需查询迁移知识与官方文档
  ├─ repair            单目标失败：诊断 → 最小修复 → 重跑
  ├─ vllm-ascend-env   需要 vLLM 服务时安装/验证 vLLM-Ascend
  ├─ (代表性验证)       capability checks + representative goals
  ├─ report-pr         证据 → 保守工程报告 / PR 文本
  └─ docs-support      证据稳定后更新仓库 NPU 支持文档
```

每个 skill 目录内的 `agents/openai.yaml` 是 Codex 接口元数据（display_name、default_prompt），不是 Claude agent 定义。

## 4. 知识分层与外部知识机制

| 层 | 位置 | 信任级别 |
| ---- | ------ | --------- |
| 内置迁移知识 | `plugins/ecosystem-code-repository-npu-migration/references/migration_knowledge/`（backend、分布式、operator、checkpoint、launcher 五主题） | 入门参考，仍须源码验证 |
| 任务级外部知识 | 用户提供的 Markdown 根目录（`--knowledge-root` 或 `external_knowledge_root:`） | 不可信参考数据 |
| 当前官方文档 | 在线 CANN/torch_npu/vLLM-Ascend 文档 | 建议刷新，按已安装版本核对 |

外部知识机制（`goal` 和 `repair` 支持）：

- 根目录必须是可读绝对路径；`discover_external_knowledge.py` 递归发现 `.md`/`.markdown`，只读、不复制文件、不建索引、不写仓库配置
- 可选的根目录 `index.md` 只作路由提示；无效或空目录报告 `BLOCKED_EXTERNAL_KNOWLEDGE`
- 外部资料不能替代真实 NPU 执行证据

## 5. `.npu-agent` 轻量工作笔记

仓库级 `goal` 可在目标仓库的 `.npu-agent/` 下维护两份笔记，仅用于 context compaction 后回看，**不是**断点恢复文件、流程控制状态或支持结论来源：

- `migration_progress.md`：目标、已完成、当前重点、剩余、阻塞、关键决定
- `compatibility.json`：按 entry/group 记录命令、日志、结论、风险

只有主 goal Agent 在里程碑后写入；repair 和子代理只返回事实。文件缺失可重建，默认不提交业务仓库。

## 6. Frontmatter 与领域映射

8 个 skill 使用仓库规范富 frontmatter（`name`、`description`、`version`、`domain: ecosystem-code-repository-npu-migration`、`function`、`tags`、`author`）。`validate_skill_design.py` 强制校验：必填字段齐全、`domain` 为 `ecosystem-code-repository-npu-migration`、`function` 与目录映射一致、`version` 为 semver、`tags` 1–5 项、`description` 以 `This skill should be used when` 开头。

name → function 映射（即 `plugin_bundle.SKILL_FUNCTION`）：

| skill | function | 说明 |
| ------- | ---------- | ------ |
| goal | model-adaptation | 迁移协调入口 |
| repair | model-adaptation | 失败诊断修复 |
| feature-model | model-analysis | 仓库功能分析 |
| env-setup | deployment | NPU 环境搭建 |
| vllm-ascend-env | deployment | vLLM 服务安装 |
| local-assets | common | 本地资产发现 |
| report-pr | common | 报告生成 |
| docs-support | common | 支持文档 |

## 7. 安装、版本与发布校验

### 安装

```bash
# 方式一：一键脚本（同时创建软链接/junction 并安装）
bash install.sh claude-code    # Unix，Claude Code
install.bat claude-code        # Windows，Claude Code
bash install.sh codex          # Unix，Codex
install.bat codex              # Windows，Codex

# 方式二：手动
claude plugin marketplace add .
claude plugin install ecosystem-code-repository-npu-migration@MindSpeed-Agent

codex plugin marketplace add .
codex plugin add ecosystem-code-repository-npu-migration@MindSpeed-Agent
```

### 版本与 bundle lock

- `set_plugin_version.py <semver>` 同步三份版本位置（`.claude-plugin/plugin.json`、`.codex-plugin/plugin.json`、根 `.claude-plugin/marketplace.json`——该市场清单同时供 Claude Code 与 Codex 消费）并自动重跑 `build_bundle_lock.py --write`
- `bundle.lock.json`（schema `ecosystem-code-repository-npu-migration-bundle-lock-v3`）记录插件清单、8 skill 名单、全部发布文件的 sha256；skill 文件经软链接读取内容、以插件相对路径 `skills/<name>/...` 为键，canonical 根路径不进入 lock
- `build_bundle_lock.py --release` 的干净度检查同时覆盖插件目录与 `skills/ecosystem-code-repository-npu-migration/`

### 校验命令（从仓库根执行）

```bash
python plugins/ecosystem-code-repository-npu-migration/scripts/build_bundle_lock.py --check
python plugins/ecosystem-code-repository-npu-migration/scripts/validate_plugin_package.py --json
python plugins/ecosystem-code-repository-npu-migration/scripts/validate_bundle.py --json
python plugins/ecosystem-code-repository-npu-migration/scripts/validate_skill_design.py --json
python -m pytest plugins/ecosystem-code-repository-npu-migration/tests
```

校验器内置 `ensure_skill_links()`：fresh clone 上直接运行会自动创建软链接（Windows 需先运行 install.bat 创建 junction）。这些检查只证明插件结构、资源引用和提示词契约；目标仓库是否真正支持昇腾 NPU，仍需实际运行证据。

## 8. 维护检查清单

新增 / 移动 / 重命名 skill 时，必须同步以下各处并重新生成 lock：

1. 正本放入 `skills/ecosystem-code-repository-npu-migration/<function>/<skill-name>/`，frontmatter 用富格式（模板见 `docs/skill-template/SKILL.md`）
2. `plugins/ecosystem-code-repository-npu-migration/scripts/plugin_bundle.py`：`EXPECTED_SKILLS` + `SKILL_FUNCTION` 映射
3. `plugins/ecosystem-code-repository-npu-migration/.claude-plugin/plugin.json`：Claude 清单 skills 数组追加 `"./skills/<skill-name>"`
4. `install.sh`：preflight 目录清单、幂等清理清单、`ln -s` 块
5. `install.bat`：preflight 块、junction 清理循环、`mklink /J` 块、`:stage_npu_skills_for_install` / `:restore_npu_skill_junctions`
6. `set_plugin_version.py <新版本>` 重生成 bundle.lock；插件 README 版本表述同步
7. 跑第 7 节全部校验命令

skill 引用插件级资源（`references/`、`scripts/`）一律写 `<plugin-root>/...` 锚点；skill 内部资源用 skill 相对路径。
