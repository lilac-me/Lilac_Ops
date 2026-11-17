# catalog/ — MindSpeed 生态「仓库 ↔ 插件」知识图谱

本目录是 MindSpeed-Agent 与 MindSpeed 社区代码仓关系的**单一真相源**，驱动
根插件 `ecosystem-routing` skill（见
`skills/common/common/ecosystem-routing/SKILL.md`）做新手意图路由。

## 文件职责

| 文件 | 内容 | 维护者 |
| --- | --- | --- |
| `repos.yaml` | 社区代码仓注册表：`id`（canonical）、`aliases`、`kind`、`backends`、`urls`、`related` | 人工 |
| `plugins.yaml` | 本仓插件注册表：`repo_affinity`（处理哪些仓）、`skills`、`workflow` | 人工 + 半自动 |
| `task-kinds.yaml` | 任务维度枚举（性能/迁移/精度/部署/压测/定位/算子/资料/验证） | 人工 |
| `intent-examples.yaml` | 新手话术黄金测试集（含期望 repo/task/plugin 断言） | 人工（真实会话回写） |
| `generator.py` | 汇总上述 + 扫 `skills/**/SKILL.md` frontmatter → `graph.json` | — |
| `lint_catalog.py` | 本地校验 CLI（别名冲突/引用合法性/金句断言） | — |
| `graph.json` | 生成产物：节点 + 边 + 路由命中（不入库，见 `.gitignore`） | 生成 |

> CI 兜底：`ci/checks/knowledge_graph.py`（由 `python ci/guard.py` 触发），与
> 本地 `python catalog/lint_catalog.py` 使用同一份检查实现（`checks_core.py`）。

## 怎么用

```bash
# 1) 修改 catalog/*.yaml 或 SKILL.md frontmatter 后重新生成
python catalog/generator.py            # 输出 catalog/graph.json
# 2) 本地校验（阻塞问题退出码 1）
python catalog/lint_catalog.py
# 3) CI 同款（覆盖 skills/plugins/marketplace 等全部检查）
python ci/guard.py
```

## 怎么加一个社区仓

1. 在 `repos.yaml` 追加节点：`id`（kebab-case、唯一）、`aliases`（口语/曾用名，
   全局唯一）、`kind`（枚举见 `checks_core.py`）、`backends`、`urls`；
2. 若有插件处理它，在 `plugins.yaml` 对应插件的 `repo_affinity` 追加；
3. 若有 skill 主要作用于它，给该 SKILL.md frontmatter 加可选字段：

```yaml
targets-repo:   # 可选；值须能解析到 repos.yaml 的 id 或 aliases
  - mindspeed-llm
```

1. 重跑 generator + lint，直至无阻塞。

## 怎么加一条新手话术（金句）

1. 把真实用户原话加入 `intent-examples.yaml`（**禁止 AI 凭空生成后直接入库**，
   必须来自真实会话或明确评审过的示例）；
2. 写明 `expected_repos` / `expected_task`（可为 `null` 表示应反问）/ `expected_plugins`；
3. 跑 lint 验证该话术能确定性消歧到期望仓（R1）与任务（R2）。

## 校验规则速览（lint 强校验项）

- `catalog.repos.alias-conflict`：归一化别名/ID 不得跨仓冲突；
- `catalog.plugins.affinity` / `catalog.plugins.skills`：引用须存在；
- `catalog.plugins.workflow`：插件 `workflow` 非空时文件必须存在；
- `catalog.plugins.execution`：`execution` 取值合法；`state-machine` 插件必须声明
  `workflow` + `entry_agent` + `agents`，且 `plugins/<dir>/agents/<agent>` 目录存在；
- `catalog.taskkinds.refs`：插件/skill 引用须存在；
- `catalog.skill.targets`：SKILL.md `targets-repo` 须可解析到任一仓；
- `catalog.intents.repo`：金句确定性消歧必须命中 `expected_repos` 且不误中无关仓。

（警告项 `catalog.repos.kind`、`catalog.intents.task`、`catalog.intents.plugin` 提示
数据问题，不阻塞。）

> 插件执行形态：`execution: skills`（缺省）为纯技能聚合，路由终点是单个 skill；
> `execution: state-machine` 表示插件自带 workflow 状态机，路由终点是
> `entry_agent`（如 inference-model-optimize 的 orchestrator），插件 `skills` 仅作
> 各阶段 agent 的知识底座。graph.json 通过 `edges.plugin_workflows` /
> `edges.plugin_agents` 暴露该关系。
