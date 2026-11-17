# 规则目录治理（rules/）

`rules.yaml` 是审查规则的**唯一事实源**。issue 必须绑定 `rule_id`；无匹配规则的发现不得计分。

## 规则结构

每条规则：`id`（R-<维度>-<序号>，全仓唯一，废弃后不复用）、`dimension`（correctness/completeness/clarity/accessibility）、`title`、`kind`（deterministic = lint 脚本判定，必填 `linter`；semantic = LLM reviewer 判定）、`severity`（类型等级：critical 命中即 critical，不因投票人数少降级）、`weight`（扣分权重）、`primary`（该对象主责维度：同一对象多维度命中时只扣主责维度的分）、`judge`（判定标准）、`evidence`（证据要求）。

## 新增规则流程

1. 在 rules/README.md 下方"规则提案区"登记提案（id 占位 `R-XXX-???`、动机、判定草案、权重建议）；
2. 评审通过后在 rules.yaml 登记（id 定号、字段齐全）；
3. deterministic 规则同步实现/扩展对应 linter；semantic 规则同步更新 `skills/doc-skills/review/doc-quality-check/references/dimension_guides.md` 与判例文件；
4. 更新结构校验器（`tools/check_doc_agent_structure.py`）的规则数量断言；
5. 全量回归：`python "<DOC_AGENT_ROOT>/linters/doc_lint.py" rules`；`<DOC_AGENT_ROOT>` 按 doc-orchestrator 的平台运行时说明解析。源码仓额外运行 `python tools/check_doc_agent_structure.py` 与 `python tools/check_doc_agent_runtime.py`。

## 废弃流程

- 规则不再适用时加 `deprecated: true` 并保留 id 与字段（不删除——历史 issue 仍引用该 id）；
- 判定说明改为"已废弃，不再执行"，kind 保持原值；
- 版本发布说明中记录废弃原因。

## 变更版本化

- rules.yaml 每次变更在文件头注释登记 `changed: <日期> <id列表> <原因>`；
- 评分口径变化（weight/severity）视为破坏性变更，需在提案区说明对存量报告的影响。

## 规则提案区（当前为空）

- （暂无提案）
