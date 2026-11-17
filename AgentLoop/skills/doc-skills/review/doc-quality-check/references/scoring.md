# 评分与裁决细则（scoring）

## 评分模型

```text
总分 = correctness×40% + completeness×25% + clarity×20% + accessibility×15%
各维度初始 100 分，按 rules.yaml 扣分权重（weight）逐项扣减，下限 0。
通过条件：verdict == "pass"（总分 ≥ 80 且各维度 ≥ 最低分：90/80/60/60）。
```

> 机器计算以引擎 score 输出为准（阈值见 rules/rules.yaml 的 scoring 段）；本文件保留数值仅供语义层参考口径。

## 投票裁决规则

| 独立发现人数 | 默认等级 |
| ------------- | --------- |
| ≥3 位 reviewer | critical |
| 2 位 reviewer | major |
| 1 位 reviewer | minor（专长维度可升级） |

## 类型等级取高

- 同一问题的最终等级取"类型等级（rules.yaml severity）"与"投票等级"的较高者。
- severity 为 critical 的规则命中即 critical，不因投票人数少而降级。
- semantic 规则由 reviewer 自评等级，经投票裁决定级。

## 同一问题判同

`(rule_id, location.file, location.line)` 三元组相同即同一问题（line 缺失时用 anchor）；仅措辞不同视为同一问题。

## 跨维度去重

dimension 得分 = 100 − Σ(该维度已确认 issue 的 weight)，下限 0；非主责维度命中同一对象只记录不扣分：

| 对象 | 主责维度 |
| ------ | --------- |
| 跨仓链接失效/分支错误 | 正确性（R-COR-001）；导航只查路径，不重复扣链接失效 |
| 图片替代文本缺失 | 完整性（R-COM-008）；易获取只查图片路径可达 |
| 必备资料缺失 | 完整性（R-COM-001） |
| 代码块内注释错误 | 正确性（R-COR-005）；易理解只查错别字/表述 |

## 循环

- 修复后重审最多 3 轮（引擎计数）；第 3 轮仍有 critical 或评分未达标 → manual_intervention（quality_report.verdict）。
- "新增 critical" = 与上一轮逐条比对后新增的 critical 问题（由账本查询得出）。
