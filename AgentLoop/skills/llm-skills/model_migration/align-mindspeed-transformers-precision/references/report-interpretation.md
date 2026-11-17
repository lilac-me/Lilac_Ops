# 精度比较报告解读

## 决策字段

| 字段 | 含义 |
| --- | --- |
| `format_version` | 报告 schema 版本 |
| `passed` | 主有限性指标是否达到阈值 |
| `primary_metric` | 决策使用的指标 |
| `threshold` / `threshold_percent` | 阈值的比例/百分比形式 |
| `metrics.average_token_cosine` | 默认主门禁：有效位置全词表逐 token 余弦均值 |

使用 `actual >= threshold` 或 `actual < threshold` 表述，不以诊断指标替换门禁。

阶段 harness 还应区分：

| 字段 | 含义 |
| --- | --- |
| `numeric_pass` | 当前阶段自身比较通过 |
| `gate_bypassed` | 直接前驱检查被显式绕过 |
| `diagnostic_only` | 仅可用于诊断 |
| `sequential_precision_certified` | 当前数值通过且前驱链未绕过并已认证 |

`numeric_pass=true` 与 `sequential_precision_certified=false` 可以同时成立。

## 输入与 shape

| 字段 | 检查 |
| --- | --- |
| `input_sha256` | 不一致则比较无效 |
| `input_shape` | `[B,S]`，两侧必须一致 |
| `logits_shape` | 裁剪明确词表 padding 后为 `[B,S,V]` |
| `valid_token_count` | 共享 mask 纳入的有效位置数 |

## 余弦指标

| 字段 | 用途 |
| --- | --- |
| `minimum/maximum/median_token_cosine` | 最差、最好和典型 token |
| `p01/p05_token_cosine` | 低分尾部 |
| `flatten_cosine` | 全局展平方向，仅辅助 |
| `first_valid_token_cosine` | 历史 token 影响前的基础诊断 |
| `last_valid_token_cosines` | 累积/长程诊断 |
| `token_positions/token_cosines` | 映射到 `[batch, position]` |

## 误差与排序

| 字段 | 用途 |
| --- | --- |
| `max_abs` | 最大单元素差，易受离群值和 logits 尺度影响 |
| `mean_abs` | 平均绝对差 |
| `rms_error` | 对大误差更敏感 |
| `worst_token_position` | token 内误差最大的位置 |
| `top1_match_rate` | argmax 相同比例，near-tie 时可能低于 100% |
| `topK_mean_overlap` | 高分候选集合重合度 |

绝对误差不存在统一阈值，应结合余弦、位置和排序 margin 解读。

## 元数据核对

确认：

- reference/candidate 标签、文件和框架；
- HF 路径、模型类型、dtype、attention backend 和加载状态；
- MCore checkpoint、词表和 dtype；
- direct/Flash/absorb/sparse/DSA 路径；
- TP/PP/CP/EP/ETP/DP/SP/VPP/MTP 和 writer rank；
- 源文件、producer 版本、输入 hash 和同步屏障。

错误 checkpoint 或执行模式产生的数值通过，不满足请求范围。

## 报告模板

```text
结果：PASS/FAIL
主指标：有效 token 平均余弦 = X（阈值 Y）
范围：min X，median X，max X；first X；last mean X
输入完整性：hash；input [B,S]；logits [B,S,V]；有效 token N
诊断：flatten cosine；mean/max abs；Top-1；Top-K
执行：HF backend/dtype；MCore backend/absorb/sparse；并行布局
结论：已对齐，或失败类别与证据
下一步：归档、扩展覆盖，或最小诊断命令
```

门禁通过时，区分“直接基准通过”和“全部优化/稀疏/长上下文/cache 路径通过”；
有绕过时，区分“数值诊断通过”和“顺序链认证通过”。
