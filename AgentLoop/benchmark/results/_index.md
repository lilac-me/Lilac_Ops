# MindSpeed-Agent 能力 benchmark — 对比报告

> 排名规则：**完成** > **效果分** > **token 更少**。效果分 = 各维度质量加权；成本（token/耗时/迭代）单独呈现，不与效果混淆。
>
> 每一行是一个 {域 × case × harness × model} 组合，所以**换 harness 或换模型就是再跑一遍、结果自动进同一张表**——对比能力靠 `harness` / `model` 两列，不需要另一套跑法或另一张榜。

| # | 域 | harness | model | case | 难度 | 完成 | 效果分 | 维度分 | token | 耗时(min) | 效果/百万token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ops | dsh | deepseek-v4 | gelu | easy | ✅ | **100.00** | completion=100 performance=100 engineering=100 convergence=100 efficiency=100 | 286674 | 59.8 | 348.8283 |
| 2 | ops | reference | gold-standard | relu | easy | ✅ | **99.12** | completion=100 performance=97 engineering=100 convergence=100 efficiency=100 | — | 35.7 | — |
| 3 | ops | dsh | deepseek-v4 | scaled-dot-product-attention | hard | ✅ | **95.68** | completion=100 performance=86 engineering=100 convergence=100 efficiency=100 | 30961 | 25.0 | 3090.3395 |
| 4 | ops | dsh | deepseek-v4 | swiglu | hard | ✅ | **95.59** | completion=100 performance=85 engineering=100 convergence=100 efficiency=100 | 61192 | 21.0 | 1562.1323 |
| 5 | ops | dsh | deepseek-v4 | matmul | hard | ✅ | **93.60** | completion=100 performance=79 engineering=100 convergence=100 efficiency=89 | 506953 | 28.0 | 184.6325 |
| 6 | ops | dsh | deepseek-v4 | rmsnorm | medium | ✅ | **75.05** | completion=100 performance=17 engineering=100 convergence=100 efficiency=100 | 118506 | 24.0 | 633.3013 |

> 覆盖范围：6 个 run —— harness/model：dsh/deepseek-v4 ×5、reference/gold-standard ×1。
> `reference` 是黄金回灌基线，不是「换个 harness 跑一遍」，因此**被测组合只有 1 个（dsh/deepseek-v4）**：本表是**该组合**的能力基线，**不是**跨 harness / 跨模型对比——Claude Code / Codex / 其它模型的补测尚未进行（6 条结果全部由 `dsh` 产出）；补测就是换环境再跑一遍同一 case；协议见 `benchmark/README.md`。

_按以上排名规则排序。维度分 = 该领域各质量维度得分（0-100）。_
