# 精度排查案例知识库（Cases）

按“问题形态 × 根因算子”独立成篇，每个用例是一个可复用知识库。统一方法论：
**复现 → 一次 Dump 整网统计比对 → 二次细粒度 tensor Dump → 逐元素配对比对 →
单算子复现/结论**。相关 msProbe 配置与文档见 `../msprobe/`。

> **适用前提**：案例假定问题已通过阶段一预检（`../precheck-flow.md` 的 L0–L6），
> 即已排除采样、图模式、权重、配置版本四类配置性根因，确认需要进入 Dump 阶段。
> 若尚未预检，先走预检漏斗，不要直接套用案例。

## 用例索引

| 文件 | 问题形态 | 根因算子 | 关键配置 |
|------|----------|----------|----------|
| [01_cann_upgrade_garbled_all_reduce.md](01_cann_upgrade_garbled_all_reduce.md) | CANN 升级后整段乱码 | `all_reduce`（Embedding 层） | 二次 tensor Dump 限定 Embedding 模块 `list` |
| [02_cann_env_diff_garbled_fused_infer_attention.md](02_cann_env_diff_garbled_fused_infer_attention.md) | CANN 环境差异后整段乱码（Moonlight-16B-A3B） | `npu_fused_infer_attention_score`（MLA FIA） | 二次 tensor Dump 限定 FIA 算子 `list` |

## 案例查询（问题形态 → 案例映射）

定位到“首个偏差算子”后，按下表匹配问题形态，提取根因方向作为参考，再决定深挖思路：

| 问题形态 | 典型首偏算子 | 对应案例/入口 |
|---------|-------------|--------------|
| CANN 升级后整段乱码 | `all_reduce`（Embedding 层） | [01_cann_upgrade_garbled_all_reduce.md](01_cann_upgrade_garbled_all_reduce.md) |
| CANN 环境差异整段乱码/复读 | `npu_fused_infer_attention_score`(FIA) | [02_cann_env_diff_garbled_fused_infer_attention.md](02_cann_env_diff_garbled_fused_infer_attention.md) |
| 图模式 vs eager 纯推理反复/错误 token | routed/shared expert 归约 all-reduce 簿记 | 见 [用例A：all_reduce 归约乱码](01_cann_upgrade_garbled_all_reduce.md) |

无匹配案例时，按通用判定法则：输入一致、输出首偏 → 根因在算子内核；输入已偏 → 回溯上游。

## 共用要点

- 首 token 即乱码 → 只采 `step: [0]` 整网即可。
- 大 Max diff 的整型张量（count/index，如 `tp_send_counts`）要排除，只信浮点张量
  / `Result != pass`。
- 二次 dump 用 `task="tensor"` + `list` 限定目标算子/模块，`data_mode:["all"]` 保留
  输入输出 `.pt`，再逐元素配对比较。
- 判定法则：输入一致、输出首偏 → 根因在算子内核；输入已偏 → 回溯上游首偏点。
