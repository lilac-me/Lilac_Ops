---
name: vla-model-adaptation-optimization
description: >-
  This skill should be used when the user asks to adapt and optimize a
  vision-language-action (VLA) model or a world model on Ascend NPU: preparing the
  CANN / torch_npu environment and offline datasets and weights, injecting
  transfer_to_npu as a one-shot entry hook, replacing flash-attn with the NPU
  attention backend, adapting device string / distributed / AMP / fp64 coupling
  points, diagnosing silent numerical errors (NaN, non-decreasing loss, mask
  polarity), collecting and analysing Ascend profiling, replacing hand-written
  RMSNorm / RoPE with fused NPU kernels, and delivering the migration patch,
  training scripts and change log. It covers models whose training pipeline runs
  on PyTorch/CUDA and needs to run on Ascend NPU.
version: 0.1.0
domain: vla-model
function: model-adaptation
tags:
  - ascend
  - npu
  - vla
  - world-model
  - migration
author: mindspeed-agent
license: MIT
compatibility: "torch 2.7.1 + torch_npu 2.7.1.postX（CANN 8.x–9.x）；transformers >= 4.51"
metadata:
  zh-description: >-
    把 GPU(CUDA) 版 VLA（视觉-语言-动作）与世界模型的训练 pipeline 适配到昇腾 NPU、
    并做性能优化与交付时使用。覆盖环境搭建、一键迁移、attention 决策、静默错排查、
    profiling 采集与分析、融合算子替换。
---

# VLA / 世界模型 NPU 适配与优化

面向**含训练 pipeline 的 VLA（视觉-语言-动作）与世界模型**：把 GPU(CUDA) 训练适配到昇腾 NPU，
做性能优化，并交付 patch、`test/` 脚本、README 与 `MIGRATION_LOG.md`。方法论提炼自昇腾社区
已适配的同类模型与官方迁移调优文档。

适用边界：含训练 pipeline 的 VLA 与世界模型。

## 流程（三步）

全程用 TodoWrite 记录阶段状态。迁移 → 优化 → 交付，不允许跳跃。

```text
迁移阶段（环境 + 代码 NPU 迁移）
1. 搭环境 + 备数据/权重 → references/flow/1-migrate.md §1-2（环境成本常高于代码改动；数据集与预训练权重都要备齐）
2. 静态分析（只读）：从启动脚本追到训练主函数；产出数据流一页纸写进模型档案卡（模板 `assets/model-card.template.md`，落到工作区，见下「产物位置」）；跑 `scripts/scan_gpu_deps.sh` 做**组件体检**（只列需先定方案的组件，结论决定入口块怎么写）
3. 查 references/ref-models.md 找 2~3 个结构最像的已适配模型，读其 patch 时只当参考线索，不当改动清单——结构/版本差异可能很大，改哪里、改成什么值都要在本模型上重新判定
4. 主函数入口块 transfer_to_npu 一键迁移 → references/flow/1-migrate.md §3.2
5. 跑通后按报错/静默错定点修复人工耦合点（设备字符串 / 分布式 / AMP / attention 决策）→ references/flow/1-migrate.md §3.3-3.4
6. 验证：训练正常迭代，loss 有限、无 NaN/Inf/恒 0/异常跳变，目标卡数跑通前向 + 反向 + optimizer step
   每处人工改动当时记一条审计凭据（见下「审计凭据」），不要留到交付时再回溯

优化阶段
7. 先采一次开箱 profiling，量出瓶颈类别（先量后猜）→ references/flow/2-optimize.md §0
8. 一次只加一项优化，每项单独做精度对齐（前后 loss 一致 / 无异常值）→ references/flow/2-optimize.md §1-8（按序：零代码 → 显存/FSDP → 融合算子 → host-bound → batch/数据 → 优化器）
9. 优化后再采一次，与开箱基线对比计算 / 通信 / 空闲（口径先注册进档案卡）；有量化正收益的项固化为模型默认 → references/flow/2-optimize.md §9

交付
10. 产出 patch + test/ + README（环境、数据集与权重获取写成可执行步骤）+ **MIGRATION_LOG.md** → references/flow/3-deliver.md（含冗余审计：凭据对账 + 逐个改动回退验证，去掉 transfer_to_npu 已覆盖的多余改动；MIGRATION_LOG 规格见该文档，是 agent 迁移的必需产物）
11. 踩坑按通用性分流：模型特有的留在该模型的踩坑记录（工作区）；通用结论按模板沉入对应阶段文档
```

第 5 步里那种 `.cuda()`→`.npu()` 的机械替换多数已被 `transfer_to_npu` 等自动化工具覆盖，交付前冗余审计会去掉（references/flow/3-deliver.md），不必逐处手改。

## ⚠️ 两条关键约束

1. **入口块加完先跑，不要逐处修改 CUDA 调用。** `transfer_to_npu` 覆盖 `.cuda()` / `.to("cuda")` / `torch.cuda.*`，以及白名单函数上的 `device` / `device_type` / `map_location` / `device_id` 实参（能兜多少、边界在哪见 references/flow/1-migrate.md §3.2；读完那一节就不必再逐个修改 CUDA 调用）。**先运行，再按报错/静默错定点修复**；逐处手改既费时，又会给交付 patch 制造冗余。
2. **`ASCEND_LAUNCH_BLOCKING=1` 绝不进性能跑与交付件**（同步下发，使 `TASK_QUEUE_ENABLE` 完全失效）。语料 67 个模型的 test 脚本里**一次都没出现过，出现即漏删**；交付前自查 `grep -rn "ASCEND_LAUNCH_BLOCKING" test/`（流程见 references/flow/3-deliver.md 交付前检查）。

## 执行与上下文预算（长任务防稀释，必守）

目标：B/C 阶段不被 A 阶段噪音稀释、C 交付审计不被长上下文冲淡。五纪律 + 审计凭据：

1. **懒加载**：只读当前阶段文档（本文件常驻，其余按文件索引按需）。当前阶段结束后不反读上一阶段文档；references/ 只在被索引点名时读，不顺手通读。
2. **产物落盘，只读尾巴**：日志 / profiling / scan 报告落到工作目录，需要时 `grep` / `tail` / 取摘要回读一小段，不整篇读回上下文。
3. **阶段边界压缩**：A→B、B→C 切换时，把上一阶段结论压成 ≤20 行写进档案卡「检查点」字段（`assets/model-card.template.md`），此后决策只引用检查点，不回溯过程细节（环境踩坑、报错排查等不重提）。
4. **尝试必记**：优化阶段每做一个实验，手段 + 数值 + 结论写进模型档案卡的「优化记录表」（references/flow/2-optimize.md §9）；**失败项也记一行**（避免重复尝试、避免遗忘该做的候选）。
5. **域外只读不写**：模型仓与工作目录之外的路径一律不读改；确需参考先复制进工作区再读。

**审计凭据（A 阶段开始记，C 阶段只对账）**：每处人工改动当时记一条：`改了什么 + 为什么改 + 工具是否覆盖不到`（格式见档案卡「审计凭据」段，`references/flow/3-deliver.md` 冗余审计）。记录时**凭据只是线索，不是豁免**：C 阶段仍须逐 hunks 回退验证（`git apply --reverse` 试删），凭据帮你快速定位「哪些是人工耦合点、哪些是工具覆盖」，但不替代验证本身。

> 并行 fan-out（可选）：B 阶段确有多个独立实验候选、且上下文有余量时，可派 2~3 个子 agent/独立会话各做一种（同种子、独立 A/B）后横向对比。

## 通过标准

| 阶段 | 标准 |
| --- | --- |
| 迁移 | 训练正常迭代，loss 有限且无异常值（NaN / Inf / 恒 0 / 异常跳变），目标卡数分布式**完整跑通前向 + 反向 + optimizer step**（≥3 步，跨过 gradient accumulation 与调度器首轮；步数记进档案卡口径，交付审计的负向验证沿用同一口径）；有 GPU 基线时对齐 Final loss 与任务指标 |
| 优化 | 至少一项优化假设走完闭合实验（归因→手段→对齐→量化）；保留/回退唯一裁判 = 量化收益 + 维护复杂度，中性/负收益一律回退并记录 |
| 交付 | patch 可 `git apply --check` 通过，README 环境/数据/权重三节可执行，精度与性能脚本各一份且参数/开关接得通（profiling 用开关控制即可，不必做成命令行参数），**MIGRATION_LOG.md 逐处改动有「改了什么 + 为什么」** |

Tier 判据（迁移阶段只动 Tier 1）：Tier 1 = 不做就无法运行或数值一定错；Tier 2 = 不做也能跑、只是慢 → 全部推迟到优化阶段。Tier 由 agent 结合运行结果判定；`scripts/scan_gpu_deps.sh` 只做组件体检，不分 Tier。阶段文档里同一件事也叫「兼容性修复 vs 换算法」（references/flow/1-migrate.md §3.4、references/ref-models.md）。

## 文件索引

按当前阶段读，不要一次读完：

| 文件 | 何时读 | 内容 |
| --- | --- | --- |
| `references/flow/1-migrate.md` | 迁移阶段 | 环境搭建（版本配对 / CANN / 依赖 / 启动方式 / 报错速查）+ 代码迁移（一键迁移 / 人工耦合点 / attention 决策 / 静默错速查） |
| `references/flow/2-optimize.md` | 优化阶段 | profiling 采集与分析、瓶颈→手段、手段库（零代码 / 显存 / FSDP 粒度 / 融合算子 / host-bound / batch / 数据 / 优化器）、对齐与收益口径 |
| `references/flow/3-deliver.md` | 交付 | 交付目录、patch 生成、**MIGRATION_LOG 规格**、冗余审计（凭据对账）、README 要求、训练脚本与 FPS 提取、交付前检查（参数透传 / 调试变量残留 / patch 顺序） |
| `references/appendix-env.md` | 按需 | **环境变量集中收录**（含义/取值/出处/语料覆盖统计；基线脚本 `assets/env_npu.sh` 只收偏离默认值的通用项） |
| `references/ref-models.md` | 按需（流程 3、交付 10） | 参考模型索引 + 扩散/DiT/TE 专项 |
| `references/ref-ops.md` | 按需（优化阶段融合算子） | 算子对照、`npu_fusion_attention` 调用约定、RMSNorm/RoPE/优化器写法 |
| 工作区产物 | 每模型一份 | 档案卡 + 踩坑记录，**不在本 skill 内**，位置见下节「产物位置」 |

工具与资产：

| 路径 | 用途 |
| --- | --- |
| `scripts/scan_gpu_deps.sh` | 组件体检：列出 FA / 图模式 / TE / 自定义算子等需先定方案的组件（references/flow/1-migrate.md §3.1） |
| `scripts/discover_fused_patchers.py` | 语义扫描 vendor 副本的 RMSNorm/RoPE，输出待改定位清单（references/flow/2-optimize.md §4） |
| `assets/env_npu.sh` | 环境变量基线脚本 |
| `assets/train_Np.sh.template` | 训练脚本模板（参数解析 + FPS 提取 + ModelZoo 收尾块） |
| `assets/README.template.md` | README 模板（references/flow/3-deliver.md 引用） |
| `assets/model-card.template.md` | 模型档案卡模板（数据流一页纸 / 基线 + 口径注册 / 优化记录表 / 检查点 / 审计凭据） |
| `assets/seed_all.py` | 全局随机种子（`seed_all(seed, is_npu=True)`，含 `torch.npu.manual_seed_all`），迁移早期接进训练入口 |
| `assets/pitfall-note.template.md` | 踩坑记录模板（基本信息 + 踩坑条目） |

## 产物位置

**本 skill 只承载方法论与模板；每次迁移的产物落在工作区，不进 skill 目录。**

| 产物 | 位置 | 说明 |
| --- | --- | --- |
| 工作区根 | `<目标仓根>/.npu-migration/`（环境变量 `NPU_MIGRATION_WORKSPACE` 可覆盖） | 起手时确定并把实际路径写进档案卡 |
| 模型档案卡 | `<工作区>/<Model>/model-card.md` | 执行状态载体（模板 `assets/model-card.template.md`），A→B→C 跨阶段唯一读入 |
| 踩坑记录 | `<工作区>/<Model>/pitfalls.md` | 知识沉淀（模板 `assets/pitfall-note.template.md`），给后续结构相近模型参考 |
| 交付件 | `<目标仓>/model_examples/<Model>/` | patch / `test/` / README / `MIGRATION_LOG.md`，随目标仓提交（见 `references/flow/3-deliver.md`） |

工作区不进版本控制（若目标仓已有 `.gitignore`，把 `.npu-migration/` 加进去）。

## 踩坑记录规范

每个模型一份踩坑记录（`<工作区>/<Model>/pitfalls.md`），只记有参考价值的踩坑，够用就行。

- 不写：进度 / 流水账、机器环境细节（镜像源、搭建耗时、个人路径、数据摆放），这些对后续模型没有参考价值
- 「是否静默」= 不报错但结果错；这类优先沉淀进 `references/flow/1-migrate.md` §3.5 静默错速查
- 通用结论按「诊断 grep → 原始 → 优化 → 边界」模板沉入阶段文档后，踩坑记录里只留一行「已沉入 <文档>」指向，不重复全文
- `references/flow/1-migrate.md` / `references/flow/2-optimize.md` 是收敛后的方法论，踩坑记录是未收敛的原始现场，不复述阶段文档已有的结论

## 已验证模型

已完成并交付的案例（结构 / 上游仓库 commit / 开箱→优化性能）见
`references/ref-models.md` §已完成案例。

## 语料

- patch 语料：`DrivingSDK/model_examples/`（67 个模型，会随仓增长，<https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples> 下拼模型名即可核对）
- 算子库与 patcher：`DrivingSDK/mx_driving/`
- 官方方法论：`DrivingSDK/docs/zh/migration_tuning/model_optimization.md`
- 算子清单：`DrivingSDK/docs/zh/api/context/`（63 个算子文档，频次与调用约定见 `references/ref-ops.md`）
