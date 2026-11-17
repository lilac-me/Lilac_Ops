---
name: veomni-model-adaptation
description: >-
  This skill should be used when the user asks to actually onboard a model into
  VeOmni: create GPU and NPU Patchgen configs, register model/config/processor,
  convert and load weights, integrate Text/VLM/DiT/Omni/RL Trainer, add multimodal
  data preprocessing, generate training YAML and toy configs, run Patchgen, and
  add model loading, forward and training tests.
version: 0.1.1
domain: common
function: model-adaptation
tags:
  - veomni
  - patchgen
  - model-adaptation
  - trainer
author: mindspeed-agent
license: MIT
compatibility: "VeOmni >= 0.1"
---

# VeOmni 模型适配

## 目的

完成模型在 VeOmni 中的实际接入，产出可运行的适配代码、训练配置与测试记录。

## 前置条件

- `model_inventory.json` 已完成，且已包含适配方案、改动文件清单与风险
- 环境检查通过（或静态分析模式下仅生成代码不执行）

## 适配流程

1. **创建 Patchgen 配置**
   - 分别创建 GPU 与 NPU patch 配置（NPU patch 中处理 torch_npu 差异、attention 实现替换等）
   - 配置文件命名与目录位置遵循 VeOmni 现有 patch 目录约定
2. **注册**
   - 模型类、config 类、Processor 注册到 VeOmni 模型注册表
   - 补全 auto_map / remote code 所需条目
3. **权重加载**
   - 接入权重加载路径；必要时编写权重转换脚本（名称映射、分片、dtype）
   - 转换脚本与映射表随适配代码一并落盘，保证可复现
4. **Trainer 接入**
   - 按 inventory 判定的类型接入 Text / VLM / DiT / Omni / RL Trainer
   - 补充多模态数据预处理与 metadata（VLM/Omni 场景）
5. **配置生成**
   - 生成训练 YAML（并行参数留待 veomni-parallel-recipe 细化）
   - 生成 toy config（小模型/小数据，用于快速功能验证）
   - Flux DiT SFT 场景须读取 `references/flux-sft-training-readiness.md`，校验并行、全局 batch、VAE 子目录、模型 JSON 与图像条件预处理的训练前置条件
6. **运行 Patchgen**
   - 执行 Patchgen 生成目录下 modeling 文件
   - **generated 文件只能通过 Patchgen 生成，禁止手工修改**；发现需要改动时回去改 patch 配置后重新生成
7. **测试**
   - 模型加载测试（权重能完整加载、无缺失/多余 key）
   - 前向测试（toy 输入，logits/输出 shape 正确）
   - 训练测试（toy config 完成 1-2 step 前反向）
   - 记录测试命令与结果

## 输出契约

全部写入 `outputs/{model}/` 与 VeOmni 工作副本：

| 产物 | 说明 |
| ------ | ------ |
| Patchgen 配置 | GPU/NPU patch 配置文件 |
| 注册改动 | registry / auto_map 代码改动 |
| 权重转换脚本 | 含映射表 |
| 训练 YAML + toy config | 可直接运行的训练配置 |
| `patch_manifest.json` | Schema：`benchmarks/veomni/schemas/patch_manifest.json`，记录 Patchgen 配置、生成文件清单、注册项和 VeOmni source revision |
| 测试记录 | 加载/前向/训练测试的命令与结果 |

## 门禁与约束

- 放行条件：Patchgen、注册、配置和测试改动完整，**未手改 generated 文件**（manifest 中的生成文件与 Patchgen 重跑结果一致）。
- 权重下载等外部动作必须先获得授权。
- 功能验证失败时回到本 Skill 修复，不允许跳过直接进入精度阶段。

## 参考资料

- `references/patchgen-flow.md`：Patchgen 主流程与 generated 文件管理细则
- `references/trainer-matrix.md`：Text / VLM / DiT / Omni / RL Trainer 接入差异矩阵
- `references/transformers-v5-migration.md`：Transformers v5 迁移检查点（模型基于 v5 结构时，创建 Patchgen 配置前先过一遍）
- `references/flux-sft-training-readiness.md`：Flux DiT SFT 的配置、权重加载、批量前向和图像条件预处理改动及验证清单
