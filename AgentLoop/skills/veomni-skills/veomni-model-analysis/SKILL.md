---
name: veomni-model-analysis
description: >-
  This skill should be used when the user asks to analyze the feasibility of
  onboarding a new model to VeOmni, classify it as Dense/MoE/VLM/Omni/DiT/RL,
  inspect config.json, modeling, processor and tokenizer files, map Patchgen,
  model registration and Trainer integration points, identify the closest
  reference model in the repo, or estimate adaptation workload, gaps and risks.
version: 0.1.0
domain: common
function: model-analysis
tags:
  - veomni
  - model-analysis
  - patchgen
  - feasibility
author: mindspeed-agent
license: MIT
compatibility: "VeOmni >= 0.1"
---

# VeOmni 模型接入分析

## 目的

分析新模型接入 VeOmni 所需的改动和风险，输出适配计划，供适配阶段直接执行。

## 分析流程

1. **模型资产盘点**
   - 读取 HF 目录：`config.json`、modeling 文件、processor、tokenizer、generation_config
   - 记录权重格式（safetensors/bin）、分片数量、总参数量
2. **模型类型判定**

   | 特征 | 判定 |
   | ------ | ------ |
   | 纯 Decoder + FFN | Dense |
   | 含 experts/router/moe 字段 | MoE（记录专家数、top-k、激活参数量） |
   | 含 vision_tower / 视觉 encoder | VLM |
   | 含 audio/omni 模块或多流输入 | Omni |
   | 含 VAE / diffusion 调度器 | DiT |
   | 训练入口为 RL Trainer / rollout 组件 | RL |

3. **接入点映射**（对照 VeOmni 仓库结构）
   - Patchgen：需要新增哪些 patch 配置（GPU 与 NPU 分开列出）
   - 模型注册：`model registry`、auto_map、config 类注册点
   - Trainer：Text / VLM / DiT / Omni / RL Trainer 中选择哪个入口
   - 数据：训练数据字段、collator、多模态预处理与 metadata 要求
4. **约束分析**
   - 算子支持：识别自定义算子/特殊 attention 实现是否已有 NPU 支持；缺失项列入交接清单
   - 并行需求：评估 FSDP2 / Ulysses SP / EP 适用性（MoE 需要 EP）
   - Checkpoint：DCP 保存/加载兼容性
5. **参考模型选择**：在 VeOmni 已支持模型中选定结构最接近者（如 Qwen3、Qwen3-MoE、Qwen3-VL、Wan2.1），作为适配模板。
6. **工作量与风险**：按改动点估算工作量（小/中/大），列出缺口、风险和依赖。

## 输出契约

- `model_inventory.json`（Schema：`benchmarks/veomni/schemas/model_inventory.json`）：

```json
{
  "model": "string",
  "model_type": "dense | moe | vlm | omni | dit | rl",
  "structure": {"layers": 0, "hidden_size": 0, "params_total": "string",
                 "params_active": "string | null", "experts": "int | null"},
  "assets": {"config": "string", "modeling": ["string"], "processor": "string | null",
              "tokenizer": "string", "weight_format": "string"},
  "integration": {"patchgen_configs": ["string"], "registry_points": ["string"],
                   "trainer": "string", "data_fields": ["string"]},
  "operator_gaps": ["string"],
  "parallel_needs": {"fsdp2": true, "ulysses_sp": false, "ep": false},
  "reference_model": "string",
  "workload": "small | medium | large",
  "risks": ["string"]
}
```

- 适配方案、改动文件清单与风险写入 `model_inventory.json` 的集成、风险和备注字段；面向人的说明由最终交付件统一汇总。

## 门禁与约束

- 放行条件：模型、数据、权重、Patchgen、算子和并行需求**均已分类**，缺一项则继续分析不得放行。
- 本 Skill 只读分析，不修改 VeOmni 仓库文件。
- 缺少 NPU 底层算子 → 在 inventory 的 `operator_gaps` 中列出，交 orchestrator 转交 Ops Agent。
