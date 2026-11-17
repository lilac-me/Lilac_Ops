---
name: veomni-precision-alignment
description: >-
  This skill should be used when the user asks to align numerical precision
  between GPU, single-card NPU and multi-card NPU VeOmni training: fix seeds/
  inputs/dtype/config, record golden manifests, compare per-operator/module/
  logits/loss/gradient/parameter-update, bisect long networks to locate the
  first divergence node, inspect NaN/Inf and FSDP2/SP/EP precision impact, or
  rerun regressions after fixes.
version: 0.1.0
domain: common
function: alignment
tags:
  - veomni
  - precision
  - golden
  - bisect
  - regression
author: mindspeed-agent
license: MIT
compatibility: "VeOmni >= 0.1"
---

# VeOmni 精度对齐

## 目的

定位 GPU、单卡 NPU 和多卡 NPU 之间的数值差异，收敛到模型定义的精度标准。

## 对齐流程

1. **条件固化**
   - 固定：代码版本（commit hash）、权重、输入数据、随机种子、dtype、训练配置
   - 所有条件记入 `golden_manifest.json`，保证 Golden 可复现
2. **Golden 生成**
   - 在 GPU（或单卡 NPU）上生成参考输出：逐层 hook 输出、logits、loss、梯度、参数更新量
   - Golden 保存方式与目录在 manifest 中记录
3. **逐层比较**
   - 比较粒度依次为：算子 → 模块 → logits → loss → gradient → 参数更新
   - 每层记录 max abs diff、相对误差、余弦相似度（logits 全词表）
4. **首个差异定位**
   - 长网络使用**二分方式**：先看中间层，差异存在则向前半段收缩，否则向后半段收缩，直到锁定第一个差异节点
   - 锁定后检查该节点：输入 shape、mask、数据预处理、算子实现差异、dtype 转换点
5. **异常检查**
   - NaN/Inf：定位首次出现位置与传播路径
   - 并行影响：对比单卡 NPU 与多卡 NPU（FSDP2/SP/EP），确认差异来自并行切分而非实现错误
6. **修复与回归**
   - 修复后重新执行相关回归；回归范围按修改影响面扩大（跨模态修改强制扩大范围）
7. **阈值**
   - 阈值根据模型、dtype 和业务指标确定，**不在本 Skill 中设置统一标准**
   - 所用阈值必须在 accuracy_report.json 中显式记录及依据

## 输出契约

- `golden_manifest.json`（Schema：`benchmarks/veomni/schemas/golden_manifest.json`）：代码版本、权重、输入、种子、生成方式
- `accuracy_report.json`（Schema：`benchmarks/veomni/schemas/accuracy_report.json`）：

```json
{
  "model": "string",
  "pairs": [
    {"baseline": "gpu", "target": "npu_single", "threshold": {"metric": "cosine", "value": 0.0},
     "first_divergence": {"node": "string", "layer": 0, "diff": {"max_abs": 0.0, "rel": 0.0}},
     "nan_inf": false, "conclusion": "PASS | FAIL"}
  ],
  "parallel_impact": {"fsdp2": "PASS | FAIL | N/A", "sp": "PASS | FAIL | N/A", "ep": "PASS | FAIL | N/A"},
  "regression": {"executed": true, "passed": true},
  "conclusion": "string"
}
```

## 门禁与约束

- 放行条件：输入和测试条件一致（manifest 核对），达到模型定义的精度标准，无非预期 NaN/Inf。
- 精度失败时输出首个差异节点定位结果，交回适配侧修复；禁止带病放行。
- 比较脚本必须使用相同的输入文件与随机种子执行，禁止用"大概一致"代替量化数据。
