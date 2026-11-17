---
name: inference-model-skills
description: "This skill should be used when the user asks Codex to locate or route to the repository's existing Ascend inference deployment, model-analysis, batch-invariance validation, benchmarking, autotuning, profiling, or performance-diagnosis skills without duplicating their implementations."
---

# 推理领域 Skill 路由

按用户意图读取对应子目录中的 `SKILL.md`，并相对该文件所在目录解析其资源：

- 部署和服务管理：`deployment/`
- 模型与显存分析：`model-analysis/`
- 批次不变性验收、回归验证与异常定位：`batch-invariance-validation/batch-invariance-validation/`，由独立插件状态机调度其余四个阶段 Skill
- benchmark、寻优和 profiling：`performance-optimization/`

端到端多 Agent 优化使用同级目录 `../inference-model-optimize/SKILL.md`。
