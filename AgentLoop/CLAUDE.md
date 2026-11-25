# AgentLoop Profile Opt Lite

只处理训练 Profiling 驱动的现有 Triton 算子优化。不要重新生成算子，不引入其他领域工作流。

- 整网任务：先读 README.md 和 skills/ops-skills/triton/profile-driven-triton-opt/docs/model-hotspot-pass.md，使用 plugins/ops/engine/model_hotspot.py run 或 prepare → 逐个 prompt.txt → finish。
- 已有 Profiling：使用 skills/ops-skills/triton/profile-driven-triton-opt/SKILL.md 与同目录 AGENTS.md，配合 analyzer 和 source assembler。
- 单算子必须使用 CANNBot verifier、latency/simulator optimizer；不自行伪造 benchmark 或优化结果。仅在工作副本修改，生成经过验证的补丁。
- 缺真实源码则跳过；弱候选人工核定；整网只在 finish 统一集成与回归，失败回滚。没有自动二次热点采集。
- 保留原有产物契约及测试；不绕过精度、性能及补丁验收。
