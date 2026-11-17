# vLLM-Ascend 安装策略

针对选定的 CANN、torch、torch_npu、Python 和架构，使用当前官方 vLLM-Ascend 安装与兼容性文档。用户明确指定的版本仍是约束；不要仅根据软件包版本号相同就推断兼容。

## 操作步骤

1. 确认选定的绝对 Python 能通过 `import torch; torch.npu.set_device(0)`。
2. 记录 torch/torch_npu/vLLM 软件包版本和导入路径。
3. 候选兼容性不明确时，检查官方 wheel 元数据，或在不安装的情况下运行 `probe_vllm_ascend_wheels.py`。
4. 使用选定 Python 安装文档规定的 vLLM 和 vLLM-Ascend 组合。
5. 重新检查受保护的软件栈并重跑 base gate。依赖解析导致漂移时，重新安装固定的官方 torch/torch_npu 组合；要求无法共存时阻塞。
6. 使用明确的本地模型，或请求 `local-assets` 在授权根目录下发现模型。
7. 运行 `vllm_ascend_smoke.py`；只有在无 CPU/CUDA fallback 的真实 NPU 生成通过时，才能将生成标为 PASS。

返回直接命令证据。软件包导入只能证明安装；生成 smoke 证明被测试的 engine/model/request 路径，不证明目标仓库集成或分布式服务。
