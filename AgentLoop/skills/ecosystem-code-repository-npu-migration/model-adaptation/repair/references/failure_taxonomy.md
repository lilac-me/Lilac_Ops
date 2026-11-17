# 失败分类与知识路由

分类第一个具体失败，不要把每条 warning 或下游症状都列为失败。
分类只用于决定检查方向；根因由当前日志和源码决定。

| 类别和常见信号 | 知识主题 | 需要刷新官方来源的内容 |
| --- | --- | --- |
| CANN 路径、runtime 注册、torch/torch_npu 不匹配、导入来源或依赖漂移 | backend 与 runtime（`<plugin-root>/references/migration_knowledge/backend_runtime.md`） | 请求的 CANN/Python/torch/torch_npu 组合和注册行为 |
| `.cuda()`、`torch.cuda.*`、设备选择、DataParallel、AMP 或 dtype | backend 与 runtime（`<plugin-root>/references/migration_knowledge/backend_runtime.md`） | 当前 TorchNPU 和 PyTorch API 签名 |
| NCCL/HCCL、rank、process group、collective 超时或 rendezvous | 分布式与服务（`<plugin-root>/references/migration_knowledge/distributed_serving.md`） | HCCL 和已安装 framework 的分布式行为 |
| Ray worker 可见性、placement、vLLM 生成或权重同步 | 分布式与服务（`<plugin-root>/references/migration_knowledge/distributed_serving.md`） | 当前 Ray 和 vLLM-Ascend API |
| 不支持的 operator、FlashAttention、Triton 或自定义 CUDA extension | operator 语义（`<plugin-root>/references/migration_knowledge/operators_semantics.md`） | 当前 operator 支持情况和精确 symbol 签名 |
| packing、varlen、mask、position ID、cache 布局或数值不匹配 | operator 语义（`<plugin-root>/references/migration_knowledge/operators_semantics.md`） | 当前 fused operator 布局和 dtype 支持 |
| 量化 CUDA backend 或隐藏的 CPU fallback | operator 语义（`<plugin-root>/references/migration_knowledge/operators_semantics.md`） | 请求的量化 backend 支持 |
| checkpoint、state dict、optimizer state、save/load 或 resume | 检查点与启动器（`<plugin-root>/references/migration_knowledge/checkpoint_launchers.md`） | 当前 framework checkpoint 格式和 reshard 支持 |
| Hydra、YAML、shell wrapper、CLI override 或解析出的 asset/device 错误 | 检查点与启动器（`<plugin-root>/references/migration_knowledge/checkpoint_launchers.md`） | 精确的 launcher/config framework 版本 |
| 普通 Python import/config/type error | 从本地 traceback 和调用方开始；只有失败 symbol 跨越上述边界时才继续某个主题 | 当失败 API 行为依赖版本时，查询拥有该 API 的软件包 |

对于 hang，在归类为通信问题前先收集对齐的 rank 或 worker 证据。对于数值失败，在修改容差前先定位第一个发生偏离的 tensor 或语义不变量。

除非 warning 与失败命令、hang、非零退出码、错误结果或用户明确报告的缺陷存在关联，否则不要仅因 warning 修改代码。
