# Backend 与 Runtime 模式

## Runtime 注册与来源

识别信号：

- 缺少 `torch.npu`、`npu` 不是可识别设备，或子进程导入了不同的 torch、torch_npu 或目标软件包。
- backend 注册依赖导入 `torch_npu`、加速器插件 hook 或软件包自动加载行为。

检查选定的绝对 Python、`torch.__file__`、`torch_npu.__file__`、目标软件包的 `__file__`、导入顺序以及 worker/子进程环境。修改显式导入行为前检查当前 TorchNPU 文档，因为注册行为会随版本变化。

优先使用仓库的统一加速器初始化 hook。如果项目同时支持 CPU 或 CUDA，将可选导入限制在 NPU 路径本地。不要全局把不可用依赖标记为已安装。

最低验证要求是：先运行选定解释器的 base gate，再运行能触发改动 hook 的最小仓库导入或 helper。这不能证明 operator、训练或分布式执行。

## 设备选择与 tensor 放置

识别信号包括 `.cuda()`、`.is_cuda`、`torch.device("cuda")`、CUDA tensor 构造器、`torch.cuda.is_available()`、`current_device`、`device_count`、`set_device`、`synchronize` 和硬编码的本地序号。

跟踪设备来源：用户选项、launcher local rank、调度器可见性、framework 抽象或自动检测 helper。在 `ASCEND_RT_VISIBLE_DEVICES` 屏蔽后保留逻辑到物理设备的映射。优先使用已有 device 抽象或窄 NPU 分支，不要全局文本替换。

Tensor 和 model 的放置必须与 optimizer state、inputs、labels、新创建 tensor 以及 checkpoint restore 一致。model 在 NPU 上不证明所有新 tensor 或 optimizer buffer 都在 NPU 上。

在可行时，先使用 CPU-safe input 验证改动后的 selector，再运行真实 NPU forward 或消费它的仓库目标。仅返回 `npu:0` 字符串不能证明放置正确。

## Stream、event 和 memory helper

CUDA stream、event、同步、内存统计和设备属性 API 可能有 TorchNPU 等价 API、不同签名，或没有必需的等价 API。检查 caller 使用 API 的原因：正确性同步、profiling、telemetry、allocator 控制或性能调优。

只有 caller 语义匹配时才使用当前文档化的 NPU API。CUDA-only telemetry 不影响正确性时可以加保护，但不要删除保护正确性的同步或 allocator 行为。单独保留 CUDA 分支。

通过受影响的异步操作验证同步，并通过真实 caller 验证 memory helper。helper 导入成功不能证明 stream 顺序或多步运行时内存有界。

## AMP 与 dtype

识别信号包括 `torch.cuda.amp`、CUDA-only `autocast`、`GradScaler`、设备能力探测、bf16/fp16 假设和意外的 dtype promotion。

检查已安装的 PyTorch/TorchNPU API、仓库版本保护、operator dtype 支持，以及选定 dtype 是否确实需要 scaling。当前版本支持时，优先使用 device-aware `torch.amp` 或仓库的加速器抽象。保持 CUDA 行为不变。

先验证窄范围的 autocast/scaler helper，再运行 loss 和梯度数值均有限的真实 forward/backward。仅成功构造 context manager 不能证明混合精度训练。

## DataParallel 与 backend 专属依赖

`torch.nn.DataParallel`、`flash-attn`、xFormers、Apex、bitsandbytes、CUDA wheel index、`CUDAExtension`、`.cu` 源码和 `nvcc` 检查都是 backend 敏感信号。确认它们是核心、可选、受保护，还是可以由仓库原生分布式路径或纯 PyTorch 路径替代。

不要让 CUDA 软件包在 NPU 上成为必需依赖，不要虚报其可用，也不要删除现有 CUDA 实现。优先使用仓库原生分布式路径，不要自行发明 NPU DataParallel 等价实现。

在项目依赖安装前后记录 torch 和 torch_npu 的版本及导入路径。发生漂移时，恢复官方匹配的固定组合并重跑 base gate。依赖解析器返回 clean 结果不能证明 runtime 兼容。
