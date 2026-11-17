# 分布式与服务模式

## 修改 backend 前确认 topology

process group 是参与 rank 共享的通信上下文；rank 是一个进程身份；collective 是所有参与 rank 都必须一致进入的操作。修改通信代码前，记录当前 entry 实际使用的 global/local rank，以及 DP、TP、PP、EP、FSDP、sequence/context-parallel 和 trainer/rollout 成员关系。

NPU tensor collective 通常使用 HCCL，CUDA collective 使用 NCCL，CPU 协调可以使用 Gloo 或其他仓库原生 channel。不要盲目替换每个 `nccl` 字符串：检查 channel 传输的是 device tensor 还是 CPU metadata，以及所有参与者是否以相同顺序、使用相同 rank 列表创建 group。

保留 CUDA/NCCL 和 CPU 路径。先验证 group 创建和一个匹配的小型 collective，再验证仓库原生多 rank 目标。通用 HCCL all-reduce 只能证明环境，不能证明仓库分布式训练。

## Collective hang 与生命周期

识别信号包括超时、某个 rank 在 collective 前停止、group 创建顺序不同、shape/dtype 不匹配、endpoint 缺失，或仍有工作未完成就 teardown。

比较每个 rank 的日志并定位第一个分叉操作。必要时检查 Python stack 和昇腾 plog。确认 device binding 发生在 communicator 创建前，初始化发生在 receiver 等待前，并在未完成工作被 fence 后清理所有者负责的 group/endpoint。

不要仅凭耗时将问题归类为 HCCL hang，也不要切换到 Gloo 来隐藏失败的 NPU tensor collective。必须看到对齐的 rank 完成情况和仓库语义结果。

## Ray NPU 可见性与 placement

识别信号包括：父进程可见 NPU 而 Ray actor 看不到设备、缺少 `ASCEND_RT_VISIBLE_DEVICES`、CUDA `num_gpus` 声明，或 NPU worker 在调度器建立可见性前导入 torch。

检查 Ray node 如何公布资源，以及每个 actor/task 如何请求资源。NPU worker 通常需要在 NPU 分支中使用类似 `resources={"NPU": 1}` 的 NPU 资源请求；CUDA 资源声明保留在 CUDA 分支。不要从父进程复制由调度器管理的可见性变量，以免覆盖 actor 分配。

通过同一调度器路径启动有界 actor，绑定其逻辑设备并分配小型 NPU tensor 来验证。父进程设备可用不能证明 worker placement 或仓库 workload。

## vLLM 与训练到推理的同步

将 rollout placement、真实 NPU 生成、权重传输和 learner update 视为独立能力。确认已安装版本中的精确 vLLM/vLLM-Ascend API，不要从软件包版本号推断。

对于权重同步，识别 producer/consumer rank、TP/PP/EP/FSDP 所有权、full 与 adapter/delta update、参数名转换、bucket metadata、同步、acknowledgement 和 cache invalidation。TP gather 不表示每个 stage 或 replica 都拥有完整参数。

仓库支持独立 channel 时，将 CPU control metadata 与 NPU tensor payload 分开。NPU 路径不要要求 CUDA IPC 或仅支持 NCCL 的 handle。

验证 consumer acknowledgement，以及一个反映新权重行为的更新后 generation 或 forward。sender 完成、软件包导入或一次独立 vLLM 生成，都不能证明仓库权重同步或 learner progress。
