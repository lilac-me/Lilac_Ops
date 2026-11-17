# 多机权重更新验证

HCCL、NPU IPC 和 Checkpoint 三个入口均可连接已启动的多机推理服务。服务启动、逐节点环境/容量预检及停止继续复用 [multi-node.md](multi-node.md) 中的 serving 入口；跨节点 PP 使用目标版本原生 launcher。验证脚本不另建 SSH 控制器，不把普通多机 HTTP 可达当作权重更新通过。

## 服务与控制范围

- `--server-url` 必须指向能通过 collective RPC 控制**整个目标集群**的 API。典型场景是统一 head API、其余节点 headless 的 internal DP，以及同一执行器管理的跨节点 TP/PP。所有节点继承同一份 baseline 特殊配置和修复 PR。
- hybrid/external DP 若每个 API 只管理局部 EngineCore，不能把其中一个 API 当作全局入口。需先使用部署方提供的全局控制入口，或按各个独立管理域分别执行并分别记录，不能在本脚本中盲目向多个可能重叠的 API 重复广播。
- 服务需要支持 `/get_world_size?include_dp=true/false`、sleep/wake、开发态 collective RPC、所选权重更新后端。DP>1 时还需要两个生成接口均支持 `X-data-parallel-rank` 请求头。执行前核对部署版本的路由实现（`_get_data_parallel_rank` / `get_core_engine_for_request`）和服务日志；旧版本忽略此头时先适配，不能声称已验证每个副本。
- 所有节点先从同一真实 checkpoint 启动，设置 `VLLM_SERVER_DEV_MODE=1`、sleep mode 和必要的模型配置。IPC 还需要所有接收端启用可信环境下的 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`。各节点有足够的 KV、图、运行时及 producer 缓冲容量。

## 公共参数和结果

三类入口增加以下参数：

| 参数 | 含义 |
| --- | --- |
| `--inference-nodes <head>,<worker>...` | 推理节点列表，用于核对和报告；不启动节点 |
| `--expected-world-size <N>` | 预期 TP×PP×DP worker 总数，与服务查询结果核对 |
| `--data-parallel-size <D>` | 预期 DP；省略时由全局 worker 数 / 单 DP worker 数推导 |

启动前从 serving 的实际配置和状态生成这些参数，不按“节点数×某个固定卡数”猜测 worker 总数。脚本在 sleep 前查询并核对拓扑。针对每个 DP 副本、每条 prompt 和两个接口分别采集更新前后输出，TP/PP 由对应副本内部的执行器执行。

报告的 `cluster` 记录 API 控制范围、节点、全局 worker 数、每副本 worker 数和 DP。`baseline_by_endpoint` / `post_update_by_endpoint` 的每条输出带 `dp_rank` 与 `prompt_index`，按 DP rank、提示词顺序排列；每接口应有 `DP×prompt数` 条结果。`outputs_identical_by_dp_rank` 分别记录各副本结果，任一副本的 text/token/logprob 不一致、输出缺失或重复均失败。README 按 DP 和提示词展示对照。

## HCCL：一个 producer 广播到全部节点

独立 producer 只需在一张未被推理进程占用的物理 NPU 上运行。推理集群使用 `--weight-transfer-config '{"backend":"hccl"}'`。通信组大小是 `TP×PP×DP + 1`，producer 为 rank 0，推理 worker 按目标后端的全局 DP/TP/PP rank 加 1 排列，不按本地卡数初始化。

```bash
ASCEND_RT_VISIBLE_DEVICES=2 python scripts/weight-update/test_hccl_weight_update.py \
  --server-url http://10.0.0.1:8000 --served-model-name policy \
  --inference-nodes 10.0.0.1,10.0.0.2 --expected-world-size 4 --data-parallel-size 2 \
  --checkpoint /models/policy --device 0 \
  --master-address 10.0.0.1 --master-port 29511 \
  --prompts-json prompts.json --report hccl_report.json
```

示例为每节点 TP2、全局 DP2。`master-address` 是 producer 实际所在节点、所有推理 worker 可达的地址；producer 位于第三台机器时填写第三台地址。多机不使用 `127.0.0.1`、`localhost` 或通配监听地址。各节点 HCCL 网卡/IP、端口和跨机通信按目标平台配置并归档；生产卡与推理卡仍需隔离。

## NPU IPC：每个节点本地共享，跨机同步元数据

IPC 内存不跨主机共享。每个推理 worker 对应一个**同节点、同物理 NPU**的 producer；跨机进程组只汇总 checkpoint 元数据和 IPC handle，并同步生命周期。`--devices` 是本节点 producer 的逻辑设备序列，按 LOCAL_RANK 取设备，不能用全局 RANK 索引本地设备。

从实际服务启动配置、可见设备和 worker 日志审计物理映射，保存为 `inference_devices.json` 并分发给各节点：

```json
[
  {"host": "10.0.0.1", "devices": [0, 1]},
  {"host": "10.0.0.2", "devices": [0, 1]}
]
```

`host` 是接收端 `npu_generate_uuid` 使用的主机地址，`devices` 为物理 chip ID，不是经过可见设备映射后的逻辑编号。两侧容器网络和可见设备映射必须匹配；物理映射以实际服务配置为依据，不能为了通过校验填写 producer 自己的错误映射。多机必须传 `--inference-devices-json`；producer 实际 UUID 集合须与此映射完全一致。

以下命令以 skill 目录为工作目录。两节点使用相同脚本、checkpoint、提示词、设备映射文件和 packed 大小；可在各节点保留本地相同路径的副本，无需共享内存或共享磁盘。

```bash
# 节点 0；此处推理和 producer 均使用物理卡 0、1。
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes=2 --nproc-per-node=2 --node-rank=0 \
  --master-addr=10.0.0.1 --master-port=29501 \
  scripts/weight-update/test_npu_ipc_weight_update.py \
  --server-url http://10.0.0.1:8000 --served-model-name policy \
  --inference-nodes 10.0.0.1,10.0.0.2 --expected-world-size 4 --data-parallel-size 2 \
  --node-size 2 --node-rank 0 --devices 0,1 \
  --inference-devices-json inference_devices.json \
  --checkpoint /models/policy --packed-buffer-mb 1024 \
  --prompts-json prompts.json --report ipc_report.json
```

节点 1 执行相同命令，将 torchrun 的 `--node-rank=0` 和脚本的 `--node-rank 0` 都改为 1；`--devices` 按该节点的实际逻辑设备映射填写。`--node-size` / `--node-rank` 仅核对 torchrun 环境，不代替 torchrun 创建全局 RANK、WORLD_SIZE 等变量。当前入口要求各节点 producer 数相同、WORLD_SIZE 等于推理 worker 总数。

sleep 前全局收集并检查：rank/LOCAL_RANK/节点 rank、物理 UUID 唯一性、推理设备对应关系、checkpoint manifest 和 packed buffer 大小。不同 buffer 大小会导致各 rank 的 chunk 集合通信失配，必须提前拒绝；不足以容纳最大张量也在 sleep 前失败。最终报告记录 `producer_nodes`，包括各节点、设备、UUID、checkpoint 和 buffer。收集所有节点的 torchrun stdout/stderr，主报告由全局 rank 0 写入。

## Checkpoint：全部 worker 从可读路径重载

不需要 producer 或额外训练卡。使用全局 collective RPC 对所有节点执行 `reload_weights(is_checkpoint_format=true)`；同一 checkpoint 路径必须在**每个推理节点**可读且内容一致，共享盘或预先分发均可。

```bash
python scripts/weight-update/test_reload_weights.py \
  --server-url http://10.0.0.1:8000 --served-model-name policy \
  --inference-nodes 10.0.0.1,10.0.0.2 --expected-world-size 4 --data-parallel-size 2 \
  --updated-model /models/policy --prompts-json prompts.json --report checkpoint_report.json
```

不以 RPC 返回的结果列表长度推断全局覆盖：某些 internal DP 实现会等待所有副本完成，但只返回第一个副本的结果。报告保留 `checkpoint_reload_result`；全局控制范围依据部署实现核查，数据正确性通过逐 DP 双接口对照及所有节点的服务日志证明。

任一节点初始化、传输、重载或输出校验失败，整项失败；休眠后更新未完成时不继续唤醒 KV cache，不自动重试部分权重。检查全节点日志后恢复或重启集群；底层集合通信阻塞由配置的进程组超时终止，保留首个错误和各节点现场。无多机 NPU 实测时只能记录离线检查通过，不能宣称多机传输已实测。
