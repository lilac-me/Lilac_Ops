# Ascend / vLLM 错误签名

## 阅读原则

从下往上找到最后一个业务异常，再向上确认它属于哪个请求阶段。不要从最外层 `EngineDeadError` 开始猜。

## EngineDeadError

```text
vllm.v1.engine.exceptions.EngineDeadError:
EngineCore encountered an issue.
```

含义：API server 发现 EngineCore 已死亡。真正原因在上方 traceback。

后续常见现象：

- `Connection refused`
- `ReadTimeout`
- 后续请求全部 `500`
- pytest 连续多条失败

处理：只复现第一条请求，查看该请求对应的 EngineCore 日志。

## aclnnGroupedMatmulV5 / 维度不匹配

```text
RuntimeError: npu_grouped_matmul ...
call aclnnGroupedMatmulV5 failed, error code is 161002
AclNN_Parameter_Error(EZ1001):
Dim 1 value of x[0] should be equal with dim 1 value of weight[0],
but now is 2048 and 2816 respectively.
```

定位方向：

- 权重 shape 与当前 hidden states 不匹配
- MoE 权重布局或合并顺序错误
- weight reload/重映射后缓冲区内容错误
- fused MoE 的派生权重未恢复
- sleep/wake 后只恢复了一部分权重

专项检查：睡眠前同请求是否成功；失败是否只发生在 `wake_up` 后第一次推理。

## MLA prefill: x must be a 2D tensor

```text
AssertionError: x must be a 2D tensor
```

栈通常位于：

```text
vllm_ascend/attention/mla_v1.py
mla_preprocess_prefill
kv_b_proj
```

优先检查：

- 测试是否全局启用了 batch-invariant
- 输入是否被错误地增加或减少维度
- 跳过模块是否在 import 阶段修改环境变量

典型误报链：FA3 测试被跳过，但模块级设置 `VLLM_BATCH_INVARIANT=1`，污染后续 MLA 用例。

## 输出乱码但运行稳定

现象：

- `temperature=0` 下多次结果完全一致
- 文本包含大量重复符号、省略号或无意义 token

解释：确定性正常，正确性异常。优先检查：

- 模型权重是否完整
- tokenizer 与模型是否匹配
- attention/MLA 后端选择
- 量化或 dtype 配置
- CANN/vLLM-Ascend 版本组合
- 是否存在错误的环境切换目录

## 输出每次不同

先区分：

1. 默认采样导致的正常随机性。
2. 同一 `temperature=0` 下仍漂移。

第二种优先检查随机 seed、batch invariant、异步执行和未初始化状态。

## `choices` 缺失

客户端看到 `KeyError: 'choices'` 时，不要只改测试。保存原始状态码和响应体：

- 200 但无 choices：响应结构或模型输出异常
- 500 + EngineDeadError：EngineCore 已崩溃
- 空响应：连接中断或服务退出

## 服务启动超时

按顺序检查：

1. `--health` 是否曾经返回 200。
2. 模型加载耗时是否超过测试超时。
3. 日志中是否出现权重加载错误。
4. 端口是否被旧进程占用。
5. NPU 显存是否被残留 EngineCore 占用。
6. 是否错误地同时启动多个服务实例。

## 模块导入失败 / 测试跳过

- 缺少 `flash_attn_npu_v3`：FA3 不可用。
- 缺少 `vllm.distributed.weight_transfer.clients`：weight transfer 用例不可用。
- `SamplingParams` 缺字段：当前 vLLM 版本不支持该 trace replay 能力。
- `NPUWorker` 缺 `synchronize_device`：设备同步用例不可用。

跳过原因必须与当前实际代码能力一致。
