# NPU 算子支持度分析

## 目录

- 必须回答
- 官方支持度查询来源
- 非 Eager 路径扫描规则
- 算子分类
- NPU 支持矩阵
- 重点验证维度
- 风险分级
- NPU 适配建议输出
- NPU 证据隔离与整网结论

## 必须回答

1. 模型实现中哪些路径是标准 PyTorch Eager；
2. 哪些路径依赖非 Eager 实现；
3. 是否使用自定义 CUDA、Triton、FlashAttention、torch.ops、torch_npu、CANN、自定义 Autograd、Hub Kernel、融合算子或第三方库；
4. 每个非 Eager 路径是否有 NPU 前向、反向支持；
5. 是否存在 GPU 专属假设、CUDA API、动态 Shape、数据类型或布局限制；
6. 不支持时是否存在 Eager Fallback；
7. Fallback 是否会带来精度、显存或性能风险；
8. 需要补充哪些 NPU 适配、融合算子或验证用例。
9. 对非 Eager 路径的评估，需要给出 NPU 官方或开源出处；否则必须明确标注为理论分析，不能表述为已完成 NPU 真实验证。

## 官方支持度查询来源

NPU 算子支持度分析允许并优先查询以下昇腾官方文档：

- CANN 算子/API 清单：`https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/operatorlist_00001.html`
- PTA（TorchNPU）接口文档：`https://www.hiascend.com/document/detail/zh/Pytorch/latest/index/index.html`

使用原则：

- 对 PyTorch 原生接口，优先通过 PTA/TorchNPU 文档确认该接口在昇腾 NPU 上的支持情况；
- 对 CANN 基础算子、融合算子或底层算子 API，查询 CANN 算子/API 清单确认支持情况及约束；
- 支持度判断必须关注文档中的产品型号、dtype、Shape/Layout、训练前向/反向等约束，不能仅依据“存在同名接口”判定为完整支持；
- 若官方文档未给出明确支持证据，则标记为 `SX` 或理论分析，不得表述为已完成 NPU 真实验证。

## 非 Eager 路径扫描规则

必须搜索：

```text
torch.ops
torch_npu
torch.compile
torch.library
autograd.Function
cpp_extension
CUDAExtension
Triton
triton.jit
flash_attn
xformers
apex
transformer_engine
use_kernel*
use_kernelized_func
custom_op
fused
npu_*
all_to_all
reduce_scatter
all_gather
tilelang
```

同时检查：

- 装饰器注册的替换 Kernel；
- Import 失败时的 Fallback；
- `is_cuda`、`device.type == "cuda"`；
- CUDA Stream/Event；
- CUDA Graph；
- CUDA 专属 dtype/layout；
- `.contiguous()`、Stride、转置布局假设；
- 自定义分布式通信。

## 算子分类

### A. 标准 PyTorch Eager

Linear、MatMul、Softmax、LayerNorm/RMSNorm、Conv、TopK、Gather/Scatter。

仍需检查 torch_npu 是否支持对应 dtype 和 Shape。

### B. PyTorch 组合算子

- 手写 Attention；
- 手写 MoE Dispatch；
- Python 循环 Recurrent Attention；
- Dense Mask；
- 多次 Transpose + MatMul；
- 大规模 Einsum。

### C. 外部或自定义 Kernel

- FlashAttention；
- Triton；
- CUDA Extension；
- Transformer Engine；
- 自定义 Hub Kernel；
- 自定义融合算子；
- CANN/Ascend C 算子。

### D. 分布式算子

- AllReduce；
- AllGather；
- ReduceScatter；
- AllToAll；
- P2P；
- CP Ring；
- Expert Dispatch/Combine。

## NPU 支持矩阵

| 模块 | 调用接口 | 实现类型 | Forward | Backward | BF16 | FP16 | FP8 | 动态 Shape | NPU Fallback | 性能风险 | 证据 |
|---|---|---|---|---|---|---|---|---|---|---|---|

状态统一使用：

- `S0`：已原生支持并完成训练验证；
- `S1`：接口支持，但未完成目标 Shape/精度验证；
- `S2`：只有 Eager Fallback；
- `S3`：仅 Forward 或仅推理支持；
- `S4`：不支持，需要新增算子；
- `SX`：信息不足。

## 重点验证维度

每个非 Eager 算子必须检查：

1. 是否支持训练前向；
2. 是否支持训练反向；
3. 支持哪些输入 dtype：BF16/FP16/FP32/FP8；
4. 支持的输入 Shape/Layout 规范，例如 FA 的 TND 等；
5. 支持的输入 Shape 泛化范围，例如长序列等；
6. 是否支持确定性计算；

## 风险分级

| 等级 | 含义 | 处理要求 |
|---|---|---|
| P0 | 无法运行或结果错误 | 必须新增/修复 |
| P1 | 前向可用但反向、精度或分布式失败 | 阻塞训练 |
| P2 | 可通过 Eager 运行但性能不可接受 | 需要融合优化 |
| P3 | 已支持但存在边界 Shape 风险 | 增加测试 |
| P4 | 已验证可直接使用 | 保留回归用例 |

## NPU 适配建议输出

| 缺口 | 根因 | 最小可运行方案 | 高性能方案 | 需要修改的代码 | 验证用例 |
|---|---|---|---|---|---|

建议必须区分：功能适配、精度适配、性能适配、分布式适配和工程依赖适配。

## NPU 证据隔离与整网结论

NPU 支持矩阵中的每一行必须增加或隐含以下证据范围：

| 证据范围 | 含义 | 允许的结论 |
|---|---|---|
| `reference_as_is` | 用户给定源码原样执行 | 可决定原始程序整体状态 |
| `reference_fallback` | 当前源码已有或可最小改写的 Eager Fallback | 只能说明功能兜底，不能说明性能可用 |
| `external_equivalent` | vLLM、MindSpeed、SGLang 等外部等价实现 | 作为替代路径，不改变原始程序状态 |

必须遵守：

1. 先给“原始整网 As-Is 状态”，再给外部推理/训练替代路径；
2. 整网 As-Is 状态由阻塞启动或主路径的最严重缺口决定，不能用部分 Eager 算子 S1 掩盖 P0 Kernel；
3. `S0` 必须附训练环境、设备、dtype、Shape、步数或测试范围；仅有部署文档的推理路径最多标记 `S3`；
4. 同名算子、相似数学功能或独立仓库示例不等价于当前调用接口可直接替换；
5. 外部报告完成训练时，应写成“外部栈 S0”，并继续保留当前推理源码无 Backward 的 `S4`；
6. 硬编码 CUDA/NCCL、GPU 专属 dtype 或无替换的主路径自定义 Kernel 存在时，原始整网至少为 `P0/S4`；
7. 每个外部证据记录来源、版本、产品型号和查询日期；缺少任一关键约束时降为 `SX`。

---
