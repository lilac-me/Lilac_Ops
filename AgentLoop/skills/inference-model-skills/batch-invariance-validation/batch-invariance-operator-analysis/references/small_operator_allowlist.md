# 小算子替换方案

当黑名单算子命中，或白名单外算子的单算子验证不满足 batch 不变性时，参考本文件检查替换方向。以下内容只说明已知实现形态，不是按 profiling 名称直接替换的映射表。实施前必须读取目标环境当前版本的模型代码、`vllm_ascend/batch_invariant.py` 或等价注册入口，确认源 API、参数、shape、dtype、layout、芯片实现和修改位置；禁止仅在算子名后拼接 `BatchInvariant`。无法得到唯一方案时，恢复状态允许按证据强度选择最小隔离修改并重验，五轮仍不能闭环才输出恢复耗尽报告。

## 不需要替换的算子

以下常见算子已验证可直接使用，无需替换为 batch invariant 版本：

- elementwise 运算：`torch.add`、`torch.mul`、`torch.sub`、`torch.div`、`torch.exp`、`torch.log`、`torch.abs`、`torch.neg`、`torch.rsqrt`
- 三角函数：`torch.cos`、`torch.sin`
- 比较/逻辑：`torch.less`、`torch.greater`、`torch.logical_and`、`torch.logical_not`
- 数据类型转换：`torch.cast`
- shape/数据搬移：`torch.reshape`、`torch.transpose`、`torch.slice`、`torch.gather`、`torch.index_select`、`torch.tile`、`torch.broadcast_to`
- 归一化（特定路径）：`torch.nn.functional.layer_norm`（固定 normalized_shape）
- 规约（特定场景）：`torch.max`（无重复极值歧义）

## add_rms_norm → add + rms_norm

融合算子 `torch_npu.npu_add_rms_norm` 无法保证 batch 不变性，拆为两个小算子：

```python
# 原融合调用
x, residual, _ = torch_npu.npu_add_rms_norm(x, residual, weight, eps)

# 拆解后
x_ = x + residual                         # 小算子: torch.add
residual_ = x_
x_, _ = torch_npu.npu_rms_norm(x_, weight, eps)  # 小算子: npu_rms_norm
```

`torch.add` 是 elementwise，不涉及 batch 轴依赖；`npu_rms_norm` 独立执行，消除了融合 kernel 内部跨 batch 样本的计算耦合。

## softmax → softmax_batch_invariant（Triton 小算子拆解）

`torch.softmax` 替换为 Triton 实现的 `softmax_batch_invariant`，内部将 softmax 拆解为小算子序列：

```python
# softmax_batch_invariant 内部逻辑
x_max = x.max(dim=-1, keepdim=True).values  # 小算子: reduce max
x_sub = x - x_max                            # 小算子: elementwise sub
x_exp = x_sub.exp()                          # 小算子: elementwise exp
x_sum = x_exp.sum(dim=-1, keepdim=True)      # 小算子: reduce sum (batch invariant 版本)
output = x_exp / x_sum                       # 小算子: elementwise div
```

`max`、`sub`、`exp`、`div` 都是 elementwise 或无关 batch 轴的标量操作，`reduce sum` 使用 batch invariant 版本。组合后 BS=1 与 BS=N 中同位置样本结果 bitwise 一致。

## 其余算子：直接替换为 batch invariant 版本

以下算子不涉及拆解，直接 dispatch 到 batch invariant 实现：

| 原算子 | 替换算子 | 实现 |
| --- | --- | --- |
| `torch.sum` | `npu_reduce_sum_batch_invariant` | AscendC |
| `torch.matmul` | `npu_matmul_batch_invariant` | AscendC（优先），Triton 回退 |
| `torch.mm` | `npu_mm_batch_invariant` | AscendC（优先），Triton 回退 |
| `torch.bmm` | `bmm_batch_invariant` | Triton |
| `torch_npu.npu_fused_infer_attention_score` | `npu_fused_infer_attention_score_batch_invariant` | AscendC |
| `torch.nn.functional.linear` | `linear_batch_invariant` | Triton（仅 AscendC 不可用时） |
| `torch.addmm` | `addmm_batch_invariant` | Triton（代码中存在但注释掉，暂未启用） |
