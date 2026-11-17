# Batch Invariance 单算子验证

命令示例以本 Skill 根目录（与 `SKILL.md` 同级）为工作目录。

本文档对应 `scripts/operator_validation/computation_ops/test_compute_ops_batch_invariance.py`，只说明准入、用例语义和执行规则；实现细节以脚本为准，避免复制代码后产生漂移。

## 目录

1. 判定目标
2. 目标模型场景卡
3. 用例与切分规则
4. 算子 batch 轴
5. 执行与判定

## 1. 判定目标

每条用例执行一次完整 batch，再沿业务 batch 轴拆分、乱序执行并按原顺序恢复，最终要求按位一致：

```text
op(full_batch_input) == restore(op(slice_0), op(slice_1), ...)
```

该测试只证明指定 API 在已覆盖 shape、dtype、layout、芯片和 TP 下的 batch 切分性质，不能替代端到端验收，也不能外推 PP、EP、DP 或混合并行支持。

## 2. 目标模型场景卡

对白名单外、黑名单外且位于关键路径的算子，先从目标模型配置、失败命令、实际调用代码和 profiling 整理场景卡，不得只使用脚本默认 shape。

| 类别 | 必填内容 |
| ------ | ---------- |
| 模型结构 | `model_type`、hidden/intermediate size、attention head、MoE、量化和 dtype |
| 并行与批处理 | 完整拓扑、TP、每 rank local shape、batch/sequence/token 数、block 和 page 参数 |
| 推理阶段 | prefill、chunked prefill 或 decode；明确 `B/S/T/N/D` 语义 |
| 算子调用 | Python API、非默认属性、输入输出 shape/dtype/layout、device、batch 轴和 profiling shape |

同一算子在阶段、TP、layout 或 MoE/attention 属性变化时走不同分支，应分别建 case。允许围绕真实 shape 做边界扰动，不得用缩小 shape 代替真实场景。

先从当前环境的源代码和接口签名确认实际 API。未被脚本覆盖的算子，新增 case builder 或独立测试后再执行；不得使用名称相近的算子或 `--dry-run` 代替真实 NPU 用例。

## 3. 用例与切分规则

- `--cases-per-operator N`：每个选中算子的有效用例目标数；正式准入固定为 `10000`。
- `--max-split-parts N`：随机切分的最大段数；`0` 表示最多可切到 batch 轴长度。
- `--force-singleton-split`：每个 batch 元素单独执行，覆盖最强但成本最高，只用于重点算子和受控 shape。
- `--progress-interval N`：进度输出间隔；大规模执行避免设为 `1`。
- `--empty-cache-interval N`：NPU 缓存清理间隔，用于减轻长时间运行的显存碎片。

切分必须同时满足：

1. 只切业务 batch 相关输入，共享权重、全局 KV cache 等输入保持不变。
2. 乱序执行 slice，恢复时按原始区间排序。
3. 会随 slice 改变的元数据必须同步重建，例如局部 `group_list`、TND 前缀和与 block table。
4. 多输出算子逐个恢复并比较；原地写入算子比较最终目标张量。
5. 每个 case 保存 seed、TP、场景、shape、切分区间和首个失败详情，保证可复现。

## 4. 算子 batch 轴

以下是当前脚本的语义摘要；确切场景名、参数和支持范围以脚本中的 case builder 为准。

| 算子 | batch 相关轴 | 特殊处理 |
| ------ | -------------- | ---------- |
| `torch_npu.npu_swiglu` | `x dim0` | 直接恢复输出 |
| `torch_npu.npu_scatter_pa_kv_cache` | key/value/slot mapping 的 `dim0` | 共享 KV cache，比较乱序写入后的最终 cache |
| `torch.scatter_` | 输入、index、source 的 `dim0` | 比较恢复后的原地结果 |
| `torch_npu.npu_rms_norm` | `x dim0` | weight 不切分，多输出分别比较 |
| `torch_npu.npu_moe_init_routing_v2` | token 相关输入 `dim0` | 与 unpermute 组成恢复 token 顺序的 oracle |
| `torch_npu.npu_moe_token_unpermute` | token 行及其 top-k 展开行 | 同步换算展开区间和局部 index |
| `torch_npu.npu_grouped_matmul` | sorted expert-token M 轴 | 按 slice 与 expert 区间重算局部 `group_list` |
| `torch_npu.npu_moe_gating_top_k` | router logits `dim0` | 只比较业务有效输出 |
| `npu_fused_infer_attention_score_batch_invariant` | 按 layout 为 T 或 B 轴 | TND 前缀和局部重置；paged cache 不切，block table 随请求切 |
| `npu_matmul_batch_invariant` | 左矩阵 M 轴 | 右侧共享权重不切分 |
| `npu_reduce_sum_batch_invariant` | `x dim0` | reduce 维必须与 batch 轴无关 |

Fused attention 需要额外区分：

- TND prefill：只有把每个 request span 作为独立 slice 才能代表真实请求；脚本的单 token 逻辑请求场景不能外推多 token 请求内部任意切分。
- TND/BSH paged decode：切 query、block table 和逐请求长度；全局 key/value cache 保持不变。
- BSH/BNSD prefill：query/key/value 和逐请求长度一起沿 B 轴切分。

## 5. 执行与判定

先做小规模 smoke，确认 API、shape 和证据格式；再对目标关键路径算子执行正式验证：

```bash
python3 scripts/operator_validation/computation_ops/test_compute_ops_batch_invariance.py \
  --cases-per-operator 10000 \
  --operators <实际算子注册名> \
  --max-split-parts 16 \
  --output <RUN_RECORD_DIR>/phase1/operator_validation/<operator>.json
```

需要强化切分时，对同一算子另行执行 `--max-split-parts 0` 或 `--force-singleton-split`；后者应配合 shape cap，不能因执行成本过高而中断正式 10,000 条结果后仍声明通过。

| 状态 | 条件 |
| ------ | ------ |
| `PASS` | 10,000 条有效 NPU 用例全部按位一致 |
| `FAIL` | 至少一条有效用例数值不一致 |
| `BLOCKED` | OOM、接口/环境错误或有效用例不足；修复条件后重跑 |

`--dry-run` 和语法检查只验证计划生成与接口完整性，不形成数值证据。最终报告必须列出实际 API、场景卡、有效用例数、输出文件和首个失败；MoE 或 AllReduce 单算子结果不得外推为 EP、DP 或 PP 支持。
