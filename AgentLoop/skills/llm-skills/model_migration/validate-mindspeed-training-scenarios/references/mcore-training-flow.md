# MCore 训练流与验证入口

## 目录

1. 适用后端
2. MCore 训练调用链
3. 场景参数的实际归属
4. 在线 Checkpoint 转换
5. 数据与 batch 路径
6. 测试和故障定位入口

## 1. 适用后端

本 skill 只验证 MindSpeed-LLM MCore 训练场景。开始前至少找到一个 MCore 证据：

- launcher 位于 `examples/mcore/`；
- 入口是 `pretrain_gpt.py`、`posttrain_gpt.py` 或明确的 MCore 模型入口；
- 命令使用 `--use-mcore-models` 或 `--spec`；
- import 链进入 Megatron/MindSpeed MCore 训练循环。

以下信号表示应停止使用本 skill：

| 信号 | 后端 |
| --- | --- |
| `train_fsdp2.py`、`examples/fsdp2/`、FSDP2 YAML | FSDP2 |
| `examples/mindspore/`、`mindspeed_llm/mindspore/` | MindSpore |

不要为了复用场景脚本而修改其他后端。用户要求跨后端对比时，应先为各后端建立
独立实验合同，再定义跨后端可比指标。

## 2. MCore 训练调用链

```text
scenario wrapper
  -> common launcher
  -> examples/mcore/<model>/*.sh 的参数合同
  -> torchrun / torch.distributed.run
  -> pretrain_gpt.py / posttrain_gpt.py / 模型专用入口
  -> mindspeed_llm.megatron_adaptor
  -> mindspeed_llm.training.training.pretrain
  -> initialize_megatron
     -> 参数与并行状态初始化
     -> tokenizer / dataset
     -> 可选 HF -> MCore 在线转换
     -> checkpoint load
  -> model_provider
     -> GPTModel / custom model + --spec
  -> train_valid_test_datasets_provider
  -> forward_step
     -> tokens / labels / loss_mask / attention_mask / position_ids
  -> model forward + loss_func
  -> backward / optimizer step
  -> checkpoint save
```

场景对比必须证明两侧在预期 toggle 之前经过相同链路。若 wrapper 最终进入不同
入口、Spec、数据 provider 或 checkpoint 分支，则不是单变量实验。

## 3. 场景参数的实际归属

| 参数类别 | 首要检查位置 | 典型风险 |
| --- | --- | --- |
| TP/PP/CP/EP/ETP/SP | launcher、参数校验、parallel state 初始化 | 只比较 shell 文本，忽略运行时默认值 |
| `num-layer-list`、VPP、noop | launcher、TransformerBlock 层偏移、converter | runtime 与 checkpoint 层身份不同 |
| recompute | launcher、transformer/checkpoint wrapper | metadata 未贯穿 Pack/DSA 路径 |
| MBS/GBS | launcher、microbatch calculator | DP 改变后梯度累积不同 |
| optimizer/lr | launcher、训练参数 | toggle 对意外改变优化合同 |
| model/Spec | `model_provider`、`mindspeed_llm/tasks/models/spec/` | wrapper 指向不同架构 |
| data/Pack | dataset provider、preprocess handler、batch/forward | token 顺序、mask 或有效 token 数不同 |
| load/save | 训练初始化、checkpoint cache、保存目录 | 一侧随机初始化或重复转换 |

`--dry-run` 应输出运行时解析后的有效值，而不是仅回显 wrapper 环境变量。必要时
追踪参数默认值和 feature patch，确认打印值等于实际消费值。

## 4. 在线 Checkpoint 转换

在线 HF→MCore 通常从初始化阶段触发。场景要求加载 HF 预训练权重时，检查：

```text
--load <hf-source>
--enable-hf2mg-convert
--model-type-hf <model-type>
--mg-save-dir <converted-cache>
--save <training-output>
```

支持性必须同时覆盖：

- launcher/feature 参数解析；
- 训练初始化 dispatcher；
- converter 的模型类型；
- `configs/checkpoint/model_cfg.json`；
- 目标 Spec 和结构参数；
- cache shard 与 marker 完整性。

转换一次后，具有相同结构 identity 的场景应直接加载同一 MCore cache。若每个场景
都重新转换，先检查 cache key 是否误包含日志、端口或场景名等非结构字段。

## 5. 数据与 batch 路径

```text
preprocess_data.py / 已有 indexed dataset
  -> mindspeed_llm/tasks/preprocess/ handler
  -> tokenizer + prompt template
  -> train_valid_test_datasets_provider
  -> batch 获取
  -> tokens / labels / loss_mask / attention_mask / position_ids
  -> TP/CP/Pack/DSA/TND 适配
  -> forward_step / loss_func
```

固定长度与 Pack 改变 token 分组、mask、position 和有效样本语义，因此只能分别
检查收敛，不能当作逐 step loss 对。

Pack/DSA/TND 失败时，沿调用链记录 cumulative lengths、实际 lengths、dtype、
device、contiguity 和 flattened-token shape。不要只检查 dataset 输出或最终 kernel。

## 6. 测试和故障定位入口

| 问题 | 优先验证 |
| --- | --- |
| 参数差异或 invariant | wrapper diff + common launcher dry-run |
| 层身份或纯函数 | `tests/ut/` |
| launcher、转换或一步训练 | `tests/st/`、模型示例 ST |
| cache 完整性、日志 parser、比较器 | `tests/test_tools/` 或工具本地测试 |
| Spec/import | 目标运行环境 import 或最小模型构造 |
| checkpoint | tiny/抽样转换、strict load |
| Pack/DSA/TND tensor | 直接消费者边界的小型行为回归 |
| loss 对齐 | 合成日志边界测试后再解析真实日志 |

故障定位顺序：

1. 环境与 launch；
2. 参数解析和后端/Spec；
3. 数据与 iteration；
4. checkpoint identity、转换和 strict load；
5. 模型构造与层所有权；
6. forward、mask、tensor 与 loss；
7. optimizer/save；
8. mean loss 和收敛指标。

后续 rank 的 NPU/HCCL 错误可能是级联。始终保留并优先分析最早的 Python 或算子
异常。
