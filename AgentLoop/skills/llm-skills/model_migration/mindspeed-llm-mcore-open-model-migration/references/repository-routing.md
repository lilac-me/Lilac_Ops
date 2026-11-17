# MindSpeed-LLM 仓库路由

## 目录

1. 后端判定
2. MCore 核心调用链
3. 模型迁移代码地图
4. Checkpoint 转换路由
5. 数据预处理路由
6. 测试与验证路由
7. 最小适配顺序

## 1. 后端判定

修改代码前，先根据请求、入口脚本和示例目录确认后端。

| 请求或仓库信号 | 目标流程 | 主要入口 |
| --- | --- | --- |
| `mcore`、Megatron、`pretrain_gpt.py`、`posttrain_gpt.py`、`examples/mcore`、`--use-mcore-models`、`--spec` | MCore | `pretrain_gpt.py`、`posttrain_gpt.py`、`mindspeed_llm/tasks/models/spec/` |
| FSDP2、YAML、`train_fsdp2.py`、`examples/fsdp2` | FSDP2 | `train_fsdp2.py`、`mindspeed_llm/fsdp2/` |
| MindSpore、`examples/mindspore` | MindSpore | `mindspeed_llm/mindspore/` |
| checkpoint、hf2mcore、mcore2hf、hf2mg、mg2hf | Checkpoint 转换 | `convert_ckpt_v2.py`、`mindspeed_llm/tasks/checkpoint/` |
| preprocess、`prompt-type`、`handler-name`、Pack、SFT 数据 | 数据预处理 | `preprocess_data.py`、`mindspeed_llm/tasks/preprocess/` |

本 skill 只处理 MCore 模型迁移。若请求明确指向 FSDP2 或 MindSpore，应停止套用
MCore 路径，改用对应后端流程。Checkpoint 或数据任务若服务于 MCore 迁移，则按
本 reference 的转换和预处理路由处理。

不要仅因文件名含 `train`、模型目录相似或参数名相同，就假定属于同一后端。
以实际启动入口、import、`--spec` 和示例目录为证据。

## 2. MCore 核心调用链

理解以下调用链后再决定修改位置：

```text
examples/mcore/<model>/*.sh
  -> torchrun / torch.distributed.run
  -> pretrain_gpt.py / posttrain_gpt.py / 模型专用 pretrain 入口
  -> mindspeed_llm.megatron_adaptor 注册 Megatron/MindSpeed patch
  -> mindspeed_llm.training.training.pretrain
  -> initialize_megatron
     -> 可选数据预处理
     -> 可选 HF -> MCore 在线转换
  -> model_provider
     -> GPTModel / MambaModel / custom model
     -> --spec 装配 attention、MLP/MoE、norm 和 transformer layer
  -> train_valid_test_datasets_provider
  -> forward_step
     -> tokens / labels / loss_mask / attention_mask / position_ids
  -> model(..., labels=labels, loss_mask=loss_mask)
  -> loss_func
  -> backward / optimizer step
  -> checkpoint 保存
  -> 可选 MCore -> HF 转换
```

按故障所在阶段选择修改边界：

- launcher 参数或路径错误：修改 `examples/mcore/<model>/`；
- 模块装配不等价：修改目标系列 Spec；
- GPTModel 无法表达结构：增加模型专用 transformer 或模型入口；
- 权重加载失败：检查转换映射、架构参数和层身份；
- token/loss mask 错误：检查数据 handler、batch/forward 和任务入口；
- 保存/导出失败：检查 checkpoint 保存格式和反向转换。

## 3. 模型迁移代码地图

| 层次 | 主要路径 | 迁移职责 |
| --- | --- | --- |
| 示例 | `examples/mcore/<model>/` | 数据、转换、预训练、微调、生成、评估和 ST launcher |
| 训练入口 | `pretrain_gpt.py`、`posttrain_gpt.py`、模型专用入口 | provider、dataset、batch、forward 和 loss |
| 训练循环 | `mindspeed_llm/training/` | 初始化、参数、tokenizer、LoRA、保存/加载和在线转换 |
| Model Spec | `mindspeed_llm/tasks/models/spec/` | 系列级模块装配和 sharded key map |
| Transformer | `mindspeed_llm/tasks/models/transformer/` | attention、MLP/MoE、norm、MLA/DSA 等实现 |
| 模型实现 | `mindspeed_llm/core/models/` | GPTModel 无法覆盖的定制模型 |
| 数据 | `preprocess_data.py`、`mindspeed_llm/tasks/preprocess/` | 原始数据、SFT/RLHF handler、Pack 和 tokenizer |
| 模板 | `configs/finetune/templates.json` | prompt/chat template |
| Checkpoint | `convert_ckpt_v2.py`、`configs/checkpoint/model_cfg.json`、`mindspeed_llm/tasks/checkpoint/` | HF↔MCore 配置、键和 shard 映射 |
| 测试 | `tests/ut/`、`tests/st/`、`tests/test_tools/` | 单元、系统和工具回归 |

仓库版本可能调整路径。若表中的路径不存在，使用 `rg --files` 和符号搜索定位同等
职责的当前实现，不凭记忆创建旧路径。

## 4. Checkpoint 转换路由

新 MCore 迁移优先检查 `convert_ckpt_v2.py`，再跟踪其实际调用到
`mindspeed_llm/tasks/checkpoint/` 的路径。

执行顺序：

1. 在 `configs/checkpoint/model_cfg.json` 检查是否已有目标模型类型；
2. LLaMA-like 模型优先继承已有基础配置，只覆盖真实差异；
3. 审计 `config_set_value`、HF 配置键和模型权重键映射；
4. 新增 `model-type-hf` 时，检查所有会再次解析该参数的 parser；
5. 需要 `--enable-hf2mg-convert` 时，检查训练初始化和 feature validation；
6. 用 tiny 或真实 checkpoint 验证 HF→MCore；
7. 用户要求导出时，再验证 MCore→HF；
8. 对两个方向比较所有结构参数和并行切分参数。

不要把 parser 接受模型名称视为转换已支持。必须同时存在配置、键映射、运行时
构造和在线转换注册。

## 5. 数据预处理路由

| 数据场景 | 优先路径 |
| --- | --- |
| 普通预训练文本 | `GeneralPretrainHandler`、`--json-keys text`，按需 `--append-eod` |
| Alpaca 风格 SFT | `AlpacaStyleInstructionHandler` + `--prompt-type` |
| ShareGPT 风格 SFT | `SharegptStyleInstructionHandler` + `--prompt-type` |
| DPO/RM 成对数据 | Alpaca/ShareGPT pairwise handler |
| 已有 LLaMA-Factory metadata | `LlamaFactoryInstructionHandler` |
| 仅 chat 语法不同 | 优先新增或配置 prompt template |
| 原始样本结构无法映射 | 再新增或扩展 handler |

先修改模板，后新增 handler。只有数据字段和语义无法被现有 handler 表达时，才扩大
预处理代码范围。

最小验证命令应使用目标仓库实际参数；典型形态为：

```bash
python preprocess_data.py \
  --input <raw-data> \
  --tokenizer-name-or-path <hf-model-or-tokenizer> \
  --output-prefix <output-prefix> \
  --handler-name <handler> \
  --tokenizer-type PretrainedFromHF \
  --prompt-type <prompt-type> \
  --workers 1 \
  --log-interval 1
```

验证生成的 `.bin/.idx`、tokenizer 和 prompt template 与训练 launcher 一致。

## 6. 测试与验证路由

| 验证目标 | 首选位置或方法 |
| --- | --- |
| 纯函数、层身份、参数映射 | `tests/ut/` |
| 转换、launcher、单步训练 | `tests/st/` 或模型示例 ST |
| 比较器、转换工具、静态审计 | `tests/test_tools/` 或对应工具测试 |
| Spec import | 目标环境 import 或最小模型构造 |
| 数据预处理 | 单 worker、小样本 `.bin/.idx` 生成 |
| Checkpoint | tiny/抽样转换、strict load、按需反向导出 |
| 端到端 | 一步 optimizer step，再扩大到目标配置 |

测试路径同样以当前仓库为准。新增回归应放在最窄、最接近缺陷所有权的位置。

## 7. 最小适配顺序

对开源因果语言模型依次判断：

1. 复用 `pretrain_gpt.py` 和 Megatron Core `GPTModel`；
2. 通过 launcher 参数表达标准结构；
3. 新增或复用目标系列 `--spec`；
4. 增加 checkpoint 配置和键映射；
5. 仅在 chat/data 语义需要时增加 template 或 handler；
6. 仅当 GPTModel + Spec 无法表达架构时增加模型专用入口或 custom model；
7. 仅当行为真正跨模型通用时修改共享框架。

每扩大一级修改范围，都要记录为什么上一层无法表达，并搜索受影响调用者。
