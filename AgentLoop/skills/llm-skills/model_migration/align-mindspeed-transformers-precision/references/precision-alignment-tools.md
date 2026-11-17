# 内置精度对齐工具

本 skill 在 `scripts/precision_alignment/` 中内置一套模型无关的
Transformers/MindSpeed-LLM 全词表 logits 对齐工具。

## 工具与依赖

| 文件 | 用途 | 运行时依赖 |
| --- | --- | --- |
| `alignment_utils.py` | payload 校验、hash、layout、dtype 和输出选择 | `torch` |
| `make_fixed_input.py` | tokenizer 只执行一次，生成共享输入 | `torch`、`transformers` |
| `dump_transformers_logits.py` | 加载 HF 权重并保存完整 logits | `torch`、`transformers`；可选 `accelerate`、`safetensors` |
| `dump_logits.py` | 分布式导出 MindSpeed-LLM/Megatron logits | 目标仓库的 `mindspeed_llm`、`megatron`、`torch`、`torch_npu` 与 NPU 环境 |
| `compare_logits.py` | 在 CPU 比较两份兼容 payload | `torch` |
| `test_alignment_tools.py` | CPU 单元回归 | 上述 HF 侧依赖、`pytest` |

`dump_logits.py` 是可移植脚本，不是 MindSpeed-LLM 运行时的替代品。运行它时应
从目标 MindSpeed-LLM 仓库根目录启动，并保证该仓库及 Megatron 在当前 Python
导入路径中。不要安装或升级用户现有的 `torch`/`torch_npu` 来迁就 skill。

## 路径解析

先确定 skill 根目录，后续命令中的 `<skill-root>` 指包含本文件所属 `SKILL.md`
的目录。所有入口都支持直接脚本执行；除 MindSpeed dumper 外，也支持包模块导入。

```bash
python <skill-root>/scripts/precision_alignment/make_fixed_input.py --help
python <skill-root>/scripts/precision_alignment/dump_transformers_logits.py --help
python <skill-root>/scripts/precision_alignment/compare_logits.py --help
```

不要假定当前工作目录就是 skill 根目录。launcher 中优先将
`<skill-root>/scripts/precision_alignment/dump_logits.py` 解析为绝对路径。

## 最小工作流

生成唯一共享输入：

```bash
python <skill-root>/scripts/precision_alignment/make_fixed_input.py \
  --tokenizer-dir <hf-model-dir> \
  --prompt "Hello" \
  --output-file alignment_input.pt
```

在隔离的 Transformers 环境导出参考 logits：

```bash
python <skill-root>/scripts/precision_alignment/dump_transformers_logits.py \
  --model-dir <hf-model-dir> \
  --input-file alignment_input.pt \
  --output-file reference_logits.pt \
  --dtype bf16 \
  --attn-implementation eager
```

在 MindSpeed-LLM 设备环境中，把内置 `dump_logits.py` 作为分布式入口，并追加
目标模型正常加载所需的 `--load`、`--spec`、模型结构和并行参数：

```bash
python -m torch.distributed.run <distributed-options> \
  <skill-root>/scripts/precision_alignment/dump_logits.py \
  --input-ids-path alignment_input.pt \
  --logits-output-path candidate_logits.pt \
  --load <mcore-checkpoint> \
  <model-and-parallel-options>
```

最后在 CPU 环境比较：

```bash
python <skill-root>/scripts/precision_alignment/compare_logits.py \
  --reference-file reference_logits.pt \
  --candidate-file candidate_logits.pt \
  --report-file alignment_report.json \
  --threshold 0.98
```

旧参数别名 `--hf-file` 和 `--mindspeed-file` 仍受支持。

## 输出合同

两侧 payload 至少包含：

```text
framework: string
input_ids: int tensor [B,S]
attention_mask: 0/1 tensor [B,S]，无 padding 或右 padding
input_sha256: 共享输入合同计算的 SHA-256
logits: finite tensor [B,S,V]
metadata: dictionary
```

尽可能保存 FP32 全词表 logits。主门禁是所有有效 token 的逐 token 全词表
余弦相似度算术平均值。

## 模型扩展边界

标准 Transformers causal LM 直接使用通用 dumper。配置、checkpoint 键、运行时
结构、forward 或诊断存在差异时，在模型独占目录继承
`TransformersAlignmentAdapter`，通过 `main(adapter=...)` 注入。

MindSpeed 模型若不能使用默认 `GPTModelInfer` 和 `ForwardStep`，只实现小型
provider/forward 适配器：

```text
--alignment-model-provider my_package.alignment:model_provider
--alignment-forward-function my_package.alignment:forward_logits
--alignment-output-selector attr:logits
--alignment-logits-layout sbv
```

模型专用分支不得写回内置通用工具。只有能力对多个模型都成立且回归覆盖充分时，
才同步扩展通用目录。

## 自检

在包含 `torch`、`transformers` 和 `pytest` 的 CPU 环境运行：

```bash
python -m pytest -q \
  <skill-root>/scripts/precision_alignment/test_alignment_tools.py
```

该测试不执行真实 NPU forward。完整可用性还必须在目标 MindSpeed-LLM 环境中
验证 `dump_logits.py` 的导入、分布式初始化、checkpoint 严格加载和 writer rank
输出。
