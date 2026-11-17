---
name: fsdp2-model-adaptation
description: 将新的 HuggingFace causal language model 适配到 MindSpeed-LLM FSDP2 后端。适用于用户要求移植、适配、接入、新增模型、一键适配模型、补齐融合算子、补齐 FLOPS/MFU、生成 FSDP2 启动脚本，或输出中文模型分析报告的场景。适配内容包括本地完整 modeling 重写、模型注册、FSDP/TP/EP/CP 并行计划、常用 NPU 融合算子、模型级 FLOPS/MFU 估算和可执行 examples/fsdp2 脚本；模型分析报告直接调用同一 Skill 套件中的 fsdp2-model-analysis 生成。
targets-repo:
  - mindspeed-llm
---

# FSDP2 模型适配

## 核心规则

做最小但完整的仓库内适配，让新模型可以通过现有 FSDP2 后端训练或推理。优先复用仓库已有模型、注册表、examples、融合算子和 FLOPS/MFU 结构，不新增无必要抽象。

新模型适配必须本地完整重写或 copy-adapt 到 `mindspeed_llm/fsdp2/models/<model_name>/modeling_*.py`。不要用 wrapper + monkey patch 作为主路径，不要依赖 `MindSpeedPatchesManager` 动态替换 `transformers` 类或函数。融合算子、EP hook、FSDP2 兼容逻辑都要写在本地 modeling 文件中，并保留 eager/reference fallback。

## 工作流

1. 先检查仓库状态：
   - `git status --short`
   - `rg --files | rg 'fsdp2|modeling|configuration|gpt_oss|qwen|mamba|deepseek|flops|examples/fsdp2|tests/st'`
   - 阅读最接近的 FSDP2 模型实现、`model_registry.py`、`model_factory.py`、`model_loader.py`、`flops_factory.py`、`flops_base.py`、已有 examples/fsdp2 脚本。

2. 识别目标模型结构：
   - 读取 `config.json`、HuggingFace modeling/configuration 源码和本地相似模型。
   - 明确 `model_type`、`architectures`、`hidden_size`、层数、attention heads、kv heads、head dim、intermediate size、MoE expert 数、top-k、shared experts、vocab、max position、RoPE、attention 类型、特殊结构字段。
   - 对 DeepSeek-V4/DeepSeek-V4-Flash 这类带 per-layer schedule 的模型，必须读取真实 checkpoint/config 的 `compress_ratios`、`compress_rate_csa`、`compress_rate_hca`、`layer_types`、`mlp_layer_types`、`num_hash_layers`。`compress_ratio=0` 是 `sliding_attention`，不是 CSA/HCA；`compress_ratio=4` 是 CSA；`compress_ratio=128` 是 HCA。若原始 `compress_ratios` 长度大于 `num_hidden_layers`，按模型构建逻辑截取前 `num_hidden_layers` 层，并在报告中说明。
   - 对 Hash-MoE 层必须写出 `tid2eid[input_ids]` 固定选择 expert 的结构；不要把 `hash_moe` 写成普通 TopK-MoE。报告、流程图、FLOPS/MFU 公式都要区分 HashRouter 和 TopKRouter。
   - 如果目标模型有 MLA/CSA/HCA、MTP、shared expert、grouped output projection、sliding window、稀疏 attention 等特殊结构，报告和 FLOPS 公式必须单独体现，不能套用普通 Qwen/Llama 公式。

3. 本地重写模型：
   - 在 `mindspeed_llm/fsdp2/models/<model_name>/` 下添加 `modeling_*.py`，必要时添加本地 `configuration_*.py`。
   - 保持 HuggingFace 上游结构可识别，但所有 FSDP2 修改都落在本地文件里。
   - 更新 `mindspeed_llm/fsdp2/models/model_registry.py`，注册稳定的 `model_id`。
   - 更新 `mindspeed_llm/fsdp2/utils/arguments.py` 的 `model_id` Literal。
   - 对齐 `fsdp_modules`、`tp_colwise`、`tp_rowwise`、`ep_modules`、`ep_fsdp_modules`、`recompute_modules` 的模块路径。
   - 只有当至少两个现有模型马上复用时，才新增共享 helper。

4. 接入常用融合算子：
   - RMSNorm：仅当语义是无 bias、最后一维 RMS 方差、乘 weight 时，用 `torch_npu.npu_rms_norm`，否则保留 eager。
   - RoPE：确认 rotate-half 约定和 cos/sin broadcast 维度后，才用 `torch_npu.npu_rotary_mul`，否则保留 eager。
   - SwiGLU/MLP：gate/up 顺序和 bias 语义一致时，才用 `torch_npu.npu_swiglu` 或 grouped GEMM。
   - MoE grouped GEMM：参考 GPT-OSS/Qwen3-MoE 模式，使用 token permute、tokens_per_expert、grouped matmul、activation、down projection、unpermute；EP 场景下参数可能是 `DTensor`，需要 `to_local()`，并提供与 dispatcher 兼容的 `ep_forward(hidden_states, split_list)`。
   - Flash Attention：确认 Q/K/V shape 顺序、causal/sliding-window mask、dropout、scale、特殊 attention 字段后再接入。
   - 所有融合路径都必须由现有参数开关控制，例如 `use_fused_rmsnorm`、`use_fused_rotary_pos_emb`、`use_flash_attn`、`use_fused_swiglu`、`moe_grouped_gemm`、`ep_dispatcher == "fused"`。

5. 补齐 FLOPS/MFU：
   - 新增或扩展 `mindspeed_llm/fsdp2/utils/flops/<model>_flops.py`。
   - 实现 `BaseFlopsEstimator.calculate_achieved_flops(tokens_sum, batch_seqlens, delta_time)`。
   - 在 `flops_factory.py` 中导入 estimator，并用真实 `config.model_type` 注册。
   - 训练日志已有链路是 `TrainMonitor -> FlopsCounter -> FlopsFactory -> BaseFlopsEstimator`，不要新增第二套 MFU logger。
   - Dense linear FLOPS 使用 `6 * tokens_sum * in_dim * out_dim`。
   - MoE expert FLOPS 使用 active top-k expert，不用全量 expert。
   - Attention QK/AV FLOPS 必须按真实 attention 类型计算。普通 full attention 才使用 `sum(seqlen ** 2 for seqlen in batch_seqlens)`；sliding attention 使用 local window pair；CSA/HCA 使用 local window pair + compressed KV pair，并按真实 per-layer schedule 统计层数。
   - HashRouter 的 `tid2eid[input_ids]` 是查表，不按 matmul FLOPS 计；learned router score 仍按 `6 * token_count * hidden_size * routed_expert_count` 计入 FLOPS。
   - Embedding lookup 默认不算 matmul FLOPS；LM head 在未 tie 或实际参与训练时计算。
   - 公式必须和报告里的 module 级公式一致。

6. 生成可执行启动脚本：
   - 在 `examples/fsdp2/<model>/` 下提供 `.yaml` 和 `.sh`。
   - 脚本风格对齐已有 FSDP2 examples，包含 `env_config.sh`、`torchrun`、日志目录、可覆盖的模型路径和数据路径。
   - YAML 中必须覆盖 `model`、`data`、`parallel`、`training`、`optimization`，并写出 FSDP/EP/E-FSDP/CP 的模块路径。
   - 只有用户明确要求 ST 时才新增 `tests/st` 文件；没有真实基线时必须说明是 placeholder。

7. 调用模型分析 Skill：
   - 模型代码、配置、FLOPS 估算和启动脚本完成后，必须调用同一 Skill 套件中的 `fsdp2-model-analysis` 生成中文模型分析报告，不在本 Skill 中复制或重新实现模型分析规则。
   - 调用时传入原生与适配后 modeling 路径、`config.json`、权重路径、模型层级结构、训练 dtype、micro batch、sequence length、optimizer、recompute 设置、允许的并行维度、单卡显存、目标昇腾产品及 CANN/torch_npu 版本。
   - 将已适配的融合算子开关、FSDP2/EP/E-FSDP/CP 模块路径、FLOPS estimator 公式和可执行环境证据一并交给分析 Skill，供其完成结构图、逐 module shape、理论显存、FLOPS、通信量、最优切分和 NPU 算子支持度分析。
   - 报告路径和结构图路径由用户指定；用户未指定时，传入目标 modeling 文件同级目录作为输出目录。
   - `fsdp2-model-analysis` 的完整工作流、公式、状态定义、报告结构和验收规则是模型分析的唯一事实来源。两者要求冲突时以模型分析 Skill 为准，禁止在本 Skill 中维护第二份分析规范。
   - 模型分析 Skill 不可用或执行失败时，将模型适配标记为未完成并报告阻塞原因，禁止生成简化报告冒充正式分析结果。
   - 适配流程只负责确认报告与结构图产物存在并记录路径，不自行改写分析结论。

8. 最小验证：
   - `PYTHONPYCACHEPREFIX=/private/tmp/mindspeed_pycache python3 -m py_compile <changed python files>`
   - `bash -n <changed shell scripts>`
   - 静态检查 `FlopsFactory.get_model_estimator(config)` 不会返回 `UnknownModelFlopsEstimator`。
   - 能无硬件运行的导入检查、语法检查、shell 检查都要跑。
   - 需要 NPU/分布式环境的训练验证如果本地跑不了，要在最终回复中明确说明未实测。

9. 精度对齐验收：
   - Reference 必须运行用户给定的原生 Transformers modeling；Candidate 必须由 MindSpeed-LLM 的 `train_fsdp2.py` 参数体系启动，并实际经过 `ModelFactory.create`、`MindSpeedParallelEngine`、FSDP2/EP 包装和 `Trainer._compute_loss` 单步前向。
   - 直接实例化适配后的 modeling 调用 forward 只能记为 standalone smoke，禁止写成 FSDP2 精度通过。
   - 默认使用 4096 个全有效 token，并比较完整词表 logits。报告必须记录 world size、FSDP/EP/TP/CP、FSDP2 wrapped module 数和 Trainer loss。
   - MoE 在加载前必须从权重索引核对实际 expert 数与 config；不一致时阻塞。核实后需要修正 config 时，必须用显式 checkpoint view override 并写入报告，禁止静默裁专家或对 HashRouter id 取模。
   - 详细执行与门禁遵循同目录 `fsdp2-precision-alignment` Skill，适配完成后不得用旧的 standalone 报告替代真实框架验收。

## 输出要求

最终回复用中文，简要说明：

- 改了哪些文件。
- 可执行脚本路径。
- 中文报告路径和结构图路径。
- 验证命令和结果。
- 哪些内容是理论估算，哪些由于本地环境限制没有实测。
