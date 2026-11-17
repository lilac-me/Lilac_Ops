---
name: fsdp2-model-adaptation
description: 将新的 HuggingFace causal language model 适配到 MindSpeed-LLM FSDP2 后端，消费接入前模型分析结论并建立可供后续精度验证与性能调优的 eager baseline。适用于移植、接入或新增模型，补齐本地 modeling、模型注册、FSDP/TP/EP/CP 并行计划、FLOPS/MFU 和可执行 examples/fsdp2 启动配置的场景；模型分析由独立的 llm-model-analysis 负责，融合算子性能优化交给专用调优 Skill。
---

# FSDP2 模型适配

## 核心规则

做最小但完整的仓库内适配，让新模型可以通过现有 FSDP2 后端训练或推理。优先复用仓库已有模型、注册表、examples 和 FLOPS/MFU 结构，不新增无必要抽象。

新模型适配必须本地完整重写或 copy-adapt 到 `mindspeed_llm/fsdp2/models/<model_name>/modeling_*.py`。不要用 wrapper + monkey patch 作为主路径，不要依赖 `MindSpeedPatchesManager` 动态替换 `transformers` 类或函数。本阶段只实现 FSDP2 兼容逻辑和 EP eager 合同，不实现本模型专属性能融合。后续融合阶段若增加 fused 分支，必须写在本地 modeling 文件并保留 eager/reference fallback；这条位置约束不构成阶段二提前实现融合的授权。

## 工作流

1. 先检查仓库状态：
   - `git status --short`
   - `rg --files | rg 'fsdp2|modeling|configuration|gpt_oss|qwen|mamba|deepseek|flops|examples/fsdp2|tests/st'`
   - 阅读最接近的 FSDP2 模型实现、`model_registry.py`、`model_factory.py`、`model_loader.py`、`flops_factory.py`、`flops_base.py`、已有 examples/fsdp2 脚本。

2. 读取接入前分析并识别目标模型结构：
   - 由生命周期主控调用时，先读取 `$llm-model-analysis` 生成的接入前 `model_analysis.md` 和 `model_analysis.json`，核对其中记录的源码路径、不可变版本和 config 与本次接入输入完全一致。
   - 将报告中的模型结构、Tensor Shape、参数量、NPU 支持缺口、FLOPS 口径和并行切分建议作为设计输入；缺少报告或版本不一致时返回 `blocked`，不要自行生成简化报告替代。
   - 独立调用本 Skill 且用户未提供分析报告时，先调用 `$llm-model-analysis` 分析原始模型，再开始修改代码。
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
   - MoE 模型必须满足 EP dispatcher 的完整合同：`num_experts`、按名可取的 `gate_up_proj`/`down_proj`（堆叠布局）、`act_fn`、`forward(hidden_states, top_k_index, top_k_weights)` 签名，以及 `ep_forward(hidden_states, tokens_per_expert)`。本阶段的 `ep_forward` 必须使用 eager 专家计算；接口存在不等于允许提前调用 fused Grouped GEMM。
   - 只有当至少两个现有模型马上复用时，才新增共享 helper。

4. 建立 eager baseline：
   - 优先实现与原生 modeling 对应的 eager/reference 数值路径。
   - 不在模型接入阶段新增或启用以性能为目的的 RMSNorm、RoPE、Flash Attention、SwiGLU、Grouped GEMM、fused dispatcher，或仅为这些融合服务的 patch/hook、参数重排和 conversion mapping。基础 modeling/EP eager 权重加载所必需的 conversion mapping 可以在本阶段完成，并须在边界审计中说明用途。
   - 若 copy-adapt 的上游实现中已经存在可选 attention backend 或融合声明，保留其 eager fallback，在接入 YAML 中明确关闭，并证明阶段二没有新增可执行 fused kernel 分支。无法证明时阻塞。
   - 将融合算子接入、性能门禁和逐项接受判断交给 `$mindspeed-llm-fsdp2-fused-op-integration`。
   - 对 `input_commit..working_tree` 的新增行执行边界审计，至少搜索 `npu_fusion_attention`、`npu_rms_norm`、fused RoPE、Grouped GEMM/GMM、融合开关和 `MindSpeedPatchesManager` 注册。每个命中必须在 `adaptation_boundary_audit` 中分类；本模型新增的性能路径一律移至后续阶段。

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

7. 核对接入结果与分析结论：
   - 将适配后的 modeling、配置、FSDP2/EP/E-FSDP/CP 模块路径、FLOPS estimator 和启动配置与接入前报告逐项核对。
   - 输出差异清单，至少记录分析结论、实际实现、源码证据、差异原因和处置状态。不得静默修改或覆盖接入前报告。
   - FSDP2 封装、并行计划和 eager fallback 等不改变模型语义的差异可以解释后接受；结构、Shape、参数和计算语义差异必须阻塞，直到修正实现或由主控重新调用 `$llm-model-analysis` 生成独立的接入后报告。
   - 向主控返回适配后 modeling/config 路径、差异清单、静态检查结果和精度对齐所需参数，不在本 Skill 内重复运行接入后完整分析。
   - 生成 forward 语义不变量矩阵，把分析报告中的模块顺序映射到具体函数和源码位置。至少覆盖 attention 的 Q/K/V projection、Q/K norm、RoPE、KV repeat、mask、attention 与输出 reshape，以及 MoE router、permute、专家计算、路由权重和 unpermute；任何分析步骤在实现中缺失或调序都必须阻塞。
   - 生成结构签名，记录 module path/class、参数名/shape/dtype、tied-weight group、state_dict key 和 MoE 专家轴布局，供后续融合候选做逐字段比较。

8. 最小验证：
   - `PYTHONPYCACHEPREFIX=/private/tmp/mindspeed_pycache python3 -m py_compile <changed python files>`
   - `bash -n <changed shell scripts>`
   - 静态检查 `FlopsFactory.get_model_estimator(config)` 不会返回 `UnknownModelFlopsEstimator`。
   - 能无硬件运行的导入检查、语法检查、shell 检查都要跑。
   - 需要 NPU/分布式环境的训练验证如果本地跑不了，要在最终回复中明确说明未实测。
   - 除静态检查外，必须执行一次"候选链路就绪检查"：完整导入链（`from train_fsdp2 import Arguments, MindSpeedAutoTrainer`）、目标 YAML 的 `fsdp2_parse_args` dry-run、以及精度 harness 依赖的合同字段存在性（如 `_mindspeed_weight_load_report`、`fsdp_implementation`）。静态检查无法发现 argparse required 字段缺失、加载报告合同违约等运行时问题。
   - 将 `adaptation_boundary_audit`、forward 语义不变量矩阵和结构签名作为必需产物；不能仅在 Markdown 中写“已核对”。具体字段和后续比较方法见 `../fused-op-integration/references/correctness-gates.md`。

9. 准备精度对齐验收：
   - Reference 必须运行用户给定的原生 Transformers modeling；Candidate 必须由 MindSpeed-LLM 的 `train_fsdp2.py` 参数体系启动，并实际经过 `ModelFactory.create`、`MindSpeedParallelEngine`、FSDP2/EP 包装和 `Trainer._compute_loss` 单步前向。
   - 直接实例化适配后的 modeling 调用 forward 只能记为 standalone smoke，禁止写成 FSDP2 精度通过。
   - 默认使用 4096 个全有效 token，并比较完整词表 logits。报告必须记录 world size、FSDP/EP/TP/CP、FSDP2 wrapped module 数和 Trainer loss。
   - MoE 在加载前必须从权重索引核对实际 expert 数与 config；不一致时阻塞。核实后需要修正 config 时，必须用显式 checkpoint view override 并写入报告，禁止静默裁专家或对 HashRouter id 取模。
   - 由生命周期主控调用时，返回上述输入和配置，由主控在下一阶段调用 `$fsdp2-precision-alignment`，本 Skill 不重复执行精度验收。
   - 独立调用本 Skill 时，在接入完成后调用 `$fsdp2-precision-alignment`；不得用 standalone 报告替代真实框架验收。

## 输出要求

最终回复用中文，简要说明：

- 改了哪些文件。
- 可执行脚本路径。
- 接入前 `model_analysis.md`、`model_analysis.json` 和接入差异清单路径。
- 验证命令和结果。
- 哪些内容是理论估算，哪些由于本地环境限制没有实测。
