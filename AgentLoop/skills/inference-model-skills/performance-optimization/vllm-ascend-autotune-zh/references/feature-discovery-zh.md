# vLLM/vLLM-Ascend 全特性发现指南

## 目标

生成“当前环境真实可发现”的特性清单，而不是维护一份会过期的静态百科。不同 vLLM/vLLM-Ascend 版本、不同容器镜像、不同模型专项分支暴露的参数可能不同，因此每次全局寻优前都应重新发现。

## 必收集来源

0. **参考文档网址**
   - 默认：`https://docs.vllm.ai/projects/ascend/en/latest/`
   - 这是用户输入项，可以改成其它版本的 vLLM-Ascend 官方文档、内部文档或本地文档镜像。
   - 发现报告必须记录本次采用的 URL，避免后续把不同版本文档混在一起判断。

1. **vLLM CLI**
   - `vllm serve --help`
   - `vllm bench serve --help`
   - 如果任务涉及其它 benchmark，也收集 `vllm bench --help` 与对应子命令 help。

2. **vLLM Python 包**
   - `import vllm; print(vllm.__version__)`
   - `import vllm; print(vllm.__file__)`

3. **vLLM-Ascend Python 包**
   - `import vllm_ascend; print(vllm_ascend.__file__)`
   - `vllm_ascend/envs.py` 中的 `VLLM_ASCEND_*`
   - `ascend_config.py` 中的 additional-config 映射
   - docs 中 configuration、release_notes、model tutorials 对相关环境变量的说明
   - 显式检索 PCP、DCP、KVCacheOffload/KV cache offload/cache offload/CPU offload 相关参数与教程。

4. **benchmark 脚本**
   - 脚本内已固定的 CLI 参数
   - 可由环境变量覆盖的参数
   - 已 export 的 `VLLM_ASCEND_*`、HCCL、ASCEND、VLLM 变量

## 使用脚本发现

优先在目标容器内运行：

```bash
python scripts/discover_vllm_features.py \
  --reference-doc-url https://docs.vllm.ai/projects/ascend/en/latest/ \
  --json-out /path/to/logs/features.json \
  --markdown-out /path/to/logs/features.md
```

如果 `vllm` 不在 PATH，指定二进制：

```bash
python scripts/discover_vllm_features.py \
  --reference-doc-url https://docs.vllm.ai/projects/ascend/en/latest/ \
  --vllm-bin /path/to/vllm \
  --json-out features.json \
  --markdown-out features.md
```

如果无法 import `vllm_ascend`，但知道源码路径：

```bash
python scripts/discover_vllm_features.py \
  --reference-doc-url https://docs.vllm.ai/projects/ascend/en/latest/ \
  --ascend-root /vllm-workspace/vllm-ascend/vllm_ascend \
  --json-out features.json \
  --markdown-out features.md
```

## 手工兜底命令

当脚本无法运行时，用这些命令手工收集：

```bash
vllm serve --help
vllm bench serve --help
python -c "import vllm, pathlib; print(vllm.__version__); print(pathlib.Path(vllm.__file__).resolve())"
python -c "import vllm_ascend, pathlib; print(pathlib.Path(vllm_ascend.__file__).resolve())"
grep -Rho "VLLM_ASCEND_[A-Z0-9_]*" /path/to/vllm_ascend /path/to/vllm-ascend/docs | sort -u
grep -RIn "additional_config\|enable_\|weight_\|torchair_\|ascend_" /path/to/vllm_ascend /path/to/vllm-ascend/docs
grep -RIn "PCP\|DCP\|pcp\|dcp\|KVCacheOffload\|kvcache.*offload\|kv.*cache.*offload\|cache.*offload\|cpu.*offload" /path/to/vllm_ascend /path/to/vllm-ascend/docs
```

## 特性分类

把发现出的特性分成四类：

1. **直接可寻优**
   - 可通过环境变量或脚本参数单独切换；
   - 不改变模型权重；
   - 不改变 benchmark 工作负载；
   - 失败后可直接回滚。

2. **条件可寻优**
   - 需要模型满足特定结构，例如 MoE、MLA、量化、长上下文；
   - 需要特定卡型或通信拓扑；
   - 可能启动失败，但不会污染环境。

3. **需要另建实验**
   - 需要换权重、量化格式、并行部署架构、Ray 集群、PD 分离、外部 DP；
   - 不能和当前 benchmark 直接横向比较。

4. **只记录不测试**
   - 安全风险高；
   - 改变输出语义；
   - 与吞吐目标无关；
   - 当前脚本无法注入。

## 进入候选矩阵的判断

优先进入候选矩阵：

- TP/DP/EP、expert parallel、parallel size；
- scheduler 相关：`max-num-seqs`、`max-num-batched-tokens`、chunked prefill、DBO、PCP、DCP；
- PCP/DCP 相关：prefill chunking/pipeline、dynamic chunked pipeline parallel、chunked pipeline parallel；
- memory/cache 相关：`gpu-memory-utilization`、`max-model-len`、prefix cache、KV cache dtype/布局、KVCacheOffload、KV cache offload、cache offload、CPU offload；
- compilation 相关：`compilation-config`、cudagraph capture sizes、eager/graph 模式；
- vLLM-Ascend 通信/融合/算子相关：FlashComm、MLAPO、Fused MC2、NZ、TopK、Prefetch MLP、Matmul AllReduce；
- sampling/输出相关但不改变固定输出有效性的项。

暂缓进入候选矩阵：

- LoRA、prompt adapter、guided decoding、structured output，除非用户目标就是这些功能；
- speculative decoding，除非 draft model 或 speculative 配置已准备好；
- quantization，除非权重和量化配置已经存在；
- 多模态参数，除非 workload 是多模态；
- API server 行为参数，除非影响吞吐或 benchmark 请求路径。

## 报告要求

报告“包含所有特性”时必须说明来源：

- 参考文档网址；
- CLI help 中发现的参数数量；
- vLLM-Ascend 环境变量数量；
- additional-config 疑似键数量；
- PCP、DCP、KVCacheOffload/KV cache offload 相关发现结果；如果没有发现，也要明确写“当前环境/文档未发现”；
- 源码/docs 路径；
- 发现失败或无法确认的来源。

如果未运行全特性发现，只能说“当前 skill 内置了常见寻优项”，不能声称“包含所有 vLLM/vLLM-Ascend 特性”。
