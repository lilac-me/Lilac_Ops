---
name: bridge-model-adaptation
description: This skill should be used when the user asks to adapt an open-source Hugging Face model into MindSpeed-Bridge, implement the Bridge, Provider, and MegatronModel modelling structure, add the corresponding recipe and example launcher scripts, verify weight import/export roundtrip, or bring up smoke training on a remote NPU environment with bridge-remote-train-run.
license: MIT
targets-repo:
  - mindspeed-bridge
---

# MindSpeed-Bridge 模型迁移适配

本 skill 负责把 Hugging Face 开源模型迁移到 MindSpeed-Bridge 仓库，实现对应的Bridge、Provider以及MegatronModel结构适配，补充相应的recipe和example启动脚本，最终交付一版能在昇腾 NPU 上拉起训练且数值正常的代码。

## 输入与输出

从用户给定的信息的获取输入：

- 开源模型名称（如 `Qwen3.5-35B-A3B`）
- 开源模型`config.json`
- 开源模型实现（Transformer源码）
- 可用的本地或远端 NPU 环境信息（无环境时只做静态交付并明确标注）。

输出（交付件）：

代码适配新增的文件，如：

```text
mindspeed_bridge
├── examples/models/<model_family>/<model>_<model_size>_<train_mode>.sh
├── models/<model_family>/
│   ├── modelling_<model>/{spec.py, model.py, attention.py, ...}
│   ├── <model>_bridge.py
│   └── <model>_provider.py
└── recipes/<model_family>/
    ├── <model_family>.py
    └── <model>_<model_size>_<train_mode>.py
```

examples中提供的启动脚本必须是全层全参脚本，需要另外提供一个不上仓的可在远端NPU环境拉起的训练脚本。
使用`$bridge-remote-train-run` 拉起该训练脚本，收集返回的训练日志和可复现步骤，总结到SUMMARY.md中，作为模型适配完成的依据。

## 工作区

在指定的位置建立`model-adaptation`工作区，默认在任务工作目录，
它是整个流程的单一事实来源，用于固化代码适配的中间产物和迭代过程，保证可追踪、可复现。工作区结构如下：

```text
model-adaptation/
├── inputs/              # 输入：开源模型名称、config.json、Transformer源码
├── errors/
│    ├── eNNN.md         # 记录遇到的问题，分析和解决方法
│    └── ...
├── iter-NNN/            # 每次迭代的工作区，NNN为迭代号
│    ├── *log            # 迭代中返回的训练日志
│    └── *.sh            # 迭代中拉起的脚本
├── status.md            # 迭代状态，包含当前适配进展，已完成的迭代等，模板见 `references/status-template.md`
└── summary.md           # 迭代总结，包含适配方案、验证脚本、验证结果, 模板见 `references/summary-template.md`。
```

如果是由`$bridge-migration-workflow`创建的工作区，`model-adaptation`已包含在`$bridge-migration-workflow`的工作区目录中。且不再单独维护`inputs/`，而是使用公共的`migration_process/inputs/`。

## 配套版本

MindSpeed-Bridge 面向昇腾平台，在 Megatron-Bridge 之上提供模型支持、精度验证与亲和优化。适配前先按当前分支核对以下依赖：

| 仓库 | 作用 |
| --- | --- |
| Megatron-Bridge | AutoBridge 机制、HF↔Megatron 权重转换、训练套件 |
| Megatron-LM | Megatron 模型后端与权重切分 |
| MegatronAdaptor | Megatron-LM 的昇腾 NPU 适配 |
| MindSpeed-Ops | 昇腾 Triton 算子优化 |
| TransformerEngineNPU | 昇腾 transformer-engine 替代 |
| flash-linear-attention-npu | 昇腾 Flash-Linear-Attention（Ascend C） |

注意分支差异：MindSpeed-Bridge `v0.3.1` 分支会加载 MindSpeed 的 `megatron_adaptor`；master 分支已移除MindSpeed，runtime 直接从 MegatronAdaptor 加载。

## 核心约定

- 大部分场景只需要修改`models`、`recipes`和`examples`范围内的代码：`mindspeed_bridge/models/<model_family>/`、`mindspeed_bridge/recipes/<model_family>/`、`mindspeed_bridge/examples/models/<model_family>/`。
- 严禁修改 `runtime/`、`scripts/` 等无关代码；唯一例外是确实需要注册新step function 时，按需修改 `runtime/entrypoints/steps.py`。
- 每个 `models/<model_family>/` 互相隔离独立；顶层 `mindspeed_bridge/models/__init__.py` 保持空 namespace，不 import 具体模型族，runtime 自动扫描 `*_bridge.py` / `*_provider.py` 注册。
- 对原生结构或算子的替换必须有开关控制，开关需在 example 启动脚本中可配置，确保不需要优化时可走原生实现。
- 若配套的 Megatron-Bridge 分支已支持该模型，文件夹命名与结构尽量与其保持一致，方便对照。
- 必须且只能使用`$bridge-remote-train-run`拉起脚本，不允许手动连接远程环境执行命令。
- 每次远程拉起脚本当作一次迭代，必须将执行的脚本归档到`iter-NNN/`中，迭代号从`001`严格递增，不允许覆盖，返回的日志结果也必须归档到`iter-NNN/`中。
- 必须遵守工作流顺序，进入工作流前确认status.md中上一工作流内容已标记完成。

## 工作流

### 1. 收集输入

从用户给定的信息收集输入，所有输入放在`model-adaptation/inputs/`，至少包括：开源模型名称、开源模型`config.json`、开源模型实现（Transformer源码）、可用的本地或远端 NPU 环境信息。

`config.json` 与 Transformers 源码可以通过Hugging Face 下载到本地；根据 `config.json` 的 `architectures` 字段在Transformers 中找到对应实现。网络不可达时可用 ModelScope 或 HF Mirror。

完成后刷新status.md的输入部分，标记当前进展为"已完成工作流 1 收集输入", 并刷新status.md中对应工作流内容。

### 2. 分析适配方案

先阅读：

- Megatron-Bridge `docs/adding-new-models.md`；
- Megatron-Bridge `skills/adding-model-support/SKILL.md`；
- MindSpeed-Bridge `docs/zh/feature/autoload_installer.md`。

结合模型结构给出方案，并额外确认：

- 目标模型是否已被配套 Megatron-Bridge 分支支持；是则优先复用原生实现，只替换部分结构/算子为 NPU 优化实现；
- 是否存在可替换为 MindSpeed-Ops、flash-linear-attention-npu 中的模块；
- 融合算子替换必须要有配置或者特性开关。

完成后刷新status.md的当前进展为"已完成工作流 2 分析适配方案"，并刷新对应工作流内容。

### 3. 完成 Provider、Bridge、modelling 适配

**Bridge**（`<model>_bridge.py`）：实现 `ModelTypeBridge`，在 `provider_bridge` 中把 HF config 转为
Megatron config 并返回 Provider，在 `mapping_registry` 中注册权重名映射。
原生映射已满足时直接复用；少量修改时继承原生类并保持类名一致；无原生实现时参考 Megatron-Bridge `skills/adding-model-support` 完成。

**Provider**（`<model>_provider.py`）：实现 `ModelTypeProvider`，在 `provide` 中完成 Megatron 模型结构组装，返回MegatronModel。
原生 Provider 满足需求时直接复用；少量修改时继承原生类并保持类名一致。

**modelling_**：`bridge.py`/`provider.py` 之外的MegatronModel结构实现（spec、model、attention 等）都放 `modelling_<model_type>/`。

完成后刷新status.md的当前进展为"已完成工作流 3 完成 Provider、Bridge、modelling 适配"，并刷新对应工作流内容。

### 4. 完成 Recipe 和 Example

**Recipe**：

- 公共训练配置放 `recipes/<model_family>/<model_family>.py`，具体配置函数放 `<model>_<model_size>_<train_mode>.py` 并在族 `__init__.py` 导出；
- 函数命名全小写（参考 `qwen35_vl_35b_a3b_sft_config`），与 Megatron-Bridge recipe 命名一致，便于 `--recipe` 引用；
- 并行度约束：`TP <= num_key_value_heads`；`EP > 1` 且 `TP > 1` 时必须 `sequence_parallel=True`；
  最少卡数 `PP * lcm(TP*CP, ETP*EP)`（MindSpeed recipe 默认 `ETP=1`，即
  `PP * lcm(TP*CP, EP)`；当 `EP` 与 `TP*CP` 满足整除关系时等于 `PP * max(TP*CP, EP)`）。
  推导：`parallel_state.py` 要求 world_size 同时被 `TP*CP*PP`（非专家部分）与
  `ETP*EP*PP`（专家部分）整除；另需 `num_experts % EP == 0`。
- **`load_weights` 必须可配置，禁止硬编码 `True`**：默认取 `HF_LOAD_WEIGHTS` 环境变量（launcher 控制），
  recipe helper 传入 `to_megatron_provider(load_weights=...)`。

**Example 启动脚本**：

- 启动脚本放在 `examples/models/<model_family>/`，命名 `<model>_<model_size>_<train_mode>_<NPU_TYPE>.sh`；
- 参考启动脚本模板 `template/launcher_template.sh` 生成；
- 禁止在脚本中使用${val:-}这种shell扩展语法获取环境变量
- examples中的脚本必须是全层全参的最优脚本

完成后刷新status.md的当前进展为"已完成工作流 4 完成 Recipe 和 Example"，并刷新对应工作流内容。

### 5. 验证

使用`$bridge-remote-train-run`以及可用的NPU环境信息，在远端NPU环境上完成以下三个验证。
验证过程中遇到的任何错误，必须第一时间记录到`errors/eNNN.md` NNN为序号，从001严格递增，不覆盖。
`eNNN.md`中必须包含报错原文，关联的迭代号，原因分析，解决方法，并标记为待刷新验证结果。
只有下一轮迭代时才能刷新上一轮迭代的验证结果。

#### 验证一：代码打通

在`examples`下生成减层单机可拉起的训练脚本，不开启任何切分（TP/PP/CP/EP=1），尽可能减少gbs，减少验证成本。
不加载权重（`HF_LOAD_WEIGHTS=False`），使用随机初始化权重。
迭代次数10~20步。
必须使用`$bridge-remote-train-run` 拉起训练脚本。

**验证目标**

训练正常拉起无报错，loss正常收敛，不为NaN值。

#### 验证二：权重双向转换

如果存在开源HF权重，参考 `Megatron-Bridge/examples/conversion/hf_megatron_roundtrip_benchmark.py` 完成`import (HF -> Megatron)` 到 `export (Megatron -> HF)`的一个roundtrip。
如果不存在开源HF权重，参考 `Megatron-Bridge/examples/conversion/create_hf_toy_model.py`将随机初始化的Megatron模型保存为HF权重后再进行roundtrip验证。
必须编写shell脚本，使用`$bridge-remote-train-run`拉起脚本，不直接运行python文件
必须在python文件中添加如下代码注入MindSpeed-Bridge才能在AutoBridge识别新适配的代码：

```python
from mindspeed_bridge.runtime.auto_register import register_mindspeed_adaptor

# MindSpeed patches must be active before importing Megatron GPU implementations.
register_mindspeed_adaptor(strict=True)
```

**验证目标**

roundtrip执行无报错

#### 验证三：加载权重拉起训练无报错，loss正常收敛无NaN值

在`examples`下生成加载权重的减层单机可拉起的训练脚本，加载开源权重或者保存的HF权重拉起训练无报错，loss收敛无NaN值。
迭代次数15步。
必须使用`$bridge-remote-train-run` 拉起该训练脚本。

**验证目标**

训练正常拉起无报错，loss正常收敛，不为NaN值

完成上述验证后刷新status.md的当前进展为"已完成工作流 5 验证"，并刷新对应工作流内容。

### 6. 总结

参考 references/summary-template.md 完成总结
