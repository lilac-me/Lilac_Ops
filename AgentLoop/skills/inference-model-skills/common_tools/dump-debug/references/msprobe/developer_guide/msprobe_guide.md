# msProbe 精度定位指南

在推理或训练过程中，我们常遇到输出偏离预期、数值行为不稳定（NaN/Inf）、预测结果与标签不再
匹配等精度异常。要定位根因，就必须监控并采集模型执行过程中产生的中间数据——特征图、权重、
激活值、各层输出。通过在特定阶段采集关键张量、记录核心层的输入输出对，并保留上下文元数据
（prompt、张量 dtype、硬件配置等），可以系统性地追查精度退化或数值误差的起点。本文介绍 AI
模型（重点面向 vllm-ascend 服务）精度问题的端到端定位流程：准备、数据采集、分析与验证。

更多资料见 [Ascend/msprobe](https://gitcode.com/Ascend/msprobe)。

## 0. 背景概念

`msprobe` 支持三种精度采集级别：

- **L0**：采集模块级张量并生成 `construct.json`，供可视化工具重建网络结构；调用时必须传入
  模型或子模块句柄。
- **L1**：仅采集 API 级统计量，适合轻量排查。
- **mix**：同时采集结构信息与算子统计量，需要同时做图重建与数值比对时使用。

## 1. 前置条件

### 1.1 安装 `msprobe`

使用 pip 安装 msProbe：

```bash
pip install mindstudio-probe
```

### 1.2 图模式 Dump（可选）

如需采集 cudagraph 图数据，需要从源码安装：

1. 从源码安装 `aclgraph_dump`：

   ```bash
   git clone https://gitcode.com/Ascend/msprobe.git
   cd msprobe
   pip install uv
   python3 build.py -e include-mod=aclgraph_dump -e no-check=true
   pip install artifacts/mindstudio_probe*.whl
   ```

## 2. 使用 `msprobe` 采集数据

采集数据整体遵循“由粗到细”的策略：先确定问题出现的 token，再决定该 token 附近需要采样的
范围。典型流程如下。

### 2.1 准备 Dump 配置内容

准备可被 `PrecisionDebugger` 解析的配置内容，以下两种方式任选其一：

- 通过 `--additional-config.dump_config` 直接传入配置对象。
- 通过 `--additional-config.dump_config_path` 传入配置文件路径。

常用字段如下：

| 字段        | 说明                                                                                                                                                                                                        | 是否必填 | Eager 模式 | 图模式 |
|:-----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------:|:-------------:|:-------------:|
| `task`      | Dump 任务类型。PyTorch 常用取值为 `"statistics"` 和 `"tensor"`：statistics 任务采集张量统计量（mean、variance、max、min 等），tensor 任务采集张量本身。 |    是    |      ✅       |      ✅       |
| `dump_path` | Dump 结果保存目录。不配置时 `msprobe` 使用默认路径。                                                                                                                                                        |    否    |      ✅       |      ✅       |
| `rank`      | 需要采集的 rank。空列表表示采集所有 rank；单卡任务必须将该字段设为 `[]`。                                                                                                                                    |    否    |      ✅       |      ✅       |
| `step`      | 需要采集的 token 迭代。空列表表示采集所有迭代。                                                                                                                                                              |    否    |      ✅       |      ❌       |
| `level`     | Dump 级别字符串（`"L0"`、`"L1"` 或 `"mix"`）。`L0` 面向 `nn.Module`，`L1` 面向 `torch.api`，`mix` 两者都采集。                                                                                                |    是    |      ✅       |      ✅       |
| `async_dump`| 是否开启异步 Dump（PyTorch `statistics`/`tensor` 任务支持）。默认 `false`。                                                                                                                                  |    否    |      ✅       |      ❌       |
| `scope`     | 需要采集的模块范围。空列表表示采集所有模块。                                                                                                                                                                 |    否    |      ✅       |      ❌       |
| `dump_enable` | 在一次训练/推理任务中动态开关 `PrecisionDebugger` 的 Dump，可在同一任务内按需开启或关闭。                                                                                                                  |    否    |      ✅       |      ❌       |
| `list`      | 需要采集的算子范围。空列表表示采集所有算子。                                                                                                                                                                 |    否    |      ✅       |      ✅       |

如需限定采集的算子，可配置 `list` 块：

- `scope`（list[str]）：在 PyTorch PyNative 场景下用于限定 Dump 范围。按工具命名规范提供两个
  模块名或 API 名即可锁定一个区间，只 Dump 两者之间的数据。示例：

  ```json
  "scope": ["Module.conv1.Conv2d.forward.0", "Module.fc2.Linear.forward.0"]
  "scope": ["Cell.conv1.Conv2d.forward.0", "Cell.fc2.Dense.forward.0"]
  "scope": ["Tensor.add.0.forward", "Functional.square.2.forward"]
  ```

  可提供的内容由 `level` 决定：`level=L0` 时提供模块，`level=L1` 时提供 API，`level=mix`
  时模块或 API 均可。

- `list`（list[str]）：自定义算子列表，可选方式包括：
    - 在 PyTorch pynative 场景下提供具体 API 的全名，只 Dump 这些 API。示例：`"list": ["Tensor.permute.1.forward", "Tensor.transpose.2.forward", "Torch.relu.3.forward"]`。
    - `level=mix` 时可提供模块名，Dump 会自动展开为该模块运行期间产生的全部数据。示例：`"list": ["Module.module.language_model.encoder.layers.0.mlp.ParallelMlp.forward.0"]`。
    - 提供子串（如 `"list": ["relu"]`）即可 Dump 所有名称包含该子串的 API；`level=mix` 时
      名称包含该子串的模块也会被展开。

配置示例：
eager 模式：

```json
{
  "task": "statistics",
  "dump_path": "/home/data_dump",
  "rank": [],
  "step": [],
  "level": "L1",
  "async_dump": false,

  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

图模式：

```json
{
  "task": "statistics",
  "level": "L1",
  "dump_path": "/home/data_dump",
  "statistics": {
    "list": []
  }
}
```

## 3. 在 vllm-ascend 中使能 `msprobe`

1. 启动 vLLM 时通过 `--additional-config` 传入 Dump 配置内容：

   ```bash
   vllm serve Qwen/Qwen2.5-0.5B-Instruct \
     --dtype bfloat16 \
     --host 0.0.0.0 \
     --port 8000 \
     --additional-config '{
       "dump_config": {
         "task": "statistics",
         "level": "L1",
         "dump_path": "/data/msprobe_dump",
         "statistics": {
           "list": []
         }
       }
     }' &
   ```

   兼容模式（沿用旧写法）仍受支持：

   ```bash
   vllm serve Qwen/Qwen2.5-0.5B-Instruct \
     --dtype bfloat16 \
     --host 0.0.0.0 \
     --port 8000 \
     --additional-config '{"dump_config_path": "/data/msprobe_config.json"}' &
   ```

## 4. 发送请求并采集 Dump

1. 按常规方式发送推理请求，例如：

   ```bash
   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "Qwen/Qwen2.5-0.5B-Instruct",
           "prompt": "Explain gravity in one sentence.",
           "max_completion_tokens": 32,
           "temperature": 0
         }' | python -m json.tool
   ```

2. 每次请求都会驱动 `msprobe: start -> forward -> stop -> step` 这一序列。runner 在每条代码
   路径上都会调用 `step()`，因此即使推理提前返回，也能拿到完整数据集。

3. Dump 文件写入 `dump_path`，通常包含：
   - 按算子/模块分组的张量文件。
   - `dump.json`，记录 dtype、shape、min/max、`requires_grad` 等元数据。
   - `construct.json`，`level` 为 `L0` 或 `mix` 时生成（可视化所需）。

   目录结构示例：
   eager 模式：

   ```text
   ├── dump_path
   │   ├── step0
   │   │   ├── rank0
   │   │   │   ├── dump_tensor_data
   │   │   │   │    ├── Tensor.permute.1.forward.pt                       # 命名格式：{api_type}.{api_name}.{call_count}.forward.{input/output}.{arg_index}。
   │   │   │   │    │                                              # arg_index 为该 API 的第 n 个输入或输出。若某个输入是列表，则用小数为下标继续编号（如 1.1 表示第一个参数的第一个元素）。
   │   │   │   │    ├── Module.conv1.Conv2d.forward.0.input.0.pt          # 命名格式：{Module}.{module_name}.{class_name}.forward.{call_count}.{input/output}.{arg_index}。
   │   │   │   │    └── Module.conv1.Conv2d.forward.0.parameters.bias.pt  # 模块参数数据：{Module}.{module_name}.{class_name}.forward.{call_count}.parameters.{parameter_name}。
   │   │   │   │                                                          # 传给 dump 的 `model` 参数为 List[torch.nn.Module] 或 Tuple[torch.nn.Module] 时，模块级数据名还会带上列表下标（{Module}.{index}.*），如 Module.0.conv1.Conv2d.forward.0.input.0.pt。
   │   │   │   ├── dump.json
   │   │   │   ├── stack.json
   │   │   │   ├── dump_error_info.log
   │   │   │   └── construct.json
   │   │   ├── rank1
   │   │   │   ├── dump_tensor_data
   │   │   │   │   └── ...
   │   │   │   ├── dump.json
   │   │   │   ├── stack.json
   │   │   │   ├── dump_error_info.log
   │   │   │   └── construct.json
   │   │   ├── ...
   │   │   │
   │   │   └── rank7
   │   ├── step1
   │   │   ├── ...
   │   ├── step2
   ```

   - `rank`：设备 ID。每张卡的数据写入对应的 `rank{ID}` 目录；非分布式场景下目录名直接为 `rank`。
   - `dump_tensor_data`：采集到的张量数据。
   - `dump.json`：各 API 或模块前向数据的统计信息，包含名称、dtype、shape、max、min、mean、
     L2 范数（L2 方差的平方根），以及 `summary_mode="md5"` 时的 CRC-32。详见
     [dump.json 文件说明](#dumpjson-文件说明)。
   - `dump_error_info.log`：仅当 Dump 工具发生错误时存在，记录失败日志。
   - `stack.json`：API/模块的调用栈。
   - `construct.json`：层次结构描述；`level=L1` 时为空。

   图模式：

   ```text
   L0_dump
   ├── step0
   │   └── rank0
   │       └── dump.json
   ├── step1
   │   └── rank0
   │       └── dump.json
   ├── step2
   │   └── rank0
   │       └── dump.json
   ├── step3
   │   └── rank0
   │       └── dump.json
   ├── step4
   │   └── rank0
   │       └── dump.json
   └── step5
       └── rank0
           └── dump.json
   ```

   - `dump.json`：详见 [dump.json 文件说明](#dumpjson-文件说明)。

## 5. 分析结果

### 5.1 前置准备

通常需要两份 Dump 数据：一份来自“问题侧”（暴露精度或数值误差的那次运行），另一份来自
“标杆侧”（正常基线）。两份数据不必完全一致——可以来自不同分支、不同框架版本，甚至不同实现
（算子替换、不同图优化开关等）。只要输入、硬件拓扑和采样点（step/token）相同或相近，
`msprobe` 就能比对它们并定位分歧节点。若找不到完全干净的基线，可以先采集问题侧数据，手工构造
最小复现用例做自比对。下文假设问题侧 Dump 为 `problem_dump`，标杆侧 Dump 为 `bench_dump`。

### 5.2 可视化

使用 `msprobe graph_visualize` 构图或图比对，再用 TensorBoard（`tb_graph_ascend` 插件）打开
生成的 `*.vis.db` 文件。

1. 确认 Dump 数据满足可视化条件：
   - Dump 级别必须是 `L0` 或 `mix`，以保证 `construct.json` 非空。
   - 每个 rank 目录应包含 `dump.json`、`stack.json` 和 `construct.json`。

2. 选择命令模式：
   - 单图构建：

     ```bash
     msprobe graph_visualize -tp <target_path> -o <output_path>
     ```

   - 图比对：

     ```bash
     msprobe graph_visualize -tp <target_path> -gp <golden_path> -o <output_path>
     ```

   - 常用可选参数：
     - `-oc` / `--overflow_check`：开启溢出标记
     - `-fm` / `--fuzzy_match`：开启节点映射的模糊匹配
     - `-lm` / `--layer_mapping [mapping.yaml]`：跨框架/跨层映射比对
     - `-tensor_log`：打印逐节点比对日志（tensor dump 场景）
     - `-progress_log`：打印详细进度日志

3. 路径粒度由 `graph_visualize` 自动识别：
   - 单 rank：`.../step0/rank0`
   - 多 rank（批量）：`.../step0`
   - 多 step（批量）：包含 `step*` 的 Dump 根路径

4. 输出文件：
   - 单图构建：`build_{timestamp}.vis.db`
   - 图比对：`compare_{timestamp}.vis.db`

5. 用输出目录启动 TensorBoard：

   ```bash
   tensorboard --logdir <output_path> --bind_all --port <optional_port>
   ```

6. 在可视化界面中查看结构与数值差异：
   - 切换 rank/step，快速定位不稳定节点。
   - 使用搜索/过滤聚焦目标算子或模块。
   - 比对模式下优先关注高差异高亮节点，并追溯其周边输入输出与参数。

## 6. 常见问题排查

- `RuntimeError: Please enforce eager mode`：重启 vLLM 并加上 `--enforce-eager` 参数。
- 没有生成 Dump 文件：确认 JSON 路径正确且各个节点都有写权限；分布式场景下设置
  `keep_all_ranks`，让每个 rank 都写入各自的 Dump。
- Dump 数据过大：先用 `statistics` 任务定位异常张量，再用
  `scope`/`list`/`tensor_list`、`filters`、`token_range` 等收敛范围。
- `TypeError: PrecisionDebugger.start() got an unexpected keyword argument 'scheduled_tokens'`：
  请将 `msprobe` 升级到 v26.2.0 或更高版本。

---

## 附录

### dump.json 文件说明

#### L0 级别

L0 级别的 `dump.json` 包含模块的前向输入输出以及参数。以 PyTorch 的 `Conv2d` 为例，网络代码为：

`output = self.conv2(input)  # self.conv2 = torch.nn.Conv2d(64, 128, 5, padding=2, bias=True)`

`dump.json` 包含以下条目：

- `Module.conv2.Conv2d.forward.0`：该模块的前向数据。`input_args` 表示位置参数输入，
  `input_kwargs` 表示关键字参数输入，`output` 保存前向输出，`parameters` 保存权重/偏置。

**说明**：传给 Dump API 的 `model` 参数为 `List[torch.nn.Module]` 或 `Tuple[torch.nn.Module]`
时，模块级名称中会带上列表下标（`{Module}.{index}.*`）。示例：`Module.0.conv1.Conv2d.forward.0`。

```json
{
 "task": "tensor",
 "level": "L0",
 "framework": "pytorch",
 "dump_data_dir": "/dump/path",
 "data": {
  "Module.conv2.Conv2d.forward.0": {
   "input_args": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      16,
      14,
      14
     ],
     "Max": 1.638758659362793,
     "Min": 0.0,
     "Mean": 0.2544615864753723,
     "Norm": 70.50277709960938,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.input.0.pt"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      32,
      10,
      10
     ],
     "Max": 1.6815717220306396,
     "Min": -1.5120246410369873,
     "Mean": -0.025344856083393097,
     "Norm": 149.65576171875,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.output.0.pt"
    }
   ],
   "parameters": {
    "weight": {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      5,
      5
     ],
     "Max": 0.05992485210299492,
     "Min": -0.05999220535159111,
     "Mean": -0.0006165213999338448,
     "Norm": 3.421217441558838,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.weight.pt"
    },
    "bias": {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32
     ],
     "Max": 0.05744686722755432,
     "Min": -0.04894155263900757,
     "Mean": 0.006410328671336174,
     "Norm": 0.17263513803482056,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.bias.pt"
    }
   }
  }
 }
}
```

#### L1 级别

L1 级别的 `dump.json` 记录 API 的前向输入输出。以 PyTorch 的 `relu` 为例
（`output = torch.nn.functional.relu(input)`），文件包含：

- `Functional.relu.0.forward`：该 API 的前向数据。`input_args` 为位置参数输入，
  `input_kwargs` 为关键字参数输入，`output` 保存前向输出。

```json
{
 "task": "tensor",
 "level": "L1",
 "framework": "pytorch",
 "dump_data_dir":"/dump/path",
 "data": {
  "Functional.relu.0.forward": {
   "input_args": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 1.3864083290100098,
     "Min": -1.3364859819412231,
     "Mean": 0.03711778670549393,
     "Norm": 236.20692443847656,
     "requires_grad": true,
     "data_name": "Functional.relu.0.forward.input.0.pt"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 1.3864083290100098,
     "Min": 0.0,
     "Mean": 0.16849493980407715,
     "Norm": 175.23345947265625,
     "requires_grad": true,
     "data_name": "Functional.relu.0.forward.output.0.pt"
    }
   ]
  }
 }
}  
```

#### mix 级别

`mix` 级别的 `dump.json` 同时包含 L0 与 L1 级别数据，文件格式与上述示例一致。
