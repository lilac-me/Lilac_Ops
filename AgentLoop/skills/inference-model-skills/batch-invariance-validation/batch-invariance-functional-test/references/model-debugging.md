# 模型异常定位

## 目录

1. 定位顺序
2. msprobe 工具 dump
3. 算子定位流程
4. 减层后继续 msprobe

## 1. 定位顺序

1. 固定失败模型、完整并行拓扑、设备、prompt、seed、temperature、长度、prefix caching 和调度参数。采样侧只改变或泛化 temperature，不引入其他采样参数。
2. 先读取 `scheduler-eviction.md` 并分析 attempt 日志。命中请求驱逐时停止数值定位，调整参数消除驱逐后重跑；当前流程不支持对驱逐场景给出 batch invariance 结论。
3. 读取 [Profiling 分诊规则](../../batch-invariance-operator-analysis/references/profiling-triage.md) 分析失败配置 profiling。
4. 按 `operator-batch-invariance.md` 验证关键路径上的白名单外算子；必要时验证 AllReduce。
5. 用相同参数测试 TP=1 或 TP=2，区分通信/切分路径和单卡计算路径。
6. Phase 2 token 或 logprob 不一致时，先读取当前 attempt 的 `failure_manifest.json`、scheduler JSON 和日志，按本文件“场景判断”选择静态 slice 或动态 request ID。缺少结构化失败或调度证据时先补齐，不猜测 dump 方法。
7. msprobe 无法安装、构建或加载，无法注入目标 vLLM，无法覆盖目标模块，或者模型 dump 结果不足以判断首个不一致边界时，按 `msprobe-dump-validation-cases.md` 运行与目标场景对应的 Demo 验证工具链。Demo 通过时回到模型场景检查配置和映射；Demo 失败或阻塞时修复条件后重验。重验后仍不可用才停止定位，并标记 `BLOCKED_MSPROBE_UNAVAILABLE` 或对应的 msprobe 证据阻塞状态；本流程不切换到手动 dump。
8. 只有全层稳定复现、msprobe 可用且全层采集耗时或数据量过大时，才考虑通过减层降低复现成本，并在减层模型中继续使用 msprobe 下沉到具体模块或算子边界。
9. 修复后重跑原失败场景；减层模型和 msprobe 调试配置不能作为正式验收证据。

## 2. msprobe 工具 dump

> **试验性能力说明**：msprobe dump debug 目前为试验性能力。如遇异常，欢迎提供可复现的问题场景，包括触发条件、预期与实际行为、复现步骤，以及必要的版本、配置、日志或 dump manifest；相关反馈将用于持续完善本 skill 的异常定位能力。反馈不能替代本流程的证据要求和 `BLOCKED_*` 判定。

继续 msprobe 定位前，必须从官方仓库**完整获取并通读**以下两份工具使用说明，不能只读取网页摘要、搜索片段、本节示例或历史缓存：

- [vLLM dump 指南](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/vllm_dump_instruct.md)
- [config.json 指南](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/config_json_introduct.md)

"完整获取"必须同时满足：

1. 取得两份 Markdown 原文全文，确认内容未被登录页、错误页或截断输出替代。
2. 记录官方仓库地址、实际 commit/ref、获取时间、两份文件的本地只读快照路径和 SHA-256。
3. 通读两份文件的全部章节，并根据目标环境确认 vLLM 实现、昇腾/其他硬件、Eager/图模式、msprobe 版本和配置字段是否匹配。
4. 将上述信息写入本次定位 manifest；缺少任一项时标记 `BLOCKED_MSPROBE_DOCUMENTATION_INCOMPLETE`，停止 msprobe 配置、dump、slice 和基于 dump 的后续定位，不得依据本节示例猜测参数。

对于 vLLM 昇腾场景，vLLM dump 指南将具体集成方式委托给 vLLM-Ascend 官方《[MSProbe 调试指南](https://docs.vllm.ai/projects/ascend/zh-cn/latest/developer_guide/performance_and_debug/msprobe_guide.html)》。因此，两份 msprobe 文档是最低前置材料；还必须完整获取并通读与当前 vLLM-Ascend 版本对应的该官方指南。无法取得或版本不匹配时标记 `BLOCKED_MSPROBE_VLLM_ASCEND_GUIDE_UNAVAILABLE`，不得继续 msprobe 定位。

### msprobe 安装

如果没有安装，Eager 模式安装使用如下手动安装方式（完整获取：https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/install_guide/msprobe_install_guide.md#23-%E6%BA%90%E7%A0%81%E5%AE%89%E8%A3%85）：

```bash
mkdir msprobe_code/
cd msprobe_code/
git clone https://gitcode.com/Ascend/msprobe.git
cd msprobe
python3 build.py -e include-mod=tb_graph_ascend -e no-check=true
```

如果提示 uv 缺失，安装：`curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
# 将生成在 artifacts/ 目录下的安装包拷贝到运行环境中，执行以下命令安装：
pip install mindstudio_probe*.whl
```

图模式需要的源码构建模块和产物路径随版本变化，必须以目标 vLLM-Ascend 版本官方 MSProbe 指南为准，不能固定沿用历史 `tb_graph_ascend` 构建命令。安装后记录 `pip show mindstudio-probe`、导入检查、实际版本和安装来源。

### dump 配置要点

完成文档门禁后，优先使用官方的 slice 或 request 语义切片能力，而不是对比两次独立运行的完整 dump 数据：

- 确认当前 vLLM-Ascend 已集成 `dump_config_path`；通过服务参数 `--additional-config` 或当前版本等价入口注入，并从启动日志验证实际生效。
- 指定请求采集前必须检查当前 vLLM-Ascend 源码：`_start_dump_data` 的所有真实前向调用点必须传入当前 `scheduler_output.num_scheduled_tokens`，并由该方法继续传给 `PrecisionDebugger.start(..., scheduled_tokens=...)`。仅存在 `dump_config_path` 或 `debugger.start(self.model)` 不代表支持 request 语义切片。能力缺失时，只允许在独立调试副本中按对应版本官方指南增加该适配；无法安全适配时标记 `BLOCKED_MSPROBE_REQUEST_ID_UNSUPPORTED`。
- `PrecisionDebugger.start()` 不接受 `scheduled_tokens` 时，当前官方 vLLM-Ascend 指南要求升级到 msprobe `v26.2.0` 或更高版本；升级后重新完成文档、版本和注入验证。
- 指定请求采集时，设置 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`，请求侧显式设置 `X-Request-ID`，并在配置中使用 vLLM 实际添加前缀后的 request ID。保存请求头、scheduler 的 `num_scheduled_tokens` 映射和 msprobe 实际命中的 request ID 作为同一次采集证据。
- **默认只做统计，不做 tensor dump**：msprobe dump 默认使用 `task: statistics`，只采集算子统计信息（shape、Min/Max/Mean/Norm、执行时间等），不保存完整 tensor 数据。只有在明确需要逐元素证据时，才使用 `statistics.tensor_list` 对疑似 API/模块保存完整 tensor，或使用 `task: tensor`。这是为了减少 dump 数据量和显存占用，避免不必要的性能开销。
- `summary_mode: md5` 在 PyTorch 场景实际是 CRC-32 校验，不能在报告中误称为 MD5。
- 完整 tensor 采集保持 `async_dump: false`。若确需异步 tensor dump，必须按官方约束配置 `list` 并先评估显存风险。

## 3. 算子定位流程（不知道哪个算子有问题时）

当 batch invariance 测试失败但不知道具体哪个算子有问题时，先按下述规则选择静态 slice 或动态 request ID，从 BS=N dump 中提取目标请求的算子统计，再与相同输入的 BS=1 dump 对比。不要直接对比未完成请求映射的两次完整 dump。

`msprobe-dump-validation-cases.md` 中的 Demo 用于参考配置和验证工具链，不是模型定位前的强制门禁。先直接在稳定失败的模型场景中使用对应的 slice/request ID 链路；当模型采集出现安装、注入、dump、切片、request ID 对齐或结果可信度问题时，再完整读取该文档并运行与目标场景对应的 Demo。静态 prefill/decode 分别使用 `STATIC-PREFILL-MIDDLE`、`STATIC-DECODE-MIDDLE`，动态 chunked prefill 和混合调度分别使用 `DYNAMIC-PREFILL-CHUNKED`、`DYNAMIC-MIXED-PD`。

对应 Demo 通过表示工具链在受控场景可用，应回到模型场景检查模型配置、请求映射和采集范围；Demo 失败或阻塞时，修复环境、安装、配置或证据问题后允许重新验证。重验后仍不可用时，报告对应 dump 能力未验证并标记相应 `BLOCKED_*` 状态，停止使用该链路。不得用 Demo 通过结果代替模型异常边界证据。

### 场景判断

只在静态 slice 与动态 request ID 两种方法之间选择：

**静态 slice**：必须同时满足以下条件：

- 关闭 chunked prefill 和异步调度
- scheduler 证据完整且无驱逐、抢占和活跃请求缩减
- `ignore_eos=True` 且所有请求实际生成指定的相同步数
- 日志证明全部请求在同一个 prefill step 完成，而不只是按 token budget 推算
- dump shape 证明 BS=1/BS=N 的 prefill 第 0 维分别为目标请求长度和全部请求长度之和
- 定位 decode 时，日志和 dump shape 还要证明目标请求始终处于同一 row

`slice.size` 是目标维长度相等时才触发规则的条件，不是要保留的长度。因此 P 阶段和 decode 阶段必须使用不同映射：P 阶段按 packed token 区间切片，decode 才按固定 row 切片。任一条件缺失、请求打包顺序无法确认或 shape 序列不符合预期时，标记 `BLOCKED_MSPROBE_SLICE_MAPPING`。

**动态 request ID**：任一静态条件不满足，或请求分波、分段 prefill、活跃集合/row 变化时使用。长输入和数据集评测通常属于此类，但必须依据实际调度证据确认，不能只按场景名称决定。

存在驱逐时停止数值 dump；调度证据缺失或 failure manifest 不完整时分别标记 `BLOCKED_SCHEDULER_EVIDENCE` 或 `BLOCKED_FAILURE_MANIFEST_INCOMPLETE`。静态映射证据不充分但 request ID 链路可验证时，选择动态方法并记录缺失的静态条件。

### 场景 A：静态批次定位流程

#### Step 1: 确认 step0 P 阶段映射

从 Phase 2 失败日志中找到不满足的请求，并在实际 scheduler 打包顺序中定位它。只采集 `step=0`，且必须确认该 step 是所有请求一次完成的 P 阶段，不是 decode 或分段 prefill。记录：

- 目标请求的 request ID、prompt token IDs、采样和调度参数。
- 每条请求进入模型的实际 token 数 `L0...L(N-1)`；长度必须来自最终 `prompt_token_ids`，包含 chat template、特殊 token 等全部输入，不能使用字符数或原始文本长度。
- scheduler 的实际打包顺序及各请求长度。提交顺序与实际顺序不一致时，以 scheduler 顺序为准。
- BS=N step0 中目标张量第 0 维等于 `T_total = sum(Lj)`，BS=1 step0 中该请求第 0 维等于 `Li` 的证据。

对于第 `i` 条请求，计算：

```text
begin = sum(L0 ... L(i-1))
end   = begin + Li
```

完整的静态和动态 prefill/decode dump 验证用例见 `msprobe-dump-validation-cases.md`。真实失败场景的 dump、切片或统计对齐链路出现问题时，使用其中对应 Demo 验证工具链。

#### Step 2: 配置 BS=N P 阶段 slice dump

只采集已确认的 prefill step，并配置 `slice=[{"dim": 0, "size": T_total, "begin": begin, "end": end}]`。完整配置模板见 `msprobe-dump-validation-cases.md` 的静态 Prefill Demo。

参数含义：

- `size`: BS=N P 阶段 packed-token 第 0 维长度 `T_total`；只有该维长度相等的张量才会被切片。
- `begin`/`end`: 目标请求在 packed-token 维中的半开区间 `[begin, end)`。
- `step`: 固定为 `[0]`，只获取 P 阶段的第一步数据。
- 仅将第 0 维确认为 packed-token 维且等于 `T_total` 的 input/output 作为请求级证据；权重、标量和其他 layout 不应依赖本 slice 判断请求差异。

#### Step 3: 配置 BS=1 dump

运行目标请求的 BS=1 场景（相同模型、token IDs、TP、dtype、seed、temperature 和调度参数，仅 batch size 不同），不配置 slice，并采集相同的 request-local step。

#### Step 4: 逐算子对比

先比较能按 token 维对齐的算子 input/output 的 shape、dtype、Max、Min、Mean 和 Norm，找到**第一个输入统计一致但输出统计不一致**的候选边界：

```text
边界判定规则：
- 输入统计一致且输出统计不一致 → 该算子是候选异常边界；仍需在完整 tensor 级别确认。
- 输入统计已经不一致 → 差异来自上游，继续向前追溯。
- 输入和输出统计均一致 → 继续检查下游；所有统计一致只能说明统计级证据未发现差异，不能证明逐元素或按位一致。
```

#### Step 5: 精细验证（可选）

对候选边界算子启用 `statistics.tensor_list`，保存完整 tensor，并在相同 rank、相同 input/output 位置上执行 `torch.equal`。只有完整 tensor 按位不一致时，才能将它作为确定的 P 阶段差异边界；完整 tensor 一致则继续检查下游算子。

#### Step 6: decode row slice（仅在需要 decode 定位时）

decode 的第 0 维稳定为 `BS=N`，可单独使用 `{"dim": 0, "size": <BS>, "begin": <target_row>, "end": <target_row+1>}`。该 row 配置不能用于 P 阶段；P 阶段必须使用本节的 `T_total` 和前缀和区间。

### 场景 B：动态调度定位流程

当 BS=N 场景的所有请求不能在一个 step 内完成 P 阶段时，slice 工具无法直接定位请求。此时使用 **request_id** 方式：

#### Step 1: 识别不满足的请求

从失败日志中找到不满足的请求 ID（`request_id`）和对应的不满足步数。

#### Step 2: 配置 BS=N request ID dump

设置 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`，请求侧显式设置 `X-Request-ID`，并在 msprobe 配置中指定 vLLM 实际使用的 request ID。完整配置与注入验证见 `msprobe-dump-validation-cases.md` 的动态 Demo。

#### Step 3: 配置 BS=1 dump

运行 BS=1 场景，使用相同 `request_id` 采集单条请求的算子统计。

#### Step 4: 逐算子对比

与场景 A 相同，逐算子对比输出统计，找到第一个不一致的算子边界。

### 对比结果解读

- **所有算子输出统计一致**：仅说明统计级别未发现差异；如需按位结论，仍需 tensor 级别验证。
- **找到并经完整 tensor 确认的边界算子**：该算子是 P 阶段差异的确定边界，检查其是否在 `batch_invariant.py` 中注册。
- **边界算子已注册但仍不一致**：可能是注册实现有误，或边界算子不是真正的根因（被前序算子传播），需要向前追溯

## 4. 减层后继续 msprobe (可选)

缩短目标模型 `config.json` 的 `num_hidden_layers`，确认问题仍能稳定复现。可以用 `load_format="dummy"` 降低定位成本，但该配置不能用于 Phase 2 或 Math500 正式验收。

确认减层模型仍能稳定复现后，继续通过 msprobe 的 `step`、`scope`、`list` 和 `tensor_list` 缩小范围，直到定位到具体小模块或算子边界。减层只是降低定位成本的手段，不改变 request/slice 映射、逐 rank 完整性和 tensor 证据要求。记录每次保留的层范围、输入输出长度、TP、seed、首个 mismatch step、dump 配置、dump 路径和日志路径。

当前正式定位流程覆盖 TP。PP、EP、DP 属于理论支持但未验证，其他并行方式属于可能支持但尚未分析；若在这些拓扑中定位，记录为实验性证据，不能据此修改本 skill 的支持状态。若减层后不再复现，回到完整模型的 msprobe 证据；不得用减层结果外推完整模型根因。
