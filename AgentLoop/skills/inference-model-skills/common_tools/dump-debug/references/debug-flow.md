# 精度定位流程（详细）

> 本文件覆盖**阶段二 Dump**（P1 → P2 → P3）；阶段一的入口分诊与 L0–L6 预检见
> `references/precheck-flow.md`，两者不重复：预检负责「确认或排除根因」，本文件负责
> 「拿到算子级/整网级证据并交付」。
>
> 定位思路遵循 infer_debug_guide 的“**先 Layer 再 OP**”，是**决策驱动的线性流程**：整网 Dump →
> 首偏定位(L0 找模块 → L1 找算子) → 保存算子输入输出 → 细粒度 tensor 核对 →
> 结论汇报（可疑算子 + dump 数据，见 final-report.md）。每一步有明确判定依据；
> 依据不足（dump 不收敛 / 无结论）时**回补 dump 证据后继续**，不靠猜测、
> 不跳过前置阶段。
>
> 说明：本文中以 `` `references/...` `` 形式给出的路径均相对于技能根目录（即 `SKILL.md` 所在目录）。

## 目录

1. 定位顺序
2. 环境确认（Dump 前置）
3. 复现与确定性判定（Dump 前置）
4. P1 分级 Dump 定位（L0 → L1）
5. P2 细粒度 tensor Dump
6. P3 结论汇报与最终输出格式
7. 环境变量变更管理
8. 操作纪律（守卫规则）
9. 常见精度异常速查

## 1. 定位顺序

1. 固定失败模型、完整并行拓扑（TP/DP/EP/DCP）、设备、prompt、seed、temperature、
   长度、prefix caching 和调度参数。
2. 记录 bad/good 各自的 `set_env` / CANN 路径与版本（`msprobe compare` 需要两套环境切换）。
3. 先复现，确认“bad 乱码 / good 正常”可复现（各发一次请求，保留响应与日志）。
4. 做确定性判定，决定后续 L0/L1 dump 是否开启确定性。
5. 按 P1 → P2 分级 dump，收敛到首个偏差算子并保存其输入输出。
6. 按 P3 输出最终交付（见 final-report.md）。

## 2. 环境确认（Dump 前置）

- 先采集两侧实际环境，别用目录名代替：
  - `bash scripts/audit_vllm_ascend_env.sh > env_<side>.txt`，记录 vllm / vllm-ascend 的
    `__file__`、版本、`git rev-parse HEAD`、是否 dirty、CANN/NNAL/ATB、driver/firmware。
  - `bash scripts/audit_msprobe_env.sh`，记录 msProbe 实际导入路径、版本、标杆注册情况。
- 确认环境中是否已安装 msprobe：
  - 检查：`which msprobe`；`msprobe --help`；`pip show mindstudio-probe`（确认安装与版本）。
  - 未安装 → 按 `references/msprobe/install_guide.md` 安装：
    - 在线：`pip install mindstudio-probe`
    - 离线：`pip install ./mindstudio_probe-{ver}-py3-none-any.whl`
    - 源码（含可选模块）：`python3 build.py -e include-mod=<模块> -e no-check=true`，
      再 `pip install mindstudio_probe*.whl`
  - ⚠️ 在线/离线基础包**不含 aclgraph_dump / nan_check / xor_checksum**；需要这些
    （尤其**图模式采集**）须源码编译把模块打入包。
- 确认本次检测的问题形态，决定是否装额外模块：
  - 若与**图模式(cudagraph)** 相关（如 分支 A：图模式 vs eager）→ 需**安装 aclgraph_dump 模块**
    （图模式/ACLGraph 整网采集），否则图模式路径的 dump 采不到/不准。
  - 低精度约束：图模式不支持 fp8/fp4 采集，建议用 fp16/bf16/fp32 常规精度。
- 校验安装：`pip show mindstudio-probe` 无输出 → 当前终端用了错误的 python 环境，
  注意 `export PATH=<安装msprobe的python>/bin:$PATH`。
- 避开 CWD 下 `profile.py` 遮蔽标准库的坑：运行目录不能放在存在 `profile.py` 的目录
  （会导致服务启动崩溃），需要 `cd` 到 vllm 目录后再启动 serve。

## 3. 复现与确定性判定（Dump 前置）

> 阶段一的入口分诊（单 curl / 仅 RL / 数据集掉点）与 L0–L6 预检见
> `references/precheck-flow.md`；本节只保留 Dump 前的复现与确定性确认，避免与预检重复。

- 收集必收信息：芯片、模型、并行(TP/DP/EP)、采样、调度、bad/good 各自的
  `set_env`/CANN 版本、失败响应样例。
- **复现前置（变量最小化，见 infer_debug_guide §3.2）**：
  - 固定随机性：`np.random.seed` / `torch.manual_seed` / `torch_npu.npu.manual_seed`；关闭 dropout。
  - 打开确定性：`torch.use_deterministic_algorithms(True)`（算子计算）+
    `export HCCL_DETERMINISTIC=TRUE`（通信）；或用 msprobe 接口
    `from msprobe.pytorch import seed_all; seed_all(seed=1234, mode=True, rm_dropout=True, is_enhanced=False)`。
- 分别启动 bad/good、**各发一次请求**，保留两份响应与日志，确认“bad 坏 / good 好”可复现。
- 若同输入多次结果漂移（不一致/抖动类），先确认确定性已开启，否则 dump 不可比。
- **确定性判定（决定后续 L0/L1 dump 开关）**：确认可复现后，先开启确定性试跑一次：
  - 若开确定性后**问题仍复现** → 与随机性/非确定性无关，bad/good 需可比，**从 L0 起 dump 就开启确定性**。
  - 若开确定性后**问题消失**（仅关确定性时坏）→ 由非确定性算子触发，后续 dump **可先不开确定性**
    （开确定性会掩蔽/规避坏路径，dump 到的不再是问题现场）。

## 4. P1 分级 Dump 定位（L0 → L1）

> 先用 L0(模块级) 找疑似模块，再用 L1(API级) 锁定具体算子。vLLM 场景建议 `level` 初始值先设
> **L0**（可保证后续 level 任意切换）；时间紧可用 `level:"mix"`(L0+L1) 一次采齐。首 token 乱时
> 只需 `step:[0]`。配置项详解见 `references/msprobe/dump/config_json_introduct.md`、采集方法见
> `references/msprobe/dump/vllm_dump_instruct.md`、`references/msprobe/developer_guide/msprobe_guide.md`（vllm-ascend）。

### P1-A：L0 模块级 Dump → 定位疑似模块

- 配置（`config.json`）：

```json
{ "task":"statistics","dump_path":"/root/workspace/dump_data","rank":[],
  "step":[0],"level":"L0",
  "statistics":{ "scope":[],"list":[],"data_mode":["all"],"summary_mode":"statistics" } }
```

- vllm-ascend 启动加：`--additional-config '{"dump_config_path": "/root/workspace/config.json"}'`
  （L0/mix 采集需 `start(model)` 传 model；vllm-ascend 已封装）。
- **确定性开关**：若 P0 判定“开确定性仍复现”，此 L0 bad/good dump **必须开启确定性**
  （保证可比）；若 P0 判定“开确定性问题消失”，则**不开确定性**，采集坏路径现场。
- bad/good 各 dump，再比对：`msprobe compare -tp <bad>/step0 -gp <good>/step0 -o <out>`
- 分析 `compare_result_rank*.csv`：排除整型 count/index 假偏差（rel≈0、res=pass）；
  只看浮点张量 / `Result!=pass`；找**首个**偏差**模块**；若前置模块输入 maxdiff=0、
  某模块输出首偏 → 分歧在该模块内部，且随层数放大。

### P1-B：L1 API 级 Dump → 锁定具体算子

- 在 P1-A 定位到的模块范围内深挖：把 `level` 改为 `"L1"`（或 `"mix"`），用
  `scope:[模块起,模块止]` / `list:[API名]` 限定；必要时 `risk_level:"CORE"`（只看高风险融合/通信算子）。
  确定性开关跟随 P0 判定：开确定性仍复现 → 开启（保证 bad/good 可比）；开确定性消失 → 不开（保留坏路径）。
- bad/good 各 dump → `msprobe compare`。
- 在模块内按 Max diff 找**首个 `Result!=pass` 的 API**（如 FIA / all_reduce / linear / matmul），
  即首个偏差的**具体算子**。
- **定位到具体算子后，必须保存该算子的输入输出**（bad/good 各一份），这是细粒度比对与结论汇报的前提。

## 5. P2 细粒度 tensor Dump（限定目标算子/模块）

**必须保存目标算子的输入输出**：配置 `task:"tensor"`，`list:[目标算子]`（或 `scope` 限模块），
`data_mode:["all"]`，bad/good 各一份，落盘保留输入输出 `.pt` 真实张量：

```json
{ "task":"tensor","dump_path":"/root/workspace/dump_fia_xxx","rank":[],"step":[0],
  "level":"L1","tensor":{ "scope":[],"list":["npu_fused_infer_attention_score"],
                          "data_mode":["all"],"summary_mode":"statistics" } }
```

- 逐元素配对比较（`torch.load` + `abs(bad-good)`，按文件名配对）。
- 需要时可用 `msprobe parse -d <dump_path> -t pt -o <out>` 将 dump 转为 `.pt/.npy` 便于分析
  （见 `references/msprobe/dump/data_parse_instruct.md`）。
- **特殊值(NaN/Inf)对比**：常规数值 diff 遇 NaN/Inf 无意义时，放弃数值比较，改做特殊值特征对比：
  - **定型方向**：good 有限、bad 含 NaN/Inf → 该算子在 bad 环境引入特殊值 = 根因/首偏算子；
    输入已含 NaN/Inf → 回溯上游找首个“有限→NaN/Inf”的点；bad/good 都有限才走数值 diff。
  - **逐元素四类规则**：`bad&good 皆 NaN` 视为相等；同号 `Inf` 视为相等；`NaN↔有限` /
    `Inf↔有限` / 异号 `Inf` 判定为不一致。
  - **mask 统计**：输出 `mismatch_mask` 定位位置，并统计
    `bad_nan/good_nan/bad_inf/good_inf/nan_mismatch/inf_mismatch` 计数。
  - **数值根因方向**：反量化 `scale/offset` 为 0 → 除零；fp16/bf16 大数相乘/归约 → 溢出 Inf；
    attention score/softmax 溢出；整型 `count/index` cast 的假 NaN 需排除。
- **判定分流（关键决策点）**：
  - 输出含 NaN/Inf 而 good 有限（或 NaN/Inf 位置/数量不一致）→ 该算子本身引入特殊值，
    直接判定为根因算子（→ P3）。
  - 输入一致、输出首偏 → 疑为**算子内核自身**数值偏差（→ P3 结论汇报，交算子/版本团队）。
  - 输入已偏、输出跟随 → **上游传播**，回溯上游找真正的首偏点，回到 P1/P2。
  - 自回归/生成步输入出现大偏 = 前 token 偏差经 KV cache **自放大**，不是新根因。

## 6. P3 结论汇报与最终输出格式

- 汇报**怀疑的具体算子**（首个偏差算子）及判定依据：
  - 输入一致、输出首偏 → 疑为**算子内核自身**数值偏差。
  - 输入已偏、输出跟随 → 已回溯到**上游首偏点**并报告该上游算子。
- **提供该算子的 dump 数据**（交付的核心产物）：
  - bad/good 各一份的**输入/输出真实张量**（`dump_xxx/step0/*/dump_tensor_data/*.pt`）。
  - `msprobe compare` 的比对结果（`compare_result_rank*.csv`）。
  - bad/good 客户端响应与生成日志（复现证据）。
  - **调用该算子的源码**（`文件:行` + 代码片段），方便算子/版本团队定位内核与复现。
- （可选）用保存的输入做**单算子复现**（CPU 参考 / bad-good 内核对比）作为附加证据，以区分
  “算子实现问题”与“多算子内存踩踏/寄存器未复位”（见 infer_debug_guide §4）。
- **最终输出格式**：按 `references/final-report.md` 的“最终输出内容”小节固定结构输出
  （定位结论 / 证据表格 / 交付数据放 `dump_output/` / 调用源码 / 备注）。
- 交付数据落盘到**运行脚本所在路径下的 `dump_output/` 目录**，包含 `data/bad|good/...*.pt`、
  `compare_result_rank*.csv`、`msprobe_config_{stat|tensor}_{bad|good}.json`、
  `serve_{bad|good}_tensor.sh`、`client.sh`。

## 7. 环境变量变更管理（L1/确定性对比必读）

L1/确定性对比常需要改 `HCCL_DETERMINISTIC`、`ASCEND_LAUNCH_BLOCKING`、`NPU_...` 等环境变量。
改前必须“记原值”，改后必须“可回退”：

1. **改前先查原值**：`echo $HCCL_DETERMINISTIC`，禁止不查直接 `export`。
2. **记入变更清单**：变量名 / 原值 / 新值 / 何时改 / 何时回退。
3. **验证后决定保留或回退**；交付 dump 数据后，确定性相关变量按需还原，保证 bad/good
   对比时环境一致、结果可比。
4. **容器重启会丢变量**：确认重启后是否已在启动脚本重新设置。
5. 变更清单模板：

```text
变量名      原值       新值        用途              状态
HCCL_DETERMINISTIC  (空)  1          L1对比开启确定性    对比完成已还原
```

> 备注：`ASCEND_LAUNCH_BLOCKING=1` 与图/cudagraph 不兼容，图模式场景勿同时开启。

## 8. 操作纪律（守卫规则）

- **方案先行、不改代码**：用户要求“先给方案”时，先分析输出方案，不立即改代码/加 dump。
- **JSON 引号**：`--additional-config`/`--compilation_config` 等 JSON 参数，
  PowerShell 单引号内双引号要 `\"` 转义，Git Bash 用单引号包裹；转义错会导致参数丢失。
- **统一 serve 脚本**：统一用已确认的 serve 脚本，改动只调整 dump/config 参数，避免引入
  与问题无关的配置漂移。
- **杀进程后查显存**：`pkill` 后确认 `npu-smi` 显存已释放再重启，僵尸占显存会 OOM。
- **SSH PTY 缓冲**：长输出会被截断/分页；小段执行、用 `grep`/`sed`/输出重定向到临时
  文件再读取；必要时用 SFTP 拉回本地分析（paramiko/scp）。
- **并发预算**：可读服务端 `Maximum concurrency` 日志推算并发档位。

## 9. 常见精度异常速查

> 进入本表前，先按 `references/precheck-flow.md` 的 L0–L6 排除「采样 / 图模式 / 权重 /
> 配置版本」四类配置性根因；本表用于**预检未命中后**的 Dump 阶段快速分流。

| 现象 | 优先排查方向 | 证据/方法 |
|------|-------------|----------|
| 图模式错、eager 对（复读/错 token） | routed/shared 归约 all-reduce 簿记 | 用例A `references/cases/01_cann_upgrade_garbled_all_reduce.md` + 单算子复现 |
| 首 token 即整段乱码（仅 CANN 环境/版本切换） | fused 算子数值偏差（all_reduce / FIA） | 分支 B + cases/ 案例库 |
| 大 Max diff 但 rel≈0、res=pass | 整型 count/index 张量（假偏差） | 只看浮点张量，排除 `tp_send_counts` 等 |
| 输入一致、输出首偏 | 算子在 bad 环境自身偏差 | tensor 级 .pt 逐元素配对比对 → 结论汇报 |
| 输入已偏、输出跟随 | 上游传播 | 回溯上游找真正的首偏点 |
| 同输入结果漂移 | 未开确定性导致 dump 对不上 | 对比前开启确定性 |
