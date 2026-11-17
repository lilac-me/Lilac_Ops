---
name: ascend-perf-tuning
description: "This skill should be used when the user asks to tune Ascend NPU performance, diagnose performance degradation or instability, investigate host-bound behavior, slow ranks, communication bottlenecks, or optimize an inference service using a decision-driven workflow."
version: 1.0.0
domain: inference-model
function: performance-optimization
tags:
  - ascend
  - performance
  - diagnosis
allowed-tools: "Read, Bash(py -3:*) , Bash(curl:*) , Bash(ssh:*) , PowerShell, Skill, Agent, Glob, Grep, TodoWrite, WebFetch, WebSearch, TaskStop, TaskOutput, AskUserQuestion, Monitor"
hooks:
  PreToolUse:
    # --- 1. 禁止自行写脚本 ---
    - tools: ["Write", "Edit"]
      action: deny
      message: |
        性能调优 skill 禁止自行编写或修改脚本文件。
        所有操作必须通过已有的 skills scripts 执行：
        - 启动服务 → vllm-ascend-serving/scripts/serve_start.py
        - 停止服务 → vllm-ascend-serving/scripts/serve_stop.py
        - 基准测试 → vllm-ascend-benchmark/scripts/bench_run.py
        - Profiling  → ascend-profiling-collection/scripts/collect_torch_profile_case.py
        - NPU 探测  → vllm-ascend-serving/scripts/serve_probe_npus.py
        如需新增脚本，应作为独立 PR 提交到对应 skill 的 scripts/ 目录。

    # --- 2. 禁止手写 vllm serve 启动命令 ---
    - tools: ["Bash", "PowerShell"]
      matcher: "(vllm serve|nohup.*vllm|vllm.entrypoints)"
      action: prompt
      message: |
        ⚠️ 禁止直接手写服务启动/停止命令。
        请改用 Skill 工具委派：
          Skill("vllm-ascend-serving", args="启动服务, 模型=..., 端口=..., TP=..., ...")
        对应的 serve_start.py / serve_stop.py 会处理环境准备、残留清理、健康检查等。

    # --- 3. 禁止手写 vllm bench serve 压测命令 ---
    - tools: ["Bash", "PowerShell"]
      matcher: "vllm bench serve"
      action: prompt
      message: |
        ⚠️ 禁止直接手写 vllm bench serve 压测命令。
        请改用 Skill 工具委派：
          Skill("vllm-ascend-benchmark", args="压测, 并发=..., input/output=..., 重复=...")
        对应的 bench_run.py 会处理参数校验、多次测试汇总、服务生命周期。

    # --- 4. 禁止手写 msprof / msprof-analyze 命令 ---
    - tools: ["Bash", "PowerShell"]
      matcher: "(\\bmsprof\\b|\\bmsprof-analyze\\b)"
      action: prompt
      message: |
        ⚠️ 禁止直接手写 msprof / msprof-analyze 命令。
        请改用 Skill 工具委派：
          - 采集 → Skill("ascend-profiling-collection", args="...")
          - 分析 → Skill("ascend-msprof-analyze-cli", args="...")

    # --- 5. Windows 下优先用 PowerShell，不要用 Git Bash ---
    - tools: ["Bash"]
      matcher: "ssh.*\\s/(home|workspace|tmp|usr|vllm)"
      action: prompt
      message: |
        ⚠️ Git Bash (MSYS) 会自动转换 Linux 路径，导致远程命令失败。
        例如 /home/vllm/weights → C:/Program Files/Git/home/vllm/weights。
        请切换到 **PowerShell** 执行 SSH 远程命令，避免路径被篡改。
        如确需用 Bash，路径前加双斜杠 `//home/vllm/weights` 可阻止 MSYS 转换。

    # --- 6. JSON 参数中的双引号转义 ---
    - tools: ["Bash", "PowerShell"]
      matcher: "(compilation_config|additional_config|profiler.config|[{}])"
      action: prompt
      message: |
        ⚠️ 命令行中的 JSON 参数需要正确转义双引号：
        - Git Bash: '{"key":"value"}' 或 "{\\"key\\":\\"value\\"}"
        - PowerShell: '{\"key\":\"value\"}' 或使用 here-string @'
        错误转义会导致 JSON 解析失败，服务启动参数丢失。

    # --- 7. 修改环境变量前先查原值 ---
    - tools: ["Bash", "PowerShell"]
      matcher: "export\\s+(TASK_QUEUE|HCCL|ASCEND|ACLNN|CPU_AFFINITY|NPU_ALLOC|MINDIE|ATB_)"
      action: prompt
      message: |
        ⚠️ 修改昇腾相关环境变量前，必须先 echo $VAR 查原值，记录到变更清单。
        流程：
        1. echo $变量名    ← 先查原值
        2. 记录到变更清单  ← Phase 2.5
        3. 再执行 export   ← 你的当前命令
        回退命令：unset 变量名 或 export 变量名=原值

    # --- 8. 容器重启后环境变量丢失 ---
    - tools: ["Bash", "PowerShell"]
      matcher: "docker (re)?start"
      action: prompt
      message: |
        ⚠️ 容器重启后内部环境变量会丢失（ASCEND_GLOBAL_LOG_LEVEL、HCCL_DETERMINISTIC 等）。
        确认已在 vllm serve 命令中通过 --extra-env 或启动脚本重新设置了 Phase 2 的环境变量。

    # --- 9. 杀进程后检查显存 ---
    - tools: ["Bash", "PowerShell"]
      matcher: "(p)?kill.*vllm"
      action: prompt
      message: |
        ⚠️ 杀掉 vllm 进程后，确认显存已释放再重启服务：
          npu-smi info -t usages -i <N> | grep HBM
        僵尸进程占用显存是 OOM / EngineCore 崩溃的常见根因。
targets-repo:
  - vllm-ascend
  - cann
---


# 昇腾NPU性能调优实操指南

> 本指南定义了一套**决策驱动的调优工作流**。每步有明确判断依据和决策点，
> 你需根据实际profiling数据做判断，而非机械执行。
> 当判断依据不充分时，**先采集数据再决策**，不要靠猜测。
> 详细案例库与盲区分析见: `perf_tuning_research/key_info.md`

---

## 工作流总览

```text
问题接入 → 场景分流 → 环境预检 → 采集profiling → 瓶颈三分类 → 查知识库(案例) → 按方向深入 → 验证闭环
                                                                         │                      │
                                                                  project-brain + Ascend KG      未达标 ─┘
                                                                                                  达标 → 结束
```

**核心原则：这是一个迭代循环，不是单向流程。** 每次优化后必须重新压测/采集 profiling 验证收益，未达标则回到瓶颈定位步骤继续。不要"一次改完所有方案再验证"——逐项修改、逐项验证、逐项决定保留或回退。

---

## 集成知识库案例查询

本指南集成了两个外部知识源，均在 **Phase 4 瓶颈识别后** 触发查询：

| 知识源 | 定位 | 接入方式 |
| -------- | ------ | --------- |
| **project-brain MCP** | 性能工厂 312 篇实战案例 | MCP 工具 `project-brain_keyword_search` |
| **Ascend KG** | 昇腾知识图谱 — 近 500 万节点、460+ Skill | REST API `ascend.wiki` |

两者查询时机一致（瓶颈分类后），互相补充。详见 Phase 4 的"瓶颈识别后查询知识库"。

### project-brain 案例查询

1. **瓶颈识别后自动搜索**：在 Phase 4（瓶颈定位）识别出瓶颈类型后，立即使用 `project-brain_keyword_search` 搜索性能工厂案例库中相似案例
2. **搜索关键词规则**：`scope` 设为 `"性能工厂"`，关键词组合为 `"性能" + <瓶颈类型/优化方案> + <场景关键词>`。典型搜索词：
   - Hostbound 场景：`"性能 hostbound"`、`"性能 下发瓶颈"`
   - 通信场景：`"性能 通信"`、`"性能 快慢卡"`、`"性能 通信阻塞"`
   - 计算场景：`"性能 TransData"`、`"性能 算子优化"`
   - 长稳劣化场景：`"性能 劣化"`、`"性能 长稳"`
   - 推理场景：`"性能 推理 服务化"`
   - 版本/升级场景：`"性能 版本 劣化"`
3. **结果筛选**：从搜索结果中选取与当前问题场景最匹配的 1-2 个案例，提取根因和解决方案作为参考
4. **输出格式**：在给出优化建议时，如匹配到相关案例，以 `📖 参考案例：<案例名称>` 格式附注在方案后
5. **无匹配时**：如果 `project-brain_keyword_search` 返回结果中无密切相关的案例，给出通用的优化建议即可，不强行关联

**注**：如果当前环境中没有配置 project-brain MCP 服务，或查询失败，应跳过此步骤，继续按标准流程给出优化建议。

---

## Phase 0: 问题接入

收集以下信息，它们决定后续所有决策：

| 必收信息 | 为什么重要 |
| --------- | ----------- |
| 训练还是推理？ | 工具链/优化手段完全不同 |
| 开箱(首次)还是长稳(劣化)？ | 开箱走A类故障树，长稳走B类 |
| 单机还是多机？ | 多机优先排查通信/快慢卡 |
| 目标来源(vs竞品/基线/线性度)？ | 决定优化力度和可接受的精度折中 |
| 芯片型号(A2/A3/310P)？ | A3通信亲和组配置不同，310P精度策略不同 |
| 版本(CANN/PyTorch/MindIE)？ | 版本差异可导致2倍性能差 |

**决策点**: 用户说不清问题现象时，**先帮跑基线测试再继续**。

### GPU 对标场景处理规则

当用户要求与 GPU（如 H100）对比性能时：

1. **必须向用户索要以下数据，缺一不可**：
   - GPU 上的 benchmark 结果（total tokens/s、TTFT、TPOT、batch size、input/output length）
   - GPU 上的 profiling 数据（如果有，用于算子级对比）
   - GPU 型号、CUDA 版本、所用推理框架版本
   - 测试条件（并发数、warmup、input/output 分布）

2. **如果用户无法提供 GPU 数据**：
   - 明确告知无法做精确的 NPU vs GPU 对比
   - 可以提供公开的 GPU 参考数据（如 MLPerf、官方 benchmark），但必须在结论中标注"参考值，非同等条件对比"
   - 建议用户先在 GPU 上跑相同的 benchmark 脚本，再在 NPU 上复现

3. **算子级性能对比**：
   - 需要用户提供 GPU 上各算子的耗时（NVIDIA Nsight Systems / Nsight Compute 采集）
   - 在 NPU 上用 msprof 采集相同算子的耗时
   - 对比时需确保：相同 shape、相同数据类型、相同算子实现

4. **结论表述规范**：
   - 有 GPU 实测数据：`NPU 达到 GPU 的 X%`（附双方数据）
   - 仅有公开参考值：`NPU 达到 H100 公开参考值的 X%`（标注数据来源）
   - 无任何对比数据：**禁止给出 NPU vs GPU 的百分比结论**

---

## Phase 0.5: 命令工具箱 [所有路径]

> 本节提供推理场景下**服务启动、压测、Profiling 采集、数据解析**的标准化命令模板。
> 所有路径在执行测试前，先从这里选择合适的命令组合。

### 0.5.1 vllm 服务启动

```bash
python -m vllm.entrypoints.openai.api_server \
  --model=<model_path> \
  --served-model-name <model_name> \
  --port 3113 \
  -tp 4 \
  -dp 2 \
  --enable-expert-parallel \    # MoE 模型必加
  --gpu-memory-utilization 0.8 \ # 推理场景 0.8~0.9
  --no-enable-log-requests \     # 关请求日志，减少 CPU 开销
  --max-num-seqs 128 \           # 增大 batch 容量（默认 64）
  --max-model-len 4096 \         # 模型最大序列长度
  --max-num-batched-tokens 65536 \ # 增大 batch token 上限
  --async-scheduling \           # 异步调度，减少 NPU 空泡
  --compilation_config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --trust-remote-code
```

**关键参数说明**：

| 参数 | 默认值 | 调优方向 | 原理 |
| ------ | -------- | --------- | ------ |
| `--max-num-seqs` | 64 | 增大（128/256） | 更大 batch → 摊薄 CPU 下发开销 → 缓解 hostbound |
| `--max-num-batched-tokens` | 32768 | 增大（65536/98304） | 更大 prefill batch → 减少 prefill 次数 |
| `--gpu-memory-utilization` | 0.9 | 0.8~0.95 | 调低可避免 OOM，调高可容纳更大 batch |
| `--compilation_config` | 无 | `FULL_DECODE_ONLY` | decode 阶段用 cudagraph 加速，跳过 prefill 编译 |
| `--async-scheduling` | 取决于版本 | 开启 | CPU 调度不阻塞 NPU 执行 |

### 0.5.2 压测（vllm bench serve）

```bash
vllm bench serve \
  --backend vllm --trust-remote-code \
  --model <model_path> \
  --tokenizer <model_path> \
  --served-model-name <model_name> \
  --dataset-name random \
  --random-input-len 2048 --random-output-len 2048 \
  --ignore-eos \
  --num-prompts 128 \
  --max-concurrency 128 \          # 必须 >= num-prompts！
  --num-warmups 50 \
  --temperature 0.0 \
  --base-url http://127.0.0.1:<port>
```

**关键参数说明**：

| 参数 | 推荐值 | 为什么 |
| ------ | -------- | -------- |
| `--max-concurrency` | >= `--num-prompts`（如 128） | 小于 num-prompts 时请求被限流，总吞吐降低 30~40% |
| `--num-warmups` | 50 | vllm warmup 有缺陷（串行发 + 重建 session），设 50 足够 |
| `--num-prompts` | 128 | 保证有足够请求让服务端进入稳态 |
| `--random-input-len` / `--random-output-len` | 2048 / 2048 | 标准 benchmark 配置，便于跨平台对比 |

**注意**：vllm bench serve 的 warmup 阶段是串行发请求，结束后会重建 HTTP session。正式测试的第一个请求仍需完整走 prefill + decode（~90-100s），后续请求在 1s 内完成。总 duration ≈ 第一个请求耗时。这是工具行为，不影响结果有效性。

### 0.5.3 Profiling 采集

#### 推理服务 Profiling（msprof 命令行）

```bash
# 先找到服务进程 PID
ps aux | grep vllm.entrypoints | grep -v grep

# 启动 profiling（采集 30 秒）
msprof --output=<output_dir> --application="bash -c 'sleep 30'" \
  --profile-ms=1 --model-execution=on --sys-hardware-mem=on \
  --runtime-api=on --task-time=on --aicpu=on \
  --l2_cache=on --op-trace=on --ddk-dir=<CANN_PATH>/opp

# 在 profiling 期间，用压测工具发请求
vllm bench serve ... --base-url http://127.0.0.1:<port>
```

#### 训练 Profiling（PyTorch 代码内）

```python
import torch_npu.profiler as prof
with prof.profile(
    profiler_level=prof.Level.Level1,
    aic_metrics=prof.AicMetrics.PipeUtilization,
    schedule=prof.schedule(wait=0, warmup=1, active=2, repeat=1),
    on_trace_ready=prof.tensorboard_trace_handler("./prof_out")) as p:
    for step in range(3):
        train_one_step()
        p.step()
```

### 0.5.4 Profiling 数据解析

```bash
# 1. 先用 advisor 做全局诊断
msprof-analyze advisor all -d <profiling_dir>

# 2. 查看关键统计文件
# 算子级耗时统计
cat <profiling_dir>/ASCEND_PROFILER_OUTPUT/op_statistic.csv

# Kernel 级详细耗时
cat <profiling_dir>/ASCEND_PROFILER_OUTPUT/kernel_details.csv

# Step 级别的计算/通信/调度分解
cat <profiling_dir>/ASCEND_PROFILER_OUTPUT/step_trace_time.csv

# 通信矩阵
cat <profiling_dir>/ASCEND_PROFILER_OUTPUT/communication_matrix.json

# 3. 用 MindStudio Insight 打开 trace_view.json 看 Timeline
# trace_view.json 路径:
<profiling_dir>/ASCEND_PROFILER_OUTPUT/trace_view.json
```

### 0.5.5 快速验证脚本（Python）

当需要自定义压测逻辑（如特定 input/output 分布、特殊 warmup 策略）时：

```python
import asyncio, aiohttp, time

API_URL = "http://127.0.0.1:3113/v1/completions"
PROMPT = "hello " * 1900  # ~1900 tokens
OUTPUT_LEN = 2048
CONCURRENCY = 128
NUM_REQUESTS = 128

async def send(session, sem):
    async with sem:
        payload = {"model": "qwen", "prompt": PROMPT,
                   "max_tokens": OUTPUT_LEN, "temperature": 0.0, "ignore_eos": True}
        async with session.post(API_URL, json=payload) as resp:
            data = await resp.json()
        return data.get("usage", {}).get("completion_tokens", 0)

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [send(session, sem) for _ in range(NUM_REQUESTS)]
        t0 = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - t0
        total_output = sum(results)
        print(f"Duration: {duration:.2f}s, "
              f"Output tok/s: {total_output/duration:.2f}")

asyncio.run(main())
```

根据Phase 0信息选择路径，**后续每个Phase会标注适用哪条路径**:

| 路径 | 场景 | 核心差异 | 后续Phase |
| ------ | ------ | --------- | ----------- |
| **Path A** | 训练+开箱 | 从零优化: 环境预检→并行策略→瓶颈三分类→按方向深入→验证→迭代 | Phase 2→3→4→5A/5B/5C→8 |
| **Path B** | 训练+长稳 | 从变更入手: 先排查近期变更，再判断波动模式定向排查 | Phase 2→3→6 |
| **Path C** | 推理+纯模型 | 类似Path A，但重点关注算子/格式/版本(非服务化调度) | Phase 2→3→4→5A/5B/5C→8 |
| **Path D** | 推理+服务化 | **前提: 纯模型已达标**，再调服务化调度参数 | Phase 2→0.5(命令工具箱)→5D→8 |

**关键决策(Path D)**: 推理场景必须先确认纯模型性能达标。未达标就调服务化是浪费精力——先跑纯模型基线对比。

**关键决策(Path D)**: 推理场景必须先确认纯模型性能达标。未达标就调服务化是浪费精力——先跑纯模型基线对比。

---

## Phase 2: 环境预检

无论哪条路径，先做预检排除低级错误。

### 推理场景预检

```bash
pip install msprechecker
msprechecker precheck    # 打屏展示: 环境变量/服务化配置/网络配置/系统配置
# 根据打屏提示逐项修正，需对照基线yaml(由解决方案提供)
```

- 依赖: device ip + rank_table; Python>3.8
- 可在服务启动后运行; 资料: https://gitee.com/ascend/msit/tree/master/msprechecker

### 训练场景预检

```bash
git clone https://gitee.com/ascend/mstt.git
cd mstt/profiler/msprof_analyze && git checkout poc-precheck
pip3 install -r requirements.txt && python3 setup.py bdist_wheel
pip3 install dist/msprof_analyze-*-py3-none-any.whl
msprof-analyze precheck env --nproc_per_node 8 --nnodes 4 --node_rank 0
# --check-type可选: env_variable|python_lib|cpu|npu|communication|io|all
```

- 依赖: 需安装torch; 只能在容器内运行; 不能在服务启动后运行
- 资料: https://gitee.com/ascend/mstt/blob/poc-precheck/profiler/msprof_analyze/precheck/README.md

**思考提示**: 预检发现问题后不要认为"修完就好了"。记录预检结果，采集profiling验证。多因素叠加是常态。

通用推荐配置(预检后仍需手动确认):

```bash
export ASCEND_GLOBAL_LOG_LEVEL=3     # ERROR级,debug/info严重拖慢
unset ASCEND_LAUNCH_BLOCKING         # 关闭流同步
export HCCL_DETERMINISTIC=false       # 关闭确定性计算
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True  # 内存池扩展
export MINDIE_LOG_LEVEL=ERROR        # 推理场景
```

---

## Phase 2.5: 环境变量变更管理 [所有路径必读]

**原则: 每次修改环境变量前，必须先记录原值，维护变更清单，验证后决定保留或回退。**

### 为什么必须这么做?

- 环境变量之间可能冲突(如CPU_AFFINITY_CONF=1对训练有益但对单卡推理有害)
- 多人协作时不知道别人设了什么，容易重复设置或互相覆盖
- 某些变量设错会导致精度问题或OOM，不记录原值就无法回退
- 调优结束时，需要确认哪些是必须保留的，哪些应该撤销

### 修改前必须操作

```bash
# 1. 先查当前值(不要上来就export!)
echo $TASK_QUEUE_ENABLE
echo $CPU_AFFINITY_CONF
echo $HCCL_DETERMINISTIC
# 或一次性导出所有相关变量:
env | grep -E "TASK_QUEUE|CPU_AFFINITY|HCCL|ASCEND|ACLNN|NPU_ALLOC|MINDIE|ATB_"

# 2. 记录原值到变更清单
# 3. 修改
# 4. 验证后决定保留或回退
```

### 变更清单模板(每次调优必须维护)

```text
=== 环境变量变更清单 ===
日期: 2026-xx-xx  场景: xxx训练  路径: Path A

[ ] TASK_QUEUE_ENABLE
    原值: 1(默认)  现值: 2
    原因: hostbound, 一级流水负载迁移至二级流水
    验证: free占比从59%→30%, step时间从2.0s→1.8s ✓
    回退: export TASK_QUEUE_ENABLE=1

[ ] ACLNN_CACHE_LIMIT
    原值: 未设置(默认)  现值: 100000
    原因: 二级流水大量getWorkspacesize
    验证: 未观察到额外收益, 回退
    回退: unset ACLNN_CACHE_LIMIT
```

### 调优结束时的清理

- 逐项审视变更清单: 哪些确认有效需持久化(写入启动脚本), 哪些应回退
- 有效配置应写入项目的启动脚本(set_env.sh等), 而非只在当前shell生效
- 确认没有遗留的临时设置(如调试用的ASCEND_LAUNCH_BLOCKING=1)

---

## Phase 3: 采集Profiling [所有路径]

> ⚠️ **硬门禁：基线跑完后的第一个动作必须是采集 profiling。禁止跳过本 Phase 直接试参数。**
>
> 没有 profiling 数据 = 不知道瓶颈方向 = 纯猜测优化。本 Phase 采集数据后，交给 Phase 4 判断方向，
> 再进入 Phase 5 实施优化。跳过 Phase 3+4 直接到 Phase 5 是本工作流最常见的错误。

### 3.0 执行采集

> 先去 `Read skills/inference-model-skills/performance-optimization/ascend-profiling-collection/SKILL.md`，
> 按其中的参数表确认采集参数，再按命令模板执行采集。

**关键参数速查**：

- 推理服务使用 L1 级别、不开 with_stack（膨胀严重）
- 采集阶段选 `middle`（decode 中段，反映稳态 TPOT）
- prompt 数量 10-32 即可，不需要全量压测
- 采集完成后拿到 `<profiling_dir>` 路径，进入 Phase 4 分析

### 3.1 核心概念解释

- **profiling**: 采集模型运行时性能数据，包括每个算子的耗时、通信耗时、CPU/NPU空闲时间等
- **L0/L1/L2**: 采集级别。L0最轻量(无膨胀, 通信/调度数据最准), L1加CANN/HCCL/AI Core指标(常用), L2加Runtime但膨胀严重
- **with_stack**: 采集Python调用栈。**膨胀极严重**, 会导致虚假的hostbound, 仅定位代码行时短暂开启
- **Timeline**: MindStudio Insight中的时间线视图, 可以看到CPU/CANN/NPU各层的算子执行时间线, 是定位问题的关键工具
- **膨胀**: profiling采集本身会影响模型性能, 导致采集数据比实际差。with_stack/L2膨胀最严重

### 采集原则

采集原则:

- **首次采集L1级, 不开with_stack**(膨胀严重, 会虚假hostbound)
- warmup=1, active=2步
- 多机先cluster分析缩小范围, 再对异常卡深入

```python
import torch_npu.profiler as prof
with prof.profile(
    profiler_level=prof.Level.Level1,
    aic_metrics=prof.AicMetrics.PipeUtilization,
    schedule=prof.schedule(wait=0, warmup=1, active=2, repeat=1),
    on_trace_ready=prof.tensorboard_trace_handler("./prof_out")) as p:
    for step in range(3):
        train_one_step(); p.step()
```

分析工具(按顺序):

1. `msprof-analyze advisor all -d {prof_path}` — 先跑advisor
2. `msprof-analyze cluster all -d {prof_path}` — 多机必跑
3. MindStudio Insight — 看Timeline细节

**思考提示**: advisor建议逐条看但不全执行，先关注影响最大的。

---

## Phase 4: 瓶颈定位 — 三选一决策 [Path A/C核心]

> ⚠️ **硬门禁：必须基于 Phase 3 采集的 profiling 数据做判断，禁止凭感觉猜瓶颈方向。**
>
> 先去 `Read skills/inference-model-skills/performance-optimization/ascend-msprof-analyze-cli/SKILL.md`，
> 按其中的分析流程解析 Phase 3 采集的 `<profiling_dir>`，拿到 Free Time%、通信占比、算子耗时分布
> 等定量数据后，再按下面的判定表做三选一。

**这是最关键决策点，方向错后面全错。**

### 先理解核心概念

- **Free Time(空闲时间)**: NPU没在执行计算/通信的时间。高Free意味着NPU在等CPU下发任务(hostbound)，或在等数据加载。看step_trace_time.csv或Insight的Summary页签
- **Hostbound**: CPU下发速度跟不上NPU执行速度，NPU出现空泡。Timeline上表现为: 下发连线近乎垂直(而非理想的倾斜弧线), NPU算子之间有空隙
- **TransData**: ND格式和NZ格式之间的数据转换算子。NPU有两种数据排布格式(ND=常规连续, NZ=分块存储利于矩阵乘), 不同算子要求不同格式, 格式不匹配时自动插入TransData转换, 消耗额外时间
- **aclnn vs aclop**: 两种算子执行路径。aclop(路径3)走图编译图执行, aclnn(路径5)走eager立即执行。关闭jit_compile后从路径3切到路径5, 通常路径5更快

### 判断依据

|------|---------|---------|
| **下发** | Free Time>10% | Free 59% + Compute 40% |
| **通信** | 通信占比异常; 卡间差异大; 带宽远低于经验值 | 通信>50%且非计算掩盖 |
| **计算** | 算子耗时异常; advisor报亲和; transdata/cast占比高 | TransData 21%+Cast 14% |

**思考提示**:

- 混合瓶颈(下发+通信)常见, 先解决占比最大的
- Free高可能是下发问题, 也可能是预处理在循环内——看Timeline确认free时CPU在干什么
- 通信占比高可能是快慢卡导致等待——先排除快慢卡

### 瓶颈识别后查询知识库

完成瓶颈分类后（识别出是下发/通信/计算/长稳/服务化问题），**必须同时查询两个知识源**获取相似案例和参考方案：

#### 源 1：project-brain MCP（性能工厂案例库）

根据瓶颈类型调用 `project-brain_keyword_search`：

| 瓶颈类型 | `scope` | `query` |
| --------- | --------- | --------- |
| Hostbound (下发) | `"性能工厂"` | `"性能 hostbound"` |
| 通信瓶颈 | `"性能工厂"` | `"性能 通信"` |
| 快慢卡 | `"性能工厂"` | `"性能 快慢卡"` |
| 计算瓶颈(TransData/Cast) | `"性能工厂"` | `"性能 TransData"` 或 `"性能 算子"` |
| 长稳劣化 | `"性能工厂"` | `"性能 劣化"` |
| 推理服务化 | `"性能工厂"` | `"性能 推理 服务化"` |

#### 源 2：Ascend KG（昇腾知识图谱）

**预检**（首次查询前执行一次）：

```bash
curl -sf --max-time 5 -H "X-API-Key: kg-dev-d733dba010c46a2b0178a5e91663c5ed91d6dc93f09e6e6f" https://ascend.wiki/health
```

返回 200 则可用，非 200 则跳过 KG。

**方式 A：加载调优 Skill 获取完整方法论**

```bash
# vLLM 服务化调优
curl -s --compressed -X POST https://ascend.wiki/source \
  -H "Content-Type: application/json" \
  -H "X-API-Key: kg-dev-d733dba010c46a2b0178a5e91663c5ed91d6dc93f09e6e6f" \
  -d '{"node_id": "vllm-ascend-performance-optimization"}'

# 推理融合算子优化
curl -s --compressed -X POST https://ascend.wiki/source \
  -H "Content-Type: application/json" \
  -H "X-API-Key: kg-dev-d733dba010c46a2b0178a5e91663c5ed91d6dc93f09e6e6f" \
  -d '{"node_id": "model-infer-fusion"}'
```

**方式 B：语义检索文档和案例**

根据瓶颈类型用中文搜索：

```bash
curl -s --compressed -X POST https://ascend.wiki/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: kg-dev-d733dba010c46a2b0178a5e91663c5ed91d6dc93f09e6e6f" \
  -d '{"query": "<瓶颈描述 + 优化关键词>", "top_k": 5, "file_type": "document", "with_neighbors": true}'
```

| 瓶颈类型 | KG 搜索 query 示例 |
| --------- | ------------------- |
| Hostbound | `"hostbound TASK_QUEUE 下发 优化"` |
| 通信 | `"HCCL 通信 带宽 优化"` |
| 计算 | `"TransData Cast 算子 性能"` |
| 推理服务化 | `"vLLM 推理 吞吐 decode 优化 max-num-seqs"` |
| MoE 模型 | `"MoE EP FLASHCOMM 多流 优化"` |

**KG 评分阈值**：

| score | 含义 | 动作 |
| ------- | ------ | ------ |
| > 0.83 | 黄金命中 | 用 `/source` 取全文，直接参考 |
| 0.70-0.83 | 模糊 | 换中/英文改写 query 重试 |
| < 0.70 | 噪声 | 换 `file_type:configuration` 查参数，或换 `file_type:concept` 查设计 |

#### 查询后输出格式

从两个源的返回中选取最匹配的 1-3 个结果，按以下格式附注在优化建议中：

```text
→ 瓶颈类型：下发瓶颈（Hostbound），Free占比59%

📖 project-brain：【智源hostbound】emu3.5day0适配 hostbound问题定位案例
  根因：动态shape导致在线编译，CPU下发排队
  方案：jit_compile=False + ACLNN_CACHE_LIMIT=100000

📖 Ascend KG：(score 0.91) vllm-ascend-performance-optimization 使用指南_hostbound调优
  建议：TASK_QUEUE_ENABLE=2，增大 max-num-batched-tokens

→ 开始下发方向优化（Phase 5A）...
```

**注**：如果某个知识源不可用（project-brain 未配置 MCP、KG 返回非 200），跳过该源继续用另一个即可，不强求两者都有结果。

---

## Phase 5A: 下发方向深入 [Path A/C]

> ⚠️ **前置检查**：确认 Phase 3（profiling 采集）和 Phase 4（瓶颈定位+查知识库）已完成。
> 如果 Phase 4 判定瓶颈不是"下发"，不应进入本方向——回到 Phase 4 的判定表选正确方向。

### 5A.1 通用优化(按收益排序, 逐一验证)

**关键原则: 每项改完单独验证记录收益, 不要一次全改。否则不知道哪项有效，也无法回退。所有修改前先查原值并记录到变更清单(Phase 2.5)。**

**案例查询指引**：在下发方向优化前，搜索匹配案例作为参考：

- project-brain：`scope="性能工厂"`, `query="性能 hostbound TASK_QUEUE"` 或 `query="性能 hostbound jit_compile"`
- Ascend KG：搜索 `"hostbound TASK_QUEUE 下发 优化"`
- 如搜索结果中有关联案例，在对应的优化项后以 `📖 参考案例：<名称>` 附注
- 适用于 5A.1~5A.4 各项优化

#### 1. TASK_QUEUE_ENABLE=2

- **做了什么**: 一级流水的"参数转换+workspace申请"任务迁移到二级流水, 增加并行度
- **为什么有效**: 原本一级流水串行做"参数转换→申请workspace→下发", 改为二级流水并行做"申请workspace", 一级流水只做下发, 减少dequeue唤醒时间
- **收益**: ~10%, 几乎所有hostbound场景有效
- **怎么判断该设**: profiling二级流水(CANN层)中存在大量getWorkspacesize操作; Free占比高
- **原值查询**: `echo $TASK_QUEUE_ENABLE` (默认1)
- **回退**: `export TASK_QUEUE_ENABLE=1`
- **注意**: 采集profiling时用1(2会改变调度行为导致采集不准), 优化运行时用2
- **⚠️ 图模式不兼容**: vllm serve 未配置 `--enforce-eager` 时默认使能图模式（graph mode），图模式下 **不能** 设置 `TASK_QUEUE_ENABLE=2`，会导致功能异常。只在 eager 模式（`--enforce-eager`）下才能使用此优化

#### 2. ACLNN_CACHE_LIMIT=100000

- **做了什么**: 扩大算子编译缓存条目数
- **为什么有效**: 每个新shape首次执行时需编译(getWorkspacesize), 编译结果缓存后复用。缓存太小导致反复编译
- **怎么判断该设**: profiling二级流水中看到大量getWorkspacesize操作; advisor报算子编译问题
- **原值查询**: `echo $ACLNN_CACHE_LIMIT` (默认较小)
- **回退**: `unset ACLNN_CACHE_LIMIT`

#### 3. CPU_AFFINITY_CONF — 根据场景选值

| 值 | 场景 | 做了什么 | 为什么有效 |
| ---- | ------ | --------- | ----------- |
| 1 | 8卡训练 | 每个进程绑定到对应NPU的NUMA节点CPU核组 | 避免跨NUMA内存访问, 减少CPU核切换开销 |
| 2 | A+K/虚拟机/细粒度 | 分离算子下发主线程与数据处理线程绑核 | 避免线程间CPU竞争(下发线程被数据处理抢占) |
| 0 | 单卡高并发推理 | 不绑核 | 多生产单消费场景, 绑核反而限制CPU调度灵活性 |

- **怎么判断该设**: hostbound + 未绑核; 或波动大 + 未绑核
- **原值查询**: `echo $CPU_AFFINITY_CONF`
- **回退**: `unset CPU_AFFINITY_CONF`

#### 4. jit_compile=False + allow_internal_format=False

- **做了什么**: 关闭在线编译(jit) + 关闭NPU私有格式(ND/NZ自动转换)
- **为什么有效(jit)**: 每次遇到新shape都在线编译, 耗时百ms级。关闭后走预编译的aclnn路径(路径5), 速度更快
- **为什么有效(格式)**: 关闭私有格式后全网统一用ND格式, 消除TransData转换开销。但部分算子在NZ格式下更快, 需权衡
- **怎么判断该设**: advisor报在线编译问题; TransData占比高
- **代码**: `torch_npu.npu.set_compile_mode(jit_compile=False)` + `torch_npu.npu.config.allow_internal_format = False`
- **注意**: 动态shape时jit_compile=False基本无收益(仍需编译); 关闭后首次step可能慢(编译阶段), 后续稳定; 310P上需实测确认哪个路径更快

### 5A.2 仍hostbound? 看Timeline确认Free时CPU行为

**原理: Timeline中NPU的Free空泡有不同成因，关键看同一时刻CPU在干什么。这是"看数据判断根因"而非"试一下看看"。**

| Free时CPU行为 | 怎么看出来 | 根因 | 方案 |
| -------------- | ----------- | ------ | ------ |
| CPU空白(无任何活动) | Timeline的CPU泳道完全空白 | GC暂停或CPU资源被其他进程抢占 | gc.set_threshold; bpytop/htop查后台进程 |
| CPU在做预处理 | CPU泳道有数据处理函数名 | 预处理在训练循环内, NPU等CPU | 移到dataloader; 增num_workers |
| CPU在下发但NPU空等 | CPU泳道有CANN下发接口, NPU泳道无算子 | 小算子太多, CPU逐个下发速度跟不上 | 融合算子; 增大BS; 亲和优化器 |
| 某卡CPU慢于他卡 | 对比多卡Timeline下发弧线(倾斜度不同) | 绑核/CPU竞争/版本不一致 | CPU_AFFINITY_CONF=2; 查版本; 关pin_memory |
| CPU做H2D搬运 | CPU泳道有memcpy/H2D操作 | pin_memory抢占或CPU算子迫使H2D->CPU->D2H | to(device,non_blocking=True); 检查是否有只在CPU实现的算子(如STFT) |

### 5A.3 代码级下发优化

```python
# 亲和优化器: 原生AdamW每个参数更新都单独H2D下发(频繁小搬运),
# 亲和版先在Host侧完成所有更新, 再统一下发到Device(少次大搬运)
optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=lr)

# 异步to: non_blocking=True让H2D异步执行, CPU不等待搬运完成就继续下发下一个算子
images = images.to(device, non_blocking=True)

# 重复tensor存类变量: 避免每个step重复创建+to(device), 打断流水
# 改前: each step creates mask_tensor and .to(device)  ← 每次100ms+
# 改后: self.mask_tensor = mask_tensor.to(device)  ← init一次, 后续复用
```

### 5A.4 特殊场景

- **freeze后劣化**: 关pin_memory + CPU_AFFINITY_CONF=2
- **版本过旧**: CANN RC2 vs RC3可2倍差异
- **推理单卡高并发**: CPU_AFFINITY_CONF=0, taskset绑奇数NUMA

---

## Phase 5B: 通信方向深入

**案例查询指引**：在通信方向优化前，AI agent 应搜索匹配的通信/快慢卡案例作为参考：

- 使用 `project-brain_keyword_search`，`scope="性能工厂"`，`query="性能 快慢卡"` 或 `query="性能 通信阻塞"`
- 使用 `project-brain_keyword_search`，`scope="性能工厂"`，`query="性能 通信 HCCS"` 或 `query="性能 多机 通信"`
- 在优化建议中引用匹配到的案例，格式：`📖 参考案例：<名称>`

### 5B.1 通信问题根因判断(关键决策)

**通信占比高 != 通信本身有问题**, 按此顺序:

```text
1. 有快慢卡? → cluster看各卡差异 → 计算差>3%或下发差>10%=有
   有快慢卡: 先解决(见5B.3), 通信等待是果不是因
2. 带宽正常? → HCCS~30GB/s, SDMA~19GB/s, RDMA>20GB/s
   达标: 通信本身无问题, 从并行策略入手
   远低于经验值: 继续排查
3. 效率低的原因?
   数据量太小 → 优化并行策略减少小包
   SDMA不对齐 → 数据需512B整除(否则降10倍)
   RDMA重传 → 单次>4s=重传, hccn_tool -i $i -stat -g|grep rty确认
   抢带宽 → MatMul+allGather并发导致
```

### 5B.2 通信环境变量(修改前先查原值! 见Phase 2.5)

```bash
# HCCL_BUFFSIZE: 两个NPU间共享数据缓存区大小
# 为什么设200: 默认200M, 通信数据量大时可增大, 通信域多时可减小(避免占模型内存)
# 怎么判断该调: 通信数据量大且通信效率低(未达经验带宽)时尝试增大
export HCCL_BUFFSIZE=200
# 原值: echo $HCCL_BUFFSIZE  回退: unset HCCL_BUFFSIZE

# HCCL_OP_EXPANSION_MODE: 通信算法编排展开位置从HOST转到AI Vector
# 为什么有效: 减少host侧通信算子下发耗时, 通信编排由NPU侧AI Vector执行
# 怎么判断该设: 通信算子下发慢(hostbound+通信占比高)
export HCCL_OP_EXPANSION_MODE="AIV"
# 原值: echo $HCCL_OP_EXPANSION_MODE  回退: unset HCCL_OP_EXPANSION_MODE
# ⚠️ MindIE 2.0.T3/T3.1/T6 使能AIV有崩溃问题, 必须unset!

# HCCL_RDMA_TIMEOUT: RDMA超时时间, 默认20(=4.3s, 公式4.096*2^value)
# 为什么有效: 默认超时4.3s太长, 重传时等待太久。减小后更快触发重传恢复
# 怎么判断该设: Timeline中出现>4s的Notify-Wait; hccn_tool查到重传
export HCCL_RDMA_TIMEOUT=xxx
# 原值: echo $HCCL_RDMA_TIMEOUT  回退: export HCCL_RDMA_TIMEOUT=20

# HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT: 通过PCIe Direct提交RDMA任务
# 为什么有效: 减少node@launch耗时, 提升通信算子下发性能
# 怎么判断该设: 通信算子下发慢(推理场景常见)
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
# 原值: echo $HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT  回退: unset HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT
```

### 5B.3 快慢卡四步定位

```text
1. 集群summary → 差异>3%/10%异常
2. 计算差异 → op_statistics找shape/算子数差异
3. 下发差异 → Timeline找空泡/接口耗时差异
4. 定点 → hccl耗时最大处入手→跳转→找相邻差距最大处对比
```

常见根因: 数据切分不均/MOE负载不均/下发慢卡(CPU竞争)/条件性预处理

### 5B.4 特殊通信场景

| 场景 | 怎么判断 | 方案 | 为什么有效 |
| ------ | --------- | ------ | ----------- |
| ZeRO3大量小包allgather | advisor报或Timeline中allgather算子数量极多, 每次数据量很小(256byte级) | 切ZeRO2 | ZeRO3每次allgather数据量小, 建链时间>传输时间; ZeRO2减少通信次数 |
| DP allGather带宽极低(<1GB/s) | 通信矩阵显示allGather带宽<1GB/s, 远低于经验值19GB/s | 字节对齐padding | SDMA数据需512B对齐, 不对齐时带宽降10倍(案例: 350ms→50ms) |
| A3线性度不足 | A3多机线性度远低于A2, 通信占比高 | 亲和组1→40组 | A3 Switch架构需要合理配置亲和组才能发挥带宽 |
| 连续多次allreduce | Timeline中同一位置连续多个allreduce | 评估合并为一次 | 减少通信启动开销(建链+同步) |
| plog IO瓶颈 | 小块多频次communication; 共享存储写入慢 | 本地写+ERROR级别 | alltoall中plog 28000条/s写入共享盘阻塞通信 |
| TLS打开 | 预检或下方查询命令返回1 | 执行下方关闭命令 | TLS加密增加通信开销 |

TLS 查询与关闭命令（`i` 为目标设备编号）：

```bash
# 查询 TLS 状态
hccn_tool -i $i -tls -g
# 关闭 TLS
hccn_tool -i $i -tls -s enable 0
```

---

## Phase 5C: 计算方向深入

**案例查询指引**：在计算方向优化前，AI agent 应搜索匹配的算子优化案例作为参考：

- 使用 `project-brain_keyword_search`，`scope="性能工厂"`，`query="性能 TransData"` 或 `query="性能 算子优化"`
- 使用 `project-brain_keyword_search`，`scope="性能工厂"`，`query="性能 AOE"` 或 `query="性能 jit_compile"`
- 在优化建议中引用匹配到的案例，格式：`📖 参考案例：<名称>`

### 5C.1 先看算子占比找大头

**优先优化占比最高的前3个算子**, 不要分散精力。

### 5C.2 常见问题

| 问题 | 方案 |
| ------ | ------ |
| TransData/Cast占比高 | allow_internal_format=False; FP32→FP16(310P) |
| 算子编译耗时 | jit_compile=False; 安装kernel包 |
| 非亲和操作 | 见替换表 |
| vector算子占比高 | 循环→矩阵运算; 融合算子 |
| 亲和优化器未用 | NpuFusedAdamW替换 |
| 非连续tensor | .contiguous() |
| 特定shape算子慢 | AOE调优(见5C.4) |
| CPU执行应在NPU的计算 | 升级版本使NPU支持 |

### 5C.3 非亲和操作替换

**思考提示**: 替换前确认是否影响精度/改变语义, 有疑问先小规模验证。

| 不亲和操作 | 替换 | 注意 |
| ----------- | ------ | ------ |
| X[mask]=0 | X*=~mask | 仅赋0 |
| X[~mask]=value | X=~mask*value+X*mask | 通用 |
| torch.where(x<y,y,x) | torch.lerp(x,y,(x<y).float()) | 反向算子更少 |
| repeat_interleave(v,s) | (zeros+v).T.flatten() | 确认语义 |
| is_nonzero判断BN | 注释 | 仅训练安全 |
| for循环内tensor | 改矩阵运算 | 需理解逻辑 |
| SyncBN | custom版/关闭 | CV常见 |
| to(device) | to(device,non_blocking=True) | 通用安全 |
| attention_mask大量copy | 评估必要性 | 多模态 |

### 5C.4 AOE调优

修改模型参数后某些算子性能骤降时:

```python
torch_npu.npu.set_aoe(dump_path)  # 简化方式
```

完整: 对比profiling→确定算子+shape→生成ge图→aoe --job_type=2→加入知识库

---

## Phase 5D: 推理服务化深入

> ⚠️ **前置检查**：确认 Phase 3（profiling 采集）和 Phase 4（瓶颈定位+查知识库）已完成。
> 推理场景最常见错误——基线跑完跳过 profiling，直接调参数。必定浪费时间。

**案例查询指引**：在推理服务化调优前，从两个知识源搜索匹配案例：

- project-brain：`scope="性能工厂"`, `query="性能 推理 服务化"` 或 `query="性能 vllm"`
- Ascend KG：加载 `vllm-ascend-performance-optimization` Skill，或搜索 `"vLLM 推理 吞吐 decode 优化"`
- 在优化建议中引用匹配到的案例，格式：`📖 参考案例：<名称>`

**前提**: 纯模型已达标。未达标先走5A/5B/5C。

### 5D.1 明确目标

| 目标 | 指标 | 方向 |
| ------ | ------ | ------ |
| 低时延 | TTFT/TPOT | 减Prefill计算; 减调度开销 |
| 高吞吐 | tokens/s | 最大化batch; 减空泡 |

**思考提示**: 时延和吞吐矛盾, 需确认优先级。不确定时先测不限速基线判断空间。

### 5D.x 服务化压测注意事项（vllm bench serve 踩坑经验）

> 以下经验基于 vllm 0.22.1 + Ascend910 + TP4 DP2 with EP 场景实测，适用于 vllm bench serve 工具。

#### 1. max-concurrency 参数决定测试行为

`--max-concurrency` 控制客户端最大并发数，它**改变了 vllm bench serve 的请求调度方式**：

| max-concurrency | 测试行为 | 典型 duration | 典型 Total tok/s |
| ---------------- | --------- | -------------- | ----------------- |
| < num-prompts（如 64） | 按并发数持续发请求，请求均匀分布 | ~140~150s | ~3500~3700 |
| >= num-prompts（如 128） | 先发第一个请求，等它完成后瞬间发完剩余，duration ≈ 第一个请求耗时 | ~90~105s | ~5000~6000 |

**结论**：测最大吞吐时，`--max-concurrency` 必须 >= `--num-prompts`（通常设 128 或 256）。

#### 2. warmup 机制

vllm bench serve 的 warmup 串行发请求，结束后重建 HTTP session。正式测试的第一个请求仍需完整 prefill + decode（~90-100s）。**手动预热对结果影响不大（±5%）**，不需要额外预热。

#### 3. 结果波动

即使参数一致，多次测试也有 ~10% 波动（5000~6000 tok/s），建议至少测 3 次取稳定值。

### 5D.2 MindIE配置(修改前先查原值! 见Phase 2.5)

```bash
# MINDIE_ASYNC_SCHEDULING_ENABLE: 前后处理/layer setup异步双线程执行
# 为什么有效: 同步调度时preprocess→forward→generateOutput串行, NPU空闲等CPU
# 异步后preprocess和generateOutput互相掩盖, 减少NPU空泡
# 怎么判断该设: decode阶段NPU空闲占比高
export MINDIE_ASYNC_SCHEDULING_ENABLE=1
# 原值: echo $MINDIE_ASYNC_SCHEDULING_ENABLE  回退: unset MINDIE_ASYNC_SCHEDULING_ENABLE

# taskset绑核: 将MindIE进程绑到奇数NUMA节点
# 为什么有效: atbmodels自动将NPU算子下发线程绑到偶数NUMA, 如果服务化进程也在偶数NUMA会竞争CPU
# 怎么判断该设: 异步调度后layer setup仍有空泡; CPU利用率不均
taskset -c 168-191 ./bin/mindieservice_daemon
# 回退: 不用taskset, 直接启动

# NPU_MEMORY_FRACTION: NPU显存使用比例
# 为什么设0.96: 最大化显存利用, 支持更大batch/更多KV cache
# 怎么判断该调: OOM时调低到0.92; 显存利用率低时调高
export NPU_MEMORY_FRACTION=0.96
# 原值: echo $NPU_MEMORY_FRACTION  回退: unset NPU_MEMORY_FRACTION

# ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE: workspace内存分配算法
# 为什么设3: 优化内存碎片, 提升显存利用率
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
# 原值: echo $ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE  回退: unset ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE
```

| 参数 | 低时延 | 高吞吐 |
| ------ | -------- | -------- |
| maxPrefillBatchSize | 小(4-16) | 与maxBS同步增 |
| supportSelectBatch | false | true |
| maxQueueDelayMicroseconds | <=100ms | 加大 |
| maxPrefillTokens | - | 1-2万最佳 |

### 5D.3 服务化瓶颈定位

用msServiceProfiler采集**高并发**profiling(不能只采单请求):

| 现象 | 根因 | 方案 |
| ------ | ------ | ------ |
| decode NPU空闲>60% | host非模型计算耗时 | 异步调度; 关后采样 |
| layer setup阻塞 | 同步调度 | 异步调度; 绑奇数NUMA |
| 算子时间=纯模型 | 调度问题 | 优化调度配置 |
| 采样阶段多aclnn算子 | 后处理差异 | do_sample:false |

---

## Phase 6: 长稳劣化排查(Path B专用)

**案例查询指引**：在排查长稳劣化前，AI agent 应搜索匹配的劣化案例作为参考：

- 使用 `project-brain_keyword_search`，`scope="性能工厂"`，`query="性能 劣化"`
- 根据用户场景增补关键词，如 `query="性能 劣化 版本"`、`query="性能 劣化 多机"`、`query="性能 劣化 GC"` 等
- 在排查建议中引用匹配到的案例，格式：`📖 相似案例：<名称>`

### 6.1 先问: 近期有什么变更?

版本升级? 集群重规划? 新增代码? → 有变更先回退验证

### 6.2 波动模式判断

**随机波动(位置不固定)**:

| 怀疑 | 验证 | 方案 |
| ------ | ------ | ------ |
| 未绑核 | 查CPU_AFFINITY_CONF | CPU_AFFINITY_CONF=1 |
| plog写共享盘 | 查plog路径 | 本地写+归档 |
| 数据读取抖动 | 查数据文件大小/存储 | 条带化; 文件>=10G; 增num_workers |
| 集群竞争 | 停其他任务观察 | 资源隔离 |
| RDMA重传 | 查Notify-Wait>4s | HCCL_RDMA_TIMEOUT; 检查交换机PFC |
| 版本不一致 | 对比各机版本 | 统一版本 |
| 后台进程 | bpytop/htop | 清理categraf等 |

**规律性波动**:

| 模式 | 方案 |
| ------ | ------ |
| 周期与CKPT对应 | 本地缓存; 减少路径访问 |
| GC导致 | gc.set_threshold(700,10,1000); CKPT后gc.collect() |
| 随step劣化 | 排查资源泄漏/MOE负载不均/第三方库循环引用 |

### 6.3 GC判断方法

- 抖动趋势多次运行一致
- profile free时段无CPU/NPU/HCCL行为
- gc.disable()后消失

解决(按代价): 增大阈值(~万分之二) > 手动GC(~千分之一) > gc.disable()(~1%)

GC检测:

```python
import gc,os,time,inspect,json
gc_info,save={},[]
def gc_cb(phase,info):
    pid,cur=os.getpid(),time.time_ns()
    if phase=='start':gc_info[pid]=cur
    elif pid in gc_info and cur-gc_info[pid]>=1e8:
        save.append({"pid":pid,"duration":(cur-gc_info[pid])/1e6,
            "stack":[f"{f.filename},{f.lineno}" for f in inspect.stack()]})
gc.callbacks.append(gc_cb)
```

---

## Phase 7: 内存问题 [所有路径]

### 环境变量(修改前先查原值! 见Phase 2.5)

```bash
# PYTORCH_NPU_ALLOC_CONF: 内存池扩展段(虚拟内存特性)
# 为什么有效: PyTorch内存池默认按块分配, 释放后产生碎片无法复用。expandable_segments开启后支持段式扩展, 减少碎片
# 怎么判断该设: 内存碎片导致申请慢; Insight内存视图有大量小碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# 原值: echo $PYTORCH_NPU_ALLOC_CONF  回退: unset PYTORCH_NPU_ALLOC_CONF

# ATB_LAYER_INTERNAL_TENSOR_REUSE: Layer间中间tensor复用
# 为什么有效: 模型forward时每层产生中间tensor, 开启后层间复用这些tensor的内存, 减少显存峰值
# 怎么判断该设: 显存利用率低/有尖刺
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
# 原值: echo $ATB_LAYER_INTERNAL_TENSOR_REUSE  回退: unset ATB_LAYER_INTERNAL_TENSOR_REUSE

# ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE: workspace内存分配算法(推理)
# 怎么判断该设: 推理场景显存碎片多
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
# 原值: echo $ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE  回退: unset ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE

# NPU_MEMORY_FRACTION: NPU显存使用比例(推理)
# 怎么判断该调: OOM→0.92; 显存利用率低→0.96~0.97
export NPU_MEMORY_FRACTION=0.96
# 原值: echo $NPU_MEMORY_FRACTION  回退: unset NPU_MEMORY_FRACTION
```

### 常见问题判断与方案

| 问题 | 怎么判断 | 方案 | 为什么有效 |
| ------ | --------- | ------ | ----------- |
| 内存碎片 | Insight内存视图有小碎片; 申请耗时异常 | expandable_segments:True | 段式扩展减少碎片 |
| checkpoint后劣化 | 保存CKPT后性能持续下降 | 检查代码释放逻辑 | 可能是未释放的tensor导致内存泄漏 |
| 内存尖刺 | Insight内存折线图有短时尖峰, 峰值远超稳态 | 削峰填谷; 增大BS | 尖刺导致显存利用率低(为峰值预留空间) |
| OOM | 运行时报OOM | `grep -nri devmm_mem_stats_show`查各模块占用; 降NPU_MEMORY_FRACTION | 定位哪个模块占用过多 |
| FSDP内存未达上限 | npu-smi info显存占用低但用了FSDP | 不建议用FSDP | FSDP增加通信成本, 内存足够时用DP更优 |

---

## Phase 8: 验证闭环与迭代决策 [Path A/C/D 核心]

> 这是调优工作流中最容易被忽略但最关键的一步。
> **每次修改后必须验证，验证后必须决策：保留还是回退。**

### 8.1 验证流程

```text
修改实施
   │
   ▼
重新压测 / 重新采集 profiling
   │
   ▼
对比修改前后的关键指标
   │
   ├── 达标（达到目标 or 正向收益）→ 保留修改，记录到变更清单
   │
   └── 未达标（无收益 or 负收益）→ 回退修改，记录到变更清单
        │
        ▼
   回到 Phase 4（瓶颈定位），尝试下一个优化方向
```

### 8.2 对比指标

| 场景 | 核心指标 | 辅助指标 |
| ------ | --------- | --------- |
| 推理-高吞吐 | Total tokens/s | Output tokens/s, Peak concurrent requests |
| 推理-低时延 | TTFT (P50/P99), TPOT (P50/P99) | ITL (P50/P99) |
| 训练 | Step time, tokens/s | Free%, 通信占比, 计算占比 |
| 通用 | 显存占用, CPU 利用率 | NPU 频率, 温度 |

### 8.3 判断标准

- **正向收益**：核心指标提升 >= 3%（考虑测试波动）
- **无收益**：核心指标变化在 ±3% 以内
- **负收益**：核心指标下降 >= 3%
- **副作用**：即使核心指标提升，也要检查辅助指标是否恶化（如 TTFT 明显变差）

### 8.4 迭代终止条件

满足以下任一条件即可终止：

1. **达到目标**：核心指标满足用户设定的目标值
2. **收益递减**：连续 3 次优化尝试均为"无收益"或"负收益"
3. **副作用不可接受**：优化带来的副作用（精度下降、显存不足、时延恶化）超过可接受范围
4. **用户确认终止**：用户认为当前结果可接受

### 8.5 迭代记录模板

每次迭代后，在变更清单中追加记录：

```text
=== 迭代记录 ===
迭代 #: 3
日期: 2026-xx-xx
修改内容: max-num-seqs 64→128, max-num-batched-tokens 32768→65536

修改前: Total tok/s = 3683 (duration=142s, TPOT=31ms)
修改后: Total tok/s = 5268 (duration=99s, TPOT=46ms)
收益: +43% ✓
副作用: TPOT 从 31ms 上升到 46ms（batch 增大导致单请求变慢，但总吞吐提升）
决策: 保留

下一步: 尝试增大 max-concurrency 256
```

| 维度 | A2 | A3 |
| ------ | ---- | ---- |
| 机内拓扑 | Fullmesh | Switch(灵渠) |
| Host-Device | PCIe(2x32GB/s) | HCCS(2x150GB/s) |
| 芯片 | 单die | 双die(SIO~140GB/s) |
| 通信配置 | TP>=8最优 | TP=2/4也最优 |
| 甜点 | 稠密模型 | MoE/长序列/大集群 |
| 亲和组 | 1组即可 | 需配40组 |

---

## 经验带宽参考

| 链路 | 带宽 | 饱和点 |
| ------ | ------ | -------- |
| HCCS | 30GB/s(单向) | 32MB |
| SDMA跨片 | ~19GB/s(>16MB时) | 16MB |
| RDMA | >20GB/s | 0.5MB |

---

## 附录A: 昇腾NPU硬件架构速查

### NPU芯片内部结构

```text
                     ┌─────────────────────────┐
                     │        AI Core           │
                     │  ┌───────┐  ┌──────────┐ │
  HBM ←──MTE2────── │  │ Cube  │  │ Vector   │ │
  (显存)  MTE3──────→ │  │(矩阵乘)│  │(向量运算) │ │
                     │  └───────┘  └──────────┘ │
                     │       Scalar(指令分发)     │
                     └─────────────────────────┘
                     ┌─────────────────────────┐
                     │       AI CPU             │
                     │  (控制类任务, 如DMA调度)   │
                     └─────────────────────────┘
```

**关键概念**:

- **Cube**: 矩阵乘法单元, 处理MatMul/Conv等密集计算, 算力最大
- **Vector**: 向量运算单元, 处理element-wise操作(add/mul/relu/cast等), 算力较小
- **AI CPU**: 控制类任务, 不做计算, 负责DMA传输调度、通信控制等
- **MTE2/MTE3**: 内存搬运引擎, MTE2=HBM→AI Core(读), MTE3=AI Core→HBM(写)
- **HBM**: 高带宽显存, 模型权重/激活值/梯度都存在这里, 容量32G/64G

**判断算子bound在哪** (profiling中kernel_details.csv的ratio字段):

- aic_mac_ratio高 → bound在Cube(正常, 矩阵乘密集)
- aiv_vec_ratio高 → bound在Vector(需优化: 减少vector操作/改用cube操作)
- aiv_mte2_ratio高 → bound在内存带宽(数据搬运慢)
- aic_scalar_ratio高 → bound在指令分发(联系算子专家)

### 一级流水 vs 二级流水

训练时, 每个算子从CPU下发到NPU执行经过两个流水级:

- **一级流水(Host侧)**: Python调用 → PyTorch算子分发 → CANN接口适配 → enqueue到NPU队列
- **二级流水(Device侧)**: dequeue出队 → getWorkspacesize(申请内存) → 编译(如需) → 执行

**TASK_QUEUE_ENABLE=2的原理**: 将"参数转换+workspace申请"从一级流水迁移到二级流水, 让一级流水只做enqueue(更快), 二级流水并行做workspace申请和执行

### ND vs NZ格式

- **ND(N-dimensional)**: 常规连续内存排布, 与GPU一致, 大多数算子输入输出用ND
- **NZ(N-Z格式)**: 分块存储(16x32块), Cube单元在NZ格式下矩阵乘效率更高
- **TransData**: ND↔NZ格式转换算子, 自动插入在格式不匹配的算子之间
- `allow_internal_format=False`: 禁止框架自动转换, 全网统一ND, 消除TransData开销
- **权衡**: NZ格式下MatMul更快, 但引入TransData开销; 实测确认哪种更优

### NUMA与绑核

```text
┌──────── NUMA Node 0 ────────┐  ┌──────── NUMA Node 1 ────────┐
│  CPU核0-23  ←→  NPU卡0,1,2,3 │  │  CPU核24-47  ←→  NPU卡4,5,6,7 │
│  内存控制器0                  │  │  内存控制器1                  │
└──────────────────────────────┘  └──────────────────────────────┘
```

- **NUMA**: 多CPU节点架构, 每个NUMA Node有本地内存, 跨NUMA访问内存慢
- **绑核原理**: 将训练进程绑定到对应NPU所在NUMA的CPU核, 避免跨NUMA内存访问
- **CPU_AFFINITY_CONF=1**: 自动绑到NPU对应NUMA核组(训练推荐)
- **CPU_AFFINITY_CONF=2**: 细粒度, 分离下发线程和数据线程到不同核(避免抢占)
- **CPU_AFFINITY_CONF=0**: 不绑核(单卡推理推荐, 因为多线程需灵活调度)

### 通信链路

| 链路 | 全称 | 用途 | 何时出现 |
| ------ | ------ | ------ | --------- |
| HCCS | Huawei Cache Coherent System | 节点内NPU间互联 | TP/PP通信 |
| SDMA | System Direct Memory Access | 节点内跨OS/NPU内存拷贝 | 跨NPU数据搬运 |
| RDMA | Remote Direct Memory Access | 节点间绕过CPU的直接内存访问 | 多机通信 |
| PCIe | Peripheral Component Interconnect Express | NPU-CPU互联 | H2D/D2H数据搬运 |

### A2 vs A3架构差异(为什么A3调优策略不同)

| 维度 | A2 | A3 | 对调优的影响 |
| ------ | ---- | ---- | ------------ |
| 机内拓扑 | Fullmesh(全互联) | Switch(灵渠交换网络) | A3需要配置亲和组才能发挥带宽 |
| Host-Device | PCIe(2x32GB/s) | HCCS Switch(2x150GB/s) | A3的H2D速度快4.7倍, hostbound场景改善 |
| 芯片封装 | 单die | 双die(SIO~140GB/s互联) | A3单卡内可形成双Ring, TP=2也高效 |
| 通信配置 | TP>=8才能发挥最优 | TP=2/4也能最优 | A3并行策略更灵活, 不需要强行TP=8 |
| 亲和组 | 1组即可 | 需配40组 | A3多机必须配置亲和组, 否则通信效率极低 |

---

## 附录B: 常见报错/现象速查

| 现象 | 可能原因 | 快速定位 |
| ------ | --------- | --------- |
| aclopCompileAndExcute耗时超长 | 动态shape未命中缓存, 触发在线编译 | kernel_details.csv看是否有大量编译 |
| step_trace中free时段CPU/NPU都空白 | GC暂停或后台进程抢占 | gc.disable()测试; bpytop查后台进程 |
| 某张卡step末尾有大段空泡 | 代码未释放内存, 导致拖尾 | 对比快慢卡Timeline, 查空泡前最后算子的调用栈 |
| 性能随step逐渐劣化 | MOE负载不均/资源泄漏/第三方库 | 对比前后step的GroupedMatmul shape差异 |
| Notify-Wait超4秒 | RDMA重传 | `hccn_tool -i $i -stat -g\|grep rty`查重传数 |
| allGather带宽<1GB/s | SDMA地址不对齐 | 检查通信数据量是否512B对齐 |
| 开with_stack后出现hostbound | 采集膨胀导致 | 改用L1不开with_stack重新采集 |
| 首step极慢(70s), 后续正常 | jit编译首次开销 | jit_compile=False后首次编译在离线完成 |
| 多机某台机器性能差一半 | CANN版本不一致 | 对比各机CANN/PTA版本 |
