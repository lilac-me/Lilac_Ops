# ops-dev Agent Benchmark（算子开发流水线）

> 先挑的「练手」域。评测对象是 `plugins/ops/` 的三阶段流水线：**Design → Implement → Optimize**（`ops-design-agent` / `ops-implement-agent` / `ops-optimize-agent`）。

## 用例清单（36 个）

`testcases/manifest.yaml` 是单源真相，按 `plugins/ops/ops.yaml` 的 `execution.testcases` 引用。难度分级沿用 ops-design 的工作量估算口径：

| 难度 | 数量 | 代表算子 | 成本预算（时间 / token） |
| ------ | :---: | ------ | ------ |
| easy (VV element-wise) | 15 | relu, gelu, silu, sigmoid, tanh, add, mul, sqrt, exp … | 1h / 150k |
| medium (reduction / 两遍法) | 10 | rmsnorm, layernorm, softmax, mean, sum, var, dropout … | 3h / 350k |
| hard (Cube / matmul / attention) | 8 | matmul, linear, sdpa, rope, swiglu, grouped-gemm … | 8h / 800k |
| complex (多 kernel / 多算子 fusion) | 3 | fused-gelu-add-norm, multi-head-attention, conv2d | 2d / 2M |

每个 case 声明了 `reference`（torch 参考实现，精度/性能 ground truth）、`kernel_type`（VV/CV）、`dtypes`、`target_speedup`（默认 1.0 = 至少平齐 torch）、以及一行 `requirement`（喂给流水线的需求原文）。

## 用法

```bash
# 查看用例
python -m benchmark.agents.ops.runner list-cases

# 给已完成的 run 记账（harness 适配器）
python -m benchmark.agents.ops.runner record-meta outputs/relu \
  --harness dsh --model deepseek-v4 \
  --input-tokens 12345 --output-tokens 6789 --wall-ms 2139052

# 记录一个 DSH run 的真实 token（harness 适配器，读 DSH 会话日志）
python -m benchmark.agents.ops.harness_adapters.dsh_tokens list
python -m benchmark.agents.ops.harness_adapters.dsh_tokens extract \
  --workspace outputs/gelu --turn 3

# 打分一个 run（默认写到 results/ops/<case>/<harness>__<model>/result.json）
python -m benchmark.agents.ops.runner score outputs/relu --case relu

# 汇总所有 result.json 成对比报告
python -m benchmark.agents.ops.report \
  --results benchmark/results --out benchmark/results/_index.md
```

> **结果布局**：`results/<域>/<case>/<harness>__<model>/result.json`。
> 跑次标识放在**目录名**里，不是文件名——同一个 case 在第二个 harness/model 下再跑
> 不会覆盖第一个；文件名统一叫 `result.json`，`report.py` 用一条 `**/result.json`
> 就能收全所有跑法（把跑次塞进文件名会让结果悄悄不出现在报告里）。
> 该布局由 `runner.default_result_path()` 生成，`--out` 可覆盖（覆盖时请沿用同一布局）。

## 当前基线

> **数字一律以 `benchmark/results/` 下的 `result.json` 为准**，本节的表格是它的可读副本；
> 改动评分或重跑后请重新生成，不要手改数字。
> 全量对比表见 `benchmark/results/_index.md`。

### 六个全流程 case 的对比（有硬件、可迭代）

> **这 6 条全部来自同一个被测组合 `dsh` + `deepseek-v4`**（`relu` 是 `reference` 黄金回灌，
> 不是第二个 harness）。所以下表是**该组合的能力基线**，不是跨模型 / 跨 harness 对比——
> Claude Code / Codex / 其它模型的补测是 TODO，见 `benchmark/README.md`。

| case | 难度 | D1 完成 | D2 性能 | D3 工程 | D4 收敛 | D5 成本 | 效果分 | deciding speedup | token | 产出（harness / model） |
| ------ | ------ | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ------ |
| gelu | easy | 100 | 100 | 100 | 100 | 100 | **100.00** | 1.0468 | 286,674 | dsh / deepseek-v4 |
| relu（reference 回灌） | easy | 100 | 97.08 | 100 | 100 | 100 | **99.12** | 0.9708 | — | reference / gold-standard |
| sdpa | hard | 100 | 86 | 100 | 100 | 100 | **95.68** | 0.8561 | 30,961 | dsh / deepseek-v4 |
| swiglu | hard | 100 | 85 | 100 | 100 | 100 | **95.59** | 0.8529 | 61,192 | dsh / deepseek-v4 |
| matmul | hard | 100 | 79 | 100 | 100 | 89.45 | **93.60** | 0.7868 | 506,953 | dsh / deepseek-v4 |
| rmsnorm | medium | 100 | 17 | 100 | 100 | 100 | **75.05** | 0.2525 | 118,506 | dsh / deepseek-v4 |

**分水岭在归约**：element-wise（easy）稳定 99–100，而 `rmsnorm` 掉到 75 ——
失分几乎全在 D2：`tl.sum` 归约比厂商核慢 5–9×，实测各档 speedup 为
fp32 `[4096,4096]` 0.2525 / fp16 `[4096,4096]` 0.3540 / fp16 `[2048,8192]` 0.3579；
且 `[16384,16384]` 因 UB 限制测不了，覆盖率闸门再压一次（2/3），
所以 D2 = 100 × 0.2525 × 0.6667 ≈ 17。

### 两条基线的逐维依据

`reference / relu`（黄金回灌，token 未采集，所以 D5 只由 wall+iter 算出）：

| 维度 | 分 | 依据 |
| ------ | :---: | ------ |
| D1 完成度 | 100 | 19 passed / 0 failed，bit-exact |
| D2 性能 | 97.08 | deciding speedup 0.9708（覆盖率闸门 1.0，3 次重复测量 std 0.0085） |
| D3 工程 | 100 | 10/10 交付物 + Triton 原生（反作弊通过）+ 12 节设计 |
| D4 收敛 | 100 | `memory_bound` + 诚实 `NO_SUGGESTIONS` |
| D5 成本 | 100 | 35.7 分钟；token 历史未采集，故只按 wall+iter 归一化 |
| **效果分** | **99.12** | 0.35·100 + 0.30·97.08 + 0.20·100 + 0.15·100 |

`dsh / deepseek-v4 / gelu`（实跑，带真实 token）：

| 维度 | 分 | 依据 |
| ------ | :---: | ------ |
| D1 完成度 | 100 | 16 passed / 0 failed，bit-exact |
| D2 性能 | 100 | deciding speedup 1.0468（fp16 反超 torch；fp32 0.70× 已登记为已知限制） |
| D3 工程 | 100 | 10/10 交付物 + 反作弊通过 + 12 节设计 |
| D4 收敛 | 100 | `memory_bound` + 真实 erf→tanh 优化（≈2.7×）+ 诚实 `TARGET_MET` |
| D5 成本 | 100 | 286,674 token < easy 档 300k 预算 + 59.8 分钟 + 迭代 1/1/1 |
| **效果分** | **100.00** | 效费比 348.8 效果/百万 token |

> **注意 D5 的口径**：easy 档预算按实测 agent run 反推标定为 300k token
> （见下方「预算校准」），因此 286,674 不超预算、D5 记 100。
> 成本预算是**相对量**：它的作用是分辨越界的 outlier，不是给"省 token"发奖。
> 真正回答"谁更划算"的是并列展示的 token 与效费比，不是 D5。
>
> 两条基线的对比即「效果 vs 成本」的实证：`gelu` 效果满分但花 286,674 token，
> `relu`（reference）效果 99.12 且无 token 记录 —— 同一 `gelu` 在 Claude Code / Codex
> 各跑一遍即可直接比效费比（`harness_adapters/` 契约已就位）。

## 测量口径与预算校准

> **怎么测**：测量实现只有 `bench_runs.py` 一份，在 NPU 容器里以算子工作区为 cwd 运行
> （脚本会自己把 cwd 放进 `sys.path`，**不需要拷贝到工作区**）：
> `cd <算子工作区> && python <repo>/benchmark/agents/ops/bench_runs.py --case gelu --runs 3`。
> 完整说明（含两条硬性前提：先确认卡空闲、必须重复 ≥3 次）见 `benchmark/README.md`。

### 必须重复测量（`bench_runs.py --runs 3`）

单次采样不可信。实测两类噪声源，其余 std < 1%：

| case | 噪声 | 最大 std | 说明 |
| ------ | ------ | :---: | ------ |
| swiglu | fp32 | 0.287（≈13%） | fp32 路径噪声大 |
| sdpa | 小 S | 0.147（≈5%） | S=64 太快，测量窗口小 |

三条结论：

1. **会话内很稳，跨会话仍会漂移**：3 次重复的 std 很小，但 `gelu` 大 shape 首次 1.13、
   后来降到 1.047（~7%）。**重复测量只能压住会话内噪声**；跨会话比较必须固定环境或取多次会话中位。
2. **评分把 std 作为证据带出来**（`performance.worst_speedup_std` + `runs`），
   噪声大的结论不会被当成精确值。
3. **跨会话复现只验证过 `relu`**：重测得 0.732/0.741/0.971 等，与历史黄金运行一致。

### ⚠️ 测量环境是第三个变量（实测踩到过）

补测 sdpa 时撞上 **device 0 被别的任务占着**（`npu-smi`：HBM 45.7GB/65.5GB、AICore 38%）：

| 卡状态 | S=64 的 std | torch S=128 | 可信度 |
| ------ | :---: | :---: | :---: |
| device 0 被占用 | **4.31（32%）** | 89.24us（异常） | ❌ |
| 空闲卡 | 0.077 | 15.31us（正常） | ✅ |

**规则：跑性能基线前必须先 `npu-smi info` 确认目标卡 AICore≈0%、HBM 占用低**，
否则数字无意义。要并行跑多个 case/多个环境时，还必须**一 job 一卡**（同卡并发会让
AICore/HBM 争抢，speedup 直接失真）。

### 预算校准（V7：难度 → 预算）

初版直接沿用 `ops-design` 的**人类工时表**（easy 1h/medium 3h/hard 8h/complex 2d），
实测 agent 表现严重偏离：hard 档预算 8h 而实测只花 ~25min；easy 档 token 预算 150k 而
`gelu` 实际用了 286k（**easy 偏紧、hard 偏松**）。按实测重标定后：

| 难度 | wall | token | 依据 |
| :---: | :---: | :---: | ------ |
| easy | 60 min | 300k | `gelu` 实测 59.8min / 286k → 刚好覆盖 |
| medium | 60 min | 250k | `rmsnorm` 实测 24min / 118k → 有余量 |
| hard | 90 min | 400k | `matmul`/`sdpa`/`swiglu` 实测 21–28min；token 中位 ~61k |
| complex | 180 min | 800k | 无实测，按 hard ×2 外推 |

校准的直接影响：`gelu` 的 D5 从 76 升到 100（easy token 预算 150k→300k），
`matmul` 从 100 降到 89.45（hard token 预算 800k→400k，实测 507k 超预算）。

**已知局限**：token 数受 turn 级上下文放大影响（`matmul` 的 507k 即如此），校准只能算近似；
要精确需「一 case 一会话」隔离跑。`complex` 档无实测，是外推值。

### 权重敏感性

权重 0.35/0.30/0.20/0.15 是人为选定、未标定的，所以要检查换权重后**名次**是否翻转。
`python -m benchmark.agents.ops.sensitivity` 用 6 种权重方案（baseline / equal /
completion_first / performance_first / convergence_first / engineering_first）重排：
**6 个 case 的排名区间都是单点，0 次翻转** —— 排名对权重选择稳健。
（该工具的输出是自检产物，故不落库；随时可重跑复现。）

## 下一步（可选接力）

1. 同一 `gelu`/`relu` case 在 Claude Code / Codex 各跑一遍，填出跨 harness 对比表（`report.py` 自动合并）；
2. 补齐 case 覆盖：36 个里只实跑 6 个，可补 `softmax` / `layernorm` 等其余 medium 档与更多 hard 档；
3. 跨 session 复现性扫描（当前只对 `relu` 验证过）；
4. **接入其它域**（`doc-agent`、`inference-model-optimize`）：本仓**故意先不摆脚手架**，
   接入要点与两个已知的坑见 `benchmark/README.md` 的 TODO 一节。
