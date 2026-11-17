# 基于服务端 Maximum concurrency 的并发预算推算

> 目的：让后续每次新模型/新拓扑调优，都能先通过服务端自身日志推算"并发预算"，
> 避免只按固定 workload 对比而漏测高并发潜力组合。

## 一、日志位置与格式

vLLM 启动完成后，服务端 stdout 会按**每个 DP rank** 打印一行：

```text
(EngineCore_DP0 pid=22288) INFO 08-21 15:10:50 [kv_cache_utils.py:2236] Maximum concurrency for 36,864 tokens per request: 33.38x
(EngineCore_DP1 pid=22289) INFO 08-21 15:10:50 [kv_cache_utils.py:2236] Maximum concurrency for 36,864 tokens per request: 33.38x
...
```

- `36,864`（或 `40,960`）= **实际的 `--max-model-len`**
- `33.38x` = 服务端估算的"打满 KV cache 时可支撑的最大并发数"（基于 max-model-len、
  KV cache 容量、每请求 token 预算）
- **每个 DP rank 一行** → 单 rank 预测 × DP 数 = 总并发预算

## 二、读取命令

```bash
# 服务端 stdout 日志路径（runtime 目录）
grep "Maximum concurrency" /home/<工号>/serving/<timestamp>_<ip>/stdout.log

# 或直接看当前/最近一次启动
grep -h "Maximum concurrency" /home/<工号>/serving/*/stdout.log | tail -20
```

## 三、总并发预算推算

```text
总并发预算 ≈ Σ(各 rank Nx) = DP 数 × 单 rank 预测
```

示例（Qwen3.5-35B-A3B，8 卡，max-model-len=36864）：

| 配置 | 每 rank 预测 | DP 数 | 总并发预算 |
| --- | --- | --- | --- |
| TP8/DP1 | 177.96x | 1 | ~178 |
| TP1/DP8 | 33.38x | 8 | ~267 |
| TP4/DP2 | 155.97x | 2 | ~312 |
| TP2/DP4 (gmu0.9, seqs128) | 118.57x | 4 | ~474 |
| TP2/DP4 (gmu0.92, seqs176) | 170.48x | 4 | ~682 |
| TP4/DP2 + DCP=2 (gmu0.9) | 289.67x | 2 | ~579 |
| TP4/DP2 + DCP=2 (gmu0.92) | 298.52x | 2 | ~597 |

关键观察（实测）：

- **DP 越大 → 总并发预算越高**（DP=4 ≈ 474-682 > DP=2 ≈ 312 > DP=1 ≈ 178）
- **DCP=2 让单 rank 预测翻倍**（155.97 → 289.67），DP=2 时总预算 ~579-597
- gmu 0.9→0.92 小幅提升（579 → 597）；max-num-seqs 从 128→176 主要受 HCCL_BUFFSIZE
  检查约束，不直接反映在预测值

## 四、选档位规则（下次新模型必须遵守）

1. 每次启动服务后**必须**读取并记录各 rank 的 `Maximum concurrency`（时序归档：
   配置 → 单 rank 预测 × DP 数 = 总预算）
2. 除用户固定 workload 对比外，**对总并发预算相近（±20% 内）的拓扑组合，加测高并发
   档位**（预算的 ~50% / ~75% / ~100%），确认真实吞吐上界
3. 不能因为"固定 workload 已达标/收敛"就跳过——低并发只能测得下限；预测能撑满的
   配置才是真实能力上限（例：本次 TP4/DP2+DCP2 预测 ~579 但仅按 128 并发对比，
   TP2/DP4 ~474-682 也仅按 128 测过，高并发潜力组合被漏测）
4. 高并发档位测试时注意 `HCCL_BUFFSIZE`（MoE dispatch workspace 检查拦截，
   NEEDED 随 max-num-seqs 线性增长，报错见 `vllm-ascend-serving` SKILL.md 已知问题
   —— 先 grep `HCCL_BUFFSIZE`/`NEEDED_HCCL`/`Check UB/window` 再归因）
5. **HCCL_BUFFSIZE 调低实验**：该值占用每 rank 显存，设得越大越浪费、可能压缩 KV
   降低并发预算。高并发档位测试前做一次调低实验：**去掉 `HCCL_BUFFSIZE` env（恢复默认
   200MB）跑一档**，报错日志会打印 `NEEDED_HCCL_BUFFSIZE=xxxMB`，据此配一个**更大的
   临近值**（NEEDED=445 → 设 600 即可；勿一步设 2400 这类大值）；调整后读回
   `Maximum concurrency` 验证 KV 预算是否改善（本例 TP4/DP2+DCP2 在 BUFFSIZE=2400 下
   预测 319.81x/rank，调低后有望更高）。

## 五、统一档位集与自动加测（推荐流程）

- **统一档位集**：`[128, 256, 384, 512, 768, 1024]`。同一任务内所有配置/候选使用
  同一套档位，保证公平比较；`--num-prompts` 恒 = `--max-concurrency`（稳态覆盖，
  无尾部空闲）。
- **自动加测**：服务启动后读预算。总预算 > 当前档位集上限对应档位时，自动补测
  预算 50%/75%/100% 最近档位，不逐次询问用户（一次授权管整个任务）。
- **加测前置条件**（否则加测无效）：
  - 服务端 `--max-num-seqs` ≥ 目标并发（调度器上限，低于 bench 并发时并发被截断）
  - `HCCL_BUFFSIZE` 按 maxBs 线性放大（maxBs 128→1024 时 NEEDED≈445MB×8≈3.6GB，
    按日志 NEEDED 值设置，注意每 rank 显存占用、KV cache 压缩）
  - 启动后读回实际 `Maximum concurrency` 校准（HCCL_BUFFSIZE/gmu 变化会改变预测值）
- **最优点判定**：每配置画"吞吐-并发"曲线，取最高档位为饱和点；跨配置比较用**统一
  档位**下的吞吐（避免"配置 A 在 128、配置 B 在 579"这种不可比比较）。

## 六、任务开始前的用户提醒模板

开始寻优/全量对比任务前，先向用户说明以下默认行为并请求获准（一次获准覆盖整个任务）：

```text
我会在每次服务启动后读取 vLLM 的 Maximum concurrency 日志推算并发预算；
若预测并发高于当前 benchmark 档位，会自动加测更高并发档位（统一档位集
128/256/384/512/768/1024 内就近落档）来寻找真实吞吐上界，不再逐次询问。
固定 workload 的对比基准会保留。
```
