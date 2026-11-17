# CANN 环境差异下 fused_infer_attention 数值偏差导致回复乱码（Moonlight-16B-A3B）

## 案例简介

同一份 vLLM 服务配置、同一权重，仅切换 CANN（Ascend 工具链）环境，推理回复从正确
变为整段乱码（复读“建建建…”）。本案例用 msProbe 的“一次 Dump 比对 → 二次细粒度
tensor Dump → 逐元素比对”流程，把整网精度问题收敛为单个算子
`npu_fused_infer_attention_score`（MLA 的 fused infer attention 内核）的数值偏差。

与 `01_cann_upgrade_garbled_all_reduce.md`（all_reduce 乱码）同属“CANN 环境/版本
差异 → 乱码”的一类，但根因算子不同，流程中的“二次 tensor Dump”用 `task="tensor"`
与 `list` 配合限定单算子，而非限定 Embedding 模块。

## 问题现象

| 场景 | 环境 | 请求回复 |
| --- | --- | --- |
| 正常（good） | CANN B050（旧/正确） | “ world by city often dubbed …”正常语义 |
| 异常（bad） | 新 CANN（高分可用版本） | “…建建建建建建建建建建…”整段乱码 |

首 token 即乱码 → 只需采集首次 forward（`step: [0]`）整网数据即可定位。

## 定位环境

- 模型：Moonlight-16B-A3B（MLA + MoE，含共享专家）
- 并行：TP=2、DP=1、EP（`--enable-expert-parallel`、`enable_shared_expert_dp:false`
  、`ascend_scheduler_config.enabled:false`）
- 采样/调度：`--enforce-eager`、`--max-model-len 8192`、`--max-num-seqs 48`、
  `--gpu-memory-utilization 0.9`、端口 8106
- 关键：Python 必须 `export PATH=/usr/local/python3.12.13/bin:$PATH`
- bad 环境 `source .../CANN/bad/.../set_env.sh`；good 用 `CANN/B050/.../set_env.sh`

## 一次 Dump 对比（statistics + 整网）

配置（首帧整网统计，`msprobe_config.json`）：

```json
{
  "task": "statistics",
  "dump_path": "/root/workspace/dump_data",
  "rank": [],
  "step": [0],
  "level": "mix",
  "statistics": { "scope": [], "list": [], "data_mode": ["all"],
                  "summary_mode": "statistics" }
}
```

服务脚本在启动参数 `--additional-config` 里加
`"dump_config_path": "/root/workspace/msprobe_config.json"`。分别用 bad / good 启动、
各发一次请求，得到 `dump_data_bad/` 与 `dump_data/`，再执行：

```bash
msprobe compare -tp /root/workspace/dump_data_bad/step0 \
                -gp /root/workspace/dump_data/step0 \
                -o /root/workspace/compare_out
```

分析 `compare_result_rank*.csv`：

- 大 Max diff 的“整数张量”（`tp_send_counts`、dispatch 的 index/count 等，如
  `npu_moe_distribute_dispatch*.output.5` maxdiff≈2e9）是**索引/计数**，rel≈0、
  res=pass，**不是真分歧**，要过滤。
- 真正看 `Result != pass` 和浮点张量的 Max diff。
- 首 token 整网对比，`layers.0`（第一层）里：embed、input_layernorm、q_proj /
  kv_a_proj / kv_b_proj 输出全部 **maxdiff=0（精确一致）**，而
  `NPU.npu_fused_infer_attention_score.0.forward.output.0` **maxdiff≈0.9863，为整网
  首个开始偏差的算子** → 分歧发生在 attention 内部，且随层数逐层放大。

## 二次细粒度 tensor Dump（限定 FIA，保留输入输出）

要确认“输入一致、输出首偏”，就只 dump FIA 的输入输出真实张量
（`task="tensor"` + `list` 限定算子），bad / good 各一份：

`msprobe_config_fia_bad.json` / `msprobe_config_fia_good.json`：

```json
{
  "task": "tensor",
  "dump_path": "/root/workspace/dump_fia_bad",
  "rank": [],
  "step": [0],
  "level": "L1",
  "tensor": { "scope": [], "list": ["npu_fused_infer_attention_score"],
              "data_mode": ["all"], "summary_mode": "statistics" }
}
```

对应两份 serve 脚本分别引用两个 config。分别起 bad / good 服务、各发一次请求，得到
`dump_fia_bad/step0/` 与 `dump_fia_good/step0/`，其下 `*_rank*/dump_tensor_data/`
保存 FIA 的输入/输出 `.pt` 真实张量（输入输出均已留存）。

### 逐元素比对（Python torch）

按文件名配对加载 bad / good 的 `.pt`，计算逐元素 |diff|：

```python
import glob, torch
def walk(root):
    d={}
    for p in glob.glob(root+"/**/dump_tensor_data/*.pt", recursive=True):
        d[p.split("dump_tensor_data/")[1]] = p
    return d
gb, gg = walk("/root/workspace/dump_fia_bad/step0"), walk("/root/workspace/dump_fia_good/step0")
rows=[]
for k in set(gb)&set(gg):
    tb=torch.load(gb[k], map_location="cpu", weights_only=True)
    tg=torch.load(gg[k], map_location="cpu", weights_only=True)
    if tb.shape!=tg.shape: continue
    d=(tb.float()-tg.float()).abs()
    rows.append((k, d.max().item(), d.mean().item(),
                 d.max().item()/(tg.float().abs().max().item() or 1.0)))
rows.sort(key=lambda r:r[1], reverse=True)
for r in rows[:40]: print(f"{r[0][:70]:70s} maxdiff={r[1]:.4g} meandiff={r[2]:.5g} rel={r[3]:.3g}")
```

### 结果解读

- 预填充（step 0）整网统计：FIA 全部输入（`input.0/1/2`、`query_rope`、`key_rope`、
  `atten_mask`、`scale`）maxdiff=0 完全一致；**output.0 maxdiff=0.9863 首个偏差**，
  `output.1`（注意力分数/softmax 侧）maxdiff=0.4172。
- 生成步的 tensor 级对比里，`kwargs.key_rope`、`input.1` 在 bad/good 间已出现
  rel≈1 的大偏差（如 key_rope maxdiff≈15）——这是 FIA 输出错误写回 KV cache 后
  **逐 token 自放大**的结果，并非新的根因。

## 结论

根因算子为 **`npu_fused_infer_attention_score`（MLA 的 fused infer attention 内核）**
：在输入完全一致的前提下，bad（新 C 仓）环境下该内核输出第一个偏离，随后经层间传播
以及 KV cache 反馈逐 token 放大，最终呈现整段乱码。可据此做单算子复现/向算子团队反馈。

## 经验总结

- 首 token 就乱 → 采集 `step: [0]` 整网即可，不必全量或多步。
- 大 Max diff 的整型张量（count/index）要排除，只看浮点张量 / `Result!=pass`。
- 根因定位后，用 `task="tensor"` + `list` 只 dump 该算子，保留输入输出 `.pt`，
  逐元素配对比较，即可区分“算子自身数值偏差”与“上游传播”。
- 若算子的**输入**在 bad/good 间已偏差，需回溯上游找真正的首偏点；若输入一致而
  输出首偏，则根因在算子内核本身。
