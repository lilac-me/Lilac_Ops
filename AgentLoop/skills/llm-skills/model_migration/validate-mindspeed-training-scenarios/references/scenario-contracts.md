# 训练场景合同

## 目录

1. 场景清单
2. Intentional delta
3. 公共 launcher
4. 拓扑与层所有权
5. Checkpoint cache
6. 日志与 loss
7. Pack/DSA/TND

## 1. 场景清单

每个场景记录：

```text
scenario_id
baseline_id
toggle_bundle
model/spec/checkpoint
data mode/tokenizer/sequence length/seeds
TP/PP/CP/EP/ETP/DP/SP/num-layer-list
MBS/GBS/num_microbatches
recompute flags
terminal/load iteration
log/checkpoint/cache path
```

推荐矩阵：

| 配对 | baseline | candidate | 允许变化 |
| --- | --- | --- | --- |
| recompute | off | on | recompute flag 与其必需参数 |
| TP/SP | TP1, SP off | TPn, SP on | TP/SP 组合及派生 DP |
| PP | PP1 | PPn | PP、合法层列表及派生 DP |
| EP | EP1 | EPn | EP/ETP 及派生 expert DP |

固定长度和 Pack 作为独立 case，不组成逐 step loss 对。

## 2. Intentional delta

启动前把 wrapper 解析为有效参数 map，忽略以下非语义差异：

- 参数顺序；
- shell 引号形式；
- 日志、端口、场景名等允许的运行隔离字段。

然后输出：

```text
parameter | baseline | candidate | declared reason | allowed?
```

未声明变化阻塞运行。特别检查：

- 数据、tokenizer、seed、seq length；
- 模型 Spec 和架构 flags；
- checkpoint/cache identity；
- optimizer、lr schedule、MBS/GBS；
- loss 计算和日志频率；
- 影响样本顺序或 token 分组的参数。

## 3. 公共 launcher

公共 launcher 拥有所有不变量。wrapper 仅导出：

```text
SCENARIO_NAME
TOGGLE_BUNDLE
TP/PP/CP/EP/ETP/SP
NUM_LAYER_LIST
RECOMPUTE_*
LOG_DIR/SAVE_DIR/PORT
```

公共 launcher 支持 `--dry-run`，打印规范化 JSON 配置并在分布式初始化前退出。
runner 串行执行，保存：

```text
scenario | command | start/end time | exit code | log | compare eligibility
```

失败场景不阻止后续独立场景，但依赖它的比较标记为 skipped。

## 4. 拓扑与层所有权

定义：

```text
dense_DP = world_size / (TP * PP * CP)
expert_DP = world_size / (ETP * EP * PP)
```

验证：

```text
world_size % (TP * PP * CP) == 0
world_size % (ETP * EP * PP) == 0
num_experts % EP == 0
GBS % (MBS * dense_DP) == 0
SP => TP > 1
```

显式 `num-layer-list`：

- 元素数等于 PP；
- 元素和等于 decoder 层数；
- PP rank `r` 偏移为前 `r` 项之和；
- shared/full/special/MTP 模块的 producer 与 consumer 所有权合法。

未指定时使用运行时均匀规则，不静默注入模型特定 split。默认 split 不合法时应
要求用户指定或调整，而不是自行替换。

## 5. Checkpoint cache

cache identity 建议包含：

```text
model/source checkpoint digest
converter code/version
TP/PP/EP/ETP
num-layer-list
vocab padding
model spec/architecture flags
MTP/MLA/DSA/MoE registration flags
```

所有 rank 在使用前共同检查：

1. marker 可读；
2. marker 指向目录存在；
3. 预期 shard 集合完整；
4. 每个 shard 非空；
5. 元数据 identity 与当前请求一致；
6. 通过 collective 汇总一致结论。

不完整 cache 不能就地部分使用。由单个受控转换阶段重新生成到临时位置，完成后
原子发布 marker，其他场景只读复用。

## 6. 日志与 loss

日志头写入规范化实验合同。loss parser：

- 解析实际 iteration 与 loss；
- 拒绝重复、缺口、非有限值；
- 允许从非零 checkpoint iteration 开始；
- 只比较相同起止且连续的交集；
- 不按文件行数或 `TRAIN_ITERS` 推断 step。

配对结果：

```text
reference_mean
candidate_mean
relative_mean_difference_percent
start/end/count
start_window_mean/end_window_mean
linear_slope
converged
passed
```

主门禁为：

```text
relative_mean_difference_percent < 1.0
reference_converged
candidate_converged
```

窗口默认 5；短于两倍窗口时明确使用可用长度或仅报告趋势不足，不能伪造收敛。

## 7. Pack/DSA/TND

记录 tensor 合同：

| Tensor | 生产者 | shape | dtype/device | 消费者 |
| --- | --- | --- | --- | --- |
| packed tokens | dataset/collator | `[T,...]` | model dtype/NPU | attention |
| cu_seqlens_q/k | collator | `[docs+1]` | int32/NPU | attention |
| actual_seq_qlen/klen | adapter | `[docs]` | contiguous int32/NPU | DSA/TND op |
| sparse indices | indexer | operator-defined | integer/NPU | sparse attention |

每个消费者确认是否需要累计零、end positions、最大长度、flattened token 或
batch 维。mask 必须隔离文档并保持各文档内部因果关系。

不支持组合要在 preflight 给出组合名、期望合同和实际值，不能等到多 rank kernel
报通用 shape 错误。
