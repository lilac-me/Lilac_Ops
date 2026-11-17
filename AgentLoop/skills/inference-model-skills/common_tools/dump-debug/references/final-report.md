# Precision Debug 最终报告

## 1. 报告原则

从问题接入开始维护本报告的运行副本，并保存到 `<RUN_RECORD_DIR>/precision_debug_final_report.md`；
每完成一个阶段立即更新，不得只在全部流程结束后汇总。报告中的证据使用相对于 `<RUN_RECORD_DIR>`
的位置；未生成的证据写“未生成”并说明原因，不使用预期位置冒充实际产物。最终回复只给出运行记录
目录和最终报告，不展示 skill 安装目录、内部脚本路径或 reference 路径。

每个被执行的命令都必须落盘保存可复现记录：完整命令或参数化脚本、执行环境/输入配置、开始与
结束时间、stdout、stderr 和退出码。敏感凭据必须脱敏。

阶段状态：

| 状态 | 含义 |
|------|------|
| `CONFIRMED` | 收敛到首个偏差算子，并完成细粒度 I/O 交付 |
| `TRIAGED` | 预检已确认根因属配置类（权重 / 版本 / 采样 / 图模式开关 / 输入模板未对齐），无需进入 Dump |
| `PARTIAL` | 仅定位到模块未收敛到算子，或部分证据缺失 |
| `BLOCKED` | 因环境、msprobe 不可用、dump 不收敛或证据缺失无法完成定位 |
| `NOT_RUN` | 用户未要求或前置阶段未执行 |

整体状态：

- `CONFIRMED`：环境确认、预检分诊、P1 分级 Dump、P2 细粒度、P3 交付全部完成且证据完整。
- `TRIAGED`：预检 L0–L3 之一命中并完成单变量回代，或 C2 判出输入模板未对齐，根因属配置类，
  明确不需要 Dump。
- `PARTIAL`：只完成部分阶段；其余阶段为 `NOT_RUN`。
- `BLOCKED`：某一必需阶段因证据不足无法判定，且无明确 `CONFIRMED` / `TRIAGED` 结论。

## 2. 最终报告模板

```markdown
# vllm-ascend 精度定位报告

## 结论
- 整体状态：<CONFIRMED|TRIAGED|PARTIAL|BLOCKED>
- 入口分支：<A 单 curl 可复现 | B 仅 RL 复现 | C 数据集掉点 | 崩溃 | 未复现>
- 问题形态：<乱码 | 复读 | 错 token>
- 输入口径：<一致 | 固定偏移 N tokens（疑 template 未对齐） | 长度一致但 token 不同 | 偏移不固定 | 未执行>
- 可疑算子：<首个偏差算子名>
- 怀疑算子清单：<L4 报出的算子（标注是否随机性算子）| L5 补出的算子 | 置信度>
- Dump 范围：<定向 Dump | 全局 Dump | 未做 dump 确认>
- 判定类型：输入一致、输出首偏 → 算子内核数值偏差；输入已偏、输出跟随 → 上游首偏点算子
- 运行记录：<RUN_RECORD_DIR>
- 模型 / 芯片 / 并行：<model path> / <SOC_VERSION + ASCEND_RT_VISIBLE_DEVICES> / <TP·DP·EP>
- bad / good 环境：<两套 set_env / CANN 路径与版本>

## 阶段汇总
| 阶段 | 状态 | 核心结论 | 主证据 |
|------|------|----------|--------|
| 环境确认 | <status> | vllm/vllm-ascend 路径与 commit、CANN、msprobe | <env_<side>.txt> |
| 入口分诊 | <status> | 单 curl / 仅 RL / 数据集掉点 / 崩溃 | <3 次响应与日志> |
| C2 输入口径核对 | <status> | 输入 token 长度是否固定偏移、token id 是否一致 | <input_len_diff.jsonl、input_len_summary.json> |
| L0–L1 数据与配置 | <status> | 权重 / 配置版本是否致因；两侧均 NPU 时 CANN 差异如何处置 | <model_diff_*.json、version_diff.txt> |
| L2–L3 开关类 | <status> | 采样 / 图模式是否致因 | <determinism_matrix.csv、graph_vs_eager_*.csv> |
| L4–L5 算子层 | <status> | 怀疑算子 / 算子选择差异；随机性算子是否已排除；用户选择的 Dump 范围 | <accuracy_checking_*.csv、op_diff.csv> |
| Dump 分流 | <status> | D1 定向 / D2 全局 / D3 batch / RE 图模式 | <分流判定记录> |
| P1 分级 Dump | <status> | 首个偏差模块 / 算子 | <compare_result_rank*.csv> |
| P2 细粒度 | <status> | 输入一致、输出首偏 | <data/bad|good/rank*/...*.pt> |
| P3 交付 | <status> | 可疑算子 + 调用源码 | <dump_output/ 产物> |

## 预检结论
- 命中层：<L0|L1|L2|L3|L4|L5|L6|C2|未命中>
- 命中类型：<采样 / 图模式 / 权重 / 配置版本 / 算子 / 算子选择差异 / template 未对齐 / 无>
- 单变量回代：<如何回代、现象是否复现>
- CANN 差异处置：<不适用 | 用户选择拆包定位（cann-package-split-diagnose） | 用户选择整网 Dump 比对>
- 随机性算子排除：<acc_check 报错但属随机性算子的列表；关闭随机源后是否仍报错>
- 已排除项：<逐层列出并附证据名>
- 未执行项及原因：<例如无 GPU 环境未做 GPU 侧 acc_check>

## 限制与未完成项
- <例如：只有单环境、无 GOOD 基线，L2/L3 为降级比对>
- <例如：GPU 侧 acc_check 与 api_precision_compare 未执行>
- <例如：训推一致性仅给出建议，未代为执行>

## 交付件清单
只登记已落盘且核验存在的文件。`SHA256` 证明引用的脚本、配置和关键结果与实际执行版本一致；
不适用时写 `N/A` 并说明原因。

| 阶段 | 类型 | 相对路径 | SHA256 | 说明 |
|------|------|----------|--------|------|
| P1 | compare csv | <compare_result_rank*.csv> | <hash> | 首个偏差行，如 `out.op.forward.output.0` maxdiff≠0 |
| P2 | I/O 张量 | <data/bad|good/rank*/...*.pt> | <hash> | 算子输入输出真实张量 |
| P2 | dump 配置 | <msprobe_config_{stat|tensor}_{bad|good}.json> | <hash> | L0/L1 dump 配置 |
| P3 | 采集脚本 | <serve_{bad|good}_tensor.sh>、<client.sh> | <hash> | 复现/采集脚本 |
```

## 3. 最终输出内容（P3.1 交付格式）

交付正文按固定小节结构输出（保证“算子 + 证据 + 数据 + 源码”四要素齐全、可复现）：

1. **定位结论**：一句话给可疑算子 + 判定类型。
2. **判定依据（证据表格）**：首偏点 bad/good 逐元素比对表（输入 maxdiff=0 / `output.0` 首偏
   res=error / 高层 KV cache 自放大），附 good 正常 / bad 乱码两行复现。
3. **交付数据路径**：固化到运行脚本所在路径下的 `dump_output/` 目录（I/O `.pt`、
   `compare_result_rank*.csv`、`msprobe_config_*.json`、`serve_*_tensor.sh`、`client.sh`）。
4. **调用该算子的源码**：`文件:行` + 关键代码片段（例：`torch_npu.npu_fused_infer_attention_score(
   query, key, value, **common_kwargs)`，见 `vllm_ascend/attention/mla_v1.py`）。
5. **备注**：环境/坑点（如 CWD 下 `profile.py` 遮蔽标准库、是否开启确定性、容器重启丢变量等）。
