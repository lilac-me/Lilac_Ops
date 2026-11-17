# S4 调优扫描循环重构 + Patch 导出设计（count-agnostic + 编排器驱动 + 主动交付）

> 状态：**设计文档（先写文档，再开发测试）**。
> 上游正本：`triton-latency-optimizer`（vendored）的 `references/Index.md` 33 点有序扫描。
> 借鉴对象：`plugins-official/triton-op-generator/AGENTS.md` **Phase 4**（性能优化迭代循环）。
> 本工作流正本：`profile-driven-triton-opt`（即本 skill）。
>
> 本文件两大部分：
>
> - **A 部（§1–§9）**：S4 调优扫描循环重构（count-agnostic 状态机、scan_trace 契约）。
> - **B 部（§10，扩展）**：Patch 导出（S4.5）——把采纳的优化作为用户可直接引用的统一补丁交付（主动交付 + 契约强制）。

## 0. 摘要（TL;DR）

把 S4 内层从「**字符串证物 + 一次只应用一个点的纪律**」升级为「**可机械推导扫描完整性的无状态状态机**」：

- **去硬编码**：不再假设"33 点"/"31 点"，点数由编排器每次从 `Index.md` **当场读取**（`N = 当前表格最大点号`），扫描区间推导全部以 `cursor` + `hit_optimization_point` 为基数，`None → [cursor, N]` 全未命中。
- **编排器持有循环**：点扫描循环由本工作流（调用方）驱动，逐轮 `Skill(triton-latency-optimizer)` 强制调用 + 官方 verify/benchmark；`scan_complete` 由编排器推导，**不采信 skill 自报**。
- **证据结构化**：`summary.json.optimizer` 增加 `scan_trace`（逐轮 `scan_from/hit/not_hit_interval`）与 `points_considered`（N），`check_summary_contract.py` 据此做 **count-agnostic 穷尽闸**——把"我合并判断了 3-31"这类手写式声明变成自洽区间序列，杜绝静默跳过。

## 1. 背景与动机

### 1.1 现状缺口（本轮暴露的三个问题）

1. **无机械扫描驱动的循环**：当前 `AGENTS.md` S4 ①③只写"按编号递增有序扫描、一次只应用一个点、循环由本工作流调用方持有"，但没有**状态变量**（cursor/scan_from/excluded/banned/scan_complete）也没有**区间推导规则**，导致实际执行时可以"合并判断 3-31 均不满足"写成 `points_hit=[2,33]` 却不留逐点扫描证据。
2. **证据是字符串、非结构化**：`points_hit` / `loaded_docs` 只要求存在，`check_summary_contract.py` 的调优执行者闸只做 presence 校验，**不做穷尽性校验**——没法区分"真逐点扫过"和"手写了一条声明"。
3. **点数硬编码 + 正本自身漂移**：prompt 层多处写死"33 点"，而 `Index.md` 表头自述"共 31 个优化点"但表格实际列了 33（32 Autotune / 33 Block-Size-Scaling），**表头与表格已不一致**——这正是"点数会增加"的现实证据。一旦优化点增删，硬编码的常量、区间上界全部失效。

### 1.2 为什么不能继续硬编码

- `Index.md` 的优化点由 `triton-latency-optimizer` 维护，未来增删点（如新增 operator 专用点、拆分子点）不需要本 skill 联动改码。
- 区间推导一旦与"最大点号"绑定，就必须**当场读表**而不是写死常量。

## 2. 目标与非目标

### 目标

- `scan_complete` 与"某段未扫描"全部可由**数值推导**，不依赖 skill 或用例自报。
- 点数变化（31→33→N）零改动兼容：编排器每轮读 `Index.md` 得 N。
- 每轮**必须真实 `Skill(triton-latency-optimizer)` 调用**，禁止 Read-then-inline。
- summary 产出**可审计的扫描轨迹**，契约脚本做自洽性穷尽校验。

### 非目标（与 generator 的刻意差异）

- **不移植 mfu 模式**：本工作流只做 `speedup_vs_user`（分母唯一=用户源）单口径。
- **不移植多-case kernel 分裂的整条复杂度**：本工作流主测试点是**单一 profiled shape**（`test_shapes.json`），无几何平均、无 `total_cases>1` 分裂再扫描的强制分支。
- **不移植 IR 多轮（ir_max_iterations/opt_iter_n_ir_k 目录）**：IR 分析优化点（当前表中 31）在普通轮内命中一次即视为普通轮；不做跨目录 IR 子循环。
- **不移植 50 轮上限**：本层默认用轻量预算（`max_opt_iterations`，默认小值），由 config 控制。

> 取舍原则：只搬 generator 能解决"扫描完整性可推导 + 执行者唯一化"的最小机械结构，不搬景观化复杂性（多 case 几何平均等）；MFU 判定通道作为可选 `metric_mode` 按需接入（见 `SKILL.md` S4 判定口径），不属于扫描状态机。

## 3. 红利——从 generator Phase 4 搬来的最小结构

generator Phase 4 已把这些做对了，扫描状态机部分照搬（不含多 case 几何平均）；mfu 判定另见 `SKILL.md` S4.0：

| 机制 | generator 原文要点 | 本工作流移植 |
| --- | --- | --- |
| 状态变量 | `cursor/scan_from`、`code_version`、`excluded`、`fail_count`、`banned_points`、`scan_complete` | 全保留 |
| 区间推导 | `return P → [cursor, P-1] not-hit`；`return None → [cursor, 31] not-hit`（本层改为动态 N） | 全保留，上界改 **N** |
| 强制 Skill 调用 | "本轮无 `Skill(triton-latency-optimizer)` 调用即视为 4.1 未执行" | 全保留 |
| scan_complete 推导 | 由编排器算，不采信 skill 自报"已扫完" | 全保留 |
| 无效命中防护 | 无提升 → `excluded += [P]`、`cursor = P+1`、`fail_count[P]++`、3 次 → `banned_points` | 全保留 |
| 终止保证 | 每轮要么 `cursor` 前进要么 `code_version` 递增（metric 单调有界） | 全保留 |
| KKB 优先级 | latency-optimizer 每轮前自行调 `triton-knowledge-retrieval` 取瓶颈→点映射、优先高收益点 | 保留为 latency-optimizer 内部行为，不重写 |
| simulator 采集驱动 | 优化耗尽且未达 target → 转 `triton-simulator-optimizer`，采 MMAD 证据 → 回喂强制命中点 | 保留（见 §4.4） |

## 4. 新 S4 内层设计

### 4.0 入口硬断言

进入调优循环前，读取该 candidate 上一验证产物，确认 `passed == total`。断言失败 → **C 类终止，不得进入调优循环**（对应 generator 4.0）。

### 4.1 状态变量（S4 内层，每 candidate 一份）

```text
opt_iteration = 0
max_opt_iterations = <config.max_opt_iterations，默认不写死 50>
no_improve_streak = 0
target_speedup = <config.target_speedup，默认 2.0>
hit_history = []

# ── 扫描状态机（编排器持有循环，扫描完整性由编排器推导）──
N = <每轮从 triton-latency-optimizer/references/Index.md 表格当场读取的最大点号>  # count-agnostic
cursor = 1                # 下次 scan_from
code_version = 0          # 每采纳一个优化 +1；变更即清空 excluded、cursor 回 1
excluded = []             # 【版本作用域】本 code_version 下已判无提升的点
fail_count = {}           # 【跨版本】各点累计"命中但无提升"次数；>=3 永久禁用
banned_points = []        # 永久禁用；与 excluded 合并为 exclude_points
scan_complete = false     # 从 cursor=1 走完一整趟仍无命中 → true（编排器推导）

# ── 迭代目录 ──
round_dir = op_{name}/opt_iter_{n}   # 每轮独立产物目录

# ── simulator 采集 ──
simulator_attempted = false
```

### 4.2 迭代循环骨架

```text
while opt_iteration < max_opt_iterations 且  not scan_complete:
   4.2.1  强制回调 Skill(triton-latency-optimizer)
           入参：baseline_code / 上一轮优化代码、opt_iteration、
                 scan_from=cursor、exclude_points=excluded ∪ banned_points
           返回：hit_optimization_point = P | None；每条变异前必须加载 P 的参考文档
   4.2.2  区间推导（编排器，数值）
           P != null → not_hit = [cursor, P-1]
           P is None → not_hit = [cursor, N]
                        若 cursor == 1 → scan_complete = true（进 4.4 终局）
                        若 cursor >  1 → cursor = 1，continue（从头整趟复查）
           P != null → 加载 P 参考文档 → 改码 → AST 预检查 → 4.2 精度验证 → 4.3 性能测定 → 4.4 判定
   4.2.3  记录 round 到 scan_trace
   4.2.4  精度/性能判定走 §4.3
```

### 4.3 判定与处理（每轮）

- **优化成功**（`speedup_vs_user > best_speedup`，best_code 更新）：
  - `code_version += 1`；`excluded = []`；`cursor = 1`（代码变了，旧 not-hit 结论失效，须允许重扫）
  - `no_improve_streak = 0`；`target_reached = (best_speedup >= target_speedup)`
  - 命中点采纳 → 记录 `best_config`。
- **无提升**（命中点 P 但 verify 未过 / benchmark 无提升 / 相等）回退 best_code：
  - `no_improve_streak += 1`
  - **无效命中防护**：`excluded += [P]`（仅本 code_version）；`cursor = P + 1`；`fail_count[P] += 1`，跨 3 个 code_version → `banned_points += [P]`
- **命中点 P 的 verify 未过校验**：不入 `excluded`（不是"无提升"，是"引入错误"），回退并按 A 类继续，`cursor` 不推进（下次仍从 P 重新评估）。走 4.5 分析。

> **终止保证**：每轮要么 `cursor` 前进（有界 ≤N），要么 `code_version` 递增（需真实提升，metric 单调有界），循环必然收敛。

**注（critical）**：命中点 P 的**位移与排除语义**要区分两类"失败"：

- verify 未过 → 代码引入错误（A 类），`cursor` 保持 = P，回退继续，从 P 重试别的方向。
- verify 过但无提升/无变化 → 该点在当前 code_version 无收益（无效命中），`cursor = P+1`，加入 excluded。

两者混用会导致要么原地打转、要么错误地永久排除有过错的点。

### 4.4 终局 / 优化耗尽

退出前置门（任一满足才可出循环）：

- (a) `scan_complete == true`（整趟无命中）；**或**
- (b) `opt_iteration >= max_opt_iterations`；**或**
- (c) `target_reached == true` 且 next-scan_complete（已扫完能给的都给了）。

**优化耗尽且未达 target 时（禁止直接判 BELOW 收场，须先走 simulator）**：

- `simulator_attempted == false` → 强制回调 `triton-simulator-optimizer`（只采集+诊断：msprof → pipe 占比 → 修复方向 = latency-optimizer 点编号；`MMAD > 50%` 判"真·硬件极限"）。
- 拿到诊断的修复方向 → 回到 4.2.1，把诊断指向的点作为**强制命中点**传入（覆盖静态命中判断），再验证一轮。
- 诊断报无对应优化点（`MMAD > 50%` 且放大 tile / bf16 化均不可行）→ 才可终局判 BELOW。
- 设置 `simulator_attempted = true`。

**终局判定（§三态）**：`best_speedup >= target_speedup 且 optimizer_rounds>0 且 非空 best_code → REACHED`；否则 `BELOW`；verify 全流产 / 环境错误 → `FAILED`。

### 4.5 与 profile 主测试点对齐

- 主测试点 = profiled shape（`assembly/{name}/test_shapes.json`），verify + benchmark 均用该 shape。
- 单 shape 下无几何平均、无 kernel 分裂再扫描分支。`Index.md` 中 `total_cases>1` 要求（点 17）在本工作流视为不命中。

## 5. summary.json 契约扩展（count-agnostic 穷尽闸的输入）

在现有 `optimizer` 块内新增两个字段，其余（tool/points_hit/loaded_docs/waiver）保留：

```jsonc
"optimizer": {
  "tool": "vendored-triton-op/triton-latency-optimizer",
  "points_hit": [2, 33],                    // 实际采纳过的点（含无提升的）
  "loaded_docs": ["...tiling_optimization.md", "...block_size_scaling.md"],
  "points_considered": 33,                  // N = 本轮从 Index.md 当场读取的最大点号（count-agnostic）
  "scan_complete": true,
  "scan_trace": [
    // mandatory = 本轮收尾作为最后一步真实执行的强制点（Index.md MANDATORY_PER_ROUND，如点 31 IR 分析）。
    // 应用改动轮（hit 非 null）必须携带；整趟复查轮（hit=null）豁免但建议也执行并记录。
    { "round": 1, "scan_from": 1,  "hit": 2,  "not_hit": "[1,1]",  "mandatory": [31] },     // cursor=1 → hit=2
    { "round": 2, "scan_from": 3,  "hit": 33, "not_hit": "[3,32]", "mandatory": [31] },    // cursor=3 → hit=33
    { "round": 3, "scan_from": 1,  "hit": null, "not_hit": "[1,33]", "scan_complete": true }
  ],
  "waiver": null
}
```

**区间推导规则（写死进契约校验，数值可验）**：

- round 的 `not_hit` 必须等于该 round 的推导值：`hit=P → [scan_from, P-1]`；`hit=None → [scan_from, points_considered]`。
- `scan_from` 序列必须连续：`next.scan_from` 只能是 `prev.hit + 1`（无效命中前进）、`1`（code_version 变更 / 整趟复查），或该轮为 `scan_complete` 终止轮。
- `scan_complete == true` 当且仅当存在**从 cursor=1 出发、最终 hit=None 且 not_hit 覆盖 `[1, N]`** 的轮次。
- `points_considered` = N 必须 ≥ 每个 not_hit 区间的上界，且 = 所有扫描区间的并集上界（防止用假 N 凑。

## 6. check_summary_contract.py 增强

在现有 `_check_optimizer_evidence`（presence 闸）之上**新增** `scan-trace 穷尽闸`：

1. **存在性**：`optimizer_rounds>0` 时必须带 `points_considered` + `scan_trace`（≥1 条）。缺 → FAIL（禁止退回旧的 `points_hit=[2,33]` 手写式声明）。
2. **self-consistency**：逐条校验 `scan_trace` 的 `not_hit` 是否等于 `[scan_from, hit-1]`（hit 非空）或 `[scan_from, points_considered]`（hit 空）；不符 → FAIL。
3. **连续性**：非终止轮的 `next.scan_from ∈ {prev.hit+1, 1}`；`prev.hit == null` 的非轮外跳（除 scan_complete）→ FAIL。
4. **scan_complete 语义**：`scan_complete=true` 当且仅当存在一个 round `scan_from=1` 且 `hit=None`（覆盖到 N）。否则 FAIL。
5. **count-agnostic**：脚本**不写死** 33/31/N；N 取 `points_considered`。若未来 `Index.md` 变 34，只需 summary 写 `points_considered=34`、trace 用 `[..,34]`，脚本零改动放行。
6. `optimizer.waiver={reason}` 仍可公示放行（PASS_WITH_WAIVER），但必须显式声明"未提供完整 scan_trace"。
7. **强制点闸（新增）**：`Index.md` 顶部机器可读标记 `MANDATORY_PER_ROUND: 31`（单一事实源，
   禁止 self-declare/硬编码）。校验时 `--index <Index.md>` 读取该标记成集合 M：
   - `optimizer_rounds>0` 且缺 `--index` → FAIL「必须从 latency-optimizer skill Index.md 读取强制点，禁止自报」；
   - `--index` 给出但读不到 `MANDATORY_PER_ROUND` → FAIL「Index 缺强制点标记，禁止编造」；
   - 对每条 `hit` 非 null 的 scan_trace 记录，要求 `M ⊆ record.mandatory`，任一应用轮缺失 → FAIL（
     `optimizer.waiver={reason}` 可公示放行）。
   - 根因：count-agnostic 单次扫描首中即停会把点 31（每轮必须执行）折进 `not_hit` 区间静默跳过，
     此闸把「必须执行的点是否真的执行」变成可数值自洽的强制约束。

> 脚本不改动既有 perf_baseline/gate_ref/bench 溯源闸；只强化调优执行者闸的**穷尽性**维度。

## 7. Prompt 层同步（文档/开发落地）

| 文件 | 变更 |
| --- | --- |
| `SKILL.md`（canonical + plugin） | §依赖 skill 的"33 点有序扫描"表述改为"读 Index.md 当前点数 N，cursor 驱动有序扫描（count-agnostic）"；编排约定补 scan_trace/points_considered 契约；写死 33 的字眼全删。 |
| `AGENTS.md`（canonical + plugin） | S4 内层整节改写为 §4 状态机：状态变量、区间推导、强制 Skill 调用、scan_complete 推导、无效命中防护、termination 保证、simulator 分支、scan_trace 记录。 |
| `scripts/check_summary_contract.py`（canonical + plugin） | 新增 scan-trace 穷尽闸（§6）+ 强制点闸（§6.7，`--index` 读 MANDATORY_PER_ROUND），保持 count-agnostic。 |
| `vendored-triton-op/triton-latency-optimizer/references/Index.md` | **不改写**。上游 Index.md 不自带 `MANDATORY_PER_ROUND` 标记，校验侧（`check_summary_contract.py --index`）在标记缺失时按表格「每轮作为最后一个优化点必须执行」行派生强制点集（count-agnostic），vendor 内容保持字节保真。 |
| `docs/scan-loop-redesign.md`（本文件，canonical） | 设计正本，供评审与后续维护。 |

**同步纪律**：canonical 与 plugin 副本当前字节一致（经 `diff` 验证），SKILL.md / AGENTS.md / `scripts/check_summary_contract.py` 每次改动必须**双写**，最终再 `diff` 确认一致。

## 8. 测试计划

1. **契约脚本单测**（headless，不需 NPU）：
   - 合法 trace（上述示例，应用轮带 `mandatory:[31]`，配 `--index`）→ PASS。
   - 手写式旧格式（`points_hit=[2,33]`、无 scan_trace）→ FAIL（穷尽闸拦截）。
   - 区间推导错误（not_hit 不等于推导值）→ FAIL。
   - **强制点闸四态**（`--index` 指向含 `MANDATORY_PER_ROUND: 31` 的 Index.md）：
     - 应用轮带 `mandatory:[31]` → PASS；
     - 应用轮缺 `mandatory`（点 31 折进 not_hit）→ FAIL（拦截本次缺陷）；
     - 同缺 mandatory 但 `optimizer.waiver` → PASS_WITH_WAIVER；
     - `optimizer_rounds>0` 但缺 `--index` → FAIL。
   - `scan_from` 跳变非法 → FAIL。
   - `scan_complete=true` 但无整趟 None 轮 → FAIL。
   - 未来点数（`points_considered=34`）编排 trace → PASS（count-agnostic 验证）。
   - 既有 PASS 用例回归：`perf_baseline`==用户源、`gate_ref`≠`perf_baseline`、bench 溯源一致。
2. **prompt 层静态校验**：diff canonical vs plugin 全一致；grep 确认无残留"33 点"/"31 个优化点"硬编码表述（仅允许在引用 Index.md 表格本身时出现）。
3. **真实 smoke**（有 NPU 时）：对 causal_conv1d_fwd_kernel 重跑 S4，确认新状态机产出 `scan_trace`，CONTRACT PASS，且结果仍 BELOW@1.336x（回归不退化）。

## 9. 验收标准（A 部：扫描循环）

- [ ] 无任何硬编码"33/31"点数的扫描逻辑；N 每轮读 Index.md。
- [ ] `scan_complete` / not-hit 区间全部数值推导，不依赖自报。
- [ ] 每轮强制 `Skill(triton-latency-optimizer)` 调用（写在 AGENTS.md 硬约束）。
- [ ] `check_summary_contract.py` 对"无 scan_trace 的手写式声明"一律 FAIL。
- [ ] canonical 与 plugin 双写后 `diff` 全一致。
- [ ] 契约单测（含 count-agnostic 用例）全过。

---

# B 部 — §10. Patch 导出（S4.5，主动交付）

> 本扩展把「采纳的优化 → 用户可直接引用的统一补丁」做成 **S4 出口的必然交付物**，
> 与 summary / best_code 平级。四决策以用户确认落定：**②主动交付 + 保留责任声明**、全套交付、
> tiling 直改启动默认值、契约强制（缺则 FAIL）。

## 10.1 定位与语义

- **边界语义 = 主动交付**：`optimizer_rounds>0 且 采纳非空 best_code`（REACHED，或 BELOW-建议收）时，
  **必须**产出一份 drop-in 的 `{op}.patch` + `apply.sh` + `README`；缺则契约 FAIL。
- **责任声明**（告知不阻断）：patch 只做**算子级归因**；未做端到端/整训练回归；
  **替换回图与验收仍是用户责任**。
  该声明**必须**写进三处：
  ① `apply.sh` 头注释；② `README.md`；③ **生成的报告**——每个候选的 per-op `report.md`
  新增《Patch 应用与责任声明》小节（§10.6），`batch_report.md` 尾部明示（§10.6）。报告落这不只是"点到"，
  是**像素级可见**：用户读报告即被明确告知边界，不确定谁交付了"能直接改 DB"这件事。
- **形态 = 全套**：`{op}.patch`（unified diff，`-p1`，路径相对用户源根）+ `apply.sh` + `README.md`
  （命中点 / 前→后 latency / 数值 MERE / 责任声明）。
- **tiling 直改启动默认值**：见 §10.2（`launch_default` 模式）。
- **契约强制**：`best_code` 非空 → patch 必须存在且字节派生自 verify 产物，否则 FAIL（§10.5）。

## 10.2 patch 内容 = 用户源码级变更（不是 wrapper）

**关键校正**：patch 的目标是**用户源码文件本身**（= `perf_baseline`），不是 standalone wrapper
（如 causal 轮的 `best_code.py` 是 `import _userpkg` 的包装，不可直接 apply 进用户仓）。
因此每个采纳的优化必须能表达成**用户源码上的最小改动**：

- **patch-mode = `launch_default`**（tiling / tile 级调优，如 causal）：
  「直改启动默认值」——在用户源 `causal_conv1d_fwd_impl` 的 launch 行，把默认 tile 从
  `BD=256/BT=32`（及 nw）覆盖为 `best_config`（`BD=64/BT=64/nw=4`）。→ diff 只有几行 launch 参数，
  apply 后立即生效。这是 causal 场景的既定形态。
- **patch-mode = `kernel_replace`**（重写类优化）：用户源中把对应 kernel/launcher 体替换为
  best_code 的实现。→ diff 较大；若无法 drop-in 则降级 advisory（§10.3 步 2），走 waiver 公示。
- **patch-mode = `null`**（未采纳任何更优变体 / 无 best_code）：不产 patch。

## 10.3 生成流程（机械、字节派生、可审计）

```text
输入：A = official/{op_name}.py       # byte-freezed 用户源副本 == perf_baseline（已过 verify）
      C = best_config                 # 采纳的优化参数（launch_default 场景）
      B = best_code.py                # 已过 verifier 的优化变体（kernel_replace 场景）
步骤：
1. 构造 P = {op}_patched.py           # 用户源码级副本：
      launch_default → 复制 A，将 launch 行 tile 覆盖为 best_config
      kernel_replace  → 复制 A，将对应 kernel 体替换为 B 的 kernel
2. 重验 P：verify.py P 必须 passed==total
      · launch_default：单 shape 快、必过（launch config 由 host 决定）
      · kernel_replace：若 P 无法 drop-in（verify 失败）→ patch-mode 降级 advisory，
        可直接 apply 的承诺取消，走 waiver 公示，绝不静默交付坏 patch
3. 生成 diff：diff -u --label a/<rel> --label b/<rel> A P > patch/{op}.patch
      · rel = A 相对用户源根（如 fsdp/ops/gdn/triton/convolution.py）→ git apply -p1
        在用户仓根即可应用
4. 写 apply.sh（git apply 示例 + patch -p1 兜底 + 责任声明）与 README.md（命中点/前后 latency/MERE）
5. 记录 patch 块到 summary.json；batch_report 增「patch」列
```

- **禁止手写 diff**：patch 字节必须由 `diff -u(A, P)` 产出；P 必须由步 1 确定性构造、步 2 重验。
- `diff_from` 恒 = 用户源冻结副本（perf_baseline）；`diff_target` = patched 副本。

## 10.4 summary.json 的 `patch` 块

```jsonc
"patch": {
  "requires_patch": true,                 // 语义上需要（best_code 非空）
  "mode": "launch_default" | "kernel_replace" | null,
  "path": "patch/causal_conv1d_fwd_kernel.patch",    // 相对 op_{name}/
  "diff_from": "official/causal_conv1d_fwd_kernel.py",
  "diff_target": "patch/causal_conv1d_fwd_kernel_patched.py",
  "apply_target": "<用户源根>/fsdp/ops/gdn/triton/convolution.py",  // 用户源绝对路径 = perf_baseline（示例，实际由本次运行填入）
  "apply_rel": "fsdp/ops/gdn/triton/convolution.py",
  "best_config_applied": {"BD":64,"BT":64,"num_warps":4},
  "resverified": {"passed":1,"total":1},
  "hunks": 1,
  "apply_note": "仅算子级归因，未做端到端/训练回归；替换回图与验收是用户责任"
}
```

## 10.5 契约（check_summary_contract）patch 闸（强制）

- **触发**：`best_code` 非空（有采纳的优化变体）→ `patch` 块必须存在且合法，否则 FAIL。
- 校验：
  - `patch.mode ∈ {launch_default, kernel_replace}`（非 null）。
  - `patch.path` 存在且是真实文件（`op_{name}/patch/{op}.patch`）。
  - `patch.diff_from` == `official/{op_name}.py`（字节派生自 perf_baseline 用户源冻结副本）。
  - `patch.resverified.passed == patch.resverified.total > 0`（patch 目标已重验）。
  - `patch.apply_target` == `perf_baseline`（patch 落点 == 用户源分母唯一）。
  - `launch_default` 模式：`patch.best_config_applied` 与总结已采纳配置（`bench.best_config` /
    summary 里该轮采纳配置）一致。
- `best_code` 为空（无采纳优化）→ 不要求 patch（patch 可缺席或 `mode=null`）。
- `patch.waiver={reason}` 可公示放行（如 kernel_replace 无法 drop-in 降级 advisory）。

## 10.6 报告落点（含责任声明，强制写入生成报告）

- **batch_report 增列**：「patch」列：REACHED / BELOW-建议收 → `patch/{op}.patch` 全路径；
  无采纳优化 → 「—」；tier=weak → 照旧进待人工核定栏。
- **per-op `report.md` 新增《Patch 应用与责任声明》小节**（每个候选，无论 REACHED/BELOW）：
  - patch 产物路径（`patch/{op}.patch`、`apply.sh`）与 apply 命令示例（`cd <源根> && git apply <patch>`）。
  - **责任声明**原文（与 §10.1 三处一致）：
    > 本 patch 仅做**算子级归因**；未做端到端/整训练回归。**替换回图与验收是用户责任**；
    > 应用前请自行做精度/性能/训练回归验证。
  - 附命中点、前→后 latency、数值 MERE（基准值参考）。
- **`batch_report.md` 尾部明示**同一责任声明（全文/摘要 + 指向各 per-op 小节）。

这样责任声明在三个交付通道都可见：`apply.sh` / `README` / **生成报告（report.md + batch_report.md）**。

## 10.7 实现落地（开发阶段，canonical/plugin 硬链接单处编辑即双写）

- `AGENTS.md`：新增《S4.5 patch 导出》小节（§10.2–10.3 流程 + 强制交付）；S5 汇总要求 per-op
  `report.md` 写《Patch 应用与责任声明》、`batch_report.md` 尾部同声明（§10.6）。
- `SKILL.md`：产物清单补 `patch/`（`{op}.patch` / `apply.sh` / `README`）；契约自检表补 patch 闸；
  报告要求补责任声明小节。
- `scripts/check_summary_contract.py`：新增 patch 闸（§10.5）。
- 可选 `scripts/gen_patch.py`：机械化步 1/3/4（构造 P、diff、apply.sh），降低手写风险。

## 10.8 测试计划（开发阶段）

1. **tiling-only（launch_default）**：构造采纳配置 → 重验 P 通过、`diff -u` 只有 launch 行 hunk、
   `git apply` 到用户仓副本成功、契约 PASS。
2. **无采纳优化**：patch 缺席 / `mode=null`，契约 PASS。
3. **负例**：缺 patch、`diff_from`≠冻结副本、`best_config_applied` 与采纳配置不一致、
   resverified 未过 → 契约 FAIL。
4. **报告落点**：per-op `report.md` 含《Patch 应用与责任声明》小节、`batch_report.md` 尾部含同一声明
   （做静态校验，缺则视为未交付）。
5. **A 部回归**：scan_trace / perf_baseline / bench 溯源等既有闸不回归。
6. **static**：canonical vs plugin diff 一致；无硬编码点数量复述。
