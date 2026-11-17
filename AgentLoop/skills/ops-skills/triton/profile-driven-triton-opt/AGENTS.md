# AGENTS.md — Profile 驱动 Triton 算子自动调优编排语义

本工作流由 `profile-driven-triton-opt` skill 承接用户入口，按 **S0→S6** 状态机驱动。
每条规则都是硬约束；遵循设计文稿 §9 不变量：算子级归因、只做调优不重新生成算子、
分母唯一=用户源、替换回图为用户责任。

## 状态模型 S0–S6

```text
S0 参数确认   读取 config.json + 判定 metric_mode；校验输入（zip 路径、源映射、NPU 可用性 preflight）；设定设备隔离
S1 解析 zip   定位 kernel_details.csv + step_trace_time.csv
S2 瓶颈识别   稳态聚合 → op 占比 → 两档 triton 判定 → candidates.json（triton-op-* 内：analyzer）
S3 源装配     用户映射名→源；校验；无源→SKIPPED_NO_SOURCE（assembler）
S4 调优闭环   逐 candidate 串行跑性能调优模块（verifier 收敛门 + latency/simulator optimizer）
S5 汇总       summary → batch_report.md
S6 会话归档   将当前 Claude Code 后台 session jsonl 归档为 session.jsonl + session.md
```

## 各阶段出口 Gate

| 阶段 | 出口产物 | 出口 Gate |
| --- | --- | --- |
| S0 | config 校验通过 | zip 存在、NPU preflight 过、**`init_run.py` 已新建 RUN_DIR(fresh)**、`metric_mode` 已定并写 `RUN_DIR/metric_mode.txt`、`ASCEND_RT_VISIBLE_DEVICES` 已按 `config.npu_device` 设定（或显式沿用环境） |
| S1 | 定位出两个 CSV | kernel_details.csv 必须存在，否则终止并提示检查 profiler 产物 |
| S2 | candidates.json | 无可用 Triton 候选 → 正常结束产「无可优化瓶颈」报告，不硬造 |
| S3 | assembly/{name}/ | 有源 → 合格输入包；无源 → 标 SKIPPED_NO_SOURCE，明确报错并跳过该 candidate |
| S4 | op_{name}/summary.json | verifier 全 shape `passed==total` 才计为成功调优底数；**须过 `scripts/check_summary_contract.py`** |
| S5 | `RUN_DIR/batch_report.md` | 汇总保留全部状态，不因单个失败中断整体；路径必须在本轮 RUN_DIR 内 |
| S6 | `RUN_DIR/session.jsonl` + `RUN_DIR/session.md` | 定位/渲染失败**非阻塞、仅告警**，不影响任务结束 |

## S4 内层（性能调优模块，不重新生成算子）

- **逐 candidate 串行**遍历 `candidates.json` 中 `tier=strong` 项，独立工作目录 `op_{name}/`。
- 复用 vendored 副本（`skills/ops-skills/vendored-triton-op/`）：
  - **收敛门** = `triton-op-verifier`（`verify.py` / `benchmark.py` / `freeze_baseline.py`）；每 variant 必须全 shape 通过。
  - **调优循环（唯一执行通道）** = `triton-latency-optimizer` + `triton-simulator-optimizer`。
    **必须按 latency-optimizer 的纪律执行，禁止以自写 tiling 脚本顶替**。
    该 skill 现为**无状态单步函数**，且**每次调用扫描前必须先做一次 KB 检索（preflight）**：
    优先尝试 KB 指向的高收益点（`exclude_points` 仍一律跳过），并返回「KB 知识检索摘要」盒装块。
    编排器不代替它检索，但必须把摘要记进 `optimizer.kb_preflight` 证物（KB 不可用需注明）。
  - `triton-precision-debug` 用于精度异常排查；`npu-arch` 用芯片约束约束方案。
  - **收敛门语义**：`triton-op-verifier` 现以**生态算子开源精度标准（eco_limits）为第一标准**、
    NPU Benchmark 标准为 fallback；golden 优先落 CPU，执行失败才回退 NPU。编排器仍只认
    `verify_result.json` 的 `passed_cases == total_cases`，不因判定标准切换而放宽。
- **基线 = 用户源本身**，`speedup_vs_user = 新 triton / 用户源`（分母唯一）。
- **判定口径由 `metric_mode` 决定**（见下"S4.0"）：`speedup` 用 `speedup_vs_user`；
  `mfu` 用 `triton-mfu-analyzer` 测出的 `mfu_achievable_pct`，不以 benchmark 加速比作判据。
- 用 profiled shape(+dtype+dispatch) 作为唯一主测试点（来自 S3 `test_shapes.json`；单 shape 下无几何平均、
  无 multi-case kernel 分裂）。
- 未达目标的算子**不自动硬推**过量轮次，可用 `max_opt_iterations` 限制。
- **调优证物**：`optimizer_rounds>0` 时，`summary.json` 必须带 `optimizer` 块：
  `tool = "triton-latency-optimizer"`、`points_hit`（本轮命中点编号数组）、`loaded_docs`（真实加载的
  参考文档路径）、`points_considered`（N）、`scan_complete`、`scan_trace`（逐轮扫描区间，应用轮带
  `record.mandatory` 记录强制点执行）。另建议带 `kb_preflight`（每次调用前强制 KB 检索的摘要：
  `called` / `cards` / `points` / `available`），缺省仅告警、不阻断。
  `check_summary_contract.py` 的调优执行者闸（presence + **scan-trace 穷尽闸** + **强制点闸**）强制校验
  （缺 tool / 非白名单 / 缺证物 / scan_trace 不自洽 / 强制点（`--index` 读 Index.md `MANDATORY_PER_ROUND`
  如点 31）在应用轮未执行 即 FAIL；`optimizer.waiver={reason}` 才可公示放行）。

### S4.0 判定口径（metric_mode 双轨）

`metric_mode ∈ {speedup, mfu}`，是 S4 唯一优化目标 / 采纳判定 / 退出判据的开关；
S0–S3 两种模式完全一致，只有本节的测定通道与判定量不同。

| 模式 | 指标 | 基线 | 每轮测定 | 达标条件 |
| --- | --- | --- | --- | --- |
| `speedup`（默认） | `speedup_vs_user` | 用户源 triton 的 latency | `verify.py` + `benchmark.py`（`implementation.avg_latency_ms`） | `best_speedup >= target_speedup` |
| `mfu` | `mfu_achievable_pct`（dtype 修正实测峰值口径） | 用户源在极限 case 上的 `baseline_mfu` | `triton-mfu-analyzer`（`mfu_decompose.json`） | `best_mfu >= target_mfu` |

- **mfu 分支入口（每个 candidate 一次）**：`triton-mfu-analyzer` 目录适配 → (case, 倍率) 极限搜索
  （`mfu_grid.json` / `extreme_case.jsonl`）→ `flops.py` 编写 + `check_flops.py` 自检
  （通过后**整轮冻结禁改**）→ 按 dtype 标定峰值（`peak.json` 冻结）→ 测 `baseline_mfu`。
  产物落在 `op_{name}/mfu/`；同一 candidate 的后续每轮复用同一套极限 case / flops / peak，
  保证 MFU 可比。**mfu 模式禁止用 benchmark 加速比作判据**（加速比只在报告里引用）。
- **结构性失败**：极限 case 上输出 shape 不符 / NaN 结构性错位（"少算工作"）→ 该点 MFU 作废，
  按 `mfu_grid.json` 候选排序换下一个通过 `check_scaled_point` 的点；全部非法则回落到 s=1
  原始宽度最优点，并在报告中标注重选原因。
- **统一判定量**：下文 `optimized_metric` / `best_metric` / `baseline_metric` / `target_metric`
  按模式取义；mfu 模式下 `best_speedup` 类字段一律对应 `best_mfu`，其余三态判定、scan 状态机、
  强制点、patch/restructure 契约与 speedup 模式完全相同。
- **产物字段**：`summary.json` 必带 `metric_mode`；mfu 模式另带 `target_mfu` / `best_mfu` /
  `baseline_mfu`。`perf_baseline` 仍必须指向用户源（分母唯一），与 `gate_ref` 分离的硬约束不变。

### S4 扫描状态机（count-agnostic，编排器持有循环）

> 点数 **不硬编码**：`N` = 本 candidate 调优期间从
> `triton-latency-optimizer/references/Index.md` 表格**当场读取的最大点号**（每次扫描前重读，
> 因此优化点增删零适配）。所有区间推导只依赖 `cursor` + `hit_optimization_point`，无常量。

**状态变量（每 candidate 一份）：**

```text
max_opt_iterations = <config.max_opt_iterations>
target_speedup     = <config.target_speedup，默认 2.0>
N                  = <随时从 Index.md 表格读取的最大点号>
cursor             = 1             # 下次 scan_from；由命中点/无命中推导推进
code_version       = 0             # 每采纳一个优化 +1；代码变更即清 empty excluded、cursor 回 1
excluded           = []            # 【版本作用域】本 code_version 下已判无提升的点
fail_count         = {}            # 【跨版本】各点累计"命中但无提升"次数；>=3 永久禁用
banned_points      = []            # 永久禁用；与 excluded 合并为 exclude_points
scan_complete      = false         # 从 cursor=1 走完一整趟仍无命中 → true（编排器推导）
simulator_attempted= false         # triton-simulator-optimizer 是否已调用（终局前置门）
restructure_applied= []            # 【restructure,§R】已应用变换签名（dedup 依据）
restructure_pending_hint = false   # 触发成立待守卫时置 true；通过即为 R 阶段落地
```

**每轮循环（编排器驱动）：**

```text
round_dir = op_{name}/opt_iter_{n}/   # 每轮独立产物目录
0. 【R 阶段·仅当 restructure.enabled 且 cursor==1 且 code_version 推进后的新基线入口】
   restructure 检查（先于步骤 1 的 vendor 扫描；详见 SKILL.md「R 阶段」）：
   a. restructure_applied 已含全部登记变换 / max_restructures 用尽 → R 阶段退出（不补新 code_version）。
   b. 触发判定（任一即置 restructure_pending_hint=true）：
        A. IR 强制点 31 收尾真实分析结果含 "gather 复用/需 kernel-body restructure" 且 ir_has_more_suggestions=true；
        B. 调用侧结构检测：last_pass.mlir 解析到同一源区间多次重叠 GM gather 的 W-tap 模式。
   c. restructure_pending_hint 且 best_speedup < target_speedup：
      构造重构 kernel（restructure_engine/ 计划）→ scripts/restructure_guard.py 4 闸
      （g1 I/O 冻结 + g2 全变体矩阵位级 diff + g3 IR 同构；g4=官方 verify/benchmark 溯源）。
      三闸任一 False → 该变换作废（不落 best_code，cursor 保持，走回步骤 1）。
      三闸全过 → 该重构 kernel 作为本 code_version 基线，cursor 保持 1，restructure_applied += 签名，
                进步骤 1 vendor 扫描（重构后必有 tune 轮）。
1. 强制 Skill(triton-latency-optimizer) 调用          # ← 无本次调用=未执行本轮，禁止 Read-then-inline
   入参：baseline_code/上一轮优化代码、opt_iteration、
        scan_from=cursor、exclude_points=excluded ∪ banned_points
   返回：hit_optimization_point = P | None（skill 不自报"扫完"）
2. 区间推导（编排器，数值可验）：
   P != null → not_hit = [cursor, P-1]；改码前必须真实加载 P 参考文档，且前置
              references/checklist.md 逐项满足 → 4.2 精度 → 4.3 性能 → 4.4 判定
   P is None → not_hit = [cursor, N]
               cursor==1 → scan_complete=true，进终局
               cursor>1  → cursor=1，continue（从头整趟复查，代码可能已变）
3. 每轮收尾执行强制点（Index.md `MANDATORY_PER_ROUND`，当前=点 31 IR 分析，命中条件=所有算子类型）：
   真实调用 latency-optimizer skill 执行（提 last_pass.mlir 分析），产出 ir_has_more_suggestions；
   有建议 → 作为下一轮命中点继续；无 → 仅记为"已执行"。**禁止把必须执行的强制点并进 not_hit 静默跳过。**
4. 记录本轮 scan_trace：{round, scan_from, hit, not_hit, mandatory:[强制点], [scan_complete]}
5. 判定（见下）后 opt_iteration++；不可原地打转（见终止保证）
```

**每轮判定：**

- **优化成功**（verify 全过 且 `speedup_vs_user > best_speedup`）→ 更新 best_code/best_speedup；
  `code_version += 1`、`excluded = []`、`cursor = 1`（旧 not-hit 结论失效，须允许重扫）、
  `no_improve_streak = 0`、`target_reached = (best_speedup >= target_speedup)`。
- **无提升**（P 命中但 verify 未过 = 引入错误 A 类 / verify 过但无收益 = 无效命中）：
  - **无效命中**（verify 过、无提升或相等）：回退 best_code；`excluded += [P]`（仅本 code_version）、
    **`cursor = P + 1`**（游标强制前进，杜绝原地重复）、`fail_count[P] += 1`、跨 3 个 code_version →
    `banned_points += [P]`（log 标注"无效重复命中"）。
  - **A 类错误**（verify 未过）：**`cursor` 保持 = P**（不过早排除可能有效但被改坏的点），回退，4.5 分析。

> **终止保证**：每轮要么 `cursor` 前进（有界 ≤N），要么 `code_version` 递增（需真实提升，metric
> 单调有界），循环必然收敛。退出只在终局前置门满足后进行，`improvement_made` 不是退出触发器。

**simulator 采集驱动（优化耗尽 & 未达 target 时的强制收手闸）：**

- latency-optimizer 整趟（cursor=1→N）已无命中且 `best_speedup < target_speedup`，
  **禁止直接判 BELOW**：须先 `Skill(triton-simulator-optimizer)`（只采集+诊断：msprof → pipe 占比 →
  修复方向 = latency-optimizer 点编号；`MMAD > 50%` 判"真·硬件极限"）；`simulator_attempted = true`。
- 有修复方向 → 回步骤 1，把诊断指向的点作为**强制命中点**传入（覆盖其静态命中判断），再验证一轮。
- 诊断报无对应优化点（`MMAD > 50%` 且放大 tile / bf16 化均不可行）→ 才可终局判 BELOW。

**终局判定（三态）：** `best_speedup >= target_speedup 且 optimizer_rounds>0 且 非空 best_code → REACHED`；
否则 `BELOW`；verify 全流产 / 环境错误 → `FAILED`。详见下行"三态判定（§7）"。

### 产物契约（防 torch 参考顶包分母）

summary.json 必须显式区分两个角色字段（S4 出口须过 `scripts/check_summary_contract.py`）：

| 字段 | 角色 | 允许 |
| --- | --- | --- |
| `gate_ref` | 收敛门真值（torch 参考） | **仅数值校验**；禁止进性能分母 |
| `perf_baseline` | 性能分母 | **必须指向用户源 triton 文件**；不得等于 `gate_ref` |

> **硬约束**：`perf_baseline` ≠ `gate_ref`；任何把 torch 参考当 `speedup_vs_user` 分母的写法，契约校验即 FAIL。

**测量溯源闸**：`bench.tool` 必须是 `scripts/benchmark.py`（torch_npu.profiler），且 `bench.baseline_result_json` /
`bench.optimized_result_json` 存在并含 `implementation.avg_latency_ms`，`bench.baseline_avg_ms`/`bench.optimized_avg_ms`
与官方产物一致。手写计时器（`torch.npu Event`/`time.perf_counter`）FAIL；`bench.waiver={reason}` → PASS_WITH_WAIVER。
最终性能数字一律回灌官方 `verify.py` + `benchmark.py`（Mode-A 适配器：`{op}_torch.py` 的 Model+get_inputs、
`{op}_{impl}.py` 的 ModelNew，verify 通过后 benchmark 取 `implementation.avg_latency_ms`）。

### 三态判定（§7）

| 态 | 条件 | 处理 |
| --- | --- | --- |
| `REACHED` | `speedup_vs_user ≥ target_speedup` **且 `optimizer_rounds>0` 且非空 `best_code`** | 正常收 best_code |
| `BELOW` | 调优成功但未达 target = **没达标、算没成**；或 REACHED 未满足前置自动降级 | 保留 best_code；报告只给「是否达标」，不吹实际收益 |
| `FAILED` | verify 未全过 或 环境错误 | 标注原因；不硬造 |

- **REACHED 硬前置**：未跑调优轮（`optimizer_rounds<=0`）或无 `best_code`，一律降 `BELOW`，不得判 `REACHED`。
- batch_report 附**次要列「建议收不收 best_code」**：仅当已有可执行的更优 variant 时给「建议收」，兜底可执行建议、不改判不夸大。

### S4.R Restructure 阶段（调用侧 kernel-body 重构，可选）

> 对应设计文档 `docs/restructure-stage.md`。当 op 瓶颈是**读放大/gather 复用**这类结构性问题，
> 而 vendor `triton-latency-optimizer` 的 33 个点（Index.md）**全部*不含 kernel-body 重构**（点 31 只
> 出诊断、不执行改写）时，唯一可继续的杠杆在**方法体内部数据流改写**。这是 vendor 自判
> *out-of-scope* 的动作，**属调用侧自持**、**vendor 字节不动**。

- **合规前提（硬）**：restructure 必须满足 **I/O 冻结（`restructure_guard` g1/g2）+ 语义同构（g3）**，
  落在「只调优、不重新生成算子」界内；默认 `config.restructure.enabled=false`（operator/用户显式开启）。
- **每种不同变换各领一个 code_version**：命中并过 4 闸 → 作为新 code_version 基线、`cursor` 回 1、
  `restructure_applied += 签名`（dedup 拦截重复）。未达标不重复同一变换；有**下一种** → 下一个 code_version。
- **不直接判 REACHED**：REACHED 仍须 `optimizer_rounds>0` 且非空 best_code——重构后必有 tune 轮。
- **契约证物**：本轮 best_code 由 restructure 而来时，`summary.json` 必须带 **正交 `restructure` 块**
  （trigger/applied_transform/engine/gates/io_freeze_basis）；`optimizer` 块只装 vendor tune trace，两者互不干扰。
  契约 `_check_restructure` + `_check_report_records` 强制（restructure 缺省 → 纯 tune 契约逐字节不变）。
- **终止**：restructure 不是"一次搞定"原子——每个不同变换各领一个 code_version 参与主循环，达标/耗尽
  （`restructure_applied` 全含登记变换或 `max_restructures` 用尽）/封顶（`max_opt_iterations`）三选一收敛，
  同一 sign 绝不重跑；tune 耗尽 + simulator `MMAD>50%` → 终局 BELOW。
- **report 落点（强制，§8）**：用户 `report.md` 必含「调优过程」+「重构过程」两节（字段与 summary 对齐）；
  纯 tune 运行至少含「调优过程」。缺 record → 契约 FAIL（视为未交付）。

### S4.5 Patch 导出（主动交付，设计文档 §10）

`optimizer_rounds>0 且 采纳非空 best_code`（REACHED，或 BELOW-建议收）时，**必须**产出一份
可直接引用的补丁 + apply 脚本 + 说明；缺则契约 FAIL。目标是**用户源码本身**，不是 standalone wrapper。

**patch-mode**：

- `launch_default`（tiling/tile 级调优，如 causal）：在用户源 launcher 的 launch 行，把默认 tile 覆盖为
  `best_config`（例 `BD=256/BT=32`→`BD=64/BT=64/nw=4`）。→ 几行 diff，apply 后立即生效。
- `kernel_replace`（重写类）：把用户源对应 kernel 体替换为 best_code 实现。→ 大 diff。
- `null`（未采纳更优变体 / 无 best_code）：不产 patch。

**流程（机械、字节派生、可审计，禁止手写 diff）**：

```text
A = official/{op_name}.py          # byte-freezed 用户源副本 == perf_baseline（已过 verify）
1. 构造 P = {op}_patched.py        # 用户源码级副本：launch_default→覆盖 launch tile；
                                   #           kernel_replace→替换 kernel 体
2. 重验 P：verify.py 必须 passed==total
      · launch_default：必过（launch config 由 host 决定）
      · kernel_replace：若无法 drop-in → patch-mode 降级 advisory，走 waiver 公示，绝不静默交付坏 patch
3. diff -u --label a/<rel> --label b/<rel> A P > patch/{op}.patch
      · rel = 相对用户源根 →`git apply -p1` 于仓根即可应用
4. 写 patch/apply.sh（git apply + patch -p1 兜底 + 责任声明头注释）与 patch/README.md
5. 记录 summary.json.patch 块；batch_report 增「patch」列
```

**summary.json.patch 块**：`mode`/`path`/`diff_from`(=official 冻结副本=`perf_baseline`)/`diff_target`/
`apply_target`(=perf_baseline 用户源绝对路径)/`apply_rel`/`best_config_applied`(launch_default)/`resverified`/
`apply_note`。契约 patch 闸强制校验（§10.5）。

**责任声明（强制写入生成报告）**：per-op `report.md` 新增《Patch 应用与责任声明》小节、
`batch_report.md` 尾部同声明（见下"报告落点"）。原文：
> 本 patch 仅做**算子级归因**；未做端到端/整训练回归。**替换回图与验收是用户责任**；应用前请自行做精度/性能/训练回归验证。

## S5 汇总 `batch_report.md`

| op | tier | target | actual(speedup_vs_user) | 是否达标 | best_code 建议 | patch | 失败/原因 | 待人工核定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| … | … | 2.0x | … | REACHED/BELOW/FAILED | 收/不收 | `patch/{op}.patch` 或「—」 | … | weak 候选单独一栏 |

- `tier=weak` 候选统一进「待人工核定」栏，给建议、不改判。
- **「patch」列**：采纳更优 best_code（REACHED / BELOW-建议收）→ `patch/{op}.patch` 全路径；
  无采纳优化 → 「—」。
- **报告落点（责任声明，强制）**：
  - 每候选 `op_{name}/report.md` 新增《Patch 应用与责任声明》小节：patch 路径 + `apply.sh` 用法
    （`cd <源根> && git apply <patch>`）+ §10 责任声明原文 + 命中点/前→后 latency/数值 MERE。
  - `batch_report.md` 结尾明示同一责任声明（全文/摘要 + 指向各 per-op 小节）。
  - 报告继续明示：仅算子级归因，不承诺整 step/整训练提速；替换回图为用户责任。

## 错误处理

| 阶段 | 错误 | 处理 |
| --- | --- | --- |
| S1 | zip 内无 kernel_details.csv | 报错终止，提示检查 profiler 产物 |
| S2 | 无 Triton 候选 / 无 op 达阈值 | 正常结束，产「无可优化 Triton 瓶颈」报告 |
| S3 | 源缺失 / 名映射缺失 | 跳过并标 SKIPPED_NO_SOURCE；禁止臆造 |
| S4 内层 | verify 未全过 / baseline 被篡改（exit 4） | A 类迭代修复，B/C 类终止该 candidate |
| S4 外层 | benchmark 超时 | 降 `--repeats`（50→20→10→5）重试，禁止编造数据 |
| S5 | 部分 candidate 失败 | 汇总保留全部状态，不中断整体 |

## 反作弊

- 加速比分母**只能**是用户提供的源本身，无其他输入来源。
- **真值/分母角色分离**：`gate_ref`(torch 参考) 只做收敛门数值校验，**禁止**进性能分母；
  `perf_baseline` 必须克服用户源 triton；二者不得混用（`check_summary_contract.py` 强制）。
- **REACHED 前置**：`optimizer_rounds>0` 且非空 `best_code`，否则强制降 `BELOW`。
- 无源 → SKIPPED_NO_SOURCE，绝不替用户补源/硬造 golden。
- 性能测量禁止 `time.perf_counter`，必须用 benchmark 脚本（warmup/repeats）。
- **调优执行者唯一化**：`optimizer_rounds>0` 必须经 `triton-latency-optimizer`（`Index.md` 有序扫描 +
  命中即加载参考文档 + checklist），禁止自写 tiling 扫描顶替；`check_summary_contract.py` 调优执行者闸强制。
- **运行隔离（fresh-run）**：每次运行必须 `scripts/init_run.py` 新建独立 `RUN_DIR`（除非显式
  `--reuse-last` 续跑）；所有阶段产物写进 `RUN_DIR`，禁止平铺到 `<out_root>` 根或被其他 run 复用。
- **fresh-source 闸**：本轮 `summary.json`/`candidates.json`/`batch_report.md` 必须指向**本次** `RUN_DIR`；
  **禁止拿历史 run 或平铺旧产物的 summary 当本轮交付**（曾因静默继承上一轮 BELOW 结果），违反即判定失效、需重跑。
- **patch 字节派生闸**：patch 必须由 `diff -u(冻结用户源副本, 重验过的 patched 副本)` 字节派生，
  **禁止手写 diff**；`apply_target` 必须 == `perf_baseline`（落点=用户源分母唯一）；patched 副本须经
  `verify.py passed==total` 重验后才交付（`check_summary_contract.py` patch 闸强制）。

## S6 会话归档（session.jsonl + session.md）

必须在 S5 之后执行，将当前 Claude Code 会话归档到工作目录便于复盘。**复用 Claude Code 后台
已存的 session jsonl，不自写脚本造会话**。渲染器为 vendored `skills/ops-skills/vendored-triton-op/render_session.py`。

> ⚠️ 并行子进程共享同一个 `/root/.claude/projects/<hash>/` 目录，**必须用工作目录绝对路径精确
> 过滤定位自己的 jsonl，禁止用时间排序（`ls -t | head -1` 会错拿其它并发进程的 jsonl）**。

```bash
# RUN_DIR = 本次由 init_run.py 新建的运行目录
MY_JSONL=$(grep -l "{RUN_DIR绝对路径}" /root/.claude/projects/*/*.jsonl 2>/dev/null | head -1)
if [ -n "$MY_JSONL" ]; then
  cp "$MY_JSONL" "$RUN_DIR/session.jsonl"
  python3 skills/ops-skills/vendored-triton-op/render_session.py \
    "$RUN_DIR/session.jsonl" "$RUN_DIR/session.md" 2>&1 || \
    echo "WARN: session render failed (non-fatal)"
else
  echo "WARN: session jsonl not located (non-fatal)"
fi
```

定位失败 / 渲染失败**均不阻塞任务，仅告警**。
