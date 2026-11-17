# S4 Restructure 阶段设计（kernel-body 重构并入 S4，不改 vendor latency-optimizer）

> 配套文档：`docs/scan-loop-redesign.md`（S4 扫描循环 + Patch 导出，count-agnostic）。
> 本文档在其之上**新增一个调用侧「restructure 阶段」**，把「kernel-body 重构」(plan B 的 load-block 改写)
> 从「人工越界的一次性尝试」升级为「S4 内首等的、有循环/终止/契约支持的调用侧子阶段」。
> 约束：**triton-latency-optimizer 为外部 vendor，禁止修改**；现有契约闸（溯源/执行者/scan-trace/强制点/patch）不削弱。

---

## 0. 摘要（TL;DR）

`triton-latency-optimizer` 的 33 个调优点（Index.md）**全部不含 kernel-body 重构**（tiling/分核/访存/
scalar-vector/dtype/Autotune/Block-Size，唯一"IR 分析"点 31 也只出诊断、不执行改写）。而 S4 已实证
（`causal_conv1d_fwd` 一轮）：当 op 的瓶颈是**读放大/gather 复用**这类结构性问题时，唯一可继续的杠杆在
**方法体内部数据流改写**（plan B：`W`-tap GM gather → 单次 `2·BT×BD` UB 窗口 + `extract_slice` 移位累加），
实测 1.2697x → **1.3980x**。这是 vendor 明确标注 *"out of tune-only scope"* 的动作，因此**不在 vendor
修改范围**，而是调用侧（profile-driven-triton-opt）自持。

本文档定义：触发 → 4 闸守卫 → 落地为**独立 code_version** 进 S4 主循环 → 重投喂 vendor 扫描 1-33 →
契约新增**正交 `restructure` 块**（不动 `optimizer` 块的现有闸）。全程零触碰 vendor。

---

## 1. 背景与缺口

### 1.1 现状（本会话实测）

- plan B load-block 重构:用户源 1.6824ms → 1.2034ms = **1.3980x**(未达 2.0x, memory-bound 上限)。
- 4 层程序化闸(临时自造)已证 I/O 逐字节守恒:gate1 签名/实参冻结、gate2 变体矩阵 12/12 位级一致、
  gate3 IR 同语义(swish/store/mask 同位)、gate4 patch+verify 1/1 + 官方 benchmark。
- 但**这是人工一次性干预**:无循环、无终止、无契约证物字段、不在 `optimizer.scan_trace` 内。

### 1.2 三个缺口

| # | 缺口 | 后果 |
|---|---|---|
| G1 | 无触发判定:没有任何 code 决定「该用 restructure」 | plan B 只能靠人肉判断,不可复现 |
| G2 | 无 4 闸守卫的正式载体 | 每次重构都要重写一遍 bref/gate2,易漏 |
| G3 | 无契约证物:summary 无 restructure 块 | 重构后的 best_code 无法溯源「改了什么+如何保证 I/O」 |

### 1.3 为什么"唯一执行通道"不拦 restructure

AGENTS.md:36「调优循环(唯一执行通道)= latency-optimizer + simulator;禁止自写 tiling 扫描顶替」。
该禁令约束的是**调优点 1-33**。restructure 是 vendor 自判 *out-of-scope* 的一类动作,
调用侧执行它不是"顶替调优",而是填补 vendor 明确不覆盖的空档;vendor 在被重构后的代码上**仍 1-33 照跑**。
合规前提(硬):restructure 必须满足 **I/O 冻结(gate1/2)+ 语义同构(gate3)**,落在"只调优、不重新生成算子"界内。

---

## 2. 目标与非目标

### 目标

1. 让「kernel-body 重构」成为 **S4 内可判定的首等阶段**(触发 + 守卫 + 循环 + 终止)。
2. 每种**不同**的重构变换各领一个 **code_version** 进主循环,`cursor` 回 1、`excluded` 清空重扫。
3. 契约新增**可选 `restructure` 块**;**纯 tune(无重构)的既有 summary 契约校验结果逐字节不变**。
4. vendor `triton-latency-optimizer` **字节不动**。

### 非目标

- 不新增"重构点"进 vendor 的 Index.md(那要改 vendor)。
- 不允许重构改变方法 I/O(签名/dtype/布局/启动/语义)——越过即拒,不落为 best_code。
- 不允许"重构直接判 REACHED":REACHED 仍须 `optimizer_rounds>0`(重构后必有 tune 轮)且非空 best_code。
- 不做端到端/整训练回归(沿用「算子级归因,替换回图=用户责任」)。

---

## 3. 位置:插入 S4 主循环何处

S4 主循环 `AGENTS.md:72-104` 是**编排器持有**的;插入点在**每个新 code_version 的入口**
(`cursor==1` 且已采纳上一基线)做 restructure 检查,**先于 vendor 扫描**:

```text
code_version 推进后的下一轮入口(cursor==1)
   │
   ▼
[R] restructure 检查(仅当 restructure.enabled 且 有新触发 & 新变换)
   │  命中 → 4闸守卫 → 产出重构 kernel 作为本 code_version 基线 → cursor 保持 1 → 进 vendor 扫描
   │  未命中 → 直接进 vendor 扫描
   ▼
   现有 S4 调优轮(vendor 扫描 1-33).....
```

复用既有语义:`AGENTS.md:64`「每采纳一个优化 +1;代码变更即清 excluded、cursor 回 1」正是 restructure
落地所需——**重构 = 一次代码变更**。未达标不重复同一变换;若有**下一种**重构手段则作为**下一个
code_version** 再来一轮。

**终止保证(不无限续)**:沿用 `AGENTS.md:104-105`——每轮要么 `cursor` 前进、要么 `code_version` 递增
(需真实提升,单调有界)。restructure 补给的是"新 code_version";同一变换被 **dedup** 拦截(见 §5.2),
天然不会重复榨;总轮数受 `config.max_opt_iterations` 封顶;tune 侧无命中且 `simulator MMAD>50%` 判
硬件极限时终局 BELOW(`AGENTS.md:107-113`)。

---

## 4. 触发(调用侧可判定)

两种信号,任一即触发;均在调用侧(CANONICAL)、不依赖 vendor 新增能力。

| 触发 | 信号来源 | 判定 |
|---|---|---|
| A. IR 信号 | S4 每轮收尾强制点 31(`check_summary_contract._read_mandatory_points` 已接管)真实调用 vendor 分析 `last_pass.mlir`;`ir_has_more_suggestions=true` **且** IR 结论含"gather 复用/需 kernel-body restructure" | 复用它原产出的 `ir_notes`,升级为 `restructure_hint=true` |
| B. 结构检测(自持) | 调用侧解析 `last_pass.mlir`(`ir_output_run<N>/...mlir`):检测同一源区间被**多次重叠 GM gather** 的 W-tap 模式 | 纯调用侧规则,确定性 |

`restructure_hint && best_speedup < target_speedup && restructure.enabled` → 进入 restructure 阶段。

---

## 5. 4 闸守卫(调用侧模块)

把 `plan_b/` 下的一次性脚本拔高为调用侧可复用模块:

| 闸 | 现 plan_b 对应 | 提升为 |
|---|---|---|
| g1 API/实参冻结(gate1) | `bref.py::assert_signature_identical` | `restructure_guard.py::g1_io_frozen` |
| g2 全变体矩阵位级 diff(gate2) | `gate2_matrix.py`(12 例) | `restructure_guard.py::g2_matrix_diff` |
| g3 IR 语义同构(gate3) | IR 抽取对比(load 数/op 链/mask/并行模型) | `restructure_guard.py::g3_ir_homology` |
| g4 官方 verify+benchmark(gate4) | verify 1/1 + benchmark.py 溯源 | 复用既有 verify/benchmark 适配器,不回写 |

`restructure_guard.py` 返回 `{g1,g2,g3}: bool`;任一 False → 该变换作废(如同 A 类错误,`cursor` 保持、
不落 best_code)。变换本身由 `restructure_engine/` 提供(load-block transform 参数化:窗口宽度/移位、
mask 策略、BT/BD)。

### 5.1 边界(硬)

读取用户源 → 变换为**方法体内部数据流重写**;对外 `fwd_impl` 签名/宿主启动/dtype/布局/语义必须
`g1/g2` 逐字节守恒。**变更 I/O 或换原语/布局/dtype → 直接拒绝**,不构造成 best_code。

### 5.2 dedup

变换以 `applied_transform` 签名(如 `load_block{win=2*BT, W=4}`)为键去重;同一签名只接受一次真实提升,
杜绝重复榨。`fail_count`/`banned_points` 语义与 tune 点共用。

---

## 6. 代码修改清单(逐文件)

> canonical 根=`skills/ops-skills/triton/profile-driven-triton-opt`;
> plugin 目录=`plugins/ops/skills/profile-driven-triton-opt` 为**软链**,单处编辑即双写(这与
> scan-loop-redesign.md §10.7 的硬链接约定同构,本阶段产物同样落 canonical 一处)。

### 6.1 `config.json`(新增 schema)

```jsonc
{
  "restructure": {
    "enabled": false,               // 默认关;operator/用户按需开
    "scope": "method-body-dataflow,io-frozen",
    "triggers": ["ir_point31_gather_reuse", "structural_gather_detector"],
    "max_restructures": 2           // 每个 candidate 最多重构变换次数(与 max_opt_iterations 协同)
  }
}
```

### 6.2 `AGENTS.md` S4 主循环(加"R"阶段小节)

- 在 `### S4 扫描状态机`(第 51 行起)的循环骨架里、**step 1 之前**插入"restructure 检查"子步:
  命中且通过 4 闸 → 作为本 code_version 基线进入步骤 1(vendor 扫描)。
- 状态变量段新增 `restructure_applied: []`(已用变换签名,dedup 依据)、`restructure_pending_hint`。
- 自检表(S4 出口)新增一条:`restructure.enabled` 且本轮产出 best_code 由 restructure 而来时,
  `summary.restructure` 块必须存在且 4 闸通过。

### 6.3 `scripts/check_summary_contract.py`(改动核心,向后兼容)

- 新增 `_check_restructure(s, errs, warnings)`;**`restructure` 块缺省→直接 return(纯 tune 路径
  契约结果逐字节不变)**。块存在时校验:
  - `trigger` 非空;`gates.{g1_io_frozen,g2_matrix,g3_ir_homology}` 全 `true`(g4=官方 verify/benchmark,
    复用既有溯源闸,不另设);`engine` 为真实文件;`applied_transform` 非空。
  - **反滥用硬断言**:`restructure` 存在但 `optimizer_rounds<=0` → FAIL(重构后必须跑过 tune 轮,
    阻止"重构+直接判 REACHED")。这与现有 `_check_optimizer_evidence` 的 REACHED 前置天然衔接。
- 新增 report 落点闸(§8.3):存在 `restructure` 块 → 强制 report.md 含「重构过程」片段;
  `optimizer_rounds>0` → 强制 report.md 含「调优过程」记录片段。复用 `_check_report_statement`(L442)
  "读 report.md + 查片段"模式,新增两套 fragment 常量(`RESTRUCTURE_REPORT_FRAG` / `OPTIMIZATION_REPORT_FRAG`)。
- 在 `main()` 第 517 行 `_check_optimizer_evidence` 之后追加 `_check_restructure(s, errs, warnings)`
  与 `_check_report_records(s, errs, warnings)`。
- 契约校验入口不变:`check_summary_contract.py <summary> --triton-source <源> --index <Index.md>`。

### 6.4 新增 `scripts/restructure_guard.py`(调用侧守卫)

把 §5 的四闸提为独立模块,入参 = 用户源冻结副本 + 重构 kernel + 变体矩阵用例 + IR 产物路径,
出参 = `{g1:bool, g2:bool, g3:bool, matrix_cases:int, ir_gm_loads_before:int, after:int}`。

### 6.5 新增 `restructure_engine/`(变换器)

`load_block.py`(参数化 load-block 变换:窗口宽 `2*BT`/移位数 `W`/mask 策略)、`transform_registry.py`
(签名→变换注册表,供 dedup、供 `applied_transform` 落 summary)。

### 6.6 `SKILL.md`(轻改)

- 「依赖 Skill」或「编排约定」下补一句:restructure 阶段(G6.2)为调用侧能力,`optimizer.restrure==null`
  的纯 tune 运行不受影响;重构须经 `restructure_guard` 4 闸 + 契约 `restructure` 块。
- 产物契约表(`SKILL.md:79`)表格新增一行:存在重构时 `restructure` 块(可选)。缺省不影响既有契约。

---

## 7. summary.json `restructure` 块(Schema,可选/正交)

```jsonc
{
  "restructure": {
    "trigger": "ir_point31_gather_reuse",
    "applied_transform": "load_block{win=2*BT=128, W=4}",
    "engine": "restructure_engine/load_block.py",
    "gates": {
      "g1_io_frozen": true,
      "g2_matrix": true, "g2_matrix_cases": 12,
      "g3_ir_homology": true, "g3_gm_loads": { "before": 5, "after": 2 }
    },
    "io_freeze_basis": "gate1(签名/实参冻结)+gate2(12例位级 torch.equal)+gate3(IR:swish/store/mask同位)"
  }
  // optimizer 块仍只装 vendor tune trace —— 现有 4 条闸全部原样通过,不受 restructure 影响
}
```

`optimizer` 块字段(`tool/points_hit/loaded_docs/points_considered/scan_complete/scan_trace`)不动;
restructure 是 **additive** 的调用侧证物,与 vendor 证物正交。REACHED 判定仍以
`best_speedup>=target && optimizer_rounds>0 && best_code!=null` 为准(`check_summary_contract.main` §4)。

---

## 8. 用户 report.md 记录要求(强制)

> 原则:summary.json 是 **machine 契约**(查真伪),report.md 是 **用户最终阅读**的落点——
> 重构记录与优化记录**两者都必须写进 report.md**。契约只校验 summary 无法让用户看到过程;
> 缺 report 记录 = 视为未交付(与 §10.6 责任声明同级别强制)。

### 8.1 必含两节(与既有 report.md 结构衔接,见本 op 现 `report.md`「三、调优过程」)

在既有报告结构上**新增「重构过程」一节**,与「调优过程」并列;**两节都强制**:

```text
## 三、调优过程(既有:vendor tune scan 记录)     ← optimization record,已有
## 三' 重构过程(新增:restructure record)         ← 本节为本设计新增
```

### 8.2 「重构过程」节必含字段(与 summary.restructure 逐项对齐,供用户肉眼核验)

```markdown
## 重构过程(restructure record)
- 触发: ir_point31_gather_reuse | structural_gather_detector
- 变换: load_block{win=2*BT=128, W=4}   (engine=restructure_engine/load_block.py)
- 4 闸:
  - g1 I/O 冻结: 签名/实参逐字节一致 ✓
  - g2 变体矩阵: 12/12 位级 torch.equal ✓
  - g3 IR 同构: GM load 5→2, swish/store/mask 同位 ✓
- I/O 冻结依据: gate1+gate2+gate3(g3 GM loads before=5 / after=2)
- 性能: baseline(用户源)=1.6824ms → 重构=1.2034ms, speedup_vs_user=1.3980x
- 与调优关系: 重构产出作为 code_version=N 基线,随后 vendor 扫描 1-33(见「调优过程」)
```

### 8.3 契约强制(不改则漏)

`check_summary_contract` 新增 report 落点闸:

- 存在 `restructure` 块 → `report.md` 必须含「重构过程」及必含字段(`restructure_report_frag` 全集,
  如 `"重构过程"`、`"g1"`、`"g2"`、`"g3"`、`"speedup_vs_user"`)。缺 → FAIL。复用
  `_check_report_statement`(现 L442)的"read report.md + 查片段"模式,新增 fragment 集合。
- `optimizer_rounds>0` → report.md 必须含「调优过程」记录(`optimization_report_frag`,如
  `"调优过程"`、`"triton-latency-optimizer"`、`"scan_complete"`)。缺 → FAIL。

---

## 9. 终止 / 非一次性语义

| 情形 | 行为 |
|---|---|
| 达标 | `best_speedup>=2.0` → REACHED,停 |
| 重构变换耗尽 | `restructure_applied` 已含全部登记变换,或 `max_restructures` 用尽 → 不再补新 code_version |
| 同变换重复 | dedup 拦截,不重复 |
| tune 耗尽+硬件极限 | vendor 1-33 无命中且 simulator `MMAD>50%` → 终局 BELOW |
| 次数封顶 | `max_opt_iterations` 硬上限(不硬推过量轮) |

即:**restructure 不是"一次搞定就完"的原子**,而是"每个不同变换各领一个 code_version 参与主循环、达标/
耗尽/封顶三选一收敛"的多轮机制;同一招绝不重跑。

---

## 10. 测试计划(开发阶段)

| 用例 | 验证点 |
|---|---|
| U1 既有纯 tune summary(无 restructure 块) | `check_summary_contract` 仍 CONTRACT PASS,输出与改动前一致(回归) |
| U2 带 restructure 块且 `optimizer_rounds>0` | CONTRACT PASS |
| U3 带 restructure 但 `optimizer_rounds<=0` | CONTRACT FAIL(反滥用) |
| U4 restructure.gates 任一 false | CONTRACT FAIL |
| U5 空 restructure 块 / 伪造 trigger | CONTRACT FAIL(缺字段) |
| U6 有 restructure 块但 report.md 缺「重构过程」片段 | CONTRACT FAIL(用户报告记录缺失) |
| U7 `optimizer_rounds>0` 但 report.md 缺「调优过程」记录 | CONTRACT FAIL(用户报告记录缺失) |
| I1 端到端:`causal_conv1d_fwd` 走 restructure(load_block)→ tune → 出 best_code + restructure 块 + report.md 两节 | REACHED/BELOW 判定 + 溯源闸 + 报告记录闸通过 |
| I2 vendor 校验 | `git status` 确认 `triton-latency-optimizer` 零改动 |

---

## 11. 验收标准

1. `triton-latency-optimizer` 目录 `git diff` 为空(vendor 未动)。
2. 纯 tune(summary 无 restructure)契约结果与本期完全一致。
3. 带 restructure 的 summary 走 `restructure > tune > best_code + 契约 pass`,`applied_transform` 可溯源。
4. 重构后的 `best_code` 经官方 `verify.py`+`benchmark.py` 溯源;`perf_baseline=用户源 ≠ gate_ref` 恒成立。
5. `causal_conv1d_fwd` 走该阶段复现 ≥1.3980x(`1.6824/1.2034`),并向达到 2.0x 继续(swish/store 为下一变换候选)。
6. 用户 report.md 必含「调优过程」+「重构过程」两节(§8),字段与 summary 对齐;纯 tune 运行至少含「调优过程」。

---

## 12. 风险与待拍板

- **重构范围的灰度**:4 闸只能保证"方法体内部数据流改写 + I/O 冻结";若日后想重构到"换原语/布局/dtype",
  g1/g2 同样会"通过"(I/O 未变)但语义等价性需 g3 更强判定——**建议把允许范围显式锁死为
  "method-body-dataflow"**(§2 非目标),超出即拒。
- **`enabled` 默认 false**:避免改动既有的默认纯-tune 行为;需要用户在 config 显式开启。
- **本阶段是否要 patch**:restructure 属 kernel_replace 类(大 diff),沿用 `AGENTS.md §10 S4.5` 的
  kernel_replace 路径(字节派生、verify 重验、`apply_target==perf_baseline`)。
