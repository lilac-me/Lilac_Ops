# Ops Plugin — 昇腾算子开发

三阶段状态机流水线：Design → Implement → Optimize (3轮迭代)。支持 **Triton**、**Ascend C** 与 **TileLang** 三种开发语言，覆盖 Atlas A2/A3（arch22）与 Ascend 950（arch35）架构。

## 一行触发

Triton 路径：

```text
按照 plugins/ops/ops.yaml 的 design→implement→optimize 流程，用 Triton 生成一个 <算子名> 算子，A3 芯片 bf16，状态机确保全流程执行
```

替换 `<算子名>` 为目标算子即可。Agent 自动完成全部三个阶段，optimize 阶段通过 `state_machine.py` 强制执行 3 轮迭代（或收敛时提前停止）。

Ascend C 路径：

```text
按照 plugins/ops/ops.yaml 的 design→implement→optimize 流程，用 Ascend C 在 <A3|Ascend 950> 上生成一个 <算子名> 算子，标杆参考 <代码链接>，服务器 <IP>/<密码>/<容器>，状态机确保全流程执行并上板验证
```

替换 `<算子名>`、`<架构>`、`<代码链接>`、服务器信息为目标值即可。Agent 自动完成全部三个阶段，optimize 阶段通过 `state_machine.py` 强制执行 3 轮迭代（或收敛时提前停止）。

TileLang 路径：

```text
按照 plugins/ops/ops.yaml 的 design→implement→optimize 流程，用 TileLang 在 <A2/A3|Ascend 950> 上生成一个 <算子名> 算子，标杆参考 <代码链接>，服务器 <IP>/<密码>/<容器>，状态机确保全流程执行并上板验证
```

替换 `<算子名>`、`<架构>`、`<代码链接>`、服务器信息为目标值即可。A2/A3 走 tilelang-mlir-ascend 后端，Ascend 950 走 tilelang USE_ASCEND 后端。

## 目录结构

```text
plugins/ops/
├── .claude-plugin/
│   └── plugin.json              # 注册 skills
├── agents/
│   ├── ops-design-agent/
│   │   ├── agent.md             # 设计详设 Agent（Triton / Ascend C / TileLang）
│   │   └── agent.yaml
│   ├── ops-implement-agent/
│   │   ├── agent.md             # 落地 + Profiling Agent（Triton / Ascend C / TileLang）
│   │   └── agent.yaml
│   └── ops-optimize-agent/
│       ├── agent.md             # 迭代优化 Agent（Triton / Ascend C / TileLang）
│       └── agent.yaml
├── engine/
│   └── state_machine.py         # 状态机引擎
├── ops.yaml                     # 流水线定义（含 dev_lang / target_arch 变量）
└── README.md
```

## 流水线

```text
用户需求
  │
  ▼
Stage 1: Design (ops-design-agent)
  ├─ 需求分析 + NPU 可行性评估
  ├─ 确定开发方式: dev_lang = triton | ascendc | tilelang
  ├─ 硬件约束 (UB/Grid/对齐)
  ├─ Tiling 方案 (BLOCK_SIZE / BM/BN/BK)
  ├─ 内存规划 + 计算调度 + Kernel 架构
  └─ Write → outputs/<op>/design.md
  │
  ▼
Stage 2: Implement (ops-implement-agent)
  ├─ 读取 design.md
  ├─ triton: 生成 Triton kernel + pytest + benchmark
  ├─ ascendc: 生成 Ascend C 代码 (直调/DSL/aclNN/CATLASS)
  ├─ tilelang: 生成 TileLang kernel (@tilelang.jit) + pytest + benchmark
  ├─ SSH 远端实测 + torch.profiler 多 shape×dtype 采集
  ├─ 分析全 NPU 算子 + pipeline 利用率 + 瓶颈定位
  └─ Write → test_report + performance_report + profiling_report + execution_log
  │
  ▼
Stage 3: Optimize (ops-optimize-agent, 3 轮迭代)
  ├─ 读取 profiling_report.md → 识别瓶颈类型
  ├─ triton: 根据瓶颈选择优化手法：
  │   Memory Bound → 增大 BLOCK_SIZE, block_ptr, compile_hint
  │   Compute Bound → autotune/max_autotune, 降精度, 向量化
  │   Launch Bound → 固定核心 grid, 消除小 tile
  │   UB 溢出    → BLOCK_SIZE_SUB, buffer 复用
  ├─ ascendc: 根据瓶颈选择优化手法：
  |   UB 溢出 / 多核不均 / 流水断裂 / 950 CV 协同 (CATLASS/Blaze) 优化
  ├─ tilelang: 根据瓶颈选择优化手法：
  |   pipeline 未重叠 → num_stages 搜索 / tiling autotune / 片上复用 / dtype / 尾轴对齐
  ├─ 每轮: 修改 kernel → SSH 远端重跑 pytest+benchmark+profiler
  ├─ 对比优化前后: latency / speedup / HBM BW / pipeline 利用率
  └─ Write → optimize_round{1,2,3}.md
  │
  ▼
收敛条件 (任一满足即停止):
  - HBM 利用率 >= 95% (已达硬件极限)
  - speedup 达到设计目标
  - 连续 2 轮提升 < 5%
```

## 集成的 Skills

| Skill | 阶段 | 职责 |
| ------ | :---: | ------ |
| `ops-design` | Design | Triton 需求分析、可行性评估、Tiling 方案、内存规划 |
| `ascendc-design` | Design | Ascend C 需求分析、按 arch22/arch35 分支的 Tiling/内存/调度/架构设计 |
| `tilelang-design` | Design | TileLang 需求分析、按 tilelang-mlir-ascend/tilelang USE_ASCEND 分支的 Grid/片上内存/T.gemm tiling/Pipelined 设计 |
| `ascendc-vector-dev` | Design + Implement | **Ascend C 向量化开发基础**：DMA 数据搬运、向量硬件数学、顺序算法向量化、同步正确性、常见易错点 |
| `ascendc-adapter-layer` | Implement | **Ascend C 适配层设计**：单算子(`<<<>>>`+torch 扩展) vs aclnn 调用选型、适配代码、交付 README（用户如何调用算子） |
| `triton-op-dev` | Implement | Triton 代码生成、NPU 适配、性能优化 |
| `ascendc-op-dev` | Implement + Optimize | Ascend C 代码生成（直调/DSL/aclNN/CATLASS）、调试、性能优化编排 |
| `tilelang-op-dev` | Implement + Optimize | TileLang 代码生成（@tilelang.jit + T.copy/T.gemm/T.Pipelined）、环境确认、性能优化 |
| `ops-test-gen` | Implement | pytest 精度/性能测试、L1-L5 验收（三条路径共用） |
| `ops-bwd-golden` | Implement | 反向 torch 标杆生成（torch 自动求导），跨语言梯度精度 golden |
| `ops-profiling` | Implement + Optimize | torch.profiler/msprof 采集、pipeline 分析、优化建议产出（共用） |
| `ops-boundary-survey` | Implement | 算子边界摸排、边界测试用例生成与验证 |

### Ascend C 路径复用的可复用 Skills（来自 cannbot / 本仓库）

Ascend C算子生成需使用cannbot中的Skills，在使用Ascend C算子生成Agent前，请按照[cannbot-skills安装指导](https://gitcode.com/cann/cannbot-skills/blob/master/docs/installation-guide.md)完成前置Skills的安装。

- 工程模板：`ascendc-direct-invoke-template`、`ascendc-registry-invoke-template`、`torch-ascendc-op-extension`、`dsl-baseline-generation`、`dsl-lowering`
- 950 Matmul/融合：`catlass-op-design`、`catlass-op-develop`、`catlass-op-perf-tune`、`ascendc-blaze-best-practice`
- 约束/API：`ascendc-tiling-design`、`npu-arch`、`ascendc-api-best-practices`、`ascendc-docs-search`、`ascendc-env-check`
- 调试：`ascendc-op-debug`、`ascendc-precision-debug`、`ascendc-runtime-debug`、`ascendc-crash-debug`
- 优化/检视：`ascendc-perf-optimize`、`code-performance-advisor`、`ascendc-code-review`、`cake-code-review`

## 远端执行

需要询问使用者，以提供可用环境

```ini
BENCH_NPU_IP=IP
BENCH_NPU_USER=root
BENCH_NPU_PASSWORD=your_password
BENCH_NPU_CONTAINER=xuxu
DEV_LANG=ascendc          # 或 triton / tilelang
TARGET_ARCH=arch35        # arch22 (A2/A3) | arch35 (Ascend 950)
```

无需手动确认——远端操作全部自动执行。

## ops.yaml 定制

`ops.yaml` 定义了流水线的阶段、Agent 映射、模板变量。修改 YAML 即可调整流程，无需改代码：

- 新增阶段：在 `pipeline` 下添加 stage，声明 `depends_on`
- 调整 prompt：修改各 stage 的 `prompt` 模板（已内置 `{dev_lang}` / `{target_arch}` 分支）
- 新增 Agent：在 `agents` 下注册，`pipeline` 中引用
- 语言/架构切换：通过 `variables.dev_lang` 与 `variables.target_arch` 控制

## 与 Benchmark 的关系

```text
plugins/ops/ops.yaml     ← 单算子开发流水线（Triton / Ascend C 双路径）
benchmark/               ← 批量评测框架（36 用例）
  └─ 加载 cases → 逐用例调 ops-dev → 五维评分 → 报告
```
