# Ops Plugin — Codex Agents

> Codex 原生 Agent 定义（Claude Code 版见 `agents/*/agent.md`）
> YAML 状态机共享: `plugins/ops/ops.yaml`
> 运行时引擎: `plugins/ops/engine/state_machine.py`

## 流水线

```text
Design → Implement → Optimize (3轮迭代)
```

状态转换由 `state_machine.py` 驱动，`ops.yaml` 单源真相。

**三语言路径**：每个阶段按 `dev_lang`（`triton` / `ascendc` / `tilelang`）分支执行，由设计阶段确定。
Ascend C 路径额外区分目标架构 `target_arch`：`arch22`（Atlas A2/A3）与 `arch35`（Ascend 950）。
TileLang 路径按 `target_arch` 区分后端：`arch22` → tilelang-mlir-ascend，`arch35` → tilelang USE_ASCEND。

```text
用户需求(算子名 + 标杆代码/参考链接 + 服务器IP/密码)
   │
   ▼
Stage 1: Design (ops-design-agent)
   ├─ dev_lang = triton   → ops-design
   ├─ dev_lang = ascendc  → ascendc-design (按 arch22/arch35 分支)
   └─ dev_lang = tilelang → tilelang-design (按 tilelang-mlir-ascend / tilelang USE_ASCEND 分支)
   │
   ▼
Stage 2: Implement (ops-implement-agent)
   ├─ triton   → triton-op-dev + ops-test-gen + ops-profiling
   ├─ ascendc  → ascendc-op-dev (+ 直调/DSL/aclNN/CATLASS) + ops-test-gen + ops-profiling
   └─ tilelang → tilelang-op-dev + ops-test-gen + ops-profiling
   │
   ▼
Stage 3: Optimize (ops-optimize-agent, 3轮迭代)
   ├─ triton   → triton-op-dev 优化手法
   ├─ ascendc  → ascendc-op-dev + ascendc-perf-optimize / catlass-op-perf-tune
   └─ tilelang → tilelang-op-dev 优化手法（num_stages/tiling/片上复用/dtype）
```

## Agent 清单

| Agent | 文件 | 职责 |
| ------ | ------ | ------ |
| ops-design-agent | `agents/ops-design-agent/agent.yaml` | 需求分析、可行性评估、Tiling 方案、内存规划（Triton / Ascend C / TileLang） |
| ops-implement-agent | `agents/ops-implement-agent/agent.yaml` | 代码生成、远端测试、Profiling 采集、报告产出（Triton / Ascend C / TileLang） |
| ops-optimize-agent | `agents/ops-optimize-agent/agent.yaml` | Profiling 瓶颈分析、定向优化、3轮迭代、收敛判断（Triton / Ascend C / TileLang） |

## Codex 调用方式

```bash
# 状态机驱动
python -m plugins.ops.engine.state_machine next \
  --yaml plugins/ops/ops.yaml \
  --workspace outputs/<op> \
  --handoff "→ handoff: DESIGN_DONE"

# 查询当前状态
python -m plugins.ops.engine.state_machine current \
  --workspace outputs/<op>

# 验证日志
python -m plugins.ops.engine.state_machine verify \
  --yaml plugins/ops/ops.yaml \
  --workspace outputs/<op>
```

## 与 Claude Code 版的差异

| | Claude Code | Codex | DSH |
| ------ | :---: | :---: | :---: |
| Agent 入口 | `.claude-plugin/plugin.json` | `.codex-plugin/plugin.json` | `.dsh/skills/` 投影 |
| Agent 定义 | `agent.md` (frontmatter + 中文) | `agent.yaml` (纯 YAML) | 角色 prompt 内联（无 agent.yaml） |
| Agent 调用 | Agent 工具 (`subagent_type`) | Codex task/subtask 系统 | DSH subagent 工具 |
| 状态机 | `engine/state_machine.py` | 同一份 `engine/state_machine.py` | 同一份 |
| 流水线 | `ops.yaml` | 同一份 `ops.yaml` | 同一份 |
