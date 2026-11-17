# 远端精度验证脚本合同

本文记录远端精度验证涉及的 13 个有效脚本。正常执行只读取对应 Skill 和入口 `--help`；只有审计脚本集合、修改接口或根据结构化失败阶段诊断时读取本文。

## 边界与入口

- 本地唯一入口：`remote-validation/scripts/run_validation.py`。
- 远端仓库 launcher：`examples/fsdp2/<model>/precision_alignment/run_complete_alignment.sh`，只负责模型参数映射。
- 精度唯一公共入口：`precision-alignment/scripts/run_alignment.py`。
- 远端 runtime：只同步 8 个运行必需 Python 文件，不同步环境工具和测试。

## 端到端调用图

```text
run_validation.py
├─ rsync MindSpeed-LLM worktree
├─ rsync 8-file precision runtime
├─ preflight + npu-smi idle gate
└─ run_complete_alignment.sh
   └─ run_alignment.py
      ├─ make_fixed_input.py ──────────────→ alignment_input.pt
      ├─ prepare_checkpoint_view.py ───────→ layers_N/checkpoint_view/ + manifest
      ├─ dump_transformers_logits.py ──────→ layers_N/reference_logits.pt
      ├─ dump_mindspeed_fsdp2_logits.py ───→ layers_N/candidate_logits.pt
      ├─ compare_logits.py ────────────────→ layers_N/comparison_report.json
      └─ write_alignment_report.py ────────→ alignment_report.md

alignment_utils.py
└─ 被固定输入、两个 dump 和比较器共同导入，不是独立入口
```

## 13 个脚本的职责

| # | 脚本 | 分类 | 主要输入 | 输出/作用 | 调用与同步 |
| --- | --- | --- | --- | --- | --- |
| 1 | `remote-validation/scripts/run_validation.py` | 本地控制器 | 私有远端配置、worktree | 同步、预检、NPU 门禁、执行、产物回收、validation report | 用户/主控调用；不远端同步自身 |
| 2 | `examples/fsdp2/<model>/precision_alignment/run_complete_alignment.sh` | 模型薄 launcher | 远端环境变量、模型专用 YAML | 将模型参数传给 `run_alignment.py` | 控制器调用；随 worktree 同步 |
| 3 | `precision-alignment/scripts/run_alignment.py` | 运行必需、唯一精度入口 | 权重、仓库、Candidate YAML、两个 Python、层数/阈值 | 编排全流程；生成 status、record、报告与分层目录 | launcher 调用；进入 runtime |
| 4 | `precision-alignment/scripts/alignment_utils.py` | 运行必需、内部库 | tensor payload | 输入校验、hash、logits layout、JSON 规范化 | 被 4 个内部步骤导入；进入 runtime |
| 5 | `precision-alignment/scripts/make_fixed_input.py` | 运行必需、内部步骤 | tokenizer、prompt、长度 | `alignment_input.pt`、摘要日志 | `run_alignment.py` 调用；进入 runtime |
| 6 | `precision-alignment/scripts/prepare_checkpoint_view.py` | 运行必需、内部步骤 | 原权重目录、层数、显式 config override | 只读 symlink checkpoint view、manifest | `run_alignment.py` 调用；进入 runtime |
| 7 | `precision-alignment/scripts/dump_transformers_logits.py` | 运行必需、内部步骤 | checkpoint view、固定输入、Reference 参数 | 原生 Transformers FP32 logits payload | Reference Python 调用；进入 runtime |
| 8 | `precision-alignment/scripts/dump_mindspeed_fsdp2_logits.py` | 运行必需、内部步骤 | Candidate YAML、checkpoint view、固定输入 | FSDP2 training-forward logits、执行和权重加载证明 | Candidate `torchrun` 调用；进入 runtime |
| 9 | `precision-alignment/scripts/compare_logits.py` | 运行必需、内部步骤 | 两份 logits payload、阈值 | comparison JSON；返回 `0` 或 `2` | Reference Python 调用；进入 runtime |
| 10 | `precision-alignment/scripts/write_alignment_report.py` | 运行必需、内部步骤 | `run_record.json` | 中文 `alignment_report.md` | Reference Python 调用；进入 runtime |
| 11 | `precision-alignment/scripts/setup_transformers_runtime.sh` | 环境工具 | Transformers commit、checkout、venv | 隔离 Reference venv、环境报告 | 仅显式准备环境时调用；不进入 runtime |
| 12 | `precision-alignment/tests/test_alignment_tools.py` | 开发测试 | PyTorch、pytest | 数值、payload、checkpoint 和比较器单元测试 | 开发者调用；不进入 runtime |
| 13 | `precision-alignment/tests/test_orchestration.py` | 开发测试 | Python 标准库 | 编排、环境隔离、日志尾部、失败 status 测试 | 开发者调用；不进入 runtime |

## 稳定输入输出

- 控制器输入合同：`remote-validation/references/remote_config_schema.json`。
- 控制器输出合同：`remote-validation/references/validation_report_schema.json`。
- 精度入口输入合同：`run_alignment.py --help`；模型 launcher 不增加第二套业务参数。
- 精度状态：`alignment_status.json`，包含 `state`、`current_stage`、`layer_count`、结构化 `error`。
- 精度执行记录：`run_record.json`，包含逐命令、退出码、日志、输出末尾、数值状态和最早失败阶段。
- 人工报告：`alignment_report.md`；不能替代 status/record 的机器判断。

## 失败诊断顺序

1. 读取本地 `validation_report.json`，区分同步、preflight、NPU、launcher 或产物回收失败。
2. launcher 已启动时读取回收的 `alignment_status.json`，确定内部阶段和层数。
3. 读取 `run_record.json` 中该阶段的命令、退出码、`output_tail` 和单个日志。
4. 只有 status、record、日志仍显示接口违约时，读取失败阶段对应的一个内部脚本。

## 已裁剪文件

`audit_alignment_harness.py` 未被入口、测试或远端控制器调用，已删除，不计入上述 13 个有效脚本。其仍有价值的环境边界由 remote-validation preflight、runtime allowlist、launcher source guard 和编排测试覆盖。
