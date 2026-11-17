# VeOmni Benchmarks

VeOmni 迁移闭环的验证资源。代码适配、训练、精度和性能阶段通过 JSON/YAML 产物进行机器门禁；最终生成 `acceptance_report.json` 作为机器验收契约，并汇总为两份 Markdown 交付件。

## 目录

| 目录 | 内容 |
| ------ | ------ |
| `quick_migration/` | 最终交付件结构校验器与单元测试 |
| `schemas/` | 环境、模型、Patch、功能、精度、MFU、性能等阶段的 JSON Schema |
| `scenarios/`、`fixtures/` | Dense、MoE、VLM、DiT、Omni 等场景与最小配置 |
| `evaluate/`、`scorer/` | 结构化产物 Schema 校验和流水线门禁评分 |

## 最终交付件校验

```bash
python benchmarks/veomni/quick_migration/check_outputs.py \
  outputs/<model>/delivery_coverage_report.md \
  outputs/<model>/migration_process_record.md

python -m unittest discover -s benchmarks/veomni/quick_migration/tests -v
```

校验器要求覆盖报告至少包含 dataset、dataloader、model、config，以及训练、精度和性能的执行状态；过程记录必须包含执行记录与授权边界。它只校验报告结构，运行结论仍以阶段 JSON/YAML 和真实日志为准。

## 阶段证据校验

```bash
python -m pip install -r benchmarks/veomni/requirements.txt
python benchmarks/veomni/run_scenarios.py
python benchmarks/veomni/evaluate/evaluate.py outputs/<model>/
python benchmarks/veomni/scorer/score.py outputs/<model>/
```

真实 NPU 训练、精度对齐和性能调优需要使用目标模型、数据与已授权环境执行。
