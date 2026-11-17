from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


CHECKER = Path(__file__).resolve().parents[1] / "check_outputs.py"
SPEC = importlib.util.spec_from_file_location("quick_migration_checker", CHECKER)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


VALID_COVERAGE = """# VeOmni 迁移交付件覆盖报告

## 范围与结论
已执行代码适配和验证。

## 模块覆盖明细
| 模块 | 源端证据 | VeOmni 目标或参考实现 | 映射/交付说明 | 状态 | 缺口与下一步 |
|---|---|---|---|---|---|
| dataset | `src/data.py: Dataset` | `veomni/data.py: Dataset` | 样本字段一致 | COVERED | 无 |
| dataloader | `src/loader.py: build` | `veomni/loader.py: build` | collate 待确认 | PARTIAL | 确认动态 padding |
| model | `src/model.py: Model` | `veomni/model.py: Model` | 未读取全部源码 | UNKNOWN | 补充源码 |
| config | `src/config.yaml: model` | `veomni/config.yaml: model` | 缺少运行配置 | BLOCKED | 提供配置 |

## 训练、精度与性能
| 项目 | 执行状态 | 结构化证据 | 结论 |
|---|---|---|---|
| 最小训练 | NOT_RUN | `functional_report.json` | 等待授权 |

## 风险与待确认事项
1. 需要确认 tokenizer。
"""

VALID_PROCESS = """# VeOmni 迁移过程记录

## 输入与可追溯信息
已记录版本。

## 执行记录
| 序号 | 阶段 | 实际操作/代码改动 | 命令或日志证据 | 结果 |
|---|---|---|---|---|
| 1 | 代码适配 | 读取模型 | `src/model.py` | 已记录 |

## 关键决策、失败与回退
1. 依据文件路径作映射。

## 未决项与授权边界
1. 无。
2. 训练需要授权。
"""


class CheckOutputsTests(unittest.TestCase):
    def test_valid_deliverables_pass(self) -> None:
        self.assertEqual(checker.check_coverage(VALID_COVERAGE), [])
        self.assertEqual(checker.check_process(VALID_PROCESS), [])

    def test_coverage_requires_all_primary_modules(self) -> None:
        errors = checker.check_coverage(VALID_COVERAGE.replace("| config |", "| settings |"))
        self.assertIn("覆盖报告未覆盖必填模块: config", errors)

    def test_missing_files_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text, errors = checker.read_markdown(Path(tmp) / "missing.md")
        self.assertIsNone(text)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
