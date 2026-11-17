#!/usr/bin/env python3
"""Validate the two Markdown deliverables of the VeOmni quick-migration workflow."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_COVERAGE_MODULES = ("dataset", "dataloader", "model", "config")
REQUIRED_COVERAGE_SECTIONS = ("范围与结论", "模块覆盖明细", "训练、精度与性能", "风险与待确认事项")
REQUIRED_PROCESS_SECTIONS = ("输入与可追溯信息", "执行记录", "关键决策、失败与回退", "未决项与授权边界")
ALLOWED_STATUSES = ("COVERED", "PARTIAL", "NOT_REQUIRED", "BLOCKED", "UNKNOWN")


def read_markdown(path: Path) -> tuple[str | None, list[str]]:
    if not path.is_file():
        return None, [f"文件不存在: {path}"]
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return text, [f"文件为空: {path}"]
    return text, []


def check_coverage(text: str) -> list[str]:
    errors: list[str] = []
    if not text.startswith("# VeOmni 迁移交付件覆盖报告"):
        errors.append("覆盖报告缺少一级标题“VeOmni 迁移交付件覆盖报告”")
    for section in REQUIRED_COVERAGE_SECTIONS:
        if f"## {section}" not in text:
            errors.append(f"覆盖报告缺少章节: {section}")
    table_modules = {
        columns[1].strip().lower()
        for line in text.splitlines()
        if line.startswith("|") and line.endswith("|")
        for columns in [line.split("|")]
        if len(columns) > 2 and columns[1].strip() not in {"模块", "---"}
        and not set(columns[1].strip()) <= {"-", ":"}
    }
    for module in REQUIRED_COVERAGE_MODULES:
        if module not in table_modules:
            errors.append(f"覆盖报告未覆盖必填模块: {module}")
    if not any(status in text for status in ALLOWED_STATUSES):
        errors.append("覆盖报告没有有效覆盖状态")
    if "| 模块 |" not in text:
        errors.append("覆盖报告缺少模块覆盖表格")
    return errors


def check_process(text: str) -> list[str]:
    errors: list[str] = []
    if not text.startswith("# VeOmni 迁移过程记录"):
        errors.append("过程记录缺少一级标题“VeOmni 迁移过程记录”")
    for section in REQUIRED_PROCESS_SECTIONS:
        if f"## {section}" not in text:
            errors.append(f"过程记录缺少章节: {section}")
    if "| 序号 |" not in text or ("操作" not in text and "阶段" not in text):
        errors.append("过程记录缺少执行记录表格")
    if "授权" not in text:
        errors.append("过程记录未记录授权边界")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 VeOmni 快速迁移的两份 Markdown 交付件")
    parser.add_argument("coverage_report", type=Path)
    parser.add_argument("process_record", type=Path)
    args = parser.parse_args()

    coverage, errors = read_markdown(args.coverage_report)
    process, process_errors = read_markdown(args.process_record)
    errors.extend(process_errors)
    if coverage is not None:
        errors.extend(check_coverage(coverage))
    if process is not None:
        errors.extend(check_process(process))

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[PASS] VeOmni 快速迁移交付件结构完整")
    return 0


if __name__ == "__main__":
    sys.exit(main())
