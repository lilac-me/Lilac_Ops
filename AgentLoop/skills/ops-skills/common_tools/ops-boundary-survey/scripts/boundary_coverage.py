#!/usr/bin/env python3
"""
boundary_coverage.py - 生成边界 UT 用例覆盖率报告

根据边界摸排报告的「边界用例表」提取全部边界场景（约束内 + 约束外），
与 UT 文件中的 test_<case_id>_* 测试方法匹配，计算 UT 用例覆盖率，
输出覆盖矩阵与覆盖率报告。

用法:
    python boundary_coverage.py --report <boundary_report.md> --ut <test_xxx_boundary.py> \
        --output <coverage_report.md>

约定:
    - 报告「边界用例表」每行一个边界场景，需含「用例 ID」「维度」列
      （约束内表: | # | 用例 ID | 维度 | ...；约束外表: 同理）
    - UT 测试方法名为 test_<case_id>_<direction>（direction 可为 forward/backward/error 等）
    - 场景覆盖判定: 该场景 ID 在 UT 中至少存在一个对应测试方法
"""
import argparse
import os
import re
import sys

# 表头分隔行（如 |---|----|）
SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_tables(md_text):
    """从 markdown 中提取所有表格（含所属小节标题），返回 [(section, header, rows)]。"""
    tables = []
    section = ""
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            i += 1
            continue
        if line.startswith("|"):
            header = None
            rows = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                content = lines[j].strip()
                if header is None:
                    header = content
                elif SEP_RE.match(content):
                    pass
                else:
                    rows.append(content)
                j += 1
            tables.append((section, header, rows))
            i = j
            continue
        i += 1
    return tables


def parse_row(row_str):
    """将表格行解析为单元格列表（去除空白与首尾空格）。"""
    inner = row_str.strip().lstrip("|").rstrip("|")
    return [c.strip() for c in inner.split("|")]


def find_column(header_str, *names):
    """在表头中定位列名，返回列索引；找不到返回 None。"""
    cells = parse_row(header_str)
    for idx, cell in enumerate(cells):
        for name in names:
            if cell.lower() == name.lower():
                return idx
    return None


def extract_scenarios(report_text):
    """从边界报告提取全部边界场景，返回 [(case_id, dimension, section)]。"""
    scenarios = []
    tables = extract_tables(report_text)
    for section, header, rows in tables:
        if "用例" not in section and "约束内" not in section and "约束外" not in section:
            continue
        id_idx = find_column(header, "用例ID", "用例 ID", "case id")
        dim_idx = find_column(header, "维度", "分类", "边界维度")
        if id_idx is None:
            continue
        for row in rows:
            cells = parse_row(row)
            if len(cells) <= id_idx:
                continue
            case_id = cells[id_idx].strip()
            if not case_id or case_id == "-" or case_id == "..." or set(case_id) == {"."}:
                continue
            dim = cells[dim_idx].strip() if dim_idx is not None and dim_idx < len(cells) else "未标注"
            scenarios.append({"id": case_id, "dim": dim, "section": section})
    return scenarios


def extract_ut_methods(ut_text):
    """从 UT 文件提取所有测试方法名。"""
    methods = re.findall(r"^\s*def\s+(test_[A-Za-z0-9_]+)\s*\(", ut_text, re.MULTILINE)
    return set(methods)


def build_coverage(scenarios, ut_methods):
    """为每个场景匹配 UT 方法，返回覆盖状态列表。"""
    covered, uncovered = [], []
    for sc in scenarios:
        matched = [m for m in ut_methods if m.startswith("test_" + sc["id"] + "_") or m == "test_" + sc["id"]]
        if matched:
            covered.append({**sc, "methods": sorted(matched)})
        else:
            uncovered.append({**sc, "methods": []})
    return covered, uncovered


def render_report(op_name, scenarios, covered, uncovered):
    """生成覆盖率报告 markdown。"""
    total = len(scenarios)
    cov = len(covered)
    rate = (cov / total * 100) if total else 0.0

    lines = []
    lines.append(f"# {op_name} 边界 UT 用例覆盖率报告")
    lines.append("")
    lines.append("## 覆盖率统计")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 边界场景总数（约束内 + 约束外） | {total} |")
    lines.append(f"| 已覆盖场景数（存在对应 UT 方法） | {cov} |")
    lines.append(f"| **UT 用例覆盖率** | **{rate:.1f}%** |")
    lines.append("")
    lines.append("> UT 用例覆盖率 = 已实现 UT 方法数 ÷ 边界报告中全部边界场景数。")
    lines.append("")

    lines.append("## 覆盖矩阵")
    lines.append("")
    groups = {}
    for sc in scenarios:
        groups.setdefault(sc["dim"], []).append(sc)
    for dim in sorted(groups):
        lines.append(f"### {dim}")
        lines.append("")
        lines.append("| # | 用例 ID | 所属表格 | UT 方法 | 覆盖状态 |")
        lines.append("|---|---------|----------|---------|:---:|")
        for idx, sc in enumerate(groups[dim], 1):
            if sc["id"] in {c["id"] for c in covered}:
                methods = "; ".join(next(c["methods"] for c in covered if c["id"] == sc["id"]))
                status = "OK"
            else:
                methods = "-"
                status = "MISS"
            lines.append(f"| {idx} | {sc['id']} | {sc['section']} | `{methods}` | {status} |")
        lines.append("")

    lines.append("## 未覆盖场景")
    lines.append("")
    if uncovered:
        lines.append("| # | 用例 ID | 维度 | 所属表格 |")
        lines.append("|---|---------|------|----------|")
        for idx, sc in enumerate(uncovered, 1):
            lines.append(f"| {idx} | {sc['id']} | {sc['dim']} | {sc['section']} |")
    else:
        lines.append("无，全部边界场景均已有对应 UT 方法。")
    lines.append("")

    lines.append("## 结论")
    lines.append("")
    if rate >= 95.0:
        lines.append(f"OK UT 用例覆盖率 {rate:.1f}% >= 95%，达标。")
    else:
        lines.append(f"FAIL UT 用例覆盖率 {rate:.1f}% < 95%，未达标。请补充以下场景的 UT 方法：")
        for sc in uncovered:
            lines.append(f"  - `{sc['id']}`（{sc['dim']}）")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="生成边界 UT 用例覆盖率报告")
    parser.add_argument("--report", required=True, help="边界摸排报告 markdown 路径")
    parser.add_argument("--ut", required=True, help="边界 UT 文件路径")
    parser.add_argument("--output", required=True, help="覆盖率报告输出路径")
    args = parser.parse_args()

    if not os.path.exists(args.report):
        print(f"错误: 报告文件不存在 {args.report}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.ut):
        print(f"错误: UT 文件不存在 {args.ut}", file=sys.stderr)
        sys.exit(1)

    report_text = read_text(args.report)
    ut_text = read_text(args.ut)

    scenarios = extract_scenarios(report_text)
    if not scenarios:
        print("错误: 未从报告中解析到任何边界场景（请确认报告含「用例 ID」「维度」列）", file=sys.stderr)
        sys.exit(1)

    ut_methods = extract_ut_methods(ut_text)
    covered, uncovered = build_coverage(scenarios, ut_methods)

    op_name = os.path.splitext(os.path.basename(args.ut))[0]
    report = render_report(op_name, scenarios, covered, uncovered)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    total, cov = len(scenarios), len(covered)
    print(f"边界场景总数: {total}, 已覆盖: {cov}, UT 用例覆盖率: {cov / total * 100:.1f}%")
    if uncovered:
        print("未覆盖场景:")
        for sc in uncovered:
            print(f"  - {sc['id']} ({sc['dim']})")


if __name__ == "__main__":
    main()
