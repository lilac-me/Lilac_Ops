#!/usr/bin/env python3
"""VeOmni 产物契约校验。

读取 outputs/<model>/ 下的结构化产物，对照
benchmarks/veomni/schemas/ 中的 JSON Schema 子集声明
（type / required / properties / enum / items /
minItems / minLength / minimum / exclusiveMinimum / maximum）
逐字段检查并收集偏差。

用法:
    python benchmarks/veomni/evaluate/evaluate.py outputs/<model>/
    python benchmarks/veomni/evaluate/evaluate.py outputs/<model>/environment_report.json
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


class YamlUnavailable(Exception):
    """PyYAML 未安装，无法解析 YAML 产物。"""


def load_json(path: Path):
    return _json.loads(path.read_text(encoding="utf-8"))


def load_artifact(path: Path):
    """按后缀加载产物；YAML 依赖 PyYAML。"""
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise YamlUnavailable(str(path))
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return load_json(path)


def find_artifact(base: Path, schema_id: str):
    """按 schema 名在产物目录中查找同名产物（.json/.yaml）。"""
    stem = Path(schema_id).stem
    for ext in (".json", ".yaml", ".yml"):
        candidate = base / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


_SCALAR_PREDICATES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def check_type(value, spec) -> bool:
    """spec 类型可以是单个或列表（含 "null"）。"""
    candidates = spec if isinstance(spec, list) else [spec]
    return any(_SCALAR_PREDICATES.get(c, lambda _: True)(value) for c in candidates)


class SchemaChecker:
    """逐字段对照 JSON Schema 子集，收集偏差描述。"""

    def __init__(self):
        self.problems: list[str] = []

    def _number_bounds(self, value, rule, where: str):
        floor = rule.get("minimum")
        if floor is not None and value < floor:
            self.problems.append(f"{where}: {value} < minimum {floor}")
        floor_excl = rule.get("exclusiveMinimum")
        if floor_excl is not None and value <= floor_excl:
            self.problems.append(f"{where}: {value} <= exclusiveMinimum {floor_excl}")
        ceil_ = rule.get("maximum")
        if ceil_ is not None and value > ceil_:
            self.problems.append(f"{where}: {value} > maximum {ceil_}")

    def _text_bounds(self, value, rule, where: str):
        floor = rule.get("minLength")
        if floor is not None and len(value) < floor:
            self.problems.append(f"{where}: 字符串长度 < minLength {floor}")

    def _list_bounds(self, value, rule, where: str):
        floor = rule.get("minItems")
        if floor is not None and len(value) < floor:
            self.problems.append(f"{where}: 数组长度 {len(value)} < minItems {floor}")

    def run(self, value, rule, where="$"):
        if rule is True or rule == {}:
            return
        allowed = rule.get("enum")
        if allowed is not None and value is not None and value not in allowed:
            self.problems.append(f"{where}: 值 {value!r} 不在枚举 {allowed}")
            return
        declared = rule.get("type")
        if declared is not None and not check_type(value, declared):
            self.problems.append(f"{where}: 期望类型 {declared}，实际 {type(value).__name__}")
            return
        if isinstance(value, dict):
            for field in rule.get("required", []):
                if field not in value:
                    self.problems.append(f"{where}: 缺少必填字段 '{field}'")
            for field, sub_rule in rule.get("properties", {}).items():
                if field in value:
                    self.run(value[field], sub_rule, f"{where}.{field}")
        elif isinstance(value, list):
            self._list_bounds(value, rule, where)
            sub_rule = rule.get("items")
            if sub_rule:
                for idx, item in enumerate(value):
                    self.run(item, sub_rule, f"{where}[{idx}]")
        elif isinstance(value, str):
            self._text_bounds(value, rule, where)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            self._number_bounds(value, rule, where)


def validate(value, schema, path="$", errors=None) -> list:
    """对照 schema 逐字段校验，返回偏差描述列表。"""
    checker = SchemaChecker()
    checker.run(value, schema, path)
    if errors is not None:
        errors.extend(checker.problems)
    return checker.problems


def evaluate_file(artifact: Path, schema: dict) -> tuple:
    try:
        data = load_artifact(artifact)
    except YamlUnavailable:
        return None, []
    except _json.JSONDecodeError as e:
        return False, [f"JSON 解析失败: {e}"]
    except OSError as e:
        return False, [f"文件读取失败: {e}"]
    checker = SchemaChecker()
    checker.run(data, schema)
    return (not checker.problems), checker.problems


def main() -> int:
    parser = argparse.ArgumentParser(description="VeOmni 产物 Schema 校验")
    parser.add_argument("target", type=Path, help="产物目录或单个产物文件")
    parser.add_argument("--schema-dir", type=Path, default=SCHEMA_DIR)
    args = parser.parse_args()

    target = args.target
    if target.is_file():
        schema_file = args.schema_dir / f"{target.stem}.json"
        if not schema_file.exists():
            print(f"[FAIL] 找不到 Schema: {schema_file}")
            return 2
        schema = load_json(schema_file)
        ok, errs = evaluate_file(target, schema)
        if ok is None:
            print(f"[SKIP] {target}: 需要 PyYAML 解析 YAML 产物（pip install pyyaml）")
            return 2
        print(f"{'[PASS]' if ok else '[FAIL]'} {target}")
        for e in errs:
            print(f"       - {e}")
        return 0 if ok else 1

    if not target.is_dir():
        print(f"[FAIL] 目标不存在: {target}")
        return 2

    total, passed, skipped = 0, 0, 0
    for schema_file in sorted(args.schema_dir.glob("*.json")):
        schema = load_json(schema_file)
        artifact = find_artifact(target, schema_file.name)
        total += 1
        if artifact is None:
            print(f"[MISS] {schema_file.stem}: 产物不存在（阶段未执行或未落盘）")
            continue
        ok, errs = evaluate_file(artifact, schema)
        if ok is None:
            skipped += 1
            print(f"[SKIP] {artifact.name}: 需要 PyYAML（pip install pyyaml）")
        elif ok:
            passed += 1
            print(f"[PASS] {artifact.name}")
        else:
            print(f"[FAIL] {artifact.name}")
            for e in errs:
                print(f"       - {e}")
    verdict = "全部合规" if passed + skipped == total and skipped == 0 else \
              (f"{skipped} 个因缺 PyYAML 未校验" if passed + skipped == total else f"{total - passed - skipped} 个缺失或不合规")
    print(f"\n校验完成: {passed}/{total} 通过（{verdict}）")
    return 0 if passed + skipped == total else 1


if __name__ == "__main__":
    sys.exit(main())
