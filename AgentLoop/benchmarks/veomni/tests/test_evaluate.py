from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1] / "evaluate"
sys.path.insert(0, str(EVAL_DIR))

import evaluate  # noqa: E402


class CheckTypeTests(unittest.TestCase):
    def test_bool_is_not_integer(self) -> None:
        # bool 是 int 的子类，必须显式排除
        self.assertFalse(evaluate.check_type(True, "integer"))
        self.assertFalse(evaluate.check_type(True, "number"))
        self.assertTrue(evaluate.check_type(True, "boolean"))

    def test_number_accepts_int_and_float(self) -> None:
        self.assertTrue(evaluate.check_type(3, "number"))
        self.assertTrue(evaluate.check_type(3.5, "number"))
        self.assertTrue(evaluate.check_type(3, "integer"))
        self.assertFalse(evaluate.check_type(3.5, "integer"))

    def test_null_in_union(self) -> None:
        self.assertTrue(evaluate.check_type(None, ["string", "null"]))
        self.assertTrue(evaluate.check_type("x", ["string", "null"]))
        self.assertFalse(evaluate.check_type(1, ["string", "null"]))


class ValidateTests(unittest.TestCase):
    def test_missing_required(self) -> None:
        schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}}
        errs = evaluate.validate({}, schema)
        self.assertTrue(any("缺少必填字段 'a'" in e for e in errs))

    def test_enum_violation(self) -> None:
        schema = {"type": "string", "enum": ["PASS", "FAIL"]}
        errs = evaluate.validate("BAD", schema)
        self.assertTrue(any("不在枚举" in e for e in errs))

    def test_enum_allows_null(self) -> None:
        schema = {"type": ["string", "null"], "enum": ["A", "B", None]}
        self.assertEqual(evaluate.validate(None, schema), [])

    def test_nested_list_paths(self) -> None:
        schema = {
            "type": "object",
            "required": ["pairs"],
            "properties": {
                "pairs": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["conclusion"],
                        "properties": {"conclusion": {"type": "string"}},
                    },
                }
            },
        }
        errs = evaluate.validate({"pairs": [{"conclusion": 1}]}, schema)
        self.assertTrue(any("$" in e and "pairs[0].conclusion" in e for e in errs))

    def test_string_and_number_bounds(self) -> None:
        self.assertTrue(evaluate.validate("ab", {"type": "string", "minLength": 3}))
        self.assertTrue(evaluate.validate(1, {"type": "number", "minimum": 2}))
        self.assertTrue(evaluate.validate(2, {"type": "number", "exclusiveMinimum": 2}))
        self.assertTrue(evaluate.validate(5, {"type": "number", "maximum": 4}))
        self.assertEqual(evaluate.validate("abc", {"type": "string", "minLength": 3}), [])


class EvaluateFileTests(unittest.TestCase):
    def test_valid_json_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            p.write_text(json.dumps({"a": "ok"}), encoding="utf-8")
            ok, errs = evaluate.evaluate_file(p, {"type": "object", "required": ["a"]})
            self.assertTrue(ok)
            self.assertEqual(errs, [])

    def test_broken_json_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            p.write_text("{not json", encoding="utf-8")
            ok, errs = evaluate.evaluate_file(p, {"type": "object"})
            self.assertFalse(ok)
            self.assertTrue(any("JSON 解析失败" in e for e in errs))

    def test_yaml_without_pyyaml_returns_none(self) -> None:
        # 无 PyYAML 环境下 YAML 产物返回 (None, [])，由调用方标记 SKIP
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.yaml"
            p.write_text("a: 1\n", encoding="utf-8")
            try:
                import yaml  # noqa: F401
            except ImportError:
                ok, errs = evaluate.evaluate_file(p, {"type": "object"})
                self.assertIsNone(ok)
                self.assertEqual(errs, [])
            else:
                ok, errs = evaluate.evaluate_file(p, {"type": "object", "required": ["a"]})
                self.assertTrue(ok)

    def test_missing_file_fails(self) -> None:
        ok, errs = evaluate.evaluate_file(Path("/nonexistent/x.json"), {"type": "object"})
        self.assertFalse(ok)
        self.assertTrue(any("读取失败" in e for e in errs))


class FindArtifactTests(unittest.TestCase):
    def test_json_preferred_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.assertIsNone(evaluate.find_artifact(base, "report.json"))
            (base / "report.yaml").write_text("a: 1\n", encoding="utf-8")
            self.assertEqual(evaluate.find_artifact(base, "report.json"), base / "report.yaml")
            (base / "report.json").write_text("{}", encoding="utf-8")
            self.assertEqual(evaluate.find_artifact(base, "report.json"), base / "report.json")


if __name__ == "__main__":
    unittest.main()