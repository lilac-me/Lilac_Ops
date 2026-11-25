#!/usr/bin/env python3
"""Discover vLLM/vLLM-Ascend features from the current runtime.

Run this inside the target environment/container whenever possible. The script
uses only the Python standard library and does not require network access.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any


def run_cmd(cmd: list[str], timeout: int) -> dict[str, Any]:
    try:
        p = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "ok": p.returncode == 0,
        }
    except Exception as e:  # noqa: BLE001 - diagnostic script
        return {"cmd": cmd, "ok": False, "error": repr(e), "stdout": "", "stderr": ""}


def import_path(module: str) -> tuple[str | None, str | None]:
    code = (
        "import importlib, pathlib\n"
        f"m=importlib.import_module({module!r})\n"
        "print(pathlib.Path(m.__file__).resolve())\n"
    )
    r = run_cmd([sys.executable, "-c", code], timeout=20)
    if r["ok"] and r["stdout"].strip():
        return r["stdout"].strip(), None
    return None, r.get("stderr") or r.get("error") or "import failed"


def installed_version(distribution: str, module: str) -> str | None:
    """Return the installed distribution version, falling back to module metadata."""
    try:
        return distribution_version(distribution)
    except PackageNotFoundError:
        pass
    except Exception:
        pass

    code = (
        "import importlib\n"
        f"m=importlib.import_module({module!r})\n"
        "print(getattr(m, '__version__', 'unknown'))\n"
    )
    result = run_cmd([sys.executable, "-c", code], timeout=20)
    if result["ok"] and result["stdout"].strip():
        return result["stdout"].strip()
    return None


def extract_help_options(help_text: str) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in help_text.splitlines():
        line = raw.rstrip()
        m = re.match(r"\s{1,}((?:-\w,\s*)?--[A-Za-z0-9][A-Za-z0-9_-]*(?:[^\n]*?))(?:\s{2,}(.+))?$", line)
        if m:
            if current:
                options.append(current)
            opt_part = m.group(1).strip()
            desc = (m.group(2) or "").strip()
            names = re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", opt_part)
            current = {"names": names, "usage": opt_part, "description": desc}
        elif current and line.startswith(" ") and line.strip():
            current["description"] = (current["description"] + " " + line.strip()).strip()
    if current:
        options.append(current)
    return options


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def discover_ascend_envs(root: Path) -> list[dict[str, Any]]:
    envs_file = root / "envs.py"
    text = read_text(envs_file)
    envs: list[dict[str, Any]] = []
    if not text:
        return envs

    # Prefer AST for the env dict when possible.
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.startswith("VLLM_ASCEND_"):
                        snippet = ast.get_source_segment(text, value) or ""
                        default = None
                        m = re.search(r'os\.getenv\(\s*["\']' + re.escape(key.value) + r'["\']\s*,\s*([^)]+)\)', snippet)
                        if m:
                            default = m.group(1).strip()
                        envs.append({"name": key.value, "default_expr": default, "source": str(envs_file)})
    except SyntaxError:
        pass

    seen = {e["name"] for e in envs}
    for name in sorted(set(re.findall(r"VLLM_ASCEND_[A-Z0-9_]+", text)) - seen):
        envs.append({"name": name, "default_expr": None, "source": str(envs_file)})
    return sorted(envs, key=lambda x: x["name"])


def discover_ascend_docs(root: Path) -> dict[str, Any]:
    repo = root.parent
    docs: dict[str, Any] = {"additional_config_keys": [], "mentioned_envs": [], "files_scanned": []}
    candidate_files = [
        repo / "docs/source/user_guide/configuration/additional_config.md",
        repo / "docs/source/user_guide/release_notes.md",
    ]
    for p in candidate_files:
        text = read_text(p)
        if not text:
            continue
        docs["files_scanned"].append(str(p))
        docs["mentioned_envs"].extend(re.findall(r"VLLM_ASCEND_[A-Z0-9_]+", text))
        docs["additional_config_keys"].extend(re.findall(r"`([a-zA-Z_][a-zA-Z0-9_]*(?:_[a-zA-Z0-9]+)*)`", text))
    docs["mentioned_envs"] = sorted(set(docs["mentioned_envs"]))
    docs["additional_config_keys"] = sorted(
        k for k in set(docs["additional_config_keys"]) if k.startswith(("enable_", "weight_", "torchair_", "ascend_"))
    )
    return docs


def grep_source_features(root: Path) -> dict[str, list[str]]:
    patterns = {
        "vllm_ascend_env_names": r"VLLM_ASCEND_[A-Z0-9_]+",
        "additional_config_like": r"\b(?:enable|use|weight|torchair|ascend|kv|cache|offload|pcp|dcp)_[a-zA-Z0-9_]+\b",
        "pcp_dcp_kv_offload": (
            r"\b(?:PCP|DCP|pcp|dcp|"
            r"kv[_-]?cache[_-]?offload|kvcache[_-]?offload|"
            r"kv[_-]?offload|cache[_-]?offload|cpu[_-]?offload|"
            r"swap[_-]?space|swap[_-]?space[_-]?bytes)\b"
        ),
    }
    out: dict[str, set[str]] = {k: set() for k in patterns}
    for path in root.rglob("*.py"):
        text = read_text(path)
        if not text:
            continue
        for key, pat in patterns.items():
            out[key].update(re.findall(pat, text))
    return {k: sorted(v) for k, v in out.items()}


def markdown_report(catalog: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# vLLM/vLLM-Ascend 特性发现报告")
    lines.append("")
    lines.append(f"- Python: `{catalog.get('python')}`")
    lines.append(f"- vLLM: `{catalog.get('vllm_version') or '未知'}`")
    lines.append(f"- vLLM-Ascend: `{catalog.get('vllm_ascend_version') or '未知'}`")
    lines.append(f"- vLLM 路径: `{catalog.get('vllm_path') or '未找到'}`")
    lines.append(f"- vLLM-Ascend 路径: `{catalog.get('vllm_ascend_path') or '未找到'}`")
    lines.append(f"- 参考文档 URL: `{catalog.get('reference_doc_url') or '未指定'}`")
    lines.append("")

    for name in ["vllm serve", "vllm bench serve"]:
        opts = catalog.get("cli_options", {}).get(name, [])
        lines.append(f"## {name} CLI 参数")
        lines.append("")
        lines.append(f"共发现 {len(opts)} 个参数。")
        lines.append("")
        for opt in opts:
            names = ", ".join(f"`{n}`" for n in opt.get("names", []))
            desc = opt.get("description", "")
            lines.append(f"- {names}: {desc}")
        lines.append("")

    envs = catalog.get("vllm_ascend_envs", [])
    lines.append("## vLLM-Ascend 环境变量")
    lines.append("")
    lines.append(f"共发现 {len(envs)} 个 `VLLM_ASCEND_*` 环境变量。")
    lines.append("")
    for e in envs:
        default = e.get("default_expr")
        lines.append(f"- `{e['name']}`" + (f" 默认表达式: `{default}`" if default else ""))
    lines.append("")

    docs = catalog.get("vllm_ascend_docs", {})
    keys = docs.get("additional_config_keys", [])
    lines.append("## additional-config 相关键")
    lines.append("")
    lines.append(f"从文档/源码中发现 {len(keys)} 个疑似 additional-config 键。")
    lines.append("")
    for key in keys:
        lines.append(f"- `{key}`")
    lines.append("")

    source = catalog.get("source_feature_grep", {})
    for section, values in source.items():
        lines.append(f"## 源码 grep: {section}")
        lines.append("")
        for value in values:
            lines.append(f"- `{value}`")
        lines.append("")

    errors = catalog.get("errors", [])
    if errors:
        lines.append("## 发现过程中的错误")
        lines.append("")
        for err in errors:
            lines.append(f"- {err}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--json-out")
    p.add_argument("--markdown-out")
    p.add_argument("--vllm-bin", default="vllm")
    p.add_argument("--ascend-root", help="Optional vllm_ascend package root")
    p.add_argument(
        "--reference-doc-url",
        default="https://docs.vllm.ai/projects/ascend/en/latest/",
        help="Reference documentation URL recorded in the feature report; override when tuning another docs version.",
    )
    args = p.parse_args()

    catalog: dict[str, Any] = {
        "python": sys.executable,
        "reference_doc_url": args.reference_doc_url,
        "cli_options": {},
        "errors": [],
    }

    vllm_file, err = import_path("vllm")
    catalog["vllm_path"] = vllm_file
    if err:
        catalog["errors"].append(f"import vllm: {err.strip()}")

    catalog["vllm_version"] = installed_version("vllm", "vllm")
    if not catalog["vllm_version"]:
        catalog["errors"].append("get vllm version failed")

    for label, cmd in {
        "vllm serve": [args.vllm_bin, "serve", "--help"],
        "vllm bench serve": [args.vllm_bin, "bench", "serve", "--help"],
    }.items():
        r = run_cmd(cmd, args.timeout)
        catalog.setdefault("raw_help_status", {})[label] = {
            "ok": r["ok"],
            "returncode": r.get("returncode"),
            "stderr_tail": (r.get("stderr") or "")[-1000:],
        }
        if r["stdout"]:
            catalog["cli_options"][label] = extract_help_options(r["stdout"])
        else:
            catalog["cli_options"][label] = []
            catalog["errors"].append(f"{label} --help produced no stdout")

    ascend_root: Path | None = Path(args.ascend_root).resolve() if args.ascend_root else None
    if not ascend_root:
        ascend_file, err = import_path("vllm_ascend")
        catalog["vllm_ascend_path"] = ascend_file
        if ascend_file:
            ascend_root = Path(ascend_file).parent
        elif err:
            catalog["errors"].append(f"import vllm_ascend: {err.strip()}")
    else:
        catalog["vllm_ascend_path"] = str(ascend_root)

    catalog["vllm_ascend_version"] = installed_version("vllm-ascend", "vllm_ascend")
    if not catalog["vllm_ascend_version"]:
        catalog["errors"].append("get vLLM-Ascend version failed")

    if ascend_root and ascend_root.exists():
        catalog["vllm_ascend_envs"] = discover_ascend_envs(ascend_root)
        catalog["vllm_ascend_docs"] = discover_ascend_docs(ascend_root)
        catalog["source_feature_grep"] = grep_source_features(ascend_root)
    else:
        catalog["vllm_ascend_envs"] = []
        catalog["vllm_ascend_docs"] = {}
        catalog["source_feature_grep"] = {}

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown_report(catalog), encoding="utf-8")

    if not args.json_out and not args.markdown_out:
        print(markdown_report(catalog))


if __name__ == "__main__":
    main()
