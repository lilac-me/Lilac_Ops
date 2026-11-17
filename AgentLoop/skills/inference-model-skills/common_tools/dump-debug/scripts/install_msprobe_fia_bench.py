#!/usr/bin/env python3
"""Install and register the npu_fused_infer_attention_score CPU benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import shutil
import sys
from pathlib import Path

OP_NAME = "npu_fused_infer_attention_score"
IMPORT_LINE = f"from msprobe.pytorch.bench_functions.{OP_NAME} import {OP_NAME}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup(path: Path) -> Path:
    target = path.with_name(path.name + ".orig")
    if not target.exists():
        shutil.copy2(path, target)
    return target


def locate_package(explicit: str | None) -> Path:
    if explicit:
        package = Path(explicit).expanduser().resolve()
    else:
        spec = importlib.util.find_spec("msprobe")
        if not spec or not spec.origin:
            raise SystemExit("msprobe is not importable; pass --package /path/to/site-packages/msprobe")
        package = Path(spec.origin).resolve().parent
    if not (package / "pytorch").is_dir():
        raise SystemExit(f"not an msprobe package directory: {package}")
    return package


def install_asset(package: Path, asset: Path) -> Path:
    bench_dir = package / "pytorch" / "bench_functions"
    if not bench_dir.is_dir():
        raise SystemExit(f"bench_functions directory not found: {bench_dir}")
    target = bench_dir / f"{OP_NAME}.py"
    if target.exists():
        backup(target)
    shutil.copy2(asset, target)
    return target


def register_in_yaml(package: Path) -> Path | None:
    config = package / "pytorch" / "bench_functions" / "fusion_operator_config.yaml"
    if not config.exists():
        return None
    text = config.read_text(encoding="utf-8")
    if f"{OP_NAME}:" in text:
        return config
    if "operators:" not in text:
        raise SystemExit(f"operators section missing in {config}")
    entry = (
        f"\n  {OP_NAME}:\n"
        f"    forward: {OP_NAME}\n"
        '    description: "NPU Fused Infer Attention Score (TND, nope + rope)"\n'
    )
    backup(config)
    config.write_text(text.rstrip() + entry, encoding="utf-8")
    return config


def register_in_factory(package: Path) -> Path | None:
    factory = package / "pytorch" / "dump" / "function_factory.py"
    if not factory.exists():
        return None
    text = factory.read_text(encoding="utf-8")
    if "npu_custom_functions = Register()" not in text:
        return None

    changed = False
    if IMPORT_LINE not in text:
        anchor = "from msprobe.pytorch.common.utils import logger"
        if anchor not in text:
            raise SystemExit(f"import anchor not found in {factory}")
        text = text.replace(anchor, f"{IMPORT_LINE}\n{anchor}", 1)
        changed = True

    pattern = re.compile(
        r"(npu_custom_functions\s*=\s*Register\(\)\s*\n"
        r"npu_custom_functions\(\[)(.*?)(\n\s*\]\))",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise SystemExit(f"npu_custom_functions registration block not found in {factory}")
    if OP_NAME not in match.group(2):
        body = match.group(2).rstrip()
        replacement = f"{match.group(1)}{body},\n    {OP_NAME}{match.group(3)}"
        text = text[:match.start()] + replacement + text[match.end():]
        changed = True

    if changed:
        backup(factory)
        factory.write_text(text, encoding="utf-8")
    return factory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", help="Explicit msprobe package directory")
    parser.add_argument(
        "--asset",
        required=True,
        help="Benchmark implementation to install (a .py file implementing "
             "npu_fused_infer_attention_score as a CPU reference)",
    )
    args = parser.parse_args()

    package = locate_package(args.package)
    asset = Path(args.asset).expanduser().resolve()
    if not asset.is_file():
        raise SystemExit(f"asset not found: {asset}")

    target = install_asset(package, asset)
    config = register_in_yaml(package)
    factory = register_in_factory(package)

    print(f"package: {package}")
    print(f"installed: {target}")
    print(f"sha256: {sha256(target)}")
    if config:
        print(f"yaml_registration: {config}")
    if factory:
        print(f"legacy_registration: {factory}")
    print("Restart the Python process before running msprobe acc_check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())