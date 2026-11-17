#!/usr/bin/env python3
"""Discover a task-scoped external Markdown knowledge root without writing files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "external-knowledge-discovery-v1"
MARKDOWN_SUFFIXES = {".md", ".markdown"}


def _display_path(value: Path) -> str:
    try:
        return str(value.expanduser().resolve(strict=False))
    except (OSError, RuntimeError):
        return str(value.expanduser())


def _blocked(root: str, errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "BLOCKED",
        "reason_code": "BLOCKED_EXTERNAL_KNOWLEDGE",
        "root": root,
        "index": None,
        "files": [],
        "markdown_count": 0,
        "errors": errors,
    }


def _resolve_root(value: Path) -> tuple[Path | None, list[str]]:
    expanded = value.expanduser()
    display = _display_path(expanded)
    if not expanded.is_absolute():
        return None, [f"external knowledge root must be absolute: {value}"]

    try:
        root = expanded.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        return None, [f"cannot resolve external knowledge root {display}: {exc}"]
    if not root.exists():
        return None, [f"external knowledge root does not exist: {root}"]
    if not root.is_dir():
        return None, [f"external knowledge root is not a directory: {root}"]

    try:
        with os.scandir(root) as entries:
            next(entries, None)
    except OSError as exc:
        return None, [f"external knowledge root is not readable: {root}: {exc}"]
    return root, []


def _relative_to_root(path: Path, root: Path) -> str | None:
    try:
        resolved = path.resolve(strict=False)
        relative = resolved.relative_to(root)
    except (OSError, RuntimeError):
        return None
    except ValueError:
        # A symlink that escapes the selected root is intentionally ignored.
        return None
    if not resolved.is_file():
        return None
    return relative.as_posix()


def discover(root_value: str | Path) -> dict[str, Any]:
    """Return a deterministic, non-persistent inventory of Markdown files."""

    value = Path(root_value)
    root, errors = _resolve_root(value)
    if root is None:
        return _blocked(_display_path(value), errors)

    relative_files: set[str] = set()

    def onerror(error: OSError) -> None:
        errors.append(f"cannot read external knowledge directory {error.filename or root}: {error}")

    for current, directories, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=onerror,
    ):
        current_path = Path(current)
        # Never recurse through a directory symlink, even when it points back
        # inside the selected root. Regular files are checked after resolving
        # them, so an escaping file symlink is ignored as well.
        directories[:] = sorted(
            name for name in directories if not (current_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            candidate = current_path / filename
            if candidate.suffix.lower() not in MARKDOWN_SUFFIXES:
                continue
            relative = _relative_to_root(candidate, root)
            if relative is not None:
                relative_files.add(relative)

    files = sorted(relative_files)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if files and not errors else "BLOCKED",
        "reason_code": "EXTERNAL_KNOWLEDGE_DISCOVERED" if files and not errors else "BLOCKED_EXTERNAL_KNOWLEDGE",
        "root": str(root),
        "index": "index.md" if "index.md" in relative_files else None,
        "files": files,
        "markdown_count": len(files),
        "errors": errors,
    }
    if not files and not errors:
        result["errors"] = [f"external knowledge root contains no Markdown files: {root}"]
    if result["errors"]:
        result["status"] = "BLOCKED"
        result["reason_code"] = "BLOCKED_EXTERNAL_KNOWLEDGE"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="absolute external knowledge root")
    parser.add_argument(
        "--json",
        action="store_true",
        help="explicitly request the machine-readable output (the default)",
    )
    args = parser.parse_args(argv)
    result = discover(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
