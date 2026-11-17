#!/usr/bin/env python3
"""Prepare an LF-normalized, base64-safe SSH/container script transport command."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any


SENSITIVE = (
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|(?:access|auth|bearer|hf)[_-]?token)\s*=\s*[^\s]+"),
    re.compile(r"https?://[^\s/:@]+:[^\s/@]+@"),
)


class TransportError(ValueError):
    pass


def prepare(
    script: bytes,
    *,
    container: str | None = None,
    workdir: str | None = None,
    ssh_target: str | None = None,
) -> dict[str, Any]:
    if b"\x00" in script:
        raise TransportError("script contains a NUL byte")
    normalized = script.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    text = normalized.decode("utf-8", errors="strict")
    if any(pattern.search(text) for pattern in SENSITIVE):
        raise TransportError("script appears to contain credentials or private-key material")
    if workdir:
        text = f"cd -- {shlex.quote(workdir)} || exit 97\n" + text
        normalized = text.encode("utf-8")
    payload = base64.b64encode(normalized).decode("ascii")
    if container:
        sink = f"docker exec -i {shlex.quote(container)} bash -s --"
    else:
        sink = "bash -s --"
    remote_command = f"printf %s {shlex.quote(payload)} | base64 -d | {sink}"
    result: dict[str, Any] = {
        "schema_version": 1,
        "payload_sha256": "sha256:" + hashlib.sha256(normalized).hexdigest(),
        "payload_bytes": len(normalized),
        "line_endings": "LF",
        "workdir_fail_closed": bool(workdir),
        "remote_command": remote_command,
    }
    if ssh_target:
        if any(character.isspace() for character in ssh_target) or "://" in ssh_target:
            raise TransportError("ssh_target must be a plain user@host-style target")
        result["ssh_argv"] = ["ssh", "--", ssh_target, remote_command]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--container")
    parser.add_argument("--workdir")
    parser.add_argument("--ssh-target")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = prepare(
            args.script.resolve().read_bytes(),
            container=args.container,
            workdir=args.workdir,
            ssh_target=args.ssh_target,
        )
    except (OSError, UnicodeDecodeError, TransportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
