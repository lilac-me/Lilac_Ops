#!/usr/bin/env python3
"""Validate one installed ecosystem-code-repository-npu-migration bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plugin_bundle import BundleIntegrityError, load_bundle_context


PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-root", default=str(PLUGIN_ROOT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        context = load_bundle_context(args.plugin_root)
        payload = {
            "status": "PASS",
            "reason_code": "BUNDLE_INTEGRITY_OK",
            "plugin_root": str(context.plugin_root),
            "plugin_version": context.version,
            "bundle_digest": context.bundle_digest,
        }
        code = 0
    except BundleIntegrityError as exc:
        payload = {"status": "BLOCKED", "reason_code": "BUNDLE_INTEGRITY_MISMATCH", "error": str(exc)}
        code = 2
    print(json.dumps(payload, sort_keys=True) if args.json else payload["reason_code"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
