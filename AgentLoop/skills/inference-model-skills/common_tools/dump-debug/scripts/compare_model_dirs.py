#!/usr/bin/env python3
"""Compare two model/checkpoint directories before blaming scripts or versions."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

CRITICAL_SUFFIXES = {
    ".json", ".model", ".tiktoken", ".txt", ".py", ".jinja", ".md",
}
DEFAULT_IGNORES = {".git", "__pycache__", ".cache"}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_snapshot(root, ignores):
    snapshot = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in ignores for part in relative.parts):
            continue
        if path.is_symlink():
            snapshot[str(relative)] = {
                "kind": "symlink",
                "target": os.readlink(path),
            }
        elif path.is_file():
            stat = path.stat()
            snapshot[str(relative)] = {
                "kind": "file",
                "size": stat.st_size,
                "path": path,
            }
        elif path.is_dir():
            snapshot[str(relative)] = {"kind": "dir"}
    return snapshot


def should_hash(entry, mode, small_limit):
    if mode == "none" or entry["kind"] != "file":
        return False
    if mode == "all":
        return True
    suffix = Path(entry["path"].name).suffix.lower()
    return entry["size"] <= small_limit or suffix in CRITICAL_SUFFIXES


def public_entry(entry):
    return {key: value for key, value in entry.items() if key != "path"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--good", required=True, help="Known-good model directory")
    parser.add_argument("--bad", required=True, help="Known-bad model directory")
    parser.add_argument("--hash-mode", choices=("none", "small", "all"), default="small")
    parser.add_argument("--small-limit-mb", type=int, default=64)
    parser.add_argument("--ignore", default=".git,__pycache__,.cache")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    good_root = Path(args.good).expanduser().resolve()
    bad_root = Path(args.bad).expanduser().resolve()
    if not good_root.is_dir() or not bad_root.is_dir():
        print(f"error: model directory missing: {good_root} or {bad_root}", file=sys.stderr)
        return 2

    ignores = {item.strip() for item in args.ignore.split(",") if item.strip()}
    ignores |= DEFAULT_IGNORES
    good = build_snapshot(good_root, ignores)
    bad = build_snapshot(bad_root, ignores)

    only_good = sorted(set(good) - set(bad))
    only_bad = sorted(set(bad) - set(good))
    shared = sorted(set(good) & set(bad))
    manifest_mismatches = []
    for relative in shared:
        if good[relative]["kind"] != bad[relative]["kind"]:
            manifest_mismatches.append(
                {"path": relative, "reason": "kind", "good": good[relative], "bad": bad[relative]}
            )
            continue
        if good[relative]["kind"] == "file" and good[relative]["size"] != bad[relative]["size"]:
            manifest_mismatches.append(
                {"path": relative, "reason": "size", "good_size": good[relative]["size"], "bad_size": bad[relative]["size"]}
            )
        if good[relative]["kind"] == "symlink" and good[relative]["target"] != bad[relative]["target"]:
            manifest_mismatches.append(
                {"path": relative, "reason": "symlink", "good": good[relative]["target"], "bad": bad[relative]["target"]}
            )

    small_limit = args.small_limit_mb * 1024 * 1024
    hashes = {}
    hash_mismatches = []
    for relative in shared:
        if relative in {item["path"] for item in manifest_mismatches}:
            continue
        if not should_hash(good[relative], args.hash_mode, small_limit):
            continue
        good_hash = sha256_file(good[relative]["path"])
        bad_hash = sha256_file(bad[relative]["path"])
        hashes[relative] = {"good": good_hash, "bad": bad_hash}
        if good_hash != bad_hash:
            hash_mismatches.append({"path": relative, "good": good_hash, "bad": bad_hash})

    same_real_path = good_root == bad_root
    identical = not only_good and not only_bad and not manifest_mismatches and not hash_mismatches
    result = {
        "good": str(good_root),
        "bad": str(bad_root),
        "same_real_path": same_real_path,
        "hash_mode": args.hash_mode,
        "files": {"good": len(good), "bad": len(bad)},
        "missing_in_bad": only_good,
        "extra_in_bad": only_bad,
        "manifest_mismatches": manifest_mismatches,
        "hashed_files": len(hashes),
        "hash_mismatches": hash_mismatches,
        "verdict": "identical" if identical else "different",
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"GOOD: {good_root}")
        print(f"BAD : {bad_root}")
        if same_real_path:
            print("warning: both arguments resolve to the same real path; mount/overlay differences cannot be detected here")
        print(f"entries: good={len(good)} bad={len(bad)}")
        print(f"missing_in_bad={len(only_good)} extra_in_bad={len(only_bad)} manifest_mismatches={len(manifest_mismatches)}")
        print(f"hash_mode={args.hash_mode} hashed_files={len(hashes)} hash_mismatches={len(hash_mismatches)}")
        for label, items in (
            ("MISSING_IN_BAD", only_good[:20]),
            ("EXTRA_IN_BAD", only_bad[:20]),
            ("MANIFEST", manifest_mismatches[:20]),
            ("HASH", hash_mismatches[:20]),
        ):
            if items:
                print(f"--- {label}")
                for item in items:
                    print(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, default=str))
        print(f"VERDICT: {result['verdict']}")

    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())