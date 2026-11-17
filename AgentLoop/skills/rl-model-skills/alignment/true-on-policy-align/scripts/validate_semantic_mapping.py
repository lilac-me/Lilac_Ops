#!/usr/bin/env python3
"""Validate a provider-neutral semantic mapping used for optional structural triage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


MAPPING_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
SHA256 = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
RELATIONS = {"1:1", "1:N", "N:1"}
ROLES = {"input", "output", "parameter", "aux_state"}
CONFIDENCE = {"high", "medium", "low"}
ORDER_KINDS = {
    "call_index",
    "stack_trace",
    "runtime_sequence",
    "source_control_flow",
    "semantic_dependency",
}
FORBIDDEN_ORDER_BASIS = {
    "json_insertion_order",
    "dict_insertion_order",
    "file_order",
    "mapping_file_order",
}


class MappingError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MappingError(f"cannot read semantic mapping {path}: {exc}") from exc


def require_string(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MappingError(f"{key} must be a non-empty string")
    return value.strip()


def require_string_list(
    parent: dict[str, Any], key: str, *, allow_empty: bool = False
) -> list[str]:
    value = parent.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise MappingError(f"{key} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise MappingError(f"{key} may not be empty")
    return [item.strip() for item in value]


def validate_endpoint(
    endpoint: Any, label: str, seen: set[tuple[str, str, int | None]]
) -> None:
    if not isinstance(endpoint, dict):
        raise MappingError(f"{label} must be an object")
    key = require_string(endpoint, "key")
    selector = endpoint.get("selector", "$")
    if not isinstance(selector, str) or not selector:
        raise MappingError(f"{label}.selector must be a non-empty string")
    call_index = endpoint.get("call_index")
    if call_index is not None and (
        isinstance(call_index, bool)
        or not isinstance(call_index, int)
        or call_index < 0
    ):
        raise MappingError(f"{label}.call_index must be a non-negative integer or null")
    identity = (key, selector, call_index)
    if identity in seen:
        raise MappingError(f"{label} duplicates endpoint {identity}")
    seen.add(identity)

    transforms = endpoint.get("transforms", [])
    if not isinstance(transforms, list) or not all(
        isinstance(item, str) and item.strip() for item in transforms
    ):
        raise MappingError(f"{label}.transforms must be a list of non-empty strings")
    if transforms:
        basis = endpoint.get("semantic_basis")
        if not isinstance(basis, str) or not basis.strip():
            raise MappingError(
                f"{label}.semantic_basis is required when transforms are declared"
            )


def validate_relation(relation: str, train_count: int, infer_count: int, label: str) -> None:
    valid = (
        (relation == "1:1" and train_count == 1 and infer_count == 1)
        or (relation == "1:N" and train_count == 1 and infer_count > 1)
        or (relation == "N:1" and train_count > 1 and infer_count == 1)
    )
    if not valid:
        raise MappingError(
            f"{label}.relation={relation} is inconsistent with "
            f"train={train_count}, infer={infer_count}"
        )


def validate(mapping: Any) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise MappingError("semantic mapping root must be an object")
    if mapping.get("schema_version") != 1:
        raise MappingError("schema_version must be 1")
    if mapping.get("phase") != "prefill":
        raise MappingError("phase must be prefill")

    order_basis = require_string(mapping, "execution_order_basis")
    if order_basis.strip().lower() in FORBIDDEN_ORDER_BASIS:
        raise MappingError(
            "execution_order_basis may not rely on JSON/dict/file insertion order"
        )

    hashes = mapping.get("source_hashes", {})
    if not isinstance(hashes, dict):
        raise MappingError("source_hashes must be an object")
    for name, value in hashes.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(value, str):
            raise MappingError("source_hashes must map non-empty names to SHA256 strings")
        if not SHA256.match(value):
            raise MappingError(f"source_hashes.{name} must be sha256:<64 hex>")

    entries = mapping.get("mappings")
    if not isinstance(entries, list) or not entries:
        raise MappingError("mappings must be a non-empty list")

    ids: set[str] = set()
    orders: dict[int, str] = {}
    relation_counts = {relation: 0 for relation in sorted(RELATIONS)}

    for index, entry in enumerate(entries):
        label = f"mappings[{index}]"
        if not isinstance(entry, dict):
            raise MappingError(f"{label} must be an object")
        mapping_id = require_string(entry, "mapping_id")
        if not MAPPING_ID.match(mapping_id):
            raise MappingError(f"{label}.mapping_id has invalid characters")
        if mapping_id in ids:
            raise MappingError(f"duplicate mapping_id {mapping_id}")
        ids.add(mapping_id)
        require_string(entry, "semantic_node")
        require_string(entry, "semantic_basis")

        relation = require_string(entry, "relation")
        if relation not in RELATIONS:
            raise MappingError(f"{label}.relation must be one of {sorted(RELATIONS)}")
        confidence = require_string(entry, "confidence")
        if confidence not in CONFIDENCE:
            raise MappingError(f"{label}.confidence must be one of {sorted(CONFIDENCE)}")

        order = entry.get("execution_order")
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise MappingError(f"{label}.execution_order must be a non-negative integer")
        if order in orders:
            raise MappingError(
                f"{label}.execution_order duplicates {orders[order]}; "
                "semantic mappings require a total order"
            )
        orders[order] = mapping_id

        order_evidence = entry.get("execution_order_evidence")
        if not isinstance(order_evidence, dict):
            raise MappingError(f"{label}.execution_order_evidence must be an object")
        order_kind = require_string(order_evidence, "kind")
        if order_kind not in ORDER_KINDS:
            raise MappingError(
                f"{label}.execution_order_evidence.kind must be one of "
                f"{sorted(ORDER_KINDS)}"
            )
        require_string(order_evidence, "detail")

        roles = require_string_list(entry, "tensor_roles")
        if len(roles) != len(set(roles)) or any(role not in ROLES for role in roles):
            raise MappingError(
                f"{label}.tensor_roles must contain unique values from {sorted(ROLES)}"
            )

        train = entry.get("train")
        infer = entry.get("infer")
        if not isinstance(train, list) or not train:
            raise MappingError(f"{label}.train must be a non-empty list")
        if not isinstance(infer, list) or not infer:
            raise MappingError(f"{label}.infer must be a non-empty list")
        endpoint_seen: dict[str, set[tuple[str, str, int | None]]] = {
            "train": set(),
            "infer": set(),
        }
        for endpoint_index, endpoint in enumerate(train):
            validate_endpoint(
                endpoint, f"{label}.train[{endpoint_index}]", endpoint_seen["train"]
            )
        for endpoint_index, endpoint in enumerate(infer):
            validate_endpoint(
                endpoint, f"{label}.infer[{endpoint_index}]", endpoint_seen["infer"]
            )
        validate_relation(relation, len(train), len(infer), label)
        relation_counts[relation] += 1

        structural = entry.get("structural_relation")
        if structural is not None:
            if not isinstance(structural, dict):
                raise MappingError(f"{label}.structural_relation must be an object")
            require_string(structural, "kind")
            require_string(structural, "basis")

    return {
        "valid": True,
        "phase": "prefill",
        "mapping_count": len(entries),
        "relation_counts": relation_counts,
        "execution_order_min": min(orders),
        "execution_order_max": max(orders),
        "source_hash_count": len(hashes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()
    mapping_path = args.mapping.resolve()
    try:
        result = validate(load_json(mapping_path))
        result["mapping_file_sha256"] = sha256_file(mapping_path)
    except (MappingError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
