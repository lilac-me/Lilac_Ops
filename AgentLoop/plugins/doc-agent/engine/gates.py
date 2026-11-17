"""Gate predicates for doc-agent state transitions.

Every predicate evaluates against a workspace directory and returns
(ok: bool, detail: str). Gates are AND-combined by the state machine.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONTRACTS = PLUGIN_ROOT / "contracts"

_SCHEMA_CACHE: dict[str, dict] = {}
REVIEWERS_BY_DEPTH = {
    "L0": set(),
    "L1": {"correctness-code-reviewer", "completeness-reviewer"},
    "L2": {
        "correctness-code-reviewer",
        "correctness-semantic-reviewer",
        "completeness-reviewer",
        "clarity-reviewer",
        "accessibility-reviewer",
    },
}


def _load_schema(name: str) -> dict:
    if name not in _SCHEMA_CACHE:
        path = CONTRACTS / f"{name}.json"
        if not path.exists():
            raise ValueError(f"unknown contract schema: {name}")
        _SCHEMA_CACHE[name] = json.loads(path.read_text(encoding="utf-8"))
    return _SCHEMA_CACHE[name]


def _type_ok(value, spec: dict) -> bool:
    t = spec.get("type")
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "array":
        return isinstance(value, list)
    if t == "object":
        return isinstance(value, dict)
    return True


def _resolve_ref(ref: str) -> dict:
    """Resolve the local contract references used by quality_report.json."""
    if not isinstance(ref, str) or not ref.endswith(".json"):
        raise ValueError(f"unsupported schema reference: {ref!r}")
    return _load_schema(pathlib.Path(ref).stem)


def _validate(value, spec: dict, path: str, errors: list[str]) -> None:
    """Validate the JSON-Schema subset used by doc-agent contracts.

    The old implementation only descended one object level and treated array
    items as a type check.  That allowed malformed nested issue/code records
    through the state-machine gates.  This validator intentionally remains
    dependency-free, but now recursively handles objects, arrays, refs and the
    scalar constraints present in the contracts.
    """
    if not isinstance(spec, dict):
        errors.append(f"{path}: invalid schema node")
        return

    if "$ref" in spec:
        try:
            _validate(value, _resolve_ref(spec["$ref"]), path, errors)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
        return

    if "enum" in spec and value not in spec["enum"]:
        errors.append(f"{path}: invalid enum value {value!r}")
    if "const" in spec and value != spec["const"]:
        errors.append(f"{path}: expected constant {spec['const']!r}")

    if not _type_ok(value, spec):
        expected = spec.get("type", "schema")
        errors.append(f"{path}: expected {expected}")
        return

    if isinstance(value, str):
        if "minLength" in spec and len(value) < int(spec["minLength"]):
            errors.append(f"{path}: shorter than minLength {spec['minLength']}")
        if "maxLength" in spec and len(value) > int(spec["maxLength"]):
            errors.append(f"{path}: longer than maxLength {spec['maxLength']}")
        if "pattern" in spec:
            try:
                matched = re.search(spec["pattern"], value)
            except re.error as exc:
                errors.append(f"{path}: invalid pattern ({exc})")
                matched = True
            if not matched:
                errors.append(f"{path}: pattern mismatch {value!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            errors.append(f"{path}: below minimum {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            errors.append(f"{path}: above maximum {spec['maximum']}")

    if isinstance(value, list):
        if "minItems" in spec and len(value) < int(spec["minItems"]):
            errors.append(f"{path}: fewer than minItems {spec['minItems']}")
        if "maxItems" in spec and len(value) > int(spec["maxItems"]):
            errors.append(f"{path}: more than maxItems {spec['maxItems']}")
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            for index, item in enumerate(value):
                _validate(item, item_spec, f"{path}[{index}]", errors)

    if isinstance(value, dict):
        for key in spec.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required field '{key}'")
        props = spec.get("properties", {})
        for key, val in value.items():
            if spec.get("additionalProperties") is False and key not in props:
                errors.append(f"{path}: unexpected field '{key}'")
                continue
            if key in props:
                _validate(val, props[key], f"{path}.{key}", errors)


def _schema_file_for(filename: str) -> str:
    name = pathlib.Path(filename).stem  # e.g. context_report.json -> context_report
    return name


def product_exists(workspace: pathlib.Path, arg: str) -> tuple[bool, str]:
    matches = list(workspace.glob(arg))
    non_empty = [m for m in matches if m.is_file() and m.stat().st_size > 0]
    if non_empty:
        return True, f"{len(non_empty)} file(s) match '{arg}'"
    return False, f"no non-empty file matches '{arg}'"


def schema_valid(workspace: pathlib.Path, arg: str) -> tuple[bool, str]:
    file = workspace / arg
    if not file.is_file() or file.stat().st_size == 0:
        return False, f"{arg}: missing or empty"
    try:
        value = json.loads(file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"{arg}: invalid JSON ({exc})"
    schema_name = _schema_file_for(arg)
    return validate_value(value, schema_name, arg)


def validate_value(value, schema_name: str, path: str = "value") -> tuple[bool, str]:
    """Validate an in-memory value against a Doc-Agent contract."""
    try:
        schema = _load_schema(schema_name)
    except ValueError as exc:
        return False, str(exc)
    errors: list[str] = []
    _validate(value, schema, path, errors)
    if errors:
        return False, "; ".join(errors[:5])
    return True, f"{path}: schema valid"


def file_sha256(path: pathlib.Path) -> str:
    """Return the SHA-256 of a file without normalizing its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_inputs_sha256(workspace: pathlib.Path, depth: str) -> str:
    """Fingerprint every mutable review input consumed by a quality report."""
    names = ["lint_report.json", "review_report.json", "issues.jsonl"]
    if depth != "L0":
        names.append("code_verify.json")
    digest = hashlib.sha256()
    for name in names:
        path = workspace / name
        if name != "issues.jsonl" and not path.is_file():
            raise OSError(f"required review input missing: {name}")
        content = path.read_bytes() if path.is_file() else b""
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def score_pass(workspace: pathlib.Path, _arg: str = "") -> tuple[bool, str]:
    ok, detail = schema_valid(workspace, "quality_report.json")
    if not ok:
        return False, detail
    file = workspace / "quality_report.json"
    try:
        data = json.loads(file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"quality_report.json invalid ({exc})"
    document = _resolve_document(workspace, str(data.get("doc", "")))
    if document is None:
        return False, "quality_report.json target document cannot be located"
    try:
        current_hash = file_sha256(document)
    except OSError as exc:
        return False, f"quality report target unreadable ({exc})"
    if data.get("doc_sha256") != current_hash:
        return False, "quality_report.json is stale for the current document"
    try:
        inputs_hash = review_inputs_sha256(workspace, str(data.get("depth", "")))
    except OSError as exc:
        return False, f"quality report inputs unavailable ({exc})"
    if data.get("inputs_sha256") != inputs_hash:
        return False, "quality_report.json is stale for the current review inputs"
    return data.get("verdict") == "pass", f"verdict={data.get('verdict')!r}"


def lint_report_state(
    workspace: pathlib.Path,
    expected_doc: str,
    expected_depth: str,
) -> tuple[bool, str]:
    """Require a successful lint report bound to the exact document revision."""
    ok, detail = schema_valid(workspace, "lint_report.json")
    if not ok:
        return False, detail
    try:
        data = json.loads((workspace / "lint_report.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"lint_report.json unreadable ({exc})"
    if data.get("status") != "completed":
        errors = data.get("errors") or []
        return False, f"lint did not complete: {errors[:3]}"
    if data.get("errors"):
        return False, "completed lint report must not contain errors"
    if data.get("depth") != expected_depth:
        return False, (
            f"lint depth {data.get('depth')!r} does not match score depth "
            f"{expected_depth!r}"
        )
    lint_document = _resolve_document(workspace, str(data.get("doc", "")))
    target_document = _resolve_document(workspace, expected_doc)
    if lint_document is None or target_document is None:
        return False, "lint or score target document cannot be located"
    if lint_document.resolve() != target_document.resolve():
        return False, "lint_report.json does not belong to the score target document"
    try:
        current_hash = file_sha256(target_document)
    except OSError as exc:
        return False, f"lint target unreadable ({exc})"
    if data.get("doc_sha256") != current_hash:
        return False, "lint_report.json is stale for the current document"
    return True, f"lint completed for {expected_depth} and current document revision"


def lint_complete(workspace: pathlib.Path, _arg: str = "") -> tuple[bool, str]:
    """Transition gate using the target/depth recorded in quality_report.json."""
    ok, detail = schema_valid(workspace, "quality_report.json")
    if not ok:
        return False, detail
    try:
        report = json.loads((workspace / "quality_report.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"quality_report.json unreadable ({exc})"
    return lint_report_state(
        workspace,
        str(report.get("doc", "")),
        str(report.get("depth", "")),
    )


def review_report_state(
    workspace: pathlib.Path,
    expected_doc: str,
    expected_depth: str,
) -> tuple[bool, str]:
    """Require every reviewer mandated by the selected depth to complete."""
    ok, detail = schema_valid(workspace, "review_report.json")
    if not ok:
        return False, detail
    try:
        data = json.loads((workspace / "review_report.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"review_report.json unreadable ({exc})"
    if data.get("status") != "completed":
        return False, f"semantic review did not complete: {(data.get('errors') or [])[:3]}"
    if data.get("errors"):
        return False, "completed review report must not contain errors"
    if data.get("depth") != expected_depth:
        return False, "review_report.json depth does not match score depth"
    expected = REVIEWERS_BY_DEPTH.get(expected_depth)
    required = data.get("required_reviewers", [])
    completed = data.get("completed_reviewers", [])
    if expected is None:
        return False, f"unsupported review depth {expected_depth!r}"
    if len(required) != len(set(required)) or set(required) != expected:
        return False, f"required reviewers do not match depth {expected_depth}"
    if len(completed) != len(set(completed)) or set(completed) != expected:
        return False, f"completed reviewers do not match depth {expected_depth}"
    review_document = _resolve_document(workspace, str(data.get("doc", "")))
    target_document = _resolve_document(workspace, expected_doc)
    if review_document is None or target_document is None:
        return False, "review or score target document cannot be located"
    if review_document.resolve() != target_document.resolve():
        return False, "review_report.json does not belong to the score target document"
    try:
        current_hash = file_sha256(target_document)
    except OSError as exc:
        return False, f"review target unreadable ({exc})"
    if data.get("doc_sha256") != current_hash:
        return False, "review_report.json is stale for the current document"
    return True, f"all reviewers required for {expected_depth} completed"


def review_complete(workspace: pathlib.Path, _arg: str = "") -> tuple[bool, str]:
    """Transition gate using target/depth recorded in quality_report.json."""
    ok, detail = schema_valid(workspace, "quality_report.json")
    if not ok:
        return False, detail
    try:
        report = json.loads((workspace / "quality_report.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"quality_report.json unreadable ({exc})"
    return review_report_state(
        workspace,
        str(report.get("doc", "")),
        str(report.get("depth", "")),
    )


def code_verify_state(
    workspace: pathlib.Path,
    depth: str = "L2",
    expected_doc: str | None = None,
) -> tuple[str, str]:
    """Return ``pass``, ``manual`` or ``fail`` for code verification.

    Presence alone is not sufficient for a PASSED transition: a manual or
    failed T1/T2/T3 result must remain visible to the orchestrator.
    """
    if depth == "L0":
        return "pass", "L0 lint-only flow: code_verify exempt"
    ok, detail = schema_valid(workspace, "code_verify.json")
    if not ok:
        return "fail", f"code_verify.json required at depth {depth}: {detail}"
    try:
        data = json.loads((workspace / "code_verify.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return "fail", f"code_verify.json unreadable: {exc}"
    blocks = data.get("blocks", [])
    statuses = [block.get("status") for block in blocks]
    if any(status == "fail" for status in statuses):
        return "fail", "one or more code blocks failed verification"
    if any(status == "manual" for status in statuses):
        return "manual", "one or more code blocks await manual verification"
    doc = str(data.get("doc", "")).strip()
    target_doc = str(expected_doc or "").strip()
    if expected_doc is not None and not target_doc:
        return "fail", "score target document is empty; cannot bind code verification"
    if not doc:
        return "fail", "code_verify.json doc is empty; cannot match code blocks"

    supplied_document = _resolve_document(workspace, doc)
    if supplied_document is None:
        return "fail", f"cannot locate document for code verification: {doc}"
    if target_doc:
        target_document = _resolve_document(workspace, target_doc)
        if target_document is None:
            return "fail", f"cannot locate score target document: {target_doc}"
        if supplied_document.resolve() != target_document.resolve():
            return "fail", (
                f"code verification document '{doc}' does not match "
                f"score target '{target_doc}'"
            )
        document = target_document
    else:
        # Direct library callers may omit an expected document.  The CLI scorer
        # supplies one from --doc or run.json before reaching this branch.
        document = supplied_document
    try:
        current_hash = file_sha256(document)
    except OSError as exc:
        return "fail", f"code verification target unreadable: {exc}"
    if data.get("doc_sha256") != current_hash:
        return "fail", "code_verify.json is stale for the current document"

    expected_lines = _fenced_code_block_lines(document)
    covered_lines: list[int] = []
    for index, block in enumerate(blocks):
        location = block.get("location", {})
        block_document = _resolve_document(workspace, str(location.get("file", "")))
        if block_document is None or block_document.resolve() != document.resolve():
            return "fail", f"code block {index} location does not target the reviewed document"
        covered_lines.append(location.get("line"))
    if sorted(covered_lines) != expected_lines:
        return "fail", (
            f"code verification locations {sorted(covered_lines)!r} do not match "
            f"fenced code block starts {expected_lines!r}"
        )
    return "pass", f"all {len(statuses)} code block(s) verified"


def _resolve_document(workspace: pathlib.Path, doc: str) -> pathlib.Path | None:
    """Resolve a document name and its draft naming variants within a run."""
    raw_doc = pathlib.Path(doc)
    if raw_doc.is_absolute():
        candidates = [raw_doc]
    else:
        candidates = [
            workspace / raw_doc,
            workspace / f"draft_{raw_doc.name}",
            workspace / f"draft_{raw_doc.stem}.md",
        ]
    return next((path for path in candidates if path.is_file()), None)


def resolve_document(workspace: pathlib.Path, doc: str) -> pathlib.Path | None:
    """Public document resolver shared by state-machine commands."""
    return _resolve_document(workspace, doc)


def _fenced_code_block_lines(document: pathlib.Path) -> list[int]:
    """Return one-based opening-line numbers for Markdown fenced code blocks."""
    in_fence = False
    fence_char = ""
    fence_len = 0
    starts: list[int] = []
    try:
        lines = document.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not in_fence:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence_char = stripped[0]
                fence_len = len(stripped) - len(stripped.lstrip(fence_char))
                in_fence = True
                starts.append(line_number)
            continue
        if stripped and stripped[0] == fence_char:
            run_len = len(stripped) - len(stripped.lstrip(fence_char))
            if run_len >= fence_len:
                in_fence = False
    return starts


def fenced_code_block_lines(document: pathlib.Path) -> list[int]:
    """Public fenced-block locator used by the doc-hash command."""
    return _fenced_code_block_lines(document)


def code_verify_present(workspace: pathlib.Path, _arg: str = "") -> tuple[bool, str]:
    """Require every code block to be verified before a review can pass,
    except for L0 lint-only flows (depth recorded in quality_report.json)."""
    report = workspace / "quality_report.json"
    depth = "L2"
    report_doc = ""
    if report.is_file():
        try:
            report_data = json.loads(report.read_text(encoding="utf-8-sig"))
            depth = report_data.get("depth", "L2")
            report_doc = str(report_data.get("doc", "")).strip()
        except json.JSONDecodeError:
            pass
    run_doc = ""
    try:
        run_data = json.loads((workspace / "run.json").read_text(encoding="utf-8-sig"))
        run_doc = str(run_data.get("doc", "")).strip()
    except (OSError, json.JSONDecodeError):
        pass
    if depth != "L0" and run_doc and report_doc:
        run_document = _resolve_document(workspace, run_doc)
        report_document = _resolve_document(workspace, report_doc)
        if (run_document is None or report_document is None or
                run_document.resolve() != report_document.resolve()):
            return False, (
                f"quality report document '{report_doc}' does not match "
                f"run target '{run_doc}'"
            )
    expected_doc = run_doc or report_doc
    if depth != "L0" and not expected_doc:
        return False, f"score target document is required at depth {depth}"
    state, detail = code_verify_state(
        workspace, depth, expected_doc=expected_doc or None
    )
    return state == "pass", detail


def iter_below_cap(workspace: pathlib.Path, arg: str, ledger=None) -> tuple[bool, str]:
    """arg format: '<state>:<cap>'. Ledger callable provided by the state machine."""
    state, _, cap_raw = arg.partition(":")
    try:
        cap = int(cap_raw)
    except ValueError:
        return False, f"bad cap '{arg}'"
    count = ledger(state) if ledger else 0
    return count < cap, f"iter({state})={count} cap={cap}"


def evaluate(workspace: pathlib.Path, gates: list, ledger=None) -> tuple[bool, str]:
    """Evaluate a list of gate specs. Each is either a one-entry dict
    {op: arg} or a bare scalar for no-arg ops (score_pass, code_verify_present)."""
    details = []
    for gate in gates:
        if isinstance(gate, str):
            op, arg = gate, ""
        else:
            op, arg = next(iter(gate.items()))
        if op == "product_exists":
            ok, detail = product_exists(workspace, arg)
        elif op == "schema_valid":
            ok, detail = schema_valid(workspace, arg)
        elif op == "score_pass":
            ok, detail = score_pass(workspace)
        elif op == "lint_complete":
            ok, detail = lint_complete(workspace)
        elif op == "review_complete":
            ok, detail = review_complete(workspace)
        elif op == "code_verify_present":
            ok, detail = code_verify_present(workspace)
        elif op == "iter_below_cap":
            ok, detail = iter_below_cap(workspace, arg, ledger)
        else:
            return False, f"unknown gate op '{op}'"
        details.append(f"{op}:{arg} -> {'OK' if ok else 'FAIL'} ({detail})")
        if not ok:
            return False, "; ".join(details)
    return True, "; ".join(details)
