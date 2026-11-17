"""Minimal block-style YAML subset loader for doc-agent config files.

Supports: nested mappings by indentation, block lists ("- item"),
inline lists ([a, b]), quoted and bare scalars, full-line and inline
comments. No flow mappings, anchors, or multi-line strings.

Used to load doc_pipeline.yaml and rules/rules.yaml without third-party
dependencies.
"""

from __future__ import annotations

import re

_QUOTED = re.compile(r"""^(['"])(.*)\1$""", re.S)


def _strip_comment(line: str) -> str:
    """Strip a trailing comment that starts with '#' outside quotes."""
    in_s = in_d = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line.rstrip()


def _parse_scalar(raw: str):
    raw = raw.strip()
    m = _QUOTED.match(raw)
    if m:
        return m.group(2)
    if raw in ("true", "True"):
        return True
    if raw in ("false", "False"):
        return False
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    if raw == "":
        return ""
    return raw


def _parse_inline(raw: str):
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x) for x in inner.split(",") if x.strip() != ""]
    return _parse_scalar(raw)


def load(text: str):
    """Load a YAML subset document. Returns dict or list."""
    lines = []
    for raw in text.splitlines():
        if "\ufeff" in raw:
            raw = raw.replace("\ufeff", "")
        stripped = _strip_comment(raw)
        if stripped.strip() == "":
            continue
        if stripped.strip() == "---":
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        content = stripped.lstrip(" ")
        if "\t" in stripped[:indent]:
            raise ValueError("tabs are not allowed for indentation")
        lines.append((indent, content))

    if not lines:
        return {}
    root_indent = lines[0][0]
    value, index = _parse_block(lines, 0, root_indent)
    return value


def _parse_block(lines, index: int, indent: int):
    if index >= len(lines):
        return None, index
    first_indent, first_content = lines[index]
    if first_indent < indent:
        return None, index
    if first_indent > indent:
        raise ValueError(f"unexpected indent at line: {first_content!r}")
    if first_content.startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_map(lines, index, indent)


def _parse_list(lines, index: int, indent: int):
    items = []
    i = index
    while i < len(lines):
        line_indent, content = lines[i]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"bad indent in list: {content!r}")
        if not content.startswith("- "):
            break
        rest = content[2:].strip()
        if rest == "":
            # nested block under "-"
            if i + 1 < len(lines) and lines[i + 1][0] > indent:
                child, i = _parse_block(lines, i + 1, lines[i + 1][0])
                items.append(child)
            else:
                items.append(None)
                i += 1
            continue
        if ":" in rest:
            key_raw, _, val_raw = rest.partition(":")
            key = _parse_scalar(key_raw)
            if val_raw.strip() == "":
                if i + 1 < len(lines) and lines[i + 1][0] > indent:
                    child, i = _parse_block(lines, i + 1, lines[i + 1][0])
                    items.append({key: child})
                else:
                    items.append({key: None})
                    i += 1
            else:
                items.append({key: _parse_inline(val_raw)})
                i += 1
                # subsequent deeper keys belong to the same list item
                if i < len(lines) and lines[i][0] > indent:
                    child, i = _parse_block(lines, i, lines[i][0])
                    if isinstance(child, dict) and isinstance(items[-1], dict):
                        items[-1].update(child)
        else:
            items.append(_parse_inline(rest))
            i += 1
    return items, i


def _parse_map(lines, index: int, indent: int):
    obj = {}
    i = index
    while i < len(lines):
        line_indent, content = lines[i]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"bad indent in map: {content!r}")
        if content.startswith("- "):
            break
        key_raw, sep, val_raw = content.partition(":")
        if not sep:
            raise ValueError(f"expected 'key: value', got: {content!r}")
        key = _parse_scalar(key_raw)
        if val_raw.strip() == "":
            if i + 1 < len(lines) and lines[i + 1][0] > indent:
                child, i = _parse_block(lines, i + 1, lines[i + 1][0])
                obj[key] = child
            else:
                obj[key] = None
                i += 1
        else:
            obj[key] = _parse_inline(val_raw)
            i += 1
    return obj, i


def dump(obj) -> str:  # pragma: no cover - not needed by pipeline
    raise NotImplementedError
