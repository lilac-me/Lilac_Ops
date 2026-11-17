#!/usr/bin/env python3
"""Check vLLM/vLLM-Ascend wheel compatibility without installing packages."""

from __future__ import annotations

import argparse
import html
import json
import platform
import re
import shlex
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from typing import Any


PYPI_JSON = "https://pypi.org/pypi/{package}/{version}/json"
TRITON_ASCEND_SIMPLE_INDEX = "https://mirrors.huaweicloud.com/ascend/repos/pypi"
TRITON_ASCEND_FALLBACK_INDEX = "https://triton-ascend.osinfra.cn/pypi/simple"
TORCH_FAMILY_DEPS = {"torch", "torch-npu", "torchvision", "torchaudio"}

try:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover - optional parser, fallback below
    Requirement = None  # type: ignore[assignment]
    default_environment = None  # type: ignore[assignment]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def arch_aliases(machine: str) -> set[str]:
    normalized = machine.lower()
    aliases = {normalized}
    if normalized in {"aarch64", "arm64"}:
        aliases.update({"aarch64", "arm64"})
    if normalized in {"x86_64", "amd64"}:
        aliases.update({"x86_64", "amd64"})
    return aliases


def is_abi3_compatible(tag: str) -> bool:
    match = re.match(r"cp(\d)(\d+)-abi3", tag)
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2))
    return sys.version_info.major == major and sys.version_info.minor >= minor


def filename_matches_python(filename: str, selected_tag: str) -> bool:
    parts = filename.split("-")
    if len(parts) < 5 or not filename.endswith(".whl"):
        return False
    tag = f"{parts[-3]}-{parts[-2]}"
    return parts[-3] == selected_tag or is_abi3_compatible(tag)


def filename_matches_arch(filename: str, aliases: set[str]) -> bool:
    if not filename.endswith(".whl"):
        return False
    platform_tag = filename[:-4].split("-")[-1].lower()
    platform_parts = set(platform_tag.split("."))
    if "any" in platform_parts:
        return True
    return any(alias and alias in platform_tag for alias in aliases)


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def base_version(value: str) -> str:
    return value.split("+", 1)[0].strip()


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def fetch_metadata(package: str, version: str, timeout: int) -> dict[str, Any]:
    url = PYPI_JSON.format(package=package, version=version)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str, timeout: int) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def simple_project_url(index_url: str, package: str) -> str:
    return index_url.rstrip("/") + "/" + normalize_package_name(package) + "/"


def wheel_version_match(filename: str, package: str, version: str) -> bool:
    normalized_prefix = normalize_package_name(package).replace("-", r"[-_]")
    return bool(re.match(rf"^{normalized_prefix}-{re.escape(version)}-", filename, re.I))


def parse_simple_links(page: str, base_url: str) -> list[str]:
    filenames: list[str] = []
    for match in re.finditer(r"""href\s*=\s*['"]([^'"]+)['"]""", page, re.I):
        href = html.unescape(match.group(1))
        path = urlparse(urljoin(base_url, href)).path
        filename = unquote(path.rsplit("/", 1)[-1])
        if filename.endswith(".whl"):
            filenames.append(filename)
    return filenames


def analyze_simple_package(
    package: str,
    version: str,
    index_urls: list[str],
    timeout: int,
    selected_tag: str,
    aliases: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "package": package,
        "version": version,
        "status": "BLOCKED",
        "index_urls": index_urls,
        "compatible_wheels": [],
        "all_wheels": [],
        "errors": [],
    }
    for index_url in index_urls:
        project_url = simple_project_url(index_url, package)
        try:
            page = fetch_text(project_url, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            result["errors"].append(f"{project_url}: {type(exc).__name__}: {exc}")
            continue
        for filename in parse_simple_links(page, project_url):
            if not wheel_version_match(filename, package, version):
                continue
            file_info = {"filename": filename, "index_url": index_url, "project_url": project_url}
            result["all_wheels"].append(file_info)
            if filename_matches_python(filename, selected_tag) and filename_matches_arch(filename, aliases):
                result["compatible_wheels"].append(file_info)

    if result["compatible_wheels"]:
        result["status"] = "PASS"
    else:
        result["error"] = (
            f"no compatible {package}=={version} wheel for selected Python tag {selected_tag} "
            f"and architecture aliases {sorted(aliases)} in approved simple indexes"
        )
    return result


def analyze_package(package: str, version: str, timeout: int, selected_tag: str, aliases: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "package": package,
        "version": version,
        "pypi_url": PYPI_JSON.format(package=package, version=version),
        "status": "BLOCKED",
        "compatible_wheels": [],
        "all_wheels": [],
        "sdists": [],
    }
    try:
        metadata = fetch_metadata(package, version, timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        result["error"] = f"metadata fetch failed: {type(exc).__name__}: {exc}"
        return result

    info = metadata.get("info", {})
    result["requires_python"] = info.get("requires_python")
    result["requires_dist"] = info.get("requires_dist") or []

    for item in metadata.get("urls", []):
        filename = str(item.get("filename") or "")
        packagetype = item.get("packagetype")
        file_info = {
            "filename": filename,
            "packagetype": packagetype,
            "python_version": item.get("python_version"),
            "requires_python": item.get("requires_python"),
            "size": item.get("size"),
        }
        if packagetype == "bdist_wheel":
            result["all_wheels"].append(file_info)
            if filename_matches_python(filename, selected_tag) and filename_matches_arch(filename, aliases):
                result["compatible_wheels"].append(file_info)
        elif packagetype == "sdist":
            result["sdists"].append(file_info)

    if result["compatible_wheels"]:
        result["status"] = "PASS"
    else:
        result["error"] = (
            f"no compatible wheel for selected Python tag {selected_tag} "
            f"and architecture aliases {sorted(aliases)}"
        )
    return result


def active_exact_requirements(package: dict[str, Any]) -> list[dict[str, str]]:
    exacts: list[dict[str, str]] = []
    for raw_req in package.get("requires_dist") or []:
        req_text = str(raw_req)
        if Requirement is not None and default_environment is not None:
            try:
                req = Requirement(req_text)
                env = default_environment()
                env["extra"] = ""
                if req.marker is not None and not req.marker.evaluate(env):
                    continue
                pinned_versions = [spec.version for spec in req.specifier if spec.operator == "=="]
                if pinned_versions:
                    exacts.append(
                        {
                            "package": package["package"],
                            "dependency": normalize_package_name(req.name),
                            "version": pinned_versions[0],
                            "requirement": req_text,
                        }
                    )
                continue
            except Exception:
                pass

        marker_text = req_text.split(";", 1)[1].lower() if ";" in req_text else ""
        if "extra ==" in marker_text or "extra==" in marker_text:
            continue
        match = re.match(r"\s*([A-Za-z0-9_.-]+)\s*==\s*([^,;\s]+)", req_text)
        if match:
            exacts.append(
                {
                    "package": package["package"],
                    "dependency": normalize_package_name(match.group(1)),
                    "version": match.group(2),
                    "requirement": req_text,
                }
            )
    return exacts


def dependency_analysis(packages: list[dict[str, Any]], selected: dict[str, str]) -> dict[str, Any]:
    pins: dict[str, list[dict[str, str]]] = {}
    for package in packages:
        for pin in active_exact_requirements(package):
            pins.setdefault(pin["dependency"], []).append(pin)

    conflicts: list[dict[str, Any]] = []
    for dependency, entries in sorted(pins.items()):
        versions = sorted({base_version(entry["version"]) for entry in entries})
        if len(versions) > 1:
            conflicts.append({"dependency": dependency, "versions": versions, "pins": entries})
            continue
        selected_version = selected.get(dependency)
        if selected_version and versions and base_version(selected_version) != versions[0]:
            conflicts.append(
                {
                    "dependency": dependency,
                    "versions": versions,
                    "selected_version": selected_version,
                    "pins": entries,
                }
            )
    return {"exact_pins": pins, "conflicts": conflicts}


def requested_triton_ascend_versions(analysis: dict[str, Any]) -> list[str]:
    pins = analysis.get("exact_pins", {}).get("triton-ascend", [])
    return sorted({pin["version"] for pin in pins})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--python", default="")
    parser.add_argument("--vllm-version", default="0.22.1")
    parser.add_argument("--vllm-ascend-version", default="0.22.1rc1")
    parser.add_argument("--torch-version", default="")
    parser.add_argument("--torch-npu-version", default="")
    parser.add_argument(
        "--triton-ascend-extra-index-url",
        action="append",
        default=[TRITON_ASCEND_SIMPLE_INDEX, TRITON_ASCEND_FALLBACK_INDEX],
        help="Approved simple index for triton-ascend wheels. May be repeated.",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--allow-torch-family-conflict",
        action="store_true",
        help=(
            "Allow conflicts limited to torch, torch-npu, torchvision, and torchaudio "
            "to pass as warnings so installers can preserve and post-check the selected torch stack."
        ),
    )
    return parser.parse_args()


def build(args: argparse.Namespace) -> dict[str, Any]:
    selected_tag = python_tag()
    machine = platform.machine()
    aliases = arch_aliases(machine)
    selected_python = str(Path(args.python).expanduser().resolve()) if args.python else sys.executable
    packages = [
        analyze_package("vllm", args.vllm_version, args.timeout, selected_tag, aliases),
        analyze_package("vllm-ascend", args.vllm_ascend_version, args.timeout, selected_tag, aliases),
    ]
    selected = {}
    if args.torch_version:
        selected["torch"] = args.torch_version
    if args.torch_npu_version:
        selected["torch-npu"] = args.torch_npu_version
    dep_analysis = dependency_analysis(packages, selected)
    triton_index_urls = unique_preserve_order(args.triton_ascend_extra_index_url)
    extra_index_checks = [
        analyze_simple_package("triton-ascend", version, triton_index_urls, args.timeout, selected_tag, aliases)
        for version in requested_triton_ascend_versions(dep_analysis)
    ]
    package_status_ok = all(item["status"] == "PASS" for item in packages)
    extra_index_ok = all(item["status"] == "PASS" for item in extra_index_checks)
    conflicts = dep_analysis["conflicts"]
    torch_family_conflicts = [
        conflict for conflict in conflicts if conflict.get("dependency") in TORCH_FAMILY_DEPS
    ]
    non_torch_family_conflicts = [
        conflict for conflict in conflicts if conflict.get("dependency") not in TORCH_FAMILY_DEPS
    ]
    dependency_ok = not conflicts or (
        args.allow_torch_family_conflict and not non_torch_family_conflicts
    )
    status = "PASS" if package_status_ok and extra_index_ok and dependency_ok else "BLOCKED"
    blocked = [f"{item['package']}: {item.get('error', 'blocked')}" for item in packages if item["status"] != "PASS"]
    blocked += [
        f"{item['package']}: {item.get('error', 'blocked')}"
        for item in extra_index_checks
        if item["status"] != "PASS"
    ]
    blocked += [
        f"dependency conflict: {conflict}"
        for conflict in conflicts
        if not args.allow_torch_family_conflict or conflict.get("dependency") not in TORCH_FAMILY_DEPS
    ]
    warnings = []
    if args.allow_torch_family_conflict and torch_family_conflicts:
        warnings.append(
            "torch-family dependency pins conflict; install may continue only with post-install "
            "torch/torch_npu import, version, and NPU visibility checks"
        )
    install_hints = []
    for item in extra_index_checks:
        if item["status"] == "PASS":
            install_hints.append(
                f"{shlex.quote(sys.executable)} -m pip install "
                f"\"{item['package']}=={item['version']}\" "
                + " ".join(f"--extra-index-url={url}" for url in item["index_urls"])
            )
    result = {
        "schema_version": "vllm-ascend-wheel-probe-v1",
        "checked_at": utc_now(),
        "status": status,
        "selected_python": selected_python,
        "selected_python_version": sys.version.split()[0],
        "selected_python_tag": selected_tag,
        "machine": machine,
        "arch_aliases": sorted(aliases),
        "probe_status": status,
        "packages": packages,
        "extra_index_checks": extra_index_checks,
        "dependency_analysis": dep_analysis,
        "dependency_conflict_policy": (
            "torch-family-warning" if args.allow_torch_family_conflict else "strict"
        ),
        "torch_family_conflicts": torch_family_conflicts,
        "non_torch_family_conflicts": non_torch_family_conflicts,
        "install_hints": install_hints,
        "non_mutating": True,
        "install_allowed": status == "PASS",
        "install_strategy": (
            "docs-order-install-with-post-install-torch-stack-check"
            if warnings
            else "docs-order-install"
        ),
        "warnings": warnings,
        "errors": blocked,
        "blocked_reason": "; ".join(blocked) if blocked else "",
    }
    return result


def self_test() -> int:
    assert filename_matches_python("demo-1.0.0-cp312-cp312-manylinux2014_aarch64.whl", "cp312")
    assert filename_matches_arch("demo-1.0.0-cp312-cp312-manylinux2014_aarch64.whl", {"aarch64"})
    assert normalize_package_name("vLLM_Ascend") == "vllm-ascend"
    print("SELF_TEST_PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    result = build(args)
    print(json.dumps(result, indent=2 if args.json else None, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
