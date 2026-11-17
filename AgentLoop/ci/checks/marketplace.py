# CI Guard — L2 marketplace 一致性检查
# ====================================
# marketplace.json 每个插件条目：source 存在、manifest 存在、
# 版本与对应 plugin.json 一致（防止改版本后忘记同步 marketplace）。

from __future__ import annotations

import json
from pathlib import Path

from ci.common import BLOCK, WARN, CheckResult, read_text


def check_marketplace(repo: Path, plugins: list) -> list:
    results: list = []
    mp_path = repo / ".claude-plugin" / "marketplace.json"
    rel = mp_path.relative_to(repo).as_posix()

    if not mp_path.exists():
        results.append(CheckResult("marketplace.missing", BLOCK, "缺少 .claude-plugin/marketplace.json", rel))
        return results

    try:
        mp = json.loads(read_text(mp_path))
    except json.JSONDecodeError as e:
        results.append(CheckResult("marketplace.parse", BLOCK, f"marketplace.json 解析失败: {e}", rel))
        return results
    if not isinstance(mp, dict):
        results.append(CheckResult("marketplace.parse", BLOCK, "marketplace.json 顶层不是 JSON 对象", rel))
        return results

    if not mp.get("name"):
        results.append(CheckResult("marketplace.fields", WARN, "marketplace 缺少 name", rel))
    if not mp.get("repository"):
        results.append(CheckResult("marketplace.fields", WARN, "marketplace 缺少 repository", rel))

    entries = mp.get("plugins")
    if not isinstance(entries, list) or not entries:
        results.append(CheckResult("marketplace.plugins", BLOCK, "marketplace plugins 列表为空或缺失", rel))
        return results

    # manifest 版本索引：{plugin name: version}
    manifest_versions: dict = {}
    for p in plugins:
        if p["data"] and isinstance(p["data"].get("name"), str):
            manifest_versions[p["data"]["name"]] = p["data"].get("version")

    for entry in entries:
        if not isinstance(entry, dict):
            results.append(CheckResult("marketplace.entry", BLOCK, "plugins 列表包含非对象条目", rel))
            continue
        name = entry.get("name")
        source = entry.get("source")
        label = f"{rel} → {name or '?'}"
        if not isinstance(name, str) or not name.strip():
            results.append(CheckResult("marketplace.entry", BLOCK, "插件条目缺少 name", label))
            continue
        if not isinstance(source, str) or not source.strip():
            results.append(CheckResult("marketplace.entry", BLOCK, f"插件 {name} 缺少 source", label))
            continue

        # source 路径与 manifest 存在性
        src = (repo / source).resolve()
        if not src.is_dir():
            results.append(CheckResult(
                "marketplace.source", BLOCK,
                f"插件 {name} 的 source 路径不存在：{source}", label))
            continue
        manifest = src / ".claude-plugin" / "plugin.json"
        if not manifest.exists():
            results.append(CheckResult(
                "marketplace.source", BLOCK,
                f"插件 {name} 的 source 目录缺少 .claude-plugin/plugin.json", label))
            continue

        # 版本一致性
        mversion = manifest_versions.get(name)
        entry_version = entry.get("version")
        if mversion is None:
            results.append(CheckResult(
                "marketplace.version", WARN,
                f"插件 {name} 未在 plugins 目录中被发现，无法核对版本", label))
        elif entry_version != mversion:
            results.append(CheckResult(
                "marketplace.version", BLOCK,
                f"marketplace 版本（{entry_version}）与 plugin.json（{mversion}）不一致", label))

    return results
