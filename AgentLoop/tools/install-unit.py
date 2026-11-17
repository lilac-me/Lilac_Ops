#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MindSpeed-Agent 按需安装器（core + 领域 unit）。

安装架构见 docs/install-on-demand-design.md：安装只装 core（mindspeed-agent，
含 ecosystem-routing + catalog），领域 unit（mindspeed-llm /
inference-model-optimize / ops / doc-agent）在 ecosystem-routing 命中后按需安装。

用法：
    python tools/install-unit.py status                          # 查看各 unit 安装状态
    python tools/install-unit.py dsh <unit>                      # DSH：投影该 unit 技能到 .dsh/skills
    python tools/install-unit.py claude <unit> [--run]           # Claude Code：打印/执行 install_hint
    python tools/install-unit.py codex <unit> [--run]            # Codex：打印/执行 install_hint
    python tools/install-unit.py link <unit>                     # 建 plugins/<unit>/skills junction 视图
                                                                 #   （供 install 脚本在安装前调用）

平台 CLI 不可用时不阻塞——本工具支持纯打印提示（默认不 --run），
便于无对应 CLI 的环境（如 CI）也能查询状态。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from catalog.loader import Catalog, load_yaml
except Exception:  # pragma: no cover - degrade to plain yaml read
    Catalog = None
    load_yaml = None

UNITS_YAML = ROOT / "catalog" / "plugins.yaml"
DSH_SKILLS = ROOT / ".dsh" / "skills"


def _units() -> list:
    if load_yaml is not None:
        return load_yaml(UNITS_YAML).get("plugins", [])
    import yaml  # fallback
    return yaml.safe_load(UNITS_YAML.read_text(encoding="utf-8")).get("plugins", [])


def unit_by_id(pid: str):
    for u in _units():
        if u.get("id") == pid:
            return u
    return None


def find_canonical_skills(root: str) -> list:
    """Return canonical skill dirs under a skills_root, each holding SKILL.md."""
    base = ROOT / root
    out = []
    if not base.is_dir():
        return out
    for sk in sorted(base.rglob("SKILL.md")):
        out.append(sk.parent)
    return out


def dsh_project(unit: dict) -> int:
    """Project the unit's canonical skills as .dsh/skills/<name> junctions/links."""
    name = unit.get("id")
    skills_root = (unit.get("install") or {}).get("skills_root")
    if not skills_root:
        print(f"ERROR: unit {name} 缺 install.skills_root", file=sys.stderr)
        return 1
    DSH_SKILLS.mkdir(parents=True, exist_ok=True)
    count = 0
    for skill_dir in find_canonical_skills(skills_root):
        target = DSH_SKILLS / skill_dir.name
        if target.exists():
            if _is_real_dir(target):
                # 真实目录而非 junction/symlink：跳过避免误删用户内容
                print(f"  SKIP(real dir): {skill_dir.name}")
                continue
            # junction/symlink：可安全替换（删除不影响源目录）
            try:
                if target.is_symlink():
                    target.unlink()
                else:
                    _remove_junction(target)
            except OSError as e:
                print(f"  FAIL remove old link: {skill_dir.name} ({e})", file=sys.stderr)
                continue
        try:
            target.symlink_to(skill_dir, target_is_directory=True)
        except OSError:
            # Windows fallback: junction via cmd
            rc = subprocess.call(
                ["cmd", "/c", "mklink", "/J", str(target), str(skill_dir)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if rc != 0:
                print(f"  FAIL link: {skill_dir.name}", file=sys.stderr)
                continue
        count += 1
    print(f"[DSH] {name}: projected {count} skills into .dsh/skills/")
    print(f"      {name} 技能可用；如当前会话已启动，请新开会话让 DSH 重扫技能目录。")
    return 0


def _is_real_dir(path: Path) -> bool:
    """True when path is a directory but not a symlink/junction.

    Windows junctions are not reported by is_symlink(); detect them via the
    reparse-point attribute (directory junction has FILE_ATTRIBUTE_REPARSE_POINT).
    """
    import ctypes
    FILE_ATTRIBUTE_REPARSE_POINT = 0x400
    if path.is_symlink():
        return False
    try:
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs != -1 and (attrs & FILE_ATTRIBUTE_REPARSE_POINT):
            return False
    except Exception:
        pass
    return path.is_dir()


def _remove_junction(path: Path) -> None:
    """Remove a Windows directory junction without touching its target."""
    rc = subprocess.call(
        ["cmd", "/c", "rmdir", str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc != 0:
        # fallback: recursive delete should only affect the link itself in rmdir
        raise OSError(f"rmdir junction failed rc={rc}")


def hint_cmd(unit: dict, platform: str) -> str:
    hint = (unit.get("install") or {}).get("install_hint") or {}
    return (hint.get(platform) or "").strip()


def _remove_link(path: Path) -> None:
    """Remove a junction/symlink safely; never touches a real directory."""
    if not path.exists():
        return
    if _is_real_dir(path):
        return
    try:
        if path.is_symlink():
            path.unlink()
        else:
            _remove_junction(path)
    except OSError as e:
        print(f"  WARN remove old link: {path.name} ({e})", file=sys.stderr)


def link_plugin_view(unit: dict) -> int:
    """Build plugins/<dir>/skills/<name> junction view for a unit (claude/codex install).

    Mirrors how ops/doc-agent plugin-local skill views are created: each
    canonical skill dir under install.skills_root is linked into the plugin's
    skills/ folder so plugin installers copy real content.
    Plugin directory may differ from the unit id when the checkout renamed the
    directory (e.g. ecosystem-code-repository-npu-migration -> plugins/ecosystem-code-repository-npu-migration);
    install.source_dir records the actual directory when it differs.
    """
    name = unit.get("id")
    skills_root = (unit.get("install") or {}).get("skills_root")
    source_dir = (unit.get("install") or {}).get("source_dir") or f"plugins/{name}"
    if not skills_root:
        print(f"ERROR: unit {name} 缺 install.skills_root", file=sys.stderr)
        return 1
    view = ROOT / source_dir / "skills"
    view.mkdir(parents=True, exist_ok=True)
    count = 0
    failures = 0
    for skill_dir in find_canonical_skills(skills_root):
        target = view / skill_dir.name
        _remove_link(target)
        if target.exists():
            print(f"  FAIL: remove or back up real directory first: {target}", file=sys.stderr)
            failures += 1
            continue
        try:
            target.symlink_to(skill_dir, target_is_directory=True)
        except OSError:
            rc = subprocess.call(
                ["cmd", "/c", "mklink", "/J", str(target), str(skill_dir)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if rc != 0:
                print(f"  FAIL link: {skill_dir.name}", file=sys.stderr)
                failures += 1
                continue
        count += 1
    print(f"[link] {name}: {count} skill views -> {source_dir}/skills/")
    return int(bool(failures) or count == 0)


def platform_install(unit: dict, platform: str, run: bool) -> int:
    name = unit.get("id")
    cmd = hint_cmd(unit, platform)
    if not cmd:
        print(f"ERROR: unit {name} 缺 {platform} install_hint", file=sys.stderr)
        return 1
    print(f"[{platform}] 按需安装 unit: {name}")
    print(f"  命令: {cmd}")
    if not run:
        print("  提示：本工具默认只打印命令；确认后加 --run 执行，或手动执行上述命令。")
        print("  提示：Claude Code / Codex 安装新插件后需【新开会话】方可加载其 skills。")
        return 0
    cli = "claude" if platform == "claude" else "codex"
    if shutil.which(cli) is None:
        print(f"ERROR: 未找到 CLI '{cli}'，无法执行（可手动执行上方命令）", file=sys.stderr)
        return 1
    print(f"  执行中...")
    if name == "mindspeed-scan-issue-agent":
        if link_plugin_view(unit) != 0:
            return 1
        path = ROOT / "plugins" / name / "scripts/package_plugin.py"
        spec = importlib.util.spec_from_file_location("issue_plugin_package", path)
        package = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(package)
        action = "install" if platform == "claude" else "add"
        with package.materialized_install():
            rc = subprocess.call(f"{cli} plugin {action} {name}@MindSpeed-Agent", shell=True)
    else:
        rc = subprocess.call(cmd, shell=True)
    if rc == 0:
        print(f"  {name} 安装成功。请新开会话后再继续任务。")
    else:
        print(f"  {name} 安装失败（exit={rc}），请手动执行上方命令。", file=sys.stderr)
    return rc


def installed_units(platform: str) -> set:
    """Best-effort: which units are currently installed on the platform."""
    if platform == "dsh":
        if not DSH_SKILLS.is_dir():
            return set()
        names = {p.name for p in DSH_SKILLS.iterdir()}
        # map projected skill names back to owning unit via plugins.yaml skills
        owned: dict = {}
        for u in _units():
            for sname in u.get("skills", []) or []:
                owned.setdefault(sname, u.get("id"))
        return {owned[n] for n in names if n in owned}
    cli = "claude" if platform == "claude" else "codex"
    if shutil.which(cli) is None:
        return set()
    try:
        out = subprocess.run([cli, "plugin", "list"], capture_output=True, text=True,
                             timeout=15).stdout
    except Exception:
        return set()
    present = set()
    for u in _units():
        if u.get("id") in out:
            present.add(u.get("id"))
    return present


def cmd_status() -> int:
    print("=== MindSpeed-Agent 按需安装单元状态 ===")
    for u in _units():
        name = u.get("id")
        inst = u.get("install") or {}
        unit = inst.get("unit")
        default = inst.get("default_installed")
        skills = u.get("skills", []) or []
        mark = "已装(core)" if unit == "core" else ("按需" if default is False else "?")
        print(f"- {name:28s} unit={unit:<10} skills={len(skills):>2}  [{mark}]")
    print()
    print("core 默认已安装；on-demand unit 在路由命中后按需安装：")
    print("  DSH:    python tools/install-unit.py dsh <unit>")
    print("  Claude: python tools/install-unit.py claude <unit> [--run]")
    print("  Codex:  python tools/install-unit.py codex <unit> [--run]")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MindSpeed-Agent 按需安装器")
    ap.add_argument("platform", nargs="?", choices=["status", "claude", "codex", "dsh", "link"],
                    help="status/link 或目标平台")
    ap.add_argument("unit", nargs="?", default=None, help="unit id（catalog/plugins.yaml 的插件 id）")
    ap.add_argument("--run", action="store_true", help="claude/codex 模式实际执行安装命令")
    args = ap.parse_args(argv)

    if args.platform == "status" or args.platform is None:
        return cmd_status()

    if not args.unit:
        print("ERROR: 需要指定 unit id（如 mindspeed-llm）", file=sys.stderr)
        return 1
    unit = unit_by_id(args.unit)
    if unit is None:
        known = ", ".join(u.get("id") for u in _units())
        print(f"ERROR: 未知 unit {args.unit!r}。可用：{known}", file=sys.stderr)
        return 1

    if args.platform == "link":
        return link_plugin_view(unit)
    if args.platform == "dsh":
        return dsh_project(unit)
    return platform_install(unit, args.platform, args.run)


if __name__ == "__main__":
    sys.exit(main())
