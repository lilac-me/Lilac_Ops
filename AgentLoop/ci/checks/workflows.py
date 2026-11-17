# CI Guard — L2 状态机工作流一致性检查
# =====================================
# 1. workflows/*.yaml：可加载、initial_state 存在、转移目标已定义；
# 2. YAML ↔ 编译产物一致性（防止改了 YAML 忘记重编译 state-machine-config.js）；
# 3. workflows/*.js：node --check 语法校验。

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from ci.common import BLOCK, WARN, CheckResult

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover
    _yaml = None

TERMINAL_STATES = {"done", "abort"}


def _iter_workflow_yamls(repo: Path):
    for pdir in sorted(repo.glob("plugins/*")):
        if not pdir.is_dir():
            continue
        for yf in sorted((pdir / "workflows").glob("*.yaml")):
            if yf.exists():
                yield yf


def check_workflows(repo: Path) -> list:
    results: list = []
    checked_js_dirs: set = set()

    # ---- 1. YAML 状态机完整性 + 2. YAML↔编译产物一致性 ----
    for yf in _iter_workflow_yamls(repo):
        rel = yf.relative_to(repo).as_posix()
        if _yaml is None:
            results.append(CheckResult("workflow.yaml", WARN, "PyYAML 缺失，跳过状态机 YAML 检查", rel))
            continue
        try:
            sm = _yaml.safe_load(yf.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            results.append(CheckResult("workflow.yaml", BLOCK, f"YAML 解析失败: {e}", rel))
            continue
        if not isinstance(sm, dict):
            results.append(CheckResult("workflow.yaml", BLOCK, "YAML 顶层不是映射", rel))
            continue
        pipeline = sm.get("pipeline", {}) if isinstance(sm.get("pipeline", {}), dict) else {}
        states = sm.get("states", [])
        if not isinstance(states, list) or not states:
            results.append(CheckResult("workflow.states", BLOCK, "states 缺失或为空", rel))
            continue
        state_ids = {s.get("id") for s in states if isinstance(s, dict) and s.get("id")}
        initial = pipeline.get("initial_state")
        if not initial or initial not in state_ids:
            results.append(CheckResult(
                "workflow.initial", BLOCK,
                f"initial_state（{initial!r}）不存在或未定义", rel))
        for s in states:
            if not isinstance(s, dict):
                continue
            sid = s.get("id")
            for t in s.get("exit_transitions", []) or []:
                if not isinstance(t, dict):
                    continue
                goto = t.get("goto")
                if not goto:
                    results.append(CheckResult(
                        "workflow.transition", BLOCK,
                        f"状态 {sid} 存在缺少 goto 的转移", rel))
                elif goto not in state_ids and goto not in TERMINAL_STATES:
                    results.append(CheckResult(
                        "workflow.transition", BLOCK,
                        f"状态 {sid} 的转移目标 {goto!r} 未在 states 中定义", rel))

        compiled = yf.with_name("state-machine-config.js")
        if compiled.exists():
            results += _check_compiled_consistency(repo, yf, compiled)

    # ---- 3. node --check 语法（每个 workflows 目录一次）----
    for pdir in sorted(repo.glob("plugins/*")):
        wdir = pdir / "workflows"
        if not wdir.is_dir() or wdir in checked_js_dirs:
            continue
        checked_js_dirs.add(wdir)
        js_files = sorted(wdir.glob("*.js"))
        if not js_files:
            continue
        node = shutil.which("node")
        if node is None:
            results.append(CheckResult(
                "workflow.js-syntax", WARN,
                "未找到 node，跳过 workflows/*.js 语法检查", wdir.relative_to(repo).as_posix()))
            continue
        for jsf in js_files:
            proc = subprocess.run(
                [node, "--check", str(jsf)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "").strip().splitlines()
                detail = msg[0] if msg else "语法错误"
                results.append(CheckResult("workflow.js-syntax", BLOCK, detail, jsf.relative_to(repo).as_posix()))

    return results


def _check_compiled_consistency(repo: Path, yaml_path: Path, compiled_path: Path) -> list:
    """用官方编译器编译 YAML，与磁盘上的 state-machine-config.js 对比。"""
    results: list = []
    plugin_dir = yaml_path.parent.parent  # workflows/ 的上一级 = 插件根
    rel = compiled_path.relative_to(repo).as_posix()
    try:
        sys.path.insert(0, str(plugin_dir))
        try:
            from engine.backends.claude_code import compile_yaml_to_js
        except BaseException as e:  # 引擎 import 失败（含 PyYAML 缺失的 SystemExit）
            results.append(CheckResult(
                "workflow.compile", WARN,
                f"无法加载编译引擎（{type(e).__name__}: {e}），跳过 YAML↔JS 一致性检查", rel))
            return results
        try:
            generated = compile_yaml_to_js(yaml_path, compiled_path.parent / "__guard_tmp_config.js")
            on_disk = compiled_path.read_text(encoding="utf-8", errors="replace")
        except BaseException as e:
            results.append(CheckResult(
                "workflow.compile", WARN, f"编译过程异常: {type(e).__name__}: {e}", rel))
            return results
        finally:
            # 清理临时文件与 sys.path
            tmp = compiled_path.parent / "__guard_tmp_config.js"
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            try:
                sys.path.remove(str(plugin_dir))
            except ValueError:
                pass

        def _norm(s: str) -> str:
            return s.replace("\r\n", "\n").strip()

        if _norm(generated) != _norm(on_disk):
            results.append(CheckResult(
                "workflow.compile", BLOCK,
                "state-machine-config.js 与 YAML 不一致——YAML 已变更但未重编译，"
                "请运行：python -m engine.backends.claude_code compile "
                f"--yaml {yaml_path.name} --out {compiled_path.name}", rel))
    except BaseException as e:  # pragma: no cover
        results.append(CheckResult(
            "workflow.compile", WARN, f"一致性检查异常: {type(e).__name__}: {e}", rel))
    return results
