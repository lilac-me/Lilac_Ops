#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""skill 包完整性自检（跨平台: 任意 python3 即可, 无需 bash/paramiko）。

用法: python check_skill.py [skill_dir]
   skill_dir 默认 = 本文件所在目录的父目录（即 skill 包根）。

检查项:
  1. 必备文件/目录结构（SKILL.md references/README.md references/config.json scripts/ references/ templates/ scripts/helpers/）
  2. references/config.json 合法 JSON；version 与 SKILL.md 头部版本一致；关键配置字段存在
  3. 所有 .sh：无 CRLF 行尾（Linux 上跑会崩）；无硬编码旧日期/容器 IP（警告级）
  4. 所有 .py：语法编译通过；无 CRLF
  5. owner_map 脱敏/动态加载检查（summarize.py 无内嵌人名 dict，运行时从 NFS owner_map.json 读取）
  6. SKILL.md / references/README.md 中引用的 scripts|templates|references 文件都存在
  7. 垃圾文件提示（__pycache__ / *.pyc）
返回码: 0=全部通过（仅警告可接受）, 1=有必须修复的问题
"""
import ast
import json
import os
import re
import sys

OK, WARN, ERR = "OK", "WARN", "ERR"
results = []


def log(level, msg):
    results.append((level, msg))
    print(f"[{level}] {msg}")


def root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


def find_all(pkg, rel="") -> list:
    """递归列出包内文件（相对路径）"""
    out = []
    base = os.path.join(pkg, rel) if rel else pkg
    for dirpath, dirnames, filenames in os.walk(base):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            out.append(os.path.relpath(full, pkg))
    return out


def main():
    pkg = sys.argv[1] if len(sys.argv) > 1 else root_dir()
    pkg = os.path.abspath(pkg)
    print(f"== check_skill: {pkg} ==")

    # ---------- 1. 必备结构 ----------
    required = [
        "SKILL.md", "references/README.md", "references/config.json",
        "scripts", "references", "templates",
        "scripts/bootstrap.sh", "scripts/run_all.sh",
        "scripts/mode_daily.sh", "scripts/mode_version.sh",
        "scripts/nightly_ut.sh", "scripts/nightly_atk_acc.sh",
        "scripts/nightly_atk_perf.sh", "scripts/ut_remain.sh",
        "scripts/pack_logs.sh", "scripts/run_retest.sh", "scripts/summarize.py",
        "scripts/summarize_retest.py", "scripts/step456_run.sh",
        "references/owner_map.md", "references/environment-setup.md",
        "references/vllm_install_notes.md",
        "templates/condarc", "scripts/helpers/ssh_cmd.py",
    ]
    missing = [r for r in required if not os.path.exists(os.path.join(pkg, r))]
    if missing:
        log(ERR, f"缺失必备文件: {', '.join(missing)}")
    else:
        log(OK, "必备文件/目录齐全")

    # ---------- 2. references/config.json ----------
    cfg_path = os.path.join(pkg, "references", "config.json")
    cfg = None
    if os.path.exists(cfg_path):
        try:
            cfg = json.loads(read(cfg_path))
            log(OK, f"references/config.json 合法（version={cfg.get('version')}）")
        except Exception as e:
            log(ERR, f"references/config.json 解析失败: {e}")
    skill_head = re.search(r">\s*版本:\s*v([0-9.]+)", read(os.path.join(pkg, "SKILL.md")))
    skv = skill_head.group(1) if skill_head else None
    if cfg and skv and cfg.get("version") != skv:
        log(ERR, f"版本不一致: config.json={cfg.get('version')} SKILL.md=v{skv}")
    elif cfg and skv:
        log(OK, f"版本一致（v{skv}）")
    else:
        log(WARN, "无法比对版本（config.json 或 SKILL.md 版本头缺失）")

    # ---------- 3/4. 脚本语法与行尾 ----------
    sh_files, py_files = [], []
    for rel in find_all(pkg):
        if rel.endswith(".sh"):
            sh_files.append(rel)
        elif rel.endswith(".py"):
            py_files.append(rel)
    for rel in sh_files:
        c = read(os.path.join(pkg, rel))
        if "\r" in c:
            log(ERR, f"{rel}: 含 CRLF 行尾（Linux 执行会坏）")
        if re.search(r"2026\d{4}", c):
            log(WARN, f"{rel}: 含硬编码日期（建议参数化/DATE 环境变量）")
    # bash 语法（有 bash 才检查; Windows 上跳过）
    if shutil_which("bash"):
        import subprocess
        bad = []
        for rel in sh_files:
            r = subprocess.run(["bash", "-n", os.path.join(pkg, rel)], capture_output=True)
            if r.returncode != 0:
                bad.append((rel, r.stderr.decode("utf-8", "replace").strip()[:300]))
        if bad:
            for rel, err in bad:
                log(ERR, f"{rel}: bash -n 语法错误: {err}")
        else:
            log(OK, f"bash -n 语法通过（{len(sh_files)} 个 .sh）")
    else:
        log(WARN, "本机无 bash, 跳过 bash -n 语法检查（部署/容器上再验）")
    py_errs = 0
    for rel in py_files:
        c = read(os.path.join(pkg, rel))
        if "\r" in c:
            log(ERR, f"{rel}: 含 CRLF 行尾")
            py_errs += 1
        try:
            ast.parse(c)
        except SyntaxError as e:
            log(ERR, f"{rel}: 语法错误 {e}")
            py_errs += 1
    if py_errs:
        log(WARN, f".py 语法检查完成（{len(py_files)} 个, {py_errs} 个问题已记录）")
    else:
        log(OK, f".py 语法检查通过（{len(py_files)} 个）")

    # ---------- 5. owner_map 脱敏 / 动态加载检查（2026-09-08 改造） ----------
    # 开源合规: summarize.py 不得内嵌负责人人名映射, 从 NFS owner_map 目录动态读取。
    sm = os.path.join(pkg, "scripts", "summarize.py")
    om_ok = False
    if os.path.exists(sm):
        tree = ast.parse(read(sm))
        has_load = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "_load_owner_map"
                       for n in tree.body)
        has_assign = any(isinstance(n, ast.Assign) and any(
                             isinstance(t, ast.Name) and t.id == "OWNER_MAP" for t in n.targets)
                         for n in tree.body)
        inline_dict = any(isinstance(n, ast.Assign) and any(
                              isinstance(t, ast.Name) and t.id == "OWNER_MAP" for t in n.targets)
                          and isinstance(n.value, ast.Dict) and n.value.keys
                          for n in tree.body)
        if has_load and has_assign and not inline_dict:
            log(OK, "owner_map 动态加载（_load_owner_map + OWNER_MAP 赋值, 无内嵌人名 dict）")
            om_ok = True
        else:
            log(ERR, "owner_map 应为动态加载: _load_owner_map() 存在 + OWNER_MAP 赋值 + 无内嵌 Dict")
    omf = os.environ.get("WS", "/mnt/share/ops_nightly_agent") + "/owner_map/owner_map.json"
    if om_ok and os.path.exists(omf):
        try:
            ext = json.load(open(omf, encoding="utf-8"))
            if isinstance(ext, dict) and ext:
                log(OK, f"NFS owner_map.json 可读（{len(ext)} 项）")
            else:
                log(WARN, "NFS owner_map.json 为空或格式异常（fallback owner=未知）")
        except Exception as e:
            log(WARN, f"NFS owner_map.json 解析失败: {e}（fallback owner=未知）")
    elif om_ok:
        log(WARN, "NFS owner_map.json 不存在（本机检查，不影响开源内容；运行时 fallback owner=未知）")

    # ---------- 6. 文档引用文件存在（只检查反引号包裹的路径引用, 排除树状图/散文叙述） ----------
    ref_pat = re.compile(r"`((?:scripts|templates|references)/[A-Za-z0-9_./\-]+?\.(?:sh|py|md))`")
    doc_files = []
    for doc in ("SKILL.md", os.path.join("references", "README.md")):
        dp = os.path.join(pkg, doc)
        if os.path.exists(dp):
            for m in ref_pat.finditer(read(dp)):
                doc_files.append((doc, m.group(1)))
    bad_refs = [(d, r) for d, r in doc_files if not os.path.exists(os.path.join(pkg, r))]
    if bad_refs:
        for d, r in bad_refs:
            log(ERR, f"{d} 引用的文件不存在: {r}")
    else:
        log(OK, f"文档引用文件存在（检查 {len(doc_files)} 处）")

    # ---------- 7. 垃圾文件 ----------
    junk = [rel for rel in find_all(pkg) if "__pycache__" in rel or rel.endswith(".pyc")]
    if junk:
        for j in junk:
            log(WARN, f"垃圾文件（建议删除）: {j}")

    # ---------- 汇总 ----------
    errs = [m for lv, m in results if lv == ERR]
    warns = [m for lv, m in results if lv == WARN]
    print("=" * 50)
    print(f"结果: {'全部通过' if not errs else f'{len(errs)} 个问题'}（警告 {len(warns)} 条）")
    return 1 if errs else 0


def shutil_which(name):
    import shutil
    return shutil.which(name)


if __name__ == "__main__":
    sys.exit(main())
