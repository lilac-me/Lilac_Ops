#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把本 skill 包整体部署到服务器 NFS 共享目录（可迁移部署入口）— 容器/服务器双模式（2026-09-09）。

用法: python deploy_to_nfs.py [skill_dir]
   skill_dir 默认 = 本文件所在目录的父目录（skill 包根）。

环境变量:
   NPC_IP / NPC_USER / NPC_PASS    SSH 凭据（密码只走环境变量, 不进日志/报告/命令行）
   NPC_SUDO=1                      服务器方式: 普通用户登录 + sudo -S -i 切 root（如 132 服务器）
                                   缺省=容器方式: root 直连（如 6 卡容器）
   SUDO_PASS                       sudo 密码（服务器方式, 缺省同 NPC_PASS）
   NFS_WS                          服务器 WS 根（默认 /mnt/share/ops_nightly_agent）

行为:
   1. 整包上传到 $NFS_WS/skill/（SKILL.md scripts/** references/** templates/**,
      剔除 __pycache__/*.pyc; README.md 与 config.json 位于 references/ 下）;
   2. 运行所必需的脚本同步到 $NFS_WS/scripts/（run_all 调用入口: nightly_*.sh run_all.sh run_retest.sh
      ut_remain.sh summarize.py summarize_retest.py）;
   3. 远端 md5sum 与本地比对, 不一致会报 ERR。
依赖: python3 + paramiko（SSH 侧无需任何安装, 脚本落在共享盘即用）。
服务器方式说明: 普通用户无 NFS root 写权限 → 先 sftp 传家目录临时区, 再 sudo cp 同步到目标后校验。
"""
import hashlib
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import paramiko

HOST = os.environ.get("NPC_IP", "")
USER = os.environ.get("NPC_USER", "root")
PASS = os.environ.get("NPC_PASS", "")
SUDO = os.environ.get("NPC_SUDO", "0") == "1"
SUDO_PASS = os.environ.get("SUDO_PASS", PASS)
NFS_WS = os.environ.get("NFS_WS", "/mnt/share/ops_nightly_agent")
TMP = f"/home/{USER}/.deploy_tmp"

# 除 skill/ 全树外, 同步到 $WS/scripts/ 的运行脚本（run_all / run_retest 直接调用这些入口）
RUNTIME_SCRIPTS = [
    "nightly_ut.sh", "nightly_atk_acc.sh", "nightly_atk_perf.sh",
    "run_all.sh", "run_retest.sh", "ut_remain.sh", "pack_logs.sh",
    "summarize.py", "summarize_retest.py",
]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_files(pkg, rel=""):
    """返回 (本地绝对路径, 相对包根的 POSIX 相对路径) 列表, 剔除 pycache/pyc"""
    out = []
    base = os.path.join(pkg, rel) if rel else pkg
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(".pyc"):
                continue
            full = os.path.join(dirpath, fn)
            out.append((full, os.path.relpath(full, pkg).replace("\\", "/")))
    return out


def main():
    pkg = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else pkg_root()
    if not HOST:
        print("[deploy] NPC_IP not set")
        return 2
    mode = "server(SUDO)" if SUDO else "container(ROOT)"
    print(f"[deploy] 本地包: {pkg}")
    print(f"[deploy] mode={mode} 目标: {NFS_WS}/skill/** + {NFS_WS}/scripts/运行入口")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=22, username=USER, password=PASS, timeout=20,
                   banner_timeout=30, auth_timeout=30)
    sftp = client.open_sftp()

    def run(cmd, timeout=120):
        if SUDO:  # 服务器方式: sudo -S -i bash -c（stdin 送密码）
            stdin, stdout, stderr = client.exec_command(f"sudo -S -i bash -c {cmd!r}", timeout=timeout)
            stdin.write(SUDO_PASS + "\n")
            stdin.flush()
        else:
            stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out

    def mkdirs(remote_dir):
        # 仅用于 TMP（家目录临时区, 普通用户可写）与容器模式 NFS 路径:
        # SUDO 模式 NFS 目标不经此函数（2.5 阶段用 sudo mkdir -p）。
        # 统一用 sftp 逐级创建 → SUDO 模式下临时区属普通用户, sftp.put 才可写。
        parts = remote_dir.strip("/").split("/")
        cur = ""
        for p in parts:
            cur += "/" + p
            try:
                sftp.stat(cur)
            except IOError:
                sftp.mkdir(cur)

    def dst(rel):
        """上传目标（SUDO 时先落家目录临时区, 最后统一 sudo cp）"""
        return f"{TMP}/{rel}" if SUDO else f"{NFS_WS}/{rel}"

    try:
        # 清理历史临时区（可能残留 root 属主目录导致 sftp 无写权限; 重建后归普通用户）
        if SUDO:
            run(f"rm -rf {TMP}")
        # ---------- 1. skill 全树 ----------
        files = collect_files(pkg)
        mkdirs(dst("skill"))
        for local, rel in files:
            rp = dst(f"skill/{rel}")
            mkdirs(os.path.dirname(rp))
            sftp.put(local, rp)
        print(f"[deploy] skill 树上传完成（{len(files)} 个文件）")

        # ---------- 2. $WS/scripts 运行入口 ----------
        mkdirs(dst("scripts"))
        scripts_dir = os.path.join(pkg, "scripts")
        runtime = []
        for name in RUNTIME_SCRIPTS:
            lp = os.path.join(scripts_dir, name)
            if os.path.exists(lp):
                sftp.put(lp, dst(f"scripts/{name}"))
                runtime.append(name)
        print(f"[deploy] $WS/scripts 运行入口同步: {', '.join(runtime)}")

        # ---------- 2.5 服务器方式: 临时区同步到 NFS 目标 ----------
        if SUDO:
            rc, out = run(f"mkdir -p {NFS_WS}/skill {NFS_WS}/scripts && "
                          f"cp -r {TMP}/skill/. {NFS_WS}/skill/ && "
                          f"cp -r {TMP}/scripts/. {NFS_WS}/scripts/ && "
                          f"rm -rf {TMP} && echo NFS_SYNC_OK")
            print(out, end="")
            if rc != 0:
                print(f"[ERR] NFS 同步失败: {out}")
                return 1

        # ---------- 3. md5 校验 ----------
        errs = 0
        for local, rel in files:
            rc, out = run(f"md5sum {NFS_WS}/skill/{rel}")
            remote_md5 = out.split()[0] if rc == 0 and out.strip() else "?"
            if remote_md5 != md5(local):
                print(f"[ERR] md5 不一致: {rel} ({md5(local)} vs {remote_md5})")
                errs += 1
        for name in runtime:
            rc, out = run(f"md5sum {NFS_WS}/scripts/{name}")
            remote_md5 = out.split()[0] if rc == 0 and out.strip() else "?"
            if remote_md5 != md5(os.path.join(scripts_dir, name)):
                print(f"[ERR] md5 不一致: scripts/{name}")
                errs += 1
        if errs:
            print(f"[deploy] 完成, 但 {errs} 个文件 md5 不一致")
            return 1
        print(f"[deploy] ALL DONE — 全部 {len(files) + len(runtime)} 个文件 md5 校验一致")
        return 0
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    sys.exit(main())
