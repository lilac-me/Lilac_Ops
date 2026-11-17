#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用 SSH 执行助手（agent 侧运维工具, skill 自带）— 容器/服务器双模式（2026-09-09）。

两种提供方式（同一接口, 环境变量切换）:
  模式A 容器方式（默认）:     root 直连
    $env:NPC_IP='<ip>'; $env:NPC_USER='root'; $env:NPC_PASS='<pwd>'
  模式B 服务器方式（普通用户 + sudo 切 root, 如 132 服务器）:
    $env:NPC_IP='<ip>'; $env:NPC_USER='<user>'; $env:NPC_PASS='<pwd>'
    $env:NPC_SUDO='1'; $env:SUDO_PASS='<sudo密码>'   # SUDO_PASS 缺省同 NPC_PASS
    模式B 下命令以 `sudo -S -i bash -c '<cmd>'` 执行; --sftp-put 先传家目录再 sudo cp 到目标。

用法:
  python ssh_cmd.py '<cmd1>' ['<cmd2>' ...]      # 依次在远端执行, $ 前缀打印, [exit:N] 标记
  python ssh_cmd.py --sftp-put <local> <remote>  # 上传单个文件（远端须为绝对路径）
依赖: python3 + paramiko (pip install paramiko)
注意: 密码只经环境变量传递, 不进日志/报告/命令行历史。
"""
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import paramiko

HOST = os.environ.get("NPC_IP", "")
USER = os.environ.get("NPC_USER", "root")
PASS = os.environ.get("NPC_PASS", "")
SUDO = os.environ.get("NPC_SUDO", "0") == "1"
SUDO_PASS = os.environ.get("SUDO_PASS", PASS)


def exec_sudo(client, cmd, timeout=3600):
    """模式B: sudo -S -i 执行; stdin 送 sudo 密码（服务器方式唯一差异点）。"""
    chan = client.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(f"sudo -S -i bash -c {cmd!r}")
    chan.send(SUDO_PASS + "\n")
    chan.shutdown_write()
    out = b""
    err = b""
    while True:
        if chan.recv_ready():
            out += chan.recv(65536)
        if chan.recv_stderr_ready():
            err += chan.recv_stderr(65536)
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
    rc = chan.recv_exit_status()
    return rc, out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")


def main():
    if not HOST:
        print("[ssh_cmd] NPC_IP not set", file=sys.stderr)
        return 2
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=22, username=USER, password=PASS, timeout=20,
                   banner_timeout=30, auth_timeout=30)
    mode = "server(SUDO)" if SUDO else "container(ROOT)"
    print(f"[ssh_cmd] mode={mode} user={USER} host={HOST}", flush=True)
    try:
        args = sys.argv[1:]
        if args and args[0] == "--sftp-put":
            local, remote = args[1], args[2]
            sftp = client.open_sftp()
            if SUDO:  # 普通用户无权限写远端 root 路径 → 先传家目录, 再 sudo cp
                tmp = f"/home/{USER}/{os.path.basename(remote)}"
                sftp.put(local, tmp)
                sftp.close()
                rc, out, err = exec_sudo(client,
                    f"sudo cp {tmp} {remote} && sudo chmod 644 {remote} && sudo rm -f {tmp} && echo SFTP_PUT_OK")
                print(out, end="")
                if err:
                    print("[stderr]", err, file=sys.stderr)
                return rc
            sftp.put(local, remote)
            sftp.close()
            print(f"[sftp-put] {local} -> {remote} OK")
            return 0
        for cmd in args:
            print(f"$ {cmd}", flush=True)
            if SUDO:
                rc, out, err = exec_sudo(client, cmd, timeout=3600)
            else:
                stdin, stdout, stderr = client.exec_command(cmd, timeout=3600, get_pty=False)
                out = stdout.read().decode("utf-8", errors="replace")
                err = stderr.read().decode("utf-8", errors="replace")
                rc = stdout.channel.recv_exit_status()
            if out:
                print(out)
            if err:
                print("[stderr]", err)
            print(f"[exit:{rc}]", flush=True)
            if rc != 0:
                return rc
        return 0
    finally:
        client.close()

if __name__ == "__main__":
    sys.exit(main())
