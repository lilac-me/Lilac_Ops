#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""容器/服务器失联监控: 长时探测 TCP:22, 恢复后自动 SSH 重连并持续跟踪 run_all 状态直到完成。
用法: python watch_container.py [DATE] [max_minutes]
   DATE    默认今日 YYYYMMDD; 产物目录 $WS/{DATE}/
   max_min 默认 45
依赖: python3 + paramiko; 连接信息走环境变量 NPC_IP/NPC_USER/NPC_PASS; WS 可用 NFS_WS 覆盖
2026-09-09 双模式: NPC_SUDO=1 时走服务器方式（普通用户 + sudo -S -i）, SUDO_PASS 缺省同 NPC_PASS。
"""
import os
import sys
import io
import time
import socket

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import paramiko

HOST = os.environ.get("NPC_IP", "")
USER = os.environ.get("NPC_USER", "root")
PASS = os.environ.get("NPC_PASS", "")
SUDO = os.environ.get("NPC_SUDO", "0") == "1"
SUDO_PASS = os.environ.get("SUDO_PASS", PASS)
WS = os.environ.get("NFS_WS", "/mnt/share/ops_nightly_agent")
DATE = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y%m%d")
MAX_MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 45
OUT = f"{WS}/{DATE}"

def tcp_ok():
    try:
        s = socket.create_connection((HOST, 22), timeout=8)
        s.close()
        return True
    except OSError:
        return False

def run(client, cmd, timeout=40):
    if SUDO:  # 服务器方式: sudo -S -i bash -c 执行（stdin 送密码）
        stdin, stdout, stderr = client.exec_command(f"sudo -S -i bash -c {cmd!r}", timeout=timeout)
        stdin.write(SUDO_PASS + "\n")
        stdin.flush()
    else:
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out

def main():
    if not HOST:
        print("[watch] NPC_IP not set")
        return 2
    t0 = time.time()
    connected = False
    client = None
    last = ""
    while time.time() - t0 < MAX_MIN * 60:
        ts = time.strftime("%H:%M:%S")
        if not connected:
            if tcp_ok():
                print(f"[{ts}] TCP 22 UP, 尝试 SSH...", flush=True)
                try:
                    client = paramiko.SSHClient()
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    client.connect(HOST, port=22, username=USER, password=PASS, timeout=20,
                                   banner_timeout=30, auth_timeout=30)
                    connected = True
                    print(f"[{ts}] SSH 恢复连接", flush=True)
                except Exception as e:
                    print(f"[{ts}] SSH 连接失败: {e}", flush=True)
                    try:
                        client.close()
                    except Exception:
                        pass
                    client = None
            else:
                print(f"[{ts}] TCP 22 DOWN (已失联 {int(time.time()-t0)}s)", flush=True)
            time.sleep(20)
            continue
        try:
            rc, phase = run(client, f"cat {OUT}/.phase 2>/dev/null || echo none")
            phase = phase.strip()
            rc2, tail = run(client, f"ls -t {WS}/logs/run_all_{DATE}_*.log | head -1 | xargs tail -6 2>/dev/null")
            rc3, cnt = run(client,
                f"echo -n 'acc:'; grep -c 'status:' {OUT}/result_atk.log 2>/dev/null || echo 0; "
                f"echo -n 'perf:'; grep -c 'status:' {OUT}/result_atk_perf.log 2>/dev/null || echo 0; "
                f"echo -n 'ut_bytes:'; stat -c%s {OUT}/result_ut.log 2>/dev/null || echo 0; "
                f"echo -n 'run_all_proc:'; ps aux | grep '[r]un_all.sh' | wc -l",
                timeout=30)
            line = f"[{ts}] phase={phase} {cnt.replace(chr(10), ' ')}"
            if line != last:
                print(line, flush=True)
                print(tail.strip()[-3:], flush=True)
                last = line
            if phase == "summarize":
                time.sleep(40)
                rc4, final = run(client, f"tail -50 {WS}/logs/run_all_{DATE}_*.log 2>/dev/null | tail -50; echo ====; ls -la {OUT}/report.md {OUT}/report.json 2>&1", timeout=40)
                print("[FINAL]", final, flush=True)
                print(f"[watch] DONE {int(time.time()-t0)}s", flush=True)
                return 0
        except Exception as e:
            print(f"[{ts}] 读取失败: {e}，尝试重连", flush=True)
            connected = False
            try:
                client.close()
            except Exception:
                pass
            client = None
            time.sleep(15)
            continue
        time.sleep(30)
    print(f"[watch] TIMEOUT {MAX_MIN}min", flush=True)
    return 1

if __name__ == "__main__":
    sys.exit(main())
