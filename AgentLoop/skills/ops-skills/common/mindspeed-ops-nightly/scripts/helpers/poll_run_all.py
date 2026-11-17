#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轮询 run_all 进度：每 <interval>s 检查 .phase / 日志 tail / 关键计数, 直到 ALL DONE 或超时。
用法: python poll_run_all.py [DATE] [max_minutes] [interval_seconds]
   DATE    默认今日 YYYYMMDD; 产物目录 $WS/{DATE}/（复测则 *_{DATE}_retest 由调用方拼接）
   max_min 默认 24*60; interval 默认 60
依赖: python3 + paramiko; 连接信息走环境变量 NPC_IP/NPC_USER/NPC_PASS; WS 可用 NFS_WS 覆盖
2026-09-09 双模式: NPC_SUDO=1 时走服务器方式（普通用户 + sudo -S -i）, SUDO_PASS 缺省同 NPC_PASS。
"""
import os
import sys
import io
import time

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
MAX_MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 24 * 60
INTERVAL = int(sys.argv[3]) if len(sys.argv) > 3 else 60

OUT = f"{WS}/{DATE}"

def run(client, cmd, timeout=30):
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
        print("[poll] NPC_IP not set")
        return 2
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=22, username=USER, password=PASS, timeout=20,
                   banner_timeout=30, auth_timeout=30)
    t0 = time.time()
    last_phase = None
    try:
        while time.time() - t0 < MAX_MIN * 60:
            ts = time.strftime("%H:%M:%S")
            try:
                rc, phase = run(client, f"cat {OUT}/.phase 2>/dev/null || echo none")
                phase = phase.strip()
                rc2, tail = run(client, f"tail -4 {WS}/logs/run_all_{DATE}_*.log 2>/dev/null | tail -4", timeout=20)
                rc3, cnt = run(client,
                    f"echo -n 'acc:'; grep -c 'status:' {OUT}/result_atk.log 2>/dev/null || echo 0; "
                    f"echo -n 'perf:'; grep -c 'status:' {OUT}/result_atk_perf.log 2>/dev/null || echo 0; "
                    f"echo -n 'utlog:'; ls -la {OUT}/result_ut.log 2>/dev/null | awk '{{print $5}}' || echo 0",
                    timeout=20)
                if phase != last_phase:
                    print(f"[{ts}] PHASE -> {phase}", flush=True)
                    last_phase = phase
                    print(f"[{ts}] tail: {tail.strip().splitlines()[-2:] if tail.strip() else ''}", flush=True)
                print(f"[{ts}] phase={phase} {cnt.replace(chr(10), ' ')}", flush=True)
                if phase == "summarize" or "ALL DONE" in tail:
                    time.sleep(30)
                    rc4, final = run(client,
                        f"tail -60 {WS}/logs/run_all_{DATE}_*.log 2>/dev/null | tail -60; echo ====; "
                        f"ls {OUT}/report.md {OUT}/report.json 2>&1", timeout=30)
                    print("[FINAL]", final, flush=True)
                    print(f"[poll] DONE in {int(time.time()-t0)}s", flush=True)
                    return 0
            except Exception as e:
                print(f"[{ts}] poll error: {e}", flush=True)
                try:
                    client.close()
                    client = paramiko.SSHClient()
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    client.connect(HOST, port=22, username=USER, password=PASS, timeout=20,
                                   banner_timeout=30, auth_timeout=30)
                    print(f"[{ts}] reconnected", flush=True)
                except Exception as e2:
                    print(f"[{ts}] reconnect failed: {e2}", flush=True)
            time.sleep(INTERVAL)
        print(f"[poll] TIMEOUT after {MAX_MIN}min (phase={last_phase})", flush=True)
        return 1
    finally:
        client.close()

if __name__ == "__main__":
    sys.exit(main())
