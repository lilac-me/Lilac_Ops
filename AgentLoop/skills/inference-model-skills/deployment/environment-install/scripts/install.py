"""Environment install - SSH连接与指令执行工具

使用 paramiko 进行密码登录 + sudo -S 切换 root 权限执行命令。
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import paramiko
except ImportError:
    logger.error("paramiko 未安装，请执行: pip install paramiko")
    paramiko = None


def connect(
    host: str,
    user: str,
    password: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Step 1: 连接服务器并检测平台

    连接策略：
    1. 使用 paramiko 密码登录
    2. 通过 echo + sudo -S 切换到 root 权限执行命令
    """
    logger.info("=" * 60)
    logger.info("Step 1: 服务器连接")
    logger.info("=" * 60)

    result = {
        "status": "pending",
        "platform": None,
        "ssh_client": None,
        "is_root": False,
        "errors": [],
        "hostname": None,
    }

    if paramiko is None:
        result["status"] = "failed"
        result["errors"].append("paramiko 未安装，请执行: pip install paramiko")
        return result

    try:
        logger.info("正在连接 %s@%s ...", user, host)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            host,
            username=user,
            password=password,
            timeout=timeout,
            look_for_keys=False,
        )

        out, _ = _sudo_run(ssh, password, "hostname", timeout=10)
        hostname = out.strip()
        result["hostname"] = hostname
        logger.info("连接成功: %s", hostname)

        platform_cmd = (
            "if command -v nvidia-smi &>/dev/null; then echo GPU; "
            "elif command -v npu-smi &>/dev/null; then echo NPU; "
            "else echo UNKNOWN; fi"
        )
        out, _ = _sudo_run(ssh, password, platform_cmd, timeout=10)
        platform = out.strip()
        result["platform"] = platform
        logger.info("平台类型: %s", platform)

        out, _ = _sudo_run(ssh, password, "whoami", timeout=10)
        current_user = out.strip()
        result["is_root"] = current_user == "root"
        logger.info("当前用户: %s", current_user)

        if not result["is_root"]:
            logger.info("非 root 用户，将通过 sudo -S 切换到 root 执行后续命令")

        result["status"] = "success"
        result["ssh_client"] = ssh
        return result

    except paramiko.AuthenticationException:
        result["status"] = "failed"
        result["errors"].append("SSH 认证失败，请检查用户名和密码")
        return result
    except paramiko.SSHException as e:
        result["status"] = "failed"
        result["errors"].append("SSH 连接异常: %s" % str(e))
        return result
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append("连接失败: %s" % str(e))
        return result


def _sudo_run(
    ssh: paramiko.SSHClient,
    password: str,
    cmd: str,
    timeout: int = 60,
) -> Tuple[str, str]:
    """
    通过 sudo -S 以 root 权限在远程服务器执行命令。

    安全性：密码通过 SSH channel 的 stdin 直接写入 sudo -S，
    不出现在命令行中，避免远程进程列表 (ps aux) 泄露密码。
    """
    # 用 -p '' 抑制 sudo 的密码提示符，避免提示符混入 stderr
    safe_cmd = cmd.replace("'", "'\\''")
    full_cmd = "sudo -S -p '' bash -c '%s'" % safe_cmd
    stdin, stdout, stderr = ssh.exec_command(full_cmd, timeout=timeout, get_pty=False)
    # 通过 channel stdin 传入密码，密码不会进入命令行参数
    stdin.write(password + "\n")
    stdin.flush()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out, err


def run(
    ssh: paramiko.SSHClient,
    password: str,
    cmd: str,
    timeout: int = 60,
    ignore_errors: bool = False,
) -> Dict[str, Any]:
    """在远程服务器上以 root 权限执行命令"""
    try:
        out, err = _sudo_run(ssh, password, cmd, timeout=timeout)
        clean_out = "\n".join(
            l for l in out.split("\n")
            if "[sudo]" not in l and "password" not in l.lower()
        )
        result = {
            "stdout": clean_out.strip(),
            "stderr": err.strip(),
            "error": None,
        }
        if "sudo: a password is required" in err:
            msg = "用户没有 sudo 权限，请配置 sudo 免密权限"
            result["error"] = msg
            if not ignore_errors:
                logger.error(msg)
        return result
    except Exception as e:
        msg = str(e)
        if not ignore_errors:
            logger.error("命令执行失败: %s", msg)
        return {"stdout": "", "stderr": "", "error": msg}


def check_env(
    ssh: paramiko.SSHClient,
    password: str,
) -> Dict[str, Any]:
    """检查远程服务器基础环境"""
    logger.info("检查服务器环境...")
    env_info = {}
    out, _ = _sudo_run(ssh, password, "df -h / /home /data 2>/dev/null | tail -5", timeout=10)
    env_info["disk"] = out.strip()
    out, _ = _sudo_run(ssh, password, "free -h", timeout=10)
    env_info["memory"] = out.strip()
    out, _ = _sudo_run(
        ssh, password,
        "which conda 2>/dev/null || ls /data/miniconda3/bin/conda 2>/dev/null "
        "|| ls /data/anaconda3/bin/conda 2>/dev/null || echo 'conda not found'",
        timeout=10,
    )
    env_info["conda_path"] = out.strip()
    out, _ = _sudo_run(ssh, password, "python3 --version 2>/dev/null", timeout=10)
    env_info["python_version"] = out.strip()
    return env_info


def install(
    ssh: paramiko.SSHClient,
    password: str,
    workspace: str,
    conda_env_name: str,
    conda_path: str = "",
    python_version: str = "3.11",
    platform: str = "GPU",
    proxy: Optional[str] = None,
    vllm_version: Optional[str] = None,
    torch_version: Optional[str] = None,
    vllm_ascend_version: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Step 2: 环境安装"""
    logger.info("=" * 60)
    logger.info("Step 2: 环境安装")
    logger.info("=" * 60)

    result = {"status": "pending", "steps": [], "errors": []}

    def run_cmd(cmd, timeout=300, step_name=""):
        logger.info("[%s] 执行中...", step_name)
        exec_result = run(ssh, password, cmd, timeout=timeout)
        if exec_result.get("error"):
            result["status"] = "failed"
            result["errors"].append("[%s] %s" % (step_name, exec_result["error"]))
            result["steps"].append({"step": step_name, "status": "failed", "error": exec_result["error"]})
            return False
        result["steps"].append({"step": step_name, "status": "success"})
        return True

    try:
        proxy_prefix = ""
        if proxy:
            proxy_prefix = "export http_proxy=%s && export https_proxy=%s && " % (proxy, proxy)

        if not conda_path:
            out, _ = _sudo_run(
                ssh, password,
                "dirname $(which conda 2>/dev/null || ls /data/miniconda3/bin/conda 2>/dev/null "
                "|| ls /data/anaconda3/bin/conda 2>/dev/null)",
                timeout=10,
            )
            conda_bin_dir = out.strip()
            if conda_bin_dir:
                conda_path = conda_bin_dir.replace("/bin", "")
            else:
                result["status"] = "failed"
                result["errors"].append("未找到 Conda 安装路径")
                return result

        conda_sh = conda_path + "/etc/profile.d/conda.sh"

        # 2.1 工作目录
        if not run_cmd("mkdir -p %s/logs" % workspace, timeout=30, step_name="2.1"):
            return result

        # 2.2 Conda 环境
        check_cmd = "%s %s/bin/conda env list | grep -q '^%s ' && echo EXISTS || echo NOT_EXISTS" % (
            proxy_prefix, conda_path, conda_env_name
        )
        out, _ = _sudo_run(ssh, password, check_cmd, timeout=30)
        if "NOT_EXISTS" in out:
            create_cmd = "%s %s/bin/conda create -n %s python=%s -y" % (
                proxy_prefix, conda_path, conda_env_name, python_version
            )
            if not run_cmd(create_cmd, timeout=600, step_name="2.2"):
                return result
        else:
            result["steps"].append({"step": "2.2", "status": "skipped", "note": "Conda 环境已存在"})

        # 2.3 安装框架
        install_cmds = [proxy_prefix] if proxy_prefix else []
        install_cmds.append("source %s" % conda_sh)
        install_cmds.append("conda activate %s" % conda_env_name)

        if platform == "NPU":
            if torch_version:
                install_cmds.append("pip install torch==%s" % torch_version)
            if vllm_ascend_version:
                install_cmds.append("pip install vllm-ascend==%s" % vllm_ascend_version)
            if vllm_version:
                install_cmds.append("pip install vllm==%s" % vllm_version)
            else:
                install_cmds.append("pip install vllm")
                if not vllm_ascend_version:
                    install_cmds.append("pip install vllm-ascend")
        elif platform == "GPU":
            if vllm_version:
                install_cmds.append("pip install vllm==%s" % vllm_version)
            else:
                install_cmds.append("pip install vllm")

        full_cmd = " && ".join(install_cmds)
        if not run_cmd(full_cmd, timeout=1800, step_name="2.3"):
            return result

        # 2.4 验证
        verify_cmd = ("%s source %s && conda activate %s && "
                      "python --version && pip list 2>/dev/null | grep -iE 'torch|vllm'") % (
            proxy_prefix, conda_sh, conda_env_name
        )
        out, _ = _sudo_run(ssh, password, verify_cmd, timeout=60)
        result["steps"].append({"step": "2.4", "status": "success", "output": out.strip()})

        result["status"] = "success"
        return result

    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="深度学习环境自动化安装")
    parser.add_argument("--action", required=True, choices=["connect", "install", "full"])
    parser.add_argument("--local-script", help="本地部署脚本路径")
    parser.add_argument("--workspace", default="", help="远程工作目录")
    parser.add_argument("--conda-env", default="", help="Conda环境名")
    parser.add_argument("--python-version", default="3.11")
    parser.add_argument("--platform", default="", help="NPU/GPU")
    parser.add_argument("--vllm-version")

    args = parser.parse_args()

    host = os.environ.get("REMOTE_SERVER_IP", "")
    user = os.environ.get("REMOTE_SERVER_USER", "root")
    password = os.environ.get("REMOTE_SERVER_PASSWORD", "")

    if args.action in ("connect", "full"):
        if not host or not password:
            print("[ERROR] 请设置 REMOTE_SERVER_IP 和 REMOTE_SERVER_PASSWORD 环境变量")
            sys.exit(1)
        conn_result = connect(host, user, password)
        print("[%s] 连接结果: platform=%s" % (conn_result["status"].upper(), conn_result.get("platform")))
        if conn_result["status"] != "success":
            sys.exit(1)
        platform = conn_result["platform"]
    else:
        platform = args.platform or "NPU"
        conn_result = {}

    if args.action in ("install", "full"):
        ssh = conn_result.get("ssh_client")
        if not ssh:
            print("[ERROR] 需要有效的SSH连接")
            sys.exit(1)
        inst_result = install(
            ssh=ssh,
            password=password,
            workspace=args.workspace,
            conda_env_name=args.conda_env,
            python_version=args.python_version,
            platform=platform,
            vllm_version=args.vllm_version,
        )
        print("[%s] 安装结果" % inst_result["status"].upper())
        for step in inst_result.get("steps", []):
            print("  Step %s: %s" % (step["step"], step["status"]))
        if inst_result["status"] != "success":
            sys.exit(1)

    print("完成!")


if __name__ == "__main__":
    main()
