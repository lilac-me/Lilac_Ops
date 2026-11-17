# Step 1: 服务器连接 (Server Connection)

## 目标

建立与目标服务器的SSH连接，验证服务器可达性，检测平台类型（NPU/GPU），检查基础环境。

## 连接策略（免密优先 → 密码 + sudo -S）

连接尝试顺序如下：

1. **密码登录 + sudo -S（优先）**：使用 paramiko（Python SSH 库）通过密码建立连接，然后使用 `echo ${SERVER_PASSWORD} | sudo -S bash -c '...'` 切换到 root 权限执行命令
   - 无需在本地安装 sshpass
   - 无需交互式终端输入密码
   - 可自然处理密码中含 `&`、`$` 等特殊字符的问题
2. **免密登录备用**：paramiko 连接失败时，尝试 `ssh -o BatchMode=yes` 免密连接
3. **备用方案**：如上方案均不可行时，提示用户在交互式终端（Git Bash / WSL）中手动连接

## 输入参数

| 参数 | 说明 |
| ------ | ------ |
| 服务器IP | 从用户文档读取 |
| 用户名 | 从用户文档读取 |
| 密码 | 从用户文档读取 |
| 代理服务器 | 从用户文档读取（如需要） |

## 执行步骤

### 1.1 密码连接（使用 paramiko，优先方式）

使用 Python paramiko 库进行密码登录。paramiko 已预装在本机环境中。

连接方式：

```python
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(SERVER_IP, username=SERVER_USER, password=SERVER_PASSWORD, 
            timeout=30, look_for_keys=False)

# 设置 keepalive 防止长时间命令执行时 SSH 连接超时断开
transport = ssh.get_transport()
transport.set_keepalive(30)

# 执行命令（使用 sudo -S 切换到 root）
cmd = f"echo '{SERVER_PASSWORD}' | sudo -S bash -c 'whoami && hostname'"
stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
print(stdout.read().decode())

ssh.close()
```

> **重要**：密码中的特殊字符（如 `&`、`$`、`"`、`'`）在 shell 传递时可能被解释。paramiko 会将密码作为字符串直接传递给 SSH 服务器，可以正确处理大部分特殊字符。但在 `echo '{PASSWORD}' | sudo -S` 中，如果密码包含单引号 `'`，需要做特殊转义处理。

#### 密码含特殊字符的处理

如果密码包含 `&` 字符（如 `g4BX&b8z6V`），使用 paramiko 的 `exec_command` 方法时需注意：

- 将密码作为 Python 字符串直接传入 SSH 命令字符串中
- 使用单引号包裹密码：`echo '${SERVER_PASSWORD}'`
- 但若密码本身包含单引号，需替换为 `'\''`

#### 从本机 Bash/SSH 使用 sudo -S

如果通过普通 SSH 进行一次性命令执行，也可直接在 shell 中使用：

```bash
echo '${SERVER_PASSWORD}' | sudo -S bash -c 'whoami && hostname'
```

注意：PowerShell 的命令行中单引号/双引号的嵌套规则与 bash 不同，建议使用 paramiko 以避免引号嵌套问题。

### 1.2 免密连接（备用）

如果 paramiko 密码连接失败，可尝试免密连接：

```bash
# 尝试免密连接（BatchMode 不提示密码）
ssh -o BatchMode=yes -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 \
    ${SERVER_USER}@${SERVER_IP} "hostname && whoami"
```

### 1.3 平台检测

```bash
# 检测服务器平台类型
ssh ${SERVER_USER}@${SERVER_IP} '
detect_platform() {
    if command -v npu-smi &>/dev/null; then
        echo "NPU"; return
    fi
    if command -v nvidia-smi &>/dev/null; then
        echo "GPU"; return
    fi
    echo "UNKNOWN"
}; detect_platform'
```

> **说明**：平台检测仅通过硬件管理命令（`npu-smi` / `nvidia-smi`）判断 NPU/GPU 硬件是否存在，**不检测任何 CANN 安装路径**（如 `/usr/local/Ascend`）。CANN 是否已安装的判断在 [step-02-install.md](step-02-install.md) 2.2.1 节中独立进行，且仅检测目标安装路径 `${ASCEND_PATH}`，避免因默认路径存在 CANN 而误判目标路径无需安装。

### 1.4 检查用户权限

```bash
# 检查当前用户是否为 root
ssh ${SERVER_USER}@${SERVER_IP} 'whoami'
if [ "$(ssh ${SERVER_USER}@${SERVER_IP} 'whoami')" != "root" ]; then
    echo "当前用户非 root，后续所有步骤将通过 sudo -S 切换到 root 执行"
    echo "命令格式: echo '\''${SERVER_PASSWORD}'\'' | sudo -S bash -c '\''...'\''"
fi
```

### 1.5 基础环境检查

```bash
# 使用 echo + sudo -S 切换到 root 执行（非 root 用户时）
# 各参数替换为从文档读取的实际值
PASSWORD='${SERVER_PASSWORD}'

# 磁盘空间（检查所有挂载点，CANN 安装需至少 10GB 可用空间）
echo "${PASSWORD}" | sudo -S bash -c "df -h | grep -E '/data|/home'"

# 根分区可用空间（CANN 安装包默认解压到 /tmp 或 /root）
echo "${PASSWORD}" | sudo -S bash -c "df -h / /home /tmp 2>/dev/null"

# 内存
echo "${PASSWORD}" | sudo -S bash -c "free -h"

# Conda
echo "${PASSWORD}" | sudo -S bash -c "which conda || ls /data/miniconda3/bin/conda 2>/dev/null"

# Python版本
echo "${PASSWORD}" | sudo -S bash -c "python3 --version"

# 网络连通性检查（GitHub / PyPI 可达性）
echo "${PASSWORD}" | sudo -S bash -c "
curl -s --connect-timeout 10 https://github.com -o /dev/null && echo 'GitHub: OK' || echo 'GitHub: UNREACHABLE'
curl -s --connect-timeout 10 https://pypi.org -o /dev/null && echo 'PyPI: OK' || echo 'PyPI: UNREACHABLE'
"
```

> **注意**：`sudo -S` 会从标准输入读取密码。如果密码中包含 `$` 符号，建议使用单引号包裹密码字符串，避免 shell 变量展开。
>
> **磁盘空间告警**: 如果根分区 `/` 可用空间 < 10GB，CANN 安装可能因解压空间不足而失败。此时应：
>
> 1. 清理 pip/conda 缓存: `pip cache purge && conda clean --all -y`
> 2. 设置 `TMPDIR` 环境变量到有足够空间的分区（详见 step-02-install 2.2.1）
>
> **网络告警**: 如果 GitHub 不可达，后续 vllm/vllm-ascend 源码克隆将失败。应提示用户配置代理或提供离线安装包。

### 1.6 完整连接示例 (Python paramiko)

以下是一个完整的连接 + 平台检测 + 环境检查的 paramiko 示例：

```python
import paramiko

SERVER_IP = "xxx.xxx.xxx.xxx"
SERVER_USER = "username"
SERVER_PASSWORD = "password"
PROXY = "http://proxy.example.com:8080"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(SERVER_IP, username=SERVER_USER, password=SERVER_PASSWORD, 
            timeout=30, look_for_keys=False)
print("连接成功")

# 设置 keepalive 防止长时间执行（如编译安装）时 SSH 超时
transport = ssh.get_transport()
transport.set_keepalive(30)

def sudo_run(cmd, timeout=60):
    """通过 sudo -S 以 root 权限执行命令

    安全性：密码通过 stdin 写入，不进入命令行，避免远程 ps 泄露。
    """
    safe_cmd = cmd.replace("'", "'\\''")
    full_cmd = f"sudo -S -p '' bash -c '{safe_cmd}'"
    stdin, stdout, stderr = ssh.exec_command(full_cmd, timeout=timeout)
    stdin.write(SERVER_PASSWORD + "\n")
    stdin.flush()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out, err

# 0) 磁盘空间预检
out, _ = sudo_run("df -h / /home 2>/dev/null")
print(f"磁盘:\n{out}")

# 1) 平台检测
out, _ = sudo_run(
    "if command -v nvidia-smi &>/dev/null; then echo GPU; "
    "elif command -v npu-smi &>/dev/null; then echo NPU; "
    "else echo UNKNOWN; fi"
)
print(f"平台: {out.strip()}")

# 2) 检查用户
out, _ = sudo_run("whoami")
print(f"当前用户: {out.strip()}")

# 3) 基础环境检查
out, _ = sudo_run("df -h /home")
print(out.strip())

out, _ = sudo_run("free -h")
print(out.strip())

ssh.close()
```

#### 使用 paramiko 安装 vllm 的完整流程

```python
import paramiko

SERVER_IP = "xxx.xxx.xxx.xxx"
SERVER_USER = "username"
SERVER_PASSWORD = "password"
PROXY = "http://proxy.example.com:8080"
CONDA_PATH = "/data/anaconda3"
CONDA_ENV = "my_env"
VLLM_VERSION = "0.20.2"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(SERVER_IP, username=SERVER_USER, password=SERVER_PASSWORD, 
            timeout=30, look_for_keys=False)

def sudo_run(cmd, timeout=300):
    full_cmd = f"echo '{SERVER_PASSWORD}' | sudo -S bash -c '{cmd}'"
    stdin, stdout, stderr = ssh.exec_command(full_cmd, timeout=timeout)
    return stdout.read().decode("utf-8", errors="replace")

# 创建 Conda 环境
sudo_run(
    f"export http_proxy={PROXY} && export https_proxy={PROXY} && "
    f"{CONDA_PATH}/bin/conda create -n {CONDA_ENV} python=3.11 -y",
    timeout=600
)

# 安装 vllm
sudo_run(
    f"export http_proxy={PROXY} && export https_proxy={PROXY} && "
    f"source {CONDA_PATH}/etc/profile.d/conda.sh && conda activate {CONDA_ENV} && "
    f"pip install vllm=={VLLM_VERSION}",
    timeout=900
)

ssh.close()
```

> **提示**：以上 Python 代码可直接保存为 `.py` 文件并在任何安装了 paramiko 的环境中运行，无需手动输入 SSH 命令。

## 产出

| 产出 | 说明 |
| ------ | ------ |
| SSH连接状态 | connected / failed |
| 平台类型 | NPU / GPU / UNKNOWN |
| 主机名 | 确认连接正确服务器 |
| 磁盘/内存 | 判断是否满足安装条件 |

## 失败处理

| 错误 | 原因 | 处理 |
| ------ | ------ | ------ |
| Connection timeout | 网络不通 | 检查网络、防火墙 |
| Permission denied (publickey) | 免密未配置 | 优先使用 paramiko 密码连接方案（见 1.1 节） |
| Permission denied (password) | 密码错误 | 确认密码 |
| sudo: a password is required | 用户无 sudo 权限 | 提示用户配置 sudo 免密，或更换工作目录到有写权限的路径 |
| Platform UNKNOWN | 检测失败 | 提示用户手动指定平台 |
| paramiko 未安装 | 本地环境缺失 | 执行 `pip install paramiko` 安装 |

## 下一步

平台检测完成后，自动进入 [Step 2: 环境安装](step-02-install.md)
