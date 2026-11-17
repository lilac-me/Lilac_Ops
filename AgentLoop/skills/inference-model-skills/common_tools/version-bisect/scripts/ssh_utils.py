import base64
import os
import paramiko
import time


class SSHClient:
    def __init__(self, host, port=22, username="root", password=None, key_path=None, passphrase=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client = None
        self.passphrase = passphrase
        self.key_path = key_path
        self.sftp = None

    def connect(self):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs = {"hostname": self.host, "port": self.port, "username": self.username, "timeout": 30}
        if self.key_path:
            kp = os.path.expanduser(self.key_path)
            if os.path.exists(kp):
                key = paramiko.RSAKey.from_private_key_file(kp, password=self.passphrase) if self.passphrase else paramiko.RSAKey.from_private_key_file(kp)
                connect_kwargs["pkey"] = key
            elif self.password:
                connect_kwargs["password"] = self.password
        else:
            connect_kwargs["password"] = self.password
        self.client.connect(**connect_kwargs)
        self.sftp = self.client.open_sftp()

    def exec(self, command, timeout=60):
        _, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        return exit_code, stdout.read().decode("utf-8", errors="replace"), stderr.read().decode("utf-8", errors="replace")

    def upload(self, local_path, remote_path):
        self.sftp.put(local_path, remote_path)

    def close(self):
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()


class DockerExecutor:
    def __init__(self, ssh: SSHClient, container_name: str):
        self.ssh = ssh
        self.container = container_name

    def exec(self, command, timeout=60):
        cmd = f'docker exec {self.container} bash -c {self._quote(command)}'
        return self.ssh.exec(cmd, timeout)

    def upload(self, local_path, remote_path_in_container):
        host_path = f"/tmp/docker_upload_{int(time.time())}_{os.path.basename(local_path)}"
        self.ssh.upload(local_path, host_path)
        self.ssh.exec(f"docker cp {host_path} {self.container}:{remote_path_in_container}", timeout=30)

    def _quote(self, cmd):
        encoded = base64.b64encode(cmd.encode()).decode()
        return f"$(echo {encoded} | base64 -d)"


class CondaExecutor:
    def __init__(self, ssh: SSHClient, env_name: str, cann_path: str = ""):
        self.ssh = ssh
        self.env = env_name
        self.cann_path = cann_path

    def exec(self, command, timeout=60):
        activate = f"conda activate {self.env} && " if self.env else ""
        cann_setup = ""
        if self.cann_path:
            cann_setup = f"source {self.cann_path}/ascend-toolkit/set_env.sh && source {self.cann_path}/nnal/atb/set_env.sh && "
        full_cmd = f"bash -c '{activate}{cann_setup}{command}'"
        return self.ssh.exec(full_cmd, timeout)

    def upload(self, local_path, remote_path):
        self.ssh.upload(local_path, remote_path)
