VERSION = "1.0.0"


class SSHClient:
    """基础SSH连接与文件上传"""

    def __init__(self, host, port=22, user="root", password=None, key_filename=None):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.key_filename = key_filename
        self._client = None
        self._sftp = None

    def connect(self):
        import paramiko
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = dict(hostname=self.host, port=self.port, username=self.user, look_for_keys=False)
        if self.password:
            kwargs["password"] = self.password
        if self.key_filename:
            kwargs["key_filename"] = self.key_filename
        self._client.connect(**kwargs)
        self._sftp = self._client.open_sftp()
        return self

    def exec(self, command, timeout=120):
        stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        return exit_code, out, err

    def exec_ok(self, command, timeout=120):
        rc, out, err = self.exec(command, timeout=timeout)
        if rc != 0:
            msg = f"Command failed (rc={rc}): {command[:200]}"
            if err:
                msg += f"\nSTDERR: {err[:500]}"
            if out:
                msg += f"\nSTDOUT: {out[:500]}"
            raise RuntimeError(msg)
        return out

    def write_file(self, remote_path, content):
        with self._sftp.open(remote_path, "w") as f:
            f.write(content)

    def upload(self, local_path, remote_path):
        self._sftp.put(local_path, remote_path)

    def close(self):
        if self._sftp:
            self._sftp.close()
        if self._client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
