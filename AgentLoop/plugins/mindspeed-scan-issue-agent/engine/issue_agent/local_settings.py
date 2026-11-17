"""Local, non-secret policy and Windows Credential Manager storage.

The web UI passes secrets to this module over stdin. Tokens are never written to
JSON, SQLite, command-line arguments, the browser, or the agent transcript.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import tempfile
from ctypes import wintypes
from pathlib import Path

from .models import RepositoryRef


TARGET = "9-17-scan-issue-agent/gitcode"


class _FILETIME(ctypes.Structure):
    _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]


class _CREDENTIAL(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
        ("LastWritten", _FILETIME), ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def _advapi():
    if os.name != "nt":
        raise RuntimeError("secure token storage currently requires Windows Credential Manager")
    return ctypes.WinDLL("Advapi32.dll", use_last_error=True)


def save_token(kind: str, token: str) -> None:
    if kind not in {"read", "write"}:
        raise ValueError("token kind must be read or write")
    if not token or token != token.strip() or "\n" in token or "\r" in token:
        raise ValueError("enter a non-empty GitCode token without whitespace at either end")
    blob = token.encode("utf-16-le")
    if len(blob) > 2560:
        raise ValueError("token exceeds Windows Credential Manager size limit")
    buffer = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
    credential = _CREDENTIAL()
    credential.Type = 1  # CRED_TYPE_GENERIC
    credential.TargetName = f"{TARGET}/{kind}"
    credential.CredentialBlobSize = len(blob)
    credential.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
    credential.Persist = 2  # CRED_PERSIST_LOCAL_MACHINE, scoped to current user
    credential.UserName = os.getenv("USERNAME", "GitCode user")
    api = _advapi()
    api.CredWriteW.argtypes = [ctypes.POINTER(_CREDENTIAL), wintypes.DWORD]
    api.CredWriteW.restype = wintypes.BOOL
    if not api.CredWriteW(ctypes.byref(credential), 0):
        raise ctypes.WinError(ctypes.get_last_error())


def get_token(kind: str) -> str:
    if kind not in {"read", "write"}:
        raise ValueError("token kind must be read or write")
    if os.name != "nt":
        return ""
    api = _advapi()
    pointer = ctypes.POINTER(_CREDENTIAL)()
    api.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                              ctypes.POINTER(ctypes.POINTER(_CREDENTIAL))]
    api.CredReadW.restype = wintypes.BOOL
    api.CredFree.argtypes = [ctypes.c_void_p]
    if not api.CredReadW(f"{TARGET}/{kind}", 1, 0, ctypes.byref(pointer)):
        if ctypes.get_last_error() == 1168:  # ERROR_NOT_FOUND
            return ""
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        credential = pointer.contents
        return ctypes.string_at(credential.CredentialBlob,
                                credential.CredentialBlobSize).decode("utf-16-le")
    finally:
        api.CredFree(pointer)


def delete_token(kind: str) -> None:
    if kind not in {"read", "write"}:
        raise ValueError("token kind must be read or write")
    api = _advapi()
    api.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    api.CredDeleteW.restype = wintypes.BOOL
    if not api.CredDeleteW(f"{TARGET}/{kind}", 1, 0) and ctypes.get_last_error() != 1168:
        raise ctypes.WinError(ctypes.get_last_error())


def read_token() -> str:
    return get_token("read") or os.getenv("GITCODE_READ_TOKEN", "") or os.getenv("GITCODE_TOKEN", "")


def write_token() -> str:
    return get_token("write") or os.getenv("GITCODE_WRITE_TOKEN", "")


class LocalSettings:
    def __init__(self, data_dir: str | Path):
        self.path = Path(data_dir).expanduser().resolve() / "agent-settings.json"

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("agent settings must be a JSON object")
        return {"default_repository": data.get("default_repository") or "",
                "repositories": data.get("repositories") or {}}

    def save(self, data: dict) -> dict:
        repos = data.get("repositories")
        if not isinstance(repos, dict) or len(repos) > 100:
            raise ValueError("repositories must be an object with at most 100 entries")
        normalized: dict[str, dict] = {}
        for name, row in repos.items():
            if not isinstance(name, str) or not re.fullmatch(
                    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name) or \
                    any(part in {".", ".."} for part in name.split("/")):
                raise ValueError(f"invalid GitCode repository: {name}")
            repo = RepositoryRef.parse(str(name))
            if repo.host != "gitcode.com" or repo.full_name != name:
                raise ValueError(f"repository must be exact GitCode owner/repo: {name}")
            if not isinstance(row, dict):
                raise ValueError("repository settings must be an object")
            normalized[name] = {
                "interactive_post": row.get("interactive_post") is True,
                "scheduled_post": row.get("scheduled_post") is True,
            }
        default = str(data.get("default_repository") or "")
        if default and default not in normalized:
            raise ValueError("default repository must be in the trusted repository list")
        result = {"default_repository": default, "repositories": normalized}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                         prefix="settings-", suffix=".tmp", delete=False) as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            temp = Path(stream.name)
        try:
            temp.replace(self.path)
        finally:
            temp.unlink(missing_ok=True)
        return result

    def ensure_repository(self, repo: RepositoryRef) -> dict:
        data = self.load()
        if repo.full_name not in data["repositories"]:
            data["repositories"][repo.full_name] = {"interactive_post": False,
                                                     "scheduled_post": False}
            if not data["default_repository"]:
                data["default_repository"] = repo.full_name
            return self.save(data)
        return data

    def policy(self, repo: RepositoryRef) -> dict:
        return self.load()["repositories"].get(repo.full_name, {})
