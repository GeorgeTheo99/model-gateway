"""Shared provider API-key file resolution, validation, and writing."""

from __future__ import annotations

import os
import re
import secrets
import stat
from pathlib import Path

_SAFE_SECRET_NAME = re.compile(r"[a-zA-Z0-9_.-]+")


def secret_dir() -> Path:
    """Directory for gateway-managed provider secrets, outside checkout/config trees."""
    configured = os.environ.get("MODEL_GATEWAY_SECRET_DIR", "").strip()
    return Path(configured or Path.home() / ".config" / "model-gateway" / "secrets").expanduser()


def default_api_key_path(name: str) -> Path:
    """Return the real path of gateway-managed secret file ``name``."""
    if name in {".", ".."} or Path(name).name != name or not _SAFE_SECRET_NAME.fullmatch(name):
        raise ValueError("secret file name must be a safe filename")
    intended = secret_dir() / name
    if intended.is_symlink():
        raise ValueError(f"refusing symlink API key target: {intended}")
    return Path(os.path.realpath(intended))


def resolve_api_key_file(raw_path: str | Path, config_path: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        config_target = Path(os.path.realpath(config_path.expanduser()))
        path = config_target.parent / path
    return Path(os.path.realpath(path))


def read_api_key_file(raw_path: str | Path, config_path: Path) -> str:
    path = resolve_api_key_file(raw_path, config_path)
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise OSError(f"API key path is not a regular file: {path}")
    if file_stat.st_mode & 0o077:
        raise OSError(f"API key file permissions must be 0600: {path}")
    return path.read_text().strip()


def write_api_key_file(path: Path, key: str) -> None:
    """Atomically write ``key`` to a mode-0600 file (creating its directory 0700)."""
    if path.is_symlink():
        raise OSError(f"refusing symlink API key target: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(key.strip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
