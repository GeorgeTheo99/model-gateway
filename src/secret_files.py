"""Shared provider API-key file resolution and validation."""

from __future__ import annotations

import os
import stat
from pathlib import Path


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
