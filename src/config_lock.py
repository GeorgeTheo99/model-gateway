"""Cross-process lock shared by onboarding and runtime config writers."""

from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def lock_path(config_path: Path) -> Path:
    resolved = Path(os.path.realpath(config_path.expanduser()))
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:20]
    root = Path(tempfile.gettempdir()) / f"model-gateway-locks-{os.getuid()}"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root / f"{digest}.lock"


@contextmanager
def config_write_lock(config_path: Path, *, blocking: bool = True):
    path = lock_path(config_path)
    with open(path, "a+") as handle:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            raise RuntimeError(f"another config write holds {path}") from exc
        try:
            yield path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
