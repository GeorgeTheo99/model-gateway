"""Cross-process lock shared by onboarding and runtime config writers."""

from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path


def lock_path(config_path: Path) -> Path:
    resolved = Path(os.path.realpath(config_path.expanduser()))
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:20]
    root = Path(tempfile.gettempdir()) / f"model-gateway-locks-{os.getuid()}"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root / f"{digest}.lock"


_process_locks_guard = threading.Lock()
_process_locks: dict[Path, threading.RLock] = {}
_thread_state = threading.local()


def _process_lock(path: Path) -> threading.RLock:
    with _process_locks_guard:
        return _process_locks.setdefault(path, threading.RLock())


@contextmanager
def config_write_lock(config_path: Path, *, blocking: bool = True):
    """Cross-process lock that is reentrant within the current thread."""
    path = lock_path(config_path)
    process_lock = _process_lock(path)
    if not process_lock.acquire(blocking=blocking):
        raise RuntimeError(f"another config write holds {path}")
    depths = getattr(_thread_state, "depths", None)
    if depths is None:
        depths = _thread_state.depths = {}
    try:
        if depths.get(path, 0):
            depths[path] += 1
            try:
                yield path
            finally:
                depths[path] -= 1
            return

        with open(path, "a+") as handle:
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(handle.fileno(), flags)
            except BlockingIOError as exc:
                raise RuntimeError(f"another config write holds {path}") from exc
            depths[path] = 1
            try:
                yield path
            finally:
                depths.pop(path, None)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        process_lock.release()
