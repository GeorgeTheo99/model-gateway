"""Cross-process lock shared by onboarding and runtime config writers.

The lock lives in ``.locks/`` beside the config's real path, so every writer
of one config resolves the same lock whatever its environment
(docs/adr/0003-scoped-access-and-lightweight-local-models.md section 7).
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import tempfile
import threading
from contextlib import ExitStack, contextmanager
from pathlib import Path

# confstr(_CS_DARWIN_USER_TEMP_DIR): the per-user temp directory a shell's
# $TMPDIR names, which launchd jobs (without $TMPDIR) cannot otherwise find.
_CS_DARWIN_USER_TEMP_DIR = 65537


def _digest(config_path: Path) -> tuple[Path, str]:
    resolved = Path(os.path.realpath(config_path.expanduser()))
    return resolved, hashlib.sha256(str(resolved).encode()).hexdigest()[:20]


def _own_directory(root: Path) -> bool:
    """Create ``root`` if needed; true only when it is a real directory owned by this user (then 0700)."""
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        return False
    if stat.S_IMODE(info.st_mode) != 0o700:
        os.chmod(root, 0o700)
    return True


def lock_path(config_path: Path) -> Path:
    resolved, digest = _digest(config_path)
    root = resolved.parent / ".locks"
    try:
        owned = _own_directory(root)
    except OSError as exc:
        raise RuntimeError(f"{root} must be a directory owned by this user: {exc}") from exc
    if not owned:
        raise RuntimeError(f"{root} must be a directory owned by this user")
    return root / f"{digest}.lock"


def _legacy_roots() -> list[Path]:
    roots = {Path("/tmp"), Path(tempfile.gettempdir())}
    try:
        if user_temp := os.confstr(_CS_DARWIN_USER_TEMP_DIR):
            roots.add(Path(user_temp))
    except (OSError, ValueError):  # not macOS
        pass
    return sorted({Path(os.path.realpath(root)) for root in roots})


def legacy_lock_paths(config_path: Path) -> list[Path]:
    """Where releases before 0.8.4 locked this config: ``tempfile.gettempdir()``,
    which is ``/tmp`` under launchd and ``$TMPDIR`` in a shell. Writers hold these
    too for one release so processes from either version exclude each other
    during an upgrade; remove after 0.8.4. Roots another user owns are skipped."""
    _resolved, digest = _digest(config_path)
    paths = []
    for root in _legacy_roots():
        directory = root / f"model-gateway-locks-{os.getuid()}"
        try:
            if _own_directory(directory):
                paths.append(directory / f"{digest}.lock")
        except OSError:
            continue
    return paths


_process_locks_guard = threading.Lock()
_process_locks: dict[Path, threading.RLock] = {}
_thread_state = threading.local()


def _process_lock(path: Path) -> threading.RLock:
    with _process_locks_guard:
        return _process_locks.setdefault(path, threading.RLock())


def _flock(stack: ExitStack, path: Path, *, blocking: bool) -> None:
    handle = stack.enter_context(open(path, "a+"))
    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(handle.fileno(), flags)
    except BlockingIOError as exc:
        raise RuntimeError(f"another config write holds {path}") from exc
    stack.callback(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)


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

        # One fixed order (new lock, then sorted legacy locks) so writers never deadlock.
        with ExitStack() as stack:
            for held in (path, *legacy_lock_paths(config_path)):
                _flock(stack, held, blocking=blocking)
            depths[path] = 1
            try:
                yield path
            finally:
                depths.pop(path, None)
    finally:
        process_lock.release()
