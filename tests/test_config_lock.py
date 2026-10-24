"""Config writers lock beside the config, whatever their environment (ADR 0003 section 7)."""

import fcntl
import os
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from src import config_lock

_real_legacy_roots = config_lock._legacy_roots


@pytest.fixture(autouse=True)
def _legacy_roots(tmp_path, monkeypatch):
    roots = [tmp_path / "launchd-tmp", tmp_path / "shell-tmp"]
    monkeypatch.setattr(config_lock, "_legacy_roots", lambda: roots)
    return roots


def _config(tmp_path):
    directory = tmp_path / "shared"
    directory.mkdir()
    path = directory / "config.yaml"
    path.write_text("providers: {}\n")
    return path


def _held_elsewhere(path) -> bool:
    """flock is per open file description, so a second open sees this process's own lock."""
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


def test_lock_lives_beside_the_config_real_path(tmp_path):
    config = _config(tmp_path)
    link = tmp_path / "current-config.yaml"
    link.symlink_to(config)

    path = config_lock.lock_path(link)

    assert path.parent == config.parent / ".locks"
    assert path == config_lock.lock_path(config)
    assert (path.parent.stat().st_mode & 0o777) == 0o700


def test_lock_does_not_depend_on_tmpdir(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "shell"))
    from_shell = config_lock.lock_path(config)
    monkeypatch.delenv("TMPDIR")
    monkeypatch.setattr(tempfile, "tempdir", None)

    assert config_lock.lock_path(config) == from_shell


def test_lock_directory_is_private(tmp_path):
    config = _config(tmp_path)
    (config.parent / ".locks").mkdir(mode=0o755)
    os.chmod(config.parent / ".locks", 0o755)

    config_lock.lock_path(config)

    assert ((config.parent / ".locks").stat().st_mode & 0o777) == 0o700


def test_refuses_a_lock_path_that_is_a_file(tmp_path):
    config = _config(tmp_path)
    (config.parent / ".locks").write_text("")
    with pytest.raises(RuntimeError, match="must be a directory owned by this user"):
        config_lock.lock_path(config)


def test_refuses_a_lock_directory_that_is_not_this_users(tmp_path, monkeypatch):
    config = _config(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (config.parent / ".locks").symlink_to(elsewhere)
    with pytest.raises(RuntimeError, match="must be a directory owned by this user"):
        config_lock.lock_path(config)

    (config.parent / ".locks").unlink()
    config_lock.lock_path(config)
    monkeypatch.setattr(os, "getuid", lambda: os.geteuid() + 1)
    with pytest.raises(RuntimeError, match="must be a directory owned by this user"):
        config_lock.lock_path(config)


def test_writers_hold_the_new_and_legacy_locks(tmp_path):
    config = _config(tmp_path)
    legacy = config_lock.legacy_lock_paths(config)
    assert [path.parent.parent for path in legacy] == [tmp_path / "launchd-tmp", tmp_path / "shell-tmp"]

    with config_lock.config_write_lock(config) as path:
        assert path == config_lock.lock_path(config)
        assert all(_held_elsewhere(held) for held in (path, *legacy))
        with config_lock.config_write_lock(config):  # reentrant in this thread
            pass
        assert all(_held_elsewhere(held) for held in (path, *legacy))

    assert not any(_held_elsewhere(held) for held in (path, *legacy))


def test_an_old_release_holding_its_lock_excludes_new_writers(tmp_path):
    config = _config(tmp_path)
    old = config_lock.legacy_lock_paths(config)[1]
    with open(old, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="another config write holds"):
            with config_lock.config_write_lock(config, blocking=False):
                pass
        # The locks taken before the busy one were released.
        assert not _held_elsewhere(config_lock.lock_path(config))
        assert not _held_elsewhere(config_lock.legacy_lock_paths(config)[0])
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    with config_lock.config_write_lock(config, blocking=False):
        pass


def test_other_threads_are_refused_while_a_write_holds_the_lock(tmp_path):
    config = _config(tmp_path)
    errors = []

    def contend():
        try:
            with config_lock.config_write_lock(config, blocking=False):
                pass
        except RuntimeError as exc:
            errors.append(exc)

    with config_lock.config_write_lock(config):
        thread = threading.Thread(target=contend)
        thread.start()
        thread.join()
    assert len(errors) == 1


def test_legacy_roots_another_user_owns_are_skipped(tmp_path, _legacy_roots):
    config = _config(tmp_path)
    hijacked = _legacy_roots[0] / f"model-gateway-locks-{os.getuid()}"
    hijacked.parent.mkdir()
    hijacked.symlink_to(tmp_path)

    assert [path.parent.parent for path in config_lock.legacy_lock_paths(config)] == [_legacy_roots[1]]


def test_other_threads_wait_for_the_writer(tmp_path):
    config = _config(tmp_path)
    order = []

    def wait():
        with config_lock.config_write_lock(config):
            order.append("waiter")

    with config_lock.config_write_lock(config):
        thread = threading.Thread(target=wait)
        thread.start()
        thread.join(0.2)
        assert thread.is_alive()
        order.append("writer")
    thread.join()
    assert order == ["writer", "waiter"]


def test_legacy_roots_cover_launchd_and_shell_temp_directories(monkeypatch):
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", None)

    roots = _real_legacy_roots()

    assert roots == sorted(set(roots))
    assert all(path == path.resolve() for path in roots)
    assert Path(os.path.realpath("/tmp")) in roots
    if sys.platform == "darwin":
        assert Path(os.path.realpath(os.confstr(65537))) in roots
