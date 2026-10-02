"""Safe read/write layer for model-gateway config files.

Two files, two deploy models (see docs/provider-pricing-sources.md and the
productionization plan):

- ``config.yaml`` (providers + auth): deployed copy is a symlink to the shared
  gitignored file. Edits are hot (after reload) and durable across deploys.
  Static provider keys written here go to mode-0600 ``api_key_file`` targets,
  never inline YAML; never echo a key back; writes are additive/preserving.
- ``model-info.json`` (models): machine-local and Git-ignored. Writes edit the
  live copy (hot) and, when configured, a second machine-local mirror
  (``MODEL_INFO_SOURCE_PATH``) used by deploy tooling or private backup flows.

All writes: validate -> backup -> atomic temp+rename. Best-effort; a write
failure raises a clear error to the admin API caller.
"""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import shutil
import stat
import time
from functools import wraps
from pathlib import Path
from typing import Any

import yaml

from src.catalog import normalize_thinking_capabilities, validate_pricing_policy
from src.config_lock import config_write_lock
from src.providers import CONFIG_PATH, MODEL_INFO_PATH, MODEL_INFO_SOURCE_PATH
from src.secret_files import default_api_key_path, resolve_api_key_file, write_api_key_file

log = logging.getLogger("model-gateway")

log_dir = Path(
    os.environ.get("MODEL_GATEWAY_LOG_DIR", str(Path.home() / "Library" / "Logs" / "model-gateway"))
)


def _config_backup_dir() -> Path:
    """Return the private config/catalog backup directory.

    Backups may contain provider credentials or a private model catalog, so a
    deployment keeps them in private state outside the diagnostic log tree.
    """
    configured = os.environ.get("MODEL_GATEWAY_BACKUP_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Library" / "Application Support" / "model-gateway" / "backups" / "config"


def _write_transaction(function):
    """Hold the shared lock across a complete read-modify-write operation."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with config_write_lock(CONFIG_PATH):
            return function(*args, **kwargs)
    return wrapped


# ── provider config (config.yaml) ───────────────────────────────────────────


def load_config_full() -> dict:
    """Load the full config.yaml (all sections, including auth/providers)."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _resolve_target(path: Path) -> Path:
    """Resolve symlinks so writes land on the real file, not replace the link.

    The deployed config.yaml is a symlink to the shared gitignored file. A
    naive os.replace would write a real file at the link location, breaking
    the link and splitting config state. Resolve to the real target first.
    """
    try:
        return Path(os.path.realpath(path))
    except OSError:
        return path


def _is_secret_path(target: Path) -> bool:
    """Whether the resolved target holds secrets (config.yaml with API keys)."""
    return target == _resolve_target(CONFIG_PATH)


def _atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically: temp file in same dir, then rename.

    Resolves symlinks first so a symlinked config file (deployed config.yaml)
    is updated in place rather than replaced with a real file. Secret-bearing
    files (config.yaml) are always clamped to 0600 — a pre-existing loose mode
    is tightened rather than preserved.
    """
    target = _resolve_target(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode & 0o777 if target.exists() else 0o600
    if _is_secret_path(target):
        mode = 0o600
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _open_private_backup_directory(path: Path) -> tuple[int, Path]:
    """Securely create/open an absolute directory without following symlinks."""
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    if not absolute.is_absolute():  # pragma: no cover - abspath is defensive
        raise RuntimeError("model-gateway backup path must be absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                os.mkdir(component, 0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        metadata = os.fstat(directory_fd)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise RuntimeError(f"model-gateway backup directory is not privately owned: {absolute}")
        os.fchmod(directory_fd, 0o700)
        return directory_fd, absolute
    except BaseException:
        os.close(directory_fd)
        raise


def _secure_backup_tree(directory_fd: int) -> None:
    """Recursively clamp an owned legacy backup tree without following links."""
    for name in os.listdir(directory_fd):
        entry_fd = os.open(
            name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            metadata = os.fstat(entry_fd)
            if metadata.st_uid != os.getuid():
                raise RuntimeError(f"legacy backup entry is not owned by this user: {name}")
            if stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise RuntimeError(f"legacy backup file has hard links: {name}")
                os.fchmod(entry_fd, 0o600)
            elif stat.S_ISDIR(metadata.st_mode):
                os.fchmod(entry_fd, 0o700)
                _secure_backup_tree(entry_fd)
            else:
                raise RuntimeError(f"legacy backup entry is not a regular file or directory: {name}")
        finally:
            os.close(entry_fd)


def _validate_backup_separation() -> tuple[Path, Path]:
    """Reject equal or nested diagnostic-log and private-backup roots."""
    logs = Path(os.path.realpath(os.path.abspath(os.fspath(log_dir.expanduser()))))
    backups = Path(os.path.realpath(os.path.abspath(os.fspath(_config_backup_dir()))))
    common = Path(os.path.commonpath((logs, backups)))
    if common in {logs, backups}:
        raise RuntimeError("model-gateway diagnostic logs and private backups must not overlap")
    return logs, backups


def _migrate_legacy_backup_directory(legacy: Path) -> int:
    if not os.path.lexists(legacy):
        return 0
    legacy_parent_fd, _ = _open_private_backup_directory(legacy.parent)
    legacy_fd, _ = _open_private_backup_directory(legacy)
    target_fd, _ = _open_private_backup_directory(_config_backup_dir())
    try:
        legacy_meta = os.fstat(legacy_fd)
        target_meta = os.fstat(target_fd)
        if (legacy_meta.st_dev, legacy_meta.st_ino) == (target_meta.st_dev, target_meta.st_ino):
            raise RuntimeError("legacy backup directory still overlaps the configured private backup root")
        if legacy_meta.st_dev != target_meta.st_dev:
            raise RuntimeError(
                "legacy and configured model-gateway backup directories must be on the same filesystem"
            )
        _secure_backup_tree(legacy_fd)
        _prune_backups(legacy_fd, "config.yaml")
        _prune_backups(legacy_fd, "model-info.json")
        moved = len(os.listdir(legacy_fd))
        destination = f"legacy-config-backups-{time.time_ns()}"
        try:
            os.stat(destination, dir_fd=target_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:  # pragma: no cover - nanosecond collision is defensive
            raise RuntimeError(f"legacy backup migration destination already exists: {destination}")
        os.rename(
            legacy.name,
            destination,
            src_dir_fd=legacy_parent_fd,
            dst_dir_fd=target_fd,
        )
        _prune_all_backup_generations(target_fd, "config.yaml")
        _prune_all_backup_generations(target_fd, "model-info.json")
        os.fsync(target_fd)
        os.fsync(legacy_parent_fd)
        return moved
    finally:
        os.close(target_fd)
        os.close(legacy_fd)
        os.close(legacy_parent_fd)


def migrate_legacy_backups() -> int:
    """Move every historical config/catalog backup root out of logs."""
    logs, _ = _validate_backup_separation()
    configured = os.environ.get("MODEL_GATEWAY_LEGACY_BACKUP_DIRS")
    if configured is None:
        candidates = [
            logs / "config-backups",
            Path.home() / "Library" / "Application Support" / "HomeServer" / "ci" / "logs" / "config-backups",
        ]
    else:
        candidates = [Path(value).expanduser() for value in configured.split(os.pathsep) if value]
    moved = 0
    seen: set[str] = set()
    for candidate in candidates:
        # Preserve the lexical source so the no-follow directory walk can
        # reject symlinked roots/components before any rename. Never realpath a
        # migration source into an unrelated same-user target.
        lexical = os.path.abspath(os.fspath(candidate))
        if lexical in seen:
            continue
        seen.add(lexical)
        moved += _migrate_legacy_backup_directory(Path(lexical))
    return moved


def _backup_retention() -> int:
    raw_retention = os.environ.get("MODEL_GATEWAY_BACKUP_RETENTION", "20").strip()
    try:
        retention = int(raw_retention)
    except ValueError as exc:
        raise RuntimeError("MODEL_GATEWAY_BACKUP_RETENTION must be an integer") from exc
    if retention < 1 or retention > 1000:
        raise RuntimeError("MODEL_GATEWAY_BACKUP_RETENTION must be between 1 and 1000")
    return retention


def _prune_backups(directory_fd: int, source_name: str) -> None:
    """Retain a bounded private history for one managed file in one directory."""
    retention = _backup_retention()
    prefix = f"{source_name}.bak."
    backups: list[tuple[int, str]] = []
    for name in os.listdir(directory_fd):
        if not name.startswith(prefix) or not name[len(prefix):].isdigit():
            continue
        entry_fd = os.open(
            name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            metadata = os.fstat(entry_fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
            ):
                raise RuntimeError(f"unsafe model-gateway backup entry: {name}")
            os.fchmod(entry_fd, 0o600)
            backups.append((metadata.st_mtime_ns, name))
        finally:
            os.close(entry_fd)
    backups.sort(reverse=True)
    for _, name in backups[retention:]:
        os.unlink(name, dir_fd=directory_fd)


def _prune_all_backup_generations(directory_fd: int, source_name: str) -> None:
    """Apply one global retention cap across current and imported generations."""
    retention = _backup_retention()
    prefix = f"{source_name}.bak."
    found: list[tuple[int, str | None, str]] = []

    def collect(parent_fd: int, parent_name: str | None) -> None:
        for entry in os.listdir(parent_fd):
            if not entry.startswith(prefix) or not entry[len(prefix):].isdigit():
                continue
            entry_fd = os.open(
                entry,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            try:
                metadata = os.fstat(entry_fd)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
                    raise RuntimeError(f"unsafe model-gateway backup entry: {entry}")
                os.fchmod(entry_fd, 0o600)
                found.append((metadata.st_mtime_ns, parent_name, entry))
            finally:
                os.close(entry_fd)

    collect(directory_fd, None)
    archives: dict[str, int] = {}
    try:
        for name in os.listdir(directory_fd):
            if not name.startswith("legacy-config-backups-"):
                continue
            archive_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            metadata = os.fstat(archive_fd)
            if metadata.st_uid != os.getuid():
                os.close(archive_fd)
                raise RuntimeError(f"unsafe imported backup archive: {name}")
            os.fchmod(archive_fd, 0o700)
            archives[name] = archive_fd
            collect(archive_fd, name)
        found.sort(reverse=True)
        for _, parent_name, name in found[retention:]:
            os.unlink(name, dir_fd=directory_fd if parent_name is None else archives[parent_name])
        for name, archive_fd in archives.items():
            if not os.listdir(archive_fd):
                os.rmdir(name, dir_fd=directory_fd)
    finally:
        for archive_fd in archives.values():
            os.close(archive_fd)


def _backup(path: Path) -> Path | None:
    """Copy a managed file to a private, bounded timestamped backup."""
    if not path.exists():
        return None
    _validate_backup_separation()
    backup_fd, backup_dir = _open_private_backup_directory(_config_backup_dir())
    name = f"{path.name}.bak.{time.time_ns()}"
    bak = backup_dir / name
    try:
        _prune_all_backup_generations(backup_fd, path.name)
        source_path = _resolve_target(path)
        source_fd = os.open(source_path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            source_meta = os.fstat(source_fd)
            if (
                not stat.S_ISREG(source_meta.st_mode)
                or source_meta.st_uid != os.getuid()
                or source_meta.st_nlink != 1
            ):
                raise RuntimeError(f"unsafe model-gateway backup source: {source_path}")
            with os.fdopen(source_fd, "rb", closefd=False) as source:
                destination_fd = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=backup_fd,
                )
                try:
                    with os.fdopen(destination_fd, "wb") as destination:
                        shutil.copyfileobj(source, destination)
                        destination.flush()
                        os.fsync(destination.fileno())
                except BaseException:
                    os.unlink(name, dir_fd=backup_fd)
                    raise
        finally:
            os.close(source_fd)
        _prune_all_backup_generations(backup_fd, path.name)
        os.fsync(backup_fd)
        return bak
    finally:
        os.close(backup_fd)


def backup_status() -> dict:
    """Summarize the private backup directory by name and mtime only.

    Read-only: never creates, opens, or re-permissions backups. Managed
    generations (``<file>.bak.<ns>``, including imported legacy archives) are
    grouped per source file; anything else, such as a manual snapshot, is
    counted as ``other``. Symlinks are ignored.
    """
    directory = _config_backup_dir()
    generations: dict[str, list[float]] = {}
    other: list[float] = []

    def scan(path: Path, *, top: bool) -> None:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                mtime = entry.stat(follow_symlinks=False).st_mtime
                source, sep, suffix = entry.name.rpartition(".bak.")
                if sep and suffix.isdigit() and entry.is_file(follow_symlinks=False):
                    generations.setdefault(source, []).append(mtime)
                elif top and entry.name.startswith("legacy-config-backups-") and entry.is_dir(follow_symlinks=False):
                    scan(Path(entry.path), top=False)
                elif top:
                    other.append(mtime)

    exists = directory.is_dir() and not directory.is_symlink()
    if exists:
        scan(directory, top=True)
    return {
        "directory": str(directory),
        "exists": exists,
        "retention": _backup_retention(),
        "generations": [
            {"file": name, "count": len(times), "newest": max(times), "oldest": min(times)}
            for name, times in sorted(generations.items())
        ],
        "other": {"count": len(other), "newest": max(other) if other else None},
    }


def snapshot_writable_files(extra_paths=()) -> dict[Path, str | None]:
    """Capture every admin-managed file (plus ``extra_paths``) for validation rollback."""
    paths = {CONFIG_PATH, MODEL_INFO_PATH, *extra_paths}
    if MODEL_INFO_SOURCE_PATH:
        paths.add(MODEL_INFO_SOURCE_PATH)
    snapshot = {}
    for path in paths:
        target = _resolve_target(path)
        snapshot[path] = target.read_text() if target.exists() else None
    return snapshot


def restore_writable_files(snapshot: dict[Path, str | None]) -> None:
    """Atomically restore files captured by :func:`snapshot_writable_files`."""
    with config_write_lock(CONFIG_PATH):
        for path, text in snapshot.items():
            if text is None:
                _resolve_target(path).unlink(missing_ok=True)
            else:
                _atomic_write(path, text)


@_write_transaction
def upsert_provider(
    provider_id: str,
    *,
    base_url: str,
    api_key: str | None = None,
    protocol: str | None = None,
    default_headers: dict | None = None,
) -> dict:
    """Create or update a provider block in config.yaml.

    ``api_key`` is write-only: if None, the existing key is preserved; if an
    empty string, the key reference is removed and a gateway-managed key file
    that nothing else references is deleted.
    A static key is stored in the provider's mode-0600 ``api_key_file``, never
    inline. Other fields are set only when provided. Returns the masked
    provider status dict (no secrets).
    """
    provider_id = (provider_id or "").strip().lower()
    if not provider_id:
        raise ValueError("provider_id is required")
    if not base_url:
        raise ValueError("base_url is required")

    config = load_config_full()
    providers = config.setdefault("providers", {}) or {}
    # Normalize: store under the canonical id. If a synonym key exists, update
    # it in place; otherwise create under provider_id.
    existing = providers.get(provider_id)
    block: dict = dict(existing) if isinstance(existing, dict) else {}

    block["base_url"] = base_url
    if protocol is not None:
        block["protocol"] = protocol
    if default_headers is not None:
        block["default_headers"] = default_headers
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError("api_key must be a string")
    retired: list[Path] = []
    if api_key is not None:
        if api_key.strip() == "":
            retired = _managed_key_files(provider_id, block)
            block.pop("api_key", None)
            block.pop("api_key_file", None)
        else:
            _store_api_key(config, provider_id, block, api_key)
    providers[provider_id] = block
    config["providers"] = providers

    _backup(CONFIG_PATH)
    _atomic_write(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    _delete_unreferenced_key_files(config, retired)
    return _masked_block(provider_id, block)


@_write_transaction
def delete_provider(provider_id: str) -> dict:
    """Remove a provider from config.yaml. Refuses if models depend on it."""
    from src.providers import _load_models  # local import to avoid cycle at load

    provider_id = (provider_id or "").strip().lower()
    config = load_config_full()
    providers = config.get("providers", {}) or {}
    if provider_id not in providers and not any(
        k.lower() == provider_id for k in providers
    ):
        raise KeyError(f"provider {provider_id!r} not found")

    # Refuse if any enabled model routes to this provider, however it is spelled.
    from src.providers import _canonical_provider, _is_model_enabled
    canonical = _canonical_provider(provider_id)
    dependents = []
    for entry in {id(v): v for v in _load_models().values()}.values():
        if _canonical_provider(entry.get("provider")) == canonical and _is_model_enabled(entry.get("name")):
            dependents.append(entry.get("name", ""))
    if dependents:
        raise ValueError(
            f"cannot delete provider {provider_id!r}: {len(dependents)} enabled "
            f"model(s) depend on it: {', '.join(sorted(set(dependents)))}. "
            "Disable or reassign them first."
        )

    # Remove the key (and any synonym key).
    retired: list[Path] = []
    for k in list(providers.keys()):
        if k.lower() == provider_id:
            retired += _managed_key_files(k.lower(), providers[k])
            del providers[k]
    config["providers"] = providers
    _backup(CONFIG_PATH)
    _atomic_write(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    _delete_unreferenced_key_files(config, retired)
    return {"id": provider_id, "deleted": True}


def _masked_block(provider_id: str, block: dict) -> dict:
    from src.providers import _safe_url

    return {
        "id": provider_id,
        "base_url": _safe_url(block.get("base_url", "")),
        "protocol": block.get("protocol", "openai"),
        "has_api_key": bool(block.get("api_key") or block.get("api_key_file")),
        "api_key_source": _configured_key_source(block),
        "default_headers": bool(block.get("default_headers")),
    }


def _configured_key_source(block: dict) -> str:
    if block.get("api_key"):
        return "inline"
    return "file" if block.get("api_key_file") else "missing"


# ── provider API-key files ──────────────────────────────────────────────────


def _uses_inline_token(block: dict) -> bool:
    """OAuth refreshers rewrite short-lived tokens in config.yaml itself."""
    return bool(block.get("auth_refresh"))


def _api_key_file_target(provider_id: str, block: dict) -> Path:
    """The real file a provider's static key lives in: its reference or the default.

    Like the default location, an existing reference must not be a symlink, so
    a key write never lands in a file the reference merely points at.
    """
    raw = block.get("api_key_file")
    if not raw:
        return default_api_key_path(f"{provider_id}.api-key")
    intended = Path(str(raw)).expanduser()
    if not intended.is_absolute():
        intended = _resolve_target(CONFIG_PATH).parent / intended
    if intended.is_symlink():
        raise ValueError(f"refusing symlink API key target: {intended}")
    return resolve_api_key_file(raw, CONFIG_PATH)


def _managed_key_files(provider_id: str, block: object) -> list[Path]:
    """The provider's own default key file, if the block references it.

    Only ``<secret dir>/<provider id>.api-key`` is ever deleted; an
    owner-supplied or custom-named key file is left alone.
    """
    if not isinstance(block, dict) or not block.get("api_key_file") or _uses_inline_token(block):
        return []
    try:
        target = _api_key_file_target(provider_id, block)
        own = default_api_key_path(f"{provider_id}.api-key")
    except ValueError:
        return []
    return [target] if target == own else []


def managed_key_files(provider_id: str) -> list[Path]:
    """Key files that clearing or deleting ``provider_id`` could delete (for rollback snapshots)."""
    provider_id = (provider_id or "").strip().lower()
    providers = load_config_full().get("providers") or {}
    return [path for key, block in providers.items() if key.lower() == provider_id
            for path in _managed_key_files(key.lower(), block)]


def _delete_unreferenced_key_files(config: dict, paths: list[Path]) -> None:
    """Delete retired key files once the saved config no longer references them.

    Runs after the config is saved, so a failure only leaves the file behind.
    Files are compared by identity: case-insensitive volumes and hard links can
    give one file several spellings.
    """
    if not paths:
        return
    still_used = [path for _owner, path in _key_file_owners(config) if path.exists()]
    for path in paths:
        try:
            if path.exists() and not any(os.path.samefile(path, used) for used in still_used):
                path.unlink()
        except OSError as exc:
            log.warning("could not delete retired provider key file %s: %s", path, exc)


def _owner_label(section: str, name: str) -> str:
    from src.providers import _canonical_provider

    return f"{section}.{_canonical_provider(str(name).strip().lower())}"


def _key_file_owners(config: dict) -> list[tuple[str, Path]]:
    """(owner, real path) for every provider, workspace, federation peer,
    consumer credential, and client keys file, so a provider key write can
    never overwrite another credential."""
    owners = []
    for section in ("providers", "workspaces"):
        entries = config.get(section) or {}
        for name, block in entries.items() if isinstance(entries, dict) else ():
            if isinstance(block, dict) and block.get("api_key_file"):
                owners.append((_owner_label(section, name), resolve_api_key_file(block["api_key_file"], CONFIG_PATH)))
    federation = config.get("federation") or {}
    peers = federation.get("peers") if isinstance(federation, dict) else None
    for peer, block in peers.items() if isinstance(peers, dict) else ():
        if isinstance(block, dict) and block.get("api_key_file"):
            owners.append((f"federation peer {peer}", resolve_api_key_file(block["api_key_file"], CONFIG_PATH)))
    auth = config.get("auth") or {}
    consumers = auth.get("consumer_credentials") if isinstance(auth, dict) else None
    for entry in consumers if isinstance(consumers, list) else ():
        if isinstance(entry, dict) and isinstance(entry.get("key_file"), str) and entry["key_file"].strip():
            owners.append((f"consumer {entry.get('id')}", resolve_api_key_file(entry["key_file"], CONFIG_PATH)))
    client_keys_file = os.environ.get("MODEL_GATEWAY_CLIENT_KEYS_FILE", "").strip()
    if client_keys_file:
        owners.append(("client keys file", resolve_api_key_file(client_keys_file, CONFIG_PATH)))
    return owners


def _check_key_file_target(config: dict, owner: str, target: Path) -> None:
    managed = {_resolve_target(CONFIG_PATH), _resolve_target(MODEL_INFO_PATH)}
    if MODEL_INFO_SOURCE_PATH:
        managed.add(_resolve_target(MODEL_INFO_SOURCE_PATH))
    if target in managed:
        raise ValueError(f"API key file must not be a gateway config or catalog file: {target}")
    for other, path in _key_file_owners(config):
        if path == target and other != owner:
            raise ValueError(f"API key file {target} is already used by {other!r}")


def _store_api_key(config: dict, provider_id: str, block: dict, api_key: str) -> None:
    if _uses_inline_token(block):
        block["api_key"] = api_key.strip()
        return
    target = _api_key_file_target(provider_id, block)
    _check_key_file_target(config, _owner_label("providers", provider_id), target)
    write_api_key_file(target, api_key)
    block.setdefault("api_key_file", str(target))
    block.pop("api_key", None)


def api_key_file_target(provider_id: str) -> Path | None:
    """Key file an ``upsert_provider`` key write for ``provider_id`` would touch."""
    provider_id = (provider_id or "").strip().lower()
    block = (load_config_full().get("providers") or {}).get(provider_id)
    block = block if isinstance(block, dict) else {}
    if not provider_id or _uses_inline_token(block):
        return None
    try:
        return _api_key_file_target(provider_id, block)
    except ValueError:
        return None


@_write_transaction
def migrate_inline_api_keys(*, dry_run: bool = False) -> list[dict]:
    """Move static inline provider/workspace keys into mode-0600 ``api_key_file``s.

    The inline key is the effective one, so it also replaces the contents of an
    existing (shadowed) file reference. OAuth-refreshed tokens stay inline. All
    files are restored if any write fails. Returns one masked entry per move.
    """
    config = load_config_full()
    planned: list[tuple[str, dict, Path]] = []
    claimed: dict[Path, str] = {}
    for section in ("providers", "workspaces"):
        entries = config.get(section) or {}
        for name, block in entries.items() if isinstance(entries, dict) else ():
            if not isinstance(block, dict) or not block.get("api_key") or _uses_inline_token(block):
                continue
            provider_id = str(name).strip().lower()
            target = _api_key_file_target(provider_id, block)
            _check_key_file_target(config, _owner_label(section, provider_id), target)
            if target in claimed:
                raise ValueError(f"providers {claimed[target]!r} and {name!r} would share API key file {target}")
            claimed[target] = str(name)
            planned.append((str(name), block, target))
    summary = [{"provider": name, "api_key_file": str(target)} for name, _, target in planned]
    if dry_run or not planned:
        return summary
    _backup(CONFIG_PATH)
    snapshot = snapshot_writable_files([target for _, _, target in planned])
    try:
        for _, block, target in planned:
            write_api_key_file(target, str(block["api_key"]))
            block.setdefault("api_key_file", str(target))
            block.pop("api_key")
        _atomic_write(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    except Exception:
        restore_writable_files(snapshot)
        raise
    return summary


# ── consumer credentials (auth.consumer_credentials) ────────────────────────

# Roles map onto the profile permissions in docs/consumer-profiles.md.
CONSUMER_ROLES = {
    "runtime": ["profiles:read", "profiles:invoke"],
    "deployer": ["profiles:read", "profiles:write"],
    # Admin-API access confined to the credential's ``providers`` allowlist.
    "manager": ["providers:manage", "models:register"],
}


def consumer_key_path(credential_id: str) -> Path:
    """Default key file for a consumer credential, beside the live config."""
    from src.auth import _PRINCIPAL_ID_RE

    if not _PRINCIPAL_ID_RE.fullmatch(credential_id or ""):
        raise ValueError("consumer credential id is invalid")
    return _resolve_target(CONFIG_PATH).parent / "secrets" / "consumers" / f"{credential_id}.key"


def consumer_credential_id(consumer: object, role: object) -> str:
    """Validate ``consumer`` and ``role`` and return credential id ``<consumer>-<role>``."""
    from src.auth import _PRINCIPAL_ID_RE

    if role not in CONSUMER_ROLES:
        raise ValueError(f"role must be one of: {', '.join(CONSUMER_ROLES)}")
    if not isinstance(consumer, str) or not _PRINCIPAL_ID_RE.fullmatch(consumer):
        raise ValueError("consumer must be 1-32 lowercase letters, digits, or hyphens")
    credential_id = f"{consumer}-{role}"
    if len(credential_id) > 32:
        raise ValueError(
            f"consumer id is too long: {credential_id!r} must be at most 32 characters, "
            f"so a {role} consumer id can have at most {31 - len(role)}"
        )
    return credential_id


def _consumer_entries(config: dict) -> list:
    auth = config.setdefault("auth", {})
    if not isinstance(auth, dict):
        raise ValueError("auth configuration must be an object")
    entries = auth.setdefault("consumer_credentials", [])
    if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
        raise ValueError("auth.consumer_credentials must be a list of objects")
    return entries


def _consumer_key_status(entry: dict) -> str:
    from src.auth import _read_consumer_key_file

    if "key" in entry:
        return "inline"
    try:
        _read_consumer_key_file(entry.get("key_file"))
    except FileNotFoundError:
        return "missing"
    except (OSError, UnicodeError, ValueError):
        return "invalid"
    return "ok"


def _consumer_summary(entry: dict) -> dict:
    """Non-secret view of one credential entry."""
    try:
        managed = _managed_consumer_key(entry, str(entry.get("id") or "")) is not None
    except ValueError:
        managed = False
    return {
        "id": entry.get("id"),
        "consumer": entry.get("consumer"),
        "namespaces": list(entry.get("namespaces") or []),
        "permissions": list(entry.get("permissions") or []),
        "allow_direct_models": bool(entry.get("allow_direct_models", False)),
        "providers": list(entry.get("providers") or []),
        "key_file": entry.get("key_file"),
        "key_status": _consumer_key_status(entry),
        "managed_key_file": managed,
    }


def list_consumer_credentials() -> list[dict]:
    """Return every consumer credential without key values."""
    return [_consumer_summary(entry) for entry in _consumer_entries(load_config_full())]


def _validate_auth_after_write(snapshot: dict[Path, str | None], *, keep_client_auth: bool = False) -> None:
    """Reload the written config and roll every file back if auth rejects it.

    ``keep_client_auth`` also refuses a write that leaves ``/v1`` with no
    client or consumer credential, which would reopen it without auth.
    """
    from src import auth, providers

    providers.reload()
    try:
        consumers, clients, _admins = auth.validate_credential_separation()
        if keep_client_auth and not consumers and not clients:
            raise ValueError("refusing to remove the last /v1 credential; add another client or consumer key first")
    except Exception:
        restore_writable_files(snapshot)
        raise
    finally:
        providers.reload()


@_write_transaction
def add_consumer_credential(
    consumer: str,
    role: str,
    *,
    namespaces: list[str] | None = None,
    allow_direct_models: bool = False,
    providers: list[str] | None = None,
    local_ai: bool = False,
    reveal_key: bool = False,
) -> dict:
    """Create credential ``<consumer>-<role>`` with a generated mode-0600 key file.

    Idempotent: an identical existing entry is reported unchanged, and an
    existing key file at the default path is adopted rather than replaced. An
    identical entry whose default key file is missing gets a new key.
    Returns the non-secret summary plus ``status`` (``created``/``repaired``/
    ``unchanged``) and ``enables_client_auth`` when ``/v1`` was open before.
    ``providers`` is required for, and only accepted by, the manager role.
    ``local_ai`` adds ``local_ai:manage`` (the local AI admin API) to any role.
    ``reveal_key`` adds ``key`` only when this call generated it, for the
    admin UI's one-time display; an adopted or unchanged key is never read.
    """
    from src.auth import _PRINCIPAL_ID_RE, _read_consumer_key_file

    credential_id = consumer_credential_id(consumer, role)
    target = consumer_key_path(credential_id)
    namespaces = list(namespaces or [consumer])
    if any(not isinstance(value, str) or not _PRINCIPAL_ID_RE.fullmatch(value) for value in namespaces) or (
        len(set(namespaces)) != len(namespaces)
    ):
        raise ValueError("namespaces must be unique 1-32 character lowercase ids")
    if role == "manager":
        if not providers:
            raise ValueError("the manager role requires a providers allowlist")
    elif providers:
        raise ValueError("providers applies only to the manager role")
    desired = {
        "id": credential_id,
        "consumer": consumer,
        "key_file": str(target),
        "namespaces": namespaces,
        "permissions": list(CONSUMER_ROLES[role]) + (["local_ai:manage"] if local_ai else []),
        "allow_direct_models": bool(allow_direct_models),
    }
    if role == "manager":
        desired["providers"] = list(providers)

    config = load_config_full()
    entries = _consumer_entries(config)
    existing = next((entry for entry in entries if entry.get("id") == credential_id), None)
    if existing is not None:
        comparable = {key: existing.get(key) for key in desired if key != "key_file"}
        comparable["allow_direct_models"] = bool(existing.get("allow_direct_models", False))
        if comparable != {key: value for key, value in desired.items() if key != "key_file"}:
            raise ValueError(
                f"consumer credential {credential_id!r} already exists with different settings; revoke it first"
            )
        summary = _consumer_summary(existing)
        if summary["key_status"] == "ok":
            return {**summary, "status": "unchanged", "enables_client_auth": False}
        if summary["key_status"] != "missing" or "key_file" not in existing or (
            resolve_api_key_file(existing["key_file"], CONFIG_PATH) != target
        ):
            raise ValueError(
                f"consumer credential {credential_id!r} has an unusable key file ({summary['key_status']}); "
                "fix its permissions or revoke and re-add it"
            )
        if target.is_symlink():
            raise ValueError(f"refusing symlink consumer key target: {target}")
        snapshot = snapshot_writable_files([target])
        token = secrets.token_hex(32)
        write_api_key_file(target, token)
        _validate_auth_after_write(snapshot)
        result = {**_consumer_summary(existing), "status": "repaired", "enables_client_auth": False}
        return {**result, "key": token} if reveal_key else result

    from src import auth, providers

    try:
        providers.reload()
        consumers_before, clients_before, _admins = auth.validate_credential_separation()
        enables_client_auth = not consumers_before and not clients_before
    except Exception:  # noqa: BLE001 — a broken config is reported by the post-write check
        enables_client_auth = False

    if target.is_symlink():
        raise ValueError(f"refusing symlink consumer key target: {target}")
    _check_key_file_target(config, f"consumer {credential_id}", target)
    for entry in entries:
        if "key_file" in entry and resolve_api_key_file(entry["key_file"], CONFIG_PATH) == target:
            raise ValueError(f"consumer key file {target} is already used by {entry.get('id')!r}")

    snapshot = snapshot_writable_files([target])
    token = None
    try:
        if target.exists():
            _read_consumer_key_file(str(target))  # adopt only a valid private key file
        else:
            token = secrets.token_hex(32)
            write_api_key_file(target, token)
        entries.append(desired)
        _backup(CONFIG_PATH)
        _atomic_write(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    except Exception:
        restore_writable_files(snapshot)
        raise
    _validate_auth_after_write(snapshot)
    result = {**_consumer_summary(desired), "status": "created", "enables_client_auth": enables_client_auth}
    return {**result, "key": token} if reveal_key and token else result


def _managed_consumer_key(entry: dict, credential_id: str) -> Path | None:
    """The entry's key file if it is this credential's default, non-symlink path."""
    raw = entry.get("key_file")
    if not raw:
        return None
    intended = Path(str(raw)).expanduser()
    if not intended.is_absolute():
        intended = _resolve_target(CONFIG_PATH).parent / intended
    if intended != consumer_key_path(credential_id) or intended.is_symlink():
        return None
    return intended


@_write_transaction
def rotate_consumer_credential(credential_id: str) -> dict:
    """Replace a credential's managed key file with a new generated key.

    The old key stops authenticating as soon as the file is replaced. Only a
    valid key at the default ``secrets/consumers/<id>.key`` path is rotated;
    inline keys, external or symlinked files, and unusable files are refused.
    Returns the non-secret summary plus the new ``key`` for one-time display.
    """
    entries = _consumer_entries(load_config_full())
    entry = next((item for item in entries if item.get("id") == credential_id), None)
    if entry is None:
        raise KeyError(f"consumer credential {credential_id!r} not found")
    target = _managed_consumer_key(entry, credential_id)
    if target is None:
        raise ValueError(
            f"consumer credential {credential_id!r} does not use its managed key file; rotate it by hand"
        )
    status = _consumer_key_status(entry)
    if status != "ok":
        raise ValueError(
            f"consumer credential {credential_id!r} has an unusable key file ({status}); "
            "repair it with add, or revoke and re-add it"
        )
    snapshot = snapshot_writable_files([target])
    token = secrets.token_hex(32)
    try:
        write_api_key_file(target, token)
    except Exception:
        restore_writable_files(snapshot)
        raise
    _validate_auth_after_write(snapshot)
    return {**_consumer_summary(entry), "status": "rotated", "key": token}


@_write_transaction
def revoke_consumer_credential(credential_id: str) -> dict:
    """Remove a consumer credential entry and delete its managed key file.

    Only the default ``secrets/consumers/<id>.key`` file is deleted; a key file
    elsewhere, still referenced, or a symlink is kept. The config is rewritten
    first, so a failed delete leaves an unused orphan file and a warning.
    """
    config = load_config_full()
    entries = _consumer_entries(config)
    index = next((i for i, entry in enumerate(entries) if entry.get("id") == credential_id), None)
    if index is None:
        raise KeyError(f"consumer credential {credential_id!r} not found")
    entry = entries.pop(index)
    summary = _consumer_summary(entry)

    snapshot = snapshot_writable_files()
    _backup(CONFIG_PATH)
    _atomic_write(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    _validate_auth_after_write(snapshot, keep_client_auth=True)

    summary["key_file_deleted"] = False
    raw = entry.get("key_file")
    if raw:
        intended = _managed_consumer_key(entry, credential_id)
        target = resolve_api_key_file(raw, CONFIG_PATH)
        managed = intended is not None
        still_used = any(
            "key_file" in other and resolve_api_key_file(other["key_file"], CONFIG_PATH) == target
            for other in entries
        ) or any(path == target for _owner, path in _key_file_owners(config))
        if managed and not still_used and intended.is_file():
            try:
                intended.unlink()
                summary["key_file_deleted"] = True
            except OSError as exc:
                summary["warning"] = f"could not delete key file: {exc.strerror or exc}"
    return {**summary, "status": "revoked"}


# ── model config (model-info.json) ──────────────────────────────────────────


def load_model_info() -> dict:
    """Load the full model-info.json document."""
    if not MODEL_INFO_PATH.exists():
        return {"llm": []}
    with open(MODEL_INFO_PATH) as f:
        return json.load(f)


def _write_model_info(doc: dict) -> list[str]:
    """Write model-info.json to the live copy and optional local mirror.

    Returns the list of paths actually written.
    """
    text = json.dumps(doc, indent=2) + "\n"
    paths = [MODEL_INFO_PATH]
    _backup(MODEL_INFO_PATH)
    _atomic_write(MODEL_INFO_PATH, text)
    if MODEL_INFO_SOURCE_PATH and MODEL_INFO_SOURCE_PATH != MODEL_INFO_PATH:
        try:
            _atomic_write(MODEL_INFO_SOURCE_PATH, text)
            paths.append(MODEL_INFO_SOURCE_PATH)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not mirror model-info.json to %s: %s", MODEL_INFO_SOURCE_PATH, exc)
    return [str(p) for p in paths]


# Fields a cloud model entry may carry. Used to validate + order writes.
# NOTE: "enabled" is intentionally absent — runtime state lives in
# config.yaml model_overrides, not the machine-local catalog.
_MODEL_FIELDS = [
    "name", "provider", "provider_model_id", "omlx_id", "alias", "context",
    "max_output_tokens", "thinking", "thinking_levels", "thinking_format", "vision", "quirks",
    "system_instruction", "pricing", "pricing_status", "desc",
]
_PRICING_RATE_FIELDS = {
    "input", "output", "cache_read", "cache_write", "cache_write_1h", "reasoning",
}
_LOCAL_PROVIDERS = {"local", "omlx", "mlx"}


def _validated_pricing(pricing: object) -> dict:
    if not isinstance(pricing, dict) or not pricing:
        raise ValueError("metered pricing must be a non-empty object")
    unknown = set(pricing) - _PRICING_RATE_FIELDS
    if unknown:
        raise ValueError("unknown pricing field(s): " + ", ".join(sorted(unknown)))
    missing = {"input", "output"} - set(pricing)
    if missing:
        raise ValueError("metered pricing requires: " + ", ".join(sorted(missing)))
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        for value in pricing.values()
    ):
        raise ValueError("pricing values must be finite non-negative numbers")
    return dict(pricing)


def _apply_pricing_update(entry: dict, fields: dict, provider: str) -> None:
    """Apply the explicit metered/unmetered/unknown pricing policy."""
    if "pricing" not in fields and "pricing_status" not in fields:
        return

    status = fields.get("pricing_status")
    if status is not None:
        status = str(status).strip().lower()
    if status not in {None, "metered", "unmetered", "unknown"}:
        raise ValueError("pricing_status must be metered, unmetered, or unknown")

    supplied_pricing = fields.get("pricing") if "pricing" in fields else entry.get("pricing")
    if status == "unmetered":
        if provider not in _LOCAL_PROVIDERS:
            raise ValueError("unmetered pricing is only valid for local/oMLX models")
        if supplied_pricing:
            raise ValueError("unmetered models cannot also define token prices")
        entry.pop("pricing", None)
        entry["pricing_status"] = "unmetered"
        return
    if status == "unknown" or (status is None and "pricing" in fields and supplied_pricing is None):
        entry.pop("pricing", None)
        entry.pop("pricing_status", None)
        return

    if status == "metered" or "pricing" in fields:
        entry["pricing"] = _validated_pricing(supplied_pricing)
        entry.pop("pricing_status", None)


@_write_transaction
def upsert_model(name: str, **fields) -> dict:
    """Create or update a model entry in model-info.json.

    ``name`` is the gateway-facing model id and the dict key. Returns the
    written entry (masked: no secrets; models carry none). Writes the live
    catalog and optional machine-local mirror.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("model name is required")
    provider = (fields.get("provider") or "").strip().lower()
    if not provider:
        raise ValueError("provider is required")
    provider_model_id = (fields.get("provider_model_id") or "").strip()
    omlx_id = (fields.get("omlx_id") or "").strip()
    if provider in {"local", "omlx", "mlx"}:
        if not omlx_id and not provider_model_id:
            raise ValueError("omlx_id or provider_model_id is required for local/oMLX models")
    elif not provider_model_id:
        raise ValueError("provider_model_id is required")

    doc = load_model_info()
    llm = doc.get("llm", [])
    entry = next((e for e in llm if e.get("name") == name), None)
    if entry is None:
        entry = {"name": name}
        llm.append(entry)
        doc["llm"] = llm

    entry["name"] = name
    entry["provider"] = provider
    if provider_model_id:
        entry["provider_model_id"] = provider_model_id
    else:
        entry.pop("provider_model_id", None)
    if omlx_id:
        entry["omlx_id"] = omlx_id
    # JSON null means "leave unchanged" for optional admin fields. In
    # particular it must not discard a narrow explicit thinking_levels list and
    # replace it with the broad legacy fallback for the unchanged mode.
    thinking_changed = fields.get("thinking") is not None and fields["thinking"] != entry.get("thinking", "")
    if thinking_changed and "thinking_levels" not in fields:
        # A mode change without an explicit level list requests the safe legacy
        # fallback for that new mode rather than retaining stale capabilities.
        entry.pop("thinking_levels", None)
    for f in ("alias", "context", "max_output_tokens", "thinking",
              "thinking_levels", "thinking_format", "quirks", "system_instruction", "desc"):
        if f in fields and fields[f] is not None:
            entry[f] = fields[f]
    _apply_pricing_update(entry, fields, provider)
    validate_pricing_policy(entry)
    entry.update(normalize_thinking_capabilities(entry))
    if "vision" in fields and fields["vision"] is not None:
        entry["vision"] = bool(fields["vision"])
    # "enabled" is handled by set_model_enabled() writing config.yaml
    # model_overrides; it is never written to the model catalog.

    paths = _write_model_info(doc)
    return {"name": name, "entry": _model_summary(entry), "written_to": paths}


@_write_transaction
def delete_model(name: str) -> dict:
    """Remove a model entry by name. Also clears any runtime override."""
    name = (name or "").strip()
    doc = load_model_info()
    llm = doc.get("llm", [])
    before = len(llm)
    llm = [e for e in llm if e.get("name") != name]
    if len(llm) == before:
        raise KeyError(f"model {name!r} not found")
    doc["llm"] = llm
    paths = _write_model_info(doc)
    # Clean up any stale runtime override for the deleted model.
    config = load_config_full()
    overrides = config.get("model_overrides") or {}
    if name in overrides:
        del overrides[name]
        config["model_overrides"] = overrides
        _backup(CONFIG_PATH)
        _atomic_write(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
        paths.append(str(CONFIG_PATH))
    return {"name": name, "deleted": True, "written_to": paths}


@_write_transaction
def set_model_enabled(name: str, enabled: bool) -> dict:
    """Toggle a model's runtime enabled state in config.yaml model_overrides.

    Writes to the Git-ignored config.yaml (symlinked shared file), not
    model-info.json — so the toggle is hot, durable across deploys,
    and doesn't dirty the repo. The catalog stays the source of truth for
    *what models exist*; this is purely runtime on/off state.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("model name is required")
    # Verify the model exists in the catalog before recording an override.
    doc = load_model_info()
    if not any(e.get("name") == name for e in doc.get("llm", [])):
        raise KeyError(f"model {name!r} not found")

    config = load_config_full()
    overrides = config.setdefault("model_overrides", {}) or {}
    overrides[name] = {"enabled": bool(enabled)}
    config["model_overrides"] = overrides
    _backup(CONFIG_PATH)
    _atomic_write(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    return {"name": name, "enabled": bool(enabled), "written_to": [str(CONFIG_PATH)]}


def _model_summary(entry: dict) -> dict:
    """Mask-free summary of a model entry (models carry no secrets)."""
    return {
        k: entry.get(k) for k in _MODEL_FIELDS if k in entry
    }
