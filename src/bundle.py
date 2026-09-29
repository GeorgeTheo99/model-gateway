"""Portable, secret-free gateway bundles for moving to another machine.

A bundle is a gzip tarball with ``manifest.json``, ``config.yaml``,
``model-info.json`` and, when present, ``consumer-profiles-registry.json``.

Export never includes secrets: it keeps only known content sections (so
``auth`` and any unknown section are dropped), strips every inline ``api_key``
and auth-like ``default_headers``, and keeps ``api_key_file`` references as
home-relative paths so the target machine supplies its own key files.
Machine-layout sections (``auth``, ``federation``, ``exports``, ``profiles``)
always stay with the target on import.

Bundles are trusted input: an imported provider ``base_url`` receives the key
in its ``api_key_file`` on this machine. Review ``import --dry-run`` first.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import tarfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from src import config_io, profiles, providers
from src.secret_files import resolve_api_key_file

BUNDLE_FORMAT = 1
_MAX_MEMBER_BYTES = 16 * 1024 * 1024
_MEMBERS = ("manifest.json", "config.yaml", "model-info.json", "consumer-profiles-registry.json")
_MACHINE_SECTIONS = ("auth", "federation", "exports", "profiles")
_CONTENT_SECTIONS = ("providers", "workspaces", "pools", "models", "model_overrides", "model_fallbacks")
_SECRET_HEADER = re.compile(r"auth|key|token|secret|cookie|session", re.IGNORECASE)


def _home_relative(value: object) -> object:
    home = str(Path.home())
    if isinstance(value, dict):
        return {key: _home_relative(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_home_relative(item) for item in value]
    if isinstance(value, str) and (value == home or value.startswith(home + "/")):
        return "~" + value[len(home):]
    return value


def portable_config(config: dict) -> tuple[dict, list[str]]:
    """Return a secret-free, home-relative copy of ``config`` and what was dropped."""
    doc = {key: copy.deepcopy(value) for key, value in config.items() if key in _CONTENT_SECTIONS}
    removed = [str(key) for key in config if key not in _CONTENT_SECTIONS]

    def strip(node: object, where: str) -> None:
        if isinstance(node, dict):
            for key in list(node):
                path = f"{where}.{key}" if where else str(key)
                if key == "api_key":
                    node.pop(key)
                    removed.append(path)
                elif key == "default_headers" and isinstance(node[key], dict):
                    for header in [name for name in node[key] if _SECRET_HEADER.search(str(name))]:
                        node[key].pop(header)
                        removed.append(f"{path}.{header}")
                else:
                    strip(node[key], path)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                strip(item, f"{where}[{index}]")

    strip(doc, "")
    return _home_relative(doc), removed


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def credential_urls(config: dict) -> list[str]:
    """Paths of URL values in a portable config that embed userinfo or a query.

    Export keeps these byte-for-byte so an import restores the exact route, so
    a bundle containing one is only safe to write locally, not to serve.
    """
    found: list[str] = []

    def walk(node: object, where: str) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                walk(item, f"{where}.{key}" if where else str(key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{where}[{index}]")
        elif isinstance(node, str) and "://" in node:
            try:
                parts = urlsplit(node)
            except ValueError:
                found.append(where)
                return
            if parts.username or parts.password or parts.query or parts.fragment:
                found.append(where)

    walk(config, "")
    return found


def build_bundle() -> tuple[dict, bytes, list[str]]:
    """Build a bundle in memory: (manifest, gzip tar bytes, credential URL paths)."""
    config, removed = portable_config(config_io.load_config_full())
    files = {
        "config.yaml": yaml.safe_dump(config, sort_keys=False, default_flow_style=False).encode(),
        "model-info.json": (json.dumps(config_io.load_model_info(), indent=2) + "\n").encode(),
    }
    registry = profiles.registry_path()
    if registry.exists():
        profiles._load(registry)  # refuse to export a corrupt registry
        files["consumer-profiles-registry.json"] = registry.read_bytes()
    manifest = {
        "format": BUNDLE_FORMAT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {name: _sha256(data) for name, data in files.items()},
        "removed": removed,
    }
    files = {"manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(), **files}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o600, int(time.time())
            tar.addfile(info, io.BytesIO(data))
    return manifest, buffer.getvalue(), credential_urls(config)


def export_bundle(out: Path) -> dict:
    """Write a bundle to ``out`` (mode 0600) and return its manifest."""
    manifest, data, _credential_urls = build_bundle()
    out = out.expanduser()
    if out.is_symlink() or out.exists():
        raise ValueError(f"refusing to overwrite existing bundle: {out}")
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return manifest


def read_bundle(path: Path) -> tuple[dict, dict[str, bytes]]:
    """Read and verify a bundle; return (manifest, files) without writing anything."""
    files: dict[str, bytes] = {}
    with tarfile.open(path.expanduser(), mode="r:gz") as tar:
        for member in tar:
            if member.name not in _MEMBERS or not member.isreg() or member.name in files:
                raise ValueError(f"unexpected bundle member: {member.name!r}")
            if member.size > _MAX_MEMBER_BYTES:
                raise ValueError(f"bundle member too large: {member.name}")
            files[member.name] = tar.extractfile(member).read()
    try:
        manifest = json.loads(files.pop("manifest.json"))
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("bundle manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError("unsupported bundle format")
    if manifest.get("files") != {name: _sha256(data) for name, data in files.items()}:
        raise ValueError("bundle contents do not match its manifest")
    if "config.yaml" not in files or "model-info.json" not in files:
        raise ValueError("bundle must contain config.yaml and model-info.json")
    return manifest, files


def _missing_key_files(config: dict) -> list[dict]:
    missing = []
    for section in ("providers", "workspaces"):
        entries = config.get(section) or {}
        for name, block in entries.items() if isinstance(entries, dict) else ():
            if isinstance(block, dict) and block.get("api_key_file"):
                target = resolve_api_key_file(block["api_key_file"], providers.CONFIG_PATH)
                if not target.is_file():
                    missing.append({"owner": f"{section}.{name}", "api_key_file": str(target)})
    return missing


def _provider_endpoints(config: dict) -> list[dict]:
    """Where each imported provider would send this machine's key, for review."""
    rows = []
    for section, url_field in (("providers", "base_url"), ("workspaces", "workspace_url")):
        entries = config.get(section) or {}
        for name, block in entries.items() if isinstance(entries, dict) else ():
            if isinstance(block, dict):
                rows.append({"owner": f"{section}.{name}", "url": block.get(url_field),
                             "api_key_file": block.get("api_key_file")})
    return rows


def _target_has_content(config: dict, registry: Path) -> bool:
    if any(value for key, value in config.items() if key not in _MACHINE_SECTIONS):
        return True
    if (config_io.load_model_info().get("llm") or []):
        return True
    return registry.exists() and bool(profiles._load(registry)["namespaces"])


def _write_rollback_file(path: Path, snapshot: dict[Path, str | None]) -> None:
    """Atomically save pre-import file contents so the operator CLI can undo a failed start."""
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({str(key): value for key, value in snapshot.items()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _restore(snapshot: dict[Path, str | None]) -> None:
    """Restore files under the config lock, then the registry lock (the order import uses)."""
    with config_io.config_write_lock(config_io.CONFIG_PATH), profiles._file_lock(exclusive=True):
        config_io.restore_writable_files(snapshot)


def restore_rollback(path: Path) -> list[str]:
    """Restore the files saved by an import's rollback file, then delete it."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        info = os.fstat(handle.fileno())
        if info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise ValueError(f"refusing unsafe rollback file: {path}")
        saved = json.load(handle)
    allowed = {str(key) for key in config_io.snapshot_writable_files([profiles.registry_path()])}
    if not isinstance(saved, dict) or any(
        key not in allowed or not (value is None or isinstance(value, str)) for key, value in saved.items()
    ):
        raise ValueError("rollback file is invalid or names files this gateway does not manage")
    _restore({Path(key): value for key, value in saved.items()})
    path.unlink()
    return sorted(saved)


def _validate_loaded_gateway() -> None:
    """The checks startup and /admin/api/reload apply to a new registry."""
    from src.auth import validate_credential_separation
    from src.server import _validate_vision_fallback_policy

    providers.reload()
    providers.snapshot_registry()
    _validate_vision_fallback_policy(log_policy=False)
    validate_credential_separation()


def import_bundle(path: Path, *, force: bool = False, dry_run: bool = False,
                  rollback_file: Path | None = None) -> dict:
    """Replace this gateway's providers, catalog and profiles from a bundle.

    Refuses to replace existing content without ``force``. Backs up, then
    validates like an admin reload and restores every file on failure.
    ``rollback_file`` also saves the previous contents for :func:`restore_rollback`.
    """
    manifest, files = read_bundle(path)
    try:
        incoming = yaml.safe_load(files["config.yaml"]) or {}
    except yaml.YAMLError as exc:
        raise ValueError("bundle config.yaml is not valid YAML") from exc
    if not isinstance(incoming, dict) or set(incoming) - set(_CONTENT_SECTIONS):
        raise ValueError("bundle config must be a mapping of known content sections only")
    catalog = json.loads(files["model-info.json"])
    if not isinstance(catalog, dict) or not isinstance(catalog.get("llm"), list):
        raise ValueError("bundle model-info.json must be an object with an llm list")
    registry_doc = None
    if "consumer-profiles-registry.json" in files:
        registry_doc = json.loads(files["consumer-profiles-registry.json"])

    with config_io.config_write_lock(config_io.CONFIG_PATH):
        current = config_io.load_config_full()
        registry = profiles.registry_path()
        if _target_has_content(current, registry) and not force:
            raise ValueError("this gateway already has providers, models, or profiles; re-run with --force to replace them")
        merged = {**incoming, **{key: current[key] for key in _MACHINE_SECTIONS if key in current}}
        summary = {
            "created_at": manifest.get("created_at"),
            "providers": sorted((incoming.get("providers") or {}).keys()),
            "models": len(catalog["llm"]),
            "profile_namespaces": sorted((registry_doc or {}).get("namespaces", {}).keys()),
            "provider_endpoints": _provider_endpoints(merged),
            "missing_key_files": _missing_key_files(merged),
            "dry_run": dry_run,
        }
        if dry_run:
            return summary

        paths = [registry] if registry_doc is not None else []
        snapshot = config_io.snapshot_writable_files(paths)
        if rollback_file is not None:
            _write_rollback_file(rollback_file, snapshot)
        try:
            config_io._backup(config_io.CONFIG_PATH)
            config_io._atomic_write(
                config_io.CONFIG_PATH, yaml.safe_dump(merged, sort_keys=False, default_flow_style=False))
            config_io._write_model_info(catalog)
            if registry_doc is not None:
                registry.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with profiles._file_lock(exclusive=True):
                    config_io._backup(registry)
                    profiles._write_atomic(registry, registry_doc)
                    profiles._load(registry)
            _validate_loaded_gateway()
        except BaseException:  # includes KeyboardInterrupt from an operator Ctrl-C
            _restore(snapshot)
            raise
        finally:
            providers.reload()
    return summary
