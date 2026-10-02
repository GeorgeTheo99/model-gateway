"""Machine-local discovery file telling consumers how to reach this gateway.

The gateway rewrites ``endpoint.json`` at startup and after an admin reload,
so every deploy path (installer, CI, restart) keeps it current. It holds the
loopback URL, the alias export path, and credential key-file *paths*, never
key values. ``MODEL_GATEWAY_ENDPOINT_FILE`` overrides the location; an empty
value disables it (e.g. a second gateway on the same machine).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src import providers
from src.secret_files import resolve_api_key_file
from src.version import CAPABILITIES, VERSION

ENDPOINT_VERSION = 1

log = logging.getLogger("model-gateway")


def endpoint_path() -> Path | None:
    configured = os.environ.get("MODEL_GATEWAY_ENDPOINT_FILE")
    if configured is not None:
        configured = configured.strip()
        return Path(configured).expanduser() if configured else None
    return Path.home() / "Library" / "Application Support" / "model-gateway" / "endpoint.json"


def _connect_origin() -> tuple[str, int]:
    host = os.environ.get("MODEL_GATEWAY_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("MODEL_GATEWAY_PORT", "9111"))
    # A wildcard bind is reachable locally through loopback.
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}", port


def endpoint_document() -> dict:
    config = providers._load_config()
    origin, port = _connect_origin()
    exports = config.get("exports") if isinstance(config.get("exports"), dict) else {}
    aliases = exports.get("model_aliases")
    auth = config.get("auth") if isinstance(config.get("auth"), dict) else {}
    consumers = {}
    for entry in auth.get("consumer_credentials") or []:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry.get("key_file"):
            consumers[entry["id"]] = {
                "consumer": entry.get("consumer"),
                "namespaces": list(entry.get("namespaces") or []),
                "permissions": list(entry.get("permissions") or []),
                "allow_direct_models": bool(entry.get("allow_direct_models", False)),
                "key_file": str(resolve_api_key_file(entry["key_file"], providers.CONFIG_PATH)),
            }
    client_key_file = os.environ.get("MODEL_GATEWAY_CLIENT_KEYS_FILE", "").strip()
    return {
        "version": ENDPOINT_VERSION,
        "service": "model-gateway",
        "gateway_version": VERSION,
        "capabilities": list(CAPABILITIES),
        "base_url": f"{origin}/v1",
        "health_url": f"{origin}/health",
        "port": port,
        "model_aliases": str(Path(str(aliases)).expanduser()) if aliases else None,
        "client_key_file": str(Path(client_key_file).expanduser()) if client_key_file else None,
        "consumers": consumers,
    }


def write_endpoint_file() -> Path | None:
    """Atomically write the discovery file (mode 0600); return its path or None if disabled."""
    path = endpoint_path()
    if path is None:
        return None
    text = json.dumps(endpoint_document(), indent=2) + "\n"
    if path.is_symlink():
        raise OSError(f"refusing symlink endpoint file: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        os.chmod(path, 0o600)
        return path
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.unlink(missing_ok=True)
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def refresh() -> None:
    """Best-effort :func:`write_endpoint_file`; a failure never blocks serving."""
    try:
        path = write_endpoint_file()
        if path is not None:
            log.info("consumer discovery file: %s", path)
    except Exception as exc:  # noqa: BLE001
        log.warning("consumer discovery file not written: %s", exc)
