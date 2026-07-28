"""Inbound gateway authentication helpers.

Auth is intentionally opt-in for trusted local clients. Set
MODEL_GATEWAY_CLIENT_KEYS (or MODEL_GATEWAY_CLIENT_KEYS_FILE) to protect
/v1/* and MODEL_GATEWAY_ADMIN_KEY to protect /admin/api/*. Both may also be configured in config.yaml under an
``auth`` section (``admin_keys`` / ``client_keys`` lists, or a comma-separated
string); env vars take precedence over config.
"""

from __future__ import annotations

import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class AuthMode:
    client_auth_enabled: bool
    admin_auth_enabled: bool
    client_key_count: int
    admin_key_configured: bool
    unsafe_admin_without_key: bool
    writes_enabled: bool
    warning: str


def _split_keys(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def _config_auth() -> dict:
    """Read the ``auth`` section from config.yaml (reloaded on /admin/api/reload).

    Importing lazily keeps :mod:`src.auth` importable in test contexts that
    stub the provider registry.
    """
    try:
        from src.providers import auth_config
        return auth_config()
    except Exception:
        return {}


def _config_keys(field: str) -> set[str]:
    raw = _config_auth().get(field)
    if isinstance(raw, str):
        return _split_keys(raw)
    if isinstance(raw, list):
        return {str(k).strip() for k in raw if str(k).strip()}
    return set()


def _key_file_values(env_name: str) -> tuple[set[str], bool]:
    configured = os.environ.get(env_name, "").strip()
    if not configured:
        return set(), True
    path = Path(configured).expanduser()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError:
        return set(), False
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
            return set(), False
        if metadata.st_size > 65536:
            return set(), False
        with os.fdopen(fd, "r", closefd=False) as handle:
            raw = handle.read(65537)
    except (OSError, UnicodeError):
        return set(), False
    finally:
        os.close(fd)
    if len(raw.encode("utf-8")) > 65536:
        return set(), False
    keys = _split_keys(raw.replace("\n", ","))
    return keys, bool(keys)


def _client_auth_configured() -> bool:
    return bool(
        os.environ.get("MODEL_GATEWAY_CLIENT_KEYS", "").strip()
        or os.environ.get("MODEL_GATEWAY_CLIENT_KEYS_FILE", "").strip()
        or _config_keys("client_keys")
    )


def _client_keys(*, file_keys: set[str] | None = None) -> set[str]:
    # Runtime key files keep generated per-host credentials out of LaunchAgent
    # plists while config.yaml remains supported for portable installations.
    keys = _split_keys(os.environ.get("MODEL_GATEWAY_CLIENT_KEYS"))
    if file_keys is None:
        file_keys, _valid = _key_file_values("MODEL_GATEWAY_CLIENT_KEYS_FILE")
    keys |= file_keys
    keys |= _config_keys("client_keys")
    return keys


def _admin_keys() -> set[str]:
    # Allow a future comma-separated admin key list without changing the env name.
    keys = _split_keys(os.environ.get("MODEL_GATEWAY_ADMIN_KEY"))
    keys |= _config_keys("admin_keys")
    return keys


def _extract_token(request: Request) -> str:
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if auth:
        return auth
    # Anthropic-compatible clients commonly use x-api-key. Accept a few common
    # aliases so enabling auth does not force client-specific gateway patches.
    for header in ("x-api-key", "api-key", "x-gateway-key"):
        value = request.headers.get(header, "").strip()
        if value:
            return value
    return ""


def _matches(token: str, keys: set[str]) -> bool:
    return bool(token) and any(secrets.compare_digest(token, key) for key in keys)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _unsafe_admin_without_key_enabled() -> bool:
    return _truthy_env("MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN")


def admin_writes_enabled() -> bool:
    """Whether the admin UI may mutate providers/models/config.

    Defaults to False (read-only dashboard). Enable with
    MODEL_GATEWAY_ADMIN_WRITES=true for explicit write access. This gates the
    writeable management endpoints and hides the management UI forms.
    """
    return _truthy_env("MODEL_GATEWAY_ADMIN_WRITES")


def auth_mode() -> AuthMode:
    file_keys, file_valid = _key_file_values("MODEL_GATEWAY_CLIENT_KEYS_FILE")
    client_keys = _client_keys(file_keys=file_keys)
    admin_keys = _admin_keys()
    unsafe_admin = _unsafe_admin_without_key_enabled()
    warning = ""
    if not file_valid:
        warning = "/v1 client auth is misconfigured; the configured client key file is unreadable or invalid."
    elif not client_keys:
        if _client_auth_configured():
            warning = "/v1 client auth is misconfigured; the configured client key source is unreadable or invalid."
        else:
            warning = "/v1 client auth is disabled; set MODEL_GATEWAY_CLIENT_KEYS before exposing beyond trusted local clients."
    if not admin_keys:
        if unsafe_admin:
            admin_warning = "/admin/api is unauthenticated because MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN is enabled."
        else:
            admin_warning = "/admin/api is locked; set MODEL_GATEWAY_ADMIN_KEY or explicitly enable MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN for local dev."
        warning = f"{warning} {admin_warning}".strip()
    return AuthMode(
        client_auth_enabled=bool(client_keys),
        admin_auth_enabled=bool(admin_keys),
        client_key_count=len(client_keys),
        admin_key_configured=bool(admin_keys),
        unsafe_admin_without_key=unsafe_admin and not admin_keys,
        writes_enabled=admin_writes_enabled(),
        warning=warning,
    )


def require_client_auth(request: Request) -> None:
    """Protect /v1/* when MODEL_GATEWAY_CLIENT_KEYS is configured.

    Admin keys are accepted as client keys too, so the admin UI can inspect
    existing /v1 debug endpoints with one token.
    """
    file_keys, file_valid = _key_file_values("MODEL_GATEWAY_CLIENT_KEYS_FILE")
    if not file_valid:
        raise HTTPException(
            status_code=503,
            detail="model-gateway client authentication is misconfigured",
        )
    keys = _client_keys(file_keys=file_keys)
    if not keys:
        if _client_auth_configured():
            raise HTTPException(
                status_code=503,
                detail="model-gateway client authentication is misconfigured",
            )
        return
    token = _extract_token(request)
    if _matches(token, keys) or _matches(token, _admin_keys()):
        return
    raise HTTPException(status_code=401, detail="Missing or invalid model-gateway client key")


def require_admin_auth(request: Request) -> None:
    """Protect /admin/api/*.

    Admin APIs fail closed unless MODEL_GATEWAY_ADMIN_KEY is configured. For
    local development only, set MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN=true.
    """
    keys = _admin_keys()
    if not keys:
        if _unsafe_admin_without_key_enabled():
            return
        raise HTTPException(
            status_code=401,
            detail="model-gateway admin API is locked until MODEL_GATEWAY_ADMIN_KEY is set",
        )
    if _matches(_extract_token(request), keys):
        return
    raise HTTPException(status_code=401, detail="Missing or invalid model-gateway admin key")


def require_admin_writes() -> None:
    """Gate mutating admin endpoints behind MODEL_GATEWAY_ADMIN_WRITES.

    Raises 403 when writes are disabled (the default read-only dashboard).
    Call after require_admin_auth so auth is still enforced first.
    """
    if not admin_writes_enabled():
        raise HTTPException(
            status_code=403,
            detail="admin writes are disabled; set MODEL_GATEWAY_ADMIN_WRITES=true to manage providers/models",
        )