"""Inbound gateway authentication helpers.

Auth is intentionally opt-in for backward compatibility with the existing local
clients. Set MODEL_GATEWAY_CLIENT_KEYS to protect /v1/* and
MODEL_GATEWAY_ADMIN_KEY to protect /admin/api/*. Legacy CLOUD_GATEWAY_* names
remain accepted during migration. Both may also be configured in config.yaml
under an ``auth`` section (``admin_keys`` / ``client_keys`` lists, or a
comma-separated string); env vars take precedence over config.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class AuthMode:
    client_auth_enabled: bool
    admin_auth_enabled: bool
    client_key_count: int
    admin_key_configured: bool
    unsafe_admin_without_key: bool
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


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _client_keys() -> set[str]:
    # Env takes precedence so operators can override without editing config,
    # but config.yaml is the canonical home for secrets (it is gitignored and
    # already holds provider API keys).
    keys = _split_keys(_env("MODEL_GATEWAY_CLIENT_KEYS", "CLOUD_GATEWAY_CLIENT_KEYS"))
    keys |= _config_keys("client_keys")
    return keys


def _admin_keys() -> set[str]:
    # Allow a future comma-separated admin key list without changing the env name.
    keys = _split_keys(_env("MODEL_GATEWAY_ADMIN_KEY", "CLOUD_GATEWAY_ADMIN_KEY"))
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
    return _truthy_env("MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN") or _truthy_env(
        "CLOUD_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN"
    )


def auth_mode() -> AuthMode:
    client_keys = _client_keys()
    admin_keys = _admin_keys()
    unsafe_admin = _unsafe_admin_without_key_enabled()
    warning = ""
    if not client_keys:
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
        warning=warning,
    )


def require_client_auth(request: Request) -> None:
    """Protect /v1/* when MODEL_GATEWAY_CLIENT_KEYS is configured.

    Admin keys are accepted as client keys too, so the admin UI can inspect
    existing /v1 debug endpoints with one token.
    """
    keys = _client_keys()
    if not keys:
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