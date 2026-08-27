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
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from fastapi import HTTPException, Request


_PRINCIPAL_ID_RE = re.compile(r"^[a-z0-9-]{1,32}$")
_PROFILE_PERMISSIONS = {"profiles:read", "profiles:write", "profiles:invoke"}
_client_auth_required_latched = False


class AuthConfigError(ValueError):
    """Inbound authentication configuration is malformed or unreadable."""


class CredentialOverlapError(ValueError):
    """One token appears in more than one inbound credential class."""


@dataclass(frozen=True)
class AuthMode:
    client_auth_enabled: bool
    admin_auth_enabled: bool
    client_key_count: int
    admin_key_configured: bool
    unsafe_admin_without_key: bool
    writes_enabled: bool
    warning: str


@dataclass(frozen=True)
class ConsumerPrincipal:
    credential_id: str
    consumer: str
    permissions: frozenset[str]
    namespaces: frozenset[str]
    allow_direct_models: bool


@dataclass(frozen=True)
class AuthIdentity:
    kind: str
    principal: ConsumerPrincipal | None = None


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
    except ImportError:
        return {}
    try:
        value = auth_config()
    except Exception as exc:  # provider config parsing/schema errors fail closed
        raise AuthConfigError("authentication configuration is unreadable") from exc
    if not isinstance(value, dict):
        raise AuthConfigError("auth configuration must be an object")
    return value


def _config_keys(field: str) -> set[str]:
    raw = _config_auth().get(field)
    if raw is None:
        return set()
    if isinstance(raw, str):
        return _split_keys(raw)
    if isinstance(raw, list):
        if any(not isinstance(value, str) or not value.strip() for value in raw):
            raise AuthConfigError(f"auth.{field} must contain only non-empty strings")
        return {value.strip() for value in raw}
    raise AuthConfigError(f"auth.{field} must be a string or list of strings")


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
    global _client_auth_required_latched
    configured = bool(
        os.environ.get("MODEL_GATEWAY_CLIENT_KEYS", "").strip()
        or os.environ.get("MODEL_GATEWAY_CLIENT_KEYS_FILE", "").strip()
        or _config_keys("client_keys")
        or _config_auth().get("consumer_credentials")
    )
    if configured:
        _client_auth_required_latched = True
    return configured


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


def _read_consumer_key_file(raw_path: object) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("consumer credential key_file must be a path")
    from src import providers

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        config_target = Path(os.path.realpath(providers.CONFIG_PATH.expanduser()))
        path = config_target.parent / path
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("consumer credential key_file must be a regular mode-0600 file")
        if metadata.st_size > 65536:
            raise ValueError("consumer credential key_file exceeds 64 KiB")
        with os.fdopen(fd, "r", closefd=False) as handle:
            value = handle.read(65537).strip()
    finally:
        os.close(fd)
    if not value or len(value.encode("utf-8")) > 65536:
        raise ValueError("consumer credential key_file is empty or too large")
    return value


def _consumer_credentials() -> list[tuple[str, ConsumerPrincipal]]:
    raw = _config_auth().get("consumer_credentials")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("auth.consumer_credentials must be a list")
    result: list[tuple[str, ConsumerPrincipal]] = []
    credential_ids: set[str] = set()
    tokens: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("consumer credential entries must be objects")
        unknown = set(item) - {"id", "consumer", "key_file", "key", "permissions", "namespaces", "allow_direct_models"}
        if unknown:
            raise ValueError(f"consumer credential has unknown fields: {sorted(unknown)}")
        credential_id = item.get("id")
        consumer = item.get("consumer")
        if not isinstance(credential_id, str) or not _PRINCIPAL_ID_RE.fullmatch(credential_id):
            raise ValueError("consumer credential id is invalid")
        if credential_id in credential_ids:
            raise ValueError("consumer credential ids must be unique")
        credential_ids.add(credential_id)
        if not isinstance(consumer, str) or not _PRINCIPAL_ID_RE.fullmatch(consumer):
            raise ValueError("consumer id is invalid")
        has_file = "key_file" in item
        has_inline = "key" in item
        if has_file == has_inline:
            raise ValueError("consumer credential requires exactly one of key_file or key")
        if has_file:
            token = _read_consumer_key_file(item["key_file"])
        else:
            token = item.get("key")
            if not isinstance(token, str) or not token.strip() or len(token.encode("utf-8")) > 65536:
                raise ValueError("consumer credential inline key is invalid")
            token = token.strip()
        if token in tokens:
            raise ValueError("a consumer token may belong to only one principal")
        tokens.add(token)
        permissions = item.get("permissions")
        namespaces = item.get("namespaces")
        if (
            not isinstance(permissions, list)
            or any(not isinstance(value, str) or value not in _PROFILE_PERMISSIONS for value in permissions)
            or len(permissions) != len(set(permissions))
        ):
            raise ValueError("consumer credential permissions are invalid")
        if (
            not isinstance(namespaces, list) or not namespaces
            or any(not isinstance(value, str) or not _PRINCIPAL_ID_RE.fullmatch(value) for value in namespaces)
            or len(namespaces) != len(set(namespaces))
        ):
            raise ValueError("consumer credential namespaces are invalid")
        allow_direct = item.get("allow_direct_models", False)
        if not isinstance(allow_direct, bool):
            raise ValueError("consumer credential allow_direct_models must be boolean")
        result.append((token, ConsumerPrincipal(
            credential_id=credential_id,
            consumer=consumer,
            permissions=frozenset(permissions),
            namespaces=frozenset(namespaces),
            allow_direct_models=allow_direct,
        )))
    return result


def _federation_keys() -> set[str]:
    from src import providers

    config = providers._load_config()
    if not isinstance(config, dict):
        raise ValueError("config root must be an object")
    federation = config.get("federation")
    if federation is None:
        federation = {}
    if not isinstance(federation, dict):
        raise ValueError("federation configuration must be an object")
    peers = federation.get("peers")
    if peers is None:
        peers = {}
    if not isinstance(peers, dict):
        raise ValueError("federation peers must be an object")
    values: set[str] = set()
    for peer in peers.values():
        if not isinstance(peer, dict):
            raise ValueError("federation peer configuration must be an object")
        has_inline = isinstance(peer.get("api_key"), str) and bool(peer["api_key"].strip())
        has_file = bool(peer.get("api_key_file"))
        if has_inline == has_file:
            raise ValueError("federation peer requires exactly one credential source")
        token = peer["api_key"].strip() if has_inline else _read_consumer_key_file(peer["api_key_file"])
        if token in values:
            raise ValueError("federation peer tokens must be unique")
        values.add(token)
    return values


def validate_credential_separation() -> tuple[list[tuple[str, ConsumerPrincipal]], set[str], set[str]]:
    """Load inbound credentials and reject every cross-class token overlap."""
    global _client_auth_required_latched

    try:
        file_keys, file_valid = _key_file_values("MODEL_GATEWAY_CLIENT_KEYS_FILE")
        if not file_valid:
            raise AuthConfigError("configured client key file is unreadable or invalid")
        clients = _client_keys(file_keys=file_keys)
        consumers = _consumer_credentials()
        consumer_tokens = {token for token, _principal in consumers}
        admins = _admin_keys()
        federation = _federation_keys()
    except CredentialOverlapError:
        raise
    except AuthConfigError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise AuthConfigError("inbound authentication configuration is invalid") from exc
    classes = {
        "legacy client": clients,
        "consumer": consumer_tokens,
        "admin": admins,
        "federation": federation,
    }
    owners: dict[str, str] = {}
    for class_name, tokens in classes.items():
        for token in tokens:
            previous = owners.get(token)
            if previous is not None:
                raise CredentialOverlapError(
                    f"credential token overlaps {previous} and {class_name} classes"
                )
            owners[token] = class_name
    if clients or consumers:
        _client_auth_required_latched = True
    return consumers, clients, admins


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
    config_valid = True
    try:
        client_keys = _client_keys(file_keys=file_keys)
        consumer_count = len(_consumer_credentials())
        admin_keys = _admin_keys()
    except (OSError, UnicodeError, ValueError, TypeError, yaml.YAMLError):
        # Keep env-held admin recovery/status usable while reporting the broken
        # config. Config-held credentials remain unavailable until repaired.
        config_valid = False
        client_keys = _split_keys(os.environ.get("MODEL_GATEWAY_CLIENT_KEYS")) | file_keys
        consumer_count = 0
        admin_keys = _split_keys(os.environ.get("MODEL_GATEWAY_ADMIN_KEY"))
    unsafe_admin = _unsafe_admin_without_key_enabled()
    warning = ""
    if not config_valid:
        warning = "Inbound authentication config is malformed; config-held credentials are unavailable."
    elif not file_valid:
        warning = "/v1 client auth is misconfigured; the configured client key file is unreadable or invalid."
    elif not client_keys and not consumer_count:
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
        client_auth_enabled=bool(client_keys or consumer_count),
        admin_auth_enabled=bool(admin_keys),
        client_key_count=len(client_keys) + consumer_count,
        admin_key_configured=bool(admin_keys),
        unsafe_admin_without_key=unsafe_admin and not admin_keys,
        writes_enabled=admin_writes_enabled(),
        warning=warning,
    )


def _is_loopback_host(host: str) -> bool:
    host = host.strip().lower()
    return host == "localhost" or host == "::1" or host.startswith("127.")


def check_bind_safety(host: str) -> None:
    """Refuse a non-loopback bind unless /v1 client auth is configured.

    An unauthenticated gateway on 0.0.0.0 (or any non-loopback interface)
    exposes provider credentials as free model access to the whole network.
    Fail closed at startup; trusted private networks may opt out explicitly
    with MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_NONLOCAL=true.
    """
    if _is_loopback_host(host):
        return
    if _client_auth_configured():
        return
    if _truthy_env("MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_NONLOCAL"):
        return
    raise SystemExit(
        f"refusing to bind {host!r} without /v1 client auth: configure "
        "MODEL_GATEWAY_CLIENT_KEYS / MODEL_GATEWAY_CLIENT_KEYS_FILE / auth.client_keys, "
        "or set MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_NONLOCAL=true for a trusted private network."
    )


def require_client_auth(request: Request) -> AuthIdentity:
    """Authenticate an ordinary client and attach any consumer principal.

    Legacy/admin/anonymous behavior remains unchanged for ordinary routes.
    Identity-aware consumer credentials are additionally accepted; their direct
    model permission is enforced by the inference handlers after parsing the
    requested selector.
    """
    try:
        consumers, keys, admins = validate_credential_separation()
    except (OSError, UnicodeError, ValueError):
        raise HTTPException(status_code=503, detail="model-gateway client authentication is misconfigured")

    token = _extract_token(request)
    for credential, principal in consumers:
        if secrets.compare_digest(token, credential):
            identity = AuthIdentity("consumer", principal)
            request.state.auth_identity = identity
            request.state.consumer_principal = principal
            return identity
    if _matches(token, keys):
        identity = AuthIdentity("legacy")
        request.state.auth_identity = identity
        return identity
    if _matches(token, admins):
        identity = AuthIdentity("admin")
        request.state.auth_identity = identity
        return identity
    if not keys and not consumers:
        if _client_auth_configured() or _client_auth_required_latched:
            raise HTTPException(status_code=503, detail="model-gateway client authentication is misconfigured")
        identity = AuthIdentity("anonymous")
        request.state.auth_identity = identity
        return identity
    raise HTTPException(status_code=401, detail="Missing or invalid model-gateway client key")


def require_consumer_auth(request: Request, *, permission: str, namespace: str) -> ConsumerPrincipal:
    """Require an identity-aware principal with namespace and permission grants."""
    identity = require_client_auth(request)
    principal = identity.principal
    if identity.kind != "consumer" or principal is None:
        raise HTTPException(status_code=401, detail="A consumer credential is required")
    if namespace not in principal.namespaces or permission not in principal.permissions:
        raise HTTPException(status_code=403, detail="Consumer principal is not authorized for this namespace")
    return principal


def require_admin_auth(request: Request) -> None:
    """Protect /admin/api/*, preserving env-key recovery for malformed config."""
    token = _extract_token(request)
    env_keys = _split_keys(os.environ.get("MODEL_GATEWAY_ADMIN_KEY"))
    if _matches(token, env_keys):
        try:
            validate_credential_separation()
        except CredentialOverlapError:
            raise HTTPException(status_code=503, detail="model-gateway authentication is misconfigured")
        except (AuthConfigError, OSError, UnicodeError, yaml.YAMLError):
            # An env-held admin key must remain usable to repair/reload a
            # malformed runtime config. No config-held identity can authenticate
            # while that file is unreadable.
            return
        return
    try:
        _consumers, _clients, keys = validate_credential_separation()
    except (OSError, UnicodeError, ValueError, yaml.YAMLError):
        raise HTTPException(status_code=503, detail="model-gateway authentication is misconfigured")
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