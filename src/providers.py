"""Provider registry — resolve model name to endpoint + credentials."""

import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml

from src import catalog

log = logging.getLogger("model-gateway")


# model-info.json is the gateway-owned source of truth. Allow the path to be
# supplied via env (set by the launcher/deploy); fall back to the checkout-local
# catalog for tests and local dev.
_DEFAULT_MODEL_INFO = Path(__file__).resolve().parents[1] / "model-info.json"
MODEL_INFO_PATH = Path(
    os.environ.get("MODEL_GATEWAY_MODEL_INFO") or str(_DEFAULT_MODEL_INFO)
).resolve()
# Optional source-repo path for durable model edits. When set, model write
# operations mirror edits here (pending commit) in addition to the deployed
# MODEL_INFO_PATH, so changes survive deploys once committed. The deployed
# copy is overwritten by git checkout on each deploy; the source copy is not.
_DEFAULT_MODEL_INFO_SOURCE = Path.home() / "local_code" / "model-gateway" / "model-info.json"
MODEL_INFO_SOURCE_PATH = (
    Path(
        os.environ.get("MODEL_GATEWAY_MODEL_INFO_SOURCE")
        or str(_DEFAULT_MODEL_INFO_SOURCE)
    ).resolve()
    if os.environ.get("MODEL_GATEWAY_MODEL_INFO_SOURCE")
    or _DEFAULT_MODEL_INFO_SOURCE.exists()
    else None
)
CONFIG_PATH = Path(
    os.environ.get("MODEL_GATEWAY_CONFIG")
    or Path(__file__).resolve().parents[1] / "config" / "config.yaml"
).resolve()

_DEFAULT_OMLX_BASE_URL = os.environ.get("MODEL_GATEWAY_OMLX_BASE_URL", "http://localhost:9110/v1")
_DEFAULT_OMLX_API_KEY = os.environ.get("MODEL_GATEWAY_OMLX_API_KEY", "omlx")


@dataclass
class ProviderInfo:
    provider: str
    base_url: str
    api_key: str
    provider_model_id: str
    protocol: str = "openai"  # "openai" (Chat Completions) or "anthropic" (Messages API)
    # Path appended to base_url for chat/messages requests. None => server default
    # ("/chat/completions" or "/messages"). "" => base_url is already a complete
    # invocation URL (e.g. endpoint_style: invocations providers).
    endpoint_suffix: str | None = None
    # Provider quirk flags from config (e.g. "no_stream_options",
    # "no_reasoning_params"). Generic mechanism; which providers need which
    # quirks is runtime config, not code.
    quirks: frozenset = frozenset()
    context: int = 0
    max_output_tokens: int = 0
    thinking: str = ""  # "", "optional", or "always"
    thinking_format: str = ""  # optional explicit gateway normalization format
    system_instruction: str = ""
    vision: bool = False  # authoritative: True if the model can natively handle image inputs
    pricing: dict = None  # $/Mtok rates: input, output, cache_read?, cache_write?, reasoning?


_config: dict | None = None
_models: dict[str, dict] | None = None

# OAuth token auto-refresh via an external CLI token cache (opt-in per provider
# with `auth_refresh:` in config.yaml). See refresh_oauth_token().
_AUTH_REFRESH_MIN_INTERVAL = 60.0  # seconds between CLI refresh attempts per provider
_token_refresh_locks: dict[str, "asyncio.Lock"] = {}
_last_token_refresh_attempt: dict[str, float] = {}

# Provider synonym table lives in ``src.catalog`` (the single source shared with
# the downstream catalog generator). ``tests/test_catalog.py`` guards drift.


def _canonical_provider(provider: str | None) -> str:
    return catalog.canonical_provider(provider)


def _find_provider_entry(config: dict, provider: str) -> tuple[str, dict] | None:
    """Return (config_key, entry) for a provider, matching canonical synonyms.

    Searches both the ``providers:`` section and the ``workspaces:`` alias
    section (same schema; "workspace" is the operator-facing name for a
    Databricks upstream in the pools design).
    """
    for section in ("providers", "workspaces"):
        providers = config.get(section, {}) or {}
        if not isinstance(providers, dict):
            continue
        direct = providers.get(provider)
        if direct:
            return provider, direct
        for key, value in providers.items():
            if _canonical_provider(key) == provider:
                return key, value or {}
    return None


def _pools(config: dict) -> dict[str, list[str]]:
    """Ordered workspace-failover pools: pool name -> [provider names]."""
    raw = config.get("pools") or {}
    if not isinstance(raw, dict):
        return {}
    pools: dict[str, list[str]] = {}
    for name, members in raw.items():
        if isinstance(members, str):
            members = [members]
        if isinstance(members, list):
            cleaned = [str(m).strip() for m in members if str(m).strip()]
            if cleaned:
                pools[str(name)] = cleaned
    return pools


def _pool_members(entry: dict, config: dict) -> list[str]:
    """Ordered candidate providers for a model entry.

    A ``pool:`` reference expands to the pool's member list; a plain
    ``provider:`` is a single-member pool. Unknown pool names fall back to the
    entry's provider so a config typo degrades to current behavior.
    """
    pool_name = entry.get("pool")
    if pool_name:
        members = _pools(config).get(str(pool_name))
        if members:
            return [_canonical_provider(m) for m in members]
        log.warning("Model %r references unknown pool %r", entry.get("name"), pool_name)
    return [_canonical_provider(entry.get("provider", "local"))]


def _resolve_provider_config(config: dict, provider: str) -> dict:
    found = _find_provider_entry(config, provider)
    return found[1] if found else {}


def _provider_defaults(provider: str) -> dict:
    """Built-in provider defaults for local backends managed on this machine."""
    if provider == "omlx":
        return {
            "base_url": _DEFAULT_OMLX_BASE_URL,
            "api_key": _DEFAULT_OMLX_API_KEY,
            "protocol": "openai",
        }
    return {}


def _provider_env_prefix(provider: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in provider.upper())
    return f"MODEL_GATEWAY_PROVIDER_{normalized}"


def _env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _provider_env_config(provider: str) -> dict:
    """Provider config from environment variables.

    Generic shape:
      MODEL_GATEWAY_PROVIDER_<PROVIDER>_{BASE_URL,API_KEY,PROTOCOL,ENABLED}

    Databricks additionally accepts the standard workspace env vars:
      DATABRICKS_HOST + DATABRICKS_TOKEN, or DATABRICKS_SERVING_BASE_URL.
    """
    prefix = _provider_env_prefix(provider)
    env_config: dict = {}
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    api_key = os.environ.get(f"{prefix}_API_KEY")
    protocol = os.environ.get(f"{prefix}_PROTOCOL")
    enabled = _env_bool(os.environ.get(f"{prefix}_ENABLED"))

    if provider == "databricks":
        db_base_url = os.environ.get("DATABRICKS_SERVING_BASE_URL")
        db_host = os.environ.get("DATABRICKS_HOST")
        db_token = os.environ.get("DATABRICKS_TOKEN")
        if not base_url:
            if db_base_url:
                base_url = db_base_url
            elif db_host:
                base_url = f"{db_host.rstrip('/')}/serving-endpoints"
        if not api_key and db_token:
            api_key = db_token
        if enabled is None and (base_url or api_key):
            enabled = True
        protocol = protocol or "openai"

    if base_url is not None:
        env_config["base_url"] = base_url
    if api_key is not None:
        env_config["api_key"] = api_key
    if protocol is not None:
        env_config["protocol"] = protocol
    if enabled is not None:
        env_config["enabled"] = enabled
    return env_config


def _effective_provider_config(config: dict, provider: str) -> dict:
    """Provider config with defaults + config.yaml + env overrides."""
    effective = _provider_defaults(provider)
    effective.update(_resolve_provider_config(config, provider))
    effective.update(_provider_env_config(provider))
    return effective


def _load_config() -> dict:
    global _config
    if _config is not None:
        return _config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            _config = yaml.safe_load(f) or {}
    else:
        _config = {}
    return _config


def _entry_routable_ids(entry: dict) -> list[str]:
    """Return every gateway-facing identifier for a catalog entry."""
    return catalog.entry_routable_ids(entry)


def _load_models() -> dict[str, dict]:
    """Load routable models from model-info.json (keyed by name/alias/id).

    The catalog + config.yaml ``models:`` overlay merge lives in
    :func:`src.catalog.load_catalog_entries` and is shared with the downstream
    catalog generator so the router and the generator can never drift.
    """
    global _models
    if _models is not None:
        return _models

    _models = {}
    if not MODEL_INFO_PATH.exists():
        log.warning("model-info.json not found at %s", MODEL_INFO_PATH)
    overlay = _load_config().get("models") or []
    entries = catalog.load_catalog_entries(MODEL_INFO_PATH, overlay=overlay if isinstance(overlay, list) else [])
    for entry in entries:
        for model_id in _entry_routable_ids(entry):
            _models[model_id] = entry

    log.info("Loaded %d routable model keys (catalog + config overlay)", len(_models))
    return _models


def routable_ids(name: str) -> list[str]:
    """Return every identifier that routes to the named model.

    Used by per-model stats to match ledger rows regardless of which alias or
    upstream id the caller sent. Includes name, alias, provider_model_id,
    omlx_id, and any alternate_ids, with empties/duplicates removed.
    """
    entry = _load_models().get(name)
    if not entry:
        return [name] if name else []
    return sorted(_entry_routable_ids(entry))


def provider_quirks(provider: str) -> frozenset:
    """Return the quirk set for a provider from live runtime config.

    Used by header construction and other call sites that only have the
    provider name (no resolved ProviderInfo).
    """
    config = _load_config()
    provider_config = _effective_provider_config(config, _canonical_provider(provider))
    quirks_raw = provider_config.get("quirks") or []
    if isinstance(quirks_raw, str):
        quirks_raw = [q.strip() for q in quirks_raw.split(",") if q.strip()]
    return frozenset(quirks_raw)


def auth_config() -> dict:
    """Return the live ``auth`` section from config.yaml.

    Re-reads after :func:`reload` resets the cached config. Used by
    :mod:`src.auth` so admin/client keys can live in the gitignored
    config.yaml instead of the launchd plist environment.
    """
    return _load_config().get("auth") or {}


def model_overrides() -> dict:
    """Return the live ``model_overrides`` section from config.yaml.

    Runtime state for models (currently just ``enabled``) lives here — NOT in
    model-info.json — so admin enable/disable toggles don't dirty the committed
    catalog and aren't reverted by deploys. Keys are model names; values are
    dicts with ``enabled: bool``. Missing entry = enabled (default true).

    Example config.yaml::

        model_overrides:
          gemini-3-flash:
            enabled: false
    """
    return _load_config().get("model_overrides") or {}


def _is_model_enabled(name: str | None) -> bool:
    """Check the runtime enabled override for a model. Default True."""
    if not name:
        return True
    overrides = model_overrides()
    entry = overrides.get(name)
    if isinstance(entry, dict):
        return bool(entry.get("enabled", True))
    return True


def reload():
    """Force reload of config and models (e.g. after config change)."""
    global _config, _models
    _config = None
    _models = None


def _databricks_cli() -> str:
    """Path to the databricks CLI (launchd PATH may not include homebrew)."""
    from shutil import which
    return which("databricks") or "/opt/homebrew/bin/databricks"


def _jwt_expiry_epoch(token: str) -> float | None:
    """Return JWT ``exp`` as epoch seconds, or None when the token is opaque."""
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode())
        exp = json.loads(decoded).get("exp")
    except Exception:
        return None
    if isinstance(exp, (int, float)):
        return float(exp)
    return None


def _persist_api_key(config_key: str, token: str) -> None:
    """Write a rotated api_key back to config.yaml, preserving comments.

    Keeps the on-disk config consistent with the in-memory one so gateway
    restarts and any external token-refresh job agree on the token.
    """
    if not CONFIG_PATH.exists():
        return
    try:
        lines = CONFIG_PATH.read_text().splitlines(keepends=True)

        # Locate the provider's own key line and its indent.
        key_re = re.compile(rf"^([ \t]*){re.escape(config_key)}:[ \t]*(#.*)?$")
        start = None
        provider_indent = ""
        for i, line in enumerate(lines):
            m = key_re.match(line.rstrip("\n"))
            if m:
                start = i
                provider_indent = m.group(1)
                break
        if start is None:
            log.warning("Could not persist refreshed api_key for %r to %s", config_key, CONFIG_PATH)
            return

        # Replace api_key only within this provider's block: lines strictly
        # more indented than the provider key. Stop at the next sibling/parent
        # so a missing api_key line can never clobber the NEXT provider's.
        api_key_re = re.compile(r"^([ \t]*api_key:[ \t]*)(\S+)(.*)$")
        for i in range(start + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped and not lines[i].startswith(provider_indent + " ") and not lines[i].startswith(provider_indent + "\t"):
                break  # left the provider block (sibling or parent)
            m = api_key_re.match(lines[i].rstrip("\n"))
            if m:
                lines[i] = f"{m.group(1)}{token}{m.group(3)}\n"
                CONFIG_PATH.write_text("".join(lines))
                return
        log.warning("Could not persist refreshed api_key for %r to %s", config_key, CONFIG_PATH)
    except OSError as exc:
        log.warning("Failed to persist refreshed api_key for %r: %s", config_key, exc)


async def refresh_oauth_token(provider: str, *, force: bool = False) -> str | None:
    """Refresh an expired OAuth token via an external CLI token cache.

    Opt-in per provider with config.yaml::

        providers:
          my_workspace:
            auth_refresh: databricks-cli   # only supported refresher today
            auth_profile: my-cli-profile   # optional; falls back to --host <base_url>

    Called by the server when an upstream returns 401/403 (e.g. a short-lived
    OAuth JWT expired). Providers without ``auth_refresh`` are untouched.
    Returns the new token if one was obtained, else None.
    """
    config = _load_config()
    found = _find_provider_entry(config, provider)
    if not found:
        return None
    config_key, entry = found
    if (entry.get("auth_refresh") or "") != "databricks-cli":
        return None
    base_url = (entry.get("base_url") or "").rstrip("/")
    current = entry.get("api_key", "")
    # Only short-lived OAuth JWTs ("eyJ...") benefit; PATs never expire this way.
    if not current.startswith("eyJ"):
        return None

    lock = _token_refresh_locks.setdefault(provider, asyncio.Lock())
    async with lock:
        # A concurrent request may have refreshed while we waited on the lock.
        latest = entry.get("api_key", "")
        if latest != current:
            return latest

        now = time.monotonic()
        if not force and now - _last_token_refresh_attempt.get(provider, 0.0) < _AUTH_REFRESH_MIN_INTERVAL:
            return None
        _last_token_refresh_attempt[provider] = now

        profile = entry.get("auth_profile", "")
        # --host wants the workspace origin, not a path under it.
        host = base_url.split("/serving-endpoints")[0] if base_url else ""
        cli_args = ["--profile", profile] if profile else ["--host", host]
        try:
            proc = await asyncio.create_subprocess_exec(
                _databricks_cli(), "auth", "token", *cli_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except (OSError, asyncio.TimeoutError) as exc:
            log.error("OAuth token refresh for %r failed: %s", provider, exc)
            return None
        if proc.returncode != 0:
            log.error(
                "OAuth token refresh for %r failed (exit %d): %s",
                provider, proc.returncode, stderr.decode(errors="replace")[:300],
            )
            return None
        try:
            token = json.loads(stdout).get("access_token", "")
        except json.JSONDecodeError:
            log.error("OAuth token refresh for %r: unparseable CLI output", provider)
            return None
        if not token or token == current:
            return None

        entry["api_key"] = token
        _persist_api_key(config_key, token)
        log.warning("Refreshed expired OAuth token for provider %r via CLI token cache", provider)
        return token


async def ensure_fresh_oauth_token(provider: str, *, min_valid_seconds: int = 300) -> str | None:
    """Refresh a configured OAuth JWT before it expires.

    PAT-backed providers and opaque API keys are left untouched. This is a
    lightweight preflight used by request routing so OAuth-backed Databricks
    workspaces do not depend on an external timer job.
    """
    config = _load_config()
    found = _find_provider_entry(config, provider)
    if not found:
        return None
    _config_key, entry = found
    if (entry.get("auth_refresh") or "") != "databricks-cli":
        return None
    current = entry.get("api_key", "")
    if not current.startswith("eyJ"):
        return None
    exp = _jwt_expiry_epoch(current)
    if exp is None or exp - time.time() > min_valid_seconds:
        return None
    return await refresh_oauth_token(provider, force=True)


def _configured_pool_members(entry: dict, config: dict) -> list[str]:
    """Pool members that have usable local config (base_url + api_key, enabled)."""
    members = []
    for member in _pool_members(entry, config):
        provider_config = _effective_provider_config(config, member)
        if provider_config.get("enabled") is False:
            continue
        if provider_config.get("base_url") and provider_config.get("api_key"):
            members.append(member)
    return members


def _availability_for_entry(model_id: str, entry: dict | None) -> dict:
    if not entry:
        return {
            "available": False,
            "reason": "model_not_found",
            "message": f"Model {model_id!r} is not in model-info.json",
        }

    name = entry.get("name") or model_id
    provider = _pool_members(entry, _load_config())[0]
    if not _is_model_enabled(name):
        return {
            "available": False,
            "reason": "model_disabled",
            "model": name,
            "provider": provider,
            "message": f"Model {name!r} is disabled by runtime model_overrides",
        }

    config = _load_config()
    configured = _configured_pool_members(entry, config)
    if configured:
        return {"available": True, "reason": "", "model": name, "provider": configured[0], "message": ""}

    # Nothing in the pool is usable — report why using the first member.
    provider_config = _effective_provider_config(config, provider)
    if provider_config.get("enabled") is False:
        return {
            "available": False,
            "reason": "provider_disabled",
            "model": name,
            "provider": provider,
            "message": f"Provider {provider!r} is disabled by runtime config",
        }

    missing = []
    if not provider_config.get("base_url"):
        missing.append("base_url")
    if not provider_config.get("api_key"):
        missing.append("api_key")
    return {
        "available": False,
        "reason": "provider_not_configured",
        "model": name,
        "provider": provider,
        "missing": missing,
        "message": f"Provider {provider!r} is missing {', '.join(missing)} in local runtime config",
    }


def model_availability(model_id: str) -> dict:
    """Return availability details for a model identifier without exposing secrets."""
    return _availability_for_entry(model_id, _load_models().get(model_id))


def is_model_available(model_id: str) -> bool:
    return bool(model_availability(model_id).get("available"))


def pool_candidates(model_id: str) -> list[str]:
    """Ordered, locally-configured pool member providers for a model id.

    Accepts any routable id (name, alias, provider_model_id). Single-provider
    models return a one-element list. Unknown models return [].
    """
    entry = _load_models().get(model_id)
    if not entry:
        return []
    return _configured_pool_members(entry, _load_config())


def resolve(model_id: str, provider_override: str | None = None) -> ProviderInfo | None:
    """Resolve a model name/alias/id to provider info.

    Pooled models route to the first configured pool member whose circuit is
    not open (per-workspace failover happens here for new requests; in-flight
    failover lives in src.upstream). ``provider_override`` pins a specific
    pool member, bypassing circuit state — used by upstream failover.
    """
    models = _load_models()
    entry = models.get(model_id)
    availability = _availability_for_entry(model_id, entry)
    if not availability["available"]:
        if availability["reason"] == "model_not_found":
            return None
        log.info("Model %r unavailable: %s", model_id, availability["message"])
        return None

    config = _load_config()
    candidates = _configured_pool_members(entry, config)
    if provider_override:
        provider = _canonical_provider(provider_override)
        if provider not in candidates:
            return None
    else:
        provider = candidates[0]
        if len(candidates) > 1:
            from src import circuit  # local import: circuit has no src imports
            for member in candidates:
                if not circuit.is_tripped(member):
                    provider = member
                    break
            if provider != candidates[0]:
                log.warning("pool: %r routing to %r (circuit open on %r)", model_id, provider, candidates[0])
    provider_config = _effective_provider_config(config, provider)

    base_url = provider_config.get("base_url", "")
    api_key = provider_config.get("api_key", "")
    protocol = provider_config.get("protocol", "openai")

    provider_model_id = entry.get("provider_model_id", "")
    if not provider_model_id:
        provider_model_id = entry.get("omlx_id", "")
    if not provider_model_id:
        provider_model_id = entry.get("name", "") or model_id

    # Per-model protocol override (e.g. an AI-gateway provider that serves both
    # Anthropic- and OpenAI-protocol models under one host).
    entry_protocol = entry.get("protocol")
    if entry_protocol:
        protocol = entry_protocol

    endpoint_suffix: str | None = None
    endpoint_style = provider_config.get("endpoint_style", "")
    if endpoint_style == "invocations":
        # base_url is a workspace host; each model has its own full invocation
        # URL: <base>/serving-endpoints/<provider_model_id>/invocations.
        base_url = base_url.rstrip("/") + f"/serving-endpoints/{provider_model_id}/invocations"
        endpoint_suffix = ""
    else:
        # Optional per-protocol path prefixes appended to base_url, e.g.
        # path_prefixes: {anthropic: /anthropic/v1, openai: /mlflow/v1}.
        path_prefixes = provider_config.get("path_prefixes") or {}
        prefix = path_prefixes.get(protocol) if isinstance(path_prefixes, dict) else None
        if prefix:
            base_url = base_url.rstrip("/") + "/" + str(prefix).strip("/")

    quirks_raw = provider_config.get("quirks") or []
    if isinstance(quirks_raw, str):
        quirks_raw = [q.strip() for q in quirks_raw.split(",") if q.strip()]

    return ProviderInfo(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        provider_model_id=provider_model_id,
        protocol=protocol,
        endpoint_suffix=endpoint_suffix,
        quirks=frozenset(quirks_raw),
        context=entry.get("context", 0),
        max_output_tokens=entry.get("max_output_tokens", 0),
        thinking=entry.get("thinking", ""),
        thinking_format=entry.get("thinking_format", ""),
        system_instruction=entry.get("system_instruction", ""),
        vision=bool(entry.get("vision", False)),
        pricing=entry.get("pricing"),
    )


def pricing_for(model_id: str) -> dict | None:
    """Return the $/Mtok pricing dict for a routable model, or None if unset.

    Keys may include: input, output, cache_read, cache_write, reasoning.
    None means cost is "unknown" for this model (ledger must not guess).
    """
    models = _load_models()
    entry = models.get(model_id)
    if not entry:
        return None
    pricing = entry.get("pricing")
    if not isinstance(pricing, dict):
        return None
    return pricing


def list_models() -> list[dict]:
    """Return all catalog model identifiers known to the gateway.

    This includes models whose provider is not configured locally so admin and
    debug views can explain why they are unavailable. Client-facing /v1/models
    should call list_available_models() instead.
    """
    models = _load_models()
    seen = set()
    result = []
    for model_id, entry in models.items():
        if model_id and model_id not in seen:
            seen.add(model_id)
            model_entry = dict(entry)
            model_entry["id"] = model_id
            result.append(model_entry)
    return result


def list_available_models() -> list[dict]:
    """Return model identifiers that are enabled and locally usable."""
    return [m for m in list_models() if is_model_available(m.get("id") or m.get("name", ""))]


def _configured_provider_ids() -> set[str]:
    config = _load_config()
    providers = config.get("providers", {}) or {}
    configured = {_canonical_provider(key) for key in providers}
    for entry in {id(v): v for v in _load_models().values()}.values():
        provider = _canonical_provider(entry.get("provider", ""))
        if _provider_defaults(provider) or _provider_env_config(provider):
            configured.add(provider)
    return configured


def _safe_url(value: str) -> str:
    """Strip URL userinfo before exposing provider config in admin APIs."""
    if not value:
        return ""
    try:
        parts = urlsplit(value)
    except ValueError:
        return ""
    if not parts.netloc:
        return value
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    # Do not expose query/fragment: provider base URLs occasionally carry
    # auth-ish parameters and admin status only needs the service origin/path.
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def provider_status() -> list[dict]:
    """Return masked provider configuration status for admin/observability APIs."""
    config = _load_config()
    configured = config.get("providers", {}) or {}
    models = list_models()

    # Count unique models per provider by canonical name. list_models()
    # exposes every routable identifier (name + alias + provider_model_id +
    # omlx_id + alternate_ids), so counting its rows would over-count models.
    # Dedupe by
    # the model `name`, which every identifier row for a model shares.
    model_names: dict[str, set[str]] = {}
    for model in models:
        provider = _canonical_provider(model.get("provider", ""))
        name = model.get("name") or model.get("id")
        if name and _is_model_enabled(name):
            model_names.setdefault(provider, set()).add(name)
    model_counts = {p: len(names) for p, names in model_names.items()}

    provider_ids = sorted(set(model_counts) | {_canonical_provider(key) for key in configured})
    result = []
    for provider in provider_ids:
        explicit_config = _resolve_provider_config(config, provider)
        env_config = _provider_env_config(provider)
        provider_config = _effective_provider_config(config, provider)
        base_url = _safe_url(provider_config.get("base_url", ""))
        has_api_key = bool(provider_config.get("api_key"))
        model_count = model_counts.get(provider, 0)
        issues = []
        if model_count and provider_config.get("enabled") is False:
            issues.append("provider_disabled")
        if model_count and provider_config.get("enabled") is not False and not base_url:
            issues.append("missing_base_url")
        if model_count and provider_config.get("enabled") is not False and not has_api_key:
            issues.append("missing_api_key")
        result.append({
            "id": provider,
            "configured": bool(explicit_config or _provider_defaults(provider) or env_config),
            "enabled_models": model_count,
            "base_url": base_url,
            "protocol": provider_config.get("protocol", "openai") if provider_config else "openai",
            "has_api_key": has_api_key,
            "ready": not issues,
            "issues": issues,
        })
    return result


def model_status() -> list[dict]:
    """Return routable model metadata with provider-config status."""
    ready = {p["id"]: p["ready"] for p in provider_status()}
    configured = _configured_provider_ids() | {"omlx"}
    result = []
    for model in list_models():
        provider = _canonical_provider(model.get("provider", ""))
        availability = model_availability(model.get("id") or model.get("name", ""))
        result.append({
            "id": model.get("id"),
            "name": model.get("name", ""),
            "alias": model.get("alias", ""),
            "provider": provider,
            "provider_model_id": model.get("provider_model_id") or model.get("omlx_id") or model.get("name", ""),
            "omlx_id": model.get("omlx_id", ""),
            "provider_configured": provider in configured,
            "provider_ready": ready.get(provider, False),
            "availability_reason": availability.get("reason", ""),
            "availability_message": availability.get("message", ""),
            "available": bool(availability.get("available")),
            "context": model.get("context", 0),
            "max_output_tokens": model.get("max_output_tokens", 0),
            "thinking": model.get("thinking", ""),
            "thinking_format": model.get("thinking_format", ""),
            "vision": bool(model.get("vision", False)),
            "pricing": model.get("pricing"),
            "enabled": _is_model_enabled(model.get("name") or model.get("id")),
        })
    return result


def config_validation() -> dict:
    """Validate current provider/model config without exposing secrets."""
    providers = provider_status()
    missing = [p for p in providers if p["enabled_models"] and p["issues"]]
    return {
        "ok": not missing,
        "model_info_path": str(MODEL_INFO_PATH),
        "config_path": str(CONFIG_PATH),
        "providers": providers,
        "issues": [
            {
                "provider": p["id"],
                "enabled_models": p["enabled_models"],
                "issues": p["issues"],
            }
            for p in missing
        ],
    }
