"""Provider registry — resolve model name to endpoint + credentials."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml

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
    context: int = 0
    max_output_tokens: int = 0
    thinking: str = ""  # "", "optional", or "always"
    thinking_format: str = ""  # optional explicit gateway normalization format
    system_instruction: str = ""
    vision: bool = False  # authoritative: True if the model can natively handle image inputs
    pricing: dict = None  # $/Mtok rates: input, output, cache_read?, cache_write?, reasoning?


_config: dict | None = None
_models: dict[str, dict] | None = None

_PROVIDER_SYNONYMS = {
    "local": "omlx",
    "omlx": "omlx",
    "mlx": "omlx",
    "openai": "openai",
    "gpt": "openai",
    "anthropic": "anthropic",
    "anthropci": "anthropic",
    "claude": "anthropic",
    "google": "google",
    "gemini": "google",
    "zhipuai": "zhipuai",
    "zai": "zhipuai",
    "bigmodel": "zhipuai",
    "databricks": "databricks",
    "dbx": "databricks",
    "dbrx": "databricks",
}


def _canonical_provider(provider: str | None) -> str:
    raw = (provider or "local").strip().lower()
    if not raw:
        return "omlx"
    return _PROVIDER_SYNONYMS.get(raw, raw)


def _resolve_provider_config(config: dict, provider: str) -> dict:
    providers = config.get("providers", {}) or {}
    direct = providers.get(provider)
    if direct:
        return direct
    for key, value in providers.items():
        if _canonical_provider(key) == provider:
            return value or {}
    return {}


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


def _load_models() -> dict[str, dict]:
    """Load routable models from model-info.json (keyed by name/alias/id)."""
    global _models
    if _models is not None:
        return _models

    _models = {}
    if not MODEL_INFO_PATH.exists():
        log.warning("model-info.json not found at %s", MODEL_INFO_PATH)
        return _models

    with open(MODEL_INFO_PATH) as f:
        data = json.load(f)

    for entry in data.get("llm", []):
        provider = _canonical_provider(entry.get("provider", "local"))
        # GGUF/llama.cpp serving has been retired on this machine. Local MLX
        # entries are routable through model-gateway as a thin proxy to oMLX.
        if provider in {"gguf", "llama", "llama_cpp", "llama.cpp"}:
            continue
        normalized_entry = dict(entry)
        normalized_entry["provider"] = provider
        name = entry.get("name")
        if name:
            _models[name] = normalized_entry

        # Also index by alias and backend model id.
        alias = entry.get("alias")
        if alias:
            _models[alias] = normalized_entry

        pmid = entry.get("provider_model_id")
        if pmid:
            _models[pmid] = normalized_entry

        # Local oMLX entries should resolve by omlx_id too.
        omlx_id = entry.get("omlx_id")
        if omlx_id:
            _models[omlx_id] = normalized_entry

    log.info("Loaded %d routable model keys from model-info.json", len(_models))
    return _models


def routable_ids(name: str) -> list[str]:
    """Return every identifier that routes to the named model.

    Used by per-model stats to match ledger rows regardless of which alias or
    upstream id the caller sent. Includes name, alias, provider_model_id, and
    omlx_id, with empties/duplicates removed.
    """
    entry = _load_models().get(name)
    if not entry:
        return [name] if name else []
    ids = {entry.get("name"), entry.get("alias"), entry.get("provider_model_id"), entry.get("omlx_id")}
    ids.discard(None)
    ids.discard("")
    return sorted(ids)


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


def _availability_for_entry(model_id: str, entry: dict | None) -> dict:
    if not entry:
        return {
            "available": False,
            "reason": "model_not_found",
            "message": f"Model {model_id!r} is not in model-info.json",
        }

    name = entry.get("name") or model_id
    provider = _canonical_provider(entry.get("provider", "local"))
    if not _is_model_enabled(name):
        return {
            "available": False,
            "reason": "model_disabled",
            "model": name,
            "provider": provider,
            "message": f"Model {name!r} is disabled by runtime model_overrides",
        }

    provider_config = _effective_provider_config(_load_config(), provider)
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
    if missing:
        return {
            "available": False,
            "reason": "provider_not_configured",
            "model": name,
            "provider": provider,
            "missing": missing,
            "message": f"Provider {provider!r} is missing {', '.join(missing)} in local runtime config",
        }

    return {"available": True, "reason": "", "model": name, "provider": provider, "message": ""}


def model_availability(model_id: str) -> dict:
    """Return availability details for a model identifier without exposing secrets."""
    return _availability_for_entry(model_id, _load_models().get(model_id))


def is_model_available(model_id: str) -> bool:
    return bool(model_availability(model_id).get("available"))


def resolve(model_id: str) -> ProviderInfo | None:
    """Resolve a model name/alias/id to provider info."""
    models = _load_models()
    entry = models.get(model_id)
    availability = _availability_for_entry(model_id, entry)
    if not availability["available"]:
        if availability["reason"] == "model_not_found":
            return None
        log.info("Model %r unavailable: %s", model_id, availability["message"])
        return None

    provider = _canonical_provider(entry.get("provider", "local"))
    provider_config = _effective_provider_config(_load_config(), provider)

    base_url = provider_config.get("base_url", "")
    api_key = provider_config.get("api_key", "")
    protocol = provider_config.get("protocol", "openai")

    provider_model_id = entry.get("provider_model_id", "")
    if not provider_model_id:
        provider_model_id = entry.get("omlx_id", "")
    if not provider_model_id:
        provider_model_id = entry.get("name", "") or model_id

    return ProviderInfo(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        provider_model_id=provider_model_id,
        protocol=protocol,
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
    # omlx_id), so counting its rows would over-count models 2-3x. Dedupe by
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
