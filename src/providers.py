"""Provider registry — resolve model name to endpoint + credentials."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml

log = logging.getLogger("model-gateway")


def _env(*names: str) -> str | None:
    """Return the first non-empty env var, preferring MODEL_GATEWAY_* names."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


# model-info.json is the gateway-owned source of truth. Allow the path to be
# supplied via env (set by the launcher/deploy); fall back to the checkout-local
# catalog for tests and local dev. CLOUD_GATEWAY_* names remain supported during
# the cloud-gateway -> model-gateway migration.
_DEFAULT_MODEL_INFO = Path(__file__).resolve().parents[1] / "model-info.json"
MODEL_INFO_PATH = Path(
    _env("MODEL_GATEWAY_MODEL_INFO", "CLOUD_GATEWAY_MODEL_INFO") or str(_DEFAULT_MODEL_INFO)
).resolve()
# Optional source-repo path for durable model edits. When set, model write
# operations mirror edits here (pending commit) in addition to the deployed
# MODEL_INFO_PATH, so changes survive deploys once committed. The deployed
# copy is overwritten by git checkout on each deploy; the source copy is not.
_DEFAULT_MODEL_INFO_SOURCE = Path.home() / "local_code" / "model-gateway" / "model-info.json"
MODEL_INFO_SOURCE_PATH = (
    Path(
        _env("MODEL_GATEWAY_MODEL_INFO_SOURCE", "CLOUD_GATEWAY_MODEL_INFO_SOURCE")
        or str(_DEFAULT_MODEL_INFO_SOURCE)
    ).resolve()
    if _env("MODEL_GATEWAY_MODEL_INFO_SOURCE", "CLOUD_GATEWAY_MODEL_INFO_SOURCE")
    or _DEFAULT_MODEL_INFO_SOURCE.exists()
    else None
)
CONFIG_PATH = Path(
    _env("MODEL_GATEWAY_CONFIG", "CLOUD_GATEWAY_CONFIG")
    or Path(__file__).resolve().parents[1] / "config" / "config.yaml"
).resolve()


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
        # model-gateway currently routes only remote/cloud providers.
        # Local MLX/oMLX models are served directly by oMLX on port 9110;
        # GGUF/llama.cpp serving has been retired on this machine.
        if provider in {"omlx", "local", "mlx", "gguf", "llama", "llama_cpp", "llama.cpp"}:
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


def resolve(model_id: str) -> ProviderInfo | None:
    """Resolve a model name/alias/id to provider info."""
    models = _load_models()
    entry = models.get(model_id)
    if not entry:
        return None

    # Runtime enabled state lives in config.yaml model_overrides (not the
    # committed catalog), so toggles survive deploys and don't dirty the repo.
    name = entry.get("name") or model_id
    if not _is_model_enabled(name):
        log.info("Model %r is disabled; not routing", name)
        return None

    provider = _canonical_provider(entry.get("provider", "local"))

    config = _load_config()
    provider_config = _resolve_provider_config(config, provider)

    default_base_url = ""
    default_api_key = ""
    if provider == "omlx":
        log.error("Provider %r is not routed by model-gateway yet; use oMLX directly", provider)
        return None

    base_url = provider_config.get("base_url", default_base_url)
    api_key = provider_config.get("api_key", default_api_key)
    protocol = provider_config.get("protocol", "openai")

    if not base_url or not api_key:
        log.error("Provider %r missing base_url or api_key in config", provider)
        return None

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
    """Return all model identifiers accepted by the gateway for /v1/models.

    Claude Code validates ANTHROPIC_MODEL against /v1/models before sending a
    request, while launchers may use aliases or upstream provider IDs. Expose
    every routable identifier so validation matches resolve().
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


def _configured_provider_ids() -> set[str]:
    config = _load_config()
    providers = config.get("providers", {}) or {}
    return {_canonical_provider(key) for key in providers}


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

    model_counts: dict[str, int] = {}
    for model in models:
        provider = _canonical_provider(model.get("provider", ""))
        model_counts[provider] = model_counts.get(provider, 0) + 1

    provider_ids = sorted(set(model_counts) | {_canonical_provider(key) for key in configured})
    result = []
    for provider in provider_ids:
        provider_config = _resolve_provider_config(config, provider)
        base_url = _safe_url(provider_config.get("base_url", ""))
        has_api_key = bool(provider_config.get("api_key"))
        issues = []
        if model_counts.get(provider, 0) and not base_url:
            issues.append("missing_base_url")
        if model_counts.get(provider, 0) and not has_api_key:
            issues.append("missing_api_key")
        result.append({
            "id": provider,
            "configured": bool(provider_config),
            "enabled_models": model_counts.get(provider, 0),
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
    configured = _configured_provider_ids()
    result = []
    for model in list_models():
        provider = _canonical_provider(model.get("provider", ""))
        result.append({
            "id": model.get("id"),
            "name": model.get("name", ""),
            "alias": model.get("alias", ""),
            "provider": provider,
            "provider_model_id": model.get("provider_model_id", ""),
            "provider_configured": provider in configured,
            "provider_ready": ready.get(provider, False),
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
