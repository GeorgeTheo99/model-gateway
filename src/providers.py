"""Provider registry — resolve model name to endpoint + credentials."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger("cloud-gateway")

# model-info.json is the shared source of truth and normally lives in the
# server repo root. The gateway no longer lives inside that tree, so allow the
# path to be supplied via env (set by the launcher/deploy); fall back to the
# legacy sibling-tree location for backward compat and local dev.
_DEFAULT_MODEL_INFO = Path(__file__).resolve().parents[2] / "model-info.json"
MODEL_INFO_PATH = Path(os.environ.get("CLOUD_GATEWAY_MODEL_INFO", str(_DEFAULT_MODEL_INFO))).resolve()
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


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
        # cloud-gateway intentionally routes only remote/cloud providers.
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

    provider = _canonical_provider(entry.get("provider", "local"))

    config = _load_config()
    provider_config = _resolve_provider_config(config, provider)

    default_base_url = ""
    default_api_key = ""
    if provider == "omlx":
        log.error("Provider %r is not routed by cloud-gateway; use oMLX directly", provider)
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
    )


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
