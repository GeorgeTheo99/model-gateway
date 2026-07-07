"""Config-driven model-level fallback for upstream saturation / missing models.

The mapping lives in config.yaml — never in code — so workspace/provider
specific model names stay out of the repo::

    model_fallbacks:
      my-primary-endpoint: my-fallback-endpoint

Keys/values are the upstream (provider) model ids as sent in the request body.
Consulted after a request has already exhausted the retry loop with a
saturation status (429/502/503/504) or a 404 "model not found".
"""

from dataclasses import dataclass

from src import providers

SATURATION_STATUSES = {429, 502, 503, 504}


@dataclass
class ModelFallbackDecision:
    fallback_model: str
    reason: str


def _fallback_map() -> dict:
    mapping = providers._load_config().get("model_fallbacks") or {}
    return mapping if isinstance(mapping, dict) else {}


def fallback_after_error(provider_model_id: str, status_code: int, body_text: str = "") -> ModelFallbackDecision | None:
    fallback = _fallback_map().get(provider_model_id)
    if not fallback:
        return None

    if status_code in SATURATION_STATUSES:
        return ModelFallbackDecision(
            fallback_model=str(fallback),
            reason=f"upstream saturation status {status_code}",
        )

    message = (body_text or "").lower()
    if status_code == 404 and "model" in message and (
        "not found" in message or "unknown" in message or "does not exist" in message
    ):
        return ModelFallbackDecision(
            fallback_model=str(fallback),
            reason="missing upstream model",
        )

    return None
