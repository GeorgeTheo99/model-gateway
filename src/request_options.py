"""Pure request options shared by live routing and workspace smoke probes."""
from __future__ import annotations


def upstream_auth_headers(api_key: str, protocol: str, quirks=()) -> dict[str, str]:
    if protocol != "anthropic":
        # An owner-added local server may take no key; an empty bearer could be refused.
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}
    headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
    if "anthropic_bearer_auth" in quirks:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["x-api-key"] = api_key
    return headers


def normalize_token_limit(request: dict, quirks) -> None:
    if "use_max_completion_tokens" not in quirks:
        return
    value = request.get("max_completion_tokens")
    if value is None:
        value = request.get("max_tokens")
    if value is None:
        value = request.get("max_output_tokens")
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        request.pop(key, None)
    if value is not None:
        request["max_completion_tokens"] = value


def apply_openrouter_routing(request: dict, quirks) -> None:
    """OpenRouter provider routing the model's quirks require, keeping the caller's other preferences."""
    routing = {}
    if "openrouter_zdr" in quirks:
        # Only zero-data-retention endpoints may serve the model.
        routing["zdr"] = True
    if "openrouter_fireworks_only" in quirks:
        # Only Fireworks may serve the model, never another host.
        routing.update(only=["fireworks"], allow_fallbacks=False)
    if routing:
        current = request.get("provider")
        request["provider"] = {**(current if isinstance(current, dict) else {}), **routing}
