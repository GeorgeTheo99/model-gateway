"""Upstream error normalization — translate provider error responses into
clean client-facing envelopes (Anthropic or OpenAI shaped).

Handles double-JSON-encoded error bodies (some AI gateways return
``{"error_code":"BAD_REQUEST","message":"{\\"message\\":\\"prompt is too long\\"}"}``
where ``message`` is itself a JSON document) and detects context-window
overflows so clients get an actionable 400 instead of an opaque 502.
"""

import json
import logging

import httpx
from fastapi.responses import JSONResponse

log = logging.getLogger("model-gateway")

# Substrings indicating the request exceeded the model's context window.
# Providers phrase this differently; match case-insensitively.
_CONTEXT_OVERFLOW_MARKERS = (
    "input is too long",
    "prompt is too long",
    "too long",
    "context length",
    "context window",
    "maximum context",
    "exceeds the maximum",
    "reduce the length",
    "string too long",
)


def _unwrap_error_message(value) -> str:
    """Pull a clean human-readable message out of a possibly nested error value.

    Handles values that are themselves JSON strings (double-encoded).
    """
    for _ in range(4):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    value = json.loads(stripped)
                    continue
                except (ValueError, TypeError):
                    return stripped
            return stripped
        if isinstance(value, dict):
            nxt = (
                value.get("message")
                or value.get("detail")
                or value.get("error")
                or value.get("error_message")
            )
            if nxt is None:
                return json.dumps(value)
            value = nxt
            continue
        return str(value)
    return str(value)


def _extract_upstream_error_message(resp: httpx.Response, body_text: str) -> str:
    try:
        payload = resp.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            msg = _unwrap_error_message(err)
            if msg.strip():
                return msg.strip()
        for key in ("message", "detail", "error", "error_message"):
            if key in payload:
                msg = _unwrap_error_message(payload.get(key))
                if msg.strip():
                    return msg.strip()

    text = (body_text or "").strip()
    return text[:500] if text else f"Upstream returned {resp.status_code}"


def _extract_upstream_error_type(resp: httpx.Response) -> tuple[str | None, str | None]:
    """Return (error.type, error.code) from an upstream error body when present.

    Recognizes both Anthropic-style ``{"type": "error", "error": {"type": ...}}``
    and OpenAI-style ``{"error": {"type": ..., "code": ...}}`` bodies so the
    original error fidelity (authentication_error, permission_error,
    overloaded_error, ...) survives the gateway instead of being relabeled.
    """
    try:
        payload = resp.json()
    except Exception:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    err = payload.get("error")
    if not isinstance(err, dict):
        return None, None
    error_type = err.get("type")
    error_code = err.get("code")
    return (
        error_type if isinstance(error_type, str) and error_type else None,
        error_code if isinstance(error_code, str) and error_code else None,
    )


def _is_context_overflow(status_code: int, message: str) -> bool:
    if status_code not in (400, 413, 422):
        return False
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS)


def _anthropic_error(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "type": "error",
            "error": {"type": error_type, "message": message},
        },
    )


def upstream_error(resp: httpx.Response, body_text: str, provider_label: str = "Provider") -> JSONResponse:
    """Anthropic-envelope error response for a failed upstream request."""
    message = _extract_upstream_error_message(resp, body_text)

    if _is_context_overflow(resp.status_code, message):
        log.warning("Context overflow from %s (status=%s): %s", provider_label, resp.status_code, message)
        return _anthropic_error(
            400,
            "invalid_request_error",
            "Input is too long for this model's context window. Reduce the "
            "conversation length (compact history, drop large tool outputs, or "
            f"split the request) and try again. Upstream detail: {message}",
        )

    upstream_type, _ = _extract_upstream_error_type(resp)

    if resp.status_code == 429:
        retry_after = resp.headers.get("retry-after")
        if retry_after:
            message = f"{message} (retry-after: {retry_after})"
        return _anthropic_error(429, upstream_type or "rate_limit_error", message)

    if resp.status_code == 529:
        return _anthropic_error(529, upstream_type or "overloaded_error", message)

    if upstream_type:
        # Preserve upstream error fidelity: original error.type + status code.
        return _anthropic_error(resp.status_code, upstream_type, message)

    if 400 <= resp.status_code < 500:
        return _anthropic_error(resp.status_code, "invalid_request_error", message)

    return _anthropic_error(502, "api_error", f"{provider_label} returned {resp.status_code}: {message}")


def upstream_error_openai(resp: httpx.Response, body_text: str, provider_label: str = "Provider") -> JSONResponse:
    """Like :func:`upstream_error` but emits the OpenAI ``{"error": {...}}``
    envelope for OpenAI-shaped passthrough routes."""
    message = _extract_upstream_error_message(resp, body_text)

    if _is_context_overflow(resp.status_code, message):
        log.warning("Context overflow (openai shape) from %s (status=%s): %s", provider_label, resp.status_code, message)
        return JSONResponse(
            status_code=400,
            content={"error": {
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "message": (
                    "Input is too long for this model's context window. Reduce the "
                    "conversation length and try again. Upstream detail: " + message
                ),
            }},
        )

    upstream_type, upstream_code = _extract_upstream_error_type(resp)

    def _openai_error(status: int, error_type: str, msg: str) -> JSONResponse:
        err: dict = {"type": error_type, "message": msg}
        if upstream_code:
            err["code"] = upstream_code
        return JSONResponse(status_code=status, content={"error": err})

    if resp.status_code == 429:
        retry_after = resp.headers.get("retry-after")
        if retry_after:
            message = f"{message} (retry-after: {retry_after})"
        return _openai_error(429, upstream_type or "rate_limit_error", message)

    if resp.status_code == 529:
        return _openai_error(529, upstream_type or "overloaded_error", message)

    if upstream_type:
        # Preserve upstream error fidelity: original error.type/code + status.
        return _openai_error(resp.status_code, upstream_type, message)

    if 400 <= resp.status_code < 500:
        return _openai_error(resp.status_code, "invalid_request_error", message)

    return _openai_error(502, "api_error", f"{provider_label} returned {resp.status_code}: {message}")
