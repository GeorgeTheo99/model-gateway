"""Robust upstream HTTP machinery — retries, backoff, circuit breaker, auth refresh.

The original retry loop was generalized to add:
  - Circuit key is the provider name (per-provider breaker, see src.circuit).
  - 401/403 triggers a single opt-in OAuth refresh via
    src.providers.refresh_oauth_token (no-op unless the provider config sets
    ``auth_refresh``).
  - Optional config-driven model fallback (config.yaml ``model_fallbacks:``)
    after saturation/missing-model failures.
"""

import asyncio
import email.utils
import logging
import random
import time
from dataclasses import dataclass

import httpx
from fastapi import Request

from src.circuit import (
    is_tripped,
    probe_done,
    record_failure,
    record_success,
    wait_for_recovery,
)
from src.model_fallback import SATURATION_STATUSES, fallback_after_error
from src.providers import ensure_fresh_oauth_token, refresh_oauth_token

log = logging.getLogger("model-gateway")

_RETRY_MAX = 3  # per-request retries for 5xx (circuit breaker handles sustained outages)
_RETRY_BASE_DELAY = 1.5  # seconds
_RETRY_MAX_DELAY = 15.0  # seconds
_RETRY_JITTER_RATIO = 0.25
_RETRY_429_MIN_DELAY = 8.0  # seconds
_RETRY_429_MAX_DELAY = 60.0  # seconds
_RETRY_429_ATTEMPTS = 6  # rate limits get more patience
_RETRY_TRANSPORT_ATTEMPTS = 4
_RETRY_TRANSPORT_MAX_DELAY = 15.0  # seconds


async def _raise_if_disconnected(request: Request | None) -> None:
    """Stop upstream work promptly when the downstream client has gone away."""
    if request is not None and await request.is_disconnected():
        raise asyncio.CancelledError("client disconnected")


async def _sleep_or_disconnect(delay: float, request: Request | None) -> None:
    """Backoff sleep that wakes early on client disconnect checks."""
    if delay <= 0:
        await _raise_if_disconnected(request)
        await asyncio.sleep(0)
        return
    deadline = time.monotonic() + delay
    while True:
        await _raise_if_disconnected(request)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 0.25))


def _circuit_key(provider: str) -> str:
    return provider or ""


def _compute_retry_delay(resp: httpx.Response, attempt: int) -> float:
    is_rate_limited = resp.status_code == 429
    max_delay = _RETRY_429_MAX_DELAY if is_rate_limited else _RETRY_MAX_DELAY
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            retry_delay = min(max(float(retry_after), 0.0), max_delay)
            if is_rate_limited:
                retry_delay = max(retry_delay, _RETRY_429_MIN_DELAY)
            return retry_delay
        except ValueError:
            retry_at = email.utils.parsedate_to_datetime(retry_after)
            if retry_at is not None:
                retry_delay = min(max(retry_at.timestamp() - time.time(), 0.0), max_delay)
                if is_rate_limited:
                    retry_delay = max(retry_delay, _RETRY_429_MIN_DELAY)
                return retry_delay
    base_multiplier = 4 if is_rate_limited else 1
    base_delay = min((_RETRY_BASE_DELAY * base_multiplier) * (2 ** attempt), max_delay)
    jitter = base_delay * _RETRY_JITTER_RATIO * random.random()
    delay = base_delay + jitter
    if is_rate_limited:
        return max(delay, _RETRY_429_MIN_DELAY)
    return delay


def _compute_transport_retry_delay(attempt: int) -> float:
    base_delay = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_TRANSPORT_MAX_DELAY)
    jitter = base_delay * _RETRY_JITTER_RATIO * random.random()
    return base_delay + jitter


def _is_retryable_exception(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    )


def _is_retryable_status(status_code: int) -> bool:
    return status_code in (408, 409, 425, 429, 500, 502, 503, 504)


def _is_auth_status(status_code: int) -> bool:
    """Auth failures that may be a stale short-lived OAuth token (some gateways return 403)."""
    return status_code in (401, 403)


def _apply_refreshed_token(headers: dict, token: str, request: Request | None) -> dict:
    """Swap the bearer/x-api-key credential in forwarded headers after an OAuth refresh."""
    headers = dict(headers)
    if "Authorization" in headers:
        headers["Authorization"] = f"Bearer {token}"
    if "x-api-key" in headers:
        headers["x-api-key"] = token
    if request is not None:
        request.state.api_key = token
    return headers


async def _preflight_oauth_token(provider: str, headers: dict, request: Request | None) -> dict:
    """Refresh nearly-expired OAuth tokens before the upstream request."""
    if not provider:
        return headers
    token = await ensure_fresh_oauth_token(provider)
    if not token:
        return headers
    log.warning("Preflight refreshed OAuth token for provider %r", provider)
    return _apply_refreshed_token(headers, token, request)


def _is_circuit_breaker_status(status_code: int) -> bool:
    """Status codes that indicate the provider is down (not just rate-limited)."""
    return status_code in (502, 503, 504)


def _probe_succeeded(status_code: int) -> bool:
    """Treat any non-circuit-breaker response as proof the provider is reachable."""
    return not _is_circuit_breaker_status(status_code)


def _max_attempts_for_status(status_code: int) -> int:
    if status_code == 429:
        return _RETRY_429_ATTEMPTS
    return _RETRY_MAX


async def _retry_post(
    client: httpx.AsyncClient, endpoint: str, *, json: dict, headers: dict,
    provider: str = "",
    request: Request | None = None,
) -> httpx.Response:
    """POST with exponential backoff + circuit breaker.

    If the provider's circuit is open, waits for recovery (up to 3 min)
    instead of sending requests into a known-down endpoint. This keeps
    errors inside the gateway so the coding harness never sees them.
    """
    circuit = _circuit_key(provider)
    probe_request = False
    if circuit and is_tripped(circuit):
        log.info("circuit[%s]: POST waiting for recovery", circuit)
        recovered = await wait_for_recovery(circuit)
        if not recovered:
            record_failure(circuit, 502, "circuit breaker timeout")
            raise httpx.ConnectError(f"Provider {circuit} unavailable (circuit open)")
        probe_request = is_tripped(circuit)

    max_attempts = _RETRY_MAX
    attempt = 0
    auth_retried = False
    headers = await _preflight_oauth_token(provider, headers, request)
    while attempt < max_attempts:
        await _raise_if_disconnected(request)
        try:
            resp = await client.post(endpoint, json=json, headers=headers)
        except Exception as exc:
            if circuit and probe_request:
                probe_done(circuit, success=False)
                probe_request = False
            max_attempts = max(max_attempts, _RETRY_TRANSPORT_ATTEMPTS)
            if not _is_retryable_exception(exc) or attempt == max_attempts - 1:
                if circuit:
                    record_failure(circuit, 0, f"transport: {type(exc).__name__}")
                raise
            if circuit:
                record_failure(circuit, 0, f"transport: {type(exc).__name__}")
            delay = _compute_transport_retry_delay(attempt)
            log.warning(
                "Transient upstream transport error %s on POST (attempt %d/%d), retrying in %.1fs",
                type(exc).__name__, attempt + 1, max_attempts, delay,
            )
            await _sleep_or_disconnect(delay, request)
            attempt += 1
            continue

        if _is_auth_status(resp.status_code) and not auth_retried and provider:
            auth_retried = True
            await resp.aread()
            token = await refresh_oauth_token(provider, force=True)
            if token:
                headers = _apply_refreshed_token(headers, token, request)
                log.warning(
                    "Upstream %d on POST — refreshed OAuth token for %r, retrying",
                    resp.status_code, provider,
                )
                continue  # immediate retry with fresh credentials, no attempt charge
            if circuit and probe_request:
                probe_done(circuit, success=_probe_succeeded(resp.status_code))
                probe_request = False
            return resp

        if not _is_retryable_status(resp.status_code):
            if circuit:
                if probe_request:
                    probe_done(circuit, success=True)
                    probe_request = False
                else:
                    record_success(circuit)
            return resp

        max_attempts = max(max_attempts, _max_attempts_for_status(resp.status_code))
        await resp.aread()

        if circuit and probe_request:
            probe_done(circuit, success=_probe_succeeded(resp.status_code))
            probe_request = False

        if circuit and _is_circuit_breaker_status(resp.status_code):
            record_failure(circuit, resp.status_code, resp.text[:200])

        if attempt == max_attempts - 1:
            return resp

        # If circuit just tripped, wait for recovery instead of blind retry
        if circuit and is_tripped(circuit):
            log.info("circuit[%s]: tripped mid-retry (POST), waiting for recovery", circuit)
            recovered = await wait_for_recovery(circuit)
            if not recovered:
                return resp
            probe_request = is_tripped(circuit)

        delay = _compute_retry_delay(resp, attempt)
        log.warning("Transient upstream status %d on POST (attempt %d/%d), retrying in %.1fs", resp.status_code, attempt + 1, max_attempts, delay)
        await _sleep_or_disconnect(delay, request)
        attempt += 1
    return resp  # unreachable, but satisfies type checkers


async def _retry_send_stream(
    client: httpx.AsyncClient, endpoint: str, *, json: dict, headers: dict,
    provider: str = "",
    request: Request | None = None,
) -> httpx.Response:
    """Streaming POST with exponential backoff + circuit breaker.

    Returns an open streaming response — caller must close it.
    Same circuit breaker semantics as _retry_post.
    """
    circuit = _circuit_key(provider)
    probe_request = False
    if circuit and is_tripped(circuit):
        log.info("circuit[%s]: stream waiting for recovery", circuit)
        recovered = await wait_for_recovery(circuit)
        if not recovered:
            record_failure(circuit, 502, "circuit breaker timeout")
            raise httpx.ConnectError(f"Provider {circuit} unavailable (circuit open)")
        probe_request = is_tripped(circuit)

    max_attempts = _RETRY_MAX
    attempt = 0
    auth_retried = False
    headers = await _preflight_oauth_token(provider, headers, request)
    while attempt < max_attempts:
        await _raise_if_disconnected(request)
        try:
            resp = await client.send(
                client.build_request("POST", endpoint, json=json, headers=headers),
                stream=True,
            )
        except Exception as exc:
            if circuit and probe_request:
                probe_done(circuit, success=False)
                probe_request = False
            max_attempts = max(max_attempts, _RETRY_TRANSPORT_ATTEMPTS)
            if not _is_retryable_exception(exc) or attempt == max_attempts - 1:
                if circuit:
                    record_failure(circuit, 0, f"transport: {type(exc).__name__}")
                raise
            if circuit:
                record_failure(circuit, 0, f"transport: {type(exc).__name__}")
            delay = _compute_transport_retry_delay(attempt)
            log.warning(
                "Transient upstream transport error %s on stream (attempt %d/%d), retrying in %.1fs",
                type(exc).__name__, attempt + 1, max_attempts, delay,
            )
            await _sleep_or_disconnect(delay, request)
            attempt += 1
            continue

        if _is_auth_status(resp.status_code) and not auth_retried and provider:
            auth_retried = True
            await resp.aread()
            await resp.aclose()
            token = await refresh_oauth_token(provider, force=True)
            if token:
                headers = _apply_refreshed_token(headers, token, request)
                log.warning(
                    "Upstream %d on stream — refreshed OAuth token for %r, retrying",
                    resp.status_code, provider,
                )
                continue  # immediate retry with fresh credentials, no attempt charge
            # No fresh token available — return a non-streamed error response as-is.
            if circuit and probe_request:
                probe_done(circuit, success=_probe_succeeded(resp.status_code))
                probe_request = False
            resp = await client.send(
                client.build_request("POST", endpoint, json=json, headers=headers),
                stream=True,
            )
            return resp

        if not _is_retryable_status(resp.status_code):
            if circuit:
                if probe_request:
                    probe_done(circuit, success=True)
                    probe_request = False
                else:
                    record_success(circuit)
            return resp

        max_attempts = max(max_attempts, _max_attempts_for_status(resp.status_code))
        await resp.aread()
        await resp.aclose()

        if circuit and probe_request:
            probe_done(circuit, success=_probe_succeeded(resp.status_code))
            probe_request = False

        if circuit and _is_circuit_breaker_status(resp.status_code):
            record_failure(circuit, resp.status_code, "")

        if attempt == max_attempts - 1:
            # Need a fresh stream response to return
            resp = await client.send(
                client.build_request("POST", endpoint, json=json, headers=headers),
                stream=True,
            )
            if circuit and probe_request:
                probe_done(circuit, success=_probe_succeeded(resp.status_code))
                probe_request = False
            if circuit and _is_circuit_breaker_status(resp.status_code):
                record_failure(circuit, resp.status_code, "")
            elif circuit and not _is_retryable_status(resp.status_code):
                record_success(circuit)
            return resp

        # If circuit just tripped, wait for recovery instead of blind retry
        if circuit and is_tripped(circuit):
            log.info("circuit[%s]: tripped mid-retry (stream), waiting for recovery", circuit)
            recovered = await wait_for_recovery(circuit)
            if not recovered:
                resp = await client.send(
                    client.build_request("POST", endpoint, json=json, headers=headers),
                    stream=True,
                )
                return resp
            probe_request = is_tripped(circuit)

        delay = _compute_retry_delay(resp, attempt)
        log.warning("Transient upstream status %d on stream (attempt %d/%d), retrying in %.1fs", resp.status_code, attempt + 1, max_attempts, delay)
        await _sleep_or_disconnect(delay, request)
        attempt += 1
    return resp  # unreachable


@dataclass
class PoolContext:
    """Workspace-pool failover context for a single upstream request.

    Pool members must be protocol-compatible for the routed model (same body
    and response shape, same header style, same endpoint suffix relative to
    base_url) — config guarantees this by pooling only like-kind workspace
    entries. Failover then reduces to swapping base_url + credentials.
    """
    model_key: str          # any routable id for the model (used to re-resolve)
    provider: str           # provider the request was originally resolved to
    base_url: str           # that provider's resolved base_url (prefix of endpoint)
    api_key: str            # that provider's credential as sent in headers


def _fallback_endpoint(endpoint: str, requested_model: str, fallback_model: str) -> str:
    """Rebuild the endpoint for a model fallback.

    ``endpoint_style: invocations`` providers embed the model in the URL
    (/serving-endpoints/<model>/invocations), so switching json["model"]
    alone would keep hitting the failed primary endpoint.
    """
    return endpoint.replace(
        f"/serving-endpoints/{requested_model}/",
        f"/serving-endpoints/{fallback_model}/",
    )


def _pool_eligible_status(status_code: int) -> bool:
    """Statuses that justify trying the next workspace in the pool."""
    if status_code in SATURATION_STATUSES:
        return True
    return status_code in (401, 403, 404)


def _rewire_for_provider(
    pool: PoolContext, endpoint: str, headers: dict, info,
) -> tuple[str, dict]:
    """Rebuild endpoint + auth headers for another pool member.

    Handles kind differences between pool members: ``endpoint_style:
    invocations`` members resolve to a complete URL (suffix ""), while
    path-prefix members need the protocol suffix appended. All pooled
    members are Databricks workspaces, which accept Bearer auth everywhere.
    """
    original_suffix = endpoint[len(pool.base_url):] if endpoint.startswith(pool.base_url) else ""
    cand_suffix = getattr(info, "endpoint_suffix", None)
    if cand_suffix is None:
        # Path-prefix member: reuse the original suffix, or derive the
        # protocol default when the original was a complete invocation URL.
        cand_suffix = original_suffix or (
            "/messages" if "anthropic-version" in headers else "/chat/completions"
        )
    new_endpoint = f"{info.base_url}{cand_suffix}"
    new_headers = dict(headers)
    # Databricks accepts Bearer on both AI-gateway and invocations endpoints.
    new_headers["Authorization"] = f"Bearer {info.api_key}"
    if "x-api-key" in new_headers:
        new_headers["x-api-key"] = info.api_key
    return new_endpoint, new_headers


def _pool_ctx(pool: PoolContext | None, request: Request | None, provider: str) -> PoolContext | None:
    """Explicit pool arg wins; otherwise use the handler-set request.state.pool_ctx.

    The state ctx is only honored when it matches the provider actually being
    called — a vision-fallback or model-fallback resolve may have changed the
    route after the ctx was stashed.
    """
    if pool is not None:
        return pool
    ctx = getattr(request.state, "pool_ctx", None) if request is not None else None
    if ctx is not None and ctx.provider == provider:
        return ctx
    return None


def _pool_failover_candidates(pool: PoolContext | None) -> list[str]:
    if pool is None or not pool.model_key:
        return []
    from src.providers import pool_candidates  # runtime import: avoid cycle
    candidates = [c for c in pool_candidates(pool.model_key) if c != pool.provider]
    # Prefer members whose circuit is closed; a known-down workspace would
    # block the failover on its recovery wait. Keep tripped members as a
    # last resort (their probe may succeed) rather than dropping them.
    healthy = [c for c in candidates if not is_tripped(c)]
    tripped = [c for c in candidates if is_tripped(c)]
    return healthy + tripped


async def _send_with_pool(
    send,
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    json: dict,
    headers: dict,
    provider: str,
    request: Request | None,
    pool: PoolContext | None,
) -> httpx.Response:
    """Run one send (post or stream-open) with ordered workspace-pool failover.

    Transport exceptions and pool-eligible statuses (429/5xx exhausted, 404,
    401/403 after refresh) advance to the next pool member. The last
    response/exception is returned/raised when every member fails.
    """
    candidates = _pool_failover_candidates(pool)
    try:
        resp = await send(
            client, endpoint, json=json, headers=headers, provider=provider, request=request,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        if not candidates:
            raise
        resp = None

    if resp is not None and (not candidates or not _pool_eligible_status(resp.status_code)):
        return resp

    from src.providers import resolve  # runtime import: avoid cycle
    for candidate in candidates:
        info = resolve(pool.model_key, provider_override=candidate)
        if info is None:
            continue
        if resp is not None:
            await resp.aread()
            await resp.aclose()
        cand_endpoint, cand_headers = _rewire_for_provider(pool, endpoint, headers, info)
        log.warning(
            "pool-failover: %s → workspace %r after failure on %r",
            pool.model_key, candidate, provider,
        )
        try:
            resp = await send(
                client, cand_endpoint, json=json, headers=cand_headers,
                provider=candidate, request=request,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            if candidate == candidates[-1]:
                raise
            resp = None
            continue
        if not _pool_eligible_status(resp.status_code):
            return resp
    if resp is None:  # every candidate raised; surface a connect error
        raise httpx.ConnectError(f"all pool workspaces failed for {pool.model_key}")
    return resp


async def _retry_post_with_model_fallback(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    json: dict,
    headers: dict,
    provider: str = "",
    request: Request | None = None,
    pool: PoolContext | None = None,
) -> httpx.Response:
    resp = await _send_with_pool(
        _retry_post, client, endpoint,
        json=json, headers=headers, provider=provider, request=request,
        pool=_pool_ctx(pool, request, provider),
    )

    requested_model = json.get("model", "")
    if not requested_model:
        return resp

    body_text = ""
    if resp.status_code != 200:
        body = await resp.aread()
        body_text = body.decode(errors="replace")

    decision = fallback_after_error(requested_model, resp.status_code, body_text)
    if not decision:
        return resp

    await resp.aclose()
    retry_json = dict(json)
    retry_json["model"] = decision.fallback_model
    log.warning(
        "model-fallback: retrying %s with %s after %s",
        requested_model, decision.fallback_model, decision.reason,
    )
    return await _retry_post(
        client, _fallback_endpoint(endpoint, requested_model, decision.fallback_model),
        json=retry_json, headers=headers, provider=provider, request=request,
    )


async def _retry_send_stream_with_model_fallback(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    json: dict,
    headers: dict,
    provider: str = "",
    request: Request | None = None,
    pool: PoolContext | None = None,
) -> httpx.Response:
    resp = await _send_with_pool(
        _retry_send_stream, client, endpoint,
        json=json, headers=headers, provider=provider, request=request,
        pool=_pool_ctx(pool, request, provider),
    )

    requested_model = json.get("model", "")
    if not requested_model:
        return resp

    body_text = ""
    if resp.status_code != 200:
        body = await resp.aread()
        body_text = body.decode(errors="replace")

    decision = fallback_after_error(requested_model, resp.status_code, body_text)
    if not decision:
        return resp

    await resp.aclose()
    retry_json = dict(json)
    retry_json["model"] = decision.fallback_model
    log.warning(
        "model-fallback: retrying %s with %s after %s",
        requested_model, decision.fallback_model, decision.reason,
    )
    return await _retry_send_stream(
        client, _fallback_endpoint(endpoint, requested_model, decision.fallback_model),
        json=retry_json, headers=headers, provider=provider, request=request,
    )
