"""Workspace-pool schema (src.providers) and failover (src.upstream) tests.

Covers: pool expansion, workspaces: section as provider alias, resolve()
skipping open circuits, provider_override pinning, pool-eligible statuses,
_send_with_pool ordered failover including transport errors and rewiring
endpoints/headers across endpoint styles.
"""

import asyncio
import copy
from types import SimpleNamespace

import httpx
import pytest

import src.circuit as circuit
import src.providers as providers
import src.upstream as upstream


POOLED_CONFIG = {
    "workspaces": {
        "ws-a": {
            "base_url": "https://a.example.com",
            "api_key": "key-a",
            "protocol": "openai",
            "path_prefixes": {"anthropic": "anthropic/v1", "openai": "mlflow/v1"},
        },
        "ws-b": {
            "base_url": "https://b.example.com",
            "api_key": "key-b",
            "protocol": "openai",
            "endpoint_style": "invocations",
        },
    },
    "pools": {
        "main-pool": ["ws-a", "ws-b"],
    },
    "models": [
        {
            "name": "pooled-model",
            "alias": "pooled",
            "provider_model_id": "databricks-pooled-model",
            "pool": "main-pool",
            "protocol": "anthropic",
            "context": 1000,
            "max_output_tokens": 100,
        },
        {
            "name": "solo-model",
            "provider": "ws-b",
            "provider_model_id": "databricks-solo-model",
            "context": 1000,
            "max_output_tokens": 100,
        },
    ],
}


@pytest.fixture
def pooled_registry(monkeypatch):
    monkeypatch.setattr(providers, "_config", POOLED_CONFIG)
    monkeypatch.setattr(providers, "_models", None)
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", providers.Path("/nonexistent-model-info.json"))
    yield
    providers._config = None
    providers._models = None


@pytest.fixture
def clean_circuits():
    for name in ("ws-a", "ws-b", "ws-c"):
        circuit._circuits.pop(name, None)
    yield
    for name in ("ws-a", "ws-b", "ws-c"):
        circuit._circuits.pop(name, None)


# ── providers: pool schema ───────────────────────────────────────────────────

def test_pool_members_expansion(pooled_registry):
    entry = providers._load_models()["pooled-model"]
    assert providers._pool_members(entry, POOLED_CONFIG) == ["ws-a", "ws-b"]


def test_pool_members_plain_provider_is_single_member(pooled_registry):
    entry = providers._load_models()["solo-model"]
    assert providers._pool_members(entry, POOLED_CONFIG) == ["ws-b"]


def test_pool_members_unknown_pool_falls_back_to_provider(pooled_registry):
    entry = {"name": "x", "pool": "no-such-pool", "provider": "ws-a"}
    assert providers._pool_members(entry, POOLED_CONFIG) == ["ws-a"]


def test_workspaces_section_resolves_as_provider(pooled_registry):
    info = providers.resolve("solo-model")
    assert info is not None
    assert info.provider == "ws-b"
    assert info.base_url == "https://b.example.com/serving-endpoints/databricks-solo-model/invocations"
    assert info.endpoint_suffix == ""


def test_resolve_pooled_prefers_first_member(pooled_registry, clean_circuits):
    info = providers.resolve("pooled-model")
    assert info is not None
    assert info.provider == "ws-a"
    assert info.base_url == "https://a.example.com/anthropic/v1"


def test_resolve_pooled_skips_open_circuit(pooled_registry, clean_circuits):
    for _ in range(circuit.TRIP_THRESHOLD):
        circuit.record_failure("ws-a", 503, "down")
    assert circuit.is_tripped("ws-a")
    info = providers.resolve("pooled-model")
    assert info is not None
    assert info.provider == "ws-b"


def test_resolve_provider_override_pins_member(pooled_registry, clean_circuits):
    info = providers.resolve("pooled-model", provider_override="ws-b")
    assert info is not None
    assert info.provider == "ws-b"
    assert info.api_key == "key-b"


def test_resolve_provider_override_rejects_non_member(pooled_registry):
    assert providers.resolve("pooled-model", provider_override="google") is None


def test_pool_candidates_by_alias_and_id(pooled_registry):
    assert providers.pool_candidates("pooled") == ["ws-a", "ws-b"]
    assert providers.pool_candidates("databricks-pooled-model") == ["ws-a", "ws-b"]
    assert providers.pool_candidates("unknown-model") == []


def test_model_status_exposes_pool_candidate_providers(pooled_registry):
    row = next(item for item in providers.model_status() if item["name"] == "pooled-model")
    assert row["candidate_providers"] == ["ws-a", "ws-b"]


def test_availability_ok_when_any_pool_member_configured(pooled_registry, monkeypatch):
    config = {
        "workspaces": {
            "ws-a": {"base_url": "", "api_key": ""},  # unconfigured
            "ws-b": POOLED_CONFIG["workspaces"]["ws-b"],
        },
        "pools": POOLED_CONFIG["pools"],
        "models": POOLED_CONFIG["models"],
    }
    monkeypatch.setattr(providers, "_config", config)
    monkeypatch.setattr(providers, "_models", None)
    availability = providers.model_availability("pooled-model")
    assert availability["available"] is True
    assert availability["provider"] == "ws-b"


# ── upstream: pool failover ──────────────────────────────────────────────────

def _pool_ctx() -> upstream.PoolContext:
    return upstream.PoolContext(
        model_key="pooled-model",
        provider="ws-a",
        base_url="https://a.example.com/anthropic/v1",
        api_key="key-a",
    )


@pytest.fixture
def fast_retries(monkeypatch):
    monkeypatch.setattr(upstream, "_RETRY_MAX", 1)
    monkeypatch.setattr(upstream, "_RETRY_429_ATTEMPTS", 1)
    monkeypatch.setattr(upstream, "_RETRY_TRANSPORT_ATTEMPTS", 1)
    monkeypatch.setattr(upstream, "_compute_retry_delay", lambda resp, attempt: 0.0)
    monkeypatch.setattr(upstream, "_compute_transport_retry_delay", lambda attempt: 0.0)


def test_pool_failover_on_saturation(pooled_registry, clean_circuits, fast_retries):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.url.host, str(req.url.path), req.headers.get("authorization")))
        if req.url.host == "a.example.com":
            return httpx.Response(503, text="down")
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await upstream._retry_post_with_model_fallback(
                client, "https://a.example.com/anthropic/v1/messages",
                json={"model": "databricks-pooled-model", "max_tokens": 1},
                headers={"Authorization": "Bearer key-a", "anthropic-version": "2023-06-01"},
                provider="ws-a", pool=_pool_ctx(),
            )
        return resp

    resp = asyncio.run(run())
    assert resp.status_code == 200
    # ws-b is invocations-style: complete URL, original suffix discarded.
    assert calls[-1][0] == "b.example.com"
    assert calls[-1][1] == "/serving-endpoints/databricks-pooled-model/invocations"
    assert calls[-1][2] == "Bearer key-b"


def test_pool_failover_on_404(pooled_registry, clean_circuits, fast_retries):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "a.example.com":
            return httpx.Response(404, text='{"message": "model not found"}')
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await upstream._retry_post_with_model_fallback(
                client, "https://a.example.com/anthropic/v1/messages",
                json={"model": "databricks-pooled-model", "max_tokens": 1},
                headers={"Authorization": "Bearer key-a"},
                provider="ws-a", pool=_pool_ctx(),
            )
        return resp

    resp = asyncio.run(run())
    assert resp.status_code == 200


def test_pool_failover_on_transport_error(pooled_registry, clean_circuits, fast_retries):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "a.example.com":
            raise httpx.ConnectError("workspace deleted (NXDOMAIN)")
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await upstream._retry_post_with_model_fallback(
                client, "https://a.example.com/anthropic/v1/messages",
                json={"model": "databricks-pooled-model", "max_tokens": 1},
                headers={"Authorization": "Bearer key-a"},
                provider="ws-a", pool=_pool_ctx(),
            )
        return resp

    resp = asyncio.run(run())
    assert resp.status_code == 200


def test_no_failover_on_success(pooled_registry, clean_circuits, fast_retries):
    hosts = []

    def handler(req: httpx.Request) -> httpx.Response:
        hosts.append(req.url.host)
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await upstream._retry_post_with_model_fallback(
                client, "https://a.example.com/anthropic/v1/messages",
                json={"model": "databricks-pooled-model", "max_tokens": 1},
                headers={"Authorization": "Bearer key-a"},
                provider="ws-a", pool=_pool_ctx(),
            )
        return resp

    resp = asyncio.run(run())
    assert resp.status_code == 200
    assert hosts == ["a.example.com"]


def test_no_failover_on_client_error(pooled_registry, clean_circuits, fast_retries):
    """400s are the caller's fault; never mask them with a pool retry."""
    hosts = []

    def handler(req: httpx.Request) -> httpx.Response:
        hosts.append(req.url.host)
        return httpx.Response(400, text="bad request")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await upstream._retry_post_with_model_fallback(
                client, "https://a.example.com/anthropic/v1/messages",
                json={"model": "databricks-pooled-model", "max_tokens": 1},
                headers={"Authorization": "Bearer key-a"},
                provider="ws-a", pool=_pool_ctx(),
            )
        return resp

    resp = asyncio.run(run())
    assert resp.status_code == 400
    assert hosts == ["a.example.com"]


def test_all_members_fail_returns_last_response(pooled_registry, clean_circuits, fast_retries):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="everything down")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await upstream._retry_post_with_model_fallback(
                client, "https://a.example.com/anthropic/v1/messages",
                json={"model": "databricks-pooled-model", "max_tokens": 1},
                headers={"Authorization": "Bearer key-a"},
                provider="ws-a", pool=_pool_ctx(),
            )
        return resp

    resp = asyncio.run(run())
    assert resp.status_code == 503


def test_stream_pool_failover(pooled_registry, clean_circuits, fast_retries):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "a.example.com":
            return httpx.Response(503, text="down")
        return httpx.Response(200, content=b"data: ok\n\n")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await upstream._retry_send_stream_with_model_fallback(
                client, "https://a.example.com/anthropic/v1/messages",
                json={"model": "databricks-pooled-model", "max_tokens": 1, "stream": True},
                headers={"Authorization": "Bearer key-a"},
                provider="ws-a", pool=_pool_ctx(),
            )
            assert resp.status_code == 200
            body = await resp.aread()
            assert b"ok" in body
            await resp.aclose()

    asyncio.run(run())
def test_single_member_pool_no_failover(pooled_registry, clean_circuits, fast_retries):
    ctx = upstream.PoolContext(
        model_key="solo-model", provider="ws-b",
        base_url="https://b.example.com/serving-endpoints/databricks-solo-model/invocations",
        api_key="key-b",
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await upstream._retry_post_with_model_fallback(
                client, ctx.base_url,
                json={"model": "databricks-solo-model", "max_tokens": 1},
                headers={"Authorization": "Bearer key-b"},
                provider="ws-b", pool=ctx,
            )
        return resp

    resp = asyncio.run(run())
    assert resp.status_code == 503

def test_model_fallback_rewrites_invocations_endpoint():
    assert upstream._fallback_endpoint(
        "https://ws.example.com/serving-endpoints/databricks-gpt-5-5/invocations",
        "databricks-gpt-5-5", "databricks-gpt-5-4",
    ) == "https://ws.example.com/serving-endpoints/databricks-gpt-5-4/invocations"
    # Path-prefix endpoints (model only in body) are unchanged.
    assert upstream._fallback_endpoint(
        "https://gw.example.com/mlflow/v1/chat/completions",
        "databricks-gpt-5-5", "databricks-gpt-5-4",
    ) == "https://gw.example.com/mlflow/v1/chat/completions"


# Use real retry budgets here: failover must not depend on shrinking them.
@pytest.fixture(params=[
    upstream._retry_post_with_model_fallback,
    upstream._retry_send_stream_with_model_fallback,
], ids=["post", "stream"])
def pool_send(request):
    return request.param


@pytest.fixture
def no_retry_waits(monkeypatch):
    async def unexpected_wait(*args, **kwargs):
        pytest.fail("workspace failover must not sleep or wait for circuit recovery")

    monkeypatch.setattr(upstream, "_sleep_or_disconnect", unexpected_wait)
    monkeypatch.setattr(upstream, "wait_for_recovery", unexpected_wait)


async def _send_pool(pool_send, client):
    return await pool_send(
        client, "https://a.example.com/anthropic/v1/messages",
        json={"model": "databricks-pooled-model", "max_tokens": 1},
        headers={"Authorization": "Bearer key-a", "anthropic-version": "2023-06-01"},
        provider="ws-a", pool=_pool_ctx(),
    )


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504, 404, 400, 413, 422])
def test_pool_tries_backup_without_retry_ladder(
    pooled_registry, clean_circuits, no_retry_waits, pool_send, status,
):
    hosts = []
    responses = []

    def handler(req):
        hosts.append(req.url.host)
        response = httpx.Response(
            status if req.url.host == "a.example.com" else 200,
            text="response", headers={"Retry-After": "60"},
        )
        responses.append(response)
        return response

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await _send_pool(pool_send, client)
            await response.aread()
            await response.aclose()
            return response.status_code

    result = asyncio.run(run())
    if status in (400, 413, 422):
        assert result == status
        assert hosts == ["a.example.com"]
    else:
        assert result == 200
        assert hosts == ["a.example.com", "b.example.com"]
    assert all(response.is_closed for response in responses)


@pytest.mark.parametrize("error", [
    httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError,
])
def test_pool_transport_failure_tries_backup_once(
    pooled_registry, clean_circuits, no_retry_waits, pool_send, error,
):
    hosts = []

    def handler(req):
        hosts.append(req.url.host)
        if req.url.host == "a.example.com":
            raise error("workspace unavailable", request=req)
        return httpx.Response(200, text="ok")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await _send_pool(pool_send, client)
            await response.aclose()
            return response.status_code

    assert asyncio.run(run()) == 200
    assert hosts == ["a.example.com", "b.example.com"]
    assert circuit.get_status()["ws-a"]["consecutive_failures"] == 1


@pytest.mark.parametrize("already_open", [False, True])
def test_pool_does_not_wait_for_primary_circuit(
    pooled_registry, clean_circuits, no_retry_waits, pool_send, already_open,
):
    for _ in range(circuit.TRIP_THRESHOLD - (0 if already_open else 1)):
        circuit.record_failure("ws-a", 503, "down")
    hosts = []

    def handler(req):
        hosts.append(req.url.host)
        return httpx.Response(503 if req.url.host == "a.example.com" else 200, text="response")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await _send_pool(pool_send, client)
            await response.aclose()
            return response.status_code

    assert asyncio.run(run()) == 200
    assert hosts == (["b.example.com"] if already_open else ["a.example.com", "b.example.com"])
    assert circuit.is_tripped("ws-a")


@pytest.mark.parametrize("connect_timeout", [900.0, 2.0, None])
def test_pool_caps_connect_timeout_but_preserves_read_budget(
    pooled_registry, clean_circuits, no_retry_waits, pool_send, connect_timeout,
):
    timeouts = []

    def handler(req):
        timeouts.append(req.extensions["timeout"])
        return httpx.Response(503 if req.url.host == "a.example.com" else 200, text="response")

    async def run():
        timeout = httpx.Timeout(900, connect=connect_timeout)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=timeout) as client:
            response = await _send_pool(pool_send, client)
            await response.aclose()
            assert client.timeout.connect == connect_timeout

    asyncio.run(run())
    assert timeouts[0] == {
        "connect": 2.0 if connect_timeout == 2.0 else 5.0,
        "read": 900, "write": 900, "pool": 900,
    }
    assert timeouts[1]["connect"] == connect_timeout


@pytest.mark.parametrize("cached_token", [None, "fresh-key"])
@pytest.mark.parametrize("status", [401, 403])
def test_pool_auth_uses_cached_refresh_without_browser_sso(
    pooled_registry, clean_circuits, no_retry_waits, pool_send, monkeypatch, cached_token, status,
):
    preflights = []
    refreshes = []
    hosts = []

    async def ensure(provider, *, allow_login):
        preflights.append((provider, allow_login))
        return None

    async def refresh(provider, *, force, allow_login):
        refreshes.append((provider, force, allow_login))
        return cached_token

    monkeypatch.setattr(upstream, "ensure_fresh_oauth_token", ensure)
    monkeypatch.setattr(upstream, "refresh_oauth_token", refresh)

    def handler(req):
        hosts.append(req.url.host)
        if req.url.host == "a.example.com" and req.headers["authorization"] != "Bearer fresh-key":
            return httpx.Response(status, text="expired")
        return httpx.Response(200, text="ok")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await _send_pool(pool_send, client)
            await response.aclose()
            return response.status_code

    assert asyncio.run(run()) == 200
    assert refreshes == [("ws-a", True, False)]
    if cached_token:
        assert hosts == ["a.example.com", "a.example.com"]
        assert preflights == [("ws-a", False)]
    else:
        assert hosts == ["a.example.com", "b.example.com"]
        assert preflights == [("ws-a", False), ("ws-b", True)]


def test_pool_final_member_keeps_retries(
    pooled_registry, clean_circuits, pool_send, monkeypatch,
):
    config = copy.deepcopy(POOLED_CONFIG)
    config["workspaces"]["ws-c"] = dict(config["workspaces"]["ws-b"], base_url="https://c.example.com")
    config["pools"]["main-pool"].append("ws-c")
    monkeypatch.setattr(providers, "_config", config)
    monkeypatch.setattr(upstream, "_compute_retry_delay", lambda resp, attempt: 0)
    hosts = []

    def handler(req):
        hosts.append(req.url.host)
        return httpx.Response(429, text="rate limited")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await _send_pool(pool_send, client)
            assert await response.aread() == b"rate limited"
            await response.aclose()
            return response.status_code

    assert asyncio.run(run()) == 429
    assert hosts == ["a.example.com", "b.example.com"] + ["c.example.com"] * upstream._RETRY_429_ATTEMPTS


@pytest.mark.parametrize("unresolvable_backup", [False, True])
def test_pool_without_usable_backup_keeps_primary_retries(
    pooled_registry, clean_circuits, pool_send, monkeypatch, unresolvable_backup,
):
    if unresolvable_backup:
        monkeypatch.setattr(providers, "resolve", lambda *args, **kwargs: None)
    else:
        monkeypatch.setattr(providers, "pool_candidates", lambda model: ["ws-a"])
    monkeypatch.setattr(upstream, "_compute_retry_delay", lambda resp, attempt: 0)
    hosts = []

    def handler(req):
        hosts.append(req.url.host)
        return httpx.Response(429, text="rate limited")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await _send_pool(pool_send, client)
            await response.aclose()
            return response.status_code

    assert asyncio.run(run()) == 429
    assert hosts == ["a.example.com"] * upstream._RETRY_429_ATTEMPTS


@pytest.mark.parametrize("error", [ValueError, asyncio.CancelledError])
def test_pool_does_not_mask_local_errors_or_cancellation(
    pooled_registry, clean_circuits, pool_send, error,
):
    hosts = []

    def handler(req):
        hosts.append(req.url.host)
        raise error("stop")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await _send_pool(pool_send, client)

    with pytest.raises(error, match="stop"):
        asyncio.run(run())
    assert hosts == ["a.example.com"]


def test_pool_resolves_backup_credentials_after_primary_finishes(
    pooled_registry, clean_circuits, pool_send, monkeypatch,
):
    config = copy.deepcopy(POOLED_CONFIG)
    monkeypatch.setattr(providers, "_config", config)
    auth = []

    def handler(req):
        auth.append(req.headers["authorization"])
        if req.url.host == "a.example.com":
            config["workspaces"]["ws-b"]["api_key"] = "rotated-key"
            return httpx.Response(503, text="down")
        status = 200 if req.headers["authorization"] == "Bearer rotated-key" else 401
        return httpx.Response(status, text="response")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await _send_pool(pool_send, client)
            await response.aclose()
            return response.status_code

    assert asyncio.run(run()) == 200
    assert auth == ["Bearer key-a", "Bearer rotated-key"]


@pytest.mark.parametrize("failure", [503, httpx.ConnectError])
def test_pool_preserves_failure_if_backup_disappears(
    pooled_registry, clean_circuits, no_retry_waits, pool_send, monkeypatch, failure,
):
    config = copy.deepcopy(POOLED_CONFIG)
    monkeypatch.setattr(providers, "_config", config)
    hosts = []

    def handler(req):
        hosts.append(req.url.host)
        config["workspaces"]["ws-b"]["enabled"] = False
        if failure == 503:
            return httpx.Response(503, text="original error")
        raise failure("original error", request=req)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await _send_pool(pool_send, client)
            assert response.status_code == 503
            assert await response.aread() == b"original error"
            await response.aclose()

    if failure == 503:
        asyncio.run(run())
    else:
        with pytest.raises(failure, match="original error"):
            asyncio.run(run())
    assert hosts == ["a.example.com"]


@pytest.mark.parametrize("disconnect", [False, True])
def test_pool_closes_lazy_error_stream_on_handoff_or_disconnect(
    pooled_registry, clean_circuits, disconnect,
):
    hosts = []

    class ErrorStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            pytest.fail("discarded error streams should not be drained")
            yield b"unused"

        async def aclose(self):
            self.closed = True

    stream = ErrorStream()
    error = httpx.Response(404, stream=stream)

    class DownstreamRequest:
        state = SimpleNamespace()
        checks = 0

        async def is_disconnected(self):
            self.checks += 1
            return disconnect and self.checks >= 3

    def handler(req):
        hosts.append(req.url.host)
        return error if req.url.host == "a.example.com" else httpx.Response(200, text="ok")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await upstream._retry_send_stream_with_model_fallback(
                client, "https://a.example.com/anthropic/v1/messages",
                json={"model": "databricks-pooled-model"},
                headers={"Authorization": "Bearer key-a"},
                provider="ws-a", pool=_pool_ctx(), request=DownstreamRequest(),
            )
            assert response.status_code == 200
            await response.aclose()

    if disconnect:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run())
        assert hosts == ["a.example.com"]
    else:
        asyncio.run(run())
        assert hosts == ["a.example.com", "b.example.com"]
    assert stream.closed
    assert error.is_closed


def test_pool_never_replays_a_started_stream(pooled_registry, clean_circuits):
    hosts = []

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: first-token\n\n"
            raise httpx.ReadError("stream interrupted")

        async def aclose(self):
            pass

    def handler(req):
        hosts.append(req.url.host)
        return httpx.Response(200, stream=BrokenStream())

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await _send_pool(upstream._retry_send_stream_with_model_fallback, client)
            assert not response.is_closed
            try:
                chunks = response.aiter_bytes()
                assert await anext(chunks) == b"data: first-token\n\n"
                with pytest.raises(httpx.ReadError, match="interrupted"):
                    await anext(chunks)
            finally:
                await response.aclose()

    asyncio.run(run())
    assert hosts == ["a.example.com"]
