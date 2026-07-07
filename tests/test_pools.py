"""Workspace-pool schema (src.providers) and failover (src.upstream) tests.

Covers: pool expansion, workspaces: section as provider alias, resolve()
skipping open circuits, provider_override pinning, pool-eligible statuses,
_send_with_pool ordered failover including transport errors and rewiring
endpoints/headers across endpoint styles.
"""

import asyncio

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
    for name in ("ws-a", "ws-b"):
        circuit._circuits.pop(name, None)
    yield
    for name in ("ws-a", "ws-b"):
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
