"""Tests for the upstream retry wrappers (src.upstream) using httpx.MockTransport.

Covers: transient 5xx retry in _retry_post, the 401 OAuth-refresh branch,
_retry_send_stream returning an open readable stream after a retry, and the
config-driven model-fallback wrapper.

All retry counts and delay computations are monkeypatched small/zero so the
suite stays fast; circuit state is cleaned via a fixture with teardown.
"""

import asyncio
import json as jsonlib

import httpx
import pytest

import src.circuit as circuit
import src.providers as providers
import src.upstream as upstream


@pytest.fixture
def fast_retries(monkeypatch):
    """Shrink retry budgets and zero all backoff delays."""
    monkeypatch.setattr(upstream, "_RETRY_MAX", 2)
    monkeypatch.setattr(upstream, "_RETRY_429_ATTEMPTS", 2)
    monkeypatch.setattr(upstream, "_RETRY_TRANSPORT_ATTEMPTS", 2)
    monkeypatch.setattr(upstream, "_compute_retry_delay", lambda resp, attempt: 0.0)
    monkeypatch.setattr(upstream, "_compute_transport_retry_delay", lambda attempt: 0.0)


@pytest.fixture
def clean_circuit():
    """Provide a circuit-state cleaner; guarantees teardown even on failure."""
    used: set[str] = set()

    def _use(provider: str) -> str:
        used.add(provider)
        circuit._circuits.pop(provider, None)
        return provider

    try:
        yield _use
    finally:
        for provider in used:
            circuit._circuits.pop(provider, None)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _DisconnectedRequest:
    def __init__(self, disconnected_after: int = 0):
        self._checks = 0
        self._disconnected_after = disconnected_after

    async def is_disconnected(self) -> bool:
        self._checks += 1
        return self._checks > self._disconnected_after


# ── (a) _retry_post: 503 then 200 ────────────────────────────────────────────


def test_retry_post_retries_503_then_succeeds(fast_retries, clean_circuit):
    provider = clean_circuit("retry-prov-a")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with _client(handler) as client:
            return await upstream._retry_post(
                client, "https://up.example.com/v1/chat/completions",
                json={"model": "m"}, headers={}, provider=provider,
            )

    resp = asyncio.run(run())
    assert resp.status_code == 200
    assert len(calls) == 2


def test_retry_post_stops_on_disconnect_before_retry(fast_retries, clean_circuit):
    provider = clean_circuit("retry-prov-disconnect-post")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, text="unavailable")

    async def run():
        async with _client(handler) as client:
            await upstream._retry_post(
                client, "https://up.example.com/v1/chat/completions",
                json={"model": "m"}, headers={}, provider=provider,
                request=_DisconnectedRequest(disconnected_after=1),
            )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert len(calls) == 1


# ── (b) _retry_post: 401 → OAuth refresh → retried with new bearer ──────────


def test_retry_post_401_refreshes_token_and_retries(fast_retries, clean_circuit, monkeypatch):
    provider = clean_circuit("retry-prov-b")
    calls = []

    async def fake_refresh(prov: str, *, force: bool = False) -> str | None:
        assert prov == provider
        assert force is True
        return "eyJnew"

    # src.upstream binds refresh_oauth_token by value at import; patch both.
    monkeypatch.setattr(providers, "refresh_oauth_token", fake_refresh)
    monkeypatch.setattr(upstream, "refresh_oauth_token", fake_refresh)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers.get("Authorization") != "Bearer eyJnew":
            return httpx.Response(401, json={"error": {"message": "expired"}})
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with _client(handler) as client:
            return await upstream._retry_post(
                client, "https://up.example.com/v1/chat/completions",
                json={"model": "m"}, headers={"Authorization": "Bearer eyJold"},
                provider=provider,
            )

    resp = asyncio.run(run())
    assert resp.status_code == 200
    assert len(calls) == 2
    assert calls[0].headers["Authorization"] == "Bearer eyJold"
    assert calls[1].headers["Authorization"] == "Bearer eyJnew"


def test_retry_post_preflight_refreshes_near_expiry_token(fast_retries, clean_circuit, monkeypatch):
    provider = clean_circuit("retry-prov-preflight")
    calls = []

    async def fake_ensure(prov: str) -> str | None:
        assert prov == provider
        return "eyJfresh"

    monkeypatch.setattr(upstream, "ensure_fresh_oauth_token", fake_ensure)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with _client(handler) as client:
            return await upstream._retry_post(
                client, "https://up.example.com/v1/chat/completions",
                json={"model": "m"}, headers={"Authorization": "Bearer eyJold"},
                provider=provider,
            )

    resp = asyncio.run(run())
    assert resp.status_code == 200
    assert len(calls) == 1
    assert calls[0].headers["Authorization"] == "Bearer eyJfresh"


# ── (c) _retry_send_stream: 503 then 200 stream ──────────────────────────────


class _SSEStream(httpx.AsyncByteStream):
    """A genuinely lazy byte stream so the response is returned unread/open."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        pass


def test_retry_send_stream_retries_503_then_streams(fast_retries, clean_circuit):
    provider = clean_circuit("retry-prov-c")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(
            200,
            stream=_SSEStream([b'data: {"choices": []}\n\n', b"data: [DONE]\n\n"]),
            headers={"content-type": "text/event-stream"},
        )

    async def run():
        async with _client(handler) as client:
            resp = await upstream._retry_send_stream(
                client, "https://up.example.com/v1/chat/completions",
                json={"model": "m", "stream": True}, headers={}, provider=provider,
            )
            assert resp.status_code == 200
            assert not resp.is_closed  # returned open for the caller to consume
            chunks = [chunk async for chunk in resp.aiter_bytes()]
            await resp.aclose()
            return chunks

    chunks = asyncio.run(run())
    assert b"[DONE]" in b"".join(chunks)
    assert len(calls) == 2


def test_retry_send_stream_stops_on_disconnect_before_retry(fast_retries, clean_circuit):
    provider = clean_circuit("retry-prov-disconnect-stream")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, text="unavailable")

    async def run():
        async with _client(handler) as client:
            await upstream._retry_send_stream(
                client, "https://up.example.com/v1/chat/completions",
                json={"model": "m", "stream": True}, headers={}, provider=provider,
                request=_DisconnectedRequest(disconnected_after=1),
            )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert len(calls) == 1


# ── (d) model fallback wrapper: model-a exhausted → model-b served ───────────


def test_retry_post_with_model_fallback_switches_model(
    fast_retries, clean_circuit, tmp_path, monkeypatch,
):
    provider = clean_circuit("retry-prov-d")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("providers: {}\nmodel_fallbacks:\n  model-a: model-b\n")
    monkeypatch.setattr(providers, "CONFIG_PATH", cfg)
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", tmp_path / "model-info.json")
    (tmp_path / "model-info.json").write_text(jsonlib.dumps({"llm": []}))
    providers.reload()

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = jsonlib.loads(request.content)
        calls.append(body["model"])
        if body["model"] == "model-a":
            return httpx.Response(503, text="saturated")
        return httpx.Response(200, json={"model": "model-b", "ok": True})

    async def run():
        async with _client(handler) as client:
            return await upstream._retry_post_with_model_fallback(
                client, "https://up.example.com/v1/chat/completions",
                json={"model": "model-a"}, headers={}, provider=provider,
            )

    try:
        resp = asyncio.run(run())
        assert resp.status_code == 200
        assert resp.json()["model"] == "model-b"
        # model-a exhausted its (shrunk) retry budget, then model-b succeeded.
        assert calls == ["model-a", "model-a", "model-b"]
    finally:
        providers.reload()
