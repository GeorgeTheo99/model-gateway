"""Tests for pricing lookup and the ledger middleware end-to-end."""

import sqlite3
from types import SimpleNamespace

import pytest

from src.providers import CompositeRoute, ProviderInfo, pricing_for, pricing_status_for
import src.server as server_module
from src.server import app
from src import ledger

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None


def test_pricing_for_known_model(monkeypatch):
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "claude-sonnet-4.6": {
            "name": "claude-sonnet-4.6", "provider": "anthropic",
            "pricing": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
        },
    })
    p = pricing_for("claude-sonnet-4.6")
    assert p is not None
    assert p["input"] == 3.0
    assert p["output"] == 15.0
    assert p["cache_read"] == 0.3
    assert p["cache_write"] == 3.75


def test_pricing_for_local_unmetered_model_is_explicit(monkeypatch):
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "qwen3.5-397b": {
            "name": "qwen3.5-397b", "provider": "omlx", "pricing_status": "unmetered",
        },
    })
    assert pricing_for("qwen3.5-397b") is None
    assert pricing_status_for("qwen3.5-397b") == "unmetered"


def test_kimi_k3_pricing_matches_moonshot_rates(monkeypatch):
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "kimi-k3": {
            "name": "kimi-k3", "provider": "moonshot",
            "pricing": {"input": 3.0, "output": 15.0, "cache_read": 0.3},
        },
    })
    assert pricing_for("kimi-k3") == {"input": 3.0, "output": 15.0, "cache_read": 0.3}
    assert pricing_status_for("kimi-k3") == "metered"


def test_pricing_for_unknown_model_is_none(monkeypatch):
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {})
    assert pricing_for("does-not-exist-xyz") is None
    assert pricing_status_for("does-not-exist-xyz") == "unknown"


def test_cloud_unmetered_marker_fails_safe_to_unknown(monkeypatch):
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "bad-cloud": {
            "name": "bad-cloud", "provider": "openai", "pricing_status": "unmetered",
        },
    })
    assert pricing_status_for("bad-cloud") == "unknown"
    monkeypatch.setattr(providers, "_models", {
        "bad-pool": {
            "name": "bad-pool", "provider": "omlx", "pool": "cloud-pool",
            "pricing_status": "unmetered",
        },
    })
    assert pricing_status_for("bad-pool") == "unknown"


def _info(provider="test", provider_model_id="upstream", vision=False):
    return ProviderInfo(
        provider=provider,
        base_url="http://up",
        api_key="k",
        provider_model_id=provider_model_id,
        protocol="openai",
        context=0,
        max_output_tokens=32768,
        thinking="",
        vision=vision,
    )


@pytest.fixture
def tmp_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_LEDGER_PATH", str(tmp_path / "ledger.db"))
    ledger.init()
    yield tmp_path / "ledger.db"


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_non_streaming_request_records_ledger_with_cost(tmp_ledger, monkeypatch):
    """A /v1/chat/completions call records tokens + cost from the response usage."""
    info = _info(provider="anthropic", provider_model_id="claude-sonnet-4-6")
    # pricing_for looks up _models by the gateway model name; seed it.
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "priced-model": {
            "name": "priced-model",
            "provider": "anthropic",
            "provider_model_id": "claude-sonnet-4-6",
            "pricing": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
        },
    })

    def fake_resolve(model):
        return info if model == "priced-model" else None

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        return server_module.JSONResponse(status_code=200, content={
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100000,
                "completion_tokens": 50000,
                "prompt_tokens_details": {"cached_tokens": 20000},
            },
        })

    monkeypatch.setattr(server_module, "resolve", fake_resolve)
    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)

    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json={
            "model": "priced-model",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt_cache_key": "raw-openai-session",
        })
        assert resp.status_code == 200

    rows = ledger.recent()
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "priced-model"
    assert r["provider"] == "anthropic"
    assert r["status"] == 200
    # OpenAI-shape: input_tokens = prompt_tokens - cached = 100k - 20k = 80k
    assert r["input_tokens"] == 80000
    assert r["output_tokens"] == 50000
    assert r["cached_read_tokens"] == 20000
    assert r["usage_reported"] == 1
    # 80k*3 + 20k*0.3 + 50k*15 per 1M = 0.24 + 0.006 + 0.75 = 0.996
    # (cached tokens priced once at cache_read, not double-counted at input rate)
    assert r["cost_usd"] == pytest.approx(0.996, abs=1e-6)
    assert r["pricing_complete"] == 1
    assert r["session_observed"] is True
    assert r["session_source"] == "prompt_cache_key"
    assert r["cache_retention_requested"] == "short"
    assert "session_fingerprint" not in r


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_native_anthropic_cache_usage_and_markers_are_preserved(tmp_ledger, monkeypatch):
    """New Anthropic usage details must not hide cache tokens from the ledger."""
    info = _info(provider="anthropic", provider_model_id="claude-opus-5")
    info.protocol = "anthropic"
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "opus5": {
            "name": "opus5",
            "provider": "anthropic",
            "provider_model_id": "claude-opus-5",
            "pricing": {
                "input": 5.0,
                "output": 25.0,
                "cache_read": 0.5,
                "cache_write": 6.25,
                "cache_write_1h": 10.0,
            },
        },
    })
    monkeypatch.setattr(server_module, "resolve", lambda model: info if model == "opus5" else None)

    async def fake_passthrough_anthropic_sync(endpoint, body, headers, **kwargs):
        assert body["model"] == "claude-opus-5"
        assert body["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        return server_module.JSONResponse(status_code=200, content={
            "type": "message",
            "model": "claude-opus-5",
            "content": [{"type": "text", "text": "OK"}],
            "usage": {
                "input_tokens": 14,
                "output_tokens": 4,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 14_403,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 0,
                    "ephemeral_1h_input_tokens": 14_403,
                },
                "output_tokens_details": {"thinking_tokens": 3},
            },
        })

    monkeypatch.setattr(
        server_module,
        "_passthrough_anthropic_sync",
        fake_passthrough_anthropic_sync,
    )

    raw_session_id = "raw-anthropic-session"
    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            headers={"x-session-affinity": raw_session_id},
            json={
                "model": "opus5",
                "max_tokens": 16,
                "system": [{
                    "type": "text",
                    "text": "stable prefix",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }],
                "messages": [{"role": "user", "content": "Reply OK"}],
            },
        )
        assert response.status_code == 200

    row = ledger.recent()[0]
    assert row["input_tokens"] == 14
    assert row["output_tokens"] == 4
    assert row["cached_read_tokens"] == 0
    assert row["cache_write_tokens"] == 0
    assert row["cache_write_1h_tokens"] == 14_403
    assert row["reasoning_tokens"] == 3
    assert row["cost_usd"] == pytest.approx(0.1442, abs=1e-9)
    assert row["pricing_complete"] == 1
    assert row["session_observed"] is True
    assert row["session_source"] == "x-session-affinity"
    assert row["cache_retention_requested"] == "long"
    assert "session_fingerprint" not in row
    with sqlite3.connect(tmp_ledger) as conn:
        fingerprint, request_started_at = conn.execute(
            "SELECT session_fingerprint, request_started_at FROM requests"
        ).fetchone()
    assert fingerprint == ledger.session_fingerprint(raw_session_id)
    assert request_started_at is not None
    assert raw_session_id not in tmp_ledger.read_bytes().decode("latin-1")

    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin-key")
    with TestClient(app) as client:
        admin_response = client.get(
            "/admin/api/requests",
            headers={"Authorization": "Bearer admin-key"},
        )
    assert admin_response.status_code == 200
    admin_text = admin_response.text
    assert raw_session_id not in admin_text
    assert fingerprint not in admin_text
    assert admin_response.json()["requests"][0]["session_observed"] is True


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_native_anthropic_stream_records_split_cache_usage(tmp_ledger, monkeypatch):
    info = _info(provider="anthropic", provider_model_id="claude-opus-5")
    info.protocol = "anthropic"
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "opus5-stream": {
            "name": "opus5-stream",
            "provider": "anthropic",
            "provider_model_id": "claude-opus-5",
            "pricing": {
                "input": 5.0,
                "output": 25.0,
                "cache_read": 0.5,
                "cache_write": 6.25,
                "cache_write_1h": 10.0,
            },
        },
    })
    monkeypatch.setattr(
        server_module,
        "resolve",
        lambda model: info if model == "opus5-stream" else None,
    )

    async def fake_passthrough_anthropic_stream(endpoint, body, headers, **kwargs):
        async def chunks():
            yield (
                'data: {"type":"message_start","message":{"usage":'
                '{"input_tokens":14,"output_tokens":0,"cache_read_input_tokens":0,'
                '"cache_creation_input_tokens":300,"cache_creation":'
                '{"ephemeral_5m_input_tokens":100,"ephemeral_1h_input_tokens":200}}}}\n\n'
            ).encode()
            yield (
                'data: {"type":"message_delta","usage":{"output_tokens":4,'
                '"output_tokens_details":{"thinking_tokens":3}}}\n\n'
            ).encode()
        return server_module.StreamingResponse(chunks(), media_type="text/event-stream")

    monkeypatch.setattr(
        server_module,
        "_passthrough_anthropic_stream",
        fake_passthrough_anthropic_stream,
    )

    with TestClient(app) as client:
        response = client.post("/v1/messages", json={
            "model": "opus5-stream",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "Reply OK"}],
        })
        assert response.status_code == 200

    row = ledger.recent()[0]
    assert row["input_tokens"] == 14
    assert row["output_tokens"] == 4
    assert row["cached_read_tokens"] == 0
    assert row["cache_write_tokens"] == 100
    assert row["cache_write_1h_tokens"] == 200
    assert row["reasoning_tokens"] == 3
    assert row["cost_usd"] == pytest.approx(0.002795, abs=1e-9)
    assert row["pricing_complete"] == 1


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_semantic_composite_request_records_requested_and_resolved_model_ids(tmp_ledger, monkeypatch):
    info = _info(provider="omlx", provider_model_id="GLM-5.2-MLX-4.5bit")
    info.composite = CompositeRoute(
        text_model="glm-5.2-4.5bit",
        vision_model="gemma4-26b",
        image_handling="extract_then_answer",
        max_images=4,
    )
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "auto-local": {
            "name": "auto-local",
            "provider": "omlx",
            "omlx_id": "auto-local",
            "pricing_status": "unmetered",
        },
    })
    monkeypatch.setattr(server_module, "resolve", lambda model: info if model == "auto-local" else None)

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        assert body["model"] == "GLM-5.2-MLX-4.5bit"
        return server_module.JSONResponse(status_code=200, content={"choices": []})

    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "auto-local", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200

    row = ledger.recent()[0]
    assert row["model"] == "auto-local"
    assert row["provider_model_id"] == "GLM-5.2-MLX-4.5bit"


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_composite_validation_error_keeps_resolution_receipt(tmp_ledger, monkeypatch):
    info = _info(provider="omlx", provider_model_id="GLM-5.2-MLX-4.5bit")
    info.composite = CompositeRoute(
        text_model="glm-5.2-4.5bit",
        vision_model="gemma4-26b",
        image_handling="extract_then_answer",
        max_images=4,
    )
    vision = _info(provider="omlx", provider_model_id="gemma4-26b-upstream", vision=True)
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "auto-local": {
            "name": "auto-local",
            "provider": "omlx",
            "omlx_id": "auto-local",
            "pricing_status": "unmetered",
        },
    })
    monkeypatch.setattr(
        server_module,
        "resolve",
        lambda model: info if model == "auto-local" else vision if model == "gemma4-26b" else None,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto-local",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/not-inline.png"},
                    }],
                }],
            },
        )
        assert response.status_code == 400

    row = ledger.recent()[0]
    assert row["model"] == "auto-local"
    assert row["provider"] == "omlx"
    assert row["provider_model_id"] == "GLM-5.2-MLX-4.5bit"


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_direct_stream_requests_usage_and_prices_cache_hits(tmp_ledger, monkeypatch):
    info = _info(provider="moonshot", provider_model_id="kimi-k3")
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "kimi-k3": {
            "name": "kimi-k3",
            "provider": "moonshot",
            "provider_model_id": "kimi-k3",
            "pricing": {"input": 3.0, "output": 15.0, "cache_read": 0.3},
        },
    })
    monkeypatch.setattr(server_module, "resolve", lambda model: info if model == "kimi-k3" else None)

    async def fake_passthrough_stream(endpoint, body, headers, **kwargs):
        assert body["stream_options"] == {"include_usage": True}

        async def chunks():
            yield b'data: {"choices":[],"us'
            yield b'age":{"prompt_tokens":100000,"completion_tokens":50000,'
            yield b'"cached_tokens":40000}}\n'
            yield b'\ndata: [DONE]\n\n'
        return server_module.StreamingResponse(chunks(), media_type="text/event-stream")

    monkeypatch.setattr(server_module, "_passthrough_stream", fake_passthrough_stream)

    with TestClient(app) as c:
        response = c.post("/v1/chat/completions", json={
            "model": "kimi-k3", "messages": [], "stream": True,
        })
        assert response.status_code == 200

    row = ledger.recent()[0]
    assert row["usage_reported"] == 1
    assert row["input_tokens"] == 60000
    assert row["cached_read_tokens"] == 40000
    assert row["output_tokens"] == 50000
    assert row["cost_usd"] == pytest.approx(0.942)
    assert row["pricing_complete"] == 1


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_streaming_error_event_is_recorded_as_failure(tmp_ledger, monkeypatch):
    info = _info(provider="openai", provider_model_id="gpt-test")
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "stream-error": {
            "name": "stream-error",
            "provider": "openai",
            "provider_model_id": "gpt-test",
            "pricing": {"input": 2.0, "output": 4.0},
        },
    })
    monkeypatch.setattr(server_module, "resolve", lambda model: info if model == "stream-error" else None)

    async def fake_passthrough_stream(endpoint, body, headers, **kwargs):
        async def chunks():
            yield server_module._stream_error_event("provider stream truncated").encode()
        return server_module.StreamingResponse(chunks(), media_type="text/event-stream")

    monkeypatch.setattr(server_module, "_passthrough_stream", fake_passthrough_stream)

    with TestClient(app) as c:
        response = c.post("/v1/chat/completions", json={
            "model": "stream-error", "messages": [], "stream": True,
        })
        assert response.status_code == 200

    row = ledger.recent()[0]
    assert row["error"] == "provider stream truncated"
    aggregate = ledger.aggregate(group_by="model")[0]
    assert aggregate["ok"] == 0
    assert aggregate["errors"] == 1


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_translated_messages_stream_error_is_recorded(tmp_ledger, monkeypatch):
    info = _info(provider="fireworks", provider_model_id="fw-test")
    import src.providers as providers
    from src.streaming import translate_stream
    monkeypatch.setattr(providers, "_models", {
        "translated-messages": {
            "name": "translated-messages",
            "provider": "fireworks",
            "provider_model_id": "fw-test",
            "pricing": {"input": 1.0, "output": 2.0},
        },
    })
    monkeypatch.setattr(server_module, "resolve", lambda model: info if model == "translated-messages" else None)

    async def fake_handle_streaming(*args, **kwargs):
        async def upstream():
            yield b'data: {"error":{"type":"api_error","message":"messages upstream failed"}}\n\n'
        return server_module.StreamingResponse(
            translate_stream(upstream(), "translated-messages"),
            media_type="text/event-stream",
        )

    monkeypatch.setattr(server_module, "_handle_streaming", fake_handle_streaming)
    with TestClient(app) as c:
        response = c.post("/v1/messages", json={
            "model": "translated-messages", "messages": [], "stream": True,
        })
        assert response.status_code == 200

    row = ledger.recent()[0]
    assert row["error"] == "messages upstream failed"
    assert ledger.summary()["errors"] == 1


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_translated_responses_stream_error_is_recorded(tmp_ledger, monkeypatch):
    info = _info(provider="fireworks", provider_model_id="fw-test")
    import src.providers as providers
    from src.responses import translate_responses_stream
    monkeypatch.setattr(providers, "_models", {
        "translated-responses": {
            "name": "translated-responses",
            "provider": "fireworks",
            "provider_model_id": "fw-test",
            "pricing": {"input": 1.0, "output": 2.0},
        },
    })
    monkeypatch.setattr(server_module, "resolve", lambda model: info if model == "translated-responses" else None)

    async def fake_handle_responses_stream(*args, **kwargs):
        async def upstream():
            yield b'data: {"error":{"type":"api_error","message":"responses upstream failed"}}\n\n'
        return server_module.StreamingResponse(
            translate_responses_stream(upstream(), "translated-responses"),
            media_type="text/event-stream",
        )

    monkeypatch.setattr(server_module, "_handle_responses_stream", fake_handle_responses_stream)
    with TestClient(app) as c:
        response = c.post("/v1/responses", json={
            "model": "translated-responses", "input": "hi", "stream": True,
        })
        assert response.status_code == 200

    row = ledger.recent()[0]
    assert row["error"] == "responses upstream failed"
    assert ledger.summary()["errors"] == 1


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_unpriced_model_records_unknown_cost(tmp_ledger, monkeypatch):
    info = _info(provider="openai", provider_model_id="gpt-5.4")
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "unpriced": {"name": "unpriced", "provider": "openai", "provider_model_id": "gpt-5.4"},
    })

    def fake_resolve(model):
        return info if model == "unpriced" else None

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        return server_module.JSONResponse(status_code=200, content={
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        })

    monkeypatch.setattr(server_module, "resolve", fake_resolve)
    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)

    with TestClient(app) as c:
        c.post("/v1/chat/completions", json={"model": "unpriced", "messages": []})

    r = ledger.recent()[0]
    assert r["input_tokens"] == 100
    assert r["cost_usd"] is None
    assert r["pricing_complete"] == 0


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_unmetered_model_records_known_zero_without_usage(tmp_ledger, monkeypatch):
    info = _info(provider="omlx", provider_model_id="local-model")
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "local-model": {
            "name": "local-model",
            "provider": "omlx",
            "omlx_id": "local-model",
            "pricing_status": "unmetered",
        },
    })

    monkeypatch.setattr(server_module, "resolve", lambda model: info if model == "local-model" else None)

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        return server_module.JSONResponse(status_code=200, content={"choices": []})

    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)

    with TestClient(app) as c:
        response = c.post("/v1/chat/completions", json={"model": "local-model", "messages": []})
        assert response.status_code == 200

    row = ledger.recent()[0]
    assert row["usage_reported"] == 0
    assert row["cost_usd"] == 0.0
    assert row["pricing_complete"] == 1


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_model_not_found_records_row_with_null_model(tmp_ledger, monkeypatch):
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {})

    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json={"model": "nope", "messages": []})
        assert resp.status_code == 404

    r = ledger.recent()[0]
    assert r["status"] == 404
    assert r["model"] is None
    assert r["usage_reported"] == 0


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_admin_usage_endpoint_requires_admin_key(monkeypatch, tmp_ledger):
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_KEY", raising=False)
    with TestClient(app) as c:
        assert c.get("/admin/api/usage").status_code == 401
        assert c.get("/admin/api/requests").status_code == 401

        monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin-key")
        response = c.get(
            "/admin/api/usage",
            headers={"Authorization": "Bearer admin-key"},
        )
        assert response.status_code == 200
        assert response.json()["cache_retention"]["min_gap_seconds"] == 300
