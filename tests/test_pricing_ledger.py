"""Tests for pricing lookup and the ledger middleware end-to-end."""

from types import SimpleNamespace

import pytest

from src.providers import ProviderInfo, pricing_for
import src.server as server_module
from src.server import app
from src import ledger

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None


def test_pricing_for_known_model():
    p = pricing_for("claude-sonnet-4.6")
    assert p is not None
    assert p["input"] == 3.0
    assert p["output"] == 15.0
    assert p["cache_read"] == 0.3
    assert p["cache_write"] == 3.75


def test_pricing_for_unpriced_model_is_none():
    # gpt-5.4 has no pricing field -> unknown cost
    assert pricing_for("gpt-5.4") is None


def test_pricing_for_unknown_model_is_none():
    assert pricing_for("does-not-exist-xyz") is None


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
    monkeypatch.setenv("CLOUD_GATEWAY_LEDGER_PATH", str(tmp_path / "ledger.db"))
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

    async def fake_passthrough_sync(endpoint, body, headers):
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


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_unpriced_model_records_unknown_cost(tmp_ledger, monkeypatch):
    info = _info(provider="openai", provider_model_id="gpt-5.4")
    import src.providers as providers
    monkeypatch.setattr(providers, "_models", {
        "unpriced": {"name": "unpriced", "provider": "openai", "provider_model_id": "gpt-5.4"},
    })

    def fake_resolve(model):
        return info if model == "unpriced" else None

    async def fake_passthrough_sync(endpoint, body, headers):
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
    monkeypatch.delenv("CLOUD_GATEWAY_ADMIN_KEY", raising=False)
    with TestClient(app) as c:
        assert c.get("/admin/api/usage").status_code == 401
        assert c.get("/admin/api/requests").status_code == 401
