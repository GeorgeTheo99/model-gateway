"""Reasoning tiers: ``model_gateway.reasoning_tier`` placed on each model's own levels."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import src.server as server
from src import federation
from src.providers import ProviderInfo
from src.reasoning_tiers import TIER_POLICY, tier_level, tier_map

from tests.test_federation import _catalog, _config, _seed

KIMI = ("low", "high", "max")
ASTRA = ("low", "medium", "high", "xhigh")
FIREWORKS = ("minimal", "low", "medium", "high", "xhigh", "max")


@pytest.mark.parametrize("thinking,levels,default,expected", [
    ("always", KIMI, "max", {"none": "low", "light": "low", "normal": "max", "hard": "max"}),
    ("always", ASTRA, "high", {"none": "low", "light": "low", "normal": "high", "hard": "xhigh"}),
    ("always", FIREWORKS, "high", {"none": "minimal", "light": "minimal", "normal": "high", "hard": "max"}),
    ("optional", ("off", "xhigh"), "xhigh", {"none": "off", "light": "xhigh", "normal": "xhigh", "hard": "xhigh"}),
    ("optional", ("off", "low", "high"), "high", {"none": "off", "light": "low", "normal": "high", "hard": "high"}),
    ("always", ("max",), "max", {"none": "max", "light": "max", "normal": "max", "hard": "max"}),
])
def test_each_tier_lands_on_the_models_own_levels(thinking, levels, default, expected):
    assert tier_map(thinking, levels, default) == expected
    # Declaration order does not matter, and a one-shot iterable is read once.
    assert tier_level("hard", thinking, iter(reversed(levels)), default) == expected["hard"]


def test_a_model_without_thinking_has_no_tiers():
    assert tier_map("", (), None) is None
    assert tier_level("hard", "", (), None) is None


def _info(levels, *, thinking="always", fmt="openai", provider="openai", protocol="openai", **extra) -> ProviderInfo:
    return ProviderInfo(
        provider=provider, base_url="http://up", api_key="k", provider_model_id="upstream-model",
        protocol=protocol, context=0, max_output_tokens=32768, thinking=thinking,
        thinking_levels=levels, thinking_format=fmt, vision=False, **extra,
    )


@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture
def upstream(monkeypatch):
    """Route every model name to its info and capture what the chat upstream receives."""
    sent = {}
    routes = {}

    async def fake_passthrough(endpoint, body, headers, **kwargs):
        sent["body"] = body
        return server.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server, "resolve", lambda model, provider_override=None: routes.get(model))
    monkeypatch.setattr(server, "_passthrough_sync", fake_passthrough)
    return routes, sent


@pytest.mark.parametrize("tier,level", [("none", "low"), ("light", "low"), ("normal", "max"), ("hard", "max")])
def test_chat_sends_the_tiers_level_and_reports_it(client, upstream, tier, level):
    routes, sent = upstream
    routes["kimi"] = _info(KIMI, fmt="kimi", provider="moonshot")
    response = client.post("/v1/chat/completions", json={
        "model": "kimi", "messages": [{"role": "user", "content": "hi"}],
        "model_gateway": {"reasoning_tier": tier},
    })
    assert response.status_code == 200
    assert sent["body"]["reasoning_effort"] == level
    assert "model_gateway" not in sent["body"]
    assert response.headers["x-model-gateway-reasoning-tier"] == tier
    assert response.headers["x-model-gateway-reasoning-level"] == level
    assert response.headers["x-model-gateway-reasoning-policy"] == TIER_POLICY


def test_none_turns_thinking_off_where_the_model_can(client, upstream):
    routes, sent = upstream
    routes["qwen"] = _info(("off", "xhigh"), thinking="optional")
    response = client.post("/v1/chat/completions", json={
        "model": "qwen", "messages": [{"role": "user", "content": "ok"}],
        "model_gateway": {"reasoning_tier": "none", "image_handling": "reroute"},
    })
    assert response.status_code == 200
    assert response.headers["x-model-gateway-reasoning-level"] == "off"
    assert sent["body"]["reasoning_effort"] == "none"


def test_a_tier_on_a_model_without_thinking_changes_nothing(client, upstream):
    routes, sent = upstream
    routes["plain"] = _info(None, thinking="")
    response = client.post("/v1/chat/completions", json={
        "model": "plain", "messages": [{"role": "user", "content": "hi"}],
        "model_gateway": {"reasoning_tier": "hard"},
    })
    assert response.status_code == 200
    assert "reasoning_effort" not in sent["body"]
    assert response.headers["x-model-gateway-reasoning-tier"] == "hard"
    assert "x-model-gateway-reasoning-level" not in response.headers


@pytest.mark.parametrize("endpoint,extra", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/responses", {"input": "hi"}),
    ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32}),
])
@pytest.mark.parametrize("controls,message", [
    ({"model_gateway": {"reasoning_tier": "auto"}}, "Unknown model_gateway.reasoning_tier 'auto'"),
    ({"model_gateway": {"reasoning_tier": "hard"}, "reasoning_effort": "low"},
     "cannot be combined with reasoning_effort"),
    ({"model_gateway": {"reasoning_tier": "hard"}, "thinking": {"type": "enabled", "budget_tokens": 2048}},
     "cannot be combined with thinking"),
])
def test_unknown_tiers_and_tier_plus_level_are_refused(client, upstream, endpoint, extra, controls, message):
    routes, _ = upstream
    routes["kimi"] = _info(KIMI, fmt="kimi", provider="moonshot")
    response = client.post(endpoint, json={"model": "kimi", **extra, **controls})
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert message in body["error"]["message"]
    if endpoint == "/v1/messages":
        assert body["type"] == "error"


def test_a_streamed_reply_carries_the_headers(client, upstream, monkeypatch):
    routes, _ = upstream
    routes["kimi"] = _info(KIMI, fmt="kimi", provider="moonshot")
    sent = {}

    async def fake_stream(endpoint, body, headers, **kwargs):
        sent["body"] = body
        return server.StreamingResponse(iter([b"data: [DONE]\n\n"]), media_type="text/event-stream")

    monkeypatch.setattr(server, "_passthrough_stream", fake_stream)
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "kimi", "stream": True, "messages": [{"role": "user", "content": "hi"}],
        "model_gateway": {"reasoning_tier": "hard"},
    }) as response:
        assert response.read() == b"data: [DONE]\n\n"
        assert response.headers["x-model-gateway-reasoning-level"] == "max"
    assert sent["body"]["reasoning_effort"] == "max"


def test_messages_native_anthropic_gets_the_level(client, upstream, monkeypatch):
    routes, _ = upstream
    routes["claude"] = _info(("off", "low", "medium", "high", "max"), thinking="optional", fmt="anthropic",
                             provider="anthropic", protocol="anthropic")
    sent = {}

    async def fake_anthropic(endpoint, body, headers, **kwargs):
        sent["body"] = body
        return server.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server, "_passthrough_anthropic_sync", fake_anthropic)
    response = client.post("/v1/messages", json={
        "model": "claude", "max_tokens": 64000, "messages": [{"role": "user", "content": "hi"}],
        "model_gateway": {"reasoning_tier": "light"},
    })
    assert response.status_code == 200
    assert response.headers["x-model-gateway-reasoning-level"] == "low"
    assert sent["body"]["thinking"]["type"] in ("enabled", "adaptive")
    assert "model_gateway" not in sent["body"] and "reasoning_effort" not in sent["body"]


@pytest.mark.parametrize("quirk,tools,level", [
    ("reasoning_none_with_tools", True, "off"),
    ("no_reasoning_params", False, None),
    ("force_reasoning_effort_max", False, "max"),
])
def test_a_quirk_that_overrides_the_level_reports_what_was_sent(client, upstream, quirk, tools, level):
    routes, sent = upstream
    routes["quirky"] = _info(ASTRA, quirks=frozenset({quirk}))
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    response = client.post("/v1/chat/completions", json={
        "model": "quirky", "messages": [{"role": "user", "content": "hi"}],
        "model_gateway": {"reasoning_tier": "light"}, **({"tools": [tool]} if tools else {}),
    })
    assert response.status_code == 200
    assert response.headers.get("x-model-gateway-reasoning-level") == level
    assert sent["body"].get("reasoning_effort") == {"off": "none", None: None, "max": "max"}[level]


def test_responses_native_passthrough_gets_the_level(client, upstream, monkeypatch):
    routes, _ = upstream
    routes["astra"] = _info(ASTRA, fmt="openai-responses")
    sent = {}

    async def fake_responses(endpoint, body, headers, is_stream, **kwargs):
        sent["body"] = body
        return server.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server, "_handle_openai_responses_passthrough", fake_responses)
    response = client.post("/v1/responses", json={
        "model": "astra", "input": "hi", "reasoning": {"summary": "auto"},
        "model_gateway": {"reasoning_tier": "hard"},
    })
    assert response.status_code == 200
    assert sent["body"]["reasoning"] == {"summary": "auto", "effort": "xhigh"}
    assert response.headers["x-model-gateway-reasoning-level"] == "xhigh"


def test_a_vision_reroute_places_the_tier_on_the_served_model(client, upstream, monkeypatch):
    routes, sent = upstream
    routes["text-model"] = _info(("off", "low"), thinking="optional")
    routes["vision-fallback"] = ProviderInfo(**{**_info(("max",)).__dict__, "vision": True})
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "vision-fallback")
    monkeypatch.setattr(server, "pool_candidates", lambda model: [])
    response = client.post("/v1/chat/completions", json={
        "model": "text-model", "model_gateway": {"reasoning_tier": "light"},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
    })
    assert response.status_code == 200
    assert sent["body"]["reasoning_effort"] == "xhigh"  # the OpenAI format sends max as xhigh
    assert response.headers["x-model-gateway-reasoning-level"] == "max"


def test_a_tier_overrides_a_profiles_default_level():
    body = {"model_gateway": {"reasoning_tier": "hard"}}
    server._apply_profile_defaults(body, {"reasoning_effort": "off"}, "openai_chat")
    assert "reasoning_effort" not in body


def test_discovery_advertises_each_models_tiers(client, monkeypatch):
    import src.providers as providers

    entry = {"name": "kimi-x", "provider": "moonshot", "provider_model_id": "kimi-x", "thinking": "always",
             "thinking_levels": list(KIMI), "thinking_format": "kimi"}
    plain = {"name": "plain-x", "provider": "moonshot", "provider_model_id": "plain-x"}
    forced = {"name": "forced-x", "provider": "forced", "provider_model_id": "forced-x", "thinking": "always",
              "thinking_levels": list(ASTRA)}
    monkeypatch.setattr(providers, "_config", {"providers": {
        "moonshot": {"base_url": "http://up", "api_key": "k"},
        "forced": {"base_url": "http://up", "api_key": "k", "quirks": ["force_reasoning_effort_max"]},
    }})
    monkeypatch.setattr(providers, "_models", {"kimi-x": entry, "plain-x": plain, "forced-x": forced})

    rows = {row["id"]: row for row in client.get("/v1/models").json()["data"]}
    assert rows["kimi-x"]["thinking_tiers"] == {"none": "low", "light": "low", "normal": "max", "hard": "max"}
    assert rows["kimi-x"]["thinking_tier_policy"] == TIER_POLICY
    assert rows["plain-x"]["thinking_tiers"] is None and rows["plain-x"]["thinking_tier_policy"] is None
    # A provider quirk that always sends max makes every tier max.
    assert set(rows["forced-x"]["thinking_tiers"].values()) == {"max"}

    canonical = client.get("/v1/models/canonical").json()["data"]
    assert all("thinking_tiers" not in row for row in canonical)
    included = {row["id"]: row for row in client.get("/v1/models/canonical?include=thinking_tiers").json()["data"]}
    assert included["kimi-x"]["thinking_tiers"]["normal"] == "max"


def test_federation_forwards_a_concrete_level_placed_on_the_peers_metadata(tmp_path, monkeypatch, client):
    captured = {}

    async def owner(request: httpx.Request):
        captured["body"] = json.loads(await request.aread())
        return httpx.Response(200, json={"ok": True})

    manager = federation.FederationManager(_config(tmp_path), transport=httpx.MockTransport(owner))
    _seed(manager, _catalog())
    monkeypatch.setattr(federation, "_manager", manager)
    monkeypatch.setattr(server, "resolve", lambda _model, provider_override=None: None)

    response = client.post("/v1/chat/completions", json={
        "model": "edge/model-a", "messages": [{"role": "user", "content": "ok"}],
        "model_gateway": {"reasoning_tier": "none"},
    })
    assert response.status_code == 200
    assert captured["body"]["reasoning_effort"] == "off"
    assert "model_gateway" not in captured["body"]
    assert response.headers["x-model-gateway-reasoning-level"] == "off"

    imported = next(row for row in client.get("/v1/models").json()["data"] if row["id"] == "edge/model-a")
    assert imported["thinking_tiers"] == {"none": "off", "light": "high", "normal": "high", "hard": "high"}
