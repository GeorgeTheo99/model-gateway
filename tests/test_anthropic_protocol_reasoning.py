"""Native Anthropic reasoning must not depend on a provider's configured name."""

import copy

import pytest
from fastapi.testclient import TestClient

from src.providers import ProviderInfo
import src.server as server


def _native_info(**overrides):
    return ProviderInfo(**{
        "provider": "workspace-proxy",
        "base_url": "http://up/anthropic/v1",
        "api_key": "test-key",
        "provider_model_id": "databricks-claude-sonnet-4-6",
        "protocol": "anthropic",
        "thinking": "optional",
        "max_output_tokens": 4096,
        **overrides,
    })


@pytest.mark.parametrize("target", ["chat", "messages", "responses"])
@pytest.mark.parametrize("provider", ["databricks", "workspace-proxy"])
def test_native_protocol_infers_anthropic_reasoning(target, provider):
    assert server._infer_thinking_format(_native_info(provider=provider), target) == "anthropic"


@pytest.mark.parametrize("target", ["chat", "messages", "responses"])
def test_named_openai_protocol_is_not_inferred_from_claude_model_name(target):
    # Some workspaces expose Claude through OpenAI Chat/serving invocations.
    assert server._infer_thinking_format(_native_info(protocol="openai"), target) == "openai"


@pytest.mark.parametrize("explicit", ["none", "openai", "anthropic"])
def test_explicit_thinking_format_still_takes_precedence(explicit):
    info = _native_info(thinking_format=explicit)
    assert server._infer_thinking_format(info, "messages") == explicit


@pytest.mark.parametrize("protocol,expected_params", [
    ("anthropic", ["thinking"]),
    ("openai", ["reasoning_effort"]),
])
def test_model_discovery_reflects_protocol_reasoning(monkeypatch, protocol, expected_params):
    entry = {
        "id": "sonnet", "name": "sonnet", "provider": "workspace-proxy",
        "provider_model_id": "databricks-claude-sonnet-4-6",
        "protocol": protocol, "thinking": "optional", "max_output_tokens": 4096,
    }
    monkeypatch.setattr(server, "list_available_models", lambda: [entry])
    monkeypatch.setattr(server, "list_routable_models", lambda: [entry])
    client = TestClient(server.app)
    for endpoint in ["/v1/models", "/v1/debug/thinking"]:
        response = client.get(endpoint)
        assert response.status_code == 200
        data = response.json()
        row = data["data"][0] if endpoint == "/v1/models" else data["models"][0]
        assert row["thinking_format"] == protocol
        assert row["forwarded_params"] == expected_params


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("controls,expected_thinking", [
    ({}, None),
    ({"thinking": {"type": "disabled"}}, None),
    ({"thinking": {"type": "enabled", "budget_tokens": 2048}},
     {"type": "enabled", "budget_tokens": 2048}),
    ({"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
     {"type": "enabled", "budget_tokens": 1024}),
    ({"reasoning_effort": "high"},
     {"type": "enabled", "budget_tokens": 3072}),  # reserve 1024 answer tokens
], ids=["auto", "off", "budget", "adaptive", "openai-input"])
def test_messages_proxy_forwards_only_native_reasoning(monkeypatch, stream, controls, expected_thinking):
    info = _native_info()
    monkeypatch.setattr(server, "resolve", lambda model: info if model == "sonnet" else None)
    tools = [{"name": "lookup", "description": "Look up a value", "input_schema": {"type": "object"}}]
    calls = []

    async def capture(endpoint, body, headers, **kwargs):
        calls.append(copy.deepcopy(body))
        assert endpoint == "http://up/anthropic/v1/messages"
        assert kwargs["provider"] == "workspace-proxy"
        assert body["model"] == "databricks-claude-sonnet-4-6"
        assert body["stream"] is stream
        assert body["tools"] == tools
        assert body.get("thinking") == expected_thinking
        assert "reasoning_effort" not in body
        assert "reasoning" not in body
        assert "output_config" not in body
        return server.JSONResponse(content={"ok": True})

    handler = "_passthrough_anthropic_stream" if stream else "_passthrough_anthropic_sync"
    monkeypatch.setattr(server, handler, capture)
    response = TestClient(server.app).post("/v1/messages", json={
        "model": "sonnet",
        "messages": [{"role": "user", "content": "Reply OK"}],
        "max_tokens": 4096,
        "stream": stream,
        "tools": tools,
        **copy.deepcopy(controls),
    })
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(calls) == 1
