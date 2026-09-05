"""Open Responses (api_style: open_responses) model routing tests.

Covers models whose upstream only accepts function tools via the workspace
Open Responses endpoint (e.g. gpt-6-astra): resolve() URL shape, pool rewire
and ordered failover on the native Responses path, reasoning controls, and
clear rejections on the chat/messages routes.
"""

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

import src.providers as providers
import src.server as server
import src.upstream as upstream
from src.server import _apply_gateway_reasoning
from src.upstream import PoolContext


OPEN_RESPONSES_CONFIG = {
    "providers": {
        "ws-e2": {
            "base_url": "https://e2.example.com",
            "api_key": "key-e2",
            "protocol": "openai",
            "endpoint_style": "invocations",
            "quirks": ["no_stream_options", "no_reasoning_params"],
        },
        "ws-west": {
            "base_url": "https://west.example.com",
            "api_key": "key-west",
            "protocol": "openai",
            "endpoint_style": "invocations",
            "quirks": ["no_stream_options", "no_reasoning_params"],
        },
    },
    "pools": {
        "astra-pool": ["ws-e2", "ws-west"],
    },
    "models": [
        {
            "name": "gpt-6-astra",
            "alias": "astra",
            "pool": "astra-pool",
            "provider_model_id": "databricks-gpt-6-astra",
            "api_style": "open_responses",
            "context": 1050000,
            "max_output_tokens": 128000,
            "thinking": "always",
            "thinking_levels": ["minimal", "low", "medium", "high", "xhigh", "max"],
            "vision": True,
        },
        {
            "name": "plain-chat-model",
            "alias": "plain",
            "provider": "ws-e2",
            "provider_model_id": "databricks-plain-chat-model",
            "context": 1000,
            "max_output_tokens": 100,
        },
    ],
}


@pytest.fixture
def open_responses_registry(monkeypatch):
    monkeypatch.setattr(providers, "_config", OPEN_RESPONSES_CONFIG)
    monkeypatch.setattr(providers, "_models", None)
    monkeypatch.setattr(
        providers, "MODEL_INFO_PATH", providers.Path("/nonexistent-model-info.json")
    )
    yield
    providers._config = None
    providers._models = None


# ── providers: resolve() URL shape ──────────────────────────────────────────

def test_resolve_builds_open_responses_url(open_responses_registry):
    info = providers.resolve("astra")
    assert info is not None
    assert info.base_url == "https://e2.example.com/serving-endpoints/open-responses"
    assert info.endpoint_suffix == ""
    assert info.api_style == "open_responses"
    assert info.provider_model_id == "databricks-gpt-6-astra"


def test_pool_rewire_targets_open_responses_on_every_member(open_responses_registry):
    assert providers.pool_candidates("astra") == ["ws-e2", "ws-west"]
    west_info = providers.resolve("astra", provider_override="ws-west")
    assert west_info is not None
    assert west_info.base_url == "https://west.example.com/serving-endpoints/open-responses"
    assert west_info.endpoint_suffix == ""


def test_pool_failover_on_404_targets_member_open_responses_url(
    open_responses_registry, monkeypatch
):
    """A dead primary fails over to the next member's open-responses URL."""
    monkeypatch.setattr(upstream, "_RETRY_MAX", 1)
    monkeypatch.setattr(upstream, "_RETRY_BASE_DELAY", 0)
    monkeypatch.setattr(upstream, "_RETRY_MAX_DELAY", 0)
    monkeypatch.setattr(upstream, "_RETRY_TRANSPORT_ATTEMPTS", 1)
    monkeypatch.setattr(upstream, "_RETRY_TRANSPORT_MAX_DELAY", 0)
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        if req.url.host == "e2.example.com":
            return httpx.Response(404, text='{"message": "endpoint not found"}')
        return httpx.Response(200, json={"ok": True, "host": req.url.host})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await upstream._retry_post_with_model_fallback(
                client, "https://e2.example.com/serving-endpoints/open-responses",
                json={"model": "databricks-gpt-6-astra"},
                headers={"Authorization": "Bearer key-e2"},
                provider="ws-e2",
                pool=PoolContext(
                    model_key="astra",
                    provider="ws-e2",
                    base_url="https://e2.example.com/serving-endpoints/open-responses",
                    api_key="key-e2",
                ),
            )

    resp = asyncio.run(run())
    assert resp.status_code == 200
    assert seen == [
        "https://e2.example.com/serving-endpoints/open-responses",
        "https://west.example.com/serving-endpoints/open-responses",
    ]


def test_plain_chat_model_unaffected(open_responses_registry):
    info = providers.resolve("plain")
    assert info is not None
    assert info.api_style == ""
    assert (
        info.base_url
        == "https://e2.example.com/serving-endpoints/databricks-plain-chat-model/invocations"
    )


# ── server: reasoning controls on the Responses path ─────────────────────────

def test_reasoning_survives_no_reasoning_params_on_responses_target(open_responses_registry):
    info = providers.resolve("astra")
    req = {"reasoning": {"effort": "high"}}
    enabled = _apply_gateway_reasoning(req, info, target_api="responses")
    assert enabled is True
    assert req["reasoning"] == {"effort": "high"}


def test_reasoning_efforts_map_verbatim_on_responses_target(open_responses_registry):
    info = providers.resolve("astra")
    req = {"reasoning": {"effort": "max"}}
    _apply_gateway_reasoning(req, info, target_api="responses")
    assert req["reasoning"] == {"effort": "max"}

    req = {"reasoning": {"effort": "minimal"}}
    _apply_gateway_reasoning(req, info, target_api="responses")
    assert req["reasoning"] == {"effort": "low"}


# ── server: chat/messages routes reject responses-only models ────────────────

def test_chat_route_rejects_responses_only_model(open_responses_registry, monkeypatch):
    monkeypatch.setattr(server, "resolve", providers.resolve)
    client = TestClient(server.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "astra", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 400
    assert "/v1/responses" in resp.json()["error"]["message"]


def test_messages_route_rejects_responses_only_model(open_responses_registry):
    client = TestClient(server.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "astra",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        },
    )
    assert resp.status_code == 400
    body = resp.json()["error"]["message"]
    assert "/v1/responses" in (body["text"] if isinstance(body, dict) else body)
