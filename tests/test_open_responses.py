"""Open Responses (api_style: open_responses) model routing tests.

Covers models whose upstream only accepts function tools via the workspace
Open Responses endpoint (e.g. gpt-6-astra): resolve() URL shape, pool rewire
and ordered failover on the native Responses path, reasoning controls, and
clear rejections on the chat/messages routes.
"""

import asyncio
import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import src.circuit as circuit
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
    config = copy.deepcopy(OPEN_RESPONSES_CONFIG)
    monkeypatch.setattr(providers, "_config", config)
    monkeypatch.setattr(providers, "_models", None)
    monkeypatch.setattr(circuit, "_circuits", {})
    for variable in (
        "GATEWAY_VISION_FALLBACK", "GATEWAY_VISION_FALLBACK_LOCAL",
        "GATEWAY_VISION_FALLBACK_CLOUD", "GATEWAY_VISION_FALLBACK_MODE",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(
        providers, "MODEL_INFO_PATH", providers.Path("/nonexistent-model-info.json")
    )
    yield config
    providers._config = None
    providers._models = None


# ── providers: resolve() URL shape ──────────────────────────────────────────

@pytest.mark.parametrize("suffix", ["", "/", "/serving-endpoints", "/serving-endpoints/"])
def test_resolve_builds_open_responses_url(open_responses_registry, suffix):
    open_responses_registry["providers"]["ws-e2"]["base_url"] = "https://e2.example.com" + suffix
    info = providers.resolve("astra")
    assert info is not None
    assert info.base_url == "https://e2.example.com/serving-endpoints/open-responses"
    assert info.endpoint_suffix == ""
    assert info.api_style == "open_responses"
    assert info.provider_model_id == "databricks-gpt-6-astra"


@pytest.mark.parametrize("variable,suffix", [
    ("DATABRICKS_HOST", ""),
    ("DATABRICKS_HOST", "/"),
    ("DATABRICKS_SERVING_BASE_URL", "/serving-endpoints"),
    ("DATABRICKS_SERVING_BASE_URL", "/serving-endpoints/"),
])
def test_resolve_open_responses_from_databricks_env(
    open_responses_registry, monkeypatch, variable, suffix,
):
    for name in (
        "DATABRICKS_HOST", "DATABRICKS_SERVING_BASE_URL",
        "MODEL_GATEWAY_PROVIDER_DATABRICKS_BASE_URL",
        "MODEL_GATEWAY_PROVIDER_DATABRICKS_API_KEY",
        "MODEL_GATEWAY_PROVIDER_DATABRICKS_PROTOCOL",
        "MODEL_GATEWAY_PROVIDER_DATABRICKS_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "https://workspace.example.com" + suffix)
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")
    model = open_responses_registry["models"][0]
    model.pop("pool")
    model["provider"] = "databricks"

    info = providers.resolve("astra")

    assert info is not None
    assert info.base_url == "https://workspace.example.com/serving-endpoints/open-responses"
    assert info.endpoint_suffix == ""
    assert info.api_key == "test-token"


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


# ── vision fallback dependencies ────────────────────────────────────────────

@pytest.mark.parametrize("variable", ["GATEWAY_VISION_FALLBACK", "GATEWAY_VISION_FALLBACK_CLOUD"])
@pytest.mark.parametrize("mode", ["reroute", "extract_then_answer"])
def test_vision_policy_rejects_responses_only_dependency(
    open_responses_registry, monkeypatch, variable, mode,
):
    monkeypatch.setenv(variable, "astra")
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK_MODE", mode)

    with pytest.raises(RuntimeError, match="Responses-only") as exc:
        server._validate_vision_fallback_policy(log_policy=False)

    assert variable in str(exc.value)
    assert "astra" in str(exc.value)


@pytest.mark.parametrize("mode", ["reroute", "extract_then_answer"])
@pytest.mark.parametrize("typed_message", [False, True])
def test_invalid_vision_dependency_fails_before_upstream_request(
    open_responses_registry, monkeypatch, mode, typed_message,
):
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "astra")
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK_MODE", mode)

    def no_client(**kwargs):
        pytest.fail("Invalid vision dependencies must be rejected before opening an upstream client")

    monkeypatch.setattr(server.httpx, "AsyncClient", no_client)
    result = TestClient(server.app).post("/v1/responses", json={
        "model": "plain",
        "input": [{**({"type": "message"} if typed_message else {}), "role": "user", "content": [
            {"type": "input_text", "text": "Describe this image"},
            {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
        ]}],
    })

    assert result.status_code == 502
    assert "Responses-only" in result.json()["error"]["message"]


# ── full Responses route: error envelopes and workspace failover ─────────────

@pytest.fixture
def mock_responses_upstream(open_responses_registry, monkeypatch):
    """Exercise the real route and retry wrappers without network/auth calls."""
    config = open_responses_registry
    config["providers"]["ws-west"]["base_url"] = "https://west.example.com/serving-endpoints/"
    config["providers"]["ws-dogfood"] = {
        **config["providers"]["ws-e2"],
        "base_url": "https://dogfood.example.com",
        "api_key": "key-dogfood",
    }
    config["pools"]["astra-pool"].append("ws-dogfood")
    monkeypatch.setattr(upstream, "_RETRY_MAX", 1)
    monkeypatch.setattr(upstream, "_RETRY_TRANSPORT_ATTEMPTS", 1)
    monkeypatch.setattr(upstream, "_max_attempts_for_status", lambda status: 1)

    async def no_token(*args, **kwargs):
        return None

    monkeypatch.setattr(upstream, "ensure_fresh_oauth_token", no_token)
    monkeypatch.setattr(upstream, "refresh_oauth_token", no_token)
    real_client = httpx.AsyncClient

    def install(handler):
        monkeypatch.setattr(
            server.httpx, "AsyncClient",
            lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
        )

    return install


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status,payload,expected_status,error_type,message,code", [
    pytest.param(400, {"error_code": "BAD_REQUEST", "message": "Invalid input"},
                 400, "invalid_request_error", "Invalid input", None, id="databricks-error"),
    pytest.param(422, {"error_code": "BAD_REQUEST", "message": json.dumps({"error": {"message": "Input is too long"}})},
                 400, "invalid_request_error", "context window", "context_length_exceeded", id="nested-context-overflow"),
    pytest.param(503, "Service unavailable", 502, "api_error", "Service unavailable", None, id="non-json-error"),
    pytest.param(429, {"error_code": "RESOURCE_EXHAUSTED", "message": "Rate limited"},
                 429, "rate_limit_error", "retry-after: 5", None, id="rate-limit"),
    pytest.param(403, {"error": {"type": "permission_error", "code": "denied", "message": "Access denied"}},
                 403, "permission_error", "Access denied", "denied", id="native-openai-error"),
])
def test_responses_errors_have_consistent_openai_envelopes(
    mock_responses_upstream, stream, status, payload, expected_status, error_type, message, code,
):
    seen = []

    def handler(request):
        seen.append(request.url.host)
        data = {"text": payload} if isinstance(payload, str) else {"json": payload}
        return httpx.Response(status, headers={"retry-after": "5"}, **data)

    mock_responses_upstream(handler)
    result = TestClient(server.app).post("/v1/responses", json={
        "model": "astra", "input": "Hello", "stream": stream,
    })

    assert result.status_code == expected_status
    assert set(result.json()) == {"error"}
    error = result.json()["error"]
    assert error["type"] == error_type
    assert message in error["message"]
    assert error.get("code") == code
    if status in (400, 422):
        assert seen == ["e2.example.com"]  # Invalid input must not trigger workspace failover.
    else:
        assert list(dict.fromkeys(seen)) == ["e2.example.com", "west.example.com", "dogfood.example.com"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", [404, 401, 429, 503, "transport"])
def test_responses_route_fails_over_through_third_workspace(
    mock_responses_upstream, stream, failure,
):
    seen = []
    expected_body = {
        "model": "databricks-gpt-6-astra", "input": "Weather in Paris?",
        "tools": [{"type": "function", "name": "weather", "parameters": {"type": "object", "properties": {}}}],
        "reasoning": {"effort": "max"}, "max_output_tokens": 300, "stream": stream,
    }
    response_body = {
        "id": "resp_test", "object": "response", "status": "completed", "model": "gpt-6-astra",
        "output": [{"type": "function_call", "name": "weather", "call_id": "call_test", "arguments": "{}"}],
        "usage": {"input_tokens": 30, "output_tokens": 20, "total_tokens": 50},
    }
    event = "event: response.completed\ndata: " + json.dumps({"type": "response.completed", "response": response_body}) + "\n\n"
    keys = {"e2.example.com": "key-e2", "west.example.com": "key-west", "dogfood.example.com": "key-dogfood"}

    def handler(request):
        seen.append(request)
        if request.url.host == "e2.example.com":
            if failure == "transport":
                raise httpx.ConnectError("Primary unavailable")
            return httpx.Response(failure, json={"error_code": "UNAVAILABLE", "message": "Primary unavailable"})
        if request.url.host == "west.example.com":
            return httpx.Response(404, json={"error_code": "NOT_FOUND", "message": "Secondary unavailable"})
        if stream:
            return httpx.Response(200, text=event, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=response_body)

    mock_responses_upstream(handler)
    result = TestClient(server.app).post("/v1/responses", json={**expected_body, "model": "astra"})

    assert result.status_code == 200, result.text
    assert list(dict.fromkeys(request.url.host for request in seen)) == [
        "e2.example.com", "west.example.com", "dogfood.example.com",
    ]
    # Assert outside the transport handler: retry wrappers catch transport
    # exceptions, which would otherwise hide a failed assertion on a member.
    for request in seen:
        assert request.url.path == "/serving-endpoints/open-responses"
        assert request.headers["authorization"] == "Bearer " + keys[request.url.host]
        assert json.loads(request.content) == expected_body
    if stream:
        assert result.text == event
    else:
        assert result.json() == response_body


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("typed_message", [False, True])
def test_astra_native_vision_stays_on_responses_api(
    mock_responses_upstream, stream, typed_message,
):
    seen = []
    body = {
        "model": "astra", "stream": stream, "reasoning": {"effort": "low"},
        "input": [{**({"type": "message"} if typed_message else {}), "role": "user", "content": [
            {"type": "input_text", "text": "Describe this image"},
            {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
        ]}],
    }

    def handler(request):
        seen.append(request)
        if stream:
            return httpx.Response(200, text='event: response.completed\ndata: {"type":"response.completed"}\n\n')
        return httpx.Response(200, json={"id": "resp_test", "status": "completed", "output": []})

    mock_responses_upstream(handler)
    result = TestClient(server.app).post("/v1/responses", json=body)

    assert result.status_code == 200
    assert len(seen) == 1
    assert seen[0].url.path == "/serving-endpoints/open-responses"
    assert json.loads(seen[0].content) == {**body, "model": "databricks-gpt-6-astra"}


def test_responses_stream_is_not_replayed_after_output_starts(mock_responses_upstream):
    seen = []

    class InterruptedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
            raise httpx.ReadError("Upstream stream interrupted")

    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(200, stream=InterruptedStream(), headers={"content-type": "text/event-stream"})

    mock_responses_upstream(handler)
    result = TestClient(server.app).post("/v1/responses", json={
        "model": "astra", "input": "Hello", "stream": True,
    })

    assert result.status_code == 200
    assert "Hello" in result.text
    assert "Provider stream failed" in result.text
    assert seen == ["e2.example.com"]
