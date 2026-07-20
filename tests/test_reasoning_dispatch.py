"""Regression tests for the gateway reasoning/thinking dispatch.

These lock in the exact upstream params each `thinking_format` forwards, so a
silent divergence (e.g. the `zai` branch dropping `reasoning_effort`) becomes a
visible failure instead of a quiet behavior change.

Run:  cd model-gateway && uv run pytest
"""

import asyncio
import base64
import io
import logging
from types import SimpleNamespace

import pytest

import src.providers as providers
from src.providers import ProviderInfo, list_models
import src.server as server_module
from src.server import _apply_gateway_reasoning, _infer_thinking_format, app
from src.translator import anthropic_to_openai_chat, openai_chat_to_anthropic

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - fastapi is a runtime dep
    TestClient = None


# ── helpers ──────────────────────────────────────────────────────────────────

def _info(fmt: str, *, thinking: str = "always", provider: str = "x",
          provider_model_id: str = "m", max_output_tokens: int = 32768,
          vision: bool = False) -> ProviderInfo:
    """Build a ProviderInfo with an explicit thinking_format."""
    return ProviderInfo(
        provider=provider,
        base_url="http://up",
        api_key="k",
        provider_model_id=provider_model_id,
        protocol="openai",
        context=0,
        max_output_tokens=max_output_tokens,
        thinking=thinking,
        thinking_format=fmt,
        vision=vision,
    )


# Thinking-related keys the dispatch may write onto the upstream request.
_THINK_KEYS = {
    "enable_thinking", "reasoning", "reasoning_effort", "thinking",
    "chat_template_kwargs", "thinking_budget",
}


def _think_view(req: dict) -> dict:
    """Return only the thinking-related keys of req (sorted for stable compare)."""
    return {k: req[k] for k in _THINK_KEYS if k in req}


def _run(fmt: str, req: dict, *, thinking: str = "always",
         target_api: str = "chat", **info_kw) -> tuple[bool, dict]:
    info = _info(fmt, thinking=thinking, **info_kw)
    enabled = _apply_gateway_reasoning(req, info, target_api=target_api)
    return enabled, _think_view(req)


# ── per-format expectations ──────────────────────────────────────────────────
# Each case: (format, target_api, input_req_fragment, expected_enabled, expected_view)
# Covers three scenarios across the matrix:
#   A. always-thinking model, no client control  -> auto-enable, effort defaults "high"
#   B. always-thinking model, client asks max    -> effort normalizes to "xhigh"
#   C. always-thinking model, client disables    -> enabled False, format-specific shape

ANTHROPIC_BUDGET_HIGH = int(32768 * 0.80)   # 26214
ANTHROPIC_BUDGET_XHIGH = int(32768 * 0.95)  # 31129


CASES = [
    # ── A: no client control, always-thinking ──────────────────────────────
    pytest.param("zai", "chat", {}, True,
                 {"enable_thinking": True, "reasoning_effort": "high"}, id="zai-always-default"),
    pytest.param("openai-responses", "responses", {}, True,
                 {"reasoning": {"effort": "high"}}, id="responses-always-default"),
    pytest.param("openai", "chat", {}, True,
                 {"reasoning_effort": "high"}, id="openai-always-default"),
    pytest.param("anthropic", "chat", {}, True,
                 {"thinking": {"type": "enabled", "budget_tokens": ANTHROPIC_BUDGET_HIGH}},
                 id="anthropic-always-default"),
    pytest.param("openrouter", "chat", {}, True,
                 {"reasoning": {"effort": "high"}}, id="openrouter-always-default"),
    pytest.param("qwen-chat-template", "chat", {}, True,
                 {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}},
                 id="qwen-ctk-always-default"),
    pytest.param("glm-chat-template", "chat", {}, True,
                 {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "high"}},
                 id="glm-ctk-always-default"),
    pytest.param("deepseek-v4-dsml", "chat", {}, True,
                 {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}},
                 id="deepseek-v4-dsml-always-default"),
    pytest.param("qwen", "chat", {}, True,
                 {"enable_thinking": True}, id="qwen-always-default"),
    pytest.param("deepseek", "chat", {}, True,
                 {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
                 id="deepseek-always-default"),
    pytest.param("none", "chat", {}, True, {}, id="none-always-default"),

    # ── B: client requests max (normalizes to xhigh internally) ────────────
    pytest.param("zai", "chat", {"reasoning_effort": "max"}, True,
                 {"enable_thinking": True, "reasoning_effort": "max"}, id="zai-max"),
    pytest.param("openai-responses", "responses", {"reasoning_effort": "max"}, True,
                 {"reasoning": {"effort": "xhigh"}}, id="responses-max"),
    pytest.param("openai", "chat", {"reasoning_effort": "max"}, True,
                 {"reasoning_effort": "xhigh"}, id="openai-max"),
    pytest.param("anthropic", "chat", {"reasoning_effort": "max"}, True,
                 {"thinking": {"type": "enabled", "budget_tokens": ANTHROPIC_BUDGET_XHIGH}},
                 id="anthropic-max"),
    pytest.param("openrouter", "chat", {"reasoning_effort": "max"}, True,
                 {"reasoning": {"effort": "xhigh"}}, id="openrouter-max"),
    pytest.param("qwen-chat-template", "chat", {"reasoning_effort": "max"}, True,
                 {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}},
                 id="qwen-ctk-max"),
    pytest.param("glm-chat-template", "chat", {"reasoning_effort": "max"}, True,
                 {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "max"}},
                 id="glm-ctk-max"),
    pytest.param("deepseek-v4-dsml", "chat", {"reasoning_effort": "max"}, True,
                 {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}},
                 id="deepseek-v4-dsml-max"),
    pytest.param("qwen", "chat", {"reasoning_effort": "max"}, True,
                 {"enable_thinking": True}, id="qwen-max"),
    pytest.param("deepseek", "chat", {"reasoning_effort": "max"}, True,
                 {"thinking": {"type": "enabled"}, "reasoning_effort": "xhigh"},
                 id="deepseek-max"),
    pytest.param("none", "chat", {"reasoning_effort": "max"}, True, {}, id="none-max"),

    # ── C: client disables thinking on an always model ────────────────────
    pytest.param("zai", "chat", {"reasoning_effort": "none"}, False,
                 {"enable_thinking": False}, id="zai-disabled"),
    pytest.param("openai-responses", "responses", {"reasoning_effort": "none"}, False,
                 {"reasoning": {"effort": "none"}}, id="responses-disabled"),
    pytest.param("openai", "chat", {"reasoning_effort": "none"}, False,
                 {"reasoning_effort": "none"}, id="openai-disabled"),
    pytest.param("anthropic", "chat", {"reasoning_effort": "none"}, False,
                 {}, id="anthropic-disabled"),
    pytest.param("openrouter", "chat", {"reasoning_effort": "none"}, False,
                 {"reasoning": {"effort": "none"}}, id="openrouter-disabled"),
    pytest.param("qwen-chat-template", "chat", {"reasoning_effort": "none"}, False,
                 {"chat_template_kwargs": {"enable_thinking": False}}, id="qwen-ctk-disabled"),
    pytest.param("glm-chat-template", "chat", {"reasoning_effort": "none"}, False,
                 {"chat_template_kwargs": {"enable_thinking": False}}, id="glm-ctk-disabled"),
    pytest.param("deepseek-v4-dsml", "chat", {"reasoning_effort": "none"}, False,
                 {"chat_template_kwargs": {"enable_thinking": False}}, id="deepseek-v4-dsml-disabled"),
    pytest.param("qwen", "chat", {"reasoning_effort": "none"}, False,
                 {"enable_thinking": False}, id="qwen-disabled"),
    pytest.param("deepseek", "chat", {"reasoning_effort": "none"}, False,
                 {"thinking": {"type": "disabled"}}, id="deepseek-disabled"),
    pytest.param("none", "chat", {"reasoning_effort": "none"}, False, {}, id="none-disabled"),
]


@pytest.mark.parametrize("fmt,target_api,fragment,exp_enabled,exp_view", CASES)
def test_dispatch_per_format(fmt, target_api, fragment, exp_enabled, exp_view):
    req = {"messages": [], **fragment}
    enabled, view = _run(fmt, req, target_api=target_api)
    assert enabled is exp_enabled
    assert view == exp_view


# ── contract: optional models get no params unless the client asks ───────────

def test_optional_model_no_control_is_noop():
    """An optional-thinking model with no client reasoning control must not be
    auto-enabled and must leave the request untouched."""
    enabled, view = _run("zai", {"messages": []}, thinking="optional")
    assert enabled is False
    assert view == {}


def test_optional_model_client_enables_is_forwarded():
    enabled, view = _run("zai", {"messages": [], "reasoning_effort": "high"},
                         thinking="optional")
    assert enabled is True
    assert view == {"enable_thinking": True, "reasoning_effort": "high"}


def test_no_thinking_no_format_strips_and_returns_false():
    """A model with neither thinking nor thinking_format strips controls and
    returns False even if the client sent reasoning params."""
    info = ProviderInfo(provider="x", base_url="http://up", api_key="k",
                        provider_model_id="m", thinking="", thinking_format="")
    req = {"messages": [], "reasoning_effort": "high", "thinking": {"type": "enabled"}}
    enabled = _apply_gateway_reasoning(req, info, target_api="chat")
    assert enabled is False
    assert _think_view(req) == {}


def test_client_reasoning_dict_effort_is_normalized():
    """reasoning.effort is read and normalized the same as reasoning_effort."""
    enabled, view = _run("openai", {"messages": [], "reasoning": {"effort": "max"}},
                         target_api="chat")
    assert enabled is True
    assert view == {"reasoning_effort": "xhigh"}


def test_glm_chat_template_reads_nested_effort_from_pi():
    enabled, view = _run("glm-chat-template", {
        "messages": [],
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "max",
        },
    }, thinking="optional")
    assert enabled is True
    assert view == {
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "max",
        }
    }


def test_disabled_glm_chat_template_strips_stale_nested_effort():
    enabled, view = _run("glm-chat-template", {
        "messages": [],
        "reasoning_effort": "none",
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "max",
            "unrelated": "kept",
        },
    })
    assert enabled is False
    assert view == {"chat_template_kwargs": {"unrelated": "kept", "enable_thinking": False}}


def test_budget_overrides_effort_ratio_for_anthropic():
    """An explicit reasoning.max_tokens budget is forwarded verbatim (>=1024)."""
    enabled, view = _run("anthropic",
                         {"messages": [], "reasoning": {"effort": "high", "max_tokens": 8000}})
    assert enabled is True
    assert view == {"thinking": {"type": "enabled", "budget_tokens": 8000}}


def test_qwen_forwards_budget_when_provided():
    enabled, view = _run("qwen",
                         {"messages": [], "reasoning": {"effort": "high", "max_tokens": 4096}})
    assert enabled is True
    assert view == {"enable_thinking": True, "thinking_budget": 4096}


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_chat_completions_text_request_does_not_500_on_vision_fallback_env(client, monkeypatch):
    """Regression: the vision-fallback env lookup must not NameError for normal text requests."""
    info = _info("none", thinking="", provider="test", provider_model_id="upstream-model")

    def fake_resolve(model):
        return info if model == "text-model" else None

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        assert endpoint == "http://up/chat/completions"
        assert body["model"] == "upstream-model"
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, "resolve", fake_resolve)
    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)

    resp = client.post("/v1/chat/completions", json={
        "model": "text-model",
        "messages": [{"role": "user", "content": "hello"}],
    })

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_chat_completions_applies_declarative_provider_onboarding_quirks(client, monkeypatch):
    info = _info("openai", provider="new-provider", provider_model_id="new-upstream", vision=True)
    info.quirks = frozenset({
        "force_reasoning_effort_max",
        "use_max_completion_tokens",
        "drop_fixed_sampling_fields",
        "inline_image_urls_only",
    })

    monkeypatch.setattr(server_module, "resolve", lambda model: info if model == "new-model" else None)

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        assert endpoint == "http://up/chat/completions"
        assert body["model"] == "new-upstream"
        assert body["reasoning_effort"] == "max"
        assert body["max_completion_tokens"] == 123
        assert body["tools"][0]["function"]["name"] == "lookup"
        for field in ("max_tokens", "temperature", "top_p", "n", "presence_penalty", "frequency_penalty"):
            assert field not in body
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)
    response = client.post("/v1/chat/completions", json={
        "model": "new-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 123,
        "temperature": 0.2,
        "top_p": 0.8,
        "n": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    })
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    public_image = client.post("/v1/chat/completions", json={
        "model": "new-model",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        ]}],
    })
    assert public_image.status_code == 400
    assert "inline data:image" in public_image.json()["error"]["message"]


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_chat_completions_local_omlx_proxies_to_upstream_model(client, monkeypatch):
    info = _info(
        "glm-chat-template",
        provider="omlx",
        provider_model_id="local-upstream",
    )

    def fake_resolve(model):
        return info if model == "local-alias" else None

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        assert endpoint == "http://up/chat/completions"
        assert headers["Authorization"] == "Bearer k"
        assert body["model"] == "local-upstream"
        assert body["chat_template_kwargs"] == {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "max",
        }
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, "resolve", fake_resolve)
    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)

    resp = client.post("/v1/chat/completions", json={
        "model": "local-alias",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "max",
    })

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_chat_completions_anthropic_model_uses_native_handler(client, monkeypatch):
    info = _info("anthropic", thinking="optional", provider="anthropic", provider_model_id="claude-upstream")
    info.protocol = "anthropic"

    def fake_resolve(model):
        return info if model == "claude-alias" else None

    async def fake_handle_chat_anthropic(body, resolved_info, model, request, is_stream):
        assert resolved_info is info
        assert model == "claude-alias"
        assert is_stream is False
        assert body["model"] == "claude-upstream"
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, "resolve", fake_resolve)
    monkeypatch.setattr(server_module, "_handle_chat_anthropic", fake_handle_chat_anthropic)

    resp = client.post("/v1/chat/completions", json={
        "model": "claude-alias",
        "messages": [{"role": "user", "content": "hello"}],
    })

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_adaptive_anthropic_chat_uses_new_thinking_shape():
    info = _info(
        "anthropic",
        provider="anthropic",
        provider_model_id="claude-fable-5",
        max_output_tokens=2048,
    )
    body = openai_chat_to_anthropic({
        "model": "anthropic/claude-fable-5",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 2048,
    })

    assert _apply_gateway_reasoning(body, info, target_api="messages") is True
    server_module._normalize_anthropic_adaptive_thinking(body, info)

    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}


def test_openai_chat_anthropic_translation_roundtrip_shapes():
    req = openai_chat_to_anthropic({
        "model": "claude-alias",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "ping"},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup", "description": "Lookup", "parameters": {"type": "object"}}}],
        "tool_choice": "required",
    })
    assert req["system"] == "be brief"
    assert req["messages"] == [{"role": "user", "content": "ping"}]
    assert req["tools"][0]["name"] == "lookup"
    assert req["tool_choice"] == {"type": "any"}
    assert req["max_tokens"] == 8192

    resp = anthropic_to_openai_chat({
        "content": [{"type": "text", "text": "pong"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }, "claude-alias")
    assert resp["model"] == "claude-alias"
    assert resp["choices"][0]["message"]["content"] == "pong"
    assert resp["choices"][0]["finish_reason"] == "stop"
    assert resp["usage"]["total_tokens"] == 5

    reasoning_req = openai_chat_to_anthropic({
        "messages": [{"role": "user", "content": "think"}],
        "reasoning_effort": "high",
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "tool_choice": "none",
    })
    assert reasoning_req["reasoning_effort"] == "high"
    assert "tools" not in reasoning_req
    enabled = _apply_gateway_reasoning(reasoning_req, _info("anthropic", thinking="optional"), target_api="messages")
    assert enabled is True
    assert reasoning_req["thinking"]["type"] == "enabled"


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_fireworks_inline_image_compression(monkeypatch):
    from PIL import Image

    raw = io.BytesIO()
    Image.new("RGB", (2200, 1200), (180, 40, 40)).save(raw, format="PNG")
    body = {"messages": [{"role": "user", "content": [{
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64.b64encode(raw.getvalue()).decode('ascii')}"},
    }]}]}
    info = _info("", provider="fireworks", vision=True)

    monkeypatch.setenv("GATEWAY_FIREWORKS_IMAGE_MAX_DIMENSION", "400")
    monkeypatch.setenv("GATEWAY_FIREWORKS_IMAGE_MAX_BYTES", "50000")
    server_module._compress_fireworks_inline_images(body, info)

    url = body["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    optimized = base64.b64decode(url.split(",", 1)[1])
    with Image.open(io.BytesIO(optimized)) as image:
        assert max(image.size) <= 400
    assert len(optimized) <= 50000


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_chat_completions_image_request_extracts_then_answers_with_opt_in_fallback(client, monkeypatch):
    text_info = _info("none", thinking="", provider="test", provider_model_id="text-upstream")
    fallback_info = _info("", thinking="optional", provider="test", provider_model_id="vision-upstream", vision=True)

    def fake_resolve(model):
        if model == "text-model":
            return text_info
        if model == "vision-fallback":
            return fallback_info
        return None

    async def fake_extract(request, body, fallback_model, fallback, error_factory, **kwargs):
        assert fallback_model == "vision-fallback"
        assert fallback is fallback_info
        assert server_module._payload_has_image(body)
        assert kwargs == {"max_images": 1, "require_inline_images": False}
        return ["Visible: a concrete wall with a crack."]

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        assert endpoint == "http://up/chat/completions"
        assert body["model"] == "text-upstream"
        assert not server_module._payload_has_image(body)
        joined = "\n".join(str(message.get("content", "")) for message in body["messages"])
        assert "Image observations from vision model" in joined
        assert "Visible: a concrete wall with a crack." in joined
        assert "gateway_image_handling" not in body
        assert "model_gateway" not in body
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "vision-fallback")
    monkeypatch.setattr(server_module, "resolve", fake_resolve)
    monkeypatch.setattr(server_module, "_extract_image_observations", fake_extract)
    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)

    resp = client.post("/v1/chat/completions", json={
        "model": "text-model",
        "gateway_image_handling": "extract_then_answer",
        "model_gateway": {"image_handling": "extract_then_answer"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what should I do?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
    })

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_chat_completions_composite_defaults_to_scoped_local_extraction(client, monkeypatch):
    text_info = _info("glm-chat-template", provider="omlx", provider_model_id="glm-upstream")
    text_info.composite = providers.CompositeRoute(
        text_model="glm-local",
        vision_model="gemma-local",
        image_handling="extract_then_answer",
        max_images=4,
    )
    vision_info = _info("", thinking="", provider="omlx", provider_model_id="gemma-upstream", vision=True)
    resolved = []

    def fake_resolve(model):
        resolved.append(model)
        if model == "best-local":
            return text_info
        if model == "gemma-local":
            return vision_info
        if model == "cloud-trap":
            raise AssertionError("composite must not use the global cloud fallback")
        return None

    async def fake_extract(request, body, fallback_model, fallback, error_factory, **kwargs):
        assert fallback_model == "gemma-local"
        assert fallback is vision_info
        assert kwargs == {"max_images": 4, "require_inline_images": True}
        return ["Dense Gemma sees a terminal window."]

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        assert body["model"] == "glm-upstream"
        assert not server_module._payload_has_image(body)
        assert "Dense Gemma sees a terminal window." in str(body["messages"])
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "cloud-trap")
    monkeypatch.setattr(server_module, "resolve", fake_resolve)
    monkeypatch.setattr(server_module, "_extract_image_observations", fake_extract)
    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)

    resp = client.post("/v1/chat/completions", json={
        "model": "best-local",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is shown?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
    })

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert resolved == ["best-local", "gemma-local"]


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses", "/v1/messages"])
def test_composite_staging_is_consistent_across_api_translations(monkeypatch, endpoint):
    from src.responses import responses_to_chat
    from src.translator import anthropic_to_openai

    text_info = _info("glm-chat-template", provider="omlx", provider_model_id="glm-upstream")
    text_info.composite = providers.CompositeRoute(
        text_model="glm-local",
        vision_model="gemma-local",
        image_handling="extract_then_answer",
        max_images=4,
    )
    vision_info = _info("", thinking="", provider="omlx", provider_model_id="gemma-upstream", vision=True)
    image_url = "data:image/png;base64,AAAA"
    if endpoint == "/v1/responses":
        chat_body = responses_to_chat({"model": "best-local", "input": [{
            "type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "inspect"},
                {"type": "input_image", "image_url": image_url},
            ],
        }]})
    elif endpoint == "/v1/messages":
        chat_body = anthropic_to_openai({"model": "best-local", "messages": [{
            "role": "user", "content": [
                {"type": "text", "text": "inspect"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            ],
        }]})
    else:
        chat_body = {"model": "best-local", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "inspect"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}]}

    async def fake_extract(request, body, fallback_model, fallback, error_factory, **kwargs):
        assert fallback_model == "gemma-local"
        assert fallback is vision_info
        return ["visible terminal"]

    monkeypatch.setattr(server_module, "resolve", lambda model: vision_info if model == "gemma-local" else None)
    monkeypatch.setattr(server_module, "_extract_image_observations", fake_extract)
    request = SimpleNamespace(headers={}, state=SimpleNamespace())

    rewritten, served_model, served_info, error = asyncio.run(
        server_module._apply_chat_vision_fallback(
            request, chat_body, "best-local", text_info, endpoint,
            server_module._error if endpoint == "/v1/messages" else server_module._error_openai,
        )
    )

    assert error is None
    assert served_model == "best-local"
    assert served_info is text_info
    assert not server_module._payload_has_image(rewritten)
    assert "visible terminal" in str(rewritten["messages"])


@pytest.mark.parametrize(
    ("headers", "controls"),
    [
        ({"x-gateway-image-handling": "reroute"}, {}),
        ({}, {"gateway_image_handling": "reroute"}),
        ({}, {"model_gateway": {"image_handling": "reroute"}}),
    ],
)
def test_composite_rejects_client_image_handling_override(headers, controls):
    text_info = _info("glm-chat-template", provider="omlx", provider_model_id="glm-upstream")
    text_info.composite = providers.CompositeRoute(
        text_model="glm-local",
        vision_model="gemma-local",
        image_handling="extract_then_answer",
        max_images=4,
    )
    body = {
        "model": "best-local",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
        **controls,
    }
    request = SimpleNamespace(headers=headers, state=SimpleNamespace())

    rewritten, served_model, served_info, error = asyncio.run(
        server_module._apply_chat_vision_fallback(
            request, body, "best-local", text_info, "/v1/chat/completions",
        )
    )

    assert error.status_code == 400
    assert b"client overrides are not allowed" in error.body
    assert served_model == "best-local"
    assert served_info is text_info
    assert "gateway_image_handling" not in rewritten
    assert "model_gateway" not in rewritten


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses", "/v1/messages"])
@pytest.mark.parametrize("control", ["header", "body", "nested"])
def test_composite_override_is_rejected_across_api_routes(
    client, monkeypatch, endpoint, control,
):
    text_info = _info("glm-chat-template", provider="omlx", provider_model_id="glm-upstream")
    text_info.composite = providers.CompositeRoute(
        text_model="glm-local",
        vision_model="gemma-local",
        image_handling="extract_then_answer",
        max_images=4,
    )
    monkeypatch.setattr(server_module, "resolve", lambda model: text_info)

    headers = {}
    if endpoint == "/v1/responses":
        payload = {"model": "best-local", "input": [{
            "type": "message", "role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            ],
        }]}
    elif endpoint == "/v1/messages":
        payload = {"model": "best-local", "max_tokens": 32, "messages": [{
            "role": "user", "content": [{
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
            }],
        }]}
    else:
        payload = {"model": "best-local", "messages": [{
            "role": "user", "content": [{
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"},
            }],
        }]}

    if control == "header":
        headers["x-gateway-image-handling"] = "reroute"
    elif control == "body":
        payload["gateway_image_handling"] = "reroute"
    else:
        payload["model_gateway"] = {"image_handling": "reroute"}

    response = client.post(endpoint, headers=headers, json=payload)

    assert response.status_code == 400
    assert "client overrides are not allowed" in response.json()["error"]["message"]


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
@pytest.mark.parametrize("control", ["body", "nested", "nested-extra", "nondict"])
@pytest.mark.parametrize("with_image", [False, True], ids=["text", "native-vision"])
def test_native_anthropic_body_controls_do_not_trigger_staging_or_forward(
    client, monkeypatch, control, with_image,
):
    native = _info(
        "anthropic", provider="anthropic", provider_model_id="claude-native",
        vision=with_image,
    )
    native.protocol = "anthropic"
    monkeypatch.setattr(server_module, "resolve", lambda model: native)

    async def fake_passthrough(endpoint, body, headers, **kwargs):
        assert "gateway_image_handling" not in body
        assert "model_gateway" not in body
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, "_passthrough_anthropic_sync", fake_passthrough)
    content = [{"type": "text", "text": "hello"}]
    if with_image:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
        })
    payload = {
        "model": "claude-native",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": content}],
    }
    if control == "body":
        payload["gateway_image_handling"] = "reroute"
    elif control == "nested":
        payload["model_gateway"] = {"image_handling": "reroute"}
    elif control == "nested-extra":
        payload["model_gateway"] = {"image_handling": "reroute", "trace": "private"}
    else:
        payload["model_gateway"] = "invalid"

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
@pytest.mark.parametrize("control", ["body", "nested", "nested-extra", "nondict"])
def test_native_openai_responses_body_controls_are_not_forwarded(
    client, monkeypatch, control,
):
    native = _info(
        "openai-responses", provider="openai", provider_model_id="gpt-native",
        vision=True,
    )
    monkeypatch.setattr(server_module, "resolve", lambda model: native)

    async def fake_passthrough(endpoint, body, headers, is_stream, **kwargs):
        assert "gateway_image_handling" not in body
        assert "model_gateway" not in body
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(
        server_module, "_handle_openai_responses_passthrough", fake_passthrough,
    )
    payload = {
        "model": "gpt-native",
        "input": [{
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }],
    }
    if control == "body":
        payload["gateway_image_handling"] = "reroute"
    elif control == "nested":
        payload["model_gateway"] = {"image_handling": "reroute"}
    elif control == "nested-extra":
        payload["model_gateway"] = {"image_handling": "reroute", "trace": "private"}
    else:
        payload["model_gateway"] = "invalid"

    response = client.post("/v1/responses", json=payload)

    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_chat_completions_image_request_uses_explicit_cloud_vision_fallback(client, monkeypatch):
    text_info = _info("none", thinking="", provider="test", provider_model_id="text-upstream")
    fallback_info = _info(
        "",
        thinking="optional",
        provider="fireworks",
        provider_model_id="accounts/fireworks/models/qwen3p7-plus",
        vision=True,
    )

    def fake_resolve(model):
        if model == "text-model":
            return text_info
        if model == "cloud-vision":
            return fallback_info
        return None

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        assert endpoint == "http://up/chat/completions"
        assert body["model"] == "accounts/fireworks/models/qwen3p7-plus"
        assert "prompt_cache_key" not in body
        assert "prompt_cache_retention" not in body
        assert "reasoning" not in body["messages"][0]
        assert "reasoning_content" not in body["messages"][0]
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "cloud-vision")
    monkeypatch.setattr(server_module, "resolve", fake_resolve)
    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)

    resp = client.post("/v1/chat/completions", json={
        "model": "text-model",
        "prompt_cache_key": "pi-session",
        "prompt_cache_retention": "24h",
        "messages": [
            {
                "role": "assistant",
                "content": "prior answer",
                "reasoning": "prior hidden reasoning",
                "reasoning_content": "prior hidden reasoning",
            },
            {"role": "user", "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]},
        ],
    })

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_lifespan_validates_vision_policy_before_initialization(monkeypatch):
    def reject_policy():
        raise RuntimeError("invalid vision policy")

    monkeypatch.setattr(server_module, "_validate_vision_fallback_policy", reject_policy)

    async def enter_lifespan():
        async with server_module._lifespan(server_module.app):
            pytest.fail("invalid policy must prevent startup")

    with pytest.raises(RuntimeError, match="invalid vision policy"):
        asyncio.run(enter_lifespan())


def test_startup_vision_fallback_disabled_by_default(monkeypatch, caplog):
    monkeypatch.delenv("GATEWAY_VISION_FALLBACK", raising=False)
    monkeypatch.setattr(
        server_module,
        "resolve",
        lambda model: pytest.fail(f"disabled policy must not resolve {model}"),
    )

    with caplog.at_level(logging.INFO, logger="model-gateway"):
        server_module._validate_vision_fallback_policy()

    assert "vision fallback policy: disabled" in caplog.text
    assert "fails closed" in caplog.text


@pytest.mark.parametrize(
    ("fallback_info", "message"),
    [
        (None, "is not resolvable"),
        (_info("none", provider="omlx", vision=False), "is not vision-capable"),
    ],
)
def test_startup_rejects_invalid_opt_in_vision_fallback(
    monkeypatch, fallback_info, message,
):
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "bad-fallback")
    monkeypatch.setattr(server_module, "resolve", lambda model: fallback_info)

    with pytest.raises(RuntimeError, match=message):
        server_module._validate_vision_fallback_policy()


def test_startup_rejects_protocol_incompatible_vision_fallback(monkeypatch):
    fallback_info = _info("none", provider="anthropic", vision=True)
    fallback_info.protocol = "anthropic"
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "anthropic-vision")
    monkeypatch.setattr(server_module, "resolve", lambda model: fallback_info)

    with pytest.raises(RuntimeError, match="OpenAI-compatible protocol"):
        server_module._validate_vision_fallback_policy()


def test_startup_rejects_mixed_local_cloud_fallback_pool(monkeypatch):
    infos = {
        "omlx": _info("none", provider="omlx", vision=True),
        "fireworks": _info("none", provider="fireworks", vision=True),
    }
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "mixed-vision")
    monkeypatch.setattr(server_module, "pool_candidates", lambda model: list(infos))
    monkeypatch.setattr(
        server_module,
        "resolve",
        lambda model, provider_override=None: infos[provider_override],
    )

    with pytest.raises(RuntimeError, match="mixes local and cloud providers"):
        server_module._validate_vision_fallback_policy()


def test_request_revalidates_opt_in_fallback_policy(monkeypatch):
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "vision-fallback")

    def reject_policy(**kwargs):
        assert kwargs == {"log_policy": False}
        raise RuntimeError("mixed locality")

    monkeypatch.setattr(server_module, "_validate_vision_fallback_policy", reject_policy)

    _, _, error = server_module._resolve_vision_fallback(
        "text-model", _info("none", provider="omlx", vision=False),
    )

    assert error.status_code == 502
    assert b"Vision fallback policy is invalid: mixed locality" in error.body


def test_startup_rejects_composite_as_global_vision_fallback(monkeypatch):
    composite_info = _info("none", provider="omlx", vision=True)
    composite_info.composite = providers.CompositeRoute(
        text_model="text-local",
        vision_model="vision-local",
        image_handling="extract_then_answer",
        max_images=4,
    )
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "best-local")
    monkeypatch.setattr(
        server_module, "resolve", lambda model, provider_override=None: composite_info,
    )

    with pytest.raises(RuntimeError, match="not a composite"):
        server_module._validate_vision_fallback_policy()


@pytest.mark.parametrize(
    ("provider", "cloud_egress", "level"),
    [("omlx", "false", logging.INFO), ("fireworks", "true", logging.WARNING)],
)
def test_startup_logs_effective_opt_in_vision_policy(
    monkeypatch, caplog, provider, cloud_egress, level,
):
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "vision-fallback")
    fallback_info = _info("none", provider=provider, vision=True)
    monkeypatch.setattr(server_module, "resolve", lambda model: fallback_info)

    with caplog.at_level(level, logger="model-gateway"):
        server_module._validate_vision_fallback_policy()

    assert "vision fallback policy: enabled model=vision-fallback" in caplog.text
    assert f"providers={provider}" in caplog.text
    assert f"cloud_egress={cloud_egress}" in caplog.text


def test_replace_image_preserves_order_and_all_content():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": "one"}},
        {"type": "custom", "value": 7},
        {"type": "text", "text": "after"},
    ]}]}
    result = server_module._replace_images_with_extracted_text(body, "visible detail", "vision")
    parts = result["messages"][0]["content"]
    assert [part["type"] for part in parts] == ["text", "text", "custom", "text"]
    assert "image 1" in parts[1]["text"]
    assert parts[0]["text"] == "before" and parts[3]["text"] == "after"
    assert body["messages"][0]["content"][1]["type"] == "image_url"


def test_replace_multiple_images_uses_matching_observations():
    result = server_module._replace_images_with_extracted_text(
        {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "one"}},
            {"type": "image_url", "image_url": {"url": "two"}},
        ]}]},
        ["first detail", "second detail"], "vision",
    )
    parts = result["messages"][0]["content"]
    assert "image 1" in parts[0]["text"] and "first detail" in parts[0]["text"]
    assert "image 2" in parts[1]["text"] and "second detail" in parts[1]["text"]


def test_replace_images_rejects_observation_count_mismatch():
    with pytest.raises(ValueError, match="count does not match"):
        server_module._replace_images_with_extracted_text(
            {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "one"}},
                {"type": "image_url", "image_url": {"url": "two"}},
            ]}]},
            ["only one detail"], "vision",
        )


def test_replace_images_rejects_empty_observations():
    with pytest.raises(ValueError, match="empty observations"):
        server_module._replace_images_with_extracted_text(
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
            "  ", "vision",
        )


def test_multi_image_extraction_is_bounded_and_ordered(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200

        def __init__(self, observation):
            self.observation = observation

        def json(self):
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": self.observation}}],
                "usage": {},
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, endpoint, json, headers):
            calls.append((endpoint, json, headers))
            return FakeResponse(f"observation {len(calls)}")

    monkeypatch.setattr(server_module.httpx, "AsyncClient", lambda **kwargs: FakeClient())
    monkeypatch.setattr(server_module, "_ledger_record", lambda *args, **kwargs: None)
    request = SimpleNamespace(state=SimpleNamespace())
    info = _info("", thinking="", provider="omlx", provider_model_id="gemma-upstream", vision=True)
    body = {"messages": [{"role": "tool", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQID"}},
    ]}]}

    result = asyncio.run(server_module._extract_image_observations(
        request, body, "gemma-local", info,
        max_images=4, require_inline_images=True,
    ))

    assert result == ["observation 1", "observation 2"]
    assert len(calls) == 2
    assert all(server_module._image_count(call[1]) == 1 for call in calls)
    assert all(len(call[1]["messages"]) == 2 for call in calls)


def test_local_composite_rejects_remote_images_before_upstream(monkeypatch):
    class ForbiddenClient:
        def __init__(self, **kwargs):
            raise AssertionError("image validation must happen before creating a client")

    monkeypatch.setattr(server_module.httpx, "AsyncClient", ForbiddenClient)
    request = SimpleNamespace(state=SimpleNamespace())
    info = _info("", thinking="", provider="omlx", provider_model_id="gemma-upstream", vision=True)
    result = asyncio.run(server_module._extract_image_observations(
        request,
        {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
        ]}]},
        "gemma-local",
        info,
        max_images=4,
        require_inline_images=True,
    ))

    assert result.status_code == 400
    assert "inline data:image" in result.body.decode()


@pytest.mark.parametrize(
    ("urls", "max_images", "expected_status", "message"),
    [
        (["data:image/png;base64,AAAA"] * 5, 4, 400, "1 through 4"),
        (["data:image/png;base64,%%%"], 4, 400, "malformed base64"),
    ],
)
def test_local_composite_rejects_invalid_image_batches_before_upstream(
    monkeypatch, urls, max_images, expected_status, message,
):
    class ForbiddenClient:
        def __init__(self, **kwargs):
            raise AssertionError("image validation must happen before creating a client")

    monkeypatch.setattr(server_module.httpx, "AsyncClient", ForbiddenClient)
    request = SimpleNamespace(state=SimpleNamespace())
    info = _info("", thinking="", provider="omlx", provider_model_id="gemma-upstream", vision=True)
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}} for url in urls
    ]}]}

    result = asyncio.run(server_module._extract_image_observations(
        request, body, "gemma-local", info,
        max_images=max_images, require_inline_images=True,
    ))

    assert result.status_code == expected_status
    assert message in result.body.decode()


def test_local_composite_enforces_per_image_and_total_byte_limits(monkeypatch):
    class ForbiddenClient:
        def __init__(self, **kwargs):
            raise AssertionError("size validation must happen before creating a client")

    monkeypatch.setattr(server_module.httpx, "AsyncClient", ForbiddenClient)
    monkeypatch.setattr(server_module, "DEFAULT_COMPOSITE_IMAGE_MAX_BYTES", 2)
    monkeypatch.setattr(server_module, "DEFAULT_COMPOSITE_IMAGE_TOTAL_MAX_BYTES", 3)
    request = SimpleNamespace(state=SimpleNamespace())
    info = _info("", thinking="", provider="omlx", provider_model_id="gemma-upstream", vision=True)

    per_image = asyncio.run(server_module._extract_image_observations(
        request,
        {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQID"}},
        ]}]},
        "gemma-local", info, max_images=4, require_inline_images=True,
    ))
    total = asyncio.run(server_module._extract_image_observations(
        request,
        {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQI="}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AwQ="}},
        ]}]},
        "gemma-local", info, max_images=4, require_inline_images=True,
    ))

    assert per_image.status_code == 413 and "per-image byte limit" in per_image.body.decode()
    assert total.status_code == 413 and "total byte limit" in total.body.decode()


@pytest.mark.parametrize("finish_reason", ["length", None, ""])
def test_local_composite_rejects_nonterminal_observations(monkeypatch, finish_reason):
    calls = 0

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [{
                    "finish_reason": finish_reason,
                    "message": {"content": "partial observation"},
                }],
                "usage": {},
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, endpoint, json, headers):
            nonlocal calls
            calls += 1
            return FakeResponse()

    monkeypatch.setattr(server_module.httpx, "AsyncClient", lambda **kwargs: FakeClient())
    monkeypatch.setattr(server_module, "_ledger_record", lambda *args, **kwargs: None)
    request = SimpleNamespace(state=SimpleNamespace())
    info = _info("", thinking="", provider="omlx", provider_model_id="gemma-upstream", vision=True)

    result = asyncio.run(server_module._extract_image_observations(
        request,
        {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]},
        "gemma-local", info, max_images=4, require_inline_images=True,
    ))

    assert calls == 1
    assert result.status_code == 502
    assert f"finish_reason={finish_reason}" in result.body.decode()


@pytest.mark.parametrize("content", [
    None,
    {},
    7,
    [{"type": "image_url", "image_url": {"url": "x"}}],
    [{"type": "text", "text": "partial"}, {"type": "image_url", "image_url": {"url": "x"}}],
])
def test_local_composite_rejects_nontext_observations(monkeypatch, content):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {},
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, endpoint, json, headers):
            return FakeResponse()

    monkeypatch.setattr(server_module.httpx, "AsyncClient", lambda **kwargs: FakeClient())
    monkeypatch.setattr(server_module, "_ledger_record", lambda *args, **kwargs: None)
    request = SimpleNamespace(state=SimpleNamespace())
    info = _info("", thinking="", provider="omlx", provider_model_id="gemma-upstream", vision=True)

    result = asyncio.run(server_module._extract_image_observations(
        request,
        {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]},
        "gemma-local", info, max_images=4, require_inline_images=True,
    ))

    assert result.status_code == 502
    assert "empty observations" in result.body.decode()


def test_responses_input_image_translates_to_real_image_block():
    from src.responses import responses_to_chat
    result = responses_to_chat({"input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "inspect"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]}]})
    assert result["messages"][0]["content"][1] == {
        "type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"},
    }


def test_responses_input_image_translates_to_anthropic_image_block():
    result = server_module._responses_to_anthropic_messages({
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "inspect"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            {"type": "input_image", "image_url": "https://example.test/image.png"},
        ]}],
    })
    parts = result["messages"][0]["content"]
    assert parts[1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    assert parts[2] == {"type": "image", "source": {"type": "url", "url": "https://example.test/image.png"}}


def test_anthropic_image_translation_preserves_interleaved_order():
    from src.translator import anthropic_to_openai
    result = anthropic_to_openai({"model": "m", "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "ONE"}},
        {"type": "text", "text": "between"},
        {"type": "image", "source": {"type": "url", "url": "https://example.test/two.png"}},
    ]}]})
    parts = result["messages"][0]["content"]
    assert [part["type"] for part in parts] == ["image_url", "text", "image_url"]
    assert parts[2]["image_url"]["url"] == "https://example.test/two.png"


def test_anthropic_tool_results_preserve_block_sequence():
    from src.translator import anthropic_to_openai
    result = anthropic_to_openai({"model": "m", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "before"},
        {"type": "tool_result", "tool_use_id": "call_1", "content": "result"},
        {"type": "text", "text": "after"},
    ]}]})
    assert [(message["role"], message["content"]) for message in result["messages"]] == [
        ("user", "before"), ("tool", "result"), ("user", "after"),
    ]


def test_anthropic_tool_result_preserves_images_for_gateway_staging():
    from src.translator import anthropic_to_openai
    result = anthropic_to_openai({"model": "m", "messages": [{"role": "user", "content": [{
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": [
            {"type": "text", "text": "screenshot"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        ],
    }]}]})
    tool = result["messages"][0]
    assert tool["role"] == "tool"
    assert tool["content"][0] == {"type": "text", "text": "screenshot"}
    assert tool["content"][1]["image_url"]["url"] == "data:image/png;base64,AAAA"
    assert server_module._image_count(result) == 1


def test_responses_tool_output_preserves_images_for_gateway_staging():
    from src.responses import responses_to_chat
    result = responses_to_chat({"input": [{
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [
            {"type": "input_text", "text": "screenshot"},
            {"type": "input_text", "text": {"page": 2}},
            {"type": "input_file", "filename": "report.pdf"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ],
    }]})
    tool = result["messages"][0]
    assert tool["role"] == "tool"
    assert tool["content"][0] == {"type": "text", "text": "screenshot"}
    assert tool["content"][1] == {"type": "text", "text": '{"page": 2}'}
    assert tool["content"][2] == {"type": "text", "text": "[file: report.pdf]"}
    assert tool["content"][3]["image_url"]["url"] == "data:image/png;base64,AAAA"
    assert server_module._image_count(result) == 1


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_responses_file_id_rejected_on_translated_route(client, monkeypatch):
    monkeypatch.setattr(server_module, "resolve", lambda model: _info("none", provider="test", vision=False))
    response = client.post("/v1/responses", json={
        "model": "text-model", "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_image", "file_id": "file_123"},
        ]}],
    })
    assert response.status_code == 400
    assert "file_id" in response.json()["error"]["message"]


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_responses_native_openai_preserves_file_id(client, monkeypatch):
    native = _info("none", provider="openai", provider_model_id="native-responses", vision=False)
    monkeypatch.setattr(server_module, "resolve", lambda model: native)

    async def fake_passthrough(endpoint, body, headers, is_stream, **kwargs):
        assert body["model"] == "native-responses"
        assert body["input"][0]["content"][0]["file_id"] == "file_123"
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, "_handle_openai_responses_passthrough", fake_passthrough)
    response = client.post("/v1/responses", json={
        "model": "vision-model", "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_image", "file_id": "file_123"},
        ]}],
    })
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
@pytest.mark.parametrize("mode", ["typo", "extract_then_answer"])
def test_responses_native_file_id_rejects_non_native_handling(client, monkeypatch, mode):
    native = _info("none", provider="openai", provider_model_id="native-responses", vision=False)
    monkeypatch.setattr(server_module, "resolve", lambda model: native)
    response = client.post("/v1/responses", json={
        "model": "vision-model", "gateway_image_handling": mode,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_image", "file_id": "file_123"},
        ]}],
    })
    assert response.status_code == 400


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_invalid_image_handling_mode_is_rejected(client, monkeypatch):
    monkeypatch.setattr(server_module, "resolve", lambda model: _info("none", provider="test", vision=False))
    response = client.post("/v1/chat/completions", json={
        "model": "text-model", "gateway_image_handling": "typo", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "x"}},
        ]}],
    })
    assert response.status_code == 400
    assert "Unsupported" in response.json()["error"]["message"]


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
@pytest.mark.parametrize("fallback_env", [None, ""], ids=["unset", "empty"])
def test_chat_text_only_image_fails_closed_by_default(client, monkeypatch, fallback_env):
    if fallback_env is None:
        monkeypatch.delenv("GATEWAY_VISION_FALLBACK", raising=False)
    else:
        monkeypatch.setenv("GATEWAY_VISION_FALLBACK", fallback_env)
    monkeypatch.setattr(server_module, "resolve", lambda model: _info("none", provider="test", vision=False))
    response = client.post("/v1/chat/completions", json={
        "model": "text-model", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "x"}},
        ]}],
    })
    assert response.status_code == 400
    assert set(response.json()) == {"error"}
    message = response.json()["error"]["message"]
    assert "text-only" in message
    assert "GATEWAY_VISION_FALLBACK=<model>" in message


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
@pytest.mark.parametrize("path,payload", [
    ("/v1/responses", {
        "model": "text-model", "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": "x"},
        ]}],
    }),
    ("/v1/messages", {
        "model": "text-model", "max_tokens": 32, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        ]}],
    }),
])
def test_other_endpoints_text_only_image_fail_closed_when_fallback_empty(client, monkeypatch, path, payload):
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "")
    monkeypatch.setattr(server_module, "resolve", lambda model: _info("none", provider="test", vision=False))
    response = client.post(path, json=payload)
    assert response.status_code == 400
    error = response.json()
    if path == "/v1/messages":
        assert error["type"] == "error"
        assert error["error"]["type"] == "invalid_request_error"
    else:
        assert set(error) == {"error"}


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "gpt-5.6"])
def test_configured_gpt_vision_models_bypass_fallback(client, monkeypatch, model):
    native = _info("openai-responses", provider="databricks", provider_model_id=f"upstream-{model}", vision=True)
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "unavailable-fallback")
    monkeypatch.setattr(server_module, "resolve", lambda requested: native if requested == model else None)

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        assert body["model"] == f"upstream-{model}"
        return server_module.JSONResponse(status_code=200, content={"served": model})

    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)
    response = client.post("/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ]}],
    })
    assert response.status_code == 200
    assert response.json()["served"] == model


def test_chat_native_vision_image_bypasses_fallback(client, monkeypatch):
    native = _info("none", thinking="", provider="test", provider_model_id="native-vision", vision=True)
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "missing-fallback")
    monkeypatch.setattr(server_module, "resolve", lambda model: native if model == "vision-model" else None)

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        assert body["model"] == "native-vision"
        assert server_module._payload_has_image(body)
        return server_module.JSONResponse(status_code=200, content={"model": body["model"]})

    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)
    response = client.post("/v1/chat/completions", json={
        "model": "vision-model", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "x"}},
        ]}],
    })
    assert response.status_code == 200
    assert response.json()["model"] == "native-vision"


# ── format inference ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("provider,model_id,explicit,expected", [
    ("anthropic", "claude-x", "", "anthropic"),
    ("openrouter", "m", "", "openrouter"),
    ("zhipuai", "glm-5.2", "", "zai"),
    ("bigmodel", "glm", "", "zai"),
    ("openai", "gpt-x", "", "openai"),            # non-responses target
    ("google", "gemini", "", "google-openai"),
    ("omlx", "qwen3", "", "qwen-chat-template"),
    ("omlx", "glm-5", "", "glm-chat-template"),
    ("zai_coding", "glm-5.2", "zai", "zai"),       # explicit wins
])
def test_infer_thinking_format(provider, model_id, explicit, expected):
    info = SimpleNamespace(provider=provider, provider_model_id=model_id,
                           thinking_format=explicit)
    assert _infer_thinking_format(info, target_api="chat") == expected


def test_openai_responses_format_infers_openai_for_chat_target():
    """explicit 'openai-responses' collapses to 'openai' for chat/completions."""
    info = SimpleNamespace(provider="openai", provider_model_id="gpt-x",
                           thinking_format="openai-responses")
    assert _infer_thinking_format(info, target_api="chat") == "openai"
    assert _infer_thinking_format(info, target_api="responses") == "openai-responses"


# ── integration: every real routable model resolves cleanly ──────────────────

def test_every_routable_model_has_known_format():
    """No model in model-info.json should resolve to an unhandled format, and a
    max-effort probe must not raise for any of them. Catches new models added
    without a thinking_format, and guards the observability endpoint's probe."""
    seen_formats = set()
    for entry in list_models():
        info = ProviderInfo(
            provider=entry.get("provider", "x"),
            base_url="http://up",
            api_key="k",
            provider_model_id=entry.get("provider_model_id", entry["id"]),
            thinking=entry.get("thinking", ""),
            thinking_format=entry.get("thinking_format", ""),
            max_output_tokens=entry.get("max_output_tokens", 32768) or 32768,
        )
        fmt = _infer_thinking_format(info, target_api="chat")
        seen_formats.add(fmt)
        # A max-effort probe (the same one the debug endpoint runs) must not raise.
        req = {"messages": [], "reasoning_effort": "max"}
        _apply_gateway_reasoning(req, info, target_api="chat")
    # Every format we exercised in the unit matrix should appear, or be absent
    # only because no model currently uses it.
    assert seen_formats.issubset({
        "zai", "openai", "openai-responses", "anthropic", "openrouter",
        "qwen", "qwen-chat-template", "glm-chat-template", "deepseek",
        "deepseek-v4-dsml", "google-openai", "none",
    })


# ── observability endpoints ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_models_endpoint_exposes_thinking_fields(client, monkeypatch):
    monkeypatch.setattr(providers, "_config", {
        "providers": {
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
            }
        }
    })
    providers._models = None
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data, "/v1/models returned no models"
    required = {"id", "thinking", "thinking_format", "thinking_levels",
                "max_reachable", "forwarded_params"}
    for m in data:
        assert required.issubset(m), f"missing thinking fields on {m.get('id')}"
        assert isinstance(m["forwarded_params"], list)
        assert isinstance(m["max_reachable"], bool)


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_debug_thinking_endpoint_matrix(client):
    resp = client.get("/v1/debug/thinking")
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["models"] == len(body["models"])
    assert body["summary"]["max_reachable"] + body["summary"]["max_unreachable"] \
        == body["summary"]["models"]
    assert "by_format" in body["summary"]


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_debug_thinking_zai_max_reachable(client):
    """The zai branch now forwards reasoning_effort, so glm-5.2-zai is
    max-reachable in the debug matrix: forwarded_params includes
    reasoning_effort alongside enable_thinking."""
    resp = client.get("/v1/debug/thinking")
    assert resp.status_code == 200
    rows = {r["name"]: r for r in resp.json()["models"]}
    assert "glm-5.2-zai" in rows, "glm-5.2-zai missing from debug matrix"
    zai = rows["glm-5.2-zai"]
    assert zai["thinking_format"] == "zai"
    assert zai["forwarded_params"] == ["enable_thinking", "reasoning_effort"]
    assert zai["max_reachable"] is True
