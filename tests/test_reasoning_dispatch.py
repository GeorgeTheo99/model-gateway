"""Regression tests for the gateway reasoning/thinking dispatch.

These lock in the exact upstream params each `thinking_format` forwards, so a
silent divergence (e.g. the `zai` branch dropping `reasoning_effort`) becomes a
visible failure instead of a quiet behavior change.

Run:  cd model-gateway && uv run pytest
"""

import base64
import io
from types import SimpleNamespace

import pytest

from src.providers import ProviderInfo, list_models
import src.server as server_module
from src.server import _apply_gateway_reasoning, _infer_thinking_format, app

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

    async def fake_passthrough_sync(endpoint, body, headers):
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
def test_chat_completions_local_omlx_proxies_to_upstream_model(client, monkeypatch):
    info = _info(
        "glm-chat-template",
        provider="omlx",
        provider_model_id="local-upstream",
    )

    def fake_resolve(model):
        return info if model == "local-alias" else None

    async def fake_passthrough_sync(endpoint, body, headers):
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
def test_chat_completions_image_request_uses_fireworks_default_vision_fallback(client, monkeypatch):
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
        if model == server_module.DEFAULT_VISION_FALLBACK_MODEL:
            return fallback_info
        return None

    async def fake_passthrough_sync(endpoint, body, headers):
        assert endpoint == "http://up/chat/completions"
        assert body["model"] == "accounts/fireworks/models/qwen3p7-plus"
        assert "prompt_cache_key" not in body
        assert "prompt_cache_retention" not in body
        assert "reasoning" not in body["messages"][0]
        assert "reasoning_content" not in body["messages"][0]
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.delenv("GATEWAY_VISION_FALLBACK", raising=False)
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
def test_models_endpoint_exposes_thinking_fields(client):
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
