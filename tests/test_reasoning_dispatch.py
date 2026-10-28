"""Regression tests for the gateway reasoning/thinking dispatch.

These lock in the exact upstream params each `thinking_format` forwards, so a
silent divergence (e.g. the `zai` branch dropping `reasoning_effort`) becomes a
visible failure instead of a quiet behavior change.

Run:  cd model-gateway && uv run pytest
"""

import asyncio
import base64
import copy
import io
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.providers as providers
from src.providers import ProviderInfo, list_models
import src.server as server_module
from src.server import _apply_gateway_reasoning, _infer_thinking_format, app
from src.translator import anthropic_to_openai, anthropic_to_openai_chat, openai_chat_to_anthropic

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - fastapi is a runtime dep
    TestClient = None


# ── helpers ──────────────────────────────────────────────────────────────────

def _info(fmt: str, *, thinking: str = "always", thinking_levels=None,
          provider: str = "x", provider_model_id: str = "m",
          max_output_tokens: int = 32768, vision: bool = False,
          quirks=frozenset()) -> ProviderInfo:
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
        thinking_levels=thinking_levels,
        thinking_format=fmt,
        vision=vision,
        quirks=frozenset(quirks),
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
#   B. always-thinking model, client asks max    -> provider-native mapping after validation
#   C. optional model, client requests Off       -> enabled False, format-specific shape

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

    # ── C: client disables thinking on an optional model ──────────────────
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
    mode = "optional" if fragment.get("reasoning_effort") == "none" else "always"
    enabled, view = _run(fmt, req, target_api=target_api, thinking=mode)
    assert enabled is exp_enabled
    assert view == exp_view


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_fireworks_always_model_advertises_minimal_but_forwards_low(client, monkeypatch):
    """Fireworks keeps the canonical minimal level public but receives low."""
    entry = {
        "name": "glm-5.3-fw",
        "provider": "fireworks",
        "provider_model_id": "accounts/fireworks/models/glm-5p3",
        "thinking": "always",
    }
    monkeypatch.setattr(providers, "_config", {
        "providers": {"fireworks": {"base_url": "http://up", "api_key": "k"}},
    })
    monkeypatch.setattr(providers, "_models", {entry["name"]: entry})

    advertised = client.get("/v1/models")
    assert advertised.status_code == 200
    model = next(row for row in advertised.json()["data"] if row["id"] == entry["name"])
    assert model["thinking_levels"] == ["minimal", "low", "medium", "high", "xhigh", "max"]

    async def fake_passthrough(endpoint, body, headers, **kwargs):
        assert endpoint == "http://up/chat/completions"
        assert body["model"] == entry["provider_model_id"]
        assert body["reasoning_effort"] == "low"
        assert body["reasoning_effort"] != "minimal"
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough)
    response = client.post("/v1/chat/completions", json={
        "model": entry["name"],
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "minimal",
    })
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.parametrize("effort,expected", [("max", "xhigh"), ("medium", "medium")])
def test_fireworks_default_format_preserves_other_effort_mappings(effort, expected):
    _, view = _run(
        "",
        {"messages": [], "reasoning_effort": effort},
        provider="fireworks",
        provider_model_id="accounts/fireworks/models/glm-5p3",
    )
    assert view["reasoning_effort"] == expected


@pytest.mark.parametrize("effort", ["minimal", "low"])
def test_zai_still_maps_finer_efforts_to_native_high(effort):
    _, view = _run("zai", {"messages": [], "reasoning_effort": effort})
    assert view["reasoning_effort"] == "high"


GLM53_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")


def test_zai_glm53_uses_required_thinking_shape_and_defaults_to_max():
    enabled, view = _run(
        "zai",
        {"messages": []},
        thinking_levels=GLM53_LEVELS,
        quirks={"zai_glm53_thinking"},
    )
    assert enabled is True
    assert view == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }


@pytest.mark.parametrize(
    "effort,expected",
    [
        ("none", "low"),
        ("minimal", "low"),
        ("light", "low"),
        ("low", "low"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "max"),
        ("max", "max"),
        ("ultra", "max"),
        ("unknown", "max"),
    ],
)
def test_zai_glm53_maps_coding_plan_efforts(effort, expected):
    enabled, view = _run(
        "zai",
        {"messages": [], "reasoning_effort": effort},
        thinking_levels=GLM53_LEVELS,
        quirks={"zai_glm53_thinking"},
    )
    assert enabled is True
    assert view == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": expected,
    }


def test_zai_glm53_explicit_effort_precedes_disabled_toggle_and_preserves_clear_thinking():
    enabled, view = _run(
        "zai",
        {
            "messages": [],
            "reasoning_effort": "high",
            "thinking": {"type": "disabled", "clear_thinking": False},
        },
        thinking_levels=GLM53_LEVELS,
        quirks={"zai_glm53_thinking"},
    )
    assert enabled is True
    assert view == {
        "thinking": {"type": "enabled", "clear_thinking": False},
        "reasoning_effort": "high",
    }


@pytest.mark.parametrize("thinking", [False, "disabled", {"type": "disabled"}])
def test_zai_glm53_disabled_toggle_becomes_low(thinking):
    enabled, view = _run(
        "zai",
        {"messages": [], "thinking": thinking},
        thinking_levels=GLM53_LEVELS,
        quirks={"zai_glm53_thinking"},
    )
    assert enabled is True
    assert view == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }


KIMI_LEVELS = ("low", "high", "max")


def test_kimi_defaults_to_max():
    for req in ({"messages": []}, {"messages": [], "thinking": True},
                {"messages": [], "thinking": {"type": "enabled"}}):
        enabled, view = _run("kimi", req, thinking_levels=KIMI_LEVELS)
        assert enabled is True
        assert view == {"reasoning_effort": "max"}


@pytest.mark.parametrize("level", KIMI_LEVELS)
@pytest.mark.parametrize("field", ["reasoning_effort", "reasoning", "output_config"])
def test_kimi_forwards_each_level_verbatim(level, field):
    req = {"messages": []}
    if field == "reasoning_effort":
        req["reasoning_effort"] = level
    elif field == "reasoning":
        req["reasoning"] = {"effort": level}
    else:
        req.update(thinking={"type": "adaptive"}, output_config={"effort": level})
    enabled, view = _run("kimi", req, thinking_levels=KIMI_LEVELS)
    assert enabled is True
    assert view == {"reasoning_effort": level}


@pytest.mark.parametrize("level", ["none", "minimal", "medium", "xhigh"])
def test_kimi_rejects_levels_it_does_not_declare(level):
    # Moonshot silently accepts unknown efforts, so the gateway must refuse them.
    with pytest.raises(server_module.ThinkingValidationError, match="not supported"):
        _run("kimi", {"messages": [], "reasoning_effort": level}, thinking_levels=KIMI_LEVELS)


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


def test_always_model_strictly_rejects_off():
    with pytest.raises(server_module.ThinkingValidationError, match="not supported"):
        _run("openai", {"messages": [], "reasoning_effort": "none"})


def test_legacy_thinking_false_is_auto_and_true_selects_supported_level():
    disabled_req = {"messages": [], "thinking": False}
    enabled, view = _run("openai", disabled_req, thinking="optional")
    assert enabled is False
    assert view == {}

    enabled, view = _run("openai", {"messages": [], "thinking": False})
    assert enabled is True
    assert view == {"reasoning_effort": "high"}

    enabled, view = _run(
        "openai", {"messages": [], "thinking": True}, thinking="optional",
        thinking_levels=("off", "max"),
    )
    assert enabled is True
    assert view == {"reasoning_effort": "xhigh"}


def test_kimi_max_only_rejects_lower_levels_and_translates_max_after_validation():
    with pytest.raises(server_module.ThinkingValidationError, match="supported levels: max"):
        _run("openai", {"messages": [], "reasoning_effort": "high"},
             thinking_levels=("max",))
    enabled, view = _run("openai", {"messages": []}, thinking_levels=("max",))
    assert enabled is True
    assert view == {"reasoning_effort": "xhigh"}


def test_kimi_max_only_legacy_enabled_budget_dispatches_canonical_max():
    enabled, view = _run(
        "openai",
        {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 4096}},
        thinking_levels=("max",),
    )
    assert enabled is True
    assert view == {"reasoning_effort": "xhigh"}


def test_dispatch_reuses_the_single_validated_reasoning_control(monkeypatch):
    original = server_module._extract_reasoning_control
    calls = 0

    def counted_extract(req, info):
        nonlocal calls
        calls += 1
        return original(req, info)

    monkeypatch.setattr(server_module, "_extract_reasoning_control", counted_extract)
    enabled, view = _run(
        "openai",
        {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 4096}},
        thinking_levels=("max",),
    )
    assert enabled is True
    assert view == {"reasoning_effort": "xhigh"}
    assert calls == 1


@pytest.mark.parametrize("model_id", [
    "claude-opus-5", "databricks-claude-opus-5", "system.ai.claude-opus-5",
    "system.ai.claude-opus-4-7", "system.ai.claude-opus-4-8",
])
def test_adaptive_anthropic_legacy_budget_preserves_prior_effort_inference(model_id):
    info = _info(
        "anthropic",
        thinking="optional",
        thinking_levels=("off", "low", "medium", "high", "xhigh", "max"),
        provider="anthropic",
        provider_model_id=model_id,
    )
    body = {
        "messages": [],
        "thinking": {"type": "enabled", "budget_tokens": 2000},
    }
    assert _apply_gateway_reasoning(body, info, target_api="chat") is True
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}


def test_opus_5_rejects_unsupported_minimal_effort():
    with pytest.raises(server_module.ThinkingValidationError, match="Thinking level 'minimal'"):
        _run(
            "anthropic",
            {"messages": [], "reasoning_effort": "minimal"},
            thinking="optional",
            thinking_levels=("off", "low", "medium", "high", "xhigh", "max"),
            provider="anthropic",
            provider_model_id="claude-opus-5",
        )


def test_fable_5_rejects_unsupported_minimal_effort():
    with pytest.raises(server_module.ThinkingValidationError, match="Thinking level 'minimal'"):
        _run(
            "anthropic",
            {"messages": [], "reasoning_effort": "minimal"},
            thinking="always",
            thinking_levels=("low", "medium", "high", "xhigh", "max"),
            provider="anthropic",
            provider_model_id="claude-fable-5",
        )


def test_adaptive_anthropic_strips_fixed_sampling_fields_when_thinking_enabled():
    info = _info(
        "anthropic",
        thinking_levels=("low", "medium", "high", "xhigh", "max"),
        provider="anthropic",
        provider_model_id="claude-fable-5",
    )
    body = {
        "messages": [],
        "reasoning_effort": "high",
        "temperature": 0,
        "top_p": 0.9,
    }

    assert _apply_gateway_reasoning(body, info, target_api="messages") is True
    assert "temperature" not in body
    assert "top_p" not in body
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}


def test_adaptive_anthropic_opus_ids_match_provider_model_ids():
    # The adaptive set is matched against info.provider_model_id (hyphenated),
    # not the catalog name (dotted). Keep the two in sync with provider_model_id.
    assert server_module._uses_adaptive_anthropic_thinking("claude-fable-5") is True
    assert server_module._uses_adaptive_anthropic_thinking("claude-opus-5") is True
    assert server_module._uses_adaptive_anthropic_thinking("claude-opus-4-7") is True
    assert server_module._uses_adaptive_anthropic_thinking("claude-opus-4-8") is True


@pytest.mark.parametrize("model_id", [
    "system.ai.claude-opus-5", "system.ai.claude-opus-4-7", "system.ai.claude-opus-4-8",
])
def test_unity_native_adaptive_thinking_preserved(model_id):
    info = _info("anthropic", provider_model_id=model_id, thinking="optional")
    body = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
    server_module._normalize_anthropic_adaptive_thinking(body, info)
    assert _apply_gateway_reasoning(body, info, target_api="messages") is True
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}


@pytest.mark.parametrize("model_id", ["system.ai.claude-sonnet-4-6", "custom.ai.claude-opus-5"])
def test_unity_adaptive_detection_does_not_match_other_models(model_id):
    assert server_module._uses_adaptive_anthropic_thinking(model_id) is False


def test_unknown_or_overridden_unsupported_level_is_never_silently_dropped():
    with pytest.raises(server_module.ThinkingValidationError, match="Unknown thinking level"):
        _run("openai", {"messages": [], "reasoning_effort": "turbo"}, thinking="optional")
    with pytest.raises(server_module.ThinkingValidationError, match="Thinking level 'low'"):
        _run(
            "openai",
            {"messages": [], "reasoning": {"effort": "low"}, "reasoning_effort": "max"},
            thinking_levels=("max",),
        )


def test_no_thinking_model_rejects_explicit_enabled_control():
    info = ProviderInfo(provider="x", base_url="http://up", api_key="k",
                        provider_model_id="m", thinking="", thinking_format="")
    req = {"messages": [], "thinking": {"type": "enabled"}}
    with pytest.raises(server_module.ThinkingValidationError, match="does not support"):
        _apply_gateway_reasoning(req, info, target_api="chat")


def test_no_thinking_model_accepts_legacy_none_as_noop():
    info = ProviderInfo(provider="x", base_url="http://up", api_key="k",
                        provider_model_id="m", thinking="", thinking_format="")
    req = {"messages": [], "reasoning_effort": "none"}
    assert _apply_gateway_reasoning(req, info, target_api="chat") is False
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
    }, thinking="optional")
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
@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "low",
        }),
        ("/v1/responses", {"input": "hi", "reasoning": {"effort": "low"}}),
        ("/v1/messages", {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
        }),
    ],
)
def test_endpoints_reject_unsupported_level_with_protocol_error_shape(
    client, monkeypatch, endpoint, payload,
):
    info = _info("openai", thinking_levels=("max",), provider="moonshot")
    monkeypatch.setattr(server_module, "resolve", lambda model: info if model == "max-only" else None)
    response = client.post(endpoint, json={"model": "max-only", **payload})
    assert response.status_code == 400
    body = response.json()
    error = body["error"]
    assert error["type"] == "invalid_request_error"
    assert "supported levels: max" in error["message"]
    if endpoint == "/v1/messages":
        assert body["type"] == "error"
    else:
        assert "type" not in body


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_chat_completions_applies_declarative_provider_onboarding_quirks(client, monkeypatch):
    info = _info("openai", provider="new-provider", provider_model_id="new-upstream", vision=True)
    info.quirks = frozenset({
        "force_reasoning_effort_max",
        "use_max_completion_tokens",
        "drop_fixed_sampling_fields",
        "named_tool_choice_as_required",
        "inline_image_urls_only",
    })

    monkeypatch.setattr(server_module, "resolve", lambda model: info if model == "new-model" else None)

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        assert endpoint == "http://up/chat/completions"
        assert body["model"] == "new-upstream"
        assert body["reasoning_effort"] == "max"
        assert body["max_completion_tokens"] == 123
        assert body["tool_choice"] == "required"
        assert [tool["function"]["name"] for tool in body["tools"]] == ["run_python"]
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
        "tools": [
            {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "run_python", "parameters": {"type": "object"}}},
        ],
        "tool_choice": {"type": "function", "function": {"name": "run_python"}},
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
def test_fireworks_kimi_max_only_forwards_native_named_tool_and_stream(client, monkeypatch):
    info = _info(
        "openai", thinking_levels=("max",), provider="fireworks",
        provider_model_id="accounts/fireworks/models/kimi-k3", vision=True,
    )
    info.quirks = frozenset({"force_reasoning_effort_max"})
    monkeypatch.setattr(server_module, "resolve", lambda model: info if model == "kimi-k3" else None)

    async def fake_stream(endpoint, body, headers, **kwargs):
        assert endpoint == "http://up/chat/completions"
        assert body["reasoning_effort"] == "max"
        assert body["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
        assert body["stream"] is True
        return server_module.JSONResponse(status_code=200, content={"stream_path": True})

    monkeypatch.setattr(server_module, "_passthrough_stream", fake_stream)
    response = client.post("/v1/chat/completions", json={
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "use the tool"}],
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
    })
    assert response.status_code == 200
    assert response.json() == {"stream_path": True}


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


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_adaptive_anthropic_chat_preserves_canonical_effort(effort):
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
        "reasoning_effort": effort,
    })

    assert _apply_gateway_reasoning(body, info, target_api="messages") is True
    server_module._normalize_anthropic_adaptive_thinking(body, info)

    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": effort}


@pytest.mark.parametrize("client_cap, expected", [(None, 128000), ({"max_tokens": 2048}, 2048)])
def test_anthropic_chat_without_client_cap_uses_model_output_limit(monkeypatch, client_cap, expected):
    info = _info(
        "anthropic",
        provider="anthropic",
        provider_model_id="claude-opus-5-5",
        max_output_tokens=128000,
    )
    captured = {}

    async def fake_stream(endpoint, body, model, headers, **kwargs):
        captured.update(body)
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, "_chat_anthropic_stream", fake_stream)
    monkeypatch.setattr(server_module, "_forward_headers", lambda *args, **kwargs: {})
    body = {
        "model": "claude-opus-5-5",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": True,
        "reasoning_effort": "high",
        **(client_cap or {}),
    }
    request = SimpleNamespace(state=SimpleNamespace())

    asyncio.run(server_module._handle_chat_anthropic(body, info, "claude-opus-5.5", request, True))

    assert captured["max_tokens"] == expected


@pytest.mark.parametrize("via_chat_body", [False, True])
@pytest.mark.parametrize("client_cap, expected", [(None, 128000), ({"max_output_tokens": 2048}, 2048)])
def test_anthropic_responses_without_client_cap_uses_model_output_limit(
    monkeypatch, client_cap, expected, via_chat_body,
):
    info = _info(
        "anthropic",
        provider="anthropic",
        provider_model_id="claude-opus-5-5",
        max_output_tokens=128000,
    )
    captured = {}

    async def fake_stream(endpoint, messages_req, model, headers, **kwargs):
        captured.update(messages_req)
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, "_handle_responses_anthropic_stream", fake_stream)
    monkeypatch.setattr(server_module, "_forward_headers", lambda *args, **kwargs: {})
    body = {"model": "claude-opus-5-5", "input": "ping", "stream": True, **(client_cap or {})}
    # Vision extraction hands over an image-free Chat request; its cap mirrors the Responses one.
    chat_body = {
        "model": "claude-opus-5-5",
        "messages": [{"role": "user", "content": "ping"}],
        **({"max_tokens": client_cap["max_output_tokens"]} if client_cap else {}),
    } if via_chat_body else None
    request = SimpleNamespace(state=SimpleNamespace())

    asyncio.run(server_module._handle_responses_anthropic(
        body, info, "claude-opus-5.5", request, chat_body=chat_body,
    ))

    assert captured["max_tokens"] == expected


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

    tool_req = openai_chat_to_anthropic({
        "messages": [{
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "screenshot"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }],
    })
    assert tool_req["messages"] == [{
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": [
                {"type": "text", "text": "screenshot"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            ],
        }],
    }]

    url_tool_req = openai_chat_to_anthropic({
        "messages": [{
            "role": "tool",
            "tool_call_id": "call_url",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "https://example.test/screenshot.png"},
            }],
        }],
    })
    assert url_tool_req["messages"][0]["content"][0]["content"] == [{
        "type": "image",
        "source": {"type": "url", "url": "https://example.test/screenshot.png"},
    }]

    json_tool_req = openai_chat_to_anthropic({
        "messages": [{"role": "tool", "tool_call_id": "call_2", "content": {"ok": True}}],
    })
    assert json_tool_req["messages"][0]["content"][0]["content"] == '{"ok": true}'

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
@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/chat/completions", {
            "model": "anthropic-text",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "first turn"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ]},
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": "follow-up without a new image"},
            ],
        }),
        ("/v1/responses", {
            "model": "anthropic-text",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "first turn"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ]},
                {"type": "message", "role": "assistant", "content": "prior answer"},
                {"type": "message", "role": "user", "content": "follow-up without a new image"},
            ],
        }),
        ("/v1/messages", {
            "model": "anthropic-text",
            "max_tokens": 128,
            "system": [{
                "type": "text", "text": "system",
                "cache_control": {"type": "ephemeral"},
            }],
            "tools": [{
                "name": "lookup", "description": "lookup",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"},
            }],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "metadata": {"user_id": "vision-regression"},
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "first turn"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": "AAAA",
                    }, "cache_control": {"type": "ephemeral"}},
                ]},
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": "follow-up without a new image"},
            ],
        }),
    ],
)
def test_text_only_anthropic_model_gets_image_notes_across_endpoints(
    client, monkeypatch, path, payload,
):
    text_info = _info(
        "anthropic", thinking="", provider="anthropic",
        provider_model_id="claude-text-upstream",
    )
    text_info.protocol = "anthropic"
    captured = {}

    async def fake_chat_anthropic(body, info, model, request, is_stream):
        captured.update(model=model, upstream_chat=body)
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    async def fake_responses_anthropic(body, info, model, request, *, chat_body=None):
        captured.update(model=model, upstream_chat=chat_body)
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    async def fake_messages_sync(endpoint, body, headers, **kwargs):
        captured.update(
            model="anthropic-text",
            upstream_chat=anthropic_to_openai(body),
            native_anthropic=copy.deepcopy(body),
        )
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, "resolve", lambda model: text_info)
    monkeypatch.setattr(server_module, "_handle_chat_anthropic", fake_chat_anthropic)
    monkeypatch.setattr(server_module, "_handle_responses_anthropic", fake_responses_anthropic)
    monkeypatch.setattr(server_module, "_passthrough_anthropic_sync", fake_messages_sync)

    response = client.post(path, json=payload)

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert captured["model"] == "anthropic-text"
    assert captured["upstream_chat"] is not None
    assert not server_module._payload_has_image(captured["upstream_chat"])
    assert server_module.IMAGE_OMITTED_NOTE in str(captured["upstream_chat"]["messages"])
    if path == "/v1/messages":
        native = captured["native_anthropic"]
        assert native["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert native["tools"][0]["cache_control"] == {"type": "ephemeral"}
        assert native["tool_choice"]["disable_parallel_tool_use"] is True
        assert native["metadata"] == {"user_id": "vision-regression"}
        replacement = native["messages"][0]["content"][1]
        assert replacement == {
            "type": "text", "text": server_module.IMAGE_OMITTED_NOTE,
            "cache_control": {"type": "ephemeral"},
        }


def test_every_chat_image_shape_is_replaced_with_the_note():
    note = {"type": "text", "text": server_module.IMAGE_OMITTED_NOTE}
    body = {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "keep"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "input_image", "image": "data:image/png;base64,AA=="},
            {"type": "unsupported_input_image_file", "file_id": "file_1"},
        ]},
        {"role": "tool", "tool_call_id": "t", "content": [{"type": "image", "source": {}}]},
    ]}
    assert server_module._image_count(body) == 4
    assert server_module._omit_chat_images(body) == 4
    assert body["messages"][0]["content"] == [{"type": "text", "text": "keep"}, note, note, note]
    assert body["messages"][1]["content"] == [note]


def test_every_responses_image_shape_is_replaced_with_the_note():
    note = {"type": "input_text", "text": server_module.IMAGE_OMITTED_NOTE}
    pdf = {"type": "input_file", "file_data": "data:application/pdf;base64,AA==", "filename": "a.pdf"}
    body = {"input": [
        {"role": "user", "content": [
            {"type": "input_text", "text": "keep"},
            {"type": "input_image", "file_id": "file_1"},
            {"type": "input_file", "file_data": "data:image/png;base64,AA==", "filename": "a.png"},
            pdf,
        ]},
        {"type": "custom_tool_call_output", "call_id": "c", "output": [
            {"type": "image_url", "image_url": "data:image/png;base64,AA=="},
        ]},
    ]}
    server_module._omit_responses_images(body)
    assert body["input"][0]["content"] == [{"type": "input_text", "text": "keep"}, note, note, pdf]
    assert body["input"][1]["output"] == [note]


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_messages_to_a_chat_model_keep_its_system_instruction(client, monkeypatch):
    info = _info("none", thinking="", provider="test", provider_model_id="text-upstream")
    info.system_instruction = "Answer briefly."
    monkeypatch.setattr(server_module, "resolve", lambda model: info)
    captured = {}

    async def fake_sync(endpoint, body, *args, **kwargs):
        captured["body"] = body
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, "_handle_sync", fake_sync)
    response = client.post("/v1/messages", json={
        "model": "text-model", "max_tokens": 32, "system": "Be kind.",
        "messages": [{"role": "user", "content": "hi"}],
    })

    assert response.status_code == 200
    system = captured["body"]["messages"][0]
    assert system["role"] == "system"
    assert "Answer briefly." in system["content"] and "Be kind." in system["content"]


def test_removed_vision_fallback_settings_only_warn(monkeypatch, caplog):
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK_CLOUD", "cloud-vision")
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK_LOCAL", "")

    with caplog.at_level(logging.WARNING, logger="model-gateway"):
        server_module._warn_removed_vision_fallback_env()

    assert "ignoring GATEWAY_VISION_FALLBACK_CLOUD:" in caplog.text
    assert "GATEWAY_VISION_FALLBACK_LOCAL" not in caplog.text


def test_historical_image_remains_visible_to_request_local_routing():
    body = {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "first turn"},
            {"type": "image_url", "image_url": {"url": "one"}},
        ]},
        {"role": "assistant", "content": "prior answer"},
        {"role": "user", "content": "follow-up without a new image"},
    ]}

    assert server_module._image_count(body) == 1


def test_request_json_reader_enforces_streaming_body_limit(monkeypatch):
    class FakeRequest:
        headers = {}

        async def stream(self):
            yield b'{"model":'
            yield b'"too-large"}'

    monkeypatch.setattr(server_module, "MODEL_GATEWAY_MAX_REQUEST_BODY_BYTES", 10)
    with pytest.raises(server_module._RequestBodyTooLarge):
        asyncio.run(server_module._read_json_body_bounded(FakeRequest()))


def test_responses_input_image_translates_to_real_image_block():
    from src.responses import responses_to_chat
    result = responses_to_chat({"input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "inspect"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "high"},
    ]}]})
    assert result["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"},
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
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "low"},
        ],
    }]})
    tool = result["messages"][0]
    assert tool["role"] == "tool"
    assert tool["content"][0] == {"type": "text", "text": "screenshot"}
    assert tool["content"][1] == {"type": "text", "text": '{"page": 2}'}
    assert tool["content"][2] == {"type": "text", "text": "[file: report.pdf]"}
    assert tool["content"][3]["image_url"] == {
        "url": "data:image/png;base64,AAAA",
        "detail": "low",
    }
    assert server_module._image_count(result) == 1


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_responses_file_id_rejected_on_translated_route(client, monkeypatch):
    monkeypatch.setattr(server_module, "resolve", lambda model: _info("none", provider="test", vision=True))
    response = client.post("/v1/responses", json={
        "model": "text-model", "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_image", "file_id": "file_123"},
        ]}],
    })
    assert response.status_code == 400
    assert "file_id" in response.json()["error"]["message"]


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_responses_native_openai_preserves_file_id(client, monkeypatch):
    native = _info("none", provider="openai", provider_model_id="native-responses", vision=True)
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
@pytest.mark.parametrize("path,payload,handler", [
    ("/v1/chat/completions", {
        "model": "text-model", "gateway_image_handling": "reroute", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
    }, "_passthrough_sync"),
    ("/v1/responses", {
        "model": "text-model", "stream": True, "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "what is this?"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]}],
    }, "_handle_responses_stream"),
    ("/v1/messages", {
        "model": "text-model", "max_tokens": 32, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        ]}],
    }, "_handle_sync"),
])
def test_text_only_model_gets_a_note_in_place_of_each_image(client, monkeypatch, path, payload, handler):
    text_info = _info("none", thinking="", provider="test", provider_model_id="text-upstream")
    monkeypatch.setattr(server_module, "resolve", lambda model: text_info)
    captured = {}

    async def fake_upstream(endpoint, body, *args, **kwargs):
        captured["body"] = body
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, handler, fake_upstream)
    response = client.post(path, json=payload)

    assert response.status_code == 200
    body = captured["body"]
    assert body["model"] == "text-upstream"
    assert not server_module._payload_has_image(body)
    assert "gateway_image_handling" not in body
    parts = [part for message in body["messages"] if isinstance(message.get("content"), list)
             for part in message["content"]]
    assert {"type": "text", "text": server_module.IMAGE_OMITTED_NOTE} in parts
    assert "what is this?" in str(parts)


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "gpt-5.6"])
def test_configured_gpt_vision_models_receive_images(client, monkeypatch, model):
    native = _info("openai-responses", provider="databricks", provider_model_id=f"upstream-{model}", vision=True)
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


def test_chat_native_vision_model_receives_images(client, monkeypatch):
    native = _info("none", thinking="", provider="test", provider_model_id="native-vision", vision=True)
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
            thinking_levels=tuple(entry.get("thinking_levels", ())),
            thinking_format=entry.get("thinking_format", ""),
            max_output_tokens=entry.get("max_output_tokens", 32768) or 32768,
        )
        fmt = _infer_thinking_format(info, target_api="chat")
        seen_formats.add(fmt)
        # The strongest model-specific effort probe used by debug must not raise.
        levels = tuple(entry.get("thinking_levels", ()))
        enabled_levels = [level for level in levels if level != "off"]
        if enabled_levels:
            req = {"messages": [], "reasoning_effort": enabled_levels[-1]}
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
    monkeypatch.setattr(providers, "_models", {
        "test-thinking": {
            "name": "test-thinking",
            "provider": "openrouter",
            "provider_model_id": "test/thinking",
            "thinking": "optional",
            "thinking_format": "openrouter",
        },
    })
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
    thinking_model = next(m for m in data if m["id"] == "test-thinking")
    assert thinking_model["thinking_levels"] == ["off", "minimal", "low", "medium", "high", "xhigh", "max"]


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_debug_thinking_endpoint_matrix(client):
    resp = client.get("/v1/debug/thinking")
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["models"] == len(body["models"])
    assert body["summary"]["max_reachable"] + body["summary"]["max_unreachable"] \
        == body["summary"]["models"]
    assert "by_format" in body["summary"]
    assert "gateway_levels" not in body
    assert all("thinking_levels" in row for row in body["models"])


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
@pytest.mark.parametrize("profile_name,provider,base_url", [
    ("moonshot-kimi-k3.1-pro", "moonshot", "https://api.moonshot.ai/v1"),
    ("fireworks-kimi-k3", "fireworks", "https://api.fireworks.ai/inference/v1"),
])
def test_kimi_profiles_advertise_their_levels_and_max_default(client, monkeypatch, profile_name, provider, base_url):
    from src.onboarding import load_profile

    profile_path = Path(__file__).resolve().parents[1] / "config" / "onboarding" / f"{profile_name}.yaml"
    entry = load_profile(profile_path)["models"][0]
    monkeypatch.setattr(providers, "_config", {
        "providers": {provider: {"base_url": base_url, "api_key": "test-key"}},
    })
    monkeypatch.setattr(
        providers,
        "_models",
        {model_id: entry for model_id in providers._entry_routable_ids(entry)},
    )

    models_response = client.get("/v1/models")
    assert models_response.status_code == 200
    advertised = next(row for row in models_response.json()["data"] if row["id"] == entry["name"])
    assert advertised["thinking_levels"] == ["low", "high", "max"]
    assert advertised["default_enabled_level"] == "max"
    assert advertised["forwarded_params"] == ["reasoning_effort"]


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_glm53_profile_observability_matches_actual_dispatch(client, monkeypatch):
    from src.onboarding import load_profile

    profile_path = Path(__file__).resolve().parents[1] / "config" / "onboarding" / "fireworks-glm-5.3.yaml"
    entry = load_profile(profile_path)["models"][0]
    monkeypatch.setattr(providers, "_config", {
        "providers": {
            "fireworks": {
                "base_url": "https://api.fireworks.ai/inference/v1",
                "api_key": "test-key",
            },
        },
    })
    monkeypatch.setattr(
        providers,
        "_models",
        {model_id: entry for model_id in providers._entry_routable_ids(entry)},
    )

    models_response = client.get("/v1/models")
    assert models_response.status_code == 200
    advertised = next(
        row for row in models_response.json()["data"]
        if row["id"] == "glm-5.3-fw"
    )
    assert advertised["default_enabled_level"] == "high"
    assert advertised["forwarded_params"] == ["reasoning_effort"]
    assert advertised["max_reachable"] is True

    debug_response = client.get("/v1/debug/thinking")
    assert debug_response.status_code == 200
    debug = next(
        row for row in debug_response.json()["models"]
        if row["name"] == "glm-5.3-fw"
    )
    assert debug["default_enabled_level"] == "high"
    assert debug["forwarded_params"] == ["reasoning_effort"]
    assert debug["max_reachable"] is True

    async def fake_passthrough(endpoint, body, headers, **kwargs):
        assert endpoint == "https://api.fireworks.ai/inference/v1/chat/completions"
        assert body["model"] == "accounts/fireworks/models/glm-5p3"
        assert "thinking" not in body
        assert body["reasoning_effort"] == "low"
        return server_module.JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough)
    routed = client.post("/v1/chat/completions", json={
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "minimal",
    })
    assert routed.status_code == 200
    assert routed.json() == {"ok": True}


@pytest.mark.skipif(TestClient is None, reason="fastapi not installed")
def test_debug_thinking_zai_max_reachable(client, monkeypatch):
    """The generic Z.ai branch forwards reasoning effort alongside thinking."""
    monkeypatch.setattr(providers, "_models", {
        "zai-always-test": {
            "name": "zai-always-test",
            "provider": "zai_coding",
            "provider_model_id": "zai-upstream-test",
            "thinking": "always",
            "thinking_format": "zai",
        },
    })
    resp = client.get("/v1/debug/thinking")
    assert resp.status_code == 200
    rows = {r["name"]: r for r in resp.json()["models"]}
    assert "zai-always-test" in rows
    zai = rows["zai-always-test"]
    assert zai["thinking_format"] == "zai"
    assert zai["forwarded_params"] == ["enable_thinking", "reasoning_effort"]
    assert zai["max_reachable"] is True


