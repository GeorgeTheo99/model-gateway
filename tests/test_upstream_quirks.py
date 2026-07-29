"""Tests for the config-driven robustness/quirk layer.

Covers: provider quirks parsing, endpoint_style invocations, path_prefixes,
stream_options quirk handling, upstream error unwrapping, context-overflow
detection, the per-provider circuit breaker, the config.yaml models: overlay
(with per-model protocol override), and config-driven model fallbacks.

No workspace-specific values appear here — providers/models use placeholder
hosts and ids, exactly as real deployments configure them via config.yaml.
"""

import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

import src.circuit as circuit
import src.model_fallback as model_fallback
import src.providers as providers
from src import admin, config_io, errors
from src.config_lock import config_write_lock
from src.server import _apply_openai_request_quirks, _maybe_stream_options, _upstream_endpoint


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Isolate config.yaml + model-info.json to temp files (no ambient env)."""
    for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_SERVING_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("providers: {}\n")
    mi = tmp_path / "model-info.json"
    mi.write_text(json.dumps({"llm": []}))
    monkeypatch.setattr(providers, "CONFIG_PATH", cfg)
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", mi)
    monkeypatch.setattr(providers, "MODEL_INFO_SOURCE_PATH", None)
    providers.reload()
    yield tmp_path
    providers.reload()


def _write_config(tmp_path, text: str) -> None:
    (tmp_path / "config.yaml").write_text(text)
    providers.reload()


def _write_models(tmp_path, entries: list) -> None:
    (tmp_path / "model-info.json").write_text(json.dumps({"llm": entries}))
    providers.reload()


# ── (a) quirks parsing ───────────────────────────────────────────────────────


def test_quirks_parse_from_provider_config(tmp_config):
    _write_config(tmp_config, """
providers:
  my_workspace:
    base_url: https://gateway.example.com
    api_key: placeholder
    quirks:
      - no_stream_options
      - no_reasoning_params
""")
    _write_models(tmp_config, [
        {"name": "m1", "provider": "my_workspace", "provider_model_id": "m1-upstream"},
    ])
    info = providers.resolve("m1")
    assert info is not None
    assert info.quirks == frozenset({"no_stream_options", "no_reasoning_params"})


def test_quirks_default_empty(tmp_config):
    _write_config(tmp_config, """
providers:
  plain:
    base_url: https://api.example.com/v1
    api_key: placeholder
""")
    _write_models(tmp_config, [
        {"name": "m2", "provider": "plain", "provider_model_id": "m2-up"},
    ])
    info = providers.resolve("m2")
    assert info.quirks == frozenset()


# ── (b) endpoint_style: invocations ──────────────────────────────────────────


def test_endpoint_style_invocations_builds_full_url(tmp_config):
    _write_config(tmp_config, """
providers:
  my_workspace:
    base_url: https://workspace.example.com
    api_key: placeholder
    endpoint_style: invocations
""")
    _write_models(tmp_config, [
        {"name": "serving-model", "provider": "my_workspace", "provider_model_id": "my-endpoint"},
    ])
    info = providers.resolve("serving-model")
    assert info.base_url == "https://workspace.example.com/serving-endpoints/my-endpoint/invocations"
    assert info.endpoint_suffix == ""
    # _upstream_endpoint must not append the default suffix.
    assert _upstream_endpoint(info, "/chat/completions") == info.base_url
    assert _upstream_endpoint(info, "/messages") == info.base_url


def test_upstream_endpoint_default_suffix(tmp_config):
    info = SimpleNamespace(base_url="https://api.example.com/v1", endpoint_suffix=None)
    assert _upstream_endpoint(info, "/chat/completions") == "https://api.example.com/v1/chat/completions"
    assert _upstream_endpoint(info, "/messages") == "https://api.example.com/v1/messages"


# ── (c) path_prefixes per-protocol routing ───────────────────────────────────


def test_path_prefixes_route_by_protocol(tmp_config):
    _write_config(tmp_config, """
providers:
  my_gateway:
    base_url: https://gateway.example.com
    api_key: placeholder
    protocol: openai
    path_prefixes:
      anthropic: /anthropic/v1
      openai: /openai/v1
""")
    _write_models(tmp_config, [
        {"name": "oai-model", "provider": "my_gateway", "provider_model_id": "oai-up"},
        {"name": "anth-model", "provider": "my_gateway", "provider_model_id": "anth-up", "protocol": "anthropic"},
    ])
    oai = providers.resolve("oai-model")
    anth = providers.resolve("anth-model")
    assert oai.base_url == "https://gateway.example.com/openai/v1"
    assert oai.protocol == "openai"
    assert anth.base_url == "https://gateway.example.com/anthropic/v1"
    assert anth.protocol == "anthropic"


# ── (d) _maybe_stream_options ────────────────────────────────────────────────


def test_maybe_stream_options_pops_for_quirk_provider():
    info = SimpleNamespace(quirks=frozenset({"no_stream_options"}))
    req = {"stream": True, "stream_options": {"include_usage": True}}
    _maybe_stream_options(req, info)
    assert "stream_options" not in req


def test_maybe_stream_options_sets_include_usage_otherwise():
    info = SimpleNamespace(quirks=frozenset())
    req = {"stream": True}
    _maybe_stream_options(req, info)
    assert req["stream_options"] == {"include_usage": True}


def test_api_key_file_resolves_relative_to_config(tmp_config):
    secret = tmp_config / "secrets" / "provider.key"
    secret.parent.mkdir()
    secret.write_text("file-secret\n")
    secret.chmod(0o600)
    _write_config(tmp_config, """
providers:
  file_auth:
    base_url: https://api.example.com/v1
    api_key_file: secrets/provider.key
""")
    _write_models(tmp_config, [
        {"name": "file-model", "provider": "file_auth", "provider_model_id": "file-up"},
    ])
    info = providers.resolve("file-model")
    assert info is not None
    assert info.api_key == "file-secret"


def test_api_key_file_rejects_group_or_world_permissions(tmp_config):
    secret = tmp_config / "unsafe.key"
    secret.write_text("secret\n")
    secret.chmod(0o644)
    _write_config(tmp_config, f"""
providers:
  file_auth:
    base_url: https://api.example.com/v1
    api_key_file: {secret}
""")
    _write_models(tmp_config, [
        {"name": "file-model", "provider": "file_auth", "provider_model_id": "file-up"},
    ])
    assert providers.resolve("file-model") is None
    assert providers.model_availability("file-model")["reason"] == "provider_not_configured"


def test_openai_request_quirks_normalize_new_provider_shape():
    info = SimpleNamespace(quirks=frozenset({
        "force_reasoning_effort_max",
        "use_max_completion_tokens",
        "drop_fixed_sampling_fields",
    }))
    req = {
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning": {"effort": "low"},
        "max_tokens": 42,
        "temperature": 0.2,
        "top_p": 0.8,
        "n": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    assert _apply_openai_request_quirks(req, info) is None
    assert req["reasoning_effort"] == "max"
    assert req["max_completion_tokens"] == 42
    assert "max_tokens" not in req
    for field in ("temperature", "top_p", "n", "presence_penalty", "frequency_penalty"):
        assert field not in req
    assert req["tools"][0]["function"]["name"] == "lookup"


def test_named_tool_choice_as_required_preserves_selected_function():
    info = SimpleNamespace(quirks=frozenset({"named_tool_choice_as_required"}))
    req = {
        "tools": [
            {"type": "function", "function": {"name": "lookup"}},
            {"type": "function", "function": {"name": "run_python"}},
        ],
        "tool_choice": {"type": "function", "function": {"name": "run_python"}},
    }

    assert _apply_openai_request_quirks(req, info) is None
    assert req["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in req["tools"]] == ["run_python"]


@pytest.mark.parametrize(
    ("tools", "function_name"),
    [
        ([{"type": "function", "function": {"name": "lookup"}}], "run_python"),
        (42, "run_python"),
        ([], ""),
        (
            [
                {"type": "function", "function": {"name": "run_python"}},
                {"type": "function", "function": {"name": "run_python"}},
            ],
            "run_python",
        ),
    ],
)
def test_named_tool_choice_as_required_rejects_invalid_selection(tools, function_name):
    info = SimpleNamespace(quirks=frozenset({"named_tool_choice_as_required"}))
    req = {
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": function_name}},
    }

    error = _apply_openai_request_quirks(req, info)

    assert error == "The requested named function must appear exactly once in tools."


def test_inline_image_only_quirk_rejects_public_url_and_accepts_data_url():
    info = SimpleNamespace(quirks=frozenset({"inline_image_urls_only"}))
    public = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]}]}
    assert "inline data:image" in _apply_openai_request_quirks(public, info)
    for invalid_url in ("file:///tmp/a.png", "ftp://example.com/a.png", "data:text/plain;base64,QQ==", "data:image/png,raw", "data:image/png;base64,not@@base64"):
        invalid = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": invalid_url}},
        ]}]}
        assert "inline data:image" in _apply_openai_request_quirks(invalid, info)
    inline = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]}]}
    assert _apply_openai_request_quirks(inline, info) is None


def test_anthropic_tool_result_blocks_rewrites_only_tool_images():
    info = SimpleNamespace(quirks=frozenset({"anthropic_tool_result_blocks"}))
    user_image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,USER"}}
    req = {"messages": [
        {"role": "user", "content": [user_image]},
        {"role": "tool", "tool_call_id": "call_1", "content": [
            {"type": "text", "text": "inline screenshot"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "image_url", "image_url": {"url": "https://example.test/screenshot.png"}},
        ]},
    ]}

    assert _apply_openai_request_quirks(req, info) is None

    assert req["messages"][0]["content"][0] == user_image
    assert req["messages"][1]["content"] == [
        {"type": "text", "text": "inline screenshot"},
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "AAAA",
        }},
        {"type": "image", "source": {
            "type": "url", "url": "https://example.test/screenshot.png",
        }},
    ]


def test_per_model_quirks_merge_with_provider_quirks(tmp_config):
    _write_config(tmp_config, """
providers:
  shared_gw:
    base_url: https://gateway.example.com
    api_key: placeholder
    quirks:
      - no_stream_options
""")
    _write_models(tmp_config, [
        {"name": "plain", "provider": "shared_gw", "provider_model_id": "plain-up"},
        {"name": "picky", "provider": "shared_gw", "provider_model_id": "picky-up",
         "quirks": ["reasoning_none_with_tools"]},
    ])
    plain = providers.resolve("plain")
    picky = providers.resolve("picky")
    # Provider quirk applies to both; per-model quirk only to the model that declares it.
    assert plain.quirks == frozenset({"no_stream_options"})
    assert picky.quirks == frozenset({"no_stream_options", "reasoning_none_with_tools"})


def test_reasoning_none_with_tools_forces_none_when_tools_present():
    from src.server import _apply_gateway_reasoning
    info = SimpleNamespace(
        provider="shared_gw", provider_model_id="gpt-5-6-sol",
        thinking="optional", thinking_format="", max_output_tokens=32768,
        quirks=frozenset({"reasoning_none_with_tools"}),
    )
    req = {
        "messages": [],
        "reasoning_effort": "high",
        "tools": [{"type": "function", "function": {"name": "t"}}],
    }
    enabled = _apply_gateway_reasoning(req, info, target_api="chat")
    assert enabled is False
    assert req["reasoning_effort"] == "none"
    assert "thinking" not in req and "reasoning" not in req


def test_reasoning_none_with_tools_leaves_reasoning_when_no_tools():
    from src.server import _apply_gateway_reasoning
    info = SimpleNamespace(
        provider="shared_gw", provider_model_id="gpt-5-6-sol",
        thinking="optional", thinking_format="", max_output_tokens=32768,
        quirks=frozenset({"reasoning_none_with_tools"}),
    )
    req = {"messages": [], "reasoning_effort": "high"}
    enabled = _apply_gateway_reasoning(req, info, target_api="chat")
    # No tools -> quirk does not fire; normal reasoning path keeps effort active.
    assert enabled is True
    assert req.get("reasoning_effort") == "high"


def test_no_reasoning_params_quirk_strips_reasoning_controls():
    from src.server import _apply_gateway_reasoning
    info = SimpleNamespace(
        provider="my_workspace", provider_model_id="m-up",
        thinking="always", thinking_format="", max_output_tokens=32768,
        quirks=frozenset({"no_reasoning_params"}),
    )
    req = {"messages": [], "reasoning_effort": "high", "thinking": {"type": "enabled", "budget_tokens": 4096}}
    enabled = _apply_gateway_reasoning(req, info, target_api="chat")
    assert enabled is True  # thinking still reported enabled for translators
    for key in ("reasoning_effort", "thinking", "reasoning", "output_config"):
        assert key not in req


# ── (e) error unwrapping ─────────────────────────────────────────────────────


def test_unwrap_error_message_double_encoded():
    payload = {"error_code": "BAD_REQUEST", "message": "{\"message\":\"prompt is too long\"}"}
    assert errors._unwrap_error_message(payload) == "prompt is too long"


def test_unwrap_error_message_plain_string():
    assert errors._unwrap_error_message("simple failure") == "simple failure"


# ── (f) context overflow detection ───────────────────────────────────────────


def test_is_context_overflow_true_for_400_prompt_too_long():
    assert errors._is_context_overflow(400, "prompt is too long") is True


def test_is_context_overflow_false_for_500():
    assert errors._is_context_overflow(500, "prompt is too long") is False


# ── (f2) upstream error fidelity ─────────────────────────────────────────────────


def _mk_resp(status: int, json_body=None, text: str = "", headers: dict | None = None) -> httpx.Response:
    req = httpx.Request("POST", "https://upstream.example.com/v1/messages")
    if json_body is not None:
        return httpx.Response(status, json=json_body, headers=headers, request=req)
    return httpx.Response(status, text=text, headers=headers, request=req)


def test_upstream_error_preserves_anthropic_error_type_and_status():
    body = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
    resp = _mk_resp(401, json_body=body)
    out = errors.upstream_error(resp, json.dumps(body), "Anthropic")
    assert out.status_code == 401
    payload = json.loads(out.body)
    assert payload["error"]["type"] == "authentication_error"
    assert "invalid x-api-key" in payload["error"]["message"]


def test_upstream_error_preserves_permission_error():
    body = {"type": "error", "error": {"type": "permission_error", "message": "no access"}}
    resp = _mk_resp(403, json_body=body)
    out = errors.upstream_error(resp, json.dumps(body), "Anthropic")
    assert out.status_code == 403
    assert json.loads(out.body)["error"]["type"] == "permission_error"


def test_upstream_error_529_keeps_status_and_overloaded_type():
    body = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    resp = _mk_resp(529, json_body=body)
    out = errors.upstream_error(resp, json.dumps(body), "Anthropic")
    assert out.status_code == 529
    assert json.loads(out.body)["error"]["type"] == "overloaded_error"
    # Even without a parseable body, 529 maps to overloaded_error, not 502 api_error.
    resp = _mk_resp(529, text="upstream saturated")
    out = errors.upstream_error(resp, "upstream saturated", "Anthropic")
    assert out.status_code == 529
    assert json.loads(out.body)["error"]["type"] == "overloaded_error"


def test_upstream_error_429_retry_after_annotation_kept():
    resp = _mk_resp(429, json_body={"error": {"type": "rate_limit_error", "message": "slow down"}}, headers={"retry-after": "7"})
    out = errors.upstream_error(resp, "", "Anthropic")
    assert out.status_code == 429
    payload = json.loads(out.body)
    assert payload["error"]["type"] == "rate_limit_error"
    assert "(retry-after: 7)" in payload["error"]["message"]


def test_upstream_error_context_overflow_special_case_wins():
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long"}}
    resp = _mk_resp(400, json_body=body)
    out = errors.upstream_error(resp, json.dumps(body), "Anthropic")
    assert out.status_code == 400
    payload = json.loads(out.body)
    assert payload["error"]["type"] == "invalid_request_error"
    assert "context window" in payload["error"]["message"]


def test_upstream_error_unlabeled_4xx_still_invalid_request():
    resp = _mk_resp(422, text="plain failure")
    out = errors.upstream_error(resp, "plain failure", "Provider")
    assert out.status_code == 422
    assert json.loads(out.body)["error"]["type"] == "invalid_request_error"


def test_upstream_error_openai_preserves_type_and_code():
    body = {"error": {"type": "insufficient_quota", "code": "insufficient_quota", "message": "quota exceeded"}}
    resp = _mk_resp(429, json_body=body)
    out = errors.upstream_error_openai(resp, json.dumps(body), "OpenAI")
    assert out.status_code == 429
    payload = json.loads(out.body)
    assert payload["error"]["type"] == "insufficient_quota"
    assert payload["error"]["code"] == "insufficient_quota"


def test_upstream_error_openai_529_overloaded():
    resp = _mk_resp(529, text="overloaded")
    out = errors.upstream_error_openai(resp, "overloaded", "Provider")
    assert out.status_code == 529
    assert json.loads(out.body)["error"]["type"] == "overloaded_error"


# ── (g) circuit breaker ──────────────────────────────────────────────────────


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


def test_circuit_opens_after_three_502s_and_blocks(clean_circuit):
    provider = clean_circuit("test-circuit-provider")
    assert circuit.is_tripped(provider) is False
    for _ in range(3):
        circuit.record_failure(provider, 502, "bad gateway")
    assert circuit.is_tripped(provider) is True
    # New requests are blocked: the recovery event is cleared.
    assert circuit._get(provider).recovery_event.is_set() is False
    # A success closes the circuit and releases waiters.
    circuit.record_success(provider)
    assert circuit.is_tripped(provider) is False
    assert circuit._get(provider).recovery_event.is_set() is True


def test_circuit_two_failures_do_not_trip(clean_circuit):
    provider = clean_circuit("test-circuit-two")
    circuit.record_failure(provider, 503, "")
    circuit.record_failure(provider, 503, "")
    assert circuit.is_tripped(provider) is False


# ── (h) config models: overlay + per-model protocol override ────────────────


def test_config_models_overlay_adds_routable_model(tmp_config):
    _write_config(tmp_config, """
providers:
  my_workspace:
    base_url: https://gateway.example.com/v1
    api_key: placeholder
    protocol: openai
models:
  - name: overlay-model
    alias: om
    provider: my_workspace
    provider_model_id: overlay-upstream
    protocol: anthropic
    context: 200000
""")
    # Overlay model routes even though model-info.json is empty.
    for model_id in ("overlay-model", "om", "overlay-upstream"):
        info = providers.resolve(model_id)
        assert info is not None
        assert info.provider_model_id == "overlay-upstream"
        # Per-model protocol override wins over the provider default.
        assert info.protocol == "anthropic"
    assert info.context == 200000


# ── (i) model_fallbacks config ───────────────────────────────────────────────


def test_model_fallbacks_config_maps_after_saturation(tmp_config):
    _write_config(tmp_config, """
providers: {}
model_fallbacks:
  primary-endpoint: fallback-endpoint
""")
    decision = model_fallback.fallback_after_error("primary-endpoint", 502)
    assert decision is not None
    assert decision.fallback_model == "fallback-endpoint"
    # 404 model-not-found also maps.
    decision = model_fallback.fallback_after_error(
        "primary-endpoint", 404, '{"message": "model not found"}')
    assert decision is not None
    # Non-mapped models never fall back.
    assert model_fallback.fallback_after_error("other-endpoint", 502) is None
    # Non-retryable statuses never fall back.
    assert model_fallback.fallback_after_error("primary-endpoint", 400) is None


def test_model_fallbacks_absent_is_noop(tmp_config):
    _write_config(tmp_config, "providers: {}\n")
    assert model_fallback.fallback_after_error("anything", 503) is None


# ── (j) _persist_api_key block-bounded rewrite ──────────────────────────────


def test_persist_api_key_does_not_touch_next_provider(tmp_config):
    """A provider stanza without api_key must not clobber the NEXT provider's."""
    _write_config(tmp_config, """providers:
  first:
    base_url: https://a.example.com
    auth_refresh: databricks-cli
  second:
    base_url: https://b.example.com
    api_key: keep-me
""")
    providers._persist_api_key("first", "eyJnew")
    text = (tmp_config / "config.yaml").read_text()
    assert "keep-me" in text
    assert "eyJnew" not in text


def test_persist_api_key_rewrites_own_block_only(tmp_config):
    _write_config(tmp_config, """providers:
  first:
    base_url: https://a.example.com
    api_key: eyJold
  second:
    base_url: https://b.example.com
    api_key: keep-me
""")
    providers._persist_api_key("first", "eyJnew")
    text = (tmp_config / "config.yaml").read_text()
    assert "api_key: eyJnew" in text
    assert "eyJold" not in text
    assert "keep-me" in text


def test_persist_api_key_is_scoped_to_provider_section(tmp_config):
    _write_config(tmp_config, """metadata:
  providers:
    first:
      api_key: nested-secret
federation:
  peers:
    first:
      api_key: peer-secret
providers:
  first:
    base_url: https://a.example.com
    api_key: eyJold
""")

    assert providers._persist_api_key("first", "eyJnew") is True

    config = (tmp_config / "config.yaml").read_text()
    assert "api_key: nested-secret" in config
    assert "api_key: peer-secret" in config
    assert "api_key: eyJnew" in config
    assert "api_key: eyJold" not in config


def test_persist_api_key_preserves_secure_config_mode(tmp_config):
    _write_config(tmp_config, """providers:
  first:
    base_url: https://a.example.com
    api_key: eyJold
""")
    path = tmp_config / "config.yaml"
    path.chmod(0o600)

    assert providers._persist_api_key("first", "eyJnew") is True

    assert path.stat().st_mode & 0o777 == 0o600


def test_persist_api_key_waits_for_shared_config_transaction(tmp_config):
    _write_config(tmp_config, """providers:
  first:
    base_url: https://a.example.com
    api_key: eyJold
""")
    started = threading.Event()
    finished = threading.Event()
    errors = []

    def persist():
        started.set()
        try:
            providers._persist_api_key("first", "eyJnew")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            finished.set()

    with config_write_lock(providers.CONFIG_PATH):
        thread = threading.Thread(target=persist)
        thread.start()
        assert started.wait(timeout=2)
        assert not finished.wait(timeout=0.05)

    thread.join(timeout=2)
    assert not thread.is_alive()
    assert errors == []
    assert "api_key: eyJnew" in (tmp_config / "config.yaml").read_text()
    assert providers._load_config()["providers"]["first"]["api_key"] == "eyJnew"


def test_refresh_oauth_token_runs_browser_sso_when_cli_cache_is_broken(tmp_config, monkeypatch):
    _write_config(tmp_config, """providers:
  ws:
    base_url: https://workspace.example.com/serving-endpoints
    api_key: eyJold
    auth_refresh: databricks-cli
    auth_profile: ws-profile
""")
    providers._last_token_refresh_attempt.clear()
    providers._last_auth_login_attempt.clear()
    monkeypatch.setattr(providers, "_databricks_cli", lambda: "databricks")
    calls = []

    class Proc:
        def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b""):
            self.returncode = returncode
            self._stdout = stdout
            self._stderr = stderr

        async def communicate(self):
            return self._stdout, self._stderr

    async def fake_exec(*args, **kwargs):
        calls.append(args)
        if args[:3] == ("databricks", "auth", "token"):
            if len([c for c in calls if c[:3] == ("databricks", "auth", "token")]) == 1:
                return Proc(1, stderr=b"OAuth is not configured for this host")
            return Proc(0, stdout=b'{"access_token":"eyJnew"}')
        if args[:3] == ("databricks", "auth", "login"):
            return Proc(0)
        raise AssertionError(f"unexpected subprocess args: {args}")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    token = asyncio.run(providers.refresh_oauth_token("ws", force=True))

    assert token == "eyJnew"
    assert calls == [
        ("databricks", "auth", "token", "--profile", "ws-profile"),
        ("databricks", "auth", "login", "--host", "https://workspace.example.com", "--profile", "ws-profile"),
        ("databricks", "auth", "token", "--profile", "ws-profile"),
    ]
    text = (tmp_config / "config.yaml").read_text()
    assert "api_key: eyJnew" in text


def test_refresh_oauth_token_uses_workspace_url_for_ai_gateway_login(tmp_config, monkeypatch):
    _write_config(tmp_config, """providers:
  ws:
    base_url: https://12345.ai-gateway.cloud.databricks.com
    workspace_url: https://workspace.example.com
    api_key: eyJold
    auth_refresh: databricks-cli
    auth_profile: ws-profile
""")
    providers._last_token_refresh_attempt.clear()
    providers._last_auth_login_attempt.clear()
    monkeypatch.setattr(providers, "_databricks_cli", lambda: "databricks")
    calls = []

    class Proc:
        def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b""):
            self.returncode = returncode
            self._stdout = stdout
            self._stderr = stderr

        async def communicate(self):
            return self._stdout, self._stderr

    async def fake_exec(*args, **kwargs):
        calls.append(args)
        if args[:3] == ("databricks", "auth", "token"):
            token_calls = [c for c in calls if c[:3] == ("databricks", "auth", "token")]
            if len(token_calls) == 1:
                return Proc(1, stderr=b"OAuth is not configured for this host")
            return Proc(0, stdout=b'{"access_token":"eyJnew"}')
        if args[:3] == ("databricks", "auth", "login"):
            return Proc(0)
        raise AssertionError(f"unexpected subprocess args: {args}")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    token = asyncio.run(providers.refresh_oauth_token("ws", force=True))

    assert token == "eyJnew"
    assert calls[1] == (
        "databricks",
        "auth",
        "login",
        "--host",
        "https://workspace.example.com",
        "--profile",
        "ws-profile",
    )


def test_stale_oauth_refresh_cannot_overwrite_admin_provider_update(
    tmp_config, monkeypatch,
):
    _write_config(tmp_config, """providers:
  ws:
    base_url: https://old.example.com/serving-endpoints
    api_key: eyJold
    auth_refresh: databricks-cli
""")
    monkeypatch.setattr(config_io, "CONFIG_PATH", tmp_config / "config.yaml")
    monkeypatch.setattr(config_io, "MODEL_INFO_PATH", tmp_config / "model-info.json")
    monkeypatch.setattr(config_io, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.setattr(config_io, "log_dir", tmp_config / "logs")
    (tmp_config / "model-info.json").write_text(json.dumps({
        "llm": [{"name": "ws-model", "provider": "ws", "provider_model_id": "ws-model"}],
    }))
    providers.reload()
    monkeypatch.delenv("GATEWAY_VISION_FALLBACK", raising=False)
    providers._last_token_refresh_attempt.clear()
    providers._last_auth_login_attempt.clear()
    monkeypatch.setattr(providers, "_databricks_cli", lambda: "databricks")
    cli_started = asyncio.Event()
    release_cli = asyncio.Event()

    class Proc:
        returncode = 0

        async def communicate(self):
            cli_started.set()
            await release_cli.wait()
            return b'{"access_token":"eyJstale"}', b""

    async def fake_exec(*args, **kwargs):
        return Proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def run():
        refresh = asyncio.create_task(providers.refresh_oauth_token("ws", force=True))
        await asyncio.wait_for(cli_started.wait(), timeout=2)
        result, reload_error = await asyncio.to_thread(
            admin._apply_registry_mutation,
            lambda: config_io.upsert_provider(
                "ws",
                base_url="https://new.example.com/serving-endpoints",
                api_key="eyJadmin",
            ),
        )
        assert result is not None
        assert reload_error is None
        release_cli.set()
        return await asyncio.wait_for(refresh, timeout=2)

    token = asyncio.run(run())

    assert token == "eyJadmin"
    config = config_io.load_config_full()["providers"]["ws"]
    assert config["api_key"] == "eyJadmin"
    assert config["base_url"] == "https://new.example.com/serving-endpoints"
    assert "eyJstale" not in (tmp_config / "config.yaml").read_text()


def test_refresh_oauth_token_respects_auth_login_false(tmp_config, monkeypatch):
    _write_config(tmp_config, """providers:
  ws:
    base_url: https://workspace.example.com
    api_key: eyJold
    auth_refresh: databricks-cli
    auth_profile: ws-profile
    auth_login: false
""")
    providers._last_token_refresh_attempt.clear()
    providers._last_auth_login_attempt.clear()
    monkeypatch.setattr(providers, "_databricks_cli", lambda: "databricks")
    calls = []

    class Proc:
        returncode = 1

        async def communicate(self):
            return b"", b"OAuth is not configured for this host"

    async def fake_exec(*args, **kwargs):
        calls.append(args)
        return Proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    token = asyncio.run(providers.refresh_oauth_token("ws", force=True))

    assert token is None
    assert calls == [("databricks", "auth", "token", "--profile", "ws-profile")]


def test_jwt_expiry_epoch_parses_exp():
    import base64
    import json

    payload = base64.urlsafe_b64encode(json.dumps({"exp": 12345}).encode()).decode().rstrip("=")
    assert providers._jwt_expiry_epoch(f"header.{payload}.sig") == 12345
    assert providers._jwt_expiry_epoch("dapi-static-pat") is None


def test_anthropic_bearer_auth_quirk(tmp_path, monkeypatch):
    """anthropic_bearer_auth quirk switches Anthropic-protocol headers to Bearer."""
    from src import providers
    from src.server import _forward_headers

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "providers:\n"
        "  gwprov:\n"
        "    base_url: https://gw.example.com\n"
        "    api_key: k\n"
        "    quirks: [anthropic_bearer_auth]\n"
        "  plainprov:\n"
        "    base_url: https://plain.example.com\n"
        "    api_key: k\n"
    )
    monkeypatch.setattr(providers, "CONFIG_PATH", cfg)
    providers.reload()

    class FakeState:
        api_key = "sekret"

    class FakeRequest:
        state = FakeState()

    h = _forward_headers(FakeRequest(), protocol="anthropic", provider="gwprov")
    assert h["Authorization"] == "Bearer sekret"
    assert "x-api-key" not in h

    h2 = _forward_headers(FakeRequest(), protocol="anthropic", provider="plainprov")
    assert h2["x-api-key"] == "sekret"
    assert "Authorization" not in h2
    providers.reload()
