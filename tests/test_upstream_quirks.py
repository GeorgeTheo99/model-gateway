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
from types import SimpleNamespace

import httpx
import pytest

import src.circuit as circuit
import src.model_fallback as model_fallback
import src.providers as providers
from src import errors
from src.server import _maybe_stream_options, _upstream_endpoint


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
