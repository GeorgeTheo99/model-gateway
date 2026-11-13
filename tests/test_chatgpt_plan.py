"""The owner's ChatGPT plan as a provider: sign-in (src/chatgpt_auth.py), the Chat-to-Responses
route (src/chatgpt_responses.py), the shipped catalog row, and the consumer restriction."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import stat
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from src import chatgpt_auth, chatgpt_responses, onboarding, providers, server, shipped, upstream

ISSUED = "oaiapp_test123"
_AsyncClient = httpx.AsyncClient  # before any test patches it
ADMIN = "admin-secret"
MYAI = "myai-secret"
OTHER = "other-secret"


def _b64(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


def _jwt(claims: dict) -> str:
    return f"{_b64({'alg': 'RS256'})}.{_b64(claims)}.sig"


def _id_token(nonce: str, *, sub: str = "user-1", aud: str = ISSUED, iss: str = chatgpt_auth.ISSUER) -> str:
    return _jwt({"iss": iss, "aud": aud, "sub": sub, "nonce": nonce, "exp": time.time() + 600,
                 "email": "owner@example.com"})


@pytest.fixture(autouse=True)
def _fresh_auth_state(monkeypatch):
    monkeypatch.setattr(chatgpt_auth, "_attempt", None)
    monkeypatch.setattr(chatgpt_auth, "_last_error", None)
    monkeypatch.setattr(chatgpt_auth, "_cache", None)
    monkeypatch.setattr(chatgpt_auth, "_lock", asyncio.Lock())
    monkeypatch.setattr(chatgpt_responses, "_reasoning", type(chatgpt_responses._reasoning)())
    yield
    if chatgpt_auth._attempt is not None and chatgpt_auth._attempt.server is not None:
        chatgpt_auth._attempt.server.close()


@pytest.fixture
def token_endpoint(monkeypatch):
    """Answer token requests; each test sets ``scope`` and inspects ``forms``."""
    state = {"forms": [], "scope": f"openid profile email offline_access resource.invoke {chatgpt_auth.PLAN_SCOPE}",
             "sub": "user-1", "error": None}

    async def post_token(form):
        state["forms"].append(dict(form))
        if state["error"]:
            raise chatgpt_auth.ChatGPTAuthError(state["error"], "refused")
        n = len(state["forms"])
        body = {"access_token": _jwt({"exp": time.time() + 3600, "https://api.openai.com/auth":
                                      {"chatgpt_plan_type": "pro"}, "n": n}),
                "refresh_token": f"refresh-{n}", "expires_in": 3600, "scope": state["scope"], "token_type": "Bearer"}
        if form["grant_type"] == "authorization_code":
            body["id_token"] = _id_token(state["nonce"], sub=state["sub"], aud=form["client_id"])
        return body

    monkeypatch.setattr(chatgpt_auth, "_post_token", post_token)
    return state


async def _sign_in(token_endpoint, **start_kwargs) -> dict:
    started = await chatgpt_auth.start(**start_kwargs)
    params = {k: v[0] for k, v in parse_qs(urlsplit(started["authorize_url"]).query).items()}
    token_endpoint["nonce"] = params["nonce"]
    callback = f"{params['redirect_uri']}?code=abc&state={params['state']}"
    if params["client_id"] == chatgpt_auth.REGISTRATION_CLIENT:
        callback += f"&client_id={ISSUED}"
    return await chatgpt_auth.complete(callback), params


# ── Sign-in ───────────────────────────────────────────────────────────────────


def test_first_sign_in_registers_and_saves_a_private_record(token_endpoint):
    async def run():
        result, params = await _sign_in(token_endpoint, agent_name="Myai")
        return result, params

    result, params = asyncio.run(run())
    assert params["client_id"] == chatgpt_auth.REGISTRATION_CLIENT
    assert params["agent_name_hint"] == "Myai"
    assert params["code_challenge_method"] == "S256" and params["resource"] == chatgpt_auth.RESOURCE
    assert params["redirect_uri"].startswith("http://127.0.0.1:") and params["redirect_uri"].endswith("/auth/callback")
    assert params["ext_agent_host_id"].startswith("urn:uuid:")
    assert set(params["scope"].split()) >= {"offline_access", "resource.invoke", chatgpt_auth.PLAN_SCOPE}
    exchange = token_endpoint["forms"][0]
    assert exchange["client_id"] == ISSUED and exchange["redirect_uri"] == params["redirect_uri"]
    assert exchange["code_verifier"] and exchange["resource"] == chatgpt_auth.RESOURCE
    assert result["signed_in"] and result["email"] == "owner@example.com" and result["plan"] == "pro"
    assert "access_token" not in json.dumps(result)
    path = chatgpt_auth.record_path()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    record = json.loads(path.read_text())
    assert record["client_id"] == ISSUED and record["ext_agent_host_id"] == params["ext_agent_host_id"]
    assert chatgpt_auth.access_token() == record["access_token"]


def test_returning_sign_in_reuses_the_client_and_host(token_endpoint):
    async def run():
        _, first = await _sign_in(token_endpoint)
        _, second = await _sign_in(token_endpoint)
        return first, second

    first, second = asyncio.run(run())
    assert second["client_id"] == ISSUED and "agent_name_hint" not in second
    assert second["ext_agent_host_id"] == first["ext_agent_host_id"]
    assert second["id_token_hint"] and second["login_hint"] == "owner@example.com"


def test_returning_sign_in_refuses_another_account(token_endpoint):
    async def run():
        await _sign_in(token_endpoint)
        token_endpoint["sub"] = "someone-else"
        with pytest.raises(chatgpt_auth.ChatGPTAuthError) as exc:
            await _sign_in(token_endpoint)
        return exc.value.code

    assert asyncio.run(run()) == "account_mismatch"


@pytest.mark.parametrize("mutate,code", [
    (lambda q: q.replace("state=", "state=x"), "state_mismatch"),
    (lambda q: q.replace("code=abc&", "error=access_denied&"), "access_denied"),
    (lambda q: q.replace(f"&client_id={ISSUED}", ""), "registration_incomplete"),
])
def test_bad_callbacks_are_refused(token_endpoint, mutate, code):
    async def run():
        started = await chatgpt_auth.start()
        params = {k: v[0] for k, v in parse_qs(urlsplit(started["authorize_url"]).query).items()}
        token_endpoint["nonce"] = params["nonce"]
        callback = f"{params['redirect_uri']}?code=abc&state={params['state']}&client_id={ISSUED}"
        with pytest.raises(chatgpt_auth.ChatGPTAuthError) as exc:
            await chatgpt_auth.complete(mutate(callback))
        return exc.value.code

    assert asyncio.run(run()) == code
    assert not chatgpt_auth.status()["signed_in"]


def test_sign_in_without_plan_scope_is_refused(token_endpoint):
    token_endpoint["scope"] = "openid profile email offline_access"

    with pytest.raises(chatgpt_auth.ChatGPTAuthError) as exc:
        asyncio.run(_sign_in(token_endpoint))
    assert exc.value.code == "plan_not_granted"
    assert chatgpt_auth.access_token() == ""


def test_id_token_checks():
    good = _id_token("n")
    assert chatgpt_auth._validate_id_token(good, client_id=ISSUED, nonce="n", now=time.time())["sub"] == "user-1"
    for token, nonce in ((good, "other"), (_id_token("n", aud="oaiapp_other"), "n"), (_id_token("n", iss="https://evil"), "n")):
        with pytest.raises(chatgpt_auth.ChatGPTAuthError):
            chatgpt_auth._validate_id_token(token, client_id=ISSUED, nonce=nonce, now=time.time())


def test_loopback_listener_finishes_the_sign_in(token_endpoint, monkeypatch):
    async def run():
        started = await chatgpt_auth.start()
        params = {k: v[0] for k, v in parse_qs(urlsplit(started["authorize_url"]).query).items()}
        token_endpoint["nonce"] = params["nonce"]
        port = urlsplit(params["redirect_uri"]).port
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"GET /auth/callback?code=abc&state={params['state']}&client_id={ISSUED} HTTP/1.1\r\n"
                     f"Host: 127.0.0.1\r\n\r\n".encode())
        await writer.drain()
        answer = await reader.read()
        writer.close()
        return answer

    answer = asyncio.run(run())
    assert answer.startswith(b"HTTP/1.1 200") and b"Signed in" in answer
    assert chatgpt_auth.status()["signed_in"]
    assert chatgpt_auth.status()["pending"] is None


def test_refresh_rotates_and_unusable_refresh_signs_out(token_endpoint):
    asyncio.run(_sign_in(token_endpoint))
    before = json.loads(chatgpt_auth.record_path().read_text())

    assert asyncio.run(chatgpt_auth.refresh()) is None  # not near expiry
    token = asyncio.run(chatgpt_auth.refresh(force=True))
    after = json.loads(chatgpt_auth.record_path().read_text())
    form = token_endpoint["forms"][-1]
    assert form == {"grant_type": "refresh_token", "client_id": ISSUED,
                    "refresh_token": before["refresh_token"], "resource": chatgpt_auth.RESOURCE}
    assert token == after["access_token"] != before["access_token"]
    assert after["refresh_token"] != before["refresh_token"]

    token_endpoint["error"] = "refresh_token_reused"
    assert asyncio.run(chatgpt_auth.refresh(force=True)) is None
    status = chatgpt_auth.status()
    assert not status["signed_in"] and status["reauth_reason"] == "refresh_token_reused" and status["registered"]


def test_transient_refresh_failure_keeps_the_session(token_endpoint):
    asyncio.run(_sign_in(token_endpoint))
    token_endpoint["error"] = "http_503"
    assert asyncio.run(chatgpt_auth.refresh(force=True)) is None
    assert chatgpt_auth.status()["signed_in"]


def test_sign_out_revokes_and_keeps_the_registration(token_endpoint, monkeypatch):
    asyncio.run(_sign_in(token_endpoint))
    refresh_token = json.loads(chatgpt_auth.record_path().read_text())["refresh_token"]
    seen = []

    def handler(request):
        if request.url.path.endswith("openid-configuration"):
            return httpx.Response(200, json={"revocation_endpoint": f"{chatgpt_auth.ISSUER}/oauth/revoke"})
        seen.append(parse_qs(request.content.decode()))
        return httpx.Response(200)

    real = httpx.AsyncClient
    monkeypatch.setattr(chatgpt_auth.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    result = asyncio.run(chatgpt_auth.sign_out())
    assert result["revocation_confirmed"] and not result["signed_in"] and result["registered"]
    assert seen == [{"token": [refresh_token], "token_type_hint": ["refresh_token"], "client_id": [ISSUED]}]
    record = json.loads(chatgpt_auth.record_path().read_text())
    assert "refresh_token" not in record and record["client_id"] == ISSUED


# ── Request and stream translation ────────────────────────────────────────────


def test_chat_request_becomes_a_plan_responses_request():
    body = {
        "model": "x", "temperature": 0.2, "max_tokens": 50, "top_p": 1, "user": "u", "metadata": {"a": 1},
        "prompt_cache_retention": "24h", "prompt_cache_key": "conv-1", "stream_options": {"include_usage": True},
        "parallel_tool_calls": False, "tool_choice": "auto",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "What's this?"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA", "detail": "high"}}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "look", "arguments": "{\"q\":1}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "a cat"},
            {"role": "system", "content": "Now answer."},
            {"role": "assistant", "content": "A cat."},
        ],
        "tools": [{"type": "function", "function": {"name": "look", "description": "Look", "parameters": {"type": "object"}}}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}, "strict": True}},
    }
    request = chatgpt_responses.chat_to_responses_request(body, "gpt-6.1-sol", {"effort": "high"})
    assert request["store"] is False and request["stream"] is True
    assert request["instructions"] == "Be brief."
    assert request["reasoning"] == {"effort": "high", "summary": "auto"}
    assert request["include"] == ["reasoning.encrypted_content"]
    assert request["tools"] == [{"type": "namespace", "name": "app", "description": chatgpt_responses._NAMESPACE_DESCRIPTION,
                                 "tools": [{"type": "function", "name": "look", "description": "Look",
                                            "parameters": {"type": "object"}, "strict": False}]}]
    assert request["tool_choice"] == "auto" and request["parallel_tool_calls"] is False
    assert request["text"] == {"format": {"type": "json_schema", "name": "r", "schema": {"type": "object"}, "strict": True}}
    assert request["prompt_cache_key"] == "conv-1"
    for dropped in ("temperature", "max_tokens", "max_output_tokens", "top_p", "user", "metadata",
                    "prompt_cache_retention", "stream_options", "messages"):
        assert dropped not in request
    assert request["input"] == [
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "What's this?"},
            {"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "high"}]},
        {"type": "function_call", "call_id": "call_1", "name": "look", "namespace": "app", "arguments": "{\"q\":1}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "a cat"},
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Now answer."}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "A cat."}]},
    ]


def test_named_tool_choice_and_unsupported_content():
    tools = [{"type": "function", "function": {"name": "f"}}, {"type": "function", "function": {"name": "g"}}]
    request = chatgpt_responses.chat_to_responses_request(
        {"messages": [{"role": "user", "content": "hi"}], "tools": tools,
         "tool_choice": {"type": "function", "function": {"name": "f"}}}, "m", None)
    # The plan can't name a namespaced function: only that one is sent, required.
    assert request["tool_choice"] == "required"
    assert [tool["name"] for tool in request["tools"][0]["tools"]] == ["f"]
    with pytest.raises(chatgpt_responses.TranslationError):
        chatgpt_responses.chat_to_responses_request(
            {"messages": [{"role": "user", "content": "hi"}], "tools": tools,
             "tool_choice": {"type": "function", "function": {"name": "h"}}}, "m", None)
    assert "reasoning" not in request
    with pytest.raises(chatgpt_responses.TranslationError):
        chatgpt_responses.chat_to_responses_request(
            {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]}, "m", None)
    with pytest.raises(chatgpt_responses.TranslationError):
        chatgpt_responses.chat_to_responses_request(
            {"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "web_search"}]}, "m", None)


def _sse(*events: dict) -> list[bytes]:
    return [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events]


async def _aiter(chunks):
    for chunk in chunks:
        yield chunk


def _collect(chunks) -> list:
    async def run():
        return [c async for c in chatgpt_responses.responses_sse_to_chat(_aiter(chunks), "gpt-6.1-sol-chatgpt")]

    out = []
    for raw in asyncio.run(run()):
        text = raw.decode()
        out.append("[DONE]" if text.startswith("data: [DONE]") else json.loads(text[6:]))
    return out


USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
         "input_tokens_details": {"cached_tokens": 80}, "output_tokens_details": {"reasoning_tokens": 5}}
REASONING = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc-1"}
TOOL_STREAM = _sse(
    {"type": "response.created", "response": {}},
    {"type": "response.reasoning_summary_part.added"},
    {"type": "response.reasoning_summary_text.delta", "delta": "Thinking"},
    {"type": "response.output_item.done", "output_index": 0, "item": REASONING},
    {"type": "response.output_item.added", "output_index": 1,
     "item": {"type": "function_call", "call_id": "call_9", "name": "look", "namespace": "app", "arguments": ""}},
    {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{\"q\""},
    {"type": "response.function_call_arguments.delta", "output_index": 1, "delta": ":2}"},
    {"type": "response.completed", "response": {"usage": USAGE}},
)


def test_stream_translation_of_a_tool_call_keeps_reasoning_for_the_next_turn():
    events = _collect(TOOL_STREAM)
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    deltas = [e["choices"][0]["delta"] for e in events[1:-2]]
    assert deltas[0] == {"reasoning_content": "Thinking"}
    assert deltas[1]["tool_calls"][0] == {"index": 0, "id": "call_9", "type": "function",
                                         "function": {"name": "look", "arguments": ""}}
    assert "".join(d["tool_calls"][0]["function"]["arguments"] for d in deltas[1:]) == "{\"q\":2}"
    final = events[-2]
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    assert final["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                              "prompt_tokens_details": {"cached_tokens": 80},
                              "completion_tokens_details": {"reasoning_tokens": 5}}
    assert events[-1] == "[DONE]"

    follow_up = chatgpt_responses.chat_to_responses_request({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "call_9", "function": {"name": "look", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_9", "content": "ok"},
    ]}, "m", None)
    assert follow_up["input"][1] == {"type": "reasoning", "summary": [], "encrypted_content": "enc-1"}
    assert follow_up["input"][2]["type"] == "function_call"


@pytest.mark.parametrize("asked,sent", [(None, False), (False, False), (True, True)])
def test_a_tool_is_strict_only_when_the_request_says_so(asked, sent):
    function = {"name": "look", "parameters": {"type": "object"}, **({} if asked is None else {"strict": asked})}
    request = chatgpt_responses.chat_to_responses_request(
        {"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": function}]}, "m", None)
    assert request["tools"][0]["tools"][0]["strict"] is sent


def _call_events(output_index, call_id, name, arguments, *, deltas=(), done=("arguments", "item")):
    """One function call's events; the plan sends a parallel call's arguments only in ``done``."""
    item = {"type": "function_call", "call_id": call_id, "name": name, "namespace": "app"}
    events = [{"type": "response.output_item.added", "output_index": output_index, "item": {**item, "arguments": ""}}]
    events += [{"type": "response.function_call_arguments.delta", "output_index": output_index, "delta": delta}
               for delta in deltas]
    if "arguments" in done:
        events.append({"type": "response.function_call_arguments.done", "output_index": output_index,
                       "arguments": arguments})
    if "item" in done:
        events.append({"type": "response.output_item.done", "output_index": output_index,
                       "item": {**item, "arguments": arguments}})
    return events


def _calls(events) -> list[dict]:
    result, error = asyncio.run(chatgpt_responses.responses_sse_to_chat_completion(_aiter(events), "m"))
    assert error is None
    return [{"id": call["id"], **call["function"]} for call in result["choices"][0]["message"]["tool_calls"]]


@pytest.mark.parametrize("done", [("arguments", "item"), ("arguments",), ("item",)])
def test_parallel_calls_take_their_arguments_from_the_done_events(done):
    calls = _calls(_sse(
        {"type": "response.output_item.done", "output_index": 0, "item": REASONING},
        *_call_events(1, "call_1", "search", '{"query":"weather"}', done=done),
        *_call_events(2, "call_2", "recall", '{"query":"city"}', done=done),
        *_call_events(3, "call_3", "list", "{}", done=done),
        {"type": "response.completed", "response": {"usage": USAGE}},
    ))
    assert calls == [{"id": "call_1", "name": "search", "arguments": '{"query":"weather"}'},
                     {"id": "call_2", "name": "recall", "arguments": '{"query":"city"}'},
                     {"id": "call_3", "name": "list", "arguments": "{}"}]


@pytest.mark.parametrize("deltas", [('{"q"', ":2}"), ('{"q"',)])
def test_done_events_add_only_the_arguments_the_deltas_did_not_send(deltas):
    events = _collect(_sse(*_call_events(0, "call_9", "look", '{"q":2}', deltas=deltas),
                           {"type": "response.completed", "response": {}}))
    sent = [call["function"]["arguments"] for event in events[1:-2]
            for call in event["choices"][0]["delta"]["tool_calls"]]
    assert sent == ["", *deltas, *([":2}"] if len(deltas) == 1 else [])]


@pytest.mark.parametrize("events,expected", [
    (_sse({"type": "response.output_text.delta", "delta": "Hel"},
          {"type": "response.failed", "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded",
                                                             "message": "Limit reached"}}}),
     {"type": "api_error", "message": "Limit reached", "code": "subscription_sharing_usage_limit_exceeded"}),
    (_sse({"type": "response.output_text.delta", "delta": "Hel"}),
     {"type": "api_error", "message": "The ChatGPT plan stream ended before the response completed"}),
])
def test_stream_failures_end_in_an_error_not_a_finished_answer(events, expected):
    out = _collect(events)
    assert out[1]["choices"][0]["delta"] == {"content": "Hel"}
    assert out[-1] == {"error": expected}


def test_incomplete_response_finishes_with_length():
    out = _collect(_sse({"type": "response.output_text.delta", "delta": "x"},
                        {"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}))
    assert out[-2]["choices"][0]["finish_reason"] == "length" and out[-1] == "[DONE]"


# ── Catalog, routing and the server ───────────────────────────────────────────


def _configure(monkeypatch, consumers=("myai",)):
    config = {
        "providers": {
            "chatgpt": {} if consumers is None else {"consumers": list(consumers)},
            "openai": {"base_url": "https://api.openai.com/v1", "api_key": "sk-api"},
        },
        "auth": {
            "admin_keys": [ADMIN],
            "consumer_credentials": [
                {"id": "myai-runtime", "consumer": "myai", "key": MYAI, "namespaces": ["myai"],
                 "permissions": ["profiles:read", "profiles:invoke"], "allow_direct_models": True},
                {"id": "other-runtime", "consumer": "dessecker", "key": OTHER, "namespaces": ["myai"],
                 "permissions": ["profiles:read", "profiles:invoke"], "allow_direct_models": True},
            ],
        },
    }
    monkeypatch.setattr(providers, "_config", config)
    monkeypatch.setattr(providers, "_models", None)
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", providers.Path("/nonexistent-model-info.json"))
    monkeypatch.setattr(shipped, "_packaged", lambda: {
        pid: p for pid, p in onboarding.packaged_profiles().items() if pid in ("chatgpt-plan", "openai-gpt")})
    return config


def test_shipped_row_needs_a_sign_in_and_never_takes_the_api_route(monkeypatch, token_endpoint):
    _configure(monkeypatch)
    assert providers.resolve("gpt-6.1-sol-chatgpt") is None
    # Listed while signed out, so a routing profile naming it stays valid; not available.
    assert providers.model_availability("gpt-6.1-sol-chatgpt")["reason"] == "chatgpt_signed_out"
    asyncio.run(_sign_in(token_endpoint))
    _configure(monkeypatch)  # signing in reloads the registry; this test's config isn't on disk
    info = providers.resolve("gpt-6.1-sol-chatgpt")
    assert info.provider == "chatgpt" and info.provider_model_id == "gpt-6.1-sol"
    assert info.api_key == chatgpt_auth.access_token() and info.base_url == chatgpt_auth.RESOURCE
    assert providers.resolve("gpt-6.1-sol").provider == "openai"


@pytest.fixture
def plan_upstream(monkeypatch, token_endpoint):
    _configure(monkeypatch)
    asyncio.run(_sign_in(token_endpoint))
    _configure(monkeypatch)
    monkeypatch.setattr(upstream, "_RETRY_MAX", 1)
    monkeypatch.setattr(upstream, "_RETRY_TRANSPORT_ATTEMPTS", 1)
    seen = []
    state = {"events": TOOL_STREAM, "status": 200}

    def handler(request):
        seen.append({"url": str(request.url), "auth": request.headers.get("authorization"),
                     "body": json.loads(request.content)})
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"error": {"type": "invalid_request_error", "message": "nope"}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"".join(state["events"]))

    real = httpx.AsyncClient
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return seen, state


def _chat(key, stream, model="gpt-6.1-sol-chatgpt"):
    return TestClient(server.app).post(
        "/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "stream": stream, "temperature": 0.2, "reasoning_effort": "high",
              "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
              "tools": [{"type": "function", "function": {"name": "look", "parameters": {"type": "object"}}}]})


@pytest.mark.parametrize("stream", [True, False])
def test_chat_completions_run_on_the_plan(plan_upstream, stream):
    seen, _ = plan_upstream
    result = _chat(MYAI, stream)
    assert result.status_code == 200
    assert seen[0]["url"] == "https://api.openai.com/v1/responses"
    assert seen[0]["auth"] == f"Bearer {chatgpt_auth.access_token()}"
    sent = seen[0]["body"]
    assert sent["model"] == "gpt-6.1-sol" and sent["store"] is False and sent["stream"] is True
    assert sent["instructions"] == "sys" and sent["reasoning"]["effort"] == "high" and "temperature" not in sent
    if stream:
        assert '"finish_reason": "tool_calls"' in result.text and result.text.rstrip().endswith("data: [DONE]")
    else:
        message = result.json()["choices"][0]["message"]
        assert message["tool_calls"][0]["function"] == {"name": "look", "arguments": "{\"q\":2}"}
        assert result.json()["usage"]["prompt_tokens"] == 100


def test_usage_limit_is_an_error_without_a_fallback(plan_upstream):
    _, state = plan_upstream
    state["events"] = _sse({"type": "response.failed", "response": {"error": {
        "code": "subscription_sharing_usage_limit_exceeded", "message": "Limit reached"}}})
    result = _chat(MYAI, stream=False)
    assert result.status_code == 429
    assert result.json()["error"]["code"] == "subscription_sharing_usage_limit_exceeded"


def test_other_consumers_and_endpoints_cannot_use_the_plan(plan_upstream):
    seen, _ = plan_upstream
    assert _chat(OTHER, stream=False).status_code == 404
    assert _chat(ADMIN, stream=False).status_code == 200
    messages = TestClient(server.app).post(
        "/v1/messages", headers={"Authorization": f"Bearer {MYAI}"},
        json={"model": "gpt-6.1-sol-chatgpt", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    assert messages.status_code == 400 and "chat/completions" in messages.text
    assert len(seen) == 1


def test_admin_api_drives_the_sign_in(monkeypatch, token_endpoint):
    _configure(monkeypatch)
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    client = TestClient(server.app)
    headers = {"Authorization": f"Bearer {ADMIN}"}
    started = client.post("/admin/api/chatgpt/sign-in", headers=headers, json={"agent_name": "Myai"})
    assert started.status_code == 200
    params = {k: v[0] for k, v in parse_qs(urlsplit(started.json()["authorize_url"]).query).items()}
    token_endpoint["nonce"] = params["nonce"]
    done = client.post("/admin/api/chatgpt/callback", headers=headers, json={
        "url": f"{params['redirect_uri']}?code=abc&state={params['state']}&client_id={ISSUED}"})
    assert done.status_code == 200 and done.json()["signed_in"]
    _configure(monkeypatch)  # signing in reloads the registry; this test's config isn't on disk
    got = client.get("/admin/api/chatgpt", headers=headers)
    assert got.json().get("email") == "owner@example.com", got.text
    rows = {r["id"]: r for r in client.get("/admin/api/onboarding/profiles", headers=headers).json()["profiles"]}
    assert rows["chatgpt-plan"]["auth"] == "chatgpt" and rows["chatgpt-plan"]["has_api_key"]
    assert rows["openai-gpt"]["auth"] == "api_key"
    refused = client.post("/admin/api/onboarding/profiles/chatgpt-plan", headers=headers, json={"api_key": "sk-x"})
    assert refused.status_code == 400 and refused.json()["code"] == "sign_in_required"


def test_a_profile_naming_the_plan_registers_signed_out_and_runs_once_signed_in(monkeypatch, token_endpoint):
    config = _configure(monkeypatch)
    config["auth"]["consumer_credentials"][0]["permissions"] += ["profiles:write"]
    manifest = {"schema_version": 1, "namespace": "myai", "source_revision": "myai@test",
                "default_profile": "myai/cloud-best",
                "profiles": [{"id": "myai/cloud-best", "locality": "cloud_explicit", "credential_policy": "gateway_managed",
                              "protocols": ["openai_chat"], "defaults": {"reasoning_effort": "high"},
                              "routes": {"text": "gpt-6.1-sol-chatgpt", "vision": "gpt-6.1-sol-chatgpt"}}]}
    client = TestClient(server.app)
    headers = {"Authorization": f"Bearer {MYAI}"}
    registered = client.put("/v1/profiles/myai/snapshot", headers={**headers, "If-None-Match": "*"}, json=manifest)
    assert registered.status_code == 200, registered.text
    signed_out = client.post("/v1/chat/completions", headers=headers, json={
        "model": "profile:myai/cloud-best", "messages": [{"role": "user", "content": "hi"}]})
    assert signed_out.status_code == 503 and "Sign in with ChatGPT" in signed_out.text
    assert "chatgpt_signed_out" in signed_out.text
    asyncio.run(_sign_in(token_endpoint))
    _configure(monkeypatch)["auth"]["consumer_credentials"][0]["permissions"] += ["profiles:write"]
    from src import profiles
    assert profiles.resolve_execution("profile:myai/cloud-best", protocol="openai_chat", has_image=False).model \
        == "gpt-6.1-sol-chatgpt"


def test_without_a_consumers_list_only_the_consumer_that_signed_in_may_use_the_plan(monkeypatch, token_endpoint):
    _configure(monkeypatch, consumers=None)
    assert providers.provider_consumers("chatgpt") == frozenset()

    async def run():
        started = await chatgpt_auth.start(consumer="myai")
        params = {k: v[0] for k, v in parse_qs(urlsplit(started["authorize_url"]).query).items()}
        token_endpoint["nonce"] = params["nonce"]
        await chatgpt_auth.complete(f"{params['redirect_uri']}?code=abc&state={params['state']}&client_id={ISSUED}")

    asyncio.run(run())
    _configure(monkeypatch, consumers=None)
    assert providers.provider_consumers("chatgpt") == frozenset({"myai"})
    assert providers.provider_consumers("openai") is None


def test_plan_429_is_not_retried_and_names_its_code(plan_upstream, monkeypatch):
    seen, state = plan_upstream
    state["status"] = 429
    calls = []
    real = _AsyncClient

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={"error": {"type": "usage_limit_error", "code": "subscription_sharing_usage_limit_exceeded",
                                                   "message": "Limit reached"}})

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(upstream, "_RETRY_MAX", 5)
    result = _chat(MYAI, stream=True)
    assert result.status_code == 429 and len(calls) == 1
    assert result.headers["x-model-gateway-error-code"] == "subscription_sharing_usage_limit_exceeded"


def test_a_401_refreshes_once_and_403_does_not(plan_upstream, monkeypatch):
    _, _ = plan_upstream
    refreshed = []

    async def refresh(*, force=False, failed_token=None):
        refreshed.append(failed_token)
        return None

    monkeypatch.setattr(chatgpt_auth, "refresh", refresh)
    real = _AsyncClient
    for status in (403, 401):
        monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kw: real(
            transport=httpx.MockTransport(lambda request: httpx.Response(status, json={"error": {"message": "no"}})), **kw))
        _chat(MYAI, stream=False)
    # Preflight checks expiry on every request; only the 401 forces a refresh, naming the refused token.
    assert chatgpt_auth.access_token() in refreshed
    assert refreshed.count(chatgpt_auth.access_token()) == 1


def test_a_registration_whose_exchange_failed_is_retried_with_its_issued_client(token_endpoint):
    async def run():
        started = await chatgpt_auth.start(agent_name="Myai")
        params = {k: v[0] for k, v in parse_qs(urlsplit(started["authorize_url"]).query).items()}
        token_endpoint["error"] = "invalid_grant"
        with pytest.raises(chatgpt_auth.ChatGPTAuthError):
            await chatgpt_auth.complete(f"{params['redirect_uri']}?code=abc&state={params['state']}&client_id={ISSUED}")
        token_endpoint["error"] = None
        again = await chatgpt_auth.start(agent_name="Myai")
        return {k: v[0] for k, v in parse_qs(urlsplit(again["authorize_url"]).query).items()}

    again = asyncio.run(run())
    assert again["client_id"] == ISSUED and "agent_name_hint" not in again


def test_status_never_carries_the_sign_in_url(token_endpoint):
    asyncio.run(_sign_in(token_endpoint))

    async def run():
        await chatgpt_auth.start()
        return chatgpt_auth.status()

    pending = asyncio.run(run())["pending"]
    assert pending is not None and "authorize_url" not in pending


def test_peers_are_not_offered_the_plan(monkeypatch, token_endpoint):
    _configure(monkeypatch)
    asyncio.run(_sign_in(token_endpoint))
    _configure(monkeypatch)
    ids = {row["id"] for row in server._federation_rows(2)}
    assert "gpt-6.1-sol-chatgpt" not in ids and "gpt-6.1-sol" in ids
