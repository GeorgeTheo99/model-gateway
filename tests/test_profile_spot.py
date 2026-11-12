"""A profile's prefer_spot: a spot tier first, falling back to the route itself (docs/consumer-profiles.md)."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from src import circuit, ledger, profiles, providers, server

client = TestClient(server.app)
TOKEN = "consumer-secret"
PROVIDER = "cloud-test"
STANDARD = {
    "name": "std-model",
    "provider": PROVIDER,
    "provider_model_id": "std-native",
    "pricing": {"input": 2.0, "output": 4.0},
}
SPOT = {
    "name": "spot-model",
    "provider": PROVIDER,
    "provider_model_id": "spot-native",
    "spot_of": "std-model",
    "pricing": {"input": 1.0, "output": 2.0},
}
VISION = {
    "name": "vision-model",
    "provider": PROVIDER,
    "provider_model_id": "vision-native",
    "vision": True,
    "pricing": {"input": 0.5, "output": 1.0},
}
CHAT = {"model": "profile:ha/cloud-best", "messages": [{"role": "user", "content": "hello"}]}


@pytest.fixture(autouse=True)
def _clean_circuit():
    circuit._circuits.pop(PROVIDER, None)
    yield
    circuit._circuits.pop(PROVIDER, None)


def _configure(monkeypatch, models=(STANDARD, SPOT, VISION), **providers_extra):
    config = {
        "providers": {PROVIDER: {"base_url": "https://cloud.invalid/v1", "api_key": "provider-secret"}, **providers_extra},
        "auth": {
            "consumer_credentials": [{
                "id": "ha-runtime",
                "consumer": "ha",
                "key": TOKEN,
                "namespaces": ["ha"],
                "permissions": ["profiles:read", "profiles:write", "profiles:invoke"],
                "allow_direct_models": True,
            }],
        },
        "models": deepcopy(list(models)),
    }
    monkeypatch.setattr(providers, "_config", config)
    monkeypatch.setattr(providers, "_models", None)
    return config


def _manifest(*, prefer_spot=True, locality="cloud_explicit", credential_policy="gateway_managed", vision=None):
    profile = {
        "id": "ha/cloud-best",
        "locality": locality,
        "credential_policy": credential_policy,
        "protocols": ["openai_chat"],
        "routes": {"text": "std-model", **({"vision": vision} if vision else {})},
        "defaults": {"temperature": 0.2},
    }
    if prefer_spot is not None:
        profile["prefer_spot"] = prefer_spot
    return {"schema_version": 1, "namespace": "ha", "source_revision": "ha@one", "profiles": [profile]}


def _headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def _register(manifest=None):
    return client.put("/v1/profiles/ha/snapshot", headers={**_headers(), "If-None-Match": "*"},
                      json=manifest or _manifest())


def _upstream(monkeypatch, statuses: dict[str, int], *, stream=False):
    """Answer each upstream model id with its status; record the models asked for."""
    calls = []

    async def handle(self, request: httpx.Request):
        body = json.loads(await request.aread())
        calls.append(body["model"])
        status = statuses.get(body["model"], 200)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": f"status {status}"}})
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000}
        message = {"role": "assistant", "content": "ok"}
        if stream:
            chunks = [
                {"id": "c", "object": "chat.completion.chunk", "model": body["model"],
                 "choices": [{"index": 0, "delta": message, "finish_reason": None}]},
                {"id": "c", "object": "chat.completion.chunk", "model": body["model"],
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
            ]
            content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content.encode())
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "usage": usage,
        })

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    return calls


# ── registration ─────────────────────────────────────────────────────────────


def test_prefer_spot_binds_the_spot_tier_without_changing_the_public_snapshot(monkeypatch):
    _configure(monkeypatch)
    response = _register()
    assert response.status_code == 200, response.text
    # Consumers check snapshot keys exactly: the spot binding stays internal.
    assert set(response.json()["profiles"][0]) == {
        "id", "locality", "credential_policy", "executable", "protocols", "routes", "defaults",
    }
    assert response.json()["profiles"][0]["routes"] == {"text": "std-model"}
    stored = json.loads(profiles.registry_path().read_text())["namespaces"]["ha"][-1]["manifest"]["profiles"][0]
    assert (stored["prefer_spot"], stored["spot_routes"]) == (True, {"text": "spot-model"})


def test_a_route_without_a_spot_tier_runs_as_named(monkeypatch):
    _configure(monkeypatch, models=(STANDARD, VISION))
    assert _register().status_code == 200
    stored = json.loads(profiles.registry_path().read_text())["namespaces"]["ha"][-1]["manifest"]["profiles"][0]
    assert stored["prefer_spot"] is True and "spot_routes" not in stored


@pytest.mark.parametrize("manifest, message", [
    (_manifest(prefer_spot="yes"), "prefer_spot must be true or false"),
    (_manifest(locality="local_only", credential_policy="gateway_local"), "prefer_spot needs a cloud_explicit profile"),
])
def test_invalid_prefer_spot_is_rejected(monkeypatch, manifest, message):
    _configure(monkeypatch)
    response = _register(manifest)
    assert response.status_code == 422
    assert message in response.text


@pytest.mark.parametrize("change", [
    {"provider": "other-cloud"},
    {"thinking_format": "kimi"},
    {"vision": True},
    {"max_output_tokens": 999},
])
def test_a_spot_tier_must_match_its_route_on_one_endpoint(monkeypatch, change):
    _configure(monkeypatch, models=(STANDARD, {**SPOT, **change}, VISION),
               **{"other-cloud": {"base_url": "https://other.invalid/v1", "api_key": "other-secret"}})
    response = _register()
    assert response.status_code == 422
    assert "must match 'std-model'" in response.text


def test_spot_of_must_name_another_model(monkeypatch):
    from src.catalog import validate_client_metadata

    with pytest.raises(ValueError, match="spot_of must name another model"):
        validate_client_metadata({**SPOT, "spot_of": "spot-model"})
    with pytest.raises(ValueError, match="take no base_model or alternate_ids"):
        validate_client_metadata({**SPOT, "family": "std", "base_model": "std-model"})
    with pytest.raises(ValueError, match="take no base_model or alternate_ids"):
        validate_client_metadata({**SPOT, "alternate_ids": ["std"]})


def test_removing_the_spot_tier_fails_closed_until_reregistration(monkeypatch):
    config = _configure(monkeypatch)
    assert _register().status_code == 200
    config["models"] = [STANDARD, VISION]
    providers._models = None
    _upstream(monkeypatch, {})
    response = client.post("/v1/chat/completions", headers=_headers(), json=CHAT)
    assert response.status_code == 409
    assert "profile_binding_changed" in response.text


# ── execution ────────────────────────────────────────────────────────────────


def test_spot_serves_when_it_has_capacity(monkeypatch):
    _configure(monkeypatch)
    assert _register().status_code == 200
    calls = _upstream(monkeypatch, {})
    response = client.post("/v1/chat/completions", headers=_headers(), json=CHAT)
    assert response.status_code == 200, response.text
    assert calls == ["spot-native"]
    assert "x-model-gateway-spot-fallback" not in response.headers
    row = ledger.recent(limit=1)[0]
    assert (row["model"], row["provider_model_id"], row["cost_usd"]) == ("spot-model", "spot-native", 1.0)
    assert row["profile_id"] == "ha/cloud-best"


@pytest.mark.parametrize("status", [401, 403, 404, 408, 429, 500, 502, 503, 504])
def test_spot_without_capacity_falls_back_once_without_charging_the_circuit(monkeypatch, status):
    _configure(monkeypatch)
    assert _register().status_code == 200
    calls = _upstream(monkeypatch, {"spot-native": status})
    for _ in range(circuit.TRIP_THRESHOLD + 1):
        response = client.post("/v1/chat/completions", headers=_headers(), json=CHAT)
        assert response.status_code == 200, response.text
    # One spot attempt per request (no retries), then the standard route.
    assert calls == ["spot-native", "std-native"] * (circuit.TRIP_THRESHOLD + 1)
    assert response.headers["x-model-gateway-spot-fallback"] == "spot-model -> std-model"
    # Spot's expected unavailability must not open the circuit the standard route shares.
    assert not circuit.is_tripped(PROVIDER)
    row = ledger.recent(limit=1)[0]
    assert (row["model"], row["provider_model_id"], row["cost_usd"]) == ("std-model", "std-native", 2.0)


def test_spot_stream_falls_back_before_any_output(monkeypatch):
    _configure(monkeypatch)
    assert _register().status_code == 200
    calls = _upstream(monkeypatch, {"spot-native": 503}, stream=True)
    with client.stream("POST", "/v1/chat/completions", headers=_headers(), json={**CHAT, "stream": True}) as response:
        body = b"".join(response.iter_bytes())
    assert response.status_code == 200
    assert b"ok" in body and body.count(b"[DONE]") == 1
    assert calls == ["spot-native", "std-native"]
    row = ledger.recent(limit=1)[0]
    assert (row["model"], row["cost_usd"]) == ("std-model", 2.0)


@pytest.mark.parametrize("status", [400, 413, 422])
def test_a_request_the_standard_route_would_reject_alike_is_returned(monkeypatch, status):
    _configure(monkeypatch)
    assert _register().status_code == 200
    calls = _upstream(monkeypatch, {"spot-native": status})
    response = client.post("/v1/chat/completions", headers=_headers(), json=CHAT)
    assert response.status_code == status
    assert calls == ["spot-native"]


def test_spot_streams_when_it_has_capacity(monkeypatch):
    _configure(monkeypatch)
    assert _register().status_code == 200
    calls = _upstream(monkeypatch, {}, stream=True)
    with client.stream("POST", "/v1/chat/completions", headers=_headers(), json={**CHAT, "stream": True}) as response:
        body = b"".join(response.iter_bytes())
    assert response.status_code == 200 and b"ok" in body
    assert "x-model-gateway-spot-fallback" not in response.headers
    assert calls == ["spot-native"]
    assert ledger.recent(limit=1)[0]["model"] == "spot-model"


def test_a_profile_registered_before_the_spot_route_existed_still_runs(monkeypatch):
    """Deploying a spot route must not change existing bindings (no 409 after an upgrade)."""
    config = _configure(monkeypatch, models=(STANDARD, VISION))
    assert _register(_manifest(prefer_spot=None)).status_code == 200
    config["models"] = [STANDARD, SPOT, VISION]
    providers._models = None
    calls = _upstream(monkeypatch, {})
    assert client.post("/v1/chat/completions", headers=_headers(), json=CHAT).status_code == 200
    assert calls == ["std-native"]


def test_spot_on_another_key_runs_the_route_itself(monkeypatch):
    """The fallback reuses spot's endpoint and headers, so both must resolve to one key."""
    _configure(monkeypatch)
    assert _register().status_code == 200
    real = providers.resolve

    def other_key(model, *args, **kwargs):
        info = real(model, *args, **kwargs)
        return replace(info, api_key="other") if model == "spot-model" and info else info

    monkeypatch.setattr(server, "resolve", other_key)
    calls = _upstream(monkeypatch, {"spot-native": 503})
    assert client.post("/v1/chat/completions", headers=_headers(), json=CHAT).status_code == 200
    assert calls == ["std-native"]


def test_only_a_profile_that_prefers_spot_falls_back(monkeypatch):
    _configure(monkeypatch)
    assert _register(_manifest(prefer_spot=None)).status_code == 200
    calls = _upstream(monkeypatch, {"spot-native": 503})
    assert client.post("/v1/chat/completions", headers=_headers(), json=CHAT).status_code == 200
    assert calls == ["std-native"]  # the profile without prefer_spot runs its route as named
    calls.clear()
    # Naming the spot route directly gets spot alone: its unavailability is the caller's to see,
    # and it never opens the circuit the provider's other routes share.
    monkeypatch.setattr("src.upstream._compute_retry_delay", lambda resp, attempt: 0.0)
    for _ in range(2):
        direct = client.post("/v1/chat/completions", headers=_headers(), json={**CHAT, "model": "spot-model"})
        assert direct.status_code == 502
    assert set(calls) == {"spot-native"}
    assert not circuit.is_tripped(PROVIDER)


def test_an_image_request_takes_the_vision_route_without_spot(monkeypatch):
    _configure(monkeypatch)
    assert _register(_manifest(vision="vision-model")).status_code == 200
    calls = _upstream(monkeypatch, {"spot-native": 503})
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    response = client.post("/v1/chat/completions", headers=_headers(), json={
        **CHAT, "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, image]}],
    })
    assert response.status_code == 200, response.text
    assert calls == ["vision-native"]


def test_spot_falls_back_on_a_transport_error(monkeypatch):
    _configure(monkeypatch)
    assert _register().status_code == 200
    calls = _upstream(monkeypatch, {})
    answer = httpx.AsyncHTTPTransport.handle_async_request

    async def gone_then_unreachable(self, request: httpx.Request):
        body = json.loads(await request.aread())
        if body["model"] == "spot-native":
            calls.append(body["model"])
            if len(calls) == 1:
                raise httpx.ReadTimeout("slow", request=request)
            raise httpx.ConnectError("refused", request=request)
        return await answer(self, request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", gone_then_unreachable)
    for _ in range(2):
        response = client.post("/v1/chat/completions", headers=_headers(), json=CHAT)
        assert response.status_code == 200, response.text
        assert response.headers["x-model-gateway-spot-fallback"] == "spot-model -> std-model"
    assert calls == ["spot-native", "std-native", "spot-native", "std-native"]
