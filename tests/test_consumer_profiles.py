from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from src import ledger, profiles, providers, server
from src.auth import AuthIdentity, ConsumerPrincipal, validate_credential_separation


client = TestClient(server.app)
TOKEN = "consumer-secret"
LEGACY = "legacy-secret"


def _configure(monkeypatch, *, permissions=None, namespaces=None, allow_direct=False, models=None, extra=None):
    config = {
        "providers": {},
        "auth": {
            "client_keys": [LEGACY],
            "consumer_credentials": [{
                "id": "ha-runtime",
                "consumer": "ha",
                "key": TOKEN,
                "namespaces": namespaces or ["ha"],
                "permissions": permissions or ["profiles:read", "profiles:write", "profiles:invoke"],
                "allow_direct_models": allow_direct,
            }],
        },
    }
    if models:
        config["models"] = deepcopy(models)
    if extra:
        config.update(deepcopy(extra))
    monkeypatch.setattr(providers, "_config", config)
    monkeypatch.setattr(providers, "_models", None)
    return config


def _manifest(*, namespace="ha", route="test-local", vision=None, locality="local_only", credential_policy="gateway_local", source_revision="ha@one"):
    routes = {"text": route}
    if vision:
        routes["vision"] = vision
    return {
        "schema_version": 1,
        "namespace": namespace,
        "source_revision": source_revision,
        "default_profile": f"{namespace}/automatic-local",
        "profiles": [{
            "id": f"{namespace}/automatic-local",
            "description": "Local automatic route",
            "locality": locality,
            "credential_policy": credential_policy,
            "protocols": ["openai_chat", "openai_responses", "anthropic_messages"],
            "routes": routes,
            "defaults": {"temperature": 0.2, "max_output_tokens": 123, "reasoning_effort": "off"},
        }],
    }


def _headers(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def _register(manifest=None):
    return client.put(
        "/v1/profiles/ha/snapshot",
        headers={**_headers(), "If-None-Match": "*"},
        json=manifest or _manifest(),
    )


def test_snapshot_etag_history_idempotency_and_redaction(monkeypatch):
    _configure(monkeypatch)
    first = _register()
    assert first.status_code == 200
    assert first.json()["gateway_version"] == 1
    assert first.headers["cache-control"] == "no-store"
    etag = first.headers["etag"]
    serialized = first.text
    assert "test-local-upstream" not in serialized
    assert "binding" not in serialized
    assert "digest" not in serialized

    latest = client.get("/v1/profiles/ha/snapshot", headers=_headers())
    assert latest.status_code == 200
    assert latest.headers["etag"] == etag
    assert "Authorization" in latest.headers["vary"]
    assert "X-API-Key" in latest.headers["vary"]
    assert "stale-if-error=86400" in latest.headers["cache-control"]
    assert client.get(
        "/v1/profiles/ha/snapshot", headers={**_headers(), "If-None-Match": etag}
    ).status_code == 304
    assert client.get(
        "/v1/profiles/ha/snapshot",
        headers={**_headers(), "If-None-Match": f'"other", W/{etag}'},
    ).status_code == 304

    missing_precondition = client.put("/v1/profiles/ha/snapshot", headers=_headers(), json=_manifest())
    assert missing_precondition.status_code == 428
    identical = client.put(
        "/v1/profiles/ha/snapshot", headers={**_headers(), "If-Match": etag}, json=_manifest()
    )
    assert identical.status_code == 200
    assert identical.json()["gateway_version"] == 1

    updated_manifest = _manifest(source_revision="ha@two")
    updated = client.put(
        "/v1/profiles/ha/snapshot",
        headers={**_headers(), "If-Match": f'"other", {etag}'},
        json=updated_manifest,
    )
    assert updated.status_code == 200
    assert updated.json()["gateway_version"] == 2
    assert client.put(
        "/v1/profiles/ha/snapshot", headers={**_headers(), "If-Match": etag}, json=updated_manifest
    ).status_code == 412

    history = client.get("/v1/profiles/ha/snapshot/1", headers=_headers())
    assert history.status_code == 200
    assert history.json()["source_revision"] == "ha@one"
    assert history.headers["cache-control"].endswith("immutable")
    assert history.headers["content-location"] == "/v1/profiles/ha/snapshot/1"
    assert updated.headers["etag"] != etag

    restored = client.put(
        "/v1/profiles/ha/snapshot",
        headers={**_headers(), "If-Match": updated.headers["etag"]},
        json=_manifest(),
    )
    assert restored.status_code == 200
    assert restored.json()["gateway_version"] == 3
    assert restored.headers["etag"] not in {etag, updated.headers["etag"]}


def test_profile_registration_rejects_non_strict_json(monkeypatch):
    _configure(monkeypatch)
    headers = {**_headers(), "If-None-Match": "*", "Content-Type": "application/json"}
    duplicate = client.put(
        "/v1/profiles/ha/snapshot",
        headers=headers,
        content=b'{"schema_version":1,"schema_version":1}',
    )
    assert duplicate.status_code == 400
    nonfinite = client.put(
        "/v1/profiles/ha/snapshot",
        headers=headers,
        content=b'{"schema_version":NaN}',
    )
    assert nonfinite.status_code == 400
    oversized = client.put(
        "/v1/profiles/ha/snapshot",
        headers=headers,
        content=b" " * 1_048_577,
    )
    assert oversized.status_code == 413


def test_profile_apis_require_consumer_identity_and_namespace_acl(monkeypatch):
    _configure(monkeypatch, namespaces=["ha", "myai"])
    assert _register().status_code == 200
    assert client.get("/v1/profiles/ha/snapshot", headers=_headers(LEGACY)).status_code == 401
    assert client.get("/v1/profiles/ha/snapshot").status_code == 401

    cross = _manifest(namespace="myai")
    response = client.put(
        "/v1/profiles/pi/snapshot",
        headers={**_headers(), "If-None-Match": "*"},
        json=cross,
    )
    assert response.status_code == 403
    assert response.headers["cache-control"] == "no-store"


def test_strict_routes_reject_alias_cloud_and_mixed_pool(monkeypatch):
    cloud = {
        "name": "cloud-model",
        "provider": "cloud-test",
        "provider_model_id": "native-cloud-id",
        "pricing": {"input": 1.0, "output": 2.0},
    }
    _configure(
        monkeypatch,
        models=[cloud],
        extra={"providers": {"cloud-test": {"base_url": "https://cloud.invalid/v1", "api_key": "secret"}}},
    )
    assert _register(_manifest(route="testlocal")).status_code == 422
    assert _register(_manifest(route="cloud-model")).status_code == 422

    pooled = {
        "name": "mixed-model",
        "pool": "mixed",
        "provider_model_id": "mixed-native",
        "pricing": {"input": 1.0, "output": 2.0},
    }
    _configure(
        monkeypatch,
        models=[pooled],
        extra={
            "providers": {"cloud-test": {"base_url": "https://cloud.invalid/v1", "api_key": "secret"}},
            "pools": {"mixed": ["omlx", "cloud-test"]},
        },
    )
    assert _register(_manifest(route="mixed-model")).status_code == 422


def test_local_profile_rejects_non_loopback_omlx_endpoint(monkeypatch):
    _configure(
        monkeypatch,
        extra={"providers": {"omlx": {"base_url": "https://cloud.invalid/v1", "api_key": "omlx"}}},
    )
    response = _register()
    assert response.status_code == 422
    assert "trusted loopback" in response.text


def test_local_profile_rejects_cloud_composite_dependency(monkeypatch):
    models = [
        {
            "name": "cloud-vision",
            "provider": "cloud-test",
            "provider_model_id": "cloud-vision-native",
            "vision": True,
            "pricing": {"input": 1.0, "output": 2.0},
        },
        {
            "name": "local-composite",
            "provider": "omlx",
            "omlx_id": "local-composite-native",
            "vision": True,
            "pricing_status": "unmetered",
            "composite": {
                "text_model": "test-local",
                "vision_model": "cloud-vision",
                "image_handling": "extract_then_answer",
            },
        },
    ]
    _configure(
        monkeypatch,
        models=models,
        extra={"providers": {"cloud-test": {"base_url": "https://cloud.invalid/v1", "api_key": "secret"}}},
    )
    response = _register(_manifest(route="local-composite", vision="local-composite"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_profile_route"


def test_local_route_with_configured_cloud_model_fallback_is_rejected(monkeypatch):
    cloud = {
        "name": "cloud-model",
        "provider": "cloud-test",
        "provider_model_id": "native-cloud-id",
        "pricing": {"input": 1.0, "output": 2.0},
    }
    _configure(
        monkeypatch,
        models=[cloud],
        extra={
            "providers": {"cloud-test": {"base_url": "https://cloud.invalid/v1", "api_key": "secret"}},
            "model_fallbacks": {"test-local-upstream": "native-cloud-id"},
        },
    )
    response = _register()
    assert response.status_code == 422
    assert "non-local provider" in response.text


@pytest.mark.parametrize(
    ("locality", "credential_policy"),
    [
        ("local_only", "gateway_managed"),
        ("local_only", "consumer_byok"),
        ("cloud_explicit", "gateway_local"),
    ],
)
def test_invalid_locality_credential_combinations_are_rejected(
    monkeypatch, locality, credential_policy,
):
    _configure(monkeypatch)
    response = _register(_manifest(locality=locality, credential_policy=credential_policy))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_profile_manifest"


def test_contracted_cloud_profile_is_discoverable_but_fails_closed(monkeypatch):
    cloud = {
        "name": "cloud-model",
        "provider": "cloud-test",
        "provider_model_id": "native-cloud-id",
        "pricing": {"input": 1.0, "output": 2.0},
    }
    _configure(
        monkeypatch,
        models=[cloud],
        extra={"providers": {"cloud-test": {"base_url": "https://cloud.invalid/v1", "api_key": "secret"}}},
    )
    response = _register(_manifest(route="cloud-model", locality="cloud_explicit", credential_policy="consumer_byok"))
    assert response.status_code == 200
    assert response.json()["profiles"][0]["executable"] is False
    invocation = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "profile:ha/automatic-local", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert invocation.status_code == 403
    assert invocation.json()["error"]["type"] == "invalid_request_error"
    assert invocation.json()["error"]["code"] == "profile_execution_unavailable"


@pytest.mark.parametrize(
    ("path", "payload", "expected_model_field"),
    [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, "max_tokens"),
        ("/v1/responses", {"input": "hi"}, "max_tokens"),
        ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}]}, "max_tokens"),
    ],
)
def test_profile_execution_resolves_all_protocols_and_applies_defaults(monkeypatch, path, payload, expected_model_field):
    _configure(monkeypatch)
    assert _register().status_code == 200
    captured = {}

    async def stop_after_profile(_request, body, requested_model, info, _endpoint, error_factory):
        captured.update(body)
        captured["resolved_model"] = requested_model
        return body, requested_model, info, error_factory(418, "invalid_request_error", "captured")

    monkeypatch.setattr(server, "_apply_chat_vision_fallback", stop_after_profile)
    response = client.post(path, headers=_headers(), json={"model": "profile:ha/automatic-local", **payload})
    assert response.status_code == 418
    assert captured["resolved_model"] == "test-local"
    assert captured["temperature"] == 0.2
    assert captured[expected_model_field] == 123
    if path == "/v1/chat/completions":
        audit = ledger.recent(limit=1)[0]
        assert audit["profile_id"] == "ha/automatic-local"
        assert audit["profile_namespace"] == "ha"
        assert audit["profile_version"] == 1


def test_profile_reasoning_default_only_applies_when_omitted():
    omitted = {}
    server._apply_profile_defaults(omitted, {"reasoning_effort": "off"}, "openai_chat")
    assert omitted["reasoning_effort"] == "off"
    explicit = {"reasoning": {"effort": "high"}}
    server._apply_profile_defaults(explicit, {"reasoning_effort": "off"}, "openai_chat")
    assert "reasoning_effort" not in explicit


def test_explicit_controls_override_profile_defaults(monkeypatch):
    _configure(monkeypatch)
    assert _register().status_code == 200
    captured = {}

    async def stop_after_profile(_request, body, requested_model, info, _endpoint, error_factory):
        captured.update(body)
        return body, requested_model, info, error_factory(418, "invalid_request_error", "captured")

    monkeypatch.setattr(server, "_apply_chat_vision_fallback", stop_after_profile)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "profile:ha/automatic-local",
            "messages": [],
            "temperature": 0.9,
            "max_completion_tokens": 77,
            "reasoning_effort": "off",
        },
    )
    assert response.status_code == 418
    assert captured["temperature"] == 0.9
    assert captured["max_completion_tokens"] == 77
    assert "max_tokens" not in captured
    assert captured["reasoning_effort"] == "off"


def test_local_vision_route_is_selected_instead_of_global_cloud_fallback(monkeypatch):
    models = [
        {
            "name": "local-vision",
            "provider": "omlx",
            "omlx_id": "local-vision-native",
            "vision": True,
            "pricing_status": "unmetered",
        },
        {
            "name": "cloud-vision",
            "provider": "cloud-test",
            "provider_model_id": "cloud-vision-native",
            "vision": True,
            "pricing": {"input": 1.0, "output": 2.0},
        },
    ]
    _configure(
        monkeypatch,
        models=models,
        extra={"providers": {"cloud-test": {"base_url": "https://cloud.invalid/v1", "api_key": "secret"}}},
    )
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "cloud-vision")
    assert _register(_manifest(vision="local-vision")).status_code == 200
    captured = {}

    async def stop_after_profile(_request, body, requested_model, info, _endpoint, error_factory):
        captured["model"] = requested_model
        captured["provider"] = info.provider
        return body, requested_model, info, error_factory(418, "invalid_request_error", "captured")

    monkeypatch.setattr(server, "_apply_chat_vision_fallback", stop_after_profile)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "profile:ha/automatic-local",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}]}],
        },
    )
    assert response.status_code == 418
    assert captured == {"model": "local-vision", "provider": "omlx"}


def test_profile_vision_without_declared_route_never_uses_global_fallback(monkeypatch):
    cloud_vision = {
        "name": "cloud-vision",
        "provider": "cloud-test",
        "provider_model_id": "cloud-vision-native",
        "vision": True,
        "pricing": {"input": 1.0, "output": 2.0},
    }
    _configure(
        monkeypatch,
        models=[cloud_vision],
        extra={"providers": {"cloud-test": {"base_url": "https://cloud.invalid/v1", "api_key": "secret"}}},
    )
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "cloud-vision")
    assert _register().status_code == 200
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "profile:ha/automatic-local",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}]}],
        },
    )
    assert response.status_code == 403
    assert "does not define a vision route" in response.text


def test_cross_namespace_invocation_direct_denial_and_legacy_compatibility(monkeypatch):
    _configure(monkeypatch, namespaces=["ha", "myai"])
    assert _register().status_code == 200
    cross = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "profile:pi/automatic-local", "messages": []},
    )
    assert cross.status_code == 403

    direct = client.post(
        "/v1/chat/completions", headers=_headers(), json={"model": "test-local", "messages": []}
    )
    assert direct.status_code == 403
    # A legacy credential keeps reaching ordinary model resolution/dispatch.
    legacy = client.post(
        "/v1/chat/completions", headers=_headers(LEGACY), json={"model": "not-a-model", "messages": []}
    )
    assert legacy.status_code == 404


def test_consumer_with_direct_grant_can_use_ordinary_routes(monkeypatch):
    _configure(monkeypatch, allow_direct=True)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "not-a-model", "messages": []},
    )
    assert response.status_code == 404


def test_profile_selector_from_federation_is_rejected_before_resolution(monkeypatch):
    request = SimpleNamespace(state=SimpleNamespace(federation_source="peer"))
    body = {"model": "profile:ha/automatic-local"}
    response = server._prepare_profile_execution(
        request, "/v1/chat/completions", body, protocol="openai_chat", has_image=False
    )
    assert response.status_code == 403


def test_identical_manifest_reregistration_repairs_binding_drift(monkeypatch):
    config = _configure(monkeypatch)
    assert _register().status_code == 200
    current = client.get("/v1/profiles/ha/snapshot", headers=_headers())
    config["providers"] = {"omlx": {"base_url": "http://127.0.0.1:9112/v1", "api_key": "omlx"}}
    providers._models = None
    drifted = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "profile:ha/automatic-local", "messages": []},
    )
    assert drifted.status_code == 409

    rebound = client.put(
        "/v1/profiles/ha/snapshot",
        headers={**_headers(), "If-Match": current.headers["etag"]},
        json=_manifest(),
    )
    assert rebound.status_code == 200
    assert rebound.json()["gateway_version"] == 2

    async def stop_after_profile(_request, body, requested_model, info, _endpoint, error_factory):
        return body, requested_model, info, error_factory(418, "invalid_request_error", "captured")

    monkeypatch.setattr(server, "_apply_chat_vision_fallback", stop_after_profile)
    repaired = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "profile:ha/automatic-local", "messages": []},
    )
    assert repaired.status_code == 418


def test_binding_drift_requires_reregistration(monkeypatch):
    config = _configure(monkeypatch)
    assert _register().status_code == 200
    config["models"] = [{
        "name": "test-local",
        "provider": "omlx",
        "omlx_id": "changed-upstream",
        "pricing_status": "unmetered",
    }]
    providers._models = None
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "profile:ha/automatic-local", "messages": []},
    )
    assert response.status_code == 409, response.text


def test_consumer_key_file_must_be_private_regular_and_non_symlinked(monkeypatch, tmp_path):
    key = tmp_path / "consumer.key"
    key.write_text(TOKEN)
    key.chmod(0o644)
    config = {
        "providers": {},
        "auth": {"consumer_credentials": [{
            "id": "ha-runtime",
            "consumer": "ha",
            "key_file": str(key),
            "namespaces": ["ha"],
            "permissions": ["profiles:read"],
            "allow_direct_models": False,
        }]},
    }
    monkeypatch.setattr(providers, "_config", config)
    monkeypatch.setattr(providers, "_models", None)
    assert client.get("/v1/profiles/ha/snapshot", headers=_headers()).status_code == 503

    key.chmod(0o600)
    assert client.get("/v1/profiles/ha/snapshot", headers=_headers()).status_code == 404
    link = tmp_path / "linked.key"
    link.symlink_to(key)
    config["auth"]["consumer_credentials"][0]["key_file"] = str(link)
    assert client.get("/v1/profiles/ha/snapshot", headers=_headers()).status_code == 503


def test_overlapping_inbound_credential_classes_fail_closed_everywhere(monkeypatch):
    config = _configure(monkeypatch)
    config["auth"]["client_keys"] = [TOKEN]
    assert client.get("/v1/models", headers=_headers()).status_code == 503

    config["auth"]["client_keys"] = [LEGACY]
    config["auth"]["admin_keys"] = [TOKEN]
    assert client.get("/admin/api/status", headers=_headers()).status_code == 503

    config["auth"].pop("admin_keys")
    config["federation"] = {
        "node_id": "main",
        "peers": {"edge": {"base_url": "https://edge.invalid", "api_key": TOKEN}},
    }
    with pytest.raises(ValueError, match="overlaps"):
        validate_credential_separation()
    assert client.get(
        "/v1/federation/catalog",
        headers={"Authorization": f"Bearer {TOKEN}", "X-Model-Gateway-Source": "edge"},
    ).status_code == 503


def test_consumer_auth_does_not_degrade_to_anonymous_after_config_removal(monkeypatch):
    config = _configure(monkeypatch)
    config["auth"].pop("client_keys")
    assert client.get("/v1/models", headers=_headers()).status_code == 200
    config.clear()
    providers._models = None
    assert client.get("/v1/models").status_code == 503


def test_rejected_registry_quota_write_preserves_last_good_snapshot(monkeypatch):
    _configure(monkeypatch)
    first = _register()
    assert first.status_code == 200
    monkeypatch.setattr(profiles, "_MAX_NAMESPACE_BYTES", 1)
    rejected = client.put(
        "/v1/profiles/ha/snapshot",
        headers={**_headers(), "If-Match": first.headers["etag"]},
        json=_manifest(source_revision="ha@too-large"),
    )
    assert rejected.status_code == 413
    latest = client.get("/v1/profiles/ha/snapshot", headers=_headers())
    assert latest.status_code == 200
    assert latest.json()["gateway_version"] == 1


def test_registry_file_is_private(monkeypatch):
    _configure(monkeypatch)
    assert _register().status_code == 200
    path = profiles.registry_path()
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.with_name(path.name + ".lock").stat().st_mode & 0o777 == 0o600
