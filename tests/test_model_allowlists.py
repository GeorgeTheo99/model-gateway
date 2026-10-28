"""Per-credential direct-model allowlists
(docs/adr/0003-scoped-access-and-lightweight-local-models.md sections 1 and 2)."""

from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src import admin, auth, config_io, discovery, federation, providers, server

client = TestClient(server.app)
TOKEN = "weather-secret"
LEGACY = "legacy-secret"
PROFILE_ONLY = "profile-only-secret"
ADMIN = "admin-secret"
IMAGE = [{"role": "user", "content": [
    {"type": "text", "text": "what is this?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
]}]

CATALOG = [
    {"name": "allowed-a", "alias": "a-alias", "provider": "cloud-test", "provider_model_id": "a-native",
     "pricing": {"input": 1.0, "output": 2.0}},
    {"name": "blocked-b", "provider": "cloud-test", "provider_model_id": "b-native",
     "pricing": {"input": 1.0, "output": 2.0}},
    {"name": "glm-x-fw", "base_model": "glm-x", "family": "glm", "provider": "cloud-test", "provider_model_id": "glm-x-native",
     "pricing": {"input": 1.0, "output": 2.0}},
    {"name": "vision-helper", "provider": "cloud-test", "provider_model_id": "helper-native", "vision": True,
     "pricing": {"input": 1.0, "output": 2.0}},
]


def _configure(monkeypatch, models=("allowed-a", "glm-x"), *, credential=None):
    entry = {
        "id": "weather-runtime", "consumer": "weather", "key": TOKEN, "namespaces": ["weather"],
        "permissions": ["profiles:read", "profiles:invoke"], "allow_direct_models": True,
    }
    if models is not None:
        entry["models"] = list(models)
    entry.update(credential or {})
    config = {
        "providers": {"cloud-test": {"base_url": "https://cloud.invalid/v1", "api_key": "secret"}},
        "models": deepcopy(CATALOG),
        "auth": {
            "client_keys": [LEGACY],
            "admin_keys": [ADMIN],
            "consumer_credentials": [entry, {
                "id": "ha-runtime", "consumer": "ha", "key": PROFILE_ONLY, "namespaces": ["ha"],
                "permissions": ["profiles:read", "profiles:invoke"],
            }],
        },
    }
    monkeypatch.setattr(providers, "_config", config)
    monkeypatch.setattr(providers, "_models", None)
    monkeypatch.delenv("GATEWAY_VISION_FALLBACK", raising=False)
    return config


@pytest.fixture
def dispatched(monkeypatch):
    """Capture what reaches an upstream instead of calling it."""
    calls = []

    async def passthrough(endpoint, body, headers, provider="", request=None, **_kwargs):
        calls.append({"endpoint": endpoint, "model": body.get("model"),
                      "fallback_disabled": getattr(request.state, "disable_model_fallback", False)})
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(server, "_passthrough_sync", passthrough)
    return calls


def _chat(model, token=TOKEN, messages=None):
    return client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"},
                       json={"model": model, "messages": messages or [{"role": "user", "content": "hi"}]})


# ── credential validation ───────────────────────────────────────────────────


@pytest.mark.parametrize("credential, message", [
    ({"allow_direct_models": False}, "requires allow_direct_models"),
    ({"permissions": ["profiles:invoke", "profiles:write"]}, "cannot be combined"),
    ({"permissions": ["models:register"], "providers": ["cloud-test"]}, "cannot be combined"),
    ({"permissions": ["providers:manage"], "providers": ["cloud-test"]}, "cannot be combined"),
    ({"models": []}, "non-empty list"),
    ({"models": ["a", "a"]}, "non-empty list"),
    ({"models": "allowed-a"}, "non-empty list"),
    ({"models": [" "]}, "non-empty list"),
])
def test_invalid_allowlists_are_rejected(monkeypatch, credential, message):
    _configure(monkeypatch, credential=credential)
    with pytest.raises(ValueError, match=message):
        auth._consumer_credentials()
    with pytest.raises(auth.AuthConfigError):
        auth.validate_credential_separation()


def test_principal_carries_its_allowlist(monkeypatch):
    _configure(monkeypatch)
    consumers, _keys, _admins = auth.validate_credential_separation()
    principals = {principal.credential_id: principal for _token, principal in consumers}
    assert principals["weather-runtime"].models == frozenset({"allowed-a", "glm-x"})
    assert principals["ha-runtime"].models is None


# ── what a listed name permits ──────────────────────────────────────────────


def test_listed_models_their_aliases_and_upstream_ids_are_served(monkeypatch, dispatched):
    _configure(monkeypatch)
    for name in ("allowed-a", "a-alias", "a-native"):
        assert _chat(name).status_code == 200, name
    assert [call["model"] for call in dispatched] == ["a-native"] * 3
    assert all(call["fallback_disabled"] for call in dispatched)


def test_a_listed_alias_permits_its_entry(monkeypatch, dispatched):
    _configure(monkeypatch, models=["a-alias"])
    assert _chat("allowed-a").status_code == 200


@pytest.mark.parametrize("path, envelope", [
    ("/v1/chat/completions", "error"),
    ("/v1/responses", "error"),
    ("/v1/messages", "error"),
    ("/v1/images/generations", "error"),
    ("/v1/videos", "error"),
])
@pytest.mark.parametrize("model", ["blocked-b", "no-such-model", "glm-x-fw", "peer/remote", 42])
def test_other_models_are_denied_identically_on_every_endpoint(monkeypatch, path, envelope, model):
    _configure(monkeypatch)
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "input": "hi", "max_tokens": 8}
    response = client.post(path, headers={"Authorization": f"Bearer {TOKEN}"}, json=body)
    assert response.status_code == 403
    error = response.json()[envelope]
    assert error["code"] == "direct_model_access_denied"
    assert error["message"] == "Consumer principal is not authorized to invoke this model"


def test_every_model_taking_endpoint_is_covered():
    """A new inference endpoint must be added to the parametrized denial test above."""
    writes = {route.path for route in server.app.routes
              if route.path.startswith("/v1") and {"POST", "PUT"} & (getattr(route, "methods", None) or set())}
    # The local federation paths admit only authenticated peer forwards, never consumer credentials.
    assert writes == {"/v1/chat/completions", "/v1/responses", "/v1/messages",
                      "/v1/images/generations", "/v1/videos",
                      "/v1/profiles/{namespace}/snapshot", "/v1/catalog/packs/{namespace}",
                      "/v1/federation/local/chat/completions", "/v1/federation/local/responses",
                      "/v1/federation/local/messages"}


def test_names_match_exactly(monkeypatch, dispatched):
    _configure(monkeypatch)
    for name in ("Allowed-A", " allowed-a", "allowed-a "):
        assert _chat(name).status_code == 403, name
    assert dispatched == []


def test_a_route_named_after_its_base_model_is_rerouted_within_the_base(monkeypatch, dispatched):
    config = _configure(monkeypatch, models=["glm-y"])
    config["models"] += [
        {"name": "glm-y", "base_model": "glm-y", "family": "glm", "provider": "unconfigured-provider",
         "provider_model_id": "glm-y-upstream", "pricing": {"input": 1.0, "output": 2.0}},
        {"name": "glm-y-fw", "base_model": "glm-y", "family": "glm", "provider": "cloud-test",
         "provider_model_id": "glm-y-native", "pricing": {"input": 1.0, "output": 2.0}},
    ]
    assert _chat("glm-y").status_code == 200
    assert dispatched[-1]["model"] == "glm-y-native"
    assert _chat("glm-y-fw").status_code == 403


def test_profiles_are_not_limited_by_the_allowlist(monkeypatch):
    _configure(monkeypatch)
    resolved = []

    def resolve_execution(selector, *, protocol, has_image):
        resolved.append(selector)
        raise server.profiles.ProfileError(418, "captured", "captured")

    monkeypatch.setattr(server.profiles, "resolve_execution", resolve_execution)
    assert _chat("profile:weather/automatic").status_code == 418
    assert _chat("profile:ha/automatic").status_code == 403  # namespace grants still apply
    assert resolved == ["profile:weather/automatic"]


def test_a_listed_base_model_permits_its_name_not_its_routes(monkeypatch, dispatched):
    _configure(monkeypatch)
    assert _chat("glm-x").status_code == 200
    assert dispatched[-1]["model"] == "glm-x-native"
    assert _chat("glm-x-fw").status_code == 403


def test_a_retired_name_is_served_only_by_a_permitted_successor(monkeypatch, dispatched):
    _configure(monkeypatch, models=["old-a"])
    monkeypatch.setattr(server, "retirement_redirect", lambda name: "blocked-b" if name == "old-a" else None)
    assert _chat("old-a").status_code == 403
    assert dispatched == []

    _configure(monkeypatch, models=["old-a", "allowed-a"])
    monkeypatch.setattr(server, "retirement_redirect", lambda name: "allowed-a" if name == "old-a" else None)
    response = _chat("old-a")
    assert response.status_code == 200
    assert dispatched[-1]["model"] == "a-native"


def test_a_successor_appearing_after_authorization_is_not_served(monkeypatch, dispatched):
    _configure(monkeypatch, models=["old-a"])
    answers = iter([None, "blocked-b"])  # authorization sees no redirect; the catalog then changes
    monkeypatch.setattr(server, "retirement_redirect", lambda name: next(answers))
    assert _chat("old-a").status_code == 404
    assert dispatched == []


def test_pack_rows_count_only_for_their_own_namespace(monkeypatch):
    rows = {
        "pack-model": {"name": "pack-model", "origin": {"layer": "pack", "namespace": "weather"}},
        "other-pack": {"name": "other-pack", "origin": {"layer": "pack", "namespace": "ha"}},
    }
    monkeypatch.setattr(providers, "_models", rows)
    allowed = frozenset({"pack-model", "other-pack"})
    assert providers.direct_model_permitted("pack-model", allowed, frozenset({"weather"}))
    assert not providers.direct_model_permitted("other-pack", allowed, frozenset({"weather"}))


def test_unrestricted_credentials_are_unchanged(monkeypatch, dispatched):
    _configure(monkeypatch, models=None)
    assert _chat("blocked-b").status_code == 200
    assert _chat("blocked-b", token=LEGACY).status_code == 200
    assert not any(call["fallback_disabled"] for call in dispatched)


# ── vision fallback ─────────────────────────────────────────────────────────

def test_images_to_a_text_only_allowed_model_are_served_by_it_as_notes(monkeypatch, dispatched):
    _configure(monkeypatch)
    response = _chat("allowed-a", messages=IMAGE)
    assert response.status_code == 200
    assert [call["model"] for call in dispatched] == ["a-native"]


# ── federation ──────────────────────────────────────────────────────────────


def test_imported_models_must_be_listed(monkeypatch):
    _configure(monkeypatch, models=["peer/listed"])
    forwarded = []

    async def forward(request, path, body):
        forwarded.append(body["model"])
        return JSONResponse({"ok": True})

    monkeypatch.setattr(server, "_forward_imported_if_known", forward)
    assert _chat("peer/listed").status_code == 200
    assert _chat("peer/other").status_code == 403
    assert forwarded == ["peer/listed"]


# ── listings ────────────────────────────────────────────────────────────────


def _ids(path, token):
    response = client.get(path, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    return {row.get("id") or row.get("name") for row in response.json().get("data") or response.json()["models"]}


def test_listings_show_only_what_a_restricted_credential_may_call(monkeypatch):
    _configure(monkeypatch)
    assert _ids("/v1/models", TOKEN) == {"allowed-a", "a-alias", "a-native", "glm-x"}
    assert _ids("/v1/models/canonical", TOKEN) == {"allowed-a", "glm-x"}
    assert _ids("/v1/debug/thinking", TOKEN) == {"allowed-a", "glm-x"}


def test_canonical_presets_reference_only_permitted_models(monkeypatch):
    _configure(monkeypatch)
    document = {"llm": [], "auto_models": {"cloud": {"model": "blocked-b"}},
                "model_presets": {"version": 1, "presets": {
                    "fast": {"label": "Fast", "cloud": {"model": "allowed-a"}},
                    "deep": {"label": "Deep", "cloud": {"model": "blocked-b"}},
                }}}
    monkeypatch.setattr(server.catalog, "load_model_info_document", lambda _path: deepcopy(document))
    body = client.get("/v1/models/canonical", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert body["auto_models"] == {}
    assert set(body["model_presets"].get("presets", {})) == {"fast"}
    full = client.get("/v1/models/canonical", headers={"Authorization": f"Bearer {LEGACY}"}).json()
    assert full["auto_models"]["cloud"]["model"] == "blocked-b"


def test_base_model_rows_carry_their_route_capabilities(monkeypatch):
    _configure(monkeypatch)
    rows = client.get("/v1/models/canonical", headers={"Authorization": f"Bearer {TOKEN}"}).json()["data"]
    glm = next(row for row in rows if row["id"] == "glm-x")
    assert glm["available"] is True and glm["scope"] == "cloud"


def test_other_callers_see_the_full_listing(monkeypatch):
    _configure(monkeypatch)
    full = {"allowed-a", "blocked-b", "glm-x-fw", "vision-helper", "test-local"}
    for token in (LEGACY, PROFILE_ONLY):
        assert full <= _ids("/v1/models/canonical", token)
        assert full <= _ids("/v1/models", token)


def test_imported_rows_are_listed_only_when_permitted(monkeypatch):
    _configure(monkeypatch, models=["allowed-a", "peer/listed"])
    rows = [{"id": "peer/listed", "object": "model"}, {"id": "peer/other", "object": "model"}]

    class Manager:
        def imported_rows(self):
            return deepcopy(rows)

    monkeypatch.setattr(federation, "manager", lambda: Manager())
    assert _ids("/v1/models", TOKEN) == {"allowed-a", "a-alias", "a-native", "peer/listed"}
    assert {"peer/listed", "peer/other"} <= _ids("/v1/models", LEGACY)


def test_federation_export_is_not_filtered(monkeypatch):
    _configure(monkeypatch)
    assert {"allowed-a", "blocked-b"} <= {row["id"] for row in server._direct_model_rows()}


# ── admin status, management and discovery ─────────────────────────────────


def test_admin_status_reports_legacy_keys_and_unmatched_names(monkeypatch):
    _configure(monkeypatch, models=["allowed-a", "glm-x", "typo-model"])
    status = client.get("/admin/api/status", headers={"Authorization": f"Bearer {ADMIN}"}).json()["auth"]
    assert status["deprecated_legacy_client_keys"] == 1
    assert status["unmatched_model_allowlists"] == {"weather-runtime": ["typo-model"]}


def test_listed_retired_names_are_matched(monkeypatch):
    _configure(monkeypatch, models=["allowed-a", "old-a"])
    providers._load_models()
    monkeypatch.setattr(providers, "_retired", {"old-a": "allowed-a"})
    assert providers.unmatched_model_names(frozenset({"allowed-a", "old-a", "typo"})) == ["typo"]


def test_admin_status_survives_a_broken_catalog(monkeypatch):
    _configure(monkeypatch)

    def broken(_names):
        raise ValueError("catalog failed to load")

    monkeypatch.setattr(admin, "unmatched_model_names", broken)
    response = client.get("/admin/api/status", headers={"Authorization": f"Bearer {ADMIN}"})
    assert response.status_code == 200
    assert response.json()["auth"]["unmatched_model_allowlists"] == {}


def test_consumer_http_api_validates_models(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    response = client.post("/admin/api/consumers", headers={"Authorization": f"Bearer {ADMIN}"},
                           json={"consumer": "bot", "role": "runtime", "allow_direct_models": True, "models": "x"})
    assert response.status_code == 400
    assert "models must be a list of strings" in response.text


def test_consumer_api_creates_restricted_runtime_credentials(tmp_path, monkeypatch):
    for name in ("MODEL_GATEWAY_CLIENT_KEYS", "MODEL_GATEWAY_CLIENT_KEYS_FILE", "MODEL_GATEWAY_ADMIN_KEY"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "gateway" / "config.yaml"
    path.parent.mkdir()
    path.write_text("auth:\n  admin_keys: [admin-token]\n  client_keys: [client-token]\nproviders: {}\n")
    monkeypatch.setattr(providers, "CONFIG_PATH", path)
    monkeypatch.setattr(config_io, "CONFIG_PATH", path)
    providers.reload()

    row = config_io.add_consumer_credential("weather", "runtime", allow_direct_models=True, models=["allowed-a", "glm-x"])
    assert row["models"] == ["allowed-a", "glm-x"]
    assert config_io.add_consumer_credential(
        "weather", "runtime", allow_direct_models=True, models=["glm-x", "allowed-a"])["status"] == "unchanged"
    with pytest.raises(ValueError, match="different settings"):
        config_io.add_consumer_credential("weather", "runtime", allow_direct_models=True, models=["glm-x"])
    with pytest.raises(ValueError, match="different settings"):
        config_io.add_consumer_credential("weather", "runtime", allow_direct_models=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        config_io.add_consumer_credential("weather", "deployer", allow_direct_models=True, models=["allowed-a"])
    assert [c["id"] for c in config_io.list_consumer_credentials()] == ["weather-runtime"]
    assert discovery.endpoint_document()["consumers"]["weather-runtime"]["models"] == ["allowed-a", "glm-x"]
