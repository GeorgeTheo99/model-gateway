"""The shipped catalog layer and the owner's precedence over it (ADR 0002)."""

import json

import pytest
import yaml

import src.providers as providers
from src import catalog, onboarding, shipped

BASE_URL = "https://api.example.test/v1"


def _profile(models, *, retire=None, profile_id="example-curated"):
    data = {
        "schema_version": 1,
        "id": profile_id,
        "auto_add": True,
        "provider": {"id": "example", "base_url": BASE_URL, "protocol": "openai"},
        "models": [
            {"provider": "example", "provider_model_id": f"up/{m['name']}", "context": 1000,
             "pricing": {"input": 1.0, "output": 2.0}, **m}
            for m in models
        ],
    }
    if retire:
        data["retire"] = retire
    return onboarding.validate_profile(data)


def _entries(tmp_path, owner_rows, profiles, config=None, ready=lambda profile, config: True):
    path = tmp_path / "model-info.json"
    path.write_text(json.dumps({"llm": owner_rows, "allow_empty": True}))
    layer = shipped.layer(config or {}, profiles=profiles, ready=ready)
    entries = catalog.load_catalog_entries(
        path, overlay=(config or {}).get("models", []), shipped=layer.entries, shipped_names=layer.names,
        excluded=layer.hidden | frozenset(layer.retired),
    )
    return {entry["name"]: entry for entry in entries}, layer


def test_shipped_models_load_by_reference(tmp_path):
    models, _ = _entries(tmp_path, [], {"p": _profile([{"name": "m1"}, {"name": "m2"}])})
    assert list(models) == ["m1", "m2"]
    assert models["m1"]["origin"] == {"layer": "shipped", "profile": "p"}


def test_not_ready_provider_loads_nothing(tmp_path):
    models, layer = _entries(tmp_path, [], {"p": _profile([{"name": "m1"}])}, ready=lambda p, c: False)
    assert models == {} and layer.names == {"m1"}


def test_owner_override_wins_per_field_and_keeps_shipped_updates(tmp_path):
    profiles = {"p": _profile([{"name": "m1", "context": 2000}])}
    owner = [{"name": "m1", "pricing": {"input": 0.5, "output": 1.0}, "shipped_base": {"pricing": {"input": 9, "output": 9}}}]
    models, _ = _entries(tmp_path, owner, profiles)
    m1 = models["m1"]
    assert m1["pricing"] == {"input": 0.5, "output": 1.0}
    assert m1["context"] == 2000  # shipped change still flows
    assert m1["origin"]["overridden"] == ["pricing"]
    assert m1["origin"]["stale_overrides"] == ["pricing"]  # shipped pricing changed since the override
    assert "shipped_base" not in m1


def test_legacy_full_copy_merges_to_the_shipped_row(tmp_path):
    profile = _profile([{"name": "m1"}])
    models, _ = _entries(tmp_path, [dict(profile["models"][0])], {"p": profile})
    assert models["m1"]["origin"]["overridden"] == []


def test_sparse_override_of_unloaded_shipped_model_is_dormant(tmp_path):
    owner = [{"name": "m1", "pricing": {"input": 0.5, "output": 1.0}, "shipped_base": {}},
             {"name": "m2", "provider": "example", "provider_model_id": "up/m2"}]
    models, _ = _entries(tmp_path, owner, {"p": _profile([{"name": "m1"}, {"name": "m2"}])},
                         ready=lambda p, c: False)
    assert list(models) == ["m2"]  # a full legacy copy still stands on its own
    assert models["m2"]["origin"] == {"layer": "owner"}


def test_owner_claims_single_identifiers_and_shadows_by_name(tmp_path):
    profiles = {"p": _profile([
        {"name": "glm-fw", "alias": "glm", "alternate_ids": ["glm-5.3", "glm-latest"]},
        {"name": "astra"},
    ])}
    owner = [
        {"name": "glm-local", "provider": "omlx", "omlx_id": "glm-local", "alternate_ids": ["glm-5.3"],
         "pricing_status": "unmetered"},
        {"name": "my-astra", "provider": "omlx", "omlx_id": "my-astra", "alternate_ids": ["astra"],
         "pricing_status": "unmetered"},
    ]
    models, _ = _entries(tmp_path, owner, profiles)
    assert models["glm-fw"]["alternate_ids"] == ["glm-latest"]
    assert models["glm-fw"]["alias"] == "glm"
    assert "astra" not in models  # the owner took its name


@pytest.mark.parametrize("selection,expected", [
    ("none", []), (["m2"], ["m2"]), (["p"], ["m1", "m2"]), ("all", ["m1", "m2"]),
])
def test_selection_mode(tmp_path, selection, expected):
    config = {"catalog": {"shipped": selection}}
    models, _ = _entries(tmp_path, [], {"p": _profile([{"name": "m1"}, {"name": "m2"}])}, config)
    assert list(models) == expected


def test_hidden_and_retired_apply_to_lower_layers_only(tmp_path):
    profiles = {
        "old": _profile([{"name": "other"}], profile_id="old",
                        retire={"models": ["gone", "pinned-one", "keeps"], "replaced_by": {"gone": "new"}}),
        "new": _profile([{"name": "new"}, {"name": "hide-me"}, {"name": "keeps"}], profile_id="new"),
    }
    owner = [{"name": n, "provider": "omlx", "omlx_id": n, "pricing_status": "unmetered"}
             for n in ("gone", "pinned-one", "mine")]
    config = {"catalog": {"hidden": ["hide-me", "mine"], "pinned": ["pinned-one"]}}
    models, layer = _entries(tmp_path, owner, profiles, config)
    # The owner's own rows always load, retired or hidden names or not.
    assert sorted(models) == ["gone", "keeps", "mine", "new", "other", "pinned-one"]
    # "keeps" is still shipped by a profile, so the older retirement does not apply.
    assert layer.retired == {"gone": "new"}


def test_replaced_by_must_name_a_live_model():
    with pytest.raises(onboarding.OnboardingError, match="replaced_by"):
        _profile([{"name": "m1"}], retire={"models": ["old"], "replaced_by": {"other": "m1"}})
    with pytest.raises(onboarding.OnboardingError, match="replaced_by"):
        _profile([{"name": "m1"}], retire={"models": ["a", "b"], "replaced_by": {"a": "b"}})


@pytest.mark.parametrize("raw", [{"shipped": "some"}, {"hidden": "x"}, {"pinned": [""]}, {"extra": 1}, []])
def test_invalid_catalog_settings_are_rejected(raw):
    with pytest.raises(ValueError, match="catalog"):
        shipped.settings({"catalog": raw})


def test_registry_loads_shipped_models_for_a_ready_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(shipped, "_packaged", lambda: {"p": _profile([{"name": "m1"}])})
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"llm": [], "allow_empty": True}))
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", info)
    config = {"providers": {"example": {"base_url": BASE_URL, "api_key": "k"}}}
    providers.CONFIG_PATH.write_text(yaml.safe_dump(config))
    providers.reload()
    assert providers.resolve("m1").provider_model_id == "up/m1"

    for change in ({"api_key": ""}, {"base_url": "https://elsewhere.test/v1"}, {"enabled": False}):
        providers.CONFIG_PATH.write_text(yaml.safe_dump(
            {"providers": {"example": {**config["providers"]["example"], **change}}}))
        providers.reload()
        assert "m1" not in providers._load_models()


def test_a_retired_name_is_served_by_its_successor(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.server as server_module

    profile = _profile([{"name": "new"}], retire={"models": ["old", "gone"], "replaced_by": {"old": "new"}})
    monkeypatch.setattr(shipped, "_packaged", lambda: {"p": profile})
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"llm": [], "allow_empty": True}))
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", info)
    providers.CONFIG_PATH.write_text(yaml.safe_dump({"providers": {"example": {"base_url": BASE_URL, "api_key": "k"}}}))
    providers.reload()
    sent = []

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        sent.append(body["model"])
        return server_module.JSONResponse(status_code=200, content={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)
    with TestClient(server_module.app) as client:
        response = client.post("/v1/chat/completions", json={"model": "old", "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 200, response.text
        assert response.headers["Deprecation"] == "true"
        assert response.headers["X-Model-Gateway-Model-Redirect"] == "old -> new"
        assert sent == ["up/new"]
        assert client.post("/v1/chat/completions", json={
            "model": "gone", "messages": [{"role": "user", "content": "hi"}]}).status_code == 404
        listed = {row["id"] for row in client.get("/v1/models").json()["data"]}
        assert "new" in listed and "old" not in listed


def _registry_with(tmp_path, monkeypatch, owner_rows, profile):
    monkeypatch.setattr(shipped, "_packaged", lambda: {"p": profile})
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"llm": owner_rows, "allow_empty": True}))
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", info)
    from src import config_io
    monkeypatch.setattr(config_io, "MODEL_INFO_PATH", info)
    monkeypatch.setattr(config_io, "MODEL_INFO_SOURCE_PATH", None)
    providers.CONFIG_PATH.write_text(yaml.safe_dump({"providers": {"example": {"base_url": BASE_URL, "api_key": "k"}}}))
    providers.reload()
    return info


def test_an_override_a_release_invalidates_leaves_the_model_unavailable(tmp_path, monkeypatch):
    # Valid when saved against an optional-thinking row; the release made thinking always-on.
    profile = _profile([{"name": "m1", "thinking": "always", "thinking_levels": ["low", "high"]}])
    _registry_with(tmp_path, monkeypatch, [{"name": "m1", "thinking_levels": ["off", "low"], "shipped_base": {}}], profile)
    entry = providers._load_models()["m1"]
    assert "off" in entry["origin"]["override_error"]
    assert providers.resolve("m1") is None  # fail closed, never the release's row instead
    assert providers.model_availability("m1")["reason"] == "override_invalid"
    assert "m1" in providers.catalog_status()["override_errors"]


def test_an_owner_route_replaces_the_lower_model_and_keeps_its_identifiers(tmp_path):
    profiles = {"p": _profile([{"name": "cloud", "alias": "c"}, {"name": "glm-fw", "alias": "glm"}])}
    owner = [
        # A different route under a shipped name: the owner's model, nothing merged in.
        {"name": "cloud", "provider": "omlx", "omlx_id": "my-local", "pricing_status": "unmetered"},
        # An override, and another owner model already using the alias the release gives it.
        {"name": "glm-fw", "desc": "mine", "shipped_base": {}},
        {"name": "glm-local", "provider": "omlx", "omlx_id": "glm-local", "alias": "glm", "pricing_status": "unmetered"},
        # An override whose shipped model is gone never becomes a model of its own.
        {"name": "dropped", "context": 5, "shipped_base": {"context": 1}},
    ]
    models, _ = _entries(tmp_path, owner, profiles)
    assert (models["cloud"]["provider"], models["cloud"]["omlx_id"]) == ("omlx", "my-local")
    assert models["cloud"]["origin"] == {"layer": "owner"} and "alias" not in models["cloud"]
    assert models["glm-fw"]["desc"] == "mine" and "alias" not in models["glm-fw"]
    assert models["glm-local"]["alias"] == "glm"
    assert "dropped" not in models


def test_a_failed_lazy_reload_serves_the_last_good_catalog(tmp_path, monkeypatch):
    info = _registry_with(tmp_path, monkeypatch, [], _profile([{"name": "m1"}]))
    assert "m1" in providers._load_models()
    info.write_text("{broken")
    providers.reload()
    assert "m1" in providers._load_models()  # request path keeps serving
    assert providers.catalog_status()["load_error"]
    with pytest.raises(ValueError):
        providers.reload()
        providers.snapshot_registry()  # validation still sees the error


def test_a_failed_reload_keeps_the_new_config_and_credentials(tmp_path, monkeypatch):
    info = _registry_with(tmp_path, monkeypatch, [], _profile([{"name": "m1"}]))
    config = {"providers": {"example": {"base_url": BASE_URL, "api_key": "k"}},
              "auth": {"client_keys": ["old-key"]}}
    providers.CONFIG_PATH.write_text(yaml.safe_dump(config))
    providers.reload()
    assert "m1" in providers._load_models()
    info.write_text("{broken")
    providers.CONFIG_PATH.write_text(yaml.safe_dump({**config, "auth": {"client_keys": ["new-key"]}}))
    providers.reload()
    assert "m1" in providers._load_models()
    assert providers._load_config()["auth"]["client_keys"] == ["new-key"]  # never the old credentials


def test_a_rejected_candidate_never_becomes_the_last_good_catalog(tmp_path, monkeypatch):
    info = _registry_with(tmp_path, monkeypatch, [], _profile([{"name": "m1"}]))
    live = providers.snapshot_registry()
    info.write_text(json.dumps({"llm": [{"name": "candidate", "provider": "omlx", "omlx_id": "c",
                                         "pricing_status": "unmetered"}], "allow_empty": True}))
    providers.reload()
    assert "candidate" in providers.snapshot_registry()[1]  # validated, then rejected by a later check
    providers.restore_registry(live)
    info.write_text("{broken")
    providers.reload()
    assert "candidate" not in providers._load_models() and "m1" in providers._load_models()


def test_review_keep_or_discard(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.server as server_module

    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin")
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    flagged = {"name": "m1", "desc": "mine", "shipped_base": {"desc": None}, "review": True}
    info = _registry_with(tmp_path, monkeypatch, [dict(flagged), {**flagged, "name": "m2"}],
                          _profile([{"name": "m1"}, {"name": "m2"}]))
    headers = {"Authorization": "Bearer admin"}
    with TestClient(server_module.app) as client:
        assert client.get("/admin/api/status", headers=headers).json()["catalog"]["review"] == ["m1", "m2"]
        assert client.post("/admin/api/models/m1/review", headers=headers, json={"keep": True}).status_code == 200
        assert client.post("/admin/api/models/m2/review", headers=headers, json={"keep": False}).status_code == 200
        assert client.post("/admin/api/models/m2/review", headers=headers, json={"keep": False}).status_code == 404
        assert client.get("/admin/api/status", headers=headers).json()["catalog"]["review"] == []
    assert json.loads(info.read_text())["llm"] == [{"name": "m1", "desc": "mine", "shipped_base": {"desc": None}}]
    assert providers._load_models()["m1"]["desc"] == "mine"


def test_base_model_references_resolve_to_an_available_route(tmp_path, monkeypatch):
    profile = _profile([
        {"name": "astra", "family": "gpt", "base_model": "astra"},
        {"name": "astra-or", "family": "gpt", "base_model": "astra", "provider_model_id": "or/astra"},
        {"name": "glm-fw", "family": "glm", "base_model": "glm"},
    ])
    _registry_with(tmp_path, monkeypatch, [], profile)
    assert providers.resolve("glm").provider_model_id == "up/glm-fw"  # a pure base-model name
    assert providers.resolve("astra").provider_model_id == "up/astra"
    assert providers.resolve_base_model("glm", locality="local_only") is None
    providers.CONFIG_PATH.write_text(yaml.safe_dump({
        "providers": {"example": {"base_url": BASE_URL, "api_key": "k"}},
        "model_overrides": {"astra": {"enabled": False}, "glm-fw": {"enabled": False}},
    }))
    providers.reload()
    assert providers.resolve("astra") is None  # the owner turned that route off
    assert providers.resolve_base_model("astra") == "astra-or"
    assert providers.resolve("glm") is None


def test_provider_preference_orders_base_model_routes(tmp_path, monkeypatch):
    first = _profile([{"name": "m-a", "family": "m", "base_model": "m"}], profile_id="a")
    second = onboarding.validate_profile({**first, "id": "b", "provider": {**first["provider"], "id": "other",
                                          "base_url": "https://other.test/v1"},
                                          "models": [{**first["models"][0], "name": "m-b", "provider": "other",
                                                      "provider_model_id": "up/m-b"}]})
    monkeypatch.setattr(shipped, "_packaged", lambda: {"a": first, "b": second})
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"llm": [], "allow_empty": True}))
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", info)
    config = {"providers": {"example": {"base_url": BASE_URL, "api_key": "k"},
                            "other": {"base_url": "https://other.test/v1", "api_key": "k"}}}
    providers.CONFIG_PATH.write_text(yaml.safe_dump(config))
    providers.reload()
    assert providers.resolve_base_model("m") == "m-a"
    providers.CONFIG_PATH.write_text(yaml.safe_dump({**config, "catalog": {"provider_preference": ["other"]}}))
    providers.reload()
    assert providers.resolve_base_model("m") == "m-b"


def test_profile_routes_may_name_a_base_model(tmp_path, monkeypatch):
    from src import profiles

    _registry_with(tmp_path, monkeypatch, [], _profile([{"name": "glm-fw", "family": "glm", "base_model": "glm"}]))

    def manifest(locality, policy):
        return {"schema_version": 1, "namespace": "ha", "source_revision": "r", "profiles": [{
            "id": "ha/p", "locality": locality, "credential_policy": policy,
            "protocols": ["openai_chat"], "routes": {"text": "glm"}}]}

    registered, _bindings = profiles.validate_manifest(manifest("cloud_explicit", "gateway_managed"))
    assert registered["profiles"][0]["routes"] == {"text": "glm-fw"}  # bound to the route available now
    with pytest.raises(profiles.ProfileError):
        profiles.validate_manifest(manifest("local_only", "gateway_local"))  # no local route of glm


def test_a_base_model_request_is_served_and_billed_as_its_route(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.server as server_module

    _registry_with(tmp_path, monkeypatch, [], _profile([{"name": "glm-fw", "family": "glm", "base_model": "glm"}]))
    served = []
    real_ctx = server_module._set_ledger_ctx

    def capture(request, model, info, is_stream=False):
        served.append(model)
        return real_ctx(request, model, info, is_stream)

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        return server_module.JSONResponse(status_code=200, content={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(server_module, "_set_ledger_ctx", capture)
    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)
    with TestClient(server_module.app) as client:
        response = client.post("/v1/chat/completions", json={"model": "glm", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200, response.text
    assert set(served) == {"glm-fw"} and "Deprecation" not in response.headers


def test_create_only_registration_cannot_restore_hidden_or_retired_names(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.server as server_module

    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin")
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    _registry_with(tmp_path, monkeypatch, [], _profile([{"name": "m1"}], retire={"models": ["old"]}))
    config = yaml.safe_load(providers.CONFIG_PATH.read_text())
    providers.CONFIG_PATH.write_text(yaml.safe_dump({**config, "catalog": {"hidden": ["m1"]}}))
    providers.reload()
    headers = {"Authorization": "Bearer admin", "If-None-Match": "*"}
    body = {"provider": "omlx", "omlx_id": "x", "pricing_status": "unmetered"}
    with TestClient(server_module.app) as client:
        for name in ("m1", "old"):
            assert client.put(f"/admin/api/models/{name}", headers=headers, json=body).status_code == 412


def test_a_profile_route_the_owner_turned_off_is_not_rebound(tmp_path, monkeypatch):
    from src import profiles

    profile = _profile([{"name": "astra", "family": "gpt", "base_model": "astra"},
                        {"name": "astra-or", "family": "gpt", "base_model": "astra", "provider_model_id": "or/astra"}])
    _registry_with(tmp_path, monkeypatch, [], profile)
    config = yaml.safe_load(providers.CONFIG_PATH.read_text())
    providers.CONFIG_PATH.write_text(yaml.safe_dump({**config, "model_overrides": {"astra": {"enabled": False}}}))
    providers.reload()
    manifest = {"schema_version": 1, "namespace": "ha", "source_revision": "r", "profiles": [{
        "id": "ha/p", "locality": "cloud_explicit", "credential_policy": "gateway_managed",
        "protocols": ["openai_chat"], "routes": {"text": "astra"}}]}
    with pytest.raises(profiles.ProfileError):
        profiles.validate_manifest(manifest)


def test_base_model_fallback_never_crosses_locality_or_bypasses_owner_choices(tmp_path, monkeypatch):
    profile = _profile([
        {"name": "glm", "family": "glm", "base_model": "glm", "thinking": "always", "thinking_levels": ["low"]},
        {"name": "glm-or", "family": "glm", "base_model": "glm", "provider_model_id": "or/glm"},
    ])
    local = {"name": "qwen", "provider": "omlx", "omlx_id": "qwen", "family": "qwen", "base_model": "qwen",
             "pricing_status": "unmetered"}
    _registry_with(tmp_path, monkeypatch, [{"name": "glm", "thinking_levels": ["off"], "shipped_base": {}}, local], profile)
    assert providers.resolve("glm") is None  # override_invalid: fail closed, not rerouted
    cloud_qwen = {"name": "qwen-fw", "provider": "example", "provider_model_id": "up/qwen", "family": "qwen",
                  "base_model": "qwen", "pricing": {"input": 1.0, "output": 2.0}}
    info = providers.MODEL_INFO_PATH
    info.write_text(json.dumps({"llm": [local, cloud_qwen], "allow_empty": True}))
    config = yaml.safe_load(providers.CONFIG_PATH.read_text())
    for omlx in ({"base_url": "http://127.0.0.1:9110/v1", "enabled": False}, None):
        providers.CONFIG_PATH.write_text(yaml.safe_dump(
            {**config, "providers": {**config["providers"], **({"omlx": omlx} if omlx else {})}}))
        providers.reload()
        assert providers.base_model_route("qwen") is None  # local traffic never goes to the cloud route


def test_overlay_rows_over_shipped_models_follow_the_owner_rules(tmp_path):
    profiles = {"p": _profile([{"name": "m1"}, {"name": "m2", "thinking": "always", "thinking_levels": ["low"]}])}
    config = {"models": [
        {"name": "m1", "provider": "omlx", "omlx_id": "local-m1", "pricing_status": "unmetered"},
        {"name": "m2", "thinking_levels": ["off"]},
    ]}
    models, _ = _entries(tmp_path, [], profiles, config)
    assert (models["m1"]["provider"], models["m1"]["omlx_id"], models["m1"]["origin"]) == ("omlx", "local-m1", {"layer": "owner"})
    assert "provider_model_id" not in models["m1"]
    assert models["m2"]["origin"]["override_error"]


def test_a_release_moving_an_identifier_onto_an_owner_row_never_breaks_the_catalog(tmp_path):
    profiles = {"p": _profile([{"name": "m1", "provider_model_id": "up/new"}])}
    owner = [{"name": "m1", "desc": "mine", "shipped_base": {}},
             {"name": "other", "provider": "example", "provider_model_id": "up/new"}]
    models, _ = _entries(tmp_path, owner, profiles)
    assert models["other"]["provider_model_id"] == "up/new" and "m1" not in models


def test_migration_flags_another_route_under_a_shipped_name_for_review(tmp_path, monkeypatch):
    from src import config_io

    profile = _profile([{"name": "m1"}])
    fork = {"name": "m1", "provider": "example", "provider_model_id": "up/m1-v1", "context": 1000,
            "pricing": {"input": 1.0, "output": 2.0}}
    info = _registry_with(tmp_path, monkeypatch, [fork], profile)
    layer = shipped.layer(providers._load_config(), include_packs=False)
    loaded = {row["name"]: {k: v for k, v in row.items() if k != "origin"} for row in layer.entries}
    assert config_io.shipped_migration_pending(loaded)
    assert config_io.migrate_to_shipped_layer(loaded, layer.names)["review"] == ["m1"]
    assert json.loads(info.read_text())["llm"] == [{**fork, "review": True}]
    providers.reload()
    assert providers.resolve("m1").provider_model_id == "up/m1-v1"
    assert providers.catalog_status()["review"] == ["m1"]


def test_an_invalid_catalog_block_during_fallback_is_not_a_server_error(tmp_path, monkeypatch):
    _registry_with(tmp_path, monkeypatch, [], _profile([{"name": "m1"}]))
    assert "m1" in providers._load_models()
    config = yaml.safe_load(providers.CONFIG_PATH.read_text())
    providers.CONFIG_PATH.write_text(yaml.safe_dump({**config, "catalog": {"shipped": "bogus"}}))
    providers.reload()
    assert providers.resolve("nope") is None
    assert providers.catalog_status()["load_error"]


def test_overlay_rows_never_merge_into_pack_models(tmp_path, monkeypatch):
    pack = {"name": "qwen-pack", "provider": "omlx", "omlx_id": "qwen-8bit", "context": 7, "alias": "qp",
            "pricing_status": "unmetered", "desc": "from the pack"}
    monkeypatch.setattr(shipped, "_registered_packs", lambda config_path=None: {"ha": {"models": [pack]}})
    config = {"models": [
        {"name": "my-qwen", "provider": "omlx", "omlx_id": "qwen-8bit", "pricing_status": "unmetered"},
    ]}
    models, _ = _entries(tmp_path, [], {}, config)
    assert models["my-qwen"]["origin"] == {"layer": "owner"}
    assert "context" not in models["my-qwen"] and "desc" not in models["my-qwen"]
    assert "qwen-pack" not in models


def test_an_overlay_route_change_that_does_not_validate_is_a_clean_error(tmp_path):
    profiles = {"p": _profile([{"name": "m1"}])}
    config = {"models": [{"name": "m1", "provider": "omlx", "omlx_id": "x", "pricing_status": "unmetered",
                          "thinking": "always", "thinking_levels": ["off"]}]}
    with pytest.raises(ValueError, match="thinking"):
        _entries(tmp_path, [], profiles, config)


def test_an_override_colliding_with_another_owner_row_is_still_an_error(tmp_path):
    profiles = {"p": _profile([{"name": "m1"}])}
    owner = [{"name": "m1", "alias": "x", "shipped_base": {}},
             {"name": "mine", "provider": "omlx", "omlx_id": "mine", "alias": "x", "pricing_status": "unmetered"}]
    with pytest.raises(ValueError, match="collides"):
        _entries(tmp_path, owner, profiles)


def test_an_owner_set_collision_fails_whatever_the_row_order(tmp_path):
    profiles = {"p": _profile([{"name": "m1"}])}
    owner = [{"name": "other", "provider": "example", "provider_model_id": "up/m1"},
             {"name": "mine", "provider": "omlx", "omlx_id": "mine", "alternate_ids": ["x"], "pricing_status": "unmetered"},
             {"name": "m1", "alternate_ids": ["x"], "shipped_base": {}}]
    with pytest.raises(ValueError, match="collides"):
        _entries(tmp_path, owner, profiles)


def test_a_displaced_shipped_model_is_reported(tmp_path, monkeypatch):
    profile = _profile([{"name": "m1", "provider_model_id": "up/new"}])
    owner = [{"name": "m1", "desc": "mine", "shipped_base": {}},
             {"name": "other", "provider": "example", "provider_model_id": "up/new"}]
    _registry_with(tmp_path, monkeypatch, owner, profile)
    assert providers.catalog_status()["displaced"] == {"other": ["m1"]}
