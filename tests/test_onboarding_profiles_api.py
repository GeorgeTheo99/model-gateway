"""Applying packaged onboarding profiles through the admin API (additive only)."""

from __future__ import annotations

import json

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from src import admin, config_io, onboarding, shipped
import src.providers as providers
from src.server import app

ADMIN = {"Authorization": "Bearer admin"}
MANAGER = {"Authorization": "Bearer manager-token"}
RUNTIME = {"Authorization": "Bearer runtime-token"}
DEPLOYER = {"Authorization": "Bearer deployer-token"}
KEYS_ONLY = {"Authorization": "Bearer keys-token"}
FIREWORKS = "https://api.fireworks.ai/inference/v1"
GLM = "fireworks-glm-5.3"
URL = f"/admin/api/onboarding/profiles/{GLM}"
GLM_MODELS = {
    "glm-5.3-fw": "accounts/fireworks/models/glm-5p3",
    "glm-5.3-flash-fw": "accounts/fireworks/models/glm-5p3-flash",
}


@pytest.fixture(autouse=True)
def _released_profiles(monkeypatch):
    """This module exercises the curated profiles a release ships."""
    monkeypatch.setattr(shipped, "_packaged", lambda: onboarding.packaged_profiles())


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A fresh package install: keyless seeded fireworks and an empty catalog."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "auth": {
            "admin_keys": ["admin"],
            "consumer_credentials": [
                {"id": "ha-manager", "consumer": "ha", "key": "manager-token", "namespaces": ["ha"],
                 "permissions": ["providers:manage", "models:register"], "providers": ["fireworks"]},
                {"id": "ha-runtime", "consumer": "ha", "key": "runtime-token", "namespaces": ["ha"],
                 "permissions": ["profiles:read", "profiles:invoke"]},
                {"id": "ha-deployer", "consumer": "ha", "key": "deployer-token", "namespaces": ["ha"],
                 "permissions": ["profiles:read", "profiles:write"]},
                {"id": "ha-keys", "consumer": "ha", "key": "keys-token", "namespaces": ["ha"],
                 "permissions": ["providers:manage"], "providers": ["fireworks"]},
            ],
        },
        "providers": {"fireworks": {"base_url": FIREWORKS, "protocol": "openai", "enabled": True}},
    }))
    path.chmod(0o600)
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"allow_empty": True, "llm": []}))
    for module in (providers, config_io):
        monkeypatch.setattr(module, "CONFIG_PATH", path)
        monkeypatch.setattr(module, "MODEL_INFO_PATH", info)
        monkeypatch.setattr(module, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.delenv("MODEL_GATEWAY_PROVIDER_FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_GATEWAY_PROVIDER_FIREWORKS_BASE_URL", raising=False)
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    config_io.log_dir = tmp_path / "logs"
    providers.reload()
    return path


@pytest.fixture
def upstream(monkeypatch):
    """Stub the provider's /models check; records the keys it was given."""
    calls: list[str] = []

    def validator(profile, key):
        calls.append(key)
        if key == "bad-key":
            raise onboarding.OnboardingError("provider model discovery failed: 401 Unauthorized")
        return sorted(model["provider_model_id"] for model in profile["models"])

    def completion(profile, key):
        if key == "no-credit-key":
            raise onboarding.OnboardingError("provider refused a test completion (HTTP 402): no credit")

    monkeypatch.setattr(onboarding, "validate_upstream_models", validator)
    monkeypatch.setattr(onboarding, "validate_upstream_completion", completion)
    return calls


@pytest.fixture
def client(cfg):
    with TestClient(app) as c:
        yield c


def _catalog(cfg) -> list[dict]:
    return json.loads((cfg.parent / "model-info.json").read_text())["llm"]


def _state(cfg) -> tuple[bytes, bytes, bool]:
    key_file = config_io.api_key_file_target("fireworks")
    return cfg.read_bytes(), (cfg.parent / "model-info.json").read_bytes(), bool(key_file and key_file.exists())


def _edit(cfg, mutate) -> None:
    config = yaml.safe_load(cfg.read_text())
    mutate(config)
    cfg.write_text(yaml.safe_dump(config))
    providers.reload()


def _seed_models(cfg, *rows) -> None:
    (cfg.parent / "model-info.json").write_text(json.dumps({"allow_empty": True, "llm": list(rows)}))
    providers.reload()


def _rows(client, headers=MANAGER) -> dict[str, dict]:
    response = client.get("/admin/api/onboarding/profiles", headers=headers)
    assert response.status_code == 200, response.text
    return {row["id"]: row for row in response.json()["profiles"]}


# ── listing ──────────────────────────────────────────────────────────────────


def test_lists_packaged_profiles_filtered_by_allowlist(client):
    assert set(_rows(client, ADMIN)) == {
        "anthropic-claude", "fireworks-glm-5.3", "fireworks-kimi-k3", "moonshot-kimi-k3-rollback",
        "moonshot-kimi-k3.1-pro",
        "openai-gpt", "openrouter-claude", "openrouter-gemini-4", "openrouter-gpt",
    }
    rows = _rows(client)
    assert set(rows) == {"fireworks-glm-5.3", "fireworks-kimi-k3"}
    assert rows[GLM] == {
        "id": GLM, "provider": "fireworks", "provider_display_name": "Fireworks", "auto_add": True,
        "models": list(GLM_MODELS),
        "model_details": [
            {"name": "glm-5.3-fw", "display_name": "GLM 5.3 (Fireworks)", "zero_data_retention": True,
             "base_model": "glm-5.3"},
            {"name": "glm-5.3-flash-fw", "display_name": "GLM 5.3 Flash (Fireworks)", "zero_data_retention": True,
             "base_model": "glm-5.3-flash"},
        ],
        "provider_defined": True, "has_api_key": False, "applied": False,
    }


def test_listing_offers_only_valid_reviewed_top_level_profiles(client, tmp_path, monkeypatch):
    packaged = tmp_path / "onboarding"
    (packaged / "drafts").mkdir(parents=True)
    source = (onboarding.PACKAGED_PROFILE_DIR / f"{GLM}.yaml").read_text()
    (packaged / f"{GLM}.yaml").write_text(source)
    (packaged / "drafts" / "fireworks-draft.yaml").write_text(source.replace(GLM, "fireworks-draft"))
    generated = yaml.safe_load(source.replace(GLM, "fireworks-generated"))
    generated["provenance"] = {"generator": "test"}
    (packaged / "fireworks-generated.yaml").write_text(yaml.safe_dump(generated))
    (packaged / "fireworks-broken.yaml").write_text("schema_version: 1\nprovider: {id: fireworks}\n")
    (packaged / "fireworks-misnamed.yaml").write_text(source)
    monkeypatch.setattr(onboarding, "PACKAGED_PROFILE_DIR", packaged)
    assert set(_rows(client, ADMIN)) == {GLM}
    assert client.post("/admin/api/onboarding/profiles/fireworks-draft", headers=MANAGER,
                       json={"api_key": "fw-key"}).status_code == 404


@pytest.mark.parametrize("headers", [RUNTIME, DEPLOYER, KEYS_ONLY])
def test_credentials_without_both_management_grants_are_refused(client, cfg, upstream, headers):
    before = _state(cfg)
    assert client.get("/admin/api/onboarding/profiles", headers=headers).status_code == 403
    assert client.post(URL, headers=headers, json={"api_key": "fw-key"}).status_code == 403
    assert client.post(URL, json={"api_key": "fw-key"}).status_code == 401
    assert _state(cfg) == before and upstream == []


def test_manager_cannot_apply_a_profile_outside_its_allowlist(client, cfg, upstream):
    before = _state(cfg)
    response = client.post("/admin/api/onboarding/profiles/moonshot-kimi-k3-rollback", headers=MANAGER,
                           json={"api_key": "moonshot-key"})
    assert response.status_code == 403
    assert _state(cfg) == before and upstream == []


def test_unknown_profile_is_404(client, upstream):
    response = client.post("/admin/api/onboarding/profiles/nope", headers=MANAGER, json={"api_key": "k"})
    assert response.status_code == 404
    assert response.json()["code"] == "profile_not_found"


# ── applying ─────────────────────────────────────────────────────────────────


def test_fresh_gateway_applies_profile_by_reference(client, cfg, upstream):
    # The profile retires glm-5.2-fw and two z.ai models; none exist here.
    response = client.post(URL, headers=MANAGER, json={"api_key": "fw-key"})
    assert response.status_code == 200, response.text
    # The key also unlocks the provider's other curated profile.
    assert response.json() == {"profile": GLM, "provider": "fireworks", "added_models": list(GLM_MODELS),
                               "auto_added_models": ["kimi-k3"],
                               "already_present": [], "key_updated": True, "reloaded": True}
    assert upstream == ["fw-key"]
    assert "fw-key" not in cfg.read_text() and "fw-key" not in response.text
    assert config_io.api_key_file_target("fireworks").read_text() == "fw-key\n"
    # Curated models load from the release; nothing is copied into the owner's catalog.
    assert _catalog(cfg) == []
    loaded = providers._load_models()
    assert loaded["glm-5.3"]["name"] == "glm-5.3-fw"
    assert loaded["glm53flashfw"]["name"] == "glm-5.3-flash-fw"
    assert loaded["kimi-k3"]["origin"] == {"layer": "shipped", "profile": "fireworks-kimi-k3"}
    row = _rows(client)[GLM]
    assert (row["provider_defined"], row["has_api_key"], row["applied"]) == (True, True, True)


def test_second_apply_is_idempotent(client, cfg, upstream):
    assert client.post(URL, headers=MANAGER, json={"api_key": "fw-key"}).status_code == 200
    before = _state(cfg)
    response = client.post(URL, headers=MANAGER)  # no body: keep the configured key
    assert response.status_code == 200, response.text
    assert response.json()["added_models"] == []
    assert response.json()["already_present"] == list(GLM_MODELS)
    assert response.json()["key_updated"] is False
    assert _state(cfg) == before
    assert upstream == ["fw-key", "fw-key"]
    # The same key again is not an update either.
    assert client.post(URL, headers=MANAGER, json={"api_key": "fw-key"}).json()["key_updated"] is False
    assert _state(cfg) == before


def test_owner_rows_are_left_untouched(client, cfg, upstream):
    present = {"name": "glm-5.3-fw", "provider": "fireworks", "provider_model_id": GLM_MODELS["glm-5.3-fw"],
               "context": 1000, "desc": "owner edited"}
    _seed_models(cfg, present)
    response = client.post(URL, headers=ADMIN, json={"api_key": "fw-key"})
    assert response.status_code == 200, response.text
    assert response.json()["added_models"] == ["glm-5.3-flash-fw"]
    assert response.json()["already_present"] == ["glm-5.3-fw"]
    assert _catalog(cfg) == [present]
    entry = providers._load_models()["glm-5.3-fw"]
    assert (entry["context"], entry["desc"]) == (1000, "owner edited")  # the owner wins per field
    assert entry["display_name"] == "GLM 5.3 (Fireworks)"  # other fields still come from the release


def test_key_is_optional_when_the_provider_has_one(client, cfg, upstream):
    before = _state(cfg)
    response = client.post(URL, headers=MANAGER, json={"api_key": None})
    assert response.status_code == 400
    assert response.json()["code"] == "api_key_required" and "key" in response.json()["detail"]
    assert _state(cfg) == before and upstream == []
    assert client.post("/admin/api/providers/fireworks", headers=MANAGER, json={"api_key": "fw-key"}).status_code == 200
    response = client.post(URL, headers=MANAGER, json={})
    assert response.status_code == 200, response.text
    assert response.json()["key_updated"] is False and upstream == ["fw-key"]


def test_a_new_key_replaces_the_providers_key(client, cfg, upstream):
    assert client.post("/admin/api/providers/fireworks", headers=MANAGER, json={"api_key": "old-key"}).status_code == 200
    response = client.post(URL, headers=MANAGER, json={"api_key": "new-key"})
    assert response.status_code == 200, response.text
    assert response.json()["key_updated"] is True
    assert config_io.api_key_file_target("fireworks").read_text() == "new-key\n"


@pytest.mark.parametrize("api_key", ["", "  ", 5, ["k"], "fw\x00secret", "fw secret", "fw\x7f", "ключ", "k" * 513])
def test_invalid_api_key_values_are_rejected(client, cfg, upstream, api_key):
    before = _state(cfg)
    assert client.post(URL, headers=MANAGER, json={"api_key": api_key}).status_code == 400
    assert _state(cfg) == before and upstream == []


def test_missing_provider_is_defined_only_with_the_reviewed_endpoint(client, cfg, upstream):
    _edit(cfg, lambda config: config["providers"].pop("fireworks"))
    response = client.post(URL, headers=MANAGER, json={"api_key": "fw-key"})
    assert response.status_code == 200, response.text
    block = yaml.safe_load(cfg.read_text())["providers"]["fireworks"]
    assert {key: block[key] for key in ("base_url", "protocol", "enabled")} == {
        "base_url": FIREWORKS, "protocol": "openai", "enabled": True}
    assert "api_key" not in block and block["api_key_file"]


def test_existing_provider_block_keeps_its_owner_settings(client, cfg, upstream):
    # Protocol-less, like an owner-written block: only the key changes.
    _edit(cfg, lambda config: config["providers"].update({"fireworks": {"base_url": FIREWORKS + "/"}}))
    assert client.post(URL, headers=MANAGER, json={"api_key": "fw-key"}).status_code == 200
    block = yaml.safe_load(cfg.read_text())["providers"]["fireworks"]
    assert block["base_url"] == FIREWORKS + "/" and "enabled" not in block and "protocol" not in block


def test_a_turned_off_provider_or_model_is_refused_not_applied(client, cfg, upstream):
    _edit(cfg, lambda config: config["providers"]["fireworks"].update({"enabled": False}))
    before = _state(cfg)
    response = client.post(URL, headers=MANAGER, json={"api_key": "fw-key"})
    assert response.status_code == 409 and response.json()["code"] == "provider_disabled"
    assert _state(cfg) == before and upstream == []
    _edit(cfg, lambda config: config["providers"]["fireworks"].update({"enabled": True}))
    _seed_models(cfg, {"name": "glm-5.3-fw", "provider": "fireworks", "provider_model_id": GLM_MODELS["glm-5.3-fw"]})
    _edit(cfg, lambda config: config.update({"model_overrides": {"glm-5.3-fw": {"enabled": False}}}))
    response = client.post(URL, headers=MANAGER, json={"api_key": "fw-key"})
    assert response.status_code == 409 and response.json()["code"] == "model_disabled"
    # A profile model not added yet but already turned off is refused too.
    _seed_models(cfg)
    _edit(cfg, lambda config: config.update({"model_overrides": {"glm-5.3-flash-fw": {"enabled": False}}}))
    response = client.post(URL, headers=MANAGER, json={"api_key": "fw-key"})
    assert response.status_code == 409 and response.json()["code"] == "model_disabled"


# ── refusals write nothing ───────────────────────────────────────────────────


@pytest.mark.parametrize("change", [
    {"base_url": "https://attacker.example/v1"},
    {"protocol": "anthropic"},
])
def test_provider_conflict_writes_nothing(client, cfg, upstream, change):
    _edit(cfg, lambda config: config["providers"]["fireworks"].update(change))
    before = _state(cfg)
    for headers in (MANAGER, ADMIN):
        response = client.post(URL, headers=headers, json={"api_key": "fw-key"})
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "provider_conflict"
    assert _state(cfg) == before and upstream == []


def test_provider_defined_outside_its_own_block_is_a_conflict(client, cfg, upstream):
    _edit(cfg, lambda config: (config["providers"].pop("fireworks"),
                               config.setdefault("workspaces", {}).update({"fireworks": {"base_url": FIREWORKS}})))
    before = _state(cfg)
    response = client.post(URL, headers=MANAGER, json={"api_key": "fw-key"})
    assert response.status_code == 409 and response.json()["code"] == "provider_conflict"
    assert _state(cfg) == before and upstream == []


@pytest.mark.parametrize("existing", [
    {"name": "glm-5.3-fw", "provider": "fireworks", "provider_model_id": "accounts/fireworks/models/other"},
    {"name": "glm-5.3-fw", "provider": "anthropic", "provider_model_id": GLM_MODELS["glm-5.3-fw"]},
])
def test_an_owner_route_under_a_curated_name_wins(client, cfg, upstream, existing):
    _seed_models(cfg, existing)
    response = client.post(URL, headers=MANAGER, json={"api_key": "fw-key"})
    assert response.status_code == 200, response.text
    assert _catalog(cfg) == [existing]
    entry = providers._load_models()["glm-5.3-fw"]
    assert (entry["provider"], entry["provider_model_id"]) == (existing["provider"], existing["provider_model_id"])


def test_an_owner_row_with_a_retired_name_keeps_working(cfg):
    _seed_models(cfg, {"name": "glm-5.3-zai", "provider": "zai", "provider_model_id": "glm-5.3", "alias": "glm53"})
    assert providers._load_models()["glm53"]["name"] == "glm-5.3-zai"
    assert providers.catalog_status()["retired_owner_rows"] == ["glm-5.3-zai"]
    _edit(cfg, lambda config: config.update(catalog={"pinned": ["glm-5.3-zai"]}))
    assert providers.catalog_status()["retired_owner_rows"] == []


def test_upstream_rejection_writes_nothing(client, cfg, upstream):
    before = _state(cfg)
    response = client.post(URL, headers=MANAGER, json={"api_key": "bad-key"})
    assert response.status_code == 400
    assert response.json()["code"] == "upstream_rejected"
    assert "401" in response.json()["detail"] and "bad-key" not in response.text
    assert _state(cfg) == before


def test_a_key_that_cannot_answer_never_replaces_a_working_one(client, cfg, upstream):
    assert client.post(URL, headers=MANAGER, json={"api_key": "fw-key"}).status_code == 200
    before = _state(cfg)
    response = client.post(URL, headers=MANAGER, json={"api_key": "no-credit-key"})
    assert response.status_code == 400 and response.json()["code"] == "upstream_rejected"
    assert "402" in response.json()["detail"] and "no-credit-key" not in response.text
    assert _state(cfg) == before
    assert config_io.api_key_file_target("fireworks").read_text() == "fw-key\n"


def test_the_completion_check_uses_the_candidate_key_on_the_profiles_model(monkeypatch):
    sent = []

    def post(url, *, json, headers, timeout):
        sent.append((url, json["model"], headers["Authorization"]))
        return httpx.Response(402, json={"error": {"message": "insufficient credit"}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(onboarding.httpx, "post", post)
    profile = onboarding.packaged_profiles()["fireworks-glm-5.3"]
    with pytest.raises(onboarding.OnboardingError, match="HTTP 402.*insufficient credit"):
        onboarding.validate_upstream_completion(profile, "candidate")
    assert sent == [(FIREWORKS + "/chat/completions", GLM_MODELS["glm-5.3-fw"], "Bearer candidate")]
    monkeypatch.setattr(onboarding.httpx, "post", lambda url, **_: httpx.Response(200, json={"choices": []}))
    with pytest.raises(onboarding.OnboardingError, match="no reply"):
        onboarding.validate_upstream_completion(profile, "candidate")
    reply = {"choices": [{"message": {"role": "assistant", "content": None}}]}
    monkeypatch.setattr(onboarding.httpx, "post", lambda url, **_: httpx.Response(200, json=reply))
    onboarding.validate_upstream_completion(profile, "candidate")


def test_the_completion_check_applies_the_models_request_quirks(monkeypatch):
    bodies = []

    def post(url, *, json, headers, timeout):
        bodies.append(json)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr(onboarding.httpx, "post", post)
    onboarding.validate_upstream_completion(onboarding.packaged_profiles()["moonshot-kimi-k3-rollback"], "candidate")
    assert "max_tokens" not in bodies[0] and bodies[0]["max_completion_tokens"] == 16
    assert bodies[0]["reasoning_effort"] == "max"


def test_the_completion_check_routes_zdr_profiles_to_zdr_endpoints(monkeypatch):
    bodies = []

    def post(url, *, json, headers, timeout):
        bodies.append(json)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr(onboarding.httpx, "post", post)
    onboarding.validate_upstream_completion(onboarding.packaged_profiles()["openrouter-gemini-4"], "candidate")
    assert bodies[0]["model"] == "google/gemini-4-argon" and bodies[0]["provider"] == {"zdr": True}


def _anthropic_profile():
    return onboarding.validate_profile({
        "schema_version": 1,
        "provider": {"id": "anthropic", "base_url": "https://api.anthropic.com/v1", "protocol": "anthropic"},
        "models": [{"name": "claude-test", "provider": "anthropic", "provider_model_id": "claude-test-1"}],
    })


def test_anthropic_profiles_check_the_key_with_the_messages_api(monkeypatch):
    calls = []

    def get(url, *, headers, params, timeout):
        calls.append(("GET", url, headers, params))
        return httpx.Response(200, json={"data": [{"id": "claude-test-1"}]}, request=httpx.Request("GET", url))

    def post(url, *, json, headers, timeout):
        calls.append(("POST", url, headers, json))
        return httpx.Response(200, json={"type": "message", "role": "assistant", "content": []})

    monkeypatch.setattr(onboarding.httpx, "get", get)
    monkeypatch.setattr(onboarding.httpx, "post", post)
    profile = _anthropic_profile()
    assert onboarding.validate_upstream_models(profile, "candidate") == ["claude-test-1"]
    onboarding.validate_upstream_completion(profile, "candidate")
    (_, models_url, models_headers, params), (_, url, headers, body) = calls
    assert models_url == "https://api.anthropic.com/v1/models" and params == {"limit": 1000}
    assert url == "https://api.anthropic.com/v1/messages"
    for sent in (models_headers, headers):
        assert sent["x-api-key"] == "candidate" and sent["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in sent
    assert body == {
        "model": "claude-test-1", "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 16, "stream": False,
    }


def test_an_anthropic_key_that_cannot_answer_is_refused(monkeypatch):
    key = "sk-ant-" + "k" * 40
    error = {"type": "error", "error": {"type": "authentication_error", "message": f"invalid x-api-key {key}"}}
    monkeypatch.setattr(onboarding.httpx, "post", lambda url, **_: httpx.Response(401, json=error))
    with pytest.raises(onboarding.OnboardingError, match="HTTP 401.*invalid x-api-key") as raised:
        onboarding.validate_upstream_completion(_anthropic_profile(), key)
    assert key not in str(raised.value)
    monkeypatch.setattr(onboarding.httpx, "post", lambda url, **_: httpx.Response(200, json={"choices": []}))
    with pytest.raises(onboarding.OnboardingError, match="no reply"):
        onboarding.validate_upstream_completion(_anthropic_profile(), "candidate")


def test_the_completion_check_sends_max_completion_tokens_to_openai(monkeypatch):
    bodies = []

    def post(url, *, json, headers, timeout):
        bodies.append((url, json))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr(onboarding.httpx, "post", post)
    onboarding.validate_upstream_completion(onboarding.packaged_profiles()["openai-gpt"], "candidate")
    [(url, body)] = bodies
    assert url == "https://api.openai.com/v1/chat/completions"
    assert body["model"] == "gpt-6-luna" and body["max_completion_tokens"] == 16 and "max_tokens" not in body


def test_packaged_routes_to_one_model_agree_on_its_family():
    families: dict[str, str] = {}
    routes: dict[str, set[str]] = {}
    for profile in onboarding.packaged_profiles().values():
        for model in profile["models"]:
            if "base_model" not in model:
                continue
            assert families.setdefault(model["base_model"], model["family"]) == model["family"]
            if profile.get("auto_add"):
                provider = model["provider"]
                assert provider not in routes.setdefault(model["base_model"], set()), model["base_model"]
                routes[model["base_model"]].add(provider)
    assert routes["claude-opus-5.5"] == {"anthropic", "openrouter"}
    assert routes["gpt-6-astra"] == {"openai", "openrouter"}
    assert routes["claude-fable-5.1"] == {"anthropic"}


def test_profiles_reject_an_unknown_protocol():
    profile = onboarding.packaged_profiles()["fireworks-glm-5.3"]
    profile["provider"]["protocol"] = "gemini"
    with pytest.raises(onboarding.OnboardingError, match="protocol must be openai or anthropic"):
        onboarding.validate_profile(profile)


def test_profiles_reject_invalid_client_metadata():
    profile = onboarding.packaged_profiles()["openrouter-gemini-4"]
    profile["models"][0]["zdr"] = False
    with pytest.raises(onboarding.OnboardingError, match="requires zdr: true"):
        onboarding.validate_profile(profile)


def test_a_key_in_a_provider_error_is_redacted_before_it_is_shortened(monkeypatch):
    key = "k" * 300
    monkeypatch.setattr(onboarding.httpx, "post", lambda url, **_: httpx.Response(
        401, json={"error": {"message": f"bad key {key}"}}))
    with pytest.raises(onboarding.OnboardingError) as raised:
        onboarding.validate_upstream_completion(onboarding.packaged_profiles()["fireworks-glm-5.3"], key)
    assert "kkkk" not in str(raised.value) and "[redacted]" in str(raised.value)


def test_writes_disabled_refuses(client, cfg, upstream, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "false")
    before = _state(cfg)
    assert client.post(URL, headers=MANAGER, json={"api_key": "fw-key"}).status_code == 403
    assert _state(cfg) == before and upstream == []
    assert client.get("/admin/api/onboarding/profiles", headers=MANAGER).status_code == 200


def test_rejected_reload_rolls_back_every_file(client, cfg, upstream, monkeypatch):
    before = _state(cfg)
    monkeypatch.setattr(admin, "_reload_registry_transactionally", lambda snapshot: "boom")
    response = client.post(URL, headers=MANAGER, json={"api_key": "fw-key"})
    assert response.status_code == 400 and response.json()["code"] == "reload_rejected"
    assert _state(cfg) == before


# ── curated models by reference (docs/adr/0002-layered-model-catalog.md) ────


def _names(cfg) -> list[str]:
    return [row["name"] for row in _catalog(cfg)]


def _loaded() -> list[str]:
    return admin._catalog_names(shipped_only=True)


def test_curated_models_load_only_for_keyed_providers(cfg, monkeypatch):
    assert _loaded() == []
    monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_FIREWORKS_API_KEY", "fw-key")
    providers.reload()
    # The Moonshot rollback profile is not curated, and OpenRouter has no key.
    assert _loaded() == ["glm-5.3-fw", "glm-5.3-flash-fw", "kimi-k3"]
    assert providers._load_models()["kimi3"]["name"] == "kimi-k3"
    assert _catalog(cfg) == []


def test_a_deleted_curated_model_stays_hidden_until_saved_again(cfg, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_FIREWORKS_API_KEY", "fw-key")
    providers.reload()
    config_io.delete_model("kimi-k3")
    providers.reload()
    assert "kimi-k3" not in _loaded()
    assert yaml.safe_load(cfg.read_text())["catalog"] == {"hidden": ["kimi-k3"]}
    config_io.upsert_model("kimi-k3", provider="fireworks",
                           provider_model_id="accounts/fireworks/models/kimi-k3", desc="mine")
    providers.reload()
    assert providers._load_models()["kimi-k3"]["desc"] == "mine"
    assert "catalog" not in yaml.safe_load(cfg.read_text())
    assert _catalog(cfg) == [{"name": "kimi-k3", "desc": "mine",
                              "shipped_base": {"desc": next(
                                  m for m in onboarding.packaged_profiles()["fireworks-kimi-k3"]["models"]
                              )["desc"]}}]


def test_an_owner_identifier_moves_from_the_curated_model(cfg, monkeypatch):
    _seed_models(cfg, {"name": "other", "provider": "fireworks", "provider_model_id": "x/other", "alias": "kimi3"})
    monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_FIREWORKS_API_KEY", "fw-key")
    providers.reload()
    models = providers._load_models()
    assert models["kimi3"]["name"] == "other"
    assert "alias" not in models["kimi-k3"]


def test_curated_models_need_the_profile_endpoint_and_an_enabled_provider(cfg, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_FIREWORKS_API_KEY", "fw-key")
    _edit(cfg, lambda config: config["providers"]["fireworks"].update(base_url="https://proxy.example/v1"))
    assert _loaded() == []
    _edit(cfg, lambda config: config["providers"]["fireworks"].update(base_url=FIREWORKS, enabled=False))
    assert _loaded() == []


def test_curated_models_load_with_admin_writes_off(cfg, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_FIREWORKS_API_KEY", "fw-key")
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "false")
    providers.reload()
    assert _loaded() == ["glm-5.3-fw", "glm-5.3-flash-fw", "kimi-k3"]
    assert _catalog(cfg) == []


def test_adding_a_provider_key_brings_its_curated_models(client, cfg, upstream):
    response = client.post("/admin/api/providers/fireworks", headers=ADMIN, json={"base_url": FIREWORKS, "api_key": "fw-key"})
    assert response.status_code == 200, response.text
    assert response.json()["auto_added_models"] == ["glm-5.3-fw", "glm-5.3-flash-fw", "kimi-k3"]


def test_startup_migrates_copied_curated_rows(cfg, monkeypatch):
    curated = {m["name"]: m for m in onboarding.packaged_profiles()[GLM]["models"]}
    drifted = {**curated["glm-5.3-flash-fw"], "desc": "owner's"}
    _seed_models(cfg, dict(curated["glm-5.3-fw"]), drifted)
    config_io.auto_add_state_path().write_text(json.dumps(
        {"version": 1, "models": ["glm-5.3-fw", "glm-5.3-flash-fw", "kimi-k3"]}))
    monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_FIREWORKS_API_KEY", "fw-key")
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "false")  # the migration is not an admin write
    providers.reload()
    with TestClient(app):
        pass
    assert _catalog(cfg) == [{"name": "glm-5.3-flash-fw", "desc": "owner's",
                              "shipped_base": {"desc": curated["glm-5.3-flash-fw"]["desc"]}, "review": True}]
    assert yaml.safe_load(cfg.read_text())["catalog"] == {"hidden": ["kimi-k3"]}  # the owner deleted it
    assert not config_io.auto_add_state_path().exists()
    models = providers._load_models()
    assert models["glm-5.3-flash-fw"]["origin"]["review"] is True
    assert "kimi-k3" not in models
    with TestClient(app):  # idempotent
        pass
    assert len(_catalog(cfg)) == 1


def test_profiles_reject_invalid_curation_fields():
    profile = onboarding.packaged_profiles()["openrouter-gemini-4"]
    with pytest.raises(onboarding.OnboardingError, match="auto_add"):
        onboarding.validate_profile({**profile, "auto_add": "yes"})
    with pytest.raises(onboarding.OnboardingError, match="display_name"):
        onboarding.validate_profile({**profile, "provider": {**profile["provider"], "display_name": ""}})
