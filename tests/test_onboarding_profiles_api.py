"""Applying packaged onboarding profiles through the admin API (additive only)."""

from __future__ import annotations

import json

import pytest
import yaml
from fastapi.testclient import TestClient

from src import admin, config_io, onboarding
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

    monkeypatch.setattr(onboarding, "validate_upstream_models", validator)
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
    assert set(_rows(client, ADMIN)) == {"fireworks-glm-5.3", "fireworks-kimi-k3", "moonshot-kimi-k3-rollback"}
    rows = _rows(client)
    assert set(rows) == {"fireworks-glm-5.3", "fireworks-kimi-k3"}
    assert rows[GLM] == {"id": GLM, "provider": "fireworks", "models": list(GLM_MODELS),
                         "provider_defined": True, "has_api_key": False, "applied": False}


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


def test_fresh_gateway_applies_profile_and_ignores_retirements(client, cfg, upstream):
    # The profile retires glm-5.2-fw and two z.ai models; none exist here.
    response = client.post(URL, headers=MANAGER, json={"api_key": "fw-key"})
    assert response.status_code == 200, response.text
    assert response.json() == {"profile": GLM, "provider": "fireworks", "added_models": list(GLM_MODELS),
                               "already_present": [], "key_updated": True, "reloaded": True}
    assert upstream == ["fw-key"]
    assert "fw-key" not in cfg.read_text() and "fw-key" not in response.text
    assert config_io.api_key_file_target("fireworks").read_text() == "fw-key\n"
    assert {row["name"]: row["provider_model_id"] for row in _catalog(cfg)} == GLM_MODELS
    # Reloaded in-process: the models route now.
    loaded = providers._load_models()
    assert loaded["glm-5.3"]["name"] == "glm-5.3-fw"
    assert loaded["glm53flashfw"]["name"] == "glm-5.3-flash-fw"
    assert _rows(client)[GLM] | {"models": None} == {
        "id": GLM, "provider": "fireworks", "models": None,
        "provider_defined": True, "has_api_key": True, "applied": True}


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


def test_existing_models_are_left_untouched_and_never_retired(client, cfg, upstream):
    present = {"name": "glm-5.3-fw", "provider": "fireworks", "provider_model_id": GLM_MODELS["glm-5.3-fw"],
               "context": 1000, "desc": "owner edited"}
    retired = {"name": "glm-5.2-fw", "provider": "fireworks", "provider_model_id": "accounts/fireworks/models/glm-5p2"}
    _seed_models(cfg, present, retired)
    response = client.post(URL, headers=ADMIN, json={"api_key": "fw-key"})
    assert response.status_code == 200, response.text
    assert response.json()["added_models"] == ["glm-5.3-flash-fw"]
    assert response.json()["already_present"] == ["glm-5.3-fw"]
    rows = {row["name"]: row for row in _catalog(cfg)}
    assert rows["glm-5.3-fw"] == present and rows["glm-5.2-fw"] == retired


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


@pytest.mark.parametrize("api_key", ["", "  ", 5, ["k"]])
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
    # Protocol-less and disabled, like an owner-written block: only the key changes.
    _edit(cfg, lambda config: config["providers"].update({"fireworks": {"base_url": FIREWORKS + "/", "enabled": False}}))
    assert client.post(URL, headers=MANAGER, json={"api_key": "fw-key"}).status_code == 200
    block = yaml.safe_load(cfg.read_text())["providers"]["fireworks"]
    assert block["base_url"] == FIREWORKS + "/" and block["enabled"] is False and "protocol" not in block


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
    {"name": "glm-5.3-zai", "provider": "zai", "provider_model_id": "glm-5.3", "alias": "glm53"},
])
def test_model_conflict_writes_nothing(client, cfg, upstream, existing):
    _seed_models(cfg, existing)
    before = _state(cfg)
    response = client.post(URL, headers=MANAGER, json={"api_key": "fw-key"})
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "model_conflict"
    assert existing["name"] in response.json()["detail"] or "glm-5.3" in response.json()["detail"]
    assert _state(cfg) == before and upstream == []


def test_upstream_rejection_writes_nothing(client, cfg, upstream):
    before = _state(cfg)
    response = client.post(URL, headers=MANAGER, json={"api_key": "bad-key"})
    assert response.status_code == 400
    assert response.json()["code"] == "upstream_rejected"
    assert "401" in response.json()["detail"] and "bad-key" not in response.text
    assert _state(cfg) == before


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
