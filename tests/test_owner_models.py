"""Models a built-in provider lists, added one at a time by a models:register credential (ADR 0005)."""

from __future__ import annotations

import json

import pytest
import yaml
from fastapi.testclient import TestClient

from src import admin, config_io, directory, onboarding_generation, profiles
import src.providers as providers
from src.server import app

MANAGER = {"Authorization": "Bearer manager-token"}
OTHER = {"Authorization": "Bearer other-token"}
KEYS_ONLY = {"Authorization": "Bearer keys-token"}
NARROW = {"Authorization": "Bearer narrow-token"}
FIREWORKS = "https://api.fireworks.ai/inference/v1"
OPENROUTER = "https://openrouter.ai/api/v1"
NEW = "accounts/fireworks/models/brand-new"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "auth": {
            "admin_keys": ["admin"],
            "consumer_credentials": [
                {"id": "myai-manager", "consumer": "myai", "key": "manager-token", "namespaces": ["myai"],
                 "permissions": ["providers:manage", "models:register"], "providers": ["fireworks", "openrouter", "chatgpt"]},
                {"id": "ha-manager", "consumer": "ha", "key": "other-token", "namespaces": ["ha"],
                 "permissions": ["providers:manage", "models:register"], "providers": ["fireworks"]},
                {"id": "myai-keys", "consumer": "myai", "key": "keys-token", "namespaces": ["myai"],
                 "permissions": ["providers:manage"], "providers": ["fireworks"]},
                {"id": "myai-narrow", "consumer": "myai", "key": "narrow-token", "namespaces": ["myai"],
                 "permissions": ["models:register"], "providers": ["openrouter"]},
            ],
        },
        "providers": {
            "fireworks": {"base_url": FIREWORKS, "api_key": "fw-secret"},
            "openrouter": {"base_url": OPENROUTER, "api_key": "or-secret"},
        },
    }))
    path.chmod(0o600)
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"llm": [
        {"name": "glm-fw", "provider": "fireworks", "provider_model_id": "accounts/fireworks/models/glm"},
    ]}))
    for module in (providers, config_io):
        monkeypatch.setattr(module, "CONFIG_PATH", path)
        monkeypatch.setattr(module, "MODEL_INFO_PATH", info)
        monkeypatch.setattr(module, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.setenv("MODEL_GATEWAY_PROFILE_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    config_io.log_dir = tmp_path / "logs"
    providers.reload()
    return path


@pytest.fixture
def upstream(monkeypatch):
    """The provider's model list and its test completions, recorded."""
    calls: dict[str, list] = {"listed": [], "tested": []}
    state = {"models": ["accounts/fireworks/models/glm", NEW], "error": None}

    def discover(base_url, api_key=None, protocol="openai", quirks=()):
        calls["listed"].append((base_url, api_key, protocol))
        if state.get("down"):
            return {"status": "network_error", "model_ids": []}
        return {"status": "verified", "model_ids": list(state["models"])}

    def smoke(row, provider_config):
        calls["tested"].append(row)
        return state["error"]

    monkeypatch.setattr(onboarding_generation, "discover_models", discover)
    monkeypatch.setattr(directory, "smoke_test", smoke)
    return calls, state


@pytest.fixture
def client(cfg):
    with TestClient(app) as c:
        yield c


def _add(client, model=NEW, headers=MANAGER, provider="fireworks"):
    return client.post(f"/admin/api/providers/{provider}/models", headers=headers, json={"model": model})


def test_a_scoped_key_lists_and_adds_a_model_its_provider_lists(client, upstream):
    calls, _ = upstream
    listed = client.post("/admin/api/providers/fireworks/discover", headers=MANAGER)
    assert listed.status_code == 200
    assert listed.json()["models"] == [{"id": "accounts/fireworks/models/glm", "registered": True},
                                       {"id": NEW, "registered": False}]
    added = _add(client)
    assert added.status_code == 200, added.text
    name = added.json()["model"]["name"]
    assert name.startswith("fireworks-") and added.json()["model"]["upstream_id"] == NEW
    stored = next(row for row in config_io.load_model_info()["llm"] if row["name"] == name)
    assert stored["registered_by"] == "myai" and stored["zdr"] is True and "pricing" not in stored
    assert calls["tested"][0]["provider_model_id"] == NEW
    # The runtime row never carries the registrant.
    assert "registered_by" not in providers._load_models()[name]
    registered = client.get("/admin/api/models/registered", headers=MANAGER).json()["models"]
    assert [(m["name"], m["provider_model_id"], m["maintained"]) for m in registered] == [(name, NEW, None)]
    assert client.get("/admin/api/models/registered", headers=OTHER).json()["models"] == []
    again = _add(client)  # already there: refused before any test completion
    assert again.status_code == 409 and again.json()["code"] == "already_added"
    assert len(calls["tested"]) == 1


def test_openrouter_rows_keep_zero_retention(client, upstream):
    _, state = upstream
    state["models"] = ["vendor/new-model"]
    added = _add(client, "vendor/new-model", provider="openrouter")
    assert added.status_code == 200, added.text
    stored = next(row for row in config_io.load_model_info()["llm"] if row["name"] == added.json()["model"]["name"])
    assert stored["quirks"] == ["openrouter_zdr"] and stored["zdr"] is True


def test_nothing_is_written_when_the_model_fails_or_isnt_listed(client, upstream):
    _, state = upstream
    before = config_io.load_model_info()
    assert _add(client, "accounts/fireworks/models/other").json()["code"] == "model_not_listed"
    state["down"] = True
    assert _add(client).json()["code"] == "listing_failed"
    state["down"] = False
    state["error"] = "HTTP 404: model fw-secret not found"
    refused = _add(client)
    assert refused.status_code == 400 and refused.json()["code"] == "upstream_rejected"
    assert "fw-secret" not in refused.text
    assert config_io.load_model_info() == before


def test_scope_and_provider_rules(client, upstream):
    assert _add(client, provider="anthropic").status_code == 403  # not on its allowlist
    assert _add(client, headers=OTHER, provider="openrouter").status_code == 403
    assert _add(client, provider="chatgpt").json()["code"] == "unsupported_provider"
    assert client.post("/admin/api/providers/chatgpt/discover", headers=MANAGER).json()["code"] == "unsupported_provider"
    assert client.post("/admin/api/providers/fireworks/models", headers=MANAGER, json={"model": ""}).status_code == 400
    assert client.post("/admin/api/providers/fireworks/models", headers={"Authorization": "Bearer nope"},
                       json={"model": NEW}).status_code == 401
    # providers:manage alone adds nothing and lists nothing.
    assert _add(client, headers=KEYS_ONLY).status_code == 403
    assert client.post("/admin/api/providers/fireworks/discover", headers=KEYS_ONLY).status_code == 403
    assert client.get("/admin/api/models/registered", headers=KEYS_ONLY).status_code == 403


def test_another_key_of_the_same_consumer_stays_within_its_providers(client, upstream):
    name = _add(client).json()["model"]["name"]  # fireworks, by myai-manager
    assert client.get("/admin/api/models/registered", headers=NARROW).json()["models"] == []
    assert client.delete(f"/admin/api/models/{name}", headers=NARROW).status_code == 403
    assert client.post(f"/admin/api/models/{name}/maintained", headers=NARROW).status_code == 403


def test_an_admins_rows_are_not_a_consumers(client, upstream, cfg):
    config = yaml.safe_load(cfg.read_text())
    config["auth"]["consumer_credentials"].append(
        {"id": "admin-ish", "consumer": "admin", "key": "admin-ish-token", "namespaces": ["x"],
         "permissions": ["models:register"], "providers": ["fireworks"]})
    cfg.write_text(yaml.safe_dump(config))
    providers.reload()
    name = _add(client, headers={"Authorization": "Bearer admin"}).json()["model"]["name"]
    stored = next(row for row in config_io.load_model_info()["llm"] if row["name"] == name)
    assert stored["registered_by"] == admin.ADMIN_REGISTRANT
    assert client.delete(f"/admin/api/models/{name}", headers={"Authorization": "Bearer admin-ish-token"}).status_code == 403


def test_owner_rows_carry_the_providers_facts_and_google_ids_are_bare(monkeypatch):
    corrections = directory.load_corrections()
    row = directory.owner_row("openrouter", "google/gemini-9", corrections)
    assert row["quirks"][0] == "openrouter_zdr" and row["zdr"] is True and "pricing" not in row
    assert row["display_name"] == "google/gemini-9 (OpenRouter)"
    assert admin._upstream_id("google", "models/gemini-9") == "gemini-9"
    assert admin._upstream_id("openrouter", "models/x") == "models/x"


def test_discover_sends_the_providers_own_auth_headers(monkeypatch):
    seen = {}

    class Response:
        status_code = 200

        def json(self):
            return {"data": [{"id": "claude-x"}]}

    def get(url, headers, timeout):
        seen.update(headers)
        return Response()
    monkeypatch.setattr(onboarding_generation.httpx, "get", get)
    result = onboarding_generation.discover_models("https://api.anthropic.com/v1", "sk-ant", "anthropic")
    assert result["model_ids"] == ["claude-x"]
    assert seen["x-api-key"] == "sk-ant" and "Authorization" not in seen and "content-type" not in seen


def test_a_model_the_directory_offers_is_turned_on_there_instead(client, upstream, monkeypatch):
    class Item:
        provider, model_id, row = "fireworks", NEW, {"name": "brand-new-fw", "provider": "fireworks", "provider_model_id": NEW}
    monkeypatch.setattr(admin, "_directory_mapped", lambda: ([Item()], None, "test"))
    refused = _add(client)
    assert refused.status_code == 409 and refused.json()["code"] == "in_directory" and "brand-new-fw" in refused.text


def test_only_its_registrant_removes_a_model_and_never_one_a_profile_uses(client, upstream, monkeypatch):
    name = _add(client).json()["model"]["name"]
    assert client.delete(f"/admin/api/models/{name}", headers=OTHER).status_code == 403
    assert client.delete("/admin/api/models/glm-fw", headers=MANAGER).status_code == 404  # not added by a key
    monkeypatch.setattr(profiles, "routed_models", lambda config_path=None: {name})
    assert client.delete(f"/admin/api/models/{name}", headers=MANAGER).status_code == 409
    monkeypatch.setattr(profiles, "routed_models", lambda config_path=None: set())
    assert client.delete(f"/admin/api/models/{name}", headers=MANAGER).status_code == 200
    assert name not in providers._load_models()


def test_switching_to_the_maintained_row_turns_it_on_and_removes_the_own_row_at_once(client, upstream, monkeypatch):
    name = _add(client).json()["model"]["name"]
    not_yet = client.post(f"/admin/api/models/{name}/maintained", headers=MANAGER)
    assert not_yet.status_code == 409 and not_yet.json()["code"] == "not_in_directory"

    class Item:
        provider, model_id, row = "fireworks", NEW, {"name": "brand-new-fw", "provider": "fireworks", "provider_model_id": NEW}
    monkeypatch.setattr(admin, "_directory_mapped", lambda: ([Item()], None, "test"))
    assert client.get("/admin/api/models/registered", headers=MANAGER).json()["models"][0]["maintained"] == "brand-new-fw"
    assert client.post(f"/admin/api/models/{name}/maintained", headers=OTHER).status_code == 403
    monkeypatch.setattr(profiles, "routed_models", lambda config_path=None: {name})
    assert client.post(f"/admin/api/models/{name}/maintained", headers=MANAGER).status_code == 409
    monkeypatch.setattr(profiles, "routed_models", lambda config_path=None: set())
    # The catalog's row is tested first: one that refuses never replaces the working row.
    _, state = upstream
    state["error"] = "HTTP 500"
    failed = client.post(f"/admin/api/models/{name}/maintained", headers=MANAGER)
    assert failed.status_code == 409 and failed.json()["code"] == "maintained_failed"
    state["error"] = None
    # A twin the catalog can't load (hidden, provider not ready, ids taken): nothing changes.
    refused = client.post(f"/admin/api/models/{name}/maintained", headers=MANAGER)
    assert refused.status_code == 400 and refused.json()["code"] == "not_loadable"
    assert name in {row["name"] for row in config_io.load_model_info()["llm"]}
    assert directory.settings(config_io.load_config_full())["models"] == []
    real = config_io.loaded_shipped_rows
    monkeypatch.setattr(config_io, "loaded_shipped_rows", lambda: [*real(), {"name": "brand-new-fw"}])
    switched = client.post(f"/admin/api/models/{name}/maintained", headers=MANAGER)
    assert switched.status_code == 200 and switched.json()["name"] == "brand-new-fw"
    assert name not in {row["name"] for row in config_io.load_model_info()["llm"]}
    assert directory.settings(config_io.load_config_full())["models"] == ["brand-new-fw"]
