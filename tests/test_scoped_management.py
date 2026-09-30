"""Scoped admin API access for manager consumer credentials."""

from __future__ import annotations

import json

import pytest
import yaml
from fastapi.testclient import TestClient

from src import auth, config_io
import src.providers as providers
from src.server import app

ADMIN = {"Authorization": "Bearer admin"}
MANAGER = {"Authorization": "Bearer manager-token"}
RUNTIME = {"Authorization": "Bearer runtime-token"}
CREATE = {"If-None-Match": "*"}
FIREWORKS = "https://api.fireworks.ai/inference/v1"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "auth": {
            "admin_keys": ["admin"],
            "consumer_credentials": [
                {"id": "ha-manager", "consumer": "ha", "key": "manager-token", "namespaces": ["ha"],
                 "permissions": ["providers:manage", "models:register"], "providers": ["fireworks"]},
                {"id": "ha-runtime", "consumer": "ha", "key": "runtime-token", "namespaces": ["ha"],
                 "permissions": ["profiles:read", "profiles:invoke"]},
            ],
        },
        "providers": {
            "anthropic": {"base_url": "https://api.anthropic.com/v1", "api_key": "anthropic-secret",
                          "protocol": "anthropic"},
            # Owner-defined, like the live config: no explicit protocol.
            "fireworks": {"base_url": FIREWORKS},
        },
    }))
    path.chmod(0o600)
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"llm": [
        {"name": "claude-test", "provider": "anthropic", "provider_model_id": "claude-test-1",
         "context": 200000, "max_output_tokens": 8192, "pricing": {"input": 3.0, "output": 15.0}},
    ]}))
    for module in (providers, config_io):
        monkeypatch.setattr(module, "CONFIG_PATH", path)
        monkeypatch.setattr(module, "MODEL_INFO_PATH", info)
        monkeypatch.setattr(module, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    config_io.log_dir = tmp_path / "logs"
    providers.reload()
    return path


@pytest.fixture
def client(cfg):
    with TestClient(app) as c:
        yield c


def _save_fireworks(client, headers=MANAGER, **body):
    return client.post("/admin/api/providers/fireworks", headers=headers,
                       json={"base_url": FIREWORKS, "protocol": "openai", "api_key": "fw-key", **body})


def _drop_provider(cfg, provider_id):
    config = yaml.safe_load(cfg.read_text())
    config["providers"].pop(provider_id, None)
    cfg.write_text(yaml.safe_dump(config))
    providers.reload()


def test_manager_status_is_reduced_and_full_admin_unchanged(client):
    scoped = client.get("/admin/api/status", headers=MANAGER)
    assert scoped.status_code == 200
    assert scoped.json() == {"service": "model-gateway", "status": "ok", "writes_enabled": True,
                             "capabilities": {"create_only_model_registration": True}}
    full = client.get("/admin/api/status", headers=ADMIN).json()
    assert "config_path" in full and "auth" in full
    assert client.get("/admin/api/status", headers=RUNTIME).status_code == 403


def test_manager_manages_only_allowlisted_provider_keys(client, cfg):
    assert _save_fireworks(client).status_code == 200
    assert "fw-key" not in cfg.read_text()
    rows = client.get("/admin/api/providers", headers=MANAGER).json()["providers"]
    assert [row["id"] for row in rows] == ["fireworks"] and rows[0]["has_api_key"] is True
    assert {row["id"] for row in client.get("/admin/api/providers", headers=ADMIN).json()["providers"]} >= {
        "anthropic", "fireworks"}
    # Rotating and clearing the key of an existing allowlisted provider are allowed.
    assert _save_fireworks(client, api_key="fw-key-2").status_code == 200
    key_file = config_io.api_key_file_target("fireworks")
    assert key_file.exists()
    assert _save_fireworks(client, api_key="").status_code == 200
    assert client.get("/admin/api/providers", headers=MANAGER).json()["providers"][0]["has_api_key"] is False
    assert not key_file.exists()
    # Other providers are neither writable nor testable.
    before = cfg.read_bytes()
    response = client.post("/admin/api/providers/anthropic", headers=MANAGER,
                           json={"base_url": "https://api.anthropic.com/v1", "api_key": "stolen"})
    assert response.status_code == 403
    assert client.post("/admin/api/providers/anthropic/validate", headers=MANAGER, json={}).status_code == 403
    assert cfg.read_bytes() == before


@pytest.mark.parametrize("change", [
    {"base_url": "https://attacker.example/v1"},
    {"protocol": "anthropic"},
    {"default_headers": {"X-Exfil": "1"}},
])
def test_manager_cannot_redirect_an_existing_provider(client, cfg, change):
    assert _save_fireworks(client).status_code == 200
    before = cfg.read_bytes()
    response = _save_fireworks(client, **change)
    assert response.status_code == 403
    assert cfg.read_bytes() == before
    # A full admin still can.
    assert _save_fireworks(client, headers=ADMIN, **change).status_code == 200


def test_manager_registers_only_new_models_for_its_providers(client):
    assert _save_fireworks(client).status_code == 200
    model = {"provider": "fireworks", "provider_model_id": "accounts/fireworks/models/glm"}
    url = "/admin/api/models/glm-fw"
    assert client.post(url, headers=MANAGER, json=model).status_code == 403  # upsert needs a full admin
    assert client.put(url, headers={**MANAGER, **CREATE}, json=model).status_code == 200
    assert client.put(url, headers={**MANAGER, **CREATE}, json=model).status_code == 412
    other = {"provider": "anthropic", "provider_model_id": "claude-new"}
    assert client.put("/admin/api/models/claude-new", headers={**MANAGER, **CREATE}, json=other).status_code == 403
    for missing in ({}, {"provider": ""}, {"provider": None}):
        response = client.put("/admin/api/models/local-x", headers={**MANAGER, **CREATE},
                              json={"omlx_id": "x", **missing})
        assert response.status_code == 403
    assert "claude-new" not in providers._load_models()


@pytest.mark.parametrize("method,path", [
    ("delete", "/admin/api/providers/fireworks"),
    ("delete", "/admin/api/models/claude-test"),
    ("post", "/admin/api/models/claude-test/disable"),
    ("post", "/admin/api/reload"),
    ("get", "/admin/api/consumers"),
    ("post", "/admin/api/consumers"),
    ("post", "/admin/api/consumers/ha-runtime/rotate"),
    ("get", "/admin/api/models"),
])
def test_manager_cannot_use_any_other_admin_action(client, method, path):
    kwargs = {"json": {}} if method == "post" else {}
    assert getattr(client, method)(path, headers=MANAGER, **kwargs).status_code == 401


def test_runtime_credentials_have_no_management_access(client):
    assert _save_fireworks(client, headers=RUNTIME).status_code == 403
    assert client.get("/admin/api/providers", headers=RUNTIME).status_code == 403


def test_manager_role_requires_allowlist_and_is_listed(client, cfg):
    created = client.post("/admin/api/consumers", headers=ADMIN,
                          json={"consumer": "hs", "role": "manager", "providers": ["fireworks"]})
    assert created.status_code == 200, created.text
    assert created.json()["providers"] == ["fireworks"]
    assert created.json()["permissions"] == ["providers:manage", "models:register"]
    missing = client.post("/admin/api/consumers", headers=ADMIN, json={"consumer": "hs2", "role": "manager"})
    assert missing.status_code == 400
    extra = client.post("/admin/api/consumers", headers=ADMIN,
                        json={"consumer": "hs3", "role": "runtime", "providers": ["fireworks"]})
    assert extra.status_code == 400
    listing = client.get("/admin/api/consumers", headers=ADMIN).json()
    assert listing["roles"]["manager"] == ["providers:manage", "models:register"]


@pytest.mark.parametrize("entry", [
    {"permissions": ["providers:manage"]},  # management without an allowlist
    {"permissions": ["profiles:read"], "providers": ["fireworks"]},  # allowlist without management
    {"permissions": ["providers:manage"], "providers": ["Fireworks!"]},
    {"permissions": ["providers:manage"], "providers": ["fireworks", "fireworks"]},
    {"permissions": ["providers:admin"], "providers": ["fireworks"]},
])
def test_invalid_management_grants_fail_closed(cfg, entry):
    config = yaml.safe_load(cfg.read_text())
    config["auth"]["consumer_credentials"] = [
        {"id": "bad-manager", "consumer": "bad", "key": "bad-token", "namespaces": ["bad"], **entry},
    ]
    cfg.write_text(yaml.safe_dump(config))
    providers.reload()
    with pytest.raises(auth.AuthConfigError):
        auth.validate_credential_separation()


def _grant(cfg, *allowed):
    config = yaml.safe_load(cfg.read_text())
    config["auth"]["consumer_credentials"][0]["providers"] = list(allowed)
    cfg.write_text(yaml.safe_dump(config))
    providers.reload()


def _set_providers(cfg, section, entries):
    config = yaml.safe_load(cfg.read_text())
    config.setdefault(section, {}).update(entries)
    cfg.write_text(yaml.safe_dump(config))
    providers.reload()


ATTACKER = "https://attacker.example/v1"


@pytest.mark.parametrize("setup,allowed,path", [
    ("synonym", "google", "google"),
    ("mixed-case", "fireworks", "fireworks"),
    ("workspace", "fireworks", "fireworks"),
    ("env", "fireworks", "fireworks"),
    ("builtin", "omlx", "omlx"),
    ("non-canonical-path", "fireworks", "Fireworks"),
    ("env-key-only", "fireworks", "fireworks"),
    ("undefined", "moonshot", "moonshot"),
])
def test_manager_cannot_redirect_providers_defined_elsewhere(client, cfg, monkeypatch, setup, allowed, path):
    _grant(cfg, allowed)
    if setup != "non-canonical-path":
        _drop_provider(cfg, "fireworks")
    if setup == "synonym":
        _set_providers(cfg, "providers", {"gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai"}})
    elif setup == "mixed-case":
        _set_providers(cfg, "providers", {"Fireworks": {"base_url": FIREWORKS}})
    elif setup == "workspace":
        _set_providers(cfg, "workspaces", {"fireworks": {"base_url": FIREWORKS}})
    elif setup == "env":
        monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_FIREWORKS_BASE_URL", FIREWORKS)
        monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_FIREWORKS_API_KEY", "env-secret")
    elif setup == "env-key-only":
        monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_FIREWORKS_API_KEY", "env-secret")
    before = cfg.read_bytes()
    response = client.post(f"/admin/api/providers/{path}", headers=MANAGER,
                           json={"base_url": ATTACKER, "api_key": "k"})
    assert response.status_code == 403, response.text
    assert cfg.read_bytes() == before
    effective = providers._effective_provider_config(providers._load_config(), allowed)
    assert effective.get("base_url") != ATTACKER


def test_manager_sets_key_on_env_defined_provider_without_moving_it(client, cfg, monkeypatch):
    _drop_provider(cfg, "fireworks")
    monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_FIREWORKS_BASE_URL", FIREWORKS)
    providers.reload()
    response = client.post("/admin/api/providers/fireworks", headers=MANAGER, json={"api_key": "fw-key"})
    assert response.status_code == 200, response.text
    assert yaml.safe_load(cfg.read_text())["providers"]["fireworks"]["base_url"] == FIREWORKS


def test_manager_updates_key_of_protocol_less_existing_provider(client, cfg):
    # The live fireworks block has no protocol; Home Server sends "openai".
    _set_providers(cfg, "providers", {"fireworks": {"base_url": FIREWORKS}})
    assert _save_fireworks(client).status_code == 200
    assert _save_fireworks(client, protocol=None, base_url=None, api_key="fw-2").status_code == 200
    assert _save_fireworks(client, base_url=FIREWORKS + "/").status_code == 200
    assert _save_fireworks(client, protocol="anthropic").status_code == 403
    for falsy in ("", False, 0):
        assert _save_fireworks(client, protocol=falsy).status_code in (400, 403), falsy
    assert yaml.safe_load(cfg.read_text())["providers"]["fireworks"].get("protocol", "openai") == "openai"
    assert _save_fireworks(client, base_url=5).status_code == 400


def test_manager_keeps_a_disabled_provider_disabled(client, cfg):
    _set_providers(cfg, "providers", {"fireworks": {"base_url": FIREWORKS, "enabled": False}})
    assert _save_fireworks(client).status_code == 200
    assert yaml.safe_load(cfg.read_text())["providers"]["fireworks"]["enabled"] is False


def test_create_only_refuses_federation_peer_namespace(client, monkeypatch):
    from types import SimpleNamespace
    from src import federation
    monkeypatch.setattr(federation.manager(), "config", SimpleNamespace(peers={"studio": object()}))
    assert _save_fireworks(client).status_code == 200
    # Names cannot contain "/" in the URL, so a peer id can only arrive as an alias or upstream id.
    for fields in ({"alias": "studio/glm"}, {"provider_model_id": "studio/glm"}):
        response = client.put("/admin/api/models/local-glm", headers={**MANAGER, **CREATE},
                              json={"provider": "fireworks", "provider_model_id": "glm", **fields})
        assert response.status_code == 409, (fields, response.text)
    assert client.put("/admin/api/models/studio-glm", headers={**ADMIN, **CREATE},
                      json={"provider": "fireworks", "provider_model_id": "glm"}).status_code == 200


def test_provider_key_write_cannot_overwrite_a_consumer_key_file(client, cfg, tmp_path):
    consumer_key = tmp_path / "ha-runtime.key"
    consumer_key.write_text("runtime-file-token\n")
    consumer_key.chmod(0o600)
    config = yaml.safe_load(cfg.read_text())
    config["auth"]["consumer_credentials"][1] = {
        **{k: v for k, v in config["auth"]["consumer_credentials"][1].items() if k != "key"},
        "key_file": str(consumer_key),
    }
    config["providers"]["fireworks"]["api_key_file"] = str(consumer_key)  # owner misconfiguration
    cfg.write_text(yaml.safe_dump(config))
    providers.reload()
    for headers in (MANAGER, ADMIN):
        assert _save_fireworks(client, headers=headers).status_code == 400
    assert consumer_key.read_text() == "runtime-file-token\n"
