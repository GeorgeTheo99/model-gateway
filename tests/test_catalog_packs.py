"""Consumer catalog packs: product-owned models between the shipped layer and the owner (ADR 0002)."""

import json

import pytest
import yaml
from fastapi.testclient import TestClient

import src.providers as providers
from src import config_io, onboarding, packs, shipped
from src.server import app

BASE_URL = "https://api.example.test/v1"
HA = {"Authorization": "Bearer ha-deployer"}
MYAI = {"Authorization": "Bearer myai-deployer"}
QWEN = {"name": "qwen-local", "provider": "omlx", "omlx_id": "qwen-8bit", "alias": "qwen",
        "pricing_status": "unmetered", "context": 32768}


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    curated = onboarding.validate_profile({
        "schema_version": 1, "id": "example-curated", "auto_add": True,
        "provider": {"id": "example", "base_url": BASE_URL, "protocol": "openai"},
        "models": [{"name": "cloud-fw", "provider": "example", "provider_model_id": "up/cloud", "alias": "cloudfw",
                    "family": "cloud", "base_model": "cloud", "pricing": {"input": 1.0, "output": 2.0}}],
    })
    monkeypatch.setattr(shipped, "_packaged", lambda: {"example-curated": curated})
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"llm": [], "allow_empty": True}))
    for module in (providers, config_io):
        monkeypatch.setattr(module, "MODEL_INFO_PATH", info)
    monkeypatch.setattr(config_io, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin")
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    credentials = [
        {"id": f"{ns}-deployer", "consumer": ns, "key": f"{ns}-deployer", "namespaces": [ns],
         "permissions": ["profiles:read", "profiles:write"]}
        for ns in ("ha", "myai")
    ]
    providers.CONFIG_PATH.write_text(yaml.safe_dump({
        "auth": {"consumer_credentials": credentials},
        "providers": {"example": {"base_url": BASE_URL, "api_key": "k"},
                      "omlx": {"base_url": "http://127.0.0.1:9110/v1", "api_key": "local"}},
    }))
    providers.reload()
    with TestClient(app) as client:
        yield client, info


def _pack(*models, requires=(), namespace="ha"):
    return {"schema_version": 1, "namespace": namespace, "source_revision": "home-server@1",
            "models": list(models), "requires": list(requires)}


def test_register_a_pack_and_report_requirements(gateway):
    client, info = gateway
    response = client.put("/v1/catalog/packs/ha", headers=HA, json=_pack(QWEN, requires=["cloud", "missing"]))
    assert response.status_code == 200, response.text
    assert response.json()["unmet_requires"] == ["missing"]  # "cloud" resolves as a base model
    models = providers._load_models()
    assert models["qwen"]["name"] == "qwen-local"
    assert models["qwen-local"]["origin"] == {"layer": "pack", "namespace": "ha"}
    assert json.loads(info.read_text())["llm"] == []  # nothing lands in the owner's catalog
    assert client.get("/v1/catalog/packs/ha", headers=HA).json()["models"][0]["name"] == "qwen-local"
    status = client.get("/admin/api/status", headers={"Authorization": "Bearer admin"}).json()
    assert status["catalog"]["packs"]["ha"] == {"source_revision": "home-server@1", "models": ["qwen-local"],
                                                "unmet_requires": ["missing"]}


@pytest.mark.parametrize("model,namespace_pack", [
    ({**QWEN, "name": "cloud-fw"}, None),            # a shipped name
    ({**QWEN, "alias": "cloudfw"}, None),            # a shipped identifier
    ({**QWEN, "name": "other", "alias": "q2"}, "myai"),  # another pack's upstream id
])
def test_conflicts_are_refused_without_writing(gateway, model, namespace_pack):
    client, _info = gateway
    if namespace_pack:
        assert client.put(f"/v1/catalog/packs/{namespace_pack}", headers=MYAI,
                          json=_pack(QWEN, namespace=namespace_pack)).status_code == 200
    before = packs.path().read_text() if packs.path().exists() else None
    response = client.put("/v1/catalog/packs/ha", headers=HA, json=_pack(model))
    assert response.status_code == 409 and response.json()["error"]["code"] == "pack_conflict", response.text
    assert (packs.path().read_text() if packs.path().exists() else None) == before


def test_a_credential_registers_only_its_namespace(gateway):
    client, _info = gateway
    assert client.put("/v1/catalog/packs/ha", headers=MYAI, json=_pack(QWEN)).status_code == 403
    assert client.put("/v1/catalog/packs/ha", json=_pack(QWEN)).status_code == 401
    assert not packs.path().exists()


@pytest.mark.parametrize("bad", [
    {**QWEN, "pricing_status": "free"}, {k: v for k, v in QWEN.items() if k != "omlx_id"},
    {**QWEN, "origin": {"layer": "shipped"}}, {**QWEN, "shipped_base": {}},
])
def test_invalid_models_are_refused(gateway, bad):
    client, _info = gateway
    response = client.put("/v1/catalog/packs/ha", headers=HA, json=_pack(bad))
    assert response.status_code == 422, response.text
    assert not packs.path().exists()


def test_the_owner_overrides_and_hides_pack_models(gateway):
    client, info = gateway
    assert client.put("/v1/catalog/packs/ha", headers=HA, json=_pack(QWEN)).status_code == 200
    config_io.upsert_model("qwen-local", provider="omlx", omlx_id="qwen-8bit", context=8192)
    providers.reload()
    entry = providers._load_models()["qwen-local"]
    assert entry["context"] == 8192 and entry["origin"]["overridden"] == ["context"]
    assert json.loads(info.read_text())["llm"] == [{"name": "qwen-local", "context": 8192, "shipped_base": {"context": 32768}}]
    config_io.delete_model("qwen-local")
    providers.reload()
    assert "qwen-local" not in providers._load_models()


def test_re_registering_replaces_and_delete_removes(gateway):
    client, _info = gateway
    second = {**QWEN, "name": "qwen-small", "omlx_id": "qwen-4bit", "alias": "qwens"}
    assert client.put("/v1/catalog/packs/ha", headers=HA, json=_pack(QWEN, second)).status_code == 200
    assert client.put("/v1/catalog/packs/ha", headers=HA, json=_pack(second)).status_code == 200
    assert "qwen-local" not in providers._load_models() and "qwen-small" in providers._load_models()
    assert client.delete("/v1/catalog/packs/ha", headers=HA).status_code == 204
    assert "qwen-small" not in providers._load_models()
    assert client.delete("/v1/catalog/packs/ha", headers=HA).status_code == 404


def test_a_pack_never_removes_or_rewrites_owner_models(gateway):
    client, info = gateway
    mine = {"name": "mine", "provider": "omlx", "omlx_id": "local-x", "pricing_status": "unmetered"}
    info.write_text(json.dumps({"llm": [dict(QWEN), mine], "allow_empty": True}))  # QWEN: a legacy import's copy
    providers.reload()
    # A pack may name the owner's models; the owner's rows still win whole.
    assert client.put("/v1/catalog/packs/ha", headers=HA, json=_pack(QWEN)).status_code == 200
    assert client.put("/v1/catalog/packs/myai", headers=MYAI, json=_pack(
        {**mine, "omlx_id": "attacker", "vision": True, "alias": "qwen-x"}, namespace="myai")).status_code == 200
    entry = providers._load_models()["mine"]
    assert (entry["omlx_id"], entry.get("vision"), entry["origin"]) == ("local-x", None, {"layer": "owner"})
    with TestClient(app):  # the startup migration ignores packs
        pass
    # A routine owner edit keeps the owner's row whole, never a sparse override of the pack's.
    config_io.upsert_model("qwen-local", provider="omlx", omlx_id="qwen-8bit", context=8192)
    for namespace, headers in (("ha", HA), ("myai", MYAI)):
        assert client.delete(f"/v1/catalog/packs/{namespace}", headers=headers).status_code == 204
    rows = {row["name"]: row for row in json.loads(info.read_text())["llm"]}
    assert rows["mine"] == mine and rows["qwen-local"]["context"] == 8192 and "shipped_base" not in rows["qwen-local"]
    assert providers.resolve("qwen-local").provider_model_id == "qwen-8bit"
    assert providers.resolve("mine").provider_model_id == "local-x"


def test_an_owner_override_does_not_block_the_products_next_registration(gateway):
    client, info = gateway
    assert client.put("/v1/catalog/packs/ha", headers=HA, json=_pack(QWEN)).status_code == 200
    config_io.upsert_model("qwen-local", provider="omlx", omlx_id="qwen-8bit", context=8192)
    providers.reload()
    assert client.put("/v1/catalog/packs/ha", headers=HA, json=_pack(QWEN)).status_code == 200
    assert providers._load_models()["qwen-local"]["context"] == 8192


def test_a_pack_yields_to_a_base_model_name_a_release_later_ships(gateway, monkeypatch):
    client, _info = gateway
    assert client.put("/v1/catalog/packs/ha", headers=HA, json=_pack({**QWEN, "name": "newbase"})).status_code == 200
    later = onboarding.validate_profile({
        "schema_version": 1, "id": "later-curated", "auto_add": True,
        "provider": {"id": "example", "base_url": BASE_URL, "protocol": "openai"},
        "models": [{"name": "newbase-fw", "provider": "example", "provider_model_id": "up/newbase",
                    "family": "newbase", "base_model": "newbase", "pricing": {"input": 1.0, "output": 2.0}}],
    })
    curated = shipped._packaged()
    monkeypatch.setattr(shipped, "_packaged", lambda: {**curated, "later-curated": later})
    providers.reload()
    assert providers.resolve("newbase").provider_model_id == "up/newbase"


@pytest.mark.parametrize("field", ["system_instruction", "quirks", "pool", "api_style", "fallbacks"])
def test_pack_models_cannot_set_fields_that_reach_other_consumers(gateway, field):
    client, _info = gateway
    response = client.put("/v1/catalog/packs/ha", headers=HA, json=_pack({**QWEN, field: "x"}))
    assert response.status_code == 422 and field in response.text


def test_base_model_names_stay_with_the_release_and_the_owner(gateway):
    client, _info = gateway
    # A pack model named after a shipped base model, or routing that base model, steers nothing.
    named = client.put("/v1/catalog/packs/myai", headers=MYAI, json=_pack({**QWEN, "name": "cloud"}, namespace="myai"))
    assert named.status_code == 409
    routed = {**QWEN, "family": "cloud", "base_model": "cloud"}
    assert client.put("/v1/catalog/packs/myai", headers=MYAI, json=_pack(routed, namespace="myai")).status_code == 200
    providers.CONFIG_PATH.write_text(yaml.safe_dump({
        **yaml.safe_load(providers.CONFIG_PATH.read_text()), "catalog": {"provider_preference": ["omlx"]}}))
    providers.reload()
    assert providers.resolve_base_model("cloud") == "cloud-fw"


def test_identifiers_of_shipped_models_not_loaded_here_are_reserved(gateway, monkeypatch):
    client, _info = gateway
    other = onboarding.validate_profile({
        "schema_version": 1, "id": "other-curated", "auto_add": True,
        "provider": {"id": "other", "base_url": "https://other.test/v1", "protocol": "openai"},
        "models": [{"name": "other-m", "provider": "other", "provider_model_id": "up/other-m"}],
    })
    curated = shipped._packaged()
    monkeypatch.setattr(shipped, "_packaged", lambda: {**curated, "other-curated": other})
    providers.reload()
    assert "other-m" not in providers._load_models()  # no key for "other" yet
    response = client.put("/v1/catalog/packs/ha", headers=HA, json=_pack({**QWEN, "name": "other-m"}))
    assert response.status_code == 409


def test_a_release_that_ships_a_pack_identifier_wins_without_breaking_the_catalog(gateway, monkeypatch):
    client, _info = gateway
    assert client.put("/v1/catalog/packs/ha", headers=HA, json=_pack({**QWEN, "name": "later"})).status_code == 200
    later = onboarding.validate_profile({
        "schema_version": 1, "id": "later-curated", "auto_add": True,
        "provider": {"id": "example", "base_url": BASE_URL, "protocol": "openai"},
        "models": [{"name": "later", "provider": "example", "provider_model_id": "up/later",
                    "pricing": {"input": 1.0, "output": 2.0}}],
    })
    curated = shipped._packaged()
    monkeypatch.setattr(shipped, "_packaged", lambda: {**curated, "later-curated": later})
    providers.reload()
    assert providers._load_models()["later"]["origin"]["layer"] == "shipped"
