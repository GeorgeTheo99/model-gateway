import json

from fastapi.testclient import TestClient

import src.providers as providers
import src.server as server


client = TestClient(server.app)


def test_canonical_models_are_logical_safe_and_include_unavailable(monkeypatch, tmp_path):
    model_info = tmp_path / "model-info.json"
    model_info.write_text(json.dumps({
        "auto_models": {
            "default_scope": "cloud",
            "default_tier": "best",
            "cloud": {
                "model": "cloud-alias",
                "vision_model": "private-upstream-id",
                "label": "Best",
                "description": "Gateway cloud pair",
                "secret": "must-not-leak",
            },
            "local": {"model": "local-model"},
        },
        "model_presets": {
            "version": 1,
            "default_scope": "cloud",
            "default_tier": "best",
            "pricing": {"secret": True},
            "presets": {
                "best": {
                    "label": "Best",
                    "intent": "Highest quality",
                    "cloud": {
                        "text_model": "cloud-alias",
                        "vision_model": "private-upstream-id",
                        "source_policy": "quality",
                        "credential": "must-not-leak",
                    },
                    "local": {"text_model": "local-model"},
                }
            },
        },
    }))
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", model_info)
    monkeypatch.setattr(server, "effective_model_inventory", lambda: [
        {
            "name": "local-model",
            "provider": "omlx",
            "effective_provider": "omlx",
            "available": True,
            "enabled": True,
            "context": 262144,
            "max_output_tokens": 32768,
            "thinking": "optional",
            "thinking_levels": ["off", "high"],
            "vision": True,
            "path": "/private/model/path",
            "provider_model_id": "local-upstream",
            "routable_ids": ["local-model", "local-alias", "local-upstream"],
            "pricing": {"input": 0},
        },
        {
            "name": "cloud-model",
            "provider": "moonshot",
            "effective_provider": "moonshot",
            "available": False,
            "enabled": True,
            "availability_reason": "provider_not_configured",
            "availability_message": "Provider is not configured",
            "context": 1048576,
            "max_output_tokens": 131072,
            "thinking": "always",
            "thinking_levels": ["max"],
            "off_supported": False,
            "default_enabled_level": "max",
            "vision": True,
            "provider_model_id": "private-upstream-id",
            "routable_ids": ["cloud-model", "cloud-alias", "private-upstream-id"],
            "api_key": "must-not-leak",
        },
    ])

    response = client.get("/v1/models/canonical")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [row["id"] for row in body["data"]] == ["local-model", "cloud-model"]
    assert body["data"][0]["scope"] == "local"
    assert body["data"][1] == {
        "id": "cloud-model",
        "object": "model",
        "created": 0,
        "owned_by": "moonshot",
        "scope": "cloud",
        "available": False,
        "enabled": True,
        "availability_reason": "provider_not_configured",
        "availability_message": "",
        "context_length": 1048576,
        "max_output_tokens": 131072,
        "thinking": "always",
        "thinking_levels": ["max"],
        "off_supported": False,
        "default_enabled_level": "max",
        "vision": True,
    }
    assert body["auto_models"] == {
        "default_scope": "cloud",
        "default_tier": "best",
        "cloud": {
            "model": "cloud-model",
            "vision_model": "cloud-model",
            "label": "Best",
            "description": "Gateway cloud pair",
        },
    }
    assert body["model_presets"] == {
        "version": 1,
        "default_scope": "cloud",
        "default_tier": "best",
        "presets": {
            "best": {
                "label": "Best",
                "intent": "Highest quality",
                "cloud": {
                    "text_model": "cloud-model",
                    "vision_model": "cloud-model",
                    "source_policy": "quality",
                },
            }
        },
    }
    serialized = response.text
    for forbidden in (
        "local-alias",
        "local-upstream",
        "private-upstream-id",
        "/private/model/path",
        "must-not-leak",
        '"pricing"',
    ):
        assert forbidden not in serialized


def test_canonical_models_fail_safe_on_malformed_policy(monkeypatch, tmp_path):
    model_info = tmp_path / "model-info.json"
    model_info.write_text("{not-json")
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", model_info)
    monkeypatch.setattr(server, "effective_model_inventory", lambda: [{
        "name": "cloud-model",
        "provider": "openai",
        "effective_provider": "openai",
        "available": True,
        "enabled": True,
        "routable_ids": ["cloud-model"],
    }])

    response = client.get("/v1/models/canonical")

    assert response.status_code == 200
    assert [row["id"] for row in response.json()["data"]] == ["cloud-model"]
    assert response.json()["auto_models"] == {}
    assert response.json()["model_presets"] == {}


def test_canonical_models_drop_dangling_cloud_policy(monkeypatch, tmp_path):
    model_info = tmp_path / "model-info.json"
    model_info.write_text(json.dumps({
        "auto_models": {
            "default_scope": "cloud",
            "default_tier": "missing-tier",
            "cloud": {
                "model": "cloud-model",
                "vision_model": "unknown-alias",
                "label": "Dangling",
            },
        },
        "model_presets": {
            "version": 1,
            "default_scope": "cloud",
            "default_tier": "missing-tier",
            "presets": {
                "best": {
                    "label": "Dangling",
                    "cloud": {
                        "text_model": "cloud-model",
                        "vision_model": "unknown-alias",
                    },
                }
            },
        },
    }))
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", model_info)
    monkeypatch.setattr(server, "effective_model_inventory", lambda: [{
        "name": "cloud-model",
        "provider": "openai",
        "effective_provider": "openai",
        "declared_providers": ["openai"],
        "available": True,
        "enabled": True,
        "routable_ids": ["cloud-model"],
    }])

    response = client.get("/v1/models/canonical")

    assert response.status_code == 200
    assert response.json()["auto_models"] == {}
    assert response.json()["model_presets"] == {}


def test_canonical_models_skip_nameless_and_overlong_ids(monkeypatch, tmp_path):
    model_info = tmp_path / "model-info.json"
    model_info.write_text("{}")
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", model_info)
    monkeypatch.setattr(server, "effective_model_inventory", lambda: [
        {
            "id": "private/upstream-id",
            "provider": "openai",
            "effective_provider": "openai",
            "available": True,
        },
        {
            "name": "x" * 257,
            "provider": "openai",
            "effective_provider": "openai",
            "available": True,
        },
        {
            "name": "canonical",
            "provider": "openai",
            "effective_provider": "openai",
            "available": True,
        },
    ])

    response = client.get("/v1/models/canonical")

    assert response.status_code == 200
    assert [row["id"] for row in response.json()["data"]] == ["canonical"]


def test_canonical_models_never_label_mixed_pool_local(monkeypatch, tmp_path):
    model_info = tmp_path / "model-info.json"
    model_info.write_text("{}")
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", model_info)
    monkeypatch.setattr(server, "effective_model_inventory", lambda: [{
        "name": "hybrid-model",
        "provider": "omlx",
        "effective_provider": "omlx",
        "declared_providers": ["omlx", "openrouter"],
        "available": True,
        "enabled": True,
    }])

    response = client.get("/v1/models/canonical")

    assert response.status_code == 200
    row = response.json()["data"][0]
    assert row["scope"] == "cloud"
    assert row["owned_by"] == "model_gateway"


def test_effective_inventory_keeps_unconfigured_declared_pool_members(monkeypatch):
    entry = {
        "name": "hybrid-model",
        "provider": "omlx",
        "provider_model_id": "hybrid-model",
        "pool": "hybrid",
    }
    monkeypatch.setattr(providers, "_models", {"hybrid-model": entry})
    monkeypatch.setattr(providers, "_config", {
        "pools": {"hybrid": ["omlx", "openrouter"]},
        "providers": {},
    })

    inventory = providers.effective_model_inventory()
    row = inventory[0]
    canonical = server._canonical_model_rows(inventory)[0]

    assert row["candidate_providers"] == ["omlx"]
    assert row["declared_providers"] == ["omlx", "openrouter"]
    assert canonical["scope"] == "cloud"
    assert canonical["owned_by"] == "model_gateway"


def test_canonical_models_use_client_auth(monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS", "canonical-key")

    assert client.get("/v1/models/canonical").status_code == 401
    response = client.get(
        "/v1/models/canonical",
        headers={"Authorization": "Bearer canonical-key"},
    )
    assert response.status_code == 200


def test_canonical_models_add_client_metadata_only_when_included(monkeypatch, tmp_path):
    model_info = tmp_path / "model-info.json"
    model_info.write_text("{}")
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", model_info)
    monkeypatch.setattr(server, "effective_model_inventory", lambda: [
        {"name": "zdr-cloud", "provider": "openrouter", "effective_provider": "openrouter",
         "available": True, "display_name": "ZDR Cloud", "zdr": True},
        {"name": "retaining-cloud", "provider": "openrouter", "effective_provider": "openrouter",
         "available": True},
        {"name": "local-model", "provider": "omlx", "effective_provider": "omlx", "available": True},
    ])

    exact = client.get("/v1/models/canonical").json()["data"]
    assert all("display_name" not in row and "zero_data_retention" not in row for row in exact)

    rows = client.get(
        "/v1/models/canonical", params={"include": "display_name,zero_data_retention,unknown"},
    ).json()["data"]
    assert [(row["display_name"], row["zero_data_retention"]) for row in rows] == [
        ("ZDR Cloud", True), ("", False), ("", True),
    ]

    names_only = client.get("/v1/models/canonical", params={"include": "display_name"}).json()["data"]
    assert names_only[0]["display_name"] == "ZDR Cloud" and "zero_data_retention" not in names_only[0]


def test_canonical_models_add_route_grouping_only_when_included(monkeypatch, tmp_path):
    model_info = tmp_path / "model-info.json"
    model_info.write_text("{}")
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", model_info)
    monkeypatch.setattr(server, "effective_model_inventory", lambda: [
        {"name": "claude-opus-5.5", "provider": "anthropic", "effective_provider": "anthropic",
         "available": True, "family": "claude", "base_model": "claude-opus-5.5"},
        {"name": "claude-opus-5.5-or", "provider": "openrouter", "effective_provider": "openrouter",
         "available": True, "family": "claude", "base_model": "claude-opus-5.5"},
        {"name": "solo", "provider": "openrouter", "effective_provider": "openrouter", "available": True},
    ])

    exact = client.get("/v1/models/canonical").json()["data"]
    assert all("family" not in row and "base_model" not in row for row in exact)
    rows = client.get("/v1/models/canonical", params={"include": "family,base_model"}).json()["data"]
    assert [(row["id"], row["owned_by"], row["family"], row["base_model"]) for row in rows] == [
        ("claude-opus-5.5", "anthropic", "claude", "claude-opus-5.5"),
        ("claude-opus-5.5-or", "openrouter", "claude", "claude-opus-5.5"),
        ("solo", "openrouter", "", ""),
    ]
