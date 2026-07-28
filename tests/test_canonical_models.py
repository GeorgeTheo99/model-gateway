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
