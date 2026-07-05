from fastapi.testclient import TestClient

import src.providers as providers
from src.server import app


client = TestClient(app)


def test_v1_auth_is_open_by_default(monkeypatch):
    monkeypatch.delenv("MODEL_GATEWAY_CLIENT_KEYS", raising=False)
    resp = client.get("/v1/models")
    assert resp.status_code == 200


def test_v1_auth_requires_configured_client_key(monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS", "other-key, good-key")

    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer bad"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer good-key"}).status_code == 200
    assert client.get("/v1/models", headers={"x-api-key": "good-key"}).status_code == 200


def test_v1_auth_accepts_admin_key(monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS", "client-key")
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin-key")

    assert client.get("/v1/debug/thinking", headers={"Authorization": "Bearer admin-key"}).status_code == 200


def test_admin_api_fails_closed_without_admin_key(monkeypatch):
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.delenv("MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN", raising=False)

    assert client.get("/admin").status_code == 200
    assert client.get("/admin/api/status").status_code == 401


def test_admin_auth_requires_configured_admin_key(monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin-key")

    assert client.get("/admin/api/status").status_code == 401
    resp = client.get("/admin/api/status", headers={"Authorization": "Bearer admin-key"})
    assert resp.status_code == 200
    assert resp.json()["auth"]["admin_auth_enabled"] is True


def test_admin_presets_endpoint_is_read_only_and_auth_protected(monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin-key")

    assert client.get("/admin/api/presets").status_code == 401
    resp = client.get("/admin/api/presets", headers={"Authorization": "Bearer admin-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["auto_models"]["cloud"]["vision_model"] == "qwen3.7-plus-fw"
    assert body["model_presets"]["presets"]["best"]["cloud"]["vision_model"] == "qwen3.7-plus-fw"


def test_admin_provider_status_never_exposes_secret_values(monkeypatch):
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.setenv("MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN", "true")
    monkeypatch.setattr(providers, "_config", {
        "providers": {
            "openai": {
                "base_url": "https://user:pass@example.invalid/v1?api_key=query-secret#token=fragment-secret",
                "api_key": "super-secret-test-key",
                "custom_token": "another-secret",
                "password": "secret-password",
                "default_headers": {"Authorization": "Bearer nested-secret"},
            }
        }
    })
    monkeypatch.setattr(providers, "_models", {
        "test-model": {
            "name": "test-model",
            "provider": "openai",
            "provider_model_id": "upstream-test-model",
        }
    })

    resp = client.get("/admin/api/providers")
    assert resp.status_code == 200
    body = resp.text
    assert "super-secret-test-key" not in body
    assert "another-secret" not in body
    assert "secret-password" not in body
    assert "nested-secret" not in body
    assert "user:pass" not in body
    assert "query-secret" not in body
    assert "fragment-secret" not in body
    provider = resp.json()["providers"][0]
    assert provider["has_api_key"] is True
    assert provider["base_url"] == "https://example.invalid/v1"
    assert "config" not in provider


def test_config_client_keys_protect_v1(monkeypatch):
    """client_keys in config.yaml protect /v1 even without env."""
    monkeypatch.delenv("MODEL_GATEWAY_CLIENT_KEYS", raising=False)
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.setattr(providers, "_config", {"auth": {"client_keys": ["cfg-cloud"]}})

    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer cfg-cloud"}).status_code == 200


def test_config_admin_keys_protect_admin_api(monkeypatch):
    """admin_keys in config.yaml protect /admin/api even without env."""
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.setattr(providers, "_config", {"auth": {"admin_keys": ["cfg-admin"]}})

    assert client.get("/admin/api/status").status_code == 401
    resp = client.get("/admin/api/status", headers={"Authorization": "Bearer cfg-admin"})
    assert resp.status_code == 200
    assert resp.json()["auth"]["admin_auth_enabled"] is True


def test_env_and_config_keys_are_merged(monkeypatch):
    """Env keys and config keys union; either is accepted."""
    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS", "env-key")
    monkeypatch.setattr(providers, "_config", {"auth": {"client_keys": ["cfg-key"]}})

    assert client.get("/v1/models", headers={"Authorization": "Bearer env-key"}).status_code == 200
    assert client.get("/v1/models", headers={"Authorization": "Bearer cfg-key"}).status_code == 200
    assert client.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_config_client_keys_accepts_comma_string(monkeypatch):
    """A comma-separated string is also accepted for config keys."""
    monkeypatch.delenv("MODEL_GATEWAY_CLIENT_KEYS", raising=False)
    monkeypatch.setattr(providers, "_config", {"auth": {"client_keys": "alpha, beta"}})

    assert client.get("/v1/models", headers={"Authorization": "Bearer alpha"}).status_code == 200
    assert client.get("/v1/models", headers={"x-api-key": "beta"}).status_code == 200
    assert client.get("/v1/models").status_code == 401
