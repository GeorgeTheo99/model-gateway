from fastapi.testclient import TestClient

import src.providers as providers
from src.server import app


client = TestClient(app)


def test_v1_auth_is_open_by_default(monkeypatch):
    monkeypatch.delenv("CLOUD_GATEWAY_CLIENT_KEYS", raising=False)
    resp = client.get("/v1/models")
    assert resp.status_code == 200


def test_v1_auth_requires_configured_client_key(monkeypatch):
    monkeypatch.setenv("CLOUD_GATEWAY_CLIENT_KEYS", "other-key, good-key")

    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer bad"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer good-key"}).status_code == 200
    assert client.get("/v1/models", headers={"x-api-key": "good-key"}).status_code == 200


def test_v1_auth_accepts_admin_key(monkeypatch):
    monkeypatch.setenv("CLOUD_GATEWAY_CLIENT_KEYS", "client-key")
    monkeypatch.setenv("CLOUD_GATEWAY_ADMIN_KEY", "admin-key")

    assert client.get("/v1/debug/thinking", headers={"Authorization": "Bearer admin-key"}).status_code == 200


def test_admin_api_fails_closed_without_admin_key(monkeypatch):
    monkeypatch.delenv("CLOUD_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.delenv("CLOUD_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN", raising=False)

    assert client.get("/admin").status_code == 200
    assert client.get("/admin/api/status").status_code == 401


def test_admin_auth_requires_configured_admin_key(monkeypatch):
    monkeypatch.setenv("CLOUD_GATEWAY_ADMIN_KEY", "admin-key")

    assert client.get("/admin/api/status").status_code == 401
    resp = client.get("/admin/api/status", headers={"Authorization": "Bearer admin-key"})
    assert resp.status_code == 200
    assert resp.json()["auth"]["admin_auth_enabled"] is True


def test_admin_provider_status_never_exposes_secret_values(monkeypatch):
    monkeypatch.delenv("CLOUD_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.setenv("CLOUD_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN", "true")
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
