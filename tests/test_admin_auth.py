import json

from fastapi.testclient import TestClient

import src.admin as admin_module
import src.auth as auth_module
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


def test_v1_auth_accepts_private_client_key_file(monkeypatch, tmp_path):
    key_file = tmp_path / "client.keys"
    key_file.write_text("file-key\n")
    key_file.chmod(0o600)
    monkeypatch.delenv("MODEL_GATEWAY_CLIENT_KEYS", raising=False)
    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS_FILE", str(key_file))

    assert client.get("/v1/models").status_code == 401
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer file-key"},
    )
    assert response.status_code == 200


def test_v1_auth_file_misconfiguration_fails_closed(monkeypatch, tmp_path):
    key_file = tmp_path / "client.keys"
    key_file.write_text("exposed-key\n")
    key_file.chmod(0o644)
    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS", "fallback-key")
    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS_FILE", str(key_file))

    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer fallback-key"},
    )

    assert response.status_code == 503
    assert "misconfigured" in response.json()["detail"]


def test_client_key_file_rejects_unsafe_and_invalid_paths(monkeypatch, tmp_path):
    missing = tmp_path / "missing.keys"
    empty = tmp_path / "empty.keys"
    empty.write_text("")
    empty.chmod(0o600)
    oversized = tmp_path / "oversized.keys"
    oversized.write_text("x" * 65537)
    oversized.chmod(0o600)
    target = tmp_path / "target.keys"
    target.write_text("target-key\n")
    target.chmod(0o600)
    symlink = tmp_path / "linked.keys"
    symlink.symlink_to(target)
    fifo = tmp_path / "fifo.keys"
    fifo.parent.mkdir(parents=True, exist_ok=True)
    import os
    os.mkfifo(fifo, 0o600)

    for path in (missing, empty, oversized, symlink, fifo):
        with monkeypatch.context() as scoped:
            scoped.setenv("MODEL_GATEWAY_CLIENT_KEYS_FILE", str(path))
            keys, valid = auth_module._key_file_values("MODEL_GATEWAY_CLIENT_KEYS_FILE")
            assert keys == set()
            assert valid is False


def test_client_key_file_uses_one_open_inode_snapshot(monkeypatch, tmp_path):
    key_file = tmp_path / "client.keys"
    key_file.write_text("original-key\n")
    key_file.chmod(0o600)
    replacement = tmp_path / "replacement.keys"
    replacement.write_text("replacement-key\n")
    replacement.chmod(0o600)
    real_open = auth_module.os.open

    def swapping_open(path, flags):
        fd = real_open(path, flags)
        replacement.replace(key_file)
        return fd

    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS_FILE", str(key_file))
    monkeypatch.setattr(auth_module.os, "open", swapping_open)

    keys, valid = auth_module._key_file_values("MODEL_GATEWAY_CLIENT_KEYS_FILE")

    assert valid is True
    assert keys == {"original-key"}
    assert key_file.read_text().strip() == "replacement-key"


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


def test_admin_presets_endpoint_is_read_only_and_auth_protected(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin-key")
    model_info = tmp_path / "model-info.json"
    model_info.write_text(json.dumps({
        "llm": [],
        "auto_models": {
            "default_scope": "cloud",
            "cloud": {"model": "cloud-model", "vision_model": "cloud-model"},
        },
        "model_presets": {
            "default_tier": "best",
            "presets": {
                "best": {"cloud": {"text_model": "cloud-model", "vision_model": "cloud-model"}},
            },
        },
    }))
    monkeypatch.setattr(admin_module, "MODEL_INFO_PATH", model_info)

    assert client.get("/admin/api/presets").status_code == 401
    resp = client.get("/admin/api/presets", headers={"Authorization": "Bearer admin-key"})
    assert resp.status_code == 200
    body = resp.json()
    auto = body["auto_models"]
    presets = body["model_presets"]["presets"]
    assert auto["default_scope"] in {"cloud", "local"}
    assert auto[auto["default_scope"]]["model"]
    assert presets[body["model_presets"]["default_tier"]][auto["default_scope"]]["text_model"]


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


# ── non-loopback bind safety ─────────────────────────────────────────────────


def _clear_client_auth(monkeypatch):
    monkeypatch.delenv("MODEL_GATEWAY_CLIENT_KEYS", raising=False)
    monkeypatch.delenv("MODEL_GATEWAY_CLIENT_KEYS_FILE", raising=False)
    monkeypatch.delenv("MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_NONLOCAL", raising=False)
    monkeypatch.setattr(providers, "_config", {})


def test_bind_safety_allows_loopback_without_auth(monkeypatch):
    _clear_client_auth(monkeypatch)
    auth_module.check_bind_safety("127.0.0.1")
    auth_module.check_bind_safety("localhost")
    auth_module.check_bind_safety("::1")


def test_bind_safety_refuses_nonlocal_without_auth(monkeypatch):
    import pytest

    _clear_client_auth(monkeypatch)
    with pytest.raises(SystemExit, match="refusing to bind"):
        auth_module.check_bind_safety("0.0.0.0")
    with pytest.raises(SystemExit, match="refusing to bind"):
        auth_module.check_bind_safety("192.168.1.20")


def test_bind_safety_allows_nonlocal_with_client_keys(monkeypatch):
    _clear_client_auth(monkeypatch)
    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS", "some-key")
    auth_module.check_bind_safety("0.0.0.0")


def test_bind_safety_allows_nonlocal_with_config_keys(monkeypatch):
    _clear_client_auth(monkeypatch)
    monkeypatch.setattr(providers, "_config", {"auth": {"client_keys": ["cfg-key"]}})
    auth_module.check_bind_safety("0.0.0.0")


def test_bind_safety_explicit_unauthenticated_optout(monkeypatch):
    _clear_client_auth(monkeypatch)
    monkeypatch.setenv("MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_NONLOCAL", "true")
    auth_module.check_bind_safety("0.0.0.0")
