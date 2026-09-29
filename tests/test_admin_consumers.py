"""Admin API for consumer credentials, profile snapshots, backups, and bundles."""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from src import admin, auth, config_io
import src.providers as providers
from src.server import app

ADMIN = {"Authorization": "Bearer admin-token"}


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for name in ("MODEL_GATEWAY_CLIENT_KEYS", "MODEL_GATEWAY_CLIENT_KEYS_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin-token")
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    path = Path(os.path.realpath(tmp_path)) / "gateway" / "config.yaml"
    path.parent.mkdir()
    path.write_text("auth:\n  client_keys: [client-token]\nproviders: {}\n")
    path.chmod(0o600)
    monkeypatch.setattr(providers, "CONFIG_PATH", path)
    monkeypatch.setattr(config_io, "CONFIG_PATH", path)
    providers.reload()
    return path


@pytest.fixture
def client(cfg):
    with TestClient(app) as c:
        yield c


def _key_file(cfg: Path, credential_id: str) -> Path:
    return cfg.parent / "secrets" / "consumers" / f"{credential_id}.key"


def _consumer_tokens() -> dict[str, str]:
    providers.reload()
    consumers, _clients, _admins = auth.validate_credential_separation()
    return {principal.credential_id: token for token, principal in consumers}


def _add(client, consumer="myai", role="runtime", **extra):
    return client.post("/admin/api/consumers", headers=ADMIN, json={"consumer": consumer, "role": role, **extra})


def test_add_returns_generated_key_once_uncached(client, cfg):
    resp = _add(client, allow_direct_models=True)
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    body = resp.json()
    assert body["status"] == "created" and body["allow_direct_models"] is True
    assert body["key"] == _key_file(cfg, "myai-runtime").read_text().strip()
    assert _consumer_tokens() == {"myai-runtime": body["key"]}

    again = _add(client, allow_direct_models=True)
    assert again.status_code == 200
    assert again.json()["status"] == "unchanged" and "key" not in again.json()

    listing = client.get("/admin/api/consumers", headers=ADMIN)
    assert listing.status_code == 200
    assert body["key"] not in listing.text
    (row,) = listing.json()["consumers"]
    assert row["id"] == "myai-runtime" and row["key_status"] == "ok"
    assert row["managed_key_file"] is True
    assert listing.json()["roles"]["deployer"] == ["profiles:read", "profiles:write"]


def test_add_never_reveals_an_adopted_key_file(client, cfg):
    target = _key_file(cfg, "pi-runtime")
    target.parent.mkdir(parents=True, mode=0o700)
    target.write_text("adopted-token\n")
    target.chmod(0o600)
    resp = _add(client, consumer="pi")
    assert resp.status_code == 200
    assert resp.json()["status"] == "created"
    assert "key" not in resp.json() and "adopted-token" not in resp.text


def test_add_rejects_bad_input_and_refreshes_discovery(client, cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(admin.discovery, "refresh", lambda: calls.append(1))
    before = cfg.read_text()
    assert _add(client, consumer="Bad Id").status_code == 400
    assert _add(client, role="owner").status_code == 400
    assert _add(client, allow_direct_models="yes").status_code == 400
    too_long = _add(client, consumer="a" * 24, role="deployer")
    assert too_long.status_code == 400
    assert "at most 23" in too_long.json()["error"]["message"]
    assert _add(client, consumer="a" * 24).status_code == 200  # 32-character runtime id
    for namespaces in ([["x"]], {}, "myai", [1]):
        resp = _add(client, consumer="lab", namespaces=namespaces)
        assert resp.status_code == 400, namespaces
    assert _add(client, consumer="lab", namespaces=["lab", "lab"]).status_code == 400
    assert calls == [1]
    calls.clear()
    cfg.write_text(before)
    providers.reload()
    assert calls == []
    assert _add(client).status_code == 200
    assert calls == [1]


def test_rotate_requires_confirmation_and_invalidates_old_key(client, cfg):
    old = _add(client).json()["key"]
    assert client.post("/admin/api/consumers/myai-runtime/rotate", headers=ADMIN).status_code == 428
    wrong = client.post("/admin/api/consumers/myai-runtime/rotate", headers=ADMIN, json={"confirm": "ha-runtime"})
    assert wrong.status_code == 428
    assert _consumer_tokens() == {"myai-runtime": old}

    resp = client.post("/admin/api/consumers/myai-runtime/rotate", headers=ADMIN, json={"confirm": "myai-runtime"})
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    new = resp.json()["key"]
    assert resp.json()["status"] == "rotated" and new != old
    assert _key_file(cfg, "myai-runtime").stat().st_mode & 0o777 == 0o600
    assert _consumer_tokens() == {"myai-runtime": new}

    # The running gateway authenticates the new key and rejects the old one.
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {new}"}).status_code == 200
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {old}"}).status_code == 401


def test_rotate_refuses_unmanaged_inline_and_unknown_credentials(client, cfg, tmp_path):
    external = tmp_path / "external.key"
    external.write_text("external-token\n")
    external.chmod(0o600)
    config = yaml.safe_load(cfg.read_text())
    config["auth"]["consumer_credentials"] = [
        {"id": "ha-runtime", "consumer": "ha", "key_file": str(external),
         "namespaces": ["ha"], "permissions": ["profiles:read", "profiles:invoke"]},
        {"id": "ha-deployer", "consumer": "ha", "key": "inline-token",
         "namespaces": ["ha"], "permissions": ["profiles:read", "profiles:write"]},
    ]
    cfg.write_text(yaml.safe_dump(config))
    providers.reload()
    for credential_id in ("ha-runtime", "ha-deployer"):
        resp = client.post(f"/admin/api/consumers/{credential_id}/rotate", headers=ADMIN,
                           json={"confirm": credential_id})
        assert resp.status_code == 400
        assert "managed key file" in resp.json()["error"]["message"]
    assert external.read_text() == "external-token\n"
    missing = client.post("/admin/api/consumers/nobody-runtime/rotate", headers=ADMIN,
                          json={"confirm": "nobody-runtime"})
    assert missing.status_code == 404


def test_rotate_rolls_back_when_registry_reload_fails(client, cfg, monkeypatch):
    old = _add(client).json()["key"]
    monkeypatch.setattr(admin, "_reload_registry_transactionally", lambda snapshot: "boom")
    resp = client.post("/admin/api/consumers/myai-runtime/rotate", headers=ADMIN, json={"confirm": "myai-runtime"})
    assert resp.status_code == 400
    assert "rolled back" in resp.json()["error"]["message"]
    assert "key" not in resp.json()
    assert _key_file(cfg, "myai-runtime").read_text().strip() == old


def test_rotate_refuses_symlinked_managed_key_file(client, cfg, tmp_path):
    _add(client)
    key_file = _key_file(cfg, "myai-runtime")
    real = tmp_path / "real.key"
    real.write_text(key_file.read_text())
    real.chmod(0o600)
    key_file.unlink()
    key_file.symlink_to(real)
    resp = client.post("/admin/api/consumers/myai-runtime/rotate", headers=ADMIN, json={"confirm": "myai-runtime"})
    assert resp.status_code == 400 and "key" not in resp.json()
    assert key_file.is_symlink() and real.read_text() == key_file.read_text()


def test_rotate_rolls_back_when_auth_validation_fails(client, cfg, monkeypatch):
    old = _add(client).json()["key"]

    def reject():
        raise auth.AuthConfigError("inbound authentication configuration is invalid")

    with monkeypatch.context() as patched:
        patched.setattr(auth, "validate_credential_separation", reject)
        resp = client.post("/admin/api/consumers/myai-runtime/rotate", headers=ADMIN,
                           json={"confirm": "myai-runtime"})
    assert resp.status_code == 400 and "key" not in resp.json()
    assert _key_file(cfg, "myai-runtime").read_text().strip() == old
    assert _consumer_tokens() == {"myai-runtime": old}


def test_add_and_revoke_roll_back_when_registry_reload_fails(client, cfg, monkeypatch):
    kept = _add(client, consumer="ha").json()["key"]
    before = cfg.read_text()
    with monkeypatch.context() as patched:
        patched.setattr(admin, "_reload_registry_transactionally", lambda snapshot: "boom")
        added = _add(client)
        revoked = client.request("DELETE", "/admin/api/consumers/ha-runtime", headers=ADMIN,
                                 json={"confirm": "ha-runtime"})
    assert added.status_code == 400 and "key" not in added.json()
    assert not _key_file(cfg, "myai-runtime").exists()
    assert revoked.status_code == 400
    assert cfg.read_text() == before
    restored = _key_file(cfg, "ha-runtime")
    assert restored.read_text().strip() == kept
    assert restored.stat().st_mode & 0o777 == 0o600
    assert _consumer_tokens() == {"ha-runtime": kept}


def test_revoke_requires_confirmation_and_keeps_last_v1_credential(client, cfg):
    _add(client)
    assert client.request("DELETE", "/admin/api/consumers/myai-runtime", headers=ADMIN).status_code == 428
    resp = client.request("DELETE", "/admin/api/consumers/myai-runtime", headers=ADMIN,
                          json={"confirm": "myai-runtime"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "revoked" and resp.json()["key_file_deleted"] is True
    assert not _key_file(cfg, "myai-runtime").exists()

    # With the legacy client key gone, the only consumer is the last /v1 key.
    _add(client, consumer="ha")
    config = yaml.safe_load(cfg.read_text())
    config["auth"].pop("client_keys")
    cfg.write_text(yaml.safe_dump(config))
    providers.reload()
    last = client.request("DELETE", "/admin/api/consumers/ha-runtime", headers=ADMIN, json={"confirm": "ha-runtime"})
    assert last.status_code == 400
    assert "last /v1 credential" in last.json()["error"]["message"]
    assert _key_file(cfg, "ha-runtime").exists()


def test_consumer_writes_need_admin_auth_and_write_mode(client, monkeypatch):
    assert client.get("/admin/api/consumers").status_code == 401
    assert client.post("/admin/api/consumers", json={"consumer": "x", "role": "runtime"}).status_code == 401
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_WRITES")
    assert _add(client).status_code == 403
    assert client.post("/admin/api/consumers/x-runtime/rotate", headers=ADMIN,
                       json={"confirm": "x-runtime"}).status_code == 403
    assert client.request("DELETE", "/admin/api/consumers/x-runtime", headers=ADMIN,
                          json={"confirm": "x-runtime"}).status_code == 403
    assert client.get("/admin/api/consumers", headers=ADMIN).status_code == 200


def _record(namespace: str, version: int, revision: str) -> dict:
    return {
        "gateway_version": version, "registered_at": f"2026-09-2{version}T00:00:00Z",
        "etag": f'"e{version}"', "manifest_digest": f"digest-{version}", "bindings": {"x": "internal-binding"},
        "manifest": {
            "schema_version": 1, "namespace": namespace, "source_revision": revision,
            "default_profile": f"{namespace}/auto",
            "profiles": [{"id": f"{namespace}/auto", "locality": "local_only", "credential_policy": "gateway_local",
                          "protocols": ["openai_chat"], "routes": {"text": "local-model"}}],
        },
    }


def test_profiles_lists_latest_snapshot_and_history(client, tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"format": 1, "namespaces": {
        "myai": [_record("myai", 1, "a"), _record("myai", 2, "b")], "empty": [],
    }}))
    monkeypatch.setenv("MODEL_GATEWAY_PROFILE_REGISTRY", str(registry))
    resp = client.get("/admin/api/profiles", headers=ADMIN)
    assert resp.status_code == 200
    (entry,) = resp.json()["namespaces"]
    assert entry["latest"]["gateway_version"] == 2
    assert entry["latest"]["source_revision"] == "b"
    assert entry["latest"]["profiles"][0]["executable"] is True
    assert [v["gateway_version"] for v in entry["versions"]] == [2, 1]
    assert "internal-binding" not in resp.text

    registry.write_text("{not json")
    assert client.get("/admin/api/profiles", headers=ADMIN).status_code == 503
    assert client.get("/admin/api/profiles").status_code == 401


def test_backups_summarizes_generations_without_reading_them(client, tmp_path, monkeypatch):
    backups = tmp_path / "config-backups"
    (backups / "legacy-config-backups-1").mkdir(parents=True)
    (backups / "manual-snapshot").mkdir()
    for name in ("config.yaml.bak.100", "config.yaml.bak.200", "legacy-config-backups-1/model-info.json.bak.300"):
        (backups / name).write_text("secret-content")
    os.utime(backups / "config.yaml.bak.100", (1000, 1000))
    os.utime(backups / "config.yaml.bak.200", (2000, 2000))
    (backups / "config.yaml.bak.999").symlink_to(backups / "config.yaml.bak.100")
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(backups))
    resp = client.get("/admin/api/backups", headers=ADMIN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["directory"] == str(backups) and body["exists"] is True and body["retention"] == 20
    by_file = {row["file"]: row for row in body["generations"]}
    assert by_file["config.yaml"]["count"] == 2
    assert (by_file["config.yaml"]["oldest"], by_file["config.yaml"]["newest"]) == (1000, 2000)
    assert by_file["model-info.json"]["count"] == 1
    assert body["other"]["count"] == 1
    assert "secret-content" not in resp.text


def test_backups_reports_a_missing_directory(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(tmp_path / "absent"))
    body = client.get("/admin/api/backups", headers=ADMIN).json()
    assert body["exists"] is False and body["generations"] == [] and body["other"]["count"] == 0
    assert not (tmp_path / "absent").exists()


def _with_provider(cfg: Path, base_url: str) -> None:
    config = yaml.safe_load(cfg.read_text())
    config["providers"] = {"openai": {"base_url": base_url, "api_key": "sk-inline-secret"}}
    cfg.write_text(yaml.safe_dump(config))
    providers.reload()


def test_bundle_manifest_and_download_are_secret_free(client, cfg):
    _with_provider(cfg, "https://api.openai.com/v1")
    manifest = client.get("/admin/api/bundle/manifest", headers=ADMIN)
    assert manifest.status_code == 200
    assert manifest.json()["downloadable"] is True
    assert "providers.openai.api_key" in manifest.json()["removed"]
    assert "auth" in manifest.json()["removed"]

    resp = client.get("/admin/api/bundle", headers=ADMIN)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/gzip"
    assert resp.headers["cache-control"] == "no-store"
    assert "attachment" in resp.headers["content-disposition"]
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        members = {m.name: tar.extractfile(m).read().decode() for m in tar.getmembers()}
    assert {"manifest.json", "config.yaml", "model-info.json"} <= set(members)
    assert "sk-inline-secret" not in members["config.yaml"]
    assert "client-token" not in members["config.yaml"]
    assert client.get("/admin/api/bundle").status_code == 401


def test_bundle_download_refuses_urls_with_embedded_credentials(client, cfg):
    _with_provider(cfg, "https://user:pass@api.example.com/v1?token=abc")
    manifest = client.get("/admin/api/bundle/manifest", headers=ADMIN).json()
    assert manifest["downloadable"] is False
    assert manifest["credential_urls"] == ["providers.openai.base_url"]
    resp = client.get("/admin/api/bundle", headers=ADMIN)
    assert resp.status_code == 409
    assert "pass" not in resp.text and "abc" not in resp.text
