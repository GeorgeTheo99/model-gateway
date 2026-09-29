"""Consumer credential management (``model-gateway consumer``)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src import auth, config_io
import src.providers as providers

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for name in ("MODEL_GATEWAY_CLIENT_KEYS", "MODEL_GATEWAY_CLIENT_KEYS_FILE", "MODEL_GATEWAY_ADMIN_KEY"):
        monkeypatch.delenv(name, raising=False)
    path = Path(os.path.realpath(tmp_path)) / "gateway" / "config.yaml"
    path.parent.mkdir()
    path.write_text("auth:\n  admin_keys: [admin-token]\n  client_keys: [client-token]\nproviders: {}\n")
    path.chmod(0o600)
    monkeypatch.setattr(providers, "CONFIG_PATH", path)
    monkeypatch.setattr(config_io, "CONFIG_PATH", path)
    providers.reload()
    return path


def _entries(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text())["auth"]["consumer_credentials"]


def test_add_creates_private_key_file_and_entry(cfg):
    row = config_io.add_consumer_credential("myai", "runtime", allow_direct_models=True)
    key_file = cfg.parent / "secrets" / "consumers" / "myai-runtime.key"
    assert row["status"] == "created"
    assert row["key_file"] == str(key_file)
    assert row["key_status"] == "ok"
    assert key_file.stat().st_mode & 0o777 == 0o600
    assert key_file.parent.stat().st_mode & 0o777 == 0o700
    token = key_file.read_text().strip()
    assert len(token) == 64
    assert token not in cfg.read_text()
    assert token not in repr(row)
    assert _entries(cfg) == [{
        "id": "myai-runtime", "consumer": "myai", "key_file": str(key_file),
        "namespaces": ["myai"], "permissions": ["profiles:read", "profiles:invoke"],
        "allow_direct_models": True,
    }]
    providers.reload()
    consumers, _clients, _admins = auth.validate_credential_separation()
    assert [(t, p.credential_id) for t, p in consumers] == [(token, "myai-runtime")]


def test_deployer_role_and_explicit_namespaces(cfg):
    config_io.add_consumer_credential("ha", "deployer", namespaces=["ha", "ha-lab"])
    (entry,) = _entries(cfg)
    assert entry["permissions"] == ["profiles:read", "profiles:write"]
    assert entry["namespaces"] == ["ha", "ha-lab"]
    assert entry["allow_direct_models"] is False


def test_add_is_idempotent_and_refuses_conflicting_settings(cfg):
    config_io.add_consumer_credential("myai", "runtime")
    key_file = cfg.parent / "secrets" / "consumers" / "myai-runtime.key"
    token = key_file.read_text()
    before = cfg.read_text()
    assert config_io.add_consumer_credential("myai", "runtime")["status"] == "unchanged"
    assert cfg.read_text() == before and key_file.read_text() == token
    with pytest.raises(ValueError, match="different settings"):
        config_io.add_consumer_credential("myai", "runtime", allow_direct_models=True)


def test_add_adopts_existing_private_key_file(cfg):
    key_file = cfg.parent / "secrets" / "consumers" / "pi-runtime.key"
    key_file.parent.mkdir(parents=True)
    key_file.write_text("existing-consumer-token\n")
    key_file.chmod(0o600)
    config_io.add_consumer_credential("pi", "runtime")
    assert key_file.read_text() == "existing-consumer-token\n"


def test_add_refuses_loose_existing_key_file_and_leaves_config(cfg):
    key_file = cfg.parent / "secrets" / "consumers" / "pi-runtime.key"
    key_file.parent.mkdir(parents=True)
    key_file.write_text("existing-consumer-token\n")
    key_file.chmod(0o644)
    before = cfg.read_text()
    with pytest.raises(ValueError, match="mode-0600"):
        config_io.add_consumer_credential("pi", "runtime")
    assert cfg.read_text() == before
    assert key_file.read_text() == "existing-consumer-token\n"


def test_add_rolls_back_when_token_overlaps_another_class(cfg):
    key_file = cfg.parent / "secrets" / "consumers" / "pi-runtime.key"
    key_file.parent.mkdir(parents=True)
    key_file.write_text("client-token\n")
    key_file.chmod(0o600)
    before = cfg.read_text()
    with pytest.raises(auth.CredentialOverlapError):
        config_io.add_consumer_credential("pi", "runtime")
    assert cfg.read_text() == before
    assert key_file.read_text() == "client-token\n"


def test_add_refuses_symlinked_key_target(cfg, tmp_path):
    outside = tmp_path / "outside.key"
    outside.write_text("x\n")
    link = cfg.parent / "secrets" / "consumers" / "pi-runtime.key"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        config_io.add_consumer_credential("pi", "runtime")


@pytest.mark.parametrize("consumer,role", [("My AI", "runtime"), ("myai", "admin"), ("x" * 30, "deployer")])
def test_add_rejects_invalid_ids_and_roles(cfg, consumer, role):
    with pytest.raises(ValueError):
        config_io.add_consumer_credential(consumer, role)
    assert "consumer_credentials" not in cfg.read_text()


def test_list_reports_key_status_without_values(cfg):
    config_io.add_consumer_credential("myai", "runtime")
    config_io.add_consumer_credential("ha", "runtime")
    (cfg.parent / "secrets" / "consumers" / "ha-runtime.key").unlink()
    rows = {row["id"]: row for row in config_io.list_consumer_credentials()}
    assert rows["myai-runtime"]["key_status"] == "ok"
    assert rows["ha-runtime"]["key_status"] == "missing"
    token = (cfg.parent / "secrets" / "consumers" / "myai-runtime.key").read_text().strip()
    assert token not in repr(rows)


def test_revoke_removes_entry_and_deletes_key_file(cfg):
    config_io.add_consumer_credential("myai", "runtime")
    config_io.add_consumer_credential("myai", "deployer")
    key_file = cfg.parent / "secrets" / "consumers" / "myai-deployer.key"
    row = config_io.revoke_consumer_credential("myai-deployer")
    assert row["status"] == "revoked" and row["key_file_deleted"] is True
    assert not key_file.exists()
    assert [entry["id"] for entry in _entries(cfg)] == ["myai-runtime"]
    with pytest.raises(KeyError):
        config_io.revoke_consumer_credential("myai-deployer")


def test_revoke_refuses_to_remove_the_last_v1_credential(cfg):
    cfg.write_text("auth:\n  admin_keys: [admin-token]\nproviders: {}\n")
    config_io.add_consumer_credential("myai", "runtime")
    before = cfg.read_text()
    with pytest.raises(ValueError, match="last /v1 credential"):
        config_io.revoke_consumer_credential("myai-runtime")
    assert cfg.read_text() == before
    assert (cfg.parent / "secrets" / "consumers" / "myai-runtime.key").exists()


def test_add_repairs_missing_default_key_and_rejects_loose_one(cfg):
    config_io.add_consumer_credential("myai", "runtime")
    key_file = cfg.parent / "secrets" / "consumers" / "myai-runtime.key"
    old = key_file.read_text()
    key_file.unlink()
    row = config_io.add_consumer_credential("myai", "runtime")
    assert row["status"] == "repaired" and row["key_status"] == "ok"
    assert key_file.read_text() != old and key_file.stat().st_mode & 0o777 == 0o600
    key_file.chmod(0o644)
    with pytest.raises(ValueError, match="unusable key file"):
        config_io.add_consumer_credential("myai", "runtime")


def test_add_reports_when_it_turns_on_client_auth(cfg):
    cfg.write_text("auth:\n  admin_keys: [admin-token]\nproviders: {}\n")
    assert config_io.add_consumer_credential("myai", "runtime")["enables_client_auth"] is True
    assert config_io.add_consumer_credential("ha", "runtime")["enables_client_auth"] is False


def test_revoke_warns_but_succeeds_when_key_delete_fails(cfg, monkeypatch):
    config_io.add_consumer_credential("myai", "runtime")
    config_io.add_consumer_credential("myai", "deployer")
    key_file = cfg.parent / "secrets" / "consumers" / "myai-deployer.key"
    real_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if self == key_file:
            raise PermissionError(13, "Permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    row = config_io.revoke_consumer_credential("myai-deployer")
    assert row["key_file_deleted"] is False and "Permission denied" in row["warning"]
    assert [entry["id"] for entry in _entries(cfg)] == ["myai-runtime"]


def test_revoke_keeps_key_files_outside_the_managed_path(cfg, tmp_path):
    custom = tmp_path / "project-owned.key"
    custom.write_text("project-token\n")
    custom.chmod(0o600)
    doc = yaml.safe_load(cfg.read_text())
    doc["auth"]["consumer_credentials"] = [{
        "id": "proj-runtime", "consumer": "proj", "key_file": str(custom),
        "namespaces": ["proj"], "permissions": ["profiles:read"],
    }]
    cfg.write_text(yaml.safe_dump(doc))
    assert config_io.revoke_consumer_credential("proj-runtime")["key_file_deleted"] is False
    assert custom.exists()


def test_revoke_keeps_symlinked_or_shared_key_files(cfg, tmp_path):
    shared = tmp_path / "shared.key"
    shared.write_text("shared-token\n")
    shared.chmod(0o600)
    doc = yaml.safe_load(cfg.read_text())
    doc["auth"]["consumer_credentials"] = [{
        "id": "legacy-runtime", "consumer": "legacy", "key_file": str(shared),
        "namespaces": ["legacy"], "permissions": ["profiles:read"],
    }]
    doc["providers"] = {"demo": {"base_url": "https://x.example/v1", "api_key_file": str(shared)}}
    cfg.write_text(yaml.safe_dump(doc))
    row = config_io.revoke_consumer_credential("legacy-runtime")
    assert row["key_file_deleted"] is False
    assert shared.exists()


def test_cli_exit_codes_and_output_never_print_keys(tmp_path):
    real = Path(os.path.realpath(tmp_path))
    config = real / "config.yaml"
    config.write_text("auth:\n  client_keys: [client-token]\nproviders: {}\n")
    config.chmod(0o600)
    env = {key: value for key, value in os.environ.items() if not key.startswith("MODEL_GATEWAY_CLIENT_KEYS")}
    env["MODEL_GATEWAY_BACKUP_DIR"] = str(real / "backups")

    def run(*args):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "consumers.py"), "--config", str(config), *args],
            env=env, capture_output=True, text=True, timeout=60,
        )

    added = run("add", "myai", "--role", "runtime", "--allow-direct-models")
    assert added.returncode == 0, added.stderr
    assert added.stdout.startswith("created myai-runtime -> ")
    token = (real / "secrets" / "consumers" / "myai-runtime.key").read_text().strip()
    assert run("add", "myai", "--role", "runtime", "--allow-direct-models").returncode == 3
    listed = run("list")
    assert listed.returncode == 0 and "myai-runtime" in listed.stdout and "key=ok" in listed.stdout
    assert run("list", "--json").returncode == 0
    conflict = run("add", "myai", "--role", "runtime")
    assert conflict.returncode == 2 and "different settings" in conflict.stderr
    revoked = run("revoke", "myai-runtime")
    assert revoked.returncode == 0 and "key file deleted" in revoked.stdout
    missing = run("revoke", "myai-runtime")
    assert missing.returncode == 2 and "not found" in missing.stderr
    outputs = [added, listed, conflict, revoked, missing]
    assert all(token not in result.stdout + result.stderr for result in outputs)


def test_cli_manager_role_requires_and_lists_provider_allowlist(tmp_path):
    real = Path(os.path.realpath(tmp_path))
    config = real / "config.yaml"
    config.write_text("auth:\n  client_keys: [client-token]\nproviders: {}\n")
    config.chmod(0o600)
    env = {key: value for key, value in os.environ.items() if not key.startswith("MODEL_GATEWAY_CLIENT_KEYS")}
    env["MODEL_GATEWAY_BACKUP_DIR"] = str(real / "backups")

    def run(*args):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "consumers.py"), "--config", str(config), *args],
            env=env, capture_output=True, text=True, timeout=60,
        )

    missing = run("add", "ha", "--role", "manager")
    assert missing.returncode == 2 and "providers allowlist" in missing.stderr
    stray = run("add", "ha", "--role", "runtime", "--provider", "fireworks")
    assert stray.returncode == 2 and "only to the manager role" in stray.stderr
    added = run("add", "ha", "--role", "manager", "--provider", "fireworks")
    assert added.returncode == 0, added.stderr
    assert "providers=fireworks" in run("list").stdout
    (entry,) = yaml.safe_load(config.read_text())["auth"]["consumer_credentials"]
    assert entry["id"] == "ha-manager" and entry["providers"] == ["fireworks"]
