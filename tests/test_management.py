"""Tests for writeable provider/model management (milestones 4 & 5)."""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src import admin, circuit, config_io
import src.providers as providers
from src.server import app

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None


# ── config_io: provider writes ──────────────────────────────────────────────


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Isolate config.yaml + model-info.json to temp dirs."""
    cfg = tmp_path / "config.yaml"
    monkeypatch.setattr(providers, "CONFIG_PATH", cfg)
    monkeypatch.setattr(config_io, "CONFIG_PATH", cfg)
    # Minimal config with one provider + auth.
    cfg.write_text("auth:\n  admin_keys:\n    - admin\nproviders:\n  anthropic:\n    base_url: https://api.anthropic.com/v1\n    api_key: secret-existing\n    protocol: anthropic\n")
    # Model-info: temp file, no source mirror.
    mi = tmp_path / "model-info.json"
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", mi)
    monkeypatch.setattr(config_io, "MODEL_INFO_PATH", mi)
    monkeypatch.setattr(providers, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.setattr(config_io, "MODEL_INFO_SOURCE_PATH", None)
    mi.write_text(json.dumps({"llm": [
        {"name": "claude-test", "provider": "anthropic", "provider_model_id": "claude-test-1",
         "context": 200000, "max_output_tokens": 8192, "pricing": {"input": 3.0, "output": 15.0}},
    ]}))
    providers.reload()
    return tmp_path


def test_upsert_provider_creates_new(tmp_config, monkeypatch):
    # Set log/backup dir to tmp so backups don't litter the real log dir.
    monkeypatch.setenv("MODEL_GATEWAY_LOG_DIR", str(tmp_config / "logs"))
    config_io.log_dir = tmp_config / "logs"
    result = config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-new")
    assert result["id"] == "openai"
    assert result["has_api_key"] is True
    assert result["api_key_source"] == "file"
    assert result["base_url"] == "https://api.openai.com/v1"
    # The key lands in a private key file, never inline in config.yaml.
    import yaml
    text = (tmp_config / "config.yaml").read_text()
    assert "sk-new" not in text
    block = yaml.safe_load(text)["providers"]["openai"]
    key_file = tmp_config / "secrets" / "openai.api-key"
    assert "api_key" not in block
    assert block["api_key_file"] == str(key_file.resolve())
    assert key_file.read_text() == "sk-new\n"
    assert key_file.stat().st_mode & 0o777 == 0o600
    assert key_file.parent.stat().st_mode & 0o777 == 0o700
    providers.reload()
    assert providers._effective_provider_config(providers._load_config(), "openai")["api_key"] == "sk-new"


def test_upsert_provider_key_replaces_inline_and_existing_file(tmp_config, monkeypatch):
    import yaml
    existing = tmp_config / "custom" / "anthropic.key"
    existing.parent.mkdir()
    existing.write_text("old-file-key\n")
    existing.chmod(0o600)
    cfg = tmp_config / "config.yaml"
    doc = yaml.safe_load(cfg.read_text())
    doc["providers"]["anthropic"]["api_key_file"] = str(existing)
    cfg.write_text(yaml.safe_dump(doc))

    config_io.upsert_provider("anthropic", base_url="https://api.anthropic.com/v1", api_key="rotated")

    block = yaml.safe_load(cfg.read_text())["providers"]["anthropic"]
    assert "api_key" not in block
    assert block["api_key_file"] == str(existing)
    assert existing.read_text() == "rotated\n"
    assert not (tmp_config / "secrets").exists()


def test_upsert_provider_refuses_key_file_owned_by_another_provider(tmp_config, monkeypatch):
    import yaml
    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-openai")
    cfg = tmp_config / "config.yaml"
    doc = yaml.safe_load(cfg.read_text())
    doc["providers"]["anthropic"]["api_key_file"] = doc["providers"]["openai"]["api_key_file"]
    cfg.write_text(yaml.safe_dump(doc))

    with pytest.raises(ValueError, match="already used by 'providers.openai'"):
        config_io.upsert_provider("anthropic", base_url="https://api.anthropic.com/v1", api_key="other")
    assert (tmp_config / "secrets" / "openai.api-key").read_text() == "sk-openai\n"


def test_upsert_provider_refuses_symlinked_default_key_file(tmp_config, monkeypatch):
    secrets_dir = tmp_config / "secrets"
    secrets_dir.mkdir()
    victim = tmp_config / "victim.txt"
    victim.write_text("keep\n")
    (secrets_dir / "openai.api-key").symlink_to(victim)

    with pytest.raises(ValueError, match="symlink"):
        config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-new")
    assert victim.read_text() == "keep\n"


def _set_provider_blocks(tmp_config, **blocks):
    import yaml
    cfg = tmp_config / "config.yaml"
    doc = yaml.safe_load(cfg.read_text())
    for name, block in blocks.items():
        if block is None:
            doc["providers"].pop(name, None)
        else:
            doc["providers"][name] = block
    cfg.write_text(yaml.safe_dump(doc))
    return cfg


def test_upsert_provider_preserves_relative_key_file_reference(tmp_config, monkeypatch):
    import yaml
    (tmp_config / "keys").mkdir()
    cfg = _set_provider_blocks(tmp_config, openai={"base_url": "https://api.openai.com/v1", "api_key_file": "keys/openai.key"})

    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-rel")

    assert yaml.safe_load(cfg.read_text())["providers"]["openai"]["api_key_file"] == "keys/openai.key"
    assert (tmp_config / "keys" / "openai.key").read_text() == "sk-rel\n"


def test_upsert_provider_refuses_symlinked_existing_key_file(tmp_config, monkeypatch):
    victim = tmp_config / "victim.txt"
    victim.write_text("keep\n")
    (tmp_config / "link.key").symlink_to(victim)
    _set_provider_blocks(tmp_config, openai={"base_url": "https://api.openai.com/v1", "api_key_file": "link.key"})

    with pytest.raises(ValueError, match="symlink"):
        config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-new")
    assert victim.read_text() == "keep\n"


def test_upsert_provider_refuses_key_file_that_is_a_gateway_file(tmp_config, monkeypatch):
    catalog_before = config_io.MODEL_INFO_PATH.read_text()
    _set_provider_blocks(tmp_config, openai={"base_url": "https://api.openai.com/v1", "api_key_file": "model-info.json"})

    with pytest.raises(ValueError, match="config or catalog"):
        config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-new")
    assert config_io.MODEL_INFO_PATH.read_text() == catalog_before


def test_upsert_provider_refuses_federation_peer_key_file(tmp_config, monkeypatch):
    import yaml
    peer_key = tmp_config / "peer.key"
    peer_key.write_text("peer-secret\n")
    peer_key.chmod(0o600)
    cfg = tmp_config / "config.yaml"
    doc = yaml.safe_load(cfg.read_text())
    doc["federation"] = {"node_id": "main", "peers": {"edge": {"base_url": "https://edge.example", "api_key_file": str(peer_key)}}}
    doc["providers"]["openai"] = {"base_url": "https://api.openai.com/v1", "api_key_file": str(peer_key)}
    cfg.write_text(yaml.safe_dump(doc))

    with pytest.raises(ValueError, match="federation peer edge"):
        config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-new")
    assert peer_key.read_text() == "peer-secret\n"


def test_upsert_provider_rejects_non_string_key(tmp_config, monkeypatch):
    with pytest.raises(ValueError, match="must be a string"):
        config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key=123)


def test_upsert_provider_keeps_oauth_refreshed_token_inline(tmp_config, monkeypatch):
    import yaml
    cfg = tmp_config / "config.yaml"
    doc = yaml.safe_load(cfg.read_text())
    doc["providers"]["ws"] = {"base_url": "https://ws.example/serving-endpoints", "auth_refresh": "databricks-cli"}
    cfg.write_text(yaml.safe_dump(doc))

    config_io.upsert_provider("ws", base_url="https://ws.example/serving-endpoints", api_key="eyJtoken")

    assert yaml.safe_load(cfg.read_text())["providers"]["ws"]["api_key"] == "eyJtoken"
    assert not (tmp_config / "secrets").exists()


def test_upsert_provider_empty_key_removes_file_reference(tmp_config, monkeypatch):
    import yaml
    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-openai")
    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="")
    block = yaml.safe_load((tmp_config / "config.yaml").read_text())["providers"]["openai"]
    assert "api_key" not in block and "api_key_file" not in block


def test_clearing_a_key_deletes_its_gateway_managed_file(tmp_config, monkeypatch):
    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-openai")
    key_file = tmp_config / "secrets" / "openai.api-key"
    assert key_file.exists()
    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="")
    assert not key_file.exists()


def test_clearing_a_key_keeps_an_owner_supplied_or_shared_file(tmp_config, monkeypatch):
    import yaml
    external = tmp_config / "owner.key"
    external.write_text("sk-owner\n")
    external.chmod(0o600)
    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-openai")
    managed = tmp_config / "secrets" / "openai.api-key"
    doc = yaml.safe_load((tmp_config / "config.yaml").read_text())
    doc["providers"]["other"] = {"base_url": "https://other.example/v1", "api_key_file": str(external)}
    doc["workspaces"] = {"ws": {"base_url": "https://ws.example/v1", "api_key_file": str(managed)}}
    (tmp_config / "config.yaml").write_text(yaml.safe_dump(doc))

    config_io.upsert_provider("other", base_url="https://other.example/v1", api_key="")
    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="")
    assert external.read_text() == "sk-owner\n"
    assert managed.read_text() == "sk-openai\n"


def test_deleting_a_provider_deletes_its_gateway_managed_key_file(tmp_config, monkeypatch):
    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-openai")
    config_io.delete_provider("openai")
    assert not (tmp_config / "secrets" / "openai.api-key").exists()


@pytest.mark.parametrize("operation", ["clear", "delete"])
def test_admin_key_removal_deletes_the_file_and_a_rejected_reload_restores_it(tmp_config, monkeypatch, operation):
    if TestClient is None:
        pytest.skip("fastapi test client unavailable")
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    key_file = tmp_config / "secrets" / "openai.api-key"
    admin_headers = {"Authorization": "Bearer admin"}

    def remove(client):
        if operation == "clear":
            return client.post("/admin/api/providers/openai", headers=admin_headers,
                               json={"base_url": "https://api.openai.com/v1", "api_key": ""})
        return client.delete("/admin/api/providers/openai", headers=admin_headers)

    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-original")
    providers.reload()
    with TestClient(app) as client:
        assert remove(client).status_code == 200
    assert not key_file.exists()

    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-original")
    providers.reload()
    seen = []

    def reject(snapshot):
        seen.append(key_file.exists())
        return "synthetic rejection"

    monkeypatch.setattr(admin, "_reload_registry_transactionally", reject)
    with TestClient(app) as client:
        response = remove(client)
    assert response.status_code == 400 and "rolled back" in response.text
    assert seen == [False], "the file must be gone before the reload is validated"
    assert key_file.read_text() == "sk-original\n"
    assert key_file.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("case", ["custom-name", "outside", "oauth", "consumer", "federation"])
def test_removal_keeps_files_that_are_not_the_providers_own_or_are_still_used(tmp_config, monkeypatch, case):
    import yaml
    secrets = tmp_config / "secrets"
    secrets.mkdir(exist_ok=True)
    kept = {"custom-name": secrets / "custom.key", "outside": tmp_config / "owner.key"}.get(
        case, secrets / "openai.api-key")
    kept.write_text("sk-kept\n")
    kept.chmod(0o600)
    doc = yaml.safe_load((tmp_config / "config.yaml").read_text())
    block = {"base_url": "https://api.openai.com/v1", "api_key_file": str(kept)}
    if case == "oauth":
        block["auth_refresh"] = "databricks-cli"
    doc["providers"]["openai"] = block
    if case == "consumer":
        doc["auth"]["consumer_credentials"] = [{"id": "c", "consumer": "c", "key_file": str(kept),
                                                "namespaces": ["c"], "permissions": ["profiles:read"]}]
    if case == "federation":
        doc["federation"] = {"peers": {"peer": {"base_url": "http://peer.example/v1", "api_key_file": str(kept)}}}
    (tmp_config / "config.yaml").write_text(yaml.safe_dump(doc))
    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="")
    assert kept.read_text() == "sk-kept\n"


def test_deleting_a_synonym_provider_still_refuses_dependent_models(tmp_config, monkeypatch):
    import yaml
    doc = yaml.safe_load((tmp_config / "config.yaml").read_text())
    doc["providers"]["claude"] = doc["providers"].pop("anthropic")
    (tmp_config / "config.yaml").write_text(yaml.safe_dump(doc))
    providers.reload()
    with pytest.raises(ValueError, match="depend on it"):
        config_io.delete_provider("claude")


def test_admin_rejected_provider_update_restores_key_file(tmp_config, monkeypatch):
    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-original")
    providers.reload()
    key_file = tmp_config / "secrets" / "openai.api-key"
    monkeypatch.setattr(admin, "_reload_registry_transactionally", lambda snapshot: "synthetic rejection")

    result, error = admin._apply_registry_mutation(
        lambda: config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-rejected"),
        extra_paths=lambda: [config_io.api_key_file_target("openai")],
    )

    assert result is None and "rolled back" in error
    assert key_file.read_text() == "sk-original\n"
    assert key_file.stat().st_mode & 0o777 == 0o600


def test_migrate_inline_api_keys(tmp_config, monkeypatch):
    import yaml
    cfg = tmp_config / "config.yaml"
    doc = yaml.safe_load(cfg.read_text())
    shadowed = tmp_config / "shadowed.key"
    shadowed.write_text("stale\n")
    shadowed.chmod(0o600)
    doc["providers"]["openai"] = {"base_url": "https://api.openai.com/v1", "api_key": "sk-inline", "api_key_file": str(shadowed)}
    doc["providers"]["ws"] = {"base_url": "https://ws.example", "api_key": "eyJ", "auth_refresh": "databricks-cli"}
    cfg.write_text(yaml.safe_dump(doc))
    before = cfg.read_text()

    planned = config_io.migrate_inline_api_keys(dry_run=True)
    assert {row["provider"] for row in planned} == {"anthropic", "openai"}
    assert cfg.read_text() == before and not (tmp_config / "secrets").exists()

    moved = config_io.migrate_inline_api_keys()
    assert moved == planned
    text = cfg.read_text()
    assert "secret-existing" not in text and "sk-inline" not in text
    after = yaml.safe_load(text)["providers"]
    assert (tmp_config / "secrets" / "anthropic.api-key").read_text() == "secret-existing\n"
    assert shadowed.read_text() == "sk-inline\n"
    assert after["openai"]["api_key_file"] == str(shadowed)
    assert after["ws"]["api_key"] == "eyJ"
    assert config_io.migrate_inline_api_keys() == []
    providers.reload()
    config = providers._load_config()
    assert providers._effective_provider_config(config, "anthropic")["api_key"] == "secret-existing"
    assert providers._effective_provider_config(config, "openai")["api_key"] == "sk-inline"


def test_migrate_refuses_providers_that_would_share_a_key_file(tmp_config, monkeypatch):
    cfg = _set_provider_blocks(
        tmp_config,
        Foo={"base_url": "https://foo.example/v1", "api_key": "one"},
        foo={"base_url": "https://foo.example/v1", "api_key": "two"},
    )
    before = cfg.read_text()

    with pytest.raises(ValueError, match="would share API key file"):
        config_io.migrate_inline_api_keys()
    assert cfg.read_text() == before
    assert not (tmp_config / "secrets").exists()


def test_migrate_covers_workspaces_section(tmp_config, monkeypatch):
    import yaml
    cfg = tmp_config / "config.yaml"
    doc = yaml.safe_load(cfg.read_text())
    doc["workspaces"] = {"ws-pat": {"base_url": "https://ws.example/serving-endpoints", "api_key": "dapi-inline"}}
    cfg.write_text(yaml.safe_dump(doc))

    moved = {row["provider"] for row in config_io.migrate_inline_api_keys()}

    assert moved == {"anthropic", "ws-pat"}
    assert "dapi-inline" not in cfg.read_text()
    assert (tmp_config / "secrets" / "ws-pat.api-key").read_text() == "dapi-inline\n"


def test_migrate_restores_files_when_config_write_fails(tmp_config, monkeypatch):
    shadowed = tmp_config / "shadowed.key"
    shadowed.write_text("stale\n")
    shadowed.chmod(0o600)
    cfg = _set_provider_blocks(
        tmp_config,
        openai={"base_url": "https://api.openai.com/v1", "api_key": "sk-inline", "api_key_file": str(shadowed)},
    )
    before = cfg.read_text()
    real_write = config_io._atomic_write

    def fail_config_write(path, text):
        if path == config_io.CONFIG_PATH and "sk-inline" not in text:
            raise OSError("synthetic config write failure")
        return real_write(path, text)

    monkeypatch.setattr(config_io, "_atomic_write", fail_config_write)
    with pytest.raises(OSError, match="synthetic"):
        config_io.migrate_inline_api_keys()

    assert cfg.read_text() == before
    assert shadowed.read_text() == "stale\n"
    assert shadowed.stat().st_mode & 0o777 == 0o600
    assert not (tmp_config / "secrets" / "anthropic.api-key").exists()


def test_migrate_script_exit_codes(tmp_path):
    import subprocess
    import sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    real = _Path(os.path.realpath(tmp_path))
    cfg = real / "config.yaml"
    cfg.write_text("providers:\n  demo:\n    base_url: https://x.example/v1\n    api_key: demo-secret\n")
    cfg.chmod(0o600)
    env = {
        **os.environ,
        "MODEL_GATEWAY_SECRET_DIR": str(real / "secrets"),
        "MODEL_GATEWAY_BACKUP_DIR": str(real / "backups"),
    }

    def run(*args):
        return subprocess.run(
            [sys.executable, str(root / "scripts" / "migrate_api_keys.py"), "--config", str(cfg), *args],
            env=env, capture_output=True, text=True, timeout=60,
        )

    dry = run("--dry-run")
    assert dry.returncode == 0 and "would move demo" in dry.stdout
    assert "demo-secret" in cfg.read_text()
    applied = run()
    assert applied.returncode == 0 and "moved demo" in applied.stdout
    assert "demo-secret" not in applied.stdout + applied.stderr + cfg.read_text()
    assert run().returncode == 3
    assert run("--dry-run").returncode == 0


def test_provider_status_reports_key_source_and_storage_warnings(tmp_config, monkeypatch):
    import yaml
    cfg = tmp_config / "config.yaml"
    doc = yaml.safe_load(cfg.read_text())
    doc["providers"]["openai"] = {"base_url": "https://api.openai.com/v1", "api_key": "sk", "api_key_file": "x.key"}
    cfg.write_text(yaml.safe_dump(doc))
    providers.reload()

    by_id = {p["id"]: p for p in providers.provider_status()}
    assert by_id["anthropic"]["api_key_source"] == "inline"
    assert by_id["anthropic"]["warnings"] == ["inline_api_key"]
    assert by_id["anthropic"]["ready"] is True
    assert by_id["openai"]["warnings"] == ["inline_api_key_shadows_file"]
    validation = providers.config_validation()
    assert validation["ok"] is True
    assert {w["provider"] for w in validation["warnings"]} == {"anthropic", "openai"}

    config_io.migrate_inline_api_keys()
    providers.reload()
    by_id = {p["id"]: p for p in providers.provider_status()}
    assert by_id["anthropic"]["api_key_source"] == "file"
    assert by_id["anthropic"]["warnings"] == []
    monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_ANTHROPIC_API_KEY", "env-key")
    assert {p["id"]: p for p in providers.provider_status()}["anthropic"]["api_key_source"] == "env"
    assert "secret-existing" not in json.dumps(providers.config_validation())


def test_config_backups_can_live_outside_log_tree(tmp_config, monkeypatch):
    logs = tmp_config / "logs"
    backups = tmp_config / "private-backups"
    monkeypatch.setattr(config_io, "log_dir", logs)
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(backups))

    config_io.upsert_provider(
        "openai",
        base_url="https://api.openai.com/v1",
        api_key="sk-new",
    )

    created = list(backups.glob("config.yaml.bak.*"))
    assert len(created) == 1
    assert created[0].stat().st_mode & 0o777 == 0o600
    assert backups.stat().st_mode & 0o777 == 0o700
    assert not logs.exists()


def test_config_backup_retention_removes_old_credentials(tmp_config, monkeypatch):
    backups = tmp_config / "private-backups"
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(backups))
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_RETENTION", "3")

    config_io.upsert_provider(
        "anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key="replacement-secret",
    )
    for index in range(5):
        config_io.upsert_provider(
            "openai",
            base_url=f"https://api{index}.openai.com/v1",
            api_key=f"new-secret-{index}",
        )

    created = list(backups.glob("config.yaml.bak.*"))
    assert len(created) == 3
    assert all("secret-existing" not in path.read_text() for path in created)


def test_config_backup_rejects_symlinked_parent(tmp_config, monkeypatch):
    real_parent = tmp_config / "real-backups"
    real_parent.mkdir()
    linked_parent = tmp_config / "linked-backups"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(linked_parent / "config"))

    with pytest.raises(OSError):
        config_io.upsert_provider(
            "openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-new",
        )

    assert not list(real_parent.rglob("*.bak.*"))


def test_legacy_backups_migrate_out_of_logs_and_become_private(tmp_config, monkeypatch):
    logs = tmp_config / "logs"
    legacy = logs / "config-backups"
    nested = legacy / "retirement-archive"
    nested.mkdir(parents=True)
    config_backup = legacy / "config.yaml.bak.100"
    catalog_backup = legacy / "model-info.json.bak.101"
    archived_catalog = nested / "private-catalog.json"
    config_backup.write_text("api_key: legacy-secret\n")
    catalog_backup.write_text('{"private": true}\n')
    archived_catalog.write_text('{"archived": true}\n')
    for path in (config_backup, catalog_backup, archived_catalog):
        path.chmod(0o644)
    nested.chmod(0o755)
    legacy.chmod(0o755)
    target = tmp_config / "private-backups"
    monkeypatch.setattr(config_io, "log_dir", logs)
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(target))

    assert config_io.migrate_legacy_backups() == 3

    assert not legacy.exists()
    imports = list(target.glob("legacy-config-backups-*"))
    assert len(imports) == 1
    imported = imports[0]
    assert (imported / config_backup.name).read_text() == "api_key: legacy-secret\n"
    assert (imported / catalog_backup.name).read_text() == '{"private": true}\n'
    assert (imported / nested.name / archived_catalog.name).read_text() == '{"archived": true}\n'
    assert target.stat().st_mode & 0o777 == 0o700
    assert imported.stat().st_mode & 0o777 == 0o700
    assert (imported / config_backup.name).stat().st_mode & 0o777 == 0o600
    assert (imported / nested.name).stat().st_mode & 0o777 == 0o700
    assert (imported / nested.name / archived_catalog.name).stat().st_mode & 0o777 == 0o600


def test_legacy_backup_migration_rejects_symlink_source_without_moving_target(tmp_config, monkeypatch):
    logs = tmp_config / "logs"
    victim = tmp_config / "victim"
    victim.mkdir()
    secret = victim / "config.yaml.bak.1"
    secret.write_text("must-stay\n")
    linked = logs / "config-backups"
    logs.mkdir()
    linked.symlink_to(victim, target_is_directory=True)
    target = tmp_config / "private-backups"
    monkeypatch.setattr(config_io, "log_dir", logs)
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(target))
    monkeypatch.setenv("MODEL_GATEWAY_LEGACY_BACKUP_DIRS", str(linked))

    with pytest.raises(OSError):
        config_io.migrate_legacy_backups()

    assert victim.is_dir()
    assert secret.read_text() == "must-stay\n"
    assert linked.is_symlink()


def test_former_package_backup_root_is_migrated_and_pruned(tmp_config, monkeypatch):
    old_package_root = tmp_config / "HomeServer" / "ci" / "logs" / "config-backups"
    old_package_root.mkdir(parents=True)
    for index in range(25):
        (old_package_root / f"config.yaml.bak.{index}").write_text(f"secret-{index}\n")
    target = tmp_config / "private-backups"
    monkeypatch.setattr(config_io, "log_dir", tmp_config / "new-logs")
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(target))
    monkeypatch.setenv("MODEL_GATEWAY_LEGACY_BACKUP_DIRS", str(old_package_root))
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_RETENTION", "20")

    assert config_io.migrate_legacy_backups() == 20

    assert not old_package_root.exists()
    imported = next(target.glob("legacy-config-backups-*"))
    assert len(list(imported.glob("config.yaml.bak.*"))) == 20
    assert imported.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in imported.glob("*.bak.*"))


def test_legacy_backup_migration_applies_retention_before_import(tmp_config, monkeypatch):
    logs = tmp_config / "logs"
    legacy = logs / "config-backups"
    legacy.mkdir(parents=True)
    for index in range(25):
        (legacy / f"config.yaml.bak.{index}").write_text(f"secret-{index}\n")
    target = tmp_config / "private-backups"
    monkeypatch.setattr(config_io, "log_dir", logs)
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(target))
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_RETENTION", "20")

    assert config_io.migrate_legacy_backups() == 20

    imported = next(target.glob("legacy-config-backups-*"))
    assert len(list(imported.glob("config.yaml.bak.*"))) == 20


def test_multiple_legacy_imports_share_one_global_retention_cap(tmp_config, monkeypatch):
    roots = [tmp_config / "legacy-one", tmp_config / "legacy-two"]
    for root_index, root in enumerate(roots):
        root.mkdir()
        for index in range(15):
            path = root / f"config.yaml.bak.{root_index * 100 + index}"
            path.write_text(f"secret-{root_index}-{index}\n")
            os.utime(path, ns=(1_000_000_000 + root_index * 100 + index,) * 2)
    target = tmp_config / "private-backups"
    monkeypatch.setattr(config_io, "log_dir", tmp_config / "logs")
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(target))
    monkeypatch.setenv("MODEL_GATEWAY_LEGACY_BACKUP_DIRS", os.pathsep.join(map(str, roots)))
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_RETENTION", "20")

    assert config_io.migrate_legacy_backups() == 30

    retained = list(target.glob("legacy-config-backups-*/config.yaml.bak.*"))
    assert len(retained) == 20


def test_backup_runtime_rejects_log_directory_overlap(tmp_config, monkeypatch):
    logs = tmp_config / "logs"
    legacy = logs / "config-backups"
    legacy.mkdir(parents=True)
    (legacy / "config.yaml.bak.1").write_text("secret\n")
    monkeypatch.setattr(config_io, "log_dir", logs)
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(legacy))

    with pytest.raises(RuntimeError, match="must not overlap"):
        config_io.migrate_legacy_backups()

    assert (legacy / "config.yaml.bak.1").read_text() == "secret\n"


def test_legacy_backup_migration_rejects_cross_filesystem_before_move(tmp_config, monkeypatch):
    logs = tmp_config / "logs"
    legacy = logs / "config-backups"
    legacy.mkdir(parents=True)
    (legacy / "config.yaml.bak.1").write_text("secret\n")
    target = tmp_config / "private-backups"
    monkeypatch.setattr(config_io, "log_dir", logs)
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(target))

    real_open = config_io._open_private_backup_directory
    real_fstat = os.fstat
    opened: dict[str, int] = {}

    def tracked_open(path):
        fd, absolute = real_open(path)
        opened[str(absolute)] = fd
        return fd, absolute

    def split_device_fstat(fd):
        metadata = real_fstat(fd)
        if fd == opened.get(str(target)):
            return SimpleNamespace(st_dev=metadata.st_dev + 1, st_ino=metadata.st_ino)
        return metadata

    monkeypatch.setattr(config_io, "_open_private_backup_directory", tracked_open)
    monkeypatch.setattr(config_io.os, "fstat", split_device_fstat)
    with pytest.raises(RuntimeError, match="same filesystem"):
        config_io.migrate_legacy_backups()

    assert legacy.is_dir()
    assert (legacy / "config.yaml.bak.1").read_text() == "secret\n"


def test_config_backup_rejects_fifo_without_blocking(tmp_config, monkeypatch):
    backups = tmp_config / "private-backups"
    backups.mkdir()
    os.mkfifo(backups / "config.yaml.bak.1", 0o600)
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(backups))

    with pytest.raises(RuntimeError, match="unsafe model-gateway backup entry"):
        config_io.upsert_provider(
            "openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-new",
        )


def test_config_backup_rejects_fifo_source_without_blocking(tmp_config, monkeypatch):
    source = tmp_config / "config-source.fifo"
    os.mkfifo(source, 0o600)
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(tmp_config / "private-backups"))

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="unsafe model-gateway backup source"):
        config_io._backup(source)
    assert time.monotonic() - started < 1


def test_provider_write_lock_covers_full_read_modify_write(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    original_load = config_io.load_config_full
    active = 0
    max_active = 0
    guard = threading.Lock()

    def slow_load():
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        try:
            return original_load()
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(config_io, "load_config_full", slow_load)
    threads = [
        threading.Thread(target=config_io.upsert_provider, args=(name,), kwargs={
            "base_url": f"https://{name}.example.com/v1", "api_key": "secret",
        })
        for name in ("one", "two")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert all(not thread.is_alive() for thread in threads)
    assert max_active == 1
    import yaml
    providers_config = yaml.safe_load((tmp_config / "config.yaml").read_text())["providers"]
    assert {"one", "two"}.issubset(providers_config)


def test_upsert_provider_preserves_existing_key(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    # Update anthropic base_url, api_key=None -> existing key preserved.
    config_io.upsert_provider("anthropic", base_url="https://api.anthropic.com/v2", api_key=None)
    import yaml
    cfg = yaml.safe_load((tmp_config / "config.yaml").read_text())
    assert cfg["providers"]["anthropic"]["api_key"] == "secret-existing"
    assert cfg["providers"]["anthropic"]["base_url"] == "https://api.anthropic.com/v2"


def test_upsert_provider_empty_key_removes(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    config_io.upsert_provider("anthropic", base_url="https://api.anthropic.com/v1", api_key="")
    import yaml
    cfg = yaml.safe_load((tmp_config / "config.yaml").read_text())
    assert "api_key" not in cfg["providers"]["anthropic"]


def test_delete_provider_refuses_if_models_depend(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    with pytest.raises(ValueError, match="depend on it"):
        config_io.delete_provider("anthropic")


def test_delete_provider_after_disabling_models(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    config_io.set_model_enabled("claude-test", False)
    providers.reload()
    result = config_io.delete_provider("anthropic")
    assert result["deleted"] is True
    import yaml
    cfg = yaml.safe_load((tmp_config / "config.yaml").read_text())
    assert "anthropic" not in cfg["providers"]


def test_delete_unknown_provider(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    with pytest.raises(KeyError):
        config_io.delete_provider("nonexistent")


# ── config_io: model writes ─────────────────────────────────────────────────


def test_upsert_model_creates(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    result = config_io.upsert_model(
        "gpt-test", provider="openai", provider_model_id="gpt-test-1",
        context=128000, max_output_tokens=16384, pricing={"input": 2.5, "output": 15.0},
    )
    assert result["name"] == "gpt-test"
    assert result["entry"]["provider_model_id"] == "gpt-test-1"
    # Verify it landed in model-info.json.
    doc = json.loads((tmp_config / "model-info.json").read_text())
    names = [e["name"] for e in doc["llm"]]
    assert "gpt-test" in names


def test_upsert_model_updates_existing(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    config_io.upsert_model("claude-test", provider="anthropic",
                           provider_model_id="claude-test-1", context=300000)
    doc = json.loads((tmp_config / "model-info.json").read_text())
    entry = next(e for e in doc["llm"] if e["name"] == "claude-test")
    assert entry["context"] == 300000
    # pricing preserved (not provided in update).
    assert entry["pricing"] == {"input": 3.0, "output": 15.0}


def test_upsert_local_model_can_be_explicitly_unmetered(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    result = config_io.upsert_model(
        "local-unmetered", provider="omlx", provider_model_id="local-upstream",
        pricing_status="unmetered", pricing=None,
    )
    assert result["entry"]["pricing_status"] == "unmetered"
    assert "pricing" not in result["entry"]
    providers.reload()
    status = next(row for row in providers.model_status() if row["name"] == "local-unmetered")
    assert status["pricing_status"] == "unmetered"
    assert status["pricing"] is None


def test_upsert_model_rejects_invalid_pricing_policy(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    with pytest.raises(ValueError, match="only valid for local"):
        config_io.upsert_model(
            "cloud-free", provider="openai", provider_model_id="cloud-free",
            pricing_status="unmetered", pricing=None,
        )
    with pytest.raises(ValueError, match="requires: output"):
        config_io.upsert_model(
            "bad-price", provider="openai", provider_model_id="bad-price",
            pricing_status="metered", pricing={"input": 1.0},
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        config_io.upsert_model(
            "infinite-price", provider="openai", provider_model_id="infinite-price",
            pricing_status="metered", pricing={"input": float("inf"), "output": 1.0},
        )
    config_io.upsert_model(
        "local-to-cloud", provider="omlx", provider_model_id="local-upstream",
        pricing_status="unmetered", pricing=None,
    )
    with pytest.raises(ValueError, match="only valid for local"):
        config_io.upsert_model(
            "local-to-cloud", provider="openai", provider_model_id="cloud-upstream",
        )


def test_set_model_enabled_writes_config_yaml(tmp_config, monkeypatch):
    """enabled state lives in config.yaml model_overrides, not model-info.json."""
    config_io.log_dir = tmp_config / "logs"
    r = config_io.set_model_enabled("claude-test", False)
    assert r["enabled"] is False
    assert str(tmp_config / "config.yaml") in r["written_to"]
    # Override landed in config.yaml.
    import yaml
    cfg = yaml.safe_load((tmp_config / "config.yaml").read_text())
    assert cfg["model_overrides"]["claude-test"]["enabled"] is False
    # model-info.json is NOT touched by the toggle.
    doc = json.loads((tmp_config / "model-info.json").read_text())
    entry = next(e for e in doc["llm"] if e["name"] == "claude-test")
    assert "enabled" not in entry


def test_atomic_write_preserves_symlink(tmp_path, monkeypatch):
    """config.yaml is a symlink to a shared file; writes must update the target,
    not replace the link with a real file (which would split config state)."""
    config_io.log_dir = tmp_path / "logs"
    # Set up: deploy dir with a symlink to a shared real file.
    shared = tmp_path / "shared.yaml"
    shared.write_text("providers:\n  anthropic:\n    base_url: https://x\n    api_key: k\n")
    link = tmp_path / "deploy" / "config.yaml"
    link.parent.mkdir(parents=True)
    link.symlink_to(shared)
    monkeypatch.setattr(providers, "CONFIG_PATH", link)
    monkeypatch.setattr(config_io, "CONFIG_PATH", link)
    providers.reload()
    # Write via config_io (upsert_provider uses _atomic_write).
    config_io.upsert_provider("anthropic", base_url="https://y", api_key="k2")
    # The link must still be a symlink.
    assert link.is_symlink(), "symlink was replaced by a real file"
    # And the shared target must hold the new content.
    import yaml
    d = yaml.safe_load(shared.read_text())
    assert d["providers"]["anthropic"]["base_url"] == "https://y"


def test_delete_model(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    config_io.delete_model("claude-test")
    doc = json.loads((tmp_config / "model-info.json").read_text())
    assert all(e["name"] != "claude-test" for e in doc["llm"])


def test_upsert_model_validates_required(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    with pytest.raises(ValueError, match="provider_model_id is required"):
        config_io.upsert_model("x", provider="openai", provider_model_id="")


def test_upsert_model_validates_and_persists_thinking_levels(tmp_config):
    result = config_io.upsert_model(
        "max-only", provider="anthropic", provider_model_id="max-upstream",
        thinking="always", thinking_levels=["max"],
    )
    assert result["entry"]["thinking_levels"] == ["max"]
    providers.reload()
    assert providers.resolve("max-only").thinking_levels == ("max",)

    with pytest.raises(ValueError, match="supports off only"):
        config_io.upsert_model(
            "invalid", provider="anthropic", provider_model_id="invalid-upstream",
            thinking="always", thinking_levels=["off", "high"],
        )


def test_upsert_model_thinking_null_preserves_narrow_levels(tmp_config):
    config_io.upsert_model(
        "max-only", provider="anthropic", provider_model_id="max-upstream",
        thinking="always", thinking_levels=["max"],
    )

    result = config_io.upsert_model(
        "max-only", provider="anthropic", provider_model_id="max-upstream",
        thinking=None,
    )

    assert result["entry"]["thinking"] == "always"
    assert result["entry"]["thinking_levels"] == ["max"]
    persisted = next(
        row for row in json.loads((tmp_config / "model-info.json").read_text())["llm"]
        if row["name"] == "max-only"
    )
    assert persisted["thinking_levels"] == ["max"]


# ── enabled field routing ───────────────────────────────────────────────────


def test_resolve_skips_disabled_model(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    assert providers.resolve("claude-test") is not None  # enabled by default
    config_io.set_model_enabled("claude-test", False)
    providers.reload()
    assert providers.resolve("claude-test") is None  # disabled -> not routable


def test_resolve_local_omlx_model_uses_builtin_proxy_defaults(tmp_config, monkeypatch):
    doc = json.loads((tmp_config / "model-info.json").read_text())
    doc["llm"].append({
        "name": "local-test",
        "alias": "lt",
        "omlx_id": "local-upstream",
        "context": 65536,
        "max_output_tokens": 4096,
        "thinking": "always",
        "thinking_format": "glm-chat-template",
    })
    (tmp_config / "model-info.json").write_text(json.dumps(doc))
    providers.reload()

    for model_id in ("local-test", "lt", "local-upstream"):
        info = providers.resolve(model_id)
        assert info is not None
        assert info.provider == "omlx"
        assert info.base_url == "http://localhost:9110/v1"
        assert info.api_key == "omlx"
        assert info.provider_model_id == "local-upstream"
        assert info.protocol == "openai"

    omlx_status = next(p for p in providers.provider_status() if p["id"] == "omlx")
    assert omlx_status["ready"] is True
    assert omlx_status["issues"] == []


def test_effective_inventory_has_one_row_with_all_routable_ids(tmp_config):
    config_io.upsert_model(
        "inventory-model", provider="anthropic", provider_model_id="inventory-upstream",
        alias="inventory-alias", vision=True,
    )
    providers.reload()
    rows = [m for m in providers.effective_model_inventory() if m["name"] == "inventory-model"]
    assert len(rows) == 1
    assert rows[0]["vision"] is True
    assert set(rows[0]["routable_ids"]) >= {"inventory-model", "inventory-alias", "inventory-upstream"}
    assert providers.resolve("inventory-alias").vision is True
    discovered = [m["id"] for m in providers.list_models() if m["name"] == "inventory-model"]
    assert set(discovered) >= {"inventory-model", "inventory-alias", "inventory-upstream"}


def test_provider_status_counts_unique_models_not_identifiers(tmp_config, monkeypatch):
    """A model with name + alias + provider_model_id + omlx_id must count as 1."""
    doc = json.loads((tmp_config / "model-info.json").read_text())
    doc["llm"].append({
        "name": "local-multi", "alias": "lm", "omlx_id": "lm-up",
        "provider_model_id": "lm-pmid", "alternate_ids": ["legacy/lm"],
        "context": 4096, "max_output_tokens": 1024,
    })
    (tmp_config / "model-info.json").write_text(json.dumps(doc))
    providers.reload()
    counts = {p["id"]: p["enabled_models"] for p in providers.provider_status()}
    # omlx has exactly one unique local model (local-multi); claude-test is anthropic.
    assert counts.get("omlx") == 1
    assert counts.get("anthropic") == 1
    # routable_ids exposes canonical identifiers plus legacy alternate IDs.
    assert set(providers.routable_ids("local-multi")) == {"local-multi", "lm", "lm-up", "lm-pmid", "legacy/lm"}
    assert providers.resolve("legacy/lm") is not None


def test_upsert_model_allows_local_omlx_id(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    result = config_io.upsert_model(
        "local-created", provider="omlx", omlx_id="local-created-upstream",
        context=32768, max_output_tokens=2048,
    )
    assert result["entry"]["omlx_id"] == "local-created-upstream"
    providers.reload()
    info = providers.resolve("local-created")
    assert info is not None
    assert info.provider_model_id == "local-created-upstream"


def test_unconfigured_provider_models_are_hidden_from_v1_models(tmp_config, monkeypatch):
    doc = json.loads((tmp_config / "model-info.json").read_text())
    doc["llm"].append({
        "name": "gpt-unconfigured",
        "provider": "openai",
        "provider_model_id": "gpt-test",
        "context": 128000,
        "max_output_tokens": 4096,
    })
    (tmp_config / "model-info.json").write_text(json.dumps(doc))
    providers.reload()

    assert providers.resolve("gpt-unconfigured") is None
    assert providers.model_availability("gpt-unconfigured")["reason"] == "provider_not_configured"

    with TestClient(app) as c:
        public_rows = c.get("/v1/models").json()["data"]
        ids = {m["id"] for m in public_rows}
        assert "claude-test" in ids
        assert "gpt-unconfigured" not in ids
        assert all("pricing" not in m and "pricing_status" not in m for m in public_rows)

        admin = {"Authorization": "Bearer admin"}
        row = next(
            m for m in c.get("/admin/api/models", headers=admin).json()["models"]
            if m["name"] == "gpt-unconfigured"
        )
        assert row["available"] is False
        assert row["availability_reason"] == "provider_not_configured"


def test_requesting_unconfigured_model_returns_clear_error(tmp_config, monkeypatch):
    doc = json.loads((tmp_config / "model-info.json").read_text())
    doc["llm"].append({"name": "gpt-unconfigured", "provider": "openai", "provider_model_id": "gpt-test"})
    (tmp_config / "model-info.json").write_text(json.dumps(doc))
    providers.reload()

    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json={"model": "gpt-unconfigured", "messages": []})
    assert resp.status_code == 404
    message = resp.json()["error"]["message"]
    assert "provider_not_configured" in message
    assert "openai" in message


def test_databricks_provider_can_be_configured_from_env(tmp_config, monkeypatch):
    # Isolate from ambient Databricks env (dev shells often export these).
    for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_SERVING_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    doc = json.loads((tmp_config / "model-info.json").read_text())
    doc["llm"].append({
        "name": "dbx-chat",
        "provider": "databricks",
        "provider_model_id": "my-serving-endpoint",
    })
    (tmp_config / "model-info.json").write_text(json.dumps(doc))
    with open(tmp_config / "config.yaml", "a") as f:
        f.write("  databricks:\n    enabled: false\n")
    providers.reload()

    assert providers.model_availability("dbx-chat")["reason"] == "provider_disabled"

    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-placeholder")
    providers.reload()

    info = providers.resolve("dbx-chat")
    assert info is not None
    assert info.provider == "databricks"
    assert info.base_url == "https://workspace.example.databricks.com/serving-endpoints"
    assert info.api_key == "dapi-placeholder"
    assert info.protocol == "openai"


# ── admin API endpoints ─────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin")
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_readonly(tmp_config, monkeypatch):
    """Admin auth on, but writes disabled (the default read-only dashboard)."""
    config_io.log_dir = tmp_config / "logs"
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin")
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_WRITES", raising=False)
    with TestClient(app) as c:
        yield c


def test_admin_writes_disabled_blocks_management(client_readonly):
    """With MODEL_GATEWAY_ADMIN_WRITES unset, mutating endpoints return 403."""
    h = {"Authorization": "Bearer admin"}
    # Provider writes
    assert client_readonly.post("/admin/api/providers/openai", headers=h, json={"base_url": "u"}).status_code == 403
    assert client_readonly.delete("/admin/api/providers/anthropic", headers=h).status_code == 403
    assert client_readonly.post("/admin/api/providers/anthropic/validate", headers=h).status_code == 403
    # Model writes
    assert client_readonly.post("/admin/api/models/new", headers=h, json={"provider": "openai", "provider_model_id": "x"}).status_code == 403
    assert client_readonly.delete("/admin/api/models/claude-test", headers=h).status_code == 403
    assert client_readonly.post("/admin/api/models/claude-test/disable", headers=h).status_code == 403
    assert client_readonly.post("/admin/api/models/claude-test/enable", headers=h).status_code == 403
    # Reload is also gated
    assert client_readonly.post("/admin/api/reload", headers=h).status_code == 403
    # Read-only endpoints still work
    assert client_readonly.get("/admin/api/status", headers=h).status_code == 200
    assert client_readonly.get("/admin/api/providers", headers=h).status_code == 200
    assert client_readonly.get("/admin/api/models", headers=h).status_code == 200
    assert client_readonly.get("/admin/api/models/claude-test/stats", headers=h).status_code == 200


def test_admin_status_reports_writes_enabled_true(client):
    h = {"Authorization": "Bearer admin"}
    assert client.get("/admin/api/status", headers=h).json()["writes_enabled"] is True


def test_admin_status_reports_writes_enabled_false(client_readonly):
    h = {"Authorization": "Bearer admin"}
    assert client_readonly.get("/admin/api/status", headers=h).json()["writes_enabled"] is False


def test_admin_workspace_pools_reports_runtime_routing_without_secrets(
    client_readonly, tmp_config, monkeypatch,
):
    config = {
        "auth": {"admin_keys": ["admin"]},
        "providers": {
            "ws-primary": {
                "base_url": "https://primary.cloud.databricks.com",
                "api_key": "secret-primary-token",
                "protocol": "openai",
                "endpoint_style": "invocations",
                "auth_refresh": "databricks-cli",
                "auth_profile": "primary-profile",
            },
            "ws-standby": {
                "base_url": "https://standby.cloud.databricks.com",
                "api_key": "secret-standby-token",
                "protocol": "openai",
                "endpoint_style": "invocations",
                "auth_refresh": "databricks-cli",
                "auth_profile": "standby-profile",
            },
        },
        "pools": {"default-pool": ["ws-primary", "ws-standby"]},
    }
    import yaml
    (tmp_config / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (tmp_config / "model-info.json").write_text(json.dumps({"llm": [{
        "name": "pooled-model",
        "provider": "ws-primary",
        "pool": "default-pool",
        "provider_model_id": "upstream-model",
    }]}))
    monkeypatch.setattr(circuit, "get_status", lambda: {
        "ws-primary": {
            "is_open": True,
            "consecutive_failures": 3,
            "last_failure_status": 503,
            "last_failure_message": "upstream unavailable",
            "seconds_since_failure": 2.0,
            "probe_in_progress": False,
        },
    })
    providers.reload()

    assert client_readonly.get("/admin/api/workspace-pools").status_code == 401
    response = client_readonly.get(
        "/admin/api/workspace-pools", headers={"Authorization": "Bearer admin"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == {
        "pools": 1, "workspaces": 2, "standalone_providers": 0,
        "healthy": 0, "degraded": 1, "down": 0,
    }
    pool = body["pools"][0]
    assert pool["active_member"] == "ws-standby"
    assert pool["models"] == ["pooled-model"]
    assert pool["members"][0]["issues"] == ["circuit_open"]
    assert pool["members"][0]["role"] == "primary"
    assert pool["members"][0]["active"] is False
    assert pool["members"][1]["role"] == "secondary"
    assert pool["members"][1]["active"] is True
    assert pool["members"][1]["auth_profile"] == "standby-profile"
    assert pool["members"][1]["credential_type"] == "oauth"
    assert "secret-primary-token" not in response.text
    assert "secret-standby-token" not in response.text


def test_admin_ui_contains_workspace_pool_view():
    html = TestClient(app).get("/admin").text
    assert 'data-tab="connections"' in html
    assert "/admin/api/workspace-pools" in html
    assert "model-gateway workspace repair" in html
    assert "Routing groups (workspace pools)" in html
    assert 'id="workspaceList"' in html
    assert "Copy repair-all command" in html
    assert 'id="standaloneCards"' not in html


def test_admin_workspace_pools_report_standalone_providers(
    client_readonly, tmp_config, monkeypatch,
):
    config = {
        "auth": {"admin_keys": ["admin"]},
        "providers": {
            "ws-primary": {
                "base_url": "https://primary.cloud.databricks.com",
                "api_key": "secret-primary-token",
            },
            "google": {
                "base_url": "https://generativelanguage.googleapis.com",
                "api_key": "secret-google-token",
            },
        },
        "pools": {"default-pool": ["ws-primary"]},
    }
    import yaml
    (tmp_config / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (tmp_config / "model-info.json").write_text(json.dumps({"llm": [
        {
            "name": "pooled-model",
            "provider": "ws-primary",
            "pool": "default-pool",
            "provider_model_id": "upstream-model",
        },
        {
            "name": "direct-model",
            "provider": "google",
            "provider_model_id": "google-model",
        },
    ]}))
    monkeypatch.setattr(circuit, "get_status", lambda: {})
    providers.reload()

    response = client_readonly.get(
        "/admin/api/workspace-pools", headers={"Authorization": "Bearer admin"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["standalone_providers"] == 1
    assert [s["id"] for s in body["standalone_providers"]] == ["google"]
    google = body["standalone_providers"][0]
    assert google["models"] == ["direct-model"]
    assert google["ready"] is True
    assert "secret-google-token" not in response.text
    assert "recent" in google

    providers_response = client_readonly.get(
        "/admin/api/providers", headers={"Authorization": "Bearer admin"},
    )
    rows = {p["id"]: p for p in providers_response.json()["providers"]}
    assert rows["ws-primary"]["pool_memberships"] == [
        {"pool": "default-pool", "position": 1},
    ]
    assert rows["google"]["pool_memberships"] == []


def test_admin_reload_rejects_invalid_vision_fallback_and_restores_registry(
    client, tmp_config, monkeypatch,
):
    model_info_path = tmp_config / "model-info.json"
    valid_catalog = {"llm": [{
        "name": "vision-fallback",
        "provider": "omlx",
        "omlx_id": "vision-upstream",
        "vision": True,
        "context": 4096,
        "max_output_tokens": 512,
    }]}
    model_info_path.write_text(json.dumps(valid_catalog))
    providers.reload()
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "vision-fallback")
    assert providers.resolve("vision-fallback").vision is True

    invalid_catalog = json.loads(json.dumps(valid_catalog))
    invalid_catalog["llm"][0]["vision"] = False
    model_info_path.write_text(json.dumps(invalid_catalog))

    response = client.post(
        "/admin/api/reload",
        headers={"Authorization": "Bearer admin"},
    )

    assert response.status_code == 400
    assert "not vision-capable" in response.text
    restored = providers.resolve("vision-fallback")
    assert restored is not None
    assert restored.vision is True


def test_admin_reload_malformed_catalog_restores_live_registry(
    client, tmp_config,
):
    assert providers.resolve("claude-test") is not None
    (tmp_config / "model-info.json").write_text("{ malformed")

    response = client.post(
        "/admin/api/reload",
        headers={"Authorization": "Bearer admin"},
    )

    assert response.status_code == 400
    assert "reload rejected" in response.text
    assert providers.resolve("claude-test") is not None


def test_admin_mutations_rollback_when_they_invalidate_vision_fallback(
    client, tmp_config, monkeypatch,
):
    import yaml

    model_info_path = tmp_config / "model-info.json"
    valid_catalog = {"llm": [{
        "name": "vision-fallback",
        "provider": "omlx",
        "omlx_id": "vision-upstream",
        "vision": True,
        "context": 4096,
        "max_output_tokens": 512,
    }]}
    model_info_path.write_text(json.dumps(valid_catalog))
    config_path = tmp_config / "config.yaml"
    model_info_path.chmod(0o600)
    config_path.chmod(0o600)
    providers.reload()
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "vision-fallback")
    assert providers.resolve("vision-fallback").vision is True
    headers = {"Authorization": "Bearer admin"}

    update_response = client.post(
        "/admin/api/models/vision-fallback",
        headers=headers,
        json={
            "provider": "omlx",
            "omlx_id": "vision-upstream",
            "vision": False,
        },
    )

    assert update_response.status_code == 400
    assert "changes rolled back" in update_response.text
    restored_catalog = json.loads(model_info_path.read_text())
    assert restored_catalog["llm"][0]["vision"] is True
    assert model_info_path.stat().st_mode & 0o777 == 0o600
    assert providers.resolve("vision-fallback").vision is True

    disable_response = client.post(
        "/admin/api/models/vision-fallback/disable",
        headers=headers,
    )

    assert disable_response.status_code == 400
    assert "changes rolled back" in disable_response.text
    config = yaml.safe_load(config_path.read_text())
    assert "vision-fallback" not in (config.get("model_overrides") or {})
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert providers.resolve("vision-fallback").vision is True


def test_registry_mutations_are_serialized_across_validation(tmp_config, monkeypatch):
    monkeypatch.delenv("GATEWAY_VISION_FALLBACK", raising=False)
    a_written = threading.Event()
    release_a = threading.Event()
    b_started = threading.Event()
    b_entered = threading.Event()
    errors = []

    def mutate_a():
        result = config_io.upsert_provider(
            "provider-a", base_url="https://a.example.test/v1", api_key="a",
        )
        a_written.set()
        assert release_a.wait(timeout=2)
        return result

    def mutate_b():
        b_entered.set()
        return config_io.upsert_provider(
            "provider-b", base_url="https://b.example.test/v1", api_key="b",
        )

    def run(mutate, started=None):
        if started is not None:
            started.set()
        try:
            result, reload_error = admin._apply_registry_mutation(mutate)
            assert reload_error is None
            assert result is not None
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread_a = threading.Thread(target=run, args=(mutate_a,))
    thread_b = threading.Thread(target=run, args=(mutate_b, b_started))
    thread_a.start()
    assert a_written.wait(timeout=2)
    thread_b.start()
    assert b_started.wait(timeout=2)
    assert not b_entered.wait(timeout=0.05)
    release_a.set()
    thread_a.join(timeout=2)
    thread_b.join(timeout=2)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert errors == []
    config = config_io.load_config_full()
    assert {"provider-a", "provider-b"}.issubset(config["providers"])


def test_admin_writes_require_admin_auth(client_readonly):
    """Writes gate runs after auth: no key -> 401, not 403."""
    assert client_readonly.post("/admin/api/reload").status_code == 401
    assert client_readonly.post("/admin/api/models/x", json={"provider": "p", "provider_model_id": "y"}).status_code == 401


def test_admin_upsert_provider_endpoint(client):
    resp = client.post("/admin/api/providers/openai",
                       headers={"Authorization": "Bearer admin"},
                       json={"base_url": "https://api.openai.com/v1", "api_key": "sk-x"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "openai"
    assert body["has_api_key"] is True
    assert body["reloaded"] is True


def test_admin_provider_endpoints_require_admin_key(client):
    assert client.post("/admin/api/providers/x", json={"base_url": "u"}).status_code == 401
    assert client.delete("/admin/api/providers/x").status_code == 401


def test_admin_upsert_model_endpoint(client):
    resp = client.post("/admin/api/models/new-model",
                       headers={"Authorization": "Bearer admin"},
                       json={"provider": "openai", "provider_model_id": "new-1",
                             "context": 128000, "max_output_tokens": 4096})
    assert resp.status_code == 200
    assert resp.json()["name"] == "new-model"
    # Catalog now contains the model (resolve also needs provider config).
    assert "new-model" in providers._load_models()


def test_create_only_status_capability_is_authenticated(client_readonly):
    assert client_readonly.get("/admin/api/status").status_code == 401
    response = client_readonly.get("/admin/api/status", headers={"Authorization": "Bearer admin"})
    assert response.status_code == 200
    assert response.json()["capabilities"]["create_only_model_registration"] is True
    assert response.json()["writes_enabled"] is False


def test_create_only_put_requires_precondition_and_preserves_existing_model(client):
    headers = {"Authorization": "Bearer admin"}
    payload = {"provider": "anthropic", "provider_model_id": "put-upstream"}
    assert client.put("/admin/api/models/put-model", headers=headers, json=payload).status_code == 428
    for invalid in ([], "not-an-object", 1):
        response = client.put("/admin/api/models/put-model", headers={**headers, "If-None-Match": "*"}, json=invalid)
        assert response.status_code == 400
    response = client.put("/admin/api/models/put-model", headers={**headers, "If-None-Match": "*", "If-Match": '"ignored"'}, json=payload)
    assert response.status_code == 400
    response = client.put("/admin/api/models/put-model", headers={**headers, "If-None-Match": "*"}, json=payload)
    assert response.status_code == 200
    before = config_io.MODEL_INFO_PATH.read_bytes()
    response = client.put("/admin/api/models/put-model", headers={**headers, "If-None-Match": "*"},
                          json={**payload, "provider_model_id": "replacement"})
    assert response.status_code == 412
    assert config_io.MODEL_INFO_PATH.read_bytes() == before


def test_create_only_model_success_then_legacy_upsert(client):
    headers = {"Authorization": "Bearer admin", "If-None-Match": " \t* "}
    payload = {"provider": "anthropic", "provider_model_id": "new-upstream", "enabled": False}
    response = client.post("/admin/api/models/%20new-model%20", headers=headers, json=payload)
    assert response.status_code == 200
    assert response.json()["name"] == "new-model"
    assert response.json()["reloaded"] is True
    assert response.json()["enabled"] is False
    assert config_io.load_config_full()["model_overrides"]["new-model"] == {"enabled": False}
    assert providers._load_models()["new-model"]["provider_model_id"] == "new-upstream"

    # Omitting the precondition still updates the same normalized canonical name.
    payload.update(provider_model_id="replacement-upstream", enabled=True)
    response = client.post(
        "/admin/api/models/%20new-model%20",
        headers={"Authorization": "Bearer admin"}, json=payload,
    )
    assert response.status_code == 200
    assert providers.resolve("new-model").provider_model_id == "replacement-upstream"
    assert len([e for e in config_io.load_model_info()["llm"] if e["name"] == "new-model"]) == 1


def test_admin_model_writes_refresh_the_alias_export(client, tmp_config, monkeypatch):
    # The exporter runs as a subprocess and reads the catalog from the environment.
    monkeypatch.setenv("MODEL_GATEWAY_MODEL_INFO", str(tmp_config / "model-info.json"))
    aliases = tmp_config / "aliases.json"
    cfg = tmp_config / "config.yaml"
    cfg.write_text(cfg.read_text() + f"exports:\n  model_aliases: {aliases}\n")
    headers = {"Authorization": "Bearer admin"}
    for name, alias in (("keep-model", "keep"), ("new-model", "newm")):
        payload = {"provider": "anthropic", "provider_model_id": f"{name}-upstream", "alias": alias,
                   "context": 1000, "max_output_tokens": 100}
        response = client.post(f"/admin/api/models/{name}", headers=headers, json=payload)
        assert response.status_code == 200
        assert response.json()["catalogs"] == "regenerated"
    assert "newm" in aliases.read_text()

    response = client.delete("/admin/api/models/new-model", headers=headers)
    assert response.status_code == 200
    assert response.json()["catalogs"] == "regenerated"
    assert "newm" not in aliases.read_text() and "keep" in aliases.read_text()


def test_admin_writes_report_skipped_and_failed_exports_without_failing(client, tmp_config, monkeypatch):
    mi = tmp_config / "model-info.json"
    monkeypatch.setenv("MODEL_GATEWAY_MODEL_INFO", str(mi))
    cfg = tmp_config / "config.yaml"
    cfg.write_text(cfg.read_text() + f"exports:\n  model_aliases: {tmp_config}/aliases.json\n")
    headers = {"Authorization": "Bearer admin"}

    # No aliased model: an unmarked catalog refuses to export, but the save stands.
    response = client.post("/admin/api/providers/openai", headers=headers,
                           json={"base_url": "https://api.openai.com/v1", "api_key": "sk-test"})
    assert response.status_code == 200
    assert response.json()["catalogs"].startswith("failed: ")
    assert "openai" in config_io.load_config_full()["providers"]

    doc = json.loads(mi.read_text())
    mi.write_text(json.dumps({**doc, "allow_empty": True}))
    response = client.delete("/admin/api/providers/openai", headers=headers)
    assert response.status_code == 200
    assert response.json()["catalogs"] == "skipped (no exportable models)"
    assert not (tmp_config / "aliases.json").exists()


def test_timed_out_catalog_export_is_killed(monkeypatch):
    import asyncio
    import sys

    from src import admin

    started = []
    real_exec = asyncio.create_subprocess_exec

    async def slow_exec(*_args, **kwargs):
        proc = await real_exec(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
        started.append(proc)
        return proc

    async def expire(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(admin.asyncio, "create_subprocess_exec", slow_exec)
    monkeypatch.setattr(admin.asyncio, "wait_for", expire)
    assert asyncio.run(admin._regenerate_catalogs()).startswith("failed")
    assert started and started[0].returncode is not None


@pytest.mark.parametrize("target,fields,overlay", [
    ("claude-test", {}, False),
    ("%20claude-test%20", {}, False),
    ("existing-alias", {}, False),
    ("claude-test-1", {}, False),
    ("alternate-id", {}, False),
    ("new-model", {"alias": "existing-alias"}, False),
    ("new-model", {"alias": "claude-test"}, False),
    ("new-model", {"provider_model_id": "claude-test-1"}, False),
    ("new-model", {"omlx_id": "alternate-id"}, False),
    ("overlay-model", {}, True),
    ("overlay-alias", {}, True),
])
def test_create_only_preserves_existing_identifiers_without_writes_or_reload(
    client, tmp_config, monkeypatch, target, fields, overlay,
):
    # Prime the registry, then change disk without a reload to simulate another
    # process's completed write. The precondition must not trust cached state.
    cached_models = providers._load_models()
    doc = config_io.load_model_info()
    doc["llm"][0].update(alias="existing-alias", alternate_ids=["alternate-id"])
    config_io.MODEL_INFO_PATH.write_text(json.dumps(doc))
    with config_io.CONFIG_PATH.open("a") as handle:
        handle.write("model_overrides:\n  claude-test:\n    enabled: false\n")
    if overlay:
        with config_io.CONFIG_PATH.open("a") as handle:
            handle.write("models:\n  - name: overlay-model\n    alias: overlay-alias\n"
                         "    provider: anthropic\n    provider_model_id: overlay-upstream\n")
    mirror = tmp_config / "mirror.json"
    mirror.write_text(json.dumps(doc))
    monkeypatch.setattr(config_io, "MODEL_INFO_SOURCE_PATH", mirror)
    paths = [config_io.CONFIG_PATH, config_io.MODEL_INFO_PATH, mirror]
    before = [(p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ino) for p in paths]

    def unexpected(*args, **kwargs):
        pytest.fail("Failed preconditions must not write, restore, or reload")

    monkeypatch.setattr(config_io, "_atomic_write", unexpected)
    monkeypatch.setattr(config_io, "_backup", unexpected)
    monkeypatch.setattr(admin, "_reload_registry_transactionally", unexpected)
    monkeypatch.setattr(admin, "restore_provider_registry", unexpected)
    payload = {"provider": "anthropic", "provider_model_id": "new-upstream", "enabled": False, **fields}
    response = client.post(
        f"/admin/api/models/{target}",
        headers={"Authorization": "Bearer admin", "If-None-Match": "*"}, json=payload,
    )
    assert response.status_code == 412
    assert [(p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_ino) for p in paths] == before
    assert providers._load_models() is cached_models


@pytest.mark.parametrize("values", [
    [""], ['"etag"'], ['W/"etag"'], ['"*"'], ["*, *"], ["*", "*"], ["*", '"etag"'],
])
def test_create_only_rejects_unsupported_preconditions(client, monkeypatch, values):
    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported preconditions must not start a mutation")

    monkeypatch.setattr(admin, "_apply_registry_mutation", unexpected)
    response = client.post(
        "/admin/api/models/new-model",
        headers=[("Authorization", "Bearer admin"), *[("If-None-Match", v) for v in values]],
        json={"provider": "anthropic", "provider_model_id": "new-upstream"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("precondition", ["*", '"unsupported"'])
def test_create_only_keeps_auth_and_write_gates(client_readonly, monkeypatch, precondition):
    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS", "consumer-key")
    payload = {"provider": "anthropic", "provider_model_id": "new-upstream"}
    for key in (None, "wrong", "consumer-key", "admin"):
        headers = {"If-None-Match": precondition}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        response = client_readonly.post("/admin/api/models/new-model", headers=headers, json=payload)
        assert response.status_code == (403 if key == "admin" else 401)
    assert not any(e["name"] == "new-model" for e in config_io.load_model_info()["llm"])


@pytest.mark.parametrize("method", ["POST", "PUT"])
@pytest.mark.parametrize("shared_alias", [False, True])
def test_create_only_competing_api_requests_have_one_winner(client, monkeypatch, shared_alias, method):
    # Separate TestClients give concurrent requests independent event loops.
    # Rendezvous immediately before the real lock so an outside-lock existence
    # check would let both requests through and fail this test deterministically.
    barrier = threading.Barrier(2)
    real_lock = admin.config_write_lock

    @contextmanager
    def competing_lock(path):
        barrier.wait(timeout=5)
        with real_lock(path):
            yield

    monkeypatch.setattr(admin, "config_write_lock", competing_lock)

    def create(index):
        competing_client = TestClient(app)
        name = f"competing-model-{index}" if shared_alias else "competing-model"
        return competing_client.request(
            method, f"/admin/api/models/{name}",
            headers={"Authorization": "Bearer admin", "If-None-Match": "*"},
            json={"provider": "anthropic", "provider_model_id": f"upstream-{index}",
                  "alias": "shared-alias" if shared_alias else f"alias-{index}", "enabled": bool(index)},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(create, index) for index in range(2)]
        responses = [future.result(timeout=10) for future in futures]
    assert sorted(r.status_code for r in responses) == [200, 412]
    winner = next(index for index, response in enumerate(responses) if response.status_code == 200)
    entries = [e for e in config_io.load_model_info()["llm"] if e["name"].startswith("competing-model")]
    assert len(entries) == 1
    name = f"competing-model-{winner}" if shared_alias else "competing-model"
    assert entries[0]["name"] == name
    assert entries[0]["provider_model_id"] == f"upstream-{winner}"
    assert entries[0]["alias"] == ("shared-alias" if shared_alias else f"alias-{winner}")
    assert config_io.load_config_full()["model_overrides"] == {name: {"enabled": bool(winner)}}
    assert providers._load_models()[name]["provider_model_id"] == f"upstream-{winner}"


def test_admin_upsert_model_applies_enabled_checkbox(client):
    h = {"Authorization": "Bearer admin"}
    payload = {
        "provider": "anthropic",
        "provider_model_id": "claude-test-1",
        "enabled": False,
    }
    response = client.post("/admin/api/models/claude-test", headers=h, json=payload)
    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert providers.resolve("claude-test") is None

    payload["enabled"] = True
    response = client.post("/admin/api/models/claude-test", headers=h, json=payload)
    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert providers.resolve("claude-test") is not None


def test_admin_disable_enable_model_endpoint(client):
    h = {"Authorization": "Bearer admin"}
    assert client.post("/admin/api/models/claude-test/disable", headers=h).status_code == 200
    assert providers.resolve("claude-test") is None
    assert client.post("/admin/api/models/claude-test/enable", headers=h).status_code == 200
    assert providers.resolve("claude-test") is not None


def test_admin_delete_model_endpoint(client):
    h = {"Authorization": "Bearer admin"}
    created = client.post(
        "/admin/api/models/replacement",
        headers=h,
        json={"provider": "anthropic", "provider_model_id": "replacement-1"},
    )
    assert created.status_code == 200
    assert client.delete("/admin/api/models/claude-test", headers=h).status_code == 200
    assert providers.resolve("claude-test") is None
    assert providers.resolve("replacement") is not None


def test_admin_delete_provider_refuses_with_dependents(client):
    h = {"Authorization": "Bearer admin"}
    resp = client.delete("/admin/api/providers/anthropic", headers=h)
    assert resp.status_code == 409
    assert "depend on it" in resp.json()["error"]["message"]


def test_admin_upsert_provider_validates_required(client):
    h = {"Authorization": "Bearer admin"}
    resp = client.post("/admin/api/providers/x", headers=h, json={"base_url": ""})
    assert resp.status_code == 400


def test_admin_model_stats_endpoint(client, monkeypatch):
    """Per-model stats returns config + usage + recent, filtered by routable ids."""
    from src import ledger
    from src.usage import Usage, CostEstimate
    db = client  # reuse tmp_config-backed app; point ledger at temp db
    import tempfile, os
    monkeypatch.setenv("MODEL_GATEWAY_LEDGER_PATH", str(tempfile.mkdtemp() + "/ledger.db"))
    ledger.init()
    # Record a request for claude-test under its name and an alias.
    for mid in ("claude-test", "claude-test-1"):
        ledger.record(endpoint="/v1/messages", method="POST", model=mid,
                      provider="anthropic", provider_model_id="claude-test-1",
                      status=200, latency_ms=120, is_stream=False,
                      usage=Usage(input_tokens=100, output_tokens=50, cached_read_tokens=0,
                                  cache_write_tokens=0, reasoning_tokens=0, reported=True),
                      cost=CostEstimate(0.012, True, []))
    h = {"Authorization": "Bearer admin"}
    resp = client.get("/admin/api/models/claude-test/stats?window=24h", headers=h)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"]["name"] == "claude-test"
    assert "claude-test" in body["routable_ids"]
    # Both rows matched via the routable id set.
    assert body["usage"]["requests"] == 2
    assert body["usage"]["input_tokens"] == 200
    assert body["usage"]["cost_usd"] == 0.024
    assert len(body["recent"]) == 2
    # Unknown model returns empty usage/recent, null model — not an error.
    resp2 = client.get("/admin/api/models/does-not-exist/stats", headers=h)
    assert resp2.status_code == 200
    assert resp2.json()["model"] is None
    assert resp2.json()["usage"] == {}
    assert resp2.json()["recent"] == []
    # Requires admin auth.
    assert client.get("/admin/api/models/claude-test/stats").status_code == 401
    # Reset the providers cache so later modules reload the real catalog.
    providers.reload()


def test_config_writes_clamp_secret_file_to_0600(tmp_config, monkeypatch):
    """A loose pre-existing config.yaml mode is tightened on every write."""
    monkeypatch.setenv("MODEL_GATEWAY_LOG_DIR", str(tmp_config / "logs"))
    config_io.log_dir = tmp_config / "logs"
    cfg = config_io.CONFIG_PATH
    cfg.chmod(0o644)

    config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-new")

    assert (cfg.stat().st_mode & 0o777) == 0o600


def test_model_info_writes_do_not_clamp(tmp_config, monkeypatch):
    """model-info.json is secret-free; its existing mode is preserved."""
    monkeypatch.setenv("MODEL_GATEWAY_LOG_DIR", str(tmp_config / "logs"))
    config_io.log_dir = tmp_config / "logs"
    mi = config_io.MODEL_INFO_PATH
    mi.chmod(0o644)

    config_io.upsert_model(
        "clamp-check", provider="anthropic", provider_model_id="clamp-check-1",
        context=1000, max_output_tokens=100, pricing={"input": 1.0, "output": 2.0},
    )

    assert (mi.stat().st_mode & 0o777) == 0o644


def test_admin_provider_stats_endpoint(client, monkeypatch):
    """Per-provider stats returns config + usage + recent, filtered by provider."""
    from src import ledger
    from src.usage import Usage, CostEstimate
    import tempfile
    monkeypatch.setenv("MODEL_GATEWAY_LEDGER_PATH", str(tempfile.mkdtemp() + "/ledger.db"))
    ledger.init()
    ledger.record(endpoint="/v1/messages", method="POST", model="claude-test",
                  provider="anthropic", provider_model_id="claude-test-1",
                  status=200, latency_ms=90, is_stream=False,
                  usage=Usage(input_tokens=10, output_tokens=5, cached_read_tokens=0,
                              cache_write_tokens=0, reasoning_tokens=0, reported=True),
                  cost=CostEstimate(0.001, True, []))
    ledger.record(endpoint="/v1/messages", method="POST", model="other-model",
                  provider="other", provider_model_id="other-1",
                  status=200, latency_ms=50, is_stream=False,
                  usage=Usage(input_tokens=1, output_tokens=1, cached_read_tokens=0,
                              cache_write_tokens=0, reasoning_tokens=0, reported=True),
                  cost=CostEstimate(0.002, True, []))
    h = {"Authorization": "Bearer admin"}
    resp = client.get("/admin/api/providers/anthropic/stats?window=24h", headers=h)
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"]["id"] == "anthropic"
    assert body["usage"]["requests"] == 1  # only the anthropic row
    assert len(body["recent"]) == 1
    assert body["recent"][0]["provider"] == "anthropic"
    # Unknown provider: null config, empty usage/recent.
    resp2 = client.get("/admin/api/providers/nope/stats", headers=h)
    assert resp2.status_code == 200
    assert resp2.json()["provider"] is None
    assert resp2.json()["recent"] == []
    # Requires admin auth.
    assert client.get("/admin/api/providers/anthropic/stats").status_code == 401
    providers.reload()


def test_admin_request_detail_endpoint(client, monkeypatch):
    """Single ledger row lookup by id; 404 on unknown id; requires auth."""
    from src import ledger
    from src.usage import Usage, CostEstimate
    import tempfile
    monkeypatch.setenv("MODEL_GATEWAY_LEDGER_PATH", str(tempfile.mkdtemp() + "/ledger.db"))
    ledger.init()
    rid = ledger.record(endpoint="/v1/chat/completions", method="POST", model="claude-test",
                        provider="anthropic", provider_model_id="claude-test-1",
                        status=500, latency_ms=42, is_stream=True,
                        usage=Usage(input_tokens=7, output_tokens=0, cached_read_tokens=0,
                                    cache_write_tokens=0, reasoning_tokens=0, reported=False),
                        cost=CostEstimate(None, False, []), error="upstream exploded")
    h = {"Authorization": "Bearer admin"}
    resp = client.get(f"/admin/api/requests/{rid}", headers=h)
    assert resp.status_code == 200
    row = resp.json()["request"]
    assert row["id"] == rid
    assert row["error"] == "upstream exploded"
    assert row["status"] == 500
    assert row["is_stream"] == 1
    assert "ts_iso" in row
    assert client.get("/admin/api/requests/nope", headers=h).status_code == 404
    assert client.get(f"/admin/api/requests/{rid}").status_code == 401


def test_admin_discover_provider_models(client, monkeypatch):
    """Discovery lists upstream ids and marks already-registered ones."""
    import src.admin as admin_module

    def fake_discover(base_url, api_key=None):
        return {"status": "verified", "http_status": 200,
                "model_ids": ["claude-test-1", "brand-new-model"]}
    import src.onboarding_generation as og
    monkeypatch.setattr(og, "discover_models", fake_discover)
    h = {"Authorization": "Bearer admin"}
    resp = client.post("/admin/api/providers/anthropic/discover", headers=h)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "verified"
    by_id = {m["id"]: m["registered"] for m in body["models"]}
    # claude-test-1 is the tmp_config catalog entry's provider_model_id.
    assert by_id == {"claude-test-1": True, "brand-new-model": False}
    # Unknown provider 404s; unauthenticated 401s.
    assert client.post("/admin/api/providers/nope/discover", headers=h).status_code == 404
    assert client.post("/admin/api/providers/anthropic/discover").status_code == 401


def test_admin_preview_model_valid(client):
    """Preview validates and resolves the route without writing anything."""
    h = {"Authorization": "Bearer admin"}
    body = {"provider": "anthropic", "provider_model_id": "claude-new-1",
            "context": 200000, "max_output_tokens": 8192,
            "pricing": {"input": 3.0, "output": 15.0}}
    resp = client.post("/admin/api/models/claude-new/preview", headers=h, json=body)
    assert resp.status_code == 200
    p = resp.json()
    assert p["ok"] is True
    assert p["routable"] is True
    assert p["issues"] == []
    assert p["clashes"] == []
    assert p["route"] == [{"provider": "anthropic", "usable": True, "reason": None}]
    assert p["exists"] is False
    assert "claude-new" in p["routable_ids"]
    # Nothing was written.
    assert providers.resolve("claude-new") is None


def test_admin_preview_model_reports_clash_and_issues(client):
    h = {"Authorization": "Bearer admin"}
    # Alias clash with the existing claude-test entry's name.
    body = {"provider": "anthropic", "provider_model_id": "x-1", "alias": "claude-test",
            "pricing": {"input": 1.0, "output": 2.0}}
    resp = client.post("/admin/api/models/x-model/preview", headers=h, json=body)
    p = resp.json()
    assert p["ok"] is False
    assert p["clashes"] == [{"id": "claude-test", "model": "claude-test"}]
    # Missing provider_model_id for a cloud provider is an issue.
    resp2 = client.post("/admin/api/models/y-model/preview", headers=h,
                        json={"provider": "anthropic", "pricing": {"input": 1.0, "output": 2.0}})
    p2 = resp2.json()
    assert p2["ok"] is False
    assert any("provider_model_id" in i for i in p2["issues"])
    # Unusable provider route is flagged.
    resp3 = client.post("/admin/api/models/z-model/preview", headers=h,
                        json={"provider": "unconfigured-prov", "provider_model_id": "z-1",
                              "pricing": {"input": 1.0, "output": 2.0}})
    p3 = resp3.json()
    assert p3["ok"] is True  # entry itself is valid
    assert p3["routable"] is False
    assert p3["route"][0]["usable"] is False


def test_admin_preview_and_discover_require_writes(client_readonly):
    h = {"Authorization": "Bearer admin"}
    assert client_readonly.post("/admin/api/models/m/preview", headers=h, json={}).status_code == 403
    assert client_readonly.post("/admin/api/providers/anthropic/discover", headers=h).status_code == 403
