"""Portable gateway bundles (``model-gateway bundle export|import``)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from src import bundle, config_io
import src.providers as providers

ROOT = Path(__file__).resolve().parents[1]
CATALOG = {"llm": [{
    "name": "claude-test", "provider": "anthropic", "provider_model_id": "claude-test-1",
    "context": 200000, "max_output_tokens": 8192, "pricing": {"input": 3.0, "output": 15.0},
}]}
REGISTRY = {"format": 1, "namespaces": {"ha": [{
    "gateway_version": 1, "registered_at": "2026-09-28T00:00:00Z", "etag": "\"e\"",
    "manifest_digest": "d", "manifest": {"namespace": "ha"}, "bindings": {},
}]}}


def _machine(root: Path, monkeypatch) -> Path:
    """Point the gateway at one simulated machine rooted at ``root``."""
    home = root / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    for name in ("MODEL_GATEWAY_CLIENT_KEYS", "MODEL_GATEWAY_CLIENT_KEYS_FILE", "MODEL_GATEWAY_ADMIN_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_GATEWAY_PROFILE_REGISTRY", str(root / "registry.json"))
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(root / "backups"))
    config = root / "config.yaml"
    info = root / "model-info.json"
    for module in (providers, config_io):
        monkeypatch.setattr(module, "CONFIG_PATH", config)
        monkeypatch.setattr(module, "MODEL_INFO_PATH", info)
        monkeypatch.setattr(module, "MODEL_INFO_SOURCE_PATH", None)
    providers.reload()
    return home


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = Path(os.path.realpath(tmp_path)) / "source"
    home = _machine(root, monkeypatch)
    key_file = home / ".config" / "model-gateway" / "secrets" / "anthropic.api-key"
    key_file.parent.mkdir(parents=True)
    key_file.write_text("sk-source-secret\n")
    key_file.chmod(0o600)
    (root / "config.yaml").write_text(yaml.safe_dump({
        "auth": {"admin_keys": ["admin-secret"], "client_keys": ["client-secret"]},
        "providers": {
            "anthropic": {"base_url": "https://api.anthropic.com/v1", "protocol": "anthropic",
                          "api_key_file": str(key_file)},
            "legacy": {"base_url": "https://x.example/v1", "api_key": "inline-secret",
                       "default_headers": {"Authorization": "Bearer hdr-secret", "HTTP-Referer": "https://me"}},
        },
        "exports": {"model_aliases": str(home / "aliases.json")},
        "federation": {"node_id": "source"},
        "model_overrides": {"claude-test": {"enabled": True}},
    }))
    (root / "model-info.json").write_text(json.dumps(CATALOG))
    (root / "registry.json").write_text(json.dumps(REGISTRY))
    return root


def test_export_is_secret_free_and_home_relative(source):
    out = source / "bundle.tar.gz"
    manifest = bundle.export_bundle(out)
    assert out.stat().st_mode & 0o777 == 0o600
    assert set(manifest["files"]) == {"config.yaml", "model-info.json", "consumer-profiles-registry.json"}
    assert {"auth", "federation", "exports", "providers.legacy.api_key",
            "providers.legacy.default_headers.Authorization"} <= set(manifest["removed"])
    raw = b""
    with tarfile.open(out) as tar:
        for member in tar.getmembers():
            raw += tar.extractfile(member).read()
    for secret in (b"sk-source-secret", b"admin-secret", b"client-secret", b"inline-secret", b"hdr-secret"):
        assert secret not in raw
    _manifest, files = bundle.read_bundle(out)
    config = yaml.safe_load(files["config.yaml"])
    assert config["providers"]["anthropic"]["api_key_file"] == "~/.config/model-gateway/secrets/anthropic.api-key"
    assert config["providers"]["legacy"]["default_headers"] == {"HTTP-Referer": "https://me"}
    with pytest.raises(ValueError, match="overwrite"):
        bundle.export_bundle(out)


def test_import_into_fresh_machine_keeps_its_auth_and_reports_missing_keys(source, tmp_path, monkeypatch):
    out = source / "bundle.tar.gz"
    bundle.export_bundle(out)
    target = Path(os.path.realpath(tmp_path)) / "target"
    _machine(target, monkeypatch)
    (target / "config.yaml").write_text(yaml.safe_dump({
        "auth": {"client_keys": ["target-client"]},
        "exports": {"model_aliases": "~/Library/Application Support/model-gateway/model-aliases.json"},
        "providers": {},
    }))
    summary = bundle.import_bundle(out)
    assert summary["providers"] == ["anthropic", "legacy"]
    assert summary["models"] == 1 and summary["profile_namespaces"] == ["ha"]
    assert [row["owner"] for row in summary["missing_key_files"]] == ["providers.anthropic"]
    config = yaml.safe_load((target / "config.yaml").read_text())
    assert config["auth"] == {"client_keys": ["target-client"]}
    assert config["exports"]["model_aliases"].startswith("~/Library/")
    assert "federation" not in config
    assert json.loads((target / "model-info.json").read_text()) == CATALOG
    assert json.loads((target / "registry.json").read_text()) == REGISTRY


def test_import_refuses_existing_content_without_force_and_dry_run_writes_nothing(source):
    out = source / "bundle.tar.gz"
    bundle.export_bundle(out)
    before = (source / "config.yaml").read_text()
    with pytest.raises(ValueError, match="--force"):
        bundle.import_bundle(out)
    assert bundle.import_bundle(out, dry_run=True, force=True)["dry_run"] is True
    assert (source / "config.yaml").read_text() == before
    bundle.import_bundle(out, force=True)
    assert yaml.safe_load((source / "config.yaml").read_text())["auth"]["admin_keys"] == ["admin-secret"]


def test_import_rolls_back_when_the_catalog_does_not_load(source, tmp_path, monkeypatch):
    out = source / "bundle.tar.gz"
    bundle.export_bundle(out)
    target = Path(os.path.realpath(tmp_path)) / "target"
    _machine(target, monkeypatch)
    (target / "config.yaml").write_text("auth:\n  client_keys: [t]\n")
    broken = target / "broken.tar.gz"
    _manifest, files = bundle.read_bundle(out)
    files["model-info.json"] = json.dumps({"llm": []}).encode()
    _write_tar(broken, files)
    with pytest.raises(Exception):
        bundle.import_bundle(broken)
    assert (target / "config.yaml").read_text() == "auth:\n  client_keys: [t]\n"
    assert not (target / "model-info.json").exists()
    assert not (target / "registry.json").exists()


def _write_tar(path: Path, files: dict[str, bytes], manifest: dict | None = None) -> None:
    if manifest is None:
        manifest = {"format": 1, "files": {name: bundle._sha256(data) for name, data in files.items()}}
    members = {"manifest.json": json.dumps(manifest).encode(), **files}
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize("case", ["traversal", "tampered", "machine-section", "missing-catalog"])
def test_read_and_import_reject_unsafe_bundles(tmp_path, case):
    files = {"config.yaml": b"providers: {}\n", "model-info.json": json.dumps(CATALOG).encode()}
    manifest = None
    if case == "traversal":
        files["../evil"] = b"x"
    elif case == "tampered":
        manifest = {"format": 1, "files": {"config.yaml": "0", "model-info.json": "0"}}
    elif case == "machine-section":
        files["config.yaml"] = b"auth:\n  client_keys: [x]\n"
    else:
        files.pop("model-info.json")
    path = tmp_path / "bad.tar.gz"
    _write_tar(path, files, manifest)
    with pytest.raises(ValueError):
        bundle.import_bundle(path, dry_run=True)


def test_cli_round_trip(source, tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("MODEL_GATEWAY_CLIENT_KEYS")}
    env.update({
        "HOME": str(source / "home"),
        "MODEL_GATEWAY_PROFILE_REGISTRY": str(source / "registry.json"),
        "MODEL_GATEWAY_MODEL_INFO": str(source / "model-info.json"),
        "MODEL_GATEWAY_BACKUP_DIR": str(source / "backups"),
    })
    out = source / "cli.tar.gz"

    def run(*args):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "bundle.py"), "--config", str(source / "config.yaml"), *args],
            env=env, capture_output=True, text=True, timeout=60,
        )

    exported = run("export", "--out", str(out))
    assert exported.returncode == 0, exported.stderr
    assert "left out: auth" in exported.stdout
    assert run("import", str(out)).returncode == 2
    dry = run("import", str(out), "--force", "--dry-run")
    assert dry.returncode == 3 and json.loads(dry.stdout)["dry_run"] is True


def test_export_allowlists_sections_and_import_rejects_unknown_ones(source, tmp_path):
    doc = yaml.safe_load((source / "config.yaml").read_text())
    doc["future_secrets"] = {"token": "future-secret"}
    doc["pools"] = {"p": ["anthropic"]}
    (source / "config.yaml").write_text(yaml.safe_dump(doc))
    out = source / "bundle.tar.gz"
    manifest = bundle.export_bundle(out)
    assert "future_secrets" in manifest["removed"]
    _manifest, files = bundle.read_bundle(out)
    assert b"future-secret" not in files["config.yaml"]
    assert "pools" in yaml.safe_load(files["config.yaml"])
    files["config.yaml"] = b"surprise: {}\n"
    _write_tar(tmp_path / "unknown.tar.gz", files)
    with pytest.raises(ValueError, match="known content sections"):
        bundle.import_bundle(tmp_path / "unknown.tar.gz", dry_run=True)
    files["config.yaml"] = b"providers: [unclosed\n"
    _write_tar(tmp_path / "yaml.tar.gz", files)
    with pytest.raises(ValueError, match="not valid YAML"):
        bundle.import_bundle(tmp_path / "yaml.tar.gz", dry_run=True)


def test_import_without_force_protects_any_existing_section(source, tmp_path, monkeypatch):
    out = source / "bundle.tar.gz"
    bundle.export_bundle(out)
    target = Path(os.path.realpath(tmp_path)) / "target"
    _machine(target, monkeypatch)
    (target / "config.yaml").write_text("auth:\n  client_keys: [t]\nmodel_fallbacks: {a: b}\n")
    with pytest.raises(ValueError, match="--force"):
        bundle.import_bundle(out)


def test_dry_run_lists_where_keys_would_be_sent(source):
    out = source / "bundle.tar.gz"
    bundle.export_bundle(out)
    summary = bundle.import_bundle(out, force=True, dry_run=True)
    endpoints = {row["owner"]: row for row in summary["provider_endpoints"]}
    assert endpoints["providers.anthropic"]["url"] == "https://api.anthropic.com/v1"
    assert endpoints["providers.anthropic"]["api_key_file"] == "~/.config/model-gateway/secrets/anthropic.api-key"


def test_rollback_file_restores_the_previous_machine_state(source, tmp_path, monkeypatch):
    out = source / "bundle.tar.gz"
    bundle.export_bundle(out)
    target = Path(os.path.realpath(tmp_path)) / "target"
    _machine(target, monkeypatch)
    original = "auth:\n  client_keys: [t]\n"
    (target / "config.yaml").write_text(original)
    rollback = target / "rollback.json"
    bundle.import_bundle(out, rollback_file=rollback)
    assert rollback.stat().st_mode & 0o777 == 0o600
    assert (target / "model-info.json").exists()
    restored = bundle.restore_rollback(rollback)
    assert str(target / "config.yaml") in restored
    assert (target / "config.yaml").read_text() == original
    assert not (target / "model-info.json").exists() and not (target / "registry.json").exists()
    assert not rollback.exists()


def test_import_rolls_back_when_vision_policy_rejects_the_catalog(source, tmp_path, monkeypatch):
    out = source / "bundle.tar.gz"
    bundle.export_bundle(out)
    target = Path(os.path.realpath(tmp_path)) / "target"
    _machine(target, monkeypatch)
    (target / "config.yaml").write_text("auth:\n  client_keys: [t]\n")
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK_CLOUD", "model-not-in-bundle")
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK_MODE", "extract_then_answer")
    with pytest.raises(Exception):
        bundle.import_bundle(out)
    assert (target / "config.yaml").read_text() == "auth:\n  client_keys: [t]\n"
    assert not (target / "model-info.json").exists()


def test_import_restores_files_on_keyboard_interrupt(source, tmp_path, monkeypatch):
    out = source / "bundle.tar.gz"
    bundle.export_bundle(out)
    target = Path(os.path.realpath(tmp_path)) / "target"
    _machine(target, monkeypatch)
    (target / "config.yaml").write_text("auth:\n  client_keys: [t]\n")

    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(bundle, "_validate_loaded_gateway", interrupted)
    with pytest.raises(KeyboardInterrupt):
        bundle.import_bundle(out)
    assert (target / "config.yaml").read_text() == "auth:\n  client_keys: [t]\n"
    assert not (target / "model-info.json").exists() and not (target / "registry.json").exists()


def test_restore_rollback_rejects_unmanaged_paths_and_hard_links(source, tmp_path):
    rollback = tmp_path / "rollback.json"
    rollback.write_text(json.dumps({str(tmp_path / "elsewhere"): "x"}))
    rollback.chmod(0o600)
    with pytest.raises(ValueError, match="does not manage"):
        bundle.restore_rollback(rollback)
    assert not (tmp_path / "elsewhere").exists()
    rollback.write_text(json.dumps({str(source / "config.yaml"): "providers: {}\n"}))
    os.link(rollback, tmp_path / "second-link")
    with pytest.raises(ValueError, match="unsafe"):
        bundle.restore_rollback(rollback)
