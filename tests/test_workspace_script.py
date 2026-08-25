from __future__ import annotations

import argparse
import importlib.util
import stat
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "workspace.py"


def load_workspace_module():
    spec = importlib.util.spec_from_file_location("model_gateway_workspace", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backup_is_owner_only_even_for_legacy_public_source(tmp_path, monkeypatch):
    workspace = load_workspace_module()
    source = tmp_path / "config.yaml"
    source.write_text("secret: sentinel\n")
    source.chmod(0o644)
    monkeypatch.setattr(workspace.time, "strftime", lambda _fmt: "20260825-170000")

    workspace._backup(source)

    backup = tmp_path / "config.yaml.bak-20260825-170000"
    assert backup.read_text() == source.read_text()
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600


def test_backup_refuses_existing_or_symlink_destination(tmp_path, monkeypatch):
    workspace = load_workspace_module()
    source = tmp_path / "config.yaml"
    source.write_text("secret: sentinel\n")
    destination = tmp_path / "config.yaml.bak-20260825-170001"
    protected = tmp_path / "protected"
    protected.write_text("unchanged")
    destination.symlink_to(protected)
    monkeypatch.setattr(workspace.time, "strftime", lambda _fmt: "20260825-170001")

    with pytest.raises(FileExistsError):
        workspace._backup(source)

    assert protected.read_text() == "unchanged"
    assert destination.is_symlink()


def test_atomic_writer_creates_owner_only_config_without_temp_remnants(tmp_path):
    workspace = load_workspace_module()
    config = tmp_path / "config.yaml"

    workspace._write_config(config, {"providers": {"x": {"api_key": "sentinel"}}})

    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert "sentinel" in config.read_text()
    assert list(tmp_path.glob(".config.yaml.*")) == []


def test_mutating_command_requires_admin_key_before_writing(tmp_path, monkeypatch):
    workspace = load_workspace_module()
    config = tmp_path / "config.yaml"
    original = "providers:\n  workspace_a:\n    base_url: https://example.invalid\npools: {}\n"
    config.write_text(original)
    monkeypatch.setattr(workspace, "ADMIN_KEY", "")

    with pytest.raises(SystemExit, match="MODEL_GATEWAY_ADMIN_KEY is required"):
        workspace.cmd_remove(argparse.Namespace(config=config, name="workspace_a"))

    assert config.read_text() == original
    assert list(tmp_path.glob("*.bak-*")) == []
