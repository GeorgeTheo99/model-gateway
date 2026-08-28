from __future__ import annotations

import argparse
import importlib.util
import stat
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "workspace.py"
CLI = Path(__file__).resolve().parents[1] / "bin" / "model-gateway"


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


def test_mutating_command_requires_activation_path_before_writing(tmp_path, monkeypatch):
    workspace = load_workspace_module()
    config = tmp_path / "config.yaml"
    original = "providers:\n  workspace_a:\n    base_url: https://example.invalid\npools: {}\n"
    config.write_text(original)
    monkeypatch.setattr(workspace, "ADMIN_KEY", "")
    monkeypatch.setattr(workspace, "RESTART_BIN", "")

    with pytest.raises(SystemExit, match="require MODEL_GATEWAY_RESTART_BIN or MODEL_GATEWAY_ADMIN_KEY"):
        workspace.cmd_remove(argparse.Namespace(config=config, name="workspace_a"))

    assert config.read_text() == original
    assert list(tmp_path.glob("*.bak-*")) == []


def test_restart_activation_does_not_require_admin_key(monkeypatch):
    workspace = load_workspace_module()
    monkeypatch.setattr(workspace, "ADMIN_KEY", "")
    monkeypatch.setattr(workspace, "RESTART_BIN", "/safe/model-gateway")

    workspace._require_activation_path()


def test_commit_activates_new_config(tmp_path, monkeypatch):
    workspace = load_workspace_module()
    config = tmp_path / "config.yaml"
    config.write_text("providers:\n  old: {}\n")
    monkeypatch.setattr(workspace.time, "strftime", lambda _fmt: "20260828-140000")
    activations = []
    monkeypatch.setattr(workspace, "activate_gateway", lambda: activations.append("ok"))

    workspace._commit_and_activate(config, {"providers": {"new": {}}})

    assert activations == ["ok"]
    assert workspace._load_config(config) == {"providers": {"new": {}}}
    backup = tmp_path / "config.yaml.bak-20260828-140000"
    assert backup.read_text() == "providers:\n  old: {}\n"
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600


def test_failed_activation_restores_and_verifies_previous_config(tmp_path, monkeypatch):
    workspace = load_workspace_module()
    config = tmp_path / "config.yaml"
    original = "providers:\n  old: {}\n"
    config.write_text(original)
    monkeypatch.setattr(workspace.time, "strftime", lambda _fmt: "20260828-140001")
    attempts = []

    def activate():
        attempts.append(config.read_text())
        if len(attempts) == 1:
            raise RuntimeError("new config did not start")

    monkeypatch.setattr(workspace, "activate_gateway", activate)

    with pytest.raises(SystemExit, match="previous configuration restored and verified"):
        workspace._commit_and_activate(config, {"providers": {"new": {}}})

    assert len(attempts) == 2
    assert workspace._load_config(config) == {"providers": {"old": {}}}
    assert config.read_text() == original


def test_atomic_write_preserves_config_symlink(tmp_path):
    workspace = load_workspace_module()
    target = tmp_path / "private-config.yaml"
    target.write_text("providers: {}\n")
    link = tmp_path / "runtime-config.yaml"
    link.symlink_to(target)

    workspace._write_config(link, {"providers": {"new": {}}})

    assert link.is_symlink()
    assert workspace._load_config(target) == {"providers": {"new": {}}}


def test_restart_activation_uses_operator_cli(monkeypatch):
    workspace = load_workspace_module()
    monkeypatch.setattr(workspace, "RESTART_BIN", "/safe/model-gateway")
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="restarted\n", stderr="")

    monkeypatch.setattr(workspace.subprocess, "run", run)

    workspace.restart_gateway()

    assert calls == [
        (["/safe/model-gateway", "restart"], {"capture_output": True, "text": True, "timeout": 90})
    ]


def test_workspace_list_excludes_unpooled_non_databricks_provider(tmp_path, capsys):
    workspace = load_workspace_module()
    config = tmp_path / "workspaces.yaml"
    config.write_text(
        "providers:\n"
        "  ws-a:\n"
        "    base_url: https://example.cloud.databricks.com\n"
        "    auth_refresh: databricks-cli\n"
        "    auth_profile: ws-a\n"
        "  google:\n"
        "    base_url: https://generativelanguage.googleapis.com/v1beta/openai\n"
        "    api_key: secret\n"
        "pools:\n"
        "  default-pool: [ws-a]\n"
    )

    workspace.cmd_list(argparse.Namespace(config=config))

    output = capsys.readouterr().out
    assert "ws-a" in output
    assert "google" not in output


def test_operator_cli_help_lists_workspace_command():
    result = subprocess.run([str(CLI), "help"], capture_output=True, text=True)

    assert result.returncode == 1
    assert "model-gateway workspace COMMAND" in result.stdout
