from __future__ import annotations

import argparse
import base64
import importlib.util
import json
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


def test_check_coverage_checks_pooled_and_directly_bound_models():
    workspace = load_workspace_module()
    config = {
        "models": [
            {"name": "a", "provider": "other", "provider_model_id": "ep-a", "alias": "a", "pool": "p1"},
            {"name": "b", "provider": "ws-x", "provider_model_id": "ep-b", "alias": "b"},
            {"name": "c", "provider": "other", "provider_model_id": "ep-c", "alias": "c"},
        ],
        "pools": {"p1": ["ws-x"]},
    }

    # Pool member + direct binding both checked; unrelated provider is not
    with pytest.raises(SystemExit, match="ep-a"):
        workspace.check_coverage(config, ["p1"], {"ep-b"}, allow_partial=False, provider_names=["ws-x"])

    # ep-b (direct) missing now → also caught
    with pytest.raises(SystemExit, match="ep-b"):
        workspace.check_coverage(config, ["p1"], {"ep-a"}, allow_partial=False, provider_names=["ws-x"])

    # unrelated model c is never checked
    workspace.check_coverage(config, ["p1"], {"ep-a", "ep-b"}, allow_partial=False, provider_names=["ws-x"])


# --- ai-gateway host derivation -------------------------------------------

def fake_jwt(claims: dict) -> str:
    def b64(segment: bytes) -> str:
        return base64.urlsafe_b64encode(segment).rstrip(b"=").decode()

    return f"{b64(b'x')}.{b64(json.dumps(claims).encode())}.sig"


def test_is_ai_gateway_host_classifies_hosts():
    workspace = load_workspace_module()

    assert workspace._is_ai_gateway_host("https://1444828305810485.ai-gateway.cloud.databricks.com")
    assert not workspace._is_ai_gateway_host("https://e2-demo-field-eng.cloud.databricks.com")
    assert not workspace._is_ai_gateway_host("https://e2-dogfood.staging.cloud.databricks.com")


def test_derive_ai_gateway_host_uses_numeric_aud_claim():
    workspace = load_workspace_module()
    token = fake_jwt({"aud": ["1444828305810485"], "iss": "https://e2-demo-field-eng.cloud.databricks.com/oidc"})

    assert workspace.derive_ai_gateway_host(token) == "https://1444828305810485.ai-gateway.cloud.databricks.com"


def test_derive_ai_gateway_host_accepts_string_aud():
    workspace = load_workspace_module()
    token = fake_jwt({"aud": "7474647777725369"})

    assert workspace.derive_ai_gateway_host(token) == "https://7474647777725369.ai-gateway.cloud.databricks.com"


def test_derive_ai_gateway_host_fails_without_numeric_aud():
    workspace = load_workspace_module()
    token = fake_jwt({"aud": ["not-numeric"]})

    with pytest.raises(SystemExit, match="pass the explicit"):
        workspace.derive_ai_gateway_host(token)


def test_resolve_ai_gateway_host_from_workspace_url(monkeypatch):
    workspace = load_workspace_module()
    token = fake_jwt({"aud": ["1444828305810485"], "iss": "https://e2-demo-field-eng.cloud.databricks.com/oidc"})
    minted = []
    monkeypatch.setattr(workspace, "ensure_auth", lambda host, profile: minted.append(host) or token)

    probe_host, gateway_base, out = workspace._resolve_ai_gateway_host(
        "https://e2-demo-field-eng.cloud.databricks.com", "e2-demo"
    )

    assert probe_host == "https://e2-demo-field-eng.cloud.databricks.com"
    assert gateway_base == "https://1444828305810485.ai-gateway.cloud.databricks.com"
    assert out == token
    assert minted == ["https://e2-demo-field-eng.cloud.databricks.com"]


def test_resolve_ai_gateway_host_recovers_workspace_from_iss(monkeypatch):
    workspace = load_workspace_module()
    token = fake_jwt({"aud": ["7474647777725369"], "iss": "https://fevm-model-exp.cloud.databricks.com/oidc"})
    monkeypatch.setattr(workspace, "ensure_auth", lambda host, profile: token)

    probe_host, gateway_base, _ = workspace._resolve_ai_gateway_host(
        "https://7474647777725369.ai-gateway.cloud.databricks.com", "fevm"
    )

    assert probe_host == "https://fevm-model-exp.cloud.databricks.com"
    assert gateway_base == "https://7474647777725369.ai-gateway.cloud.databricks.com"


def test_resolve_ai_gateway_host_requires_iss_for_gateway_host_input(monkeypatch):
    workspace = load_workspace_module()
    token = fake_jwt({"aud": ["1444828305810485"]})  # no iss
    monkeypatch.setattr(workspace, "ensure_auth", lambda host, profile: token)

    with pytest.raises(SystemExit, match="pass the workspace URL instead"):
        workspace._resolve_ai_gateway_host(
            "https://1444828305810485.ai-gateway.cloud.databricks.com", "e2-demo"
        )


def test_cmd_add_ai_gateway_style_derives_base_url_and_sets_workspace_url(tmp_path, monkeypatch):
    workspace = load_workspace_module()
    config = tmp_path / "config.yaml"
    config.write_text(
        "providers:\n"
        "  old: {}\n"
        "pools:\n"
        "  default-pool: []\n"
        "models:\n"
        "  - name: gpt-5.4\n"
        "    provider: new-ws\n"
        "    provider_model_id: databricks-gpt-5-4\n"
        "    alias: gpt54\n"
        "    pool: default-pool\n"
    )
    token = fake_jwt({"aud": ["1444828305810485"], "iss": "https://e2-demo-field-eng.cloud.databricks.com/oidc"})
    monkeypatch.setattr(workspace, "ADMIN_KEY", "")
    monkeypatch.setattr(workspace, "RESTART_BIN", "/safe/model-gateway")
    monkeypatch.setattr(workspace, "ensure_auth", lambda host, profile: token)
    monkeypatch.setattr(workspace, "probe_endpoints", lambda host, tk: {"databricks-gpt-5-4"})
    monkeypatch.setattr(workspace, "smoke_test", lambda host, tk, names: None)
    monkeypatch.setattr(workspace, "restart_gateway", lambda: None)
    verify_hosts = []

    def spy_verify(cfg, host, profile, pools, allow_partial, token=None, provider_names=()):
        verify_hosts.append(host)
        assert provider_names == ["new-ws"]

    monkeypatch.setattr(workspace, "_verify", spy_verify)

    workspace.cmd_add(argparse.Namespace(
        config=config, name="new-ws",
        host="https://e2-demo-field-eng.cloud.databricks.com",
        pools="default-pool", position=None, profile=None,
        style="ai-gateway", allow_partial=False,
    ))

    assert verify_hosts == ["https://e2-demo-field-eng.cloud.databricks.com"]
    entry = workspace._load_config(config)["providers"]["new-ws"]
    assert entry["base_url"] == "https://1444828305810485.ai-gateway.cloud.databricks.com"
    assert entry["workspace_url"] == "https://e2-demo-field-eng.cloud.databricks.com"
    assert entry["api_key"] == token
    assert entry["path_prefixes"] == {"anthropic": "anthropic/v1", "openai": "mlflow/v1"}
    assert entry["quirks"] == ["anthropic_bearer_auth"]
    assert entry.get("endpoint_style") is None
    assert workspace._load_config(config)["pools"]["default-pool"] == ["new-ws"]


def test_cmd_add_invocations_style_keeps_workspace_as_base_url(tmp_path, monkeypatch):
    workspace = load_workspace_module()
    config = tmp_path / "config.yaml"
    config.write_text("providers:\n  old: {}\npools: {}\n")
    token = fake_jwt({"aud": ["1444828305810485"]})
    monkeypatch.setattr(workspace, "ADMIN_KEY", "")
    monkeypatch.setattr(workspace, "RESTART_BIN", "/safe/model-gateway")
    monkeypatch.setattr(workspace, "ensure_auth", lambda host, profile: token)
    monkeypatch.setattr(workspace, "probe_endpoints", lambda host, tk: set())
    monkeypatch.setattr(workspace, "smoke_test", lambda host, tk, names: None)
    monkeypatch.setattr(workspace, "restart_gateway", lambda: None)
    monkeypatch.setattr(workspace, "_verify", lambda *a, **k: None)

    workspace.cmd_add(argparse.Namespace(
        config=config, name="e2",
        host="https://e2-demo-field-eng.cloud.databricks.com",
        pools="", position=None, profile=None,
        style="invocations", allow_partial=False,
    ))

    entry = workspace._load_config(config)["providers"]["e2"]
    assert entry["base_url"] == "https://e2-demo-field-eng.cloud.databricks.com"
    assert entry["endpoint_style"] == "invocations"
    assert "workspace_url" not in entry


def test_cmd_replace_ai_gateway_entry_derives_gateway_host(tmp_path, monkeypatch):
    workspace = load_workspace_module()
    config = tmp_path / "config.yaml"
    config.write_text(
        "providers:\n"
        "  dead-aigw:\n"
        "    base_url: https://0.ai-gateway.cloud.databricks.com\n"
        "    protocol: openai\n"
        "    path_prefixes:\n"
        "      anthropic: anthropic/v1\n"
        "      openai: mlflow/v1\n"
        "    auth_refresh: databricks-cli\n"
        "    auth_profile: dead\n"
        "    quirks: [anthropic_bearer_auth]\n"
        "pools:\n"
        "  default-pool: [dead-aigw]\n"
    )
    token = fake_jwt({"aud": ["1444828305810485"], "iss": "https://e2-demo-field-eng.cloud.databricks.com/oidc"})
    monkeypatch.setattr(workspace, "ADMIN_KEY", "")
    monkeypatch.setattr(workspace, "RESTART_BIN", "/safe/model-gateway")
    monkeypatch.setattr(workspace, "ensure_auth", lambda host, profile: token)
    monkeypatch.setattr(workspace, "probe_endpoints", lambda host, tk: set())
    monkeypatch.setattr(workspace, "smoke_test", lambda host, tk, names: None)
    monkeypatch.setattr(workspace, "restart_gateway", lambda: None)
    monkeypatch.setattr(workspace, "_verify", lambda *a, **k: None)

    workspace.cmd_replace(argparse.Namespace(
        config=config, old_name="dead-aigw",
        host="https://e2-demo-field-eng.cloud.databricks.com",
        name=None, profile=None, allow_partial=False,
    ))

    providers = workspace._load_config(config)["providers"]
    # --name defaults to the old name: in-place replacement keeps the key
    entry = providers["dead-aigw"]
    assert entry["base_url"] == "https://1444828305810485.ai-gateway.cloud.databricks.com"
    assert entry["workspace_url"] == "https://e2-demo-field-eng.cloud.databricks.com"
    assert entry["auth_profile"] == "dead-aigw"  # profile defaults to new name
