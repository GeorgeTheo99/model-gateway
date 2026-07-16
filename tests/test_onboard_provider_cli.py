import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.onboarding import OnboardingError


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "onboard_provider.py"
SPEC = importlib.util.spec_from_file_location("onboard_provider_cli", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CLI)


def test_operator_cli_resolves_its_installed_symlink(tmp_path):
    link = tmp_path / "model-gateway"
    link.symlink_to(SCRIPT.parents[1] / "bin" / "model-gateway")
    env = dict(os.environ)
    env["MODEL_GATEWAY_PLIST_DIR"] = str(tmp_path / "plists")
    result = subprocess.run([str(link), "env"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert f"ROOT_DIR={SCRIPT.parents[1]}" in result.stdout


def test_prompt_secret_fails_clearly_without_tty(monkeypatch):
    monkeypatch.setattr(CLI.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(OnboardingError, match="no interactive terminal"):
        CLI._prompt_secret()


def test_prompt_secret_rejects_empty_input(monkeypatch):
    monkeypatch.setattr(CLI.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(CLI.getpass, "getpass", lambda _prompt: "")
    with pytest.raises(OnboardingError, match="cannot be empty"):
        CLI._prompt_secret()


def test_missing_keychain_item_falls_back_to_hidden_prompt(tmp_path, monkeypatch, capsys):
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "schema_version: 1\nid: test\n"
        "provider:\n  id: test\n  base_url: https://api.example.com/v1\n"
        "models:\n  - name: test\n    provider: test\n    provider_model_id: test\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text("providers: {}\n")
    model_info = tmp_path / "model-info.json"
    model_info.write_text('{"llm": []}')
    captured = {}

    monkeypatch.setattr(CLI.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(CLI, "_keychain_secret", lambda *_args: (_ for _ in ()).throw(OnboardingError("missing")))
    monkeypatch.setattr(CLI, "_prompt_secret", lambda: "prompt-secret")
    monkeypatch.setattr(CLI, "apply_profile", lambda _profile, **kwargs: captured.update(kwargs) or {"ok": True})
    monkeypatch.setattr(CLI.sys, "argv", [
        str(SCRIPT), str(profile), "--config", str(config), "--model-info", str(model_info),
        "--api-key-keychain-service", "missing-service",
    ])

    assert CLI.main() == 0
    assert captured["api_key"] == "prompt-secret"
    assert "falling back to the secure terminal prompt" in capsys.readouterr().err


def _generation_files(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("providers: {}\n")
    model_info = tmp_path / "model-info.json"
    model_info.write_text('{"llm": []}')
    return config, model_info


def test_generate_cli_writes_secret_free_reviewable_draft(tmp_path, monkeypatch):
    config, model_info = _generation_files(tmp_path)
    output = tmp_path / "draft.yaml"
    monkeypatch.setattr(CLI, "discover_models", lambda *_args: {
        "status": "verified",
        "source": "provider_models",
        "model_ids": ["kimi-k3"],
        "http_status": 200,
    })
    result = CLI._generate_main([
        "--provider", "moonshot",
        "--base-url", "https://api.moonshot.ai/v1",
        "--model", "kimi-k3",
        "--output", str(output),
        "--config", str(config),
        "--model-info", str(model_info),
        "--non-interactive",
    ])
    assert result["draft"] == str(output.resolve())
    assert result["applied"] is False
    profile = yaml.safe_load(output.read_text())
    assert profile["provenance"]["fields"]["/models/0/provider_model_id"]["confidence"] == "verified"
    assert "api_key" not in output.read_text()


def test_generate_apply_passes_saved_profile_unchanged_to_transaction_engine(tmp_path, monkeypatch):
    config, model_info = _generation_files(tmp_path)
    output = tmp_path / "draft.yaml"
    captured = {}
    monkeypatch.setenv("PROVIDER_API_KEY", "secret")
    monkeypatch.setattr(CLI, "discover_models", lambda *_args: {
        "status": "verified",
        "source": "provider_models",
        "model_ids": ["model-a"],
    })
    monkeypatch.setattr(CLI, "apply_profile", lambda profile, **kwargs: captured.update(profile=profile, kwargs=kwargs) or {"ok": True})
    result = CLI._generate_main([
        "--provider", "example",
        "--base-url", "https://api.example.com/v1",
        "--model", "model-a",
        "--output", str(output),
        "--apply",
        "--non-interactive",
        "--yes",
        "--api-key-env", "PROVIDER_API_KEY",
        "--config", str(config),
        "--model-info", str(model_info),
    ])
    assert result["applied"] is True
    assert captured["profile"] == yaml.safe_load(output.read_text())
    assert captured["kwargs"]["check_upstream"] is True
    assert captured["kwargs"]["api_key"] == "secret"


def test_generate_output_cannot_overwrite_provider_secret(tmp_path, monkeypatch):
    config, model_info = _generation_files(tmp_path)
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    secret = secret_dir / "example.api-key"
    secret.write_text("original-secret\n")
    secret.chmod(0o600)
    monkeypatch.setenv("MODEL_GATEWAY_SECRET_DIR", str(secret_dir))
    with pytest.raises(OnboardingError, match="protected runtime file"):
        CLI._generate_main([
            "--provider", "example",
            "--base-url", "https://api.example.com/v1",
            "--model", "model-a",
            "--no-discover",
            "--output", str(secret),
            "--force",
            "--config", str(config),
            "--model-info", str(model_info),
        ])
    assert secret.read_text() == "original-secret\n"


def test_generate_output_cannot_overwrite_any_configured_secret(tmp_path, monkeypatch):
    config, model_info = _generation_files(tmp_path)
    custom = tmp_path / "custom-provider.key"
    custom.write_text("custom-secret\n")
    custom.chmod(0o600)
    config.write_text(
        f"providers:\n  other:\n    base_url: https://other.example/v1\n    api_key_file: {custom}\n"
    )
    monkeypatch.setenv("MODEL_GATEWAY_SECRET_DIR", str(tmp_path / "new-secrets"))
    with pytest.raises(OnboardingError, match="protected runtime file"):
        CLI._generate_main([
            "--provider", "example",
            "--base-url", "https://api.example.com/v1",
            "--model", "model-a",
            "--no-discover",
            "--output", str(custom),
            "--force",
            "--config", str(config),
            "--model-info", str(model_info),
        ])
    assert custom.read_text() == "custom-secret\n"


def test_generate_non_interactive_probe_requires_explicit_yes(tmp_path, monkeypatch):
    config, model_info = _generation_files(tmp_path)
    monkeypatch.setenv("PROVIDER_API_KEY", "secret")
    with pytest.raises(OnboardingError, match="probes were not approved"):
        CLI._generate_main([
            "--provider", "example",
            "--base-url", "https://api.example.com/v1",
            "--model", "model-a",
            "--no-discover",
            "--probe", "text",
            "--non-interactive",
            "--api-key-env", "PROVIDER_API_KEY",
            "--output", str(tmp_path / "draft.yaml"),
            "--config", str(config),
            "--model-info", str(model_info),
        ])


def test_generate_stdout_cannot_be_applied(tmp_path):
    config, model_info = _generation_files(tmp_path)
    with pytest.raises(OnboardingError, match="must be saved"):
        CLI._generate_main([
            "--provider", "example",
            "--base-url", "https://api.example.com/v1",
            "--model", "model-a",
            "--no-discover",
            "--stdout",
            "--apply",
            "--config", str(config),
            "--model-info", str(model_info),
        ])


def test_generation_validates_url_before_loading_or_sending_secrets(tmp_path, monkeypatch):
    config, model_info = _generation_files(tmp_path)
    config.write_text(
        "providers:\n  example:\n    base_url: https://api.example.com/v1\n    api_key: must-not-send\n"
    )
    monkeypatch.setattr(
        CLI,
        "discover_models",
        lambda *_args: (_ for _ in ()).throw(AssertionError("network must not be called")),
    )
    with pytest.raises(OnboardingError, match="HTTPS URL"):
        CLI._generate_main([
            "--provider", "example",
            "--base-url", "http://attacker.example/v1",
            "--model", "model-a",
            "--output", str(tmp_path / "draft.yaml"),
            "--config", str(config),
            "--model-info", str(model_info),
        ])


def test_configured_secret_requires_matching_base_url(tmp_path):
    config, _model_info = _generation_files(tmp_path)
    secret = tmp_path / "provider.key"
    secret.write_text("secret\n")
    secret.chmod(0o600)
    config.write_text(
        f"providers:\n  example:\n    base_url: https://old.example/v1\n    api_key_file: {secret}\n"
    )
    assert CLI._configured_secret(
        config, "example", "https://new.example/v1"
    ) == ""
    assert CLI._configured_secret(
        config, "example", "https://old.example/v1"
    ) == "secret"


def test_generate_parser_disables_apply_abbreviation():
    with pytest.raises(SystemExit):
        CLI._generate_parser().parse_args([
            "--provider", "example",
            "--base-url", "https://api.example.com/v1",
            "--model", "model-a",
            "--app",
        ])


def test_generated_noninteractive_apply_requires_exact_safety_acknowledgements(tmp_path):
    config, model_info = _generation_files(tmp_path)
    args = SimpleNamespace(
        allow_metadata_removal=False,
        confirm_replace=[],
        confirm_retire=[],
        non_interactive=True,
        yes=True,
    )
    profile = {
        "models": [{"name": "model-a"}],
        "retire": {"models": ["old"]},
        "provenance": {
            "generator": "model-gateway onboard generate",
            "safety": {
                "metadata_removals": {"model-a": ["alias"]},
                "retirements": ["old"],
                "catalog_fingerprints": {"model-a": None, "old": None},
            },
        }
    }
    allow, replacements, retirements = CLI._interactive_profile_approvals(
        args,
        profile,
        {
            "metadata_removals": {"model-a": ["alias"]},
            "replaced_models": ["model-a"],
        },
        generated=True,
    )
    assert allow is False
    assert replacements == set()
    assert retirements == set()
