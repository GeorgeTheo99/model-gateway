import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.onboarding import OnboardingError


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "onboard_provider.py"
SPEC = importlib.util.spec_from_file_location("onboard_provider_cli", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CLI)


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
