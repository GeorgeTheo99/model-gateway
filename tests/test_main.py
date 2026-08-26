"""Entrypoint logging safety tests."""

from __future__ import annotations

import runpy
from pathlib import Path


def test_uvicorn_access_log_is_disabled(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    runpy.run_module("src.main", run_name="__main__")

    assert captured["host"] == "127.0.0.1"
    assert captured["access_log"] is False


def test_portable_launchagent_uses_private_log_and_backup_contract():
    script = (Path(__file__).parents[1] / "bin" / "model-gateway").read_text()

    assert "umask 077" in script
    assert "<key>MODEL_GATEWAY_LOG_DIR</key><string>${e_log_dir}</string>" in script
    assert "<key>MODEL_GATEWAY_BACKUP_DIR</key><string>${e_backup}</string>" in script
    assert "<key>MODEL_GATEWAY_LEGACY_BACKUP_DIRS</key><string>${e_legacy_backups}</string>" in script
    assert "<key>Umask</key><integer>63</integer>" in script
    assert 'ensure_private_dir "$LOG_DIR"' in script
    assert 'ensure_private_dir "$BACKUP_DIR"' in script
    assert 'ensure_private_file "$LOG_FILE"' in script
    assert "ensure_log_backup_separation" in script
    assert 'chmod 600 "$plist"' in script
    assert "$HOME/.claude" not in script
    assert "~/srv/model-gateway/shared/model-aliases.json" in script
