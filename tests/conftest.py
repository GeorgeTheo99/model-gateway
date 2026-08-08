"""Shared test fixtures.

Isolate the request ledger to a per-test temp database by default so no test
ever writes to the production ledger at ~/srv/model-gateway/shared/ledger.db.
Tests that need to inspect the ledger can still override the path.
"""

import os
from pathlib import Path

import pytest

# Provider paths are resolved at import time. Point them at a committed,
# machine-neutral fixture before importing the registry so the suite never
# depends on the operator's Git-ignored model-info.json.
os.environ["MODEL_GATEWAY_MODEL_INFO"] = str(Path(__file__).parent / "fixtures" / "model-info.json")

import src.providers as providers


@pytest.fixture(autouse=True)
def _isolate_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_LEDGER_PATH", str(tmp_path / "test-ledger.db"))
    yield


@pytest.fixture(autouse=True)
def _isolate_runtime_config(tmp_path, monkeypatch):
    """Keep gitignored local provider config from changing test behavior."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("providers: {}\n")
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(tmp_path / "config-backups"))
    monkeypatch.setenv("MODEL_GATEWAY_LEGACY_BACKUP_DIRS", str(tmp_path / "logs" / "config-backups"))
    from src import config_io
    monkeypatch.setattr(config_io, "log_dir", tmp_path / "logs")
    monkeypatch.setattr(providers, "CONFIG_PATH", cfg)
    providers.reload()
    yield
    providers.reload()
