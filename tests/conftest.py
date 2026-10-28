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

import src.auth as auth
import src.providers as providers
import src.server as server


@pytest.fixture(autouse=True)
def _clear_vision_observation_cache():
    """Extraction results are cached process-wide; keep tests independent."""
    server._vision_observation_cache.clear()
    yield
    server._vision_observation_cache.clear()


@pytest.fixture(autouse=True)
def _isolate_local_runtime(tmp_path, monkeypatch):
    """Local AI state stays in tmp; launchctl, sysctl, uv, and the network never run."""
    import subprocess

    from src import local_runtime

    monkeypatch.setenv("MODEL_GATEWAY_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MODEL_GATEWAY_PLIST_DIR", str(tmp_path / "LaunchAgents"))

    def launchctl(*args):
        if args[:1] == ("print",):
            return subprocess.CompletedProcess(["launchctl", *args], 113, "", "")
        raise AssertionError(f"test attempted launchctl {args}")

    def forbidden(*args, **kwargs):
        raise AssertionError("test attempted a local AI system or network call")

    monkeypatch.setattr(local_runtime, "_launchctl", launchctl)
    monkeypatch.setattr(local_runtime, "_memory_bytes", lambda: 0)
    monkeypatch.setattr(local_runtime, "_port_in_use", lambda port: False)
    for name in ("_run", "_request", "_download_opener"):
        monkeypatch.setattr(local_runtime, name, forbidden)
    from src import media_runtime

    monkeypatch.setattr(media_runtime, "_run", forbidden)
    yield


@pytest.fixture(autouse=True)
def _no_real_upstreams(monkeypatch):
    """No test reaches a real upstream: this Mac runs a production oMLX and gateway on the
    default ports. Tests stub upstream calls or use httpx.MockTransport."""
    import httpx

    attempts = []

    def refuse(request):
        # Recorded as well as raised: a handler's own except clause may turn the raise into an error response.
        attempts.append(f"{request.method} {request.url.host}:{request.url.port}")
        raise AssertionError(f"test attempted a real upstream request to {request.url.host}:{request.url.port}")

    async def refuse_async(self, request):
        refuse(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: refuse(request))
    yield
    assert not attempts, f"test attempted real upstream requests: {attempts}"


@pytest.fixture(autouse=True)
def _isolate_shipped_layer(monkeypatch):
    """Released curated profiles load only in tests that opt in (patch ``shipped._packaged``)."""
    from src import shipped

    monkeypatch.setattr(shipped, "_packaged", lambda: {})
    yield


@pytest.fixture(autouse=True)
def _isolate_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_LEDGER_PATH", str(tmp_path / "test-ledger.db"))
    yield


@pytest.fixture(autouse=True)
def _isolate_runtime_config(tmp_path, monkeypatch):
    """Keep gitignored local provider/config state from changing test behavior."""
    cfg = tmp_path / "config.yaml"
    monkeypatch.setenv("MODEL_GATEWAY_PROFILE_REGISTRY", str(tmp_path / "consumer-profiles-registry.json"))
    cfg.write_text("providers: {}\n")
    monkeypatch.setenv("MODEL_GATEWAY_BACKUP_DIR", str(tmp_path / "config-backups"))
    # Provider key writes must never touch ~/.config/model-gateway/secrets.
    monkeypatch.setenv("MODEL_GATEWAY_SECRET_DIR", str(tmp_path / "secrets"))
    # Startup writes the consumer discovery file; never touch the real one.
    monkeypatch.setenv("MODEL_GATEWAY_ENDPOINT_FILE", str(tmp_path / "endpoint.json"))
    monkeypatch.setenv("MODEL_GATEWAY_LEGACY_BACKUP_DIRS", str(tmp_path / "logs" / "config-backups"))
    from src import config_io
    monkeypatch.setattr(config_io, "log_dir", tmp_path / "logs")
    monkeypatch.setattr(providers, "CONFIG_PATH", cfg)
    monkeypatch.setattr(config_io, "CONFIG_PATH", cfg)
    monkeypatch.setattr(auth, "_client_auth_required_latched", False)
    monkeypatch.setattr(providers, "_last_good", None)
    monkeypatch.setattr(providers, "_load_error", None)
    providers.reload()
    yield
    providers.reload()
