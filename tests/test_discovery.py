"""Consumer discovery file (``endpoint.json``)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src import config_io, discovery
import src.providers as providers


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for name in ("MODEL_GATEWAY_HOST", "MODEL_GATEWAY_PORT", "MODEL_GATEWAY_CLIENT_KEYS", "MODEL_GATEWAY_CLIENT_KEYS_FILE"):
        monkeypatch.delenv(name, raising=False)
    path = Path(os.path.realpath(tmp_path)) / "config.yaml"
    path.write_text(
        "auth:\n  client_keys: [client-token]\n"
        "exports:\n  model_aliases: ~/aliases/model-aliases.json\n"
        "providers: {}\n"
    )
    path.chmod(0o600)
    monkeypatch.setattr(providers, "CONFIG_PATH", path)
    monkeypatch.setattr(config_io, "CONFIG_PATH", path)
    providers.reload()
    return path


def test_document_lists_urls_paths_and_consumers_without_keys(cfg, tmp_path, monkeypatch):
    config_io.add_consumer_credential("myai", "runtime", allow_direct_models=True)
    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS_FILE", str(tmp_path / "client.key"))
    providers.reload()
    doc = discovery.endpoint_document()
    key_file = cfg.parent / "secrets" / "consumers" / "myai-runtime.key"
    assert doc["version"] == 1 and doc["service"] == "model-gateway"
    assert doc["base_url"] == "http://127.0.0.1:9111/v1"
    assert doc["health_url"] == "http://127.0.0.1:9111/health"
    assert doc["port"] == 9111
    assert doc["model_aliases"] == str(Path.home() / "aliases" / "model-aliases.json")
    assert doc["client_key_file"] == str(tmp_path / "client.key")
    assert doc["consumers"] == {"myai-runtime": {
        "consumer": "myai", "namespaces": ["myai"],
        "permissions": ["profiles:read", "profiles:invoke"],
        "allow_direct_models": True, "key_file": str(key_file),
    }}
    assert key_file.read_text().strip() not in json.dumps(doc)
    assert "client-token" not in json.dumps(doc)


@pytest.mark.parametrize("host,origin", [
    ("0.0.0.0", "http://127.0.0.1:19111"),
    ("::", "http://127.0.0.1:19111"),
    ("100.100.1.2", "http://100.100.1.2:19111"),
    ("::1", "http://[::1]:19111"),
])
def test_document_uses_a_connectable_origin(cfg, monkeypatch, host, origin):
    monkeypatch.setenv("MODEL_GATEWAY_HOST", host)
    monkeypatch.setenv("MODEL_GATEWAY_PORT", "19111")
    assert discovery.endpoint_document()["base_url"] == f"{origin}/v1"


def test_write_is_private_atomic_and_can_be_disabled(cfg, tmp_path, monkeypatch):
    target = tmp_path / "state" / "endpoint.json"
    monkeypatch.setenv("MODEL_GATEWAY_ENDPOINT_FILE", str(target))
    assert discovery.write_endpoint_file() == target
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.parent.stat().st_mode & 0o777 == 0o700
    assert json.loads(target.read_text())["port"] == 9111
    monkeypatch.setenv("MODEL_GATEWAY_ENDPOINT_FILE", "")
    assert discovery.endpoint_path() is None and discovery.write_endpoint_file() is None


def test_write_refuses_symlink_and_refresh_never_raises(cfg, tmp_path, monkeypatch, caplog):
    real = tmp_path / "elsewhere.json"
    real.write_text("{}")
    link = tmp_path / "endpoint.json"
    link.symlink_to(real)
    monkeypatch.setenv("MODEL_GATEWAY_ENDPOINT_FILE", str(link))
    with pytest.raises(OSError, match="symlink"):
        discovery.write_endpoint_file()
    discovery.refresh()
    assert real.read_text() == "{}"
    assert "discovery file not written" in caplog.text


def test_default_path_is_application_support(monkeypatch):
    monkeypatch.delenv("MODEL_GATEWAY_ENDPOINT_FILE", raising=False)
    assert discovery.endpoint_path() == Path.home() / "Library" / "Application Support" / "model-gateway" / "endpoint.json"


def test_startup_writes_the_discovery_file(tmp_path):
    from fastapi.testclient import TestClient
    from src.server import app

    with TestClient(app):
        pass
    assert json.loads((tmp_path / "endpoint.json").read_text())["service"] == "model-gateway"
