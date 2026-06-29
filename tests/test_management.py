"""Tests for writeable provider/model management (milestones 4 & 5)."""

import json

import pytest

from src import config_io
import src.providers as providers
from src.server import app

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None


# ── config_io: provider writes ──────────────────────────────────────────────


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Isolate config.yaml + model-info.json to temp dirs."""
    cfg = tmp_path / "config.yaml"
    monkeypatch.setattr(providers, "CONFIG_PATH", cfg)
    monkeypatch.setattr(config_io, "CONFIG_PATH", cfg)
    # Minimal config with one provider + auth.
    cfg.write_text("auth:\n  admin_keys:\n    - admin\nproviders:\n  anthropic:\n    base_url: https://api.anthropic.com/v1\n    api_key: secret-existing\n    protocol: anthropic\n")
    # Model-info: temp file, no source mirror.
    mi = tmp_path / "model-info.json"
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", mi)
    monkeypatch.setattr(config_io, "MODEL_INFO_PATH", mi)
    monkeypatch.setattr(providers, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.setattr(config_io, "MODEL_INFO_SOURCE_PATH", None)
    mi.write_text(json.dumps({"llm": [
        {"name": "claude-test", "provider": "anthropic", "provider_model_id": "claude-test-1",
         "context": 200000, "max_output_tokens": 8192, "pricing": {"input": 3.0, "output": 15.0}},
    ]}))
    providers.reload()
    return tmp_path


def test_upsert_provider_creates_new(tmp_config, monkeypatch):
    # Set log/backup dir to tmp so backups don't litter the real log dir.
    monkeypatch.setenv("CLOUD_GATEWAY_LOG_DIR", str(tmp_config / "logs"))
    config_io.log_dir = tmp_config / "logs"
    result = config_io.upsert_provider("openai", base_url="https://api.openai.com/v1", api_key="sk-new")
    assert result["id"] == "openai"
    assert result["has_api_key"] is True
    assert result["base_url"] == "https://api.openai.com/v1"
    # Verify it landed in config.yaml.
    import yaml
    cfg = yaml.safe_load((tmp_config / "config.yaml").read_text())
    assert cfg["providers"]["openai"]["api_key"] == "sk-new"


def test_upsert_provider_preserves_existing_key(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    # Update anthropic base_url, api_key=None -> existing key preserved.
    config_io.upsert_provider("anthropic", base_url="https://api.anthropic.com/v2", api_key=None)
    import yaml
    cfg = yaml.safe_load((tmp_config / "config.yaml").read_text())
    assert cfg["providers"]["anthropic"]["api_key"] == "secret-existing"
    assert cfg["providers"]["anthropic"]["base_url"] == "https://api.anthropic.com/v2"


def test_upsert_provider_empty_key_removes(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    config_io.upsert_provider("anthropic", base_url="https://api.anthropic.com/v1", api_key="")
    import yaml
    cfg = yaml.safe_load((tmp_config / "config.yaml").read_text())
    assert "api_key" not in cfg["providers"]["anthropic"]


def test_delete_provider_refuses_if_models_depend(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    with pytest.raises(ValueError, match="depend on it"):
        config_io.delete_provider("anthropic")


def test_delete_provider_after_disabling_models(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    config_io.set_model_enabled("claude-test", False)
    providers.reload()
    result = config_io.delete_provider("anthropic")
    assert result["deleted"] is True
    import yaml
    cfg = yaml.safe_load((tmp_config / "config.yaml").read_text())
    assert "anthropic" not in cfg["providers"]


def test_delete_unknown_provider(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    with pytest.raises(KeyError):
        config_io.delete_provider("nonexistent")


# ── config_io: model writes ─────────────────────────────────────────────────


def test_upsert_model_creates(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    result = config_io.upsert_model(
        "gpt-test", provider="openai", provider_model_id="gpt-test-1",
        context=128000, max_output_tokens=16384, pricing={"input": 2.5, "output": 15.0},
    )
    assert result["name"] == "gpt-test"
    assert result["entry"]["provider_model_id"] == "gpt-test-1"
    # Verify it landed in model-info.json.
    doc = json.loads((tmp_config / "model-info.json").read_text())
    names = [e["name"] for e in doc["llm"]]
    assert "gpt-test" in names


def test_upsert_model_updates_existing(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    config_io.upsert_model("claude-test", provider="anthropic",
                           provider_model_id="claude-test-1", context=300000)
    doc = json.loads((tmp_config / "model-info.json").read_text())
    entry = next(e for e in doc["llm"] if e["name"] == "claude-test")
    assert entry["context"] == 300000
    # pricing preserved (not provided in update).
    assert entry["pricing"] == {"input": 3.0, "output": 15.0}


def test_set_model_enabled_writes_config_yaml(tmp_config, monkeypatch):
    """enabled state lives in config.yaml model_overrides, not model-info.json."""
    config_io.log_dir = tmp_config / "logs"
    r = config_io.set_model_enabled("claude-test", False)
    assert r["enabled"] is False
    assert str(tmp_config / "config.yaml") in r["written_to"]
    # Override landed in config.yaml.
    import yaml
    cfg = yaml.safe_load((tmp_config / "config.yaml").read_text())
    assert cfg["model_overrides"]["claude-test"]["enabled"] is False
    # model-info.json is NOT touched by the toggle.
    doc = json.loads((tmp_config / "model-info.json").read_text())
    entry = next(e for e in doc["llm"] if e["name"] == "claude-test")
    assert "enabled" not in entry


def test_atomic_write_preserves_symlink(tmp_path, monkeypatch):
    """config.yaml is a symlink to a shared file; writes must update the target,
    not replace the link with a real file (which would split config state)."""
    config_io.log_dir = tmp_path / "logs"
    # Set up: deploy dir with a symlink to a shared real file.
    shared = tmp_path / "shared.yaml"
    shared.write_text("providers:\n  anthropic:\n    base_url: https://x\n    api_key: k\n")
    link = tmp_path / "deploy" / "config.yaml"
    link.parent.mkdir(parents=True)
    link.symlink_to(shared)
    monkeypatch.setattr(providers, "CONFIG_PATH", link)
    monkeypatch.setattr(config_io, "CONFIG_PATH", link)
    providers.reload()
    # Write via config_io (upsert_provider uses _atomic_write).
    config_io.upsert_provider("anthropic", base_url="https://y", api_key="k2")
    # The link must still be a symlink.
    assert link.is_symlink(), "symlink was replaced by a real file"
    # And the shared target must hold the new content.
    import yaml
    d = yaml.safe_load(shared.read_text())
    assert d["providers"]["anthropic"]["base_url"] == "https://y"


def test_delete_model(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    config_io.delete_model("claude-test")
    doc = json.loads((tmp_config / "model-info.json").read_text())
    assert all(e["name"] != "claude-test" for e in doc["llm"])


def test_upsert_model_validates_required(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    with pytest.raises(ValueError, match="provider_model_id is required"):
        config_io.upsert_model("x", provider="openai", provider_model_id="")


# ── enabled field routing ───────────────────────────────────────────────────


def test_resolve_skips_disabled_model(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    assert providers.resolve("claude-test") is not None  # enabled by default
    config_io.set_model_enabled("claude-test", False)
    providers.reload()
    assert providers.resolve("claude-test") is None  # disabled -> not routable


# ── admin API endpoints ─────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    monkeypatch.setenv("CLOUD_GATEWAY_ADMIN_KEY", "admin")
    with TestClient(app) as c:
        yield c


def test_admin_upsert_provider_endpoint(client):
    resp = client.post("/admin/api/providers/openai",
                       headers={"Authorization": "Bearer admin"},
                       json={"base_url": "https://api.openai.com/v1", "api_key": "sk-x"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "openai"
    assert body["has_api_key"] is True
    assert body["reloaded"] is True


def test_admin_provider_endpoints_require_admin_key(client):
    assert client.post("/admin/api/providers/x", json={"base_url": "u"}).status_code == 401
    assert client.delete("/admin/api/providers/x").status_code == 401


def test_admin_upsert_model_endpoint(client):
    resp = client.post("/admin/api/models/new-model",
                       headers={"Authorization": "Bearer admin"},
                       json={"provider": "openai", "provider_model_id": "new-1",
                             "context": 128000, "max_output_tokens": 4096})
    assert resp.status_code == 200
    assert resp.json()["name"] == "new-model"
    # Catalog now contains the model (resolve also needs provider config).
    assert "new-model" in providers._load_models()


def test_admin_disable_enable_model_endpoint(client):
    h = {"Authorization": "Bearer admin"}
    assert client.post("/admin/api/models/claude-test/disable", headers=h).status_code == 200
    assert providers.resolve("claude-test") is None
    assert client.post("/admin/api/models/claude-test/enable", headers=h).status_code == 200
    assert providers.resolve("claude-test") is not None


def test_admin_delete_model_endpoint(client):
    h = {"Authorization": "Bearer admin"}
    assert client.delete("/admin/api/models/claude-test", headers=h).status_code == 200
    assert providers.resolve("claude-test") is None


def test_admin_delete_provider_refuses_with_dependents(client):
    h = {"Authorization": "Bearer admin"}
    resp = client.delete("/admin/api/providers/anthropic", headers=h)
    assert resp.status_code == 409
    assert "depend on it" in resp.json()["error"]["message"]


def test_admin_upsert_provider_validates_required(client):
    h = {"Authorization": "Bearer admin"}
    resp = client.post("/admin/api/providers/x", headers=h, json={"base_url": ""})
    assert resp.status_code == 400
