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
    monkeypatch.setenv("MODEL_GATEWAY_LOG_DIR", str(tmp_config / "logs"))
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


def test_resolve_local_omlx_model_uses_builtin_proxy_defaults(tmp_config, monkeypatch):
    doc = json.loads((tmp_config / "model-info.json").read_text())
    doc["llm"].append({
        "name": "local-test",
        "alias": "lt",
        "omlx_id": "local-upstream",
        "context": 65536,
        "max_output_tokens": 4096,
        "thinking": "always",
        "thinking_format": "glm-chat-template",
    })
    (tmp_config / "model-info.json").write_text(json.dumps(doc))
    providers.reload()

    for model_id in ("local-test", "lt", "local-upstream"):
        info = providers.resolve(model_id)
        assert info is not None
        assert info.provider == "omlx"
        assert info.base_url == "http://localhost:9110/v1"
        assert info.api_key == "omlx"
        assert info.provider_model_id == "local-upstream"
        assert info.protocol == "openai"

    omlx_status = next(p for p in providers.provider_status() if p["id"] == "omlx")
    assert omlx_status["ready"] is True
    assert omlx_status["issues"] == []


def test_provider_status_counts_unique_models_not_identifiers(tmp_config, monkeypatch):
    """A model with name + alias + provider_model_id + omlx_id must count as 1."""
    doc = json.loads((tmp_config / "model-info.json").read_text())
    doc["llm"].append({
        "name": "local-multi", "alias": "lm", "omlx_id": "lm-up",
        "provider_model_id": "lm-pmid", "context": 4096, "max_output_tokens": 1024,
    })
    (tmp_config / "model-info.json").write_text(json.dumps(doc))
    providers.reload()
    counts = {p["id"]: p["enabled_models"] for p in providers.provider_status()}
    # omlx has exactly one unique local model (local-multi); claude-test is anthropic.
    assert counts.get("omlx") == 1
    assert counts.get("anthropic") == 1
    # routable_ids exposes all four identifiers for the local model.
    assert set(providers.routable_ids("local-multi")) == {"local-multi", "lm", "lm-up", "lm-pmid"}


def test_upsert_model_allows_local_omlx_id(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    result = config_io.upsert_model(
        "local-created", provider="omlx", omlx_id="local-created-upstream",
        context=32768, max_output_tokens=2048,
    )
    assert result["entry"]["omlx_id"] == "local-created-upstream"
    providers.reload()
    info = providers.resolve("local-created")
    assert info is not None
    assert info.provider_model_id == "local-created-upstream"


def test_unconfigured_provider_models_are_hidden_from_v1_models(tmp_config, monkeypatch):
    doc = json.loads((tmp_config / "model-info.json").read_text())
    doc["llm"].append({
        "name": "gpt-unconfigured",
        "provider": "openai",
        "provider_model_id": "gpt-test",
        "context": 128000,
        "max_output_tokens": 4096,
    })
    (tmp_config / "model-info.json").write_text(json.dumps(doc))
    providers.reload()

    assert providers.resolve("gpt-unconfigured") is None
    assert providers.model_availability("gpt-unconfigured")["reason"] == "provider_not_configured"

    with TestClient(app) as c:
        ids = {m["id"] for m in c.get("/v1/models").json()["data"]}
        assert "claude-test" in ids
        assert "gpt-unconfigured" not in ids

        admin = {"Authorization": "Bearer admin"}
        row = next(
            m for m in c.get("/admin/api/models", headers=admin).json()["models"]
            if m["name"] == "gpt-unconfigured"
        )
        assert row["available"] is False
        assert row["availability_reason"] == "provider_not_configured"


def test_requesting_unconfigured_model_returns_clear_error(tmp_config, monkeypatch):
    doc = json.loads((tmp_config / "model-info.json").read_text())
    doc["llm"].append({"name": "gpt-unconfigured", "provider": "openai", "provider_model_id": "gpt-test"})
    (tmp_config / "model-info.json").write_text(json.dumps(doc))
    providers.reload()

    with TestClient(app) as c:
        resp = c.post("/v1/chat/completions", json={"model": "gpt-unconfigured", "messages": []})
    assert resp.status_code == 404
    message = resp.json()["error"]["message"]
    assert "provider_not_configured" in message
    assert "openai" in message


def test_databricks_provider_can_be_configured_from_env(tmp_config, monkeypatch):
    doc = json.loads((tmp_config / "model-info.json").read_text())
    doc["llm"].append({
        "name": "dbx-chat",
        "provider": "databricks",
        "provider_model_id": "my-serving-endpoint",
    })
    (tmp_config / "model-info.json").write_text(json.dumps(doc))
    with open(tmp_config / "config.yaml", "a") as f:
        f.write("  databricks:\n    enabled: false\n")
    providers.reload()

    assert providers.model_availability("dbx-chat")["reason"] == "provider_disabled"

    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-placeholder")
    providers.reload()

    info = providers.resolve("dbx-chat")
    assert info is not None
    assert info.provider == "databricks"
    assert info.base_url == "https://workspace.example.databricks.com/serving-endpoints"
    assert info.api_key == "dapi-placeholder"
    assert info.protocol == "openai"


# ── admin API endpoints ─────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin")
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_readonly(tmp_config, monkeypatch):
    """Admin auth on, but writes disabled (the default read-only dashboard)."""
    config_io.log_dir = tmp_config / "logs"
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin")
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_WRITES", raising=False)
    with TestClient(app) as c:
        yield c


def test_admin_writes_disabled_blocks_management(client_readonly):
    """With MODEL_GATEWAY_ADMIN_WRITES unset, mutating endpoints return 403."""
    h = {"Authorization": "Bearer admin"}
    # Provider writes
    assert client_readonly.post("/admin/api/providers/openai", headers=h, json={"base_url": "u"}).status_code == 403
    assert client_readonly.delete("/admin/api/providers/anthropic", headers=h).status_code == 403
    assert client_readonly.post("/admin/api/providers/anthropic/validate", headers=h).status_code == 403
    # Model writes
    assert client_readonly.post("/admin/api/models/new", headers=h, json={"provider": "openai", "provider_model_id": "x"}).status_code == 403
    assert client_readonly.delete("/admin/api/models/claude-test", headers=h).status_code == 403
    assert client_readonly.post("/admin/api/models/claude-test/disable", headers=h).status_code == 403
    assert client_readonly.post("/admin/api/models/claude-test/enable", headers=h).status_code == 403
    # Reload is also gated
    assert client_readonly.post("/admin/api/reload", headers=h).status_code == 403
    # Read-only endpoints still work
    assert client_readonly.get("/admin/api/status", headers=h).status_code == 200
    assert client_readonly.get("/admin/api/providers", headers=h).status_code == 200
    assert client_readonly.get("/admin/api/models", headers=h).status_code == 200
    assert client_readonly.get("/admin/api/models/claude-test/stats", headers=h).status_code == 200


def test_admin_status_reports_writes_enabled_true(client):
    h = {"Authorization": "Bearer admin"}
    assert client.get("/admin/api/status", headers=h).json()["writes_enabled"] is True


def test_admin_status_reports_writes_enabled_false(client_readonly):
    h = {"Authorization": "Bearer admin"}
    assert client_readonly.get("/admin/api/status", headers=h).json()["writes_enabled"] is False


def test_admin_writes_require_admin_auth(client_readonly):
    """Writes gate runs after auth: no key -> 401, not 403."""
    assert client_readonly.post("/admin/api/reload").status_code == 401
    assert client_readonly.post("/admin/api/models/x", json={"provider": "p", "provider_model_id": "y"}).status_code == 401


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


def test_admin_model_stats_endpoint(client, monkeypatch):
    """Per-model stats returns config + usage + recent, filtered by routable ids."""
    from src import ledger
    from src.usage import Usage, CostEstimate
    db = client  # reuse tmp_config-backed app; point ledger at temp db
    import tempfile, os
    monkeypatch.setenv("MODEL_GATEWAY_LEDGER_PATH", str(tempfile.mkdtemp() + "/ledger.db"))
    ledger.init()
    # Record a request for claude-test under its name and an alias.
    for mid in ("claude-test", "claude-test-1"):
        ledger.record(endpoint="/v1/messages", method="POST", model=mid,
                      provider="anthropic", provider_model_id="claude-test-1",
                      status=200, latency_ms=120, is_stream=False,
                      usage=Usage(input_tokens=100, output_tokens=50, cached_read_tokens=0,
                                  cache_write_tokens=0, reasoning_tokens=0, reported=True),
                      cost=CostEstimate(0.012, True, []))
    h = {"Authorization": "Bearer admin"}
    resp = client.get("/admin/api/models/claude-test/stats?window=24h", headers=h)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"]["name"] == "claude-test"
    assert "claude-test" in body["routable_ids"]
    # Both rows matched via the routable id set.
    assert body["usage"]["requests"] == 2
    assert body["usage"]["input_tokens"] == 200
    assert body["usage"]["cost_usd"] == 0.024
    assert len(body["recent"]) == 2
    # Unknown model returns empty usage/recent, null model — not an error.
    resp2 = client.get("/admin/api/models/does-not-exist/stats", headers=h)
    assert resp2.status_code == 200
    assert resp2.json()["model"] is None
    assert resp2.json()["usage"] == {}
    assert resp2.json()["recent"] == []
    # Requires admin auth.
    assert client.get("/admin/api/models/claude-test/stats").status_code == 401
    # Reset the providers cache so later modules reload the real catalog.
    providers.reload()
