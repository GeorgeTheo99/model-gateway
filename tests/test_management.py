"""Tests for writeable provider/model management (milestones 4 & 5)."""

import json
import threading
import time

import pytest

from src import admin, config_io
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


def test_provider_write_lock_covers_full_read_modify_write(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    original_load = config_io.load_config_full
    active = 0
    max_active = 0
    guard = threading.Lock()

    def slow_load():
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        try:
            return original_load()
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(config_io, "load_config_full", slow_load)
    threads = [
        threading.Thread(target=config_io.upsert_provider, args=(name,), kwargs={
            "base_url": f"https://{name}.example.com/v1", "api_key": "secret",
        })
        for name in ("one", "two")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert all(not thread.is_alive() for thread in threads)
    assert max_active == 1
    import yaml
    providers_config = yaml.safe_load((tmp_config / "config.yaml").read_text())["providers"]
    assert {"one", "two"}.issubset(providers_config)


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


def test_upsert_local_model_can_be_explicitly_unmetered(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    result = config_io.upsert_model(
        "local-unmetered", provider="omlx", provider_model_id="local-upstream",
        pricing_status="unmetered", pricing=None,
    )
    assert result["entry"]["pricing_status"] == "unmetered"
    assert "pricing" not in result["entry"]
    providers.reload()
    status = next(row for row in providers.model_status() if row["name"] == "local-unmetered")
    assert status["pricing_status"] == "unmetered"
    assert status["pricing"] is None


def test_upsert_model_rejects_invalid_pricing_policy(tmp_config, monkeypatch):
    config_io.log_dir = tmp_config / "logs"
    with pytest.raises(ValueError, match="only valid for local"):
        config_io.upsert_model(
            "cloud-free", provider="openai", provider_model_id="cloud-free",
            pricing_status="unmetered", pricing=None,
        )
    with pytest.raises(ValueError, match="requires: output"):
        config_io.upsert_model(
            "bad-price", provider="openai", provider_model_id="bad-price",
            pricing_status="metered", pricing={"input": 1.0},
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        config_io.upsert_model(
            "infinite-price", provider="openai", provider_model_id="infinite-price",
            pricing_status="metered", pricing={"input": float("inf"), "output": 1.0},
        )
    config_io.upsert_model(
        "local-to-cloud", provider="omlx", provider_model_id="local-upstream",
        pricing_status="unmetered", pricing=None,
    )
    with pytest.raises(ValueError, match="only valid for local"):
        config_io.upsert_model(
            "local-to-cloud", provider="openai", provider_model_id="cloud-upstream",
        )


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


def test_effective_inventory_has_one_row_with_all_routable_ids(tmp_config):
    config_io.upsert_model(
        "inventory-model", provider="anthropic", provider_model_id="inventory-upstream",
        alias="inventory-alias", vision=True,
    )
    providers.reload()
    rows = [m for m in providers.effective_model_inventory() if m["name"] == "inventory-model"]
    assert len(rows) == 1
    assert rows[0]["vision"] is True
    assert set(rows[0]["routable_ids"]) >= {"inventory-model", "inventory-alias", "inventory-upstream"}
    assert providers.resolve("inventory-alias").vision is True
    discovered = [m["id"] for m in providers.list_models() if m["name"] == "inventory-model"]
    assert set(discovered) >= {"inventory-model", "inventory-alias", "inventory-upstream"}


def test_provider_status_counts_unique_models_not_identifiers(tmp_config, monkeypatch):
    """A model with name + alias + provider_model_id + omlx_id must count as 1."""
    doc = json.loads((tmp_config / "model-info.json").read_text())
    doc["llm"].append({
        "name": "local-multi", "alias": "lm", "omlx_id": "lm-up",
        "provider_model_id": "lm-pmid", "alternate_ids": ["legacy/lm"],
        "context": 4096, "max_output_tokens": 1024,
    })
    (tmp_config / "model-info.json").write_text(json.dumps(doc))
    providers.reload()
    counts = {p["id"]: p["enabled_models"] for p in providers.provider_status()}
    # omlx has exactly one unique local model (local-multi); claude-test is anthropic.
    assert counts.get("omlx") == 1
    assert counts.get("anthropic") == 1
    # routable_ids exposes canonical identifiers plus legacy alternate IDs.
    assert set(providers.routable_ids("local-multi")) == {"local-multi", "lm", "lm-up", "lm-pmid", "legacy/lm"}
    assert providers.resolve("legacy/lm") is not None


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
        public_rows = c.get("/v1/models").json()["data"]
        ids = {m["id"] for m in public_rows}
        assert "claude-test" in ids
        assert "gpt-unconfigured" not in ids
        assert all("pricing" not in m and "pricing_status" not in m for m in public_rows)

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
    # Isolate from ambient Databricks env (dev shells often export these).
    for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_SERVING_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
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


def test_admin_reload_rejects_invalid_vision_fallback_and_restores_registry(
    client, tmp_config, monkeypatch,
):
    model_info_path = tmp_config / "model-info.json"
    valid_catalog = {"llm": [{
        "name": "vision-fallback",
        "provider": "omlx",
        "omlx_id": "vision-upstream",
        "vision": True,
        "context": 4096,
        "max_output_tokens": 512,
    }]}
    model_info_path.write_text(json.dumps(valid_catalog))
    providers.reload()
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "vision-fallback")
    assert providers.resolve("vision-fallback").vision is True

    invalid_catalog = json.loads(json.dumps(valid_catalog))
    invalid_catalog["llm"][0]["vision"] = False
    model_info_path.write_text(json.dumps(invalid_catalog))

    response = client.post(
        "/admin/api/reload",
        headers={"Authorization": "Bearer admin"},
    )

    assert response.status_code == 400
    assert "not vision-capable" in response.text
    restored = providers.resolve("vision-fallback")
    assert restored is not None
    assert restored.vision is True


def test_admin_reload_malformed_catalog_restores_live_registry(
    client, tmp_config,
):
    assert providers.resolve("claude-test") is not None
    (tmp_config / "model-info.json").write_text("{ malformed")

    response = client.post(
        "/admin/api/reload",
        headers={"Authorization": "Bearer admin"},
    )

    assert response.status_code == 400
    assert "reload rejected" in response.text
    assert providers.resolve("claude-test") is not None


def test_admin_mutations_rollback_when_they_invalidate_vision_fallback(
    client, tmp_config, monkeypatch,
):
    import yaml

    model_info_path = tmp_config / "model-info.json"
    valid_catalog = {"llm": [{
        "name": "vision-fallback",
        "provider": "omlx",
        "omlx_id": "vision-upstream",
        "vision": True,
        "context": 4096,
        "max_output_tokens": 512,
    }]}
    model_info_path.write_text(json.dumps(valid_catalog))
    config_path = tmp_config / "config.yaml"
    model_info_path.chmod(0o600)
    config_path.chmod(0o600)
    providers.reload()
    monkeypatch.setenv("GATEWAY_VISION_FALLBACK", "vision-fallback")
    assert providers.resolve("vision-fallback").vision is True
    headers = {"Authorization": "Bearer admin"}

    update_response = client.post(
        "/admin/api/models/vision-fallback",
        headers=headers,
        json={
            "provider": "omlx",
            "omlx_id": "vision-upstream",
            "vision": False,
        },
    )

    assert update_response.status_code == 400
    assert "changes rolled back" in update_response.text
    restored_catalog = json.loads(model_info_path.read_text())
    assert restored_catalog["llm"][0]["vision"] is True
    assert model_info_path.stat().st_mode & 0o777 == 0o600
    assert providers.resolve("vision-fallback").vision is True

    disable_response = client.post(
        "/admin/api/models/vision-fallback/disable",
        headers=headers,
    )

    assert disable_response.status_code == 400
    assert "changes rolled back" in disable_response.text
    config = yaml.safe_load(config_path.read_text())
    assert "vision-fallback" not in (config.get("model_overrides") or {})
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert providers.resolve("vision-fallback").vision is True


def test_registry_mutations_are_serialized_across_validation(tmp_config, monkeypatch):
    monkeypatch.delenv("GATEWAY_VISION_FALLBACK", raising=False)
    a_written = threading.Event()
    release_a = threading.Event()
    b_started = threading.Event()
    b_entered = threading.Event()
    errors = []

    def mutate_a():
        result = config_io.upsert_provider(
            "provider-a", base_url="https://a.example.test/v1", api_key="a",
        )
        a_written.set()
        assert release_a.wait(timeout=2)
        return result

    def mutate_b():
        b_entered.set()
        return config_io.upsert_provider(
            "provider-b", base_url="https://b.example.test/v1", api_key="b",
        )

    def run(mutate, started=None):
        if started is not None:
            started.set()
        try:
            result, reload_error = admin._apply_registry_mutation(mutate)
            assert reload_error is None
            assert result is not None
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread_a = threading.Thread(target=run, args=(mutate_a,))
    thread_b = threading.Thread(target=run, args=(mutate_b, b_started))
    thread_a.start()
    assert a_written.wait(timeout=2)
    thread_b.start()
    assert b_started.wait(timeout=2)
    assert not b_entered.wait(timeout=0.05)
    release_a.set()
    thread_a.join(timeout=2)
    thread_b.join(timeout=2)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert errors == []
    config = config_io.load_config_full()
    assert {"provider-a", "provider-b"}.issubset(config["providers"])


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


def test_admin_upsert_model_applies_enabled_checkbox(client):
    h = {"Authorization": "Bearer admin"}
    payload = {
        "provider": "anthropic",
        "provider_model_id": "claude-test-1",
        "enabled": False,
    }
    response = client.post("/admin/api/models/claude-test", headers=h, json=payload)
    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert providers.resolve("claude-test") is None

    payload["enabled"] = True
    response = client.post("/admin/api/models/claude-test", headers=h, json=payload)
    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert providers.resolve("claude-test") is not None


def test_admin_disable_enable_model_endpoint(client):
    h = {"Authorization": "Bearer admin"}
    assert client.post("/admin/api/models/claude-test/disable", headers=h).status_code == 200
    assert providers.resolve("claude-test") is None
    assert client.post("/admin/api/models/claude-test/enable", headers=h).status_code == 200
    assert providers.resolve("claude-test") is not None


def test_admin_delete_model_endpoint(client):
    h = {"Authorization": "Bearer admin"}
    created = client.post(
        "/admin/api/models/replacement",
        headers=h,
        json={"provider": "anthropic", "provider_model_id": "replacement-1"},
    )
    assert created.status_code == 200
    assert client.delete("/admin/api/models/claude-test", headers=h).status_code == 200
    assert providers.resolve("claude-test") is None
    assert providers.resolve("replacement") is not None


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
