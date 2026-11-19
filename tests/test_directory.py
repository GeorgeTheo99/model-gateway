"""The models.dev directory: snapshot, validation, sync and mapping (ADR 0005)."""

from __future__ import annotations

import json

import pytest
import yaml
from fastapi.testclient import TestClient

import src.providers as providers
from src import catalog, config_io, directory
from src.server import app

ADMIN = {"Authorization": "Bearer admin"}


def _model(model_id, **fields):
    model = {
        "id": model_id, "name": model_id.title(), "reasoning": False, "tool_call": True,
        "modalities": {"input": ["text", "image"], "output": ["text"]},
        "limit": {"context": 200_000, "output": 32_000},
        "cost": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25},
        "release_date": "2026-10-01",
    }
    model.update(fields)
    return model


def _api(**providers_models):
    return {dev_id: {"id": dev_id, "name": dev_id, "models": {m["id"]: m for m in models}}
            for dev_id, models in providers_models.items()}


@pytest.fixture
def corrections():
    return directory.load_corrections()


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(directory, "directory_dir", lambda: tmp_path / "models-dev")
    monkeypatch.setattr(directory, "BUNDLED_SNAPSHOT", tmp_path / "bundled.json")
    return tmp_path / "models-dev"


def _snapshot(corrections, api, labs=None):
    return directory.build_snapshot(api, labs or {}, corrections, fetched_at="2026-11-19T00:00:00Z")


def _map(corrections, dev_provider, model, labs=None):
    snapshot = _snapshot(corrections, _api(**{dev_provider: [model]}), labs)
    return directory.map_model(dev_provider, snapshot["providers"][dev_provider]["models"][model["id"]],
                               snapshot, corrections)


def test_snapshot_keeps_mapped_providers_and_the_lab_names_they_reference(corrections):
    api = _api(anthropic=[_model("claude-x-1", canonical_model_id="anthropic/claude-x-1", extra="dropped")],
               someone=[_model("other")])
    labs = {"anthropic/claude-x-1": {"name": "Claude X 1", "release_date": "2026-10-01", "noise": 1},
            "unrelated/model": {"name": "Unrelated"}}
    snapshot = _snapshot(corrections, api, labs)
    assert set(snapshot["providers"]) == {"anthropic"}
    assert "extra" not in snapshot["providers"]["anthropic"]["models"]["claude-x-1"]
    assert snapshot["labs"] == {"anthropic/claude-x-1": {"name": "Claude X 1", "release_date": "2026-10-01"}}


def test_update_validation_refuses_lost_providers_and_mass_disappearance(corrections):
    old = _snapshot(corrections, _api(anthropic=[_model(f"m{i}") for i in range(10)], openai=[_model("g")]))
    directory.validate_update(_snapshot(corrections, _api(anthropic=[_model(f"m{i}") for i in range(9)],
                                                          openai=[_model("g")])), old)
    with pytest.raises(directory.DirectoryError, match="no models for 'openai'"):
        directory.validate_update(_snapshot(corrections, _api(anthropic=[_model("m0")])), old)
    with pytest.raises(directory.DirectoryError, match="would disappear"):
        directory.validate_update(_snapshot(corrections, _api(anthropic=[_model("m0")], openai=[_model("g")])), old)
    with pytest.raises(directory.DirectoryError, match="no providers"):
        directory.validate_update(_snapshot(corrections, {}), old)


def test_thinking_follows_reasoning_options(corrections):
    toggle = _map(corrections, "anthropic", _model("a", reasoning=True, reasoning_options=[
        {"type": "toggle"}, {"type": "effort", "values": ["low", "medium", "high", "xhigh", "max"]}]))
    assert (toggle.row["thinking"], toggle.row["thinking_levels"]) == ("optional", ["off", "low", "medium", "high", "xhigh", "max"])
    effort_only = _map(corrections, "openai", _model("b", reasoning=True, reasoning_options=[
        {"type": "effort", "values": ["low", "high"]}]))
    assert (effort_only.row["thinking"], effort_only.row["thinking_levels"]) == ("always", ["low", "high"])
    with_none = _map(corrections, "openai", _model("c", reasoning=True, reasoning_options=[
        {"type": "effort", "values": ["none", "low"]}]))
    assert (with_none.row["thinking"], with_none.row["thinking_levels"]) == ("optional", ["off", "low"])
    plain = _map(corrections, "openai", _model("d"))
    assert "thinking" not in plain.row and "thinking_levels" not in plain.row


def test_prices_tiers_and_extended_cache_writes(corrections):
    model = _model("claude-y-2", canonical_model_id="anthropic/claude-y-2", cost={
        "input": 0.1, "output": 0.5, "cache_read": 0.01, "cache_write": 0.125, "reasoning": 0.5,
        "tiers": [{"input": 0.5, "output": 2.5, "cache_read": 0.05, "cache_write": 0.625,
                   "tier": {"type": "context", "size": 100_000}}]})
    row = _map(corrections, "anthropic", model).row
    assert row["pricing"] == {
        "input": 0.1, "output": 0.5, "cache_read": 0.01, "cache_write": 0.125, "cache_write_1h": 0.2,
        "long_context": {"above_prompt_tokens": 100_000, "input": 0.5, "output": 2.5, "cache_read": 0.05,
                         "cache_write": 0.625, "cache_write_1h": 1.0},
    }
    assert catalog.pricing_problem(row["pricing"]) is None


def test_routes_are_named_after_the_lab_model_with_provider_suffixes(corrections):
    labs = {"anthropic/claude-z-5-5": {"name": "Claude Z 5.5"}, "google/gemini-9-x": {"name": "Gemini 9 X"}}
    direct = _map(corrections, "anthropic", _model("claude-z-5-5", canonical_model_id="anthropic/claude-z-5-5"), labs)
    via_or = _map(corrections, "openrouter",
                  _model("anthropic/claude-z-5.5", canonical_model_id="anthropic/claude-z-5-5"), labs)
    gemini = _map(corrections, "openrouter", _model("google/gemini-9-x", canonical_model_id="google/gemini-9-x",
                                                    limit={"context": 1_048_576, "output": 943_718}), labs)
    assert (direct.row["name"], direct.row["base_model"], direct.row["display_name"]) == (
        "claude-z-5.5", "claude-z-5.5", "Claude Z 5.5 (Anthropic)")
    assert (via_or.row["name"], via_or.row["quirks"], via_or.row["zdr"]) == ("claude-z-5.5-or", ["openrouter_zdr"], True)
    # Gemini keeps bare names on OpenRouter's Vertex routes, capped at Vertex's 64K output, without
    # OpenRouter's Google cache_write.
    assert (gemini.row["name"], gemini.row["max_output_tokens"]) == ("gemini-9-x", 65536)
    assert "cache_write" not in gemini.row["pricing"]
    assert direct.line == "anthropic/claude z" and gemini.line == "google/gemini x"


def test_models_that_cannot_be_offered_say_why(corrections):
    assert _map(corrections, "openai", _model("old", status="deprecated")).skipped == "deprecated"
    assert _map(corrections, "openai", _model("free", cost={})).skipped == "no_price"
    assert _map(corrections, "openai", _model("emb", type="embedding")).skipped == "not_chat"
    image = _model("img", modalities={"input": ["text"], "output": ["image"]})
    assert _map(corrections, "openai", image).skipped == "no_text_output"


def test_base_names_fall_back_to_the_lab_id_when_names_collide(corrections):
    api = _api(openai=[_model("x-1", canonical_model_id="openai/x-1"), _model("x-1-0101", canonical_model_id="openai/x-1-0101")])
    labs = {"openai/x-1": {"name": "X 1"}, "openai/x-1-0101": {"name": "X 1"}}
    assert directory.base_names(_snapshot(corrections, api, labs)) == {"openai/x-1": "x-1", "openai/x-1-0101": "x-1-0101"}


def test_one_provider_serving_a_model_twice_keeps_the_plain_id_on_the_name(corrections):
    api = _api(openai=[_model("x-1", canonical_model_id="openai/x-1"),
                       _model("x-latest", canonical_model_id="openai/x-1")])
    mapped = directory.map_snapshot(_snapshot(corrections, api, {"openai/x-1": {"name": "X 1"}}), corrections)
    assert {m.model_id: m.row["name"] for m in mapped} == {"x-1": "x-1", "x-latest": "x-latest"}


def test_a_line_follows_its_newest_released_model(corrections):
    labs = {f"openai/gpt-{v}-sol": {"name": f"GPT-{v} Sol", "release_date": d}
            for v, d in (("6", "2026-09-01"), ("6.1", "2026-10-01"), ("6.2", "2026-11-17"), ("7", "2026-12-01"))}
    api = _api(openai=[
        _model("gpt-6-sol", canonical_model_id="openai/gpt-6-sol"),
        _model("gpt-6.1-sol", canonical_model_id="openai/gpt-6.1-sol"),
        _model("gpt-6.2-sol", canonical_model_id="openai/gpt-6.2-sol"),
        _model("gpt-7-sol-preview", canonical_model_id="openai/gpt-7-sol", status="beta"),
    ])
    mapped = directory.map_snapshot(_snapshot(corrections, api, labs), corrections)
    assert directory.current_models(mapped) == {"openai/gpt sol": "openai/gpt-6.2-sol"}


def test_sync_stores_a_valid_snapshot_and_keeps_the_last_good_one_on_failure(corrections, state_dir):
    api = _api(anthropic=[_model("m1"), _model("m2")])
    payloads = {directory.API_URL: json.dumps(api).encode(), directory.LABS_URL: b"{}"}
    state = directory.sync(fetch=payloads.__getitem__, corrections=corrections)
    assert state["error"] is None and state["changed"] is True and state["models"] == 2
    snapshot, origin = directory.load_snapshot()
    assert origin == "synced" and set(snapshot["providers"]["anthropic"]["models"]) == {"m1", "m2"}

    assert directory.sync(fetch=payloads.__getitem__, corrections=corrections)["changed"] is False

    payloads[directory.API_URL] = b"not json"
    state = directory.sync(fetch=payloads.__getitem__, corrections=corrections)
    assert state["error"].startswith("DirectoryError")
    assert directory.load_snapshot()[0]["providers"]["anthropic"]["models"].keys() == {"m1", "m2"}


def test_without_a_sync_the_release_snapshot_is_used(corrections, state_dir):
    assert directory.load_snapshot() == (None, "none")
    bundled = _snapshot(corrections, _api(openai=[_model("g")]))
    directory.BUNDLED_SNAPSHOT.write_text(json.dumps(bundled))
    assert directory.load_snapshot() == (bundled, "bundled")


def test_sync_is_due_daily():
    from datetime import datetime, timezone

    then = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()
    assert directory.sync_due({})
    assert not directory.sync_due({"last_attempt": "2026-11-19T00:00:00Z"}, now=then + 3600)
    assert directory.sync_due({"last_attempt": "2026-11-19T00:00:00Z"}, now=then + 86_400)


def test_directory_settings_validate():
    assert directory.settings({}) == {"sync": "daily"}
    assert directory.settings({"catalog": {"directory": {"sync": "off"}}}) == {"sync": "off"}
    with pytest.raises(ValueError, match="daily or off"):
        directory.settings({"catalog": {"directory": {"sync": "hourly"}}})
    with pytest.raises(ValueError, match="unknown field"):
        directory.settings({"catalog": {"directory": {"lines": []}}})


def test_every_offered_model_in_the_release_snapshot_is_a_valid_catalog_row(corrections):
    snapshot = json.loads(directory.BUNDLED_SNAPSHOT.read_text())
    mapped = directory.map_snapshot(snapshot, corrections)
    offered = [item.row for item in mapped if item.row is not None]
    assert len(offered) > 100
    names: dict[str, tuple] = {}
    for row in offered:
        catalog.validate_pricing_policy(row)
        catalog.validate_client_metadata(row)
        catalog.normalize_thinking_capabilities(row)
        key = (row["provider"], row["provider_model_id"])
        assert names.setdefault(row["name"], key) == key, f"two routes named {row['name']}"


def test_corrections_keep_established_route_identities(corrections):
    snapshot = json.loads(directory.BUNDLED_SNAPSHOT.read_text())
    rows = {(item.provider, item.model_id): item.row for item in directory.map_snapshot(snapshot, corrections)}
    expected = {
        ("anthropic", "claude-opus-5-5"): "claude-opus-5.5",
        ("openrouter", "anthropic/claude-opus-5.5"): "claude-opus-5.5-or",
        ("openai", "gpt-6.1-sol"): "gpt-6.1-sol",
        ("openrouter", "google/gemini-4-argon"): "gemini-4-argon",
        ("fireworks", "accounts/fireworks/models/glm-5p3"): "glm-5.3-fw",
        ("fireworks", "accounts/fireworks/models/kimi-k3"): "kimi-k3",
        ("moonshot", "kimi-k3.1-pro"): "kimi-k3.1-pro",
        ("openrouter", "z-ai/glm-5.3-flash"): "glm-5.3-flash-or",
    }
    assert {key: rows[key]["name"] for key in expected} == expected
    assert rows[("fireworks", "accounts/fireworks/models/glm-5p3")]["alias"] == "glm53fw"
    assert rows[("moonshot", "kimi-k3.1-pro")]["pricing"]["cache_write_1h"] == 4.0


@pytest.fixture
def client(tmp_path, monkeypatch, state_dir, corrections):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"auth": {"admin_keys": ["admin"]}}))
    path.chmod(0o600)
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"allow_empty": True, "llm": []}))
    for module in (providers, config_io):
        monkeypatch.setattr(module, "CONFIG_PATH", path)
        monkeypatch.setattr(module, "MODEL_INFO_PATH", info)
        monkeypatch.setattr(module, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    providers.reload()
    labs = {"openai/gpt-6.2-sol": {"name": "GPT-6.2 Sol", "release_date": "2026-11-17"},
            "openai/gpt-6.1-sol": {"name": "GPT-6.1 Sol", "release_date": "2026-09-29"}}
    bundled = _snapshot(corrections, _api(openai=[
        _model("gpt-6.2-sol", canonical_model_id="openai/gpt-6.2-sol"),
        _model("gpt-6.1-sol", canonical_model_id="openai/gpt-6.1-sol"),
        _model("emb", type="embedding"),
    ]), labs)
    directory.BUNDLED_SNAPSHOT.write_text(json.dumps(bundled))
    with TestClient(app) as c:
        yield c


def test_admin_directory_status_and_browse(client):
    status = client.get("/admin/api/directory", headers=ADMIN).json()
    assert status["snapshot"] == {"origin": "bundled", "fetched_at": "2026-11-19T00:00:00Z", "models": 3, "offered": 2}
    assert status["lines"] == [{"line": "openai/gpt sol", "current": "openai/gpt-6.2-sol", "providers": ["openai"]}]
    browse = client.get("/admin/api/directory/models", headers=ADMIN, params={"offered": "1"}).json()
    assert browse["total"] == 2
    assert [(m["model_id"], m["current"], m["loaded"]) for m in browse["models"]] == [
        ("gpt-6.2-sol", True, False), ("gpt-6.1-sol", False, False)]
    assert browse["models"][0]["row"]["name"] == "gpt-6.2-sol"
    assert client.get("/admin/api/directory/models", headers=ADMIN, params={"q": "emb"}).json()["models"][0]["skipped"] == "not_chat"
    assert client.get("/admin/api/directory", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_admin_directory_sync_runs_a_sync(client, monkeypatch):
    monkeypatch.setattr(directory, "sync", lambda: {"error": None, "changed": True, "models": 3})
    assert client.post("/admin/api/directory/sync", headers=ADMIN).json() == {"error": None, "changed": True, "models": 3}
