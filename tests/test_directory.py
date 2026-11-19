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
    monkeypatch.setattr(directory, "directory_dir", lambda config_path=None: tmp_path / "models-dev")
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


def test_a_first_sync_stores_its_snapshot_even_when_the_release_has_the_same_one(corrections, state_dir):
    api = _api(anthropic=[_model("m1")])
    payloads = {directory.API_URL: json.dumps(api).encode(), directory.LABS_URL: b"{}"}
    import hashlib

    bundled = _snapshot(corrections, api)
    bundled["source"]["sha256"] = hashlib.sha256(payloads[directory.API_URL]).hexdigest()
    directory.BUNDLED_SNAPSHOT.write_text(json.dumps(bundled))
    assert directory.sync(fetch=payloads.__getitem__, corrections=corrections)["changed"] is True
    assert directory.load_snapshot()[1] == "synced"


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
    assert directory.settings({}) == {"sync": "daily", "lines": {}, "models": []}
    on = {"catalog": {"directory": {"sync": "off", "lines": {"openai/gpt sol": False}, "models": ["x"]}}}
    assert directory.settings(on) == {"sync": "off", "lines": {"openai/gpt sol": False}, "models": ["x"]}
    for bad, message in (({"sync": "hourly"}, "daily or off"), ({"extra": 1}, "unknown field"),
                         ({"lines": ["a"]}, "true or false"), ({"models": "x"}, "list of model names")):
        with pytest.raises(ValueError, match=message):
            directory.settings({"catalog": {"directory": bad}})
    assert directory.provider_policy({}) == {"openrouter": {"zero_retention_only": True}}
    with pytest.raises(ValueError, match="openrouter only"):
        directory.provider_policy({"catalog": {"provider_policy": {"fireworks": {"zero_retention_only": True}}}})


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
    assert status["lines"] == [{"line": "openai/gpt sol", "current": "openai/gpt-6.2-sol", "current_name": "GPT-6.2 Sol",
                                "label": "GPT Sol", "providers": ["openai"], "followed": True, "owner": None}]
    assert status["loaded"] == [] and status["ranking"][0] == "anthropic"  # openai has no key here
    browse = client.get("/admin/api/directory/models", headers=ADMIN, params={"offered": "1"}).json()
    assert browse["total"] == 2
    assert [(m["model_id"], m["current"], m["loaded"]) for m in browse["models"]] == [
        ("gpt-6.2-sol", True, False), ("gpt-6.1-sol", False, False)]
    assert browse["models"][0]["row"]["name"] == "gpt-6.2-sol"
    assert client.get("/admin/api/directory/models", headers=ADMIN, params={"q": "emb"}).json()["models"][0]["skipped"] == "not_chat"
    assert client.get("/admin/api/directory", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_admin_directory_sync_runs_a_sync(client, monkeypatch):
    monkeypatch.setattr(directory, "sync", lambda **kwargs: {"error": None, "changed": True, "models": 3})
    assert client.post("/admin/api/directory/sync", headers=ADMIN).json() == {
        "error": None, "changed": True, "models": 3, "catalogs": "regenerated"}


# ── The directory layer (stage 2) ──────────────────────────────────────────

LABS = {
    "openai/gpt-6-sol": {"name": "GPT-6 Sol", "release_date": "2026-09-01"},
    "openai/gpt-6.2-sol": {"name": "GPT-6.2 Sol", "release_date": "2026-11-17"},
    "openai/gpt-6-luna": {"name": "GPT-6 Luna", "release_date": "2026-09-22"},
    "openai/gpt-6-cyber": {"name": "GPT-6 Cyber", "release_date": "2026-11-05"},
    "openai/gpt-4o": {"name": "GPT-4o", "release_date": "2024-05-13"},
}


def _layer_snapshot(corrections):
    def m(model_id, canonical):
        return _model(model_id, canonical_model_id=canonical)

    api = _api(
        openai=[m("gpt-6-sol", "openai/gpt-6-sol"), m("gpt-6.2-sol", "openai/gpt-6.2-sol"),
                m("gpt-6-luna", "openai/gpt-6-luna"), m("gpt-6-cyber", "openai/gpt-6-cyber"),
                m("gpt-4o", "openai/gpt-4o")],
        openrouter=[m("openai/gpt-6.2-sol", "openai/gpt-6.2-sol"), m("openai/gpt-6-sol", "openai/gpt-6-sol")],
    )
    snapshot = _snapshot(corrections, api, LABS)
    snapshot["fetched_at"] = "2026-11-19T00:00:00Z"
    return snapshot


def _ready(*providers_ready):
    return lambda profile, config: profile["provider"]["id"] in providers_ready


def test_followed_lines_load_their_current_model_on_every_ready_provider(corrections):
    found = directory.layer({}, ready=_ready("openai", "openrouter"), snapshot=_layer_snapshot(corrections),
                            corrections=corrections)
    assert [row["name"] for row in found.rows] == ["gpt-6-luna", "gpt-6.2-sol", "gpt-6.2-sol-or"]
    assert found.rows[0]["origin"] == {"layer": "shipped", "profile": "models.dev", "line": "openai/gpt luna"}
    # gpt-4o is too old to be followed by default and GPT-6 Cyber is special-purpose.
    assert found.followed == ("openai/gpt luna", "openai/gpt sol")
    assert found.retired == {"gpt-6-sol": "gpt-6.2-sol", "gpt-6-sol-or": "gpt-6.2-sol-or"}
    assert ("openai", "gpt-4o") in found.known_routes


def test_owner_choices_follow_unfollow_turn_on_keep_and_rank(corrections):
    config = {"catalog": {
        "directory": {"lines": {"openai/gpt luna": False, "openai/gpt cyber": True}, "models": ["gpt-4o"]},
        "provider_preference": ["openrouter", "openai"],
    }}
    found = directory.layer(config, ready=_ready("openai", "openrouter"), snapshot=_layer_snapshot(corrections),
                            corrections=corrections, keep={"gpt-6-sol"})
    assert [row["name"] for row in found.rows] == ["gpt-6.2-sol-or", "gpt-4o", "gpt-6-cyber", "gpt-6-sol", "gpt-6.2-sol"]
    assert found.retired == {"gpt-6-sol-or": "gpt-6.2-sol-or"}


def test_openrouter_policy_controls_zero_retention_routing(corrections):
    snapshot = _layer_snapshot(corrections)
    zdr = directory.layer({}, ready=_ready("openrouter"), snapshot=snapshot, corrections=corrections)
    assert zdr.rows[0]["quirks"] == ["openrouter_zdr"] and zdr.rows[0]["zdr"] is True
    config = {"catalog": {"provider_policy": {"openrouter": {"zero_retention_only": False}}}}
    open_routes = directory.layer(config, ready=_ready("openrouter"), snapshot=snapshot, corrections=corrections)
    assert "quirks" not in open_routes.rows[0] and "zdr" not in open_routes.rows[0]


def test_a_provider_counts_as_ready_only_at_the_expected_endpoint_with_a_key(corrections, monkeypatch):
    from src import shipped

    config = {"providers": {"openai": {"base_url": "https://api.openai.com/v1", "api_key": "k"},
                            "openrouter": {"base_url": "https://elsewhere.test/v1", "api_key": "k"},
                            "fireworks": {"base_url": "https://api.fireworks.ai/inference/v1"}}}
    for name in ("OPENAI", "OPENROUTER", "FIREWORKS"):
        monkeypatch.delenv(f"MODEL_GATEWAY_PROVIDER_{name}_API_KEY", raising=False)
    assert directory.ready_providers(config, corrections, shipped.provider_ready) == {"openai"}


def test_shipped_layer_loads_directory_rows_and_curated_rows_yield(corrections, monkeypatch, tmp_path):
    from src import onboarding, shipped

    snapshot = _layer_snapshot(corrections)
    monkeypatch.setattr(shipped, "_directory", lambda config, ready, keep, config_path=None: directory.layer(
        config, ready=_ready("openai"), keep=keep, snapshot=snapshot, corrections=corrections))
    profile = onboarding.validate_profile({
        "schema_version": 1, "id": "openai-curated", "auto_add": True,
        "provider": {"id": "openai", "base_url": "https://api.openai.com/v1", "protocol": "openai"},
        "models": [
            {"name": "gpt-6-luna", "provider": "openai", "provider_model_id": "gpt-6-luna", "context": 1000,
             "pricing": {"input": 9.0, "output": 9.0}},
            {"name": "curated-only", "provider": "openai", "provider_model_id": "not-in-directory", "context": 1000,
             "pricing": {"input": 1.0, "output": 2.0}},
        ],
    })
    found = shipped.layer({"catalog": {"pinned": []}}, profiles={"openai-curated": profile},
                          ready=lambda p, c: True, config_path=tmp_path / "config.yaml")
    names = [row["name"] for row in found.entries]
    assert names == ["gpt-6-luna", "gpt-6.2-sol", "curated-only"]
    assert found.entries[0]["pricing"]["input"] == 1.0  # the directory's row, not the curated copy
    assert found.retired == {"gpt-6-sol": "gpt-6.2-sol"}


def test_default_ranking_resolves_base_models_when_the_owner_sets_none(corrections):
    from src import shipped

    assert shipped.provider_preference({}) == list(corrections.ranking)
    assert shipped.provider_preference({"catalog": {"provider_preference": ["openrouter"]}}) == ["openrouter"]


def test_admin_directory_settings_round_trip(client):
    response = client.put("/admin/api/directory/settings", headers=ADMIN, json={
        "lines": {"openai/gpt sol": False}, "provider_preference": ["openrouter", "openai"],
        "provider_policy": {"openrouter": {"zero_retention_only": True}}})
    assert response.status_code == 200, response.text
    assert response.json()["settings"]["lines"] == {"openai/gpt sol": False}
    status = client.get("/admin/api/directory", headers=ADMIN).json()
    assert status["ranking"] == ["openrouter", "openai"]
    assert [line["followed"] for line in status["lines"]] == [False]
    restored = client.put("/admin/api/directory/settings", headers=ADMIN,
                          json={"lines": {"openai/gpt sol": None}, "provider_preference": []}).json()
    assert restored["settings"]["lines"] == {}
    stored = yaml.safe_load(providers.CONFIG_PATH.read_text())
    assert "directory" not in stored["catalog"] and "provider_preference" not in stored["catalog"]
    bad = client.put("/admin/api/directory/settings", headers=ADMIN, json={"sync": "hourly"})
    assert bad.status_code == 400 and "daily or off" in bad.text


def test_a_lower_ranked_route_whose_upstream_id_is_a_loaded_name_is_left_out(corrections):
    labs = {"moonshotai/kimi-k3": {"name": "Kimi K3", "release_date": "2026-11-01"}}
    api = _api(**{"fireworks-ai": [_model("accounts/fireworks/models/kimi-k3", canonical_model_id="moonshotai/kimi-k3")],
                  "moonshotai": [_model("kimi-k3", canonical_model_id="moonshotai/kimi-k3")]})
    snapshot = _snapshot(corrections, api, labs)
    found = directory.layer({}, ready=_ready("fireworks", "moonshot"), snapshot=snapshot, corrections=corrections)
    # Fireworks' route is named kimi-k3 (corrections); Moonshot's upstream id kimi-k3 would collide.
    assert [row["name"] for row in found.rows] == ["kimi-k3"]


def test_latest_and_preview_names_never_become_a_line_current_model(corrections):
    labs = {"anthropic/claude-q-1": {"name": "Claude Q 1", "release_date": "2026-10-01"},
            "anthropic/claude-q-latest": {"name": "Claude Q (latest)", "release_date": "2026-11-01"}}
    api = _api(anthropic=[_model("claude-q-1", canonical_model_id="anthropic/claude-q-1"),
                          _model("claude-q-latest-alias", canonical_model_id="anthropic/claude-q-latest")])
    mapped = directory.map_snapshot(_snapshot(corrections, api, labs), corrections)
    assert directory.current_models(mapped) == {"anthropic/claude q": "anthropic/claude-q-1"}


# ── Holds and smoke tests (stage 3) ────────────────────────────────────────


def _hold_snapshots(corrections, new_cost=None, drop=False):
    old = _snapshot(corrections, _api(openai=[_model("g1"), _model("g2")]))
    models = [] if drop else [_model("g1", **({"cost": new_cost} if new_cost is not None else {}))]
    new = _snapshot(corrections, _api(openai=[*models, _model("g2", cost={"input": 9.0, "output": 9.0})]))
    return old, new


@pytest.mark.parametrize("new_cost, drop, kind", [
    ({"input": 1.6, "output": 5.0}, False, "price_change"),
    ({"input": 1.0, "output": 3.0}, False, "price_change"),
    ({}, False, "price_missing"),
    (None, True, "removed"),
])
def test_suspicious_changes_to_loaded_models_are_held(corrections, new_cost, drop, kind):
    old, new = _hold_snapshots(corrections, new_cost, drop)
    holds = directory.apply_holds(new, old, {("openai", "g1")}, corrections, {})
    assert list(holds) == ["openai:g1"] and holds["openai:g1"]["kind"] == kind
    assert holds["openai:g1"]["status"] == "pending"
    assert new["providers"]["openai"]["models"]["g1"] == old["providers"]["openai"]["models"]["g1"]
    # g2 is not loaded, so its 9x price change goes through.
    assert new["providers"]["openai"]["models"]["g2"]["cost"]["input"] == 9.0


def test_small_changes_pass_and_a_rejected_change_stays_rejected_until_it_changes(corrections):
    old, new = _hold_snapshots(corrections, {"input": 1.4, "output": 6.0})
    assert directory.apply_holds(new, old, {("openai", "g1")}, corrections, {}) == {}
    old, new = _hold_snapshots(corrections, {"input": 3.0, "output": 5.0})
    first = directory.apply_holds(new, old, {("openai", "g1")}, corrections, {})
    rejected = {"openai:g1": {**first["openai:g1"], "status": "rejected"}}
    old, again = _hold_snapshots(corrections, {"input": 3.0, "output": 5.0})
    assert directory.apply_holds(again, old, {("openai", "g1")}, corrections, rejected)["openai:g1"]["status"] == "rejected"
    old, other = _hold_snapshots(corrections, {"input": 4.0, "output": 5.0})
    assert directory.apply_holds(other, old, {("openai", "g1")}, corrections, rejected)["openai:g1"]["status"] == "pending"


def test_sync_holds_changes_to_protected_routes_and_approval_applies_them(corrections, state_dir):
    payloads = {directory.API_URL: json.dumps(_api(openai=[_model("g1"), _model("g2")])).encode(),
                directory.LABS_URL: b"{}"}
    directory.sync(fetch=payloads.__getitem__, corrections=corrections)
    payloads[directory.API_URL] = json.dumps(_api(openai=[_model("g1", cost={"input": 5.0, "output": 25.0}),
                                                          _model("g2")])).encode()
    state = directory.sync(fetch=payloads.__getitem__, corrections=corrections, protect={("openai", "g1")})
    assert state["holds"]["openai:g1"]["kind"] == "price_change"
    assert directory.load_snapshot()[0]["providers"]["openai"]["models"]["g1"]["cost"]["input"] == 1.0
    assert directory.resolve_hold("openai:g1", approve=True)["status"] == "approved"
    assert directory.load_snapshot()[0]["providers"]["openai"]["models"]["g1"]["cost"]["input"] == 5.0
    assert directory.load_state()["holds"] == {}
    with pytest.raises(KeyError):
        directory.resolve_hold("openai:g1", approve=False)


def test_smoke_records_decide_retests_and_failures_make_a_route_unavailable(corrections, state_dir, monkeypatch):
    row = {"name": "gpt-6.2-sol", "provider": "openai", "provider_model_id": "gpt-6.2-sol"}
    assert directory.needs_smoke_test(row, {}, retest_failed=False)
    assert directory.record_smoke({"gpt-6.2-sol": ("openai:gpt-6.2-sol", "provider refused (HTTP 403)")})
    smoke = directory.load_state()["smoke"]
    assert not directory.needs_smoke_test(row, smoke, retest_failed=False)
    assert directory.needs_smoke_test(row, smoke, retest_failed=True)
    assert not directory.record_smoke({"gpt-6.2-sol": ("openai:gpt-6.2-sol", "still refused")})

    found = directory.layer({}, ready=_ready("openai"), snapshot=_layer_snapshot(corrections), corrections=corrections)
    by_name = {r["name"]: r for r in found.rows}
    assert by_name["gpt-6.2-sol"]["origin"]["smoke_error"] == "still refused"
    availability = providers._availability_for_entry("gpt-6.2-sol", by_name["gpt-6.2-sol"])
    assert (availability["available"], availability["reason"]) == (False, "smoke_test_failed")
    assert "smoke_error" not in by_name["gpt-6-luna"]["origin"]


def test_smoke_test_reports_the_provider_error(monkeypatch):
    from src import onboarding

    seen = {}

    def refuse(profile, key):
        seen.update(profile=profile, key=key)
        raise onboarding.OnboardingError("provider refused a test completion (HTTP 404): no such model")

    monkeypatch.setattr(onboarding, "validate_upstream_completion", refuse)
    row = {"name": "x", "provider": "openrouter", "provider_model_id": "a/x", "quirks": ["openrouter_zdr"]}
    error = directory.smoke_test(row, {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"})
    assert error.endswith("no such model")
    assert seen["profile"]["models"] == [row] and seen["key"] == "k"
    monkeypatch.setattr(onboarding, "validate_upstream_completion", lambda profile, key: None)
    assert directory.smoke_test(row, {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"}) is None


def test_admin_holds_endpoint(client, monkeypatch):
    calls = []

    def resolve(hold_id, approve):
        calls.append((hold_id, approve))
        if hold_id == "missing":
            raise KeyError(hold_id)
        return {"id": hold_id, "status": "approved" if approve else "rejected", "fingerprint": "x"}

    monkeypatch.setattr(directory, "resolve_hold", resolve)
    ok = client.post("/admin/api/directory/holds/openrouter:google/gemini-x", headers=ADMIN, json={"approve": False})
    assert ok.status_code == 200 and ok.json()["hold"] == {"id": "openrouter:google/gemini-x", "status": "rejected"}
    assert client.post("/admin/api/directory/holds/missing", headers=ADMIN, json={"approve": True}).status_code == 404
    assert client.post("/admin/api/directory/holds/x", headers=ADMIN, json={}).status_code == 400
    assert calls == [("openrouter:google/gemini-x", False), ("missing", True)]


def test_a_base_named_route_that_failed_its_test_falls_back_to_the_next_ranked_route(monkeypatch):
    entries = {
        "kimi-k3.1-pro": {"name": "kimi-k3.1-pro", "provider": "moonshot", "base_model": "kimi-k3.1-pro",
                          "origin": {"layer": "shipped", "profile": "models.dev", "smoke_error": "suspended"}},
        "kimi-k3.1-pro-fw": {"name": "kimi-k3.1-pro-fw", "provider": "fireworks", "base_model": "kimi-k3.1-pro",
                             "origin": {"layer": "shipped", "profile": "models.dev"}},
    }
    monkeypatch.setattr(providers, "_load_models", lambda **kwargs: entries)
    monkeypatch.setattr(providers, "resolve_base_model",
                        lambda base, locality=None: "kimi-k3.1-pro-fw" if base == "kimi-k3.1-pro" else None)
    assert providers.base_model_route("kimi-k3.1-pro") == "kimi-k3.1-pro-fw"


@pytest.mark.parametrize("name, label", [
    ("GPT-6.2 Sol", "GPT Sol"), ("Claude Opus 5.5", "Claude Opus"), ("Kimi K3.1 Pro", "Kimi Pro"),
    ("GLM-5.3-Flash", "GLM-Flash"), ("DeepSeek V4.1 Flash", "DeepSeek Flash"), ("Gemini 4 Neon", "Gemini Neon"),
])
def test_line_labels_drop_versions(name, label):
    assert directory.line_label(name) == label
