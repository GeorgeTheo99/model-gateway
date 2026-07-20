"""Tests for src.catalog — the shared catalog merge.

Validates that the generator and the router share one merge, that overlay wins
on id clash, that the synonym table stays in sync with src.providers, and that
GGUF/llama.cpp entries are skipped by default.
"""

from __future__ import annotations

import json

import pytest

from src import catalog


def _write_model_info(path, llm, **extra):
    doc = {"llm": llm}
    doc.update(extra)
    path.write_text(json.dumps(doc))


def test_catalog_rejects_malformed_model_entries_and_overlays(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, ["not-a-model"])
    with pytest.raises(ValueError, match="entries must be objects"):
        catalog.load_catalog_entries(mi)

    mi.write_text(json.dumps({"llm": {}}))
    with pytest.raises(ValueError, match="llm must be a list"):
        catalog.load_catalog_entries(mi)

    _write_model_info(mi, [{}])
    with pytest.raises(ValueError, match="routable identifier"):
        catalog.load_catalog_entries(mi)

    _write_model_info(mi, [{"name": {"bad": "id"}, "provider": "openai"}])
    with pytest.raises(ValueError, match="name must be a string"):
        catalog.load_catalog_entries(mi)

    _write_model_info(mi, [{"name": "bad-alternates", "provider": "openai", "alternate_ids": ["ok", 7]}])
    with pytest.raises(ValueError, match="alternate_ids must contain only strings"):
        catalog.load_catalog_entries(mi)

    _write_model_info(mi, [])
    with pytest.raises(ValueError, match="effective model catalog is empty"):
        catalog.load_catalog_entries(mi, require_nonempty=True)

    _write_model_info(mi, [{"name": "valid", "provider": "openai"}])
    cfg = tmp_path / "config.yaml"
    cfg.write_text("models: not-a-list\n")
    with pytest.raises(ValueError, match="overlay must be a list"):
        catalog.load_catalog_entries(mi, cfg)
    cfg.write_text("models: {}\n")
    with pytest.raises(ValueError, match="overlay must be a list"):
        catalog.load_catalog_entries(mi, cfg)
    with pytest.raises(ValueError, match="overlay entries must be objects"):
        catalog.load_catalog_entries(mi, overlay=["not-a-model"])


def test_overlay_empty_when_no_config(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [{"name": "a", "alias": "a", "provider": "openrouter"}])
    entries = catalog.load_catalog_entries(mi, tmp_path / "missing.yaml")
    assert [e["name"] for e in entries] == ["a"]


def test_base_catalog_rejects_duplicate_routable_ids(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [
        {"name": "first", "alias": "shared", "provider": "openai"},
        {"name": "second", "provider_model_id": "shared", "provider": "openai"},
    ])
    with pytest.raises(ValueError, match="collides between 'first' and 'second'"):
        catalog.load_catalog_entries(mi)


def test_merge_overlay_wins_on_name_clash(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [{"name": "glm-5.2", "alias": "glm52", "provider": "zai", "context": 1000}])
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n  - name: glm-5.2\n    alias: glm52fw\n    provider: fireworks\n    context: 2000\n"
    )
    entries = catalog.load_catalog_entries(mi, cfg)
    # Overlay wins: single entry, overlay fields.
    assert len(entries) == 1
    assert entries[0]["name"] == "glm-5.2"
    assert entries[0]["alias"] == "glm52fw"
    assert entries[0]["provider"] == "fireworks"
    assert entries[0]["context"] == 2000


def test_merge_overlay_inherits_omitted_capabilities(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(
        mi,
        [{"name": "gpt-vision", "alias": "gv", "provider": "openai", "vision": True, "context": 1000}],
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text("models:\n  - name: gpt-vision\n    provider: databricks\n    context: 2000\n")
    entry = catalog.load_catalog_entries(mi, cfg)[0]
    assert entry["provider"] == "databricks"
    assert entry["context"] == 2000
    assert entry["vision"] is True
    assert entry["alias"] == "gv"


def test_merge_overlay_evicts_all_ids_of_replaced_entry(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(
        mi,
        [{"name": "glm-5.2", "alias": "glm52", "provider_model_id": "glm-5.2-zai", "provider": "zai"}],
    )
    cfg = tmp_path / "config.yaml"
    # Overlay claims only the name; the catalog entry's alias/provider_model_id
    # must be evicted so the old alias doesn't linger pointing at the old entry.
    cfg.write_text("models:\n  - name: glm-5.2\n    alias: glm52fw\n    provider: fireworks\n")
    entries = catalog.load_catalog_entries(mi, cfg)
    assert len(entries) == 1
    assert entries[0]["alias"] == "glm52fw"
    # The old alias must not produce a second entry.
    names = [e["name"] for e in entries]
    assert names == ["glm-5.2"]


def test_overlay_same_name_inherits_when_other_ids_collide(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(
        mi,
        [
            {"name": "new-model", "alias": "new", "provider": "openai", "vision": True},
            {"name": "old-model", "alias": "old", "alternate_ids": ["shared"], "provider": "openai"},
        ],
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n  - name: new-model\n    alias: shared\n    provider: databricks\n"
    )
    entries = catalog.load_catalog_entries(mi, cfg)
    assert [e["name"] for e in entries] == ["new-model"]
    assert entries[0]["vision"] is True
    assert entries[0]["alias"] == "shared"


def test_overlay_ambiguous_multi_collision_without_name_match_fails(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(
        mi,
        [
            {"name": "first", "alias": "first-alias", "provider": "openai", "vision": True},
            {"name": "second", "provider_model_id": "second-upstream", "provider": "openai"},
        ],
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n  - name: replacement\n    alias: first-alias\n"
        "    provider_model_id: second-upstream\n    provider: databricks\n"
    )
    with pytest.raises(ValueError, match="ambiguously collides"):
        catalog.load_catalog_entries(mi, cfg)


def test_overlay_only_entry_added(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [{"name": "a", "alias": "a", "provider": "openrouter"}])
    cfg = tmp_path / "config.yaml"
    cfg.write_text("models:\n  - name: b\n    alias: b\n    provider: openrouter\n")
    entries = catalog.load_catalog_entries(mi, cfg)
    assert [e["name"] for e in entries] == ["a", "b"]


def test_retired_gguf_skipped_by_default(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(
        mi,
        [
            {"name": "gguf-one", "alias": "ggufone", "provider": "gguf"},
            {"name": "mlx-one", "alias": "mlxone", "provider": "local"},
        ],
    )
    entries = catalog.load_catalog_entries(mi)
    assert [e["name"] for e in entries] == ["mlx-one"]
    all_entries = catalog.load_catalog_entries(mi, include_retired=True)
    assert [e["name"] for e in all_entries] == ["gguf-one", "mlx-one"]


def test_provider_canonicalization(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(
        mi,
        [
            {"name": "local-one", "alias": "lone", "provider": "local"},
            {"name": "claude-one", "alias": "cone", "provider": "claude"},
        ],
    )
    entries = {e["name"]: e for e in catalog.load_catalog_entries(mi)}
    assert entries["local-one"]["provider"] == "omlx"
    assert entries["claude-one"]["provider"] == "anthropic"


def test_pricing_policy_accepts_local_unmetered_and_metered_cloud(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [
        {"name": "local", "provider": "omlx", "pricing_status": "unmetered"},
        {"name": "cloud", "provider": "openai", "pricing": {"input": 1.0, "output": 2.0}},
    ])
    entries = catalog.load_catalog_entries(mi)
    assert [catalog.validate_pricing_policy(entry) for entry in entries] == ["unmetered", "metered"]


def test_pricing_policy_rejects_nonfinite_rates(tmp_path):
    mi = tmp_path / "model-info.json"
    for value in (float("nan"), float("inf"), float("-inf")):
        _write_model_info(mi, [{
            "name": "bad-price", "provider": "openai",
            "pricing": {"input": value, "output": 2.0},
        }])
        with pytest.raises(ValueError, match="finite non-negative"):
            catalog.load_catalog_entries(mi)


def test_pricing_policy_rejects_cloud_unmetered_and_conflicts(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [{"name": "cloud", "provider": "openai", "pricing_status": "unmetered"}])
    with pytest.raises(ValueError, match="only valid for local"):
        catalog.load_catalog_entries(mi)

    _write_model_info(mi, [{
        "name": "local", "provider": "omlx", "pricing_status": "unmetered",
        "pricing": {"input": 0, "output": 0},
    }])
    with pytest.raises(ValueError, match="cannot combine"):
        catalog.load_catalog_entries(mi)

    _write_model_info(mi, [{
        "name": "pooled-local", "provider": "omlx", "pool": "cloud-pool",
        "pricing_status": "unmetered",
    }])
    with pytest.raises(ValueError, match="cannot use a provider pool"):
        catalog.load_catalog_entries(mi)


def test_cloud_overlay_must_clear_inherited_unmetered_marker(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [{"name": "local", "provider": "omlx", "pricing_status": "unmetered"}])
    cfg = tmp_path / "config.yaml"
    cfg.write_text("models:\n  - name: local\n    provider: openai\n")
    with pytest.raises(ValueError, match="only valid for local"):
        catalog.load_catalog_entries(mi, cfg)

    cfg.write_text("models:\n  - name: local\n    provider: openai\n    pricing_status: null\n")
    assert catalog.load_catalog_entries(mi, cfg)[0]["provider"] == "openai"


def test_routable_ids_includes_alternate_ids():
    ids = catalog.entry_routable_ids(
        {"name": "n", "alias": "a", "provider_model_id": "pm", "omlx_id": "omlx", "alternate_ids": ["x", "y"]}
    )
    assert ids == ["n", "a", "pm", "omlx", "x", "y"]


def test_provider_registry_rejects_empty_catalog_without_publishing(tmp_path, monkeypatch):
    import src.providers as providers

    monkeypatch.setattr(providers, "MODEL_INFO_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(providers, "_models", None)
    with pytest.raises(ValueError, match="effective model catalog is empty"):
        providers._load_models()
    assert providers._models is None


def test_routable_ids_match_providers_module():
    """catalog and providers share the same routable-id logic (providers delegates)."""
    import src.providers as providers

    sample = {"name": "n", "alias": "a", "provider_model_id": "p", "omlx_id": "o", "alternate_ids": ["x"]}
    assert catalog.entry_routable_ids(sample) == providers._entry_routable_ids(sample)
    assert catalog.canonical_provider("claude") == providers._canonical_provider("claude")
    assert catalog.canonical_provider("local") == providers._canonical_provider("local")


def test_load_model_info_document(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [], routing_profiles={"x": 1}, model_presets={"y": 2})
    doc = catalog.load_model_info_document(mi)
    assert doc["routing_profiles"] == {"x": 1}
    assert doc["model_presets"] == {"y": 2}
    assert catalog.load_model_info_document(tmp_path / "missing.json") == {}
