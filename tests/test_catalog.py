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


def test_overlay_empty_when_no_config(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [{"name": "a", "alias": "a", "provider": "openrouter"}])
    entries = catalog.load_catalog_entries(mi, tmp_path / "missing.yaml")
    assert [e["name"] for e in entries] == ["a"]


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


def test_routable_ids_includes_alternate_ids():
    ids = catalog.entry_routable_ids(
        {"name": "n", "alias": "a", "provider_model_id": "pm", "omlx_id": "omlx", "alternate_ids": ["x", "y"]}
    )
    assert ids == ["n", "a", "pm", "omlx", "x", "y"]


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
