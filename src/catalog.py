"""Shared catalog merge — the single source of truth for model entries.

Both the gateway router (``src.providers``) and the downstream catalog generator
(``scripts/export_catalogs.py``) read the same way: the committed
``model-info.json`` ``llm`` list with the per-machine ``config.yaml`` ``models:``
overlay merged on top (overlay wins on any routable-id clash). Keeping the merge
in one place means the generator and the router can never drift.

This module is pure: it only reads files and returns data. It has no dependency
on the provider registry, logging config, or runtime state, so it is safe to
import from standalone scripts (``uv run python scripts/export_catalogs.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

# Provider synonyms mirrored from ``src.providers``. Kept here (not imported)
# so this module stays dependency-free for standalone script use; the router
# delegates to these via ``catalog.canonical_provider``.
_PROVIDER_SYNONYMS = {
    "local": "omlx",
    "omlx": "omlx",
    "mlx": "omlx",
    "openai": "openai",
    "gpt": "openai",
    "anthropic": "anthropic",
    "anthropci": "anthropic",
    "claude": "anthropic",
    "google": "google",
    "gemini": "google",
    "zhipuai": "zhipuai",
    "zai": "zhipuai",
    "bigmodel": "zhipuai",
    "databricks": "databricks",
    "dbx": "databricks",
    "dbrx": "databricks",
}

# Retired local serving backends. Entries whose canonical provider is one of
# these are skipped by default (matches ``src.providers._load_models``).
_RETIRED_LOCAL_PROVIDERS = {"gguf", "llama", "llama_cpp", "llama.cpp"}


def canonical_provider(provider: str | None) -> str:
    """Normalize a provider name to its canonical form."""
    raw = (provider or "local").strip().lower()
    if not raw:
        return "omlx"
    return _PROVIDER_SYNONYMS.get(raw, raw)


def entry_routable_ids(entry: dict) -> list[str]:
    """Return every gateway-facing identifier for a catalog entry.

    Mirrors ``src.providers._entry_routable_ids`` so the merge keys the same way
    the router does.
    """
    ids = [entry.get("name"), entry.get("alias"), entry.get("provider_model_id"), entry.get("omlx_id")]
    extra_ids = entry.get("alternate_ids") or []
    if isinstance(extra_ids, str):
        ids.append(extra_ids)
    elif isinstance(extra_ids, list):
        ids.extend(extra_ids)
    result: list[str] = []
    seen: set[str] = set()
    for value in ids:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _load_model_info(model_info_path: Path) -> list[dict]:
    if not model_info_path.exists():
        return []
    with open(model_info_path) as f:
        data = json.load(f)
    llm = data.get("llm") or []
    return [e for e in llm if isinstance(e, dict)]


def _load_overlay(config_path: Path) -> list[dict]:
    if not config_path.exists():
        return []
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    overlay = config.get("models") or []
    if not isinstance(overlay, list):
        return []
    return [e for e in overlay if isinstance(e, dict)]


def load_catalog_entries(
    model_info_path: Path | str,
    config_path: Path | str | None = None,
    *,
    overlay: list[dict] | None = None,
    include_retired: bool = False,
) -> list[dict]:
    """Return merged, deduplicated catalog entries (overlay wins on id clash).

    Reads ``model-info.json`` (committed catalog) and merges the ``models:``
    overlay from ``config_path`` on top. Entries are deduplicated by routable id
    (name / alias / provider_model_id / omlx_id / alternate_ids); when an
    overlay entry shares any id with a catalog entry, the overlay entry wins
    completely (it replaces the catalog entry for every id it claims).

    The returned list is ordered: catalog entries (in file order) that survive
    the overlay, followed by overlay-only entries (in overlay order). Each
    entry has its ``provider`` field canonicalized.

    ``overlay`` (if given) takes precedence over reading ``models:`` from
    ``config_path``; pass it directly when the caller already has the loaded
    config (e.g. the gateway router reads via its cached config object).

    ``include_retired`` defaults to False, skipping GGUF/llama.cpp entries to
    match the gateway router. Generators that want the full raw catalog
    (e.g. for reporting) may pass True.
    """
    model_info_path = Path(model_info_path)

    catalog_entries = _load_model_info(model_info_path)
    if overlay is None:
        config_path = Path(config_path) if config_path else None
        overlay_entries = _load_overlay(config_path) if config_path else []
    else:
        overlay_entries = [e for e in overlay if isinstance(e, dict)]

    # Index catalog entries by each routable id.
    by_id: dict[str, dict] = {}
    order: list[str] = []  # first-seen id per entry, to preserve order
    for entry in catalog_entries:
        provider = canonical_provider(entry.get("provider", "local"))
        if not include_retired and provider in _RETIRED_LOCAL_PROVIDERS:
            continue
        normalized = dict(entry)
        normalized["provider"] = provider
        primary: str | None = None
        for model_id in entry_routable_ids(normalized):
            if primary is None:
                primary = model_id
            by_id[model_id] = normalized
        if primary is not None and primary not in order:
            order.append(primary)

    # Apply overlay: an overlay entry wins every id it claims, replacing any
    # catalog entry that shared one. The overlay entry's primary id is appended
    # to the order list if it is new.
    for entry in overlay_entries:
        provider = canonical_provider(entry.get("provider", "local"))
        if not include_retired and provider in _RETIRED_LOCAL_PROVIDERS:
            continue
        normalized = dict(entry)
        normalized["provider"] = provider
        ids = entry_routable_ids(normalized)
        # If any overlay id collides with an existing catalog entry, evict all
        # of that catalog entry's ids first so the overlay fully replaces it.
        collided_primaries: set[str] = set()
        for model_id in ids:
            existing = by_id.get(model_id)
            if existing is not None and existing is not normalized:
                # find the existing entry's primary id (first id in order that maps to it)
                for prim in order:
                    if by_id.get(prim) is existing:
                        collided_primaries.add(prim)
                        break
        for prim in collided_primaries:
            existing = by_id.get(prim)
            if existing is None:
                continue
            for model_id in entry_routable_ids(existing):
                if by_id.get(model_id) is existing:
                    del by_id[model_id]
            order.remove(prim)
        primary: str | None = None
        for model_id in ids:
            if primary is None:
                primary = model_id
            by_id[model_id] = normalized
        if primary is not None and primary not in order:
            order.append(primary)

    return [by_id[primary] for primary in order]


def load_model_info_document(model_info_path: Path | str) -> dict[str, Any]:
    """Load the full ``model-info.json`` document (not just the llm list).

    Used by generators that also need ``routing_profiles`` / ``model_presets``
    / ``stt`` / ``tts`` etc.
    """
    model_info_path = Path(model_info_path)
    if not model_info_path.exists():
        return {}
    with open(model_info_path) as f:
        return json.load(f) or {}
