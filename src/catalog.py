"""Shared catalog merge — the single source of truth for model entries.

Both the gateway router (``src.providers``) and the downstream catalog generator
(``scripts/export_catalogs.py``) read the same way: the machine-specific
``model-info.json`` ``llm`` list with the per-machine ``config.yaml`` ``models:``
overlay merged on top. On a routable-id clash, overlay fields override the base
entry while omitted capability metadata is inherited. Keeping the merge in one
place means the generator and the router cannot drift.

This module is pure: it only reads files and returns data. It has no dependency
on the provider registry, logging config, or runtime state, so it is safe to
import from standalone scripts (``uv run python scripts/export_catalogs.py``).
"""

from __future__ import annotations

import json
import math
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
_PRICING_RATE_FIELDS = {"input", "output", "cache_read", "cache_write", "reasoning"}


def canonical_provider(provider: str | None) -> str:
    """Normalize a provider name to its canonical form."""
    raw = (provider or "local").strip().lower()
    if not raw:
        return "omlx"
    return _PROVIDER_SYNONYMS.get(raw, raw)


def validate_pricing_policy(entry: dict) -> str:
    """Validate and classify a catalog entry's pricing policy.

    Source catalogs use numeric ``pricing`` for metered models,
    ``pricing_status: unmetered`` for local/oMLX models, and neither field when
    cost is unknown. This validation is deliberately shared by catalog load,
    admin writes, onboarding, and install checks so a cloud route can never be
    silently treated as free.
    """
    pricing = entry.get("pricing")
    marker = entry.get("pricing_status")
    provider = canonical_provider(entry.get("provider"))

    if pricing is not None:
        if not isinstance(pricing, dict) or not pricing:
            raise ValueError(f"model {entry.get('name')!r} pricing must be a non-empty object")
        unknown = set(pricing) - _PRICING_RATE_FIELDS
        if unknown:
            raise ValueError(
                f"model {entry.get('name')!r} has unknown pricing field(s): "
                + ", ".join(sorted(unknown))
            )
        missing = {"input", "output"} - set(pricing)
        if missing:
            raise ValueError(
                f"model {entry.get('name')!r} pricing requires: "
                + ", ".join(sorted(missing))
            )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            for value in pricing.values()
        ):
            raise ValueError(f"model {entry.get('name')!r} pricing values must be finite non-negative numbers")
        if marker is not None:
            raise ValueError(f"model {entry.get('name')!r} cannot combine pricing with pricing_status")
        return "metered"

    if marker is None:
        return "unknown"
    if marker != "unmetered":
        raise ValueError(f"model {entry.get('name')!r} pricing_status must be unmetered when present")
    if provider != "omlx":
        raise ValueError(f"model {entry.get('name')!r} unmetered pricing is only valid for local/oMLX models")
    if entry.get("pool"):
        raise ValueError(f"model {entry.get('name')!r} unmetered pricing cannot use a provider pool")
    return "unmetered"


def entry_routable_ids(entry: dict) -> list[str]:
    """Return every gateway-facing identifier for a catalog entry.

    Mirrors ``src.providers._entry_routable_ids`` so the merge keys the same way
    the router does.
    """
    ids = []
    for field in ("name", "alias", "provider_model_id", "omlx_id"):
        value = entry.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"model {entry.get('name')!r} {field} must be a string")
        ids.append(value)
    extra_ids = entry.get("alternate_ids")
    if extra_ids is None:
        extra_ids = []
    if isinstance(extra_ids, str):
        ids.append(extra_ids)
    elif isinstance(extra_ids, list):
        if any(not isinstance(value, str) for value in extra_ids):
            raise ValueError(f"model {entry.get('name')!r} alternate_ids must contain only strings")
        ids.extend(extra_ids)
    else:
        raise ValueError(f"model {entry.get('name')!r} alternate_ids must be a string or list of strings")
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
    if not isinstance(data, dict):
        raise ValueError("model-info.json root must be an object")
    llm = data.get("llm", [])
    if not isinstance(llm, list):
        raise ValueError("model-info.json llm must be a list")
    if any(not isinstance(entry, dict) for entry in llm):
        raise ValueError("model-info.json llm entries must be objects")
    return llm


def _load_overlay(config_path: Path) -> list[dict]:
    if not config_path.exists():
        return []
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("config root must be an object")
    overlay = config.get("models", [])
    if not isinstance(overlay, list):
        raise ValueError("config models overlay must be a list")
    if any(not isinstance(entry, dict) for entry in overlay):
        raise ValueError("config models overlay entries must be objects")
    return overlay


def load_catalog_entries(
    model_info_path: Path | str,
    config_path: Path | str | None = None,
    *,
    overlay: list[dict] | None = None,
    include_retired: bool = False,
    require_nonempty: bool = False,
) -> list[dict]:
    """Return merged, deduplicated catalog entries (overlay wins on id clash).

    Reads the machine-specific ``model-info.json`` catalog and merges the
    ``models:`` overlay from ``config_path`` on top. Entries are deduplicated by
    routable id (name / alias / provider_model_id / omlx_id / alternate_ids).
    When an overlay entry shares an id with one catalog entry, its fields
    override that entry while omitted fields (notably capabilities such as
    ``vision``) are inherited.

    The returned list is ordered: catalog entries (in file order) that survive
    the overlay, followed by overlay-only entries (in overlay order). Each
    entry has its ``provider`` field canonicalized.

    ``overlay`` (if given) takes precedence over reading ``models:`` from
    ``config_path``; pass it directly when the caller already has the loaded
    config (e.g. the gateway router reads via its cached config object).

    ``include_retired`` defaults to False, skipping GGUF/llama.cpp entries to
    match the gateway router. Generators that want the full raw catalog
    (e.g. for reporting) may pass True. ``require_nonempty`` is enabled by the
    runtime and installer so production cannot publish an empty registry.
    """
    model_info_path = Path(model_info_path)

    catalog_entries = _load_model_info(model_info_path)
    if overlay is None:
        config_path = Path(config_path) if config_path else None
        overlay_entries = _load_overlay(config_path) if config_path else []
    else:
        if not isinstance(overlay, list):
            raise ValueError("config models overlay must be a list")
        if any(not isinstance(entry, dict) for entry in overlay):
            raise ValueError("config models overlay entries must be objects")
        overlay_entries = list(overlay)

    # Index catalog entries by each routable id.
    by_id: dict[str, dict] = {}
    order: list[str] = []  # first-seen id per entry, to preserve order
    for entry in catalog_entries:
        provider = canonical_provider(entry.get("provider", "local"))
        if not include_retired and provider in _RETIRED_LOCAL_PROVIDERS:
            continue
        normalized = dict(entry)
        normalized["provider"] = provider
        validate_pricing_policy(normalized)
        routable = entry_routable_ids(normalized)
        if not routable:
            raise ValueError("catalog model entries require at least one routable identifier")
        primary: str | None = None
        for model_id in routable:
            if primary is None:
                primary = model_id
            existing = by_id.get(model_id)
            if existing is not None and existing is not normalized:
                raise ValueError(
                    f"routable id {model_id!r} collides between "
                    f"{existing.get('name')!r} and {normalized.get('name')!r}"
                )
            by_id[model_id] = normalized
        if primary is not None and primary not in order:
            order.append(primary)

    # Apply overlay: an overlay entry wins every id it claims, replacing any
    # catalog entry that shared one. The overlay entry's primary id is appended
    # to the order list if it is new.
    for entry in overlay_entries:
        overlay_ids = entry_routable_ids(entry)
        collided_primaries: set[str] = set()
        for model_id in overlay_ids:
            existing = by_id.get(model_id)
            if existing is not None:
                for prim in order:
                    if by_id.get(prim) is existing:
                        collided_primaries.add(prim)
                        break
        # Prefer the same-name base for capability inheritance. Other collisions
        # are still evicted because the overlay claims those identifiers (for
        # example an alternate id retiring an older logical catalog entry).
        named_primary = str(entry.get("name") or "")
        base_primary = named_primary if named_primary in collided_primaries else None
        if base_primary is None and len(collided_primaries) == 1:
            base_primary = next(iter(collided_primaries))
        if base_primary is None and len(collided_primaries) > 1:
            raise ValueError(
                f"overlay model {entry.get('name')!r} ambiguously collides with "
                f"multiple catalog models: {sorted(collided_primaries)}"
            )
        base = by_id.get(base_primary) if base_primary else None
        normalized = {**base, **entry} if base else dict(entry)
        provider = canonical_provider(normalized.get("provider", "local"))
        if not include_retired and provider in _RETIRED_LOCAL_PROVIDERS:
            continue
        normalized["provider"] = provider
        validate_pricing_policy(normalized)
        ids = entry_routable_ids(normalized)
        if not ids:
            raise ValueError("catalog model entries require at least one routable identifier")
        # Evict the base entry's old identifier index, then index the merged
        # entry under its inherited and overridden routable identifiers.
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

    result = [by_id[primary] for primary in order]
    if require_nonempty and not result:
        raise ValueError("effective model catalog is empty")
    return result


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
