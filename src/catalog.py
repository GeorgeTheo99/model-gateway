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
import re
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
_PRICING_RATE_FIELDS = {
    "input", "output", "cache_read", "cache_write", "cache_write_1h", "reasoning",
}

# Canonical gateway thinking levels. ``off`` is a real explicit control while
# Auto is represented by omitting a reasoning control from the request.
# What a model produces (docs/adr/0004-local-image-and-video-generation.md). Absent is chat;
# only chat models appear in default discovery or answer the chat endpoints.
MODALITIES = ("chat", "image", "video")
# Providers that run on this Mac, whose models may be unmetered.
LOCAL_PROVIDERS = frozenset({"omlx", "media"})
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
ENABLED_THINKING_LEVELS = THINKING_LEVELS[1:]
_THINKING_MODES = {"", "optional", "always"}


def _canonical_thinking_mode(entry: dict) -> str:
    """Map legacy no-thinking spellings to the canonical empty mode."""
    thinking = entry.get("thinking", "")
    if thinking is None or thinking == "never":
        return ""
    if not isinstance(thinking, str) or thinking not in _THINKING_MODES:
        raise ValueError(
            f"model {entry.get('name')!r} thinking must be optional, always, or empty"
        )
    return thinking


def normalized_thinking_levels(entry: dict) -> list[str]:
    """Validate and return one model's effective canonical thinking levels.

    Older catalog entries did not carry ``thinking_levels``. Keep those rows
    readable with a conservative fallback derived from their existing
    ``thinking`` mode; newly written rows can persist the returned explicit
    list. No-thinking models intentionally expose no levels.
    """
    thinking = _canonical_thinking_mode(entry)

    if "thinking_levels" not in entry:
        if thinking == "optional":
            return list(THINKING_LEVELS)
        if thinking == "always":
            return list(ENABLED_THINKING_LEVELS)
        return []

    raw = entry.get("thinking_levels")
    if not isinstance(raw, list) or any(not isinstance(level, str) for level in raw):
        raise ValueError(f"model {entry.get('name')!r} thinking_levels must be a list of strings")
    if len(raw) != len(set(raw)):
        raise ValueError(f"model {entry.get('name')!r} thinking_levels must not contain duplicates")
    unknown = set(raw) - set(THINKING_LEVELS)
    if unknown:
        raise ValueError(
            f"model {entry.get('name')!r} has unknown thinking level(s): "
            + ", ".join(sorted(unknown))
        )

    levels = [level for level in THINKING_LEVELS if level in raw]
    if not thinking and levels:
        raise ValueError(f"model {entry.get('name')!r} cannot define thinking_levels without thinking")
    if thinking and not any(level != "off" for level in levels):
        raise ValueError(f"model {entry.get('name')!r} thinking requires at least one enabled thinking level")
    if "off" in levels and thinking != "optional":
        raise ValueError(f"model {entry.get('name')!r} supports off only when thinking is optional")
    return levels


def normalize_thinking_capabilities(entry: dict) -> dict:
    """Return a copy with canonical thinking mode and effective levels."""
    normalized = dict(entry)
    if "thinking" in normalized:
        normalized["thinking"] = _canonical_thinking_mode(normalized)
    normalized["thinking_levels"] = normalized_thinking_levels(normalized)
    return normalized


def modality(entry: dict) -> str:
    """``chat``, ``image`` or ``video``; validated at load by :func:`validate_client_metadata`."""
    return entry.get("modality") or "chat"


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
    if provider not in LOCAL_PROVIDERS:
        raise ValueError(f"model {entry.get('name')!r} unmetered pricing is only valid for local models")
    if entry.get("pool"):
        raise ValueError(f"model {entry.get('name')!r} unmetered pricing cannot use a provider pool")
    return "unmetered"


_DISPLAY_NAME_MAX = 80
_ROUTE_GROUP_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,63}$")


def validate_client_metadata(entry: dict) -> None:
    """Validate the optional fields consumers may show or group by.

    ``zdr: true`` declares that the route keeps zero data retention upstream.
    The ``openrouter_zdr`` quirk enforces it on OpenRouter, so it requires the
    declaration. ``base_model`` names the upstream model independently of the
    route (every provider's route to it shares the key), and ``family`` groups
    models a consumer offers one provider choice for; they come together.
    """
    name = entry.get("name")
    display_name = entry.get("display_name")
    if display_name is not None and (
        not isinstance(display_name, str)
        or not display_name.strip()
        or len(display_name) > _DISPLAY_NAME_MAX
        or not display_name.isprintable()
    ):
        raise ValueError(
            f"model {name!r} display_name must be printable text of 1-{_DISPLAY_NAME_MAX} characters"
        )
    zdr = entry.get("zdr")
    if zdr is not None and not isinstance(zdr, bool):
        raise ValueError(f"model {name!r} zdr must be true or false")
    quirks = entry.get("quirks") or []
    if isinstance(quirks, str):
        quirks = [quirk.strip() for quirk in quirks.split(",")]
    if "openrouter_zdr" in quirks and zdr is not True:
        raise ValueError(f"model {name!r} openrouter_zdr quirk requires zdr: true")
    for field in ("family", "base_model"):
        value = entry.get(field)
        if value is not None and (not isinstance(value, str) or not _ROUTE_GROUP_RE.fullmatch(value)):
            raise ValueError(
                f"model {name!r} {field} must be 1-64 lowercase letters, digits, dots, or hyphens"
            )
    if (entry.get("family") is None) != (entry.get("base_model") is None):
        raise ValueError(f"model {name!r} family and base_model must be set together")
    if entry.get("modality") is not None and entry["modality"] not in MODALITIES:
        raise ValueError(f"model {name!r} modality must be one of: {', '.join(MODALITIES)}")


def entry_routable_ids(entry: dict) -> list[str]:
    """Return every gateway-facing identifier for a catalog entry.

    Mirrors ``src.providers._entry_routable_ids`` so the merge keys the same way
    the router does. An owner-added (``custom-``) provider's upstream id is not
    one: it could otherwise claim another route's identifier.
    """
    ids = []
    custom = canonical_provider(entry.get("provider")).startswith("custom-")
    for field in ("name", "alias", "provider_model_id", "omlx_id"):
        value = entry.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"model {entry.get('name')!r} {field} must be a string")
        if not (custom and field == "provider_model_id"):
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


def allows_empty(model_info_path: Path | str) -> bool:
    """Whether the catalog explicitly opts in to having no models.

    Fresh installs write ``"allow_empty": true`` so the gateway can start before
    the first model is registered; other catalogs must stay non-empty.
    """
    path = Path(model_info_path)
    if not path.exists():
        return False
    with open(path) as f:
        data = json.load(f)
    return isinstance(data, dict) and data.get("allow_empty") is True


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


# Owner-row bookkeeping (ADR 0002). ``origin`` is computed, never trusted from a file.
OWNER_META_FIELDS = frozenset({"shipped_base", "review", "origin"})
# Fields that say where a model routes. An owner row that changes one replaces the
# lower row outright; merging the rest of a different route over it could misroute.
ROUTE_FIELDS = ("provider", "pool", "provider_model_id", "omlx_id")
_ID_FIELDS = ("name", "alias", "provider_model_id", "omlx_id", "alternate_ids", "provider")
LOWER_LAYERS = frozenset({"shipped", "pack"})


def same_field(field: str, value: object, other: object) -> bool:
    """Whether two values of a catalog field agree (providers compared canonically)."""
    if field == "provider":
        return canonical_provider(value) == canonical_provider(other)
    return value == other


def route_differs(fields: dict, base: dict) -> bool:
    """Whether ``fields`` sets one of :data:`ROUTE_FIELDS` to another value than ``base``."""
    return any(field in fields and not same_field(field, fields[field], base.get(field)) for field in ROUTE_FIELDS)


def _alternates(row: dict) -> list:
    value = row.get("alternate_ids") or []
    return [value] if isinstance(value, str) else list(value)


def _strip_ids(row: dict, claimed: set[str]) -> dict:
    """``row`` without the alias/alternate ids in ``claimed``."""
    row = dict(row)
    if row.get("alias") in claimed:
        row.pop("alias")
    if row.get("alternate_ids"):
        row["alternate_ids"] = [value for value in _alternates(row) if value not in claimed]
    return row


def _fold_shipped(shipped: list[dict], owner: list[dict], shipped_names: frozenset[str]) -> list[dict]:
    """Merge owner rows over the lower layers (shipped, then packs): the owner wins.

    An owner row named after a loaded lower model overrides only the fields it
    sets, unless it changes the route (:data:`ROUTE_FIELDS`): then it replaces
    that model. An override that no longer validates leaves the model
    unavailable (``origin.override_error``) rather than serving the lower row. A
    row recording ``shipped_base``/``review`` is an override and never loads on
    its own. Identifiers the owner sets win per identifier: another model's
    alias or alternate id moves to the owner; its name or upstream id makes the
    owner's row replace it.
    """
    by_name = {row["name"]: row for row in shipped}
    folded: list[tuple[dict, set[str], dict | None]] = []  # (row, owner-set ids, lower row it merges)
    replaced: set[str] = set()
    for row in owner:
        name = row.get("name")
        fields = {k: v for k, v in row.items() if k not in OWNER_META_FIELDS}
        own_ids = set(entry_routable_ids({k: fields[k] for k in _ID_FIELDS if k in fields}))
        sparse = "shipped_base" in row
        owned = {"layer": "owner", **({"review": True} if row.get("review") is True else {})}
        base = by_name.get(name) if isinstance(name, str) else None
        if base is None:
            if sparse or (name in shipped_names and not row.get("provider")):
                continue  # an override whose lower model is not loaded here
            folded.append(({**fields, "origin": owned}, own_ids, None))
            continue
        replaced.add(name)
        # Another route under the name, or the owner's own row for a pack's name (a pack
        # never shapes an owner model), replaces the lower model: nothing is merged in.
        if route_differs(fields, base) or (base["origin"]["layer"] == "pack" and not sparse):
            folded.append(({**fields, "origin": owned}, own_ids, None))
            continue
        merged = {**base, **fields}
        if "thinking" in fields and fields["thinking"] != base.get("thinking") and "thinking_levels" not in fields:
            merged.pop("thinking_levels", None)
        changed = sorted(k for k in fields if k != "name" and not same_field(k, fields[k], base.get(k)))
        recorded = row.get("shipped_base") if isinstance(row.get("shipped_base"), dict) else {}
        origin = {**base["origin"], "overridden": changed,
                  "stale_overrides": sorted(k for k in changed
                                            if k in recorded and not same_field(k, recorded[k], base.get(k)))}
        if row.get("review") is True:
            origin["review"] = True
        try:
            validate_pricing_policy(merged)
            validate_client_metadata(merged)
            normalize_thinking_capabilities(merged)
        except ValueError as exc:
            # A release changed the row under a once-valid override. Fail closed: the
            # lower row's data, unavailable, with the reason.
            merged, origin = dict(base), {**origin, "override_error": str(exc)}
        merged["origin"] = origin
        merged["_owner_ids"] = sorted(own_ids)  # which identifiers the owner set (dropped before returning)
        folded.append((merged, own_ids, base))
    # Identifiers a merged row takes from its lower row yield to ones another owner row sets.
    result: list[dict] = []
    for index, (row, _own, base) in enumerate(folded):
        if base is not None:
            others = set().union(*(own for i, (_r, own, _b) in enumerate(folded) if i != index))
            row = _strip_ids(row, others - _own)
        result.append(row)
    claimed: set[str] = set()
    for row in result:
        claimed.update(entry_routable_ids(row))
    kept: list[dict] = []
    for row in shipped:
        if row["name"] in replaced:
            continue
        movable = {str(row.get("alias") or ""), *_alternates(row)}
        if any(model_id in claimed for model_id in entry_routable_ids(row) if model_id not in movable):
            continue
        kept.append(_strip_ids(row, claimed))
    return kept + result


def _is_lower(entry: dict) -> bool:
    return (entry.get("origin") or {}).get("layer") in LOWER_LAYERS


def _yields(entry: dict, shared: set[str]) -> bool:
    """Whether ``entry`` gives way on a collision over ``shared``: shipped/pack rows do,
    unless an owner override of them set one of those identifiers itself."""
    return _is_lower(entry) and not shared.intersection(entry.get("_owner_ids") or ())


def _record_yield(lower: dict, winner: dict) -> None:
    """Log a displaced lower model and note it on the winner's origin for admin status."""
    import logging

    logging.getLogger("model-gateway").warning(
        "catalog: %s model %r yields to %r, which uses one of its identifiers",
        (lower.get("origin") or {}).get("layer"), lower.get("name"), winner.get("name"),
    )
    origin = dict(winner.get("origin") or {"layer": "owner"})
    origin["displaced"] = sorted({*origin.get("displaced", []), str(lower.get("name"))})
    winner["origin"] = origin


def load_catalog_entries(
    model_info_path: Path | str,
    config_path: Path | str | None = None,
    *,
    overlay: list[dict] | None = None,
    shipped: list[dict] | None = None,
    shipped_names: frozenset[str] = frozenset(),
    excluded: frozenset[str] = frozenset(),
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

    ``shipped`` rows (see :mod:`src.shipped`) load beneath the owner's catalog,
    which wins per field and per identifier (:func:`_fold_shipped`).
    ``shipped_names`` lists every shipped name, loaded or not. Shipped and pack
    models named in ``excluded`` (hidden or retired) are dropped from the result.

    ``include_retired`` defaults to False, skipping GGUF/llama.cpp entries to
    match the gateway router. Generators that want the full raw catalog
    (e.g. for reporting) may pass True. ``require_nonempty`` is enabled by the
    runtime and installer so production cannot publish an empty registry,
    unless the catalog explicitly sets ``"allow_empty": true`` (fresh installs).
    """
    model_info_path = Path(model_info_path)

    catalog_entries = _load_model_info(model_info_path)
    if shipped is not None:
        catalog_entries = _fold_shipped(list(shipped), catalog_entries, shipped_names)
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
        validate_client_metadata(normalized)
        normalized = normalize_thinking_capabilities(normalized)
        routable = entry_routable_ids(normalized)
        if not routable:
            raise ValueError("catalog model entries require at least one routable identifier")
        clashes = {id(by_id[model_id]): by_id[model_id] for model_id in routable if model_id in by_id}
        if clashes:
            # A shipped or pack row yields to whatever it collides with, on identifiers the
            # release or pack supplied; any collision on an identifier the owner set is an error.
            if all(_yields(normalized, set(routable) & set(entry_routable_ids(c))) for c in clashes.values()):
                _record_yield(normalized, next(iter(clashes.values())))
                continue
            for existing in clashes.values():
                shared = set(routable) & set(entry_routable_ids(existing))
                if not _yields(existing, shared):
                    raise ValueError(
                        f"routable id {sorted(shared)[0]!r} collides between "
                        f"{existing.get('name')!r} and {normalized.get('name')!r}"
                    )
                _record_yield(existing, normalized)
                for model_id in entry_routable_ids(existing):
                    if by_id.get(model_id) is existing:
                        del by_id[model_id]
                order[:] = [prim for prim in order if prim in by_id]
        for model_id in routable:
            by_id[model_id] = normalized
        if routable[0] not in order:
            order.append(routable[0])

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
        fields = {k: v for k, v in entry.items() if k != "origin"}
        if base is not None and _is_lower(base) and (
                (base.get("origin") or {}).get("layer") == "pack"
                or route_differs(fields, base)):
            # Another route under a shipped name, or any overlay row over a pack model, is
            # the owner's own model: nothing is merged in.
            base = None
        lower = base is not None and _is_lower(base)
        normalized = {**base, **fields} if base else {**fields, "origin": {"layer": "owner"}}
        if (
            base
            and "thinking" in entry
            and entry.get("thinking") != base.get("thinking")
            and "thinking_levels" not in entry
        ):
            normalized.pop("thinking_levels", None)
        provider = canonical_provider(normalized.get("provider", "local"))
        if not include_retired and provider in _RETIRED_LOCAL_PROVIDERS:
            continue
        normalized["provider"] = provider
        try:
            validate_pricing_policy(normalized)
            validate_client_metadata(normalized)
            normalized = normalize_thinking_capabilities(normalized)
        except ValueError as exc:
            if not lower:
                raise
            # An overlay a release invalidated: fail closed on the lower row, as for owner rows.
            normalized = {**base, "origin": {**base["origin"], "override_error": str(exc)}}
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

    # Hiding and retirement apply to shipped and pack models; the owner's own rows always load.
    result = [
        {k: v for k, v in by_id[primary].items() if k != "_owner_ids"} for primary in order
        if by_id[primary].get("name") not in excluded
        or (by_id[primary].get("origin") or {}).get("layer") not in LOWER_LAYERS
    ]
    if require_nonempty and not result and not allows_empty(model_info_path):
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
