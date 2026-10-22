"""The shipped catalog layer (docs/adr/0002-layered-model-catalog.md).

Curated release profiles (``auto_add: true``) are read on every catalog load
instead of being copied into the owner's ``model-info.json``, so a release's
model fixes, additions and retirements reach every install while owner rows
still win field by field (see :func:`src.catalog.load_catalog_entries`).

The owner's choices live in the config ``catalog:`` block::

    catalog:
      shipped: all            # or none, or a list of profile ids / model names
      hidden: [model-name]    # deleted shipped models; never loaded
      pinned: [model-name]    # kept even when a release retires them
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class ShippedLayer:
    entries: list[dict] = field(default_factory=list)
    """Shipped rows this install loads, in profile order."""
    names: frozenset[str] = frozenset()
    """Every shipped model name, whether or not it loads here."""
    retired: dict[str, str | None] = field(default_factory=dict)
    """Retired name -> its replacement (or None), excluding pinned names."""
    hidden: frozenset[str] = frozenset()


def _names(settings: dict, key: str) -> frozenset[str]:
    value = settings.get(key, [])
    if not isinstance(value, list) or any(not isinstance(name, str) or not name for name in value):
        raise ValueError(f"config catalog.{key} must be a list of model names")
    return frozenset(value)


def settings(config: dict) -> tuple[str | frozenset[str], frozenset[str], frozenset[str]]:
    """Validated ``(selection, hidden, pinned)`` from the config ``catalog:`` block."""
    raw = config.get("catalog")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("config catalog must be an object")
    unknown = set(raw) - {"shipped", "hidden", "pinned"}
    if unknown:
        raise ValueError("config catalog has unknown field(s): " + ", ".join(sorted(unknown)))
    selection = raw.get("shipped", "all")
    if isinstance(selection, list):
        if any(not isinstance(name, str) or not name for name in selection):
            raise ValueError("config catalog.shipped list must hold profile ids or model names")
        selection = frozenset(selection)
    elif selection not in ("all", "none"):
        raise ValueError("config catalog.shipped must be all, none, or a list")
    return selection, _names(raw, "hidden"), _names(raw, "pinned")


def provider_ready(profile: dict, config: dict) -> bool:
    """Whether this install serves the profile's provider: on, at the profile's endpoint, with a key."""
    import src.providers as providers

    wanted = profile["provider"]
    provider_id = providers._canonical_provider(str(wanted["id"]))
    location = providers._find_provider_location(config, provider_id)
    if location is not None and isinstance(location[2], dict) and location[2].get("enabled") is False:
        return False
    effective = providers._effective_provider_config(config, provider_id)
    return (
        str(effective.get("base_url") or "").rstrip("/") == str(wanted["base_url"]).rstrip("/")
        and (effective.get("protocol") or "openai") == wanted.get("protocol", "openai")
        and bool(str(effective.get("api_key") or "").strip())
    )


def _packaged() -> dict[str, dict]:
    from src.onboarding import packaged_profiles

    return packaged_profiles()


def layer(
    config: dict,
    *,
    profiles: dict[str, dict] | None = None,
    ready: Callable[[dict, dict], bool] = provider_ready,
) -> ShippedLayer:
    """Compute the shipped rows, names and retirements for one config."""
    if profiles is None:
        profiles = _packaged()
    selection, hidden, pinned = settings(config)
    entries: list[dict] = []
    names: set[str] = set()
    retired: dict[str, str | None] = {}
    for profile_id, profile in profiles.items():
        if not profile.get("auto_add"):
            continue
        retire = profile.get("retire") or {}
        replaced_by = retire.get("replaced_by") or {}
        for name in retire.get("models") or []:
            if name not in pinned:
                retired[name] = replaced_by.get(name)
        models = profile["models"]
        names.update(model["name"] for model in models)
        if selection == "none":
            continue
        if selection != "all":
            models = [m for m in models if profile_id in selection or m["name"] in selection]
        if not models or not ready(profile, config):
            continue
        entries.extend(
            {**model, "origin": {"layer": "shipped", "profile": profile_id}} for model in models
        )
    # A name another profile still ships is alive, whatever an older profile retired.
    retired = {name: successor for name, successor in retired.items() if name not in names}
    return ShippedLayer(entries, frozenset(names), retired, hidden)
