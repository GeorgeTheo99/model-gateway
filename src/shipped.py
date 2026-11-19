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
      provider_preference: [provider-id]  # route order for base-model references
      directory: {sync: daily, lines: {line: bool}, models: [name]}  # models.dev (src.directory)
      provider_policy: {openrouter: {zero_retention_only: true}}
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


def provider_preference(config: dict) -> list[str]:
    """The provider order for resolving a base-model reference (ADR 0002 R1): the owner's,
    else the directory's default ranking (docs/adr/0005-models-dev-directory.md)."""
    settings(config)
    from src import directory

    return directory.ranking(config, directory.load_corrections())


def settings(config: dict) -> tuple[str | frozenset[str], frozenset[str], frozenset[str]]:
    """Validated ``(selection, hidden, pinned)`` from the config ``catalog:`` block."""
    raw = config.get("catalog")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("config catalog must be an object")
    unknown = set(raw) - {"shipped", "hidden", "pinned", "provider_preference", "directory", "provider_policy"}
    if unknown:
        raise ValueError("config catalog has unknown field(s): " + ", ".join(sorted(unknown)))
    selection = raw.get("shipped", "all")
    if isinstance(selection, list):
        if any(not isinstance(name, str) or not name for name in selection):
            raise ValueError("config catalog.shipped list must hold profile ids or model names")
        selection = frozenset(selection)
    elif selection not in ("all", "none"):
        raise ValueError("config catalog.shipped must be all, none, or a list")
    _names(raw, "provider_preference")
    from src import directory

    directory.settings(config)
    directory.provider_policy(config)
    return selection, _names(raw, "hidden"), _names(raw, "pinned")


def provider_ready(profile: dict, config: dict) -> bool:
    """Whether this install serves the profile's provider: on, at the profile's endpoint, with a key.

    A ChatGPT plan's rows load signed out too, unavailable until the owner signs in, so a routing
    profile that names one stays valid when its session ends (src/chatgpt_auth.py).
    """
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
        and (wanted.get("auth") == "chatgpt" or bool(str(effective.get("api_key") or "").strip()))
    )


def _directory(config: dict, ready: Callable[[dict, dict], bool], keep: frozenset[str], config_path=None):
    from src import directory

    return directory.layer(config, ready=ready, keep=keep, config_path=config_path)


def _bound_routes(config_path=None) -> frozenset[str]:
    """Route names registered consumer profiles use: kept when the directory retires them."""
    from src import profiles

    return frozenset(profiles.routed_models(config_path))


def providers_canonical(provider: str) -> str:
    from src.catalog import canonical_provider

    return canonical_provider(provider)


def _packaged() -> dict[str, dict]:
    from src.onboarding import packaged_profiles

    return packaged_profiles()


def _registered_packs(config_path=None) -> dict[str, dict]:
    from src import packs

    return packs.load(config_path)


def _local_ai_rows(config: dict, *, installed_only: bool = True) -> list[dict]:
    from src import local_runtime

    return local_runtime.catalog_rows(config, installed_only=installed_only)


def _media_rows(config: dict, *, installed_only: bool = True) -> list[dict]:
    from src import media_runtime

    return media_runtime.catalog_rows(config, installed_only=installed_only)


def _pinned_local_ai_names() -> set[str]:
    from src import local_runtime

    return set(local_runtime.pinned_models())


def all_shipped_rows(profiles: dict[str, dict] | None = None) -> list[dict]:
    """Every model a release ships, loaded on this install or not."""
    profiles = _packaged() if profiles is None else profiles
    return [*(model for profile in profiles.values() if profile.get("auto_add") for model in profile["models"]),
            *_local_ai_rows({}, installed_only=False), *_media_rows({}, installed_only=False)]


def layer(
    config: dict,
    *,
    profiles: dict[str, dict] | None = None,
    ready: Callable[[dict, dict], bool] = provider_ready,
    include_packs: bool = True,
    config_path=None,
) -> ShippedLayer:
    """Compute the rows beneath the owner's catalog: shipped, then consumer packs (src.packs)."""
    if profiles is None:
        profiles = _packaged()
    selection, hidden, pinned = settings(config)
    entries: list[dict] = []
    names: set[str] = set()
    retired: dict[str, str | None] = {}
    # models.dev directory rows (docs/adr/0005-models-dev-directory.md) load as shipped rows of
    # the pseudo-profile "models.dev"; a curated row for a route the directory maps yields to it.
    found = _directory(config, ready, frozenset(pinned) | _bound_routes(config_path), config_path)
    if selection != "none":
        directory_rows = [row for row in found.rows if selection == "all" or "models.dev" in selection
                          or row["name"] in selection]
        entries.extend(directory_rows)
        names.update(row["name"] for row in directory_rows)
    retired.update({name: successor for name, successor in found.retired.items() if name not in pinned})
    for profile_id, profile in profiles.items():
        if not profile.get("auto_add"):
            continue
        retire = profile.get("retire") or {}
        replaced_by = retire.get("replaced_by") or {}
        for name in retire.get("models") or []:
            if name not in pinned:
                retired[name] = replaced_by.get(name)
        models = [model for model in profile["models"]
                  if (providers_canonical(model["provider"]), model.get("provider_model_id")) not in found.known_routes]
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
    # Gateway local AI (src.local_runtime) loads once installed, whatever the selection:
    # the owner added it. Its names count only then, so an owner's own row under one
    # (another oMLX setup) is never mistaken for an override of an absent model. A model
    # the owner pinned (docs/adr/0003-scoped-access-and-lightweight-local-models.md
    # section 6) wins over a release row a later release ships under one of its identifiers.
    from src.catalog import entry_routable_ids

    from src.catalog import _record_yield

    pins = _pinned_local_ai_names()
    for model in _local_ai_rows(config):
        pinned = model["name"] in pins
        row = {**model, "origin": {"layer": "shipped", "profile": "local-ai-pin" if pinned else "local-ai"}}
        if pinned:
            claimed = set(entry_routable_ids(model))
            for entry in [entry for entry in entries if claimed.intersection(entry_routable_ids(entry))]:
                _record_yield(entry, row)  # logged, and listed in admin status as displaced
                entries.remove(entry)
        names.add(model["name"])
        entries.append(row)
    # Gateway media generation (src.media_runtime), likewise once installed (ADR 0004 section 3).
    for model in _media_rows(config):
        names.add(model["name"])
        entries.append({**model, "origin": {"layer": "shipped", "profile": "media"}})
    # A name another profile still ships is alive, whatever an older profile retired.
    retired = {name: successor for name, successor in retired.items() if name not in names}
    if include_packs:
        shipped_ids = {model_id for row in entries for model_id in entry_routable_ids(row)}
        shipped_ids.update(str(row["base_model"]) for row in entries if row.get("base_model"))
        for namespace, pack in sorted(_registered_packs(config_path).items()):
            for model in pack.get("models") or []:
                # A release that later ships an identifier or base-model name a pack uses
                # wins; the pack model yields.
                if shipped_ids.intersection(entry_routable_ids(model)):
                    continue
                names.add(model["name"])
                entries.append({**model, "origin": {"layer": "pack", "namespace": namespace}})
    return ShippedLayer(entries, frozenset(names), retired, hidden)
