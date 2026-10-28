"""Reasoning tiers: a client says how hard a request is, the gateway picks the model's level.

A client that classifies its own requests sends ``model_gateway.reasoning_tier``
instead of a concrete level, so it never has to know each model's levels. The
tiers place a request on the model's own ordered levels (docs/reasoning-tiers.md):

- ``none``: off when the model can turn thinking off, else its cheapest level;
- ``light``: its cheapest enabled level;
- ``normal``: its default enabled level (``default_enabled_level``);
- ``hard``: its deepest level.

A model with one level always gets it. ``TIER_POLICY`` names these semantics;
any change to them is a new policy name, never a silent change to this one.
"""
from __future__ import annotations

from collections.abc import Iterable

from src import catalog

TIER_POLICY = "tier-v1"
REASONING_TIERS = ("none", "light", "normal", "hard")


def tier_level(tier: str, thinking: str, levels: Iterable[str], default: str | None) -> str | None:
    """The concrete level for one tier, or None when the model has no thinking control."""
    declared = set(levels)
    ordered = [level for level in catalog.THINKING_LEVELS if level in declared]
    enabled = [level for level in ordered if level != "off"]
    if not thinking or not enabled:
        return None
    if tier == "none" and "off" in ordered:
        return "off"
    if tier in ("none", "light"):
        return enabled[0]
    if tier == "hard":
        return enabled[-1]
    return default if default in enabled else enabled[-1]


def tier_map(thinking: str, levels: Iterable[str], default: str | None) -> dict[str, str] | None:
    """Every tier's level for one model, as advertised in discovery; None without thinking."""
    levels = tuple(levels)
    if tier_level("normal", thinking, levels, default) is None:
        return None
    return {tier: tier_level(tier, thinking, levels, default) for tier in REASONING_TIERS}
