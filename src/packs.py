"""Consumer catalog packs (docs/adr/0002-layered-model-catalog.md).

A product (Home Server, My AI, any program) registers the models it needs that
no release ships, as one snapshot per consumer namespace. Packs load between
the shipped layer and the owner's catalog: they cannot override a shipped
model, and the owner can override or hide a pack model like any other.
``requires`` names the shipped models or base models the product's profiles
need; unmet requirements are reported, never forced on.

Snapshots live in ``catalog-packs.json`` beside ``config.yaml``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src import catalog

_NAMESPACE_RE = re.compile(r"^[a-z0-9-]{1,32}$")
_FIELDS = {"schema_version", "namespace", "source_revision", "models", "requires"}
MAX_MODELS = 100
MAX_REQUIRES = 50


class PackError(ValueError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


def path() -> Path:
    import src.providers as providers

    return providers.CONFIG_PATH.parent / "catalog-packs.json"


def load() -> dict[str, dict]:
    """Registered packs by namespace (empty when none were registered)."""
    target = path()
    if not target.exists():
        return {}
    doc = json.loads(target.read_text())
    packs = doc.get("packs") if isinstance(doc, dict) else None
    if not isinstance(packs, dict) or any(not isinstance(pack, dict) for pack in packs.values()):
        raise ValueError(f"{target} must hold {{\"packs\": {{namespace: pack}}}}")
    return packs


def validate(namespace: str, raw: object, *, shipped_rows: list[dict], other_packs: dict[str, dict]) -> dict:
    """A normalized pack, or :class:`PackError` explaining why it cannot load."""
    def refuse(message: str, code: str = "invalid_pack", status: int = 422):
        raise PackError(status, code, message)

    if not isinstance(raw, dict) or set(raw) - _FIELDS:
        refuse("pack must be an object with schema_version, namespace, source_revision, models, requires")
    if raw.get("schema_version") != 1:
        refuse("pack schema_version must be 1")
    if raw.get("namespace") != namespace or not _NAMESPACE_RE.fullmatch(namespace):
        refuse("pack namespace must match the path")
    revision = raw.get("source_revision")
    if not isinstance(revision, str) or not revision.strip() or len(revision) > 512:
        refuse("pack source_revision must be a non-empty string")
    models, requires = raw.get("models", []), raw.get("requires", [])
    if not isinstance(models, list) or len(models) > MAX_MODELS or any(not isinstance(m, dict) for m in models):
        refuse(f"pack models must be a list of at most {MAX_MODELS} objects")
    if (not isinstance(requires, list) or len(requires) > MAX_REQUIRES
            or any(not isinstance(name, str) or not name.strip() for name in requires)):
        refuse(f"pack requires must be a list of at most {MAX_REQUIRES} model names")

    taken: dict[str, str] = {}
    for row in shipped_rows:
        taken.update(dict.fromkeys(catalog.entry_routable_ids(row), "a shipped model"))
    for other, pack in other_packs.items():
        for row in pack.get("models") or []:
            taken.update(dict.fromkeys(catalog.entry_routable_ids(row), f"pack {other!r}"))
    normalized: list[dict] = []
    for model in models:
        name = model.get("name")
        if not isinstance(name, str) or not name.strip():
            refuse("every pack model needs a name")
        if not model.get("provider") or not (model.get("provider_model_id") or model.get("omlx_id")):
            refuse(f"pack model {name!r} needs provider and provider_model_id or omlx_id")
        if catalog.OWNER_META_FIELDS & set(model):
            refuse(f"pack model {name!r} cannot set origin or override bookkeeping")
        try:
            row = {**model, "provider": catalog.canonical_provider(model["provider"])}
            catalog.validate_pricing_policy(row)
            catalog.validate_client_metadata(row)
            row = catalog.normalize_thinking_capabilities(row)
            ids = catalog.entry_routable_ids(row)
        except ValueError as exc:
            refuse(str(exc))
        for model_id in ids:
            if model_id in taken:
                refuse(f"pack model {name!r} identifier {model_id!r} is used by {taken[model_id]}",
                       code="pack_conflict", status=409)
            taken[model_id] = f"pack model {name!r}"
        normalized.append(row)
    return {"schema_version": 1, "namespace": namespace, "source_revision": revision.strip(),
            "models": normalized, "requires": list(dict.fromkeys(name.strip() for name in requires))}


def write(packs: dict[str, dict]) -> str:
    """Persist every pack (call under the config write lock)."""
    from src import config_io

    target = path()
    config_io._atomic_write(target, json.dumps({"version": 1, "packs": packs}, indent=2, sort_keys=True) + "\n")
    return str(target)
