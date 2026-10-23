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
# What a pack model may describe: its route, capabilities, pricing and labels. Fields
# that reach other consumers' requests or routing (system_instruction, quirks, pool,
# api_style, fallbacks, ...) or make public claims (zdr) stay with the owner.
MODEL_FIELDS = frozenset({
    "name", "provider", "provider_model_id", "omlx_id", "alias", "alternate_ids", "context",
    "max_output_tokens", "thinking", "thinking_levels", "thinking_format", "vision", "pricing",
    "pricing_status", "desc", "display_name", "family", "base_model",
    "params", "active", "format", "quant", "size_gb", "path", "hf_repo",
})
MAX_MODELS = 100
MAX_REQUIRES = 50


class PackError(ValueError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


def path(config_path: Path | None = None) -> Path:
    if config_path is None:
        import src.providers as providers

        config_path = providers.CONFIG_PATH
    return Path(config_path).parent / "catalog-packs.json"


def load(config_path: Path | None = None) -> dict[str, dict]:
    """Registered packs by namespace (empty when none were registered)."""
    target = path(config_path)
    if not target.exists():
        return {}
    doc = json.loads(target.read_text())
    packs = doc.get("packs") if isinstance(doc, dict) else None
    if not isinstance(packs, dict) or any(not isinstance(pack, dict) for pack in packs.values()):
        raise ValueError(f"{target} must hold {{\"packs\": {{namespace: pack}}}}")
    return packs


def _normalized(row: dict) -> dict:
    row = {**row, "provider": catalog.canonical_provider(row.get("provider"))}
    catalog.validate_pricing_policy(row)
    catalog.validate_client_metadata(row)
    return catalog.normalize_thinking_capabilities(row)


def validate(
    namespace: str,
    raw: object,
    *,
    shipped_rows: list[dict],
    owner_rows: list[dict],
    other_packs: dict[str, dict],
) -> dict:
    """A normalized pack, or :class:`PackError` explaining why it cannot load.

    ``shipped_rows`` is every shipped row, loaded here or not: a pack may not reuse
    their identifiers or base-model names, nor another pack's identifiers. Owner
    rows (``owner_rows``) are never a conflict: the owner's rows always win at
    load, so a pack only reserves their base-model names.
    """
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
    for label, rows in (("a shipped model", shipped_rows), ("an owner model", owner_rows)):
        for row in rows:
            if row.get("base_model"):
                taken.setdefault(str(row["base_model"]), f"{label}'s base model")
    for other, pack in other_packs.items():
        for row in pack.get("models") or []:
            taken.update(dict.fromkeys(catalog.entry_routable_ids(row), f"pack {other!r}"))
    normalized: list[dict] = []
    for model in models:
        name = model.get("name")
        if not isinstance(name, str) or not name.strip():
            refuse("every pack model needs a name")
        if set(model) - MODEL_FIELDS:
            refuse(f"pack model {name!r} cannot set: " + ", ".join(sorted(set(model) - MODEL_FIELDS)))
        if not model.get("provider") or not (model.get("provider_model_id") or model.get("omlx_id")):
            refuse(f"pack model {name!r} needs provider and provider_model_id or omlx_id")
        try:
            row = _normalized(model)
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
