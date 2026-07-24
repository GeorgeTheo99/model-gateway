"""Safe read/write layer for model-gateway config files.

Two files, two deploy models (see docs/provider-pricing-sources.md and the
productionization plan):

- ``config.yaml`` (providers + auth): deployed copy is a symlink to the shared
  gitignored file. Edits are hot (after reload) and durable across deploys.
  Holds secrets — never echo ``api_key`` back; writes are additive/preserving.
- ``model-info.json`` (models): machine-local and Git-ignored. Writes edit the
  live copy (hot) and, when configured, a second machine-local mirror
  (``MODEL_INFO_SOURCE_PATH``) used by deploy tooling or private backup flows.

All writes: validate -> backup -> atomic temp+rename. Best-effort; a write
failure raises a clear error to the admin API caller.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from functools import wraps
from pathlib import Path
from typing import Any

import yaml

from src.catalog import normalize_thinking_capabilities, validate_pricing_policy
from src.config_lock import config_write_lock
from src.providers import CONFIG_PATH, MODEL_INFO_PATH, MODEL_INFO_SOURCE_PATH

log_dir = Path(os.environ.get("MODEL_GATEWAY_LOG_DIR", str(Path.home() / ".claude")))


def _write_transaction(function):
    """Hold the shared lock across a complete read-modify-write operation."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with config_write_lock(CONFIG_PATH):
            return function(*args, **kwargs)
    return wrapped


# ── provider config (config.yaml) ───────────────────────────────────────────


def load_config_full() -> dict:
    """Load the full config.yaml (all sections, including auth/providers)."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _resolve_target(path: Path) -> Path:
    """Resolve symlinks so writes land on the real file, not replace the link.

    The deployed config.yaml is a symlink to the shared gitignored file. A
    naive os.replace would write a real file at the link location, breaking
    the link and splitting config state. Resolve to the real target first.
    """
    try:
        return Path(os.path.realpath(path))
    except OSError:
        return path


def _atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically: temp file in same dir, then rename.

    Resolves symlinks first so a symlinked config file (deployed config.yaml)
    is updated in place rather than replaced with a real file.
    """
    target = _resolve_target(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode & 0o777 if target.exists() else 0o600
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _backup(path: Path) -> Path | None:
    """Copy path to a timestamped .bak in the log dir; return the backup path."""
    if not path.exists():
        return None
    backup_dir = log_dir / "config-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    bak = backup_dir / f"{path.name}.bak.{int(time.time())}"
    shutil.copy2(path, bak)
    return bak


def snapshot_writable_files() -> dict[Path, str | None]:
    """Capture every admin-managed file for validation rollback."""
    paths = {CONFIG_PATH, MODEL_INFO_PATH}
    if MODEL_INFO_SOURCE_PATH:
        paths.add(MODEL_INFO_SOURCE_PATH)
    snapshot = {}
    for path in paths:
        target = _resolve_target(path)
        snapshot[path] = target.read_text() if target.exists() else None
    return snapshot


def restore_writable_files(snapshot: dict[Path, str | None]) -> None:
    """Atomically restore files captured by :func:`snapshot_writable_files`."""
    with config_write_lock(CONFIG_PATH):
        for path, text in snapshot.items():
            if text is None:
                _resolve_target(path).unlink(missing_ok=True)
            else:
                _atomic_write(path, text)


@_write_transaction
def upsert_provider(
    provider_id: str,
    *,
    base_url: str,
    api_key: str | None = None,
    protocol: str | None = None,
    default_headers: dict | None = None,
) -> dict:
    """Create or update a provider block in config.yaml.

    ``api_key`` is write-only: if None, the existing key is preserved; if an
    empty string, it is removed. Other fields are set only when provided.
    Returns the masked provider status dict (no secrets).
    """
    provider_id = (provider_id or "").strip().lower()
    if not provider_id:
        raise ValueError("provider_id is required")
    if not base_url:
        raise ValueError("base_url is required")

    config = load_config_full()
    providers = config.setdefault("providers", {}) or {}
    # Normalize: store under the canonical id. If a synonym key exists, update
    # it in place; otherwise create under provider_id.
    existing = providers.get(provider_id)
    block: dict = dict(existing) if isinstance(existing, dict) else {}

    block["base_url"] = base_url
    if protocol is not None:
        block["protocol"] = protocol
    if default_headers is not None:
        block["default_headers"] = default_headers
    if api_key is not None:
        if api_key == "":
            block.pop("api_key", None)
        else:
            block["api_key"] = api_key
    providers[provider_id] = block
    config["providers"] = providers

    _backup(CONFIG_PATH)
    _atomic_write(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    return _masked_block(provider_id, block)


@_write_transaction
def delete_provider(provider_id: str) -> dict:
    """Remove a provider from config.yaml. Refuses if models depend on it."""
    from src.providers import _load_models  # local import to avoid cycle at load

    provider_id = (provider_id or "").strip().lower()
    config = load_config_full()
    providers = config.get("providers", {}) or {}
    if provider_id not in providers and not any(
        k.lower() == provider_id for k in providers
    ):
        raise KeyError(f"provider {provider_id!r} not found")

    # Refuse if any enabled model routes to this provider.
    from src.providers import _canonical_provider, _is_model_enabled
    dependents = []
    for entry in {id(v): v for v in _load_models().values()}.values():
        if _canonical_provider(entry.get("provider")) == provider_id and _is_model_enabled(entry.get("name")):
            dependents.append(entry.get("name", ""))
    if dependents:
        raise ValueError(
            f"cannot delete provider {provider_id!r}: {len(dependents)} enabled "
            f"model(s) depend on it: {', '.join(sorted(set(dependents)))}. "
            "Disable or reassign them first."
        )

    # Remove the key (and any synonym key).
    for k in list(providers.keys()):
        if k.lower() == provider_id:
            del providers[k]
    config["providers"] = providers
    _backup(CONFIG_PATH)
    _atomic_write(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    return {"id": provider_id, "deleted": True}


def _masked_block(provider_id: str, block: dict) -> dict:
    from src.providers import _safe_url

    return {
        "id": provider_id,
        "base_url": _safe_url(block.get("base_url", "")),
        "protocol": block.get("protocol", "openai"),
        "has_api_key": bool(block.get("api_key")),
        "default_headers": bool(block.get("default_headers")),
    }


# ── model config (model-info.json) ──────────────────────────────────────────


def load_model_info() -> dict:
    """Load the full model-info.json document."""
    if not MODEL_INFO_PATH.exists():
        return {"llm": []}
    with open(MODEL_INFO_PATH) as f:
        return json.load(f)


def _write_model_info(doc: dict) -> list[str]:
    """Write model-info.json to the live copy and optional local mirror.

    Returns the list of paths actually written.
    """
    text = json.dumps(doc, indent=2) + "\n"
    paths = [MODEL_INFO_PATH]
    _backup(MODEL_INFO_PATH)
    _atomic_write(MODEL_INFO_PATH, text)
    if MODEL_INFO_SOURCE_PATH and MODEL_INFO_SOURCE_PATH != MODEL_INFO_PATH:
        try:
            _atomic_write(MODEL_INFO_SOURCE_PATH, text)
            paths.append(MODEL_INFO_SOURCE_PATH)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not mirror model-info.json to %s: %s", MODEL_INFO_SOURCE_PATH, exc)
    return [str(p) for p in paths]


# Fields a cloud model entry may carry. Used to validate + order writes.
# NOTE: "enabled" is intentionally absent — runtime state lives in
# config.yaml model_overrides, not the machine-local catalog.
_MODEL_FIELDS = [
    "name", "provider", "provider_model_id", "omlx_id", "alias", "context",
    "max_output_tokens", "thinking", "thinking_levels", "thinking_format", "vision", "quirks",
    "system_instruction", "pricing", "pricing_status", "desc",
]
_PRICING_RATE_FIELDS = {
    "input", "output", "cache_read", "cache_write", "cache_write_1h", "reasoning",
}
_LOCAL_PROVIDERS = {"local", "omlx", "mlx"}


def _validated_pricing(pricing: object) -> dict:
    if not isinstance(pricing, dict) or not pricing:
        raise ValueError("metered pricing must be a non-empty object")
    unknown = set(pricing) - _PRICING_RATE_FIELDS
    if unknown:
        raise ValueError("unknown pricing field(s): " + ", ".join(sorted(unknown)))
    missing = {"input", "output"} - set(pricing)
    if missing:
        raise ValueError("metered pricing requires: " + ", ".join(sorted(missing)))
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        for value in pricing.values()
    ):
        raise ValueError("pricing values must be finite non-negative numbers")
    return dict(pricing)


def _apply_pricing_update(entry: dict, fields: dict, provider: str) -> None:
    """Apply the explicit metered/unmetered/unknown pricing policy."""
    if "pricing" not in fields and "pricing_status" not in fields:
        return

    status = fields.get("pricing_status")
    if status is not None:
        status = str(status).strip().lower()
    if status not in {None, "metered", "unmetered", "unknown"}:
        raise ValueError("pricing_status must be metered, unmetered, or unknown")

    supplied_pricing = fields.get("pricing") if "pricing" in fields else entry.get("pricing")
    if status == "unmetered":
        if provider not in _LOCAL_PROVIDERS:
            raise ValueError("unmetered pricing is only valid for local/oMLX models")
        if supplied_pricing:
            raise ValueError("unmetered models cannot also define token prices")
        entry.pop("pricing", None)
        entry["pricing_status"] = "unmetered"
        return
    if status == "unknown" or (status is None and "pricing" in fields and supplied_pricing is None):
        entry.pop("pricing", None)
        entry.pop("pricing_status", None)
        return

    if status == "metered" or "pricing" in fields:
        entry["pricing"] = _validated_pricing(supplied_pricing)
        entry.pop("pricing_status", None)


@_write_transaction
def upsert_model(name: str, **fields) -> dict:
    """Create or update a model entry in model-info.json.

    ``name`` is the gateway-facing model id and the dict key. Returns the
    written entry (masked: no secrets; models carry none). Writes the live
    catalog and optional machine-local mirror.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("model name is required")
    provider = (fields.get("provider") or "").strip().lower()
    if not provider:
        raise ValueError("provider is required")
    provider_model_id = (fields.get("provider_model_id") or "").strip()
    omlx_id = (fields.get("omlx_id") or "").strip()
    if provider in {"local", "omlx", "mlx"}:
        if not omlx_id and not provider_model_id:
            raise ValueError("omlx_id or provider_model_id is required for local/oMLX models")
    elif not provider_model_id:
        raise ValueError("provider_model_id is required")

    doc = load_model_info()
    llm = doc.get("llm", [])
    entry = next((e for e in llm if e.get("name") == name), None)
    if entry is None:
        entry = {"name": name}
        llm.append(entry)
        doc["llm"] = llm

    entry["name"] = name
    entry["provider"] = provider
    if provider_model_id:
        entry["provider_model_id"] = provider_model_id
    else:
        entry.pop("provider_model_id", None)
    if omlx_id:
        entry["omlx_id"] = omlx_id
    # JSON null means "leave unchanged" for optional admin fields. In
    # particular it must not discard a narrow explicit thinking_levels list and
    # replace it with the broad legacy fallback for the unchanged mode.
    thinking_changed = fields.get("thinking") is not None and fields["thinking"] != entry.get("thinking", "")
    if thinking_changed and "thinking_levels" not in fields:
        # A mode change without an explicit level list requests the safe legacy
        # fallback for that new mode rather than retaining stale capabilities.
        entry.pop("thinking_levels", None)
    for f in ("alias", "context", "max_output_tokens", "thinking",
              "thinking_levels", "thinking_format", "quirks", "system_instruction", "desc"):
        if f in fields and fields[f] is not None:
            entry[f] = fields[f]
    _apply_pricing_update(entry, fields, provider)
    validate_pricing_policy(entry)
    entry.update(normalize_thinking_capabilities(entry))
    if "vision" in fields and fields["vision"] is not None:
        entry["vision"] = bool(fields["vision"])
    # "enabled" is handled by set_model_enabled() writing config.yaml
    # model_overrides; it is never written to the model catalog.

    paths = _write_model_info(doc)
    return {"name": name, "entry": _model_summary(entry), "written_to": paths}


@_write_transaction
def delete_model(name: str) -> dict:
    """Remove a model entry by name. Also clears any runtime override."""
    name = (name or "").strip()
    doc = load_model_info()
    llm = doc.get("llm", [])
    before = len(llm)
    llm = [e for e in llm if e.get("name") != name]
    if len(llm) == before:
        raise KeyError(f"model {name!r} not found")
    doc["llm"] = llm
    paths = _write_model_info(doc)
    # Clean up any stale runtime override for the deleted model.
    config = load_config_full()
    overrides = config.get("model_overrides") or {}
    if name in overrides:
        del overrides[name]
        config["model_overrides"] = overrides
        _backup(CONFIG_PATH)
        _atomic_write(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
        paths.append(str(CONFIG_PATH))
    return {"name": name, "deleted": True, "written_to": paths}


@_write_transaction
def set_model_enabled(name: str, enabled: bool) -> dict:
    """Toggle a model's runtime enabled state in config.yaml model_overrides.

    Writes to the Git-ignored config.yaml (symlinked shared file), not
    model-info.json — so the toggle is hot, durable across deploys,
    and doesn't dirty the repo. The catalog stays the source of truth for
    *what models exist*; this is purely runtime on/off state.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("model name is required")
    # Verify the model exists in the catalog before recording an override.
    doc = load_model_info()
    if not any(e.get("name") == name for e in doc.get("llm", [])):
        raise KeyError(f"model {name!r} not found")

    config = load_config_full()
    overrides = config.setdefault("model_overrides", {}) or {}
    overrides[name] = {"enabled": bool(enabled)}
    config["model_overrides"] = overrides
    _backup(CONFIG_PATH)
    _atomic_write(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    return {"name": name, "enabled": bool(enabled), "written_to": [str(CONFIG_PATH)]}


def _model_summary(entry: dict) -> dict:
    """Mask-free summary of a model entry (models carry no secrets)."""
    return {
        k: entry.get(k) for k in _MODEL_FIELDS if k in entry
    }
