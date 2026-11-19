"""models.dev directory: sync, snapshot, and mapping to catalog rows (docs/adr/0005-models-dev-directory.md).

The directory is data, not catalog rows. A sync fetches models.dev's v1 ``api.json`` and
``models.json``, keeps only the providers the gateway can map, validates the result against
the last good snapshot and stores it beside ``config.yaml``. :func:`map_model` turns one
directory model into a catalog row, applying the gateway's own facts from
``config/directory/corrections.yaml``. No LLM runs here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx
import yaml

from src.catalog import ENABLED_THINKING_LEVELS

log = logging.getLogger("model-gateway")

API_URL = "https://models.dev/api.json"
LABS_URL = "https://models.dev/models.json"
SNAPSHOT_SCHEMA = 1
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 60.0
# A sync that loses more than this share of the mapped models at once is rejected.
MAX_DISAPPEARED_SHARE = 0.2
SYNC_INTERVAL_SECONDS = 24 * 60 * 60

_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "config" / "directory"
BUNDLED_SNAPSHOT = _PACKAGE_DIR / "models-dev-snapshot.json"
CORRECTIONS_PATH = _PACKAGE_DIR / "corrections.yaml"

# Fields kept per directory model; everything else in models.dev is dropped from the snapshot.
_MODEL_FIELDS = (
    "id", "name", "family", "type", "reasoning", "reasoning_options", "tool_call", "modalities",
    "limit", "cost", "release_date", "status", "canonical_model_id",
)
_LAB_FIELDS = ("name", "family", "release_date", "status")
_SKIPPED_ID = re.compile(r"(?:^|[-_/.])(?:preview|exp|experimental|latest)(?:$|[-_/.])")
_SLUG_OK = re.compile(r"^[a-z0-9][a-z0-9.-]{0,63}$")


class DirectoryError(ValueError):
    """A snapshot or sync result the gateway refuses."""


# ── Corrections ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Corrections:
    providers: dict[str, dict]
    """Gateway provider id -> provider facts (models_dev id, display name, suffix, …)."""
    labs: dict[str, dict]
    """models.dev lab id -> lab facts (home provider, family)."""
    rules: list[dict] = field(default_factory=list)

    @property
    def by_models_dev_id(self) -> dict[str, str]:
        return {facts["models_dev"]: provider for provider, facts in self.providers.items()}


def load_corrections(path: Path = CORRECTIONS_PATH) -> Corrections:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("schema_version") != 1:
        raise DirectoryError(f"{path}: schema_version must be 1")
    providers = data.get("providers") or {}
    labs = data.get("labs") or {}
    rules = data.get("rules") or []
    if not isinstance(providers, dict) or not isinstance(labs, dict) or not isinstance(rules, list):
        raise DirectoryError(f"{path}: providers and labs must be objects, rules a list")
    for provider, facts in providers.items():
        if not isinstance(facts, dict) or not facts.get("models_dev") or not facts.get("display_name"):
            raise DirectoryError(f"{path}: provider {provider!r} needs models_dev and display_name")
    for rule in rules:
        match = rule.get("match") if isinstance(rule, dict) else None
        if not isinstance(match, dict) or not match or set(match) - {"provider", "lab", "id", "id_prefix"}:
            raise DirectoryError(f"{path}: every rule needs a match on provider, lab, id or id_prefix")
        if set(rule) - {"match", "set", "quirks", "drop_pricing", "extended_cache_write", "max_output_tokens",
                        "route_suffix"}:
            raise DirectoryError(f"{path}: rule {match} has unknown field(s)")
    return Corrections(providers=providers, labs=labs, rules=rules)


# ── Snapshot ───────────────────────────────────────────────────────────────


def _trim_model(model: dict) -> dict:
    return {key: model[key] for key in _MODEL_FIELDS if key in model}


def build_snapshot(api: dict, labs: dict, corrections: Corrections, *, fetched_at: str | None = None,
                   source_sha256: str = "") -> dict:
    """Keep the mappable providers of a models.dev ``api.json``, plus lab names they reference."""
    if not isinstance(api, dict) or not isinstance(labs, dict):
        raise DirectoryError("models.dev returned an unexpected shape")
    providers: dict[str, dict] = {}
    referenced: set[str] = set()
    for dev_id in sorted(corrections.by_models_dev_id):
        entry = api.get(dev_id)
        if not isinstance(entry, dict) or not isinstance(entry.get("models"), dict):
            continue
        models = {}
        for model_id, model in entry["models"].items():
            if isinstance(model, dict) and isinstance(model.get("id"), str):
                models[model_id] = _trim_model(model)
                if isinstance(model.get("canonical_model_id"), str):
                    referenced.add(model["canonical_model_id"])
        providers[dev_id] = {"name": entry.get("name", dev_id), "models": models}
    lab_models = {cid: {key: labs[cid][key] for key in _LAB_FIELDS if key in labs[cid]}
                  for cid in sorted(referenced) if isinstance(labs.get(cid), dict)}
    return {
        "schema": SNAPSHOT_SCHEMA,
        "source": {"api": API_URL, "labs": LABS_URL, "sha256": source_sha256},
        "fetched_at": fetched_at or _utc_now(),
        "providers": providers,
        "labs": lab_models,
    }


def model_count(snapshot: dict) -> int:
    return sum(len(entry.get("models") or {}) for entry in (snapshot.get("providers") or {}).values())


def validate_update(new: dict, old: dict | None) -> None:
    """Refuse a sync that lost a provider or too many models at once (ADR 0005)."""
    if new.get("schema") != SNAPSHOT_SCHEMA or not new.get("providers"):
        raise DirectoryError("snapshot has no providers")
    if not old:
        return
    for dev_id, entry in (old.get("providers") or {}).items():
        if entry.get("models") and not (new["providers"].get(dev_id) or {}).get("models"):
            raise DirectoryError(f"models.dev lists no models for {dev_id!r} any more")
    old_keys = {(dev_id, model_id) for dev_id, entry in old["providers"].items() for model_id in entry["models"]}
    new_keys = {(dev_id, model_id) for dev_id, entry in new["providers"].items() for model_id in entry["models"]}
    gone = len(old_keys - new_keys)
    if old_keys and gone / len(old_keys) > MAX_DISAPPEARED_SHARE:
        raise DirectoryError(f"{gone} of {len(old_keys)} models would disappear at once")


def directory_dir() -> Path:
    """Runtime directory state, beside ``config.yaml`` (never the release's ``config/directory``)."""
    from src import providers

    return Path(providers.CONFIG_PATH).parent / "models-dev"


def snapshot_path() -> Path:
    return directory_dir() / "snapshot.json"


def state_path() -> Path:
    return directory_dir() / "state.json"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("directory: unreadable %s: %s", path, exc)
        return None
    return value if isinstance(value, dict) else None


def load_snapshot() -> tuple[dict | None, str]:
    """The last good synced snapshot, else the release's bundled one; with which it is."""
    synced = _read_json(snapshot_path())
    if synced and synced.get("schema") == SNAPSHOT_SCHEMA:
        return synced, "synced"
    bundled = _read_json(BUNDLED_SNAPSHOT)
    if bundled and bundled.get("schema") == SNAPSHOT_SCHEMA:
        return bundled, "bundled"
    return None, "none"


def load_state() -> dict:
    return _read_json(state_path()) or {}


# ── Settings ───────────────────────────────────────────────────────────────


def settings(config: dict) -> dict:
    """Validated ``catalog.directory`` settings: ``{"sync": "daily" | "off"}``."""
    raw = (config.get("catalog") or {}).get("directory")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("config catalog.directory must be an object")
    unknown = set(raw) - {"sync"}
    if unknown:
        raise ValueError("config catalog.directory has unknown field(s): " + ", ".join(sorted(unknown)))
    sync_mode = raw.get("sync", "daily")
    if sync_mode not in ("daily", "off"):
        raise ValueError("config catalog.directory.sync must be daily or off")
    return {"sync": sync_mode}


# ── Sync ───────────────────────────────────────────────────────────────────


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _download(url: str) -> bytes:
    with httpx.Client(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True,
                      headers={"User-Agent": "model-gateway-directory/1", "Accept": "application/json"}) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise DirectoryError(f"{url} is larger than {MAX_DOWNLOAD_BYTES} bytes")
                chunks.append(chunk)
            return b"".join(chunks)


def sync(*, fetch: Callable[[str], bytes] = _download, corrections: Corrections | None = None) -> dict:
    """Fetch models.dev and store the snapshot if it passes validation. Returns the new state."""
    corrections = corrections or load_corrections()
    state = load_state()
    state["last_attempt"] = _utc_now()
    try:
        api_bytes = fetch(API_URL)
        labs_bytes = fetch(LABS_URL)
        try:
            api, labs = json.loads(api_bytes), json.loads(labs_bytes)
        except ValueError as exc:
            raise DirectoryError(f"models.dev returned invalid JSON: {exc}") from exc
        digest = hashlib.sha256(api_bytes).hexdigest()
        snapshot = build_snapshot(api, labs, corrections, source_sha256=digest)
        previous, _ = load_snapshot()
        validate_update(snapshot, previous)
        if previous and previous.get("source", {}).get("sha256") == digest:
            state.update(last_success=state["last_attempt"], error=None, changed=False)
        else:
            _atomic_json(snapshot_path(), snapshot)
            state.update(last_success=state["last_attempt"], error=None, changed=True,
                         models=model_count(snapshot), sha256=digest)
    except (DirectoryError, httpx.HTTPError, OSError) as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        log.warning("directory sync failed: %s", state["error"])
    _atomic_json(state_path(), state)
    return state


def sync_due(state: dict, now: float | None = None, interval: float = SYNC_INTERVAL_SECONDS) -> bool:
    last = state.get("last_attempt")
    if not last:
        return True
    try:
        then = datetime.fromisoformat(last.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return True
    return (now if now is not None else time.time()) - then >= interval


# ── Mapping ────────────────────────────────────────────────────────────────


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-.")
    return re.sub(r"-{2,}", "-", value)


def line_key(lab: str, lab_name: str) -> str:
    """A model line: the lab plus the version-free words of the name (ADR 0005)."""
    words = [word for word in re.split(r"[\s\-_/()]+", lab_name.lower()) if word and not re.search(r"\d", word)]
    return f"{lab}/{' '.join(words)}"


@dataclass(frozen=True)
class Mapped:
    provider: str
    """Gateway provider id."""
    model_id: str
    canonical: str
    lab: str
    line: str
    release_date: str
    row: dict | None
    skipped: str | None = None
    notes: tuple[str, ...] = ()
    status: str = ""

    @property
    def followable(self) -> bool:
        """Whether this model can be a line's current model: not preview, experimental or alpha/beta."""
        return (self.row is not None and self.status not in ("alpha", "beta")
                and not _SKIPPED_ID.search(self.model_id.lower()) and not _SKIPPED_ID.search(self.canonical.lower()))


def _rules_for(corrections: Corrections, provider: str, lab: str, model_id: str) -> list[dict]:
    matched = []
    for rule in corrections.rules:
        match = rule["match"]
        if "provider" in match and match["provider"] != provider:
            continue
        if "lab" in match and match["lab"] != lab:
            continue
        if "id" in match and match["id"] != model_id:
            continue
        if "id_prefix" in match and not model_id.startswith(match["id_prefix"]):
            continue
        matched.append(rule)
    return matched


def _thinking(model: dict) -> tuple[str, list[str] | None]:
    if not model.get("reasoning"):
        return "", None
    options = model.get("reasoning_options") or []
    toggle = any(isinstance(option, dict) and option.get("type") == "toggle" for option in options)
    efforts: list[str] = []
    for option in options:
        if isinstance(option, dict) and option.get("type") == "effort":
            efforts.extend(value for value in option.get("values") or [] if isinstance(value, str))
    optional = toggle or "none" in efforts
    levels = [level for level in ENABLED_THINKING_LEVELS if level in efforts]
    if not levels:
        levels = ["high"] if toggle else ["low", "medium", "high"]
    return ("optional", ["off", *levels]) if optional else ("always", levels)


def _pricing(cost: dict, notes: list[str]) -> dict | None:
    if not isinstance(cost, dict) or not all(isinstance(cost.get(key), (int, float)) for key in ("input", "output")):
        return None
    classes = ("input", "output", "cache_read", "cache_write")
    pricing = {key: float(cost[key]) for key in classes if isinstance(cost.get(key), (int, float))}
    tiers = [tier for tier in cost.get("tiers") or [] if isinstance(tier, dict)]
    context_tiers = [tier for tier in tiers if (tier.get("tier") or {}).get("type") == "context"]
    if len(context_tiers) == 1 and len(tiers) == 1:
        tier = context_tiers[0]
        size = tier["tier"].get("size")
        rates = {key: float(tier[key]) for key in classes if isinstance(tier.get(key), (int, float))}
        if isinstance(size, int) and size > 0 and set(rates) == set(pricing):
            pricing["long_context"] = {"above_prompt_tokens": size, **rates}
        else:
            notes.append("long_context_tier_unusable")
    elif tiers:
        notes.append("pricing_tiers_unsupported")
    return pricing


def map_model(dev_provider: str, model: dict, snapshot: dict, corrections: Corrections,
              base_names: dict[str, str] | None = None) -> Mapped:
    """Turn one directory model into a catalog row, or say why it is not offered."""
    provider = corrections.by_models_dev_id[dev_provider]
    facts = corrections.providers[provider]
    model_id = model["id"]
    canonical = model.get("canonical_model_id") or f"{dev_provider}/{model_id}"
    lab = canonical.split("/", 1)[0]
    lab_entry = (snapshot.get("labs") or {}).get(canonical) or {}
    lab_name = lab_entry.get("name") or model.get("name") or model_id
    release = str(lab_entry.get("release_date") or model.get("release_date") or "")
    line = line_key(lab, lab_name)

    status = str(model.get("status") or "")

    def skipped(reason: str) -> Mapped:
        return Mapped(provider, model_id, canonical, lab, line, release, None, reason, status=status)

    if model.get("type") not in (None, "chat"):
        return skipped("not_chat")
    if "text" not in ((model.get("modalities") or {}).get("output") or []):
        return skipped("no_text_output")
    if status == "deprecated":
        return skipped("deprecated")
    notes: list[str] = []
    pricing = _pricing(model.get("cost") or {}, notes)
    if pricing is None:
        return skipped("no_price")
    limit = model.get("limit") or {}
    if not isinstance(limit.get("context"), int) or not isinstance(limit.get("output"), int):
        return skipped("no_limits")

    base = (base_names or {}).get(canonical) or slug(lab_name)
    if not _SLUG_OK.match(base):
        return skipped("unusable_name")
    lab_facts = corrections.labs.get(lab) or {}
    rules = _rules_for(corrections, provider, lab, model_id)
    suffix = facts.get("route_suffix", "")
    for rule in rules:
        suffix = rule.get("route_suffix", suffix)
    name = base if lab_facts.get("home") == provider else base + suffix
    thinking, levels = _thinking(model)
    row: dict = {
        "name": name,
        "provider": provider,
        "provider_model_id": model_id,
        "display_name": f"{lab_name} ({facts['display_name']})"[:80],
        "family": lab_facts.get("family") or slug(lab),
        "base_model": base,
        "context": limit["context"],
        "max_output_tokens": limit["output"],
        "vision": "image" in ((model.get("modalities") or {}).get("input") or []),
        "pricing": pricing,
        "desc": f"{lab_name} via {facts['display_name']} (models.dev)",
    }
    if thinking:
        row["thinking"], row["thinking_levels"] = thinking, levels
    if facts.get("thinking_format") and thinking:
        row["thinking_format"] = facts["thinking_format"]
    quirks = list(facts.get("quirks") or [])
    if facts.get("zdr"):
        row["zdr"] = True

    for rule in rules:
        for key in rule.get("drop_pricing") or []:
            pricing.pop(key, None)
            (pricing.get("long_context") or {}).pop(key, None)
        multiplier = rule.get("extended_cache_write")
        if multiplier is not None and "cache_write" in pricing:
            pricing["cache_write_1h"] = round(pricing["input"] * float(multiplier), 6)
            tier = pricing.get("long_context")
            if tier is not None:
                tier["cache_write_1h"] = round(tier["input"] * float(multiplier), 6)
        if rule.get("max_output_tokens"):
            row["max_output_tokens"] = min(row["max_output_tokens"], int(rule["max_output_tokens"]))
        quirks.extend(quirk for quirk in rule.get("quirks") or [] if quirk not in quirks)
        row.update(rule.get("set") or {})
    if quirks:
        row["quirks"] = quirks
    if not row.get("thinking"):
        row.pop("thinking_levels", None)
    return Mapped(provider, model_id, canonical, lab, line, release, row, None, tuple(notes), status)


def base_names(snapshot: dict) -> dict[str, str]:
    """Lab model id -> base-model name; a name two lab models share falls back to the lab id."""
    by_slug: dict[str, set[str]] = {}
    for dev_id, entry in (snapshot.get("providers") or {}).items():
        for model in entry["models"].values():
            canonical = model.get("canonical_model_id") or f"{dev_id}/{model['id']}"
            lab_name = ((snapshot.get("labs") or {}).get(canonical) or {}).get("name") or model.get("name") or ""
            by_slug.setdefault(slug(lab_name), set()).add(canonical)
    names = {}
    for name, canonicals in by_slug.items():
        for canonical in canonicals:
            names[canonical] = name if len(canonicals) == 1 else slug(canonical.split("/", 1)[1])
    return names


def _alias_like(model_id: str) -> bool:
    return bool(_SKIPPED_ID.search(model_id.lower())) or "/routers/" in model_id


def map_snapshot(snapshot: dict, corrections: Corrections) -> list[Mapped]:
    """Every directory model of a snapshot, mapped or with the reason it is not offered.

    Route names are unique: when one provider serves a lab model under several ids (a dated
    snapshot, a ``-latest`` router), the plain id keeps the name and the others are named
    after their own id. A name two providers would share is a corrections bug; the later
    one is not offered.
    """
    names = base_names(snapshot)
    mapped = []
    for dev_id, entry in sorted((snapshot.get("providers") or {}).items()):
        if dev_id not in corrections.by_models_dev_id:
            continue
        for model_id in sorted(entry["models"]):
            mapped.append(map_model(dev_id, entry["models"][model_id], snapshot, corrections, names))
    claimed: dict[str, Mapped] = {}
    result = []
    def precedence(m: Mapped) -> tuple:
        own_id = m.row is not None and slug(m.model_id.rsplit("/", 1)[-1]) in (m.row["base_model"], m.row["name"])
        return (m.row is None, m.provider, not own_id, _alias_like(m.model_id), m.model_id)

    for item in sorted(mapped, key=precedence):
        if item.row is not None and item.row["name"] in claimed:
            owner = claimed[item.row["name"]]
            renamed = None
            if owner.provider == item.provider:
                suffix = item.row["name"][len(item.row["base_model"]):]
                candidate = slug(item.model_id.rsplit("/", 1)[-1]) + suffix
                if _SLUG_OK.match(candidate) and candidate not in claimed:
                    renamed = replace(item, row={**item.row, "name": candidate})
            item = renamed or replace(item, row=None, skipped="name_conflict")
        if item.row is not None:
            claimed[item.row["name"]] = item
        result.append(item)
    return result


def current_models(mapped: list[Mapped]) -> dict[str, str]:
    """Line -> the lab model id of its newest followable model."""
    newest: dict[str, tuple[str, str]] = {}
    for item in mapped:
        if not item.followable:
            continue
        best = newest.get(item.line)
        if best is None or (item.release_date, item.canonical) > best:
            newest[item.line] = (item.release_date, item.canonical)
    return {line: canonical for line, (_, canonical) in newest.items()}
