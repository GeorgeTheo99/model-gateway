"""Transactional onboarding for OpenAI-compatible providers and models.

Profiles are tracked, secret-free YAML files. Applying one validates the
upstream model list, writes a provider secret to a mode-0600 file, updates the
runtime config and model catalog together, and restores every touched file if
any write fails.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import httpx
import yaml

from src.catalog import entry_routable_ids, validate_pricing_policy
from src.config_lock import config_write_lock
from src.secret_files import read_api_key_file, resolve_api_key_file


class OnboardingError(ValueError):
    """A profile or onboarding operation is unsafe or invalid."""


def load_profile(path: Path) -> dict:
    return validate_profile(yaml.safe_load(path.read_text()) or {})


def validate_profile(data: object) -> dict:
    """Validate a parsed profile for both CLI and direct API callers."""
    if not isinstance(data, dict):
        raise OnboardingError("profile root must be an object")
    if data.get("schema_version") != 1:
        raise OnboardingError("profile schema_version must be 1")
    provider = data.get("provider")
    models = data.get("models")
    if not isinstance(provider, dict) or not provider.get("id") or not provider.get("base_url"):
        raise OnboardingError("profile provider requires id and base_url")
    provider_id = str(provider["id"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", provider_id):
        raise OnboardingError("provider id must contain only lowercase letters, digits, underscores, and hyphens")
    parsed_url = urlsplit(str(provider["base_url"]))
    if (
        parsed_url.scheme != "https"
        or not parsed_url.netloc
        or parsed_url.username
        or parsed_url.password
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise OnboardingError(
            "provider base_url must be an HTTPS URL without credentials, query, or fragment"
        )
    if not isinstance(models, list) or not models:
        raise OnboardingError("profile requires at least one model")
    names: set[str] = set()
    identifiers: dict[str, str] = {}
    for model in models:
        if not isinstance(model, dict):
            raise OnboardingError("every profile model must be an object")
        for field in ("name", "provider", "provider_model_id"):
            value = model.get(field)
            if not isinstance(value, str) or not value.strip():
                raise OnboardingError(f"profile model requires non-empty string {field}")
        name = model["name"]
        if name in names:
            raise OnboardingError(f"duplicate profile model name: {name}")
        names.add(name)
        try:
            routable_ids = entry_routable_ids(model)
        except ValueError as exc:
            raise OnboardingError(str(exc)) from exc
        for identifier in routable_ids:
            owner = identifiers.get(identifier)
            if owner and owner != name:
                raise OnboardingError(f"duplicate profile model identifier: {identifier}")
            identifiers[identifier] = name
        if model["provider"] != provider["id"]:
            raise OnboardingError(
                f"model {model['name']!r} provider must match profile provider {provider['id']!r}"
            )
        for field in ("context", "max_output_tokens"):
            if field in model and (not isinstance(model[field], int) or model[field] <= 0):
                raise OnboardingError(f"model {model['name']!r} {field} must be a positive integer")
        pricing = model.get("pricing")
        if pricing is not None:
            if not isinstance(pricing, dict) or not pricing:
                raise OnboardingError(f"model {model['name']!r} pricing must be a non-empty object")
            unknown_pricing = set(pricing) - {"input", "output", "cache_read", "cache_write", "reasoning"}
            if unknown_pricing:
                raise OnboardingError(
                    f"model {model['name']!r} has unknown pricing field(s): "
                    + ", ".join(sorted(unknown_pricing))
                )
            missing_pricing = {"input", "output"} - set(pricing)
            if missing_pricing:
                raise OnboardingError(
                    f"model {model['name']!r} pricing requires: "
                    + ", ".join(sorted(missing_pricing))
                )
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                for value in pricing.values()
            ):
                raise OnboardingError(f"model {model['name']!r} pricing values must be finite non-negative numbers")
        try:
            validate_pricing_policy(model)
        except ValueError as exc:
            raise OnboardingError(str(exc)) from exc
    provenance = data.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise OnboardingError("profile provenance must be an object")
    retire = data.get("retire") or {}
    retired = retire.get("models") or []
    if not isinstance(retired, list) or any(not isinstance(name, str) or not name for name in retired):
        raise OnboardingError("retire.models must be a list of non-empty model names")
    if len(set(retired)) != len(retired):
        raise OnboardingError("retire.models contains duplicates")
    if set(retired) & names:
        raise OnboardingError("a model cannot be both added and retired")
    if provenance is not None:
        from src.onboarding_generation import generated_safety

        generated_safety(data)
    return data


def _real_target(path: Path) -> Path:
    return Path(os.path.realpath(path.expanduser()))


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {"llm": []}
    return json.loads(path.read_text())


def secret_path_for_provider(config_path: Path, provider: dict) -> Path:
    del config_path  # secrets intentionally live outside checkout/config trees
    name = str(provider.get("secret_name") or f"{provider['id']}.api-key")
    if Path(name).name != name or not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
        raise OnboardingError("provider secret_name must be a safe filename")
    root = Path(
        os.environ.get("MODEL_GATEWAY_SECRET_DIR")
        or Path.home() / ".config" / "model-gateway" / "secrets"
    ).expanduser()
    intended = root / name
    if intended.is_symlink():
        raise OnboardingError(f"refusing symlink API key target: {intended}")
    return _real_target(intended)


def _load_existing_secret(
    config: dict,
    provider_id: str,
    config_path: Path,
    expected_base_url: str,
) -> str:
    block = (config.get("providers") or {}).get(provider_id) or {}
    if str(block.get("base_url") or "").rstrip("/") != expected_base_url.rstrip("/"):
        return ""
    if block.get("api_key"):
        return str(block["api_key"]).strip()
    raw = block.get("api_key_file")
    if not raw:
        return ""
    try:
        return read_api_key_file(raw, config_path)
    except OSError:
        return ""


def validate_upstream_models(profile: dict, api_key: str) -> list[str]:
    """Require every profile model to appear in the provider's /models list."""
    provider = profile["provider"]
    url = f"{str(provider['base_url']).rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "model-gateway-onboard/1",
    }
    try:
        response = httpx.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise OnboardingError(f"provider model discovery failed: {exc}") from exc
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise OnboardingError("provider /models response did not contain a model list")
    ids = {
        str(row.get("id"))
        for row in rows
        if isinstance(row, dict) and row.get("id")
    }
    missing = [m["provider_model_id"] for m in profile["models"] if m["provider_model_id"] not in ids]
    if missing:
        raise OnboardingError(f"provider does not advertise model(s): {', '.join(missing)}")
    return sorted(ids)


def _build_candidates(profile: dict, config: dict, model_doc: dict, *, secret_file: Path) -> tuple[dict, dict, dict]:
    provider = profile["provider"]
    provider_id = str(provider["id"]).strip().lower()
    models = [dict(model) for model in profile["models"]]
    for model in models:
        try:
            validate_pricing_policy(model)
        except ValueError as exc:
            raise OnboardingError(str(exc)) from exc
    retired = set((profile.get("retire") or {}).get("models") or [])
    new_names = {str(model["name"]) for model in models}

    current_llm = model_doc.get("llm") or []
    if not isinstance(current_llm, list):
        raise OnboardingError("model-info.json llm must be a list")
    current_by_name = {
        str(row["name"]): row
        for row in current_llm
        if isinstance(row, dict) and row.get("name")
    }
    current_names = set(current_by_name)
    missing_retired = retired - current_names
    # A previously applied profile may legitimately add reviewed metadata
    # (for example pricing) after its retired predecessor is already gone.
    # Permit that update only when every replacement still has the same
    # structural owner/upstream identity; a coincidental name from another
    # provider must not bypass first-run retirement checks.
    replacement_already_present = all(
        current_by_name.get(str(model["name"])) is not None
        and current_by_name[str(model["name"])].get("provider") == model.get("provider")
        and current_by_name[str(model["name"])].get("provider_model_id") == model.get("provider_model_id")
        for model in models
    )
    if missing_retired and not replacement_already_present:
        raise OnboardingError(
            "expected retired model(s) not found: " + ", ".join(sorted(missing_retired))
        )

    candidate_config = dict(config)
    providers = dict(candidate_config.get("providers") or {})
    existing_provider = dict(providers.get(provider_id) or {})
    existing_provider.update(
        {
            "base_url": str(provider["base_url"]).rstrip("/"),
            "protocol": provider.get("protocol", "openai"),
            "api_key_file": str(secret_file),
            "enabled": True,
        }
    )
    existing_provider.pop("api_key", None)
    providers[provider_id] = existing_provider
    candidate_config["providers"] = providers

    overlays = candidate_config.get("models") or []
    if isinstance(overlays, list):
        candidate_config["models"] = [
            row for row in overlays
            if not isinstance(row, dict) or row.get("name") not in retired | new_names
        ]
    overrides = candidate_config.get("model_overrides") or {}
    if isinstance(overrides, dict):
        candidate_config["model_overrides"] = {
            name: value for name, value in overrides.items() if name not in retired
        }

    candidate_doc = dict(model_doc)
    llm = [
        dict(row) for row in current_llm
        if isinstance(row, dict) and row.get("name") not in retired | new_names
    ]
    llm.extend(models)
    candidate_doc["llm"] = llm
    candidate_doc["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Fail closed on duplicate gateway-facing identifiers after replacement.
    seen: dict[str, str] = {}
    for row in llm + [r for r in candidate_config.get("models", []) if isinstance(r, dict)]:
        name = str(row.get("name") or "")
        identifiers = [row.get("name"), row.get("alias"), row.get("provider_model_id"), row.get("omlx_id")]
        extra = row.get("alternate_ids") or []
        identifiers.extend([extra] if isinstance(extra, str) else extra)
        for value in identifiers:
            if not value:
                continue
            model_id = str(value)
            owner = seen.get(model_id)
            if owner and owner != name:
                raise OnboardingError(f"routable id {model_id!r} collides between {owner!r} and {name!r}")
            seen[model_id] = name

    summary = {
        "profile": profile.get("id", ""),
        "provider": provider_id,
        "added_models": sorted(new_names),
        "retired_models": sorted(retired & current_names),
        "already_retired": sorted(retired - current_names),
        "secret_file": str(secret_file),
    }
    return candidate_config, candidate_doc, summary


def _snapshot(path: Path) -> tuple[bool, bytes, int]:
    if not path.exists():
        return False, b"", 0o600
    stat = path.stat()
    return True, path.read_bytes(), stat.st_mode & 0o777


def _atomic_write(path: Path, content: bytes, mode: int) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}")
    fd = None
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        tmp.unlink(missing_ok=True)


def _restore(path: Path, snapshot: tuple[bool, bytes, int]) -> None:
    existed, content, mode = snapshot
    if existed:
        _atomic_write(path, content, mode)
    else:
        path.unlink(missing_ok=True)


def _apply_profile_unlocked(
    profile: dict,
    *,
    config_path: Path,
    model_info_path: Path,
    model_info_source_path: Path | None = None,
    api_key: str | None = None,
    dry_run: bool = False,
    check_upstream: bool = True,
    upstream_validator: Callable[[dict, str], list[str]] = validate_upstream_models,
    allow_metadata_removal: bool = False,
    confirmed_replacements: set[str] | None = None,
    confirmed_retirements: set[str] | None = None,
    post_apply: Callable[[], None] | None = None,
    post_rollback: Callable[[], None] | None = None,
) -> dict:
    """Apply an onboarding profile and rollback every touched file on failure."""
    profile = validate_profile(profile)
    config_path = _real_target(config_path)
    model_info_path = _real_target(model_info_path)
    source_path = _real_target(model_info_source_path) if model_info_source_path else None
    config = _read_yaml(config_path)
    model_doc = _read_json(model_info_path)
    provider_id = str(profile["provider"]["id"]).strip().lower()
    secret_file = secret_path_for_provider(config_path, profile["provider"])
    configured_providers = config.get("providers") or {}
    if not isinstance(configured_providers, dict):
        raise OnboardingError("config providers must be an object")
    for owner, block in configured_providers.items():
        if owner == provider_id or not isinstance(block, dict) or not block.get("api_key_file"):
            continue
        owned_secret = resolve_api_key_file(block["api_key_file"], config_path)
        if owned_secret == secret_file:
            raise OnboardingError(
                f"provider secret target is already owned by provider {owner!r}: {secret_file}"
            )
    canonical_targets = [config_path, model_info_path, secret_file]
    if source_path and source_path != model_info_path:
        canonical_targets.append(source_path)
    if len(set(canonical_targets)) != len(canonical_targets):
        raise OnboardingError("config, model catalog, source mirror, and secret targets must be distinct")
    key = (
        api_key
        or _load_existing_secret(
            config,
            provider_id,
            config_path,
            str(profile["provider"]["base_url"]),
        )
    ).strip()

    candidate_config, candidate_doc, summary = _build_candidates(
        profile, config, model_doc, secret_file=secret_file
    )
    from src.onboarding_generation import (
        current_metadata_removals,
        generated_safety,
        validate_generated_approvals,
        validate_generated_catalog_state,
    )

    metadata_removals = current_metadata_removals(profile, model_doc)
    generated = bool(generated_safety(profile))
    current_rows = {
        str(row["name"]): row
        for row in model_doc.get("llm") or []
        if isinstance(row, dict) and row.get("name")
    }
    replacements = {
        str(model["name"])
        for model in profile["models"]
        if str(model["name"]) in current_rows
        and current_rows[str(model["name"])] != model
    }
    retirements = set((profile.get("retire") or {}).get("models") or [])
    summary["dry_run"] = bool(dry_run)
    summary["metadata_removals"] = metadata_removals
    summary["replaced_models"] = sorted(replacements)
    summary["requires_metadata_removal_approval"] = bool(metadata_removals)
    summary["requires_replacement_confirmation"] = bool(replacements)
    summary["requires_retirement_confirmation"] = bool(retirements)
    if dry_run:
        if generated:
            validate_generated_approvals(
                profile,
                current_model_doc=model_doc,
                allow_metadata_removal=True,
                confirmed_retirements=set((profile.get("retire") or {}).get("models") or []),
            )
        return summary
    if generated:
        validate_generated_catalog_state(profile, model_doc)
    if metadata_removals and not allow_metadata_removal:
        details = "; ".join(
            f"{name}: {', '.join(fields)}"
            for name, fields in sorted(metadata_removals.items())
        )
        raise OnboardingError(
            "profile would remove existing metadata; rerun with explicit approval: " + details
        )
    confirmed_replacement_set = set(confirmed_replacements or set())
    identical_models = {
        str(model["name"])
        for model in profile["models"]
        if current_rows.get(str(model["name"])) == model
    }
    if (
        not replacements.issubset(confirmed_replacement_set)
        or not confirmed_replacement_set.issubset(replacements | identical_models)
    ):
        raise OnboardingError(
            "model replacements require exact confirmation: "
            + ", ".join(sorted(replacements))
        )
    if retirements != set(confirmed_retirements or set()):
        raise OnboardingError(
            "model retirements require exact confirmation: "
            + ", ".join(sorted(retirements))
        )
    if generated:
        validate_generated_approvals(
            profile,
            current_model_doc=model_doc,
            allow_metadata_removal=allow_metadata_removal,
            confirmed_retirements=confirmed_retirements or set(),
        )
    if not key:
        raise OnboardingError("API key is required (use a hidden prompt, environment variable, or Keychain)")
    if check_upstream:
        upstream_validator(profile, key)

    targets: dict[Path, tuple[bytes, int]] = {
        secret_file: (key.encode() + b"\n", 0o600),
        config_path: (
            yaml.safe_dump(candidate_config, sort_keys=False, default_flow_style=False).encode(),
            0o600,
        ),
        model_info_path: ((json.dumps(candidate_doc, indent=2) + "\n").encode(), 0o644),
    }
    if source_path and source_path != model_info_path:
        targets[source_path] = targets[model_info_path]
    snapshots = {path: _snapshot(path) for path in targets}
    written: list[Path] = []
    try:
        for path, (content, mode) in targets.items():
            # Track before writing: os.replace may succeed even if a later
            # durability fsync raises, and that target must still be restored.
            written.append(path)
            _atomic_write(path, content, mode)
        if post_apply:
            post_apply()
    except Exception as exc:
        rollback_errors: list[str] = []
        for path in reversed(written):
            try:
                _restore(path, snapshots[path])
            except Exception as restore_exc:
                rollback_errors.append(f"restore {path}: {restore_exc}")
        if post_rollback:
            try:
                post_rollback()
            except Exception as reload_exc:
                rollback_errors.append(f"reload previous configuration: {reload_exc}")
        if rollback_errors:
            raise OnboardingError(
                f"onboarding failed ({exc}); rollback incomplete: {'; '.join(rollback_errors)}"
            ) from exc
        raise

    summary["written_to"] = [str(path) for path in targets]
    return summary


def apply_profile(
    profile: dict,
    *,
    config_path: Path,
    model_info_path: Path,
    model_info_source_path: Path | None = None,
    api_key: str | None = None,
    dry_run: bool = False,
    check_upstream: bool = True,
    upstream_validator: Callable[[dict, str], list[str]] = validate_upstream_models,
    allow_metadata_removal: bool = False,
    confirmed_replacements: set[str] | None = None,
    confirmed_retirements: set[str] | None = None,
    post_apply: Callable[[], None] | None = None,
    post_rollback: Callable[[], None] | None = None,
) -> dict:
    """Serialize onboarding operations for a config and apply one profile."""
    kwargs = dict(
        config_path=config_path,
        model_info_path=model_info_path,
        model_info_source_path=model_info_source_path,
        api_key=api_key,
        dry_run=dry_run,
        check_upstream=check_upstream,
        upstream_validator=upstream_validator,
        allow_metadata_removal=allow_metadata_removal,
        confirmed_replacements=confirmed_replacements,
        confirmed_retirements=confirmed_retirements,
        post_apply=post_apply,
        post_rollback=post_rollback,
    )
    if dry_run:
        return _apply_profile_unlocked(profile, **kwargs)
    try:
        with config_write_lock(config_path, blocking=False):
            return _apply_profile_unlocked(profile, **kwargs)
    except RuntimeError as exc:
        raise OnboardingError(str(exc)) from exc
