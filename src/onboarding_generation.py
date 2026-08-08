"""Safe draft generation for OpenAI-compatible provider onboarding.

Generation is deliberately separate from transactional application. It may
observe a provider's model list and run explicitly requested probes, but it
only writes a secret-free schema-v1 profile for review.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

import httpx
import yaml

from src.catalog import normalize_thinking_capabilities
from src.onboarding import OnboardingError

STRUCTURAL_MODEL_FIELDS = {"name", "provider", "provider_model_id"}
OPTIONAL_MODEL_FIELDS = (
    "alias",
    "context",
    "max_output_tokens",
    "thinking",
    "thinking_levels",
    "thinking_format",
    "vision",
    "quirks",
    "system_instruction",
    "pricing",
    "desc",
)
ALLOWED_QUIRKS = {
    "no_stream_options",
    "no_reasoning_params",
    "reasoning_none_with_tools",
    "force_reasoning_effort_max",
    "native_minimal_reasoning",
    "use_max_completion_tokens",
    "drop_fixed_sampling_fields",
    "named_tool_choice_as_required",
    "inline_image_urls_only",
    "anthropic_tool_result_blocks",
}
ALLOWED_PRICING_FIELDS = {
    "input", "output", "cache_read", "cache_write", "cache_write_1h", "reasoning",
}
DISCOVERY_OVERRIDE_STATES = {
    "unavailable",
    "rate_limited",
    "provider_error",
    "malformed",
    "network_error",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_generation_inputs(provider_id: str, base_url: str, model_ids: Iterable[str]) -> tuple[str, str, list[str]]:
    provider_id = provider_id.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", provider_id):
        raise OnboardingError(
            "provider id must contain only lowercase letters, digits, underscores, and hyphens"
        )
    base_url = base_url.strip().rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise OnboardingError(
            "provider base_url must be an HTTPS URL without credentials, query, or fragment"
        )
    models = [str(value).strip() for value in model_ids]
    if not models or any(not value for value in models):
        raise OnboardingError("at least one non-empty provider model id is required")
    if len(set(models)) != len(models):
        raise OnboardingError("provider model ids contain duplicates")
    return provider_id, base_url, models


def discover_models(base_url: str, api_key: str | None = None) -> dict:
    """Observe an OpenAI-compatible /models endpoint without guessing fields."""
    url = f"{base_url.rstrip('/')}/models"
    headers = {
        "Accept": "application/json",
        "User-Agent": "model-gateway-onboard-generate/1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = httpx.get(url, headers=headers, timeout=30)
    except httpx.HTTPError as exc:
        return {
            "status": "network_error",
            "source": "provider_models",
            "url": url,
            "observed_at": _utc_now(),
            "detail": type(exc).__name__,
            "model_ids": [],
        }

    base = {
        "source": "provider_models",
        "url": url,
        "observed_at": _utc_now(),
        "http_status": response.status_code,
        "model_ids": [],
    }
    if response.status_code in {401, 403}:
        return {**base, "status": "authentication_failed"}
    if response.status_code in {404, 405}:
        return {**base, "status": "unavailable"}
    if response.status_code == 429:
        return {**base, "status": "rate_limited"}
    if response.status_code >= 500:
        return {**base, "status": "provider_error"}
    if response.status_code >= 400:
        return {**base, "status": "http_error"}
    try:
        payload = response.json()
    except ValueError:
        return {**base, "status": "malformed"}
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {**base, "status": "malformed"}
    ids = sorted({
        str(row["id"])
        for row in rows
        if isinstance(row, dict) and row.get("id")
    })
    return {**base, "status": "verified", "model_ids": ids}


def _probe_payload(model_id: str, kind: str) -> dict:
    payload: dict = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 16,
        "stream": False,
    }
    if kind == "tools":
        payload.update({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "probe_ok",
                    "description": "Return a successful capability probe.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            }],
            "tool_choice": "required",
        })
    elif kind == "vision":
        payload["messages"] = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Reply with OK."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZrWQAAAAASUVORK5CYII="
                    },
                },
            ],
        }]
    elif kind == "reasoning":
        payload["reasoning_effort"] = "low"
    return payload


def _valid_probe_response(kind: str, payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return False
    if kind == "tools":
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            return False
        for call in tool_calls:
            if (
                not isinstance(call, dict)
                or call.get("type") != "function"
                or not isinstance(call.get("function"), dict)
                or call["function"].get("name") != "probe_ok"
                or not isinstance(call["function"].get("arguments"), str)
            ):
                return False
            try:
                arguments = json.loads(call["function"]["arguments"])
            except ValueError:
                return False
            if arguments != {}:
                return False
        return True
    return any(
        isinstance(message.get(field), str) and bool(message[field].strip())
        for field in ("content", "reasoning_content")
    )


def run_probe(base_url: str, model_id: str, kind: str, api_key: str) -> dict:
    """Run one bounded, explicitly authorized probe and record only its outcome."""
    if kind not in {"text", "tools", "vision", "reasoning"}:
        raise OnboardingError(f"unsupported probe: {kind}")
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "model-gateway-onboard-probe/1",
    }
    try:
        response = httpx.post(url, headers=headers, json=_probe_payload(model_id, kind), timeout=60)
    except httpx.HTTPError as exc:
        return {
            "kind": kind,
            "model_id": model_id,
            "status": "network_error",
            "observed_at": _utc_now(),
            "detail": type(exc).__name__,
        }
    if response.status_code in {401, 403}:
        status = "authentication_failed"
    elif response.status_code == 429:
        status = "rate_limited"
    elif response.status_code >= 500:
        status = "provider_error"
    elif not 200 <= response.status_code < 300:
        status = "rejected"
    else:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        status = "observed_success" if _valid_probe_response(kind, payload) else "malformed"
    return {
        "kind": kind,
        "model_id": model_id,
        "status": status,
        "http_status": response.status_code,
        "observed_at": _utc_now(),
    }


def _profile_id(provider_id: str, model_ids: list[str]) -> str:
    raw = f"{provider_id}-{'-'.join(model_ids)}"
    slug = re.sub(r"[^a-z0-9_-]+", "-", raw.lower()).strip("-_")
    if len(slug) <= 120:
        return slug
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{slug[:107].rstrip('-_')}-{digest}"


def _catalog_fingerprint(row: dict | None) -> str | None:
    if row is None:
        return None
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _field_evidence(source: str, confidence: str, **extra) -> dict:
    return {"source": source, "confidence": confidence, **extra}


def _existing_rows(model_info_path: Path) -> dict[str, dict]:
    if not model_info_path.exists():
        return {}
    try:
        payload = json.loads(model_info_path.read_text())
    except (OSError, ValueError) as exc:
        raise OnboardingError(f"could not read model catalog for generation: {exc}") from exc
    rows = payload.get("llm") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise OnboardingError("model-info.json llm must be a list")
    return {
        str(row["name"]): dict(row)
        for row in rows
        if isinstance(row, dict) and row.get("name")
    }


def build_draft(
    *,
    provider_id: str,
    base_url: str,
    model_ids: Iterable[str],
    model_info_path: Path,
    discovery: dict | None = None,
    probes: list[dict] | None = None,
    docs: Iterable[str] = (),
    documented_fields: dict[str, str] | None = None,
    aliases: dict[str, str] | None = None,
    context: int | None = None,
    max_output_tokens: int | None = None,
    thinking: str | None = None,
    thinking_levels: Iterable[str] | None = None,
    thinking_format: str | None = None,
    vision: bool | None = None,
    quirks: Iterable[str] = (),
    pricing: dict | None = None,
    description: str | None = None,
    retirements: Iterable[str] = (),
    preserve_existing_metadata: bool = False,
    drop_existing_metadata: Iterable[str] = (),
) -> dict:
    """Build a secret-free schema-v1 draft with top-level provenance."""
    provider_id, base_url, models = validate_generation_inputs(provider_id, base_url, model_ids)
    if context is not None and context <= 0:
        raise OnboardingError("context must be a positive integer")
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise OnboardingError("max_output_tokens must be a positive integer")
    dropped = set(drop_existing_metadata)
    retired = [str(value).strip() for value in retirements]
    if any(not value for value in retired) or len(set(retired)) != len(retired):
        raise OnboardingError("retired model names must be non-empty and unique")

    discovery = discovery or {
        "status": "not_attempted",
        "source": "provider_models",
        "model_ids": [],
    }
    advertised = set(discovery.get("model_ids") or [])
    successful_text_probes = {
        str(row.get("model_id"))
        for row in (probes or [])
        if row.get("kind") == "text" and row.get("status") == "observed_success"
    }
    existing = _existing_rows(model_info_path)
    existing_metadata_fields = {
        key
        for model_id in models
        for key in (existing.get(model_id) or {})
        if key not in STRUCTURAL_MODEL_FIELDS
    }
    unknown_drops = dropped - set(OPTIONAL_MODEL_FIELDS) - existing_metadata_fields
    if unknown_drops:
        raise OnboardingError("unknown metadata field(s): " + ", ".join(sorted(unknown_drops)))
    aliases = aliases or {}
    documented_fields = documented_fields or {}
    allowed_documented = set(OPTIONAL_MODEL_FIELDS)
    unknown_documented = set(documented_fields) - allowed_documented
    if unknown_documented:
        raise OnboardingError("unknown documented field(s): " + ", ".join(sorted(unknown_documented)))
    fields: dict[str, dict] = {
        "/provider/id": _field_evidence("operator", "confirmed"),
        "/provider/base_url": _field_evidence("operator", "confirmed"),
        "/provider/protocol": _field_evidence("deterministic_default", "default"),
        "/provider/secret_name": _field_evidence("deterministic_default", "default"),
    }
    removals: dict[str, list[str]] = {}
    profile_models: list[dict] = []

    explicit_common = {
        "context": context,
        "max_output_tokens": max_output_tokens,
        "thinking": thinking,
        "thinking_levels": list(thinking_levels) if thinking_levels is not None else None,
        "thinking_format": thinking_format,
        "vision": vision,
        "pricing": pricing,
        "desc": description,
    }
    explicit_quirks = list(quirks)
    unknown_quirks = set(explicit_quirks) - ALLOWED_QUIRKS
    if unknown_quirks:
        raise OnboardingError("unsupported request quirk(s): " + ", ".join(sorted(unknown_quirks)))
    if pricing is not None:
        unknown_pricing = set(pricing) - ALLOWED_PRICING_FIELDS
        if unknown_pricing:
            raise OnboardingError("unknown pricing field(s): " + ", ".join(sorted(unknown_pricing)))
        missing_pricing = {"input", "output"} - set(pricing)
        if missing_pricing:
            raise OnboardingError("pricing requires: " + ", ".join(sorted(missing_pricing)))
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            for value in pricing.values()
        ):
            raise OnboardingError("pricing values must be finite non-negative numbers")
    for index, model_id in enumerate(models):
        row: dict = {
            "name": model_id,
            "provider": provider_id,
            "provider_model_id": model_id,
        }
        pointer = f"/models/{index}"
        identity_confidence = "verified" if model_id in advertised else "observed" if model_id in successful_text_probes else "confirmed"
        identity_source = "provider_models" if model_id in advertised else "probe" if model_id in successful_text_probes else "operator"
        fields[f"{pointer}/name"] = _field_evidence("deterministic_default", "default")
        fields[f"{pointer}/provider"] = _field_evidence("operator", "confirmed")
        fields[f"{pointer}/provider_model_id"] = _field_evidence(identity_source, identity_confidence)

        old = existing.get(model_id) or {}
        if preserve_existing_metadata:
            for key, value in old.items():
                if key not in STRUCTURAL_MODEL_FIELDS and key not in dropped:
                    row[key] = value
                    fields[f"{pointer}/{key}"] = _field_evidence("existing_catalog", "confirmed")

        if model_id in aliases:
            row["alias"] = aliases[model_id]
            source = "documentation" if "alias" in documented_fields else "operator"
            evidence = {"url": documented_fields["alias"]} if source == "documentation" else {}
            fields[f"{pointer}/alias"] = _field_evidence(source, "confirmed", **evidence)
        for key, value in explicit_common.items():
            if value is not None:
                row[key] = value
                source = "documentation" if key in documented_fields else "operator"
                evidence = {"url": documented_fields[key]} if source == "documentation" else {}
                fields[f"{pointer}/{key}"] = _field_evidence(source, "confirmed", **evidence)
        if explicit_quirks:
            row["quirks"] = explicit_quirks
            source = "documentation" if "quirks" in documented_fields else "operator"
            evidence = {"url": documented_fields["quirks"]} if source == "documentation" else {}
            fields[f"{pointer}/quirks"] = _field_evidence(source, "confirmed", **evidence)
        if thinking in {"", "never"} and thinking_levels is None:
            # An explicit disabled mode must not inherit stale enabled levels
            # through --preserve-existing-metadata.
            row.pop("thinking_levels", None)
            fields.pop(f"{pointer}/thinking_levels", None)
        try:
            row.update(normalize_thinking_capabilities(row))
        except ValueError as exc:
            raise OnboardingError(str(exc)) from exc
        if f"{pointer}/thinking_levels" not in fields:
            fields[f"{pointer}/thinking_levels"] = _field_evidence(
                "deterministic_default", "default"
            )
        removed = sorted(
            key for key in old
            if key not in STRUCTURAL_MODEL_FIELDS and key not in row
        )
        if removed:
            removals[model_id] = removed
        profile_models.append(row)

    unresolved = [
        field
        for field in OPTIONAL_MODEL_FIELDS
        if not all(field in row for row in profile_models)
    ]
    missing_retirements = sorted(set(retired) - set(existing))
    catalog_fingerprints = {
        name: _catalog_fingerprint(existing.get(name))
        for name in dict.fromkeys([*models, *retired])
    }
    status = "needs_review" if unresolved or removals or retired or missing_retirements else "ready"
    profile: dict = {
        "schema_version": 1,
        "id": _profile_id(provider_id, models),
        "provider": {
            "id": provider_id,
            "base_url": base_url,
            "protocol": "openai",
            "secret_name": f"{provider_id}.api-key",
        },
        "models": profile_models,
        "provenance": {
            "generator": "model-gateway onboard generate",
            "generated_at": _utc_now(),
            "status": status,
            "fields": fields,
            "discovery": {key: value for key, value in discovery.items() if key != "model_ids"},
            "probes": list(probes or []),
            "documentation": list(dict.fromkeys([*docs, *documented_fields.values()])),
            "unresolved": unresolved,
            "safety": {
                "metadata_removals": removals,
                "retirements": retired,
                "missing_retirements": missing_retirements,
                "catalog_fingerprints": catalog_fingerprints,
            },
        },
    }
    if retired:
        profile["retire"] = {"models": retired}
    return profile


def write_draft(
    profile: dict,
    path: Path,
    *,
    force: bool = False,
    protected_paths: Iterable[Path] = (),
) -> Path:
    """Write a draft atomically after canonical collision and symlink checks.

    Existing parent-directory symlinks are resolved deliberately; the leaf may
    not itself be a symlink. Runtime config/catalog targets are protected even
    when either spelling traverses a symlink.
    """
    path = path.expanduser()
    if path.is_symlink():
        raise OnboardingError(f"refusing symlink draft target: {path}")
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    path = path.parent.resolve() / path.name
    protected = {
        Path(os.path.realpath(value.expanduser()))
        for value in protected_paths
        if value is not None
    }
    if path in protected:
        raise OnboardingError(f"draft target collides with protected runtime file: {path}")
    if path.exists() and not force:
        raise OnboardingError(f"draft already exists: {path} (use --force to replace it)")
    text = yaml.safe_dump(profile, sort_keys=False, default_flow_style=False)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path.resolve()


def generated_safety(profile: dict) -> dict:
    provenance = profile.get("provenance")
    if provenance is None:
        return {}
    if not isinstance(provenance, dict):
        raise OnboardingError("profile provenance must be an object")
    if provenance.get("generator") != "model-gateway onboard generate":
        return {}
    safety = provenance.get("safety")
    if not isinstance(safety, dict):
        raise OnboardingError("generated profile requires a safety object")
    removals = safety.get("metadata_removals")
    retirements = safety.get("retirements")
    missing_retirements = safety.get("missing_retirements", [])
    catalog_fingerprints = safety.get("catalog_fingerprints")
    if not isinstance(removals, dict) or any(
        not isinstance(name, str)
        or not isinstance(fields, list)
        or any(not isinstance(field, str) for field in fields)
        for name, fields in removals.items()
    ):
        raise OnboardingError("generated profile metadata_removals are malformed")
    if not isinstance(retirements, list) or any(not isinstance(name, str) for name in retirements):
        raise OnboardingError("generated profile retirements are malformed")
    if not isinstance(missing_retirements, list) or any(
        not isinstance(name, str) for name in missing_retirements
    ):
        raise OnboardingError("generated profile missing_retirements are malformed")
    target_names = {
        str(model.get("name")) for model in profile.get("models") or []
    } | set(retirements)
    if (
        not isinstance(catalog_fingerprints, dict)
        or set(catalog_fingerprints) != target_names
        or any(
            not isinstance(name, str)
            or (fingerprint is not None and not re.fullmatch(r"[0-9a-f]{64}", str(fingerprint)))
            for name, fingerprint in catalog_fingerprints.items()
        )
    ):
        raise OnboardingError("generated profile catalog_fingerprints are malformed")
    actual_retirements = list((profile.get("retire") or {}).get("models") or [])
    if sorted(retirements) != sorted(actual_retirements):
        raise OnboardingError("generated profile safety retirements do not match retire.models")
    return safety


def _current_rows(model_doc: dict) -> dict[str, dict]:
    rows = model_doc.get("llm") if isinstance(model_doc, dict) else None
    if not isinstance(rows, list):
        raise OnboardingError("model-info.json llm must be a list")
    return {
        str(row["name"]): row
        for row in rows
        if isinstance(row, dict) and row.get("name")
    }


def current_metadata_removals(profile: dict, model_doc: dict) -> dict[str, list[str]]:
    current = _current_rows(model_doc)
    removals: dict[str, list[str]] = {}
    for model in profile.get("models") or []:
        name = str(model.get("name") or "")
        old = current.get(name) or {}
        missing = sorted(
            key for key in old
            if key not in STRUCTURAL_MODEL_FIELDS and key not in model
        )
        if missing:
            removals[name] = missing
    return removals


def validate_generated_catalog_state(profile: dict, model_doc: dict) -> bool:
    """Ensure catalog rows used to generate a draft have not changed.

    Return True when the complete generated state is already applied.
    """
    safety = generated_safety(profile)
    if not safety:
        return False
    current = _current_rows(model_doc)
    models = [dict(model) for model in profile.get("models") or []]
    retired = set((profile.get("retire") or {}).get("models") or [])
    already_applied = (
        all(current.get(str(model["name"])) == model for model in models)
        and all(name not in current for name in retired)
    )
    if already_applied:
        return True
    expected = safety["catalog_fingerprints"]
    for name, fingerprint in expected.items():
        if _catalog_fingerprint(current.get(name)) != fingerprint:
            raise OnboardingError(
                f"model catalog entry {name!r} changed after draft generation; regenerate the profile before applying"
            )
    return False


def validate_generated_approvals(
    profile: dict,
    *,
    current_model_doc: dict | None = None,
    allow_metadata_removal: bool = False,
    confirmed_retirements: Iterable[str] = (),
) -> None:
    """Fail closed on generated-profile policy decisions before mutation."""
    safety = generated_safety(profile)
    if not safety:
        return
    already_applied = False
    if current_model_doc is not None:
        already_applied = validate_generated_catalog_state(profile, current_model_doc)
    recorded_removals = {
        name: sorted(fields)
        for name, fields in (safety.get("metadata_removals") or {}).items()
    }
    if current_model_doc is not None and not already_applied:
        actual_removals = current_metadata_removals(profile, current_model_doc)
        if recorded_removals != actual_removals:
            raise OnboardingError(
                "model catalog changed after draft generation; regenerate the profile before applying"
            )
    if recorded_removals and not already_applied and not allow_metadata_removal:
        details = "; ".join(
            f"{name}: {', '.join(fields)}" for name, fields in sorted(recorded_removals.items())
        )
        raise OnboardingError(
            "generated profile would remove existing metadata; explicitly approve it: " + details
        )
    retired = set(safety.get("retirements") or [])
    confirmed = set(confirmed_retirements)
    if retired != confirmed:
        raise OnboardingError(
            "generated profile retirements require exact confirmation: " + ", ".join(sorted(retired))
        )


def discovery_allows_override(profile: dict) -> bool:
    provenance = profile.get("provenance") or {}
    discovery = provenance.get("discovery") or {}
    status = discovery.get("status")
    if status == "authentication_failed":
        return False
    if status in DISCOVERY_OVERRIDE_STATES:
        return True
    if status != "conflict":
        return False
    probes = provenance.get("probes") or []
    model_ids = {str(model.get("provider_model_id")) for model in profile.get("models") or []}
    successful = {
        str(row.get("model_id"))
        for row in probes
        if isinstance(row, dict) and row.get("kind") == "text" and row.get("status") == "observed_success"
    }
    return bool(model_ids) and model_ids.issubset(successful)
