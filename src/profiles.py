"""Consumer-owned profile validation, persistence, and execution binding.

The public snapshot is deliberately separate from the internal route binding:
clients see canonical gateway model names only, while invocation revalidates the
stored closure digest against the live provider/catalog configuration.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import ipaddress
import json
import math
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src import catalog, providers

_ID_RE = re.compile(r"^[a-z0-9-]{1,32}$")
_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
_PROTOCOLS = {"openai_chat", "openai_responses", "anthropic_messages"}
_LOCALITIES = {"local_only", "cloud_explicit"}
_CREDENTIAL_POLICIES = {"gateway_local", "gateway_managed", "consumer_byok"}
_ROOT_FIELDS = {"schema_version", "namespace", "source_revision", "default_profile", "profiles"}
_PROFILE_FIELDS = {"id", "description", "locality", "credential_policy", "protocols", "routes", "defaults"}
_ROUTE_FIELDS = {"text", "vision"}
_DEFAULT_FIELDS = {"temperature", "max_output_tokens", "reasoning_effort"}
_CACHE_LATEST = "private, max-age=60, stale-if-error=86400"
_CACHE_IMMUTABLE = "private, max-age=31536000, immutable"
_MAX_REGISTRY_BYTES = 16 * 1024 * 1024
_MAX_NAMESPACE_BYTES = 4 * 1024 * 1024

_lock = threading.RLock()


@dataclass
class ProfileError(Exception):
    status: int
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class ExecutionProfile:
    selector: str
    profile_id: str
    namespace: str
    gateway_version: int
    model: str
    defaults: dict


def is_selector(value: object) -> bool:
    return isinstance(value, str) and value.startswith("profile:")


def parse_selector(value: object) -> tuple[str, str]:
    if not is_selector(value):
        raise ProfileError(404, "profile_not_found", "Profile selector not found")
    resource = value[len("profile:"):]
    if resource.count("/") != 1:
        raise ProfileError(404, "profile_not_found", "Invalid profile selector")
    namespace, name = resource.split("/", 1)
    if not _ID_RE.fullmatch(namespace) or not _NAME_RE.fullmatch(name):
        raise ProfileError(404, "profile_not_found", "Invalid profile selector")
    return namespace, resource


def registry_path() -> Path:
    override = os.environ.get("MODEL_GATEWAY_PROFILE_REGISTRY", "").strip()
    if override:
        return Path(override).expanduser()
    config = providers._load_config()
    raw = (config.get("profiles") or {}).get("registry_path", "consumer-profiles-registry.json")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        config_target = Path(os.path.realpath(providers.CONFIG_PATH.expanduser()))
        path = config_target.parent / path
    return path


@contextmanager
def _file_lock(*, exclusive: bool):
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = Path(f"{path}.lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield path
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _empty_registry() -> dict:
    return {"format": 1, "namespaces": {}}


def _load(path: Path) -> dict:
    if not path.exists():
        return _empty_registry()
    try:
        raw = path.read_bytes()
        if len(raw) > _MAX_REGISTRY_BYTES:
            raise ValueError("registry exceeds size limit")
        data = json.loads(raw)
        if (
            not isinstance(data, dict)
            or data.get("format") != 1
            or not isinstance(data.get("namespaces"), dict)
        ):
            raise ValueError("invalid registry format")
        for namespace, versions in data["namespaces"].items():
            if not isinstance(namespace, str) or not isinstance(versions, list):
                raise ValueError("invalid registry namespace")
            for expected_version, record in enumerate(versions, start=1):
                if (
                    not isinstance(record, dict)
                    or record.get("gateway_version") != expected_version
                    or not isinstance(record.get("registered_at"), str)
                    or not isinstance(record.get("etag"), str)
                    or not isinstance(record.get("manifest_digest"), str)
                    or not isinstance(record.get("manifest"), dict)
                    or record["manifest"].get("namespace") != namespace
                    or not isinstance(record.get("bindings"), dict)
                ):
                    raise ValueError("invalid registry snapshot")
        return data
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProfileError(503, "profile_registry_unavailable", "Profile registry is unavailable") from exc


def _write_atomic(path: Path, data: dict) -> None:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    temp_name = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o600)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError as exc:
        raise ProfileError(503, "profile_registry_unavailable", "Profile registry is unavailable") from exc
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _etag(value: object) -> str:
    return f'"{_digest(value)}"'


def _reject_unknown(value: dict, allowed: set[str], where: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ProfileError(422, "invalid_profile_manifest", f"{where} has unknown field(s): {', '.join(sorted(unknown))}")


def _bounded_text(value: object, where: str, *, maximum: int, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ProfileError(422, "invalid_profile_manifest", f"{where} must be a string")
    text = value.strip()
    if (required and not text) or len(text) > maximum or any(ord(ch) < 32 for ch in text):
        raise ProfileError(422, "invalid_profile_manifest", f"{where} is invalid")
    return text


def _inventory_by_name() -> dict[str, dict]:
    return {
        str(row.get("name")): row
        for row in providers.effective_model_inventory()
        if isinstance(row.get("name"), str) and row.get("name")
    }


def _effective_endpoint_binding(
    model: str,
    *,
    locality: str,
    provider_override: str | None = None,
) -> dict:
    info = providers.resolve(model, provider_override=provider_override)
    if info is None:
        raise ProfileError(422, "invalid_profile_route", f"Route target '{model}' is unavailable")
    parsed = urlsplit(info.base_url)
    hostname = (parsed.hostname or "").lower()
    is_loopback = hostname == "localhost"
    if hostname and not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if locality == "local_only" and (
        catalog.canonical_provider(info.provider) != "omlx"
        or parsed.scheme not in {"http", "https"}
        or not is_loopback
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise ProfileError(
            422,
            "invalid_profile_route",
            f"Local-only route '{model}' does not use a trusted loopback oMLX endpoint",
        )
    return {
        "provider": catalog.canonical_provider(info.provider),
        "scheme": parsed.scheme.lower(),
        "host": hostname,
        "port": parsed.port,
        "path": parsed.path.rstrip("/"),
        "protocol": info.protocol,
        "endpoint_suffix": info.endpoint_suffix,
    }


def _fallback_target(entry: dict, inventory: dict[str, dict]) -> str | None:
    upstream_id = entry.get("provider_model_id") or entry.get("omlx_id") or entry.get("name")
    mapping = providers._load_config().get("model_fallbacks") or {}
    if not isinstance(mapping, dict) or upstream_id not in mapping:
        return None
    target = str(mapping[upstream_id])
    matches = [
        name for name, row in inventory.items()
        if target in (row.get("routable_ids") or [])
    ]
    if len(matches) != 1:
        raise ProfileError(422, "invalid_profile_route", "A configured model fallback does not resolve canonically")
    return matches[0]


def _route_closure(model: str, *, locality: str, inventory: dict[str, dict], stack: tuple[str, ...] = ()) -> dict:
    entry = inventory.get(model)
    if entry is None:
        raise ProfileError(422, "invalid_profile_route", f"Route target '{model}' is not a canonical gateway model")
    if model in stack:
        raise ProfileError(422, "invalid_profile_route", "Profile route closure contains a cycle")
    providers_declared = sorted({catalog.canonical_provider(p) for p in entry.get("declared_providers") or []})
    if not providers_declared:
        providers_declared = [catalog.canonical_provider(entry.get("provider"))]
    if locality == "local_only" and set(providers_declared) != {"omlx"}:
        raise ProfileError(422, "invalid_profile_route", f"Local-only route '{model}' can reach a non-local provider")
    if locality == "cloud_explicit" and "omlx" in providers_declared:
        raise ProfileError(422, "invalid_profile_route", f"Cloud-explicit route '{model}' can reach a local provider")

    configured_candidates = providers.pool_candidates(model)
    endpoint_bindings = [
        _effective_endpoint_binding(
            model,
            locality=locality,
            provider_override=provider,
        )
        for provider in configured_candidates
    ]
    if not endpoint_bindings:
        endpoint_bindings = [_effective_endpoint_binding(model, locality=locality)]

    result: dict[str, Any] = {
        "model": model,
        "providers": providers_declared,
        "endpoints": endpoint_bindings,
        "provider_model_id": entry.get("provider_model_id") or entry.get("omlx_id") or entry.get("name"),
        "vision": bool(entry.get("vision")),
    }
    composite = entry.get("composite")
    if composite is not None:
        if not isinstance(composite, dict):
            raise ProfileError(422, "invalid_profile_route", f"Composite route '{model}' is invalid")
        text_model = composite.get("text_model")
        vision_model = composite.get("vision_model")
        if not isinstance(text_model, str) or not isinstance(vision_model, str):
            raise ProfileError(422, "invalid_profile_route", f"Composite route '{model}' is invalid")
        result["composite"] = {
            "image_handling": composite.get("image_handling", "extract_then_answer"),
            "max_images": composite.get("max_images", 4),
            "text": _route_closure(text_model, locality=locality, inventory=inventory, stack=(*stack, model)),
            "vision": _route_closure(vision_model, locality=locality, inventory=inventory, stack=(*stack, model)),
        }
    fallback = _fallback_target(entry, inventory)
    if fallback:
        result["fallback"] = _route_closure(fallback, locality=locality, inventory=inventory, stack=(*stack, model))
    return result


def _profile_is_executable(profile: dict) -> bool:
    return (
        profile["locality"] == "local_only"
        and profile["credential_policy"] == "gateway_local"
    ) or (
        profile["locality"] == "cloud_explicit"
        and profile["credential_policy"] == "gateway_managed"
    )


def _validate_defaults_for_routes(profile: dict, inventory: dict[str, dict]) -> None:
    defaults = profile.get("defaults") or {}
    reasoning_effort = defaults.get("reasoning_effort")
    max_output_tokens = defaults.get("max_output_tokens")
    for model in set(profile["routes"].values()):
        entry = inventory[model]
        if reasoning_effort is not None:
            try:
                levels = catalog.normalized_thinking_levels(entry)
            except ValueError as exc:
                raise ProfileError(422, "invalid_profile_route", f"Route target '{model}' has invalid thinking capabilities") from exc
            if reasoning_effort == "off":
                supported = not levels or "off" in levels
            else:
                supported = reasoning_effort in levels
            if not supported:
                raise ProfileError(
                    422,
                    "invalid_profile_manifest",
                    f"reasoning_effort default is not supported by route '{model}'",
                )
        route_limit = entry.get("max_output_tokens")
        if (
            max_output_tokens is not None
            and isinstance(route_limit, int)
            and not isinstance(route_limit, bool)
            and route_limit > 0
            and max_output_tokens > route_limit
        ):
            raise ProfileError(
                422,
                "invalid_profile_manifest",
                f"max_output_tokens default exceeds route '{model}' limit",
            )


def _binding_for(profile: dict, inventory: dict[str, dict]) -> str:
    closure = {
        role: _route_closure(model, locality=profile["locality"], inventory=inventory)
        for role, model in sorted(profile["routes"].items())
    }
    vision_model = profile["routes"].get("vision")
    if vision_model:
        vision_entry = inventory[vision_model]
        if not vision_entry.get("vision") and not isinstance(vision_entry.get("composite"), dict):
            raise ProfileError(422, "invalid_profile_route", f"Vision route '{vision_model}' is not vision-capable")
    _validate_defaults_for_routes(profile, inventory)
    return hashlib.sha256(_canonical_bytes(closure)).hexdigest()


def validate_manifest(raw: object) -> tuple[dict, dict[str, str]]:
    if not isinstance(raw, dict):
        raise ProfileError(422, "invalid_profile_manifest", "Manifest must be an object")
    _reject_unknown(raw, _ROOT_FIELDS, "manifest")
    required = {"schema_version", "namespace", "source_revision", "profiles"}
    if required - set(raw):
        raise ProfileError(422, "invalid_profile_manifest", f"Manifest is missing: {', '.join(sorted(required - set(raw)))}")
    if raw.get("schema_version") != 1 or isinstance(raw.get("schema_version"), bool):
        raise ProfileError(422, "invalid_profile_manifest", "schema_version must be 1")
    namespace = raw.get("namespace")
    if not isinstance(namespace, str) or not _ID_RE.fullmatch(namespace):
        raise ProfileError(422, "invalid_profile_manifest", "namespace is invalid")
    source_revision = _bounded_text(raw.get("source_revision"), "source_revision", maximum=512)
    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles or len(raw_profiles) > 256:
        raise ProfileError(422, "invalid_profile_manifest", "profiles must be a non-empty list")

    normalized_profiles = []
    seen: set[str] = set()
    for index, value in enumerate(raw_profiles):
        if not isinstance(value, dict):
            raise ProfileError(422, "invalid_profile_manifest", f"profiles[{index}] must be an object")
        _reject_unknown(value, _PROFILE_FIELDS, f"profiles[{index}]")
        missing = {"id", "locality", "credential_policy", "protocols", "routes"} - set(value)
        if missing:
            raise ProfileError(422, "invalid_profile_manifest", f"profiles[{index}] is missing: {', '.join(sorted(missing))}")
        profile_id = value.get("id")
        if not isinstance(profile_id, str) or profile_id.count("/") != 1:
            raise ProfileError(422, "invalid_profile_manifest", f"profiles[{index}].id is invalid")
        owner, name = profile_id.split("/", 1)
        if owner != namespace or not _NAME_RE.fullmatch(name) or profile_id in seen:
            raise ProfileError(422, "invalid_profile_manifest", f"profiles[{index}].id is invalid")
        seen.add(profile_id)
        locality = value.get("locality")
        credential_policy = value.get("credential_policy")
        if locality not in _LOCALITIES or credential_policy not in _CREDENTIAL_POLICIES:
            raise ProfileError(422, "invalid_profile_manifest", f"profiles[{index}] has an unsupported policy value")
        valid_policy = (
            locality == "local_only" and credential_policy == "gateway_local"
        ) or (
            locality == "cloud_explicit"
            and credential_policy in {"gateway_managed", "consumer_byok"}
        )
        if not valid_policy:
            raise ProfileError(
                422,
                "invalid_profile_manifest",
                f"profiles[{index}] has an invalid locality/credential policy combination",
            )
        protocols = value.get("protocols")
        if (
            not isinstance(protocols, list) or not protocols
            or any(not isinstance(p, str) or p not in _PROTOCOLS for p in protocols)
            or len(protocols) != len(set(protocols))
        ):
            raise ProfileError(422, "invalid_profile_manifest", f"profiles[{index}].protocols is invalid")
        routes = value.get("routes")
        if not isinstance(routes, dict):
            raise ProfileError(422, "invalid_profile_manifest", f"profiles[{index}].routes must be an object")
        _reject_unknown(routes, _ROUTE_FIELDS, f"profiles[{index}].routes")
        if "text" not in routes or any(not isinstance(v, str) or not v for v in routes.values()):
            raise ProfileError(422, "invalid_profile_manifest", f"profiles[{index}].routes is invalid")
        defaults = value.get("defaults", {})
        if not isinstance(defaults, dict):
            raise ProfileError(422, "invalid_profile_manifest", f"profiles[{index}].defaults must be an object")
        _reject_unknown(defaults, _DEFAULT_FIELDS, f"profiles[{index}].defaults")
        if "temperature" in defaults and (
            isinstance(defaults["temperature"], bool) or not isinstance(defaults["temperature"], (int, float))
            or not math.isfinite(defaults["temperature"])
        ):
            raise ProfileError(422, "invalid_profile_manifest", "temperature default must be finite")
        if "max_output_tokens" in defaults and (
            isinstance(defaults["max_output_tokens"], bool) or not isinstance(defaults["max_output_tokens"], int)
            or defaults["max_output_tokens"] <= 0
        ):
            raise ProfileError(422, "invalid_profile_manifest", "max_output_tokens default must be a positive integer")
        if "reasoning_effort" in defaults and defaults["reasoning_effort"] not in catalog.THINKING_LEVELS:
            raise ProfileError(422, "invalid_profile_manifest", "reasoning_effort default is invalid")
        profile = {
            "id": profile_id,
            "locality": locality,
            "credential_policy": credential_policy,
            "protocols": list(protocols),
            "routes": dict(routes),
            "defaults": copy.deepcopy(defaults),
        }
        if "description" in value:
            profile["description"] = _bounded_text(value["description"], f"profiles[{index}].description", maximum=1024, required=False)
        normalized_profiles.append(profile)

    default_profile = raw.get("default_profile")
    if default_profile is not None and default_profile not in seen:
        raise ProfileError(422, "invalid_profile_manifest", "default_profile must name a profile in this manifest")
    manifest = {
        "schema_version": 1,
        "namespace": namespace,
        "source_revision": source_revision,
        "profiles": normalized_profiles,
    }
    if default_profile is not None:
        manifest["default_profile"] = default_profile

    inventory = _inventory_by_name()
    bindings = {profile["id"]: _binding_for(profile, inventory) for profile in normalized_profiles}
    return manifest, bindings


def _public_snapshot(record: dict) -> dict:
    manifest = record["manifest"]
    public_profiles = []
    for stored in manifest["profiles"]:
        profile = {
            "id": stored["id"],
            "locality": stored["locality"],
            "credential_policy": stored["credential_policy"],
            "executable": _profile_is_executable(stored),
            "protocols": copy.deepcopy(stored["protocols"]),
            "routes": copy.deepcopy(stored["routes"]),
            "defaults": copy.deepcopy(stored.get("defaults") or {}),
        }
        if "description" in stored:
            profile["description"] = stored["description"]
        public_profiles.append(profile)
    # Explicit projection prevents internal bindings or future storage fields
    # from accidentally becoming part of the consumer-facing contract.
    return {
        "schema_version": manifest["schema_version"],
        "namespace": manifest["namespace"],
        "source_revision": manifest["source_revision"],
        "gateway_version": record["gateway_version"],
        "registered_at": record["registered_at"],
        **({"default_profile": manifest["default_profile"]} if "default_profile" in manifest else {}),
        "profiles": public_profiles,
    }


def register(namespace: str, raw: object, *, if_match: str | None, if_none_match: str | None) -> tuple[dict, str, bool]:
    manifest, bindings = validate_manifest(raw)
    if manifest["namespace"] != namespace:
        raise ProfileError(422, "invalid_profile_manifest", "Path namespace must match manifest namespace")
    with _lock, _file_lock(exclusive=True) as path:
        data = _load(path)
        versions = data["namespaces"].get(namespace, [])
        current = versions[-1] if versions else None
        if current is None:
            if if_none_match is None and if_match is None:
                raise ProfileError(428, "precondition_required", "First registration requires If-None-Match: *")
            if "*" not in _etag_values(if_none_match) or if_match is not None:
                raise ProfileError(412, "precondition_failed", "Profile snapshot precondition failed")
        else:
            if if_match is None:
                if if_none_match is None:
                    raise ProfileError(428, "precondition_required", "Update requires If-Match")
                raise ProfileError(412, "precondition_failed", "Profile snapshot precondition failed")
            if not if_match_succeeds(if_match, current["etag"]):
                raise ProfileError(412, "precondition_failed", "Profile snapshot precondition failed")
            if if_none_match_matches(if_none_match, current["etag"]):
                raise ProfileError(412, "precondition_failed", "Profile snapshot precondition failed")
        manifest_digest = _digest(manifest)
        if (
            current is not None
            and current["manifest_digest"] == manifest_digest
            and current["bindings"] == bindings
        ):
            return _public_snapshot(current), current["etag"], False
        record = {
            "gateway_version": (current["gateway_version"] + 1) if current else 1,
            "registered_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "manifest_digest": manifest_digest,
            "manifest": manifest,
            "bindings": bindings,
        }
        record["etag"] = _etag(_public_snapshot(record))
        versions.append(record)
        data["namespaces"][namespace] = versions
        if len(_canonical_bytes(versions)) > _MAX_NAMESPACE_BYTES:
            raise ProfileError(413, "profile_history_too_large", "Profile namespace history exceeds its storage quota")
        if len(_canonical_bytes(data)) > _MAX_REGISTRY_BYTES:
            raise ProfileError(413, "profile_registry_too_large", "Profile registry exceeds its storage quota")
        _write_atomic(path, data)
        return _public_snapshot(record), record["etag"], True


def get_snapshot(namespace: str, version: int | None = None) -> tuple[dict, str]:
    with _lock, _file_lock(exclusive=False) as path:
        data = _load(path)
        versions = data["namespaces"].get(namespace, [])
        if not versions:
            raise ProfileError(404, "profile_not_found", "Profile namespace not found")
        if version is None:
            record = versions[-1]
        else:
            record = next((item for item in versions if item.get("gateway_version") == version), None)
            if record is None:
                raise ProfileError(404, "profile_not_found", "Profile snapshot version not found")
        return _public_snapshot(record), record["etag"]


def cache_control(*, immutable: bool) -> str:
    return _CACHE_IMMUTABLE if immutable else _CACHE_LATEST


def _etag_values(header: str | None) -> list[str]:
    if not header:
        return []
    return [value.strip() for value in header.split(",") if value.strip()]


def if_match_succeeds(header: str | None, current: str) -> bool:
    values = _etag_values(header)
    return "*" in values or current in values


def if_none_match_matches(header: str | None, current: str) -> bool:
    current_weak = current.removeprefix("W/")
    for value in _etag_values(header):
        if value == "*" or value.removeprefix("W/") == current_weak:
            return True
    return False


def resolve_execution(selector: object, *, protocol: str, has_image: bool) -> ExecutionProfile:
    namespace, profile_id = parse_selector(selector)
    with _lock, _file_lock(exclusive=False) as path:
        data = _load(path)
        versions = data["namespaces"].get(namespace, [])
        if not versions:
            raise ProfileError(404, "profile_not_found", "Profile namespace not found")
        record = versions[-1]
        profile = next((item for item in record["manifest"]["profiles"] if item["id"] == profile_id), None)
        if profile is None:
            raise ProfileError(404, "profile_not_found", "Profile not found")
        if protocol not in profile["protocols"]:
            raise ProfileError(403, "profile_protocol_denied", "Profile is not enabled for this protocol")
        if not _profile_is_executable(profile):
            raise ProfileError(403, "profile_execution_unavailable", "Profile execution is unavailable for this policy")
        role = "vision" if has_image else "text"
        model = profile["routes"].get(role)
        if not model:
            raise ProfileError(403, "profile_vision_unavailable", "Profile does not define a vision route")
        try:
            live_binding = _binding_for(profile, _inventory_by_name())
        except ProfileError as exc:
            raise ProfileError(409, "profile_binding_changed", "Profile route binding changed; re-register the profile") from exc
        if live_binding != record["bindings"].get(profile_id):
            raise ProfileError(409, "profile_binding_changed", "Profile route binding changed; re-register the profile")
        if not providers.resolve(model):
            raise ProfileError(503, "profile_route_unavailable", "Profile route is unavailable")
        return ExecutionProfile(
            selector=str(selector),
            profile_id=profile_id,
            namespace=namespace,
            gateway_version=record["gateway_version"],
            model=model,
            defaults=copy.deepcopy(profile.get("defaults") or {}),
        )
