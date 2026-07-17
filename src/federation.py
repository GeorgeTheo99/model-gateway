"""Explicit, one-hop federation between model-gateway nodes.

Federated catalogs and routes deliberately live outside the provider registry.
A node exports only its direct local routes; importers retain a validated
last-known-good catalog and address imported models as ``<owner>/<direct_id>``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.secret_files import read_api_key_file, resolve_api_key_file

log = logging.getLogger("model-gateway")

SCHEMA_VERSION = 1
SOURCE_HEADER = "X-Model-Gateway-Source"
OWNER_HEADER = "X-Model-Gateway-Owner"
_NODE_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_REFRESH_SECONDS = 30.0
_DEFAULT_STALE_SECONDS = 300.0
_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_STREAM_IDLE_TIMEOUT_SECONDS = 900.0
_DEFAULT_MAX_CATALOG_BYTES = 1_048_576
_DEFAULT_MAX_MODELS = 1_000
_MAX_DIRECT_ID_LENGTH = 512

# Metadata allowed to cross the trust boundary. URLs, arbitrary provider data,
# and status/error text are intentionally excluded.
_MODEL_METADATA_FIELDS = {
    "object",
    "created",
    "thinking",
    "thinking_format",
    "thinking_levels",
    "max_reachable",
    "forwarded_params",
    "vision",
}
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class FederationConfigError(ValueError):
    """Invalid operator federation configuration."""


class CatalogValidationError(ValueError):
    """A peer returned a catalog that is unsafe or violates the contract."""


@dataclass(frozen=True)
class PeerConfig:
    node_id: str
    base_url: str
    api_key: str = field(repr=False)


@dataclass(frozen=True)
class FederationConfig:
    node_id: str
    peers: dict[str, PeerConfig]
    refresh_interval_seconds: float
    stale_after_seconds: float
    request_timeout_seconds: float
    cache_path: Path
    max_catalog_bytes: int
    max_models_per_peer: int
    stream_idle_timeout_seconds: float = _DEFAULT_STREAM_IDLE_TIMEOUT_SECONDS


@dataclass
class PeerState:
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    revision: int | None = None
    digest: str = ""
    generated_at: str = ""
    last_success_at: float | None = None
    last_attempt_at: float | None = None
    healthy: bool = False
    last_error: str = "not_refreshed"


@dataclass(frozen=True)
class ImportedRoute:
    owner_node: str
    direct_model_id: str
    peer: PeerConfig

    @property
    def route_id(self) -> str:
        return f"{self.owner_node}/{self.direct_model_id}"


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _number(block: dict, name: str, default: float, *, integer: bool = False) -> float | int:
    value = block.get(name, default)
    if not _is_finite_number(value) or value <= 0:
        raise FederationConfigError(f"federation.{name} must be a positive finite number")
    if integer:
        if not isinstance(value, int):
            raise FederationConfigError(f"federation.{name} must be a positive integer")
        return value
    return float(value)


def _validate_node_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _NODE_RE.fullmatch(value):
        raise FederationConfigError(
            f"{label} must be a lowercase DNS label (letters, digits, and interior hyphens; max 63 chars)"
        )
    return value


def _validate_base_url(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FederationConfigError(f"{label}.base_url is required")
    try:
        parsed = urlsplit(value.strip())
    except ValueError as exc:
        raise FederationConfigError(f"{label}.base_url is invalid") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise FederationConfigError(f"{label}.base_url must be an HTTP(S) URL with a host")
    try:
        parsed.port
    except ValueError as exc:
        raise FederationConfigError(f"{label}.base_url has an invalid port") from exc
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise FederationConfigError(f"{label}.base_url must not contain credentials, query, or fragment")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def load_config(config_path: Path | str) -> FederationConfig | None:
    """Load and strictly validate the optional ``federation:`` block."""
    config_path = Path(config_path)
    config_target = Path(os.path.realpath(config_path.expanduser()))
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            document = yaml.safe_load(f) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise FederationConfigError(f"could not read federation configuration: {exc}") from exc
    if not isinstance(document, dict):
        raise FederationConfigError("configuration root must be a mapping")
    block = document.get("federation")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise FederationConfigError("federation must be a mapping")

    allowed = {
        "node_id",
        "refresh_interval_seconds",
        "stale_after_seconds",
        "request_timeout_seconds",
        "stream_idle_timeout_seconds",
        "cache_path",
        "max_catalog_bytes",
        "max_models_per_peer",
        "peers",
    }
    unknown = set(block) - allowed
    if unknown:
        raise FederationConfigError(f"unknown federation setting(s): {', '.join(sorted(unknown))}")

    node_id = _validate_node_id(block.get("node_id"), "federation.node_id")
    raw_peers = block.get("peers")
    if not isinstance(raw_peers, dict):
        raise FederationConfigError("federation.peers must be an explicit mapping")

    peers: dict[str, PeerConfig] = {}
    peer_key_paths: dict[str, Path] = {}
    for raw_id, raw_peer in raw_peers.items():
        peer_id = _validate_node_id(raw_id, "federation peer ID")
        if peer_id == node_id:
            raise FederationConfigError(f"federation peer {peer_id!r} cannot be this node")
        if not isinstance(raw_peer, dict):
            raise FederationConfigError(f"federation.peers.{peer_id} must be a mapping")
        peer_unknown = set(raw_peer) - {"base_url", "api_key", "api_key_file"}
        if peer_unknown:
            raise FederationConfigError(
                f"unknown setting(s) for federation peer {peer_id!r}: {', '.join(sorted(peer_unknown))}"
            )
        has_key = "api_key" in raw_peer
        has_file = "api_key_file" in raw_peer
        if has_key == has_file:
            raise FederationConfigError(
                f"federation peer {peer_id!r} must configure exactly one of api_key or api_key_file"
            )
        if has_file:
            raw_key_file = raw_peer["api_key_file"]
            if not isinstance(raw_key_file, str) or not raw_key_file.strip():
                raise FederationConfigError(f"federation peer {peer_id!r} api_key_file must be a non-empty path")
            key_path = resolve_api_key_file(raw_key_file, config_path)
            try:
                api_key = read_api_key_file(raw_key_file, config_path)
            except OSError as exc:
                raise FederationConfigError(f"federation peer {peer_id!r} api_key_file is unusable: {exc}") from exc
            peer_key_paths[peer_id] = key_path
        else:
            raw_key = raw_peer["api_key"]
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise FederationConfigError(f"federation peer {peer_id!r} api_key must be a non-empty string")
            api_key = raw_key.strip()
        if not api_key:
            raise FederationConfigError(f"federation peer {peer_id!r} credential is empty")
        peers[peer_id] = PeerConfig(
            node_id=peer_id,
            base_url=_validate_base_url(raw_peer.get("base_url"), f"federation.peers.{peer_id}"),
            api_key=api_key,
        )

    raw_cache_path = block.get("cache_path", "federation-cache.json")
    if not isinstance(raw_cache_path, str) or not raw_cache_path.strip():
        raise FederationConfigError("federation.cache_path must be a non-empty path")
    # Use the credential resolver so relative paths and symlinks have exactly
    # the same canonicalization as peer key-file reads.
    cache_path = resolve_api_key_file(raw_cache_path, config_path)
    if cache_path == config_target:
        raise FederationConfigError("federation.cache_path must not resolve to the configuration file")
    for peer_id, key_path in peer_key_paths.items():
        if cache_path == key_path:
            raise FederationConfigError(
                f"federation.cache_path must not resolve to peer {peer_id!r} api_key_file"
            )
    # Provider credentials are loaded independently and may intentionally be
    # absent until deployment. Resolve declared paths without reading them so
    # the federation cache can never create or replace a provider secret.
    for section_name in ("providers", "workspaces"):
        entries = document.get(section_name)
        if not isinstance(entries, dict):
            continue
        for entry_id, entry in entries.items():
            if not isinstance(entry, dict) or not entry.get("api_key_file"):
                continue
            key_path = resolve_api_key_file(entry["api_key_file"], config_path)
            if cache_path == key_path:
                singular = section_name.removesuffix("s")
                raise FederationConfigError(
                    f"federation.cache_path must not resolve to {singular} {entry_id!r} api_key_file"
                )

    return FederationConfig(
        node_id=node_id,
        peers=peers,
        refresh_interval_seconds=float(_number(block, "refresh_interval_seconds", _DEFAULT_REFRESH_SECONDS)),
        stale_after_seconds=float(_number(block, "stale_after_seconds", _DEFAULT_STALE_SECONDS)),
        request_timeout_seconds=float(_number(block, "request_timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)),
        stream_idle_timeout_seconds=float(
            _number(block, "stream_idle_timeout_seconds", _DEFAULT_STREAM_IDLE_TIMEOUT_SECONDS)
        ),
        cache_path=cache_path,
        max_catalog_bytes=int(_number(block, "max_catalog_bytes", _DEFAULT_MAX_CATALOG_BYTES, integer=True)),
        max_models_per_peer=int(_number(block, "max_models_per_peer", _DEFAULT_MAX_MODELS, integer=True)),
    )


def catalog_digest(models: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        models, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _valid_direct_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_DIRECT_ID_LENGTH
        or value != value.strip()
        or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)
    ):
        raise CatalogValidationError("catalog contains an invalid direct_model_id")
    return value


def _sanitize_model(raw: Any, owner: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CatalogValidationError("catalog models must be objects")
    direct_id = _valid_direct_id(raw.get("direct_model_id"))
    if raw.get("owner_node") != owner:
        raise CatalogValidationError(f"catalog model {direct_id!r} has the wrong owner_node")

    model: dict[str, Any] = {"direct_model_id": direct_id, "owner_node": owner}
    for key in _MODEL_METADATA_FIELDS:
        if key not in raw:
            continue
        value = raw[key]
        # Payload size is bounded separately; this type gate prevents arbitrary
        # nested status/config objects from entering discovery responses/cache.
        if key in {"thinking_levels", "forwarded_params"}:
            if (
                not isinstance(value, list)
                or not all(
                    isinstance(item, str)
                    and len(item) <= 128
                    and not any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in item)
                    for item in value
                )
            ):
                raise CatalogValidationError(f"catalog model {direct_id!r} has invalid {key}")
        elif key == "created":
            if not _is_finite_number(value) or value < 0:
                raise CatalogValidationError(f"catalog model {direct_id!r} has invalid {key}")
        elif key in {"max_reachable", "vision"}:
            if not isinstance(value, bool):
                raise CatalogValidationError(f"catalog model {direct_id!r} has invalid {key}")
        elif (
            not isinstance(value, str)
            or len(value) > 128
            or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)
            or (key == "object" and value != "model")
        ):
            raise CatalogValidationError(f"catalog model {direct_id!r} has invalid {key}")
        model[key] = value
    model.setdefault("object", "model")
    model.setdefault("created", 0)
    return model


def validate_catalog(
    payload: Any,
    expected_node: str,
    *,
    max_models: int,
    previous_revision: int | None = None,
    previous_digest: str = "",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate a peer catalog and return its normalized payload/model index."""
    if not isinstance(payload, dict):
        raise CatalogValidationError("catalog must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CatalogValidationError("unsupported federation catalog schema_version")
    if payload.get("node_id") != expected_node:
        raise CatalogValidationError("federation catalog node_id does not match the configured peer")

    revision = payload.get("revision")
    digest = payload.get("digest")
    generated_at = payload.get("generated_at")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise CatalogValidationError("catalog revision must be a non-negative integer")
    if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
        raise CatalogValidationError("catalog digest must be a lowercase SHA-256 hex digest")
    if not isinstance(generated_at, str):
        raise CatalogValidationError("catalog generated_at must be an ISO-8601 timestamp")
    try:
        generated = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CatalogValidationError("catalog generated_at must be an ISO-8601 timestamp") from exc
    if generated.tzinfo is None:
        raise CatalogValidationError("catalog generated_at must include a timezone")

    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise CatalogValidationError("catalog models must be a list")
    if len(raw_models) > max_models:
        raise CatalogValidationError("catalog exceeds the configured model-count limit")

    models: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_models:
        model = _sanitize_model(raw, expected_node)
        direct_id = model["direct_model_id"]
        if direct_id in by_id:
            raise CatalogValidationError(f"catalog contains duplicate direct_model_id {direct_id!r}")
        models.append(model)
        by_id[direct_id] = model
    if not secrets.compare_digest(catalog_digest(models), digest):
        raise CatalogValidationError("catalog digest does not match its models")

    if previous_revision is not None:
        if revision < previous_revision:
            raise CatalogValidationError("catalog revision moved backwards")
        if revision == previous_revision and previous_digest and not secrets.compare_digest(digest, previous_digest):
            raise CatalogValidationError("catalog digest changed without a revision change")

    normalized = {
        "schema_version": SCHEMA_VERSION,
        "node_id": expected_node,
        "revision": revision,
        "digest": digest,
        "generated_at": generated_at,
        "models": models,
    }
    return normalized, by_id


def _iso_timestamp(epoch: float | None = None) -> str:
    value = time.time() if epoch is None else epoch
    if not _is_finite_number(value):
        raise ValueError("timestamp must be finite")
    return datetime.fromtimestamp(value, tz=UTC).isoformat().replace("+00:00", "Z")


def _reject_json_constant(value: str) -> None:
    raise CatalogValidationError(f"invalid JSON numeric constant {value!r}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CatalogValidationError("JSON number exceeds the finite numeric range")
    return parsed


def _json_loads_finite(value: bytes | str) -> Any:
    return json.loads(value, parse_constant=_reject_json_constant, parse_float=_finite_json_float)


def _cache_timestamp(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if not _is_finite_number(value) or value < 0:
        raise CatalogValidationError(f"cached {label} must be a non-negative finite timestamp or null")
    timestamp = float(value)
    try:
        datetime.fromtimestamp(timestamp, UTC)
    except (OSError, OverflowError, ValueError) as exc:
        raise CatalogValidationError(f"cached {label} is outside the supported timestamp range") from exc
    return timestamp


def _extract_bearer(request: Request) -> str:
    values = request.headers.getlist("authorization")
    if len(values) != 1:
        return ""
    auth = values[0].strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _single_header(request: Request, name: str) -> str:
    values = request.headers.getlist(name)
    if len(values) != 1 or not values[0].strip() or "," in values[0]:
        raise HTTPException(status_code=400, detail=f"exactly one {name} header is required")
    return values[0].strip()


def _error_kind(exc: Exception) -> str:
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return "timeout"
    if isinstance(exc, CatalogValidationError):
        return "invalid_catalog"
    if isinstance(exc, httpx.HTTPStatusError):
        return "http_error"
    if isinstance(exc, (httpx.NetworkError, httpx.RemoteProtocolError)):
        return "unreachable"
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        return "invalid_json"
    return "refresh_failed"


class FederationManager:
    def __init__(
        self,
        config: FederationConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.states: dict[str, PeerState] = {
            node_id: PeerState() for node_id in (config.peers if config else {})
        }
        self._transport = transport
        self._refresh_task: asyncio.Task | None = None
        self._cache_write_lock = threading.Lock()
        self._owns_cache_writes = True
        self._catalog_revision = 0
        self._catalog_digest = ""
        self._catalog_generated_at = ""
        self._catalog_models: list[dict[str, Any]] = []
        if config:
            self._load_cache()

    @property
    def enabled(self) -> bool:
        return self.config is not None

    def status(self) -> dict[str, Any]:
        if not self.config:
            return {"enabled": False}
        return {
            "enabled": True,
            "node_id": self.config.node_id,
            "peers": sorted(self.config.peers),
            "cache_path": str(self.config.cache_path),
        }

    def _client(self, *, streaming: bool = False) -> httpx.AsyncClient:
        assert self.config is not None
        if streaming:
            timeout = httpx.Timeout(
                connect=self.config.request_timeout_seconds,
                read=self.config.stream_idle_timeout_seconds,
                write=self.config.request_timeout_seconds,
                pool=self.config.request_timeout_seconds,
            )
        else:
            timeout = httpx.Timeout(self.config.request_timeout_seconds)
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    def authenticate_catalog_request(self, request: Request) -> str:
        """Authenticate a catalog read against the source peer credential."""
        if not self.config:
            raise HTTPException(status_code=401, detail="federation is not configured")
        source = _single_header(request, SOURCE_HEADER)
        peer = self.config.peers.get(source)
        token = _extract_bearer(request)
        if peer is None or not token or not secrets.compare_digest(token, peer.api_key):
            raise HTTPException(status_code=401, detail="missing or invalid federation peer credential")
        return source

    def has_forwarding_headers(self, request: Request) -> bool:
        return bool(request.headers.get(SOURCE_HEADER) or request.headers.get(OWNER_HEADER))

    def authenticate_forwarded_request(self, request: Request) -> str:
        """Authenticate and structurally validate a one-hop peer forward."""
        if not self.config:
            raise HTTPException(status_code=401, detail="federation is not configured")
        source = _single_header(request, SOURCE_HEADER)
        owner = _single_header(request, OWNER_HEADER)
        via = _single_header(request, "Via")
        if owner != self.config.node_id:
            raise HTTPException(status_code=400, detail="federation forward has the wrong owner")
        if source == self.config.node_id or source == owner:
            raise HTTPException(status_code=400, detail="federation forwarding loop rejected")
        if via != f"1.1 {source}-model-gateway":
            raise HTTPException(status_code=400, detail="federation forwarding must contain exactly one direct Via hop")
        peer = self.config.peers.get(source)
        token = _extract_bearer(request)
        if peer is None or not token or not secrets.compare_digest(token, peer.api_key):
            raise HTTPException(status_code=401, detail="missing or invalid federation peer credential")
        request.state.federation_source = source
        return source

    def validate_inbound_direct_model(self, model: Any) -> None:
        if not self.config:
            raise HTTPException(status_code=400, detail="federation forward requires a direct local model")
        try:
            _valid_direct_id(model)
        except CatalogValidationError as exc:
            raise HTTPException(status_code=400, detail="federation forward requires a valid direct local model") from exc

    def build_catalog(self, local_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a stable revisioned catalog from direct local discovery rows."""
        if not self.config:
            raise HTTPException(status_code=404, detail="federation is not configured")
        models: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in local_rows:
            direct_id = _valid_direct_id(row.get("id"))
            if direct_id in seen:
                raise CatalogValidationError(f"local catalog contains duplicate direct model ID {direct_id!r}")
            seen.add(direct_id)
            raw = {
                "direct_model_id": direct_id,
                "owner_node": self.config.node_id,
                **{key: row[key] for key in _MODEL_METADATA_FIELDS if key in row},
            }
            models.append(_sanitize_model(raw, self.config.node_id))
        models.sort(key=lambda item: item["direct_model_id"])
        digest = catalog_digest(models)
        if not self._catalog_digest or not secrets.compare_digest(digest, self._catalog_digest):
            # Epoch milliseconds remain exactly comparable in JSON/JavaScript;
            # +1 handles multiple catalog changes within the same millisecond.
            # The value is cached below, preserving monotonicity across restarts.
            self._catalog_revision = max(time.time_ns() // 1_000_000, self._catalog_revision + 1)
            self._catalog_digest = digest
            self._catalog_generated_at = _iso_timestamp()
            self._catalog_models = models
            self._write_cache()
        return {
            "schema_version": SCHEMA_VERSION,
            "node_id": self.config.node_id,
            "revision": self._catalog_revision,
            "digest": self._catalog_digest,
            "generated_at": self._catalog_generated_at,
            "models": self._catalog_models,
        }

    async def refresh_peer(self, node_id: str) -> bool:
        if not self.config or node_id not in self.config.peers:
            return False
        peer = self.config.peers[node_id]
        state = self.states[node_id]
        state.last_attempt_at = time.time()
        endpoint = f"{peer.base_url}/v1/federation/catalog"
        try:
            # HTTPX read timeouts bound inactivity, not total wall time. This
            # outer deadline covers headers and the complete catalog body so a
            # peer cannot keep startup/reload alive with a slow drip feed.
            async with asyncio.timeout(self.config.request_timeout_seconds):
                async with self._client() as client:
                    async with client.stream(
                        "GET",
                        endpoint,
                        headers={
                            "Authorization": f"Bearer {peer.api_key}",
                            SOURCE_HEADER: self.config.node_id,
                            "Accept": "application/json",
                        },
                    ) as response:
                        response.raise_for_status()
                        if response.headers.get(SOURCE_HEADER) != node_id:
                            raise CatalogValidationError(
                                "catalog response source header does not match the configured peer"
                            )
                        declared = response.headers.get("content-length")
                        if declared:
                            try:
                                if int(declared) > self.config.max_catalog_bytes:
                                    raise CatalogValidationError("catalog exceeds the configured byte limit")
                            except ValueError as exc:
                                raise CatalogValidationError("catalog has an invalid Content-Length") from exc
                        chunks: list[bytes] = []
                        size = 0
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > self.config.max_catalog_bytes:
                                raise CatalogValidationError("catalog exceeds the configured byte limit")
                            chunks.append(chunk)
                payload = _json_loads_finite(b"".join(chunks))
                normalized, by_id = validate_catalog(
                    payload,
                    node_id,
                    max_models=self.config.max_models_per_peer,
                    previous_revision=state.revision,
                    previous_digest=state.digest,
                )
                state.models = by_id
                state.revision = normalized["revision"]
                state.digest = normalized["digest"]
                state.generated_at = normalized["generated_at"]
                state.last_success_at = time.time()
                state.healthy = True
                state.last_error = ""
                self._write_cache()
                return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - refresh failure preserves LKG
            state.healthy = False
            state.last_error = _error_kind(exc)
            log.warning("federation catalog refresh failed for %s (%s)", node_id, state.last_error)
            self._write_cache()
            return False

    async def refresh_all(self) -> None:
        if not self.config or not self.config.peers:
            return
        await asyncio.gather(*(self.refresh_peer(node_id) for node_id in self.config.peers))

    async def run_refresh_loop(self) -> None:
        assert self.config is not None
        try:
            while True:
                await asyncio.sleep(self.config.refresh_interval_seconds)
                await self.refresh_all()
        except asyncio.CancelledError:
            raise

    def start_background_refresh(self) -> None:
        if self.config and self.config.peers and self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self.run_refresh_loop(), name="federation-catalog-refresh")

    async def stop(self) -> None:
        task, self._refresh_task = self._refresh_task, None
        if task:
            current = asyncio.current_task()
            cancelling_before = current.cancelling() if current else 0
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Suppress the refresh task's expected cancellation, but do
                # not swallow a new cancellation of the lifecycle operation.
                if current and current.cancelling() > cancelling_before:
                    raise

    def resolve_imported(self, route_id: Any) -> ImportedRoute | None:
        if not self.config or not isinstance(route_id, str) or "/" not in route_id:
            return None
        owner, direct_id = route_id.split("/", 1)
        peer = self.config.peers.get(owner)
        state = self.states.get(owner)
        if not peer or not state or not direct_id or direct_id not in state.models:
            return None
        return ImportedRoute(owner_node=owner, direct_model_id=direct_id, peer=peer)

    def imported_rows(self, *, now: float | None = None) -> list[dict[str, Any]]:
        if not self.config:
            return []
        current = time.time() if now is None else now
        rows: list[dict[str, Any]] = []
        for owner, state in self.states.items():
            stale = state.last_success_at is None or current - state.last_success_at > self.config.stale_after_seconds
            available = bool(state.healthy and not stale)
            health = "ready" if available else "stale" if stale else "unreachable"
            status = {
                "state": health,
                "last_attempt_at": _iso_timestamp(state.last_attempt_at) if state.last_attempt_at else None,
                "last_success_at": _iso_timestamp(state.last_success_at) if state.last_success_at else None,
                "catalog_revision": state.revision,
                "catalog_digest": state.digest,
            }
            for direct_id, model in state.models.items():
                row = {
                    "id": f"{owner}/{direct_id}",
                    "object": model.get("object", "model"),
                    "created": model.get("created", 0),
                    "owned_by": owner,
                    **{key: model[key] for key in _MODEL_METADATA_FIELDS if key in model and key not in {"object", "created"}},
                    "federated": True,
                    "owner_node": owner,
                    "direct_model_id": direct_id,
                    "available": available,
                    "stale": stale,
                    "status": status,
                }
                rows.append(row)
        return rows

    async def forward(self, request: Request, path: str, body: dict[str, Any], route: ImportedRoute) -> Response:
        """Forward one request directly to the owning node without translation."""
        assert self.config is not None
        forwarded_body = dict(body)
        forwarded_body["model"] = route.direct_model_id
        is_stream = body.get("stream") is True
        try:
            # Serialize before allocating a client or creating a request. The
            # default JSON encoder accepts non-finite numbers, while ASCII
            # escaping would hide lone surrogates instead of rejecting them.
            serialized_body = json.dumps(
                forwarded_body,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            return _forward_invalid_request(path, "Federated request body is not valid finite UTF-8 JSON")

        headers = {
            "Authorization": f"Bearer {route.peer.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if is_stream else "application/json",
            "Accept-Encoding": "identity",
            SOURCE_HEADER: self.config.node_id,
            OWNER_HEADER: route.owner_node,
            "Via": f"1.1 {self.config.node_id}-model-gateway",
        }
        endpoint = f"{route.peer.base_url}{path}"
        client = self._client(streaming=is_stream)

        async def close(response: httpx.Response | None = None) -> None:
            try:
                if response is not None:
                    await response.aclose()
            finally:
                await client.aclose()

        response: httpx.Response | None = None
        try:
            upstream_request = client.build_request("POST", endpoint, content=serialized_body, headers=headers)
            # Streaming requests get a hard deadline through response headers.
            # Non-streaming requests retain the same deadline through the full
            # body read. Streaming body reads use the client's idle timeout.
            async with asyncio.timeout(self.config.request_timeout_seconds):
                response = await client.send(upstream_request, stream=True)
                if not is_stream:
                    if response.is_stream_consumed:
                        content = response.content
                    else:
                        content = b"".join([chunk async for chunk in response.aiter_raw()])
            response_headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower() not in _HOP_BY_HOP_HEADERS
                and (not is_stream or key.lower() != "content-length")
            }
        except asyncio.CancelledError:
            await close(response)
            raise
        except (TimeoutError, httpx.TimeoutException):
            await close(response)
            message = (
                "Federated owner response timed out"
                if response is not None and not is_stream
                else "Federated owner request timed out"
            )
            return _forward_error(path, 504, message)
        except httpx.HTTPError:
            await close(response)
            message = (
                "Federated owner response failed"
                if response is not None and not is_stream
                else "Cannot connect to federated model owner"
            )
            return _forward_error(path, 502, message)
        except Exception:  # noqa: BLE001 - never leak owner/client details
            await close(response)
            message = (
                "Federated owner response failed"
                if response is not None and not is_stream
                else "Cannot connect to federated model owner"
            )
            return _forward_error(path, 502, message)

        if not is_stream:
            await close(response)
            return Response(content=content, status_code=response.status_code, headers=response_headers)

        async def relay():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            except (TimeoutError, httpx.TimeoutException):
                # Response headers are already committed, so the only safe
                # behavior is to terminate the incomplete downstream stream.
                log.warning("federated owner stream timed out for %s", route.owner_node)
            finally:
                await close(response)

        return StreamingResponse(relay(), status_code=response.status_code, headers=response_headers)

    def _cache_document(self) -> dict[str, Any]:
        assert self.config is not None
        peers: dict[str, Any] = {}
        for node_id, state in self.states.items():
            if state.revision is None:
                continue
            peers[node_id] = {
                "revision": state.revision,
                "digest": state.digest,
                "generated_at": state.generated_at,
                "last_success_at": state.last_success_at,
                "last_attempt_at": state.last_attempt_at,
                "last_error": state.last_error,
                "models": list(state.models.values()),
            }
        local_catalog = None
        if self._catalog_digest:
            local_catalog = {
                "revision": self._catalog_revision,
                "digest": self._catalog_digest,
                "generated_at": self._catalog_generated_at,
                "models": self._catalog_models,
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "node_id": self.config.node_id,
            "local_catalog": local_catalog,
            "peers": peers,
        }

    def relinquish_cache_write_ownership(self) -> None:
        """Finish any active write and prevent this manager from starting another."""
        with self._cache_write_lock:
            self._owns_cache_writes = False

    def restore_cache_write_ownership(self) -> None:
        """Restore writes after an unsuccessful lifecycle handoff."""
        with self._cache_write_lock:
            self._owns_cache_writes = True

    def _write_cache(self) -> None:
        if not self.config:
            return
        # Ownership changes use the same lock and therefore cannot complete
        # while a write is active. Once relinquishment returns, an in-flight
        # request retaining this manager can update its private state but can
        # no longer replace the shared cache.
        with self._cache_write_lock:
            if not self._owns_cache_writes:
                return
            path = self.config.cache_path
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
                data = (
                    json.dumps(self._cache_document(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                ).encode()
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(tmp, path)
                finally:
                    if tmp.exists():
                        tmp.unlink()
            except (OSError, ValueError) as exc:
                log.warning("could not write federation cache: %s", type(exc).__name__)

    def _load_cache(self) -> None:
        assert self.config is not None
        path = self.config.cache_path
        if not path.exists():
            return
        try:
            raw = path.read_bytes()
            if len(raw) > self.config.max_catalog_bytes * max(1, len(self.config.peers) + 1):
                raise CatalogValidationError("cache exceeds its size limit")
            document = _json_loads_finite(raw)
            if (
                not isinstance(document, dict)
                or document.get("schema_version") != SCHEMA_VERSION
                or document.get("node_id") != self.config.node_id
                or not isinstance(document.get("peers"), dict)
            ):
                raise CatalogValidationError("cache identity/schema mismatch")

            restored_local: dict[str, Any] | None = None
            local_cache = document.get("local_catalog")
            if local_cache is not None:
                try:
                    if not isinstance(local_cache, dict):
                        raise CatalogValidationError("cached local catalog is invalid")
                    local_payload = {
                        "schema_version": SCHEMA_VERSION,
                        "node_id": self.config.node_id,
                        "revision": local_cache.get("revision"),
                        "digest": local_cache.get("digest"),
                        "generated_at": local_cache.get("generated_at"),
                        "models": local_cache.get("models"),
                    }
                    restored_local, _ = validate_catalog(
                        local_payload,
                        self.config.node_id,
                        max_models=len(local_cache.get("models", []))
                        if isinstance(local_cache.get("models"), list)
                        else 0,
                    )
                except (CatalogValidationError, TypeError, ValueError):
                    log.warning("ignoring invalid local section in federation cache")

            restored_peers: dict[
                str, tuple[dict[str, Any], dict[str, dict[str, Any]], float | None, float | None]
            ] = {}
            for node_id, peer_cache in document["peers"].items():
                if node_id not in self.config.peers:
                    continue
                try:
                    if not isinstance(peer_cache, dict):
                        raise CatalogValidationError("cached peer section must be an object")
                    payload = {
                        "schema_version": SCHEMA_VERSION,
                        "node_id": node_id,
                        "revision": peer_cache.get("revision"),
                        "digest": peer_cache.get("digest"),
                        "generated_at": peer_cache.get("generated_at"),
                        "models": peer_cache.get("models"),
                    }
                    normalized, by_id = validate_catalog(
                        payload,
                        node_id,
                        max_models=self.config.max_models_per_peer,
                    )
                    success = _cache_timestamp(peer_cache.get("last_success_at"), "last_success_at")
                    attempt = _cache_timestamp(peer_cache.get("last_attempt_at"), "last_attempt_at")
                    restored_peers[node_id] = (normalized, by_id, success, attempt)
                except (CatalogValidationError, TypeError, ValueError):
                    log.warning("ignoring invalid federation cache section for peer %s", node_id)

            # Apply only after every section has been validated into temporary
            # state, so no exception can leave a half-restored section.
            if restored_local is not None:
                self._catalog_revision = restored_local["revision"]
                self._catalog_digest = restored_local["digest"]
                self._catalog_generated_at = restored_local["generated_at"]
                self._catalog_models = restored_local["models"]
            for node_id, (normalized, by_id, success, attempt) in restored_peers.items():
                state = self.states[node_id]
                state.models = by_id
                state.revision = normalized["revision"]
                state.digest = normalized["digest"]
                state.generated_at = normalized["generated_at"]
                state.last_success_at = success
                state.last_attempt_at = attempt
                # Reachability is never trusted across process restarts.
                state.healthy = False
                state.last_error = "not_refreshed"
        except Exception as exc:  # noqa: BLE001 - corrupt cache is non-fatal
            log.warning("ignoring invalid federation cache (%s)", type(exc).__name__)


def _forward_error(path: str, status: int, message: str) -> JSONResponse:
    error = {"type": "federation_error", "message": message}
    if path == "/v1/messages":
        return JSONResponse(status_code=status, content={"type": "error", "error": error})
    return JSONResponse(status_code=status, content={"error": error})


def _forward_invalid_request(path: str, message: str) -> JSONResponse:
    error = {"type": "invalid_request_error", "message": message}
    if path == "/v1/messages":
        return JSONResponse(status_code=400, content={"type": "error", "error": error})
    return JSONResponse(status_code=400, content={"error": error})


_manager = FederationManager()
_lifecycle_lock: asyncio.Lock | None = None
_lifecycle_loop: asyncio.AbstractEventLoop | None = None
_lifecycle_generation = 0
_lifecycle_state_lock = threading.Lock()


def manager() -> FederationManager:
    return _manager


def _lifecycle_guard() -> tuple[asyncio.Lock, int]:
    """Return a lock bound to the running loop and its lifecycle generation."""
    global _lifecycle_generation, _lifecycle_lock, _lifecycle_loop
    loop = asyncio.get_running_loop()
    with _lifecycle_state_lock:
        if _lifecycle_lock is None or _lifecycle_loop is not loop:
            _lifecycle_lock = asyncio.Lock()
            _lifecycle_loop = loop
            _lifecycle_generation += 1
        return _lifecycle_lock, _lifecycle_generation


def _lifecycle_generation_is_current(generation: int) -> bool:
    with _lifecycle_state_lock:
        return generation == _lifecycle_generation


async def _activate_config(config: FederationConfig | None) -> None:
    """Quiesce cache writes, then refresh and atomically publish a replacement."""
    global _manager
    lifecycle_lock, generation = _lifecycle_guard()
    async with lifecycle_lock:
        previous = _manager
        replacement: FederationManager | None = None
        try:
            # Drain the owned background refresh first, then wait for any cache
            # write already in progress and reject later writes from requests
            # retaining the previous manager. Reads and routing remain live.
            await previous.stop()
            previous.relinquish_cache_write_ownership()
            if not _lifecycle_generation_is_current(generation):
                return

            # Construct only after exclusive cache-write ownership is available
            # so cache loading and the initial refresh share one ordered view.
            replacement = FederationManager(config)
            if replacement.enabled:
                await replacement.refresh_all()
            if not _lifecycle_generation_is_current(generation):
                replacement.relinquish_cache_write_ownership()
                await replacement.stop()
                return

            # Starting the task and publishing the manager are synchronous, so
            # readers can never observe a replacement before it is fully live.
            with _lifecycle_state_lock:
                if generation != _lifecycle_generation:
                    stale = True
                else:
                    stale = False
                    if replacement.enabled:
                        replacement.start_background_refresh()
                    _manager = replacement
            if stale:
                replacement.relinquish_cache_write_ownership()
                await replacement.stop()
                return
        except BaseException:
            # Revoke the unpublished replacement before restoring the previous
            # owner. Restoration is synchronous so cancellation cannot strand
            # the still-published manager without cache writes or refresh.
            if replacement is not None:
                replacement.relinquish_cache_write_ownership()
            if (
                _lifecycle_generation_is_current(generation)
                and _manager is previous
            ):
                previous.restore_cache_write_ownership()
                if previous.enabled:
                    previous.start_background_refresh()
            if replacement is not None:
                await replacement.stop()
            raise

        if replacement.enabled:
            log.info("federation ready: node=%s peers=%s", replacement.config.node_id, sorted(replacement.config.peers))


async def start(config_path: Path | str | None = None) -> None:
    """Configure federation, load LKG state, refresh once, and start refresh."""
    if config_path is None:
        from src import providers
        config_path = providers.CONFIG_PATH
    await _activate_config(load_config(config_path))


_PREVALIDATED_UNSET = object()


async def reconfigure(
    config_path: Path | str | None = None,
    *,
    config: FederationConfig | None | object = _PREVALIDATED_UNSET,
) -> dict[str, Any]:
    """Reconfigure from a path or an already validated config document."""
    if config is _PREVALIDATED_UNSET:
        if config_path is None:
            from src import providers
            config_path = providers.CONFIG_PATH
        validated = load_config(config_path)
    elif config is None or isinstance(config, FederationConfig):
        validated = config
    else:
        raise TypeError("config must be a FederationConfig or None")
    await _activate_config(validated)
    return _manager.status()


async def stop() -> None:
    lifecycle_lock, generation = _lifecycle_guard()
    async with lifecycle_lock:
        if not _lifecycle_generation_is_current(generation):
            return
        current = _manager
        try:
            await current.stop()
        except BaseException:
            if _lifecycle_generation_is_current(generation) and _manager is current and current.enabled:
                current.start_background_refresh()
            raise
        if _lifecycle_generation_is_current(generation) and _manager is current:
            current.relinquish_cache_write_ownership()
