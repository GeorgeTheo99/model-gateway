"""Databricks workspaces as one directory provider (docs/adr/0006-databricks-directory.md).

models.dev lists the Databricks Foundation Model APIs (prices, limits, reasoning
controls), but not which workspace serves what. Every configured workspace is
asked which models it serves (legacy serving endpoints, plus Unity model
services on an ``/ai-gateway`` route), and a model's route is the pool of every
workspace that serves it, in configured order: the reserved pool
:data:`AUTO_POOL`. A workspace joins a model's pool only once its route answers
a test completion; one that fails is left out, and tested again after a day.

Discovery in the background uses the CLI's cached authentication only: it never
starts browser sign-in. A workspace whose token can't be minted keeps the models
it was last seen serving and is reported as needing sign-in.

The coverage file (``models-dev/databricks.json`` beside ``config.yaml``) holds
model names and test results, never tokens.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path

import httpx

PROVIDER = "databricks"
AUTO_POOL = "databricks:auto"
"""The pool of every configured workspace that serves a row's model."""
COVERAGE_SCHEMA = 1
REDISCOVER_SECONDS = 24 * 60 * 60
FAILED_ROUTE_SECONDS = 24 * 60 * 60
LIST_TIMEOUT_SECONDS = 20.0
SMOKE_TIMEOUT_SECONDS = 60.0
_HOST_SUFFIXES = (".databricks.com", ".azuredatabricks.net")
_UNITY_PREFIX = "system.ai."
_LEGACY_PREFIX = "databricks-"

log = logging.getLogger("model-gateway")


class AuthError(Exception):
    """The workspace refused the token (401/403)."""


# ── Workspaces ─────────────────────────────────────────────────────────────


def _is_workspace(provider_id: str, block: dict) -> bool:
    if provider_id == PROVIDER or block.get("auth_refresh") == "databricks-cli":
        return True
    host = urllib.parse.urlsplit(str(block.get("base_url") or "")).hostname or ""
    return host.endswith(_HOST_SUFFIXES)


def workspaces(config: dict) -> list[str]:
    """Configured, enabled Databricks workspaces with a token, in config order."""
    from src import providers

    ids: list[str] = []
    for section in ("providers", "workspaces"):
        entries = config.get(section)
        if isinstance(entries, dict):
            ids.extend(providers._canonical_provider(str(key)) for key in entries)
    ids.append(PROVIDER)  # DATABRICKS_HOST / DATABRICKS_TOKEN configure it without a config block
    found: list[str] = []
    for provider_id in dict.fromkeys(ids):
        # Identify a workspace before resolving its key, which may read a key file.
        raw = {**providers._resolve_provider_config(config, provider_id), **providers._provider_env_config(provider_id)}
        if (raw.get("enabled") is False or raw.get("managed_by") == "custom" or not raw.get("base_url")
                or not _is_workspace(provider_id, raw)):
            continue
        if str(providers._effective_provider_config(config, provider_id).get("api_key") or "").strip():
            found.append(provider_id)
    return found


def style(block: dict) -> str:
    """``unity`` (an /ai-gateway route with path prefixes), ``invocations`` or ``serving``."""
    prefixes = block.get("path_prefixes")
    if isinstance(prefixes, dict) and prefixes:
        return "unity"
    return "invocations" if block.get("endpoint_style") == "invocations" else "serving"


def fingerprint(block: dict) -> str:
    """What a workspace's coverage depends on (no secrets): a changed route invalidates it."""
    native = block.get("invocations_native_protocols") or []
    facts = [str(block.get("base_url") or "").rstrip("/"), str(block.get("workspace_url") or "").rstrip("/"),
             style(block), block.get("path_prefixes") if isinstance(block.get("path_prefixes"), dict) else {},
             sorted([native] if isinstance(native, str) else native)]
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()[:16]


def supports(block: dict, protocol: str, api_style: str) -> bool:
    """Whether a workspace route can carry a model's wire API."""
    prefixes = block.get("path_prefixes") if isinstance(block.get("path_prefixes"), dict) else {}
    if api_style == "open_responses":
        return bool(prefixes.get("responses") or prefixes.get("openai"))
    if protocol == "anthropic":
        if block.get("endpoint_style") == "invocations":
            native = block.get("invocations_native_protocols") or []
            return "anthropic" in ([native] if isinstance(native, str) else native)
        return bool(prefixes.get("anthropic"))
    return True


def _jwt_claims(token: str) -> dict:
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError):
        return {}
    return claims if isinstance(claims, dict) else {}


def api_root(block: dict, token: str) -> str:
    """The workspace origin that serves the REST API (an AI Gateway hostname doesn't)."""
    workspace_url = str(block.get("workspace_url") or "").rstrip("/")
    if workspace_url:
        return workspace_url
    parts = urllib.parse.urlsplit(str(block.get("base_url") or ""))
    if (parts.hostname or "").endswith(".ai-gateway.cloud.databricks.com"):
        issuer = str(_jwt_claims(token).get("iss") or "")
        if issuer.endswith("/oidc"):
            return issuer.removesuffix("/oidc")
        raise ValueError("set workspace_url: an AI Gateway hostname doesn't serve the workspace API")
    return f"{parts.scheme}://{parts.netloc}"


# ── Model ids ──────────────────────────────────────────────────────────────


def twins(model_id: str) -> list[str]:
    """The names one Databricks model goes by: the legacy serving endpoint and the Unity model service."""
    core = model_id.removeprefix(_UNITY_PREFIX).removeprefix(_LEGACY_PREFIX)
    return list(dict.fromkeys([model_id, _LEGACY_PREFIX + core, _UNITY_PREFIX + core,
                               _UNITY_PREFIX + _LEGACY_PREFIX + core]))


def bind_directory_row(row: dict, model: dict) -> None:
    """Make a mapped models.dev Databricks row a route over every serving workspace (in place).

    models.dev names a model's wire API by its AI SDK package: Anthropic Messages for some
    Claude models, OpenAI Responses for a few GPT models, OpenAI Chat otherwise.
    """
    row["pool"] = AUTO_POOL
    npm = ((model.get("provider") or {}) if isinstance(model.get("provider"), dict) else {}).get("npm")
    if npm == "@ai-sdk/anthropic":
        row["protocol"] = "anthropic"
    elif npm == "@ai-sdk/openai":
        row["api_style"] = "open_responses"
    core = row["provider_model_id"].removeprefix(_UNITY_PREFIX).removeprefix(_LEGACY_PREFIX)
    alternates = [*(row.get("alternate_ids") or []), _LEGACY_PREFIX + core, _UNITY_PREFIX + core]
    row["alternate_ids"] = [alt for alt in dict.fromkeys(alternates) if alt != row["provider_model_id"]]


# ── Coverage ───────────────────────────────────────────────────────────────


def coverage_path(config_path: Path | None = None) -> Path:
    from src import directory

    return directory.directory_dir(config_path) / "databricks.json"


_cache: dict[str, tuple[int, dict]] = {}
_coverage_lock = threading.Lock()
"""Serializes read-modify-write of the coverage file across the event loop and worker threads."""


def _empty() -> dict:
    return {"schema": COVERAGE_SCHEMA, "workspaces": {}, "routes": {}}


def load_coverage(config_path: Path | None = None) -> dict:
    """The coverage file (cached by modification time): ``{workspaces: {...}, routes: {...}}``."""
    path = coverage_path(config_path)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return _empty()
    cached = _cache.get(str(path))
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("databricks: unreadable %s: %s", path, exc)
        value = None
    if not isinstance(value, dict) or value.get("schema") != COVERAGE_SCHEMA:
        value = _empty()
    value.setdefault("workspaces", {})
    value.setdefault("routes", {})
    _cache[str(path)] = (mtime, value)
    return value


def update_coverage(change: Callable[[dict], None], config_path: Path | None = None) -> bool:
    """Apply ``change`` to a fresh copy of the coverage file and write it (mode 0600) when it changed."""
    path = coverage_path(config_path)
    with _coverage_lock:
        current = load_coverage(config_path)
        value = {"schema": COVERAGE_SCHEMA, "workspaces": dict(current["workspaces"]),
                 "routes": dict(current["routes"])}
        before = json.dumps(value, sort_keys=True)
        change(value)
        if json.dumps(value, sort_keys=True) == before:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(value, indent=1, sort_keys=True) + "\n")
            os.replace(tmp, path)
        finally:
            Path(tmp).unlink(missing_ok=True)
        _cache.pop(str(path), None)
        return True


def route_key(workspace: str, block: dict, upstream: str, row: dict) -> str:
    return "|".join((workspace, upstream, row.get("protocol") or "openai", row.get("api_style") or "",
                     fingerprint(block)))


def _failed(record: dict | None, now: float) -> bool:
    return bool(record) and not record.get("ok") and now - float(record.get("at_epoch") or 0) < FAILED_ROUTE_SECONDS


_members_cache: dict = {"key": None, "found": {}}


def _serving(row: dict, config: dict, coverage: dict) -> list[tuple[str, str, str]]:
    """``(workspace, upstream id, route key)`` for every workspace that lists the row's model and whose
    route carries its wire API, in configured order."""
    from src import providers

    protocol, api_style = row.get("protocol") or "openai", row.get("api_style") or ""
    names = twins(str(row.get("provider_model_id") or row.get("name") or ""))
    found = []
    for workspace in workspaces(config):
        block = providers._effective_provider_config(config, workspace)
        record = coverage["workspaces"].get(workspace)
        route = fingerprint(block)
        if not record or record.get("fingerprint") != route or not supports(block, protocol, api_style):
            continue
        served = set(record.get("served") or ())
        upstream = next((name for name in names if name in served), None)
        if upstream is not None:
            found.append((workspace, upstream, route_key(workspace, block, upstream, row)))
    return found


def serving(row: dict, config: dict) -> list[tuple[str, str, str]]:
    """``(workspace, upstream id, route key)`` for every workspace listing the row's model, whatever its tests."""
    return _serving(row, config, load_coverage())


def forget_routes(keys: set[str], config_path: Path | None = None) -> None:
    """Drop these routes' test results, so they are tested again."""
    update_coverage(lambda coverage: coverage.update(
        routes={key: value for key, value in coverage["routes"].items() if key not in keys}), config_path)


def candidates(row: dict, config: dict, coverage: dict | None = None, *,
               now: float | None = None) -> list[tuple[str, str]]:
    """Workspaces serving the row's model whose route hasn't failed its test completion within a day."""
    coverage = load_coverage() if coverage is None else coverage
    if not coverage["workspaces"]:
        return []
    now = time.time() if now is None else now
    return [(workspace, upstream) for workspace, upstream, key in _serving(row, config, coverage)
            if not _failed(coverage["routes"].get(key), now)]


def members(row: dict, config: dict, coverage: dict | None = None, *,
            now: float | None = None) -> list[tuple[str, str]]:
    """``(workspace, upstream id)`` of the row's pool: every workspace serving its model over its wire API
    whose route answered its test completion, in configured order.

    Memoized per config object, coverage file and minute: requests resolve it several times."""
    coverage = load_coverage() if coverage is None else coverage
    if not coverage["workspaces"]:
        return []
    now = time.time() if now is None else now
    key = (id(config), id(coverage), int(now // 60))
    cache = _members_cache
    if cache["key"] != key or cache.get("config") is not config or cache.get("coverage") is not coverage:
        cache.update(key=key, config=config, coverage=coverage, found={})
    found_map = cache["found"]  # one generation for the read and the write
    row_key = (row.get("provider_model_id"), row.get("name"), row.get("protocol"), row.get("api_style"))
    found = found_map.get(row_key)
    if found is None:
        routes = coverage["routes"]
        found = [(workspace, upstream) for workspace, upstream, route in _serving(row, config, coverage)
                 if (routes.get(route) or {}).get("ok")]
        found_map[row_key] = found
    return list(found)


# ── Discovery ──────────────────────────────────────────────────────────────


def _get_json(url: str, token: str) -> dict:
    response = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=LIST_TIMEOUT_SECONDS)
    if response.status_code in (401, 403):
        raise AuthError(f"HTTP {response.status_code}")
    response.raise_for_status()
    value = response.json()
    return value if isinstance(value, dict) else {}


def list_served(root: str, token: str, *, unity: bool) -> set[str]:
    """Model names a workspace serves: its serving endpoints, plus Unity model services when asked.

    A workspace without the Unity API (404) or without access to it (403) lists endpoints only; any
    other failure fails the check, so the previous list is kept."""
    served = {str(endpoint.get("name")) for endpoint in _get_json(f"{root}/api/2.0/serving-endpoints", token)
              .get("endpoints") or [] if isinstance(endpoint, dict) and endpoint.get("name")}
    if not unity:
        return served
    page, seen = "", set()
    while True:
        query = urllib.parse.urlencode({"parent": "schemas/system.ai", **({"page_token": page} if page else {})})
        try:
            data = _get_json(f"{root}/api/2.1/unity-catalog/model-services?{query}", token)
        except AuthError:
            log.info("databricks: no access to Unity model services at %s", root)
            return served
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            log.info("databricks: no Unity model services API at %s", root)
            return served
        for service in data.get("model_services") or []:
            name = str(service.get("name") or "") if isinstance(service, dict) else ""
            if name:
                served.add(name.removeprefix("model-services/"))
        page = str(data.get("next_page_token") or "")
        if not page or page in seen:
            return served
        seen.add(page)


def _stale(record: dict | None, block: dict, now: float) -> bool:
    return (not record or record.get("fingerprint") != fingerprint(block)
            or now - float(record.get("checked_epoch") or 0) >= REDISCOVER_SECONDS)


def _utc(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


async def _check(workspace: str, record: dict | None) -> dict:
    """List one workspace's models with cached CLI authentication only; the new coverage record."""
    from src import providers

    now = time.time()
    await providers.ensure_fresh_oauth_token(workspace, allow_login=False)
    served, error, auth_needed = None, None, False
    block: dict = {}
    for attempt in range(2):
        block = providers._effective_provider_config(providers._load_config(), workspace)
        token = str(block.get("api_key") or "")
        try:
            served = await asyncio.to_thread(list_served, api_root(block, token), token,
                                             unity=style(block) == "unity")
            break
        except AuthError as exc:
            error, auth_needed = f"the workspace refused its token ({exc})", True
            if attempt or not await providers.refresh_oauth_token(workspace, force=True, allow_login=False,
                                                                  failed_token=token):
                break
        except Exception as exc:  # noqa: BLE001 — one workspace's failure never stops the others
            error = f"{type(exc).__name__}: {exc}"
            if token:
                error = error.replace(token, "[redacted]")
            break
    route = fingerprint(block)
    if served is not None:
        return {"fingerprint": route, "style": style(block), "served": sorted(served),
                "checked_at": _utc(now), "checked_epoch": now, "error": None, "auth_needed": False}
    same_route = bool(record) and record.get("fingerprint") == route
    log.warning("databricks: %s not checked: %s", workspace, error)
    return {**(record if same_route else {}), "fingerprint": route, "style": style(block),
            "served": sorted((record or {}).get("served") or ()) if same_route else [],
            "error": (error or "")[:300] or None, "auth_needed": auth_needed, "attempted_at": _utc(now)}


async def discover(*, only: set[str] | None = None, stale_only: bool = False, clear_failures: bool = False,
                   config_path: Path | None = None) -> dict:
    """Ask workspaces which models they serve, with cached CLI authentication only.

    ``stale_only`` skips workspaces checked within a day whose route is unchanged. ``clear_failures``
    forgets the checked workspaces' failed test completions, so they are tried again. Only the
    workspaces checked here are written back. Returns ``{changed, workspaces: {id: summary}}``.
    """
    from src import providers

    configured = workspaces(providers._load_config())
    current = load_coverage(config_path)
    checked: dict[str, dict] = {}
    for workspace in configured:
        if only is not None and workspace not in only:
            continue
        record = current["workspaces"].get(workspace)
        block = providers._effective_provider_config(providers._load_config(), workspace)
        if stale_only and not _stale(record, block, time.time()):
            continue
        checked[workspace] = await _check(workspace, record)

    def change(coverage: dict) -> None:
        coverage["workspaces"].update(checked)
        drop = set(coverage["workspaces"]) - set(configured)  # a removed workspace's coverage goes with it
        for gone in drop:
            coverage["workspaces"].pop(gone)
        coverage["routes"] = {
            key: value for key, value in coverage["routes"].items()
            if key.split("|", 1)[0] not in drop
            and not (clear_failures and key.split("|", 1)[0] in checked and not value.get("ok"))}

    changed = update_coverage(change, config_path)
    summary = {workspace: {"served": len(record["served"]), "error": record.get("error"),
                           "auth_needed": record.get("auth_needed", False), "style": record["style"]}
               for workspace, record in checked.items()}
    return {"changed": changed, "workspaces": summary}


# ── Test completions ───────────────────────────────────────────────────────


def smoke_route(row: dict, workspace: str, upstream: str, config: dict) -> str | None:
    """One cheap completion through the exact route a request would take; the error, or None.

    Raises :class:`AuthError` when the workspace refuses its token: that says nothing about the route."""
    from src import providers
    from src.request_options import normalize_token_limit, upstream_auth_headers

    entry = {key: value for key, value in row.items() if key != "pool"}
    entry["provider_model_id"] = upstream
    info = providers.route_info(entry, str(row.get("name") or upstream), workspace, config)
    anthropic = info.protocol == "anthropic"
    if info.api_style == "open_responses":
        url = info.base_url
        body: dict = {"model": upstream, "input": "Reply with OK.", "max_output_tokens": 16, "stream": False}
    else:
        body = {"model": upstream, "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 16, "stream": False}
        if not anthropic:
            normalize_token_limit(body, info.quirks)
        url = info.base_url if info.endpoint_suffix == "" else \
            info.base_url.rstrip("/") + ("/messages" if anthropic else "/chat/completions")
    headers = {**upstream_auth_headers(info.api_key, "anthropic" if anthropic else "openai", info.quirks),
               "User-Agent": "model-gateway-databricks/1"}
    try:
        response = httpx.post(url, json=body, headers=headers, timeout=SMOKE_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        return f"test completion failed: {type(exc).__name__}"
    if response.status_code == 401:  # a 403 is the endpoint's permission: that route's verdict
        raise AuthError("HTTP 401")
    if response.status_code != 200:
        text = response.text[:300]
        if info.api_key:
            text = text.replace(info.api_key, "[redacted]")
        return f"the workspace refused a test completion (HTTP {response.status_code}): {text[:200]}"
    try:
        payload = response.json()
    except ValueError:
        return "test completion returned no JSON"
    if not isinstance(payload, dict):
        return "test completion returned no reply"
    if info.api_style == "open_responses":
        return None if isinstance(payload.get("output"), list) else "test completion returned no reply"
    if anthropic:
        return None if isinstance(payload.get("content"), list) else "test completion returned no reply"
    choices = payload.get("choices")
    return None if isinstance(choices, list) and choices and isinstance(choices[0], dict) else \
        "test completion returned no reply"


def untested(row: dict, config: dict, coverage: dict | None = None, *,
             now: float | None = None) -> list[tuple[str, str, str]]:
    """``(workspace, upstream id, route key)`` for routes serving the row's model that have no test result,
    or whose failure is over a day old."""
    coverage = load_coverage() if coverage is None else coverage
    now = time.time() if now is None else now
    found = []
    for workspace, upstream, key in _serving(row, config, coverage):
        record = coverage["routes"].get(key)
        if record is None or (not record.get("ok") and not _failed(record, now)):
            found.append((workspace, upstream, key))
    return found


def record_routes(results: dict[str, str | None], config_path: Path | None = None) -> bool:
    """Store test results (``{route key: error or None}``); True when a route's pool membership changed."""
    if not results:
        return False
    now = time.time()
    before: dict[str, object] = {}

    def change(coverage: dict) -> None:
        for key, error in results.items():
            before[key] = (coverage["routes"].get(key) or {}).get("ok")
            coverage["routes"][key] = {"ok": error is None, "error": error, "at": _utc(now), "at_epoch": now}

    update_coverage(change, config_path)
    return any(before[key] != (error is None) for key, error in results.items())


def restore_coverage(previous: bytes | None, config_path: Path | None = None) -> None:
    """Put back the coverage file the catalog accepted, keeping test results recorded since."""
    try:
        old = json.loads(previous) if previous else _empty()
    except ValueError:
        old = _empty()

    def change(coverage: dict) -> None:
        coverage["workspaces"] = dict(old.get("workspaces") or {})

    update_coverage(change, config_path)


def test_routes(rows: list[dict], config: dict) -> bool:
    """Test every untested route of the Databricks rows given; True when a pool changed."""
    coverage = load_coverage()
    results: dict[str, str | None] = {}
    refused: set[str] = set()
    for row in rows:
        if row.get("pool") != AUTO_POOL:
            continue
        for workspace, upstream, key in untested(row, config, coverage):
            if workspace in refused or key in results:
                continue
            try:
                results[key] = smoke_route(row, workspace, upstream, config)
            except AuthError:
                refused.add(workspace)  # a token problem, reported by discovery: no route verdict
                continue
            if results[key]:
                log.warning("databricks: %s on %s failed its test completion: %s",
                            row.get("name"), workspace, results[key])
    return record_routes(results)


def status(config: dict, rows: list[dict]) -> dict:
    """Secret-free coverage for the admin API: each workspace and each Databricks route's pool."""
    from src import providers

    coverage = load_coverage()
    result = []
    for workspace in workspaces(config):
        block = providers._effective_provider_config(config, workspace)
        record = coverage["workspaces"].get(workspace) or {}
        current = record.get("fingerprint") == fingerprint(block)
        result.append({"id": workspace, "style": style(block), "checked": current,
                       "served": len(record.get("served") or ()) if current else 0,
                       "checked_at": record.get("checked_at") if current else None,
                       "error": record.get("error") if current else None,
                       "auth_needed": bool(record.get("auth_needed")) if current else False})
    failures = {key: value for key, value in coverage["routes"].items() if not value.get("ok")}
    models = [{"name": row["name"], "provider_model_id": row.get("provider_model_id"),
               "pool": [{"workspace": workspace, "upstream_id": upstream}
                        for workspace, upstream in members(row, config, coverage)]}
              for row in rows if row.get("pool") == AUTO_POOL]
    return {"workspaces": result, "models": models, "failed_routes": failures}


def unmapped(config: dict, mapped_ids: set[str]) -> dict[str, list[str]]:
    """Served model names the directory doesn't map, with the workspaces serving each."""
    coverage = load_coverage()
    known = {name for model_id in mapped_ids for name in twins(model_id)}
    found: dict[str, list[str]] = {}
    for workspace in workspaces(config):
        for name in (coverage["workspaces"].get(workspace) or {}).get("served") or ():
            if name not in known:
                found.setdefault(name, []).append(workspace)
    return dict(sorted(found.items()))
