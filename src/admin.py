"""Admin API and lightweight UI for model-gateway."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src.auth import admin_writes_enabled, auth_mode, require_admin_auth, require_admin_writes
from src.config_lock import config_write_lock
from src.providers import (
    MODEL_INFO_PATH,
    config_validation,
    model_status,
    provider_status,
    standalone_provider_status,
    workspace_pool_status,
    registry_transaction as provider_registry_transaction,
    reload as reload_provider_registry,
    restore_registry as restore_provider_registry,
    routable_ids,
    snapshot_registry as snapshot_provider_registry,
)
from src import config_io, federation, ledger

router = APIRouter()
_STARTED_AT = time.time()


@router.get("/admin", response_class=HTMLResponse)
async def admin_ui():
    return HTMLResponse(_ADMIN_HTML)


@router.get("/admin/api/status")
async def admin_status(request: Request):
    require_admin_auth(request)
    mode = auth_mode()
    return {
        "service": "model-gateway",
        "status": "ok",
        "pid": os.getpid(),
        "uptime_seconds": round(time.time() - _STARTED_AT, 3),
        "model_info_path": str(MODEL_INFO_PATH),
        "config_path": str(config_io.CONFIG_PATH),
        "writes_enabled": mode.writes_enabled,
        "auth": {
            "client_auth_enabled": mode.client_auth_enabled,
            "admin_auth_enabled": mode.admin_auth_enabled,
            "client_key_count": mode.client_key_count,
            "admin_key_configured": mode.admin_key_configured,
            "unsafe_admin_without_key": mode.unsafe_admin_without_key,
            "warning": mode.warning,
        },
    }


@router.get("/admin/api/providers")
async def admin_providers(request: Request):
    require_admin_auth(request)
    return {"providers": provider_status()}


@router.get("/admin/api/models")
async def admin_models(request: Request):
    require_admin_auth(request)
    return {"models": _admin_model_status()}


def _admin_model_status() -> list[dict]:
    """Add configured image routing without probing or exporting private profiles."""
    # Delayed import avoids the admin/server module cycle at startup.
    from src.server import _configured_vision_fallback_model, _configured_vision_fallback_mode

    rows = model_status()
    for row in rows:
        row["vision_route"] = None
        if not row.get("vision") and not row.get("composite") and row.get("locality") in {"local", "cloud"}:
            helper = _configured_vision_fallback_model(row["locality"])
            if helper:
                row["vision_route"] = {
                    "model": helper,
                    "mode": _configured_vision_fallback_mode(),
                }
    return rows


@router.get("/admin/api/workspace-pools")
async def admin_workspace_pools(request: Request):
    """Workspace routing state plus bounded, redacted recent activity."""
    require_admin_auth(request)
    pools = workspace_pool_status()
    for pool in pools:
        for member in pool["members"]:
            member["recent"] = _recent_activity(member["id"])
    standalone = standalone_provider_status()
    for provider in standalone:
        provider["recent"] = _recent_activity(provider["id"])
    states = {"healthy": 0, "degraded": 0, "down": 0}
    workspace_ids = set()
    for pool in pools:
        states[pool["state"]] += 1
        workspace_ids.update(member["id"] for member in pool["members"])
    return {
        "summary": {
            "pools": len(pools),
            "workspaces": len(workspace_ids),
            "standalone_providers": len(standalone),
            **states,
        },
        "pools": pools,
        "standalone_providers": standalone,
    }


def _recent_activity(provider_id: str) -> dict:
    """Bounded, redacted recent-request summary for a single provider."""
    recent = ledger.recent(limit=100, provider=provider_id)
    failures = [
        row for row in recent
        if row.get("error") or (row.get("status") is not None and row["status"] >= 400)
    ]
    rate_limits = sum(1 for row in recent if row.get("status") == 429)
    return {
        "requests": len(recent),
        "failures": len(failures),
        "rate_limits": rate_limits,
        "last_status": recent[0].get("status") if recent else None,
        "last_request_at": recent[0].get("ts") if recent else None,
        "last_failure_status": failures[0].get("status") if failures else None,
        "last_failure_at": failures[0].get("ts") if failures else None,
    }
    states = {"healthy": 0, "degraded": 0, "down": 0}
    workspace_ids = set()
    for pool in pools:
        states[pool["state"]] += 1
        workspace_ids.update(member["id"] for member in pool["members"])
    return {
        "summary": {
            "pools": len(pools),
            "workspaces": len(workspace_ids),
            **states,
        },
        "pools": pools,
    }


@router.get("/admin/api/presets")
async def admin_presets(request: Request):
    """Read-only gateway-owned model aggregate/preset definitions."""
    require_admin_auth(request)
    if not MODEL_INFO_PATH.exists():
        return {"auto_models": {}, "model_presets": {}, "routing_profiles": {}}
    with open(MODEL_INFO_PATH) as f:
        data = json.load(f)
    return {
        "auto_models": data.get("auto_models") or {},
        "model_presets": data.get("model_presets") or {},
        "routing_profiles": data.get("routing_profiles") or {},
    }


@router.get("/admin/api/config/validation")
async def admin_config_validation(request: Request):
    require_admin_auth(request)
    return config_validation()


def _reload_registry_transactionally(provider_snapshot) -> str | None:
    """Eagerly load and validate a new registry, restoring live state on error."""
    with provider_registry_transaction():
        reload_provider_registry()
        try:
            snapshot_provider_registry()
            # Local import avoids the admin ↔ server module cycle at import time.
            from src.auth import validate_credential_separation
            from src.server import _clear_vision_observation_cache, _validate_vision_fallback_policy
            _validate_vision_fallback_policy()
            validate_credential_separation()
            _clear_vision_observation_cache()
        except Exception as exc:  # noqa: BLE001 — malformed registries must roll back
            restore_provider_registry(provider_snapshot)
            return str(exc)
        return None


def _apply_registry_mutation(mutate):
    """Run one locked admin file mutation and publish it only after validation."""
    with config_write_lock(config_io.CONFIG_PATH):
        provider_snapshot = snapshot_provider_registry()
        file_snapshot = config_io.snapshot_writable_files()
        try:
            result = mutate()
        except Exception:
            config_io.restore_writable_files(file_snapshot)
            restore_provider_registry(provider_snapshot)
            raise
        reload_error = _reload_registry_transactionally(provider_snapshot)
        if reload_error is None:
            return result, None
        try:
            config_io.restore_writable_files(file_snapshot)
        except Exception as exc:  # noqa: BLE001
            return None, f"{reload_error}; file rollback failed: {exc}"
        return None, f"{reload_error}; changes rolled back"


@router.post("/admin/api/reload")
async def admin_reload(request: Request):
    require_admin_auth(request)
    require_admin_writes()
    # Validate federation before invalidating the live provider registry. A
    # malformed shared YAML/federation block must leave both registries intact.
    with config_write_lock(config_io.CONFIG_PATH):
        try:
            federation_config = federation.load_config(config_io.CONFIG_PATH)
        except federation.FederationConfigError as exc:
            return _bad_request(str(exc))
        provider_snapshot = snapshot_provider_registry()
        reload_error = _reload_registry_transactionally(provider_snapshot)
        if reload_error is not None:
            return _bad_request(f"Provider registry reload rejected: {reload_error}")
    federation_status = await federation.reconfigure(config=federation_config)
    catalogs = await _regenerate_catalogs()
    return {
        "status": "ok",
        "message": "provider registry and federation reloaded",
        "catalogs": catalogs,
        "federation": federation_status,
    }


async def _regenerate_catalogs() -> str:
    """Re-render the downstream alias catalog (model-aliases.json) after a
    config change. Best-effort: catalog drift is never allowed to fail a reload."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_catalogs.py"
    if not script.exists():
        return "skipped (script missing)"
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script), "--config", str(config_io.CONFIG_PATH),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode == 0:
            return "regenerated"
        return f"failed: {out.decode(errors='replace')[:200]}"
    except Exception as exc:  # noqa: BLE001 — reload must survive catalog errors
        return f"failed: {exc}"


@router.get("/admin/api/usage")
async def admin_usage(request: Request):
    """Aggregate usage/cost by provider, requested model, route, endpoint, and status.

    Query params:
      since   - epoch seconds (inclusive)
      until   - epoch seconds (exclusive)
      window  - shorthand: '1h', '24h', '7d', '30d' (sets `since`)
    """
    require_admin_auth(request)
    since, until = _time_bounds(request.query_params)

    return {
        "summary": ledger.summary(since=since, until=until),
        "by_provider": ledger.aggregate(since=since, until=until, group_by="provider"),
        # Retain requested-model aggregation for API compatibility. The dashboard
        # uses by_route so aliases for one configured model appear only once.
        "by_model": ledger.aggregate(since=since, until=until, group_by="model"),
        "by_route": ledger.aggregate(since=since, until=until, group_by="route"),
        "by_endpoint": ledger.aggregate(since=since, until=until, group_by="endpoint"),
        "by_status": ledger.aggregate(since=since, until=until, group_by="status"),
    }


@router.get("/admin/api/requests")
async def admin_requests(request: Request):
    """Recent redacted ledger rows (no prompt/completion content)."""
    require_admin_auth(request)
    limit = 50
    try:
        limit = max(1, min(500, int(request.query_params.get("limit", 50))))
    except ValueError:
        pass
    return {"requests": ledger.recent(limit=limit)}


def _time_bounds(params) -> tuple[float | None, float | None]:
    """Parse window ('1h','24h','7d','30d') / since / until query params."""
    import time as _time

    since = None
    until = None
    window = params.get("window")
    if window:
        units = {"h": 3600, "d": 86400}
        try:
            secs = int(window[:-1]) * units[window[-1].lower()]
            since = _time.time() - secs
        except (KeyError, ValueError):
            pass
    if params.get("since"):
        try:
            since = float(params["since"])
        except ValueError:
            pass
    if params.get("until"):
        try:
            until = float(params["until"])
        except ValueError:
            pass
    return since, until


@router.get("/admin/api/requests/{request_id}")
async def admin_request_detail(request_id: str, request: Request):
    """One redacted ledger row by id, for the request drill-down panel."""
    require_admin_auth(request)
    row = ledger.get_request(request_id)
    if row is None:
        return _bad_request(f"request {request_id!r} not found", status=404)
    return {"request": row}


@router.get("/admin/api/providers/{provider_id}/stats")
async def admin_provider_stats(provider_id: str, request: Request):
    """Per-provider config + usage/cost + recent requests for a window."""
    require_admin_auth(request)
    since, until = _time_bounds(request.query_params)
    rows = [p for p in provider_status() if p.get("id") == provider_id]
    provider = rows[0] if rows else None
    if provider:
        usage = ledger.summary(since=since, until=until, provider=provider_id)
        recent = ledger.recent(limit=25, provider=provider_id)
    else:
        usage = {}
        recent = []
    return {"provider": provider, "usage": usage, "recent": recent}


@router.get("/admin/api/models/{model_name}/stats")
async def admin_model_stats(model_name: str, request: Request):
    """Per-model config + usage/cost + recent requests for a window.

    Query params: window ('1h','24h','7d','30d'), since/until epoch seconds.
    Matches ledger rows by every routable identifier for the model (name,
    alias, provider_model_id, omlx_id) so stats follow whichever alias the
    caller sent. Requires admin auth.
    """
    require_admin_auth(request)
    since, until = _time_bounds(request.query_params)

    # Resolve the model config from model_status (find by name).
    rows = [m for m in _admin_model_status() if (m.get("name") or m.get("id")) == model_name]
    model = rows[0] if rows else None
    # Only query the ledger when the model actually exists; an unknown name
    # would otherwise fall back to a self-named id and return NULL-filled rows.
    if model:
        ids = routable_ids(model_name)
        usage = ledger.summary(since=since, until=until, models=ids) if ids else {}
        recent = ledger.recent(limit=25, models=ids) if ids else []
    else:
        ids = []
        usage = {}
        recent = []
    return {
        "model": model,
        "routable_ids": ids,
        "usage": usage,
        "recent": recent,
    }


# ── Provider management (writeable) ──────────────────────────────────────────


@router.post("/admin/api/providers/{provider_id}")
async def admin_upsert_provider(provider_id: str, request: Request):
    """Create or update a provider block in config.yaml.

    Body: {base_url, protocol?, api_key?, default_headers?}. ``api_key`` is
    write-only (None preserves the existing key; "" removes it). Reloads the
    provider registry after writing.
    """
    require_admin_auth(request)
    require_admin_writes()
    try:
        body = await request.json()
    except Exception:
        return _bad_request("Invalid JSON body")
    try:
        result, reload_error = _apply_registry_mutation(lambda: config_io.upsert_provider(
            provider_id,
            base_url=body.get("base_url", ""),
            api_key=body.get("api_key"),
            protocol=body.get("protocol"),
            default_headers=body.get("default_headers"),
        ))
    except ValueError as exc:
        return _bad_request(str(exc))
    if reload_error is not None:
        return _bad_request(f"Provider registry update rejected: {reload_error}")
    result["reloaded"] = True
    return result


@router.delete("/admin/api/providers/{provider_id}")
async def admin_delete_provider(provider_id: str, request: Request):
    """Remove a provider. Refuses if enabled models depend on it."""
    require_admin_auth(request)
    require_admin_writes()
    try:
        result, reload_error = _apply_registry_mutation(
            lambda: config_io.delete_provider(provider_id)
        )
    except KeyError as exc:
        return _bad_request(str(exc), status=404)
    except ValueError as exc:
        return _bad_request(str(exc), status=409)
    if reload_error is not None:
        return _bad_request(f"Provider registry update rejected: {reload_error}")
    return result


@router.post("/admin/api/providers/{provider_id}/validate")
async def admin_validate_provider(provider_id: str, request: Request):
    """Lightweight upstream validation: authenticated GET {base_url}/models.

    Read-only upstream probe. Returns {ok, status_code, model_count?, error?}.
    """
    require_admin_auth(request)
    require_admin_writes()
    import httpx
    import src.providers as providers

    provider_id = providers._canonical_provider(provider_id.strip().lower())
    config = providers._load_config()
    block = providers._effective_provider_config(config, provider_id)
    if not block:
        return _bad_request(f"provider {provider_id!r} not configured", status=404)
    base_url = block.get("base_url", "")
    api_key = block.get("api_key", "")
    protocol = block.get("protocol", "openai")
    if not base_url or not api_key:
        return _bad_request("provider missing base_url or api_key")

    headers = {"Authorization": f"Bearer {api_key}"}
    if protocol == "anthropic":
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    url = f"{base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers=headers)
        ok = resp.status_code == 200
        model_count = None
        if ok:
            try:
                data = resp.json()
                items = data.get("data") if isinstance(data, dict) else data
                model_count = len(items) if isinstance(items, list) else None
            except Exception:  # noqa: BLE001
                pass
        return {"ok": ok, "status_code": resp.status_code, "model_count": model_count, "error": None if ok else resp.text[:300]}
    except httpx.HTTPError as exc:
        return {"ok": False, "status_code": None, "model_count": None, "error": str(exc)}


@router.post("/admin/api/providers/{provider_id}/discover")
async def admin_discover_provider_models(provider_id: str, request: Request):
    """List upstream model ids from the provider's /models endpoint.

    Read-only upstream probe (no config writes). Marks which ids are already
    registered in the catalog so the UI can offer only new ones.
    """
    require_admin_auth(request)
    require_admin_writes()
    import src.providers as providers
    from src.onboarding_generation import discover_models

    provider_id = providers._canonical_provider(provider_id.strip().lower())
    config = providers._load_config()
    block = providers._effective_provider_config(config, provider_id)
    if not block:
        return _bad_request(f"provider {provider_id!r} not configured", status=404)
    base_url = block.get("base_url", "")
    if not base_url:
        return _bad_request("provider missing base_url")

    result = await asyncio.to_thread(
        discover_models, base_url, block.get("api_key") or None
    )
    registered: set[str] = set()
    for entry in {id(v): v for v in providers._load_models().values()}.values():
        for key in ("provider_model_id", "omlx_id", "name", "alias"):
            value = entry.get(key)
            if value:
                registered.add(str(value))
    models = [
        {"id": mid, "registered": mid in registered}
        for mid in result.get("model_ids", [])
    ]
    return {
        "status": result.get("status"),
        "http_status": result.get("http_status"),
        "models": models,
    }


# ── Model management (writeable) ────────────────────────────────────────────


@router.post("/admin/api/models/{model_name}/preview")
async def admin_preview_model(model_name: str, request: Request):
    """Dry-run a model upsert: validate + resolve routing without writing.

    Accepts the same body as the upsert endpoint. Returns the normalized
    entry, validation issues, routable-id clashes with other models, and the
    provider route (pool members + which are usable) the entry would take.
    """
    require_admin_auth(request)
    require_admin_writes()
    from src.catalog import (
        entry_routable_ids,
        normalize_thinking_capabilities,
        validate_pricing_policy,
    )
    import src.providers as providers

    try:
        body = await request.json()
    except Exception:
        return _bad_request("Invalid JSON body")

    name = (model_name or "").strip()
    issues: list[str] = []
    entry: dict = {"name": name}
    if not name:
        issues.append("model name is required")
    provider = (body.get("provider") or "").strip().lower()
    if not provider:
        issues.append("provider is required")
    else:
        entry["provider"] = provider
    for field in ("provider_model_id", "omlx_id", "alias", "context",
                  "max_output_tokens", "thinking", "thinking_levels",
                  "thinking_format", "desc", "pool"):
        value = body.get(field)
        if value is not None and value != "":
            entry[field] = value
    if provider and provider not in {"local", "omlx", "mlx"} and not entry.get("provider_model_id"):
        issues.append("provider_model_id is required")
    if provider in {"local", "omlx", "mlx"} and not entry.get("omlx_id") and not entry.get("provider_model_id"):
        issues.append("omlx_id or provider_model_id is required for local/oMLX models")
    # Mirror config_io._apply_pricing_update: 'unmetered' is the only stored
    # marker; 'metered' keeps numeric pricing; 'unknown' stores neither.
    pricing_status = (body.get("pricing_status") or "").strip().lower()
    if pricing_status == "unmetered":
        entry["pricing_status"] = "unmetered"
    elif body.get("pricing") is not None and pricing_status != "unknown":
        entry["pricing"] = body["pricing"]
    if body.get("vision") is not None:
        entry["vision"] = bool(body["vision"])

    try:
        validate_pricing_policy(entry)
        entry.update(normalize_thinking_capabilities(entry))
    except ValueError as exc:
        issues.append(str(exc))

    # Routable-id clashes with *other* catalog entries.
    clashes: list[dict] = []
    try:
        candidate_ids = entry_routable_ids(entry)
    except ValueError as exc:
        candidate_ids = []
        issues.append(str(exc))
    models = providers._load_models()
    for rid in candidate_ids:
        existing = models.get(rid)
        if existing is not None and existing.get("name") != name:
            clashes.append({"id": rid, "model": existing.get("name")})

    # Provider route the entry would take.
    config = providers._load_config()
    route: list[dict] = []
    if provider:
        for member in providers._pool_members(entry, config):
            member_config = providers._effective_provider_config(config, member)
            usable = bool(
                member_config.get("base_url")
                and member_config.get("api_key")
                and member_config.get("enabled") is not False
            )
            reason = None
            if not member_config:
                reason = "not configured"
            elif member_config.get("enabled") is False:
                reason = "disabled"
            elif not member_config.get("base_url"):
                reason = "missing base_url"
            elif not member_config.get("api_key"):
                reason = "missing api_key"
            route.append({"provider": member, "usable": usable, "reason": reason})
    routable = not issues and not clashes and any(m["usable"] for m in route)

    return {
        "ok": not issues and not clashes,
        "routable": routable,
        "issues": issues,
        "clashes": clashes,
        "route": route,
        "routable_ids": candidate_ids,
        "entry": {k: v for k, v in entry.items() if k != "api_key"},
        "exists": any(m.get("name") == name for m in {id(v): v for v in models.values()}.values()),
    }


@router.post("/admin/api/models/{model_name}")
async def admin_upsert_model(model_name: str, request: Request):
    """Create or update a model entry in model-info.json.

    Body fields: provider, provider_model_id (or omlx_id for local/oMLX),
    alias, context, max_output_tokens, thinking, thinking_levels, thinking_format, vision,
    system_instruction, pricing, pricing_status, desc, enabled.
    Writes the live catalog and optional machine-local mirror; reloads registry.
    """
    require_admin_auth(request)
    require_admin_writes()
    try:
        body = await request.json()
    except Exception:
        return _bad_request("Invalid JSON body")
    def mutate_model():
        result = config_io.upsert_model(model_name, **body)
        if "enabled" in body:
            enabled_result = config_io.set_model_enabled(model_name, bool(body["enabled"]))
            result["enabled"] = enabled_result["enabled"]
            result["written_to"] = list(dict.fromkeys([
                *result.get("written_to", []),
                *enabled_result.get("written_to", []),
            ]))
        return result

    try:
        result, reload_error = _apply_registry_mutation(mutate_model)
    except ValueError as exc:
        return _bad_request(str(exc))
    if reload_error is not None:
        return _bad_request(f"Provider registry update rejected: {reload_error}")
    result["reloaded"] = True
    return result


@router.delete("/admin/api/models/{model_name}")
async def admin_delete_model(model_name: str, request: Request):
    """Remove a model entry by name."""
    require_admin_auth(request)
    require_admin_writes()
    try:
        result, reload_error = _apply_registry_mutation(
            lambda: config_io.delete_model(model_name)
        )
    except KeyError as exc:
        return _bad_request(str(exc), status=404)
    if reload_error is not None:
        return _bad_request(f"Provider registry update rejected: {reload_error}")
    return result


@router.post("/admin/api/models/{model_name}/enable")
async def admin_enable_model(model_name: str, request: Request):
    require_admin_auth(request)
    require_admin_writes()
    try:
        result, reload_error = _apply_registry_mutation(
            lambda: config_io.set_model_enabled(model_name, True)
        )
    except KeyError as exc:
        return _bad_request(str(exc), status=404)
    if reload_error is not None:
        return _bad_request(f"Provider registry update rejected: {reload_error}")
    return result


@router.post("/admin/api/models/{model_name}/disable")
async def admin_disable_model(model_name: str, request: Request):
    require_admin_auth(request)
    require_admin_writes()
    try:
        result, reload_error = _apply_registry_mutation(
            lambda: config_io.set_model_enabled(model_name, False)
        )
    except KeyError as exc:
        return _bad_request(str(exc), status=404)
    if reload_error is not None:
        return _bad_request(f"Provider registry update rejected: {reload_error}")
    return result


def _bad_request(message: str, status: int = 400):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status, content={"error": {"message": message}})


_ADMIN_HTML = r"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Model Gateway</title>
    <style>
      :root {
        color-scheme: light dark;
        --bg: oklch(98% 0.006 220);
        --surface: oklch(99.5% 0.002 220);
        --surface-2: oklch(95.5% 0.008 220);
        --text: oklch(26% 0.018 220);
        --muted: oklch(46% 0.02 220);
        --rule: oklch(85% 0.015 220);
        --accent: oklch(43% 0.085 220);
        --accent-bg: oklch(93% 0.024 220);
        --ok: oklch(39% 0.09 155);
        --ok-bg: oklch(95% 0.025 155);
        --warn: oklch(42% 0.09 65);
        --warn-bg: oklch(96% 0.035 80);
        --bad: oklch(43% 0.14 25);
        --bad-bg: oklch(96% 0.02 25);
        --radius: 8px;
      }
      @media (prefers-color-scheme: dark) {
        :root {
          --bg: oklch(18% 0.008 220);
          --surface: oklch(21% 0.008 220);
          --surface-2: oklch(25% 0.012 220);
          --text: oklch(94% 0.008 220);
          --muted: oklch(75% 0.015 220);
          --rule: oklch(39% 0.02 220);
          --accent: oklch(80% 0.09 210);
          --accent-bg: oklch(29% 0.04 220);
          --ok: oklch(82% 0.1 155);
          --ok-bg: oklch(27% 0.025 155);
          --warn: oklch(85% 0.1 80);
          --warn-bg: oklch(29% 0.03 80);
          --bad: oklch(82% 0.09 25);
          --bad-bg: oklch(28% 0.035 25);
        }
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font:
          0.9375rem/1.55 -apple-system,
          BlinkMacSystemFont,
          "Segoe UI",
          sans-serif;
      }
      button,
      input,
      select,
      textarea {
        font: inherit;
      }
      button,
      a,
      input,
      select,
      textarea,
      summary {
        touch-action: manipulation;
      }
      button {
        cursor: pointer;
      }
      button:disabled {
        cursor: wait;
        opacity: 0.6;
      }
      a {
        color: var(--accent);
        text-underline-offset: 3px;
      }
      :focus-visible {
        outline: 3px solid var(--accent);
        outline-offset: 3px;
      }
      h1,
      h2,
      h3,
      p {
        margin: 0;
      }
      h1 {
        font-size: 1.75rem;
        letter-spacing: -0.035em;
        line-height: 1.25;
      }
      h2 {
        font-size: 1.125rem;
        line-height: 1.4;
      }
      h3 {
        font-size: 1rem;
      }
      p {
        max-width: 75ch;
      }
      code,
      .id,
      pre {
        font:
          0.8125rem/1.6 ui-monospace,
          SFMono-Regular,
          Consolas,
          monospace;
        overflow-wrap: anywhere;
      }
      pre {
        white-space: pre-wrap;
        background: var(--surface-2);
        padding: 16px;
        border-radius: var(--radius);
      }
      .hidden,
      [hidden] {
        display: none !important;
      }
      .muted,
      .small,
      .meta,
      .detail,
      .k {
        color: var(--muted);
      }
      .small,
      .meta,
      .detail,
      .k {
        font-size: 0.8125rem;
      }
      .ok-c {
        color: var(--ok);
      }
      .warn-c {
        color: var(--warn);
      }
      .bad-c {
        color: var(--bad);
      }
      .num {
        font-variant-numeric: tabular-nums;
        text-align: right;
      }
      .skip {
        position: absolute;
        left: 16px;
        top: -100px;
        background: var(--surface);
        padding: 12px;
        z-index: 20;
      }
      .skip:focus {
        top: 8px;
      }
      .topbar {
        border-bottom: 1px solid var(--rule);
        background: var(--surface);
        padding: 16px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
      }
      .brand {
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
      }
      .toolbar,
      .auth-controls,
      .filters,
      .actions {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
      }
      .auth-controls input {
        width: 190px;
      }
      body[data-unlocked="true"] .auth-controls label,
      body[data-unlocked="true"] .auth-controls input,
      body[data-unlocked="true"] #keyToggle {
        display: none;
      }
      body[data-unlocked="true"] .auth-controls {
        width: auto;
      }
      .btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 40px;
        padding: 7px 14px;
        border: 1px solid var(--accent);
        border-radius: 6px;
        background: var(--accent);
        color: var(--surface);
        font-weight: 600;
        text-decoration: none;
        font-size: 0.8125rem;
        gap: 6px;
      }
      .btn.secondary {
        background: var(--surface);
        border-color: var(--rule);
        color: var(--text);
      }
      .btn.danger {
        color: var(--bad);
        background: var(--surface);
        border-color: var(--bad);
      }
      .btn:hover {
        filter: brightness(0.94);
      }
      .linklike {
        border: 0;
        background: transparent;
        color: var(--accent);
        padding: 4px 0;
        text-align: left;
        font: inherit;
        text-decoration: underline;
        text-underline-offset: 3px;
        overflow-wrap: anywhere;
      }
      .linklike:hover {
        text-decoration-thickness: 2px;
      }
      input,
      select,
      textarea {
        min-height: 40px;
        max-width: 100%;
        border: 1px solid var(--rule);
        border-radius: 6px;
        padding: 8px 10px;
        background: var(--surface);
        color: var(--text);
        font-size: 0.875rem;
      }
      input::placeholder,
      textarea::placeholder {
        color: var(--muted);
        opacity: 1;
      }
      textarea {
        width: 100%;
        min-height: 90px;
      }
      input[type="checkbox"] {
        min-height: 20px;
        width: 20px;
        height: 20px;
        accent-color: var(--accent);
      }
      .shell {
        max-width: 1440px;
        margin: auto;
        padding: 0 20px 48px;
      }
      .nav {
        display: flex;
        gap: 4px;
        align-items: center;
        padding: 16px 0;
        border-bottom: 1px solid var(--rule);
        overflow-x: auto;
      }
      .nav button {
        border: 0;
        border-radius: 6px;
        background: transparent;
        color: var(--muted);
        padding: 10px 14px;
        white-space: nowrap;
        font-weight: 600;
      }
      .nav button[aria-selected="true"] {
        color: var(--accent);
        background: var(--accent-bg);
      }
      .nav .settings-nav {
        margin-left: auto;
        font-weight: 400;
      }
      .statusbar {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
        padding: 12px 0;
        font-size: 0.8125rem;
        color: var(--muted);
      }
      .statusbar .updated {
        margin-left: auto;
      }
      .statusbar strong {
        color: var(--text);
      }
      .page-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        flex-wrap: wrap;
        margin: 24px 0;
      }
      .page-head p {
        margin-top: 8px;
        color: var(--muted);
      }
      .sec-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
        margin: 24px 0 12px;
      }
      .tab-panel {
        display: none;
      }
      .tab-panel.active {
        display: block;
      }
      .filters {
        margin: 16px 0;
        gap: 12px;
      }
      .filters label {
        font-size: 0.8125rem;
        color: var(--muted);
        display: grid;
        gap: 4px;
      }
      .filters input {
        min-width: 220px;
      }
      .filters select {
        min-width: 130px;
      }
      .scroll {
        overflow: auto;
        border: 1px solid var(--rule);
        border-radius: var(--radius);
        background: var(--surface);
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
      }
      th {
        text-align: left;
        font-size: 0.75rem;
        letter-spacing: 0.03em;
        color: var(--muted);
        font-weight: 600;
        background: var(--surface-2);
        white-space: nowrap;
      }
      th,
      td {
        padding: 12px 16px;
        border-bottom: 1px solid var(--rule);
      }
      tbody tr:last-child td {
        border-bottom: 0;
      }
      td {
        vertical-align: top;
      }
      td .small {
        display: block;
        margin-top: 3px;
      }
      .clickable-row:hover {
        background: var(--surface-2);
      }
      .empty td,
      .empty-state {
        padding: 28px;
        color: var(--muted);
      }
      .empty-state {
        border: 1px dashed var(--rule);
        border-radius: var(--radius);
      }
      .empty-state h2,
      .empty-state h3 {
        color: var(--text);
        margin-bottom: 8px;
      }
      .empty-state .actions {
        margin-top: 16px;
      }
      .pill {
        display: inline-block;
        border-radius: 4px;
        padding: 2px 7px;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1.6;
        background: var(--surface-2);
        color: var(--muted);
      }
      .pill.ok {
        background: var(--ok-bg);
        color: var(--ok);
      }
      .pill.warn {
        background: var(--warn-bg);
        color: var(--warn);
      }
      .pill.bad {
        background: var(--bad-bg);
        color: var(--bad);
      }
      .pill.accent {
        background: var(--accent-bg);
        color: var(--accent);
      }
      .pill.id {
        font-weight: 500;
        background: none;
        padding: 0;
        color: inherit;
      }
      .dot {
        display: none;
      }
      .strip {
        display: flex;
        gap: 24px;
        flex-wrap: wrap;
        padding: 20px 0;
        border-block: 1px solid var(--rule);
      }
      .stat {
        display: grid;
        gap: 4px;
        min-width: 130px;
        flex: 1;
      }
      .stat .label {
        font-size: 0.8125rem;
        color: var(--muted);
      }
      .stat .value {
        font-size: 1.375rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
      }
      .stat .detail {
        font-size: 0.8125rem;
      }
      .overview-grid {
        display: grid;
        gap: 24px;
      }
      .notice,
      .inline-err {
        padding: 12px 16px;
        border: 1px solid var(--rule);
        border-radius: 6px;
        background: var(--surface);
      }
      .inline-err {
        color: var(--bad);
        background: var(--bad-bg);
        border-color: var(--bad);
        margin: 12px 0;
      }
      .notice.warn {
        background: var(--warn-bg);
        color: var(--warn);
      }
      .attention-list {
        list-style: none;
        padding: 0;
        margin: 0;
      }
      .attention-list li {
        padding: 16px 0;
        border-bottom: 1px solid var(--rule);
      }
      .attention-list li:last-child {
        border: 0;
      }
      .attention-list .small {
        margin: 4px 0 8px;
      }
      .skeleton {
        background: var(--surface-2);
        border-radius: 4px;
        min-height: 48px;
        margin: 8px 0;
      }
      .view-status {
        font-size: 0.8125rem;
        color: var(--muted);
        margin: 8px 0;
      }
      details {
        margin: 16px 0;
      }
      summary {
        cursor: pointer;
        font-weight: 600;
        min-height: 44px;
        padding: 10px 0;
      }
      details > p {
        margin: 8px 0;
      }
      .formset {
        border-block: 1px solid var(--rule);
        padding: 8px 0;
      }
      .formgrid,
      .kv-grid {
        display: grid;
        gap: 16px;
      }
      .field {
        display: grid;
        gap: 6px;
        align-content: start;
      }
      .field label {
        font-size: 0.875rem;
        font-weight: 600;
      }
      .field .hint {
        font-size: 0.8125rem;
        color: var(--muted);
      }
      .check {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .msg {
        margin: 12px 0;
        min-height: 24px;
        overflow-wrap: anywhere;
      }
      .formset .toolbar {
        margin: 16px 0;
      }
      .kv-item {
        display: grid;
        gap: 4px;
        min-width: 0;
      }
      .kv-item .v {
        overflow-wrap: anywhere;
      }
      .kv-grid {
        margin: 12px 0;
      }
      .formset > summary {
        color: var(--accent);
      }
      .mgmt-hint {
        display: none;
      }
      body:not([data-writes="true"]) [data-mgmt],
      body:not([data-writes="true"]) .formset {
        display: none !important;
      }
      body:not([data-writes="true"]) .mgmt-hint {
        display: block;
      }
      .preset-grid {
        display: grid;
        gap: 16px;
      }
      .preset-card {
        padding: 12px 0;
        border-bottom: 1px solid var(--rule);
      }
      .pool-grid {
        display: grid;
        gap: 16px;
      }
      .pool-card {
        border: 1px solid var(--rule);
        border-radius: var(--radius);
        padding: 20px;
        background: var(--surface);
      }
      .pool-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
      }
      .route-order {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        list-style: none;
        padding: 0;
        margin: 16px 0;
      }
      .route-order li {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .route-order li:not(:last-child)::after {
        content: "→";
        color: var(--muted);
      }
      .route-order .skipped {
        text-decoration: line-through;
        color: var(--muted);
      }
      .assigned-models {
        display: flex;
        gap: 8px;
        align-items: baseline;
        flex-wrap: wrap;
        font-size: 0.875rem;
      }
      .workspace-row {
        padding: 20px 0;
        border-bottom: 1px solid var(--rule);
      }
      .workspace-row:last-child {
        border: 0;
      }
      .workspace-row h3 {
        margin-bottom: 8px;
      }
      .subhead {
        font-size: 1rem;
        font-weight: 600;
        margin: 24px 0 12px;
      }
      .segmented {
        display: flex;
        gap: 4px;
        flex-wrap: wrap;
      }
      .segmented button {
        border: 1px solid var(--rule);
        border-radius: 6px;
        background: var(--surface);
        color: var(--muted);
        padding: 8px 12px;
        min-height: 40px;
      }
      .segmented button[aria-pressed="true"] {
        color: var(--accent);
        background: var(--accent-bg);
        border-color: var(--accent);
      }
      .detail-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 12px;
        flex-wrap: wrap;
        border-bottom: 1px solid var(--rule);
        padding-bottom: 16px;
        margin-bottom: 20px;
      }
      .drawer .detail-head {
        position: sticky;
        top: 0;
        z-index: 1;
        background: var(--surface);
        padding-top: 12px;
      }
      .detail-head h2 {
        font-size: 1.375rem;
        overflow-wrap: anywhere;
      }
      .detail-head .close {
        background: var(--surface);
        color: var(--text);
        border: 1px solid var(--rule);
        border-radius: 6px;
        padding: 8px 12px;
        min-height: 40px;
      }
      .drawer {
        position: fixed;
        inset: 0 0 0 auto;
        width: min(780px, 100vw);
        height: 100dvh;
        z-index: 30;
        background: var(--surface);
        border-left: 1px solid var(--rule);
        overflow: auto;
        padding: 24px;
      }
      .drawer-scrim {
        position: fixed;
        inset: 0;
        background: oklch(15% 0.008 220 / 0.45);
        z-index: 29;
      }
      .drawer-open {
        overflow: hidden;
      }
      .loading-bar {
        padding: 24px;
        background: var(--surface-2);
        border-radius: 6px;
      }
      .detail-section {
        margin: 24px 0;
      }
      .detail-section > h3 {
        margin-bottom: 12px;
      }
      .toast {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: var(--text);
        color: var(--surface);
        padding: 10px 16px;
        border-radius: 6px;
        z-index: 40;
        max-width: 90vw;
      }
      .locked {
        max-width: 540px;
        margin: 64px auto;
      }
      .locked h1 {
        margin-bottom: 12px;
      }
      @media (min-width: 700px) {
        .shell {
          padding-inline: 32px;
        }
        .formgrid,
        .kv-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .field.wide {
          grid-column: 1/-1;
        }
        .overview-grid {
          grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        }
      }
      @media (max-width: 699px) {
        .topbar {
          padding: 12px 20px;
        }
        .nav {
          gap: 0;
          flex-wrap: wrap;
          overflow: visible;
        }
        .nav button {
          padding: 10px 6px;
          flex: 1;
          font-size: 0.8125rem;
        }
        .nav .settings-nav {
          margin-left: 0;
          flex: 0 0 100%;
          text-align: left;
        }
        .statusbar .updated {
          margin-left: 0;
          width: 100%;
        }
        .btn,
        button,
        input,
        select {
          min-height: 44px;
        }
        .page-head {
          margin-top: 20px;
        }
        h1 {
          font-size: 1.5rem;
        }
        .filters {
          align-items: stretch;
        }
        .filters label {
          flex: 1;
          min-width: 140px;
        }
        .filters input {
          min-width: 0;
          width: 100%;
        }
        .drawer {
          padding: 20px;
        }
        .inventory thead {
          display: none;
        }
        .inventory,
        .inventory tbody,
        .inventory tr,
        .inventory td {
          display: block;
        }
        .inventory tr {
          border-bottom: 1px solid var(--rule);
          padding: 12px 16px;
        }
        .inventory td {
          border: 0;
          padding: 4px 0;
          text-align: left;
        }
        .inventory td[data-label]::before {
          content: attr(data-label) ": ";
          font-size: 0.8125rem;
          color: var(--muted);
        }
        .inventory td:first-child {
          font-weight: 600;
        }
        .inventory .empty td {
          padding: 12px 0;
        }
        .strip {
          gap: 16px;
        }
        .stat {
          min-width: 120px;
        }
        .pool-card {
          padding: 16px;
        }
        .auth-controls {
          width: 100%;
        }
        .auth-controls input {
          flex: 1;
          min-width: 80px;
        }
        .hide-narrow {
          display: none;
        }
      }
      @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
          scroll-behavior: auto !important;
          animation: none !important;
          transition: none !important;
        }
      }
    </style>
  </head>
  <body data-writes="false">
    <a class="skip" href="#main-content">Skip to content</a>
    <div id="appChrome">
      <header class="topbar">
        <div class="brand">Model Gateway</div>
        <div class="auth-controls">
          <label for="adminKey" class="small">Admin key</label
          ><input
            id="adminKey"
            type="password"
            autocomplete="off"
            spellcheck="false"
          /><button
            id="keyToggle"
            class="btn secondary"
            type="button"
            aria-label="Show admin key"
          >
            Show</button
          ><button id="unlockBtn" class="btn" type="button">Unlock</button
          ><button id="lockBtn" class="btn secondary hidden" type="button">
            Lock</button
          ><button
            id="refreshBtn"
            class="btn secondary"
            type="button"
            title="Refresh (R)"
          >
            Refresh
          </button>
        </div>
      </header>
      <main class="shell" id="main-content" tabindex="-1">
        <section id="lockedView" class="locked">
          <h1>One place for your models</h1>
          <p class="muted">
            Unlock to check routing, find a model, and investigate requests.
            Your admin key stays in this browser tab for the session.
          </p>
          <div id="lockedErr" class="inline-err hidden" role="alert"></div>
        </section>
        <div id="dash" class="hidden">
          <nav class="nav" role="tablist" aria-label="Gateway sections">
            <button
              id="tab-overview"
              role="tab"
              data-tab="overview"
              aria-controls="panel-overview"
              aria-selected="true"
              tabindex="0"
            >
              Overview</button
            ><button
              id="tab-models"
              role="tab"
              data-tab="models"
              aria-controls="panel-models"
              aria-selected="false"
              tabindex="-1"
            >
              Models</button
            ><button
              id="tab-connections"
              role="tab"
              data-tab="connections"
              aria-controls="panel-connections"
              aria-selected="false"
              tabindex="-1"
            >
              Connections</button
            ><button
              id="tab-activity"
              role="tab"
              data-tab="activity"
              aria-controls="panel-activity"
              aria-selected="false"
              tabindex="-1"
            >
              Activity</button
            ><button
              id="tab-settings"
              role="tab"
              data-tab="settings"
              aria-controls="panel-settings"
              aria-selected="false"
              tabindex="-1"
              class="settings-nav"
            >
              Settings <span class="hide-narrow">& diagnostics</span>
            </button>
          </nav>
          <div class="statusbar">
            <span id="sService">Checking gateway…</span
            ><span id="sConfig">Configuration not loaded</span
            ><span id="writeMode">Read-only</span
            ><span class="updated" id="healthMeta">Not refreshed yet</span>
          </div>
          <div id="dashErr" class="inline-err hidden" role="alert"></div>
          <section
            id="panel-overview"
            class="tab-panel active"
            data-tab-panel="overview"
            role="tabpanel"
            aria-labelledby="tab-overview"
            tabindex="0"
          >
            <div class="page-head">
              <div>
                <h1>Overview</h1>
                <p>What needs attention, and what happened recently.</p>
              </div>
              <div class="actions">
                <button class="btn secondary" data-go="models">
                  Find a model</button
                ><button class="btn" data-mgmt data-add-provider>
                  Add a connection
                </button>
              </div>
            </div>
            <div id="overviewSetup" class="empty-state hidden">
              <h2>Connect your first provider</h2>
              <p>
                Add a cloud connection or local runtime, check it, then register
                a model. You can inspect the request route before saving.
              </p>
              <div class="actions">
                <button class="btn" data-mgmt data-add-provider>
                  Add a connection</button
                ><button class="btn secondary" data-go="connections">
                  View connections
                </button>
              </div>
              <p class="mgmt-hint">
                This gateway is read-only. Setup instructions are in Settings.
              </p>
            </div>
            <div class="overview-grid">
              <section>
                <div class="sec-head">
                  <h2>Needs attention</h2>
                  <button class="linklike" data-view-failures>
                    View recent failures
                  </button>
                </div>
                <div id="attention">
                  <div
                    class="skeleton"
                    aria-label="Loading routing status"
                  ></div>
                </div>
              </section>
              <section>
                <div class="sec-head"><h2>Available inventory</h2></div>
                <div id="inventorySummary" class="notice">
                  Loading connections and models…
                </div>
                <p class="small" style="margin-top: 12px">
                  Configuration checks do not verify upstream availability.
                  Inspect a connection for recent outcomes or run an explicit
                  check.
                </p>
              </section>
            </div>
            <div class="sec-head">
              <h2>Last 24 hours</h2>
              <button class="linklike" data-go="activity">View activity</button>
            </div>
            <div id="overviewStats" class="strip">
              <span class="muted">Loading usage…</span>
            </div>
            <div
              class="view-status"
              id="overviewUsageStatus"
              role="status"
            ></div>
            <div class="sec-head">
              <h2>Latest requests</h2>
              <span class="meta">Latest 5, independent of usage window</span>
            </div>
            <div class="scroll">
              <table id="overviewRequests" class="inventory">
                <thead>
                  <tr>
                    <th>Request</th>
                    <th>Model</th>
                    <th>Outcome</th>
                    <th class="num">Duration</th>
                  </tr>
                </thead>
                <tbody>
                  <tr class="empty">
                    <td colspan="4">Loading requests…</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
          <section
            id="panel-models"
            class="tab-panel"
            data-tab-panel="models"
            role="tabpanel"
            aria-labelledby="tab-models"
            tabindex="0"
          >
            <div class="page-head">
              <div>
                <h1>Models</h1>
                <p>Find a model, understand its route, and copy a request.</p>
              </div>
              <button class="btn" data-mgmt data-add-model>
                Register a model
              </button>
            </div>
            <div id="modelsStatus" class="view-status" role="status"></div>
            <div class="filters">
              <label for="modelFilter"
                >Search<input
                  id="modelFilter"
                  type="search"
                  placeholder="Name, alias, or upstream ID" /></label
              ><label for="modelLocality"
                >Location<select id="modelLocality">
                  <option value="">All locations</option>
                  <option value="local">Local</option>
                  <option value="cloud">Cloud</option>
                  <option value="mixed">Mixed</option>
                  <option value="composite">Composite</option>
                </select></label
              ><label for="modelProviderFilter"
                >Connection<select id="modelProviderFilter">
                  <option value="">All connections</option>
                </select></label
              ><label for="modelCapability"
                >Capability<select id="modelCapability">
                  <option value="">All capabilities</option>
                  <option value="vision">Vision</option>
                  <option value="tools">Tools (explicitly recorded)</option>
                </select></label
              ><span id="modelsMeta" class="meta"></span>
            </div>
            <div class="scroll">
              <table id="models" class="inventory">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Location</th>
                    <th>Capabilities</th>
                    <th>Connection</th>
                    <th>Availability</th>
                  </tr>
                </thead>
                <tbody>
                  <tr class="empty">
                    <td colspan="5">Loading models…</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <details id="presetSection">
              <summary>Model groups & presets</summary>
              <p class="small">
                Gateway-owned text and vision selections, when configured.
                Individual models are listed above.
              </p>
              <div id="presetsMeta" class="meta"></div>
              <div id="presetSummary" class="preset-grid"></div>
              <div class="scroll">
                <table id="presets">
                  <thead>
                    <tr>
                      <th>Tier</th>
                      <th>Scope</th>
                      <th>Text model</th>
                      <th>Vision model</th>
                      <th>Policy</th>
                      <th>Memory</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody></tbody>
                </table>
              </div>
            </details>
            <details class="formset" id="modelFormset">
              <summary id="modelFormTitle">Register a model</summary>
              <p class="small">
                Changes apply immediately. Preview the route first. Existing
                workspace pool membership is preserved. Change pool membership
                using the transactional workspace CLI.
              </p>
              <div class="formgrid">
                <div class="field">
                  <label for="mName">Client-facing model ID</label
                  ><input id="mName" placeholder="my-model" />
                </div>
                <div class="field">
                  <label for="mProvider">Connection</label
                  ><select id="mProvider"></select>
                </div>
                <div class="field">
                  <label for="mPmid">Upstream model ID</label
                  ><input id="mPmid" placeholder="Provider's model name" />
                </div>
                <div class="field">
                  <label for="mAlias">Alias (optional)</label
                  ><input id="mAlias" />
                </div>
                <div class="field">
                  <label for="mPool">Routing group (read-only)</label
                  ><select id="mPool">
                    <option value="">Direct connection</option></select
                  ><span class="hint"
                    >Pool assignment is managed by the operator CLI. New models
                    registered here use a direct connection.</span
                  >
                </div>
                <div class="field">
                  <label for="mDesc">Description</label><input id="mDesc" />
                </div>
                <div class="field">
                  <label for="mContext">Context limit (tokens)</label
                  ><input id="mContext" type="number" min="1" />
                </div>
                <div class="field">
                  <label for="mMaxOut">Maximum output (tokens)</label
                  ><input id="mMaxOut" type="number" min="1" />
                </div>
                <label class="check"
                  ><input id="mVision" type="checkbox" />Accepts images
                  natively</label
                ><label class="check"
                  ><input id="mEnabled" type="checkbox" checked />Enabled</label
                >
              </div>
              <details>
                <summary>Reasoning & pricing</summary>
                <div class="formgrid">
                  <div class="field">
                    <label for="mThinking">Reasoning mode</label
                    ><input
                      id="mThinking"
                      placeholder="always, optional, or none"
                    />
                  </div>
                  <div class="field">
                    <label for="mThinkingLevels">Reasoning levels</label
                    ><input
                      id="mThinkingLevels"
                      placeholder="low, medium, high"
                    />
                  </div>
                  <div class="field">
                    <label for="mThinkingFmt">Reasoning format</label
                    ><input id="mThinkingFmt" />
                  </div>
                  <div class="field">
                    <label for="mPricingStatus">Pricing status</label
                    ><select id="mPricingStatus">
                      <option value="unknown">Unknown</option>
                      <option value="metered">Metered</option>
                      <option value="unmetered">Unmetered</option>
                    </select>
                  </div>
                  <div class="field wide">
                    <label for="mPricing"
                      >Pricing JSON (USD per million tokens)</label
                    ><textarea
                      id="mPricing"
                      placeholder='{"input": 1.0, "output": 2.0}'
                    ></textarea>
                  </div>
                </div>
              </details>
              <div class="toolbar">
                <button class="btn secondary" id="previewModelBtn">
                  Preview route</button
                ><button class="btn" id="saveModelBtn">Save model</button
                ><button class="btn secondary" data-cancel-model>Cancel</button
                ><button class="btn danger" id="deleteModelBtn">
                  Delete model
                </button>
              </div>
              <div id="mMsg" class="msg" role="status"></div>
              <div id="previewPanel" class="hidden"></div>
            </details>
            <p class="mgmt-hint small">
              Read-only access. To register or edit models, see Settings.
            </p>
          </section>
          <section
            id="panel-connections"
            class="tab-panel"
            data-tab-panel="connections"
            role="tabpanel"
            aria-labelledby="tab-connections"
            tabindex="0"
          >
            <div class="page-head">
              <div>
                <h1>Connections</h1>
                <p>
                  Cloud providers, local runtimes, and the models they serve.
                </p>
              </div>
              <button class="btn" data-mgmt data-add-provider>
                Add a connection
              </button>
            </div>
            <div id="providersStatus" class="view-status" role="status"></div>
            <div class="filters">
              <label for="providerFilter"
                >Search<input
                  id="providerFilter"
                  type="search"
                  placeholder="Connection or host" /></label
              ><span class="meta"
                >Configuration readiness is not an upstream health check.</span
              >
            </div>
            <div class="scroll">
              <table id="providers" class="inventory">
                <thead>
                  <tr>
                    <th>Connection</th>
                    <th>Models</th>
                    <th>Configuration</th>
                    <th>Recent outcomes</th>
                    <th>Last check</th>
                  </tr>
                </thead>
                <tbody>
                  <tr class="empty">
                    <td colspan="5">Loading connections…</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <details id="routingGroups">
              <summary>Routing groups (workspace pools)</summary>
              <p>
                Try these workspaces in order for the same model. Pools provide
                ordered failover, not load balancing.
              </p>
              <p class="small">
                Preference is based on local configuration and circuit state.
                Upstream model availability is not verified by this view.
              </p>
              <div id="poolsStatus" class="view-status" role="status"></div>
              <div id="poolCards" class="pool-grid"></div>
              <div class="sec-head">
                <h2>Workspaces</h2>
                <span class="meta"
                  >Each member shown once across all groups</span
                >
              </div>
              <div id="workspaceList"></div>
              <details id="workspaceRecovery">
                <summary>Databricks recovery instructions</summary>
                <p>
                  The repair command checks all OAuth-backed workspaces, not
                  just one routing group. It may prompt for login or
                  replacement, test models, and activate configuration changes
                  with rollback verification.
                </p>
                <pre>model-gateway workspace repair</pre>
                <button
                  class="btn secondary"
                  data-copy-command="model-gateway workspace repair"
                >
                  Copy repair-all command
                </button>
                <p class="small">
                  Copied commands run in your terminal, never automatically in
                  this dashboard. Workspace tests authenticate and may make real
                  inference requests with usage charges.
                </p>
              </details>
            </details>
            <details class="formset" id="providerFormset">
              <summary id="providerFormTitle">Add a connection</summary>
              <p class="small">
                Save the connection, check access, discover its models, then
                choose one to register. Saving applies immediately; API keys are
                write-only.
              </p>
              <div class="formgrid">
                <div class="field">
                  <label for="pId">Connection ID</label
                  ><input id="pId" placeholder="my-provider" />
                </div>
                <div class="field">
                  <label for="pProtocol">Protocol</label
                  ><select id="pProtocol">
                    <option value="openai">OpenAI-compatible</option>
                    <option value="anthropic">Anthropic-compatible</option>
                  </select>
                </div>
                <div class="field wide">
                  <label for="pBaseUrl">Base URL</label
                  ><input
                    id="pBaseUrl"
                    type="url"
                    placeholder="https://api.example.com/v1"
                  />
                </div>
                <div class="field wide">
                  <label for="pApiKey">API key</label
                  ><input
                    id="pApiKey"
                    type="password"
                    autocomplete="new-password"
                  /><span class="hint"
                    >Leave blank when editing to keep the saved key. Existing
                    keys are never shown.</span
                  >
                </div>
              </div>
              <div class="toolbar">
                <button class="btn" id="saveProviderBtn">Save connection</button
                ><button class="btn secondary" id="validateProviderBtn">
                  Check saved connection</button
                ><button class="btn secondary" id="discoverBtn">
                  Discover models</button
                ><button class="btn secondary" data-cancel-provider>
                  Cancel</button
                ><button class="btn danger" id="deleteProviderBtn">
                  Delete connection
                </button>
              </div>
              <p class="small">
                Check and Discover use the saved connection, not unsaved form
                changes. The check reads the upstream model inventory; it does
                not test inference.
              </p>
              <div id="pMsg" class="msg" role="status"></div>
              <div id="discoverPanel" class="hidden">
                <h3>Discovered models</h3>
                <p id="discoverMeta" class="meta"></p>
                <div class="scroll">
                  <table id="discoverTable">
                    <thead>
                      <tr>
                        <th>Upstream ID</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody></tbody>
                  </table>
                </div>
              </div>
            </details>
            <p class="mgmt-hint small">
              Read-only access. To add, edit, check, or discover a connection,
              see Settings. Workspace CLI commands are available as instructions
              only.
            </p>
          </section>
          <section
            id="panel-activity"
            class="tab-panel"
            data-tab-panel="activity"
            role="tabpanel"
            aria-labelledby="tab-activity"
            tabindex="0"
          >
            <div class="page-head">
              <div>
                <h1>Activity</h1>
                <p>Trace requests and understand usage and cost.</p>
              </div>
            </div>
            <div class="segmented" aria-label="Activity view">
              <button data-activity="requests" aria-pressed="true">
                Requests</button
              ><button data-activity="usage" aria-pressed="false">
                Usage & cost
              </button>
            </div>
            <div id="usageErr" class="inline-err hidden" role="alert"></div>
            <div id="requestsView">
              <div class="sec-head">
                <h2>Recent requests</h2>
                <span class="meta">Latest 50 across all time</span>
              </div>
              <div class="filters">
                <label for="requestFilter"
                  >Search latest 50<input
                    id="requestFilter"
                    type="search"
                    placeholder="Model, connection, or status" /></label
                ><label for="requestOutcome"
                  >Outcome<select id="requestOutcome">
                    <option value="">All outcomes</option>
                    <option value="failed">Failures</option>
                    <option value="success">Successful</option>
                  </select></label
                >
              </div>
              <div id="requestsStatus" class="view-status" role="status"></div>
              <div class="scroll">
                <table id="recentReq" class="inventory">
                  <thead>
                    <tr>
                      <th>Request</th>
                      <th>Requested model</th>
                      <th>Recorded connection</th>
                      <th>Outcome</th>
                      <th class="num">Duration</th>
                      <th class="num">Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr class="empty">
                      <td colspan="6">Loading requests…</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div id="usageView" hidden>
              <div class="sec-head">
                <h2>Usage & cost</h2>
                <div class="segmented" id="winSeg" aria-label="Usage window">
                  <button data-w="1h" aria-pressed="false">1h</button
                  ><button data-w="24h" aria-pressed="true">24h</button
                  ><button data-w="7d" aria-pressed="false">7d</button
                  ><button data-w="30d" aria-pressed="false">30d</button
                  ><button data-w="" aria-pressed="false">All time</button>
                </div>
              </div>
              <p id="usageRange" class="meta">Last 24 hours</p>
              <div class="strip">
                <div class="stat">
                  <span class="label">Requests</span
                  ><span class="value" id="uRequests">Not loaded</span
                  ><span class="detail" id="uRequestsDetail"></span>
                </div>
                <div class="stat">
                  <span class="label">Tokens</span
                  ><span class="value" id="uTokens">Not loaded</span
                  ><span class="detail" id="uTokensDetail"></span>
                </div>
                <div class="stat">
                  <span class="label">Known cost</span
                  ><span class="value" id="uCost">Not loaded</span
                  ><span class="detail" id="uCostDetail"></span>
                </div>
                <div class="stat">
                  <span class="label">Average duration</span
                  ><span class="value" id="uLatency">Not loaded</span
                  ><span class="detail" id="uLatencyDetail"></span>
                </div>
              </div>
              <div class="sec-head"><h2>By model route</h2></div>
              <div class="scroll">
                <table id="usageByModel">
                  <thead>
                    <tr>
                      <th>Model / actual route</th>
                      <th class="num">Requests</th>
                      <th class="num">Errors</th>
                      <th class="num">Known cost</th>
                      <th class="num">Duration</th>
                    </tr>
                  </thead>
                  <tbody></tbody>
                </table>
              </div>
              <details>
                <summary>Token accounting by route</summary>
                <div class="scroll">
                  <table id="tokenAccounting">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th class="num">Input</th>
                        <th class="num">Output</th>
                        <th class="num">Cached read</th>
                        <th class="num">Usage reported</th>
                        <th class="num">Cost complete</th>
                      </tr>
                    </thead>
                    <tbody></tbody>
                  </table>
                </div>
              </details>
              <div class="sec-head"><h2>By connection</h2></div>
              <div class="scroll">
                <table id="usageByProvider">
                  <thead>
                    <tr>
                      <th>Connection</th>
                      <th class="num">Requests</th>
                      <th class="num">Errors</th>
                      <th class="num">Known cost</th>
                    </tr>
                  </thead>
                  <tbody></tbody>
                </table>
              </div>
            </div>
          </section>
          <section
            id="panel-settings"
            class="tab-panel"
            data-tab-panel="settings"
            role="tabpanel"
            aria-labelledby="tab-settings"
            tabindex="0"
          >
            <div class="page-head">
              <div>
                <h1>Settings & diagnostics</h1>
                <p>Access controls and technical details for this gateway.</p>
              </div>
            </div>
            <section class="detail-section">
              <h2>Management access</h2>
              <p id="managementState" style="margin-top: 12px"></p>
              <p class="small">
                The operator controls write access with
                <code>MODEL_GATEWAY_ADMIN_WRITES=true</code> in the gateway
                service environment. This browser cannot enable it. Writes
                update the machine-local configuration immediately.
              </p>
            </section>
            <section class="detail-section">
              <h2>Runtime</h2>
              <div id="runtimeDetails" class="kv-grid"></div>
            </section>
            <details>
              <summary>Protocol diagnostics</summary>
              <p>
                <a
                  id="thinkingLink"
                  href="/v1/debug/thinking"
                  target="_blank"
                  rel="noopener"
                  >Open reasoning diagnostics</a
                >
              </p>
              <p class="small">
                Client authentication is separate. If this link returns
                Unauthorized, use an authenticated HTTP client. No key is
                included in this URL.
              </p>
            </details>
          </section>
        </div>
      </main>
    </div>
    <div id="drawerScrim" class="drawer-scrim" hidden></div>
    <aside
      id="drawer"
      class="drawer"
      role="dialog"
      aria-modal="true"
      aria-label="Details"
      tabindex="-1"
      hidden
    >
      <div id="drawerBody" aria-live="polite"></div>
    </aside>
    <div id="toast" class="toast hidden" role="status"></div>
    <script>
      (() => {
        "use strict";
        const $ = (id) => document.getElementById(id);
        const STORAGE_KEY = "mg-admin-key";
        const BASE_PATH = (() => {
          const i = location.pathname.indexOf("/admin");
          return i > 0 ? location.pathname.slice(0, i).replace(/\/$/, "") : "";
        })();
        const api = (path) => BASE_PATH + path;
        const esc = (value) =>
          String(value ?? "").replace(
            /[&<>"']/g,
            (c) =>
              ({
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#39;",
              })[c],
          );
        const text = (value) =>
          value === null || value === undefined || value === ""
            ? "Not recorded"
            : String(value);
        const num = (value) =>
          value == null ? "Not recorded" : Number(value).toLocaleString();
        const cost = (value) =>
          value == null ? "Unknown" : "$" + Number(value).toFixed(4);
        function duration(value) {
          if (value == null) return "Not recorded";
          const s = Number(value) / 1000;
          return s < 1
            ? Math.round(value) + " ms"
            : s < 60
              ? s.toFixed(1) + " s"
              : Math.floor(s / 60) + "m " + Math.round(s % 60) + "s";
        }
        const when = (r) =>
          r.ts ? new Date(r.ts * 1000).toLocaleString() : text(r.ts_iso);
        const stamp = () => new Date().toLocaleTimeString();
        const pill = (kind, label) =>
          '<span class="pill ' + kind + '">' + esc(label) + "</span>";
        const button = (attr, id, label, cls = "linklike") =>
          '<button type="button" class="' +
          cls +
          '" ' +
          attr +
          '="' +
          esc(id) +
          '">' +
          esc(label) +
          "</button>";
        const kv = (label, value) =>
          '<div class="kv-item"><span class="k">' +
          esc(label) +
          '</span><span class="v">' +
          value +
          "</span></div>";
        const windows = {
          "1h": "Last hour",
          "24h": "Last 24 hours",
          "7d": "Last 7 days",
          "30d": "Last 30 days",
          "": "All time",
        };
        const tabs = [
          "overview",
          "models",
          "connections",
          "activity",
          "settings",
        ];
        const aliases = {
          providers: "connections",
          pools: "connections",
          presets: "models",
          usage: "activity",
          debug: "settings",
        };
        const cache = {},
          failures = {},
          fresh = {},
          loadedGeneration = {},
          checks = new Map();
        let unlocked = false,
          generation = 0,
          sessionEpoch = 0,
          mutationBusy = false,
          refreshing = false,
          currentTab = "overview",
          currentWindow = "24h",
          usageSeq = 0,
          detailSeq = 0,
          detailKind = null,
          detailName = null,
          returnFocus = null,
          toastTimer;
        let editingProvider = null,
          editingModel = null;
        const modelName = (m) => m.name || m.id;
        const models = () => {
          const seen = new Set();
          return (cache.models?.models || []).filter((m) => {
            const n = modelName(m);
            if (!n || seen.has(n)) return false;
            seen.add(n);
            return true;
          });
        };
        const providers = () => cache.providers?.providers || [];
        const pools = () => cache.pools?.pools || [];
        const memberRows = () => pools().flatMap((p) => p.members || []);
        const findModel = (name) =>
          models().find((m) =>
            [
              modelName(m),
              m.alias,
              m.provider_model_id,
              m.omlx_id,
              ...(m.routable_ids || []),
            ].includes(name),
          );
        const modelLink = (name) =>
          findModel(name)
            ? button("data-open-model", modelName(findModel(name)), name)
            : esc(text(name));
        const providerLink = (id) =>
          providers().some((p) => p.id === id)
            ? button("data-open-provider", id, id)
            : esc(text(id));
        const modelPools = (m) =>
          pools().filter((p) => (p.models || []).includes(modelName(m)));
        function locality(m) {
          if (m.composite) return "composite";
          if (m.locality) return m.locality;
          const ids = m.declared_providers || [
            m.configured_provider || m.provider,
          ];
          const local = ids.some((id) => id === "omlx");
          return local
            ? ids.some((id) => id !== "omlx")
              ? "mixed"
              : "local"
            : "cloud";
        }
        function modelAvailability(m) {
          return m.enabled === false
            ? pill("muted", "Disabled")
            : m.available
              ? pill("ok", "Configured")
              : pill("warn", "Needs attention");
        }
        function issuesText(issues) {
          const names = {
            circuit_open: "temporarily skipped after failures",
            missing_api_key: "API key missing",
            missing_base_url: "base URL missing",
            provider_disabled: "disabled",
            disabled: "disabled",
          };
          return (issues || [])
            .map((i) => names[i] || i.replace(/_/g, " "))
            .join(", ");
        }
        function associatedModels(p) {
          return models().filter(
            (m) =>
              m.enabled !== false &&
              ((
                m.declared_providers || [m.configured_provider || m.provider]
              ).includes(p.id) ||
                modelPools(m).some((g) =>
                  g.members.some((w) => w.id === p.id),
                )),
          );
        }
        function recentFor(id) {
          return (
            memberRows().find((p) => p.id === id)?.recent ||
            (cache.pools?.standalone_providers || []).find((p) => p.id === id)
              ?.recent
          );
        }
        function configReady(p) {
          return p.ready && !!p.base_url && p.has_api_key;
        }
        function outcome(r) {
          return r.error || (r.status != null && r.status >= 400)
            ? "failed"
            : r.status >= 200 && r.status < 300
              ? "success"
              : "unknown";
        }
        const outcomePill = (r) =>
          pill(
            outcome(r) === "failed"
              ? "bad"
              : outcome(r) === "success"
                ? "ok"
                : "muted",
            outcome(r) === "failed"
              ? "Failed" + (r.status ? " · " + r.status : "")
              : outcome(r) === "success"
                ? "Success"
                : "Not recorded",
          );
        function toast(message) {
          clearTimeout(toastTimer);
          $("toast").textContent = message;
          $("toast").classList.remove("hidden");
          toastTimer = setTimeout(
            () => $("toast").classList.add("hidden"),
            3500,
          );
        }
        async function copy(value) {
          try {
            if (navigator.clipboard?.writeText)
              await navigator.clipboard.writeText(value);
            else {
              const t = document.createElement("textarea");
              t.value = value;
              t.style.position = "fixed";
              t.style.left = "-10000px";
              document.body.append(t);
              t.select();
              try {
                if (!document.execCommand("copy"))
                  throw new Error("Copy unavailable");
              } finally {
                t.remove();
              }
            }
            toast("Copied to clipboard");
          } catch {
            toast("Copy unavailable. Select and copy the displayed text.");
          }
        }
        class AuthError extends Error {}
        async function request(path, method = "GET", body) {
          const epoch = sessionEpoch;
          const headers = {};
          const key = $("adminKey").value.trim();
          if (key) headers.Authorization = "Bearer " + key;
          if (body !== undefined) headers["Content-Type"] = "application/json";
          const response = await fetch(api(path), {
            method,
            headers,
            ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
          });
          if (response.status === 401)
            throw new AuthError(
              "Your admin key was rejected. Unlock with a valid key.",
            );
          if (method !== "GET" && epoch !== sessionEpoch)
            throw new Error(
              "The admin session changed while the operation was in progress. Refresh to inspect the result.",
            );
          const raw = await response.text();
          let data;
          try {
            data = JSON.parse(raw);
          } catch {
            data = {};
          }
          if (!response.ok)
            throw new Error(
              data.error?.message ||
                (typeof data.detail === "string" ? data.detail : "") ||
                "Request failed (HTTP " + response.status + "). Try again.",
            );
          return data;
        }
        function setStatus(id, message, bad = false) {
          if (!id) return;
          $(id).textContent = message;
          $(id).classList.toggle("warn-c", bad);
        }
        function updateErrors() {
          const entries = Object.entries(failures);
          $("dashErr").textContent = entries
            .map(([name, msg]) => name + ": " + msg)
            .join(" ");
          $("dashErr").classList.toggle("hidden", !entries.length);
        }
        function renderAll() {
          renderHealth();
          renderOverview();
          renderModels();
          renderProviders();
          renderPools();
          renderPresets();
          renderRequests();
        }
        async function loadResource(key, path, label, statusId, gen) {
          setStatus(statusId, "Refreshing " + label.toLowerCase() + "…");
          try {
            const data = await request(path);
            if (gen !== generation || !unlocked) return;
            cache[key] = data;
            loadedGeneration[key] = gen;
            fresh[key] = stamp();
            delete failures[label];
            setStatus(statusId, "Updated " + fresh[key]);
            renderAll();
          } catch (e) {
            if (gen !== generation) return;
            if (e instanceof AuthError) {
              lock(e.message);
              return;
            }
            failures[label] =
              (cache[key]
                ? "Showing stale data from " + fresh[key] + ". "
                : "Data unavailable. ") + e.message;
            setStatus(statusId, failures[label], true);
            renderAll();
          } finally {
            if (gen === generation) updateErrors();
          }
        }
        async function refresh() {
          if (!unlocked || refreshing) return;
          refreshing = true;
          const gen = ++generation;
          $("refreshBtn").disabled = true;
          document.body.dataset.writes = "false";
          try {
            await Promise.all([
              loadResource("status", "/admin/api/status", "Gateway", null, gen),
              loadResource(
                "providers",
                "/admin/api/providers",
                "Connections",
                "providersStatus",
                gen,
              ),
              loadResource(
                "models",
                "/admin/api/models",
                "Models",
                "modelsStatus",
                gen,
              ),
              loadResource(
                "pools",
                "/admin/api/workspace-pools",
                "Routing groups",
                "poolsStatus",
                gen,
              ),
              loadResource(
                "presets",
                "/admin/api/presets",
                "Model groups",
                null,
                gen,
              ),
              loadResource(
                "validation",
                "/admin/api/config/validation",
                "Configuration",
                null,
                gen,
              ),
              loadResource(
                "requests",
                "/admin/api/requests?limit=50",
                "Requests",
                "requestsStatus",
                gen,
              ),
              loadResource(
                "overviewUsage",
                "/admin/api/usage?window=24h",
                "Overview usage",
                "overviewUsageStatus",
                gen,
              ),
              loadUsage(currentWindow, gen),
            ]);
          } finally {
            refreshing = false;
            $("refreshBtn").disabled = false;
            if (gen === generation) {
              renderAll();
              applyHash();
            }
          }
        }
        function lock(message = "") {
          ++sessionEpoch;
          ++generation;
          ++detailSeq;
          ++usageSeq;
          unlocked = false;
          document.body.dataset.unlocked = "false";
          document.body.dataset.writes = "false";
          closeDrawer({ skipHash: true });
          for (const key of Object.keys(cache)) delete cache[key];
          for (const key of Object.keys(failures)) delete failures[key];
          for (const key of Object.keys(fresh)) delete fresh[key];
          checks.clear();
          resetProvider();
          resetModel();
          $("dash").classList.add("hidden");
          $("lockedView").classList.remove("hidden");
          $("lockBtn").classList.add("hidden");
          $("unlockBtn").classList.remove("hidden");
          $("adminKey").value = "";
          $("adminKey").type = "password";
          $("keyToggle").textContent = "Show";
          $("keyToggle").setAttribute("aria-label", "Show admin key");
          $("lockedErr").textContent = message;
          $("lockedErr").classList.toggle("hidden", !message);
          try {
            sessionStorage.removeItem(STORAGE_KEY);
          } catch {}
          $("adminKey").focus();
        }
        async function unlock() {
          if (refreshing) return;
          unlocked = true;
          document.body.dataset.unlocked = "true";
          $("lockedView").classList.add("hidden");
          $("dash").classList.remove("hidden");
          $("unlockBtn").classList.add("hidden");
          $("lockBtn").classList.remove("hidden");
          renderAll();
          await refresh();
          if (unlocked && $("drawer").hidden) $("tab-" + currentTab).focus();
          if (unlocked)
            try {
              sessionStorage.setItem(STORAGE_KEY, $("adminKey").value.trim());
            } catch {}
        }
        function setHash(tab, detail) {
          const name = aliases[tab] || tab;
          const hash =
            "#" + name + (detail ? "/" + encodeURIComponent(detail) : "");
          if (location.hash !== hash) history.pushState(null, "", hash);
        }
        function showTab(tab, opts = {}) {
          const mapped = aliases[tab] || tab;
          currentTab = tabs.includes(mapped) ? mapped : "overview";
          for (const b of document.querySelectorAll("[data-tab]")) {
            const selected = b.dataset.tab === currentTab;
            b.setAttribute("aria-selected", String(selected));
            b.tabIndex = selected ? 0 : -1;
          }
          for (const p of document.querySelectorAll("[data-tab-panel]"))
            p.classList.toggle("active", p.dataset.tabPanel === currentTab);
          if (tab === "pools") $("routingGroups").open = true;
          if (tab === "presets") $("presetSection").open = true;
          if (tab === "usage") showActivity("usage");
          if (!opts.skipHash) setHash(currentTab);
        }
        function applyHash() {
          const [tab, raw] = location.hash.replace(/^#/, "").split("/");
          if (tab === "main-content") return;
          showTab(tab || "overview", { skipHash: true });
          if (!unlocked) return;
          let name = "";
          try {
            name = decodeURIComponent(raw || "");
          } catch {
            toast("This detail link is invalid.");
          }
          if (name && tab === "models")
            showModelDetail(name, { skipHash: true });
          else if (name && ["providers", "connections"].includes(tab))
            showProviderDetail(name, { skipHash: true });
          else if (name && ["usage", "activity"].includes(tab))
            showRequestDetail(name, { skipHash: true });
          else closeDrawer({ skipHash: true });
        }
        function showActivity(view) {
          $("requestsView").hidden = view !== "requests";
          $("usageView").hidden = view !== "usage";
          for (const b of document.querySelectorAll("[data-activity]"))
            b.setAttribute("aria-pressed", String(b.dataset.activity === view));
        }
        function renderHealth() {
          const s = cache.status,
            v = cache.validation;
          $("sService").innerHTML = s
            ? pill(
                failures.Gateway ? "warn" : "ok",
                failures.Gateway
                  ? "Gateway status stale"
                  : "Gateway responding",
              )
            : "Checking gateway…";
          $("sConfig").textContent = v
            ? failures.Configuration
              ? "Configuration status stale"
              : v.ok
                ? "Configuration complete"
                : "Configuration needs attention"
            : "Configuration not loaded";
          const writable =
            !!s?.writes_enabled &&
            loadedGeneration.status === generation &&
            !failures.Gateway;
          document.body.dataset.writes = String(writable);
          $("writeMode").textContent = writable
            ? "Management enabled"
            : "Read-only";
          $("healthMeta").textContent = fresh.status
            ? "Gateway checked " + fresh.status
            : "Not refreshed yet";
          $("managementState").textContent = writable
            ? "Management is enabled. Saving a connection or model applies changes immediately."
            : "Management is read-only. Browsing does not change configuration or probe providers.";
          $("runtimeDetails").innerHTML = s
            ? kv("Uptime", duration(s.uptime_seconds * 1000)) +
              kv("Process ID", esc(s.pid)) +
              kv(
                "Admin authentication",
                s.auth?.admin_auth_enabled ? "Enabled" : "Not enabled",
              ) +
              kv(
                "Client authentication",
                s.auth?.client_auth_enabled ? "Enabled" : "Not enabled",
              ) +
              kv(
                "Configuration path",
                "<code>" + esc(s.config_path) + "</code>",
              ) +
              kv(
                "Model catalog path",
                "<code>" + esc(s.model_info_path) + "</code>",
              )
            : '<p class="muted">Runtime details not loaded.</p>';
        }
        function renderOverview() {
          const ps = providers(),
            ms = models();
          const items = [];
          for (const p of ps.filter(
            (p) => associatedModels(p).length && !configReady(p),
          ))
            items.push(
              "<li><strong>" +
                esc(p.id) +
                ' needs configuration</strong><p class="small">' +
                esc(issuesText(p.issues) || "URL or API key missing") +
                ". " +
                associatedModels(p).length +
                " assigned model(s).</p>" +
                button("data-open-provider", p.id, "Inspect connection") +
                "</li>",
            );
          for (const p of pools().filter((p) => p.state !== "healthy"))
            items.push(
              "<li><strong>" +
                esc(p.id) +
                ": " +
                (p.active_member
                  ? "backup capacity reduced"
                  : "no immediately ready workspace") +
                '</strong><p class="small">' +
                (p.active_member
                  ? "Next preference: " + esc(p.active_member) + ". "
                  : "Recovery or probing may still run. ") +
                (p.models || []).length +
                " assigned model(s).</p>" +
                button("data-show-pool", p.id, "Inspect routing group") +
                "</li>",
            );
          for (const m of ms.filter((m) => m.enabled !== false && !m.available))
            items.push(
              "<li><strong>" +
                esc(modelName(m)) +
                ' is not ready</strong><p class="small">' +
                esc(
                  m.availability_message ||
                    "Inspect the model configuration and route.",
                ) +
                "</p>" +
                button("data-open-model", modelName(m), "Inspect model") +
                "</li>",
            );
          const ready = cache.providers && cache.models && cache.pools;
          $("attention").innerHTML = items.length
            ? '<ul class="attention-list">' + items.join("") + "</ul>"
            : ready
              ? '<p class="notice">No configuration or routing-readiness issues detected.</p>'
              : '<p class="muted">Waiting for configuration and routing data.</p>';
          $("overviewSetup").classList.toggle(
            "hidden",
            !cache.models || !cache.providers || ms.some((m) => m.available),
          );
          $('overviewSetup').querySelector('h2').textContent = ps.length ? 'No models are ready' : 'Connect your first provider';
          $('overviewSetup').querySelector('p').textContent = ps.length ? 'Inspect an existing connection, then register or enable a model. Configuration changes are available only with management enabled.' : 'Add a cloud connection or local runtime, check it, then register a model. You can inspect the request route before saving.';
          $("inventorySummary").innerHTML =
            cache.models && cache.providers
              ? "<strong>" +
                ms.filter((m) => m.enabled !== false).length +
                " enabled models</strong> across " +
                ps.length +
                " connections.<br>" +
                button("data-go", "models", "Browse models") +
                " · " +
                button("data-go", "connections", "Inspect connections")
              : "Loading connections and models…";
          const u = cache.overviewUsage?.summary;
          $("overviewStats").innerHTML = u
            ? stats(u)
            : '<span class="muted">Usage not loaded.</span>';
          $("overviewRequests").querySelector("tbody").innerHTML =
            (cache.requests?.requests || [])
              .slice(0, 5)
              .map(
                (r) =>
                  "<tr><td>" +
                  button("data-open-request", r.id, when(r)) +
                  '</td><td data-label="Model">' +
                  modelLink(r.model) +
                  '</td><td data-label="Outcome">' +
                  outcomePill(r) +
                  '</td><td class="num" data-label="Duration">' +
                  duration(r.latency_ms) +
                  "</td></tr>",
              )
              .join("") ||
            '<tr class="empty"><td colspan="4">' +
              (cache.requests
                ? "No requests recorded yet. Send a request using an example from Models."
                : "Requests not loaded.") +
              "</td></tr>";
        }
        function stats(u) {
          return [
            ["Requests", num(u.requests), num(u.errors || 0) + " errors"],
            [
              "Known cost",
              cost(u.cost_usd),
              num(u.known_cost_requests || 0) +
                "/" +
                num(u.requests || 0) +
                " with a known cost",
            ],
            [
              "Average duration",
              duration(u.avg_latency_ms),
              "Recorded request latency",
            ],
          ]
            .map(
              ([label, value, detail]) =>
                '<div class="stat"><span class="label">' +
                label +
                '</span><span class="value">' +
                value +
                '</span><span class="detail">' +
                detail +
                "</span></div>",
            )
            .join("");
        }
        function fillSelect(id, values, emptyLabel) {
          const el = $(id),
            selected = el.value;
          el.innerHTML =
            (emptyLabel !== null
              ? '<option value="">' + esc(emptyLabel) + "</option>"
              : "") +
            values
              .map(
                (v) => '<option value="' + esc(v) + '">' + esc(v) + "</option>",
              )
              .join("");
          if (values.includes(selected)) el.value = selected;
        }
        function renderModels() {
          const q = $("modelFilter").value.trim().toLowerCase(),
            loc = $("modelLocality").value,
            provider = $("modelProviderFilter").value,
            cap = $("modelCapability").value;
          const all = models();
          const rows = all.filter(
            (m) =>
              (!q ||
                [
                  modelName(m),
                  m.alias,
                  m.provider_model_id,
                  m.omlx_id,
                  ...(m.routable_ids || []),
                ]
                  .join(" ")
                  .toLowerCase()
                  .includes(q)) &&
              (!loc || locality(m) === loc) &&
              (!provider ||
                (m.declared_providers || [m.provider]).includes(provider)) &&
              (!cap || (cap === "vision" ? m.vision : m.tools === true)),
          );
          $("models").querySelector("tbody").innerHTML =
            rows
              .map((m) => {
                const groups = modelPools(m);
                return (
                  '<tr class="clickable-row"><td>' +
                  button("data-open-model", modelName(m), modelName(m)) +
                  '<span class="small">' +
                  (m.alias ? "Alias: " + esc(m.alias) + " · " : "") +
                  (m.context
                    ? esc(
                        new Intl.NumberFormat(undefined, {
                          notation: "compact",
                          maximumFractionDigits: 1,
                        }).format(m.context),
                      ) + " context"
                    : "Context not recorded") +
                  '</span></td><td data-label="Location">' +
                  esc(locality(m)) +
                  '</td><td data-label="Capabilities">' +
                  [
                    m.vision ? "Vision" : "Text",
                    m.tools === true ? "Tools" : null,
                    m.thinking ? "Reasoning" : null,
                  ]
                    .filter(Boolean)
                    .join(" · ") +
                  '</td><td data-label="Connection">' +
                  (groups.length
                    ? groups
                        .map((g) => button("data-show-pool", g.id, g.id))
                        .join(" ")
                    : providerLink(m.provider)) +
                  '</td><td data-label="Availability">' +
                  modelAvailability(m) +
                  "</td></tr>"
                );
              })
              .join("") ||
            '<tr class="empty"><td colspan="5">' +
              (!cache.models
                ? "Models not loaded."
                : all.length
                  ? "No models match these filters. Clear search or choose All."
                  : "No models registered. Add a connection, discover its models, then register one.") +
              "</td></tr>";
          $("modelsMeta").textContent =
            rows.length + " of " + all.length + " models";
          const ids = providers().map((p) => p.id);
          fillSelect("modelProviderFilter", ids, "All connections"); // Keep open form values intact during refresh.
          if (!$("modelFormset").open) {
            fillSelect("mProvider", ids, "Choose a connection");
            fillSelect(
              "mPool",
              pools().map((p) => p.id),
              "Direct connection",
            );
          }
        }
        function renderProviders() {
          const q = $("providerFilter").value.trim().toLowerCase();
          const rows = providers().filter((p) =>
            (p.id + " " + p.base_url).toLowerCase().includes(q),
          );
          $("providers").querySelector("tbody").innerHTML =
            rows
              .map((p) => {
                const n = associatedModels(p).length,
                  recent = recentFor(p.id),
                  check = checks.get(p.id);
                return (
                  '<tr class="clickable-row"><td>' +
                  button("data-open-provider", p.id, p.id) +
                  '<span class="small id">' +
                  esc(p.base_url || "URL not configured") +
                  '</span></td><td data-label="Models">' +
                  (n ? num(n) : pill("muted", "Unused")) +
                  '</td><td data-label="Configuration">' +
                  pill(
                    configReady(p) ? "ok" : "warn",
                    configReady(p) ? "Complete" : "Needs configuration",
                  ) +
                  '</td><td data-label="Recent outcomes">' +
                  (recent?.requests
                    ? num(recent.failures) +
                      " failures / " +
                      num(recent.requests) +
                      ' recorded<span class="small">Latest 100, all time</span>'
                    : recent
                      ? "No recorded requests"
                      : "Recent outcomes unavailable") +
                  '</td><td data-label="Last check">' +
                  (check
                    ? esc(check.label) +
                      '<span class="small">' +
                      esc(check.at) +
                      " · this session</span>"
                    : "Not recorded") +
                  "</td></tr>"
                );
              })
              .join("") ||
            '<tr class="empty"><td colspan="5">' +
              (!cache.providers
                ? "Connections not loaded."
                : providers().length
                  ? "No connections match your search."
                  : "No connections configured. Add a cloud provider or local runtime.") +
              "</td></tr>";
        }
        function poolMarkup(p) {
          const members = p.members || [],
            first = members[0],
            active = members.find((m) => m.id === p.active_member);
          const label = !active
            ? "No immediately ready workspace"
            : active.id === first?.id
              ? "Primary available"
              : "Primary skipped; backup preferred";
          return (
            '<article class="pool-card"><div class="pool-head"><h3>' +
            esc(p.id) +
            "</h3>" +
            pill(
              !active ? "bad" : p.state === "healthy" ? "ok" : "warn",
              label,
            ) +
            '</div><ol class="route-order">' +
            members
              .map(
                (m) =>
                  "<li><span>" +
                  m.position +
                  ".</span>" +
                  button("data-open-provider", m.id, m.id) +
                  (m.ready ? "" : pill("warn", "Skipped")) +
                  (m.id === p.active_member
                    ? pill("accent", "Next preference")
                    : "") +
                  "</li>",
              )
              .join("") +
            '</ol><p class="small">' +
            (active
              ? num(Math.max(0, p.ready_members - 1)) +
                " other ready workspace(s)."
              : "Recovery or probing may still run; no workspace currently has a ready circuit.") +
            ' Upstream availability not verified.</p><div class="assigned-models"><span class="small">Assigned models:</span>' +
            ((p.models || []).map(modelLink).join(" · ") || "None") +
            "</div><details><summary>Why this preference?</summary>" +
            members
              .map(
                (m) =>
                  '<p class="small"><strong>' +
                  esc(m.id) +
                  "</strong>: " +
                  esc(
                    m.ready
                      ? "configured and circuit closed"
                      : issuesText(m.issues) || "not ready",
                  ) +
                  ".</p>",
              )
              .join("") +
            "</details></article>"
          );
        }
        function workspaceMarkup(m) {
          const groups = pools().filter((p) =>
            p.members.some((w) => w.id === m.id),
          );
          const assigned = [...new Set(groups.flatMap((p) => p.models || []))];
          const r = m.recent || {};
          const token =
            m.token_state === "opaque"
              ? "expiry not inspectable"
              : m.token_state || "not recorded";
          return (
            '<section class="workspace-row"><h3>' +
            providerLink(m.id) +
            "</h3><p>" +
            pill(
              m.ready ? "ok" : "warn",
              m.ready ? "Ready to route" : "Temporarily unavailable",
            ) +
            " " +
            esc(m.credential_type || "Unknown authentication") +
            " · " +
            esc(token) +
            '</p><p class="small">Groups: ' +
            groups.map((p) => esc(p.id)).join(", ") +
            '</p><p class="small">Assigned models: ' +
            assigned.map(modelLink).join(" · ") +
            '</p><p class="small">' +
            num(r.failures || 0) +
            " failures / " +
            num(r.requests || 0) +
            " recorded requests (latest 100). Last request: " +
            (r.last_request_at
              ? esc(new Date(r.last_request_at * 1000).toLocaleString())
              : "Not recorded") +
            '.</p><details><summary>Diagnostics & test instructions</summary><div class="kv-grid">' +
            kv("Host", "<code>" + esc(m.base_url) + "</code>") +
            kv("CLI profile", esc(text(m.auth_profile))) +
            kv("Endpoint style", esc(text(m.endpoint_style))) +
            kv(
              "Circuit",
              esc(
                m.circuit?.is_open
                  ? "Open: temporarily skipped"
                  : "Closed: eligible for a request",
              ),
            ) +
            kv("Consecutive failures", num(m.circuit?.consecutive_failures)) +
            kv(
              "Last failure status",
              esc(text(m.circuit?.last_failure_status || null)),
            ) +
            '</div><p class="small">This command authenticates, checks endpoint coverage, and may make real inference requests with usage charges. Run it on the gateway host.</p><pre>' +
            esc("model-gateway workspace test " + shellQuote(m.id)) +
            "</pre>" +
            button(
              "data-copy-command",
              "model-gateway workspace test " + shellQuote(m.id),
              "Copy workspace test command",
              "btn secondary",
            ) +
            "</details></section>"
          );
        }
        function renderPools() {
          if (!cache.pools) {
            $("poolCards").innerHTML =
              '<p class="notice">Routing group data is not available yet. Refresh to retry.</p>';
            $("workspaceList").innerHTML = "";
            return;
          }
          $("poolCards").innerHTML =
            pools()
              .map(
                (p) =>
                  '<div id="pool-' +
                  esc(p.id) +
                  '">' +
                  poolMarkup(p) +
                  "</div>",
              )
              .join("") ||
            '<div class="empty-state"><h3>No workspace pools configured</h3><p>Direct connections do not need a pool. For Databricks redundancy, the operator CLI can add a workspace to an ordered pool.</p><pre>model-gateway workspace add --help</pre><p class="small">Run this on the gateway host to see setup options. No configuration is changed by this dashboard.</p></div>';
          const seen = new Set();
          const unique = memberRows().filter((m) => {
            if (seen.has(m.id)) return false;
            seen.add(m.id);
            return true;
          });
          $("workspaceList").innerHTML =
            unique.map(workspaceMarkup).join("") ||
            '<p class="small">No pooled workspaces.</p>';
          $("workspaceRecovery").hidden = !unique.some(
            (m) =>
              m.auth_refresh === "databricks-cli" ||
              m.credential_type === "oauth",
          );
        }
        function renderPresets() {
          const data = cache.presets || {},
            auto = data.auto_models || {},
            mp = data.model_presets || {};
          $("presetsMeta").textContent = mp.default_tier
            ? "Default group: " + mp.default_tier
            : "";
          $("presetSummary").innerHTML = ["local", "cloud"]
            .filter((s) => auto[s]?.model || auto[s]?.vision_model)
            .map(
              (s) =>
                '<div class="preset-card"><h3>' +
                esc(auto[s].label || s) +
                "</h3><p>Text: " +
                modelLink(auto[s].model) +
                " · Vision: " +
                modelLink(auto[s].vision_model) +
                "</p></div>",
            )
            .join("");
          const rows = [];
          for (const [tier, p] of Object.entries(mp.presets || {}))
            for (const scope of ["local", "cloud"]) {
              const c = p?.[scope];
              if (!c || (!c.text_model && !c.vision_model)) continue;
              rows.push(
                "<tr><td>" +
                  esc(tier) +
                  "</td><td>" +
                  scope +
                  "</td><td>" +
                  modelLink(c.text_model) +
                  "</td><td>" +
                  modelLink(c.vision_model) +
                  "</td><td>" +
                  esc(c.source_policy || c.residency_mode || "Not recorded") +
                  "</td><td>" +
                  esc(
                    c.designed_memory_gb != null
                      ? c.designed_memory_gb + " GB"
                      : "Not recorded",
                  ) +
                  "</td><td>" +
                  esc(c.description || p.intent || "") +
                  "</td></tr>",
              );
            }
          $("presets").querySelector("tbody").innerHTML =
            rows.join("") ||
            '<tr class="empty"><td colspan="7">No preset groups configured. Use the individual model IDs above.</td></tr>';
        }
        function renderRequests() {
          const q = $("requestFilter").value.trim().toLowerCase(),
            wanted = $("requestOutcome").value,
            all = cache.requests?.requests || [];
          const rows = all.filter(
            (r) =>
              (!q ||
                [r.model, r.provider, r.provider_model_id, r.status, r.endpoint]
                  .join(" ")
                  .toLowerCase()
                  .includes(q)) &&
              (!wanted || outcome(r) === wanted),
          );
          $("recentReq").querySelector("tbody").innerHTML =
            rows
              .map(
                (r) =>
                  '<tr class="clickable-row"><td>' +
                  button("data-open-request", r.id, when(r)) +
                  '</td><td data-label="Requested">' +
                  modelLink(r.model) +
                  '</td><td data-label="Recorded connection">' +
                  providerLink(r.provider) +
                  '</td><td data-label="Outcome">' +
                  outcomePill(r) +
                  '</td><td class="num" data-label="Duration">' +
                  duration(r.latency_ms) +
                  '</td><td class="num" data-label="Cost">' +
                  cost(r.cost_usd) +
                  (r.cost_usd != null && !r.pricing_complete
                    ? " · partial"
                    : "") +
                  "</td></tr>",
              )
              .join("") ||
            '<tr class="empty"><td colspan="6">' +
              (!cache.requests
                ? "Requests not loaded."
                : all.length
                  ? "No matching requests in the latest 50. Clear search or choose All outcomes."
                  : "No requests recorded yet. Copy an example from a model to get started.") +
              "</td></tr>";
        }
        async function loadUsage(window, gen = generation) {
          currentWindow = window;
          const seq = ++usageSeq;
          for (const b of document.querySelectorAll("#winSeg button"))
            b.setAttribute("aria-pressed", String(b.dataset.w === window));
          $("usageRange").textContent =
            "Loading " +
            windows[window].toLowerCase() +
            "…" +
            (cache.usage
              ? " Previous data: " + windows[cache.usageWindow] + "."
              : "");
          try {
            const u = await request(
              "/admin/api/usage" +
                (window ? "?window=" + encodeURIComponent(window) : ""),
            );
            if (seq !== usageSeq || gen !== generation || !unlocked) return;
            cache.usage = u;
            cache.usageWindow = window;
            delete failures.Usage;
            $("usageErr").classList.add("hidden");
            renderUsage(u);
            $("usageRange").textContent =
              windows[window] + " · updated " + stamp();
          } catch (e) {
            if (seq !== usageSeq || gen !== generation) return;
            if (e instanceof AuthError) {
              lock(e.message);
              return;
            }
            failures.Usage = e.message;
            $("usageErr").textContent = "Could not refresh usage. " + e.message;
            $("usageErr").classList.remove("hidden");
            $("usageRange").textContent = cache.usage
              ? "Stale data: " +
                windows[cache.usageWindow] +
                ". Retry by choosing a window."
              : "Usage unavailable. Choose a window to retry.";
          } finally {
            if (gen === generation) updateErrors();
          }
        }
        function renderUsage(u) {
          const s = u.summary || {};
          $("uRequests").textContent = num(s.requests);
          $("uRequestsDetail").textContent =
            num(s.ok || 0) + " successful / " + num(s.errors || 0) + " errors";
          $("uTokens").textContent = num(
            (s.input_tokens || 0) + (s.output_tokens || 0),
          );
          $("uTokensDetail").textContent =
            num(s.usage_reported_requests || 0) +
            "/" +
            num(s.requests || 0) +
            " reported";
          $("uCost").textContent = cost(s.cost_usd);
          $("uCostDetail").textContent =
            num(s.known_cost_requests || 0) +
            "/" +
            num(s.requests || 0) +
            " known · " +
            num(s.partial_pricing_requests || 0) +
            " partial";
          $("uLatency").textContent = duration(s.avg_latency_ms);
          $("uLatencyDetail").textContent = "Recorded request latency";
          const rows = u.by_route || u.by_model || [];
          $("usageByModel").querySelector("tbody").innerHTML =
            rows
              .map(
                (r) =>
                  "<tr><td>" +
                  modelLink(r.dim) +
                  '<span class="small">' +
                  esc(
                    [r.provider, r.provider_model_id]
                      .filter(Boolean)
                      .join(" · ") || "Route not recorded",
                  ) +
                  '</span></td><td class="num">' +
                  num(r.requests) +
                  '</td><td class="num">' +
                  num(r.errors || 0) +
                  '</td><td class="num">' +
                  cost(r.cost_usd) +
                  '</td><td class="num">' +
                  duration(r.avg_latency_ms) +
                  "</td></tr>",
              )
              .join("") ||
            '<tr class="empty"><td colspan="5">No requests in this window.</td></tr>';
          $("tokenAccounting").querySelector("tbody").innerHTML =
            rows
              .map(
                (r) =>
                  "<tr><td>" +
                  esc(r.dim) +
                  '</td><td class="num">' +
                  num(r.input_tokens) +
                  '</td><td class="num">' +
                  num(r.output_tokens) +
                  '</td><td class="num">' +
                  num(r.cached_read_tokens) +
                  '</td><td class="num">' +
                  num(r.usage_reported_requests || 0) +
                  "/" +
                  num(r.requests) +
                  '</td><td class="num">' +
                  num(r.complete_pricing_requests || 0) +
                  "/" +
                  num(r.requests) +
                  "</td></tr>",
              )
              .join("") ||
            '<tr class="empty"><td colspan="6">No accounting data.</td></tr>';
          $("usageByProvider").querySelector("tbody").innerHTML =
            (u.by_provider || [])
              .map(
                (r) =>
                  "<tr><td>" +
                  providerLink(r.dim) +
                  '</td><td class="num">' +
                  num(r.requests) +
                  '</td><td class="num">' +
                  num(r.errors || 0) +
                  '</td><td class="num">' +
                  cost(r.cost_usd) +
                  "</td></tr>",
              )
              .join("") ||
            '<tr class="empty"><td colspan="4">No requests in this window.</td></tr>';
        }
        function openDrawer(kind, name) {
          detailKind = kind;
          detailName = name;
          if ($("drawer").hidden) {
            returnFocus = document.activeElement;
            $("drawer").hidden = false;
            $("drawerScrim").hidden = false;
            $("appChrome").inert = true;
            document.body.classList.add("drawer-open");
          }
          $("drawer").setAttribute("aria-label", kind + ": " + name);
          $("drawer").focus();
        }
        function closeDrawer(opts = {}) {
          ++detailSeq;
          $("drawer").hidden = true;
          $("drawerScrim").hidden = true;
          $("appChrome").inert = false;
          document.body.classList.remove("drawer-open");
          $("drawerBody").innerHTML = "";
          detailKind = null;
          detailName = null;
          if (!opts.skipHash) setHash(currentTab);
          if (
            returnFocus?.isConnected &&
            returnFocus.getClientRects().length &&
            !returnFocus.closest("[inert]")
          )
            returnFocus.focus();
          else if (unlocked) $("tab-" + currentTab).focus();
          returnFocus = null;
        }
        function detailHeader(name, kind) {
          return (
            '<div class="detail-head"><div><p class="small">' +
            esc(kind) +
            "</p><h2>" +
            esc(name) +
            '</h2></div><button class="close" data-close-detail type="button">Close</button></div>'
          );
        }
        async function detail(kind, name, path, render, opts = {}) {
          const tab = {
            model: "models",
            provider: "connections",
            request: "activity",
          }[kind];
          showTab(tab, { skipHash: true });
          if (!opts.skipHash) setHash(tab, name);
          openDrawer(kind, name);
          const seq = ++detailSeq,
            gen = generation;
          $("drawerBody").innerHTML =
            detailHeader(name, kind) +
            '<div class="loading-bar">Loading details…</div>';
          try {
            const data = await request(path);
            if (seq === detailSeq && gen === generation && unlocked)
              render(name, data);
          } catch (e) {
            if (seq !== detailSeq || gen !== generation) return;
            if (e instanceof AuthError) {
              lock(e.message);
              return;
            }
            $("drawerBody").innerHTML =
              detailHeader(name, kind) +
              '<div class="inline-err" role="alert">' +
              esc(e.message) +
              "</div>" +
              button(
                "data-retry-detail",
                kind,
                "Retry details",
                "btn secondary",
              );
          }
        }
        function showModelDetail(name, opts) {
          return detail(
            "model",
            name,
            "/admin/api/models/" +
              encodeURIComponent(name) +
              "/stats?window=24h",
            renderModelDetail,
            opts,
          );
        }
        function showProviderDetail(name, opts) {
          return detail(
            "provider",
            name,
            "/admin/api/providers/" +
              encodeURIComponent(name) +
              "/stats?window=24h",
            renderProviderDetail,
            opts,
          );
        }
        function showRequestDetail(name, opts) {
          return detail(
            "request",
            name,
            "/admin/api/requests/" + encodeURIComponent(name),
            (n, d) => renderRequestDetail(d.request || {}),
            opts,
          );
        }
        function shellQuote(value) {
          return "'" + String(value).replace(/'/g, "'\\''") + "'";
        }
        function usageExample(name) {
          return (
            "curl " +
            shellQuote(location.origin + api("/v1/chat/completions")) +
            ' \\\n  -H "Authorization: Bearer $MODEL_GATEWAY_API_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d ' +
            shellQuote(
              JSON.stringify({
                model: name,
                messages: [{ role: "user", content: "Hello" }],
              }),
            )
          );
        }
        function recentTable(rows) {
          return (
            '<p class="small">Latest 25 recorded requests, independent of the 24-hour usage summary.</p><div class="scroll"><table class="inventory"><thead><tr><th>Request</th><th>Model</th><th>Outcome</th><th>Duration</th></tr></thead><tbody>' +
            ((rows || [])
              .map(
                (r) =>
                  "<tr><td>" +
                  button("data-open-request", r.id, when(r)) +
                  "</td><td>" +
                  modelLink(r.model) +
                  "</td><td>" +
                  outcomePill(r) +
                  "</td><td>" +
                  duration(r.latency_ms) +
                  "</td></tr>",
              )
              .join("") ||
              '<tr class="empty"><td colspan="4">No requests recorded.</td></tr>') +
            "</tbody></table></div>"
          );
        }
        function renderModelDetail(name, data) {
          if (!data.model) {
            $("drawerBody").innerHTML =
              detailHeader(name, "Model") +
              '<p class="empty-state">This model is no longer in the catalog.</p>';
            return;
          }
          const m = data.model,
            groups = modelPools(m),
            example = usageExample(name),
            price = m.pricing_status || "unknown";
          let html =
            detailHeader(name, "Model") +
            '<div class="toolbar">' +
            modelAvailability(m) +
            button("data-edit-model", name, "Edit model", "btn secondary") +
            "</div>";
          html +=
            '<section class="detail-section"><h3>Use this model</h3><p>Client-facing ID: <code>' +
            esc(name) +
            "</code> " +
            button("data-copy-command", name, "Copy ID") +
            '</p><p class="small">Accepted IDs: ' +
            esc((data.routable_ids || m.routable_ids || [name]).join(", ")) +
            '</p><p class="small">Set <code>MODEL_GATEWAY_API_KEY</code> to an authorized client key in your terminal. Running this example sends a real request and may incur charges.</p><pre>' +
            esc(example) +
            "</pre>" +
            button(
              "data-copy-command",
              example,
              "Copy request example",
              "btn secondary",
            ) +
            "</section>";
          html +=
            '<section class="detail-section"><h3>Capabilities & cost</h3><div class="kv-grid">' +
            kv(
              "Vision",
              m.vision ? "Native image input" : "No native image input",
            ) +
            kv(
              "Tools",
              m.tools == null
                ? "Support not recorded"
                : m.tools
                  ? "Supported"
                  : "Not supported",
            ) +
            kv("Reasoning", esc(m.thinking || "Not recorded")) +
            kv(
              "Reasoning levels",
              esc((m.thinking_levels || []).join(", ") || "Not recorded"),
            ) +
            kv("Context limit", num(m.context) + " tokens") +
            kv("Maximum output", num(m.max_output_tokens) + " tokens") +
            kv("Pricing status", esc(price)) +
            kv(
              "USD / million tokens",
              m.pricing
                ? Object.entries(m.pricing)
                    .map(([k, v]) => esc(k.replace(/_/g, " ")) + ": $" + esc(v))
                    .join(" · ")
                : price === "unmetered"
                  ? "Unmetered; recorded cost is zero"
                  : "Pricing not configured",
            ) +
            "</div></section>";
          html +=
            '<section class="detail-section"><h3>Routing</h3>' +
            (groups.length
              ? groups.map(poolMarkup).join("")
              : m.pool
                ? "<p>Routing group: " +
                  esc(m.pool) +
                  ". Workspace details are unavailable; refresh routing data.</p>"
                : "<p>Direct connection: " +
                  providerLink(m.configured_provider || m.provider) +
                  '</p><p class="small">Configuration readiness does not verify upstream availability.</p>') +
            '<div class="kv-grid">' +
            kv(
              "Upstream model ID",
              "<code>" +
                esc(text(m.provider_model_id || m.omlx_id)) +
                "</code>",
            ) +
            kv(
              "Configured model fallback",
              m.fallback_model
                ? modelLink(m.fallback_model)
                : "None configured",
            ) +
            kv("Image handling", visionRouting(m)) +
            "</div></section>";
          html +=
            '<section class="detail-section"><h3>Last 24 hours</h3><div class="strip">' +
            stats(data.usage || {}) +
            '</div></section><section class="detail-section"><h3>Recent requests</h3>' +
            recentTable(data.recent) +
            "</section>";
          $("drawerBody").innerHTML = html;
          $("drawerBody")
            .querySelector("[data-edit-model]")
            .setAttribute("data-mgmt", "");
        }
        function visionRouting(m) {
          if (m.composite)
            return (
              "Composite: text " +
              modelLink(m.composite.text_model) +
              "; images " +
              modelLink(m.composite.vision_model) +
              ". " +
              (m.composite.image_handling === "reroute"
                ? "The vision model answers image requests."
                : "The vision model extracts observations; the text model answers.")
            );
          if (m.vision) return "Native image input; no helper needed.";
          if (!m.vision_route)
            return "No image helper configured for this route.";
          return (
            "Configured helper: " +
            modelLink(m.vision_route.model) +
            ". " +
            (m.vision_route.mode === "extract_then_answer"
              ? "The helper extracts observations; this model remains the answerer."
              : "Image requests are answered by the helper instead.")
          );
        }
        function renderProviderDetail(id, data) {
          const p = data.provider;
          if (!p) {
            $("drawerBody").innerHTML =
              detailHeader(id, "Connection") +
              '<p class="empty-state">This connection is no longer configured.</p>';
            return;
          }
          const member = memberRows().find((m) => m.id === id),
            assigned = associatedModels(p),
            groups = pools().filter((g) => g.members.some((m) => m.id === id));
          let html =
            detailHeader(id, "Connection") +
            '<div class="toolbar">' +
            pill(
              configReady(p) ? "ok" : "warn",
              configReady(p) ? "Configuration complete" : "Needs configuration",
            ) +
            button(
              "data-edit-provider",
              id,
              "Edit connection",
              "btn secondary",
            ) +
            '</div><p class="small" style="margin-top:12px">Readiness is a local configuration check, not an upstream availability test.</p>';
          const check = checks.get(id);
          html +=
            '<section class="detail-section"><h3>Connection</h3><div class="kv-grid">' +
            kv(
              "Base URL",
              "<code>" + esc(p.base_url || "Not configured") + "</code>",
            ) +
            kv("Protocol", esc(p.protocol)) +
            kv("API key", p.has_api_key ? "Present (hidden)" : "Missing") +
            kv(
              "Last explicit check",
              esc(
                check
                  ? check.label + " · " + check.at + " (this browser session)"
                  : "Not recorded",
              ),
            ) +
            "</div>" +
            (p.issues?.length
              ? '<p class="notice warn">' + esc(issuesText(p.issues)) + "</p>"
              : "") +
            '</section><section class="detail-section"><h3>Assigned models</h3><div class="assigned-models">' +
            (assigned.map((m) => modelLink(modelName(m))).join(" · ") ||
              "Unused: no enabled models assigned.") +
            "</div></section>";
          if (groups.length)
            html +=
              '<section class="detail-section"><h3>Routing groups (workspace pools)</h3>' +
              groups.map(poolMarkup).join("") +
              "</section>";
          if (member)
            html +=
              '<section class="detail-section"><h3>Workspace diagnostics</h3>' +
              workspaceMarkup(member) +
              "</section>";
          html +=
            '<section class="detail-section"><h3>Last 24 hours</h3><div class="strip">' +
            stats(data.usage || {}) +
            '</div></section><section class="detail-section"><h3>Recent requests</h3>' +
            recentTable(data.recent) +
            "</section>";
          $("drawerBody").innerHTML = html;
          $("drawerBody")
            .querySelector("[data-edit-provider]")
            .setAttribute("data-mgmt", "");
        }
        function renderRequestDetail(r) {
          let html =
            detailHeader("Request " + text(r.id), "Activity") +
            "<p>" +
            outcomePill(r) +
            " " +
            esc(when(r)) +
            "</p>" +
            (r.error ? '<p class="inline-err">' + esc(r.error) + "</p>" : "");
          html +=
            '<section class="detail-section"><h3>Recorded routing</h3><div class="kv-grid">' +
            kv("Requested model", modelLink(r.model)) +
            kv("Recorded connection", providerLink(r.provider)) +
            kv(
              "Recorded upstream ID",
              "<code>" + esc(text(r.provider_model_id)) + "</code>",
            ) +
            kv("Failover / attempt chain", "Not recorded") +
            kv("Vision extraction link", "Not recorded") +
            kv("Endpoint", "<code>" + esc(text(r.endpoint)) + "</code>") +
            '</div><p class="small">The ledger does not record the complete attempt chain or guarantee the final workspace after failover. Do not infer that no failover occurred.</p></section>';
          html +=
            '<section class="detail-section"><h3>Timing & cost</h3><div class="kv-grid">' +
            kv("Recorded latency", duration(r.latency_ms)) +
            kv("Mode", r.is_stream ? "Streaming" : "Non-streaming") +
            kv("Known cost", cost(r.cost_usd)) +
            kv(
              "Accounting",
              r.cost_usd == null
                ? "Cost unknown"
                : r.pricing_complete
                  ? "Complete"
                  : "Partial: some pricing classes missing",
            ) +
            '</div><p class="small">Recorded latency is not a time-to-first-token or a guaranteed full stream duration.</p></section><details><summary>Token accounting & protocol details</summary><div class="kv-grid">' +
            kv("Usage reported", r.usage_reported ? "Yes" : "No") +
            kv("Input tokens", num(r.input_tokens)) +
            kv("Output tokens", num(r.output_tokens)) +
            kv("Cached read", num(r.cached_read_tokens)) +
            kv("Cache write", num(r.cache_write_tokens)) +
            kv("1-hour cache write", num(r.cache_write_1h_tokens)) +
            kv("Reasoning tokens", num(r.reasoning_tokens)) +
            kv(
              "Missing pricing classes",
              esc(
                Array.isArray(r.missing_pricing_classes)
                  ? r.missing_pricing_classes.join(", ") || "None"
                  : text(r.missing_pricing_classes),
              ),
            ) +
            kv("HTTP status", esc(text(r.status))) +
            "</div></details>";
          $("drawerBody").innerHTML = html;
        }
        function resetProvider() {
          editingProvider = null;
          for (const id of ["pId", "pBaseUrl", "pApiKey"]) $(id).value = "";
          $("pId").readOnly = false;
          $("pProtocol").value = "openai";
          $("pMsg").textContent = "";
          $("providerFormTitle").textContent = "Add a connection";
          $("deleteProviderBtn").hidden = true;
          $("discoverPanel").classList.add("hidden");
        }
        function resetModel() {
          editingModel = null;
          for (const id of [
            "mName",
            "mPmid",
            "mAlias",
            "mContext",
            "mMaxOut",
            "mThinking",
            "mThinkingLevels",
            "mThinkingFmt",
            "mPricing",
            "mDesc",
          ])
            $(id).value = "";
          $("mName").readOnly = false;
          $("mVision").checked = false;
          $("mEnabled").checked = true;
          $("mPricingStatus").value = "unknown";
          $("mMsg").textContent = "";
          $("modelFormTitle").textContent = "Register a model";
          $("deleteModelBtn").hidden = true;
          $("previewPanel").classList.add("hidden");
          fillSelect(
            "mProvider",
            providers().map((p) => p.id),
            "Choose a connection",
          );
          fillSelect(
            "mPool",
            pools().map((p) => p.id),
            "Direct connection",
          );
          $("mPool").disabled = true;
        }
        function openForm(kind) {
          if (mutationBusy) {
            toast("Wait for the current operation to finish.");
            return false;
          }
          if (document.body.dataset.writes !== "true") {
            toast("Management is read-only. See Settings.");
            return false;
          }
          closeDrawer({ skipHash: true });
          showTab(kind === "model" ? "models" : "connections");
          const form = $(kind === "model" ? "modelFormset" : "providerFormset");
          form.open = true;
          form.scrollIntoView({ block: "start" });
          $(kind === "model" ? "mName" : "pId").focus();
          return true;
        }
        function editProvider(id) {
          if (mutationBusy) {
            toast("Wait for the current operation to finish.");
            return;
          }
          const p = providers().find((p) => p.id === id);
          if (!p || failures.Connections) {
            toast("Refresh connections successfully before editing.");
            return;
          }
          resetProvider();
          editingProvider = id;
          $("pId").value = id;
          $("pId").readOnly = true;
          $("pBaseUrl").value = p.base_url || "";
          $("pProtocol").value = p.protocol || "openai";
          $("pApiKey").value = "";
          if (p.base_url_redacted)
            setMsg(
              "pMsg",
              "This URL contains hidden authentication components. Enter a complete replacement URL before saving; the displayed redacted URL cannot safely replace it.",
            );
          $("providerFormTitle").textContent = "Edit connection: " + id;
          $("deleteProviderBtn").hidden = false;
          openForm("provider");
        }
        function editModel(name) {
          if (mutationBusy) {
            toast("Wait for the current operation to finish.");
            return;
          }
          const m = findModel(name);
          if (
            !m ||
            failures.Models ||
            !cache.pools ||
            failures["Routing groups"]
          ) {
            toast(
              "Refresh models and routing groups successfully before editing.",
            );
            return;
          }
          resetModel();
          editingModel = m;
          $("mName").value = modelName(m);
          $("mName").readOnly = true;
          $("mProvider").value = m.configured_provider || m.provider || "";
          for (const [field, value] of Object.entries({
            mPmid: m.provider_model_id || m.omlx_id,
            mAlias: m.alias,
            mContext: m.context,
            mMaxOut: m.max_output_tokens,
            mThinking: m.thinking,
            mThinkingLevels: (m.thinking_levels || []).join(", "),
            mThinkingFmt: m.thinking_format,
            mPricing: m.pricing ? JSON.stringify(m.pricing) : "",
            mDesc: m.desc,
          }))
            $(field).value = value ?? "";
          $("mVision").checked = !!m.vision;
          $("mEnabled").checked = m.enabled !== false;
          $("mPricingStatus").value = m.pricing_status || "unknown";
          $("mPool").value = m.pool || modelPools(m)[0]?.id || "";
          $("mPool").disabled = true;
          $("modelFormTitle").textContent = "Edit model: " + modelName(m);
          $("deleteModelBtn").hidden = false;
          openForm("model");
        }
        function setMsg(id, message, ok = false) {
          $(id).textContent = message;
          $(id).className = "msg " + (ok ? "ok-c" : "bad-c");
        }
        async function mutation(buttonId, messageId, fn) {
          if (
            document.body.dataset.writes !== "true" ||
            mutationBusy ||
            $(buttonId).disabled
          )
            return;
          mutationBusy = true;
          const epoch = sessionEpoch;
          const controls = [
            ...document.querySelectorAll(
              ".formset input, .formset select, .formset textarea, .formset button, [data-add-model], [data-add-provider], [data-edit-model], [data-edit-provider]",
            ),
          ].map((el) => [el, el.disabled]);
          for (const [el] of controls) el.disabled = true;
          setMsg(messageId, "Working…", true);
          try {
            await fn();
          } catch (e) {
            if (epoch !== sessionEpoch) return;
            if (e instanceof AuthError) lock(e.message);
            else setMsg(messageId, e.message);
          } finally {
            mutationBusy = false;
            for (const [el, disabled] of controls) el.disabled = disabled;
            $("mPool").disabled = true;
          }
        }
        function collectModel() {
          const name = $("mName").value.trim(),
            provider = $("mProvider").value,
            upstream = $("mPmid").value.trim();
          if (!name || !provider || !upstream)
            throw new Error(
              "Enter a model ID, connection, and upstream model ID.",
            );
          const body = {
            provider,
            provider_model_id: upstream,
            vision: $("mVision").checked,
            enabled: $("mEnabled").checked,
            alias: $("mAlias").value.trim(),
            desc: $("mDesc").value.trim(),
            thinking: $("mThinking").value.trim(),
            thinking_levels: $("mThinkingLevels")
              .value.split(",")
              .map((v) => v.trim())
              .filter(Boolean),
            thinking_format: $("mThinkingFmt").value.trim(),
            pricing_status: $("mPricingStatus").value,
          };
          for (const [field, key] of [
            ["mContext", "context"],
            ["mMaxOut", "max_output_tokens"],
          ]) {
            if ($(field).value) {
              const n = Number($(field).value);
              if (!Number.isSafeInteger(n) || n <= 0)
                throw new Error("Token limits must be positive whole numbers.");
              body[key] = n;
            }
          }
          if (body.pricing_status === "metered") {
            try {
              body.pricing = JSON.parse($("mPricing").value);
            } catch {
              throw new Error(
                "Enter valid pricing JSON with input and output rates.",
              );
            }
          } else body.pricing = null;
          return { name, body };
        }
        async function saveProvider() {
          return mutation("saveProviderBtn", "pMsg", async () => {
            const id = $("pId").value.trim().toLowerCase();
            if (!id || !$("pBaseUrl").value.trim())
              throw new Error("Enter a connection ID and base URL.");
            const existing = providers().find((p) => p.id === id);
            if (
              existing?.base_url_redacted &&
              $("pBaseUrl").value.trim() === existing.base_url
            )
              throw new Error(
                "The displayed URL is redacted. Enter a complete replacement URL or use the operator CLI to preserve its hidden components.",
              );
            const body = {
              base_url: $("pBaseUrl").value.trim(),
              protocol: $("pProtocol").value,
            };
            if ($("pApiKey").value) body.api_key = $("pApiKey").value;
            await request(
              "/admin/api/providers/" + encodeURIComponent(id),
              "POST",
              body,
            );
            $("pApiKey").value = "";
            checks.delete(id);
            editingProvider = id;
            $("pId").readOnly = true;
            $("deleteProviderBtn").hidden = false;
            setMsg(
              "pMsg",
              "Saved and activated. Check the saved connection, then discover models.",
              true,
            );
            await refresh();
          });
        }
        async function validateProvider() {
          return mutation("validateProviderBtn", "pMsg", async () => {
            const id = $("pId").value.trim();
            if (!providers().some((p) => p.id === id))
              throw new Error("Save this connection before checking it.");
            const r = await request(
              "/admin/api/providers/" + encodeURIComponent(id) + "/validate",
              "POST",
            );
            checks.set(id, {
              at: stamp(),
              label: r.ok ? "Inventory accessible" : "Inventory check failed",
            });
            renderProviders();
            setMsg(
              "pMsg",
              r.ok
                ? "Model inventory accessible: " +
                    num(r.model_count) +
                    " models. Inference was not tested."
                : "Inventory check failed: " + text(r.error || r.status_code),
              !!r.ok,
            );
          });
        }
        async function discoverModels() {
          return mutation("discoverBtn", "pMsg", async () => {
            const id = $("pId").value.trim();
            if (!providers().some((p) => p.id === id))
              throw new Error(
                "Save this connection before discovering models.",
              );
            const r = await request(
              "/admin/api/providers/" + encodeURIComponent(id) + "/discover",
              "POST",
            );
            if (r.status !== "verified")
              throw new Error(
                "Discovery did not verify this connection: " +
                  text(r.status) +
                  ". Check authentication and protocol support.",
              );
            $("discoverPanel").classList.remove("hidden");
            $("discoverMeta").textContent =
              (r.models || []).length +
              " upstream IDs. Review capabilities and pricing before registering.";
            $("discoverTable").querySelector("tbody").innerHTML =
              (r.models || [])
                .map(
                  (m) =>
                    "<tr><td><code>" +
                    esc(m.id) +
                    "</code></td><td>" +
                    (m.registered
                      ? "Already registered"
                      : button(
                          "data-register-discovered",
                          m.id,
                          "Register",
                          "btn secondary",
                        )) +
                    "</td></tr>",
                )
                .join("") ||
              '<tr class="empty"><td colspan="2">No models were discovered.</td></tr>';
            setMsg(
              "pMsg",
              "Discovery complete. Select an unregistered model below.",
              true,
            );
          });
        }
        async function saveModel() {
          return mutation("saveModelBtn", "mMsg", async () => {
            const { name, body } = collectModel();
            await request(
              "/admin/api/models/" + encodeURIComponent(name),
              "POST",
              body,
            );
            setMsg(
              "mMsg",
              "Saved and activated. Changes are live in the machine-local catalog.",
              true,
            );
            $("previewPanel").classList.add("hidden");
            await refresh();
            editingModel = findModel(name);
            $("mName").readOnly = true;
            $("deleteModelBtn").hidden = false;
          });
        }
        async function previewModel() {
          return mutation("previewModelBtn", "mMsg", async () => {
            const { name, body } = collectModel();
            const pool =
              editingModel?.pool || modelPools(editingModel || { name })[0]?.id;
            if (pool) body.pool = pool;
            const r = await request(
              "/admin/api/models/" + encodeURIComponent(name) + "/preview",
              "POST",
              body,
            );
            $("previewPanel").classList.remove("hidden");
            $("previewPanel").innerHTML =
              "<h3>Configuration preview</h3><p>" +
              pill(
                r.routable ? "ok" : "warn",
                r.routable ? "Locally routable" : "Needs attention",
              ) +
              '</p><p class="small">No upstream request was made. Circuit state is not checked. No configuration was saved.</p><ol>' +
              (r.route || [])
                .map(
                  (p) =>
                    "<li>" +
                    esc(p.provider) +
                    ": " +
                    esc(
                      p.usable
                        ? "configuration complete"
                        : p.reason || "unavailable",
                    ) +
                    "</li>",
                )
                .join("") +
              "</ol>" +
              (r.issues || [])
                .concat(
                  (r.clashes || []).map(
                    (c) => "ID " + c.id + " belongs to " + c.model,
                  ),
                )
                .map((i) => '<p class="bad-c">' + esc(i) + "</p>")
                .join("");
            setMsg(
              "mMsg",
              r.ok
                ? "Preview complete. Review before saving."
                : "Resolve the preview issues before saving.",
              !!r.ok,
            );
          });
        }
        async function deleteProvider() {
          return mutation("deleteProviderBtn", "pMsg", async () => {
            const id = $("pId").value.trim(),
              p = providers().find((p) => p.id === id);
            if (!p) throw new Error("Select an existing connection.");
            if (p.pool_memberships?.length)
              throw new Error(
                "This connection belongs to workspace pools. Use the transactional workspace CLI to change membership before deletion.",
              );
            const n = associatedModels(p).length;
            if (
              !confirm(
                "Delete connection " +
                  id +
                  "? " +
                  n +
                  " enabled model(s) reference it. Configuration is reloaded immediately.",
              )
            ) {
              setMsg("pMsg", "Deletion canceled.", true);
              return;
            }
            await request(
              "/admin/api/providers/" + encodeURIComponent(id),
              "DELETE",
            );
            resetProvider();
            setMsg(
              "pMsg",
              "Connection deleted and configuration reloaded.",
              true,
            );
            await refresh();
          });
        }
        async function deleteModel() {
          return mutation("deleteModelBtn", "mMsg", async () => {
            const name = $("mName").value.trim();
            if (!name) throw new Error("Select a model.");
            if (
              !confirm(
                "Delete " +
                  name +
                  " and its aliases from the catalog? Clients using these IDs will no longer be able to route requests.",
              )
            ) {
              setMsg("mMsg", "Deletion canceled.", true);
              return;
            }
            await request(
              "/admin/api/models/" + encodeURIComponent(name),
              "DELETE",
            );
            resetModel();
            setMsg("mMsg", "Model deleted from the catalog.", true);
            await refresh();
          });
        }
        document.addEventListener("click", (e) => {
          const b = e.target.closest("button");
          if (!b) return;
          if (b.dataset.tab) {
            closeDrawer({ skipHash: true });
            showTab(b.dataset.tab);
            return;
          }
          if (b.dataset.go) {
            closeDrawer({ skipHash: true });
            showTab(b.dataset.go);
            return;
          }
          if (b.hasAttribute("data-view-failures")) {
            showTab("activity");
            showActivity("requests");
            $("requestOutcome").value = "failed";
            renderRequests();
            return;
          }
          if (b.dataset.activity) {
            showActivity(b.dataset.activity);
            return;
          }
          if (b.hasAttribute("data-close-detail")) {
            closeDrawer();
            return;
          }
          if (b.hasAttribute("data-copy-command")) {
            copy(b.dataset.copyCommand);
            return;
          }
          if (b.dataset.openModel) {
            showModelDetail(b.dataset.openModel);
            return;
          }
          if (b.dataset.openProvider) {
            showProviderDetail(b.dataset.openProvider);
            return;
          }
          if (b.dataset.openRequest) {
            showRequestDetail(b.dataset.openRequest);
            return;
          }
          if (b.dataset.showPool) {
            closeDrawer({ skipHash: true });
            showTab("connections");
            $("routingGroups").open = true;
            document
              .getElementById("pool-" + b.dataset.showPool)
              ?.scrollIntoView({ block: "start" });
            return;
          }
          if (b.dataset.editProvider) {
            editProvider(b.dataset.editProvider);
            return;
          }
          if (b.dataset.editModel) {
            editModel(b.dataset.editModel);
            return;
          }
          if (b.hasAttribute("data-add-provider")) {
            if (mutationBusy) return;
            resetProvider();
            openForm("provider");
            return;
          }
          if (b.hasAttribute("data-add-model")) {
            if (mutationBusy) return;
            resetModel();
            openForm("model");
            return;
          }
          if (b.hasAttribute("data-cancel-provider")) {
            resetProvider();
            $("providerFormset").open = false;
            return;
          }
          if (b.hasAttribute("data-cancel-model")) {
            resetModel();
            $("modelFormset").open = false;
            return;
          }
          if (b.dataset.registerDiscovered) {
            const p = $("pId").value;
            resetModel();
            $("mName").value = b.dataset.registerDiscovered.split("/").pop();
            $("mProvider").value = p;
            $("mPmid").value = b.dataset.registerDiscovered;
            openForm("model");
            setMsg(
              "mMsg",
              "Discovered IDs do not include capabilities or prices. Review the fields and preview the route.",
              true,
            );
            return;
          }
          if (b.dataset.retryDetail) {
            const f = {
              model: showModelDetail,
              provider: showProviderDetail,
              request: showRequestDetail,
            }[b.dataset.retryDetail];
            f?.(detailName);
          }
        });
        for (const [id, event, fn] of [
          ["modelFilter", "input", renderModels],
          ["modelLocality", "change", renderModels],
          ["modelProviderFilter", "change", renderModels],
          ["modelCapability", "change", renderModels],
          ["providerFilter", "input", renderProviders],
          ["requestFilter", "input", renderRequests],
          ["requestOutcome", "change", renderRequests],
        ])
          $(id).addEventListener(event, fn);
        for (const [id, fn] of [
          ["unlockBtn", unlock],
          ["refreshBtn", refresh],
          ["lockBtn", () => lock()],
          ["saveProviderBtn", saveProvider],
          ["validateProviderBtn", validateProvider],
          ["discoverBtn", discoverModels],
          ["deleteProviderBtn", deleteProvider],
          ["saveModelBtn", saveModel],
          ["previewModelBtn", previewModel],
          ["deleteModelBtn", deleteModel],
        ])
          $(id).addEventListener("click", fn);
        $("keyToggle").addEventListener("click", () => {
          const show = $("adminKey").type === "password";
          $("adminKey").type = show ? "text" : "password";
          $("keyToggle").textContent = show ? "Hide" : "Show";
          $("keyToggle").setAttribute(
            "aria-label",
            show ? "Hide admin key" : "Show admin key",
          );
        });
        $("adminKey").addEventListener("keydown", (e) => {
          if (e.key === "Enter") unlock();
        });
        $("drawerScrim").addEventListener("click", () => closeDrawer());
        document.querySelector(".nav").addEventListener("keydown", (e) => {
          const target = e.target.closest("[data-tab]");
          if (!target) return;
          let i = tabs.indexOf(target.dataset.tab);
          if (e.key === "ArrowRight") i = (i + 1) % tabs.length;
          else if (e.key === "ArrowLeft")
            i = (i + tabs.length - 1) % tabs.length;
          else if (e.key === "Home") i = 0;
          else if (e.key === "End") i = tabs.length - 1;
          else return;
          e.preventDefault();
          showTab(tabs[i]);
          $("tab-" + tabs[i]).focus();
        });
        document.addEventListener("keydown", (e) => {
          if (!$("drawer").hidden) {
            if (e.key === "Escape") {
              e.preventDefault();
              closeDrawer();
            }
            if (e.key === "Tab") {
              const els = [
                ...$("drawer").querySelectorAll(
                  'button:not(:disabled),a[href],input,select,textarea,summary,[tabindex="0"]',
                ),
              ].filter((el) => el.getClientRects().length);
              const first = els[0],
                last = els.at(-1);
              if (!first) {
                e.preventDefault();
                $("drawer").focus();
              } else if (
                e.shiftKey &&
                (document.activeElement === first ||
                  document.activeElement === $("drawer"))
              ) {
                e.preventDefault();
                last.focus();
              } else if (
                !e.shiftKey &&
                (document.activeElement === last ||
                  document.activeElement === $("drawer"))
              ) {
                e.preventDefault();
                first.focus();
              }
            }
            return;
          }
          if (
            e.key === "r" &&
            !/INPUT|TEXTAREA|SELECT/.test(e.target.tagName) &&
            !e.metaKey &&
            !e.ctrlKey &&
            !e.altKey
          )
            refresh();
        });
        for (const b of document.querySelectorAll("#winSeg button"))
          b.addEventListener("click", () => loadUsage(b.dataset.w));
        window.addEventListener("popstate", applyHash);
        window.addEventListener("hashchange", applyHash);
        $("thinkingLink").href = api("/v1/debug/thinking");
        resetProvider();
        resetModel();
        showTab(location.hash.slice(1).split("/")[0] || "overview", {
          skipHash: true,
        });
        let stored = "";
        try {
          stored = sessionStorage.getItem(STORAGE_KEY) || "";
        } catch {}
        if (stored) {
          $("adminKey").value = stored;
          unlock();
        } else $("adminKey").focus();
      })();
    </script>
  </body>
</html>
"""
