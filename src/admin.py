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
    return {"models": model_status()}


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
            from src.server import _validate_vision_fallback_policy
            _validate_vision_fallback_policy()
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
    """Aggregate usage/cost by provider, model, endpoint, and status.

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
        "by_model": ledger.aggregate(since=since, until=until, group_by="model"),
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
    rows = [m for m in model_status() if (m.get("name") or m.get("id")) == model_name]
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
      color-scheme: light;
      /* Tinted neutrals (cool slate, low chroma) — never pure white/black. */
      --bg: oklch(98.5% 0.004 250);
      --surface: oklch(100% 0.002 250);
      --surface-2: oklch(97% 0.006 250);
      --rule: oklch(90% 0.012 250);
      --rule-strong: oklch(82% 0.016 250);
      --text: oklch(28% 0.025 250);
      --text-2: oklch(45% 0.020 250);
      --muted: oklch(58% 0.018 250);
      /* Single restrained accent: a steady teal-blue, used for primary actions + state. */
      --accent: oklch(54% 0.110 220);
      --accent-ink: oklch(100% 0.002 250);
      --accent-soft: oklch(94% 0.04 220);
      /* Semantic, restrained. Paired with text/shape, never color alone. */
      --ok: oklch(52% 0.110 155);
      --ok-soft: oklch(94% 0.045 155);
      --warn: oklch(62% 0.120 60);
      --warn-soft: oklch(95% 0.05 65);
      --bad: oklch(54% 0.155 25);
      --bad-soft: oklch(95% 0.05 25);
      --radius: 8px;
      --radius-sm: 5px;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: oklch(20% 0.012 250);
        --surface: oklch(23.5% 0.014 250);
        --surface-2: oklch(26% 0.016 250);
        --rule: oklch(32% 0.018 250);
        --rule-strong: oklch(40% 0.020 250);
        --text: oklch(93% 0.008 250);
        --text-2: oklch(80% 0.012 250);
        --muted: oklch(66% 0.014 250);
        --accent: oklch(70% 0.110 220);
        --accent-ink: oklch(18% 0.02 250);
        --accent-soft: oklch(30% 0.05 220);
        --ok: oklch(72% 0.110 155);
        --ok-soft: oklch(30% 0.05 155);
        --warn: oklch(78% 0.115 65);
        --warn-soft: oklch(30% 0.06 65);
        --bad: oklch(72% 0.140 25);
        --bad-soft: oklch(30% 0.07 25);
      }
    }
    *, *::before, *::after { box-sizing: border-box; }
    html, body { margin: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, Inter, sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }
    a { color: var(--accent); }
    h1, h2, h3 { margin: 0; font-weight: 650; letter-spacing: -0.01em; color: var(--text); }
    h2 { font-size: 15px; }
    p { margin: 0; }
    .muted { color: var(--muted); }
    .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }

    /* ── Top bar ─────────────────────────────────────────────── */
    .topbar {
      position: sticky; top: 0; z-index: 10;
      display: flex; align-items: center; gap: 16px;
      padding: 12px clamp(16px, 4vw, 40px);
      background: color-mix(in oklab, var(--surface) 88%, transparent);
      backdrop-filter: saturate(140%) blur(8px);
      border-bottom: 1px solid var(--rule);
    }
    .brand { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
    .brand h1 { font-size: 16px; font-weight: 700; letter-spacing: -0.015em; }
    .brand .sub { font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .topbar .spacer { flex: 1; }
    .keybar { display: flex; align-items: center; gap: 8px; }
    .field {
      display: inline-flex; align-items: center; gap: 6px;
      background: var(--surface-2); border: 1px solid var(--rule); border-radius: var(--radius-sm);
      padding: 6px 9px;
    }
    .field:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    .field input {
      border: 0; background: transparent; outline: 0; color: inherit; font: inherit;
      width: 180px; padding: 0;
    }
    .field input::placeholder { color: var(--muted); }
    .iconbtn {
      appearance: none; border: 1px solid var(--rule); background: var(--surface-2); color: var(--text-2);
      border-radius: var(--radius-sm); padding: 6px 8px; font: inherit; font-size: 13px; cursor: pointer;
      display: inline-flex; align-items: center; gap: 6px;
    }
    .iconbtn:hover { background: var(--surface); border-color: var(--rule-strong); color: var(--text); }
    .iconbtn:focus-visible { outline: none; box-shadow: 0 0 0 3px var(--accent-soft); border-color: var(--accent); }
    .btn {
      appearance: none; border: 1px solid transparent; background: var(--accent); color: var(--accent-ink);
      border-radius: var(--radius-sm); padding: 7px 13px; font: inherit; font-weight: 600; font-size: 13px; cursor: pointer;
    }
    .btn:hover { filter: brightness(1.06); }
    .btn:focus-visible { outline: none; box-shadow: 0 0 0 3px var(--accent-soft); }
    .btn.secondary { background: var(--surface-2); color: var(--text); border-color: var(--rule); }
    .btn.secondary:hover { border-color: var(--rule-strong); background: var(--surface); }
    .btn.danger { background: var(--bad); color: var(--accent-ink); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .showtoggle { appearance: none; border: 0; background: transparent; color: var(--muted); cursor: pointer; padding: 2px; font-size: 12px; }
    .showtoggle:hover { color: var(--text-2); }

    /* ── Layout ──────────────────────────────────────────────── */
    main { padding: 20px clamp(16px, 4vw, 40px) 56px; display: grid; gap: 28px; max-width: 1320px; margin: 0 auto; }
    section { display: grid; gap: 12px; }
    .sec-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
    .sec-head h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); font-weight: 650; }
    .sec-head .meta { font-size: 12px; color: var(--muted); }

    /* ── Status strip ────────────────────────────────────────── */
    .strip {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      border: 1px solid var(--rule); border-radius: var(--radius); overflow: hidden;
      background: var(--surface);
    }
    .stat { padding: 12px 14px; border-right: 1px solid var(--rule); display: flex; flex-direction: column; gap: 2px; }
    .stat:last-child { border-right: 0; }
    .stat .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); }
    .stat .value { font-size: 18px; font-weight: 650; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
    .stat .detail { font-size: 12px; color: var(--muted); }
    @media (max-width: 640px) { .stat { border-right: 0; border-bottom: 1px solid var(--rule); } }

    /* ── Tables ──────────────────────────────────────────────── */
    .scroll { overflow-x: auto; border: 1px solid var(--rule); border-radius: var(--radius); background: var(--surface); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    thead th {
      text-align: left; padding: 9px 12px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--muted); font-weight: 600; background: var(--surface-2);
      border-bottom: 1px solid var(--rule); white-space: nowrap; position: sticky; top: 0;
    }
    tbody td { padding: 8px 12px; border-bottom: 1px solid var(--rule); vertical-align: middle; font-variant-numeric: tabular-nums; }
    tbody tr:last-child td { border-bottom: 0; }
    tbody tr:hover { background: var(--surface-2); }
    td.num, th.num { text-align: right; }
    .group-row td { background: var(--surface-2); font-weight: 650; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-2); }

    /* ── Pills / state ───────────────────────────────────────── */
    .pill {
      display: inline-flex; align-items: center; gap: 5px;
      font-size: 12px; font-weight: 550; padding: 2px 8px; border-radius: 999px;
      border: 1px solid var(--rule); background: var(--surface-2); white-space: nowrap;
      font-variant-numeric: normal;
    }
    .pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); }
    .pill.ok { background: var(--ok-soft); border-color: color-mix(in oklab, var(--ok) 40%, var(--rule)); color: var(--ok); }
    .pill.ok .dot { background: var(--ok); }
    .pill.warn { background: var(--warn-soft); border-color: color-mix(in oklab, var(--warn) 40%, var(--rule)); color: var(--warn); }
    .pill.warn .dot { background: var(--warn); }
    .pill.bad { background: var(--bad-soft); border-color: color-mix(in oklab, var(--bad) 40%, var(--rule)); color: var(--bad); }
    .pill.bad .dot { background: var(--bad); }
    .pill.accent { background: var(--accent-soft); border-color: color-mix(in oklab, var(--accent) 40%, var(--rule)); color: var(--accent); }
    .id { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }
    .ok-c { color: var(--ok); } .warn-c { color: var(--warn); } .bad-c { color: var(--bad); }

    /* ── Forms ───────────────────────────────────────────────── */
    details.formset {
      border: 1px solid var(--rule); border-radius: var(--radius); background: var(--surface);
    }
    details.formset > summary {
      cursor: pointer; list-style: none; padding: 10px 14px; font-size: 13px; color: var(--text-2); font-weight: 550;
      display: flex; align-items: center; gap: 8px;
    }
    details.formset > summary::-webkit-details-marker { display: none; }
    details.formset > summary::before { content: "+"; color: var(--accent); font-weight: 700; font-size: 15px; }
    details.formset[open] > summary::before { content: "\2013"; }
    details.formset[open] > summary { border-bottom: 1px solid var(--rule); }
    .form-body { padding: 14px; display: grid; gap: 10px; }
    .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }
    .control { display: flex; flex-direction: column; gap: 3px; }
    .control > label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
    .control input[type=text], .control input[type=password], .control input[type=number], .control select, input[type=text].ctl, input[type=password].ctl, input[type=number].ctl {
      font: inherit; font-size: 13px; color: var(--text); background: var(--surface-2);
      border: 1px solid var(--rule); border-radius: var(--radius-sm); padding: 7px 9px; width: 100%;
    }
    .control input:focus, .control select:focus, input.ctl:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    .checkrow { display: flex; gap: 18px; flex-wrap: wrap; align-items: center; }
    .check { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text-2); }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .msg { font-size: 12px; color: var(--muted); }
    .msg.ok-c { color: var(--ok); } .msg.bad-c { color: var(--bad); }
    .hint { font-size: 12px; color: var(--muted); max-width: 70ch; }

    /* ── Segmented control (usage windows) ───────────────────── */
    .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: var(--radius-sm); overflow: hidden; background: var(--surface); }
    .seg button {
      appearance: none; border: 0; background: transparent; color: var(--text-2);
      padding: 6px 12px; font: inherit; font-size: 12px; font-weight: 550; cursor: pointer; border-right: 1px solid var(--rule);
    }
    .seg button:last-child { border-right: 0; }
    .seg button:hover { background: var(--surface-2); color: var(--text); }
    .seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); }
    .seg button:focus-visible { outline: none; box-shadow: inset 0 0 0 2px var(--accent-soft); }

    /* ── Tabs ────────────────────────────────────────────────── */
    .tabs { display: flex; gap: 6px; flex-wrap: wrap; border-bottom: 1px solid var(--rule); padding-bottom: 8px; }
    .tab {
      appearance: none; border: 1px solid var(--rule); background: var(--surface); color: var(--text-2);
      border-radius: 999px; padding: 7px 12px; font: inherit; font-size: 13px; font-weight: 600; cursor: pointer;
    }
    .tab:hover { background: var(--surface-2); color: var(--text); }
    .tab[aria-selected="true"] { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
    .tab-panel { display: none; }
    .tab-panel.active { display: grid; }
    .linklike { appearance: none; border: 0; background: transparent; padding: 0; color: inherit; font: inherit; cursor: pointer; }
    .linklike:hover .pill { border-color: var(--accent); color: var(--accent); }

    /* ── States: locked, loading, empty, error ───────────────── */
    .locked {
      display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px;
      padding: 64px 24px; text-align: center; border: 1px dashed var(--rule-strong); border-radius: var(--radius); background: var(--surface);
    }
    .locked .lockglyph { width: 38px; height: 38px; border-radius: 50%; background: var(--surface-2); border: 1px solid var(--rule); display: grid; place-items: center; color: var(--muted); font-size: 18px; }
    .locked h2 { font-size: 16px; }
    .locked p { color: var(--muted); max-width: 46ch; }
    .inline-err {
      font-size: 12px; color: var(--bad); background: var(--bad-soft);
      border: 1px solid color-mix(in oklab, var(--bad) 35%, var(--rule));
      border-radius: var(--radius-sm); padding: 7px 10px; max-width: 80ch;
    }
    .skeleton { color: transparent; background: linear-gradient(90deg, var(--surface-2), var(--rule), var(--surface-2)); background-size: 200% 100%; animation: shimmer 1.3s ease-in-out infinite; border-radius: 4px; }
    @media (prefers-reduced-motion: reduce) { .skeleton { animation: none; background: var(--surface-2); } }
    @keyframes shimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }
    .empty td { text-align: center; color: var(--muted); padding: 28px 12px; font-size: 13px; }
    .hidden { display: none !important; }
    .two-col { display: grid; grid-template-columns: 1.4fr 1fr; gap: 28px; align-items: start; }
    @media (max-width: 980px) { .two-col { grid-template-columns: 1fr; } }
    code.kv { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; background: var(--surface-2); border: 1px solid var(--rule); border-radius: var(--radius-sm); padding: 1px 5px; }

    /* ── Model detail panel ──────────────────────────────────── */
    .detail {
      border: 1px solid var(--rule); border-radius: var(--radius); background: var(--surface);
      display: grid; gap: 16px; padding: 16px;
    }
    .detail .detail-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
    .detail .detail-head h3 { font-size: 16px; font-weight: 700; letter-spacing: -0.015em; }
    .detail .detail-head .close { appearance: none; border: 0; background: transparent; color: var(--muted); cursor: pointer; font-size: 18px; line-height: 1; padding: 2px 6px; border-radius: var(--radius-sm); }
    .detail .detail-head .close:hover { color: var(--text); background: var(--surface-2); }
    .kv-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px 18px; }
    .kv-item { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
    .kv-item .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
    .kv-item .v { font-size: 13px; font-variant-numeric: tabular-nums; word-break: break-all; }
    .detail .subhead { font-size: 12px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); font-weight: 600; margin-top: 2px; }
    .cost-callout { display: flex; align-items: baseline; gap: 10px; padding: 12px 14px; background: var(--accent-soft); border: 1px solid color-mix(in oklab, var(--accent) 30%, var(--rule)); border-radius: var(--radius); }
    .cost-callout .cost-value { font-size: 24px; font-weight: 700; letter-spacing: -0.02em; color: var(--accent); font-variant-numeric: tabular-nums; }
    .cost-callout .cost-label { font-size: 12px; color: var(--text-2); }
    .clickable { cursor: pointer; }
    .clickable:hover { color: var(--accent); }
    tbody tr.clickable-row { cursor: pointer; }
    tbody tr.clickable-row:hover { background: var(--accent-soft); }
    .bymodel-clickable { cursor: pointer; }
    .bymodel-clickable:hover td { background: var(--accent-soft); color: var(--text); }
    /* Visible affordance chevron on drill-down rows. */
    td.chev { width: 22px; color: var(--muted); text-align: right; padding-right: 14px; font-size: 12px; }
    tr.clickable-row:hover td.chev, tr.bymodel-clickable:hover td.chev { color: var(--accent); }
    /* Filter box */
    .filterbar { display: flex; align-items: center; gap: 8px; }
    .filterbar input { width: 200px; }
    tr.filtered-out { display: none; }
    .loading-bar { font-size: 12px; color: var(--muted); padding: 8px 0; }

    /* ── Detail drawer (right-side overlay) ───────────────── */
    .drawer-scrim { position: fixed; inset: 0; z-index: 40; background: color-mix(in oklab, oklch(20% 0.02 250) 42%, transparent); }
    .drawer {
      position: fixed; top: 0; right: 0; bottom: 0; z-index: 41;
      width: min(640px, 94vw); overflow-y: auto; overscroll-behavior: contain;
      background: var(--surface); border-left: 1px solid var(--rule);
      box-shadow: -12px 0 32px color-mix(in oklab, oklch(20% 0.02 250) 18%, transparent);
    }
    .drawer .drawer-body { border: 0; border-radius: 0; padding: 0 22px 28px; min-width: 0; }
    .drawer .drawer-body > * { min-width: 0; }
    .drawer .drawer-body .kv-grid, .drawer .drawer-body .strip { grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); }
    .drawer .detail-head {
      position: sticky; top: 0; z-index: 1; background: var(--surface);
      padding: 18px 22px 10px; margin: 0 -22px; border-bottom: 1px solid var(--rule);
    }
    body.drawer-open { overflow: hidden; }
    @media (prefers-reduced-motion: no-preference) {
      .drawer:not([hidden]) { animation: drawer-in 0.18s ease-out; }
      .drawer-scrim:not([hidden]) { animation: scrim-in 0.18s ease-out; }
    }
    @keyframes drawer-in { from { transform: translateX(28px); opacity: 0.4; } to { transform: none; opacity: 1; } }
    @keyframes scrim-in { from { opacity: 0; } to { opacity: 1; } }
    .preset-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .preset-card { border: 1px solid var(--rule); border-radius: var(--radius); background: var(--surface); padding: 13px 14px; display: grid; gap: 8px; }
    .preset-card h3 { font-size: 14px; }
    .small { font-size: 12px; color: var(--muted); }

    /* ── Read-only mode (writes disabled) ─────────────────────── */
    body[data-writes="false"] .formset { display: none; }
    body[data-writes="false"] [data-mgmt] { display: none; }
    body[data-writes="false"] .mgmt-hint { display: block; }
    .mgmt-hint { display: none; font-size: 12px; color: var(--muted); padding: 2px 0; }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <h1>Model Gateway</h1>
      <span class="sub" id="brandSub">Router health, models, usage</span>
    </div>
    <div class="spacer"></div>
    <div class="keybar">
      <div class="field">
        <input id="adminKey" type="password" placeholder="Admin key" autocomplete="off" aria-label="Admin key" />
        <button class="showtoggle" id="keyToggle" type="button" aria-label="Show admin key">show</button>
      </div>
      <button class="btn" id="unlockBtn" type="button">Unlock</button>
      <button class="iconbtn" id="refreshBtn" type="button" title="Refresh (R)" aria-label="Refresh">↻</button>
    </div>
  </header>

  <main id="main">
    <!-- Locked state shown until an admin key is supplied. -->
    <div id="lockedView" class="locked">
      <div class="lockglyph" aria-hidden="true">⚿</div>
      <h2>Enter an admin key to load the dashboard</h2>
      <p>Provider status, model inventory, usage, and recent requests live behind admin-authenticated endpoints. Paste an admin key above and choose Unlock.</p>
      <div id="lockedErr" class="inline-err hidden"></div>
    </div>

    <!-- Dashboard (hidden until unlocked). -->
    <div id="dash" class="hidden" style="display:grid; gap:28px;">
      <section>
        <div class="sec-head"><h2>Health</h2><span class="meta" id="healthMeta">—</span></div>
        <div class="strip" id="statusStrip">
          <div class="stat"><span class="label">Service</span><span class="value" id="sService">—</span><span class="detail" id="sServiceDetail">—</span></div>
          <div class="stat"><span class="label">Uptime</span><span class="value" id="sUptime">—</span><span class="detail">since last restart</span></div>
          <div class="stat"><span class="label">Providers</span><span class="value" id="sProviders">—</span><span class="detail" id="sProvidersDetail">—</span></div>
          <div class="stat"><span class="label">Models</span><span class="value" id="sModels">—</span><span class="detail" id="sModelsDetail">routable</span></div>
          <div class="stat"><span class="label">Config</span><span class="value" id="sConfig">—</span><span class="detail" id="sConfigDetail">—</span></div>
        </div>
      </section>

      <nav class="tabs" role="tablist" aria-label="Admin sections">
        <button class="tab" role="tab" type="button" data-tab="presets" aria-selected="true">Presets</button>
        <button class="tab" role="tab" type="button" data-tab="models" aria-selected="false">Models</button>
        <button class="tab" role="tab" type="button" data-tab="providers" aria-selected="false">Providers</button>
        <button class="tab" role="tab" type="button" data-tab="usage" aria-selected="false">Usage</button>
        <button class="tab" role="tab" type="button" data-tab="debug" aria-selected="false">Debug</button>
      </nav>

      <section class="tab-panel" data-tab-panel="providers" role="tabpanel">
        <div class="sec-head"><h2>Providers</h2>
          <div class="filterbar"><div class="field"><input id="providerFilter" type="search" placeholder="Filter providers" aria-label="Filter providers" /></div><span class="meta">config.yaml readiness</span></div>
        </div>
        <div class="scroll">
          <table id="providers"><thead><tr><th>ID</th><th class="num">Models</th><th>Protocol</th><th>Base URL</th><th>Key</th><th>State</th><th>Issues</th><th></th><th class="chev"></th></tr></thead><tbody></tbody></table>
        </div>
        <details class="formset"><summary>Add or edit a provider</summary>
          <div class="form-body">
            <div class="form-grid">
              <div class="control"><label>Provider id</label><input id="pId" type="text" placeholder="anthropic" /></div>
              <div class="control"><label>Base URL</label><input id="pBaseUrl" type="text" placeholder="https://api.anthropic.com" /></div>
              <div class="control"><label>Protocol</label><input id="pProtocol" type="text" placeholder="openai | anthropic" /></div>
              <div class="control"><label>API key (write-only)</label><input id="pApiKey" type="password" placeholder="leave blank to keep" /></div>
            </div>
            <div class="toolbar">
              <button class="btn" id="saveProviderBtn" type="button">Save provider</button>
              <button class="btn secondary" id="validateProviderBtn" type="button">Validate connection</button>
              <button class="btn secondary" id="discoverBtn" type="button">Discover models</button>
              <button class="btn danger" id="deleteProviderBtn" type="button">Delete</button>
              <span id="pMsg" class="msg"></span>
            </div>
            <div id="discoverPanel" class="hidden">
              <div class="subhead">Upstream models <span class="meta" id="discoverMeta"></span></div>
              <div class="scroll" style="max-height:260px;"><table id="discoverTable"><thead><tr><th>Upstream id</th><th>Status</th><th></th></tr></thead><tbody></tbody></table></div>
              <p class="hint">Register pre-fills the model form below with this provider + upstream id; review pricing/limits before saving.</p>
            </div>
            <p class="hint">The api_key is write-only: leave blank to keep the existing key. Provider config lives in the gitignored config.yaml and persists across deploys.</p>
          </div>
        </details>
        <p class="mgmt-hint">Provider management is read-only. Set <code>MODEL_GATEWAY_ADMIN_WRITES=true</code> to add or edit providers.</p>
      </section>

      <section class="tab-panel" data-tab-panel="models" role="tabpanel">
        <div class="sec-head"><h2>Models</h2>
          <div class="filterbar"><div class="field"><input id="modelFilter" type="search" placeholder="Filter models" aria-label="Filter models" /></div><span class="meta" id="modelsMeta">grouped local / cloud</span></div>
        </div>
        <div class="scroll">
          <table id="models"><thead><tr><th>Name</th><th>Upstream id</th><th>Provider</th><th>Pricing</th><th class="num">Context</th><th class="num">Max out</th><th>Thinking</th><th>Vision</th><th>State</th><th></th><th class="chev"></th></tr></thead><tbody></tbody></table>
        </div>
        <details class="formset"><summary>Add or edit a model</summary>
          <div class="form-body">
            <div class="form-grid">
              <div class="control"><label>Name (gateway id)</label><input id="mName" type="text" placeholder="claude-sonnet-4.5" /></div>
              <div class="control"><label>Provider</label><input id="mProvider" type="text" placeholder="anthropic" /></div>
              <div class="control"><label>Provider model id</label><input id="mPmid" type="text" placeholder="claude-sonnet-4.5-20250929" /></div>
              <div class="control"><label>Alias</label><input id="mAlias" type="text" placeholder="optional" /></div>
              <div class="control"><label>Context</label><input id="mContext" type="number" placeholder="tokens" /></div>
              <div class="control"><label>Max output tokens</label><input id="mMaxOut" type="number" placeholder="tokens" /></div>
              <div class="control"><label>Thinking</label><input id="mThinking" type="text" placeholder="| optional | always" /></div>
              <div class="control"><label>Thinking levels</label><input id="mThinkingLevels" type="text" placeholder="off, low, medium, high" /></div>
              <div class="control"><label>Thinking format</label><input id="mThinkingFmt" type="text" placeholder="glm-chat-template" /></div>
              <div class="control"><label>Pricing status</label><select id="mPricingStatus"><option value="metered">metered</option><option value="unmetered">unmetered local</option><option value="unknown">unknown</option></select></div>
              <div class="control"><label>Pricing JSON ($/Mtok)</label><input id="mPricing" class="ctl" type="text" placeholder='{"input":3,"output":15}' /></div>
              <div class="control" style="grid-column: 1 / -1;"><label>Description (no $ prices)</label><input id="mDesc" class="ctl" type="text" placeholder="short note" /></div>
            </div>
            <div class="checkrow">
              <label class="check"><input id="mVision" type="checkbox" /> vision</label>
              <label class="check"><input id="mEnabled" type="checkbox" checked /> enabled</label>
            </div>
            <div class="toolbar">
              <button class="btn secondary" id="previewModelBtn" type="button">Preview routing</button>
              <button class="btn" id="saveModelBtn" type="button">Save model</button>
              <button class="btn danger" id="deleteModelBtn" type="button">Delete</button>
              <span id="mMsg" class="msg"></span>
            </div>
            <div id="previewPanel" class="hidden"></div>
            <p class="hint">Writes are hot (immediate) and mirrored to the configured machine-local catalog copy. Back up reviewed catalog changes through the operator's private configuration workflow. Disabled models are hidden from /v1/models.</p>
          </div>
        </details>
        <p class="mgmt-hint">Model management is read-only. Set <code>MODEL_GATEWAY_ADMIN_WRITES=true</code> to add or edit models.</p>
      </section>

      <section class="tab-panel active" data-tab-panel="presets" role="tabpanel">
        <div class="sec-head"><h2>Presets</h2><span class="meta" id="presetsMeta">read-only model aggregates</span></div>
        <div id="presetSummary" class="preset-summary"></div>
        <div class="scroll">
          <table id="presets"><thead><tr><th>Tier</th><th>Scope</th><th>Text model</th><th>Vision model</th><th>Policy / residency</th><th>Memory</th><th>Description</th></tr></thead><tbody></tbody></table>
        </div>
        <p class="hint">Presets are gateway-owned aggregates. This page is read-only for now; consumers should choose these aggregate IDs instead of duplicating text/vision pairing logic.</p>
      </section>

      <section class="tab-panel" data-tab-panel="usage" role="tabpanel">
        <div class="sec-head"><h2>Usage &amp; cost</h2>
          <div class="toolbar">
            <div class="seg" id="winSeg" role="group" aria-label="Time window">
              <button data-w="1h" type="button">1h</button>
              <button data-w="24h" type="button" aria-pressed="true">24h</button>
              <button data-w="7d" type="button">7d</button>
              <button data-w="30d" type="button">30d</button>
              <button data-w="" type="button">All</button>
            </div>
            <span id="usageRange" class="meta"></span>
          </div>
        </div>
        <div class="strip" id="usageStrip">
          <div class="stat"><span class="label">Requests</span><span class="value" id="uRequests">—</span><span class="detail" id="uRequestsDetail">ok / errors</span></div>
          <div class="stat"><span class="label">Tokens</span><span class="value" id="uTokens">—</span><span class="detail" id="uTokensDetail">in / out (cached)</span></div>
          <div class="stat"><span class="label">Est. cost</span><span class="value" id="uCost">—</span><span class="detail" id="uCostDetail">cost coverage —</span></div>
          <div class="stat"><span class="label">Avg latency</span><span class="value" id="uLatency">—</span><span class="detail" id="uLatencyDetail">over window</span></div>
        </div>
        <div class="scroll"><table id="usageByModel"><thead><tr><th>Model</th><th class="num">Requests</th><th class="num">Usage</th><th class="num">Costed</th><th class="num">Errors</th><th class="num">In tok</th><th class="num">Out tok</th><th class="num">Cached</th><th class="num">Cost</th><th class="num">Avg ms</th></tr></thead><tbody></tbody></table></div>
        <div class="sec-head" style="margin-top:4px;"><h2>Recent requests</h2><span class="meta">last 50 · click a row for details</span></div>
        <div class="scroll"><table id="recentReq"><thead><tr><th>Time</th><th>Endpoint</th><th>Model</th><th class="num">Status</th><th>Stream</th><th>Coverage</th><th class="num">In</th><th class="num">Out</th><th class="num">Cached</th><th class="num">Cost</th><th class="num">Latency</th><th class="chev"></th></tr></thead><tbody></tbody></table></div>
        <div id="usageErr" class="inline-err hidden"></div>
      </section>

      <section class="tab-panel" data-tab-panel="debug" role="tabpanel">
        <div class="sec-head"><h2>Debug</h2><span class="meta">read-only probes</span></div>
        <p class="hint">Reasoning matrix: <a id="thinkingLink" href="/v1/debug/thinking">/v1/debug/thinking</a>. If client auth is enabled, open it with a key-capable HTTP client.</p>
        <div id="dashErr" class="inline-err hidden"></div>
      </section>
    </div>
  </main>
  <div id="drawerScrim" class="drawer-scrim" hidden></div>
  <aside id="drawer" class="drawer" role="dialog" aria-modal="true" aria-label="Details" tabindex="-1" hidden>
    <div id="drawerBody" class="detail drawer-body" aria-live="polite"></div>
  </aside>
<script>
(function () {
  const STORAGE_KEY = 'mg-admin-key';
  const keyInput = document.getElementById('adminKey');
  const keyToggle = document.getElementById('keyToggle');
  const unlockBtn = document.getElementById('unlockBtn');
  const refreshBtn = document.getElementById('refreshBtn');
  const lockedView = document.getElementById('lockedView');
  const lockedErr = document.getElementById('lockedErr');
  const dash = document.getElementById('dash');
  const dashErr = document.getElementById('dashErr');
  const usageErr = document.getElementById('usageErr');

  const BASE_PATH = (() => {
    const marker = '/admin';
    const path = window.location.pathname;
    const idx = path.indexOf(marker);
    return idx > 0 ? path.slice(0, idx).replace(/\/$/, '') : '';
  })();
  function api(path){ return BASE_PATH + path; }
  document.getElementById('thinkingLink').href = api('/v1/debug/thinking');

  let unlocked = false;
  let currentWindow = '24h';
  let currentTab = 'presets';
  let _modelsCache = [];  // deduped model rows, for resolving ledger dims to model names

  function headers(){ const key = keyInput.value.trim(); return key ? {'Authorization':'Bearer '+key} : {}; }
  function text(v){ return (v === undefined || v === null || v === '') ? '—' : String(v); }
  function escapeHtml(value){ return text(value).replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch])); }
  function fmtNum(v){ return (v===null||v===undefined) ? '—' : Number(v).toLocaleString(); }
  function fmtCost(v){ return (v===null||v===undefined) ? '—' : '$'+Number(v).toFixed(4); }
  function fmtMs(v){ return (v===null||v===undefined) ? '—' : Math.round(v)+'ms'; }
  function fmtUptime(s){ if (s===undefined||s===null) return '—'; s = Math.round(s); const d=Math.floor(s/86400), h=Math.floor((s%86400)/3600), m=Math.floor((s%3600)/60); if (d) return d+'d '+h+'h'; if (h) return h+'h '+m+'m'; return m+'m'; }
  function statePill(kind, label){ return '<span class="pill '+kind+'"><span class="dot"></span>'+escapeHtml(label)+'</span>'; }
  function requestCoverage(r){
    if (r.error) return statePill('bad','stream error');
    const usage = r.usage_reported ? statePill('ok','usage') : statePill('warn','no usage');
    let cost = statePill('warn','cost unknown');
    if (r.cost_usd !== null && r.cost_usd !== undefined) cost = r.pricing_complete ? statePill('ok','costed') : statePill('warn','partial');
    return usage+' '+cost;
  }
  function idPill(v){ return '<span class="pill id">'+escapeHtml(v)+'</span>'; }

  async function get(path){
    const res = await fetch(api(path), {headers: headers()});
    if (res.status === 401) throw new AuthError();
    if (!res.ok) throw new Error(path + ' → HTTP ' + res.status + ': ' + (await res.text()).slice(0,200));
    return res.json();
  }
  class AuthError extends Error { constructor(){ super('admin key required or invalid'); this.name='AuthError'; } }

  function showLocked(message){
    unlocked = false;
    dash.classList.add('hidden');
    lockedView.classList.remove('hidden');
    unlockBtn.textContent = 'Unlock';
    if (message) { lockedErr.textContent = message; lockedErr.classList.remove('hidden'); } else { lockedErr.classList.add('hidden'); }
    keyInput.focus();
  }
  function showDash(){
    unlocked = true;
    lockedView.classList.add('hidden');
    dash.classList.remove('hidden');
    unlockBtn.textContent = 'Locked in';
  }

  function handleFatal(e){
    if (e instanceof AuthError) showLocked('That key was rejected, or no key was sent. Paste a valid admin key and choose Unlock.');
    else { dashErr.textContent = e.message; dashErr.classList.remove('hidden'); }
  }

  // ── Loaders ─────────────────────────────────────────────────
  async function loadHealth(){
    dashErr.classList.add('hidden');
    try {
      const [status, providers, models, presets, validation] = await Promise.all([
        get('/admin/api/status'), get('/admin/api/providers'), get('/admin/api/models'), get('/admin/api/presets'), get('/admin/api/config/validation')
      ]);
      renderHealth(status, providers.providers||[], models.models||[], validation);
      renderProviders(providers.providers||[]);
      renderModels(models.models||[]);
      renderPresets(presets);
    } catch (e) { handleFatal(e); throw e; }
  }

  function renderHealth(status, providerRows, modelRows, validation){
    document.body.setAttribute('data-writes', String(status.writes_enabled === true));
    document.getElementById('sService').textContent = status.status || '—';
    document.getElementById('sService').className = 'value ' + (status.status === 'ok' ? 'ok-c' : 'bad-c');
    document.getElementById('sServiceDetail').textContent = 'pid ' + status.pid + ' · admin auth ' + (status.auth.admin_auth_enabled ? 'on' : 'off') + ' · client auth ' + (status.auth.client_auth_enabled ? 'on' : 'off');
    document.getElementById('sUptime').textContent = fmtUptime(status.uptime_seconds);
    const bad = providerRows.filter(p => (p.issues||[]).length);
    document.getElementById('sProviders').textContent = providerRows.length;
    document.getElementById('sProvidersDetail').textContent = bad.length ? bad.length + ' need config' : 'all configured';
    document.getElementById('sProviders').className = 'value ' + (bad.length ? 'warn-c' : 'ok-c');
    const uniq = dedupeModels(modelRows);
    const enabled = uniq.filter(m => m.enabled !== false).length;
    document.getElementById('sModels').textContent = enabled + '/' + uniq.length;
    document.getElementById('sModelsDetail').textContent = 'enabled / total';
    document.getElementById('sConfig').textContent = validation.ok ? 'ready' : 'issues';
    document.getElementById('sConfig').className = 'value ' + (validation.ok ? 'ok-c' : 'bad-c');
    document.getElementById('sConfigDetail').textContent = validation.ok ? 'no missing credentials' : (validation.issues||[]).length + ' provider(s)';
    document.getElementById('healthMeta').textContent = 'updated ' + new Date().toLocaleTimeString();
  }

  function renderProviders(rows){
    document.querySelector('#providers tbody').innerHTML = rows.map(p => {
      const ready = !(p.issues||[]).length;
      const issues = (p.issues||[]).map(i => idPill(i)).join(' ') || '<span class="muted">—</span>';
      const state = ready ? statePill('ok','ready') : statePill('warn','config');
      const keyCell = p.has_api_key ? '<span class="ok-c">present</span>' : '<span class="bad-c">missing</span>';
      return '<tr class="clickable-row" data-open-provider="'+escapeHtml(p.id)+'"><td>'+idPill(p.id)+'</td><td class="num">'+escapeHtml(p.enabled_models)+'</td><td>'+escapeHtml(p.protocol)+'</td><td class="id">'+escapeHtml(p.base_url)+'</td><td>'+keyCell+'</td><td>'+state+'</td><td>'+issues+'</td><td><button class="btn secondary" data-mgmt data-edit-provider="'+escapeHtml(p.id)+'">edit</button></td><td class="chev">›</td></tr>';
    }).join('') || '<tr class="empty"><td colspan="9">No providers configured.</td></tr>';
    applyFilter('providerFilter', '#providers');
  }

  function dedupeModels(rows){
    const seen = new Set(); const out = [];
    for (const m of rows) { const n = m.name || m.id; if (n && !seen.has(n)) { seen.add(n); out.push(m); } }
    return out;
  }

  function renderModels(rows){
    const uniq = dedupeModels(rows);
    _modelsCache = uniq;
    const local = uniq.filter(m => String(m.provider).toLowerCase() === 'omlx');
    const cloud = uniq.filter(m => String(m.provider).toLowerCase() !== 'omlx');
    const body = document.querySelector('#models tbody');
    const parts = [];
    function group(label, items){
      if (!items.length) return;
      parts.push('<tr class="group-row"><td colspan="11">'+escapeHtml(label)+' · '+items.length+'</td></tr>');
      for (const m of items) {
        const en = m.enabled !== false;
        const nm = escapeHtml(m.name || m.id);
        const up = escapeHtml(m.provider_model_id || m.omlx_id || m.name || '');
        const th = escapeHtml(m.thinking || m.thinking_format || '');
        const state = en ? statePill('ok','on') : statePill('bad','off');
        const priceState = m.pricing_status || (m.pricing ? 'metered' : 'unknown');
        const price = statePill(priceState === 'unknown' ? 'warn' : 'ok', priceState);
        const toggle = '<button class="btn secondary" data-mgmt data-toggle-model="'+nm+'" data-enable="'+(!en)+'">'+(en?'disable':'enable')+'</button> <button class="btn secondary" data-mgmt data-edit-model="'+nm+'">edit</button>';
        parts.push('<tr class="clickable-row" data-open-model="'+nm+'"><td>'+idPill(m.name || m.id)+'</td><td class="id">'+up+'</td><td>'+escapeHtml(m.provider)+'</td><td>'+price+'</td><td class="num">'+escapeHtml(m.context)+'</td><td class="num">'+escapeHtml(m.max_output_tokens)+'</td><td>'+th+'</td><td>'+(m.vision?'yes':'no')+'</td><td>'+state+'</td><td>'+toggle+'</td><td class="chev">›</td></tr>');
      }
    }
    group('Local (oMLX)', local);
    group('Cloud', cloud);
    body.innerHTML = parts.join('') || '<tr class="empty"><td colspan="11">No models configured.</td></tr>';
    document.getElementById('modelsMeta').textContent = local.length + ' local · ' + cloud.length + ' cloud';
    applyFilter('modelFilter', '#models');
  }

  function resolveModelName(dim){
    if (!dim) return null;
    for (const m of _modelsCache) {
      const name = m.name || m.id;
      if (!name) continue;
      if (dim === name || dim === m.alias || dim === m.provider_model_id || dim === m.omlx_id || (m.routable_ids || []).includes(dim)) return name;
    }
    return null;
  }

  function modelExists(name){ return !!resolveModelName(name); }
  function modelLink(name){
    if (!name) return '<span class="muted">—</span>';
    const exists = modelExists(name);
    const label = exists ? idPill(name) : '<span class="pill warn"><span class="dot"></span>'+escapeHtml(name)+' missing</span>';
    return exists ? '<button class="linklike" type="button" data-open-model="'+escapeHtml(name)+'">'+label+'</button>' : label;
  }
  function compactList(values){
    const arr = (values || []).filter(Boolean);
    return arr.length ? arr.map(v => idPill(v)).join(' ') : '<span class="muted">—</span>';
  }
  function renderPresets(data){
    data = data || {};
    const auto = data.auto_models || {};
    const mp = data.model_presets || {};
    const presets = mp.presets || {};
    const defaultScope = mp.default_scope || auto.default_scope || '—';
    const defaultTier = mp.default_tier || auto.default_tier || '—';
    document.getElementById('presetsMeta').textContent = 'default '+defaultScope+' / '+defaultTier+' · version '+(mp.version || '—');

    const autoCards = [];
    for (const scope of ['cloud','local']) {
      const cfg = auto[scope] || {};
      if (!cfg.model && !cfg.vision_model) continue;
      autoCards.push('<div class="preset-card"><h3>'+escapeHtml((cfg.label || scope)+' · '+scope)+'</h3>'+
        '<div class="small">auto_models default pair</div>'+
        '<div>Text '+modelLink(cfg.model)+'</div>'+
        '<div>Vision '+modelLink(cfg.vision_model)+'</div>'+
        '<p class="small">'+escapeHtml(cfg.description || '—')+'</p></div>');
    }
    document.getElementById('presetSummary').innerHTML = autoCards.join('') || '<div class="preset-card"><h3>No auto models</h3><p class="small">No auto_models block found.</p></div>';

    const rows = [];
    const tierNames = Object.keys(presets);
    for (const tier of tierNames) {
      const preset = presets[tier] || {};
      for (const scope of ['cloud','local']) {
        const cfg = preset[scope] || {};
        if (!cfg.text_model && !cfg.vision_model) continue;
        const policy = cfg.source_policy || [cfg.residency_mode, cfg.residency_group].filter(Boolean).join(' · ') || '—';
        const mem = [
          cfg.designed_memory_gb != null ? 'designed '+cfg.designed_memory_gb+'GB' : '',
          cfg.text_memory_gb != null ? 'text '+cfg.text_memory_gb+'GB' : '',
          cfg.vision_memory_gb != null ? 'vision '+cfg.vision_memory_gb+'GB' : '',
          cfg.keep_hot ? 'keep hot' : '',
        ].filter(Boolean).join(' · ') || '—';
        const extras = cfg.allow_parallel_resident ? '<div class="small">parallel '+compactList(cfg.allow_parallel_resident)+'</div>' : '';
        rows.push('<tr><td>'+idPill(tier)+'<div class="small">'+escapeHtml(preset.label || '')+'</div></td>'+
          '<td>'+escapeHtml(scope)+'</td><td>'+modelLink(cfg.text_model)+'</td><td>'+modelLink(cfg.vision_model)+'</td>'+
          '<td>'+escapeHtml(policy)+extras+'</td><td>'+escapeHtml(mem)+'</td><td>'+escapeHtml(cfg.description || preset.intent || '—')+'</td></tr>');
      }
    }
    document.querySelector('#presets tbody').innerHTML = rows.join('') || '<tr class="empty"><td colspan="7">No presets configured.</td></tr>';
  }

  const TABS = ['presets','models','providers','usage','debug'];
  function showTab(tab, opts){
    currentTab = TABS.includes(tab) ? tab : 'presets';
    for (const b of document.querySelectorAll('[data-tab]')) b.setAttribute('aria-selected', String(b.dataset.tab === currentTab));
    for (const p of document.querySelectorAll('[data-tab-panel]')) p.classList.toggle('active', p.dataset.tabPanel === currentTab);
    if (!opts || !opts.skipHash) setHash(currentTab);
  }

  // ── Hash routing: #tab or #tab/detailName ───────────────────
  function setHash(tab, detail){
    const h = '#' + tab + (detail ? '/' + encodeURIComponent(detail) : '');
    if (window.location.hash === h) return;
    // pushState (not location.hash=) so the browser never anchor-scrolls
    // to elements whose ids match tab names (e.g. <table id="models">).
    history.pushState(null, '', h);
  }
  function applyHash(){
    const raw = (window.location.hash || '').replace(/^#/, '');
    if (!raw) return;
    const [tab, detail] = raw.split('/');
    showTab(tab, {skipHash: true});
    if (!unlocked) return;
    const name = detail ? decodeURIComponent(detail) : '';
    if (tab === 'models' && name) showModelDetail(name, {skipHash: true});
    else if (tab === 'providers' && name) showProviderDetail(name, {skipHash: true});
    else if (tab === 'usage' && name) showRequestDetail(name, {skipHash: true});
    else closeDrawer({skipHash: true});
  }
  window.addEventListener('hashchange', applyHash);

  // ── Table filter ───────────────────────────────────────
  function applyFilter(inputId, tableSel){
    const q = (document.getElementById(inputId).value || '').trim().toLowerCase();
    for (const tr of document.querySelectorAll(tableSel + ' tbody tr')) {
      if (tr.classList.contains('empty')) continue;
      if (tr.classList.contains('group-row')) { tr.classList.toggle('filtered-out', !!q); continue; }
      tr.classList.toggle('filtered-out', !!q && !tr.textContent.toLowerCase().includes(q));
    }
  }
  document.getElementById('modelFilter').addEventListener('input', () => applyFilter('modelFilter', '#models'));
  document.getElementById('providerFilter').addEventListener('input', () => applyFilter('providerFilter', '#providers'));

  // ── Usage ───────────────────────────────────────────────────
  const _winLabel = {'1h':'Last hour','24h':'Last 24 hours','7d':'Last 7 days','30d':'Last 30 days','':'All time'};
  async function loadUsage(window){
    currentWindow = window;
    for (const b of document.querySelectorAll('#winSeg button')) b.setAttribute('aria-pressed', String(b.dataset.w === window));
    document.getElementById('usageRange').textContent = _winLabel[window] || window || 'All';
    usageErr.classList.add('hidden');
    try {
      const q = window ? ('?window='+encodeURIComponent(window)) : '';
      const [usage, recent] = await Promise.all([ get('/admin/api/usage'+q), get('/admin/api/requests?limit=50') ]);
      const s = usage.summary || {};
      const ok = s.ok || 0, errs = s.errors || 0;
      const uReq = document.getElementById('uRequests');
      uReq.textContent = fmtNum(s.requests); uReq.className = 'value ' + (errs ? 'warn-c' : 'ok-c');
      document.getElementById('uRequestsDetail').textContent = ok + ' ok / ' + errs + ' errors';
      const uTok = document.getElementById('uTokens');
      uTok.textContent = fmtNum((s.input_tokens||0) + (s.output_tokens||0)); uTok.className = 'value ok-c';
      document.getElementById('uTokensDetail').textContent = fmtNum(s.input_tokens)+' in / '+fmtNum(s.output_tokens)+' out · '+fmtNum(s.usage_reported_requests)+'/'+fmtNum(s.requests)+' reported';
      const uCost = document.getElementById('uCost');
      uCost.textContent = fmtCost(s.cost_usd); uCost.className = 'value ' + ((s.unknown_cost_requests||0) ? 'warn-c' : 'ok-c');
      document.getElementById('uCostDetail').textContent = fmtNum(s.known_cost_requests)+'/'+fmtNum(s.requests)+' known · '+fmtNum(s.missing_pricing_requests)+' missing price';
      const uLat = document.getElementById('uLatency');
      uLat.textContent = fmtMs(s.avg_latency_ms); uLat.className = 'value ok-c';
      document.getElementById('uLatencyDetail').textContent = s.requests ? 'over '+s.requests+' requests' : 'no requests';
      const byModel = usage.by_model || [];
      document.querySelector('#usageByModel tbody').innerHTML = byModel.map(r => {
        const name = resolveModelName(r.dim);
        const cls = name ? 'bymodel-clickable' : '';
        const attr = name ? ' data-open-model="'+escapeHtml(name)+'"' : '';
        return '<tr class="'+cls+'"'+attr+'><td>'+idPill(r.dim||'—')+'</td><td class="num">'+fmtNum(r.requests)+'</td><td class="num">'+fmtNum(r.usage_reported_requests)+'</td><td class="num">'+fmtNum(r.known_cost_requests)+'</td><td class="num '+(r.errors?'bad-c':'ok-c')+'">'+fmtNum(r.errors)+'</td><td class="num">'+fmtNum(r.input_tokens)+'</td><td class="num">'+fmtNum(r.output_tokens)+'</td><td class="num">'+fmtNum(r.cached_read_tokens)+'</td><td class="num">'+fmtCost(r.cost_usd)+'</td><td class="num">'+fmtMs(r.avg_latency_ms)+'</td></tr>';
      }).join('') || '<tr class="empty"><td colspan="10">No requests in this window.</td></tr>';
      const reqs = recent.requests || [];
      document.querySelector('#recentReq tbody').innerHTML = reqs.map(r => '<tr class="clickable-row" data-open-request="'+escapeHtml(r.id||'')+'"><td>'+escapeHtml(r.ts_iso||'—')+'</td><td>'+escapeHtml(r.endpoint||'—')+'</td><td>'+(resolveModelName(r.model) ? modelLink(resolveModelName(r.model)) : idPill(r.model||'—'))+'</td><td class="num '+(!r.error && r.status && r.status < 400 ? 'ok-c' : 'bad-c')+'">'+(r.status||'—')+'</td><td>'+(r.is_stream?'stream':'sync')+'</td><td>'+requestCoverage(r)+'</td><td class="num">'+fmtNum(r.input_tokens)+'</td><td class="num">'+fmtNum(r.output_tokens)+'</td><td class="num">'+fmtNum(r.cached_read_tokens)+'</td><td class="num">'+fmtCost(r.cost_usd)+'</td><td class="num">'+fmtMs(r.latency_ms)+'</td><td class="chev">›</td></tr>').join('') || '<tr class="empty"><td colspan="12">No requests yet.</td></tr>';
    } catch (e) {
      if (e instanceof AuthError) { return; } // loadHealth drives the locked state on auth failure
      usageErr.textContent = e.message; usageErr.classList.remove('hidden');
    }
  }

  // ── CRUD ────────────────────────────────────────────────────
  async function send(method, path, body){
    const opts = {method, headers: headers()};
    if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
    const res = await fetch(api(path), opts);
    const txt = await res.text();
    let j = null; try { j = JSON.parse(txt); } catch(e) { j = {raw: txt}; }
    if (res.status === 401) throw new AuthError();
    if (!res.ok) throw new Error((j && j.error && j.error.message) || txt || ('HTTP '+res.status));
    return j;
  }
  function setMsg(id, msg, ok){ const el = document.getElementById(id); el.textContent = msg; el.className = 'msg '+(ok ? 'ok-c' : 'bad-c'); }

  async function saveProvider(){
    const id = document.getElementById('pId').value.trim().toLowerCase();
    if (!id) return setMsg('pMsg', 'provider id required', false);
    const body = {base_url: document.getElementById('pBaseUrl').value.trim()};
    const proto = document.getElementById('pProtocol').value.trim(); if (proto) body.protocol = proto;
    const key = document.getElementById('pApiKey').value; if (key) body.api_key = key;
    try { await send('POST', '/admin/api/providers/'+encodeURIComponent(id), body); setMsg('pMsg', 'saved + reloaded', true); loadHealth(); }
    catch(e){ if (e instanceof AuthError) return showLocked(); setMsg('pMsg', e.message, false); }
  }
  async function deleteProvider(){
    const id = document.getElementById('pId').value.trim().toLowerCase();
    if (!id) return setMsg('pMsg', 'provider id required', false);
    if (!confirm('Delete provider '+id+'?')) return;
    try { await send('DELETE', '/admin/api/providers/'+encodeURIComponent(id)); setMsg('pMsg', 'deleted + reloaded', true); loadHealth(); }
    catch(e){ if (e instanceof AuthError) return showLocked(); setMsg('pMsg', e.message, false); }
  }
  async function validateProvider(){
    const id = document.getElementById('pId').value.trim().toLowerCase();
    if (!id) return setMsg('pMsg', 'provider id required', false);
    setMsg('pMsg', 'validating…', true);
    try { const r = await send('POST', '/admin/api/providers/'+encodeURIComponent(id)+'/validate'); setMsg('pMsg', r.ok ? ('OK · '+r.status_code+' · '+(r.model_count??'?')+' models') : ('failed: '+(r.error||r.status_code)), r.ok); }
    catch(e){ if (e instanceof AuthError) return showLocked(); setMsg('pMsg', e.message, false); }
  }
  function editProvider(id){
    document.getElementById('pId').value = id;
    setMsg('pMsg', 'editing '+id+' — fill base_url (blank=keep) + key (blank=keep)', true);
  }
  function editModel(name){
    const m = _modelsCache.find(row => (row.name || row.id) === name);
    if (!m) return setMsg('mMsg', 'could not find '+name, false);
    document.getElementById('mName').value = name;
    document.getElementById('mProvider').value = m.configured_provider || m.provider || '';
    document.getElementById('mPmid').value = m.provider_model_id || m.omlx_id || '';
    document.getElementById('mAlias').value = m.alias || '';
    document.getElementById('mContext').value = m.context || '';
    document.getElementById('mMaxOut').value = m.max_output_tokens || '';
    document.getElementById('mThinking').value = m.thinking || '';
    document.getElementById('mThinkingLevels').value = (m.thinking_levels || []).join(', ');
    document.getElementById('mThinkingFmt').value = m.thinking_format || '';
    document.getElementById('mPricingStatus').value = m.pricing_status || (m.pricing ? 'metered' : 'unknown');
    document.getElementById('mPricing').value = m.pricing ? JSON.stringify(m.pricing) : '';
    document.getElementById('mVision').checked = m.vision === true;
    document.getElementById('mEnabled').checked = m.enabled !== false;
    setMsg('mMsg', 'editing '+name, true);
  }
  function collectModelForm(){
    const name = document.getElementById('mName').value.trim();
    if (!name) { setMsg('mMsg', 'name required', false); return null; }
    const body = {provider: document.getElementById('mProvider').value.trim(), provider_model_id: document.getElementById('mPmid').value.trim()};
    if (!body.provider || !body.provider_model_id) { setMsg('mMsg', 'provider + provider_model_id required', false); return null; }
    const alias = document.getElementById('mAlias').value.trim(); if (alias) body.alias = alias;
    const ctx = parseInt(document.getElementById('mContext').value); if (!isNaN(ctx)) body.context = ctx;
    const mo = parseInt(document.getElementById('mMaxOut').value); if (!isNaN(mo)) body.max_output_tokens = mo;
    const th = document.getElementById('mThinking').value.trim(); if (th) body.thinking = th;
    const levels = document.getElementById('mThinkingLevels').value.trim();
    if (levels) body.thinking_levels = levels.split(',').map(value => value.trim()).filter(Boolean);
    const tf = document.getElementById('mThinkingFmt').value.trim(); if (tf) body.thinking_format = tf;
    const pricingStatus = document.getElementById('mPricingStatus').value;
    body.pricing_status = pricingStatus;
    const pr = document.getElementById('mPricing').value.trim();
    if (pricingStatus === 'metered') {
      if (!pr) { setMsg('mMsg', 'metered pricing JSON required', false); return null; }
      try { body.pricing = JSON.parse(pr); } catch(e){ setMsg('mMsg', 'pricing JSON invalid', false); return null; }
    } else {
      body.pricing = null;
    }
    const desc = document.getElementById('mDesc').value.trim(); if (desc) body.desc = desc;
    body.vision = document.getElementById('mVision').checked;
    body.enabled = document.getElementById('mEnabled').checked;
    return {name, body};
  }
  async function saveModel(){
    const form = collectModelForm();
    if (!form) return;
    try { await send('POST', '/admin/api/models/'+encodeURIComponent(form.name), form.body); setMsg('mMsg', 'saved + reloaded (machine-local catalog updated)', true); document.getElementById('previewPanel').classList.add('hidden'); loadHealth(); }
    catch(e){ if (e instanceof AuthError) return showLocked(); setMsg('mMsg', e.message, false); }
  }
  async function previewModel(){
    const form = collectModelForm();
    if (!form) return;
    const panel = document.getElementById('previewPanel');
    panel.classList.remove('hidden');
    panel.innerHTML = '<div class="loading-bar">Previewing…</div>';
    try {
      const p = await send('POST', '/admin/api/models/'+encodeURIComponent(form.name)+'/preview', form.body);
      let html = '<div class="subhead">Routing preview</div>';
      const verdict = p.routable ? statePill('ok','will route') : (p.ok ? statePill('warn','valid, not routable') : statePill('bad','invalid'));
      html += '<div class="toolbar">'+verdict+'<span class="meta">'+(p.exists ? 'updates existing entry' : 'creates new entry')+'</span></div>';
      if ((p.issues||[]).length) html += '<div class="inline-err">'+p.issues.map(escapeHtml).join('<br>')+'</div>';
      if ((p.clashes||[]).length) html += '<div class="inline-err">Id clash: '+p.clashes.map(c => escapeHtml(c.id)+' → '+escapeHtml(c.model)).join(', ')+'</div>';
      html += '<div class="kv-grid">';
      html += '<div class="kv-item"><span class="k">Route</span><span class="v">'+((p.route||[]).map(r => r.usable ? statePill('ok', r.provider) : statePill('bad', r.provider+' · '+(r.reason||'unusable'))).join(' ') || '—')+'</span></div>';
      html += '<div class="kv-item"><span class="k">Routable ids</span><span class="v id">'+escapeHtml((p.routable_ids||[]).join(', ')||'—')+'</span></div>';
      html += '<div class="kv-item"><span class="k">Thinking</span><span class="v">'+escapeHtml((p.entry&&p.entry.thinking)||'off')+' · levels '+escapeHtml(((p.entry&&p.entry.thinking_levels)||[]).join(', ')||'—')+'</span></div>';
      html += '</div>';
      panel.innerHTML = html;
      setMsg('mMsg', p.routable ? 'preview ok — safe to save' : 'preview found problems', p.routable);
    } catch(e){
      if (e instanceof AuthError) return showLocked();
      panel.innerHTML = '<div class="inline-err">'+escapeHtml(e.message)+'</div>';
    }
  }
  async function discoverModels(){
    const id = document.getElementById('pId').value.trim().toLowerCase();
    if (!id) return setMsg('pMsg', 'provider id required', false);
    const panel = document.getElementById('discoverPanel');
    panel.classList.remove('hidden');
    document.getElementById('discoverMeta').textContent = 'querying…';
    document.querySelector('#discoverTable tbody').innerHTML = '';
    try {
      const r = await send('POST', '/admin/api/providers/'+encodeURIComponent(id)+'/discover');
      const models = r.models || [];
      document.getElementById('discoverMeta').textContent = r.status + ' · ' + models.length + ' models';
      document.querySelector('#discoverTable tbody').innerHTML = models.map(m =>
        '<tr><td class="id">'+escapeHtml(m.id)+'</td><td>'+(m.registered ? statePill('ok','registered') : statePill('warn','new'))+'</td>'+
        '<td>'+(m.registered ? '' : '<button class="btn secondary" data-register-upstream="'+escapeHtml(m.id)+'" data-register-provider="'+escapeHtml(id)+'">register</button>')+'</td></tr>'
      ).join('') || '<tr class="empty"><td colspan="3">No models reported ('+escapeHtml(r.status||'unknown')+').</td></tr>';
    } catch(e){
      if (e instanceof AuthError) return showLocked();
      document.getElementById('discoverMeta').textContent = '';
      document.querySelector('#discoverTable tbody').innerHTML = '<tr class="empty"><td colspan="3">'+escapeHtml(e.message)+'</td></tr>';
    }
  }
  function registerFromDiscovery(provider, upstreamId){
    // Pre-fill the model form; operator reviews limits/pricing then saves.
    const modelForm = document.querySelector('[data-tab-panel="models"] .formset');
    showTab('models');
    modelForm.open = true;
    const short = upstreamId.split('/').pop();
    document.getElementById('mName').value = short;
    document.getElementById('mProvider').value = provider;
    document.getElementById('mPmid').value = upstreamId;
    document.getElementById('mAlias').value = '';
    document.getElementById('mPricingStatus').value = 'unknown';
    document.getElementById('mPricing').value = '';
    document.getElementById('mEnabled').checked = true;
    setMsg('mMsg', 'pre-filled from discovery — set context/limits/pricing, preview, then save', true);
    modelForm.scrollIntoView({behavior:'smooth', block:'start'});
  }
  async function deleteModel(){
    const name = document.getElementById('mName').value.trim();
    if (!name) return setMsg('mMsg', 'name required', false);
    if (!confirm('Delete model '+name+'?')) return;
    try { await send('DELETE', '/admin/api/models/'+encodeURIComponent(name)); setMsg('mMsg', 'deleted + reloaded', true); loadHealth(); }
    catch(e){ if (e instanceof AuthError) return showLocked(); setMsg('mMsg', e.message, false); }
  }
  async function toggleModel(name, enable){
    const ep = enable ? 'enable' : 'disable';
    try { await send('POST', '/admin/api/models/'+encodeURIComponent(name)+'/'+ep); setMsg('mMsg', name+' '+ep+'d', true); loadHealth(); }
    catch(e){ if (e instanceof AuthError) return showLocked(); setMsg('mMsg', e.message, false); }
  }

  // ── Model detail click-through ─────────────────────────────
  const drawer = document.getElementById('drawer');
  const drawerScrim = document.getElementById('drawerScrim');
  const drawerBody = document.getElementById('drawerBody');
  let drawerKind = null;   // 'model' | 'provider' | 'request' | null
  let drawerName = null;   // current entity name/id shown in the drawer
  let _drawerReturnFocus = null;
  function openDrawer(kind, name){
    drawerKind = kind; drawerName = name;
    if (drawer.hidden) {
      _drawerReturnFocus = document.activeElement;
      drawer.hidden = false; drawerScrim.hidden = false;
      document.body.classList.add('drawer-open');
      drawer.focus({preventScroll: true});
    }
  }
  function closeDrawer(opts){
    if (drawer.hidden) return;
    drawer.hidden = true; drawerScrim.hidden = true;
    document.body.classList.remove('drawer-open');
    drawerBody.innerHTML = '';
    drawerKind = null; drawerName = null;
    if (!opts || !opts.skipHash) setHash(currentTab);
    if (_drawerReturnFocus && _drawerReturnFocus.isConnected) { try { _drawerReturnFocus.focus(); } catch(e) {} }
    _drawerReturnFocus = null;
  }
  drawerScrim.addEventListener('click', () => closeDrawer());
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !drawer.hidden) { e.preventDefault(); closeDrawer(); } });
  function openFormset(tab){
    const fs = document.querySelector('[data-tab-panel="'+tab+'"] .formset');
    if (fs) { fs.open = true; fs.scrollIntoView({behavior:'smooth', block:'start'}); }
  }
  function pricingText(p){
    if (!p) return '—';
    const parts = [];
    if (p.input != null) parts.push('in $'+Number(p.input).toFixed(2));
    if (p.output != null) parts.push('out $'+Number(p.output).toFixed(2));
    if (p.cache_read != null) parts.push('cache read $'+Number(p.cache_read).toFixed(3));
    if (p.cache_write != null) parts.push('cache write 5m $'+Number(p.cache_write).toFixed(3));
    if (p.cache_write_1h != null) parts.push('cache write 1h $'+Number(p.cache_write_1h).toFixed(3));
    if (p.reasoning != null) parts.push('reason $'+Number(p.reasoning).toFixed(3));
    return parts.length ? parts.join(' · ') : JSON.stringify(p);
  }
  async function showModelDetail(name, opts){
    if (!name) return;
    if (currentTab !== 'models') showTab('models', {skipHash: true});
    if (!opts || !opts.skipHash) setHash('models', name);
    openDrawer('model', name);
    drawerBody.innerHTML = '<div class="loading-bar">Loading '+escapeHtml(name)+'…</div>';
    try {
      const q = currentWindow ? ('?window='+encodeURIComponent(currentWindow)) : '';
      const data = await get('/admin/api/models/'+encodeURIComponent(name)+'/stats'+q);
      if (drawerKind === 'model' && drawerName === name) renderModelDetail(name, data);
    } catch (e) {
      if (e instanceof AuthError) { showLocked(); return; }
      drawerBody.innerHTML = '<div class="inline-err">Failed to load stats: '+escapeHtml(e.message)+'</div>';
    }
  }
  function renderModelDetail(name, data){
    const m = data.model || {};
    const u = data.usage || {};
    const ids = data.routable_ids || [];
    const en = m.enabled !== false;
    const reqs = fmtNum(u.requests);
    const errs = u.errors || 0;
    const cost = fmtCost(u.cost_usd);
    const winLabel = _winLabel[currentWindow] || currentWindow || 'All';
    const kv = (k, v) => '<div class="kv-item"><span class="k">'+k+'</span><span class="v">'+v+'</span></div>';
    let html = '<div class="detail-head"><h3>'+idPill(name)+'</h3>';
    html += '<div class="toolbar"><span class="meta">'+escapeHtml(m.provider||'—')+' · '+(en?statePill('ok','on'):statePill('bad','off'))+'</span>';
    html += '<button class="btn secondary" data-mgmt data-edit-model="'+escapeHtml(name)+'">Edit</button>';
    html += '<button class="close" type="button" data-close-detail aria-label="Close">×</button></div></div>';
    // Cost callout
    html += '<div class="cost-callout"><span class="cost-value">'+cost+'</span><span class="cost-label">estimated cost · '+escapeHtml(winLabel)+' · '+fmtNum(u.known_cost_requests)+'/'+fmtNum(u.requests)+' requests costed</span></div>';
    // Usage strip
    html += '<div class="strip"><div class="stat"><span class="label">Requests</span><span class="value '+(errs?'warn-c':'ok-c')+'">'+reqs+'</span><span class="detail">'+fmtNum(u.ok)+' ok / '+fmtNum(errs)+' errors</span></div>';
    html += '<div class="stat"><span class="label">Tokens</span><span class="value ok-c">'+fmtNum((u.input_tokens||0)+(u.output_tokens||0))+'</span><span class="detail">'+fmtNum(u.input_tokens)+' in / '+fmtNum(u.output_tokens)+' out · '+fmtNum(u.usage_reported_requests)+'/'+fmtNum(u.requests)+' reported</span></div>';
    html += '<div class="stat"><span class="label">Avg latency</span><span class="value ok-c">'+fmtMs(u.avg_latency_ms)+'</span><span class="detail">over '+fmtNum(u.requests)+' requests</span></div>';
    html += '<div class="stat"><span class="label">Reasoning tok</span><span class="value ok-c">'+fmtNum(u.reasoning_tokens)+'</span><span class="detail">thinking output</span></div></div>';
    // Config grid
    html += '<div><div class="subhead">Configuration</div><div class="kv-grid">';
    html += kv('Upstream id', '<span class="id">'+escapeHtml(m.provider_model_id || m.omlx_id || '—')+'</span>');
    html += kv('Provider', escapeHtml(m.provider||'—'));
    html += kv('Context', escapeHtml(m.context));
    html += kv('Max output', escapeHtml(m.max_output_tokens));
    html += kv('Thinking', escapeHtml(m.thinking || m.thinking_format || '—'));
    html += kv('Thinking levels', escapeHtml((m.thinking_levels || []).join(', ') || '—'));
    html += kv('Vision', m.vision ? 'yes' : 'no');
    html += kv('Pricing status', statePill(m.pricing_status === 'unknown' ? 'warn' : 'ok', m.pricing_status || 'unknown'));
    html += kv('Pricing ($/Mtok)', '<span class="id">'+escapeHtml(pricingText(m.pricing))+'</span>');
    html += kv('Routable ids', '<span class="id">'+escapeHtml(ids.join(', ')||'—')+'</span>');
    html += '</div></div>';
    // Recent requests for this model
    const reqs2 = data.recent || [];
    html += '<div><div class="subhead">Recent requests · this model · '+winLabel+'</div><div class="scroll"><table><thead><tr><th>Time</th><th class="num">Status</th><th>Stream</th><th>Coverage</th><th class="num">In</th><th class="num">Out</th><th class="num">Cached</th><th class="num">Cost</th><th class="num">Latency</th></tr></thead><tbody>';
    html += reqs2.map(r => '<tr><td>'+escapeHtml(r.ts_iso||'—')+'</td><td class="num '+(!r.error && r.status && r.status < 400 ? 'ok-c' : 'bad-c')+'">'+(r.status||'—')+'</td><td>'+(r.is_stream?'stream':'sync')+'</td><td>'+requestCoverage(r)+'</td><td class="num">'+fmtNum(r.input_tokens)+'</td><td class="num">'+fmtNum(r.output_tokens)+'</td><td class="num">'+fmtNum(r.cached_read_tokens)+'</td><td class="num">'+fmtCost(r.cost_usd)+'</td><td class="num">'+fmtMs(r.latency_ms)+'</td></tr>').join('') || '<tr class="empty"><td colspan="9">No requests for this model in '+escapeHtml(winLabel)+'.</td></tr>';
    html += '</tbody></table></div></div>';
    drawerBody.innerHTML = html;
  }

  // ── Provider detail click-through ─────────────────────────
  async function showProviderDetail(id, opts){
    if (!id) return;
    if (currentTab !== 'providers') showTab('providers', {skipHash: true});
    if (!opts || !opts.skipHash) setHash('providers', id);
    openDrawer('provider', id);
    drawerBody.innerHTML = '<div class="loading-bar">Loading '+escapeHtml(id)+'…</div>';
    try {
      const q = currentWindow ? ('?window='+encodeURIComponent(currentWindow)) : '';
      const data = await get('/admin/api/providers/'+encodeURIComponent(id)+'/stats'+q);
      if (drawerKind === 'provider' && drawerName === id) renderProviderDetail(id, data);
    } catch (e) {
      if (e instanceof AuthError) { showLocked(); return; }
      drawerBody.innerHTML = '<div class="inline-err">Failed to load stats: '+escapeHtml(e.message)+'</div>';
    }
  }
  function renderProviderDetail(id, data){
    const p = data.provider || {};
    const u = data.usage || {};
    const winLabel = _winLabel[currentWindow] || currentWindow || 'All';
    const kv = (k, v) => '<div class="kv-item"><span class="k">'+k+'</span><span class="v">'+v+'</span></div>';
    const errs = u.errors || 0;
    const ready = !(p.issues||[]).length;
    let html = '<div class="detail-head"><h3>'+idPill(id)+'</h3>';
    html += '<div class="toolbar"><span class="meta">'+(ready?statePill('ok','ready'):statePill('warn','config'))+'</span>';
    html += '<button class="btn secondary" data-mgmt data-edit-provider="'+escapeHtml(id)+'">Edit</button>';
    html += '<button class="close" type="button" data-close-detail aria-label="Close">×</button></div></div>';
    html += '<div class="cost-callout"><span class="cost-value">'+fmtCost(u.cost_usd)+'</span><span class="cost-label">estimated cost · '+escapeHtml(winLabel)+' · '+fmtNum(u.known_cost_requests)+'/'+fmtNum(u.requests)+' requests costed</span></div>';
    html += '<div class="strip"><div class="stat"><span class="label">Requests</span><span class="value '+(errs?'warn-c':'ok-c')+'">'+fmtNum(u.requests)+'</span><span class="detail">'+fmtNum(u.ok)+' ok / '+fmtNum(errs)+' errors</span></div>';
    html += '<div class="stat"><span class="label">Tokens</span><span class="value ok-c">'+fmtNum((u.input_tokens||0)+(u.output_tokens||0))+'</span><span class="detail">'+fmtNum(u.input_tokens)+' in / '+fmtNum(u.output_tokens)+' out</span></div>';
    html += '<div class="stat"><span class="label">Avg latency</span><span class="value ok-c">'+fmtMs(u.avg_latency_ms)+'</span><span class="detail">over '+fmtNum(u.requests)+' requests</span></div>';
    html += '<div class="stat"><span class="label">Models</span><span class="value ok-c">'+fmtNum(p.enabled_models)+'</span><span class="detail">enabled via this provider</span></div></div>';
    html += '<div><div class="subhead">Configuration</div><div class="kv-grid">';
    html += kv('Base URL', '<span class="id">'+escapeHtml(p.base_url||'—')+'</span>');
    html += kv('Protocol', escapeHtml(p.protocol||'—'));
    html += kv('API key', p.has_api_key ? '<span class="ok-c">present</span>' : '<span class="bad-c">missing</span>');
    html += kv('Issues', (p.issues||[]).map(i => idPill(i)).join(' ') || '<span class="muted">none</span>');
    html += '</div></div>';
    const reqs = data.recent || [];
    html += '<div><div class="subhead">Recent requests · this provider</div><div class="scroll"><table><thead><tr><th>Time</th><th>Model</th><th class="num">Status</th><th>Stream</th><th class="num">In</th><th class="num">Out</th><th class="num">Cost</th><th class="num">Latency</th></tr></thead><tbody>';
    html += reqs.map(r => '<tr><td>'+escapeHtml(r.ts_iso||'—')+'</td><td>'+(resolveModelName(r.model) ? modelLink(resolveModelName(r.model)) : idPill(r.model||'—'))+'</td><td class="num '+(!r.error && r.status && r.status < 400 ? 'ok-c' : 'bad-c')+'">'+(r.status||'—')+'</td><td>'+(r.is_stream?'stream':'sync')+'</td><td class="num">'+fmtNum(r.input_tokens)+'</td><td class="num">'+fmtNum(r.output_tokens)+'</td><td class="num">'+fmtCost(r.cost_usd)+'</td><td class="num">'+fmtMs(r.latency_ms)+'</td></tr>').join('') || '<tr class="empty"><td colspan="8">No requests for this provider yet.</td></tr>';
    html += '</tbody></table></div></div>';
    drawerBody.innerHTML = html;
  }

  // ── Request detail click-through ──────────────────────────
  async function showRequestDetail(id, opts){
    if (!id) return;
    if (currentTab !== 'usage') showTab('usage', {skipHash: true});
    if (!opts || !opts.skipHash) setHash('usage', id);
    openDrawer('request', id);
    drawerBody.innerHTML = '<div class="loading-bar">Loading request…</div>';
    try {
      const data = await get('/admin/api/requests/'+encodeURIComponent(id));
      if (drawerKind === 'request' && drawerName === id) renderRequestDetail(data.request || {});
    } catch (e) {
      if (e instanceof AuthError) { showLocked(); return; }
      drawerBody.innerHTML = '<div class="inline-err">Failed to load request: '+escapeHtml(e.message)+'</div>';
    }
  }
  function renderRequestDetail(r){
    const kv = (k, v) => '<div class="kv-item"><span class="k">'+k+'</span><span class="v">'+v+'</span></div>';
    const okReq = !r.error && r.status && r.status < 400;
    let html = '<div class="detail-head"><h3>'+idPill(r.id||'request')+'</h3>';
    html += '<div class="toolbar"><span class="meta">'+escapeHtml(r.ts_iso||'—')+' · '+(okReq?statePill('ok', String(r.status||'ok')):statePill('bad', String(r.status||'error')))+'</span>';
    html += '<button class="close" type="button" data-close-detail aria-label="Close">×</button></div></div>';
    if (r.error) html += '<div class="inline-err">'+escapeHtml(r.error)+'</div>';
    html += '<div><div class="subhead">Request</div><div class="kv-grid">';
    html += kv('Endpoint', '<span class="id">'+escapeHtml(r.endpoint||'—')+'</span>');
    html += kv('Model', resolveModelName(r.model) ? modelLink(resolveModelName(r.model)) : escapeHtml(r.model||'—'));
    html += kv('Provider', escapeHtml(r.provider||'—'));
    html += kv('Upstream id', '<span class="id">'+escapeHtml(r.provider_model_id||'—')+'</span>');
    html += kv('Mode', r.is_stream ? 'stream' : 'sync');
    html += kv('Status', escapeHtml(r.status));
    html += kv('Latency', fmtMs(r.latency_ms));
    html += kv('Coverage', requestCoverage(r));
    html += '</div></div>';
    html += '<div><div class="subhead">Tokens &amp; cost</div><div class="kv-grid">';
    html += kv('Input tokens', fmtNum(r.input_tokens));
    html += kv('Output tokens', fmtNum(r.output_tokens));
    html += kv('Cached read', fmtNum(r.cached_read_tokens));
    html += kv('Cache write', fmtNum(r.cache_write_tokens));
    html += kv('Reasoning tokens', fmtNum(r.reasoning_tokens));
    html += kv('Cost', fmtCost(r.cost_usd));
    html += kv('Pricing complete', r.pricing_complete ? 'yes' : 'no');
    const mpc = r.missing_pricing_classes;
    html += kv('Missing pricing', mpc && mpc.length ? escapeHtml(Array.isArray(mpc) ? mpc.join(', ') : String(mpc)) : '<span class="muted">none</span>');
    html += '</div></div>';
    drawerBody.innerHTML = html;
  }

  // ── Wiring ──────────────────────────────────────────────────
  function unlock(){
    const key = keyInput.value.trim();
    if (!key) { lockedErr.textContent = 'Enter an admin key first.'; lockedErr.classList.remove('hidden'); keyInput.focus(); return; }
    try { sessionStorage.setItem(STORAGE_KEY, key); } catch(e) {}
    lockedErr.classList.add('hidden');
    showDash();
    // Load health first so _modelsCache is populated before loadUsage renders
    // the by-model table (whose click-through resolves dims to model names).
    loadHealth().then(() => loadUsage(currentWindow)).then(() => applyHash()).catch(()=>{});
  }
  function refresh(){ if (!unlocked) return; closeDrawer(); loadHealth(); loadUsage(currentWindow); }

  unlockBtn.addEventListener('click', unlock);
  refreshBtn.addEventListener('click', refresh);
  keyInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') unlock(); });
  keyToggle.addEventListener('click', () => {
    const show = keyInput.type === 'password';
    keyInput.type = show ? 'text' : 'password';
    keyToggle.textContent = show ? 'hide' : 'show';
  });

  // Delegated clicks for table action buttons and click-throughs.
  document.addEventListener('click', (e) => {
    const t = e.target;
    if (t.closest && t.closest('[data-close-detail]')) { closeDrawer(); return; }
    const tabBtn = t.closest && t.closest('[data-tab]');
    if (tabBtn) { closeDrawer({skipHash: true}); showTab(tabBtn.dataset.tab); return; }
    // Buttons inside clickable rows act on the row's entity, not the drill-down.
    if (t.tagName === 'BUTTON') {
      const ep = t.getAttribute('data-edit-provider');
      const em = t.getAttribute('data-edit-model');
      const tm = t.getAttribute('data-toggle-model');
      const ru = t.getAttribute('data-register-upstream');
      if (ep) { closeDrawer({skipHash: true}); showTab('providers'); openFormset('providers'); editProvider(ep); return; }
      if (em) { closeDrawer({skipHash: true}); showTab('models'); openFormset('models'); editModel(em); return; }
      if (tm) { toggleModel(tm, t.getAttribute('data-enable') === 'true'); return; }
      if (ru) { registerFromDiscovery(t.getAttribute('data-register-provider'), ru); return; }
    }
    const openModel = t.closest && t.closest('[data-open-model]');
    if (openModel) { const nm = openModel.getAttribute('data-open-model'); if (nm) { showModelDetail(nm); return; } }
    const openProvider = t.closest && t.closest('[data-open-provider]');
    if (openProvider) { const id = openProvider.getAttribute('data-open-provider'); if (id) { showProviderDetail(id); return; } }
    const openRequest = t.closest && t.closest('[data-open-request]');
    if (openRequest) { const id = openRequest.getAttribute('data-open-request'); if (id) { showRequestDetail(id); return; } }
  });

  // Annotate model rows with dataset for editModel lookups after render.
  const _origRender = renderModels;
  renderModels = function(rows){
    _origRender(rows);
    const uniq = dedupeModels(rows);
    const byName = {};
    for (const m of uniq) byName[m.name || m.id] = m;
    document.querySelectorAll('#models tbody tr').forEach(tr => {
      const idCell = tr.querySelector('td .id');
      if (!idCell) return;
      const name = idCell.textContent.trim();
      const m = byName[name];
      if (!m) return;
      tr.dataset.modelName = name;
      tr.dataset.provider = m.provider || '';
      tr.dataset.upstream = m.provider_model_id || m.omlx_id || '';
      tr.dataset.context = (m.context === undefined || m.context === null || m.context === '') ? '—' : String(m.context);
      tr.dataset.thinking = m.thinking || m.thinking_format || '—';
      tr.dataset.enabled = String(m.enabled !== false);
    });
  };

  document.getElementById('saveProviderBtn').addEventListener('click', saveProvider);
  document.getElementById('deleteProviderBtn').addEventListener('click', deleteProvider);
  document.getElementById('validateProviderBtn').addEventListener('click', validateProvider);
  document.getElementById('saveModelBtn').addEventListener('click', saveModel);
  document.getElementById('deleteModelBtn').addEventListener('click', deleteModel);
  document.getElementById('previewModelBtn').addEventListener('click', previewModel);
  document.getElementById('discoverBtn').addEventListener('click', discoverModels);
  document.querySelectorAll('#winSeg button').forEach(b => b.addEventListener('click', () => {
    loadUsage(b.dataset.w);
    if (drawerKind === 'model' && drawerName) showModelDetail(drawerName, {skipHash: true});
    else if (drawerKind === 'provider' && drawerName) showProviderDetail(drawerName, {skipHash: true});
  }));

  document.addEventListener('keydown', (e) => {
    if (e.key === 'r' && !/INPUT|TEXTAREA|SELECT/.test((e.target.tagName||'').toUpperCase()) && !e.metaKey && !e.ctrlKey) { refresh(); }
  });

  // ── Boot ────────────────────────────────────────────────────
  let stored = '';
  try { stored = sessionStorage.getItem(STORAGE_KEY) || ''; } catch(e) {}
  const initialHash = (window.location.hash || '').replace(/^#/, '').split('/')[0];
  showTab(TABS.includes(initialHash) ? initialHash : currentTab, {skipHash: true});
  if (stored) { keyInput.value = stored; unlock(); }
  else showLocked();
})();
</script>
</body>
</html>
"""
