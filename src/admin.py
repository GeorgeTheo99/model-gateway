"""Admin API and lightweight UI for cloud-gateway."""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src.auth import auth_mode, require_admin_auth
from src.providers import (
    CONFIG_PATH,
    MODEL_INFO_PATH,
    config_validation,
    model_status,
    provider_status,
    reload as reload_provider_registry,
)
from src import config_io, ledger

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
        "service": "cloud-gateway",
        "status": "ok",
        "pid": os.getpid(),
        "uptime_seconds": round(time.time() - _STARTED_AT, 3),
        "model_info_path": str(MODEL_INFO_PATH),
        "config_path": str(CONFIG_PATH),
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


@router.get("/admin/api/config/validation")
async def admin_config_validation(request: Request):
    require_admin_auth(request)
    return config_validation()


@router.post("/admin/api/reload")
async def admin_reload(request: Request):
    require_admin_auth(request)
    reload_provider_registry()
    return {"status": "ok", "message": "provider registry reloaded"}


@router.get("/admin/api/usage")
async def admin_usage(request: Request):
    """Aggregate usage/cost by provider, model, endpoint, and status.

    Query params:
      since   - epoch seconds (inclusive)
      until   - epoch seconds (exclusive)
      window  - shorthand: '1h', '24h', '7d', '30d' (sets `since`)
    """
    require_admin_auth(request)
    import time as _time

    params = request.query_params
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


# ── Provider management (writeable) ──────────────────────────────────────────


@router.post("/admin/api/providers/{provider_id}")
async def admin_upsert_provider(provider_id: str, request: Request):
    """Create or update a provider block in config.yaml.

    Body: {base_url, protocol?, api_key?, default_headers?}. ``api_key`` is
    write-only (None preserves the existing key; "" removes it). Reloads the
    provider registry after writing.
    """
    require_admin_auth(request)
    try:
        body = await request.json()
    except Exception:
        return _bad_request("Invalid JSON body")
    try:
        result = config_io.upsert_provider(
            provider_id,
            base_url=body.get("base_url", ""),
            api_key=body.get("api_key"),
            protocol=body.get("protocol"),
            default_headers=body.get("default_headers"),
        )
    except ValueError as exc:
        return _bad_request(str(exc))
    reload_provider_registry()
    result["reloaded"] = True
    return result


@router.delete("/admin/api/providers/{provider_id}")
async def admin_delete_provider(provider_id: str, request: Request):
    """Remove a provider. Refuses if enabled models depend on it."""
    require_admin_auth(request)
    try:
        result = config_io.delete_provider(provider_id)
    except KeyError as exc:
        return _bad_request(str(exc), status=404)
    except ValueError as exc:
        return _bad_request(str(exc), status=409)
    reload_provider_registry()
    return result


@router.post("/admin/api/providers/{provider_id}/validate")
async def admin_validate_provider(provider_id: str, request: Request):
    """Lightweight upstream validation: authenticated GET {base_url}/models.

    Read-only upstream probe. Returns {ok, status_code, model_count?, error?}.
    """
    require_admin_auth(request)
    import httpx
    from src.providers import resolve as _resolve  # noqa: F401 (kept for clarity)
    import src.providers as providers

    provider_id = provider_id.strip().lower()
    config = providers._load_config()
    block = providers._resolve_provider_config(config, provider_id)
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


# ── Model management (writeable) ────────────────────────────────────────────


@router.post("/admin/api/models/{model_name}")
async def admin_upsert_model(model_name: str, request: Request):
    """Create or update a model entry in model-info.json.

    Body fields (provider + provider_model_id required): provider,
    provider_model_id, alias, context, max_output_tokens, thinking,
    thinking_format, vision, system_instruction, pricing, desc, enabled.
    Writes deploy to the live catalog + source-repo mirror; reloads registry.
    """
    require_admin_auth(request)
    try:
        body = await request.json()
    except Exception:
        return _bad_request("Invalid JSON body")
    try:
        result = config_io.upsert_model(model_name, **body)
    except ValueError as exc:
        return _bad_request(str(exc))
    reload_provider_registry()
    result["reloaded"] = True
    return result


@router.delete("/admin/api/models/{model_name}")
async def admin_delete_model(model_name: str, request: Request):
    """Remove a model entry by name."""
    require_admin_auth(request)
    try:
        result = config_io.delete_model(model_name)
    except KeyError as exc:
        return _bad_request(str(exc), status=404)
    reload_provider_registry()
    return result


@router.post("/admin/api/models/{model_name}/enable")
async def admin_enable_model(model_name: str, request: Request):
    require_admin_auth(request)
    try:
        result = config_io.set_model_enabled(model_name, True)
    except KeyError as exc:
        return _bad_request(str(exc), status=404)
    reload_provider_registry()
    return result


@router.post("/admin/api/models/{model_name}/disable")
async def admin_disable_model(model_name: str, request: Request):
    require_admin_auth(request)
    try:
        result = config_io.set_model_enabled(model_name, False)
    except KeyError as exc:
        return _bad_request(str(exc), status=404)
    reload_provider_registry()
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
  <title>Cloud Gateway Admin</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1020; --panel:#121a2f; --muted:#94a3b8; --text:#e5e7eb; --ok:#34d399; --warn:#f59e0b; --bad:#fb7185; --line:#26324c; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: radial-gradient(circle at top left, #1e3a8a55, transparent 34rem), var(--bg); color: var(--text); }
    header { padding: 28px clamp(18px, 4vw, 52px); border-bottom: 1px solid var(--line); display:flex; justify-content:space-between; gap:16px; align-items:center; }
    h1 { margin: 0 0 6px; font-size: clamp(28px, 4vw, 44px); letter-spacing: -0.04em; }
    h2 { margin: 0 0 14px; font-size: 18px; }
    p { margin: 0; color: var(--muted); }
    main { padding: 26px clamp(18px, 4vw, 52px) 48px; display: grid; gap: 18px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px; }
    .card { background: color-mix(in oklab, var(--panel) 92%, black); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: 0 16px 60px #0005; }
    .metric { font-size: 34px; font-weight: 760; letter-spacing: -0.04em; }
    .muted { color: var(--muted); }
    .ok { color: var(--ok); } .warn { color: var(--warn); } .bad { color: var(--bad); }
    .toolbar { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    input { background:#0f172a; color:var(--text); border:1px solid var(--line); border-radius:12px; padding:10px 12px; min-width:260px; }
    button { background:#2563eb; color:white; border:0; border-radius:12px; padding:10px 14px; font-weight:700; cursor:pointer; }
    button.secondary { background:#334155; }
    table { width:100%; border-collapse:collapse; font-size: 14px; }
    th, td { text-align:left; padding:10px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
    th { color:#cbd5e1; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
    code, .pill { background:#0f172a; border:1px solid var(--line); border-radius:999px; padding:3px 7px; white-space:nowrap; }
    .scroll { overflow:auto; }
    .error { border-color: color-mix(in oklab, var(--bad), var(--line)); }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Cloud Gateway</h1>
      <p>Local model router admin: provider status, model inventory, config validation, and debug entry points.</p>
    </div>
    <div class="toolbar">
      <input id="adminKey" type="password" placeholder="Admin key" autocomplete="off" />
      <button onclick="loadAll()">Use key</button>
      <button class="secondary" onclick="loadAll()">Refresh</button>
    </div>
  </header>
  <main>
    <section class="grid">
      <div class="card"><h2>Status</h2><div id="statusMetric" class="metric muted">—</div><p id="statusSub">Loading…</p></div>
      <div class="card"><h2>Providers</h2><div id="providerMetric" class="metric muted">—</div><p id="providerSub">Configured and missing keys.</p></div>
      <div class="card"><h2>Models</h2><div id="modelMetric" class="metric muted">—</div><p id="modelSub">Routable aliases exposed by /v1/models.</p></div>
      <div class="card"><h2>Validation</h2><div id="validationMetric" class="metric muted">—</div><p id="validationSub">Provider config readiness.</p></div>
    </section>

    <section class="card">
      <h2>Providers</h2>
      <div class="scroll"><table id="providers"><thead><tr><th>ID</th><th>Models</th><th>Protocol</th><th>Base URL</th><th>API key</th><th>Issues</th><th>Actions</th></tr></thead><tbody></tbody></table></div>
      <details style="margin-top:12px;"><summary class="muted">Add / edit provider</summary>
        <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top:10px; gap:10px;">
          <input id="pId" placeholder="provider id (e.g. anthropic)" />
          <input id="pBaseUrl" placeholder="base_url (https://...)" style="min-width:240px;" />
          <input id="pProtocol" placeholder="protocol (openai|anthropic)" />
          <input id="pApiKey" type="password" placeholder="api_key (write-only)" />
        </div>
        <div class="toolbar" style="margin-top:8px;">
          <button onclick="saveProvider()">Save provider</button>
          <button class="secondary" onclick="validateProvider()">Validate connection</button>
          <button class="secondary" onclick="deleteProvider()">Delete</button>
          <span id="pMsg" class="muted"></span>
        </div>
        <p class="muted" style="font-size:12px; margin-top:6px;">api_key is write-only: leave blank to keep the existing key. Provider config is stored in the gitignored config.yaml and is durable across deploys.</p>
      </details>
    </section>

    <section class="card">
      <h2>Models</h2>
      <div class="scroll"><table id="models"><thead><tr><th>Name</th><th>Provider</th><th>Upstream</th><th>Context</th><th>Max out</th><th>Thinking</th><th>Vision</th><th>Enabled</th><th>Actions</th></tr></thead><tbody></tbody></table></div>
      <details style="margin-top:12px;"><summary class="muted">Add / edit model</summary>
        <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); margin-top:10px; gap:10px;">
          <input id="mName" placeholder="name (gateway id)" />
          <input id="mProvider" placeholder="provider" />
          <input id="mPmid" placeholder="provider_model_id" />
          <input id="mAlias" placeholder="alias" />
          <input id="mContext" type="number" placeholder="context" />
          <input id="mMaxOut" type="number" placeholder="max_output_tokens" />
          <input id="mThinking" placeholder="thinking (|optional|always)" />
          <input id="mThinkingFmt" placeholder="thinking_format" />
          <input id="mPricing" placeholder="pricing JSON" style="min-width:200px;" />
        </div>
        <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); margin-top:8px; gap:10px;">
          <label class="muted"><input id="mVision" type="checkbox" /> vision</label>
          <label class="muted"><input id="mEnabled" type="checkbox" checked /> enabled</label>
          <input id="mDesc" placeholder="desc (no $ prices)" style="min-width:240px;" />
        </div>
        <div class="toolbar" style="margin-top:8px;">
          <button onclick="saveModel()">Save model</button>
          <button class="secondary" onclick="deleteModel()">Delete</button>
          <span id="mMsg" class="muted"></span>
        </div>
        <p class="muted" style="font-size:12px; margin-top:6px;">Writes are hot (immediate) and mirrored to the source repo (pending commit). Pricing should come from the official provider — use the cloud-gateway-add-model skill to fetch it. Disabled models are hidden from /v1/models.</p>
      </details>
    </section>

    <section class="card">
      <h2>Debug</h2>
      <p>Existing reasoning matrix: <a href="/v1/debug/thinking">/v1/debug/thinking</a>. If client auth is enabled, open it with an admin/client key-capable HTTP client.</p>
      <pre id="errors" class="muted"></pre>
    </section>

    <section class="card">
      <h2>Usage &amp; Cost</h2>
      <div class="toolbar" style="margin-bottom:12px;">
        <button class="secondary" onclick="loadUsage('1h')">1h</button>
        <button class="secondary" onclick="loadUsage('24h')">24h</button>
        <button class="secondary" onclick="loadUsage('7d')">7d</button>
        <button class="secondary" onclick="loadUsage('30d')">30d</button>
        <button class="secondary" onclick="loadUsage('')">All</button>
        <span id="usageRange" class="muted" style="margin-left:auto;"></span>
      </div>
      <div class="grid" style="margin-bottom:14px;">
        <div class="card"><h2>Requests</h2><div id="uRequests" class="metric muted">—</div><p id="uRequestsSub" class="muted">ok / errors</p></div>
        <div class="card"><h2>Tokens</h2><div id="uTokens" class="metric muted">—</div><p id="uTokensSub" class="muted">in / out (cached read)</p></div>
        <div class="card"><h2>Est. cost</h2><div id="uCost" class="metric muted">—</div><p id="uCostSub" class="muted">USD, priced rows only</p></div>
        <div class="card"><h2>Avg latency</h2><div id="uLatency" class="metric muted">—</div><p id="uLatencySub" class="muted">ms over window</p></div>
      </div>
      <h2>By model</h2>
      <div class="scroll"><table id="usageByModel"><thead><tr><th>Model</th><th>Requests</th><th>Errors</th><th>In tok</th><th>Out tok</th><th>Cached</th><th>Cost</th><th>Avg ms</th></tr></thead><tbody></tbody></table></div>
      <h2 style="margin-top:16px;">Recent requests</h2>
      <div class="scroll"><table id="recentReq"><thead><tr><th>Time</th><th>Endpoint</th><th>Model</th><th>Status</th><th>Stream</th><th>In</th><th>Out</th><th>Cached</th><th>Cost</th><th>Latency</th></tr></thead><tbody></tbody></table></div>
      <pre id="usageErrors" class="muted"></pre>
    </section>
  </main>
<script>
const keyInput = document.getElementById('adminKey');
function headers(){ const key = keyInput.value.trim(); return key ? {'Authorization':'Bearer '+key} : {}; }
function escapeHtml(value){ return text(value).replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch])); }
async function get(path){
  const res = await fetch(path, {headers: headers()});
  if(!res.ok) throw new Error(path + ' -> HTTP ' + res.status + ': ' + await res.text());
  return res.json();
}
function text(v){ return (v === undefined || v === null || v === '') ? '—' : String(v); }
function cls(ok){ return ok ? 'ok' : 'bad'; }
function pill(v){ return '<span class="pill">'+escapeHtml(v)+'</span>'; }
async function loadAll(){
  const err = document.getElementById('errors'); err.textContent = '';
  try {
    const [status, providers, models, validation] = await Promise.all([
      get('/admin/api/status'), get('/admin/api/providers'), get('/admin/api/models'), get('/admin/api/config/validation')
    ]);
    document.getElementById('statusMetric').textContent = status.status;
    document.getElementById('statusMetric').className = 'metric ok';
    document.getElementById('statusSub').textContent = 'Uptime ' + Math.round(status.uptime_seconds) + 's · client auth ' + (status.auth.client_auth_enabled ? 'on' : 'off') + ' · admin auth ' + (status.auth.admin_auth_enabled ? 'on' : 'off');
    const providerRows = providers.providers || [];
    const modelRows = models.models || [];
    const badProviders = providerRows.filter(p => (p.issues || []).length);
    document.getElementById('providerMetric').textContent = providerRows.length;
    document.getElementById('providerMetric').className = 'metric ' + (badProviders.length ? 'warn' : 'ok');
    document.getElementById('providerSub').textContent = badProviders.length ? badProviders.length + ' provider(s) need config' : 'All providers with models look configured';
    document.getElementById('modelMetric').textContent = modelRows.length;
    document.getElementById('modelMetric').className = 'metric ok';
    document.getElementById('validationMetric').textContent = validation.ok ? 'ok' : 'issues';
    document.getElementById('validationMetric').className = 'metric ' + (validation.ok ? 'ok' : 'bad');
    document.getElementById('validationSub').textContent = validation.issues?.length ? JSON.stringify(validation.issues) : 'No missing provider credentials detected';
    document.querySelector('#providers tbody').innerHTML = providerRows.map(p => `<tr><td>${pill(p.id)}</td><td>${escapeHtml(p.enabled_models)}</td><td>${escapeHtml(p.protocol)}</td><td>${escapeHtml(p.base_url)}</td><td class="${cls(p.has_api_key)}">${p.has_api_key ? 'present' : 'missing'}</td><td>${(p.issues||[]).map(pill).join(' ') || '—'}</td><td><button class="secondary" onclick="editProvider('${escapeHtml(p.id)}')">edit</button></td></tr>`).join('');
    // Dedupe models by name (list_models returns name+alias+id rows).
    const seen = new Set(); const uniq = [];
    for (const m of modelRows) { const n = m.name || m.id; if (n && !seen.has(n)) { seen.add(n); uniq.push(m); } }
    document.querySelector('#models tbody').innerHTML = uniq.map(m => {
      const en = m.enabled !== false;
      const nm = escapeHtml(m.name || m.id);
      return `<tr><td>${pill(m.name || m.id)}</td><td>${escapeHtml(m.provider)}</td><td>${escapeHtml(m.provider_model_id)}</td><td>${escapeHtml(m.context)}</td><td>${escapeHtml(m.max_output_tokens)}</td><td>${escapeHtml(m.thinking || m.thinking_format)}</td><td>${m.vision ? 'yes' : 'no'}</td><td class="${cls(en)}">${en ? 'yes' : 'no'}</td><td><button class="secondary" onclick="editModel('${nm}')">edit</button> <button class="secondary" onclick="toggleModel('${nm}', ${!en})">${en ? 'disable' : 'enable'}</button></td></tr>`;
    }).join('');
  } catch(e) {
    err.textContent = e.message;
    document.getElementById('statusMetric').textContent = 'error';
    document.getElementById('statusMetric').className = 'metric bad';
    document.getElementById('statusSub').textContent = 'Check admin key or gateway logs.';
  }
}
loadAll();

// ── Provider/model CRUD ───────────────────────────────────────────────
async function send(method, path, body){
  const opts = {method, headers: headers()};
  if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  const res = await fetch(path, opts);
  const txt = await res.text();
  let j = null; try { j = JSON.parse(txt); } catch(e) { j = {raw: txt}; }
  if (!res.ok) throw new Error((j && j.error && j.error.message) || txt || ('HTTP '+res.status));
  return j;
}
function setMsg(id, msg, ok){ const el = document.getElementById(id); el.textContent = msg; el.className = ok ? 'ok' : 'bad'; }
async function saveProvider(){
  const id = document.getElementById('pId').value.trim().toLowerCase();
  if (!id) return setMsg('pMsg', 'provider id required', false);
  const body = {base_url: document.getElementById('pBaseUrl').value.trim()};
  const proto = document.getElementById('pProtocol').value.trim(); if (proto) body.protocol = proto;
  const key = document.getElementById('pApiKey').value; if (key) body.api_key = key;
  try { await send('POST', '/admin/api/providers/'+encodeURIComponent(id), body); setMsg('pMsg', 'saved + reloaded', true); loadAll(); }
  catch(e){ setMsg('pMsg', e.message, false); }
}
async function deleteProvider(){
  const id = document.getElementById('pId').value.trim().toLowerCase();
  if (!id) return setMsg('pMsg', 'provider id required', false);
  if (!confirm('Delete provider '+id+'?')) return;
  try { await send('DELETE', '/admin/api/providers/'+encodeURIComponent(id)); setMsg('pMsg', 'deleted + reloaded', true); loadAll(); }
  catch(e){ setMsg('pMsg', e.message, false); }
}
async function validateProvider(){
  const id = document.getElementById('pId').value.trim().toLowerCase();
  if (!id) return setMsg('pMsg', 'provider id required', false);
  setMsg('pMsg', 'validating…', true);
  try { const r = await send('POST', '/admin/api/providers/'+encodeURIComponent(id)+'/validate'); setMsg('pMsg', r.ok ? ('OK · '+r.status_code+' · '+(r.model_count??'?')+' models') : ('failed: '+(r.error||r.status_code)), r.ok); }
  catch(e){ setMsg('pMsg', e.message, false); }
}
function editProvider(id){
  document.getElementById('pId').value = id;
  // Fetch current values from the loaded table row is fragile; leave fields for user to fill.
  setMsg('pMsg', 'editing '+id+' — fill base_url (blank=keep) + key (blank=keep)', true);
}
function editModel(name){
  // Populate the form from the loaded model row.
  const rows = document.querySelectorAll('#models tbody tr');
  let m = null;
  for (const r of rows) { if (r.children[0].textContent.replace(/[^a-z0-9.-]/gi,'') === name || r.children[0].textContent.includes(name)) { m = r.children; break; } }
  if (!m) return setMsg('mMsg', 'could not find '+name, false);
  document.getElementById('mName').value = name;
  document.getElementById('mProvider').value = m[1].textContent === '—' ? '' : m[1].textContent;
  document.getElementById('mPmid').value = m[2].textContent === '—' ? '' : m[2].textContent;
  document.getElementById('mContext').value = m[3].textContent === '—' ? '' : m[3].textContent;
  document.getElementById('mMaxOut').value = '';
  document.getElementById('mThinking').value = m[5].textContent === '—' ? '' : m[5].textContent;
  document.getElementById('mEnabled').checked = m[7].textContent === 'yes';
  setMsg('mMsg', 'editing '+name, true);
}
async function saveModel(){
  const name = document.getElementById('mName').value.trim();
  if (!name) return setMsg('mMsg', 'name required', false);
  const body = {provider: document.getElementById('mProvider').value.trim(), provider_model_id: document.getElementById('mPmid').value.trim()};
  if (!body.provider || !body.provider_model_id) return setMsg('mMsg', 'provider + provider_model_id required', false);
  const alias = document.getElementById('mAlias').value.trim(); if (alias) body.alias = alias;
  const ctx = parseInt(document.getElementById('mContext').value); if (!isNaN(ctx)) body.context = ctx;
  const mo = parseInt(document.getElementById('mMaxOut').value); if (!isNaN(mo)) body.max_output_tokens = mo;
  const th = document.getElementById('mThinking').value.trim(); if (th) body.thinking = th;
  const tf = document.getElementById('mThinkingFmt').value.trim(); if (tf) body.thinking_format = tf;
  const pr = document.getElementById('mPricing').value.trim(); if (pr) { try { body.pricing = JSON.parse(pr); } catch(e){ return setMsg('mMsg', 'pricing JSON invalid', false); } }
  const desc = document.getElementById('mDesc').value.trim(); if (desc) body.desc = desc;
  body.vision = document.getElementById('mVision').checked;
  body.enabled = document.getElementById('mEnabled').checked;
  try { await send('POST', '/admin/api/models/'+encodeURIComponent(name), body); setMsg('mMsg', 'saved + reloaded (hot; commit source repo to persist)', true); loadAll(); }
  catch(e){ setMsg('mMsg', e.message, false); }
}
async function deleteModel(){
  const name = document.getElementById('mName').value.trim();
  if (!name) return setMsg('mMsg', 'name required', false);
  if (!confirm('Delete model '+name+'?')) return;
  try { await send('DELETE', '/admin/api/models/'+encodeURIComponent(name)); setMsg('mMsg', 'deleted + reloaded', true); loadAll(); }
  catch(e){ setMsg('mMsg', e.message, false); }
}
async function toggleModel(name, enable){
  const ep = enable ? 'enable' : 'disable';
  try { await send('POST', '/admin/api/models/'+encodeURIComponent(name)+'/'+ep); setMsg('mMsg', name+' '+ep+'d', true); loadAll(); }
  catch(e){ setMsg('mMsg', e.message, false); }
}

const _winLabel = {'1h':'Last hour','24h':'Last 24 hours','7d':'Last 7 days','30d':'Last 30 days','':'All time'};
function fmtNum(v){ return (v===null||v===undefined) ? '—' : Number(v).toLocaleString(); }
function fmtCost(v){ return (v===null||v===undefined) ? '—' : '$'+Number(v).toFixed(4); }
function fmtMs(v){ return (v===null||v===undefined) ? '—' : Math.round(v)+'ms'; }
async function loadUsage(window){
  const err = document.getElementById('usageErrors'); err.textContent = '';
  document.getElementById('usageRange').textContent = _winLabel[window] || window || 'All';
  try {
    const q = window ? ('?window='+encodeURIComponent(window)) : '';
    const [usage, recent] = await Promise.all([ get('/admin/api/usage'+q), get('/admin/api/requests?limit=50') ]);
    const s = usage.summary || {};
    const ok = s.ok || 0, errs = s.errors || 0;
    document.getElementById('uRequests').textContent = fmtNum(s.requests);
    document.getElementById('uRequests').className = 'metric ' + (errs ? 'warn' : 'ok');
    document.getElementById('uRequestsSub').textContent = ok + ' ok / ' + errs + ' errors';
    document.getElementById('uTokens').textContent = fmtNum((s.input_tokens||0) + (s.output_tokens||0));
    document.getElementById('uTokens').className = 'metric ok';
    document.getElementById('uTokensSub').textContent = fmtNum(s.input_tokens) + ' in / ' + fmtNum(s.output_tokens) + ' out (' + fmtNum(s.cached_read_tokens) + ' cached)';
    document.getElementById('uCost').textContent = fmtCost(s.cost_usd);
    document.getElementById('uCost').className = 'metric ok';
    document.getElementById('uCostSub').textContent = 'unpriced rows excluded';
    document.getElementById('uLatency').textContent = fmtMs(s.avg_latency_ms);
    document.getElementById('uLatency').className = 'metric ok';
    document.getElementById('uLatencySub').textContent = s.requests ? 'over ' + s.requests + ' requests' : 'no requests';
    const byModel = usage.by_model || [];
    document.querySelector('#usageByModel tbody').innerHTML = byModel.map(r => `<tr><td>${pill(r.dim || '—')}</td><td>${fmtNum(r.requests)}</td><td class="${cls(!(r.errors))}">${fmtNum(r.errors)}</td><td>${fmtNum(r.input_tokens)}</td><td>${fmtNum(r.output_tokens)}</td><td>${fmtNum(r.cached_read_tokens)}</td><td>${fmtCost(r.cost_usd)}</td><td>${fmtMs(r.avg_latency_ms)}</td></tr>`).join('') || '<tr><td class="muted" colspan="8">No requests in window</td></tr>';
    const reqs = recent.requests || [];
    document.querySelector('#recentReq tbody').innerHTML = reqs.map(r => `<tr><td>${escapeHtml(r.ts_iso||'—')}</td><td>${escapeHtml(r.endpoint||'—')}</td><td>${pill(r.model||'—')}</td><td class="${cls(r.status && r.status < 400)}">${r.status||'—'}</td><td>${r.is_stream ? 'stream' : 'sync'}</td><td>${fmtNum(r.input_tokens)}</td><td>${fmtNum(r.output_tokens)}</td><td>${fmtNum(r.cached_read_tokens)}</td><td>${fmtCost(r.cost_usd)}</td><td>${fmtMs(r.latency_ms)}</td></tr>`).join('') || '<tr><td class="muted" colspan="10">No requests yet</td></tr>';
  } catch(e) {
    err.textContent = e.message;
  }
}
loadUsage('24h');
</script>
</body>
</html>
"""
