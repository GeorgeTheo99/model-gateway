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
      <div class="scroll"><table id="providers"><thead><tr><th>ID</th><th>Models</th><th>Protocol</th><th>Base URL</th><th>API key</th><th>Issues</th></tr></thead><tbody></tbody></table></div>
    </section>

    <section class="card">
      <h2>Models</h2>
      <div class="scroll"><table id="models"><thead><tr><th>ID</th><th>Provider</th><th>Upstream model</th><th>Context</th><th>Thinking</th><th>Vision</th><th>Provider config</th></tr></thead><tbody></tbody></table></div>
    </section>

    <section class="card">
      <h2>Debug</h2>
      <p>Existing reasoning matrix: <a href="/v1/debug/thinking">/v1/debug/thinking</a>. If client auth is enabled, open it with an admin/client key-capable HTTP client.</p>
      <pre id="errors" class="muted"></pre>
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
    document.querySelector('#providers tbody').innerHTML = providerRows.map(p => `<tr><td>${pill(p.id)}</td><td>${escapeHtml(p.enabled_models)}</td><td>${escapeHtml(p.protocol)}</td><td>${escapeHtml(p.base_url)}</td><td class="${cls(p.has_api_key)}">${p.has_api_key ? 'present' : 'missing'}</td><td>${(p.issues||[]).map(pill).join(' ') || '—'}</td></tr>`).join('');
    document.querySelector('#models tbody').innerHTML = modelRows.map(m => `<tr><td>${pill(m.id)}</td><td>${escapeHtml(m.provider)}</td><td>${escapeHtml(m.provider_model_id)}</td><td>${escapeHtml(m.context)}</td><td>${escapeHtml(m.thinking || m.thinking_format)}</td><td>${m.vision ? 'yes' : 'no'}</td><td class="${cls(m.provider_ready)}">${m.provider_ready ? 'ready' : (m.provider_configured ? 'incomplete' : 'missing')}</td></tr>`).join('');
  } catch(e) {
    err.textContent = e.message;
    document.getElementById('statusMetric').textContent = 'error';
    document.getElementById('statusMetric').className = 'metric bad';
    document.getElementById('statusSub').textContent = 'Check admin key or gateway logs.';
  }
}
loadAll();
</script>
</body>
</html>
"""
