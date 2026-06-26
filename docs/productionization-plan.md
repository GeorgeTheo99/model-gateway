# Cloud Gateway Productionization Plan

## Outcome

Turn the local cloud gateway into a managed model-router product: a small admin UI, durable provider/model configuration, explicit access control, live observability, and cost tracking while preserving the existing OpenAI-compatible, Anthropic-compatible, and Responses-compatible APIs.

## Current state

- Runtime is a FastAPI service in `src/server.py` with:
  - `GET /health`
  - `GET /v1/models`
  - `GET /v1/debug/thinking`
  - `POST /v1/chat/completions`
  - `POST /v1/messages`
  - `POST /v1/responses`
- Provider/model resolution lives in `src/providers.py`:
  - model catalog: `model-info.json`
  - secrets/config: `config/config.yaml`
  - provider protocol: `openai` or `anthropic`
- Existing observability is reasoning-focused only: `/v1/models` exposes thinking support and `/v1/debug/thinking` shows forwarding behavior.
- There is no gateway admin UI, no explicit inbound gateway auth, no request ledger, no cost ledger, and provider/model changes require file edits + process reload/restart.

## Non-goals for the first production pass

- Do not add OAuth initially. Provider API keys are enough for Anthropic/OpenAI-compatible providers today.
- Do not expose provider API keys back to the browser/API after save.
- Do not replace the stable request translation path unless needed for management/observability hooks.
- Do not make this a multi-tenant internet service; assume local/private deployment with explicit client/admin tokens.

## Target architecture

```text
Admin UI ── /admin/* ─┐
CLI/tools ─ /admin/api/* ├── Admin auth ── config store ── providers/models/pricing
Clients ─── /v1/* ───────┘       │
                                 ├── request middleware / route hooks
                                 ├── usage + cost ledger
                                 └── provider health / validation probes

Upstream providers: Anthropic, OpenAI, Fireworks, Z.ai/Zhipu, OpenRouter, future OpenAI-compatible endpoints
```

### 1. Configuration model

Replace implicit free-form config with validated schemas while keeping file compatibility.

- `providers` table/config:
  - `id`: stable provider key, e.g. `anthropic`, `openai`, `fireworks`, `zai_coding`, `custom-openai-1`
  - `display_name`
  - `kind`: `openai-compatible`, `anthropic`, later `oauth` or `custom`
  - `base_url`
  - `api_key_ref`: reference to secret store; never returned in plaintext
  - `enabled`
  - `default_headers` / feature flags when needed
- `models` table/config:
  - `id`: gateway-facing model id/alias
  - `provider_id`
  - `provider_model_id`
  - `protocol`: request contract to upstream when provider supports multiple
  - `context`, `max_output_tokens`, `vision`, `thinking`, `thinking_format`
  - `enabled`
  - optional `system_instruction`
- `pricing` table/config:
  - `model_id` or `provider_model_id`
  - `input_per_million`, `output_per_million`
  - optional `cached_input_per_million`, `reasoning_per_million`, `currency`
  - `effective_date`, `source_url`/`source_note`

Recommended store for v1: SQLite + encrypted/permissioned local secret file, with import/export to the existing YAML/JSON formats. This enables UI edits, history, and request ledgers without introducing a heavier service.

### 2. Access control

Add two explicit token classes before exposing write endpoints:

- `CLOUD_GATEWAY_CLIENT_KEYS`: tokens allowed to call `/v1/*`.
- `CLOUD_GATEWAY_ADMIN_KEY`: token allowed to call `/admin/api/*` and load the UI.

For backward compatibility, allow an env flag like `CLOUD_GATEWAY_ALLOW_UNAUTHENTICATED_LOCAL=true` only when bound to loopback/private local dev.

### 3. Admin API

Read-only first, write endpoints after auth is in place.

- `GET /admin/api/status`
  - gateway version, uptime, config path/store, model catalog path, auth mode, ledger mode.
- `GET /admin/api/providers`
  - provider configs with `has_api_key`, never the secret.
- `POST /admin/api/providers`
  - create/update provider + secret.
- `POST /admin/api/providers/{id}/validate`
  - lightweight upstream validation, e.g. list models or minimal authenticated request.
- `DELETE /admin/api/providers/{id}`
  - disable/delete after confirming no enabled models depend on it.
- `GET /admin/api/models`
  - merged model catalog with provider status, thinking/vision/context, enabled state.
- `POST /admin/api/models`
  - create/update custom model mapping.
- `POST /admin/api/models/import`
  - import from provider list-models where available.
- `GET /admin/api/usage`
  - aggregate requests/tokens/cost/latency/errors by time window, provider, model, endpoint, status.
- `GET /admin/api/requests`
  - recent redacted request ledger.

### 4. UI

Serve a local admin UI from the gateway process to minimize deployment friction.

Initial pages:

- **Overview**: health, uptime, enabled providers, routable models, recent errors, spend today/month.
- **Providers**: add/edit provider, set API key, validate connection, enabled/disabled state.
- **Models**: gateway id, provider model id, context, output, vision, thinking, protocol, enable/disable, import/discover.
- **Observability**: request volume, latency, errors, token totals, estimated cost, recent requests.
- **Debug**: existing thinking matrix, provider config validation, routing preview for a given model id.

Recommended frontend v1: static React/Vite or plain TypeScript served by FastAPI. If keeping dependencies minimal matters more than rich UX, use static HTML/CSS/JS with `/admin/api/*`.

### 5. Observability and cost

Add a redacted request ledger around every `/v1/*` route.

Fields:

- timestamp, request id, endpoint, gateway model id, provider id, upstream model id
- HTTP status, error type, retry/fallback marker
- latency_ms, stream/non-stream
- input_tokens, output_tokens, cached_input_tokens, reasoning_tokens when provider returns usage
- estimated_cost_usd and pricing version

For streaming responses, wrap the stream and parse final usage chunks where providers provide them; otherwise store status/latency and mark token/cost estimate as unavailable.

Cost policy:

- Prefer exact provider-reported usage when present.
- Use configured pricing per model/provider.
- Mark estimates clearly as estimates.
- Show unknown cost when usage or pricing is missing; do not guess silently.

### 6. Provider extensibility

Support future providers through provider adapters rather than hard-coded branches everywhere.

Adapter interface:

- `build_headers(provider_config)`
- `rewrite_request(route, body, model_config)`
- `rewrite_response(route, response)` when needed
- `validate(provider_config)`
- `discover_models(provider_config)` optional
- `extract_usage(response_or_stream)`

Initial adapters:

- `openai-compatible`: OpenAI, Fireworks, Z.ai/Zhipu, OpenRouter/custom endpoints
- `anthropic`: native Anthropic Messages API

Provider-specific quirks currently in `src/server.py` (Fireworks cleanup/compression, Z.ai thinking format, OpenRouter/Gemini signatures) can migrate behind adapters incrementally.

### 7. OAuth posture

Defer OAuth until a real provider requires it.

If/when needed:

- add provider `auth_type`: `api_key` or `oauth2`
- store refresh tokens in the same local secret store
- UI can run device-code flow or local callback flow
- adapter owns token refresh

## Compatibility / breakage flags

The gateway is already used by local launchers and services, so productionization must be staged.

### Safe in the first slice

- `/v1/*` remains unauthenticated unless `CLOUD_GATEWAY_CLIENT_KEYS` is set.
- `/admin/api/*` now fails closed unless `CLOUD_GATEWAY_ADMIN_KEY` is set, or `CLOUD_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN=true` is explicitly set for local development.
- Existing model ids, aliases, request bodies, and upstream routing behavior are unchanged.
- Provider API keys are still read from `config/config.yaml`; the new admin APIs only report masked status.

### What will break if client auth is enabled

Set `CLOUD_GATEWAY_CLIENT_KEYS=cloud` to match the current local launchers before requiring auth.

Known clients that should continue working because they already send a token:

- `server/local_claude/zshrc-launcher.zsh` Claude path sends `ANTHROPIC_AUTH_TOKEN=cloud`.
- `server/local_claude/zshrc-launcher.zsh` Codex path sends `OPENAI_API_KEY=cloud`.
- Pi generated `models.json` uses `apiKey: cloud` for the cloud-gateway provider.

Clients that will break until updated:

- raw `curl` calls without `Authorization: Bearer <key>` or `x-api-key: <key>`.
- scripts that call `http://localhost:9111/v1/*` with no API key.
- health/model probes that use `/v1/models` without auth after `CLOUD_GATEWAY_CLIENT_KEYS` is configured.

`/health` intentionally stays unauthenticated for launchd and load-balancer-style checks.

### What will break if admin auth is enabled

- Browser access to `/admin` still loads the static UI, but `/admin/api/*` calls return `401` until `CLOUD_GATEWAY_ADMIN_KEY` is configured and entered in the UI.
- Programmatic admin calls need `Authorization: Bearer $CLOUD_GATEWAY_ADMIN_KEY` or `x-api-key: $CLOUD_GATEWAY_ADMIN_KEY`.
- For temporary local-only development, set `CLOUD_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN=true`; do not use that when binding beyond trusted loopback/private development.

### Future breaking changes to avoid or stage carefully

- Moving provider/model config from YAML/JSON into SQLite should start as import/export-compatible; do not remove file support immediately.
- Enabling writeable provider/key management must not echo existing secrets back to clients.
- Refactoring provider adapters must preserve provider-specific behavior currently embedded in `src/server.py`, especially Fireworks image compression, OpenRouter/Gemini signatures, and Z.ai thinking controls.
- Cost/usage ledgers must redact prompts by default; logging full prompts would be a privacy/security behavior change.

## Milestones

1. **Hardening baseline**
   - Sanitize examples/docs.
   - Add opt-in inbound client/admin auth.
   - Add config validation and masked provider status endpoint.
2. **Read-only admin UI**
   - Overview, providers, models, existing thinking matrix.
   - No secret writes yet.
3. **Request/usage ledger**
   - SQLite schema, request ids, latency/status/error logging.
   - Token/cost extraction for non-streaming first.
4. **Provider/key management**
   - Add/edit provider configs and API keys via admin API/UI.
   - Validate provider connection.
   - Hot reload provider/model registry.
5. **Model management**
   - Add/import/enable/disable model mappings.
   - Routing preview and config validation.
6. **Cost dashboard**
   - Pricing config, daily/monthly spend, per-provider/model breakdown.
   - Streaming usage support where possible.
7. **Adapter refactor**
   - Move provider-specific behavior out of route bodies incrementally.

## First implementation slice

Implement milestone 1 with tests before UI work:

- `src/admin.py` or equivalent for admin API.
- `src/config_store.py` for typed load/validate/masked status.
- `src/auth.py` for admin/client token checks.
- Tests for:
  - no provider secret leaks
  - missing provider config surfaced in validation
  - `/v1/*` rejects unauthorized clients when client auth is enabled
  - admin endpoints require admin token when configured

This creates the safe foundation needed before adding writeable UI/key management.
