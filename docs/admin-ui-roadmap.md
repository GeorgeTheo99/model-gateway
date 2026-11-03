# Admin UI

The approved direction is documented in [admin-ui-design-brief.md](admin-ui-design-brief.md).
The UI remains embedded in `_ADMIN_HTML` in `src/admin.py`: no frontend build step,
external runtime assets, or framework dependency.

## Implemented structure

- **Overview**: **Running now** (requests the gateway is serving, oMLX models in
  memory with size and last use, the media-gen model and queue; refreshed every
  5 seconds while Overview or Models is open), actionable configuration/routing
  issues, first-connection guidance, inventory links, a 7-day summary with a
  daily cost chart and top models by cost, and the latest five requests.
- **Models**: search and location/connection/capability/30-day-use filters, and
  sorting by recent use, requests, or cost. Each row shows live status (in
  memory, loading, not loaded, configured), last use, and 30-day requests and
  cost. Model details add client IDs, copyable request examples,
  pricing/capabilities, configured routing, image handling, and usage with a
  24-hour/7/30/90-day window and cost chart. Gateway-owned presets are a
  disclosure here.
- **Usage**: cost and requests for 24 hours, 7/30/90 days, any number of days,
  or all time; a per-day (hourly for two days or less) cost chart for all models
  or one model; a sortable, filterable per-model table with share of cost; token
  accounting; and per-connection totals. Rows group every alias and upstream ID
  under its catalog model.
- **Requests**: the latest 50, 200, or 500 requests with search and outcome
  filters, relative times, tokens, duration, and cost (local runtimes show
  "Local").
- **Connections**: provider/local-runtime inventory with 7-day requests,
  failures, last request, and cost; safe edit/discovery flows; and contextual
  routing groups (workspace pools). Workspace diagnostics are separate from the
  preference order; each workspace appears once in the group inventory.
- **Consumers**: consumer credentials (role, namespaces, key-file status) with
  add, rotate, and revoke in write mode, and read-only profile namespaces with
  the latest snapshot and version history.
- **Settings & diagnostics**: secondary access to management-mode instructions,
  runtime metadata, backup status, portable-bundle download, and authenticated
  protocol diagnostics.

Legacy links remain supported: `#providers[/name]`, `#pools`, `#presets`,
`#usage/<request-id>`, and `#debug`. Primary links are `#overview`,
`#models[/name]`, `#usage`, `#activity[/request-id]` (Requests),
`#connections[/name]`, and `#consumers`.

### Admin APIs behind the dashboard

- `GET /admin/api/runtime`: requests in progress (process-local, from the ledger
  middleware), oMLX `/v1/models/status`, and media-gen `/health`, each runtime
  asked with a 3-second timeout. An unreachable runtime is reported, not shown as
  idle. Runtimes with no catalog models are not asked.
- `GET /admin/api/usage?window=<N>h|<N>d`: adds `by_catalog_model` and `series`
  (gap-filled buckets in the gateway's local time, with `cost_by_model`) to the
  existing aggregates.
- `GET /admin/api/models/<name>/stats?window=...`: adds the model's `series`.

## Safety and evidence boundaries

- Admin authentication and `MODEL_GATEWAY_ADMIN_WRITES` still gate API access.
  Read-only mode hides management actions, including explicit provider inventory
  checks and discovery. CLI instructions only copy text; they never execute commands.
- Saves activate configuration immediately. Pool membership remains CLI-managed;
  model edit preserves it and preview includes it. Browser connection deletion is
  refused for pool members because the operator CLI has the transactional checks.
- Provider switching resets all editable fields and clears the write-only key.
  Saving an unchanged redacted URL is refused to avoid losing hidden authentication
  components. Model discovery resets previous model fields before prefill.
- Refresh failures are visible globally and by section. Retained values are marked
  stale. Independent sections can load even if another endpoint fails.
- Pool preference is local configuration plus circuit state, not upstream verification.
  An all-open pool means no immediately ready member, not proof that recovery cannot run.
- Model metadata is explicitly whitelisted. Tools support is unknown unless explicitly
  recorded; a tool-parser setting is not evidence of model support. Configured model
  fallback and scoped image routing are not execution traces.
- Provider inventory checks read the upstream model list, not inference. Their results
  are session-local, not a persistent verification history.
- Requests contain the recorded route, not a guaranteed final failover workspace or
  full attempt chain. Missing evidence is labelled **Not recorded**. Duration is the
  recorded request latency, not TTFT or guaranteed stream-completion time.
- Request filters operate on the latest 50, 200, or 500 records across all time.
  Detail histories are the latest 25, independent of their usage windows.
- "Running now" lists requests this gateway process is serving; requests sent to
  oMLX or media-gen directly are not included. oMLX memory figures are its own
  resident estimates.
- Known cost is a subtotal; accounting completeness and unknown/partial costs are
  shown separately. No admin or client credentials are embedded in copyable examples.
- Consumer key values are never listed. A key generated by add or rotate is returned
  once (`Cache-Control: no-store`), kept only in the page until dismissed, reloaded,
  or locked, and never written to browser storage. Rotate and revoke require the
  credential id as confirmation, in the UI and in the API body. Only managed key
  files are rotated; revoke keeps the last `/v1` credential.
- Backup status reads names and timestamps only, never backup contents. Bundle
  download is refused when a URL embeds userinfo, a query string, or a fragment.

## Verification

Python API, metadata, and structural regressions:

```sh
uv run pytest -q -p no:cacheprovider
```

Browser behavior uses an isolated in-memory API with synthetic data and no production
credentials. Requires an installed Playwright package and its Chromium browser
(a cached `npx playwright` install works via `NODE_PATH`):

```sh
node tests/admin_ui_browser.mjs
```

If Playwright is installed outside this repository, set `NODE_PATH` to its containing
`node_modules` directory. Set `ADMIN_UI_SCREENSHOT_DIR` to an external artifact directory
to capture desktop, tablet, mobile, detail-view, and dark-theme screenshots.

For local visual inspection only:

```sh
node tests/admin_ui_browser.mjs --serve
```

This serves the current embedded UI at `http://127.0.0.1:9127/admin`, using the public
fixture key `fixture-admin`. All writes affect only in-memory synthetic data. Stop the
process when finished. This is not the production gateway or a deployment command.

## Still separate work

- Browser-managed workspace authentication, replacement, or pool reordering needs
  dedicated safe APIs; the UI deliberately does not pretend those operations exist.
- Server-side request pagination/search and persistent provider verification history.
- Fully guided fresh-install onboarding beyond the existing save → check → discover
  → register → preview flow.
- Federation management and profile editing require their own scope and
  authorization design. Profiles stay read-only here; consumers publish them with
  their deployer credential.

## History

Earlier embedded-UI improvements included affordances/deep links (`0d2f961`),
provider discovery and model preview (`2bed7fe`), and detail drawers
(`0da759e`, `97109e7`). This redesign supersedes the old Presets-first tab structure
and the standalone Workspace Pools tab, preserving their APIs and legacy links.
