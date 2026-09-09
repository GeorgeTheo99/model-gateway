# Admin UI

The approved direction is documented in [admin-ui-design-brief.md](admin-ui-design-brief.md).
The UI remains embedded in `_ADMIN_HTML` in `src/admin.py`: no frontend build step,
external runtime assets, or framework dependency.

## Implemented structure

- **Overview**: actionable configuration/routing issues, first-connection guidance,
  inventory links, fixed 24-hour usage summary, and the latest five requests.
- **Models**: search and location/connection/capability filters; model details with
  client IDs, copyable request examples, pricing/capabilities, configured routing,
  image handling, and recent activity. Gateway-owned presets are a disclosure here.
- **Connections**: provider/local-runtime inventory, safe edit/discovery flows, and
  contextual routing groups (workspace pools). Workspace diagnostics are separate
  from the preference order; each workspace appears once in the group inventory.
- **Activity**: latest-50 request search/outcome filters and a separate time-windowed
  usage/cost view. Token accounting is progressively disclosed.
- **Settings & diagnostics**: secondary access to management-mode instructions,
  runtime metadata, and authenticated protocol diagnostics.

Legacy links remain supported: `#providers[/name]`, `#pools`, `#presets`,
`#usage[/request-id]`, and `#debug`. New primary links are `#overview`,
`#models[/name]`, `#connections[/name]`, and `#activity[/request-id]`.

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
  fallback, composite targets, and scoped image routing are not execution traces.
- Provider inventory checks read the upstream model list, not inference. Their results
  are session-local, not a persistent verification history.
- Requests contain the recorded route, not a guaranteed final failover workspace or
  full attempt chain. Missing evidence is labelled **Not recorded**. Duration is the
  recorded request latency, not TTFT or guaranteed stream-completion time.
- Request filters operate on the latest 50 records across all time. Detail histories
  are the latest 25, independent of their fixed 24-hour usage summaries.
- Known cost is a subtotal; accounting completeness and unknown/partial costs are
  shown separately. No admin or client credentials are embedded in copyable examples.

## Verification

Python API, metadata, and structural regressions:

```sh
uv run pytest -q -p no:cacheprovider
```

Browser behavior uses an isolated in-memory API with synthetic data and no production
credentials. Requires an installed Playwright package and its Chromium browser:

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
- Federation and consumer-private profile management require their own scope and
  authorization design; they are not silently added to the generic admin inventory.

## History

Earlier embedded-UI improvements included affordances/deep links (`0d2f961`),
provider discovery and model preview (`2bed7fe`), and detail drawers
(`0da759e`, `97109e7`). This redesign supersedes the old Presets-first tab structure
and the standalone Workspace Pools tab, preserving their APIs and legacy links.
