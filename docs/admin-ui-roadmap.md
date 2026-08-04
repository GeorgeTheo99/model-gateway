# Admin UI Roadmap

Phased improvement plan for the embedded admin UI (`_ADMIN_HTML` in
`src/admin.py`). Constraints that apply to every phase:

- Single-file UI: HTML/CSS/JS embedded in `src/admin.py`. No build tooling.
- Design language per `PRODUCT.md`: calm, dense, single teal-blue accent,
  WCAG AA, reduced-motion friendly, no gradients/glassmorphism.
- Write endpoints stay gated behind `MODEL_GATEWAY_ADMIN_WRITES=true`
  (env-var gate; revisit at packaging time). Read-only mode
  (`body[data-writes="false"]`) hides all management controls.
- Deep links `#<tab>` and `#<tab>/<name>` must keep working across phases.

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Affordances, deep links, drill-downs, filters | ✅ `0d2f961` |
| 2 | Provider discover/validate + model preview write flows | ✅ `2bed7fe` |
| — | Detail drawer overlay (UX fix, replaced inline panels) | ✅ `0da759e`, `97109e7` |
| 3 | First-run wizard | Documented below; not started |
| 4 | Circuit / pool / federation visibility | Not started |

## Phase 3 — First-run wizard

### Problem

A fresh install (`./install.sh`) leaves a running gateway with no usable
providers: the bootstrap catalog is a starter stub, and setup requires
either CLI onboarding (`model-gateway onboard generate` → review draft →
apply) or prior knowledge of the admin Providers/Models forms. The admin
UI greets a new operator with empty tables and "N need config" warnings —
the opposite of "trustworthy at a glance".

### Scope

Guided setup flow inside the admin UI for the unconfigured state.

1. **Detection** — the UI recognizes "needs setup" (no ready providers or
   no routable models) and offers the wizard instead of empty tables.
   Backend: small addition to `/admin/api/status` (a `needs_setup`
   boolean or equivalent derived signal).
2. **Step 1: Add a provider** — protocol picker (OpenAI- or
   Anthropic-compatible), base URL, API key, with inline **Validate**
   reusing `POST /admin/api/providers/{id}/validate`
   (already returns `OK · 200 · N models`).
3. **Step 2: Discover & register models** — run
   `POST /admin/api/providers/{id}/discover`, present the model list,
   register selected models (reusing the Phase 2 register pre-fill flow
   plus `POST /admin/api/models/{name}` writes).
4. **Step 3: Verify routing** — `POST /admin/api/models/{name}/preview`
   (`will route`), show the client-facing base URL and an example `curl`.
5. **Exit** — wizard dismisses permanently once ≥1 provider is ready and
   ≥1 model routes; the normal dashboard takes over.

### Success criteria

- A fresh install can go from zero to one routable model entirely in the
  browser (writes enabled), with validation at each step.
- With writes disabled, the wizard degrades gracefully (see open
  questions) rather than presenting dead controls.
- No regressions: deep links, drawer, read-only mode, full test suite.

### Notes

- Nearly all backend endpoints already exist from Phase 2; this is
  primarily UI work plus the "needs setup" status signal.
- The existing right-side drawer could host the wizard steps.

### Open design questions (decide before building)

1. Presentation: full-screen takeover vs. dismissible banner + drawer
   flow.
2. Writes-off installs: show a read-only "run these CLI commands"
   version, or instructions for enabling `MODEL_GATEWAY_ADMIN_WRITES`
   for the setup session?
3. Re-entry: should "re-run wizard" stay accessible after first run
   (e.g., from the Debug tab)?

## Phase 4 — Circuit / pool / federation visibility

Surface runtime routing state that currently only exists in logs/config:
circuit-breaker states per provider, workspace pool health (see
`docs/workspace-pools-design.md`), and federation route status (see
`docs/federation.md`). Read-only; no write flows planned. Scope to be
detailed when Phase 3 ships.
