# Design Sketch: Workspace Pools, Catalog Generation, and Foolproof Workspace Replacement

Status: DRAFT (sketch only, nothing implemented)
Date: 2026-07-07

## Problem

1. **Single-workspace routing.** Every model binds to exactly one provider
   (`databricks` = AI Gateway workspace, `databricks-e2` = e2-demo-field-eng).
   If a workspace is deleted, rate-limited to death, or its endpoints are
   removed, those models hard-fail. `model_fallbacks` is a single-hop,
   same-catalog map and is the only cross-endpoint escape hatch.
2. **Catalog drift.** The same model list is hand-maintained in 4 places:
   gateway `config.yaml`, `isaac_manage/local_claude/model-aliases.json`
   (pi-list / claude-* / codex-* / pi-* launchers), and
   `pi-local/config/pi-models/models.json` (Pi `/model` picker), plus
   `model-info.json`. The fable incident (in gateway + aliases, missing from
   Pi catalog) is the canonical symptom.
3. **Workspace replacement is manual and undocumented.** Standing up a new
   workspace route today means editing YAML by hand, knowing about
   `auth_refresh`/`auth_profile`/`endpoint_style`/quirks, minting a token,
   restarting the gateway, and then separately fixing the two downstream
   catalogs.

## Goals

- A workspace outage or deletion degrades to "requests transparently route to
  the next workspace in the pool" for every Databricks FMAPI model.
- One command adds/replaces a workspace end-to-end (auth → config → validate
  → regenerate downstream catalogs → reload).
- `pi-list`, `pi-<alias>` launchers, and Pi `/model` are always generated,
  never hand-edited, and therefore always agree with the gateway.

## Non-goals

- Auto-exposing every serving endpoint on a workspace. The catalog stays
  curated (context/max_output/thinking metadata is not discoverable and wrong
  metadata = broken coding model). Discovery is used for *validation and
  suggestions*, not live exposure.
- Load balancing / cost-based routing. Pools are ordered failover only.

---

## 1. Config schema: workspace pools

### 1a. Workspaces become first-class, providers become thin

```yaml
# config.yaml (new shape — old shape stays supported, see Migration)
workspaces:
  fevm-model-exp:                  # PRIMARY. NOTE: the existing "AI Gateway"
    # provider URL 7474647777725369.ai-gateway.cloud.databricks.com is this
    # same workspace (org id 7474647777725369 = fevm-model-exp) reached via
    # its AI Gateway hostname — so "primary" is already fevm-model-exp today.
    base_url: https://7474647777725369.ai-gateway.cloud.databricks.com
    workspace_url: https://fevm-model-exp.cloud.databricks.com  # for auth/probes
    kind: ai-gateway               # ai-gateway | serving-invocations
    auth: oauth-cli                # pat | oauth-cli
    auth_profile: fevm-model-exp   # databricks CLI profile
    api_key: dapi...               # PAT, or last-known OAuth JWT
    path_prefixes: {anthropic: anthropic/v1, openai: mlflow/v1}
    quirks: [anthropic_bearer_auth]
  e2-demo:
    base_url: https://e2-demo-field-eng.cloud.databricks.com
    kind: serving-invocations
    auth: oauth-cli
    auth_profile: e2-demo-west     # databricks CLI profile
    api_key: eyJ...                # rotated in place by refresh
    quirks: [no_stream_options, no_reasoning_params]
  dogfood:
    base_url: https://adb-2548836972759138.18.azuredatabricks.net
    kind: serving-invocations
    auth: oauth-cli
    auth_profile: logfood

pools:
  # Ordered failover. Different models prefer different workspaces —
  # "across many workspaces" is the normal case, not the exception:
  default-pool:  [fevm-model-exp, e2-demo]   # sonnet/opus/gpt: primary first
  fable-pool:    [e2-demo, fevm-model-exp]   # fable lives on e2-demo
  glm-pool:      [dogfood]                   # GLM52 only exists on dogfood
```

Model→workspace affinity falls out of per-model `pool:` references: `fable`
→ `fable-pool`, `glm-5.2` → `glm-pool`, everything else → `default-pool`.
A pool may be a single workspace (glm-pool) when a model exists nowhere else
— it still gets the circuit breaker, per-workspace auth refresh, and the
interactive replacement flow below; it just has no automatic failover target.

Key insight that makes pooling cheap: **FMAPI pay-per-token endpoint names are
identical across workspaces** (`databricks-claude-sonnet-4-6`,
`databricks-gpt-5-5`, ... exist on every FMAPI-enabled workspace). So one
`provider_model_id` is valid against every member of a pool; only the host,
auth, and endpoint style differ — which is exactly what the workspace entry
captures.

### 1b. Models reference a pool (or a single workspace)

```yaml
models:
- name: claude-fable-5
  alias: fable
  provider_model_id: databricks-claude-fable-5
  pool: e2-first-pool            # NEW: ordered workspace failover
  protocol: anthropic
  context: 1000000
  max_output_tokens: 128000
  vision: true
- name: gpt-5.5
  alias: gpt
  provider_model_id: databricks-gpt-5-5
  pool: databricks-openai-pool
  fallback_model: databricks-gpt-5-4   # model-level fallback still exists,
                                       # consulted only after the whole pool fails
  ...
```

`provider:` (single workspace) remains valid for non-pooled things (google,
omlx). `pool:` and `provider:` are mutually exclusive.

### 1c. Failover semantics (routing algorithm)

Per request, in `resolve()` + `upstream.py`:

1. Take the pool's workspace list, skip any workspace whose circuit is OPEN
   (circuit breaker becomes **per-workspace**, which it effectively already is
   since circuits key on provider name).
2. Send to the first healthy workspace. Existing retry ladder applies
   (5xx x3, 429 x6 with backoff, transport x4), plus 401/403 → oauth-cli
   refresh for that workspace.
3. Workspace-level failover triggers — move to the next pool member — on:
   - retry ladder exhausted with 429/5xx (today this falls through to
     `model_fallbacks`; pool comes first now),
   - 404 model-not-found (endpoint deleted on that workspace),
   - DNS/TLS/connect errors (workspace deleted → NXDOMAIN routes here),
   - 401/403 *after* a failed refresh attempt,
   - circuit already OPEN (skip without sending).
4. Only after **all** pool members fail: consult `fallback_model` /
   `model_fallbacks` (different model, e.g. gpt-5.5 → gpt-5.4, resolved
   through its own pool).
5. Sticky preference (optional, phase 2): remember "workspace X is serving
   model M" for N minutes after a failover so every request doesn't re-probe
   the dead primary; the circuit breaker's probe loop already gives us most
   of this for free.

Failovers are logged + counted in the ledger (`failover_from`,
`failover_to`, reason) and surfaced on the admin dashboard.

### 1d. Auth per workspace

- `auth: oauth-cli` → generalize today's `refresh_oauth_token()`:
  preflight (<5 min JWT validity) + reactive on 401/403, using
  `databricks auth token --profile <auth_profile>`, persisting back to
  config.yaml. This already exists for e2; it just becomes per-workspace.
- `auth: pat` → static; validation warns if the PAT fails a probe.
- Browser SSO (dead refresh token) stays an interactive concern: the
  `workspace add/auth` CLI (below) and the zsh launcher preflight
  (`_ensure_e2_auth`, generalized to `_ensure_ws_auth <profile>`) own it.
  The gateway itself never blocks on a browser.

---

## 2. Catalog generation: gateway is the single source of truth

```
config.yaml (curated, hand-edited via admin UI or CLI)
        │
        ▼
GET /v1/models  ──(or direct YAML read)──►  scripts/export_catalogs.py
        │
        ├──► isaac_manage/local_claude/model-aliases.json   (pi-list, claude-*, codex-*, pi-*)
        ├──► pi-local/config/pi-models/models.json          (Pi /model via ~/.pi/agent symlink)
        └──► drift check (fails loudly if targets were hand-edited)
```

- **`scripts/export_catalogs.py`** renders both downstream files from the
  gateway catalog. Mapping rules:
  - gateway `protocol: anthropic` → Pi provider `databricks-anthropic`
    (anthropic-messages, `baseUrl http://localhost:9111`);
  - everything else Databricks → Pi provider `databricks`
    (openai-completions, `baseUrl http://localhost:9111/v1`);
  - google models → Pi provider `google`;
  - `alias` → `cloud:<name>` keys in model-aliases.json;
  - context/max_output/vision/thinking copied verbatim.
  Output files get a `"_generated": "by model-gateway export_catalogs — do not hand-edit"` header key.
- **Triggers:** run automatically on gateway start (`run.sh`), on
  `/admin/api/reload`, and from `manage.sh models sync`. `pi-list` reloads via
  the existing `_load_cloud_models` re-eval (`reload-cloud-models` alias);
  Pi picks the new catalog up next session (models.json is read at startup).
- **Drift check** (mirrors `aidk_drift_check.py`): compares generated files
  against a fresh render; wired into `manage.sh status` and the admin
  dashboard config-readiness panel.
- pi-launcher note: `_pi_cloud` currently derives Pi provider from the
  `cloud:` id prefix (`claude-*` → databricks-anthropic). Generation makes
  this explicit instead: model-aliases.json entries carry a
  `pi_provider` field the launcher reads directly. Fable's entry gets
  `pi_provider: databricks-anthropic` (it's a claude model behind an
  invocations endpoint — the prefix heuristic already handled it, but
  explicit beats implicit).

---

## 3. Foolproof workspace replacement

### 3a. Interactive workspace replacement on OAuth failure (paste-a-URL flow)

Requirement: when OAuth to an existing workspace fails unrecoverably, the
operator is **prompted to paste a new workspace URL** and the system rewires
itself. Two contexts:

**Interactive (zsh launcher — the main path).** Generalize `_ensure_e2_auth`
to `_ensure_ws_auth <workspace>` and run it for whichever workspaces the
requested model's pool needs (derived from the generated aliases file):

```
1. databricks auth token --profile <auth_profile>      # silent refresh
2. on failure → databricks auth login --host <workspace_url>   # browser SSO
3. on failure (workspace deleted / SSO dead / 404 host) →
     "Workspace '<name>' (<url>) is unreachable or auth is dead."
     "Paste a replacement workspace URL (or Enter to skip): "
4. paste https://new-ws.cloud.databricks.com[/?o=...]  →
     normalize URL → manage.sh workspace replace <name> --host <url>
     (auth → probe → coverage check → smoke test → config rewrite:
      the new workspace takes the dead one's place in every pool it was in
      → gateway reload → regenerate catalogs)
5. skip → launcher continues; pooled models degrade to remaining members,
     single-workspace pools (glm-pool) fail with a clear error.
```

The prompt happens in the **launcher/CLI**, never inside the gateway — the
gateway is headless and must not block requests on a human. The same prompt
is reachable on demand via `manage.sh workspace replace` (below).

**Headless (gateway runtime).** When refresh fails server-side, the workspace
circuit stays OPEN and the admin dashboard + `manage.sh status` show
`auth-dead: run 'manage.sh workspace replace <name>'`. Next launcher
invocation triggers the interactive flow in step 2–4 automatically, because
preflight consults circuit/auth state via `/admin/api/stats`.

### 3b. One command: `manage.sh workspace add` (and `replace`)

```
$ manage.sh workspace add fe-sandbox \
    --host https://fe-sandbox-theodem.cloud.databricks.com \
    --pools databricks-anthropic-pool,databricks-openai-pool \
    [--position 2] [--profile fe-sandbox-theodem] [--kind serving-invocations]
```

Steps (idempotent, each verified before the next):

1. **Auth**: ensure a CLI profile exists (`databricks auth login --host ...`
   → browser SSO if needed); mint a token; write the workspace entry with
   `auth: oauth-cli`.
2. **Probe**: `GET /api/2.0/serving-endpoints` on the new workspace; verify
   reachability + list endpoint names.
3. **Validate coverage**: for every catalog model in the requested pools,
   check `provider_model_id` exists on the new workspace. Print a coverage
   table (`fable ✓  sonnet ✓  gpt-5.5 ✗ (missing)`), and refuse to insert the
   workspace into a pool position where it can't serve the pool's models
   unless `--allow-partial`.
4. **Smoke test**: send one tiny real completion per protocol
   (anthropic + openai) through the new workspace directly.
5. **Commit**: write config.yaml, `POST /admin/api/reload` (no restart needed
   for config reads; restart only if the process predates pool support).
6. **Regenerate**: run `export_catalogs.py` + drift check.
7. **Report**: final table of pools with member order and per-model coverage.

`manage.sh workspace replace <old> --host <new-url>` = `add` with the new
workspace inheriting every pool membership (same positions) of the old one,
then the old entry is quarantined (kept in config for audit, excluded from
pools). This is the command the paste-a-URL prompt invokes.

`manage.sh workspace remove <name>` is the inverse of add: refuse if it would
leave any pool empty; otherwise remove, reload, regenerate.

`manage.sh workspace test <name>` = steps 2–4 only (use in cron/launchd for
early warning that a standby workspace has drifted).

URL normalization: accept pasted URLs in any of the common shapes —
`https://host/?o=123`, `https://host/browse/...`, bare `host` — strip path
and query, keep the `o=` org id for verification against
`/api/2.0/preview/scim/v2/Me` / workspace-conf probes.

### 3c. Deleted-workspace runbook (the failure you asked about)

If a routed workspace is deleted *right now*:

- **Automatic (already-configured pool):** DNS/connect errors trip that
  workspace's circuit after 3 failures; all pooled models silently route to
  the next member. Admin dashboard shows the circuit OPEN with reason. Zero
  user-visible failures if the pool has a healthy member. Probes keep testing
  every 10s and will never recover a deleted workspace — a `workspace
  quarantine` state (auto after N hours OPEN, or manual) stops probe noise.
- **Recovery (bring capacity back):** either the paste-a-URL prompt on next
  launcher use (3a), or explicitly `manage.sh workspace replace <dead> --host
  <new-url>`. Both handle SSO, validation, smoke test, config write, reload,
  and catalog regeneration in one pass.
- **Single-workspace pools (e.g. glm-pool/dogfood):** there is no failover
  target, so the paste-a-URL prompt is the *primary* recovery mechanism, and
  the coverage check pivots: instead of requiring the new workspace to serve
  all pool models, it reports which currently-orphaned models the new
  workspace can serve and wires only those.
- **Standby recommendation:** keep pools at 2 active members and validate a
  3rd cold-standby profile weekly via `workspace test` (candidates already in
  `~/.databrickscfg`: fe-sandbox-theodem, logfood, ai-devtools). Adding the
  standby to pools then requires no new auth ceremony.

### 3d. What makes it "foolproof"

- Auth, endpoint-name coverage, and a real completion are verified **before**
  the workspace is trusted with traffic.
- Pools can't be emptied; partial coverage requires an explicit flag.
- Downstream catalogs can't drift (generated + drift-checked).
- The zsh launcher's e2-specific auth hack (`_E2_BACKED_ALIASES=(fable)`)
  is replaced by generic per-workspace preflight derived from the generated
  aliases file (`ws_auth: <profile>` field), so a workspace swap does not
  require editing zshrc-launcher.zsh at all.

---

## 4. Migration & phasing

- **Back-compat:** `providers:` (current shape) keeps working; a `provider:`
  on a model is treated as a 1-member pool. Migration is
  `providers.databricks → workspaces.ai-gateway`, `providers.databricks-e2 →
  workspaces.e2-demo` plus two pool definitions — mechanical.
- **Phase 1 (biggest win, smallest diff):** catalog generation +
  drift check (kills the 4-way hand-maintenance; fable-class bugs impossible).
- **Phase 2:** workspace pools + per-workspace circuit/failover in
  `upstream.py`/`providers.py` (~ the existing `model_fallback` retry-wrapper
  pattern, applied one level up).
- **Phase 3:** `manage.sh workspace add/remove/test` + quarantine +
  admin-UI pool panel.

## Open questions

1. Pool membership per model class is now decided (primary fevm-model-exp;
   fable→e2-demo-first; GLM52→dogfood-only). Remaining: should default-pool
   get a 3rd cold-standby member, and does e2-demo/dogfood have FMAPI parity
   for the default models? `workspace test` output should confirm.
2. AI-gateway workspace (`ai-gateway` kind) uses different path shapes than
   `serving-invocations` — pooling across kinds is supported by design
   (resolve() already branches on endpoint_style per provider), but the first
   implementation could restrict pools to same-kind members for simplicity.
3. Should Pi's `/model` list pooled models once (gateway hides the pool —
   recommended) — vs. exposing per-workspace variants for debugging? Sketch
   assumes once; a `debug:` block in the generated Pi catalog could add
   `fable@e2-demo`-style pinned ids later.
4. `model-info.json` — fold into config.yaml as part of Phase 1 so there is
   exactly one curated file? (It duplicates thinking/vision metadata today.)
