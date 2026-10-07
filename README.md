# Model Gateway

A self-hosted model router. One local endpoint in front of every model you
use — cloud providers (Anthropic, OpenAI, Google, OpenRouter, Fireworks, …)
and local oMLX models — speaking three client protocols with streaming
translation between them.

```text
  Clients
    ├─ OpenAI Chat:      POST /v1/chat/completions
    ├─ OpenAI Responses: POST /v1/responses
    └─ Anthropic:        POST /v1/messages
               │
               ▼
        auth + model resolution
               │
        policy / translation (reasoning · vision · tools)
               │
        retries · pools · circuit breakers · fallback
               │
               ▼
   cloud providers / local oMLX / federated gateways
               │
               ▼
   usage, cost, latency and error ledger
```

## Features

- **Three client protocols** — OpenAI Chat Completions, OpenAI Responses, and
  Anthropic Messages, including full SSE streaming translation in every
  direction.
- **Unified model catalog** — logical model names, aliases, capability
  metadata (context, vision, reasoning levels), and pricing in one registry.
- **Reliability** — per-provider retries, circuit breakers, provider pools,
  and model-level fallback routes. Pools try a backup on the first transient
  failure, reserving normal retries for the last healthy workspace before
  last-resort circuit-recovery probes.
- **Admin dashboard** — health, provider/model inventory, Databricks workspace
  pool routing, usage and cost, and gated write controls at `/admin`.
- **Transactional onboarding** — `model-gateway onboard` discovers upstream
  models, writes a reviewable secret-free profile, stores credentials in
  0600 files, applies atomically, and rolls back on verification failure.
- **Databricks workspace management** — `model-gateway workspace` validates
  OAuth, endpoint coverage, and a real smoke completion before changing an
  ordered workspace pool; failed activation rolls back automatically.
- **Usage/cost ledger** — tokens, provider cost, estimates, latency and
  errors per request, in SQLite.
- **Federation** — explicitly configured gateway-to-gateway routes.
- **Consumer profiles** — authenticated, namespace-scoped, immutable routing
  snapshots with enforced gateway-local or gateway-managed cloud execution.
  Consumer repositories own text/vision route and default policy; the gateway
  catalog does not.

## Requirements

- macOS (the bundled service manager uses launchd)
- [Homebrew](https://brew.sh), or for a git install: [uv](https://docs.astral.sh/uv/), `git`, `curl`, `python3`

## Quick start

```bash
brew install georgetheo99/tap/model-gateway
model-gateway install   # create config + empty catalog, start the LaunchAgent, verify /health
model-gateway admin     # copy the generated admin key and open the admin UI
```

In the admin UI, add a connection (base URL + API key), discover its models,
and register the ones you want. Point any OpenAI- or Anthropic-compatible
client at `http://127.0.0.1:9111`.

Config, catalog, secrets and the usage ledger live in
`~/Library/Application Support/model-gateway/`, so upgrades never touch them.
Upgrade with `brew upgrade model-gateway && model-gateway restart`.

To run from a git checkout instead (`model-gateway update` then pulls it):

```bash
git clone https://github.com/GeorgeTheo99/model-gateway.git
cd model-gateway
./install.sh
```

Providers can also be onboarded from the CLI:

```bash
model-gateway onboard generate \
  --provider example \
  --base-url https://api.example.com/v1 \
  --model example-model

# review the generated secret-free draft (path is printed), then:
model-gateway onboard <draft.yaml> --dry-run
model-gateway onboard <draft.yaml>
```

Day-to-day management:

```bash
model-gateway status    # launchd state + /health probe
model-gateway logs -f   # follow the service log
model-gateway restart   # restart + verify
model-gateway update    # git installs: git pull + uv sync + restart + verify
model-gateway workspace list                    # show workspace pool order
model-gateway workspace test <name>             # auth, coverage, smoke test
model-gateway workspace repair                  # repair dead OAuth/workspaces
```

**Add Databricks workspaces through the CLI**, on the gateway host. The admin
UI at `/admin#connections` shows routing groups and workspace diagnostics;
its generic **Add a connection** form does not configure Databricks OAuth or
workspace-pool membership.

```bash
model-gateway workspace list  # find the existing pool names
model-gateway workspace add backup-workspace \
  --host https://<workspace-host> \
  --profile <databricks-cli-profile> \
  --pools <existing-pool-a>,<existing-pool-b> \
  --style auto
model-gateway workspace test backup-workspace
```

Omitting `--position` appends the workspace as a backup. Supply `--pools`:
without it, the workspace is registered but is not added to any routing pool.
This registers an existing Databricks workspace; it does not provision one.
Authentication may open browser SSO, and validation makes small real inference
requests. The command activates the change, so it can restart the gateway.

For a workspace **already registered** in the gateway, attach it without
recreating or overwriting its connection settings:

```bash
model-gateway workspace pool add-member fable-pool dogfood --dry-run
model-gateway workspace pool add-member fable-pool dogfood
```

This appends one backup to an existing nonempty pool; repeating it is a no-op.
To create a new pool instead, use `workspace pool create` (below).
It uses the live gateway's model catalog and routing rules, rejects disabled or
wire-incompatible members, and tests **every enabled pool model** through the
candidate's configured route. Missing models or failed probes leave config
unchanged. Provider settings and other pools are preserved. `--dry-run` still
performs authentication and small billable inference probes, but never writes
config or restarts the gateway. Existing OAuth profile/host selection and
`auth_login: false` are respected. Unset shell provider routing/credential
overrides before using this command: preflight must validate the saved
connection, not unrelated credentials inherited by an interactive shell.

To give a model its **own failover pool**, register any extra routes under new
names, then create the pool and bind the model in one step:

```bash
# Optional: a second route to an already-registered workspace, e.g. to match
# the model's current wire format (existing entries are left untouched).
model-gateway workspace add opus55-e2-west \
  --host https://e2-demo-west.cloud.databricks.com \
  --profile e2-demo-west-ws --style invocations

model-gateway workspace pool create opus55-pool \
  --members opus55-fevm,opus55-e2-west,opus55-e2,dogfood-invocations \
  --model opus55 --dry-run
# Then run the same command without --dry-run.
```

The first member is the primary. `--model` accepts a name, alias, or upstream
ID and may be repeated. Only routing fields change; model metadata stays in
place, and a catalog-only model gets a minimal overlay entry. Each member must
resolve to the model's current upstream ID and protocol/API style, so existing
client sessions keep working. A protocol-changing member is rejected with
instructions to register a compatible route. Coverage accepts either legacy
serving endpoints or Unity Catalog model services, and every member receives a
small real completion through its exact route. Re-running an applied command is
a no-op; a model already in another pool or a same-named pool with different
members is rejected.

An admin **read** key (`auth.admin_keys` or `MODEL_GATEWAY_ADMIN_KEY`) is required
to verify the gateway's config identity and live pool order/readiness (plus, for
`pool create`, that the pool serves the bound models). The CLI
uses restart activation even when admin API writes are disabled. Activation or
live-verification failure restores the backup and reactivates the previous
config, unless a concurrent edit makes automatic rollback unsafe. Concurrent
config edits during preflight cause the command to abort without writing.

Mutating Databricks workspace commands are similarly available as
`workspace add`, `workspace replace`, and `workspace remove`. They verify the
candidate before writing `config.yaml`, restart and health-check the gateway,
and restore the prior configuration if activation fails. Run
`model-gateway workspace --help` for the full syntax. For
`--style ai-gateway`, pass the **workspace URL** as `--host`; the routed
`<org-id>.ai-gateway.cloud.databricks.com` base URL is derived from the
workspace token (an explicit gateway hostname is still accepted).

`--host` must be the workspace URL from your browser (`https://…databricks.com`,
`?o=` and paths are stripped); a bare workspace ID is rejected with guidance.
The smoke test runs through the **exact runtime route** that will be committed.
`--style auto` (on `add` and `replace`) tries the derived AI Gateway route with
a real completion and falls back to direct `/serving-endpoints/…/invocations`
when that host/path is unusable, printing the reason. `--allow-partial`
accepts a workspace that lacks some catalog models, but only when a bootstrap
model answers through the runtime route; the served IDs are recorded on the
provider as `available_model_ids`, and `scripts/export_catalogs.py` omits the
rest so downstream launchers (`pi-list`) never advertise dead models.

### Failover timing

When another configured, resolvable pool member with a closed circuit and
compatible declared protocol/API style remains, streaming and non-streaming
requests move to it on the first transient HTTP/transport failure, missing
endpoint (404), or auth rejection after one cached-token refresh attempt.
They do not wait through same-workspace backoff, an open circuit's recovery,
browser SSO, or another in-progress OAuth refresh. Connection setup is capped
at 5 seconds on these attempts (an already-shorter timeout is respected).
The last healthy member retains its normal retries before any last-resort
open-circuit probes; single-workspace routes retain their original policy.

Failover swaps URL/credentials, not request-body formats. For example, an
OpenAI-only invocations endpoint is skipped for a prepared native Anthropic
request. Pool members still need compatible model capabilities and provider
quirks; automatic cross-protocol body translation is not part of pool failover.
Discarded streaming error responses are closed without draining their bodies.

This is **not a total request deadline**: cached-token CLI calls can take up
to 30 seconds, and existing read/write timeouts are preserved for large prompts
and slow reasoning. Non-streaming HTTP calls still buffer the upstream response.
A successful stream is never replayed on a backup after it has started.
Model-level fallback remains after workspace failover.

## Configuration

Two layers, both in `~/Library/Application Support/model-gateway/` (installs
that predate 0.2 keep their checkout-local `config/config.yaml` and
`model-info.json`). See `config/config.yaml.example` and `docs/`:

| Layer | File | Contents |
|---|---|---|
| Providers | `config.yaml` | Base URLs, credentials (or 0600 key-file refs), protocol, headers, quirks, pools, auth keys, consumer principals |
| Model catalog | `model-info.json` (+ optional `models:` overlay in config) | Gateway model IDs, aliases, upstream IDs, context/output limits, capabilities, pricing, fallbacks |

A model is exposed only when its referenced provider is configured and
usable. Both files are managed for you by the onboarding flow and the admin
API; direct edits are also supported.

## Security defaults

- Binds to `127.0.0.1` by default.
- Binding to a non-loopback host **refuses to start** unless `/v1` client
  keys are configured (override for trusted private networks with
  `MODEL_GATEWAY_ALLOW_UNAUTHENTICATED_NONLOCAL=true`).
- `/admin/api` fails closed unless `MODEL_GATEWAY_ADMIN_KEY` is set; admin
  writes additionally require `MODEL_GATEWAY_ADMIN_WRITES=true`, which
  a fresh `model-gateway install` sets unless installed with
  `MODEL_GATEWAY_ADMIN_WRITES=false`.
- `config.yaml` and secret key files are kept at mode `0600`.
- Provider keys are never returned by the admin API or UI after save.
- Consumer profile APIs require identity-aware credentials; legacy, admin,
  anonymous, and federation credentials cannot read, register, or invoke them.
- A consumer principal without `allow_direct_models: true` cannot bypass its
  profiles through an ordinary direct model invocation.

There is no bundled TLS or rate limiting. Consumer profile namespaces provide
an authorization boundary, but ordinary direct routes are not general-purpose
multi-tenant isolation; run the gateway on loopback or behind a trusted proxy.

## Documentation

| Doc | Contents |
|---|---|
| `docs/deployment.md` | Deployment layout and operations |
| `docs/unity-gateway.md` | Unity Gateway model-service routing and migration checks |
| `docs/deployment-auth.md` | Inbound auth configuration |
| `docs/provider-onboarding.md` | Provider/model onboarding flow |
| `docs/workspace-pools-design.md` | Databricks workspace pools and operations |
| `docs/federation.md` | Gateway-to-gateway federation |
| `docs/consumer-profiles.md` | Consumer credential, snapshot API, and profile execution contract |
| `docs/adr/0001-consumer-profile-security.md` | Consumer-profile security boundary decision |
| `docs/productionization-plan.md` | Admin/control-plane roadmap |
| `docs/admin-ui-roadmap.md` | Admin UI phases (first-run wizard, runtime visibility) |
| `PRODUCT.md` | Product definition and design principles |

## Development

```bash
uv sync
uv run python -m pytest -q     # test suite
uv run python -m src.main      # run in the foreground
```

## Status & support

Single-operator, self-hosted software in active development. No support
policy or compatibility guarantees yet; interfaces may change between
commits.

## License

[Apache-2.0](LICENSE).
