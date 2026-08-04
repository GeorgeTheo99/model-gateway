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
  and model-level fallback routes.
- **Admin dashboard** — health, provider/model inventory, usage and cost, and
  gated write controls at `/admin`.
- **Transactional onboarding** — `model-gateway onboard` discovers upstream
  models, writes a reviewable secret-free profile, stores credentials in
  0600 files, applies atomically, and rolls back on verification failure.
- **Usage/cost ledger** — tokens, provider cost, estimates, latency and
  errors per request, in SQLite.
- **Federation** — explicitly configured gateway-to-gateway routes.

## Requirements

- macOS (the bundled service manager uses launchd)
- [uv](https://docs.astral.sh/uv/), `git`, `curl`, `python3`

## Quick start

```bash
git clone <repo-url> model-gateway
cd model-gateway
./install.sh
```

The installer creates a config, bootstraps a starter model catalog
(`model-info.json`) if none exists, installs and starts a LaunchAgent, and
verifies `/health`. Then register your first provider and model:

```bash
model-gateway onboard generate \
  --provider example \
  --base-url https://api.example.com/v1 \
  --model example-model

# review the generated secret-free draft, then:
model-gateway onboard config/onboarding/drafts/example-example-model.yaml --dry-run
model-gateway onboard config/onboarding/drafts/example-example-model.yaml
```

Point any OpenAI- or Anthropic-compatible client at
`http://127.0.0.1:9111`.

Day-to-day management:

```bash
model-gateway status    # launchd state + /health probe
model-gateway logs -f   # follow the service log
model-gateway restart   # restart + verify
model-gateway update    # git pull + uv sync + restart + verify
```

## Configuration

Two layers (see `config/config.yaml.example` and `docs/`):

| Layer | File | Contents |
|---|---|---|
| Providers | `config/config.yaml` | Base URLs, credentials (or 0600 key-file refs), protocol, headers, quirks, pools, auth keys |
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
  writes additionally require `MODEL_GATEWAY_ADMIN_WRITES=true`.
- `config/config.yaml` and secret key files are kept at mode `0600`.
- Provider keys are never returned by the admin API or UI after save.

There is no bundled TLS, rate limiting, or multi-tenancy: run it on
loopback or behind your own reverse proxy on networks you trust.

## Documentation

| Doc | Contents |
|---|---|
| `docs/deployment.md` | Deployment layout and operations |
| `docs/deployment-auth.md` | Inbound auth configuration |
| `docs/provider-onboarding.md` | Provider/model onboarding flow |
| `docs/federation.md` | Gateway-to-gateway federation |
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

Proprietary, source-available. See [LICENSE](LICENSE) — viewing and
personal non-commercial evaluation permitted; redistribution, commercial
use, and derivative works require written permission.
