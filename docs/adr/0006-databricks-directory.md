# ADR 0006: Databricks Workspaces in the models.dev Directory

- **Status:** Accepted (owner, 2026-11-27)
- **Date:** 2026-11-27
- **Supersedes:** the "auto-exposing every serving endpoint" non-goal of the
  [workspace pools design](../workspace-pools-design.md)
- **Extends:** [ADR 0005](0005-models-dev-directory.md) (models.dev directory)

## Context

Databricks models were only ever hand-written `models:` entries. The pools design
ruled out exposing a workspace's models automatically because Databricks doesn't
report context, output or thinking metadata, and wrong metadata breaks coding
models.

models.dev now lists Databricks as a provider (`databricks`, 61 models). It gives
prices with long-context tiers, context and output limits, reasoning controls,
vision and the lab-level `canonical_model_id` (`anthropic/claude-opus-5-5`)
that every provider of a model shares. Each model also names its AI SDK
package, which identifies the one wire API some models need:
`@ai-sdk/anthropic` for Messages and `@ai-sdk/openai` for Responses. Without
a package, a model uses OpenAI Chat Completions. The metadata objection no
longer holds.

What models.dev cannot know is which of the owner's workspaces serves a model.
It also doesn't know whether a workspace serves the model under its legacy
serving-endpoint name (`databricks-gpt-6-2-sol`) or its Unity model-service name
(`system.ai.gpt-6-2-sol`).

## Decision

### Databricks is one directory provider

- `databricks` is a provider in `config/directory/corrections.yaml`. Its route
  suffix is `-dbx` and its display name is "Databricks".
- Its rows map like any directory row (ADR 0005): `claude-opus-5.5-dbx`,
  `base_model: claude-opus-5.5`, with the directory's price, limits and thinking.
  It has no single endpoint (`workspaces: true`).
- Followed lines load their current model automatically. Every other model is
  turned on one at a time (`catalog.directory.models`), as for every directory
  provider.
- Default ranking puts Databricks after the labs' own APIs and before OpenRouter.
  `catalog.provider_preference` overrides it.
- A row's wire API comes from the model's AI SDK package:
  - `@ai-sdk/anthropic` gives `protocol: anthropic`;
  - `@ai-sdk/openai` gives `api_style: open_responses`;
  - anything else uses OpenAI Chat.
- A row carries the model's legacy and Unity names in `alternate_ids`, so
  either name routes.

### A route is the pool of every workspace that serves the model

- A Databricks row's `pool` is the reserved `databricks:auto`. Its members are
  every configured Databricks workspace that serves the model, in config order
  (`src/databricks_models.py`). Existing pool failover, circuit breakers and
  per-workspace auth refresh apply unchanged.
- A provider counts as a Databricks workspace when it is named `databricks`,
  sets `auth_refresh: databricks-cli`, or has a `*.databricks.com` or
  `*.azuredatabricks.net` base URL. It must also be enabled and have a token.
- Each member route uses the name its own workspace lists the model by. The
  model.dev id is preferred, then the legacy name, then the Unity name.
- A workspace joins a model's pool only when its route can carry the model's
  wire API:
  - Messages needs an `anthropic` path prefix, or an invocations route that
    lists `anthropic` in `invocations_native_protocols`;
  - Responses needs a `responses` or `openai` prefix;
  - Chat works on any route.
- A row loads once some workspace lists its model. It becomes available
  (`no_serving_workspace` until then) when a member route has answered its test
  completion.
- A row whose model no workspace serves isn't offered. A bare base-model name
  then falls through to the next ranked provider.
- Pool failover during a request rewrites the body's `model` to the backup
  workspace's own name for the model. The ledger records that name too.

### Coverage: which workspace serves what

- Every workspace is asked which models it serves. All workspaces list their
  serving endpoints (`/api/2.0/serving-endpoints`). Unity routes (an
  `/ai-gateway` base with `path_prefixes`) also list Unity model services
  (`/api/2.1/unity-catalog/model-services`).
- The result is stored in `models-dev/databricks.json` beside `config.yaml`.
  The file holds model names, a fingerprint of each workspace's route, and test
  results. It never holds a token.
- If a workspace's route changes (its base URL, prefixes or style), its coverage
  no longer counts until the workspace is checked again.
- A removed workspace's coverage is deleted.
- Discovery runs:
  - when the gateway starts;
  - hourly for any workspace not checked in a day;
  - after `POST /admin/api/reload` (which `model-gateway workspace add` and
    `replace` call), for new or changed workspaces;
  - on every models.dev sync;
  - on `POST /admin/api/databricks/discover`.
- **Adding a model checks every workspace.** Turning directory models or lines
  on (`PUT /admin/api/directory/settings`) first re-checks every configured
  workspace, and so does adding an unmapped model.
- When coverage changes, the catalog reloads. If the catalog rejects the
  change, the previous coverage file is restored.

### Test completions per workspace

- Every member route answers one cheap completion before it joins the pool. The
  completion goes through the exact URL a request would use: path prefix,
  invocations URL or Responses URL, with the workspace's quirks.
- A route that fails is left out of that model's pool. After a day it is
  tested again, and it rejoins only if it answers. An explicit
  `POST /admin/api/databricks/discover` clears the workspace's failures at once.
- Before a test pass, workspace tokens are refreshed from the CLI cache. A
  workspace that refuses its token in a test gets no verdict on the route; it is
  tested again in the next pass.
- Directory rows on Databricks have no row-level test. A failing workspace
  removes only itself from the pool.

### Authentication in the background

- Background discovery uses the Databricks CLI's cached authentication only
  (`allow_login=False`), so it never opens a browser sign-in.
- If a workspace refuses its token, the gateway tries one cached refresh. If that
  fails, the workspace keeps the models it was last seen serving and is reported
  as `auth_needed` (`GET /admin/api/databricks`).
- Request-time auth (browser SSO when no backup workspace remains) is unchanged.

### Hand-curated Databricks rows keep working

- A directory row yields to an owner row that uses any of its identifiers (name,
  legacy or Unity name, alias). So an existing hand-written `models:` entry for
  a Databricks model keeps routing exactly as before, and the directory row
  isn't loaded.
- Pools in `pools:`, `available_model_ids` and every workspace CLI command are
  unchanged.

### Models the directory doesn't map

- These are never added automatically. `GET /admin/api/databricks` lists them
  under `unmapped`, with the workspaces serving each.
- `POST /admin/api/providers/databricks/discover` lists every model the
  workspaces serve. Each entry shows the workspaces serving it, the directory
  route that offers it (if any) and whether it is registered.
- `POST /admin/api/providers/databricks/models` adds one as an owner row
  (ADR 0005). Only an admin key may use this and the listing endpoint: their
  reach is every workspace, whatever a scoped credential's provider allowlist
  says. It works in these steps:
  1. Re-check every workspace.
  2. Test the model on each workspace that serves it, and refuse it if none
     answers.
  3. Write a row named `databricks-<model>` with `pool: databricks:auto`.

  Like any owner row it has no price or capability claims. Its pool follows
  coverage as workspaces start or stop serving it.

## Consequences

- An install with Databricks workspaces gets the directory's Databricks models
  automatically, priced and with correct limits. Each model fails over across
  every workspace that serves it, with no `models:` or `pools:` editing.
- Databricks requests for a reasoning model use the directory's thinking
  levels. How reasoning parameters are carried follows the existing rules: native
  Messages on an Anthropic route, `reasoning_effort` on OpenAI Chat.
- This was not verified against a live workspace when it was built. The tests
  mock the CLI and HTTP. The first live check should be `GET /admin/api/databricks`
  after `POST /admin/api/databricks/discover`.
- `/health` is unchanged.
