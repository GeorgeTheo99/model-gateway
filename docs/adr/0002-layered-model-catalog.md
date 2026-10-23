# ADR 0002: Layered Model Catalog

- **Status:** Accepted
- **Date:** 2026-10-22

## Context

The catalog is one flat list. Curated (`auto_add`) profiles are **copied** into the
owner's `model-info.json` the first time a model is offered; after that the gateway
"never retires, replaces, or moves anything". On the production install, 16 of 45
rows are identical copies of shipped rows, one has drifted, and nothing records
which rows the owner wrote. Shipped fixes (pricing, context, thinking levels,
retirements) never reach existing installs, and Home Server ships its own private
`model-info.json` beside the gateway's.

Owners and programs must be able to add, change, and remove models, and still
`git pull` / `brew upgrade` and receive shipped model updates.

## Decision

### Layers

| Layer | Owner | Source | On update |
|---|---|---|---|
| Shipped | Model Gateway releases | `config/onboarding/*.yaml` with `auto_add: true` | Replaced wholesale |
| Consumer packs | My AI, Home Server, any program | Per-namespace snapshots registered over the admin API | Replaced when the product re-registers |
| Owner | User, admin UI, scoped API callers | `model-info.json` `llm` rows, config `models:` overlay, config `catalog:` | Never touched |

The effective catalog is computed on every load: shipped, then packs, then owner
rows, then the config `models:` overlay (whose existing semantics are unchanged).
Nothing from a lower layer is copied into a higher one.

### Owner wins, field by field

- An owner row whose `name` is a shipped model is an **override**: it is merged
  over the shipped row (`{**shipped, **owner}`), so fields it does not set keep
  receiving shipped updates.
- An owner row that changes where the model routes (`provider`, `pool`,
  `provider_model_id`, `omlx_id`) is the owner's own model and **replaces** the
  shipped row; nothing is merged into it. Admin saves store it whole.
- An override a release invalidates leaves the model **unavailable**
  (`override_invalid`) instead of serving the shipped row.
- Admin writes store overrides sparsely, with `shipped_base` recording the shipped
  value each overridden field replaced. The effective entry reports
  `origin.stale_overrides` when the shipped value has since changed
  ("update available under your override").
- A legacy full copy of a shipped row merges to the same result, so installs load
  before migration.
- A sparse override (with `shipped_base` or `review`) whose shipped row is not
  currently loaded is ignored, never promoted to a standalone model.

### Identifiers (R6)

Resolution is per identifier. An owner row may claim a shipped row's `alias` or
`alternate_ids`; the shipped row loses only that identifier, including when it
is merged under an override. Claiming a shipped row's `name` or upstream id
replaces the whole shipped row. Collisions inside one layer remain an error;
shipped collisions are a release bug caught by tests.

### Which shipped models load

- `catalog.shipped` in config: `all` (default), `none`, or a list of profile ids
  and model names.
- With `all`, a shipped model loads when its profile's provider is turned on,
  uses the profile's endpoint and protocol, and has a key — the same rule
  `auto_add` used. A release therefore **auto-enables** new shipped models.
- With `none` or a list, only the named shipped models load; new shipped models
  are not enabled. Named models are still references, so their fixes and
  retirements still apply. Pack `requires` are reported, never loaded for it.
- Deleting a shipped model adds it to `catalog.hidden`; the replaced
  `auto-added-models.json` ledger is migrated into it.

### Retirement (R3)

- A shipped profile retires names with `retire.models` and may map each to a
  successor with `retire.replaced_by: {old: new}`.
- A retired name is hidden from every listing and export unless it is in
  `catalog.pinned`. Retirement and `catalog.hidden` apply to shipped and pack
  models only: an owner's own row under a retired name keeps working and is listed
  in admin status (`retired_owner_rows`). Saving a model with a retired name
  through the admin API pins it; create-only registration cannot reuse a hidden
  or retired name.
- A request for a retired, unpinned name is redirected to its `replaced_by` when
  that model is available, with `Deprecation: true` and a
  `X-Model-Gateway-Model-Redirect` header. Without a replacement it is not found.

### Consumers reference base models (R1)

Provider-suffixed names (`glm-5.3-fw`, `gpt-6-astra-or`) are routes, not
contracts. Consumers reference a `base_model` (`glm-5.3`, `gpt-6-astra`); the
gateway resolves it to an available route, preferring the owner's
`catalog.provider_preference` order, then catalog order.

- A route name is never rerouted to another provider (that could change
  retention or cost); only a base-model name is, including a route named after
  its base model when that route is unavailable. A model the owner turned off is
  not rerouted. Requests are rewritten to the concrete route, so pricing, the
  ledger and pool failover see the model used. Pack routes never answer a
  base-model name.
- A consumer profile route may name a base model. Registration binds it to the
  route available then, within the profile's locality, and stores that route, so
  ADR 0001's snapshot binding is unchanged; re-registering picks up a new
  preference.

### Consumer packs

A pack is a per-namespace snapshot (`schema_version`, `namespace`,
`source_revision`, `models`, `requires`) registered with
`PUT /v1/catalog/packs/{namespace}` by a consumer credential holding
`profiles:write` on that namespace (a product's routing policy and the models it
needs ship together). It is stored in `catalog-packs.json` beside `config.yaml`
and replaced whole on each registration.

- Pack models may set only route, capability, pricing and label fields; fields
  that reach other consumers (`system_instruction`, `quirks`, `pool`,
  `api_style`, ...) stay with the owner.
- They cannot reuse an identifier or base-model name of any shipped model
  (loaded here or not), of the owner's catalog (except an identical copy, as an
  older install's import left), or of another pack (409).
- A release that later ships a pack model's identifier wins; the pack model
  yields instead of failing the catalog.
- The owner can override or hide pack models like shipped ones. The startup
  migration never treats pack rows as shipped, so a pack cannot remove owner rows. `requires` lists shipped models or base models the pack's profiles
need; unmet requirements are reported (`unmet_requires` on the pack and in admin
status), never forced on — the owner still wins.

### Failure handling (R7)

An invalid owner layer never silently falls back to shipped-only:

- Admin writes and `/admin/api/reload` validate strictly and roll back, as before.
- Any other reload whose catalog fails keeps serving the last good models with
  the newly loaded config (credential and provider changes are never reverted;
  a config that does not parse still fails) and reports `catalog.load_error`.
  The last good models are part of each registry snapshot, so a candidate an
  admin write rejected never becomes them.
- An override a release invalidates (for example a thinking level the shipped row
  no longer allows) is skipped for that model only; the shipped row loads and
  `origin.override_error` / `catalog.override_errors` say why.
- Startup still refuses an invalid catalog: there is no earlier registry to keep.

Admin status also lists overrides awaiting review (resolved with
`POST /admin/api/models/{name}/review` `{"keep": bool}`), stale overrides, and
retirements.

### Pi

`pi-catalog` owns only the providers in Pi's `models.json` that point at the
gateway, and preserves other providers and the owner's `modelOverrides`.
pi-shared's native-provider defaults (Codex context ceilings) already fill only
absent values.

## Migration

Run once at startup, under the config write lock, with a backup:

1. Rows identical to a shipped row are removed from `model-info.json`.
2. Differing rows become sparse overrides of only the differing fields, marked
   `review: true` and listed in admin status (R5). Their origin is unknown — an
   owner edit or a stale copy — so they are neither silently pinned nor dropped;
   the owner keeps or discards each one.
3. Ledger names absent from the catalog become `catalog.hidden`; the ledger is
   then retired.

## Consequences

- Shipped fixes, new models and retirements reach every install on update, with
  owner choices preserved.
- Admin writers and conflict checks must operate on the effective catalog, not
  the raw owner file.
- Registered consumer profiles bind the resolved route; a shipped change to a
  bound field (upstream id, endpoint, vision) returns `409 profile_binding_changed`
  until the consumer re-registers, as today.
- Home Server stops shipping a private catalog and registers a pack.
