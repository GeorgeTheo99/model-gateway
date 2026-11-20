# ADR 0005: models.dev Directory, Provider Ranking and Follow-Latest

- **Status:** Accepted (owner, 2026-11-19); built in stages
- **Date:** 2026-11-19
- **Supersedes:** the 2026-10-16 decision that new models arrive only in gateway
  releases, and the tabled [model updates](../model-updates.md) proposal.
- **Extends:** [ADR 0002](0002-layered-model-catalog.md) (layered catalog)

## Context

Every cloud model the gateway offers is hand-written: curated onboarding profiles
(`config/onboarding/*.yaml`) plus owner rows. Adding Google's Gemini 4 fleet,
retiring Gemini 3.x and fixing three long-prompt price tiers took an afternoon of
reading pricing pages and testing quirks by hand, and new models reach installs
only in a release.

[models.dev](https://models.dev) is an MIT-licensed, community-maintained
directory (about 10,000 models from 324 providers, updated many times a day)
that Pi itself builds its model list from. Its `api.json` has, per provider and
model: prices including long-context tiers, context and output limits,
modalities, tool support, reasoning controls (`reasoning_options`: a `toggle`
means reasoning can be turned off, `effort` lists the levels), `release_date`,
`status` (`alpha`, `beta`, `deprecated`), `family` and a lab-level
`canonical_model_id` shared by every provider that serves the model. Checked
against what we found by testing: its reasoning controls agree on every current
route (no off switch for Gemini 4, Claude Sonnet 5.5 or GPT-6.1 Sol; Haiku 5.5
and GPT-6 Luna can turn reasoning off).

It does not know, and the gateway must keep: data retention (OpenRouter
zero-retention routing), extended cache-write prices (Anthropic 1-hour, OpenAI
12-hour), Fireworks spot routes, output caps of the routes we actually use (Google
Vertex via OpenRouter caps output at 64K; models.dev lists the 1M AI Studio limit),
request quirks, and our established names and aliases.

## Decision

### Directory (source of facts)

- The gateway syncs `https://models.dev/api.json` (the stable v1 API) once a day
  and on demand (`POST /admin/api/directory/sync`). No LLM runs on the gateway.
- Only providers the gateway can map are kept: `anthropic`, `openai`,
  `openrouter`, `fireworks-ai`→`fireworks`, `moonshotai`→`moonshot`, `zhipuai`
  (open.bigmodel.cn), `google`, `deepseek`, `xai`. Specialised model types,
  models without text output, and `deprecated` models are not offered.
- A sync is accepted whole or not at all. It is rejected (the last good snapshot
  stays) when it does not parse, when a mapped provider that had models has none,
  or when more than 20% of the mapped models disappear at once.
- `MODEL_GATEWAY_DIRECTORY=off` turns the directory layer off (the test suite does).
- The accepted snapshot is stored as `directory/snapshot.json` beside
  `config.yaml`; every release also ships a filtered snapshot
  (`config/directory/models-dev-snapshot.json`) used until the first sync
  succeeds, so an offline install works.

### Mapping a directory model to a catalog row

| Row field | From |
|---|---|
| `provider`, `provider_model_id` | mapped provider id, models.dev model id |
| `base_model` | slug of the lab-level model name (`Claude Opus 5.5` → `claude-opus-5.5`); the lab model id when two names collide |
| `name` | `base_model` on the lab's own API; otherwise `base_model` + the provider's route suffix (`-or`, `-fw`) |
| `display_name` | `<model name> (<provider display name>)`, the existing convention |
| `context`, `max_output_tokens` | `limit.context`, `limit.output` (capped by corrections) |
| `vision` | `image` in `modalities.input` |
| `thinking`, `thinking_levels` | none without `reasoning`; `optional` (with `off`) when `reasoning_options` has a `toggle` or the effort values include `none`; else `always`. Levels are the effort values mapped onto the gateway's levels |
| `pricing` | `cost.input/output/cache_read/cache_write`; one `cost.tiers` context tier becomes `long_context`. Extended cache-write prices come from corrections. `cost.reasoning` and audio prices are not used (reasoning is billed as output) |
| `family` | the lab id (`anthropic`, `google`, …), as today |

A model without `cost.input` and `cost.output` is not offered: an unknown price
is never treated as free.

### Corrections (gateway-owned facts)

`config/directory/corrections.yaml`, shipped in releases, holds:

- **Per provider:** route suffix, display name, `thinking_format`, quirks (for
  example `openrouter_zdr` with `zdr: true` on every OpenRouter row), which
  directory cache-write classes to drop (OpenRouter's Google `cache_write` is not a
  per-token rate), and extended cache-write multipliers per lab (Anthropic 1-hour
  writes are 2× input; OpenAI 12-hour writes are 2.5× input where a 30-minute write
  is priced).
- **Per model** (keyed by provider and models.dev id, or a lab-model prefix): the
  established `name`, `alias` and `display_name` of models we already serve, output
  caps, quirks, and adaptive-thinking flags.

Corrections win over directory fields; owner rows still win over both (ADR 0002).

### Model lines and follow-latest

- A **line** is a lab plus the version-free words of the model name: `Claude Opus
  5.5` and `Claude Opus 4.8` are line `anthropic/claude opus`; `Gemini 4 Neon` is
  `google/gemini neon`. A line's newest model (by `release_date`, ignoring
  preview, experimental, `-latest` and `alpha`/`beta` models) is its current model.
- **Followed lines** load their current model on every ready, ranked provider that
  serves it (a lower-ranked route whose upstream id is a loaded route's name or alias is
  left out). The default follows every line of the featured labs (Anthropic,
  OpenAI, Google, Zhipu, Moonshot, DeepSeek, xAI) whose current model is under six
  months old, except special-purpose lines (cyber, coding, realtime, mini and similar;
  `featured` in the corrections). The owner follows or unfollows lines and turns individual models on in
  Settings (`catalog.directory.lines`, `catalog.directory.models`).
- When a line's current model changes, the previous one is **retired** with
  `replaced_by` its successor (ADR 0002 R3 redirects), unless it is pinned
  (`catalog.pinned`) or a registered consumer profile binds it; then it keeps
  loading and admin status reports the successor.

### Provider ranking and policy

- `catalog.provider_preference` (ADR 0002 R1) is the ranking. Its default is the
  lab's own API first (Anthropic; the owner's ChatGPT plan, then the OpenAI key
  (owner, 2026-11-19); Google), then Fireworks, Moonshot, Zhipu, then OpenRouter.
- A bare model name (`gpt-6.1-sol`, `claude-opus-5.5`) asks for the model: it is
  served by the first ready provider in the ranking that serves it, and falls back
  to the next. Each bare-named route also answers to a provider-suffixed id
  (`gpt-6.1-sol-openai`, `claude-opus-5.5-anthropic`) that never changes provider,
  as other route names never do.
- The ChatGPT plan's models come from models.dev's experimental v2 API
  (`openai-codex`, fetched with each sync; when it fails the plan keeps the models
  it had). The plan serves Chat Completions and Responses (not Anthropic Messages),
  so bare names resolve to it only there, and only for callers allowed to use it. When the plan answers a bare name
  with a usage limit, the request runs again on the next provider and bare names
  skip the plan for 15 minutes. The plan gets no test completion: its sign-in is
  its availability.
- `catalog.provider_policy` holds per-provider rules. `openrouter:
  {zero_retention_only: true}` (the default, as today) adds `openrouter_zdr` to every
  OpenRouter directory row.
- Turning a provider off (existing provider `enabled: false`) removes its routes.

### Holds and smoke tests

- A sync that would change a loaded model's input or output price by more than 50%,
  remove a loaded model, or drop its price holds that change: the model keeps its
  previous values until the owner approves or rejects the change in Settings.
- A model newly loaded from the directory gets one cheap completion straight to its
  provider, with the route's request quirks (the onboarding key check). A failure
  marks it unavailable (`smoke_test_failed`, with the provider's reason) and is
  retried after the next sync. Results are kept in `models-dev/state.json`.
- A rejected change stays rejected while models.dev proposes the same entry; a
  different proposal opens a new hold. A change models.dev withdraws closes its hold.

### Settings and API

The gateway owns the settings and serves them over the admin API; the admin UI and
My AI's Settings page both use it:

| Endpoint | Purpose |
|---|---|
| `GET /admin/api/directory` | Sync status, providers with ranking and policy, lines, held changes |
| `GET /admin/api/directory/models` | Browse directory models (filters: provider, lab, line, search) with their mapped rows and whether they load |
| `POST /admin/api/directory/sync` | Sync now |
| `PUT /admin/api/directory/settings` | Ranking, policies, followed lines, turned-on models |
| `POST /admin/api/directory/holds/{id}` | Approve or reject a held change |

### Scale

The directory is data, not catalog rows: only followed lines and turned-on models
load (about 50 rows by default), which keeps `/v1/models/canonical` inside My AI's
512-row and Pi's 1,000-row limits.

## Migration

- The curated profiles' models become directory rows: corrections give them their
  established names, aliases and display names, so routes, aliases, ledger
  history and consumer bindings keep their identity. A curated row for a route the
  directory maps yields to the directory row, so only routes models.dev cannot
  know (the ChatGPT plan, Fireworks spot) are still served from a profile.
- The profiles stay as onboarding manifests: their provider blocks drive key
  entry and the key check (a test completion on the first model), and consumers
  find the profile that serves their routes by its model names (My AI's Cloud
  Best setup). Their model lists are therefore kept, not removed as first
  planned; their prices no longer reach the catalog for directory routes.
- Owner rows are untouched; a directory row whose name an owner row claims yields
  as shipped rows do.
- Lines the owner already retired by hand are unfollowed (Gemini 3.x Flash and
  Flash-Lite, 2026-11-18).

## Stages

1. Directory sync, snapshot, mapping and corrections; a read-only browse endpoint.
2. Directory layer in the catalog with provider ranking and policy; curated models
   move onto it.
3. Follow-latest retirement, holds and smoke tests.
4. Admin Settings: providers and the model browser.
5. My AI Settings and Pi shortcuts on the same API (`catalog:manage`, generated
   Pi shortcuts); release.

## Consequences

- New models and price changes reach every install within a day, without a
  release; corrections and the code that handles new request shapes still ship in
  releases.
- The gateway trusts a community source for prices. Holds, smoke tests and the
  provider-reported cost (preferred by the ledger where available) bound the damage
  of a wrong entry.
- The gateway makes one outbound request a day to models.dev.
