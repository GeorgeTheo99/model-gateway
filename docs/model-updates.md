# Model updates (proposed)

> **Status:** proposed and tabled 2026-10-22; nothing is built. Reviewed with GPT-6 Astra and
> GLM 5.3. Companion proposal: Myai's coding jobs (`my-ai/docs/coding-jobs.md`), whose weekly
> model scout would produce this feature's curated models.

An opt-in feature that tells a gateway's owner which models are new, which are being replaced,
and which local models fit their Mac, and applies the owner's choices.

## Owner decisions

| Topic | Decision |
| --- | --- |
| Shipping | New curated models still ship in gateway releases (2026-10-16). No separate model feed; if release lag becomes a problem, a signed feed needs signatures, expiry, anti-replay and revocation |
| Opt-in | Off until the owner turns it on, per install. Deterministic: no LLM runs on the gateway |
| Who decides | Each app owns its routes. Adding a gateway route never changes what an app uses |
| Follow newest | Off by default for cloud and local. Apps offer a one-time "switch?" prompt showing price, capability and retention changes |
| Clients | One HTTP API. The admin UI, the CLI, Home Server and Myai all call it; apps never run the CLI |
| Updating the gateway | No updater of its own. Home Server's updater owns component installs; Homebrew installs update through Homebrew |

## What it reports

- **New curated models** from the installed release's onboarding profiles and `local-models/`
  manifests.
- **Not reviewed:** models the owner's keyed providers now list (`/models`) that the catalog
  doesn't have, with retention shown as unknown. Listing a model doesn't prove access, price,
  compatibility or retention.
- **Replacements and deprecations** (below).
- **Local fit** for curated manifests: weights + KV cache for the model's context + peak load +
  models already loaded + staging disk, against RAM and free disk; and whether the installed
  oMLX supports the model's `model_type`.

## Replacements and deprecation

- Replacement links are curated catalog fields, never derived from `family`/`base_model`: the
  route they replace, the workloads they suit (text, vision, tools, voice), required
  capabilities, minimum gateway/oMLX versions, and the evidence. Chains (A→B→C) are followed with
  a cycle guard. An unknown successor is a no-op, never an error.
- A replaced route gets `deprecated_at` and `retire_after`. Within the window the gateway keeps
  serving it while any active profile references it, and reports it so apps can prompt. An
  upstream that removes a model early triggers emergency retirement.

## Applying

- Cloud: with the opt-in on, curated models are added for keyed providers (as `auto_add` does
  today, with offered names recorded so deletes stick).
- Local: opt-in, within download and storage budgets the owner sets, within the gateway's download rate cap. The
  previous model stays until the new one passes a smoke completion through the gateway.
- Apply takes a plan with an immutable id and preconditions, is idempotent, is recorded for
  audit, and names the gateway instance it targets.
- `model-gateway update` and install verification hold if a release would drop a model a
  registered profile still references.

## API, permissions, CLI, UI

| Piece | Proposal |
| --- | --- |
| `GET /admin/api/model-updates` | The report above |
| `POST /admin/api/model-updates` | Apply a plan |
| Permissions | `models:updates:read`, `models:updates:apply`, bounded by the credential's provider allowlist; local installs still need `local_ai:manage`, profile changes `profiles:write` |
| CLI | `model-gateway models updates`, `apply <plan>`, `settings` |
| Admin UI | A Model updates tab |

## App changes

- **Home Server:** read deprecations and offer the switch prompt. Needs real migration work:
  `CLOUD_MODEL` is hardcoded and an existing profile returns early
  (`home-automation/src/api/settings_gateway.py`).
- **Myai:** the same prompt and re-snapshot flow in Settings.

## Related: a `chatgpt` provider

A provider using OpenAI's Sign in with ChatGPT (open-source, locally hosted apps; Responses API
with `store: false`) would give every gateway client GPT from the owner's ChatGPT plan. Pi itself
already has the same flow: its `openai` provider's Sign in with ChatGPT (Pi 1.0.3).

## Out of scope

LLM judgment or web research on end-user installs, cross-provider pooling, a gateway self-updater.
