# ADR 0003: Scoped Model Access and Lightweight Local Models

- **Status:** Accepted; implemented in 0.8.4 (phases 0-1) and 0.8.5 (phases 2-5)
- **Date:** 2026-10-23

## Context

Model Gateway should serve any model a user, or a program the owner builds,
needs, and stay lightweight: it should reach many local models without users
installing model files themselves, and a program that picks a specific model
should be able to use only that model.

Today:

- **Access.** A consumer credential is either profile-only or may call every
  direct model (`allow_direct_models`, `src/auth.py`). Profile-only credentials
  reach just their own namespace's profiles (ADR 0001). Every client sees the
  whole catalog in `/v1/models`, `/v1/models/canonical` and `/v1/debug/thinking`.
  Several steps can change the model a direct request reaches: retirement
  redirects, base-model reroutes inside `providers.resolve`, `model_fallbacks`
  (upstream ids, `src/model_fallback.py`), vision fallback (reroute or
  `extract_then_answer`, selectable by the caller with a header or body field)
  and federation forwarding.
- **Local AI.** The gateway-managed runtime serves one pinned model
  (`local_runtime.MODELS`, Qwen 3.8 27B 8-bit) through oMLX with
  `--max-concurrent-requests 1`. `install` refuses a model directory holding
  anything else, eligibility is one 48 GiB floor, and adding it (CLI, admin API
  or a `local_ai:manage` credential) ends by force-restarting the gateway
  (`launchctl kickstart -k`).
- **Federation.** One-hop peers are addressable only as `<peer>/<id>`
  (`federation.resolve_imported`); base-model resolution
  (`providers.resolve_base_model`, ADR 0002 R1) never considers them. A peer
  exports every available identifier, cloud routes and aliases included,
  without locality or `base_model`. Importers recompute the catalog digest over
  the fields they know (`federation.validate_catalog`). A forwarded request runs
  the peer's ordinary direct pipeline, fallbacks included, and peer URLs may be
  plain `http`.
- **Config lock.** `config_lock.lock_path` uses `tempfile.gettempdir()`, which
  is `/tmp` under launchd and `$TMPDIR` in a shell, so the CLI and the running
  gateway do not exclude each other.

oMLX already holds several models at once: an engine pool with LRU eviction, a
memory guard, and per-model `is_pinned` and `ttl_seconds` settings (checked in
the installed oMLX 0.7.1; the managed runtime pins 0.7.0).

## Decision

### 1. Per-credential model allowlists

A consumer credential may list the direct models it can call:

```yaml
auth:
  consumer_credentials:
    - id: weather-bot
      key_file: ...
      namespaces: [weather]
      permissions: [profiles:invoke]
      allow_direct_models: true
      models: [qwen3.8-27b, glm-5.3]
```

- `models` is optional and valid only with `allow_direct_models: true` and
  without `profiles:write`, `providers:manage` or `models:register`. A
  credential that could register profiles could route its own profiles
  anywhere, and one that could register models could reshape the names it
  lists; those scopes are trusted like the owner. Absent, direct access is
  unrestricted, as today. Present, it is a non-empty list of unique names.
- **What a name allows.** A listed base model allows requests by that
  base-model name, resolved as ADR 0002 describes; a route name is allowed only
  when it is listed itself, so a restricted program cannot pick a provider the
  owner's preference would not. Aliases and alternate ids count as the entry
  they name. Pack rows match only for credentials holding that pack's
  namespace, so one product cannot steer a name another product's credential
  lists.
- **When the check runs.** The requested identifier is mapped to its catalog
  entry, or imported row, before retirement redirect, resolution, federation
  forwarding or any availability-dependent error. A name that is not allowed
  is `403 direct_model_access_denied` whether or not it exists. Every later
  step that changes the model is checked against the entry resolution actually
  chose, not the request string:
  - retirement redirect: the successor must be allowed;
  - base-model reroute: allowed by the listed base model;
  - vision fallback (reroute and `extract_then_answer`): the helper model must
    be allowed, otherwise an image sent to a text-only model fails;
  - federation: the imported `<peer>/<id>` name, or under section 4 its base
    model, must be allowed.
- `model_fallbacks` is disabled for restricted credentials, as for profile
  requests. Its targets are upstream ids, not catalog models, so they cannot be
  checked by name. Pool failover keeps the model and is unaffected.
- Error bodies for restricted credentials never name other models.
- Profiles are unchanged: `models` does not apply to `profile:` selectors.
  A restricted credential's profiles are bounded by its namespace grants, and
  it cannot write them.
- Legacy shared keys, admin keys and anonymous access keep full access. ADR
  0001 already schedules legacy keys for removal; admin status now reports them
  as deprecated, along with `models` entries that match no catalog name or base
  model.
- The check lives in the helper all three inference endpoints call
  (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`); a test enumerates
  the app's routes so a new model-taking endpoint cannot skip it.

### 2. Listings for restricted credentials

For a credential with `models`, `/v1/models`, `/v1/models/canonical` (rows,
`auto_models`, `model_presets`) and `/v1/debug/thinking` list only entries
section 1 allows. A listed base model is listed under its base-model name,
with the capabilities of the route it currently resolves to; routes it does
not list are not shown. Imported federation rows appear only in `/v1/models`,
as today.

Every other caller, profile-only credentials included, sees the listing it
sees today. Products register profiles from the catalog (Home Server's
`model_gateway_inventory.py` and my-ai's `gateway_discovery.py` read
`/v1/models/canonical`), so hiding it from them would break registration.

The filter is applied per caller in the handlers, not in `_direct_model_rows`,
which also builds the federation export.

### 3. Several managed local models, curated

The release ships a curated set of local models (`local_runtime.MODELS` plus a
verified manifest per model in `local-models/`), each with its own
`min_memory_bytes`, disk size and a `base_model`.

- A model is eligible on Apple silicon with at least its own memory floor. The
  48 GiB global floor becomes Qwen's per-model value.
- `local-ai add <model>` and `local-ai remove <model>` work per model, with
  status per model. `start_add`'s "already serves" refusal and `check`'s
  single-model assertion become per-model. Removing the last model tears the
  runtime down, as `remove` does today. One setup job runs at a time.
- The first add builds the runtime as today. Adding or removing a model on a
  running runtime reloads the gateway's catalog and oMLX's model list without
  restarting either. oMLX 0.7.0 rescans on `POST /admin/api/reload` (after a
  login with its API key), which unloads its loaded models; the gateway polls a
  `catalog-generation` file the job bumps.
- Downloads stage outside oMLX's `--model-dir`, and the model directory
  invariant becomes "exactly the verified payloads of the installed models".
- `model_settings.json` lists every installed model. oMLX's engine pool and
  memory guard decide what stays loaded, except that models bound by a
  registered `local_only` or `trusted_local` profile on this node are pinned.
  A profile registration that would pin payloads larger than 75% of the Mac's
  memory is refused (`409 local_memory_exceeded`); an add pins nothing, since
  profiles bind only models already served.
- `--max-concurrent-requests 1` serializes every local model behind one slot.
  Kept: on the memory-bound Macs local AI targets, a second concurrent request
  grows the KV cache, and pins keep the models profiles depend on resident.
- The curated set is Qwen3.8-27B in 8-bit (48 GiB Macs) and 4-bit (32 GiB
  Macs); both have base model `qwen3.8-27b`, and `add` without a model installs
  the first this Mac can run.
- Each installed model loads as a shipped-layer row (origin profile
  `local-ai`), as ADR 0002 describes for the single model today.
- Installs keep the existing download rules: HTTPS only, resumable, hash
  verified, rate-limited (`DOWNLOAD_BYTES_PER_SECOND`, 150 Mbps), and a disk
  check of remaining bytes plus `DISK_MARGIN`.

### 4. Trusted-local peers serve before downloading

**Catalog schema 2.** Each exported model gains `locality` and `base_model`.
`locality` is `local` only when every pool member is loopback oMLX and the
row has no `model_fallbacks` target; otherwise `cloud`. Because importers
recompute the digest over the fields they know, the fields cannot be added to
schema 1: a schema 1 importer would reject the whole catalog. Importers ask
with `?schema_version=2` and accept either version; exporters keep revision
and digest state per schema, and the importer cache records each peer's
schema and resets its revision and digest baseline when that schema changes.
A schema 1 catalog is treated as all `cloud` with no base models.

**Trust.** The owner marks a peer as trusted for local work; such a peer's
`base_url` must be `https`:

```yaml
federation:
  peers:
    studio:
      base_url: https://...
      trusted_local: true
```

**The peer keeps the request local.** The importer picks the peer row by
`base_model` (else `direct_model_id`) and forwards that row's
`direct_model_id` to a local-only federation path that only schema 2 peers
serve, authenticated by the federation credential. A peer that has been
downgraded before the importer's next refresh therefore refuses it instead of
running it through its ordinary pipeline. On that path the peer serves only
from loopback oMLX, with no redirect, reroute, `model_fallbacks` or vision
fallback, and otherwise refuses with 409.

**Direct requests.** A curated local model name keeps local locality even
when it is not installed here: it resolves only to this node's oMLX, then a
healthy, fresh `local` row from a `trusted_local` peer (in peer configuration
order), then on-demand install (section 5), and never to a cloud route that
shares its base model. A plain base-model name follows ADR 0002's preference
on this node, then trusted-local peer rows.

**Profiles.** `local_only` keeps ADR 0001's meaning: this Mac, loopback oMLX.
A new profile locality, `trusted_local` (with `gateway_local` credentials),
also admits a `local` row of a `trusted_local` peer, so a product opts in
explicitly and the owner's peer flag never widens existing profiles. Bindings
follow ADR 0002 R1: the route available at registration is bound and stays
bound until the profile is re-registered.

- A peer row's binding records the peer node and the last-known-good exported
  row. A stale or unreachable peer is `503 profile_route_unavailable`; a
  changed row is `409 profile_binding_changed`.
- Profile selectors are still never forwarded; the peer receives a direct
  model call.

A peer that is not trusted keeps today's behaviour: addressable as
`<peer>/<id>`, never part of resolution or a profile closure.

### 5. On-demand install

The owner lists curated models that may install on first use:

```yaml
local_ai:
  auto_install: [qwen3.8-27b]
```

- On-demand install adds models to a runtime that exists. Building the
  runtime (the first add) is never request-triggered, and models are offered
  for on-demand install only once the runtime exists.
- A listed, eligible, not-installed model appears in `/v1/models/canonical`,
  filtered like every other row, with `"local_install": "on_demand"` only
  when the caller asks with `?include=local_install` (consumers such as my-ai
  reject unknown canonical fields). `/v1/models` keeps listing only available
  models.
- A direct request that resolves to it, with no local route and no trusted
  peer serving it, starts the setup job and returns immediately:
  `503 model_installing` with `Retry-After` estimated from bytes remaining and
  the download rate. No request is held open for a download.
- Requests read cached job state; apart from one deduplicated job start, they
  never run subprocesses. While a job for another model runs, the answer is
  `503 model_install_busy`. An ineligible model, one blocked by disk space,
  or one whose last job failed is `503 model_not_installable` with a reason;
  requests do not retry a failed job, the owner does. An interrupted job (its
  heartbeat went stale, for example across a reboot) is resumed by the next
  request, from its partial download.
- Requests can trigger downloads only of models the owner listed. Explicit
  adds remain with the owner and `local_ai:manage` credentials.
- Profiles bind only routes available at registration (section 4), so a
  product needing a not-yet-installed model names it in its pack `requires`
  and registers once the model is served.
- A listed model whose base model an installed model already serves is served
  by that model (section 4's order), so it installs on demand only for a
  request that names it and may not use the other model.

### 6. Open local models (owner opt-in)

With `local_ai.open_models: true`, a full admin may pin a Hugging Face MLX
repository at a 40-character revision.

- The gateway builds the manifest at pin time from the repository listing at
  that revision. LFS files carry Hugging Face's SHA-256. Non-LFS files are
  downloaded and checked against their git blob id before hashing. The
  manifest is stored outside the payload directory (as `model_payload`
  requires), and every later install and check verifies against it.
- Only weight, configuration, tokenizer and template files are accepted
  (`.safetensors`, `.json`, `.jinja`, `.txt`, tokenizer models). A repository
  with Python files, `auto_map` or `trust_remote_code` is refused. A manifest
  lists either `model.safetensors.index.json` and the shards it names (the
  curated rule) or exactly one `model.safetensors` and no index.
- The memory floor is derived from the payload size, and eligibility requires
  it as for curated models. The pin supplies catalog fields (context, output
  limit, thinking, vision, license). Its name may not collide with any
  identifier or base-model name, and it has no `base_model`, so a pin never
  joins base-model resolution or widens an allowlist.
- Pins are kept beside the runtime (`local-ai-pins/`), so removing local AI
  keeps them. Pinned models load beside the curated ones once installed (origin
  `local-ai-pin`), so a pin that is not installed routes nowhere, never to a
  route that shares its name as a base model. A later
  release catalog row that uses one of a pin's identifiers loses it to the pin,
  as an owner row would win (ADR 0002); a later release's own local AI model
  with the pin's name wins over the pin, whose payload `unpin` then removes.
- Packs cannot pin open models. They may name one in `requires`, which
  reports it as unmet until the owner pins it (ADR 0002).
- Pinned models then behave like curated models for eligibility, add, remove,
  `auto_install` and listings.

### 7. Config lock beside the config

`config_lock.lock_path` places the lock in `.locks/` (mode 0700) beside the
real path of `config.yaml`, keyed as today, and refuses a `.locks/` owned by
another user. Every process writing one config file resolves the same lock
whatever its environment, as the profile registry's lock beside its file
already does. Until no install can still run 0.8.3, writers hold the new lock and both legacy
locations (`/tmp/model-gateway-locks-<uid>/` and the per-user Darwin temp
directory), in that fixed order, so processes from either version exclude
each other during an upgrade whatever environment they started in.

## Rollout

Each phase ships and is verified on its own.

| Phase | Scope | Depends on |
|---|---|---|
| 0 | Config lock beside the config (section 7) | none |
| 1 | Allowlists and restricted listings (sections 1, 2) | none |
| 2 | Curated managed local models (section 3) | none |
| 3 | Federation schema 2, trusted-local peers, `trusted_local` profiles (section 4) | 2 for curated names |
| 4 | On-demand install (section 5) | 2; 3 for peer-first order |
| 5 | Open local models (section 6) | 2 |

- Phase 1 is backward compatible: no existing credential has `models`, and
  other callers' listings are unchanged. Its tests are adversarial: every
  model-changing step in section 1 for a restricted credential, aliases, pack
  rows, retired names, image requests and error bodies.
- Phase 2 verifies oMLX 0.7.0's multi-model behaviour and new-model discovery,
  or moves the pin, before shipping. This Mac's production oMLX is not
  gateway-managed and stays that way; it keeps its owner rows, and moving it
  under management is a separate, later decision.
- Phase 3 ships to importers before exporters are asked for schema 2; old
  peers keep working on schema 1. Trusted-local peers need HTTPS certificates.
- Home Server customers receive each phase through the next Home Server
  package build.

## Consequences

- A program with its own restricted credential can call and see only the
  models it lists. Its profiles stay bounded by namespace grants, which the
  owner controls by not giving it `profiles:write`.
- A program listing a model that is later retired breaks unless the retired
  name's successor is also listed; admin status reports unmatched entries.
- A Mac without the memory or disk for a model can use it from a trusted peer,
  and a Mac with room can fetch it on first use without the user installing
  anything.
- The managed runtime holds several large payloads. Disk use grows with what
  the owner installs or lists for `auto_install`.
- Local privacy grows a second class: `trusted_local` profiles may run on
  machines the owner trusts, while `local_only` stays on this Mac.
- Federation gains a negotiated schema version, and both versions must be
  supported until every peer upgrades.
- Open mode trusts Hugging Face's reported hashes once, at pin time. After
  that, files are always verified against the stored manifest.
