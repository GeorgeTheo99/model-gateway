# Provider onboarding

OpenAI-compatible providers and replacement models can be added with a
secret-free YAML profile instead of hand-editing runtime files. Profiles may be
written by hand or generated as reviewable drafts.

## Generate a draft

For a simple OpenAI-compatible provider, start with only the provider, HTTPS
base URL, and exact upstream model ID:

```bash
model-gateway onboard generate \
  --provider moonshot \
  --base-url https://api.moonshot.ai/v1 \
  --model kimi-k3
```

Generation is non-destructive. By default it attempts read-only `/models`
discovery and saves:

```text
config/onboarding/drafts/moonshot-kimi-k3.yaml
```

The file is never overwritten unless `--force` is explicit. Use `--output
PATH` to select another path, or `--stdout` for inspection. `--stdout` cannot
be combined with `--apply`: every applied generated profile must first exist as
a reviewable file. Output parents are canonicalized through existing symlinks,
and draft output may never collide with the runtime config, live/source model
catalog, the new provider's secret destination, or any existing configured
provider secret file—even with `--force`.

Discovery uses, in order, an explicitly configured environment/Keychain
secret, an existing provider secret whose configured canonical base URL
matches the requested URL, or a hidden interactive prompt after an
unauthenticated endpoint rejects access. A secret is never reused for a changed
endpoint. Generation never stores a newly entered secret. Use `--no-discover`
for an offline minimal draft or
`--require-discovery` when an inconclusive result must fail the command.

### Evidence and unknown fields

Generated profiles remain schema version 1. A top-level `provenance` block
records categorical evidence for generated fields without leaking into model
catalog rows:

```yaml
provenance:
  generator: model-gateway onboard generate
  status: needs_review
  fields:
    /provider/base_url:
      source: operator
      confidence: confirmed
    /models/0/provider_model_id:
      source: provider_models
      confidence: verified
  discovery:
    source: provider_models
    status: verified
    http_status: 200
  unresolved:
    - alias
    - context
    - max_output_tokens
    - thinking
    - vision
    - quirks
    - pricing
```

Sources are `operator`, `provider_models`, `documentation`, `probe`,
`existing_catalog`, or `deterministic_default`. Confidence is categorical
(`verified`, `confirmed`, `observed`, or `default`), never a guessed score.
Provider responses, credentials, authorization headers, prompts, and completion
content are not stored.

Only harmless structure is derived automatically: schema/profile IDs, the
OpenAI protocol, the secret filename, and a gateway model name equal to the
upstream ID. Aliases, limits, thinking behavior, vision, quirks, pricing,
descriptions, and retirement are omitted unless supplied explicitly.

Examples:

```bash
model-gateway onboard generate \
  --provider example \
  --base-url https://api.example.com/v1 \
  --model example-model \
  --alias example \
  --context 128000 \
  --max-output-tokens 16000 \
  --thinking optional \
  --vision \
  --quirk use_max_completion_tokens \
  --pricing-json '{"input": 1.0, "output": 3.0}' \
  --description 'Example model via Example Provider'
```

To attribute an explicit value to documentation, cite the field directly:

```bash
--context 128000 \
--documented context=https://docs.example.com/models/example-model
```

`--docs HTTPS_URL` records a general supporting citation without assigning it
to a field.

### Optional probes

Probes are never automatic because they may consume tokens and can produce
false conclusions. Request only the behavior needed:

```bash
--probe text
--probe tools
--probe vision
--probe reasoning
```

Interactive use asks before sending probes. Non-interactive use requires
`--yes` and a non-prompt credential source. A probe records only its kind,
model ID, timestamp, HTTP status, and outcome. Success demonstrates that exact
request only; it does not automatically add capabilities or request quirks to
the profile.

### Existing models

Applying a model row replaces the prior catalog row. The generator therefore
detects every existing non-structural field—including unknown catalog metadata
—and records any field the draft would remove. Apply is blocked until those
removals are approved. The draft records canonical fingerprints of every
added/replaced and retired catalog row. The transaction engine recomputes the
actual diff and fingerprints while holding the configuration lock; if any
field name or value changed after generation, apply stops and requires a
regenerated draft rather than trusting stale provenance. The same actual-diff
metadata-removal gate also protects hand-written profiles, so deleting the
provenance block cannot bypass it.

Preserve current metadata explicitly with:

```bash
--preserve-existing-metadata
```

Individual preserved fields can still be intentionally dropped:

```bash
--preserve-existing-metadata --drop-existing-metadata vision
```

A provider change may make old capabilities or quirks invalid, so preserved
values remain visibly annotated as `existing_catalog` evidence and must be
reviewed.

## Apply a profile

A reviewed generated draft and a hand-written profile use the same command:

```bash
model-gateway onboard config/onboarding/drafts/example-model.yaml --dry-run
model-gateway onboard config/onboarding/drafts/example-model.yaml
```

Generation can also apply its saved draft in one invocation:

```bash
model-gateway onboard generate ... --apply
```

`--apply` always saves and reloads the draft first, shows/requests the generated
profile's safety decisions, then passes that same loaded schema-v1 mapping to
the existing transactional `apply_profile()` engine. The engine:

1. writes the key outside the checkout to
   `~/.config/model-gateway/secrets/<provider>.api-key` with mode `0600`, after
   rejecting any canonical target already owned by another configured provider;
2. adds the provider by `api_key_file` reference;
3. atomically replaces added and explicitly retired models in
   `model-info.json`;
4. mirrors the catalog to `MODEL_GATEWAY_MODEL_INFO_SOURCE`;
5. restarts the service, checks configured exports, and verifies health and
   model presence.

Configuration writers share one lock. Every touched file is snapshotted in
memory. If a handled write, service reload, catalog-export, or health
verification error occurs, the command restores those snapshots and reloads
the previous configuration. Abrupt process or machine termination remains
outside this rollback guarantee.

Profiles are idempotent: rerunning one updates the same provider/model without
requiring an already retired model to remain present.

## Explicit retirement

Retirement accepts exact gateway model names only:

```bash
--retire-model example-model-old
```

The generated profile contains:

```yaml
retire:
  models:
    - example-model-old
```

There is no wildcard, prefix, alias-family, or automatic older-version match.
Interactive apply confirms the exact set separately. Non-interactive apply
requires matching `--confirm-retire MODEL` arguments. These exact confirmations
apply to generated and hand-written profiles; the YAML retirement list remains
the source of intent. If any profile changes an existing model row, apply also
requires an exact `--confirm-replace MODEL` acknowledgement. If replacement
omits current metadata, dry-run lists `metadata_removals` and real apply also
requires `--allow-metadata-removal`. Exact already-applied model rows remain
idempotent and require no replacement acknowledgement.

## Inconclusive discovery

Generation still saves a reviewable draft when `/models` is absent, malformed,
rate-limited, unavailable, or temporarily fails. Model identity remains
unverified and apply retains the normal strict upstream check.

A valid `/models` list that omits the requested model is a conflict, not an
inconclusive result. Authentication failures also remain hard failures for
apply. A successful explicitly requested text probe can provide narrower model
existence evidence when a list endpoint is incomplete. Probe success requires
a valid OpenAI completion shape with non-empty content; the tools probe must
return exactly one `type=function` call to `probe_ok` with arguments exactly
`{}`.

For providers that genuinely lack a usable list endpoint, generated-profile
apply may use:

```bash
--allow-inconclusive-model-check
```

This narrow override is accepted only when the saved provenance shows a real
inconclusive endpoint attempt or successful text probes. `--no-discover` alone
is not overrideable. An authentication failure is always a hard stop, even if
stored probe evidence claims success. A valid model-list conflict requires a
successful targeted text probe. The CLI exposes no broad upstream-skip option.

## Non-interactive and CI use

`--non-interactive` never prompts or falls back to hidden input. Generation may
save a draft with unknown fields, but probes require `--yes`. Apply requires:

- `--yes`;
- an existing configured secret, `--api-key-env NAME`, or Keychain source;
- `--allow-metadata-removal` when the actual catalog diff lists removals;
- one matching `--confirm-replace MODEL` per actual existing-row replacement;
- one matching `--confirm-retire MODEL` per profile retirement;
- the narrow discovery override only when strict `/models` validation is
  intentionally unavailable.

For reproducible CI, generate and review a profile once, commit the secret-free
profile, and apply that fixed file. Avoid regenerating against live provider
metadata on every deployment. If CI intentionally generates, use an explicit
`--output`, `--non-interactive`, `--require-discovery`, and explicit metadata
flags so unresolved or drifting provider data fails visibly.

## Supplying secrets safely

Never put a key directly in a command argument. The default hidden prompt works
without Keychain. For non-interactive automation, refer to an environment
variable:

```bash
model-gateway onboard PROFILE --api-key-env PROVIDER_API_KEY
```

An existing macOS Keychain item may be read with
`--api-key-keychain-service SERVICE`. Keychain creation can fail with `User
interaction is not allowed` in SSH/headless sessions; interactive use falls
back to the hidden prompt.

## Hand-written profile format

Complex providers and stable CI workflows may continue using hand-written
profiles in `config/onboarding/*.yaml`:

```yaml
schema_version: 1
id: example-model
provider:
  id: example
  base_url: https://api.example.com/v1
  protocol: openai
  secret_name: example.api-key
models:
  - name: example-model
    provider: example
    provider_model_id: example-model
    alias: example
    context: 128000
    max_output_tokens: 16000
    thinking: optional
    vision: false
    quirks: [no_stream_options]
retire:
  models: [example-model-old]
```

Supported reusable request quirks are:

- `no_stream_options`
- `no_reasoning_params`
- `reasoning_none_with_tools`
- `force_reasoning_effort_max`
- `use_max_completion_tokens`
- `drop_fixed_sampling_fields`
- `named_tool_choice_as_required`
- `inline_image_urls_only`

Provider-specific request code should only be added when the upstream cannot
be described by these generic compatibility flags.
