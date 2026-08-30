# ADR 0001: Consumer-Owned Profiles as a Gateway Security Boundary

- **Status:** Accepted
- **Date:** 2026-08-27

## Context

Home Automation, my-ai, and Pi currently share model-gateway credentials and can address the same catalog. The gateway authenticates possession of a key but does not establish a consumer identity, authorize a profile namespace, or enforce locality across every routing transition. Its canonical policy API also omits local preset policy, which leaves consumers reading or duplicating gateway catalog files.

This means `ha/*`, `myai/*`, and `pi/*` are naming conventions, not security boundaries. It also means a request described as local-only could escape through a later provider pool, model fallback, global vision fallback, or composite dependency.

## Decision

### Ownership

- Consumer repositories own version-controlled policy manifests for their namespace.
- model-gateway validates, registers, versions, authorizes, publishes, and executes registered profiles.
- model-gateway continues to own canonical model inventory, provider health, retries, execution, and usage accounting.
- Consumer intent and privacy classification remain outside model-gateway.

### Identity and authorization

- Each consumer uses a distinct credential mapped by the gateway to a consumer principal.
- Principals carry explicit namespace and permission grants.
- A consumer credential cannot read, register, or invoke another consumer's profiles.
- Direct catalog invocation is a separate permission. HA runtime credentials will not receive it.
- Legacy shared keys remain temporarily supported only for non-profile routes during staged migration. They cannot access profiles.

### Profile and federation namespaces

Profile resource IDs remain consumer namespaced, for example `ha/automatic-local`. Execution uses the reserved selector `profile:ha/automatic-local`.

The `profile:` prefix is intentional. Federation already owns the unprefixed `<owner-node>/<direct-model-id>` syntax, so a profile selector can never be interpreted as a federated model route. Profile selectors are resolved before ordinary catalog and federation lookup and are never forwarded to peers.

### Locality and credentials

A profile declares both:

- `locality`: `local_only` or `cloud_explicit`
- `credential_policy`: `gateway_local`, `gateway_managed`, or `consumer_byok`

The gateway executes `local_only` + `gateway_local` and `cloud_explicit` + `gateway_managed`. Gateway-managed cloud execution resolves the registered canonical route with the gateway's provider credential; the inbound consumer credential is never forwarded upstream. `cloud_explicit` + `consumer_byok` remains representable but fails closed until an explicit credential-reference contract exists.

Locality is derived from the authenticated principal and registered profile. Request headers and body fields cannot weaken it.

For a local-only profile, the gateway validates the complete route closure: selected text/vision models, provider pools, composite dependencies, and configured model fallbacks. Every reachable provider must be local oMLX. Runtime checks remain in force after registration. Global vision fallback is disabled for profile execution; a profile must name a native vision model or an explicitly local composite.

Ordinary retries may repeat an already-authorized route. A retry, pool transition, model fallback, vision route, or composite leg may not change locality or credential class. Cross-model fallback remains disabled for profile requests rather than allowing an unproven transition. Same-route pool failover is permitted only when every candidate remains within the profile's validated locality and credential class.

### Versioning and degraded reads

The gateway stores immutable, server-versioned namespace snapshots. Registration replaces one namespace atomically and uses ETag preconditions. Read APIs return ETags and private cache directives. Consumers keep their own last-known-good snapshot and may use it during a bounded gateway discovery outage; they must not invent policy or read gateway files directly.

A cached profile snapshot is discovery metadata, not offline execution authority. Inference still requires the gateway, which re-authorizes the current registered profile on every request.

### Defaults

There is no gateway-wide routing default suitable for every consumer. In particular, a global `default_scope: cloud` cannot govern HA. Each consumer manifest owns its semantic default within its namespace. HA will register a local-only automatic default; my-ai and Pi retain their own policy.

## Consequences

### Positive

- HA local-only behavior becomes enforceable rather than advisory.
- Consumer profile updates cannot affect another namespace.
- Consumers can discover local profiles through an authenticated API with stale-cache support.
- Gateway catalog and provider behavior remain centralized without moving consumer intent/privacy policy into the gateway.
- Existing direct routes remain available as a rollback path during migration.

### Costs

- Gateway authentication becomes identity-aware.
- Profile registration and snapshot persistence become operational state requiring backup.
- Catalog changes that alter a registered profile's route closure can fail closed until the profile is re-registered.
- Separate runtime credentials must be provisioned and rotated for each consumer.

## Rollout

1. Deploy identity, profile registry/API, and local-only enforcement without configuring consumer principals.
2. Provision separate principals and register consumer manifests while legacy traffic is unchanged.
3. Adversarially verify namespace and locality boundaries.
4. Migrate HA behind its existing ownership flags while retaining direct oMLX as a disabled rollback path.
5. Migrate my-ai to profile/catalog APIs with a bounded last-known-good cache.
6. Observe each consumer independently.
7. Remove legacy shared keys, direct HA oMLX, duplicated `best-local` catalogs, and rollback flags only after each dependent consumer completes migration. Gateway-owned auto/preset policy has been removed; registered consumer snapshots are authoritative.

For `my-ai`, the profile-only compatibility floor is commit `4c17be5` plus a complete three-profile snapshot. A rollback below that floor is coupled: restore the prior gateway `auto_models`/`model_presets` catalog blocks and a compatible two-profile registry snapshot before starting the older consumer binary. Current binaries roll back with the complete snapshot and do not require gateway-owned policy.
