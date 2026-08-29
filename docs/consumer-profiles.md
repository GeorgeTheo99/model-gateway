# Consumer Profile Contract

This document defines the version 1 contract for consumer-owned, gateway-enforced routing profiles. See [ADR 0001](adr/0001-consumer-profile-security.md) for the ownership and security decision.

## Identifiers

- Consumer and namespace IDs: lowercase `a-z`, digits, and `-`; 1–32 characters.
- Profile resource ID: `<namespace>/<name>`, for example `ha/automatic-local`.
- Execution selector: `profile:<namespace>/<name>`.
- Profile names use lowercase `a-z`, digits, and `-`; 1–64 characters.
- The `profile:` selector is reserved and never participates in ordinary model or federation resolution.

## Consumer credentials

Identity-aware credentials are configured under `auth.consumer_credentials`. Secret values should be stored in owner-only regular files.

```yaml
auth:
  # Existing shared keys remain migration-only and cannot access profiles.
  client_keys: []
  consumer_credentials:
    - id: ha-runtime
      consumer: ha
      key_file: secrets/ha-runtime.key
      namespaces: [ha]
      permissions: [profiles:read, profiles:invoke]
      allow_direct_models: false
    - id: ha-deployer
      consumer: ha
      key_file: secrets/ha-deployer.key
      namespaces: [ha]
      permissions: [profiles:read, profiles:write]
      allow_direct_models: false
```

Rules:

- Credential IDs and consumer IDs are not secrets.
- Prefer `key_file`; inline `key` exists only for tests and portable configuration.
- `key_file` must be a non-symlinked regular file with mode `0600`, bounded to 64 KiB.
- A token may belong to only one credential class/principal.
- Profile APIs require an identity-aware consumer credential. Legacy client keys, unauthenticated loopback access, admin keys, and federation peer keys cannot access consumer profiles.
- A principal may operate only on namespaces listed in its credential record.
- `allow_direct_models` controls ordinary catalog routes separately from profiles.

## Manifest schema

Registration replaces one consumer namespace atomically.

```json
{
  "schema_version": 1,
  "namespace": "ha",
  "source_revision": "home-server@<git-sha>",
  "default_profile": "ha/automatic-local",
  "profiles": [
    {
      "id": "ha/automatic-local",
      "description": "Automatic local Home Automation route",
      "locality": "local_only",
      "credential_policy": "gateway_local",
      "protocols": ["openai_chat", "openai_responses", "anthropic_messages"],
      "routes": {
        "text": "glm-5.2-4.5bit",
        "vision": "best-local"
      },
      "defaults": {
        "temperature": 0.2,
        "max_output_tokens": 4096,
        "reasoning_effort": "off"
      }
    }
  ]
}
```

### Required fields

- Root: `schema_version`, `namespace`, `source_revision`, `profiles`.
- Profile: `id`, `locality`, `credential_policy`, `protocols`, `routes`.
- `routes.text` is required. `routes.vision` is optional; image requests fail closed when absent.

### Allowed values

- `schema_version`: `1`.
- `locality`: `local_only`, `cloud_explicit`.
- `credential_policy`: `gateway_local`, `gateway_managed`, `consumer_byok`.
- `protocols`: one or more of `openai_chat`, `openai_responses`, `anthropic_messages`.

Valid combinations are:

- `local_only` + `gateway_local`
- `cloud_explicit` + `gateway_managed`
- `cloud_explicit` + `consumer_byok`

`local_only` + `gateway_local` and `cloud_explicit` + `gateway_managed` are executable. `cloud_explicit` + `consumer_byok` remains a contracted, non-executable capability and returns `403 profile_execution_unavailable` until the gateway provides an explicit BYOK credential-reference contract. Other combinations are rejected with `422`.

### Route validation

- Route targets must be canonical gateway model names, not aliases, provider-native IDs, profile selectors, or federated IDs.
- A `local_only` route must have a complete closure containing only oMLX providers whose effective endpoint is trusted loopback (`localhost` or a loopback IP). Merely naming a remote provider `omlx` is rejected.
- Provider pools must not mix local and cloud candidates.
- Every dependency of a `local_only` route must resolve exclusively through trusted local oMLX providers.
- Every dependency of a `cloud_explicit` route must exclude oMLX and resolve through gateway-configured cloud providers.
- Process-wide/global vision fallback is never used for a profile request.
- Cross-model fallback is disabled for profile execution. Same-route transport retry and provider-pool failover are allowed only within the registered route's validated locality closure.
- The gateway records a digest of the validated route closure, including the effective non-secret endpoint identity and protocol. If catalog/config drift changes that closure, invocation fails closed until re-registration. Re-registering the unchanged manifest creates a new bound version when the closure changed.

### Defaults

Defaults apply only when the request omitted an equivalent explicit control. The gateway never overwrites an explicit caller value. Supported defaults are:

- `temperature`
- `max_output_tokens`
- `reasoning_effort`

Protocol translation maps `max_output_tokens` to the native request field without changing the semantic limit.

## Registry storage

Configure the machine-local durable registry under `profiles.registry_path` (a
relative path resolves beside `config.yaml`) or with
`MODEL_GATEWAY_PROFILE_REGISTRY`. The gateway writes immutable versions using
an owner-only lock and atomic replace/fsync. Both the registry and its `.lock`
file are operational state: back them up and never commit them.

```yaml
profiles:
  registry_path: consumer-profiles-registry.json
```

## Snapshot APIs

All endpoints require an identity-aware consumer credential and namespace authorization.

### Register latest namespace snapshot

`PUT /v1/profiles/{namespace}/snapshot`

- First registration requires `If-None-Match: *`.
- Updates require `If-Match: "<current-etag>"`.
- Request bodies are strict JSON (duplicate keys and non-finite values are rejected) and limited to 1 MiB.
- Missing precondition: `428 Precondition Required`.
- ETag mismatch: `412 Precondition Failed`.
- Identical canonical content is idempotent and returns the current snapshot/version.
- Successful mutation returns `200` and `Cache-Control: no-store`.
- Namespace history and the shared registry have bounded storage quotas; an over-quota update is rejected without replacing the last-known-good registry.

### Read latest snapshot

`GET /v1/profiles/{namespace}/snapshot`

- Honors `If-None-Match` with `304`.
- Returns `ETag`, `Vary` for every accepted credential header, and:
  `Cache-Control: private, max-age=60, stale-if-error=86400`.

### Read immutable version

`GET /v1/profiles/{namespace}/snapshot/{gateway_version}`

- Honors `If-None-Match` with `304`.
- Returns `Cache-Control: private, max-age=31536000, immutable`.

### Response shape

```json
{
  "schema_version": 1,
  "namespace": "ha",
  "source_revision": "home-server@<git-sha>",
  "gateway_version": 3,
  "registered_at": "2026-08-27T00:00:00Z",
  "default_profile": "ha/automatic-local",
  "profiles": [
    {
      "id": "ha/automatic-local",
      "locality": "local_only",
      "credential_policy": "gateway_local",
      "executable": true,
      "protocols": ["openai_chat", "openai_responses", "anthropic_messages"],
      "routes": {"text": "glm-5.2-4.5bit", "vision": "best-local"},
      "defaults": {}
    }
  ]
}
```

Snapshots never expose tokens, provider credentials, provider URLs, filesystem paths, provider-native IDs, or internal route-binding material.

## Execution

Clients invoke a profile by placing its execution selector in the normal `model` field:

```json
{"model": "profile:ha/automatic-local", "messages": [{"role": "user", "content": "..."}]}
```

Request flow:

1. Authenticate the consumer principal.
2. Parse the reserved profile selector.
3. Authorize `profiles:invoke` and the selector namespace.
4. Load the latest registered snapshot.
5. Verify protocol permission and executable credential/locality policy.
6. Select text or vision route from the actual request modality.
7. Revalidate the route binding and defaults against the live catalog/config.
8. Resolve the route with gateway-local or gateway-managed provider credentials.
9. Apply missing defaults and execute the concrete model without forwarding the consumer credential upstream.
10. Keep the original profile identity in gateway audit metadata.

Headers such as `x-gateway-image-handling`, locality hints, or namespace hints cannot relax profile policy. A profile selector received from a federation peer is rejected and is never forwarded.

## Error contract

- `401`: no valid consumer credential.
- `403`: missing permission, cross-namespace attempt, direct-model denial, or contracted-but-disabled profile execution.
- `404`: authorized namespace/profile/version does not exist.
- `409`: registered route binding no longer matches live gateway configuration.
- `412`: mutation ETag mismatch.
- `422`: invalid manifest or unsupported profile policy combination.
- `428`: mutation precondition missing.
- `503`: profile registry unavailable or an otherwise valid local route is unavailable.

Errors use the existing OpenAI or Anthropic envelope for inference endpoints and `Cache-Control: no-store` for profile API errors.

## Consumer stale-cache behavior

Consumers cache only authenticated snapshot responses, keyed by namespace and ETag. On discovery failure they may use a validated last-known-good snapshot for at most the advertised `stale-if-error` window. A stale snapshot may populate selectors/default labels, but it cannot authorize or execute a route without model-gateway. Consumers must never fall back to reading `model-info.json` or inventing concrete model policy.
