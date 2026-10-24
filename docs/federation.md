# Gateway federation

Federation connects a small, explicitly configured set of symmetric
model-gateway nodes. It is intentionally one hop: each node exports only models
it can route directly, and imported models are never re-exported.

## Route contract

- Local model IDs do not change and always win resolution conflicts.
- An imported model ID is `<owner_node>/<direct_model_id>`.
- The delimiter is split once, so a direct ID may itself contain any number of
  `/` characters. A direct ID may also begin with this node's or a peer's node
  ID; prefixes inside a direct ID are opaque model syntax, not another hop.
- There are no unqualified aliases for imported models in v1.
- Only `/v1/chat/completions`, `/v1/responses`, and `/v1/messages` forward.
- The importer changes only `model` to the owner's direct ID. The owner then
  performs its normal provider routing, translation, policy, and ledger work.
  Importers do not apply provider quirks, vision/reasoning transforms,
  fallbacks, or pools.

For example, `edge/org/models/large` routes directly to node `edge` with
`model: org/models/large`.

## Configuration

Federation is optional. Configure reciprocal peers in each node's gitignored
`config/config.yaml`:

```yaml
federation:
  node_id: main
  refresh_interval_seconds: 30
  stale_after_seconds: 300
  request_timeout_seconds: 10
  stream_idle_timeout_seconds: 900
  cache_path: federation-cache.json
  max_catalog_bytes: 1048576
  max_models_per_peer: 1000
  peers:
    edge:
      base_url: https://edge-gateway.example.com
      api_key_file: secrets/main-edge.api-key
```

On `edge`, use `node_id: edge` and configure a `main` peer with the reciprocal
gateway URL and the same shared credential. Use a unique secret for every node
pair; do not reuse a client, admin, provider, or another pair's secret.
Possession of a pair secret is a capability to read the two nodes' exported
catalogs and invoke direct model routes as that configured peer. It grants no
admin access and no transitive peer identity. `api_key` may be used instead of
`api_key_file`, but exactly one is required. Secret files use the provider
secret-file rules: relative paths resolve beside `config.yaml`, the target must
be a regular file, and permissions must be `0600`.

Node and peer IDs must be lowercase DNS labels (letters, digits, and interior
hyphens, at most 63 characters). A node cannot list itself as a peer. Peer base
URLs must be HTTP(S), without embedded credentials, query strings, or
fragments. Use HTTPS outside a trusted loopback/private transport.

`request_timeout_seconds` is a hard wall-clock deadline for a complete catalog
refresh. For forwarding, it covers the request and response headers for a
streaming response, or the request plus complete response body for a
non-streaming response. After streaming headers arrive,
`stream_idle_timeout_seconds` bounds the wait between response bytes; it
defaults to 900 seconds, aligned with the gateway's existing provider stream
timeout. Both settings must be positive finite numbers.

`cache_path` is relative to the real `config.yaml` directory unless absolute.
It must not resolve (including through symlinks) to `config.yaml` or any peer,
provider, or workspace `api_key_file`. Provider/workspace key paths need not
exist for this collision check. The default is `config/federation-cache.json`,
which is ignored by Git. Cache writes are atomic and mode `0600`; the document contains only
catalog metadata and timestamps—never peer URLs or credentials.

Provider ownership remains ordinary local config. If cloud routes should exist
only on `main`, configure those providers/models only on `main`; the software is
otherwise identical on every node.

## Discovery and health

An authenticated peer reads:

```text
GET /v1/federation/catalog
X-Model-Gateway-Source: edge
Authorization: Bearer <configured edge credential>
```

The endpoint always requires a configured peer credential, even when normal
`/v1` client auth is open. It returns catalog schema 1 (or 2, below), the source node ID, a
monotonically comparable revision, a SHA-256 digest, generation time, and only
direct local routes. The response repeats `X-Model-Gateway-Source` for node
identity validation.

Importers validate the schema, expected node/owner, revision/digest, payload and
model-count limits, unique direct IDs, finite numeric metadata, and direct-ID
syntax before replacing the last-known-good catalog. Slash-containing IDs are
valid and are not interpreted as nested namespaces. A bad or unreachable
refresh never erases the last-known-good catalog.

`GET /v1/models` preserves every local row as-is and appends imported rows with:

- `federated`, `owner_node`, and `direct_model_id`
- `available` for current peer catalog reachability
- `stale` based on `stale_after_seconds`
- sanitized `status` timestamps and catalog revision/digest

Catalog presence is separate from route health. Cached rows remain visible when
a refresh fails or grows stale, and their exact namespaced routes remain
eligible for a direct forwarding attempt.

A lower peer revision is rejected as a possible rollback while the importer has
a higher last-known-good revision. If an owner intentionally loses/reset its
revision state, stop the importer, remove its `federation-cache.json`, and
restart it to establish the owner's new revision baseline. This also discards
all cached peer routes until they refresh; do not edit revision values by hand.

## Forwarding security

The importer constructs outbound headers from scratch. Client authorization
and arbitrary client headers are never forwarded. Direct peer requests contain:

```text
Authorization: Bearer <configured peer credential>
X-Model-Gateway-Source: main
X-Model-Gateway-Owner: edge
Via: 1.1 main-model-gateway
```

The owner requires the configured credential for the named source, requires
itself as owner, and accepts exactly one matching `Via` hop. It rejects unknown
models, repeated/multiple hops, self loops, and wrong owners. It does not reject
a direct ID because an internal slash segment resembles a node prefix.
One-hop safety is structural: authenticated inbound forwards resolve only
against direct local routes and never enter imported forwarding, while exports
never contain imported routes. There is no transitive federation.

Sync response bytes/status and streaming SSE bytes/status are relayed without
translation. Streaming uses an async iterator for backpressure and closes the
owner response/client on completion, disconnect, cancellation, or timeout. A
timeout before response headers is returned as `504`. Once streaming headers
have been sent, HTTP status can no longer change; an idle timeout therefore
closes/truncates the downstream stream rather than returning a `504` envelope.
Clients must treat a stream without its protocol completion marker as failed.

## Trusted-local peers (catalog schema 2)

A peer the owner trusts with local work is marked `trusted_local`; its
`base_url` must be `https`:

```yaml
federation:
  peers:
    studio:
      base_url: https://studio-gateway.example.com
      api_key_file: secrets/main-studio.api-key
      trusted_local: true
```

Importers ask every peer for `GET /v1/federation/catalog?schema_version=2`.
Schema 2 adds each model's `locality` (`local` when its whole route closure is
loopback oMLX and it has no `model_fallbacks` target, else `cloud`) and
`base_model`. A node that predates schema 2 ignores the query and answers
schema 1, which is treated as all `cloud`. Each schema keeps its own revision
and digest, and an importer starts a new revision baseline when a peer's
schema changes (an upgrade or a rollback), so neither looks like it moved
backwards.

A trusted peer serves local work this node cannot:

- A request for a base-model name, or for a curated local AI model this Mac
  has not installed, with no route here goes to the first fresh `local` row
  serving that name (its `base_model`, else its direct ID) on a
  `trusted_local` peer, in configuration order. A curated local AI model name
  never falls back to a cloud route that shares its base model.
- A profile with locality `trusted_local` (credential policy `gateway_local`)
  binds this Mac's local route first, then a trusted peer's `local` row as
  `<peer>/<direct_id>`. The binding records the exported row: a stale or
  unreachable peer is `503 profile_route_unavailable`, a changed row or a peer
  no longer trusted is `409 profile_binding_changed`. `local_only` profiles
  keep meaning this Mac only.

These requests go to `/v1/federation/local/{chat/completions,messages,responses}`,
which only schema 2 nodes serve, so a peer downgraded since the importer's last
refresh refuses them instead of running its ordinary pipeline. On that path the
peer serves exactly the named model, available and local there, with no
retirement redirect, base-model reroute, `model_fallbacks` or vision fallback;
otherwise it answers `409`. Only authenticated peer forwards may use it, and
it is recorded in the serving node's ledger under its own path.
Explicit `<peer>/<id>` requests keep the ordinary forwarding path above, and
`/v1/models` shows `locality` and `base_model` only for trusted peers, where
they govern routing. A pack model's `base_model` is never exported, as it never
answers a base-model name on its own node.

Rolling an importer back to a release without schema 2 drops the cached
schema 2 peer catalogs (their digests no longer validate) until the next
refresh; its own schema 1 catalog is kept.

## Lifecycle and operations

Startup loads the last-known-good cache, performs one concurrent peer refresh,
and starts periodic refreshes. Shutdown cancels the refresh task cleanly.
`POST /admin/api/reload` revalidates/reconfigures federation after an operator
edits `config.yaml`; there are no federation admin write endpoints.

Useful checks:

```bash
# Local client view (normal client auth still applies)
curl -fsS -H 'Authorization: Bearer cloud' \
  http://127.0.0.1:9111/v1/models | jq '.data[] | select(.federated == true)'

# Peer-only direct catalog check
curl -fsS \
  -H 'X-Model-Gateway-Source: edge' \
  -H 'Authorization: Bearer <peer-key>' \
  http://127.0.0.1:9111/v1/federation/catalog | jq
```
