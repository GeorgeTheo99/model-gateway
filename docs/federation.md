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
`/v1` client auth is open. It returns schema version 1, the source node ID, a
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
