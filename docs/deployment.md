# Deployment

`model-gateway` is the canonical gateway service. `cloud-gateway` is retired; do not recreate legacy checkouts, bare repos, LaunchAgents, or environment variables.

## Portable macOS install (consumer machines)

A second Mac can run the gateway directly from a clone — no `server-ci`, bare repo, or CI hook required:

```bash
git clone <repo-url> ~/local_code/model-gateway
cd ~/local_code/model-gateway
cp <reviewed-private-catalog> model-info.json
./install.sh              # uv sync + launchd plist + start + /health verify
# or: ./install.sh --no-start
```

Before installing, provision that machine's Git-ignored `model-info.json` catalog (for example, copy a reviewed catalog from the machine's private configuration backup). The repository intentionally does not ship model routes, local model paths, or per-machine pricing metadata. The installer does not create this file.

The installer creates a repo-local `config/config.yaml` if missing, installs the `com.local.model-gateway` LaunchAgent, symlinks `model-gateway` into `~/.local/bin`, and runs from the clone with:

```bash
uv run python -m src.main
```

Operator commands:

```bash
model-gateway status
model-gateway logs -f
model-gateway restart
model-gateway update      # git pull --ff-only + uv sync + restart + verify
model-gateway env
```

Portable defaults are env-overridable. During `install`, the resolved bind host and port are persisted in the owner-only `~/Library/Application Support/model-gateway/install.env`; later `update`, `restart`, `status`, and fresh shell sessions recover that assignment before falling back to the legacy defaults. Updates re-exec the newly pulled operator script before rewriting the LaunchAgent, so future installer changes use the current resolution logic. An explicit environment value still takes precedence when intentionally re-running `install`.

When upgrading from a version that predates persisted bind configuration, first pull the checkout directly with `git -C <model-gateway-checkout> pull --ff-only`, then run `MODEL_GATEWAY_PORT=<currently-installed-port> model-gateway install --no-start` (and include `MODEL_GATEWAY_HOST` if customized). Normal `model-gateway update` commands are safe afterward. This one-time step is necessary because an already-running older Bash script cannot adopt update logic that has not yet been pulled.

- `MODEL_GATEWAY_CONFIG=<repo>/config/config.yaml`
- `MODEL_GATEWAY_MODEL_INFO=<repo>/model-info.json`
- `MODEL_GATEWAY_MODEL_INFO_SOURCE=<repo>/model-info.json`
- `MODEL_GATEWAY_HOST=127.0.0.1`, `MODEL_GATEWAY_PORT=9111`
- `MODEL_GATEWAY_LEDGER_PATH=~/srv/model-gateway/shared/ledger.db`
- `MODEL_GATEWAY_LOG_DIR=~/Library/Logs/model-gateway`
- `MODEL_GATEWAY_BACKUP_DIR=~/Library/Application Support/model-gateway/backups/config`
  (always keep this private state outside diagnostic log trees)

The ledger database and any SQLite WAL/SHM sidecars are restricted to mode
`0600`. Configuration/catalog backups are stored in a mode-`0700` directory as
mode-`0600` files and retain 20 generations per managed file by default
(`MODEL_GATEWAY_BACKUP_RETENTION`, bounded to 1–1000). On startup, legacy
`<log-dir>/config-backups` and the former Home Server package path
`~/Library/Application Support/HomeServer/ci/logs/config-backups` are validated,
retention-pruned, clamped private, and atomically moved
to the configured backup directory before serving requests. Log and backup
roots may not overlap. Uvicorn request access logging is disabled because request
URLs can contain temporary capabilities; structured usage remains available in
the ledger.

After install, edit `config/config.yaml` to add provider secrets/auth. A fresh generated config intentionally has `providers: {}`, so catalog entries remain unavailable until providers are configured. If you deliberately expose the gateway beyond loopback (`MODEL_GATEWAY_HOST=0.0.0.0`), configure `auth.client_keys` and firewall rules first.

The installer refuses to overwrite/stop/remove an existing `com.local.model-gateway` plist whose `WorkingDirectory` points somewhere else (for example a CI-managed runtime checkout). Use `model-gateway install --force` or `MODEL_GATEWAY_FORCE=1` only when you intentionally want this clone to adopt that LaunchAgent label.

## Dev-server deploy flow (maintainer)

The maintainer's own CI-driven dev-server deployment is documented separately
in [deployment-dev-server.md](deployment-dev-server.md). Consumer machines do
not need it — use the portable install above.

## Provider config

`model-info.json` is a machine-local, Git-ignored catalog. `MODEL_GATEWAY_MODEL_INFO` selects the live copy. `MODEL_GATEWAY_MODEL_INFO_SOURCE` may select a second machine-local mirror used by admin/onboarding writes; it is not a Git ownership boundary and may point to the same file on portable installs. Operators are responsible for backing up and propagating reviewed catalog changes between machines.

A model is exposed from `/v1/models` only when its provider is enabled and has the required local config/secrets in `config/config.yaml` or provider environment variables. Missing providers do not block startup; requests for catalog models whose provider is unavailable return a clear `provider_not_configured` / `provider_disabled` error.

Databricks is optional and disabled/unconfigured on this machine. A work machine can enable Databricks model serving with Git-ignored catalog/config or environment variables (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, optional `DATABRICKS_SERVING_BASE_URL`) without changing repository-owned files.

## Vision routing policy

Image input to a raw text-only model fails closed by default. Use a native vision
model or an explicit gateway composite when images are expected. On machines
that install the standard local preset contract, callers send `auto-local` for
the canonical Local Best route; the gateway expands it to GLM-5.2 text plus
Gemma 4 26B vision using `extract_then_answer`. The legacy `best-local` ID stays
distinct during migration and shares the explicit `detail-local` route backed by
Gemma 4 31B vision. Composite models always use their declared image-handling
mode and cannot be redirected by client headers or request fields.

For compatibility clients, an operator may configure locality-scoped helpers:

```text
GATEWAY_VISION_FALLBACK_LOCAL=<native-local-vision-model>
GATEWAY_VISION_FALLBACK_CLOUD=<native-cloud-vision-model>
```

A text-only local route can use only the local helper, and a text-only cloud
route can use only the cloud helper. If its matching variable is empty, that
route fails closed. Source or fallback pools that mix local oMLX and cloud
providers are rejected. Each fallback must be a native vision model using an
OpenAI-compatible protocol; the gateway validates every configured pool
candidate at startup, on admin reload, and again before each fallback request.
Cloud helper use is explicit cloud egress and is logged as such.

Scoped fallbacks default to `extract_then_answer`: the helper receives the image
and returns bounded observations, then the originally requested text model
answers. A caller can explicitly request `reroute`, or an operator can set
`GATEWAY_VISION_FALLBACK_MODE=reroute`, to send the complete translated request
to the fallback instead. The mode must be `reroute` or `extract_then_answer`.
Only `extract_then_answer` keeps the originally requested text model as the
answering model; `reroute` lets the fallback model answer the complete request.
Extraction accepts only inline `data:image/...;base64` payloads so the gateway
can enforce byte bounds without performing server-side URL fetches. It accepts
up to 4 images per request by default; operators can adjust the limit with
`GATEWAY_VISION_FALLBACK_MAX_IMAGES=<1-32>` (validated at startup with the other
fallback policy checks). Inline images retain the existing 20 MB per-image and
32 MB aggregate decoded-byte bounds in both composite and process-wide fallback
modes. Gateway API request bodies are streamed into a bounded 64 MB buffer;
vision-helper responses are streamed with a 1 MB cap before JSON parsing.
Observation text is capped per image and per request, and the complete
multi-image extraction is bounded by `GATEWAY_VISION_EXTRACTION_TOTAL_TIMEOUT_SECONDS`
(default 900 seconds, range 1-3600).

Successful inline-image observations are cached per process in a 256-entry LRU,
keyed by decoded image bytes, media/detail options, and the complete
extractor/provider/prompt identity. Cache entries expire after
`GATEWAY_VISION_OBSERVATION_CACHE_TTL_SECONDS` (default 3600 seconds, range
0-86400; 0 disables caching), and a successful admin registry reload clears the
cache. Pi/session history still owns the original image payloads; this cache
holds only bounded observation text and does not provide durable image memory.

The legacy `GATEWAY_VISION_FALLBACK=<native-vision-model>` remains supported for
existing deployments and retains its historical default mode of `reroute`. It
cannot be combined with either scoped variable. New deployments should use the
scoped variables so local images cannot cross into cloud providers implicitly.
Consumer profile requests remain governed by their explicit profile vision
route and never use these process-wide fallbacks.

## Optional federation

Nodes may import direct routes from explicitly configured peer gateways. Add a
`federation:` block to the gitignored `config/config.yaml`; no deploy-time
secrets or node-specific routes belong in the repository. Imported IDs are
always namespaced as `<owner_node>/<direct_model_id>`, local routing wins, and
imports are never included in downstream catalog exports.

Startup loads the atomic last-known-good cache, performs an initial peer catalog
refresh, and starts periodic refreshes. The default cache is
`config/federation-cache.json` (Git-ignored). Service shutdown cleans up the
refresh task. An authenticated, write-enabled `POST /admin/api/reload`
reconfigures federation after manual config edits; there are no federation
admin write APIs. See [federation.md](federation.md) for the full config,
security, discovery, and forwarding contract.

## Downstream catalog exports

`scripts/export_catalogs.py` renders the downstream alias catalog from the same
merge the router uses (`model-info.json` + the `config.yaml` `models:` overlay,
overlay wins on id clash):

- `exports.model_aliases` → `~/srv/model-gateway/shared/model-aliases.json`, the **public
  contract** consumed by Pi-side renderers (`pi-shared/bin/pi-catalog`) and any
  other tool that needs the model catalog.

Pi-specific artifacts (Pi `models.json` and `pi-launchers.zsh`) are NO LONGER
rendered by the gateway — they live in `pi-shared/bin/pi-catalog`, which reads
the alias file. The gateway stays generic (no Pi config-schema knowledge).

Exports are **opt-in** via the gitignored `config.yaml`. Machines that don't
need an alias file omit the `exports:` section and the generator is a no-op.
Generation runs on gateway start (`src/server.py` lifespan) and on
`/admin/api/reload`; drift is checked with `scripts/export_catalogs.py --check`.

On machines that run a local oMLX service, a machine-local `fan_out_settings.py`
(kept in deployment shared state, outside this repository — see
[deployment-dev-server.md](deployment-dev-server.md)) owns the oMLX-local
concerns (`~/.omlx/model_settings.json` sync + oMLX restart) but no longer
generates aliases itself — it delegates to `export_catalogs.py --aliases-out`,
so a manual `fan_out` run stays consistent with the gateway-generated catalog.

The catalog path is resolved directly from `MODEL_GATEWAY_MODEL_INFO` (or the
checkout-local `model-info.json` default); there is no tracked runtime catalog
or compatibility symlink.

## Pi launcher integration

For Pi users, configure an alias export in `config/config.yaml` when desired:

```yaml
exports:
  model_aliases: ~/srv/model-gateway/shared/model-aliases.json
```

The gateway regenerates that generic alias file on startup. Render Pi-specific artifacts separately with `pi-shared/bin/pi-catalog`; generated launchers define `pi-restart model-gw`, which now delegates to the portable `model-gateway restart` command when available and falls back to `server-ci restart --model-gw` on the maintainer's dev-server install. Pi-owned generated artifacts should live under `~/.pi/` (for example, `~/.pi/generated/pi-launchers.zsh`), never under this repository or a model-gateway runtime directory.

Pi removes image blocks before transport for models declared text-only. A route that deliberately relies on the validated process-wide fallback must therefore opt in through catalog metadata:

```yaml
pi:
  image_input: gateway-assisted
```

The alias exporter carries this field through unchanged; `pi-shared/bin/pi-catalog` renders the route with image input enabled and labels it `assisted vision`. Use it only for direct routes whose locality-scoped fallback is guaranteed on that deployment. Native models continue to use `vision: true`, and explicit composites continue to advertise their own public vision capability.
