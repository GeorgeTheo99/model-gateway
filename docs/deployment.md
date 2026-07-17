# Deployment

`model-gateway` is the canonical gateway service. The old `cloud-gateway` name is retired; do not create new `cloud-gateway` checkouts, bare repos, LaunchAgents, or environment variables.

## Portable macOS install (consumer machines)

A second Mac can run the gateway directly from a clone — no `server-ci`, bare repo, or CI hook required:

```bash
git clone <repo-url> ~/local_code/model-gateway
cd ~/local_code/model-gateway
./install.sh              # uv sync + launchd plist + start + /health verify
# or: ./install.sh --no-start
```

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

Portable defaults are env-overridable:

- `MODEL_GATEWAY_CONFIG=<repo>/config/config.yaml`
- `MODEL_GATEWAY_MODEL_INFO=<repo>/model-info.json`
- `MODEL_GATEWAY_MODEL_INFO_SOURCE=<repo>/model-info.json`
- `MODEL_GATEWAY_HOST=127.0.0.1`, `MODEL_GATEWAY_PORT=9111`
- `MODEL_GATEWAY_LEDGER_PATH=~/srv/model-gateway/shared/ledger.db`
- `MODEL_GATEWAY_LOG_DIR=~/Library/Logs/model-gateway`

After install, edit `config/config.yaml` to add provider secrets/auth. A fresh generated config intentionally has `providers: {}`, so catalog entries remain unavailable until providers are configured. If you deliberately expose the gateway beyond loopback (`MODEL_GATEWAY_HOST=0.0.0.0`), configure `auth.client_keys` and firewall rules first.

The installer refuses to overwrite/stop/remove an existing `com.local.model-gateway` plist whose `WorkingDirectory` points somewhere else (for example an ls99 `server-ci` runtime checkout). Use `model-gateway install --force` or `MODEL_GATEWAY_FORCE=1` only when you intentionally want this clone to adopt that LaunchAgent label.

## ls99 dev-server deploy flow

ls99 still uses the dev/runtime split:

- Development checkout: `~/local_code/model-gateway`
- Bare repo: `~/repos/model-gateway.git`
- Runtime checkout: `~/srv/model-gateway/current`
- Shared secrets/config: `~/srv/model-gateway/shared/config/config.yaml`
- LaunchAgent: `com.local.model-gateway`
- Service port: `127.0.0.1:9111`

Pushes to `main` on `~/repos/model-gateway.git` run `hooks/post-receive`, which delegates to:

```bash
/Users/localserver99/ci/server/bin/server-ci run-model-gateway <oldrev> <newrev> refs/heads/main
```

The dev-server pipeline updates the runtime checkout, keeps `config/config.yaml` symlinked to shared config, syncs a runtime `.venv`, imports the app as a smoke check, restarts `com.local.model-gateway` for runtime-affecting changes, and verifies `/health`.

## Provider config

`model-info.json` is the committed portable catalog. Upstream providers are local runtime config: a model is exposed from `/v1/models` only when its provider is enabled and has the required local config/secrets in `config/config.yaml` or provider env vars. Missing providers do not block startup; requests for catalog models whose provider is unavailable return a clear `provider_not_configured` / `provider_disabled` error.

Databricks is optional and disabled/unconfigured on this machine. A work machine can enable Databricks model serving with gitignored config or env (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, optional `DATABRICKS_SERVING_BASE_URL`) without changing the committed catalog.

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

Manual deploy:

```bash
/Users/localserver99/ci/server/bin/server-ci deploy-model-gateway main
```

## Downstream catalog exports

`scripts/export_catalogs.py` renders the downstream alias catalog from the same
merge the router uses (`model-info.json` + the `config.yaml` `models:` overlay,
overlay wins on id clash):

- `exports.model_aliases` → `~/.claude/model-aliases.json`, the **public
  contract** consumed by Pi-side renderers (`pi-shared/bin/pi-catalog`) and any
  other tool that needs the model catalog.

Pi-specific artifacts (Pi `models.json` and `pi-launchers.zsh`) are NO LONGER
rendered by the gateway — they live in `pi-shared/bin/pi-catalog`, which reads
the alias file. The gateway stays generic (no Pi config-schema knowledge).

Exports are **opt-in** via the gitignored `config.yaml`. Machines that don't
need an alias file omit the `exports:` section and the generator is a no-op.
Generation runs on gateway start (`src/server.py` lifespan) and on
`/admin/api/reload`; drift is checked with `scripts/export_catalogs.py --check`.

`runtime/omlx-config/fan_out_settings.py` still owns the oMLX-local concerns
(`~/.omlx/model_settings.json` sync + oMLX restart) but no longer generates
aliases itself — it delegates to `export_catalogs.py --aliases-out`, so a manual
`fan_out` run stays consistent with the gateway-generated catalog.

`runtime/model-info.json` is a symlink to the repo-root `model-info.json`; the
committed root catalog is the single source of truth.

## Pi launcher integration

For Pi users, configure an alias export in `config/config.yaml` when desired:

```yaml
exports:
  model_aliases: ~/.claude/model-aliases.json
```

The gateway regenerates that generic alias file on startup. Render Pi-specific artifacts separately with `pi-shared/bin/pi-catalog`; generated launchers define `pi-restart model-gw`, which now delegates to the portable `model-gateway restart` command when available and falls back to `server-ci restart --model-gw` on ls99/dev-server installs. Pi-owned generated artifacts should live under `~/.pi/` (for example, `~/.pi/generated/pi-launchers.zsh`), never under this repository or a model-gateway runtime directory.
