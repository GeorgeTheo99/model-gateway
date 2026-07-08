# Deployment

`model-gateway` is the canonical gateway service. The old `cloud-gateway` name is retired; do not create new `cloud-gateway` checkouts, bare repos, LaunchAgents, or environment variables.

## Local layout

- Development checkout: `~/local_code/model-gateway`
- Bare repo: `~/repos/model-gateway.git`
- Runtime checkout: `~/srv/model-gateway/current`
- Shared secrets/config: `~/srv/model-gateway/shared/config/config.yaml`
- LaunchAgent: `com.local.model-gateway`
- Service port: `127.0.0.1:9111`

## Deploy flow

Pushes to `main` on `~/repos/model-gateway.git` run `hooks/post-receive`, which delegates to:

```bash
/Users/localserver99/ci/server/bin/server-ci run-model-gateway <oldrev> <newrev> refs/heads/main
```

The deploy pipeline updates the runtime checkout, keeps `config/config.yaml` symlinked to shared config, syncs a runtime `.venv`, imports the app as a smoke check, restarts `com.local.model-gateway` for runtime-affecting changes, and verifies `/health`.

## Provider config

`model-info.json` is the committed portable catalog. Upstream providers are local runtime config: a model is exposed from `/v1/models` only when its provider is enabled and has the required local config/secrets in `config/config.yaml` or provider env vars. Missing providers do not block startup; requests for catalog models whose provider is unavailable return a clear `provider_not_configured` / `provider_disabled` error.

Databricks is optional and disabled/unconfigured on this machine. A work machine can enable Databricks model serving with gitignored config or env (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, optional `DATABRICKS_SERVING_BASE_URL`) without changing the committed catalog.

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

## First machine setup (summary)

There is no one-shot installer; a new machine is wired by hand plus `server-ci`:
1. Create the bare repo (`git init --bare ~/repos/model-gateway.git`) and dev
   checkout (`~/local_code/model-gateway`, remote → the bare repo).
2. `server-ci deploy-model-gateway main` clones the runtime checkout at
   `~/srv/model-gateway/current`, seeds an empty `config/config.yaml` (symlinked
   to `~/srv/model-gateway/shared/config/config.yaml`), syncs the venv.
3. Hand-add provider secrets/auth to `config.yaml` (OpenRouter, Fireworks,
   Databricks, oMLX base URL, etc.). Every model is gated by its provider being
   configured, so a fresh config exposes 0 models until secrets are added.
4. `server-ci install-launchagents --reload` installs `com.local.model-gateway`
   (port 9111).
5. (Optional) add an `exports:` section to `config.yaml` to turn on the alias
   catalog; it regenerates on gateway start.
6. (Optional) source a Pi launcher from `~/.zshrc` — generate one with
   `pi-shared/bin/pi-catalog --aliases ~/.claude/model-aliases.json ...`
   (the legacy `runtime/pi-launcher.zsh` also still works).

Onboarding gaps (no templated first-run config, no automated `~/.zshrc`
sourcing, alias symlink owned by the `local-directory` service) are tracked as
follow-ups and are orthogonal to the catalog-export system.
