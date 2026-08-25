# Author dev-server deploy flow (ls99)

This documents the maintainer's own CI-driven deployment on the ls99 dev
server. It is **not** required to run model-gateway — consumer machines should
use the portable install in [deployment.md](deployment.md).

## Layout

The dev server uses a dev/runtime split:

- Development checkout: `~/local_code/model-gateway`
- Bare repo: `~/repos/model-gateway.git`
- Runtime checkout: `~/srv/model-gateway/current`
- Shared secrets/config: `~/srv/model-gateway/shared/config/config.yaml`
- oMLX runtime config: `~/srv/model-gateway/shared/omlx-config/` (machine-local
  shared state; not part of this repository)
- LaunchAgents: `com.local.model-gateway` (gateway, `127.0.0.1:9111`) and
  `com.local.omlx` (local oMLX model server, `:9110`)

## Repository authority and publishing

The development checkout keeps two distinct remotes:

- `origin` — `~/repos/model-gateway.git`, the authoritative local bare
  repository and deployment source
- `github` — the public GitHub mirror used for distribution

GitHub synchronization is intentionally **manual**. A push to the local bare
repository deploys this server, but it does not push to GitHub. Publish the same
commit separately when it is ready to be public:

```bash
git push origin main    # update authoritative bare repo and deploy locally
git push github main    # update public mirror
```

Verify alignment with `git rev-parse HEAD origin/main github/main`. Consumer
machines clone or update from GitHub using the portable flow in
[deployment.md](deployment.md); they do not use this maintainer topology.

Secret scanning is enforced in three layers: tracked pre-commit/pre-push hooks,
the authoritative bare repository's pre-receive hook, and the pinned GitHub
Gitleaks workflow. Install Gitleaks and activate the worktree hooks once per
clone:

```bash
brew install gitleaks
git config core.hooksPath .githooks
```

The custom configuration detects Zhipu/Z.ai's provider-specific key shape that
GitHub's provider scanner may not recognize. Never bypass a failed scan or
allowlist a whole credential-bearing file.

## Push-to-deploy

Pushes to `main` on `~/repos/model-gateway.git` run `hooks/post-receive`, which
delegates to a machine-local CI runner (untracked, lives outside this repo):

```bash
~/ci/server/bin/server-ci run-model-gateway <oldrev> <newrev> refs/heads/main
```

The pipeline updates the runtime checkout, keeps `config/config.yaml` symlinked
to shared config, syncs a runtime `.venv`, imports the app as a smoke check,
restarts `com.local.model-gateway` for runtime-affecting changes, and verifies
`/health`.

Manual deploy:

```bash
~/ci/server/bin/server-ci deploy-model-gateway main
```

## oMLX companion service

`com.local.omlx` runs `omlx serve` for local MLX models and is a routing
backend of the gateway (the built-in `omlx` provider defaults to
`http://localhost:9110/v1`; override with `MODEL_GATEWAY_OMLX_BASE_URL`).

Its runtime configuration (`fan_out_settings.py`, patches, converters) lives in
machine-local shared state at `~/srv/model-gateway/shared/omlx-config/` — set
via `SERVER_MODEL_GATEWAY_OMLX_CONFIG` in the CI runner's untracked
`config/install.env`. `fan_out_settings.py` locates the deployed checkout via
`MODEL_GATEWAY_DEPLOY_TREE` (default `~/srv/model-gateway/current`) and
delegates alias generation to `scripts/export_catalogs.py`.
