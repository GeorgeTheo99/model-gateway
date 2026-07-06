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

Manual deploy:

```bash
/Users/localserver99/ci/server/bin/server-ci deploy-model-gateway main
```
