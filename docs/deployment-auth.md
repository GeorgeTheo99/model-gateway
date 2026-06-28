# Cloud Gateway Deployment and Auth Rollout

## Current deployed state

The production launchd service is `com.local.cloud-gateway`.

- Working directory: `~/srv/cloud-gateway/current`
- Source repo: `~/local_code/cloud-gateway`
- Deployed via push to local bare repo: `~/repos/cloud-gateway.git`
- Runtime port: `9111`
- Health check: `GET http://127.0.0.1:9111/health`
- Admin UI: `GET http://127.0.0.1:9111/admin`

As of commit `e5ba25a`, the live service has:

- `/admin` available.
- `/admin/api/*` fail-closed with `401` until an admin key is configured.
- `/v1/*` protected with `401` unless a valid client (or admin) key is sent.

Both admin and client auth are **currently enabled** on the deployed service.
Keys live in the gitignored runtime config `~/srv/cloud-gateway/shared/config/config.yaml`
(symlinked into the deploy tree as `config/config.yaml`) under the `auth:` section.
The launchd plist does **not** carry auth env vars, keeping secrets out of the
repo and the plist.

## Where keys are configured

Admin/client keys are read from two sources, merged at runtime (env takes
precedence over config):

1. `auth:` section in `config/config.yaml` (canonical; gitignored, holds provider keys too):

   ```yaml
   auth:
     admin_keys:
       - "<admin-key>"
     client_keys:
       - "cloud"
   ```

   Each field accepts a list or a single comma-separated string. Reloaded on
   `POST /admin/api/reload` (and on service restart).

2. Env vars `CLOUD_GATEWAY_ADMIN_KEY` / `CLOUD_GATEWAY_CLIENT_KEYS`
   (comma-separated), which override/extend config. Useful for ad-hoc overrides
   without editing the file.

The launchd plist (`com.local.cloud-gateway`) only sets `CLOUD_GATEWAY_MODEL_INFO`;
auth env vars are intentionally not wired into the plist template in `server-ci`
`install-launchagents` to avoid committing secrets.

## Auth environment variables

These are optional overrides for the config-file keys above.

### `CLOUD_GATEWAY_ADMIN_KEY`

Protects the admin API only.

- Affects: `/admin/api/*`
- Does not affect: `/v1/models`, `/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/health`
- Safe first rollout step.

Programmatic admin calls must include one of:

```bash
Authorization: Bearer $CLOUD_GATEWAY_ADMIN_KEY
x-api-key: $CLOUD_GATEWAY_ADMIN_KEY
```

The browser UI at `/admin` still loads without a key, but its API calls return `401` until the key is entered in the UI.

For temporary local development only, admin auth can be bypassed with:

```bash
CLOUD_GATEWAY_ALLOW_UNAUTHENTICATED_ADMIN=true
```

Do not use that when the gateway is reachable beyond trusted local development.

### `CLOUD_GATEWAY_CLIENT_KEYS`

Protects client/model APIs.

- Affects: `/v1/models`, `/v1/debug/thinking`, `/v1/chat/completions`, `/v1/messages`, `/v1/responses`
- Does not affect: `/health`
- Format: comma-separated accepted tokens.

Recommended least-breakage value when enabling client auth:

```bash
CLOUD_GATEWAY_CLIENT_KEYS=cloud
```

Clients must then send one of:

```bash
Authorization: Bearer cloud
x-api-key: cloud
```

Admin keys are also accepted on `/v1/*`, so an admin token can be used for debugging.

## Known consumer compatibility

Known clients that already send the default `cloud` token and should continue working if `CLOUD_GATEWAY_CLIENT_KEYS=cloud` is enabled:

- `server/local_claude/zshrc-launcher.zsh`
  - Claude path sends `ANTHROPIC_AUTH_TOKEN=cloud`.
  - Codex path sends `OPENAI_API_KEY=cloud`.
  - Pi generated `models.json` uses `apiKey: cloud` for the cloud-gateway provider.
- `server/directory/install.sh`
  - Remote Claude/Codex launcher functions use `LS99_CLOUD_KEY`.
- `server/voice-gateway/nlu/chat_client.py`
  - Defaults `MODEL_GATEWAY_API_KEY` to `cloud`.
- `server/scripts/probe-thinking-shapes.py`
  - Cloud endpoint default token is `cloud`.

Known no-header probes/checks (updated to use `/health`, which stays unauthenticated):

- `server/scripts/install-home-server.sh`
  - Now probes `http://127.0.0.1:9111/health` (liveness) instead of `/v1/models`.
- `server/docs/operations/HOME_SERVER_BACKEND_DISTRIBUTION.md`
  - Documents `curl -fsS http://127.0.0.1:9111/health` for unauthenticated liveness,
    plus an authenticated `/v1/models` example.
- Any ad-hoc `curl http://127.0.0.1:9111/v1/*` without `Authorization` or `x-api-key`
  will now return `401`; use `/health` for liveness or send `Authorization: Bearer cloud`.

## Recommended rollout order

This rollout is now complete. The steps are retained as a reference for
re-running on a fresh install or another machine.

1. Deploy the code first while leaving `/v1/*` open (no `auth:` section in config).
2. Configure `auth.admin_keys` in `config/config.yaml` and restart the service.
3. Verify:

   ```bash
   curl -fsS http://127.0.0.1:9111/health
   curl -fsS http://127.0.0.1:9111/admin
   curl -i http://127.0.0.1:9111/admin/api/status   # 401
   curl -fsS -H "Authorization: Bearer $ADMIN_KEY" \
     http://127.0.0.1:9111/admin/api/status
   ```

4. Update no-header probes to use `/health` (unauthenticated liveness).
5. Add `auth.client_keys: ["cloud"]` to config and restart.
6. Verify:

   ```bash
   curl -i http://127.0.0.1:9111/v1/models                       # 401
   curl -fsS -H "Authorization: Bearer cloud" http://127.0.0.1:9111/v1/models
   ```

## Service commands

Restart/status are managed by `server-ci`:

```bash
server-ci restart --cloud-gw
server-ci restart --status
```

Launchd inspection:

```bash
launchctl print gui/$(id -u)/com.local.cloud-gateway
```

Live service smoke check:

```bash
curl -fsS http://127.0.0.1:9111/health
curl -i http://127.0.0.1:9111/admin/api/status                       # 401 without key
curl -fsS -H "Authorization: Bearer cloud" http://127.0.0.1:9111/v1/models | jq '.data | length'
```
