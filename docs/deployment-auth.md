# Cloud Gateway Deployment and Auth Rollout

## Current deployed state

The production launchd service is `com.local.cloud-gateway`.

- Working directory: `~/srv/cloud-gateway/current`
- Source repo: `~/local_code/cloud-gateway`
- Deployed via push to local bare repo: `~/repos/cloud-gateway.git`
- Runtime port: `9111`
- Health check: `GET http://127.0.0.1:9111/health`
- Admin UI: `GET http://127.0.0.1:9111/admin`

As of commit `b8880cf`, the live service has:

- `/admin` available.
- `/admin/api/*` fail-closed with `401` until `CLOUD_GATEWAY_ADMIN_KEY` is configured.
- `/v1/*` unchanged and open unless `CLOUD_GATEWAY_CLIENT_KEYS` is configured.

## Auth environment variables

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

Known no-header probes/checks that will break if client auth is enabled before updating them:

- `server/scripts/install-home-server.sh`
  - Probes `http://127.0.0.1:9111/v1/models` without auth.
- `server/docs/operations/HOME_SERVER_BACKEND_DISTRIBUTION.md`
  - Documents `curl -fsS http://127.0.0.1:9111/v1/models` without auth.
- Any ad-hoc `curl http://127.0.0.1:9111/v1/*` without `Authorization` or `x-api-key`.

## Recommended rollout order

1. Deploy the code first while leaving `/v1/*` open.
2. Configure `CLOUD_GATEWAY_ADMIN_KEY` and restart the service.
3. Verify:

   ```bash
   curl -fsS http://127.0.0.1:9111/health
   curl -fsS http://127.0.0.1:9111/admin
   curl -i http://127.0.0.1:9111/admin/api/status
   curl -fsS -H "Authorization: Bearer $CLOUD_GATEWAY_ADMIN_KEY" \
     http://127.0.0.1:9111/admin/api/status
   curl -fsS http://127.0.0.1:9111/v1/models
   ```

4. Update no-header probes to send the future client token.
5. Set `CLOUD_GATEWAY_CLIENT_KEYS=cloud` and restart.
6. Verify:

   ```bash
   curl -i http://127.0.0.1:9111/v1/models
   curl -fsS -H "Authorization: Bearer cloud" http://127.0.0.1:9111/v1/models
   ```

Expected after client auth is enabled:

- First command returns `401`.
- Second command returns model data.

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
curl -i http://127.0.0.1:9111/admin/api/status
curl -fsS http://127.0.0.1:9111/v1/models | jq '.data | length'
```
