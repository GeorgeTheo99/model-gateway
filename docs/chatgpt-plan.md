# ChatGPT plan provider

The `chatgpt` provider serves models on the owner's ChatGPT plan through OpenAI's
[Sign in with ChatGPT](https://developers.openai.com/siwc/token-sharing-open-source) for
open-source and locally hosted apps (checked 2026-11-12). Requests use the plan's Work and
Codex allowance, not an API key. Capability: `chatgpt_plan.v1`.

Plan usage is for the owner's own use. Who may use the provider:

- **An admin key.** It is the owner, so it is always allowed.
- **The consumer whose manager signed the plan in.** It is allowed when `consumers` is not set,
  for example Myai's `myai-manager` grants `myai`.
- **The consumers that `consumers` names.** The list replaces that grant:

  ```yaml
  providers:
    chatgpt:
      consumers: [myai, pi]
  ```

- **Nobody else.** Any other consumer, a legacy key, an anonymous caller and every federation peer
  is refused. Peers aren't offered the plan's models at all.

`consumers` works on any provider; elsewhere, leaving it unset allows everyone.

## Models

`config/onboarding/chatgpt-plan.yaml` ships `gpt-6.1-sol-chatgpt`, which runs upstream
`gpt-6.1-sol`. The upstream id is not routable (`src/catalog.py`), so `gpt-6.1-sol` stays the
API-key route and no name switches billing. The rows are priced at zero: the meter is
ChatGPT › Settings › Usage.

The rows load signed out too. They are unavailable then (`chatgpt_signed_out`), which keeps a
routing profile that names one valid when the session ends. A ChatGPT row alone doesn't make a
catalog non-empty.

## Sign-in (`src/chatgpt_auth.py`)

| Endpoint | Body | Result |
|---|---|---|
| `GET /admin/api/chatgpt` | — | Status: `signed_in`, `email`, `plan`, `pending` (no URL: it can carry the ID token), `last_error`, `reauth_reason`, `usage_url`; never a token |
| `POST /admin/api/chatgpt/sign-in` | `{agent_name?, new_account?}` | `authorize_url` to open, `redirect_uri`, `expires_at` |
| `POST /admin/api/chatgpt/callback` | `{url}` | Finishes a sign-in made on another device |
| `POST /admin/api/chatgpt/sign-out` | — | Revokes the refresh token, then clears the tokens |

These need an admin key or a manager holding `providers:manage` and `models:register` with
`chatgpt` in its providers, plus admin writes.

The sign-in works like this:

- **Registration.** The first sign-in registers a client (`client_id=dynamic_agent_client`,
  `agent_name_hint`). It sends a stable `ext_agent_host_id` (`urn:uuid:…`), which is saved before
  the sign-in starts. Later sign-ins reuse the issued `oaiapp_…` client id with `id_token_hint`.
- **PKCE and the callback.** The gateway uses authorization code with PKCE (S256), with its own
  listener on `http://127.0.0.1:1455/auth/callback` (or another free port) for 10 minutes. A
  browser on another device can't reach that listener: the owner posts the address it ended on to
  `/callback`.
- **Token checks.** The ID token comes straight from the token endpoint over TLS, so TLS
  establishes its issuer (OpenID Connect Core §3.1.3.7). The gateway still checks its issuer,
  audience, expiry and nonce. A returning sign-in must be the same account. The token response must
  grant `chatgpt.tokens.use.direct`.
- **Storage.** The record is `<secret dir>/chatgpt-oauth.json`, mode 0600, written atomically.
- **Interrupted sign-ins.** An issued client id is kept the moment the callback brings it
  (`pending_client_id`), so a sign-in whose exchange failed is retried with it. `new_account`
  registers another account, which replaces the saved one and revokes its session.
- **Refresh.** Requests refresh the access token within 5 minutes of expiry, and once again after a
  401. A 403 is an eligibility refusal and doesn't trigger a refresh. Refreshes are serialized
  because refresh tokens rotate, and a 401 for a token another request already replaced uses the
  new one.
  - An unusable refresh token (`invalid_grant`, `refresh_token_reused`, …) signs the plan out with a
    `reauth_reason`.
  - A transient failure keeps the session.

## Requests (`src/chatgpt_responses.py`)

Only `POST /v1/chat/completions` serves these models. `/v1/responses` and `/v1/messages`
return 400. Each request is translated to the plan's Responses API:

- **Request shape.** Every request is sent with `store: false` and `stream: true`, with the full
  history in `input`.
- **System prompts.** Leading system messages become `instructions`; later ones become developer
  messages.
- **Tools.** Function tools are grouped in one namespace (`app`), and calls carry it. Hosted tools
  are refused.
- **Tool call arguments.** A single call's arguments stream as deltas. Parallel calls send
  theirs only whole, in `response.function_call_arguments.done` and `response.output_item.done`,
  so the translation sends whatever part of the whole arguments the deltas didn't.
- **Dropped fields.** `temperature`, `top_p`, `max_tokens`, `metadata`, `user` and
  `prompt_cache_retention` are not sent.
- **Kept fields.** `prompt_cache_key`, `parallel_tool_calls`, `tool_choice`, `response_format`
  (as `text.format`) and `verbosity` are sent.
- **Reasoning.** The reasoning summary streams as `reasoning_content`. Encrypted reasoning items
  are kept by tool call id and sent again before that call, so reasoning carries across a tool loop.
- **Success and failure.** Success is `response.completed`. Nothing falls back to another route.
- **Usage limit.** A usage limit (`subscription_sharing_usage_limit_exceeded`) is never retried.
  - Before streaming, it is a 429.
  - Mid-stream, it ends the stream with an error event, or becomes a 429 when the client asked for
    no stream.
  - Error responses name `subscription_sharing_*` codes in an `X-Model-Gateway-Error-Code` header.
- **Signed out.** A profile that names the plan registers and stays registered while signed out
  (its endpoint is fixed). Running it then returns 503 `chatgpt_signed_out`.
