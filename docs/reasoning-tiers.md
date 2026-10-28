# Reasoning tiers

A client that classifies its own requests (a greeting, an ordinary question, a hard
problem) can send a **tier** instead of a concrete thinking level. The gateway places
the tier on the served model's own levels, so clients never copy each model's ladder.

```json
{"model": "kimi-k3.1-pro", "messages": [...], "model_gateway": {"reasoning_tier": "hard"}}
```

## Tiers (policy `tier-v1`)

| Tier | Level |
|---|---|
| `none` | `off` when the model can turn thinking off, else its cheapest level |
| `light` | its cheapest enabled level |
| `normal` | its default enabled level (`default_enabled_level`) |
| `hard` | its deepest level |

A model with one level always gets it. A model without thinking ignores the tier.
`tier-v1` names these semantics; a change to them ships as a new policy name.

Examples: Kimi K3.1 Pro (`low`, `high`, `max`; default `max`) gets low, low, max, max;
GPT-6 Astra (`low` to `xhigh`; default `high`) gets low, low, high, xhigh.

## Rules

- `model_gateway.reasoning_tier` works on `/v1/chat/completions`, `/v1/responses` and
  `/v1/messages`. There is no `auto`: classification belongs to the client.
- A tier and a concrete level in one request (`reasoning_effort`, `reasoning.effort`,
  `thinking`, `output_config.effort`, `chat_template_kwargs`) is a 400; so is an
  unknown tier. A client that steers a model mid-run sends concrete levels from then on.
- A tier overrides a consumer profile's default `reasoning_effort`.
- The tier is placed on the model that serves the request: after a vision reroute,
  the fallback model's levels.
- Federation: the gateway that receives the tier places it once, on the peer's
  published levels and thinking format, and forwards a concrete level. A peer that then
  reroutes the request for vision checks that level against its fallback model.

## Discovery

Every `/v1/models` row carries `thinking_tiers` (each tier's level for that model) and
`thinking_tier_policy`, both `null` for a model without thinking. `/v1/models/canonical`
adds them with `?include=thinking_tiers`. A client that needs the level before sending
(to show it, or to start steering from it) reads it here.

## Response

A request with a tier answers with:

| Header | Value |
|---|---|
| `X-Model-Gateway-Reasoning-Tier` | the requested tier |
| `X-Model-Gateway-Reasoning-Level` | the level sent upstream, after provider quirks (absent when none is sent) |
| `X-Model-Gateway-Reasoning-Policy` | `tier-v1` |

Headers arrive before a streamed body. Usage and cost come from the provider's token
counts, never from the tier.
