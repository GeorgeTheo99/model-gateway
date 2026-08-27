# Provider Pricing Sources

Authoritative sources for token pricing used when adding a model to
`model-gateway/model-info.json` or reviewing prices for drift. The gateway
routes to the providers below; **price each model from the provider we actually
route to**, not from aggregators (LiteLLM, morphllm, etc.). Aggregators are
useful only as a cross-check, never as the source of record.

## Pricing field schema (in model-info.json)

```json
"pricing": {
  "input": 3.0,
  "output": 15.0,
  "cache_read": 0.3,
  "cache_write": 3.75,
  "cache_write_1h": 6.0
}
```

- All values are **USD per 1,000,000 tokens** ($/Mtok).
- `input` and `output` are required for a priced entry.
- `cache_read` / `cache_write` / `cache_write_1h`: include only when the
  provider reports those token classes **and** the gateway receives them.
  `cache_write` means the default 5-minute write; `cache_write_1h` means the
  1-hour write. The gateway receives:
  - Anthropic: `cache_read_input_tokens` + `cache_creation_input_tokens`, split
    by `cache_creation.ephemeral_5m_input_tokens` and
    `cache_creation.ephemeral_1h_input_tokens` ✓
  - OpenAI / Fireworks / OpenRouter / Z.ai: `prompt_tokens_details.cached_tokens`
    (cache reads only; cache writes are not reported by these providers) — set
    `cache_read` when the provider charges a distinct cache-read rate, omit
    `cache_write`.
- `reasoning`: omit unless the provider bills reasoning tokens at a separate
  rate from output. None of the current providers do.
- If a rate is genuinely unknown, **omit the whole `pricing` field** — the
  ledger records `cost_usd = NULL` (unknown). Never guess or copy a rate from a
  different model/provider.
- Local/oMLX models have no provider token charge. Mark them explicitly with
  `"pricing_status": "unmetered"` and omit `pricing`; the ledger records known
  `$0` cost while tracking token-reporting coverage separately.

## Provider → official pricing source

| Provider (in model-info.json) | Official pricing source | Format | Cache fields |
|---|---|---|---|
| `anthropic` | https://www.anthropic.com/pricing | HTML, per-model blocks | Write + Read |
| `openai` | https://platform.openai.com/docs/pricing | HTML table | cached input (read only) |
| `fireworks` | https://docs.fireworks.ai/serverless/pricing | HTML table (per-model: input / cached input / output) | cached input (read only) |
| `openrouter` | https://openrouter.ai/api/v1/models | **JSON API** (machine-readable) | `prompt_cache_read` / `prompt_cache_write` (often null) |
| `moonshot` | https://platform.kimi.ai/docs/pricing/chat-k3 | HTML table | cached input (read only) |
| `zai_coding` | https://docs.z.ai/guides/overview/pricing | HTML | none reported |
| `zhipuai` | https://open.bigmodel.cn/pricing | HTML (Chinese site) | none reported |

### Parsing notes per provider

**Anthropic** (verified 2026-07-24): the page has a block per model
with input/output and prompt-caching rates. Map directly: input→input,
output→output, 5-minute write→cache_write, 1-hour write→cache_write_1h, and
cache hit/refresh→cache_read. Anthropic documents cache_read = 0.1× input,
cache_write = 1.25× input, and cache_write_1h = 2× input; prefer concrete
numbers from the current official page.

**OpenAI**: the pricing page lists models in a table with Input / Output /
Cached input per 1M tokens. GPT-5.x models report a cached-input rate (0.5×
input typical) but no cache-write field. Map: input→input, output→output,
Cached input→cache_read. Omit cache_write. Same OpenAI-shape cost-model caveat
as Fireworks applies (see above).

**Fireworks** (verified 2026-08-01): the serverless pricing docs page
(https://docs.fireworks.ai/serverless/pricing) lists each headline model with
three per-1M-token figures: **input / cached input / output**. Kimi K3 Standard
is `$3.00 / $0.30 / $15.00`. The `https://fireworks.ai/pricing` marketing page
defers to the docs page for actual rates — use the docs page. Map: input→input,
"cached input"→cache_read, output→output. Omit cache_write (Fireworks reports
cache reads only; the gateway receives `prompt_tokens_details.cached_tokens`).

⚠️ **Cost-model caveat (OpenAI-shape providers: Fireworks, OpenAI, OpenRouter):**
`prompt_tokens` *includes* cached tokens, while Anthropic's `input_tokens`
*excludes* them. `src/usage.py` `extract_usage()` normalizes this so
`input_tokens` is always cache-miss input (it subtracts `cached_tokens` from
`prompt_tokens` for OpenAI-shape responses). This makes `estimate_cost()`
uniform across providers and avoids double-counting cached tokens. If you ever
change that normalization, re-verify the cost math for both shapes.

> ⚠️ Common drift bug: Fireworks model prices differ from the same model on
> Z.ai direct or OpenRouter. Always read the price off the **Fireworks** docs
> page for `provider: fireworks` models, not Z.ai's page. (This is how
> `glm-5.1-fw` got the wrong rate: the Z.ai GLM-5 price $1/$3 was used instead
> of the Fireworks GLM-5.1 price $1.40/$4.40.)

**OpenRouter** (JSON, verified 2026-06-28): `GET
https://openrouter.ai/api/v1/models` returns `{data: [...]}`. Each model has
`id` (matches `provider_model_id` in model-info.json, e.g.
`google/gemini-3-flash-preview`) and `pricing` with `prompt`, `completion`
(both **$/token** — multiply ×1,000,000 for $/Mtok), and optionally
`prompt_cache_read` / `prompt_cache_write`. Match by `provider_model_id`.
This is the only provider with a machine-readable source — prefer it for
`openrouter` models. For `prompt_cache_read`/`prompt_cache_write`: if null,
omit the cache field (Google via OpenRouter does not report cache tokens
reliably). Same OpenAI-shape cost-model caveat as Fireworks applies (see
above) when `prompt_cache_read` is non-null.

**Moonshot** (verified 2026-07-20): Kimi K3 is priced per 1M tokens at
`$3.00` cache-miss input, `$0.30` cache-hit input, and `$15.00` output. Map
these to `input`, `cache_read`, and `output` respectively. Automatic context
caching is included; no separate cache-write token class is reported.

**Z.ai (`zai_coding`)**: the Z.ai docs pricing page lists GLM models with
input/output per 1M tokens. No cache fields. Match by `provider_model_id`
(e.g. `glm-5.2`). Note: the `zai_coding` provider points at
`api.z.ai/api/coding/paas/v4` (the coding tier) — confirm the price is for the
**coding** endpoint, which sometimes differs from the standard Z.ai API tier.
GLM-5.3's Coding Plan publishes credit multipliers rather than USD/Mtok prices,
so `glm-5.3-zai` intentionally omits `pricing` until a directly comparable
price is published. Its current compatibility route retains upstream ID
`glm-5.2`, which Z.ai documents as automatically routed to GLM-5.3.
GLM-5.3-Flash likewise omits USD pricing for the Coding Plan; Z.ai documents it
in quota/points terms (with 3× the available quota compared with GLM-5.3).

**ZhipuAI (`zhipuai`)**: the BigModel pricing page
(https://open.bigmodel.cn/pricing) lists GLM models in CNY per 1M tokens.
Convert CNY→USD at the current rate and note the conversion date in the commit
message. `glm-5-turbo` is the current relevant ID; the former `glm-5.1` route
was retired from this gateway catalog.

## Model → provider → upstream lookup (current catalog)

Use this to know which source to read for each model. Regenerate with:
`python3 -c "import json; [print(f\"{e['name']:24} {e.get('provider','?'):12} {e.get('provider_model_id','?')}\") for e in json.load(open('model-gateway/model-info.json'))['llm'] if e.get('provider')]"`

| Gateway model | Provider | Upstream model id | Source page |
|---|---|---|---|
| claude-fable-5 | anthropic | claude-fable-5 | anthropic |
| claude-opus-5 / 4.8 / 4.7 / 4.6 / 4.5 | anthropic | claude-opus-* | anthropic |
| claude-sonnet-4.6 | anthropic | claude-sonnet-4-6 | anthropic |
| gpt-5.4 | openai | gpt-5.4 | openai |
| gpt-5.4-mini | openai | gpt-5.4-mini | openai |
| deepseek-v4-pro-fw | fireworks | accounts/fireworks/models/deepseek-v4-pro | fireworks |
| glm-5.2-fw | fireworks | accounts/fireworks/models/glm-5p2 | fireworks |
| kimi-k3 | fireworks | accounts/fireworks/models/kimi-k3 | fireworks |
| minimax-m3-fw | fireworks | accounts/fireworks/models/minimax-m3 | fireworks |
| qwen3.7-plus-fw | fireworks | accounts/fireworks/models/qwen3p7-plus | fireworks |
| deepseek-v4-flash-or | openrouter | deepseek/deepseek-v4-flash | openrouter API |
| gemini-3.1-pro | openrouter | google/gemini-3.1-pro-preview | openrouter API |
| gemini-3-flash | openrouter | google/gemini-3-flash-preview | openrouter API |
| gemini-3.5-flash | openrouter | google/gemini-3.5-flash | openrouter API |
| glm-5.3-zai | zai_coding | glm-5.2 (documented GLM-5.3 compatibility route) | z.ai Coding Plan |
| glm-5.3-flash-zai | zai_coding | glm-5.3-flash | z.ai Coding Plan |
| glm-5-zai-turbo | zhipuai | glm-5-turbo | zhipuai (CNY→USD) |
