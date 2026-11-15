# Usage tags: what a client's own requests cost

A client labels a model request with `X-Model-Gateway-Usage-Tag: <tag>`, then asks the gateway what the
requests with that label cost. Myai uses it for a run's receipt, its check-in amount and the owner's
monthly limit. The gateway still prices every request (the ledger, `src/usage.py`), so the client
never has to work out prices itself.

## Tagging a request

- **Header:** `X-Model-Gateway-Usage-Tag`, 1 to 120 characters from `A-Z a-z 0-9 . _ : -`. An invalid value is
  ignored: the request runs and is recorded untagged, never refused.
- **Where:** every ledgered request (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`, image and video
  generation). The ledger row keeps the tag (`usage_tag`) and who sent it (`usage_owner`: `consumer:<credential
  id>` for a consumer credential, else the kind of key: `legacy`, `admin` or `anonymous`).
- **Choosing tags:** a tag is a client's own name; make it unguessable and unique where it matters, for example
  `myai-<install id>:run:<run id>`.

## Reading it back

`GET /v1/usage`, with the same client key that sent the requests (another key's rows are never counted):

| Query | Meaning |
|---|---|
| `tag=<tag>` | Requests with exactly this tag |
| `tag_prefix=<prefix>` | Requests whose tag starts with it (at least 8 characters; `%` and `_` match only themselves) |
| `since=<epoch seconds>` | Optional: only requests from then on |

Pass exactly one of `tag` and `tag_prefix`; anything else is `400`. The answer (`Cache-Control: no-store`):

```json
{"requests": 3, "cost_usd": 0.0421, "unpriced_requests": 0,
 "input_tokens": 120000, "output_tokens": 4100, "cached_read_tokens": 80000}
```

`cost_usd` sums the priced requests. `unpriced_requests` counts requests whose cost is unknown or only partly
priced (no pricing for the model, or usage not reported), so a client can say "at least" instead of presenting an
incomplete total as complete. A plan-billed or local model (`pricing_status: unmetered`) costs `0`. A streamed
request is recorded when its stream ends, so read a run's total after its last response has finished.
