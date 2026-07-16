# Provider onboarding

OpenAI-compatible providers and replacement models can be added with one
tracked, secret-free profile instead of hand-editing runtime files.

```bash
model-gateway onboard moonshot-kimi-k3 --dry-run
model-gateway onboard moonshot-kimi-k3
```

The non-dry run prompts for the API key with hidden input, validates that the
provider's authenticated `/models` endpoint advertises every requested model,
then:

1. writes the key outside the checkout to `~/.config/model-gateway/secrets/<provider>.api-key` with mode `0600`;
2. adds the provider by `api_key_file` reference;
3. atomically replaces the listed retired models in `model-info.json`;
4. mirrors the catalog to `MODEL_GATEWAY_MODEL_INFO_SOURCE`;
5. restarts the service, regenerates configured exports, and verifies health.

Every touched file is snapshotted in memory. If a handled write, service
reload, catalog-export, or health verification error occurs, the command
restores those snapshots and reloads the previous configuration. An abrupt
process or machine termination is outside this rollback guarantee. Profiles
are idempotent: rerunning one updates the same
provider/model without requiring the retired model to still exist.

## Supplying secrets safely

Never put a key in a command argument. For automation, use an environment
variable or macOS Keychain:

```bash
model-gateway onboard PROFILE --api-key-env PROVIDER_API_KEY
model-gateway onboard PROFILE \
  --api-key-keychain-service model-gateway-provider
```

To load a key into Keychain without shell history or terminal echo:

```bash
read -s PROVIDER_API_KEY
security add-generic-password -U -a "$USER" \
  -s model-gateway-provider -w "$PROVIDER_API_KEY"
unset PROVIDER_API_KEY
```

## Profile format

Profiles live in `config/onboarding/*.yaml`:

```yaml
schema_version: 1
id: example-model
provider:
  id: example
  base_url: https://api.example.com/v1
  protocol: openai
  secret_name: example.api-key
models:
  - name: example-model
    provider: example
    provider_model_id: example-model
    alias: example
    context: 128000
    max_output_tokens: 16000
    thinking: optional
    vision: false
    quirks: [no_stream_options]
retire:
  models: [example-model-old]
```

Supported reusable request quirks include:

- `no_stream_options`
- `no_reasoning_params`
- `reasoning_none_with_tools`
- `force_reasoning_effort_max`
- `use_max_completion_tokens`
- `drop_fixed_sampling_fields`
- `inline_image_urls_only`

Add a new profile, run its dry-run, and then onboard it. Provider-specific
request code should only be added when the upstream cannot be described by
these generic compatibility flags.
