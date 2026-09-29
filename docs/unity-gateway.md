# Unity Gateway routing

Databricks Unity Gateway model services use fully qualified Unity Catalog names
(such as `system.ai.gpt-6-astra`), not legacy serving-endpoint names. Query model
services through the workspace's `/ai-gateway` data plane. OAuth authentication
and refresh still use the workspace root.

Example private configuration (supply credentials locally):

```yaml
providers:
  workspace:
    base_url: https://<workspace-url>/ai-gateway
    workspace_url: https://<workspace-url>
    auth_refresh: databricks-cli
    auth_profile: <existing-profile>
    protocol: openai
    path_prefixes:
      openai: mlflow/v1
      anthropic: anthropic/v1
      responses: openai/v1
    quirks: [anthropic_bearer_auth]

models:
  - name: gpt-6-astra
    alias: astra
    provider: workspace
    provider_model_id: system.ai.gpt-6-astra
    alternate_ids: [databricks-gpt-6-astra]
    api_style: open_responses
```

For `api_style: open_responses`, `path_prefixes.responses` selects the Responses
API prefix. If absent, `path_prefixes.openai` selects the unified Responses API.
The resolver appends `/responses` and returns a complete endpoint for both
native passthrough and pool failover. Providers with neither prefix retain the
legacy `/serving-endpoints/open-responses` behavior for backward compatibility;
a Unity-only installation must not leave such providers in any model pool.

- Native OpenAI Responses: `/ai-gateway/openai/v1/responses`.
- Unified Chat Completions: `/ai-gateway/mlflow/v1/chat/completions`.
- Native Anthropic Messages: `/ai-gateway/anthropic/v1/messages`.
- Preserve old endpoint identifiers in `alternate_ids` when changing
  `provider_model_id`, so existing clients and active sessions keep routing.
- Update `available_model_ids` and both sides of `model_fallbacks` to the
  discovered Unity model service names; fallback keys are upstream IDs, not aliases.
- Remove `endpoint_style: invocations` and legacy-only provider quirks when
  converting a provider; preserve model-specific quirks unless tested otherwise.
- Convert **every** pool member, not just each model's primary provider.
- A workspace URL alone is not evidence of legacy routing: the API path matters.

## Verification and activation

Discover service names and supported APIs with the existing workspace profile:

```sh
databricks ai-gateway list-model-services \
  --parent schemas/system.ai --profile <existing-profile> -o json
```

Stage the private config separately. Resolve every enabled model/member pair
against that candidate and reject any URL containing `/serving-endpoints/`.
Send a bounded completion to every resolved Unity route, including Responses,
then verify streaming and function-tool traffic through the local gateway.
Activate only after testing, with a private backup and rollback available.

The existing `workspace add/replace/test` bootstrap checks discover legacy
serving endpoints; they are **not** proof of Unity model-service coverage or
Responses compatibility. Do not use `--style auto` for a Unity-only migration:
it may fall back to legacy invocations. Use explicit, verified private config
until those workflows support Unity model-service discovery end to end.

Reference: [Query model APIs (model services)](https://docs.databricks.com/aws/en/ai-gateway/query-model-services).
