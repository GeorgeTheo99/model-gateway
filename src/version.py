"""Release version and capabilities, for products that attach to this gateway.

Consumers compare these against their own minimum before using the gateway;
``/health`` stays version-free because older installers match it exactly.
"""

VERSION = "0.13.0"

# Contracts a consumer may require. Add a name when a feature ships; never
# remove or change the meaning of one.
CAPABILITIES = (
    "consumer_credentials.v1",
    "create_only_model_registration",
    "catalog_packs.v1",
    "custom_providers.v1",
    "endpoint_file.v1",
    "federation_catalog.v2",
    "layered_catalog.v1",
    "legacy_home_server_import.v1",
    "local_runtime.v1",
    "local_runtime_models.v1",
    "local_runtime_on_demand.v1",
    "local_runtime_pins.v1",
    "media_generation.v1",
    "model_allowlists.v1",
    "onboarding_auto_add.v1",
    "onboarding_profiles.v1",
    "profiles.v1",
    "reasoning_tiers.v1",
    "scoped_manager.v1",
    "text_only_image_notes.v1",
    "trusted_local_profiles.v1",
)
