"""Release version and capabilities, for products that attach to this gateway.

Consumers compare these against their own minimum before using the gateway;
``/health`` stays version-free because older installers match it exactly.
"""

VERSION = "0.8.0"

# Contracts a consumer may require. Add a name when a feature ships; never
# remove or change the meaning of one.
CAPABILITIES = (
    "consumer_credentials.v1",
    "create_only_model_registration",
    "catalog_packs.v1",
    "custom_providers.v1",
    "endpoint_file.v1",
    "layered_catalog.v1",
    "legacy_home_server_import.v1",
    "local_runtime.v1",
    "onboarding_auto_add.v1",
    "onboarding_profiles.v1",
    "profiles.v1",
    "scoped_manager.v1",
)
