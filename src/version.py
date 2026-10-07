"""Release version and capabilities, for products that attach to this gateway.

Consumers compare these against their own minimum before using the gateway;
``/health`` stays version-free because older installers match it exactly.
"""

VERSION = "0.5.1"

# Contracts a consumer may require. Add a name when a feature ships; never
# remove or change the meaning of one.
CAPABILITIES = (
    "consumer_credentials.v1",
    "create_only_model_registration",
    "endpoint_file.v1",
    "legacy_home_server_import.v1",
    "local_runtime.v1",
    "profiles.v1",
    "scoped_manager.v1",
)
