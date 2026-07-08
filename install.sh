#!/usr/bin/env bash
# Bootstrap entrypoint for the model-gateway service.
#
# Usage:
#   ./install.sh             # bootstrap + start + verify
#   ./install.sh --no-start  # bootstrap only (no service start)
#
# Prerequisites: macOS, Homebrew uv, git, python3, curl.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/bin/model-gateway" install "$@"
