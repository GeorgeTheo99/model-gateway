#!/bin/bash
# Rollback to previous patched CLI version
set -euo pipefail

DST="$(dirname "$0")/patched-claude"

if [ ! -f "$DST/cli.js.bak" ]; then
  echo "ERROR: No backup found at $DST/cli.js.bak"
  exit 1
fi

cp "$DST/cli.js.bak" "$DST/cli.js"
echo "Rolled back to previous version."
echo "Version: $(node -e "console.log(require('$DST/package.json').version)")"
