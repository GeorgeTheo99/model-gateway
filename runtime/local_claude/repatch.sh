#!/bin/bash
# Repatch Claude Code CLI for local model usage
# Run this after: npm update -g @anthropic-ai/claude-code
set -euo pipefail

SRC="/opt/homebrew/lib/node_modules/@anthropic-ai/claude-code"
DST="$(dirname "$0")/patched-claude"
TARGET='case"macos":return"/Library/Application Support/ClaudeCode"'
REPLACEMENT='case"macos":return"/dev/null/NoManagedSettings"'

echo "Source version: $(node -e "console.log(require('$SRC/package.json').version)")"
echo "Patched CLI:    $DST/cli.js"

# Back up current cli.js before overwriting (for rollback.sh)
if [ -f "$DST/cli.js" ]; then
  cp "$DST/cli.js" "$DST/cli.js.bak"
  echo "Backup saved:   $DST/cli.js.bak"
  echo "  (run ./rollback.sh to restore if upgrade fails)"
fi

# Copy fresh CLI
cp "$SRC/cli.js" "$DST/cli.js"

# Verify target string exists
if ! grep -q "$TARGET" "$DST/cli.js"; then
  echo "ERROR: Target string not found in cli.js - Claude Code may have changed the managed settings path."
  echo "Search manually: rg 'Application Support' $DST/cli.js"
  exit 1
fi

# Apply patch
sed -i '' "s|$TARGET|$REPLACEMENT|" "$DST/cli.js"

# Verify
if grep -q 'NoManagedSettings' "$DST/cli.js"; then
  echo "Patch applied successfully."
else
  echo "ERROR: Patch verification failed."
  exit 1
fi

# Refresh symlinks (in case structure changed)
for f in node_modules resvg.wasm tree-sitter-bash.wasm tree-sitter.wasm vendor package.json; do
  rm -f "$DST/$f"
  if [ -e "$SRC/$f" ]; then
    ln -s "$SRC/$f" "$DST/$f"
  fi
done

echo "Symlinks refreshed. Done."
echo "New version: $(node -e "console.log(require('$DST/package.json').version)")"
