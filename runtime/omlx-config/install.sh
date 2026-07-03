#!/bin/bash
# Install oMLX and configure it as the Claude Code inference backend.
# Run on localserver99: bash ~/local_code/server/claude-proxy/install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OMLX_PORT=9110
CACHE_DIR="$HOME/.omlx/cache"
MODEL_DIR="$HOME/models/mlx"
LOG_DIR="$HOME/.claude"
PLIST_LABEL="com.local.claude-proxy"
PLIST_SRC="$SCRIPT_DIR/$PLIST_LABEL.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_LABEL.plist"

echo "=== Installing oMLX for Claude Code ==="

# 1. Install oMLX via pip (into user site-packages)
if ! command -v omlx &>/dev/null; then
    echo "Installing oMLX..."
    pip3 install omlx
else
    echo "oMLX already installed: $(omlx --version 2>/dev/null || echo 'unknown version')"
    echo "Upgrading..."
    pip3 install --upgrade omlx
fi

# Verify
if ! command -v omlx &>/dev/null; then
    echo "ERROR: omlx not found in PATH after install"
    echo "You may need to add pip's bin dir to PATH"
    exit 1
fi

# 2. Create cache directory
mkdir -p "$CACHE_DIR"
echo "SSD cache dir: $CACHE_DIR"

# 3. Verify model directory
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model directory not found: $MODEL_DIR"
    exit 1
fi
echo "Model dir: $MODEL_DIR ($(ls -d "$MODEL_DIR"/*/ 2>/dev/null | wc -l | tr -d ' ') models)"

# 4. Install launchd plist
if [ ! -f "$PLIST_SRC" ]; then
    echo "ERROR: plist not found: $PLIST_SRC"
    exit 1
fi
cp "$PLIST_SRC" "$PLIST_DST"
echo "Installed launchd plist: $PLIST_DST"

# 5. Load the service
LAUNCHD_TARGET="gui/$(id -u)/$PLIST_LABEL"
launchctl bootout "$LAUNCHD_TARGET" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
echo "Service loaded: $PLIST_LABEL"

# 6. Wait for health
echo -n "Waiting for oMLX on port $OMLX_PORT..."
elapsed=0
while [ $elapsed -lt 30 ]; do
    if curl -sf --max-time 2 "http://localhost:$OMLX_PORT/health" >/dev/null 2>&1; then
        echo " ready (${elapsed}s)"
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
if [ $elapsed -ge 30 ]; then
    echo " TIMEOUT"
    echo "Check logs: $LOG_DIR/claude-proxy.log"
    exit 1
fi

# 7. Symlink model-aliases.json from local_claude
ALIASES_SRC="$SCRIPT_DIR/../local_claude/model-aliases.json"
ALIASES_DST="$LOG_DIR/model-aliases.json"
if [ -f "$ALIASES_SRC" ]; then
    ln -sf "$ALIASES_SRC" "$ALIASES_DST"
    echo "Symlinked model aliases: $ALIASES_DST -> $ALIASES_SRC"
else
    echo "WARNING: model-aliases.json not found at $ALIASES_SRC"
fi

echo ""
echo "=== Done ==="
echo "oMLX running on port $OMLX_PORT with SSD cache at $CACHE_DIR"
echo "Use claude-<model> to launch Claude Code, or claude-list to see options"
