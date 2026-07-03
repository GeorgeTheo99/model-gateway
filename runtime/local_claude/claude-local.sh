#!/bin/bash
# Launch Claude Code with shared subscription login from ~/.claude
set -euo pipefail

unset ANTHROPIC_BASE_URL
unset ANTHROPIC_API_KEY
unset ANTHROPIC_AUTH_TOKEN
unset CLAUDE_CONFIG_DIR
unset CLAUDE_CODE_MAX_OUTPUT_TOKENS
unset API_TIMEOUT_MS

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export ANTHROPIC_MODEL="claude-opus-4-6"
export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-4-6"
export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4-6"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-sonnet-4-6"

exec claude "$@"
