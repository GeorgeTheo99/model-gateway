#!/usr/bin/env bash
# PreToolUse hook: block WebSearch/WebFetch for local/cloud model sessions.
# Anthropic's built-in WebSearch/WebFetch only work against the real Anthropic API.
# When ANTHROPIC_BASE_URL is set (oMLX or cloud gateway), they silently fail or
# return 0 results. This hook forces Claude to use the MCP websearch tools instead.
#
# When ANTHROPIC_BASE_URL is unset (claude-default / Anthropic direct),
# the built-in tools work fine, so this hook exits 0 (allow).
#
# Exit codes: 0 = allow, 2 = block (Claude sees stderr and picks a different tool)

if [ -z "${ANTHROPIC_BASE_URL:-}" ]; then
  exit 0  # Direct Anthropic session — built-in tools work
fi

echo "Blocked: Built-in WebSearch/WebFetch don't work with local/cloud models. Use MCP tools mcp__websearch__web_search or mcp__websearch__web_fetch instead." >&2
exit 2
