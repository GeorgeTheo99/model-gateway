# local_claude

Compatibility launcher for running Claude Code with shared subscription login.

## What this does

- Launches the standard `claude` CLI using the shared `~/.claude` config
- Clears local/oMLX auth environment variables so saved `/login` credentials can be used
- Keeps the `claude-local.sh` entrypoint available for compatibility

## Usage

```bash
# Launch with subscription login
./claude-local.sh

# Or add a shell alias to ~/.zshrc:
alias claude-local='~/local_code/localserver99/local_claude/claude-local.sh'
```

## Configuration

Login state lives in `~/.claude/.credentials.json`.

The launcher clears these environment variables before starting Claude Code:
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_AUTH_TOKEN`
- `CLAUDE_CONFIG_DIR`
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS`
- `API_TIMEOUT_MS`

## Architecture

```
claude-local.sh
  → clears local auth overrides
  → claude
    → ~/.claude/.credentials.json
      → Anthropic subscription login
```

## oMLX Local Model Launcher

The `_claude_local` function in `~/.zshrc` launches Claude Code against local models via oMLX. Model aliases (e.g. `claude-qwen35`) are generated dynamically from `~/.claude/model-aliases.json`.

### MCP Servers for Local Models

Claude Code's built-in tools (WebSearch, WebFetch) do not work with local models — they require Anthropic's backend. MCP servers also do not load from `~/.claude/settings.json` when using a custom `ANTHROPIC_BASE_URL`.

### WebSearch/WebFetch hook

A `PreToolUse` hook in `settings.json` mechanically blocks `WebSearch` and `WebFetch` when `ANTHROPIC_BASE_URL` is set (local/cloud sessions). The model sees the error message and reroutes to the MCP `websearch` tools instead. When `ANTHROPIC_BASE_URL` is unset (`claude-default` / direct Anthropic), the hook allows the built-in tools through.

Hook script: `local_claude/block-cloud-tools.sh` — exits 0 if `ANTHROPIC_BASE_URL` is unset, exits 2 (block) otherwise.

### Current MCP servers

- **websearch** — SearXNG wrapper (`~/local_code/local-search/mcp-websearch/server.py`), provides `web_search` and `web_fetch` tools. Runs in a uv-managed Python 3.12 venv at `~/local_code/local-search/mcp-websearch/.venv/`.

### Server-specific Claude instructions

`~/.claude/CLAUDE.md` on the server contains instructions that only apply to server sessions (e.g. directing the model to use MCP `web_search` instead of built-in `WebSearch`).
