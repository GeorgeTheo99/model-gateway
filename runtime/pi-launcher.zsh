# ── Pi / Claude Code / Codex Launcher (oMLX + model-gateway) ──
# Interactive sessions route through model-gateway (port 9111); local MLX
# requests are proxied onward to oMLX (port 9110). oMLX still handles model
# loading, SSD KV cache, and local inference. Model metadata comes from oMLX
# API + model-info.json. Aliases come from ~/.claude/model-aliases.json.

OMLX_PORT=9110
OMLX_URL="http://localhost:$OMLX_PORT"
OMLX_API_KEY="omlx"
CLOUD_GW_PORT=9111
CLOUD_GW_URL="http://localhost:$CLOUD_GW_PORT"
CLOUD_GW_API_KEY="cloud"
_ALIAS_FILE="${HOME}/.claude/model-aliases.json"
_MODEL_INFO_FILE="${MODEL_INFO_PATH:-${MODEL_GATEWAY_RUNTIME_MODEL_INFO:-${HOME}/local_code/model-gateway/runtime/model-info.json}}"

# ── Load models from oMLX API + alias file ──────────────────────────

typeset -gA _OMLX_CTX _OMLX_MAX_OUT _OMLX_ALIAS _OMLX_DESC _OMLX_LOADED _OMLX_UNSUPPORTED_DESC _OMLX_UNSUPPORTED_NOTE
typeset -ga _OMLX_IDS _OMLX_UNKNOWN _OMLX_UNSUPPORTED_IDS

_load_omlx_models() {
  _OMLX_IDS=()
  _OMLX_UNKNOWN=()
  _OMLX_UNSUPPORTED_IDS=()
  _OMLX_CTX=()
  _OMLX_MAX_OUT=()
  _OMLX_ALIAS=()
  _OMLX_DESC=()
  _OMLX_LOADED=()
  _OMLX_UNSUPPORTED_DESC=()
  _OMLX_UNSUPPORTED_NOTE=()

  local json
  json=$(curl -sf --max-time 3 -H "Authorization: Bearer $OMLX_API_KEY" "$OMLX_URL/v1/models/status" 2>/dev/null)

  if [ -n "$json" ]; then
    # Parse oMLX API response. Models without an entry in the alias file
    # are reported as "unknown" — never invent an alias by splitting the id,
    # because that silently collapses sibling models (e.g. gemma4-26b vs
    # gemma4-31b) into a single claude-<family> function.
    eval "$(OMLX_JSON="$json" ALIAS_FILE="$_ALIAS_FILE" python3 -c "
import json, os, shlex, sys
try:
    data = json.loads(os.environ['OMLX_JSON'])
except Exception:
    sys.exit(1)
aliases = {}
try:
    with open(os.environ['ALIAS_FILE']) as f:
        aliases = json.load(f)
except Exception:
    pass
for m in data.get('models', []):
    mid = m['id']
    ctx = m.get('max_context_window', 32768)
    mt = m.get('max_tokens', 32768)
    loaded = 'yes' if m.get('loaded') else 'no'
    a = aliases.get(mid)
    if not a:
        print(f'_OMLX_UNKNOWN+=({shlex.quote(mid)})')
        continue
    if a.get('supported') is False:
        qmid = shlex.quote(mid)
        desc = a.get('desc') or mid
        note = a.get('support_note') or 'unsupported by current runtime'
        print(f'_OMLX_UNSUPPORTED_IDS+=({qmid})')
        print(f'_OMLX_UNSUPPORTED_DESC[{qmid}]={shlex.quote(desc)}')
        print(f'_OMLX_UNSUPPORTED_NOTE[{qmid}]={shlex.quote(note)}')
        continue
    alias = a['alias']
    desc = a.get('desc', mid)
    qmid = shlex.quote(mid)
    print(f'_OMLX_IDS+=({qmid})')
    print(f'_OMLX_CTX[{qmid}]={ctx}')
    print(f'_OMLX_MAX_OUT[{qmid}]={mt}')
    print(f'_OMLX_ALIAS[{qmid}]={shlex.quote(alias)}')
    print(f'_OMLX_DESC[{qmid}]={shlex.quote(desc)}')
    print(f'_OMLX_LOADED[{qmid}]={loaded}')
" 2>/dev/null)"
  else
    # oMLX unreachable — fall back to alias file only (no ctx/max_out/loaded)
    if [ -f "$_ALIAS_FILE" ]; then
      eval "$(ALIAS_FILE="$_ALIAS_FILE" python3 -c "
import json, os, shlex
try:
    with open(os.environ['ALIAS_FILE']) as f:
        aliases = json.load(f)
except Exception:
    raise SystemExit(1)
for mid, a in aliases.items():
    if mid.startswith('cloud:'):
        continue
    if a.get('supported') is False:
        qmid = shlex.quote(mid)
        desc = a.get('desc') or mid
        note = a.get('support_note') or 'unsupported by current runtime'
        print(f'_OMLX_UNSUPPORTED_IDS+=({qmid})')
        print(f'_OMLX_UNSUPPORTED_DESC[{qmid}]={shlex.quote(desc)}')
        print(f'_OMLX_UNSUPPORTED_NOTE[{qmid}]={shlex.quote(note)}')
        continue
    if not a.get('alias'):
        continue
    alias = a['alias']
    desc = a.get('desc', mid)
    qmid = shlex.quote(mid)
    print(f'_OMLX_IDS+=({qmid})')
    print(f'_OMLX_CTX[{qmid}]=0')
    print(f'_OMLX_MAX_OUT[{qmid}]=0')
    print(f'_OMLX_ALIAS[{qmid}]={shlex.quote(alias)}')
    print(f'_OMLX_DESC[{qmid}]={shlex.quote(desc)}')
    print(f'_OMLX_LOADED[{qmid}]=unknown')
" 2>/dev/null)"
    fi
  fi
}

_load_omlx_models

# ── Load cloud models from alias file ────────────────────────────────

typeset -gA _CLOUD_ALIAS _CLOUD_DESC _CLOUD_NAME _CLOUD_PROVIDER _CLOUD_PROVIDER_MODEL _CLOUD_CTX _CLOUD_MAX_OUT
typeset -ga _CLOUD_IDS _REMOTE_CLOUD_IDS

_load_cloud_models() {
  _CLOUD_IDS=()
  _REMOTE_CLOUD_IDS=()
  _CLOUD_ALIAS=()
  _CLOUD_DESC=()
  _CLOUD_NAME=()
  _CLOUD_PROVIDER=()
  _CLOUD_PROVIDER_MODEL=()
  _CLOUD_CTX=()
  _CLOUD_MAX_OUT=()

  if [ -f "$_ALIAS_FILE" ]; then
    eval "$(ALIAS_FILE="$_ALIAS_FILE" python3 -c "
import json, os, shlex
try:
    with open(os.environ['ALIAS_FILE']) as f:
        aliases = json.load(f)
except Exception:
    raise SystemExit(1)
for key, a in aliases.items():
    if not key.startswith('cloud:'):
        continue
    provider = a.get('provider', '')
    alias = a.get('alias', '')
    desc = a.get('desc', '')
    name = a.get('name', '')
    pm = a.get('provider_model_id', '')
    ctx = a.get('context', 32768)
    mo = a.get('max_output_tokens', 32768)
    qkey = shlex.quote(key)
    print(f'_CLOUD_IDS+=({qkey})')
    print(f'_CLOUD_PROVIDER[{qkey}]={shlex.quote(provider)}')
    print(f'_CLOUD_ALIAS[{qkey}]={shlex.quote(alias)}')
    print(f'_CLOUD_DESC[{qkey}]={shlex.quote(desc)}')
    print(f'_CLOUD_NAME[{qkey}]={shlex.quote(name)}')
    print(f'_CLOUD_PROVIDER_MODEL[{qkey}]={shlex.quote(pm)}')
    print(f'_CLOUD_CTX[{qkey}]={ctx}')
    print(f'_CLOUD_MAX_OUT[{qkey}]={mo}')
    if provider == 'gguf':
        continue
    print(f'_REMOTE_CLOUD_IDS+=({qkey})')
" 2>/dev/null)"
  fi
}

_load_cloud_models

# ── Core launcher ────────────────────────────────────────────────────

_LS99_AI_HELPER="$HOME/local_code/server/scripts/ai-session-env.sh"

_ls99_with_ai_env() {
  local tool="$1"
  local provider="$2"
  local model_id="$3"
  local model_alias="$4"
  local launcher="$5"
  shift 5

  (
    if [ -f "$_LS99_AI_HELPER" ]; then
      source "$_LS99_AI_HELPER"
      ls99_ai_session_env \
        --tool "$tool" \
        --provider "$provider" \
        --model-id "$model_id" \
        --model-alias "$model_alias" \
        --launcher "$launcher"
    else
      echo "Warning: AI attribution helper missing at $_LS99_AI_HELPER" >&2
    fi
    exec "$@"
  )
}

_ensure_omlx() {
  if curl -sf --max-time 2 "$OMLX_URL/health" >/dev/null 2>&1; then
    return 0
  fi
  echo "Starting oMLX..."
  local label="gui/$(id -u)/com.local.claude-proxy"
  if launchctl print "$label" >/dev/null 2>&1; then
    launchctl kickstart -k "$label" >/dev/null 2>&1 || true
  else
    launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.local.claude-proxy.plist 2>/dev/null || true
  fi
  local elapsed=0
  while [ $elapsed -lt 30 ]; do
    curl -sf --max-time 2 "$OMLX_URL/health" >/dev/null 2>&1 && { echo "oMLX ready"; return 0; }
    sleep 2; elapsed=$((elapsed + 2))
  done
  echo "WARNING: oMLX may not be ready — check ~/.claude/claude-proxy.log"
  return 1
}

_ensure_model_gateway() {
  if curl -sf --max-time 2 "$CLOUD_GW_URL/health" >/dev/null 2>&1; then
    return 0
  fi
  echo "Starting model-gateway..."
  local label="gui/$(id -u)/com.local.model-gateway"
  if launchctl print "$label" >/dev/null 2>&1; then
    launchctl kickstart -k "$label" >/dev/null 2>&1 || true
  else
    launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.local.model-gateway.plist 2>/dev/null || true
  fi
  local elapsed=0
  while [ $elapsed -lt 30 ]; do
    curl -sf --max-time 2 "$CLOUD_GW_URL/health" >/dev/null 2>&1 && { echo "model-gateway ready"; return 0; }
    sleep 2; elapsed=$((elapsed + 2))
  done
  echo "WARNING: model-gateway may not be ready — check ~/.claude/model-gateway.log"
  return 1
}

_LS99_QWEN_SESSION_INSTRUCTION='Session discipline: follow the latest user instruction only. If the user requests a handoff, stop coding immediately and only write the handoff plus fresh-session instructions. Before any edit/commit/push, restate the current task and verify it matches the latest user message. If interrupted and resumed, re-read the latest user message and discard stale plans.'

_ls99_session_instruction_for_alias() {
  local alias="$1"
  case "$alias" in
    qwen36|qwen36fw)
      print -r -- "$_LS99_QWEN_SESSION_INSTRUCTION"
      ;;
    *)
      print -r -- ""
      ;;
  esac
}

_ls99_toml_escape() {
  local raw="$1"
  raw="${raw//\\/\\\\}"
  raw="${raw//\"/\\\"}"
  raw="${raw//$'\n'/\\n}"
  print -r -- "$raw"
}

_ls99_auto_compact_limit() {
  local ctx="$1"
  if [[ "$ctx" != <-> ]] || [ "$ctx" -le 0 ]; then
    print -r -- 0
    return
  fi
  print -r -- $((ctx * 95 / 100))
}

_claude_local() {
  local model_id="$1"; shift

  if [ -z "${_OMLX_ALIAS[$model_id]}" ]; then
    echo "Error: unknown model '$model_id'"
    return 1
  fi

  _ensure_omlx || return 1
  _ensure_model_gateway || return 1

  # Fetch fresh context/max_out from oMLX if we had stale data (fallback mode)
  local ctx="${_OMLX_CTX[$model_id]}"
  local max_out="${_OMLX_MAX_OUT[$model_id]}"
  if [ "$ctx" = "0" ] || [ -z "$ctx" ]; then
    # Re-query oMLX now that it's running
    local fresh
    fresh=$(curl -sf --max-time 3 -H "Authorization: Bearer $OMLX_API_KEY" "$OMLX_URL/v1/models/status" 2>/dev/null)
    if [ -n "$fresh" ]; then
      ctx=$(python3 -c "
import json
data = json.loads('''$fresh''')
for m in data.get('models', []):
    if m['id'] == '$model_id':
        print(m.get('max_context_window', 32768))
        break
else:
    print(32768)
" 2>/dev/null)
      max_out=$(python3 -c "
import json
data = json.loads('''$fresh''')
for m in data.get('models', []):
    if m['id'] == '$model_id':
        print(m.get('max_tokens', 32768))
        break
else:
    print(32768)
" 2>/dev/null)
    fi
    ctx="${ctx:-32768}"
    max_out="${max_out:-32768}"
  fi

  local model_alias="${_OMLX_ALIAS[$model_id]}"
  local session_instruction
  session_instruction="$(_ls99_session_instruction_for_alias "$model_alias")"
  local base_url="$CLOUD_GW_URL"
  local auth_token="$CLOUD_GW_API_KEY"
  local backend_label="model-gateway → oMLX"
  local -a claude_cmd
  claude_cmd=(claude --permission-mode bypassPermissions)
  if [ -n "$session_instruction" ]; then
    claude_cmd+=(--append-system-prompt "$session_instruction")
  fi
  claude_cmd+=("$@")

  echo "Claude Code → ${backend_label} (${_OMLX_ALIAS[$model_id]}, ${ctx} context, ${max_out} max output)"
  _ls99_with_ai_env \
    claude \
    model-gateway \
    "$model_id" \
      "${_OMLX_ALIAS[$model_id]}" \
      "claude-${_OMLX_ALIAS[$model_id]}" \
      env \
      ANTHROPIC_BASE_URL="$base_url" \
      ANTHROPIC_AUTH_TOKEN="$auth_token" \
      ANTHROPIC_MODEL="$model_id" \
      ANTHROPIC_DEFAULT_OPUS_MODEL="$model_id" \
      ANTHROPIC_DEFAULT_SONNET_MODEL="$model_id" \
      ANTHROPIC_DEFAULT_HAIKU_MODEL="$model_id" \
      CLAUDE_CODE_MAX_OUTPUT_TOKENS="$max_out" \
      API_TIMEOUT_MS=43200000 \
      CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
      CLAUDE_CODE_ATTRIBUTION_HEADER=0 \
      "${claude_cmd[@]}"
}

# ── Cloud gateway launcher ───────────────────────────────────────────

_claude_cloud() {
  local cloud_id="$1"; shift

  if [ -z "${_CLOUD_ALIAS[$cloud_id]}" ]; then
    echo "Error: unknown cloud model '$cloud_id'"
    return 1
  fi

  _ensure_model_gateway || return 1

  local provider_model="${_CLOUD_PROVIDER_MODEL[$cloud_id]}"
  local ctx="${_CLOUD_CTX[$cloud_id]}"
  local max_out="${_CLOUD_MAX_OUT[$cloud_id]}"
  local model_alias="${_CLOUD_ALIAS[$cloud_id]}"
  local session_instruction
  session_instruction="$(_ls99_session_instruction_for_alias "$model_alias")"
  local -a claude_cmd
  claude_cmd=(claude --permission-mode bypassPermissions)
  if [ -n "$session_instruction" ]; then
    claude_cmd+=(--append-system-prompt "$session_instruction")
  fi
  claude_cmd+=("$@")

  echo "Claude Code → model-gateway (${_CLOUD_ALIAS[$cloud_id]}, ${ctx} context, ${max_out} max output)"
  _ls99_with_ai_env \
    claude \
    model-gateway \
    "$provider_model" \
    "${_CLOUD_ALIAS[$cloud_id]}" \
    "claude-${_CLOUD_ALIAS[$cloud_id]}" \
    env \
      ANTHROPIC_BASE_URL="$CLOUD_GW_URL" \
      ANTHROPIC_AUTH_TOKEN="$CLOUD_GW_API_KEY" \
      ANTHROPIC_MODEL="$provider_model" \
      ANTHROPIC_DEFAULT_OPUS_MODEL="$provider_model" \
      ANTHROPIC_DEFAULT_SONNET_MODEL="$provider_model" \
      ANTHROPIC_DEFAULT_HAIKU_MODEL="$provider_model" \
      CLAUDE_CODE_MAX_OUTPUT_TOKENS="$max_out" \
      API_TIMEOUT_MS=43200000 \
      CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
      CLAUDE_CODE_ATTRIBUTION_HEADER=0 \
      "${claude_cmd[@]}"
}

# ── Anthropic cloud ─────────────────────────────────────────────────

claude-default() {
  echo "Claude Code → Anthropic cloud (native defaults, bypassPermissions)"
  _ls99_with_ai_env \
    claude \
    anthropic \
    "default" \
    "default" \
    "claude-default" \
    env \
      -u ANTHROPIC_BASE_URL \
      -u ANTHROPIC_AUTH_TOKEN \
      -u ANTHROPIC_MODEL \
      -u ANTHROPIC_DEFAULT_OPUS_MODEL \
      -u ANTHROPIC_DEFAULT_SONNET_MODEL \
      -u ANTHROPIC_DEFAULT_HAIKU_MODEL \
      -u CLAUDE_CONFIG_DIR \
      -u CLAUDE_CODE_MAX_OUTPUT_TOKENS \
      -u API_TIMEOUT_MS \
      CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
      claude --permission-mode bypassPermissions "$@"
}

# ── Auto-generate claude-<alias>() for each model ───────────────────

for _mid in "${_OMLX_IDS[@]}"; do
  eval "claude-${_OMLX_ALIAS[$_mid]}(){ _claude_local $_mid \"\$@\"; }"
done
for _cid in "${_CLOUD_IDS[@]}"; do
  eval "claude-${_CLOUD_ALIAS[$_cid]}(){ _claude_cloud $_cid \"\$@\"; }"
done
unset _mid _cid

# ── Utilities ────────────────────────────────────────────────────────

claude-list() {
  echo "Claude Code Launcher Commands:"
  echo ""
  echo "  claude-default       Anthropic cloud (native Claude defaults, bypassPermissions)"
  echo ""
  echo "  Local MLX models (via oMLX on port $OMLX_PORT):"
  for _mid in "${_OMLX_IDS[@]}"; do
    local _lbl=""
    if [ "${_OMLX_LOADED[$_mid]}" = "yes" ]; then
      _lbl=" [loaded]"
    fi
    printf "  %-22s %s%s\n" "claude-${_OMLX_ALIAS[$_mid]}" "${_OMLX_DESC[$_mid]}" "$_lbl"
  done
  if [ ${#_OMLX_UNSUPPORTED_IDS[@]} -gt 0 ]; then
    echo ""
    echo "  Unsupported local MLX models (present, no claude-* launcher):"
    for _mid in "${_OMLX_UNSUPPORTED_IDS[@]}"; do
      printf "    %-28s %s\n" "$_mid" "${_OMLX_UNSUPPORTED_DESC[$_mid]}"
      printf "      %s\n" "${_OMLX_UNSUPPORTED_NOTE[$_mid]}"
    done
  fi
  if [ ${#_REMOTE_CLOUD_IDS[@]} -gt 0 ]; then
    echo ""
    echo "  Cloud models (via model-gateway on port $CLOUD_GW_PORT):"
    for _cid in "${_REMOTE_CLOUD_IDS[@]}"; do
      printf "  %-22s %s\n" "claude-${_CLOUD_ALIAS[$_cid]}" "${_CLOUD_DESC[$_cid]}"
    done
  fi
  if [ ${#_OMLX_UNKNOWN[@]} -gt 0 ]; then
    echo ""
    echo "  Unknown models (missing alias in $_ALIAS_FILE — no claude-* function created):"
    for _mid in "${_OMLX_UNKNOWN[@]}"; do
      printf "    %s\n" "$_mid"
    done
    echo "    Fix: add an entry in ~/local_code/model-gateway/runtime/model-info.json then run"
    echo "         python3 ~/local_code/model-gateway/runtime/omlx-config/fan_out_settings.py && claude-reload"
  fi
  echo ""
  echo "Utilities:"
  echo "  claude-restart        restart oMLX"
  echo "  claude-reload         reload model list"
  echo "  claude-list           show this help"
}

claude-restart() {
  echo "Restarting oMLX..."
  local label="gui/$(id -u)/com.local.claude-proxy"
  launchctl kickstart -k "$label" 2>/dev/null || {
    echo "Bootstrapping service..."
    launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.local.claude-proxy.plist 2>/dev/null || true
  }
  local elapsed=0
  while [ $elapsed -lt 20 ]; do
    curl -sf --max-time 2 "$OMLX_URL/health" >/dev/null 2>&1 && { echo "oMLX ready (${elapsed}s)"; return 0; }
    sleep 2; elapsed=$((elapsed + 2))
  done
  echo "WARNING: oMLX may not be ready — check ~/.claude/claude-proxy.log"
}

claude-reload() {
  echo "Reloading model list from oMLX..."
  # Remove old aliases
  for _mid in "${_OMLX_IDS[@]}"; do
    unfunction "claude-${_OMLX_ALIAS[$_mid]}" 2>/dev/null
  done
  # Reload
  _load_omlx_models
  # Recreate aliases
  for _mid in "${_OMLX_IDS[@]}"; do
    eval "claude-${_OMLX_ALIAS[$_mid]}(){ _claude_local $_mid \"\$@\"; }"
  done
  unset _mid
  echo "Loaded ${#_OMLX_IDS[@]} models (${#_OMLX_UNSUPPORTED_IDS[@]} unsupported, ${#_OMLX_UNKNOWN[@]} unknown)"
  claude-list
}

# ── Legacy compatibility (home-automation still uses these) ──────────

# anthropic-proxy management (for home-automation chat, NOT Claude Code)
_start_anthropic_proxy() {
  local launchd_label="gui/$(id -u)/com.local.anthropic-proxy"
  if launchctl print "$launchd_label" > /dev/null 2>&1; then
    launchctl kickstart -k "$launchd_label" >/dev/null 2>&1 || true
  else
    nohup uv run --directory ~/local_code/server/anthropic-proxy python server.py >> ~/.claude/anthropic-proxy.log 2>&1 &
  fi
  local elapsed=0
  while [ $elapsed -lt 15 ]; do
    curl -sf --max-time 2 http://localhost:9000/health > /dev/null 2>&1 && { echo "anthropic-proxy ready"; return 0; }
    sleep 1; elapsed=$((elapsed + 1))
  done
  echo "WARNING: anthropic-proxy may not be ready"
}

# ── End Claude Code Launcher ──

# ── Codex CLI Launcher ───────────────────────────────────────────────

_codex_local() {
  local model_id="$1"; shift

  if [ -z "${_OMLX_ALIAS[$model_id]}" ]; then
    echo "Error: unknown model '$model_id'"
    return 1
  fi

  _ensure_omlx || return 1
  _ensure_model_gateway || return 1

  local ctx="${_OMLX_CTX[$model_id]}"
  local max_out="${_OMLX_MAX_OUT[$model_id]}"
  ctx="${ctx:-32768}"
  max_out="${max_out:-32768}"
  local auto_compact
  auto_compact="$(_ls99_auto_compact_limit "$ctx")"
  local model_alias="${_OMLX_ALIAS[$model_id]}"
  local session_instruction
  session_instruction="$(_ls99_session_instruction_for_alias "$model_alias")"
  local session_instruction_escaped
  session_instruction_escaped="$(_ls99_toml_escape "$session_instruction")"
  local -a codex_cmd
  codex_cmd=(
    codex --dangerously-bypass-approvals-and-sandbox
    -c 'model_provider="ls99_models"'
    -c 'model_providers.ls99_models.name="LS99 model-gateway"'
    -c "model_providers.ls99_models.base_url=\"$CLOUD_GW_URL/v1\""
    -c 'model_providers.ls99_models.env_key="OPENAI_API_KEY"'
    -c 'model_providers.ls99_models.supports_websockets=false'
    -c "model_context_window=$ctx"
    -m "$model_id"
  )
  if [ "$auto_compact" -gt 0 ]; then
    codex_cmd+=(-c "model_auto_compact_token_limit=$auto_compact")
  fi
  if [ -n "$session_instruction_escaped" ]; then
    codex_cmd+=(-c "developer_instructions=\"$session_instruction_escaped\"")
  fi
  codex_cmd+=("$@")

  echo "Codex → model-gateway → oMLX (${_OMLX_ALIAS[$model_id]}, ${ctx} context, ${max_out} max output)"
  _ls99_with_ai_env \
    codex \
    model-gateway \
    "$model_id" \
    "${_OMLX_ALIAS[$model_id]}" \
    "codex-${_OMLX_ALIAS[$model_id]}" \
    env \
      CODEX_HOME="$HOME/.codex-omlx" \
      OPENAI_API_KEY="$CLOUD_GW_API_KEY" \
      OPENAI_BASE_URL="$CLOUD_GW_URL/v1" \
      "${codex_cmd[@]}"
}

_codex_cloud() {
  local cloud_id="$1"; shift

  if [ -z "${_CLOUD_ALIAS[$cloud_id]}" ]; then
    echo "Error: unknown cloud model '$cloud_id'"
    return 1
  fi

  _ensure_model_gateway || return 1

  local provider_model="${_CLOUD_PROVIDER_MODEL[$cloud_id]}"
  local gateway_model="${_CLOUD_NAME[$cloud_id]:-$provider_model}"
  local ctx="${_CLOUD_CTX[$cloud_id]}"
  local max_out="${_CLOUD_MAX_OUT[$cloud_id]}"
  ctx="${ctx:-32768}"
  max_out="${max_out:-32768}"
  local auto_compact
  auto_compact="$(_ls99_auto_compact_limit "$ctx")"
  local model_alias="${_CLOUD_ALIAS[$cloud_id]}"
  local session_instruction
  session_instruction="$(_ls99_session_instruction_for_alias "$model_alias")"
  local session_instruction_escaped
  session_instruction_escaped="$(_ls99_toml_escape "$session_instruction")"
  local -a codex_cmd
  codex_cmd=(
    codex --dangerously-bypass-approvals-and-sandbox
    -c 'model_provider="ls99_models"'
    -c 'model_providers.ls99_models.name="LS99 model-gateway"'
    -c "model_providers.ls99_models.base_url=\"$CLOUD_GW_URL/v1\""
    -c 'model_providers.ls99_models.env_key="OPENAI_API_KEY"'
    -c 'model_providers.ls99_models.supports_websockets=false'
    -c "model_context_window=$ctx"
    -m "$gateway_model"
  )
  if [ "$auto_compact" -gt 0 ]; then
    codex_cmd+=(-c "model_auto_compact_token_limit=$auto_compact")
  fi
  if [ -n "$session_instruction_escaped" ]; then
    codex_cmd+=(-c "developer_instructions=\"$session_instruction_escaped\"")
  fi
  codex_cmd+=("$@")

  echo "Codex → model-gateway (${_CLOUD_ALIAS[$cloud_id]}, ${ctx} context, ${max_out} max output)"
  _ls99_with_ai_env \
    codex \
    model-gateway \
    "$provider_model" \
    "${_CLOUD_ALIAS[$cloud_id]}" \
    "codex-${_CLOUD_ALIAS[$cloud_id]}" \
    env \
      CODEX_HOME="$HOME/.codex-omlx" \
      OPENAI_API_KEY="$CLOUD_GW_API_KEY" \
      OPENAI_BASE_URL="$CLOUD_GW_URL/v1" \
      "${codex_cmd[@]}"
}

codex-default() {
  echo "Codex → OpenAI cloud (default model from config.toml, ChatGPT subscription)"
  _ls99_with_ai_env \
    codex \
    openai \
    "default" \
    "default" \
    "codex-default" \
    env \
      -u OPENAI_BASE_URL \
      -u OPENAI_API_KEY \
      -u CODEX_HOME \
      codex --dangerously-bypass-approvals-and-sandbox "$@"
}

# Auto-generate codex-<alias>() for each model
for _mid in "${_OMLX_IDS[@]}"; do
  eval "codex-${_OMLX_ALIAS[$_mid]}(){ _codex_local $_mid \"\$@\"; }"
done
for _cid in "${_CLOUD_IDS[@]}"; do
  eval "codex-${_CLOUD_ALIAS[$_cid]}(){ _codex_cloud $_cid \"\$@\"; }"
done
unset _mid _cid

codex-list() {
  echo "Codex CLI Launcher Commands:"
  echo ""
  echo "  codex-default        OpenAI cloud (default model)"
  echo ""
  echo "  Local MLX models (via oMLX on port $OMLX_PORT):"
  for _mid in "${_OMLX_IDS[@]}"; do
    local _lbl=""
    if [ "${_OMLX_LOADED[$_mid]}" = "yes" ]; then
      _lbl=" [loaded]"
    fi
    printf "  %-22s %s%s\n" "codex-${_OMLX_ALIAS[$_mid]}" "${_OMLX_DESC[$_mid]}" "$_lbl"
  done
  if [ ${#_OMLX_UNSUPPORTED_IDS[@]} -gt 0 ]; then
    echo ""
    echo "  Unsupported local MLX models (present, no codex-* launcher):"
    for _mid in "${_OMLX_UNSUPPORTED_IDS[@]}"; do
      printf "    %-28s %s\n" "$_mid" "${_OMLX_UNSUPPORTED_DESC[$_mid]}"
      printf "      %s\n" "${_OMLX_UNSUPPORTED_NOTE[$_mid]}"
    done
  fi
  if [ ${#_REMOTE_CLOUD_IDS[@]} -gt 0 ]; then
    echo ""
    echo "  Cloud models (via model-gateway on port $CLOUD_GW_PORT):"
    for _cid in "${_REMOTE_CLOUD_IDS[@]}"; do
      printf "  %-22s %s\n" "codex-${_CLOUD_ALIAS[$_cid]}" "${_CLOUD_DESC[$_cid]}"
    done
  fi
  if [ ${#_OMLX_UNKNOWN[@]} -gt 0 ]; then
    echo ""
    echo "  Unknown models (missing alias — no codex-* function created):"
    for _mid in "${_OMLX_UNKNOWN[@]}"; do
      printf "    %s\n" "$_mid"
    done
  fi
  echo ""
  echo "Utilities:"
  echo "  codex-list            show this help"
}

# Override claude-reload to also reload codex and cloud aliases
claude-reload() {
  echo "Reloading model list..."
  for _mid in "${_OMLX_IDS[@]}"; do
    unfunction "claude-${_OMLX_ALIAS[$_mid]}" 2>/dev/null
    unfunction "codex-${_OMLX_ALIAS[$_mid]}" 2>/dev/null
  done
  for _cid in "${_CLOUD_IDS[@]}"; do
    unfunction "claude-${_CLOUD_ALIAS[$_cid]}" 2>/dev/null
    unfunction "codex-${_CLOUD_ALIAS[$_cid]}" 2>/dev/null
  done
  _load_omlx_models
  _load_cloud_models
  for _mid in "${_OMLX_IDS[@]}"; do
    eval "claude-${_OMLX_ALIAS[$_mid]}(){ _claude_local $_mid \"\$@\"; }"
    eval "codex-${_OMLX_ALIAS[$_mid]}(){ _codex_local $_mid \"\$@\"; }"
  done
  for _cid in "${_CLOUD_IDS[@]}"; do
    eval "claude-${_CLOUD_ALIAS[$_cid]}(){ _claude_cloud $_cid \"\$@\"; }"
    eval "codex-${_CLOUD_ALIAS[$_cid]}(){ _codex_cloud $_cid \"\$@\"; }"
  done
  unset _mid _cid
  echo "Loaded ${#_OMLX_IDS[@]} MLX + ${#_REMOTE_CLOUD_IDS[@]} cloud models (${#_OMLX_UNSUPPORTED_IDS[@]} unsupported, ${#_OMLX_UNKNOWN[@]} unknown)"
  claude-list
  echo ""
  codex-list
}

# ── End Codex CLI Launcher ──

# ── Pi CLI Launcher ─────────────────────────────────────────────────

# Keep Pi's provider-side prompt cache hot for expensive cloud models.
# Pi maps this to Anthropic 1h cache TTL (instead of 5m) and OpenAI 24h
# retention where supported. Preserve an explicit user override.
export PI_CACHE_RETENTION="${PI_CACHE_RETENTION:-long}"

_PI_AGENT_DIR="$HOME/.pi-omlx/agent"

# Vision fallback for text-only cloud models is handled entirely by the
# model-gateway: when a non-vision model receives an image payload, the gateway
# reroutes the request to GATEWAY_VISION_FALLBACK (default google/gemini-3.5-flash
# via OpenRouter) in server.py /v1/chat/completions. No client-side env vars needed.

_PI_MODELS_PROVIDER="ls99-models"
_PI_SHARED_REPAIR="$HOME/local_code/pi-shared/bin/pi-omlx-repair"

_pi_repair_profile() {
  if [ -x "$_PI_SHARED_REPAIR" ]; then
    PI_OMLX_AGENT_DIR="$_PI_AGENT_DIR" "$_PI_SHARED_REPAIR" >/dev/null || echo "Warning: failed to repair Pi oMLX profile" >&2
  fi
}

_pi_write_models() {
  mkdir -p "$_PI_AGENT_DIR"
  _pi_repair_profile
  local _pi_omlx_status
  _pi_omlx_status=$(curl -sf --max-time 3 -H "Authorization: Bearer $OMLX_API_KEY" "$OMLX_URL/v1/models/status" 2>/dev/null)
  ALIAS_FILE="$_ALIAS_FILE" \
  MODEL_INFO_FILE="$_MODEL_INFO_FILE" \
  OMLX_STATUS="$_pi_omlx_status" \
  OMLX_URL="$OMLX_URL" \
  OMLX_API_KEY="$OMLX_API_KEY" \
  CLOUD_GW_URL="$CLOUD_GW_URL" \
  CLOUD_GW_API_KEY="$CLOUD_GW_API_KEY" \
  PI_MODELS_PROVIDER="$_PI_MODELS_PROVIDER" \
  python3 - "$_PI_AGENT_DIR/models.json" <<'PY'
import json
import os
import sys

out_path = sys.argv[1]
alias_file = os.environ["ALIAS_FILE"]
with open(alias_file) as f:
    aliases = json.load(f)

omlx_status = {}
try:
    omlx_status = {
        m["id"]: m
        for m in json.loads(os.environ.get("OMLX_STATUS") or "{}").get("models", [])
        if "id" in m
    }
except Exception:
    omlx_status = {}

model_info_by_key = {}
model_info_file = os.environ.get("MODEL_INFO_FILE")
try:
    if model_info_file and os.path.exists(model_info_file):
        with open(model_info_file) as f:
            model_info = json.load(f)
        for entry in model_info.get("llm", []):
            provider = (entry.get("provider", "local") or "local").strip().lower()
            if provider not in {"local", "omlx", "mlx"}:
                provider_model = entry.get("provider_model_id")
                if provider_model:
                    model_info_by_key[f"cloud:{provider_model}"] = entry
                    model_info_by_key[provider_model] = entry
            else:
                omlx_id = entry.get("omlx_id") or entry.get("provider_model_id")
                if omlx_id:
                    model_info_by_key[omlx_id] = entry
            alias = entry.get("alias")
            if alias:
                model_info_by_key[alias] = entry
except Exception:
    model_info_by_key = {}

def merged_meta(key, alias_meta):
    # model-info.json is the source of truth.  The alias file is generated
    # from it and may be stale between fan-out runs, so use aliases as a
    # fallback and let model-info override capability/request-shape metadata.
    meta = dict(alias_meta)
    provider_model = alias_meta.get("provider_model_id")
    if provider_model:
        meta.update(model_info_by_key.get(provider_model, {}))
    meta.update(model_info_by_key.get(key, {}))
    return meta

def cost():
    return {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}

compat = {
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "supportsUsageInStreaming": False,
    "maxTokensField": "max_tokens",
}

THINKING_VALUES = {"optional", "always"}
ANTHROPIC_PROVIDERS = {"anthropic"}

# Keep this conservative: model-gateway is shared by other services, so Pi's
# generated config should only send model-specific thinking parameters where
# the backend path is already known to tolerate them.
def norm_provider(meta):
    return (meta.get("provider") or "local").strip().lower()

def is_qwen_family(key, meta):
    haystack = " ".join(str(meta.get(k, "")) for k in ("alias", "name", "omlx_id", "provider_model_id"))
    return "qwen" in f"{key} {haystack}".lower()

def is_local_key(key, meta):
    return not key.startswith("cloud:") and norm_provider(meta) in {"local", "omlx", "mlx"}

def reasoning_kind(key, meta, status):
    provider = norm_provider(meta)
    thinking = meta.get("thinking") or ""
    thinking_format = (meta.get("thinking_format") or "").strip().lower()
    provider_model = meta.get("provider_model_id") or ""
    chat_template_kwargs = meta.get("chat_template_kwargs") if isinstance(meta.get("chat_template_kwargs"), dict) else {}

    if meta.get("enable_thinking") is False or chat_template_kwargs.get("enable_thinking") is False:
        return ""

    if provider in ANTHROPIC_PROVIDERS:
        return "anthropic"

    if is_local_key(key, meta):
        # oMLX Qwen/GLM/Gemma-style templates accept request-level
        # chat_template_kwargs.enable_thinking. Only enable non-Qwen families
        # when model-info explicitly records the probed thinking_format.
        if thinking in THINKING_VALUES and thinking_format == "qwen-chat-template":
            return "local-qwen"
        # GLM-5.2 chat template additionally accepts a graded
        # chat_template_kwargs.reasoning_effort ("high"/"max"), so expose
        # real effort levels instead of a bare on/off toggle. Pi 0.80.3+
        # supports this natively through configurable chat-template kwargs.
        if thinking in THINKING_VALUES and thinking_format == "glm-chat-template":
            return "local-glm"
        if thinking in THINKING_VALUES and thinking_format == "deepseek-v4-dsml":
            return "local-deepseek-v4-dsml"
        # For Qwen models discovered by oMLX as thinking_default=True but not
        # marked in model-info, treat thinking as optional so Pi can still turn
        # it off for a session.
        if is_qwen_family(key, meta) and (thinking in THINKING_VALUES or status.get("thinking_default") is True):
            return "local-qwen"
        return ""

    if provider == "gguf":
        return ""

    if provider == "fireworks" and thinking in THINKING_VALUES:
        # Pi's upstream registry uses Anthropic Messages shape for Fireworks
        # reasoning models. Our gateway's /v1/messages path already translates
        # that shape and strips Anthropic-only cache controls.
        return "fireworks-messages"

    if provider in {"zhipuai", "zai", "zai_coding", "bigmodel"} and thinking in THINKING_VALUES:
        return "zai"

    if provider == "openrouter" and thinking in THINKING_VALUES:
        # model-gateway normalizes Pi/OpenAI-style thinking controls to
        # OpenRouter's unified reasoning object, including Gemini thinking.
        return "openrouter"

    if provider == "openai" and thinking in THINKING_VALUES:
        # Route GPT reasoning models through model-gateway's Responses endpoint;
        # GPT-5.x rejects some reasoning+tools shapes on Chat Completions.
        return "openai-responses"

    # Leave all other routes disabled until their request shape has been
    # explicitly probed and recorded in model-info.json.
    return ""

def api_type_for(kind, provider):
    if provider in ANTHROPIC_PROVIDERS or kind == "fireworks-messages":
        return "anthropic-messages"
    if kind == "openai-responses":
        return "openai-responses"
    return "openai-completions"

def apply_reasoning(model, kind, meta):
    if not kind:
        return
    model["reasoning"] = True
    thinking = meta.get("thinking") or ""
    level_map = {}
    if thinking == "always":
        level_map["off"] = None

    if kind == "local-qwen":
        model["compat"] = {"thinkingFormat": "qwen-chat-template"}
        # This Pi format is a boolean chat_template_kwargs.enable_thinking
        # toggle, not a graded effort/budget control. Expose high as the on
        # state and avoid implying that other levels produce different local
        # oMLX behavior.
        level_map.update({"minimal": None, "low": None, "medium": None, "xhigh": None})
    elif kind == "local-glm":
        # GLM-5.2 template supports graded reasoning_effort (high/max) via
        # chat_template_kwargs. Use Pi's configurable chat-template thinking
        # mode so the generated request carries both enable_thinking and the
        # mapped reasoning_effort without the glm-thinking extension shim.
        # Mirrors the cloud "zai" effort mapping.
        model["compat"] = {
            "thinkingFormat": "chat-template",
            "chatTemplateKwargs": {
                "enable_thinking": {"$var": "thinking.enabled"},
                "preserve_thinking": True,
                "reasoning_effort": {"$var": "thinking.effort", "omitWhenOff": True},
            },
        }
        level_map.update({"minimal": "high", "low": "high", "medium": "high", "high": "high", "xhigh": "max"})
    elif kind == "local-deepseek-v4-dsml":
        model["compat"] = {
            "thinkingFormat": "qwen-chat-template",
            "stripDsmlToolMarkup": True,
        }
        level_map.update({"minimal": None, "low": None, "medium": None, "xhigh": None})
    elif kind == "zai":
        model["compat"] = {"supportsDeveloperRole": False, "thinkingFormat": "zai"}
        # Gateway now forwards reasoning_effort (maps internal "xhigh"→"max");
        # expose max as the highest selector level. Clamp lower levels to the
        # default "high" effort Z.ai applies when enable_thinking is on.
        level_map.update({"minimal": "high", "low": "high", "medium": "high", "high": "high", "xhigh": "max"})
    elif kind == "openrouter":
        model["compat"] = {"thinkingFormat": "openrouter"}
        if str(meta.get("provider_model_id", "")).startswith("deepseek/"):
            model["compat"]["requiresReasoningContentOnAssistantMessages"] = True
            level_map.update({"minimal": None, "low": None, "medium": None, "high": "high", "xhigh": "max"})

    if level_map:
        model["thinkingLevelMap"] = level_map

local_models = []
cloud_models = []
seen_local = set()
seen_cloud = set()

for key, alias_meta in aliases.items():
    meta = merged_meta(key, alias_meta)
    alias = meta.get("alias")
    if not alias or meta.get("supported") is False:
        continue
    desc = meta.get("desc") or key
    status = omlx_status.get(key, {})
    ctx = int(
        meta.get("context")
        or status.get("max_context_window")
        or meta.get("max_context_window")
        or 32768
    )
    max_out = int(
        meta.get("max_output_tokens")
        or status.get("max_tokens")
        or meta.get("max_tokens")
        or 32768
    )
    provider = norm_provider(meta)
    if provider == "gguf":
        continue
    kind = reasoning_kind(key, meta, status)
    api_type = api_type_for(kind, provider)
    is_anthropic = provider in ANTHROPIC_PROVIDERS
    # Cloud models route through the model-gateway, which natively handles
    # vision fallback (transparently rerouting to gemini-3.1-pro) for text-only
    # models. So mark every cloud model image-capable to let Pi send images.
    # Local VL models get image input via their `vision` flag or a VL/gemma name
    # heuristic (model-info.json's vision flag is incomplete); local text-only
    # models stay text-only (no cloud reroute available).
    is_cloud = provider not in {"local", "omlx", "mlx", "gguf"}
    _hay = " ".join(
        str(meta.get(k, ""))
        for k in ("alias", "name", "omlx_id", "provider_model_id", "desc")
    ).lower()
    is_vision = bool(meta.get("vision")) or "vl" in _hay or "gemma" in _hay
    model = {
        "id": key,
        "name": desc,
        "api": api_type,
        "reasoning": False,
        "input": (["text", "image"] if (is_anthropic or is_cloud or is_vision) else ["text"]),
        "contextWindow": ctx,
        "maxTokens": max_out,
        "cost": cost(),
    }
    if key.startswith("cloud:") and api_type == "anthropic-messages":
        # Pi's Anthropic client appends /v1/messages, while OpenAI-compatible
        # clients append /chat/completions to a /v1 base URL. Keep the provider
        # base URL at /v1 for OpenAI-shaped models, but override Anthropic-shaped
        # cloud models to the gateway root to avoid /v1/v1/messages.
        model["baseUrl"] = os.environ["CLOUD_GW_URL"].rstrip("/")
    apply_reasoning(model, kind, meta)
    if key.startswith("cloud:"):
        provider_model = meta.get("provider_model_id")
        if not provider_model or provider_model in seen_cloud:
            continue
        model["id"] = provider_model
        seen_cloud.add(provider_model)
        cloud_models.append(model)
    else:
        if key in seen_local:
            continue
        seen_local.add(key)
        local_models.append(model)

data = {
    "providers": {
        os.environ["PI_MODELS_PROVIDER"]: {
            "baseUrl": os.environ["CLOUD_GW_URL"].rstrip("/") + "/v1",
            "api": "openai-completions",
            "apiKey": os.environ["CLOUD_GW_API_KEY"],
            "compat": compat,
            "models": local_models + cloud_models,
        },
    }
}

with open(out_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
}

_pi_local() {
  local model_id="$1"; shift

  if [ -z "${_OMLX_ALIAS[$model_id]}" ]; then
    echo "Error: unknown model '$model_id'. Run pi-list for available models."
    return 1
  fi

  _ensure_omlx || return 1
  _ensure_model_gateway || return 1
  _pi_write_models || return 1

  local ctx="${_OMLX_CTX[$model_id]}"
  local max_out="${_OMLX_MAX_OUT[$model_id]}"
  ctx="${ctx:-32768}"
  max_out="${max_out:-32768}"

  echo "Pi → model-gateway → oMLX (${_OMLX_ALIAS[$model_id]}, ${ctx} context, ${max_out} max output)"
  _ls99_with_ai_env \
    pi \
    model-gateway \
    "$model_id" \
    "${_OMLX_ALIAS[$model_id]}" \
    "pi-${_OMLX_ALIAS[$model_id]}" \
    env \
      PI_CODING_AGENT_DIR="$_PI_AGENT_DIR" \
      pi --provider "$_PI_MODELS_PROVIDER" --model "$model_id" "$@"
}

_pi_cloud() {
  local cloud_id="$1"; shift

  if [ -z "${_CLOUD_ALIAS[$cloud_id]}" ]; then
    echo "Error: unknown cloud model '$cloud_id'. Run pi-list for available models."
    return 1
  fi

  _ensure_model_gateway || return 1
  _pi_write_models || return 1

  local provider_model="${_CLOUD_PROVIDER_MODEL[$cloud_id]}"
  local ctx="${_CLOUD_CTX[$cloud_id]}"
  local max_out="${_CLOUD_MAX_OUT[$cloud_id]}"
  ctx="${ctx:-32768}"
  max_out="${max_out:-32768}"

  echo "Pi → model-gateway (${_CLOUD_ALIAS[$cloud_id]}, ${ctx} context, ${max_out} max output)"
  _ls99_with_ai_env \
    pi \
    model-gateway \
    "$provider_model" \
    "${_CLOUD_ALIAS[$cloud_id]}" \
    "pi-${_CLOUD_ALIAS[$cloud_id]}" \
    env \
      PI_CODING_AGENT_DIR="$_PI_AGENT_DIR" \
      pi --provider "$_PI_MODELS_PROVIDER" --model "$provider_model" "$@"
}

pi-default() {
  echo "Pi → default provider/model from Pi settings"
  _ls99_with_ai_env \
    pi \
    default \
    "default" \
    "default" \
    "pi-default" \
    env \
      -u PI_CODING_AGENT_DIR \
      pi "$@"
}

pi-openai() {
  echo "Pi → OpenAI subscription (ChatGPT Plus/Pro via /login OAuth)"
  _ls99_with_ai_env \
    pi \
    openai \
    "openai-subscription" \
    "openai" \
    "pi-openai" \
    env \
      -u PI_CODING_AGENT_DIR \
      -u OPENAI_API_KEY \
      -u OPENAI_BASE_URL \
      pi --provider openai "$@"
}

# Auto-generate pi-<alias>() for each model
for _mid in "${_OMLX_IDS[@]}"; do
  eval "pi-${_OMLX_ALIAS[$_mid]}(){ _pi_local $_mid \"\$@\"; }"
done
for _cid in "${_CLOUD_IDS[@]}"; do
  eval "pi-${_CLOUD_ALIAS[$_cid]}(){ _pi_cloud $_cid \"\$@\"; }"
done
unset _mid _cid

pi-restart() {
  # Restart the model-gateway (and oMLX/other services) via the canonical
  # server-ci interface, which is already on PATH and maps flags to launchd
  # labels (see `server-ci restart --help`). Defaults to model-gw so the
  # common case is just `pi-restart`.
  local svc="${1:-model-gw}"
  if [ "$svc" = "-h" ] || [ "$svc" = "--help" ]; then
    echo "Usage: pi-restart [service]   (default: model-gw)"
    echo ""
    echo "Wraps \`server-ci restart --<service>\`. Common services:"
    echo "  model-gw            Cloud LLM gateway (port $CLOUD_GW_PORT)  [default]"
    echo "  omlx                oMLX inference server (port $OMLX_PORT)"
    echo "  proxy               Caddy reverse proxy (port 8080)"
    echo "  ha-hub              Home automation dashboard (port 8100)"
    echo "  searxng             SearXNG meta-search (port 8888)"
    echo "  finance-dashboard   Finance land dashboard (port 8010)"
    echo "  mcp-ws              MCP websearch HTTP (port 8889)"
    echo "  directory           Directory API (port 9080)"
    echo "  content-lib         Content Library (port 8120)"
    echo "  all                 All of the above"
    echo ""
    echo "  status              Show status of all services (no restart)"
    echo ""
    echo "Full list: server-ci restart --help"
    return 0
  fi
  if [ "$svc" = "status" ]; then
    server-ci restart --status
    return $?
  fi
  if ! command -v server-ci >/dev/null 2>&1; then
    echo "Error: server-ci not found on PATH" >&2
    return 1
  fi
  server-ci restart --"$svc"
  local rc=$?
  if [ $rc -eq 0 ] && [ "$svc" != "all" ] && [ "$svc" != "status" ]; then
    # restart_launchd returns immediately after launchctl kickstart, before
    # the service is listening. Poll server-ci status until the port reports
    # UP (or ~25s elapse), so we don't print a misleading DOWN line during
    # the boot window.
    local port=""
    case "$svc" in
      model-gw) port=9111 ;;
      omlx) port=9110 ;;
      proxy) port=8080 ;;
      ha-hub) port=8100 ;;
      finance-dashboard) port=8010 ;;
      searxng) port=8888 ;;
      mcp-ws) port=8889 ;;
      directory) port=9080 ;;
      content-lib) port=8120 ;;
    esac
    if [ -n "$port" ]; then
      local elapsed=0 line=""
      while [ $elapsed -lt 25 ]; do
        line=$(server-ci restart --status 2>/dev/null | grep -E "^[[:space:]]*$port " | head -1)
        if echo "$line" | grep -qi "UP"; then
          break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
      done
      echo ""
      if [ -n "$line" ]; then
        echo "$line"
      else
        echo "  $port ($svc):       status unknown (not found in server-ci status)"
      fi
      if [ $elapsed -ge 25 ]; then
        echo "WARNING: $svc did not report UP within 25s — check ~/.claude/${svc//-/}.log"
      fi
    fi
  fi
  return $rc
}

pi-list() {
  echo "Pi CLI Launcher Commands:"
  echo ""
  echo "  pi-default           Pi default provider/model"
  echo "  pi-openai            OpenAI subscription (ChatGPT Plus/Pro via /login OAuth)"
  echo ""
  echo "  Local MLX models (via oMLX on port $OMLX_PORT):"
  for _mid in "${_OMLX_IDS[@]}"; do
    local _lbl=""
    if [ "${_OMLX_LOADED[$_mid]}" = "yes" ]; then
      _lbl=" [loaded]"
    fi
    printf "  %-22s %s%s\n" "pi-${_OMLX_ALIAS[$_mid]}" "${_OMLX_DESC[$_mid]}" "$_lbl"
  done
  if [ ${#_OMLX_UNSUPPORTED_IDS[@]} -gt 0 ]; then
    echo ""
    echo "  Unsupported local MLX models (present, no pi-* launcher):"
    for _mid in "${_OMLX_UNSUPPORTED_IDS[@]}"; do
      printf "    %-28s %s\n" "$_mid" "${_OMLX_UNSUPPORTED_DESC[$_mid]}"
      printf "      %s\n" "${_OMLX_UNSUPPORTED_NOTE[$_mid]}"
    done
  fi
  if [ ${#_REMOTE_CLOUD_IDS[@]} -gt 0 ]; then
    echo ""
    echo "  Cloud models (via model-gateway on port $CLOUD_GW_PORT):"
    for _cid in "${_REMOTE_CLOUD_IDS[@]}"; do
      printf "  %-22s %s\n" "pi-${_CLOUD_ALIAS[$_cid]}" "${_CLOUD_DESC[$_cid]}"
    done
  fi
  if [ ${#_OMLX_UNKNOWN[@]} -gt 0 ]; then
    echo ""
    echo "  Unknown models (missing alias — no pi-* function created):"
    for _mid in "${_OMLX_UNKNOWN[@]}"; do
      printf "    %s\n" "$_mid"
    done
  fi
  echo ""
  echo "Utilities:"
  echo "  pi-restart [service]  restart a service via server-ci (default: model-gw); 'pi-restart status' or '-h' for more"
  echo "  pi-list               show this help"
}

# Override claude-reload to also reload pi aliases
claude-reload() {
  echo "Reloading model list..."
  for _mid in "${_OMLX_IDS[@]}"; do
    unfunction "claude-${_OMLX_ALIAS[$_mid]}" 2>/dev/null
    unfunction "codex-${_OMLX_ALIAS[$_mid]}" 2>/dev/null
    unfunction "pi-${_OMLX_ALIAS[$_mid]}" 2>/dev/null
  done
  for _cid in "${_CLOUD_IDS[@]}"; do
    unfunction "claude-${_CLOUD_ALIAS[$_cid]}" 2>/dev/null
    unfunction "codex-${_CLOUD_ALIAS[$_cid]}" 2>/dev/null
    unfunction "pi-${_CLOUD_ALIAS[$_cid]}" 2>/dev/null
  done
  _load_omlx_models
  _load_cloud_models
  for _mid in "${_OMLX_IDS[@]}"; do
    eval "claude-${_OMLX_ALIAS[$_mid]}(){ _claude_local $_mid \"\$@\"; }"
    eval "codex-${_OMLX_ALIAS[$_mid]}(){ _codex_local $_mid \"\$@\"; }"
    eval "pi-${_OMLX_ALIAS[$_mid]}(){ _pi_local $_mid \"\$@\"; }"
  done
  for _cid in "${_CLOUD_IDS[@]}"; do
    eval "claude-${_CLOUD_ALIAS[$_cid]}(){ _claude_cloud $_cid \"\$@\"; }"
    eval "codex-${_CLOUD_ALIAS[$_cid]}(){ _codex_cloud $_cid \"\$@\"; }"
    eval "pi-${_CLOUD_ALIAS[$_cid]}(){ _pi_cloud $_cid \"\$@\"; }"
  done
  unset _mid _cid
  _pi_write_models
  echo "Loaded ${#_OMLX_IDS[@]} MLX + ${#_REMOTE_CLOUD_IDS[@]} cloud models (${#_OMLX_UNSUPPORTED_IDS[@]} unsupported, ${#_OMLX_UNKNOWN[@]} unknown)"
  claude-list
  echo ""
  codex-list
  echo ""
  pi-list
}

_pi_write_models 2>/dev/null || true

# ── End Pi CLI Launcher ──
