#!/usr/bin/env python3
"""Generate downstream model catalogs from the gateway catalog (single source of truth).

Renders:
  1. model-aliases.json — ~/.claude/model-aliases.json (claude-*, codex-*, pi-*
     launchers and the legacy pi-launcher.zsh read this).
  2. pi-local/config/pi-models/models.json — Pi /model picker (via the
     ~/.pi/agent/models.json symlink)
  3. model-gateway-runtime/pi-launchers.zsh — pi-<alias>() shell functions
     (pi-fable, pi-glm52, …) + pi-list, sourced from ~/.zshrc

All three are rendered from the SAME merge the gateway router uses:
``model-info.json`` (committed catalog) with the ``config.yaml`` ``models:``
overlay applied on top (overlay wins on id clash). The merge lives in
``src.catalog.load_catalog_entries`` and is shared with ``src.providers``, so
the generator and the router can never drift.

Exports are OPT-IN via config.yaml — machines without Pi (or that don't want
shell launchers) simply omit the section and this script is a no-op::

    exports:
      model_aliases: ~/.claude/model-aliases.json
      pi_models: ~/local_code/pi-local/config/pi-models/models.json
      pi_launchers: ~/local_code/model-gateway-runtime/pi-launchers.zsh

CLI flags (--aliases-out / --pi-out / --launchers-out) override config for
ad-hoc runs.

Per-model export controls in config.yaml (all optional)::

    models:
    - name: claude-opus-4.8
      alias: opus48
      ...
      export: true            # false = gateway-only model, skip all catalogs
      pi:
        name: pi-opus48       # display name override in Pi
        reasoning: false      # Pi thinking-param toggle (default: true)
        id: some-alternate-id # Pi model id override (default: provider_model_id)

Modes:
    export_catalogs.py                 # write all configured catalogs
    export_catalogs.py --check         # drift check: exit 1 if outputs are stale
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # uv run from repo root always has it; bare python3 may not
    sys.exit("export_catalogs: pyyaml is required (run via `uv run` from the gateway repo)")

# Ensure the repo root (parent of scripts/) is importable so `src.catalog` can
# be loaded when this script is run directly as a subprocess (Python puts the
# script's own directory on sys.path, not the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src import catalog as catalog_mod
except ImportError:  # bare python3 outside the repo / without the venv
    catalog_mod = None  # type: ignore[assignment]

HOME = Path.home()
# Same resolution as the gateway itself: MODEL_GATEWAY_CONFIG env, else the
# checkout-local config (repo/config/config.yaml, symlinked to shared config
# in deployed layouts).
DEFAULT_CONFIG = Path(
    os.environ.get("MODEL_GATEWAY_CONFIG")
    or Path(__file__).resolve().parents[1] / "config" / "config.yaml"
)
# Same model-info resolution as src.providers: env override, else the
# checkout-local catalog. This is the committed source of truth.
DEFAULT_MODEL_INFO = Path(
    os.environ.get("MODEL_GATEWAY_MODEL_INFO")
    or Path(__file__).resolve().parents[1] / "model-info.json"
)


def _export_targets(config: dict, args) -> dict[str, Path | None]:
    """Resolve export destinations: CLI flag > config.yaml `exports:` > disabled.

    Machines without Pi simply have no `exports:` section — nothing is written
    and the gateway keeps serving normally.
    """
    exports = config.get("exports") or {}
    if not isinstance(exports, dict):
        exports = {}

    def _target(flag_value: Path | None, key: str) -> Path | None:
        if flag_value is not None:
            return flag_value
        raw = exports.get(key)
        return Path(str(raw)).expanduser() if raw else None

    return {
        "model_aliases": _target(args.aliases_out, "model_aliases"),
        "pi_models": _target(args.pi_out, "pi_models"),
        "pi_launchers": _target(args.launchers_out, "pi_launchers"),
    }

# Pi provider templates. Keys are Pi provider names; the gateway is always the
# endpoint. Kept in one place so a schema change lands everywhere at once.
PI_PROVIDER_TEMPLATES = {
    "databricks": {
        "baseUrl": "http://localhost:9111/v1",
        "apiKey": "cloud",
        "api": "openai-completions",
        "compat": {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": True,
            "maxTokensField": "max_tokens",
        },
    },
    "databricks-anthropic": {
        "baseUrl": "http://localhost:9111",
        "apiKey": "cloud",
        "api": "anthropic-messages",
        "compat": {
            "supportsEagerToolInputStreaming": False,
        },
    },
    "google": {
        "baseUrl": "http://localhost:9111/v1",
        "apiKey": "cloud",
        "api": "openai-completions",
        "compat": {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "maxTokensField": "max_tokens",
        },
    },
}


def _pi_provider_for(entry: dict) -> str:
    # Anthropic-protocol models (and any claude-* model behind an invocations
    # endpoint) talk anthropic-messages to the gateway from Pi.
    if (entry.get("protocol") or "") == "anthropic" or str(entry.get("name", "")).startswith("claude"):
        return "databricks-anthropic"
    if str(entry.get("name", "")).startswith("gemini"):
        return "google"
    return "databricks"


def _display_name(entry: dict) -> str:
    pi = entry.get("pi") or {}
    if pi.get("name"):
        return str(pi["name"])
    name = str(entry.get("name", ""))
    pretty = name.replace("-", " ").title().replace("Gpt", "GPT").replace("Glm", "GLM")
    return f"{pretty} via Databricks"


def _merged_entries(model_info_path: Path, config: dict) -> list[dict]:
    """Return the merged catalog (model-info.json + config.yaml models: overlay).

    Uses the same merge as the gateway router (``src.catalog``). Falls back to
    reading the overlay alone if ``src.catalog`` is not importable (rare: bare
    python3 outside the venv).
    """
    overlay = config.get("models") or []
    if not isinstance(overlay, list):
        overlay = []
    if catalog_mod is not None:
        return catalog_mod.load_catalog_entries(model_info_path, overlay=overlay)
    # Fallback: overlay only (no merge with the committed catalog).
    return [
        {**e, "provider": e.get("provider", "local")} for e in overlay if isinstance(e, dict)
    ]


def _exported_models(entries: list[dict]) -> list[dict]:
    """Filter merged entries to those exported to downstream catalogs.

    - ``export: false``  → gateway-only model, skip all catalogs.
    - no ``alias``        → launcher functions are alias-derived; skip.
    """
    out = []
    for entry in entries:
        if entry.get("export") is False:
            continue
        if not entry.get("alias"):
            continue
        out.append(entry)
    return out


def render_pi_models(entries: list[dict]) -> dict:
    """Pi models.json (providers → models) targeting the gateway."""
    providers: dict = {}
    for entry in _exported_models(entries):
        pi_provider = _pi_provider_for(entry)
        if pi_provider not in providers:
            providers[pi_provider] = dict(PI_PROVIDER_TEMPLATES[pi_provider], models=[])
        pi = entry.get("pi") or {}
        model: dict = {
            "id": pi.get("id") or entry.get("provider_model_id", entry["name"]),
            "name": _display_name(entry),
            "reasoning": bool(pi.get("reasoning", True)),
        }
        if entry.get("vision"):
            model["input"] = ["text", "image"]
        model["contextWindow"] = entry.get("context", 32768)
        model["maxTokens"] = entry.get("max_output_tokens", 32768)
        providers[pi_provider]["models"].append(model)
    return {"providers": providers}


def render_pi_launchers(entries: list[dict]) -> str:
    """zsh snippet defining pi-<alias>() quick-start functions + pi-list."""
    lines = [
        "# Generated by model-gateway scripts/export_catalogs.py — do not hand-edit.",
        "# Source from ~/.zshrc. Regenerated on gateway start / admin reload.",
        "",
        "_pi_gw_launch() {",
        "  local pi_provider=\"$1\" model=\"$2\" alias_name=\"$3\"; shift 3",
        "  if ! curl -sf --max-time 2 http://localhost:9111/health >/dev/null 2>&1; then",
        "    echo \"WARNING: model-gateway not healthy on :9111 — try: launchctl kickstart -k gui/$(id -u)/com.local.model-gateway (or mg-workspace repair)\"",
        "  fi",
        "  echo \"Pi → model-gateway (${alias_name} → ${model})\"",
        "  pi --provider \"$pi_provider\" --model \"$model\" \"$@\"",
        "}",
        "",
    ]
    entries_out = []
    for entry in _exported_models(entries):
        alias = str(entry["alias"])
        pi = entry.get("pi") or {}
        model_id = pi.get("id") or entry.get("provider_model_id", entry["name"])
        pi_provider = _pi_provider_for(entry)
        entries_out.append((alias, pi_provider, model_id, str(entry.get("name", ""))))
        lines.append(
            f"pi-{alias}() {{ _pi_gw_launch {pi_provider!r} {model_id!r} {alias!r} \"$@\"; }}"
        )
    lines += [
        "",
        "pi-list() {",
        '  echo "Pi quick-start commands (via model-gateway :9111):"',
    ]
    width = max(len(a) for a, *_ in entries_out) + 3
    for alias, _prov, model_id, name in entries_out:
        lines.append(f'  printf "  %-{width}s %s\\n" "pi-{alias}" {name + " (" + model_id + ")"!r}')
    lines += [
        '  echo ""',
        '  echo "  mg-workspace list|repair|test    gateway workspace management"',
        '  echo "  pi-restart [service]           restart gateway/oMLX/services (default: model-gw)"',
        "}",
        "",
        "pi-restart() {",
        "  # Restart model-gateway (and other services) via the canonical server-ci",
        "  # interface, which maps flags to launchd labels. Defaults to model-gw.",
        '  local svc="${1:-model-gw}"',
        '  if [ "$svc" = "-h" ] || [ "$svc" = "--help" ]; then',
        '    echo "Usage: pi-restart [service]   (default: model-gw)"',
        '    echo ""',
        r'    echo "Wraps \`server-ci restart --<service>\`. Common services:"',
        '    echo "  model-gw  Cloud LLM gateway (port 9111)  [default]"',
        '    echo "  omlx      oMLX inference server (port 9110)"',
        '    echo "  all       All services"',
        '    echo "  status    Show status of all services (no restart)"',
        '    echo "Full list: server-ci restart --help"',
        "    return 0",
        "  fi",
        '  if [ "$svc" = "status" ]; then',
        "    server-ci restart --status",
        "    return $?",
        "  fi",
        '  if ! command -v server-ci >/dev/null 2>&1; then',
        '    echo "Error: server-ci not found on PATH" >&2',
        "    return 1",
        "  fi",
        '  server-ci restart --"$svc"',
        "  local rc=$?",
        '  if [ $rc -eq 0 ] && [ "$svc" != "all" ] && [ "$svc" != "status" ]; then',
        "    # Poll until the service port reports UP (or ~25s elapse).",
        '    local port=""',
        '    case "$svc" in',
        "      model-gw) port=9111 ;;",
        "      omlx) port=9110 ;;",
        "    esac",
        '    if [ -n "$port" ]; then',
        "      local elapsed=0 line=\"\"",
        "      while [ $elapsed -lt 25 ]; do",
        '        line=$(server-ci restart --status 2>/dev/null | grep -E "^[[:space:]]*$port " | head -1)',
        '        if echo "$line" | grep -qi "UP"; then',
        "          break",
        "        fi",
        "        sleep 2",
        "        elapsed=$((elapsed + 2))",
        "      done",
        '      echo ""',
        '      if [ -n "$line" ]; then',
        "        echo \"$line\"",
        "      else",
        '        echo "  $port ($svc): status unknown"',
        "      fi",
        '      if [ $elapsed -ge 25 ]; then',
        '        echo "WARNING: $svc did not report UP within 25s"',
        "      fi",
        "    fi",
        "  fi",
        "  return $rc",
        "}",
        "",
    ]
    return "\n".join(lines)


# Fields carried through to the alias file so Pi/Codex/Claude launchers can make
# UI/request-shape decisions without calling the gateway. Mirrors
# runtime/omlx-config/fan_out_settings.py::build_aliases — keep in sync.
_ALIAS_FIELDS = (
    "thinking",
    "thinking_format",
    "enable_thinking",
    "chat_template_kwargs",
    "vision",
    "format",
    "context",
    "max_output_tokens",
    "omlx_id",
    "provider_model_id",
)


def render_model_aliases(entries: list[dict]) -> dict:
    """Render ~/.claude/model-aliases.json from merged catalog entries.

    Local models (provider ``omlx``/``local``) are keyed by ``omlx_id``; cloud
    models by ``cloud:<provider_model_id>`` — exactly the schema the legacy
    ``runtime/pi-launcher.zsh`` and ``fan_out_settings.py`` produce, so all
    existing ``claude-*`` / ``codex-*`` / ``pi-*`` consumers keep working.

    Hard-fails (sys.exit 2) on duplicate aliases, matching fan_out_settings.py.
    """
    aliases: dict[str, dict] = {}
    seen: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    for entry in entries:
        if entry.get("export") is False:
            continue  # gateway-only model; not exposed to any downstream launcher
        alias = entry.get("alias")
        supported = entry.get("supported", True)
        provider = entry.get("provider", "omlx")
        is_cloud = provider not in ("omlx", "local") and bool(provider)

        if is_cloud:
            if not alias:
                continue
            key = f"cloud:{entry.get('provider_model_id', alias)}"
        else:
            key = entry.get("omlx_id")
            if not key:
                continue
            if not alias and supported is not False:
                continue

        if alias:
            if alias in seen:
                collisions.append((alias, seen[alias], key))
                continue
            seen[alias] = key

        alias_entry: dict = {
            "desc": entry.get("desc", ""),
            "name": entry.get("name", ""),
        }
        if alias:
            alias_entry["alias"] = alias

        for field in _ALIAS_FIELDS:
            if entry.get(field) is not None:
                alias_entry[field] = entry.get(field)

        if supported is False:
            alias_entry["supported"] = False
            if entry.get("support_note"):
                alias_entry["support_note"] = entry.get("support_note")
        if is_cloud:
            alias_entry["provider"] = provider
            alias_entry["provider_model_id"] = entry.get("provider_model_id", "")
            alias_entry["context"] = entry.get("context", 32768)
            alias_entry["max_output_tokens"] = entry.get("max_output_tokens", 32768)
        aliases[key] = alias_entry

    if collisions:
        sys.stderr.write("ERROR: duplicate aliases in catalog:\n")
        for alias, first, second in collisions:
            sys.stderr.write(f"  '{alias}' used by both '{first}' and '{second}'\n")
        sys.exit(2)

    return aliases


def _resolve_write_target(path: Path) -> Path:
    """Resolve symlinks to the final target so we write THROUGH the link.

    ``~/.claude/model-aliases.json`` is a symlink to
    ``~/srv/server/shared/local_claude/model-aliases.json``. A naive
    ``tmp.replace(path)`` would replace the symlink itself; we must write to the
    resolved target so the symlink stays intact and the shared file updates.
    """
    if path.is_symlink() or path.exists():
        try:
            return path.resolve()
        except OSError:
            return path
    return path


def _dump(data: dict) -> str:
    return json.dumps(data, indent=2) + "\n"


def _check_one(path: Path, rendered: str, label: str) -> bool:
    # Compare against the resolved symlink target so drift checks see the real
    # on-disk content, not the symlink itself.
    target = _resolve_write_target(path)
    current = target.read_text() if target.exists() else ""
    if current == rendered:
        return True
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        rendered.splitlines(keepends=True),
        fromfile=f"{label} (on disk)",
        tofile=f"{label} (rendered from catalog)",
    )
    sys.stderr.write("".join(list(diff)[:60]))
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-info", type=Path, default=DEFAULT_MODEL_INFO,
                        help="override model-info.json path (default: MODEL_GATEWAY_MODEL_INFO env or checkout catalog)")
    parser.add_argument("--aliases-out", type=Path, default=None,
                        help="override exports.model_aliases from config")
    parser.add_argument("--pi-out", type=Path, default=None,
                        help="override exports.pi_models from config")
    parser.add_argument("--launchers-out", type=Path, default=None,
                        help="override exports.pi_launchers from config")
    parser.add_argument("--check", action="store_true", help="drift check only; exit 1 when stale")
    args = parser.parse_args()

    if not args.config.exists():
        sys.exit(f"export_catalogs: config not found: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f) or {}

    targets = _export_targets(config, args)
    if not any(targets.values()):
        print("export_catalogs: no exports configured (config.yaml `exports:` section absent) — nothing to do")
        return 0

    if not args.model_info.exists():
        sys.exit(f"export_catalogs: model-info.json not found: {args.model_info}")
    entries = _merged_entries(args.model_info, config)
    exported = _exported_models(entries)
    if not exported:
        sys.exit("export_catalogs: refusing to render an empty catalog (no exportable models in model-info.json + config overlay)")

    renders: list[tuple[Path, str, str]] = []
    if targets["model_aliases"]:
        renders.append((targets["model_aliases"], _dump(render_model_aliases(entries)), "model-aliases.json"))
    if targets["pi_models"]:
        renders.append((targets["pi_models"], _dump(render_pi_models(entries)), "pi models.json"))
    if targets["pi_launchers"]:
        renders.append((targets["pi_launchers"], render_pi_launchers(entries), "pi-launchers.zsh"))

    if args.check:
        ok = all(_check_one(path, content, label) for path, content, label in renders)
        if not ok:
            print("export_catalogs: DRIFT — regenerate with scripts/export_catalogs.py", file=sys.stderr)
            return 1
        print(f"export_catalogs: in sync ({len(exported)} models, {len(renders)} exports)")
        return 0

    for path, content, label in renders:
        target = _resolve_write_target(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # PID-suffixed temp so concurrent runs (admin reload + manual fan_out)
        # can't clobber each other; matches src/config_io.py's pattern.
        tmp = target.with_suffix(f"{target.suffix}.tmp.{os.getpid()}")
        tmp.write_text(content)
        tmp.replace(target)
        shown = path if path == target else f"{path} → {target}"
        print(f"export_catalogs: wrote {label} → {shown}")
    print(f"export_catalogs: {len(exported)} models exported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
