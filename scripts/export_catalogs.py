#!/usr/bin/env python3
"""Generate downstream model catalogs from the gateway config (single source of truth).

Renders:
  1. pi-local/config/pi-models/models.json — Pi /model picker (via the
     ~/.pi/agent/models.json symlink)
  2. model-gateway-runtime/pi-launchers.zsh — pi-<alias>() shell functions
     (pi-fable, pi-glm52, …) + pi-list, sourced from ~/.zshrc

from the ``models:`` overlay in the gateway runtime config.yaml. Never hand-edit
the generated files; edit config.yaml and re-run (or let the gateway deploy hook
run it).

Exports are OPT-IN via config.yaml — machines without Pi (or that don't want
shell launchers) simply omit the section and this script is a no-op::

    exports:
      pi_models: ~/local_code/pi-local/config/pi-models/models.json
      pi_launchers: ~/local_code/model-gateway-runtime/pi-launchers.zsh

CLI flags (--pi-out / --launchers-out) override config for ad-hoc runs.

Per-model export controls in config.yaml (all optional)::

    models:
    - name: claude-opus-4.8
      alias: opus48
      ...
      export: true            # false = gateway-only model, skip both catalogs
      pi:
        name: pi-opus48       # display name override in Pi
        reasoning: false      # Pi thinking-param toggle (default: true)
        id: some-alternate-id # Pi model id override (default: provider_model_id)

Modes:
    export_catalogs.py                 # write both catalogs
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

HOME = Path.home()
# Same resolution as the gateway itself: MODEL_GATEWAY_CONFIG env, else the
# checkout-local config (repo/config/config.yaml, symlinked to shared config
# in deployed layouts).
DEFAULT_CONFIG = Path(
    os.environ.get("MODEL_GATEWAY_CONFIG")
    or Path(__file__).resolve().parents[1] / "config" / "config.yaml"
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


def _exported_models(config: dict) -> list[dict]:
    models = config.get("models") or []
    out = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if entry.get("export") is False:
            continue
        if not entry.get("alias"):
            continue  # launcher functions are alias-derived; no alias = gateway-only
        out.append(entry)
    return out


def render_pi_models(config: dict) -> dict:
    """Pi models.json (providers → models) targeting the gateway."""
    providers: dict = {}
    for entry in _exported_models(config):
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


def render_pi_launchers(config: dict) -> str:
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
    entries = []
    for entry in _exported_models(config):
        alias = str(entry["alias"])
        pi = entry.get("pi") or {}
        model_id = pi.get("id") or entry.get("provider_model_id", entry["name"])
        pi_provider = _pi_provider_for(entry)
        entries.append((alias, pi_provider, model_id, str(entry.get("name", ""))))
        lines.append(
            f"pi-{alias}() {{ _pi_gw_launch {pi_provider!r} {model_id!r} {alias!r} \"$@\"; }}"
        )
    lines += [
        "",
        "pi-list() {",
        '  echo "Pi quick-start commands (via model-gateway :9111):"',
    ]
    width = max(len(a) for a, *_ in entries) + 3
    for alias, _prov, model_id, name in entries:
        lines.append(f'  printf "  %-{width}s %s\\n" "pi-{alias}" {name + " (" + model_id + ")"!r}')
    lines += [
        '  echo ""',
        '  echo "  mg-workspace list|repair|test    gateway workspace management"',
        "}",
        "",
    ]
    return "\n".join(lines)


def _dump(data: dict) -> str:
    return json.dumps(data, indent=2) + "\n"


def _check_one(path: Path, rendered: str, label: str) -> bool:
    current = path.read_text() if path.exists() else ""
    if current == rendered:
        return True
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        rendered.splitlines(keepends=True),
        fromfile=f"{label} (on disk)",
        tofile=f"{label} (rendered from config.yaml)",
    )
    sys.stderr.write("".join(list(diff)[:60]))
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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

    exported = _exported_models(config)
    if not exported:
        sys.exit("export_catalogs: refusing to render an empty catalog (no exportable models in config)")

    renders: list[tuple[Path, str, str]] = []
    if targets["pi_models"]:
        renders.append((targets["pi_models"], _dump(render_pi_models(config)), "pi models.json"))
    if targets["pi_launchers"]:
        renders.append((targets["pi_launchers"], render_pi_launchers(config), "pi-launchers.zsh"))

    if args.check:
        ok = all(_check_one(path, content, label) for path, content, label in renders)
        if not ok:
            print("export_catalogs: DRIFT — regenerate with scripts/export_catalogs.py", file=sys.stderr)
            return 1
        print(f"export_catalogs: in sync ({len(exported)} models, {len(renders)} exports)")
        return 0

    for path, content, label in renders:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content)
        tmp.replace(path)
        print(f"export_catalogs: wrote {label} → {path}")
    print(f"export_catalogs: {len(exported)} models exported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
