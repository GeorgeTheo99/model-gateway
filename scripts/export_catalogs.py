#!/usr/bin/env python3
"""Generate downstream model catalogs from the gateway config (single source of truth).

Renders:
  pi-local/config/pi-models/models.json — Pi /model picker (via the
  ~/.pi/agent/models.json symlink)

from the ``models:`` overlay in the gateway runtime config.yaml. Never hand-edit
the generated files; edit config.yaml and re-run (or let the gateway deploy hook
run it).

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
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # uv run from repo root always has it; bare python3 may not
    sys.exit("export_catalogs: pyyaml is required (run via `uv run` from the gateway repo)")

HOME = Path.home()
DEFAULT_CONFIG = HOME / "local_code" / "model-gateway-runtime" / "config" / "config.yaml"
DEFAULT_PI_OUT = HOME / "local_code" / "pi-local" / "config" / "pi-models" / "models.json"

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
    parser.add_argument("--pi-out", type=Path, default=DEFAULT_PI_OUT)
    parser.add_argument("--check", action="store_true", help="drift check only; exit 1 when stale")
    args = parser.parse_args()

    if not args.config.exists():
        sys.exit(f"export_catalogs: config not found: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f) or {}

    exported = _exported_models(config)
    if not exported:
        sys.exit("export_catalogs: refusing to render an empty catalog (no exportable models in config)")

    pi_models = _dump(render_pi_models(config))

    if args.check:
        ok = _check_one(args.pi_out, pi_models, "pi models.json")
        if not ok:
            print("export_catalogs: DRIFT — regenerate with scripts/export_catalogs.py", file=sys.stderr)
            return 1
        print(f"export_catalogs: in sync ({len(exported)} models)")
        return 0

    for path, content, label in (
        (args.pi_out, pi_models, "pi models.json"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content)
        tmp.replace(path)
        print(f"export_catalogs: wrote {label} → {path}")
    print(f"export_catalogs: {len(exported)} models exported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
