#!/usr/bin/env python3
"""Fan out model metadata from model-info.json into derived configs and restart oMLX.

Reads the authoritative model-info.json and:
  1. Syncs oMLX model_settings.json — context windows, max output tokens,
     optional thinking budgets, sampling defaults, and chat-template controls
     (preserves oMLX-managed fields like is_default, is_pinned, etc.)
  2. Regenerates model-aliases.json — short aliases and descriptions for ~/.zshrc launcher
  3. Restarts oMLX so it rescans model directories and picks up changes

Usage:
    python3 fan_out_settings.py                  # auto-detect paths
    python3 fan_out_settings.py --dry-run        # preview without writing
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_INFO_DEFAULT = REPO_ROOT / "model-info.json"
OMLX_SETTINGS_DEFAULT = Path.home() / ".omlx" / "model_settings.json"
MODEL_ALIASES_DEFAULT = Path.home() / ".claude" / "model-aliases.json"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def sync_omlx_settings(llm_entries: list, omlx_path: Path, dry_run: bool) -> None:
    """Sync model-info limits into oMLX model_settings.json."""
    if omlx_path.exists():
        omlx_settings = load_json(omlx_path)
    else:
        omlx_settings = {"version": 1, "models": {}}

    models = omlx_settings.setdefault("models", {})
    updated = []

    for entry in llm_entries:
        provider = entry.get("provider", "local")
        if provider != "local" and provider:
            continue
        omlx_id = entry.get("omlx_id")
        if not omlx_id:
            continue

        ctx = entry.get("context")
        max_out = entry.get("max_output_tokens")
        thinking_budget = entry.get("thinking_budget_tokens")
        optional_settings = {
            "temperature": entry.get("temperature"),
            "top_p": entry.get("top_p"),
            "top_k": entry.get("top_k"),
            "repetition_penalty": entry.get("repetition_penalty"),
            "min_p": entry.get("min_p"),
            "presence_penalty": entry.get("presence_penalty"),
            "force_sampling": entry.get("force_sampling"),
            "enable_thinking": entry.get("enable_thinking"),
            "chat_template_kwargs": entry.get("chat_template_kwargs"),
            "forced_ct_kwargs": entry.get("forced_ct_kwargs"),
        }
        if (
            ctx is None
            and max_out is None
            and thinking_budget is None
            and all(v is None for v in optional_settings.values())
        ):
            continue

        model_cfg = models.setdefault(omlx_id, {})
        changed = False
        if ctx is not None and ctx != model_cfg.get("max_context_window"):
            model_cfg["max_context_window"] = ctx
            changed = True
        if max_out is not None and max_out != model_cfg.get("max_tokens"):
            model_cfg["max_tokens"] = max_out
            changed = True
        if thinking_budget is not None:
            if model_cfg.get("thinking_budget_enabled") is not True:
                model_cfg["thinking_budget_enabled"] = True
                changed = True
            if thinking_budget != model_cfg.get("thinking_budget_tokens"):
                model_cfg["thinking_budget_tokens"] = thinking_budget
                changed = True
        elif "thinking_budget_tokens" in model_cfg or model_cfg.get("thinking_budget_enabled"):
            model_cfg["thinking_budget_enabled"] = False
            model_cfg.pop("thinking_budget_tokens", None)
            changed = True
        for key, value in optional_settings.items():
            if value is not None and value != model_cfg.get(key):
                model_cfg[key] = value
                changed = True
        if changed:
            updated.append(omlx_id)

    known_ids = {
        e["omlx_id"]
        for e in llm_entries
        if e.get("omlx_id") and not (e.get("provider", "local") != "local" and e.get("provider"))
    }
    stale = [mid for mid in models if mid not in known_ids]
    for mid in stale:
        del models[mid]

    print(f"\n[oMLX settings] {omlx_path}")
    if stale:
        print(f"  Removed stale: {', '.join(sorted(stale))}")
    if not updated and not stale:
        print("  No changes.")
    else:
        for mid in updated:
            m = models[mid]
            thinking = ""
            if m.get("thinking_budget_enabled") and m.get("thinking_budget_tokens"):
                thinking = f"  thinking_budget={m.get('thinking_budget_tokens')}"
            print(
                f"  {mid}: ctx={m.get('max_context_window')}  "
                f"max_tokens={m.get('max_tokens')}{thinking}"
            )
        if not dry_run:
            save_json(omlx_path, omlx_settings)
            print(f"  Wrote {len(models)} entries ({len(updated)} updated, {len(stale)} removed).")


def build_aliases(llm_entries: list) -> dict:
    """Build alias map from model-info entries, enforcing uniqueness.

    Local models are keyed by omlx_id.
    Cloud models (provider != "local") are keyed by "cloud:<provider_model_id>".
    """
    aliases: dict = {}
    seen: dict = {}
    collisions: list = []
    for entry in llm_entries:
        alias = entry.get("alias")
        supported = entry.get("supported", True)

        # Determine the key: omlx_id for local, "cloud:<provider_model_id>" for cloud
        provider = entry.get("provider", "local")
        if provider != "local" and provider:
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

        alias_entry = {
            "desc": entry.get("desc", ""),
            "name": entry.get("name", ""),
        }
        if alias:
            alias_entry["alias"] = alias

        # Carry non-secret capability metadata through to generated launcher
        # configs.  This lets Pi/Codex/Claude launchers make UI/request-shape
        # decisions without changing model-gateway itself.
        for field in (
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
        ):
            if entry.get(field) is not None:
                alias_entry[field] = entry.get(field)

        if supported is False:
            alias_entry["supported"] = False
            if entry.get("support_note"):
                alias_entry["support_note"] = entry.get("support_note")
        if provider != "local" and provider:
            alias_entry["provider"] = provider
            alias_entry["provider_model_id"] = entry.get("provider_model_id", "")
            alias_entry["context"] = entry.get("context", 32768)
            alias_entry["max_output_tokens"] = entry.get("max_output_tokens", 32768)
        aliases[key] = alias_entry

    if collisions:
        print("\nERROR: duplicate aliases in model-info.json:", file=sys.stderr)
        for alias, first, second in collisions:
            print(f"  '{alias}' used by both '{first}' and '{second}'", file=sys.stderr)
        sys.exit(2)

    return aliases


def diff_aliases(label: str, path: Path, aliases: dict, dry_run: bool) -> None:
    """Write aliases to path, printing a summary of changes."""
    existing = load_json(path) if path.exists() else {}

    print(f"\n[{label}] {path}")
    if aliases == existing:
        print(f"  No changes ({len(aliases)} aliases).")
        return

    added = set(aliases) - set(existing)
    removed = set(existing) - set(aliases)
    changed = {k for k in set(aliases) & set(existing) if aliases[k] != existing[k]}
    if added:
        print(f"  Added:   {', '.join(sorted(added))}")
    if removed:
        print(f"  Removed: {', '.join(sorted(removed))}")
    if changed:
        print(f"  Changed: {', '.join(sorted(changed))}")
    if not dry_run:
        save_json(path, aliases)
        print(f"  Wrote {len(aliases)} aliases.")


def restart_omlx(dry_run: bool) -> None:
    """Restart oMLX so it picks up settings and rescans model directories."""
    if dry_run:
        print("\n[dry-run] Would restart oMLX.")
        return
    print("\nRestarting oMLX...")
    uid = subprocess.check_output(["id", "-u"]).decode().strip()
    subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{uid}/com.local.claude-proxy"],
        capture_output=True,
    )
    print("  oMLX restarted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-info", type=Path, default=MODEL_INFO_DEFAULT,
                        help=f"Path to model-info.json (default: {MODEL_INFO_DEFAULT})")
    parser.add_argument("--omlx-settings", type=Path, default=OMLX_SETTINGS_DEFAULT,
                        help=f"Path to oMLX model_settings.json (default: {OMLX_SETTINGS_DEFAULT})")
    parser.add_argument("--model-aliases", type=Path, default=MODEL_ALIASES_DEFAULT,
                        help=f"Path to model-aliases.json (default: {MODEL_ALIASES_DEFAULT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing")
    args = parser.parse_args()

    if not args.model_info.exists():
        print(f"Error: {args.model_info} not found", file=sys.stderr)
        sys.exit(1)

    model_info = load_json(args.model_info)
    llm_entries = model_info.get("llm", [])
    print(f"Source: {args.model_info} ({len(llm_entries)} LLM entries)")

    if args.dry_run:
        print("[dry-run mode]")

    sync_omlx_settings(llm_entries, args.omlx_settings, args.dry_run)

    aliases = build_aliases(llm_entries)
    diff_aliases("model-aliases", args.model_aliases, aliases, args.dry_run)

    restart_omlx(args.dry_run)

    if not args.dry_run:
        print("Reload claude-launcher aliases:    claude-reload")


if __name__ == "__main__":
    main()
