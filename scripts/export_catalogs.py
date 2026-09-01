#!/usr/bin/env python3
"""Generate the downstream model catalog (alias file) from the gateway catalog.

Renders:
  1. model-aliases.json — the gateway-owned PUBLIC CONTRACT
     consumed by Pi-side renderers (pi-shared/bin/pi-catalog) and any other
     tool that needs the model catalog. The gateway owns this generic catalog;
     Pi-specific artifacts (models.json, pi-launchers.zsh) are rendered by
     pi-shared from this file, not here.

Rendered from the SAME merge the gateway router uses: the machine-local,
Git-ignored ``model-info.json`` with the ``config.yaml`` ``models:`` overlay applied on
top (overlay wins on id clash). The merge lives in
``src.catalog.load_catalog_entries`` and is shared with ``src.providers``, so
the generator and the router can never drift.

Exports are OPT-IN via config.yaml — machines that don't need an alias file
simply omit the section and this script is a no-op::

    exports:
      model_aliases: ~/srv/model-gateway/shared/model-aliases.json

The ``--aliases-out`` CLI flag overrides config for ad-hoc runs.

Per-model export controls in config.yaml (all optional)::

    models:
    - name: claude-opus-4.8
      alias: opus48
      ...
      export: true            # false = gateway-only model, skip the alias catalog
      pi:
        image_input: disabled # opt out of derived gateway-assisted vision

Modes:
    export_catalogs.py                 # write the configured alias catalog
    export_catalogs.py --check         # drift check: exit 1 if output is stale
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import plistlib
import stat
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
    from src.secret_files import read_api_key_file
except ImportError:  # bare python3 outside the repo / without the venv
    catalog_mod = None  # type: ignore[assignment]
    read_api_key_file = None  # type: ignore[assignment]

HOME = Path.home()
# Same resolution as the gateway itself: MODEL_GATEWAY_CONFIG env, else the
# checkout-local config (repo/config/config.yaml, symlinked to shared config
# in deployed layouts).
DEFAULT_CONFIG = Path(
    os.environ.get("MODEL_GATEWAY_CONFIG")
    or Path(__file__).resolve().parents[1] / "config" / "config.yaml"
)
# Same model-info resolution as src.providers: env override, else the
# checkout-local machine catalog.
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
    }


def _merged_entries(model_info_path: Path, config: dict) -> list[dict]:
    """Return the merged catalog (model-info.json + config.yaml models: overlay).

    Uses the same merge as the gateway router (``src.catalog``). Falls back to
    reading the overlay alone if ``src.catalog`` is not importable (rare: bare
    python3 outside the venv).
    """
    overlay = config.get("models", [])
    if not isinstance(overlay, list):
        raise ValueError("config models overlay must be a list")
    if catalog_mod is not None:
        return catalog_mod.load_catalog_entries(model_info_path, overlay=overlay)
    # Fallback: overlay only (no merge with the machine catalog).
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


# Fields carried through to the generic alias contract so downstream consumers
# can make UI and request-shape decisions without calling the gateway.
PI_IMAGE_INPUT_ASSISTED = "gateway-assisted"
PI_IMAGE_INPUT_DISABLED = "disabled"


_ALIAS_FIELDS = (
    "thinking",
    "thinking_levels",
    "thinking_format",
    "enable_thinking",
    "chat_template_kwargs",
    "vision",
    "format",
    "context",
    "max_output_tokens",
    "omlx_id",
    "provider_model_id",
    "pi",
)


_VISION_POLICY_KEYS = (
    "GATEWAY_VISION_FALLBACK",
    "GATEWAY_VISION_FALLBACK_LOCAL",
    "GATEWAY_VISION_FALLBACK_CLOUD",
    "GATEWAY_VISION_FALLBACK_MODE",
)


def _installed_vision_policy(environ: dict[str, str]) -> dict[str, str]:
    """Read exact policy values from a private, owner-controlled LaunchAgent."""
    plist_dir = Path(
        environ.get("MODEL_GATEWAY_PLIST_DIR")
        or Path(environ.get("HOME") or Path.home()) / "Library" / "LaunchAgents"
    )
    plist_path = plist_dir / "com.local.model-gateway.plist"
    try:
        file_stat = plist_path.lstat()
        if (
            plist_path.is_symlink()
            or not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != os.getuid()
            or file_stat.st_mode & 0o022
        ):
            return {}
        with plist_path.open("rb") as handle:
            document = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException, ValueError):
        return {}
    values = document.get("EnvironmentVariables") if isinstance(document, dict) else None
    if not isinstance(values, dict):
        return {}
    return {
        key: str(values[key])
        for key in _VISION_POLICY_KEYS
        if isinstance(values.get(key), str)
    }


def _scoped_assisted_vision_localities(environ: dict[str, str] | None = None) -> set[str]:
    """Localities whose validated process policy provides extract-then-answer."""
    env = dict(os.environ if environ is None else environ)
    installed = _installed_vision_policy(env)
    values = {
        key: str(env[key]) if key in env else installed.get(key, "")
        for key in _VISION_POLICY_KEYS
    }
    legacy = values["GATEWAY_VISION_FALLBACK"].strip()
    local = values["GATEWAY_VISION_FALLBACK_LOCAL"].strip()
    cloud = values["GATEWAY_VISION_FALLBACK_CLOUD"].strip()
    if legacy or not (local or cloud):
        return set()
    mode = values["GATEWAY_VISION_FALLBACK_MODE"].strip().lower()
    if mode and mode != "extract_then_answer":
        return set()
    return {
        locality
        for locality, fallback in (("local", local), ("cloud", cloud))
        if fallback
    }


def _route_locality(entry: dict, config: dict) -> str | None:
    """Return a direct route's stable locality, or None for mixed/empty pools."""
    pool_name = entry.get("pool")
    members = (config.get("pools") or {}).get(str(pool_name)) if pool_name else None
    if isinstance(members, str):
        members = [members]
    providers = list(members or [_effective_provider(entry, config)])
    if not providers:
        return None
    localities = set()
    for provider in providers:
        normalized = str(provider).strip().lower()
        if catalog_mod is not None:
            normalized = catalog_mod.canonical_provider(normalized)
        localities.add("local" if normalized in {"omlx", "local"} else "cloud")
    return next(iter(localities)) if len(localities) == 1 else None


def _apply_assisted_vision_policy(
    entry: dict,
    alias_entry: dict,
    config: dict,
    assisted_localities: set[str],
) -> None:
    """Derive Pi's effective image input from validated scoped gateway policy."""
    hints = dict(alias_entry.get("pi") or {})
    image_input = hints.get("image_input")
    if image_input not in {None, PI_IMAGE_INPUT_ASSISTED, PI_IMAGE_INPUT_DISABLED}:
        raise ValueError(
            f"model {entry.get('name')!r} pi.image_input must be "
            f"'{PI_IMAGE_INPUT_ASSISTED}' or '{PI_IMAGE_INPUT_DISABLED}'"
        )
    if entry.get("composite") is not None:
        if entry.get("vision") is not True:
            raise ValueError(
                f"composite model {entry.get('name')!r} must declare vision: true"
            )
        if image_input is not None:
            raise ValueError(
                f"composite model {entry.get('name')!r} cannot set pi.image_input"
            )
        return
    if entry.get("vision") is True:
        if image_input is not None:
            raise ValueError(
                f"native vision model {entry.get('name')!r} cannot set pi.image_input"
            )
        return
    if image_input == PI_IMAGE_INPUT_DISABLED:
        alias_entry["pi"] = hints
        return
    locality = _route_locality(entry, config)
    if locality in assisted_localities:
        hints["image_input"] = PI_IMAGE_INPUT_ASSISTED
        alias_entry["pi"] = hints
        return
    if image_input == PI_IMAGE_INPUT_ASSISTED:
        raise ValueError(
            f"model {entry.get('name')!r} requests assisted image input without a "
            "matching locality-scoped extract_then_answer fallback"
        )


def _effective_provider(entry: dict, config: dict) -> str:
    """The provider a model actually routes to.

    Pooled models (``pool:`` with no meaningful ``provider:``) resolve to the
    pool's first member — the same primary the router picks. Without this,
    pooled entries fall back to the merge default (``omlx``) and get dropped
    from the alias file as local models with no ``omlx_id``.
    """
    pool_name = entry.get("pool")
    if pool_name:
        members = (config.get("pools") or {}).get(str(pool_name)) or []
        if isinstance(members, str):
            members = [members]
        if members:
            return str(members[0])
    return entry.get("provider", "omlx")


def _effective_protocol(entry: dict, provider: str, config: dict) -> str:
    """Request-shape protocol for a model: entry override, else provider config."""
    if entry.get("protocol"):
        return str(entry["protocol"])
    provider_config = (config.get("providers") or {}).get(provider) or {}
    return str(provider_config.get("protocol") or "openai")


def _provider_serveable(provider: str, config: dict, config_path: Path = DEFAULT_CONFIG) -> bool:
    """Can THIS gateway serve models routed to ``provider``?

    Mirrors the router's usable-member rule (enabled + base_url + api_key,
    with the built-in omlx default). Machines export only the models they can
    actually serve, so downstream launchers never list dead models. When the
    config has no ``providers:`` section at all (ad-hoc/test renders), the
    filter is skipped — serveability cannot be determined.
    """
    providers = config.get("providers")
    if not isinstance(providers, dict) or not providers:
        return True
    provider_config = dict(providers.get(provider) or {})
    if provider_config.get("enabled") is False:
        return False
    if provider in ("omlx", "local") and not provider_config:
        return True  # omlx has built-in defaults (local oMLX on :9110)
    has_key = bool(provider_config.get("api_key"))
    if not has_key and provider_config.get("api_key_file") and read_api_key_file is not None:
        try:
            has_key = bool(read_api_key_file(provider_config["api_key_file"], config_path))
        except OSError:
            has_key = False
    return bool(provider_config.get("base_url")) and has_key


def render_model_aliases(
    entries: list[dict], config: dict | None = None, config_path: Path = DEFAULT_CONFIG,
) -> dict:
    """Render the gateway-owned model-aliases.json from merged catalog entries.

    Local models (provider ``omlx``/``local``) are keyed by ``omlx_id``; cloud
    models by ``cloud:<provider_model_id>`` — the public schema consumed by
    downstream renderers such as ``pi-shared/bin/pi-catalog`` (which owns Pi
    ``models.json`` and ``pi-launchers.zsh`` rendering).

    Hard-fails (sys.exit 2) on duplicate aliases, matching fan_out_settings.py.
    """
    config = config or {}
    aliases: dict[str, dict] = {}
    seen: dict[str, str] = {}
    assisted_localities = _scoped_assisted_vision_localities()
    collisions: list[tuple[str, str, str]] = []
    for entry in entries:
        if entry.get("export") is False:
            continue  # gateway-only model; not exposed to any downstream launcher
        alias = entry.get("alias")
        supported = entry.get("supported", True)
        provider = _effective_provider(entry, config)
        if not _provider_serveable(provider, config, config_path):
            continue  # this machine's gateway cannot route the model
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
        _apply_assisted_vision_policy(entry, alias_entry, config, assisted_localities)

        if supported is False:
            alias_entry["supported"] = False
            if entry.get("support_note"):
                alias_entry["support_note"] = entry.get("support_note")
        if is_cloud:
            alias_entry["provider"] = provider
            alias_entry["protocol"] = _effective_protocol(entry, provider, config)
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

    Deployments may expose the gateway-owned catalog through a compatibility
    symlink. A naive ``tmp.replace(path)`` would replace the symlink itself;
    writing to the resolved target preserves that link while updating the
    canonical shared file.
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
    try:
        entries = _merged_entries(args.model_info, config)
    except ValueError as exc:
        parser.error(str(exc))
    exported = _exported_models(entries)
    if not exported:
        sys.exit("export_catalogs: refusing to render an empty catalog (no exportable models in model-info.json + config overlay)")

    renders: list[tuple[Path, str, str]] = []
    alias_doc: dict | None = None
    if targets["model_aliases"]:
        alias_doc = render_model_aliases(entries, config, args.config)
        if not alias_doc:
            sys.exit("export_catalogs: refusing to render an empty alias catalog (no serveable exportable models)")
        renders.append((targets["model_aliases"], _dump(alias_doc), "model-aliases.json"))

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
    print(f"export_catalogs: {len(alias_doc) if alias_doc is not None else len(exported)} models exported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
