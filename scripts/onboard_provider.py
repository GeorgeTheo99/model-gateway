#!/usr/bin/env python3
"""Generate or apply secret-free provider onboarding profiles."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.onboarding import (  # noqa: E402
    OnboardingError,
    apply_profile,
    load_profile,
    secret_path_for_provider,
)
from src.onboarding_generation import (  # noqa: E402
    build_draft,
    discover_models,
    discovery_allows_override,
    generated_safety,
    run_probe,
    validate_generation_inputs,
    write_draft,
)
from src.secret_files import read_api_key_file, resolve_api_key_file  # noqa: E402


def _profile_path(value: str) -> Path:
    direct = Path(value).expanduser()
    if direct.exists():
        return direct.resolve()
    name = value if value.endswith((".yaml", ".yml")) else f"{value}.yaml"
    bundled = _REPO_ROOT / "config" / "onboarding" / name
    if bundled.exists():
        return bundled.resolve()
    raise argparse.ArgumentTypeError(f"onboarding profile not found: {value}")


def _prompt_secret() -> str:
    if not sys.stdin.isatty():
        raise OnboardingError(
            "API key is not configured and no interactive terminal is available; "
            "rerun from a terminal or use --api-key-env NAME"
        )
    try:
        key = getpass.getpass("Provider API key (hidden; stored only when applying): ").strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise OnboardingError("API key prompt was cancelled") from exc
    if not key:
        raise OnboardingError("API key cannot be empty")
    return key


def _confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        raise OnboardingError("confirmation requires a terminal; use --non-interactive with explicit acknowledgements")
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt) as exc:
        raise OnboardingError("confirmation was cancelled") from exc


def _keychain_secret(service: str, account: str | None) -> str:
    command = ["security", "find-generic-password", "-w", "-s", service]
    if account:
        command.extend(["-a", account])
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise OnboardingError(f"could not read API key from Keychain service {service!r}") from exc


def _configured_secret(config_path: Path, provider_id: str, expected_base_url: str) -> str:
    if not config_path.exists():
        return ""
    config = yaml.safe_load(config_path.read_text()) or {}
    block = (config.get("providers") or {}).get(provider_id) or {}
    configured_base_url = str(block.get("base_url") or "").rstrip("/")
    if configured_base_url != expected_base_url.rstrip("/"):
        return ""
    if block.get("api_key"):
        return str(block["api_key"]).strip()
    if block.get("api_key_file"):
        try:
            return read_api_key_file(str(block["api_key_file"]), config_path)
        except OSError:
            return ""
    return ""


def _configured_secret_paths(config_path: Path) -> list[Path]:
    if not config_path.exists():
        return []
    config = yaml.safe_load(config_path.read_text()) or {}
    providers = config.get("providers") or {}
    if not isinstance(providers, dict):
        raise OnboardingError("config providers must be an object")
    return [
        resolve_api_key_file(block["api_key_file"], config_path)
        for block in providers.values()
        if isinstance(block, dict) and block.get("api_key_file")
    ]


def _explicit_secret(args: argparse.Namespace) -> str:
    if args.api_key_env:
        key = os.environ.get(args.api_key_env, "")
        if not key:
            raise OnboardingError(f"environment variable {args.api_key_env!r} is empty or unset")
        return key.strip()
    if args.api_key_keychain_service:
        try:
            return _keychain_secret(args.api_key_keychain_service, args.keychain_account)
        except OnboardingError:
            if args.non_interactive or not sys.stdin.isatty():
                raise
            print(
                "Keychain item is unavailable in this session; falling back to the secure terminal prompt.",
                file=sys.stderr,
            )
            return _prompt_secret()
    return ""


def _launchd_pid(target: str) -> int | None:
    result = subprocess.run(["launchctl", "print", target], check=True, capture_output=True, text=True)
    match = re.search(r"^[\t ]*pid = (\d+)$", result.stdout, re.MULTILINE)
    return int(match.group(1)) if match else None


def _client_headers(config_path: Path) -> dict[str, str]:
    config = yaml.safe_load(config_path.read_text()) or {}
    keys = os.environ.get("MODEL_GATEWAY_CLIENT_KEYS") or (config.get("auth") or {}).get("client_keys") or []
    if isinstance(keys, str):
        keys = [part.strip() for part in keys.split(",") if part.strip()]
    return {"Authorization": f"Bearer {keys[0]}"} if keys else {}


def _service_reloader(
    label: str,
    health_url: str,
    *,
    config_path: Path,
    model_info_path: Path,
    expected_models: set[str] | None = None,
    absent_models: set[str] | None = None,
):
    def reload_and_verify() -> None:
        target = f"gui/{os.getuid()}/{label}"
        previous_pid = _launchd_pid(target)
        subprocess.run(["launchctl", "kickstart", "-k", target], check=True, capture_output=True, text=True)
        deadline = time.monotonic() + 30
        last_error = "service did not become healthy"
        models_url = health_url.rsplit("/health", 1)[0] + "/v1/models"
        headers = _client_headers(config_path)
        while time.monotonic() < deadline:
            try:
                current_pid = _launchd_pid(target)
                if current_pid is None:
                    raise RuntimeError("launchd service has no running pid")
                if previous_pid is not None and current_pid == previous_pid:
                    raise RuntimeError("launchd process has not restarted yet")
                with urllib.request.urlopen(health_url, timeout=3) as response:
                    payload = json.load(response)
                if payload.get("status") != "ok" or payload.get("service") != "model-gateway":
                    raise RuntimeError(f"unexpected health response: {payload!r}")
                request = urllib.request.Request(models_url, headers=headers)
                with urllib.request.urlopen(request, timeout=3) as response:
                    model_payload = json.load(response)
                ids = {str(row.get("id")) for row in model_payload.get("data", []) if isinstance(row, dict)}
                missing = (expected_models or set()) - ids
                present = (absent_models or set()) & ids
                if missing:
                    raise RuntimeError(f"onboarded models unavailable: {', '.join(sorted(missing))}")
                if present:
                    raise RuntimeError(f"retired models still available: {', '.join(sorted(present))}")
                subprocess.run(
                    [sys.executable, str(_REPO_ROOT / "scripts" / "export_catalogs.py"),
                     "--check", "--config", str(config_path), "--model-info", str(model_info_path)],
                    check=True, capture_output=True, text=True,
                )
                return
            except Exception as exc:  # service may still be restarting
                last_error = str(exc)
            time.sleep(1)
        raise OnboardingError(f"gateway verification failed after reload: {last_error}")

    return reload_and_verify


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-info", type=Path, required=True)
    parser.add_argument("--model-info-source", type=Path)
    parser.add_argument("--launchd-label", help="restart this launchd service and verify before commit")
    parser.add_argument("--health-url", default="http://127.0.0.1:9111/health")


def _add_secret_args(parser: argparse.ArgumentParser) -> None:
    secret = parser.add_mutually_exclusive_group()
    secret.add_argument("--api-key-env", metavar="NAME")
    secret.add_argument("--api-key-keychain-service", metavar="SERVICE")
    parser.add_argument("--keychain-account", default=os.environ.get("USER"))


def _reloaders(args: argparse.Namespace, profile: dict):
    new_models = {str(model["name"]) for model in profile["models"]}
    retired_models = set((profile.get("retire") or {}).get("models") or [])
    if not args.launchd_label:
        return None, None
    common = dict(config_path=args.config, model_info_path=args.model_info)
    return (
        _service_reloader(
            args.launchd_label,
            args.health_url,
            expected_models=new_models,
            absent_models=retired_models,
            **common,
        ),
        _service_reloader(args.launchd_label, args.health_url, **common),
    )


def _interactive_profile_approvals(
    args: argparse.Namespace,
    profile: dict,
    preflight: dict,
    *,
    generated: bool,
) -> tuple[bool, set[str], set[str]]:
    removals = preflight.get("metadata_removals") or {}
    replacements = set(preflight.get("replaced_models") or [])
    retirements = set((profile.get("retire") or {}).get("models") or [])
    allow_removal = bool(args.allow_metadata_removal)
    confirmed_replacements = set(args.confirm_replace or [])
    confirmed_retirements = set(args.confirm_retire or [])
    if args.non_interactive:
        if generated and not args.yes:
            raise OnboardingError("generated-profile --non-interactive --apply requires --yes")
        return allow_removal, confirmed_replacements, confirmed_retirements
    if removals and not allow_removal:
        detail = "; ".join(f"{name}: {', '.join(fields)}" for name, fields in sorted(removals.items()))
        allow_removal = _confirm(f"Remove existing model metadata ({detail})?")
        if not allow_removal:
            raise OnboardingError("metadata removal was not approved")
    if confirmed_replacements and not replacements.issubset(confirmed_replacements):
        raise OnboardingError("--confirm-replace values do not cover the actual replacement set")
    if replacements and not confirmed_replacements:
        names = ", ".join(sorted(replacements))
        if _confirm(f"Replace exactly these existing gateway models: {names}?"):
            confirmed_replacements = replacements
        else:
            raise OnboardingError("model replacement was not approved")
    if confirmed_retirements and confirmed_retirements != retirements:
        raise OnboardingError("--confirm-retire values do not match retire.models")
    if retirements and not confirmed_retirements:
        names = ", ".join(sorted(retirements))
        if _confirm(f"Retire exactly these gateway models: {names}?"):
            confirmed_retirements = retirements
        else:
            raise OnboardingError("retirement was not approved")
    if generated and not args.yes and not _confirm("Apply this saved profile transactionally?"):
        raise OnboardingError("onboarding apply was not approved")
    return allow_removal, confirmed_replacements, confirmed_retirements


def _apply_loaded_profile(args: argparse.Namespace, profile: dict, key: str | None = None) -> dict:
    generated = bool(generated_safety(profile))
    allow_removal = bool(args.allow_metadata_removal)
    confirmed_replacements = set(args.confirm_replace or [])
    confirmed_retirements = set(args.confirm_retire or [])
    if not args.dry_run:
        preflight = apply_profile(
            profile,
            config_path=args.config,
            model_info_path=args.model_info,
            model_info_source_path=args.model_info_source,
            dry_run=True,
        )
        allow_removal, confirmed_replacements, confirmed_retirements = _interactive_profile_approvals(
            args,
            profile,
            preflight,
            generated=generated,
        )
    if args.allow_inconclusive_model_check:
        if not generated or not discovery_allows_override(profile):
            raise OnboardingError(
                "--allow-inconclusive-model-check is only valid for a generated profile with inconclusive discovery or successful text probes"
            )
    if not key:
        key = _explicit_secret(args)
    if not key and not args.dry_run:
        provider_id = str(profile["provider"]["id"])
        key = _configured_secret(args.config, provider_id, str(profile["provider"]["base_url"]))
    if not key and not args.dry_run:
        if args.non_interactive:
            raise OnboardingError("non-interactive apply requires an existing secret or --api-key-env/Keychain")
        key = _prompt_secret()
    reload_service, rollback_service = _reloaders(args, profile)
    return apply_profile(
        profile,
        config_path=args.config,
        model_info_path=args.model_info,
        model_info_source_path=args.model_info_source,
        api_key=key,
        dry_run=args.dry_run,
        check_upstream=not args.allow_inconclusive_model_check,
        allow_metadata_removal=allow_removal,
        confirmed_replacements=confirmed_replacements,
        confirmed_retirements=confirmed_retirements,
        post_apply=reload_service,
        post_rollback=rollback_service,
    )


def _apply_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a secret-free onboarding profile transactionally.",
        allow_abbrev=False,
    )
    parser.add_argument("profile", type=_profile_path)
    _add_runtime_args(parser)
    _add_secret_args(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--allow-metadata-removal", action="store_true")
    parser.add_argument("--confirm-replace", action="append", default=[], metavar="MODEL")
    parser.add_argument("--confirm-retire", action="append", default=[], metavar="MODEL")
    parser.add_argument("--allow-inconclusive-model-check", action="store_true")
    return parser


def _apply_main(argv: list[str]) -> dict:
    args = _apply_parser().parse_args(argv)
    profile = load_profile(args.profile)
    if "drafts" in args.profile.parts and not generated_safety(profile):
        raise OnboardingError("a profile under a drafts directory must retain valid generated provenance")
    return _apply_loaded_profile(args, profile)


def _generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover metadata and save a reviewable onboarding draft.",
        allow_abbrev=False,
    )
    parser.add_argument("--provider", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", action="append", required=True, dest="models")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path)
    output.add_argument("--stdout", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-discover", action="store_true")
    parser.add_argument("--require-discovery", action="store_true")
    parser.add_argument("--probe", action="append", choices=("text", "tools", "vision", "reasoning"), default=[])
    parser.add_argument("--docs", action="append", default=[], metavar="HTTPS_URL")
    parser.add_argument("--documented", action="append", default=[], metavar="FIELD=HTTPS_URL")
    parser.add_argument("--alias", action="append", default=[], metavar="[MODEL=]ALIAS")
    parser.add_argument("--context", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--thinking", choices=("always", "optional", "never"))
    parser.add_argument(
        "--thinking-level", action="append", dest="thinking_levels",
        choices=("off", "minimal", "low", "medium", "high", "xhigh", "max"),
        help="repeat to declare model-specific explicit thinking levels",
    )
    parser.add_argument("--thinking-format")
    parser.add_argument("--vision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--quirk", action="append", default=[])
    parser.add_argument("--pricing-json", metavar="JSON_OBJECT")
    parser.add_argument("--description")
    parser.add_argument("--retire-model", action="append", default=[])
    parser.add_argument("--preserve-existing-metadata", action="store_true")
    parser.add_argument("--drop-existing-metadata", action="append", default=[], metavar="FIELD")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--allow-metadata-removal", action="store_true")
    parser.add_argument("--confirm-replace", action="append", default=[], metavar="MODEL")
    parser.add_argument("--confirm-retire", action="append", default=[], metavar="MODEL")
    parser.add_argument("--allow-inconclusive-model-check", action="store_true")
    parser.set_defaults(dry_run=False)
    _add_runtime_args(parser)
    _add_secret_args(parser)
    return parser


def _parse_aliases(values: list[str], models: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" in value:
            model, alias = value.split("=", 1)
        elif len(models) == 1:
            model, alias = models[0], value
        else:
            raise OnboardingError("--alias requires MODEL=ALIAS when generating multiple models")
        if model not in models or not alias.strip():
            raise OnboardingError(f"invalid alias assignment: {value}")
        if model in result:
            raise OnboardingError(f"duplicate alias assignment for model: {model}")
        result[model] = alias.strip()
    return result


def _parse_documented(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise OnboardingError("--documented requires FIELD=HTTPS_URL")
        field, url = value.split("=", 1)
        if not field or not url or field in result:
            raise OnboardingError(f"invalid documented field citation: {value}")
        result[field] = url
    return result


def _parse_pricing(value: str | None) -> dict | None:
    if value is None:
        return None
    try:
        pricing = json.loads(value)
    except json.JSONDecodeError as exc:
        raise OnboardingError(f"--pricing-json is not valid JSON: {exc}") from exc
    if not isinstance(pricing, dict):
        raise OnboardingError("--pricing-json must contain a JSON object")
    return pricing


def _validate_docs(urls: list[str]) -> None:
    for url in urls:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise OnboardingError(
                f"documentation citation must be an HTTPS URL without credentials, query, or fragment: {url}"
            )


def _generate_main(argv: list[str]) -> dict | None:
    args = _generate_parser().parse_args(argv)
    provider_id, base_url, models = validate_generation_inputs(args.provider, args.base_url, args.models)
    args.provider = provider_id
    args.base_url = base_url
    args.models = models
    documented = _parse_documented(args.documented)
    _validate_docs([*args.docs, *documented.values()])
    aliases = _parse_aliases(args.alias, args.models)
    pricing = _parse_pricing(args.pricing_json)
    key = ""
    if not args.no_discover or args.probe or args.apply:
        key = _explicit_secret(args)
        if not key:
            key = _configured_secret(args.config, args.provider, args.base_url)

    if args.no_discover:
        discovery = {"status": "not_attempted", "source": "provider_models", "model_ids": []}
    else:
        discovery = discover_models(args.base_url, key or None)
        if discovery["status"] == "authentication_failed" and not key and not args.non_interactive:
            key = _prompt_secret()
            discovery = discover_models(args.base_url, key)
        if discovery["status"] == "verified":
            missing = sorted(set(args.models) - set(discovery.get("model_ids") or []))
            if missing:
                discovery = {**discovery, "status": "conflict", "missing_model_ids": missing}
    if args.require_discovery and discovery["status"] != "verified":
        raise OnboardingError(f"required provider discovery was not verified: {discovery['status']}")

    probes: list[dict] = []
    if args.probe:
        if not key:
            if args.non_interactive:
                raise OnboardingError("non-interactive probes require an existing secret or --api-key-env/Keychain")
            key = _prompt_secret()
        if not args.yes:
            if args.non_interactive or not _confirm("Run the requested probes? They may consume provider tokens"):
                raise OnboardingError("provider probes were not approved")
        for model_id in args.models:
            for kind in args.probe:
                probes.append(run_probe(args.base_url, model_id, kind, key))

    profile = build_draft(
        provider_id=args.provider,
        base_url=args.base_url,
        model_ids=args.models,
        model_info_path=args.model_info,
        discovery=discovery,
        probes=probes,
        docs=args.docs,
        documented_fields=documented,
        aliases=aliases,
        context=args.context,
        max_output_tokens=args.max_output_tokens,
        thinking="" if args.thinking == "never" else args.thinking,
        thinking_levels=args.thinking_levels,
        thinking_format=args.thinking_format,
        vision=args.vision,
        quirks=args.quirk,
        pricing=pricing,
        description=args.description,
        retirements=args.retire_model,
        preserve_existing_metadata=args.preserve_existing_metadata,
        drop_existing_metadata=args.drop_existing_metadata,
    )
    if args.stdout:
        if args.apply:
            raise OnboardingError("--stdout cannot be combined with --apply; applied profiles must be saved")
        print(yaml.safe_dump(profile, sort_keys=False, default_flow_style=False), end="")
        return None

    default_name = f"{profile['id']}.yaml"
    output = args.output or (_REPO_ROOT / "config" / "onboarding" / "drafts" / default_name)
    draft_path = write_draft(
        profile,
        output,
        force=args.force,
        protected_paths=[
            args.config,
            args.model_info,
            args.model_info_source,
            secret_path_for_provider(args.config, profile["provider"]),
            *_configured_secret_paths(args.config),
        ],
    )
    profile = load_profile(draft_path)
    result: dict = {
        "draft": str(draft_path),
        "status": (profile.get("provenance") or {}).get("status"),
        "discovery": discovery["status"],
        "probes": probes,
        "applied": False,
    }
    if args.apply:
        result["apply"] = _apply_loaded_profile(args, profile, key=key)
        result["applied"] = True
    return result


def main() -> int:
    try:
        argv = sys.argv[1:]
        if argv and argv[0] == "generate":
            result = _generate_main(argv[1:])
        else:
            result = _apply_main(argv)
    except (OnboardingError, OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"onboard_provider: {exc}", file=sys.stderr)
        return 1
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
