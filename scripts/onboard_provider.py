#!/usr/bin/env python3
"""Apply a tracked provider onboarding profile without exposing API keys."""

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

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.onboarding import OnboardingError, apply_profile, load_profile  # noqa: E402


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
        key = getpass.getpass("Provider API key (hidden; stored in a mode-0600 file): ").strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise OnboardingError("API key prompt was cancelled") from exc
    if not key:
        raise OnboardingError("API key cannot be empty")
    return key


def _keychain_secret(service: str, account: str | None) -> str:
    command = ["security", "find-generic-password", "-w", "-s", service]
    if account:
        command.extend(["-a", account])
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise OnboardingError(f"could not read API key from Keychain service {service!r}") from exc


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=_profile_path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-info", type=Path, required=True)
    parser.add_argument("--model-info-source", type=Path)
    secret = parser.add_mutually_exclusive_group()
    secret.add_argument("--api-key-env", metavar="NAME")
    secret.add_argument("--api-key-keychain-service", metavar="SERVICE")
    parser.add_argument("--keychain-account", default=os.environ.get("USER"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--launchd-label", help="restart this launchd service and verify before commit")
    parser.add_argument("--health-url", default="http://127.0.0.1:9111/health")
    parser.add_argument("--skip-upstream-check", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        profile = load_profile(args.profile)
        key = None
        if args.api_key_env:
            key = os.environ.get(args.api_key_env, "")
            if not key:
                raise OnboardingError(f"environment variable {args.api_key_env!r} is empty or unset")
        elif args.api_key_keychain_service:
            try:
                key = _keychain_secret(args.api_key_keychain_service, args.keychain_account)
            except OnboardingError:
                if not sys.stdin.isatty():
                    raise
                print(
                    "Keychain item is unavailable in this session; falling back to the secure terminal prompt.",
                    file=sys.stderr,
                )
                key = _prompt_secret()
        elif not args.dry_run:
            key = _prompt_secret()

        new_models = {str(model["name"]) for model in profile["models"]}
        retired_models = set((profile.get("retire") or {}).get("models") or [])
        reload_service = _service_reloader(
            args.launchd_label,
            args.health_url,
            config_path=args.config,
            model_info_path=args.model_info,
            expected_models=new_models,
            absent_models=retired_models,
        ) if args.launchd_label else None
        rollback_service = _service_reloader(
            args.launchd_label,
            args.health_url,
            config_path=args.config,
            model_info_path=args.model_info,
        ) if args.launchd_label else None
        result = apply_profile(
            profile,
            config_path=args.config,
            model_info_path=args.model_info,
            model_info_source_path=args.model_info_source,
            api_key=key,
            dry_run=args.dry_run,
            check_upstream=not args.skip_upstream_check,
            post_apply=reload_service,
            post_rollback=rollback_service,
        )
    except (OnboardingError, OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"onboard_provider: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
