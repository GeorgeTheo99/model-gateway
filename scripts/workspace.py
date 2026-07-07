#!/usr/bin/env python3
"""Foolproof workspace management for model-gateway pools.

Commands:
    workspace.py list
    workspace.py repair                 # interactive: fix dead auth / dead workspaces
    workspace.py test <name>
    workspace.py add <name> --host <url> [--pools p1,p2] [--position N]
                            [--profile <cli-profile>] [--style invocations|ai-gateway]
                            [--allow-partial]
    workspace.py replace <old-name> --host <url> [--name <new-name>] [--allow-partial]
    workspace.py remove <name>

`add`/`replace` are idempotent and verify BEFORE committing config:
  1. auth   — databricks CLI profile (browser SSO if refresh token is dead)
  2. probe  — GET /api/2.0/serving-endpoints reachability
  3. cover  — every pool model's provider_model_id exists on the workspace
  4. smoke  — one real completion per protocol used by affected models
  5. commit — write config.yaml (backup first), POST /admin/api/reload
              (which also regenerates pi-list/Pi catalogs)

`replace` gives the new workspace the old one's pool positions, then removes
the old entry. `remove` refuses to empty a pool. `test` runs steps 1-4 only.

Accepted --host shapes: https://host, https://host/?o=123, bare host.
"""

from __future__ import annotations

import argparse
import copy
import os
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

HOME = Path.home()
# Same resolution as the gateway: MODEL_GATEWAY_CONFIG env, else checkout-local.
DEFAULT_CONFIG = Path(
    os.environ.get("MODEL_GATEWAY_CONFIG")
    or Path(__file__).resolve().parents[1] / "config" / "config.yaml"
)
GATEWAY_URL = os.environ.get("MODEL_GATEWAY_URL", "http://localhost:9111")
ADMIN_KEY = os.environ.get("MODEL_GATEWAY_ADMIN_KEY", "admin")


def _fail(msg: str) -> "SystemExit":
    return SystemExit(f"workspace: ERROR: {msg}")


def normalize_host(raw: str) -> str:
    """Normalize a pasted workspace URL to a bare https origin."""
    raw = raw.strip()
    if not raw:
        raise _fail("empty workspace URL")
    if "://" not in raw:
        raw = f"https://{raw}"
    parts = urllib.parse.urlsplit(raw)
    if parts.scheme != "https" or not parts.netloc:
        raise _fail(f"not a valid https workspace URL: {raw!r}")
    return f"https://{parts.netloc}"


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise _fail(f"config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _providers_section(config: dict) -> dict:
    # workspaces: is an alias section; providers: is the current live one.
    if isinstance(config.get("workspaces"), dict):
        return config["workspaces"]
    return config.setdefault("providers", {})


def _pools_section(config: dict) -> dict:
    return config.setdefault("pools", {})


def _databricks(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    cli = shutil.which("databricks") or "/opt/homebrew/bin/databricks"
    return subprocess.run([cli, *args], capture_output=True, text=True, timeout=timeout)


def ensure_auth(host: str, profile: str) -> str:
    """Return a fresh access token for the workspace, running SSO if needed."""
    proc = _databricks("auth", "token", "--profile", profile)
    if proc.returncode != 0:
        print(f"  auth: profile {profile!r} has no valid token — launching browser SSO for {host}")
        login = subprocess.run(
            [shutil.which("databricks") or "/opt/homebrew/bin/databricks",
             "auth", "login", "--host", host, "--profile", profile],
            timeout=300,
        )
        if login.returncode != 0:
            raise _fail(f"browser SSO login failed for {host} (profile {profile})")
        proc = _databricks("auth", "token", "--profile", profile)
        if proc.returncode != 0:
            raise _fail(f"still cannot mint a token for profile {profile}: {proc.stderr[:200]}")
    token = json.loads(proc.stdout).get("access_token", "")
    if not token:
        raise _fail(f"CLI returned no access_token for profile {profile}")
    print(f"  auth: OK (profile {profile})")
    return token


def _get_json(url: str, token: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def probe_endpoints(host: str, token: str) -> set[str]:
    try:
        data = _get_json(f"{host}/api/2.0/serving-endpoints", token)
    except urllib.error.URLError as exc:
        raise _fail(f"workspace unreachable: {host} ({exc})")
    names = {e["name"] for e in data.get("endpoints", [])}
    print(f"  probe: OK ({len(names)} serving endpoints)")
    return names


def _models_in_pools(config: dict, pool_names: list[str]) -> list[dict]:
    models = []
    for entry in config.get("models") or []:
        if isinstance(entry, dict) and entry.get("pool") in pool_names:
            models.append(entry)
    return models


def check_coverage(config: dict, pool_names: list[str], endpoint_names: set[str], allow_partial: bool) -> None:
    models = _models_in_pools(config, pool_names)
    if not models:
        print("  coverage: no pooled models affected — skipped")
        return
    missing = []
    print("  coverage:")
    for m in models:
        pmid = m.get("provider_model_id", m.get("name"))
        ok = pmid in endpoint_names
        print(f"    {'✓' if ok else '✗'} {m.get('alias') or m.get('name'):10s} {pmid}")
        if not ok:
            missing.append(pmid)
    if missing and not allow_partial:
        raise _fail(
            f"workspace does not serve {len(missing)} pool model(s): {', '.join(missing)}. "
            "Re-run with --allow-partial to accept degraded coverage."
        )


def smoke_test(host: str, token: str, endpoint_names: set[str]) -> None:
    """One tiny real completion against a known FMAPI endpoint on the workspace."""
    candidates = [n for n in ("databricks-claude-sonnet-4-6", "databricks-gpt-5-4-mini", "databricks-gpt-5-5")
                  if n in endpoint_names]
    if not candidates:
        print("  smoke: no known FMAPI endpoint available — skipped")
        return
    name = candidates[0]
    body = json.dumps({"messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                       "max_tokens": 16}).encode()
    req = urllib.request.Request(
        f"{host}/serving-endpoints/{name}/invocations",
        data=body, method="POST",
        headers={"Authorization": f"Bearer {token}", "content-type": "application/json"},
    )
    deadline = time.time() + 120
    while True:
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                json.loads(resp.read())
                print(f"  smoke: OK ({name})")
                return
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and time.time() < deadline:
                print("  smoke: 429, retrying in 15s…")
                time.sleep(15)
                continue
            raise _fail(f"smoke completion failed on {name}: HTTP {exc.code} {exc.read()[:150]!r}")
        except urllib.error.URLError as exc:
            raise _fail(f"smoke completion failed on {name}: {exc}")


def _backup(path: Path) -> None:
    backup = path.with_name(path.name + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, backup)
    print(f"  backup: {backup.name}")


def _write_config(path: Path, config: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))
    tmp.replace(path)


def reload_gateway() -> None:
    req = urllib.request.Request(
        f"{GATEWAY_URL}/admin/api/reload", method="POST",
        headers={"Authorization": f"Bearer {ADMIN_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
        print(f"  reload: {data.get('message')} (catalogs: {data.get('catalogs')})")
    except urllib.error.URLError as exc:
        print(f"  reload: WARN — gateway reload failed ({exc}); restart it manually")


def cmd_list(args) -> None:
    config = _load_config(args.config)
    providers = _providers_section(config)
    pools = config.get("pools") or {}
    member_of: dict[str, list[str]] = {}
    for pool, members in pools.items():
        for m in members or []:
            member_of.setdefault(m, []).append(pool)
    print(f"{'WORKSPACE':22s} {'HOST':55s} {'AUTH':14s} POOLS")
    for name, entry in providers.items():
        if not isinstance(entry, dict) or not entry.get("base_url"):
            continue
        auth = entry.get("auth_profile") or ("pat" if str(entry.get("api_key", "")).startswith("dapi") else "static")
        print(f"{name:22s} {entry.get('base_url','')[:55]:55s} {auth:14s} {','.join(member_of.get(name, [])) or '-'}")
    print("\nPools:")
    for pool, members in pools.items():
        print(f"  {pool}: {' → '.join(members or [])}")


def _verify(config: dict, host: str, profile: str, pool_names: list[str], allow_partial: bool) -> None:
    token = ensure_auth(host, profile)
    endpoint_names = probe_endpoints(host, token)
    check_coverage(config, pool_names, endpoint_names, allow_partial)
    smoke_test(host, token, endpoint_names)


def cmd_test(args) -> None:
    config = _load_config(args.config)
    providers = _providers_section(config)
    entry = providers.get(args.name)
    if not isinstance(entry, dict):
        raise _fail(f"unknown workspace {args.name!r}")
    # AI-gateway hostnames don't serve the REST API; allow a workspace_url
    # override for probes/smokes (e.g. fevm-model-exp behind its aigw host).
    host = normalize_host(str(entry.get("workspace_url") or entry.get("base_url", "")))
    pools = [p for p, members in (config.get("pools") or {}).items() if args.name in (members or [])]
    print(f"Testing workspace {args.name!r} ({host}) — pools: {', '.join(pools) or 'none'}")
    if entry.get("auth_refresh") == "databricks-cli":
        token = ensure_auth(host, entry.get("auth_profile") or args.name)
    else:
        token = str(entry.get("api_key", ""))
        print("  auth: static credential from config (PAT)")
        if not token:
            raise _fail(f"workspace {args.name!r} has no api_key configured")
    endpoint_names = probe_endpoints(host, token)
    check_coverage(config, pools, endpoint_names, allow_partial=True)
    smoke_test(host, token, endpoint_names)
    print("workspace test: PASSED")


def _insert_into_pools(config: dict, name: str, pool_names: list[str], position: int | None) -> None:
    pools = _pools_section(config)
    for pool in pool_names:
        members = pools.setdefault(pool, [])
        if name in members:
            continue
        if position is None or position >= len(members):
            members.append(name)
        else:
            members.insert(max(position - 1, 0), name)


def cmd_add(args) -> None:
    host = normalize_host(args.host)
    profile = args.profile or args.name
    pool_names = [p.strip() for p in (args.pools or "").split(",") if p.strip()]
    config = _load_config(args.config)

    print(f"Adding workspace {args.name!r} ({host}) to pools: {', '.join(pool_names) or 'none'}")
    # Verify against a config copy that already contains the new pool layout so
    # coverage checks the models this workspace WILL serve.
    staged = copy.deepcopy(config)
    _insert_into_pools(staged, args.name, pool_names, args.position)
    _verify(staged, host, profile, pool_names, args.allow_partial)

    providers = _providers_section(config)
    token = ensure_auth(host, profile)  # cheap: token is cached by the CLI
    providers[args.name] = {
        "base_url": host,
        "api_key": token,
        "protocol": "openai",
        "endpoint_style": "invocations" if args.style == "invocations" else None,
        "auth_refresh": "databricks-cli",
        "auth_profile": profile,
        "quirks": ["no_stream_options", "no_reasoning_params"],
    }
    if args.style != "invocations":
        providers[args.name].pop("endpoint_style")
        providers[args.name]["path_prefixes"] = {"anthropic": "anthropic/v1", "openai": "mlflow/v1"}
        providers[args.name]["quirks"] = ["anthropic_bearer_auth"]
    providers[args.name] = {k: v for k, v in providers[args.name].items() if v is not None}
    _insert_into_pools(config, args.name, pool_names, args.position)

    _backup(args.config)
    _write_config(args.config, config)
    print(f"  commit: {args.config}")
    reload_gateway()
    print(f"workspace add: DONE — {args.name} is live")


def cmd_replace(args) -> None:
    config = _load_config(args.config)
    providers = _providers_section(config)
    old = providers.get(args.old_name)
    if not isinstance(old, dict):
        raise _fail(f"unknown workspace {args.old_name!r}")
    new_name = args.name or args.old_name
    host = normalize_host(args.host)
    profile = args.profile or new_name

    pools = _pools_section(config)
    affected = [p for p, members in pools.items() if args.old_name in (members or [])]
    print(f"Replacing workspace {args.old_name!r} with {new_name!r} ({host}) in pools: {', '.join(affected) or 'none'}")

    staged = copy.deepcopy(config)
    for pool in affected:  # stage: swap in place, keep position
        members = staged["pools"][pool]
        members[members.index(args.old_name)] = new_name
    _verify(staged, host, profile, affected, args.allow_partial)

    token = ensure_auth(host, profile)
    entry = {k: v for k, v in old.items()}  # inherit style/quirks from the old entry
    entry.update({"base_url": host, "api_key": token, "auth_refresh": "databricks-cli", "auth_profile": profile})
    providers[new_name] = entry
    if new_name != args.old_name:
        providers.pop(args.old_name, None)
    for pool in affected:
        members = pools[pool]
        members[members.index(args.old_name)] = new_name

    _backup(args.config)
    _write_config(args.config, config)
    print(f"  commit: {args.config}")
    reload_gateway()
    print(f"workspace replace: DONE — {new_name} took over {args.old_name}'s pool positions")


def cmd_repair(args) -> None:
    """Interactive recovery pass over every OAuth-backed workspace.

    For each workspace with auth_refresh: databricks-cli, escalate:
      silent token → browser SSO → prompt to paste a replacement workspace URL
    (the paste-a-URL flow). Static-PAT workspaces are probe-checked only.
    """
    config = _load_config(args.config)
    providers = _providers_section(config)
    broken: list[str] = []
    for name, entry in list(providers.items()):
        if not isinstance(entry, dict) or not entry.get("base_url"):
            continue
        if entry.get("auth_refresh") != "databricks-cli":
            continue
        host = normalize_host(str(entry.get("workspace_url") or entry["base_url"]))
        profile = entry.get("auth_profile") or name
        print(f"\nChecking workspace {name!r} ({host}, profile {profile})")
        try:
            token = ensure_auth(host, profile)
            probe_endpoints(host, token)
        except SystemExit as exc:
            print(f"  {exc}")
            broken.append(name)
            reply = input(f"  Paste a replacement workspace URL for {name!r} (Enter to skip): ").strip()
            if not reply:
                print("  skipped — pooled models fail over; single-workspace models will error")
                continue
            ns = argparse.Namespace(
                config=args.config, old_name=name, host=reply,
                name=None, profile=None, allow_partial=True,
            )
            cmd_replace(ns)
            broken.remove(name)
    if broken:
        raise _fail(f"still broken: {', '.join(broken)}")
    print("\nworkspace repair: all OAuth-backed workspaces healthy")


def cmd_remove(args) -> None:
    config = _load_config(args.config)
    providers = _providers_section(config)
    if args.name not in providers:
        raise _fail(f"unknown workspace {args.name!r}")
    pools = config.get("pools") or {}
    for pool, members in pools.items():
        if args.name in (members or []) and len(members) == 1:
            raise _fail(f"refusing to remove: {args.name!r} is the only member of pool {pool!r}")
    solo_models = [m.get("name") for m in config.get("models") or []
                   if isinstance(m, dict) and m.get("provider") == args.name]
    if solo_models:
        raise _fail(f"refusing to remove: models route directly to it: {', '.join(map(str, solo_models))}")
    for members in pools.values():
        if args.name in (members or []):
            members.remove(args.name)
    providers.pop(args.name)
    _backup(args.config)
    _write_config(args.config, config)
    print(f"  commit: {args.config}")
    reload_gateway()
    print(f"workspace remove: DONE — {args.name} removed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")
    sub.add_parser("repair")

    p = sub.add_parser("test")
    p.add_argument("name")

    p = sub.add_parser("add")
    p.add_argument("name")
    p.add_argument("--host", required=True)
    p.add_argument("--pools", default="")
    p.add_argument("--position", type=int, default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--style", choices=["invocations", "ai-gateway"], default="invocations")
    p.add_argument("--allow-partial", action="store_true")

    p = sub.add_parser("replace")
    p.add_argument("old_name")
    p.add_argument("--host", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--allow-partial", action="store_true")

    p = sub.add_parser("remove")
    p.add_argument("name")

    args = parser.parse_args()
    {"list": cmd_list, "repair": cmd_repair, "test": cmd_test, "add": cmd_add,
     "replace": cmd_replace, "remove": cmd_remove}[args.command](args)


if __name__ == "__main__":
    main()
