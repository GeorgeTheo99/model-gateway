#!/usr/bin/env python3
"""Foolproof workspace management for model-gateway pools.

Commands:
    model-gateway workspace list
    model-gateway workspace repair      # interactive: fix dead auth / workspaces
    model-gateway workspace test <name>
    model-gateway workspace add <name> --host <url> [--pools p1,p2]
                                [--position N] [--profile <cli-profile>]
                                [--style invocations|ai-gateway|auto] [--allow-partial]
    model-gateway workspace replace <old-name> --host <url> [--name <new-name>]
                                      [--style inherit|invocations|ai-gateway|auto]
                                      [--allow-partial]
    model-gateway workspace remove <name>

`add`/`replace` are idempotent and verify BEFORE committing config. If gateway
activation fails, the previous config is restored and activated automatically:
  1. auth   — databricks CLI profile (browser SSO if refresh token is dead)
  2. probe  — GET /api/2.0/serving-endpoints reachability
  3. cover  — every pool model's provider_model_id exists on the workspace
  4. smoke  — one real completion through the EXACT route the gateway will
              use (derived AI Gateway host + path prefix, or the direct
              /serving-endpoints/<id>/invocations URL) — not a proxy for it
  5. commit — write config.yaml (backup first), then restart+verify through
              the operator CLI or use the authenticated admin reload API

`replace` gives the new workspace the old one's pool positions, then removes
the old entry. `remove` refuses to empty a pool. `test` runs steps 1-4 only.

Accepted --host shapes: https://host, https://host/?o=123, bare host. A bare
workspace ID (e.g. 7474651766001209) is rejected with instructions to paste
the browser URL instead.

--style auto probes the derived AI Gateway route with a real completion and
falls back to direct invocations when that host/path is unusable (DNS, TLS,
404, protocol); the chosen style and the reason are printed. --allow-partial
records which catalog models the workspace actually serves
(``available_model_ids``) so exported launchers omit the rest.
For --style ai-gateway, pass the WORKSPACE URL (e.g. https://e2-demo-field-eng
.cloud.databricks.com); the routed <org-id>.ai-gateway.cloud.databricks.com
base_url is derived from the workspace token's aud claim. An explicit
gateway hostname is still accepted (the workspace URL is then recovered
from the token's iss claim for probes/auth).
"""

from __future__ import annotations

import argparse
import base64
import copy
import os
import json
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
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
ADMIN_KEY = os.environ.get("MODEL_GATEWAY_ADMIN_KEY", "").strip()
RESTART_BIN = os.environ.get("MODEL_GATEWAY_RESTART_BIN", "").strip()


def _fail(msg: str) -> "SystemExit":
    return SystemExit(f"workspace: ERROR: {msg}")


WORKSPACE_URL_HELP = (
    "Expected the full workspace URL copied from your browser, e.g. "
    "https://my-workspace.cloud.databricks.com or "
    "https://adb-1234567890123456.7.azuredatabricks.net "
    "(a '?o=<id>' suffix and any path are fine and are stripped)."
)


def normalize_host(raw: str) -> str:
    """Normalize a pasted workspace URL to a bare https origin.

    Accepts a browser URL (with ``?o=`` query, path, trailing slash) or a bare
    hostname. Rejects a bare workspace/org ID (e.g. ``7474651766001209``), an
    ``adb-`` prefix without a domain, ``http://`` and anything that is not a
    dotted DNS hostname, and says exactly what to paste instead.
    """
    raw = raw.strip()
    if not raw:
        raise _fail(f"empty workspace URL. {WORKSPACE_URL_HELP}")
    if re.fullmatch(r"(o=)?\d{6,}", raw) or re.fullmatch(r"adb-\d+(\.\d+)?", raw):
        raise _fail(
            f"{raw!r} looks like a workspace ID, not a workspace URL. {WORKSPACE_URL_HELP}"
        )
    if "://" not in raw:
        raw = f"https://{raw}"
    parts = urllib.parse.urlsplit(raw)
    host = parts.hostname or ""
    if parts.scheme != "https":
        raise _fail(f"workspace URL must use https, got {parts.scheme!r}. {WORKSPACE_URL_HELP}")
    if not host or "." not in host or not re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+", host.lower()):
        raise _fail(f"not a valid workspace hostname: {host or raw!r}. {WORKSPACE_URL_HELP}")
    if parts.port:
        raise _fail(f"workspace URL must not include a port: {raw!r}. {WORKSPACE_URL_HELP}")
    return f"https://{host.lower()}"


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


def _is_ai_gateway_host(host: str) -> bool:
    """Whether host is an explicit <org-id>.ai-gateway.cloud.databricks.com origin."""
    netloc = urllib.parse.urlsplit(host).netloc
    prefix = netloc.split(".", 1)[0]
    return netloc.endswith(".ai-gateway.cloud.databricks.com") and prefix.isdigit()


def _jwt_claims(token: str) -> dict:
    """Decode the unverified payload of a Databricks CLI OAuth JWT (for routing hints only)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims if isinstance(claims, dict) else {}
    except (IndexError, ValueError):
        return {}


def derive_ai_gateway_host(token: str) -> str:
    """Derive https://<org-id>.ai-gateway.cloud.databricks.com from a workspace token.

    The workspace's org id is the numeric `aud` claim of its OAuth token
    (verified against e2-demo-field-eng and fevm-model-exp), and the AI
    Gateway hostname for a workspace is <org-id>.ai-gateway.cloud.databricks.com.
    """
    aud = _jwt_claims(token).get("aud")
    candidates = [str(a) for a in (aud if isinstance(aud, list) else [aud]) if str(a).isdigit()]
    if len(candidates) != 1:
        raise _fail(
            "cannot derive the AI Gateway host from the workspace token "
            f"(aud={aud!r}); pass the explicit <org-id>.ai-gateway.cloud.databricks.com "
            "hostname as --host instead"
        )
    return f"https://{candidates[0]}.ai-gateway.cloud.databricks.com"


def _resolve_ai_gateway_host(host: str, profile: str) -> tuple[str, str, str]:
    """Resolve (workspace_host, gateway_base_url, token) for ai-gateway style.

    Accepts either the workspace URL (auth/probes against it; the gateway
    base_url is derived from the token's aud claim) or an explicit gateway
    hostname (the workspace URL is recovered from the token's iss claim,
    since gateway hostnames don't serve the REST API).
    """
    token = ensure_auth(host, profile)
    claims = _jwt_claims(token)
    if _is_ai_gateway_host(host):
        iss = str(claims.get("iss") or "")
        workspace = iss[: -len("/oidc")] if iss.endswith("/oidc") else ""
        if not workspace:
            raise _fail(
                f"cannot recover the workspace URL from the token (iss={iss!r}); "
                f"pass the workspace URL instead of the {host} gateway hostname"
            )
        return workspace, host, token
    return host, derive_ai_gateway_host(token), token


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


def _affected_models(config: dict, pool_names: list[str], provider_names: list[str] = ()) -> list[dict]:
    """Models served by the given pools OR bound directly to the given providers."""
    models = []
    for entry in config.get("models") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("pool") in pool_names or entry.get("provider") in provider_names:
            models.append(entry)
    return models


def check_coverage(config: dict, pool_names: list[str], endpoint_names: set[str], allow_partial: bool,
                   provider_names: list[str] = ()) -> list[str] | None:
    """Print per-model coverage; return the sorted provider_model_ids this
    workspace serves. Fails unless every affected model is served or
    ``allow_partial`` is set (then the missing ones are listed as degraded)."""
    models = _affected_models(config, pool_names, provider_names)
    if not models:
        print("  coverage: no models bound to these pools/providers — skipped")
        return None
    missing, available = [], []
    print("  coverage:")
    for m in models:
        pmid = m.get("provider_model_id", m.get("name"))
        ok = pmid in endpoint_names
        print(f"    {'✓' if ok else '✗'} {m.get('alias') or m.get('name'):10s} {pmid}")
        (available if ok else missing).append(pmid)
    if missing:
        if not allow_partial:
            raise _fail(
                f"workspace does not serve {len(missing)} of {len(models)} model(s): {', '.join(missing)}. "
                "Re-run with --allow-partial to accept degraded coverage (those models will be "
                "left out of the exported catalogs)."
            )
        print(f"  coverage: DEGRADED — {len(missing)} model(s) not served here and will be omitted "
              f"from catalogs: {', '.join(missing)}")
    return sorted(set(available))


def _model_protocols(config: dict, pool_names: list[str], provider_names: list[str] = ()) -> set[str]:
    """Wire protocols the affected models use on an AI-gateway route."""
    return {str(m.get("protocol") or "openai") for m in _affected_models(config, pool_names, provider_names)}


def _stamp_coverage(entry: dict, available: list[str] | None) -> None:
    """Persist which catalog models the workspace serves (no secrets). ``None``
    means coverage was not checked (no bound models); drop stale values."""
    entry.pop("available_model_ids", None)
    entry.pop("coverage_checked_at", None)
    if available is None:
        return
    entry["available_model_ids"] = list(available)
    entry["coverage_checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


SMOKE_CANDIDATES = ("databricks-claude-sonnet-4-6", "databricks-gpt-5-4-mini", "databricks-gpt-5-5")

AI_GATEWAY_PATH_PREFIXES = {"anthropic": "anthropic/v1", "openai": "mlflow/v1"}


class RoutePreflightError(Exception):
    """A runtime data-plane probe failed; ``kind`` classifies the failure."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def _classify_url_error(exc: urllib.error.URLError) -> tuple[str, str]:
    reason = exc.reason
    if isinstance(reason, socket.gaierror) or "nodename nor servname" in str(reason) \
            or "Name or service not known" in str(reason) or "[Errno 8]" in str(reason):
        return "dns", f"hostname does not resolve ({reason})"
    if isinstance(reason, ssl.SSLError) or "SSL" in str(reason).upper():
        return "tls", str(reason)
    return "connect", str(reason)


def _smoke_request(url: str, body: dict, headers: dict, deadline: float) -> None:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    while True:
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                json.loads(resp.read())
                return
        except urllib.error.HTTPError as exc:
            snippet = exc.read()[:150]
            if exc.code == 429 and time.time() < deadline:
                print("  smoke: 429, retrying in 15s…")
                time.sleep(15)
                continue
            if exc.code in (401, 403):
                raise RoutePreflightError("auth", f"HTTP {exc.code} {snippet!r}")
            if exc.code == 404:
                raise RoutePreflightError("path", f"HTTP 404 (path/model not served here) {snippet!r}")
            if exc.code == 429:
                raise RoutePreflightError("rate_limited", "HTTP 429 persisted for 120s")
            raise RoutePreflightError("http", f"HTTP {exc.code} {snippet!r}")
        except urllib.error.URLError as exc:
            kind, detail = _classify_url_error(exc)
            raise RoutePreflightError(kind, detail)
        except socket.timeout:
            raise RoutePreflightError("connect", "timed out")


def runtime_preflight(style: str, workspace_host: str, base_url: str, token: str,
                      endpoint_names: set[str], protocols: set[str] | None = None) -> str:
    """One real completion through the EXACT route the gateway will use.

    ``style`` is ``invocations`` (POST <workspace>/serving-endpoints/<id>/invocations,
    bearer auth, OpenAI body) or ``ai-gateway`` (POST <gateway>/<prefix>/... per
    protocol with bearer auth). Returns the endpoint name used. Raises
    RoutePreflightError with a classified ``kind`` (dns/tls/connect/auth/path/
    rate_limited/http/missing_model) so callers can decide whether to fall back.
    """
    candidates = [n for n in SMOKE_CANDIDATES if n in endpoint_names]
    if not candidates:
        raise RoutePreflightError("missing_model",
                                  f"none of the bootstrap models {', '.join(SMOKE_CANDIDATES)} is served")
    name = candidates[0]
    deadline = time.time() + 120
    headers = {"Authorization": f"Bearer {token}"}
    openai_body = {"messages": [{"role": "user", "content": "Reply with exactly: OK"}], "max_tokens": 16}
    if style == "invocations":
        _smoke_request(f"{workspace_host}/serving-endpoints/{name}/invocations", openai_body, headers, deadline)
        return name
    if style != "ai-gateway":
        raise ValueError(f"unknown route style {style!r}")
    for protocol in sorted(protocols or {"openai"}):
        prefix = AI_GATEWAY_PATH_PREFIXES[protocol]
        if protocol == "anthropic":
            url = f"{base_url}/{prefix}/messages"
            body = {"model": name, **openai_body}
            _smoke_request(url, body, {**headers, "anthropic-version": "2023-06-01"}, deadline)
        else:
            url = f"{base_url}/{prefix}/chat/completions"
            _smoke_request(url, {"model": name, **openai_body}, headers, deadline)
    return name


def smoke_test(host: str, token: str, endpoint_names: set[str], *, style: str = "invocations",
               base_url: str | None = None, protocols: set[str] | None = None,
               require_bootstrap: bool = False) -> None:
    """Runtime smoke through the committed route; SystemExit on failure.

    ``require_bootstrap`` (set for partial-coverage activations) turns "no
    bootstrap model to smoke" into a failure: degraded coverage is only
    acceptable when at least one known model provably works end to end.
    """
    try:
        name = runtime_preflight(style, host, base_url or host, token, endpoint_names, protocols)
    except RoutePreflightError as exc:
        if exc.kind == "missing_model":
            if require_bootstrap:
                raise _fail(f"partial coverage refused: {exc.detail}; a bootstrap model must exist "
                            "and answer through the runtime route")
            print(f"  smoke: skipped — {exc.detail}")
            return
        raise _fail(f"runtime smoke via {style} route failed ({exc.kind}): {exc.detail}")
    print(f"  smoke: OK ({name} via {style})")


def _config_target(path: Path) -> Path:
    """Return the real config target so atomic writes preserve symlinks."""
    return Path(os.path.realpath(path))


def _backup(path: Path) -> Path:
    """Create a collision-safe, owner-only backup of secret-bearing config."""
    path = _config_target(path)
    backup = path.with_name(path.name + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(backup, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with path.open("rb") as source, os.fdopen(fd, "wb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        backup.unlink(missing_ok=True)
        raise
    print(f"  backup: {backup.name}")
    return backup


def _write_config(path: Path, config: dict) -> None:
    """Persist secret-bearing config atomically with owner-only permissions."""
    path = _config_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        tmp.unlink(missing_ok=True)
        raise


def _restore_backup(path: Path, backup: Path) -> None:
    """Atomically restore a raw config backup without replacing symlinks."""
    path = _config_target(path)
    backup = _config_target(backup)
    if not backup.is_file():
        raise RuntimeError(f"rollback backup is missing: {backup}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.rollback.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with backup.open("rb") as source, os.fdopen(fd, "wb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        tmp.replace(path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        tmp.unlink(missing_ok=True)
        raise


def _require_activation_path() -> None:
    if not RESTART_BIN and not ADMIN_KEY:
        raise _fail(
            "mutating workspace commands require MODEL_GATEWAY_RESTART_BIN "
            "or MODEL_GATEWAY_ADMIN_KEY"
        )


def reload_gateway() -> None:
    if not ADMIN_KEY:
        raise RuntimeError("MODEL_GATEWAY_ADMIN_KEY is required for API reload")
    req = urllib.request.Request(
        f"{GATEWAY_URL}/admin/api/reload", method="POST",
        headers={"Authorization": f"Bearer {ADMIN_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
        print(f"  reload: {data.get('message')} (catalogs: {data.get('catalogs')})")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"gateway reload failed: {exc}") from exc


def restart_gateway() -> None:
    if not RESTART_BIN:
        raise RuntimeError("MODEL_GATEWAY_RESTART_BIN is not configured")
    try:
        proc = subprocess.run(
            [RESTART_BIN, "restart"],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"gateway restart failed: {exc}") from exc
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "unknown error").strip()[-1000:]
        raise RuntimeError(f"gateway restart failed (exit {proc.returncode}): {detail}")
    print("  activation: gateway restarted and verified")


def activate_gateway() -> None:
    if RESTART_BIN:
        restart_gateway()
    else:
        reload_gateway()


def _commit_and_activate(path: Path, config: dict) -> None:
    """Commit verified config, activate it, and roll back on activation failure."""
    backup = _backup(path)
    _write_config(path, config)
    print(f"  commit: {_config_target(path)}")
    try:
        activate_gateway()
    except Exception as activation_error:
        print(f"  activation: FAILED — {activation_error}")
        print(f"  rollback: restoring {backup.name}")
        try:
            _restore_backup(path, backup)
            activate_gateway()
        except Exception as rollback_error:
            raise _fail(
                "activation failed and rollback could not be activated; "
                f"config was restored from {backup}, but the gateway needs manual recovery: "
                f"{rollback_error}"
            ) from rollback_error
        raise _fail(
            f"activation failed; previous configuration restored and verified: {activation_error}"
        ) from activation_error


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
        host = str(entry.get("workspace_url") or entry.get("base_url", ""))
        if (
            name not in member_of
            and entry.get("auth_refresh") != "databricks-cli"
            and "databricks" not in host
        ):
            continue
        auth = entry.get("auth_profile") or ("pat" if str(entry.get("api_key", "")).startswith("dapi") else "static")
        print(f"{name:22s} {entry.get('base_url','')[:55]:55s} {auth:14s} {','.join(member_of.get(name, [])) or '-'}")
    print("\nPools:")
    for pool, members in pools.items():
        print(f"  {pool}: {' → '.join(members or [])}")


def _verify(config: dict, host: str, profile: str, pool_names: list[str], allow_partial: bool,
            token: str | None = None, provider_names: list[str] = (), *,
            style: str = "invocations", base_url: str | None = None) -> list[str] | None:
    """Control-plane (auth + endpoint coverage) then data-plane (one completion
    through the exact committed route). Returns the served provider_model_ids."""
    token = token or ensure_auth(host, profile)
    endpoint_names = probe_endpoints(host, token)
    available = check_coverage(config, pool_names, endpoint_names, allow_partial, provider_names)
    protocols = _model_protocols(config, pool_names, provider_names) if style == "ai-gateway" else None
    smoke_test(host, token, endpoint_names, style=style, base_url=base_url, protocols=protocols,
               require_bootstrap=allow_partial)
    return available


def _entry_style(entry: dict) -> str:
    if entry.get("endpoint_style") == "invocations":
        return "invocations"
    return "ai-gateway" if isinstance(entry.get("path_prefixes"), dict) else "invocations"


def _resolve_route(style: str, host: str, profile: str, config: dict, pool_names: list[str],
                   provider_names: list[str]) -> tuple[str, str, str, str]:
    """Pick the concrete route for ``style`` (invocations|ai-gateway|auto).

    Returns (final_style, workspace_host, base_url, token). ``auto`` tries the
    derived AI Gateway host with a real data-plane probe first and falls back
    to direct invocations when that route is unusable (DNS, TLS, connect, path
    or protocol failures); auth failures are NOT masked by fallback.
    """
    if style == "invocations":
        return "invocations", host, host, ensure_auth(host, profile)
    verify_host, base_url, token = _resolve_ai_gateway_host(host, profile)
    if style == "ai-gateway":
        print(f"  ai-gateway: {base_url} (workspace {verify_host})")
        return "ai-gateway", verify_host, base_url, token
    # auto
    endpoint_names = probe_endpoints(verify_host, token)
    protocols = _model_protocols(config, pool_names, provider_names)
    try:
        used = runtime_preflight("ai-gateway", verify_host, base_url, token, endpoint_names, protocols)
        print(f"  route: ai-gateway {base_url} OK ({used})")
        return "ai-gateway", verify_host, base_url, token
    except RoutePreflightError as exc:
        if exc.kind == "auth":
            raise _fail(f"AI Gateway route rejected the workspace credential ({exc.detail}); not falling back")
        print(f"  route: ai-gateway {base_url} unusable ({exc.kind}: {exc.detail})")
        print(f"  route: falling back to direct invocations on {verify_host}")
        return "invocations", verify_host, verify_host, token


def _provider_entry(style: str, base_url: str, token: str, profile: str, workspace_host: str,
                    inherit: dict | None = None) -> dict:
    """Build a complete provider entry for ``style``, dropping fields that
    belong to the other style so a converted entry has no stale routing."""
    entry = dict(inherit or {})
    for k in ("endpoint_style", "path_prefixes", "workspace_url", "quirks", "protocol"):
        entry.pop(k, None)
    entry.update({"base_url": base_url, "api_key": token, "protocol": "openai",
                  "auth_refresh": "databricks-cli", "auth_profile": profile})
    if style == "invocations":
        entry["endpoint_style"] = "invocations"
        entry["quirks"] = ["no_stream_options", "no_reasoning_params"]
    else:
        entry["path_prefixes"] = dict(AI_GATEWAY_PATH_PREFIXES)
        entry["quirks"] = ["anthropic_bearer_auth"]
        entry["workspace_url"] = workspace_host
    return entry


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
    check_coverage(config, pools, endpoint_names, allow_partial=True, provider_names=[args.name])
    style = _entry_style(entry)
    protocols = _model_protocols(config, pools, [args.name]) if style == "ai-gateway" else None
    smoke_test(host, token, endpoint_names, style=style, base_url=str(entry.get("base_url", "")),
               protocols=protocols)
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
    _require_activation_path()
    host = normalize_host(args.host)
    profile = args.profile or args.name
    pool_names = [p.strip() for p in (args.pools or "").split(",") if p.strip()]
    config = _load_config(args.config)

    staged = copy.deepcopy(config)
    _insert_into_pools(staged, args.name, pool_names, args.position)
    style, verify_host, base_url, token = _resolve_route(
        args.style, host, profile, staged, pool_names, [args.name])
    print(f"Adding workspace {args.name!r} ({verify_host}, {style}) to pools: {', '.join(pool_names) or 'none'}")
    available = _verify(staged, verify_host, profile, pool_names, args.allow_partial, token=token,
                        provider_names=[args.name], style=style, base_url=base_url)

    providers = _providers_section(config)
    entry = _provider_entry(style, base_url, token, profile, verify_host)
    _stamp_coverage(entry, available)
    providers[args.name] = entry
    _insert_into_pools(config, args.name, pool_names, args.position)

    _commit_and_activate(args.config, config)
    print(f"workspace add: DONE — {args.name} is live via {style}")


def cmd_replace(args) -> None:
    _require_activation_path()
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

    requested = getattr(args, "style", "inherit") or "inherit"
    style = _entry_style(old) if requested == "inherit" else requested

    staged = copy.deepcopy(config)
    for pool in affected:  # stage: swap in place, keep position
        members = staged["pools"][pool]
        members[members.index(args.old_name)] = new_name
    style, verify_host, base_url, token = _resolve_route(style, host, profile, staged, affected, [new_name])
    print(f"Replacing workspace {args.old_name!r} with {new_name!r} ({verify_host}, {style}) "
          f"in pools: {', '.join(affected) or 'none'}")
    available = _verify(staged, verify_host, profile, affected, args.allow_partial, token=token,
                        provider_names=[new_name], style=style, base_url=base_url)

    entry = _provider_entry(style, base_url, token, profile, verify_host, inherit=old)
    _stamp_coverage(entry, available)
    providers[new_name] = entry
    if new_name != args.old_name:
        providers.pop(args.old_name, None)
    for pool in affected:
        members = pools[pool]
        members[members.index(args.old_name)] = new_name

    _commit_and_activate(args.config, config)
    print(f"workspace replace: DONE — {new_name} took over {args.old_name}'s pool positions via {style}")


def cmd_repair(args) -> None:
    """Interactive recovery pass over every OAuth-backed workspace.

    For each workspace with auth_refresh: databricks-cli, escalate:
      silent token → browser SSO → prompt to paste a replacement workspace URL
    (the paste-a-URL flow). Static-PAT workspaces are probe-checked only.
    """
    _require_activation_path()
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
                name=None, profile=None, allow_partial=True, style="inherit",
            )
            cmd_replace(ns)
            broken.remove(name)
    if broken:
        raise _fail(f"still broken: {', '.join(broken)}")
    print("\nworkspace repair: all OAuth-backed workspaces healthy")


def cmd_remove(args) -> None:
    _require_activation_path()
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
    _commit_and_activate(args.config, config)
    print(f"workspace remove: DONE — {args.name} removed")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="model-gateway workspace",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")
    sub.add_parser("repair")

    p = sub.add_parser("test")
    p.add_argument("name")

    p = sub.add_parser("add")
    p.add_argument("name")
    p.add_argument("--host", required=True,
                   help="workspace URL (recommended for both styles; for --style "
                        "ai-gateway the <org-id>.ai-gateway host is derived from it)")
    p.add_argument("--pools", default="")
    p.add_argument("--position", type=int, default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--style", choices=["invocations", "ai-gateway", "auto"], default="invocations",
                   help="route style; auto = try the derived AI Gateway route with a real "
                        "completion, fall back to direct invocations if it is unusable")
    p.add_argument("--allow-partial", action="store_true")

    p = sub.add_parser("replace")
    p.add_argument("old_name")
    p.add_argument("--host", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--style", choices=["inherit", "invocations", "ai-gateway", "auto"], default="inherit",
                   help="route style for the replacement (default: inherit the old entry's style)")
    p.add_argument("--allow-partial", action="store_true")

    p = sub.add_parser("remove")
    p.add_argument("name")

    args = parser.parse_args()
    {"list": cmd_list, "repair": cmd_repair, "test": cmd_test, "add": cmd_add,
     "replace": cmd_replace, "remove": cmd_remove}[args.command](args)


if __name__ == "__main__":
    main()
