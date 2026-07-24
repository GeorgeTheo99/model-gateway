"""Provider registry — resolve model name to endpoint + credentials."""

import asyncio
import base64
import copy
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml

from src import catalog
from src.config_lock import config_write_lock
from src.secret_files import read_api_key_file

log = logging.getLogger("model-gateway")

_registry_lock = threading.RLock()


def _registry_locked(function):
    """Keep readers from observing a candidate registry before validation."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _registry_lock:
            return function(*args, **kwargs)
    return wrapped


@contextmanager
def registry_transaction():
    """Serialize registry replacement with all registry-backed readers."""
    with _registry_lock:
        yield


# model-info.json is the machine-local model catalog. Allow the live path to be
# supplied by the launcher/deploy; fall back to the checkout-local Git-ignored
# catalog for portable installs, tests, and local development.
_DEFAULT_MODEL_INFO = Path(__file__).resolve().parents[1] / "model-info.json"
MODEL_INFO_PATH = Path(
    os.environ.get("MODEL_GATEWAY_MODEL_INFO") or str(_DEFAULT_MODEL_INFO)
).resolve()
# Optional machine-local mirror for durable model edits. When set, admin and
# onboarding writes update it in addition to MODEL_INFO_PATH. Deploy tooling may
# use separate live/mirror paths; portable installs normally use one file.
_DEFAULT_MODEL_INFO_SOURCE = Path.home() / "local_code" / "model-gateway" / "model-info.json"
MODEL_INFO_SOURCE_PATH = (
    Path(
        os.environ.get("MODEL_GATEWAY_MODEL_INFO_SOURCE")
        or str(_DEFAULT_MODEL_INFO_SOURCE)
    ).resolve()
    if os.environ.get("MODEL_GATEWAY_MODEL_INFO_SOURCE")
    or _DEFAULT_MODEL_INFO_SOURCE.exists()
    else None
)
CONFIG_PATH = Path(
    os.environ.get("MODEL_GATEWAY_CONFIG")
    or Path(__file__).resolve().parents[1] / "config" / "config.yaml"
).resolve()

_DEFAULT_OMLX_BASE_URL = os.environ.get("MODEL_GATEWAY_OMLX_BASE_URL", "http://localhost:9110/v1")
_DEFAULT_OMLX_API_KEY = os.environ.get("MODEL_GATEWAY_OMLX_API_KEY", "omlx")


@dataclass(frozen=True)
class CompositeRoute:
    """Gateway-owned local text+vision composition policy."""

    text_model: str
    vision_model: str
    image_handling: str = "extract_then_answer"
    max_images: int = 4


@dataclass
class ProviderInfo:
    provider: str
    base_url: str
    api_key: str
    provider_model_id: str
    protocol: str = "openai"  # "openai" (Chat Completions) or "anthropic" (Messages API)
    # Path appended to base_url for chat/messages requests. None => server default
    # ("/chat/completions" or "/messages"). "" => base_url is already a complete
    # invocation URL (e.g. endpoint_style: invocations providers).
    endpoint_suffix: str | None = None
    # Provider quirk flags from config (e.g. "no_stream_options",
    # "no_reasoning_params"). Generic mechanism; which providers need which
    # quirks is runtime config, not code.
    quirks: frozenset = frozenset()
    context: int = 0
    max_output_tokens: int = 0
    thinking: str = ""  # "", "optional", or "always"
    # None preserves compatibility for callers constructing ProviderInfo from
    # old entries; catalog-loaded models always carry a validated explicit tuple.
    thinking_levels: tuple[str, ...] | None = None
    thinking_format: str = ""  # optional explicit gateway normalization format
    system_instruction: str = ""
    vision: bool = False  # authoritative: True if the model can natively handle image inputs
    pricing: dict = None  # $/Mtok rates: input, output, cache_read?, cache_write?, reasoning?
    # Logical composites deliberately keep ``vision=False`` here because the
    # resolved upstream is the text model. The public catalog entry remains
    # ``vision: true`` so clients preserve image blocks for gateway staging.
    composite: CompositeRoute | None = None


_config: dict | None = None
_models: dict[str, dict] | None = None

# OAuth token auto-refresh via an external CLI token cache (opt-in per provider
# with `auth_refresh:` in config.yaml). See refresh_oauth_token().
_AUTH_REFRESH_MIN_INTERVAL = 60.0  # seconds between CLI refresh attempts per provider
_AUTH_LOGIN_MIN_INTERVAL = 300.0  # seconds between browser SSO attempts per provider
_AUTH_LOGIN_TIMEOUT = 300.0  # seconds to wait for the browser SSO callback
_token_refresh_locks: dict[str, "asyncio.Lock"] = {}
_last_token_refresh_attempt: dict[str, float] = {}
_last_auth_login_attempt: dict[str, float] = {}

# Provider synonym table lives in ``src.catalog`` (the single source shared with
# the downstream catalog generator). ``tests/test_catalog.py`` guards drift.


def _canonical_provider(provider: str | None) -> str:
    return catalog.canonical_provider(provider)


def _find_provider_location(config: dict, provider: str) -> tuple[str, str, dict] | None:
    """Return (section, config_key, entry), matching canonical synonyms."""
    for section in ("providers", "workspaces"):
        entries = config.get(section, {}) or {}
        if not isinstance(entries, dict):
            continue
        direct = entries.get(provider)
        if direct:
            return section, provider, direct
        for key, value in entries.items():
            if _canonical_provider(key) == provider:
                return section, key, value or {}
    return None


def _find_provider_entry(config: dict, provider: str) -> tuple[str, dict] | None:
    """Return (config_key, entry) for a provider, matching canonical synonyms."""
    found = _find_provider_location(config, provider)
    return (found[1], found[2]) if found else None


def _pools(config: dict) -> dict[str, list[str]]:
    """Ordered workspace-failover pools: pool name -> [provider names]."""
    raw = config.get("pools") or {}
    if not isinstance(raw, dict):
        return {}
    pools: dict[str, list[str]] = {}
    for name, members in raw.items():
        if isinstance(members, str):
            members = [members]
        if isinstance(members, list):
            cleaned = [str(m).strip() for m in members if str(m).strip()]
            if cleaned:
                pools[str(name)] = cleaned
    return pools


def _pool_members(entry: dict, config: dict) -> list[str]:
    """Ordered candidate providers for a model entry.

    A ``pool:`` reference expands to the pool's member list; a plain
    ``provider:`` is a single-member pool. Unknown pool names fall back to the
    entry's provider so a config typo degrades to current behavior.
    """
    pool_name = entry.get("pool")
    if pool_name:
        members = _pools(config).get(str(pool_name))
        if members:
            return [_canonical_provider(m) for m in members]
        log.warning("Model %r references unknown pool %r", entry.get("name"), pool_name)
    return [_canonical_provider(entry.get("provider", "local"))]


def _resolve_provider_config(config: dict, provider: str) -> dict:
    found = _find_provider_entry(config, provider)
    return found[1] if found else {}


def _provider_defaults(provider: str) -> dict:
    """Built-in provider defaults for local backends managed on this machine."""
    if provider == "omlx":
        return {
            "base_url": _DEFAULT_OMLX_BASE_URL,
            "api_key": _DEFAULT_OMLX_API_KEY,
            "protocol": "openai",
        }
    return {}


def _provider_env_prefix(provider: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in provider.upper())
    return f"MODEL_GATEWAY_PROVIDER_{normalized}"


def _env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _provider_env_config(provider: str) -> dict:
    """Provider config from environment variables.

    Generic shape:
      MODEL_GATEWAY_PROVIDER_<PROVIDER>_{BASE_URL,API_KEY,PROTOCOL,ENABLED}

    Databricks additionally accepts the standard workspace env vars:
      DATABRICKS_HOST + DATABRICKS_TOKEN, or DATABRICKS_SERVING_BASE_URL.
    """
    prefix = _provider_env_prefix(provider)
    env_config: dict = {}
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    api_key = os.environ.get(f"{prefix}_API_KEY")
    protocol = os.environ.get(f"{prefix}_PROTOCOL")
    enabled = _env_bool(os.environ.get(f"{prefix}_ENABLED"))

    if provider == "databricks":
        db_base_url = os.environ.get("DATABRICKS_SERVING_BASE_URL")
        db_host = os.environ.get("DATABRICKS_HOST")
        db_token = os.environ.get("DATABRICKS_TOKEN")
        if not base_url:
            if db_base_url:
                base_url = db_base_url
            elif db_host:
                base_url = f"{db_host.rstrip('/')}/serving-endpoints"
        if not api_key and db_token:
            api_key = db_token
        if enabled is None and (base_url or api_key):
            enabled = True
        protocol = protocol or "openai"

    if base_url is not None:
        env_config["base_url"] = base_url
    if api_key is not None:
        env_config["api_key"] = api_key
    if protocol is not None:
        env_config["protocol"] = protocol
    if enabled is not None:
        env_config["enabled"] = enabled
    return env_config


def _effective_provider_config(config: dict, provider: str) -> dict:
    """Provider config with defaults + config.yaml + env overrides.

    ``api_key_file`` keeps long-lived secrets out of YAML. Relative paths are
    resolved beside config.yaml; an explicit environment API key still wins.
    """
    effective = _provider_defaults(provider)
    effective.update(_resolve_provider_config(config, provider))
    effective.update(_provider_env_config(provider))
    if not effective.get("api_key") and effective.get("api_key_file"):
        try:
            effective["api_key"] = read_api_key_file(effective["api_key_file"], CONFIG_PATH)
        except OSError as exc:
            log.warning("Provider %r api_key_file is unusable: %s", provider, exc)
            effective["api_key"] = ""
    return effective


@_registry_locked
def _load_config() -> dict:
    global _config
    if _config is not None:
        return _config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            _config = yaml.safe_load(f) or {}
    else:
        _config = {}
    return _config


def _entry_routable_ids(entry: dict) -> list[str]:
    """Return every gateway-facing identifier for a catalog entry."""
    return catalog.entry_routable_ids(entry)


@_registry_locked
def _load_models() -> dict[str, dict]:
    """Load routable models from model-info.json (keyed by name/alias/id).

    The catalog + config.yaml ``models:`` overlay merge lives in
    :func:`src.catalog.load_catalog_entries` and is shared with the downstream
    catalog generator so the router and the generator can never drift.
    """
    global _models
    if _models is not None:
        return _models

    config = _load_config()
    overlay = config.get("models", [])
    if not isinstance(overlay, list):
        raise ValueError("config models overlay must be a list")
    entries = catalog.load_catalog_entries(
        MODEL_INFO_PATH,
        overlay=overlay,
        require_nonempty=True,
    )
    loaded: dict[str, dict] = {}
    for entry in entries:
        for model_id in _entry_routable_ids(entry):
            loaded[model_id] = entry

    _models = loaded
    log.info("Loaded %d routable model keys (catalog + config overlay)", len(_models))
    return _models


@_registry_locked
def routable_ids(name: str) -> list[str]:
    """Return every identifier that routes to the named model.

    Used by per-model stats to match ledger rows regardless of which alias or
    upstream id the caller sent. Includes name, alias, provider_model_id,
    omlx_id, and any alternate_ids, with empties/duplicates removed.
    """
    entry = _load_models().get(name)
    if not entry:
        return [name] if name else []
    return sorted(_entry_routable_ids(entry))


@_registry_locked
def provider_quirks(provider: str) -> frozenset:
    """Return the quirk set for a provider from live runtime config.

    Used by header construction and other call sites that only have the
    provider name (no resolved ProviderInfo).
    """
    config = _load_config()
    provider_config = _effective_provider_config(config, _canonical_provider(provider))
    quirks_raw = provider_config.get("quirks") or []
    if isinstance(quirks_raw, str):
        quirks_raw = [q.strip() for q in quirks_raw.split(",") if q.strip()]
    return frozenset(quirks_raw)


@_registry_locked
def auth_config() -> dict:
    """Return the live ``auth`` section from config.yaml.

    Re-reads after :func:`reload` resets the cached config. Used by
    :mod:`src.auth` so admin/client keys can live in the gitignored
    config.yaml instead of the launchd plist environment.
    """
    return _load_config().get("auth") or {}


@_registry_locked
def model_overrides() -> dict:
    """Return the live ``model_overrides`` section from config.yaml.

    Runtime state for models (currently just ``enabled``) lives here, not in
    model-info.json, so admin enable/disable toggles do not rewrite the catalog.
    Keys are model names; values are
    dicts with ``enabled: bool``. Missing entry = enabled (default true).

    Example config.yaml::

        model_overrides:
          gemini-3-flash:
            enabled: false
    """
    return _load_config().get("model_overrides") or {}


def _is_model_enabled(name: str | None) -> bool:
    """Check the runtime enabled override for a model. Default True."""
    if not name:
        return True
    overrides = model_overrides()
    entry = overrides.get(name)
    if isinstance(entry, dict):
        return bool(entry.get("enabled", True))
    return True


@_registry_locked
def snapshot_registry() -> tuple[dict, dict[str, dict]]:
    """Capture the loaded registry so a rejected admin reload can roll back."""
    return _load_config(), _load_models()


@_registry_locked
def restore_registry(snapshot: tuple[dict, dict[str, dict]]) -> None:
    """Restore a snapshot returned by :func:`snapshot_registry`."""
    global _config, _models
    _config, _models = snapshot


@_registry_locked
def reload():
    """Force reload of config and models (e.g. after config change)."""
    global _config, _models
    _config = None
    _models = None


def _databricks_cli() -> str:
    """Path to the databricks CLI (launchd PATH may not include homebrew)."""
    from shutil import which
    return which("databricks") or "/opt/homebrew/bin/databricks"


def _jwt_expiry_epoch(token: str) -> float | None:
    """Return JWT ``exp`` as epoch seconds, or None when the token is opaque."""
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode())
        exp = json.loads(decoded).get("exp")
    except Exception:
        return None
    if isinstance(exp, (int, float)):
        return float(exp)
    return None


def _replace_provider_api_key_line(
    lines: list[str], section: str, config_key: str, token: str,
) -> bool:
    """Replace only a direct ``api_key`` child of the selected provider block."""
    section_re = re.compile(rf"^([ \t]*){re.escape(section)}:[ \t]*(?:#.*)?$")
    key_re = re.compile(rf"^([ \t]*){re.escape(config_key)}:[ \t]*(?:#.*)?$")
    section_matches = []
    for index, line in enumerate(lines):
        match = section_re.match(line.rstrip("\n"))
        if match and not match.group(1):
            section_matches.append(index)
    if len(section_matches) != 1:
        return False
    section_start = section_matches[0]
    section_indent = 0

    provider_start = None
    provider_indent = None
    direct_indent = None
    for index in range(section_start + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[index]) - len(lines[index].lstrip(" \t"))
        if indent <= section_indent:
            break
        if direct_indent is None:
            direct_indent = indent
        if indent == direct_indent and key_re.match(lines[index].rstrip("\n")):
            provider_start = index
            provider_indent = indent
            break
    if provider_start is None or provider_indent is None:
        return False

    api_key_re = re.compile(r"^([ \t]*api_key:[ \t]*)(\S+)(.*)$")
    field_indent = None
    for index in range(provider_start + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[index]) - len(lines[index].lstrip(" \t"))
        if indent <= provider_indent:
            break
        if field_indent is None:
            field_indent = indent
        if indent != field_indent:
            continue
        match = api_key_re.match(lines[index].rstrip("\n"))
        if match:
            lines[index] = f"{match.group(1)}{token}{match.group(3)}\n"
            return True
    return False


def _atomic_replace_config_text(text: str) -> None:
    """Replace the config target without widening its secret-bearing mode."""
    target = Path(os.path.realpath(CONFIG_PATH))
    mode = target.stat().st_mode & 0o777 if target.exists() else 0o600
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _persist_api_key(
    config_key: str,
    token: str,
    *,
    expected_entry: dict | None = None,
) -> bool:
    """Compare-and-swap a rotated key under the shared config transaction."""
    with config_write_lock(CONFIG_PATH):
        with registry_transaction():
            try:
                if not CONFIG_PATH.exists():
                    return False
                text = CONFIG_PATH.read_text()
                disk_config = yaml.safe_load(text) or {}
                disk_location = _find_provider_location(disk_config, config_key)
                cached_location = _find_provider_location(_load_config(), config_key)
                if not disk_location or not cached_location:
                    return False
                section, actual_key, disk_entry = disk_location
                if expected_entry is not None and (
                    disk_entry != expected_entry or cached_location[2] != expected_entry
                ):
                    log.warning(
                        "Discarding stale OAuth refresh for provider %r after config changed",
                        config_key,
                    )
                    return False

                lines = text.splitlines(keepends=True)
                if not _replace_provider_api_key_line(lines, section, actual_key, token):
                    log.warning(
                        "Could not persist refreshed api_key for %r to %s",
                        config_key,
                        CONFIG_PATH,
                    )
                    return False
                _atomic_replace_config_text("".join(lines))
                cached_location[2]["api_key"] = token
                return True
            except (OSError, yaml.YAMLError) as exc:
                log.warning("Failed to persist refreshed api_key for %r: %s", config_key, exc)
                return False


async def refresh_oauth_token(provider: str, *, force: bool = False) -> str | None:
    """Refresh an expired OAuth token via an external CLI token cache.

    Opt-in per provider with config.yaml::

        providers:
          my_workspace:
            auth_refresh: databricks-cli   # only supported refresher today
            auth_profile: my-cli-profile   # optional; falls back to --host <base_url>

    Called by the server before expiry and when an upstream returns 401/403
    (e.g. a short-lived OAuth JWT expired). If the CLI token cache itself is
    broken, the gateway runs ``databricks auth login`` once per cooldown window,
    then retries token minting. Providers without ``auth_refresh`` are
    untouched. Returns the new token if one was obtained, else None.
    """
    config = _load_config()
    found = _find_provider_entry(config, provider)
    if not found:
        return None
    _config_key, entry = found
    if (entry.get("auth_refresh") or "") != "databricks-cli":
        return None
    current = entry.get("api_key", "")
    # Only short-lived OAuth JWTs ("eyJ...") benefit; PATs never expire this way.
    if not current.startswith("eyJ"):
        return None

    lock = _token_refresh_locks.setdefault(provider, asyncio.Lock())
    async with lock:
        # Re-read after acquiring the async refresh lease. Admin reloads replace
        # the registry while an earlier caller may be waiting here.
        latest_found = _find_provider_entry(_load_config(), provider)
        if not latest_found:
            return None
        config_key, entry = latest_found
        latest = entry.get("api_key", "")
        if latest != current:
            return latest
        if (entry.get("auth_refresh") or "") != "databricks-cli":
            return None
        expected_entry = copy.deepcopy(entry)
        base_url = (entry.get("base_url") or "").rstrip("/")

        now = time.monotonic()
        if not force and now - _last_token_refresh_attempt.get(provider, 0.0) < _AUTH_REFRESH_MIN_INTERVAL:
            return None
        _last_token_refresh_attempt[provider] = now

        profile = entry.get("auth_profile", "")
        # --host wants the workspace origin, not a path under it.
        host = base_url.split("/serving-endpoints")[0] if base_url else ""
        cli_args = ["--profile", profile] if profile else ["--host", host]

        async def run_token() -> tuple[int, bytes, bytes]:
            proc = await asyncio.create_subprocess_exec(
                _databricks_cli(), "auth", "token", *cli_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            return proc.returncode, stdout, stderr

        try:
            returncode, stdout, stderr = await run_token()
        except (OSError, asyncio.TimeoutError) as exc:
            log.error("OAuth token refresh for %r failed: %s", provider, exc)
            return None

        if returncode != 0 and entry.get("auth_login", True) is not False:
            log.error(
                "OAuth token refresh for %r failed (exit %d): %s",
                provider, returncode, stderr.decode(errors="replace")[:300],
            )
            login_now = time.monotonic()
            if not host:
                log.error("Cannot launch Databricks browser SSO for %r: missing base_url", provider)
            elif login_now - _last_auth_login_attempt.get(provider, 0.0) >= _AUTH_LOGIN_MIN_INTERVAL:
                _last_auth_login_attempt[provider] = login_now
                login_args = ["auth", "login", "--host", host]
                if profile:
                    login_args.extend(["--profile", profile])
                log.warning(
                    "Launching Databricks browser SSO for provider %r (%s, profile %r)",
                    provider, host, profile or "<host>",
                )
                try:
                    login = await asyncio.create_subprocess_exec(
                        _databricks_cli(), *login_args,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    login_stdout, login_stderr = await asyncio.wait_for(
                        login.communicate(), timeout=_AUTH_LOGIN_TIMEOUT,
                    )
                except (OSError, asyncio.TimeoutError) as exc:
                    log.error("Databricks browser SSO for %r failed: %s", provider, exc)
                    return None
                if login.returncode != 0:
                    detail = (login_stderr or login_stdout).decode(errors="replace")[:300]
                    log.error(
                        "Databricks browser SSO for %r failed (exit %d): %s",
                        provider, login.returncode, detail,
                    )
                    return None
                try:
                    returncode, stdout, stderr = await run_token()
                except (OSError, asyncio.TimeoutError) as exc:
                    log.error("OAuth token refresh for %r failed after SSO: %s", provider, exc)
                    return None
            else:
                log.warning(
                    "Skipping Databricks browser SSO for %r; attempted within the last %.0fs",
                    provider, _AUTH_LOGIN_MIN_INTERVAL,
                )
        if returncode != 0:
            log.error(
                "OAuth token refresh for %r failed (exit %d): %s",
                provider, returncode, stderr.decode(errors="replace")[:300],
            )
            return None
        try:
            token = json.loads(stdout).get("access_token", "")
        except json.JSONDecodeError:
            log.error("OAuth token refresh for %r: unparseable CLI output", provider)
            return None
        if not token or token == current:
            return None

        if not _persist_api_key(config_key, token, expected_entry=expected_entry):
            latest_found = _find_provider_entry(_load_config(), provider)
            latest = latest_found[1].get("api_key", "") if latest_found else ""
            return latest if latest and latest != current else None
        log.warning("Refreshed expired OAuth token for provider %r via CLI token cache", provider)
        return token


async def ensure_fresh_oauth_token(provider: str, *, min_valid_seconds: int = 300) -> str | None:
    """Refresh a configured OAuth JWT before it expires.

    PAT-backed providers and opaque API keys are left untouched. This is a
    lightweight preflight used by request routing so OAuth-backed Databricks
    workspaces do not depend on an external timer job.
    """
    config = _load_config()
    found = _find_provider_entry(config, provider)
    if not found:
        return None
    _config_key, entry = found
    if (entry.get("auth_refresh") or "") != "databricks-cli":
        return None
    current = entry.get("api_key", "")
    if not current.startswith("eyJ"):
        return None
    exp = _jwt_expiry_epoch(current)
    if exp is None or exp - time.time() > min_valid_seconds:
        return None
    return await refresh_oauth_token(provider, force=True)


def _configured_pool_members(entry: dict, config: dict) -> list[str]:
    """Pool members that have usable local config (base_url + api_key, enabled)."""
    members = []
    for member in _pool_members(entry, config):
        provider_config = _effective_provider_config(config, member)
        if provider_config.get("enabled") is False:
            continue
        if provider_config.get("base_url") and provider_config.get("api_key"):
            members.append(member)
    return members


def _parse_composite_route(entry: dict) -> CompositeRoute | None:
    raw = entry.get("composite")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("composite must be an object")
    text_model = str(raw.get("text_model") or "").strip()
    vision_model = str(raw.get("vision_model") or "").strip()
    image_handling = str(raw.get("image_handling") or "extract_then_answer").strip().lower()
    max_images = raw.get("max_images", 4)
    if not text_model or not vision_model:
        raise ValueError("composite requires text_model and vision_model")
    if text_model == vision_model:
        raise ValueError("composite text_model and vision_model must differ")
    if image_handling not in {"extract_then_answer", "reroute"}:
        raise ValueError("composite image_handling must be extract_then_answer or reroute")
    if isinstance(max_images, bool) or not isinstance(max_images, int) or not 1 <= max_images <= 8:
        raise ValueError("composite max_images must be an integer from 1 through 8")
    return CompositeRoute(
        text_model=text_model,
        vision_model=vision_model,
        image_handling=image_handling,
        max_images=max_images,
    )


def _availability_for_entry(model_id: str, entry: dict | None) -> dict:
    if not entry:
        return {
            "available": False,
            "reason": "model_not_found",
            "message": f"Model {model_id!r} is not in model-info.json",
        }

    name = entry.get("name") or model_id
    provider = _pool_members(entry, _load_config())[0]
    if not _is_model_enabled(name):
        return {
            "available": False,
            "reason": "model_disabled",
            "model": name,
            "provider": provider,
            "message": f"Model {name!r} is disabled by runtime model_overrides",
        }

    try:
        composite = _parse_composite_route(entry)
    except ValueError as exc:
        return {
            "available": False,
            "reason": "invalid_composite",
            "model": name,
            "provider": provider,
            "message": f"Composite model {name!r} is invalid: {exc}",
        }
    if composite is not None:
        models = _load_models()
        dependencies = (
            ("text_model", composite.text_model, False),
            ("vision_model", composite.vision_model, True),
        )
        dependency_provider = ""
        for role, dependency_id, requires_vision in dependencies:
            dependency = models.get(dependency_id)
            if dependency is entry or (dependency and dependency.get("composite") is not None):
                return {
                    "available": False,
                    "reason": "invalid_composite",
                    "model": name,
                    "provider": provider,
                    "message": f"Composite model {name!r} cannot contain nested or cyclic composites",
                }
            dependency_members = _pool_members(dependency, _load_config()) if dependency else []
            if not dependency_members or any(_canonical_provider(member) != "omlx" for member in dependency_members):
                return {
                    "available": False,
                    "reason": "invalid_composite",
                    "model": name,
                    "provider": dependency_members[0] if dependency_members else provider,
                    "message": f"Composite model {name!r} dependencies must use only local oMLX providers",
                }
            availability = _availability_for_entry(dependency_id, dependency)
            if not availability.get("available"):
                return {
                    "available": False,
                    "reason": f"composite_{role}_unavailable",
                    "model": name,
                    "provider": availability.get("provider") or provider,
                    "message": (
                        f"Composite model {name!r} dependency {dependency_id!r} is unavailable: "
                        f"{availability.get('message', availability.get('reason', 'unknown error'))}"
                    ),
                }
            dependency_provider = availability.get("provider") or dependency_provider
            if _canonical_provider(dependency_provider) != "omlx":
                return {
                    "available": False,
                    "reason": "invalid_composite",
                    "model": name,
                    "provider": dependency_provider,
                    "message": f"Composite model {name!r} dependencies must use local oMLX",
                }
            if requires_vision and not bool(dependency.get("vision", False)):
                return {
                    "available": False,
                    "reason": "invalid_composite",
                    "model": name,
                    "provider": dependency_provider,
                    "message": f"Composite vision dependency {dependency_id!r} is not vision-capable",
                }
        return {
            "available": True,
            "reason": "",
            "model": name,
            "provider": "omlx",
            "message": "",
        }

    config = _load_config()
    configured = _configured_pool_members(entry, config)
    if configured:
        return {"available": True, "reason": "", "model": name, "provider": configured[0], "message": ""}

    # Nothing in the pool is usable — report why using the first member.
    provider_config = _effective_provider_config(config, provider)
    if provider_config.get("enabled") is False:
        return {
            "available": False,
            "reason": "provider_disabled",
            "model": name,
            "provider": provider,
            "message": f"Provider {provider!r} is disabled by runtime config",
        }

    missing = []
    if not provider_config.get("base_url"):
        missing.append("base_url")
    if not provider_config.get("api_key"):
        missing.append("api_key")
    return {
        "available": False,
        "reason": "provider_not_configured",
        "model": name,
        "provider": provider,
        "missing": missing,
        "message": f"Provider {provider!r} is missing {', '.join(missing)} in local runtime config",
    }


@_registry_locked
def model_availability(model_id: str) -> dict:
    """Return availability details for a model identifier without exposing secrets."""
    return _availability_for_entry(model_id, _load_models().get(model_id))


@_registry_locked
def is_model_available(model_id: str) -> bool:
    return bool(model_availability(model_id).get("available"))


@_registry_locked
def pool_candidates(model_id: str) -> list[str]:
    """Ordered, locally-configured pool member providers for a model id.

    Accepts any routable id (name, alias, provider_model_id). Single-provider
    models return a one-element list. Unknown models return [].
    """
    entry = _load_models().get(model_id)
    if not entry:
        return []
    return _configured_pool_members(entry, _load_config())


@_registry_locked
def resolve(model_id: str, provider_override: str | None = None) -> ProviderInfo | None:
    """Resolve a model name/alias/id to provider info.

    Pooled models route to the first configured pool member whose circuit is
    not open (per-workspace failover happens here for new requests; in-flight
    failover lives in src.upstream). ``provider_override`` pins a specific
    pool member, bypassing circuit state — used by upstream failover.
    """
    models = _load_models()
    entry = models.get(model_id)
    availability = _availability_for_entry(model_id, entry)
    if not availability["available"]:
        if availability["reason"] == "model_not_found":
            return None
        log.info("Model %r unavailable: %s", model_id, availability["message"])
        return None

    try:
        composite = _parse_composite_route(entry)
    except ValueError:
        return None
    if composite is not None:
        text_info = resolve(composite.text_model, provider_override=provider_override)
        vision_info = resolve(composite.vision_model)
        if (
            not text_info
            or not vision_info
            or text_info.provider != "omlx"
            or vision_info.provider != "omlx"
            or not vision_info.vision
        ):
            return None
        return replace(text_info, vision=False, composite=composite)

    config = _load_config()
    candidates = _configured_pool_members(entry, config)
    if provider_override:
        provider = _canonical_provider(provider_override)
        if provider not in candidates:
            return None
    else:
        provider = candidates[0]
        if len(candidates) > 1:
            from src import circuit  # local import: circuit has no src imports
            for member in candidates:
                if not circuit.is_tripped(member):
                    provider = member
                    break
            if provider != candidates[0]:
                log.warning("pool: %r routing to %r (circuit open on %r)", model_id, provider, candidates[0])
    provider_config = _effective_provider_config(config, provider)

    base_url = provider_config.get("base_url", "")
    api_key = provider_config.get("api_key", "")
    protocol = provider_config.get("protocol", "openai")

    provider_model_id = entry.get("provider_model_id", "")
    if not provider_model_id:
        provider_model_id = entry.get("omlx_id", "")
    if not provider_model_id:
        provider_model_id = entry.get("name", "") or model_id

    # Per-model protocol override (e.g. an AI-gateway provider that serves both
    # Anthropic- and OpenAI-protocol models under one host).
    entry_protocol = entry.get("protocol")
    if entry_protocol:
        protocol = entry_protocol

    endpoint_suffix: str | None = None
    endpoint_style = provider_config.get("endpoint_style", "")
    if endpoint_style == "invocations":
        # base_url is a workspace host; each model has its own full invocation
        # URL: <base>/serving-endpoints/<provider_model_id>/invocations.
        base_url = base_url.rstrip("/") + f"/serving-endpoints/{provider_model_id}/invocations"
        endpoint_suffix = ""
    else:
        # Optional per-protocol path prefixes appended to base_url, e.g.
        # path_prefixes: {anthropic: /anthropic/v1, openai: /mlflow/v1}.
        path_prefixes = provider_config.get("path_prefixes") or {}
        prefix = path_prefixes.get(protocol) if isinstance(path_prefixes, dict) else None
        if prefix:
            base_url = base_url.rstrip("/") + "/" + str(prefix).strip("/")

    quirks_raw = provider_config.get("quirks") or []
    if isinstance(quirks_raw, str):
        quirks_raw = [q.strip() for q in quirks_raw.split(",") if q.strip()]
    quirks_set = set(quirks_raw)
    # Per-model quirks merge on top of provider quirks so a single model can
    # opt into a quirk (e.g. "no_reasoning_params" for gpt-5.6-sol which rejects
    # reasoning_effort + function tools on /v1/chat/completions) without
    # affecting other models routed through the same shared provider.
    model_quirks_raw = entry.get("quirks") or []
    if isinstance(model_quirks_raw, str):
        model_quirks_raw = [q.strip() for q in model_quirks_raw.split(",") if q.strip()]
    quirks_set.update(model_quirks_raw)

    return ProviderInfo(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        provider_model_id=provider_model_id,
        protocol=protocol,
        endpoint_suffix=endpoint_suffix,
        quirks=frozenset(quirks_set),
        context=entry.get("context", 0),
        max_output_tokens=entry.get("max_output_tokens", 0),
        thinking=entry.get("thinking", ""),
        thinking_levels=(tuple(entry["thinking_levels"]) if "thinking_levels" in entry else None),
        thinking_format=entry.get("thinking_format", ""),
        system_instruction=entry.get("system_instruction", ""),
        vision=bool(entry.get("vision", False)),
        pricing=entry.get("pricing"),
    )


def _pricing_status(entry: dict | None) -> str:
    """Classify one catalog entry as metered, unmetered, or unknown.

    Numeric rates take precedence over the marker so a malformed conflicting
    entry can never be treated as free. Catalog/admin validation prevents that
    conflict for newly written entries.
    """
    if not entry:
        return "unknown"
    pricing = entry.get("pricing")
    if isinstance(pricing, dict) and pricing:
        return "metered"
    if (
        entry.get("pricing_status") == "unmetered"
        and not entry.get("pool")
        and catalog.canonical_provider(entry.get("provider")) == "omlx"
    ):
        return "unmetered"
    return "unknown"


@_registry_locked
def pricing_for(model_id: str) -> dict | None:
    """Return the $/Mtok pricing dict for a routable model, or None if unset."""
    entry = _load_models().get(model_id)
    if not entry or _pricing_status(entry) != "metered":
        return None
    return entry["pricing"]


@_registry_locked
def pricing_status_for(model_id: str) -> str:
    """Return ``metered``, ``unmetered``, or ``unknown`` for a routable id."""
    return _pricing_status(_load_models().get(model_id))


@_registry_locked
def effective_model_inventory() -> list[dict]:
    """Return one effective runtime row per logical model.

    This is the shared discovery/admin inventory: merged catalog capabilities,
    runtime enablement, pool-aware availability, and the effective provider are
    computed once. Routing still accepts every identifier indexed by
    :func:`_load_models`.
    """
    result = []
    seen_entries: set[int] = set()
    for entry in _load_models().values():
        identity = id(entry)
        if identity in seen_entries:
            continue
        seen_entries.add(identity)
        row = dict(entry)
        name = row.get("name") or next(iter(_entry_routable_ids(row)), "")
        availability = model_availability(name)
        row["id"] = name
        row["routable_ids"] = _entry_routable_ids(row)
        row["enabled"] = _is_model_enabled(name)
        row["available"] = bool(availability.get("available"))
        row["availability_reason"] = availability.get("reason", "")
        row["availability_message"] = availability.get("message", "")
        row["effective_provider"] = availability.get("provider") or _pool_members(row, _load_config())[0]
        result.append(row)
    return result


@_registry_locked
def list_models() -> list[dict]:
    """Return all routable identifiers, preserving discovery compatibility."""
    result = []
    for model in effective_model_inventory():
        for model_id in model["routable_ids"]:
            row = dict(model)
            row["id"] = model_id
            result.append(row)
    return result


@_registry_locked
def list_available_models() -> list[dict]:
    """Return routable identifiers for enabled, locally usable models."""
    return [m for m in list_models() if m["available"]]


def _configured_provider_ids() -> set[str]:
    config = _load_config()
    providers = config.get("providers", {}) or {}
    configured = {_canonical_provider(key) for key in providers}
    for entry in {id(v): v for v in _load_models().values()}.values():
        provider = _canonical_provider(entry.get("provider", ""))
        if _provider_defaults(provider) or _provider_env_config(provider):
            configured.add(provider)
    return configured


def _safe_url(value: str) -> str:
    """Strip URL userinfo before exposing provider config in admin APIs."""
    if not value:
        return ""
    try:
        parts = urlsplit(value)
    except ValueError:
        return ""
    if not parts.netloc:
        return value
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    # Do not expose query/fragment: provider base URLs occasionally carry
    # auth-ish parameters and admin status only needs the service origin/path.
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


@_registry_locked
def provider_status() -> list[dict]:
    """Return masked provider configuration status for admin/observability APIs."""
    config = _load_config()
    configured = config.get("providers", {}) or {}
    models = effective_model_inventory()
    model_counts: dict[str, int] = {}
    for model in models:
        provider = _canonical_provider(model.get("effective_provider") or model.get("provider", ""))
        if model.get("enabled"):
            model_counts[provider] = model_counts.get(provider, 0) + 1

    provider_ids = sorted(set(model_counts) | {_canonical_provider(key) for key in configured})
    result = []
    for provider in provider_ids:
        explicit_config = _resolve_provider_config(config, provider)
        env_config = _provider_env_config(provider)
        provider_config = _effective_provider_config(config, provider)
        base_url = _safe_url(provider_config.get("base_url", ""))
        has_api_key = bool(provider_config.get("api_key"))
        model_count = model_counts.get(provider, 0)
        issues = []
        if model_count and provider_config.get("enabled") is False:
            issues.append("provider_disabled")
        if model_count and provider_config.get("enabled") is not False and not base_url:
            issues.append("missing_base_url")
        if model_count and provider_config.get("enabled") is not False and not has_api_key:
            issues.append("missing_api_key")
        result.append({
            "id": provider,
            "configured": bool(explicit_config or _provider_defaults(provider) or env_config),
            "enabled_models": model_count,
            "base_url": base_url,
            "protocol": provider_config.get("protocol", "openai") if provider_config else "openai",
            "has_api_key": has_api_key,
            "ready": not issues,
            "issues": issues,
        })
    return result


@_registry_locked
def model_status() -> list[dict]:
    """Return the shared effective inventory in the admin API shape."""
    ready = {p["id"]: p["ready"] for p in provider_status()}
    configured = _configured_provider_ids() | {"omlx"}
    result = []
    for model in effective_model_inventory():
        provider = _canonical_provider(model.get("effective_provider") or model.get("provider", ""))
        result.append({
            "id": model.get("id"),
            "name": model.get("name", ""),
            "alias": model.get("alias", ""),
            "routable_ids": model.get("routable_ids", []),
            "provider": provider,
            "configured_provider": _canonical_provider(model.get("provider", "")),
            "provider_model_id": model.get("provider_model_id") or model.get("omlx_id") or model.get("name", ""),
            "omlx_id": model.get("omlx_id", ""),
            "provider_configured": provider in configured,
            "provider_ready": ready.get(provider, False),
            "availability_reason": model.get("availability_reason", ""),
            "availability_message": model.get("availability_message", ""),
            "available": model.get("available", False),
            "context": model.get("context", 0),
            "max_output_tokens": model.get("max_output_tokens", 0),
            "thinking": model.get("thinking", ""),
            "thinking_levels": list(model.get("thinking_levels", [])),
            "thinking_format": model.get("thinking_format", ""),
            "vision": bool(model.get("vision", False)),
            "pricing": model.get("pricing"),
            "pricing_status": _pricing_status(model),
            "enabled": model.get("enabled", True),
        })
    return result


@_registry_locked
def config_validation() -> dict:
    """Validate current provider/model config without exposing secrets."""
    providers = provider_status()
    missing = [p for p in providers if p["enabled_models"] and p["issues"]]
    return {
        "ok": not missing,
        "model_info_path": str(MODEL_INFO_PATH),
        "config_path": str(CONFIG_PATH),
        "providers": providers,
        "issues": [
            {
                "provider": p["id"],
                "enabled_models": p["enabled_models"],
                "issues": p["issues"],
            }
            for p in missing
        ],
    }
