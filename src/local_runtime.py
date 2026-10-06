"""Gateway-owned local AI: a pinned oMLX runtime serving a verified MLX model.

One Model Gateway per Mac is shared by every product attached to it, so local
AI is installed here rather than by each product. ``start_add`` launches a
one-shot setup job (from ``local-ai/launchd``, never ``~/Library/LaunchAgents``,
so it does not rerun at login) that downloads the pinned payload (HTTPS only,
resumable, rate-limited, hash-verified), builds oMLX from the locked project
in ``local-runtime/`` with uv, installs the ``com.local.omlx`` LaunchAgent,
adds the ``omlx`` provider and model to this gateway, and requires a real
completion through the gateway (or, when /v1 needs a key the job does not
hold, a completion from oMLX while the gateway's config and catalog route the
model to it; see ``check``).

Ownership: the oMLX LaunchAgent and the setup job record
``<state directory>#<gateway launchd label>`` under ``ModelGatewayRoot``, so
a second gateway on the same Mac (another label) never treats this one's
runtime as its own. A ``com.local.omlx`` without this gateway's value (such as
a developer's own oMLX) is never modified: status reports ``managed: false``
and add and remove refuse. launchd jobs are stopped only when launchd loaded
them from this gateway's plist.

Every launchctl, sysctl, network, and subprocess edge is a module-level
function so tests replace them. Messages never include keys or response bodies.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import plistlib
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import yaml

from src import config_io
from src.catalog import canonical_provider
from src.config_lock import config_write_lock
from src.model_payload import VerificationError, read_json, validate_manifest, verify_payload
from src.secret_files import resolve_api_key_file


def _package_root() -> Path:
    """This install's root, unresolved, so a package's version-independent path survives upgrades.

    The CLI passes its root (for the component package, ``current``): Python
    imports this module through the resolved working directory, so ``__file__``
    names the release directory itself.
    """
    configured = os.environ.get("MODEL_GATEWAY_PACKAGE_ROOT", "").strip()
    if configured and os.path.isabs(configured):
        return Path(configured)
    resolved = Path(os.path.abspath(__file__)).parents[1]
    # The server runs from the resolved release too; a package's .package names
    # its version-independent ROOT, exactly as the CLI reads it.
    try:
        lines = (resolved / ".package").read_text(encoding="utf-8").splitlines()
    except OSError:
        return resolved
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip() == "ROOT" and os.path.isabs(value.strip()):
            return Path(value.strip())
    return resolved


PACKAGE_ROOT = _package_root()
MODELS_DIR = PACKAGE_ROOT / "local-models"
RUNTIME_PROJECT = PACKAGE_ROOT / "local-runtime"
OMLX_VERSION = "0.7.0"
OMLX_LABEL = "com.local.omlx"
SETUP_LABEL = "com.local.model-gateway.local-ai-setup"
GATEWAY_LABEL = "com.local.model-gateway"
OWNER_KEY = "ModelGatewayRoot"
DEFAULT_PORT = 9110
MIN_MEMORY_BYTES = 48 * 1024**3
# Beyond the remaining download: the oMLX environment plus the bounded 8 GB KV cache.
DISK_MARGIN = 12 * 1024**3
# The documented aggregate download budget for this network.
DOWNLOAD_BYTES_PER_SECOND = 25_000_000
DOWNLOAD_ATTEMPTS = 5
HEARTBEAT_SECONDS = 5
# An active state whose heartbeat is older than this, with no running job, was interrupted.
STALE_SECONDS = 60
# How long cancel and remove wait for a stopped setup job to exit.
JOB_STOP_SECONDS = 30
ACTIVE_STATES = frozenset({"queued", "downloading", "installing"})
METAL_SMOKE = ("import mlx.core as m; assert m.metal.is_available(); "
               "a=m.ones((64,64)); b=a@a; m.eval(b); assert b[0,0].item()==64")


class LocalRuntimeError(ValueError):
    """A refusal or failure whose message is safe to show."""


class Cancelled(Exception):
    """launchd stopped the setup job (cancel)."""


class _Busy(LocalRuntimeError):
    """Another process holds a local AI lock."""


def safe_message(exc: BaseException) -> str:
    """A message for operators and API clients: never config contents, keys, or bodies."""
    if isinstance(exc, LocalRuntimeError):
        return str(exc)
    if isinstance(exc, yaml.YAMLError):
        # YAML errors quote the offending line, which may hold a key.
        return "The gateway configuration is not valid YAML"
    if isinstance(exc, ValueError):
        return "The gateway configuration or local AI state is invalid"
    return str(exc)


@dataclass(frozen=True)
class LocalModel:
    name: str  # gateway model name
    manifest: str  # file under local-models/
    catalog: dict  # model-info.json fields besides provider/omlx_id
    settings: dict  # oMLX model_settings.json entry


MODELS = {
    "qwen3.8-27b": LocalModel(
        name="qwen3.8-27b",
        manifest="qwen3.8-27b-8bit.json",
        catalog={
            "context": 32768, "max_output_tokens": 4096, "thinking": "optional",
            "thinking_levels": ["off", "xhigh"], "thinking_format": "qwen-chat-template",
            "vision": True, "pricing_status": "unmetered",
            "desc": "Qwen3.8-27B dense 8-bit MLX VL, served by Model Gateway local AI",
        },
        settings={"max_context_window": 32768, "max_tokens": 4096,
                  "enable_thinking": True, "preserve_thinking": True},
    ),
}
DEFAULT_MODEL = "qwen3.8-27b"


# ── paths ────────────────────────────────────────────────────────────────────


def state_dir() -> Path:
    """The gateway's machine state directory (the CLI's APP_DIR)."""
    configured = os.environ.get("MODEL_GATEWAY_STATE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Library" / "Application Support" / "model-gateway"


def runtime_root() -> Path:
    return state_dir() / "local-ai"


def launch_agents_dir() -> Path:
    configured = os.environ.get("MODEL_GATEWAY_PLIST_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / "Library" / "LaunchAgents"


def omlx_plist_path() -> Path:
    return launch_agents_dir() / f"{OMLX_LABEL}.plist"


def setup_plist_path() -> Path:
    return runtime_root() / "launchd" / f"{SETUP_LABEL}.plist"


def status_path() -> Path:
    return runtime_root() / "status.json"


def venv_dir() -> Path:
    return runtime_root() / f"omlx-{OMLX_VERSION}"


def mlx_dir() -> Path:
    return runtime_root() / "models" / "mlx"


def cache_dir() -> Path:
    return runtime_root() / "models" / "cache"


def inference_dir() -> Path:
    """oMLX --base-path: its settings, model settings, and API key."""
    return runtime_root() / "inference"


def api_key_path() -> Path:
    return inference_dir() / "api.key"


def owner_value() -> str:
    """``ModelGatewayRoot``: identifies this gateway among any others on the Mac."""
    return f"{state_dir()}#{gateway_label()}"


def gateway_label() -> str:
    return os.environ.get("MODEL_GATEWAY_LAUNCHD_LABEL", "").strip() or GATEWAY_LABEL


def _require_known_gateway() -> None:
    """Refuse to guess the label of a gateway whose LaunchAgent predates recording it.

    The setup job restarts the gateway by label, and the label is part of the
    ownership value, so without ``MODEL_GATEWAY_LAUNCHD_LABEL`` the default
    LaunchAgent must be the one running this gateway's config.
    """
    if os.environ.get("MODEL_GATEWAY_LAUNCHD_LABEL", "").strip():
        return
    environment = (_plist(launch_agents_dir() / f"{GATEWAY_LABEL}.plist") or {}).get("EnvironmentVariables")
    configured = environment.get("MODEL_GATEWAY_CONFIG") if isinstance(environment, dict) else None
    if not isinstance(configured, str) or (
            os.path.realpath(os.path.expanduser(configured)) != os.path.realpath(config_io.CONFIG_PATH)):
        raise LocalRuntimeError(
            "This gateway's LaunchAgent label is unknown; run 'model-gateway install' to record it, then retry")


def gateway_origin() -> str:
    from src.discovery import _connect_origin  # discovery imports this module

    return _connect_origin()[0]


# ── private files ────────────────────────────────────────────────────────────


def _no_symlinks(path: Path, base: Path) -> None:
    """Refuse a symlink at ``path`` or any directory up to and including ``base``."""
    for part in (path, *path.parents):
        if part.is_symlink():
            raise LocalRuntimeError(f"Symlink is not allowed: {part}")
        if part == base:
            return


def _regular(path: Path, base: Path | None = None) -> None:
    _no_symlinks(path, base or runtime_root())
    if path.exists() and (not path.is_file() or path.stat().st_uid != os.getuid()):
        raise LocalRuntimeError(f"Expected a user-owned regular file: {path}")


def _atomic(path: Path, data: str | bytes, base: Path | None = None) -> None:
    """Write a private (0600) file atomically; unchanged content is left in place."""
    _regular(path, base)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = data.encode() if isinstance(data, str) else data
    if path.exists() and path.read_bytes() == raw:
        os.chmod(path, 0o600)
        return
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _create_new(path: Path, data: bytes) -> None:
    """Write a new private file; never replace anything that appeared at ``path``."""
    _no_symlinks(path.parent, path.parent)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # Unlike rename, link fails with EEXIST instead of replacing the target.
        os.link(name, path)
    except FileExistsError:
        raise LocalRuntimeError(f"{path} appeared during setup; Model Gateway will not replace it") from None
    finally:
        Path(name).unlink(missing_ok=True)


def _key(path: Path, *, create: bool = False) -> str:
    _regular(path)
    if create and not path.exists():
        _atomic(path, secrets.token_urlsafe(32) + "\n")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise LocalRuntimeError(f"Key must have mode 0600: {path}")
    value = path.read_text().strip()
    if not value or len(value) > 4096 or any(char.isspace() for char in value):
        raise LocalRuntimeError(f"Invalid key file: {path}")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── edges (replaced in tests) ────────────────────────────────────────────────


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/launchctl", *args], capture_output=True, text=True)


def _memory_bytes() -> int:
    # Absolute: launchd jobs and the gateway run without /usr/sbin on PATH.
    return int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True))


def _apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _run(*args: str, env: dict | None = None) -> None:
    subprocess.run(args, check=True, env=env)


def _disk_free(path: Path) -> int:
    return shutil.disk_usage(path).free


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """The model host redirects to its CDN; never follow a downgrade."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise urllib.error.HTTPError(newurl, code, "Refusing a non-HTTPS redirect", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), HttpsOnlyRedirect)


def _request(url: str, token: str = "", body: dict | None = None, *, timeout: int = 15) -> dict:
    """JSON request to a loopback service: no proxy, no redirects."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


# ── launchd ──────────────────────────────────────────────────────────────────


def _target(label: str) -> str:
    return f"gui/{os.getuid()}/{label}"


def _loaded(label: str) -> bool:
    return _launchctl("print", _target(label)).returncode == 0


def _job_running() -> bool:
    result = _launchctl("print", _target(SETUP_LABEL))
    return result.returncode == 0 and "state = running" in (result.stdout or "")


def _loaded_path(output: str) -> str | None:
    """The plist path in ``launchctl print`` output."""
    for line in output.splitlines():
        name, _, value = line.strip().partition(" = ")
        if name == "path" and value:
            return value
    return None


def _bootout(label: str, plist: Path) -> None:
    """Stop ``label`` only when launchd loaded it from ``plist``."""
    result = _launchctl("print", _target(label))
    if result.returncode != 0:
        return
    loaded = _loaded_path(result.stdout or "")
    if loaded is None or os.path.realpath(loaded) != os.path.realpath(plist):
        raise LocalRuntimeError(
            f"{label} is loaded from {loaded or 'an unknown plist'}, not {plist}; Model Gateway will not stop it")
    if _launchctl("bootout", _target(label)).returncode != 0:
        raise LocalRuntimeError(f"Could not stop {label}")


def _bootstrap(plist: Path, label: str) -> None:
    _bootout(label, plist)
    # bootout returns before the job releases its launchd slot; bootstrap then
    # reports EIO (5) for a moment.
    for attempt in range(30):
        result = _launchctl("bootstrap", f"gui/{os.getuid()}", str(plist))
        if result.returncode == 0:
            return
        if result.returncode != 5 or attempt == 29:
            raise LocalRuntimeError(f"{label} bootstrap failed (exit {result.returncode})")
        time.sleep(1)


def _plist(path: Path) -> dict | None:
    try:
        value = plistlib.loads(path.read_bytes())
    except (OSError, ValueError, plistlib.InvalidFileException):
        return None
    return value if isinstance(value, dict) else None


def omlx_owner() -> str | None:
    """``gateway`` when this gateway owns com.local.omlx, ``external`` for anyone else, None if absent."""
    path = omlx_plist_path()
    if path.is_symlink() or (path.exists() and not path.is_file()):
        return "external"
    if path.exists():
        value = _plist(path)
        return "gateway" if value is not None and value.get(OWNER_KEY) == owner_value() else "external"
    # Loaded from somewhere else entirely.
    return "external" if _loaded(OMLX_LABEL) else None


def _require_setup_owned() -> None:
    """Refuse a setup job that another gateway sharing this state directory queued."""
    path = setup_plist_path()
    if path.exists() and (_plist(path) or {}).get(OWNER_KEY) != owner_value():
        raise LocalRuntimeError(f"Local AI setup in {runtime_root()} belongs to another Model Gateway")


def _require_owned() -> None:
    if omlx_owner() == "external":
        raise LocalRuntimeError(
            f"{OMLX_LABEL} is managed outside Model Gateway ({omlx_plist_path()}); "
            "Model Gateway will not modify it")
    _require_setup_owned()


def _configured_port() -> int:
    raw = os.environ.get("MODEL_GATEWAY_LOCAL_AI_PORT", "").strip() or str(DEFAULT_PORT)
    if not raw.isdigit() or not 1024 <= int(raw) <= 65535:
        raise LocalRuntimeError("MODEL_GATEWAY_LOCAL_AI_PORT must be an integer between 1024 and 65535")
    return int(raw)


def _installed_port() -> int | None:
    """The port in this gateway's own oMLX LaunchAgent, if it has one."""
    if omlx_owner() != "gateway":
        return None
    args = (_plist(omlx_plist_path()) or {}).get("ProgramArguments") or []
    with contextlib.suppress(ValueError, IndexError):
        return int(args[args.index("--port") + 1])
    return None


def port() -> int:
    return _installed_port() or _configured_port()


# ── models ───────────────────────────────────────────────────────────────────


def model(name: str | None) -> LocalModel:
    found = MODELS.get(name or DEFAULT_MODEL)
    if found is None:
        raise LocalRuntimeError(f"Unknown local model {name!r}; available: {', '.join(sorted(MODELS))}")
    return found


def manifest(local: LocalModel) -> dict:
    try:
        # The strict reader refuses symlinked ancestors; a package's root is the `current` link.
        value = read_json(Path(os.path.realpath(MODELS_DIR / local.manifest)), "model manifest")
        validate_manifest(value)
    except VerificationError as exc:
        raise LocalRuntimeError(f"Invalid model manifest for {local.name}: {exc}") from None
    return value


def installed_model() -> LocalModel | None:
    """The model whose verified payload this gateway's runtime serves."""
    if omlx_owner() != "gateway":
        return None
    for local in MODELS.values():
        if (mlx_dir() / manifest(local)["model_id"]).is_dir():
            return local
    return None


def eligible() -> bool:
    """Local AI needs Apple silicon with at least 48 GiB of memory."""
    try:
        return _apple_silicon() and _memory_bytes() >= MIN_MEMORY_BYTES
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


# ── status ───────────────────────────────────────────────────────────────────


def read_status() -> dict:
    path = status_path()
    _regular(path)
    try:
        value = json.loads(path.read_text()) if path.exists() else {}
    except (ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_status(value: dict) -> None:
    _atomic(status_path(), json.dumps(value) + "\n")


def _stale(value: dict) -> bool:
    try:
        updated = datetime.fromisoformat(str(value.get("updated_at")))
    except ValueError:
        return True
    return (datetime.now(timezone.utc) - updated).total_seconds() > STALE_SECONDS


def record_failure(message: str) -> None:
    value = read_status()
    value.update(schema_version=1, state="failed", message=message[:300], updated_at=_now())
    _write_status(value)


def status() -> dict:
    """Eligibility, ownership, and setup progress; never keys."""
    owner = omlx_owner()
    current = read_status()
    installed = installed_model()
    state = current.get("state")
    if state in ACTIVE_STATES and _stale(current) and not _job_running():
        # e.g. the Mac restarted mid-download; adding again resumes it.
        state = "interrupted"
    if state is None and installed is not None:
        state = "done"
    message = str(current.get("message") or "")
    if owner == "external":
        message = f"{OMLX_LABEL} is managed outside Model Gateway"
    return {
        "eligible": eligible(),
        "installed": installed is not None,
        "managed": owner != "external",
        "model": installed.name if installed else current.get("model"),
        "state": state,
        "bytes_done": int(current.get("bytes_done") or 0),
        "bytes_total": int(current.get("bytes_total") or 0),
        "message": message,
        "base_url": f"http://127.0.0.1:{port()}" if installed else None,
        "loaded": _model_loaded(installed) if installed is not None and owner != "external" else None,
    }


def _model_loaded(local: LocalModel) -> bool | None:
    """Whether oMLX has the installed model in memory, so products can say it must load.

    Unknown (None) whenever the runtime cannot be asked; never claims a reload.
    """
    try:
        model_id = manifest(local)["model_id"]
        inventory = _request(f"http://127.0.0.1:{port()}/v1/models/status", _key(api_key_path()), timeout=3)
    except Exception:  # noqa: BLE001 — status must never fail on a busy or stopped runtime
        return None
    for row in inventory.get("models", []) if isinstance(inventory, dict) else []:
        if isinstance(row, dict) and row.get("id") == model_id and isinstance(row.get("loaded"), bool):
            return row["loaded"]
    return None


def endpoint_info() -> dict:
    """The endpoint.json ``local_runtime`` block: URLs only, never keys."""
    try:
        owner = omlx_owner()
        base_url = f"http://127.0.0.1:{port()}" if installed_model() is not None else None
    except Exception:  # noqa: BLE001 — discovery must never fail on local AI state
        return {"managed": False, "base_url": None, "health_url": None}
    return {"managed": owner != "external", "base_url": base_url,
            "health_url": f"{base_url}/health" if base_url else None}


class Progress:
    """Writes job state atomically, with a heartbeat so readers can detect a dead job."""

    def __init__(self, local: LocalModel, total: int):
        self.lock = threading.Lock()
        self.value = {"schema_version": 1, "state": "downloading", "model": local.name, "bytes_done": 0,
                      "bytes_total": total, "message": "", "started_at": _now()}
        self.stop = threading.Event()
        self.heartbeat = threading.Thread(target=self._beat, daemon=True)

    def update(self, **fields) -> None:
        with self.lock:
            self.value.update(fields, updated_at=_now())
            # After remove, never recreate the state it deleted.
            if runtime_root().is_dir():
                _write_status(self.value)

    def _beat(self) -> None:
        while not self.stop.wait(HEARTBEAT_SECONDS):
            self.update()

    def __enter__(self):
        self.update()
        self.heartbeat.start()
        return self

    def close(self) -> None:
        self.stop.set()
        self.heartbeat.join()

    def __exit__(self, *exc) -> None:
        self.close()


@contextlib.contextmanager
def _lock(name: str, timeout: float | None, busy: str):
    """Hold an exclusive flock on ``local-ai/<name>``, waiting up to ``timeout`` seconds (None: forever)."""
    path = runtime_root() / name
    _regular(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if timeout is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise _Busy(busy) from None
                    time.sleep(0.2)
        yield
    finally:
        os.close(fd)


def provision_lock(*, wait: bool, busy: str = "Local AI setup is installing"):
    """Serialize installing and removing between the setup job, CLI, and admin API."""
    return _lock(".provision.lock", None if wait else 0, busy)


def job_lock(timeout: float, busy: str = "Local AI setup is still stopping; retry in a moment"):
    """Held by the setup job for its whole run, so its holder knows no job is writing."""
    return _lock(".job.lock", timeout, busy)


def _job_alive() -> bool:
    try:
        with job_lock(0):
            return False
    except _Busy:
        return True


# ── gateway configuration ────────────────────────────────────────────────────


def _our_provider(block: object) -> bool:
    if not isinstance(block, dict) or not isinstance(block.get("api_key_file"), str):
        return False
    configured = resolve_api_key_file(block["api_key_file"], config_io.CONFIG_PATH)
    return configured == Path(os.path.realpath(api_key_path()))


def _our_catalog_entry(entry: object, local: LocalModel) -> bool:
    return (isinstance(entry, dict) and entry.get("name") == local.name
            and canonical_provider(entry.get("provider")) == "omlx"
            and entry.get("omlx_id") == manifest(local)["model_id"])


def gateway_conflict(local: LocalModel) -> str | None:
    """Why this gateway's config cannot take the local runtime, if it cannot."""
    config = config_io.load_config_full()
    configured = config.get("providers") or {}
    if not isinstance(configured, dict):
        return "The gateway providers configuration is not a mapping"
    for provider_id, block in configured.items():
        if canonical_provider(str(provider_id)) == "omlx" and not _our_provider(block):
            return (f"The gateway already has an oMLX provider ({provider_id!r}) that Model Gateway "
                    "local AI does not manage; remove it first")
    for entry in config_io.load_model_info().get("llm", []):
        if isinstance(entry, dict) and entry.get("name") == local.name and not _our_catalog_entry(entry, local):
            return f"The gateway catalog already has a different model named {local.name!r}"
    return None


def _restart_gateway(*, required: bool) -> None:
    """Restart the gateway LaunchAgent so it loads the new config, like the CLI does."""
    label = gateway_label()
    if not _loaded(label):
        if required:
            raise LocalRuntimeError(f"Model Gateway ({label}) is not running; start it and retry")
        return
    if _launchctl("kickstart", "-k", _target(label)).returncode != 0:
        raise LocalRuntimeError(f"Could not restart Model Gateway ({label})")
    _wait_health(f"{gateway_origin()}/health")


def _write_config(config: dict) -> None:
    config_io._backup(config_io.CONFIG_PATH)
    config_io._atomic_write(config_io.CONFIG_PATH,
                            yaml.safe_dump(config, sort_keys=False, default_flow_style=False))


def configure_gateway(local: LocalModel, omlx_port: int) -> None:
    """Add the omlx provider and model, restart the gateway, roll back on failure."""
    with config_write_lock(config_io.CONFIG_PATH):
        conflict = gateway_conflict(local)
        if conflict:
            raise LocalRuntimeError(conflict)
        snapshot = config_io.snapshot_writable_files()
        try:
            config = config_io.load_config_full()
            config.setdefault("providers", {})
            if config["providers"] is None:
                config["providers"] = {}
            # An empty api_key overrides the built-in omlx default so api_key_file applies;
            # managed_by makes admin provider key edits and deletes refuse this block.
            config["providers"]["omlx"] = {
                "base_url": f"http://127.0.0.1:{omlx_port}/v1", "protocol": "openai", "api_key": "",
                "api_key_file": str(api_key_path()), "enabled": True,
                "managed_by": config_io.LOCAL_AI_MANAGED,
            }
            _write_config(config)
            config_io.upsert_model(local.name, provider="omlx", omlx_id=manifest(local)["model_id"],
                                   **local.catalog)
            _restart_gateway(required=True)
        except BaseException:
            config_io.restore_writable_files(snapshot)
            with contextlib.suppress(Exception):
                _restart_gateway(required=False)
            raise


def unconfigure_gateway() -> bool:
    """Remove only this runtime's provider and models from the files; return whether anything changed.

    The running gateway keeps its loaded config until the caller restarts it.
    """
    with config_write_lock(config_io.CONFIG_PATH):
        snapshot = config_io.snapshot_writable_files()
        changed = False
        try:
            config = config_io.load_config_full()
            configured = config.get("providers")
            configured = configured if isinstance(configured, dict) else {}
            # With someone else's oMLX provider, a matching catalog entry routes to it, so it is theirs.
            foreign = any(canonical_provider(str(provider_id)) == "omlx" and not _our_provider(block)
                          for provider_id, block in configured.items())
            for local in () if foreign else MODELS.values():
                entries = config_io.load_model_info().get("llm", [])
                if any(_our_catalog_entry(entry, local) for entry in entries):
                    config_io.delete_model(local.name)
                    changed = True
            if _our_provider(configured.get("omlx")):
                del configured["omlx"]
                _write_config(config)
                changed = True
        except BaseException:
            config_io.restore_writable_files(snapshot)
            raise
        return changed


# ── download ─────────────────────────────────────────────────────────────────


def download_file(url: str, path: Path, entry: dict, progress: Progress, done: int, opener) -> None:
    """Resume one file to its exact size, hashing as it goes.

    A dropped connection keeps the partial file for the next attempt; only a
    complete file with the wrong hash, or oversized content, is removed.
    """
    size, digest = entry["size_bytes"], hashlib.sha256()
    have = path.stat().st_size if path.exists() else 0
    if have > size:
        path.unlink()
        have = 0
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600), "ab") as out:
        if have:
            with open(path, "rb") as existing:
                while chunk := existing.read(1024 * 1024):
                    digest.update(chunk)
            progress.value["bytes_done"] = done + have
        if have < size:
            request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with opener.open(request, timeout=60) as response:
                if have and response.status == 200:
                    # The host ignored the range: start this file over.
                    out.truncate(0)
                    digest, have = hashlib.sha256(), 0
                elif response.status != (206 if have else 200) or (
                        have and not str(response.headers.get("Content-Range", "")).startswith(f"bytes {have}-")):
                    raise RuntimeError("Model host returned an unexpected response")
                started, received = time.monotonic(), 0
                while chunk := response.read(1024 * 1024):
                    if have + received + len(chunk) > size:
                        out.flush()
                        path.unlink()
                        raise RuntimeError("Model host sent more data than expected")
                    out.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    progress.value["bytes_done"] = done + have + received
                    # Stay within the aggregate download budget.
                    ahead = received / DOWNLOAD_BYTES_PER_SECOND - (time.monotonic() - started)
                    if ahead > 0:
                        time.sleep(ahead)
            out.flush()
            os.fsync(out.fileno())
    if path.stat().st_size < size:
        raise RuntimeError("Model host closed the connection early")
    if digest.hexdigest() != entry["sha256"].lower():
        path.unlink()
        raise RuntimeError(f"Downloaded {entry['path']} did not match the release manifest")


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def download_payload(payload: dict, progress: Progress) -> None:
    """Fetch the pinned payload into the .partial directory that install verifies and promotes."""
    destination = mlx_dir() / payload["model_id"]
    if destination.exists():
        progress.update(bytes_done=payload["total_bytes"])
        return
    partial = destination.with_name(payload["model_id"] + ".partial")
    _no_symlinks(partial, runtime_root())
    partial.mkdir(parents=True, exist_ok=True, mode=0o700)
    opener = _download_opener()
    base = f"https://huggingface.co/{payload['hf_repo']}/resolve/{payload['hf_revision']}/"
    done = 0
    for entry in payload["files"]:
        path = partial / entry["path"]
        _regular(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        failures, size = 0, _size(path)
        while True:
            try:
                download_file(base + urllib.parse.quote(entry["path"]), path, entry, progress, done, opener)
                break
            except (OSError, RuntimeError, http.client.HTTPException) as exc:
                # Only consecutive attempts that added nothing count toward giving up.
                if (current := _size(path)) > size:
                    failures, size = 0, current
                failures += 1
                if failures == DOWNLOAD_ATTEMPTS:
                    raise RuntimeError(f"Could not download the local AI model ({exc})") from None
                time.sleep(min(60, 5 * 2 ** (failures - 1)))
        done += entry["size_bytes"]
        progress.update(bytes_done=done)


# ── install ──────────────────────────────────────────────────────────────────


def omlx_plist(omlx_port: int) -> dict:
    root = runtime_root()
    return {
        "Label": OMLX_LABEL, OWNER_KEY: owner_value(),
        "ProgramArguments": [str(venv_dir() / "bin" / "omlx"), "serve",
                             "--base-path", str(inference_dir()), "--model-dir", str(mlx_dir()),
                             "--host", "127.0.0.1", "--port", str(omlx_port), "--no-hf-cache",
                             "--max-concurrent-requests", "1", "--memory-guard", "safe",
                             "--paged-ssd-cache-dir", str(cache_dir()),
                             "--paged-ssd-cache-max-size", "8GB"],
        "WorkingDirectory": str(inference_dir()),
        "EnvironmentVariables": {
            "PATH": f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "StandardOutPath": str(root / "logs" / "omlx.log"),
        "StandardErrorPath": str(root / "logs" / "omlx.log"),
        "Umask": 0o77, "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 10,
    }


def _job_environment(omlx_port: int) -> dict:
    """What the setup job needs to address this exact gateway."""
    env = {
        "PATH": f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "MODEL_GATEWAY_STATE_DIR": str(state_dir()),
        "MODEL_GATEWAY_CONFIG": str(config_io.CONFIG_PATH),
        "MODEL_GATEWAY_MODEL_INFO": str(config_io.MODEL_INFO_PATH),
        "MODEL_GATEWAY_PLIST_DIR": str(launch_agents_dir()),
        "MODEL_GATEWAY_LAUNCHD_LABEL": gateway_label(),
        "MODEL_GATEWAY_LOCAL_AI_PORT": str(omlx_port),
        "MODEL_GATEWAY_PACKAGE_ROOT": str(PACKAGE_ROOT),
        "MODEL_GATEWAY_HOST": os.environ.get("MODEL_GATEWAY_HOST", "").strip() or "127.0.0.1",
        "MODEL_GATEWAY_PORT": os.environ.get("MODEL_GATEWAY_PORT", "").strip() or "9111",
    }
    if config_io.MODEL_INFO_SOURCE_PATH:
        env["MODEL_GATEWAY_MODEL_INFO_SOURCE"] = str(config_io.MODEL_INFO_SOURCE_PATH)
    for name in ("MODEL_GATEWAY_BACKUP_DIR", "MODEL_GATEWAY_UV_BIN"):
        if os.environ.get(name, "").strip():
            env[name] = os.environ[name].strip()
    return env


def _job_command(name: str) -> list[str]:
    return ["/bin/bash", str(PACKAGE_ROOT / "bin" / "model-gateway"), "_local-ai-job", name]


def setup_plist(local: LocalModel, omlx_port: int) -> dict:
    root = runtime_root()
    return {
        "Label": SETUP_LABEL, OWNER_KEY: owner_value(),
        "ProgramArguments": _job_command(local.name),
        "WorkingDirectory": str(root),
        "EnvironmentVariables": _job_environment(omlx_port),
        "StandardOutPath": str(root / "logs" / "setup.log"),
        "StandardErrorPath": str(root / "logs" / "setup.log"),
        "Umask": 0o77, "RunAtLoad": True, "ProcessType": "Background",
    }


def _uv() -> str:
    return (os.environ.get("MODEL_GATEWAY_UV_BIN", "").strip() or shutil.which("uv")
            or "/opt/homebrew/bin/uv")


def _wait_health(url: str, seconds: int = 90) -> None:
    deadline = time.monotonic() + seconds
    while True:
        try:
            if _request(url, timeout=3).get("status") in {"ok", "healthy"}:
                return
        except (OSError, ValueError, http.client.HTTPException):
            pass
        if time.monotonic() >= deadline:
            raise LocalRuntimeError(f"Service did not become healthy: {url}")
        time.sleep(2)


def install(local: LocalModel, payload: dict) -> None:
    """Verify and promote the payload, build oMLX, start it, and wire it into the gateway."""
    _require_owned()
    model_id = payload["model_id"]
    destination = mlx_dir() / model_id
    partial = destination.with_name(model_id + ".partial")
    staged = destination if destination.exists() else partial
    _no_symlinks(staged, runtime_root())
    try:
        verify_payload(staged, payload)
    except VerificationError as exc:
        raise LocalRuntimeError(f"The local model failed verification: {exc}") from None
    if staged == partial:
        partial.rename(destination)
    # oMLX serves every model in --model-dir; never a forgotten or foreign one.
    if {path.name for path in mlx_dir().iterdir()} != {model_id}:
        raise LocalRuntimeError(f"{mlx_dir()} must contain only the verified {model_id} payload")
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(venv_dir()))
    _run(_uv(), "sync", "--locked", "--no-dev", "--project", str(RUNTIME_PROJECT), env=env)
    _run(str(venv_dir() / "bin" / "python"), "-c", METAL_SMOKE)
    api_key = _key(api_key_path(), create=True)
    _atomic(inference_dir() / "settings.json", json.dumps({
        "auth": {"api_key": api_key}, "logging": {"log_dir": str(runtime_root() / "logs" / "omlx")}},
        indent=2) + "\n")
    _atomic(inference_dir() / "model_settings.json", json.dumps({
        "version": 1, "models": {model_id: {"is_default": True, **local.settings}}}, indent=2) + "\n")
    cache_dir().mkdir(parents=True, exist_ok=True, mode=0o700)
    (runtime_root() / "logs").mkdir(parents=True, exist_ok=True, mode=0o700)
    omlx_port = port()
    # Again: a com.local.omlx may have appeared while this job downloaded and built.
    owner = omlx_owner()
    _require_owned()
    plist = omlx_plist_path()
    if owner is None:
        _create_new(plist, plistlib.dumps(omlx_plist(omlx_port)))
    else:
        _atomic(plist, plistlib.dumps(omlx_plist(omlx_port)), base=plist.parent)
    _bootstrap(plist, OMLX_LABEL)
    _wait_health(f"http://127.0.0.1:{omlx_port}/health")
    configure_gateway(local, omlx_port)


def _gateway_token() -> str:
    """The gateway admin key, which /v1 accepts, when this process can read one."""
    from src import auth

    keys = sorted(auth._admin_keys())
    return keys[0] if keys else ""


def _anonymous_refused(gateway: str) -> bool:
    """Whether the gateway's /v1 requires a client key this process does not hold."""
    try:
        _request(f"{gateway}/v1/models")
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return True
        raise
    return False


def _check_direct(local: LocalModel, payload: dict) -> dict:
    """Without a gateway key: complete against oMLX and require the gateway's config to route to it.

    Package installs seed no admin key, and once a product adds consumer
    credentials /v1 refuses anonymous clients. Rather than mint a standing
    credential (which would also lock /v1 for every anonymous local client),
    the job proves the model with oMLX's own key and checks that the config and
    catalog the gateway was restarted on enable this job's managed ``omlx``
    provider and route the model to it.
    """
    omlx_port = port()
    provider = (config_io.load_config_full().get("providers") or {}).get("omlx")
    if (not _our_provider(provider) or provider.get("enabled") is not True
            or provider.get("managed_by") != config_io.LOCAL_AI_MANAGED
            or provider.get("base_url") != f"http://127.0.0.1:{omlx_port}/v1"
            or not any(_our_catalog_entry(entry, local) for entry in config_io.load_model_info().get("llm", []))):
        raise LocalRuntimeError(f"The gateway does not route {local.name} to local AI")
    return _request(f"http://127.0.0.1:{omlx_port}/v1/chat/completions", _key(api_key_path()), {
        "model": payload["model_id"], "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}, "stream": False}, timeout=300)


def check(local: LocalModel, payload: dict) -> None:
    """oMLX serves exactly the verified payload and the gateway completes with it."""
    model_id = payload["model_id"]
    inventory = _request(f"http://127.0.0.1:{port()}/v1/models/status", _key(api_key_path()))
    rows = [row for row in inventory.get("models", []) if isinstance(row, dict) and row.get("source_type") != "builtin"]
    if (len(rows) != 1 or rows[0].get("id") != model_id
            or os.path.realpath(str(rows[0].get("model_path"))) != os.path.realpath(mlx_dir() / model_id)):
        raise LocalRuntimeError("oMLX is not serving the sole verified local model")
    gateway, token = gateway_origin(), _gateway_token()
    if not token and _anonymous_refused(gateway):
        _completion_ok(_check_direct(local, payload))
        return
    deadline = time.monotonic() + 60
    while True:
        listing = _request(f"{gateway}/v1/models", token)
        if local.name in {row.get("id") for row in listing.get("data", []) if isinstance(row, dict)}:
            break
        if time.monotonic() >= deadline:
            raise LocalRuntimeError(f"The gateway does not list {local.name}")
        time.sleep(2)
    completion = _request(f"{gateway}/v1/chat/completions", token, {
        "model": local.name, "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 32, "reasoning_effort": "off", "stream": False}, timeout=300)
    _completion_ok(completion)


def _completion_ok(completion: dict) -> None:
    try:
        content = completion["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = None
    if not isinstance(content, str) or content.strip().rstrip(".") != "OK":
        raise LocalRuntimeError("The local model did not pass the completion check")


# ── operations ───────────────────────────────────────────────────────────────


def start_add(name: str | None = None) -> dict:
    """Queue the setup job; reuse a running one. Partial downloads resume."""
    local = model(name)
    _require_owned()
    installed = installed_model()
    if installed is not None and installed != local:
        raise LocalRuntimeError(f"Local AI already serves {installed.name}; remove it first")
    current = status()
    if current["installed"] and current["state"] == "done":
        return current
    if not eligible():
        raise LocalRuntimeError("This Mac needs Apple silicon and at least 48 GB of memory for local AI")
    _require_known_gateway()
    root = runtime_root()
    _no_symlinks(root, root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(_lock(".start.lock", 0, ""))
        except _Busy:
            return status()  # another add is starting the job right now
        if _job_running() or _job_alive():
            return status()
        conflict = gateway_conflict(local)
        if conflict:
            raise LocalRuntimeError(conflict)
        omlx_port = port()
        if omlx_owner() is None and _port_in_use(omlx_port):
            raise LocalRuntimeError(f"Port {omlx_port} is in use; set MODEL_GATEWAY_LOCAL_AI_PORT to a free port")
        payload = manifest(local)
        partial = mlx_dir() / (payload["model_id"] + ".partial")
        staged = sum((partial / e["path"]).stat().st_size for e in payload["files"]
                     if (partial / e["path"]).is_file() and (partial / e["path"]).stat().st_size <= e["size_bytes"])
        remaining = 0 if (mlx_dir() / payload["model_id"]).exists() else payload["total_bytes"] - staged
        needed = remaining + DISK_MARGIN
        if _disk_free(root) < needed:
            raise LocalRuntimeError(f"Local AI needs {needed / 1e9:.0f} GB of free disk space")
        value = {"schema_version": 1, "state": "queued", "model": local.name, "bytes_done": 0,
                 "bytes_total": payload["total_bytes"], "message": "", "started_at": _now(), "updated_at": _now()}
        _write_status(value)
        try:
            _atomic(setup_plist_path(), plistlib.dumps(setup_plist(local, omlx_port)))
            (root / "logs").mkdir(parents=True, exist_ok=True, mode=0o700)
            _bootstrap(setup_plist_path(), SETUP_LABEL)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            record_failure(safe_message(exc))
            raise
    return status()


def cancel() -> dict:
    """Stop a download; the partial payload stays so a later add resumes it."""
    if not runtime_root().exists():
        return status()
    _require_setup_owned()
    # The job holds the provision lock while it installs; stopping it then would leave a half-configured install.
    with provision_lock(wait=False, busy="Local AI is being installed and can no longer be cancelled"):
        _bootout(SETUP_LABEL, setup_plist_path())
        with job_lock(JOB_STOP_SECONDS):
            current = read_status()
            if current.get("state") in ACTIVE_STATES:
                current.update(state="cancelled", message="", updated_at=_now())
                _write_status(current)
    return status()


def _teardown() -> bool:
    """Unwire, stop, and delete this runtime; return whether the gateway must restart."""
    # Edit the gateway's files first so nothing routes to the runtime once it restarts.
    changed = unconfigure_gateway()
    if omlx_owner() == "gateway":
        _bootout(OMLX_LABEL, omlx_plist_path())
        omlx_plist_path().unlink()
        changed = True
    shutil.rmtree(runtime_root())
    return changed


def remove() -> dict:
    """Remove this gateway's runtime, its provider and model, and all local AI state.

    The gateway restarts once, after teardown, so it and endpoint.json stop
    advertising the runtime. Without a loaded gateway (uninstalled), only the
    files change.
    """
    _require_owned()
    root = runtime_root()
    _no_symlinks(root, root)
    tearing_down = False
    try:
        with provision_lock(wait=False, busy="Local AI is being installed; wait for it to finish, then remove it"):
            _bootout(SETUP_LABEL, setup_plist_path())
            with job_lock(JOB_STOP_SECONDS):
                tearing_down = True
                restart = _teardown()
    except BaseException:
        if tearing_down:
            # Teardown may have unwired the config before failing; the gateway should match it.
            with contextlib.suppress(Exception):
                _restart_gateway(required=False)
        raise
    if restart:
        _restart_gateway(required=False)
    return status()


def _source_digest() -> str:
    """The installed code and data the job runs, to notice an upgrade mid-download."""
    digest = hashlib.sha256()
    paths = [*PACKAGE_ROOT.glob("src/**/*.py"), PACKAGE_ROOT / "bin" / "model-gateway",
             *MODELS_DIR.glob("*.json"), RUNTIME_PROJECT / "pyproject.toml", RUNTIME_PROJECT / "uv.lock"]
    for path in sorted(paths):
        digest.update(path.relative_to(PACKAGE_ROOT).as_posix().encode() + b"\0")
        with contextlib.suppress(OSError):
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _reexec(name: str) -> None:
    """Replace this job with the installed release's job; the job lock is released on exec."""
    command = _job_command(name)
    os.execv(command[0], command)


def run_job(name: str) -> None:
    """The one-shot setup job: download, verify, install, and prove local AI."""
    # Imported now, with the rest of this release, so an upgrade cannot mix code mid-job.
    from src import auth, discovery  # noqa: F401

    source = _source_digest()
    local = model(name)
    payload = manifest(local)

    def stop(_signal, _frame):
        raise Cancelled

    previous = signal.signal(signal.SIGTERM, stop)
    with contextlib.ExitStack() as cleanup:
        cleanup.callback(signal.signal, signal.SIGTERM, previous)
        cleanup.enter_context(job_lock(0, busy="Local AI setup is already running"))
        progress = cleanup.enter_context(Progress(local, payload["total_bytes"]))
        try:
            _require_owned()
            if not eligible():
                raise LocalRuntimeError("This Mac needs Apple silicon and at least 48 GB of memory for local AI")
            download_payload(payload, progress)
            if _source_digest() != source:
                # Model Gateway was upgraded during the download: install with the new release's code.
                progress.update()
                progress.close()  # so no heartbeat write straddles the exec
                _reexec(local.name)
            with provision_lock(wait=True):
                progress.update(state="installing")
                install(local, payload)
                check(local, payload)
            progress.update(state="done", message="")
        except Cancelled:
            if progress.value["state"] == "installing":
                progress.update(state="failed", message="Setup stopped while installing; add local AI again to repair it")
            else:
                progress.update(state="cancelled", message="")
        except Exception as exc:
            # Our own messages or network/OS errors; never response bodies, config contents, or keys.
            progress.update(state="failed", message=safe_message(exc)[:300])
            raise


# ── CLI (bin/model-gateway local-ai) ─────────────────────────────────────────


def _print_status(value: dict) -> None:
    progress = ""
    if value["bytes_total"]:
        progress = f" ({value['bytes_done'] / 1e9:.1f} / {value['bytes_total'] / 1e9:.1f} GB)"
    rows = [
        ("eligible", "yes" if value["eligible"] else "no (needs Apple silicon and 48 GB of memory)"),
        ("managed", "yes" if value["managed"] else "no"),
        ("installed", "yes" if value["installed"] else "no"),
        ("model", value["model"] or "-"),
        ("state", f"{value['state'] or 'none'}{progress}"),
        ("base_url", value["base_url"] or "-"),
    ]
    if value["message"]:
        rows.append(("message", value["message"]))
    for label, text in rows:
        print(f"{label + ':':<12}{text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-gateway local-ai", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("status", help="eligibility, ownership, and setup progress")
    show.add_argument("--json", action="store_true")
    add = commands.add_parser("add", help="download and install local AI in the background")
    add.add_argument("model", nargs="?", default=DEFAULT_MODEL, choices=sorted(MODELS))
    commands.add_parser("cancel", help="stop a download (a later add resumes it)")
    commands.add_parser("remove", help="remove the runtime, its model, and its gateway routes")
    job = commands.add_parser("job", help=argparse.SUPPRESS)
    job.add_argument("model", choices=sorted(MODELS))
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if args.command == "job":
            run_job(args.model)
            return 0
        if args.command == "status":
            value = status()
            if args.json:
                print(json.dumps(value, indent=2))
            else:
                _print_status(value)
            return 0
        if args.command == "add":
            value = start_add(args.model)
            print(f"Local AI setup {value['state'] or 'started'} for {args.model}; "
                  "follow it with 'model-gateway local-ai status'")
            return 0
        if args.command == "cancel":
            print(f"Local AI setup {cancel()['state'] or 'not running'}")
            return 0
        remove()
        print("Local AI removed")
        return 0
    except (ValueError, yaml.YAMLError, OSError, RuntimeError, subprocess.SubprocessError,
            http.client.HTTPException) as exc:
        message = safe_message(exc)
        # The job failed before it could report progress; a busy job lock means another job is.
        if args.command == "job" and not isinstance(exc, _Busy):
            with contextlib.suppress(OSError, ValueError):
                if read_status().get("state") in ACTIVE_STATES:
                    record_failure(message)
        print(f"error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
