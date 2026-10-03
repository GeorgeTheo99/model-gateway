"""Import a Home Server 0.4.x bundled gateway, and its local AI, into the component layout.

The component package helper runs this as the user, in phases:

  preflight   refuse anything unexpected; change nothing (system Python, stdlib only)
  prepare     journal, build oMLX, stop the legacy jobs, snapshot, remove the legacy
              LaunchAgents, copy the gateway state, and move local AI
  finish      start the component's oMLX and verify the consumers and local runtime
  rollback    undo exactly what the journal recorded (system Python, stdlib only)

The helper installs and verifies the component gateway between prepare and
finish. Every step and every path this import creates or moves is recorded in
``<state root>/state/migration-journal.json`` before it happens, so a re-run
after an interruption resumes and a failure rolls back only this import's
changes. The legacy gateway state is copied, never moved; models and the oMLX
inference files are renamed on the same volume. A snapshot without models goes
to ``<Home Server root>/backups/w3-migration-<timestamp>-<suffix>/`` and is kept;
a rollback moves the component state it created into its
``component-at-rollback/`` rather than deleting it.

launchctl and curl come from PATH (as in the helper), so tests substitute them.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time

SCHEMA_VERSION = 1
INFERENCE_LABEL = "com.local.home-server-inference"
SETUP_LABEL = "com.local.home-server-local-ai-setup"
OMLX_LABEL = "com.local.omlx"
ACTIVE_STATES = frozenset({"queued", "downloading", "installing"})
# A legacy setup status older than this with no running job was interrupted.
STALE_SECONDS = 60
# The oMLX environment the import rebuilds, and slack beyond the copies it makes.
VENV_BYTES = 3 * 1024**3
MARGIN_BYTES = 256 * 1024**2
CONSUMER_PERMISSIONS = {"ha-manager": ["providers:manage", "models:register"]}
SAFE_KEY_NAME = re.compile(r"[a-z0-9_][a-z0-9_.-]*")
LEDGER_FILES = ("ledger.db", "ledger.db-wal", "ledger.db-shm", "ledger.db-journal")
# Exit codes shared with the package helper.
REFUSED = 2
ALREADY_IMPORTED = 10
ROLLBACK_PENDING = 11
# A rollback that started (and may have been interrupted) or did not finish.
ROLLBACK_STATES = frozenset({"rolling-back", "rollback-failed"})


class ImportFailure(RuntimeError):
    """A failure whose message is safe to show (no keys or file contents)."""


class Refused(ImportFailure):
    """Preflight found something this import must not touch; nothing changed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── paths ────────────────────────────────────────────────────────────────────


class Paths:
    """The legacy Home Server layout and this component's layout."""

    def __init__(self, root: Path):
        self.root = root
        self.server_ci = root / "ci" / "bin" / "server-ci"
        self.install_env = root / "ci" / "config" / "install.env"
        self.setup_plist = root / "ci" / "launchd" / f"{SETUP_LABEL}.plist"
        shared = root / "runtime" / "shared"
        self.gateway = shared / "model-gateway"
        self.gateway_tree = root / "runtime" / "current" / "model-gateway"
        self.legacy_config = self.gateway / "config" / "config.yaml"
        self.inference = shared / "inference"
        self.setup_status = shared / "home-automation" / "config-data" / "local-ai-setup.json"
        self.models = root / "models"
        self.backups = root / "backups"
        self.label = os.environ.get("MODEL_GATEWAY_LAUNCHD_LABEL", "").strip() or "com.local.model-gateway"
        self.agents = Path(os.environ.get("MODEL_GATEWAY_PLIST_DIR", "").strip()
                           or Path.home() / "Library" / "LaunchAgents")
        self.gateway_plist = self.agents / f"{self.label}.plist"
        self.inference_plist = self.agents / f"{INFERENCE_LABEL}.plist"
        self.omlx_plist = self.agents / f"{OMLX_LABEL}.plist"
        self.state = Path(os.environ.get("MODEL_GATEWAY_STATE_DIR", "").strip()
                          or Path.home() / "Library" / "Application Support" / "model-gateway")
        self.journal = self.state / "state" / "migration-journal.json"
        self.config = self.state / "config.yaml"
        self.model_info = self.state / "model-info.json"
        self.ledger = self.state / "ledger.db"
        self.secrets = self.state / "secrets"
        self.registry = self.state / "state" / "consumer-profiles.json"
        self.state_install_env = self.state / "install.env"
        self.endpoint = self.state / "endpoint.json"
        self.local_ai = self.state / "local-ai"
        self.current = self.state / "current"

    @property
    def owner(self) -> str:
        """``ModelGatewayRoot`` of this gateway's oMLX, as src.local_runtime writes it."""
        return f"{self.state}#{self.label}"


# ── small edges ──────────────────────────────────────────────────────────────


def _fault(step: str) -> None:
    """Test-only fault injection; ignored unless MODEL_GATEWAY_MIGRATION_TEST=1."""
    if os.environ.get("MODEL_GATEWAY_MIGRATION_TEST") != "1":
        return
    if step in os.environ.get("MODEL_GATEWAY_MIGRATION_FAIL_AT", "").split(","):
        raise ImportFailure(f"injected failure at {step}")
    if os.environ.get("MODEL_GATEWAY_MIGRATION_INTERRUPT_AT") == step:
        # Like a reboot: kill everything up to the package helper, so nothing rolls back.
        pid, chain = os.getppid(), []
        while pid > 1:
            chain.append(pid)
            command = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
            if "model-gateway-install-from-pkg" in command:
                for victim in chain:
                    os.kill(victim, signal.SIGKILL)
                os._exit(137)
            pid = int(subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout)
        raise ImportFailure("no package helper to interrupt")


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def _target(label: str) -> str:
    return f"gui/{os.getuid()}/{label}"


def _loaded(label: str) -> tuple[bool, str | None, bool]:
    """Whether ``label`` is loaded, the plist launchd loaded it from, and whether it is running."""
    result = _launchctl("print", _target(label))
    if result.returncode != 0:
        return False, None, False
    path, running = None, False
    for line in (result.stdout or "").splitlines():
        name, _, value = line.strip().partition(" = ")
        if name == "path" and value and path is None:
            path = value
        elif name == "state" and value == "running":
            running = True
    return True, path, running


def _same(a: str | Path | None, b: Path) -> bool:
    return a is not None and os.path.realpath(str(a)) == os.path.realpath(b)


def _device(path: Path) -> int:
    return os.stat(path).st_dev


def _stop(label: str, plist: Path) -> bool:
    """Boot out ``label`` only when launchd loaded it from ``plist``; return whether it was loaded."""
    loaded, path, _running = _loaded(label)
    if not loaded:
        return False
    if not _same(path, plist):
        raise ImportFailure(f"{label} is loaded from {path or 'an unknown plist'}, not {plist}; not stopping it")
    if _launchctl("bootout", _target(label)).returncode != 0:
        raise ImportFailure(f"could not stop {label}")
    for _ in range(30):
        if not _loaded(label)[0]:
            return True
        time.sleep(1)
    raise ImportFailure(f"{label} is still loaded after bootout")


def _start(plist: Path, label: str) -> None:
    # bootstrap reports EIO (5) while a just-stopped job releases its launchd slot.
    for attempt in range(30):
        result = _launchctl("bootstrap", f"gui/{os.getuid()}", str(plist))
        if result.returncode == 0:
            return
        if result.returncode != 5 or attempt == 29:
            raise ImportFailure(f"{label} bootstrap failed (exit {result.returncode})")
        time.sleep(1)


def _health(url: str) -> dict | None:
    result = subprocess.run(["curl", "-fsS", "--max-time", "3", url], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        value = json.loads(result.stdout)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _gateway_healthy(port: int) -> bool:
    body = _health(f"http://127.0.0.1:{port}/health")
    return body is not None and body.get("status") == "ok" and body.get("service") == "model-gateway"


def _wait_health(url: str, seconds: int = 90) -> None:
    deadline = time.monotonic() + seconds
    while True:
        body = _health(url)
        if body is not None and body.get("status") in {"ok", "healthy"}:
            return
        if time.monotonic() >= deadline:
            raise ImportFailure(f"service did not become healthy: {url}")
        time.sleep(2)


# ── private files ────────────────────────────────────────────────────────────


def _no_symlinks(path: Path, base: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink():
            raise Refused(f"refusing a symlinked path: {part}")
        if part == base:
            return


def _user_file(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise Refused(f"expected a regular file owned by this user: {path}")


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, 0o600)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _sqlite_copy(source: Path, destination: Path) -> None:
    """A consistent copy of a SQLite database through the online backup API."""
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(temporary)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)


def _tree_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for directory, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except OSError:
                pass
    return total


def _read_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() and not name.lstrip().startswith("#"):
            values[name.strip()] = value.strip()
    return values


def _plist(path: Path) -> dict | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = plistlib.loads(path.read_bytes())
    except (OSError, ValueError, plistlib.InvalidFileException):
        return None
    return value if isinstance(value, dict) else None


def _port(value: object, name: str) -> int:
    if not isinstance(value, str) or not value.isdigit() or not 1024 <= int(value) <= 65535:
        raise Refused(f"{name} must be a port from 1024 to 65535")
    return int(value)


# ── journal ──────────────────────────────────────────────────────────────────


class Journal:
    """Every step and path of one import, saved before each change."""

    def __init__(self, paths: Paths, data: dict):
        self.paths, self.data = paths, data

    @classmethod
    def load(cls, paths: Paths) -> Journal | None:
        path = paths.journal
        if not path.exists() and not path.is_symlink():
            return None
        _user_file(path)
        try:
            data = json.loads(path.read_text())
        except ValueError:
            raise ImportFailure(f"the migration journal is not valid JSON: {path}") from None
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise ImportFailure(f"unsupported migration journal: {path}")
        return cls(paths, data)

    def save(self) -> None:
        self.data["updated_at"] = _now()
        _no_symlinks(self.paths.journal.parent, self.paths.state)
        _write_private(self.paths.journal, (json.dumps(self.data, indent=2) + "\n").encode())

    @property
    def state(self) -> str:
        return str(self.data.get("state"))

    def done(self, step: str) -> bool:
        return step in self.data["steps"]

    def mark(self, step: str) -> None:
        _fault(step)
        self.data["steps"].append(step)
        self.save()

    def claim(self, path: Path) -> None:
        """Record ``path`` as this import's before creating it; refuse anything else's."""
        created = self.data["created"]
        if str(path) in created:
            return
        if path.exists() or path.is_symlink():
            raise ImportFailure(f"{path} already exists; the import will not replace it")
        created.append(str(path))
        self.save()

    def claim_dirs(self, path: Path) -> None:
        """Claim each missing directory from the state root down to ``path``."""
        missing = []
        for part in (path, *path.parents):
            if part.exists():
                break
            missing.append(part)
        for part in reversed(missing):
            self.claim(part)
            part.mkdir(mode=0o700)

    def move(self, source: Path, destination: Path, *, rewritten: bool = False) -> None:
        """Rename on the same volume; recorded first so rollback can rename back."""
        record = [str(source), str(destination)]
        if record not in self.data["moved"]:
            self.data["moved"].append(record)
            if rewritten:
                self.data["rewritten"].append(str(destination))
            self.save()
        if destination.exists() or destination.is_symlink():
            if source.exists() or source.is_symlink():
                raise ImportFailure(f"both {source} and {destination} exist; resolve this by hand")
            return  # moved by an earlier, interrupted run
        self.claim_dirs(destination.parent)
        os.rename(source, destination)


# ── preflight ────────────────────────────────────────────────────────────────


def _component_installed(paths: Paths) -> bool:
    value = _plist(paths.gateway_plist) or {}
    return (value.get("ModelGatewayComponentRoot") == str(paths.state)
            and value.get("WorkingDirectory") == str(paths.current))


def inspect(paths: Paths) -> dict:
    """What the import will do, after every fail-closed check. Changes nothing."""
    root = paths.root
    if not root.is_absolute() or root.is_symlink() or not root.is_dir() or root.stat().st_uid != os.getuid():
        raise Refused(f"not a Home Server root owned by this user: {root}")
    _user_file(paths.install_env)
    installed = _read_env(paths.install_env)
    if installed.get("INSTALLED_SERVER_CI_PROFILE") != "home-server":
        raise Refused(f"{paths.install_env} is not a home-server install")
    if installed.get("INSTALLED_SERVER_MODEL_GATEWAY_MODE", "bundled") != "bundled":
        raise Refused("this Home Server is not using its bundled gateway")
    gateway_port = _port(installed.get("INSTALLED_SERVER_MODEL_GATEWAY_PORT", "9111"), "the gateway port")
    omlx_port = _port(installed.get("INSTALLED_SERVER_OMLX_PORT", "9110"), "the oMLX port")
    if gateway_port == omlx_port:
        raise Refused("the gateway and oMLX ports must differ")

    owner = str(paths.server_ci)
    gateway = _plist(paths.gateway_plist)
    working = str((gateway or {}).get("WorkingDirectory") or "")
    if (gateway is None or gateway.get("HomeServerCIPath") != owner or gateway.get("Label") != paths.label
            or not (working == str(paths.gateway_tree) or working.startswith(f"{paths.gateway_tree}/"))):
        raise Refused(f"{paths.gateway_plist} is not this Home Server's bundled gateway ({owner})")
    environment = gateway.get("EnvironmentVariables")
    environment = environment if isinstance(environment, dict) else {}
    if environment.get("MODEL_GATEWAY_PORT", str(gateway_port)) != str(gateway_port):
        raise Refused(f"{paths.gateway_plist} does not use the gateway port in {paths.install_env}")
    if environment.get("MODEL_GATEWAY_CONFIG", str(paths.legacy_config)) != str(paths.legacy_config):
        raise Refused(f"{paths.gateway_plist} does not use {paths.legacy_config}")

    local_ai = paths.inference_plist.exists() or paths.inference_plist.is_symlink()
    model_id = None
    if local_ai:
        inference = _plist(paths.inference_plist)
        if inference is None or inference.get("HomeServerCIPath") != owner or inference.get("Label") != INFERENCE_LABEL:
            raise Refused(f"{paths.inference_plist} is not this Home Server's inference LaunchAgent ({owner})")
        arguments = inference.get("ProgramArguments") or []
        if "--port" not in arguments or arguments[arguments.index("--port") + 1:][:1] != [str(omlx_port)]:
            raise Refused(f"{paths.inference_plist} does not use the oMLX port in {paths.install_env}")
        mlx = paths.models / "mlx"
        found = sorted(p.name for p in mlx.iterdir() if p.is_dir() and not p.is_symlink()
                       and not p.name.endswith(".partial")) if mlx.is_dir() else []
        if len(found) != 1:
            raise Refused(f"expected exactly one installed model in {mlx}")
        model_id = found[0]
        if model_id not in _managed_model_ids():
            raise Refused(f"the installed model {model_id} is not one this gateway manages")
        _no_symlinks(paths.inference / "api.key", paths.root)
        _user_file(paths.inference / "api.key")
        if paths.omlx_plist.exists() or paths.omlx_plist.is_symlink() or _loaded(OMLX_LABEL)[0]:
            raise Refused(f"{OMLX_LABEL} already exists on this Mac; the import will not replace it")

    _no_symlinks(paths.legacy_config, paths.root)
    _user_file(paths.legacy_config)
    _user_file(paths.gateway / "model-info.json")
    for existing in (paths.config, paths.model_info, *(paths.state / name for name in LEDGER_FILES),
                     paths.state_install_env, paths.secrets, paths.local_ai, paths.registry, paths.current):
        if existing.exists() or existing.is_symlink():
            raise Refused(f"Model Gateway state already exists at {existing}; the import will not replace it")

    if not _gateway_healthy(gateway_port):
        raise Refused(f"the Home Server gateway is not healthy on port {gateway_port}")
    if _setup_active(paths):
        raise Refused("Home Server local AI setup is running; wait for it to finish or cancel it, then retry")

    # Everything the import renames must be on the state root's volume.
    state_volume = paths.state if paths.state.exists() else paths.state.parent
    renamed = [root]
    if local_ai:
        renamed += [paths.models / "mlx" / model_id, paths.inference]
        if (paths.models / "cache").is_dir():
            renamed.append(paths.models / "cache")
    for path in renamed:
        if _device(path) != _device(state_volume):
            raise Refused(f"{path} and {paths.state} must be on the same volume")
    needed = (2 * _tree_size(paths.gateway) + _tree_size(paths.inference) + MARGIN_BYTES
              + (VENV_BYTES if local_ai else 0))
    free = shutil.disk_usage(state_volume).free
    if free < needed:
        raise Refused(f"the import needs {needed / 1e9:.1f} GB of free disk space; {free / 1e9:.1f} GB is free")
    return {"gateway_port": gateway_port, "omlx_port": omlx_port, "local_ai": local_ai, "model_id": model_id}


def _managed_model_ids() -> set[str]:
    """Model ids of this package's local-models manifests (stdlib only, for preflight)."""
    from src import model_payload

    ids = set()
    for path in sorted((Path(__file__).resolve().parents[1] / "local-models").glob("*.json")):
        try:
            value = model_payload.read_json(path, "model manifest")
            model_payload.validate_manifest(value)
        except model_payload.VerificationError:
            continue
        ids.add(value["model_id"])
    return ids


def _setup_active(paths: Paths) -> bool:
    """Whether the Home Server local AI setup job is running or recently reported progress."""
    _loaded_setup, _path, running = _loaded(SETUP_LABEL)
    status = {}
    if paths.setup_status.is_file() and not paths.setup_status.is_symlink():
        try:
            status = json.loads(paths.setup_status.read_text())
        except ValueError:
            status = {}
    return running or (isinstance(status, dict) and status.get("state") in ACTIVE_STATES and not _stale(status))


def _stale(status: dict) -> bool:
    try:
        updated = datetime.fromisoformat(str(status.get("updated_at")))
    except ValueError:
        return True
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - updated).total_seconds() > STALE_SECONDS


def preflight(paths: Paths) -> int:
    journal = Journal.load(paths)
    if journal is not None and journal.state == "completed":
        if journal.data.get("home_server_root") == str(paths.root) and _component_installed(paths):
            print(f"Already imported from {paths.root} ({paths.journal})")
            return ALREADY_IMPORTED
        raise Refused(f"{paths.journal} records a completed import, but {paths.gateway_plist} is not this "
                      "component's gateway")
    if journal is not None and (journal.state == "in-progress" or journal.state in ROLLBACK_STATES):
        _require_same_import(journal)
        if journal.state in ROLLBACK_STATES:
            print(f"An earlier import did not finish rolling back ({paths.journal})")
            return ROLLBACK_PENDING
        print(f"Resuming the import recorded in {paths.journal}")
        return 0
    plan = inspect(paths)
    print(f"Importing the Home Server gateway from {paths.root} (port {plan['gateway_port']}, "
          f"local AI: {'yes' if plan['local_ai'] else 'no'})")
    return 0


def _require_same_import(journal: Journal) -> None:
    """A recorded import resumes or rolls back only for its own Home Server root and label."""
    paths = journal.paths
    if journal.data.get("home_server_root") != str(paths.root):
        raise Refused(f"{paths.journal} records an import from {journal.data.get('home_server_root')}")
    if journal.data.get("label") != paths.label:
        raise Refused(f"{paths.journal} records an import for {journal.data.get('label')}, not {paths.label}")


# ── prepare ──────────────────────────────────────────────────────────────────


def _begin(paths: Paths) -> Journal:
    journal = Journal.load(paths)
    if journal is not None and journal.state in ROLLBACK_STATES:
        raise ImportFailure(f"an earlier import did not finish rolling back ({paths.journal})")
    if journal is not None and journal.state == "in-progress":
        _require_same_import(journal)
        return journal
    plan = inspect(paths)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    journal = Journal(paths, {
        "schema_version": SCHEMA_VERSION, "state": "in-progress", "home_server_root": str(paths.root),
        "label": paths.label, "started_at": _now(), **plan,
        "backup_dir": str(paths.backups / f"w3-migration-{stamp}-{os.getpid()}-{os.urandom(3).hex()}"),
        "steps": [], "created": [], "moved": [], "rewritten": [], "stopped": [], "removed_plists": [],
        "consumers": [],
    })
    _no_symlinks(paths.journal.parent, paths.state)
    paths.journal.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    journal.save()
    # The gateway writes it on start; a rollback must not leave a component discovery file behind.
    if not paths.endpoint.exists():
        journal.claim(paths.endpoint)
    return journal


def _build_runtime(journal: Journal) -> None:
    from src import local_runtime

    if journal.data["model_id"] != local_runtime.manifest(local_runtime.model(None))["model_id"]:
        raise ImportFailure(f"the installed model {journal.data['model_id']} is not one this gateway manages")
    venv = local_runtime.venv_dir()
    journal.claim_dirs(venv.parent)
    journal.claim(venv)
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(venv), UV_PYTHON_PREFERENCE="only-managed", UV_NO_CONFIG="1")
    try:
        subprocess.run([local_runtime._uv(), "sync", "--locked", "--no-dev", "--project",
                        str(local_runtime.RUNTIME_PROJECT)], check=True, env=env)
        subprocess.run([str(venv / "bin" / "python"), "-c", local_runtime.METAL_SMOKE], check=True)
    except (OSError, subprocess.CalledProcessError):
        raise ImportFailure("could not build the oMLX environment from the locked local runtime") from None


def _stop_legacy(journal: Journal) -> None:
    paths = journal.paths
    # Preflight saw no setup running; one that started since is not stopped mid-install.
    if _setup_active(paths):
        raise ImportFailure("Home Server local AI setup started during the import; not stopping it")
    jobs = [(SETUP_LABEL, paths.setup_plist), (paths.label, paths.gateway_plist)]
    if journal.data["local_ai"]:
        jobs.append((INFERENCE_LABEL, paths.inference_plist))
    for label, plist in jobs:
        record = [label, str(plist)]
        loaded, path, _running = _loaded(label)
        if loaded and _same(path, plist) and label != SETUP_LABEL and record not in journal.data["stopped"]:
            journal.data["stopped"].append(record)
            journal.save()
        if _stop(label, plist):
            print(f"Stopped {label}")


def _snapshot(journal: Journal) -> None:
    paths = journal.paths
    backup = Path(journal.data["backup_dir"])
    if backup.exists():
        shutil.rmtree(backup)  # an interrupted snapshot of this import
    paths.backups.mkdir(mode=0o700, exist_ok=True)
    backup.mkdir(mode=0o700)
    shutil.copytree(paths.gateway, backup / "model-gateway", symlinks=True,
                    ignore=lambda _d, names: [name for name in names if name in LEDGER_FILES])
    if paths.inference.is_dir():
        shutil.copytree(paths.inference, backup / "inference", symlinks=True)
    shutil.copy2(paths.install_env, backup / "install.env")
    (backup / "LaunchAgents").mkdir(mode=0o700)
    for plist in (paths.gateway_plist, paths.inference_plist):
        if plist.is_file():
            shutil.copy2(plist, backup / "LaunchAgents" / plist.name)
    ledger = paths.gateway / "state" / "ledger.db"
    if ledger.is_file():
        _sqlite_copy(ledger, backup / "model-gateway" / "state" / "ledger.db")
    print(f"Snapshot (without models): {backup}")


def _under(path: Path, base: Path) -> bool:
    return os.path.realpath(path) == os.path.realpath(base) or \
        os.path.realpath(path).startswith(os.path.realpath(base) + os.sep)


def _copy_secret(journal: Journal, source: Path, destination: Path) -> None:
    _no_symlinks(source, journal.paths.root)
    _user_file(source)
    journal.claim_dirs(destination.parent)
    journal.claim(destination)
    _write_private(destination, source.read_bytes())


def _legacy_references(value: object, root: Path, where: str = "") -> list[str]:
    if isinstance(value, dict):
        return [hit for key, item in value.items() for hit in _legacy_references(item, root, f"{where}.{key}")]
    if isinstance(value, list):
        return [hit for index, item in enumerate(value) for hit in _legacy_references(item, root, f"{where}[{index}]")]
    if isinstance(value, str) and str(root) in os.path.expanduser(value):
        return [where.lstrip(".")]
    return []


def _copy_state(journal: Journal) -> None:
    import yaml

    from src import config_io, local_runtime

    paths = journal.paths
    legacy_dir = paths.legacy_config.parent
    try:
        config = yaml.safe_load(paths.legacy_config.read_text()) or {}
    except yaml.YAMLError:
        raise ImportFailure(f"{paths.legacy_config} is not valid YAML") from None
    if not isinstance(config, dict):
        raise ImportFailure(f"{paths.legacy_config} is not a mapping")
    providers_dir = Path(os.path.realpath(paths.secrets / "providers"))

    auth = config.get("auth") or {}
    entries = (auth.get("consumer_credentials") or []) if isinstance(auth, dict) else []
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ImportFailure("the legacy auth.consumer_credentials is not a list of objects")
    ids = []
    for entry in entries:
        ids.append(str(entry.get("id")))
        if isinstance(entry.get("key_file"), str):
            source = Path(os.path.expanduser(entry["key_file"]))
            source = source if source.is_absolute() else legacy_dir / source
            if not _under(source, paths.root):
                continue
            try:
                destination = config_io.consumer_key_path(entry.get("id"))
            except (TypeError, ValueError):
                raise ImportFailure(f"the legacy consumer credential id {entry.get('id')!r} is invalid") from None
            _copy_secret(journal, source, destination)
            entry["key_file"] = str(destination)
        # Attached Home Server's Settings also manages the gateway-owned local AI.
        if entry.get("id") == "ha-manager" and entry.get("permissions") == CONSUMER_PERMISSIONS["ha-manager"]:
            entry["permissions"] = [*entry["permissions"], "local_ai:manage"]
    journal.data["consumers"] = ids
    journal.save()

    # Each provider key goes to <provider id>.api-key; two different keys never share a file.
    copied: dict[str, str] = {}

    def copy_provider_key(source: Path, destination: Path) -> None:
        key, origin = destination.name.lower(), os.path.realpath(source)
        if copied.setdefault(key, origin) != origin:
            raise ImportFailure(f"two different legacy provider keys would both be imported as {destination}")
        _copy_secret(journal, source, destination)

    providers = config.get("providers")
    providers = providers if isinstance(providers, dict) else {}
    for provider_id, block in list(providers.items()):
        if str(provider_id).lower() == "omlx":
            if not journal.data["local_ai"]:
                raise ImportFailure("the legacy gateway has an oMLX provider but no local AI LaunchAgent")
            # Exactly the block src.local_runtime.configure_gateway writes for its own runtime.
            providers[provider_id] = {
                "base_url": f"http://127.0.0.1:{journal.data['omlx_port']}/v1", "protocol": "openai",
                "api_key": "", "api_key_file": os.path.realpath(local_runtime.api_key_path()), "enabled": True,
                "managed_by": "local_ai"}
            continue
        if isinstance(block, dict) and isinstance(block.get("api_key_file"), str):
            source = Path(os.path.expanduser(block["api_key_file"]))
            source = source if source.is_absolute() else legacy_dir / source
            if _under(source, paths.root):
                name = f"{str(provider_id).lower()}.api-key"
                if not SAFE_KEY_NAME.fullmatch(name):
                    raise ImportFailure(f"the legacy provider id {provider_id!r} is not a safe file name")
                destination = providers_dir / name
                copy_provider_key(source, destination)
                block["api_key_file"] = str(destination)
    legacy_providers = paths.gateway / "secrets" / "providers"
    if legacy_providers.is_dir():
        for source in sorted(legacy_providers.iterdir()):
            if source.is_file() and not source.is_symlink() and os.path.realpath(source) not in copied.values():
                copy_provider_key(source, providers_dir / source.name)

    profiles = config.get("profiles")
    config["profiles"] = {**(profiles if isinstance(profiles, dict) else {}), "registry_path": str(paths.registry)}
    exports = config.get("exports")
    if isinstance(exports, dict) and isinstance(exports.get("model_aliases"), str) and \
            str(paths.root) in os.path.expanduser(exports["model_aliases"]):
        exports["model_aliases"] = str(paths.state / "model-aliases.json")
    leftovers = _legacy_references(config, paths.root)
    if leftovers:
        raise ImportFailure(f"the legacy config still refers to {paths.root} at: {', '.join(leftovers)}")

    journal.claim(paths.config)
    _write_private(paths.config, yaml.safe_dump(config, sort_keys=False, default_flow_style=False).encode())
    journal.claim(paths.model_info)
    _write_private(paths.model_info, (paths.gateway / "model-info.json").read_bytes())
    registry = paths.gateway / "state" / "consumer-profiles.json"
    if registry.is_file():
        _copy_secret(journal, registry, paths.registry)
    client_keys = paths.gateway / "secrets" / "client.keys"
    if client_keys.is_file() and client_keys.read_text().strip():
        _copy_secret(journal, client_keys, paths.secrets / "client.keys")
    ledger = paths.gateway / "state" / "ledger.db"
    if ledger.is_file():
        # The component gateway's SQLite sidecars are this import's too, so a rollback takes them.
        for name in LEDGER_FILES:
            journal.claim(paths.state / name)
        _sqlite_copy(ledger, paths.ledger)
    journal.claim(paths.state_install_env)
    _write_private(paths.state_install_env,
                   f"MODEL_GATEWAY_HOST=127.0.0.1\nMODEL_GATEWAY_PORT={journal.data['gateway_port']}\n"
                   "MODEL_GATEWAY_ADMIN_WRITES=true\n".encode())
    print(f"Copied the gateway state to {paths.state}")


def _rewrite_paths(value: object, replacements: list[tuple[str, str]]) -> object:
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements:
            if value == old or value.startswith(old + "/"):
                return new + value[len(old):]
    return value


def _move_local_ai(journal: Journal) -> None:
    from src import local_runtime

    paths = journal.paths
    model_id = journal.data["model_id"]
    journal.move(paths.models / "mlx" / model_id, local_runtime.mlx_dir() / model_id)
    if (paths.models / "cache").is_dir() or local_runtime.cache_dir().exists():
        journal.move(paths.models / "cache", local_runtime.cache_dir())
    else:
        journal.claim_dirs(local_runtime.cache_dir())
    inference = local_runtime.inference_dir()
    replacements = [(str(paths.inference), str(inference)),
                    (str(paths.models / "mlx"), str(local_runtime.mlx_dir())),
                    (str(paths.models / "cache"), str(local_runtime.cache_dir())),
                    (str(paths.root / "logs" / "inference"), str(local_runtime.runtime_root() / "logs" / "omlx"))]
    journal.move(paths.inference / "api.key", inference / "api.key")
    os.chmod(inference / "api.key", 0o600)
    for name in ("settings.json", "model_settings.json"):
        source, destination = paths.inference / name, inference / name
        if not source.exists() and not destination.exists():
            continue
        journal.move(source, destination, rewritten=True)
        try:
            value = json.loads(destination.read_text())
        except ValueError:
            raise ImportFailure(f"{source} is not valid JSON") from None
        value = _rewrite_paths(value, replacements)
        if name == "settings.json" and isinstance(value, dict):
            logging = value.get("logging") if isinstance(value.get("logging"), dict) else {}
            value["logging"] = {**logging, "log_dir": str(local_runtime.runtime_root() / "logs" / "omlx")}
        leftovers = _legacy_references(value, paths.root)
        if leftovers:
            raise ImportFailure(f"{source} still refers to {paths.root} at: {', '.join(leftovers)}")
        _write_private(destination, (json.dumps(value, indent=2) + "\n").encode())
    journal.claim_dirs(local_runtime.runtime_root() / "logs")
    if paths.setup_status.is_file() and not paths.setup_status.is_symlink() and not local_runtime.status_path().exists():
        try:
            status = json.loads(paths.setup_status.read_text())
        except ValueError:
            status = None
        if isinstance(status, dict):
            status["model"] = local_runtime.model(None).name
            journal.claim(local_runtime.status_path())
            _write_private(local_runtime.status_path(), (json.dumps(status) + "\n").encode())
    journal.claim(local_runtime.omlx_plist_path())
    _write_private(local_runtime.omlx_plist_path(),
                   plistlib.dumps(local_runtime.omlx_plist(journal.data["omlx_port"])))
    print(f"Moved local AI into {local_runtime.runtime_root()}")


def _remove_legacy_plists(journal: Journal) -> None:
    paths = journal.paths
    backup = Path(journal.data["backup_dir"]) / "LaunchAgents"
    plists = [paths.gateway_plist] + ([paths.inference_plist] if journal.data["local_ai"] else [])
    for plist in plists:
        if not plist.exists():
            continue
        if (_plist(plist) or {}).get("HomeServerCIPath") != str(paths.server_ci):
            raise ImportFailure(f"{plist} changed owner during the import")
        if not (backup / plist.name).is_file():
            raise ImportFailure(f"the snapshot has no copy of {plist}")
        record = [str(plist), str(backup / plist.name)]
        if record not in journal.data["removed_plists"]:
            journal.data["removed_plists"].append(record)
            journal.save()
        plist.unlink()


def prepare(paths: Paths) -> None:
    journal = _begin(paths)
    if journal.data["local_ai"] and not journal.done("runtime"):
        # Before stopping anything: building oMLX may need the network.
        _build_runtime(journal)
        journal.mark("runtime")
    if not journal.done("remove-legacy-plists"):
        # Every resume: a restart may have loaded the legacy LaunchAgents again.
        _stop_legacy(journal)
        _fault("stop-legacy")
    elif journal.data["local_ai"] and not journal.done("local-ai") and _setup_active(paths):
        # A resume after a restart: a setup job may have started before the models move.
        raise ImportFailure("Home Server local AI setup started during the import; not moving its models")
    # The plists go right after the snapshot that holds them, so a restart mid-import
    # cannot load the legacy gateway or inference again.
    for step, function in (("snapshot", _snapshot), ("remove-legacy-plists", _remove_legacy_plists),
                           ("copy-state", _copy_state), ("local-ai", _move_local_ai)):
        if step == "local-ai" and not journal.data["local_ai"]:
            continue
        if not journal.done(step):
            function(journal)
            journal.mark(step)


# ── finish ───────────────────────────────────────────────────────────────────


def finish(paths: Paths) -> None:
    from src import config_io, local_runtime

    journal = Journal.load(paths)
    if journal is None or journal.state != "in-progress":
        raise ImportFailure("there is no import in progress")
    if not journal.done("install"):
        journal.mark("install")
    if journal.data["local_ai"] and not journal.done("start-local-ai"):
        if not _loaded(OMLX_LABEL)[0]:
            _start(local_runtime.omlx_plist_path(), OMLX_LABEL)
        _wait_health(f"http://127.0.0.1:{journal.data['omlx_port']}/health")
        journal.mark("start-local-ai")
    listed = {row["id"]: row for row in config_io.list_consumer_credentials()}
    for credential in journal.data["consumers"]:
        row = listed.get(credential)
        if row is None or row["key_status"] not in {"ok", "inline"}:
            raise ImportFailure(f"the imported consumer credential {credential} is missing or has no usable key")
    if journal.data["local_ai"]:
        current = local_runtime.status()
        if not (current["installed"] and current["managed"]):
            raise ImportFailure("the imported local AI is not installed and managed by this gateway")
    _fault("verify")
    journal.data.update(state="completed", completed_at=_now())
    journal.save()
    print(f"Imported the Home Server gateway; consumers: {', '.join(journal.data['consumers']) or 'none'}")


# ── rollback ─────────────────────────────────────────────────────────────────


def _kept_at_rollback(paths: Paths, path: Path) -> bool:
    """Component state a rollback keeps in the snapshot: config, ledger, secrets, and state but the journal."""
    if _same(path, paths.config) or _under(path, paths.secrets):
        return True
    if any(_same(path, paths.state / name) for name in LEDGER_FILES):
        return True
    state = paths.journal.parent
    return _under(path, state) and not _same(path, state) and not _same(path, paths.journal)


def _rollback_failed(journal: Journal, errors: list[str]) -> int:
    journal.data.update(state="rollback-failed", errors=errors)
    journal.save()
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1


def rollback(paths: Paths) -> int:
    journal = Journal.load(paths)
    if journal is None or not (journal.state == "in-progress" or journal.state in ROLLBACK_STATES):
        print("No import to roll back")
        return 0
    # Recorded first: an interrupted rollback is retried, never resumed as an import.
    journal.data["state"] = "rolling-back"
    journal.save()
    errors: list[str] = []

    def attempt(description: str, function, *args) -> bool:
        try:
            function(*args)
        except Exception as exc:  # noqa: BLE001 - every step is attempted; failures are recorded
            errors.append(f"{description}: {exc}")
            return False
        return True

    attempt("rolling back", _fault, "rollback")

    def stop_gateway() -> None:
        if (_plist(paths.gateway_plist) or {}).get("ModelGatewayComponentRoot") == str(paths.state):
            _stop(paths.label, paths.gateway_plist)
            paths.gateway_plist.unlink()
        # Preflight found no component install, so `current` is this import's.
        if paths.current.is_symlink():
            paths.current.unlink()

    def stop_omlx() -> None:
        if (_plist(paths.omlx_plist) or {}).get("ModelGatewayRoot") == paths.owner:
            _stop(OMLX_LABEL, paths.omlx_plist)
            paths.omlx_plist.unlink()

    gateway_stopped = attempt("stopping the component gateway", stop_gateway)
    # While the component's oMLX may still serve local AI, its files stay and the legacy jobs stay stopped.
    omlx_stopped = attempt("stopping the component oMLX", stop_omlx)
    if not gateway_stopped:
        # A running gateway would recreate the state being moved, and the legacy one needs its port:
        # nothing more is undone until a retry can stop it.
        return _rollback_failed(journal, errors)

    def held(path: Path) -> bool:
        return not omlx_stopped and (_under(path, paths.local_ai) or _same(path, paths.omlx_plist)
                                     or _same(path, paths.inference_plist))

    backup = Path(journal.data["backup_dir"])
    for source, destination in reversed(journal.data["moved"] if omlx_stopped else []):
        source, destination = Path(source), Path(destination)

        def move_back(source=source, destination=destination) -> None:
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.rename(destination, source)
            if str(destination) in journal.data["rewritten"] and (backup / "inference" / source.name).is_file():
                shutil.copy2(backup / "inference" / source.name, source)

        attempt(f"moving {destination} back", move_back)
    kept = backup / "component-at-rollback"
    for created in journal.data["created"]:
        path = Path(created)
        if not _kept_at_rollback(paths, path) or not os.path.lexists(path):
            continue  # a parent directory already moved it

        def keep(path=path) -> None:
            relative = Path(os.path.relpath(os.path.realpath(path.parent), os.path.realpath(paths.state)))
            target = kept / relative / path.name
            if os.path.lexists(target):
                raise ImportFailure(f"{target} already exists; not moving {path} there")
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.rename(path, target)

        attempt(f"moving {path} into {kept}", keep)
    pending = [Path(destination) for _source, destination in journal.data["moved"] if Path(destination).exists()]
    for created in reversed(journal.data["created"]):
        path = Path(created)
        if _kept_at_rollback(paths, path) or held(path):
            continue

        def remove(path=path) -> None:
            if any(_under(item, path) for item in pending):
                raise ImportFailure(f"{path} still holds moved files; not removing it")
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

        attempt(f"removing {path}", remove)
    for plist, saved in journal.data["removed_plists"]:
        plist, saved = Path(plist), Path(saved)
        if held(plist):
            continue

        def restore(plist=plist, saved=saved) -> None:
            if not plist.exists():
                shutil.copy2(saved, plist)

        attempt(f"restoring {plist}", restore)
    for label, plist in journal.data["stopped"] if omlx_stopped else []:
        plist = Path(plist)

        def start(label=label, plist=plist) -> None:
            if not _loaded(label)[0]:
                _start(plist, label)

        attempt(f"starting {label}", start)
    if errors:
        return _rollback_failed(journal, errors)
    journal.data.pop("errors", None)
    journal.data.update(state="rolled-back", rolled_back_at=_now())
    journal.save()
    print(f"Rolled back the import; the Home Server gateway in {paths.root} is restored")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.legacy_import", description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=["preflight", "prepare", "finish", "rollback"])
    parser.add_argument("--root", required=True, type=Path, help="the Home Server root")
    args = parser.parse_args(argv)
    os.umask(0o077)
    paths = Paths(args.root)
    try:
        if args.phase == "preflight":
            return preflight(paths)
        if args.phase == "rollback":
            return rollback(paths)
        if args.phase == "prepare":
            prepare(paths)
        else:
            finish(paths)
        return 0
    except Refused as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return REFUSED
    except (ImportFailure, OSError, ValueError, shutil.Error, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
