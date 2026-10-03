"""Importing a Home Server 0.4.x bundled gateway with the component package helper.

Like test_component_pkg, nothing touches the real launchd, ~/Library, or the
network: the helper runs with a fake HOME and fake launchctl, curl, lsof,
sleep, and uv first on PATH. The fake launchctl remembers which plist each
label was loaded from (``launchctl print`` reports it), and the fake curl
answers /health for whichever loaded plist owns the requested port.
"""

from __future__ import annotations

import json
import os
import plistlib
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.version import VERSION
from test_component_pkg import LABEL, SUPPORT, SYSTEM_PATH, release_name, staged, write_fake  # noqa: F401

GATEWAY_PORT = 9171
OMLX_PORT = 9187
MODEL_ID = "qwen3.8-27b-8bit-30gb"
INFERENCE_LABEL = "com.local.home-server-inference"
SETUP_LABEL = "com.local.home-server-local-ai-setup"
CONSUMERS = ("ha-runtime", "ha-deployer", "ha-manager")

FAKE_LAUNCHCTL = """#!/bin/bash
# Fake launchctl: loaded labels are files holding their plist path.
state="$HOME/.fake"
mkdir -p "$state/loaded"
printf '%s\\n' "$*" >> "$state/launchctl.log"
interrupt() {
  local pid=$PPID chain=()
  while [ "$pid" -gt 1 ]; do
    chain+=("$pid")
    case "$(ps -o command= -p "$pid")" in
      *model-gateway-install-from-pkg*) kill -KILL "${chain[@]}"; exit 1 ;;
    esac
    pid="$(ps -o ppid= -p "$pid" | tr -d ' ')"
  done
  exit 1
}
start() {
  if [ -e "$state/interrupt" ]; then
    rm -f "$state/interrupt"
    interrupt
  fi
  "$FAKE_PYTHON" - "$1" <<'PY'
import json, os, plistlib, sys
from pathlib import Path
plist = plistlib.loads(Path(sys.argv[1]).read_bytes())
home = Path(os.environ["HOME"])
(home / ".fake/loaded" / plist["Label"]).write_text(sys.argv[1])
root = plist.get("ModelGatewayComponentRoot")
if not root:
    raise SystemExit(0)
app = Path(root)
version = [line.split('"')[1] for line in (app / "current/src/version.py").read_text().splitlines()
           if line.startswith("VERSION = ")][0]
fail = home / ".fake/fail-version"
if fail.exists() and fail.read_text().strip() == version:
    raise SystemExit(0)
port = int(plist["EnvironmentVariables"]["MODEL_GATEWAY_PORT"])
(app / "endpoint.json").write_text(json.dumps(
    {"version": 1, "service": "model-gateway", "gateway_version": version, "port": port}))
(app / "endpoint.json").chmod(0o600)
PY
}
label="${2##*/}"
case "$1" in
  print)
    [ -e "$state/loaded/$label" ] || exit 113
    run="waiting"
    [ ! -e "$state/running-$label" ] || run="running"
    printf '%s = {\\n\\tpath = %s\\n\\tstate = %s\\n}\\n' "$2" "$(cat "$state/loaded/$label")" "$run"
    ;;
  bootout) rm -f "$state/loaded/$label" ;;
  bootstrap) start "$3" ;;
  load) start "$2" ;;
  kickstart) label="${3##*/}"; start "$(cat "$state/loaded/$label" 2>/dev/null || echo "$HOME/Library/LaunchAgents/$label.plist")" ;;
  enable) exit 0 ;;
  *) exit 1 ;;
esac
"""

FAKE_CURL = """#!$FAKE_PYTHON
# Fake curl: /health answers for the loaded LaunchAgent that serves the requested port.
import json, os, plistlib, sys
from pathlib import Path
home = Path(os.environ["HOME"])
port = sys.argv[-1].split("://", 1)[1].split("/", 1)[0].rsplit(":", 1)[1]
loaded = home / ".fake/loaded"
for marker in sorted(loaded.iterdir()) if loaded.is_dir() else ():
    plist = plistlib.loads(Path(marker.read_text().strip()).read_bytes())
    arguments = plist.get("ProgramArguments") or []
    if (plist.get("EnvironmentVariables") or {}).get("MODEL_GATEWAY_PORT") == port:
        root = plist.get("ModelGatewayComponentRoot")
        if root:
            version = [line.split('"')[1] for line in (Path(root) / "current/src/version.py").read_text().splitlines()
                       if line.startswith("VERSION = ")][0]
            fail = home / ".fake/fail-version"
            if fail.exists() and fail.read_text().strip() == version:
                continue
        print(json.dumps({"status": "ok", "service": "model-gateway"}, separators=(",", ":")), end="")
        raise SystemExit(0)
    if "--port" in arguments and arguments[arguments.index("--port") + 1] == port:
        print('{"status":"healthy"}', end="")
        raise SystemExit(0)
raise SystemExit(7)
"""

FAKE_UV = """#!/bin/bash
# Fake uv: a gateway venv runs this test's Python; the oMLX venv's python passes the Metal smoke.
printf 'UV_PYTHON_PREFERENCE=%s UV_PROJECT_ENVIRONMENT=%s %s\\n' "${UV_PYTHON_PREFERENCE:-}" \\
  "$UV_PROJECT_ENVIRONMENT" "$*" >> "$HOME/.fake/uv.log"
[ ! -e "$HOME/.fake/uv-fail" ] || exit 1
[ "$1" = sync ] || exit 2
mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
case "$(basename "$UV_PROJECT_ENVIRONMENT")" in
  omlx-*)
    printf '#!/bin/sh\\nexit 0\\n' > "$UV_PROJECT_ENVIRONMENT/bin/python"
    printf '#!/bin/sh\\nexit 0\\n' > "$UV_PROJECT_ENVIRONMENT/bin/omlx"
    chmod 755 "$UV_PROJECT_ENVIRONMENT/bin/omlx"
    ;;
  *) printf '#!/bin/sh\\nexec "%s" "$@"\\n' "$FAKE_PYTHON" > "$UV_PROJECT_ENVIRONMENT/bin/python" ;;
esac
chmod 755 "$UV_PROJECT_ENVIRONMENT/bin/python"
"""


def private(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o600)
    return path


class Legacy:
    """A fake user account with a Home Server 0.4.x bundled gateway (and optionally local AI)."""

    def __init__(self, base: Path, *, local_ai: bool = True):
        self.home = base / "home"
        self.fakes = base / "fakes"
        self.fake = self.home / ".fake"
        (self.fake / "loaded").mkdir(parents=True)
        for name, text in (("launchctl", FAKE_LAUNCHCTL), ("curl", FAKE_CURL), ("uv", FAKE_UV),
                           ("lsof", "#!/bin/bash\nexit 1\n"), ("sleep", "#!/bin/bash\nexit 0\n")):
            write_fake(self.fakes, name, text.replace("$FAKE_PYTHON", sys.executable))
        self.agents = self.home / "Library/LaunchAgents"
        self.app = self.home / "Library/Application Support/model-gateway"
        self.root = self.home / "Library/Application Support/HomeServer"
        self.shared = self.root / "runtime/shared"
        self.gateway = self.shared / "model-gateway"
        self.inference = self.shared / "inference"
        self.local_ai = local_ai
        self.build()

    @property
    def gateway_plist(self) -> Path:
        return self.agents / f"{LABEL}.plist"

    @property
    def inference_plist(self) -> Path:
        return self.agents / f"{INFERENCE_LABEL}.plist"

    @property
    def omlx_plist(self) -> Path:
        return self.agents / "com.local.omlx.plist"

    def build(self) -> None:
        root, gateway = self.root, self.gateway
        server_ci = root / "ci/bin/server-ci"
        server_ci.parent.mkdir(parents=True)
        server_ci.write_text("#!/bin/bash\nexit 0\n")
        server_ci.chmod(0o755)
        private(root / "ci/config/install.env",
                "INSTALLED_SERVER_CI_PROFILE=home-server\nINSTALLED_SERVER_HOME_AUTOMATION_PORT=8100\n"
                f"INSTALLED_SERVER_MODEL_GATEWAY_PORT={GATEWAY_PORT}\nINSTALLED_SERVER_OMLX_PORT={OMLX_PORT}\n")
        (root / "runtime/current/model-gateway/src").mkdir(parents=True)
        (root / "runtime/current/model-gateway/src/version.py").write_text('VERSION = "0.2.1"\n')
        secrets = gateway / "secrets"
        for name in CONSUMERS:
            private(secrets / f"{name}.key", f"key-{name}\n")
        private(secrets / "client.keys", "legacy-client-key\n")
        private(secrets / "providers/fireworks", "fw-secret\n")
        private(secrets / "providers/unreferenced", "other-secret\n")
        credentials = [
            {"id": "ha-runtime", "consumer": "ha", "key_file": str(secrets / "ha-runtime.key"), "namespaces": ["ha"],
             "permissions": ["profiles:read", "profiles:invoke"], "allow_direct_models": False},
            {"id": "ha-deployer", "consumer": "ha", "key_file": str(secrets / "ha-deployer.key"), "namespaces": ["ha"],
             "permissions": ["profiles:read", "profiles:write"], "allow_direct_models": False},
            {"id": "ha-manager", "consumer": "ha", "key_file": str(secrets / "ha-manager.key"), "namespaces": ["ha"],
             "permissions": ["providers:manage", "models:register"], "providers": ["fireworks"],
             "allow_direct_models": False},
        ]
        providers = {"fireworks": {"base_url": "https://api.fireworks.ai/inference/v1", "protocol": "openai",
                                   "enabled": True, "api_key_file": str(secrets / "providers/fireworks")}}
        if self.local_ai:
            providers["omlx"] = {"base_url": f"http://127.0.0.1:{OMLX_PORT}/v1", "api_key": "",
                                 "api_key_file": str(self.inference / "api.key"), "enabled": True}
        private(gateway / "config/config.yaml", yaml.safe_dump({
            "auth": {"admin_keys": [], "client_keys": [], "consumer_credentials": credentials},
            "providers": providers,
            "profiles": {"registry_path": str(gateway / "state/consumer-profiles.json")},
            "exports": {"model_aliases": str(gateway / "model-aliases.json")},
        }, sort_keys=False))
        private(gateway / "model-info.json", json.dumps({"updated": "2026-09-01", "llm": [
            {"name": "qwen3.8-27b", "omlx_id": MODEL_ID, "context": 32768, "max_output_tokens": 4096,
             "vision": True, "pricing_status": "unmetered"},
            {"name": "glm-5.3-flash-fw", "provider": "fireworks",
             "provider_model_id": "accounts/fireworks/models/glm-5p3-flash", "context": 131072},
        ]}))
        private(gateway / "state/consumer-profiles.json", json.dumps({"namespaces": {"ha": {"profiles": []}}}))
        ledger = sqlite3.connect(gateway / "state/ledger.db")
        ledger.execute("CREATE TABLE usage (id INTEGER PRIMARY KEY, model TEXT)")
        ledger.execute("INSERT INTO usage (model) VALUES ('qwen3.8-27b'), ('glm-5.3-flash-fw')")
        ledger.commit()
        ledger.close()
        self.write_plist(self.gateway_plist, {
            "Label": LABEL, "HomeServerCIPath": str(server_ci),
            "ProgramArguments": ["uv", "run", "--locked", "python", "-m", "src.main"],
            "WorkingDirectory": str(root / "runtime/current/model-gateway"),
            "EnvironmentVariables": {"MODEL_GATEWAY_PORT": str(GATEWAY_PORT),
                                     "MODEL_GATEWAY_CONFIG": str(gateway / "config/config.yaml"),
                                     "MODEL_GATEWAY_CLIENT_KEYS_FILE": str(secrets / "client.keys")}})
        if not self.local_ai:
            return
        private(self.inference / "api.key", "omlx-key\n")
        private(self.inference / "settings.json", json.dumps({
            "auth": {"api_key": "omlx-key"}, "logging": {"log_dir": str(root / "logs/inference")},
            "model": {"model_dirs": [str(root / "models/mlx")]}}))
        private(self.inference / "model_settings.json", json.dumps(
            {"version": 1, "models": {MODEL_ID: {"is_default": True}}}))
        (self.inference / ".provision.lock").touch()
        model = root / "models/mlx" / MODEL_ID
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (root / "models/cache").mkdir()
        (root / "models/cache/block").write_text("kv")
        (root / "models/omlx-0.6.3/bin").mkdir(parents=True)
        private(self.shared / "home-automation/config-data/local-ai-setup.json", json.dumps({
            "schema_version": 1, "state": "done", "model": MODEL_ID, "updated_at": "2026-09-01T00:00:00+00:00"}))
        self.write_plist(self.inference_plist, {
            "Label": INFERENCE_LABEL, "HomeServerCIPath": str(server_ci),
            "ProgramArguments": [str(root / "models/omlx-0.6.3/bin/omlx"), "serve", "--base-path", str(self.inference),
                                 "--model-dir", str(root / "models/mlx"), "--port", str(OMLX_PORT)],
            "WorkingDirectory": str(self.inference)})

    def write_plist(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(plistlib.dumps(value))
        path.chmod(0o600)
        (self.fake / "loaded" / value["Label"]).write_text(str(path))

    def run(self, support: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
        environment = {"HOME": str(self.home), "PATH": f"{self.fakes}:{SYSTEM_PATH}", "USER": "tester",
                       "TMPDIR": os.environ.get("TMPDIR", "/tmp"), "MODEL_GATEWAY_PKG_SUPPORT_DIR": str(support),
                       "MODEL_GATEWAY_LAUNCHD_LABEL": LABEL, **env}
        return subprocess.run([str(support / "bin/model-gateway-install-from-pkg"), *args], capture_output=True,
                              text=True, env=environment, timeout=180)

    def import_(self, support: Path, **env: str) -> subprocess.CompletedProcess:
        return self.run(support, "--import-legacy-home-server", str(self.root), **env)

    def loaded(self, label: str) -> str | None:
        marker = self.fake / "loaded" / label
        return marker.read_text() if marker.exists() else None

    def journal(self) -> dict:
        return json.loads((self.app / "state/migration-journal.json").read_text())

    def snapshot(self) -> dict[str, bytes]:
        """Every legacy file outside the backups, with its contents."""
        files = {}
        for top in (self.root, self.agents):
            for path in top.rglob("*"):
                if path.is_file() and "backups" not in path.relative_to(self.root if top == self.root else top).parts:
                    files[str(path)] = path.read_bytes()
        return files


@pytest.fixture
def legacy(tmp_path) -> Legacy:
    return Legacy(tmp_path)


@pytest.fixture
def cloud_only(tmp_path) -> Legacy:
    return Legacy(tmp_path, local_ai=False)


def imported(machine: Legacy, staged: Path) -> subprocess.CompletedProcess:  # noqa: F811
    result = machine.import_(staged / SUPPORT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MODEL_GATEWAY_IMPORT=imported" in result.stdout
    return result


def assert_component_running(machine: Legacy, support: Path) -> None:
    plist = plistlib.loads(machine.gateway_plist.read_bytes())
    assert plist["ModelGatewayComponentRoot"] == str(machine.app)
    assert plist["WorkingDirectory"] == str(machine.app / "current")
    assert plist["EnvironmentVariables"]["MODEL_GATEWAY_PORT"] == str(GATEWAY_PORT)
    assert machine.loaded(LABEL) == str(machine.gateway_plist)
    assert os.readlink(machine.app / "current") == f"releases/{release_name(support)}"
    endpoint = json.loads((machine.app / "endpoint.json").read_text())
    assert endpoint["port"] == GATEWAY_PORT and endpoint["gateway_version"] == VERSION


def assert_legacy_restored(machine: Legacy, before: dict[str, bytes]) -> None:
    assert machine.snapshot() == before
    assert machine.loaded(LABEL) == str(machine.gateway_plist)
    assert "HomeServerCIPath" in plistlib.loads(machine.gateway_plist.read_bytes())
    if machine.local_ai:
        assert machine.loaded(INFERENCE_LABEL) == str(machine.inference_plist)
    assert machine.loaded("com.local.omlx") is None and not machine.omlx_plist.exists()
    for name in ("config.yaml", "model-info.json", "ledger.db", "install.env", "secrets", "local-ai", "current",
                 "endpoint.json"):
        assert not (machine.app / name).exists(), name
    assert machine.journal()["state"] == "rolled-back"


# ── happy paths ──────────────────────────────────────────────────────────────


def test_imports_a_bundled_gateway_with_local_ai(legacy, staged):
    support = staged / SUPPORT
    result = imported(legacy, staged)
    assert f"Model Gateway {VERSION} imported the Home Server gateway" in result.stdout
    assert_component_running(legacy, support)
    app = legacy.app

    config = yaml.safe_load((app / "config.yaml").read_text())
    credentials = {row["id"]: row for row in config["auth"]["consumer_credentials"]}
    assert sorted(credentials) == sorted(CONSUMERS)
    for name in CONSUMERS:
        key = app / "secrets/consumers" / f"{name}.key"
        assert credentials[name]["key_file"] == str(key)
        assert key.read_text() == f"key-{name}\n" and key.stat().st_mode & 0o777 == 0o600
    assert credentials["ha-manager"]["permissions"] == ["providers:manage", "models:register", "local_ai:manage"]
    assert credentials["ha-manager"]["providers"] == ["fireworks"]
    assert config["providers"]["fireworks"]["api_key_file"] == str(app / "secrets/providers/fireworks")
    assert (app / "secrets/providers/fireworks").read_text() == "fw-secret\n"
    assert (app / "secrets/providers/unreferenced").stat().st_mode & 0o777 == 0o600
    assert config["providers"]["omlx"] == {
        "base_url": f"http://127.0.0.1:{OMLX_PORT}/v1", "protocol": "openai", "api_key": "",
        "api_key_file": str(app / "local-ai/inference/api.key"), "enabled": True, "managed_by": "local_ai"}
    assert config["profiles"] == {"registry_path": str(app / "state/consumer-profiles.json")}
    assert config["exports"]["model_aliases"] == str(app / "model-aliases.json")
    assert str(legacy.root) not in (app / "config.yaml").read_text()
    assert (app / "config.yaml").stat().st_mode & 0o777 == 0o600
    assert (app / "model-info.json").read_bytes() == (legacy.gateway / "model-info.json").read_bytes()
    assert (app / "state/consumer-profiles.json").read_bytes() == \
        (legacy.gateway / "state/consumer-profiles.json").read_bytes()
    assert (app / "install.env").read_text() == \
        f"MODEL_GATEWAY_HOST=127.0.0.1\nMODEL_GATEWAY_PORT={GATEWAY_PORT}\nMODEL_GATEWAY_ADMIN_WRITES=true\n"
    rows = sqlite3.connect(app / "ledger.db").execute("SELECT model FROM usage ORDER BY id").fetchall()
    assert rows == [("qwen3.8-27b",), ("glm-5.3-flash-fw",)]
    # The legacy client key keeps working: the component LaunchAgent reads the imported copy.
    plist = plistlib.loads(legacy.gateway_plist.read_bytes())
    assert plist["EnvironmentVariables"]["MODEL_GATEWAY_CLIENT_KEYS_FILE"] == str(app / "secrets/client.keys")
    assert (app / "secrets/client.keys").read_text() == "legacy-client-key\n"

    # Local AI moved (not copied) and owned by this gateway.
    local = app / "local-ai"
    assert (local / "models/mlx" / MODEL_ID / "config.json").exists()
    assert not (legacy.root / "models/mlx" / MODEL_ID).exists()
    assert (local / "models/cache/block").read_text() == "kv" and not (legacy.root / "models/cache").exists()
    assert (local / "inference/api.key").read_text() == "omlx-key\n"
    assert not (legacy.inference / "api.key").exists()
    settings = json.loads((local / "inference/settings.json").read_text())
    assert settings == {"auth": {"api_key": "omlx-key"}, "logging": {"log_dir": str(local / "logs/omlx")},
                        "model": {"model_dirs": [str(local / "models/mlx")]}}
    assert json.loads((local / "inference/model_settings.json").read_text())["models"] == {MODEL_ID: {"is_default": True}}
    status = json.loads((local / "status.json").read_text())
    assert status["state"] == "done" and status["model"] == "qwen3.8-27b"
    omlx = plistlib.loads(legacy.omlx_plist.read_bytes())
    assert omlx["ModelGatewayRoot"] == f"{app}#{LABEL}"
    assert omlx["ProgramArguments"][0] == str(local / "omlx-0.6.3/bin/omlx")
    assert omlx["ProgramArguments"][omlx["ProgramArguments"].index("--port") + 1] == str(OMLX_PORT)
    assert legacy.loaded("com.local.omlx") == str(legacy.omlx_plist)
    uv = (legacy.fake / "uv.log").read_text()
    assert f"UV_PYTHON_PREFERENCE=only-managed UV_PROJECT_ENVIRONMENT={local / 'omlx-0.6.3'} sync --locked" in uv

    # The legacy jobs are gone, their state stays for the Home Server package.
    assert not legacy.inference_plist.exists() and legacy.loaded(INFERENCE_LABEL) is None
    assert (legacy.gateway / "config/config.yaml").exists() and (legacy.inference / "settings.json").exists() is False
    assert (legacy.inference / ".provision.lock").exists()
    journal = legacy.journal()
    assert journal["state"] == "completed" and journal["home_server_root"] == str(legacy.root)
    assert journal["steps"] == ["runtime", "snapshot", "copy-state", "local-ai", "remove-legacy-plists",
                                "install", "start-local-ai"]
    backup = Path(journal["backup_dir"])
    assert backup.parent == legacy.root / "backups" and backup.name.startswith("w3-migration-")
    assert (backup / "model-gateway/config/config.yaml").read_bytes() == \
        (legacy.gateway / "config/config.yaml").read_bytes()
    assert (backup / "inference/api.key").read_text() == "omlx-key\n"
    assert (backup / "install.env").exists()
    assert sorted(path.name for path in (backup / "LaunchAgents").iterdir()) == sorted(
        [legacy.gateway_plist.name, legacy.inference_plist.name])
    assert sqlite3.connect(backup / "model-gateway/state/ledger.db").execute("SELECT count(*) FROM usage").fetchone() == (2,)
    assert not any(backup.rglob(MODEL_ID))


def test_imports_a_cloud_only_gateway(cloud_only, staged):
    support = staged / SUPPORT
    imported(cloud_only, staged)
    assert_component_running(cloud_only, support)
    config = yaml.safe_load((cloud_only.app / "config.yaml").read_text())
    assert "omlx" not in config["providers"]
    assert not (cloud_only.app / "local-ai").exists() and not cloud_only.omlx_plist.exists()
    assert "omlx-" not in (cloud_only.fake / "uv.log").read_text()
    assert cloud_only.journal()["steps"] == ["snapshot", "copy-state", "remove-legacy-plists", "install"]


def test_rerunning_after_an_import_is_a_no_op(legacy, staged):
    imported(legacy, staged)
    calls = (legacy.fake / "launchctl.log").read_text()
    again = legacy.import_(staged / SUPPORT)
    assert again.returncode == 0, again.stderr
    assert "MODEL_GATEWAY_IMPORT=already-imported" in again.stdout
    assert (legacy.fake / "launchctl.log").read_text() == calls
    # The plain package helper sees its own component install.
    plain = legacy.run(staged / SUPPORT)
    assert plain.returncode == 0 and "already installed and running" in plain.stdout


# ── preflight ────────────────────────────────────────────────────────────────


def refused(machine: Legacy, staged: Path, message: str) -> None:  # noqa: F811
    before = machine.snapshot()
    result = machine.import_(staged / SUPPORT)
    assert result.returncode == 2, result.stdout + result.stderr
    assert message in result.stderr and "MODEL_GATEWAY_IMPORT=refused" in result.stdout
    assert machine.snapshot() == before
    assert not (machine.app / "state").exists()
    assert machine.loaded(LABEL) == str(machine.gateway_plist)


def test_refuses_plists_another_home_server_root_owns(legacy, staged):
    plist = plistlib.loads(legacy.gateway_plist.read_bytes())
    plist["HomeServerCIPath"] = str(legacy.home / "Other/ci/bin/server-ci")
    legacy.write_plist(legacy.gateway_plist, plist)
    refused(legacy, staged, "is not this Home Server's bundled gateway")


def test_refuses_an_inference_plist_another_root_owns(legacy, staged):
    plist = plistlib.loads(legacy.inference_plist.read_bytes())
    plist["HomeServerCIPath"] = str(legacy.home / "Other/ci/bin/server-ci")
    legacy.write_plist(legacy.inference_plist, plist)
    refused(legacy, staged, "is not this Home Server's inference LaunchAgent")


def test_refuses_an_unhealthy_legacy_gateway(legacy, staged):
    (legacy.fake / "loaded" / LABEL).unlink()
    before = legacy.snapshot()
    result = legacy.import_(staged / SUPPORT)
    assert result.returncode == 2 and "is not healthy" in result.stderr
    assert legacy.snapshot() == before


def test_refuses_while_local_ai_setup_runs(legacy, staged):
    setup = legacy.root / "ci/launchd" / f"{SETUP_LABEL}.plist"
    legacy.write_plist(setup, {"Label": SETUP_LABEL})
    (legacy.fake / f"running-{SETUP_LABEL}").touch()
    refused(legacy, staged, "local AI setup is running")


def test_refuses_existing_component_state_or_omlx(legacy, staged):
    legacy.omlx_plist.write_bytes(plistlib.dumps({"Label": "com.local.omlx"}))
    refused(legacy, staged, "com.local.omlx already exists")
    legacy.omlx_plist.unlink()
    private(legacy.app / "config.yaml", "auth: {}\n")
    refused(legacy, staged, "Model Gateway state already exists")


def test_refuses_a_gateway_port_that_disagrees_with_install_env(legacy, staged):
    private(legacy.root / "ci/config/install.env",
            "INSTALLED_SERVER_CI_PROFILE=home-server\nINSTALLED_SERVER_MODEL_GATEWAY_PORT=9172\n"
            f"INSTALLED_SERVER_OMLX_PORT={OMLX_PORT}\n")
    refused(legacy, staged, "does not use the gateway port")


def test_usage_errors_change_nothing(legacy, staged):
    result = legacy.run(staged / SUPPORT, "--import-legacy-home-server", "relative/root")
    assert result.returncode == 2 and "absolute" in result.stderr
    assert not legacy.app.exists()


# ── rollback and resume ──────────────────────────────────────────────────────


TEST = {"MODEL_GATEWAY_MIGRATION_TEST": "1"}
STEPS = ["runtime", "stop-legacy", "snapshot", "copy-state", "local-ai", "remove-legacy-plists", "install",
         "start-local-ai", "verify"]


@pytest.mark.parametrize("step", STEPS)
def test_a_failure_at_each_step_rolls_back_to_the_legacy_gateway(legacy, staged, step):
    before = legacy.snapshot()
    result = legacy.import_(staged / SUPPORT, MODEL_GATEWAY_MIGRATION_FAIL_AT=step, **TEST)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"injected failure at {step}" in result.stderr
    assert "MODEL_GATEWAY_IMPORT=rolled-back" in result.stdout
    assert_legacy_restored(legacy, before)
    # A later attempt starts over and succeeds.
    imported(legacy, staged)
    assert_component_running(legacy, staged / SUPPORT)


def test_a_component_that_does_not_start_is_rolled_back(legacy, staged):
    before = legacy.snapshot()
    (legacy.fake / "fail-version").write_text(VERSION)
    result = legacy.import_(staged / SUPPORT)
    assert result.returncode == 1 and "did not start on the imported state" in result.stderr
    assert_legacy_restored(legacy, before)


def test_fault_injection_is_ignored_without_the_test_marker(legacy, staged):
    result = legacy.import_(staged / SUPPORT, MODEL_GATEWAY_MIGRATION_FAIL_AT="copy-state")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("step", STEPS[:-1])
def test_an_interrupted_import_resumes(legacy, staged, step):
    interrupted = legacy.import_(staged / SUPPORT, MODEL_GATEWAY_MIGRATION_INTERRUPT_AT=step, **TEST)
    assert interrupted.returncode != 0
    assert legacy.journal()["state"] == "in-progress"
    # Until it resumes, the plain package helper leaves the half-done import alone.
    plain = legacy.run(staged / SUPPORT)
    assert plain.returncode == 0 and "has not finished" in plain.stdout
    result = imported(legacy, staged)
    assert "Resuming the import" in result.stdout
    assert_component_running(legacy, staged / SUPPORT)
    assert (legacy.app / "local-ai/models/mlx" / MODEL_ID / "config.json").exists()
    assert legacy.loaded("com.local.omlx") == str(legacy.omlx_plist)
    assert not legacy.inference_plist.exists() and legacy.loaded(INFERENCE_LABEL) is None
    config = yaml.safe_load((legacy.app / "config.yaml").read_text())
    assert config["auth"]["consumer_credentials"][0]["key_file"] == str(legacy.app / "secrets/consumers/ha-runtime.key")


def test_an_interrupted_import_can_still_roll_back(legacy, staged):
    before = legacy.snapshot()
    legacy.import_(staged / SUPPORT, MODEL_GATEWAY_MIGRATION_INTERRUPT_AT="local-ai", **TEST)
    # A resumed run that then fails rolls back everything, including the earlier run's work.
    result = legacy.import_(staged / SUPPORT, MODEL_GATEWAY_MIGRATION_FAIL_AT="install", **TEST)
    assert result.returncode == 1, result.stdout + result.stderr
    assert_legacy_restored(legacy, before)


def test_rollback_restores_rewritten_inference_settings(legacy, staged):
    original = (legacy.inference / "settings.json").read_bytes()
    result = legacy.import_(staged / SUPPORT, MODEL_GATEWAY_MIGRATION_FAIL_AT="remove-legacy-plists", **TEST)
    assert result.returncode == 1
    assert (legacy.inference / "settings.json").read_bytes() == original
    assert (legacy.root / "models/mlx" / MODEL_ID / "config.json").exists()
