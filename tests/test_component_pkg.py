"""Model Gateway component package: build, payload verification, and the per-user helper.

Nothing here touches the real launchd, ~/Library, or the network: the build
runs on a throwaway Git copy of this checkout, and the helper runs with a fake
HOME and fake launchctl, curl, lsof, sleep, and uv first on PATH, against a
test-only LaunchAgent label.
"""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.version import CAPABILITIES, VERSION

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "packaging" / "component"
IDENTIFIER = "com.local.model-gateway.component"
SUPPORT = Path("payload/Library/Application Support/ModelGateway")
RUNTIME_TOP = ["LICENSE", "README.md", "bin", "config", "local-models", "local-runtime", "pyproject.toml",
               "scripts", "src", "uv.lock"]
LABEL = "com.local.model-gateway-pkgtest"
HEALTH = '{"status":"ok","service":"model-gateway"}'
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                           "-c", "commit.gpgSign=false", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def make_repo(destination: Path) -> Path:
    """Commit this checkout's runtime files and packaging (including uncommitted edits) to a new repo."""
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-co", "--exclude-standard", "--",
                             *RUNTIME_TOP, "packaging/component"],
                            check=True, capture_output=True, text=True).stdout.splitlines()
    for relative in listed:
        source = ROOT / relative
        if source.is_file() and not source.is_symlink():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    _git(destination, "init", "-q")
    _git(destination, "add", "-A")
    _git(destination, "commit", "-qm", "release")
    return destination


def build(repo: Path, out: Path, *args: str, path: str = SYSTEM_PATH) -> subprocess.CompletedProcess:
    env = {"HOME": str(out.parent), "PATH": path, "TMPDIR": os.environ.get("TMPDIR", "/tmp")}
    return subprocess.run([str(repo / "packaging/component/scripts/build-component-pkg.sh"),
                           "--out-dir", str(out), *args], capture_output=True, text=True, env=env, timeout=120)


@pytest.fixture(scope="module")
def staged(tmp_path_factory) -> Path:
    """A dry-run staging of the current release: payload/ and scripts/."""
    base = tmp_path_factory.mktemp("component")
    repo = make_repo(base / "repo")
    result = build(repo, base / "out", "--version", VERSION, "--dry-run")
    assert result.returncode == 0, result.stderr
    return base / "out" / f"ModelGateway-{VERSION}.dry-run"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_lines(support: Path) -> list[str]:
    files = sorted(f"./{path.relative_to(support).as_posix()}" for top in ("bin", "package/gateway")
                   for path in (support / top).rglob("*") if path.is_file())
    return [f"{sha256(support / name[2:])}  {name}" for name in files]


def rebind(support: Path, **release: object) -> None:
    """Rewrite the manifest and release.plist after editing a staged payload, as a build would."""
    (support / "manifest.sha256").write_text("\n".join(manifest_lines(support)) + "\n")
    path = support / "package" / "release.plist"
    metadata = plistlib.loads(path.read_bytes())
    metadata.update(release, manifest_sha256=sha256(support / "manifest.sha256"))
    path.write_bytes(plistlib.dumps(metadata, sort_keys=True))


def variant(staged: Path, destination: Path, version: str) -> Path:
    """A copy of the staged support directory relabeled as another gateway version."""
    support = destination / "ModelGateway"
    shutil.copytree(staged / SUPPORT, support)
    version_py = support / "package/gateway/src/version.py"
    version_py.write_text(version_py.read_text().replace(f'VERSION = "{VERSION}"', f'VERSION = "{version}"'))
    commit = plistlib.loads((support / "package/release.plist").read_bytes())["source_commit"]
    rebind(support, package_version=version, release_name=f"{version}-{commit[:12]}")
    return support


# ── build ────────────────────────────────────────────────────────────────────


def test_dry_run_stages_only_runtime_files_with_bound_release_metadata(staged):
    support = staged / SUPPORT
    gateway = support / "package" / "gateway"
    assert sorted(path.name for path in gateway.iterdir()) == RUNTIME_TOP
    assert [path.name for path in (gateway / "bin").iterdir()] == ["model-gateway"]
    assert sorted(path.name for path in support.iterdir()) == ["bin", "manifest.sha256", "package"]
    assert [path.name for path in (support / "bin").iterdir()] == ["model-gateway-install-from-pkg"]
    assert [path.name for path in (staged / "scripts").iterdir()] == ["postinstall"]
    assert not any(path.is_symlink() for path in support.rglob("*"))
    assert all(not path.stat().st_mode & 0o022 for path in support.rglob("*"))

    lines = (support / "manifest.sha256").read_text().splitlines()
    assert lines == manifest_lines(support)  # sorted, complete, and exact
    release = plistlib.loads((support / "package/release.plist").read_bytes())
    commit = release["source_commit"]
    assert release == {
        "format_version": 1, "package_identifier": IDENTIFIER, "package_version": VERSION,
        "release_name": f"{VERSION}-{commit[:12]}", "source_commit": commit,
        "manifest_sha256": sha256(support / "manifest.sha256"),
        "helper_sha256": sha256(support / "bin/model-gateway-install-from-pkg"),
        "postinstall_sha256": sha256(staged / "scripts/postinstall"),
        "capabilities": sorted(CAPABILITIES), "helper_timeout": 1500, "postinstall_timeout": 1560,
    }
    assert (support / "bin/model-gateway-install-from-pkg").read_bytes() == \
        (COMPONENT / "bin/model-gateway-install-from-pkg").read_bytes()


def test_builds_of_one_revision_have_identical_inventory_and_metadata(staged, tmp_path):
    repo = make_repo(tmp_path / "repo")
    result = build(repo, tmp_path / "out", "--version", VERSION, "--dry-run")
    assert result.returncode == 0, result.stderr
    again = tmp_path / "out" / f"ModelGateway-{VERSION}.dry-run" / SUPPORT
    first = staged / SUPPORT
    assert (again / "manifest.sha256").read_bytes() == (first / "manifest.sha256").read_bytes()
    release, other = (plistlib.loads((root / "package/release.plist").read_bytes()) for root in (first, again))
    assert release.pop("source_commit") and other.pop("source_commit")  # a fresh repo, so a new commit id
    release.pop("release_name"), other.pop("release_name")
    assert release == other


def test_build_refuses_dirty_trees_mismatched_versions_and_symlinks(tmp_path):
    repo = make_repo(tmp_path / "repo")
    out = tmp_path / "out"
    mismatch = build(repo, out, "--version", "9.9.9", "--dry-run")
    assert mismatch.returncode != 0 and "does not match src/version.py" in mismatch.stderr
    (repo / "src" / "scratch.py").write_text("x = 1\n")
    dirty = build(repo, out, "--version", VERSION, "--dry-run")
    assert dirty.returncode != 0 and "dirty" in dirty.stderr
    (repo / "src" / "scratch.py").unlink()
    (repo / "src" / "link.py").symlink_to("main.py")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "symlink")
    linked = build(repo, out, "--version", VERSION, "--dry-run")
    assert linked.returncode != 0 and "symlink" in linked.stderr
    assert not out.exists() or not any(out.iterdir())


FAKE_PKGBUILD = """#!/bin/bash
# Fake pkgbuild: a tar "package" of PackageInfo, Payload/, and Scripts/.
set -euo pipefail
while [ $# -gt 1 ]; do
  case "$1" in
    --root) root="$2"; shift 2 ;;
    --scripts) scripts="$2"; shift 2 ;;
    --identifier) identifier="$2"; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    --install-location|--ownership) shift 2 ;;
    *) echo "unexpected pkgbuild argument: $1" >&2; exit 2 ;;
  esac
done
work="$(mktemp -d)"
cp -R "$root" "$work/Payload"
cp -R "$scripts" "$work/Scripts"
printf '<?xml version="1.0"?>\\n<pkg-info identifier="%s" version="%s" install-location="/"><scripts><postinstall file="./postinstall"/></scripts></pkg-info>\\n' \\
  "$identifier" "$version" > "$work/PackageInfo"
tar -cf "$1" -C "$work" .
"""

FAKE_PKGUTIL = """#!/bin/bash
set -euo pipefail
case "$1" in
  --expand|--expand-full) mkdir -p "$3"; tar -xf "$2" -C "$3" ;;
  --flatten) tar -cf "$3" -C "$2" . ;;
  *) exit 2 ;;
esac
"""


def write_fake(directory: Path, name: str, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text)
    (directory / name).chmod(0o755)


def test_build_and_verify_a_component_package_with_fake_pkgbuild(tmp_path):
    repo = make_repo(tmp_path / "repo")
    fakes = tmp_path / "fakes"
    write_fake(fakes, "pkgbuild", FAKE_PKGBUILD)
    write_fake(fakes, "pkgutil", FAKE_PKGUTIL)
    out = tmp_path / "out"
    result = build(repo, out, "--version", VERSION, path=f"{fakes}:{SYSTEM_PATH}")
    assert result.returncode == 0, result.stderr
    pkg = out / f"ModelGateway-component-{VERSION}.pkg"
    assert (out / f"{pkg.name}.sha256").read_text().split()[0] == sha256(pkg)
    listing = subprocess.run(["tar", "-tf", str(pkg)], check=True, capture_output=True, text=True).stdout
    assert "./Scripts/postinstall" in listing.splitlines()
    verifier = repo / "packaging/component/scripts/verify-component-pkg.sh"
    env = {"PATH": f"{fakes}:{SYSTEM_PATH}", "HOME": str(tmp_path)}
    verified = subprocess.run([str(verifier), str(pkg)], capture_output=True, text=True, env=env)
    assert verified.returncode == 0, verified.stderr
    assert f"{IDENTIFIER} {VERSION}" in verified.stdout
    # The build patched the postinstall timeout into PackageInfo.
    expanded = tmp_path / "expanded"
    subprocess.run([str(fakes / "pkgutil"), "--expand", str(pkg), str(expanded)], check=True)
    assert 'timeout="1560"' in (expanded / "PackageInfo").read_text()
    # A package whose payload was altered fails verification.
    gateway_main = expanded / "Payload/Library/Application Support/ModelGateway/package/gateway/src/main.py"
    gateway_main.write_text(gateway_main.read_text() + "# tampered\n")
    tampered = tmp_path / "tampered.pkg"
    subprocess.run([str(fakes / "pkgutil"), "--flatten", str(expanded), str(tampered)], check=True)
    refused = subprocess.run([str(verifier), str(tampered)], capture_output=True, text=True, env=env)
    assert refused.returncode != 0 and "verification failed" in refused.stderr
    assert build(repo, out, "--version", VERSION, path=f"{fakes}:{SYSTEM_PATH}").returncode != 0  # never overwrites


# ── postinstall payload verification ─────────────────────────────────────────


def verify_payload(staged: Path, support: Path, postinstall: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([str(postinstall or staged / "scripts/postinstall"), "--verify-payload", str(support)],
                          capture_output=True, text=True, env={"PATH": SYSTEM_PATH})


def tamper_cases():
    def edit_source(support):
        path = support / "package/gateway/src/main.py"
        path.write_text(path.read_text() + "# tampered\n")

    def extra_file(support):
        (support / "package/gateway/src/extra.py").write_text("x = 1\n")

    def extra_rebound(support):
        # Even a consistent manifest cannot hide a file: release.plist binds the manifest hash.
        (support / "package/gateway/src/extra.py").write_text("x = 1\n")
        (support / "manifest.sha256").write_text("\n".join(manifest_lines(support)) + "\n")

    def symlink(support):
        (support / "package/gateway/src/link.py").symlink_to("main.py")

    def missing_file(support):
        (support / "package/gateway/uv.lock").unlink()

    def stray_package_file(support):
        (support / "package/notes.txt").write_text("hi\n")

    def helper(support):
        path = support / "bin/model-gateway-install-from-pkg"
        path.write_text(path.read_text() + "# tampered\n")
        rebind(support)

    def version_mismatch(support):
        path = support / "package/release.plist"
        metadata = plistlib.loads(path.read_bytes())
        metadata["package_version"] = "99.0.0"
        path.write_bytes(plistlib.dumps(metadata))

    return [edit_source, extra_file, extra_rebound, symlink, missing_file, stray_package_file, helper,
            version_mismatch]


def test_postinstall_accepts_the_staged_payload(staged):
    result = verify_payload(staged, staged / SUPPORT)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("tamper", tamper_cases(), ids=lambda case: case.__name__)
def test_postinstall_refuses_a_tampered_payload(staged, tmp_path, tamper):
    support = tmp_path / "ModelGateway"
    shutil.copytree(staged / SUPPORT, support, symlinks=True)
    tamper(support)
    result = verify_payload(staged, support)
    assert result.returncode != 0
    assert "ERROR:" in result.stderr


def test_postinstall_refuses_a_payload_it_was_not_built_with(staged, tmp_path):
    other = tmp_path / "postinstall"
    other.write_text((staged / "scripts/postinstall").read_text() + "# different\n")
    other.chmod(0o755)
    result = verify_payload(staged, staged / SUPPORT, other)
    assert result.returncode != 0 and "postinstall checksum" in result.stderr


def test_postinstall_installs_only_on_the_startup_disk_and_as_root(staged):
    postinstall = str(staged / "scripts/postinstall")
    other_volume = subprocess.run([postinstall, "pkg", "/", "/Volumes/Other", "/"], capture_output=True, text=True)
    assert other_volume.returncode != 0 and "startup disk" in other_volume.stderr
    if os.getuid() != 0:
        not_root = subprocess.run([postinstall, "pkg", "/", "/", "/"], capture_output=True, text=True)
        assert not_root.returncode != 0 and "must run as root" in not_root.stderr


def test_postinstall_and_helper_share_the_payload_contract():
    postinstall = (COMPONENT / "scripts/postinstall").read_text()
    helper = (COMPONENT / "bin/model-gateway-install-from-pkg").read_text()
    for check in ("shasum -a 256 -c manifest.sha256",
                  "<(find ./bin ./package/gateway -type f -print | LC_ALL=C sort)",
                  "\"./package/release.plist\" ] || exit 1",
                  "format_version"):
        assert check in postinstall and check in helper
    assert "target-user.plist" in postinstall and "launchctl asuser" in postinstall
    assert "/usr/bin/env -i" in postinstall


# ── per-user helper ──────────────────────────────────────────────────────────


FAKE_LAUNCHCTL = """#!/bin/bash
# Fake launchctl: loaded labels are files; starting writes endpoint.json like the gateway.
state="$HOME/.fake"
mkdir -p "$state/loaded"
printf '%s\\n' "$*" >> "$state/launchctl.log"
start() {
  local plist="$1" label
  label="$(basename "$plist" .plist)"
  touch "$state/loaded/$label"
  "$FAKE_PYTHON" - "$plist" <<'PY'
import json, os, plistlib, sys
from pathlib import Path
app = Path(os.environ["HOME"]) / "Library/Application Support/model-gateway"
version = [line.split('"')[1] for line in (app / "current/src/version.py").read_text().splitlines()
           if line.startswith("VERSION = ")][0]
fail = Path(os.environ["HOME"]) / ".fake/fail-version"
if fail.exists() and fail.read_text().strip() == version:
    raise SystemExit(0)
port = int(plistlib.loads(Path(sys.argv[1]).read_bytes())["EnvironmentVariables"]["MODEL_GATEWAY_PORT"])
path = app / "endpoint.json"
path.write_text(json.dumps({"version": 1, "service": "model-gateway", "gateway_version": version, "port": port}))
path.chmod(0o600)
PY
}
case "$1" in
  print) [ -e "$state/loaded/${2##*/}" ] ;;
  bootout) rm -f "$state/loaded/${2##*/}" ;;
  bootstrap) start "$3" ;;
  load) start "$2" ;;
  kickstart) start "$HOME/Library/LaunchAgents/${3##*/}.plist" ;;
  enable) exit 0 ;;
  *) exit 1 ;;
esac
"""

FAKE_CURL = """#!/bin/bash
# Fake curl: the gateway answers /health while loaded, unless its version is set to fail.
app="$HOME/Library/Application Support/model-gateway"
version="$(sed -n 's/^VERSION = "\\(.*\\)"$/\\1/p' "$app/current/src/version.py" 2>/dev/null)"
[ -n "$(ls "$HOME/.fake/loaded" 2>/dev/null)" ] || exit 7
[ "$(cat "$HOME/.fake/fail-version" 2>/dev/null)" != "$version" ] || exit 7
printf '%s' '{"status":"ok","service":"model-gateway"}'
"""

FAKE_UV = """#!/bin/bash
# Fake uv: a venv whose python is this test run's interpreter.
printf '%s\\n' "$*" >> "$HOME/.fake/uv.log"
[ ! -e "$HOME/.fake/uv-fail" ] || exit 1
[ "$1" = sync ] || exit 2
mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
printf '#!/bin/sh\\nexec "%s" "$@"\\n' "$FAKE_PYTHON" > "$UV_PROJECT_ENVIRONMENT/bin/python"
chmod 755 "$UV_PROJECT_ENVIRONMENT/bin/python"
"""


class Machine:
    """A fake user account: HOME, LaunchAgents, and fake system tools."""

    def __init__(self, base: Path):
        self.base = base
        self.home = base / "home"
        self.fakes = base / "fakes"
        self.state = self.home / ".fake"
        self.state.mkdir(parents=True)
        python = sys.executable
        for name, text in (("launchctl", FAKE_LAUNCHCTL), ("curl", FAKE_CURL), ("uv", FAKE_UV),
                           ("lsof", "#!/bin/bash\nexit 1\n"), ("sleep", "#!/bin/bash\nexit 0\n")):
            write_fake(self.fakes, name, text.replace("$FAKE_PYTHON", python))
        self.app = self.home / "Library/Application Support/model-gateway"
        self.plist = self.home / "Library/LaunchAgents" / f"{LABEL}.plist"

    def run(self, support: Path, **env: str) -> subprocess.CompletedProcess:
        environment = {"HOME": str(self.home), "PATH": f"{self.fakes}:{SYSTEM_PATH}", "USER": "tester",
                       "TMPDIR": os.environ.get("TMPDIR", "/tmp"), "MODEL_GATEWAY_PKG_SUPPORT_DIR": str(support),
                       "MODEL_GATEWAY_LAUNCHD_LABEL": LABEL, **env}
        return subprocess.run([str(support / "bin/model-gateway-install-from-pkg")], capture_output=True,
                              text=True, env=environment, timeout=180)

    def launchctl_calls(self) -> list[str]:
        log = self.state / "launchctl.log"
        return log.read_text().splitlines() if log.exists() else []

    def current(self) -> str:
        return os.readlink(self.app / "current")


@pytest.fixture
def machine(tmp_path) -> Machine:
    return Machine(tmp_path)


def installed(machine: Machine, staged: Path) -> subprocess.CompletedProcess:
    result = machine.run(staged / SUPPORT)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def release_name(support: Path) -> str:
    return plistlib.loads((support / "package/release.plist").read_bytes())["release_name"]


def test_fresh_install_creates_a_component_owned_gateway(machine, staged):
    result = installed(machine, staged)
    support = staged / SUPPORT
    name = release_name(support)
    assert f"Model Gateway {VERSION} installed and verified" in result.stdout
    assert machine.current() == f"releases/{name}"
    release = machine.app / "releases" / name
    package = dict(line.split("=", 1) for line in (release / ".package").read_text().splitlines())
    assert package["MANAGER"] == "model-gateway-pkg"
    assert package["ROOT"] == str(machine.app / "current")
    assert package["PYTHON"] == str(machine.app / "current/.venv/bin/python")
    assert package["COMPONENT_ROOT"] == str(machine.app)
    assert package["UV"] == str(machine.fakes / "uv")
    plist = plistlib.loads(machine.plist.read_bytes())
    assert plist["ModelGatewayComponentRoot"] == str(machine.app)
    assert plist["WorkingDirectory"] == str(machine.app / "current")
    assert plist["ProgramArguments"][0] == package["PYTHON"]
    # A config with no admin key, the first free port persisted, and discovery written.
    config = yaml.safe_load((machine.app / "config.yaml").read_text())
    assert config["auth"] == {"client_keys": []} and config["providers"] == {}
    assert oct((machine.app / "config.yaml").stat().st_mode & 0o777) == "0o600"
    port = dict(line.split("=", 1) for line in (machine.app / "install.env").read_text().splitlines())
    assert 9111 <= int(port["MODEL_GATEWAY_PORT"]) <= 9159
    assert json.loads((machine.app / "endpoint.json").read_text())["gateway_version"] == VERSION
    assert os.readlink(machine.home / ".local/bin/model-gateway") == str(machine.app / "current/bin/model-gateway")
    assert any(call.startswith("bootstrap ") for call in machine.launchctl_calls())
    sync = (machine.state / "uv.log").read_text()
    assert "sync --project" in sync and "--frozen --no-dev --no-install-project" in sync


def test_rerunning_the_same_package_is_a_no_op(machine, staged):
    installed(machine, staged)
    calls, plist = machine.launchctl_calls(), machine.plist.read_bytes()
    again = installed(machine, staged)
    assert "already installed" in again.stdout
    assert machine.launchctl_calls() == calls and machine.plist.read_bytes() == plist


def test_first_install_honors_an_explicit_port(machine, staged):
    result = machine.run(staged / SUPPORT, MODEL_GATEWAY_PORT="9137")
    assert result.returncode == 0, result.stderr
    assert "MODEL_GATEWAY_PORT=9137" in (machine.app / "install.env").read_text()
    assert plistlib.loads(machine.plist.read_bytes())["EnvironmentVariables"]["MODEL_GATEWAY_PORT"] == "9137"


def test_upgrade_swaps_current_restarts_and_keeps_the_previous_release(machine, staged, tmp_path):
    installed(machine, staged)
    old = machine.current()
    port = (machine.app / "install.env").read_text()
    newer = variant(staged, tmp_path / "newer", "99.0.0")
    result = machine.run(newer)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "upgraded to 99.0.0" in result.stdout
    assert machine.current() == f"releases/{release_name(newer)}"
    assert (machine.app / old).is_dir()  # the rollback target stays
    assert (machine.app / "install.env").read_text() == port
    assert json.loads((machine.app / "endpoint.json").read_text())["gateway_version"] == "99.0.0"
    assert plistlib.loads(machine.plist.read_bytes())["ModelGatewayComponentRoot"] == str(machine.app)


def test_failed_upgrade_rolls_back_to_the_previous_release(machine, staged, tmp_path):
    installed(machine, staged)
    old = machine.current()
    newer = variant(staged, tmp_path / "newer", "99.0.0")
    (machine.state / "fail-version").write_text("99.0.0")
    result = machine.run(newer)
    assert result.returncode != 0
    assert f"rolled back to {VERSION}" in result.stderr
    assert machine.current() == old
    assert json.loads((machine.app / "endpoint.json").read_text())["gateway_version"] == VERSION
    assert (machine.state / "loaded" / LABEL).exists()


def test_failed_fresh_install_removes_the_new_launchagent_but_keeps_state(machine, staged):
    (machine.state / "fail-version").write_text(VERSION)
    result = machine.run(staged / SUPPORT)
    assert result.returncode != 0 and "failed to install" in result.stderr
    assert not machine.plist.exists() and not (machine.app / "current").exists()
    assert (machine.app / "config.yaml").exists()
    (machine.state / "fail-version").unlink()
    installed(machine, staged)  # a retry reuses the staged release
    assert (machine.state / "uv.log").read_text().count("sync ") == 1


def test_a_newer_component_install_is_never_downgraded(machine, staged, tmp_path):
    newer = variant(staged, tmp_path / "newer", "99.0.0")
    assert machine.run(newer).returncode == 0
    current, calls = machine.current(), machine.launchctl_calls()
    result = installed(machine, staged)
    assert "refusing to downgrade" in result.stdout
    assert machine.current() == current and machine.launchctl_calls() == calls


def foreign_plist(machine: Machine, working_directory: Path | str, **extra) -> bytes:
    machine.plist.parent.mkdir(parents=True, exist_ok=True)
    raw = plistlib.dumps({"Label": LABEL, "ProgramArguments": ["python", "-m", "src.main"],
                          "WorkingDirectory": str(working_directory), **extra})
    machine.plist.write_bytes(raw)
    return raw


def git_checkout(machine: Machine) -> Path:
    checkout = machine.home / "local_code/model-gateway"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "src").mkdir()
    (checkout / "src/version.py").write_text('VERSION = "0.2.1"\n')
    return checkout


@pytest.mark.parametrize("owner, setup", [
    ("a git checkout", lambda m: {"working_directory": git_checkout(m)}),
    ("Homebrew", lambda m: {"working_directory": "/opt/homebrew/Cellar/model-gateway/0.2.1/libexec"}),
    ("server-ci", lambda m: {"working_directory": m.home / "srv/model-gateway/current",
                             "HomeServerCIPath": str(m.home / "ci/bin/server-ci")}),
    ("a Model Gateway package for", lambda m: {"working_directory": m.home / "elsewhere/current",
                                               "ModelGatewayComponentRoot": str(m.home / "elsewhere")}),
])
def test_a_foreign_gateway_is_attach_only_and_never_modified(machine, staged, owner, setup):
    raw = foreign_plist(machine, **setup(machine))
    result = installed(machine, staged)
    assert f"owned by {owner}" in result.stdout and "Attach-only" in result.stdout
    assert machine.plist.read_bytes() == raw
    assert machine.launchctl_calls() == [] and not machine.app.exists()
    if owner == "a git checkout":
        assert "version:           0.2.1" in result.stdout


def test_a_legacy_home_server_gateway_is_left_for_the_home_server_package(machine, staged):
    legacy = machine.home / "Library/Application Support/HomeServer/runtime/current/model-gateway"
    raw = foreign_plist(machine, legacy, HomeServerCIPath=str(machine.home / "HomeServer/ci/bin/server-ci"))
    result = installed(machine, staged)
    assert "Legacy Home Server gateway; migration is handled by the Home Server package" in result.stdout
    assert machine.plist.read_bytes() == raw and machine.launchctl_calls() == [] and not machine.app.exists()


def test_a_symlinked_plist_is_treated_as_foreign(machine, staged, tmp_path):
    target = tmp_path / "elsewhere.plist"
    target.write_bytes(plistlib.dumps({"Label": LABEL}))
    machine.plist.parent.mkdir(parents=True)
    machine.plist.symlink_to(target)
    result = installed(machine, staged)
    assert "Attach-only" in result.stdout and machine.plist.is_symlink() and not machine.app.exists()


def test_install_fails_clearly_without_uv(machine, staged):
    result = machine.run(staged / SUPPORT, MODEL_GATEWAY_UV_BIN=str(machine.base / "missing-uv"))
    assert result.returncode != 0 and "uv is required" in result.stderr
    assert not machine.plist.exists()


def test_helper_refuses_a_tampered_package(machine, staged, tmp_path):
    support = tmp_path / "ModelGateway"
    shutil.copytree(staged / SUPPORT, support)
    path = support / "package/gateway/src/server.py"
    path.write_text(path.read_text() + "# tampered\n")
    result = machine.run(support)
    assert result.returncode != 0 and "verification failed" in result.stderr
    assert not machine.app.exists() and machine.launchctl_calls() == []


def test_a_damaged_staged_release_is_rebuilt(machine, staged, tmp_path):
    (machine.state / "fail-version").write_text(VERSION)
    assert machine.run(staged / SUPPORT).returncode != 0
    release = machine.app / "releases" / release_name(staged / SUPPORT)
    (release / "src/main.py").write_text("tampered\n")
    (machine.state / "fail-version").unlink()
    installed(machine, staged)
    assert (release / "src/main.py").read_bytes() == (staged / SUPPORT / "package/gateway/src/main.py").read_bytes()
