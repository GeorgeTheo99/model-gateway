"""Gateway-owned local AI runtime: offline, in tmp, with every system edge faked.

conftest points the state and LaunchAgents directories at tmp and makes
launchctl (other than a "not loaded" print), sysctl, uv, and the network fail.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tomllib

import pytest
import yaml
from fastapi.testclient import TestClient

from src import config_io, discovery, local_runtime as lr
import src.providers as providers
from src.server import app

ROOT = Path(__file__).resolve().parents[1]
REAL_MEMORY_BYTES = lr._memory_bytes
QWEN = lr.MODELS[lr.DEFAULT_MODEL]
QWEN_ID = "qwen3.8-27b-8bit-30gb"
GIB = 1024**3


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """A private gateway config and catalog, with admin writes enabled."""
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "auth": {
            "admin_keys": ["admin"],
            "consumer_credentials": [
                {"id": "app-runtime", "consumer": "app", "key": "runtime-token", "namespaces": ["app"],
                 "permissions": ["profiles:read", "profiles:invoke"]},
                {"id": "app-manager", "consumer": "app", "key": "manager-token", "namespaces": ["app"],
                 "permissions": ["providers:manage", "models:register"], "providers": ["fireworks"]},
                {"id": "app-local", "consumer": "app", "key": "local-token", "namespaces": ["app"],
                 "permissions": ["profiles:read", "local_ai:manage"]},
            ],
        },
        "providers": {"fireworks": {"base_url": "https://api.fireworks.ai/inference/v1"}},
    }))
    config.chmod(0o600)
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"allow_empty": True, "llm": []}))
    for module in (providers, config_io):
        monkeypatch.setattr(module, "CONFIG_PATH", config)
        monkeypatch.setattr(module, "MODEL_INFO_PATH", info)
        monkeypatch.setattr(module, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    providers.reload()
    return config


@pytest.fixture
def eligible(monkeypatch):
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)
    monkeypatch.setattr(lr, "_memory_bytes", lambda: 64 * GIB)


@pytest.fixture
def launchctl(monkeypatch):
    """Record launchctl calls; ``loaded`` names labels whose print succeeds."""
    calls, loaded = [], set()

    def fake(*args):
        calls.append(args)
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[1]
            return subprocess.CompletedProcess(args, 0 if label in loaded else 113, "state = running\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(lr, "_launchctl", fake)
    fake.calls, fake.loaded = calls, loaded
    return fake


def write_plist(path: Path, value: dict) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = plistlib.dumps(value)
    path.write_bytes(raw)
    return raw


def external_omlx(**extra) -> bytes:
    return write_plist(lr.omlx_plist_path(), {"Label": lr.OMLX_LABEL, "ProgramArguments": ["omlx", "serve"], **extra})


def owned_install(port: int = 9123) -> None:
    write_plist(lr.omlx_plist_path(), lr.omlx_plist(port))
    (lr.mlx_dir() / QWEN_ID).mkdir(parents=True)


# ── status and ownership ─────────────────────────────────────────────────────


def test_status_on_a_mac_that_cannot_run_local_ai():
    assert lr.status() == {"eligible": False, "installed": False, "managed": True, "model": None, "state": None,
                           "bytes_done": 0, "bytes_total": 0, "message": "", "base_url": None}
    assert not lr.runtime_root().exists()  # reading status creates nothing


def test_status_when_eligible_and_when_installed(eligible):
    assert lr.status()["eligible"] is True
    owned_install(9123)
    status = lr.status()
    assert (status["installed"], status["managed"], status["model"], status["state"], status["base_url"]) == (
        True, True, "qwen3.8-27b", "done", "http://127.0.0.1:9123")


@pytest.mark.parametrize("owner", [None, "/somewhere/else"])
def test_an_external_omlx_is_reported_unmanaged(eligible, owner):
    external_omlx(**({lr.OWNER_KEY: owner} if owner else {}))
    (lr.mlx_dir() / QWEN_ID).mkdir(parents=True)  # even with a payload, it is not ours
    status = lr.status()
    assert (status["managed"], status["installed"], status["base_url"]) == (False, False, None)
    assert "managed outside Model Gateway" in status["message"]


def test_an_omlx_loaded_from_elsewhere_is_external(launchctl):
    launchctl.loaded.add(lr.OMLX_LABEL)
    assert lr.omlx_owner() == "external"
    assert lr.status()["managed"] is False


def test_add_and_remove_refuse_an_external_omlx_without_touching_it(eligible, gateway, launchctl):
    raw = external_omlx()
    for operation in (lr.start_add, lr.remove):
        with pytest.raises(lr.LocalRuntimeError, match="managed outside Model Gateway"):
            operation()
    assert lr.omlx_plist_path().read_bytes() == raw
    assert all(call[0] == "print" for call in launchctl.calls)
    assert not lr.runtime_root().exists()


def test_eligibility_needs_48_gib_and_uses_absolute_sysctl(monkeypatch):
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)
    for memory, expected in ((48 * GIB, True), (48 * GIB - 1, False), (16 * GIB, False)):
        calls = []
        monkeypatch.setattr(lr.subprocess, "check_output", lambda args, **kw: calls.append(args) or f"{memory}\n")
        monkeypatch.setattr(lr, "_memory_bytes", REAL_MEMORY_BYTES)
        assert lr.eligible() is expected
        # Absolute, because launchd jobs and the gateway have no /usr/sbin on PATH.
        assert calls == [["/usr/sbin/sysctl", "-n", "hw.memsize"]]
    monkeypatch.setattr(lr, "_apple_silicon", lambda: False)
    assert lr.eligible() is False


def test_interrupted_when_an_active_job_stopped_heartbeating(eligible):
    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(timespec="seconds")
    lr._atomic(lr.status_path(), json.dumps({"state": "downloading", "model": "qwen3.8-27b",
                                             "bytes_done": 5, "bytes_total": 10, "updated_at": old}))
    assert lr.status()["state"] == "interrupted"
    lr._atomic(lr.status_path(), json.dumps({"state": "downloading", "updated_at": lr._now()}))
    assert lr.status()["state"] == "downloading"


# ── download ─────────────────────────────────────────────────────────────────


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int, headers: dict):
        super().__init__(body)
        self.status, self.headers = status, headers


class FakeOpener:
    """Serves one fixture file, honouring (or ignoring) byte ranges; ``cut`` drops the connection early."""

    def __init__(self, body: bytes, *, honour_range: bool = True, cut: int | None = None):
        self.body, self.honour_range, self.cut, self.requests = body, honour_range, cut, []

    def open(self, request, timeout):
        self.requests.append(request.get_header("Range"))
        start = int(request.get_header("Range")[6:-1]) if request.get_header("Range") and self.honour_range else 0
        headers = {"Content-Range": f"bytes {start}-{len(self.body) - 1}/{len(self.body)}"} if start else {}
        return FakeResponse(self.body[start:self.cut], 206 if start else 200, headers)


BODY = b"model weights fixture" * 1000
ENTRY = {"path": "model.safetensors", "size_bytes": len(BODY), "sha256": hashlib.sha256(BODY).hexdigest()}


@pytest.fixture
def progress():
    return lr.Progress(QWEN, len(BODY))


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(lr.time, "sleep", calls.append)
    return calls


def download(progress, opener, path=None):
    path = path or lr.runtime_root() / "partial/model.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    lr.download_file("https://host/model", path, ENTRY, progress, 0, opener)
    return path


def test_download_writes_and_verifies_a_fresh_private_file(progress, sleeps):
    opener = FakeOpener(BODY)
    path = download(progress, opener)
    assert path.read_bytes() == BODY and path.stat().st_mode & 0o777 == 0o600
    assert opener.requests == [None]
    assert progress.value["bytes_done"] == len(BODY)


def test_download_resumes_a_partial_file_with_a_range(progress, sleeps):
    path = lr.runtime_root() / "partial/model.safetensors"
    lr._atomic(path, BODY[:5000])
    opener = FakeOpener(BODY)
    download(progress, opener, path)
    assert (path.read_bytes(), opener.requests) == (BODY, ["bytes=5000-"])


def test_a_dropped_connection_keeps_the_partial_file_for_the_next_attempt(progress, sleeps):
    path = lr.runtime_root() / "partial/model.safetensors"
    with pytest.raises(RuntimeError, match="closed the connection early"):
        download(progress, FakeOpener(BODY, cut=7000), path)
    assert path.read_bytes() == BODY[:7000]
    opener = FakeOpener(BODY)
    download(progress, opener, path)
    assert (path.read_bytes(), opener.requests) == (BODY, ["bytes=7000-"])


def test_a_resume_from_the_wrong_offset_is_refused(progress, sleeps):
    path = lr.runtime_root() / "partial/model.safetensors"
    lr._atomic(path, BODY[:5000])
    opener = FakeOpener(BODY)
    opener.open = lambda request, timeout: FakeResponse(BODY[10:], 206, {"Content-Range": "bytes 10-20999/21000"})
    with pytest.raises(RuntimeError, match="unexpected response"):
        download(progress, opener, path)
    assert path.read_bytes() == BODY[:5000]


def test_download_restarts_when_the_host_ignores_the_range(progress, sleeps):
    path = lr.runtime_root() / "partial/model.safetensors"
    lr._atomic(path, BODY[:5000])
    download(progress, FakeOpener(BODY, honour_range=False), path)
    assert path.read_bytes() == BODY


def test_a_complete_file_is_hash_verified_without_downloading(progress, sleeps):
    path = lr.runtime_root() / "partial/model.safetensors"
    lr._atomic(path, BODY)
    opener = FakeOpener(BODY)
    download(progress, opener, path)
    assert opener.requests == []


def test_corrupt_or_oversized_downloads_are_removed(progress, sleeps):
    path = lr.runtime_root() / "partial/model.safetensors"
    with pytest.raises(RuntimeError, match="did not match"):
        download(progress, FakeOpener(b"x" * len(BODY)), path)
    assert not path.exists()
    with pytest.raises(RuntimeError, match="more data"):
        download(progress, FakeOpener(BODY + b"extra"), path)
    assert not path.exists()


def test_download_is_rate_limited(progress, sleeps, monkeypatch):
    monkeypatch.setattr(lr, "DOWNLOAD_BYTES_PER_SECOND", 1000)
    download(progress, FakeOpener(BODY))
    # Fake sleeps take no time, so the limiter asks for the whole budget: 21000 bytes at 1000 B/s.
    assert sleeps and 20 < sleeps[-1] <= len(BODY) / 1000
    sleeps.clear()
    monkeypatch.setattr(lr, "DOWNLOAD_BYTES_PER_SECOND", 25_000_000)
    download(progress, FakeOpener(BODY), lr.runtime_root() / "partial/other.safetensors")
    assert all(delay < 0.01 for delay in sleeps)


def test_payload_download_retries_then_reports_a_bounded_error(progress, sleeps, monkeypatch):
    manifest = {"model_id": QWEN_ID, "hf_repo": "org/model", "hf_revision": "abc123",
                "total_bytes": len(BODY), "files": [ENTRY]}
    calls = []

    def failing(url, *args):
        calls.append(url)
        raise OSError("network down")

    monkeypatch.setattr(lr, "_download_opener", lambda: object())
    monkeypatch.setattr(lr, "download_file", failing)
    with pytest.raises(RuntimeError, match="Could not download the local AI model"):
        lr.download_payload(manifest, progress)
    assert calls == ["https://huggingface.co/org/model/resolve/abc123/model.safetensors"] * lr.DOWNLOAD_ATTEMPTS
    assert (lr.mlx_dir() / f"{QWEN_ID}.partial").is_dir()
    # An already promoted payload is never downloaded again.
    (lr.mlx_dir() / QWEN_ID).mkdir()
    calls.clear()
    lr.download_payload(manifest, progress)
    assert calls == [] and progress.value["bytes_done"] == len(BODY)


def test_redirects_must_stay_on_https():
    handler = lr.HttpsOnlyRedirect()
    request = lr.urllib.request.Request("https://huggingface.co/x")
    with pytest.raises(lr.urllib.error.HTTPError):
        handler.redirect_request(request, None, 302, "Found", {}, "http://cdn.example/x")
    assert handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.example/x").full_url == \
        "https://cdn.example/x"


# ── LaunchAgents ─────────────────────────────────────────────────────────────


def test_omlx_launch_agent_is_loopback_offline_bounded_and_owned():
    value = plistlib.loads(plistlib.dumps(lr.omlx_plist(9123)))
    args = value["ProgramArguments"]
    root = lr.runtime_root()
    assert value["Label"] == "com.local.omlx"
    assert value[lr.OWNER_KEY] == str(lr.state_dir())
    assert args[:2] == [str(root / f"omlx-{lr.OMLX_VERSION}/bin/omlx"), "serve"]
    expected = {"--base-path": str(root / "inference"), "--model-dir": str(root / "models/mlx"),
                "--host": "127.0.0.1", "--port": "9123", "--max-concurrent-requests": "1",
                "--memory-guard": "safe", "--paged-ssd-cache-dir": str(root / "models/cache"),
                "--paged-ssd-cache-max-size": "8GB"}
    assert {flag: args[args.index(flag) + 1] for flag in expected} == expected
    assert "--no-hf-cache" in args and "--api-key" not in args
    assert value["EnvironmentVariables"]["HF_HUB_OFFLINE"] == "1"
    assert (value["Umask"], value["KeepAlive"], value["RunAtLoad"]) == (0o77, True, True)


def test_bootstrap_retries_an_asynchronous_bootout(launchctl, sleeps, monkeypatch):
    launchctl.loaded.add(lr.OMLX_LABEL)
    results = iter([5, 0])
    real = launchctl

    def flaky(*args):
        if args[0] == "bootstrap":
            real.calls.append(args)
            return subprocess.CompletedProcess(args, next(results), "", "private diagnostic")
        return real(*args)

    monkeypatch.setattr(lr, "_launchctl", flaky)
    lr._bootstrap(Path("/tmp/x.plist"), lr.OMLX_LABEL)
    assert [call[0] for call in launchctl.calls] == ["print", "bootout", "bootstrap", "bootstrap"]
    assert sleeps == [1]
    results = iter([5] * 30)
    with pytest.raises(lr.LocalRuntimeError, match="bootstrap failed") as error:
        lr._bootstrap(Path("/tmp/x.plist"), lr.OMLX_LABEL)
    assert "private diagnostic" not in str(error.value)


def queue(monkeypatch, *, free=10**15):
    bootstraps = []
    monkeypatch.setattr(lr, "_bootstrap", lambda plist, label: bootstraps.append((plist, label)))
    monkeypatch.setattr(lr, "_disk_free", lambda path: free)
    return bootstraps


def test_add_queues_a_one_shot_job_outside_launch_agents(eligible, gateway, launchctl, monkeypatch):
    bootstraps = queue(monkeypatch)
    status = lr.start_add()
    assert (status["state"], status["model"], status["bytes_total"]) == ("queued", "qwen3.8-27b", 29531521322)
    plist = lr.runtime_root() / f"launchd/{lr.SETUP_LABEL}.plist"
    assert bootstraps == [(plist, lr.SETUP_LABEL)]
    job = plistlib.loads(plist.read_bytes())
    assert job["ProgramArguments"] == ["/bin/bash", str(lr.PACKAGE_ROOT / "bin/model-gateway"),
                                       "_local-ai-job", "qwen3.8-27b"]
    assert "KeepAlive" not in job and job[lr.OWNER_KEY] == str(lr.state_dir())
    env = job["EnvironmentVariables"]
    assert env["MODEL_GATEWAY_CONFIG"] == str(gateway)
    assert env["MODEL_GATEWAY_STATE_DIR"] == str(lr.state_dir())
    assert "/usr/sbin" in env["PATH"]
    assert plist.stat().st_mode & 0o777 == 0o600
    assert not (lr.launch_agents_dir() / f"{lr.SETUP_LABEL}.plist").exists()
    assert lr.read_status()["state"] == "queued"


def test_add_refuses_ineligible_macs_full_disks_and_conflicts(gateway, launchctl, monkeypatch):
    bootstraps = queue(monkeypatch, free=lr.DISK_MARGIN)
    with pytest.raises(lr.LocalRuntimeError, match="48 GB"):
        lr.start_add()
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)
    monkeypatch.setattr(lr, "_memory_bytes", lambda: 64 * GIB)
    with pytest.raises(lr.LocalRuntimeError, match="free disk space"):
        lr.start_add()
    monkeypatch.setattr(lr, "_disk_free", lambda path: 10**15)
    monkeypatch.setattr(lr, "_port_in_use", lambda port: port == 9110)
    with pytest.raises(lr.LocalRuntimeError, match="Port 9110 is in use"):
        lr.start_add()
    with pytest.raises(lr.LocalRuntimeError, match="Unknown local model"):
        lr.start_add("other")
    monkeypatch.setattr(lr, "_port_in_use", lambda port: False)
    config = yaml.safe_load(gateway.read_text())
    config["providers"]["omlx"] = {"base_url": "http://192.168.1.5:9110/v1"}
    gateway.write_text(yaml.safe_dump(config))
    with pytest.raises(lr.LocalRuntimeError, match="oMLX provider"):
        lr.start_add()
    del config["providers"]["omlx"]
    gateway.write_text(yaml.safe_dump(config))
    config_io.MODEL_INFO_PATH.write_text(json.dumps({"llm": [
        {"name": "qwen3.8-27b", "provider": "fireworks", "provider_model_id": "qwen"}]}))
    with pytest.raises(lr.LocalRuntimeError, match="different model named"):
        lr.start_add()
    assert bootstraps == []


def test_add_reuses_a_running_job(eligible, gateway, launchctl, monkeypatch):
    bootstraps = queue(monkeypatch)
    launchctl.loaded.add(lr.SETUP_LABEL)
    lr.start_add()
    assert bootstraps == []


def test_a_failed_launch_is_recorded_rather_than_left_queued(eligible, gateway, launchctl, monkeypatch):
    queue(monkeypatch)
    monkeypatch.setattr(lr, "_bootstrap", lambda *a: (_ for _ in ()).throw(lr.LocalRuntimeError("bootstrap failed (exit 5)")))
    with pytest.raises(lr.LocalRuntimeError):
        lr.start_add()
    assert lr.read_status()["state"] == "failed"


def test_cancel_stops_a_download_but_not_an_install(launchctl):
    lr._atomic(lr.status_path(), json.dumps({"state": "downloading", "updated_at": lr._now()}))
    launchctl.loaded.add(lr.SETUP_LABEL)
    assert lr.cancel()["state"] == "cancelled"
    assert ("bootout", f"gui/{os.getuid()}/{lr.SETUP_LABEL}") in launchctl.calls
    launchctl.calls.clear()
    lr._atomic(lr.status_path(), json.dumps({"state": "installing", "updated_at": lr._now()}))
    with lr.provision_lock(wait=True):
        with pytest.raises(lr.LocalRuntimeError, match="can no longer be cancelled"):
            lr.cancel()
    assert not [call for call in launchctl.calls if call[0] == "bootout"]
    assert lr.read_status()["state"] == "installing"


def test_cancel_without_any_local_ai_creates_nothing():
    assert lr.cancel()["state"] is None
    assert not lr.runtime_root().exists()


# ── the setup job ────────────────────────────────────────────────────────────


def test_job_downloads_then_installs_and_checks(eligible, monkeypatch):
    events = []
    monkeypatch.setattr(lr, "download_payload", lambda payload, progress: events.append("download"))
    monkeypatch.setattr(lr, "install", lambda local, payload: events.append(("install", lr.read_status()["state"])))
    monkeypatch.setattr(lr, "check", lambda local, payload: events.append("check"))
    handler = lr.signal.getsignal(lr.signal.SIGTERM)
    lr.run_job("qwen3.8-27b")
    assert events == ["download", ("install", "installing"), "check"]
    assert lr.read_status()["state"] == "done"
    assert lr.signal.getsignal(lr.signal.SIGTERM) is handler


def test_a_cancelled_download_records_cancelled_and_installs_nothing(eligible, monkeypatch):
    monkeypatch.setattr(lr, "download_payload", lambda *a: (_ for _ in ()).throw(lr.Cancelled()))
    monkeypatch.setattr(lr, "install", lambda *a: pytest.fail("installed after cancel"))
    lr.run_job("qwen3.8-27b")
    assert lr.read_status()["state"] == "cancelled"


def test_stopping_during_install_records_a_repairable_failure(eligible, monkeypatch):
    monkeypatch.setattr(lr, "download_payload", lambda *a: None)
    monkeypatch.setattr(lr, "install", lambda *a: (_ for _ in ()).throw(lr.Cancelled()))
    lr.run_job("qwen3.8-27b")
    status = lr.read_status()
    assert status["state"] == "failed" and "add local AI again" in status["message"]


def test_job_records_a_failure(eligible, monkeypatch):
    monkeypatch.setattr(lr, "download_payload",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("Could not download the local AI model")))
    with pytest.raises(RuntimeError):
        lr.run_job("qwen3.8-27b")
    status = lr.read_status()
    assert (status["state"], status["message"]) == ("failed", "Could not download the local AI model")


def test_job_refuses_an_external_omlx_and_ineligible_macs(monkeypatch):
    with pytest.raises(lr.LocalRuntimeError, match="48 GB"):
        lr.run_job("qwen3.8-27b")
    assert lr.read_status()["state"] == "failed"
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)
    monkeypatch.setattr(lr, "_memory_bytes", lambda: 64 * GIB)
    external_omlx()
    with pytest.raises(lr.LocalRuntimeError, match="managed outside"):
        lr.run_job("qwen3.8-27b")


def test_cli_job_reports_failures_without_a_traceback(eligible, monkeypatch, capsys):
    monkeypatch.setattr(lr, "download_payload", lambda *a: (_ for _ in ()).throw(RuntimeError("network down")))
    assert lr.main(["job", "qwen3.8-27b"]) == 2
    assert "error: network down" in capsys.readouterr().err


# ── install and gateway wiring ───────────────────────────────────────────────


def tiny_payload() -> dict:
    """A verified two-file payload staged in .partial, plus its manifest."""
    partial = lr.mlx_dir() / f"{QWEN_ID}.partial"
    partial.mkdir(parents=True)
    files = {"model-00001.safetensors": b"weights",
             "model.safetensors.index.json": json.dumps({"weight_map": {"a": "model-00001.safetensors"}}).encode()}
    entries = []
    for name, data in files.items():
        (partial / name).write_bytes(data)
        entries.append({"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return {"schema_version": 1, "model_id": QWEN_ID, "hf_repo": "org/model", "hf_revision": "abc", "license": "x",
            "total_bytes": sum(e["size_bytes"] for e in entries), "files": entries}


def test_install_verifies_builds_starts_and_wires_the_gateway(eligible, gateway, launchctl, monkeypatch):
    payload = tiny_payload()
    runs, health = [], []
    monkeypatch.setattr(lr, "manifest", lambda local: payload)
    monkeypatch.setattr(lr, "_run", lambda *args, env=None: runs.append((args, env and env.get("UV_PROJECT_ENVIRONMENT"))))
    monkeypatch.setattr(lr, "_wait_health", health.append)
    monkeypatch.setattr(lr, "_uv", lambda: "/fake/uv")
    launchctl.loaded.add(lr.GATEWAY_LABEL)
    lr.install(QWEN, payload)

    assert (lr.mlx_dir() / QWEN_ID).is_dir() and not (lr.mlx_dir() / f"{QWEN_ID}.partial").exists()
    assert runs[0] == (("/fake/uv", "sync", "--locked", "--no-dev", "--project", str(lr.RUNTIME_PROJECT)),
                       str(lr.venv_dir()))
    assert runs[1][0][:2] == (str(lr.venv_dir() / "bin/python"), "-c")
    key = lr.api_key_path()
    assert key.stat().st_mode & 0o777 == 0o600
    settings = json.loads((lr.inference_dir() / "model_settings.json").read_text())
    assert settings["models"][QWEN_ID]["is_default"] is True
    agent = plistlib.loads(lr.omlx_plist_path().read_bytes())
    assert agent[lr.OWNER_KEY] == str(lr.state_dir()) and lr.omlx_owner() == "gateway"
    assert ("bootstrap", f"gui/{os.getuid()}", str(lr.omlx_plist_path())) in launchctl.calls
    assert ("kickstart", "-k", f"gui/{os.getuid()}/{lr.GATEWAY_LABEL}") in launchctl.calls
    assert health == ["http://127.0.0.1:9110/health", "http://127.0.0.1:9111/health"]

    config = yaml.safe_load(gateway.read_text())
    assert config["providers"]["omlx"] == {"base_url": "http://127.0.0.1:9110/v1", "protocol": "openai",
                                           "api_key": "", "api_key_file": str(key), "enabled": True}
    assert config["providers"]["fireworks"]  # untouched
    assert key.read_text().strip() not in gateway.read_text()
    (entry,) = json.loads(config_io.MODEL_INFO_PATH.read_text())["llm"]
    assert (entry["name"], entry["provider"], entry["omlx_id"], entry["pricing_status"]) == (
        "qwen3.8-27b", "omlx", QWEN_ID, "unmetered")
    providers.reload()
    routed = providers.resolve("qwen3.8-27b")
    assert (routed.base_url, routed.api_key, routed.provider_model_id) == (
        "http://127.0.0.1:9110/v1", key.read_text().strip(), QWEN_ID)
    assert lr.status()["installed"] is True


def test_install_rejects_a_payload_that_fails_verification(eligible, gateway, monkeypatch):
    payload = tiny_payload()
    (lr.mlx_dir() / f"{QWEN_ID}.partial/model-00001.safetensors").write_bytes(b"tampered")
    with pytest.raises(lr.LocalRuntimeError, match="failed verification"):
        lr.install(QWEN, payload)
    assert not (lr.mlx_dir() / QWEN_ID).exists()


def test_gateway_wiring_rolls_back_when_the_gateway_does_not_restart(gateway, launchctl, monkeypatch):
    monkeypatch.setattr(lr, "manifest", lambda local: {"model_id": QWEN_ID})
    before = (gateway.read_text(), config_io.MODEL_INFO_PATH.read_text())
    with pytest.raises(lr.LocalRuntimeError, match="not running"):
        lr.configure_gateway(QWEN, 9110)
    assert (gateway.read_text(), config_io.MODEL_INFO_PATH.read_text()) == before


def test_check_requires_the_sole_payload_and_a_real_completion(eligible, monkeypatch):
    owned_install(9123)
    lr._key(lr.api_key_path(), create=True)
    payload = {"model_id": QWEN_ID}
    path = str(lr.mlx_dir() / QWEN_ID)
    replies = {
        "http://127.0.0.1:9123/v1/models/status": {"models": [{"id": QWEN_ID, "model_path": path}]},
        "http://127.0.0.1:9111/v1/models": {"data": [{"id": "qwen3.8-27b"}]},
        "http://127.0.0.1:9111/v1/chat/completions": {"choices": [{"message": {"content": "OK"}}]},
    }
    seen = []
    monkeypatch.setattr(lr, "_request", lambda url, token="", body=None, **kw: seen.append((url, body)) or replies[url])
    lr.check(QWEN, payload)
    assert seen[-1][1]["model"] == "qwen3.8-27b"
    replies["http://127.0.0.1:9111/v1/chat/completions"] = {"choices": [{"message": {"content": "Sure!"}}]}
    with pytest.raises(lr.LocalRuntimeError, match="completion check"):
        lr.check(QWEN, payload)
    replies["http://127.0.0.1:9123/v1/models/status"] = {"models": [{"id": QWEN_ID, "model_path": path},
                                                                    {"id": "other", "model_path": "/x"}]}
    with pytest.raises(lr.LocalRuntimeError, match="sole verified"):
        lr.check(QWEN, payload)


def test_remove_unwires_only_its_own_routes_and_deletes_local_state(eligible, gateway, launchctl, monkeypatch):
    owned_install(9110)
    key = lr._key(lr.api_key_path(), create=True)
    config = yaml.safe_load(gateway.read_text())
    config["providers"]["omlx"] = {"base_url": "http://127.0.0.1:9110/v1", "api_key_file": str(lr.api_key_path())}
    gateway.write_text(yaml.safe_dump(config))
    config_io.MODEL_INFO_PATH.write_text(json.dumps({"llm": [
        {"name": "qwen3.8-27b", "provider": "omlx", "omlx_id": QWEN_ID, "pricing_status": "unmetered"},
        {"name": "glm", "provider": "fireworks", "provider_model_id": "glm"}]}))
    launchctl.loaded.update({lr.OMLX_LABEL, lr.GATEWAY_LABEL})
    monkeypatch.setattr(lr, "_wait_health", lambda url: None)
    assert lr.remove()["installed"] is False
    assert "omlx" not in yaml.safe_load(gateway.read_text())["providers"]
    assert [row["name"] for row in json.loads(config_io.MODEL_INFO_PATH.read_text())["llm"]] == ["glm"]
    assert ("bootout", f"gui/{os.getuid()}/{lr.OMLX_LABEL}") in launchctl.calls
    assert not lr.omlx_plist_path().exists() and not lr.runtime_root().exists()
    assert key not in gateway.read_text()


# ── discovery, admin API, credentials, and package data ──────────────────────


def test_endpoint_document_reports_the_local_runtime_without_keys(eligible, gateway):
    assert discovery.endpoint_document()["local_runtime"] == {"managed": True, "base_url": None, "health_url": None}
    owned_install(9123)
    key = lr._key(lr.api_key_path(), create=True)
    document = discovery.endpoint_document()
    assert document["local_runtime"] == {"managed": True, "base_url": "http://127.0.0.1:9123",
                                         "health_url": "http://127.0.0.1:9123/health"}
    assert key not in json.dumps(document)
    lr.omlx_plist_path().write_bytes(plistlib.dumps({"Label": lr.OMLX_LABEL}))
    assert discovery.endpoint_document()["local_runtime"] == {"managed": False, "base_url": None, "health_url": None}


def test_admin_local_ai_needs_local_ai_manage_or_a_full_admin(gateway, monkeypatch):
    calls = []
    monkeypatch.setattr(lr, "start_add", lambda model=None: calls.append(("add", model)) or {"state": "queued"})
    monkeypatch.setattr(lr, "cancel", lambda: calls.append(("cancel",)) or {"state": "cancelled"})
    client = TestClient(app)
    for token in ("runtime-token", "manager-token"):
        headers = {"Authorization": f"Bearer {token}"}
        assert client.get("/admin/api/local-ai", headers=headers).status_code == 403
        assert client.post("/admin/api/local-ai", headers=headers, json={"action": "add"}).status_code == 403
    assert client.get("/admin/api/local-ai").status_code == 401
    for token in ("local-token", "admin"):
        headers = {"Authorization": f"Bearer {token}"}
        status = client.get("/admin/api/local-ai", headers=headers)
        assert status.status_code == 200 and status.json()["managed"] is True
        assert client.post("/admin/api/local-ai", headers=headers, json={"action": "add"}).json() == {"state": "queued"}
        assert client.post("/admin/api/local-ai", headers=headers, json={"action": "cancel"}).status_code == 200
    assert calls == [("add", None), ("cancel",)] * 2
    local = {"Authorization": "Bearer local-token"}
    assert client.post("/admin/api/local-ai", headers=local, json={"action": "remove"}).status_code == 400
    # The scoped credential reaches nothing else in the admin API.
    assert client.get("/admin/api/providers", headers=local).status_code == 403
    assert client.get("/admin/api/models", headers=local).status_code == 401
    assert set(client.get("/admin/api/status", headers=local).json()) == {
        "service", "status", "writes_enabled", "version", "capabilities"}


def test_admin_local_ai_reports_refusals_and_respects_read_only_mode(gateway, monkeypatch):
    client = TestClient(app)
    local = {"Authorization": "Bearer local-token"}
    refused = client.post("/admin/api/local-ai", headers=local, json={"action": "add"})
    assert refused.status_code == 409 and "48 GB" in refused.json()["error"]["message"]
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "false")
    assert client.post("/admin/api/local-ai", headers=local, json={"action": "add"}).status_code == 403


def test_local_ai_scope_needs_no_provider_allowlist(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("auth:\n  client_keys: [client-token]\nproviders: {}\n")
    config.chmod(0o600)
    env = {key: value for key, value in os.environ.items() if not key.startswith("MODEL_GATEWAY_CLIENT_KEYS")}
    env["MODEL_GATEWAY_BACKUP_DIR"] = str(tmp_path / "backups")
    added = subprocess.run([sys.executable, str(ROOT / "scripts/consumers.py"), "--config", str(config),
                            "add", "app", "--role", "runtime", "--local-ai"],
                           env=env, capture_output=True, text=True, timeout=60)
    assert added.returncode == 0, added.stderr
    (entry,) = yaml.safe_load(config.read_text())["auth"]["consumer_credentials"]
    assert entry["permissions"] == ["profiles:read", "profiles:invoke", "local_ai:manage"]
    assert "providers" not in entry


def test_cli_usage_lists_local_ai_and_rejects_unknown_subcommands(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_GATEWAY_", "GATEWAY_VISION"))}
    env.update(HOME=str(tmp_path), MODEL_GATEWAY_LAUNCHD_LABEL="com.local.model-gateway-test-does-not-exist")
    cli = str(ROOT / "bin/model-gateway")
    usage = subprocess.run([cli, "help"], capture_output=True, text=True, env=env)
    assert "model-gateway local-ai status|add|cancel|remove" in usage.stdout
    bogus = subprocess.run([cli, "local-ai", "bogus"], capture_output=True, text=True, env=env)
    assert bogus.returncode == 1 and "usage: model-gateway local-ai" in bogus.stderr
    helped = subprocess.run([cli, "local-ai", "--help"], capture_output=True, text=True, env=env)
    assert helped.returncode == 0 and "add [MODEL]" in helped.stdout
    assert not (tmp_path / "Library").exists()


def test_shipped_payload_manifest_and_locked_runtime_match_the_code():
    payload = lr.manifest(QWEN)
    assert (payload["model_id"], payload["license"]) == (QWEN_ID, "apache-2.0")
    assert (lr.MODELS_DIR / "LICENSE-Qwen-Apache-2.0.txt").is_file() and (lr.MODELS_DIR / "NOTICE.md").is_file()
    project = tomllib.loads((lr.RUNTIME_PROJECT / "pyproject.toml").read_text())
    assert project["project"]["dependencies"] == [f"omlx=={lr.OMLX_VERSION}"]
    lock = (lr.RUNTIME_PROJECT / "uv.lock").read_text()
    assert f'name = "omlx"\nversion = "{lr.OMLX_VERSION}"' in lock
    assert 'name = "model-gateway-local-runtime"' in lock
