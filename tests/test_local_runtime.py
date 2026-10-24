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
import shutil
import subprocess
import sys
import tomllib

import pytest
import yaml
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src import config_io, discovery, local_runtime as lr
import src.providers as providers
from src.server import app

ROOT = Path(__file__).resolve().parents[1]
REAL_MEMORY_BYTES = lr._memory_bytes
REAL_DOWNLOAD_OPENER = lr._download_opener
REAL_MANIFEST = lr.manifest
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
    # As the CLI and the gateway's own LaunchAgent set it.
    monkeypatch.setenv("MODEL_GATEWAY_LAUNCHD_LABEL", lr.GATEWAY_LABEL)
    providers.reload()
    return config


@pytest.fixture
def eligible(monkeypatch):
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)
    monkeypatch.setattr(lr, "_memory_bytes", lambda: 64 * GIB)


@pytest.fixture
def launchctl(monkeypatch):
    """Record launchctl calls; ``loaded`` names labels whose print succeeds.

    A loaded job reports this gateway's plist path unless ``paths`` overrides it.
    """
    calls, loaded, paths = [], set(), {}

    def fake(*args):
        calls.append(args)
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[1]
            default = {lr.OMLX_LABEL: lr.omlx_plist_path(), lr.SETUP_LABEL: lr.setup_plist_path()}.get(
                label, lr.launch_agents_dir() / f"{label}.plist")
            output = f"{args[1]} = {{\n\tpath = {paths.get(label, default)}\n\tstate = running\n}}\n"
            return subprocess.CompletedProcess(args, 0 if label in loaded else 113, output, "")
        if args[0] == "bootout":
            loaded.discard(args[1].rsplit("/", 1)[1])
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(lr, "_launchctl", fake)
    fake.calls, fake.loaded, fake.paths = calls, loaded, paths
    return fake


def qwen_manifest(payload: dict):
    """``lr.manifest`` with ``payload`` for the default model; other models keep their own."""
    return lambda local: payload if local.name == QWEN.name else REAL_MANIFEST(local)


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
    value = lr.status()
    models = value.pop("models")
    assert (value.pop("auto_install"), value.pop("auto_install_error")) == ([], "")
    assert value == {"eligible": False, "installed": False, "managed": True, "model": None, "state": None,
                     "bytes_done": 0, "bytes_total": 0, "message": "", "base_url": None, "loaded": None}
    assert models == [{"name": name, "eligible": False, "installed": False, "min_memory_gb": gb, "loaded": None,
                       "source": "release"}
                      for name, gb in (("qwen3.8-27b", 48), ("qwen3.8-27b-4bit", 32))]
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


def test_a_second_gateway_never_treats_this_gateways_runtime_as_its_own(eligible, gateway, launchctl,
                                                                        monkeypatch):
    owned_install(9123)
    lr._atomic(lr.setup_plist_path(), plistlib.dumps(lr.setup_plist(QWEN, 9123)))
    assert lr.omlx_owner() == "gateway"
    # Same state directory, another LaunchAgent label: another gateway.
    monkeypatch.setenv("MODEL_GATEWAY_LAUNCHD_LABEL", "com.local.model-gateway-second")
    assert lr.omlx_owner() == "external" and lr.status()["managed"] is False
    for operation in (lr.start_add, lr.remove):
        with pytest.raises(lr.LocalRuntimeError, match="managed outside Model Gateway"):
            operation()
    lr.omlx_plist_path().unlink()
    for operation in (lr.start_add, lr.cancel, lr.remove):
        with pytest.raises(lr.LocalRuntimeError, match="belongs to another Model Gateway"):
            operation()
    assert all(call[0] == "print" for call in launchctl.calls)
    assert (lr.mlx_dir() / QWEN_ID).is_dir()


def test_add_refuses_to_guess_the_label_of_an_unlabelled_gateway(eligible, gateway, launchctl, monkeypatch):
    bootstraps = queue(monkeypatch)
    monkeypatch.delenv("MODEL_GATEWAY_LAUNCHD_LABEL")
    with pytest.raises(lr.LocalRuntimeError, match="label is unknown"):
        lr.start_add()
    # The default LaunchAgent runs another config: still not this gateway.
    default = lr.launch_agents_dir() / f"{lr.GATEWAY_LABEL}.plist"
    write_plist(default, {"Label": lr.GATEWAY_LABEL, "EnvironmentVariables": {"MODEL_GATEWAY_CONFIG": "/other.yaml"}})
    with pytest.raises(lr.LocalRuntimeError, match="label is unknown"):
        lr.start_add()
    assert bootstraps == []
    # As `model-gateway install` records it (this test's state directory is not the default).
    write_plist(default, {"Label": lr.GATEWAY_LABEL, "EnvironmentVariables": {
        "MODEL_GATEWAY_CONFIG": str(gateway), "MODEL_GATEWAY_STATE_DIR": str(lr.state_dir())}})
    assert lr.start_add()["state"] == "queued" and len(bootstraps) == 1


def test_eligibility_is_per_model_and_uses_absolute_sysctl(monkeypatch):
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)
    small = lr.MODELS["qwen3.8-27b-4bit"]
    for memory, any_model, eight_bit, preferred in ((48 * GIB, True, True, QWEN), (48 * GIB - 1, True, False, small),
                                                   (32 * GIB, True, False, small), (32 * GIB - 1, False, False, QWEN)):
        calls = []
        monkeypatch.setattr(lr.subprocess, "check_output", lambda args, **kw: calls.append(args) or f"{memory}\n")
        monkeypatch.setattr(lr, "_memory_bytes", REAL_MEMORY_BYTES)
        assert lr.eligible() is any_model
        assert lr.eligible(QWEN) is eight_bit
        assert lr.preferred_model() == preferred  # the default when nothing fits
        # Absolute, because launchd jobs and the gateway have no /usr/sbin on PATH.
        assert calls[0] == ["/usr/sbin/sysctl", "-n", "hw.memsize"]
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
    monkeypatch.setattr(lr, "DOWNLOAD_BYTES_PER_SECOND", 18_750_000)
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
    assert (lr.staging_dir() / f"{QWEN_ID}.partial").is_dir()
    assert not lr.mlx_dir().exists()  # oMLX never sees a partial payload
    # An already promoted payload is never downloaded again.
    (lr.mlx_dir() / QWEN_ID).mkdir(parents=True)
    calls.clear()
    lr.download_payload(manifest, progress)
    assert calls == [] and progress.value["bytes_done"] == len(BODY)


def test_retries_restart_their_budget_whenever_the_file_grew(progress, sleeps, monkeypatch):
    manifest = {"model_id": QWEN_ID, "hf_repo": "org/model", "hf_revision": "0" * 40,
                "total_bytes": len(BODY), "files": [ENTRY]}
    calls = []

    def flaky(url, path, entry, progress, done, opener):
        # Each connection drops after 1000 more bytes, far more often than DOWNLOAD_ATTEMPTS.
        calls.append(url)
        have = path.stat().st_size if path.exists() else 0
        with path.open("ab") as out:
            out.write(BODY[have:have + 1000])
        if have + 1000 < len(BODY):
            raise RuntimeError("Model host closed the connection early")

    monkeypatch.setattr(lr, "_download_opener", lambda: object())
    monkeypatch.setattr(lr, "download_file", flaky)
    lr.download_payload(manifest, progress)
    assert len(calls) == 21 > lr.DOWNLOAD_ATTEMPTS
    assert (lr.staging_dir() / f"{QWEN_ID}.partial/model.safetensors").read_bytes() == BODY
    # A fresh budget backs off from the start again.
    assert set(sleeps) == {5}


def test_redirects_must_stay_on_https():
    handler = lr.HttpsOnlyRedirect()
    request = lr.urllib.request.Request("https://huggingface.co/x")
    with pytest.raises(lr.urllib.error.HTTPError):
        handler.redirect_request(request, None, 302, "Found", {}, "http://cdn.example/x")
    assert handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.example/x").full_url == \
        "https://cdn.example/x"


def test_the_download_opener_bypasses_proxies_and_refuses_a_redirect_off_https(monkeypatch):
    import http.server
    import threading

    seen = []

    class Redirect(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9/plain")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Redirect)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # A proxy would never answer; the opener must ignore it and reach the host directly.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    try:
        opener = REAL_DOWNLOAD_OPENER()
        with pytest.raises(lr.urllib.error.HTTPError, match="non-HTTPS redirect"):
            opener.open(f"http://127.0.0.1:{server.server_port}/model", timeout=5)
    finally:
        server.shutdown()
        server.server_close()
    assert seen == ["/model"]


# ── LaunchAgents ─────────────────────────────────────────────────────────────


def test_omlx_launch_agent_is_loopback_offline_bounded_and_owned():
    value = plistlib.loads(plistlib.dumps(lr.omlx_plist(9123)))
    args = value["ProgramArguments"]
    root = lr.runtime_root()
    assert value["Label"] == "com.local.omlx"
    assert value[lr.OWNER_KEY] == f"{lr.state_dir()}#com.local.model-gateway" == lr.owner_value()
    assert args[:2] == [str(root / f"omlx-{lr.OMLX_VERSION}/bin/omlx"), "serve"]
    expected = {"--base-path": str(root / "inference"), "--model-dir": str(root / "models/mlx"),
                "--host": "127.0.0.1", "--port": "9123", "--max-concurrent-requests": "1",
                "--memory-guard": "safe", "--paged-ssd-cache-dir": str(root / "models/cache"),
                "--paged-ssd-cache-max-size": "8GB"}
    assert {flag: args[args.index(flag) + 1] for flag in expected} == expected
    assert "--no-hf-cache" in args and "--api-key" not in args
    assert value["EnvironmentVariables"]["HF_HUB_OFFLINE"] == "1"
    # Absolute tools such as sysctl live in the sbin directories.
    assert value["EnvironmentVariables"]["PATH"].split(":")[-2:] == ["/usr/sbin", "/sbin"]
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
    launchctl.paths[lr.OMLX_LABEL] = "/tmp/x.plist"
    lr._bootstrap(Path("/tmp/x.plist"), lr.OMLX_LABEL)
    assert [call[0] for call in launchctl.calls] == ["print", "bootout", "bootstrap", "bootstrap"]
    assert sleeps == [1]
    results = iter([5] * 30)
    with pytest.raises(lr.LocalRuntimeError, match="bootstrap failed") as error:
        lr._bootstrap(Path("/tmp/x.plist"), lr.OMLX_LABEL)
    assert "private diagnostic" not in str(error.value)


def test_bootout_requires_launchd_to_have_loaded_this_gateways_plist(launchctl):
    launchctl.loaded.add(lr.OMLX_LABEL)
    launchctl.paths[lr.OMLX_LABEL] = "/Users/someone/Library/LaunchAgents/com.local.omlx.plist"
    with pytest.raises(lr.LocalRuntimeError, match="will not stop it"):
        lr._bootout(lr.OMLX_LABEL, lr.omlx_plist_path())
    del launchctl.paths[lr.OMLX_LABEL]
    lr._bootout(lr.OMLX_LABEL, lr.omlx_plist_path())
    assert [call[0] for call in launchctl.calls] == ["print", "print", "bootout"]


def test_a_new_plist_never_replaces_a_file_that_appeared(tmp_path):
    path = tmp_path / "LaunchAgents/com.local.omlx.plist"
    lr._create_new(path, b"ours")
    assert path.read_bytes() == b"ours" and path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(lr.LocalRuntimeError, match="appeared during setup"):
        lr._create_new(path, b"replacement")
    assert path.read_bytes() == b"ours"
    assert [entry.name for entry in path.parent.iterdir()] == [path.name]


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
    assert "KeepAlive" not in job and job[lr.OWNER_KEY] == lr.owner_value()
    env = job["EnvironmentVariables"]
    assert env["MODEL_GATEWAY_CONFIG"] == str(gateway)
    assert env["MODEL_GATEWAY_PACKAGE_ROOT"] == str(lr.PACKAGE_ROOT)
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
    config_io.MODEL_INFO_PATH.write_text(json.dumps({"allow_empty": True, "llm": []}))
    # A gateway LaunchAgent with another state directory is refused before any download.
    write_plist(lr.launch_agents_dir() / f"{lr.GATEWAY_LABEL}.plist",
                {"Label": lr.GATEWAY_LABEL, "EnvironmentVariables": {"MODEL_GATEWAY_STATE_DIR": "/elsewhere"}})
    with pytest.raises(lr.LocalRuntimeError, match="another state directory"):
        lr.start_add()
    assert bootstraps == []


def test_add_disk_check_counts_a_partly_downloaded_payload(eligible, gateway, launchctl, monkeypatch):
    payload = lr.manifest(QWEN)
    first, second = payload["files"][:2]
    partial = lr.mlx_dir() / f"{QWEN_ID}.partial"
    partial.mkdir(parents=True)
    (partial / first["path"]).write_bytes(b"x" * first["size_bytes"])
    (partial / second["path"]).write_bytes(b"x" * (second["size_bytes"] + 1))  # oversized: downloaded again
    needed = payload["total_bytes"] - first["size_bytes"] + lr.DISK_MARGIN
    bootstraps = queue(monkeypatch, free=needed - 1)
    with pytest.raises(lr.LocalRuntimeError, match="free disk space"):
        lr.start_add()
    monkeypatch.setattr(lr, "_disk_free", lambda path: needed)
    assert lr.start_add()["state"] == "queued" and len(bootstraps) == 1


def test_add_does_nothing_when_local_ai_is_installed_and_done(eligible, gateway, launchctl, monkeypatch):
    bootstraps = queue(monkeypatch)
    monkeypatch.setattr(lr, "_disk_free", lambda path: pytest.fail("checked disk space"))
    owned_install(9110)
    for recorded in (None, "done"):
        if recorded:
            lr._atomic(lr.status_path(), json.dumps({"state": recorded, "model": "qwen3.8-27b"}))
        status = lr.start_add()
        assert (status["installed"], status["state"]) == (True, "done")
    assert bootstraps == [] and not lr.setup_plist_path().exists()
    assert not [call for call in launchctl.calls if call[0] != "print"]


def test_concurrent_adds_start_one_job(eligible, gateway, launchctl, monkeypatch):
    bootstraps = queue(monkeypatch)
    with lr._lock(".start.lock", 0, ""):
        lr.start_add()
    assert bootstraps == [] and not lr.setup_plist_path().exists()
    with lr.job_lock(0):  # a job is running even if launchd has not said so yet
        lr.start_add()
    assert bootstraps == []
    lr.start_add()
    assert len(bootstraps) == 1


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


def test_the_job_holds_the_job_lock_for_its_whole_run(eligible, monkeypatch):
    held = []
    monkeypatch.setattr(lr, "download_payload", lambda *a: held.append(lr._job_alive()))
    monkeypatch.setattr(lr, "install", lambda *a: held.append(lr._job_alive()))
    monkeypatch.setattr(lr, "check", lambda *a: None)
    lr.run_job("qwen3.8-27b")
    assert held == [True, True] and lr._job_alive() is False
    # A second job never overwrites the running one's progress.
    lr._atomic(lr.status_path(), json.dumps({"state": "downloading", "bytes_done": 7, "updated_at": lr._now()}))
    with lr.job_lock(0):
        assert lr.main(["job", "qwen3.8-27b"]) == 2
    assert (lr.read_status()["state"], lr.read_status()["bytes_done"]) == ("downloading", 7)


def test_the_job_reexecs_when_the_installed_release_changed_during_download(eligible, monkeypatch):
    class Exec(BaseException):
        pass

    digests = iter(["before", "after"])
    execs = []
    monkeypatch.setattr(lr, "_source_digest", lambda: next(digests))
    monkeypatch.setattr(lr, "download_payload", lambda *a: None)
    monkeypatch.setattr(lr, "install", lambda *a: pytest.fail("installed with the old release's code"))
    monkeypatch.setattr(lr, "_reexec", lambda name: execs.append(name) or (_ for _ in ()).throw(Exec()))
    with pytest.raises(Exec):
        lr.run_job("qwen3.8-27b")
    assert execs == ["qwen3.8-27b"]
    assert lr._job_command("qwen3.8-27b") == ["/bin/bash", str(lr.PACKAGE_ROOT / "bin/model-gateway"),
                                              "_local-ai-job", "qwen3.8-27b"]


def test_source_digest_covers_code_models_and_the_runtime_lock(tmp_path, monkeypatch):
    package = tmp_path / "package"
    files = ["src/local_runtime.py", "src/nested/x.py", "bin/model-gateway", "local-models/m.json",
             "local-runtime/pyproject.toml", "local-runtime/uv.lock"]
    for name in files:
        (package / name).parent.mkdir(parents=True, exist_ok=True)
        (package / name).write_text(name)
    monkeypatch.setattr(lr, "PACKAGE_ROOT", package)
    monkeypatch.setattr(lr, "MODELS_DIR", package / "local-models")
    monkeypatch.setattr(lr, "RUNTIME_PROJECT", package / "local-runtime")
    digest = lr._source_digest()
    (package / "src/__pycache__").mkdir()
    (package / "src/__pycache__/x.pyc").write_text("compiled")
    assert lr._source_digest() == digest
    for name in files:
        before = lr._source_digest()
        (package / name).write_text(name + " upgraded")
        assert lr._source_digest() != before


PACKAGE_ROOT_PROBE = """
import json, os
from pathlib import Path
from src import local_runtime as lr

root = Path(os.environ["MODEL_GATEWAY_PACKAGE_ROOT"])
before = lr._source_digest()
target = os.readlink(root)
os.unlink(root)
os.symlink(target.replace("A", "B"), root)
print(json.dumps({"root": str(lr.PACKAGE_ROOT), "models": str(lr.MODELS_DIR), "runtime": str(lr.RUNTIME_PROJECT),
                  "command": lr._job_command("m"), "changed": lr._source_digest() != before}))
"""


def test_a_package_root_from_the_cli_follows_current_across_an_upgrade(tmp_path):
    # Python imports this module from the resolved release; the CLI's unresolved root must win.
    for release in ("A", "B"):
        for name in ("bin/model-gateway", "src/x.py", "local-models/m.json", "local-runtime/uv.lock"):
            path = tmp_path / "releases" / release / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{release} {name}")
    current = tmp_path / "current"
    current.symlink_to("releases/A")
    env = {**os.environ, "MODEL_GATEWAY_PACKAGE_ROOT": str(current)}
    result = subprocess.run([sys.executable, "-c", PACKAGE_ROOT_PROBE], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout)
    assert probe == {"root": str(current), "models": str(current / "local-models"),
                     "runtime": str(current / "local-runtime"),
                     "command": ["/bin/bash", str(current / "bin/model-gateway"), "_local-ai-job", "m"],
                     "changed": True}
    relative = {**os.environ, "MODEL_GATEWAY_PACKAGE_ROOT": "current"}
    fallback = subprocess.run([sys.executable, "-c", "from src import local_runtime as lr; print(lr.PACKAGE_ROOT)"],
                              cwd=ROOT, env=relative, capture_output=True, text=True, timeout=60)
    assert fallback.stdout.strip() == str(ROOT)


def test_cli_passes_its_unresolved_package_root_to_local_ai(tmp_path):
    release = tmp_path / "releases/0.4.0-0123456789ab"
    (release / "bin").mkdir(parents=True)
    shutil.copy2(ROOT / "bin/model-gateway", release / "bin/model-gateway")
    current = tmp_path / "current"
    current.symlink_to("releases/0.4.0-0123456789ab")
    (release / ".package").write_text(f"MANAGER=model-gateway-pkg\nROOT={current}\nPYTHON={sys.executable}\n")
    harness = tmp_path / "harness.sh"
    harness.write_text('cli="$1"\nset -- env\nsource "$cli" >/dev/null\n'
                       'run_python() { printf "%s|%s\\n" "$MODEL_GATEWAY_PACKAGE_ROOT" "$PWD"; }\n'
                       'run_local_ai status\n')
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_GATEWAY_", "GATEWAY_VISION"))}
    env.update(HOME=str(tmp_path), MODEL_GATEWAY_LAUNCHD_LABEL="com.local.model-gateway-test-does-not-exist",
               MODEL_GATEWAY_PLIST_DIR=str(tmp_path / "LaunchAgents"))
    result = subprocess.run(["bash", str(harness), str(current / "bin/model-gateway")],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{current}|{current}\n"


@pytest.mark.parametrize("custom", [False, True])
def test_the_gateway_launch_agent_carries_only_a_custom_state_dir(tmp_path, custom):
    # The server loads local AI's model from the state directory the CLI installed it in.
    harness = tmp_path / "harness.sh"
    harness.write_text('cli="$1"\nset -- env\nsource "$cli" >/dev/null\n'
                       'persist_install_config() { :; }\nwrite_plist >/dev/null\n')
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_GATEWAY_", "GATEWAY_VISION"))}
    env.update(HOME=str(tmp_path), MODEL_GATEWAY_LAUNCHD_LABEL="com.local.model-gateway-test-does-not-exist",
               MODEL_GATEWAY_PLIST_DIR=str(tmp_path / "LaunchAgents"))
    expected = str(tmp_path / "state") if custom else None
    # A later command from a plain shell (an update) keeps the recorded directory.
    for state in (expected, None):
        run_env = {**env, "MODEL_GATEWAY_STATE_DIR": state} if state else env
        result = subprocess.run(["bash", str(harness), str(ROOT / "bin/model-gateway")],
                                capture_output=True, text=True, env=run_env, timeout=60)
        assert result.returncode == 0, result.stderr
        plist = plistlib.loads((tmp_path / "LaunchAgents/com.local.model-gateway-test-does-not-exist.plist").read_bytes())
        assert plist["EnvironmentVariables"].get("MODEL_GATEWAY_STATE_DIR") == expected


def test_progress_never_recreates_removed_state(eligible):
    lr.runtime_root().mkdir(parents=True)
    progress = lr.Progress(QWEN, 10)
    progress.update(bytes_done=1)
    assert lr.read_status()["bytes_done"] == 1
    lr.shutil.rmtree(lr.runtime_root())
    progress.update(bytes_done=2)
    assert not lr.runtime_root().exists()


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


def tiny_payload(model_id: str = QWEN_ID, *, staging: Path | None = None) -> dict:
    """A verified two-file payload staged in .partial, plus its manifest."""
    partial = (staging or lr.staging_dir()) / f"{model_id}.partial"
    partial.mkdir(parents=True)
    files = {"model-00001.safetensors": b"weights",
             "model.safetensors.index.json": json.dumps({"weight_map": {"a": "model-00001.safetensors"}}).encode()}
    entries = []
    for name, data in files.items():
        (partial / name).write_bytes(data)
        entries.append({"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return {"schema_version": 1, "model_id": model_id, "hf_repo": "org/model", "hf_revision": "0" * 40, "license": "x",
            "total_bytes": sum(e["size_bytes"] for e in entries), "files": entries}


def test_install_verifies_builds_starts_and_wires_the_gateway(eligible, gateway, launchctl, monkeypatch):
    payload = tiny_payload()
    runs, health = [], []
    monkeypatch.setattr(lr, "manifest", qwen_manifest(payload))
    monkeypatch.setattr(lr, "_run", lambda *args, env=None: runs.append((args, env and env.get("UV_PROJECT_ENVIRONMENT"))))
    monkeypatch.setattr(lr, "_wait_health", health.append)
    monkeypatch.setattr(lr, "_uv", lambda: "/fake/uv")
    launchctl.loaded.add(lr.GATEWAY_LABEL)
    lr.install(QWEN, payload)

    assert (lr.mlx_dir() / QWEN_ID).is_dir() and not (lr.staging_dir() / f"{QWEN_ID}.partial").exists()
    assert runs[0] == (("/fake/uv", "sync", "--locked", "--no-dev", "--project", str(lr.RUNTIME_PROJECT)),
                       str(lr.venv_dir()))
    assert runs[1][0][:2] == (str(lr.venv_dir() / "bin/python"), "-c")
    key = lr.api_key_path()
    assert key.stat().st_mode & 0o777 == 0o600
    settings = json.loads((lr.inference_dir() / "model_settings.json").read_text())
    assert settings["models"][QWEN_ID]["is_default"] is True
    agent = plistlib.loads(lr.omlx_plist_path().read_bytes())
    assert agent[lr.OWNER_KEY] == lr.owner_value() and lr.omlx_owner() == "gateway"
    assert ("bootstrap", f"gui/{os.getuid()}", str(lr.omlx_plist_path())) in launchctl.calls
    assert ("kickstart", "-k", f"gui/{os.getuid()}/{lr.GATEWAY_LABEL}") in launchctl.calls
    assert health == ["http://127.0.0.1:9110/health", "http://127.0.0.1:9111/health"]

    config = yaml.safe_load(gateway.read_text())
    assert config["providers"]["omlx"] == {"base_url": "http://127.0.0.1:9110/v1", "protocol": "openai",
                                           "api_key": "", "api_key_file": str(key), "enabled": True,
                                           "managed_by": "local_ai"}
    assert config["providers"]["fireworks"]  # untouched
    assert key.read_text().strip() not in gateway.read_text()
    # The model loads from the release (shipped layer), never as a copy in the owner's catalog.
    assert json.loads(config_io.MODEL_INFO_PATH.read_text())["llm"] == []
    providers.reload()
    (entry,) = [row for row in providers.effective_model_inventory() if row["name"] == "qwen3.8-27b"]
    assert (entry["provider"], entry["omlx_id"], entry["pricing_status"], entry["origin"]) == (
        "omlx", QWEN_ID, "unmetered", {"layer": "shipped", "profile": "local-ai"})
    routed = providers.resolve("qwen3.8-27b")
    assert (routed.base_url, routed.api_key, routed.provider_model_id) == (
        "http://127.0.0.1:9110/v1", key.read_text().strip(), QWEN_ID)
    assert lr.status()["installed"] is True


def test_install_refuses_an_external_plist_that_appears_after_download(eligible, gateway, launchctl, monkeypatch):
    payload = tiny_payload()
    monkeypatch.setattr(lr, "manifest", qwen_manifest(payload))
    monkeypatch.setattr(lr, "_uv", lambda: "/fake/uv")
    monkeypatch.setattr(lr, "_wait_health", lambda url: pytest.fail("started oMLX"))
    appeared = []

    def build(*args, env=None):
        # A developer installs their own oMLX while the runtime builds.
        if not appeared:
            appeared.append(external_omlx())

    monkeypatch.setattr(lr, "_run", build)
    with pytest.raises(lr.LocalRuntimeError, match="managed outside Model Gateway"):
        lr.install(QWEN, payload)
    assert lr.omlx_plist_path().read_bytes() == appeared[0]
    # And when it appears after the ownership check, creating ours fails instead of replacing it.
    monkeypatch.setattr(lr, "omlx_owner", lambda: None)
    with pytest.raises(lr.LocalRuntimeError, match="appeared during setup"):
        lr.install(QWEN, payload)
    assert lr.omlx_plist_path().read_bytes() == appeared[0]
    assert all(call[0] == "print" for call in launchctl.calls)
    assert "omlx" not in yaml.safe_load(gateway.read_text())["providers"]


def test_install_rejects_a_payload_that_fails_verification(eligible, gateway, monkeypatch):
    payload = tiny_payload()
    (lr.staging_dir() / f"{QWEN_ID}.partial/model-00001.safetensors").write_bytes(b"tampered")
    with pytest.raises(lr.LocalRuntimeError, match="failed verification"):
        lr.install(QWEN, payload)
    assert not (lr.mlx_dir() / QWEN_ID).exists()


def test_gateway_wiring_rolls_back_when_the_gateway_does_not_restart(gateway, launchctl, monkeypatch):
    monkeypatch.setattr(lr, "manifest", qwen_manifest({"model_id": QWEN_ID}))
    before = (gateway.read_text(), config_io.MODEL_INFO_PATH.read_text())
    with pytest.raises(lr.LocalRuntimeError, match="not running"):
        lr.configure_gateway(QWEN, 9110)
    assert (gateway.read_text(), config_io.MODEL_INFO_PATH.read_text()) == before
    # The restored files still get their one-time adoption.
    assert not (lr.runtime_root() / lr.CATALOG_LAYER_MARKER).exists()


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
    with pytest.raises(lr.LocalRuntimeError, match="exactly the verified"):
        lr.check(QWEN, payload)


def test_check_without_a_gateway_key_completes_against_omlx_and_requires_gateway_routing(
        eligible, gateway, monkeypatch):
    # A package install: no admin key, and a product's consumers make /v1 refuse anonymous clients.
    config = yaml.safe_load(gateway.read_text())
    del config["auth"]["admin_keys"]
    gateway.write_text(yaml.safe_dump(config))
    providers.reload()
    owned_install(9123)
    omlx_key = lr._key(lr.api_key_path(), create=True)
    payload = {"model_id": QWEN_ID}
    path = str(lr.mlx_dir() / QWEN_ID)
    seen = []

    def fake_request(url, token="", body=None, **kw):
        seen.append((url, token, body))
        if url == "http://127.0.0.1:9111/v1/models":
            raise lr.urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
        return {
            "http://127.0.0.1:9123/v1/models/status": {"models": [{"id": QWEN_ID, "model_path": path}]},
            "http://127.0.0.1:9123/v1/chat/completions": {"choices": [{"message": {"content": "OK"}}]},
        }[url]

    monkeypatch.setattr(lr, "_request", fake_request)
    with pytest.raises(lr.LocalRuntimeError, match="does not route"):
        lr.check(QWEN, payload)
    assert not any(url.endswith("/chat/completions") for url, _token, _body in seen)

    monkeypatch.setattr(lr, "manifest", qwen_manifest(payload))
    monkeypatch.setattr(lr, "_restart_gateway", lambda required: None)
    lr.configure_gateway(QWEN, 9123)
    seen.clear()
    lr.check(QWEN, payload)
    url, token, body = seen[-1]
    assert (url, token, body["model"]) == ("http://127.0.0.1:9123/v1/chat/completions", omlx_key, QWEN_ID)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    # No standing credential was added to the gateway.
    assert [c["id"] for c in yaml.safe_load(gateway.read_text())["auth"]["consumer_credentials"]] == [
        "app-runtime", "app-manager", "app-local"]

    # The routing must be this job's enabled, local-AI-managed provider.
    wired = yaml.safe_load(gateway.read_text())
    for change in ({"enabled": False}, {"enabled": "true"}, {"managed_by": None}, {"managed_by": "operator"}):
        edited = {**wired, "providers": {**wired["providers"], "omlx": {**wired["providers"]["omlx"], **change}}}
        if change.get("managed_by", "") is None:
            del edited["providers"]["omlx"]["managed_by"]
        gateway.write_text(yaml.safe_dump(edited))
        seen.clear()
        with pytest.raises(lr.LocalRuntimeError, match="does not route"):
            lr.check(QWEN, payload)
        assert not any(url.endswith("/chat/completions") for url, _token, _body in seen)


def test_check_through_the_gateway_surfaces_other_gateway_errors(eligible, monkeypatch):
    owned_install(9123)
    lr._key(lr.api_key_path(), create=True)
    path = str(lr.mlx_dir() / QWEN_ID)

    def fake_request(url, token="", body=None, **kw):
        if url == "http://127.0.0.1:9111/v1/models":
            raise lr.urllib.error.HTTPError(url, 503, "Misconfigured", {}, None)
        return {"models": [{"id": QWEN_ID, "model_path": path}]}

    monkeypatch.setattr(lr, "_request", fake_request)
    with pytest.raises(lr.urllib.error.HTTPError):
        lr.check(QWEN, {"model_id": QWEN_ID})


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


def test_remove_restarts_the_gateway_once_after_teardown_so_endpoint_json_drops_the_runtime(
        eligible, gateway, launchctl, monkeypatch, tmp_path):
    endpoint = tmp_path / "endpoint.json"
    monkeypatch.setenv("MODEL_GATEWAY_ENDPOINT_FILE", str(endpoint))
    owned_install(9110)
    lr._key(lr.api_key_path(), create=True)
    config = yaml.safe_load(gateway.read_text())
    config["providers"]["omlx"] = {"base_url": "http://127.0.0.1:9110/v1", "api_key_file": str(lr.api_key_path()),
                                   "managed_by": "local_ai"}
    gateway.write_text(yaml.safe_dump(config))
    discovery.write_endpoint_file()
    assert json.loads(endpoint.read_text())["local_runtime"]["base_url"] == "http://127.0.0.1:9110"
    launchctl.loaded.update({lr.OMLX_LABEL, lr.GATEWAY_LABEL})
    real = lr._launchctl

    def restarting(*args):
        if args[0] == "kickstart":
            # The restarted gateway rewrites endpoint.json from what is on disk now.
            providers.reload()
            discovery.write_endpoint_file()
        return real(*args)

    monkeypatch.setattr(lr, "_launchctl", restarting)
    monkeypatch.setattr(lr, "_wait_health", lambda url: None)
    lr.remove()
    actions = [call[0] for call in launchctl.calls if call[0] != "print"]
    assert actions == ["bootout", "kickstart"]
    assert json.loads(endpoint.read_text())["local_runtime"] == {"managed": True, "base_url": None,
                                                                "health_url": None}


def test_remove_works_without_an_installed_gateway(eligible, gateway, launchctl, monkeypatch):
    owned_install(9110)
    launchctl.loaded.add(lr.OMLX_LABEL)
    monkeypatch.setattr(lr, "_wait_health", lambda url: pytest.fail("waited for a gateway"))
    lr.remove()
    assert not lr.omlx_plist_path().exists() and not lr.runtime_root().exists()
    assert ("kickstart", "-k", f"gui/{os.getuid()}/{lr.GATEWAY_LABEL}") not in launchctl.calls


def test_remove_refuses_an_omlx_that_launchd_loaded_from_elsewhere(eligible, gateway, launchctl, monkeypatch):
    owned_install(9110)
    launchctl.loaded.add(lr.OMLX_LABEL)
    launchctl.paths[lr.OMLX_LABEL] = "/Library/LaunchDaemons/com.local.omlx.plist"
    with pytest.raises(lr.LocalRuntimeError, match="will not stop it"):
        lr.remove()
    assert not [call for call in launchctl.calls if call[0] == "bootout"]
    assert lr.omlx_plist_path().exists() and (lr.mlx_dir() / QWEN_ID).is_dir()


def test_remove_and_cancel_wait_for_a_stopped_job_to_exit(eligible, gateway, launchctl, monkeypatch):
    import threading

    owned_install(9110)
    lr._atomic(lr.status_path(), json.dumps({"state": "downloading", "updated_at": lr._now()}))
    monkeypatch.setattr(lr, "JOB_STOP_SECONDS", 0.3)
    with lr.job_lock(0):
        for operation in (lr.cancel, lr.remove):
            with pytest.raises(lr.LocalRuntimeError, match="still stopping"):
                operation()
    assert lr.omlx_plist_path().exists() and lr.read_status()["state"] == "downloading"

    monkeypatch.setattr(lr, "JOB_STOP_SECONDS", 10)
    job = lr.job_lock(0)
    job.__enter__()
    timer = threading.Timer(0.3, job.__exit__, (None, None, None))
    timer.start()
    try:
        lr.remove()
    finally:
        timer.join()
    assert not lr.omlx_plist_path().exists() and not lr.runtime_root().exists()


def test_unconfigure_leaves_a_foreign_omlx_provider_and_its_models_untouched(gateway):
    config = yaml.safe_load(gateway.read_text())
    config["providers"]["omlx"] = {"base_url": "http://127.0.0.1:8000/v1", "api_key_file": "/Users/dev/omlx.key"}
    gateway.write_text(yaml.safe_dump(config))
    catalog = json.dumps({"llm": [{"name": "qwen3.8-27b", "provider": "omlx", "omlx_id": QWEN_ID}]})
    config_io.MODEL_INFO_PATH.write_text(catalog)
    before = gateway.read_text()
    assert lr.unconfigure_gateway() is False
    assert gateway.read_text() == before and config_io.MODEL_INFO_PATH.read_text() == catalog


# ── the local model in the shipped catalog layer (ADR 0002) ─────────────────


def wire_local_ai(gateway, monkeypatch) -> None:
    """The gateway as configure_gateway leaves it, with the payload installed."""
    monkeypatch.setattr(lr, "manifest", qwen_manifest({"model_id": QWEN_ID}))
    lr._key(lr.api_key_path(), create=True)
    (lr.mlx_dir() / QWEN_ID).mkdir(parents=True)
    config = yaml.safe_load(gateway.read_text())
    config["providers"]["omlx"] = {"base_url": "http://127.0.0.1:9110/v1", "protocol": "openai", "api_key": "",
                                   "api_key_file": str(lr.api_key_path()), "enabled": True,
                                   "managed_by": "local_ai"}
    gateway.write_text(yaml.safe_dump(config))
    providers.reload()


def local_entry() -> dict | None:
    providers.reload()
    return next((row for row in providers.effective_model_inventory() if row["name"] == "qwen3.8-27b"), None)


def test_the_local_model_loads_from_the_release_and_takes_its_fixes(gateway, monkeypatch):
    assert local_entry() is None
    wire_local_ai(gateway, monkeypatch)
    entry = local_entry()
    assert (entry["omlx_id"], entry["origin"]) == (QWEN_ID, {"layer": "shipped", "profile": "local-ai"})
    # A release that changes the model's row reaches this install with nothing to migrate.
    fixed = lr.LocalModel(QWEN.name, QWEN.manifest, {**QWEN.catalog, "context": 65536}, QWEN.settings,
                          QWEN.min_memory_bytes)
    monkeypatch.setitem(lr.MODELS, QWEN.name, fixed)
    assert local_entry()["context"] == 65536
    # The owner added it, so it loads whatever the shipped selection.
    config = yaml.safe_load(gateway.read_text())
    gateway.write_text(yaml.safe_dump({**config, "catalog": {"shipped": "none"}}))
    assert local_entry() is not None
    # Once the gateway no longer manages oMLX, or without the payload, it is gone.
    config = yaml.safe_load(gateway.read_text())
    managed = config["providers"].pop("omlx")
    gateway.write_text(yaml.safe_dump(config))
    assert local_entry() is None
    config["providers"]["omlx"] = managed
    gateway.write_text(yaml.safe_dump(config))
    assert local_entry() is not None
    (lr.mlx_dir() / QWEN_ID).rmdir()
    assert local_entry() is None


def test_startup_migration_retires_copies_of_the_local_model(gateway, monkeypatch):
    from src import admin

    wire_local_ai(gateway, monkeypatch)
    # What configure_gateway used to write.
    config_io.MODEL_INFO_PATH.write_text(json.dumps({"llm": [
        {"name": "qwen3.8-27b", "provider": "omlx", "omlx_id": QWEN_ID, **QWEN.catalog}]}))
    assert admin.migrate_catalog()["dropped"] == ["qwen3.8-27b"]
    assert json.loads(config_io.MODEL_INFO_PATH.read_text())["llm"] == []
    assert local_entry()["origin"]["layer"] == "shipped"
    # A Home Server import (null provider, its own alias and fields) becomes a sparse,
    # reviewable override: the route and the release's later fixes stay the gateway's.
    imported = {"name": "qwen3.8-27b", "provider": None, "omlx_id": QWEN_ID, "alias": "qwen38mlx",
                **QWEN.catalog, "desc": "Home Server's Qwen"}
    config_io.MODEL_INFO_PATH.write_text(json.dumps({"llm": [imported]}))
    assert admin.migrate_catalog()["review"] == ["qwen3.8-27b"]
    (override,) = json.loads(config_io.MODEL_INFO_PATH.read_text())["llm"]
    assert override == {"name": "qwen3.8-27b", "alias": "qwen38mlx", "desc": "Home Server's Qwen",
                        "shipped_base": {"alias": None, "desc": QWEN.catalog["desc"]}, "review": True}
    entry = local_entry()
    assert (entry["alias"], entry["desc"], entry["origin"]["overridden"]) == (
        "qwen38mlx", "Home Server's Qwen", ["alias", "desc"])
    assert providers.resolve("qwen38mlx").provider_model_id == QWEN_ID


def test_an_owner_qwen_without_gateway_local_ai_is_untouched(gateway):
    from src import admin

    # A developer gateway routing its own oMLX: the row is the owner's model.
    config = yaml.safe_load(gateway.read_text())
    config["providers"]["omlx"] = {"base_url": "http://127.0.0.1:8000/v1", "api_key_file": "/Users/dev/omlx.key"}
    gateway.write_text(yaml.safe_dump(config))
    catalog = json.dumps({"llm": [{"name": "qwen3.8-27b", "provider": None, "omlx_id": QWEN_ID}]})
    config_io.MODEL_INFO_PATH.write_text(catalog)
    assert admin.migrate_catalog() == {}
    assert config_io.MODEL_INFO_PATH.read_text() == catalog
    assert local_entry()["origin"] == {"layer": "owner"}


def test_adding_local_ai_again_restores_a_hidden_model_and_tolerates_overrides(gateway, monkeypatch):
    monkeypatch.setattr(lr, "manifest", qwen_manifest({"model_id": QWEN_ID}))
    monkeypatch.setattr(lr, "_restart_gateway", lambda required: None)
    config = yaml.safe_load(gateway.read_text())
    gateway.write_text(yaml.safe_dump({**config, "catalog": {"hidden": ["qwen3.8-27b", "other"]}}))
    config_io.MODEL_INFO_PATH.write_text(json.dumps({"llm": [
        {"name": "qwen3.8-27b", "desc": "mine", "shipped_base": {"desc": "x"}}]}))
    lr.configure_gateway(QWEN, 9110)
    assert yaml.safe_load(gateway.read_text())["catalog"]["hidden"] == ["other"]
    # Another route under the name is still the owner's own model, and blocks local AI.
    config_io.MODEL_INFO_PATH.write_text(json.dumps({"llm": [
        {"name": "qwen3.8-27b", "provider": "fireworks", "provider_model_id": "qwen"}]}))
    assert "different model" in lr.gateway_conflict(QWEN)


def test_upgrading_keeps_a_local_model_the_owner_deleted_deleted_once(gateway, monkeypatch):
    from src import admin

    wire_local_ai(gateway, monkeypatch)
    # Earlier releases always wrote the model: no row means the owner deleted it.
    assert lr.catalog_adoption_pending()
    admin.migrate_catalog()
    assert yaml.safe_load(gateway.read_text())["catalog"]["hidden"] == ["qwen3.8-27b"]
    assert local_entry() is None and not lr.catalog_adoption_pending()
    # Only once: adding local AI again (or this release's own installs) is never hidden.
    monkeypatch.setattr(lr, "_restart_gateway", lambda required: None)
    lr.configure_gateway(QWEN, 9110)
    assert "catalog" not in yaml.safe_load(gateway.read_text())
    admin.migrate_catalog()
    assert local_entry() is not None


def test_adoption_keeps_a_model_that_still_has_its_copy(gateway, monkeypatch):
    from src import admin

    wire_local_ai(gateway, monkeypatch)
    config_io.MODEL_INFO_PATH.write_text(json.dumps({"llm": [
        {"name": "qwen3.8-27b", "provider": "omlx", "omlx_id": QWEN_ID, **QWEN.catalog}]}))
    admin.migrate_catalog()
    assert "catalog" not in yaml.safe_load(gateway.read_text())
    assert json.loads(config_io.MODEL_INFO_PATH.read_text())["llm"] == []
    assert local_entry()["origin"]["layer"] == "shipped" and not lr.catalog_adoption_pending()


def test_an_admin_save_of_the_local_model_stores_only_what_changed(gateway, monkeypatch):
    wire_local_ai(gateway, monkeypatch)
    lr.mark_catalog_layer()
    config_io.upsert_model("qwen3.8-27b", provider="local", omlx_id=QWEN_ID, desc="mine")
    (row,) = json.loads(config_io.MODEL_INFO_PATH.read_text())["llm"]
    assert row == {"name": "qwen3.8-27b", "desc": "mine", "shipped_base": {"desc": QWEN.catalog["desc"]}}
    assert local_entry()["origin"]["overridden"] == ["desc"]


def wire_local_ai_config(gateway) -> None:
    config = yaml.safe_load(gateway.read_text())
    config["providers"]["omlx"] = {"base_url": "http://127.0.0.1:9110/v1", "api_key_file": str(lr.api_key_path()),
                                   "enabled": True, "managed_by": "local_ai"}
    gateway.write_text(yaml.safe_dump(config))


def test_remove_drops_old_copies_but_keeps_overrides_and_settings(gateway, monkeypatch):
    wire_local_ai(gateway, monkeypatch)
    override = {"name": "qwen3.8-27b", "desc": "mine", "shipped_base": {"desc": "x"}}
    copy = {"name": "qwen3.8-27b", "provider": "omlx", "omlx_id": QWEN_ID}
    for rows, kept in (([copy], []), ([override], [override])):
        wire_local_ai_config(gateway)
        config_io.MODEL_INFO_PATH.write_text(json.dumps({"llm": rows}))
        assert lr.unconfigure_gateway() is True
        written = yaml.safe_load(gateway.read_text())
        assert "omlx" not in written["providers"] and "catalog" not in written
        assert json.loads(config_io.MODEL_INFO_PATH.read_text())["llm"] == kept


def test_routing_check_follows_what_the_gateway_loads(gateway, monkeypatch):
    wire_local_ai(gateway, monkeypatch)
    assert lr._catalog_routes(QWEN)
    config = yaml.safe_load(gateway.read_text())
    for change in ({"catalog": {"hidden": ["qwen3.8-27b"]}},
                   {"models": [{"name": "qwen3.8-27b", "provider": "fireworks", "provider_model_id": "q"}]}):
        gateway.write_text(yaml.safe_dump({**config, **change}))
        assert not lr._catalog_routes(QWEN)
    # An overlay row on the same route (any spelling of the local provider) is an override.
    gateway.write_text(yaml.safe_dump({**config, "models": [{"name": "qwen3.8-27b", "provider": "local",
                                                              "omlx_id": QWEN_ID, "desc": "overlay"}]}))
    assert lr._catalog_routes(QWEN)


def test_a_broken_local_manifest_never_takes_the_catalog_down(gateway, monkeypatch, caplog):
    wire_local_ai(gateway, monkeypatch)

    def broken(local):
        raise lr.LocalRuntimeError("Invalid model manifest for qwen3.8-27b: fixture")

    monkeypatch.setattr(lr, "manifest", broken)
    config_io.MODEL_INFO_PATH.write_text(json.dumps({"llm": [
        {"name": "glm", "provider": "fireworks", "provider_model_id": "glm"}]}))
    assert local_entry() is None
    assert [row["name"] for row in providers.effective_model_inventory()] == ["glm"]
    assert "local AI model qwen3.8-27b not loaded" in caplog.text


def test_setup_converts_copies_so_the_restarted_gateway_never_waits_for_its_lock(gateway, monkeypatch):
    from src import shipped

    monkeypatch.setattr(lr, "manifest", qwen_manifest({"model_id": QWEN_ID}))
    lr._key(lr.api_key_path(), create=True)
    (lr.mlx_dir() / QWEN_ID).mkdir(parents=True)
    # A Home Server import made before local AI was added on this Mac.
    config_io.MODEL_INFO_PATH.write_text(json.dumps({"llm": [
        {"name": "qwen3.8-27b", "provider": None, "omlx_id": QWEN_ID, "alias": "qwen38mlx", **QWEN.catalog}]}))
    seen = []

    def restart(required):
        layer = shipped.layer(config_io.load_config_full(), include_packs=False)
        loaded = {row["name"]: {k: v for k, v in row.items() if k != "origin"} for row in layer.entries}
        seen.append((config_io.shipped_migration_pending(loaded), lr.catalog_adoption_pending()))

    monkeypatch.setattr(lr, "_restart_gateway", restart)
    lr.configure_gateway(QWEN, 9110)
    assert seen == [(False, False)]
    (override,) = json.loads(config_io.MODEL_INFO_PATH.read_text())["llm"]
    assert override["alias"] == "qwen38mlx" and override["review"] is True and "provider" not in override


def test_startup_skips_the_migration_while_another_writer_holds_the_lock(gateway, monkeypatch, caplog):
    import threading

    from src.config_lock import config_write_lock

    wire_local_ai(gateway, monkeypatch)
    lr.mark_catalog_layer()
    copy = json.dumps({"llm": [{"name": "qwen3.8-27b", "provider": "omlx", "omlx_id": QWEN_ID, **QWEN.catalog}]})
    config_io.MODEL_INFO_PATH.write_text(copy)
    held, done = threading.Event(), threading.Event()

    def writer():
        with config_write_lock(config_io.CONFIG_PATH):
            held.set()
            done.wait(30)

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        held.wait(5)
        with TestClient(app):
            pass
    finally:
        done.set()
        thread.join()
    assert "catalog migration failed or skipped" in caplog.text
    assert config_io.MODEL_INFO_PATH.read_text() == copy
    with TestClient(app):  # the next start migrates
        pass
    assert json.loads(config_io.MODEL_INFO_PATH.read_text())["llm"] == []


def test_setup_refuses_a_gateway_launch_agent_with_another_state_dir(gateway, monkeypatch):
    monkeypatch.setattr(lr, "manifest", qwen_manifest({"model_id": QWEN_ID}))
    monkeypatch.setattr(lr, "_restart_gateway", lambda required: None)
    plist = lr.launch_agents_dir() / f"{lr.GATEWAY_LABEL}.plist"
    write_plist(plist, {"Label": lr.GATEWAY_LABEL, "EnvironmentVariables": {"MODEL_GATEWAY_STATE_DIR": "/elsewhere"}})
    with pytest.raises(lr.LocalRuntimeError, match="another state directory"):
        lr.configure_gateway(QWEN, 9110)
    assert "omlx" not in yaml.safe_load(gateway.read_text())["providers"]
    write_plist(plist, {"Label": lr.GATEWAY_LABEL, "EnvironmentVariables": {"MODEL_GATEWAY_STATE_DIR": str(lr.state_dir())}})
    lr.configure_gateway(QWEN, 9110)
    assert yaml.safe_load(gateway.read_text())["providers"]["omlx"]["managed_by"] == "local_ai"


def test_a_pack_cannot_claim_a_local_ai_model():
    from src import packs, shipped

    pack = {"schema_version": 1, "namespace": "app", "source_revision": "r", "requires": [],
            "models": [{"name": "qwen3.8-27b", "provider": "omlx", "omlx_id": "other-payload"}]}
    with pytest.raises(packs.PackError, match="used by a shipped model"):
        packs.validate("app", pack, shipped_rows=shipped.all_shipped_rows(), owner_rows=[], other_packs={})


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


def test_invalid_config_is_a_safe_422_in_the_admin_api_and_cli(eligible, gateway, launchctl, monkeypatch, capsys):
    queue(monkeypatch)
    client = TestClient(app)
    admin = {"Authorization": "Bearer admin"}
    # A broken catalog file is a ValueError from deep inside start_add.
    config_io.MODEL_INFO_PATH.write_text('{"llm": [sk-secret')
    refused = client.post("/admin/api/local-ai", headers=admin, json={"action": "add"})
    assert refused.status_code == 422 and "sk-secret" not in refused.text
    assert refused.json()["error"]["message"] == "The gateway configuration or local AI state is invalid"
    assert lr.main(["add"]) == 2
    assert capsys.readouterr().err == "error: The gateway configuration or local AI state is invalid\n"
    # YAML errors quote the offending line.
    with pytest.raises(yaml.YAMLError) as invalid:
        yaml.safe_load("providers: {omlx: [sk-secret-value\n")
    assert "sk-secret" in str(invalid.value)
    monkeypatch.setattr(lr, "gateway_conflict", lambda local: (_ for _ in ()).throw(invalid.value))
    refused = client.post("/admin/api/local-ai", headers=admin, json={"action": "add"})
    assert refused.status_code == 422
    assert refused.json()["error"]["message"] == "The gateway configuration is not valid YAML"
    assert lr.main(["add"]) == 2
    assert capsys.readouterr().err == "error: The gateway configuration is not valid YAML\n"
    monkeypatch.setattr(lr, "status", lambda name=None: (_ for _ in ()).throw(ValueError("invalid literal: 'sk-secret'")))
    status = client.get("/admin/api/local-ai", headers=admin)
    assert status.status_code == 422 and "sk-secret" not in status.text
    assert lr.main(["status"]) == 2 and "sk-secret" not in capsys.readouterr().err


def test_admin_refuses_key_edits_and_deletes_of_the_local_ai_provider(gateway):
    key_file = str(lr.api_key_path())
    config = yaml.safe_load(gateway.read_text())
    config["providers"]["omlx"] = {"base_url": "http://127.0.0.1:9110/v1", "api_key": "", "api_key_file": key_file,
                                   "managed_by": "local_ai"}
    gateway.write_text(yaml.safe_dump(config))
    before = gateway.read_text()
    client = TestClient(app)
    admin = {"Authorization": "Bearer admin"}
    for body in ({"base_url": "http://127.0.0.1:9110/v1", "api_key": "new"},
                 {"base_url": "http://127.0.0.1:9110/v1", "api_key": ""}):
        edited = client.post("/admin/api/providers/omlx", headers=admin, json=body)
        assert edited.status_code == 400 and "managed by Model Gateway local AI" in edited.json()["error"]["message"]
    deleted = client.delete("/admin/api/providers/omlx", headers=admin)
    assert deleted.status_code == 409 and "model-gateway local-ai" in deleted.json()["error"]["message"]
    assert gateway.read_text() == before
    assert yaml.safe_load(gateway.read_text())["providers"]["omlx"]["api_key_file"] == key_file


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


def _source_cli(tmp_path, body: str, **env_extra) -> subprocess.CompletedProcess:
    harness = tmp_path / "harness.sh"
    harness.write_text('cli="$1"\nset -- env\nsource "$cli" >/dev/null\n' + body)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_GATEWAY_", "GATEWAY_VISION"))}
    env.update(HOME=str(tmp_path), MODEL_GATEWAY_LAUNCHD_LABEL="com.local.model-gateway-test-does-not-exist",
               MODEL_GATEWAY_PLIST_DIR=str(tmp_path / "LaunchAgents"), **env_extra)
    return subprocess.run(["bash", str(harness), str(ROOT / "bin/model-gateway")],
                          capture_output=True, text=True, env=env, timeout=60)


def test_cli_local_ai_remove_needs_no_gateway_launch_agent(tmp_path):
    result = _source_cli(tmp_path, 'run_local_ai() { echo "RAN $*"; }\ncmd_local_ai remove\n')
    assert result.returncode == 0, result.stderr
    assert result.stdout == "RAN remove\n"


def test_cli_uninstall_warns_about_this_gateways_local_ai(tmp_path):
    label = "com.local.model-gateway-test-does-not-exist"
    state = tmp_path / "state"
    omlx = tmp_path / "LaunchAgents/com.local.omlx.plist"
    body = "remove_plist() { :; }\ncmd_uninstall\n"
    for owner, warned in ((f"{state}#{label}", True), (f"{state}#com.local.model-gateway", False), (None, False)):
        write_plist(omlx, {"Label": "com.local.omlx", **({lr.OWNER_KEY: owner} if owner else {})})
        result = _source_cli(tmp_path, body, MODEL_GATEWAY_STATE_DIR=str(state))
        assert result.returncode == 0, result.stderr
        assert ("model-gateway local-ai remove" in result.stderr) is warned
    assert omlx.exists()


def test_the_manifest_is_read_through_a_current_symlink(tmp_path, monkeypatch):
    release = tmp_path / "releases/0.4.0-0123456789ab"
    shutil.copytree(ROOT / "local-models", release / "local-models")
    (tmp_path / "current").symlink_to("releases/0.4.0-0123456789ab")
    monkeypatch.setattr(lr, "MODELS_DIR", tmp_path / "current/local-models")
    assert lr.manifest(lr.MODELS["qwen3.8-27b"])["model_id"] == "qwen3.8-27b-8bit-30gb"


def test_shipped_payload_manifest_and_locked_runtime_match_the_code():
    payload = lr.manifest(QWEN)
    assert (payload["model_id"], payload["license"]) == (QWEN_ID, "apache-2.0")
    assert (lr.MODELS_DIR / "LICENSE-Qwen-Apache-2.0.txt").is_file() and (lr.MODELS_DIR / "NOTICE.md").is_file()
    project = tomllib.loads((lr.RUNTIME_PROJECT / "pyproject.toml").read_text())
    assert project["project"]["dependencies"] == [f"omlx=={lr.OMLX_VERSION}"]
    lock = (lr.RUNTIME_PROJECT / "uv.lock").read_text()
    assert f'name = "omlx"\nversion = "{lr.OMLX_VERSION}"' in lock
    assert 'name = "model-gateway-local-runtime"' in lock


def test_package_root_follows_the_package_root_file_when_run_from_a_release(tmp_path, monkeypatch):
    """The server imports from the resolved release; .package names the stable root."""
    release = tmp_path / "releases" / "r1"
    (release / "src").mkdir(parents=True)
    module = release / "src" / "local_runtime.py"
    module.write_text("")
    current = tmp_path / "current"
    (release / ".package").write_text(f"MANAGER=model-gateway-pkg\nROOT={current}\n")
    monkeypatch.delenv("MODEL_GATEWAY_PACKAGE_ROOT", raising=False)
    monkeypatch.setattr(lr, "__file__", str(module))
    assert lr._package_root() == current
    (release / ".package").unlink()
    assert lr._package_root() == release


def test_status_reports_whether_the_installed_model_is_loaded(monkeypatch):
    local = lr.MODELS["qwen3.8-27b"]
    monkeypatch.setattr(lr, "omlx_owner", lambda: "ours")
    monkeypatch.setattr(lr, "installed_models", lambda: [local])
    monkeypatch.setattr(lr, "_key", lambda *a, **k: "omlx-key")
    model_id = lr.manifest(local)["model_id"]
    for rows, expected in (([{"id": model_id, "loaded": False}], False),
                           ([{"id": model_id, "loaded": True}], True),
                           ([{"id": "other", "loaded": True}], None)):
        monkeypatch.setattr(lr, "_request", lambda *a, rows=rows, **k: {"models": rows})
        assert lr.status()["loaded"] is expected
    def unreachable(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr(lr, "_request", unreachable)
    assert lr.status()["loaded"] is None
    monkeypatch.setattr(lr, "omlx_owner", lambda: "external")
    assert lr.status()["loaded"] is None


# ── several models (docs/adr/0003-scoped-access-and-lightweight-local-models.md section 3) ──

SMALL = lr.MODELS["qwen3.8-27b-4bit"]
SMALL_ID = "qwen3.8-27b-4bit-16gb"


def running_runtime(gateway, launchctl, monkeypatch) -> list[str]:
    """This gateway's runtime built, loaded and wired with the 8-bit model; records oMLX reloads."""
    owned_install(9110)
    lr._key(lr.api_key_path(), create=True)
    (lr.venv_dir() / "bin").mkdir(parents=True)
    (lr.venv_dir() / "bin" / "omlx").write_text("")
    wire_local_ai_config(gateway)
    launchctl.loaded.update({lr.OMLX_LABEL, lr.GATEWAY_LABEL})
    reloads = []
    monkeypatch.setattr(lr, "_omlx_reload", lambda: reloads.append("reload"))
    return reloads


def test_a_second_model_joins_the_running_runtime_without_a_restart(eligible, gateway, launchctl, monkeypatch):
    reloads = running_runtime(gateway, launchctl, monkeypatch)
    payload = tiny_payload(SMALL_ID)
    monkeypatch.setattr(lr, "manifest", lambda local: payload if local is SMALL else REAL_MANIFEST(local))
    monkeypatch.setattr(lr, "_run", lambda *a, **k: pytest.fail("rebuilt oMLX"))
    before = lr.generation()
    lr.install(SMALL, payload)

    assert sorted(path.name for path in lr.mlx_dir().iterdir()) == sorted([QWEN_ID, SMALL_ID])
    settings = json.loads((lr.inference_dir() / "model_settings.json").read_text())["models"]
    assert {model_id: row["is_default"] for model_id, row in settings.items()} == {QWEN_ID: True, SMALL_ID: False}
    assert reloads == ["reload"]
    assert lr.generation() != before  # the running gateway reloads its catalog
    assert not [call for call in launchctl.calls if call[0] in {"kickstart", "bootstrap", "bootout"}]
    providers.reload()
    names = {row["name"]: row for row in providers.effective_model_inventory() if row.get("provider") == "omlx"}
    assert set(names) == {"qwen3.8-27b", "qwen3.8-27b-4bit"}
    assert names["qwen3.8-27b-4bit"]["omlx_id"] == SMALL_ID
    assert [m.name for m in lr.installed_models()] == ["qwen3.8-27b", "qwen3.8-27b-4bit"]


def test_the_base_model_name_is_served_by_whichever_qwen_is_installed(gateway, launchctl, monkeypatch):
    lr._key(lr.api_key_path(), create=True)
    wire_local_ai_config(gateway)
    write_plist(lr.omlx_plist_path(), lr.omlx_plist(9110))
    (lr.mlx_dir() / SMALL_ID).mkdir(parents=True)
    providers.reload()
    routed = providers.resolve("qwen3.8-27b")
    assert routed is not None and routed.provider_model_id == SMALL_ID


def test_promotion_refuses_a_foreign_directory_and_moves_old_partials(eligible, gateway, monkeypatch):
    payload = tiny_payload(staging=lr.mlx_dir())  # where earlier releases downloaded
    monkeypatch.setattr(lr, "manifest", qwen_manifest(payload))
    assert lr.partial_dir(QWEN_ID) == lr.staging_dir() / f"{QWEN_ID}.partial"
    assert lr.partial_dir(QWEN_ID).is_dir() and not (lr.mlx_dir() / f"{QWEN_ID}.partial").exists()
    (lr.mlx_dir() / "someone-elses-model").mkdir()
    with pytest.raises(lr.LocalRuntimeError, match="remove someone-elses-model"):
        lr._promote(payload)


def test_removing_one_of_several_models_keeps_the_runtime(eligible, gateway, launchctl, monkeypatch):
    reloads = running_runtime(gateway, launchctl, monkeypatch)
    (lr.mlx_dir() / SMALL_ID).mkdir()
    before = lr.generation()
    status = lr.remove("qwen3.8-27b-4bit")
    assert status["installed"] is False and status["model"] == "qwen3.8-27b-4bit"
    assert [path.name for path in lr.mlx_dir().iterdir()] == [QWEN_ID]
    assert list(json.loads((lr.inference_dir() / "model_settings.json").read_text())["models"]) == [QWEN_ID]
    assert reloads == ["reload"] and lr.generation() != before
    assert lr.omlx_plist_path().exists() and "omlx" in yaml.safe_load(gateway.read_text())["providers"]
    with pytest.raises(lr.LocalRuntimeError, match="is not installed"):
        lr.remove("qwen3.8-27b-4bit")


def test_removing_the_last_model_removes_the_runtime(eligible, gateway, launchctl, monkeypatch):
    running_runtime(gateway, launchctl, monkeypatch)
    monkeypatch.setattr(lr, "_wait_health", lambda url: None)
    lr.remove("qwen3.8-27b")
    assert not lr.runtime_root().exists() and not lr.omlx_plist_path().exists()
    assert "omlx" not in yaml.safe_load(gateway.read_text())["providers"]


def test_add_defaults_to_the_best_model_this_mac_can_run(gateway, launchctl, monkeypatch):
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)
    monkeypatch.setattr(lr, "_memory_bytes", lambda: 36 * GIB)
    bootstraps = queue(monkeypatch)
    status = lr.start_add()
    assert (status["model"], status["state"], len(bootstraps)) == ("qwen3.8-27b-4bit", "queued", 1)
    with pytest.raises(lr.LocalRuntimeError, match="needs Apple silicon and at least 48 GB"):
        lr.start_add("qwen3.8-27b")


def test_one_setup_job_at_a_time(eligible, gateway, launchctl, monkeypatch):
    queue(monkeypatch)
    lr.start_add("qwen3.8-27b")
    monkeypatch.setattr(lr, "_job_alive", lambda: True)
    with pytest.raises(lr.LocalRuntimeError, match="installing qwen3.8-27b; add qwen3.8-27b-4bit when it finishes"):
        lr.start_add("qwen3.8-27b-4bit")
    assert lr.start_add("qwen3.8-27b")["state"] == "queued"


def test_status_describes_each_model(eligible, monkeypatch):
    monkeypatch.setattr(lr, "omlx_owner", lambda: "gateway")
    monkeypatch.setattr(lr, "installed_models", lambda: [QWEN, SMALL])
    monkeypatch.setattr(lr, "_key", lambda *a, **k: "omlx-key")
    monkeypatch.setattr(lr, "_request", lambda *a, **k: {"models": [
        {"id": QWEN_ID, "loaded": True}, {"id": SMALL_ID, "loaded": False}]})
    value = lr.status("qwen3.8-27b-4bit")
    assert (value["model"], value["installed"], value["loaded"], value["state"]) == (
        "qwen3.8-27b-4bit", True, False, "done")
    assert [(row["name"], row["loaded"]) for row in value["models"]] == [
        ("qwen3.8-27b", True), ("qwen3.8-27b-4bit", False)]
    # A job for one model never reports as another's progress.
    lr._atomic(lr.status_path(), json.dumps({"state": "failed", "model": "qwen3.8-27b-4bit", "message": "x",
                                             "updated_at": lr._now()}))
    assert lr.status("qwen3.8-27b")["state"] == "done"
    assert lr.status("qwen3.8-27b-4bit")["state"] == "failed"


# ── pins: models local profiles use stay loaded ──────────────────────────────


def pins(installed, names, memory=512 * GIB):
    settings = lr.model_settings(installed, names, memory)["models"]
    return {model_id: (row["is_default"], row["is_pinned"]) for model_id, row in settings.items()}


def test_model_settings_pin_models_that_local_profiles_route_to():
    assert pins([QWEN, SMALL], {"qwen3.8-27b-4bit"}) == {QWEN_ID: (True, False), SMALL_ID: (False, True)}
    # A route to the 8-bit is served by the 4-bit once only the 4-bit is installed, so that is pinned.
    assert pins([SMALL], {"qwen3.8-27b"}) == {SMALL_ID: (True, True)}
    # Never more than the memory budget, whatever registered profiles route to.
    assert pins([QWEN, SMALL], {"qwen3.8-27b", "qwen3.8-27b-4bit"}, 48 * GIB) == {
        QWEN_ID: (True, True), SMALL_ID: (False, False)}


def test_pins_must_fit_the_macs_memory(monkeypatch):
    monkeypatch.setattr(lr, "installed_models", lambda: [QWEN, SMALL])
    monkeypatch.setattr(lr, "_memory_bytes", lambda: 48 * GIB)
    assert lr.pin_budget_error({"qwen3.8-27b"}) is None
    assert lr.pin_budget_error({"not-local"}) is None
    error = lr.pin_budget_error({"qwen3.8-27b", "qwen3.8-27b-4bit"})
    assert error is not None and "more than 75% of this Mac's memory" in error


def test_sync_pins_rewrites_settings_and_reloads_only_on_change(eligible, gateway, launchctl, monkeypatch):
    reloads = running_runtime(gateway, launchctl, monkeypatch)
    pinned = set()
    monkeypatch.setattr(lr, "_pinned_names", lambda: pinned)
    assert lr.sync_pins() is True and reloads == ["reload"]
    assert lr.sync_pins() is False and reloads == ["reload"]
    pinned.add("qwen3.8-27b")
    assert lr.sync_pins() is True
    assert json.loads((lr.inference_dir() / "model_settings.json").read_text())["models"][QWEN_ID]["is_pinned"]
    with lr.provision_lock(wait=True):  # the setup job writes the same settings
        pinned.clear()
        assert lr.sync_pins() is False


def test_sync_pins_does_nothing_without_this_gateways_runtime(monkeypatch):
    monkeypatch.setattr(lr, "_omlx_reload", lambda: pytest.fail("reloaded"))
    assert lr.sync_pins() is False
    lr.runtime_root().mkdir(parents=True)
    external_omlx()
    assert lr.sync_pins() is False


def test_omlx_reload_logs_in_with_the_runtime_key_then_reloads(monkeypatch):
    lr._key(lr.api_key_path(), create=True)
    key = lr.api_key_path().read_text().strip()
    seen = []

    class Headers(list):
        def items(self):
            return iter(self)

    class Response:
        status = 200

        def __init__(self, headers=()):
            self.headers = Headers(headers)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Opener:
        def open(self, request, timeout):
            seen.append((request.full_url, request.data, dict(request.header_items())))
            if request.full_url.endswith("/login"):
                return Response([("Set-Cookie", "omlx_admin_session=abc; HttpOnly; Path=/")])
            return Response()

    monkeypatch.setattr(lr.urllib.request, "build_opener", lambda *handlers: Opener())
    lr._omlx_reload()
    assert seen[0][0] == "http://127.0.0.1:9110/admin/api/login" and json.loads(seen[0][1]) == {"api_key": key}
    assert seen[1][0] == "http://127.0.0.1:9110/admin/api/reload" and seen[1][2]["Cookie"] == "omlx_admin_session=abc"


def test_the_gateway_reloads_its_catalog_when_the_local_models_change(monkeypatch):
    import asyncio

    from src import admin

    class Done(Exception):
        pass

    sequence = iter(["a", "a", "b", "b", "c", "c"])

    def generation():
        try:
            return next(sequence)
        except StopIteration:
            raise Done from None

    reloads, errors = [], iter([None, "rejected", None])

    def reload():
        reloads.append("reload")
        return next(errors)

    async def regenerate():
        return "regenerated"

    monkeypatch.setattr(lr, "generation", generation)
    monkeypatch.setattr(admin, "_reload_for_local_ai", reload)
    monkeypatch.setattr(admin, "_regenerate_catalogs", regenerate)
    monkeypatch.setattr(admin.discovery, "refresh", lambda: reloads.append("refresh"))

    async def run():
        with pytest.raises(Done):
            await admin.watch_local_ai(poll_seconds=0)

    asyncio.run(run())
    # a -> b reloads; b -> c is rejected once, then retried and applied.
    assert reloads == ["reload", "refresh", "reload", "reload", "refresh"]


def test_registering_a_local_profile_pins_its_model_within_the_memory_budget(monkeypatch):
    from src import profiles
    from tests.test_consumer_profiles import _configure, _manifest, _register

    _configure(monkeypatch)
    budget = []
    monkeypatch.setattr(lr, "pin_budget_error", lambda names: budget.append(set(names)) or None)
    synced = []
    monkeypatch.setattr(lr, "sync_pins", lambda: synced.append(1) or True)
    first = _register()
    assert first.status_code == 200
    assert budget == [{"test-local"}] and synced == [1]
    assert profiles.local_route_models() == {"test-local"}

    etag = first.headers["ETag"]
    monkeypatch.setattr(lr, "pin_budget_error", lambda names: "Keeping test-local loaded needs 99 GB")
    # A new revision routing to the same models pins nothing new, so it is not re-checked.
    from tests.test_consumer_profiles import _headers, client
    same = client.put("/v1/profiles/ha/snapshot", headers={**_headers(), "If-Match": etag},
                      json=_manifest(source_revision="ha@two"))
    assert same.status_code == 200
    monkeypatch.setattr(profiles, "_local_routes", lambda namespaces: {"test-local", "another-local"}
                        if namespaces.get("ha", [{}])[-1].get("manifest", {}).get("source_revision") == "ha@three"
                        else {"test-local"})
    refused = client.put("/v1/profiles/ha/snapshot", headers={**_headers(), "If-Match": same.headers["ETag"]},
                         json=_manifest(source_revision="ha@three"))
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "local_memory_exceeded"


def test_a_failed_omlx_reload_is_retried(eligible, gateway, launchctl, monkeypatch):
    running_runtime(gateway, launchctl, monkeypatch)
    pinned = {"qwen3.8-27b"}
    monkeypatch.setattr(lr, "_pinned_names", lambda: pinned)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("oMLX is busy")

    monkeypatch.setattr(lr, "_omlx_reload", flaky)
    before = lr._read_settings()
    with pytest.raises(OSError):
        lr.sync_pins()
    assert lr._read_settings() == before  # restored, so the retry sees the change again
    assert lr.sync_pins() is True and len(attempts) == 2


def test_a_failed_join_takes_the_new_model_back_out(eligible, gateway, launchctl, monkeypatch):
    reloads = running_runtime(gateway, launchctl, monkeypatch)
    payload = tiny_payload(SMALL_ID)
    monkeypatch.setattr(lr, "manifest", lambda local: payload if local is SMALL else REAL_MANIFEST(local))
    monkeypatch.setattr(lr, "configure_gateway", lambda *a, **k: (_ for _ in ()).throw(
        lr.LocalRuntimeError("The gateway catalog already has a different model named 'qwen3.8-27b-4bit'")))
    before = lr.generation()
    with pytest.raises(lr.LocalRuntimeError, match="different model"):
        lr.install(SMALL, payload)
    assert [path.name for path in lr.mlx_dir().iterdir()] == [QWEN_ID]
    assert (lr.staging_dir() / f"{SMALL_ID}.partial").is_dir()  # kept for the next add
    assert list(json.loads(lr._read_settings())["models"]) == [QWEN_ID]
    assert reloads == ["reload", "reload"] and lr.generation() != before


def test_a_failed_removal_keeps_serving_the_model(eligible, gateway, launchctl, monkeypatch):
    running_runtime(gateway, launchctl, monkeypatch)
    (lr.mlx_dir() / SMALL_ID).mkdir()
    lr._write_model_settings(lr._promoted_models())
    monkeypatch.setattr(lr, "_omlx_reload", lambda: (_ for _ in ()).throw(OSError("oMLX is busy")))
    with pytest.raises(OSError):
        lr.remove("qwen3.8-27b-4bit")
    assert sorted(path.name for path in lr.mlx_dir().iterdir()) == sorted([QWEN_ID, SMALL_ID])
    assert set(json.loads(lr._read_settings())["models"]) == {QWEN_ID, SMALL_ID}


def test_default_status_reports_a_failed_add_of_a_model_not_installed(eligible, monkeypatch):
    monkeypatch.setattr(lr, "omlx_owner", lambda: "gateway")
    monkeypatch.setattr(lr, "installed_models", lambda: [QWEN])
    monkeypatch.setattr(lr, "_loaded_models", lambda installed: {})
    lr._atomic(lr.status_path(), json.dumps({"state": "failed", "model": "qwen3.8-27b-4bit", "message": "no disk",
                                             "updated_at": lr._now()}))
    value = lr.status()
    assert (value["model"], value["state"], value["message"]) == ("qwen3.8-27b-4bit", "failed", "no disk")
    lr._atomic(lr.status_path(), json.dumps({"state": "done", "model": "qwen3.8-27b-4bit", "updated_at": lr._now()}))
    assert lr.status()["model"] == "qwen3.8-27b"


def test_the_watcher_outlives_a_reload_that_raises(monkeypatch):
    import asyncio

    from src import admin

    class Done(Exception):
        pass

    sequence = iter(["a", "b", "b"])

    def generation():
        try:
            return next(sequence)
        except StopIteration:
            raise Done from None

    outcomes = iter([RuntimeError("another config write holds the lock"), None])
    reloads = []

    def reload():
        reloads.append(1)
        outcome = next(outcomes)
        if outcome:
            raise outcome

    async def regenerate():
        return "ok"

    monkeypatch.setattr(lr, "generation", generation)
    monkeypatch.setattr(admin, "_reload_for_local_ai", reload)
    monkeypatch.setattr(admin, "_regenerate_catalogs", regenerate)
    monkeypatch.setattr(admin.discovery, "refresh", lambda: None)

    async def run():
        with pytest.raises(Done):
            await admin.watch_local_ai(poll_seconds=0)

    asyncio.run(run())
    assert len(reloads) == 2


def test_status_api_refuses_an_unknown_model(gateway):
    response = TestClient(app).get("/admin/api/local-ai?model=bogus", headers={"Authorization": "Bearer admin"})
    assert response.status_code == 400 and "qwen3.8-27b-4bit" in response.text


def test_a_foreign_directory_stops_promotion_before_the_payload_moves(eligible, gateway, monkeypatch):
    payload = tiny_payload()
    (lr.mlx_dir() / "someone-elses-model").mkdir(parents=True)
    with pytest.raises(lr.LocalRuntimeError, match="remove someone-elses-model"):
        lr._promote(payload)
    assert not (lr.mlx_dir() / QWEN_ID).exists() and lr.partial_dir(QWEN_ID).is_dir()


def test_removal_succeeds_even_when_pins_cannot_sync_afterwards(eligible, gateway, launchctl, monkeypatch):
    running_runtime(gateway, launchctl, monkeypatch)
    (lr.mlx_dir() / SMALL_ID).mkdir()
    lr._atomic(lr.status_path(), json.dumps({"state": "done", "model": "qwen3.8-27b-4bit", "updated_at": lr._now()}))
    monkeypatch.setattr(lr, "sync_pins", lambda: (_ for _ in ()).throw(OSError("oMLX is busy")))
    value = lr.remove("qwen3.8-27b-4bit")
    assert (value["installed"], value["state"]) == (False, None)


# ── on-demand install (docs/adr/0003-scoped-access-and-lightweight-local-models.md section 5) ──


@pytest.fixture
def on_demand(eligible, gateway, launchctl, monkeypatch):
    """The 8-bit installed and running; the 4-bit listed for on-demand install.

    Here the 4-bit has a base model of its own: with the curated one (the 8-bit's), the
    installed 8-bit serves its name and nothing installs (ADR 0003 section 4's order).
    """
    from dataclasses import replace

    monkeypatch.setitem(lr.MODELS, SMALL.name, replace(SMALL, catalog={**SMALL.catalog, "base_model": "qwen-small"}))
    running_runtime(gateway, launchctl, monkeypatch)
    config = yaml.safe_load(gateway.read_text())
    config["local_ai"] = {"auto_install": ["qwen3.8-27b-4bit"]}
    gateway.write_text(yaml.safe_dump(config))
    providers.reload()
    lr._on_demand_started.clear()
    starts = []

    def start(name):
        starts.append(name)
        lr._atomic(lr.status_path(), json.dumps({
            "state": "queued", "model": name, "bytes_done": 0, "bytes_total": 16_000_000_000,
            "updated_at": lr._now()}))
        return lr.status(name)

    monkeypatch.setattr(lr, "start_add", start)
    return starts


def chat(model, token="admin"):
    return TestClient(app).post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"},
                                json={"model": model, "messages": [{"role": "user", "content": "hi"}]})


def test_a_request_starts_installing_a_listed_model_and_returns_at_once(on_demand):
    response = chat("qwen3.8-27b-4bit")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_installing"
    assert int(response.headers["Retry-After"]) == 16_000_000_000 // lr.DOWNLOAD_BYTES_PER_SECOND + 60
    # Later requests follow the job instead of starting another.
    again = chat("qwen3.8-27b-4bit")
    assert again.json()["error"]["code"] == "model_installing" and on_demand == ["qwen3.8-27b-4bit"]


def test_only_listed_models_install_on_demand(on_demand, gateway):
    config = yaml.safe_load(gateway.read_text())
    config["local_ai"] = {"auto_install": []}
    gateway.write_text(yaml.safe_dump(config))
    providers.reload()
    assert chat("qwen3.8-27b-4bit").status_code == 404 and on_demand == []


def test_a_busy_or_failed_job_is_reported_not_retried(on_demand):
    lr._atomic(lr.status_path(), json.dumps({"state": "downloading", "model": "qwen3.8-27b", "bytes_done": 1,
                                             "bytes_total": 2, "updated_at": lr._now()}))
    assert chat("qwen3.8-27b-4bit").json()["error"]["code"] == "model_install_busy"
    lr._atomic(lr.status_path(), json.dumps({"state": "failed", "model": "qwen3.8-27b-4bit", "message": "no disk",
                                             "updated_at": lr._now()}))
    response = chat("qwen3.8-27b-4bit")
    assert response.json()["error"]["code"] == "model_not_installable"
    assert "no disk" not in response.text  # why is for the owner (local AI status)
    assert "Retry-After" not in response.headers and on_demand == []


def test_an_interrupted_download_resumes_on_the_next_request(on_demand):
    stale = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    lr._atomic(lr.status_path(), json.dumps({"state": "downloading", "model": "qwen3.8-27b-4bit", "bytes_done": 5,
                                             "bytes_total": 10, "updated_at": stale}))
    assert chat("qwen3.8-27b-4bit").json()["error"]["code"] == "model_installing"
    assert on_demand == ["qwen3.8-27b-4bit"]


def test_a_refused_start_is_answered_from_cache_until_the_window_passes(on_demand, monkeypatch):
    attempts = []

    def refuse(name):
        attempts.append(name)
        raise lr.LocalRuntimeError(f"Local AI needs 30 GB of free disk space in {lr.runtime_root()}")

    monkeypatch.setattr(lr, "start_add", refuse)
    for _ in range(3):
        response = chat("qwen3.8-27b-4bit")
        assert response.json()["error"]["code"] == "model_not_installable"
        assert "free disk space" not in response.text and str(lr.runtime_root()) not in response.text
    assert attempts == ["qwen3.8-27b-4bit"]
    started_at, answer = lr._on_demand_started["qwen3.8-27b-4bit"]
    lr._on_demand_started["qwen3.8-27b-4bit"] = (started_at - lr.ON_DEMAND_START_SECONDS, answer)
    chat("qwen3.8-27b-4bit")
    assert attempts == ["qwen3.8-27b-4bit"] * 2


def test_another_models_live_job_is_busy_not_refused(on_demand, monkeypatch):
    monkeypatch.setattr(lr, "start_add", lambda name: (_ for _ in ()).throw(lr._InstallBusy("installing x")))
    assert chat("qwen3.8-27b-4bit").json()["error"]["code"] == "model_install_busy"


@pytest.mark.parametrize("path, body", [
    ("/v1/responses", {"input": "hi"}),
    ("/v1/messages", {"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
])
def test_every_endpoint_installs_on_demand(on_demand, path, body):
    response = TestClient(app).post(path, headers={"Authorization": "Bearer admin"},
                                    json={"model": "qwen3.8-27b-4bit", **body})
    assert response.status_code == 503 and "Retry-After" in response.headers
    assert on_demand == ["qwen3.8-27b-4bit"]


def test_a_disabled_or_installed_model_never_installs_on_demand(on_demand, gateway, monkeypatch):
    config = yaml.safe_load(gateway.read_text())
    config["model_overrides"] = {"qwen3.8-27b": {"enabled": False}}
    config["local_ai"] = {"auto_install": ["qwen3.8-27b-4bit", "qwen3.8-27b"]}
    gateway.write_text(yaml.safe_dump(config))
    providers.reload()
    # The 8-bit exists here but is turned off: it stays off, nothing downloads.
    assert chat("qwen3.8-27b").status_code == 404
    assert lr.on_demand_model("qwen3.8-27b", config) is None
    assert on_demand == []


def test_auto_install_errors_are_reported_to_the_owner(gateway):
    config = yaml.safe_load(gateway.read_text())
    config["local_ai"] = {"auto_install": ["gpt-6"]}
    gateway.write_text(yaml.safe_dump(config))
    value = lr.status()
    assert value["auto_install"] == [] and "local_ai.auto_install must list" in value["auto_install_error"]


def test_nothing_installs_on_demand_without_a_runtime(gateway, eligible, monkeypatch):
    config = yaml.safe_load(gateway.read_text())
    config["local_ai"] = {"auto_install": ["qwen3.8-27b-4bit"]}
    gateway.write_text(yaml.safe_dump(config))
    providers.reload()
    monkeypatch.setattr(lr, "start_add", lambda name: pytest.fail("installed on demand"))
    assert chat("qwen3.8-27b-4bit").status_code == 404
    assert lr.on_demand_models(config) == []


def test_restricted_credentials_install_on_demand_only_what_they_list(on_demand, gateway):
    config = yaml.safe_load(gateway.read_text())
    config["auth"]["consumer_credentials"].append({
        "id": "bot-runtime", "consumer": "bot", "key": "bot-token", "namespaces": ["bot"],
        "permissions": ["profiles:invoke"], "allow_direct_models": True, "models": ["qwen3.8-27b"]})
    gateway.write_text(yaml.safe_dump(config))
    providers.reload()
    assert chat("qwen3.8-27b-4bit", token="bot-token").status_code == 403
    assert on_demand == []


def test_canonical_lists_on_demand_models_only_when_asked(on_demand):
    client = TestClient(app)
    plain = client.get("/v1/models/canonical", headers={"Authorization": "Bearer admin"}).json()["data"]
    assert "qwen3.8-27b-4bit" not in {row["id"] for row in plain}
    assert all("local_install" not in row for row in plain)
    rows = {row["id"]: row for row in client.get("/v1/models/canonical?include=local_install",
                                                 headers={"Authorization": "Bearer admin"}).json()["data"]}
    assert rows["qwen3.8-27b-4bit"]["local_install"] == "on_demand"
    assert (rows["qwen3.8-27b-4bit"]["available"], rows["qwen3.8-27b-4bit"]["scope"]) == (False, "local")
    assert rows["qwen3.8-27b"]["local_install"] == ""
    models = client.get("/v1/models", headers={"Authorization": "Bearer admin"}).json()["data"]
    assert "qwen3.8-27b-4bit" not in {row["id"] for row in models}


def test_auto_install_must_name_local_ai_models():
    with pytest.raises(ValueError, match="local_ai.auto_install"):
        lr.auto_install_names({"local_ai": {"auto_install": ["gpt-6"]}})
    assert lr.auto_install_names({}) == []


def test_a_model_whose_base_is_served_here_is_never_installed_on_demand(eligible, gateway, launchctl, monkeypatch):
    running_runtime(gateway, launchctl, monkeypatch)
    config = yaml.safe_load(gateway.read_text())
    config["local_ai"] = {"auto_install": ["qwen3.8-27b-4bit"]}
    gateway.write_text(yaml.safe_dump(config))
    providers.reload()
    monkeypatch.setattr(lr, "start_add", lambda name: pytest.fail("installed on demand"))
    served = []

    async def passthrough(endpoint, body, headers, provider="", request=None, **_kwargs):
        served.append(body["model"])
        return JSONResponse({"choices": []})

    from src import server
    monkeypatch.setattr(server, "_passthrough_sync", passthrough)
    assert chat("qwen3.8-27b-4bit").status_code == 200
    assert served == [QWEN_ID]  # the installed 8-bit serves the 4-bit's base model


def test_a_credential_limited_to_the_missing_model_installs_it(eligible, gateway, launchctl, monkeypatch):
    running_runtime(gateway, launchctl, monkeypatch)
    config = yaml.safe_load(gateway.read_text())
    config["local_ai"] = {"auto_install": ["qwen3.8-27b-4bit"]}
    config["auth"]["consumer_credentials"].append({
        "id": "bot-runtime", "consumer": "bot", "key": "bot-token", "namespaces": ["bot"],
        "permissions": ["profiles:invoke"], "allow_direct_models": True, "models": ["qwen3.8-27b-4bit"]})
    gateway.write_text(yaml.safe_dump(config))
    providers.reload()
    lr._on_demand_started.clear()
    starts = []
    monkeypatch.setattr(lr, "start_add", lambda name: starts.append(name) or lr.status(name))
    response = chat("qwen3.8-27b-4bit", token="bot-token")
    assert response.status_code == 503 and response.json()["error"]["code"] == "model_installing"
    assert starts == ["qwen3.8-27b-4bit"]


# ── owner-pinned open models (docs/adr/0003-scoped-access-and-lightweight-local-models.md section 6) ──

PIN_REV = "c" * 40


def pin_host(*, files=None):
    from tests.test_hf_manifest import Host, REV

    weights = b"single-file weights"
    host = Host({"config.json": json.dumps({"model_type": "llama"}).encode(), **(files or {})},
                lfs={"model.safetensors": (hashlib.sha256(weights).hexdigest(), len(weights))},
                info={"sha": PIN_REV, "gated": False, "cardData": {"license": "mit"}})
    # The fake serves revision REV paths; answer for PIN_REV too.
    original = host.open

    def open_(request, timeout):
        request.full_url = request.full_url.replace(PIN_REV, REV)
        return original(request, timeout)

    host.open = open_
    host.weights = weights
    return host


@pytest.fixture
def open_models(gateway):
    config = yaml.safe_load(gateway.read_text())
    config["local_ai"] = {"open_models": True}
    gateway.write_text(yaml.safe_dump(config))
    providers.reload()
    return gateway


def test_pinning_needs_the_owners_opt_in(gateway):
    with pytest.raises(lr.LocalRuntimeError, match="local_ai.open_models: true"):
        lr.pin_model("tiny", "org/model", PIN_REV, opener=pin_host())


def test_a_pin_records_a_verified_manifest_and_joins_the_local_models(open_models):
    from tests.test_hf_manifest import REPO

    host = pin_host()
    pinned = lr.pin_model("tiny", REPO, PIN_REV, context=8192, opener=host)
    assert (pinned["license"], pinned["min_memory_gb"]) == ("mit", 16)
    path = lr.pins_dir() / "tiny.json"
    assert path.stat().st_mode & 0o777 == 0o600
    local = lr.models()["tiny"]
    assert lr.manifest(local)["model_id"] == "pin-tiny"
    assert "base_model" not in local.catalog and local.settings["max_context_window"] == 8192
    assert [row["source"] for row in lr.status()["models"]] == ["release", "release", "pin"]
    # A single-file payload verifies against the pin, and installs like a curated model.
    staged = lr.partial_dir("pin-tiny")
    staged.mkdir(parents=True)
    (staged / "config.json").write_bytes(json.dumps({"model_type": "llama"}).encode())
    (staged / "model.safetensors").write_bytes(host.weights)
    with pytest.raises(lr.LocalRuntimeError, match="failed verification"):
        lr._promote(lr.manifest(local))  # the curated rule: a single file needs the pin's own rule
    lr._promote(lr.manifest(local), single_file_ok=True)
    assert (lr.mlx_dir() / "pin-tiny").is_dir()


@pytest.mark.parametrize("name, message", [
    ("Tiny", "lowercase"),
    ("qwen3.8-27b-4bit", "already a model"),
    ("qwen3.8-27b", "already a model"),
    ("test-local", "already a model"),  # an identifier in the owner's catalog
])
def test_pins_never_reuse_a_name(open_models, monkeypatch, name, message):
    monkeypatch.setattr(providers, "_models", {"test-local": {"name": "test-local", "provider": "omlx"}})
    with pytest.raises(lr.LocalRuntimeError, match=message):
        lr.pin_model(name, "org/model", PIN_REV, opener=pin_host())


def test_a_repository_with_code_cannot_be_pinned(open_models):
    from tests.test_hf_manifest import REPO

    with pytest.raises(lr.LocalRuntimeError, match="cannot be pinned"):
        lr.pin_model("tiny", REPO, PIN_REV, opener=pin_host(files={"modeling.py": b"import os\n"}))
    assert not (lr.pins_dir() / "tiny.json").exists()


def test_a_pinned_model_loads_once_installed_and_wins_over_a_release_row(open_models, gateway, monkeypatch):
    from src import shipped
    from tests.test_hf_manifest import REPO

    lr.pin_model("tiny", REPO, PIN_REV, opener=pin_host())
    wire_local_ai_config(gateway)
    lr._key(lr.api_key_path(), create=True)
    write_plist(lr.omlx_plist_path(), lr.omlx_plist(9110))
    # A later release ships its own "tiny" for a provider this install serves.
    monkeypatch.setattr(shipped, "_packaged", lambda: {"later": {
        "auto_add": True, "provider": {"id": "omlx", "base_url": "http://127.0.0.1:9110/v1"},
        "models": [{"name": "tiny", "provider": "omlx", "omlx_id": "someone-elses-tiny",
                    "pricing_status": "unmetered"}]}})
    providers.reload()
    # Not installed: the pin does not load, and the release's row does.
    (row,) = [row for row in providers.effective_model_inventory() if row["name"] == "tiny"]
    assert row["omlx_id"] == "someone-elses-tiny"
    (lr.mlx_dir() / "pin-tiny").mkdir(parents=True)
    providers.reload()
    (row,) = [row for row in providers.effective_model_inventory() if row["name"] == "tiny"]
    assert (row["omlx_id"], row["origin"]["profile"]) == ("pin-tiny", "local-ai-pin")
    assert row["origin"]["displaced"] == ["tiny"]  # visible in admin status, not silent
    assert "base_model" not in row


def test_unpinning_needs_the_model_removed_first(open_models):
    from tests.test_hf_manifest import REPO

    lr.pin_model("tiny", REPO, PIN_REV, opener=pin_host())
    (lr.mlx_dir() / "pin-tiny").mkdir(parents=True)
    with pytest.raises(lr.LocalRuntimeError, match="remove it first"):
        lr.unpin_model("tiny")
    (lr.mlx_dir() / "pin-tiny").rmdir()
    lr.unpin_model("tiny")
    assert "tiny" not in lr.models()
    with pytest.raises(lr.LocalRuntimeError, match="not a pinned model"):
        lr.unpin_model("tiny")


def test_only_a_full_admin_pins_over_the_api(open_models, monkeypatch):
    from tests.test_hf_manifest import REPO

    monkeypatch.setattr("src.hf_manifest._opener", lambda: pin_host())
    client = TestClient(app)
    body = {"name": "tiny", "repo": REPO, "revision": PIN_REV}
    assert client.post("/admin/api/local-ai/pins", headers={"Authorization": "Bearer local-token"},
                       json=body).status_code in {401, 403}
    created = client.post("/admin/api/local-ai/pins", headers={"Authorization": "Bearer admin"}, json=body)
    assert created.status_code == 200 and created.json()["name"] == "tiny"
    again = client.post("/admin/api/local-ai/pins", headers={"Authorization": "Bearer admin"}, json=body)
    assert again.status_code == 409
    assert client.delete("/admin/api/local-ai/pins/tiny", headers={"Authorization": "Bearer admin"}).json() == {
        "unpinned": "tiny"}


def test_removing_local_ai_keeps_the_owners_pins(open_models, gateway, launchctl, monkeypatch):
    from tests.test_hf_manifest import REPO

    lr.pin_model("tiny", REPO, PIN_REV, opener=pin_host())
    running_runtime(gateway, launchctl, monkeypatch)
    monkeypatch.setattr(lr, "_wait_health", lambda url: None)
    lr.remove()
    assert not lr.runtime_root().exists() and "tiny" in lr.models()


def test_a_pin_not_installed_never_reroutes_to_the_cloud(open_models, monkeypatch):
    from tests.test_hf_manifest import REPO

    lr.pin_model("tiny", REPO, PIN_REV, opener=pin_host())
    config = yaml.safe_load(config_io.CONFIG_PATH.read_text())
    config["providers"]["cloud-test"] = {"base_url": "https://cloud.invalid/v1", "api_key": "secret"}
    # A later release ships a cloud model whose base model has the pin's name.
    config["models"] = [{"name": "tiny-cloud", "base_model": "tiny", "family": "tiny", "provider": "cloud-test",
                         "provider_model_id": "tiny-native", "pricing": {"input": 1.0, "output": 2.0}}]
    config_io.CONFIG_PATH.write_text(yaml.safe_dump(config))
    providers.reload()
    assert providers.base_model_route("tiny") is None


def test_an_invalid_pin_blocks_nothing_and_can_be_unpinned(open_models, monkeypatch):
    lr.pins_dir().mkdir(parents=True)
    (lr.pins_dir() / "broken.json").write_text(json.dumps({"schema_version": 1, "name": "broken",
                                                           "manifest": {"model_id": "elsewhere"}}))
    assert "broken" not in lr.models()
    payload = tiny_payload()
    monkeypatch.setattr(lr, "manifest", qwen_manifest(payload))
    (lr.mlx_dir() / "pin-broken").mkdir(parents=True)  # its payload, installed before the pin broke
    lr._promote(payload)  # a pin-* payload is ours, never foreign
    lr.unpin_model("broken")  # and unpin takes it out with the broken pin
    assert not (lr.mlx_dir() / "pin-broken").exists() and not (lr.pins_dir() / "broken.json").exists()


def test_a_pin_file_cannot_reshape_routing(open_models):
    from tests.test_hf_manifest import REPO

    lr.pin_model("tiny", REPO, PIN_REV, opener=pin_host())
    path = lr.pins_dir() / "tiny.json"
    document = json.loads(path.read_text())
    document["catalog"].update(base_model="qwen3.8-27b", alias="gpt-6", provider="cloud-test", context="big")
    document["settings"].update(is_default=True)
    path.write_text(json.dumps(document))
    local = lr.models()["tiny"]
    assert set(local.catalog) <= set(lr._PIN_CATALOG_FIELDS) and "context" not in local.catalog
    assert set(local.settings) <= set(lr._PIN_SETTINGS_FIELDS)
    document["manifest"]["files"].append({"path": "modeling.py", "size_bytes": 0, "sha256": "0" * 64})
    path.write_text(json.dumps(document))
    assert "tiny" not in lr.models()


def test_a_release_model_of_the_same_name_shadows_a_pin(open_models, monkeypatch):
    lr.pins_dir().mkdir(parents=True)
    (lr.pins_dir() / "qwen3.8-27b-4bit.json").write_text("{}")
    assert lr.models()["qwen3.8-27b-4bit"] is lr.MODELS["qwen3.8-27b-4bit"]


def test_a_listed_pin_cannot_be_unpinned(open_models, gateway):
    from tests.test_hf_manifest import REPO

    lr.pin_model("tiny", REPO, PIN_REV, opener=pin_host())
    config = yaml.safe_load(gateway.read_text())
    config["local_ai"]["auto_install"] = ["tiny"]
    gateway.write_text(yaml.safe_dump(config))
    with pytest.raises(lr.PinError, match="auto_install"):
        lr.unpin_model("tiny")


def test_pin_api_status_codes(gateway, monkeypatch):
    from tests.test_hf_manifest import REPO

    client = TestClient(app)
    admin = {"Authorization": "Bearer admin"}
    body = {"name": "tiny", "repo": REPO, "revision": PIN_REV}
    assert client.post("/admin/api/local-ai/pins", headers=admin, json=body).status_code == 403
    config = yaml.safe_load(gateway.read_text())
    config["local_ai"] = {"open_models": True}
    gateway.write_text(yaml.safe_dump(config))
    assert client.post("/admin/api/local-ai/pins", headers=admin, json={**body, "context": 0}).status_code == 400
    assert client.delete("/admin/api/local-ai/pins/nothing", headers=admin).status_code == 404
    monkeypatch.setattr("src.hf_manifest._opener", lambda: pin_host(files={"run.py": b""}))
    assert client.post("/admin/api/local-ai/pins", headers=admin, json=body).status_code == 422
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_WRITES")
    assert client.post("/admin/api/local-ai/pins", headers=admin, json=body).status_code == 403


def test_unpin_takes_out_an_orphaned_payload(open_models, gateway, launchctl, monkeypatch):
    reloads = running_runtime(gateway, launchctl, monkeypatch)
    lr.pins_dir().mkdir(parents=True)
    (lr.pins_dir() / "broken.json").write_text("{}")  # the pin broke after it was installed
    (lr.mlx_dir() / "pin-broken").mkdir()
    before = lr.generation()
    lr.unpin_model("broken")
    assert not (lr.mlx_dir() / "pin-broken").exists() and not (lr.pins_dir() / "broken.json").exists()
    assert reloads == ["reload"] and lr.generation() != before


def test_an_uninstalled_pin_is_never_served_by_another_local_route(open_models, monkeypatch):
    from tests.test_hf_manifest import REPO

    lr.pin_model("tiny", REPO, PIN_REV, opener=pin_host())
    config = yaml.safe_load(config_io.CONFIG_PATH.read_text())
    config["providers"]["omlx"] = {"base_url": "http://127.0.0.1:9110/v1", "api_key": "x"}
    config["models"] = [{"name": "owner-tiny", "base_model": "tiny", "family": "tiny", "provider": "omlx",
                         "omlx_id": "owner-tiny-mlx", "pricing_status": "unmetered"}]
    config_io.CONFIG_PATH.write_text(yaml.safe_dump(config))
    providers.reload()
    assert providers.base_model_route("tiny") is None
