"""Gateway-managed media generation (docs/adr/0004-local-image-and-video-generation.md section 3):
offline, in tmp, with every system edge faked."""

from __future__ import annotations

import json
import plistlib
import subprocess
from pathlib import Path

import pytest
import yaml

import src.providers as providers
from src import config_io, local_runtime as lr, media_runtime as mr, shipped

GIB = 1024**3
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"providers": {"fireworks": {"base_url": "https://api.fireworks.ai/v1"}}}))
    config.chmod(0o600)
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"allow_empty": True, "llm": []}))
    for module in (providers, config_io):
        monkeypatch.setattr(module, "CONFIG_PATH", config)
        monkeypatch.setattr(module, "MODEL_INFO_PATH", info)
        monkeypatch.setattr(module, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.setenv("MODEL_GATEWAY_LAUNCHD_LABEL", lr.GATEWAY_LABEL)
    providers.reload()
    return config


@pytest.fixture
def mac(monkeypatch, tmp_path):
    """An eligible Mac whose launchd, uv, media-gen and network are recorded fakes."""
    loaded: set[str] = set()
    calls: list[tuple] = []
    served: dict[str, bool] = {}

    def launchctl(*args):
        calls.append(("launchctl", *args))
        label = args[1].rsplit("/", 1)[-1] if len(args) > 1 else ""
        if args[0] == "print":
            output = f"path = {lr.launch_agents_dir() / f'{label}.plist'}\n"
            return subprocess.CompletedProcess(args, 0 if label in loaded else 113, output, "")
        if args[0] == "bootstrap":
            loaded.add(plistlib.loads(Path(args[2]).read_bytes())["Label"])
        if args[0] == "bootout":
            loaded.discard(label)
        return subprocess.CompletedProcess(args, 0, "", "")

    def run(*args, env=None):
        calls.append(("run", *args))
        if len(args) > 2 and args[1] == "pull":
            served[args[2]] = True
        if len(args) > 2 and args[1] == "remove":
            served.pop(args[2], None)

    def request(url, token="", body=None, *, timeout=15):
        calls.append(("request", url.split("/", 3)[-1], bool(token)))
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/v1/models"):
            return {"data": [{"id": name, "installed": True} for name in served]}
        if url.endswith("/v1/images/generations"):
            return {"data": [{"b64_json": "aGk="}]}
        raise AssertionError(url)

    monkeypatch.setattr(lr, "_launchctl", launchctl)
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)
    monkeypatch.setattr(lr, "_memory_bytes", lambda: 128 * GIB)
    monkeypatch.setattr(lr, "_disk_free", lambda path: 500 * GIB)
    monkeypatch.setattr(lr, "_request", request)
    monkeypatch.setattr(lr, "_uv", lambda: "uv")
    monkeypatch.setattr(mr, "_run", run)
    monkeypatch.setattr(lr.time, "sleep", lambda seconds: None)
    project = tmp_path / "media-runtime"
    project.mkdir()
    (project / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(mr, "RUNTIME_PROJECT", project)

    def built_venv(*args, env=None):
        run(*args, env=env)
        if args[1:2] == ("sync",):
            (mr.venv_dir() / "bin").mkdir(parents=True, exist_ok=True)
            (mr.venv_dir() / "bin" / "media-gen").write_text("")

    monkeypatch.setattr(mr, "_run", built_venv)
    run.calls, run.loaded, run.served = calls, loaded, served
    return run


def _config() -> dict:
    return yaml.safe_load(config_io.CONFIG_PATH.read_text())


def test_add_builds_downloads_starts_and_wires_the_gateway(gateway, mac):
    status = mr.add("flux3-image-klein")

    commands = [c[1:4] for c in mac.calls if c[0] == "run"]
    assert commands[0] == ("uv", "sync", "--locked")
    # A new build first deletes the files of models its release retired.
    assert commands[2] == (str(mr.venv_dir() / "bin" / "media-gen"), "prune")
    assert commands[3] == (str(mr.venv_dir() / "bin" / "media-gen"), "pull", "flux3-image-klein")
    plist = plistlib.loads(mr.plist_path().read_bytes())
    assert plist[lr.OWNER_KEY] == lr.owner_value()
    assert plist["ProgramArguments"] == [str(mr.venv_dir() / "bin" / "media-gen"), "serve"]
    assert plist["EnvironmentVariables"]["MEDIA_GEN_HOST"] == "127.0.0.1"
    assert "MEDIA_GEN_OMLX_URL" not in plist["EnvironmentVariables"]  # no gateway-managed oMLX here
    assert mr.api_key_path().stat().st_mode & 0o777 == 0o600
    # The check generated a real image through media-gen, with its key.
    assert ("request", "v1/images/generations", True) in mac.calls

    provider = _config()["providers"]["media"]
    assert provider["managed_by"] == config_io.MEDIA_MANAGED
    assert provider["base_url"] == "http://127.0.0.1:9112/v1"
    assert lr.generation()  # the running gateway reloads

    providers.reload()
    info = providers.resolve("flux3-image-klein")
    assert info.provider == "media" and info.modality == "image"
    assert providers.pricing_status_for("flux3-image-klein") == "unmetered"
    assert [m["name"] for m in status["models"] if m["installed"]] == ["flux3-image-klein"]
    assert status["managed"] and status["running"]


def test_a_second_model_joins_and_video_skips_the_generation_check(gateway, mac):
    mr.add("flux3-image-klein")
    mac.calls.clear()
    mr.add("ltx-2.5")
    assert ("request", "v1/images/generations", True) not in mac.calls
    # Same build, same plist, service healthy: no rebuild, no restart of running jobs.
    assert not [c for c in mac.calls if c[0] == "run" and c[2:3] == ("sync",)]
    assert not [c for c in mac.calls if c[:2] == ("launchctl", "bootstrap")]
    providers.reload()
    assert providers.resolve("ltx-2.5").modality == "video"
    assert mr.installed() == ["flux3-image-klein", "ltx-2.5"]


def test_rows_load_only_while_this_gateway_manages_media(gateway, mac):
    assert mr.catalog_rows(_config()) == []
    mr.add("qwen-image-2.1")
    rows = shipped.layer(_config()).entries
    [row] = [r for r in rows if r["name"] == "qwen-image-2.1"]
    assert row["origin"] == {"layer": "shipped", "profile": "media"}
    assert row["modality"] == "image"
    names = {r["name"] for r in shipped.all_shipped_rows({})}
    assert set(mr.MODELS) <= names


def test_refuses_a_foreign_service_or_provider(gateway, mac):
    mr.plist_path().parent.mkdir(parents=True, exist_ok=True)
    mr.plist_path().write_bytes(plistlib.dumps({"Label": mr.MEDIA_LABEL}))
    with pytest.raises(lr.LocalRuntimeError, match="managed outside Model Gateway"):
        mr.add("flux3-image-klein")
    mr.plist_path().unlink()
    config = _config()
    config["providers"]["media"] = {"base_url": "http://elsewhere/v1", "api_key": "x"}
    config_io.CONFIG_PATH.write_text(yaml.safe_dump(config))
    with pytest.raises(lr.LocalRuntimeError, match="does not manage"):
        mr.add("flux3-image-klein")
    assert [c for c in mac.calls if c[0] == "run"] == []


def test_an_external_omlx_can_be_named(gateway, mac, monkeypatch, tmp_path):
    key = tmp_path / "omlx.key"
    key.write_text("k")
    monkeypatch.setenv("MODEL_GATEWAY_MEDIA_OMLX_URL", "http://127.0.0.1:9110")
    monkeypatch.setenv("MODEL_GATEWAY_MEDIA_OMLX_API_KEY_FILE", str(key))
    environment = mr.media_plist()["EnvironmentVariables"]
    assert environment["MEDIA_GEN_OMLX_URL"] == "http://127.0.0.1:9110"
    assert environment["MEDIA_GEN_OMLX_API_KEY_FILE"] == str(key.resolve())
    assert "k" not in environment.values()


def test_a_failed_gateway_wiring_unlists_the_model(gateway, mac, monkeypatch):
    def broken_config(config):
        raise OSError("disk full")

    monkeypatch.setattr(lr, "_write_config", broken_config)
    with pytest.raises(OSError):
        mr.add("flux3-image-klein")
    assert mr.installed() == []
    assert "media" not in _config()["providers"]


def test_refuses_a_mac_that_is_too_small(gateway, mac, monkeypatch):
    monkeypatch.setattr(lr, "_memory_bytes", lambda: 48 * GIB)
    needed = mr.MODELS["ltx-2.5"].memory_gib + mr.SYSTEM_MEMORY // GIB
    with pytest.raises(lr.LocalRuntimeError, match=f"at least {needed} GB"):
        mr.add("ltx-2.5")
    monkeypatch.setattr(lr, "_memory_bytes", lambda: 128 * GIB)
    monkeypatch.setattr(lr, "_disk_free", lambda path: 20 * GIB)
    with pytest.raises(lr.LocalRuntimeError, match="free disk space"):
        mr.add("ltx-2.5")


def test_failed_check_leaves_the_gateway_unchanged(gateway, mac, monkeypatch):
    def broken(url, token="", body=None, *, timeout=15):
        if url.endswith("/health"):
            return {"status": "ok"}
        return {"data": []}

    monkeypatch.setattr(lr, "_request", broken)
    with pytest.raises(lr.LocalRuntimeError, match="does not report"):
        mr.add("flux3-image-klein")
    assert "media" not in _config()["providers"]
    assert mr.installed() == []
    # The first install's service is stopped again; the build is kept for a retry.
    assert mr.MEDIA_LABEL not in mac.loaded
    assert not mr.plist_path().exists()
    assert mr._built()


def test_a_new_release_rebuilds_and_restarts(gateway, mac):
    mr.add("flux3-image-klein")
    (mr.RUNTIME_PROJECT / "uv.lock").write_text("version = 2\n")
    mac.calls.clear()
    mr.add("qwen-image-2.1")
    assert [c for c in mac.calls if c[0] == "run" and c[2:3] == ("sync",)]
    assert [c for c in mac.calls if c[:2] == ("launchctl", "bootstrap")]


def test_remove_one_model_then_everything(gateway, mac):
    mr.add("flux3-image-klein")
    mr.add("qwen-image-2.1")
    mr.remove("flux3-image-klein")
    assert mr.installed() == ["qwen-image-2.1"]
    assert ("run", str(mr.venv_dir() / "bin" / "media-gen"), "remove", "flux3-image-klein") in mac.calls
    assert "media" in _config()["providers"]

    mr.remove()
    assert not mr.root().exists()
    assert not mr.plist_path().exists()
    assert mr.MEDIA_LABEL not in mac.loaded
    assert "media" not in _config()["providers"]
    with pytest.raises(lr.LocalRuntimeError, match="not installed"):
        mr.remove()


def test_admin_cannot_edit_the_managed_provider(gateway, mac):
    mr.add("flux3-image-klein")
    with pytest.raises(ValueError, match="model-gateway media"):
        config_io._refuse_local_ai_provider(_config()["providers"], "media")


def test_cli(gateway, mac, capsys, monkeypatch):
    # The owner accepts the model's license first: asked at a terminal, else --accept-license.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert mr.main(["add", "flux3-image-klein"]) == 1
    out, err = capsys.readouterr()
    assert "FLUX Kommunity License" in out and "hf auth login" in out and "--accept-license" in err
    assert not mr.installed()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert mr.main(["add", "qwen-image-2.1"]) == 1
    assert "Not installed" in capsys.readouterr().out
    assert mr.main(["add", "flux3-image-klein", "--accept-license"]) == 0
    assert "flux3-image-klein is ready" in capsys.readouterr().out
    assert mr.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["managed"] is True
    assert mr.main(["remove", "qwen-image-2.1"]) == 2
    assert "not installed" in capsys.readouterr().err


def test_the_locked_runtime_pins_media_gen_by_commit():
    import tomllib

    project = tomllib.loads((mr.RUNTIME_PROJECT / "pyproject.toml").read_text())
    assert project["project"]["dependencies"] == ["media-gen[engines]"]
    source = project["tool"]["uv"]["sources"]["media-gen"]
    assert source["git"] == "https://github.com/GeorgeTheo99/media-gen" and len(source["rev"]) == 40
    lock = (mr.RUNTIME_PROJECT / "uv.lock").read_text()
    assert f'source = {{ git = "{source["git"]}?rev={source["rev"]}#{source["rev"]}" }}' in lock
    for engine in ('name = "mflux"\nversion = "0.24.0"', 'name = "ltx-pipelines-mlx"\nversion = "0.18.0"',
                   'name = "media-gen"\nversion = "0.4.0"', 'name = "mlx-video"\nversion = "0.0.1"'):
        assert engine in lock
    wan_source = project["tool"]["uv"]["sources"]["mlx-video"]
    assert wan_source["git"] == "https://github.com/Blaizzy/mlx-video"
    assert len(wan_source["rev"]) == 40
    assert f'{wan_source["git"]}?rev={wan_source["rev"]}#{wan_source["rev"]}' in lock
    assert set(mr.MODELS) == {"flux3-image-klein", "qwen-image-2.1", "ltx-2.5", "wan2.2-i2v-a14b"}
    wan = mr.MODELS["wan2.2-i2v-a14b"]
    assert wan.modality == "video" and wan.license == "Apache-2.0"
    assert wan.download_bytes == 42_702_440_820


def test_a_new_release_forgets_the_models_it_retired(gateway, mac):
    mr.add("flux3-image-klein")
    mr.installed_path().write_text(json.dumps(["flux3-image-klein", "z-image-turbo"]))
    mr.build_path().write_text("an older release\n")
    mr.add("ltx-2.5")
    assert json.loads(mr.installed_path().read_text()) == ["flux3-image-klein", "ltx-2.5"]



@pytest.fixture
def profiles(mac, monkeypatch):
    """media-gen's `custom` command, faked: its profiles, keyed by id."""
    state: dict[str, dict] = {}
    run = mr._run

    def custom_run(*args, env=None):
        if args[1:2] != ("custom",):
            return run(*args, env=env)
        mac.calls.append(("run", *args))
        action, rest = args[2], args[3:]
        if action == "import":
            options = dict(zip(rest[1::2], rest[2::2], strict=False))
            state[f"custom-{options['--name']}"] = {
                "id": f"custom-{options['--name']}", "base": options["--base"],
                "display_name": options.get("--display-name", options["--name"]), "enabled": False,
                "strength": float(options["--strength"]), "license": options["--license"],
                "source": options["--source"]}
        elif action in ("enable", "disable"):
            state[rest[0]]["enabled"] = action == "enable"
            if action == "enable":
                mac.served[rest[0]] = True
            else:
                mac.served.pop(rest[0], None)
        elif action == "remove":
            state.pop(rest[0])
            mac.served.pop(rest[0], None)

    def output(*args, env=None):
        assert args[1:] == ("custom", "list", "--json")
        assert env["MEDIA_GEN_HOME"] == str(mr.home_dir())
        return json.dumps(list(state.values()))

    monkeypatch.setattr(mr, "_run", custom_run)
    monkeypatch.setattr(mr, "_output", output)
    return state


def _custom_rows() -> list[dict]:
    return [row for row in shipped.layer(_config()).entries if row["name"].startswith("custom-")]


def test_custom_profiles_route_only_once_enabled(gateway, mac, profiles):
    mr.add("ltx-2.5")
    mr.custom("import", "/tmp/style.safetensors", "--base", "ltx-2.5", "--name", "style", "--license", "MIT",
              "--source", "my training", "--strength", "0.8", "--display-name", "Style")
    assert profiles["custom-style"]["enabled"] is False
    assert _custom_rows() == []
    assert mr.custom_profiles() == []

    mr.custom("enable", "custom-style")
    [row] = _custom_rows()
    assert row["custom_base"] == "ltx-2.5" and row["modality"] == "video"
    assert row["provider_model_id"] == "custom-style" and row["display_name"] == "Style"
    assert row["origin"] == {"layer": "shipped", "profile": "media"}
    providers.reload()
    assert providers.resolve("custom-style").provider == "media"
    # Never part of what a release ships.
    assert "custom-style" not in {r["name"] for r in shipped.all_shipped_rows({})}
    assert mr.status()["custom"] == [{"name": "custom-style", "base": "ltx-2.5", "display_name": "Style"}]
    # The vanilla model is unchanged beside it.
    assert providers.resolve("ltx-2.5").provider_model_id == "ltx-2.5"

    mr.custom("disable", "custom-style")
    assert _custom_rows() == []
    mr.custom("enable", "custom-style")
    mr.custom("remove", "custom-style")
    assert _custom_rows() == [] and profiles == {}


def test_a_custom_profile_follows_its_base_model(gateway, mac, profiles):
    mr.add("ltx-2.5")
    mr.add("flux3-image-klein")
    mr.custom("import", "/tmp/a.safetensors", "--base", "ltx-2.5", "--name", "a", "--license", "MIT",
              "--source", "x", "--strength", "1.0")
    mr.custom("enable", "custom-a")
    mr.remove("ltx-2.5")
    assert _custom_rows() == []
    with pytest.raises(lr.LocalRuntimeError, match="not installed"):
        mr.custom("enable", "custom-a")


def test_custom_enable_is_checked_and_rolled_back(gateway, mac, profiles, monkeypatch):
    mr.add("ltx-2.5")
    with pytest.raises(lr.LocalRuntimeError, match="No custom profile"):
        mr.custom("enable", "custom-missing")
    mr.custom("import", "/tmp/a.safetensors", "--base", "ltx-2.5", "--name", "a", "--license", "MIT",
              "--source", "x", "--strength", "1.0")
    request = lr._request

    def unserved(url, token="", body=None, *, timeout=15):
        if url.endswith("/v1/models"):
            return {"data": [{"id": "ltx-2.5", "installed": True}]}
        return request(url, token, body, timeout=timeout)

    monkeypatch.setattr(lr, "_request", unserved)
    with pytest.raises(lr.LocalRuntimeError, match="left disabled"):
        mr.custom("enable", "custom-a")
    assert profiles["custom-a"]["enabled"] is False
    assert _custom_rows() == []


def test_custom_needs_a_current_media_gen(gateway, mac, profiles):
    with pytest.raises(lr.LocalRuntimeError, match="not installed"):
        mr.custom("list")
    mr.add("ltx-2.5")
    (mr.RUNTIME_PROJECT / "uv.lock").write_text("version = 2\n")
    with pytest.raises(lr.LocalRuntimeError, match="media add ltx-2.5"):
        mr.custom("list")


def test_custom_cli(gateway, mac, profiles, capsys):
    mr.add("ltx-2.5")
    capsys.readouterr()
    assert mr.main(["custom", "import", "~/style.safetensors", "--base", "ltx-2.5", "--name", "style",
                    "--license", "MIT", "--source", "my training"]) == 0
    out = capsys.readouterr().out
    assert "never publishes or syncs it" in out and "custom enable custom-style" in out
    [call] = [c for c in mac.calls if c[2:4] == ("custom", "import")]
    assert call[4] == str(Path("~/style.safetensors").expanduser().resolve())
    assert mr.main(["custom", "enable", "custom-style"]) == 0
    assert mr.main(["custom", "list"]) == 0
    assert "custom-style" in capsys.readouterr().out
    assert mr.main(["custom", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["enabled"] is True


def test_custom_enable_rolls_back_when_media_gen_is_unreachable(gateway, mac, profiles, monkeypatch):
    mr.add("ltx-2.5")
    mr.custom("import", "/tmp/a.safetensors", "--base", "ltx-2.5", "--name", "a", "--license", "MIT",
              "--source", "x", "--strength", "1.0")
    generation = (lr.runtime_root() / lr.GENERATION_FILE).read_text()

    def down(url, token="", body=None, *, timeout=15):
        raise OSError("connection refused")

    monkeypatch.setattr(lr, "_request", down)
    with pytest.raises(OSError):
        mr.custom("enable", "custom-a")
    assert profiles["custom-a"]["enabled"] is False
    assert _custom_rows() == [] and mr.custom_profiles() == []
    assert (lr.runtime_root() / lr.GENERATION_FILE).read_text() != generation


def test_the_last_model_is_kept_while_custom_profiles_exist(gateway, mac, profiles):
    mr.add("ltx-2.5")
    adapters = mr.home_dir() / "custom" / "adapters"
    adapters.mkdir(parents=True)
    (adapters / f"{'0' * 64}.safetensors").write_bytes(b"x")
    with pytest.raises(lr.LocalRuntimeError, match="custom profiles"):
        mr.remove("ltx-2.5")
    assert mr.installed() == ["ltx-2.5"] and adapters.exists()
    mr.remove()
    assert not mr.root().exists()
