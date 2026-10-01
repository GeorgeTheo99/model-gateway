"""Portable operator CLI configuration tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "model-gateway"


def _run_env(home: Path, overrides: dict[str, str] | None = None) -> str:
    env = dict(os.environ)
    for key in (
        "MODEL_GATEWAY_HOST",
        "MODEL_GATEWAY_PORT",
        "MODEL_GATEWAY_ADMIN_WRITES",
        "MODEL_GATEWAY_PLIST_DIR",
        "MODEL_GATEWAY_LEDGER_PATH",
        "MODEL_GATEWAY_CONFIG",
        "MODEL_GATEWAY_MODEL_INFO",
        "MODEL_GATEWAY_MODEL_INFO_SOURCE",
        "GATEWAY_VISION_FALLBACK",
        "GATEWAY_VISION_FALLBACK_LOCAL",
        "GATEWAY_VISION_FALLBACK_CLOUD",
        "GATEWAY_VISION_FALLBACK_MODE",
        "GATEWAY_VISION_FALLBACK_MAX_IMAGES",
        "GATEWAY_VISION_OBSERVATION_CACHE_TTL_SECONDS",
        "GATEWAY_VISION_EXTRACTION_TOTAL_TIMEOUT_SECONDS",
    ):
        env.pop(key, None)
    env["HOME"] = str(home)
    env.update(overrides or {})
    completed = subprocess.run(
        [str(SCRIPT), "env"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout


def test_env_uses_legacy_bind_defaults_without_an_install_config(tmp_path: Path) -> None:
    output = _run_env(tmp_path)
    assert "MODEL_GATEWAY_HOST=127.0.0.1" in output
    assert "MODEL_GATEWAY_PORT=9111" in output


def test_env_recovers_persisted_bind_assignment_in_a_fresh_shell(tmp_path: Path) -> None:
    install_config = (
        tmp_path / "Library" / "Application Support" / "model-gateway" / "install.env"
    )
    install_config.parent.mkdir(parents=True)
    install_config.write_text(
        "MODEL_GATEWAY_HOST=127.0.0.2\nMODEL_GATEWAY_PORT=19111\n",
        encoding="utf-8",
    )
    install_config.chmod(0o600)

    output = _run_env(tmp_path)
    assert "MODEL_GATEWAY_HOST=127.0.0.2" in output
    assert "MODEL_GATEWAY_PORT=19111" in output
    assert f"MODEL_GATEWAY_INSTALL_CONFIG={install_config}" in output


def test_fresh_shell_ignores_nonprivate_install_config(tmp_path: Path) -> None:
    install_config = (
        tmp_path / "Library" / "Application Support" / "model-gateway" / "install.env"
    )
    install_config.parent.mkdir(parents=True)
    install_config.write_text(
        "MODEL_GATEWAY_HOST=127.0.0.2\nMODEL_GATEWAY_PORT=19111\n",
        encoding="utf-8",
    )
    install_config.chmod(0o644)

    output = _run_env(tmp_path)
    assert "MODEL_GATEWAY_HOST=127.0.0.1" in output
    assert "MODEL_GATEWAY_PORT=9111" in output


def test_explicit_environment_overrides_the_persisted_assignment(tmp_path: Path) -> None:
    install_config = (
        tmp_path / "Library" / "Application Support" / "model-gateway" / "install.env"
    )
    install_config.parent.mkdir(parents=True)
    install_config.write_text(
        "MODEL_GATEWAY_HOST=127.0.0.2\nMODEL_GATEWAY_PORT=19111\n",
        encoding="utf-8",
    )

    output = _run_env(
        tmp_path,
        {"MODEL_GATEWAY_HOST": "127.0.0.3", "MODEL_GATEWAY_PORT": "29111"},
    )
    assert "MODEL_GATEWAY_HOST=127.0.0.3" in output
    assert "MODEL_GATEWAY_PORT=29111" in output


def test_fresh_install_enables_admin_writes_and_upgrades_keep_read_only(tmp_path: Path) -> None:
    assert "MODEL_GATEWAY_ADMIN_WRITES=true\n" in _run_env(tmp_path)
    install_config = (
        tmp_path / "Library" / "Application Support" / "model-gateway" / "install.env"
    )
    install_config.parent.mkdir(parents=True)
    install_config.write_text("MODEL_GATEWAY_HOST=127.0.0.1\nMODEL_GATEWAY_PORT=9111\n", encoding="utf-8")
    install_config.chmod(0o600)
    # An install that predates the setting was read-only; an update keeps it so.
    assert "MODEL_GATEWAY_ADMIN_WRITES=false\n" in _run_env(tmp_path)
    install_config.write_text(
        "MODEL_GATEWAY_HOST=127.0.0.1\nMODEL_GATEWAY_PORT=9111\nMODEL_GATEWAY_ADMIN_WRITES=true\n",
        encoding="utf-8",
    )
    assert "MODEL_GATEWAY_ADMIN_WRITES=true\n" in _run_env(tmp_path)
    assert "MODEL_GATEWAY_ADMIN_WRITES=false\n" in _run_env(
        tmp_path, {"MODEL_GATEWAY_ADMIN_WRITES": "false"}
    )

    script = SCRIPT.read_text(encoding="utf-8")
    write_plist = script.split("write_plist() {", 1)[1].split("service_target() {", 1)[0]
    persist = script.split("persist_install_config() {", 1)[1].split("check_macos() {", 1)[0]
    validate = script.split("validate_bind_config() {", 1)[1].split("persist_install_config() {", 1)[0]
    assert "<key>MODEL_GATEWAY_ADMIN_WRITES</key><string>${e_admin_writes}</string>" in write_plist
    assert "MODEL_GATEWAY_ADMIN_WRITES=%s" in persist
    assert "MODEL_GATEWAY_ADMIN_WRITES must be true or false" in validate


def test_env_exposes_scoped_vision_fallback_configuration(tmp_path: Path) -> None:
    output = _run_env(
        tmp_path,
        {
            "GATEWAY_VISION_FALLBACK_LOCAL": "local-vision",
            "GATEWAY_VISION_FALLBACK_CLOUD": "cloud-vision",
            "GATEWAY_VISION_FALLBACK_MODE": "extract_then_answer",
            "GATEWAY_VISION_FALLBACK_MAX_IMAGES": "12",
            "GATEWAY_VISION_OBSERVATION_CACHE_TTL_SECONDS": "600",
            "GATEWAY_VISION_EXTRACTION_TOTAL_TIMEOUT_SECONDS": "1200",
        },
    )
    assert "GATEWAY_VISION_FALLBACK_LOCAL=local-vision" in output
    assert "GATEWAY_VISION_FALLBACK_CLOUD=cloud-vision" in output
    assert "GATEWAY_VISION_FALLBACK_MODE=extract_then_answer" in output
    assert "GATEWAY_VISION_FALLBACK_MAX_IMAGES=12" in output
    assert "GATEWAY_VISION_OBSERVATION_CACHE_TTL_SECONDS=600" in output
    assert "GATEWAY_VISION_EXTRACTION_TOTAL_TIMEOUT_SECONDS=1200" in output


def test_explicit_empty_vision_setting_clears_persisted_plist_value(tmp_path: Path) -> None:
    plist_dir = tmp_path / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True)
    (plist_dir / "com.local.model-gateway.plist").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.local.model-gateway</string>
<key>EnvironmentVariables</key><dict>
  <key>GATEWAY_VISION_FALLBACK</key><string>legacy-vision</string>
  <key>GATEWAY_VISION_FALLBACK_MODE</key><string>reroute</string>
</dict>
</dict></plist>
""",
        encoding="utf-8",
    )

    output = _run_env(
        tmp_path,
        {
            "GATEWAY_VISION_FALLBACK": "",
            "GATEWAY_VISION_FALLBACK_LOCAL": "local-vision",
            "GATEWAY_VISION_FALLBACK_MODE": "",
        },
    )

    assert "GATEWAY_VISION_FALLBACK=\n" in output
    assert "GATEWAY_VISION_FALLBACK_LOCAL=local-vision" in output
    assert "GATEWAY_VISION_FALLBACK_MODE=\n" in output
    assert "legacy-vision" not in output


def test_install_and_post_pull_update_validate_before_atomic_plist_write() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    install = script.split("cmd_install() {", 1)[1].split("cmd_uninstall() {", 1)[0]
    update_after_pull = script.split("cmd_update_after_pull() {", 1)[1].split("cmd_update() {", 1)[0]
    update = script.split("cmd_update() {", 1)[1].split("cmd_onboard() {", 1)[0]
    write_plist = script.split("write_plist() {", 1)[1].split("service_target() {", 1)[0]
    persist = script.split("persist_install_config() {", 1)[1].split("check_macos() {", 1)[0]
    assert install.index("validate_bind_config") < install.index("write_plist")
    assert update_after_pull.index("validate_bind_config") < update_after_pull.index("write_plist")
    assert write_plist.index("plutil -lint") < write_plist.index("persist_install_config")
    assert write_plist.index("persist_install_config") < write_plist.index('mv -f "$plist_tmp" "$plist"')
    assert "ensure_private_file" not in persist
    assert persist.index("mktemp") < persist.index('mv -f "$tmp" "$INSTALL_CONFIG"')
    assert "GATEWAY_VISION_FALLBACK_LOCAL" in write_plist
    assert "GATEWAY_VISION_FALLBACK_CLOUD" in write_plist
    assert "GATEWAY_VISION_FALLBACK_MODE" in write_plist
    assert "GATEWAY_VISION_FALLBACK_MAX_IMAGES" in write_plist
    assert "GATEWAY_VISION_OBSERVATION_CACHE_TTL_SECONDS" in write_plist
    assert "GATEWAY_VISION_EXTRACTION_TOTAL_TIMEOUT_SECONDS" in write_plist
    assert 'exec "$ROOT_DIR/bin/model-gateway" _update-after-pull' in update


def test_fresh_install_config_enables_the_alias_export() -> None:
    import yaml

    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index('cat >"$CONFIG_PATH" <<EOF\n') + len('cat >"$CONFIG_PATH" <<EOF\n')
    template = text[start:text.index("\nEOF\n", start)]
    # The generated admin key is the only shell expansion in the template.
    assert template.count("$") == 1
    config = yaml.safe_load(template.replace("${admin_key}", "mg-admin-test"))
    assert config["exports"] == {
        "model_aliases": "~/Library/Application Support/model-gateway/model-aliases.json"
    }
    assert config["auth"] == {"admin_keys": ["mg-admin-test"], "client_keys": []}


def _packaged_cli(tmp_path: Path) -> Path:
    """Lay out a Homebrew-style libexec with a version-independent root."""
    root = tmp_path / "libexec"
    (root / "bin").mkdir(parents=True)
    script = root / "bin" / "model-gateway"
    script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)
    (root / ".package").write_text(
        f"MANAGER=homebrew\nROOT={root}\nPYTHON={tmp_path}/venv/bin/python\nUPGRADE=brew upgrade model-gateway\n",
        encoding="utf-8",
    )
    return script


def test_packaged_install_keeps_state_in_application_support(tmp_path: Path) -> None:
    script = _packaged_cli(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_GATEWAY_", "GATEWAY_VISION"))}
    env["HOME"] = str(home)
    output = subprocess.run([str(script), "env"], check=True, capture_output=True, text=True, env=env).stdout
    app = home / "Library" / "Application Support" / "model-gateway"
    assert "PACKAGE_MANAGER=homebrew\n" in output
    assert f"ROOT_DIR={tmp_path / 'libexec'}\n" in output
    assert f"MODEL_GATEWAY_CONFIG={app / 'config.yaml'}\n" in output
    assert f"MODEL_GATEWAY_MODEL_INFO={app / 'model-info.json'}\n" in output
    assert f"MODEL_GATEWAY_MODEL_INFO_SOURCE={app / 'model-info.json'}\n" in output
    assert f"MODEL_GATEWAY_LEDGER_PATH={app / 'ledger.db'}\n" in output

    update = subprocess.run([str(script), "update"], capture_output=True, text=True, env=env)
    assert update.returncode == 1
    assert "update with: brew upgrade model-gateway && model-gateway restart" in update.stderr


def test_existing_legacy_ledger_stays_in_use(tmp_path: Path) -> None:
    legacy = tmp_path / "srv" / "model-gateway" / "shared" / "ledger.db"
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    assert f"MODEL_GATEWAY_LEDGER_PATH={legacy}\n" in _run_env(tmp_path)
    current = tmp_path / "Library" / "Application Support" / "model-gateway" / "ledger.db"
    current.parent.mkdir(parents=True)
    current.touch()
    assert f"MODEL_GATEWAY_LEDGER_PATH={current}\n" in _run_env(tmp_path)


def _cli(home: Path, *args: str, config: Path | None = None) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_GATEWAY_", "GATEWAY_VISION"))}
    env["HOME"] = str(home)
    # launchctl is not HOME-scoped: never let a test address the real service.
    env["MODEL_GATEWAY_LAUNCHD_LABEL"] = "com.local.model-gateway-test-does-not-exist"
    if config is not None:
        env["MODEL_GATEWAY_CONFIG"] = str(config)
        env["MODEL_GATEWAY_MODEL_INFO"] = str(config.parent / "model-info.json")
    return subprocess.run([str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=120)


def test_consumer_and_bundle_reject_unknown_subcommands(tmp_path: Path) -> None:
    for command in ("consumer", "bundle"):
        result = _cli(tmp_path, command, "bogus")
        assert result.returncode == 1 and "usage: model-gateway " + command in result.stderr


def test_read_only_consumer_and_bundle_commands_need_no_service(tmp_path: Path) -> None:
    real = Path(os.path.realpath(tmp_path))
    config = real / "config.yaml"
    config.write_text("auth:\n  client_keys: [client-token]\nproviders: {}\n")
    config.chmod(0o600)
    listed = _cli(real, "consumer", "list", config=config)
    assert listed.returncode == 0, listed.stderr
    assert "no consumer credentials configured" in listed.stdout
    # A mutating command refuses without the installed, loaded LaunchAgent.
    added = _cli(real, "consumer", "add", "myai", "--role", "runtime", config=config)
    assert added.returncode != 0 and "not loaded" in added.stderr
    assert "consumer_credentials" not in config.read_text()


def test_bundle_export_and_dry_run_import_through_the_cli(tmp_path: Path) -> None:
    import json

    real = Path(os.path.realpath(tmp_path))
    config = real / "config.yaml"
    config.write_text("auth:\n  client_keys: [client-token]\nproviders: {}\n")
    config.chmod(0o600)
    (real / "model-info.json").write_text(json.dumps({"llm": []}))
    out = real / "bundle.tar.gz"
    exported = _cli(real, "bundle", "export", "--out", str(out), config=config)
    assert exported.returncode == 0, exported.stderr
    dry = _cli(real, "bundle", "import", str(out), "--dry-run", config=config)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["dry_run"] is True


def _bundle_import_harness(tmp_path: Path, **failures: str) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """Run ``cmd_bundle import`` with launchd steps stubbed out."""
    import json

    real = Path(os.path.realpath(tmp_path))
    source, target = real / "source", real / "target"
    catalog = {"llm": [{"name": "claude-test", "provider": "anthropic", "provider_model_id": "claude-test-1",
                        "context": 200000, "max_output_tokens": 8192,
                        "pricing": {"input": 3.0, "output": 15.0}}]}
    for machine, providers in ((source, "{anthropic: {base_url: 'https://api.anthropic.com/v1'}}"), (target, "{}")):
        machine.mkdir()
        (machine / "config.yaml").write_text(f"auth:\n  client_keys: [client-token]\nproviders: {providers}\n")
        (machine / "config.yaml").chmod(0o600)
    (source / "model-info.json").write_text(json.dumps(catalog))
    bundle_path = real / "bundle.tar.gz"
    assert _cli(real, "bundle", "export", "--out", str(bundle_path), config=source / "config.yaml").returncode == 0
    harness = real / "harness.sh"
    harness.write_text(
        'cli="$1" bundle="$2"\nset -- env\nsource "$cli" >/dev/null\n'
        "ensure_runtime_paths_safe() { :; }\n"
        'eval "orig_$(declare -f run_service_python)"\n'
        'run_service_python() {\n'
        '  [ "$2:${FAIL_RESTORE:-}" != restore-rollback:1 ] || return 1\n'
        '  orig_run_service_python "$@" || return $?\n'
        '  [ "$2:${SIGNAL_AFTER_IMPORT:-}" != import:1 ] || kill -TERM $$\n'
        '}\n'
        'restart_service() { echo RESTARTED >&2; [ -z "${FAIL_RESTART:-}" ] || die "bootstrap failed"; }\n'
        'verify() { [ -z "${FAIL_VERIFY:-}" ]; }\n'
        'cmd_bundle import "$bundle" --force\n'
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_GATEWAY_", "GATEWAY_VISION"))}
    env.update({
        "HOME": str(real),
        "MODEL_GATEWAY_LAUNCHD_LABEL": "com.local.model-gateway-test-does-not-exist",
        "MODEL_GATEWAY_CONFIG": str(target / "config.yaml"),
        "MODEL_GATEWAY_MODEL_INFO": str(target / "model-info.json"),
        "MODEL_GATEWAY_BACKUP_DIR": str(real / "backups"),
        **failures,
    })
    result = subprocess.run(["bash", str(harness), str(SCRIPT), str(bundle_path)],
                            capture_output=True, text=True, env=env, timeout=120)
    return result, target, real / "backups"


def test_bundle_import_success_leaves_no_rollback_file(tmp_path: Path) -> None:
    result, target, backups = _bundle_import_harness(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "anthropic" in (target / "config.yaml").read_text()
    assert not list(backups.glob("bundle-rollback.*"))


def test_bundle_import_restores_previous_files_when_restart_or_verify_fails(tmp_path: Path) -> None:
    for failure in ("FAIL_RESTART", "FAIL_VERIFY"):
        case = tmp_path / failure
        case.mkdir()
        result, target, backups = _bundle_import_harness(case, **{failure: "1"})
        assert result.returncode != 0
        assert "restoring the previous files" in result.stderr
        assert (target / "config.yaml").read_text() == "auth:\n  client_keys: [client-token]\nproviders: {}\n"
        assert not (target / "model-info.json").exists()
        assert not list(backups.glob("bundle-rollback.*"))


def test_bundle_import_validation_failure_restores_without_restarting(tmp_path: Path) -> None:
    result, target, backups = _bundle_import_harness(
        tmp_path, GATEWAY_VISION_FALLBACK_CLOUD="model-not-in-bundle",
        GATEWAY_VISION_FALLBACK_MODE="extract_then_answer")
    assert result.returncode != 0
    assert "RESTARTED" not in result.stderr
    assert (target / "config.yaml").read_text() == "auth:\n  client_keys: [client-token]\nproviders: {}\n"
    assert not (target / "model-info.json").exists()
    assert not list(backups.glob("bundle-rollback.*"))


def test_bundle_import_keeps_rollback_file_when_restore_fails(tmp_path: Path) -> None:
    result, _target, backups = _bundle_import_harness(tmp_path, FAIL_VERIFY="1", FAIL_RESTORE="1")
    assert result.returncode != 0
    saved = list(backups.glob("bundle-rollback.*"))
    assert len(saved) == 1 and saved[0].stat().st_size > 0
    assert f"model-gateway bundle restore-rollback {saved[0]}" in result.stderr


def test_restore_rollback_recovers_while_the_service_is_unloaded(tmp_path: Path) -> None:
    import plistlib

    real = Path(os.path.realpath(tmp_path))
    config = real / "config.yaml"
    original = "auth:\n  client_keys: [client-token]\nproviders: {}\n"
    config.write_text("auth:\n  client_keys: [client-token]\nproviders: {imported: {}}\n")
    config.chmod(0o600)
    rollback = real / "rollback.json"
    rollback.write_text(json_dumps({str(config): original, str(real / "model-info.json"): None}))
    rollback.chmod(0o600)
    plist_dir = real / "LaunchAgents"
    plist_dir.mkdir()
    label = "com.local.model-gateway-test-does-not-exist"
    (plist_dir / f"{label}.plist").write_bytes(plistlib.dumps({
        "Label": label, "WorkingDirectory": str(ROOT),
        "EnvironmentVariables": {"MODEL_GATEWAY_CONFIG": str(config),
                                 "MODEL_GATEWAY_MODEL_INFO": str(real / "model-info.json"),
                                 "MODEL_GATEWAY_MODEL_INFO_SOURCE": str(real / "model-info.json")},
    }))
    harness = real / "harness.sh"
    harness.write_text(
        'cli="$1" file="$2"\nset -- env\nsource "$cli" >/dev/null\n'
        "restart_service() { echo RESTARTED >&2; }\nverify() { :; }\n"
        'cmd_bundle restore-rollback "$file"\n'
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_GATEWAY_", "GATEWAY_VISION"))}
    env.update({"HOME": str(real), "MODEL_GATEWAY_LAUNCHD_LABEL": label, "MODEL_GATEWAY_PLIST_DIR": str(plist_dir),
                "MODEL_GATEWAY_CONFIG": str(config), "MODEL_GATEWAY_MODEL_INFO": str(real / "model-info.json"),
                "MODEL_GATEWAY_MODEL_INFO_SOURCE": str(real / "model-info.json")})
    result = subprocess.run(["bash", str(harness), str(SCRIPT), str(rollback)],
                            capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "RESTARTED" in result.stderr
    assert config.read_text() == original
    assert not rollback.exists()


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value)


def test_bundle_import_restores_when_terminated_after_files_are_written(tmp_path: Path) -> None:
    result, target, backups = _bundle_import_harness(tmp_path, SIGNAL_AFTER_IMPORT="1")
    assert result.returncode != 0
    assert "restoring the previous files" in result.stderr
    assert "RESTARTED" not in result.stderr
    assert (target / "config.yaml").read_text() == "auth:\n  client_keys: [client-token]\nproviders: {}\n"
    assert not (target / "model-info.json").exists()
    assert not list(backups.glob("bundle-rollback.*"))


def test_onboard_passes_the_service_client_key_source_to_reload_verification(tmp_path: Path) -> None:
    import plistlib

    real = Path(os.path.realpath(tmp_path))
    key_file = real / "client.key"
    plist_dir = real / "LaunchAgents"
    plist_dir.mkdir()
    label = "com.local.model-gateway-test-does-not-exist"
    (plist_dir / f"{label}.plist").write_bytes(plistlib.dumps({
        "Label": label,
        "EnvironmentVariables": {"MODEL_GATEWAY_CLIENT_KEYS_FILE": str(key_file)},
    }))
    captured = real / "captured.env"
    fake_uv = real / "uv"
    fake_uv.write_text(
        '#!/bin/sh\nprintf "keys=%s\\nfile=%s\\n" "${MODEL_GATEWAY_CLIENT_KEYS-unset}" '
        f'"$MODEL_GATEWAY_CLIENT_KEYS_FILE" > "{captured}"\n'
    )
    fake_uv.chmod(0o700)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_GATEWAY_", "GATEWAY_VISION"))}
    env.update({"HOME": str(real), "UV_BIN": str(fake_uv), "MODEL_GATEWAY_LAUNCHD_LABEL": label,
                "MODEL_GATEWAY_PLIST_DIR": str(plist_dir), "MODEL_GATEWAY_CONFIG": str(real / "config.yaml"),
                "MODEL_GATEWAY_MODEL_INFO": str(real / "model-info.json")})
    result = subprocess.run([str(SCRIPT), "onboard", "example", "--dry-run"],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stderr
    assert captured.read_text() == f"keys=\nfile={key_file}\n"

    env["MODEL_GATEWAY_CLIENT_KEYS"] = "operator-token"
    result = subprocess.run([str(SCRIPT), "onboard", "example", "--dry-run"],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stderr
    assert captured.read_text() == f"keys=operator-token\nfile={key_file}\n"


def _write_plist(home: Path, label: str, working_dir: str, env: dict[str, str]) -> Path:
    plist = home / "Library" / "LaunchAgents" / f"{label}.plist"
    plist.parent.mkdir(parents=True, exist_ok=True)
    entries = "".join(f"<key>{key}</key><string>{value}</string>" for key, value in env.items())
    plist.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        f'<plist version="1.0"><dict><key>Label</key><string>{label}</string>'
        f"<key>WorkingDirectory</key><string>{working_dir}</string>"
        f"<key>EnvironmentVariables</key><dict>{entries}</dict></dict></plist>\n",
        encoding="utf-8",
    )
    return plist


def test_installed_launchagent_paths_win_over_application_support(tmp_path: Path) -> None:
    app = tmp_path / "Library" / "Application Support" / "model-gateway"
    app.mkdir(parents=True)
    for name in ("config.yaml", "model-info.json", "ledger.db"):
        (app / name).touch()
    pinned = tmp_path / "checkout"
    _write_plist(tmp_path, "com.local.model-gateway", str(pinned), {
        "MODEL_GATEWAY_CONFIG": f"{pinned}/config/config.yaml",
        "MODEL_GATEWAY_MODEL_INFO": f"{pinned}/model-info.json",
        "MODEL_GATEWAY_LEDGER_PATH": f"{pinned}/ledger.db",
    })
    output = _run_env(tmp_path)
    assert f"MODEL_GATEWAY_CONFIG={pinned}/config/config.yaml\n" in output
    assert f"MODEL_GATEWAY_MODEL_INFO={pinned}/model-info.json\n" in output
    assert f"MODEL_GATEWAY_LEDGER_PATH={pinned}/ledger.db\n" in output


def test_install_refuses_a_foreign_launchagent_before_creating_state(tmp_path: Path) -> None:
    home = Path(os.path.realpath(tmp_path))
    label = "com.local.model-gateway-test-does-not-exist"
    _write_plist(home, label, "/some/other/checkout", {})
    result = _cli(home, "install", "--no-start")
    assert result.returncode == 1
    assert "Refusing to install" in result.stderr
    assert not (home / "Library" / "Application Support" / "model-gateway" / "config.yaml").exists()
    assert not (home / "Library" / "Application Support" / "model-gateway" / "model-info.json").exists()


def test_packaged_launchagent_runs_the_package_python(tmp_path: Path) -> None:
    import plistlib

    script = _packaged_cli(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    harness = tmp_path / "harness.sh"
    harness.write_text('cli="$1"\nset -- env\nsource "$cli" >/dev/null\nwrite_plist >/dev/null\n', encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_GATEWAY_", "GATEWAY_VISION"))}
    env.update({"HOME": str(home), "MODEL_GATEWAY_LAUNCHD_LABEL": "com.local.model-gateway-test-does-not-exist"})
    result = subprocess.run(["bash", str(harness), str(script)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    plist = plistlib.loads(
        (home / "Library" / "LaunchAgents" / "com.local.model-gateway-test-does-not-exist.plist").read_bytes()
    )
    assert plist["ProgramArguments"] == [f"{tmp_path}/venv/bin/python", "-m", "src.main"]
    assert plist["WorkingDirectory"] == str(tmp_path / "libexec")


def test_packaged_cli_refuses_a_missing_package_root(tmp_path: Path) -> None:
    script = _packaged_cli(tmp_path)
    package = script.parents[1] / ".package"
    package.write_text(package.read_text().replace(f"ROOT={tmp_path / 'libexec'}", f"ROOT={tmp_path / 'gone'}"))
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_GATEWAY_", "GATEWAY_VISION"))}
    env["HOME"] = str(tmp_path)
    result = subprocess.run([str(script), "env"], capture_output=True, text=True, env=env)
    assert result.returncode == 1
    assert "package root is missing" in result.stderr
