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
        "MODEL_GATEWAY_PLIST_DIR",
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
    assert "$" not in template
    config = yaml.safe_load(template)
    assert config["exports"] == {
        "model_aliases": "~/Library/Application Support/model-gateway/model-aliases.json"
    }
    assert config["auth"] == {"admin_keys": [], "client_keys": []}


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
