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
        },
    )
    assert "GATEWAY_VISION_FALLBACK_LOCAL=local-vision" in output
    assert "GATEWAY_VISION_FALLBACK_CLOUD=cloud-vision" in output
    assert "GATEWAY_VISION_FALLBACK_MODE=extract_then_answer" in output


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
    assert 'exec "$ROOT_DIR/bin/model-gateway" _update-after-pull' in update
