"""Tests for scripts/export_catalogs.py — the downstream alias catalog generator.

Covers: merge with model-info.json + overlay, alias rendering (full schema incl.
omlx_id), duplicate-alias hard failure, symlink-safe writes, --check drift
detection, opt-in exports. (Pi-specific models.json/launcher rendering lives in
pi-shared/lib/pi_catalog.py and has its own test suite.)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "export_catalogs.py"


def _write_model_info(path: Path, llm: list[dict], **extra) -> None:
    doc = {"llm": llm}
    doc.update(extra)
    path.write_text(json.dumps(doc))


def _run(config_path: Path, model_info_path: Path, *extra, env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + e.get("PYTHONPATH", ""))
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path), "--model-info", str(model_info_path), *extra],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=e,
    )


def test_no_exports_is_noop(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [{"name": "a", "alias": "a", "provider": "openrouter"}])
    cfg = tmp_path / "config.yaml"
    cfg.write_text("providers: {}\n")
    r = _run(cfg, mi)
    assert r.returncode == 0
    assert "no exports configured" in r.stdout


def test_renders_alias_file(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(
        mi,
        [
            {"name": "qwen3.6-35b", "alias": "qwen36", "provider": "local", "omlx_id": "qwen3.6-35b-mlx", "context": 262144, "max_output_tokens": 32768, "thinking": "always", "format": "mlx", "desc": "local qwen"},
            {"name": "glm-5.2", "alias": "glm52", "provider": "zai", "provider_model_id": "glm-5.2-zai", "context": 1000000, "max_output_tokens": 128000, "thinking": "always", "desc": "GLM 5.2"},
        ],
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"exports:\n"
        f"  model_aliases: {tmp_path}/aliases.json\n"
    )
    r = _run(cfg, mi)
    assert r.returncode == 0, r.stderr
    aliases = json.loads((tmp_path / "aliases.json").read_text())
    # local keyed by omlx_id, cloud by cloud:<provider_model_id>
    assert "qwen3.6-35b-mlx" in aliases
    assert "cloud:glm-5.2-zai" in aliases
    local_entry = aliases["qwen3.6-35b-mlx"]
    assert local_entry["omlx_id"] == "qwen3.6-35b-mlx"
    assert local_entry["alias"] == "qwen36"
    assert local_entry["thinking"] == "always"
    assert local_entry["thinking_levels"] == ["minimal", "low", "medium", "high", "xhigh", "max"]
    assert local_entry["format"] == "mlx"
    assert local_entry["context"] == 262144
    assert local_entry["max_output_tokens"] == 32768
    cloud_entry = aliases["cloud:glm-5.2-zai"]
    assert cloud_entry["provider"] == "zhipuai"  # canonicalized from zai
    assert cloud_entry["provider_model_id"] == "glm-5.2-zai"


def test_composite_exports_as_ordinary_local_vision_alias(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [{
        "name": "best-local",
        "alias": "best-local",
        "provider": "omlx",
        "omlx_id": "best-local",
        "vision": True,
        "context": 202752,
        "max_output_tokens": 32768,
        "thinking": "always",
        "thinking_format": "glm-chat-template",
        "composite": {
            "text_model": "glm-5.2-4.5bit",
            "vision_model": "gemma4-31b",
            "image_handling": "extract_then_answer",
            "max_images": 4,
        },
    }])
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"exports:\n  model_aliases: {tmp_path}/aliases.json\n")

    result = _run(cfg, mi)

    assert result.returncode == 0, result.stderr
    entry = json.loads((tmp_path / "aliases.json").read_text())["best-local"]
    assert entry["alias"] == "best-local"
    assert entry["vision"] is True
    assert entry["thinking_format"] == "glm-chat-template"
    assert entry["context"] == 202752
    assert "composite" not in entry


def test_balanced_and_legacy_detail_composites_export_as_distinct_routes(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [
        {
            "name": "auto-local",
            "alias": "auto-local",
            "provider": "omlx",
            "omlx_id": "auto-local",
            "vision": True,
            "composite": {
                "text_model": "glm-local",
                "vision_model": "vision-balanced",
            },
        },
        {
            "name": "detail-local",
            "alias": "detail-local",
            "provider": "omlx",
            "omlx_id": "detail-local",
            "vision": True,
            "composite": {
                "text_model": "glm-local",
                "vision_model": "vision-detail",
            },
        },
        {
            "name": "best-local",
            "alias": "best-local",
            "provider": "omlx",
            "omlx_id": "best-local",
            "vision": True,
            "composite": {
                "text_model": "glm-local",
                "vision_model": "vision-detail",
            },
        },
    ])
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"exports:\n  model_aliases: {tmp_path}/aliases.json\n")

    result = _run(cfg, mi)

    assert result.returncode == 0, result.stderr
    aliases = json.loads((tmp_path / "aliases.json").read_text())
    assert aliases["auto-local"]["alias"] == "auto-local"
    assert aliases["detail-local"]["alias"] == "detail-local"
    assert aliases["best-local"]["alias"] == "best-local"
    assert aliases["auto-local"]["name"] != aliases["detail-local"]["name"]


def test_relative_api_key_file_uses_selected_config_target(tmp_path):
    real_dir = tmp_path / "shared"
    real_dir.mkdir()
    secret_dir = real_dir / "secrets"
    secret_dir.mkdir()
    secret = secret_dir / "provider.key"
    secret.write_text("secret\n")
    secret.chmod(0o600)
    cfg_target = real_dir / "config.yaml"
    cfg_link = tmp_path / "config.yaml"
    cfg_link.unlink()
    cfg_link.symlink_to(cfg_target)
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [
        {"name": "cloud-model", "alias": "cloud", "provider": "cloud", "provider_model_id": "cloud-up"},
    ])
    cfg_target.write_text(
        "providers:\n  cloud:\n    base_url: https://api.example.com/v1\n"
        "    api_key_file: secrets/provider.key\n"
        f"exports:\n  model_aliases: {tmp_path}/aliases.json\n"
    )
    r = _run(cfg_link, mi)
    assert r.returncode == 0, r.stderr
    assert "cloud:cloud-up" in json.loads((tmp_path / "aliases.json").read_text())


def test_overlay_merge_used_by_generator(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [{"name": "model-v1", "alias": "model-old", "provider": "zai", "provider_model_id": "upstream-old", "context": 1000}])
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  - name: model-v1\n"
        "    alias: model-new\n"
        "    provider: fireworks\n"
        "    provider_model_id: upstream-new\n"
        "    context: 2000\n"
        f"exports:\n  model_aliases: {tmp_path}/aliases.json\n"
    )
    r = _run(cfg, mi)
    assert r.returncode == 0, r.stderr
    aliases = json.loads((tmp_path / "aliases.json").read_text())
    # Overlay wins: the cloud key is now the replacement provider_model_id.
    assert "cloud:upstream-new" in aliases
    assert "cloud:upstream-old" not in aliases  # catalog entry fully evicted
    assert aliases["cloud:upstream-new"]["alias"] == "model-new"
    assert aliases["cloud:upstream-new"]["context"] == 2000


def test_pooled_model_exported_with_effective_provider(tmp_path):
    """Pooled models (pool:, no provider:) must not be dropped as omlx-locals.

    Regression: pooled entries defaulted to provider 'omlx' in the merge and
    were then skipped for lacking an omlx_id, so every pooled databricks model
    vanished from the alias file (and downstream Pi launchers).
    """
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [])
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "providers:\n"
        "  db-west:\n"
        "    base_url: https://west.example.com\n"
        "    api_key: k\n"
        "    protocol: openai\n"
        "  db-east:\n"
        "    base_url: https://east.example.com\n"
        "    api_key: k\n"
        "    protocol: openai\n"
        "pools:\n"
        "  my-pool:\n"
        "  - db-west\n"
        "  - db-east\n"
        "models:\n"
        "  - name: claude-fable-5\n"
        "    pool: my-pool\n"
        "    alias: fable\n"
        "    provider_model_id: databricks-claude-fable-5\n"
        "    protocol: anthropic\n"
        "    context: 1000000\n"
        "    pi:\n"
        "      name: Fable via Databricks\n"
        "  - name: gpt-5.5\n"
        "    pool: my-pool\n"
        "    alias: gpt\n"
        "    provider_model_id: databricks-gpt-5-5\n"
        f"exports:\n  model_aliases: {tmp_path}/aliases.json\n"
    )
    r = _run(cfg, mi)
    assert r.returncode == 0, r.stderr
    aliases = json.loads((tmp_path / "aliases.json").read_text())
    fable = aliases["cloud:databricks-claude-fable-5"]
    # Effective provider = first pool member; protocol from entry override.
    assert fable["provider"] == "db-west"
    assert fable["protocol"] == "anthropic"
    assert fable["alias"] == "fable"
    assert fable["pi"] == {"name": "Fable via Databricks"}
    gpt = aliases["cloud:databricks-gpt-5-5"]
    # Protocol falls back to the provider config's protocol.
    assert gpt["protocol"] == "openai"


def test_duplicate_alias_hard_fails(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(
        mi,
        [
            {"name": "a", "alias": "dup", "provider": "openrouter", "provider_model_id": "a-id"},
            {"name": "b", "alias": "dup", "provider": "openrouter", "provider_model_id": "b-id"},
        ],
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"exports:\n  model_aliases: {tmp_path}/aliases.json\n")
    r = _run(cfg, mi)
    assert r.returncode == 2
    assert "routable id 'dup' collides" in r.stderr


def test_export_false_skips_model(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(
        mi,
        [
            {"name": "a", "alias": "a", "provider": "openrouter", "provider_model_id": "a-id", "export": False},
            {"name": "b", "alias": "b", "provider": "openrouter", "provider_model_id": "b-id"},
        ],
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"exports:\n  model_aliases: {tmp_path}/aliases.json\n")
    r = _run(cfg, mi)
    assert r.returncode == 0, r.stderr
    aliases = json.loads((tmp_path / "aliases.json").read_text())
    assert "cloud:a-id" not in aliases
    assert "cloud:b-id" in aliases


def test_symlink_safe_write(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [{"name": "a", "alias": "a", "provider": "openrouter", "provider_model_id": "a-id"}])
    # Create a symlink target and a symlink pointing at it.
    target = tmp_path / "real-aliases.json"
    target.write_text("{}")
    link = tmp_path / "link-aliases.json"
    link.symlink_to(target)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"exports:\n  model_aliases: {link}\n")
    r = _run(cfg, mi)
    assert r.returncode == 0, r.stderr
    # The symlink must still be a symlink (not replaced with a regular file),
    # and the target must hold the new content.
    assert link.is_symlink()
    assert json.loads(target.read_text()) == json.loads(link.read_text())
    assert "cloud:a-id" in json.loads(target.read_text())


def test_check_drift_detection(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [{"name": "a", "alias": "a", "provider": "openrouter", "provider_model_id": "a-id"}])
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"exports:\n  model_aliases: {tmp_path}/aliases.json\n")
    # First run writes the file.
    r = _run(cfg, mi)
    assert r.returncode == 0
    # Second run with --check: in sync.
    r = _run(cfg, mi, "--check")
    assert r.returncode == 0, r.stderr
    assert "in sync" in r.stdout
    # Mutate the file → drift.
    (tmp_path / "aliases.json").write_text("{}")
    r = _run(cfg, mi, "--check")
    assert r.returncode == 1
    assert "DRIFT" in r.stderr or "DRIFT" in r.stdout


def test_refuses_empty_catalog(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [])  # no entries
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"exports:\n  model_aliases: {tmp_path}/aliases.json\n")
    r = _run(cfg, mi)
    assert r.returncode != 0
    assert "empty catalog" in r.stderr


def test_gateway_does_not_own_pi_launchers():
    assert not (REPO_ROOT / "runtime" / "pi-launcher.zsh").exists()
    assert not (REPO_ROOT / "runtime" / "local_claude" / "zshrc-launcher.zsh").exists()


def test_committed_fixture_catalog_renders(tmp_path):
    """Smoke test against the machine-neutral committed model fixture."""
    mi = REPO_ROOT / "tests" / "fixtures" / "model-info.json"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"exports:\n"
        f"  model_aliases: {tmp_path}/aliases.json\n"
    )

    result = _run(cfg, mi)

    assert result.returncode == 0, result.stderr
    aliases = json.loads((tmp_path / "aliases.json").read_text())
    assert aliases["test-local-upstream"]["alias"] == "testlocal"
    assert aliases["test-local-upstream"]["omlx_id"] == "test-local-upstream"
