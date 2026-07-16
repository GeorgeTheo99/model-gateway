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
    assert local_entry["format"] == "mlx"
    assert local_entry["context"] == 262144
    assert local_entry["max_output_tokens"] == 32768
    cloud_entry = aliases["cloud:glm-5.2-zai"]
    assert cloud_entry["provider"] == "zhipuai"  # canonicalized from zai
    assert cloud_entry["provider_model_id"] == "glm-5.2-zai"


def test_overlay_merge_used_by_generator(tmp_path):
    mi = tmp_path / "model-info.json"
    _write_model_info(mi, [{"name": "glm-5.2", "alias": "glm52zai", "provider": "zai", "provider_model_id": "glm-5.2-zai", "context": 1000}])
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  - name: glm-5.2\n"
        "    alias: glm52fw\n"
        "    provider: fireworks\n"
        "    provider_model_id: accounts/fireworks/models/glm-5p2\n"
        "    context: 2000\n"
        f"exports:\n  model_aliases: {tmp_path}/aliases.json\n"
    )
    r = _run(cfg, mi)
    assert r.returncode == 0, r.stderr
    aliases = json.loads((tmp_path / "aliases.json").read_text())
    # Overlay wins: the cloud key is now the fireworks provider_model_id.
    assert "cloud:accounts/fireworks/models/glm-5p2" in aliases
    assert "cloud:glm-5.2-zai" not in aliases  # catalog entry fully evicted
    assert aliases["cloud:accounts/fireworks/models/glm-5p2"]["alias"] == "glm52fw"
    assert aliases["cloud:accounts/fireworks/models/glm-5p2"]["context"] == 2000


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
    assert "duplicate aliases" in r.stderr


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


def test_machine_catalog_renders(tmp_path):
    """Smoke test against this machine's optional, gitignored catalog."""
    mi = REPO_ROOT / "model-info.json"
    if not mi.exists():
        pytest.skip("model-info.json not present")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"exports:\n"
        f"  model_aliases: {tmp_path}/aliases.json\n"
    )
    r = _run(cfg, mi)
    assert r.returncode == 0, r.stderr
    aliases = json.loads((tmp_path / "aliases.json").read_text())
    assert aliases
    catalog = json.loads(mi.read_text())
    has_local = any(
        str(entry.get("provider") or "omlx").lower() in {"local", "omlx"}
        for entry in catalog.get("llm", [])
    )
    assert any(k.startswith("cloud:") for k in aliases)
    assert any(not k.startswith("cloud:") for k in aliases) is has_local
