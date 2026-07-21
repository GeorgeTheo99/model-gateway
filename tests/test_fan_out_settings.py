"""Tests for the machine-local oMLX settings fan-out."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "runtime"
    / "omlx-config"
    / "fan_out_settings.py"
)
_SPEC = importlib.util.spec_from_file_location("fan_out_settings", _SCRIPT)
assert _SPEC and _SPEC.loader
fan_out_settings = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fan_out_settings)


def test_sync_omlx_settings_propagates_model_type_override(tmp_path):
    settings_path = tmp_path / "model_settings.json"
    settings_path.write_text(json.dumps({"version": 1, "models": {}}))

    fan_out_settings.sync_omlx_settings(
        [
            {
                "name": "laguna-s-2.1-6bit",
                "omlx_id": "Laguna-S-2.1-MLX-6bit",
                "context": 262_144,
                "max_output_tokens": 32_768,
                "model_type_override": "llm",
            }
        ],
        settings_path,
        dry_run=False,
    )

    model = json.loads(settings_path.read_text())["models"]["Laguna-S-2.1-MLX-6bit"]
    assert model["max_context_window"] == 262_144
    assert model["max_tokens"] == 32_768
    assert model["model_type_override"] == "llm"
