"""Secret-free, additive metadata for the admin model inventory."""

import json

import pytest

from src import providers


@pytest.fixture
def registry(monkeypatch, tmp_path):
    config = {
        "providers": {
            "openai": {"base_url": "https://cloud.example/v1", "api_key": "provider-secret"},
            "omlx": {"base_url": "http://localhost:9110/v1", "api_key": "local-secret"},
            "disabled": {"enabled": False, "api_key": "disabled-secret"},
        },
        "models": [],
    }
    monkeypatch.setattr(providers, "_config", config)
    monkeypatch.setattr(providers, "_models", None)
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", tmp_path / "absent.json")
    return config


def status_by_name():
    return {row["name"]: row for row in providers.model_status()}


def test_model_status_metadata_is_whitelisted(registry):
    registry["models"] = [{
        "name": "cloud-model",
        "provider": "gpt",
        "desc": "A model description",
        "tools": True,
        "api_key": "model-secret",
        "system_instruction": "private-system-instruction",
        "custom_metadata": {"token": "nested-secret"},
        "tool_parser": "private-parser-setting",
    }]

    row = status_by_name()["cloud-model"]
    assert row["desc"] == "A model description"
    assert row["tools"] is True
    assert row["pool"] == ""
    assert row["declared_providers"] == ["openai"]
    assert row["locality"] == "cloud"
    assert row["composite"] is None
    assert row["fallback_model"] is None
    serialized = json.dumps(row)
    for private in (
        "api_key", "system_instruction", "custom_metadata", "tool_parser",
        "provider-secret", "model-secret", "private-system-instruction",
        "nested-secret", "private-parser-setting", "cloud.example",
    ):
        assert private not in serialized


@pytest.mark.parametrize("tools, expected", [(True, True), (False, False), ("true", None), (1, None), ({"secret": "hidden"}, None), (None, None)])
def test_tools_requires_an_explicit_boolean(registry, tools, expected):
    registry["models"] = [{"name": "model", "tools": tools, "tool_parser": "known-parser"}]
    assert status_by_name()["model"]["tools"] is expected


def test_missing_tools_does_not_infer_from_parser_and_fields_are_strings(registry):
    registry["models"] = [{
        "name": "model", "tool_parser": "known-parser",
        "desc": {"secret": "hidden"}, "pool": {"secret": "hidden"},
    }]
    row = status_by_name()["model"]
    assert row["tools"] is None
    assert row["desc"] == ""
    assert row["pool"] == ""
    assert row["locality"] == "local"
    assert "hidden" not in json.dumps(row)


@pytest.mark.parametrize(
    "members, declared, locality",
    [
        (["gpt", "disabled", "missing"], ["openai", "disabled", "missing"], "cloud"),
        (["disabled", "mlx", "missing"], ["disabled", "omlx", "missing"], "mixed"),
        (["local", "mlx"], ["omlx", "omlx"], "local"),
        (["disabled", "missing"], ["disabled", "missing"], "cloud"),
    ],
)
def test_declared_pool_order_includes_unavailable_members(registry, members, declared, locality):
    registry["pools"] = {"ordered-pool": members}
    registry["models"] = [{"name": "model", "pool": "ordered-pool"}]
    row = status_by_name()["model"]
    assert row["pool"] == "ordered-pool"
    assert row["declared_providers"] == declared
    assert row["candidate_providers"] == [p for p in declared if p in {"openai", "omlx"}]
    assert row["locality"] == locality


@pytest.mark.parametrize("vision_provider", ["omlx", "openai"])
def test_composite_targets_are_whitelisted_without_claiming_wrapper_locality(registry, vision_provider):
    registry["models"] = [
        {"name": "text", "provider": "omlx"},
        {"name": "vision", "provider": vision_provider, "vision": True},
        {
            "name": "combined", "provider": "omlx", "vision": True,
            "composite": {
                "text_model": "text", "vision_model": "vision",
                "system_instruction": "private-composite-instruction",
                "api_key": "composite-secret", "unknown": {"secret": "nested-secret"},
            },
        },
    ]
    registry["model_fallbacks"] = {"combined": "not-an-upstream-fallback"}
    row = status_by_name()["combined"]
    assert row["composite"] == {"text_model": "text", "vision_model": "vision", "image_handling": "extract_then_answer"}
    assert row["locality"] is None
    assert row["fallback_model"] is None
    assert "secret" not in json.dumps(row)
    assert "private-composite-instruction" not in json.dumps(row)


@pytest.mark.parametrize("id_field", ["provider_model_id", "omlx_id", "name"])
def test_fallback_uses_upstream_model_id_not_alias(registry, id_field):
    model = {"name": "logical", "alias": "short", id_field: "upstream"}
    registry["models"] = [model]
    registry["model_fallbacks"] = {"upstream": "fallback-target", "short": "wrong-target"}
    row = status_by_name()[model["name"]]
    assert row["fallback_model"] == "fallback-target"


@pytest.mark.parametrize("mapping", [[], "invalid", {"model": {"api_key": "hidden"}}, {"model": False}, {"model": ""}])
def test_fallback_metadata_never_serializes_non_string_configuration(registry, mapping):
    registry["models"] = [{"name": "model"}]
    registry["model_fallbacks"] = mapping
    row = status_by_name()["model"]
    assert row["fallback_model"] is None
    assert "hidden" not in json.dumps(row)


def test_provider_status_marks_redacted_urls_without_exposing_components(registry):
    registry["models"] = [{"name": "model", "provider": "openai"}]
    registry["providers"]["openai"]["base_url"] = "https://user:private-pass@cloud.example/v1?key=private-query"
    row = next(p for p in providers.provider_status() if p["id"] == "openai")
    assert row["base_url"] == "https://cloud.example/v1"
    assert row["base_url_redacted"] is True
    assert "private" not in json.dumps(row)


def test_composite_image_reroute_mode_is_preserved(registry):
    registry["models"] = [
        {"name": "text"}, {"name": "vision", "vision": True},
        {"name": "combo", "composite": {"text_model": "text", "vision_model": "vision", "image_handling": "reroute"}},
    ]
    assert status_by_name()["combo"]["composite"]["image_handling"] == "reroute"
