"""Gateway-owned logical text+vision composite model tests."""

from copy import deepcopy

import pytest

import src.providers as providers


BASE_MODELS = [
    {
        "name": "text-local",
        "alias": "textlocal",
        "provider": "omlx",
        "omlx_id": "text-upstream",
        "context": 2048,
        "max_output_tokens": 256,
        "thinking": "always",
        "thinking_format": "glm-chat-template",
    },
    {
        "name": "vision-local",
        "alias": "visionlocal",
        "provider": "omlx",
        "omlx_id": "vision-upstream",
        "vision": True,
        "context": 4096,
        "max_output_tokens": 512,
    },
    {
        "name": "best-local",
        "alias": "best-local",
        "provider": "omlx",
        "omlx_id": "best-local",
        "vision": True,
        "context": 2048,
        "max_output_tokens": 256,
        "thinking": "always",
        "thinking_format": "glm-chat-template",
        "composite": {
            "text_model": "text-local",
            "vision_model": "vision-local",
            "image_handling": "extract_then_answer",
            "max_images": 4,
        },
    },
]


def _install_registry(monkeypatch, models):
    monkeypatch.setattr(providers, "_config", {"models": models})
    monkeypatch.setattr(providers, "_models", None)
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", providers.Path("/nonexistent-model-info.json"))


@pytest.fixture(autouse=True)
def reset_registry():
    yield
    providers._config = None
    providers._models = None


def test_composite_resolves_to_text_upstream_with_scoped_policy(monkeypatch):
    _install_registry(monkeypatch, deepcopy(BASE_MODELS))

    info = providers.resolve("best-local")

    assert info is not None
    assert info.provider == "omlx"
    assert info.provider_model_id == "text-upstream"
    assert info.vision is False
    assert info.thinking_format == "glm-chat-template"
    assert info.composite == providers.CompositeRoute(
        text_model="text-local",
        vision_model="vision-local",
        image_handling="extract_then_answer",
        max_images=4,
    )
    assert providers.resolve("textlocal").composite is None


def test_composite_inventory_is_publicly_vision_capable(monkeypatch):
    _install_registry(monkeypatch, deepcopy(BASE_MODELS))

    row = next(model for model in providers.list_available_models() if model["id"] == "best-local")

    assert row["vision"] is True
    assert row["effective_provider"] == "omlx"
    assert row["available"] is True


def test_composite_forces_staging_even_if_text_dependency_is_multimodal(monkeypatch):
    models = deepcopy(BASE_MODELS)
    models[0]["vision"] = True
    _install_registry(monkeypatch, models)

    assert providers.resolve("text-local").vision is True
    assert providers.resolve("best-local").vision is False


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda models: models[2]["composite"].update(vision_model="missing"), "invalid_composite"),
        (lambda models: models[1].update(vision=False), "invalid_composite"),
        (lambda models: models[1].update(provider="fireworks"), "invalid_composite"),
        (lambda models: models[1].update(composite={"text_model": "text-local", "vision_model": "vision-local"}), "invalid_composite"),
        (lambda models: models[2]["composite"].update(max_images=0), "invalid_composite"),
    ],
)
def test_invalid_or_unavailable_composite_fails_closed(monkeypatch, mutate, reason):
    models = deepcopy(BASE_MODELS)
    mutate(models)
    _install_registry(monkeypatch, models)

    availability = providers.model_availability("best-local")

    assert availability["available"] is False
    assert availability["reason"] == reason
    assert providers.resolve("best-local") is None


def test_disabled_companion_disables_composite(monkeypatch):
    models = deepcopy(BASE_MODELS)
    _install_registry(monkeypatch, models)
    providers._config["model_overrides"] = {"vision-local": {"enabled": False}}
    providers._models = None

    availability = providers.model_availability("best-local")

    assert availability["available"] is False
    assert availability["reason"] == "composite_vision_model_unavailable"
    assert "disabled" in availability["message"]


def test_composite_rejects_dependency_pool_with_any_cloud_member(monkeypatch):
    models = deepcopy(BASE_MODELS)
    models[0].pop("provider", None)
    models[0]["pool"] = "mixed"
    config = {
        "providers": {
            "fireworks": {"base_url": "https://example.test/v1", "api_key": "secret"},
        },
        "pools": {"mixed": ["omlx", "fireworks"]},
        "models": models,
    }
    monkeypatch.setattr(providers, "_config", config)
    monkeypatch.setattr(providers, "_models", None)
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", providers.Path("/nonexistent-model-info.json"))

    availability = providers.model_availability("best-local")

    assert availability["available"] is False
    assert availability["reason"] == "invalid_composite"
    assert "only local oMLX" in availability["message"]
    assert providers.resolve("best-local") is None
