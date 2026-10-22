"""The shipped catalog layer and the owner's precedence over it (ADR 0002)."""

import json

import pytest
import yaml

import src.providers as providers
from src import catalog, onboarding, shipped

BASE_URL = "https://api.example.test/v1"


def _profile(models, *, retire=None, profile_id="example-curated"):
    data = {
        "schema_version": 1,
        "id": profile_id,
        "auto_add": True,
        "provider": {"id": "example", "base_url": BASE_URL, "protocol": "openai"},
        "models": [
            {"provider": "example", "provider_model_id": f"up/{m['name']}", "context": 1000,
             "pricing": {"input": 1.0, "output": 2.0}, **m}
            for m in models
        ],
    }
    if retire:
        data["retire"] = retire
    return onboarding.validate_profile(data)


def _entries(tmp_path, owner_rows, profiles, config=None, ready=lambda profile, config: True):
    path = tmp_path / "model-info.json"
    path.write_text(json.dumps({"llm": owner_rows, "allow_empty": True}))
    layer = shipped.layer(config or {}, profiles=profiles, ready=ready)
    entries = catalog.load_catalog_entries(
        path, overlay=(config or {}).get("models", []), shipped=layer.entries, shipped_names=layer.names,
        excluded=layer.hidden | frozenset(layer.retired),
    )
    return {entry["name"]: entry for entry in entries}, layer


def test_shipped_models_load_by_reference(tmp_path):
    models, _ = _entries(tmp_path, [], {"p": _profile([{"name": "m1"}, {"name": "m2"}])})
    assert list(models) == ["m1", "m2"]
    assert models["m1"]["origin"] == {"layer": "shipped", "profile": "p"}


def test_not_ready_provider_loads_nothing(tmp_path):
    models, layer = _entries(tmp_path, [], {"p": _profile([{"name": "m1"}])}, ready=lambda p, c: False)
    assert models == {} and layer.names == {"m1"}


def test_owner_override_wins_per_field_and_keeps_shipped_updates(tmp_path):
    profiles = {"p": _profile([{"name": "m1", "context": 2000}])}
    owner = [{"name": "m1", "pricing": {"input": 0.5, "output": 1.0}, "shipped_base": {"pricing": {"input": 9, "output": 9}}}]
    models, _ = _entries(tmp_path, owner, profiles)
    m1 = models["m1"]
    assert m1["pricing"] == {"input": 0.5, "output": 1.0}
    assert m1["context"] == 2000  # shipped change still flows
    assert m1["origin"]["overridden"] == ["pricing"]
    assert m1["origin"]["stale_overrides"] == ["pricing"]  # shipped pricing changed since the override
    assert "shipped_base" not in m1


def test_legacy_full_copy_merges_to_the_shipped_row(tmp_path):
    profile = _profile([{"name": "m1"}])
    models, _ = _entries(tmp_path, [dict(profile["models"][0])], {"p": profile})
    assert models["m1"]["origin"]["overridden"] == []


def test_sparse_override_of_unloaded_shipped_model_is_dormant(tmp_path):
    owner = [{"name": "m1", "pricing": {"input": 0.5, "output": 1.0}, "shipped_base": {}},
             {"name": "m2", "provider": "example", "provider_model_id": "up/m2"}]
    models, _ = _entries(tmp_path, owner, {"p": _profile([{"name": "m1"}, {"name": "m2"}])},
                         ready=lambda p, c: False)
    assert list(models) == ["m2"]  # a full legacy copy still stands on its own
    assert models["m2"]["origin"] == {"layer": "owner"}


def test_owner_claims_single_identifiers_and_shadows_by_name(tmp_path):
    profiles = {"p": _profile([
        {"name": "glm-fw", "alias": "glm", "alternate_ids": ["glm-5.3", "glm-latest"]},
        {"name": "astra"},
    ])}
    owner = [
        {"name": "glm-local", "provider": "omlx", "omlx_id": "glm-local", "alternate_ids": ["glm-5.3"],
         "pricing_status": "unmetered"},
        {"name": "my-astra", "provider": "omlx", "omlx_id": "my-astra", "alternate_ids": ["astra"],
         "pricing_status": "unmetered"},
    ]
    models, _ = _entries(tmp_path, owner, profiles)
    assert models["glm-fw"]["alternate_ids"] == ["glm-latest"]
    assert models["glm-fw"]["alias"] == "glm"
    assert "astra" not in models  # the owner took its name


@pytest.mark.parametrize("selection,expected", [
    ("none", []), (["m2"], ["m2"]), (["p"], ["m1", "m2"]), ("all", ["m1", "m2"]),
])
def test_selection_mode(tmp_path, selection, expected):
    config = {"catalog": {"shipped": selection}}
    models, _ = _entries(tmp_path, [], {"p": _profile([{"name": "m1"}, {"name": "m2"}])}, config)
    assert list(models) == expected


def test_hidden_retired_and_pinned(tmp_path):
    profiles = {
        "old": _profile([{"name": "other"}], profile_id="old",
                        retire={"models": ["gone", "pinned-one", "keeps"], "replaced_by": {"gone": "new"}}),
        "new": _profile([{"name": "new"}, {"name": "hide-me"}, {"name": "keeps"}], profile_id="new"),
    }
    owner = [{"name": n, "provider": "omlx", "omlx_id": n, "pricing_status": "unmetered"}
             for n in ("gone", "pinned-one")]
    config = {"catalog": {"hidden": ["hide-me"], "pinned": ["pinned-one"]}}
    models, layer = _entries(tmp_path, owner, profiles, config)
    assert sorted(models) == ["keeps", "new", "other", "pinned-one"]
    # "keeps" is still shipped by a profile, so the older retirement does not apply.
    assert layer.retired == {"gone": "new"}


def test_replaced_by_must_name_a_live_model():
    with pytest.raises(onboarding.OnboardingError, match="replaced_by"):
        _profile([{"name": "m1"}], retire={"models": ["old"], "replaced_by": {"other": "m1"}})
    with pytest.raises(onboarding.OnboardingError, match="replaced_by"):
        _profile([{"name": "m1"}], retire={"models": ["a", "b"], "replaced_by": {"a": "b"}})


@pytest.mark.parametrize("raw", [{"shipped": "some"}, {"hidden": "x"}, {"pinned": [""]}, {"extra": 1}, []])
def test_invalid_catalog_settings_are_rejected(raw):
    with pytest.raises(ValueError, match="catalog"):
        shipped.settings({"catalog": raw})


def test_registry_loads_shipped_models_for_a_ready_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(shipped, "_packaged", lambda: {"p": _profile([{"name": "m1"}])})
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"llm": [], "allow_empty": True}))
    monkeypatch.setattr(providers, "MODEL_INFO_PATH", info)
    config = {"providers": {"example": {"base_url": BASE_URL, "api_key": "k"}}}
    providers.CONFIG_PATH.write_text(yaml.safe_dump(config))
    providers.reload()
    assert providers.resolve("m1").provider_model_id == "up/m1"

    for change in ({"api_key": ""}, {"base_url": "https://elsewhere.test/v1"}, {"enabled": False}):
        providers.CONFIG_PATH.write_text(yaml.safe_dump(
            {"providers": {"example": {**config["providers"]["example"], **change}}}))
        providers.reload()
        assert "m1" not in providers._load_models()
