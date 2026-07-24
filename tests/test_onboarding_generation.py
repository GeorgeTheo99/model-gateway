import json
from pathlib import Path

import pytest
import yaml

from src import onboarding_generation as generation
from src.onboarding import OnboardingError, apply_profile, load_profile


def _catalog(path: Path, rows: list[dict] | None = None) -> Path:
    path.write_text(json.dumps({"llm": rows or []}))
    return path


def test_minimal_draft_is_secret_free_and_marks_unknown_fields(tmp_path):
    profile = generation.build_draft(
        provider_id="moonshot",
        base_url="https://api.moonshot.ai/v1/",
        model_ids=["kimi-k3"],
        model_info_path=_catalog(tmp_path / "model-info.json"),
        discovery={
            "status": "verified",
            "source": "provider_models",
            "model_ids": ["kimi-k3"],
            "http_status": 200,
        },
    )
    assert profile["provider"] == {
        "id": "moonshot",
        "base_url": "https://api.moonshot.ai/v1",
        "protocol": "openai",
        "secret_name": "moonshot.api-key",
    }
    assert profile["models"] == [{
        "name": "kimi-k3",
        "provider": "moonshot",
        "provider_model_id": "kimi-k3",
        "thinking_levels": [],
    }]
    assert profile["provenance"]["fields"]["/models/0/provider_model_id"] == {
        "source": "provider_models",
        "confidence": "verified",
    }
    assert "alias" in profile["provenance"]["unresolved"]
    assert "api_key" not in yaml.safe_dump(profile)


def test_draft_validates_and_writes_thinking_levels(tmp_path):
    profile = generation.build_draft(
        provider_id="moonshot",
        base_url="https://api.moonshot.ai/v1",
        model_ids=["kimi-k3"],
        model_info_path=_catalog(tmp_path / "model-info.json"),
        thinking="always",
        thinking_levels=["max"],
    )
    assert profile["models"][0]["thinking_levels"] == ["max"]
    with pytest.raises(OnboardingError, match="supports off only"):
        generation.build_draft(
            provider_id="moonshot",
            base_url="https://api.moonshot.ai/v1",
            model_ids=["bad"],
            model_info_path=_catalog(tmp_path / "other-model-info.json"),
            thinking="always",
            thinking_levels=["off", "high"],
        )


def test_draft_uses_safe_bounded_profile_id(tmp_path):
    model_id = "accounts/example/models/" + "x" * 200
    profile = generation.build_draft(
        provider_id="example",
        base_url="https://api.example.com/v1",
        model_ids=[model_id],
        model_info_path=_catalog(tmp_path / "model-info.json"),
    )
    assert "/" not in profile["id"]
    assert len(profile["id"]) <= 120


def test_existing_metadata_is_never_removed_silently(tmp_path):
    catalog = _catalog(tmp_path / "model-info.json", [{
        "name": "model-a",
        "provider": "old-provider",
        "provider_model_id": "old-a",
        "alias": "a",
        "context": 1000,
        "vision": True,
        "params": "legacy-metadata",
    }])
    minimal = generation.build_draft(
        provider_id="new-provider",
        base_url="https://api.example.com/v1",
        model_ids=["model-a"],
        model_info_path=catalog,
    )
    assert minimal["provenance"]["safety"]["metadata_removals"] == {
        "model-a": ["alias", "context", "params", "vision"]
    }
    with pytest.raises(OnboardingError, match="remove existing metadata"):
        generation.validate_generated_approvals(minimal)

    preserved = generation.build_draft(
        provider_id="new-provider",
        base_url="https://api.example.com/v1",
        model_ids=["model-a"],
        model_info_path=catalog,
        preserve_existing_metadata=True,
        drop_existing_metadata=["vision"],
    )
    assert preserved["models"][0]["alias"] == "a"
    assert preserved["models"][0]["params"] == "legacy-metadata"
    assert "vision" not in preserved["models"][0]
    assert preserved["provenance"]["safety"]["metadata_removals"] == {"model-a": ["vision"]}
    generation.validate_generated_approvals(preserved, allow_metadata_removal=True)


def test_explicit_disabled_thinking_clears_preserved_levels(tmp_path):
    catalog = _catalog(tmp_path / "model-info.json", [{
        "name": "model-a",
        "provider": "example",
        "provider_model_id": "model-a",
        "thinking": "always",
        "thinking_levels": ["max"],
    }])

    preserved = generation.build_draft(
        provider_id="example",
        base_url="https://api.example.com/v1",
        model_ids=["model-a"],
        model_info_path=catalog,
        preserve_existing_metadata=True,
    )
    assert preserved["models"][0]["thinking"] == "always"
    assert preserved["models"][0]["thinking_levels"] == ["max"]

    disabled = generation.build_draft(
        provider_id="example",
        base_url="https://api.example.com/v1",
        model_ids=["model-a"],
        model_info_path=catalog,
        thinking="",
        preserve_existing_metadata=True,
    )
    assert disabled["models"][0]["thinking"] == ""
    assert disabled["models"][0]["thinking_levels"] == []


def test_generated_retirements_require_exact_confirmation(tmp_path):
    profile = generation.build_draft(
        provider_id="example",
        base_url="https://api.example.com/v1",
        model_ids=["new"],
        model_info_path=_catalog(tmp_path / "model-info.json"),
        retirements=["old-a", "old-b"],
    )
    with pytest.raises(OnboardingError, match="exact confirmation"):
        generation.validate_generated_approvals(profile, confirmed_retirements={"old-a"})
    generation.validate_generated_approvals(
        profile,
        confirmed_retirements={"old-a", "old-b"},
    )


def test_write_draft_refuses_overwrite_and_symlink(tmp_path):
    profile = {"schema_version": 1}
    path = tmp_path / "draft.yaml"
    generation.write_draft(profile, path)
    with pytest.raises(OnboardingError, match="already exists"):
        generation.write_draft(profile, path)
    generation.write_draft({"schema_version": 2}, path, force=True)
    assert yaml.safe_load(path.read_text())["schema_version"] == 2

    link = tmp_path / "link.yaml"
    link.symlink_to(path)
    with pytest.raises(OnboardingError, match="symlink"):
        generation.write_draft(profile, link, force=True)


def test_discover_models_classifies_results(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {"data": [{"id": "b"}, {"id": "a"}, {"ignored": True}]}

    monkeypatch.setattr(generation.httpx, "get", lambda *args, **kwargs: Response())
    result = generation.discover_models("https://api.example.com/v1", "secret")
    assert result["status"] == "verified"
    assert result["model_ids"] == ["a", "b"]
    assert "secret" not in json.dumps(result)

    Response.status_code = 401
    result = generation.discover_models("https://api.example.com/v1", "secret")
    assert result["status"] == "authentication_failed"


def test_probe_records_only_bounded_outcome(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "probe_ok", "arguments": "{}"},
            }]}}]}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(generation.httpx, "post", fake_post)
    result = generation.run_probe(
        "https://api.example.com/v1", "model-a", "tools", "top-secret"
    )
    assert result["status"] == "observed_success"
    assert result["kind"] == "tools"
    assert "top-secret" not in json.dumps(result)
    assert captured["json"]["tool_choice"] == "required"
    assert captured["timeout"] == 60


def test_probe_rejects_redirects_and_malformed_success(monkeypatch):
    class Response:
        status_code = 302

        def json(self):
            return {"choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr(generation.httpx, "post", lambda *args, **kwargs: Response())
    assert generation.run_probe(
        "https://api.example.com/v1", "model-a", "text", "secret"
    )["status"] == "rejected"

    Response.status_code = 200
    for malformed in (
        {},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": ""}}]},
    ):
        Response.json = lambda self, value=malformed: value
        assert generation.run_probe(
            "https://api.example.com/v1", "model-a", "text", "secret"
        )["status"] == "malformed"
    Response.json = lambda self: {"choices": [{"message": {"tool_calls": [{
        "type": "function",
        "function": {"name": "wrong", "arguments": "not-json"},
    }]}}]}
    assert generation.run_probe(
        "https://api.example.com/v1", "model-a", "tools", "secret"
    )["status"] == "malformed"
    Response.json = lambda self: {"choices": [{"message": {"content": "OK"}}]}
    assert generation.run_probe(
        "https://api.example.com/v1", "model-a", "text", "secret"
    )["status"] == "observed_success"
    assert generation.discovery_allows_override({
        "models": [{"provider_model_id": "model-a"}],
        "provenance": {
            "generator": "model-gateway onboard generate",
            "discovery": {"status": "conflict"},
            "probes": [{"kind": "text", "model_id": "model-a", "status": "malformed"}],
            "safety": {"metadata_removals": {}, "retirements": []},
        },
    }) is False


def test_apply_engine_enforces_generated_safety_gates(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_SECRET_DIR", str(tmp_path / "secrets"))
    model_info = _catalog(tmp_path / "model-info.json", [{
        "name": "same",
        "provider": "old",
        "provider_model_id": "old-same",
        "alias": "old-alias",
    }])
    config = tmp_path / "config.yaml"
    config.write_text("providers: {}\n")
    profile = generation.build_draft(
        provider_id="new",
        base_url="https://api.example.com/v1",
        model_ids=["same"],
        model_info_path=model_info,
    )
    with pytest.raises(OnboardingError, match="remove existing metadata"):
        apply_profile(
            profile,
            config_path=config,
            model_info_path=model_info,
            api_key="secret",
            check_upstream=False,
        )
    result = apply_profile(
        profile,
        config_path=config,
        model_info_path=model_info,
        api_key="secret",
        check_upstream=False,
        allow_metadata_removal=True,
        confirmed_replacements={"same"},
    )
    assert result["added_models"] == ["same"]
    repeated = apply_profile(
        profile,
        config_path=config,
        model_info_path=model_info,
        api_key="secret",
        check_upstream=False,
        confirmed_replacements={"same"},
    )
    assert repeated["added_models"] == ["same"]


def test_metadata_removal_gate_survives_removed_generated_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_SECRET_DIR", str(tmp_path / "secrets"))
    model_info = _catalog(tmp_path / "model-info.json", [{
        "name": "model-a",
        "provider": "example",
        "provider_model_id": "model-a",
        "alias": "must-not-disappear",
    }])
    config = tmp_path / "config.yaml"
    config.write_text("providers: {}\n")
    profile = {
        "schema_version": 1,
        "id": "edited-generated-profile",
        "provider": {
            "id": "example",
            "base_url": "https://api.example.com/v1",
        },
        "models": [{
            "name": "model-a",
            "provider": "example",
            "provider_model_id": "model-a",
        }],
    }
    dry_run = apply_profile(
        profile,
        config_path=config,
        model_info_path=model_info,
        dry_run=True,
    )
    assert dry_run["metadata_removals"] == {"model-a": ["alias"]}
    assert dry_run["requires_metadata_removal_approval"] is True
    with pytest.raises(OnboardingError, match="remove existing metadata"):
        apply_profile(
            profile,
            config_path=config,
            model_info_path=model_info,
            api_key="secret",
            check_upstream=False,
        )
    profile["models"][0]["alias"] = "stale-value"
    with pytest.raises(OnboardingError, match="replacements require exact confirmation"):
        apply_profile(
            profile,
            config_path=config,
            model_info_path=model_info,
            api_key="secret",
            check_upstream=False,
        )


def test_apply_engine_rejects_catalog_drift_after_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_SECRET_DIR", str(tmp_path / "secrets"))
    model_info = _catalog(tmp_path / "model-info.json")
    config = tmp_path / "config.yaml"
    config.write_text("providers: {}\n")
    profile = generation.build_draft(
        provider_id="example",
        base_url="https://api.example.com/v1",
        model_ids=["model-a"],
        model_info_path=model_info,
    )
    model_info.write_text(json.dumps({"llm": [{
        "name": "model-a",
        "provider": "example",
        "provider_model_id": "model-a",
        "alias": "added-later",
    }]}))
    with pytest.raises(OnboardingError, match="changed after draft generation"):
        apply_profile(
            profile,
            config_path=config,
            model_info_path=model_info,
            api_key="secret",
            check_upstream=False,
            allow_metadata_removal=True,
        )


def test_apply_engine_rejects_preserved_value_drift_after_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_SECRET_DIR", str(tmp_path / "secrets"))
    model_info = _catalog(tmp_path / "model-info.json", [{
        "name": "model-a",
        "provider": "example",
        "provider_model_id": "model-a",
        "alias": "v1",
    }])
    config = tmp_path / "config.yaml"
    config.write_text("providers: {}\n")
    profile = generation.build_draft(
        provider_id="example",
        base_url="https://api.example.com/v1",
        model_ids=["model-a"],
        model_info_path=model_info,
        preserve_existing_metadata=True,
    )
    model_info.write_text(json.dumps({"llm": [{
        "name": "model-a",
        "provider": "example",
        "provider_model_id": "model-a",
        "alias": "v2",
    }]}))
    with pytest.raises(OnboardingError, match="changed after draft generation"):
        apply_profile(
            profile,
            config_path=config,
            model_info_path=model_info,
            dry_run=True,
        )
    with pytest.raises(OnboardingError, match="changed after draft generation"):
        apply_profile(
            profile,
            config_path=config,
            model_info_path=model_info,
            api_key="secret",
            check_upstream=False,
        )


def test_generated_profile_round_trips_through_schema_v1_loader(tmp_path):
    catalog = _catalog(tmp_path / "model-info.json")
    profile = generation.build_draft(
        provider_id="example",
        base_url="https://api.example.com/v1",
        model_ids=["model-a"],
        model_info_path=catalog,
        context=128000,
        documented_fields={"context": "https://docs.example.com/model-a"},
        pricing={"input": 1.0, "output": 2.0},
        quirks=["use_max_completion_tokens"],
    )
    path = generation.write_draft(profile, tmp_path / "draft.yaml")
    loaded = load_profile(path)
    assert loaded == profile
    assert loaded["models"][0]["pricing"]["output"] == 2.0
    assert loaded["provenance"]["fields"]["/models/0/context"] == {
        "source": "documentation",
        "confidence": "confirmed",
        "url": "https://docs.example.com/model-a",
    }


def test_invalid_generated_metadata_is_rejected(tmp_path):
    catalog = _catalog(tmp_path / "model-info.json")
    with pytest.raises(OnboardingError, match="unsupported request quirk"):
        generation.build_draft(
            provider_id="example",
            base_url="https://api.example.com/v1",
            model_ids=["model-a"],
            model_info_path=catalog,
            quirks=["invented_quirk"],
        )
    with pytest.raises(OnboardingError, match="pricing values"):
        generation.build_draft(
            provider_id="example",
            base_url="https://api.example.com/v1",
            model_ids=["model-a"],
            model_info_path=catalog,
            pricing={"input": -1, "output": 2},
        )
    with pytest.raises(OnboardingError, match="finite non-negative"):
        generation.build_draft(
            provider_id="example",
            base_url="https://api.example.com/v1",
            model_ids=["model-a"],
            model_info_path=catalog,
            pricing={"input": float("nan"), "output": 2},
        )
    with pytest.raises(OnboardingError, match="pricing requires: output"):
        generation.build_draft(
            provider_id="example",
            base_url="https://api.example.com/v1",
            model_ids=["model-a"],
            model_info_path=catalog,
            pricing={"input": 1},
        )


def test_base_url_rejects_query_fragment_and_embedded_credentials(tmp_path):
    catalog = _catalog(tmp_path / "model-info.json")
    for url in (
        "https://api.example.com/v1?api_key=secret",
        "https://api.example.com/v1#fragment",
        "https://user:secret@api.example.com/v1",
    ):
        with pytest.raises(OnboardingError, match="query, or fragment"):
            generation.build_draft(
                provider_id="example",
                base_url=url,
                model_ids=["model-a"],
                model_info_path=catalog,
            )


def test_draft_target_cannot_collide_with_runtime_files(tmp_path):
    runtime = tmp_path / "config.yaml"
    runtime.write_text("providers: {}\n")
    with pytest.raises(OnboardingError, match="protected runtime file"):
        generation.write_draft(
            {"schema_version": 1},
            runtime,
            force=True,
            protected_paths=[runtime],
        )
    alias_dir = tmp_path / "alias"
    alias_dir.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OnboardingError, match="protected runtime file"):
        generation.write_draft(
            {"schema_version": 1},
            alias_dir / "config.yaml",
            force=True,
            protected_paths=[runtime],
        )


def test_discovery_override_requires_real_inconclusive_state_and_never_bypasses_auth():
    base = {
        "models": [{"provider_model_id": "model-a"}],
        "provenance": {
            "generator": "model-gateway onboard generate",
            "probes": [],
        },
    }
    base["provenance"]["discovery"] = {"status": "not_attempted"}
    assert generation.discovery_allows_override(base) is False
    base["provenance"]["discovery"] = {"status": "unavailable"}
    assert generation.discovery_allows_override(base) is True
    base["provenance"]["discovery"] = {"status": "authentication_failed"}
    base["provenance"]["probes"] = [{
        "kind": "text",
        "model_id": "model-a",
        "status": "observed_success",
    }]
    assert generation.discovery_allows_override(base) is False


def test_generated_safety_must_match_profile_retirements(tmp_path):
    catalog = _catalog(tmp_path / "model-info.json", [{
        "name": "old",
        "provider": "example",
        "provider_model_id": "old",
    }])
    profile = generation.build_draft(
        provider_id="example",
        base_url="https://api.example.com/v1",
        model_ids=["new"],
        model_info_path=catalog,
        retirements=["old"],
    )
    profile["retire"]["models"] = ["different"]
    with pytest.raises(OnboardingError, match="do not match"):
        generation.generated_safety(profile)
