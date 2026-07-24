import json
import stat
from pathlib import Path

import pytest
import yaml

from src import onboarding


@pytest.fixture(autouse=True)
def _isolate_secret_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_SECRET_DIR", str(tmp_path / "secrets"))


def _profile():
    return {
        "schema_version": 1,
        "id": "new-provider-model",
        "provider": {
            "id": "new-provider",
            "base_url": "https://api.example.com/v1",
            "protocol": "openai",
            "secret_name": "new-provider.api-key",
        },
        "models": [
            {
                "name": "new-model",
                "provider": "new-provider",
                "provider_model_id": "new-upstream",
                "alias": "new",
                "context": 128000,
                "max_output_tokens": 16000,
                "quirks": ["use_max_completion_tokens"],
            }
        ],
        "retire": {"models": ["old-model"]},
    }


def _files(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "providers:\n  old-provider:\n    base_url: https://old.example/v1\n    api_key: old-secret\n"
        "model_overrides:\n  old-model:\n    enabled: false\n"
    )
    model_info = tmp_path / "model-info.json"
    model_info.write_text(json.dumps({"llm": [{"name": "old-model", "provider": "old-provider", "provider_model_id": "old-up"}]}))
    source = tmp_path / "source-model-info.json"
    source.write_text(model_info.read_text())
    return config, model_info, source


def test_load_profile_validates_schema(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(_profile(), sort_keys=False))
    assert onboarding.load_profile(path)["id"] == "new-provider-model"
    path.write_text("schema_version: 2\n")
    with pytest.raises(onboarding.OnboardingError, match="schema_version"):
        onboarding.load_profile(path)


def test_dry_run_makes_no_changes_and_needs_no_key(tmp_path):
    config, model_info, source = _files(tmp_path)
    before = [path.read_bytes() for path in (config, model_info, source)]
    result = onboarding.apply_profile(
        _profile(), config_path=config, model_info_path=model_info,
        model_info_source_path=source, dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["added_models"] == ["new-model"]
    assert result["retired_models"] == ["old-model"]
    assert before == [path.read_bytes() for path in (config, model_info, source)]


def test_same_live_and_source_model_catalog_is_valid(tmp_path):
    config, model_info, _source = _files(tmp_path)
    result = onboarding.apply_profile(
        _profile(), config_path=config, model_info_path=model_info,
        model_info_source_path=model_info, dry_run=True,
    )
    assert result["added_models"] == ["new-model"]


def test_profile_rejects_cloud_unmetered_pricing(tmp_path):
    profile = _profile()
    profile["models"][0]["pricing_status"] = "unmetered"
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile, sort_keys=False))
    with pytest.raises(onboarding.OnboardingError, match="only valid for local"):
        onboarding.load_profile(path)

    config, model_info, source = _files(tmp_path)
    with pytest.raises(onboarding.OnboardingError, match="only valid for local"):
        onboarding.apply_profile(
            profile,
            config_path=config,
            model_info_path=model_info,
            model_info_source_path=source,
            dry_run=True,
        )


def test_profile_validates_thinking_capabilities_before_writes(tmp_path):
    profile = _profile()
    profile["models"][0].update({"thinking": "always", "thinking_levels": ["off", "high"]})
    with pytest.raises(onboarding.OnboardingError, match="supports off only"):
        onboarding.validate_profile(profile)


@pytest.mark.parametrize("legacy_mode", ["never", None])
def test_profile_canonicalizes_legacy_no_thinking_modes(legacy_mode):
    profile = _profile()
    profile["models"][0]["thinking"] = legacy_mode

    validated = onboarding.validate_profile(profile)

    assert validated["models"][0]["thinking"] == ""
    assert validated["models"][0]["thinking_levels"] == []


def test_direct_apply_validates_profile_before_using_credentials(tmp_path):
    config, model_info, source = _files(tmp_path)
    profile = _profile()
    profile["schema_version"] = 999
    profile["provider"]["base_url"] = "http://user:password@example.com/v1"
    profile["models"].append(dict(profile["models"][0]))
    with pytest.raises(onboarding.OnboardingError, match="schema_version"):
        onboarding.apply_profile(
            profile,
            config_path=config,
            model_info_path=model_info,
            model_info_source_path=source,
            api_key="must-not-be-sent",
            dry_run=True,
        )


def test_direct_apply_rejects_malformed_identifiers_without_writes(tmp_path):
    config, model_info, source = _files(tmp_path)
    before = (config.read_bytes(), model_info.read_bytes(), source.read_bytes())
    profile = _profile()
    profile["models"][0]["alias"] = {"bad": "id"}
    profile["models"][0]["alternate_ids"] = ["valid", 7]

    with pytest.raises(onboarding.OnboardingError, match="alias must be a string"):
        onboarding.apply_profile(
            profile,
            config_path=config,
            model_info_path=model_info,
            model_info_source_path=source,
            api_key="must-not-be-written",
            confirmed_retirements={"old-model"},
        )

    assert (config.read_bytes(), model_info.read_bytes(), source.read_bytes()) == before
    assert not list((tmp_path / "secrets").glob("*"))


def test_apply_profile_retires_old_model_and_uses_secret_file(tmp_path):
    config, model_info, source = _files(tmp_path)
    validated = []

    def validator(profile, key):
        validated.append((profile["id"], key))
        return ["new-upstream"]

    result = onboarding.apply_profile(
        _profile(), config_path=config, model_info_path=model_info,
        model_info_source_path=source, api_key="new-secret",
        upstream_validator=validator,
        confirmed_retirements={"old-model"},
    )
    assert validated == [("new-provider-model", "new-secret")]
    assert result["retired_models"] == ["old-model"]

    cfg = yaml.safe_load(config.read_text())
    block = cfg["providers"]["new-provider"]
    assert "api_key" not in block
    secret = Path(block["api_key_file"])
    assert secret.read_text().strip() == "new-secret"
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    assert "old-model" not in cfg.get("model_overrides", {})

    live = json.loads(model_info.read_text())
    assert [row["name"] for row in live["llm"]] == ["new-model"]
    assert json.loads(source.read_text())["llm"] == live["llm"]


def test_apply_profile_is_idempotent(tmp_path):
    config, model_info, source = _files(tmp_path)
    kwargs = dict(
        config_path=config, model_info_path=model_info,
        model_info_source_path=source, api_key="secret", check_upstream=False,
        confirmed_retirements={"old-model"},
    )
    onboarding.apply_profile(_profile(), **kwargs)
    result = onboarding.apply_profile(_profile(), **kwargs)
    assert result["already_retired"] == ["old-model"]
    assert [row["name"] for row in json.loads(model_info.read_text())["llm"]] == ["new-model"]


def test_apply_profile_can_update_metadata_after_retirement(tmp_path):
    config, model_info, source = _files(tmp_path)
    kwargs = dict(
        config_path=config, model_info_path=model_info,
        model_info_source_path=source, api_key="secret", check_upstream=False,
        confirmed_retirements={"old-model"},
    )
    onboarding.apply_profile(_profile(), **kwargs)
    updated = _profile()
    updated["models"][0]["pricing"] = {"input": 3.0, "output": 15.0}
    result = onboarding.apply_profile(updated, confirmed_replacements={"new-model"}, **kwargs)
    assert result["already_retired"] == ["old-model"]
    row = json.loads(model_info.read_text())["llm"][0]
    assert row["pricing"] == {"input": 3.0, "output": 15.0}


def test_apply_profile_requires_expected_retirement_on_first_run(tmp_path):
    config, model_info, source = _files(tmp_path)
    model_info.write_text(json.dumps({"llm": []}))
    with pytest.raises(onboarding.OnboardingError, match="expected retired"):
        onboarding.apply_profile(
            _profile(), config_path=config, model_info_path=model_info,
            model_info_source_path=source, dry_run=True,
        )


def test_existing_replacement_name_does_not_fake_retirement_idempotency(tmp_path):
    config, model_info, source = _files(tmp_path)
    profile = _profile()
    model_info.write_text(json.dumps({"llm": [{
        "name": "new-model",
        "provider": "different-provider",
        "provider_model_id": "different-upstream",
    }]}))
    with pytest.raises(onboarding.OnboardingError, match="expected retired"):
        onboarding.apply_profile(
            profile,
            config_path=config,
            model_info_path=model_info,
            model_info_source_path=source,
            dry_run=True,
        )


def test_existing_secret_is_not_reused_for_changed_base_url(tmp_path):
    config, model_info, source = _files(tmp_path)
    profile = _profile()
    profile["provider"]["id"] = "old-provider"
    profile["provider"]["base_url"] = "https://different.example/v1"
    profile["models"][0]["provider"] = "old-provider"
    with pytest.raises(onboarding.OnboardingError, match="API key is required"):
        onboarding.apply_profile(
            profile,
            config_path=config,
            model_info_path=model_info,
            model_info_source_path=source,
            check_upstream=False,
            confirmed_retirements={"old-model"},
        )


def test_direct_apply_requires_exact_retirement_confirmation(tmp_path):
    config, model_info, source = _files(tmp_path)
    with pytest.raises(onboarding.OnboardingError, match="retirements require exact confirmation"):
        onboarding.apply_profile(
            _profile(),
            config_path=config,
            model_info_path=model_info,
            model_info_source_path=source,
            api_key="secret",
            check_upstream=False,
        )


def test_apply_profile_rolls_back_all_written_files(tmp_path, monkeypatch):
    config, model_info, source = _files(tmp_path)
    before = {path: path.read_bytes() for path in (config, model_info, source)}
    real_write = onboarding._atomic_write
    calls = 0

    def fail_third(path, content, mode):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected write failure")
        real_write(path, content, mode)

    monkeypatch.setattr(onboarding, "_atomic_write", fail_third)
    with pytest.raises(OSError, match="injected"):
        onboarding.apply_profile(
            _profile(), config_path=config, model_info_path=model_info,
            model_info_source_path=source, api_key="secret", check_upstream=False,
            confirmed_retirements={"old-model"},
        )
    assert before == {path: path.read_bytes() for path in (config, model_info, source)}
    assert not (tmp_path / "secrets" / "new-provider.api-key").exists()


def test_apply_profile_rolls_back_when_post_apply_verification_fails(tmp_path):
    config, model_info, source = _files(tmp_path)
    before = {path: path.read_bytes() for path in (config, model_info, source)}
    rollback_calls = []

    def fail_verification():
        raise RuntimeError("health failed")

    with pytest.raises(onboarding.OnboardingError, match="health failed"):
        onboarding.apply_profile(
            _profile(), config_path=config, model_info_path=model_info,
            model_info_source_path=source, api_key="secret", check_upstream=False,
            post_apply=fail_verification,
            post_rollback=lambda: rollback_calls.append(True),
            confirmed_retirements={"old-model"},
        )
    assert rollback_calls == [True]
    assert before == {path: path.read_bytes() for path in (config, model_info, source)}
    assert not (tmp_path / "secrets" / "new-provider.api-key").exists()


def test_profile_rejects_unsafe_provider_and_duplicate_models(tmp_path):
    profile = _profile()
    profile["provider"]["id"] = "../../escape"
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(onboarding.OnboardingError, match="provider id"):
        onboarding.load_profile(path)

    profile = _profile()
    profile["models"].append(dict(profile["models"][0]))
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(onboarding.OnboardingError, match="duplicate profile model"):
        onboarding.load_profile(path)


def test_profile_rejects_unsafe_secret_name(tmp_path):
    profile = _profile()
    profile["provider"]["secret_name"] = "../escape.key"
    path = tmp_path / "unsafe-secret.yaml"
    path.write_text(yaml.safe_dump(profile))
    loaded = onboarding.load_profile(path)
    config, model_info, source = _files(tmp_path)
    with pytest.raises(onboarding.OnboardingError, match="safe filename"):
        onboarding.apply_profile(
            loaded, config_path=config, model_info_path=model_info,
            model_info_source_path=source, dry_run=True,
        )


def test_apply_profile_rejects_overlapping_targets(tmp_path, monkeypatch):
    config, model_info, source = _files(tmp_path)
    profile = _profile()
    profile["provider"]["secret_name"] = config.name
    monkeypatch.setenv("MODEL_GATEWAY_SECRET_DIR", str(config.parent))
    with pytest.raises(onboarding.OnboardingError, match="must be distinct"):
        onboarding.apply_profile(
            profile, config_path=config, model_info_path=model_info,
            model_info_source_path=source, dry_run=True,
        )


def test_provider_secret_target_cannot_be_owned_by_another_provider(tmp_path):
    config, model_info, source = _files(tmp_path)
    target = tmp_path / "secrets" / "new-provider.api-key"
    target.parent.mkdir()
    target.write_text("old-secret\n")
    target.chmod(0o600)
    alias = tmp_path / "other-provider.key"
    alias.symlink_to(target)
    for raw_path in (str(target), "secrets/new-provider.api-key", str(alias)):
        config.write_text(yaml.safe_dump({
            "providers": {
                "other-provider": {
                    "base_url": "https://other.example/v1",
                    "api_key_file": raw_path,
                },
            },
        }))
        with pytest.raises(onboarding.OnboardingError, match="already owned"):
            onboarding.apply_profile(
                _profile(),
                config_path=config,
                model_info_path=model_info,
                model_info_source_path=source,
                dry_run=True,
            )
        assert target.read_text() == "old-secret\n"


def test_apply_profile_rolls_back_when_directory_fsync_fails_after_replace(tmp_path, monkeypatch):
    config, model_info, source = _files(tmp_path)
    before = {path: path.read_bytes() for path in (config, model_info, source)}
    real_fsync = onboarding.os.fsync
    calls = 0

    def fail_first_directory_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        return real_fsync(fd)

    monkeypatch.setattr(onboarding.os, "fsync", fail_first_directory_fsync)
    with pytest.raises(OSError, match="directory fsync failed"):
        onboarding.apply_profile(
            _profile(), config_path=config, model_info_path=model_info,
            model_info_source_path=source, api_key="secret", check_upstream=False,
            confirmed_retirements={"old-model"},
        )
    assert before == {path: path.read_bytes() for path in (config, model_info, source)}
    assert not (tmp_path / "secrets" / "new-provider.api-key").exists()


def test_apply_profile_reports_incomplete_rollback(tmp_path, monkeypatch):
    config, model_info, source = _files(tmp_path)
    real_restore = onboarding._restore
    failed = False

    def fail_one_restore(path, snapshot):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("restore failed")
        real_restore(path, snapshot)

    monkeypatch.setattr(onboarding, "_restore", fail_one_restore)
    with pytest.raises(onboarding.OnboardingError, match="rollback incomplete"):
        onboarding.apply_profile(
            _profile(), config_path=config, model_info_path=model_info,
            model_info_source_path=source, api_key="secret", check_upstream=False,
            post_apply=lambda: (_ for _ in ()).throw(RuntimeError("verify failed")),
            confirmed_retirements={"old-model"},
        )
    assert failed is True


def test_validate_upstream_rejects_missing_model(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "another-model"}]}

    monkeypatch.setattr(onboarding.httpx, "get", lambda *args, **kwargs: Response())
    with pytest.raises(onboarding.OnboardingError, match="does not advertise"):
        onboarding.validate_upstream_models(_profile(), "secret")
