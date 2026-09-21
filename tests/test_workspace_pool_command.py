"""Existing-member pool management: no real services, tokens or inference."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import workspace as w
from src import providers


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # Never let a developer's credentials/routing overrides enter fake probes
    # or assertion output. Explicit override tests must use synthetic values.
    for key in list(os.environ):
        if key.startswith(("MODEL_GATEWAY_PROVIDER_", "DATABRICKS_")):
            monkeypatch.delenv(key)
    path = tmp_path / "config.yaml"
    catalog = tmp_path / "models.json"
    catalog.write_text('{"llm": []}')
    config = {
        "auth": {"admin_keys": ["admin-sentinel"]},
        "providers": {
            name: {"base_url": f"https://{name}.example.com", "api_key": f"key-{name}",
                   "protocol": "openai", "endpoint_style": "invocations",
                   "quirks": ["no_stream_options"], "custom_setting": "preserve"}
            for name in ("first", "second", "third")
        },
        "pools": {"fable-pool": ["first", "second"], "other": ["third"]},
        "models": [{"name": name, "provider_model_id": name, "pool": "fable-pool"}
                   for name in ("fable-5", "fable-5-1")],
    }
    path.write_text(yaml.safe_dump(config))
    args = argparse.Namespace(config=path, pool="fable-pool", name="third", dry_run=False)
    calls = {"activate": 0, "smoke": [], "endpoints": 0}
    monkeypatch.setattr(w, "ADMIN_KEY", "")
    monkeypatch.setattr(w, "RESTART_BIN", "/test/model-gateway")
    monkeypatch.setattr(w, "GATEWAY_URL", "http://gateway.invalid")

    def get_json(url, key):
        assert key == "admin-sentinel"
        if url.endswith("/status"):
            return {"status": "ok", "config_path": str(path), "model_info_path": str(catalog),
                    "writes_enabled": False}
        assert url.endswith("/workspace-pools")
        current = yaml.safe_load(path.read_text())
        return {"pools": [{"id": "fable-pool", "members": [
            {"id": m, "ready": True} for m in current["pools"]["fable-pool"]]}]}

    def smoke(url, body, headers, deadline):
        calls["smoke"].append((url, body, headers))
        return {"choices": [{"message": {"content": "OK"}}]}

    def endpoints(host, token):
        calls["endpoints"] += 1
        assert host == "https://third.example.com"
        assert token == "key-third"
        return {"fable-5", "fable-5-1"}

    def activate():
        calls["activate"] += 1

    monkeypatch.setattr(w, "_get_json", get_json)
    monkeypatch.setattr(w, "_smoke_request", smoke)
    monkeypatch.setattr(w, "probe_endpoints", endpoints)
    monkeypatch.setattr(w, "activate_gateway", activate)
    return args, config, catalog, calls


def test_add_existing_member_preserves_settings_and_smokes_actual_models(setup):
    args, before, _, calls = setup
    w.cmd_pool_add_member(args)
    expected = copy.deepcopy(before)
    expected["pools"]["fable-pool"].append("third")
    assert yaml.safe_load(args.config.read_text()) == expected
    assert calls["activate"] == 1
    assert [body["model"] for _, body, _ in calls["smoke"]] == ["fable-5", "fable-5-1"]
    assert all(url.endswith(f"/{body['model']}/invocations") for url, body, _ in calls["smoke"])
    assert all(headers["Authorization"] == "Bearer key-third" for _, _, headers in calls["smoke"])
    assert len(list(args.config.parent.glob("*.bak-*"))) == 1
    assert args.config.stat().st_mode & 0o777 == 0o600


def test_dry_run_does_not_write_or_activate(setup):
    args, _, _, calls = setup
    original = args.config.read_bytes()
    args.dry_run = True
    w.cmd_pool_add_member(args)
    assert args.config.read_bytes() == original
    assert calls["activate"] == 0
    assert len(calls["smoke"]) == 2
    assert not list(args.config.parent.glob("*.bak-*"))


def test_duplicate_is_noop_without_auth_or_restart(setup, monkeypatch):
    args, _, _, calls = setup
    args.name = "first"
    monkeypatch.setattr(w, "_get_json", lambda *a: pytest.fail("network on no-op"))
    monkeypatch.setattr(w, "RESTART_BIN", "")
    original = args.config.read_bytes()
    w.cmd_pool_add_member(args)
    assert args.config.read_bytes() == original
    assert calls == {"activate": 0, "smoke": [], "endpoints": 0}


@pytest.mark.parametrize("change,match", [
    ("unknown_pool", "unknown or empty pool"),
    ("unknown_member", "unknown workspace"),
    ("disabled", "disabled"),
    ("empty_pool", "unknown or empty pool"),
    ("no_models", "no enabled models"),
    ("no_admin", "admin read key"),
    ("reload_disabled", "admin writes are disabled"),
])
def test_preflight_errors_never_write(setup, monkeypatch, change, match):
    args, config, _, calls = setup
    if change == "unknown_pool":
        args.pool = "typo"
    elif change == "unknown_member":
        args.name = "typo"
    elif change == "disabled":
        config["providers"]["third"]["enabled"] = False
    elif change == "empty_pool":
        config["pools"]["fable-pool"] = []
    elif change == "no_models":
        config["model_overrides"] = {m["name"]: {"enabled": False} for m in config["models"]}
    elif change == "no_admin":
        config["auth"] = {}
    elif change == "reload_disabled":
        monkeypatch.setattr(w, "RESTART_BIN", "")
        monkeypatch.setattr(w, "ADMIN_KEY", "admin-sentinel")
    args.config.write_text(yaml.safe_dump(config))
    original = args.config.read_bytes()
    with pytest.raises(SystemExit, match=match):
        w.cmd_pool_add_member(args)
    assert args.config.read_bytes() == original
    assert calls["activate"] == 0
    assert not list(args.config.parent.glob("*.bak-*"))


def test_missing_fable_fails_before_smoke(setup, monkeypatch):
    args, _, _, calls = setup
    original = args.config.read_bytes()
    monkeypatch.setattr(w, "probe_endpoints", lambda *a: {"sonnet"})
    with pytest.raises(SystemExit, match="does not serve pool models: fable-5, fable-5-1"):
        w.cmd_pool_add_member(args)
    assert args.config.read_bytes() == original
    assert calls["smoke"] == []
    assert calls["activate"] == 0


@pytest.mark.parametrize("failure", ["404", "invalid_success"])
def test_failed_model_probe_preserves_config(setup, monkeypatch, failure):
    args, _, _, calls = setup
    original = args.config.read_bytes()

    def smoke(*a):
        if failure == "404":
            raise w.RoutePreflightError("path", "HTTP 404")
        return {"error": "model unavailable"}

    monkeypatch.setattr(w, "_smoke_request", smoke)
    with pytest.raises(SystemExit, match="smoke"):
        w.cmd_pool_add_member(args)
    assert args.config.read_bytes() == original
    assert calls["activate"] == 0


def test_live_verification_failure_rolls_back(setup, monkeypatch):
    args, _, _, calls = setup
    original = args.config.read_bytes()

    def fail(*a):
        raise RuntimeError("live order mismatch")

    monkeypatch.setattr(w, "_verify_live_pool", fail)
    with pytest.raises(SystemExit, match="previous configuration restored and verified"):
        w.cmd_pool_add_member(args)
    assert args.config.read_bytes() == original
    assert calls["activate"] == 2


def test_concurrent_edit_during_probes_is_not_overwritten(setup, monkeypatch):
    args, _, _, calls = setup
    concurrent = args.config.read_bytes() + b"# another writer\n"

    def smoke(*a):
        args.config.write_bytes(concurrent)
        return {"choices": [{}]}

    monkeypatch.setattr(w, "_smoke_request", smoke)
    with pytest.raises(SystemExit, match="config changed during validation"):
        w.cmd_pool_add_member(args)
    assert args.config.read_bytes() == concurrent
    assert calls["activate"] == 0
    assert not list(args.config.parent.glob("*.bak-*"))


def test_rollback_does_not_overwrite_concurrent_edit(setup, monkeypatch):
    args, _, _, _ = setup
    concurrent = b"# concurrent edit\nproviders: {}\n"

    def activate():
        args.config.write_bytes(concurrent)
        raise RuntimeError("activation failed")

    monkeypatch.setattr(w, "activate_gateway", activate)
    with pytest.raises(SystemExit, match="refusing to overwrite concurrent changes"):
        w.cmd_pool_add_member(args)
    assert args.config.read_bytes() == concurrent


def test_preview_uses_merged_catalog_and_restores_registry(setup, monkeypatch):
    args, config, catalog, _ = setup
    catalog.write_text(json.dumps({"llm": [{"name": "catalog-only", "pool": "fable-pool"}]}))
    config["pools"]["fable-pool"].append("third")
    cached_config, cached_models = {"original": True}, {"unchanged": {}}
    monkeypatch.setattr(providers, "_config", cached_config)
    monkeypatch.setattr(providers, "_models", cached_models)
    paths = providers.CONFIG_PATH, providers.MODEL_INFO_PATH
    routes = providers.preview_pool_member(config, args.config, catalog, args.pool, args.name)
    assert {r.provider_model_id for r in routes} == {"catalog-only", "fable-5", "fable-5-1"}
    assert providers._config is cached_config and providers._models is cached_models
    assert (providers.CONFIG_PATH, providers.MODEL_INFO_PATH) == paths


@pytest.mark.parametrize("failure", ["wire", "coverage"])
def test_preview_rejects_ineligible_routes_and_restores_registry(setup, monkeypatch, failure):
    args, config, catalog, _ = setup
    config["pools"]["fable-pool"].append("third")
    if failure == "wire":
        config["providers"]["third"]["protocol"] = "anthropic"
    else:
        config["providers"]["third"]["available_model_ids"] = ["sonnet"]
    saved = providers._config, providers._models, providers.CONFIG_PATH, providers.MODEL_INFO_PATH
    with pytest.raises(ValueError, match="incompatible|excluded"):
        providers.preview_pool_member(config, args.config, catalog, args.pool, args.name)
    assert (providers._config, providers._models, providers.CONFIG_PATH, providers.MODEL_INFO_PATH) == saved


@pytest.mark.parametrize("protocol,api_style,prefix,field", [
    ("openai", "", "/custom/openai", "choices"),
    ("anthropic", "", "/custom/anthropic", "content"),
    ("openai", "open_responses", "", "output"),
])
def test_smoke_respects_resolved_route(protocol, api_style, prefix, field, monkeypatch):
    calls = []
    route = providers.ProviderInfo(provider="third", base_url="https://third.example.com" + prefix,
                                   api_key="key-third", provider_model_id="fable", protocol=protocol,
                                   api_style=api_style)

    def smoke(url, body, headers, deadline):
        calls.append((url, body, headers))
        return {field: [{}]}

    monkeypatch.setattr(w, "_smoke_request", smoke)
    w._smoke_pool_routes([route], "refreshed")
    url, body, headers = calls[0]
    if api_style:
        assert url == route.base_url and body["input"] == "Reply OK"
    else:
        assert url == route.base_url + ("/messages" if protocol == "anthropic" else "/chat/completions")
        assert body["messages"]
    if protocol == "anthropic":
        assert headers["x-api-key"] == "refreshed" and "Authorization" not in headers
    else:
        assert headers["Authorization"] == "Bearer refreshed"
    assert ("anthropic-version" in headers) == (protocol == "anthropic")


def test_nested_cli_help_and_parse(monkeypatch):
    result = subprocess.run([sys.executable, str(Path(w.__file__)), "pool", "add-member", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--dry-run" in result.stdout
    seen = []
    monkeypatch.setattr(w, "cmd_pool_add_member", lambda args: seen.append(args))
    monkeypatch.setattr(sys, "argv", ["workspace", "--config", "/tmp/config", "pool", "add-member",
                                     "fable-pool", "third", "--dry-run"])
    w.main()
    assert seen[0].pool == "fable-pool" and seen[0].name == "third" and seen[0].dry_run


def test_oauth_probes_use_refreshed_token_without_replacing_provider(setup, monkeypatch):
    args, config, _, calls = setup
    config["providers"]["third"].update(auth_refresh="databricks-cli", auth_profile="existing-profile")
    args.config.write_text(yaml.safe_dump(config))
    auth = []
    monkeypatch.setattr(w, "ensure_auth", lambda host, profile, **kwargs: auth.append((host, profile)) or "fresh")
    monkeypatch.setattr(w, "probe_endpoints", lambda host, token: {"fable-5", "fable-5-1"} if token == "fresh" else set())
    w.cmd_pool_add_member(args)
    assert auth == [("https://third.example.com", "existing-profile")]
    assert all(headers["Authorization"] == "Bearer fresh" for _, _, headers in calls["smoke"])
    assert yaml.safe_load(args.config.read_text())["providers"] == config["providers"]


def test_file_credentials_resolve_relative_to_selected_config(setup):
    args, config, catalog, _ = setup
    secret = args.config.parent / "secret"
    secret.write_text("file-secret")
    secret.chmod(0o600)
    del config["providers"]["third"]["api_key"]
    config["providers"]["third"]["api_key_file"] = "secret"
    config["pools"]["fable-pool"].append("third")
    routes = providers.preview_pool_member(config, args.config, catalog, args.pool, args.name)
    assert all(route.api_key == "file-secret" for route in routes)


def test_wrong_live_config_aborts_before_probes(setup, monkeypatch):
    args, _, catalog, calls = setup
    original = args.config.read_bytes()
    monkeypatch.setattr(w, "_get_json", lambda *a: {
        "status": "ok", "config_path": "/wrong/config.yaml", "model_info_path": str(catalog)})
    with pytest.raises(SystemExit, match="not serving the selected config"):
        w.cmd_pool_add_member(args)
    assert args.config.read_bytes() == original
    assert calls == {"activate": 0, "smoke": [], "endpoints": 0}


@pytest.mark.parametrize("live_members,ready", [(["first", "second"], True),
                                              (["third", "first", "second"], True),
                                              (["first", "second", "third"], False)])
def test_live_pool_verifier_rejects_wrong_order_or_unready_member(setup, monkeypatch, live_members, ready):
    args, _, _, _ = setup
    get_json = w._get_json
    monkeypatch.setattr(w, "_get_json", lambda url, key: (
        {"pools": [{"id": args.pool, "members": [{"id": m, "ready": ready} for m in live_members]}]}
        if url.endswith("/workspace-pools") else get_json(url, key)))
    with pytest.raises(RuntimeError, match="order/readiness"):
        w._verify_live_pool(args.config, "admin-sentinel", args.pool, ["first", "second", "third"])


def test_sys_exit_from_post_activation_identity_check_rolls_back(setup, monkeypatch):
    args, _, _, calls = setup
    original = args.config.read_bytes()
    monkeypatch.setattr(w, "_verify_live_pool", lambda *a: (_ for _ in ()).throw(SystemExit("identity mismatch")))
    with pytest.raises(SystemExit, match="previous configuration restored and verified"):
        w.cmd_pool_add_member(args)
    assert args.config.read_bytes() == original
    assert calls["activate"] == 2


def test_disabled_model_is_not_probed(setup):
    args, config, _, calls = setup
    config["model_overrides"] = {"fable-5": {"enabled": False}}
    args.config.write_text(yaml.safe_dump(config))
    w.cmd_pool_add_member(args)
    assert [body["model"] for _, body, _ in calls["smoke"]] == ["fable-5-1"]


def test_workspaces_alias_and_config_symlink_are_preserved(setup):
    args, config, _, _ = setup
    config["workspaces"] = config.pop("providers")
    args.config.write_text(yaml.safe_dump(config))
    target = args.config
    args.config = target.with_name("config-link.yaml")
    args.config.symlink_to(target)
    w.cmd_pool_add_member(args)
    assert args.config.is_symlink()
    result = yaml.safe_load(target.read_text())
    assert result["workspaces"] == config["workspaces"]
    assert "providers" not in result
    assert result["pools"]["fable-pool"] == ["first", "second", "third"]


def test_mixed_provider_sections_use_runtime_precedence(setup, monkeypatch):
    args, config, _, _ = setup
    config["workspaces"] = {"third": {"base_url": "https://wrong.example.com", "api_key": "wrong"}}
    args.config.write_text(yaml.safe_dump(config))
    w.cmd_pool_add_member(args)  # fixture requires third.example.com and key-third
    assert yaml.safe_load(args.config.read_text())["workspaces"] == config["workspaces"]


def test_canonical_alias_duplicate_is_noop(setup, monkeypatch):
    args, config, _, calls = setup
    config["providers"]["databricks"] = config["providers"].pop("third")
    config["pools"]["fable-pool"].append("dbx")
    args.name = "databricks"
    args.config.write_text(yaml.safe_dump(config))
    original = args.config.read_bytes()
    w.cmd_pool_add_member(args)
    assert args.config.read_bytes() == original and calls["activate"] == 0


def test_aliases_in_live_verification_are_canonicalized(setup, monkeypatch):
    args, _, _, _ = setup
    get_json = w._get_json
    monkeypatch.setattr(w, "_get_json", lambda url, key: (
        {"pools": [{"id": args.pool, "members": [{"id": name, "ready": True}
                                                for name in ["databricks", "third"]]}]}
        if url.endswith("/workspace-pools") else get_json(url, key)))
    w._verify_live_pool(args.config, "admin-sentinel", args.pool, ["dbx", "third"])


def test_appending_alias_stores_canonical_member(setup, monkeypatch):
    args, config, _, _ = setup
    config["providers"]["databricks"] = config["providers"].pop("third")
    args.name = "dbx"
    args.config.write_text(yaml.safe_dump(config))
    w.cmd_pool_add_member(args)
    assert yaml.safe_load(args.config.read_text())["pools"]["fable-pool"][-1] == "databricks"


def test_profileless_auth_uses_host_and_disables_browser_login(monkeypatch):
    calls = []

    def cli(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="expired")

    monkeypatch.setattr(w, "_databricks", cli)
    monkeypatch.setattr(w.subprocess, "run", lambda *a, **k: pytest.fail("browser login disabled"))
    with pytest.raises(SystemExit, match="auth_login is disabled"):
        w.ensure_auth("https://third.example.com", "", allow_login=False)
    assert calls == [("auth", "token", "--host", "https://third.example.com")]


def test_command_preserves_auth_login_policy_and_host_selection(setup, monkeypatch):
    args, config, _, _ = setup
    config["providers"]["third"].update(auth_refresh="databricks-cli", auth_login=False)
    args.config.write_text(yaml.safe_dump(config))
    calls = []
    monkeypatch.setattr(w, "ensure_auth", lambda host, profile, **kw: calls.append((host, profile, kw)) or "key-third")
    w.cmd_pool_add_member(args)
    assert calls == [("https://third.example.com", "", {"allow_login": False})]


@pytest.mark.parametrize("protocol,quirks", [("anthropic", {"anthropic_bearer_auth"}),
                                             ("openai", {"use_max_completion_tokens", "force_reasoning_effort_max"})])
def test_smoke_normalization_matches_runtime(protocol, quirks, monkeypatch):
    from src.server import _apply_openai_request_quirks, _forward_headers
    from starlette.requests import Request
    route = providers.ProviderInfo(provider="third", base_url="https://third.example.com", api_key="token",
                                   provider_model_id="fable", protocol=protocol, quirks=frozenset(quirks))
    calls = []
    monkeypatch.setattr(w, "_smoke_request", lambda url, body, headers, deadline:
                        calls.append((body, headers)) or {"content" if protocol == "anthropic" else "choices": [{}]})
    monkeypatch.setattr("src.server.provider_quirks", lambda provider: quirks)
    w._smoke_pool_routes([route], "token", auth_quirks=quirks)
    body, headers = calls[0]
    request = Request({"type": "http", "headers": []})
    request.state.api_key = "token"
    assert headers == _forward_headers(request, protocol=protocol, provider="third")
    if protocol == "openai":
        expected = {"model": "fable", "max_tokens": 16, "messages": [{"role": "user", "content": "Reply OK"}]}
        assert _apply_openai_request_quirks(expected, route) is None
        assert body == expected


def test_model_only_auth_quirk_does_not_override_provider_auth(monkeypatch):
    from src.server import _forward_headers
    from starlette.requests import Request
    route = providers.ProviderInfo(provider="third", base_url="https://third.example.com", api_key="synthetic",
                                   provider_model_id="fable", protocol="anthropic",
                                   quirks=frozenset({"anthropic_bearer_auth"}))
    calls = []
    monkeypatch.setattr(w, "_smoke_request", lambda url, body, headers, deadline:
                        calls.append(headers) or {"content": [{}]})
    monkeypatch.setattr("src.server.provider_quirks", lambda provider: frozenset())
    w._smoke_pool_routes([route], "synthetic", auth_quirks=())
    request = Request({"type": "http", "headers": []})
    request.state.api_key = "synthetic"
    assert calls == [_forward_headers(request, protocol="anthropic", provider="third")]
    assert calls[0]["x-api-key"] == "synthetic"


def test_shell_credentials_cannot_override_config_probe(setup, monkeypatch, capsys):
    args, _, _, calls = setup
    original = args.config.read_bytes()
    monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_THIRD_API_KEY", "synthetic-secret-never-log")
    with pytest.raises(SystemExit, match="shell routing/credential overrides") as error:
        w.cmd_pool_add_member(args)
    assert args.config.read_bytes() == original
    assert calls == {"activate": 0, "smoke": [], "endpoints": 0}
    assert "synthetic-secret-never-log" not in str(error.value) + capsys.readouterr().out
