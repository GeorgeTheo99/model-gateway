"""`workspace pool create`: no real services, tokens or inference."""
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

REAL_VERIFY_LIVE_POOL = w._verify_live_pool


def _provider(name: str, **extra) -> dict:
    return {"base_url": f"https://{name}.example.com", "api_key": f"key-{name}",
            "protocol": "openai", "endpoint_style": "invocations",
            "quirks": ["no_stream_options"], "custom_setting": "preserve", **extra}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith(("MODEL_GATEWAY_PROVIDER_", "DATABRICKS_")):
            monkeypatch.delenv(key)
    path = tmp_path / "config.yaml"
    catalog = tmp_path / "models.json"
    catalog.write_text(json.dumps({"llm": [
        {"name": "catalog-model", "alias": "cat", "provider": "first",
         "provider_model_id": "databricks-catalog-model", "context": 1000, "vision": True},
    ]}))
    config = {
        "auth": {"admin_keys": ["admin-sentinel"]},
        "providers": {name: _provider(name) for name in ("first", "second", "third")},
        "pools": {"other": ["third", "first"]},
        "models": [
            {"name": "opus", "alias": "op", "provider": "first",
             "provider_model_id": "databricks-opus", "protocol": "anthropic",
             "quirks": ["anthropic_tool_result_blocks"], "thinking": "always"},
            {"name": "untouched", "provider": "third", "pool": "other", "provider_model_id": "u"},
        ],
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    args = argparse.Namespace(config=path, pool="opus-pool", members="first,second,third",
                              model=["op"], dry_run=False)
    calls = {"activate": 0, "smoke": [], "probes": [], "verified": []}
    monkeypatch.setattr(w, "ADMIN_KEY", "")
    monkeypatch.setattr(w, "RESTART_BIN", "/test/model-gateway")
    monkeypatch.setattr(w, "GATEWAY_URL", "http://gateway.invalid")

    def get_json(url, key):
        assert key == "admin-sentinel"
        assert url.endswith("/status")
        return {"status": "ok", "config_path": str(path), "model_info_path": str(catalog),
                "writes_enabled": False}

    def smoke(url, body, headers, deadline):
        calls["smoke"].append((url, body["model"], headers["Authorization"]))
        return {"choices": [{"message": {"content": "OK"}}]}

    served = {"databricks-opus", "databricks-catalog-model"}
    monkeypatch.setattr(w, "_get_json", get_json)
    monkeypatch.setattr(w, "_smoke_request", smoke)
    monkeypatch.setattr(w, "probe_endpoints", lambda host, token: calls["probes"].append(host) or served)
    monkeypatch.setattr(w, "probe_model_services", lambda host, token: set())
    monkeypatch.setattr(w, "activate_gateway", lambda: calls.__setitem__("activate", calls["activate"] + 1))
    monkeypatch.setattr(w, "_verify_live_pool",
                        lambda path, key, pool, members, models=(): calls["verified"].append((pool, members, models)))
    return args, config, catalog, calls


def test_create_binds_model_preserving_everything_else(setup):
    args, before, _, calls = setup
    w.cmd_pool_create(args)
    after = yaml.safe_load(args.config.read_text())
    expected = copy.deepcopy(before)
    expected["pools"]["opus-pool"] = ["first", "second", "third"]
    expected["models"][0].update(provider="first", pool="opus-pool")
    assert after == expected
    assert calls["activate"] == 1
    assert calls["verified"] == [("opus-pool", ["first", "second", "third"], ["opus"])]
    # Every member probed through its own resolved invocation route and credential.
    assert calls["smoke"] == [
        (f"https://{m}.example.com/serving-endpoints/databricks-opus/invocations", "databricks-opus",
         f"Bearer key-{m}") for m in ("first", "second", "third")]
    assert args.config.stat().st_mode & 0o777 == 0o600
    assert len(list(args.config.parent.glob("*.bak-*"))) == 1


def test_catalog_only_model_gets_minimal_overlay(setup):
    args, _, _, calls = setup
    args.model = ["cat"]
    w.cmd_pool_create(args)
    entry = yaml.safe_load(args.config.read_text())["models"][-1]
    assert entry == {"name": "catalog-model", "provider": "first", "pool": "opus-pool"}
    assert {model for _, model, _ in calls["smoke"]} == {"databricks-catalog-model"}


def test_dry_run_probes_but_never_writes(setup):
    args, _, _, calls = setup
    args.dry_run = True
    original = args.config.read_bytes()
    w.cmd_pool_create(args)
    assert args.config.read_bytes() == original
    assert calls["activate"] == 0 and len(calls["smoke"]) == 3
    assert not list(args.config.parent.glob("*.bak-*"))


def test_rerun_is_noop_without_probes_or_restart(setup):
    args, _, _, calls = setup
    w.cmd_pool_create(args)
    applied = args.config.read_bytes()
    calls.update(activate=0, smoke=[], probes=[])
    w.cmd_pool_create(args)
    assert args.config.read_bytes() == applied
    assert calls["activate"] == 0 and calls["smoke"] == [] and calls["probes"] == []


def test_unity_model_services_satisfy_coverage(setup, monkeypatch):
    args, _, _, calls = setup
    monkeypatch.setattr(w, "probe_endpoints", lambda host, token: set())
    monkeypatch.setattr(w, "probe_model_services", lambda host, token: {"databricks-opus"})
    w.cmd_pool_create(args)
    assert calls["activate"] == 1


@pytest.mark.parametrize("change,match", [
    ("one_member", "at least two"),
    ("duplicate", "must not repeat"),
    ("unknown_member", "unknown workspace"),
    ("disabled", "disabled"),
    ("unknown_model", "unknown model"),
    ("other_pool", "already uses pool 'other'"),
    ("existing_pool", "already exists with different members"),
    ("wire_change", "speaks anthropic/default but the current route speaks openai"),
    ("missing_model", "does not serve pool models: databricks-opus"),
    ("no_admin", "admin read key"),
    ("reload_disabled", "admin writes are disabled"),
    ("shell_override", "shell routing/credential overrides"),
])
def test_rejections_never_write(setup, monkeypatch, change, match):
    args, config, _, calls = setup
    if change == "one_member":
        args.members = "first"
    elif change == "duplicate":
        args.members = "first,first"
    elif change == "unknown_member":
        args.members = "first,typo"
    elif change == "disabled":
        config["providers"]["second"]["enabled"] = False
    elif change == "unknown_model":
        args.model = ["nope"]
    elif change == "other_pool":
        args.model = ["untouched"]
    elif change == "existing_pool":
        config["pools"]["opus-pool"] = ["second", "first"]
    elif change == "wire_change":
        # A native-Anthropic AI-gateway route would change the wire format active
        # sessions were created with (the current route is an invocations route).
        config["providers"]["second"] = {
            "base_url": "https://second.example.com/ai-gateway", "api_key": "key-second",
            "protocol": "openai", "path_prefixes": {"anthropic": "anthropic/v1", "openai": "mlflow/v1"}}
    elif change == "missing_model":
        monkeypatch.setattr(w, "probe_endpoints", lambda host, token: set())
    elif change == "no_admin":
        config["auth"] = {}
    elif change == "reload_disabled":
        monkeypatch.setattr(w, "RESTART_BIN", "")
        monkeypatch.setattr(w, "ADMIN_KEY", "admin-sentinel")
    elif change == "shell_override":
        monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_SECOND_API_KEY", "synthetic")
    args.config.write_text(yaml.safe_dump(config, sort_keys=False))
    original = args.config.read_bytes()
    with pytest.raises(SystemExit, match=match):
        w.cmd_pool_create(args)
    assert args.config.read_bytes() == original
    assert calls["activate"] == 0
    assert not list(args.config.parent.glob("*.bak-*"))


def test_failed_smoke_on_backup_preserves_config(setup, monkeypatch):
    args, _, _, calls = setup
    original = args.config.read_bytes()

    def smoke(url, body, headers, deadline):
        if "third" in url:
            raise w.RoutePreflightError("rate_limited", "HTTP 429 persisted for 120s")
        return {"choices": [{}]}

    monkeypatch.setattr(w, "_smoke_request", smoke)
    with pytest.raises(SystemExit, match="rate_limited"):
        w.cmd_pool_create(args)
    assert args.config.read_bytes() == original and calls["activate"] == 0


def test_live_verification_failure_rolls_back(setup, monkeypatch):
    args, _, _, calls = setup
    original = args.config.read_bytes()
    monkeypatch.setattr(w, "_verify_live_pool", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("not serving")))
    with pytest.raises(SystemExit, match="previous configuration restored and verified"):
        w.cmd_pool_create(args)
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
        w.cmd_pool_create(args)
    assert args.config.read_bytes() == concurrent and calls["activate"] == 0


def test_live_pool_verifier_requires_bound_models(setup, monkeypatch):
    args, _, _, _ = setup
    monkeypatch.setattr(w, "_get_json", lambda url, key: (
        {"status": "ok", "config_path": str(args.config), "model_info_path": "/m.json"}
        if url.endswith("/status") else
        {"pools": [{"id": "opus-pool", "models": [], "members": [{"id": "first", "ready": True},
                                                                  {"id": "second", "ready": True}]}]}))
    with pytest.raises(RuntimeError, match="not serving model"):
        REAL_VERIFY_LIVE_POOL(args.config, "k", "opus-pool", ["first", "second"], ["opus"])


def test_probe_model_services_paginates_and_strips_prefix(monkeypatch):
    pages = {"": {"model_services": [{"name": "model-services/system.ai.a"}], "next_page_token": "p2"},
             "p2": {"model_services": [{"name": "model-services/system.ai.b"}]}}
    seen = []

    def get_json(url, token):
        seen.append(url)
        return pages["p2" if "page_token=p2" in url else ""]

    monkeypatch.setattr(w, "_get_json", get_json)
    assert w.probe_model_services("https://ws.example.com", "t") == {"system.ai.a", "system.ai.b"}
    assert all(u.startswith("https://ws.example.com/api/2.1/unity-catalog/model-services?parent=schemas%2Fsystem.ai")
               for u in seen)


def test_probe_model_services_tolerates_unsupported_workspace(monkeypatch):
    def fail(url, token):
        raise w.urllib.error.HTTPError(url, 404, "not found", {}, None)

    monkeypatch.setattr(w, "_get_json", fail)
    assert w.probe_model_services("https://ws.example.com", "t") == set()


def test_cli_help_and_dispatch(monkeypatch):
    result = subprocess.run([sys.executable, str(Path(w.__file__)), "pool", "create", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "--members" in result.stdout and "--model" in result.stdout
    seen = []
    monkeypatch.setattr(w, "cmd_pool_create", lambda args: seen.append(args))
    monkeypatch.setattr(sys, "argv", ["workspace", "--config", "/tmp/c", "pool", "create", "p",
                                     "--members", "a,b", "--model", "x", "--model", "y", "--dry-run"])
    w.main()
    assert seen[0].pool == "p" and seen[0].members == "a,b" and seen[0].model == ["x", "y"] and seen[0].dry_run


@pytest.mark.parametrize("selector", ["cat", "databricks-catalog-model", "catalog-model"])
def test_rerun_of_catalog_only_model_is_noop_for_any_identifier(setup, selector):
    args, _, _, calls = setup
    args.model = ["cat"]
    w.cmd_pool_create(args)
    applied = args.config.read_bytes()
    calls.update(activate=0, smoke=[], probes=[])
    args.model = [selector]
    w.cmd_pool_create(args)
    assert args.config.read_bytes() == applied
    assert calls["activate"] == 0 and calls["smoke"] == []
    names = [m["name"] for m in yaml.safe_load(applied)["models"]]
    assert names.count("catalog-model") == 1


def test_existing_overlay_entry_is_updated_not_duplicated(setup):
    args, config, _, _ = setup
    config["models"].append({"name": "catalog-model", "thinking": "always"})
    args.config.write_text(yaml.safe_dump(config, sort_keys=False))
    args.model = ["cat"]
    w.cmd_pool_create(args)
    rows = [m for m in yaml.safe_load(args.config.read_text())["models"] if m["name"] == "catalog-model"]
    assert rows == [{"name": "catalog-model", "thinking": "always", "provider": "first", "pool": "opus-pool"}]


def test_second_model_joins_applied_pool(setup):
    args, _, _, calls = setup
    w.cmd_pool_create(args)
    calls.update(activate=0, smoke=[])
    args.model = ["op", "cat"]
    w.cmd_pool_create(args)
    after = yaml.safe_load(args.config.read_text())
    assert after["pools"]["opus-pool"] == ["first", "second", "third"]
    assert {m["name"] for m in after["models"] if m.get("pool") == "opus-pool"} == {"opus", "catalog-model"}
    assert calls["activate"] == 1


def test_duplicate_model_selectors_are_deduplicated(setup):
    args, _, _, calls = setup
    args.model = ["op", "opus", "databricks-opus"]
    w.cmd_pool_create(args)
    assert calls["verified"] == [("opus-pool", ["first", "second", "third"], ["opus"])]


def test_null_pools_section_is_accepted(setup):
    args, config, _, _ = setup
    config["pools"] = None
    config["models"] = [config["models"][0]]
    args.config.write_text(yaml.safe_dump(config, sort_keys=False))
    w.cmd_pool_create(args)
    assert yaml.safe_load(args.config.read_text())["pools"] == {"opus-pool": ["first", "second", "third"]}


def test_disabled_model_is_rejected_clearly(setup):
    args, config, _, calls = setup
    config["model_overrides"] = {"opus": {"enabled": False}}
    args.config.write_text(yaml.safe_dump(config, sort_keys=False))
    original = args.config.read_bytes()
    with pytest.raises(SystemExit, match="disabled by runtime model_overrides: opus"):
        w.cmd_pool_create(args)
    assert args.config.read_bytes() == original and calls["smoke"] == []


def test_probe_model_services_stops_on_repeated_page_token(monkeypatch):
    monkeypatch.setattr(w, "_get_json", lambda url, token: {
        "model_services": [{"name": "model-services/system.ai.a"}], "next_page_token": "same"})
    assert w.probe_model_services("https://ws.example.com", "t") == {"system.ai.a"}


def test_probe_model_services_tolerates_read_timeout(monkeypatch):
    monkeypatch.setattr(w, "_get_json", lambda url, token: (_ for _ in ()).throw(TimeoutError("read")))
    assert w.probe_model_services("https://ws.example.com", "t") == set()


def test_backups_within_one_second_do_not_collide(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("a: 1\n")
    monkeypatch.setattr(w.time, "strftime", lambda fmt: "20260101-000000")
    first, second = w._backup(config), w._backup(config)
    assert first != second and second.name.endswith("-1")
    assert first.read_text() == second.read_text() == "a: 1\n"
    assert second.stat().st_mode & 0o777 == 0o600
