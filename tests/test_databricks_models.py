"""Databricks workspaces as one directory provider (docs/adr/0006-databricks-directory.md)."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from src import admin, circuit, config_io, directory, providers, shipped
from src import databricks_models as dbx
from src.server import app

ADMIN = {"Authorization": "Bearer admin"}
LABS = {
    "openai/gpt-6.2-sol": {"name": "GPT-6.2 Sol", "release_date": "2026-11-17"},
    "anthropic/claude-opus-5-5": {"name": "Claude Opus 5.5", "release_date": "2026-09-22"},
    "openai/gpt-5.5-pro": {"name": "GPT-5.5 Pro", "release_date": "2026-11-01"},
}


def _model(model_id, canonical, npm=None):
    model = {
        "id": model_id, "name": model_id, "reasoning": True,
        "reasoning_options": [{"type": "effort", "values": ["low", "medium", "high"]}],
        "tool_call": True, "modalities": {"input": ["text", "image"], "output": ["text"]},
        "limit": {"context": 1_000_000, "output": 128_000},
        "cost": {"input": 2.0, "output": 10.0}, "release_date": "2026-11-17", "canonical_model_id": canonical,
    }
    if npm:
        model["provider"] = {"npm": npm, "api": "https://${DATABRICKS_HOST}/serving-endpoints"}
    return model


def _snapshot(corrections):
    api = {"databricks": {"id": "databricks", "name": "Databricks", "models": {m["id"]: m for m in [
        _model("system.ai.gpt-6-2-sol", "openai/gpt-6.2-sol"),
        _model("databricks-claude-opus-5-5", "anthropic/claude-opus-5-5", "@ai-sdk/anthropic"),
        _model("databricks-gpt-5-5-pro", "openai/gpt-5.5-pro", "@ai-sdk/openai"),
    ]}}}
    snapshot = directory.build_snapshot(api, LABS, corrections, fetched_at="2026-11-27T00:00:00Z")
    return snapshot


# ws-a: a Unity /ai-gateway route; ws-b: a legacy serving-endpoints route; ws-c: no Databricks.
WORKSPACES = {
    "ws-a": {"base_url": "https://a.cloud.databricks.com/ai-gateway", "workspace_url": "https://a.cloud.databricks.com",
             "api_key": "token-a", "auth_refresh": "databricks-cli", "auth_profile": "a",
             "path_prefixes": {"openai": "mlflow/v1", "anthropic": "anthropic/v1", "responses": "openai/v1"}},
    "ws-b": {"base_url": "https://adb-1.2.azuredatabricks.net/serving-endpoints", "api_key": "token-b"},
    "fireworks": {"base_url": "https://api.fireworks.ai/inference/v1", "api_key": "fw"},
}


@pytest.fixture
def corrections():
    return directory.load_corrections()


@pytest.fixture
def env(tmp_path, monkeypatch, corrections):
    """Two workspaces, the directory turned on over a Databricks-only snapshot, no network."""
    for name in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_SERVING_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"auth": {"admin_keys": ["admin"]}, "providers": WORKSPACES}))
    path.chmod(0o600)
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"allow_empty": True, "llm": []}))
    for module in (providers, config_io):
        monkeypatch.setattr(module, "CONFIG_PATH", path)
        monkeypatch.setattr(module, "MODEL_INFO_PATH", info)
        monkeypatch.setattr(module, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.setattr(directory, "directory_dir", lambda config_path=None: tmp_path / "models-dev")
    snapshot = _snapshot(corrections)
    monkeypatch.setattr(shipped, "_directory", lambda config, ready, keep, config_path=None: directory.layer(
        config, ready=ready, keep=keep, snapshot=snapshot, corrections=corrections, config_path=config_path))
    monkeypatch.setattr(directory, "load_snapshot", lambda config_path=None: (snapshot, "synced"))
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    monkeypatch.setattr(admin, "_regenerate_catalogs", _no_catalogs)
    for name in ("ws-a", "ws-b"):
        circuit._circuits.pop(name, None)
    providers.reload()
    yield path
    for name in ("ws-a", "ws-b"):
        circuit._circuits.pop(name, None)
    providers.reload()


async def _no_catalogs():
    return {}


@pytest.fixture
def listing(monkeypatch):
    """What each workspace's REST API lists; requests and token refreshes are recorded."""
    served = {
        "https://a.cloud.databricks.com": {"endpoints": ["databricks-claude-opus-5-5", "databricks-gpt-5-5-pro"],
                                           "unity": ["system.ai.gpt-6-2-sol", "system.ai.custom-llm"]},
        "https://adb-1.2.azuredatabricks.net": {"endpoints": ["databricks-gpt-6-2-sol", "databricks-claude-opus-5-5"],
                                               "unity": ["system.ai.never-listed"]},
    }
    calls = {"get": [], "refresh": [], "refused": set()}

    def get_json(url, token):
        calls["get"].append((url, token))
        root = url.split("/api/")[0]
        if root in calls["refused"]:
            raise dbx.AuthError("HTTP 403")
        if "/serving-endpoints" in url:
            return {"endpoints": [{"name": name} for name in served[root]["endpoints"]]}
        return {"model_services": [{"name": f"model-services/{name}"} for name in served[root]["unity"]]}

    async def ensure(provider, *, min_valid_seconds=300, allow_login=True):
        calls["refresh"].append(("ensure", provider, allow_login))

    async def refresh(provider, *, force=False, allow_login=True, failed_token=None):
        calls["refresh"].append(("refresh", provider, allow_login))

    monkeypatch.setattr(dbx, "_get_json", get_json)
    monkeypatch.setattr(providers, "ensure_fresh_oauth_token", ensure)
    monkeypatch.setattr(providers, "refresh_oauth_token", refresh)
    return served, calls


@pytest.fixture
def smoke(monkeypatch):
    """Test completions: every route answers unless listed in ``failing`` (workspace ids)."""
    record = {"calls": [], "failing": set()}

    def post(url, json=None, headers=None, timeout=None):
        record["calls"].append((url, json["model"]))
        workspace = "ws-a" if "a.cloud" in url else "ws-b"
        if workspace in record["failing"]:
            return httpx.Response(404, json={"error": "ENDPOINT_NOT_FOUND"}, request=httpx.Request("POST", url))
        if url.endswith("/messages"):
            body = {"type": "message", "content": [{"type": "text", "text": "OK"}]}
        elif url.endswith("/responses"):
            body = {"output": []}
        else:
            body = {"choices": [{"message": {"content": "OK"}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(dbx.httpx, "post", post)
    return record


def _discover(**kwargs):
    return asyncio.run(dbx.discover(**kwargs))


def _test_all(*rows):
    """Run the test completions of these rows (or every loaded row) and reload."""
    rows = rows or tuple({id(v): v for v in providers._load_models().values()}.values())
    changed = dbx.test_routes(list(rows), providers._load_config())
    providers.reload()
    return changed


# ── Mapping ────────────────────────────────────────────────────────────────


def test_models_dev_databricks_models_become_routes_over_every_workspace(corrections):
    rows = {item.model_id: item.row for item in directory.map_snapshot(_snapshot(corrections), corrections)}
    sol, opus, pro = rows["system.ai.gpt-6-2-sol"], rows["databricks-claude-opus-5-5"], rows["databricks-gpt-5-5-pro"]
    assert (sol["name"], sol["provider"], sol["pool"], sol["base_model"]) == (
        "gpt-6.2-sol-dbx", "databricks", dbx.AUTO_POOL, "gpt-6.2-sol")
    assert sol["alternate_ids"] == ["databricks-gpt-6-2-sol"]
    assert sol["display_name"] == "GPT-6.2 Sol (Databricks)" and sol["pricing"] == {"input": 2.0, "output": 10.0}
    assert (sol["context"], sol["max_output_tokens"], sol["thinking"]) == (1_000_000, 128_000, "always")
    assert "protocol" not in sol and "api_style" not in sol
    assert opus["protocol"] == "anthropic" and opus["alternate_ids"] == ["system.ai.claude-opus-5-5"]
    assert pro["api_style"] == "open_responses"


def test_the_release_snapshot_lists_databricks_models(corrections):
    snapshot = json.loads(directory.BUNDLED_SNAPSHOT.read_text())
    rows = [item.row for item in directory.map_snapshot(snapshot, corrections)
            if item.provider == "databricks" and item.row is not None]
    assert len(rows) > 20 and all(row["pool"] == dbx.AUTO_POOL and row["name"].endswith("-dbx") for row in rows)
    assert {"claude-opus-5.5-dbx", "gpt-6.2-sol-dbx"} <= {row["name"] for row in rows}


def test_twins_cover_legacy_and_unity_names():
    assert dbx.twins("system.ai.gpt-6-2-sol")[:2] == ["system.ai.gpt-6-2-sol", "databricks-gpt-6-2-sol"]
    assert dbx.twins("databricks-gpt-6-2-sol")[:2] == ["databricks-gpt-6-2-sol", "system.ai.gpt-6-2-sol"]


# ── Workspaces and coverage ────────────────────────────────────────────────


def test_workspaces_are_found_by_cli_auth_host_or_name_in_config_order(monkeypatch):
    for name in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_SERVING_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    config = {"providers": {
        **WORKSPACES,
        "proxy": {"base_url": "https://proxy.example.com", "api_key": "k", "auth_refresh": "databricks-cli"},
        "off": {"base_url": "https://off.cloud.databricks.com", "api_key": "k", "enabled": False},
        "keyless": {"base_url": "https://keyless.cloud.databricks.com"},
        "databricks": {"base_url": "https://named.example.com/serving-endpoints", "api_key": "k"},
    }}
    assert dbx.workspaces(config) == ["ws-a", "ws-b", "proxy", "databricks"]
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "env-token")
    assert dbx.workspaces({}) == ["databricks"]


def test_discovery_checks_every_workspace_with_cached_sign_in_only(env, listing):
    _, calls = listing
    result = _discover()
    assert result["changed"] is True
    assert result["workspaces"]["ws-a"] == {"served": 4, "error": None, "auth_needed": False, "style": "unity"}
    # A serving-endpoints route lists endpoints only: Unity names need an /ai-gateway route.
    assert result["workspaces"]["ws-b"]["served"] == 2
    assert not any("adb-1.2" in url and "model-services" in url for url, _ in calls["get"])
    assert calls["refresh"] == [("ensure", "ws-a", False), ("ensure", "ws-b", False)]
    # Nothing secret is stored.
    assert "token-" not in dbx.coverage_path().read_text()
    # Checked within a day and unchanged: skipped.
    calls["get"].clear()
    assert _discover(stale_only=True) == {"changed": False, "workspaces": {}}
    assert calls["get"] == []


def test_a_workspace_refusing_its_token_keeps_what_it_served_and_never_opens_a_browser(env, listing):
    _, calls = listing
    _discover()
    calls["refused"].add("https://adb-1.2.azuredatabricks.net")
    result = _discover()
    assert result["workspaces"]["ws-b"]["auth_needed"] is True
    assert result["workspaces"]["ws-b"]["served"] == 2
    assert ("refresh", "ws-b", False) in calls["refresh"]
    assert all(allow_login is False for _, _, allow_login in calls["refresh"])


def test_a_removed_workspace_leaves_coverage_and_a_changed_route_needs_a_new_check(env, listing):
    _discover()
    config = providers._load_config()
    row = {"name": "x", "provider_model_id": "databricks-gpt-6-2-sol", "pool": dbx.AUTO_POOL}
    assert [ws for ws, _ in dbx.candidates(row, config)] == ["ws-a", "ws-b"]
    changed = {**config, "providers": {**config["providers"],
                                        "ws-b": {**WORKSPACES["ws-b"], "base_url": "https://adb-9.9.azuredatabricks.net/serving-endpoints"}}}
    assert [ws for ws, _ in dbx.candidates(row, changed)] == ["ws-a"]
    env.write_text(yaml.safe_dump({"auth": {"admin_keys": ["admin"]}, "providers": {"ws-a": WORKSPACES["ws-a"]}}))
    providers.reload()
    _discover()
    assert set(dbx.load_coverage()["workspaces"]) == {"ws-a"}


# ── Pools ──────────────────────────────────────────────────────────────────


def test_every_serving_workspace_joins_the_pool_under_its_own_name(env, listing, smoke):
    _discover()
    config = providers._load_config()
    sol = {"name": "gpt-6.2-sol-dbx", "provider_model_id": "system.ai.gpt-6-2-sol", "pool": dbx.AUTO_POOL}
    opus = {"name": "o", "provider_model_id": "databricks-claude-opus-5-5", "pool": dbx.AUTO_POOL, "protocol": "anthropic"}
    pro = {"name": "p", "provider_model_id": "databricks-gpt-5-5-pro", "pool": dbx.AUTO_POOL, "api_style": "open_responses"}
    nowhere = {"name": "n", "provider_model_id": "databricks-not-served", "pool": dbx.AUTO_POOL}
    # Served, but no route has answered its test completion yet: the pool is empty.
    assert dbx.candidates(sol, config) == [("ws-a", "system.ai.gpt-6-2-sol"), ("ws-b", "databricks-gpt-6-2-sol")]
    assert dbx.members(sol, config) == []
    assert _test_all(sol, opus, pro, nowhere) is True
    config = providers._load_config()
    assert dbx.members(sol, config) == [("ws-a", "system.ai.gpt-6-2-sol"), ("ws-b", "databricks-gpt-6-2-sol")]
    # Anthropic Messages needs a route with an anthropic prefix; Responses one with a responses prefix.
    assert dbx.members(opus, config) == [("ws-a", "databricks-claude-opus-5-5")]
    assert dbx.members(pro, config) == [("ws-a", "databricks-gpt-5-5-pro")]
    assert dbx.members(nowhere, config) == []
    assert len(smoke["calls"]) == 4  # sol twice, opus and pro once; tested routes aren't tested again
    assert _test_all(sol, opus, pro) is False and len(smoke["calls"]) == 4


def test_a_route_that_failed_its_test_is_left_out_and_tested_again_after_a_day(env, listing, smoke):
    _discover()
    row = {"name": "x", "provider_model_id": "system.ai.gpt-6-2-sol", "pool": dbx.AUTO_POOL}
    smoke["failing"].add("ws-a")
    assert _test_all(row) is True
    config = providers._load_config()
    assert [ws for ws, _ in dbx.members(row, config)] == ["ws-b"]
    assert dbx.untested(row, config) == []
    later = time.time() + dbx.FAILED_ROUTE_SECONDS + 1
    # A day later the failure doesn't rejoin the pool by itself: it is due for another test.
    assert [ws for ws, _ in dbx.members(row, config, now=later)] == ["ws-b"]
    assert [ws for ws, _, _ in dbx.untested(row, config, now=later)] == ["ws-a"]
    # A check without clear_failures keeps it; an explicit one clears it, and the retest decides.
    _discover()
    assert [ws for ws, _, _ in dbx.untested(row, config)] == []
    _discover(clear_failures=True)
    smoke["failing"].clear()
    assert _test_all(row) is True
    assert [ws for ws, _ in dbx.members(row, providers._load_config())] == ["ws-a", "ws-b"]


def test_a_workspace_refusing_its_token_in_a_test_gets_no_route_verdict(env, listing, smoke, monkeypatch):
    _discover()
    row = {"name": "x", "provider_model_id": "system.ai.gpt-6-2-sol", "pool": dbx.AUTO_POOL}
    original = dbx.httpx.post

    def post(url, json=None, headers=None, timeout=None):
        if "a.cloud" in url:
            return httpx.Response(401, json={"error": "expired"}, request=httpx.Request("POST", url))
        return original(url, json=json, headers=headers, timeout=timeout)

    monkeypatch.setattr(dbx.httpx, "post", post)
    _test_all(row)
    config = providers._load_config()
    assert [ws for ws, _ in dbx.members(row, config)] == ["ws-b"]
    assert [ws for ws, _, _ in dbx.untested(row, config)] == ["ws-a"]  # tried again next pass


def test_concurrent_coverage_writes_keep_every_change(env):
    import threading

    def write(index):
        dbx.update_coverage(lambda coverage: coverage["routes"].update({f"ws|m{index}": {"ok": True}}))

    threads = [threading.Thread(target=write, args=(index,)) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(dbx.load_coverage()["routes"]) == 20
    assert dbx.coverage_path().stat().st_mode & 0o777 == 0o600


def test_directory_rows_load_route_and_fail_over_across_workspaces(env, listing, smoke):
    assert "gpt-6.2-sol-dbx" not in providers._load_models()  # no coverage yet: not offered
    _discover()
    providers.reload()
    # Loaded once a workspace serves it, available once a route answered its test completion.
    assert providers.model_availability("gpt-6.2-sol-dbx")["reason"] == "no_serving_workspace"
    _test_all()
    models = providers._load_models()
    assert {"gpt-6.2-sol-dbx", "claude-opus-5.5-dbx", "gpt-5.5-pro-dbx"} <= set(models)
    assert models["databricks-gpt-6-2-sol"]["name"] == "gpt-6.2-sol-dbx"  # either name routes
    assert providers.pool_candidates("gpt-6.2-sol-dbx") == ["ws-a", "ws-b"]
    first = providers.resolve("gpt-6.2-sol-dbx")
    assert (first.provider, first.provider_model_id, first.base_url) == (
        "ws-a", "system.ai.gpt-6-2-sol", "https://a.cloud.databricks.com/ai-gateway/mlflow/v1")
    backup = providers.resolve("gpt-6.2-sol-dbx", provider_override="ws-b")
    assert (backup.provider_model_id, backup.base_url) == (
        "databricks-gpt-6-2-sol", "https://adb-1.2.azuredatabricks.net/serving-endpoints")
    for _ in range(10):
        circuit.record_failure("ws-a")
    assert providers.resolve("gpt-6.2-sol-dbx").provider == "ws-b"
    opus = providers.resolve("claude-opus-5.5-dbx")
    assert (opus.protocol, opus.base_url) == ("anthropic", "https://a.cloud.databricks.com/ai-gateway/anthropic/v1")
    # A bare base-model name with no lab API configured is served by Databricks.
    assert providers.base_model_route("gpt-6.2-sol") == "gpt-6.2-sol-dbx"


def test_a_hand_curated_databricks_row_wins_over_the_directory(env, listing):
    _discover()
    env_config = yaml.safe_load(env.read_text())
    env_config["models"] = [{"name": "sol", "provider": "ws-b", "provider_model_id": "databricks-gpt-6-2-sol",
                             "context": 1000, "max_output_tokens": 100}]
    env.write_text(yaml.safe_dump(env_config))
    providers.reload()
    models = providers._load_models()
    assert "gpt-6.2-sol-dbx" not in models and models["databricks-gpt-6-2-sol"]["name"] == "sol"
    assert providers.resolve("sol").provider == "ws-b"


# ── Admin ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client(env):
    with TestClient(app) as c:
        yield c


def test_smoke_tests_each_member_and_a_failing_workspace_leaves_the_pool(client, listing, smoke):
    smoke["failing"].add("ws-b")
    result = client.post("/admin/api/databricks/discover", headers=ADMIN)
    assert result.status_code == 200, result.text
    tested = {(url, model) for url, model in smoke["calls"]}
    assert ("https://a.cloud.databricks.com/ai-gateway/mlflow/v1/chat/completions", "system.ai.gpt-6-2-sol") in tested
    assert ("https://adb-1.2.azuredatabricks.net/serving-endpoints/chat/completions", "databricks-gpt-6-2-sol") in tested
    assert ("https://a.cloud.databricks.com/ai-gateway/anthropic/v1/messages", "databricks-claude-opus-5-5") in tested
    assert ("https://a.cloud.databricks.com/ai-gateway/openai/v1/responses", "databricks-gpt-5-5-pro") in tested
    assert providers.pool_candidates("gpt-6.2-sol-dbx") == ["ws-a"]
    status = client.get("/admin/api/databricks", headers=ADMIN).json()
    assert [w["id"] for w in status["workspaces"]] == ["ws-a", "ws-b"]
    pools = {m["name"]: [p["workspace"] for p in m["pool"]] for m in status["models"]}
    assert pools["gpt-6.2-sol-dbx"] == ["ws-a"]
    assert status["unmapped"] == {"system.ai.custom-llm": ["ws-a"]}
    directory_status = client.get("/admin/api/directory", headers=ADMIN).json()
    assert {"id": "databricks", "display_name": "Databricks", "ready": True} in directory_status["providers"]


def test_listing_and_adding_a_model_the_directory_doesnt_map_checks_every_workspace(client, listing, smoke):
    served, calls = listing
    served["https://adb-1.2.azuredatabricks.net"]["endpoints"].append("databricks-custom-llm")
    listed = client.post("/admin/api/providers/databricks/discover", headers=ADMIN)
    assert listed.status_code == 200, listed.text
    models = {m["id"]: m for m in listed.json()["models"]}
    assert models["system.ai.gpt-6-2-sol"]["directory"] == "gpt-6.2-sol-dbx"
    assert models["system.ai.custom-llm"] == {"id": "system.ai.custom-llm", "registered": False,
                                              "workspaces": ["ws-a"], "directory": None}
    mapped = client.post("/admin/api/providers/databricks/models", headers=ADMIN, json={"model": "databricks-gpt-6-2-sol"})
    assert mapped.status_code == 409 and mapped.json()["code"] == "in_directory"
    smoke["calls"].clear()
    calls["get"].clear()
    added = client.post("/admin/api/providers/databricks/models", headers=ADMIN, json={"model": "system.ai.custom-llm"})
    assert added.status_code == 200, added.text
    # Adding re-checks every configured workspace, not just the first that lists the model.
    assert {url.split("/api/")[0] for url, _ in calls["get"]} == {
        "https://a.cloud.databricks.com", "https://adb-1.2.azuredatabricks.net"}
    name = added.json()["model"]["name"]
    assert name == "databricks-custom-llm"
    # Both workspaces serve it (one by its legacy name) and each answered a test first.
    assert sorted(model for _, model in smoke["calls"]) == ["databricks-custom-llm", "system.ai.custom-llm"]
    stored = next(row for row in config_io.load_model_info()["llm"] if row["name"] == name)
    assert stored["pool"] == dbx.AUTO_POOL and stored["provider"] == "databricks" and "pricing" not in stored
    assert providers.pool_candidates(name) == ["ws-a", "ws-b"]
    missing = client.post("/admin/api/providers/databricks/models", headers=ADMIN, json={"model": "databricks-nope"})
    assert missing.status_code == 400 and missing.json()["code"] == "model_not_listed"


def test_exported_aliases_list_a_databricks_route_on_its_first_serving_workspace(env, listing, smoke):
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_catalogs", providers.Path(__file__).parents[1]
                                                  / "scripts" / "export_catalogs.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    _discover()
    row = {"name": "gpt-6.2-sol-dbx", "provider": "databricks", "pool": dbx.AUTO_POOL,
           "provider_model_id": "system.ai.gpt-6-2-sol", "context": 1000, "max_output_tokens": 100}
    _test_all(row)
    config = providers._load_config()
    # A legacy coverage stamp from `workspace add --allow-partial` doesn't hide a directory route.
    config["providers"]["ws-b"]["available_model_ids"] = ["databricks-other"]
    assert export._effective_provider(row, config, env) == "ws-a"
    assert export._model_available(row, "ws-a", config) is True
    nowhere = {**row, "provider_model_id": "databricks-not-served"}
    assert export._effective_provider(nowhere, config, env) == "databricks"
    assert export._provider_serveable("databricks", config, env) is False


def test_failover_names_the_model_by_the_backup_workspace_s_own_id(env, listing, smoke, monkeypatch):
    from src import upstream

    _discover()
    _test_all()
    for name, value in (("_RETRY_MAX", 1), ("_RETRY_429_ATTEMPTS", 1), ("_RETRY_TRANSPORT_ATTEMPTS", 1)):
        monkeypatch.setattr(upstream, name, value)
    monkeypatch.setattr(upstream, "_compute_retry_delay", lambda resp, attempt: 0.0)
    info = providers.resolve("gpt-6.2-sol-dbx")
    sent = []

    def handler(request):
        sent.append((request.url.host, json.loads(request.content)["model"]))
        if request.url.host == "a.cloud.databricks.com":
            return httpx.Response(503, text="down")
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    pool = upstream.PoolContext("gpt-6.2-sol-dbx", info.provider, info.base_url, info.api_key,
                                info.protocol, info.api_style, info.provider_model_id)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await upstream._retry_post_with_model_fallback(
                client, f"{info.base_url}/chat/completions",
                json={"model": info.provider_model_id, "max_tokens": 1},
                headers={"Authorization": f"Bearer {info.api_key}"}, provider=info.provider, pool=pool)

    assert asyncio.run(run()).status_code == 200
    assert sent == [("a.cloud.databricks.com", "system.ai.gpt-6-2-sol"),
                    ("adb-1.2.azuredatabricks.net", "databricks-gpt-6-2-sol")]


def test_a_scoped_credential_cannot_reach_every_workspace(env, listing, smoke):
    config = yaml.safe_load(env.read_text())
    config["auth"]["consumer_credentials"] = [
        {"id": "app-manager", "consumer": "app", "key": "app-token", "namespaces": ["app"],
         "permissions": ["models:register"], "providers": ["databricks"]}]
    env.write_text(yaml.safe_dump(config))
    providers.reload()
    with TestClient(app) as client:
        scoped = {"Authorization": "Bearer app-token"}
        listed = client.post("/admin/api/providers/databricks/discover", headers=scoped)
        added = client.post("/admin/api/providers/databricks/models", headers=scoped, json={"model": "system.ai.x"})
    assert (listed.status_code, added.status_code) == (403, 403)
    assert listing[1]["get"] == [] and smoke["calls"] == []


def test_a_403_fails_only_that_route_and_the_workspace_s_other_routes_are_still_tested(env, listing, smoke, monkeypatch):
    _discover()
    original = dbx.httpx.post

    def post(url, json=None, headers=None, timeout=None):
        if "a.cloud" in url and json["model"] == "system.ai.gpt-6-2-sol":
            return httpx.Response(403, json={"error": "no Query permission"}, request=httpx.Request("POST", url))
        return original(url, json=json, headers=headers, timeout=timeout)

    monkeypatch.setattr(dbx.httpx, "post", post)
    sol = {"name": "s", "provider_model_id": "system.ai.gpt-6-2-sol", "pool": dbx.AUTO_POOL}
    opus = {"name": "o", "provider_model_id": "databricks-claude-opus-5-5", "pool": dbx.AUTO_POOL, "protocol": "anthropic"}
    _test_all(sol, opus)
    config = providers._load_config()
    assert [ws for ws, _ in dbx.members(sol, config)] == ["ws-b"]
    assert [ws for ws, _ in dbx.members(opus, config)] == ["ws-a"]


def test_a_transient_unity_listing_failure_keeps_the_previous_list(env, listing, monkeypatch):
    _discover()
    before = dbx.load_coverage()["workspaces"]["ws-a"]["served"]
    original = dbx._get_json

    def get_json(url, token):
        if "model-services" in url:
            request = httpx.Request("GET", url)
            raise httpx.HTTPStatusError("503", request=request, response=httpx.Response(503, request=request))
        return original(url, token)

    monkeypatch.setattr(dbx, "_get_json", get_json)
    result = _discover()
    assert result["workspaces"]["ws-a"]["error"].startswith("HTTPStatusError")
    assert dbx.load_coverage()["workspaces"]["ws-a"]["served"] == before


def test_an_invalid_settings_change_is_refused_before_any_workspace_is_asked(client, listing, smoke):
    _, calls = listing
    refused = client.put("/admin/api/directory/settings", headers=ADMIN, json={"models": [""]})
    assert refused.status_code == 400 and calls["get"] == []
    turned_on = client.put("/admin/api/directory/settings", headers=ADMIN, json={"models": ["gpt-6.2-sol-dbx"]})
    assert turned_on.status_code == 200, turned_on.text
    assert {url.split("/api/")[0] for url, _ in calls["get"]} == {
        "https://a.cloud.databricks.com", "https://adb-1.2.azuredatabricks.net"}
