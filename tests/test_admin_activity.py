"""Admin "Running now" and usage-by-catalog-model APIs."""

import json
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

import src.server as server_module
from src import admin, config_io, inflight, ledger, providers
from src.providers import ProviderInfo
from src.server import app
from src.usage import CostEstimate, Usage

ADMIN = {"Authorization": "Bearer admin"}
MODELS = [
    {"name": "glm-fw", "alias": "glmfw", "provider": "fireworks", "provider_model_id": "accounts/fw/glm",
     "pricing": {"input": 1.0, "output": 2.0}},
    {"name": "kimi-direct", "provider": "moonshot", "provider_model_id": "kimi-k3", "pricing": {"input": 1.0, "output": 2.0}},
    {"name": "local-glm", "alias": "lglm", "provider": "omlx", "provider_model_id": "GLM-Local-6bit",
     "pricing_status": "unmetered"},
    {"name": "local-qwen", "provider": "omlx", "provider_model_id": "Qwen-Local-8bit", "pricing_status": "unmetered"},
    {"name": "pic", "provider": "media", "provider_model_id": "flux-klein", "modality": "image",
     "pricing_status": "unmetered"},
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "auth:\n  admin_keys:\n    - admin\nproviders:\n"
        "  fireworks:\n    base_url: https://fw.test/v1\n    api_key: k\n"
        "  moonshot:\n    base_url: https://moon.test/v1\n    api_key: k\n"
        "  omlx:\n    base_url: http://omlx.test/v1\n    api_key: omlx-key\n"
        "  media:\n    base_url: http://media.test/v1\n    api_key: media-key\n"
    )
    catalog = tmp_path / "model-info.json"
    catalog.write_text(json.dumps({"llm": MODELS}))
    for module in (providers, config_io):
        monkeypatch.setattr(module, "CONFIG_PATH", cfg)
        monkeypatch.setattr(module, "MODEL_INFO_PATH", catalog)
        monkeypatch.setattr(module, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.setenv("MODEL_GATEWAY_LEDGER_PATH", str(tmp_path / "ledger.db"))
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin")
    providers.reload()
    ledger.init()
    with TestClient(app) as c:
        yield c
    providers.reload()


def _record(model, provider, upstream, cost=0.0, ts=None):
    rid = ledger.record(
        endpoint="/v1/chat/completions", method="POST", model=model, provider=provider,
        provider_model_id=upstream, status=200, latency_ms=100, is_stream=False,
        usage=Usage(input_tokens=10, output_tokens=5, cached_read_tokens=0, cache_write_tokens=0,
                    reasoning_tokens=0, reported=True),
        cost=CostEstimate(cost, True, []),
    )
    if ts is not None:
        with ledger._connect() as conn:
            conn.execute("UPDATE requests SET ts = ? WHERE id = ?", (ts, rid))


def test_catalog_namer_maps_every_routable_id_to_its_model(client):
    name = admin._catalog_namer(providers.model_status())
    assert name("glm-fw") == "glm-fw"
    assert name("glmfw") == "glm-fw"
    assert name("accounts/fw/glm") == "glm-fw"
    assert name("lglm") == "local-glm"
    # Like model stats, history recorded under a name the catalog no longer routes stays separate.
    assert name("old-glm-name") is None
    assert name(None) is None


def test_media_and_omlx_models_are_local(client):
    rows = {row["name"]: row for row in providers.model_status()}
    assert rows["pic"]["locality"] == "local"
    assert rows["local-glm"]["locality"] == "local"
    assert rows["glm-fw"]["locality"] == "cloud"


def test_usage_groups_every_alias_under_its_catalog_model_with_a_daily_series(client):
    now = time.time()
    _record("glm-fw", "fireworks", "accounts/fw/glm", 1.25, now - 3600)
    _record("glmfw", "fireworks", "accounts/fw/glm", 0.75, now - 2 * 86400)
    _record("kimi-k3", "moonshot", "kimi-k3", 0.5, now - 60)
    _record("lglm", "omlx", "GLM-Local-6bit", 0.0, now - 60)
    _record("retired-model", "fireworks", "gone", 0.1, now - 60)
    _record("glm-fw", "fireworks", "accounts/fw/glm", 9.0, now - 30 * 86400)

    body = client.get("/admin/api/usage?window=7d", headers=ADMIN).json()
    groups = {g["model"]: g for g in body["by_catalog_model"]}
    assert [g["model"] for g in body["by_catalog_model"]][:2] == ["glm-fw", "kimi-direct"]
    assert groups["glm-fw"]["requests"] == 2
    assert groups["glm-fw"]["cost_usd"] == pytest.approx(2.0)
    assert groups["glm-fw"]["providers"] == ["fireworks"]
    assert groups["glm-fw"]["avg_latency_ms"] == 100
    assert groups["local-glm"]["locality"] == "local"
    assert groups["retired-model"]["in_catalog"] is False
    assert "latency_ms_sum" not in groups["glm-fw"]

    series = body["series"]
    assert series["bucket"] == "day"
    assert len(series["points"]) in (8, 9)  # every local day in the window, gaps filled
    assert sum(p["requests"] for p in series["points"]) == 5
    assert sum(p["cost_usd"] for p in series["points"]) == pytest.approx(2.6)
    assert sum(series["cost_by_model"]["glm-fw"]) == pytest.approx(2.0)
    assert len(series["cost_by_model"]["glm-fw"]) == len(series["points"])
    assert body["range"]["bucket"] == "day"

    # Any number of days works; short windows bucket by hour.
    assert client.get("/admin/api/usage?window=45d", headers=ADMIN).json()["summary"]["requests"] == 6
    hourly = client.get("/admin/api/usage?window=24h", headers=ADMIN).json()["series"]
    assert hourly["bucket"] == "hour" and len(hourly["points"]) in (24, 25)


def test_long_windows_start_at_the_first_request_and_bucket_by_month(client):
    now = time.time()
    _record("glm-fw", "fireworks", "accounts/fw/glm", 1.0, now - 400 * 86400)
    _record("glm-fw", "fireworks", "accounts/fw/glm", 2.0, now - 60)
    huge = client.get("/admin/api/usage?window=1000000d", headers=ADMIN)
    assert huge.status_code == 200
    series = huge.json()["series"]
    assert series["bucket"] == "month" and 13 <= len(series["points"]) <= 15
    assert sum(p["cost_usd"] for p in series["points"]) == pytest.approx(3.0)
    assert client.get("/admin/api/usage", headers=ADMIN).json()["series"]["points"] == series["points"]
    # A window up to 180 days is filled from its start, even before the first request.
    assert len(client.get("/admin/api/usage?window=90d", headers=ADMIN).json()["series"]["points"]) in (91, 92)


def test_series_plan():
    now = time.time()
    assert admin._series_plan(now - 3600, None, now - 60) == (now - 3600, "hour")
    assert admin._series_plan(now - 90 * 86400, None, now - 60) == (now - 90 * 86400, "day")
    assert admin._series_plan(now - 3650 * 86400, None, now - 30 * 86400) == (now - 30 * 86400, "day")
    assert admin._series_plan(None, None, now - 365 * 86400)[1] == "month"
    assert admin._series_plan(None, None, None) == (None, "hour")


def test_model_stats_include_a_series_for_the_window(client):
    now = time.time()
    _record("glm-fw", "fireworks", "accounts/fw/glm", 1.0, now - 86400)
    _record("glmfw", "fireworks", "accounts/fw/glm", 2.0, now - 60)
    body = client.get("/admin/api/models/glm-fw/stats?window=30d", headers=ADMIN).json()
    assert body["usage"]["cost_usd"] == pytest.approx(3.0)
    assert body["series"]["bucket"] == "day"
    assert sum(p["cost_usd"] for p in body["series"]["points"]) == pytest.approx(3.0)
    assert client.get("/admin/api/models/missing/stats", headers=ADMIN).json()["series"] is None


def _runtime_transport(omlx=None, media=None):
    def handler(request: httpx.Request):
        if request.url.host == "omlx.test":
            assert request.url.path == "/v1/models/status"
            assert request.headers["authorization"] == "Bearer omlx-key"
            return omlx(request) if callable(omlx) else httpx.Response(200, json=omlx)
        if request.url.host == "media.test":
            assert request.url.path == "/health"
            return media(request) if callable(media) else httpx.Response(200, json=media)
        raise AssertionError(request.url)
    return httpx.MockTransport(handler)


def test_runtime_reports_loaded_local_models_and_media_generation(client, monkeypatch):
    omlx = {"current_model_memory": 300_000_000_000, "final_ceiling": 500_000_000_000, "models": [
        {"id": "GLM-Local-6bit", "loaded": True, "is_loading": False, "resident_estimated_size": 276_000_000_000,
         "actual_size": 263_000_000_000, "last_access": 1_700_000_000.5, "pinned": False},
        {"id": "Qwen-Local-8bit", "loaded": False, "is_loading": True, "estimated_size": 31_000_000_000},
        {"id": "Unlisted-Model", "loaded": False, "estimated_size": 9},
        {"id": "MarkItDown", "loaded": True, "estimated_size": 0},
        {"id": "Manual-Load", "loaded": True, "estimated_size": 5_000_000_000},
    ]}
    media = {"status": "ok", "scheduler": {"state": "running", "loaded_model": "flux-klein",
                                           "running": "job-1", "queued": 2}}
    monkeypatch.setattr(admin, "_runtime_transport", _runtime_transport(omlx, media))
    body = client.get("/admin/api/runtime", headers=ADMIN).json()
    local = body["local"]
    assert local["reachable"] is True and local["memory_bytes"] == 300_000_000_000
    rows = {row["id"]: row for row in local["models"]}
    assert set(rows) == {"GLM-Local-6bit", "Qwen-Local-8bit", "Manual-Load"}
    assert rows["GLM-Local-6bit"] == {
        "id": "GLM-Local-6bit", "catalog_models": ["local-glm"], "loaded": True, "loading": False,
        "size_bytes": 276_000_000_000, "last_access": 1_700_000_000.5, "pinned": False,
    }
    assert rows["Qwen-Local-8bit"]["loading"] is True
    assert rows["Manual-Load"]["catalog_models"] == []
    assert body["media"] == {"configured": True, "reachable": True, "state": "running",
                             "loaded_model": "flux-klein", "catalog_models": ["pic"], "busy": True, "queued": 2}
    assert body["in_flight"] == []
    assert client.get("/admin/api/runtime").status_code == 401


def test_runtime_reports_unreachable_runtimes_without_failing(client, monkeypatch):
    def refused(request):
        raise httpx.ConnectError("refused", request=request)
    monkeypatch.setattr(admin, "_runtime_transport", _runtime_transport(
        refused, lambda request: httpx.Response(503, json={})))
    body = client.get("/admin/api/runtime", headers=ADMIN).json()
    assert body["local"] == {"configured": True, "reachable": False, "error": "not reachable"}
    assert body["media"] == {"configured": True, "reachable": False, "error": "HTTP 503"}


def test_runtime_skips_runtimes_without_catalog_models(client, monkeypatch):
    catalog = json.loads(providers.MODEL_INFO_PATH.read_text())
    catalog["llm"] = [row for row in catalog["llm"] if row["provider"] not in {"omlx", "media"}]
    providers.MODEL_INFO_PATH.write_text(json.dumps(catalog))
    providers.reload()
    monkeypatch.setattr(admin, "_runtime_transport", _runtime_transport())
    body = client.get("/admin/api/runtime", headers=ADMIN).json()
    assert body["local"] == {"configured": False} and body["media"] == {"configured": False}


def _info():
    return ProviderInfo(provider="fireworks", base_url="http://up", api_key="k", provider_model_id="accounts/fw/glm",
                        protocol="openai", context=0, max_output_tokens=1024, thinking="", vision=False)


def test_requests_are_listed_as_running_until_they_are_recorded(client, monkeypatch):
    monkeypatch.setattr(admin, "_runtime_transport", _runtime_transport(
        {"models": []}, {"scheduler": {}}))
    monkeypatch.setattr(server_module, "resolve", lambda model: _info() if model == "glmfw" else None)
    seen = []

    async def fake_passthrough_sync(endpoint, body, headers, **kwargs):
        seen.append(inflight.snapshot())
        return server_module.JSONResponse(status_code=200, content={
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })

    async def fake_passthrough_stream(endpoint, body, headers, **kwargs):
        async def chunks():
            seen.append(inflight.snapshot())
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"
        return server_module.StreamingResponse(chunks(), media_type="text/event-stream")

    monkeypatch.setattr(server_module, "_passthrough_sync", fake_passthrough_sync)
    monkeypatch.setattr(server_module, "_passthrough_stream", fake_passthrough_stream)
    body = {"model": "glmfw", "messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert client.post("/v1/chat/completions", json={**body, "stream": True}).status_code == 200

    (sync_row,), (stream_row,) = seen
    assert sync_row["model"] == "glmfw" and sync_row["provider_model_id"] == "accounts/fw/glm"
    assert sync_row["provider"] == "fireworks" and sync_row["is_stream"] is False
    assert sync_row["endpoint"] == "/v1/chat/completions" and sync_row["elapsed_ms"] >= 0
    assert stream_row["is_stream"] is True
    assert inflight.snapshot() == []
    assert len(ledger.recent()) == 2


def test_runtime_names_in_flight_requests_by_catalog_model(client, monkeypatch):
    monkeypatch.setattr(admin, "_runtime_transport", _runtime_transport({"models": []}, {"scheduler": {}}))

    state = SimpleNamespace(ledger_ctx={"model": "glmfw", "provider": "fireworks",
                                        "provider_model_id": "accounts/fw/glm", "is_stream": True})
    resolving, served = inflight.start("/v1/messages", object()), inflight.start("/v1/chat/completions", state)
    try:
        rows = client.get("/admin/api/runtime", headers=ADMIN).json()["in_flight"]
    finally:
        inflight.finish(resolving)
        inflight.finish(served)
    assert [(row["endpoint"], row["model"], row["catalog_model"]) for row in rows] == [
        ("/v1/messages", None, None), ("/v1/chat/completions", "glmfw", "glm-fw")]
    assert rows[1]["is_stream"] is True


def test_a_failing_handler_does_not_stay_listed_as_running(client, monkeypatch):
    monkeypatch.setattr(server_module, "resolve", lambda model: _info())

    async def boom(endpoint, body, headers, **kwargs):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(server_module, "_passthrough_sync", boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.post("/v1/chat/completions", json={
            "model": "glmfw", "messages": [{"role": "user", "content": "hi"}]}).status_code == 500
    assert inflight.snapshot() == []


def test_stale_in_flight_entries_are_dropped(monkeypatch):
    stale = inflight.start("/v1/messages", object())
    later = time.time() + inflight.STALE_AFTER_SECONDS + 1
    monkeypatch.setattr(inflight.time, "time", lambda: later)
    fresh = inflight.start("/v1/messages", object())
    try:
        assert stale not in inflight._active  # pruned when the next request starts
        assert [row["id"] for row in inflight.snapshot()] == [fresh]
    finally:
        inflight.finish(fresh)
