"""Trusted-local peers and catalog schema 2
(docs/adr/0003-scoped-access-and-lightweight-local-models.md section 4)."""

from __future__ import annotations

import asyncio
import json
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src import federation, profiles, providers, server

client = TestClient(server.app)
CLIENT = "client-secret"
CONSUMER = "consumer-secret"
PEER = "peer-secret"
PEER_HEADERS = {"Authorization": f"Bearer {PEER}", federation.SOURCE_HEADER: "edge",
                federation.OWNER_HEADER: "main", "Via": "1.1 edge-model-gateway"}

LOCAL_ROWS = [
    {"name": "qwen-local", "base_model": "qwen-x", "family": "qwen", "provider": "omlx", "omlx_id": "qwen-local-mlx",
     "pricing_status": "unmetered", "vision": True},
    {"name": "glm-cloud", "base_model": "glm-x", "family": "glm", "provider": "cloud-test",
     "provider_model_id": "glm-native", "pricing": {"input": 1.0, "output": 2.0}},
    {"name": "text-local", "provider": "omlx", "omlx_id": "text-local-mlx", "pricing_status": "unmetered"},
]


def _configure(monkeypatch, *, models=LOCAL_ROWS, extra=None):
    config = {
        "providers": {
            "omlx": {"base_url": "http://127.0.0.1:9110/v1", "api_key": "omlx"},
            "cloud-test": {"base_url": "https://cloud.invalid/v1", "api_key": "secret"},
        },
        "models": deepcopy(models),
        "auth": {
            "client_keys": [CLIENT],
            "consumer_credentials": [{
                "id": "ha-runtime", "consumer": "ha", "key": CONSUMER, "namespaces": ["ha"],
                "permissions": ["profiles:read", "profiles:write", "profiles:invoke"],
            }],
        },
        **(extra or {}),
    }
    monkeypatch.setattr(providers, "_config", config)
    monkeypatch.setattr(providers, "_models", None)
    return config


def _peer_config(tmp_path: Path, *, trusted: bool = True, peers=("edge",)) -> federation.FederationConfig:
    return federation.FederationConfig(
        node_id="main",
        peers={peer: federation.PeerConfig(peer, f"https://{peer}.example", PEER, trusted_local=trusted)
               for peer in peers},
        refresh_interval_seconds=60, stale_after_seconds=30, request_timeout_seconds=2,
        cache_path=tmp_path / "federation-cache.json", max_catalog_bytes=100_000, max_models_per_peer=100,
    )


def _v2_catalog(node="edge", rows=None, revision=1) -> dict:
    rows = rows if rows is not None else [
        {"direct_model_id": "qwen-peer", "locality": "local", "base_model": "qwen-x", "vision": True},
        {"direct_model_id": "glm-peer", "locality": "cloud", "base_model": "glm-x"},
        {"direct_model_id": "big-local", "locality": "local"},
    ]
    models = [{"owner_node": node, "object": "model", "created": 0, **row} for row in rows]
    models.sort(key=lambda m: m["direct_model_id"])
    return {"schema_version": 2, "node_id": node, "revision": revision, "digest": federation.catalog_digest(models),
            "generated_at": "2026-01-01T00:00:00Z", "models": models}


def _seed(manager, payload, *, fresh=True):
    normalized, by_id = federation.validate_catalog(payload, payload["node_id"], max_models=100)
    state = manager.states[payload["node_id"]]
    state.schema_version, state.models = normalized["schema_version"], by_id
    state.revision, state.digest = normalized["revision"], normalized["digest"]
    state.generated_at = normalized["generated_at"]
    state.last_success_at = time.time() if fresh else time.time() - 3600
    state.healthy = fresh


@pytest.fixture
def importer(tmp_path, monkeypatch):
    """This node ("main") with trusted peer "edge" whose schema 2 catalog is loaded."""
    forwarded = []

    async def owner(request: httpx.Request):
        forwarded.append((request.url.path, json.loads(await request.aread())["model"]))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "from edge"}}]})

    manager = federation.FederationManager(_peer_config(tmp_path), transport=httpx.MockTransport(owner))
    _seed(manager, _v2_catalog())
    monkeypatch.setattr(federation, "_manager", manager)
    manager.forwarded = forwarded
    return manager


def _chat(model, token=CLIENT, **extra):
    return client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}", **extra},
                       json={"model": model, "messages": [{"role": "user", "content": "hi"}]})


# ── configuration and schema 2 ──────────────────────────────────────────────


def test_trusted_local_peers_need_https(tmp_path):
    config = tmp_path / "config.yaml"
    for url, trusted, error in (("http://edge.local:9111", True, "must use https"),
                                ("https://edge.local", "yes", "must be true or false")):
        config.write_text(f"federation:\n  node_id: main\n  peers:\n    edge:\n      base_url: {url}\n"
                          f"      api_key: k\n      trusted_local: {json.dumps(trusted)}\n")
        with pytest.raises(federation.FederationConfigError, match=error):
            federation.load_config(config)
    config.write_text("federation:\n  node_id: main\n  peers:\n    edge:\n      base_url: https://edge.local\n"
                      "      api_key: k\n      trusted_local: true\n")
    assert federation.load_config(config).peers["edge"].trusted_local is True


def test_schema_2_requires_a_locality_and_keeps_base_models():
    normalized, by_id = federation.validate_catalog(_v2_catalog(), "edge", max_models=100)
    assert normalized["schema_version"] == 2
    assert by_id["qwen-peer"]["locality"] == "local" and by_id["qwen-peer"]["base_model"] == "qwen-x"
    payload = _v2_catalog(rows=[{"direct_model_id": "x", "locality": "remote"}])
    with pytest.raises(federation.CatalogValidationError, match="invalid locality"):
        federation.validate_catalog(payload, "edge", max_models=100)
    # Schema 1 never carries the new fields: a schema 1 catalog that adds them fails its digest.
    v1 = {**_v2_catalog(), "schema_version": 1}
    with pytest.raises(federation.CatalogValidationError, match="digest"):
        federation.validate_catalog(v1, "edge", max_models=100)


def test_the_export_adds_locality_and_base_model_only_for_schema_2(tmp_path, monkeypatch):
    _configure(monkeypatch, extra={"model_fallbacks": {"text-local-mlx": "glm-native"}})
    manager = federation.FederationManager(_peer_config(tmp_path))
    monkeypatch.setattr(federation, "_manager", manager)
    v1 = client.get("/v1/federation/catalog", headers={"Authorization": f"Bearer {PEER}",
                                                         federation.SOURCE_HEADER: "edge"}).json()
    assert v1["schema_version"] == 1 and all("locality" not in m for m in v1["models"])
    v2 = client.get("/v1/federation/catalog?schema_version=2",
                    headers={"Authorization": f"Bearer {PEER}", federation.SOURCE_HEADER: "edge"}).json()
    by_id = {m["direct_model_id"]: m for m in v2["models"]}
    assert (by_id["qwen-local"]["locality"], by_id["qwen-local"]["base_model"]) == ("local", "qwen-x")
    assert by_id["qwen-local-mlx"]["locality"] == "local"
    assert by_id["glm-cloud"]["locality"] == "cloud"
    # A local model with a configured fallback can leave this Mac, so it is not local.
    assert by_id["text-local"]["locality"] == "cloud"
    # Each schema has its own revision and digest.
    assert v2["digest"] != v1["digest"]
    again = client.get("/v1/federation/catalog", headers={"Authorization": f"Bearer {PEER}",
                                                           federation.SOURCE_HEADER: "edge"}).json()
    assert again["revision"] == v1["revision"]


def test_importers_ask_for_schema_2_and_accept_an_older_peer(tmp_path):
    seen, answers = [], [_v2_catalog(revision=100), {**_v2_catalog(rows=[]), "schema_version": 1, "revision": 5,
                                                     "digest": federation.catalog_digest([])}]

    async def owner(request):
        seen.append(str(request.url))
        return httpx.Response(200, headers={federation.SOURCE_HEADER: "edge"}, json=answers.pop(0))

    async def run():
        manager = federation.FederationManager(_peer_config(tmp_path), transport=httpx.MockTransport(owner))
        assert await manager.refresh_peer("edge") is True
        assert manager.states["edge"].schema_version == 2
        # A rollback to an older release answers schema 1 with a lower revision: a new baseline.
        assert await manager.refresh_peer("edge") is True
        assert manager.states["edge"].schema_version == 1
        restored = federation.FederationManager(_peer_config(tmp_path))
        assert restored.states["edge"].schema_version == 1

    asyncio.run(run())
    assert seen[0] == "https://edge.example/v1/federation/catalog?schema_version=2"


# ── resolution through trusted peers ─────────────────────────────────────────


def test_trusted_local_route_matches_local_rows_of_fresh_trusted_peers(tmp_path):
    manager = federation.FederationManager(_peer_config(tmp_path, peers=("edge", "studio")))
    _seed(manager, _v2_catalog())
    _seed(manager, _v2_catalog(node="studio"))
    route = manager.trusted_local_route("qwen-x")
    assert (route.owner_node, route.direct_model_id, route.local) == ("edge", "qwen-peer", True)
    assert manager.trusted_local_route("big-local").direct_model_id == "big-local"
    assert manager.trusted_local_route("glm-x") is None  # cloud rows never serve local work
    _seed(manager, _v2_catalog(), fresh=False)
    assert manager.trusted_local_route("qwen-x").owner_node == "studio"  # configuration order, fresh only
    untrusted = federation.FederationManager(_peer_config(tmp_path, trusted=False))
    _seed(untrusted, _v2_catalog())
    assert untrusted.trusted_local_route("qwen-x") is None


def test_a_name_with_no_route_here_goes_to_the_peers_local_path(monkeypatch, importer):
    _configure(monkeypatch, models=[row for row in LOCAL_ROWS if row["name"] != "qwen-local"])
    response = _chat("qwen-x")
    assert response.status_code == 200 and response.json()["choices"][0]["message"]["content"] == "from edge"
    assert importer.forwarded == [("/v1/federation/local/chat/completions", "qwen-peer")]


def test_a_route_here_wins_over_a_trusted_peer(monkeypatch, importer):
    _configure(monkeypatch)
    served = []

    async def passthrough(endpoint, body, headers, provider="", request=None, **_kwargs):
        served.append(body["model"])
        return JSONResponse({"choices": []})

    monkeypatch.setattr(server, "_passthrough_sync", passthrough)
    assert _chat("qwen-x").status_code == 200
    assert served == ["qwen-local-mlx"] and importer.forwarded == []


def test_explicit_imports_still_use_the_ordinary_path(monkeypatch, importer):
    _configure(monkeypatch)
    assert _chat("edge/big-local").status_code == 200
    assert importer.forwarded == [("/v1/chat/completions", "big-local")]


def test_a_curated_local_model_never_reroutes_to_the_cloud(monkeypatch):
    cloud_qwen = {"name": "qwen3.8-27b-fw", "base_model": "qwen3.8-27b", "family": "qwen", "provider": "cloud-test",
                  "provider_model_id": "qwen-cloud", "pricing": {"input": 1.0, "output": 2.0}}
    _configure(monkeypatch, models=[*LOCAL_ROWS, cloud_qwen])
    assert providers.base_model_route("qwen3.8-27b-4bit") is None
    assert providers.base_model_route("qwen3.8-27b") is None
    # A plain base-model name still follows ADR 0002's preference.
    providers.reload()
    _configure(monkeypatch, models=[*LOCAL_ROWS, cloud_qwen])
    assert providers.resolve_base_model("qwen3.8-27b") == "qwen3.8-27b-fw"


# ── the exporter's local-only path ───────────────────────────────────────────


@pytest.fixture
def exporter(tmp_path, monkeypatch):
    """This node ("main") serving its peer "edge"."""
    manager = federation.FederationManager(_peer_config(tmp_path))
    monkeypatch.setattr(federation, "_manager", manager)
    served = []

    async def passthrough(endpoint, body, headers, provider="", request=None, **_kwargs):
        served.append((endpoint, body["model"], getattr(request.state, "disable_model_fallback", False)))
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(server, "_passthrough_sync", passthrough)
    return served


def _local(model, path="/v1/federation/local/chat/completions", headers=PEER_HEADERS, messages=None):
    return client.post(path, headers=headers,
                       json={"model": model, "messages": messages or [{"role": "user", "content": "hi"}]})


def test_the_local_path_serves_only_local_models_exactly(monkeypatch, exporter):
    _configure(monkeypatch)
    assert _local("qwen-local").status_code == 200
    assert exporter == [("http://127.0.0.1:9110/v1/chat/completions", "qwen-local-mlx", True)]
    for refused in ("glm-cloud", "qwen-x", "not-a-model"):
        response = _local(refused)
        assert response.status_code == 409, refused
    assert len(exporter) == 1


def test_the_local_path_needs_a_peer_forward(monkeypatch, exporter):
    _configure(monkeypatch)
    response = _local("qwen-local", headers={"Authorization": f"Bearer {CLIENT}"})
    assert response.status_code == 403
    response = _local("qwen-local", headers={**PEER_HEADERS, "Authorization": "Bearer wrong"})
    assert response.status_code == 401
    assert exporter == []


def test_the_local_path_serves_images_to_a_text_model_as_notes_and_never_redirects(monkeypatch, exporter):
    _configure(monkeypatch)
    image = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]
    assert _local("text-local", messages=image).status_code == 200
    assert [served[1] for served in exporter] == ["text-local-mlx"]
    monkeypatch.setattr(server, "retirement_redirect", lambda name: "qwen-local")
    assert _local("old-name").status_code == 409
    assert len(exporter) == 1


# ── trusted_local profiles ───────────────────────────────────────────────────


def _manifest(route, locality="trusted_local", revision="ha@one"):
    return {"schema_version": 1, "namespace": "ha", "source_revision": revision, "profiles": [{
        "id": "ha/assistant", "locality": locality, "credential_policy": "gateway_local",
        "protocols": ["openai_chat"], "routes": {"text": route}}]}


def _register(manifest, etag=None):
    headers = {"Authorization": f"Bearer {CONSUMER}"}
    headers.update({"If-Match": etag} if etag else {"If-None-Match": "*"})
    return client.put("/v1/profiles/ha/snapshot", headers=headers, json=manifest)


def test_a_trusted_local_profile_binds_this_macs_route_first(monkeypatch, importer):
    _configure(monkeypatch)
    response = _register(_manifest("qwen-x"))
    assert response.status_code == 200
    assert response.json()["profiles"][0]["routes"]["text"] == "qwen-local"


def test_a_trusted_local_profile_binds_and_reaches_a_peer(monkeypatch, importer):
    _configure(monkeypatch, models=[row for row in LOCAL_ROWS if row["name"] != "qwen-local"])
    response = _register(_manifest("qwen-x"))
    assert response.status_code == 200
    assert response.json()["profiles"][0]["routes"]["text"] == "edge/qwen-peer"
    assert _chat("profile:ha/assistant", token=CONSUMER).status_code == 200
    assert importer.forwarded == [("/v1/federation/local/chat/completions", "qwen-peer")]
    # Stale: unavailable, not a binding change.
    _seed(importer, _v2_catalog(), fresh=False)
    assert _chat("profile:ha/assistant", token=CONSUMER).status_code == 503
    # The peer changed the row (now a text model): re-register.
    rows = [{"direct_model_id": "qwen-peer", "locality": "local", "base_model": "qwen-x", "vision": False}]
    _seed(importer, _v2_catalog(rows=rows, revision=2))
    assert _chat("profile:ha/assistant", token=CONSUMER).status_code == 409
    # The owner stops trusting the peer.
    importer.config = replace(importer.config, peers={"edge": replace(importer.config.peers["edge"],
                                                                        trusted_local=False)})
    assert _chat("profile:ha/assistant", token=CONSUMER).status_code == 409


def test_local_only_profiles_never_bind_a_peer(monkeypatch, importer):
    _configure(monkeypatch, models=[row for row in LOCAL_ROWS if row["name"] != "qwen-local"])
    assert _register(_manifest("qwen-x", locality="local_only")).status_code == 422
    assert _register(_manifest("edge/qwen-peer", locality="local_only")).status_code == 422
    assert _register(_manifest("edge/glm-peer")).status_code == 422  # a peer's cloud row is never local


# ── review follow-ups ────────────────────────────────────────────────────────


def test_local_path_requests_are_recorded_in_the_serving_nodes_ledger(monkeypatch, exporter):
    _configure(monkeypatch)
    recorded = []
    monkeypatch.setattr(server, "_ledger_record", lambda path, method, model, *args: recorded.append((path, model)))
    assert _local("qwen-local").status_code == 200
    assert recorded == [("/v1/federation/local/chat/completions", "qwen-local")]


@pytest.mark.parametrize("path, body", [
    ("/v1/federation/local/messages", {"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/federation/local/responses", {"input": "hi"}),
])
def test_every_local_path_refuses_a_cloud_model(monkeypatch, exporter, path, body):
    _configure(monkeypatch)
    response = client.post(path, headers=PEER_HEADERS, json={"model": "glm-cloud", **body})
    assert response.status_code == 409 and exporter == []


def test_a_packs_base_model_is_never_exported(tmp_path, monkeypatch):
    _configure(monkeypatch)
    manager = federation.FederationManager(_peer_config(tmp_path))
    monkeypatch.setattr(federation, "_manager", manager)
    real = server._direct_model_rows
    monkeypatch.setattr(server, "_direct_model_rows", lambda models=None: [*real(), {"id": "pack-model"}])
    loaded = providers._load_models()
    monkeypatch.setitem(loaded, "pack-model", {"name": "pack-model", "base_model": "qwen-x",
                                                "origin": {"layer": "pack", "namespace": "ha"}})
    rows = {row["id"]: row for row in server._federation_rows(2)}
    assert "base_model" not in rows["pack-model"] and rows["qwen-local"]["base_model"] == "qwen-x"


def test_imported_rows_show_locality_only_for_trusted_peers(tmp_path):
    for trusted, shown in ((True, True), (False, False)):
        manager = federation.FederationManager(_peer_config(tmp_path, trusted=trusted))
        _seed(manager, _v2_catalog())
        row = next(row for row in manager.imported_rows() if row["direct_model_id"] == "qwen-peer")
        assert ("locality" in row) is shown and ("base_model" in row) is shown


def test_a_restricted_credential_reaches_a_trusted_peer_only_for_listed_names(monkeypatch, importer):
    config = _configure(monkeypatch, models=[row for row in LOCAL_ROWS if row["name"] != "qwen-local"])
    config["auth"]["consumer_credentials"].append({
        "id": "bot-runtime", "consumer": "bot", "key": "bot-secret", "namespaces": ["bot"],
        "permissions": ["profiles:invoke"], "allow_direct_models": True, "models": ["qwen-x"]})
    assert _chat("qwen-x", token="bot-secret").status_code == 200
    assert _chat("big-local", token="bot-secret").status_code == 403
    assert importer.forwarded == [("/v1/federation/local/chat/completions", "qwen-peer")]


def test_a_listed_local_ai_model_never_permits_its_sibling(monkeypatch):
    four_bit = {"name": "qwen3.8-27b", "base_model": "qwen3.8-27b", "family": "qwen", "provider": "omlx",
                "omlx_id": "q8", "pricing_status": "unmetered"}
    config = _configure(monkeypatch, models=[four_bit])
    config["auth"]["consumer_credentials"].append({
        "id": "bot-runtime", "consumer": "bot", "key": "bot-secret", "namespaces": ["bot"],
        "permissions": ["profiles:invoke"], "allow_direct_models": True, "models": ["qwen3.8-27b-4bit"]})
    # The 4-bit is not installed; the 8-bit serves its base model, but the credential listed only the 4-bit.
    served = []

    async def passthrough(endpoint, body, headers, provider="", request=None, **_kwargs):
        served.append(body["model"])
        return JSONResponse({"choices": []})

    monkeypatch.setattr(server, "_passthrough_sync", passthrough)
    assert providers.base_model_route("qwen3.8-27b-4bit") == "qwen3.8-27b"
    assert _chat("qwen3.8-27b-4bit", token="bot-secret").status_code == 404
    assert _chat("qwen3.8-27b-4bit").status_code == 200  # unrestricted callers are served by the 8-bit
    assert served == ["q8"]


def test_registration_needs_the_peer_reachable_now(monkeypatch, importer):
    _configure(monkeypatch, models=[row for row in LOCAL_ROWS if row["name"] != "qwen-local"])
    _seed(importer, _v2_catalog(), fresh=False)
    response = _register(_manifest("edge/qwen-peer"))
    assert response.status_code == 422 and "not reachable now" in response.text


def test_release_details_of_a_peer_row_do_not_change_its_binding(monkeypatch, importer):
    _configure(monkeypatch, models=[row for row in LOCAL_ROWS if row["name"] != "qwen-local"])
    assert _register(_manifest("qwen-x")).status_code == 200
    rows = [{"direct_model_id": "qwen-peer", "locality": "local", "base_model": "qwen-x", "vision": True,
             "forwarded_params": ["reasoning_effort"], "max_reachable": True, "created": 123}]
    _seed(importer, _v2_catalog(rows=rows, revision=2))
    assert _chat("profile:ha/assistant", token=CONSUMER).status_code == 200


def test_the_cache_holds_both_local_catalogs_and_every_peer(tmp_path, monkeypatch):
    config = replace(_peer_config(tmp_path), max_catalog_bytes=2_000)
    manager = federation.FederationManager(config)
    _seed(manager, _v2_catalog())
    rows = [{"id": f"model-{index}", "locality": "local"} for index in range(6)]
    manager.build_catalog(rows, 1)
    manager.build_catalog(rows, 2)
    restored = federation.FederationManager(config)
    assert set(restored._local) == {1, 2}
    assert restored.states["edge"].schema_version == 2 and "qwen-peer" in restored.states["edge"].models
