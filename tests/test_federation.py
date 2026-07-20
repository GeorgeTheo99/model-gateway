"""Focused federation MVP contract tests."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from src import federation
import src.admin as admin
import src.server as server


def _config(tmp_path: Path, *, node_id: str = "main", transport_peer: str = "edge") -> federation.FederationConfig:
    peer = federation.PeerConfig(transport_peer, "https://edge.example", "peer-secret")
    return federation.FederationConfig(
        node_id=node_id,
        peers={transport_peer: peer},
        refresh_interval_seconds=60,
        stale_after_seconds=30,
        request_timeout_seconds=2,
        stream_idle_timeout_seconds=900,
        cache_path=tmp_path / "federation-cache.json",
        max_catalog_bytes=100_000,
        max_models_per_peer=100,
    )


def _catalog(node_id: str = "edge", direct_ids: tuple[str, ...] = ("model-a",), revision: int = 1) -> dict:
    models = [
        {
            "direct_model_id": direct_id,
            "owner_node": node_id,
            "object": "model",
            "created": 0,
            "thinking": "optional",
            "thinking_format": "openai",
            "thinking_levels": ["none", "high"],
            "max_reachable": True,
            "forwarded_params": ["reasoning_effort"],
            "vision": False,
        }
        for direct_id in direct_ids
    ]
    return {
        "schema_version": 1,
        "node_id": node_id,
        "revision": revision,
        "digest": federation.catalog_digest(models),
        "generated_at": "2026-01-01T00:00:00Z",
        "models": models,
    }


def _seed(manager: federation.FederationManager, payload: dict | None = None) -> None:
    payload = payload or _catalog()
    normalized, by_id = federation.validate_catalog(
        payload,
        payload["node_id"],
        max_models=manager.config.max_models_per_peer,
    )
    state = manager.states[payload["node_id"]]
    state.models = by_id
    state.revision = normalized["revision"]
    state.digest = normalized["digest"]
    state.generated_at = normalized["generated_at"]
    state.last_attempt_at = time.time()
    state.last_success_at = time.time()
    state.healthy = True
    state.last_error = ""


@pytest.fixture
def installed_manager(tmp_path, monkeypatch):
    manager = federation.FederationManager(_config(tmp_path))
    _seed(manager)
    monkeypatch.setattr(federation, "_manager", manager)
    return manager


def test_config_is_optional(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("providers: {}\n")
    assert federation.load_config(config) is None


def test_config_loads_key_file_and_settings(tmp_path):
    key_file = tmp_path / "edge.key"
    key_file.write_text("file-secret\n")
    os.chmod(key_file, 0o600)
    config = tmp_path / "config.yaml"
    config.write_text(
        """federation:
  node_id: main
  refresh_interval_seconds: 12
  stale_after_seconds: 34
  request_timeout_seconds: 5
  stream_idle_timeout_seconds: 456
  cache_path: state/cache.json
  max_catalog_bytes: 12345
  max_models_per_peer: 99
  peers:
    edge:
      base_url: https://edge.example/gateway/
      api_key_file: edge.key
"""
    )
    loaded = federation.load_config(config)
    assert loaded.node_id == "main"
    assert loaded.refresh_interval_seconds == 12
    assert loaded.stale_after_seconds == 34
    assert loaded.request_timeout_seconds == 5
    assert loaded.stream_idle_timeout_seconds == 456
    assert loaded.cache_path == tmp_path / "state/cache.json"
    assert loaded.peers["edge"].base_url == "https://edge.example/gateway"
    assert loaded.peers["edge"].api_key == "file-secret"
    assert "file-secret" not in repr(loaded.peers["edge"])


def test_config_rejects_cache_path_resolving_to_config_without_modifying_it(tmp_path):
    config_target = tmp_path / "real-config.yaml"
    config_link = tmp_path / "federation.yaml"
    config_target.write_text(
        """federation:
  node_id: main
  cache_path: federation.yaml
  peers:
    edge: {base_url: https://edge.example, api_key: peer-secret}
"""
    )
    config_link.symlink_to(config_target.name)
    original = config_target.read_bytes()

    with pytest.raises(federation.FederationConfigError, match="configuration file"):
        federation.load_config(config_link)

    assert config_target.read_bytes() == original
    assert config_link.is_symlink()


def test_config_rejects_cache_path_resolving_to_peer_key_without_modifying_it(tmp_path):
    key_target = tmp_path / "edge.key"
    key_link = tmp_path / "edge-key-link"
    key_target.write_text("peer-secret\n")
    os.chmod(key_target, 0o600)
    key_link.symlink_to(key_target.name)
    config = tmp_path / "config.yaml"
    config.write_text(
        """federation:
  node_id: main
  cache_path: edge-key-link
  peers:
    edge:
      base_url: https://edge.example
      api_key_file: edge.key
"""
    )
    original_config = config.read_bytes()
    original_key = key_target.read_bytes()

    with pytest.raises(federation.FederationConfigError, match="api_key_file"):
        federation.load_config(config)

    assert config.read_bytes() == original_config
    assert key_target.read_bytes() == original_key
    assert key_link.is_symlink()


def test_config_rejects_cache_path_resolving_to_missing_provider_key(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """federation:
  node_id: main
  cache_path: secrets/cloud.key
  peers: {}
providers:
  cloud:
    api_key_file: secrets/cloud.key
"""
    )

    with pytest.raises(federation.FederationConfigError, match="provider 'cloud' api_key_file"):
        federation.load_config(config)

    assert not (tmp_path / "secrets/cloud.key").exists()


def test_config_rejects_cache_symlink_resolving_to_workspace_key_without_modifying_it(tmp_path):
    credentials = tmp_path / "credentials"
    credentials.mkdir()
    key_target = credentials / "workspace.key"
    key_target.write_text("workspace-secret\n")
    cache_link = tmp_path / "federation-cache-link"
    cache_link.symlink_to(Path("credentials/workspace.key"))
    config = tmp_path / "config.yaml"
    config.write_text(
        """federation:
  node_id: main
  cache_path: federation-cache-link
  peers: {}
workspaces:
  analytics:
    api_key_file: credentials/workspace.key
"""
    )
    original_key = key_target.read_bytes()

    with pytest.raises(federation.FederationConfigError, match="workspace 'analytics' api_key_file"):
        federation.load_config(config)

    assert key_target.read_bytes() == original_key
    assert cache_link.is_symlink()


@pytest.mark.parametrize(
    "yaml_text,match",
    [
        ("federation: {node_id: Main, peers: {}}", "lowercase DNS label"),
        ("federation: {node_id: main, peers: {main: {base_url: https://x, api_key: k}}}", "cannot be this node"),
        ("federation: {node_id: main, peers: {edge: {base_url: ftp://x, api_key: k}}}", "HTTP\\(S\\)"),
        ("federation: {node_id: main, peers: {edge: {base_url: https://x, api_key: k, api_key_file: x}}}", "exactly one"),
        ("federation: {node_id: main, peers: {edge: {base_url: https://x}}}", "exactly one"),
        ("federation: {node_id: main, peers: {}, unexpected: true}", "unknown federation"),
    ],
)
def test_config_rejects_unsafe_or_ambiguous_values(tmp_path, yaml_text, match):
    config = tmp_path / "config.yaml"
    config.write_text(yaml_text)
    with pytest.raises(federation.FederationConfigError, match=match):
        federation.load_config(config)


def test_catalog_validation_accepts_direct_ids_with_arbitrary_slashes():
    direct_ids = ("org/models/model-a", "edge/model-a", "main/model-a")
    payload = _catalog(direct_ids=direct_ids)
    normalized, models = federation.validate_catalog(payload, "edge", max_models=10)
    assert normalized["revision"] == 1
    assert list(models) == list(direct_ids)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p.update(schema_version=2), "schema_version"),
        (lambda p: p.update(node_id="wrong"), "does not match"),
        (lambda p: p["models"][0].update(owner_node="wrong"), "wrong owner"),
        (lambda p: p["models"].append(dict(p["models"][0])), "duplicate"),
    ],
)
def test_catalog_validation_rejects_bad_schema_owner_ids_and_duplicates(mutate, match):
    payload = _catalog()
    mutate(payload)
    payload["digest"] = federation.catalog_digest(payload["models"])
    with pytest.raises(federation.CatalogValidationError, match=match):
        federation.validate_catalog(payload, "edge", max_models=10)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_catalog_validation_rejects_nonfinite_model_numbers(value):
    payload = _catalog()
    payload["models"][0]["created"] = value
    with pytest.raises(federation.CatalogValidationError, match="created"):
        federation.validate_catalog(payload, "edge", max_models=10)


def test_catalog_validation_rejects_digest_mismatch_limits_and_rollback():
    payload = _catalog()
    payload["digest"] = "0" * 64
    with pytest.raises(federation.CatalogValidationError, match="digest"):
        federation.validate_catalog(payload, "edge", max_models=10)

    payload = _catalog(direct_ids=("a", "b"))
    with pytest.raises(federation.CatalogValidationError, match="model-count"):
        federation.validate_catalog(payload, "edge", max_models=1)

    payload = _catalog(revision=2)
    with pytest.raises(federation.CatalogValidationError, match="backwards"):
        federation.validate_catalog(payload, "edge", max_models=10, previous_revision=3)


def test_catalog_endpoint_fails_closed_and_checks_source_peer(tmp_path, monkeypatch):
    manager = federation.FederationManager(_config(tmp_path))
    monkeypatch.setattr(federation, "_manager", manager)
    monkeypatch.setattr(server, "_direct_model_rows", lambda: [{
        "id": "local/model",
        "object": "model",
        "created": 0,
        "owned_by": "local-provider",
        "thinking": "",
        "thinking_format": "none",
        "thinking_levels": [],
        "max_reachable": False,
        "forwarded_params": [],
        "vision": False,
    }])
    client = TestClient(server.app)

    assert client.get("/v1/federation/catalog").status_code == 400
    assert client.get(
        "/v1/federation/catalog",
        headers={federation.SOURCE_HEADER: "edge", "Authorization": "Bearer wrong"},
    ).status_code == 401
    response = client.get(
        "/v1/federation/catalog",
        headers={federation.SOURCE_HEADER: "edge", "Authorization": "Bearer peer-secret"},
    )
    assert response.status_code == 200
    assert response.headers[federation.SOURCE_HEADER] == "main"
    assert response.json()["models"][0]["direct_model_id"] == "local/model"
    assert "base_url" not in response.text
    assert "peer-secret" not in response.text

    # Ordinary client auth being open never opens the federation catalog.
    monkeypatch.setattr(federation, "_manager", federation.FederationManager())
    assert client.get("/v1/federation/catalog").status_code == 401


def test_catalog_revision_is_stable_across_restart_and_advances_on_change(tmp_path):
    config = _config(tmp_path)
    manager = federation.FederationManager(config)
    first = manager.build_catalog([{"id": "a"}])
    again = manager.build_catalog([{"id": "a"}])
    restarted = federation.FederationManager(config)
    after_restart = restarted.build_catalog([{"id": "a"}])
    changed = restarted.build_catalog([{"id": "b"}])
    assert first == again == after_restart
    assert changed["revision"] > first["revision"]
    assert changed["digest"] != first["digest"]


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf", "1.0e+999"])
def test_config_rejects_nonfinite_numbers(tmp_path, value):
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""federation:
  node_id: main
  stream_idle_timeout_seconds: {value}
  peers: {{}}
"""
    )
    with pytest.raises(federation.FederationConfigError, match="positive finite"):
        federation.load_config(config)


def test_config_wraps_yaml_parser_and_read_errors(tmp_path):
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("federation: [unterminated\n")
    with pytest.raises(federation.FederationConfigError, match="could not read federation configuration"):
        federation.load_config(malformed)
    with pytest.raises(federation.FederationConfigError, match="could not read federation configuration"):
        federation.load_config(tmp_path)


def test_lkg_cache_is_atomic_secret_free_and_visible_after_restart(tmp_path):
    config = _config(tmp_path)
    manager = federation.FederationManager(config)
    _seed(manager, _catalog(direct_ids=("org/model",)))
    manager._write_cache()

    cache_text = config.cache_path.read_text()
    assert "peer-secret" not in cache_text
    assert "edge.example" not in cache_text
    assert not list(tmp_path.glob(".*.tmp.*"))
    assert config.cache_path.stat().st_mode & 0o077 == 0

    restored = federation.FederationManager(config)
    route = restored.resolve_imported("edge/org/model")
    assert route is not None
    assert route.direct_model_id == "org/model"
    row = restored.imported_rows()[0]
    assert row["id"] == "edge/org/model"
    assert row["available"] is False
    assert row["federated"] is True


def test_cached_local_catalog_ignores_low_inbound_peer_limit(tmp_path):
    config = replace(_config(tmp_path), max_models_per_peer=1)
    local_rows = [{"id": "local/a"}, {"id": "local/b"}]
    manager = federation.FederationManager(config)
    original = manager.build_catalog(local_rows)

    restored = federation.FederationManager(config)
    assert restored.build_catalog(local_rows) == original
    assert [model["direct_model_id"] for model in restored._catalog_models] == ["local/a", "local/b"]


def test_cache_restores_valid_sections_atomically_beside_corrupt_peer(tmp_path):
    config = replace(
        _config(tmp_path),
        peers={
            "edge": federation.PeerConfig("edge", "https://edge.example", "edge-secret"),
            "west": federation.PeerConfig("west", "https://west.example", "west-secret"),
        },
    )
    manager = federation.FederationManager(config)
    local = manager.build_catalog([{"id": "local/a"}])
    _seed(manager, _catalog("edge", ("edge/a",), revision=4))
    _seed(manager, _catalog("west", ("west/a",), revision=7))
    manager._write_cache()

    document = json.loads(config.cache_path.read_text())
    document["peers"]["edge"]["digest"] = "0" * 64
    config.cache_path.write_text(json.dumps(document))

    restored = federation.FederationManager(config)
    assert restored._catalog_revision == local["revision"]
    assert restored.resolve_imported("west/west/a") is not None
    edge_state = restored.states["edge"]
    assert edge_state.models == {}
    assert edge_state.revision is None
    assert edge_state.last_success_at is None


@pytest.mark.parametrize("poison", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_poisoned_cache_is_rejected_and_models_endpoint_stays_healthy(tmp_path, monkeypatch, poison):
    config = _config(tmp_path)
    manager = federation.FederationManager(config)
    _seed(manager)
    manager._write_cache()
    document = json.loads(config.cache_path.read_text())
    document["peers"]["edge"]["last_success_at"] = "POISON"
    raw = json.dumps(document).replace('"POISON"', poison)
    config.cache_path.write_text(raw)

    restored = federation.FederationManager(config)
    assert restored.resolve_imported("edge/model-a") is None
    monkeypatch.setattr(federation, "_manager", restored)
    monkeypatch.setattr(server, "_direct_model_rows", lambda: [])
    monkeypatch.setattr(server, "list_routable_models", lambda: [])
    response = TestClient(server.app).get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def test_cache_isolates_peer_with_finite_unrepresentable_timestamp(tmp_path, monkeypatch):
    config = replace(
        _config(tmp_path),
        peers={
            "edge": federation.PeerConfig("edge", "https://edge.example", "edge-secret"),
            "west": federation.PeerConfig("west", "https://west.example", "west-secret"),
        },
    )
    manager = federation.FederationManager(config)
    _seed(manager, _catalog("edge", ("edge/a",)))
    _seed(manager, _catalog("west", ("west/a",)))
    manager._write_cache()
    document = json.loads(config.cache_path.read_text())
    document["peers"]["edge"]["last_success_at"] = 1e20
    config.cache_path.write_text(json.dumps(document))

    restored = federation.FederationManager(config)
    assert restored.resolve_imported("edge/edge/a") is None
    assert restored.resolve_imported("west/west/a") is not None
    monkeypatch.setattr(federation, "_manager", restored)
    monkeypatch.setattr(server, "_direct_model_rows", lambda: [])
    monkeypatch.setattr(server, "list_routable_models", lambda: [])
    response = TestClient(server.app).get("/v1/models")
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["data"]] == ["west/west/a"]


def test_cache_rejects_nonfinite_timestamp_before_restoration(tmp_path):
    config = replace(
        _config(tmp_path),
        peers={
            "edge": federation.PeerConfig("edge", "https://edge.example", "edge-secret"),
            "west": federation.PeerConfig("west", "https://west.example", "west-secret"),
        },
    )
    manager = federation.FederationManager(config)
    _seed(manager, _catalog("edge", ("edge/a",)))
    _seed(manager, _catalog("west", ("west/a",)))
    manager._write_cache()
    document = json.loads(config.cache_path.read_text())
    document["peers"]["edge"]["last_attempt_at"] = float("inf")
    # Use the standards-compliant overflow spelling to exercise parse_float.
    raw = json.dumps(document).replace("Infinity", "1e999")
    config.cache_path.write_text(raw)

    restored = federation.FederationManager(config)
    assert restored.resolve_imported("edge/edge/a") is None
    # Strict JSON numeric parsing rejects the poisoned document before any
    # section can be trusted.
    assert restored.resolve_imported("west/west/a") is None


def test_peer_refresh_uses_explicit_identity_and_accepts_valid_catalog(tmp_path):
    captured = {}

    async def owner(request):
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            headers={federation.SOURCE_HEADER: "edge"},
            json=_catalog(direct_ids=("org/model",)),
        )

    async def run():
        manager = federation.FederationManager(_config(tmp_path), transport=httpx.MockTransport(owner))
        assert await manager.refresh_peer("edge") is True
        assert manager.resolve_imported("edge/org/model") is not None
        assert manager.imported_rows()[0]["available"] is True

    asyncio.run(run())
    assert captured["headers"]["authorization"] == "Bearer peer-secret"
    assert captured["headers"][federation.SOURCE_HEADER.lower()] == "main"


@pytest.mark.parametrize("poison", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_refresh_preserves_lkg_and_models_endpoint_health(tmp_path, monkeypatch, poison):
    payload = _catalog(revision=2)
    raw = json.dumps(payload).replace('"created": 0', f'"created": {poison}', 1).encode()

    async def owner(_request):
        return httpx.Response(200, headers={federation.SOURCE_HEADER: "edge"}, content=raw)

    manager = federation.FederationManager(_config(tmp_path), transport=httpx.MockTransport(owner))
    _seed(manager)
    assert asyncio.run(manager.refresh_peer("edge")) is False
    assert manager.resolve_imported("edge/model-a") is not None
    monkeypatch.setattr(federation, "_manager", manager)
    monkeypatch.setattr(server, "_direct_model_rows", lambda: [])
    monkeypatch.setattr(server, "list_routable_models", lambda: [])

    response = TestClient(server.app).get("/v1/models")
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["data"]] == ["edge/model-a"]


def test_peer_refresh_enforces_payload_byte_limit(tmp_path):
    async def owner(_request):
        return httpx.Response(
            200,
            headers={federation.SOURCE_HEADER: "edge"},
            content=b"{" + (b"x" * 100) + b"}",
        )

    async def run():
        config = replace(_config(tmp_path), max_catalog_bytes=10)
        manager = federation.FederationManager(config, transport=httpx.MockTransport(owner))
        assert await manager.refresh_peer("edge") is False
        assert manager.states["edge"].last_error == "invalid_catalog"

    asyncio.run(run())


def test_transient_refresh_failure_keeps_lkg_and_marks_unavailable(tmp_path):
    async def run():
        async def fail(_request):
            raise httpx.ConnectError("host and secret must not leak")

        manager = federation.FederationManager(_config(tmp_path), transport=httpx.MockTransport(fail))
        _seed(manager)
        assert await manager.refresh_peer("edge") is False
        assert manager.resolve_imported("edge/model-a") is not None
        row = manager.imported_rows()[0]
        assert row["available"] is False
        assert row["stale"] is False
        assert row["status"]["state"] == "unreachable"
        assert "host" not in json.dumps(row["status"])
        stale_row = manager.imported_rows(now=manager.states["edge"].last_success_at + 31)[0]
        assert stale_row["stale"] is True
        assert stale_row["status"]["state"] == "stale"

    asyncio.run(run())


def test_models_appends_namespaced_import_without_changing_local_rows(installed_manager, monkeypatch):
    local_row = {
        "id": "local-a",
        "object": "model",
        "created": 0,
        "owned_by": "local",
        "thinking": "",
        "thinking_format": "none",
        "thinking_levels": [],
        "max_reachable": False,
        "forwarded_params": [],
        "vision": False,
    }
    monkeypatch.setattr(server, "_direct_model_rows", lambda: [dict(local_row)])
    response = TestClient(server.app).get("/v1/models")
    assert response.status_code == 200
    rows = response.json()["data"]
    assert rows[0] == local_row
    imported = rows[1]
    assert imported["id"] == "edge/model-a"
    assert imported["owner_node"] == "edge"
    assert imported["direct_model_id"] == "model-a"
    assert imported["available"] is True
    assert imported["stale"] is False
    assert imported["federated"] is True
    assert installed_manager.resolve_imported("model-a") is None


class _TrackedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class _FailingStream(httpx.AsyncByteStream):
    def __init__(self, error: Exception):
        self.error = error
        self.closed = False

    async def __aiter__(self):
        raise self.error
        yield b""  # pragma: no cover - keeps this an async generator

    async def aclose(self):
        self.closed = True


class _TimeoutStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield b"first"
        raise httpx.ReadTimeout("idle")

    async def aclose(self):
        self.closed = True


class _BlockingStream(httpx.AsyncByteStream):
    def __init__(self):
        self.waiting = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield b"first"
        self.waiting.set()
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


class _DripStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], delay: float):
        self.chunks = chunks
        self.delay = delay
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            yield chunk

    async def aclose(self):
        self.closed = True


def test_catalog_refresh_hard_deadline_stops_slow_drip_and_closes_response(tmp_path):
    async def run():
        raw = json.dumps(_catalog()).encode()
        drip = _DripStream([raw[index:index + 8] for index in range(0, len(raw), 8)], 0.01)

        async def owner(_request):
            return httpx.Response(200, headers={federation.SOURCE_HEADER: "edge"}, stream=drip)

        config = replace(_config(tmp_path), request_timeout_seconds=0.025)
        manager = federation.FederationManager(config, transport=httpx.MockTransport(owner))
        assert await manager.refresh_peer("edge") is False
        assert manager.states["edge"].last_error == "timeout"
        assert drip.closed

    asyncio.run(run())


def test_stream_idle_timeout_is_configured_and_finite(tmp_path):
    async def run():
        manager = federation.FederationManager(replace(_config(tmp_path), stream_idle_timeout_seconds=0.25))
        client = manager._client(streaming=True)
        try:
            assert client.timeout.read == 0.25
        finally:
            await client.aclose()

    asyncio.run(run())


def test_forward_send_timeout_maps_to_504_and_closes_client(tmp_path):
    async def run():
        async def owner(request):
            raise httpx.ReadTimeout("idle", request=request)

        manager = federation.FederationManager(_config(tmp_path))
        _seed(manager)
        client = httpx.AsyncClient(transport=httpx.MockTransport(owner))
        manager._client = lambda **_kwargs: client
        response = await manager.forward(
            None, "/v1/chat/completions", {"model": "edge/model-a"}, manager.resolve_imported("edge/model-a")
        )
        assert response.status_code == 504
        assert json.loads(response.body) == {
            "error": {"type": "federation_error", "message": "Federated owner request timed out"}
        }
        assert client.is_closed

    asyncio.run(run())


def test_forward_header_hard_deadline_maps_builtin_timeout_to_504_and_closes_client(tmp_path):
    async def run():
        entered = asyncio.Event()

        async def owner(_request):
            entered.set()
            await asyncio.sleep(1)

        manager = federation.FederationManager(replace(_config(tmp_path), request_timeout_seconds=0.02))
        _seed(manager)
        client = httpx.AsyncClient(transport=httpx.MockTransport(owner))
        manager._client = lambda **_kwargs: client
        response = await manager.forward(
            None,
            "/v1/chat/completions",
            {"model": "edge/model-a", "stream": True},
            manager.resolve_imported("edge/model-a"),
        )
        assert entered.is_set()
        assert response.status_code == 504
        assert json.loads(response.body)["error"]["type"] == "federation_error"
        assert client.is_closed

    asyncio.run(run())


def test_forward_body_timeout_maps_to_504_and_closes_response_client(tmp_path):
    async def run():
        failing = _FailingStream(httpx.ReadTimeout("idle"))

        async def owner(_request):
            return httpx.Response(200, stream=failing)

        manager = federation.FederationManager(_config(tmp_path))
        _seed(manager)
        client = httpx.AsyncClient(transport=httpx.MockTransport(owner))
        manager._client = lambda **_kwargs: client
        response = await manager.forward(
            None, "/v1/messages", {"model": "edge/model-a"}, manager.resolve_imported("edge/model-a")
        )
        assert response.status_code == 504
        assert json.loads(response.body) == {
            "type": "error",
            "error": {"type": "federation_error", "message": "Federated owner response timed out"},
        }
        assert failing.closed
        assert client.is_closed

    asyncio.run(run())


def test_nonstream_forward_hard_deadline_stops_slow_drip(tmp_path):
    async def run():
        drip = _DripStream([b"a", b"b", b"c", b"d"], 0.01)

        async def owner(_request):
            return httpx.Response(200, stream=drip)

        manager = federation.FederationManager(replace(_config(tmp_path), request_timeout_seconds=0.025))
        _seed(manager)
        client = httpx.AsyncClient(transport=httpx.MockTransport(owner))
        manager._client = lambda **_kwargs: client
        response = await manager.forward(
            None, "/v1/responses", {"model": "edge/model-a"}, manager.resolve_imported("edge/model-a")
        )
        assert response.status_code == 504
        assert json.loads(response.body) == {
            "error": {"type": "federation_error", "message": "Federated owner response timed out"}
        }
        assert drip.closed
        assert client.is_closed

    asyncio.run(run())


def test_forward_cancellation_during_initial_send_closes_client(tmp_path):
    async def run():
        entered = asyncio.Event()

        async def owner(_request):
            entered.set()
            await asyncio.Event().wait()

        manager = federation.FederationManager(_config(tmp_path))
        _seed(manager)
        client = httpx.AsyncClient(transport=httpx.MockTransport(owner))
        manager._client = lambda **_kwargs: client
        task = asyncio.create_task(manager.forward(
            None, "/v1/chat/completions", {"model": "edge/model-a"}, manager.resolve_imported("edge/model-a")
        ))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.is_closed

    asyncio.run(run())


def test_post_header_stream_timeout_truncates_and_closes_response_client(tmp_path):
    async def run():
        timed = _TimeoutStream()

        async def owner(_request):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=timed)

        manager = federation.FederationManager(_config(tmp_path))
        _seed(manager)
        client = httpx.AsyncClient(transport=httpx.MockTransport(owner))
        manager._client = lambda **_kwargs: client
        response = await manager.forward(
            None,
            "/v1/chat/completions",
            {"model": "edge/model-a", "stream": True},
            manager.resolve_imported("edge/model-a"),
        )
        assert [chunk async for chunk in response.body_iterator] == [b"first"]
        assert timed.closed
        assert client.is_closed

    asyncio.run(run())


def test_downstream_stream_cancellation_closes_response_and_client(tmp_path):
    async def run():
        blocking = _BlockingStream()

        async def owner(_request):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=blocking)

        manager = federation.FederationManager(_config(tmp_path))
        _seed(manager)
        client = httpx.AsyncClient(transport=httpx.MockTransport(owner))
        manager._client = lambda **_kwargs: client
        response = await manager.forward(
            None,
            "/v1/chat/completions",
            {"model": "edge/model-a", "stream": True},
            manager.resolve_imported("edge/model-a"),
        )
        iterator = response.body_iterator
        assert await anext(iterator) == b"first"
        pending = asyncio.create_task(anext(iterator))
        await blocking.waiting.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert blocking.closed
        assert client.is_closed

    asyncio.run(run())


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "/v1/messages"])
@pytest.mark.parametrize("streaming", [False, True])
def test_all_api_paths_forward_raw_with_only_model_rewritten(tmp_path, monkeypatch, path, streaming):
    captured = {}
    tracked = _TrackedStream([b"data: {\"part\":1}\n\n", b"data: [DONE]\n\n"])

    async def owner(request: httpx.Request):
        captured["path"] = request.url.path
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(await request.aread())
        if streaming:
            return httpx.Response(206, headers={"content-type": "text/event-stream", "x-owner-request": "abc"}, stream=tracked)
        return httpx.Response(207, headers={"content-type": "application/x-owner", "x-owner-request": "abc"}, content=b"owner-raw-body")

    manager = federation.FederationManager(_config(tmp_path), transport=httpx.MockTransport(owner))
    _seed(manager)
    monkeypatch.setattr(federation, "_manager", manager)
    monkeypatch.setattr(server, "resolve", lambda _model: None)
    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS", "client-secret")

    body = {
        "model": "edge/model-a",
        "stream": streaming,
        "messages": [{"role": "user", "content": "unchanged"}],
        "reasoning_effort": "max",
        "gateway_image_handling": "leave-alone",
        "arbitrary": {"nested": [1, True, None]},
    }
    client = TestClient(server.app)
    headers = {
        "Authorization": "Bearer client-secret",
        "x-client-only": "must-not-forward",
    }
    with client.stream("POST", path, json=body, headers=headers) as response:
        raw = response.read()
        assert response.status_code == (206 if streaming else 207)
        assert response.headers["x-owner-request"] == "abc"

    expected = dict(body)
    expected["model"] = "model-a"
    assert captured["body"] == expected
    assert captured["path"] == path
    assert captured["headers"]["authorization"] == "Bearer peer-secret"
    assert captured["headers"][federation.SOURCE_HEADER.lower()] == "main"
    assert captured["headers"][federation.OWNER_HEADER.lower()] == "edge"
    assert captured["headers"]["via"] == "1.1 main-model-gateway"
    assert "x-client-only" not in captured["headers"]
    assert "client-secret" not in json.dumps(captured)
    assert raw == (b"data: {\"part\":1}\n\ndata: [DONE]\n\n" if streaming else b"owner-raw-body")
    assert tracked.closed is streaming


@pytest.mark.parametrize(
    "path,payload,expected",
    [
        (
            "/v1/chat/completions",
            b'{"model":"edge/model-a","messages":[],"temperature":NaN}',
            {
                "error": {
                    "type": "invalid_request_error",
                    "message": "Federated request body is not valid finite UTF-8 JSON",
                }
            },
        ),
        (
            "/v1/responses",
            b'{"model":"edge/model-a","input":Infinity}',
            {
                "error": {
                    "type": "invalid_request_error",
                    "message": "Federated request body is not valid finite UTF-8 JSON",
                }
            },
        ),
        (
            "/v1/messages",
            b'{"model":"edge/model-a","messages":[{"role":"user","content":"\\ud800"}],"max_tokens":1}',
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Federated request body is not valid finite UTF-8 JSON",
                },
            },
        ),
    ],
)
def test_imported_requests_reject_nonfinite_or_non_utf8_json_before_client_send(
    tmp_path, monkeypatch, path, payload, expected
):
    sent = False

    async def owner(_request):
        nonlocal sent
        sent = True
        return httpx.Response(200, content=b"unexpected")

    manager = federation.FederationManager(_config(tmp_path), transport=httpx.MockTransport(owner))
    _seed(manager)
    client_created = False

    def forbidden_client(**_kwargs):
        nonlocal client_created
        client_created = True
        raise AssertionError("invalid imported body must not allocate an HTTP client")

    manager._client = forbidden_client
    monkeypatch.setattr(federation, "_manager", manager)
    monkeypatch.setattr(server, "resolve", lambda _model: None)
    monkeypatch.setattr(server, "model_availability", lambda _model: {"reason": "model_not_found"})

    response = TestClient(server.app).post(path, content=payload, headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert response.json() == expected
    assert client_created is False
    assert sent is False


def _local_info():
    return SimpleNamespace(
        provider="local-provider",
        base_url="http://local-upstream",
        api_key="local-key",
        provider_model_id="upstream-direct",
        protocol="openai",
        endpoint_suffix=None,
        quirks=frozenset(),
        context=0,
        max_output_tokens=0,
        thinking="",
        thinking_format="none",
        system_instruction="",
        vision=False,
        pricing=None,
    )


@pytest.mark.parametrize("model", ["main/direct-model", "edge/direct-model"])
@pytest.mark.parametrize(
    "path,body",
    [
        ("/v1/chat/completions", {"messages": []}),
        ("/v1/responses", {"input": "hello"}),
        ("/v1/messages", {"messages": [], "max_tokens": 1}),
    ],
)
def test_authenticated_peer_handlers_accept_slash_direct_ids(tmp_path, monkeypatch, model, path, body):
    manager = federation.FederationManager(_config(tmp_path))
    monkeypatch.setattr(federation, "_manager", manager)
    info = _local_info()
    info.provider = "openai"
    monkeypatch.setattr(server, "resolve", lambda candidate: info if candidate == model else None)

    async def served(*_args, **_kwargs):
        return server.JSONResponse(content={"served": model})

    monkeypatch.setattr(server, "_passthrough_sync", served)
    monkeypatch.setattr(server, "_handle_openai_responses_passthrough", served)
    monkeypatch.setattr(server, "_handle_sync", served)
    headers = {
        "Authorization": "Bearer peer-secret",
        federation.SOURCE_HEADER: "edge",
        federation.OWNER_HEADER: "main",
        "Via": "1.1 edge-model-gateway",
    }
    response = TestClient(server.app).post(path, json={"model": model, **body}, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"served": model}


def test_peer_forward_errors_use_endpoint_specific_envelopes(tmp_path, monkeypatch):
    manager = federation.FederationManager(_config(tmp_path))
    monkeypatch.setattr(federation, "_manager", manager)
    monkeypatch.setattr(server, "resolve", lambda _model: None)
    monkeypatch.setattr(server, "model_availability", lambda _model: {
        "available": False,
        "reason": "model_not_found",
        "message": "not found",
    })
    client = TestClient(server.app)
    valid = {
        federation.SOURCE_HEADER: "edge",
        federation.OWNER_HEADER: "main",
        "Via": "1.1 edge-model-gateway",
    }

    unauthorized = client.post(
        "/v1/chat/completions",
        json={"model": "missing", "messages": []},
        headers={**valid, "Authorization": "Bearer wrong"},
    )
    assert unauthorized.status_code == 401
    assert unauthorized.json() == {
        "error": {
            "type": "authentication_error",
            "message": "missing or invalid federation peer credential",
        }
    }

    wrong_owner = client.post(
        "/v1/messages",
        json={"model": "missing", "messages": [], "max_tokens": 1},
        headers={**valid, federation.OWNER_HEADER: "edge", "Authorization": "Bearer peer-secret"},
    )
    assert wrong_owner.status_code == 400
    assert wrong_owner.json() == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "federation forward has the wrong owner"},
    }

    not_found = client.post(
        "/v1/responses",
        json={"model": "missing", "input": "hello"},
        headers={**valid, "Authorization": "Bearer peer-secret"},
    )
    assert not_found.status_code == 404
    assert not_found.json() == {
        "error": {"type": "invalid_request_error", "message": "Model 'missing' not found in gateway"}
    }


def test_forward_connection_and_timeout_errors_use_api_envelopes(tmp_path, monkeypatch):
    async def unreachable(request):
        if request.url.path == "/v1/messages":
            raise httpx.ConnectError("down", request=request)
        raise httpx.ReadTimeout("idle", request=request)

    manager = federation.FederationManager(_config(tmp_path), transport=httpx.MockTransport(unreachable))
    _seed(manager)
    monkeypatch.setattr(federation, "_manager", manager)
    monkeypatch.setattr(server, "resolve", lambda _model: None)
    monkeypatch.setattr(server, "model_availability", lambda _model: {
        "available": False,
        "reason": "model_not_found",
        "message": "not found",
    })
    client = TestClient(server.app)

    unavailable = client.post(
        "/v1/messages",
        json={"model": "edge/model-a", "messages": [], "max_tokens": 1},
    )
    assert unavailable.status_code == 502
    assert unavailable.json() == {
        "type": "error",
        "error": {"type": "federation_error", "message": "Cannot connect to federated model owner"},
    }

    timed_out = client.post(
        "/v1/chat/completions",
        json={"model": "edge/model-a", "messages": []},
    )
    assert timed_out.status_code == 504
    assert timed_out.json() == {
        "error": {"type": "federation_error", "message": "Federated owner request timed out"}
    }


def test_ordinary_client_auth_keeps_existing_fastapi_detail_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(federation, "_manager", federation.FederationManager(_config(tmp_path)))
    monkeypatch.setenv("MODEL_GATEWAY_CLIENT_KEYS", "client-secret")
    response = TestClient(server.app).post(
        "/v1/chat/completions", json={"model": "missing", "messages": []}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing or invalid model-gateway client key"}


def test_owner_peer_auth_and_direct_route_validation(tmp_path, monkeypatch):
    manager = federation.FederationManager(_config(tmp_path))
    monkeypatch.setattr(federation, "_manager", manager)
    info = _local_info()
    monkeypatch.setattr(server, "resolve", lambda model: info if model in {"direct-model", "edge/direct-model"} else None)

    async def local_handler(endpoint, body, headers, **_kwargs):
        assert endpoint == "http://local-upstream/chat/completions"
        assert body["model"] == "upstream-direct"
        return server.JSONResponse(content={"served": "locally"})

    monkeypatch.setattr(server, "_passthrough_sync", local_handler)
    client = TestClient(server.app)
    good_headers = {
        "Authorization": "Bearer peer-secret",
        federation.SOURCE_HEADER: "edge",
        federation.OWNER_HEADER: "main",
        "Via": "1.1 edge-model-gateway",
    }
    body = {"model": "direct-model", "messages": []}
    assert client.post("/v1/chat/completions", json=body, headers=good_headers).json() == {"served": "locally"}

    bad_auth = dict(good_headers, Authorization="Bearer wrong")
    assert client.post("/v1/chat/completions", json=body, headers=bad_auth).status_code == 401
    wrong_owner = dict(good_headers)
    wrong_owner[federation.OWNER_HEADER] = "edge"
    assert client.post("/v1/chat/completions", json=body, headers=wrong_owner).status_code == 400
    repeated = dict(good_headers, Via="1.1 edge-model-gateway, 1.1 other")
    assert client.post("/v1/chat/completions", json=body, headers=repeated).status_code == 400
    slash_id = dict(body, model="edge/direct-model")
    assert client.post("/v1/chat/completions", json=slash_id, headers=good_headers).json() == {"served": "locally"}
    unknown = dict(body, model="unknown")
    assert client.post("/v1/chat/completions", json=unknown, headers=good_headers).status_code == 404


def test_imported_listing_does_not_duplicate_a_local_identifier(installed_manager, monkeypatch):
    local_row = {"id": "edge/model-a", "object": "model", "created": 0, "owned_by": "local"}
    monkeypatch.setattr(server, "_direct_model_rows", lambda: [local_row])
    monkeypatch.setattr(server, "list_routable_models", lambda: [{"id": "edge/model-a"}])
    rows = TestClient(server.app).get("/v1/models").json()["data"]
    assert rows == [local_row]


def test_unavailable_local_route_still_wins_over_import(installed_manager, monkeypatch):
    monkeypatch.setattr(server, "resolve", lambda _model: None)
    monkeypatch.setattr(server, "model_availability", lambda _model: {
        "available": False,
        "reason": "provider_disabled",
        "message": "local route is disabled",
    })
    response = TestClient(server.app).post(
        "/v1/chat/completions",
        json={"model": "edge/model-a", "messages": []},
    )
    assert response.status_code == 404
    assert "local route is disabled" in response.json()["error"]["message"]


def test_local_routing_wins_over_same_imported_id(installed_manager, monkeypatch):
    info = _local_info()
    monkeypatch.setattr(server, "resolve", lambda model: info if model == "edge/model-a" else None)

    async def local_handler(_endpoint, body, _headers, **_kwargs):
        assert body["model"] == "upstream-direct"
        return server.JSONResponse(content={"route": "local"})

    monkeypatch.setattr(server, "_passthrough_sync", local_handler)
    response = TestClient(server.app).post(
        "/v1/chat/completions",
        json={"model": "edge/model-a", "messages": []},
    )
    assert response.status_code == 200
    assert response.json() == {"route": "local"}


@pytest.mark.parametrize(
    "contents",
    [
        "federation: [unterminated\n",
        "federation: {node_id: main, peers: {}, unexpected: true}\n",
    ],
)
def test_failed_admin_reload_preserves_provider_and_federation_state(tmp_path, monkeypatch, contents):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(contents)
    previous = federation.FederationManager(_config(tmp_path))
    _seed(previous)
    monkeypatch.setattr(federation, "_manager", previous)
    monkeypatch.setattr(admin.config_io, "CONFIG_PATH", config_path)
    provider_reload_calls = []
    monkeypatch.setattr(admin, "reload_provider_registry", lambda: provider_reload_calls.append(True))
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin-secret")
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")

    response = TestClient(server.app).post(
        "/admin/api/reload", headers={"Authorization": "Bearer admin-secret"}
    )
    assert response.status_code == 400
    assert provider_reload_calls == []
    assert federation.manager() is previous
    assert federation.manager().resolve_imported("edge/model-a") is not None


def test_admin_reload_passes_prevalidated_federation_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("providers: {}\n")
    validated = _config(tmp_path)
    captured = {}

    async def reconfigure(*, config):
        captured["config"] = config
        return {"enabled": True}

    async def regenerate():
        return "regenerated"

    monkeypatch.setattr(admin.config_io, "CONFIG_PATH", config_path)
    monkeypatch.setattr(admin.federation, "load_config", lambda path: validated)
    monkeypatch.setattr(admin.federation, "reconfigure", reconfigure)
    monkeypatch.setattr(admin, "reload_provider_registry", lambda: captured.setdefault("provider_reloaded", True))
    monkeypatch.setattr(admin, "_regenerate_catalogs", regenerate)
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_KEY", "admin-secret")
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")

    response = TestClient(server.app).post(
        "/admin/api/reload", headers={"Authorization": "Bearer admin-secret"}
    )
    assert response.status_code == 200
    assert captured == {"provider_reloaded": True, "config": validated}


def test_activation_orders_replacement_behind_previous_cache_revision(tmp_path, monkeypatch):
    async def run():
        config = _config(tmp_path)
        release_old = asyncio.Event()
        release_replacement = asyncio.Event()
        old_started = asyncio.Event()
        replacement_started = asyncio.Event()
        old_revision_persisted = asyncio.Event()
        request_count = 0

        async def owner(_request):
            nonlocal request_count
            request_count += 1
            if request_count == 1:
                old_started.set()
                try:
                    await release_old.wait()
                except asyncio.CancelledError:
                    # Model a response that became ready while shutdown was
                    # cancelling the old background refresh.
                    pass
                payload = _catalog(direct_ids=("fresh",), revision=2)
            else:
                replacement_started.set()
                await release_replacement.wait()
                payload = _catalog(direct_ids=("stale",), revision=1)
            return httpx.Response(200, headers={federation.SOURCE_HEADER: "edge"}, json=payload)

        transport = httpx.MockTransport(owner)

        def client(_self, *, streaming=False):
            return httpx.AsyncClient(transport=transport, timeout=config.request_timeout_seconds)

        monkeypatch.setattr(federation.FederationManager, "_client", client)
        previous = federation.FederationManager(config)
        _seed(previous, _catalog(direct_ids=("baseline",), revision=0))
        previous._write_cache()
        monkeypatch.setattr(federation, "_manager", previous)

        original_write_cache = federation.FederationManager._write_cache

        def tracked_write_cache(manager):
            original_write_cache(manager)
            if manager is previous and manager.states["edge"].revision == 2:
                old_revision_persisted.set()

        monkeypatch.setattr(federation.FederationManager, "_write_cache", tracked_write_cache)
        previous._refresh_task = asyncio.create_task(
            previous.refresh_peer("edge"), name="federation-catalog-refresh"
        )
        await old_started.wait()

        activation = asyncio.create_task(federation._activate_config(config))
        await replacement_started.wait()
        assert federation.manager() is previous

        # With the old lifecycle order, the replacement has already loaded
        # revision 0 here and is waiting to publish its revision-1 response.
        # The previous manager then writes revision 2 first.
        release_old.set()
        await old_revision_persisted.wait()
        assert previous.states["edge"].revision == 2
        assert previous.resolve_imported("edge/fresh") is not None
        revision_two_cache = json.loads(config.cache_path.read_text())
        assert revision_two_cache["peers"]["edge"]["revision"] == 2

        # A request retaining the handed-off manager may finish, but cannot
        # replace the shared cache after ownership has been relinquished.
        previous.states["edge"].last_error = "retired-write"
        previous._write_cache()
        assert json.loads(config.cache_path.read_text())["peers"]["edge"]["last_error"] == ""

        release_replacement.set()
        await activation

        current = federation.manager()
        final_cache = json.loads(config.cache_path.read_text())
        assert current is not previous
        assert current.states["edge"].revision == 2
        assert current.resolve_imported("edge/fresh") is not None
        assert current.resolve_imported("edge/stale") is None
        assert final_cache["peers"]["edge"]["revision"] == 2
        assert [model["direct_model_id"] for model in final_cache["peers"]["edge"]["models"]] == ["fresh"]

        await federation.stop()

    asyncio.run(run())


def test_cancelled_activation_restores_previous_cache_owner_and_refresh(tmp_path, monkeypatch):
    async def run():
        config = _config(tmp_path)
        replacement_started = asyncio.Event()

        async def owner(_request):
            replacement_started.set()
            await asyncio.Event().wait()

        transport = httpx.MockTransport(owner)

        def client(_self, *, streaming=False):
            return httpx.AsyncClient(transport=transport, timeout=config.request_timeout_seconds)

        monkeypatch.setattr(federation.FederationManager, "_client", client)
        previous = federation.FederationManager(config)
        _seed(previous, _catalog(direct_ids=("baseline",), revision=1))
        previous._write_cache()
        previous.start_background_refresh()
        monkeypatch.setattr(federation, "_manager", previous)

        activation = asyncio.create_task(federation._activate_config(config))
        await replacement_started.wait()
        assert federation.manager() is previous
        assert previous._refresh_task is None

        activation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await activation

        assert federation.manager() is previous
        assert previous._refresh_task is not None
        assert not previous._refresh_task.done()
        _seed(previous, _catalog(direct_ids=("restored",), revision=2))
        previous._write_cache()
        restored_cache = json.loads(config.cache_path.read_text())
        assert restored_cache["peers"]["edge"]["revision"] == 2
        assert restored_cache["peers"]["edge"]["models"][0]["direct_model_id"] == "restored"

        await federation.stop()
        assert previous._refresh_task is None
        _seed(previous, _catalog(direct_ids=("after-stop",), revision=3))
        previous._write_cache()
        assert json.loads(config.cache_path.read_text()) == restored_cache

    asyncio.run(run())


def test_concurrent_activation_and_reload_leave_only_current_refresh_task(tmp_path, monkeypatch):
    async def run():
        stop_entered = asyncio.Event()
        release_stop = asyncio.Event()

        class BlockingManager(federation.FederationManager):
            async def stop(self):
                stop_entered.set()
                await release_stop.wait()
                await super().stop()

        async def refresh_without_network(self):
            await asyncio.sleep(0)

        previous = BlockingManager()
        monkeypatch.setattr(federation, "_manager", previous)
        monkeypatch.setattr(federation.FederationManager, "refresh_all", refresh_without_network)
        first_config = replace(_config(tmp_path), node_id="first")
        second_config = replace(_config(tmp_path), node_id="second")

        activation = asyncio.create_task(federation._activate_config(first_config))
        await stop_entered.wait()
        reload_task = asyncio.create_task(federation.reconfigure(config=second_config))
        await asyncio.sleep(0)
        release_stop.set()
        await asyncio.gather(activation, reload_task)

        current = federation.manager()
        refresh_tasks = [
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "federation-catalog-refresh" and not task.done()
        ]
        assert current.config.node_id == "second"
        assert refresh_tasks == [current._refresh_task]

        await federation.stop()
        assert current._refresh_task is None
        assert not [
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "federation-catalog-refresh" and not task.done()
        ]

    asyncio.run(run())


def test_background_refresh_stops_cleanly(tmp_path):
    async def run():
        config = replace(_config(tmp_path), peers={})
        manager = federation.FederationManager(config)
        manager._refresh_task = asyncio.create_task(manager.run_refresh_loop())
        await manager.stop()
        assert manager._refresh_task is None

    asyncio.run(run())
