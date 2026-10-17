"""Owner-added OpenAI-compatible providers (providers:custom)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src import config_io, custom_providers, onboarding, server
import src.providers as providers
from src.server import app

ADMIN = {"Authorization": "Bearer admin"}
CUSTOM = {"Authorization": "Bearer custom-token"}
OTHER = {"Authorization": "Bearer other-token"}
MANAGER = {"Authorization": "Bearer manager-token"}
URL = "/admin/api/custom-providers"
BOX = "https://llm.example.com/v1"
ADDRESSES = {"llm.example.com": {"93.184.216.34"}, "lan.example.com": {"192.168.1.9"}}


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "auth": {
            "admin_keys": ["admin"],
            "consumer_credentials": [
                {"id": "myai-manager", "consumer": "myai", "key": "custom-token", "namespaces": ["myai"],
                 "permissions": ["providers:manage", "models:register", "providers:custom"],
                 "providers": ["fireworks"]},
                {"id": "ha-custom", "consumer": "ha", "key": "other-token", "namespaces": ["ha"],
                 "permissions": ["providers:custom"]},
                {"id": "ha-manager", "consumer": "ha", "key": "manager-token", "namespaces": ["ha"],
                 "permissions": ["providers:manage", "models:register"], "providers": ["fireworks"]},
            ],
        },
        "providers": {"openai": {"base_url": "https://api.openai.com/v1", "protocol": "openai"}},
    }))
    path.chmod(0o600)
    info = tmp_path / "model-info.json"
    info.write_text(json.dumps({"allow_empty": True, "llm": [
        {"name": "gpt-6-luna", "provider": "openai", "provider_model_id": "gpt-6-luna"},
    ]}))
    for module in (providers, config_io):
        monkeypatch.setattr(module, "CONFIG_PATH", path)
        monkeypatch.setattr(module, "MODEL_INFO_PATH", info)
        monkeypatch.setattr(module, "MODEL_INFO_SOURCE_PATH", None)
    monkeypatch.delenv("MODEL_GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "true")
    config_io.log_dir = tmp_path / "logs"
    providers.reload()
    return path


@pytest.fixture
def addresses(monkeypatch):
    """Resolve hosts from a table instead of DNS."""
    table = dict(ADDRESSES)

    def resolve(host, port):
        if host in {"localhost", "127.0.0.1"}:
            return {"127.0.0.1"}
        if host in table:
            return set(table[host])
        try:
            return {str(__import__("ipaddress").ip_address(host))}
        except ValueError:
            raise custom_providers.CustomProviderError("unreachable", f"Can't find {host}") from None

    monkeypatch.setattr(custom_providers, "_resolve", resolve)
    return table


@pytest.fixture
def upstream(monkeypatch, addresses):
    """A provider listing two models that answers only with the right key."""
    calls = []

    def discover(url, key):
        calls.append(("models", url, key))
        if key == "bad-key":
            raise custom_providers.CustomProviderError("key_rejected", "The provider refused the key (HTTP 401)")
        return ["llama-4-70b", "gpt-6-luna"]

    def completion(profile, key):
        calls.append(("completion", profile["provider"]["base_url"], profile["models"][0]["provider_model_id"], key))
        if key == "no-credit-key":
            raise onboarding.OnboardingError("provider refused a test completion (HTTP 402): no-credit-key spent")

    monkeypatch.setattr(custom_providers, "discover", discover)
    monkeypatch.setattr(onboarding, "validate_upstream_completion", completion)
    return calls


@pytest.fixture
def client(cfg):
    with TestClient(app) as c:
        yield c


def _state(cfg):
    return cfg.read_bytes(), (cfg.parent / "model-info.json").read_bytes()


def _add(client, headers=CUSTOM, **fields):
    body = {"name": "My Box", "base_url": BOX, "api_key": "box-key", "models": ["llama-4-70b"], **fields}
    return client.post(URL, headers=headers, json=body)


# ── destinations ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("url", "allow_lan", "network"), [
    ("https://llm.example.com/v1", False, "public"),
    ("HTTPS://LLM.Example.com/v1/", False, "public"),
    ("http://localhost:11434/v1", False, "loopback"),
    ("http://127.0.0.1:1234/v1/", False, "loopback"),
    ("http://[::1]:1234/v1", False, "loopback"),
    ("http://192.168.1.20:11434/v1", True, "lan"),
    ("https://10.0.0.5/v1", True, "lan"),
    ("http://100.101.102.103:8080/v1", True, "lan"),
    ("http://[fd12::1]:8080/v1", True, "lan"),
    ("https://lan.example.com/v1", True, "lan"),
])
def test_allowed_destinations(addresses, url, allow_lan, network):
    assert custom_providers.check_destination(url, allow_lan) == network


@pytest.mark.parametrize(("url", "allow_lan", "code"), [
    ("http://llm.example.com/v1", False, "https_required"),
    ("HTTP://llm.example.com/v1", False, "https_required"),
    ("http://93.184.216.34/v1", False, "https_required"),
    # A host name gets plain http nowhere, so a DNS answer that changes later can't redirect it.
    ("http://lan.example.com/v1", True, "https_required"),
    ("https://lan.example.com/v1", False, "lan_not_allowed"),
    ("http://[fd00:ec2::254]/latest", True, "blocked_address"),
    ("http://100.100.100.200/latest", True, "blocked_address"),
    ("https://llm.exa mple.com/v1", False, "invalid_url"),
    ("https://exämple.com/v1", False, "invalid_url"),
    ("http://192.168.1.20:11434/v1", False, "lan_not_allowed"),
    ("http://100.101.102.103/v1", False, "lan_not_allowed"),
    ("http://169.254.169.254/latest", True, "blocked_address"),
    ("http://[::ffff:169.254.169.254]/latest", True, "blocked_address"),
    ("http://[fe80::1]/v1", True, "blocked_address"),
    ("http://0.0.0.0:8080/v1", True, "blocked_address"),
    ("https://224.0.0.1/v1", True, "blocked_address"),
    ("ftp://llm.example.com/v1", False, "invalid_url"),
    ("https://user:pass@llm.example.com/v1", False, "invalid_url"),
    ("https://llm.example.com/v1?key=x", False, "invalid_url"),
    ("https://llm.example.com/v1#x", False, "invalid_url"),
    ("llm.example.com/v1", False, "invalid_url"),
    ("https://nowhere.invalid/v1", False, "unreachable"),
])
def test_refused_destinations(addresses, url, allow_lan, code):
    with pytest.raises(custom_providers.CustomProviderError) as raised:
        custom_providers.check_destination(url, allow_lan)
    assert raised.value.code == code


def test_a_host_with_any_blocked_address_is_refused(addresses):
    addresses["mixed.example.com"] = {"93.184.216.34", "169.254.169.254"}
    with pytest.raises(custom_providers.CustomProviderError, match="never connects"):
        custom_providers.check_destination("https://mixed.example.com/v1", True)


def test_requests_recheck_the_saved_address_without_dns(cfg, monkeypatch):
    monkeypatch.setattr(custom_providers, "_resolve", lambda *_: pytest.fail("requests must not resolve"))

    def save(base_url):
        config = yaml.safe_load(cfg.read_text())
        config["providers"]["custom-box"] = {"base_url": base_url, "managed_by": "custom", "allow_lan": False}
        cfg.write_text(yaml.safe_dump(config))
        providers.reload()
        return SimpleNamespace(provider="custom-box", base_url=base_url, endpoint_suffix=None)

    info = save(BOX)
    assert server._upstream_endpoint(info, "/chat/completions") == f"{BOX}/chat/completions"
    for edited in ("http://169.254.169.254/v1", "http://192.168.1.20/v1", "http://llm.example.com/v1"):
        with pytest.raises(HTTPException) as raised:
            server._upstream_endpoint(save(edited), "/chat/completions")
        assert raised.value.status_code == 502 and "custom-box" in raised.value.detail


# ── credentials ──────────────────────────────────────────────────────────────


def test_custom_providers_need_their_own_permission(client, upstream, cfg):
    before = _state(cfg)
    assert _add(client, MANAGER).status_code == 403
    assert client.get(URL, headers=MANAGER).status_code == 403
    assert _state(cfg) == before and upstream == []


def test_the_permission_needs_no_provider_allowlist(cfg):
    row = config_io.add_consumer_credential("pi", "runtime", custom_providers=True)
    assert row["permissions"] == ["profiles:read", "profiles:invoke", "providers:custom"]


# ── adding ───────────────────────────────────────────────────────────────────


def test_discover_lists_models_and_writes_nothing(client, upstream, cfg):
    before = _state(cfg)
    response = client.post(f"{URL}/discover", headers=CUSTOM, json={"base_url": BOX + "/", "api_key": "box-key"})
    assert response.status_code == 200, response.text
    assert response.json() == {"base_url": BOX, "network": "public", "models": ["llama-4-70b", "gpt-6-luna"]}
    assert _state(cfg) == before


def test_add_saves_an_owned_provider_after_a_test_completion(client, upstream, cfg):
    response = _add(client)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "id": "custom-my-box", "display_name": "My Box", "base_url": BOX, "allow_lan": False, "has_api_key": True,
        "models": [{"name": "custom-my-box-llama-4-70b", "upstream_id": "llama-4-70b"}],
    }
    assert upstream == [("models", BOX, "box-key"), ("completion", BOX, "llama-4-70b", "box-key")]
    block = yaml.safe_load(cfg.read_text())["providers"]["custom-my-box"]
    assert {key: block[key] for key in ("base_url", "protocol", "managed_by", "owner", "display_name", "allow_lan")} == {
        "base_url": BOX, "protocol": "openai", "managed_by": "custom", "owner": "myai-manager",
        "display_name": "My Box", "allow_lan": False,
    }
    assert "api_key" not in block and config_io.api_key_file_target("custom-my-box").read_text() == "box-key\n"
    info = providers.resolve("custom-my-box-llama-4-70b")
    assert (info.provider, info.provider_model_id) == ("custom-my-box", "llama-4-70b")
    assert providers.resolve("llama-4-70b") is None


def test_an_upstream_id_never_takes_over_another_route(client, upstream, cfg):
    assert _add(client, models=["gpt-6-luna"]).status_code == 200
    routes = providers._load_models()
    assert routes["gpt-6-luna"]["provider"] == "openai"
    assert routes["custom-my-box-gpt-6-luna"]["provider"] == "custom-my-box"


def test_a_keyless_local_server_needs_no_key_and_routes(client, upstream, cfg):
    from src.request_options import upstream_auth_headers

    response = _add(client, base_url="http://localhost:11434/v1", api_key=None, name="Ollama")
    assert response.status_code == 200, response.text
    assert response.json()["has_api_key"] is False and upstream[-1][-1] == ""
    info = providers.resolve("custom-ollama-llama-4-70b")
    assert (info.provider, info.base_url, info.api_key) == ("custom-ollama", "http://localhost:11434/v1", "")
    assert upstream_auth_headers(info.api_key, info.protocol) == {}


def test_refusals_write_nothing(client, upstream, cfg):
    before = _state(cfg)
    cases = [
        ({"models": ["missing-model"]}, "model_not_listed"),
        ({"api_key": "bad-key"}, "key_rejected"),
        ({"api_key": "no-credit-key"}, "upstream_rejected"),
        ({"base_url": "http://192.168.1.20:11434/v1"}, "lan_not_allowed"),
        ({"base_url": "http://169.254.169.254/v1", "allow_lan": True}, "blocked_address"),
        ({"name": ""}, "invalid_name"),
        ({"models": []}, "invalid_models"),
        ({"api_key": "has space"}, "invalid_key"),
    ]
    for fields, code in cases:
        response = _add(client, **fields)
        assert (response.status_code, response.json()["code"]) == (400, code), (fields, response.text)
        assert "no-credit-key" not in response.text
    assert _state(cfg) == before


def test_names_are_numbered_and_the_count_is_capped(client, upstream, cfg, monkeypatch):
    assert _add(client).json()["id"] == "custom-my-box"
    assert _add(client).json()["id"] == "custom-my-box-2"
    monkeypatch.setattr(custom_providers, "MAX_PROVIDERS", 2)
    response = _add(client)
    assert (response.status_code, response.json()["code"]) == (409, "too_many")


def test_writes_disabled_refuses(client, upstream, cfg, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_ADMIN_WRITES", "false")
    assert _add(client).status_code == 403 and upstream == []


# ── owning ───────────────────────────────────────────────────────────────────


def test_only_the_creator_or_a_full_admin_sees_and_changes_it(client, upstream, cfg):
    assert _add(client).status_code == 200
    assert [row["id"] for row in client.get(URL, headers=CUSTOM).json()["providers"]] == ["custom-my-box"]
    assert client.get(URL, headers=OTHER).json() == {"providers": []}
    assert [row["id"] for row in client.get(URL, headers=ADMIN).json()["providers"]] == ["custom-my-box"]
    before = _state(cfg)
    for response in (
        client.delete(f"{URL}/custom-my-box", headers=OTHER),
        client.post(f"{URL}/custom-my-box/key", headers=OTHER, json={"api_key": "x"}),
        client.post(f"{URL}/custom-my-box/models", headers=OTHER, json={"models": ["gpt-6-luna"]}),
        client.delete(f"{URL}/openai", headers=CUSTOM),
    ):
        assert (response.status_code, response.json()["code"]) == (404, "not_found")
    assert _state(cfg) == before


def test_new_ids_avoid_every_provider_in_use(client, upstream, cfg, monkeypatch):
    monkeypatch.setenv("MODEL_GATEWAY_PROVIDER_CUSTOM_MY_BOX_BASE_URL", "https://elsewhere.example/v1")
    assert _add(client).json()["id"] == "custom-my-box-2"


def test_a_provider_has_a_model_cap(client, upstream, cfg, monkeypatch):
    assert _add(client).status_code == 200
    monkeypatch.setattr(custom_providers, "MAX_MODELS", 1)
    response = client.post(f"{URL}/custom-my-box/models", headers=CUSTOM, json={"models": ["gpt-6-luna"]})
    assert (response.status_code, response.json()["code"]) == (409, "too_many")


def test_add_models_adds_only_new_ones(client, upstream, cfg):
    assert _add(client).status_code == 200
    response = client.post(f"{URL}/custom-my-box/models", headers=CUSTOM, json={"models": ["llama-4-70b", "gpt-6-luna"]})
    assert response.status_code == 200, response.text
    assert [row["upstream_id"] for row in response.json()["models"]] == ["llama-4-70b", "gpt-6-luna"]
    assert upstream[-1] == ("completion", BOX, "gpt-6-luna", "box-key")


def test_available_lists_models_with_the_saved_key(client, upstream, cfg):
    assert _add(client).status_code == 200
    response = client.get(f"{URL}/custom-my-box/available", headers=CUSTOM)
    assert response.status_code == 200 and response.json() == {"models": ["llama-4-70b", "gpt-6-luna"]}
    assert upstream[-1] == ("models", BOX, "box-key")
    assert client.get(f"{URL}/custom-my-box/available", headers=OTHER).status_code == 404


def test_key_changes_only_after_the_new_key_answers(client, upstream, cfg):
    assert _add(client).status_code == 200
    key_file = config_io.api_key_file_target("custom-my-box")
    response = client.post(f"{URL}/custom-my-box/key", headers=CUSTOM, json={"api_key": "no-credit-key"})
    assert response.status_code == 400 and key_file.read_text() == "box-key\n"
    response = client.post(f"{URL}/custom-my-box/key", headers=CUSTOM, json={"api_key": "new-key"})
    assert response.status_code == 200 and key_file.read_text() == "new-key\n"


def test_delete_removes_models_provider_and_key(client, upstream, cfg):
    assert _add(client, models=["llama-4-70b", "gpt-6-luna"]).status_code == 200
    key_file = config_io.api_key_file_target("custom-my-box")
    response = client.delete(f"{URL}/custom-my-box", headers=CUSTOM)
    assert response.status_code == 200, response.text
    assert "custom-my-box" not in yaml.safe_load(cfg.read_text())["providers"]
    assert [row["name"] for row in json.loads((cfg.parent / "model-info.json").read_text())["llm"]] == ["gpt-6-luna"]
    assert not key_file.exists() and providers.resolve("custom-my-box-llama-4-70b") is None


# ── upstream calls ───────────────────────────────────────────────────────────


def _stream(monkeypatch, response, sent=None):
    from contextlib import contextmanager

    @contextmanager
    def stream(method, url, *, headers, timeout):
        if sent is not None:
            sent.append((method, url, dict(headers)))
        yield response

    monkeypatch.setattr(custom_providers.httpx, "stream", stream)


def test_discover_reads_the_model_list_without_following_redirects(monkeypatch):
    import httpx

    sent = []
    listing = {"data": [{"id": "a"}, {"id": "a"}, {"id": "bad id"}, {"id": 3}, {"id": "b"}]}
    _stream(monkeypatch, httpx.Response(200, json=listing), sent)
    assert custom_providers.discover(BOX, "") == ["a", "b"]
    assert sent[0][:2] == ("GET", f"{BOX}/models") and "Authorization" not in sent[0][2]
    assert custom_providers.discover(BOX, "k") == ["a", "b"] and sent[1][2]["Authorization"] == "Bearer k"
    for response, code in (
        (httpx.Response(401), "key_rejected"),
        (httpx.Response(302, headers={"Location": "http://169.254.169.254/"}), "not_openai_compatible"),
        (httpx.Response(200, text="<html>"), "not_openai_compatible"),
        (httpx.Response(200, json={"data": []}), "no_models"),
        (httpx.Response(200, content=b"x" * 2_000_001), "not_openai_compatible"),
    ):
        _stream(monkeypatch, response)
        with pytest.raises(custom_providers.CustomProviderError) as raised:
            custom_providers.discover(BOX, "k")
        assert raised.value.code == code


def test_a_keyless_test_completion_sends_no_authorization(monkeypatch):
    import httpx

    sent = []

    def post(url, *, json, headers, timeout):
        sent.append(headers)
        return httpx.Response(400, json={"error": {"message": "model not loaded"}})

    monkeypatch.setattr(onboarding.httpx, "post", post)
    probe = {"provider": {"id": "custom-probe", "base_url": BOX, "protocol": "openai"},
             "models": [{"provider_model_id": "llama-4-70b"}]}
    with pytest.raises(onboarding.OnboardingError, match=r"HTTP 400\): model not loaded$"):
        onboarding.validate_upstream_completion(probe, "")
    assert "Authorization" not in sent[0]
