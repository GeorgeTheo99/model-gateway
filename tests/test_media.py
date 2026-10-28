"""Image and video generation routes (docs/adr/0004-local-image-and-video-generation.md)."""

from __future__ import annotations

import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient

from src import catalog, ledger, media, providers, server

client = TestClient(server.app)
ALICE = "alice-secret"
BOB = "bob-secret"
LIMITED = "limited-secret"
MEDIA_KEY = "media-gen-key"

CATALOG = [
    {"name": "chat-a", "provider": "cloud-test", "provider_model_id": "chat-native",
     "pricing": {"input": 1.0, "output": 2.0}},
    {"name": "pic", "alias": "pic-alias", "provider": "media", "provider_model_id": "z-image-turbo",
     "modality": "image", "pricing_status": "unmetered", "display_name": "Picture"},
    {"name": "clip", "provider": "media", "provider_model_id": "ltx-2.3-distilled",
     "modality": "video", "pricing_status": "unmetered"},
]


def _credential(credential_id: str, key: str, **extra) -> dict:
    return {"id": credential_id, "consumer": credential_id, "key": key, "namespaces": [credential_id],
            "permissions": ["profiles:read"], "allow_direct_models": True, **extra}


class FakeMediaGen:
    """Enough of media-gen's API to see what the gateway forwards."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.videos: dict[str, dict] = {}
        self.fail_with: httpx.Response | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.host == "media.test"
        assert request.headers["authorization"] == f"Bearer {MEDIA_KEY}"
        if self.fail_with is not None:
            return self.fail_with
        path, method = request.url.path, request.method
        if path == "/v1/images/generations":
            body = json.loads(request.content)
            return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "aGk="}],
                                             "model_seen": body["model"]})
        if path == "/v1/videos" and method == "POST":
            video_id = f"video_{len(self.videos) + 1}"
            if request.headers["content-type"].startswith("multipart/form-data"):
                model = "multipart"
                assert b'name="input_reference"' in request.content
            else:
                model = json.loads(request.content)["model"]
            self.videos[video_id] = {"id": video_id, "object": "video", "model": model, "status": "queued"}
            return httpx.Response(200, json=self.videos[video_id])
        video_id = path.split("/")[3]
        if video_id not in self.videos:
            return httpx.Response(404, json={"error": {"message": "gone", "code": "video_not_found"}})
        if path.endswith("/content"):
            return httpx.Response(200, content=b"\x00\x00\x00\x18ftypmp42", headers={"content-type": "video/mp4"})
        if method == "DELETE":
            del self.videos[video_id]
            return httpx.Response(200, json={"id": video_id, "object": "video.deleted", "deleted": True})
        return httpx.Response(200, json=self.videos[video_id])


@pytest.fixture
def media_gen(monkeypatch):
    config = {
        "providers": {
            "cloud-test": {"base_url": "https://cloud.invalid/v1", "api_key": "secret"},
            "media": {"base_url": "http://media.test/v1", "api_key": MEDIA_KEY},
        },
        "models": deepcopy(CATALOG),
        "auth": {"client_keys": ["legacy-one", "legacy-two"], "consumer_credentials": [
            _credential("alice", ALICE),
            _credential("bob", BOB),
            _credential("limited", LIMITED, models=["chat-a", "clip"]),
        ]},
    }
    monkeypatch.setattr(providers, "_config", config)
    monkeypatch.setattr(providers, "_models", None)
    fake = FakeMediaGen()
    monkeypatch.setattr(media, "_transport", httpx.MockTransport(fake.handler))
    return fake


def _auth(key: str = ALICE) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _image(model: str = "pic", key: str = ALICE, **extra):
    return client.post("/v1/images/generations", headers=_auth(key), json={"model": model, "prompt": "x", **extra})


def _video(model: str = "clip", key: str = ALICE):
    return client.post("/v1/videos", headers=_auth(key), json={"model": model, "prompt": "waves"})


def test_modality_is_validated():
    with pytest.raises(ValueError, match="modality"):
        catalog.validate_client_metadata({"name": "x", "modality": "audio"})
    assert catalog.modality({"name": "x"}) == "chat"
    assert catalog.validate_pricing_policy({"name": "x", "provider": "media", "pricing_status": "unmetered"}) \
        == "unmetered"


def test_image_generation_is_forwarded_with_the_provider_model_id(media_gen):
    response = _image(model="pic-alias", seed=3)
    assert response.status_code == 200, response.text
    assert response.json()["model_seen"] == "z-image-turbo"
    [forwarded] = media_gen.requests
    assert str(forwarded.url) == "http://media.test/v1/images/generations"
    assert json.loads(forwarded.content)["seed"] == 3


def test_generation_is_recorded_in_the_ledger(media_gen):
    _image()
    _video()
    with ledger._connect() as conn:
        recorded = conn.execute(
            "SELECT endpoint, model, provider, status FROM requests ORDER BY ts DESC LIMIT 2"
        ).fetchall()
    assert [tuple(r) for r in recorded] == [("/v1/videos", "clip", "media", 200),
                                            ("/v1/images/generations", "pic", "media", 200)]


def test_wrong_modality_and_unknown_models_are_refused(media_gen):
    assert _image(model="clip").json()["error"]["code"] == "wrong_modality"
    assert _video(model="pic").json()["error"]["code"] == "wrong_modality"
    assert _image(model="chat-a").status_code == 400
    assert _image(model="nope").status_code == 404
    assert _image(model="profile:alice/draw").status_code == 400
    assert media_gen.requests == []


def test_chat_endpoints_refuse_media_models(media_gen):
    for path in ("/v1/chat/completions", "/v1/responses"):
        response = client.post(path, headers=_auth(), json={"model": "pic", "input": "hi",
                                                            "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 400
        assert "/v1/images/generations" in response.json()["error"]["message"]
    response = client.post("/v1/messages", headers=_auth(), json={
        "model": "clip", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 400
    assert "/v1/videos" in response.json()["error"]["message"]
    assert media_gen.requests == []


def test_allowlists_apply(media_gen):
    assert _image(key=LIMITED).status_code == 403
    assert _video(key=LIMITED).status_code == 200


def test_media_is_hidden_from_default_discovery(media_gen):
    media_ids = {"pic", "pic-alias", "z-image-turbo", "clip", "ltx-2.3-distilled"}
    ids = {row["id"] for row in client.get("/v1/models", headers=_auth()).json()["data"]}
    assert "chat-a" in ids
    assert not ids & media_ids
    canonical = client.get("/v1/models/canonical", headers=_auth()).json()["data"]
    assert "chat-a" in {row["id"] for row in canonical}
    assert not {row["id"] for row in canonical} & media_ids
    assert all("modality" not in row for row in canonical)
    thinking = client.get("/v1/debug/thinking", headers=_auth()).json()
    assert not {row["name"] for row in thinking["models"]} & media_ids


def test_media_is_listed_on_request(media_gen):
    rows = {row["id"]: row for row in
            client.get("/v1/models?include=modality", headers=_auth()).json()["data"]}
    assert {k: rows[k]["modality"] for k in ("chat-a", "pic", "pic-alias", "clip")} == {
        "chat-a": "chat", "pic": "image", "pic-alias": "image", "clip": "video"}
    assert all("modality" in row for row in rows.values())
    canonical = {row["id"]: row for row in
                 client.get("/v1/models/canonical?include=modality,display_name", headers=_auth()).json()["data"]}
    assert canonical["pic"]["modality"] == "image"
    assert canonical["pic"]["scope"] == "local"
    assert canonical["pic"]["display_name"] == "Picture"
    assert canonical["chat-a"]["modality"] == "chat"
    limited = client.get("/v1/models?include=modality", headers=_auth(LIMITED)).json()["data"]
    # Routable ids include each route's provider model id, as for chat rows.
    assert {row["id"] for row in limited} == {"chat-a", "chat-native", "clip", "ltx-2.3-distilled"}
    limited_canonical = client.get("/v1/models/canonical?include=modality", headers=_auth(LIMITED)).json()
    assert {row["id"] for row in limited_canonical["data"]} == {"chat-a", "clip"}


def test_video_jobs_belong_to_their_creator(media_gen):
    created = _video()
    assert created.status_code == 200
    video = created.json()
    assert video["model"] == "clip"
    assert media_gen.videos[video["id"]]["model"] == "ltx-2.3-distilled"

    assert client.get(f"/v1/videos/{video['id']}", headers=_auth()).json()["status"] == "queued"
    content = client.get(f"/v1/videos/{video['id']}/content", headers=_auth())
    assert content.status_code == 200
    assert content.headers["content-type"] == "video/mp4"
    assert content.content.startswith(b"\x00\x00\x00\x18ftyp")

    for path in (f"/v1/videos/{video['id']}", f"/v1/videos/{video['id']}/content"):
        assert client.get(path, headers=_auth(BOB)).status_code == 404
    assert client.delete(f"/v1/videos/{video['id']}", headers=_auth(BOB)).status_code == 404
    assert client.get("/v1/videos", headers=_auth(BOB)).json()["data"] == []

    listed = client.get("/v1/videos", headers=_auth()).json()
    assert [v["id"] for v in listed["data"]] == [video["id"]]
    assert client.delete(f"/v1/videos/{video['id']}", headers=_auth()).json()["deleted"] is True
    assert client.get(f"/v1/videos/{video['id']}", headers=_auth()).status_code == 404


def test_video_list_paginates_and_forgets_expired(media_gen):
    ids = [_video().json()["id"] for _ in range(3)]
    page = client.get("/v1/videos?limit=2", headers=_auth()).json()
    assert page["has_more"] is True
    rest = client.get(f"/v1/videos?limit=2&after={page['last_id']}", headers=_auth()).json()
    assert {v["id"] for v in page["data"] + rest["data"]} == set(ids)
    del media_gen.videos[ids[0]]  # expired at media-gen
    assert {v["id"] for v in client.get("/v1/videos", headers=_auth()).json()["data"]} == set(ids[1:])
    assert client.get(f"/v1/videos/{ids[0]}", headers=_auth()).status_code == 404


def test_multipart_video_with_reference_image(media_gen):
    response = client.post(
        "/v1/videos", headers=_auth(),
        data={"model": "clip", "prompt": "animate"},
        files={"input_reference": ("frame.png", b"\x89PNG fake", "image/png")},
    )
    assert response.status_code == 200, response.text
    assert media_gen.videos[response.json()["id"]]["model"] == "multipart"


def test_provider_errors_are_relayed_once(media_gen):
    media_gen.fail_with = httpx.Response(
        503, json={"error": {"code": "insufficient_memory", "message": "busy"}}, headers={"retry-after": "60"})
    response = _image()
    assert response.status_code == 503
    assert response.headers["retry-after"] == "60"
    assert response.json()["error"]["code"] == "insufficient_memory"
    assert len(media_gen.requests) == 1  # never retried


def test_unreachable_provider_is_502(media_gen, monkeypatch):
    def down(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(media, "_transport", httpx.MockTransport(down))
    response = _image()
    assert response.status_code == 502
    assert "media" in response.json()["error"]["message"]


def test_profiles_cannot_route_to_media_models(media_gen):
    from src import profiles

    assert "pic" not in profiles._inventory_by_name()
    assert "chat-a" in profiles._inventory_by_name()


def test_each_legacy_key_owns_its_own_videos(media_gen):
    video = _video(key="legacy-one").json()
    assert client.get(f"/v1/videos/{video['id']}", headers=_auth("legacy-one")).status_code == 200
    assert client.get(f"/v1/videos/{video['id']}", headers=_auth("legacy-two")).status_code == 404
    assert client.get("/v1/videos", headers=_auth("legacy-two")).json()["data"] == []


def test_upstream_auth_failures_are_not_the_clients(media_gen):
    media_gen.fail_with = httpx.Response(401, json={"error": {"code": "invalid_api_key", "message": "no"}})
    response = _image()
    assert response.status_code == 502
    assert "credentials" in response.json()["error"]["message"]


def test_a_page_of_expired_videos_still_continues(media_gen):
    ids = [_video().json()["id"] for _ in range(3)]
    for video_id in ids[1:]:
        del media_gen.videos[video_id]  # the two newest expired at media-gen
    page = client.get("/v1/videos?limit=2", headers=_auth()).json()
    assert page["data"] == [] and page["has_more"] is True and page["last_id"] == ids[1]
    rest = client.get(f"/v1/videos?limit=2&after={page['last_id']}", headers=_auth()).json()
    assert [v["id"] for v in rest["data"]] == [ids[0]]
