"""Image and video generation routes (docs/adr/0004-local-image-and-video-generation.md).

The gateway authenticates, applies model allowlists, records the ledger and
forwards each request once to the model's provider (media-gen on this Mac):
generation is expensive and ``POST /v1/videos`` is not idempotent, so there
are no retries, pools, fallbacks, profiles or federation. A video belongs to
the credential that created it; any other caller gets 404.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src import federation, ledger, profiles, providers
from src.auth import AuthIdentity, _extract_token, require_client_auth
from src.request_options import upstream_auth_headers

log = logging.getLogger(__name__)

router = APIRouter()

MAX_JSON_BYTES = 1_000_000
# A reference image (20 MiB at media-gen) plus multipart framing.
MAX_VIDEO_UPLOAD_BYTES = 21 * 1024 * 1024
IMAGE_TIMEOUT = httpx.Timeout(1800, connect=10)
CALL_TIMEOUT = httpx.Timeout(120, connect=10)
# Ownership rows outlive any media-gen retention, so a page cursor never points at a purged row.
VIDEO_ROW_SECONDS = 7 * 24 * 3600
# Tests replace this with an httpx.MockTransport.
_transport: httpx.AsyncBaseTransport | None = None


def _error(status: int, message: str, code: str | None = None, error_type: str | None = None) -> JSONResponse:
    if error_type is None:
        error_type = "invalid_request_error" if status < 500 else "api_error"
    error = {"message": message, "type": error_type}
    if code:
        error["code"] = code
    return JSONResponse(status_code=status, content={"error": error})


class _Refused(Exception):
    def __init__(self, response: JSONResponse):
        self.response = response


# Video ownership ------------------------------------------------------------

_db_lock = threading.Lock()


def _db_path() -> Path:
    return ledger.ledger_path().with_name("media-videos.db")


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.close(fd)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS videos (id TEXT PRIMARY KEY, owner TEXT NOT NULL, model TEXT NOT NULL,"
        " created_at INTEGER NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS videos_owner ON videos (owner, created_at)")
    return conn


def _query(sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    with _db_lock, closing(_connect()) as conn:
        return conn.execute(sql, args).fetchall()


def _owner(request: Request, identity: AuthIdentity) -> str:
    """The credential a video belongs to: a consumer credential's id, or a digest of the
    legacy or admin key itself, so holders of different keys never share videos."""
    if identity.principal is not None:
        return f"consumer:{identity.principal.credential_id}"
    if identity.kind == "anonymous":
        return "anonymous"
    return f"{identity.kind}:{hashlib.sha256(_extract_token(request).encode()).hexdigest()[:32]}"


def _owned_video(video_id: str, owner: str) -> sqlite3.Row:
    rows = _query("SELECT * FROM videos WHERE id = ? AND owner = ?", (video_id, owner))
    if not rows:
        raise _Refused(_error(404, f"No video with id '{video_id}'", "video_not_found", "not_found_error"))
    return rows[0]


# Request checks -------------------------------------------------------------


def _authenticate(request: Request) -> AuthIdentity:
    if federation.manager().has_forwarding_headers(request):
        raise _Refused(_error(403, "Image and video generation is not available through federation"))
    return require_client_auth(request)


def _caller(request: Request) -> str:
    return _owner(request, _authenticate(request))


def _authorize_model(identity: AuthIdentity, model: object, modality: str) -> providers.ProviderInfo:
    # The allowlist comes first, so a name the credential may not use reveals nothing
    # about it (docs/adr/0003-scoped-access-and-lightweight-local-models.md section 1).
    principal = identity.principal
    if principal is not None and (
        not principal.allow_direct_models
        or (principal.models is not None
            and not providers.direct_model_permitted(model, principal.models, principal.namespaces))
    ):
        raise _Refused(_error(403, "Consumer principal is not authorized to invoke this model",
                              "direct_model_access_denied"))
    if not isinstance(model, str) or not model:
        raise _Refused(_error(400, "`model` is required", "missing_required_parameter"))
    if profiles.is_selector(model):
        raise _Refused(_error(400, "Profiles route chat models; name an image or video model directly"))
    info = providers.resolve(model)
    if info is None:
        reason = providers.model_availability(model).get("reason") or "unavailable"
        if reason == "model_not_found":
            raise _Refused(_error(404, f"Model '{model}' not found in gateway", "model_not_found"))
        raise _Refused(_error(404, f"Model '{model}' is unavailable ({reason})", "model_unavailable"))
    if info.modality != modality:
        raise _Refused(_error(400, f"Model '{model}' is a {info.modality} model, not a {modality} model",
                              "wrong_modality"))
    return info


async def _json_body(request: Request) -> dict:
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_JSON_BYTES:
            raise _Refused(_error(413, "Request body exceeds the byte limit"))
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = None
    if not isinstance(body, dict):
        raise _Refused(_error(400, "The request body must be a JSON object"))
    return body


def _set_ledger(request: Request, model: str, info: providers.ProviderInfo) -> None:
    request.state.ledger_ctx = {
        "model": model,
        "provider": info.provider,
        "provider_model_id": info.provider_model_id,
        "is_stream": False,
        # Images are megabytes of base64 with no token usage: record without reading the body.
        "record_without_body": True,
    }


# Upstream -------------------------------------------------------------------


def _client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, transport=_transport)


def _url(info: providers.ProviderInfo, path: str) -> str:
    return info.base_url.rstrip("/") + path


def _headers(info: providers.ProviderInfo) -> dict[str, str]:
    return upstream_auth_headers(info.api_key, "openai", ())


def _unreachable(info: providers.ProviderInfo, exc: httpx.HTTPError) -> JSONResponse:
    log.warning("media provider %s failed: %s", info.provider, type(exc).__name__)
    if isinstance(exc, httpx.TimeoutException):
        return _error(504, f"Provider '{info.provider}' timed out")
    return _error(502, f"Provider '{info.provider}' is unreachable")


def _relay(upstream: httpx.Response, content: bytes | None = None) -> Response:
    if upstream.status_code in (401, 403):
        # The gateway's provider key, not the client's: never let a client read it as its own.
        log.warning("media provider rejected the gateway's key (HTTP %s)", upstream.status_code)
        return _error(502, "The media provider rejected the gateway's credentials")
    headers = {}
    if retry_after := upstream.headers.get("retry-after"):
        headers["Retry-After"] = retry_after
    return Response(
        content=upstream.content if content is None else content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
        headers=headers,
    )


def _as_gateway_video(video: dict, model: str) -> dict:
    """The provider's video object, under the model name the client used."""
    return {**video, "model": model}


def _video_info(row: sqlite3.Row) -> providers.ProviderInfo:
    info = providers.resolve(row["model"])
    if info is None or info.modality != "video":
        raise _Refused(_error(503, f"Model '{row['model']}' is no longer available", "model_unavailable"))
    return info


# Routes ---------------------------------------------------------------------


@router.post("/v1/images/generations")
async def images_generations(request: Request):
    try:
        identity = _authenticate(request)
        body = await _json_body(request)
        info = _authorize_model(identity, body.get("model"), "image")
    except _Refused as refused:
        return refused.response
    model = body["model"]
    _set_ledger(request, model, info)
    body["model"] = info.provider_model_id
    log.info("image %s -> %s", model, info.provider)
    try:
        async with _client(IMAGE_TIMEOUT) as client:
            upstream = await client.post(_url(info, "/images/generations"), json=body, headers=_headers(info))
    except httpx.HTTPError as exc:
        return _unreachable(info, exc)
    return _relay(upstream)


async def _video_create_request(request: Request) -> tuple[dict, dict | None]:
    """The create fields, and the multipart reference image if one was sent."""
    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        return await _json_body(request), None
    length = request.headers.get("content-length", "")
    if not length.isdigit() or int(length) > MAX_VIDEO_UPLOAD_BYTES:
        raise _Refused(_error(413, "Request body exceeds the byte limit"))
    form = await request.form(max_files=1, max_fields=16)
    fields: dict = {}
    files = None
    for key, value in form.multi_items():
        if isinstance(value, str):
            fields[key] = value
        elif key == "input_reference":
            files = {key: (value.filename or "reference", await value.read(), value.content_type)}
        else:
            raise _Refused(_error(400, f"Unexpected file field '{key}'"))
    return fields, files


@router.post("/v1/videos")
async def create_video(request: Request):
    try:
        identity = _authenticate(request)
        fields, files = await _video_create_request(request)
        info = _authorize_model(identity, fields.get("model"), "video")
    except _Refused as refused:
        return refused.response
    model = fields["model"]
    _set_ledger(request, model, info)
    fields["model"] = info.provider_model_id
    log.info("video %s -> %s", model, info.provider)
    try:
        async with _client(CALL_TIMEOUT) as client:
            if files is None:
                upstream = await client.post(_url(info, "/videos"), json=fields, headers=_headers(info))
            else:
                upstream = await client.post(_url(info, "/videos"), data=fields, files=files,
                                             headers=_headers(info))
    except httpx.HTTPError as exc:
        return _unreachable(info, exc)
    if upstream.status_code != 200:
        return _relay(upstream)
    video = upstream.json()
    await asyncio.to_thread(
        _query, "INSERT INTO videos (id, owner, model, created_at) VALUES (?, ?, ?, ?)",
        (video["id"], _owner(request, identity), model, int(time.time())),
    )
    return JSONResponse(_as_gateway_video(video, model))


async def _fetch_video(client: httpx.AsyncClient, row: sqlite3.Row, info: providers.ProviderInfo) -> dict | None:
    """The provider's video, or None once it expired there."""
    upstream = await client.get(_url(info, f"/videos/{row['id']}"), headers=_headers(info))
    if upstream.status_code == 404:
        return None
    upstream.raise_for_status()
    return _as_gateway_video(upstream.json(), row["model"])


async def _list_entry(client: httpx.AsyncClient, row: sqlite3.Row) -> dict | None:
    try:
        info = _video_info(row)
    except _Refused:
        return None  # its model is gone from the catalog, so the video cannot be reached
    return await _fetch_video(client, row, info)


@router.get("/v1/videos")
async def list_videos(request: Request, limit: int = 20, after: str | None = None, order: str = "desc"):
    try:
        owner = _caller(request)
        if not 1 <= limit <= 100:
            raise _Refused(_error(400, "`limit` must be from 1 to 100"))
        if order not in ("asc", "desc"):
            raise _Refused(_error(400, "`order` must be asc or desc"))
        await asyncio.to_thread(_query, "DELETE FROM videos WHERE created_at < ?",
                                (int(time.time()) - VIDEO_ROW_SECONDS,))
        query = "SELECT * FROM videos WHERE owner = ?"
        args: list = [owner]
        if after:
            anchor = await asyncio.to_thread(_owned_video, after, owner)
            query += " AND (created_at, id) " + ("<" if order == "desc" else ">") + " (?, ?)"
            args += [anchor["created_at"], anchor["id"]]
        direction = "DESC" if order == "desc" else "ASC"
        query += f" ORDER BY created_at {direction}, id {direction} LIMIT ?"
        rows = await asyncio.to_thread(_query, query, (*args, limit + 1))
        async with _client(CALL_TIMEOUT) as client:
            videos = await asyncio.gather(*(_list_entry(client, row) for row in rows[:limit]))
    except _Refused as refused:
        return refused.response
    except httpx.HTTPError as exc:
        log.warning("media provider failed while listing videos: %s", type(exc).__name__)
        return _error(502, "A media provider is unreachable")
    data = [video for video in videos if video is not None]
    page = rows[:limit]
    # The cursor follows the rows read, so a page whose videos all expired still continues.
    return {
        "object": "list",
        "data": data,
        "first_id": page[0]["id"] if page else None,
        "last_id": page[-1]["id"] if page else None,
        "has_more": len(rows) > limit,
    }


@router.get("/v1/videos/{video_id}")
async def get_video(request: Request, video_id: str):
    try:
        row = await asyncio.to_thread(_owned_video, video_id, _caller(request))
        info = _video_info(row)
        async with _client(CALL_TIMEOUT) as client:
            video = await _fetch_video(client, row, info)
    except _Refused as refused:
        return refused.response
    except httpx.HTTPError as exc:
        return _unreachable(info, exc)
    if video is None:
        return _error(404, f"No video with id '{video_id}'", "video_not_found", "not_found_error")
    return video


@router.get("/v1/videos/{video_id}/content")
async def video_content(request: Request, video_id: str, variant: str = "video"):
    try:
        row = await asyncio.to_thread(_owned_video, video_id, _caller(request))
        info = _video_info(row)
    except _Refused as refused:
        return refused.response
    client = _client(httpx.Timeout(60, connect=10))
    try:
        upstream = await client.send(
            client.build_request("GET", _url(info, f"/videos/{video_id}/content"),
                                 params={"variant": variant}, headers=_headers(info)),
            stream=True,
        )
    except httpx.HTTPError as exc:
        await client.aclose()
        return _unreachable(info, exc)
    if upstream.status_code != 200:
        content = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return _relay(upstream, content)

    async def body():
        # Closed however the stream ends, a client disconnect included.
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    headers = {}
    if "content-disposition" in upstream.headers:
        headers["Content-Disposition"] = upstream.headers["content-disposition"]
    if "content-length" in upstream.headers and "content-encoding" not in upstream.headers:
        headers["Content-Length"] = upstream.headers["content-length"]
    return StreamingResponse(body(), media_type="video/mp4", headers=headers)


@router.delete("/v1/videos/{video_id}")
async def delete_video(request: Request, video_id: str):
    try:
        row = await asyncio.to_thread(_owned_video, video_id, _caller(request))
        info = _video_info(row)
    except _Refused as refused:
        return refused.response
    try:
        async with _client(CALL_TIMEOUT) as client:
            upstream = await client.delete(_url(info, f"/videos/{video_id}"), headers=_headers(info))
    except httpx.HTTPError as exc:
        return _unreachable(info, exc)
    if upstream.status_code not in (200, 404):
        return _relay(upstream)
    await asyncio.to_thread(_query, "DELETE FROM videos WHERE id = ?", (video_id,))
    return {"id": video_id, "object": "video.deleted", "deleted": True}

