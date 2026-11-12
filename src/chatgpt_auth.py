"""Sign in with ChatGPT: the owner's ChatGPT plan as a gateway provider.

OpenAI's flow for open-source and locally hosted apps
(developers.openai.com/siwc/token-sharing-open-source, checked 2026-11-12):

* The first sign-in registers a client: ``client_id=dynamic_agent_client`` with an
  ``agent_name_hint`` and this host's stable ``ext_agent_host_id``. The callback returns
  the issued ``client_id`` (``oaiapp_...``), which every later sign-in and refresh reuses.
* Authorization code + PKCE (S256) to an HTTP loopback callback on ``127.0.0.1`` (only the
  port may vary). The gateway runs that listener itself. A browser on another device
  cannot reach it, so the final callback URL may also be pasted back (``complete``).
* The token response must grant ``chatgpt.tokens.use.direct``; the access token then
  serves ``POST https://api.openai.com/v1/responses`` (``src.chatgpt_responses``).
* Refresh tokens rotate: refreshes are serialized and the replacement is saved before use.
* Sign-out revokes the refresh token, clears the tokens, and keeps the client and host ids.

The ID token arrives over TLS directly from the token endpoint, so its issuer is
established by TLS (OpenID Connect Core 1.0, section 3.1.3.7, item 6); its issuer,
audience, expiry and nonce are still checked here. The gateway carries no JWT library.

One ChatGPT account per gateway. The record lives in ``<secret dir>/chatgpt-oauth.json``
(mode 0600, written atomically); tokens never appear in logs, URLs or API responses.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from src.secret_files import secret_dir

log = logging.getLogger("model-gateway")

PROVIDER = "chatgpt"
ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {PLAN_SCOPE}"
REGISTRATION_CLIENT = "dynamic_agent_client"
DEFAULT_AGENT_NAME = "Model Gateway"
CALLBACK_PATH = "/auth/callback"
PREFERRED_PORT = 1455
USAGE_URL = "https://chatgpt.com/settings/usage"

_RECORD_NAME = "chatgpt-oauth.json"
_ATTEMPT_SECONDS = 600
_REFRESH_MARGIN_SECONDS = 300
_HTTP_TIMEOUT = 30.0
# Refresh errors that mean the session is gone: sign in again (errors-and-recovery).
_UNUSABLE_REFRESH = frozenset({
    "invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired",
    "refresh_token_invalidated", "refresh_token_reused",
})


class ChatGPTAuthError(Exception):
    """A sign-in step failed; ``code`` is machine-readable, the message is safe to show."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass
class _Attempt:
    state: str
    nonce: str
    verifier: str
    redirect_uri: str
    authorize_url: str
    # None for a new registration: the callback supplies the issued client id.
    client_id: str | None
    expected_subject: str | None
    expires_at: float
    # The consumer whose manager started the sign-in; it may then use the plan. None: an admin.
    consumer: str | None
    server: asyncio.Server | None = None


_attempt: _Attempt | None = None
_listeners: list = []  # called when signing in or out changes whether the provider is ready
_last_error: dict | None = None
_lock = asyncio.Lock()  # one sign-in, refresh, or sign-out at a time
_cache: tuple[float, dict] | None = None  # (record mtime, record)


# ── Record storage ────────────────────────────────────────────────────────────


def record_path() -> Path:
    return secret_dir() / _RECORD_NAME


def _read_record() -> dict:
    """The saved record, or {} when there is none. Cached by file mtime."""
    global _cache
    path = record_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        _cache = None
        return {}
    if _cache is not None and _cache[0] == stat.st_mtime_ns:
        return dict(_cache[1])
    if stat.st_mode & 0o077 or path.is_symlink():
        log.warning("ChatGPT sign-in record ignored: it must be a 0600 regular file")
        return {}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("ChatGPT sign-in record unreadable: %s", type(exc).__name__)
        return {}
    if not isinstance(record, dict):
        return {}
    _cache = (stat.st_mtime_ns, record)
    return dict(record)


def _write_record(record: dict) -> None:
    global _cache
    path = record_path()
    if path.is_symlink():
        raise OSError(f"refusing symlink target: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _cache = None


def _host_id(record: dict) -> str:
    """This gateway's stable ``ext_agent_host_id``, created before its first sign-in."""
    host_id = record.get("ext_agent_host_id")
    if isinstance(host_id, str) and host_id.startswith("urn:uuid:"):
        return host_id
    return f"urn:uuid:{uuid.uuid4()}"


def _signed_in(record: dict) -> bool:
    return bool(record.get("access_token") and record.get("refresh_token")
                and PLAN_SCOPE in (record.get("scopes") or []))


def access_token() -> str:
    """The current access token for the provider config, or "" when signed out.

    It may be near expiry: requests refresh it first (``ensure_fresh``).
    """
    record = _read_record()
    return str(record["access_token"]) if _signed_in(record) else ""


def on_change(listener) -> None:
    """Call ``listener()`` whenever the plan becomes usable or stops being usable."""
    _listeners.append(listener)


def _changed() -> None:
    for listener in _listeners:
        listener()


# ── Tokens ────────────────────────────────────────────────────────────────────


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _jwt_claims(token: object) -> dict:
    """Unverified claims of a JWT (see the module docstring for why that is sound here)."""
    if not isinstance(token, str) or token.count(".") != 2:
        return {}
    payload = token.split(".")[1]
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (ValueError, UnicodeDecodeError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _validate_id_token(id_token: object, *, client_id: str, nonce: str, now: float) -> dict:
    claims = _jwt_claims(id_token)
    audience = claims.get("aud")
    audiences = audience if isinstance(audience, list) else [audience]
    exp = claims.get("exp")
    if (
        claims.get("iss") != ISSUER
        or client_id not in audiences
        or not isinstance(exp, (int, float)) or exp <= now
        or claims.get("nonce") != nonce
        or not isinstance(claims.get("sub"), str) or not claims["sub"]
    ):
        raise ChatGPTAuthError("invalid_id_token", "OpenAI's sign-in answer didn't check out. Try signing in again.")
    return claims


def _token_error_code(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("code") or error.get("type") or "")
    return str(error or body.get("code") or "")


async def _post_token(form: dict) -> dict:
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        resp = await client.post(TOKEN_URL, data=form, headers={"Accept": "application/json"})
    if resp.status_code != 200:
        code = _token_error_code(resp) or f"http_{resp.status_code}"
        raise ChatGPTAuthError(code, f"OpenAI refused the token request ({code}).",
                               status=502 if resp.status_code >= 500 else 400)
    try:
        body = resp.json()
    except ValueError as exc:
        raise ChatGPTAuthError("invalid_token_response", "OpenAI's token answer was unreadable.", 502) from exc
    if not isinstance(body, dict) or not body.get("access_token"):
        raise ChatGPTAuthError("invalid_token_response", "OpenAI's token answer had no access token.", 502)
    return body


def _token_fields(body: dict, now: float, previous: dict) -> dict:
    scope = body.get("scope")
    scopes = sorted(scope.split()) if isinstance(scope, str) and scope.strip() else previous.get("scopes") or []
    expires_in = body.get("expires_in")
    expires_at = now + float(expires_in) if isinstance(expires_in, (int, float)) and expires_in > 0 else 0.0
    if not expires_at:
        exp = _jwt_claims(body["access_token"]).get("exp")
        expires_at = float(exp) if isinstance(exp, (int, float)) else now + 3600
    return {
        "access_token": body["access_token"],
        # Refresh tokens rotate; a refresh answer without one keeps the current one.
        "refresh_token": body.get("refresh_token") or previous.get("refresh_token"),
        "id_token": body.get("id_token") or previous.get("id_token"),
        "token_type": body.get("token_type") or "Bearer",
        "scopes": scopes,
        "expires_at": expires_at,
        "saved_at": now,
    }


# ── Sign-in ───────────────────────────────────────────────────────────────────


def _close_attempt() -> None:
    global _attempt
    attempt, _attempt = _attempt, None
    if attempt is not None and attempt.server is not None:
        attempt.server.close()


async def _listen() -> tuple[asyncio.Server, int]:
    """Start the loopback callback listener, on 1455 when it's free."""
    for port in (PREFERRED_PORT, 0):
        try:
            server = await asyncio.start_server(_handle_callback_connection, "127.0.0.1", port)
        except OSError:
            continue
        return server, server.sockets[0].getsockname()[1]
    raise ChatGPTAuthError("listener_unavailable", "Couldn't open a local port for the sign-in callback.", 500)


def granted_consumers() -> frozenset[str]:
    """Consumers that signed the plan in through their manager; they may use it (an admin may always)."""
    granted = _read_record().get("consumers")
    return frozenset(str(name) for name in granted if name) if isinstance(granted, list) else frozenset()


async def start(*, agent_name: str | None = None, new_account: bool = False, consumer: str | None = None) -> dict:
    """Begin a sign-in: start the callback listener and return the URL to open.

    A saved account signs in again with its issued client id; ``new_account`` registers a
    new client (another ChatGPT account replaces the saved one when it completes).
    """
    global _attempt, _last_error
    async with _lock:
        _close_attempt()
        _last_error = None
        record = _read_record()
        host_id = _host_id(record)
        if record.get("ext_agent_host_id") != host_id:
            # Persisted before the first sign-in, as the flow requires.
            _write_record({**record, "ext_agent_host_id": host_id})
            record = _read_record()
        reuse = bool(record.get("client_id")) and not new_account
        # A registration whose sign-in didn't finish: OpenAI says to retry with its issued client id.
        pending_client = None if reuse else record.get("pending_client_id")
        server, port = await _listen()
        redirect_uri = f"http://127.0.0.1:{port}{CALLBACK_PATH}"
        state, nonce, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_urlsafe(64)
        params = {
            "response_type": "code",
            "client_id": record["client_id"] if reuse else pending_client or REGISTRATION_CLIENT,
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge": _b64url(hashlib.sha256(verifier.encode("ascii")).digest()),
            "code_challenge_method": "S256",
            "ext_agent_host_id": host_id,
        }
        if reuse:
            if record.get("id_token"):
                params["id_token_hint"] = record["id_token"]
            if record.get("email"):
                params["login_hint"] = record["email"]
        elif not pending_client:
            params["agent_name_hint"] = (agent_name or DEFAULT_AGENT_NAME).strip()[:80] or DEFAULT_AGENT_NAME
        _attempt = _Attempt(
            state=state, nonce=nonce, verifier=verifier, redirect_uri=redirect_uri,
            authorize_url=f"{AUTHORIZE_URL}?{urlencode(params)}",
            client_id=record["client_id"] if reuse else pending_client,
            expected_subject=record.get("subject") if reuse else None,
            expires_at=time.time() + _ATTEMPT_SECONDS, server=server, consumer=consumer,
        )
        asyncio.get_running_loop().call_later(_ATTEMPT_SECONDS, _expire, _attempt)
        log.info("ChatGPT sign-in started (callback port %d, %s)", port,
                 "returning account" if reuse else "new registration")
        return {"authorize_url": _attempt.authorize_url, "redirect_uri": redirect_uri,
                "expires_at": _attempt.expires_at}


def _expire(attempt: _Attempt) -> None:
    global _last_error
    if _attempt is attempt:
        _close_attempt()
        _last_error = {"code": "sign_in_expired", "message": "The sign-in wasn't finished in time."}


def _query_from(callback: str) -> dict[str, str]:
    """Callback parameters from a full callback URL or its query string."""
    text = callback.strip()
    query = urlsplit(text).query if "://" in text or text.startswith("/") else text.lstrip("?")
    return {key: values[0] for key, values in parse_qs(query, keep_blank_values=True).items() if values}


async def complete(callback: str) -> dict:
    """Finish a sign-in from the callback's parameters (the listener or a pasted URL)."""
    global _last_error
    params = _query_from(callback)
    async with _lock:
        attempt = _attempt
        if attempt is None or time.time() > attempt.expires_at:
            raise ChatGPTAuthError("no_sign_in", "There's no sign-in in progress. Start again.", 409)
        if not secrets.compare_digest(params.get("state", ""), attempt.state):
            raise ChatGPTAuthError("state_mismatch", "That sign-in link doesn't belong to this sign-in.", 400)
        try:
            result = await _finish(attempt, params)
        except ChatGPTAuthError as exc:
            _last_error = {"code": exc.code, "message": str(exc)}
            _close_attempt()
            raise
        _last_error = None
        _close_attempt()
        return result


async def _finish(attempt: _Attempt, params: dict[str, str]) -> dict:
    if params.get("error"):
        if params["error"] == "access_denied":
            raise ChatGPTAuthError("access_denied", "The sign-in was declined, so the plan can't be used.")
        raise ChatGPTAuthError(params["error"][:80], "OpenAI ended the sign-in with an error.")
    code = params.get("code")
    if not code:
        raise ChatGPTAuthError("missing_code", "The sign-in answer had no authorization code.")
    issued = params.get("client_id")
    if attempt.client_id is None:
        if not issued or issued == REGISTRATION_CLIENT:
            raise ChatGPTAuthError("registration_incomplete", "OpenAI didn't finish registering this gateway.")
        client_id = issued
        _write_record({**_read_record(), "pending_client_id": issued})
    else:
        if issued and issued != attempt.client_id:
            raise ChatGPTAuthError("client_mismatch", "The sign-in came back for a different registration.")
        client_id = attempt.client_id
    body = await _post_token({
        "grant_type": "authorization_code", "client_id": client_id, "code": code,
        "code_verifier": attempt.verifier, "redirect_uri": attempt.redirect_uri, "resource": RESOURCE,
    })
    now = time.time()
    claims = _validate_id_token(body.get("id_token"), client_id=client_id, nonce=attempt.nonce, now=now)
    if attempt.expected_subject and claims["sub"] != attempt.expected_subject:
        raise ChatGPTAuthError("account_mismatch", "That's a different ChatGPT account than the saved one.")
    tokens = _token_fields(body, now, {})
    if PLAN_SCOPE not in tokens["scopes"]:
        raise ChatGPTAuthError("plan_not_granted", "Signed in, but using your ChatGPT plan wasn't allowed.")
    record = _read_record()
    if record.get("refresh_token") and record.get("client_id") and record["client_id"] != client_id:
        # Another account replaces the saved one: end the replaced session.
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            await _revoke(client, record["client_id"], record["refresh_token"])
    record.pop("pending_client_id", None)
    record.update(tokens)
    record.update({"issuer": ISSUER, "client_id": client_id, "subject": claims["sub"],
                   "email": claims.get("email") if isinstance(claims.get("email"), str) else None,
                   "ext_agent_host_id": _host_id(record)})
    record.pop("reauth_reason", None)
    if attempt.consumer:
        record["consumers"] = sorted({*(record.get("consumers") or []), attempt.consumer})
    _write_record(record)
    _changed()
    log.warning("ChatGPT plan signed in for provider %r", PROVIDER)
    return status()


_PAGE = """<!doctype html><meta charset="utf-8"><title>{title}</title>
<body style="font:16px -apple-system,system-ui,sans-serif;margin:4rem auto;max-width:32rem">
<h1 style="font-size:1.4rem">{title}</h1><p>{body}</p></body>"""


async def _handle_callback_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    status_line, title, text = "404 Not Found", "Not found", "This address only finishes a ChatGPT sign-in."
    try:
        request_line = (await asyncio.wait_for(reader.readline(), 10)).decode("latin-1")
        with contextlib.suppress(asyncio.TimeoutError):
            while (await asyncio.wait_for(reader.readline(), 10)) not in (b"\r\n", b"\n", b""):
                pass
        parts = request_line.split()
        if len(parts) >= 2 and parts[0] == "GET" and urlsplit(parts[1]).path == CALLBACK_PATH:
            try:
                await complete(parts[1])
                status_line, title, text = "200 OK", "Signed in", "Your ChatGPT plan is connected. You can close this tab."
            except ChatGPTAuthError as exc:
                status_line, title, text = "400 Bad Request", "Sign-in didn't finish", str(exc)
    except (asyncio.TimeoutError, OSError):
        return
    finally:
        page = _PAGE.format(title=title, body=text).encode()
        with contextlib.suppress(OSError):
            writer.write(
                f"HTTP/1.1 {status_line}\r\nContent-Type: text/html; charset=utf-8\r\n"
                f"Content-Length: {len(page)}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n".encode() + page
            )
            await writer.drain()
            writer.close()


# ── Refresh and sign-out ──────────────────────────────────────────────────────


async def refresh(*, force: bool = False, failed_token: str | None = None) -> str | None:
    """Refresh the access token when near expiry (or ``force``); the new token, or None.

    ``failed_token`` is the token an upstream refused: when it's no longer the current one,
    another request already refreshed it, and that token is returned without a refresh.
    """
    record = _read_record()
    if not _signed_in(record):
        return None
    if failed_token and record.get("access_token") != failed_token:
        return str(record["access_token"])
    if not force and float(record.get("expires_at") or 0) - time.time() > _REFRESH_MARGIN_SECONDS:
        return None
    async with _lock:
        latest = _read_record()
        if not _signed_in(latest):
            return None
        if latest.get("access_token") != record.get("access_token"):
            return str(latest["access_token"])  # another request refreshed it meanwhile
        try:
            body = await _post_token({
                "grant_type": "refresh_token", "client_id": latest["client_id"],
                "refresh_token": latest["refresh_token"], "resource": RESOURCE,
            })
        except ChatGPTAuthError as exc:
            if exc.code in _UNUSABLE_REFRESH:
                _clear_tokens(latest, reason=exc.code)
                log.warning("ChatGPT plan session ended (%s): sign in again", exc.code)
            else:
                log.warning("ChatGPT token refresh failed (%s); keeping the session", exc.code)
            return None
        except httpx.HTTPError as exc:
            log.warning("ChatGPT token refresh failed: %s", type(exc).__name__)
            return None
        latest.update(_token_fields(body, time.time(), latest))
        _write_record(latest)
        return str(latest["access_token"])


def _clear_tokens(record: dict, *, reason: str | None) -> None:
    for key in ("access_token", "refresh_token", "id_token", "expires_at", "scopes", "token_type", "saved_at"):
        record.pop(key, None)
    if reason:
        record["reauth_reason"] = reason
    _write_record(record)
    _changed()


async def _revocation_endpoint(client: httpx.AsyncClient) -> str:
    resp = await client.get(DISCOVERY_URL)
    resp.raise_for_status()
    endpoint = resp.json().get("revocation_endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith(f"{ISSUER}/"):
        raise ValueError("no revocation endpoint")
    return endpoint


async def _revoke(client: httpx.AsyncClient, client_id: str, refresh_token: str) -> bool:
    """Revoke a renewable session, retrying network failures and 5xx; whether OpenAI confirmed it."""
    for attempt in range(3):
        try:
            endpoint = await _revocation_endpoint(client)
            resp = await client.post(endpoint, data={
                "token": refresh_token, "token_type_hint": "refresh_token", "client_id": client_id,
            })
        except (httpx.HTTPError, ValueError):
            resp = None
        if resp is not None and resp.status_code == 200:
            return True
        if resp is not None and resp.status_code < 500:
            return False
        await asyncio.sleep(0.5 * 2 ** attempt)
    return False


async def sign_out() -> dict:
    """Revoke the session, then clear its tokens (the client and host ids stay)."""
    async with _lock:
        _close_attempt()
        record = _read_record()
        revoked = True
        if record.get("refresh_token") and record.get("client_id"):
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                revoked = await _revoke(client, record["client_id"], record["refresh_token"])
        if record:
            _clear_tokens(record, reason=None)
        log.warning("ChatGPT plan signed out (revocation %s)", "confirmed" if revoked else "not confirmed")
        result = status()
        result["revocation_confirmed"] = revoked
        return result


# ── Status ────────────────────────────────────────────────────────────────────


def status() -> dict:
    """What a settings screen shows; never a token."""
    record = _read_record()
    signed_in = _signed_in(record)
    plan = None
    if signed_in:
        auth_claims = _jwt_claims(record.get("access_token")).get("https://api.openai.com/auth")
        if isinstance(auth_claims, dict) and isinstance(auth_claims.get("chatgpt_plan_type"), str):
            plan = auth_claims["chatgpt_plan_type"]
    pending = None
    if _attempt is not None and time.time() <= _attempt.expires_at:
        # Not its URL: a returning sign-in's carries the ID token as id_token_hint.
        pending = {"redirect_uri": _attempt.redirect_uri, "expires_at": _attempt.expires_at}
    return {
        "provider": PROVIDER,
        "signed_in": signed_in,
        "email": record.get("email") if signed_in else None,
        "plan": plan,
        "registered": bool(record.get("client_id")),
        "reauth_reason": record.get("reauth_reason"),
        "pending": pending,
        "last_error": _last_error,
        "usage_url": USAGE_URL,
    }
