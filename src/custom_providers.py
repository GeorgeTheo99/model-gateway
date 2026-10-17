"""User-defined OpenAI-compatible providers.

A credential holding ``providers:custom`` (or a full admin) may point the
gateway at any OpenAI-compatible base URL the owner enters. Such providers live
under reserved ``custom-<slug>`` ids, record the credential that created them
(only it, or a full admin, may change them), speak the OpenAI protocol with a
bearer key and nothing else, and declare no data retention. Their models route
only by their gateway names, so an upstream id can never take over another
route's identifier.

Destinations are checked when saved and again before each request: https for
public hosts, plain http only for loopback or an address on the owner's network
(which needs ``allow_lan``), and never link-local, multicast, or other
non-routable addresses (cloud metadata endpoints among them).
"""

from __future__ import annotations

import ipaddress
import re
import socket
import threading
import time
from urllib.parse import urlsplit

import httpx

PREFIX = "custom-"
MANAGED = "custom"
MAX_PROVIDERS = 20
MAX_MODELS = 50
_MAX_LISTED = 500
_NAME_MAX = 40
_RECHECK_SECONDS = 30.0
_SHARED_ADDRESS_SPACE = ipaddress.ip_network("100.64.0.0/10")  # CGNAT, including tailnets
_UPSTREAM_ID_RE = re.compile(r"^[\x21-\x7e]{1,200}$")

_checked: dict[tuple[str, bool], float] = {}
_checked_lock = threading.Lock()


class CustomProviderError(ValueError):
    """A request the gateway refuses, with a stable code for clients."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def is_custom(provider_id: object) -> bool:
    return isinstance(provider_id, str) and provider_id.startswith(PREFIX)


def is_managed(block: object) -> bool:
    return isinstance(block, dict) and block.get("managed_by") == MANAGED


def display_name(value: object) -> str:
    name = value.strip() if isinstance(value, str) else ""
    if not name or len(name) > _NAME_MAX or not name.isprintable():
        raise CustomProviderError("invalid_name", f"Name the provider with 1-{_NAME_MAX} printable characters")
    return name


def provider_id_for(name: str, taken: set[str]) -> str:
    """``custom-<slug>`` from a display name, numbered when the slug is taken."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:_NAME_MAX].strip("-") or "provider"
    candidate, number = f"{PREFIX}{slug}", 2
    while candidate in taken:
        candidate, number = f"{PREFIX}{slug}-{number}", number + 1
    return candidate


def model_name_for(provider_id: str, upstream_id: str, taken: set[str]) -> str:
    """A gateway model name under the provider's reserved prefix."""
    slug = re.sub(r"[^a-z0-9.]+", "-", upstream_id.lower()).strip("-.")[:80] or "model"
    candidate, number = f"{provider_id}-{slug}", 2
    while candidate in taken:
        candidate, number = f"{provider_id}-{slug}-{number}", number + 1
    return candidate


def _address_class(address: str) -> str:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
        return "blocked"
    if ip.is_private or (isinstance(ip, ipaddress.IPv4Address) and ip in _SHARED_ADDRESS_SPACE):
        return "lan"
    return "public" if ip.is_global else "blocked"


def _resolve(host: str, port: int) -> set[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise CustomProviderError("unreachable", f"Can't find {host}: {exc}") from exc
    return {str(info[4][0]) for info in infos}


def normalize_url(base_url: object) -> tuple[str, str, int]:
    """The base URL without a trailing slash, its host, and its port."""
    raw = base_url.strip() if isinstance(base_url, str) else ""
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise CustomProviderError("invalid_url", f"That address isn't a valid URL: {exc}") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or len(raw) > 2048
    ):
        raise CustomProviderError(
            "invalid_url", "Enter an http(s) address without a login, query, or #fragment",
        )
    return raw.rstrip("/"), parsed.hostname, port or (443 if parsed.scheme == "https" else 80)


def check_destination(base_url: str, allow_lan: bool) -> str:
    """Refuse an unsafe destination; return its network: loopback, lan, or public."""
    url, host, port = normalize_url(base_url)
    classes = {_address_class(address) for address in _resolve(host, port)}
    if "blocked" in classes:
        raise CustomProviderError("blocked_address", f"{host} resolves to an address the gateway never connects to")
    if "lan" in classes and not allow_lan:
        raise CustomProviderError(
            "lan_not_allowed", f"{host} is on your network; confirm that to connect to it",
        )
    if url.startswith("http://") and "public" in classes:
        raise CustomProviderError("https_required", "Use https for a provider on the internet")
    if "public" in classes:
        return "public"
    return "lan" if "lan" in classes else "loopback"


def recheck(provider_id: str, block: dict) -> None:
    """Re-check a saved provider's destination before a request (answers kept briefly)."""
    base_url = str(block.get("base_url") or "")
    key = (base_url, bool(block.get("allow_lan")))
    now = time.monotonic()
    with _checked_lock:
        if _checked.get(key, 0.0) > now:
            return
    try:
        check_destination(base_url, bool(block.get("allow_lan")))
    except CustomProviderError as exc:
        raise CustomProviderError("blocked_address", f"provider {provider_id!r}: {exc}", status=502) from exc
    with _checked_lock:
        _checked[key] = now + _RECHECK_SECONDS


def discover(base_url: str, api_key: str) -> list[str]:
    """Upstream model ids from ``{base_url}/models`` (redirects are not followed)."""
    headers = {"Accept": "application/json", "User-Agent": "model-gateway-custom/1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = httpx.get(f"{base_url}/models", headers=headers, timeout=15)
    except httpx.HTTPError as exc:
        raise CustomProviderError("unreachable", f"Couldn't reach the provider: {type(exc).__name__}") from exc
    if response.status_code in {401, 403}:
        raise CustomProviderError("key_rejected", f"The provider refused the key (HTTP {response.status_code})")
    if response.status_code != 200:
        raise CustomProviderError("not_openai_compatible", f"{base_url}/models answered HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise CustomProviderError("not_openai_compatible", "The provider's /models answer isn't JSON") from exc
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise CustomProviderError("not_openai_compatible", "The provider's /models answer has no model list")
    ids = []
    for row in rows[:_MAX_LISTED]:
        value = row.get("id") if isinstance(row, dict) else None
        if isinstance(value, str) and _UPSTREAM_ID_RE.fullmatch(value) and value not in ids:
            ids.append(value)
    if not ids:
        raise CustomProviderError("no_models", "The provider lists no models")
    return ids


def catalog_entry(name: str, provider_id: str, provider_name: str, upstream_id: str) -> dict:
    """A model with only what the gateway knows: no pricing, capabilities, or retention claim."""
    return {
        "name": name,
        "provider": provider_id,
        "provider_model_id": upstream_id,
        "display_name": f"{upstream_id} ({provider_name})"[:80],
        "desc": f"{upstream_id} on {provider_name}, a provider added by its owner",
    }
