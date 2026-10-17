"""User-defined OpenAI-compatible providers.

A credential holding ``providers:custom`` (or a full admin) may point the
gateway at any OpenAI-compatible base URL the owner enters. Such providers live
under reserved ``custom-<slug>`` ids, record the credential that created them
(only it, or a full admin, may change them), speak the OpenAI protocol with a
bearer key and nothing else, and declare no data retention. Their models route
only by their gateway names, so an upstream id can never take over another
route's identifier.

Destinations are checked when saved and again before each request. Plain http
is allowed only to an IP address (or ``localhost``): loopback, or one on the
owner's network with ``allow_lan``. A host name must use https, so TLS, not a
DNS answer that could change after the check, decides what the gateway talks
to. Link-local, multicast, cloud metadata and other non-routable addresses are
never allowed. ``allow_lan`` records the owner's consent in the app; the
credential holding ``providers:custom`` is the security boundary.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
from urllib.parse import urlsplit, urlunsplit

import httpx

PREFIX = "custom-"
MANAGED = "custom"
MAX_PROVIDERS = 20
MAX_MODELS = 50
_MAX_LISTED = 500
_MAX_LIST_BYTES = 2_000_000
_NAME_MAX = 40
_SHARED_ADDRESS_SPACE = ipaddress.ip_network("100.64.0.0/10")  # CGNAT, including tailnets
# Metadata services inside otherwise private ranges (AWS IPv6, Alibaba Cloud).
_METADATA = {ipaddress.ip_address("fd00:ec2::254"), ipaddress.ip_address("100.100.100.200")}
_UPSTREAM_ID_RE = re.compile(r"^[\x21-\x7e]{1,200}$")


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


def provider_id_for(name: str, in_use) -> str:
    """``custom-<slug>`` from a display name, numbered while ``in_use(candidate)``."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:_NAME_MAX].strip("-") or "provider"
    candidate, number = f"{PREFIX}{slug}", 2
    while in_use(candidate):
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
    if ip in _METADATA:
        return "blocked"
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


def normalize_url(base_url: object) -> tuple[str, str, str, int]:
    """The base URL (lowercase scheme and host, no trailing slash), its scheme, host, and port."""
    raw = base_url.strip() if isinstance(base_url, str) else ""
    if not raw or len(raw) > 2048 or not raw.isascii() or any(ord(char) <= 0x20 or ord(char) == 0x7F for char in raw):
        raise CustomProviderError("invalid_url", "Enter the address as plain text, without spaces")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise CustomProviderError("invalid_url", f"That address isn't a valid URL: {exc}") from exc
    if (
        parsed.scheme not in {"http", "https"}  # urlsplit lowercases the scheme
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise CustomProviderError(
            "invalid_url", "Enter an http(s) address without a login, query, or #fragment",
        )
    url = urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))
    return url, parsed.scheme, parsed.hostname, port or (443 if parsed.scheme == "https" else 80)


def _refuse(classes: set[str], scheme: str, host: str, allow_lan: bool) -> None:
    if "blocked" in classes:
        raise CustomProviderError("blocked_address", f"{host} is an address the gateway never connects to")
    if "lan" in classes and not allow_lan:
        raise CustomProviderError("lan_not_allowed", f"{host} is on your network; confirm that to connect to it")
    if scheme == "http" and "public" in classes:
        raise CustomProviderError("https_required", "Use https for a provider on the internet")


def _literal_class(host: str) -> str | None:
    """The address class of an IP literal or ``localhost``; None for any other host name."""
    if host == "localhost":
        return "loopback"
    try:
        return _address_class(host)
    except ValueError:
        return None


def _check_url(base_url: object, allow_lan: bool) -> tuple[str, str, int, str | None]:
    """The checks that need no DNS: (url, host, port, network of an IP literal or None)."""
    url, scheme, host, port = normalize_url(base_url)
    network = _literal_class(host)
    if network is None:
        if scheme != "https":
            raise CustomProviderError(
                "https_required", "Use https, or the server's IP address for plain http on your network",
            )
        return url, host, port, None
    _refuse({network}, scheme, host, allow_lan)
    return url, host, port, network


def check_destination(base_url: object, allow_lan: bool) -> str:
    """Refuse an unsafe destination; return its network: loopback, lan, or public."""
    _url, host, port, network = _check_url(base_url, allow_lan)
    if network is not None:
        return network
    classes = {_address_class(address) for address in _resolve(host, port)}
    _refuse(classes, "https", host, allow_lan)
    if "public" in classes:
        return "public"
    return "lan" if "lan" in classes else "loopback"


def recheck(provider_id: str, block: dict) -> None:
    """Re-check a saved provider's address before a request, without DNS (see the module docstring)."""
    try:
        _check_url(block.get("base_url"), bool(block.get("allow_lan")))
    except CustomProviderError as exc:
        raise CustomProviderError("blocked_address", f"provider {provider_id!r}: {exc}", status=502) from exc


def discover(base_url: str, api_key: str) -> list[str]:
    """Upstream model ids from ``{base_url}/models`` (redirects are not followed)."""
    headers = {"Accept": "application/json", "User-Agent": "model-gateway-custom/1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = bytearray()
    try:
        with httpx.stream("GET", f"{base_url}/models", headers=headers, timeout=15) as response:
            status = response.status_code
            if status == 200:
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > _MAX_LIST_BYTES:
                        raise CustomProviderError("not_openai_compatible", "The provider's model list is too large")
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise CustomProviderError("unreachable", f"Couldn't reach the provider: {type(exc).__name__}") from exc
    if status in {401, 403}:
        raise CustomProviderError("key_rejected", f"The provider refused the key (HTTP {status})")
    if status != 200:
        raise CustomProviderError("not_openai_compatible", f"{base_url}/models answered HTTP {status}")
    try:
        payload = json.loads(body)
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
