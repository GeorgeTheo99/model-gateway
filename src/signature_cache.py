"""In-memory cache for Gemini thought_signature by tool_call_id.

Gemini 3 models require thoughtSignature on functionCall parts when replaying
conversation history, but clients (Claude Code, Codex CLI) strip unknown fields
from tool_use blocks. This cache stores signatures returned by Gemini and injects
them back into outbound requests, making the fix client-agnostic.
"""

import time
import logging

log = logging.getLogger("model-gateway")

# tool_call_id -> (thought_signature, timestamp)
_cache: dict[str, tuple[str, float]] = {}

# TTL in seconds — 24h covers long Claude Code sessions
_TTL = 86400


def store(tool_call_id: str, thought_signature: str) -> None:
    """Cache a thought_signature for a tool_call_id."""
    if not tool_call_id or not thought_signature:
        return
    _cache[tool_call_id] = (thought_signature, time.monotonic())
    log.debug("signature_cache: stored signature for %s", tool_call_id)


def lookup(tool_call_id: str) -> str | None:
    """Look up a cached thought_signature. Returns None if not found or expired."""
    entry = _cache.get(tool_call_id)
    if entry is None:
        return None
    sig, ts = entry
    if time.monotonic() - ts > _TTL:
        del _cache[tool_call_id]
        return None
    return sig


def store_from_extra_content(tool_call_id: str, extra_content: dict | None) -> None:
    """Extract thought_signature from extra_content and cache it.

    Gemini returns signatures as:
      extra_content.google.thought_signature
    """
    if not extra_content:
        return
    ts = (extra_content.get("google") or {}).get("thought_signature")
    if ts:
        store(tool_call_id, ts)


def inject_into_tool_call(tc: dict) -> dict:
    """Inject cached thought_signature into a tool_call dict if it lacks one.

    Checks for existing extra_content.google.thought_signature first.
    If absent, looks up the cache by tool_call id.
    """
    # Already has a signature — nothing to do
    existing_ec = tc.get("extra_content")
    if existing_ec and (existing_ec.get("google") or {}).get("thought_signature"):
        return tc

    tc_id = tc.get("id", "")
    sig = lookup(tc_id)
    if sig:
        if existing_ec:
            if "google" not in existing_ec:
                existing_ec["google"] = {}
            existing_ec["google"]["thought_signature"] = sig
        else:
            tc["extra_content"] = {"google": {"thought_signature": sig}}
        log.debug("signature_cache: injected signature for %s", tc_id)

    return tc


def cleanup() -> None:
    """Remove expired entries. Called lazily on store operations."""
    now = time.monotonic()
    expired = [k for k, (_, ts) in _cache.items() if now - ts > _TTL]
    for k in expired:
        del _cache[k]
    if expired:
        log.debug("signature_cache: cleaned up %d expired entries", len(expired))
