"""Model requests this gateway process is serving right now (admin "Running now").

The ledger middleware registers each recorded request when it arrives and
removes it once the ledger row is written. Entries keep a reference to the
request state, so the served model shows up as soon as the handler resolves it.
Process-local and in memory only: nothing is persisted and no request content
is kept.
"""

from __future__ import annotations

import itertools
import threading
import time

# A streamed response that is never iterated never reaches its cleanup; drop
# such entries rather than report them as running forever.
STALE_AFTER_SECONDS = 6 * 3600

_lock = threading.Lock()
_ids = itertools.count(1)
_active: dict[int, tuple[float, str, object]] = {}


def start(endpoint: str, state: object) -> int:
    """Register a request; ``state`` is its ``request.state``. Returns a token for :func:`finish`."""
    token = next(_ids)
    now = time.time()
    with _lock:
        _prune(now)
        _active[token] = (now, endpoint, state)
    return token


def finish(token: int) -> None:
    with _lock:
        _active.pop(token, None)


def _prune(now: float) -> None:
    for token in [t for t, (started, _, _) in _active.items() if now - started > STALE_AFTER_SECONDS]:
        del _active[token]


def snapshot() -> list[dict]:
    """Requests in progress, oldest first. ``model`` is None until the handler resolves it."""
    now = time.time()
    with _lock:
        _prune(now)
        entries = sorted(_active.items(), key=lambda item: item[1][0])
    rows = []
    for token, (started, endpoint, state) in entries:
        ctx = getattr(state, "ledger_ctx", None) or {}
        rows.append({
            "id": token,
            "started_at": started,
            "elapsed_ms": int((now - started) * 1000),
            "endpoint": endpoint,
            "model": ctx.get("model"),
            "provider": ctx.get("provider"),
            "provider_model_id": ctx.get("provider_model_id"),
            "is_stream": bool(ctx.get("is_stream")),
        })
    return rows
