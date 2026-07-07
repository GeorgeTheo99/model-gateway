"""Per-provider circuit breaker with coalesced retries.

When an upstream provider starts failing (502/503/504), we don't want every
concurrent request independently retrying — that wastes time and hammers a
sick endpoint. Instead:

1. After TRIP_THRESHOLD consecutive failures, the circuit OPENS.
2. While open, incoming requests WAIT instead of hitting the endpoint.
3. One probe request tests the endpoint every PROBE_INTERVAL seconds.
4. If the probe succeeds, the circuit CLOSES and all waiters are released.
5. If waiters exceed WAIT_TIMEOUT, they get the last error (user sees failure).

This keeps the gateway absorbing transient outages silently — critical for
agentic coding where surfacing a 429/502 forces the user to type "continue".
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger("model-gateway")

# --- Tuning ---
TRIP_THRESHOLD = 3        # consecutive failures before circuit opens
PROBE_INTERVAL = 10.0     # seconds between probe attempts while open
WAIT_TIMEOUT = 180.0      # max seconds a request will wait for recovery (3 min)
SUCCESS_RESET = 1         # consecutive successes to fully close circuit


@dataclass
class _ProviderCircuit:
    """State for a single provider's circuit."""
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    is_open: bool = False
    last_failure_time: float = 0.0
    last_failure_status: int = 0
    last_failure_message: str = ""
    probe_in_progress: bool = False
    # Event that waiters block on; set when circuit closes or probe completes
    recovery_event: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self):
        # Start with event set (circuit closed = go ahead)
        self.recovery_event.set()


_circuits: dict[str, _ProviderCircuit] = {}


def _get(provider: str) -> _ProviderCircuit:
    if provider not in _circuits:
        _circuits[provider] = _ProviderCircuit()
    return _circuits[provider]


def record_success(provider: str) -> None:
    """Record a successful request to a provider."""
    c = _get(provider)
    c.consecutive_failures = 0
    c.consecutive_successes += 1
    if c.is_open:
        if c.consecutive_successes >= SUCCESS_RESET:
            log.info("circuit[%s]: CLOSED — provider recovered", provider)
            c.is_open = False
            c.probe_in_progress = False
            c.recovery_event.set()  # release all waiters


def record_failure(provider: str, status: int = 0, message: str = "") -> None:
    """Record a failed request. May trip the circuit open."""
    c = _get(provider)
    c.consecutive_successes = 0
    c.consecutive_failures += 1
    c.last_failure_time = time.monotonic()
    c.last_failure_status = status
    c.last_failure_message = message

    if not c.is_open and c.consecutive_failures >= TRIP_THRESHOLD:
        log.warning(
            "circuit[%s]: OPEN — %d consecutive failures (last: %d %s)",
            provider, c.consecutive_failures, status, message[:100],
        )
        c.is_open = True
        c.recovery_event.clear()  # block new requests


def is_tripped(provider: str) -> bool:
    """Check if circuit is open (provider is considered down)."""
    c = _get(provider)
    return c.is_open


def should_probe(provider: str) -> bool:
    """Check if it's time to send a probe request.

    Returns True (and marks probe in-progress) if no probe is running
    and enough time has passed since the last failure.
    """
    c = _get(provider)
    if not c.is_open:
        return False
    if c.probe_in_progress:
        return False
    elapsed = time.monotonic() - c.last_failure_time
    if elapsed >= PROBE_INTERVAL:
        c.probe_in_progress = True
        return True
    return False


def probe_done(provider: str, success: bool) -> None:
    """Called when a probe request completes."""
    c = _get(provider)
    c.probe_in_progress = False
    if success:
        record_success(provider)
    else:
        c.last_failure_time = time.monotonic()
        log.info("circuit[%s]: probe failed, still OPEN", provider)


async def wait_for_recovery(provider: str) -> bool:
    """Wait for the circuit to close. Returns True if recovered, False if timed out."""
    c = _get(provider)
    if not c.is_open:
        return True

    deadline = time.monotonic() + WAIT_TIMEOUT
    while c.is_open:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log.warning("circuit[%s]: waiter timed out after %.0fs", provider, WAIT_TIMEOUT)
            return False
        try:
            await asyncio.wait_for(c.recovery_event.wait(), timeout=min(remaining, PROBE_INTERVAL))
            if not c.is_open:
                return True
        except asyncio.TimeoutError:
            # Check if we should be the one to probe
            if should_probe(provider):
                return True  # caller will probe and report back
            continue
    return True


def get_status() -> dict:
    """Return circuit state for all providers (for /api/stats or dashboard)."""
    return {
        provider: {
            "is_open": c.is_open,
            "consecutive_failures": c.consecutive_failures,
            "last_failure_status": c.last_failure_status,
            "last_failure_message": c.last_failure_message,
            "seconds_since_failure": round(time.monotonic() - c.last_failure_time, 1) if c.last_failure_time else None,
            "probe_in_progress": c.probe_in_progress,
        }
        for provider, c in _circuits.items()
    }
