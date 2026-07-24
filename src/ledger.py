"""SQLite-backed request/usage ledger for model-gateway.

Records one row per ``/v1/*`` request with timing, status, normalized token
usage, and an estimated cost. Prompts/completions are never stored (privacy:
see docs/productionization-plan.md "redact prompts by default").

The database lives in the shared runtime state dir so it survives deploys:
``~/srv/model-gateway/shared/ledger.db`` by default, overridable via
``MODEL_GATEWAY_LEDGER_PATH``. Writes are best-effort: a ledger failure must
never break a model request.

Schema is intentionally flat (one table) for v1. Migration is additive: if
the schema gains columns later, ``_ensure_schema`` checks ``PRAGMA
table_info`` and adds missing columns.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.usage import CostEstimate, Usage

_DEFAULT_LEDGER_DIR = Path.home() / "srv" / "model-gateway" / "shared"
_DEFAULT_LEDGER_PATH = _DEFAULT_LEDGER_DIR / "ledger.db"
_SESSION_FINGERPRINT_KEY_BYTES = 32
_SESSION_FINGERPRINT_KEY_NAME = "session-fingerprint.key"
_MAX_SESSION_ID_CHARS = 2048
_SESSION_FINGERPRINT_RE = re.compile(r"^h1:[0-9a-f]{64}$")
_SESSION_SOURCES = frozenset({
    "x-session-affinity",
    "x-session-id",
    "session_id",
    "x-client-request-id",
    "prompt_cache_key",
})
_CACHE_RETENTIONS = frozenset({"short", "long"})

_lock = threading.Lock()
log = logging.getLogger("model-gateway.ledger")

# Columns added after the initial schema. Each is checked/applied on startup
# so older ledger.db files upgrade in place.
_ADDITIVE_COLUMNS: dict[str, str] = {
    "cache_write_1h_tokens": "INTEGER NOT NULL DEFAULT 0",
    "request_started_at": "REAL",
    "session_fingerprint": "TEXT",
    "session_source": "TEXT",
    "cache_retention_requested": "TEXT",
}


def ledger_path() -> Path:
    return Path(os.environ.get("MODEL_GATEWAY_LEDGER_PATH", str(_DEFAULT_LEDGER_PATH)))


def session_fingerprint_key_path() -> Path:
    """Return the stable local HMAC key path used for session pseudonyms."""
    configured = os.environ.get("MODEL_GATEWAY_SESSION_FINGERPRINT_KEY_FILE")
    if configured:
        return Path(configured).expanduser()
    return ledger_path().with_name(_SESSION_FINGERPRINT_KEY_NAME)


def _load_or_create_session_fingerprint_key() -> bytes | None:
    """Load a private stable key, creating it once with mode 0600.

    Invalid or permissive key files disable fingerprinting rather than falling
    back to an ephemeral key that would silently corrupt longitudinal metrics.
    """
    path = session_fingerprint_key_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as fh:
                fh.write(secrets.token_bytes(_SESSION_FINGERPRINT_KEY_BYTES))
                fh.flush()
                os.fsync(fh.fileno())

        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            log.warning(
                "session fingerprint key has unsafe permissions %o; expected 600: %s",
                mode,
                path,
            )
            return None
        key = path.read_bytes()
        if len(key) != _SESSION_FINGERPRINT_KEY_BYTES:
            log.warning(
                "session fingerprint key must contain exactly %d bytes: %s",
                _SESSION_FINGERPRINT_KEY_BYTES,
                path,
            )
            return None
        return key
    except OSError as exc:
        log.warning("session fingerprinting disabled: %s", exc)
        return None


def session_fingerprint(session_id: str | None) -> str | None:
    """Return a stable HMAC pseudonym without retaining the raw session id."""
    if not isinstance(session_id, str) or not session_id or len(session_id) > _MAX_SESSION_ID_CHARS:
        return None
    key = _load_or_create_session_fingerprint_key()
    if key is None:
        return None
    digest = hmac.new(key, session_id.encode("utf-8", errors="surrogatepass"), hashlib.sha256).hexdigest()
    return f"h1:{digest}"


def _restrict_sqlite_permissions(path: Path) -> None:
    """Keep the ledger and any SQLite sidecars private to the service user."""
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            candidate.chmod(0o600)
        except FileNotFoundError:
            continue


@contextmanager
def _connect():
    """Yield a private sqlite3 connection with WAL + reasonable pragmas."""
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        os.close(fd)
    _restrict_sqlite_permissions(path)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _restrict_sqlite_permissions(path)
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()
        _restrict_sqlite_permissions(path)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            request_started_at REAL,
            endpoint TEXT NOT NULL,
            method TEXT NOT NULL,
            model TEXT,
            provider TEXT,
            provider_model_id TEXT,
            status INTEGER,
            latency_ms INTEGER,
            is_stream INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cached_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_1h_tokens INTEGER NOT NULL DEFAULT 0,
            reasoning_tokens INTEGER NOT NULL DEFAULT 0,
            usage_reported INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL,
            pricing_complete INTEGER NOT NULL DEFAULT 0,
            missing_pricing_classes TEXT,
            session_fingerprint TEXT,
            session_source TEXT,
            cache_retention_requested TEXT,
            error TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_provider ON requests(provider)")
    # Additive column migration for older DBs.
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(requests)")}
    for col, decl in _ADDITIVE_COLUMNS.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE requests ADD COLUMN {col} {decl}")
    # This index must be created after additive migration on existing ledgers.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_requests_session_model_started "
        "ON requests(session_fingerprint, model, request_started_at, ts)"
    )


def init() -> None:
    """Create the schema and stable session-fingerprint key if needed."""
    with _lock, _connect() as conn:
        _ensure_schema(conn)
    _load_or_create_session_fingerprint_key()


def record(
    *,
    endpoint: str,
    method: str,
    model: str | None,
    provider: str | None,
    provider_model_id: str | None,
    status: int | None,
    latency_ms: int | None,
    is_stream: bool,
    usage: Usage,
    cost: CostEstimate,
    request_started_at: float | None = None,
    session_fingerprint: str | None = None,
    session_source: str | None = None,
    cache_retention_requested: str | None = None,
    error: str | None = None,
) -> str:
    """Insert one ledger row. Best-effort: logs and never raises.

    Returns the generated request id (also usable as a correlation id by the
    caller before insertion).
    """
    rid = uuid.uuid4().hex
    if session_fingerprint is not None and not _SESSION_FINGERPRINT_RE.fullmatch(session_fingerprint):
        # Enforce pseudonymization at the persistence boundary. Never log the
        # rejected value because an internal caller may have passed a raw id.
        session_fingerprint = None
    if session_fingerprint is None or session_source not in _SESSION_SOURCES:
        session_source = None
    if cache_retention_requested not in _CACHE_RETENTIONS:
        cache_retention_requested = None
    try:
        with _lock, _connect() as conn:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO requests (
                    id, ts, request_started_at, endpoint, method, model, provider, provider_model_id,
                    status, latency_ms, is_stream,
                    input_tokens, output_tokens, cached_read_tokens,
                    cache_write_tokens, cache_write_1h_tokens, reasoning_tokens,
                    usage_reported, cost_usd, pricing_complete,
                    missing_pricing_classes, session_fingerprint, session_source,
                    cache_retention_requested, error
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rid,
                    time.time(),
                    request_started_at,
                    endpoint,
                    method,
                    model,
                    provider,
                    provider_model_id,
                    status,
                    latency_ms,
                    1 if is_stream else 0,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cached_read_tokens,
                    usage.cache_write_tokens,
                    usage.cache_write_1h_tokens,
                    usage.reasoning_tokens,
                    1 if usage.reported else 0,
                    cost.cost_usd,
                    1 if cost.pricing_complete else 0,
                    json.dumps(cost.missing_classes) if cost.missing_classes else None,
                    session_fingerprint,
                    session_source,
                    cache_retention_requested,
                    error,
                ),
            )
    except Exception as exc:  # noqa: BLE001 - ledger must never break requests
        # Late import avoids a circular dependency at module load.
        import logging

        logging.getLogger("model-gateway.ledger").warning("ledger record failed: %s", exc)
    return rid


def recent(limit: int = 50, *, models: list[str] | None = None) -> list[dict]:
    """Return the most recent ledger rows (newest first).

    If ``models`` is given, restrict to rows whose ``model`` column matches any
    of the supplied identifiers (used for per-model stats click-through).
    """
    clauses: list[str] = []
    params: list = []
    if models:
        placeholders = ",".join("?" for _ in models)
        clauses.append(f"model IN ({placeholders})")
        params.extend(models)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _lock, _connect() as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            f"SELECT * FROM requests {where} ORDER BY ts DESC LIMIT ?",
            (*params, int(limit)),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def aggregate(
    *,
    since: float | None = None,
    until: float | None = None,
    group_by: str = "provider",
) -> list[dict]:
    """Aggregate requests/tokens/cost/latency/errors by a dimension.

    ``group_by`` is one of: provider, model, endpoint, status. Time bounds are
    epoch seconds (inclusive). Cost sums ignore NULL (unknown) rows.
    """
    allowed = {"provider", "model", "endpoint", "status"}
    if group_by not in allowed:
        raise ValueError(f"group_by must be one of {sorted(allowed)}")
    clauses = []
    params: list = []
    if since is not None:
        clauses.append("ts >= ?")
        params.append(since)
    if until is not None:
        clauses.append("ts < ?")
        params.append(until)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    sql = f"""
        SELECT
            {group_by} AS dim,
            COUNT(*) AS requests,
            SUM(CASE WHEN status >= 200 AND status < 300 AND error IS NULL THEN 1 ELSE 0 END) AS ok,
            SUM(CASE WHEN error IS NOT NULL OR (status IS NOT NULL AND status >= 400) THEN 1 ELSE 0 END) AS errors,
            SUM(input_tokens) AS input_tokens,
            SUM(output_tokens) AS output_tokens,
            SUM(cached_read_tokens) AS cached_read_tokens,
            SUM(cache_write_tokens) AS cache_write_tokens,
            SUM(cache_write_1h_tokens) AS cache_write_1h_tokens,
            SUM(reasoning_tokens) AS reasoning_tokens,
            SUM(cost_usd) AS cost_usd,
            SUM(CASE WHEN usage_reported = 1 THEN 1 ELSE 0 END) AS usage_reported_requests,
            SUM(CASE WHEN usage_reported = 0 THEN 1 ELSE 0 END) AS missing_usage_requests,
            SUM(CASE WHEN cost_usd IS NOT NULL THEN 1 ELSE 0 END) AS known_cost_requests,
            SUM(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END) AS unknown_cost_requests,
            SUM(CASE WHEN usage_reported = 1 AND cost_usd IS NULL THEN 1 ELSE 0 END) AS missing_pricing_requests,
            SUM(CASE WHEN pricing_complete = 1 THEN 1 ELSE 0 END) AS complete_pricing_requests,
            SUM(CASE WHEN cost_usd IS NOT NULL AND pricing_complete = 0 THEN 1 ELSE 0 END) AS partial_pricing_requests,
            AVG(latency_ms) AS avg_latency_ms,
            MAX(latency_ms) AS max_latency_ms
        FROM requests
        {where}
        GROUP BY {group_by}
        ORDER BY cost_usd DESC, requests DESC
    """
    with _lock, _connect() as conn:
        _ensure_schema(conn)
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def summary(*, since: float | None = None, until: float | None = None, models: list[str] | None = None) -> dict:
    """Return top-level totals for the dashboard header.

    If ``models`` is given, restrict to rows whose ``model`` column matches any
    of the supplied identifiers (per-model stats).
    """
    clauses = []
    params: list = []
    if since is not None:
        clauses.append("ts >= ?")
        params.append(since)
    if until is not None:
        clauses.append("ts < ?")
        params.append(until)
    if models:
        placeholders = ",".join("?" for _ in models)
        clauses.append(f"model IN ({placeholders})")
        params.extend(models)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT
            COUNT(*) AS requests,
            SUM(CASE WHEN status >= 200 AND status < 300 AND error IS NULL THEN 1 ELSE 0 END) AS ok,
            SUM(CASE WHEN error IS NOT NULL OR (status IS NOT NULL AND status >= 400) THEN 1 ELSE 0 END) AS errors,
            SUM(input_tokens) AS input_tokens,
            SUM(output_tokens) AS output_tokens,
            SUM(cached_read_tokens) AS cached_read_tokens,
            SUM(cache_write_tokens) AS cache_write_tokens,
            SUM(cache_write_1h_tokens) AS cache_write_1h_tokens,
            SUM(reasoning_tokens) AS reasoning_tokens,
            SUM(cost_usd) AS cost_usd,
            SUM(CASE WHEN usage_reported = 1 THEN 1 ELSE 0 END) AS usage_reported_requests,
            SUM(CASE WHEN usage_reported = 0 THEN 1 ELSE 0 END) AS missing_usage_requests,
            SUM(CASE WHEN cost_usd IS NOT NULL THEN 1 ELSE 0 END) AS known_cost_requests,
            SUM(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END) AS unknown_cost_requests,
            SUM(CASE WHEN usage_reported = 1 AND cost_usd IS NULL THEN 1 ELSE 0 END) AS missing_pricing_requests,
            SUM(CASE WHEN pricing_complete = 1 THEN 1 ELSE 0 END) AS complete_pricing_requests,
            SUM(CASE WHEN cost_usd IS NOT NULL AND pricing_complete = 0 THEN 1 ELSE 0 END) AS partial_pricing_requests,
            AVG(latency_ms) AS avg_latency_ms,
            MIN(ts) AS first_ts,
            MAX(ts) AS last_ts
        FROM requests
        {where}
    """
    with _lock, _connect() as conn:
        _ensure_schema(conn)
        row = conn.execute(sql, params).fetchone()
    if row is None:
        return {}
    d = dict(row)
    # Normalize averages/sums that may be NULL when the table is empty.
    for k in ("cost_usd",):
        if d.get(k) is not None:
            d[k] = round(float(d[k]), 6)
    if d.get("avg_latency_ms") is not None:
        d["avg_latency_ms"] = round(float(d["avg_latency_ms"]), 1)
    return d


def cache_retention_analysis(
    *,
    since: float | None = None,
    until: float | None = None,
    min_gap_seconds: float = 300,
    max_gap_seconds: float = 3600,
) -> dict:
    """Aggregate conservative Anthropic short-cache rewrite candidates.

    Ordering uses request ingress time rather than response/stream completion.
    ``LAG`` sees every successful same-session/model request, including long or
    usage-unreported requests and a predecessor just before ``since``. A gap is
    eligible only when both contiguous requests selected short retention and
    the predecessor reported real cache activity. Fingerprints never leave SQL.
    """
    if min_gap_seconds < 0 or max_gap_seconds < min_gap_seconds:
        raise ValueError("gap window must satisfy 0 <= min_gap_seconds <= max_gap_seconds")
    scan_since = since - max_gap_seconds if since is not None else None
    sql = """
        WITH raw_observed AS (
            SELECT
                *,
                COALESCE(request_started_at, ts) AS event_ts
            FROM requests
            WHERE session_fingerprint IS NOT NULL
              AND endpoint = '/v1/messages'
              AND provider = 'anthropic'
              AND status >= 200 AND status < 300
              AND error IS NULL
        ),
        scan_observed AS (
            SELECT * FROM raw_observed
            WHERE (:scan_since IS NULL OR event_ts >= :scan_since)
              AND (:until IS NULL OR event_ts < :until)
        ),
        ordered AS (
            SELECT
                *,
                LAG(event_ts) OVER (
                    PARTITION BY session_fingerprint, model ORDER BY event_ts, ts, id
                ) AS previous_ts,
                LAG(cache_retention_requested) OVER (
                    PARTITION BY session_fingerprint, model ORDER BY event_ts, ts, id
                ) AS previous_retention,
                LAG(usage_reported) OVER (
                    PARTITION BY session_fingerprint, model ORDER BY event_ts, ts, id
                ) AS previous_usage_reported,
                LAG(cached_read_tokens + cache_write_tokens + cache_write_1h_tokens) OVER (
                    PARTITION BY session_fingerprint, model ORDER BY event_ts, ts, id
                ) AS previous_cache_tokens
            FROM scan_observed
        ),
        observed_window AS (
            SELECT * FROM ordered
            WHERE (:since IS NULL OR event_ts >= :since)
        ),
        short_requests AS (
            SELECT * FROM observed_window
            WHERE cache_retention_requested = 'short' AND usage_reported = 1
        )
        SELECT
            (SELECT COUNT(*) FROM observed_window) AS session_observed_requests,
            (SELECT COUNT(*) FROM observed_window WHERE cache_retention_requested = 'long') AS long_requests,
            COUNT(*) AS short_requests,
            COUNT(DISTINCT session_fingerprint || char(0) || COALESCE(model, '')) AS short_sessions,
            COALESCE(SUM(CASE
                WHEN previous_ts IS NOT NULL
                 AND previous_retention = 'short'
                 AND previous_usage_reported = 1
                 AND previous_cache_tokens > 0
                 AND event_ts - previous_ts >= :min_gap
                 AND event_ts - previous_ts <= :max_gap
                THEN 1 ELSE 0 END), 0) AS eligible_gap_requests,
            COALESCE(SUM(CASE
                WHEN previous_ts IS NOT NULL
                 AND previous_retention = 'short'
                 AND previous_usage_reported = 1
                 AND previous_cache_tokens > 0
                 AND event_ts - previous_ts >= :min_gap
                 AND event_ts - previous_ts <= :max_gap
                 AND cached_read_tokens = 0 AND cache_write_tokens > 0
                THEN 1 ELSE 0 END), 0) AS rewrite_after_gap_requests,
            COALESCE(SUM(CASE
                WHEN previous_ts IS NOT NULL
                 AND previous_retention = 'short'
                 AND previous_usage_reported = 1
                 AND previous_cache_tokens > 0
                 AND event_ts - previous_ts >= :min_gap
                 AND event_ts - previous_ts <= :max_gap
                 AND cached_read_tokens = 0 AND cache_write_tokens > 0
                THEN cache_write_tokens ELSE 0 END), 0) AS rewrite_after_gap_tokens
        FROM short_requests
    """
    query_params = {
        "scan_since": scan_since,
        "since": since,
        "until": until,
        "min_gap": min_gap_seconds,
        "max_gap": max_gap_seconds,
    }
    with _lock, _connect() as conn:
        _ensure_schema(conn)
        row = conn.execute(sql, query_params).fetchone()
    result = dict(row) if row is not None else {}
    result["min_gap_seconds"] = min_gap_seconds
    result["max_gap_seconds"] = max_gap_seconds
    return result


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    fingerprint = d.pop("session_fingerprint", None)
    d["session_observed"] = fingerprint is not None
    if d.get("missing_pricing_classes"):
        try:
            d["missing_pricing_classes"] = json.loads(d["missing_pricing_classes"])
        except (TypeError, json.JSONDecodeError):
            pass
    if "ts" in d and d["ts"] is not None:
        d["ts_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(d["ts"]))
    return d
