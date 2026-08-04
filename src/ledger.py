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

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.usage import CostEstimate, Usage

_DEFAULT_LEDGER_DIR = Path.home() / "srv" / "model-gateway" / "shared"
_DEFAULT_LEDGER_PATH = _DEFAULT_LEDGER_DIR / "ledger.db"

_lock = threading.Lock()

# Columns added after the initial schema. Each is checked/applied on startup
# so older ledger.db files upgrade in place.
_ADDITIVE_COLUMNS: dict[str, str] = {
    "cache_write_1h_tokens": "INTEGER NOT NULL DEFAULT 0",
}


def ledger_path() -> Path:
    return Path(os.environ.get("MODEL_GATEWAY_LEDGER_PATH", str(_DEFAULT_LEDGER_PATH)))


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


def init() -> None:
    """Create the schema if needed. Safe to call at startup."""
    with _lock, _connect() as conn:
        _ensure_schema(conn)


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
    error: str | None = None,
) -> str:
    """Insert one ledger row. Best-effort: logs and never raises.

    Returns the generated request id (also usable as a correlation id by the
    caller before insertion).
    """
    rid = uuid.uuid4().hex
    try:
        with _lock, _connect() as conn:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO requests (
                    id, ts, endpoint, method, model, provider, provider_model_id,
                    status, latency_ms, is_stream,
                    input_tokens, output_tokens, cached_read_tokens,
                    cache_write_tokens, cache_write_1h_tokens, reasoning_tokens,
                    usage_reported, cost_usd, pricing_complete,
                    missing_pricing_classes, error
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rid,
                    time.time(),
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
                    error,
                ),
            )
    except Exception as exc:  # noqa: BLE001 - ledger must never break requests
        # Late import avoids a circular dependency at module load.
        import logging

        logging.getLogger("model-gateway.ledger").warning("ledger record failed: %s", exc)
    return rid


def recent(
    limit: int = 50,
    *,
    models: list[str] | None = None,
    provider: str | None = None,
) -> list[dict]:
    """Return the most recent ledger rows (newest first).

    If ``models`` is given, restrict to rows whose ``model`` column matches any
    of the supplied identifiers (used for per-model stats click-through).
    If ``provider`` is given, restrict to rows routed through that provider.
    """
    clauses: list[str] = []
    params: list = []
    if models:
        placeholders = ",".join("?" for _ in models)
        clauses.append(f"model IN ({placeholders})")
        params.extend(models)
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _lock, _connect() as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            f"SELECT * FROM requests {where} ORDER BY ts DESC LIMIT ?",
            (*params, int(limit)),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_request(request_id: str) -> dict | None:
    """Return one ledger row by id (redacted like :func:`recent`), or None."""
    with _lock, _connect() as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT * FROM requests WHERE id = ?", (request_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def aggregate(
    *,
    since: float | None = None,
    until: float | None = None,
    group_by: str = "provider",
) -> list[dict]:
    """Aggregate requests/tokens/cost/latency/errors by a dimension.

    ``group_by`` is one of: provider, model, route, endpoint, status. ``route``
    groups requested aliases by the initially resolved
    ``(provider, provider_model_id)`` pair and falls back to the requested model
    when route metadata is unavailable. Time bounds are epoch seconds
    (inclusive). Cost sums ignore NULL (unknown) rows.
    """
    allowed = {"provider", "model", "route", "endpoint", "status"}
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

    if group_by == "route":
        complete_route_sql = (
            "NULLIF(provider, '') IS NOT NULL "
            "AND NULLIF(provider_model_id, '') IS NOT NULL"
        )
        dimension_sql = (
            f"CASE WHEN {complete_route_sql} "
            "THEN provider_model_id ELSE model END"
        )
        route_columns_sql = (
            f", CASE WHEN {complete_route_sql} THEN 1 ELSE 0 END AS route_complete, "
            "NULLIF(provider, '') AS provider, "
            f"CASE WHEN {complete_route_sql} "
            "THEN provider_model_id END AS provider_model_id"
        )
        group_sql = (
            f"CASE WHEN {complete_route_sql} THEN 1 ELSE 0 END, "
            f"NULLIF(provider, ''), {dimension_sql}"
        )
    else:
        dimension_sql = group_by
        route_columns_sql = ""
        group_sql = group_by

    sql = f"""
        SELECT
            {dimension_sql} AS dim
            {route_columns_sql},
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
        GROUP BY {group_sql}
        ORDER BY cost_usd DESC, requests DESC
    """
    with _lock, _connect() as conn:
        _ensure_schema(conn)
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def summary(
    *,
    since: float | None = None,
    until: float | None = None,
    models: list[str] | None = None,
    provider: str | None = None,
) -> dict:
    """Return top-level totals for the dashboard header.

    If ``models`` is given, restrict to rows whose ``model`` column matches any
    of the supplied identifiers (per-model stats). If ``provider`` is given,
    restrict to rows routed through that provider (per-provider stats).
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
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
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


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    # Older ledgers may retain dormant cache-observability columns. Never
    # expose their historical session metadata through recent-request APIs.
    for column in (
        "request_started_at",
        "session_fingerprint",
        "session_source",
        "cache_retention_requested",
    ):
        d.pop(column, None)
    if d.get("missing_pricing_classes"):
        try:
            d["missing_pricing_classes"] = json.loads(d["missing_pricing_classes"])
        except (TypeError, json.JSONDecodeError):
            pass
    if "ts" in d and d["ts"] is not None:
        d["ts_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(d["ts"]))
    return d
