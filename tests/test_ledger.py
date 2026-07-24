"""Tests for the SQLite request ledger."""

import os
import sqlite3
import stat
import time

import pytest

from src import ledger
from src.usage import Usage, CostEstimate


@pytest.fixture(autouse=True)
def tmp_ledger(tmp_path, monkeypatch):
    db = tmp_path / "ledger.db"
    monkeypatch.setenv("MODEL_GATEWAY_LEDGER_PATH", str(db))
    # Reset any cached path by re-importing is unnecessary; ledger_path() reads env live.
    ledger.init()
    yield db


def _usage(reported=True, **kw):
    base = dict(input_tokens=0, output_tokens=0, cached_read_tokens=0,
                cache_write_tokens=0, reasoning_tokens=0, reported=reported)
    base.update(kw)
    return Usage(**base)


def test_record_and_recent(tmp_ledger):
    ledger.record(
        endpoint="/v1/chat/completions", method="POST", model="claude-sonnet-4.6",
        provider="anthropic", provider_model_id="claude-sonnet-4-6",
        status=200, latency_ms=123, is_stream=False,
        usage=_usage(input_tokens=1000, output_tokens=500, cached_read_tokens=200),
        cost=CostEstimate(cost_usd=0.012, pricing_complete=True, missing_classes=[]),
    )
    rows = ledger.recent()
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "claude-sonnet-4.6"
    assert r["provider"] == "anthropic"
    assert r["status"] == 200
    assert r["input_tokens"] == 1000
    assert r["output_tokens"] == 500
    assert r["cached_read_tokens"] == 200
    assert r["cost_usd"] == 0.012
    assert r["is_stream"] == 0
    assert "ts_iso" in r


def test_record_unreported_usage_null_cost(tmp_ledger):
    ledger.record(
        endpoint="/v1/messages", method="POST", model="glm-5.2-zai",
        provider="zai_coding", provider_model_id="glm-5.2",
        status=200, latency_ms=50, is_stream=True,
        usage=_usage(reported=False),
        cost=CostEstimate(cost_usd=None, pricing_complete=False, missing_classes=[]),
    )
    r = ledger.recent()[0]
    assert r["usage_reported"] == 0
    assert r["cost_usd"] is None
    assert r["is_stream"] == 1


def test_record_missing_pricing_classes_stored_as_json(tmp_ledger):
    ledger.record(
        endpoint="/v1/chat/completions", method="POST", model="x",
        provider="openai", provider_model_id="gpt-5.4",
        status=200, latency_ms=10, is_stream=False,
        usage=_usage(cache_write_tokens=50),
        cost=CostEstimate(cost_usd=0.001, pricing_complete=False, missing_classes=["cache_write"]),
    )
    r = ledger.recent()[0]
    assert r["missing_pricing_classes"] == ["cache_write"]
    assert r["pricing_complete"] == 0


def test_aggregate_by_model(tmp_ledger):
    for _ in range(3):
        ledger.record(endpoint="/v1/chat/completions", method="POST", model="a",
                      provider="anthropic", provider_model_id="a", status=200,
                      latency_ms=100, is_stream=False,
                      usage=_usage(input_tokens=100, output_tokens=50),
                      cost=CostEstimate(0.01, True, []))
    ledger.record(endpoint="/v1/chat/completions", method="POST", model="b",
                  provider="openai", provider_model_id="b", status=200,
                  latency_ms=200, is_stream=True,
                  usage=_usage(reported=False),
                  cost=CostEstimate(None, False, []), error="stream failed")
    by_model = ledger.aggregate(group_by="model")
    dims = {row["dim"]: row for row in by_model}
    assert dims["a"]["requests"] == 3
    assert dims["a"]["ok"] == 3
    assert dims["a"]["errors"] == 0
    assert dims["a"]["input_tokens"] == 300
    assert dims["a"]["cost_usd"] == 0.03
    assert dims["a"]["usage_reported_requests"] == 3
    assert dims["a"]["known_cost_requests"] == 3
    assert dims["a"]["complete_pricing_requests"] == 3
    assert dims["b"]["requests"] == 1
    assert dims["b"]["errors"] == 1
    assert dims["b"]["cost_usd"] is None
    assert dims["b"]["missing_usage_requests"] == 1
    assert dims["b"]["unknown_cost_requests"] == 1


def test_aggregate_window_filters_by_time(tmp_ledger):
    ledger.record(endpoint="/v1/messages", method="POST", model="a",
                  provider="anthropic", provider_model_id="a", status=200,
                  latency_ms=10, is_stream=False, usage=_usage(input_tokens=10),
                  cost=CostEstimate(0.001, True, []))
    # Future cutoff excludes everything
    future = time.time() + 3600
    assert ledger.aggregate(since=future, group_by="model") == []


def test_summary_totals(tmp_ledger):
    ledger.record(endpoint="/v1/chat/completions", method="POST", model="a",
                  provider="anthropic", provider_model_id="a", status=200,
                  latency_ms=100, is_stream=False,
                  usage=_usage(input_tokens=1000, output_tokens=500),
                  cost=CostEstimate(0.05, True, []))
    s = ledger.summary()
    assert s["requests"] == 1
    assert s["ok"] == 1
    assert s["input_tokens"] == 1000
    assert s["cost_usd"] == 0.05
    assert s["usage_reported_requests"] == 1
    assert s["known_cost_requests"] == 1
    assert s["unknown_cost_requests"] == 0
    assert s["complete_pricing_requests"] == 1
    assert s["first_ts"] is not None


def test_record_best_effort_does_not_raise_on_bad_path(monkeypatch, tmp_path):
    # Point ledger at a path inside a file (not a dir) to force a sqlite error.
    bad = tmp_path / "notadir"
    bad.write_text("x")
    monkeypatch.setenv("MODEL_GATEWAY_LEDGER_PATH", str(bad / "ledger.db"))
    # Should not raise.
    ledger.record(
        endpoint="/v1/chat/completions", method="POST", model="a",
        provider="anthropic", provider_model_id="a", status=200,
        latency_ms=1, is_stream=False, usage=_usage(),
        cost=CostEstimate(None, False, []),
    )


def test_group_by_validation():
    with pytest.raises(ValueError):
        ledger.aggregate(group_by="bogus")


def test_session_fingerprint_is_stable_private_and_redacted(tmp_ledger):
    key_path = ledger.session_fingerprint_key_path()
    assert key_path.exists()
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert len(key_path.read_bytes()) == 32

    raw_session_id = "session-raw-value-must-not-be-stored"
    fingerprint = ledger.session_fingerprint(raw_session_id)
    assert fingerprint == ledger.session_fingerprint(raw_session_id)
    assert fingerprint != raw_session_id
    assert fingerprint.startswith("h1:")
    assert len(fingerprint) == 67

    ledger.record(
        endpoint="/v1/messages", method="POST", model="opus5",
        provider="anthropic", provider_model_id="claude-opus-5", status=200,
        latency_ms=10, is_stream=False, usage=_usage(input_tokens=10),
        cost=CostEstimate(0.001, True, []), session_fingerprint=fingerprint,
        session_source="x-session-affinity", cache_retention_requested="short",
    )
    recent = ledger.recent()[0]
    assert recent["session_observed"] is True
    assert recent["session_source"] == "x-session-affinity"
    assert recent["cache_retention_requested"] == "short"
    assert "session_fingerprint" not in recent

    with sqlite3.connect(tmp_ledger) as conn:
        stored = conn.execute(
            "SELECT session_fingerprint FROM requests"
        ).fetchone()[0]
    assert stored == fingerprint
    assert raw_session_id not in tmp_ledger.read_bytes().decode("latin-1")


def test_record_rejects_unversioned_or_raw_session_fingerprints(tmp_ledger):
    ledger.record(
        endpoint="/v1/messages", method="POST", model="opus5",
        provider="anthropic", provider_model_id="claude-opus-5", status=200,
        latency_ms=10, is_stream=False, usage=_usage(input_tokens=10),
        cost=CostEstimate(0.001, True, []),
        session_fingerprint="raw-internal-caller",
        session_source="RAW-SESSION-SOURCE",
        cache_retention_requested="RAW-RETENTION",
    )
    valid_fingerprint = ledger.session_fingerprint("valid-session")
    ledger.record(
        endpoint="/v1/messages", method="POST", model="opus5",
        provider="anthropic", provider_model_id="claude-opus-5", status=200,
        latency_ms=10, is_stream=False, usage=_usage(input_tokens=10),
        cost=CostEstimate(0.001, True, []),
        session_fingerprint=valid_fingerprint,
        session_source="RAW-SESSION-SOURCE",
        cache_retention_requested="RAW-RETENTION",
    )
    recent = ledger.recent()
    assert recent[0]["session_observed"] is True
    assert recent[1]["session_observed"] is False
    assert all(row["session_source"] is None for row in recent)
    assert all(row["cache_retention_requested"] is None for row in recent)
    with sqlite3.connect(tmp_ledger) as conn:
        stored = conn.execute(
            "SELECT session_fingerprint, session_source, cache_retention_requested "
            "FROM requests ORDER BY ts"
        ).fetchall()
    assert stored[0] == (None, None, None)
    assert stored[1] == (valid_fingerprint, None, None)
    contents = tmp_ledger.read_bytes().decode("latin-1")
    assert "raw-internal-caller" not in contents
    assert "RAW-SESSION-SOURCE" not in contents
    assert "RAW-RETENTION" not in contents


def test_session_fingerprinting_fails_closed_for_permissive_key(tmp_path, monkeypatch):
    key_path = tmp_path / "unsafe.key"
    key_path.write_bytes(b"x" * 32)
    key_path.chmod(0o644)
    monkeypatch.setenv("MODEL_GATEWAY_SESSION_FINGERPRINT_KEY_FILE", str(key_path))
    assert ledger.session_fingerprint("session") is None


def test_additive_migration_adds_cache_observation_columns(tmp_path, monkeypatch):
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE requests (
                id TEXT PRIMARY KEY, ts REAL NOT NULL, endpoint TEXT NOT NULL,
                method TEXT NOT NULL, model TEXT, provider TEXT,
                provider_model_id TEXT, status INTEGER, latency_ms INTEGER,
                is_stream INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cached_read_tokens INTEGER NOT NULL DEFAULT 0,
                cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                cache_write_1h_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                usage_reported INTEGER NOT NULL DEFAULT 0, cost_usd REAL,
                pricing_complete INTEGER NOT NULL DEFAULT 0,
                missing_pricing_classes TEXT, error TEXT
            )
            """
        )
    db.chmod(0o644)
    monkeypatch.setenv("MODEL_GATEWAY_LEDGER_PATH", str(db))
    ledger.init()
    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(requests)")}
    assert {
        "request_started_at", "session_fingerprint", "session_source",
        "cache_retention_requested",
    } <= columns
    assert "idx_requests_session_model_started" in indexes
    assert stat.S_IMODE(db.stat().st_mode) == 0o600
    ledger.record(
        endpoint="/v1/messages", method="POST", model="opus5",
        provider="anthropic", provider_model_id="claude-opus-5", status=200,
        latency_ms=10, is_stream=False, usage=_usage(input_tokens=10),
        cost=CostEstimate(0.001, True, []), request_started_at=123.0,
        session_fingerprint=ledger.session_fingerprint("migrated-session"),
        session_source="x-session-affinity", cache_retention_requested="short",
    )
    assert ledger.recent()[0]["session_observed"] is True


def test_ledger_and_sqlite_sidecars_are_private(tmp_ledger):
    ledger.record(
        endpoint="/v1/messages", method="POST", model="opus5",
        provider="anthropic", provider_model_id="claude-opus-5", status=200,
        latency_ms=10, is_stream=False, usage=_usage(input_tokens=10),
        cost=CostEstimate(0.001, True, []),
    )
    paths = [
        tmp_ledger,
        tmp_ledger.with_name(f"{tmp_ledger.name}-wal"),
        tmp_ledger.with_name(f"{tmp_ledger.name}-shm"),
    ]
    for path in paths:
        if path.exists():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_cache_retention_analysis_counts_only_same_session_short_rewrites(tmp_ledger):
    fingerprint = ledger.session_fingerprint("session-a")

    def record_at(ts, *, retention="short", cache_write=0, cache_read=0, provider="anthropic"):
        ledger.record(
            endpoint="/v1/messages", method="POST", model="opus5",
            provider=provider, provider_model_id="claude-opus-5", status=200,
            latency_ms=10, is_stream=False,
            usage=_usage(input_tokens=10, cache_write_tokens=cache_write,
                         cached_read_tokens=cache_read),
            cost=CostEstimate(0.001, True, []), request_started_at=ts,
            session_fingerprint=fingerprint, session_source="x-session-affinity",
            cache_retention_requested=retention,
        )

    record_at(1_000, cache_write=500)
    record_at(1_300, cache_write=400)  # inclusive 5-minute boundary
    record_at(4_900, cache_write=300)  # inclusive 1-hour boundary
    record_at(8_501, cache_write=200)  # outside the window
    record_at(9_000, retention="long", cache_write=0, cache_read=500)
    record_at(1_300, cache_write=999, provider="openai")  # excluded provider

    result = ledger.cache_retention_analysis()
    assert result == {
        "session_observed_requests": 5,
        "long_requests": 1,
        "short_requests": 4,
        "short_sessions": 1,
        "eligible_gap_requests": 2,
        "rewrite_after_gap_requests": 2,
        "rewrite_after_gap_tokens": 700,
        "min_gap_seconds": 300,
        "max_gap_seconds": 3600,
    }
    windowed = ledger.cache_retention_analysis(since=1_200, until=5_000)
    assert windowed["eligible_gap_requests"] == 2
    assert windowed["rewrite_after_gap_tokens"] == 700
    with pytest.raises(ValueError):
        ledger.cache_retention_analysis(min_gap_seconds=10, max_gap_seconds=5)


def test_cache_retention_analysis_requires_contiguous_short_active_predecessor(tmp_ledger):
    fingerprint_a = ledger.session_fingerprint("session-a")
    fingerprint_b = ledger.session_fingerprint("session-b")

    def record_at(session, ts, *, retention="short", reported=True, cache_write=0, cache_read=0):
        ledger.record(
            endpoint="/v1/messages", method="POST", model="opus5",
            provider="anthropic", provider_model_id="claude-opus-5", status=200,
            latency_ms=60_000, is_stream=True,
            usage=_usage(reported=reported, input_tokens=10,
                         cache_write_tokens=cache_write, cached_read_tokens=cache_read),
            cost=CostEstimate(0.001, True, []), request_started_at=ts,
            session_fingerprint=session, session_source="x-session-affinity",
            cache_retention_requested=retention,
        )

    record_at(fingerprint_a, 1_000, cache_write=500)
    record_at(fingerprint_a, 1_400, retention="long", cache_read=500)
    record_at(fingerprint_a, 1_700, cache_write=400)  # previous request was long
    record_at(fingerprint_a, 2_000, reported=False)
    record_at(fingerprint_a, 2_400, cache_write=300)  # previous usage was unreported

    record_at(fingerprint_b, 1_000)  # no prior cache activity
    record_at(fingerprint_b, 1_300, cache_write=200)

    result = ledger.cache_retention_analysis()
    assert result["eligible_gap_requests"] == 0
    assert result["rewrite_after_gap_requests"] == 0
    assert result["rewrite_after_gap_tokens"] == 0


def test_summary_and_recent_filtered_by_model(tmp_ledger):
    ledger.record(endpoint="/v1/messages", method="POST", model="glm-5.2",
                  provider="zai_coding", provider_model_id="glm-5.2", status=200,
                  latency_ms=100, is_stream=False,
                  usage=_usage(input_tokens=100, output_tokens=50),
                  cost=CostEstimate(0.02, True, []))
    ledger.record(endpoint="/v1/messages", method="POST", model="claude",
                  provider="anthropic", provider_model_id="claude", status=200,
                  latency_ms=80, is_stream=False,
                  usage=_usage(input_tokens=10, output_tokens=5),
                  cost=CostEstimate(0.01, True, []))
    # Filter to glm-5.2 across all its routable ids.
    s = ledger.summary(models=["glm-5.2", "glm52"])
    assert s["requests"] == 1
    assert s["input_tokens"] == 100
    assert s["cost_usd"] == 0.02
    # recent() respects the same filter.
    rows = ledger.recent(limit=10, models=["glm-5.2"])
    assert len(rows) == 1
    assert rows[0]["model"] == "glm-5.2"
    # No matches for an unknown id set.
    assert ledger.summary(models=["does-not-exist"])["requests"] == 0
    assert ledger.recent(limit=10, models=["does-not-exist"]) == []
    # Empty/None filter returns everything.
    assert ledger.summary()["requests"] == 2
