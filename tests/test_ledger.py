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


def test_aggregate_by_route_collapses_aliases_without_merging_providers(tmp_ledger):
    rows = [
        ("glm-5.2-zai", "zai_coding", "glm-5.2", 100, 0.01),
        ("glm-5.2", "zai_coding", "glm-5.2", 200, 0.02),
        ("glm52zai", "zai_coding", "glm-5.2", 300, 0.03),
        # The same upstream id at another provider remains a separate route.
        ("other-glm", "other_provider", "glm-5.2", 400, 0.04),
        # Missing route metadata falls back to the requested model. Empty and
        # NULL route values normalize into the same group.
        ("unresolved-a", None, None, 500, None),
        ("unresolved-a", "", "", 700, None),
        ("unresolved-b", None, None, 900, None),
        ("unresolved-c", None, "upstream-only", 1000, None),
        ("unresolved-d", "", "upstream-only", 1100, None),
        # Complete and incomplete identities with the same visible values must
        # remain separate namespaces.
        ("logical-alias", "collision-provider", "collision-id", 1200, 0.05),
        ("collision-id", "collision-provider", None, 1300, 0.06),
    ]
    for model, provider, provider_model_id, latency, cost in rows:
        ledger.record(
            endpoint="/v1/chat/completions", method="POST", model=model,
            provider=provider, provider_model_id=provider_model_id, status=200,
            latency_ms=latency, is_stream=False,
            usage=_usage(input_tokens=10, output_tokens=5),
            cost=CostEstimate(cost, cost is not None, []),
        )

    aggregated = ledger.aggregate(group_by="route")
    routes = {
        (row["provider"], row["dim"]): row
        for row in aggregated
        if row["provider"] != "collision-provider"
    }
    zai = routes[("zai_coding", "glm-5.2")]
    assert zai["provider_model_id"] == "glm-5.2"
    assert zai["requests"] == 3
    assert zai["input_tokens"] == 30
    assert zai["cost_usd"] == pytest.approx(0.06)
    assert zai["avg_latency_ms"] == 200
    assert routes[("other_provider", "glm-5.2")]["requests"] == 1
    assert routes[(None, "unresolved-a")]["requests"] == 2
    assert routes[(None, "unresolved-a")]["provider_model_id"] is None
    assert routes[(None, "unresolved-b")]["requests"] == 1
    assert routes[(None, "unresolved-c")]["requests"] == 1
    assert routes[(None, "unresolved-c")]["provider_model_id"] is None
    assert routes[(None, "unresolved-d")]["requests"] == 1

    collisions = [
        row for row in aggregated
        if row["provider"] == "collision-provider" and row["dim"] == "collision-id"
    ]
    assert len(collisions) == 2
    assert {row["route_complete"] for row in collisions} == {0, 1}
    assert {row["requests"] for row in collisions} == {1}


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


def test_ledger_and_sqlite_sidecars_are_private(tmp_ledger):
    ledger.record(
        endpoint="/v1/messages", method="POST", model="opus5",
        provider="anthropic", provider_model_id="claude-opus-5", status=200,
        latency_ms=10, is_stream=False, usage=_usage(input_tokens=10),
        cost=CostEstimate(0.001, True, []),
    )
    with ledger._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE requests SET latency_ms = latency_ms")
        paths = [
            tmp_ledger,
            tmp_ledger.with_name(f"{tmp_ledger.name}-wal"),
            tmp_ledger.with_name(f"{tmp_ledger.name}-shm"),
        ]
        for path in paths:
            assert path.exists()
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        conn.rollback()


def test_recent_redacts_dormant_cache_observability_columns(tmp_ledger):
    ledger.record(
        endpoint="/v1/messages", method="POST", model="opus5",
        provider="anthropic", provider_model_id="claude-opus-5", status=200,
        latency_ms=10, is_stream=False, usage=_usage(input_tokens=10),
        cost=CostEstimate(0.001, True, []),
    )
    with sqlite3.connect(tmp_ledger) as conn:
        conn.execute("ALTER TABLE requests ADD COLUMN request_started_at REAL")
        conn.execute("ALTER TABLE requests ADD COLUMN session_fingerprint TEXT")
        conn.execute("ALTER TABLE requests ADD COLUMN session_source TEXT")
        conn.execute("ALTER TABLE requests ADD COLUMN cache_retention_requested TEXT")
        conn.execute(
            "UPDATE requests SET request_started_at=?, session_fingerprint=?, "
            "session_source=?, cache_retention_requested=?",
            (123.0, "h1:not-returned", "x-session-affinity", "short"),
        )
    row = ledger.recent()[0]
    assert "request_started_at" not in row
    assert "session_fingerprint" not in row
    assert "session_source" not in row
    assert "cache_retention_requested" not in row


def test_group_by_validation():
    with pytest.raises(ValueError):
        ledger.aggregate(group_by="bogus")


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
