"""Tests for the SQLite request ledger."""

import os
import time

import pytest

from src import ledger
from src.usage import Usage, CostEstimate


@pytest.fixture(autouse=True)
def tmp_ledger(tmp_path, monkeypatch):
    db = tmp_path / "ledger.db"
    monkeypatch.setenv("CLOUD_GATEWAY_LEDGER_PATH", str(db))
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
                  provider="openai", provider_model_id="b", status=500,
                  latency_ms=200, is_stream=False,
                  usage=_usage(reported=False),
                  cost=CostEstimate(None, False, []))
    by_model = ledger.aggregate(group_by="model")
    dims = {row["dim"]: row for row in by_model}
    assert dims["a"]["requests"] == 3
    assert dims["a"]["ok"] == 3
    assert dims["a"]["errors"] == 0
    assert dims["a"]["input_tokens"] == 300
    assert dims["a"]["cost_usd"] == 0.03
    assert dims["b"]["requests"] == 1
    assert dims["b"]["errors"] == 1
    assert dims["b"]["cost_usd"] is None


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
    assert s["first_ts"] is not None


def test_record_best_effort_does_not_raise_on_bad_path(monkeypatch, tmp_path):
    # Point ledger at a path inside a file (not a dir) to force a sqlite error.
    bad = tmp_path / "notadir"
    bad.write_text("x")
    monkeypatch.setenv("CLOUD_GATEWAY_LEDGER_PATH", str(bad / "ledger.db"))
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
