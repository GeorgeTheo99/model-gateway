"""Tests for usage normalization and cost estimation."""

import pytest

from src.usage import (
    Usage,
    anthropic_usage_to_openai_chat,
    anthropic_usage_to_responses,
    extract_usage,
    estimate_cost,
    openai_chat_usage_to_anthropic,
    openai_chat_usage_to_responses,
    usage_has_ledger_data,
    usage_was_reported,
)


def test_extract_openai_chat_shape():
    # OpenAI prompt_tokens INCLUDES cached tokens; extract_usage subtracts
    # them so input_tokens is cache-miss input (1000 - 200 = 800).
    resp = {"usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": 200},
        "completion_tokens_details": {"reasoning_tokens": 50},
    }}
    u = extract_usage(resp)
    assert u.reported is True
    assert u.input_tokens == 800
    assert u.output_tokens == 500
    assert u.cached_read_tokens == 200
    assert u.cache_write_tokens == 0
    assert u.reasoning_tokens == 50


def test_extract_moonshot_top_level_cache_shape():
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "cached_tokens": 200,
    }
    expected = Usage(
        input_tokens=800,
        output_tokens=500,
        cached_read_tokens=200,
        reported=True,
    )
    assert extract_usage({"usage": usage}) == expected
    assert openai_chat_usage_to_anthropic(usage) == {
        "input_tokens": 800,
        "output_tokens": 500,
        "cache_read_input_tokens": 200,
    }
    assert extract_usage({"usage": openai_chat_usage_to_responses(usage)}) == expected

    # Prefer the standard nested value if a provider emits both shapes.
    usage["prompt_tokens_details"] = {"cached_tokens": 150}
    assert extract_usage({"usage": usage}).cached_read_tokens == 150


def test_extract_anthropic_shape():
    resp = {"usage": {
        "input_tokens": 800,
        "output_tokens": 300,
        "cache_read_input_tokens": 150,
        "cache_creation_input_tokens": 400,
        "output_tokens_details": {"thinking_tokens": 75},
    }}
    u = extract_usage(resp)
    assert u.reported is True
    assert u.input_tokens == 800
    assert u.output_tokens == 300
    assert u.cached_read_tokens == 150
    assert u.cache_write_tokens == 400
    assert u.reasoning_tokens == 75


def test_extract_anthropic_shape_without_cache_fields():
    u = extract_usage({"usage": {"input_tokens": 11, "output_tokens": 7}})
    assert u.reported is True
    assert u.input_tokens == 11
    assert u.output_tokens == 7


def test_extract_responses_shape():
    # OpenAI Responses input_tokens INCLUDES cached; subtracted to cache-miss
    # (1200 - 300 = 900).
    resp = {"usage": {
        "input_tokens": 1200,
        "output_tokens": 600,
        "input_tokens_details": {"cached_tokens": 300},
        "output_tokens_details": {"reasoning_tokens": 100},
        "total_tokens": 1800,
    }}
    u = extract_usage(resp)
    assert u.reported is True
    assert u.input_tokens == 900
    assert u.output_tokens == 600
    assert u.cached_read_tokens == 300
    assert u.cache_write_tokens == 0
    assert u.reasoning_tokens == 100


def test_extract_no_usage_returns_unreported():
    u = extract_usage({"choices": []})
    assert u.reported is False
    assert u.input_tokens == 0
    assert extract_usage(None).reported is False
    assert usage_was_reported({}) is False
    assert usage_was_reported({"input_tokens": 0, "output_tokens": 0}) is True


def test_extract_provider_reported_cost_without_fabricating_token_coverage():
    usage = extract_usage({"usage": {"cost": 0.0004604}})
    assert usage.provider_cost_usd == 0.0004604
    assert usage.reported is False
    assert usage_has_ledger_data({"cost": 0.0004604}) is True
    assert usage_was_reported({"cost": 0.0004604}) is False


@pytest.mark.parametrize(
    "value",
    [-1, float("nan"), float("inf"), -float("inf"), 10**400, True, "0.5", None],
)
def test_extract_rejects_invalid_provider_reported_cost(value):
    usage = extract_usage({"usage": {"cost": value}})
    assert usage.provider_cost_usd is None
    assert usage_has_ledger_data({"cost": value}) is False


def test_usage_shape_conversions_preserve_provider_cost():
    chat = {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0004604}
    anthropic = openai_chat_usage_to_anthropic(chat)
    responses = openai_chat_usage_to_responses(chat)
    assert anthropic["cost"] == 0.0004604
    assert responses["cost"] == 0.0004604
    assert anthropic_usage_to_openai_chat(anthropic)["cost"] == 0.0004604
    assert anthropic_usage_to_responses(anthropic)["cost"] == 0.0004604
    assert openai_chat_usage_to_anthropic({"cost": 0.0004604}) == {"cost": 0.0004604}
    assert openai_chat_usage_to_responses({"cost": 0.0004604}) == {"cost": 0.0004604}


def test_usage_shape_conversions_preserve_cache_semantics():
    chat = {
        "prompt_tokens": 1000,
        "completion_tokens": 300,
        "prompt_tokens_details": {"cached_tokens": 150, "cache_write_tokens": 50},
        "completion_tokens_details": {"reasoning_tokens": 25},
    }
    anthropic = openai_chat_usage_to_anthropic(chat)
    assert anthropic == {
        "input_tokens": 800,
        "output_tokens": 300,
        "cache_read_input_tokens": 150,
        "cache_creation_input_tokens": 50,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 50,
            "ephemeral_1h_input_tokens": 0,
        },
        "output_tokens_details": {"thinking_tokens": 25},
    }
    responses = openai_chat_usage_to_responses(chat)
    assert extract_usage({"usage": responses}) == Usage(
        input_tokens=800, output_tokens=300, cached_read_tokens=150,
        cache_write_tokens=50, reasoning_tokens=25, reported=True,
    )

    round_trip_chat = anthropic_usage_to_openai_chat(anthropic)
    assert extract_usage({"usage": round_trip_chat}) == Usage(
        input_tokens=800, output_tokens=300, cached_read_tokens=150,
        cache_write_tokens=50, reasoning_tokens=25, reported=True,
    )
    round_trip_responses = anthropic_usage_to_responses(anthropic)
    assert extract_usage({"usage": round_trip_responses}) == Usage(
        input_tokens=800, output_tokens=300, cached_read_tokens=150,
        cache_write_tokens=50, reasoning_tokens=25, reported=True,
    )


def test_extract_anthropic_cache_write_ttls_and_costs_separately():
    usage = extract_usage({"usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 500,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 100,
            "ephemeral_1h_input_tokens": 400,
        },
    }})
    assert usage.cache_write_tokens == 100
    assert usage.cache_write_1h_tokens == 400
    cost = estimate_cost(usage, {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.5,
        "cache_write": 6.25,
        "cache_write_1h": 10.0,
    })
    assert cost.cost_usd == 0.006475
    assert cost.pricing_complete is True


def test_provider_reported_cost_overrides_configured_and_unmetered_pricing():
    usage = Usage(
        input_tokens=250_000,
        output_tokens=10,
        provider_cost_usd=0.0004604,
        reported=True,
    )
    for pricing_status in ("metered", "unmetered"):
        cost = estimate_cost(
            usage,
            {"input": 2.0, "output": 6.0},
            pricing_status=pricing_status,
        )
        assert cost.cost_usd == 0.0004604
        assert cost.pricing_complete is True
        assert cost.missing_classes == []


def test_invalid_provider_cost_falls_back_to_configured_pricing():
    usage = extract_usage({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cost": "not-a-number",
    }})
    cost = estimate_cost(usage, {"input": 3.0, "output": 15.0})
    assert cost.cost_usd == round((100 * 3.0 + 50 * 15.0) / 1_000_000, 6)
    assert cost.pricing_complete is True


def test_estimate_cost_full_pricing():
    # input_tokens is cache-miss (1M), cached_read 200k, cache_write 100k.
    # reasoning_tokens (50k) is a SUBSET of output_tokens for every current
    # provider, so it is NOT billed separately — it's already covered by the
    # output rate. Including it would double-count.
    usage = Usage(input_tokens=1_000_000, output_tokens=500_000,
                  cached_read_tokens=200_000, cache_write_tokens=100_000,
                  reasoning_tokens=50_000, reported=True)
    pricing = {"input": 3.0, "output": 15.0, "cache_read": 0.3,
               "cache_write": 3.75}
    cost = estimate_cost(usage, pricing)
    assert cost.pricing_complete is True
    assert cost.missing_classes == []
    # 1M*3 + 500k*15 + 200k*0.3 + 100k*3.75 (reasoning not billed separately)
    expected = 3.0 + 7.5 + 0.06 + 0.375
    assert abs(cost.cost_usd - round(expected, 6)) < 1e-6


def test_reasoning_tokens_not_flagged_as_missing():
    """reasoning_tokens must not make pricing_complete False (subset of output)."""
    usage = Usage(input_tokens=100, output_tokens=50, reasoning_tokens=30, reported=True)
    pricing = {"input": 3.0, "output": 15.0}  # no reasoning key, none needed
    cost = estimate_cost(usage, pricing)
    assert cost.pricing_complete is True
    assert cost.missing_classes == []
    # cost covers input + output only (reasoning already in output)
    assert cost.cost_usd == round((100 * 3.0 + 50 * 15.0) / 1_000_000, 6)


def test_estimate_cost_missing_pricing_class():
    usage = Usage(input_tokens=100, cache_write_tokens=50, reported=True)
    pricing = {"input": 3.0, "output": 15.0}  # no cache_write rate
    cost = estimate_cost(usage, pricing)
    assert cost.pricing_complete is False
    assert "cache_write" in cost.missing_classes
    # input still priced
    assert cost.cost_usd == round(100 * 3.0 / 1_000_000, 6)


def test_estimate_cost_no_pricing_is_unknown():
    usage = Usage(input_tokens=100, output_tokens=50, reported=True)
    cost = estimate_cost(usage, None)
    assert cost.cost_usd is None
    assert cost.pricing_complete is False


def test_long_context_without_valid_provider_cost_or_pricing_stays_unknown():
    for provider_cost in (None, -1, "0.5"):
        usage_block = {"prompt_tokens": 250_000, "completion_tokens": 50}
        if provider_cost is not None:
            usage_block["cost"] = provider_cost
        usage = extract_usage({"usage": usage_block})
        cost = estimate_cost(usage, None)
        assert usage.provider_cost_usd is None
        assert cost.cost_usd is None
        assert cost.pricing_complete is False


def test_estimate_cost_unreported_usage_is_unknown():
    usage = Usage(reported=False)
    cost = estimate_cost(usage, {"input": 3.0, "output": 15.0})
    assert cost.cost_usd is None


def test_estimate_cost_unmetered_is_known_zero_without_usage():
    cost = estimate_cost(Usage(reported=False), None, pricing_status="unmetered")
    assert cost.cost_usd == 0.0
    assert cost.pricing_complete is True
    assert cost.missing_classes == []


def test_estimate_cost_zero_tokens_complete():
    usage = Usage(reported=True)  # all zero
    cost = estimate_cost(usage, {"input": 3.0, "output": 15.0})
    assert cost.cost_usd == 0.0
    assert cost.pricing_complete is True
