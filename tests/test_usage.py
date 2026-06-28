"""Tests for usage normalization and cost estimation."""

from src.usage import Usage, extract_usage, estimate_cost


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


def test_extract_anthropic_shape():
    resp = {"usage": {
        "input_tokens": 800,
        "output_tokens": 300,
        "cache_read_input_tokens": 150,
        "cache_creation_input_tokens": 400,
    }}
    u = extract_usage(resp)
    assert u.reported is True
    assert u.input_tokens == 800
    assert u.output_tokens == 300
    assert u.cached_read_tokens == 150
    assert u.cache_write_tokens == 400
    assert u.reasoning_tokens == 0


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


def test_estimate_cost_full_pricing():
    # input_tokens is cache-miss (1M), cached_read 200k, cache_write 100k.
    usage = Usage(input_tokens=1_000_000, output_tokens=500_000,
                  cached_read_tokens=200_000, cache_write_tokens=100_000,
                  reasoning_tokens=50_000, reported=True)
    pricing = {"input": 3.0, "output": 15.0, "cache_read": 0.3,
               "cache_write": 3.75, "reasoning": 15.0}
    cost = estimate_cost(usage, pricing)
    assert cost.pricing_complete is True
    assert cost.missing_classes == []
    # 1M*3 + 500k*15 + 200k*0.3 + 100k*3.75 + 50k*15
    expected = 3.0 + 7.5 + 0.06 + 0.375 + 0.75
    assert abs(cost.cost_usd - round(expected, 6)) < 1e-6


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


def test_estimate_cost_unreported_usage_is_unknown():
    usage = Usage(reported=False)
    cost = estimate_cost(usage, {"input": 3.0, "output": 15.0})
    assert cost.cost_usd is None


def test_estimate_cost_zero_tokens_complete():
    usage = Usage(reported=True)  # all zero
    cost = estimate_cost(usage, {"input": 3.0, "output": 15.0})
    assert cost.cost_usd == 0.0
    assert cost.pricing_complete is True
