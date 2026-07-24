"""Unified token-usage extraction and cost estimation.

The gateway speaks three response shapes (OpenAI Chat Completions, Anthropic
Messages, OpenAI Responses). Providers report token usage and prompt-cache
breakdowns differently in each. This module normalizes any of them into a
single :class:`Usage` record suitable for the request ledger, and computes an
estimated cost from a model's ``pricing`` dict ($/Mtok).

Design rules (from docs/productionization-plan.md):
- Prefer exact provider-reported usage when present.
- Use configured pricing per model.
- Show unknown cost when usage or pricing is missing; never guess silently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Usage:
    """Normalized token usage for one request.

    All fields are 0 when the provider did not report them. ``reported`` is
    False when no usage block was present at all (e.g. a streamed response
    whose final usage chunk was lost), so the ledger can mark tokens/cost as
    unavailable rather than zero.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    reasoning_tokens: int = 0
    reported: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _anthropic_cache_write_tokens(usage: dict) -> tuple[int, int]:
    """Return Anthropic cache writes split into 5-minute and 1-hour classes."""
    total = _as_int(usage.get("cache_creation_input_tokens"))
    details = usage.get("cache_creation")
    if not isinstance(details, dict) or not (
        {"ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"} & set(details)
    ):
        return total, 0
    write_5m = _as_int(details.get("ephemeral_5m_input_tokens"))
    write_1h = _as_int(details.get("ephemeral_1h_input_tokens"))
    # Older/compatible providers may report only part of the aggregate
    # breakdown. Treat any remainder as the default 5-minute class.
    write_5m += max(0, total - write_5m - write_1h)
    return write_5m, write_1h


_OPENAI_CHAT_USAGE_KEYS = {
    "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens",
    "prompt_tokens_details", "completion_tokens_details",
}
_ANTHROPIC_USAGE_KEYS = {
    "input_tokens", "output_tokens", "cache_read_input_tokens",
    "cache_creation_input_tokens",
}
_RESPONSES_USAGE_KEYS = {
    "input_tokens", "output_tokens", "total_tokens",
    "input_tokens_details", "output_tokens_details",
}


def usage_was_reported(usage: object) -> bool:
    """Return whether a usage object contains recognized provider fields.

    Field presence, not token magnitude, distinguishes a valid explicit-zero
    report from a fabricated or absent empty object.
    """
    return isinstance(usage, dict) and bool(
        set(usage) & (_OPENAI_CHAT_USAGE_KEYS | _ANTHROPIC_USAGE_KEYS | _RESPONSES_USAGE_KEYS)
    )


def _openai_chat_cached_tokens(usage: dict) -> tuple[int, bool]:
    """Return cache-read tokens from standard or Moonshot Chat usage."""
    details = usage.get("prompt_tokens_details") or {}
    if "cached_tokens" in details:
        return _as_int(details.get("cached_tokens")), True
    if "cached_tokens" in usage:
        return _as_int(usage.get("cached_tokens")), True
    return 0, False


def openai_chat_usage_to_anthropic(usage: object) -> dict | None:
    """Convert authoritative Chat usage to Anthropic's exclusive-input shape."""
    if not isinstance(usage, dict) or not (set(usage) & _OPENAI_CHAT_USAGE_KEYS):
        return None
    details = usage.get("prompt_tokens_details") or {}
    cached_read, cache_read_reported = _openai_chat_cached_tokens(usage)
    cache_write = _as_int(details.get("cache_write_tokens"))
    cache_write_1h = _as_int(details.get("cache_write_1h_tokens"))
    completion_details = usage.get("completion_tokens_details") or {}
    result = {
        "input_tokens": max(
            0,
            _as_int(usage.get("prompt_tokens")) - cached_read - cache_write - cache_write_1h,
        ),
        "output_tokens": _as_int(usage.get("completion_tokens")),
    }
    if cache_read_reported:
        result["cache_read_input_tokens"] = cached_read
    if "cache_write_tokens" in details or "cache_write_1h_tokens" in details:
        result["cache_creation_input_tokens"] = cache_write + cache_write_1h
        result["cache_creation"] = {
            "ephemeral_5m_input_tokens": cache_write,
            "ephemeral_1h_input_tokens": cache_write_1h,
        }
    if "reasoning_tokens" in completion_details:
        result["output_tokens_details"] = {
            "thinking_tokens": _as_int(completion_details.get("reasoning_tokens")),
        }
    return result


def openai_chat_usage_to_responses(usage: object) -> dict | None:
    """Convert authoritative Chat usage to the Responses inclusive-input shape."""
    if not isinstance(usage, dict) or not (set(usage) & _OPENAI_CHAT_USAGE_KEYS):
        return None
    prompt = _as_int(usage.get("prompt_tokens"))
    completion = _as_int(usage.get("completion_tokens"))
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    cached_read, _ = _openai_chat_cached_tokens(usage)
    input_details = {"cached_tokens": cached_read}
    if "cache_write_tokens" in prompt_details:
        input_details["cache_write_tokens"] = _as_int(prompt_details.get("cache_write_tokens"))
    if "cache_write_1h_tokens" in prompt_details:
        input_details["cache_write_1h_tokens"] = _as_int(prompt_details.get("cache_write_1h_tokens"))
    return {
        "input_tokens": prompt,
        "input_tokens_details": input_details,
        "output_tokens": completion,
        "output_tokens_details": {
            "reasoning_tokens": _as_int(completion_details.get("reasoning_tokens")),
        },
        "total_tokens": _as_int(usage.get("total_tokens")) or prompt + completion,
    }


def anthropic_usage_to_openai_chat(usage: object) -> dict | None:
    """Convert authoritative Anthropic usage to Chat's inclusive-input shape."""
    if not isinstance(usage, dict) or not (set(usage) & _ANTHROPIC_USAGE_KEYS):
        return None
    input_tokens = _as_int(usage.get("input_tokens"))
    output_tokens = _as_int(usage.get("output_tokens"))
    cached_read = _as_int(usage.get("cache_read_input_tokens"))
    cache_write, cache_write_1h = _anthropic_cache_write_tokens(usage)
    prompt_tokens = input_tokens + cached_read + cache_write + cache_write_1h
    details = {"cached_tokens": cached_read}
    if "cache_creation_input_tokens" in usage:
        # Additive gateway extensions preserve exact Anthropic billing when the
        # client-facing API has no standard cache-write fields.
        details["cache_write_tokens"] = cache_write
        details["cache_write_1h_tokens"] = cache_write_1h
    result = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "prompt_tokens_details": details,
    }
    output_details = usage.get("output_tokens_details") or {}
    if "thinking_tokens" in output_details:
        result["completion_tokens_details"] = {
            "reasoning_tokens": _as_int(output_details.get("thinking_tokens")),
        }
    return result


def anthropic_usage_to_responses(usage: object) -> dict | None:
    """Convert authoritative Anthropic usage to Responses usage."""
    chat_usage = anthropic_usage_to_openai_chat(usage)
    return openai_chat_usage_to_responses(chat_usage) if chat_usage is not None else None


def extract_usage(resp: dict | None) -> Usage:
    """Normalize a response body's ``usage`` block into a :class:`Usage`.

    Accepts any of:
      - OpenAI Chat Completions shape (``prompt_tokens`` / ``completion_tokens``)
      - Anthropic Messages shape (``input_tokens`` / ``output_tokens`` +
        ``cache_read_input_tokens`` / ``cache_creation_input_tokens``)
      - OpenAI Responses shape (``input_tokens`` / ``output_tokens`` +
        ``input_tokens_details`` / ``output_tokens_details``)

    A Responses-shaped block is distinguished from an Anthropic-shaped one by
    ``input_tokens_details``. Newer Anthropic models also return
    ``output_tokens_details`` (with ``thinking_tokens``), while keeping
    ``cache_read_input_tokens`` / ``cache_creation_input_tokens`` at the top
    level of ``usage``.
    """
    if not isinstance(resp, dict):
        return Usage()
    usage = resp.get("usage")
    if not usage_was_reported(usage):
        # Some OpenAI stream chunks put usage at the top level after reassembly.
        if set(resp) & _OPENAI_CHAT_USAGE_KEYS:
            usage = resp
        else:
            return Usage()

    # OpenAI Responses shape: input_tokens_details is protocol-specific. Do not
    # use output_tokens_details as the discriminator because newer Anthropic
    # models expose that key too.
    if "input_tokens_details" in usage:
        in_details = usage.get("input_tokens_details") or {}
        out_details = usage.get("output_tokens_details") or {}
        cached = _as_int(in_details.get("cached_tokens"))
        cache_write = _as_int(in_details.get("cache_write_tokens"))
        cache_write_1h = _as_int(in_details.get("cache_write_1h_tokens"))
        return Usage(
            input_tokens=max(
                0,
                _as_int(usage.get("input_tokens")) - cached - cache_write - cache_write_1h,
            ),
            output_tokens=_as_int(usage.get("output_tokens")),
            cached_read_tokens=cached,
            cache_write_tokens=cache_write,
            cache_write_1h_tokens=cache_write_1h,
            reasoning_tokens=_as_int(out_details.get("reasoning_tokens")),
            reported=True,
        )

    # Anthropic Messages shape. A block containing only input_tokens and/or
    # output_tokens is still Anthropic-shaped and must not fall through to the
    # Chat parser. Anthropic input excludes cache reads and writes.
    if set(usage) & _ANTHROPIC_USAGE_KEYS and not (set(usage) & {"prompt_tokens", "completion_tokens"}):
        out_details = usage.get("output_tokens_details") or {}
        cache_write, cache_write_1h = _anthropic_cache_write_tokens(usage)
        return Usage(
            input_tokens=_as_int(usage.get("input_tokens")),
            output_tokens=_as_int(usage.get("output_tokens")),
            cached_read_tokens=_as_int(usage.get("cache_read_input_tokens")),
            cache_write_tokens=cache_write,
            cache_write_1h_tokens=cache_write_1h,
            reasoning_tokens=_as_int(out_details.get("thinking_tokens")),
            reported=True,
        )

    # OpenAI Chat Completions shape: prompt_tokens / completion_tokens +
    # cached tokens in prompt_tokens_details (OpenAI) or at the top level
    # (Moonshot), plus completion_tokens_details.reasoning_tokens.
    # prompt_tokens INCLUDES cached tokens; subtract to get cache-miss input.
    in_details = usage.get("prompt_tokens_details") or {}
    out_details = usage.get("completion_tokens_details") or {}
    cached, _ = _openai_chat_cached_tokens(usage)
    cache_write = _as_int(in_details.get("cache_write_tokens"))
    cache_write_1h = _as_int(in_details.get("cache_write_1h_tokens"))
    return Usage(
        input_tokens=max(
            0,
            _as_int(usage.get("prompt_tokens")) - cached - cache_write - cache_write_1h,
        ),
        output_tokens=_as_int(usage.get("completion_tokens")),
        cached_read_tokens=cached,
        cache_write_tokens=cache_write,
        cache_write_1h_tokens=cache_write_1h,
        reasoning_tokens=_as_int(out_details.get("reasoning_tokens")),
        reported=True,
    )


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost for one request.

    ``cost_usd`` is None when pricing or usage is missing (unknown cost).
    ``pricing_complete`` is False when the usage contained a token class the
    pricing dict did not rate (e.g. 1-hour cache-write tokens appeared but no
    ``cache_write_1h`` price is configured); the partial cost is still returned.
    """

    cost_usd: float | None
    pricing_complete: bool
    missing_classes: list[str]

    def as_dict(self) -> dict:
        return {
            "cost_usd": self.cost_usd,
            "pricing_complete": self.pricing_complete,
            "missing_classes": self.missing_classes,
        }


# Map Usage fields to pricing keys.
#
# ``reasoning_tokens`` is intentionally NOT in this list: for every provider
# the gateway currently routes to, reasoning tokens are a SUBSET of
# output_tokens (OpenAI ``completion_tokens_details.reasoning_tokens`` and
# Anthropic ``output_tokens_details.thinking_tokens`` are both included in
# output tokens). Billing them separately would double-count, and flagging them
# as a missing pricing class made ``pricing_complete`` misleadingly false for every
# thinking request. reasoning_tokens is still recorded in the ledger for
# observability; if a provider that bills reasoning SEPARATELY from output is
# ever added, revisit this.
_CLASS_FIELDS = [
    ("input", "input_tokens"),
    ("output", "output_tokens"),
    ("cache_read", "cached_read_tokens"),
    ("cache_write", "cache_write_tokens"),
    ("cache_write_1h", "cache_write_1h_tokens"),
]


def estimate_cost(
    usage: Usage,
    pricing: dict | None,
    *,
    pricing_status: str = "unknown",
) -> CostEstimate:
    """Estimate USD cost from normalized usage and a model pricing policy.

    ``pricing`` keys are $/Mtok: ``input``, ``output``, ``cache_read``,
    ``cache_write`` (5-minute), ``cache_write_1h``, ``reasoning`` (any may be
    absent). ``pricing_status`` may
    be ``unmetered`` for local models whose marginal provider charge is known
    to be zero. Unmetered cost is complete even when token usage was not
    reported; token coverage remains represented separately by
    :attr:`Usage.reported`.

    Otherwise, returns ``cost_usd=None`` when pricing is missing entirely or
    usage was not reported. When a non-zero token class has no matching price,
    the cost is still computed from the priced classes but
    ``pricing_complete`` is False and the unpriced classes are listed in
    ``missing_classes``.
    """
    if pricing_status == "unmetered":
        return CostEstimate(cost_usd=0.0, pricing_complete=True, missing_classes=[])
    if not usage.reported:
        return CostEstimate(cost_usd=None, pricing_complete=False, missing_classes=[])
    if not pricing:
        return CostEstimate(cost_usd=None, pricing_complete=False, missing_classes=[])

    total = 0.0
    missing: list[str] = []
    for price_key, usage_field in _CLASS_FIELDS:
        tokens = getattr(usage, usage_field, 0)
        if not tokens:
            continue
        rate = pricing.get(price_key)
        if rate is None:
            missing.append(price_key)
            continue
        total += tokens * float(rate) / 1_000_000

    return CostEstimate(
        cost_usd=round(total, 6),
        pricing_complete=not missing,
        missing_classes=missing,
    )
