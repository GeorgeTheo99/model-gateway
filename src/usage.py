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
    reasoning_tokens: int = 0
    reported: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_usage(resp: dict | None) -> Usage:
    """Normalize a response body's ``usage`` block into a :class:`Usage`.

    Accepts any of:
      - OpenAI Chat Completions shape (``prompt_tokens`` / ``completion_tokens``)
      - Anthropic Messages shape (``input_tokens`` / ``output_tokens`` +
        ``cache_read_input_tokens`` / ``cache_creation_input_tokens``)
      - OpenAI Responses shape (``input_tokens`` / ``output_tokens`` +
        ``input_tokens_details`` / ``output_tokens_details``)

    A Responses-shaped block is distinguished from an Anthropic-shaped one by
    the presence of ``input_tokens_details`` / ``output_tokens_details``
    (Anthropic uses ``cache_read_input_tokens`` / ``cache_creation_input_tokens``
    at the top level of ``usage``).
    """
    if not isinstance(resp, dict):
        return Usage()
    usage = resp.get("usage")
    if not isinstance(usage, dict) or not usage:
        # Some OpenAI stream chunks put usage at the top level after reassembly.
        if isinstance(resp, dict) and any(
            k in resp for k in ("prompt_tokens", "completion_tokens")
        ):
            usage = resp
        else:
            return Usage()

    # Anthropic Messages shape: top-level cache_read/cache_creation keys.
    # Anthropic's input_tokens EXCLUDES cached tokens (cache reads and writes
    # are reported separately), so no subtraction needed.
    if "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        return Usage(
            input_tokens=_as_int(usage.get("input_tokens")),
            output_tokens=_as_int(usage.get("output_tokens")),
            cached_read_tokens=_as_int(usage.get("cache_read_input_tokens")),
            cache_write_tokens=_as_int(usage.get("cache_creation_input_tokens")),
            reasoning_tokens=0,
            reported=True,
        )

    # OpenAI Responses shape: nested *_tokens_details.
    # OpenAI's input_tokens INCLUDES cached tokens, so subtract cached to get
    # cache-miss input (consistent with Anthropic semantics). This prevents
    # estimate_cost from billing cached tokens at both the input and cache_read
    # rates.
    if "input_tokens_details" in usage or "output_tokens_details" in usage:
        in_details = usage.get("input_tokens_details") or {}
        out_details = usage.get("output_tokens_details") or {}
        cached = _as_int(in_details.get("cached_tokens"))
        return Usage(
            input_tokens=max(0, _as_int(usage.get("input_tokens")) - cached),
            output_tokens=_as_int(usage.get("output_tokens")),
            cached_read_tokens=cached,
            cache_write_tokens=0,
            reasoning_tokens=_as_int(out_details.get("reasoning_tokens")),
            reported=True,
        )

    # OpenAI Chat Completions shape: prompt_tokens / completion_tokens +
    # prompt_tokens_details.cached_tokens + completion_tokens_details.reasoning_tokens.
    # prompt_tokens INCLUDES cached tokens; subtract to get cache-miss input.
    in_details = usage.get("prompt_tokens_details") or {}
    out_details = usage.get("completion_tokens_details") or {}
    cached = _as_int(in_details.get("cached_tokens"))
    return Usage(
        input_tokens=max(0, _as_int(usage.get("prompt_tokens")) - cached),
        output_tokens=_as_int(usage.get("completion_tokens")),
        cached_read_tokens=cached,
        cache_write_tokens=0,
        reasoning_tokens=_as_int(out_details.get("reasoning_tokens")),
        reported=True,
    )


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost for one request.

    ``cost_usd`` is None when pricing or usage is missing (unknown cost).
    ``pricing_complete`` is False when the usage contained a token class the
    pricing dict did not rate (e.g. cache_write tokens appeared but no
    ``cache_write`` price is configured); the partial cost is still returned.
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
_CLASS_FIELDS = [
    ("input", "input_tokens"),
    ("output", "output_tokens"),
    ("cache_read", "cached_read_tokens"),
    ("cache_write", "cache_write_tokens"),
    ("reasoning", "reasoning_tokens"),
]


def estimate_cost(usage: Usage, pricing: dict | None) -> CostEstimate:
    """Estimate USD cost from a Usage record and a model pricing dict.

    ``pricing`` keys are $/Mtok: ``input``, ``output``, ``cache_read``,
    ``cache_write``, ``reasoning`` (any may be absent). Returns
    ``cost_usd=None`` when pricing is missing entirely or usage was not
    reported. When a non-zero token class has no matching price, the cost is
    still computed from the priced classes but ``pricing_complete`` is False
    and the unpriced classes are listed in ``missing_classes``.
    """
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
