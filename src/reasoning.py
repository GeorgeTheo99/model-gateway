"""Shape-safe helpers for OpenAI-compatible reasoning payloads."""

from __future__ import annotations

from typing import Any

_REASONING_ALIASES = ("reasoning_content", "reasoning", "reasoning_details")


def reasoning_text(value: Any) -> str:
    """Flatten common structured reasoning values without leaking containers."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(reasoning_text(item) for item in value)
    if not isinstance(value, dict):
        return ""
    for key in ("text", "thinking", "summary"):
        text = reasoning_text(value.get(key))
        if text:
            return text
    return ""


def reasoning_alias_text(container: Any) -> str:
    """Read provider reasoning aliases in precedence order without duplication."""
    if not isinstance(container, dict):
        return ""
    for key in _REASONING_ALIASES:
        text = reasoning_text(container.get(key))
        if text:
            return text
    return ""
