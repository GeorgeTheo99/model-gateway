"""Translate OpenAI SSE streaming to Anthropic SSE streaming.

Adapted from !old/anthropic-proxy/streaming.py — simplified for cloud providers
(no tool injection fallback, no <think> tag parsing).
"""

import json
import logging
import secrets
from collections.abc import AsyncIterator

from src.reasoning import reasoning_alias_text, reasoning_text
from src.signature_cache import store_from_extra_content
from src.usage import openai_chat_usage_to_anthropic, usage_has_ledger_data, usage_was_reported

log = logging.getLogger("model-gateway")


def _gen_msg_id() -> str:
    return "msg_" + secrets.token_hex(12)


def _gen_tool_id() -> str:
    return "toolu_" + secrets.token_hex(12)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _list_content_channels(parts: list) -> list[tuple[str, str]]:
    """Return list-valued OpenAI content as ordered text/reasoning runs."""
    channels: list[tuple[str, str]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        channel = ""
        text = ""
        if ptype in {"text", "output_text"}:
            channel = "text"
            value = part.get("text", "")
            text = value if isinstance(value, str) else ""
        elif ptype == "reasoning":
            channel = "reasoning"
            text = reasoning_text(part)
        elif ptype in {"reasoning_content", "reasoning_text"}:
            channel = "reasoning"
            text = reasoning_text(part)
        if channel and text:
            if channels and channels[-1][0] == channel:
                previous_channel, previous_text = channels[-1]
                channels[-1] = (previous_channel, previous_text + text)
            else:
                channels.append((channel, text))
    return channels


def _flatten_list_content(parts: list) -> tuple[str, str]:
    """Flatten list-valued OpenAI content into grouped text and reasoning."""
    channels = _list_content_channels(parts)
    return (
        "".join(text for channel, text in channels if channel == "text"),
        "".join(text for channel, text in channels if channel == "reasoning"),
    )


async def translate_stream(
    openai_stream: AsyncIterator[bytes],
    model: str,
    has_tools: bool = False,
    thinking_enabled: bool = False,
) -> AsyncIterator[str]:
    """Consume OpenAI SSE stream and yield Anthropic SSE events."""
    msg_id = _gen_msg_id()
    started = False
    next_block_index = 0
    active_channel: str | None = None
    active_block_index: int | None = None
    tool_blocks: dict[int, dict] = {}
    output_tokens = 0
    input_tokens = 0
    cached_tokens = 0
    cache_write_tokens = 0
    cache_write_1h_tokens = 0
    cache_write_reported = False
    reasoning_tokens = 0
    reasoning_reported = False
    provider_cost_usd: float | None = None
    usage_reported = False
    finish_reason = None
    saw_finish = False

    async for raw_line in _iter_sse_lines(openai_stream):
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].lstrip()
        if payload == "[DONE]":
            break

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if isinstance(chunk, dict) and (chunk.get("error") is not None or chunk.get("type") == "error"):
            raw_error = chunk.get("error")
            if isinstance(raw_error, dict):
                error = raw_error
            else:
                error = {
                    "type": "api_error",
                    "message": chunk.get("message") or str(raw_error or "upstream stream error"),
                }
            yield _sse("error", {"type": "error", "error": error})
            return

        # Extract only authoritative upstream usage. Field presence preserves a
        # valid explicit-zero report; an absent/empty block stays unknown.
        u = chunk.get("usage")
        if usage_has_ledger_data(u):
            converted = openai_chat_usage_to_anthropic(u) or {}
            if "cost" in converted:
                provider_cost_usd = converted["cost"]
            if usage_was_reported(u):
                input_tokens = converted.get("input_tokens", input_tokens)
                output_tokens = converted.get("output_tokens", output_tokens)
                cached_tokens = converted.get("cache_read_input_tokens", cached_tokens)
                if "cache_creation_input_tokens" in converted:
                    cache_creation = converted.get("cache_creation") or {}
                    cache_write_tokens = cache_creation.get(
                        "ephemeral_5m_input_tokens",
                        converted["cache_creation_input_tokens"],
                    )
                    cache_write_1h_tokens = cache_creation.get("ephemeral_1h_input_tokens", 0)
                    cache_write_reported = True
                output_details = converted.get("output_tokens_details") or {}
                if "thinking_tokens" in output_details:
                    reasoning_tokens = output_details["thinking_tokens"]
                    reasoning_reported = True
                usage_reported = True

        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        fr = choice.get("finish_reason")
        if fr:
            finish_reason = fr
            saw_finish = True

        # Emit message_start on first chunk
        if not started:
            started = True
            yield _sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            })

        # Reasoning/thinking content. Providers use several field names
        # (OpenAI-compatible local servers: reasoning_content; OpenRouter:
        # reasoning/reasoning_details).
        reasoning_delta = reasoning_alias_text(delta)
        # Preserve list-valued content-part order instead of grouping all
        # reasoning before all text. A separate provider reasoning field is
        # emitted first, matching the wire order of Chat Completions deltas.
        content_channels: list[tuple[str, str]] = []
        raw_content = delta.get("content")
        if isinstance(raw_content, list):
            list_channels = _list_content_channels(raw_content)
            list_reasoning = "".join(
                text for channel, text in list_channels if channel == "reasoning"
            )
            if reasoning_delta and reasoning_delta != list_reasoning:
                content_channels.append(("reasoning", reasoning_delta))
            content_channels.extend(list_channels)
        else:
            if reasoning_delta:
                content_channels.append(("reasoning", reasoning_delta))
            if isinstance(raw_content, str) and raw_content:
                content_channels.append(("text", raw_content))

        for channel, channel_delta in content_channels:
            if channel == "reasoning" and not thinking_enabled:
                continue
            anthropic_channel = "thinking" if channel == "reasoning" else "text"
            if active_channel != anthropic_channel:
                if active_channel is not None and active_block_index is not None:
                    yield _sse("content_block_stop", {
                        "type": "content_block_stop",
                        "index": active_block_index,
                    })
                active_channel = anthropic_channel
                active_block_index = next_block_index
                next_block_index += 1
                content_block = (
                    {"type": "thinking", "thinking": ""}
                    if channel == "reasoning"
                    else {"type": "text", "text": ""}
                )
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": active_block_index,
                    "content_block": content_block,
                })
            if channel == "reasoning":
                event_delta = {"type": "thinking_delta", "thinking": channel_delta}
            else:
                event_delta = {"type": "text_delta", "text": channel_delta}
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": active_block_index,
                "delta": event_delta,
            })

        # Tool calls in delta
        tc_list = delta.get("tool_calls")
        if tc_list:
            for tc in tc_list:
                tc_idx = tc.get("index", 0)
                fn = tc.get("function", {})

                # Capture Google thought_signature from any tool_call delta
                # (may arrive on first or subsequent deltas)
                ec = tc.get("extra_content")
                if ec:
                    existing_ts = (ec.get("google") or {}).get("thought_signature")
                    if existing_ts and tc_idx in tool_blocks:
                        # Update cache with signature from a later delta
                        store_from_extra_content(tool_blocks[tc_idx]["id"], ec)

                if tc_idx not in tool_blocks:
                    # Tool calls start a new Anthropic content channel.
                    if active_channel is not None and active_block_index is not None:
                        yield _sse("content_block_stop", {
                            "type": "content_block_stop",
                            "index": active_block_index,
                        })
                        active_channel = None
                        active_block_index = None

                    tool_id = tc.get("id") or _gen_tool_id()
                    tool_name = fn.get("name", "")
                    tool_blocks[tc_idx] = {
                        "block_index": next_block_index,
                        "id": tool_id,
                        "name": tool_name,
                    }
                    # Capture Google thought_signature for Gemini models
                    content_block: dict = {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                    }
                    ts = (ec or {}).get("google", {}).get("thought_signature")
                    if ts:
                        content_block["thought_signature"] = ts
                        store_from_extra_content(tool_id, ec)
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": next_block_index,
                        "content_block": content_block,
                    })
                    next_block_index += 1

                args_delta = fn.get("arguments", "")
                if args_delta:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": tool_blocks[tc_idx]["block_index"],
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": args_delta,
                        },
                    })

    if not saw_finish:
        yield _sse("error", {
            "type": "error",
            "error": {"type": "api_error", "message": "Upstream stream ended before a finish marker"},
        })
        return

    # Close any open blocks
    if active_channel is not None and active_block_index is not None:
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": active_block_index,
        })
    for tb in tool_blocks.values():
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": tb["block_index"],
        })

    # Map finish reason
    has_tool_use = bool(tool_blocks)
    if has_tool_use:
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    # Ensure message_start was emitted (empty response guard)
    if not started:
        yield _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        })

    # Log cache hit rate
    prompt_tokens = input_tokens + cached_tokens + cache_write_tokens + cache_write_1h_tokens
    if cached_tokens and prompt_tokens:
        log.info("Cache hit: %d/%d prompt tokens cached (%.0f%%)", cached_tokens, prompt_tokens, cached_tokens / prompt_tokens * 100)

    final_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason},
    }
    if usage_reported or provider_cost_usd is not None:
        final_delta["usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            **({"cache_read_input_tokens": cached_tokens} if cached_tokens else {}),
            **(
                {
                    "cache_creation_input_tokens": cache_write_tokens + cache_write_1h_tokens,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": cache_write_tokens,
                        "ephemeral_1h_input_tokens": cache_write_1h_tokens,
                    },
                }
                if cache_write_reported else {}
            ),
            **(
                {"output_tokens_details": {"thinking_tokens": reasoning_tokens}}
                if reasoning_reported else {}
            ),
            **({"cost": provider_cost_usd} if provider_cost_usd is not None else {}),
        }
    yield _sse("message_delta", final_delta)

    yield _sse("message_stop", {"type": "message_stop"})


async def _iter_sse_lines(stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Iterate over SSE lines from a byte stream."""
    buffer = ""
    async for chunk in stream:
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line
    if buffer:
        yield buffer
