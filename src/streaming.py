"""Translate OpenAI SSE streaming to Anthropic SSE streaming.

Adapted from !old/anthropic-proxy/streaming.py — simplified for cloud providers
(no tool injection fallback, no <think> tag parsing).
"""

import json
import logging
import secrets
from collections.abc import AsyncIterator

from src.signature_cache import store_from_extra_content
from src.usage import openai_chat_usage_to_anthropic, usage_was_reported

log = logging.getLogger("model-gateway")


def _gen_msg_id() -> str:
    return "msg_" + secrets.token_hex(12)


def _gen_tool_id() -> str:
    return "toolu_" + secrets.token_hex(12)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _flatten_list_content(parts: list) -> tuple[str, str]:
    """Flatten list-valued OpenAI delta content into (text, reasoning).

    Some upstreams (e.g. native serving invocations for reasoning models) emit
    delta.content as a list of OpenAI content-part blocks instead of a string.
    Shape-safe: only called when content is a list, so applied unconditionally.
    """
    text_segments: list[str] = []
    reasoning_segments: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            text_segments.append(part.get("text", ""))
        elif ptype == "reasoning":
            for s in part.get("summary", []) or []:
                if isinstance(s, dict):
                    reasoning_segments.append(s.get("text", ""))
        elif ptype == "reasoning_content":
            reasoning_segments.append(part.get("text", ""))
    return "".join(text_segments), "".join(reasoning_segments)


async def translate_stream(
    openai_stream: AsyncIterator[bytes],
    model: str,
    has_tools: bool = False,
    thinking_enabled: bool = False,
) -> AsyncIterator[str]:
    """Consume OpenAI SSE stream and yield Anthropic SSE events."""
    msg_id = _gen_msg_id()
    started = False
    block_index = 0
    text_block_open = False
    thinking_block_open = False
    tool_blocks: dict[int, dict] = {}
    output_tokens = 0
    input_tokens = 0
    cached_tokens = 0
    cache_write_tokens = 0
    cache_write_reported = False
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
        if usage_was_reported(u):
            converted = openai_chat_usage_to_anthropic(u) or {}
            input_tokens = converted.get("input_tokens", input_tokens)
            output_tokens = converted.get("output_tokens", output_tokens)
            cached_tokens = converted.get("cache_read_input_tokens", cached_tokens)
            if "cache_creation_input_tokens" in converted:
                cache_write_tokens = converted["cache_creation_input_tokens"]
                cache_write_reported = True
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
        reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
        if not reasoning_delta and delta.get("reasoning_details"):
            parts = []
            for item in delta.get("reasoning_details") or []:
                if isinstance(item, dict):
                    parts.append(item.get("text") or item.get("summary") or "")
            reasoning_delta = "".join(parts)
        # Some upstreams emit delta.content as a list of content-part blocks
        # (mixing text and reasoning). Flatten before string handling below.
        text_delta = delta.get("content")
        if isinstance(text_delta, list):
            text_delta, list_reasoning = _flatten_list_content(text_delta)
            if list_reasoning:
                reasoning_delta = (reasoning_delta or "") + list_reasoning
        if reasoning_delta and thinking_enabled:
            if not thinking_block_open:
                thinking_block_open = True
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                })
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "thinking_delta", "thinking": reasoning_delta},
            })

        # Text content
        if text_delta:
            # Close thinking block if it was open
            if thinking_block_open:
                yield _sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": block_index,
                })
                block_index += 1
                thinking_block_open = False

            if not text_block_open:
                text_block_open = True
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                })
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": text_delta},
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
                    # Close text block if open
                    if text_block_open:
                        yield _sse("content_block_stop", {
                            "type": "content_block_stop",
                            "index": block_index,
                        })
                        block_index += 1
                        text_block_open = False

                    tool_id = tc.get("id") or _gen_tool_id()
                    tool_name = fn.get("name", "")
                    tool_blocks[tc_idx] = {
                        "block_index": block_index,
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
                        "index": block_index,
                        "content_block": content_block,
                    })
                    block_index += 1

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
    if thinking_block_open:
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": block_index,
        })
    if text_block_open:
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": block_index,
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
    prompt_tokens = input_tokens + cached_tokens + cache_write_tokens
    if cached_tokens and prompt_tokens:
        log.info("Cache hit: %d/%d prompt tokens cached (%.0f%%)", cached_tokens, prompt_tokens, cached_tokens / prompt_tokens * 100)

    final_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason},
    }
    if usage_reported:
        final_delta["usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            **({"cache_read_input_tokens": cached_tokens} if cached_tokens else {}),
            **({"cache_creation_input_tokens": cache_write_tokens} if cache_write_reported else {}),
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
