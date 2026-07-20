"""Usage-preservation tests across translated Chat, Messages, and Responses."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.responses import chat_to_responses, translate_responses_stream
from src.server import (
    _anthropic_messages_to_responses,
    _anthropic_sse_to_openai_chat,
    _collect_anthropic_stream,
    _collect_stream,
    _sse_error_from_line,
    _usage_fragment_from_sse_line,
)
from src.streaming import translate_stream
from src.translator import anthropic_to_openai_chat, openai_to_anthropic
from src.usage import Usage, extract_usage


async def _byte_stream(*chunks: bytes):
    for chunk in chunks:
        yield chunk


async def _collect(iterator):
    return [chunk async for chunk in iterator]


def _sse_payloads(chunks) -> list[dict]:
    text = "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in chunks)
    payloads = []
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payloads.append(json.loads(line[6:]))
    return payloads


def test_messages_stream_does_not_fabricate_final_usage():
    upstream = _byte_stream(
        b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    )
    output = asyncio.run(_collect(translate_stream(upstream, "test")))
    final = next(p for p in _sse_payloads(output) if p.get("type") == "message_delta")
    assert "usage" not in final


def test_messages_stream_preserves_explicit_zero_and_cached_input():
    upstream = _byte_stream(
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":0,',
        b'"prompt_tokens_details":{"cached_tokens":200}}}\n\n',
        b"data: [DONE]\n\n",
    )
    output = asyncio.run(_collect(translate_stream(upstream, "test")))
    final = next(p for p in _sse_payloads(output) if p.get("type") == "message_delta")
    assert final["usage"] == {
        "input_tokens": 800,
        "output_tokens": 0,
        "cache_read_input_tokens": 200,
    }


def test_messages_stream_preserves_cache_write_usage():
    upstream = _byte_stream(
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":3,',
        b'"prompt_tokens_details":{"cached_tokens":200,"cache_write_tokens":50}}}\n\n',
    )
    output = asyncio.run(_collect(translate_stream(upstream, "test")))
    final = next(p for p in _sse_payloads(output) if p.get("type") == "message_delta")
    assert final["usage"] == {
        "input_tokens": 750,
        "output_tokens": 3,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 50,
    }


def test_responses_stream_absent_vs_explicit_zero_usage():
    missing = _byte_stream(
        b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
    )
    missing_output = asyncio.run(_collect(translate_responses_stream(missing, "test")))
    missing_completed = next(
        p["response"] for p in _sse_payloads(missing_output)
        if p.get("type") == "response.completed"
    )
    assert missing_completed["usage"] is None

    explicit_zero = _byte_stream(
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0}}\n\n'
    )
    zero_output = asyncio.run(_collect(translate_responses_stream(explicit_zero, "test")))
    zero_completed = next(
        p["response"] for p in _sse_payloads(zero_output)
        if p.get("type") == "response.completed"
    )
    assert zero_completed["usage"]["input_tokens"] == 0
    assert extract_usage(zero_completed).reported is True


def test_translated_openai_stream_errors_do_not_become_success():
    upstream_error = _byte_stream(
        b'data: {"error":{"type":"api_error","message":"provider failed"}}\n\n',
    )
    message_output = asyncio.run(_collect(translate_stream(upstream_error, "test")))
    message_payloads = _sse_payloads(message_output)
    assert message_payloads[-1]["type"] == "error"
    assert all(payload.get("type") != "message_stop" for payload in message_payloads)

    premature = _byte_stream(
        b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n',
    )
    response_output = asyncio.run(_collect(translate_responses_stream(premature, "test")))
    response_payloads = _sse_payloads(response_output)
    assert response_payloads[-1]["type"] == "error"
    assert all(payload.get("type") != "response.completed" for payload in response_payloads)


def test_anthropic_stream_merges_initial_and_final_usage():
    upstream = _byte_stream(
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":11,',
        b'"output_tokens":0,"cache_read_input_tokens":3,"cache_creation_input_tokens":2}}}\n\n',
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}\n\n',
        b'data: {"type":"message_stop"}\n\n',
    )
    output = asyncio.run(_collect(_anthropic_sse_to_openai_chat(upstream, "test")))
    final = next(p for p in _sse_payloads(output) if p.get("usage"))
    assert extract_usage(final) == Usage(
        input_tokens=11,
        output_tokens=7,
        cached_read_tokens=3,
        cache_write_tokens=2,
        reported=True,
    )


def test_anthropic_stream_without_final_usage_remains_unreported():
    upstream = _byte_stream(
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":0,"output_tokens":0}}}\n\n',
        b'data: {"type":"message_stop"}\n\n',
    )
    output = asyncio.run(_collect(_anthropic_sse_to_openai_chat(upstream, "test")))
    assert all("usage" not in payload for payload in _sse_payloads(output))


def test_anthropic_stream_errors_are_propagated_with_optional_sse_space():
    upstream = _byte_stream(
        b'data:{"type":"error","error":{"type":"api_error","message":"anthropic failed"}}\n\n',
    )
    output = asyncio.run(_collect(_anthropic_sse_to_openai_chat(upstream, "test")))
    assert _sse_payloads(output)[-1]["error"]["message"] == "anthropic failed"

    response = _FakeResponse([
        b'data:{"type":"error","error":{"message":"collector failed"}}\n\n',
    ])
    with pytest.raises(ValueError, match="collector failed"):
        asyncio.run(_collect_anthropic_stream(response))


def test_anthropic_chat_stream_reports_truncated_eof_as_error():
    upstream = _byte_stream(
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":1,"output_tokens":0}}}\n\n',
    )
    output = asyncio.run(_collect(_anthropic_sse_to_openai_chat(upstream, "test")))
    payloads = _sse_payloads(output)
    assert payloads[-1]["type"] == "error"
    assert all(
        not (payload.get("choices") and payload["choices"][0].get("finish_reason"))
        for payload in payloads
    )


def test_sse_error_parser_accepts_nested_and_native_responses_shapes():
    assert _sse_error_from_line(
        'data: {"type":"error","error":{"message":"nested failure"}}'
    ) == "nested failure"
    assert _sse_error_from_line(
        'data: {"type":"error","message":"native responses failure","code":"server_error"}'
    ) == "native responses failure"


def test_sse_usage_fragments_distinguish_anthropic_start_and_final():
    initial = _usage_fragment_from_sse_line(
        'data: {"type":"message_start","message":{"usage":{"input_tokens":0,"output_tokens":0}}}'
    )
    final = _usage_fragment_from_sse_line(
        'data: {"type":"message_delta","usage":{"output_tokens":0}}'
    )
    assert initial == ("initial", {"input_tokens": 0, "output_tokens": 0})
    assert final == ("final", {"output_tokens": 0})


class _FakeResponse:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


def test_collect_anthropic_stream_keeps_initial_input_and_final_output():
    response = _FakeResponse([
        b'data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":11,"cache_read_input_tokens":3}}}\n\n',
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}\n\n',
        b'data: {"type":"message_stop"}\n\n',
    ])
    result = asyncio.run(_collect_anthropic_stream(response))
    assert result["usage"] == {
        "input_tokens": 11,
        "cache_read_input_tokens": 3,
        "output_tokens": 7,
    }


def test_collect_openai_stream_requires_finish_and_accepts_optional_sse_space():
    response = _FakeResponse([
        b'data:{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
    ])
    result = asyncio.run(_collect_stream(response))
    assert result["choices"][0]["message"]["content"] == "ok"

    truncated = _FakeResponse([
        b'data:{"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n',
    ])
    with pytest.raises(ValueError, match="finish marker"):
        asyncio.run(_collect_stream(truncated))


def test_collect_anthropic_stream_rejects_truncated_usage():
    response = _FakeResponse([
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":11}}}\n\n',
    ])
    with pytest.raises(ValueError, match="ended before final usage"):
        asyncio.run(_collect_anthropic_stream(response))


def test_nonstream_translators_preserve_absence_and_cache_semantics():
    chat = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    assert "usage" not in openai_to_anthropic(chat, "test")
    assert chat_to_responses(chat, "test")["usage"] is None

    anthropic = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
    assert "usage" not in anthropic_to_openai_chat(anthropic, "test")
    assert _anthropic_messages_to_responses(anthropic, "test")["usage"] is None

    anthropic["usage"] = {
        "input_tokens": 8,
        "output_tokens": 3,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 1,
    }
    translated_chat = anthropic_to_openai_chat(anthropic, "test")
    translated_responses = _anthropic_messages_to_responses(anthropic, "test")
    expected = Usage(
        input_tokens=8,
        output_tokens=3,
        cached_read_tokens=2,
        cache_write_tokens=1,
        reported=True,
    )
    assert extract_usage(translated_chat) == expected
    assert extract_usage(translated_responses) == expected
