"""Usage-preservation tests across translated Chat, Messages, and Responses."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.responses import chat_to_responses, responses_result_events, translate_responses_stream
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reasoning_content", "why"),
        ("reasoning", "why"),
        ("reasoning", {"text": "why"}),
        ("reasoning_details", [{"text": "wh"}, {"summary": "y"}]),
        ("reasoning_details", [{"summary": {"text": "wh"}}, {"summary": [{"text": "y"}]}]),
    ],
)
def test_chat_to_responses_preserves_reasoning_aliases(field, value):
    result = chat_to_responses({
        "choices": [{
            "message": {"content": "answer", field: value},
            "finish_reason": "stop",
        }],
    }, "test")

    assert [item["type"] for item in result["output"]] == ["reasoning", "message"]
    assert result["output"][0]["summary"] == [{"type": "summary_text", "text": "why"}]
    assert result["output"][1]["content"][0]["text"] == "answer"


def test_chat_to_responses_preserves_alternating_list_content_order():
    result = chat_to_responses({
        "choices": [{
            "message": {"content": [
                {"type": "text", "text": "A"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "R"}]},
                {"type": "output_text", "text": "B"},
            ]},
            "finish_reason": "stop",
        }],
    }, "test")

    assert [item["type"] for item in result["output"]] == [
        "message", "reasoning", "message",
    ]
    assert result["output"][0]["content"][0]["text"] == "A"
    assert result["output"][1]["summary"][0]["text"] == "R"
    assert result["output"][2]["content"][0]["text"] == "B"


def test_responses_stream_preserves_alternating_list_content_order():
    upstream = _byte_stream((
        "data: " + json.dumps({
            "choices": [{
                "delta": {"content": [
                    {"type": "text", "text": "A"},
                    {"type": "reasoning", "summary": [{"text": "R"}]},
                    {"type": "output_text", "text": "B"},
                ]},
                "finish_reason": "stop",
            }],
        }) + "\n\n"
    ).encode())

    payloads = _sse_payloads(asyncio.run(_collect(
        translate_responses_stream(upstream, "test"),
    )))
    completed = next(
        payload["response"] for payload in payloads
        if payload.get("type") == "response.completed"
    )

    assert [item["type"] for item in completed["output"]] == [
        "message", "reasoning", "message",
    ]
    assert [
        payload["output_index"] for payload in payloads
        if payload.get("type") == "response.output_item.added"
    ] == [0, 1, 2]
    assert [
        payload["output_index"] for payload in payloads
        if payload.get("type") == "response.output_item.done"
    ] == [0, 1, 2]


def test_responses_stream_preserves_interleaved_text_and_reasoning_deltas():
    chunks = []
    for delta, finish_reason in [
        ({"reasoning": {"text": "R1"}}, None),
        ({"content": "A"}, None),
        ({"reasoning_details": [{"summary": {"text": "R2"}}]}, None),
        ({"content": [
            {"type": "reasoning_content", "text": "R3"},
            {"type": "text", "text": "B"},
        ]}, None),
        ({"content": "C"}, "stop"),
    ]:
        chunks.append((
            "data: " + json.dumps({
                "choices": [{"delta": delta, "finish_reason": finish_reason}],
            }) + "\n\n"
        ).encode())

    output = asyncio.run(_collect(translate_responses_stream(_byte_stream(*chunks), "test")))
    payloads = _sse_payloads(output)
    deltas = [
        ("reasoning" if payload["type"] == "response.reasoning_summary_text.delta" else "text", payload["delta"])
        for payload in payloads
        if payload.get("type") in {
            "response.reasoning_summary_text.delta", "response.output_text.delta",
        }
    ]
    assert deltas == [
        ("reasoning", "R1"), ("text", "A"), ("reasoning", "R2"),
        ("reasoning", "R3"), ("text", "B"), ("text", "C"),
    ]
    completed = next(
        payload["response"] for payload in payloads
        if payload.get("type") == "response.completed"
    )
    assert [item["type"] for item in completed["output"]] == [
        "reasoning", "message", "reasoning", "message",
    ]
    assert completed["output"][0]["summary"][0]["text"] == "R1"
    assert completed["output"][1]["content"][0]["text"] == "A"
    assert completed["output"][2]["summary"][0]["text"] == "R2R3"
    assert completed["output"][3]["content"][0]["text"] == "BC"
    added = [
        (payload["output_index"], payload["item"]["type"])
        for payload in payloads
        if payload.get("type") == "response.output_item.added"
    ]
    done = [
        (payload["output_index"], payload["item"]["type"])
        for payload in payloads
        if payload.get("type") == "response.output_item.done"
    ]
    assert added == done == [
        (0, "reasoning"), (1, "message"),
        (2, "reasoning"), (3, "message"),
    ]


def test_anthropic_stream_reopens_blocks_on_every_reasoning_text_transition():
    chunks = []
    for delta, finish_reason in [
        ({"reasoning": {"text": "R1"}}, None),
        ({"content": "A"}, None),
        ({"reasoning_details": [{"summary": {"text": "R2"}}]}, None),
        ({"content": "B"}, "stop"),
    ]:
        chunks.append((
            "data: " + json.dumps({
                "choices": [{"delta": delta, "finish_reason": finish_reason}],
            }) + "\n\n"
        ).encode())

    output = asyncio.run(_collect(translate_stream(
        _byte_stream(*chunks), "test", thinking_enabled=True,
    )))
    payloads = _sse_payloads(output)
    starts = [
        (payload["index"], payload["content_block"]["type"])
        for payload in payloads
        if payload.get("type") == "content_block_start"
    ]
    deltas = [
        (payload["index"], payload["delta"].get("thinking") or payload["delta"].get("text"))
        for payload in payloads
        if payload.get("type") == "content_block_delta"
    ]
    stops = [
        payload["index"] for payload in payloads
        if payload.get("type") == "content_block_stop"
    ]
    assert starts == [(0, "thinking"), (1, "text"), (2, "thinking"), (3, "text")]
    assert deltas == [(0, "R1"), (1, "A"), (2, "R2"), (3, "B")]
    assert stops == [0, 1, 2, 3]


def test_anthropic_responses_stream_emits_interleaved_reasoning_and_text_items():
    result = _anthropic_messages_to_responses({
        "content": [
            {"type": "thinking", "thinking": "R1"},
            {"type": "text", "text": "A"},
            {"type": "thinking", "thinking": "R2"},
            {"type": "text", "text": "B"},
        ],
        "stop_reason": "end_turn",
    }, "test")

    payloads = _sse_payloads(list(responses_result_events(result)))
    deltas = [
        ("reasoning" if payload["type"] == "response.reasoning_summary_text.delta" else "text", payload["delta"])
        for payload in payloads
        if payload.get("type") in {
            "response.reasoning_summary_text.delta", "response.output_text.delta",
        }
    ]
    assert deltas == [
        ("reasoning", "R1"), ("text", "A"),
        ("reasoning", "R2"), ("text", "B"),
    ]
    assert [item["type"] for item in result["output"]] == [
        "reasoning", "message", "reasoning", "message",
    ]


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
        b'data:{"choices":[{"delta":{"reasoning":{"text":"R1"}},"finish_reason":null}]}\n\n',
        b'data:{"choices":[{"delta":{"reasoning_details":[{"summary":{"text":"R2"}}],"content":"ok"},"finish_reason":"stop"}]}\n\n',
    ])
    result = asyncio.run(_collect_stream(response))
    message = result["choices"][0]["message"]
    assert message["content"] == "ok"
    assert message["reasoning_content"] == "R1R2"

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


@pytest.mark.parametrize(
    "reasoning",
    [
        {"reasoning": {"text": "why"}},
        {"reasoning_details": [{"summary": {"text": "wh"}}, {"summary": "y"}]},
    ],
)
def test_messages_sync_translation_flattens_structured_reasoning(reasoning):
    result = openai_to_anthropic({
        "choices": [{
            "message": {"content": "answer", **reasoning},
            "finish_reason": "stop",
        }],
    }, "test", thinking_enabled=True)

    assert result["content"] == [
        {"type": "thinking", "thinking": "why"},
        {"type": "text", "text": "answer"},
    ]


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
