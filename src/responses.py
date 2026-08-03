"""Translate between OpenAI Responses API and Chat Completions API.

Codex CLI (v0.120+) uses the Responses API exclusively. This module translates
those requests into Chat Completions calls that Fireworks supports, then
translates the results back into the Responses API format.

Key differences between the two APIs:
  - Responses uses `input` (array of items) instead of `messages`
  - Responses uses `instructions` instead of `system` message
  - Responses output items use `type: "function_call"` with `call_id`
    instead of `tool_calls[]` with `id`
  - Responses streaming uses typed SSE events (response.created,
    response.output_text.delta, etc.) instead of Chat Completions delta chunks
"""

import json
import logging
import secrets
import time
import uuid

from src.reasoning import reasoning_alias_text
from src.signature_cache import inject_into_tool_call, store_from_extra_content
from src.streaming import _list_content_channels
from src.usage import openai_chat_usage_to_responses, usage_has_ledger_data

log = logging.getLogger("model-gateway")


def _gen_id(prefix: str = "resp") -> str:
    return f"{prefix}_" + secrets.token_hex(16)


def _gen_msg_id() -> str:
    return f"msg_{secrets.token_hex(16)}"


def _gen_call_id() -> str:
    return f"call_{secrets.token_hex(12)}"


def _ordered_content_channels(container: dict) -> list[tuple[str, str]]:
    """Return provider reasoning and content in their exact channel order."""
    primary_reasoning = reasoning_alias_text(container)
    raw_content = container.get("content")
    if isinstance(raw_content, list):
        channels = _list_content_channels(raw_content)
        listed_reasoning = "".join(
            text for channel, text in channels if channel == "reasoning"
        )
        # Some providers mirror list reasoning into a top-level alias. Prefer
        # the ordered list in that case rather than emitting it twice.
        if primary_reasoning and primary_reasoning != listed_reasoning:
            return [("reasoning", primary_reasoning), *channels]
        return channels

    channels = []
    if primary_reasoning:
        channels.append(("reasoning", primary_reasoning))
    if isinstance(raw_content, str) and raw_content:
        channels.append(("text", raw_content))
    return channels


def _reasoning_item(text: str, *, item_id: str | None = None, status: str = "completed") -> dict:
    return {
        "type": "reasoning",
        "id": item_id or _gen_id("rs"),
        "status": status,
        "summary": [{"type": "summary_text", "text": text}],
    }


def responses_to_chat(body: dict) -> dict:
    """Convert OpenAI Responses API request to Chat Completions format.

    Handles:
    - instructions → system message
    - input items (EasyInputMessage, FunctionCallOutput) → messages + tool results
    - tools (Function type) → Chat Completions tools
    - tool_choice mapping
    """
    messages = []

    # Instructions → system message
    instructions = body.get("instructions")
    if instructions:
        if isinstance(instructions, str) and instructions:
            messages.append({"role": "system", "content": instructions})

    # Convert input items to messages
    input_items = body.get("input", [])
    if isinstance(input_items, str):
        messages.append({"role": "user", "content": input_items})
    elif isinstance(input_items, list):
        for item in input_items:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue

            item_type = item.get("type", "")

            if item_type == "message":
                role = item.get("role", "user")
                content = item.get("content", "")

                if isinstance(content, str):
                    messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    # Array of content parts (input_text, input_image, etc.)
                    chat_parts = []
                    for part in content:
                        part_type = part.get("type", "")
                        if part_type == "input_text":
                            chat_parts.append({"type": "text", "text": part.get("text", "")})
                        elif part_type == "input_image":
                            # Preserve images as multimodal blocks; never turn them into prompt text.
                            url = part.get("image_url")
                            if url:
                                chat_parts.append({"type": "image_url", "image_url": {"url": url}})
                            elif part.get("file_id"):
                                # Translated Chat paths cannot dereference OpenAI file IDs safely.
                                # Preserve a sentinel so the endpoint can reject explicitly.
                                chat_parts.append({"type": "unsupported_input_image_file", "file_id": part["file_id"]})
                        elif part_type == "input_file":
                            chat_parts.append({"type": "text", "text": f"[file: {part.get('filename', 'unknown')}]"})
                    if chat_parts:
                        if all(part["type"] == "text" for part in chat_parts):
                            messages.append({"role": role, "content": "\n".join(part["text"] for part in chat_parts)})
                        else:
                            messages.append({"role": role, "content": chat_parts})

            elif item_type == "function_call":
                # Previous function call from the model
                raw_args = item.get("arguments", "{}")
                if not isinstance(raw_args, str) or (isinstance(raw_args, str) and raw_args.strip() and not raw_args.strip().startswith("{")):
                    log.warning("function_call %s: arguments type=%s, value=%r", item.get("name"), type(raw_args).__name__, str(raw_args)[:200])
                # Ensure arguments is a valid JSON string
                if raw_args is None:
                    raw_args = "{}"
                elif isinstance(raw_args, dict):
                    raw_args = json.dumps(raw_args)
                elif isinstance(raw_args, str):
                    if not raw_args.strip():
                        raw_args = "{}"
                    else:
                        try:
                            json.loads(raw_args)
                        except json.JSONDecodeError:
                            # Attempt to salvage: wrap bare value as {"input": ...}
                            raw_args = json.dumps({"input": raw_args})
                else:
                    raw_args = json.dumps({"input": str(raw_args)})

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [inject_into_tool_call({
                        "id": item.get("call_id", item.get("id", _gen_call_id())),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": raw_args,
                        },
                        **({"extra_content": item["extra_content"]} if item.get("extra_content") else {}),
                    })],
                })
                # Log if we injected a cached signature
                tc_out = messages[-1]["tool_calls"][0]
                if not item.get("extra_content") and tc_out.get("extra_content"):
                    log.info("signature_cache: injected cached thought_signature for function_call %s (%s)", item.get("call_id", ""), item.get("name", ""))
                elif not item.get("extra_content") and not tc_out.get("extra_content"):
                    log.warning("signature_cache: NO cached thought_signature for function_call %s (%s) — Gemini may reject", item.get("call_id", ""), item.get("name", ""))

            elif item_type == "function_call_output":
                # Tool result back to the model. Preserve image parts so the
                # gateway can stage them through a logical vision companion.
                output = item.get("output", "")
                if isinstance(output, list):
                    chat_parts = []
                    has_image = False
                    for part in output:
                        if not isinstance(part, dict):
                            chat_parts.append({"type": "text", "text": json.dumps(part)})
                            continue
                        part_type = part.get("type", "")
                        if part_type in {"input_text", "output_text", "text"}:
                            text = part.get("text", "")
                            if not isinstance(text, str):
                                text = json.dumps(text, sort_keys=True)
                            chat_parts.append({"type": "text", "text": text})
                        elif part_type in {"input_image", "image_url"}:
                            image_url = part.get("image_url")
                            if isinstance(image_url, str) and image_url:
                                chat_parts.append({"type": "image_url", "image_url": {"url": image_url}})
                                has_image = True
                            elif part.get("file_id"):
                                chat_parts.append({"type": "unsupported_input_image_file", "file_id": part["file_id"]})
                                has_image = True
                            else:
                                chat_parts.append({"type": "text", "text": json.dumps(part, sort_keys=True)})
                        elif part_type == "input_file":
                            file_label = part.get("filename") or part.get("file_id") or "unknown"
                            chat_parts.append({"type": "text", "text": f"[file: {file_label}]"})
                        else:
                            chat_parts.append({"type": "text", "text": json.dumps(part, sort_keys=True)})
                    output = (
                        chat_parts
                        if has_image
                        else "\n".join(part["text"] for part in chat_parts)
                    )
                elif not isinstance(output, str):
                    output = json.dumps(output)
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": output,
                })

    # Build Chat Completions request
    chat_req: dict = {
        "model": body.get("model", ""),
        "messages": messages,
    }

    if body.get("max_output_tokens"):
        chat_req["max_tokens"] = body["max_output_tokens"]

    # Pass-through params
    for key in ("temperature", "top_p", "stream"):
        if key in body:
            chat_req[key] = body[key]

    # Reasoning — map Responses API reasoning to Chat Completions
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort:
            chat_req["reasoning_effort"] = effort

    # Tools — extract only "function" type tools (what Fireworks supports)
    resp_tools = body.get("tools", [])
    if resp_tools:
        chat_tools = []
        for t in resp_tools:
            if t.get("type") == "function":
                chat_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    },
                })
        if chat_tools:
            chat_req["tools"] = chat_tools

    # Tool choice mapping
    tc = body.get("tool_choice")
    if tc:
        if isinstance(tc, str):
            # "auto", "none", "required"
            if tc == "required":
                chat_req["tool_choice"] = "required"
            else:
                chat_req["tool_choice"] = tc
        elif isinstance(tc, dict):
            tc_type = tc.get("type", "")
            if tc_type == "function":
                chat_req["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc.get("name", "")},
                }
            elif tc_type in ("auto", "required"):
                chat_req["tool_choice"] = tc_type
            elif tc_type == "allowed_tools":
                # Subset of tools — just use "auto" for now
                chat_req["tool_choice"] = "auto"

    # Parallel tool calls
    if "parallel_tool_calls" in body:
        chat_req["parallel_tool_calls"] = body["parallel_tool_calls"]

    return chat_req


def chat_to_responses(resp: dict, model: str, stream: bool = False) -> dict:
    """Convert Chat Completions response to Responses API format.

    Maps:
    - choices[0].message.content → output item type "message" with "output_text"
    - choices[0].message.tool_calls → output items type "function_call"
    - usage → Responses usage format
    """
    now = int(time.time())
    resp_id = _gen_id("resp")
    choice = resp.get("choices", [{}])[0]
    message = choice.get("message", {})

    output_items = []

    # Build output items. List-valued content can alternate text and reasoning;
    # every channel transition becomes a distinct Responses output item.
    for channel, text in _ordered_content_channels(message):
        if not text:
            continue
        if output_items and channel == "reasoning" and output_items[-1]["type"] == "reasoning":
            output_items[-1]["summary"][0]["text"] += text
        elif output_items and channel == "text" and output_items[-1]["type"] == "message":
            output_items[-1]["content"][0]["text"] += text
        elif channel == "reasoning":
            output_items.append(_reasoning_item(text))
        else:
            output_items.append({
                "type": "message",
                "id": _gen_msg_id(),
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            })

    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        fn = tc.get("function", {})
        fc_item: dict = {
            "type": "function_call",
            "id": tc.get("id", _gen_id("fc")),
            "call_id": tc.get("id", _gen_call_id()),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
            "status": "completed",
        }
        # Preserve Google thought_signature for Gemini models
        if tc.get("extra_content"):
            fc_item["extra_content"] = tc["extra_content"]
            # Cache signature for later injection on outbound requests
            store_from_extra_content(tc.get("id", ""), tc["extra_content"])
        output_items.append(fc_item)

    if not output_items:
        # Empty response
        output_items.append({
            "type": "message",
            "id": _gen_msg_id(),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        })

    # Usage remains absent when the upstream did not report it. Explicit zero
    # fields are preserved as a reported usage object.
    response_usage = openai_chat_usage_to_responses(resp.get("usage"))

    # Finish reason mapping
    finish_reason = choice.get("finish_reason", "stop")

    result = {
        "id": resp_id,
        "object": "response",
        "created_at": now,
        "status": "completed",
        "completed_at": now,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": output_items,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": True,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": response_usage,
        "user": None,
        "metadata": {},
    }

    return result


def _build_response_skeleton(model: str) -> dict:
    """Build the initial response object for streaming."""
    now = int(time.time())
    resp_id = _gen_id("resp")
    return {
        "id": resp_id,
        "object": "response",
        "created_at": now,
        "status": "in_progress",
        "completed_at": None,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": [],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": True,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": None,
        "user": None,
        "metadata": {},
    }


async def translate_responses_stream(chat_stream, model: str):
    """Translate Chat Completions SSE into text, reasoning, and tool events."""
    skeleton = _build_response_skeleton(model)

    yield _sse("response.created", {"type": "response.created", "response": skeleton})
    yield _sse("response.in_progress", {"type": "response.in_progress", "response": skeleton})

    full_tool_calls: dict[int, dict] = {}
    indexed_output: dict[int, dict] = {}
    usage_data = {}
    current_content: dict | None = None
    next_output_index = 0
    saw_finish = False
    stream_done = False

    def start_content_events(channel: str) -> list[bytes]:
        nonlocal current_content, next_output_index
        output_index = next_output_index
        next_output_index += 1
        if channel == "reasoning":
            item_id = _gen_id("rs")
            current_content = {
                "channel": channel,
                "id": item_id,
                "output_index": output_index,
                "text": "",
            }
            return [
                _sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "reasoning",
                        "status": "in_progress",
                        "summary": [],
                    },
                }),
                _sse("response.reasoning_summary_part.added", {
                    "type": "response.reasoning_summary_part.added",
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                }),
            ]

        item_id = _gen_msg_id()
        current_content = {
            "channel": channel,
            "id": item_id,
            "output_index": output_index,
            "text": "",
        }
        return [
            _sse("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            }),
            _sse("response.content_part.added", {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }),
        ]

    def finish_content_events() -> list[bytes]:
        nonlocal current_content
        if current_content is None:
            return []
        state = current_content
        current_content = None
        item_id = state["id"]
        output_index = state["output_index"]
        text = state["text"]
        if state["channel"] == "reasoning":
            item = _reasoning_item(text, item_id=item_id)
            events = [
                _sse("response.reasoning_summary_text.done", {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "text": text,
                }),
                _sse("response.reasoning_summary_part.done", {
                    "type": "response.reasoning_summary_part.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": text},
                }),
            ]
        else:
            item = {
                "type": "message",
                "id": item_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
            events = [
                _sse("response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                }),
                _sse("response.content_part.done", {
                    "type": "response.content_part.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": item["content"][0],
                }),
            ]
        indexed_output[output_index] = item
        events.append(_sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": item,
        }))
        return events

    buffer = ""
    async for chunk in chat_stream:
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].lstrip()
            if payload == "[DONE]":
                stream_done = True
                break
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if isinstance(data, dict) and (data.get("error") is not None or data.get("type") == "error"):
                raw_error = data.get("error")
                error = raw_error if isinstance(raw_error, dict) else {
                    "type": "api_error",
                    "message": data.get("message") or str(raw_error or "upstream stream error"),
                }
                yield _sse("error", {"type": "error", "error": error})
                return

            if usage_has_ledger_data(data.get("usage")):
                usage_data = {
                    **usage_data,
                    **data["usage"],
                }

            choices = data.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            for channel, content_delta in _ordered_content_channels(delta):
                if current_content is not None and current_content["channel"] != channel:
                    for event in finish_content_events():
                        yield event
                if current_content is None:
                    for event in start_content_events(channel):
                        yield event
                assert current_content is not None
                current_content["text"] += content_delta
                if channel == "reasoning":
                    yield _sse("response.reasoning_summary_text.delta", {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": current_content["id"],
                        "output_index": current_content["output_index"],
                        "summary_index": 0,
                        "delta": content_delta,
                    })
                else:
                    yield _sse("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": current_content["id"],
                        "output_index": current_content["output_index"],
                        "content_index": 0,
                        "delta": content_delta,
                    })

            tool_call_deltas = delta.get("tool_calls", [])
            if tool_call_deltas and current_content is not None:
                for event in finish_content_events():
                    yield event
            for tc in tool_call_deltas:
                idx = tc.get("index", 0)
                fn = tc.get("function", {})
                ec = tc.get("extra_content")
                if idx not in full_tool_calls:
                    tc_id = tc.get("id") or _gen_id("fc")
                    tc_output_index = next_output_index
                    next_output_index += 1
                    full_tool_calls[idx] = {
                        "id": tc_id,
                        "call_id": tc_id,
                        "name": fn.get("name", ""),
                        "arguments": "",
                        "output_index": tc_output_index,
                    }
                    yield _sse("response.output_item.added", {
                        "type": "response.output_item.added",
                        "output_index": tc_output_index,
                        "item": {
                            "type": "function_call",
                            "id": tc_id,
                            "call_id": tc_id,
                            "name": fn.get("name", ""),
                            "arguments": "",
                            "status": "in_progress",
                        },
                    })
                tc_item = full_tool_calls[idx]
                if fn.get("name"):
                    tc_item["name"] = fn["name"]
                if ec and (ec.get("google") or {}).get("thought_signature"):
                    tc_item["extra_content"] = ec
                    store_from_extra_content(tc_item["id"], ec)
                if fn.get("arguments"):
                    tc_item["arguments"] += fn["arguments"]
                    yield _sse("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "item_id": tc_item["id"],
                        "output_index": tc_item["output_index"],
                        "call_id": tc_item["call_id"],
                        "delta": fn["arguments"],
                    })

            if finish_reason:
                saw_finish = True

        if stream_done:
            break

    if not saw_finish:
        yield _sse("error", {
            "type": "error",
            "error": {"type": "api_error", "message": "Upstream stream ended before a finish marker"},
        })
        return

    for event in finish_content_events():
        yield event

    for idx in sorted(full_tool_calls):
        tc_item = full_tool_calls[idx]
        yield _sse("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "item_id": tc_item["id"],
            "output_index": tc_item["output_index"],
            "call_id": tc_item["call_id"],
            "arguments": tc_item["arguments"],
        })
        fc_item: dict = {
            "type": "function_call",
            "id": tc_item["id"],
            "call_id": tc_item["call_id"],
            "name": tc_item["name"],
            "arguments": tc_item["arguments"],
            "status": "completed",
        }
        if tc_item.get("extra_content"):
            fc_item["extra_content"] = tc_item["extra_content"]
        indexed_output[tc_item["output_index"]] = fc_item
        yield _sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": tc_item["output_index"],
            "item": fc_item,
        })

    output_items = [indexed_output[index] for index in sorted(indexed_output)]
    completed = dict(skeleton)
    completed["status"] = "completed"
    completed["completed_at"] = int(time.time())
    completed["output"] = output_items
    completed["usage"] = openai_chat_usage_to_responses(usage_data)
    yield _sse("response.completed", {
        "type": "response.completed",
        "response": completed,
    })


def responses_result_events(result: dict):
    """Emit a complete translated response as protocol-correct Responses SSE."""
    in_progress = {
        **result,
        "status": "in_progress",
        "completed_at": None,
        "output": [],
        "usage": None,
    }
    yield _sse("response.created", {"type": "response.created", "response": in_progress})
    yield _sse("response.in_progress", {"type": "response.in_progress", "response": in_progress})

    for output_index, item in enumerate(result.get("output", [])):
        item_type = item.get("type")
        if item_type == "reasoning":
            yield _sse("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {**item, "status": "in_progress", "summary": []},
            })
            for summary_index, part in enumerate(item.get("summary", [])):
                text = part.get("text", "")
                yield _sse("response.reasoning_summary_part.added", {
                    "type": "response.reasoning_summary_part.added",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "summary_index": summary_index,
                    "part": {"type": "summary_text", "text": ""},
                })
                yield _sse("response.reasoning_summary_text.delta", {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "summary_index": summary_index,
                    "delta": text,
                })
                yield _sse("response.reasoning_summary_text.done", {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "summary_index": summary_index,
                    "text": text,
                })
                yield _sse("response.reasoning_summary_part.done", {
                    "type": "response.reasoning_summary_part.done",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "summary_index": summary_index,
                    "part": part,
                })
        elif item_type == "message":
            yield _sse("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {**item, "status": "in_progress", "content": []},
            })
            for content_index, part in enumerate(item.get("content", [])):
                text = part.get("text", "")
                yield _sse("response.content_part.added", {
                    "type": "response.content_part.added",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                })
                yield _sse("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": text,
                })
                yield _sse("response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": content_index,
                    "text": text,
                })
                yield _sse("response.content_part.done", {
                    "type": "response.content_part.done",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": part,
                })
        elif item_type == "function_call":
            yield _sse("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {**item, "status": "in_progress", "arguments": ""},
            })
            yield _sse("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"],
                "output_index": output_index,
                "call_id": item.get("call_id", ""),
                "delta": item.get("arguments", ""),
            })
            yield _sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"],
                "output_index": output_index,
                "call_id": item.get("call_id", ""),
                "arguments": item.get("arguments", ""),
            })
        yield _sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": item,
        })

    yield _sse("response.completed", {"type": "response.completed", "response": result})


def _sse(event: str, data: dict) -> bytes:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
