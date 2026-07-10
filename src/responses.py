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

from src.signature_cache import inject_into_tool_call, store_from_extra_content

log = logging.getLogger("model-gateway")


def _gen_id(prefix: str = "resp") -> str:
    return f"{prefix}_" + secrets.token_hex(16)


def _gen_msg_id() -> str:
    return f"msg_{secrets.token_hex(16)}"


def _gen_call_id() -> str:
    return f"call_{secrets.token_hex(12)}"


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
                # Tool result back to the model
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": item.get("output", "") if isinstance(item.get("output"), str) else json.dumps(item.get("output", "")),
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

    # Build output items
    text = message.get("content")
    tool_calls = message.get("tool_calls", [])

    if tool_calls:
        # If there's text alongside tool calls, add it as a message item
        if text:
            output_items.append({
                "type": "message",
                "id": _gen_msg_id(),
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            })

        # Add function call items
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
    elif text:
        output_items.append({
            "type": "message",
            "id": _gen_msg_id(),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    else:
        # Empty response
        output_items.append({
            "type": "message",
            "id": _gen_msg_id(),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        })

    # Usage
    usage = resp.get("usage", {})
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}

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
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": prompt_details.get("cached_tokens", 0)},
            "output_tokens": usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": completion_details.get("reasoning_tokens", 0)},
            "total_tokens": usage.get("total_tokens", 0),
        },
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
    """Translate Chat Completions SSE stream into Responses API SSE events.

    Emits the standard Responses API event sequence:
    1. response.created
    2. response.in_progress
    3. response.output_item.added  (for message or function_call)
    4. response.content_part.added  (for output_text)
    5. response.output_text.delta / response.function_call_arguments.delta
    6. response.output_text.done / response.function_call_arguments.done
    7. response.content_part.done
    8. response.output_item.done
    9. response.completed
    """
    skeleton = _build_response_skeleton(model)
    msg_id = _gen_msg_id()
    now = skeleton["created_at"]

    # 1. response.created
    yield _sse("response.created", {"type": "response.created", "response": skeleton})
    # 2. response.in_progress
    yield _sse("response.in_progress", {"type": "response.in_progress", "response": skeleton})

    # State for accumulating content
    full_text = ""
    full_tool_calls: dict[int, dict] = {}  # index -> {id, call_id, name, arguments}
    usage_data = {}
    output_items = []

    # Track what output items we've announced
    current_msg_started = False
    current_tool_call_started: set[int] = set()

    buffer = ""
    async for chunk in chat_stream:
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if data.get("usage"):
                usage_data = data["usage"]

            choices = data.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # Handle text content
            if delta.get("content"):
                if not current_msg_started:
                    # Emit output_item.added and content_part.added
                    yield _sse("response.output_item.added", {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "id": msg_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    })
                    yield _sse("response.content_part.added", {
                        "type": "response.content_part.added",
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    })
                    current_msg_started = True

                text_delta = delta["content"]
                full_text += text_delta
                yield _sse("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text_delta,
                })

            # Handle tool calls
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                fn = tc.get("function", {})

                # Capture Google thought_signature from any tool_call delta
                # (may arrive on first or subsequent deltas)
                ec = tc.get("extra_content")
                if ec and idx in full_tool_calls:
                    existing_ts = (ec.get("google") or {}).get("thought_signature")
                    if existing_ts:
                        full_tool_calls[idx]["extra_content"] = ec
                        store_from_extra_content(full_tool_calls[idx]["id"], ec)

                if idx not in full_tool_calls:
                    tc_id = tc.get("id", _gen_id("fc"))
                    full_tool_calls[idx] = {
                        "id": tc_id,
                        "call_id": tc_id,
                        "name": fn.get("name", ""),
                        "arguments": "",
                    }
                    # Capture Google thought_signature for Gemini models
                    if ec:
                        full_tool_calls[idx]["extra_content"] = ec
                        store_from_extra_content(tc_id, ec)

                    # Emit output_item.added for the function call
                    tc_output_idx = len(output_items) + (1 if current_msg_started else 0)
                    yield _sse("response.output_item.added", {
                        "type": "response.output_item.added",
                        "output_index": tc_output_idx,
                        "item": {
                            "type": "function_call",
                            "id": tc_id,
                            "call_id": tc_id,
                            "name": fn.get("name", ""),
                            "arguments": "",
                            "status": "in_progress",
                        },
                    })
                    current_tool_call_started.add(idx)

                if fn.get("arguments"):
                    full_tool_calls[idx]["arguments"] += fn["arguments"]
                    tc_output_idx = len(output_items) + (1 if current_msg_started else 0)
                    yield _sse("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "item_id": full_tool_calls[idx]["id"],
                        "output_index": tc_output_idx,
                        "call_id": full_tool_calls[idx]["call_id"],
                        "delta": fn["arguments"],
                    })

            # Finish
            if finish_reason:
                break

    # Close any open content parts
    if current_msg_started:
        yield _sse("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "text": full_text,
        })
        yield _sse("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": full_text, "annotations": []},
        })

    # Close tool call argument streams
    for idx in sorted(current_tool_call_started):
        tc_output_idx = len(output_items) + (1 if current_msg_started else 0)
        tc_item = full_tool_calls[idx]
        yield _sse("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "item_id": tc_item["id"],
            "output_index": tc_output_idx,
            "call_id": tc_item["call_id"],
            "arguments": tc_item["arguments"],
        })

    # Build final output items
    if current_msg_started:
        output_items.append({
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": full_text, "annotations": []}],
        })

    for idx in sorted(full_tool_calls):
        tc_item = full_tool_calls[idx]
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
        output_items.append(fc_item)

    # Emit output_item.done for each item
    for i, item in enumerate(output_items):
        yield _sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": i,
            "item": item,
        })

    # Build usage
    prompt_details = usage_data.get("prompt_tokens_details") or {}
    completion_details = usage_data.get("completion_tokens_details") or {}
    resp_usage = {
        "input_tokens": usage_data.get("prompt_tokens", 0),
        "input_tokens_details": {"cached_tokens": prompt_details.get("cached_tokens", 0)},
        "output_tokens": usage_data.get("completion_tokens", 0),
        "output_tokens_details": {"reasoning_tokens": completion_details.get("reasoning_tokens", 0)},
        "total_tokens": usage_data.get("total_tokens", 0),
    }

    # Final completed response
    completed = _build_response_skeleton(model)
    completed["status"] = "completed"
    completed["completed_at"] = int(time.time())
    completed["output"] = output_items
    completed["usage"] = resp_usage

    yield _sse("response.completed", {
        "type": "response.completed",
        "response": completed,
    })


def _sse(event: str, data: dict) -> bytes:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
