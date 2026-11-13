"""Chat Completions over the ChatGPT plan's Responses API (``src.chatgpt_auth``).

The plan serves only ``POST /v1/responses`` with OpenAI's preview rules
(developers.openai.com/siwc/token-sharing-open-source/preview-limitations, 2026-11-12):

* ``store: false`` and ``stream: true`` on every request; the full history in ``input``.
* Instructions or developer messages only: a ``system`` input item is rejected.
* Function tools grouped in a namespace; hosted tools are unavailable.
* No ``temperature``, ``top_p``, ``max_output_tokens``, ``metadata``, ``user``,
  ``prompt_cache_retention``, ``safety_identifier``, ``truncation`` and the like.
* Success is ``response.completed``; a usage limit can arrive as ``response.failed``
  after streaming began, and is never switched to another billing path.

Reasoning items come back encrypted (``include: reasoning.encrypted_content``). A Chat
client drops them, so they are kept by tool call id and sent again before that call,
which keeps the model's reasoning across a tool loop.
"""

from __future__ import annotations

import json
import secrets
import time
from collections import OrderedDict

NAMESPACE = "app"
_NAMESPACE_DESCRIPTION = "Functions provided by the app you are talking to."

_REASONING_TTL = 86400.0
_REASONING_MAX = 4096
# first tool call id of an assistant turn -> (reasoning items before it, stored at)
_reasoning: OrderedDict[str, tuple[list[dict], float]] = OrderedDict()

_FINISH = {"max_output_tokens": "length", "content_filter": "content_filter"}


class TranslationError(ValueError):
    """The Chat request uses something this route can't carry."""


# ── Reasoning carry-over ──────────────────────────────────────────────────────


def _remember_reasoning(call_id: str, items: list[dict]) -> None:
    if not call_id or not items:
        return
    _reasoning[call_id] = (items, time.monotonic())
    _reasoning.move_to_end(call_id)
    while len(_reasoning) > _REASONING_MAX:
        _reasoning.popitem(last=False)


def _recall_reasoning(call_id: str) -> list[dict]:
    entry = _reasoning.get(call_id)
    if entry is None:
        return []
    items, stored = entry
    if time.monotonic() - stored > _REASONING_TTL:
        del _reasoning[call_id]
        return []
    return items


def _replayable(item: dict) -> dict | None:
    """A reasoning output item as input: its encrypted content, without the unstored id."""
    if not item.get("encrypted_content"):
        return None
    return {"type": "reasoning", "summary": item.get("summary") or [], "encrypted_content": item["encrypted_content"]}


# ── Request ───────────────────────────────────────────────────────────────────


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") for part in content
                         if isinstance(part, dict) and part.get("type") == "text")
    return ""


def _input_parts(content) -> list[dict]:
    """User-side Chat content as Responses input parts."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}] if content else []
    parts = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            parts.append({"type": "input_text", "text": str(part.get("text") or "")})
        elif kind == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if url:
                item = {"type": "input_image", "image_url": url}
                if isinstance(image, dict) and image.get("detail"):
                    item["detail"] = image["detail"]
                parts.append(item)
        elif kind == "file" and isinstance(part.get("file"), dict):
            file = part["file"]
            item = {"type": "input_file"}
            for key in ("file_data", "filename"):
                if file.get(key):
                    item[key] = file[key]
            if "file_data" not in item:
                raise TranslationError("This model takes files only as inline file_data")
            parts.append(item)
        else:
            raise TranslationError(f"This model can't take {kind!r} message content")
    return parts


def _tool_output(content):
    """A tool message's content as a function_call_output ``output``."""
    if isinstance(content, str):
        return content
    parts = _input_parts(content)
    if all(part["type"] == "input_text" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts


def _tools(tools) -> list[dict]:
    functions = []
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise TranslationError("This model takes only function tools")
        spec = tool["function"]
        function = {"type": "function", "name": spec.get("name"),
                    "parameters": spec.get("parameters") or {"type": "object", "properties": {}}}
        if spec.get("description"):
            function["description"] = spec["description"]
        if isinstance(spec.get("strict"), bool):
            function["strict"] = spec["strict"]
        functions.append(function)
    if not functions:
        return []
    return [{"type": "namespace", "name": NAMESPACE, "description": _NAMESPACE_DESCRIPTION, "tools": functions}]


def _tool_choice(choice) -> tuple[str | None, str | None]:
    """(the Responses tool_choice, the one function a named choice forces, if any).

    The plan's Responses API can't name a function inside a namespace (checked 2026-11-12:
    "Tool choice 'x' not found"), so a named choice sends only that function, required.
    """
    if choice in (None, "auto", "none", "required"):
        return choice, None
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or {}).get("name") if isinstance(choice.get("function"), dict) else None
        if name:
            return "required", name
    raise TranslationError("Unsupported tool_choice for this model")


def _text_format(response_format) -> dict | None:
    if not isinstance(response_format, dict):
        return None
    kind = response_format.get("type")
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema" and isinstance(response_format.get("json_schema"), dict):
        spec = response_format["json_schema"]
        out = {"type": "json_schema", "name": spec.get("name") or "response", "schema": spec.get("schema") or {}}
        if isinstance(spec.get("strict"), bool):
            out["strict"] = spec["strict"]
        return out
    return None


def chat_to_responses_request(body: dict, upstream_model: str, reasoning: dict | None) -> dict:
    """A Chat Completions body as a plan-route Responses request."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TranslationError("messages must be a non-empty list")
    instructions: list[str] = []
    items: list[dict] = []
    leading = True
    for message in messages:
        if not isinstance(message, dict):
            raise TranslationError("every message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "developer"):
            text = _text_of(content)
            if leading:
                if text:
                    instructions.append(text)
            elif text:
                items.append({"type": "message", "role": "developer",
                              "content": [{"type": "input_text", "text": text}]})
            continue
        leading = False
        if role == "user":
            parts = _input_parts(content)
            if parts:
                items.append({"type": "message", "role": "user", "content": parts})
        elif role == "assistant":
            calls = [call for call in message.get("tool_calls") or [] if isinstance(call, dict)]
            if calls:
                # In the order the model wrote them: its reasoning, then any text, then the calls.
                items.extend(_recall_reasoning(str(calls[0].get("id") or "")))
            text = _text_of(content)
            if text:
                items.append({"type": "message", "role": "assistant",
                              "content": [{"type": "output_text", "text": text}]})
            for call in calls:
                function = call.get("function") or {}
                arguments = function.get("arguments")
                items.append({
                    "type": "function_call", "call_id": str(call.get("id") or ""),
                    "name": function.get("name"), "namespace": NAMESPACE,
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
                })
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": str(message.get("tool_call_id") or ""),
                          "output": _tool_output(content)})
        else:
            raise TranslationError(f"Unsupported message role {role!r}")

    request: dict = {"model": upstream_model, "input": items, "store": False, "stream": True,
                     "include": ["reasoning.encrypted_content"]}
    if instructions:
        request["instructions"] = "\n\n".join(instructions)
    tools = _tools(body.get("tools"))
    if tools:
        choice, forced = _tool_choice(body.get("tool_choice"))
        if forced is not None:
            functions = [tool for tool in tools[0]["tools"] if tool["name"] == forced]
            if not functions:
                raise TranslationError(f"tool_choice names {forced!r}, which isn't in tools")
            tools = [{**tools[0], "tools": functions}]
        request["tools"] = tools
        if choice is not None:
            request["tool_choice"] = choice
        if isinstance(body.get("parallel_tool_calls"), bool):
            request["parallel_tool_calls"] = body["parallel_tool_calls"]
    if reasoning:
        request["reasoning"] = {**reasoning, "summary": "auto"}
    text: dict = {}
    text_format = _text_format(body.get("response_format"))
    if text_format:
        text["format"] = text_format
    if body.get("verbosity") in ("low", "medium", "high"):
        text["verbosity"] = body["verbosity"]
    if text:
        request["text"] = text
    if isinstance(body.get("prompt_cache_key"), str) and body["prompt_cache_key"]:
        request["prompt_cache_key"] = body["prompt_cache_key"]
    return request


# ── Response ──────────────────────────────────────────────────────────────────


def chat_usage(usage: object) -> dict | None:
    """Responses usage in Chat Completions shape (the ledger reads this)."""
    if not isinstance(usage, dict):
        return None
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    out = {"prompt_tokens": prompt, "completion_tokens": completion,
           "total_tokens": int(usage.get("total_tokens") or prompt + completion)}
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    if isinstance(cached, int):
        out["prompt_tokens_details"] = {"cached_tokens": cached}
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    if isinstance(reasoning, int):
        out["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return out


def _error_payload(error: object, fallback: str) -> dict:
    error = error if isinstance(error, dict) else {}
    out = {"type": str(error.get("type") or "api_error"), "message": str(error.get("message") or fallback)}
    if error.get("code"):
        out["code"] = str(error["code"])
    return out


async def _events(byte_iter):
    """Parsed SSE ``data`` payloads."""
    buffer = ""
    async for raw in byte_iter:
        buffer += raw.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield data


async def responses_sse_to_chat(byte_iter, model: str):
    """Translate a plan Responses stream to Chat Completions chunks (bytes).

    Ends with ``[DONE]`` after ``response.completed`` or ``response.incomplete``; any other
    end is an error event, so a cut-off stream never looks like a finished answer.
    """
    chat_id = "chatcmpl_" + secrets.token_hex(12)
    created = int(time.time())
    tool_index: dict[int, int] = {}  # output_index -> chat tool_calls index
    call_ids: list[str] = []
    reasoning_items: list[dict] = []
    summary_parts = 0

    def chunk(delta: dict, finish_reason=None, usage: dict | None = None) -> bytes:
        payload = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n".encode()

    def error(payload: dict) -> bytes:
        return f"data: {json.dumps({'error': payload})}\n\n".encode()

    yield chunk({"role": "assistant"})
    async for data in _events(byte_iter):
        kind = data.get("type")
        if kind == "response.output_text.delta" or kind == "response.refusal.delta":
            if data.get("delta"):
                yield chunk({"content": data["delta"]})
        elif kind == "response.reasoning_summary_part.added":
            if summary_parts:
                yield chunk({"reasoning_content": "\n\n"})
            summary_parts += 1
        elif kind == "response.reasoning_summary_text.delta":
            if data.get("delta"):
                yield chunk({"reasoning_content": data["delta"]})
        elif kind == "response.output_item.added":
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                index = len(tool_index)
                tool_index[int(data.get("output_index") or 0)] = index
                call_ids.append(str(item.get("call_id") or ""))
                yield chunk({"tool_calls": [{"index": index, "id": item.get("call_id") or "", "type": "function",
                                             "function": {"name": item.get("name") or "", "arguments": ""}}]})
        elif kind == "response.function_call_arguments.delta":
            index = tool_index.get(int(data.get("output_index") or 0))
            if index is not None and data.get("delta"):
                yield chunk({"tool_calls": [{"index": index, "function": {"arguments": data["delta"]}}]})
        elif kind == "response.output_item.done":
            item = data.get("item") or {}
            if item.get("type") == "reasoning" and (replay := _replayable(item)) is not None:
                reasoning_items.append(replay)
        elif kind in ("response.completed", "response.incomplete"):
            response = data.get("response") or {}
            if call_ids:
                _remember_reasoning(call_ids[0], reasoning_items)
            if kind == "response.incomplete":
                reason = (response.get("incomplete_details") or {}).get("reason")
                finish = _FINISH.get(reason, "length")
            else:
                finish = "tool_calls" if call_ids else "stop"
            yield chunk({}, finish_reason=finish, usage=chat_usage(response.get("usage")))
            yield b"data: [DONE]\n\n"
            return
        elif kind == "response.failed":
            yield error(_error_payload((data.get("response") or {}).get("error"), "The model's response failed"))
            return
        elif kind == "error":
            yield error(_error_payload(data.get("error") if isinstance(data.get("error"), dict) else data,
                                       "The model's response failed"))
            return
    yield error({"type": "api_error", "message": "The ChatGPT plan stream ended before the response completed"})


async def responses_sse_to_chat_completion(byte_iter, model: str) -> tuple[dict | None, dict | None]:
    """Collect a plan stream into one Chat Completions response: (response, error)."""
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict] = {}
    finish, usage = None, None
    async for raw in responses_sse_to_chat(byte_iter, model):
        text = raw.decode()
        if not text.startswith("data: ") or text.startswith("data: [DONE]"):
            continue
        data = json.loads(text[6:])
        if "error" in data:
            return None, data["error"]
        usage = data.get("usage") or usage
        choice = data["choices"][0]
        finish = choice.get("finish_reason") or finish
        delta = choice.get("delta") or {}
        if delta.get("content"):
            content.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        for call in delta.get("tool_calls") or []:
            slot = calls.setdefault(call["index"], {"id": "", "type": "function",
                                                    "function": {"name": "", "arguments": ""}})
            if call.get("id"):
                slot["id"] = call["id"]
            function = call.get("function") or {}
            if function.get("name"):
                slot["function"]["name"] = function["name"]
            slot["function"]["arguments"] += function.get("arguments") or ""
    message: dict = {"role": "assistant", "content": "".join(content) or None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    result = {"id": "chatcmpl_" + secrets.token_hex(12), "object": "chat.completion", "created": int(time.time()),
              "model": model, "choices": [{"index": 0, "message": message, "finish_reason": finish or "stop"}]}
    if usage:
        result["usage"] = usage
    return result, None
