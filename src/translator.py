"""Translate between Anthropic Messages API and OpenAI Chat Completions API.

Adapted from !old/anthropic-proxy/translator.py — simplified for cloud providers
(no tool injection, no system reminder normalization, no ࿜ tag parsing).
"""

import json
import logging
import secrets
import time

from src.signature_cache import inject_into_tool_call

log = logging.getLogger("model-gateway")


def _gen_msg_id() -> str:
    return "msg_" + secrets.token_hex(12)


def anthropic_to_openai(body: dict) -> dict:
    """Convert Anthropic Messages API request to OpenAI Chat Completions format.

    Strips cache_control blocks (Fireworks uses automatic prefix caching,
    not Anthropic's explicit cache_control markers).
    """
    messages = []

    # System message — strip cache_control (Fireworks doesn't support it)
    system = body.get("system")
    if system:
        if isinstance(system, list):
            # Grab text from text blocks, silently ignore cache_control
            system_text = " ".join(
                b["text"] for b in system if b.get("type") == "text"
            )
        else:
            system_text = system
        if system_text:
            messages.append({"role": "system", "content": system_text})

    # Convert messages — strip cache_control from content blocks
    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content) if content else ""})
            continue

        # Content is an array of blocks
        if role == "user":
            # Preserve block order, flushing contiguous text/image content around
            # tool_result blocks so translated conversation sequencing is stable.
            pending_parts = []
            pending_has_image = False

            def flush_pending():
                nonlocal pending_parts, pending_has_image
                if not pending_parts:
                    return
                if pending_has_image:
                    messages.append({"role": "user", "content": pending_parts})
                else:
                    messages.append({"role": "user", "content": "\n".join(p["text"] for p in pending_parts)})
                pending_parts = []
                pending_has_image = False

            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    pending_parts.append({"type": "text", "text": block["text"]})
                elif block_type == "image":
                    source = block.get("source", {})
                    source_type = source.get("type", "base64")
                    if source_type == "url" and source.get("url"):
                        image_url = source["url"]
                    else:
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        image_url = f"data:{media_type};base64,{data}"
                    pending_parts.append({
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    })
                    pending_has_image = True
                elif block_type == "tool_result":
                    flush_pending()
                    tr_content = block.get("content", "")
                    if isinstance(tr_content, list):
                        tool_parts = []
                        has_image = False
                        for item in tr_content:
                            if not isinstance(item, dict):
                                continue
                            if item.get("type") == "text":
                                tool_parts.append({"type": "text", "text": item.get("text", "")})
                            elif item.get("type") == "image":
                                source = item.get("source", {})
                                if source.get("type") == "url" and source.get("url"):
                                    image_url = source["url"]
                                else:
                                    media_type = source.get("media_type", "image/png")
                                    image_url = f"data:{media_type};base64,{source.get('data', '')}"
                                tool_parts.append({"type": "image_url", "image_url": {"url": image_url}})
                                has_image = True
                        tr_content = (
                            tool_parts
                            if has_image
                            else "\n".join(part["text"] for part in tool_parts)
                        )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": tr_content if isinstance(tr_content, list) else str(tr_content),
                    })
            flush_pending()

        elif role == "assistant":
            text_parts = []
            tool_calls = []
            reasoning_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "thinking":
                    reasoning_parts.append(block.get("thinking", ""))
                elif block.get("type") == "tool_use":
                    tc: dict = {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    }
                    # Preserve Google thought_signature for Gemini models
                    if block.get("thought_signature"):
                        tc["extra_content"] = {"google": {"thought_signature": block["thought_signature"]}}
                    # Inject cached signature if client didn't preserve it
                    had_ec = bool(tc.get("extra_content"))
                    inject_into_tool_call(tc)
                    if not had_ec and tc.get("extra_content"):
                        log.info("signature_cache: injected cached thought_signature for tool_use %s (%s)", block["id"], block["name"])
                    elif not had_ec and not tc.get("extra_content") and not tool_calls:
                        # First tool_call in this step lacks a signature — Gemini will reject this
                        log.warning("signature_cache: NO cached thought_signature for first tool_use %s (%s) — Gemini may reject", block["id"], block["name"])
                    tool_calls.append(tc)

            msg_out: dict = {"role": "assistant"}
            msg_out["content"] = "\n".join(text_parts) if text_parts else None
            if reasoning_parts:
                msg_out["reasoning_content"] = "\n".join(reasoning_parts)
            if tool_calls:
                msg_out["tool_calls"] = tool_calls
            messages.append(msg_out)

    # Build OpenAI request
    openai_req: dict = {
        "model": body.get("model", ""),
        "messages": messages,
        "max_tokens": body.get("max_tokens", 32768),
    }

    # Pass-through params
    for key in ("temperature", "top_p", "stream"):
        if key in body:
            openai_req[key] = body[key]

    # Reasoning — forward thinking/reasoning_effort to Fireworks
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        if thinking_type == "enabled":
            budget = thinking.get("budget_tokens")
            if budget:
                openai_req["thinking"] = {"type": "enabled", "budget_tokens": budget}
            else:
                openai_req["reasoning_effort"] = "high"
        elif thinking_type == "adaptive":
            # Map adaptive thinking to reasoning_effort for OpenAI-compatible providers
            effort = (body.get("output_config") or {}).get("effort", "high")
            openai_req["reasoning_effort"] = effort
        elif thinking_type == "disabled":
            openai_req["thinking"] = {"type": "disabled"}

    if "stop_sequences" in body:
        openai_req["stop"] = body["stop_sequences"]

    # Tools — strip cache_control from tool definitions
    if "tools" in body:
        openai_req["tools"] = []
        for t in body["tools"]:
            tool_def = {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            openai_req["tools"].append(tool_def)

        # Tool choice
        tc = body.get("tool_choice")
        if tc:
            if isinstance(tc, dict):
                tc_type = tc.get("type")
                if tc_type == "auto":
                    openai_req["tool_choice"] = "auto"
                elif tc_type == "any":
                    openai_req["tool_choice"] = "required"
                elif tc_type == "tool":
                    openai_req["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tc["name"]},
                    }
            elif isinstance(tc, str):
                if tc == "any":
                    openai_req["tool_choice"] = "required"
                else:
                    openai_req["tool_choice"] = tc

    return openai_req


def _anthropic_content_from_openai(content) -> str | list:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""

    blocks = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in {"text", "input_text"}:
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif ptype in {"image_url", "input_image"} or "image_url" in part:
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str) and isinstance(part.get("image"), str):
                url = part["image"]
            if isinstance(url, str) and url.startswith("data:image/") and ";base64," in url:
                header, data = url.split(";base64,", 1)
                media_type = header.removeprefix("data:") or "image/png"
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                })
    return blocks or ""


def openai_chat_to_anthropic(body: dict) -> dict:
    """Convert an OpenAI Chat Completions request to Anthropic Messages."""
    system_parts = []
    messages = []
    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.extend(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") in {"text", "input_text"})
            continue
        if role == "tool":
            tool_content = content if isinstance(content, str) else json.dumps(content)
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": tool_content,
                }],
            })
            continue
        if role == "assistant":
            blocks = []
            if content:
                blocks.append({"type": "text", "text": content if isinstance(content, str) else json.dumps(content)})
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if reasoning:
                blocks.insert(0, {"type": "thinking", "thinking": str(reasoning)})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", "toolu_" + secrets.token_hex(12)),
                    "name": fn.get("name", ""),
                    "input": args if isinstance(args, dict) else {},
                })
            messages.append({"role": "assistant", "content": blocks or ""})
            continue
        if role == "user":
            messages.append({"role": "user", "content": _anthropic_content_from_openai(content)})

    req: dict = {"messages": messages}
    if system_parts:
        req["system"] = "\n\n".join(part for part in system_parts if part)
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or body.get("max_output_tokens") or 8192
    req["max_tokens"] = max_tokens
    for key in ("temperature", "top_p", "stream"):
        if key in body:
            req[key] = body[key]
    if "stop" in body:
        stop = body["stop"]
        req["stop_sequences"] = stop if isinstance(stop, list) else [stop]

    tool_choice = body.get("tool_choice")
    if tool_choice != "none":
        tools = []
        for tool in body.get("tools") or []:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function") or {}
            tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {}),
            })
        if tools:
            req["tools"] = tools

    if tool_choice:
        if tool_choice == "required":
            req["tool_choice"] = {"type": "any"}
        elif tool_choice == "auto":
            req["tool_choice"] = {"type": "auto"}
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            req["tool_choice"] = {"type": "tool", "name": (tool_choice.get("function") or {}).get("name", "")}

    for key in ("reasoning", "reasoning_effort", "thinking", "output_config", "chat_template_kwargs"):
        if key in body:
            req[key] = body[key]

    return req


def anthropic_to_openai_chat(resp: dict, model: str) -> dict:
    """Convert an Anthropic Messages response to OpenAI Chat Completions."""
    content_parts = []
    reasoning_parts = []
    tool_calls = []
    for block in resp.get("content") or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            content_parts.append(block.get("text", ""))
        elif btype == "thinking":
            reasoning_parts.append(block.get("thinking", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", "toolu_" + secrets.token_hex(12)),
                "type": "function",
                "function": {"name": block.get("name", ""), "arguments": json.dumps(block.get("input") or {})},
            })

    message: dict = {"role": "assistant", "content": "".join(content_parts) if content_parts else None}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    stop_reason = resp.get("stop_reason")
    finish_reason = "tool_calls" if stop_reason == "tool_use" else "length" if stop_reason == "max_tokens" else "stop"
    usage = resp.get("usage") or {}
    return {
        "id": "chatcmpl_" + secrets.token_hex(12),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def openai_to_anthropic(resp: dict, model: str, has_tools: bool = False, thinking_enabled: bool = False) -> dict:
    """Convert OpenAI Chat Completions response to Anthropic Messages format.

    Passes through cache usage (cached_tokens) so Claude Code can see
    Fireworks prompt cache hit rates.
    """
    choice = resp.get("choices", [{}])[0]
    message = choice.get("message", {})

    content = []
    text = message.get("content")
    reasoning_content = message.get("reasoning_content") or message.get("reasoning")

    # Some upstreams (e.g. native serving invocations for reasoning models)
    # return message.content as a list of OpenAI content-part blocks rather
    # than a string, mixing reasoning and text. Flatten so downstream string
    # operations are safe. Shape-safe: only triggers when content is a list.
    if isinstance(text, list):
        from src.streaming import _flatten_list_content
        flat_text, flat_reasoning = _flatten_list_content(text)
        text = flat_text
        if flat_reasoning and not reasoning_content:
            reasoning_content = flat_reasoning

    if not reasoning_content and message.get("reasoning_details"):
        parts = []
        for item in message.get("reasoning_details") or []:
            if isinstance(item, dict):
                parts.append(item.get("text") or item.get("summary") or "")
        reasoning_content = "".join(parts)
    tool_calls_raw = message.get("tool_calls")

    # Thinking/reasoning content
    if reasoning_content and thinking_enabled:
        content.append({"type": "thinking", "thinking": reasoning_content})

    # Tool calls
    if tool_calls_raw:
        if text:
            content.append({"type": "text", "text": text})
        for tc in tool_calls_raw:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            block: dict = {
                "type": "tool_use",
                "id": tc.get("id", "toolu_" + secrets.token_hex(12)),
                "name": fn.get("name", ""),
                "input": args,
            }
            # Preserve Google thought_signature for Gemini models
            ts = (tc.get("extra_content") or {}).get("google", {}).get("thought_signature")
            if ts:
                block["thought_signature"] = ts
            content.append(block)
    elif text:
        content.append({"type": "text", "text": text})
    else:
        content.append({"type": "text", "text": ""})

    # Stop reason
    finish = choice.get("finish_reason", "stop")
    has_tool_use = any(b.get("type") == "tool_use" for b in content)
    if has_tool_use:
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    usage = resp.get("usage", {})

    # Build Anthropic usage — pass through cache stats from Fireworks
    anthropic_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }

    # Fireworks returns cached_tokens in prompt_tokens_details
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens", 0)
    if cached_tokens:
        anthropic_usage["cache_read_input_tokens"] = cached_tokens

    return {
        "id": _gen_msg_id(),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage,
    }
