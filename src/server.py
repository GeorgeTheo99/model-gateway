"""Model Gateway — OpenAI + Anthropic API across cloud backends.

Presents the same dual API contract on port 9111:
  /v1/chat/completions  (OpenAI format — passthrough with auth injection)
  /v1/messages          (Anthropic format — translate to OpenAI, forward, translate back)
  /v1/models            (list routable models)
  /health               (health check)
"""

import base64
import hashlib
import io
import json
import logging
import os
import secrets
import time
from types import SimpleNamespace

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.admin import router as admin_router
from src.auth import require_client_auth
from src.providers import list_models as list_routable_models, resolve
from src.responses import chat_to_responses, responses_to_chat, translate_responses_stream
from src.signature_cache import store_from_extra_content
from src.streaming import translate_stream
from src.translator import anthropic_to_openai, openai_to_anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cloud-gateway")

app = FastAPI(title="Model Gateway")
app.include_router(admin_router)

DEFAULT_VISION_FALLBACK_MODEL = "qwen3.7-plus-fw"
DEFAULT_FIREWORKS_IMAGE_MAX_BYTES = 1_000_000
DEFAULT_FIREWORKS_IMAGE_MAX_DIMENSION = 1600
DEFAULT_FIREWORKS_IMAGE_TOTAL_MAX_BYTES = 8_000_000


def _session_affinity_id(request: Request) -> str:
    """Derive a stable affinity ID for Fireworks x-session-affinity.

    Fireworks caches prompts per-replica; the x-session-affinity header pins
    requests to the same replica so repeated prefixes hit the cache.

    We hash the API key so ALL requests from the same account land on the
    same replica. This means a new Claude Code session benefits from the
    cache warmed by the previous session (the system prompt / tool defs
    are identical). Using per-session IDs would mean every new session
    starts with a cold cache.
    """
    api_key = getattr(request.state, "api_key", "")
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def _forward_headers(request: Request, protocol: str = "openai", provider: str = "") -> dict:
    """Build headers for outbound request to upstream provider.

    Includes Authorization and x-session-affinity for Fireworks prompt caching.
    Uses x-api-key + anthropic-version for Anthropic native API.
    """
    if protocol == "anthropic":
        return {
            "x-api-key": request.state.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    headers = {
        "Authorization": f"Bearer {request.state.api_key}",
    }
    if provider == "fireworks":
        headers["x-session-affinity"] = _session_affinity_id(request)
    return headers


def _error(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "type": "error",
            "error": {"type": error_type, "message": message},
        },
    )


ADAPTIVE_THINKING_ANTHROPIC_MODELS = {
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
}


def _uses_adaptive_anthropic_thinking(provider_model_id: str) -> bool:
    return provider_model_id in ADAPTIVE_THINKING_ANTHROPIC_MODELS


def _remap_max_tokens_for_provider(req: dict, provider: str) -> None:
    """Remap max_tokens → max_completion_tokens for OpenAI models.

    OpenAI's gpt-5.x models reject max_tokens; they require max_completion_tokens.
    Other providers (Fireworks, Google, ZhipuAI) still use max_tokens.
    """
    if provider != "openai":
        return
    if "max_tokens" in req:
        req["max_completion_tokens"] = req.pop("max_tokens")


def _merge_instruction(prefix: str, existing: str) -> str:
    prefix = (prefix or "").strip()
    existing = (existing or "").strip()
    if not prefix:
        return existing
    if not existing:
        return prefix
    if prefix in existing:
        return existing
    return f"{prefix}\n\n{existing}"


def _inject_openai_system_instruction(req: dict, instruction: str) -> None:
    if not instruction:
        return
    messages = req.get("messages")
    if not isinstance(messages, list):
        req["messages"] = [{"role": "system", "content": instruction}]
        return
    for msg in messages:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            msg["content"] = _merge_instruction(instruction, msg["content"])
            return
    messages.insert(0, {"role": "system", "content": instruction})


def _inject_anthropic_system_instruction(req: dict, instruction: str) -> None:
    if not instruction:
        return
    system = req.get("system")
    if isinstance(system, str):
        req["system"] = _merge_instruction(instruction, system)
        return
    if isinstance(system, list):
        existing = " ".join(
            block.get("text", "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        req["system"] = _merge_instruction(instruction, existing)
        return
    req["system"] = instruction


def _inject_responses_instruction(req: dict, instruction: str) -> None:
    if not instruction:
        return
    existing = req.get("instructions")
    if isinstance(existing, str):
        req["instructions"] = _merge_instruction(instruction, existing)
    else:
        req["instructions"] = instruction


def _is_openrouter_gemini(info) -> bool:
    return info.provider == "openrouter" and info.provider_model_id.startswith("google/gemini")


def _strip_fireworks_unsupported_message_fields(req: dict, info) -> None:
    """Remove OpenAI/Pi metadata that Fireworks rejects as extra inputs."""
    if getattr(info, "provider", "") != "fireworks":
        return

    removed = 0
    for key in ("prompt_cache_key", "prompt_cache_retention"):
        if key in req:
            req.pop(key, None)
            removed += 1

    messages = req.get("messages")
    if not isinstance(messages, list):
        if removed:
            log.info("Fireworks request cleanup: stripped %d unsupported field(s)", removed)
        return

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            if key in msg:
                msg.pop(key, None)
                removed += 1
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and "extra_content" in tool_call:
                    tool_call.pop("extra_content", None)
                    removed += 1
    if removed:
        log.info("Fireworks request cleanup: stripped %d unsupported field(s)", removed)


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
        return value if value > 0 else default
    except Exception:
        return default


def _image_url_value(part: dict) -> str | None:
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url")
    else:
        url = image_url
    return url if isinstance(url, str) else None


def _set_image_url_value(part: dict, url: str) -> None:
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        image_url["url"] = url
    else:
        part["image_url"] = {"url": url}


def _inline_image_payloads(req: dict) -> list[tuple[dict, bytes]]:
    messages = req.get("messages")
    if not isinstance(messages, list):
        return []

    payloads = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            url = _image_url_value(part)
            if not url or not url.startswith("data:image/") or ";base64," not in url:
                continue
            _, b64_data = url.split(";base64,", 1)
            try:
                payloads.append((part, base64.b64decode(b64_data, validate=False)))
            except Exception:
                continue
    return payloads


def _compress_fireworks_inline_images(req: dict, info) -> None:
    """Downsample base64 images for Fireworks' documented image/request limits."""
    if getattr(info, "provider", "") != "fireworks":
        return

    payloads = _inline_image_payloads(req)
    if not payloads:
        return

    try:
        from PIL import Image, ImageOps
    except Exception as exc:  # pragma: no cover - dependency/runtime guard
        log.warning("Fireworks image compression skipped: Pillow unavailable: %s", exc)
        return

    max_dim = _env_int("GATEWAY_FIREWORKS_IMAGE_MAX_DIMENSION", DEFAULT_FIREWORKS_IMAGE_MAX_DIMENSION)
    max_bytes = _env_int("GATEWAY_FIREWORKS_IMAGE_MAX_BYTES", DEFAULT_FIREWORKS_IMAGE_MAX_BYTES)
    total_max = _env_int("GATEWAY_FIREWORKS_IMAGE_TOTAL_MAX_BYTES", DEFAULT_FIREWORKS_IMAGE_TOTAL_MAX_BYTES)
    total_before = sum(len(raw) for _, raw in payloads)
    target_bytes = min(max_bytes, max(total_max // max(len(payloads), 1), 200_000))
    force_compress = total_before > total_max

    changed = 0
    total_after = total_before
    for part, raw in payloads:
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image = ImageOps.exif_transpose(image)
                original_size = image.size
                needs_resize = max(original_size) > max_dim
                if not force_compress and len(raw) <= target_bytes and not needs_resize:
                    continue

                image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    alpha = image.getchannel("A") if image.mode in {"RGBA", "LA"} else None
                    background.paste(image.convert("RGBA"), mask=alpha)
                    image = background
                elif image.mode != "RGB":
                    image = image.convert("RGB")

                best = None
                for quality in (85, 75, 65, 55, 45):
                    out = io.BytesIO()
                    image.save(out, format="JPEG", quality=quality, optimize=True)
                    candidate = out.getvalue()
                    best = candidate
                    if len(candidate) <= target_bytes:
                        break
                if not best or len(best) >= len(raw) and not needs_resize:
                    continue

                encoded = base64.b64encode(best).decode("ascii")
                _set_image_url_value(part, f"data:image/jpeg;base64,{encoded}")
                total_after += len(best) - len(raw)
                changed += 1
        except Exception as exc:
            log.warning("Fireworks image compression skipped one image: %s", exc)

    if changed:
        log.info(
            "Fireworks image compression: compressed %d/%d inline image(s), raw bytes %d -> %d",
            changed, len(payloads), total_before, total_after,
        )


_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}
_EFFORT_ALIASES = {"off": "none", "disabled": "none", "max": "xhigh"}
_EFFORT_RATIOS = {"minimal": 0.10, "low": 0.20, "medium": 0.50, "high": 0.80, "xhigh": 0.95, "max": 0.95}


def _normalize_effort(value) -> str | None:
    if value is None:
        return None
    effort = str(value).strip().lower()
    effort = _EFFORT_ALIASES.get(effort, effort)
    if effort in _REASONING_EFFORTS or effort == "none":
        return effort
    return None


def _reasoning_budget(req: dict, effort: str | None, explicit_budget: int | None, info) -> int:
    if explicit_budget:
        return max(1024, int(explicit_budget))
    max_out = (
        req.get("max_tokens")
        or req.get("max_completion_tokens")
        or req.get("max_output_tokens")
        or getattr(info, "max_output_tokens", 0)
        or 32768
    )
    try:
        max_out = int(max_out)
    except Exception:
        max_out = 32768
    ratio = _EFFORT_RATIOS.get(effort or "high", 0.80)
    # Anthropic requires room for final answer tokens above the thinking budget.
    return max(1024, min(int(max_out * ratio), max(max_out - 1024, 1024), 128000))


def _extract_reasoning_control(req: dict, info) -> dict:
    """Read common reasoning controls without mutating the request."""
    enabled = None
    effort = None
    budget = None
    exclude = None

    reasoning = req.get("reasoning")
    if isinstance(reasoning, dict):
        effort = _normalize_effort(reasoning.get("effort")) or effort
        if reasoning.get("max_tokens") is not None:
            try:
                budget = int(reasoning.get("max_tokens"))
            except Exception:
                budget = None
        if reasoning.get("exclude") is not None:
            exclude = bool(reasoning.get("exclude"))
        if reasoning.get("enabled") is not None:
            enabled = bool(reasoning.get("enabled"))
        if effort == "none":
            enabled = False
        elif effort or budget:
            enabled = True

    effort_param = _normalize_effort(req.get("reasoning_effort"))
    if effort_param:
        effort = effort_param
        enabled = effort_param != "none"

    thinking = req.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        if thinking_type in ("enabled", "adaptive"):
            enabled = True
            if thinking.get("budget_tokens") is not None:
                try:
                    budget = int(thinking.get("budget_tokens"))
                except Exception:
                    pass
            output_effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
            effort = _normalize_effort(output_effort) or effort
        elif thinking_type == "disabled":
            enabled = False
            effort = "none"

    chat_template_kwargs = req.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict) and "enable_thinking" in chat_template_kwargs:
        enabled = bool(chat_template_kwargs.get("enable_thinking"))

    if enabled is None and getattr(info, "thinking", "") == "always":
        enabled = True
    if enabled and not effort and not budget:
        effort = "high"

    return {"enabled": enabled, "effort": effort, "budget": budget, "exclude": exclude}


def _infer_thinking_format(info, target_api: str) -> str:
    explicit = (getattr(info, "thinking_format", "") or "").strip().lower()
    if explicit:
        if explicit == "openai-responses" and target_api != "responses":
            return "openai"
        return explicit
    provider = getattr(info, "provider", "")
    model_id = (getattr(info, "provider_model_id", "") or "").lower()
    if provider == "anthropic":
        return "anthropic"
    if provider == "openrouter":
        return "openrouter"
    if provider in {"zhipuai", "zai", "bigmodel"}:
        return "zai"
    if provider == "openai":
        return "openai-responses" if target_api == "responses" else "openai"
    if provider in {"omlx", "local", "mlx"}:
        if "qwen" in model_id:
            return "qwen-chat-template"
        if "glm" in model_id:
            return "zai"
    if provider == "google":
        return "google-openai"
    return "openai"


def _strip_reasoning_controls(req: dict) -> None:
    req.pop("reasoning_effort", None)
    req.pop("thinking", None)
    req.pop("output_config", None)
    # Preserve other chat_template_kwargs, but drop the thinking toggle unless
    # the selected backend format explicitly re-adds it.
    ctk = req.get("chat_template_kwargs")
    if isinstance(ctk, dict) and "enable_thinking" in ctk:
        ctk = dict(ctk)
        ctk.pop("enable_thinking", None)
        if ctk:
            req["chat_template_kwargs"] = ctk
        else:
            req.pop("chat_template_kwargs", None)


def _payload_has_image(payload: dict) -> bool:
    """Check if the payload contains any image content."""
    msgs = payload.get("messages", [])
    if not msgs:
        msgs = payload.get("contents", [])
    for m in msgs:
        content = m.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in {"image", "image_url"} or "image_url" in part:
                return True
    return False


def _apply_gateway_reasoning(req: dict, info, target_api: str = "chat") -> bool:
    """Normalize common thinking params for the resolved upstream provider.

    Returns whether reasoning/thinking should be considered enabled for response
    translators.  The function is intentionally additive/backward-compatible:
    optional-thinking models receive no new params unless the client requested
    them; always-thinking models keep the prior auto-enable behavior.
    """
    control = _extract_reasoning_control(req, info)
    enabled = control["enabled"]
    if enabled is None:
        return False

    if not getattr(info, "thinking", "") and not getattr(info, "thinking_format", ""):
        _strip_reasoning_controls(req)
        if target_api != "responses":
            req.pop("reasoning", None)
        return False

    fmt = _infer_thinking_format(info, target_api)
    effort = control["effort"] or ("high" if enabled else "none")
    budget = control["budget"]
    exclude = control["exclude"]

    _strip_reasoning_controls(req)
    if target_api != "responses":
        req.pop("reasoning", None)

    if fmt in {"none", "disabled", "google-openai"}:
        return bool(enabled)

    if target_api == "responses" or fmt == "openai-responses":
        if enabled:
            req["reasoning"] = {"effort": effort if effort != "max" else "high"}
        elif effort == "none":
            req["reasoning"] = {"effort": "none"}
        return bool(enabled)

    if fmt == "anthropic":
        if enabled:
            req["thinking"] = {
                "type": "enabled",
                "budget_tokens": _reasoning_budget(req, effort, budget, info),
            }
        return bool(enabled)

    if fmt == "openrouter":
        reasoning = {}
        if budget:
            reasoning["max_tokens"] = budget
        elif effort:
            reasoning["effort"] = _EFFORT_ALIASES.get(effort, effort)
        else:
            reasoning["enabled"] = bool(enabled)
        if exclude is not None:
            reasoning["exclude"] = exclude
        req["reasoning"] = reasoning
        return bool(enabled)

    if fmt == "qwen-chat-template":
        ctk = dict(req.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = bool(enabled)
        if enabled:
            ctk.setdefault("preserve_thinking", True)
        req["chat_template_kwargs"] = ctk
        if budget:
            req["thinking_budget"] = budget
        return bool(enabled)

    if fmt == "qwen":
        req["enable_thinking"] = bool(enabled)
        if budget:
            req["thinking_budget"] = budget
        return bool(enabled)

    if fmt == "zai":
        req["enable_thinking"] = bool(enabled)
        if enabled and effort and effort != "none":
            # Z.ai GLM-5.x accepts two effort levels: "high" (default) and
            # "max". Internally "max" is normalized to "xhigh" (see
            # _EFFORT_ALIASES); map it back to the literal Z.ai expects, and
            # clamp finer-grained gateway levels (minimal/low/medium) to "high"
            # — matching Z.ai's own Claude Code effort map (low/med/high→high,
            # xhigh→max).
            req["reasoning_effort"] = "max" if effort == "xhigh" else "high"
        return bool(enabled)

    if fmt == "deepseek":
        req["thinking"] = {"type": "enabled" if enabled else "disabled"}
        if enabled and effort:
            req["reasoning_effort"] = effort
        return bool(enabled)

    # Default OpenAI-compatible shape.
    if enabled and effort and effort != "none":
        req["reasoning_effort"] = effort if effort != "max" else "high"
    elif effort == "none":
        req["reasoning_effort"] = "none"
    return bool(enabled)


# ── thinking observability ─────────────────────────────────────────────────
# These helpers introspect the dispatch at runtime so the matrix exposed by
# /v1/models and /v1/debug/thinking stays accurate as _apply_gateway_reasoning
# evolves, rather than duplicating the per-format forwarding logic.

_THINK_UPSTREAM_KEYS = ("enable_thinking", "reasoning", "reasoning_effort",
                        "thinking", "chat_template_kwargs", "thinking_budget")
# Params that carry an effort/budget *level* (not just an on/off toggle).
_THINK_LEVEL_KEYS = ("reasoning", "reasoning_effort", "thinking", "thinking_budget")
# Effort levels the gateway recognizes and normalizes (see _REASONING_EFFORTS).
_GATEWAY_EFFORT_LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"]


def _probe_forwarded_params(entry: dict) -> set[str]:
    """Probe the dispatch with max-effort requests; return the set of
    thinking-related upstream params that actually get forwarded.

    Two probes (effort-only and effort+budget) are unioned so format-specific
    budget forwarding (qwen thinking_budget, anthropic thinking.budget_tokens)
    is detected even when effort alone is not forwarded (e.g. the zai branch).
    """
    info = SimpleNamespace(
        provider=entry.get("provider", ""),
        provider_model_id=entry.get("provider_model_id", entry.get("name", "")),
        thinking=entry.get("thinking", ""),
        thinking_format=entry.get("thinking_format", ""),
        max_output_tokens=entry.get("max_output_tokens", 0) or 32768,
    )
    forwarded: set[str] = set()
    for probe in (
        {"messages": [], "reasoning_effort": "max"},
        {"messages": [], "reasoning": {"effort": "max", "max_tokens": 8000}},
    ):
        req = dict(probe)
        _apply_gateway_reasoning(req, info, target_api="chat")
        forwarded |= {k for k in _THINK_UPSTREAM_KEYS if k in req}
    return forwarded


def _thinking_capabilities(entry: dict) -> dict:
    """Runtime introspection of what thinking control a model forwards upstream."""
    forwarded = _probe_forwarded_params(entry)
    info = SimpleNamespace(
        provider=entry.get("provider", ""),
        provider_model_id=entry.get("provider_model_id", entry.get("name", "")),
        thinking_format=entry.get("thinking_format", ""),
    )
    fmt = entry.get("thinking_format", "") or _infer_thinking_format(info, target_api="chat")
    return {
        "thinking": entry.get("thinking", ""),
        "thinking_format": fmt,
        "forwarded_params": sorted(forwarded),
        "max_reachable": bool(forwarded & set(_THINK_LEVEL_KEYS)),
        "gateway_levels": _GATEWAY_EFFORT_LEVELS,
    }


def _has_cache_control(content) -> bool:
    if isinstance(content, list):
        return any(isinstance(block, dict) and "cache_control" in block for block in content)
    return False


def _cacheable_text(text: str) -> bool:
    # Gemini cache writes have token minimums. This conservative character
    # threshold avoids adding cache markers to tiny prompts that cannot benefit.
    return len(text or "") >= 4000


def _enable_openrouter_gemini_prompt_cache(req: dict) -> None:
    """Mark stable prompt prefix content for OpenRouter Gemini prompt caching.

    OpenRouter's Gemini integration uses explicit cache_control breakpoints on
    message content blocks. We prefer system/instruction content because it is
    the most stable prefix across agent turns. OpenRouter uses the last Gemini
    breakpoint, so only add one when the request has no explicit marker.

    Disabled for now: when OpenRouter turns the marked prefix into Google's
    cachedContent, later requests from Codex still include tools and system
    instructions. Google's Gemini API rejects that shape with:
    "Tool config, tools and system instruction should not be set in the
    request when using cached content."
    """
    return


@app.get("/health")
async def health():
    return {"status": "ok", "service": "model-gateway"}


@app.get("/v1/models")
async def list_models(request: Request):
    require_client_auth(request)
    models = list_routable_models()
    data = []
    for m in models:
        caps = _thinking_capabilities(m)
        data.append({
            "id": m.get("id") or m["name"],
            "object": "model",
            "created": 0,
            "owned_by": m.get("provider", "cloud"),
            "thinking": caps["thinking"],
            "thinking_format": caps["thinking_format"],
            "thinking_levels": caps["gateway_levels"],
            "max_reachable": caps["max_reachable"],
            "forwarded_params": caps["forwarded_params"],
        })
    return {"object": "list", "data": data}


@app.get("/v1/debug/thinking")
async def debug_thinking(request: Request):
    require_client_auth(request)
    """Read-only matrix of per-model thinking forwarding, derived at runtime.

    Surfaces, for every routable model, which thinking params the dispatch
    actually forwards upstream under a max-effort probe — making silent gaps
    (e.g. zai dropping reasoning_effort) visible without reading the source.
    """
    seen: dict[str, dict] = {}
    for m in list_routable_models():
        seen.setdefault(m.get("name") or m["id"], m)
    rows = []
    for name, m in seen.items():
        caps = _thinking_capabilities(m)
        rows.append({
            "name": name,
            "provider": m.get("provider", ""),
            "provider_model_id": m.get("provider_model_id", ""),
            "thinking": caps["thinking"],
            "thinking_format": caps["thinking_format"],
            "forwarded_params": caps["forwarded_params"],
            "max_reachable": caps["max_reachable"],
        })
    by_format: dict[str, int] = {}
    for r in rows:
        by_format[r["thinking_format"]] = by_format.get(r["thinking_format"], 0) + 1
    reachable = sum(1 for r in rows if r["max_reachable"])
    return {
        "note": "Runtime introspection of _apply_gateway_reasoning. "
                "forwarded_params = upstream keys actually sent for a max-effort probe. "
                "max_reachable = a level-carrying param (effort/budget) reaches upstream.",
        "gateway_levels": _GATEWAY_EFFORT_LEVELS,
        "summary": {
            "models": len(rows),
            "max_reachable": reachable,
            "max_unreachable": len(rows) - reachable,
            "by_format": by_format,
        },
        "models": rows,
    }


# ── OpenAI Responses API endpoint (Codex CLI) ────────────────────────────────


@app.post("/v1/responses")
async def create_response(request: Request):
    """OpenAI Responses API — for Codex CLI compatibility.

    Translates Responses API requests to Chat Completions, forwards to cloud
    provider, then translates results back to Responses format.
    """
    require_client_auth(request)
    try:
        body = await request.json()
    except Exception:
        return _error(400, "invalid_request_error", "Invalid JSON body")

    model = body.get("model", "")
    is_stream = body.get("stream", False)

    info = resolve(model)
    if not info:
        return _error(404, "invalid_request_error", f"Model '{model}' not found in gateway")

    _inject_responses_instruction(body, info.system_instruction)

    # Anthropic models don't support Responses API directly — translate via Messages
    if info.protocol == "anthropic":
        # Translate Responses → Anthropic Messages, forward, translate back
        return await _handle_responses_anthropic(body, info, model, request)

    # OpenAI models support Responses natively. Do not translate Codex's
    # Responses+tools requests to Chat Completions: GPT-5.4 rejects function
    # tools with reasoning_effort on /chat/completions.
    if info.provider == "openai":
        body["model"] = info.provider_model_id
        _apply_gateway_reasoning(body, info, target_api="responses")
        request.state.api_key = info.api_key
        fwd = _forward_headers(request, provider=info.provider)
        endpoint = f"{info.base_url}/responses"
        log.info("Responses %s -> openai native (stream=%s)", model, is_stream)
        return await _handle_openai_responses_passthrough(endpoint, body, fwd, is_stream)

    # Translate Responses → Chat Completions
    chat_req = responses_to_chat(body)
    _inject_openai_system_instruction(chat_req, info.system_instruction)
    chat_req["model"] = info.provider_model_id
    _remap_max_tokens_for_provider(chat_req, info.provider)
    endpoint = f"{info.base_url}/chat/completions"

    for key in ("reasoning", "reasoning_effort", "thinking", "output_config", "chat_template_kwargs"):
        if key in body and key not in chat_req:
            chat_req[key] = body[key]
    thinking_enabled = _apply_gateway_reasoning(chat_req, info, target_api="chat")
    _strip_fireworks_unsupported_message_fields(chat_req, info)
    _compress_fireworks_inline_images(chat_req, info)
    if _is_openrouter_gemini(info):
        _enable_openrouter_gemini_prompt_cache(chat_req)
    if thinking_enabled:
        log.info("  Reasoning enabled for %s via %s", model, _infer_thinking_format(info, "chat"))

    # Debug: log tool_calls in messages to diagnose invalid JSON errors
    for msg in chat_req.get("messages", []):
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "")
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except (json.JSONDecodeError, TypeError):
                log.error("INVALID tool_call args for %s: type=%s value=%r", fn.get("name"), type(args).__name__, str(args)[:300])

    # Attach API key to request.state so _forward_headers can use it
    request.state.api_key = info.api_key
    fwd = _forward_headers(request, provider=info.provider)

    log.info("Responses %s -> %s (stream=%s)", model, info.provider, is_stream)
    resp_tools = body.get("tools", [])
    if resp_tools:
        tool_names = [t.get("name", "?") for t in resp_tools]
        log.info("  Tools: %s", tool_names)

    # Google's OpenAI-compatible endpoint may not include extra_content.google.thought_signature
    # in streaming deltas. For tool requests, use non-streaming upstream to guarantee we capture
    # thought_signatures (Gemini 3 requires them on functionCall parts).
    google_with_tools = (info.provider == "google" or _is_openrouter_gemini(info)) and bool(resp_tools)

    if is_stream and not google_with_tools:
        # Force streaming for the upstream request
        chat_req["stream"] = True
        return await _handle_responses_stream(endpoint, chat_req, model, fwd)
    if is_stream and google_with_tools:
        # Use non-streaming upstream to capture thought_signatures reliably
        chat_req.pop("stream", None)
        return await _handle_responses_stream_google(endpoint, chat_req, model, fwd)

    # Non-streaming client response
    if google_with_tools:
        # Non-streaming upstream for Google+tools to capture thought_signatures
        async with httpx.AsyncClient(timeout=300) as client:
            try:
                resp = await client.post(endpoint, json=chat_req, headers=fwd)
            except httpx.ConnectError:
                return _error(502, "api_error", "Cannot connect to Google API")
            except Exception as e:
                return _error(502, "api_error", f"Google error: {e}")

        if resp.status_code != 200:
            return _error(502, "api_error", f"Google returned {resp.status_code}: {resp.text[:500]}")

        openai_resp = resp.json()

        # Capture and cache thought_signatures
        for tc in (openai_resp.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []):
            ec = tc.get("extra_content")
            if ec:
                tc_id = tc.get("id", "")
                store_from_extra_content(tc_id, ec)
                ts = (ec.get("google") or {}).get("thought_signature")
                if ts:
                    log.info("signature_cache: captured thought_signature from non-streaming response for %s", tc_id)
    else:
        # Stream from Fireworks (required for max_tokens > 4096)
        chat_req["stream"] = True
        chat_req["stream_options"] = {"include_usage": True}

        client = httpx.AsyncClient(timeout=300)
        try:
            resp = await client.send(
                client.build_request("POST", endpoint, json=chat_req, headers=fwd),
                stream=True,
            )
        except httpx.ConnectError:
            await client.aclose()
            return _error(502, "api_error", "Cannot connect to cloud provider")
        except Exception as e:
            await client.aclose()
            return _error(502, "api_error", f"Provider error: {e}")

        if resp.status_code != 200:
            err_body = await resp.aread()
            await resp.aclose()
            await client.aclose()
            return _error(502, "api_error", f"Provider returned {resp.status_code}: {err_body.decode()[:500]}")

        try:
            openai_resp = await _collect_stream(resp)
        except Exception as e:
            return _error(502, "api_error", f"Stream collection failed: {e}")
        finally:
            await resp.aclose()
            await client.aclose()

    result = chat_to_responses(openai_resp, model)

    # Log cache hit rate
    usage = openai_resp.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if prompt_tokens and cached:
        log.info("Cache hit: %d/%d prompt tokens cached (%.0f%%)", cached, prompt_tokens, cached / prompt_tokens * 100)

    return JSONResponse(content=result)


async def _handle_openai_responses_passthrough(endpoint: str, body: dict, headers: dict, is_stream: bool):
    """Forward Responses API requests directly to OpenAI."""
    if is_stream:
        client = httpx.AsyncClient(timeout=300)
        try:
            resp = await client.send(
                client.build_request("POST", endpoint, json=body, headers=headers),
                stream=True,
            )
        except httpx.ConnectError:
            await client.aclose()
            return _error(502, "api_error", "Cannot connect to OpenAI API")
        except Exception as e:
            await client.aclose()
            return _error(502, "api_error", f"OpenAI error: {e}")

        if resp.status_code != 200:
            err_body = await resp.aread()
            await resp.aclose()
            await client.aclose()
            return _error(502, "api_error", f"OpenAI returned {resp.status_code}: {err_body.decode()[:500]}")

        async def event_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async with httpx.AsyncClient(timeout=300) as client:
        try:
            resp = await client.post(endpoint, json=body, headers=headers)
        except httpx.ConnectError:
            return _error(502, "api_error", "Cannot connect to OpenAI API")
        except Exception as e:
            return _error(502, "api_error", f"OpenAI error: {e}")

    try:
        content = resp.json()
    except ValueError:
        return _error(502, "api_error", f"OpenAI returned invalid JSON: {resp.text[:500]}")

    return JSONResponse(status_code=resp.status_code, content=content)


async def _handle_responses_stream(endpoint: str, chat_req: dict, model: str, headers: dict):
    """Handle streaming Responses API request."""
    client = httpx.AsyncClient(timeout=300)
    try:
        resp = await client.send(
            client.build_request("POST", endpoint, json=chat_req, headers=headers),
            stream=True,
        )
    except httpx.ConnectError:
        await client.aclose()
        return _error(502, "api_error", "Cannot connect to cloud provider")
    except Exception as e:
        await client.aclose()
        return _error(502, "api_error", f"Provider error: {e}")

    if resp.status_code != 200:
        err_body = await resp.aread()
        await resp.aclose()
        await client.aclose()
        return _error(502, "api_error", f"Provider returned {resp.status_code}: {err_body.decode()[:500]}")

    async def event_generator():
        try:
            async for event in translate_responses_stream(resp.aiter_bytes(), model):
                yield event
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _handle_responses_stream_google(endpoint: str, chat_req: dict, model: str, headers: dict):
    """Handle streaming Responses API for Google+tools: non-streaming upstream
    to guarantee thought_signature capture, then generate Responses SSE events."""
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            resp = await client.post(endpoint, json=chat_req, headers=headers)
        except httpx.ConnectError:
            return _error(502, "api_error", "Cannot connect to Google API")
        except Exception as e:
            return _error(502, "api_error", f"Google error: {e}")

    if resp.status_code != 200:
        return _error(502, "api_error", f"Google returned {resp.status_code}: {resp.text[:500]}")

    openai_resp = resp.json()

    # Capture and cache thought_signatures from all tool_calls
    for tc in (openai_resp.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []):
        ec = tc.get("extra_content")
        if ec:
            tc_id = tc.get("id", "")
            store_from_extra_content(tc_id, ec)
            ts = (ec.get("google") or {}).get("thought_signature")
            if ts:
                log.info("signature_cache: captured thought_signature from non-streaming response for %s", tc_id)

    # Convert to Responses format
    result = chat_to_responses(openai_resp, model)

    # Generate Responses API SSE events from the complete response
    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def generate_sse():
        # response.created
        yield _sse("response.created", {"type": "response.created", "response": result})

        # Output items
        for idx, item in enumerate(result.get("output", [])):
            item_type = item.get("type", "message")
            if item_type == "function_call":
                yield _sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": {"type": "function_call", "id": item.get("id", ""), "call_id": item.get("call_id", ""), "name": item.get("name", ""), "arguments": ""},
                })
                yield _sse("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "output_index": idx,
                    "item_id": item.get("id", ""),
                    "call_id": item.get("call_id", ""),
                    "delta": item.get("arguments", ""),
                })
                yield _sse("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "output_index": idx,
                    "item_id": item.get("id", ""),
                    "call_id": item.get("call_id", ""),
                    "arguments": item.get("arguments", ""),
                })
                yield _sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": item,
                })
            elif item_type == "message":
                for cidx, content_part in enumerate(item.get("content", [])):
                    yield _sse("response.output_item.added", {
                        "type": "response.output_item.added",
                        "output_index": idx,
                        "item": {"type": "message", "id": item.get("id", ""), "role": item.get("role", "assistant"), "content": []},
                    })
                    yield _sse("response.content_part.added", {
                        "type": "response.content_part.added",
                        "output_index": idx,
                        "content_index": cidx,
                        "part": {"type": content_part.get("type", "output_text"), "text": ""},
                    })
                    yield _sse("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "output_index": idx,
                        "content_index": cidx,
                        "delta": content_part.get("text", ""),
                    })
                    yield _sse("response.output_text.done", {
                        "type": "response.output_text.done",
                        "output_index": idx,
                        "content_index": cidx,
                        "text": content_part.get("text", ""),
                    })
                yield _sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": item,
                })

        # response.completed
        yield _sse("response.completed", {"type": "response.completed", "response": result})

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ── OpenAI-compatible endpoint ──────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible passthrough — forward to resolved provider with auth."""
    require_client_auth(request)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
        )

    model = body.get("model", "")
    info = resolve(model)
    if not info:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"Model '{model}' not found in gateway", "type": "invalid_request_error"}},
        )

    # --- Vision Fallback Routing ---
    # When a text-only model receives an image, transparently reroute the request
    # to a vision-capable model via the gateway. Vision capability is authoritative:
    # it comes from the `vision` flag in model-info.json (propagated onto
    # ProviderInfo), not from a keyword heuristic. Keep model-info.json accurate.
    vision_capable = info.vision

    fallback_model = os.environ.get("GATEWAY_VISION_FALLBACK", DEFAULT_VISION_FALLBACK_MODEL)
    if not vision_capable and _payload_has_image(body) and fallback_model:
        log.info("Vision fallback: rerouting '%s' to '%s'", model, fallback_model)
        fallback_info = resolve(fallback_model)
        if not fallback_info:
            # Fail loud rather than silently forwarding the image to the original
            # text-only model (which would 400 upstream with an opaque error and
            # hide that the vision fallback is misconfigured).
            log.error(
                "Vision fallback model '%s' is not resolvable in model-info.json; "
                "cannot reroute image request for text-only model '%s'",
                fallback_model, model,
            )
            return _error(
                502,
                "invalid_request_error",
                f"Vision fallback model '{fallback_model}' is not available; "
                f"cannot route image input for text-only model '{model}'. "
                f"Set GATEWAY_VISION_FALLBACK to a resolvable vision-capable model.",
            )
        if not fallback_info.vision:
            log.error(
                "Vision fallback model '%s' is not vision-capable (vision flag is false); "
                "cannot reroute image request for text-only model '%s'",
                fallback_model, model,
            )
            return _error(
                502,
                "invalid_request_error",
                f"Vision fallback model '{fallback_model}' is not vision-capable; "
                f"cannot route image input for text-only model '{model}'. "
                f"Set GATEWAY_VISION_FALLBACK to a vision-capable model.",
            )
        info = fallback_info

    # Swap model to the provider's model ID
    body["model"] = info.provider_model_id

    _inject_openai_system_instruction(body, info.system_instruction)
    is_stream = body.get("stream", False)

    thinking_enabled = _apply_gateway_reasoning(body, info, target_api="chat")
    _strip_fireworks_unsupported_message_fields(body, info)
    _compress_fireworks_inline_images(body, info)
    if _is_openrouter_gemini(info):
        _enable_openrouter_gemini_prompt_cache(body)

    # Attach API key to request.state so _forward_headers can use it
    request.state.api_key = info.api_key
    fwd = _forward_headers(request, protocol=info.protocol, provider=info.provider)

    # Anthropic models don't support OpenAI Chat Completions
    if info.protocol == "anthropic":
        return _error(400, "invalid_request_error",
                       f"Model '{model}' uses Anthropic Messages API — use /v1/messages endpoint instead")

    _remap_max_tokens_for_provider(body, info.provider)
    endpoint = f"{info.base_url}/chat/completions"

    log.info("OpenAI %s -> %s (stream=%s, thinking=%s)", model, info.provider, is_stream, thinking_enabled)

    if is_stream:
        return await _passthrough_stream(endpoint, body, fwd)
    return await _passthrough_sync(endpoint, body, fwd)


async def _passthrough_sync(endpoint: str, body: dict, headers: dict) -> JSONResponse:
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            resp = await client.post(
                endpoint,
                json=body,
                headers=headers,
            )
        except httpx.ConnectError:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "Cannot connect to cloud provider", "type": "api_error"}},
            )
        except Exception as e:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": str(e), "type": "api_error"}},
            )

    return JSONResponse(status_code=resp.status_code, content=resp.json())


async def _passthrough_stream(endpoint: str, body: dict, headers: dict):
    client = httpx.AsyncClient(timeout=300)
    try:
        resp = await client.send(
            client.build_request(
                "POST",
                endpoint,
                json=body,
                headers=headers,
            ),
            stream=True,
        )
    except httpx.ConnectError:
        await client.aclose()
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Cannot connect to cloud provider", "type": "api_error"}},
        )
    except Exception as e:
        await client.aclose()
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(e), "type": "api_error"}},
        )

    if resp.status_code != 200:
        err_body = await resp.aread()
        await resp.aclose()
        await client.aclose()
        return JSONResponse(
            status_code=resp.status_code,
            content={"error": {"message": err_body.decode()[:500], "type": "api_error"}},
        )

    async def stream_generator():
        try:
            async for line in resp.aiter_lines():
                yield (line + "\n").encode()
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


async def _collect_stream(resp: httpx.Response) -> dict:
    """Consume an OpenAI SSE stream and reassemble a complete chat completion.

    Returns a dict shaped like a non-streaming OpenAI response so existing
    translation logic works unchanged.
    """
    content = ""
    reasoning_content = ""
    tool_calls: dict[int, dict] = {}  # index -> {id, function: {name, arguments}}
    finish_reason = None
    usage = {}

    buffer = ""
    async for chunk in resp.aiter_bytes():
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
                usage = data["usage"]

            choice = (data.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

            if delta.get("content"):
                content += delta["content"]
            reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
            if not reasoning_delta and delta.get("reasoning_details"):
                parts = []
                for item in delta.get("reasoning_details") or []:
                    if isinstance(item, dict):
                        parts.append(item.get("text") or item.get("summary") or "")
                reasoning_delta = "".join(parts)
            if reasoning_delta:
                reasoning_content += reasoning_delta

            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                fn = tc.get("function", {})
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {"name": fn.get("name", ""), "arguments": ""},
                    }
                # Capture Google thought_signature from any tool_call delta
                # (may arrive on first or subsequent deltas)
                if tc.get("extra_content"):
                    tool_calls[idx]["extra_content"] = tc["extra_content"]
                    store_from_extra_content(tc.get("id") or tool_calls[idx].get("id", ""), tc["extra_content"])
                tool_calls[idx]["function"]["arguments"] += fn.get("arguments", "")

    # Assemble response in OpenAI non-streaming shape
    message: dict = {"role": "assistant"}
    if content:
        message["content"] = content
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason or "stop"}],
        "usage": usage,
    }


# ── Anthropic Messages endpoint ─────────────────────────────────────────────


@app.post("/v1/messages")
async def messages(request: Request):
    """Anthropic Messages API — route based on provider protocol.

    Anthropic models: passthrough directly to Anthropic's native Messages API.
    Other providers: translate to OpenAI, forward, translate back.
    """
    require_client_auth(request)
    try:
        body = await request.json()
    except Exception:
        return _error(400, "invalid_request_error", "Invalid JSON body")

    model = body.get("model", "")
    is_stream = body.get("stream", False)
    has_tools = bool(body.get("tools"))

    info = resolve(model)
    if not info:
        return _error(404, "invalid_request_error", f"Model '{model}' not found in gateway")

    _inject_anthropic_system_instruction(body, info.system_instruction)

    # Check/normalize thinking — from request or model config. Native Anthropic
    # requests are normalized here; non-Anthropic requests are normalized after
    # translation to the upstream OpenAI-compatible shape.
    if info.protocol == "anthropic":
        thinking_enabled = _apply_gateway_reasoning(body, info, target_api="messages")
    else:
        control = _extract_reasoning_control(body, info)
        thinking_enabled = bool(control["enabled"])
    thinking_param = body.get("thinking")

    # Rewrite thinking param for models that require adaptive thinking.
    # Claude Opus 4.6 and earlier:       thinking.type = "enabled" + budget_tokens
    # Opus 4.7+, Fable 5, and Mythos 5:  thinking.type = "adaptive" + output_config.effort
    _uses_adaptive_thinking = _uses_adaptive_anthropic_thinking(info.provider_model_id)

    if _uses_adaptive_thinking and isinstance(thinking_param, dict):
        if thinking_param.get("type") == "enabled":
            # Convert old-style "enabled" to "adaptive" + output_config.effort
            body["thinking"] = {"type": "adaptive"}
            budget = thinking_param.get("budget_tokens")
            if budget and budget >= 10000:
                body["output_config"] = {"effort": "high"}
            elif budget:
                body["output_config"] = {"effort": "medium"}
            else:
                body["output_config"] = {"effort": "high"}
        elif thinking_param.get("type") == "disabled":
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": "low"}
    elif not _uses_adaptive_thinking and isinstance(thinking_param, dict) and thinking_param.get("type") == "adaptive":
        # Convert new-style "adaptive" back to "enabled" for older models
        effort = (body.get("output_config") or {}).get("effort", "high")
        budget_map = {"high": 10000, "medium": 5000, "low": 2000}
        body["thinking"] = {"type": "enabled", "budget_tokens": budget_map.get(effort, 10000)}
        body.pop("output_config", None)

    # Attach API key to request.state so _forward_headers can use it
    request.state.api_key = info.api_key

    # Anthropic native passthrough — no translation needed
    if info.protocol == "anthropic":
        body["model"] = info.provider_model_id
        fwd = _forward_headers(request, protocol="anthropic", provider=info.provider)
        endpoint = f"{info.base_url}/messages"

        log.info("Messages %s -> %s native (stream=%s, tools=%s, thinking=%s)", model, info.provider, is_stream, has_tools, thinking_enabled)
        if has_tools:
            tool_names = [t.get("name", "?") for t in body.get("tools", [])]
            log.info("  Tools: %s", tool_names)

        if is_stream:
            return await _passthrough_anthropic_stream(endpoint, body, fwd)
        return await _passthrough_anthropic_sync(endpoint, body, fwd)

    # Non-Anthropic providers: translate Anthropic → OpenAI → forward → translate back
    openai_req = anthropic_to_openai(body)
    openai_req["model"] = info.provider_model_id
    _remap_max_tokens_for_provider(openai_req, info.provider)

    # Preserve common reasoning controls that Anthropic→OpenAI translation may
    # not understand yet, then normalize them for the selected upstream.
    for key in ("reasoning", "reasoning_effort", "thinking", "output_config", "chat_template_kwargs"):
        if key in body and key not in openai_req:
            openai_req[key] = body[key]
    thinking_enabled = _apply_gateway_reasoning(openai_req, info, target_api="chat") or thinking_enabled
    _strip_fireworks_unsupported_message_fields(openai_req, info)
    _compress_fireworks_inline_images(openai_req, info)
    if _is_openrouter_gemini(info):
        _enable_openrouter_gemini_prompt_cache(openai_req)

    endpoint = f"{info.base_url}/chat/completions"
    fwd = _forward_headers(request, protocol=info.protocol, provider=info.provider)

    log.info("Messages %s -> %s (stream=%s, tools=%s, thinking=%s)", model, info.provider, is_stream, has_tools, thinking_enabled)
    if has_tools:
        tool_names = [t.get("name", "?") for t in body.get("tools", [])]
        log.info("  Tools: %s", tool_names)

    # Google's OpenAI-compatible endpoint may not include extra_content.google.thought_signature
    # in streaming deltas. For tool requests, use non-streaming upstream to guarantee we capture
    # thought_signatures (Gemini 3 requires them on functionCall parts).
    google_with_tools = (info.provider == "google" or _is_openrouter_gemini(info)) and has_tools

    if is_stream and not google_with_tools:
        return await _handle_streaming(endpoint, openai_req, model, fwd, has_tools, thinking_enabled)
    if is_stream and google_with_tools:
        return await _handle_streaming_google(endpoint, openai_req, model, fwd, has_tools, thinking_enabled)
    return await _handle_sync(endpoint, openai_req, model, fwd, has_tools, thinking_enabled)


# ── Anthropic native passthrough helpers ─────────────────────────────────────


async def _passthrough_anthropic_sync(endpoint: str, body: dict, headers: dict) -> JSONResponse:
    """Forward request directly to Anthropic Messages API (non-streaming)."""
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            resp = await client.post(endpoint, json=body, headers=headers)
        except httpx.ConnectError:
            return _error(502, "api_error", "Cannot connect to Anthropic API")
        except Exception as e:
            return _error(502, "api_error", f"Anthropic error: {e}")

    return JSONResponse(status_code=resp.status_code, content=resp.json())


async def _passthrough_anthropic_stream(endpoint: str, body: dict, headers: dict):
    """Forward streaming request directly to Anthropic Messages API."""
    client = httpx.AsyncClient(timeout=300)
    try:
        resp = await client.send(
            client.build_request("POST", endpoint, json=body, headers=headers),
            stream=True,
        )
    except httpx.ConnectError:
        await client.aclose()
        return _error(502, "api_error", "Cannot connect to Anthropic API")
    except Exception as e:
        await client.aclose()
        return _error(502, "api_error", f"Anthropic error: {e}")

    if resp.status_code != 200:
        err_body = await resp.aread()
        await resp.aclose()
        await client.aclose()
        return _error(502, "api_error", f"Anthropic returned {resp.status_code}: {err_body.decode()[:500]}")

    async def stream_generator():
        try:
            async for line in resp.aiter_lines():
                yield (line + "\n").encode()
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _handle_responses_anthropic(body: dict, info, model: str, request: Request):
    """Translate Responses API request to Anthropic Messages, forward, translate back.

    Used when Codex CLI targets an Anthropic model via the Responses API.
    """
    # Responses → Anthropic Messages translation
    messages_req = _responses_to_anthropic_messages(body)
    _inject_anthropic_system_instruction(messages_req, info.system_instruction)
    messages_req["model"] = info.provider_model_id

    # Map reasoning_effort from Responses API to Anthropic thinking param
    _uses_adaptive = _uses_adaptive_anthropic_thinking(info.provider_model_id)
    reasoning_effort = body.get("reasoning", {}).get("effort") if isinstance(body.get("reasoning"), dict) else None

    if reasoning_effort or info.thinking in ("optional", "always"):
        effort = reasoning_effort or "high"
        if _uses_adaptive:
            messages_req["thinking"] = {"type": "adaptive"}
            messages_req["output_config"] = {"effort": effort}
        else:
            budget_map = {"high": 10000, "medium": 5000, "low": 2000}
            messages_req["thinking"] = {"type": "enabled", "budget_tokens": budget_map.get(effort, 10000)}

    is_stream = body.get("stream", False)
    request.state.api_key = info.api_key
    fwd = _forward_headers(request, protocol="anthropic", provider=info.provider)
    endpoint = f"{info.base_url}/messages"

    log.info("Responses %s -> %s anthropic (stream=%s)", model, info.provider, is_stream)

    if is_stream:
        # Stream from Anthropic, collect, translate to Responses stream format
        messages_req["stream"] = True
        return await _handle_responses_anthropic_stream(endpoint, messages_req, model, fwd)

    # Non-streaming: forward to Anthropic, translate response to Responses format
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            resp = await client.post(endpoint, json=messages_req, headers=fwd)
        except httpx.ConnectError:
            return _error(502, "api_error", "Cannot connect to Anthropic API")
        except Exception as e:
            return _error(502, "api_error", f"Anthropic error: {e}")

    if resp.status_code != 200:
        return _error(502, "api_error", f"Anthropic returned {resp.status_code}: {resp.text[:500]}")

    anthropic_resp = resp.json()
    result = _anthropic_messages_to_responses(anthropic_resp, model)
    return JSONResponse(content=result)


def _responses_to_anthropic_messages(body: dict) -> dict:
    """Convert OpenAI Responses API request to Anthropic Messages format."""
    system_text = ""
    instructions = body.get("instructions")
    if instructions and isinstance(instructions, str):
        system_text = instructions

    messages = []
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
                    text_parts = []
                    for part in content:
                        if part.get("type") == "input_text":
                            text_parts.append(part.get("text", ""))
                    if text_parts:
                        messages.append({"role": role, "content": "\n".join(text_parts)})

            elif item_type == "function_call":
                raw_args = item.get("arguments", "{}")
                if not isinstance(raw_args, str):
                    raw_args = json.dumps(raw_args)
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": item.get("call_id", item.get("id", "toolu_" + secrets.token_hex(12))),
                        "name": item.get("name", ""),
                        "input": json.loads(raw_args) if isinstance(raw_args, str) else {},
                    }],
                })

            elif item_type == "function_call_output":
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": item.get("call_id", ""),
                        "content": item.get("output", "") if isinstance(item.get("output"), str) else json.dumps(item.get("output", "")),
                    }],
                })

    req: dict = {"messages": messages}
    if system_text:
        req["system"] = system_text
    if body.get("max_output_tokens"):
        req["max_tokens"] = body["max_output_tokens"]
    else:
        # Anthropic requires max_tokens — use a generous default
        req["max_tokens"] = 16384
    for key in ("temperature", "top_p", "stream"):
        if key in body:
            req[key] = body[key]

    # Tools
    resp_tools = body.get("tools", [])
    if resp_tools:
        anthropic_tools = []
        for t in resp_tools:
            if t.get("type") == "function":
                anthropic_tools.append({
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {}),
                })
        if anthropic_tools:
            req["tools"] = anthropic_tools

    # Tool choice
    tc = body.get("tool_choice")
    if tc:
        if isinstance(tc, str):
            if tc == "required":
                req["tool_choice"] = {"type": "any"}
            elif tc == "none":
                pass  # Anthropic doesn't have a direct "none" — just omit
            else:
                req["tool_choice"] = {"type": tc}
        elif isinstance(tc, dict):
            tc_type = tc.get("type", "")
            if tc_type == "function":
                req["tool_choice"] = {"type": "tool", "name": tc.get("name", "")}
            elif tc_type in ("auto", "required"):
                req["tool_choice"] = {"type": "any" if tc_type == "required" else "auto"}

    return req


def _anthropic_messages_to_responses(resp: dict, model: str) -> dict:
    """Convert Anthropic Messages response to OpenAI Responses API format."""
    now = int(__import__("time").time())
    resp_id = "resp_" + secrets.token_hex(16)
    content = resp.get("content", [])

    output_items = []
    for block in content:
        block_type = block.get("type", "")
        if block_type == "text":
            output_items.append({
                "type": "message",
                "id": resp.get("id", "msg_" + secrets.token_hex(16)),
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": block.get("text", ""), "annotations": []}],
            })
        elif block_type == "tool_use":
            output_items.append({
                "type": "function_call",
                "id": block.get("id", "fc_" + secrets.token_hex(12)),
                "call_id": block.get("id", "call_" + secrets.token_hex(12)),
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {})),
                "status": "completed",
            })

    if not output_items:
        output_items.append({
            "type": "message",
            "id": "msg_" + secrets.token_hex(16),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        })

    usage = resp.get("usage", {})

    return {
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
            "input_tokens": usage.get("input_tokens", 0),
            "input_tokens_details": {"cached_tokens": usage.get("cache_read_input_tokens", 0)},
            "output_tokens": usage.get("output_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
        "user": None,
        "metadata": {},
    }


async def _handle_responses_anthropic_stream(endpoint: str, messages_req: dict, model: str, headers: dict):
    """Handle streaming Responses API for Anthropic models.

    Collects Anthropic's SSE stream, then emits it as Responses API events.
    """
    client = httpx.AsyncClient(timeout=300)
    try:
        resp = await client.send(
            client.build_request("POST", endpoint, json=messages_req, headers=headers),
            stream=True,
        )
    except httpx.ConnectError:
        await client.aclose()
        return _error(502, "api_error", "Cannot connect to Anthropic API")
    except Exception as e:
        await client.aclose()
        return _error(502, "api_error", f"Anthropic error: {e}")

    if resp.status_code != 200:
        err_body = await resp.aread()
        await resp.aclose()
        await client.aclose()
        return _error(502, "api_error", f"Anthropic returned {resp.status_code}: {err_body.decode()[:500]}")

    # Collect the full Anthropic response from the SSE stream
    try:
        anthropic_resp = await _collect_anthropic_stream(resp)
    except Exception as e:
        return _error(502, "api_error", f"Anthropic stream collection failed: {e}")
    finally:
        await resp.aclose()
        await client.aclose()

    result = _anthropic_messages_to_responses(anthropic_resp, model)

    # Emit as Responses API SSE events
    async def event_generator():
        yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': result})}\n\n".encode()
        yield f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': result})}\n\n".encode()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _collect_anthropic_stream(resp: httpx.Response) -> dict:
    """Consume an Anthropic SSE stream and reassemble a complete Messages response."""
    msg_data: dict = {}
    content_blocks: dict[int, dict] = {}  # index -> block
    usage: dict = {}

    buffer = ""
    async for chunk in resp.aiter_bytes():
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")

            if event_type == "message_start":
                msg_data = data.get("message", {})
            elif event_type == "content_block_start":
                block = data.get("content_block", {})
                idx = data.get("index", len(content_blocks))
                content_blocks[idx] = block
            elif event_type == "content_block_delta":
                idx = data.get("index", 0)
                delta = data.get("delta", {})
                if idx in content_blocks:
                    if delta.get("type") == "text_delta":
                        content_blocks[idx]["text"] = content_blocks[idx].get("text", "") + delta.get("text", "")
                    elif delta.get("type") == "thinking_delta":
                        content_blocks[idx]["thinking"] = content_blocks[idx].get("thinking", "") + delta.get("thinking", "")
                    elif delta.get("type") == "input_json_delta":
                        existing = content_blocks[idx].get("_partial_input", "")
                        content_blocks[idx]["_partial_input"] = existing + delta.get("partial_json", "")
            elif event_type == "message_delta":
                delta = data.get("delta", {})
                if "stop_reason" in delta:
                    msg_data["stop_reason"] = delta["stop_reason"]
                usage_update = data.get("usage", {})
                if usage_update:
                    usage.update(usage_update)

    # Assemble content blocks, parsing any accumulated JSON input
    content = []
    for idx in sorted(content_blocks):
        block = content_blocks[idx]
        block_type = block.get("type", "")
        if block_type == "text":
            content.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "thinking":
            content.append({"type": "thinking", "thinking": block.get("thinking", "")})
        elif block_type == "tool_use":
            raw_input = block.get("_partial_input", "{}")
            try:
                parsed_input = json.loads(raw_input)
            except json.JSONDecodeError:
                parsed_input = {}
            content.append({
                "type": "tool_use",
                "id": block.get("id", "toolu_" + secrets.token_hex(12)),
                "name": block.get("name", ""),
                "input": parsed_input,
            })

    msg_data["content"] = content
    if usage:
        msg_data["usage"] = usage

    return msg_data


async def _handle_sync(
    endpoint: str, openai_req: dict, model: str, headers: dict, has_tools: bool, thinking_enabled: bool,
) -> JSONResponse:
    """Handle non-streaming Anthropic request.

    Fireworks requires stream=true for max_tokens > 4096, so we always stream
    from the provider and reassemble into a single response.
    """
    openai_req["stream"] = True
    openai_req["stream_options"] = {"include_usage": True}

    client = httpx.AsyncClient(timeout=300)
    try:
        resp = await client.send(
            client.build_request("POST", endpoint, json=openai_req, headers=headers),
            stream=True,
        )
    except httpx.ConnectError:
        await client.aclose()
        return _error(502, "api_error", "Cannot connect to cloud provider")
    except Exception as e:
        await client.aclose()
        return _error(502, "api_error", f"Provider error: {e}")

    if resp.status_code != 200:
        err_body = await resp.aread()
        await resp.aclose()
        await client.aclose()
        return _error(502, "api_error", f"Provider returned {resp.status_code}: {err_body.decode()[:500]}")

    # Consume SSE stream and reassemble into a single OpenAI-shaped response
    try:
        openai_resp = await _collect_stream(resp)
    except Exception as e:
        return _error(502, "api_error", f"Stream collection failed: {e}")
    finally:
        await resp.aclose()
        await client.aclose()

    result = openai_to_anthropic(openai_resp, model, has_tools=has_tools, thinking_enabled=thinking_enabled)

    # Log cache hit rate
    usage = openai_resp.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if prompt_tokens and cached:
        log.info("Cache hit: %d/%d prompt tokens cached (%.0f%%)", cached, prompt_tokens, cached / prompt_tokens * 100)

    return JSONResponse(content=result)


async def _handle_streaming(
    endpoint: str, openai_req: dict, model: str, headers: dict, has_tools: bool, thinking_enabled: bool,
):
    openai_req["stream"] = True

    client = httpx.AsyncClient(timeout=300)
    try:
        resp = await client.send(
            client.build_request(
                "POST",
                endpoint,
                json=openai_req,
                headers=headers,
            ),
            stream=True,
        )
    except httpx.ConnectError:
        await client.aclose()
        return _error(502, "api_error", "Cannot connect to cloud provider")
    except Exception as e:
        await client.aclose()
        return _error(502, "api_error", f"Provider error: {e}")

    if resp.status_code != 200:
        err_body = await resp.aread()
        await resp.aclose()
        await client.aclose()
        return _error(502, "api_error", f"Provider returned {resp.status_code}: {err_body.decode()[:500]}")

    async def event_generator():
        try:
            async for event in translate_stream(
                resp.aiter_bytes(), model, has_tools=has_tools, thinking_enabled=thinking_enabled,
            ):
                yield event
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _handle_streaming_google(
    endpoint: str, openai_req: dict, model: str, headers: dict, has_tools: bool, thinking_enabled: bool,
):
    """Handle streaming for Google+tools: use non-streaming upstream to guarantee
    thought_signature capture, then generate Anthropic SSE events from the response."""
    # Force non-streaming upstream to capture extra_content reliably
    openai_req.pop("stream", None)

    async with httpx.AsyncClient(timeout=300) as client:
        try:
            resp = await client.post(endpoint, json=openai_req, headers=headers)
        except httpx.ConnectError:
            return _error(502, "api_error", "Cannot connect to Google API")
        except Exception as e:
            return _error(502, "api_error", f"Google error: {e}")

    if resp.status_code != 200:
        return _error(502, "api_error", f"Google returned {resp.status_code}: {resp.text[:500]}")

    openai_resp = resp.json()

    # Capture and cache thought_signatures from all tool_calls
    for tc in (openai_resp.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []):
        ec = tc.get("extra_content")
        if ec:
            tc_id = tc.get("id", "")
            store_from_extra_content(tc_id, ec)
            ts = (ec.get("google") or {}).get("thought_signature")
            if ts:
                log.info("signature_cache: captured thought_signature from non-streaming response for %s", tc_id)

    # Convert to Anthropic format
    anthropic_msg = openai_to_anthropic(openai_resp, model, has_tools=has_tools, thinking_enabled=thinking_enabled)

    # Log cache hit rate
    usage = openai_resp.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if prompt_tokens and cached:
        log.info("Cache hit: %d/%d prompt tokens cached (%.0f%%)", cached, prompt_tokens, cached / prompt_tokens * 100)

    # Generate Anthropic SSE events from the complete response
    msg_id = anthropic_msg["id"]
    content = anthropic_msg.get("content", [])
    stop_reason = anthropic_msg.get("stop_reason", "end_turn")
    input_tokens = anthropic_msg.get("usage", {}).get("input_tokens", 0)
    output_tokens = anthropic_msg.get("usage", {}).get("output_tokens", 0)
    cached_tokens = anthropic_msg.get("usage", {}).get("cache_read_input_tokens", 0)

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def generate_sse():
        # message_start
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

        # Content blocks
        for idx, block in enumerate(content):
            block_type = block.get("type", "text")

            # content_block_start
            if block_type == "text":
                start_block = {"type": "text", "text": ""}
            elif block_type == "thinking":
                start_block = {"type": "thinking", "thinking": ""}
            elif block_type == "tool_use":
                start_block = {"type": "tool_use", "id": block["id"], "name": block["name"]}
                if block.get("thought_signature"):
                    start_block["thought_signature"] = block["thought_signature"]
            else:
                start_block = dict(block)

            yield _sse("content_block_start", {
                "type": "content_block_start",
                "index": idx,
                "content_block": start_block,
            })

            # content_block_delta — send full content as a single delta
            if block_type == "text":
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": block.get("text", "")},
                })
            elif block_type == "thinking":
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "thinking_delta", "thinking": block.get("thinking", "")},
                })
            elif block_type == "tool_use":
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))},
                })

            # content_block_stop
            yield _sse("content_block_stop", {
                "type": "content_block_stop",
                "index": idx,
            })

        # message_delta
        usage_out = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        if cached_tokens:
            usage_out["cache_read_input_tokens"] = cached_tokens
        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": usage_out,
        })

        # message_stop
        yield _sse("message_stop", {"type": "message_stop"})

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
