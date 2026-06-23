"""
Anthropic → OpenAI-compatible proxy server.

Claude Code (or any Anthropic SDK client) sends requests here.
We translate them to the OpenAI chat-completions format and forward
to ANY OpenAI-compatible endpoint (OpenAI, NVIDIA NIM, Together,
Groq, Azure, Ollama, OpenRouter, …).

Run:
    uv run uvicorn src.proxy:app --host 0.0.0.0 --port 8082

Point Claude Code at it:
    export ANTHROPIC_BASE_URL=http://localhost:8082
    export ANTHROPIC_API_KEY=dummy   # proxy ignores this
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("proxy")

# ---------------------------------------------------------------------------
# Config (all from env-vars so the same binary works for any provider)
# ---------------------------------------------------------------------------
UPSTREAM_BASE_URL: str = os.environ.get(
    "UPSTREAM_BASE_URL", "https://api.openai.com"
).rstrip("/")

UPSTREAM_API_KEY: str | None = os.environ.get("UPSTREAM_API_KEY")

# If True, forward the x-api-key from the incoming request as Bearer token
UPSTREAM_API_KEY_PASSTHROUGH: bool = (
    os.environ.get("UPSTREAM_API_KEY_PASSTHROUGH", "false").lower() == "true"
)

# Model mapping: "claude-opus-4-6=gpt-4o;claude-haiku-4-5=gpt-4o-mini"
_RAW_MODEL_MAP = os.environ.get("MODEL_MAP", "")
MODEL_MAP: dict[str, str] = {}
if _RAW_MODEL_MAP:
    for pair in _RAW_MODEL_MAP.split(";"):
        if "=" in pair:
            src, dst = pair.split("=", 1)
            MODEL_MAP[src.strip()] = dst.strip()

# Fallback models when no explicit mapping exists
BIG_MODEL: str = os.environ.get("BIG_MODEL", "gpt-4o")
SMALL_MODEL: str = os.environ.get("SMALL_MODEL", "gpt-4o-mini")
REASONING_MODEL: str | None = os.environ.get("REASONING_MODEL")

# Patterns whose keys identify "big" claude models → BIG_MODEL
BIG_MODEL_PATTERNS = ("opus", "sonnet")
SMALL_MODEL_PATTERNS = ("haiku",)

# Proxy listen port (also used by the GH Actions workflow)
PORT: int = int(os.environ.get("PORT", "8082"))

# ---------------------------------------------------------------------------
# HTTP client (connection-pooled, async)
# ---------------------------------------------------------------------------
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _client


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Anthropic→OpenAI Proxy", version="1.0.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_model(claude_model: str) -> str:
    """Map a Claude model name to whatever upstream model we want."""
    if claude_model in MODEL_MAP:
        return MODEL_MAP[claude_model]
    name = claude_model.lower()
    if any(p in name for p in BIG_MODEL_PATTERNS):
        return BIG_MODEL
    if any(p in name for p in SMALL_MODEL_PATTERNS):
        return SMALL_MODEL
    # Check if extended thinking requested → use reasoning model if set
    return BIG_MODEL


def has_thinking(body: dict) -> bool:
    thinking = body.get("thinking") or {}
    return bool(thinking.get("type") == "enabled")


def anthropic_content_to_openai(content: Any) -> str | list[dict]:
    """
    Convert Anthropic `content` field to OpenAI `content`.

    Anthropic content can be:
      - str
      - list of blocks: {"type": "text"/"image"/"tool_result"/"tool_use", ...}
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content)

    parts: list[dict] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            src = block.get("source", {})
            if src.get("type") == "base64":
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{src['media_type']};base64,{src['data']}"
                    },
                })
            elif src.get("type") == "url":
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": src["url"]},
                })
        elif btype == "tool_result":
            # Flatten tool results as text for providers that don't support them
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                text = " ".join(
                    b.get("text", "") for b in tool_content if b.get("type") == "text"
                )
            else:
                text = str(tool_content)
            parts.append({"type": "text", "text": text})
        else:
            # tool_use blocks etc. – skip (handled separately in tool_calls)
            pass

    # If all parts are plain text, collapse to string for better compat
    if all(p["type"] == "text" for p in parts):
        return " ".join(p["text"] for p in parts)
    return parts if parts else ""


def anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


def build_openai_messages(body: dict) -> list[dict]:
    """
    Convert Anthropic messages (+ optional system prompt) to OpenAI messages list.
    """
    messages: list[dict] = []

    # System prompt
    system = body.get("system")
    if system:
        if isinstance(system, list):
            # List of system blocks
            sys_text = " ".join(
                b.get("text", "") for b in system if b.get("type") == "text"
            )
        else:
            sys_text = str(system)
        if sys_text.strip():
            messages.append({"role": "system", "content": sys_text})

    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        # Check for tool_use blocks inside assistant content
        if role == "assistant" and isinstance(content, list):
            tool_calls = []
            text_parts = []
            for block in content:
                if block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{block['name']}"),
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            oai_msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                oai_msg["content"] = " ".join(text_parts)
            else:
                oai_msg["content"] = None
            if tool_calls:
                oai_msg["tool_calls"] = tool_calls
            messages.append(oai_msg)

        # Tool result messages → tool role in OpenAI
        elif role == "user" and isinstance(content, list) and any(
            b.get("type") == "tool_result" for b in content
        ):
            for block in content:
                if block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        text = " ".join(
                            b.get("text", "")
                            for b in tool_content
                            if b.get("type") == "text"
                        )
                    else:
                        text = str(tool_content)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", "unknown"),
                        "content": text,
                    })
                else:
                    # Regular user content alongside tool results
                    converted = anthropic_content_to_openai([block])
                    if converted:
                        messages.append({"role": "user", "content": converted})
        else:
            messages.append({
                "role": role,
                "content": anthropic_content_to_openai(content),
            })

    return messages


def build_openai_payload(body: dict) -> dict:
    """Build the full OpenAI /v1/chat/completions payload."""
    model_in = body.get("model", "claude-sonnet-4-6")

    # If extended thinking is on, prefer reasoning model
    if has_thinking(body) and REASONING_MODEL:
        model_out = REASONING_MODEL
    else:
        model_out = resolve_model(model_in)

    payload: dict[str, Any] = {
        "model": model_out,
        "messages": build_openai_messages(body),
        "stream": body.get("stream", False),
        "max_tokens": body.get("max_tokens", 4096),
    }

    if "temperature" in body:
        payload["temperature"] = body["temperature"]
    if "top_p" in body:
        payload["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        payload["stop"] = body["stop_sequences"]

    tools = body.get("tools")
    if tools:
        payload["tools"] = anthropic_tools_to_openai(tools)
        tool_choice = body.get("tool_choice", {})
        tc_type = tool_choice.get("type", "auto")
        if tc_type == "any":
            payload["tool_choice"] = "required"
        elif tc_type == "tool":
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
        else:
            payload["tool_choice"] = "auto"

    return payload


def openai_to_anthropic_response(oai: dict, original_model: str) -> dict:
    """Convert a non-streaming OpenAI response to Anthropic /v1/messages format."""
    choice = oai["choices"][0]
    msg = choice["message"]

    content_blocks: list[dict] = []

    # Text content
    text = msg.get("content") or ""
    if text:
        content_blocks.append({"type": "text", "text": text})

    # Tool calls
    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        try:
            input_data = json.loads(fn["arguments"])
        except json.JSONDecodeError:
            input_data = {"raw": fn["arguments"]}
        content_blocks.append({
            "type": "tool_use",
            "id": tc["id"],
            "name": fn["name"],
            "input": input_data,
        })

    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    stop_reason = stop_reason_map.get(choice.get("finish_reason", "stop"), "end_turn")

    usage = oai.get("usage", {})
    return {
        "id": f"msg_{oai.get('id', 'unknown')}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": original_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def stream_openai_to_anthropic(
    response: httpx.Response,
    original_model: str,
    message_id: str,
) -> AsyncIterator[str]:
    """
    Convert OpenAI SSE stream → Anthropic SSE stream.

    Anthropic SSE events:
      message_start, content_block_start, ping,
      content_block_delta, content_block_stop, message_delta, message_stop
    """

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    # message_start
    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": original_model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield sse("ping", {"type": "ping"})

    # Track blocks
    text_block_started = False
    tool_blocks: dict[int, dict] = {}  # index → tool block state
    input_tokens = 0
    output_tokens = 0

    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw == "[DONE]":
            break
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            continue

        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Usage (some providers send it in the final chunk)
        if "usage" in chunk:
            u = chunk["usage"]
            input_tokens = u.get("prompt_tokens", input_tokens)
            output_tokens = u.get("completion_tokens", output_tokens)

        # Text delta
        text = delta.get("content")
        if text:
            if not text_block_started:
                yield sse("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
                text_block_started = True
            yield sse("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            })

        # Tool call deltas
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            block_idx = idx + 1  # offset from text block

            if idx not in tool_blocks:
                fn = tc.get("function", {})
                tool_blocks[idx] = {
                    "id": tc.get("id", f"call_{idx}"),
                    "name": fn.get("name", ""),
                    "args_buf": fn.get("arguments", ""),
                }
                yield sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_blocks[idx]["id"],
                        "name": tool_blocks[idx]["name"],
                        "input": {},
                    },
                })
                # Start input_json_delta
                if tool_blocks[idx]["args_buf"]:
                    yield sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tool_blocks[idx]["args_buf"],
                        },
                    })
            else:
                fn = tc.get("function", {})
                args_chunk = fn.get("arguments", "")
                if args_chunk:
                    tool_blocks[idx]["args_buf"] += args_chunk
                    yield sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": idx + 1,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": args_chunk,
                        },
                    })

        # Finish
        if finish_reason:
            stop_reason_map = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
            }
            stop_reason = stop_reason_map.get(finish_reason, "end_turn")

            if text_block_started:
                yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})

            for idx in tool_blocks:
                yield sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": idx + 1,
                })

            yield sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            })
            yield sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM_BASE_URL}


@app.get("/v1/models")
async def list_models():
    """Return a minimal models list so Claude Code doesn't complain."""
    return {
        "object": "list",
        "data": [
            {"id": "claude-opus-4-6", "object": "model", "created": 0, "owned_by": "proxy"},
            {"id": "claude-sonnet-4-6", "object": "model", "created": 0, "owned_by": "proxy"},
            {"id": "claude-haiku-4-5", "object": "model", "created": 0, "owned_by": "proxy"},
        ],
    }


@app.post("/v1/messages")
async def messages(request: Request):
    body: dict = await request.json()
    stream: bool = body.get("stream", False)

    # Resolve upstream API key
    if UPSTREAM_API_KEY_PASSTHROUGH:
        api_key = request.headers.get("x-api-key", "")
    elif UPSTREAM_API_KEY:
        api_key = UPSTREAM_API_KEY
    else:
        raise HTTPException(status_code=500, detail="No upstream API key configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = build_openai_payload(body)
    original_model = body.get("model", "claude-sonnet-4-6")
    message_id = f"msg_{int(time.time() * 1000)}"

    upstream_url = f"{UPSTREAM_BASE_URL}/v1/chat/completions"

    log.info(
        "→ %s  model=%s → %s  stream=%s  tools=%d",
        upstream_url,
        original_model,
        payload["model"],
        stream,
        len(payload.get("tools", [])),
    )

    client = get_client()

    if stream:
        async def generate():
            async with client.stream(
                "POST", upstream_url, json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    log.error("Upstream error %d: %s", resp.status_code, err)
                    yield f"data: {json.dumps({'error': err.decode()})}\n\n"
                    return
                async for chunk in stream_openai_to_anthropic(resp, original_model, message_id):
                    yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        resp = await client.post(upstream_url, json=payload, headers=headers)
        if resp.status_code != 200:
            log.error("Upstream error %d: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        oai_response = resp.json()
        anthropic_response = openai_to_anthropic_response(oai_response, original_model)
        return anthropic_response
