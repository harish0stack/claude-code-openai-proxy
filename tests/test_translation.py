"""Tests for the Anthropic → OpenAI translation logic."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.proxy import (
    anthropic_content_to_openai,
    anthropic_tools_to_openai,
    build_openai_messages,
    build_openai_payload,
    openai_to_anthropic_response,
    resolve_model,
)


# ── resolve_model ─────────────────────────────────────────────────────────────

def test_resolve_model_opus_maps_to_big():
    assert resolve_model("claude-opus-4-6") == "gpt-4o"

def test_resolve_model_haiku_maps_to_small():
    assert resolve_model("claude-haiku-4-5") == "gpt-4o-mini"

def test_resolve_model_sonnet_maps_to_big():
    assert resolve_model("claude-sonnet-4-6") == "gpt-4o"


# ── anthropic_content_to_openai ───────────────────────────────────────────────

def test_string_content_passthrough():
    assert anthropic_content_to_openai("hello") == "hello"

def test_text_block_list_collapses():
    result = anthropic_content_to_openai([
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ])
    assert result == "hello world"

def test_image_block_becomes_image_url():
    result = anthropic_content_to_openai([
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}},
    ])
    assert isinstance(result, list)
    assert result[0]["type"] == "image_url"
    assert "data:image/png;base64,abc123" in result[0]["image_url"]["url"]


# ── build_openai_messages ─────────────────────────────────────────────────────

def test_system_prompt_becomes_system_message():
    body = {
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "hi"}],
    }
    msgs = build_openai_messages(body)
    assert msgs[0]["role"] == "system"
    assert "helpful" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"

def test_tool_use_in_assistant_content():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tc_1", "name": "get_weather", "input": {"city": "Delhi"}},
                ],
            }
        ]
    }
    msgs = build_openai_messages(body)
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    args = json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"])
    assert args["city"] == "Delhi"

def test_tool_result_becomes_tool_message():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tc_1", "content": "Sunny, 35°C"},
                ],
            }
        ]
    }
    msgs = build_openai_messages(body)
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "tc_1"
    assert "Sunny" in msgs[0]["content"]


# ── anthropic_tools_to_openai ─────────────────────────────────────────────────

def test_tools_conversion():
    tools = [
        {
            "name": "search",
            "description": "Search the web",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    result = anthropic_tools_to_openai(tools)
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "search"
    assert "query" in result[0]["function"]["parameters"]["properties"]


# ── build_openai_payload ──────────────────────────────────────────────────────

def test_payload_includes_model_and_messages():
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload = build_openai_payload(body)
    assert payload["model"] == "gpt-4o"
    assert payload["max_tokens"] == 1024
    assert any(m["role"] == "user" for m in payload["messages"])

def test_stream_flag_propagated():
    body = {
        "model": "claude-haiku-4-5",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload = build_openai_payload(body)
    assert payload["stream"] is True
    assert payload["model"] == "gpt-4o-mini"


# ── openai_to_anthropic_response ──────────────────────────────────────────────

def test_response_conversion_basic():
    oai = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello!", "tool_calls": None},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    result = openai_to_anthropic_response(oai, "claude-sonnet-4-6")
    assert result["type"] == "message"
    assert result["role"] == "assistant"
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "Hello!"
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] == 10

def test_response_tool_call_conversion():
    oai = {
        "id": "chatcmpl-456",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Mumbai"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15},
    }
    result = openai_to_anthropic_response(oai, "claude-sonnet-4-6")
    tool_block = next(b for b in result["content"] if b["type"] == "tool_use")
    assert tool_block["name"] == "get_weather"
    assert tool_block["input"]["city"] == "Mumbai"
    assert result["stop_reason"] == "tool_use"
