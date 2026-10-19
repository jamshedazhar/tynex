"""Anthropic prompt caching: cache_control breakpoints in request bodies (offline)."""

import json

import httpx2

from tynex import http
from tynex.ai.models import Model
from tynex.ai.providers.anthropic import AnthropicProvider, add_cache_control
from tynex.ai.providers.base import LlmContext, StreamOptions
from tynex.ai.types import SystemMessage, UserMessage


def make_model() -> Model:
    return Model(
        id="claude-sonnet-4-5",
        name="Claude Sonnet 4.5",
        provider="anthropic",
        api="anthropic-messages",
        context_window=200_000,
        max_tokens=8_192,
    )


def install_transport(captured: dict) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
            {"type": "message_stop"},
        ]
        body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    http.set_client(httpx2.Client(transport=httpx2.MockTransport(handler)))


def count_breakpoints(body: dict) -> int:
    total = 0
    for block in body.get("system") or []:
        total += "cache_control" in block
    for tool in body.get("tools") or []:
        total += "cache_control" in tool
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            total += sum("cache_control" in block for block in content if isinstance(block, dict))
    return total


def test_anthropic_requests_carry_cache_breakpoints(monkeypatch):
    captured: dict = {}
    install_transport(captured)
    context = LlmContext(
        messages=[
            SystemMessage(content="You are tynex, a coding agent."),
            UserMessage(content="first question"),
            UserMessage(content="second question"),
        ],
        tools=[
            {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {}}},
            {"name": "write", "description": "Write a file", "parameters": {"type": "object", "properties": {}}},
        ],
    )
    provider = AnthropicProvider()
    events = list(provider.stream(make_model(), context, StreamOptions(api_key="test-key")))
    assert any(event.get("type") == "done" for event in events)

    body = captured["body"]
    # System prompt and the tool block prefix are cached
    assert body["system"] == [
        {"type": "text", "text": "You are tynex, a coding agent.", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in body["tools"][0]
    # The newest message ends with a breakpoint (incremental caching)
    assert body["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # Earlier messages stay clean and the total stays within Anthropic's four
    for message in body["messages"][:-1]:
        assert all("cache_control" not in block for block in message["content"])
    assert count_breakpoints(body) == 3


def test_add_cache_control_is_conservative():
    body: dict = {"messages": [{"role": "user", "content": "plain string"}]}
    add_cache_control(body)
    assert body["messages"][0]["content"] == "plain string"
    assert "system" not in body and "tools" not in body

    body = {
        "system": "",
        "tools": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "x"}]},
            {"role": "assistant", "content": []},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": []}]},
        ],
    }
    add_cache_control(body)
    assert body["messages"][0]["content"][0].get("cache_control") is None
    assert body["messages"][1]["content"] == []
    assert body["messages"][2]["content"][-1]["cache_control"] == {"type": "ephemeral"}
