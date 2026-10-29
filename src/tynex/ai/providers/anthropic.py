"""Anthropic Messages API provider (streaming)."""

import json
from collections.abc import Iterator
from typing import Any

from tynex.ai.models import Model, provider_base_url
from tynex.ai.providers.base import (
    Chunk,
    LlmContext,
    Provider,
    StreamOptions,
    finish_chunks,
    raise_for_status,
    thinking_budget,
    tool_declarations,
)
from tynex.ai.types import ImageContent, TextContent, ThinkingContent, ToolCall, ToolResultMessage, Usage
from tynex.http import iter_json_events, stream

STEALTH_TOOL_NAMES = {
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "bash": "Bash",
    "grep": "Grep",
    "find": "Find",
    "ls": "LS",
    "websearch": "WebSearch",
}
STEALTH_TOOL_NAMES_REVERSE = {v.lower(): k for k, v in STEALTH_TOOL_NAMES.items()}
CLAUDE_CODE_VERSION = "2.1.2"


def _map_tool_name(name: str, stealth: bool) -> str:
    if not stealth:
        return name
    return STEALTH_TOOL_NAMES.get(name, name)


def _unmap_tool_name(name: str, stealth: bool) -> str:
    if not stealth:
        return name
    return STEALTH_TOOL_NAMES_REVERSE.get(name.lower(), name)


CACHEABLE_BLOCK_TYPES = ("text", "image", "tool_result", "tool_use", "document")


def add_cache_control(body: dict[str, Any]) -> None:
    """Mark cache breakpoints (system, tools, last message) for prompt caching.

    Anthropic caches the longest matching prefix of previous requests, so a
    breakpoint after the stable system+tools prefix plus one at the end of the
    conversation covers both the fixed harness prompt and incremental turns.
    Up to four breakpoints are allowed; three are used here.
    """
    system = body.get("system")
    if isinstance(system, str) and system:
        body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    tools = body.get("tools")
    if isinstance(tools, list) and tools and isinstance(tools[-1], dict):
        tools[-1]["cache_control"] = {"type": "ephemeral"}
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        content = messages[-1].get("content")
        if (
            isinstance(content, list)
            and content
            and isinstance(content[-1], dict)
            and content[-1].get("type") in CACHEABLE_BLOCK_TYPES
        ):
            content[-1]["cache_control"] = {"type": "ephemeral"}


class AnthropicProvider(Provider):
    name = "anthropic"

    # -- message conversion -------------------------------------------------
    def _convert_messages(self, context: LlmContext, stealth: bool) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for message in context.conversation():
            role = getattr(message, "role", None)
            if role == "user":
                blocks: list[dict[str, Any]] = []
                content = message.content
                if isinstance(content, str):
                    if content:
                        blocks.append({"type": "text", "text": content})
                else:
                    for part in content:
                        if isinstance(part, TextContent):
                            blocks.append({"type": "text", "text": part.text})
                        elif isinstance(part, ImageContent):
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": part.mime_type, "data": part.data},
                                }
                            )
                if blocks:
                    out.append({"role": "user", "content": blocks})
            elif role == "assistant":
                blocks = []
                for part in message.content:
                    if isinstance(part, TextContent) and part.text:
                        blocks.append({"type": "text", "text": part.text})
                    elif isinstance(part, ThinkingContent):
                        if part.redacted:
                            blocks.append({"type": "redacted_thinking", "data": part.data or ""})
                        elif part.signature:
                            blocks.append({"type": "thinking", "thinking": part.thinking, "signature": part.signature})
                    elif isinstance(part, ToolCall):
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": part.id,
                                "name": _map_tool_name(part.name, stealth),
                                "input": part.arguments,
                            }
                        )
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
            elif role == "toolResult":
                assert isinstance(message, ToolResultMessage)
                result_blocks: list[dict[str, Any]] = []
                for part in message.content:
                    if isinstance(part, TextContent):
                        result_blocks.append({"type": "text", "text": part.text})
                if not result_blocks:
                    result_blocks.append({"type": "text", "text": "(no output)"})
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": result_blocks,
                                "is_error": message.is_error,
                            }
                        ],
                    }
                )
        return out

    def _convert_tools(self, context: LlmContext, stealth: bool) -> list[dict[str, Any]]:
        declarations = tool_declarations(context, lambda name: _map_tool_name(name, stealth))
        return [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in declarations
        ]

    # -- streaming ------------------------------------------------------------
    def _chunks(self, model: Model, context: LlmContext, options: StreamOptions) -> Iterator[Chunk]:
        api_key = self.resolve_api_key(model, options)
        if not api_key:
            raise RuntimeError("No Anthropic API key. Set ANTHROPIC_API_KEY or run /login in tynex.")
        base_url = (options.base_url or model.base_url or provider_base_url("anthropic") or "").rstrip("/")
        url = f"{base_url}/messages" if base_url.endswith("/v1") else f"{base_url}/v1/messages"
        max_tokens = options.max_tokens or model.max_tokens
        body: dict[str, Any] = {
            "model": model.id,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": self._convert_messages(context, options.stealth),
        }
        system = context.system_text()
        if system:
            body["system"] = system
        tools = self._convert_tools(context, options.stealth)
        if tools:
            body["tools"] = tools
        add_cache_control(body)
        budget = thinking_budget(options.thinking)
        if budget and model.reasoning:
            budget = min(budget, max(1024, max_tokens - 1024))
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            body["max_tokens"] = max(max_tokens, budget + 4096)
        elif options.temperature is not None:
            body["temperature"] = options.temperature
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        if options.stealth:
            headers["user-agent"] = f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)"
            headers["x-app"] = "cli"
        usage = Usage()
        stop_reason: str | None = None
        saw_tool_use = False
        with stream("POST", url, json_body=body, headers=headers) as response:
            raise_for_status(response)
            for event in iter_json_events(response.iter_lines()):
                etype = event.get("type")
                if etype == "message_start":
                    u = (event.get("message") or {}).get("usage") or {}
                    usage.input = u.get("input_tokens", 0)
                    usage.cache_read = u.get("cache_read_input_tokens", 0) or 0
                    usage.cache_write = u.get("cache_creation_input_tokens", 0) or 0
                elif etype == "content_block_start":
                    block = event.get("content_block") or {}
                    btype = block.get("type")
                    if btype == "tool_use":
                        saw_tool_use = True
                        yield Chunk(
                            kind="toolcall_start",
                            tool_id=block.get("id"),
                            tool_name=_unmap_tool_name(block.get("name", ""), options.stealth),
                        )
                        initial = block.get("input")
                        if initial:
                            yield Chunk(kind="toolcall_delta", json_fragment=json.dumps(initial))
                    elif btype == "redacted_thinking":
                        yield Chunk(kind="thinking")
                        yield Chunk(kind="thinking_signature")
                elif etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    dtype = delta.get("type")
                    if dtype == "text_delta":
                        yield Chunk(kind="text", text=delta.get("text", ""))
                    elif dtype == "thinking_delta":
                        yield Chunk(kind="thinking", text=delta.get("thinking", ""))
                    elif dtype == "signature_delta":
                        yield Chunk(kind="thinking_signature", signature=delta.get("signature", ""))
                    elif dtype == "input_json_delta":
                        yield Chunk(kind="toolcall_delta", json_fragment=delta.get("partial_json", ""))
                elif etype == "message_delta":
                    d = event.get("delta") or {}
                    if d.get("stop_reason"):
                        stop_reason = d["stop_reason"]
                    u = event.get("usage") or {}
                    usage.output = u.get("output_tokens", usage.output)
                elif etype == "error":
                    err = (event.get("error") or {}).get("message", "Anthropic stream error")
                    yield Chunk(kind="error", error=err)
                    return
        stop_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "toolUse",
            "refusal": "stop",
        }
        yield from finish_chunks(model, usage, stop_reason, stop_map, saw_tool_use)
