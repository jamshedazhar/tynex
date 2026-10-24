"""OpenAI Chat Completions provider."""

import json
from collections.abc import Callable, Iterator
from typing import Any

from tynex.ai.models import Model, provider_base_url
from tynex.ai.providers.base import (
    Chunk,
    LlmContext,
    Provider,
    StreamOptions,
    clamp_thinking,
    finish_chunks,
    raise_for_status,
    tool_declarations,
)
from tynex.ai.types import ImageContent, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage
from tynex.http import iter_json_events, stream

# Thinking levels clamped to the OpenAI ``reasoning_effort`` values every
# reasoning model accepts ("minimal"/"xhigh"/"max" are sent as their nearest
# broadly-supported neighbour).
REASONING_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}

# Expands a raw SSE event into one or more OpenAI-shaped events.
EventNormalizer = Callable[[dict[str, Any]], list[dict[str, Any]]]


class OpenAICompatProvider(Provider):
    name = "openai"

    # -- message conversion ---------------------------------------------------
    def _convert_messages(self, context: LlmContext) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        system = context.system_text()
        if system:
            out.append({"role": "system", "content": system})
        for message in context.conversation():
            role = getattr(message, "role", None)
            if role == "user":
                assert isinstance(message, UserMessage)
                content = message.content
                if isinstance(content, str):
                    out.append({"role": "user", "content": content})
                else:
                    parts: list[dict[str, Any]] = []
                    for part in content:
                        if isinstance(part, TextContent):
                            parts.append({"type": "text", "text": part.text})
                        elif isinstance(part, ImageContent):
                            parts.append(
                                {"type": "image_url", "image_url": {"url": f"data:{part.mime_type};base64,{part.data}"}}
                            )
                    out.append({"role": "user", "content": parts})
            elif role == "assistant":
                text = "".join(p.text for p in message.content if isinstance(p, TextContent))
                tool_calls = [
                    {"id": p.id, "type": "function", "function": {"name": p.name, "arguments": json.dumps(p.arguments)}}
                    for p in message.content
                    if isinstance(p, ToolCall)
                ]
                entry: dict[str, Any] = {"role": "assistant", "content": text or None}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                if text or tool_calls:
                    out.append(entry)
            elif role == "toolResult":
                assert isinstance(message, ToolResultMessage)
                out.append(
                    {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.text() or "(no output)"}
                )
        return out

    def _convert_tools(self, context: LlmContext) -> list[dict[str, Any]]:
        return [{"type": "function", "function": tool} for tool in tool_declarations(context)]

    # -- request shape --------------------------------------------------------
    def _uses_max_completion_tokens(self, model: Model) -> bool:
        return model.provider == "openai"

    def _allows_temperature(self, model: Model) -> bool:
        return model.provider != "openai"

    def _supports_reasoning_effort(self, model: Model) -> bool:
        return model.provider == "openai"

    def _reasoning_effort(self, model: Model, thinking: str | None) -> str | None:
        """The ``reasoning_effort`` value for this model, if it takes one."""
        if model.thinking_levels:
            return "none" if not thinking or thinking == "off" else clamp_thinking(thinking, model.thinking_levels)
        if not model.reasoning or not self._supports_reasoning_effort(model):
            return None
        return REASONING_EFFORT.get(thinking or "")

    def _chat_body(self, model: Model, context: LlmContext, options: StreamOptions) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model.id, "messages": self._convert_messages(context), "stream": True}
        tools = self._convert_tools(context)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self._uses_max_completion_tokens(model):
            body["max_completion_tokens"] = options.max_tokens or model.max_tokens
        else:
            body["max_tokens"] = options.max_tokens or model.max_tokens
        # Usage in the final stream chunk (supported by OpenAI and most gateways).
        body["stream_options"] = {"include_usage": True}
        if options.temperature is not None and self._allows_temperature(model):
            body["temperature"] = options.temperature
        effort = self._reasoning_effort(model, options.thinking)
        if effort:
            body["reasoning_effort"] = effort
        return body

    def _chat_headers(self, model: Model, api_key: str | None) -> dict[str, str]:
        headers = {"content-type": "application/json", "accept": "text/event-stream"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        return headers

    # -- streaming ------------------------------------------------------------
    def _stream_chat(
        self,
        model: Model,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        *,
        normalize: EventNormalizer | None = None,
    ) -> Iterator[Chunk]:
        usage = Usage()
        finish_reason: str | None = None
        tool_names: dict[int, str] = {}
        tool_ids: dict[int, str] = {}
        saw_tool_calls = False
        with stream("POST", url, json_body=body, headers=headers) as response:
            raise_for_status(response)
            for raw in iter_json_events(response.iter_lines()):
                for event in normalize(raw) if normalize is not None else [raw]:
                    if event.get("error"):
                        err = event["error"]
                        message = err.get("message") if isinstance(err, dict) else str(err)
                        yield Chunk(kind="error", error=message)
                        return
                    if event.get("usage"):
                        u = event["usage"]
                        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
                        usage.input = max(0, u.get("prompt_tokens", 0) - cached)
                        usage.output = u.get("completion_tokens", 0)
                        usage.cache_read = cached
                        usage.total_tokens = usage.input + usage.output
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            yield Chunk(kind="text", text=delta["content"])
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning:
                            yield Chunk(kind="thinking", text=reasoning)
                        for tc in delta.get("tool_calls") or []:
                            index = tc.get("index", 0)
                            fn = tc.get("function") or {}
                            is_new = index not in tool_ids or (fn.get("name") and not tool_names.get(index))
                            if is_new:
                                saw_tool_calls = True
                                tool_ids[index] = tc.get("id") or tool_ids.get(index) or f"call_{index}"
                                tool_names[index] = fn.get("name") or tool_names.get(index) or ""
                                yield Chunk(kind="toolcall_start", tool_id=tool_ids[index], tool_name=tool_names[index])
                            if fn.get("arguments"):
                                yield Chunk(kind="toolcall_delta", json_fragment=fn["arguments"])
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
        stop_map = {"stop": "stop", "length": "length", "tool_calls": "toolUse", "content_filter": "stop"}
        yield from finish_chunks(model, usage, finish_reason, stop_map, saw_tool_calls)

    def _chunks(self, model: Model, context: LlmContext, options: StreamOptions) -> Iterator[Chunk]:
        api_key = self.resolve_api_key(model, options)
        provider = model.provider
        base_url = (options.base_url or model.base_url or provider_base_url(provider) or "").rstrip("/")
        if (
            not api_key
            and provider not in ("ollama", "lmstudio")
            and "localhost" not in base_url
            and "127.0.0.1" not in base_url
        ):
            raise RuntimeError(f"No API key for {provider}. Set the provider API key env var or run /login in tynex.")
        yield from self._stream_chat(
            model,
            f"{base_url}/chat/completions",
            self._chat_body(model, context, options),
            self._chat_headers(model, api_key),
        )
