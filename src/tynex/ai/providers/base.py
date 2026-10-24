"""Provider base classes and the shared streaming state machine."""

import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from tynex.ai.models import Model, provider_api_key_env
from tynex.ai.types import (
    AbortedError,
    AbortSignal,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
    message_text,
)
from tynex.http import Response, is_transport_error


class ProviderError(Exception):
    """Base class for provider errors."""


class ProviderHttpError(ProviderError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:2000]}")
        self.status = status
        self.body = body


def raise_for_status(response: Response) -> None:
    """Raise :class:`ProviderHttpError` carrying the response body for HTTP errors."""
    if response.status_code >= 400:
        body = response.read().decode("utf-8", errors="replace")
        raise ProviderHttpError(response.status_code, body)


@dataclass
class LlmContext:
    """What the model sees: system prompt + conversation + tool declarations."""

    messages: list[Any] = field(default_factory=list)  # tynex.ai.types.Message list
    tools: list[dict[str, Any]] = field(default_factory=list)  # {"name","description","parameters"}

    def system_text(self) -> str:
        parts = [m.content for m in self.messages if getattr(m, "role", None) == "system"]
        return "\n\n".join(p for p in parts if p)

    def conversation(self) -> list[Any]:
        return [m for m in self.messages if getattr(m, "role", None) != "system"]


@dataclass
class StreamOptions:
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    thinking: str = "off"  # off | minimal | low | medium | high | xhigh | max
    signal: AbortSignal | None = None
    stealth: bool = False  # Anthropic: rename tools to Claude Code names


@dataclass
class Chunk:
    """One low-level streaming delta in provider wire order."""

    kind: (
        str  # text | thinking | thinking_signature | signature | toolcall_start | toolcall_delta | usage | stop | error
    )
    text: str | None = None
    tool_id: str | None = None
    tool_name: str | None = None
    json_fragment: str | None = None
    usage: Usage | None = None
    stop_reason: str | None = None
    error: str | None = None
    signature: str | None = None


# Thinking levels accepted for ``--thinking`` and ``/thinking``.
THINKING_LEVELS: tuple[str, ...] = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def clamp_thinking(level: str, available: list[str] | tuple[str, ...]) -> str:
    """The closest level ``available`` offers to ``level`` (higher on a tie)."""
    levels = [candidate for candidate in available if candidate in THINKING_LEVELS]
    if not levels:
        return "off"
    if level in levels:
        return level
    if level not in THINKING_LEVELS:
        return levels[0]
    target = THINKING_LEVELS.index(level)
    return min(
        levels,
        key=lambda candidate: (abs(THINKING_LEVELS.index(candidate) - target), -THINKING_LEVELS.index(candidate)),
    )


def thinking_budget(level: str) -> int | None:
    """Anthropic/Google thinking budget per level (None = thinking off)."""
    return {"minimal": 1024, "low": 4096, "medium": 8192, "high": 16384, "xhigh": 24576, "max": 32000}.get(level)


def compute_cost(model: Model, usage: Usage) -> None:
    """Fill ``usage.cost`` from the model's per-million-token pricing."""
    cost = model.cost
    usage.cost = (
        usage.input * cost.input
        + usage.output * cost.output
        + usage.cache_read * cost.cache_read
        + usage.cache_write * cost.cache_write
    ) / 1_000_000.0


def tool_declarations(context: LlmContext, name_mapper: Callable[[str], str] | None = None) -> list[dict[str, Any]]:
    """Provider-neutral tool definitions (``name``/``description``/``parameters``)."""
    rename = name_mapper or (lambda name: name)
    return [
        {
            "name": rename(t["name"]),
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {"type": "object", "properties": {}}),
        }
        for t in context.tools
    ]


def finish_chunks(
    model: Model, usage: Usage, stop_reason: str | None, stop_map: dict[str, str], saw_tool: bool
) -> Iterator[Chunk]:
    """Yield the terminal usage/stop chunks every streaming provider ends with."""
    usage.total_tokens = usage.total_tokens or usage.input + usage.output
    compute_cost(model, usage)
    yield Chunk(kind="usage", usage=usage)
    yield Chunk(kind="stop", stop_reason=stop_map.get(stop_reason or "", "toolUse" if saw_tool else "stop"))


def _parse_tool_arguments(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        # Best-effort repair for truncated streaming JSON.
        repaired = _repair_json(raw)
        if repaired is not None:
            return repaired
        return {"__raw": raw}


def _repair_json(raw: str) -> dict[str, Any] | None:
    s = raw
    # Drop a trailing comma before a closing bracket.
    for _ in range(20):
        s2 = s.rstrip()
        if s2.endswith(","):
            s = s2[:-1]
            continue
        break
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")
    if s and not s.rstrip().endswith(("}", "]")):
        s = s.rstrip() + '"' if s.count('"') % 2 == 1 else s.rstrip()
    s += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class StreamState:
    """Accumulates chunks into a partial AssistantMessage and emits events."""

    def __init__(self, model: Model, thinking: str) -> None:
        self.model = model
        self.message = AssistantMessage(api=model.api, provider=model.provider, model=model.id, thinking_level=thinking)
        self._open_tool: ToolCall | None = None
        self._open_tool_raw: list[str] = []
        self._tin = False
        self._chunk_stop_reason: str | None = None

    # -- helpers -----------------------------------------------------------
    @property
    def partial(self) -> AssistantMessage:
        return self.message

    def _emit(self, etype: str, **kw: Any) -> dict[str, Any]:
        return {"type": etype, "partial": self.message, **kw}

    def start(self) -> dict[str, Any]:
        return self._emit("start")

    def apply(self, chunk: Chunk) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        m = self.message
        if chunk.kind == "text":
            idx = self._find_or_open(TextContent)
            block = m.content[idx]
            assert isinstance(block, TextContent)
            prefix = len(block.text)
            block.text += chunk.text or ""
            if chunk.signature:
                block.signature = chunk.signature
            etype = "text_start" if prefix == 0 else "text_delta"
            events.append(self._emit(etype, content_index=idx, delta=chunk.text))
            if etype == "text_start":
                events.append(self._emit("text_delta", content_index=idx, delta=chunk.text))
        elif chunk.kind == "thinking":
            if not self._tin:
                m.content.append(ThinkingContent(thinking=""))
                self._tin = True
            idx = len(m.content) - 1
            block = m.content[idx]
            assert isinstance(block, ThinkingContent)
            prefix = len(block.thinking)
            block.thinking += chunk.text or ""
            if chunk.signature:
                block.signature = chunk.signature
            etype = "thinking_start" if prefix == 0 else "thinking_delta"
            events.append(self._emit(etype, content_index=idx))
            if chunk.text:
                events.append(self._emit("thinking_delta", content_index=idx, delta=chunk.text))
        elif chunk.kind == "thinking_signature":
            if self._tin and isinstance(m.content[-1], ThinkingContent):
                m.content[-1].signature = chunk.signature
        elif chunk.kind == "signature":
            # A trailing signature-only part belongs to the response; attach it
            # to the last block that can carry it (Gemini 3 text responses).
            last = m.content[-1] if m.content else None
            if isinstance(last, (TextContent, ThinkingContent, ToolCall)) and last.signature is None:
                last.signature = chunk.signature
        elif chunk.kind == "toolcall_start":
            self._close_tool()
            self._tin = False
            call = ToolCall(
                id=chunk.tool_id or f"call_{len(m.content)}",
                name=chunk.tool_name or "",
                arguments={},
                signature=chunk.signature,
            )
            m.content.append(call)
            self._open_tool = call
            self._open_tool_raw = []
            events.append(self._emit("toolcall_start", content_index=len(m.content) - 1))
        elif chunk.kind == "toolcall_delta":
            if self._open_tool is not None:
                self._open_tool_raw.append(chunk.json_fragment or "")
                events.append(self._emit("toolcall_delta", content_index=m.content.index(self._open_tool)))
        elif chunk.kind == "usage":
            if chunk.usage:
                m.usage.input += chunk.usage.input
                m.usage.output += chunk.usage.output
                m.usage.cache_read += chunk.usage.cache_read
                m.usage.cache_write += chunk.usage.cache_write
                m.usage.total_tokens = m.usage.input + m.usage.output
                m.usage.cost += chunk.usage.cost
        elif chunk.kind == "stop":
            self._chunk_stop_reason = chunk.stop_reason
        elif chunk.kind == "error":
            m.stop_reason = "error"
            m.error_message = chunk.error
        return events

    def _find_or_open(self, cls: type) -> int:
        m = self.message
        if m.content and isinstance(m.content[-1], cls):
            return len(m.content) - 1
        # A new text block starts unless the last block is a same-kind block.
        self._tin = False
        m.content.append(cls(text="") if cls is TextContent else cls(thinking=""))
        return len(m.content) - 1

    def _close_tool(self) -> None:
        if self._open_tool is not None:
            self._open_tool.arguments = _parse_tool_arguments("".join(self._open_tool_raw))
            self._open_tool = None
            self._open_tool_raw = []

    def finish(self, stop_reason: str | None, error: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Close open blocks; return the terminal event and any close events."""
        events: list[dict[str, Any]] = []
        m = self.message
        if m.content and isinstance(m.content[-1], TextContent):
            events.append(self._emit("text_end", content_index=len(m.content) - 1))
        if self._tin and m.content and isinstance(m.content[-1], ThinkingContent):
            events.append(self._emit("thinking_end", content_index=len(m.content) - 1))
        if self._open_tool is not None:
            self._close_tool()
            events.append(self._emit("toolcall_end", content_index=len(m.content) - 1))
        self._tin = False
        if error is not None or m.stop_reason == "error":
            m.stop_reason = "error"
            m.error_message = error or m.error_message or "Unknown provider error"
            events.append(self._emit("error", error=m.error_message))
        else:
            if stop_reason is None:
                stop_reason = self._chunk_stop_reason
            calls = m.tool_calls()
            if stop_reason == "toolUse" or (stop_reason is None and calls):
                m.stop_reason = "toolUse"
            else:
                m.stop_reason = stop_reason or "stop"
            if not m.usage.total_tokens:
                m.usage.total_tokens = m.usage.input + m.usage.output
            events.append(self._emit("done", reason=m.stop_reason))
        return events[-1], events[:-1]


RETRYABLE_STATUSES = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_ATTEMPTS = 3


class Provider:
    """Base provider. Subclasses implement ``_chunks()``."""

    name: str = "base"

    def _chunks(self, model: Model, context: LlmContext, options: StreamOptions) -> Iterator[Chunk]:
        raise NotImplementedError

    def stream(self, model: Model, context: LlmContext, options: StreamOptions) -> Iterator[dict[str, Any]]:
        for attempt in range(MAX_ATTEMPTS):
            state = StreamState(model, options.thinking)
            started = False
            emitted_content = False

            def ensure_started(current: StreamState = state) -> Iterator[dict[str, Any]]:
                nonlocal started
                if not started:
                    started = True
                    yield current.start()

            try:
                for chunk in self._chunks(model, context, options):
                    if options.signal is not None and options.signal.aborted:
                        raise AbortedError("Aborted")
                    if not emitted_content:
                        yield from ensure_started()
                    yield from state.apply(chunk)
                    emitted_content = True
                yield from ensure_started()
                terminal, closers = state.finish(None)
                yield from closers
                yield terminal
                return
            except AbortedError:
                yield from ensure_started()
                terminal, closers = state.finish("aborted")
                yield from closers
                yield terminal
                return
            except Exception as exc:
                retryable = (
                    isinstance(exc, ProviderHttpError) and exc.status in RETRYABLE_STATUSES
                ) or is_transport_error(exc)
                if retryable and not emitted_content and attempt + 1 < MAX_ATTEMPTS:
                    time.sleep(min(2.0, 0.5 * (2**attempt)))
                    continue
                yield from ensure_started()
                terminal, closers = state.finish(None, error=str(exc))
                yield from closers
                yield terminal
                return

    def resolve_api_key(self, model: Model, options: StreamOptions) -> str | None:
        if options.api_key:
            return options.api_key
        for env_name in provider_api_key_env(model.provider):
            value = os.environ.get(env_name)
            if value:
                return value
        return None


# ---------------------------------------------------------------------------
# Small shared conversions
# ---------------------------------------------------------------------------


def last_user_text(context: LlmContext) -> str:
    for message in reversed(context.conversation()):
        if getattr(message, "role", None) == "user":
            return message_text(message)
    return ""
