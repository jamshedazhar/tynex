"""Core message and content types."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Self


class AbortedError(Exception):
    """Raised when an operation is aborted."""

    def __init__(self, message: str = "Operation aborted") -> None:
        super().__init__(message)


class AbortSignal:
    """A threading.Event with a small convenience API."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason: str | None = None

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def abort(self, reason: str | None = None) -> None:
        self.reason = reason
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def throw_if_aborted(self) -> None:
        if self.aborted:
            raise AbortedError(self.reason or "Operation aborted")


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


@dataclass
class TextContent:
    text: str
    type: Literal["text"] = "text"
    # Provider round-trip signature (Gemini thought signatures).
    signature: str | None = None


@dataclass
class ImageContent:
    """Base64 encoded image, e.g. produced by the read tool."""

    data: str
    mime_type: str
    type: Literal["image"] = "image"


@dataclass
class ThinkingContent:
    thinking: str
    type: Literal["thinking"] = "thinking"
    # Anthropic extended thinking round-trip fields.
    signature: str | None = None
    redacted: bool = False
    data: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["toolCall"] = "toolCall"
    # Gemini 3 requires the thought signature echoed back on function calls.
    signature: str | None = None


ContentPart = TextContent | ImageContent | ThinkingContent | ToolCall

StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


# ---------------------------------------------------------------------------
# Usage / cost
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def add(self, other: Self) -> None:
        self.input += other.input
        self.output += other.output
        self.cache_read += other.cache_read
        self.cache_write += other.cache_write
        self.total_tokens += other.total_tokens or (other.input + other.output)
        self.cost += other.cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "output": self.output,
            "cacheRead": self.cache_read,
            "cacheWrite": self.cache_write,
            "totalTokens": self.total_tokens,
            "cost": round(self.cost, 6),
        }

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> Self:
        d = d or {}
        return Usage(
            input=d.get("input", 0),
            output=d.get("output", 0),
            cache_read=d.get("cacheRead", 0),
            cache_write=d.get("cacheWrite", 0),
            total_tokens=d.get("totalTokens", 0),
            cost=d.get("cost", 0.0),
        )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class UserMessage:
    content: str | list[ContentPart]
    role: Literal["user"] = "user"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(c.text for c in self.content if isinstance(c, TextContent))


@dataclass
class SystemMessage:
    content: str
    role: Literal["system"] = "system"
    sections: dict[str, str] | None = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class AssistantMessage:
    content: list[ContentPart] = field(default_factory=list)
    role: Literal["assistant"] = "assistant"
    api: str = ""
    provider: str = ""
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason | None = None
    error_message: str | None = None
    thinking_level: str = "off"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def text(self) -> str:
        return "".join(c.text for c in self.content if isinstance(c, TextContent))

    def tool_calls(self) -> list[ToolCall]:
        return [c for c in self.content if isinstance(c, ToolCall)]


@dataclass
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: list[ContentPart] = field(default_factory=list)
    is_error: bool = False
    details: dict[str, Any] | None = None
    duration_ms: int | None = None
    role: Literal["toolResult"] = "toolResult"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def text(self) -> str:
        return "\n".join(c.text for c in self.content if isinstance(c, TextContent))


@dataclass
class BashExecutionMessage:
    """A shell command run through the ``!`` / ``!!`` input mode."""

    command: str
    output: str = ""
    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
    full_output_path: str | None = None
    exclude_from_context: bool = False
    role: Literal["bashExecution"] = "bashExecution"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


Message = UserMessage | SystemMessage | AssistantMessage | ToolResultMessage | BashExecutionMessage


def bash_execution_to_text(message: BashExecutionMessage) -> str:
    """Render a bash execution as model-facing text."""
    text = f"Ran `{message.command}`\n"
    if message.output:
        text += f"```\n{message.output}\n```"
    else:
        text += "(no output)"
    if message.cancelled:
        text += "\n\n(command cancelled)"
    elif message.exit_code is not None and message.exit_code != 0:
        text += f"\n\nCommand exited with code {message.exit_code}"
    if message.truncated and message.full_output_path:
        text += f"\n\n[Output truncated. Full output: {message.full_output_path}]"
    return text


def message_text(message: Message) -> str:
    if isinstance(message, SystemMessage):
        return message.content
    if isinstance(message, UserMessage):
        return message.text()
    if isinstance(message, AssistantMessage):
        return message.text()
    if isinstance(message, BashExecutionMessage):
        return bash_execution_to_text(message)
    return message.text()


# ---------------------------------------------------------------------------
# Serialization (session files, JSON mode)
# ---------------------------------------------------------------------------


def content_to_dict(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextContent):
        payload: dict[str, Any] = {"type": "text", "text": part.text}
        if part.signature is not None:
            payload["signature"] = part.signature
        return payload
    if isinstance(part, ImageContent):
        return {"type": "image", "data": part.data, "mimeType": part.mime_type}
    if isinstance(part, ThinkingContent):
        payload = {"type": "thinking", "thinking": part.thinking}
        if part.signature is not None:
            payload["signature"] = part.signature
        if part.redacted:
            payload["redacted"] = True
        if part.data is not None:
            payload["data"] = part.data
        return payload
    assert isinstance(part, ToolCall)
    payload = {"type": "toolCall", "id": part.id, "name": part.name, "arguments": part.arguments}
    if part.signature is not None:
        payload["signature"] = part.signature
    return payload


def content_from_dict(d: dict[str, Any]) -> ContentPart:
    t = d.get("type")
    if t == "text":
        return TextContent(text=d.get("text", ""), signature=d.get("signature"))
    if t == "image":
        return ImageContent(data=d.get("data", ""), mime_type=d.get("mimeType", "application/octet-stream"))
    if t == "thinking":
        return ThinkingContent(
            thinking=d.get("thinking", ""),
            signature=d.get("signature"),
            redacted=bool(d.get("redacted", False)),
            data=d.get("data"),
        )
    if t == "toolCall":
        return ToolCall(
            id=d.get("id", ""), name=d.get("name", ""), arguments=d.get("arguments") or {}, signature=d.get("signature")
        )
    raise ValueError(f"Unknown content part type: {t!r}")


def message_to_dict(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        content: Any = (
            message.content if isinstance(message.content, str) else [content_to_dict(p) for p in message.content]
        )
        return {"role": "user", "content": content, "timestamp": message.timestamp}
    if isinstance(message, SystemMessage):
        d = {"role": "system", "content": message.content, "timestamp": message.timestamp}
        if message.sections is not None:
            d["sections"] = message.sections
        return d
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": [content_to_dict(p) for p in message.content],
            "api": message.api,
            "provider": message.provider,
            "model": message.model,
            "usage": message.usage.to_dict(),
            "stopReason": message.stop_reason,
            "errorMessage": message.error_message,
            "thinkingLevel": message.thinking_level,
            "timestamp": message.timestamp,
        }
    if isinstance(message, BashExecutionMessage):
        d = {
            "role": "bashExecution",
            "command": message.command,
            "output": message.output,
            "exitCode": message.exit_code,
            "cancelled": message.cancelled,
            "truncated": message.truncated,
            "timestamp": message.timestamp,
        }
        if message.full_output_path is not None:
            d["fullOutputPath"] = message.full_output_path
        if message.exclude_from_context:
            d["excludeFromContext"] = True
        return d
    assert isinstance(message, ToolResultMessage)
    return {
        "role": "toolResult",
        "toolCallId": message.tool_call_id,
        "toolName": message.tool_name,
        "content": [content_to_dict(p) for p in message.content],
        "isError": message.is_error,
        "details": message.details,
        "durationMs": message.duration_ms,
        "timestamp": message.timestamp,
    }


def message_from_dict(d: dict[str, Any]) -> Message:
    role = d.get("role")
    if role == "user":
        content = d.get("content", "")
        if isinstance(content, list):
            content = [content_from_dict(p) for p in content]
        return UserMessage(content=content, timestamp=d.get("timestamp", 0))
    if role == "system":
        return SystemMessage(content=d.get("content", ""), sections=d.get("sections"), timestamp=d.get("timestamp", 0))
    if role == "assistant":
        return AssistantMessage(
            content=[content_from_dict(p) for p in d.get("content", [])],
            api=d.get("api", ""),
            provider=d.get("provider", ""),
            model=d.get("model", ""),
            usage=Usage.from_dict(d.get("usage")),
            stop_reason=d.get("stopReason"),
            error_message=d.get("errorMessage"),
            thinking_level=d.get("thinkingLevel", "off"),
            timestamp=d.get("timestamp", 0),
        )
    if role == "toolResult":
        return ToolResultMessage(
            tool_call_id=d.get("toolCallId", ""),
            tool_name=d.get("toolName", ""),
            content=[content_from_dict(p) for p in d.get("content", [])],
            is_error=bool(d.get("isError", False)),
            details=d.get("details"),
            duration_ms=d.get("durationMs"),
            timestamp=d.get("timestamp", 0),
        )
    if role == "bashExecution":
        return BashExecutionMessage(
            command=d.get("command", ""),
            output=d.get("output", ""),
            exit_code=d.get("exitCode"),
            cancelled=bool(d.get("cancelled", False)),
            truncated=bool(d.get("truncated", False)),
            full_output_path=d.get("fullOutputPath"),
            exclude_from_context=bool(d.get("excludeFromContext", False)),
            timestamp=d.get("timestamp", 0),
        )
    raise ValueError(f"Unknown message role: {role!r}")
