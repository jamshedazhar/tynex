"""Conversation compaction: estimate context size and summarize older messages."""

from collections.abc import Callable
from dataclasses import dataclass

from tynex.ai.models import Model
from tynex.ai.providers.base import LlmContext, StreamOptions
from tynex.ai.types import (
    AssistantMessage,
    BashExecutionMessage,
    ImageContent,
    Message,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    bash_execution_to_text,
)
from tynex.coding.session import SessionManager

DEFAULT_KEEP_TOKENS = 12_000
MIN_MESSAGES_TO_COMPACT = 8

SUMMARIZATION_SYSTEM = (
    "You are a context summarization assistant. Your task is to summarize a coding session "
    "transcript so another engineer (or agent) can continue the work without the original "
    "conversation. Capture: the user's goal(s), key decisions made, files created or modified "
    "(with paths), important context and constraints, current state of the work, and the "
    "immediate next steps. Be concise but complete. Do not mention the summarization process."
)


@dataclass
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    messages_summarized: int


def estimate_message_tokens(message: Message) -> int:
    if isinstance(message, SystemMessage):
        return len(message.content) // 4 + 4
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return len(message.content) // 4 + 4
        total = 4
        for part in message.content:
            if isinstance(part, TextContent):
                total += len(part.text) // 4
            elif isinstance(part, ImageContent):
                total += 800
        return total
    if isinstance(message, AssistantMessage):
        total = 4
        for part in message.content:
            if isinstance(part, TextContent):
                total += len(part.text) // 4
            elif isinstance(part, ThinkingContent):
                total += len(part.thinking) // 4
            elif isinstance(part, ToolCall):
                total += (len(part.name) + len(str(part.arguments))) // 4
        return total
    if isinstance(message, ToolResultMessage):
        return sum(len(p.text) // 4 for p in message.content if isinstance(p, TextContent)) + 4
    if isinstance(message, BashExecutionMessage):
        return len(bash_execution_to_text(message)) // 4 + 4
    return 4


def estimate_tokens(messages: list[Message]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def _render_transcript(messages: list[Message], max_chars: int = 400_000) -> str:
    lines: list[str] = []
    total = 0
    for message in messages:
        if isinstance(message, UserMessage):
            body = message.text()
            line = f"## User\n{body}"
        elif isinstance(message, AssistantMessage):
            parts: list[str] = []
            for part in message.content:
                if isinstance(part, TextContent) and part.text:
                    parts.append(part.text)
                elif isinstance(part, ToolCall):
                    parts.append(f"[tool call] {part.name}({str(part.arguments)[:500]})")
            line = "## Assistant\n" + "\n".join(parts)
        elif isinstance(message, ToolResultMessage):
            body = message.text()
            line = f"## Tool result ({message.tool_name})\n{body[:4000]}"
        elif isinstance(message, BashExecutionMessage):
            line = f"## Shell\n{bash_execution_to_text(message)[:4000]}"
        elif isinstance(message, SystemMessage):
            line = f"## System\n{message.content[:2000]}"
        else:
            continue
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n\n".join(lines)


def _collect_text(stream_fn: Callable, model: Model, context: LlmContext, api_key: str | None) -> str:
    options = StreamOptions(api_key=api_key, thinking="off")
    last: AssistantMessage | None = None
    for event in stream_fn(model, context, options):
        partial = event.get("partial")
        if isinstance(partial, AssistantMessage):
            last = partial
    return last.text().strip() if last else ""


def compact(
    session: SessionManager,
    stream_fn: Callable,
    model: Model,
    api_key: str | None,
    keep_tokens: int = DEFAULT_KEEP_TOKENS,
    custom_instructions: str | None = None,
) -> CompactionResult | None:
    """Summarize older messages; returns None when there is nothing to compact."""
    message_entries = [e for e in session.message_entries() if e.message is not None]
    if len(message_entries) < MIN_MESSAGES_TO_COMPACT:
        return None
    # Walk backwards to find the kept window.
    split = 0
    tokens = 0
    for i in range(len(message_entries) - 1, -1, -1):
        tokens += estimate_message_tokens(message_entries[i].message)
        if tokens > keep_tokens:
            split = i + 1
            break
    start = split
    while start < len(message_entries) and getattr(message_entries[start].message, "role", None) != "user":
        start += 1
    if start >= len(message_entries):
        start = next(
            (
                i
                for i in range(len(message_entries) - 1, -1, -1)
                if getattr(message_entries[i].message, "role", None) == "user"
            ),
            0,
        )
    split = start
    dropped = message_entries[:split]
    kept = message_entries[split:]
    if not dropped or not kept:
        return None
    dropped_messages = [entry.message for entry in dropped]
    prior_summary = next(
        (entry.data.get("summary") for entry in reversed(session.path_entries()) if entry.type == "compaction"), None
    )
    transcript = _render_transcript(dropped_messages)
    prompt_parts = [f"[Conversation to summarize]\n{transcript}", "[End of conversation]"]
    if prior_summary:
        prompt_parts.insert(0, f"[Summary of even earlier context]\n{prior_summary}")
    if custom_instructions:
        prompt_parts.append(f"Additional instructions: {custom_instructions}")
    prompt_parts.append("Now write the summary.")
    context = LlmContext(
        messages=[SystemMessage(content=SUMMARIZATION_SYSTEM), UserMessage(content="\n\n".join(prompt_parts))], tools=[]
    )
    summary = _collect_text(stream_fn, model, context, api_key)
    if not summary:
        return None
    return CompactionResult(
        summary=summary,
        first_kept_entry_id=kept[0].id,
        tokens_before=estimate_tokens(dropped_messages),
        messages_summarized=len(dropped_messages),
    )
