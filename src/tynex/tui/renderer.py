"""Rich-based rendering of agent events for the terminal UI."""

import json
import os
import time
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from tynex.ai.models import format_model
from tynex.ai.types import AssistantMessage
from tynex.coding.agent_session import AgentSession
from tynex.coding.session_commands import format_tokens
from tynex.coding.session_commands import short as _short


def display_cwd(cwd: str, limit: int = 48) -> str:
    """Compact working directory for the footer: ``~`` for home, long paths elided."""
    text = cwd.replace("\\", "/")
    home = os.path.expanduser("~").replace("\\", "/").rstrip("/")
    if home and home != "~" and (text == home or text.startswith(home + "/")):
        text = "~" + text[len(home) :]
    if len(text) > limit:
        text = "…" + text[-(limit - 1) :]
    return text


def _context_label(context: tuple[int, int] | None) -> str:
    if not context or not context[1]:
        return ""
    pct = min(100, round(100 * context[0] / context[1]))
    return f"ctx {pct}%" + (" · /compact" if pct >= 80 else "")


def footer_text(session: AgentSession, context: tuple[int, int] | None = None) -> str:
    """Footer: model, think mode, cwd, usage, context (cheap while streaming)."""
    model = format_model(session.model) if session.model is not None else "(no model)"
    mode = session.permission_gate.mode
    think = f"think {session.thinking}" + (f" · {mode}" if mode != "accept-edits" else "")
    if session.running:
        where = _context_label(context)
        return f"{model} · {think} · {display_cwd(session.cwd)}" + (f" · {where}" if where else "")
    stats = session.stats()
    tokens = f"{format_tokens(stats['input'] + stats['cacheRead'])} in · {format_tokens(stats['output'])} out"
    where = _context_label(context) or f"session {stats['sessionId'][:8]}"
    queued = session.pending_steering
    suffix = f" · {queued} queued" if queued else ""
    return (
        f"{model} · {think} · {display_cwd(session.cwd)} · {stats['messages']} msgs · {tokens} · "
        f"${stats['cost']:.4f} · {where}{suffix}"
    )


def running_text(session: AgentSession, activity: str | None = None) -> str:
    """The status line shown above the input while a turn is running."""
    queued = session.pending_steering
    suffix = f" · {queued} queued" if queued else ""
    return f"● running · {activity or 'working…'}{suffix} · esc to interrupt"


def tool_header(name: str, args: dict[str, Any]) -> str:
    if name == "bash":
        return f"$ {_short(args.get('command', ''), 100)}"
    if name in ("pwsh", "powershell"):
        return f"PS> {_short(args.get('command', ''), 100)}"
    if name in ("read", "write", "edit", "ls"):
        return str(args.get("path", ""))
    if name in ("grep", "find"):
        return f"{args.get('pattern', '')} in {args.get('path', '.')}"
    if name == "websearch":
        return _short(args.get("query", ""), 100)
    return _short(json.dumps(args, ensure_ascii=False), 100)


class Renderer:
    """Maps agent events to terminal output above the prompt."""

    def __init__(self, session: AgentSession, console: Console) -> None:
        self.session = session
        self.console = console
        # When input is typed at the prompt, prompt_toolkit leaves the line on
        # screen, so echoing the user message again would print it twice.
        self.echo_user = True
        self._text = ""  # assistant text of the current message
        self._printed = 0  # characters already rendered
        self._scan = 0  # paragraph scan position
        self._fence = False  # inside a fenced code block
        self._block_done = False  # a Markdown block was already printed
        self._thinking = ""  # tail shown in the footer
        self._tool: str | None = None
        self._tool_started: float | None = None

    def activity(self) -> str | None:
        """Short live hint for the footer while a turn runs."""
        if self._tool:
            return f"⚙ {self._tool}"
        source, mark = (self._text, "✎") if self._text.strip() else (self._thinking, "✳")
        lines = source.strip().splitlines()
        return f"{mark} {_short(lines[-1], 46)}" if lines else None

    def _stream_block(self, *, final: bool = False) -> None:
        end = len(self._text) if final else self._paragraph_end()
        if end > self._printed:
            block = self._text[self._printed : end].strip()
            if block:
                if self._block_done:  # keep the blank line between paragraphs
                    self.console.print()
                self.console.print(Markdown(block))
                self._block_done = True
            self._printed = end
        if final:
            self._text = self._thinking = ""
            self._printed = self._scan = 0
            self._fence = self._block_done = False

    def _paragraph_end(self) -> int:
        pos, end = self._scan, 0
        while (newline := self._text.find("\n", pos)) >= 0:
            line = self._text[pos:newline]
            if line.lstrip().startswith("```"):
                self._fence = not self._fence
            pos = newline + 1
            if not self._fence and not line.strip():
                end = pos
        self._scan = pos
        return end

    def handle(self, event: dict[str, Any]) -> None:
        try:
            self._handle(event)
        except Exception:  # one bad event must not kill the interactive session
            self._tool = None

    def _handle(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "message_start":
            message = event.get("message")
            if getattr(message, "role", None) == "user":
                if self.echo_user:
                    text = message.text() if hasattr(message, "text") else str(message)
                    self.console.print(f"[bold cyan]❯[/] {text}")  # noqa: RUF001 — intentional prompt glyph
            elif isinstance(message, AssistantMessage):
                self._text, self._printed, self._scan, self._fence = "", 0, 0, False
                self._block_done = False
                self._thinking = ""
        elif etype == "message_update":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                self._text += delta.get("delta") or ""
                self._stream_block()
            elif delta.get("type") == "thinking_delta":
                # Only the tail is shown; cap it so long reasoning stays cheap.
                self._thinking = (self._thinking + (delta.get("delta") or ""))[-4000:]
        elif etype == "message_end":
            message = event.get("message")
            if isinstance(message, AssistantMessage):
                self._stream_block(final=True)
                if message.stop_reason == "error":
                    self.console.print(f"[red]✗ {message.error_message or 'Provider error'}[/]")
                elif message.stop_reason == "aborted":
                    self.console.print("[yellow]■ aborted[/]")
                self._print_usage(message)
        elif etype == "tool_execution_start":
            name = event.get("toolName", "?")
            self._tool = name
            self._tool_started = time.monotonic()
            self.console.print(f"[bold green]●[/] [bold]{name}[/] [dim]{tool_header(name, event.get('args') or {})}[/]")
        elif etype == "tool_execution_end":
            self._tool = None
            result = event.get("result")
            text = result.content[0].text if result and result.content and hasattr(result.content[0], "text") else ""
            took = f" · {time.monotonic() - self._tool_started:.2f}s" if self._tool_started else ""
            self._tool_started = None
            if event.get("isError"):
                self.console.print(f"  [red]✗ {_short(text.splitlines()[0] if text else 'error', 160)}{took}[/]")
            else:
                self.console.print(f"  [dim]↳ {self._tool_result_summary(event.get('toolName'), text)}{took}[/]")
        elif etype == "compaction":
            self.console.print(
                f"[magenta]── compacted {event.get('messagesSummarized', 0)} messages "
                f"(~{format_tokens(int(event.get('tokensBefore') or 0))} tokens) ──[/]"
            )
        elif etype == "permission_decision":
            icon = "✓" if event.get("action") == "allow" else "⊘"
            style = "dim" if event.get("action") == "allow" else "yellow"
            self.console.print(f"[{style}]{icon} permission: {_short(str(event.get('reason') or ''), 120)}[/]")
        elif etype == "limit_reached":
            self.console.print(f"[red]■ stopped — {_short(str(event.get('reason') or 'limit reached'), 140)}[/]")
        elif etype == "agent_end":
            self._tool = self._tool_started = None

    def _tool_result_summary(self, name: str | None, text: str) -> str:
        if name == "read":
            return f"{len(text.splitlines())} lines"
        if name in ("bash", "pwsh", "powershell") and text:
            last = text.strip().splitlines()[-1]
            if last.startswith("Command exited"):
                return last
            return _short(last, 120)
        first = text.strip().splitlines()[0] if text.strip() else "(no output)"
        return _short(first, 140)

    def _print_usage(self, message: AssistantMessage) -> None:
        usage = message.usage
        if usage.total_tokens or usage.cost:
            inp = usage.input + usage.cache_read
            cost = f" · ${usage.cost:.4f}" if usage.cost else ""
            cache = f" · {format_tokens(usage.cache_read)} cached" if usage.cache_read else ""
            self.console.print(f"[dim]  {format_tokens(inp)} in · {format_tokens(usage.output)} out{cache}{cost}[/]")
