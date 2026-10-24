"""Transcript rendering of agent events for the docked terminal UI."""

import json
import os
import time
from typing import Any

from rich.markup import escape

from tynex.ai.models import format_model
from tynex.ai.types import AssistantMessage
from tynex.coding.agent_session import AgentSession
from tynex.coding.session_commands import format_tokens
from tynex.coding.session_commands import short as _short
from tynex.tui.transcript import Block, Transcript


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
    think = f"think {session.thinking} · {mode}"
    if session.running:
        where = _context_label(context)
        return f"{model} · {think} · {display_cwd(session.cwd)}" + (f" · {where}" if where else "")
    stats = session.stats()
    tokens = f"{format_tokens(stats['input'] + stats['cacheRead'] + stats['cacheWrite'])} in · "
    tokens += f"{format_tokens(stats['output'])} out"
    where = _context_label(context) or f"session {stats['sessionId'][:8]}"
    queued = session.pending_steering
    suffix = f" · {queued} queued" if queued else ""
    if session.pending_plan is not None:
        suffix += " · plan awaiting /approve"
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


def tool_input(name: str, args: dict[str, Any]) -> str:
    """The full tool input, as shown when a tool block is expanded."""
    if name in ("bash", "pwsh", "powershell"):
        return str(args.get("command", ""))
    if name in ("read", "ls"):
        return f"{args.get('path', '.')}"
    if name == "write":
        return f"{args.get('path', '')}\n\n{args.get('content', '')}"
    if name == "edit":
        lines = [str(args.get("path", "")), ""]
        lines += [f"- {line}" for line in str(args.get("oldString", "")).splitlines()]
        lines += [f"+ {line}" for line in str(args.get("newString", "")).splitlines()]
        return "\n".join(lines)
    return json.dumps(args, indent=2, ensure_ascii=False)


def _indented(text: str) -> str:
    return "\n".join("  " + line for line in text.rstrip("\n").splitlines())


class Renderer:
    """Maps agent events to transcript blocks."""

    def __init__(self, session: AgentSession, transcript: Transcript) -> None:
        self.session = session
        self.transcript = transcript
        # When input is typed at the prompt, the prompt line already shows it;
        # echoing the user message again would print it twice.
        self.echo_user = True
        self._text = ""  # assistant text of the current message
        self._printed = 0  # characters already rendered
        self._scan = 0  # paragraph scan position
        self._fence = False  # inside a fenced code block
        self._block_done = False  # a Markdown block was already printed
        self._thinking = ""  # tail shown in the status line
        self._tool: str | None = None
        self._tool_block: Block | None = None
        self._tool_started: float | None = None

    def activity(self) -> str | None:
        """Short live hint for the status line while a turn runs."""
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
                self.transcript.add(Block(kind="assistant", text=block))
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

    def _notice(self, markup: str, *, error: bool = False) -> None:
        self.transcript.add(Block(kind="markup", text=markup, error=error))

    def handle(self, event: dict[str, Any]) -> None:
        try:
            self._handle(event)
        except Exception:  # one bad event must not kill the interactive session
            self._tool = None
            self._tool_block = None

    def _handle(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "message_start":
            message = event.get("message")
            if getattr(message, "role", None) == "user":
                if self.echo_user:
                    text = message.text() if hasattr(message, "text") else str(message)
                    self.transcript.add(Block(kind="user", text=text))
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
                    self._notice(f"[red]✗ {escape(message.error_message or 'Provider error')}[/]", error=True)
                elif message.stop_reason == "aborted":
                    self._notice("[yellow]■ aborted[/]")
                self._print_usage(message)
        elif etype == "tool_execution_start":
            self._tool_start(event.get("toolName", "?"), event.get("args") or {})
        elif etype == "tool_execution_end":
            self._tool_end(event)
        elif etype == "compaction":
            self._notice(
                f"[magenta]── compacted {event.get('messagesSummarized', 0)} messages "
                f"(~{format_tokens(int(event.get('tokensBefore') or 0))} tokens) ──[/]"
            )
        elif etype == "permission_decision":
            icon = "✓" if event.get("action") == "allow" else "⊘"
            style = "dim" if event.get("action") == "allow" else "yellow"
            self._notice(f"[{style}]{icon} permission: {escape(_short(str(event.get('reason') or ''), 120))}[/]")
        elif etype == "plan_ready":
            self._notice("[green]── plan submitted for approval ──[/]")
        elif etype == "limit_reached":
            self._notice(f"[red]■ stopped — {escape(_short(str(event.get('reason') or 'limit reached'), 140))}[/]")
        elif etype == "agent_end":
            self._tool = self._tool_started = None
            self._tool_block = None

    # -- tools ---------------------------------------------------------------
    def _tool_start(self, name: str, args: dict[str, Any]) -> None:
        self._tool = name
        self._tool_started = time.monotonic()
        self._tool_block = self.transcript.start_tool(
            f"{name}  {tool_header(name, args)}", detail=f"input:\n{_indented(tool_input(name, args))}"
        )

    def _tool_end(self, event: dict[str, Any]) -> None:
        block = self._tool_block
        name = event.get("toolName")
        result = event.get("result")
        first = result.content[0] if result is not None and result.content else None
        text = first.text if hasattr(first, "text") else ""
        took = f"{time.monotonic() - self._tool_started:.2f}s" if self._tool_started else ""
        is_error = bool(event.get("isError"))
        if block is not None:
            summary = (
                f"✗ {_short(text.splitlines()[0] if text else 'error', 160)}"
                if is_error
                else f"↳ {self._tool_result_summary(name, text)}"
            )
            detail = f"{block.detail}\n\noutput:\n{_indented(text)}" if text else block.detail
            self.transcript.finish_tool(block, summary=summary, detail=detail, took=took, error=is_error)
        self._tool = None
        self._tool_started = None
        self._tool_block = None

    def _tool_result_summary(self, name: str | None, text: str) -> str:
        lines = text.strip().splitlines()
        if name in ("read", "write"):
            return f"{len(lines)} lines"
        if name in ("bash", "pwsh", "powershell") and lines:
            # A long command is summarised by its size; one line is shown as is.
            if lines[-1].startswith("Command exited"):
                return lines[-1]
            return f"{len(lines)} lines" if len(lines) > 1 else _short(lines[0], 120)
        return _short(lines[0], 140) if lines else "(no output)"

    def _print_usage(self, message: AssistantMessage) -> None:
        usage = message.usage
        # Gate on any reported usage: Anthropic never sets ``total_tokens``, so
        # gating on it alone hid the line on every Claude turn.
        if not (usage.total_tokens or usage.cost or usage.input or usage.output or usage.cache_write):
            return
        inp = usage.input + usage.cache_read + usage.cache_write
        cost = f" · ${usage.cost:.4f}" if usage.cost else ""
        cache = f" · {format_tokens(usage.cache_read)} cached" if usage.cache_read else ""
        written = f" · {format_tokens(usage.cache_write)} cache write" if usage.cache_write else ""
        self._notice(f"[dim]  {format_tokens(inp)} in · {format_tokens(usage.output)} out{cache}{written}{cost}[/]")
