"""Tests for the interactive TUI mode: helpers, renderer events, command dispatch.

End-to-end prompt-session behaviour (Tab/Enter, sign-in menus) lives in
test_completion.py and test_integration.py; this file covers the remaining
branches: pure helpers, the Renderer event mapping and the InteractiveApp
command/model/login plumbing.
"""

import threading
import time
from pathlib import Path
from typing import Any

from prompt_toolkit.formatted_text import to_formatted_text

from tests.conftest import faux_scenario, text_scenario
from tynex.agent.permissions import PermissionRequest
from tynex.ai.models import Model, format_model, resolve_model
from tynex.ai.types import (
    AbortSignal,
    AssistantMessage,
    BashExecutionMessage,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from tynex.coding import session_commands
from tynex.coding import signin as coding_signin
from tynex.coding.agent_session import AgentSession
from tynex.coding.signin import provider_label
from tynex.coding.tools.shell import ShellTool, shell_name, shell_prompt
from tynex.tui import app as interactive
from tynex.tui.app import InteractiveApp
from tynex.tui.prompt import _prompt_key_bindings
from tynex.tui.renderer import Renderer, display_cwd, footer_text, running_text
from tynex.tui.transcript import Transcript

# Keep Rich from wrapping long temporary paths differently on each platform.
CONSOLE_WIDTH = 300


class FakePrompt:
    """Stands in for the prompt_toolkit session in ``InteractiveApp._input``."""

    def __init__(self, *responses: str | BaseException) -> None:
        self.responses = list(responses)
        self.asked: list[tuple[str, bool]] = []
        self.defaults: list[str] = []

    def prompt(self, text: str, *, is_password: bool = False, **kwargs: Any) -> str:
        self.asked.append((text, is_password))
        self.defaults.append(kwargs.get("default") or "")
        item = self.responses.pop(0) if self.responses else EOFError()
        if isinstance(item, BaseException):
            raise item
        return item


class Entry:
    """Minimal stand-in for ``tynex.coding.session.SessionEntry``."""

    def __init__(self, id: str, summary: str) -> None:
        self.id = id
        self._summary = summary

    def summary_line(self, max_len: int = 60) -> str:
        return self._summary


def make_app(
    tmp_path: Path, *, model: Model | None = None, terminal: bool = False, tool_names: list[str] | None = None
) -> InteractiveApp:
    session = AgentSession(cwd=str(tmp_path), model=model, tool_names=tool_names or [], no_session=True)
    # The console writes into the transcript, so command output is inspectable.
    return InteractiveApp(session, terminal=terminal)


def output(app: InteractiveApp) -> str:
    return app.transcript.text()


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def make_renderer(tmp_path: Path, **session_kwargs: Any) -> tuple[Renderer, Transcript]:
    session = AgentSession(cwd=str(tmp_path), tool_names=[], no_session=True, **session_kwargs)
    transcript = Transcript()
    return Renderer(session, transcript), transcript


def test_renderer_echoes_user_message_and_honours_echo_flag(tmp_path):
    renderer, buffer = make_renderer(tmp_path)
    renderer.handle({"type": "message_start", "message": UserMessage(content="hello there")})
    assert "❯ hello there" in buffer.text()  # noqa: RUF001 — intentional prompt glyph

    renderer.echo_user = False
    renderer.handle({"type": "message_start", "message": UserMessage(content="hidden message")})
    assert "hidden message" not in buffer.text()


def test_renderer_streams_markdown_paragraphs_without_splitting_code(tmp_path):
    renderer, buffer = make_renderer(tmp_path)
    message = AssistantMessage(api="faux", provider="faux", model="test")

    renderer.handle({"type": "message_start", "message": message})
    renderer.handle(
        {
            "type": "message_update",
            "delta": {"type": "text_delta", "delta": "Intro\n\n```\nx = 1\n\ny = 2\n```\n\nOutro"},
        }
    )
    # The first two blocks are complete (the blank lines are outside fences),
    # so they render while the rest is still streaming.
    assert "Intro" in buffer.text() and "x = 1" in buffer.text() and "y = 2" in buffer.text()
    assert "Outro" not in buffer.text()
    assert renderer._text.endswith("Outro")

    message.content = [TextContent(text="Intro")]
    message.stop_reason = "stop"
    renderer.handle({"type": "message_end", "message": message})
    out = buffer.text()
    assert out.count("Intro") == 1 and out.count("Outro") == 1  # no duplicate render
    assert renderer._text == ""


def test_renderer_streams_many_chunks_without_duplicates(tmp_path):
    renderer, buffer = make_renderer(tmp_path)
    message = AssistantMessage()
    renderer.handle({"type": "message_start", "message": message})
    text = "\n\n".join(f"Paragraph {index}." for index in range(30)) + "\n\ntail"
    for offset in range(0, len(text), 7):  # deliberately split mid-word
        renderer.handle({"type": "message_update", "delta": {"type": "text_delta", "delta": text[offset : offset + 7]}})
    message.stop_reason = "stop"
    renderer.handle({"type": "message_end", "message": message})

    out = buffer.text()
    assert all(out.count(f"Paragraph {index}.") == 1 for index in range(30))
    assert out.count("tail") == 1


def test_renderer_flushes_a_short_message_on_end_once(tmp_path):
    renderer, buffer = make_renderer(tmp_path)
    message = AssistantMessage(api="faux", provider="faux", model="test")
    renderer.handle({"type": "message_start", "message": message})
    renderer.handle({"type": "message_update", "delta": {"type": "text_delta", "delta": "Hello **world**"}})
    assert buffer.text() == ""  # no blank line yet: wait for the end

    message.content = [TextContent(text="Hello **world**")]
    message.stop_reason = "stop"
    renderer.handle({"type": "message_end", "message": message})
    assert buffer.text().count("Hello") == 1


def test_renderer_activity_reports_thinking_text_and_tools(tmp_path):
    renderer, buffer = make_renderer(tmp_path)
    renderer.handle({"type": "message_start", "message": AssistantMessage()})
    assert renderer.activity() is None

    renderer.handle({"type": "message_update", "delta": {"type": "thinking_delta", "delta": "weighing\noptions"}})
    assert renderer.activity() == "✳ options"

    # A new assistant turn clears the previous partial state.
    renderer.handle({"type": "message_start", "message": AssistantMessage()})
    assert renderer.activity() is None

    renderer.handle({"type": "tool_execution_start", "toolName": "bash", "args": {"command": "ls"}})
    assert renderer.activity() == "⚙ bash"
    assert "$ ls" in buffer.text()
    renderer.handle({"type": "tool_execution_end", "toolName": "bash"})
    assert renderer.activity() is None
    assert "(no output)" in buffer.text()

    # Streamed text shows its latest line while it arrives.
    renderer.handle({"type": "message_start", "message": AssistantMessage()})
    renderer.handle({"type": "message_update", "delta": {"type": "text_delta", "delta": "first line\nsecond"}})
    assert renderer.activity() == "✎ second"


def test_renderer_message_end_error_aborted_and_tool_turn(tmp_path):
    renderer, buffer = make_renderer(tmp_path)
    renderer.handle({"type": "message_end", "message": AssistantMessage(stop_reason="error", error_message="boom")})
    assert "✗ boom" in buffer.text()

    renderer, buffer = make_renderer(tmp_path)
    renderer.handle({"type": "message_end", "message": AssistantMessage(stop_reason="error")})
    assert "Provider error" in buffer.text()

    renderer, buffer = make_renderer(tmp_path)
    renderer.handle({"type": "message_end", "message": AssistantMessage(stop_reason="aborted")})
    assert "aborted" in buffer.text()

    # A tool-use turn with no text stays quiet (no empty markdown, no error line).
    renderer, buffer = make_renderer(tmp_path)
    renderer.handle({"type": "message_end", "message": AssistantMessage(stop_reason="toolUse")})
    assert buffer.text().strip() == ""


def test_renderer_tool_lifecycle_prints_header_and_summaries(tmp_path):
    renderer, buffer = make_renderer(tmp_path)

    renderer.handle({"type": "tool_execution_start", "toolName": "read", "args": {"path": "notes.txt"}})
    assert "read" in buffer.text() and "notes.txt" in buffer.text()
    renderer.handle(
        {
            "type": "tool_execution_end",
            "toolName": "read",
            "result": ToolResultMessage(tool_call_id="c2", tool_name="read", content=[TextContent(text="a\nb\nc")]),
        }
    )
    assert "↳ 3 lines · 0.0" in buffer.text()

    renderer.handle({"type": "tool_execution_start", "toolName": "bash", "args": {"command": "false"}})
    renderer.handle(
        {
            "type": "tool_execution_end",
            "toolName": "bash",
            "isError": True,
            "result": ToolResultMessage(
                tool_call_id="c3", tool_name="bash", content=[TextContent(text="line one\nline two")], is_error=True
            ),
        }
    )
    assert "✗ line one" in buffer.text()

    renderer.handle({"type": "tool_execution_start", "toolName": "ls", "args": {"path": "."}})
    renderer.handle({"type": "tool_execution_end", "toolName": "ls"})
    assert "(no output)" in buffer.text()


def test_renderer_keeps_the_full_tool_input_and_output_for_expansion(tmp_path):
    renderer, transcript = make_renderer(tmp_path)
    command = "python -c 'print(1)' " + "arg" * 120  # far past the 100-char header
    output = "\n".join(f"line {index}" for index in range(1, 60))
    renderer.handle({"type": "tool_execution_start", "toolName": "bash", "args": {"command": command}})
    renderer.handle(
        {
            "type": "tool_execution_end",
            "toolName": "bash",
            "result": ToolResultMessage(tool_call_id="c4", tool_name="bash", content=[TextContent(text=output)]),
        }
    )

    block = transcript.last_tool()
    assert block is not None and block.expandable
    # Nothing is cut: the block holds the whole command and the whole output.
    assert command in block.detail
    assert all(f"line {index}" in block.detail for index in range(1, 60))
    # Collapsed, the transcript shows the one-line header and summary only.
    assert "line 30" not in transcript.text()
    assert transcript.toggle(block) and block.expanded


def test_renderer_prints_file_contents_as_text_when_expanded(tmp_path):
    renderer, transcript = make_renderer(tmp_path)
    content = "\n".join(f"line {index} of the new file" for index in range(1, 120))
    renderer.handle(
        {"type": "tool_execution_start", "toolName": "write", "args": {"path": "new.py", "content": content}}
    )
    block = transcript.last_tool()
    assert block is not None
    assert "line 90 of the new file" not in transcript.text()  # collapsed

    transcript.toggle(block)
    expanded = transcript.text()
    # The content is shown as text, not as one JSON-escaped line.
    assert "line 90 of the new file" in expanded and "line 119 of the new file" in expanded
    assert "\\n" not in expanded


def test_renderer_compaction_and_agent_end(tmp_path):
    renderer, buffer = make_renderer(tmp_path)
    renderer.handle({"type": "compaction", "messagesSummarized": 4, "tokensBefore": 12_000})
    assert "compacted 4 messages" in buffer.text() and "12.0k tokens" in buffer.text()

    renderer.handle({"type": "tool_execution_start", "toolName": "ls", "args": {}})
    renderer.handle({"type": "agent_end"})
    assert renderer.activity() is None


def test_renderer_swallows_handler_errors(tmp_path, monkeypatch):
    renderer, _ = make_renderer(tmp_path)

    def boom(event: dict[str, Any]) -> None:
        raise RuntimeError("bad event")

    monkeypatch.setattr(renderer, "_handle", boom)
    renderer.handle({"type": "message_start"})  # must not raise
    assert renderer.activity() is None


# ---------------------------------------------------------------------------
# InteractiveApp: input, banner, toolbar
# ---------------------------------------------------------------------------


def test_status_line_shows_the_run_and_the_prompt_stays_single_line(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    idle = "".join(fragment for _, fragment in to_formatted_text(app._prompt()))
    assert "\n" not in idle and idle.endswith("❯ ")  # noqa: RUF001
    assert "".join(fragment for _, fragment in to_formatted_text(app._status())) == ""

    monkeypatch.setattr(app, "_busy", lambda: True)
    app.renderer._tool = "bash"
    app.session.steer("later")
    running = "".join(fragment for _, fragment in to_formatted_text(app._status()))
    assert running.startswith("● running")
    assert "⚙ bash" in running and "1 queued" in running and "esc to interrupt" in running
    # The status is its own row in the docked layout; the prompt stays one line.
    assert "\n" not in "".join(fragment for _, fragment in to_formatted_text(app._prompt()))


def test_usage_line_counts_a_cache_write_and_needs_no_total(tmp_path):
    # Anthropic's first turn reports a cache write and no ``total_tokens``.
    renderer, transcript = make_renderer(tmp_path)
    usage = Usage(input=10, output=5, cache_write=30_000)
    renderer.handle(
        {
            "type": "message_end",
            "message": AssistantMessage(provider="anthropic", api="anthropic", model="x", usage=usage),
        }
    )
    text = transcript.text()
    assert "in · 5 out" in text  # the line is printed at all
    assert "30.0k in" in text  # input + cache write, not just input
    assert "30.0k cache write" in text


def test_footer_text_reports_model_think_usage_and_context(tmp_path):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    text = footer_text(app.session)
    assert "faux/test" in text and f"think {app.session.thinking}" in text
    assert display_cwd(app.session.cwd) in text
    assert "0 in · 0 out" in text and "$0.0000" in text and "session" in text
    assert "running" not in text

    app.session._running = True
    text = footer_text(app.session)
    assert f"think {app.session.thinking}" in text
    assert "msgs" not in text and "esc to interrupt" not in text

    assert "⚙ bash" in running_text(app.session, "⚙ bash")
    app.session.steer("later")
    assert "1 queued" in running_text(app.session, "⚙ bash")

    app.session._running = False
    assert "1 queued" in footer_text(app.session)  # survives an aborted/ended turn
    assert "ctx 85%" in footer_text(app.session, context=(85_000, 100_000))
    assert "/compact" in footer_text(app.session, context=(85_000, 100_000))

    # A pending plan is the next action, so the footer names the command.
    assert "plan awaiting /approve" not in footer_text(app.session)
    app.session.pending_plan = {"id": "e1", "plan": "do it"}
    assert "plan awaiting /approve" in footer_text(app.session)


# ---------------------------------------------------------------------------
# InteractiveApp: main loop and line dispatch
# ---------------------------------------------------------------------------


def test_run_loop_skips_blank_lines_and_exits_on_quit(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    prompts: list[tuple[str, bool]] = []
    monkeypatch.setattr(app, "_run_prompt", lambda text, echo_user=True: prompts.append((text, echo_user)))

    app._input = FakePrompt("", "hello", "/quit")
    assert app.run() == 0
    assert prompts == [("hello", False)]  # typed lines are not echoed twice
    assert "bye" in output(app)


def test_run_loop_continues_after_interrupt_and_exits_on_eof(tmp_path):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    app._input = FakePrompt(KeyboardInterrupt(), "", EOFError())
    assert app.run() == 0
    assert "bye" in output(app)


def test_run_dispatches_initial_message_before_the_loop(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    prompts: list[tuple[str, bool]] = []
    monkeypatch.setattr(app, "_run_prompt", lambda text, echo_user=True: prompts.append((text, echo_user)))

    app._input = FakePrompt(EOFError())
    assert app.run("fix the tests") == 0
    # CLI messages were never on screen, so they are echoed.
    assert prompts == [("fix the tests", True)]


def test_handle_line_routes_bash_lines(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    ran: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        app, "_run_bash", lambda command, *, exclude_from_context=False: ran.append((command, exclude_from_context))
    )

    app._handle_line("!echo hi")
    app._handle_line("!!echo hi")
    app._handle_line("! echo hi")
    app._handle_line("!!  echo hi")
    assert ran == [("echo hi", False), ("echo hi", True), ("echo hi", False), ("echo hi", True)]

    prompts: list[str] = []
    monkeypatch.setattr(app, "_submit", lambda text, echo_user=True: prompts.append(text))
    app._handle_line("!")  # not a bash line: sent to the model
    app._handle_line("!!")  # empty command: sent to the model as written
    assert prompts == ["!", "!!"]


def test_handle_line_expands_prompt_template(tmp_path, monkeypatch):
    prompts_dir = tmp_path / ".tynex" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "fix.md").write_text("Fix {{1}} now. Args: {{args}}", encoding="utf-8")

    app = make_app(tmp_path)
    prompts: list[tuple[str, bool]] = []
    monkeypatch.setattr(app, "_submit", lambda text, echo_user=True: prompts.append((text, echo_user)))

    app._handle_line("/fix tests now")
    assert prompts == [("Fix tests now. Args: tests now", True)]


# ---------------------------------------------------------------------------
# InteractiveApp: _handle_command dispatch
# ---------------------------------------------------------------------------


def test_handle_command_quit_empty_and_unknown(tmp_path):
    app = make_app(tmp_path)
    assert app._handle_command("/quit") is False
    assert app._handle_command("/exit") is False
    assert app._handle_command("/q") is False
    assert app._handle_command("/") is None
    assert app._handle_command("/nope") is None
    assert "Unknown command /nope" in output(app)
    # A near miss names the command the user probably meant.
    assert app._handle_command("/permisions") is None
    assert "Did you mean /permissions?" in output(app)


def test_handle_command_help_stats_and_cwd(tmp_path):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    assert app._handle_command("/help") is None
    assert "Commands" in output(app)

    assert app._handle_command("/stats") is None
    out = output(app)
    assert "faux/test" in out and "Session" in out and "Tokens" in out

    assert app._handle_command("/session") is None
    assert "Messages" in output(app)

    assert app._handle_command("/cwd") is None
    assert app.session.cwd in output(app)


def test_handle_command_model_switches_and_reports_errors(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    picks: list[bool] = []
    monkeypatch.setattr(app, "_pick_model", lambda: picks.append(True))

    app._handle_command("/model faux/test")  # real pattern switch
    assert "Switched to" in output(app)

    def fail(pattern: str) -> Model:
        raise ValueError(f"unknown model {pattern}")

    monkeypatch.setattr(app.session, "set_model_by_pattern", fail)
    app._handle_command("/model nope")
    assert "unknown model nope" in output(app)

    app._handle_command("/model")
    assert picks == [True]


def test_handle_command_routes_helpers(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    called: list[str] = []
    monkeypatch.setattr(session_commands, "resume", lambda *a: called.append(f"resume:{a[2]}"))
    monkeypatch.setattr(session_commands, "thinking", lambda *a: called.append(f"thinking:{a[2]}"))
    monkeypatch.setattr(session_commands, "export_session", lambda *a: called.append(f"export:{a[2]}"))
    monkeypatch.setattr(session_commands, "session_info", lambda *a: called.append("session"))
    monkeypatch.setattr(coding_signin, "login", lambda *a: called.append(f"login:{a[1]}"))
    monkeypatch.setattr(coding_signin, "logout", lambda *a: called.append(f"logout:{a[1]}"))
    monkeypatch.setattr(app.session, "compact_now", lambda arg=None: arg != "skip")
    monkeypatch.setattr(app.session, "new_session", lambda: called.append("new"))

    app._handle_command("/resume 2020")
    app._handle_command("/sessions")
    app._handle_command("/login anthropic")
    app._handle_command("/logout")
    app._handle_command("/thinking high")
    app._handle_command("/export out.html")
    app._handle_command("/session")
    app._handle_command("/stats")
    app._handle_command("/compact notes")
    assert "Compacted." in output(app)
    app._handle_command("/compact skip")
    assert "Nothing to compact." in output(app)
    app._handle_command("/new")
    assert "New session" in output(app)
    assert called == [
        "resume:2020",
        "resume:",
        "login:anthropic",
        "logout:",
        "thinking:high",
        "export:out.html",
        "session",
        "session",
        "new",
    ]


def test_handle_command_name_sets_and_shows_label(tmp_path):
    app = make_app(tmp_path)
    assert app._handle_command("/name my branch") is None
    assert "Session named 'my branch'" in output(app)
    assert app.session.session_label() == "my branch"

    assert app._handle_command("/name") is None
    assert "Session label: my branch" in output(app)


def test_handle_command_extension_commands(tmp_path):
    app = make_app(tmp_path)
    seen: list[str] = []
    app.session.extensions.add_command("greet", lambda arg: seen.append(arg))
    assert app._handle_command("/greet world") is None
    assert seen == ["world"]

    def broken(arg: str) -> None:
        raise RuntimeError("kaput")

    app.session.extensions.add_command("broken", broken)
    assert app._handle_command("/broken") is None
    assert "Extension command failed: kaput" in output(app)


# ---------------------------------------------------------------------------
# InteractiveApp: model commands
# ---------------------------------------------------------------------------


def test_list_models_marks_current_and_notes_missing_catalogs(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("anthropic/claude-sonnet-4-5"))
    app._list_models()
    out = output(app)
    assert "claude-sonnet-4-5" in out and "claude-haiku-4-5" in out
    assert "●" in out  # the active model is marked
    assert "/model <pattern> to switch." in out

    # Providers without a catalog explain why they are empty.
    monkeypatch.setattr(session_commands, "ensure_models", lambda provider, settings, **kw: ([], "not configured"))
    app._list_models()
    assert "(not configured)" in output(app)


def test_pick_model_without_model_lists_catalog(tmp_path):
    app = make_app(tmp_path)  # no model selected
    assert app.session.model is None
    app._pick_model()
    out = output(app)
    assert "claude-sonnet-4-5" in out and "/model <pattern> to switch." in out


def test_pick_model_without_catalog_prints_note(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    monkeypatch.setattr(session_commands, "ensure_models", lambda provider, settings, **kw: ([], "catalog offline"))
    app._pick_model()
    out = output(app)
    assert "No models available for faux" in out and "catalog offline" in out


def test_pick_model_keeps_current_when_picker_cancels(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("anthropic/claude-sonnet-4-5"), terminal=True)

    def interrupt(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(interactive, "select_option", interrupt)
    app._pick_model()
    assert "Model unchanged" not in output(app)
    assert app.session.model is not None and app.session.model.id == "claude-sonnet-4-5"


def test_pick_model_selecting_current_model_is_a_noop(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("anthropic/claude-sonnet-4-5"), terminal=True)
    monkeypatch.setattr(interactive, "select_option", lambda *args, **kwargs: 0)
    app._pick_model()
    assert "is already active." in output(app)
    assert app.session.model is not None and app.session.model.id == "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# InteractiveApp: /resume
# ---------------------------------------------------------------------------


def test_resume_without_sessions(tmp_path):
    app = make_app(tmp_path)
    app._handle_command("/resume")
    assert "No sessions for this directory." in output(app)


def test_resume_by_identifier(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    session_file = str(tmp_path / "2026-01-01_tynex-abc.jsonl")
    monkeypatch.setattr(
        session_commands,
        "list_sessions",
        lambda cwd: [{"id": "abc12345", "modified": time.time(), "messages": 3, "label": "", "file": session_file}],
    )
    switched: list[str] = []
    monkeypatch.setattr(app.session, "switch_session", switched.append)

    monkeypatch.setattr(session_commands, "find_session", lambda cwd, arg: None)
    app._handle_command("/resume missing")
    assert "No session matching 'missing'" in output(app)

    monkeypatch.setattr(session_commands, "find_session", lambda cwd, arg: session_file)
    app._handle_command("/resume abc")
    assert switched == [session_file]
    assert f"Resumed {Path(session_file).name}" in output(app)


def test_resume_lists_and_picks_a_number(tmp_path, monkeypatch):
    sessions = [
        {"id": "aaaa0001", "modified": 1_700_000_000.0, "messages": 2, "label": "work", "file": "a.jsonl"},
        {"id": "bbbb0002", "modified": 1_700_000_100.0, "messages": 5, "label": "", "file": "b.jsonl"},
    ]
    monkeypatch.setattr(session_commands, "list_sessions", lambda cwd: sessions)

    app = make_app(tmp_path)
    switched: list[str] = []
    monkeypatch.setattr(app.session, "switch_session", switched.append)

    app._input = FakePrompt("2")
    app._handle_command("/resume")
    out = output(app)
    assert "aaaa0001" in out and "work" in out and "bbbb0002" in out
    assert switched == ["b.jsonl"]
    assert "Resumed b.jsonl" in output(app)

    # Non-numeric, out-of-range and interrupted choices all cancel quietly.
    app._input = FakePrompt("nope")
    app._handle_command("/resume")
    app._input = FakePrompt("9")
    app._handle_command("/resume")
    app._input = FakePrompt(KeyboardInterrupt())
    app._handle_command("/resume")
    assert switched == ["b.jsonl"]


# ---------------------------------------------------------------------------
# InteractiveApp: agent runs
# ---------------------------------------------------------------------------


def test_submit_without_model_asks_to_sign_in(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    assert app.session.model is None

    def must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the model must not run without a model selection")

    monkeypatch.setattr(app.session, "prompt", must_not_run)
    app._submit("hello")
    assert "No model selected" in output(app)
    assert app._busy() is False


def test_run_prompt_streams_output_and_restores_echo(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))

    def fake_prompt(text: str, emit: Any = None) -> None:
        emit({"type": "message_start", "message": AssistantMessage()})
        emit({"type": "message_update", "delta": {"type": "text_delta", "delta": "Hi\n\nThere"}})
        emit({"type": "message_end", "message": AssistantMessage(stop_reason="stop")})

    monkeypatch.setattr(app.session, "prompt", fake_prompt)
    app._run_prompt("hello")

    assert "Hi" in output(app) and "There" in output(app)
    assert app.renderer.echo_user is True  # restored by the finally block


def test_run_prompt_prints_unexpected_errors_instead_of_raising(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))

    def boom(text: str, emit: Any = None) -> None:
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(app.session, "prompt", boom)
    app._run_prompt("hello")
    assert "provider exploded" in output(app)


def test_interrupt_run_aborts_the_active_turn_once(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    assert app._interrupt_run() is None  # idle: nothing to interrupt

    monkeypatch.setattr(app, "_busy", lambda: True)
    app.session._signal = AbortSignal()
    app._interrupt_run()
    assert app.session._signal.aborted
    assert output(app).count("Interrupting…") == 1

    app._interrupt_run()  # already aborting: stays quiet
    assert output(app).count("Interrupting…") == 1


def test_messages_typed_during_a_run_are_queued(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    started = threading.Event()
    release = threading.Event()

    def fake_run(text: str, echo_user: bool = True) -> None:
        started.set()
        release.wait(5)

    monkeypatch.setattr(app, "_run_prompt", fake_run)
    app._handle_line("first message")
    assert started.wait(5)
    assert app._busy()

    app._handle_line("follow up", echo_user=False)
    assert [message.content for message in app.session._steering] == ["follow up"]
    assert "queued" in output(app)

    release.set()
    app._run_thread.join(5)
    assert not app._busy()


def test_commands_typed_during_a_run_go_back_to_the_editor(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    started = threading.Event()
    release = threading.Event()

    def fake_run(text: str, echo_user: bool = True) -> None:
        started.set()
        release.wait(5)

    monkeypatch.setattr(app, "_run_prompt", fake_run)
    prompt = FakePrompt()
    app._input = prompt
    app._handle_line("first message")
    assert started.wait(5)

    app._handle_line("/model other")
    # prompt() resets the buffer on every call, so the line must go back as the
    # next prompt's default (the run loop passes ``default=self._prefill``).
    assert app._prefill == "/model other"
    assert "press esc" in output(app)
    assert not app.session._steering  # commands are not sent to the model

    release.set()
    app._run_thread.join(5)


def test_quit_while_a_run_is_active_still_exits(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    started = threading.Event()
    release = threading.Event()

    def fake_run(text: str, echo_user: bool = True) -> None:
        started.set()
        release.wait(5)

    monkeypatch.setattr(app, "_run_prompt", fake_run)
    app._handle_line("first message")
    assert started.wait(5)

    assert app._handle_line("/quit") is False
    release.set()
    app._run_thread.join(5)


def test_shutdown_run_aborts_and_waits_for_the_worker(tmp_path):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    started = threading.Event()
    finished = threading.Event()

    def fake_run() -> None:
        started.set()
        while app.session.signal is not None and not app.session.signal.aborted:
            time.sleep(0.01)
        finished.set()

    app.session._signal = AbortSignal()
    thread = threading.Thread(target=fake_run)
    app._run_thread = thread
    thread.start()
    assert started.wait(5)

    app._shutdown_run()
    assert app.session._signal.aborted
    assert finished.is_set()
    assert not thread.is_alive()


def test_prompt_key_bindings_interrupt_on_escape_and_ctrl_c():
    busy = [False]
    calls: list[str] = []
    bindings = _prompt_key_bindings(interrupt=lambda: calls.append("interrupt"), is_busy=lambda: busy[0])
    interrupt_bindings = [binding for binding in bindings.bindings if binding.keys in (("escape",), ("c-c",))]
    assert [binding.keys for binding in interrupt_bindings] == [("escape",), ("c-c",)]

    for binding in interrupt_bindings:
        assert binding.filter() is False  # idle: default Esc/Ctrl+C behaviour
    busy[0] = True
    for binding in interrupt_bindings:
        assert binding.filter() is True
        binding.handler(None)
    assert calls == ["interrupt", "interrupt"]


def test_prompt_key_bindings_alt_enter_inserts_a_newline():
    from types import SimpleNamespace

    bindings = _prompt_key_bindings()
    binding = next(binding for binding in bindings.bindings if binding.keys == ("escape", "c-m"))
    inserted: list[str] = []
    binding.handler(SimpleNamespace(current_buffer=SimpleNamespace(insert_text=inserted.append)))
    assert inserted == ["\n"]


def test_transcript_writer_feeds_console_output_into_the_transcript(tmp_path):
    app = make_app(tmp_path)
    changes: list[int] = []
    app.transcript.on_change = lambda: changes.append(1)

    app.console.print("streamed [bold]output[/] from the agent thread")
    assert "streamed output from the agent thread" in output(app)
    assert changes, "the docked prompt must be asked to repaint"

    # The writer is a real file object: no console escape codes, no fileno.
    writer = interactive._TranscriptWriter(app.transcript)
    writer.write("raw line\n")
    writer.flush()
    assert "raw line" in output(app)
    assert writer.isatty() is False and writer.encoding == "utf-8"


def test_prompt_accepts_input_and_escape_interrupts_while_running(tmp_path, monkeypatch):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    app = make_app(tmp_path, model=resolve_model("faux/test"))
    started = threading.Event()

    def fake_prompt(text: str, emit: Any = None) -> None:
        app.session._running = True
        app.session._signal = AbortSignal()
        started.set()
        while not app.session._signal.aborted:
            time.sleep(0.01)
        app.session._running = False
        app.session._signal = None

    def wait_until(check: Any, what: str, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if check():
                return
            time.sleep(0.01)
        raise AssertionError(f"timed out waiting for {what}")

    monkeypatch.setattr(app.session, "prompt", fake_prompt)
    with create_pipe_input() as pipe:
        app._input = interactive.DockedPrompt(
            app.transcript,
            status=app._status,
            key_bindings=_prompt_key_bindings(interrupt=app._interrupt_run, is_busy=app._busy),
            input=pipe,
            output=DummyOutput(),
        )
        answers: list[str] = []
        prompt_thread = threading.Thread(target=lambda: answers.append(app.input.prompt(app._prompt())))
        prompt_thread.start()
        wait_until(lambda: app.input.app.is_running, "the prompt to start")

        # The run starts while the prompt is active, like a CLI message does.
        app._submit("first message")
        assert started.wait(5)

        # The prompt stays live: Esc aborts the run.
        pipe.send_text("\x1b")
        wait_until(lambda: not app._busy(), "escape to abort the run")

        # Ctrl+C does the same for the next run.
        started.clear()
        app._submit("second message")
        assert started.wait(5)
        pipe.send_text("\x03")
        wait_until(lambda: not app._busy(), "ctrl+c to abort the run")

        # A line submitted after the interrupts still goes through.
        pipe.send_text("still typing\r")
        prompt_thread.join(10)
        assert answers == ["still typing"]
    app._run_thread.join(5)


def test_ask_masks_only_the_password_question(tmp_path):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    app = make_app(tmp_path, model=resolve_model("faux/test"))
    answers: list[str] = []
    with create_pipe_input() as pipe:
        session = interactive.DockedPrompt(app.transcript, input=pipe, output=DummyOutput())
        app._input = session

        def ask(text: str, **kwargs: Any) -> threading.Thread:
            thread = threading.Thread(target=lambda: answers.append(app._ask(text, **kwargs)))
            thread.start()
            deadline = time.monotonic() + 10
            while not session.app.is_running and time.monotonic() < deadline:
                time.sleep(0.01)
            return thread

        # The API-key question hides what is typed...
        thread = ask("api key: ", password=True)
        assert session.is_password is True
        pipe.send_text("sk-secret\r")
        thread.join(10)
        assert answers == ["sk-secret"]
        # ...but the mask must not stick to the shared session: the next
        # prompt (and the main input line) renders normal characters again.
        assert session.is_password is False

        thread = ask("message: ")
        assert session.is_password is False
        pipe.send_text("hello\r")
        thread.join(10)
        assert answers == ["sk-secret", "hello"]


# ---------------------------------------------------------------------------
# InteractiveApp: !bash / !!
# ---------------------------------------------------------------------------


def test_run_bash_executes_even_without_the_bash_tool(tmp_path):
    # Shell execution is independent of the tool loadout, so ``!`` still works
    # when the bash tool is not enabled for the model.
    app = make_app(tmp_path, tool_names=[])
    app._run_bash("echo hello-tynex-bang")
    out = output(app)
    assert shell_prompt in out
    assert "echo hello-tynex-bang" in out
    assert "hello-tynex-bang" in out
    assert isinstance(app.session.context.messages[-1], BashExecutionMessage)


def test_run_bash_prints_output_and_errors(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    ok = BashExecutionMessage(command="echo hi", output="hi there", exit_code=0)
    monkeypatch.setattr(app.session, "execute_bash", lambda command, **kwargs: ok)
    app._run_bash("echo hi")
    out = output(app)
    assert "echo hi" in out and "hi there" in out

    bad = BashExecutionMessage(command="false", output="boom", exit_code=1)
    monkeypatch.setattr(app.session, "execute_bash", lambda command, **kwargs: bad)
    app._run_bash("false")
    out = output(app)
    assert "boom" in out
    assert "command exited with code 1" in out

    def boom(command, **kwargs):
        raise RuntimeError("no shell")

    monkeypatch.setattr(app.session, "execute_bash", boom)
    app._run_bash("nope")
    assert "no shell" in output(app)


def test_resume_uses_arrow_picker_on_a_terminal(tmp_path, monkeypatch):
    sessions = [
        {"id": "aaaa0001", "modified": 1_700_000_000.0, "messages": 2, "label": "work", "file": "a.jsonl"},
        {"id": "bbbb0002", "modified": 1_700_000_100.0, "messages": 5, "label": "", "file": "b.jsonl"},
    ]
    monkeypatch.setattr(session_commands, "list_sessions", lambda cwd: sessions)
    app = make_app(tmp_path, terminal=True)
    switched: list[str] = []
    monkeypatch.setattr(app.session, "switch_session", switched.append)
    seen: dict[str, Any] = {}

    def fake_select(title: str, options: list[str], **kwargs: Any) -> int:
        seen["title"], seen["options"] = title, options
        return 1

    monkeypatch.setattr(interactive, "select_option", fake_select)
    app._handle_command("/resume")
    assert seen["title"] == "Resume session"
    assert "aaaa0001" in seen["options"][0] and "work" in seen["options"][0]
    assert switched == ["b.jsonl"]
    assert "Resumed b.jsonl" in output(app)


def test_resume_picker_interrupt_does_not_resume(tmp_path, monkeypatch):
    # An interrupted picker must not be taken as a row index: -1 would silently
    # resume the last session in the list.
    sessions = [{"id": "aaaa0001", "modified": 1_700_000_000.0, "messages": 2, "label": "work", "file": "a.jsonl"}]
    monkeypatch.setattr(session_commands, "list_sessions", lambda cwd: sessions)
    app = make_app(tmp_path, terminal=True)
    switched: list[str] = []
    monkeypatch.setattr(app.session, "switch_session", switched.append)

    def interrupted(title: str, options: list[str], **kwargs: Any) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(interactive, "select_option", interrupted)
    app._handle_command("/resume")
    assert switched == []


def test_run_bash_excluded_and_truncated_output(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    seen: dict[str, bool] = {}

    def fake(command, *, exclude_from_context=False):
        seen["exclude"] = exclude_from_context
        return BashExecutionMessage(
            command=command,
            output="lots",
            exit_code=0,
            truncated=True,
            full_output_path="C:/tmp/full.txt",
            exclude_from_context=exclude_from_context,
        )

    monkeypatch.setattr(app.session, "execute_bash", fake)
    app._run_bash("echo lots", exclude_from_context=True)
    out = output(app)
    assert seen["exclude"] is True
    assert "output truncated — full output: C:/tmp/full.txt" in out
    assert "not sent to the model" in out


# ---------------------------------------------------------------------------
# InteractiveApp: sign-in flows
# ---------------------------------------------------------------------------


def test_suggest_login_skip_interrupt_and_success(tmp_path, monkeypatch):
    app = make_app(tmp_path)  # no model at all
    monkeypatch.setattr(coding_signin, "interactive_login", lambda **kwargs: None)
    app._suggest_login()
    out = output(app)
    assert "No model is available" in out
    assert "Skipped" in out

    # Ctrl+C during the menu exits the flow silently.
    def interrupted(**kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(coding_signin, "interactive_login", interrupted)
    app._suggest_login()

    # A successful sign-in reloads credentials and reports the switch.
    switched = resolve_model("anthropic/claude-sonnet-4-5")
    monkeypatch.setattr(coding_signin, "interactive_login", lambda **kwargs: "anthropic")
    monkeypatch.setattr(app.session, "reload_api_key", lambda: None)
    monkeypatch.setattr(app, "_switch_to_provider_if_unusable", lambda provider: switched)
    app._suggest_login()
    assert f"Signed in to {provider_label('anthropic')} — switched to " in output(app)

    monkeypatch.setattr(app, "_switch_to_provider_if_unusable", lambda provider: None)
    app._suggest_login()
    assert f"Signed in to {provider_label('anthropic')}." in output(app)


def test_suggest_login_when_model_has_no_credentials(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("anthropic/claude-sonnet-4-5"))
    monkeypatch.setattr(coding_signin, "interactive_login", lambda **kwargs: None)
    monkeypatch.setattr(app.session, "reload_api_key", lambda: None)
    app._suggest_login()
    assert "No provider credentials configured" in output(app)


def test_login_without_provider_suggests(tmp_path, monkeypatch):
    app = make_app(tmp_path)  # no model, no argument
    suggested: list[bool] = []
    monkeypatch.setattr(app, "_suggest_login", lambda: suggested.append(True))
    app._login("")
    assert suggested == [True]


def test_login_keyless_provider_without_catalog(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr(coding_signin, "ensure_models", lambda provider, settings, **kw: ([], None))
    monkeypatch.setattr(coding_signin, "default_model_for_provider", lambda provider: None)
    app._login("faux")
    assert "No faux models available" in output(app)


def test_login_stores_api_key_and_reports_each_outcome(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("anthropic/claude-sonnet-4-5"))
    stored: list[tuple[str, str]] = []
    monkeypatch.setattr(coding_signin, "store_api_key", lambda provider, key: stored.append((provider, key)))
    monkeypatch.setattr(app.session, "reload_api_key", lambda: None)

    # The signed-in provider is already active: the key is used immediately.
    monkeypatch.setattr(app, "_switch_to_provider_if_unusable", lambda provider: None)
    app._input = FakePrompt("sk-secret")
    app._login("anthropic")
    assert stored == [("anthropic", "sk-secret")]
    assert "using it now" in output(app)

    # A different provider: plain confirmation when no switch is needed.
    app._input = FakePrompt("sk-other")
    app._login("openai")
    assert "API key stored." in output(app)

    # An unusable current model is switched to the new provider.
    switched = resolve_model("openai/gpt-5")
    monkeypatch.setattr(app, "_switch_to_provider_if_unusable", lambda provider: switched)
    app._input = FakePrompt("sk-third")
    app._login("openai")
    assert f"switched to {format_model(switched)}" in output(app)

    # The key is handed to the session in all cases.
    assert stored == [("anthropic", "sk-secret"), ("openai", "sk-other"), ("openai", "sk-third")]


def test_login_aborts_on_blank_key_and_interrupt(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("anthropic/claude-sonnet-4-5"))
    stored: list[tuple[str, str]] = []
    monkeypatch.setattr(coding_signin, "store_api_key", lambda provider, key: stored.append((provider, key)))

    app._input = FakePrompt("   ")
    app._login("anthropic")
    app._input = FakePrompt(KeyboardInterrupt())
    app._login("anthropic")
    assert stored == []
    assert "Storing an API key for" in output(app)


def test_login_copilot_reports_readiness(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("anthropic/claude-sonnet-4-5"))
    monkeypatch.setattr(coding_signin, "copilot_login", lambda **kwargs: False)
    app._login_copilot()
    assert "GitHub Copilot" not in output(app)

    def fake_login(**kwargs: Any) -> bool:
        kwargs["echo"]("device code: ABCD")  # the flow talks through the renderer
        return True

    monkeypatch.setattr(coding_signin, "copilot_login", fake_login)
    monkeypatch.setattr(app.session, "reload_api_key", lambda: None)
    monkeypatch.setattr(app, "_switch_to_provider_if_unusable", lambda provider: None)
    app._login_copilot()
    assert "device code: ABCD" in output(app)
    assert "Use /model to pick one of its models." in output(app)

    switched = Model(id="copilot-model", name="Copilot Model", provider="github-copilot", api="github-copilot")
    monkeypatch.setattr(app, "_switch_to_provider_if_unusable", lambda provider: switched)
    app._login_copilot()
    assert f"switched to {format_model(switched)}" in output(app)


def test_switch_to_provider_if_unusable(tmp_path, monkeypatch):
    # A credential-free current provider (faux) never needs a switch.
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    assert app._switch_to_provider_if_unusable("anthropic") is None

    # No target available: explain why and keep the current state.
    app = make_app(tmp_path)  # no model at all
    monkeypatch.setattr(coding_signin, "ensure_models", lambda provider, settings, **kw: ([], "no catalog"))
    monkeypatch.setattr(coding_signin, "default_model_for_provider", lambda provider: None)
    assert app._switch_to_provider_if_unusable("anthropic") is None
    assert "Could not select a anthropic model" in output(app)

    # A usable target model is selected.
    target = resolve_model("anthropic/claude-sonnet-4-5")
    monkeypatch.setattr(coding_signin, "ensure_models", lambda provider, settings, **kw: ([target], None))
    monkeypatch.setattr(coding_signin, "default_model_for_provider", lambda provider: target)
    assert app._switch_to_provider_if_unusable("anthropic") is target
    assert app.session.model is target


# ---------------------------------------------------------------------------
# InteractiveApp: thinking, /session, logout, export
# ---------------------------------------------------------------------------


def make_persistent_app(tmp_path: Path, *, session_file: str = "session.jsonl") -> InteractiveApp:
    session = AgentSession(
        cwd=str(tmp_path), model=resolve_model("faux/test"), tool_names=[], session_file=str(tmp_path / session_file)
    )
    return InteractiveApp(session)


def append_exchange(app: InteractiveApp, user_text: str, assistant_text: str):
    user = app.session.session.append_message(UserMessage(content=user_text))
    assistant = AssistantMessage(content=[TextContent(text=assistant_text)])
    assistant.usage = Usage(input=1_000, output=200, cache_read=100, cache_write=50, total_tokens=1_200, cost=0.01)
    app.session.session.append_message(assistant)
    return user, assistant


def test_thinking_command_sets_level_and_reports_errors(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    monkeypatch.setattr(
        app.session, "available_thinking_levels", lambda: ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
    )

    app._handle_command("/thinking high")
    assert "Thinking level: high" in output(app)
    assert app.session.thinking == "high"
    changes = [e for e in app.session.session.entries if e.type == "thinking_level_change"]
    assert len(changes) == 1 and changes[0].data["thinkingLevel"] == "high"
    from tynex.coding.settings import read_settings_file, settings_path

    assert read_settings_file(settings_path())["thinking"] == "high"  # remembered for new sessions

    app._handle_command("/thinking HIGH")  # case-insensitive and unchanged: no new entry
    assert len([e for e in app.session.session.entries if e.type == "thinking_level_change"]) == 1

    app._handle_command("/thinking turbo")
    out = output(app)
    assert 'Unknown thinking level "turbo"' in out
    assert "off, minimal, low, medium, high, xhigh, max" in out

    app._handle_command("/thinking")  # non-terminal: lists instead of picking
    assert "available:" in output(app)


def test_thinking_command_rejects_levels_the_model_lacks(tmp_path):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    assert app.session.available_thinking_levels() == ["off"]

    app._handle_command("/thinking high")
    out = output(app)
    assert "needs a reasoning model" in out
    assert "faux/test does not support thinking" in out
    assert "/model" in out and "/thinking off" in out
    assert app.session.thinking == "off"
    assert not [e for e in app.session.session.entries if e.type == "thinking_level_change"]

    # Without a model at all the same question has a different answer.
    unset = make_app(tmp_path)
    assert unset.session.model is None
    unset._handle_command("/thinking high")
    assert "none is selected" in output(unset)


def test_thinking_levels_follow_the_catalog_and_clamp_to_the_nearest(tmp_path):
    # DeepSeek's catalog lists low/high/max; the old code offered only "off".
    model = Model(
        id="deepseek-flash",
        name="DeepSeek-V4.1-Flash",
        provider="deepseek",
        api="openai-completions",
        reasoning=True,
        thinking_levels=("low", "high", "max"),
    )
    app = make_app(tmp_path, model=model)
    assert app.session.available_thinking_levels() == ["off", "low", "high", "max"]

    app._handle_command("/thinking")  # non-terminal: lists instead of picking
    assert "available: off, low, high, max" in output(app)

    app._handle_command("/thinking medium")  # not offered: the nearest level wins
    out = output(app)
    assert "medium is not offered" in out and "using high" in out
    assert app.session.thinking == "high"


def test_session_command_reports_pi_stats(tmp_path):
    app = make_persistent_app(tmp_path)
    append_exchange(app, "hello", "hi")
    app.session.session.append_message(
        ToolResultMessage(tool_call_id="t1", tool_name="read", content=[TextContent(text="x")])
    )
    app.session.session.append_label("my label")

    app._handle_command("/session")
    out = output(app)
    assert "my label" in out
    assert app.session.session.session_id in out
    assert str(tmp_path / "session.jsonl") in out
    assert "3 total" in out
    assert "1 user" in out and "1 assistant" in out
    assert "1 tool results" in out
    assert "1.2k total" in out
    assert "$0.0100" in out


def test_budget_command_shows_limits_and_hides_unused_rows(tmp_path):
    from tynex.agent.budget import BudgetLimits

    app = make_app(tmp_path)
    app._handle_command("/budget")
    out = output(app)
    assert "No budget set" in out
    assert "Budget" not in out and "$0.0000" not in out  # no limits, no usage: nothing to show

    session = AgentSession(cwd=str(tmp_path), tool_names=[], no_session=True, budget=BudgetLimits(max_tokens=5))
    limited = InteractiveApp(session)
    limited._handle_command("/budget")
    out = output(limited)
    assert "tokens" in out and "0 / 5" in out
    assert "cost" not in out  # no cost limit


def test_logout_removes_stored_credentials(tmp_path):
    from tynex.coding.settings import load_auth, store_api_key

    store_api_key("anthropic", "sk-a")
    store_api_key("openai", "sk-b")
    app = make_app(tmp_path)

    app._handle_command("/logout anthropic")
    out = output(app)
    assert "Removed stored API key for anthropic" in out
    assert set(load_auth()) == {"openai"}

    app._handle_command("/logout anthropic")
    assert "No stored credentials for anthropic" in output(app)

    app._handle_command("/logout")  # non-terminal: only one provider left
    assert "Removed stored API key for openai" in output(app)
    assert load_auth() == {}

    app._handle_command("/logout")
    assert "No stored credentials to remove" in output(app)


def test_logout_reports_oauth_providers(tmp_path):
    from tynex.coding.settings import store_api_key

    store_api_key("github-copilot", "gho_test")
    app = make_app(tmp_path)
    app._handle_command("/logout github-copilot")
    assert "Logged out of GitHub Copilot" in output(app)


def test_export_writes_jsonl_and_html(tmp_path):
    import json

    app = make_persistent_app(tmp_path)
    append_exchange(app, "hello <world>", "hi there")
    app.session.session.append_message(
        BashExecutionMessage(command="echo hi", output="hi", exit_code=0, exclude_from_context=True)
    )

    jsonl_path = tmp_path / "export.jsonl"
    app._handle_command(f"/export {jsonl_path}")
    out = output(app)
    assert "Session exported to:" in out and "export.jsonl" in out
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines()]
    assert records[0]["type"] == "session"
    assert records[0]["id"] == app.session.session.session_id
    assert records[0]["cwd"] == app.session.cwd
    assert [r["type"] for r in records[1:]] == ["message", "message", "message"]
    assert records[1]["parentId"] == records[0]["id"]
    assert records[2]["parentId"] == records[1]["id"]

    html_path = tmp_path / "transcript.html"
    app._handle_command(f"/export {html_path}")
    assert "Session exported to:" in output(app)
    page = html_path.read_text(encoding="utf-8")
    assert "tynex session transcript" in page
    assert "hello &lt;world&gt;" in page
    assert f"{shell_prompt} echo hi" in page
    assert "not sent to the model" in page

    app._handle_command("/export")  # default HTML name, resolved against the session cwd
    assert list(tmp_path.glob("tynex-session-*.html"))


def test_run_prefills_the_editor(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    prompt = FakePrompt("edited question", EOFError())
    app._input = prompt
    monkeypatch.setattr(app, "_maybe_ask_login", lambda: None)
    seen: list[str] = []
    monkeypatch.setattr(app, "_handle_line", lambda line, echo_user=True: seen.append(line))
    app._prefill = "first question"

    assert app.run() == 0
    assert seen == ["edited question"]
    assert prompt.defaults == ["first question", ""]


# ---------------------------------------------------------------------------
# InteractiveApp: plan approval
# ---------------------------------------------------------------------------


def test_drain_plans_approves_once_and_skips_stale_events(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"), terminal=True)
    app.session.pending_plan = {"id": "plan-1", "plan": "## Plan"}
    app._plans.put({"type": "plan_ready", "planId": "plan-1", "plan": "## Plan"})
    app._plans.put({"type": "plan_ready", "planId": "plan-1", "plan": "## Plan"})  # already decided
    monkeypatch.setattr(app, "_ask", lambda text: "y")
    submitted: list[str] = []
    monkeypatch.setattr(app, "_submit", lambda text, **kwargs: submitted.append(text))

    app._drain_plans()

    assert app.session.permission_gate.mode == "accept-edits"
    assert submitted == ["Execute the approved plan."]


def test_submit_queues_until_the_active_run_finishes(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    started = threading.Event()
    release = threading.Event()
    seen: list[str] = []

    def fake_prompt(text: str, emit: Any = None) -> None:
        seen.append(text)
        started.set()
        release.wait(5)

    monkeypatch.setattr(app.session, "prompt", fake_prompt)
    app._submit("first")
    assert started.wait(5)
    app._submit("second")
    assert seen == ["first"]  # queued, not a concurrent session.prompt()
    assert app._busy()

    release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and (seen != ["first", "second"] or app._busy()):
        time.sleep(0.01)
    assert seen == ["first", "second"]
    assert not app._busy()


# ---------------------------------------------------------------------------
# InteractiveApp: tool approvals
# ---------------------------------------------------------------------------


def test_approval_drain_answers_queued_requests(tmp_path):
    app = make_app(tmp_path)
    entry = interactive._ApprovalEntry(PermissionRequest(tool_name="bash", kind="shell", subject="echo hi"))
    app._approvals.put(entry)
    asked: list[str] = []
    app._prompt_approval = lambda request: asked.append(request.summary()) or "session"

    app._drain_approvals()
    assert entry.done.is_set() and entry.result == "session"
    assert asked == ["bash: echo hi"]


def test_permission_gate_asks_through_the_app_from_the_agent_thread(tmp_path):
    app = make_app(tmp_path)
    app.session.permission_gate.set_mode("ask")
    app._prompt_approval = lambda request: "allow"
    decisions: list = []

    def check() -> None:
        decisions.append(app.session.permission_gate.check(ShellTool("."), {"command": "echo hi"}))

    worker = threading.Thread(target=check)
    worker.start()
    deadline = time.monotonic() + 10.0
    while app._approvals.empty() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not app._approvals.empty(), "the agent thread should have queued an approval"

    app._drain_approvals()  # what the prompt loop does before each prompt()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert decisions[0].allowed and decisions[0].origin == "approval"


def test_run_loop_skips_the_approval_sentinel(tmp_path, monkeypatch):
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    monkeypatch.setattr(app, "_maybe_ask_login", lambda: None)
    monkeypatch.setattr(app, "_run_prompt", lambda text, echo_user=True: None)
    app._input = FakePrompt(interactive.APPROVAL_SENTINEL, "/quit")

    assert app.run() == 0
    assert "bye" in output(app)


def test_tui_run_asks_for_approval_and_runs_the_tool(tmp_path, monkeypatch):
    from tynex.ai.providers import faux

    app = make_app(tmp_path, model=resolve_model("faux/test"), tool_names=[shell_name])
    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "toolcall", "id": "c1", "name": shell_name, "arguments": {"command": "echo tui-approved"}},
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("approved and done"),
        ]
    )
    asked: list[str] = []
    app._prompt_approval = lambda request: asked.append(request.summary()) or "allow"
    monkeypatch.setattr(app, "_refresh", lambda: None)  # no live prompt_toolkit app under pytest

    app._submit("run it")
    deadline = time.monotonic() + 20.0
    while app._busy() and time.monotonic() < deadline:
        app._drain_approvals()  # the prompt loop answers approvals while the agent runs
        time.sleep(0.02)
    assert not app._busy()
    assert asked and asked[0].startswith(f"{shell_name}: echo tui-approved")
    permissions = [entry for entry in app.session.session.entries if entry.type == "permission"]
    assert [entry.data["action"] for entry in permissions] == ["allow"]
    assert any("tui-approved" in str(getattr(message, "content", "")) for message in app.session.context.messages)


def test_prompt_approval_asks_on_the_input_line_and_maps_answers(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    answers = iter(["y", "yes", " a ", "always", "", "n"])
    monkeypatch.setattr(app, "_ask", lambda text: next(answers))
    request = PermissionRequest(tool_name="bash", kind="shell", subject="echo hi")

    assert [app._prompt_approval(request) for _ in range(6)] == ["allow", "allow", "session", "session", "deny", "deny"]
    assert "approval needed" in output(app)

    def interrupted(text: str) -> str:
        raise EOFError

    monkeypatch.setattr(app, "_ask", interrupted)
    assert app._prompt_approval(request) == "deny"


def test_approval_falls_back_to_stdin_without_a_prompt_session(tmp_path, monkeypatch):
    # Piped input and the window before the first prompt have no live editor, so
    # the question goes through Rich's input() instead.
    app = make_app(tmp_path)
    monkeypatch.setattr(app.console, "input", lambda *args, **kwargs: "a")
    request = PermissionRequest(tool_name="bash", kind="shell", subject="echo hi")
    assert app._approval_from_stdin(request) == "session"
    assert "approval needed" in output(app)

    def interrupted(*args: Any, **kwargs: Any) -> str:
        raise EOFError

    monkeypatch.setattr(app.console, "input", interrupted)
    assert app._approval_from_stdin(request) == "deny"


def test_run_exits_cleanly_when_the_prompt_returns_none(tmp_path):
    # A programmatic app.exit() without a result (an extension key binding, say)
    # hands None back to the loop instead of a line.
    app = make_app(tmp_path, model=resolve_model("faux/test"))
    app._input = FakePrompt(None)
    assert app.run() == 0
    assert "bye" in output(app)
