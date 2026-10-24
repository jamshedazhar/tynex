"""End-to-end TUI scenarios driven with real keystrokes.

``test_interactive.py`` calls ``InteractiveApp`` methods one at a time. These
tests own the real prompt session and feed keys through a pipe, so a change to
key bindings, completion, the run loop or a command's output shows up the way a
user sees it: type a line, then read the transcript.

``COMMAND_SCENARIOS`` is the coverage contract. Every entry in ``COMMANDS`` must
have one (see ``test_every_command_has_an_end_to_end_scenario``), so a new
command cannot ship without an automated check, and ``/help`` never lists a
command nobody ever ran through the prompt.
"""

import contextlib
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from tests.conftest import faux_scenario, text_scenario
from tynex.ai.models import resolve_model
from tynex.ai.types import ToolResultMessage, UserMessage
from tynex.coding.agent_session import AgentSession
from tynex.coding.tools.shell import shell_name
from tynex.tui.app import InteractiveApp
from tynex.tui.prompt import ALIASES, COMMANDS, TynexCompleter
from tynex.tui.select import select_option

TIMEOUT = 20.0
SETTLE = 0.05  # the editor must stop changing for this long before we press a key
WIDTH = 200  # wide enough that Rich never wraps an assertion target
ENTER = "\r"
TAB = "\t"
ESCAPE = "\x1b"
CTRL_D = "\x04"
CTRL_O = "\x0f"


def wait_until(check: Callable[[], bool], what: str, timeout: float = TIMEOUT) -> None:
    """Poll ``check`` until it holds. State-based waits only: no fixed sleeps."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out after {timeout:g}s waiting for {what}")


def menu_ready(buffer: Any) -> bool:
    state = buffer.complete_state
    return state is not None and bool(state.completions) and state.original_document.text == buffer.text


def nothing_to_add(text: str, completion: Any) -> bool:
    """True when applying ``completion`` would leave ``text`` unchanged."""
    return text[len(text) + completion.start_position :] == completion.text


class RecordingOutput(DummyOutput):
    """Dummy output that remembers writes made after the prompt stopped."""

    def __init__(self) -> None:
        super().__init__()
        self.written: list[str] = []

    def write(self, data: str) -> None:
        self.written.append(data)


class Ui:
    """The interactive app, run in its own thread and driven over a pipe."""

    def __init__(
        self, cwd: Any, *, model: str | None = "faux/test", tool_names: list[str] | None = None, mode: str | None = None
    ) -> None:
        session = AgentSession(
            cwd=str(cwd),
            model=resolve_model(model) if model else None,
            tool_names=tool_names or [],
            no_session=True,
            # The same keyword ``--permission-mode`` wires into, so the real
            # precedence chain (flag > settings > permissions.toml) is exercised.
            permission_mode=mode,
        )
        # Not a terminal: selectors and confirmations use their text path, so no
        # handler blocks on a real terminal. Everything the app prints lands in
        # its transcript, which is what the assertions read.
        self.app = InteractiveApp(session)
        self.exit_code: int | None = None
        self.output = RecordingOutput()
        self._stack = contextlib.ExitStack()
        self._pipe = self._stack.enter_context(create_pipe_input())
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> "Ui":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.wait(lambda: self.prompt_running, "the prompt to start")
        return self

    def _run(self) -> None:
        # The app session must be created inside this thread: prompt_toolkit
        # keeps it in a contextvar, which does not cross threads.
        with create_app_session(input=self._pipe, output=self.output):
            self.exit_code = self.app.run()

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            with contextlib.suppress(Exception):
                self._pipe.send_text(CTRL_D)
            self._thread.join(5)
        if self._thread is not None and self._thread.is_alive():
            with contextlib.suppress(Exception):
                self._pipe.send_text("\x03")
            self._thread.join(5)
        self._stack.close()

    # -- observation ---------------------------------------------------------
    @property
    def prompt_running(self) -> bool:
        prompt = self.app._input  # don't build a session outside the app context
        return prompt is not None and prompt.app.is_running

    @property
    def buffer(self) -> Any:
        return self.app.input.default_buffer

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def transcript(self) -> str:
        return self.app.transcript.text(WIDTH)

    def terminal_text(self) -> str:
        return "".join(self.output.written)

    def wait(self, check: Callable[[], bool], what: str, timeout: float = TIMEOUT) -> None:
        try:
            wait_until(check, what, timeout)
        except AssertionError:
            raise AssertionError(f"{what}\n--- transcript ---\n{self.transcript()}") from None

    def wait_for(self, text: str) -> None:
        self.wait(lambda: text in self.transcript(), f"{text!r} in the transcript")

    def wait_idle(self) -> None:
        self.wait(lambda: not self.app._busy(), "the agent run to finish")

    def _settle(self) -> None:
        """Wait until the editor stops changing (completion runs as a task)."""
        last: Any = None
        since = time.monotonic()
        deadline = since + TIMEOUT
        while time.monotonic() < deadline:
            state = self.buffer.complete_state
            snapshot = (
                self.buffer.text,
                state.complete_index if state else None,
                len(state.completions) if state else 0,
            )
            now = time.monotonic()
            if snapshot != last:
                last, since = snapshot, now
            elif now - since >= SETTLE:
                return
            time.sleep(0.005)
        raise AssertionError("the editor never settled")

    # -- input ---------------------------------------------------------------
    def send(self, keys: str) -> None:
        self._pipe.send_text(keys)

    def type(self, text: str) -> None:
        """Type ``text`` and wait for the editor (and its completion menu)."""
        self.send(text)
        self.wait(lambda: self.buffer.text == text, f"{text!r} in the editor")
        self._settle()
        expected = list(TynexCompleter(self.app.session).get_completions(Document(text), None))
        # prompt_toolkit drops a lone completion that would not change the text
        # (typing "/help" exactly), so the menu stays closed for those.
        if len(expected) == 1 and nothing_to_add(text, expected[0]):
            expected = []
        if expected:
            self.wait(lambda: menu_ready(self.buffer), f"the completion menu for {text!r}")
        else:
            self.wait(lambda: self.buffer.complete_state is None, f"no completion menu for {text!r}")

    def enter(self) -> None:
        """Press Enter like a user: an open menu fills first, then submits."""
        buffer = self.buffer
        for _ in range(4):
            before = buffer.text
            self.send(ENTER)
            # Accepting resets the editor or closes the prompt; an open menu
            # only fills the completion in, leaving the prompt live.
            self.wait(lambda typed=before: buffer.text != typed or not self.prompt_running, "Enter to take effect")
            if buffer.text == "" or not self.prompt_running:
                break
        else:  # pragma: no cover - defensive: a menu that never settles
            raise AssertionError("the line was never submitted")
        self.wait(lambda: self.prompt_running or not self.alive, "the next prompt")

    def run_line(self, line: str) -> None:
        self.type(line)
        self.enter()

    def answer(self, text: str) -> None:
        """Type one line at a nested question (an approval, a sign-in key)."""
        self.send(text + ENTER)


# ---------------------------------------------------------------------------
# The command coverage contract
# ---------------------------------------------------------------------------

# name -> (lines to type, text that must appear in the transcript afterwards)
COMMAND_SCENARIOS: dict[str, tuple[list[str], str]] = {
    "help": (["/help"], "Commands"),
    "model": (["/model faux/test"], "Switched to"),
    "thinking": (["/thinking off"], "Thinking level: off"),
    "compact": (["/compact"], "Nothing to compact."),
    "new": (["/new"], "New session"),
    "resume": (["/resume"], "No sessions for this directory."),
    "export": (["/export transcript.html"], "Session exported to:"),
    "name": (["/name demo"], "Session named 'demo'"),
    "session": (["/session"], "Messages"),
    "permissions": (["/permissions"], "No rules"),
    "plan": (["/plan"], "Plan mode"),
    "approve": (["/approve"], "No plan is waiting for approval."),
    "budget": (["/budget"], "No budget set"),
    "undo": (["/undo"], "Nothing to undo."),
    "mcp": (["/mcp"], "No MCP servers connected"),
    "login": (["/login anthropic", "sk-e2e-secret"], "API key stored."),
    "logout": (["/logout"], "No stored credentials to remove."),
    "skills": (["/skills"], "code-review"),
    "cwd": (["/cwd"], "__CWD__"),
    "quit": (["/quit"], "bye"),
}


@pytest.fixture
def ui(tmp_path):
    """Factory: each call starts an app on its own pipe, closed at teardown."""
    started: list[Ui] = []

    def start(**kwargs: Any) -> Ui:
        app = Ui(tmp_path, **kwargs).start()
        started.append(app)
        return app

    yield start
    for app in started:
        app.close()


@pytest.mark.parametrize("name", sorted(COMMAND_SCENARIOS))
def test_command_through_the_prompt(name, ui, tmp_path):
    lines, expected = COMMAND_SCENARIOS[name]
    app = ui()
    for line in lines:
        app.run_line(line)
    app.wait_for(str(tmp_path) if expected == "__CWD__" else expected)
    if name == "quit":
        app.wait(lambda: not app.alive, "the app to exit")
        assert app.exit_code == 0


@pytest.mark.parametrize("alias", sorted(ALIASES))
def test_alias_through_the_prompt(alias, ui):
    target = ALIASES[alias]
    app = ui()
    app.run_line(f"/{alias}")
    app.wait_for(COMMAND_SCENARIOS[target][1])


def test_every_command_has_an_end_to_end_scenario():
    missing = sorted(set(COMMANDS) - set(COMMAND_SCENARIOS))
    assert not missing, f"add an end-to-end scenario for: {missing}"
    stale = sorted(set(COMMAND_SCENARIOS) - set(COMMANDS))
    assert not stale, f"these scenarios no longer match a command: {stale}"


# ---------------------------------------------------------------------------
# A real turn
# ---------------------------------------------------------------------------


def test_plain_message_streams_the_reply(ui):
    from tynex.ai.providers import faux

    faux.reset_script([text_scenario("Hello from the faux model.")])
    app = ui()
    app.run_line("hi there")
    app.wait_for("Hello from the faux model.")
    app.wait_idle()
    texts = [message.text() for message in app.app.session.context.messages if isinstance(message, UserMessage)]
    assert texts == ["hi there"]
    assert "Hi there" not in app.transcript()  # typed lines are not echoed twice


def test_permissions_switches_without_the_mode_keyword(ui):
    # Reported: `/permissions bypass` reprinted the summary and changed nothing.
    app = ui()
    app.run_line("/permissions bypass")
    app.wait_for("Permission mode: bypass")
    assert app.app.session.permission_gate.mode == "bypass"

    app.run_line("/permissions")  # the listing now shows the new mode
    app.wait_for("Permissions mode bypass")

    app.run_line("/permissions mode plan")  # the long form still works
    app.wait_for("Permission mode: plan")
    assert app.app.session.permission_gate.mode == "plan"

    app.run_line("/permissions yolo")
    app.wait_for("Unknown permission mode")
    assert app.app.session.permission_gate.mode == "plan"


def test_plan_command_switches_and_explains_the_gate(ui):
    app = ui()
    app.run_line("/plan")
    app.wait_for("Plan mode — read-only")
    assert app.app.session.permission_gate.mode == "plan"
    assert "/approve executes it" in app.transcript()

    app.run_line("/plan")
    app.wait_for("Already in plan mode")
    assert app.app.session.permission_gate.mode == "plan"


def test_thinking_high_explains_a_model_without_reasoning(ui):
    # Reported verbatim: "/thinking high" used to answer "not available for this
    # model (choose off)." with no hint about why or what to do.
    app = ui()  # faux/test has no reasoning mode
    app.run_line("/thinking high")
    app.wait_for("needs a reasoning model")
    assert "faux/test does not support thinking" in app.transcript()
    assert "/model" in app.transcript() and "/thinking off" in app.transcript()
    assert app.app.session.thinking == "off"


def test_tab_accepts_the_completion_and_submits(ui):
    app = ui()
    app.type("/he")
    assert menu_ready(app.buffer)  # the menu opened while typing
    app.send(TAB)
    app.wait(lambda: app.buffer.text == "/help", "Tab to accept the highlighted completion")
    app.enter()
    app.wait_for("Commands")


def test_at_mention_completes_a_path_and_enter_submits(ui, tmp_path):
    from tynex.ai.providers import faux

    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    faux.reset_script([text_scenario("read it")])
    app = ui()
    app.type("@REA")
    assert menu_ready(app.buffer)  # Enter must complete the open menu first
    app.enter()
    app.wait_for("read it")
    app.wait_idle()
    texts = [message.text() for message in app.app.session.context.messages if isinstance(message, UserMessage)]
    assert texts == ["@README.md"]


def test_alt_enter_inserts_a_newline(ui):
    from tynex.ai.providers import faux

    faux.reset_script([text_scenario("got two lines")])
    app = ui()
    app.type("first line")
    app.send(ESCAPE + ENTER)  # Alt+Enter
    app.wait(lambda: app.buffer.text == "first line\n", "Alt+Enter to insert a newline")
    app.send("second line")
    app.wait(lambda: app.buffer.text == "first line\nsecond line", "the second line in the editor")
    app.enter()
    app.wait_for("got two lines")
    app.wait_idle()
    texts = [message.text() for message in app.app.session.context.messages if isinstance(message, UserMessage)]
    assert texts == ["first line\nsecond line"]


def test_plan_is_submitted_then_approved_from_the_prompt(ui, tmp_path):
    """Whole gate through the real prompt loop: submit, decision, execution."""
    from tynex.ai.providers import faux

    plan = "## Goal\nAdd the helper."
    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "toolcall", "id": "call_1", "name": "submit_plan", "arguments": {"plan": plan}},
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            faux_scenario(
                [
                    {
                        "type": "toolcall",
                        "id": "call_2",
                        "name": "write",
                        "arguments": {"path": "app.py", "content": "x"},
                    },
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("done"),
        ]
    )
    app = ui(tool_names=["read", "write", "submit_plan"], mode="plan")

    app.run_line("plan the change")
    app.wait_for("plan submitted for approval")
    app.wait_for("Plan recorded. Run /approve")
    assert app.app.session.pending_plan is not None
    assert not (tmp_path / "app.py").exists()

    app.run_line("/approve")
    app.wait_for("Plan approved")
    assert app.app.session.permission_gate.mode == "accept-edits"

    app.run_line("execute it")
    app.wait_for("done")
    app.wait_idle()
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x"


def test_ctrl_d_on_an_empty_line_exits(ui):
    app = ui()
    app.send(CTRL_D)
    app.wait(lambda: not app.alive, "the app to exit")
    assert app.exit_code == 0
    assert "bye" in app.transcript()
    assert "bye" in app.terminal_text()


# ---------------------------------------------------------------------------
# While a turn is running
# ---------------------------------------------------------------------------


def blocking_turn(app: Ui, started: threading.Event, release: threading.Event) -> Callable[..., None]:
    """Replace ``session.prompt`` with a turn that stays active until released."""

    def prompt(text: str, emit: Any = None) -> None:
        del text, emit
        started.set()
        release.wait(TIMEOUT)
        app.app.session._running = False

    return prompt


def test_typing_while_a_turn_runs_queues_the_message(ui, monkeypatch):
    from tynex.ai.types import AbortSignal

    started, release = threading.Event(), threading.Event()
    app = ui()
    monkeypatch.setattr(app.app.session, "prompt", blocking_turn(app, started, release))
    app.app.session._signal = AbortSignal()

    app.run_line("first message")
    assert started.wait(TIMEOUT)
    app.run_line("queued note")
    app.wait_for("queued")
    assert app.app.session.pending_steering == 1

    release.set()
    app.wait_idle()


def test_escape_interrupts_a_running_turn(ui, monkeypatch):
    from tynex.ai.types import AbortSignal

    started, release = threading.Event(), threading.Event()
    app = ui()
    monkeypatch.setattr(app.app.session, "prompt", blocking_turn(app, started, release))
    app.app.session._signal = AbortSignal()

    app.run_line("long job")
    assert started.wait(TIMEOUT)
    app.send(ESCAPE)
    app.wait(lambda: app.app.session.signal is not None and app.app.session.signal.aborted, "Esc to abort the turn")
    app.wait_for("Interrupting")
    release.set()
    app.wait_idle()


# ---------------------------------------------------------------------------
# Tool approvals through the real loop
# ---------------------------------------------------------------------------


def test_approval_panel_answers_and_runs_the_tool(ui):
    from tynex.ai.providers import faux

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
    app = ui(tool_names=[shell_name], mode="ask")
    assert "permissions ask" in app.transcript()  # the banner shows the active mode
    app.run_line("run it")
    app.wait_for("approval needed")
    app.answer("y")  # typed into the approval question on the input line
    app.wait_for("approved and done")
    app.wait_idle()

    decisions = [entry for entry in app.app.session.session.entries if getattr(entry, "type", "") == "permission"]
    assert [entry.data["action"] for entry in decisions] == ["allow"]
    assert "tui-approved" in app.transcript()


def test_session_approval_skips_the_second_question(ui):
    from tynex.ai.providers import faux

    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "toolcall", "id": "c1", "name": shell_name, "arguments": {"command": "echo one"}},
                    {"type": "toolcall", "id": "c2", "name": shell_name, "arguments": {"command": "echo two"}},
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("both ran"),
        ]
    )
    app = ui(tool_names=[shell_name], mode="ask")
    app.run_line("run both")
    app.wait_for("approval needed")
    app.answer("a")  # allow for the rest of the session
    app.wait_for("both ran")
    app.wait_idle()
    assert app.transcript().count("approval needed") == 1, "a session grant must not ask again"
    assert app.app.session.permission_gate.session_grants() == [shell_name]
    assert "echo two" in app.transcript()


def test_denied_approval_does_not_run_the_tool(ui):
    from tynex.ai.providers import faux

    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "toolcall", "id": "c1", "name": shell_name, "arguments": {"command": "echo nope"}},
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("understood"),
        ]
    )
    app = ui(tool_names=[shell_name], mode="ask")
    app.run_line("run it")
    app.wait_for("approval needed")
    app.answer("n")
    app.wait_for("understood")
    app.wait_idle()
    errors = [
        message
        for message in app.app.session.context.messages
        if isinstance(message, ToolResultMessage) and message.is_error
    ]
    assert errors and "denied by the user" in errors[0].content[0].text


def test_plan_mode_denies_the_shell_without_asking(ui):
    """A read-only mode must hold for the whole run, with no approval prompt."""
    from tynex.ai.providers import faux

    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "toolcall", "id": "c1", "name": shell_name, "arguments": {"command": "echo planned"}},
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("here is the plan"),
        ]
    )
    app = ui(tool_names=[shell_name], mode="plan")
    assert "permissions plan" in app.transcript()
    app.run_line("run it")
    app.wait_for("here is the plan")
    app.wait_idle()
    assert "approval needed" not in app.transcript()
    errors = [
        message
        for message in app.app.session.context.messages
        if isinstance(message, ToolResultMessage) and message.is_error
    ]
    assert errors and "plan mode is read-only" in errors[0].content[0].text


# ---------------------------------------------------------------------------
# Tool blocks: collapsed summary, full input and output once expanded
# ---------------------------------------------------------------------------


def test_tool_output_expands_with_ctrl_o_and_a_click(ui):
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

    from tynex.ai.providers import faux

    long_output = "\n".join(f"output line {index}" for index in range(1, 41))
    faux.reset_script(
        [
            faux_scenario(
                [
                    {
                        "type": "toolcall",
                        "id": "c1",
                        "name": shell_name,
                        "arguments": {"command": 'for i in $(seq 1 40); do echo "output line $i"; done'},
                    },
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            # The run ends with a visible reply, so the assertions wait on text
            # that only appears after the tool block was finished.
            text_scenario(f"done: {long_output.count(chr(10)) + 1} lines"),
        ]
    )
    app = ui(tool_names=[shell_name], mode="bypass")
    app.run_line("run it")
    app.wait_for("done")
    app.wait_idle()

    # Collapsed: one summary line (the last line of output), nothing more.
    collapsed = app.transcript()
    assert "output line 20" not in collapsed and "output line 10" not in collapsed
    assert "click or ctrl+o to expand" in collapsed

    # Ctrl+O expands the last block: the whole input and output are there.
    app.send(CTRL_O)
    app.wait_for("output line 20")  # a middle line: only the expanded view has it
    expanded = app.transcript()
    assert "output line 1" in expanded and "output line 20" in expanded and "output line 39" in expanded
    assert "seq 1 40" in expanded  # the full command, not the shortened header
    assert "click or ctrl+o to collapse" in expanded
    assert app.app.transcript.last_tool().expanded is True

    # A click on the block toggles it back, without touching the editor text.
    view = app.app.input.view
    lines = app.app.transcript.lines(app.app.transcript.width)
    tool_index = next(index for index, block in enumerate(app.app.transcript.blocks) if block.kind == "tool")
    row = next(row for row, (block, _line) in enumerate(lines) if block == tool_index)
    view._top = row
    typed_before = app.buffer.text
    view.mouse_handler(
        MouseEvent(position=Point(x=0, y=0), event_type=MouseEventType.MOUSE_UP, button=MouseButton.LEFT, modifiers=0)
    )
    app.wait(lambda: "output line 20" not in app.transcript(), "the click to collapse the block")
    assert app.buffer.text == typed_before


# ---------------------------------------------------------------------------
# The arrow-key selector (select.py) with real keys
# ---------------------------------------------------------------------------


def drive_select(keys: list[str], **kwargs: Any) -> Any:
    """Run ``select_option`` on a pipe and return its result."""
    with create_pipe_input() as pipe:
        results: list[Any] = []
        thread = threading.Thread(
            target=lambda: results.append(
                select_option("Pick", ["alpha", "beta", "gamma"], input=pipe, output=DummyOutput(), **kwargs)
            )
        )
        thread.start()
        for chunk in keys:
            pipe.send_text(chunk)
        thread.join(TIMEOUT)
        assert not thread.is_alive(), "select_option never returned"
        return results[0]


def test_select_option_arrows_and_wrapping():
    assert drive_select(["\x1b[B", ENTER]) == 1  # down
    assert drive_select(["\x1b[A", ENTER]) == 2  # up wraps to the last row
    assert drive_select([ENTER], selected=2) == 2  # preselected row
    assert drive_select([ESCAPE]) is None  # cancel


def test_select_option_vim_keys():
    assert drive_select(["j", "j", ENTER]) == 2
    assert drive_select(["j", "k", ENTER]) == 0


def test_select_option_cancels_on_ctrl_c():
    assert drive_select(["\x03"]) is None


def test_select_option_without_options_returns_none():
    assert select_option("Pick", []) is None
