"""Tests for the docked prompt shell (transcript above the input)."""

import contextlib
import threading
import time
from typing import Any

import pytest
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput

from tynex.tui.docked import DockedPrompt
from tynex.tui.transcript import Transcript

TIMEOUT = 10.0


def wait_until(check: Any, what: str, timeout: float = TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def mouse(kind: MouseEventType, row: int = 0, button: MouseButton = MouseButton.NONE) -> MouseEvent:
    return MouseEvent(position=Point(x=0, y=row), event_type=kind, button=button, modifiers=frozenset())


@pytest.fixture
def docked():
    """Build prompts that share a pipe, closed at teardown."""
    stack = contextlib.ExitStack()

    def make(**kwargs: Any) -> DockedPrompt:
        pipe = stack.enter_context(create_pipe_input())
        return DockedPrompt(Transcript(), input=pipe, output=DummyOutput(), **kwargs)

    yield make
    stack.close()


class Session:
    """Run one prompt in a thread, sending keys from the test."""

    def __init__(self, prompt: DockedPrompt, text: Any = "", **kwargs: Any) -> None:
        self.prompt = prompt
        self.text = text
        self.kwargs = kwargs
        self.results: list[Any] = []
        self.thread = threading.Thread(target=self._run)

    def _run(self) -> None:
        try:
            self.results.append(self.prompt.prompt(self.text, **self.kwargs))
        except BaseException as exc:  # the test inspects what came back
            self.results.append(exc)

    def __enter__(self) -> "Session":
        self.thread.start()
        wait_until(lambda: self.prompt.app.is_running or bool(self.results), "the prompt to start")
        if self.results:  # surface the failure instead of timing out on it
            raise AssertionError(f"the prompt exited before any input: {self.results[0]!r}")
        return self

    def send(self, keys: str) -> None:
        self.prompt.app.input.send_text(keys)  # type: ignore[union-attr]

    def __exit__(self, *exc: Any) -> None:
        self.thread.join(TIMEOUT)  # the test already sent Enter
        if self.thread.is_alive():  # failure path: unblock, then report
            self.send("\x04")
            self.thread.join(TIMEOUT)
        assert not self.thread.is_alive(), "the prompt never returned"


def test_accepts_a_line_and_can_run_again(docked):
    prompt = docked()
    with Session(prompt) as session:
        session.send("hello\r")
    assert session.results == ["hello"]

    with Session(prompt) as session:
        session.send("again\r")
    assert session.results == ["again"]


def test_ctrl_d_on_an_empty_line_raises_eof(docked):
    with Session(docked()) as session:
        session.send("\x04")
    assert isinstance(session.results[0], EOFError)


def test_ctrl_d_with_text_does_not_exit(docked):
    with Session(docked()) as session:
        session.send("abc\x04\r")
    assert session.results == ["abc"]


def test_ctrl_c_interrupts(docked):
    with Session(docked()) as session:
        session.send("\x03")
    assert isinstance(session.results[0], KeyboardInterrupt)


def test_the_question_replaces_the_prompt_glyph(docked):
    prompt = docked()
    with Session(prompt, "allow? y=once ") as session:
        assert prompt._prompt_text() == "allow? y=once "
        session.send("y\r")
    assert session.results == ["y"]


def test_click_expands_a_tool_block_and_the_wheel_scrolls(docked):
    prompt = docked()
    block = prompt.transcript.start_tool("bash  echo hi")
    prompt.transcript.finish_tool(block, summary="hi", detail="input: echo hi\noutput: hi")

    with Session(prompt) as session:
        wait_until(lambda: bool(prompt.view._view), "the transcript to render")
        assert not block.expanded
        assert prompt.view.mouse_handler(mouse(MouseEventType.MOUSE_UP, button=MouseButton.LEFT)) is None
        assert block.expanded
        assert "output: hi" in prompt.transcript.text()

        prompt.view.mouse_handler(mouse(MouseEventType.MOUSE_UP, button=MouseButton.LEFT))
        assert not block.expanded

        prompt.view.mouse_handler(mouse(MouseEventType.SCROLL_UP))
        assert prompt.view.scroll > 0
        prompt.view.mouse_handler(mouse(MouseEventType.SCROLL_DOWN))
        assert prompt.view.scroll == 0

        session.send("\r")
    assert session.results == [""]


def test_click_on_plain_text_does_nothing(docked):
    prompt = docked()
    prompt.transcript.write("just a notice\n")
    with Session(prompt) as session:
        wait_until(lambda: bool(prompt.view._view), "the transcript to render")
        assert prompt.view.mouse_handler(mouse(MouseEventType.MOUSE_UP, button=MouseButton.LEFT)) is NotImplemented
        session.send("\r")
    assert session.results == [""]


def test_scrolling_is_bounded_by_the_content(docked):
    prompt = docked()
    prompt.transcript.write("only line\n")
    prompt.view.scroll_by(100)
    prompt.view.create_content(40, 10)
    assert prompt.view.scroll == 0  # nothing above the first line

    prompt.transcript.write("\n".join(f"line {index}" for index in range(50)) + "\n")
    prompt.view.scroll_by(10)
    prompt.view.create_content(40, 10)
    assert prompt.view.scroll == 10

    prompt.view.scroll_by(1_000)
    prompt.view.create_content(40, 10)
    assert prompt.view.scroll == 41  # clamped to the top, not past it


def test_ctrl_o_asks_the_app_to_expand(docked):
    calls: list[str] = []
    prompt = docked(on_expand=lambda: calls.append("expand"))
    with Session(prompt) as session:
        session.send("\x0f")  # Ctrl+O
        wait_until(lambda: calls == ["expand"], "Ctrl+O to expand")
        session.send("done\r")
    assert session.results == ["done"]
