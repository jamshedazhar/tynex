"""Docked prompt: the transcript above the input, in one prompt_toolkit Application.

``PromptSession`` can only print finished output *above* its own region, and a
scrollback line cannot be clicked. This module owns the whole bottom region —
transcript, status, input, toolbar — so tool blocks are real widgets: clicking
one expands its full input and output, and Ctrl+O expands the last one.
"""

from collections.abc import Callable
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import AnyFormattedText, FormattedText
from prompt_toolkit.history import History
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyBindingsBase, merge_key_bindings
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput, ConditionalProcessor, PasswordProcessor
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style

from tynex.tui.prompt import _prompt_key_bindings
from tynex.tui.transcript import Transcript

COMPLETION_MENU_ROWS = 5
CHROME_ROWS = 5  # status + input + toolbar + breathing room
SCROLL_STEP = 3


class TranscriptControl(UIControl):
    """Scrollable, clickable view of the transcript."""

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self.scroll = 0  # lines scrolled up from the tail
        self._view: list[tuple[int, Any]] = []
        self._top = 0

    def is_focusable(self) -> bool:
        return False

    def reset_scroll(self) -> None:
        self.scroll = 0

    def scroll_by(self, delta: int) -> None:
        self.scroll = max(0, self.scroll + delta)

    def create_content(self, width: int, height: int) -> UIContent:
        lines = self.transcript.lines(width)
        height = max(1, height)
        # Scrolling is bounded by the content, so PgUp past the top does not
        # need an equal number of PgDowns to come back.
        self.scroll = min(self.scroll, max(0, len(lines) - height))
        self._top = max(0, len(lines) - height - self.scroll)
        self._view = lines[self._top : self._top + height]

        # Lines wrap, so serve the whole transcript and let the cursor (the last
        # visible line) decide what is on screen; a sliced view would run out of
        # lines as soon as one is taller than a terminal row.
        def get_line(index: int) -> AnyFormattedText:
            # An empty transcript still gets asked for line 0; serve a blank one.
            return lines[index][1] if 0 <= index < len(lines) else FormattedText()

        return UIContent(
            get_line=get_line,
            line_count=len(lines),
            cursor_position=Point(x=0, y=max(0, self._top + len(self._view) - 1)),
        )

    def block_at(self, row: int) -> Any:
        index = self._top + row
        lines = self.transcript.lines(self.transcript.width)
        if not 0 <= index < len(lines):
            return None
        block_index = lines[index][0]
        return self.transcript.blocks[block_index] if block_index < len(self.transcript.blocks) else None

    def mouse_handler(self, event: MouseEvent) -> Any:
        if event.event_type == MouseEventType.SCROLL_UP:
            self.scroll_by(SCROLL_STEP)
            return None
        if event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll_by(-SCROLL_STEP)
            return None
        if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
            block = self.block_at(event.position.y)
            if block is not None and self.transcript.toggle(block):
                return None
        return NotImplemented


class DockedPrompt:
    """A prompt_toolkit application with the transcript docked above the input."""

    def __init__(
        self,
        transcript: Transcript,
        *,
        history: History | None = None,
        completer: Any = None,
        auto_suggest: Any = None,
        style: Style | None = None,
        key_bindings: KeyBindingsBase | None = None,
        prompt: Callable[[], AnyFormattedText] | None = None,
        status: Callable[[], AnyFormattedText] | None = None,
        toolbar: Callable[[], AnyFormattedText] | None = None,
        placeholder: AnyFormattedText | None = None,
        on_expand: Callable[[], None] | None = None,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.transcript = transcript
        self.is_password = False
        self._default_prompt = prompt or (lambda: "❯ ")  # noqa: RUF001
        self._prefix: str | Callable[[], AnyFormattedText] | None = None
        self._status = status or (lambda: "")
        self._toolbar = toolbar or (lambda: "")
        self._on_expand = on_expand
        self.view = TranscriptControl(transcript)
        self.default_buffer = Buffer(
            name=DEFAULT_BUFFER,  # the shared key bindings filter on it
            history=history,
            completer=completer,
            auto_suggest=auto_suggest,
            complete_while_typing=True,
            multiline=True,
            accept_handler=self._accept,
        )
        self.default_buffer.placeholder = placeholder
        layout = self._build_layout(key_bindings)
        self.app: Application[Any] = Application(
            layout=layout,
            key_bindings=self._bindings,
            style=style,
            full_screen=False,
            mouse_support=True,
            erase_when_done=False,
            input=input,
            output=output,
        )

    # -- layout --------------------------------------------------------------
    def _prompt_text(self) -> AnyFormattedText:
        return self._prefix if self._prefix is not None else self._default_prompt()

    def _build_layout(self, extra: KeyBindingsBase | None) -> Layout:
        bindings = KeyBindings()

        @bindings.add("c-d")
        def _eof(event: Any) -> None:
            if not self.default_buffer.text:
                event.app.exit(result=None)

        @bindings.add("c-c")
        def _interrupt(event: Any) -> None:
            event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

        @bindings.add("c-o")
        def _expand(event: Any) -> None:
            del event
            if self._on_expand is not None:
                self._on_expand()

        @bindings.add("pageup")
        def _page_up(event: Any) -> None:
            del event
            self.view.scroll_by(10)

        @bindings.add("pagedown")
        def _page_down(event: Any) -> None:
            del event
            self.view.scroll_by(-10)

        # Enter/Esc+Enter live in the shared prompt bindings; the app passes its
        # own set (they also carry the interrupt bindings), tests get the default.
        extra = extra if extra is not None else _prompt_key_bindings()
        self._bindings = merge_key_bindings([bindings, extra])
        self._input = BufferControl(
            buffer=self.default_buffer,
            input_processors=[
                BeforeInput(self._prompt_text),
                ConditionalProcessor(PasswordProcessor(), Condition(lambda: self.is_password)),
            ],
        )
        body = HSplit(
            [
                Window(
                    self.view,
                    height=lambda: Dimension(min=1, max=self._transcript_rows()),
                    wrap_lines=True,
                    dont_extend_height=True,
                ),
                Window(FormattedTextControl(self._status), height=1, style="class:status", dont_extend_height=True),
                Window(self._input, height=Dimension(min=1, max=8), dont_extend_height=True),
                Window(
                    FormattedTextControl(self._toolbar), height=1, style="class:bottom-toolbar", dont_extend_height=True
                ),
            ]
        )
        floats = FloatContainer(
            content=body,
            floats=[
                Float(CompletionsMenu(max_height=COMPLETION_MENU_ROWS, display_arrows=True), xcursor=True, ycursor=True)
            ],
        )
        return Layout(floats, focused_element=self.default_buffer)

    @staticmethod
    def _transcript_rows() -> int:
        try:
            rows = get_app().output.get_size().rows
        except Exception:  # pragma: no cover - app not running yet
            rows = 24
        return max(3, rows - CHROME_ROWS)

    # -- prompt API (mirrors PromptSession) -----------------------------------
    def _accept(self, buffer: Buffer) -> bool:
        self.app.exit(result=buffer.text)
        return True

    def prompt(
        self, text: AnyFormattedText = "", *, is_password: bool = False, default: str = "", bottom_toolbar: Any = None
    ) -> str:
        """Ask one line. The toolbar is part of the layout, so it is ignored."""
        del bottom_toolbar
        self._prefix = text or None
        self.is_password = is_password
        self.default_buffer.reset(Document(default or ""))
        self.view.reset_scroll()
        try:
            result = self.app.run()
        finally:
            self.is_password = False
        if result is None:  # Ctrl+D on an empty line
            raise EOFError
        return str(result)

    def invalidate(self) -> None:
        self.app.invalidate()
