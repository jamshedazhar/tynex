"""Transcript model for the docked TUI."""

import io
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from prompt_toolkit.formatted_text import ANSI, FormattedText, to_formatted_text
from prompt_toolkit.formatted_text.utils import to_plain_text
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text as RichText

# SGR sequences and OSC-8 links: enough to turn Rich output back into plain text.
_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*(?:\x07|\x1b\\)")

EXPAND_HINT = "click or ctrl+o to expand"
COLLAPSE_HINT = "click or ctrl+o to collapse"
INDENT = "  "


def strip_ansi(text: str) -> str:
    return _ESCAPE.sub("", text)


def to_lines(ansi: str) -> list[FormattedText]:
    """Split Rich's ANSI output into one ``FormattedText`` per rendered line."""
    raw = ansi.split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    return [to_formatted_text(ANSI(line)) for line in raw]


def indent(line: FormattedText, prefix: str = INDENT) -> FormattedText:
    return FormattedText([("", prefix), *line])


def render(renderable: object, width: int) -> str:
    """Render a Rich renderable to ANSI at ``width``."""
    buffer = io.StringIO()
    Console(
        file=buffer,
        width=max(20, width),
        force_terminal=True,
        color_system="truecolor",
        highlight=False,
        soft_wrap=False,
    ).print(renderable)
    return buffer.getvalue()


@dataclass
class Block:
    """One entry in the transcript."""

    kind: str = "ansi"  # ansi | markup | markdown | user | tool
    text: str = ""
    title: str = ""
    summary: str = ""
    took: str = ""
    detail: str = ""
    number: int = 0
    expanded: bool = False
    error: bool = False
    _cache: tuple[int, list[FormattedText]] | None = field(default=None, repr=False)

    @property
    def expandable(self) -> bool:
        return self.kind == "tool" and bool(self.detail.strip())


class Transcript:
    """Ordered blocks, lazily rendered for the current width."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.width = 80
        # Set by the app to repaint after a change, whichever thread made it.
        self.on_change: Callable[[], None] | None = None
        self._pending = ""  # console output that has not reached a newline yet
        self._open: Block | None = None  # the notice block console output appends to
        self._tools = 0

    # -- writing -------------------------------------------------------------
    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def add(self, block: Block) -> Block:
        self._open = None
        self.blocks.append(block)
        self._changed()
        return block

    def write(self, text: str) -> None:
        """Append raw console output (ANSI), one block per contiguous run."""
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._append_line(line + "\n")

    def flush(self) -> None:
        if self._pending:
            self._append_line(self._pending)
            self._pending = ""

    def _append_line(self, line: str) -> None:
        if self._open is None:
            self._open = self.add(Block(kind="ansi"))
        self._open.text += line
        self._open._cache = None
        self._changed()

    def clear(self) -> None:
        self.blocks.clear()
        self._pending = ""
        self._open = None

    # -- tool blocks ---------------------------------------------------------
    def start_tool(self, title: str, *, detail: str = "") -> Block:
        self._tools += 1
        return self.add(Block(kind="tool", title=title, detail=detail, number=self._tools))

    def finish_tool(self, block: Block, *, summary: str, detail: str, took: str = "", error: bool = False) -> None:
        block.summary = summary
        block.detail = detail
        block.took = took
        block.error = error
        block._cache = None
        self._changed()

    def toggle(self, block: Block) -> bool:
        if not block.expandable:
            return False
        block.expanded = not block.expanded
        block._cache = None
        self._changed()
        return True

    def last_tool(self) -> Block | None:
        return next((block for block in reversed(self.blocks) if block.kind == "tool"), None)

    def tool_by_number(self, number: int) -> Block | None:
        return next((block for block in self.blocks if block.kind == "tool" and block.number == number), None)

    # -- rendering -----------------------------------------------------------
    def _render(self, block: Block, width: int) -> list[FormattedText]:
        if block._cache is not None and block._cache[0] == width:
            return block._cache[1]
        if block.kind == "tool":
            lines = self._render_tool(block, width)
        elif block.kind == "assistant":
            lines = to_lines(render(Markdown(block.text), width))
        elif block.kind == "user":
            lines = to_lines(render(RichText.assemble(("❯ ", "bold cyan"), (block.text, "")), width))  # noqa: RUF001
        elif block.kind == "markup":
            lines = to_lines(render(RichText.from_markup(block.text), width))
        else:
            lines = to_lines(block.text)
        block._cache = (width, lines)
        return lines

    def _render_tool(self, block: Block, width: int) -> list[FormattedText]:
        marker = "▾" if block.expanded else "▸"
        header = f"{marker} {block.number}. {block.title}"
        if block.summary:
            header += f"  {block.summary}"
        if block.took:
            header += f" · {block.took}"
        if block.expandable:
            header += f"  ({COLLAPSE_HINT if block.expanded else EXPAND_HINT})"
        style = "bold red" if block.error else "bold green"
        lines = to_lines(render(RichText(header, style=style, overflow="fold"), width))
        if block.expanded and block.detail:
            body = RichText(block.detail.rstrip("\n"), style="dim", overflow="fold")
            lines.extend(indent(line) for line in to_lines(render(body, width)))
        return lines

    def lines(self, width: int) -> list[tuple[int, FormattedText]]:
        """Every rendered line paired with the index of the block it belongs to."""
        self.width = width
        result: list[tuple[int, FormattedText]] = []
        for index, block in enumerate(self.blocks):
            # A streamed answer is split into one block per paragraph; keep the
            # blank line that used to separate them on the terminal.
            if block.kind == "assistant" and index and self.blocks[index - 1].kind == "assistant":
                result.append((index, FormattedText()))
            result.extend((index, line) for line in self._render(block, width))
        if self._pending:
            result.append((len(self.blocks), to_formatted_text(ANSI(self._pending))))
        return result

    def text(self, width: int = 100) -> str:
        """Plain text of the whole transcript (assertions, tests)."""
        return "\n".join(to_plain_text(line) for _, line in self.lines(width))
