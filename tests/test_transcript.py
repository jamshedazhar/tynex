"""Tests for the docked transcript model (blocks, expansion, rendering)."""

from tynex.tui.transcript import COLLAPSE_HINT, EXPAND_HINT, Block, Transcript, strip_ansi


def test_tool_block_starts_collapsed_and_expands_on_toggle():
    transcript = Transcript()
    block = transcript.start_tool("bash  echo hi")
    transcript.finish_tool(block, summary="hi", detail='input:\n  {"command": "echo hi"}\noutput:\n  hi', took="0.10s")

    collapsed = transcript.text()
    assert "▸ 1. bash  echo hi" in collapsed
    assert "hi" in collapsed and "0.10s" in collapsed
    assert EXPAND_HINT in collapsed
    assert "output:" not in collapsed  # the body is hidden

    assert transcript.toggle(block) is True
    expanded = transcript.text()
    assert "▾ 1." in expanded and COLLAPSE_HINT in expanded
    assert '{"command": "echo hi"}' in expanded
    assert "output:" in expanded

    assert transcript.toggle(block) is True
    assert "output:" not in transcript.text()


def test_tool_without_detail_is_not_expandable():
    transcript = Transcript()
    block = transcript.start_tool("read notes.txt")
    transcript.finish_tool(block, summary="3 lines", detail="")

    assert not block.expandable
    assert transcript.toggle(block) is False
    assert block.expanded is False
    assert EXPAND_HINT not in transcript.text()


def test_tools_are_numbered_and_addressable():
    transcript = Transcript()
    first = transcript.start_tool("first")
    second = transcript.start_tool("second")

    assert (first.number, second.number) == (1, 2)
    assert transcript.last_tool() is second
    assert transcript.tool_by_number(1) is first
    assert transcript.tool_by_number(2) is second
    assert transcript.tool_by_number(9) is None


def test_console_output_coalesces_until_a_block_is_added():
    transcript = Transcript()
    transcript.write("one\n")
    transcript.write("two\n")

    assert [block.kind for block in transcript.blocks] == ["ansi"]
    assert transcript.blocks[0].text == "one\ntwo\n"

    # A half-written line is still shown, without becoming a block of its own.
    transcript.write("partial")
    assert "partial" in transcript.text()

    transcript.add(Block(kind="markup", text="[green]done[/]"))
    transcript.write("three\n")
    assert [block.kind for block in transcript.blocks] == ["ansi", "markup", "ansi"]
    assert "done" in transcript.text()


def test_clear_resets_blocks_and_pending_output():
    transcript = Transcript()
    transcript.write("stale\npending")
    transcript.clear()

    assert transcript.blocks == []
    assert transcript.text() == ""


def test_assistant_markdown_and_user_blocks_render():
    transcript = Transcript()
    transcript.add(Block(kind="assistant", text="## Plan\n\n- first step"))
    transcript.add(Block(kind="user", text="hello there"))

    text = transcript.text()
    assert "Plan" in text and "first step" in text
    assert "❯ hello there" in text  # noqa: RUF001 - intentional prompt glyph


def test_streamed_paragraphs_keep_a_blank_line_between_them():
    # The renderer adds one block per paragraph while a message streams.
    transcript = Transcript()
    transcript.add(Block(kind="assistant", text="First paragraph."))
    transcript.add(Block(kind="assistant", text="Second paragraph."))
    transcript.add(Block(kind="user", text="and a reply"))

    lines = [line.strip() for line in transcript.text().splitlines()]
    assert lines[:3] == ["First paragraph.", "", "Second paragraph."]
    assert "and a reply" in lines[3]  # no stray blank line before a user block


def test_error_tool_blocks_are_marked_and_width_changes_rerender():
    transcript = Transcript()
    block = transcript.start_tool("bash  false")
    transcript.finish_tool(block, summary="command failed", detail="boom", error=True)

    narrow = transcript.text(width=40)
    assert "command failed" in narrow
    wide = transcript.text(width=120)  # a width change must invalidate the cache
    assert "command failed" in wide
    assert block._cache is not None and block._cache[0] == 120


def test_strip_ansi_removes_colours_and_links():
    assert strip_ansi("\x1b[31mred\x1b[0m") == "red"
    assert strip_ansi("\x1b]8;;https://x\x07link\x1b]8;;\x07") == "link"
