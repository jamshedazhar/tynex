"""Tests for the core tools: read, write, edit, shell, grep, find, ls."""

import os

import pytest

from tynex.coding.tools import create_tools
from tynex.coding.tools.edit import EditError, EditTool
from tynex.coding.tools.read import ReadTool
from tynex.coding.tools.shell import (
    _UPDATE_PREVIEW_BYTES,
    ShellTool,
    _format_shell_result,
    _live_preview,
    _ShellFlavor,
    _ShellResult,
)
from tynex.coding.tools.truncate import DEFAULT_MAX_BYTES, format_size, truncate_head, truncate_tail

_resolved_shell = _ShellFlavor.get_shell()
# The default flavor (path "bash") is the fallback when no shell was found; a real shell has a real path.
requires_bash = pytest.mark.skipif(
    _resolved_shell.name != "bash" or _resolved_shell.path == "bash", reason="no POSIX shell is available"
)


def make_tools(cwd):
    return {tool.name: tool for tool in create_tools(str(cwd))}


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


def test_write_creates_parent_directories(tmp_path):
    tools = make_tools(tmp_path)
    result = tools["write"].execute("c1", {"path": "a/b/hello.txt", "content": "hi there"})
    assert not result.is_error
    assert (tmp_path / "a" / "b" / "hello.txt").read_text(encoding="utf-8") == "hi there"
    assert "Successfully wrote to a/b/hello.txt" in result.content[0].text


def test_write_rejects_directories(tmp_path):
    with pytest.raises(IsADirectoryError):
        make_tools(tmp_path)["write"].execute("c", {"path": ".", "content": "x"})


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_read_full_file(tmp_path):
    file = tmp_path / "f.txt"
    file.write_text("\n".join(f"line {i}" for i in range(1, 11)), encoding="utf-8")
    result = ReadTool(str(tmp_path)).execute("c", {"path": "f.txt"})
    text = result.content[0].text
    assert text.startswith("line 1")
    assert "line 10" in text


def test_read_offset_and_limit_continuation(tmp_path):
    file = tmp_path / "f.txt"
    file.write_text("\n".join(str(i) for i in range(1, 101)), encoding="utf-8")
    result = ReadTool(str(tmp_path)).execute("c", {"path": "f.txt", "offset": 50, "limit": 10})
    text = result.content[0].text
    assert text.splitlines()[0] == "50"
    assert "50-59" not in text  # no truncation note, just the user-limit note
    assert "41 more lines in file. Use offset=60 to continue." in text


def test_read_offset_beyond_end(tmp_path):
    file = tmp_path / "f.txt"
    file.write_text("one\ntwo", encoding="utf-8")
    try:
        ReadTool(str(tmp_path)).execute("c", {"path": "f.txt", "offset": 99})
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "beyond end of file" in str(exc)


def test_read_truncates_large_files(tmp_path):
    file = tmp_path / "big.txt"
    file.write_text("\n".join(f"{i}" for i in range(1, 2501)), encoding="utf-8")
    result = ReadTool(str(tmp_path)).execute("c", {"path": "big.txt"})
    text = result.content[0].text
    assert "[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]" in text


def test_read_counts_lines_without_a_trailing_phantom(tmp_path):
    file = tmp_path / "big.txt"
    file.write_text("".join(f"{i}\n" for i in range(1, 2002)), encoding="utf-8")

    text = ReadTool(str(tmp_path)).execute("c", {"path": "big.txt"}).content[0].text

    assert "of 2001. Use offset=2001 to continue." in text


def test_read_normalizes_crlf_and_legacy_cr(tmp_path):
    file = tmp_path / "mixed.txt"
    file.write_bytes(b"one\r\ntwo\rthree")
    text = ReadTool(str(tmp_path)).execute("c", {"path": "mixed.txt"}).content[0].text
    assert "one\ntwo\nthree" in text
    assert "\r" not in text


def test_read_missing_file(tmp_path):
    try:
        ReadTool(str(tmp_path)).execute("c", {"path": "nope.txt"})
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "nope.txt" in str(exc)


def test_read_directory_error(tmp_path):
    try:
        ReadTool(str(tmp_path)).execute("c", {"path": "."})
        raise AssertionError("expected IsADirectoryError")
    except IsADirectoryError:
        pass


def test_read_partial_does_not_load_the_whole_file(tmp_path):
    import tracemalloc

    line = "x" * 1000
    file = tmp_path / "big.txt"
    file.write_text("\n".join(line for _ in range(20_000)), encoding="utf-8")

    tracemalloc.start()
    try:
        text = ReadTool(str(tmp_path)).execute("c", {"path": "big.txt", "offset": 19_999, "limit": 1}).content[0].text
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert text.splitlines()[0] == line
    assert peak < 8 * 1024 * 1024  # far below the ~20MB file size


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


def test_edit_single_replacement(tmp_path):
    file = tmp_path / "e.py"
    file.write_text("def foo():\n    return 1\n", encoding="utf-8")
    result = EditTool(str(tmp_path)).execute(
        "c", {"path": "e.py", "edits": [{"oldText": "def foo():", "newText": "def bar():"}]}
    )
    assert "Successfully replaced 1 block(s)" in result.content[0].text
    assert file.read_text(encoding="utf-8") == "def bar():\n    return 1\n"
    assert result.details["diff"]
    assert "def bar()" in result.details["patch"]


def test_edit_multiple_disjoint_edits(tmp_path):
    file = tmp_path / "e.py"
    file.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    result = EditTool(str(tmp_path)).execute(
        "c",
        {
            "path": "e.py",
            "edits": [{"oldText": "a = 1", "newText": "a = 10"}, {"oldText": "c = 3", "newText": "c = 30"}],
        },
    )
    assert not result.is_error
    assert file.read_text(encoding="utf-8") == "a = 10\nb = 2\nc = 30\n"


def test_edit_rejects_directories(tmp_path):
    with pytest.raises(EditError, match="directory"):
        EditTool(str(tmp_path)).execute("c", {"path": ".", "edits": [{"oldText": "a", "newText": "b"}]})


def test_edit_rejects_non_utf8_files(tmp_path):
    (tmp_path / "latin.py").write_bytes(b"caf\xe9 = 1\n")

    with pytest.raises(EditError, match="UTF-8"):
        EditTool(str(tmp_path)).execute("c", {"path": "latin.py", "edits": [{"oldText": "caf", "newText": "x"}]})


def test_edit_requires_unique_match(tmp_path):
    file = tmp_path / "e.py"
    file.write_text("x = 1\nx = 1\n", encoding="utf-8")
    try:
        EditTool(str(tmp_path)).execute("c", {"path": "e.py", "edits": [{"oldText": "x = 1", "newText": "x = 2"}]})
        raise AssertionError("expected EditError")
    except Exception as exc:
        assert "not unique" in str(exc)


def test_edit_not_found(tmp_path):
    file = tmp_path / "e.py"
    file.write_text("hello\n", encoding="utf-8")
    try:
        EditTool(str(tmp_path)).execute("c", {"path": "e.py", "edits": [{"oldText": "goodbye", "newText": "hi"}]})
        raise AssertionError("expected EditError")
    except Exception as exc:
        assert "Could not find" in str(exc)


def test_edit_preserves_crlf_and_bom(tmp_path):
    file = tmp_path / "win.txt"
    file.write_bytes("\ufeffline1\r\nline2\r\n".encode("utf-8"))
    EditTool(str(tmp_path)).execute(
        "c", {"path": "win.txt", "edits": [{"oldText": "line2", "newText": "line2 changed"}]}
    )
    data = file.read_bytes()
    assert data.startswith("\ufeff".encode())
    assert b"line2 changed\r\n" in data


def test_edit_normalizes_crlf_in_old_and_new_text(tmp_path):
    file = tmp_path / "win.txt"
    file.write_bytes(b"line1\r\nline2\r\n")
    EditTool(str(tmp_path)).execute(
        "c", {"path": "win.txt", "edits": [{"oldText": "line1\r\n", "newText": "changed\r\n"}]}
    )
    data = file.read_bytes()
    assert data == b"changed\r\nline2\r\n"  # exactly one CR per line, on every OS


def test_edit_preserves_legacy_cr_line_endings(tmp_path):
    file = tmp_path / "mac.txt"
    file.write_bytes(b"line1\rline2\r")
    EditTool(str(tmp_path)).execute("c", {"path": "mac.txt", "edits": [{"oldText": "line1", "newText": "changed"}]})
    assert file.read_bytes() == b"changed\rline2\r"


def test_edit_prepare_arguments_legacy_shape():
    tool = EditTool(".")
    prepared = tool.prepare_arguments({"path": "a.txt", "oldText": "old", "newText": "new"})
    assert prepared["edits"] == [{"oldText": "old", "newText": "new"}]

    prepared = tool.prepare_arguments({"path": "a.txt", "edits": '[{"oldText": "a", "newText": "b"}]'})
    assert prepared["edits"] == [{"oldText": "a", "newText": "b"}]

    prepared = tool.prepare_arguments({"path": "a.txt", "edits": {"oldText": "a", "newText": "b"}})
    assert prepared["edits"] == [{"oldText": "a", "newText": "b"}]


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


def test_get_shell_finds_bash_on_windows_when_a_posix_shell_exists(monkeypatch):
    monkeypatch.setattr("tynex.coding.tools.shell.os.name", "nt")
    monkeypatch.setattr(
        "tynex.coding.tools.shell.shutil.which", lambda name: "C:/Git/bin/bash.exe" if name == "bash" else None
    )
    flavour = _ShellFlavor.get_shell()
    assert flavour.name == "bash"
    assert flavour.path == "C:/Git/bin/bash.exe"


def test_get_shell_skips_the_system32_wsl_bash(monkeypatch):
    monkeypatch.setattr("tynex.coding.tools.shell.os.name", "nt")
    monkeypatch.setattr(
        "tynex.coding.tools.shell.shutil.which", lambda name: "C:/Windows/System32/bash.exe" if name == "bash" else None
    )
    assert _ShellFlavor.get_shell() == _ShellFlavor()


def test_get_shell_falls_back_to_powershell_on_windows(monkeypatch):
    monkeypatch.setattr("tynex.coding.tools.shell.os.name", "nt")
    monkeypatch.setattr(
        "tynex.coding.tools.shell.shutil.which", lambda name: "C:/powershell.exe" if name == "powershell" else None
    )
    flavour = _ShellFlavor.get_shell()
    assert flavour.name == "powershell"
    assert flavour.path == "C:/powershell.exe"


def test_get_shell_falls_back_to_the_default_flavor(monkeypatch):
    monkeypatch.setattr("tynex.coding.tools.shell.shutil.which", lambda name: None)
    assert _ShellFlavor.get_shell() == _ShellFlavor()


@requires_bash
def test_get_shell_resolves_a_real_posix_shell():
    flavour = _ShellFlavor.get_shell()
    assert flavour.name == "bash"
    assert os.path.isfile(flavour.path)


def test_shell_tool_shapes_argv_and_command_per_flavor(monkeypatch, tmp_path):
    calls: list[tuple[str, list[str]]] = []

    def fake_run(command, *, argv, **kwargs):
        calls.append((command, argv))
        return _ShellResult(text="ok", exit_code=0)

    monkeypatch.setattr("tynex.coding.tools.shell._run_shell_command", fake_run)
    monkeypatch.setattr("tynex.coding.tools.shell.os.name", "nt")
    monkeypatch.setattr("tynex.coding.tools.shell.shutil.which", lambda name: "C:/pwsh.exe" if name == "pwsh" else None)
    powershell = _ShellFlavor.get_shell()
    assert powershell.name == "pwsh"
    bash = _ShellFlavor(path="/bin/bash", args=["-c"])

    for flavour, command in ((bash, "echo hi"), (powershell, "Get-Date")):
        monkeypatch.setattr("tynex.coding.tools.shell.SHELL", flavour)
        ShellTool(str(tmp_path)).execute("c", {"command": command})

    prefix = "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\n"
    assert calls == [
        ("echo hi", ["/bin/bash", "-c"]),
        (
            prefix + "Get-Date",
            ["C:/pwsh.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"],
        ),
    ]


@requires_bash
def test_bash_echo(tmp_path):
    result = ShellTool(str(tmp_path)).execute("c", {"command": "echo hello-tynex"})
    assert not result.is_error
    assert "hello-tynex" in result.content[0].text
    assert result.details["exitCode"] == 0


def test_shell_live_updates_decode_only_a_bounded_tail():
    buffer = bytearray(b"x" * (_UPDATE_PREVIEW_BYTES * 3))
    assert len(_live_preview(buffer).encode("utf-8")) <= _UPDATE_PREVIEW_BYTES

    # A slice that starts mid-codepoint is advanced to the next character boundary.
    buffer = bytearray(b"xx" + ("é" * (_UPDATE_PREVIEW_BYTES // 2)).encode("utf-8"))
    preview = _live_preview(buffer)
    assert preview.startswith("é")
    assert "\ufffd" not in preview


@requires_bash
def test_bash_exit_code(tmp_path):
    result = ShellTool(str(tmp_path)).execute("c", {"command": "exit 3"})
    assert result.is_error
    assert "Command exited with code 3" in result.content[0].text


@requires_bash
def test_bash_timeout(tmp_path):
    result = ShellTool(str(tmp_path)).run_capture("sleep 10", timeout=1)
    assert result.timed_out
    assert "timed out after 1 seconds" in _format_shell_result(result).content[0].text


def test_shell_overflow_reports_only_the_partial_capture(tmp_path):
    from tynex.coding.tools.shell import _MAX_OUTPUT_BYTES

    truncation = truncate_tail("x" * (DEFAULT_MAX_BYTES + 1))
    captured = tmp_path / "captured.txt"
    result = _ShellResult(
        text=truncation.content,
        exit_code=0,
        overflowed=True,
        truncated=True,
        full_output_path=str(captured),
        truncation=truncation,
    )
    text = _format_shell_result(result).content[0].text
    assert "Full output" not in text
    assert f"Output exceeded {format_size(_MAX_OUTPUT_BYTES)}" in text
    assert f"was saved to {captured}" in text
    assert "discarded" not in text


@requires_bash
def test_bash_session_env(tmp_path):
    from tynex.agent.types import ToolContext

    ctx = ToolContext(cwd=str(tmp_path), session_id="sess-123", provider="faux", model="test")
    result = ShellTool(str(tmp_path)).execute("c", {"command": 'echo "$TYNEX_SESSION_ID"'}, None, None, ctx)
    assert "sess-123" in result.content[0].text


@requires_bash
def test_bash_cwd_is_respected(tmp_path):
    from tynex.agent.types import ToolContext

    sub = tmp_path / "sub"
    sub.mkdir()
    probe = 'echo "${PWD##*/}"'

    # Commands run in the configured project directory.
    result = ShellTool(str(tmp_path)).execute("c", {"command": probe})
    assert not result.is_error
    assert tmp_path.name in result.content[0].text

    # `cd` inside a single command works (each invocation is a fresh shell).
    result = ShellTool(str(tmp_path)).execute("c", {"command": f"cd sub && {probe}"})
    assert not result.is_error
    assert "sub" in result.content[0].text

    # A ToolContext cwd overrides the tool default.
    ctx = ToolContext(cwd=str(sub), session_id="s", provider="faux", model="test")
    result = ShellTool(str(tmp_path)).execute("c", {"command": probe}, None, None, ctx)
    assert not result.is_error
    assert "sub" in result.content[0].text


# ---------------------------------------------------------------------------
# read-only tools
# ---------------------------------------------------------------------------


def test_grep_find_ls(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    pass\n# TODO: fix me\n", encoding="utf-8")
    (tmp_path / "src" / "util.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("TODO: docs\n", encoding="utf-8")

    from tynex.coding.tools.readonly import FindTool, GrepTool, LsTool

    grep = GrepTool(str(tmp_path))
    result = grep.execute("c", {"pattern": "TODO", "path": "."})
    text = result.content[0].text
    assert "app.py:3" in text
    assert "notes.md:1" in text

    find = FindTool(str(tmp_path))
    result = find.execute("c", {"pattern": "**/*.py"})
    assert "src/app.py" in result.content[0].text
    assert "notes.md" not in result.content[0].text

    ls = LsTool(str(tmp_path))
    result = ls.execute("c", {"path": "."})
    assert "src/" in result.content[0].text
    assert "notes.md" in result.content[0].text


# ---------------------------------------------------------------------------
# truncation primitives
# ---------------------------------------------------------------------------


def test_truncate_head_lines():
    content = "\n".join(str(i) for i in range(1, 101))
    result = truncate_head(content, max_lines=10, max_bytes=100_000)
    assert result.truncated
    assert result.truncated_by == "lines"
    assert result.output_lines == 10
    assert result.content.splitlines()[-1] == "10"


def test_truncate_head_bytes():
    content = "\n".join("x" * 100 for _ in range(10))
    result = truncate_head(content, max_lines=1000, max_bytes=250)
    assert result.truncated_by == "bytes"
    assert len(result.content.encode("utf-8")) <= 250


def test_truncate_tail_keeps_end():
    content = "\n".join(str(i) for i in range(1, 101))
    result = truncate_tail(content, max_lines=5, max_bytes=100_000)
    assert result.content.splitlines()[0] == "96"
    assert result.content.splitlines()[-1] == "100"
