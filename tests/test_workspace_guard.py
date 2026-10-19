"""Workspace guard: file tools must not reach outside the working directory."""

from __future__ import annotations

import os
import subprocess

import pytest
from prompt_toolkit.document import Document

from tynex.agent.types import ToolContext
from tynex.coding.agent_session import AgentSession
from tynex.coding.tools import create_tools
from tynex.coding.tools.path_utils import WorkspaceBoundaryError, resolve_path
from tynex.tui.prompt import TynexCompleter


def make_tool(tmp_path, name):
    return {tool.name: tool for tool in create_tools(str(tmp_path), [name])}[name]


def outside_file(tmp_path):
    path = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    path.write_text("top secret", encoding="utf-8")
    return path


def make_directory_link(link, target):
    """Create a directory symlink, using an NTFS junction on Windows when symlinks are denied."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        if os.name != "nt":
            pytest.skip("directory links are not available on this system")
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("directory links are not available on this system")


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


def test_resolve_path_allows_paths_inside_cwd(tmp_path):
    inside = tmp_path / "sub" / "file.txt"
    inside.parent.mkdir()
    inside.write_text("ok", encoding="utf-8")

    assert resolve_path("sub/file.txt", str(tmp_path)) == os.path.realpath(inside)
    assert resolve_path(str(inside), str(tmp_path)) == os.path.realpath(inside)
    assert resolve_path("sub/../sub/file.txt", str(tmp_path)) == os.path.realpath(inside)
    assert resolve_path(".", str(tmp_path)) == os.path.realpath(tmp_path)


def test_resolve_path_rejects_escapes_and_sibling_prefixes(tmp_path):
    sibling = tmp_path.with_name(tmp_path.name + "-sibling")
    sibling.mkdir()
    (sibling / "file.txt").write_text("x", encoding="utf-8")
    try:
        with pytest.raises(WorkspaceBoundaryError, match="outside the working directory"):
            resolve_path("../outside.txt", str(tmp_path))
        with pytest.raises(WorkspaceBoundaryError):
            resolve_path(str(tmp_path.parent / "outside.txt"), str(tmp_path))
        with pytest.raises(WorkspaceBoundaryError):
            resolve_path(str(sibling / "file.txt"), str(tmp_path))
        with pytest.raises(ValueError, match="Path must not be empty"):
            resolve_path("", str(tmp_path))
    finally:
        (sibling / "file.txt").unlink()
        sibling.rmdir()


def test_resolve_path_honours_a_tool_context_cwd(tmp_path):
    tool = make_tool(tmp_path, "read")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "inside.txt").write_text("inside", encoding="utf-8")
    ctx = ToolContext(cwd=str(sub), session_id="s", provider="faux", model="test")

    assert "inside" in tool.execute("c", {"path": "inside.txt"}, None, None, ctx).content[0].text
    with pytest.raises(WorkspaceBoundaryError):
        tool.execute("c", {"path": "../outside.txt"}, None, None, ctx)


def test_resolve_path_returns_the_resolved_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    inside = target / "file.txt"
    inside.write_text("ok", encoding="utf-8")
    link = tmp_path / "link"
    make_directory_link(link, target)
    try:
        assert resolve_path("link/file.txt", str(tmp_path)) == os.path.realpath(inside)
    finally:
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            os.rmdir(link)


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("read", {"path": "../outside.txt"}),
        ("write", {"path": "../escaped.txt", "content": "x"}),
        ("edit", {"path": "../outside.txt", "edits": [{"oldText": "a", "newText": "b"}]}),
        ("grep", {"pattern": "secret", "path": ".."}),
        ("find", {"pattern": "*.txt", "path": ".."}),
        ("ls", {"path": ".."}),
    ],
)
def test_file_tools_reject_paths_outside_the_workspace(tmp_path, name, args):
    tool = make_tool(tmp_path, name)
    with pytest.raises(WorkspaceBoundaryError, match="outside the working directory"):
        tool.execute("c", args)


def test_write_outside_does_not_create_the_file(tmp_path):
    escaped = tmp_path.parent / f"{tmp_path.name}-escaped.txt"
    with pytest.raises(WorkspaceBoundaryError):
        make_tool(tmp_path, "write").execute("c", {"path": str(escaped), "content": "x"})
    assert not escaped.exists()


def test_absolute_path_inside_the_workspace_is_allowed(tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")
    result = make_tool(tmp_path, "read").execute("c", {"path": str(target)})
    assert "hello" in result.content[0].text


def test_symlink_escape_is_rejected(tmp_path):
    """A symlinked file inside the workspace must not expose its outside target."""
    secret = outside_file(tmp_path)
    link = tmp_path / "link.txt"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are not available on this system")

    with pytest.raises(WorkspaceBoundaryError, match="outside the working directory"):
        make_tool(tmp_path, "read").execute("c", {"path": "link.txt"})


def test_symlinked_directory_escape_is_rejected(tmp_path):
    """A link to an outside directory must be rejected by every file tool."""
    outside_dir = tmp_path.parent / f"{tmp_path.name}-outside-dir"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("secret", encoding="utf-8")
    link = tmp_path / "linked"
    make_directory_link(link, outside_dir)
    try:
        for name, args in (
            ("read", {"path": "linked/secret.txt"}),
            ("grep", {"pattern": "secret", "path": "linked"}),
            ("find", {"pattern": "*", "path": "linked"}),
            ("ls", {"path": "linked"}),
        ):
            with pytest.raises(WorkspaceBoundaryError, match="outside the working directory"):
                make_tool(tmp_path, name).execute("c", args)
    finally:
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            os.rmdir(link)
        (outside_dir / "secret.txt").unlink()
        outside_dir.rmdir()


# ---------------------------------------------------------------------------
# Completion and prompt
# ---------------------------------------------------------------------------


def test_path_completion_stays_inside_the_workspace(tmp_path):
    (tmp_path / "visible.txt").write_text("x", encoding="utf-8")
    outside_file(tmp_path)
    session = AgentSession(cwd=str(tmp_path), tool_names=[], no_session=True)
    completer = TynexCompleter(session)

    inside = [c.text for c in completer.get_completions(Document("@vis"), None)]
    assert inside == ["visible.txt"]
    assert list(completer.get_completions(Document("@../"), None)) == []


def test_system_prompt_contains_the_workspace_rule(tmp_path):
    session = AgentSession(cwd=str(tmp_path), tool_names=["read"], no_session=True)
    assert "Only read or modify files inside the working directory" in session.system_prompt_text()
