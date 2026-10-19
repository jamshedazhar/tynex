"""Ignore semantics of grep/find, backed by the ripgrep-rs bindings.

The tests stay git-free: ripgrep's ignore engine only needs a ``.git``
directory to exist for a tree to count as a repository. Both tools search
hidden files and directories (``.github``, ``.env``, …).

Outside a git repository the bindings offer no equivalent of fd's
``--no-require-git``, so ``find`` does not apply ``.gitignore`` there.
(``grep`` does not apply it either.)
"""

from pathlib import Path

import pytest

from tynex.coding.tools.readonly import FindTool, GrepTool


def make_repo(root: Path) -> Path:
    """A directory the ignore engine treats as a repository (no git binary needed)."""
    (root / ".git").mkdir(parents=True, exist_ok=True)
    return root


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def grep(cwd: Path, **args) -> str:
    return GrepTool(str(cwd)).execute("c", args).content[0].text


def find(cwd: Path, **args) -> str:
    return FindTool(str(cwd)).execute("c", args).content[0].text


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


def test_grep_respects_gitignore_inside_repo(tmp_path):
    repo = make_repo(tmp_path)
    write(repo / ".gitignore", "ignored.py\nbuild/\n")
    write(repo / "kept.py", "# TODO kept\n")
    write(repo / "ignored.py", "# TODO ignored\n")
    write(repo / "build" / "gen.py", "# TODO generated\n")

    text = grep(repo, pattern="TODO")

    assert "kept.py:1: # TODO kept" in text
    assert "ignored.py" not in text
    assert "gen.py" not in text


def test_grep_applies_gitignore_only_inside_repos_and_rgignore_anywhere(tmp_path):
    write(tmp_path / ".gitignore", "ignored.py\n")
    write(tmp_path / ".rgignore", "notes.txt\n")
    write(tmp_path / "ignored.py", "TODO ignored\n")
    write(tmp_path / "notes.txt", "TODO notes\n")
    write(tmp_path / "kept.py", "TODO kept\n")

    text = grep(tmp_path, pattern="TODO")

    assert "ignored.py" in text  # no repository: .gitignore is not consulted
    assert "notes.txt" not in text  # .rgignore is always respected
    assert "kept.py" in text


def test_grep_searches_hidden_files_and_directories(tmp_path):
    repo = make_repo(tmp_path)
    write(repo / ".github" / "workflows" / "ci.yml", "node-version: TODO\n")
    write(repo / ".env", "TODO secret\n")
    write(repo / "app.py", "TODO visible\n")

    text = grep(repo, pattern="TODO")

    assert "app.py:1: TODO visible" in text
    assert ".env:1: TODO secret" in text
    assert ".github/workflows/ci.yml:1: node-version: TODO" in text


def test_grep_context_lines(tmp_path):
    write(tmp_path / "m.py", "before\nneedle\nafter\n")

    text = grep(tmp_path, pattern="needle", context=1)

    assert "m.py-1- before" in text
    assert "m.py:2: needle" in text
    assert "m.py-3- after" in text


def test_grep_truncates_long_lines(tmp_path):
    write(tmp_path / "long.txt", "y" * 600 + " needle\n")

    text = grep(tmp_path, pattern="needle", context=1)

    assert "... [truncated]" in text
    assert "Some lines truncated to 500 chars. Use read tool to see full lines" in text


def test_grep_literal_and_case_insensitive(tmp_path):
    write(tmp_path / "m.py", "a+b (c) HELLO\n")

    literal = grep(tmp_path, pattern="a+b (c)", literal=True)
    assert "a+b (c) HELLO" in literal

    insensitive = grep(tmp_path, pattern="hello", ignoreCase=True)
    assert "HELLO" in insensitive


def test_grep_invalid_regex_raises(tmp_path):
    write(tmp_path / "m.py", "x\n")

    with pytest.raises(ValueError, match="regex parse error"):
        grep(tmp_path, pattern="(")


def test_grep_limit_and_notice(tmp_path):
    write(tmp_path / "m.py", "".join(f"needle {i}\n" for i in range(5)))

    text = grep(tmp_path, pattern="needle", limit=2)

    assert text.count("needle") == 2
    assert "[2 matches limit reached. Use limit=4 for more, or refine pattern]" in text


def test_grep_exact_limit_does_not_report_a_limit(tmp_path):
    write(tmp_path / "m.py", "".join(f"needle {i}\n" for i in range(5)))

    text = grep(tmp_path, pattern="needle", limit=5)

    assert "limit reached" not in text
    assert text.count("needle") == 5


def test_grep_glob_filter(tmp_path):
    write(tmp_path / "a.py", "TODO\n")
    write(tmp_path / "b.md", "TODO\n")

    text = grep(tmp_path, pattern="TODO", glob="*.py")

    assert "a.py" in text
    assert "b.md" not in text


def test_grep_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Path not found"):
        grep(tmp_path, pattern="x", path="nope")


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------


def test_find_respects_gitignore_and_matches_nested_globs(tmp_path):
    repo = make_repo(tmp_path)
    write(repo / ".gitignore", "dist/\n")
    write(repo / "app.py", "x\n")
    write(repo / "dist" / "bundle.py", "x\n")
    write(repo / "a" / "sub" / "deep.py", "x\n")

    text = find(repo, pattern="*.py")

    assert "app.py" in text
    assert "a/sub/deep.py" in text
    assert "bundle.py" not in text

    nested = find(repo, pattern="sub/*.py")

    assert "a/sub/deep.py" in nested  # "**/" prefix: path patterns match at any depth


def test_find_respects_gitignore_for_files(tmp_path):
    repo = make_repo(tmp_path)
    write(repo / ".gitignore", "ignored.py\n")
    write(repo / "kept.py", "x\n")
    write(repo / "ignored.py", "x\n")

    text = find(repo, pattern="*.py")
    exact = find(repo, pattern="ignored.py")

    assert "kept.py" in text
    assert "ignored.py" not in text
    assert "No files found" in exact


def test_find_traverses_nested_repos_and_lists_binaries(tmp_path):
    repo = make_repo(tmp_path)
    nested_repo = make_repo(repo / "vendor" / "lib")
    write(repo / "top.py", "x\n")
    write(nested_repo / "inner.py", "x\n")
    (repo / "blob.bin").write_bytes(b"\x00\x01")

    text = find(repo, pattern="*.py")
    assert "top.py" in text
    assert "vendor/lib/inner.py" in text

    binaries = find(repo, pattern="*.bin")
    assert "blob.bin" in binaries


def test_find_searches_hidden_and_honors_limit(tmp_path):
    write(tmp_path / ".hidden" / "x.py", "x\n")
    for i in range(3):
        write(tmp_path / f"f{i}.py", "x\n")

    tool = FindTool(str(tmp_path))
    text = tool.execute("c", {"pattern": "*.py"}).content[0].text
    assert ".hidden/x.py" in text

    limited = tool.execute("c", {"pattern": "*.py", "limit": 2}).content[0].text
    assert "[2 results limit reached. Use limit=4 for more, or refine pattern]" in limited


def test_find_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Path not found"):
        find(tmp_path, pattern="*.py", path="nope")


def test_search_tools_report_paths_relative_to_the_search_directory(tmp_path):
    write(tmp_path / "sub" / "deep.py", "TODO deep\n")

    text = grep(tmp_path, pattern="TODO", path="sub")
    assert "deep.py:1: TODO deep" in text
    assert "sub/deep.py" not in text  # paths are relative to the search path, not cwd

    assert find(tmp_path, pattern="*.py", path="sub") == "deep.py"


# ---------------------------------------------------------------------------
# error paths, abort handling, output formatting
# ---------------------------------------------------------------------------


def test_search_tools_reject_empty_patterns(tmp_path):
    write(tmp_path / "a.py", "x\n")

    with pytest.raises(ValueError, match="pattern must not be empty"):
        grep(tmp_path, pattern="")
    with pytest.raises(ValueError, match="pattern must not be empty"):
        find(tmp_path, pattern="")


def test_search_tools_no_result_messages(tmp_path):
    write(tmp_path / "a.py", "x\n")

    assert grep(tmp_path, pattern="zzz") == "No matches found"
    assert find(tmp_path, pattern="*.rs") == "No files found matching pattern"


def test_search_tools_check_abort_signal(tmp_path, monkeypatch):
    from tynex.coding.tools import readonly

    write(tmp_path / "a.py", "needle\n")

    class Signal:
        def __init__(self):
            self.aborted = False

        def throw_if_aborted(self):
            raise RuntimeError("aborted")

    aborted = Signal()
    aborted.aborted = True
    with pytest.raises(RuntimeError, match="aborted"):
        GrepTool(str(tmp_path)).execute("c", {"pattern": "needle"}, aborted)
    with pytest.raises(RuntimeError, match="aborted"):
        FindTool(str(tmp_path)).execute("c", {"pattern": "*.py"}, aborted)

    # abort raised mid-call (the binding cannot be interrupted, it is checked after)
    flipped = Signal()

    def fake_search(*args, **kwargs):
        flipped.aborted = True
        return []

    monkeypatch.setattr(readonly.rg, "search_structured", fake_search)
    with pytest.raises(RuntimeError, match="aborted"):
        GrepTool(str(tmp_path)).execute("c", {"pattern": "needle"}, flipped)


def test_grep_byte_truncation_notice(tmp_path):
    write(tmp_path / "big.txt", "".join("needle " + "y" * 700 + "\n" for _ in range(120)))

    text = grep(tmp_path, pattern="needle", limit=500)

    assert "[50.0KB limit reached" in text
