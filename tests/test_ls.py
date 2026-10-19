"""Tests for the ls tool (one directory level via ripgrep-rs file metadata)."""

import pytest

from tynex.coding.tools.readonly import LsTool


def ls(cwd, **args) -> str:
    return LsTool(str(cwd)).execute("c", args).content[0].text


def touch(path):
    path.write_text("", encoding="utf-8")
    return path


def test_ls_sorted_with_directory_suffix_and_dotfiles(tmp_path):
    touch(tmp_path / "b.txt")
    touch(tmp_path / "a.txt")
    (tmp_path / "Sub").mkdir()
    touch(tmp_path / ".hidden")

    lines = ls(tmp_path, path=".").splitlines()

    assert lines == [".hidden", "a.txt", "b.txt", "Sub/"]  # case-insensitive alphabetical


def test_ls_shows_ignored_and_hidden_entries(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    touch(tmp_path / "ignored.py")
    touch(tmp_path / "kept.py")

    text = ls(tmp_path, path=".")

    assert ".git/" in text  # readdir semantics: no ignore rules apply to ls
    assert "ignored.py" in text
    assert "kept.py" in text


def test_ls_limit_notice(tmp_path):
    for i in range(5):
        touch(tmp_path / f"f{i}.txt")

    text = ls(tmp_path, path=".", limit=2)

    assert text.count(".txt") == 2
    assert "[2 entries limit reached. Use limit=4 for more]" in text


def test_ls_empty_directory(tmp_path):
    assert ls(tmp_path, path=".") == "(empty directory)"


def test_ls_errors(tmp_path):
    touch(tmp_path / "file.txt")

    with pytest.raises(FileNotFoundError, match="Path not found"):
        ls(tmp_path, path="nope")
    with pytest.raises(NotADirectoryError, match="Not a directory"):
        ls(tmp_path, path="file.txt")
