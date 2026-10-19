"""Tests for tynex.coding.settings (tynex.coding.paths merged in): the single place the ~/.tynex layout is built."""

from pathlib import Path

from tynex.coding import settings as paths


def _write(base: Path, name: str, text: str) -> Path:
    path = base / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_config_dir_honours_env_and_creates_dir(monkeypatch, tmp_path):
    other = tmp_path / "elsewhere"
    monkeypatch.setenv(paths.ENV_CONFIG_DIR, str(other))
    assert paths.config_dir() == other
    assert other.is_dir()


def test_well_known_files_live_in_config_dir(tynex_config):
    assert paths.config_dir() == tynex_config
    assert paths.settings_path() == tynex_config / "settings.json"
    assert paths.auth_path() == tynex_config / "auth.json"
    assert paths.models_cache_path() == tynex_config / "models.json"
    assert paths.input_history_path() == tynex_config / "input_history.txt"
    assert paths.sessions_dir() == tynex_config / "sessions"
    assert paths.sessions_dir().is_dir()


def test_find_files_searches_project_then_cwd(builtin_root, tynex_config, tmp_path):
    project = tmp_path / "proj"
    _write(project / ".tynex", "prompts/c.md", "project c")
    _write(project, "prompts/d.md", "cwd d")
    _write(tynex_config, "prompts/b.md", "config b")  # app dir: never searched

    found = paths.find_files("prompts/*.md", str(project))

    assert [path.name for path in found] == ["c.md", "d.md"]
    assert found[0].parent.parent == project / ".tynex"
    assert found[1].parent.parent == project


def test_find_files_defaults_to_process_cwd(monkeypatch, builtin_root, tynex_config, tmp_path):
    repo = tmp_path / "repo"
    _write(repo, "prompts/local.md", "local")
    monkeypatch.chdir(repo)
    _write(tynex_config, "prompts/a.md", "config a")

    found = paths.find_files("prompts/*.md")

    assert [path.name for path in found] == ["local.md"]
    assert paths.find_files("missing/*.md", str(tmp_path)) == []


def test_find_files_one_returns_most_specific_match(builtin_root, tynex_config, tmp_path):
    project = tmp_path / "proj"
    _write(project / ".tynex", "SYSTEM.md", "project prompt")
    _write(tynex_config, "SYSTEM.md", "config prompt")

    assert paths.find_files("SYSTEM.md", str(project), one=True) == project / ".tynex" / "SYSTEM.md"
    # the config dir holds app state only, so its files are never discovered
    assert paths.find_files("SYSTEM.md", str(tmp_path / "elsewhere"), one=True) is None


def test_find_files_falls_back_to_builtin(builtin_root, tmp_path):
    _write(builtin_root, "prompts/z.md", "builtin z")
    project = tmp_path / "proj"
    _write(project / ".tynex", "prompts/c.md", "project c")

    assert [path.name for path in paths.find_files("prompts/*.md", str(project))] == ["c.md", "z.md"]
    assert paths.find_files("prompts/*.md", str(project), one=True) == project / ".tynex" / "prompts" / "c.md"
    # with nothing local, the built-in file is the match
    assert paths.find_files("prompts/*.md", str(tmp_path / "empty"), one=True) == builtin_root / "prompts" / "z.md"


def test_find_files_deduplicates_builtin_equal_to_cwd(builtin_root):
    _write(builtin_root, "prompts/x.md", "x")

    found = paths.find_files("prompts/*.md", str(builtin_root))

    assert [path.name for path in found] == ["x.md"]  # cwd == built-in dir: no duplicate


def test_read_user_text_handles_missing_empty_and_unreadable(tmp_path):
    (tmp_path / "dir.md").mkdir()

    assert paths.read_user_text(tmp_path / "missing.md") is None
    assert paths.read_user_text(tmp_path / "dir.md") is None
    assert paths.read_user_text(_write(tmp_path, "empty.md", "   \n\n")) is None
    assert paths.read_user_text(_write(tmp_path, "text.md", "  hello\nworld  \n")) == "hello\nworld"
