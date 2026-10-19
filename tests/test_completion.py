"""Tests for completion menus: /commands, /skill:<name> and @file mentions."""

import io
import threading
import time
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit.document import Document
from rich.console import Console

from tynex.coding.agent_session import AgentSession
from tynex.tui.app import InteractiveApp
from tynex.tui.prompt import COMPLETION_MENU_ROWS, TynexCompleter, _cap_completion_menu, _prompt_key_bindings


def make_session(tmp_path: Path) -> AgentSession:
    return AgentSession(cwd=str(tmp_path), tool_names=["read"], no_session=True)


def make_completer(tmp_path: Path) -> TynexCompleter:
    return TynexCompleter(make_session(tmp_path))


def texts(completer: TynexCompleter, text: str) -> list[str]:
    return [completion.text for completion in completer.get_completions(Document(text), None)]


def apply_completion(text: str, completion) -> str:
    """Simulate prompt_toolkit applying a completion to the document."""
    start = len(text) + completion.start_position
    return text[:start] + completion.text


def write_skill(tmp_path: Path, name: str = "release", description: str = "Cut a release") -> Path:
    skill_dir = tmp_path / ".tynex" / "skills" / name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n", encoding="utf-8")
    return skill_file


def make_app(tmp_path: Path) -> InteractiveApp:
    return InteractiveApp(make_session(tmp_path), console=Console(file=io.StringIO(), width=110))


def console_output(app: InteractiveApp) -> str:
    return app.console.file.getvalue()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# /commands
# ---------------------------------------------------------------------------


def test_completes_slash_commands(tmp_path):
    completer = make_completer(tmp_path)
    everything = texts(completer, "/")
    assert {"/help", "/login", "/model", "/quit", "/skills"} <= set(everything)

    assert texts(completer, "/lo") == ["/login", "/logout"]
    completion = next(iter(completer.get_completions(Document("/lo"), None)))
    assert completion.start_position == -3
    assert completion.display_text == "/login"
    assert "sign in" in completion.display_meta_text


def test_command_completion_anywhere_in_line(tmp_path):
    completer = make_completer(tmp_path)
    completion = next(iter(completer.get_completions(Document("please run /log"), None)))
    assert completion.text == "/login"
    assert apply_completion("please run /log", completion) == "please run /login"


def test_completes_extension_commands(tmp_path):
    session = make_session(tmp_path)
    session.extensions.add_command("greet", lambda args: None, help="Say hello")
    completer = TynexCompleter(session)

    assert texts(completer, "/gr") == ["/greet"]
    completion = next(iter(completer.get_completions(Document("/gr"), None)))
    assert "extension" in completion.display_meta_text and "Say hello" in completion.display_meta_text


def test_builtin_commands_win_name_collisions(tmp_path):
    session = make_session(tmp_path)
    session.extensions.add_command("help", lambda args: None, help="Shadowed")
    completer = TynexCompleter(session)

    matches = list(completer.get_completions(Document("/help"), None))
    assert [c.text for c in matches] == ["/help"]
    assert "extension" not in matches[0].display_meta_text


# ---------------------------------------------------------------------------
# Prompt templates and /skill:<name>
# ---------------------------------------------------------------------------


def test_completes_templates_and_skills(tmp_path):
    prompts = tmp_path / ".tynex" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "review.md").write_text("Review the diff carefully.", encoding="utf-8")
    write_skill(tmp_path)

    completer = make_completer(tmp_path)
    assert texts(completer, "/rev") == ["/review"]
    assert texts(completer, "/skill:rel") == ["/skill:release"]
    assert texts(completer, "/skill:") == ["/skill:release", "/skill:code-review"]

    metas = {c.text: c.display_meta_text for c in completer.get_completions(Document("/"), None)}
    assert "template" in metas["/review"]
    assert "skill" in metas["/skill:release"] and "Cut a release" in metas["/skill:release"]
    assert "skill" in metas["/skill:code-review"]


def test_skill_commands_list_and_load_instructions(tmp_path):
    skill_file = write_skill(tmp_path)
    app = make_app(tmp_path)

    # /skills lists available skills with the command that runs them.
    assert app._handle_command("/skills") is None
    listing = console_output(app)
    assert "/skill:release" in listing and "Cut a release" in listing

    # Unknown skill names are reported.
    assert app._handle_command("/skill:nope") is None
    assert "Unknown skill 'nope'" in console_output(app)

    # /skill:<name> [args] loads the SKILL.md instructions directly.
    result = app._handle_command("/skill:release bump the version")
    assert isinstance(result, str)
    assert 'Skill "release"' in result
    assert str(skill_file.parent).replace("\\", "/") in result
    assert "# release" in result  # full instructions, not just a path
    assert "description: Cut a release" not in result  # frontmatter removed
    assert result.endswith("bump the version")

    bare = app._handle_command("/skill:release")
    assert isinstance(bare, str) and "# release" in bare

    # The plural form remains as a compatibility alias.
    legacy = app._handle_command("/skills:release legacy")
    assert isinstance(legacy, str) and "# release" in legacy and legacy.endswith("legacy")

    skill_file.unlink()
    assert app._handle_command("/skill:release") is None
    assert "Could not read skill 'release'" in console_output(app)


def test_skill_command_starts_a_turn_with_its_instructions(tmp_path):
    write_skill(tmp_path)
    app = make_app(tmp_path)
    submitted: list[str] = []
    app._submit = lambda text, **kwargs: submitted.append(text)

    app._handle_line("/skill:release fix the build")

    assert len(submitted) == 1
    assert "# release" in submitted[0]
    assert submitted[0].endswith("fix the build")


# ---------------------------------------------------------------------------
# @ file mentions
# ---------------------------------------------------------------------------


def test_at_completes_files_and_directories(tmp_path):
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("x", encoding="utf-8")
    (src / "main.py").write_text("x", encoding="utf-8")
    (src / "models").mkdir()

    completer = make_completer(tmp_path)
    assert texts(completer, "@REA") == ["README.md"]
    assert texts(completer, "@sr") == ["src/"]

    inside = texts(completer, "@src/")
    assert {"app.py", "main.py", "models/"} <= set(inside)
    assert inside.index("models/") < inside.index("app.py")  # directories first

    assert texts(completer, "@src/ma") == ["main.py"]
    completion = next(iter(completer.get_completions(Document("@src/ma"), None)))
    assert completion.start_position == -2
    assert apply_completion("@src/ma", completion) == "@src/main.py"


def test_at_mention_mid_line_and_at_alone(tmp_path):
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("x", encoding="utf-8")

    completer = make_completer(tmp_path)
    completion = next(iter(completer.get_completions(Document("explain @src/ma"), None)))
    assert apply_completion("explain @src/ma", completion) == "explain @src/main.py"
    assert "README.md" in texts(completer, "@")  # bare "@" lists the directory


def test_plain_words_complete_paths(tmp_path):
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("x", encoding="utf-8")

    completer = make_completer(tmp_path)
    assert texts(completer, "REA") == ["README.md"]
    assert texts(completer, "src/ma") == ["main.py"]
    assert texts(completer, "explain REA") == ["README.md"]
    completion = next(iter(completer.get_completions(Document("src/ma"), None)))
    assert apply_completion("src/ma", completion) == "src/main.py"


def test_plain_first_word_completes_commands_shortcuts_and_skills(tmp_path):
    prompts = tmp_path / ".tynex" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "review.md").write_text("Review the diff.", encoding="utf-8")
    write_skill(tmp_path)
    completer = make_completer(tmp_path)

    assert "/help" in texts(completer, "he")
    assert "/stats" in texts(completer, "stat")  # alias
    assert "/review" in texts(completer, "rev")  # prompt template
    assert "/skill:release" in texts(completer, "rel")  # skill suffix

    completion = next(iter(completer.get_completions(Document("he"), None)))
    assert apply_completion("he", completion) == "/help"

    # A command word in the middle of prose stays prose (no menu hijacking Enter).
    assert texts(completer, "please he") == []


def test_hidden_entries_only_with_dot_prefix(tmp_path):
    (tmp_path / ".env").write_text("x", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("x", encoding="utf-8")

    completer = make_completer(tmp_path)
    assert texts(completer, "@e") == []
    assert ".env" in texts(completer, "@.")
    assert texts(completer, "@vi") == ["visible.txt"]


# ---------------------------------------------------------------------------
# End to end: Tab inside a real prompt session
# ---------------------------------------------------------------------------


_MENU = object()  # drive_prompt marker: wait for the completion menu before the next key


def drive_prompt(session: AgentSession, *chunks: object) -> str:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    _cap_completion_menu(COMPLETION_MENU_ROWS)  # mirrors the app

    with create_pipe_input() as pipe:
        prompt = PromptSession(
            completer=TynexCompleter(session),
            complete_while_typing=True,  # mirrors the app: menus pop up while typing
            key_bindings=_prompt_key_bindings(),  # mirrors the app: Enter completes an open menu
            input=pipe,
            output=DummyOutput(),
        )
        buffer = prompt.default_buffer
        errors: list[AssertionError] = []

        def wait_until(check: Callable[[], bool], what: str, timeout: float = 10.0) -> None:
            # Completion is a background task on the event loop, so feed each
            # chunk only once the buffer shows the previous one took effect.
            # Fixed sleeps make this test flaky on a loaded machine; waiting on
            # observable state cannot be "too early".
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if check():
                    return
                time.sleep(0.01)
            errors.append(AssertionError(f"timed out after {timeout:g}s waiting for {what}"))

        def menu_ready() -> bool:
            return (
                buffer.complete_state is not None
                and bool(buffer.complete_state.completions)
                and buffer.complete_state.original_document.text == buffer.text
            )

        def type_line() -> None:
            for chunk in chunks:
                if chunk is _MENU:
                    wait_until(menu_ready, "the completion menu to show entries")
                    continue
                assert isinstance(chunk, str)
                accept_key = chunk in ("\t", "\x1b[B")
                if accept_key:
                    # Tab/Down only act on the menu once it lists entries.
                    wait_until(menu_ready, "the completion menu to show entries")
                before = buffer.text
                pipe.send_text(chunk)
                if chunk == "\r":
                    continue  # submitting closes the prompt; nothing to observe
                if accept_key:
                    wait_until(lambda prev=before: buffer.text != prev, f"{chunk!r} to change the text")
                else:
                    wait_until(lambda prev=before, typed=chunk: buffer.text == prev + typed, f"{chunk!r} to be entered")

        thread = threading.Thread(target=type_line)
        thread.start()
        try:
            result = prompt.prompt("> ")
        finally:
            thread.join(timeout=60)
        if errors:
            raise errors[0]
        return result


def test_menus_and_accept_keys_in_prompt_session(tmp_path):
    (tmp_path / "main.py").write_text("x", encoding="utf-8")
    write_skill(tmp_path)
    session = make_session(tmp_path)

    # Tab accepts the highlighted completion, then Enter submits.
    assert drive_prompt(session, "@main", "\t", "\r") == "@main.py"
    assert drive_prompt(session, "/lo", "\t", "\r") == "/login"
    assert drive_prompt(session, "/skill:rel", "\t", "\r") == "/skill:release"
    # With the menu open, the first Enter fills the completion into the input
    # and the next Enter submits it: "/mo" -> "/model", then run the command.
    assert drive_prompt(session, "/mo", _MENU, "\r", "\r") == "/model"
    assert drive_prompt(session, "@main", _MENU, "\r", "\r") == "@main.py"
    # Down selects a row and previews it in the input; Enter then submits it.
    assert drive_prompt(session, "@main", "\x1b[B", "\r") == "@main.py"
    # A bare trigger takes the first row, which the next Enter submits.
    assert drive_prompt(session, "/", _MENU, "\r", "\r") == "/help"
    # An exact match has nothing to complete, so one Enter submits it.
    assert drive_prompt(session, "/model", "\r") == "/model"


def test_app_session_opens_the_menu_while_typing(tmp_path):
    """The app's own prompt session must keep ``complete_while_typing``: setting
    ``enable_history_search`` makes prompt_toolkit silently disable it."""
    from prompt_toolkit.application.current import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    session = make_session(tmp_path)
    app = InteractiveApp(session, console=Console(file=io.StringIO(), width=80))
    seen: list[str] = []

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        prompt = app.input
        buffer = prompt.default_buffer

        def type_line() -> None:
            pipe.send_text("/he")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                state = buffer.complete_state
                if state is not None and state.completions and state.original_document.text == buffer.text:
                    seen.extend(completion.text for completion in state.completions)
                    break
                time.sleep(0.01)
            pipe.send_text("\x03")  # ctrl-c aborts the prompt

        thread = threading.Thread(target=type_line)
        thread.start()
        try:
            prompt.prompt("> ")
        except KeyboardInterrupt:
            pass
        finally:
            thread.join(10)

    assert "/help" in seen
