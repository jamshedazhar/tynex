"""In-process tests for the CLI (``tynex.cli``).

The subprocess CLI tests in ``test_integration.py`` cover the real end-to-end
flows; these tests exercise every branch in-process (and are what coverage
measures).
"""

import json
from types import SimpleNamespace
from typing import Any, Self

import click
import pytest
from click.testing import CliRunner

import tynex.cli as cli_mod
from tynex.agent.permissions import PermissionGate
from tynex.coding.settings import load_auth, store_api_key

runner = CliRunner()


class FakeAgentSession:
    """Captures the arguments ``cli()`` wires into the real session."""

    last: Self | None = None

    def __init__(self, **kwargs: Any) -> None:
        FakeAgentSession.last = self
        self.kwargs = kwargs
        self.model = kwargs.get("model")
        self.cwd = kwargs["cwd"]
        self.stealth = True
        self.tools: list[Any] = []
        self.permission_gate = PermissionGate(kwargs.get("permission_mode") or "accept-edits")


@pytest.fixture
def fake_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    FakeAgentSession.last = None
    monkeypatch.setattr(cli_mod, "AgentSession", FakeAgentSession)
    # Keep the dispatch tests away from the real modes (which read stdin / run the agent).
    monkeypatch.setattr("tynex.cli.run_print", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tynex.cli.run_json", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tynex.cli.run_rpc", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tynex.tui.app.run_interactive", lambda *args, **kwargs: 0)
    return FakeAgentSession


# ---------------------------------------------------------------------------
# login / logout helpers
# ---------------------------------------------------------------------------


def test_cli_ask_masks_only_on_a_tty(monkeypatch):
    seen: dict[str, Any] = {}

    def fake_prompt(text, **kwargs):
        seen.update(kwargs)
        return "value"

    monkeypatch.setattr(cli_mod.click, "prompt", fake_prompt)
    monkeypatch.setattr(cli_mod.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    assert cli_mod._cli_ask("key: ", password=True) == "value"
    assert seen["hide_input"] is True

    monkeypatch.setattr(cli_mod.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    cli_mod._cli_ask("key: ", password=True)
    assert seen["hide_input"] is False


def test_cli_select_needs_a_terminal(monkeypatch):
    monkeypatch.setattr(cli_mod.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(cli_mod.sys, "stdout", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(cli_mod.CliError, match="needs a terminal"):
        cli_mod._cli_select("Pick", ["a"])

    monkeypatch.setattr(cli_mod.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(cli_mod, "select_option", lambda title, options: 1)
    assert cli_mod._cli_select("Pick", ["a", "b"]) == 1


def test_login_command_unknown_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli_mod.login_command, ["bogus"])
    assert result.exit_code == 2
    assert "Unknown provider 'bogus'" in result.stderr


def test_login_command_keyless_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reported: list[str] = []
    monkeypatch.setattr(cli_mod, "_report_default_model", lambda provider, settings: reported.append(provider))
    result = runner.invoke(cli_mod.login_command, ["ollama"])
    assert result.exit_code == 0
    assert "no credentials needed" in result.stdout
    assert reported == ["ollama"]


def test_login_command_api_key_cancel_and_replace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "_report_default_model", lambda provider, settings: None)

    monkeypatch.setattr(cli_mod, "_cli_ask", lambda text, password=False: "   ")
    result = runner.invoke(cli_mod.login_command, ["anthropic"])
    assert "Cancelled." in result.stdout
    assert load_auth() == {}

    store_api_key("anthropic", "sk-old")
    monkeypatch.setattr(cli_mod, "_cli_ask", lambda text, password=False: " sk-new ")
    result = runner.invoke(cli_mod.login_command, ["anthropic"])
    assert "Replacing the stored credentials for anthropic." in result.stdout
    assert "Stored the API key for anthropic" in result.stdout
    assert load_auth()["anthropic"]["key"] == "sk-new"


def test_login_command_copilot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "_report_default_model", lambda provider, settings: None)
    monkeypatch.setattr(cli_mod, "copilot_login", lambda **kwargs: False)
    assert cli_mod.login_command.callback("github-copilot") == 1

    monkeypatch.setattr(cli_mod, "copilot_login", lambda **kwargs: True)
    assert cli_mod.login_command.callback("github-copilot") == 0


def test_login_command_without_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "interactive_login", lambda **kwargs: None)
    result = runner.invoke(cli_mod.login_command, [])
    assert "Nothing to do." in result.stdout

    reported: list[str] = []
    monkeypatch.setattr(cli_mod, "interactive_login", lambda **kwargs: "anthropic")
    monkeypatch.setattr(cli_mod, "_report_default_model", lambda provider, settings: reported.append(provider))
    result = runner.invoke(cli_mod.login_command, [])
    assert result.exit_code == 0
    assert reported == ["anthropic"]


def test_logout_command_without_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli_mod.logout_command, [])
    assert "No stored credentials" in result.stdout


def test_logout_command_removes_and_notes_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tynex").mkdir()
    (tmp_path / ".tynex" / "settings.json").write_text(
        json.dumps({"apiKeys": {"groq": "from-settings"}}), encoding="utf-8"
    )
    store_api_key("groq", "sk-groq")
    monkeypatch.setenv("GROQ_API_KEY", "from-env")

    result = runner.invoke(cli_mod.logout_command, ["groq"])
    assert result.exit_code == 0
    assert "Signed out of groq" in result.stdout
    assert "settings apiKeys still configure groq" in result.stdout
    assert "GROQ_API_KEY is still set" in result.stdout
    assert load_auth() == {}


def test_logout_command_unknown_provider_and_picker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store_api_key("anthropic", "sk-a")
    store_api_key("openai", "sk-o")

    result = runner.invoke(cli_mod.logout_command, ["nope"])
    assert "No stored credentials for nope." in result.stdout

    monkeypatch.setattr(cli_mod, "_cli_select", lambda title, options: None)
    result = runner.invoke(cli_mod.logout_command, [])
    assert "Nothing to do." in result.stdout

    monkeypatch.setattr(cli_mod, "_cli_select", lambda title, options: 1)
    result = runner.invoke(cli_mod.logout_command, [])
    assert "Signed out of openai" in result.stdout
    assert set(load_auth()) == {"anthropic"}


# ---------------------------------------------------------------------------
# cli(): session/model/tool wiring and mode dispatch
# ---------------------------------------------------------------------------


def test_print_models_rejects_unknown_provider():
    with pytest.raises(cli_mod.CliError, match="Unknown provider 'nope'"):
        cli_mod.print_models({}, "nope")


def test_print_models_filters_to_one_provider(monkeypatch, capsys):
    seen: list[str] = []
    monkeypatch.setattr(cli_mod, "ensure_models", lambda name, settings: (seen.append(name), ([], "no creds"))[1])
    cli_mod.print_models({}, "google")
    assert seen == ["google"]
    assert "google:" in capsys.readouterr().out


def test_cli_models_and_sessions_subcommands(fake_session, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(cli_mod, "print_models", lambda settings, provider: calls.append("models"))
    monkeypatch.setattr(cli_mod, "print_sessions", lambda cwd: calls.append("sessions"))

    assert cli_mod.main(["models"]) == 0
    assert cli_mod.main(["sessions"]) == 0
    assert calls == ["models", "sessions"]


def test_cli_model_resolution_error(fake_session, monkeypatch):
    def fail(pattern, settings):
        raise ValueError(f"no model matches {pattern!r}")

    monkeypatch.setattr(cli_mod, "resolve_model_retry", fail)
    result = runner.invoke(cli_mod.cli, ["-p", "--model", "nope/none", "hi"])
    assert result.exit_code == 2
    assert "tynex: no model matches 'nope/none'" in result.stderr


def test_cli_session_resolution_errors(fake_session, monkeypatch):
    monkeypatch.setattr(cli_mod, "latest_session_file", lambda cwd: None)
    result = runner.invoke(cli_mod.cli, ["-c", "-p", "hi"])
    assert result.exit_code == 2
    assert "no session to resume" in result.stderr
    result = runner.invoke(cli_mod.cli, ["-r", "-p", "hi"])
    assert result.exit_code == 2
    assert "no session to resume" in result.stderr

    monkeypatch.setattr(cli_mod, "find_session", lambda cwd, ident: None)
    result = runner.invoke(cli_mod.cli, ["-r", "abc", "-p", "hi"])
    assert result.exit_code == 2
    assert "no session matching 'abc'" in result.stderr


def test_cli_resolves_existing_sessions(fake_session, tmp_path, monkeypatch):
    latest = str(tmp_path / "latest.jsonl")
    found = str(tmp_path / "found.jsonl")
    monkeypatch.setattr(cli_mod, "latest_session_file", lambda cwd: latest)
    monkeypatch.setattr(cli_mod, "find_session", lambda cwd, ident: found)

    assert runner.invoke(cli_mod.cli, ["-c", "-p", "--model", "faux/test", "hi"]).exit_code == 0
    assert FakeAgentSession.last.kwargs["session_file"] == latest

    assert runner.invoke(cli_mod.cli, ["-r", "abc", "-p", "--model", "faux/test", "hi"]).exit_code == 0
    assert FakeAgentSession.last.kwargs["session_file"] == found


def test_cli_wires_options_and_dispatches_interactive(fake_session, tmp_path, monkeypatch):
    calls: list[tuple[str, Any, Any]] = []

    def interactive(session, initial_message=None):
        calls.append(("interactive", session, initial_message))
        return 11

    monkeypatch.setattr("tynex.tui.app.run_interactive", interactive)
    result = runner.invoke(
        cli_mod.cli,
        ["--model", "faux/test", "--thinking", "high", "--append-system-prompt", "more rules", "hello", "world"],
    )
    assert result.exit_code == 0
    assert calls == [("interactive", FakeAgentSession.last, "hello world")]
    session = FakeAgentSession.last
    assert session is not None
    assert session.kwargs["thinking"] == "high"
    assert session.kwargs["append_system_prompt"] == "more rules"
    assert session.kwargs["session_file"] is None


def test_cli_settings_defaults_for_model_and_passes_through_thinking(fake_session, tmp_path):
    (tmp_path / ".tynex").mkdir()
    (tmp_path / ".tynex" / "settings.json").write_text(
        json.dumps({"model": "faux/test", "thinking": "low"}), encoding="utf-8"
    )
    assert runner.invoke(cli_mod.cli, []).exit_code == 0
    session = FakeAgentSession.last
    assert session is not None
    assert session.model is not None and session.model.id == "test"
    # AgentSession resolves the level (flag > session history > settings); the CLI
    # only forwards an explicit --thinking flag.
    assert session.kwargs["thinking"] is None


def test_cli_non_interactive_requires_a_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "AgentSession", FakeAgentSession)
    result = runner.invoke(cli_mod.cli, ["-p", "hi"])
    assert result.exit_code == 2
    assert "no model available" in result.stderr


def test_cli_print_mode_dispatch(fake_session, monkeypatch):
    calls = {}

    def run_print(session, prompt):
        calls["print"] = (session, prompt)
        return 7

    monkeypatch.setattr("tynex.cli.run_print", run_print)
    result = runner.invoke(cli_mod.cli, ["-p", "--model", "faux/test", "say", "hi"])
    assert result.exit_code == 0
    assert calls["print"][1] == "say hi"

    # Without an argument the prompt comes from stdin.
    result = runner.invoke(cli_mod.cli, ["-p", "--model", "faux/test"], input="from stdin\n")
    assert result.exit_code == 0
    assert calls["print"][1] == "from stdin"

    # Empty stdin is an error.
    result = runner.invoke(cli_mod.cli, ["-p", "--model", "faux/test"], input="")
    assert result.exit_code == 2
    assert "print mode requires a prompt" in result.stderr


def test_cli_json_and_rpc_dispatch(fake_session, monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def run_json(session, prompt):
        calls.append(("json", prompt))
        return 0

    def run_rpc(session):
        calls.append(("rpc", None))
        return 0

    monkeypatch.setattr("tynex.cli.run_json", run_json)
    monkeypatch.setattr("tynex.cli.run_rpc", run_rpc)

    result = runner.invoke(cli_mod.cli, ["--json", "--model", "faux/test", "hi"])
    assert result.exit_code == 0

    # JSON mode also accepts the prompt on stdin, like print mode.
    result = runner.invoke(cli_mod.cli, ["--json", "--model", "faux/test"], input="from stdin\n")
    assert result.exit_code == 0

    result = runner.invoke(cli_mod.cli, ["--json", "--model", "faux/test"])
    assert result.exit_code == 2
    assert "json mode requires a prompt" in result.stderr

    result = runner.invoke(cli_mod.cli, ["--rpc", "--model", "faux/test"])
    assert result.exit_code == 0
    assert calls == [("json", "hi"), ("json", "from stdin"), ("rpc", None)]

    conflict = runner.invoke(cli_mod.cli, ["-p", "--json", "--model", "faux/test", "hi"])
    assert conflict.exit_code == 2
    assert "choose one of --print, --json or --rpc" in conflict.stderr


# ---------------------------------------------------------------------------
# _ensure_utf8_streams / main()
# ---------------------------------------------------------------------------


def test_ensure_utf8_streams(monkeypatch):
    class Stream:
        def __init__(self, fail: bool = False) -> None:
            self.calls: list[dict[str, Any]] = []
            self.fail = fail

        def reconfigure(self, **kwargs):
            self.calls.append(kwargs)
            if self.fail:
                raise ValueError("cannot reconfigure")

    out = Stream()
    err = Stream(fail=True)
    monkeypatch.setattr(cli_mod.sys, "stdout", out)
    monkeypatch.setattr(cli_mod.sys, "stderr", err)
    cli_mod._ensure_utf8_streams()
    assert out.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert err.calls  # the failure is suppressed

    monkeypatch.setattr(cli_mod.sys, "stdout", object())  # no reconfigure attribute
    cli_mod._ensure_utf8_streams()


def test_main_dispatches_auth_commands_and_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "_ensure_utf8_streams", lambda: None)

    assert cli_mod.main(["logout", "nope"]) == 0
    assert "No stored credentials for nope." in capsys.readouterr().out

    assert cli_mod.main(["--version"]) == 0
    assert "tynex" in capsys.readouterr().out

    assert cli_mod.main(["--definitely-not-an-option"]) == 2
    assert "no such option" in capsys.readouterr().err.lower()

    def raise_abort(**kwargs):
        raise click.Abort

    monkeypatch.setattr(cli_mod.cli, "main", raise_abort)
    assert cli_mod.main([]) == 1
    assert "Aborted!" in capsys.readouterr().err

    def raise_interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod.cli, "main", raise_interrupt)
    assert cli_mod.main([]) == 130
    assert "Aborted!" in capsys.readouterr().err

    monkeypatch.setattr(cli_mod.cli, "main", lambda **kwargs: None)
    assert cli_mod.main([]) == 0
