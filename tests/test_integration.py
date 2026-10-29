"""End-to-end tests: AgentSession, print/JSON modes, CLI subprocesses."""

import json
import os
import subprocess
import sys

from tests.conftest import PROVIDER_KEY_ENVS, faux_scenario, models_cache_payload, text_scenario
from tynex.ai import providers
from tynex.ai.models import Model, register_models, resolve_model
from tynex.ai.providers import faux
from tynex.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage
from tynex.cli import run_json, run_print, serialize_event
from tynex.coding.agent_session import AgentSession
from tynex.coding.compaction import compact
from tynex.coding.session import SessionManager
from tynex.coding.settings import save_auth
from tynex.coding.system_prompt import load_context_files
from tynex.coding.tools.shell import shell_name
from tynex.tui.app import InteractiveApp


def faux_settings() -> dict:
    return {
        "providers": {
            "faux": {
                "api": "faux",
                "models": [{"id": "test", "name": "Faux", "contextWindow": 200_000, "maxTokens": 8_192}],
            }
        }
    }


# ---------------------------------------------------------------------------
# AgentSession
# ---------------------------------------------------------------------------


def test_agent_session_tool_round_trip(tmp_path, faux_model):
    faux.reset_script(
        [
            faux_scenario(
                [
                    {
                        "type": "toolcall",
                        "id": "call_1",
                        "name": "write",
                        "arguments": {"path": "hello.txt", "content": "from the model"},
                    },
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("All done."),
        ]
    )
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read", "write"])
    session.prompt("write hello.txt")

    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "from the model"
    assert session.context.messages[-1].text() == "All done."

    # The session was persisted as JSONL.
    assert session.session.file_path
    with open(session.session.file_path, encoding="utf-8") as fh:
        lines = fh.read().strip().splitlines()
    entries = [json.loads(line) for line in lines]
    assert entries[0]["type"] == "session"
    assert sum(1 for e in entries if e["type"] == "message") == 4

    # A second prompt continues the same context (and reuses the session file).
    session.prompt("what now?")
    assert session.context.messages[-1].text().startswith("[faux]")
    assert len(session.context.messages) >= 6
    stats = session.stats()
    assert stats["messages"] == len(session.context.messages)


def test_agent_session_steering_delivered(tmp_path, faux_model):
    faux.reset_script([text_scenario("handled both messages")])
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"])
    session.steer("also do the other thing")
    session.prompt("do the thing")

    user_messages = [m.text() for m in session.context.messages if isinstance(m, UserMessage)]
    assert user_messages == ["do the thing", "also do the other thing"]


def test_system_prompt_sections_and_agents_md(tmp_path, faux_model):
    (tmp_path / "AGENTS.md").write_text("Always run tests with pytest.", encoding="utf-8")
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read", shell_name, "edit", "write"])
    prompt = session.system_prompt_text()

    assert "<tools>" in prompt
    assert "- read: Read file contents" in prompt
    assert "<rules>" in prompt
    assert f"Use {shell_name} for file operations" in prompt
    assert "<project_context>" in prompt
    assert "Always run tests with pytest." in prompt
    assert "<cwd>" in prompt


def test_agents_md_nearest_file_wins(tmp_path):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    (outer / "AGENTS.md").write_text("outer rules", encoding="utf-8")
    (inner / "AGENTS.md").write_text("inner rules", encoding="utf-8")

    assert [entry["content"] for entry in load_context_files(str(inner))] == ["inner rules"]


def test_agents_md_project_dir_wins_over_cwd(tmp_path):
    (tmp_path / "AGENTS.md").write_text("cwd rules", encoding="utf-8")
    (tmp_path / ".tynex").mkdir()
    (tmp_path / ".tynex" / "AGENTS.md").write_text("project rules", encoding="utf-8")

    assert [entry["content"] for entry in load_context_files(str(tmp_path))] == ["project rules"]

    (tmp_path / ".tynex" / "AGENTS.md").write_text("", encoding="utf-8")
    assert load_context_files(str(tmp_path)) == []  # an empty nearest file adds nothing


def test_system_md_replaces_preamble(tmp_path, faux_model):
    (tmp_path / ".tynex").mkdir()
    (tmp_path / ".tynex" / "SYSTEM.md").write_text("You are a battle-tested systems engineer.", encoding="utf-8")
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"])
    prompt = session.system_prompt_text()
    assert prompt.startswith("You are a battle-tested systems engineer.")
    assert "coding agent harness" not in prompt


def test_read_only_tool_selection_rules(tmp_path, faux_model):
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["grep", "find", "ls"])
    prompt = session.system_prompt_text()
    assert "- grep:" in prompt
    assert "for file operations" not in prompt


# ---------------------------------------------------------------------------
# Context estimation
# ---------------------------------------------------------------------------


def _usage_session(tmp_path, faux_model, usage: Usage) -> AgentSession:
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
    session.context.messages = [
        UserMessage(content="hi"),
        AssistantMessage(content=[TextContent(text="ok")], usage=usage),
    ]
    return session


def test_estimate_context_tokens_counts_cache_writes(tmp_path, faux_model):
    # Anthropic reports the prompt prefix under cache_creation_input_tokens and
    # keeps usage.input free of it, so a cache-miss turn must still see ~30k.
    session = _usage_session(tmp_path, faux_model, Usage(input=100, output=50, cache_write=30_000))
    assert session.estimate_context_tokens() >= 30_150


def test_estimate_context_tokens_counts_cache_reads(tmp_path, faux_model):
    session = _usage_session(tmp_path, faux_model, Usage(input=100, output=50, cache_read=30_000))
    assert session.estimate_context_tokens() >= 30_150


def test_estimate_context_tokens_does_not_double_count_system_prompt(tmp_path, faux_model):
    # usage.input already contains the system prompt and tool schemas.
    session = _usage_session(tmp_path, faux_model, Usage(input=10_000, output=200))
    assert session.estimate_context_tokens() == 10_200


def test_estimate_context_tokens_falls_back_to_total_tokens(tmp_path, faux_model):
    session = _usage_session(tmp_path, faux_model, Usage(output=200, total_tokens=20_000))
    assert session.estimate_context_tokens() == 20_000


def test_estimate_context_tokens_without_usage_uses_heuristic(tmp_path, faux_model):
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
    session.context.messages = [UserMessage(content="x" * 40_000)]
    estimate = session.estimate_context_tokens()
    assert estimate >= 10_000  # 40k chars at ~4 chars per token, plus the system prompt
    assert estimate < len(session.system_prompt_text()) // 4 + 11_000


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def test_compact_summarizes_and_rewrites_transcript(tmp_path, faux_model):
    from tynex.ai.types import AssistantMessage, TextContent

    session = SessionManager(str(tmp_path))
    for i in range(1, 9):
        session.append_message(UserMessage(content=f"question {i} " + "x" * 200))
        session.append_message(AssistantMessage(content=[TextContent(text=f"answer {i} " + "y" * 200)]))

    faux.reset_script([text_scenario("THE SUMMARY")])
    result = compact(session, providers.stream, faux_model, None, keep_tokens=300)
    assert result is not None
    assert result.summary == "THE SUMMARY"
    assert result.messages_summarized > 0

    session.append_compaction(result.summary, result.first_kept_entry_id, result.tokens_before)
    messages = session.branch_messages()
    assert "THE SUMMARY" in messages[0].text()
    assert len(messages) < 16  # older messages were replaced by the summary


def test_compact_mid_turn_keeps_the_current_turn(tmp_path, faux_model):
    session = SessionManager(str(tmp_path))
    session.append_message(UserMessage(content="old question"))
    session.append_message(AssistantMessage(content=[TextContent(text="old answer")]))
    current = session.append_message(UserMessage(content="start"))
    for index in range(3):
        session.append_message(
            AssistantMessage(
                content=[ToolCall(id=f"c{index}", name="bash", arguments={"command": "ls"})],
                api="faux",
                provider="faux",
                model="test",
                stop_reason="toolUse",
            )
        )
        session.append_message(
            ToolResultMessage(tool_call_id=f"c{index}", tool_name="bash", content=[TextContent(text="x" * 60_000)])
        )

    faux.reset_script([text_scenario("THE SUMMARY")])
    result = compact(session, providers.stream, faux_model, None, keep_tokens=30_000)
    assert result is not None
    assert result.first_kept_entry_id == current.id
    assert result.messages_summarized == 2


# ---------------------------------------------------------------------------
# Print / JSON modes
# ---------------------------------------------------------------------------


def test_print_mode_writes_response_to_stdout(tmp_path, faux_model, capsys):
    faux.reset_script([text_scenario("the answer is 42")])
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
    code = run_print(session, "what is the answer?")
    assert code == 0
    out = capsys.readouterr().out
    assert out.strip() == "the answer is 42"


def test_print_mode_reports_errors(tmp_path, faux_model, capsys):
    faux.reset_script([faux_scenario([{"type": "error", "error": "api key missing"}])])
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
    code = run_print(session, "hello")
    assert code == 1
    assert "api key missing" in capsys.readouterr().err


def test_json_mode_emits_events(tmp_path, faux_model, capsys):
    faux.reset_script([text_scenario("json answer")])
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
    code = run_json(session, "hi")
    assert code == 0
    out = capsys.readouterr().out
    events = [json.loads(line) for line in out.splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert types[0] == "agent_start"
    assert types[-1] == "agent_end"
    assert any(e["type"] == "message_end" and e["message"]["role"] == "assistant" for e in events)


def test_serialize_event_is_json_safe(tmp_path, faux_model):
    from tynex.ai.types import AssistantMessage

    message = AssistantMessage(content=[], provider="faux", model="test")
    event = {"type": "message_end", "message": message}
    serialized = serialize_event(event)
    assert json.dumps(serialized)
    assert serialized["message"]["role"] == "assistant"

    result_event = {
        "type": "tool_execution_end",
        "toolCallId": "c1",
        "toolName": "read",
        "result": __import__("tynex.agent.types", fromlist=["ToolResult"]).ToolResult.text("content"),
        "isError": False,
        "durationMs": 3,
    }
    serialized = serialize_event(result_event)
    assert serialized["result"]["content"][0]["text"] == "content"
    assert json.dumps(serialized)


# ---------------------------------------------------------------------------
# Model auto-selection (auth.json / env vars / settings)
# ---------------------------------------------------------------------------


def test_detection_prefers_newest_auth_entry():
    from tynex.coding.settings import detect_configured_provider, detect_default_model

    save_auth(
        {
            "anthropic": {"type": "api_key", "key": "sk-a", "updatedAt": 100},
            "openai": {"type": "api_key", "key": "sk-o", "updatedAt": 200},
        }
    )
    assert detect_configured_provider({}) == "openai"
    model = detect_default_model({})
    assert model is not None
    assert (model.provider, model.id) == ("openai", "gpt-5")

    # Without timestamps the most recently added entry wins.
    save_auth({"anthropic": {"type": "api_key", "key": "sk-a"}, "openai": {"type": "api_key", "key": "sk-o"}})
    assert detect_configured_provider({}) == "openai"


def test_detection_falls_back_to_env_and_settings(monkeypatch):
    from tynex.coding.settings import detect_configured_provider

    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert detect_configured_provider({}) == "google"
    assert detect_configured_provider({"apiKeys": {"mistral": "k"}}) == "mistral"
    save_auth({"anthropic": {"type": "api_key", "key": "sk-a"}})
    assert detect_configured_provider({}) == "anthropic"


def test_detection_without_credentials():
    from tynex.coding.settings import detect_configured_provider, detect_default_model

    assert detect_configured_provider({}) is None
    assert detect_default_model({}) is None


def test_agent_session_uses_configured_credentials(tmp_path):
    save_auth({"anthropic": {"type": "api_key", "key": "sk-a"}})
    session = AgentSession(cwd=str(tmp_path), tool_names=["read"], no_session=True)
    assert session.model.provider == "anthropic"
    assert session.model.id == "claude-sonnet-4-5"
    assert session.api_key == "sk-a"


def test_settings_model_beats_auth_detection(tmp_path, isolated_config):
    save_auth({"anthropic": {"type": "api_key", "key": "sk-a"}})
    (isolated_config / "settings.json").write_text(json.dumps({"model": "google/gemini-2.5-pro"}), encoding="utf-8")
    session = AgentSession(cwd=str(tmp_path), tool_names=["read"], no_session=True)
    assert session.model.provider == "google"


def test_agent_session_falls_back_without_credentials(tmp_path):
    # Nothing configured: no model is selected (the interactive mode offers
    # sign-in; the provider catalog itself is fetched from the models API).
    session = AgentSession(cwd=str(tmp_path), tool_names=["read"], no_session=True)
    assert session.model is None
    assert session.api_key is None


def test_interactive_offers_login_when_no_credentials(tmp_path, monkeypatch):
    from tynex.coding import signin as signin_module
    from tynex.tui import app as interactive_module

    session = AgentSession(
        cwd=str(tmp_path), model=resolve_model("anthropic/claude-sonnet-4-5"), tool_names=["read"], no_session=True
    )
    assert session.api_key is None
    app = interactive_module.InteractiveApp(session)
    # Pick "OAuth" in the sign-in method menu; there is a single OAuth provider,
    # so the device flow starts right away.
    monkeypatch.setattr(interactive_module, "select_option", lambda title, options: 0)
    calls: list[str] = []

    def fake_login(**_kwargs) -> bool:
        # A real copilot login fetches the account's catalog from the Copilot API.
        register_models(
            "github-copilot",
            [Model(id="copilot-model", name="Copilot Model", provider="github-copilot", api="github-copilot")],
        )
        calls.append("copilot")
        return True

    monkeypatch.setattr(signin_module, "copilot_login", fake_login)
    app._maybe_ask_login()
    assert calls == ["copilot"]
    # The unusable model was replaced by the provider just signed in to.
    assert session.model is not None and session.model.provider == "github-copilot"


def test_interactive_model_picker_switches_current_provider(tmp_path, monkeypatch):
    from io import StringIO

    from rich.console import Console

    from tynex.tui import app as interactive_module

    session = AgentSession(
        cwd=str(tmp_path), model=resolve_model("anthropic/claude-sonnet-4-5"), tool_names=["read"], no_session=True
    )
    console = Console(file=StringIO(), force_terminal=True, no_color=True, highlight=False, width=110)
    app = interactive_module.InteractiveApp(session, console=console)
    calls: dict[str, object] = {}

    def fake_select(title: str, options: list[str], *, selected: int = 0) -> int:
        if title == "Providers":
            calls["providers"] = options
            return 0  # keep the current provider
        calls.update(title=title, options=options, selected=selected)
        return 1  # switch to the second model of the current provider

    monkeypatch.setattr(interactive_module, "select_option", fake_select)
    app._handle_line("/model")

    providers_shown = calls["providers"]
    assert isinstance(providers_shown, list)
    assert any("Anthropic (anthropic)" in option for option in providers_shown)
    options = calls["options"]
    assert isinstance(options, list)
    assert calls["title"] == "Models — Anthropic"
    assert calls["selected"] == 0  # highlights the active model
    assert "claude-sonnet-4-5" in options[0] and "← current" in options[0]
    assert session.model is not None and session.model.id == "claude-haiku-4-5"
    assert "Switched to anthropic/claude-haiku-4-5" in console.file.getvalue()  # type: ignore[union-attr]


def test_cli_uses_model_from_auth_json(tmp_path):
    env = cli_env(tmp_path)
    (tmp_path / "cli-config" / "auth.json").write_text(
        json.dumps({"faux": {"type": "api_key", "key": "k"}}), encoding="utf-8"
    )
    env["TYNEX_FAUX_SCRIPT"] = json.dumps([text_scenario("picked from auth.json")])
    result = subprocess.run(
        [sys.executable, "-m", "tynex", "-p", "hi"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "picked from auth.json" in result.stdout


# ---------------------------------------------------------------------------
# Interactive mode routing
# ---------------------------------------------------------------------------


def test_interactive_initial_slash_command_is_dispatched(tmp_path, faux_model):
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
    app = InteractiveApp(session)
    # `tynex /quit` must run the command instead of being sent to the model.
    assert app.run("/quit") == 0
    assert session.context.messages == []


def test_interactive_login_copilot_routes_to_device_flow(tmp_path, faux_model, monkeypatch):
    from tynex.coding import signin as signin_module
    from tynex.tui import app as interactive_module

    calls: list[dict] = []

    def fake_login(*, ask, echo, settings, **kwargs) -> bool:
        calls.append({"ask": ask, "echo": echo, "settings": settings})
        return True

    monkeypatch.setattr(signin_module, "copilot_login", fake_login)
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
    app = interactive_module.InteractiveApp(session)
    app._handle_line("/login github-copilot")
    assert len(calls) == 1
    # Credential-free providers (faux/ollama/lmstudio) are never switched away from.
    assert session.model.provider == "faux"


def test_interactive_login_copilot_switches_off_unusable_model(tmp_path, monkeypatch):
    from tynex.coding import signin as signin_module
    from tynex.tui import app as interactive_module

    def fake_login(*, ask, echo, settings, **kwargs) -> bool:
        # The device flow fetches the account's catalog from the Copilot API.
        register_models(
            "github-copilot",
            [Model(id="copilot-model", name="Copilot Model", provider="github-copilot", api="github-copilot")],
        )
        return True

    monkeypatch.setattr(signin_module, "copilot_login", fake_login)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    model = resolve_model("anthropic/claude-sonnet-4-5")
    session = AgentSession(cwd=str(tmp_path), model=model, tool_names=["read"], no_session=True)
    assert session.api_key is None
    app = interactive_module.InteractiveApp(session)
    app._handle_line("/login github-copilot")
    # The current model cannot run without a key, so sign-in lands on Copilot.
    assert session.model is not None and session.model.provider == "github-copilot"


def test_interactive_typed_message_is_not_echoed_twice(tmp_path, faux_model):

    def make_app():
        session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
        return InteractiveApp(session)

    # Input typed at the prompt is already on screen, so the renderer stays quiet.
    faux.reset_script([text_scenario("done")])
    app = make_app()
    app._handle_line("typed message", echo_user=False)
    app._run_thread.join(10)
    assert "typed message" not in app.transcript.text()
    assert "done" in app.transcript.text()

    # Messages passed on the CLI were never displayed, so they are echoed once.
    faux.reset_script([text_scenario("done")])
    app = make_app()
    app._handle_line("cli message")
    app._run_thread.join(10)
    assert app.transcript.text().count("cli message") == 1


def test_run_interactive_reports_missing_terminal(tmp_path, faux_model, monkeypatch):
    from tynex.tui import app as interactive_module

    def fake_run(self, initial_message=None):
        raise interactive_module._NoConsoleScreenBufferError

    monkeypatch.setattr(interactive_module.InteractiveApp, "run", fake_run)
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
    assert interactive_module.run_interactive(session) == 1


# ---------------------------------------------------------------------------
# CLI subprocess smoke tests
# ---------------------------------------------------------------------------


def cli_env(tmp_path):
    config = tmp_path / "cli-config"
    config.mkdir(exist_ok=True)
    settings = faux_settings()
    settings["mcpServers"] = {"context7": {"enabled": False}}  # tests stay offline
    (config / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    # A fresh on-disk catalog cache; production tynex would fetch these from the
    # providers' models APIs, tests must stay offline.
    (config / "models.json").write_text(json.dumps(models_cache_payload()), encoding="utf-8")
    env = dict(os.environ)
    env["TYNEX_CONFIG_DIR"] = str(config)
    for name in PROVIDER_KEY_ENVS:
        env.pop(name, None)
    return env


def run_cli(tmp_path, args, stdin_text=None, script=None, env_extra=None):
    env = cli_env(tmp_path)
    if script is not None:
        env["TYNEX_FAUX_SCRIPT"] = json.dumps(script)
    else:
        env.pop("TYNEX_FAUX_SCRIPT", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "tynex", *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(tmp_path),
    )


def test_cli_print_mode(tmp_path):
    result = run_cli(tmp_path, ["-p", "--model", "faux/test", "say hi"], script=[text_scenario("scripted answer")])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "scripted answer"


def test_cli_plan_mode_prints_the_plan(tmp_path):
    plan = "## Goal\nShip the plan gate."
    script = [
        faux_scenario(
            [
                {"type": "toolcall", "id": "call_1", "name": "submit_plan", "arguments": {"plan": plan}},
                {"type": "stop", "reason": "toolUse"},
            ]
        )
    ]
    result = run_cli(tmp_path, ["-p", "--plan", "--model", "faux/test", "plan it"], script=script)
    assert result.returncode == 0, result.stderr
    assert plan in result.stdout
    assert "tynex -c --permission-mode accept-edits" in result.stderr


def test_cli_plan_conflicts_with_an_explicit_mode(tmp_path):
    result = run_cli(tmp_path, ["-p", "--plan", "--permission-mode", "ask", "--model", "faux/test", "hi"])
    assert result.returncode == 2
    assert "--plan selects plan mode" in result.stderr


def test_cli_continue_restores_session_model(tmp_path):
    first = run_cli(tmp_path, ["-p", "--model", "faux/test", "say hi"], script=[text_scenario("first answer")])
    assert first.returncode == 0, first.stderr
    assert "first answer" in first.stdout

    # Resuming without --model restores the model from the session transcript,
    # so the faux provider is used again instead of the default Anthropic model.
    second = run_cli(tmp_path, ["-c", "-p", "again"], script=[text_scenario("second answer")])
    assert second.returncode == 0, second.stderr
    assert "second answer" in second.stdout


def test_cli_json_mode(tmp_path):
    result = run_cli(tmp_path, ["--json", "--model", "faux/test", "say hi"], script=[text_scenario("json answer")])
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert events[0]["type"] == "agent_start"
    assert events[-1]["type"] == "agent_end"


def test_cli_models_and_sessions_subcommands(tmp_path):
    models = run_cli(tmp_path, ["models"])
    assert models.returncode == 0, models.stderr
    assert "anthropic" in models.stdout
    sessions = run_cli(tmp_path, ["sessions"])
    assert sessions.returncode == 0, sessions.stderr


def test_cli_help_and_error_exit_codes(tmp_path):
    help_result = run_cli(tmp_path, ["-h"])
    assert help_result.returncode == 0
    assert "Usage: tynex" in help_result.stdout
    assert "Commands:" in help_result.stdout
    for command in ("login", "logout", "mcp", "models", "sessions"):
        assert command in help_result.stdout
    for option in ("--resume", "--json", "--permission-mode", "--budget"):
        assert option in help_result.stdout

    bad_model = run_cli(tmp_path, ["-p", "--model", "nope/none", "hi"])
    assert bad_model.returncode == 2
    assert "tynex: No model matches" in bad_model.stderr

    # Bare `-r` (optional value) must not swallow the following option; with no
    # sessions yet it fails cleanly instead of treating `-p` as the id.
    no_session = run_cli(tmp_path, ["-r", "-p", "hi"])
    assert no_session.returncode == 2
    assert "no session to resume" in no_session.stderr


def test_cli_resume_bare_flag_uses_latest_session(tmp_path):
    first = run_cli(tmp_path, ["-p", "--model", "faux/test", "say hi"], script=[text_scenario("answer one")])
    assert first.returncode == 0, first.stderr

    # `-r` without an id resumes the latest session (and restores its model).
    second = run_cli(tmp_path, ["-r", "-p", "again"], script=[text_scenario("answer two")])
    assert second.returncode == 0, second.stderr
    assert "answer two" in second.stdout


# ---------------------------------------------------------------------------
# tynex login / tynex logout
# ---------------------------------------------------------------------------


def test_cli_login_stores_api_key(tmp_path):
    result = run_cli(tmp_path, ["login", "anthropic"], stdin_text="sk-test\n")
    assert result.returncode == 0, result.stderr
    auth = json.loads((tmp_path / "cli-config" / "auth.json").read_text(encoding="utf-8"))
    assert auth["anthropic"]["key"] == "sk-test"
    assert "anthropic/claude-sonnet-4-5" in result.stdout


def test_cli_login_without_terminal_needs_explicit_provider(tmp_path):
    # The sign-in menu is an arrow-key selector, so piping a number no longer works.
    result = run_cli(tmp_path, ["login"], stdin_text="1\n")
    assert result.returncode == 2
    assert "needs a terminal" in result.stderr


def test_cli_logout_removes_stored_credentials(tmp_path):
    login = run_cli(tmp_path, ["login", "openai"], stdin_text="sk-o\n")
    assert login.returncode == 0, login.stderr

    result = run_cli(tmp_path, ["logout", "openai"])
    assert result.returncode == 0, result.stderr
    assert "Signed out of openai" in result.stdout
    assert json.loads((tmp_path / "cli-config" / "auth.json").read_text(encoding="utf-8")) == {}

    # Idempotent when nothing is stored.
    again = run_cli(tmp_path, ["logout", "openai"])
    assert again.returncode == 0, again.stderr
    assert "No stored credentials for openai" in again.stdout


def test_cli_login_copilot_failure_exits_1(tmp_path, monkeypatch):
    import tynex.cli as cli_module

    monkeypatch.setattr(cli_module, "copilot_login", lambda **kw: False)
    assert cli_module.login_command.main(args=["github-copilot"], prog_name="tynex login", standalone_mode=False) == 1


# ---------------------------------------------------------------------------
# tynex permissions, budgets and checkpoints
# ---------------------------------------------------------------------------


def call_scenario(name, arguments, call_id="call_1"):
    return faux_scenario(
        [
            {"type": "toolcall", "id": call_id, "name": name, "arguments": arguments},
            {"type": "stop", "reason": "toolUse"},
        ]
    )


def test_cli_print_mode_explains_denied_tool_calls(tmp_path):
    script = [call_scenario(shell_name, {"command": "echo hi"}), text_scenario("blocked, as expected")]
    result = run_cli(tmp_path, ["-p", "--model", "faux/test", "run it"], script=script)
    assert result.returncode == 0, result.stderr
    assert "blocked, as expected" in result.stdout
    # The user is warned up front and told why the call was denied.
    assert "[tynex] permission mode 'accept-edits'" in result.stderr
    assert f"[permission] deny: {shell_name}" in result.stderr


def test_cli_plan_mode_refuses_edits(tmp_path):
    script = [call_scenario("write", {"path": "planned.txt", "content": "x"}), text_scenario("just a plan")]
    result = run_cli(tmp_path, ["-p", "--model", "faux/test", "--permission-mode", "plan", "plan it"], script=script)
    assert result.returncode == 0, result.stderr
    assert "just a plan" in result.stdout
    assert not (tmp_path / "planned.txt").exists()
    # Read-only modes have nothing to warn about.
    assert "[tynex] permission mode" not in result.stderr


def test_cli_invalid_permission_mode_is_a_usage_error(tmp_path):
    result = run_cli(tmp_path, ["-p", "--model", "faux/test", "--permission-mode", "yolo", "hi"])
    assert result.returncode == 2


def test_cli_budget_stops_with_exit_code_1(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    loop_round = faux_scenario(
        [
            {"type": "usage", "usage": {"input": 10, "output": 5}},
            {"type": "toolcall", "id": "call_1", "name": "read", "arguments": {"path": "a.txt"}},
            {"type": "stop", "reason": "toolUse"},
        ]
    )
    script = [loop_round, loop_round, text_scenario("never reached")]
    result = run_cli(tmp_path, ["-p", "--model", "faux/test", "--budget", "tokens=10", "go"], script=script)
    assert result.returncode == 1
    assert "[limit] token limit reached (15/10 tokens)" in result.stderr
    assert "tynex: stopped" in result.stderr and "token limit reached (15/10 tokens)" in result.stderr


def test_cli_invalid_budget_is_a_usage_error(tmp_path):
    result = run_cli(tmp_path, ["-p", "--model", "faux/test", "--budget", "steps=3", "hi"])
    assert result.returncode == 2
    assert "unknown budget field" in result.stderr


def test_cli_warns_about_broken_mcp_servers_but_keeps_running(tmp_path):
    project = tmp_path / ".tynex"
    project.mkdir()
    (project / "settings.json").write_text(
        json.dumps({"mcpServers": {"broken": {"command": "definitely-not-a-real-command-xyz"}}}), encoding="utf-8"
    )
    script = [text_scenario("answered anyway")]
    result = run_cli(tmp_path, ["-p", "--model", "faux/test", "hi"], script=script)
    assert result.returncode == 0, result.stderr
    assert "answered anyway" in result.stdout
    assert "[tynex] MCP server 'broken'" in result.stderr

    quiet = run_cli(tmp_path, ["-p", "--model", "faux/test", "--no-mcp", "hi"], script=script)
    assert quiet.returncode == 0, quiet.stderr
    assert "MCP server" not in quiet.stderr


def test_cli_accepts_permission_mode_after_the_prompt(tmp_path):
    # A Click group stops parsing options at the first prompt word; the run
    # flags must still apply when they follow it.
    script = [call_scenario("write", {"path": "after.txt", "content": "x"}), text_scenario("planned")]
    result = run_cli(tmp_path, ["-p", "--model", "faux/test", "plan it", "--permission-mode", "plan"], script=script)
    assert result.returncode == 0, result.stderr
    assert "planned" in result.stdout
    assert not (tmp_path / "after.txt").exists()


def test_cli_warns_when_thinking_is_unavailable(tmp_path):
    # The session clamps to the model, so the flag must say it was not honoured.
    result = run_cli(
        tmp_path, ["-p", "--model", "faux/test", "--thinking", "high", "say hi"], script=[text_scenario("ok")]
    )
    assert result.returncode == 0, result.stderr
    assert "needs a reasoning model" in result.stderr
    assert "using 'off'" in result.stderr

    result = run_cli(
        tmp_path, ["-p", "--model", "faux/test", "--thinking", "off", "say hi"], script=[text_scenario("ok")]
    )
    assert result.returncode == 0, result.stderr
    assert "needs a reasoning model" not in result.stderr
