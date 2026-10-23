"""Tests for the tynex-alignment features: bash execution messages, thinking
levels and session export."""

import json
from pathlib import Path

import pytest

from tynex.ai.models import resolve_model
from tynex.ai.providers.base import THINKING_LEVELS
from tynex.ai.types import (
    AssistantMessage,
    BashExecutionMessage,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
    bash_execution_to_text,
    message_from_dict,
    message_text,
    message_to_dict,
)
from tynex.coding.agent_session import AgentSession
from tynex.coding.session import (
    SessionManager,
    export_session_to_html,
    export_session_to_jsonl,
    serialize_session_branch,
)
from tynex.coding.settings import remember_setting
from tynex.coding.tools.shell import shell_name, shell_prompt


def make_agent_session(tmp_path: Path, *, session_file: str = "session.jsonl") -> AgentSession:
    return AgentSession(
        cwd=str(tmp_path), model=resolve_model("faux/test"), tool_names=[], session_file=str(tmp_path / session_file)
    )


def append_exchange(session: SessionManager, user_text: str, assistant_text: str):
    user = session.append_message(UserMessage(content=user_text))
    assistant = session.append_message(AssistantMessage(content=[TextContent(text=assistant_text)]))
    return user, assistant


# ---------------------------------------------------------------------------
# BashExecutionMessage
# ---------------------------------------------------------------------------


def test_bash_execution_message_serializes_round_trip():
    message = BashExecutionMessage(
        command="echo hi",
        output="hi",
        exit_code=0,
        truncated=True,
        full_output_path="C:/tmp/full.txt",
        exclude_from_context=True,
        timestamp=1234,
    )
    data = message_to_dict(message)
    assert data["role"] == "bashExecution"
    assert data["command"] == "echo hi"
    assert data["exitCode"] == 0
    assert data["truncated"] is True
    assert data["fullOutputPath"] == "C:/tmp/full.txt"
    assert data["excludeFromContext"] is True

    restored = message_from_dict(data)
    assert isinstance(restored, BashExecutionMessage)
    assert restored == message


def test_bash_execution_to_text_matches_pi():
    assert bash_execution_to_text(BashExecutionMessage(command="true")) == "Ran `true`\n(no output)"
    assert (
        bash_execution_to_text(BashExecutionMessage(command="ls", output="a\nb", exit_code=0))
        == "Ran `ls`\n```\na\nb\n```"
    )
    assert (
        bash_execution_to_text(BashExecutionMessage(command="false", output="boom", exit_code=2))
        == "Ran `false`\n```\nboom\n```\n\nCommand exited with code 2"
    )
    assert (
        bash_execution_to_text(BashExecutionMessage(command="sleep 9", output="", cancelled=True))
        == "Ran `sleep 9`\n(no output)\n\n(command cancelled)"
    )
    truncated = BashExecutionMessage(command="yes", output="y", truncated=True, full_output_path="C:/tmp/out.txt")
    assert bash_execution_to_text(truncated).endswith("[Output truncated. Full output: C:/tmp/out.txt]")
    assert message_text(BashExecutionMessage(command="true")) == "Ran `true`\n(no output)"


def test_convert_to_llm_includes_bang_and_skips_bangbang(tmp_path):
    session = make_agent_session(tmp_path)
    included = BashExecutionMessage(command="echo hi", output="hi", exit_code=0, timestamp=42)
    excluded = BashExecutionMessage(command="echo secret", output="secret", exclude_from_context=True)

    converted = session._convert_to_llm([UserMessage(content="hello"), included, excluded])
    # system prompt + plain user + bash execution
    assert len(converted) == 3
    assert converted[1].content == "hello"
    assert converted[2].content == "Ran `echo hi`\n```\nhi\n```"
    assert converted[2].timestamp == 42


def test_convert_to_llm_reuses_its_prefix_without_stale_messages(tmp_path):
    session = make_agent_session(tmp_path)
    messages = [UserMessage(content="one"), AssistantMessage(content=[TextContent(text="two")])]
    session._convert_to_llm(messages)

    # Append, and replace the last message in place: the cached prefix must be
    # reused while the mutable tail is converted again.
    messages.append(UserMessage(content="three"))
    assert [m.text() for m in session._convert_to_llm(messages)[1:]] == ["one", "two", "three"]
    messages[-1] = UserMessage(content="replaced")
    assert [m.text() for m in session._convert_to_llm(messages)[1:]] == ["one", "two", "replaced"]

    # A rebuilt list with the same objects (how the loop calls it) still hits.
    rebuilt = [*messages, UserMessage(content="four")]
    assert [m.text() for m in session._convert_to_llm(rebuilt)[1:]] == ["one", "two", "replaced", "four"]

    # A different history that merely starts with the same message must not
    # inherit the cached conversion.
    other = [messages[0], UserMessage(content="different")]
    assert [m.text() for m in session._convert_to_llm(other)[1:]] == ["one", "different"]


def test_estimate_context_tokens_tracks_appends(tmp_path):
    from tynex.coding.compaction import estimate_tokens

    session = make_agent_session(tmp_path)
    messages = [UserMessage(content="a" * 400), UserMessage(content="b" * 400)]
    session.context.messages = messages
    system = len(session.system_prompt_text()) // 4
    assert session.estimate_context_tokens() == estimate_tokens(messages) + system

    messages.append(UserMessage(content="c" * 400))
    assert session.estimate_context_tokens() == estimate_tokens(messages) + system

    messages[-1] = UserMessage(content="d" * 800)
    assert session.estimate_context_tokens() == estimate_tokens(messages) + system


def test_execute_bash_records_bang_and_bangbang(tmp_path):
    session = make_agent_session(tmp_path)

    echo = "echo bang-output" if shell_name == "bash" else "Write-Output bang-output"
    message = session.execute_bash(echo)
    assert message.output.strip() == "bang-output"
    assert message.exit_code == 0
    assert message.cancelled is False
    assert message.role == "bashExecution"
    assert session.context.messages[-1] is message
    assert session.session.entries[-1].message.role == "bashExecution"

    before = list(session.context.messages)
    secret = "echo secret-output" if shell_name == "bash" else "Write-Output secret-output"
    excluded = session.execute_bash(secret, exclude_from_context=True)
    assert excluded.exclude_from_context is True
    assert session.context.messages == before  # !! stays out of the model context
    assert session.session.entries[-1].message.exclude_from_context is True

    failed = session.execute_bash("exit 3")
    assert failed.exit_code == 3
    assert "Command exited with code 3" in bash_execution_to_text(failed)


# ---------------------------------------------------------------------------
# Thinking levels
# ---------------------------------------------------------------------------


def test_thinking_levels_append_change_entries(tmp_path):
    session = make_agent_session(tmp_path)
    session.model = resolve_model("anthropic/claude-sonnet-4-5")

    assert session.available_thinking_levels() == list(THINKING_LEVELS)
    assert session.set_thinking("high") == "high"
    assert session.thinking == "high"
    entry = session.session.entries[-1]
    assert entry.type == "thinking_level_change"
    assert entry.data["thinkingLevel"] == "high"
    assert session.session.thinking_level() == "high"

    session.set_thinking("HIGH")  # unchanged: no duplicate entry
    assert len([e for e in session.session.entries if e.type == "thinking_level_change"]) == 1

    assert session.set_thinking("minimal") == "minimal"
    assert session.session.thinking_level() == "minimal"

    with pytest.raises(ValueError, match='Unknown thinking level "turbo"'):
        session.set_thinking("turbo")


def test_thinking_clamps_for_models_without_reasoning(tmp_path):
    session = make_agent_session(tmp_path)
    assert session.available_thinking_levels() == ["off"]
    assert session.set_thinking("high") == "off"
    assert session.thinking == "off"
    assert not [e for e in session.session.entries if e.type == "thinking_level_change"]


def test_thinking_level_survives_resume(tmp_path):
    path = str(tmp_path / "session.jsonl")
    model = resolve_model("anthropic/claude-sonnet-4-5")
    first = AgentSession(cwd=str(tmp_path), model=model, tool_names=[], session_file=path)
    assert first.set_thinking("medium") == "medium"

    resumed = AgentSession(cwd=str(tmp_path), model=model, tool_names=[], session_file=path)
    assert resumed.thinking == "medium"

    # An explicit choice (the CLI --thinking flag) still wins over the session.
    overridden = AgentSession(cwd=str(tmp_path), model=model, tool_names=[], session_file=path, thinking="off")
    assert overridden.thinking == "off"


def test_thinking_level_falls_back_to_settings(tmp_path):
    remember_setting("thinking", "low")
    model = resolve_model("anthropic/claude-sonnet-4-5")
    session = AgentSession(cwd=str(tmp_path), model=model, tool_names=[], no_session=True)
    assert session.thinking == "low"


def test_thinking_level_is_off_when_the_resumed_model_lacks_reasoning(tmp_path):
    path = str(tmp_path / "session.jsonl")
    reasoning_model = resolve_model("anthropic/claude-sonnet-4-5")
    first = AgentSession(cwd=str(tmp_path), model=reasoning_model, tool_names=[], session_file=path)
    assert first.set_thinking("high") == "high"

    resumed = AgentSession(cwd=str(tmp_path), model=resolve_model("faux/test"), tool_names=[], session_file=path)
    assert resumed.thinking == "off"


def test_thinking_budgets_are_monotonic_and_bounded():
    from tynex.ai.providers.base import thinking_budget

    assert thinking_budget("off") is None
    budgets = [thinking_budget(level) for level in THINKING_LEVELS if level != "off"]
    assert all(budget is not None for budget in budgets)
    assert budgets == sorted(budgets)
    assert budgets[-1] <= 32_000


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_serialize_session_branch_rewrites_header_and_parents(tmp_path):
    session = SessionManager(str(tmp_path))
    first, second = append_exchange(session, "hello", "hi")
    text = serialize_session_branch(session)
    records = [json.loads(line) for line in text.strip().splitlines()]

    assert records[0]["type"] == "session"
    assert records[0]["id"] == session.session_id
    assert records[0]["parentId"] is None
    assert [record["id"] for record in records[1:]] == [first.id, second.id]
    assert records[1]["parentId"] == records[0]["id"]
    assert records[2]["parentId"] == records[1]["id"]
    assert text.endswith("\n")


def test_export_jsonl_writes_branch_and_default_name(tmp_path):
    session = make_agent_session(tmp_path)
    append_exchange(session.session, "hello", "hi")
    target = tmp_path / "out" / "export.jsonl"

    path = export_session_to_jsonl(session.session, str(target))
    assert Path(path) == target
    assert b"\r\n" not in target.read_bytes()  # JSONL is LF on every OS
    records = [json.loads(line) for line in target.read_text(encoding="utf-8").strip().splitlines()]
    assert records[0]["id"] == session.session.session_id

    default = export_session_to_jsonl(session.session, cwd=str(tmp_path))
    assert Path(default).name.startswith("session-")
    assert default.endswith(".jsonl")


def test_export_html_escapes_content_and_renders_all_entry_types(tmp_path):
    session = make_agent_session(tmp_path)
    assistant = AssistantMessage(content=[TextContent(text="hello <b>world</b>")])
    session.session.append_message(UserMessage(content="<script>alert(1)</script>"))
    session.session.append_message(assistant)
    session.session.append_message(
        BashExecutionMessage(command="echo hi", output="hi", exit_code=0, exclude_from_context=True)
    )
    session.session.append_message(
        ToolResultMessage(tool_call_id="t1", tool_name="read", content=[TextContent(text="file body")], is_error=True)
    )
    session.session.append_label("demo label")

    path = export_session_to_html(session.session, cwd=str(tmp_path))
    page = Path(path).read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "hello &lt;b&gt;world&lt;/b&gt;" in page
    assert f"{shell_prompt} echo hi" in page
    assert "not sent to the model" in page
    assert "tool result — read" in page
    assert "demo label" in page
    assert session.session.session_id in page
    assert Path(path).name.startswith("tynex-session-")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_aggregate_all_entries(tmp_path):
    session = make_agent_session(tmp_path)
    _first, assistant = append_exchange(session.session, "hello", "hi")
    usage = Usage(input=1_000, output=200, cache_read=100, cache_write=50, cost=0.01)
    assistant.data["message"]["usage"] = usage.to_dict()
    session.session.append_message(
        ToolResultMessage(tool_call_id="t1", tool_name="read", content=[TextContent(text="x")])
    )
    session.session.append_message(BashExecutionMessage(command="true"))

    stats = session.stats()
    assert stats["sessionId"] == session.session.session_id
    assert stats["userMessages"] == 1
    assert stats["assistantMessages"] == 1
    assert stats["toolResults"] == 1
    assert stats["totalMessages"] == 3  # bash executions are not message counts in tynex
    assert stats["toolCalls"] == 0
    assert stats["input"] == 1_000
    assert stats["output"] == 200
    assert stats["cacheRead"] == 100
    assert stats["cacheWrite"] == 50
    assert stats["totalTokens"] == 1_200
    assert stats["cost"] == 0.01
    assert stats["thinking"] == "off"
    assert stats["messages"] == len(session.context.messages)
