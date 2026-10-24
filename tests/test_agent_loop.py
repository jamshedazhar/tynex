"""Tests for the agent loop: tool execution, steering, errors, truncation."""

import json

from tests.conftest import faux_scenario, text_scenario
from tynex.agent.loop import run_agent_loop
from tynex.agent.types import AgentContext, AgentLoopConfig
from tynex.ai import providers
from tynex.ai.providers import faux
from tynex.ai.types import AssistantMessage, SystemMessage, ToolResultMessage, UserMessage
from tynex.coding.tools import create_tools
from tynex.coding.tools.shell import shell_name


def make_config(model, tools, **overrides):
    def convert(messages):
        return [SystemMessage(content="test system prompt"), *messages]

    config = AgentLoopConfig(model=model, tools=tools, stream_fn=providers.stream, convert_to_llm=convert)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_loop_executes_tool_and_finishes(tmp_path, faux_model):
    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "text", "text": "Writing the file."},
                    {
                        "type": "toolcall",
                        "id": "call_1",
                        "name": "write",
                        "arguments": {"path": "out/hello.txt", "content": "hi there"},
                    },
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("Done."),
        ]
    )
    tools = create_tools(str(tmp_path))
    context = AgentContext(messages=[], tools=tools)
    events: list[dict] = []

    new_messages = run_agent_loop(
        [UserMessage(content="write hello")], context, make_config(faux_model, tools), None, events.append
    )

    assert (tmp_path / "out" / "hello.txt").read_text(encoding="utf-8") == "hi there"
    roles = [m.role for m in new_messages]
    assert roles == ["user", "assistant", "toolResult", "assistant"]
    assert new_messages[-1].text() == "Done."
    assert isinstance(new_messages[2], ToolResultMessage)
    assert new_messages[2].tool_name == "write"
    assert not new_messages[2].is_error

    # The loop writes the final context back into the caller's AgentContext.
    assert context.messages == new_messages

    event_types = [e["type"] for e in events]
    assert event_types[0] == "agent_start"
    assert event_types[-1] == "agent_end"
    assert "tool_execution_start" in event_types
    assert "tool_execution_end" in event_types
    assert "turn_end" in event_types


def test_faux_script_from_file_path(tmp_path, faux_model, monkeypatch):
    """TYNEX_FAUX_SCRIPT accepts a path to a JSON file, not just inline JSON."""
    script_file = tmp_path / "script.json"
    script_file.write_text(json.dumps([text_scenario("from a file")]), encoding="utf-8")
    monkeypatch.setenv("TYNEX_FAUX_SCRIPT", str(script_file))
    faux.reset_script(None)

    tools = create_tools(str(tmp_path))
    context = AgentContext(messages=[], tools=tools)
    new_messages = run_agent_loop(
        [UserMessage(content="hi")], context, make_config(faux_model, tools), None, lambda e: None
    )
    assert new_messages[-1].text() == "from a file"


def test_loop_delivers_steering_message_after_tool(tmp_path, faux_model):
    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "toolcall", "id": "call_1", "name": shell_name, "arguments": {"command": "echo first"}},
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("Responding to the steering message."),
        ]
    )
    tools = create_tools(str(tmp_path))
    pending: list = []

    def before_tool_call(payload, signal):
        if not pending:
            pending.append(UserMessage(content="steer: change of plans"))
        return None

    def take_steering():
        items = list(pending)
        pending.clear()
        return items

    config = make_config(faux_model, tools, before_tool_call=before_tool_call, get_steering_messages=take_steering)
    context = AgentContext(messages=[], tools=tools)
    new_messages = run_agent_loop([UserMessage(content="do something")], context, config)

    assert any(isinstance(m, UserMessage) and m.text() == "steer: change of plans" for m in new_messages)
    assert new_messages[-1].text() == "Responding to the steering message."


def test_loop_length_stop_skips_tool_execution(tmp_path, faux_model):
    faux.reset_script(
        [
            faux_scenario(
                [
                    {
                        "type": "toolcall",
                        "id": "call_1",
                        "name": "write",
                        "arguments": {"path": "never.txt", "content": "x"},
                    },
                    {"type": "stop", "reason": "length"},
                ]
            ),
            text_scenario("Retried."),
        ]
    )
    tools = create_tools(str(tmp_path))
    context = AgentContext(messages=[], tools=tools)
    new_messages = run_agent_loop([UserMessage(content="write")], context, make_config(faux_model, tools))

    assert not (tmp_path / "never.txt").exists()
    result = next(m for m in new_messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    assert "output token limit" in result.content[0].text


def test_loop_terminate_from_a_blocked_call_ends_the_turn(tmp_path, faux_model):
    """One terminable call ends the batch turn even if another call runs."""
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "toolcall", "id": "t1", "name": "read", "arguments": {"path": "a.txt"}},
                    {"type": "toolcall", "id": "t2", "name": "read", "arguments": {"path": "a.txt"}},
                    {"type": "stop", "reason": "toolUse"},
                ]
            )
        ]
    )
    tools = create_tools(str(tmp_path))

    def before_tool_call(payload, signal):
        if payload["toolCall"].id == "t1":
            return {"block": True, "terminate": True, "reason": "stop after this call"}
        return None

    config = make_config(faux_model, tools, before_tool_call=before_tool_call)
    context = AgentContext(messages=[], tools=tools)
    new_messages = run_agent_loop([UserMessage(content="go")], context, config)

    # Both calls are answered, but the loop does not ask the model again.
    assert [message.role for message in new_messages] == ["user", "assistant", "toolResult", "toolResult"]
    results = [message for message in new_messages if isinstance(message, ToolResultMessage)]
    assert results[0].is_error and "stop after this call" in results[0].text()
    assert not results[1].is_error


def test_loop_provider_error_ends_turn(faux_model):
    faux.reset_script([faux_scenario([{"type": "error", "error": "boom"}])])
    tools = create_tools(".")
    context = AgentContext(messages=[], tools=tools)
    new_messages = run_agent_loop([UserMessage(content="hi")], context, make_config(faux_model, tools))

    assistant = new_messages[-1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.stop_reason == "error"
    assert assistant.error_message == "boom"


def test_loop_unknown_tool_reported(faux_model, tmp_path):
    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "toolcall", "id": "call_1", "name": "does_not_exist", "arguments": {}},
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("ok"),
        ]
    )
    tools = create_tools(str(tmp_path))
    context = AgentContext(messages=[], tools=tools)
    new_messages = run_agent_loop([UserMessage(content="x")], context, make_config(faux_model, tools))
    result = next(m for m in new_messages if isinstance(m, ToolResultMessage))
    assert result.is_error
    assert "not found" in result.content[0].text


def test_loop_abort_stops_gracefully(faux_model, tmp_path):
    from tynex.ai.types import AbortSignal

    signal = AbortSignal()
    signal.abort()
    tools = create_tools(str(tmp_path))
    context = AgentContext(messages=[], tools=tools)
    new_messages = run_agent_loop([UserMessage(content="x")], context, make_config(faux_model, tools), signal)
    assistant = new_messages[-1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.stop_reason == "aborted"


def test_loop_abort_mid_tool_batch_answers_every_call(faux_model, tmp_path):
    from typing import ClassVar

    from tynex.agent.types import AgentTool, ToolResult
    from tynex.ai.types import AbortSignal

    class First(AgentTool):
        name = "first"
        label = "first"
        description = "first tool"
        permission_kind = "read"
        parameters: ClassVar[dict] = {"type": "object", "properties": {}}

        def execute(self, tool_call_id, args, signal=None, on_update=None, ctx=None):
            signal.abort()
            return ToolResult.text("first ran")

    class Second(AgentTool):
        name = "second"
        label = "second"
        description = "second tool"
        permission_kind = "read"
        parameters: ClassVar[dict] = {"type": "object", "properties": {}}

        def execute(self, tool_call_id, args, signal=None, on_update=None, ctx=None):
            return ToolResult.text("second ran")

    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "toolcall", "id": "t1", "name": "first", "arguments": {}},
                    {"type": "toolcall", "id": "t2", "name": "second", "arguments": {}},
                    {"type": "stop", "reason": "toolUse"},
                ]
            )
        ]
    )
    tools = [First(), Second()]
    context = AgentContext(messages=[], tools=tools)
    signal = AbortSignal()
    new_messages = run_agent_loop([UserMessage(content="go")], context, make_config(faux_model, tools), signal)

    results = [m for m in new_messages if isinstance(m, ToolResultMessage)]
    assert [result.tool_call_id for result in results] == ["t1", "t2"]
    assert results[0].text() == "first ran"
    assert results[1].text() == "Operation aborted"
    assert results[1].is_error
    assert isinstance(new_messages[-1], AssistantMessage)
    assert new_messages[-1].stop_reason == "aborted"
