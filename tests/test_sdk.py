"""The embedding SDK: tool decorator, sync/async agents, budgets, approvals."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

import pytest

from tests.conftest import faux_scenario, text_scenario
from tynex.agent.budget import BudgetLimits
from tynex.agent.types import AgentTool, ToolResult
from tynex.ai.providers import faux
from tynex.sdk import Agent, McpServerConfig, SchemaError, as_tool, tool

# ---------------------------------------------------------------------------
# Tool decorator
# ---------------------------------------------------------------------------


def test_schema_from_type_hints_and_docstring():
    @tool
    def search(
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        mode: Literal["fast", "deep"] = "fast",
        meta: dict[str, int] | None = None,
    ) -> list[dict]:
        """Search things.

        Args:
            query: What to look for.
            limit: How many results.
            tags: Optional filters.
        """
        return []

    assert search.name == "search"
    assert search.description == "Search things."
    schema = search.parameters
    assert schema["required"] == ["query"]
    assert schema["properties"]["query"] == {"type": "string", "description": "What to look for."}
    assert schema["properties"]["limit"] == {"type": "integer", "description": "How many results.", "default": 10}
    assert schema["properties"]["tags"]["type"] == "array"
    assert schema["properties"]["tags"]["items"] == {"type": "string"}
    assert schema["properties"]["mode"] == {"enum": ["fast", "deep"], "type": "string", "default": "fast"}
    assert schema["properties"]["meta"] == {"type": "object", "additionalProperties": {"type": "integer"}}
    # Optional parameters are not "required" even without defaults
    assert "tags" not in schema["required"]


def test_tool_decorator_with_options_and_name_sanitizing():
    @tool(name="my tool!", description="Custom", read_only=True)
    def whatever(flag: bool) -> str:
        return "ok"

    assert whatever.name == "my_tool_"
    assert whatever.description == "Custom"
    assert whatever.permission_kind == "read"


def test_unsupported_signatures_raise_schema_error():
    with pytest.raises(SchemaError, match=r"\*args"):

        @tool
        def bad(*args):
            return None

    with pytest.raises(SchemaError, match="union"):

        @tool
        def also_bad(value: int | str) -> None:
            return None


def test_execute_coerces_results_and_reports_errors():
    @tool
    def scalars(text: str, count: int) -> dict:
        if text == "boom":
            raise ValueError("nope")
        return {"count": count, "text": text}

    assert json.loads(scalars.execute("id", {"text": "a", "count": 2}).content[0].text) == {"count": 2, "text": "a"}
    missing = scalars.execute("id", {"text": "a"})
    assert missing.is_error and "missing required argument 'count'" in missing.content[0].text
    failed = scalars.execute("id", {"text": "boom", "count": 1})
    assert failed.is_error and "ValueError: nope" in failed.content[0].text

    @tool
    def nothing() -> None:
        """Do nothing."""

    assert nothing.execute("id", {}).content[0].text == "(no output)"

    @tool
    def passthrough() -> ToolResult:
        """Return a result as-is."""
        return ToolResult.text("direct", details={"x": 1})

    result = passthrough.execute("id", {})
    assert result.content[0].text == "direct" and result.details == {"x": 1}


def test_async_tool_runs_from_sync_and_async_context():
    @tool
    async def double(value: int) -> int:
        await asyncio.sleep(0)
        return value * 2

    assert double.execute("id", {"value": 21}).content[0].text == "42"

    async def main():
        return double.execute("id", {"value": 5}).content[0].text

    assert asyncio.run(main()) == "10"


def test_async_tool_error_propagates_from_a_running_loop():
    @tool
    async def failing(value: int) -> int:
        """Always raises."""
        await asyncio.sleep(0)
        raise ValueError(f"boom {value}")

    async def main():
        result = failing.execute("id", {"value": 1})
        assert result.is_error
        assert "ValueError: boom 1" in result.content[0].text
        assert "async tool failed to complete" not in result.content[0].text

    asyncio.run(main())


def test_as_tool_wraps_callables_only():
    def plain(text: str = "hi") -> str:
        """Plain function."""
        return text.upper()

    wrapped = as_tool(plain)
    assert wrapped.name == "plain" and wrapped.execute("id", {}).content[0].text == "HI"
    assert isinstance(as_tool(wrapped), AgentTool)
    with pytest.raises(TypeError):
        as_tool(42)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def tool_call(name, arguments, call_id="call_1"):
    return {"type": "toolcall", "id": call_id, "name": name, "arguments": arguments}


def make_agent(faux_model, **kwargs) -> Agent:
    return Agent(model=faux_model, cwd=kwargs.pop("cwd", None), **kwargs)


def test_run_executes_tools_and_reports_a_result(tmp_path, faux_model):
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    faux.reset_script(
        [
            faux_scenario([tool_call("add", {"a": 2, "b": 3}), {"type": "stop", "reason": "toolUse"}]),
            text_scenario("The answer is 5."),
        ]
    )
    events: list[str] = []
    agent = make_agent(faux_model, cwd=str(tmp_path), tools=[add], on_event=lambda event: events.append(event["type"]))
    result = agent.run("What is 2 + 3?")
    assert result.ok and result.text == "The answer is 5."
    assert result.steps == 2 and result.tool_calls == 1
    assert result.stop_reason == "stop"
    assert result.session_id
    assert "tool_execution_end" in events and "message_end" in events
    assert result.usage.total_tokens > 0


def test_run_without_a_model_explains_what_to_do(tmp_path):
    agent = Agent(model=None, cwd=str(tmp_path))
    with pytest.raises(ValueError, match="no model configured"):
        agent.run("hello")


def test_stream_yields_events_then_ends(tmp_path, faux_model):
    faux.reset_script([text_scenario("streamed answer")])
    agent = make_agent(faux_model, cwd=str(tmp_path))
    events = list(agent.stream("go"))
    kinds = [event["type"] for event in events]
    assert kinds[0] == "agent_start" and kinds[-1] == "agent_end"
    assert any(event["type"] == "message_update" for event in events)
    assert agent.messages[-1].text() == "streamed answer"


def test_async_surface(tmp_path, faux_model):
    faux.reset_script([text_scenario("async answer"), text_scenario("async stream")])
    agent = make_agent(faux_model, cwd=str(tmp_path))

    async def main():
        result = await agent.arun("one")
        assert result.text == "async answer"
        seen = [event["type"] async for event in agent.astream("two")]
        assert "agent_end" in seen

    asyncio.run(main())


def test_budget_cap_sets_limit_reason(tmp_path, faux_model):
    @tool
    def noop() -> str:
        """Do nothing."""
        return "ok"

    loop_round = faux_scenario(
        [
            {"type": "usage", "usage": {"input": 10, "output": 5}},
            tool_call("noop", {}),
            {"type": "stop", "reason": "toolUse"},
        ]
    )
    faux.reset_script([loop_round, loop_round, text_scenario("never")])
    agent = make_agent(faux_model, cwd=str(tmp_path), tools=[noop], budget=BudgetLimits(max_tokens=10))
    result = agent.run("loop")
    assert result.ok is False
    assert result.limit_reason and "token limit" in result.limit_reason


def test_approval_callback_can_deny(tmp_path, faux_model):
    asked: list[str] = []

    @tool
    def risky(path: str) -> str:
        """Write something."""
        return f"wrote {path}"

    def approve(request):
        asked.append(request.summary())
        return "deny"

    faux.reset_script(
        [
            faux_scenario([tool_call("risky", {"path": "x"}), {"type": "stop", "reason": "toolUse"}]),
            text_scenario("I could not do that."),
        ]
    )
    agent = make_agent(
        faux_model,
        cwd=str(tmp_path),
        tools=[risky],
        permission_mode="ask",
        approval=approve,
        # the default SDK mode is "bypass"; opt into asking for this test
    )
    result = agent.run("try")
    assert asked == ["risky: x"]
    assert result.text == "I could not do that."
    assert result.ok


def test_coding_tools_and_permission_modes(tmp_path, faux_model):
    (tmp_path / "data.txt").write_text("payload", encoding="utf-8")
    faux.reset_script(
        [
            faux_scenario([tool_call("read", {"path": "data.txt"}), {"type": "stop", "reason": "toolUse"}]),
            text_scenario("read it"),
        ]
    )
    agent = make_agent(faux_model, cwd=str(tmp_path), coding_tools=["read", "write"])
    assert [some.name for some in agent.tools] == ["read", "write"]
    assert agent.permission_mode == "bypass"  # SDK default: tools you provided just run
    result = agent.run("read the file")
    assert result.text == "read it"
    assert any("payload" in str(getattr(message, "content", "")) for message in result.messages)

    assert agent.set_permission_mode("plan") == "plan"
    assert agent.permission_mode == "plan"


def test_add_tool_accepts_plain_functions(tmp_path, faux_model):
    def greet(name: str = "world") -> str:
        """Greet someone."""
        return f"hello {name}"

    agent = make_agent(faux_model, cwd=str(tmp_path))
    added = agent.add_tool(greet)
    assert added.name == "greet"
    assert agent.tools == [added]
    faux.reset_script(
        [
            faux_scenario([tool_call("greet", {"name": "tynex"}), {"type": "stop", "reason": "toolUse"}]),
            text_scenario("greeted"),
        ]
    )
    assert agent.run("hi").text == "greeted"


def test_usage_cost_and_reset(tmp_path, faux_model):
    faux.reset_script([text_scenario("hi"), text_scenario("again")])
    agent = make_agent(faux_model, cwd=str(tmp_path))
    agent.run("one")
    assert agent.steps == 1
    first_usage = agent.usage
    assert first_usage.total_tokens > 0
    agent.reset_budget()
    assert agent.steps == 0 and agent.usage.total_tokens == first_usage.total_tokens  # totals are from the transcript


def test_sessions_persist_when_requested(tmp_path, faux_model):
    transcript = tmp_path / "session.jsonl"
    faux.reset_script([text_scenario("first"), text_scenario("second")])
    agent = make_agent(faux_model, cwd=str(tmp_path), session=str(transcript))
    agent.run("hello")
    assert transcript.exists()

    resumed = make_agent(faux_model, cwd=str(tmp_path), session=str(transcript))
    assert any("hello" in str(getattr(message, "content", "")) for message in resumed.messages)
    assert resumed.run("again").text == "second"


def test_error_turns_are_not_ok(tmp_path, faux_model):
    faux.reset_script([faux_scenario([{"type": "error", "error": "provider exploded"}])])
    agent = make_agent(faux_model, cwd=str(tmp_path))
    result = agent.run("go")
    assert result.ok is False
    assert result.stop_reason == "error"


def test_agent_context_manager_closes(tmp_path, faux_model):
    with make_agent(faux_model, cwd=str(tmp_path)) as agent:
        assert agent.model is not None


# ---------------------------------------------------------------------------
# Prompt/context wiring and MCP servers
# ---------------------------------------------------------------------------

FAKE_MCP_SERVER = '''"""Minimal stdio MCP server: initialize, tools/list, tools/call."""

import json
import sys

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method is None or "id" not in message:
        continue
    result = None
    if method == "initialize":
        result = {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "serverInfo": {"name": "fake", "version": "1.0"},
        }
    elif method == "tools/list":
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        result = {"tools": [{"name": "echo", "description": "Echo text back.", "inputSchema": schema}]}
    elif method == "tools/call":
        text = (message["params"].get("arguments") or {}).get("text", "")
        result = {"content": [{"type": "text", "text": f"echo:{text}"}]}
    reply = {"jsonrpc": "2.0", "id": message["id"]}
    if result is None:
        reply["error"] = {"code": -32601, "message": "not supported"}
    else:
        reply["result"] = result
    print(json.dumps(reply), flush=True)
'''


def _write_fake_mcp(tmp_path) -> str:
    script = tmp_path / "fake_mcp_server.py"
    script.write_text(FAKE_MCP_SERVER, encoding="utf-8")
    return str(script)


def test_mcp_servers_connect_and_expose_tools(tmp_path, faux_model):
    script = _write_fake_mcp(tmp_path)
    server = McpServerConfig(name="fake", command=sys.executable, args=[script])
    faux.reset_script(
        [
            faux_scenario([tool_call("mcp__fake__echo", {"text": "hi"}), {"type": "stop", "reason": "toolUse"}]),
            text_scenario("echoed"),
        ]
    )
    with make_agent(faux_model, cwd=str(tmp_path), mcp_servers=[server]) as agent:
        assert [tool.name for tool in agent.tools] == ["mcp__fake__echo"]
        assert agent.mcp_warnings == []
        result = agent.run("echo hi")
    assert result.text == "echoed"
    assert any("echo:hi" in str(getattr(message, "content", "")) for message in result.messages)


def test_mcp_true_connects_settings_servers(tmp_path, faux_model):
    script = _write_fake_mcp(tmp_path)
    tynex_dir = tmp_path / ".tynex"
    tynex_dir.mkdir()
    settings = {"mcpServers": {"fake": {"command": sys.executable, "args": [script]}}}
    (tynex_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    with make_agent(faux_model, cwd=str(tmp_path), mcp=True) as agent:
        assert [tool.name for tool in agent.tools] == ["mcp__fake__echo"]


def test_context_blocks_reach_the_system_prompt(tmp_path, faux_model):
    agent = make_agent(faux_model, cwd=str(tmp_path), context=["Answer as a haiku poet.", "Never mention the year."])
    prompt = agent._session.system_prompt_text()
    assert "<project_instructions" in prompt
    assert "Answer as a haiku poet." in prompt and "Never mention the year." in prompt


def test_extra_guidelines_apply_with_a_custom_prompt(tmp_path, faux_model):
    agent = make_agent(faux_model, cwd=str(tmp_path), system="You are terse.", extra_guidelines=["Never use emoji."])
    prompt = agent._session.system_prompt_text()
    assert "You are terse." in prompt
    assert "<guidelines>\n- Never use emoji.\n</guidelines>" in prompt


def test_load_project_reads_agents_md(tmp_path, faux_model):
    (tmp_path / "AGENTS.md").write_text("Project rule: always run tests.", encoding="utf-8")
    agent = make_agent(faux_model, cwd=str(tmp_path), load_project=True)
    assert "Project rule: always run tests." in agent._session.system_prompt_text()


def test_example_script_runs_end_to_end_offline(tmp_path):
    """examples/sdk_agent.py is executed as documented, against the faux provider."""
    import os
    import subprocess
    import sys

    from tests.conftest import PROVIDER_KEY_ENVS

    repo = Path(__file__).resolve().parent.parent
    config = tmp_path / "cfg"
    config.mkdir()
    (config / "settings.json").write_text(
        json.dumps(
            {
                "providers": {
                    "faux": {
                        "api": "faux",
                        "models": [{"id": "test", "name": "Faux", "contextWindow": 200000, "maxTokens": 8192}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    script = [
        faux_scenario(
            [
                {"type": "toolcall", "id": "c1", "name": "word_count", "arguments": {"path": "README.md"}},
                {"type": "stop", "reason": "toolUse"},
            ]
        ),
        text_scenario("Done touring the repository."),
    ]
    env = dict(os.environ, TYNEX_CONFIG_DIR=str(config), TYNEX_FAUX_SCRIPT=json.dumps(script))
    for name in PROVIDER_KEY_ENVS:
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, str(repo / "examples" / "sdk_agent.py"), "--model", "faux/test", "tour the repo"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=repo,
    )
    assert result.returncode == 0, result.stderr
    assert "Done touring the repository." in result.stdout
    assert "-> word_count" in result.stderr
    assert "model call(s)" in result.stderr
