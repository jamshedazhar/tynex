"""Tynex's embedding SDK: run agents from Python.

    from tynex.sdk import Agent, tool

    @tool
    def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    agent = Agent(model="anthropic/claude-sonnet-4-5", tools=[add])
    print(agent.run("What is 2 + 3?").text)

    # async
    result = await agent.arun("And 10 + 20?")
    async for event in agent.astream("Now 7 + 8?"):
        print(event["type"])

Everything else the CLI does is available through the same building blocks:
``coding_tools=True`` for file/shell tools, ``permission_mode``/``approval`` for
tool approvals, ``budget=BudgetLimits(max_cost_usd=...)`` for opt-in caps,
``session="path.jsonl"`` for persistent transcripts,
``mcp_servers=[McpServerConfig(...)]`` for your own MCP servers, and
``load_project=True`` to pick up the project files (AGENTS.md, skills,
extensions).
"""

from tynex.agent.budget import BudgetLimits
from tynex.agent.permissions import ApprovalOutcome, PermissionRequest, PermissionRule
from tynex.agent.types import AgentTool, ToolContext, ToolResult
from tynex.coding.mcp_settings import McpServerConfig
from tynex.sdk.agent import Agent, RunResult
from tynex.sdk.tools import FunctionTool, SchemaError, as_tool, tool

__all__ = [
    "Agent",
    "AgentTool",
    "ApprovalOutcome",
    "BudgetLimits",
    "FunctionTool",
    "McpServerConfig",
    "PermissionRequest",
    "PermissionRule",
    "RunResult",
    "SchemaError",
    "ToolContext",
    "ToolResult",
    "as_tool",
    "tool",
]
