"""Tool registry (core tools, read-only toolset and web search)."""

from collections.abc import Callable

from tynex.agent.types import AgentTool
from tynex.coding.tools.edit import EditTool
from tynex.coding.tools.plan import PLAN_TOOL_NAME, SubmitPlanTool
from tynex.coding.tools.read import ReadTool
from tynex.coding.tools.readonly import FindTool, GrepTool, LsTool
from tynex.coding.tools.shell import ShellTool, shell_name
from tynex.coding.tools.websearch import WebSearchTool
from tynex.coding.tools.write import WriteTool

DEFAULT_TOOL_NAMES = ["read", shell_name, "edit", "write", "websearch", PLAN_TOOL_NAME]
READ_ONLY_TOOL_NAMES = ["grep", "find", "ls"]
ALL_TOOL_NAMES = list(dict.fromkeys(DEFAULT_TOOL_NAMES + READ_ONLY_TOOL_NAMES))

_FACTORIES: dict[str, Callable[[str], AgentTool]] = {
    "read": ReadTool,
    "write": WriteTool,
    "edit": EditTool,
    shell_name: ShellTool,
    "grep": GrepTool,
    "find": FindTool,
    "ls": LsTool,
    "websearch": lambda _cwd: WebSearchTool(),
    PLAN_TOOL_NAME: lambda _cwd: SubmitPlanTool(),
}


def create_tools(cwd: str, names: list[str] | None = None) -> list[AgentTool]:
    """Instantiate tools for the given names (default: Tynex's core toolset)."""
    selected = names if names is not None else DEFAULT_TOOL_NAMES
    try:
        return [_FACTORIES[name](cwd) for name in selected]
    except KeyError as exc:
        raise ValueError(f"Unknown tool: {exc.args[0]!r}. Known tools: {', '.join(ALL_TOOL_NAMES)}") from None
