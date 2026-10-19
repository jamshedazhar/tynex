"""Agent core types: messages, events, tools, and context."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

from tynex.ai.models import Model
from tynex.ai.providers.base import LlmContext, StreamOptions
from tynex.ai.types import AbortSignal, ContentPart, Message, TextContent

StreamFn = Callable[[Model, LlmContext, StreamOptions], Iterator[dict[str, Any]]]


@dataclass
class ToolResult:
    content: list[ContentPart] = field(default_factory=list)
    details: dict[str, Any] | None = None
    is_error: bool = False
    terminate: bool = False

    @staticmethod
    def text(text: str, *, is_error: bool = False, details: dict[str, Any] | None = None) -> Self:
        return ToolResult(content=[TextContent(text=text)], is_error=is_error, details=details)


@dataclass
class ToolContext:
    """Execution context passed to tools (session metadata exposed to bash as TYNEX_*)."""

    cwd: str
    session_id: str | None = None
    session_file: str | None = None
    provider: str | None = None
    model: str | None = None
    thinking_level: str = "off"


UpdateSink = Callable[[ToolResult], None]


class AgentTool:
    """Base class for tools. Subclasses set the metadata attributes and implement execute()."""

    name: str = ""
    label: str = ""
    description: str = ""
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
    prompt_snippet: str | None = None
    prompt_guidelines: ClassVar[list[str]] = []

    # Permission metadata: which policy class the tool belongs to and whether
    # it can change state. Read by tynex.agent.permissions.PermissionGate.
    permission_kind: str = "tool"  # read | edit | shell | web | mcp | tool
    mutating: bool = False

    def permission_subject(self, args: dict[str, Any]) -> str:
        """Primary argument shown in approval prompts and matched by rules."""
        for key in ("command", "path", "url", "query"):
            value = args.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    def prepare_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        """Hook for tolerant argument parsing (e.g. stringified arrays)."""
        return args

    def declaration(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: UpdateSink | None = None,
        ctx: ToolContext | None = None,
    ) -> ToolResult:
        raise NotImplementedError


@dataclass
class AgentContext:
    messages: list[Message] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)


AgentEvent = dict[str, Any]
EventSink = Callable[[AgentEvent], None]


@dataclass
class PrepareNextTurnResult:
    context: AgentContext | None = None
    messages: list[Message] | None = None
    model: Model | None = None
    thinking_level: str | None = None


@dataclass
class PrepareNextTurnContext:
    message: Any
    tool_results: list[Message]
    context: AgentContext
    new_messages: list[Message]


@dataclass
class AgentLoopConfig:
    model: Model
    tools: list[AgentTool]
    stream_fn: StreamFn
    convert_to_llm: Callable[[list[Message]], list[Message]]
    api_key: str | None = None
    thinking: str = "off"
    temperature: float | None = None
    max_tokens: int | None = None
    stealth: bool = False
    tool_context: ToolContext | None = None
    get_steering_messages: Callable[[], list[Message]] = lambda: []
    get_follow_up_messages: Callable[[], list[Message]] = lambda: []
    prepare_next_turn: Callable[[PrepareNextTurnContext], PrepareNextTurnResult | None] | None = None
    before_tool_call: Callable[[dict[str, Any], AbortSignal | None], dict[str, Any] | None] | None = None
    # Called before each model request; a truthy return stops the run with that
    # reason (budget caps) and emits a ``limit_reached`` event.
    before_model_call: Callable[[], str | None] | None = None
