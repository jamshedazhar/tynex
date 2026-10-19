"""Embeddable agent API.

    import asyncio
    from tynex.sdk import Agent, tool

    @tool
    def get_weather(city: str) -> str:
        \"\"\"Current weather for a city.\"\"\"
        return f"22°C and sunny in {city}"

    agent = Agent(model="anthropic/claude-sonnet-4-5", tools=[get_weather])
    result = agent.run("How is the weather in Paris?")
    print(result.text)

    async def main():
        async for event in agent.astream("And in Berlin?"):
            ...

Tools run without prompts by default (like calling a function in your own
program). Pass ``permission_mode="ask"`` (or a name like ``"plan"``) plus an
``approval`` callback when the agent should ask before acting. ``coding_tools``
adds Tynex's built-in file/shell tools; pair them with a budget for unattended
runs.

Custom wiring: ``mcp_servers=[McpServerConfig(...)]`` connects your own MCP
servers (``mcp=True`` uses the ones in settings.json), ``context=[...]`` adds
project-context blocks to the system prompt, and ``load_project=True`` opts
into the project files (AGENTS.md, SYSTEM.md, skills, extensions).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any

from tynex.agent.budget import BudgetLimits
from tynex.agent.permissions import ApprovalFn, PermissionRule, normalize_mode
from tynex.agent.types import AgentTool, ToolResult
from tynex.ai.models import Model, register_custom_providers
from tynex.ai.types import AssistantMessage, Message, Usage
from tynex.coding.agent_session import AgentSession
from tynex.coding.mcp_settings import McpServerConfig
from tynex.coding.model_catalog import resolve_model_retry
from tynex.coding.settings import load_settings
from tynex.sdk.tools import FunctionTool, as_tool, tool

Event = dict[str, Any]

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous AI agent embedded in a Python application. "
    "Use the provided tools to accomplish the user's request, calling several tools in sequence when needed. "
    "Be concise; answer with the facts or the outcome, not a narration of your process."
)


@dataclass
class RunResult:
    """Everything one :meth:`Agent.run` produced."""

    text: str
    messages: list[Message]
    stop_reason: str
    usage: Usage
    cost: float
    steps: int
    tool_calls: int
    session_id: str
    limit_reason: str | None = None

    @property
    def ok(self) -> bool:
        """False when the run ended on an error, an abort or a budget cap."""
        return self.limit_reason is None and self.stop_reason not in ("error", "aborted")


class Agent:
    """A reusable agent: one session, any number of prompts, sync or async."""

    def __init__(
        self,
        model: str | Model | None = None,
        *,
        tools: list[Any] | None = None,
        system: str | None = None,
        append_system: str | None = None,
        cwd: str | None = None,
        thinking: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        permission_mode: str | None = "bypass",
        permission_rules: list[PermissionRule] | None = None,
        approval: ApprovalFn | None = None,
        budget: BudgetLimits | None = None,
        session: str | None = None,
        checkpoints: bool = False,
        mcp: bool = False,
        mcp_servers: list[McpServerConfig] | None = None,
        coding_tools: bool | list[str] = False,
        context: str | list[str] | None = None,
        extra_guidelines: list[str] | None = None,
        load_project: bool = False,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        self.cwd = os.path.abspath(cwd or os.getcwd())
        self.on_event = on_event
        self._run_lock = threading.Lock()
        resolved = self._resolve_model(model, self.cwd)
        builtin: list[AgentTool] = []
        if coding_tools:
            from tynex.coding.tools import create_tools  # only needed when the coding tools are requested

            names = None if coding_tools is True else list(coding_tools)
            builtin = create_tools(self.cwd, names)
        if system is not None:
            system_prompt: str | None = system
        elif load_project:
            system_prompt = None  # SYSTEM.md, or the harness default prompt
        else:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        blocks = [context] if isinstance(context, str) else list(context or [])
        self._session = AgentSession(
            cwd=self.cwd,
            model=resolved,
            tool_names=[],  # SDK tools only; use coding_tools=True for the built-ins
            thinking=thinking,
            no_session=session is None,
            session_file=session,
            system_prompt=system_prompt,
            append_system_prompt=append_system,
            extra_guidelines=extra_guidelines,
            extra_context=blocks,
            permission_mode=normalize_mode(permission_mode) if permission_mode else None,
            permission_rules=permission_rules or [],
            approval=approval,
            budget=budget,
            temperature=temperature,
            max_tokens=max_tokens,
            checkpoints=checkpoints,
            load_project=load_project,
            mcp=mcp,
            mcp_servers=mcp_servers,
        )
        for item in [*builtin, *(tools or [])]:
            self.add_tool(item)
        self._last_limit: str | None = None

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def _resolve_model(model: str | Model | None, cwd: str) -> Model | None:
        if model is None or isinstance(model, Model):
            return model
        settings = load_settings(cwd)
        register_custom_providers(settings.get("providers") or {})
        try:
            return resolve_model_retry(str(model), settings)
        except ValueError as exc:
            raise ValueError(f"could not resolve model {model!r}: {exc}") from exc

    # -- tools ---------------------------------------------------------------
    def add_tool(self, item: Any, *, read_only: bool = False, mutating: bool = False) -> AgentTool:
        """Add a function or AgentTool to the session; returns the AgentTool."""
        converted = item if isinstance(item, AgentTool) else as_tool(item, read_only=read_only, mutating=mutating)
        self._session.tools.append(converted)
        return converted

    @property
    def tools(self) -> list[AgentTool]:
        return self._session.tools

    # -- state ---------------------------------------------------------------
    def set_model(self, model: str | Model) -> Model:
        """Switch models (a pattern string is resolved against settings)."""
        if isinstance(model, Model):
            self._session.set_model(model)
            return model
        return self._session.set_model_by_pattern(str(model))

    @property
    def model(self) -> Model | None:
        return self._session.model

    @property
    def messages(self) -> list[Message]:
        return self._session.context.messages

    @property
    def usage(self) -> Usage:
        return self._session.usage_totals()

    @property
    def cost(self) -> float:
        return self.usage.cost

    @property
    def session_id(self) -> str:
        return self._session.session.session_id

    @property
    def permission_mode(self) -> str:
        return self._session.permission_gate.mode

    @property
    def mcp_warnings(self) -> list[str]:
        """Setup warnings from MCP servers (missing runtimes, failed connects)."""
        return self._session.mcp_warnings

    def set_permission_mode(self, mode: str) -> str:
        return self._session.set_permission_mode(mode)

    @property
    def budget(self) -> BudgetLimits | None:
        return self._session.budget_limits

    @property
    def steps(self) -> int:
        return self._session.budget.steps

    @property
    def tool_calls(self) -> int:
        return self._session.budget.tool_calls

    def compact(self, instructions: str | None = None) -> bool:
        """Summarize older messages; True when a compaction happened."""
        return self._session.compact_now(instructions)

    def abort(self) -> None:
        """Ask the current run to stop at the next safe point."""
        self._session.abort()

    def reset_budget(self) -> None:
        """Restart token/cost/step counters (limits stay in place)."""
        self._session.budget.reset()

    def close(self) -> None:
        self.abort()
        session = getattr(self, "_session", None)
        if session is not None and session.mcp_manager is not None:
            session.mcp_manager.close()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- running -------------------------------------------------------------
    def run(self, prompt: str | list[Any], *, on_event: Callable[[Event], None] | None = None) -> RunResult:
        """Send one message and run the agent loop to completion (blocking)."""
        if self._session.model is None:
            raise ValueError("no model configured — pass model='provider/model-id' or sign in with `tynex login`")
        collected: list[Message] = []
        self._last_limit = None

        def emit(event: Event) -> None:
            if event.get("type") == "limit_reached":
                self._last_limit = str(event.get("reason") or "limit reached")
            handler = on_event or self.on_event
            if handler is not None:
                handler(event)

        with self._run_lock:
            collected = self._session.prompt(prompt, emit=emit)
        last = next((m for m in reversed(collected) if isinstance(m, AssistantMessage)), None)
        usage = self._session.usage_totals()
        return RunResult(
            text=last.text().strip() if last is not None and last.text() else "",
            messages=list(collected),
            stop_reason=(last.stop_reason if last is not None and last.stop_reason else "stop"),
            usage=usage,
            cost=usage.cost,
            steps=self._session.budget.steps,
            tool_calls=self._session.budget.tool_calls,
            session_id=self.session_id,
            limit_reason=self._last_limit,
        )

    def stream(self, prompt: str | list[Any], *, on_event: Callable[[Event], None] | None = None) -> Iterator[Event]:
        """Run one prompt in a worker thread and yield its events (blocking iterator)."""
        events: queue.Queue[Event | None] = queue.Queue()
        handler = on_event or self.on_event

        def emit(event: Event) -> None:
            if event.get("type") == "limit_reached":
                self._last_limit = str(event.get("reason") or "limit reached")
            if handler is not None:
                handler(event)
            events.put(event)

        def worker() -> None:
            try:
                self.run(prompt, on_event=emit)
            finally:
                events.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                item = events.get()
                if item is None:
                    break
                yield item
        finally:
            if thread.is_alive():  # consumer stopped early: abort the run
                self.abort()
            thread.join(timeout=10)

    # -- async surface -------------------------------------------------------
    async def arun(self, prompt: str | list[Any], *, on_event: Callable[[Event], None] | None = None) -> RunResult:
        """Async wrapper around :meth:`run` (runs the loop in a worker thread)."""
        return await asyncio.to_thread(self.run, prompt, on_event=on_event)

    async def astream(
        self, prompt: str | list[Any], *, on_event: Callable[[Event], None] | None = None
    ) -> AsyncIterator[Event]:
        """Async iterator over :meth:`stream` events."""
        iterator = self.stream(prompt, on_event=on_event)
        sentinel = object()
        while True:
            item = await asyncio.to_thread(next, iterator, sentinel)
            if item is sentinel:
                break
            yield item

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


__all__ = ["Agent", "FunctionTool", "RunResult", "ToolResult", "as_tool", "tool"]
