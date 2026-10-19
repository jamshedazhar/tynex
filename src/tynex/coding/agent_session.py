"""AgentSession: wires the agent loop to tools, sessions, compaction and prompts."""

import contextlib
import importlib
import os
import sys
import threading
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tynex.agent.budget import BudgetLimits, BudgetState
from tynex.agent.loop import run_agent_loop
from tynex.agent.permissions import DEFAULT_PERMISSION_MODE, ApprovalFn, PermissionGate, PermissionRule
from tynex.agent.types import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    PrepareNextTurnContext,
    PrepareNextTurnResult,
    ToolContext,
)
from tynex.ai import providers
from tynex.ai.models import Model, default_model_for_provider, format_model
from tynex.ai.providers.base import THINKING_LEVELS, LlmContext, StreamOptions
from tynex.ai.types import (
    AbortSignal,
    AssistantMessage,
    BashExecutionMessage,
    ImageContent,
    Message,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    bash_execution_to_text,
    message_text,
)
from tynex.coding.checkpoints import CheckpointGroup, CheckpointStore
from tynex.coding.mcp_settings import McpServerConfig, resolve_mcp_servers
from tynex.coding.mcp_tools import McpManager
from tynex.coding.model_catalog import ensure_models, resolve_model_retry
from tynex.coding.permissions import PermissionConfig, load_permission_config
from tynex.coding.session import SessionManager
from tynex.coding.settings import detect_configured_provider, find_files, load_settings, resolve_api_key
from tynex.coding.signin import register_copilot_provider
from tynex.coding.skills import PromptTemplate, discover_prompts, discover_skills, format_skills_for_prompt
from tynex.coding.system_prompt import build_system_prompt, load_context_files, load_system_md
from tynex.coding.tools import DEFAULT_TOOL_NAMES, create_tools
from tynex.coding.tools.shell import ShellTool

EventSink = Callable[[dict[str, Any]], None]

PLAN_MODE_NOTE = (
    "Plan mode is active (permission mode 'plan'): investigate with read-only tools and present a plan. "
    "File writes, edits and shell commands are denied until the user switches permission modes."
)


class AgentSession:
    def __init__(
        self,
        cwd: str,
        model: Model | None = None,
        tool_names: list[str] | None = None,
        thinking: str | None = None,
        session_file: str | None = None,
        no_session: bool = False,
        system_prompt: str | None = None,
        append_system_prompt: str | None = None,
        extra_guidelines: list[str] | None = None,
        *,
        permission_mode: str | None = None,
        permission_rules: list[PermissionRule] | None = None,
        approval: ApprovalFn | None = None,
        budget: BudgetLimits | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        checkpoints: bool | None = None,
        load_project: bool = True,
        mcp: bool = True,
        mcp_servers: list[McpServerConfig] | None = None,
        extra_context: list[str] | None = None,
    ) -> None:
        self.cwd = os.path.abspath(cwd)
        self.settings = load_settings(self.cwd)
        register_copilot_provider(self.settings)
        # Model selection: explicit > last used in session > settings > the
        # configured provider's catalog (fetched from its models API on demand).
        # ``None`` when nothing is usable yet — the interactive mode then offers
        # sign-in, other modes refuse to run.
        self.session = SessionManager(self.cwd, session_file=session_file, no_session=no_session)
        self.model: Model | None = model or self._initial_model()
        self.api_key = resolve_api_key(self.model.provider, self.settings) if self.model is not None else None
        # Thinking: explicit > the session's last choice > settings > off.
        # Clamp to the model so a resumed level cannot reach a model that lacks it.
        level = (
            thinking
            if thinking is not None
            else (self.session.thinking_level() or self.settings.get("thinking") or "off")
        )
        available = self.available_thinking_levels()
        level = str(level).strip().lower()
        self.thinking = level if level in available else available[0]
        # Tools
        names = tool_names if tool_names is not None else list(DEFAULT_TOOL_NAMES)
        self.tools: list[AgentTool] = create_tools(self.cwd, names)
        # Extensions & project config (skipped for embedded/SDK use)
        self.extensions = ExtensionAPI()
        self.loaded_extensions = load_extensions(self.cwd, self.extensions) if load_project else []
        self.tools.extend(self.extensions.tools)
        self.prompt_templates: dict[str, PromptTemplate] = discover_prompts(self.cwd) if load_project else {}
        self.skills = discover_skills(self.cwd) if load_project else []
        # MCP servers: explicit servers win, else the settings.json mcpServers entries
        self.mcp_manager: McpManager | None = None
        self.mcp_warnings: list[str] = []
        if mcp_servers is not None:
            servers, setup_warnings = list(mcp_servers), []
        elif mcp:
            servers, setup_warnings = resolve_mcp_servers(self.settings, self.cwd)
        else:
            servers, setup_warnings = [], []
        if servers:
            self.mcp_manager = McpManager(servers)
            self.tools.extend(self.mcp_manager.connect_all())
            setup_warnings = [*setup_warnings, *self.mcp_manager.warnings]
        self.mcp_warnings = setup_warnings
        for warning in self.mcp_warnings:
            print(f"[tynex] {warning}", file=sys.stderr)
        # Permissions: explicit > settings.json > permissions.toml > default.
        self.permission_config: PermissionConfig = (
            load_permission_config(self.cwd) if load_project else PermissionConfig()
        )
        mode = (
            permission_mode
            or self.settings.get("permissionMode")
            or self.permission_config.mode
            or DEFAULT_PERMISSION_MODE
        )
        self.permission_gate = PermissionGate(
            mode, [*self.permission_config.rules, *(permission_rules or [])], approval
        )
        # Budgets and checkpoints
        self.budget = BudgetState()
        self.budget_limits = budget
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._ephemeral_id = uuid.uuid4().hex[:12]
        enabled = checkpoints if checkpoints is not None else bool(self.settings.get("checkpoints", True))
        self.checkpoints_enabled = enabled
        self.checkpoints: CheckpointStore | None = None
        self._bind_checkpoints()
        # System prompt
        self.custom_system_prompt = system_prompt or (load_system_md(self.cwd) if load_project else None)
        self.append_system_prompt = append_system_prompt or self.settings.get("appendSystemPrompt")
        self.extra_guidelines = extra_guidelines or []
        self.context_files = load_context_files(self.cwd) if load_project else []
        blocks = [block.strip() for block in extra_context or [] if block and block.strip()]
        for index, text in enumerate(blocks):
            label = "context" if len(blocks) == 1 else f"context[{index}]"
            self.context_files.append({"path": label, "content": text})
        # Conversation state
        messages = self.session.branch_messages() if not no_session else []
        self.context = AgentContext(messages=messages, tools=self.tools)
        self.stealth = bool(self.settings.get("anthropicStealth", True))
        self._steering: deque[Message] = deque()
        self._follow_ups: deque[Message] = deque()
        self._signal: AbortSignal | None = None
        self._abort_requested = False
        self._active_sink: EventSink | None = None
        self._lock = threading.RLock()
        self._running = False
        self.extensions.run_hook("session_start", {"session": self})

    # -- setup helpers -------------------------------------------------------
    def _bind_checkpoints(self) -> None:
        """Point the checkpoint store at the current session's namespace."""
        if not self.checkpoints_enabled:
            self.checkpoints = None
            return
        namespace = self.session.session_id
        if namespace == "ephemeral":
            namespace = self._ephemeral_id
        self.checkpoints = CheckpointStore(self.cwd, namespace, enabled=True)

    def _initial_model(self) -> Model | None:
        change = self.session.selection()
        if change:
            with contextlib.suppress(ValueError):
                return resolve_model_retry(f"{change[0]}/{change[1]}", self.settings)
        default = self.settings.get("model")
        if default:
            with contextlib.suppress(ValueError):
                return resolve_model_retry(default, self.settings)
        # No explicit choice: use whatever credentials are configured and take
        # the provider's latest model from its (fetched/cached) catalog.
        provider = detect_configured_provider(self.settings)
        if provider is not None:
            ensure_models(provider, self.settings)
            detected = default_model_for_provider(provider)
            if detected is not None:
                return detected
        return None

    def system_prompt_text(self) -> str:
        skills_lines = format_skills_for_prompt(self.skills, "read")
        extra_sections = dict(self.extensions.prompt_sections)
        if self.permission_gate.mode == "plan":
            extra_sections["plan_mode"] = PLAN_MODE_NOTE
        if self.mcp_manager is not None:
            lines = [
                f"{info['name']} ({len(info.get('tools') or [])} tools, {info['transport']})"
                for info in self.mcp_manager.summaries()
                if info.get("connected")
            ]
            if lines:
                extra_sections["mcp"] = (
                    "MCP servers connected for this session: "
                    + ", ".join(lines)
                    + ". Their tools are named mcp__<server>__<tool> and go through the normal permission policy."
                )
        text, _sections = build_system_prompt(
            cwd=self.cwd,
            tools=self.tools,
            custom_prompt=self.custom_system_prompt,
            append_system_prompt=self.append_system_prompt,
            context_files=self.context_files,
            skills_lines=skills_lines,
            extra_sections=extra_sections,
            extra_guidelines=self.extra_guidelines,
        )
        return text

    def tool_context(self) -> ToolContext:
        model = self.model
        return ToolContext(
            cwd=self.cwd,
            session_id=self.session.session_id,
            session_file=self.session.file_path,
            provider=model.provider if model is not None else "",
            model=model.id if model is not None else "",
            thinking_level=self.thinking,
        )

    def _loop_config(self) -> AgentLoopConfig:
        assert self.model is not None  # prompt() refuses to run without a model
        model = self.model
        return AgentLoopConfig(
            model=model,
            tools=self.tools,
            stream_fn=providers.stream,
            convert_to_llm=self._convert_to_llm,
            api_key=self.api_key,
            thinking=self.thinking,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stealth=self.stealth and model.provider == "anthropic",
            tool_context=self.tool_context(),
            get_steering_messages=self._take_steering,
            get_follow_up_messages=self._take_follow_ups,
            prepare_next_turn=self._prepare_next_turn,
            before_tool_call=self._before_tool_call,
            before_model_call=self._check_budget,
        )

    def _check_budget(self) -> str | None:
        """Called before every model request; stops the run on an exceeded cap."""
        self.budget.begin()
        reason = self.budget.exceeded(self.budget_limits)
        if reason is not None:
            return reason
        self.budget.steps += 1
        return None

    def _convert_to_llm(self, messages: list[Message]) -> list[Message]:
        system = SystemMessage(content=self.system_prompt_text())
        converted: list[Message] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            if isinstance(message, BashExecutionMessage):
                # ``!!`` executions stay in the transcript but never reach the model.
                if message.exclude_from_context:
                    continue
                converted.append(UserMessage(content=bash_execution_to_text(message), timestamp=message.timestamp))
                continue
            converted.append(message)
        return [system, *converted]

    # -- queues ---------------------------------------------------------------
    def _take_steering(self) -> list[Message]:
        with self._lock:
            items = list(self._steering)
            self._steering.clear()
            return items

    def _take_follow_ups(self) -> list[Message]:
        with self._lock:
            items = list(self._follow_ups)
            self._follow_ups.clear()
            return items

    def steer(self, content: str | list) -> None:
        """Queue a steering message: delivered after the current tool finishes."""
        with self._lock:
            self._steering.append(UserMessage(content=content))

    def follow_up(self, content: str | list) -> None:
        """Queue a follow-up: delivered after the agent finishes this turn."""
        with self._lock:
            self._follow_ups.append(UserMessage(content=content))

    def abort(self) -> None:
        with self._lock:
            if self._signal is not None:
                self._signal.abort()
            elif self._running:
                # The prompt set _running but has not created its signal yet.
                self._abort_requested = True

    @property
    def signal(self) -> AbortSignal | None:
        return self._signal

    @property
    def running(self) -> bool:
        return self._running

    @property
    def pending_steering(self) -> int:
        """Number of queued steering messages the model has not read yet."""
        with self._lock:
            return len(self._steering)

    def switch_session(self, session_file: str) -> None:
        """Load a different session file and rebuild the conversation."""
        self.session = SessionManager(self.cwd, session_file=session_file)
        change = self.session.selection()
        if change:
            with contextlib.suppress(ValueError):
                self.set_model(resolve_model_retry(f"{change[0]}/{change[1]}", self.settings))
        self.context = AgentContext(messages=self.session.branch_messages(), tools=self.tools)
        self.budget.reset()
        self._bind_checkpoints()

    def new_session(self) -> None:
        self.session = SessionManager(self.cwd)
        self.context = AgentContext(messages=[], tools=self.tools)
        self.budget.reset()
        self._bind_checkpoints()
        if self.model is not None:
            self.session.append_model_change(self.model.provider, self.model.id)

    # -- fork / clone -----------------------------------------------------------
    def _replace_session(self, session_file: str | None) -> None:
        if session_file is None:
            self.context = AgentContext(messages=self.session.branch_messages(), tools=self.tools)
        else:
            self.session = SessionManager(self.cwd, session_file=session_file)
            self.context = AgentContext(messages=self.session.branch_messages(), tools=self.tools)
        self.budget.reset()
        self._bind_checkpoints()

    def clone_session(self) -> str | None:
        """Duplicate the session at its current position."""
        leaf = self.session.current_leaf_id
        if leaf is None:
            return None
        new_file = self.session.create_branched_session(leaf)
        self._replace_session(new_file)
        return new_file

    def user_messages_for_forking(self) -> list[tuple[str, str]]:
        """User messages in the session file that ``/fork`` can branch from."""
        found: list[tuple[str, str]] = []
        for entry in self.session.entries:
            if entry.type != "message":
                continue
            message = entry.message
            if message is None or getattr(message, "role", None) != "user":
                continue
            text = message_text(message)
            if text.strip():
                found.append((entry.id, text))
        return found

    def fork_session(self, entry_id: str) -> tuple[str | None, str]:
        """Fork before the user message ``entry_id``."""
        entry = self.session.find_entry(entry_id)
        if entry is None or entry.type != "message" or getattr(entry.message, "role", None) != "user":
            raise ValueError("Invalid entry ID for forking")
        selected = message_text(entry.message)
        # Forking before the first user message means branching at its parent
        # (usually the session header), i.e. an empty conversation.
        target = entry.parent_id or self.session.header_id
        if target is None:
            raise ValueError("Invalid entry ID for forking")
        new_file = self.session.create_branched_session(target)
        self._replace_session(new_file)
        return new_file, selected

    # -- prompt ---------------------------------------------------------------
    def prompt(self, content: str | list, emit: EventSink | None = None) -> list[Message]:
        """Send a user message and run the agent loop to completion."""
        if self.model is None:
            raise RuntimeError("No model selected — sign in with `tynex login <provider>` or pass --model.")
        with self._lock:
            if self._running:
                raise RuntimeError("Agent is already running; use steer()")
            self._running = True
        signal = AbortSignal()
        if self._abort_requested:
            signal.abort()
            self._abort_requested = False
        self._signal = signal
        sink = self._make_sink(emit)
        self._active_sink = sink
        user_message = UserMessage(content=content)
        try:
            new_messages = run_agent_loop([user_message], self.context, self._loop_config(), signal, sink)
            return new_messages
        finally:
            self._running = False
            self._signal = None
            self._active_sink = None

    def finalize_aborted_turn(self, emit: EventSink | None = None) -> None:
        """Finalize and persist a dangling partial assistant message after a hard interrupt."""
        messages = self.context.messages
        if not messages:
            return
        last = messages[-1]
        if isinstance(last, AssistantMessage) and last.stop_reason is None:
            last.stop_reason = "aborted"
            last.error_message = last.error_message or "Aborted"
            self.session.append_message(last)
            if emit:
                emit({"type": "message_end", "message": last})

    # -- user bash (! / !!) -----------------------------------------------------
    def execute_bash(
        self,
        command: str,
        *,
        exclude_from_context: bool = False,
        signal: AbortSignal | None = None,
        on_update: Callable[[Any], None] | None = None,
    ) -> BashExecutionMessage:
        """Run a ``!``/``!!`` command in the platform shell and record it."""
        result = ShellTool(self.cwd).run_capture(command, signal=signal, on_update=on_update, ctx=self.tool_context())
        message = BashExecutionMessage(
            command=command,
            output=result.text,
            exit_code=None if result.aborted else result.exit_code,
            cancelled=result.aborted,
            truncated=result.truncated,
            full_output_path=result.full_output_path,
            exclude_from_context=exclude_from_context,
        )
        self.session.append_message(message)
        if not exclude_from_context:
            self.context.messages.append(message)
        return message

    # -- event sink -----------------------------------------------------------
    def _make_sink(self, user_emit: EventSink | None) -> EventSink:
        def sink(event: dict[str, Any]) -> None:
            etype = event.get("type")
            if etype == "message_end":
                message = event.get("message")
                if isinstance(message, AssistantMessage):
                    self.budget.add_usage(message.usage)
                    self.budget.tool_calls += len(message.tool_calls())
                if message is not None and not isinstance(message, SystemMessage):
                    self.session.append_message(message)
            if etype == "agent_end":
                self.extensions.run_hook("agent_end", {"session": self, "messages": event.get("messages", [])})
            if user_emit:
                user_emit(event)

        return sink

    # -- compaction -----------------------------------------------------------
    def compaction_threshold(self) -> int:
        assert self.model is not None
        reserve = max(16_384, self.model.max_tokens)
        return max(4_000, self.model.context_window - reserve)

    def estimate_context_tokens(self) -> int:
        system_tokens = len(self.system_prompt_text()) // 4
        messages = self.context.messages
        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]
            if isinstance(message, AssistantMessage) and (message.usage.input or message.usage.total_tokens):
                base = message.usage.input + message.usage.cache_read + message.usage.output
                return base + estimate_tokens(messages[i + 1 :]) + system_tokens
        return estimate_tokens(messages) + system_tokens

    def _prepare_next_turn(self, _last_turn: PrepareNextTurnContext) -> PrepareNextTurnResult | None:
        try:
            if self.estimate_context_tokens() <= self.compaction_threshold():
                return None
            assert self.model is not None  # the loop only runs with a model
            result = compact(self.session, providers.stream, self.model, self.api_key)
            if result is None:
                return None
            self.session.append_compaction(result.summary, result.first_kept_entry_id, result.tokens_before)
            self._emit_compaction(result)
            messages = self.session.branch_messages()
            return PrepareNextTurnResult(context=AgentContext(messages=messages, tools=self.tools), messages=[])
        except Exception:
            return None

    def compact_now(self, instructions: str | None = None) -> bool:
        if self.model is None:
            return False
        result = compact(self.session, providers.stream, self.model, self.api_key, custom_instructions=instructions)
        if result is None:
            return False
        self.session.append_compaction(result.summary, result.first_kept_entry_id, result.tokens_before)
        self.context = AgentContext(messages=self.session.branch_messages(), tools=self.tools)
        self._emit_compaction(result)
        return True

    def _emit_compaction(self, result: "CompactionResult") -> None:
        if self._active_sink:
            self._active_sink(
                {
                    "type": "compaction",
                    "summary": result.summary,
                    "tokensBefore": result.tokens_before,
                    "messagesSummarized": result.messages_summarized,
                }
            )

    # -- model / auth -----------------------------------------------------------
    def set_model(self, model: Model) -> None:
        self.model = model
        self.api_key = resolve_api_key(model.provider, self.settings)
        self.session.append_model_change(model.provider, model.id)
        self.context.tools = self.tools

    def set_model_by_pattern(self, pattern: str) -> Model:
        preferred = self.model.provider if self.model is not None else None
        model = resolve_model_retry(pattern, self.settings, preferred_provider=preferred)
        self.set_model(model)
        return model

    def reload_api_key(self) -> None:
        self.api_key = resolve_api_key(self.model.provider, self.settings) if self.model is not None else None

    # -- thinking level ---------------------------------------------------------
    def available_thinking_levels(self) -> list[str]:
        """Thinking levels the current model accepts."""
        if self.model is None or not self.model.reasoning:
            return ["off"]
        return list(THINKING_LEVELS)

    def set_thinking(self, level: str) -> str:
        """Set the thinking level and record a ``thinking_level_change`` entry."""
        available = self.available_thinking_levels()
        normalized = level.strip().lower()
        if normalized not in THINKING_LEVELS:
            raise ValueError(f'Unknown thinking level "{level}". Available levels: {", ".join(available)}.')
        effective = normalized if normalized in available else available[0]
        if effective != self.thinking:
            self.thinking = effective
            self.session.append_thinking_level_change(effective)
        return effective

    # -- extension hook ----------------------------------------------------------
    def _before_tool_call(self, payload: dict[str, Any], signal: AbortSignal | None) -> dict[str, Any] | None:
        blocked = self.extensions.run_hook("tool_call", payload)
        if blocked and blocked.get("block"):
            return blocked
        tool = payload.get("tool")
        args = payload.get("args") or {}
        decision = self.permission_gate.check(tool, args) if tool is not None else None
        if decision is not None:
            if not decision.allowed:
                self._record_permission(payload.get("toolCall"), decision)
                reason = f"{decision.reason}."
                if self.permission_gate.mode == "plan":
                    reason += " Plan mode is active — explore with read-only tools and present a plan; no changes are allowed."
                else:
                    reason += (
                        " Continue with other actions, ask the user to change permissions, or add an allow rule"
                        " to permissions.toml."
                    )
                return {"block": True, "reason": reason}
            if decision.origin != "mode":
                self._record_permission(payload.get("toolCall"), decision)
            if self.checkpoints is not None and getattr(tool, "mutating", False):
                path = args.get("path")
                if getattr(tool, "permission_kind", "") == "edit" and isinstance(path, str) and path:
                    self.checkpoints.capture(getattr(tool, "name", ""), [path])
        return None

    def _record_permission(self, tool_call: Any, decision: Any) -> None:
        """Persist and broadcast a permission decision (audit trail + live UI)."""
        entry = {
            "tool": decision.request.tool_name,
            "action": decision.action,
            "origin": decision.origin,
            "reason": decision.reason,
            "subject": decision.request.subject,
            "callId": getattr(tool_call, "id", None),
        }
        with contextlib.suppress(Exception):
            self.session.append("permission", **entry)
        if self._active_sink is not None:
            self._active_sink({"type": "permission_decision", **entry})

    # -- permissions / budget -----------------------------------------------------
    def set_permission_mode(self, mode: str) -> str:
        applied = self.permission_gate.set_mode(mode)
        with contextlib.suppress(Exception):
            self.session.append("permission_mode", mode=applied)
        return applied

    def permission_summary(self) -> dict[str, Any]:
        config = self.permission_config
        return {
            "mode": self.permission_gate.mode,
            "rules": [
                {"tool": rule.tool, "match": rule.match, "action": rule.action, "source": rule.source}
                for rule in self.permission_gate.rules
            ],
            "grants": self.permission_gate.session_grants(),
            "files": list(config.files),
            "warnings": list(config.warnings),
            "checkpoints": self.checkpoints_enabled,
            "mcp": self.mcp_manager.summaries() if self.mcp_manager is not None else [],
        }

    def undo_last(self, count: int = 1) -> list[CheckpointGroup]:
        """Restore the files changed by the last ``count`` tool calls."""
        if self.checkpoints is None:
            return []
        groups = self.checkpoints.undo(count)
        if not groups:
            return []
        files = [f.path for group in groups for f in group.files if f.action != "skipped"]
        with contextlib.suppress(Exception):
            self.session.append("undo", groups=[group.to_dict() for group in groups], files=files)
        note = UserMessage(
            content=(
                f"[undo] The user reverted {len(groups)} change(s): {', '.join(files) if files else '(no files)'}. "
                "The working tree now reflects the state before those tool calls."
            )
        )
        self.session.append_message(note)
        self.context.messages.append(note)
        return groups

    # -- stats ----------------------------------------------------------------
    def usage_totals(self) -> Usage:
        """Token/cost totals over every assistant message in the session file."""
        total = Usage()
        for entry in self.session.entries:
            message = entry.message
            if isinstance(message, AssistantMessage):
                total.add(message.usage)
        total.total_tokens = total.input + total.output
        return total

    def stats(self) -> dict[str, Any]:
        counts = {"user": 0, "assistant": 0, "toolResult": 0}
        tool_calls = 0
        for entry in self.session.entries:
            message = entry.message
            role = getattr(message, "role", None)
            if role in counts:
                counts[role] += 1
            if isinstance(message, AssistantMessage):
                tool_calls += len(message.tool_calls())
        usage = self.usage_totals()
        return {
            "sessionId": self.session.session_id,
            "sessionFile": self.session.file_path,
            "label": self.session.label,
            "model": format_model(self.model) if self.model is not None else "(no model)",
            "thinking": self.thinking,
            "messages": len(self.context.messages),
            "totalMessages": sum(counts.values()),
            "userMessages": counts["user"],
            "assistantMessages": counts["assistant"],
            "toolCalls": tool_calls,
            "toolResults": counts["toolResult"],
            "input": usage.input,
            "output": usage.output,
            "cacheRead": usage.cache_read,
            "cacheWrite": usage.cache_write,
            "totalTokens": usage.total_tokens,
            "cost": round(usage.cost, 6),
            "permissionMode": self.permission_gate.mode,
            "budget": self.budget.snapshot(self.budget_limits),
        }

    def session_label(self) -> str | None:
        return self.session.label


DEFAULT_KEEP_TOKENS = 12_000
MIN_MESSAGES_TO_COMPACT = 8

SUMMARIZATION_SYSTEM = (
    "You are a context summarization assistant. Your task is to summarize a coding session "
    "transcript so another engineer (or agent) can continue the work without the original "
    "conversation. Capture: the user's goal(s), key decisions made, files created or modified "
    "(with paths), important context and constraints, current state of the work, and the "
    "immediate next steps. Be concise but complete. Do not mention the summarization process."
)


@dataclass
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    messages_summarized: int


def estimate_message_tokens(message: Message) -> int:
    if isinstance(message, SystemMessage):
        return len(message.content) // 4 + 4
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return len(message.content) // 4 + 4
        total = 4
        for part in message.content:
            if isinstance(part, TextContent):
                total += len(part.text) // 4
            elif isinstance(part, ImageContent):
                total += 800
        return total
    if isinstance(message, AssistantMessage):
        total = 4
        for part in message.content:
            if isinstance(part, TextContent):
                total += len(part.text) // 4
            elif isinstance(part, ThinkingContent):
                total += len(part.thinking) // 4
            elif isinstance(part, ToolCall):
                total += (len(part.name) + len(str(part.arguments))) // 4
        return total
    if isinstance(message, ToolResultMessage):
        return sum(len(p.text) // 4 for p in message.content if isinstance(p, TextContent)) + 4
    if isinstance(message, BashExecutionMessage):
        return len(bash_execution_to_text(message)) // 4 + 4
    return 4


def estimate_tokens(messages: list[Message]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def _render_transcript(messages: list[Message], max_chars: int = 400_000) -> str:
    lines: list[str] = []
    total = 0
    for message in messages:
        if isinstance(message, UserMessage):
            body = message.text()
            line = f"## User\n{body}"
        elif isinstance(message, AssistantMessage):
            parts: list[str] = []
            for part in message.content:
                if isinstance(part, TextContent) and part.text:
                    parts.append(part.text)
                elif isinstance(part, ToolCall):
                    parts.append(f"[tool call] {part.name}({str(part.arguments)[:500]})")
            line = "## Assistant\n" + "\n".join(parts)
        elif isinstance(message, ToolResultMessage):
            body = message.text()
            line = f"## Tool result ({message.tool_name})\n{body[:4000]}"
        elif isinstance(message, BashExecutionMessage):
            line = f"## Shell\n{bash_execution_to_text(message)[:4000]}"
        elif isinstance(message, SystemMessage):
            line = f"## System\n{message.content[:2000]}"
        else:
            continue
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n\n".join(lines)


def _collect_text(stream_fn: Callable, model: Model, context: LlmContext, api_key: str | None) -> str:
    options = StreamOptions(api_key=api_key, thinking="off")
    last: AssistantMessage | None = None
    for event in stream_fn(model, context, options):
        partial = event.get("partial")
        if isinstance(partial, AssistantMessage):
            last = partial
    return last.text().strip() if last else ""


def compact(
    session: SessionManager,
    stream_fn: Callable,
    model: Model,
    api_key: str | None,
    keep_tokens: int = DEFAULT_KEEP_TOKENS,
    custom_instructions: str | None = None,
) -> CompactionResult | None:
    """Summarize older messages; returns None when there is nothing to compact."""
    message_entries = [e for e in session.message_entries() if e.message is not None]
    if len(message_entries) < MIN_MESSAGES_TO_COMPACT:
        return None
    # Walk backwards to find the kept window.
    split = 0
    tokens = 0
    for i in range(len(message_entries) - 1, -1, -1):
        tokens += estimate_message_tokens(message_entries[i].message)
        if tokens > keep_tokens:
            split = i + 1
            break
    # Snap forward to the next user message so tool call/result groups are never
    # split. Mid-turn there is no later user message: snap back to the last one
    # so the current turn is kept and older turns are dropped instead.
    start = split
    while start < len(message_entries) and getattr(message_entries[start].message, "role", None) != "user":
        start += 1
    if start >= len(message_entries):
        start = next(
            (
                i
                for i in range(len(message_entries) - 1, -1, -1)
                if getattr(message_entries[i].message, "role", None) == "user"
            ),
            0,
        )
    split = start
    dropped = message_entries[:split]
    kept = message_entries[split:]
    if not dropped or not kept:
        return None
    dropped_messages = [entry.message for entry in dropped]
    prior_summary = next(
        (entry.data.get("summary") for entry in reversed(session.path_entries()) if entry.type == "compaction"), None
    )
    transcript = _render_transcript(dropped_messages)
    prompt_parts = [f"[Conversation to summarize]\n{transcript}", "[End of conversation]"]
    if prior_summary:
        prompt_parts.insert(0, f"[Summary of even earlier context]\n{prior_summary}")
    if custom_instructions:
        prompt_parts.append(f"Additional instructions: {custom_instructions}")
    prompt_parts.append("Now write the summary.")
    context = LlmContext(
        messages=[SystemMessage(content=SUMMARIZATION_SYSTEM), UserMessage(content="\n\n".join(prompt_parts))], tools=[]
    )
    summary = _collect_text(stream_fn, model, context, api_key)
    if not summary:
        return None
    return CompactionResult(
        summary=summary,
        first_kept_entry_id=kept[0].id,
        tokens_before=estimate_tokens(dropped_messages),
        messages_summarized=len(dropped_messages),
    )


HOOK_EVENTS = ("tool_call", "session_start", "agent_end")


class ExtensionAPI:
    def __init__(self) -> None:
        self.tools: list[AgentTool] = []
        self.commands: dict[str, dict[str, Any]] = {}
        self.hooks: dict[str, list[Callable[[dict[str, Any]], Any]]] = {event: [] for event in HOOK_EVENTS}
        self.prompt_sections: dict[str, str] = {}

    def add_tool(self, tool: AgentTool) -> AgentTool:
        self.tools.append(tool)
        return tool

    def add_command(self, name: str, handler: Callable[[str], Any], help: str = "") -> None:
        self.commands[name] = {"handler": handler, "help": help or f"Extension command {name}"}

    def on(self, event: str, handler: Callable[[dict[str, Any]], Any]) -> Callable:
        if event not in self.hooks:
            raise ValueError(f"Unknown extension hook {event!r}. Known: {', '.join(HOOK_EVENTS)}")
        self.hooks[event].append(handler)
        return handler

    def add_prompt_section(self, name: str, text: str) -> None:
        self.prompt_sections[name] = text

    def run_hook(self, event: str, payload: dict[str, Any]) -> Any:
        for handler in self.hooks.get(event, []):
            try:
                result = handler(payload)
            except Exception:
                print(f"[tynex] extension hook {event!r} failed:\n{traceback.format_exc()}", file=sys.stderr)
                continue
            if result is not None:
                return result
        return None


def load_extensions(cwd: str, api: ExtensionAPI) -> list[str]:
    """Load all extension modules; returns the names of loaded extensions."""
    loaded: list[str] = []
    seen: set[str] = set()
    for file in find_files("extensions/*.py", cwd):
        if file.name.startswith("_"):
            continue
        if file.stem in seen:
            continue  # project extensions shadow built-in ones with the same name
        seen.add(file.stem)
        module_name = f"pi_extension_{file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            setup = getattr(module, "setup", None)
            if callable(setup):
                setup(api)
                loaded.append(file.stem)
        except Exception:
            sys.modules.pop(module_name, None)
            print(f"[tynex] failed to load extension {file}:\n{traceback.format_exc()}", file=sys.stderr)
    return loaded
