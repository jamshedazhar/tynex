"""The agent loop: streaming, tool execution, and turn coordination."""

import time
from typing import Any

from tynex.agent.types import AgentContext, AgentLoopConfig, EventSink, PrepareNextTurnContext, ToolResult
from tynex.ai.providers.base import LlmContext, StreamOptions
from tynex.ai.types import AbortSignal, AssistantMessage, Message, TextContent, ToolCall, ToolResultMessage


def _default_sink(event: dict[str, Any]) -> None:
    """Discard events when the caller did not supply a sink."""


def _emit_ready(message: Message, emit: EventSink) -> None:
    emit({"type": "message_start", "message": message})
    emit({"type": "message_end", "message": message})


def run_agent_loop(
    prompts: list[Message],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None = None,
    emit: EventSink | None = None,
) -> list[Message]:
    """Start a turn with one or more new messages (usually a single user message)."""
    sink = emit or _default_sink
    new_messages: list[Message] = list(prompts)
    current = AgentContext(messages=[*context.messages, *prompts], tools=context.tools)
    sink({"type": "agent_start"})
    sink({"type": "turn_start"})
    for message in prompts:
        _emit_ready(message, sink)
    final_context = _run_loop(current, new_messages, config, signal, sink)
    context.messages = final_context.messages
    context.tools = final_context.tools
    return new_messages


def _run_loop(
    initial_context: AgentContext,
    new_messages: list[Message],
    initial_config: AgentLoopConfig,
    signal: AbortSignal | None,
    sink: EventSink,
) -> AgentContext:
    current = initial_context
    config = initial_config
    last_completed_turn: PrepareNextTurnContext | None = None
    pending: list[Message] = list(config.get_steering_messages())
    while True:
        has_more_tool_calls = True
        while has_more_tool_calls or pending:
            if config.before_model_call is not None:
                stop_reason = config.before_model_call()
                if stop_reason:
                    sink({"type": "limit_reached", "reason": stop_reason})
                    sink({"type": "agent_end", "messages": new_messages})
                    return current
            prepared: list[Message] = []
            if last_completed_turn is not None:
                result = config.prepare_next_turn(last_completed_turn) if config.prepare_next_turn else None
                if result:
                    if result.context is not None:
                        current = result.context
                    prepared = list(result.messages or [])
                    if result.model is not None:
                        config.model = result.model
                    if result.thinking_level is not None:
                        config.thinking = result.thinking_level
                if not pending:
                    pending = list(config.get_steering_messages())
                sink({"type": "turn_start"})
            for message in [*prepared, *pending]:
                _emit_ready(message, sink)
                current.messages.append(message)
                new_messages.append(message)
            pending = []
            message = _stream_assistant_response(current, config, signal, sink)
            new_messages.append(message)
            if message.stop_reason in ("error", "aborted"):
                last_completed_turn = PrepareNextTurnContext(message, [], current, new_messages)
                sink({"type": "turn_end", "message": message, "toolResults": []})
                sink({"type": "agent_end", "messages": new_messages})
                return current
            tool_calls = message.tool_calls()
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                if message.stop_reason == "length":
                    batch = _fail_truncated_tool_calls(tool_calls, sink)
                else:
                    batch = _execute_tool_calls(current, message, config, signal, sink)
                tool_results = batch["messages"]
                has_more_tool_calls = not batch["terminate"]
                for result_message in tool_results:
                    current.messages.append(result_message)
                    new_messages.append(result_message)
            last_completed_turn = PrepareNextTurnContext(message, list(tool_results), current, new_messages)
            sink({"type": "turn_end", "message": message, "toolResults": tool_results})
            pending = list(config.get_steering_messages())
        follow_ups = list(config.get_follow_up_messages())
        if follow_ups:
            pending = follow_ups
            continue
        break
    sink({"type": "agent_end", "messages": new_messages})
    return current


# ---------------------------------------------------------------------------
# Assistant response streaming
# ---------------------------------------------------------------------------


def _stream_assistant_response(
    current: AgentContext, config: AgentLoopConfig, signal: AbortSignal | None, sink: EventSink
) -> AssistantMessage:
    llm_messages = config.convert_to_llm(current.messages)
    llm_context = LlmContext(messages=llm_messages, tools=[tool.declaration() for tool in config.tools])
    options = StreamOptions(
        api_key=config.api_key,
        thinking=config.thinking,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        signal=signal,
        stealth=config.stealth,
    )
    partial: AssistantMessage | None = None
    added_partial = False
    final: AssistantMessage | None = None
    for event in config.stream_fn(config.model, llm_context, options):
        etype = event.get("type")
        if etype == "start":
            partial = event["partial"]
            current.messages.append(partial)
            added_partial = True
            sink({"type": "message_start", "message": partial})
        elif etype in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            partial = event["partial"]
            if added_partial:
                current.messages[-1] = partial
            sink({"type": "message_update", "message": partial, "delta": event})
        elif etype in ("done", "error"):
            final = event["partial"]
            if added_partial:
                current.messages[-1] = final
            else:
                current.messages.append(final)
                sink({"type": "message_start", "message": final})
            sink({"type": "message_end", "message": final})
            return final
    final = partial or AssistantMessage(
        content=[TextContent(text="")],
        api=config.model.api,
        provider=config.model.provider,
        model=config.model.id,
        stop_reason="error",
        error_message="Provider produced no response",
    )
    if added_partial:
        current.messages[-1] = final
    else:
        current.messages.append(final)
    sink({"type": "message_end", "message": final})
    return final


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _fail_truncated_tool_calls(tool_calls: list[ToolCall], sink: EventSink) -> dict[str, Any]:
    messages: list[ToolResultMessage] = []
    for call in tool_calls:
        sink({"type": "tool_execution_start", "toolCallId": call.id, "toolName": call.name, "args": call.arguments})
        result = ToolResult.text(
            f'Tool call "{call.name}" was not executed: the response hit the output token limit, '
            "so its arguments may be truncated. Re-issue the tool call with complete arguments.",
            is_error=True,
        )
        finalized = {"toolCall": call, "result": result, "isError": True, "durationMs": 0}
        sink(
            {
                "type": "tool_execution_end",
                "toolCallId": call.id,
                "toolName": call.name,
                "result": result,
                "isError": True,
                "durationMs": 0,
            }
        )
        message = _tool_result_message(finalized)
        _emit_ready(message, sink)
        messages.append(message)
    return {"messages": messages, "terminate": False}


def _execute_tool_calls(
    current: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    sink: EventSink,
) -> dict[str, Any]:
    tool_calls = assistant_message.tool_calls()
    finalized_calls: list[dict[str, Any]] = []
    messages: list[ToolResultMessage] = []
    for call in tool_calls:
        sink({"type": "tool_execution_start", "toolCallId": call.id, "toolName": call.name, "args": call.arguments})
        prepared = _prepare_tool_call(current, assistant_message, call, config, signal)
        if prepared["kind"] == "immediate":
            finalized = {
                "toolCall": call,
                "result": prepared["result"],
                "isError": prepared["isError"],
                "durationMs": 0,
            }
        else:
            finalized = _execute_prepared_tool_call(prepared, signal, _update_sink(call, sink))
        sink(
            {
                "type": "tool_execution_end",
                "toolCallId": call.id,
                "toolName": call.name,
                "result": finalized["result"],
                "isError": finalized["isError"],
                "durationMs": finalized.get("durationMs"),
            }
        )
        message = _tool_result_message(finalized)
        _emit_ready(message, sink)
        finalized_calls.append(finalized)
        messages.append(message)
        if signal is not None and signal.aborted:
            break
    # A batch interrupted mid-way must still answer every tool call: providers
    # reject a replayed tool_use block without a matching tool_result.
    for call in tool_calls[len(messages) :]:
        sink({"type": "tool_execution_start", "toolCallId": call.id, "toolName": call.name, "args": call.arguments})
        result = ToolResult.text("Operation aborted", is_error=True)
        finalized = {"toolCall": call, "result": result, "isError": True, "durationMs": 0}
        sink(
            {
                "type": "tool_execution_end",
                "toolCallId": call.id,
                "toolName": call.name,
                "result": result,
                "isError": True,
                "durationMs": 0,
            }
        )
        message = _tool_result_message(finalized)
        _emit_ready(message, sink)
        finalized_calls.append(finalized)
        messages.append(message)
    terminate = bool(finalized_calls) and all(f["result"].terminate for f in finalized_calls)
    return {"messages": messages, "terminate": terminate}


def _prepare_tool_call(
    current: AgentContext,
    assistant_message: AssistantMessage,
    call: ToolCall,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
) -> dict[str, Any]:
    tool = next((t for t in config.tools if t.name == call.name), None)
    if tool is None:
        return {
            "kind": "immediate",
            "result": ToolResult.text(f"Tool {call.name} not found", is_error=True),
            "isError": True,
        }
    try:
        if tool.prepare_arguments:
            prepared_args = tool.prepare_arguments(call.arguments)
            if prepared_args is not call.arguments:
                call = ToolCall(id=call.id, name=call.name, arguments=prepared_args)
        if config.before_tool_call:
            outcome = config.before_tool_call(
                {
                    "assistantMessage": assistant_message,
                    "toolCall": call,
                    "tool": tool,
                    "args": call.arguments,
                    "context": current,
                },
                signal,
            )
            if outcome and outcome.get("block"):
                result = ToolResult.text(outcome.get("reason") or "Tool execution was blocked", is_error=True)
                if outcome.get("terminate"):
                    result.terminate = True
                return {"kind": "immediate", "result": result, "isError": True}
        if signal is not None and signal.aborted:
            return {"kind": "immediate", "result": ToolResult.text("Operation aborted", is_error=True), "isError": True}
        return {"kind": "prepared", "toolCall": call, "tool": tool, "args": call.arguments, "ctx": config.tool_context}
    except Exception as exc:
        return {"kind": "immediate", "result": ToolResult.text(str(exc), is_error=True), "isError": True}


def _execute_prepared_tool_call(prepared: dict[str, Any], signal: AbortSignal | None, on_update) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = prepared["tool"].execute(
            prepared["toolCall"].id, prepared["args"], signal, on_update, prepared.get("ctx")
        )
        is_error = result.is_error
    except Exception as exc:
        result, is_error = ToolResult.text(str(exc), is_error=True), True
    return {
        "toolCall": prepared["toolCall"],
        "result": result,
        "isError": is_error,
        "durationMs": int((time.monotonic() - started) * 1000),
    }


def _update_sink(call: ToolCall, sink: EventSink):
    def on_update(partial: ToolResult) -> None:
        sink(
            {
                "type": "tool_execution_update",
                "toolCallId": call.id,
                "toolName": call.name,
                "args": call.arguments,
                "partialResult": partial,
            }
        )

    return on_update


def _tool_result_message(finalized: dict[str, Any]) -> ToolResultMessage:
    result: ToolResult = finalized["result"]
    return ToolResultMessage(
        tool_call_id=finalized["toolCall"].id,
        tool_name=finalized["toolCall"].name,
        content=list(result.content or []),
        is_error=bool(finalized["isError"]),
        details=result.details,
        duration_ms=finalized.get("durationMs"),
    )
