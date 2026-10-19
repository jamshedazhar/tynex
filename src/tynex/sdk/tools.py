"""Turn plain Python functions into agent tools.

The JSON schema is derived from type hints (with docstring ``Args:`` sections
becoming parameter descriptions), so a tool is just a typed function:

    from tynex.sdk import tool

    @tool
    def search_orders(customer: str, limit: int = 10) -> list[dict]:
        \"\"\"Find recent orders for a customer.

        Args:
            customer: Customer name or id.
            limit: Maximum number of orders to return.
        \"\"\"
        return db.search(customer, limit)

Async functions are supported: they run on their own event loop, also when the
agent itself is driven from asyncio.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
import types
import typing
from typing import Any

from tynex.agent.types import AgentTool, ToolContext, ToolResult
from tynex.ai.types import TextContent

_SIMPLE_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_ARG_LINE_RE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


class SchemaError(TypeError):
    """A function signature that cannot be turned into a tool schema."""


get_origin = typing.get_origin


def _schema_for(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is None or annotation is Any:
        return {}
    if isinstance(annotation, type) and annotation in _SIMPLE_TYPES:
        return {"type": _SIMPLE_TYPES[annotation]}
    origin = get_origin(annotation)
    if origin is typing.Literal:
        values = list(typing.get_args(annotation))
        schema: dict[str, Any] = {"enum": values}
        if values and all(isinstance(value, bool) for value in values):
            schema["type"] = "boolean"
        elif values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            schema["type"] = "integer"
        elif values and all(isinstance(value, str) for value in values):
            schema["type"] = "string"
        return schema
    if origin in (typing.Union, types.UnionType):
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _schema_for(args[0])
        raise SchemaError(f"unsupported union annotation {annotation!r} — only Optional[X] is supported")
    if origin in (list, set, tuple, frozenset) or annotation in (list, set, tuple, frozenset):
        args = typing.get_args(annotation)
        items = _schema_for(args[0]) if args and args[0] is not Ellipsis else {}
        schema = {"type": "array"}
        if items:
            schema["items"] = items
        return schema
    if origin is dict or annotation is dict:
        args = typing.get_args(annotation)
        if len(args) == 2:
            value_schema = _schema_for(args[1])
            if value_schema:
                return {"type": "object", "additionalProperties": value_schema}
        return {"type": "object"}
    raise SchemaError(
        f"unsupported annotation {annotation!r} — use str, int, float, bool, list[...], dict[...], Literal or Optional"
    )


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return (summary, {parameter: description}) from a Google-style docstring."""
    if not doc:
        return "", {}
    description: list[str] = []
    arg_docs: dict[str, str] = {}
    section: str | None = None
    current: str | None = None
    for line in inspect.cleandoc(doc).splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered in ("args:", "arguments:", "parameters:"):
            section, current = "args", None
            continue
        if lowered in ("returns:", "yields:", "raises:", "examples:", "example:", "note:"):
            section, current = None, None
            continue
        if section == "args":
            match = _ARG_LINE_RE.match(stripped)
            if match:
                current = match.group(1)
                arg_docs[current] = match.group(2).strip()
            elif current and stripped:
                arg_docs[current] = f"{arg_docs[current]} {stripped}".strip()
            continue
        if section is None and stripped:
            description.append(stripped)
    return " ".join(description).strip(), arg_docs


def _coerce_result(value: Any) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    if value is None:
        return ToolResult.text("(no output)")
    if isinstance(value, str):
        return ToolResult.text(value)
    if isinstance(value, (dict, list, tuple, int, float, bool)):
        return ToolResult.text(json.dumps(value, ensure_ascii=False, default=str))
    text = getattr(value, "text", None)
    if isinstance(text, str):  # duck-typed content parts
        return ToolResult(content=[TextContent(text=text)])
    return ToolResult.text(str(value))


def _run_async(coro: Any) -> Any:
    """Run a coroutine from synchronous code, even inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised on the calling thread below
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        error = box["error"]
        raise error.with_traceback(error.__traceback__)
    if "value" not in box:
        raise RuntimeError("async tool failed to complete")
    return box["value"]


class FunctionTool(AgentTool):
    """An :class:`AgentTool` backed by a plain Python function."""

    def __init__(
        self,
        func: Any,
        *,
        name: str | None = None,
        description: str | None = None,
        read_only: bool = False,
        mutating: bool = False,
    ) -> None:
        if not callable(func):
            raise TypeError("tool() expects a callable")
        raw_name = name or getattr(func, "__name__", "") or ""
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)
        if not _NAME_RE.match(sanitized):
            raise SchemaError(f"invalid tool name {raw_name!r} — use letters, digits, '_' or '-'")
        self.func = func
        self.name = sanitized
        self.label = sanitized
        doc_summary, arg_docs = _parse_docstring(inspect.getdoc(func))
        self.description = description or doc_summary or sanitized
        try:
            hints = typing.get_type_hints(func)
        except Exception:  # unresolvable annotations: fall back to raw annotations
            hints = {}
        signature = inspect.signature(func)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param in signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise SchemaError(f"{sanitized}: *args/**kwargs parameters are not supported")
            annotation = hints.get(param_name, param.annotation)
            schema = _schema_for(annotation)
            doc = arg_docs.get(param_name)
            if doc:
                schema = {**schema, "description": doc}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            elif isinstance(param.default, (str, int, float, bool)):
                schema = {**schema, "default": param.default}
            properties[param_name] = schema
        self.parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            self.parameters["required"] = required
        self._params = signature.parameters
        self._is_async = inspect.iscoroutinefunction(func)
        self.read_only = bool(read_only)
        self.mutating = bool(mutating)
        self.permission_kind = "read" if read_only else ("edit" if mutating else "tool")

    def permission_subject(self, args: dict[str, Any]) -> str:
        for key in ("path", "command", "query", "url", "name"):
            value = args.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
        ctx: ToolContext | None = None,
    ) -> ToolResult:
        if not isinstance(args, dict):
            return ToolResult.text(f"tool {self.name}: arguments must be an object", is_error=True)
        kwargs: dict[str, Any] = {}
        for param_name, param in self._params.items():
            if param_name in args:
                kwargs[param_name] = args[param_name]
            elif param.default is inspect.Parameter.empty:
                return ToolResult.text(f"tool {self.name}: missing required argument '{param_name}'", is_error=True)
        try:
            result = self.func(**kwargs)
            if self._is_async:
                result = _run_async(result)
        except Exception as exc:
            return ToolResult.text(f"tool {self.name} failed: {type(exc).__name__}: {exc}", is_error=True)
        return _coerce_result(result)


def tool(
    func: Any = None,
    *,
    name: str | None = None,
    description: str | None = None,
    read_only: bool = False,
    mutating: bool = False,
) -> Any:
    """Decorator (with or without arguments) that builds a :class:`FunctionTool`."""

    def wrap(target: Any) -> FunctionTool:
        return FunctionTool(target, name=name, description=description, read_only=read_only, mutating=mutating)

    return wrap(func) if func is not None else wrap


def as_tool(item: Any, *, read_only: bool = False, mutating: bool = False) -> AgentTool:
    """Pass an AgentTool through; wrap a plain function/callable as a tool."""
    if isinstance(item, AgentTool):
        return item
    if callable(item):
        return FunctionTool(item, read_only=read_only, mutating=mutating)
    raise TypeError(f"expected a function or AgentTool, got {type(item).__name__}")
