"""MCP servers as tynex tools: bridging, permissions and parallel startup.

:class:`McpManager` connects the configured servers in parallel (failures
become warnings, not session errors) and exposes each server tool as
``mcp__<server>__<tool>`` behind permission kind ``mcp`` — denied unless
allowed. Results map onto the agent's content types.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from tynex.agent.types import AgentTool, ToolContext, ToolResult
from tynex.ai.types import ImageContent, TextContent
from tynex.coding.mcp import McpConnection
from tynex.coding.mcp_settings import McpServerConfig


def _tool_name(server: str, tool: str) -> str:
    """Sanitize ``mcp__<server>__<tool>`` into a portable, truncated tool name."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", f"mcp__{server}__{tool}")[:64]


class McpTool(AgentTool):
    """A server tool exposed to the model as ``mcp__<server>__<tool>``."""

    permission_kind = "mcp"

    def __init__(self, connection: McpConnection, definition: dict[str, Any], call_timeout: float) -> None:
        self.connection = connection
        self.server_name = connection.config.name
        self.remote_name = str(definition.get("name"))
        self.name = _tool_name(self.server_name, self.remote_name)
        self.label = self.name
        description = str(definition.get("description") or f"{self.remote_name} (MCP server {self.server_name})")
        self.description = f"[MCP:{self.server_name}] {description}"
        schema = definition.get("inputSchema")
        self.parameters = schema if isinstance(schema, dict) else {"type": "object", "properties": {}}
        self.call_timeout = call_timeout

    def permission_subject(self, args: dict[str, Any]) -> str:
        return f"{self.server_name}/{self.remote_name}"

    def execute(
        self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None, ctx: ToolContext | None = None
    ) -> ToolResult:
        return _map_result(self.connection.call_tool(self.remote_name, args, self.call_timeout))


def _map_result(result: dict[str, Any]) -> ToolResult:
    """Map MCP tool content onto the agent's content types."""
    parts: list[Any] = []
    for item in result.get("content") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(TextContent(text=str(item.get("text", ""))))
        elif kind == "image":
            parts.append(
                ImageContent(data=str(item.get("data", "")), mime_type=str(item.get("mimeType") or "image/png"))
            )
        elif kind == "resource":
            resource = item.get("resource") or {}
            text = resource.get("text")
            parts.append(TextContent(text=text if isinstance(text, str) else f"[resource {resource.get('uri', '')}]"))
        elif kind == "audio":
            parts.append(TextContent(text=f"[audio content: {item.get('mimeType', 'audio')}]"))
    structured = result.get("structuredContent")
    if not parts:
        text = json.dumps(structured, ensure_ascii=False)[:2000] if structured else "(no output)"
        parts.append(TextContent(text=text))
    details = {"structuredContent": structured} if structured else None
    return ToolResult(content=parts, details=details, is_error=bool(result.get("isError")))


class McpManager:
    """Connects configured servers in parallel and exposes their tools."""

    def __init__(self, servers: list[McpServerConfig], *, connect_timeout: float = 8.0, call_timeout: float = 120.0):
        self.servers = servers
        self.connect_timeout = connect_timeout
        self.call_timeout = call_timeout
        self.connections: dict[str, McpConnection] = {}
        self.warnings: list[str] = []
        self.tools: list[McpTool] = []

    def _register(self, connection: McpConnection, definition: dict[str, Any]) -> None:
        """Add one tool; remote names that sanitize alike get a numeric suffix."""
        tool = McpTool(connection, definition, self.call_timeout)
        base, suffix = tool.name, 2
        while any(tool.name == known.name for known in self.tools):
            tool.name = f"{base}_{suffix}"[:64]
            suffix += 1
        self.tools.append(tool)

    def connect_all(self) -> list[McpTool]:
        """Connect every server in parallel; failures and timeouts become warnings."""
        connections = {config.name: McpConnection(config, self.connect_timeout) for config in self.servers}
        outcomes: dict[str, Any] = {}
        lock = threading.Lock()

        def worker(config: McpServerConfig) -> None:
            connection = connections[config.name]
            try:
                outcome: Any = (connection, connection.connect())
            except Exception as exc:
                connection.close()
                outcome = exc
            with lock:
                outcomes[config.name] = outcome

        threads = [threading.Thread(target=worker, args=(config,), daemon=True) for config in self.servers]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + self.connect_timeout + 1.0
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        for config in self.servers:
            outcome = outcomes.get(config.name)
            if outcome is None:
                self.warnings.append(f"MCP server '{config.name}' timed out while connecting")
                connections[config.name].close()  # a slow worker must not leak it
            elif isinstance(outcome, Exception):
                self.warnings.append(f"MCP server '{config.name}' unavailable: {outcome}")
            else:
                connection, tool_defs = outcome
                self.connections[config.name] = connection
                for definition in tool_defs:
                    self._register(connection, definition)
        return self.tools

    def summaries(self) -> list[dict[str, Any]]:
        """Per-server connection state for ``/mcp`` and the system prompt."""
        rows: list[dict[str, Any]] = []
        for config in self.servers:
            connection = self.connections.get(config.name)
            row: dict[str, Any] = {
                "name": config.name,
                "transport": config.transport,
                "connected": connection is not None,
            }
            if connection is not None:
                row["server"] = connection.label()
                row["tools"] = [tool.remote_name for tool in self.tools if tool.server_name == config.name]
            rows.append(row)
        return rows

    def close(self) -> None:
        for connection in self.connections.values():
            connection.close()
