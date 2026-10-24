"""MCP client: config parsing, stdio + HTTP transports, tool bridging (offline).

The stdio server is a small script launched with the test interpreter; the HTTP
server is a real ``http.server`` bound to 127.0.0.1. No external network.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.conftest import faux_scenario, text_scenario
from tynex.agent.permissions import PermissionRule
from tynex.ai.providers import faux
from tynex.ai.types import ImageContent, TextContent
from tynex.coding.agent_session import AgentSession
from tynex.coding.mcp import McpConnection, McpError
from tynex.coding.mcp_settings import (
    McpServerConfig,
    builtin_catalog,
    builtin_status,
    load_mcp_servers,
    resolve_mcp_servers,
)
from tynex.coding.mcp_tools import McpManager, McpTool, _map_result, _tool_name
from tynex.coding.settings import load_settings, project_dir

FAKE_SERVER = """
import json
import sys

TOOLS = [
    {"name": "echo", "description": "Echo text back", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "boom", "description": "Always fails", "inputSchema": {"type": "object", "properties": {}}},
]


def send(message):
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        continue
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": request_id, "result": {"protocolVersion": "2025-06-18", "serverInfo": {"name": "fake", "version": "1.2.3"}}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        if (message.get("params") or {}).get("cursor"):
            send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS[1:]}})
        else:
            send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS[:1], "nextCursor": "page-2"}})
    elif method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name == "echo":
            send({"jsonrpc": "2.0", "id": request_id, "result": {"content": [{"type": "text", "text": "echo:" + str(arguments.get("text", ""))}]}})
        elif name == "boom":
            send({"jsonrpc": "2.0", "id": request_id, "result": {"content": [{"type": "text", "text": "kaboom"}], "isError": True}})
        else:
            send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": "unknown tool"}})
    else:
        send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "method not found"}})
"""


def write_fake_server(tmp_path: Path) -> Path:
    script = tmp_path / "fake_mcp_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    return script


def stdio_config(tmp_path: Path, name: str = "fake") -> McpServerConfig:
    script = write_fake_server(tmp_path)
    return McpServerConfig(name=name, command=sys.executable, args=[str(script)])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_load_mcp_servers_parses_settings():
    servers = load_mcp_servers(
        {
            "mcpServers": {
                "local": {"command": "uvx", "args": ["srv", "--flag"], "env": {"TOKEN": "x"}, "cwd": "/tmp"},
                "remote": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer x"}},
                "off": {"command": "nope", "enabled": False},
                "junk": "not a table",
            }
        }
    )
    assert [server.name for server in servers] == ["local", "remote"]
    local, remote = servers
    assert local.transport == "stdio" and local.args == ["srv", "--flag"]
    assert local.env == {"TOKEN": "x"} and local.cwd == "/tmp"
    assert remote.transport == "http" and remote.headers["Authorization"] == "Bearer x"
    assert load_mcp_servers({}) == []


def test_tool_names_are_prefixed_and_sanitized():
    assert _tool_name("github", "create-issue") == "mcp__github__create-issue"
    assert _tool_name("my server", "do/thing") == "mcp__my_server__do_thing"


def test_builtin_catalog_is_shipped_data():
    catalog = builtin_catalog()
    assert "context7" in catalog
    for name, entry in catalog.items():
        assert entry.get("command") or entry.get("url"), name
        assert entry["description"], name
        assert isinstance(entry["enabled"], bool), name
    assert catalog["context7"]["url"] == "https://mcp.context7.com/mcp"
    assert catalog["context7"]["enabled"] is True


def test_builtin_status_reports_http_and_enabled_state(monkeypatch):
    monkeypatch.setattr("tynex.coding.mcp_settings.shutil.which", lambda name: None)
    statuses = {status["name"]: status for status in builtin_status({"mcpServers": {"context7": {"enabled": True}}})}
    assert statuses["context7"]["available"] is True
    assert statuses["context7"]["runtime"] == "http"
    assert statuses["context7"]["enabled"] is True


def test_builtin_status_reports_missing_command_runtime(monkeypatch):
    monkeypatch.setattr(
        "tynex.coding.mcp_settings.builtin_catalog", lambda: {"fake": {"command": "npx", "description": "fake"}}
    )
    monkeypatch.setattr("tynex.coding.mcp_settings.shutil.which", lambda name: None)
    statuses = {status["name"]: status for status in builtin_status({})}
    assert statuses["fake"]["available"] is False
    assert statuses["fake"]["runtime"] == "npx"


def test_load_mcp_servers_expands_workspace_placeholders(tmp_path):
    settings = {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "pkg", "{cwd}", "${workspaceFolder}"]}}}
    servers = load_mcp_servers(settings, str(tmp_path))
    assert servers[0].args == ["-y", "pkg", str(tmp_path), str(tmp_path)]


def _enable_builtin(tmp_path: Path, name: str, **fields: object) -> Path:
    """Write a project settings.json enabling ``name``, optionally overriding fields."""
    directory = tmp_path / ".tynex"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "settings.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    servers = data.setdefault("mcpServers", {})
    servers[name] = {"enabled": True, **fields}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_settings_layers_enable_and_customize_builtins(tmp_path, monkeypatch):
    monkeypatch.setenv("TYNEX_CONFIG_DIR", str(tmp_path / "config"))
    _enable_builtin(tmp_path, "context7", headers={"X-Test": "1"})
    entry = load_settings(str(tmp_path))["mcpServers"]["context7"]
    assert entry["enabled"] is True
    assert entry["url"] == "https://mcp.context7.com/mcp"  # inherited from the shipped defaults
    assert entry["headers"] == {"X-Test": "1"}  # overridden by the project layer


def test_resolve_mcp_servers_ignores_disabled_builtins(tmp_path, monkeypatch):
    monkeypatch.setenv("TYNEX_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr("tynex.coding.mcp_settings.shutil.which", lambda name: f"/fake/{name}")
    servers, warnings = resolve_mcp_servers(load_settings(str(tmp_path)), str(tmp_path))
    assert servers == [] and warnings == []


def test_resolve_mcp_servers_enables_builtin_from_shipped_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TYNEX_CONFIG_DIR", str(tmp_path / "config"))
    _enable_builtin(tmp_path, "context7")
    servers, warnings = resolve_mcp_servers(load_settings(str(tmp_path)), str(tmp_path))
    assert [(server.name, server.url) for server in servers] == [("context7", "https://mcp.context7.com/mcp")]
    assert warnings == []


def test_resolve_mcp_servers_skips_missing_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr("tynex.coding.mcp_settings.builtin_catalog", lambda: {"fetch": {"command": "uvx"}})
    monkeypatch.setattr("tynex.coding.mcp_settings.shutil.which", lambda name: None)
    servers, warnings = resolve_mcp_servers(
        {"mcpServers": {"fetch": {"enabled": True, "command": "uvx"}}}, str(tmp_path)
    )
    assert servers == []
    assert len(warnings) == 1 and "uvx" in warnings[0] and "fetch" in warnings[0]


def test_resolve_mcp_servers_keeps_custom_servers(monkeypatch, tmp_path):
    monkeypatch.setattr("tynex.coding.mcp_settings.shutil.which", lambda name: None)
    servers, warnings = resolve_mcp_servers({"mcpServers": {"mine": {"command": "whatever"}}}, str(tmp_path))
    assert [server.name for server in servers] == ["mine"]
    assert warnings == []


def test_map_result_content_types():
    result = _map_result(
        {
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "data": "aGk=", "mimeType": "image/png"},
                {"type": "resource", "resource": {"uri": "file:///x", "text": "resource text"}},
                {"type": "resource", "resource": {"uri": "file:///y"}},
                {"type": "audio", "mimeType": "audio/wav"},
            ]
        }
    )
    assert isinstance(result.content[0], TextContent) and result.content[0].text == "hello"
    assert isinstance(result.content[1], ImageContent) and result.content[1].data == "aGk="
    assert result.content[2].text == "resource text"
    assert "[resource file:///y]" in result.content[3].text
    assert "audio" in result.content[4].text
    assert result.is_error is False

    structured = _map_result({"structuredContent": {"ok": True}})
    assert json.loads(structured.content[0].text) == {"ok": True}
    assert structured.details["structuredContent"] == {"ok": True}
    assert _map_result({}).content[0].text == "(no output)"
    assert _map_result({"content": [], "isError": True}).is_error is True


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------


def test_stdio_connection_lists_paginated_tools_and_calls_them(tmp_path):
    connection = McpConnection(stdio_config(tmp_path), timeout=20.0)
    try:
        definitions = connection.connect()
        assert [definition["name"] for definition in definitions] == ["echo", "boom"]  # cursor pagination
        assert connection.label() == "fake 1.2.3"

        tools = {definition["name"]: definition for definition in definitions}
        echo = McpTool(connection, tools["echo"], 20.0)
        assert echo.name == "mcp__fake__echo"
        assert echo.permission_kind == "mcp"
        assert echo.permission_subject({}) == "fake/echo"
        assert echo.description.startswith("[MCP:fake]")
        assert echo.parameters["required"] == ["text"]
        result = echo.execute("call_1", {"text": "hi"})
        assert result.content[0].text == "echo:hi"

        boom = McpTool(connection, tools["boom"], 20.0)
        failure = boom.execute("call_2", {})
        assert failure.is_error is True and failure.content[0].text == "kaboom"

        with pytest.raises(McpError, match="unknown tool"):
            connection.call_tool("missing", {}, timeout=20.0)
    finally:
        connection.close()


def test_stdio_unknown_command_and_timeout(tmp_path):
    broken = McpServerConfig(name="broken", command="definitely-not-a-real-command-xyz")
    with pytest.raises(McpError, match="failed to start"):
        McpConnection(broken, timeout=5.0).connect()

    sleeping = McpServerConfig(name="sleepy", command=sys.executable, args=["-c", "import time; time.sleep(60)"])
    connection = McpConnection(sleeping, timeout=0.5)
    try:
        with pytest.raises(McpError, match="did not answer"):
            connection.connect()
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# streamable HTTP transport
# ---------------------------------------------------------------------------


class _HttpMcpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append((payload.get("method"), dict(self.headers)))  # type: ignore[attr-defined]
        method, request_id = payload.get("method"), payload.get("id")
        if method == "notifications/initialized":
            self._respond(202, b"", "application/json")
        elif method == "initialize":
            body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"protocolVersion": "2025-06-18", "serverInfo": {"name": "fake-http"}},
                }
            ).encode()
            self._respond(200, body, "application/json", session="sess-1")
        elif method == "tools/list":
            event = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {"name": "ping", "description": "Ping", "inputSchema": {"type": "object", "properties": {}}}
                        ]
                    },
                }
            )
            self._respond(200, f"event: message\ndata: {event}\n\n".encode(), "text/event-stream")
        elif method == "tools/call":
            params = payload.get("params") or {}
            arguments = params.get("arguments") or {}
            body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "pong:" + str(arguments.get("value", ""))}]},
                }
            ).encode()
            self._respond(200, body, "application/json")
        else:
            self._respond(200, b'{"jsonrpc": "2.0", "result": {}}', "application/json")

    def _respond(self, status: int, body: bytes, content_type: str, session: str | None = None) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        if session:
            self.send_header("mcp-session-id", session)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, *_args) -> None:  # keep test output clean
        pass


@pytest.fixture
def http_mcp_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HttpMcpHandler)
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_http_transport_initializes_and_calls_tools(http_mcp_server):
    config = McpServerConfig(name="remote", url=f"http://127.0.0.1:{http_mcp_server.server_port}/mcp")
    connection = McpConnection(config, timeout=10.0)
    try:
        definitions = connection.connect()  # tools/list arrives as an SSE stream
        assert [definition["name"] for definition in definitions] == ["ping"]
        tool = McpTool(connection, definitions[0], 10.0)
        assert tool.name == "mcp__remote__ping"
        assert tool.execute("call_1", {"value": "42"}).content[0].text == "pong:42"
    finally:
        connection.close()

    methods = [method for method, _headers in http_mcp_server.requests]  # type: ignore[attr-defined]
    assert methods == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
    headers_by_method = dict(http_mcp_server.requests)  # type: ignore[attr-defined]
    assert headers_by_method["tools/call"]["mcp-session-id"] == "sess-1"
    assert headers_by_method["tools/call"]["mcp-protocol-version"] == "2025-06-18"


def test_http_server_error_is_reported(http_mcp_server, monkeypatch):
    def boom(*_args, **_kwargs):
        raise McpError("connection refused")

    connection = McpConnection(McpServerConfig(name="remote", url="http://127.0.0.1:1/mcp"), timeout=1.0)
    try:
        monkeypatch.setattr(connection, "request", boom)
        with pytest.raises(McpError, match="connection refused"):
            connection.connect()
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Manager + session integration
# ---------------------------------------------------------------------------


def test_manager_connects_in_parallel_and_reports_failures(tmp_path):
    servers = [stdio_config(tmp_path, "good"), McpServerConfig(name="bad", command="definitely-not-a-real-command-xyz")]
    manager = McpManager(servers, connect_timeout=20.0)
    try:
        tools = manager.connect_all()
        assert [tool.name for tool in tools] == ["mcp__good__echo", "mcp__good__boom"]
        summaries = {entry["name"]: entry for entry in manager.summaries()}
        assert summaries["good"]["connected"] is True
        assert summaries["good"]["server"] == "fake 1.2.3"
        assert summaries["good"]["tools"] == ["echo", "boom"]
        assert summaries["bad"]["connected"] is False
        assert any("bad" in warning for warning in manager.warnings)
    finally:
        manager.close()


def test_session_registers_mcp_tools_and_gates_them(tmp_path, faux_model):
    project = project_dir(tmp_path)
    project.mkdir(parents=True)
    script = write_fake_server(tmp_path)
    (project / "settings.json").write_text(
        json.dumps({"mcpServers": {"fake": {"command": sys.executable, "args": [str(script)]}}}), encoding="utf-8"
    )
    faux.reset_script(
        [
            faux_scenario(
                [
                    {
                        "type": "toolcall",
                        "id": "call_1",
                        "name": "mcp__fake__echo",
                        "arguments": {"text": "from model"},
                    },
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("done"),
        ]
    )
    session = AgentSession(
        cwd=str(tmp_path),
        model=faux_model,
        tool_names=["read"],
        no_session=True,
        permission_rules=[PermissionRule(tool="mcp__fake__echo", action="allow")],
    )
    try:
        assert "mcp__fake__echo" in [tool.name for tool in session.tools]
        events: list[dict] = []
        session.prompt("go", emit=events.append)
        decisions = [event for event in events if event["type"] == "permission_decision"]
        assert decisions and decisions[0]["action"] == "allow"
        endings = [event for event in events if event["type"] == "tool_execution_end"]
        assert endings and endings[0]["isError"] is False
        assert endings[0]["result"].content[0].text == "echo:from model"
    finally:
        session.mcp_manager.close()


def test_session_without_allow_rule_denies_mcp_tool(tmp_path, faux_model):
    project = project_dir(tmp_path)
    project.mkdir(parents=True)
    script = write_fake_server(tmp_path)
    (project / "settings.json").write_text(
        json.dumps({"mcpServers": {"fake": {"command": sys.executable, "args": [str(script)]}}}), encoding="utf-8"
    )
    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "toolcall", "id": "call_1", "name": "mcp__fake__echo", "arguments": {"text": "no"}},
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("done"),
        ]
    )
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
    try:
        events: list[dict] = []
        session.prompt("go", emit=events.append)
        endings = [event for event in events if event["type"] == "tool_execution_end"]
        assert endings and endings[0]["isError"] is True
        assert "approval required" in endings[0]["result"].content[0].text
    finally:
        session.mcp_manager.close()


def test_session_can_disable_mcp(tmp_path, faux_model):
    project = project_dir(tmp_path)
    project.mkdir(parents=True)
    script = write_fake_server(tmp_path)
    (project / "settings.json").write_text(
        json.dumps({"mcpServers": {"fake": {"command": sys.executable, "args": [str(script)]}}}), encoding="utf-8"
    )
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True, mcp=False)
    assert session.mcp_manager is None
    assert not [tool for tool in session.tools if tool.name.startswith("mcp__")]
    assert session.mcp_warnings == []


def test_mcp_command_renders(tmp_path, faux_model):
    from io import StringIO

    from rich.console import Console

    from tynex.coding import session_commands

    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True, mcp=False)
    buffer = StringIO()
    console = Console(file=buffer, width=100)
    session_commands.mcp_servers(session, console)
    # Silence here used to make "no servers" indistinguishable from a broken command.
    assert "No MCP servers connected" in buffer.getvalue()


def test_session_enables_configured_server_and_describes_it_in_prompt(tmp_path, faux_model):
    project = project_dir(tmp_path)
    project.mkdir(parents=True)
    script = write_fake_server(tmp_path)
    _enable_builtin(tmp_path, "fake", command=sys.executable, args=[str(script)])
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=[], no_session=True)
    try:
        assert "mcp__fake__echo" in [tool.name for tool in session.tools]
        prompt = session.system_prompt_text()
        assert "<mcp>" in prompt
        assert "fake (2 tools, stdio)" in prompt
        assert "mcp__<server>__<tool>" in prompt
    finally:
        session.mcp_manager.close()
