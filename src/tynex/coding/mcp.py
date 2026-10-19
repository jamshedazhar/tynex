"""Minimal MCP (Model Context Protocol) client: stdio and streamable HTTP.

Tynex speaks the client side of the 2025-06-18 spec directly with no extra
dependency; :mod:`tynex.coding.mcp_tools` bridges connected servers into the
agent's tool list. Servers come from ``settings.json`` via
:mod:`tynex.coding.mcp_settings`; stdio servers share the shell tools' process
management (:mod:`tynex.coding.process`).
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import shutil
import subprocess
import threading
from typing import Any

from tynex import __version__
from tynex.coding.mcp_settings import McpServerConfig
from tynex.coding.process import detached_options, kill_process_tree
from tynex.http import HTTPError, iter_sse, new_client

PROTOCOL_VERSION = "2025-06-18"
CLIENT_NAME = "tynex"
LIST_PAGES_LIMIT = 20
NOT_SUPPORTED = {"code": -32601, "message": "method not supported by this client"}


class McpError(RuntimeError):
    """A transport, protocol or server error while talking to an MCP server."""


class McpConnection:
    """One MCP server over stdio (``command``) or streamable HTTP (``url``).

    Send a JSON-RPC message and wait for the matching response; stdio servers
    are launched on first use and drained by one reader thread.
    """

    def __init__(self, config: McpServerConfig, timeout: float) -> None:
        self.config = config
        self.timeout = timeout
        self.server_info: dict[str, Any] = {}
        self._ids = itertools.count(1)
        self._write_lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._pending: dict[int, tuple[threading.Event, dict[str, Any]]] = {}
        self._client = new_client(timeout, connect=min(timeout, 10.0)) if config.url else None
        self._session_id: str | None = None

    def request(self, method: str, params: dict[str, Any] | None, timeout: float | None = None) -> dict[str, Any]:
        """Send one request; errors become ``McpError`` and the result is returned."""
        request_id = next(self._ids)
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        exchange = self._http_exchange if self.config.url else self._stdio_exchange
        message = exchange(payload, request_id, timeout)
        if message is None:
            raise McpError(f"MCP server '{self.config.name}' did not answer '{method}'")
        error = message.get("error")
        if error:
            raise McpError(f"MCP server '{self.config.name}': {error.get('message') or error}")
        return message.get("result") or {}

    def notify(self, method: str, params: dict[str, Any] | None) -> None:
        """Fire-and-forget notification: a server that is already gone is not an error."""
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if self.config.url:
            with contextlib.suppress(McpError):
                self._post(payload, None)
        else:
            if self._proc is None:
                self._start_stdio()
            self._write_stdio(payload)

    def connect(self) -> list[dict[str, Any]]:
        """Initialize the session, then fetch the tool list, following cursors."""
        client = {"name": CLIENT_NAME, "version": __version__}
        params = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": client}
        self.server_info = self.request("initialize", params).get("serverInfo") or {}
        self.notify("notifications/initialized", {})
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(LIST_PAGES_LIMIT):
            result = self.request("tools/list", {"cursor": cursor} if cursor else {})
            entries = result.get("tools")
            if isinstance(entries, list):
                tools.extend(entry for entry in entries if isinstance(entry, dict) and entry.get("name"))
            if not (cursor := result.get("nextCursor")):
                break
        return tools

    def call_tool(self, name: str, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": args}, timeout)

    def label(self) -> str:
        """Human-readable server title for ``/mcp``."""
        title = self.server_info.get("title") or self.server_info.get("name") or self.config.name
        version = self.server_info.get("version")
        return f"{title} {version}" if version else str(title)

    def close(self) -> None:
        """Stop the server; closing never raises."""
        with contextlib.suppress(Exception):
            if self._client is not None:
                self._client.close()
            proc = self._proc
            if proc is not None:
                kill_process_tree(proc)
                proc.wait(timeout=5)

    # ---- stdio wire ----------------------------------------------------

    def _start_stdio(self) -> None:
        command = self.config.command or ""
        resolved = shutil.which(command)
        if os.name == "nt" and resolved and resolved.lower().endswith((".cmd", ".bat")):
            argv = ["cmd", "/c", resolved, *self.config.args]  # npx/uvx shims need cmd.exe
        else:
            argv = [resolved or command, *self.config.args]
        options = {"text": True, "encoding": "utf-8", "errors": "replace", "bufsize": 1, "cwd": self.config.cwd}
        options.update(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        options.update(env={**os.environ, **self.config.env}, **detached_options())
        try:
            self._proc = subprocess.Popen(argv, **options)
        except OSError as exc:
            raise McpError(f"failed to start MCP server '{self.config.name}': {exc}") from exc
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def _write_stdio(self, message: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise McpError(f"MCP server '{self.config.name}' is not running")
        try:
            with self._write_lock:
                proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                proc.stdin.flush()
        except OSError as exc:
            raise McpError(f"MCP server '{self.config.name}' write failed: {exc}") from exc

    def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for raw in self._proc.stdout:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            if "method" not in message:  # a response to one of our requests
                pending = self._pending.get(message.get("id"))
                if pending is not None:
                    event, holder = pending
                    holder["message"] = message
                    event.set()
            elif "id" in message:  # a server-initiated request (sampling/roots): decline
                self._write_stdio({"jsonrpc": "2.0", "id": message["id"], "error": NOT_SUPPORTED})

    def _stdio_exchange(self, payload: dict[str, Any], request_id: int, timeout: float | None) -> dict[str, Any] | None:
        """Write one message and wait for its response; ``None`` on timeout."""
        if self._proc is None:
            self._start_stdio()
        event = threading.Event()
        holder: dict[str, Any] = {}
        self._pending[request_id] = (event, holder)
        try:
            self._write_stdio(payload)
            if not event.wait(timeout or self.timeout):
                return None
        finally:
            self._pending.pop(request_id, None)
        return holder.get("message")

    # ---- HTTP wire -----------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": PROTOCOL_VERSION,
            **self.config.headers,
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    def _post(self, payload: dict[str, Any], timeout: float | None) -> Any:
        assert self._client is not None
        try:
            response = self._client.post(
                self.config.url or "", json=payload, headers=self._headers(), timeout=timeout or self.timeout
            )
        except HTTPError as exc:
            raise McpError(f"MCP server '{self.config.name}' HTTP error: {exc}") from exc
        if session_id := response.headers.get("mcp-session-id"):
            self._session_id = session_id
        return response

    def _http_exchange(self, payload: dict[str, Any], request_id: int, timeout: float | None) -> dict[str, Any] | None:
        """Post one message and decode the JSON or SSE answer; ``None`` when absent."""
        response = self._post(payload, timeout)
        if response.status_code >= 400:
            body = response.text[:300]
            raise McpError(f"MCP server '{self.config.name}' returned HTTP {response.status_code}: {body}")
        if response.status_code == 202 or not response.content:
            return {}
        if "text/event-stream" not in response.headers.get("content-type", ""):
            try:
                decoded = response.json()
            except ValueError as exc:
                raise McpError(f"MCP server '{self.config.name}' sent invalid JSON: {exc}") from exc
            return decoded if isinstance(decoded, dict) else None
        for data in iter_sse(response.iter_lines()):
            try:
                candidate = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get("id") == request_id:
                return candidate
        return None
