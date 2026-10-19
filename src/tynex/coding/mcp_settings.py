"""MCP server configuration: ``mcpServers`` entries from settings.json.

Built-ins are ordinary ``mcpServers`` entries shipped in tynex's default
settings with ``enabled: false``. Settings layers (shipped defaults, global
``~/.tynex/agent/settings.json``, project ``.tynex/settings.json``) deep-merge
over them, so enabling a built-in inherits the shipped command and arguments.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any

from tynex.coding.settings import default_settings


@dataclass
class McpServerConfig:
    """One ``mcpServers`` entry: stdio with a command, HTTP with a URL."""

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def transport(self) -> str:
        return "http" if self.url else "stdio"


def _string_map(value: Any) -> dict[str, str]:
    """Coerce a settings value into a string map; anything else becomes empty."""
    return {str(key): str(item) for key, item in value.items()} if isinstance(value, dict) else {}


def _expand_workspace(value: str, base: str | None) -> str:
    """Expand ``${workspaceFolder}`` (or the ``{cwd}`` alias) to the session directory."""
    return value.replace("${workspaceFolder}", base).replace("{cwd}", base) if base else value


def load_mcp_servers(settings: dict[str, Any], cwd: str | None = None) -> list[McpServerConfig]:
    """Enabled ``mcpServers`` entries; malformed and disabled ones are ignored."""
    raw = settings.get("mcpServers")
    if not isinstance(raw, dict):
        return []
    base = os.path.abspath(cwd) if cwd else None
    servers: list[McpServerConfig] = []
    for name, entry in raw.items():
        if not isinstance(entry, dict) or entry.get("enabled") is False:
            continue
        url, command = entry.get("url"), entry.get("command")
        if isinstance(url, str) and url:
            servers.append(McpServerConfig(name=str(name), url=url, headers=_string_map(entry.get("headers"))))
        elif isinstance(command, str) and command:
            config = McpServerConfig(name=str(name), command=command, env=_string_map(entry.get("env")))
            config.args = [_expand_workspace(str(arg), base) for arg in entry.get("args") or []]
            config.cwd = str(entry["cwd"]) if entry.get("cwd") else None
            servers.append(config)
    return servers


def builtin_catalog() -> dict[str, dict[str, Any]]:
    """Built-ins: the ``mcpServers`` entries shipped in tynex's default settings."""
    servers = default_settings().get("mcpServers")
    if not isinstance(servers, dict):
        return {}
    return {str(name): entry for name, entry in servers.items() if isinstance(entry, dict)}


def runtime_path(runtime: str) -> str | None:
    """Resolved path of a server runtime (``npx``, ``uvx``, …), or ``None`` when missing."""
    return shutil.which(runtime)


def builtin_status(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per built-in for ``tynex mcp list``: runtime, availability and enabled state."""
    merged = settings.get("mcpServers")
    merged = merged if isinstance(merged, dict) else {}
    rows: list[dict[str, Any]] = []
    for name, entry in builtin_catalog().items():
        command = str(entry.get("command") or "")
        url = entry.get("url")
        active = merged.get(name)
        rows.append(
            {
                "name": name,
                "description": str(entry.get("description") or ""),
                "runtime": command or ("http" if url else "?"),
                "available": bool(url) or (bool(command) and runtime_path(command) is not None),
                "enabled": isinstance(active, dict) and active.get("enabled") is not False,
            }
        )
    return rows


def resolve_mcp_servers(settings: dict[str, Any], cwd: str | None = None) -> tuple[list[McpServerConfig], list[str]]:
    """Active ``mcpServers`` for a session as ``(servers, warnings)``.

    A built-in whose runtime is missing is skipped with a warning instead of
    failing the session; other servers pass through unchanged.
    """
    catalog = builtin_catalog()
    servers: list[McpServerConfig] = []
    warnings: list[str] = []
    for server in load_mcp_servers(settings, cwd or os.getcwd()):
        if server.command and server.name in catalog and runtime_path(server.command) is None:
            warnings.append(f"built-in '{server.name}' needs '{server.command}' on PATH — install it or disable it")
        else:
            servers.append(server)
    return servers, warnings
