"""Built-in MCP servers and the ``tynex mcp`` command group (offline)."""

import json

from tynex import cli as cli_mod


def _run(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(cli_mod, "_ensure_utf8_streams", lambda: None)
    return cli_mod.main(list(argv))


def test_mcp_add_enable_list_remove_builtin(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYNEX_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: f"/fake/{name}")

    assert _run(monkeypatch, "mcp", "add", "context7", "--project") == 0
    settings_file = tmp_path / ".tynex" / "settings.json"
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert data["mcpServers"] == {"context7": {"enabled": True}}
    assert "Enabled built-in MCP server 'context7' (http)" in capsys.readouterr().out

    assert _run(monkeypatch, "mcp", "add", "context7", "--project") == 0
    assert "already enabled" in capsys.readouterr().out

    assert _run(monkeypatch, "mcp", "list") == 0
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if "context7" in line)
    assert "http" in line and "(enabled: project settings)" in line
    assert "install" not in line
    assert "\nConfigured servers (mcpServers):\n  (none)" in out  # built-ins are not repeated

    assert _run(monkeypatch, "mcp", "remove", "context7", "--project") == 0
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "mcpServers" not in data
    assert "Removed MCP server 'context7'" in capsys.readouterr().out


def test_mcp_add_custom_stdio_and_http_servers(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYNEX_CONFIG_DIR", str(tmp_path / "config"))

    argv = ["mcp", "add", "fetchlike", "--command", "uvx", "--arg", "mcp-server-fetch", "--env", "TOKEN=x", "--project"]
    assert _run(monkeypatch, *argv) == 0
    argv = [
        "mcp",
        "add",
        "wiki",
        "--url",
        "https://mcp.deepwiki.com/mcp",
        "--header",
        "Authorization=Bearer z",
        "--project",
    ]
    assert _run(monkeypatch, *argv) == 0
    data = json.loads((tmp_path / ".tynex" / "settings.json").read_text(encoding="utf-8"))
    assert data["mcpServers"]["fetchlike"] == {"command": "uvx", "args": ["mcp-server-fetch"], "env": {"TOKEN": "x"}}
    assert data["mcpServers"]["wiki"] == {
        "url": "https://mcp.deepwiki.com/mcp",
        "headers": {"Authorization": "Bearer z"},
    }
    assert "enabled" not in data["mcpServers"]["fetchlike"]  # custom servers are on by default
    assert "enabled" not in data["mcpServers"]["wiki"]

    assert _run(monkeypatch, "mcp", "list") == 0
    out = capsys.readouterr().out
    assert "fetchlike" in out and "uvx mcp-server-fetch" in out
    assert "wiki" in out and "http https://mcp.deepwiki.com/mcp" in out

    assert _run(monkeypatch, "mcp", "remove", "fetchlike", "--project") == 0
    assert _run(monkeypatch, "mcp", "remove", "wiki", "--project") == 0
    data = json.loads((tmp_path / ".tynex" / "settings.json").read_text(encoding="utf-8"))
    assert "mcpServers" not in data


def test_mcp_add_unknown_builtin_and_bad_pairs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYNEX_CONFIG_DIR", str(tmp_path / "config"))

    assert _run(monkeypatch, "mcp", "add", "nope", "--project") == 2
    assert "unknown built-in MCP server 'nope'" in capsys.readouterr().err

    assert _run(monkeypatch, "mcp", "add", "x", "--command", "uvx", "--env", "BROKEN", "--project") == 2
    assert "KEY=VALUE" in capsys.readouterr().err


def test_mcp_add_rejects_options_that_need_their_counterpart(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYNEX_CONFIG_DIR", str(tmp_path / "config"))
    assert _run(monkeypatch, "mcp", "add", "x", "--header", "A=B", "--project") == 2
    assert "--header requires --url" in capsys.readouterr().err
    assert _run(monkeypatch, "mcp", "add", "x", "--arg", "a", "--project") == 2
    assert "--arg and --env require --command" in capsys.readouterr().err


def test_mcp_add_http_builtin_works_without_runtime(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYNEX_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(cli_mod.shutil, "which", lambda name: None)

    assert _run(monkeypatch, "mcp", "add", "context7", "--project") == 0
    captured = capsys.readouterr()
    assert "Enabled built-in MCP server 'context7' (http)" in captured.out
    assert "on PATH" not in captured.err
    data = json.loads((tmp_path / ".tynex" / "settings.json").read_text(encoding="utf-8"))
    assert data["mcpServers"] == {"context7": {"enabled": True}}

    assert _run(monkeypatch, "mcp", "list") == 0
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if "context7" in line)
    assert "install" not in line


def test_mcp_add_writes_global_settings_by_default(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYNEX_CONFIG_DIR", str(tmp_path / "config"))

    assert _run(monkeypatch, "mcp", "add", "context7") == 0
    global_file = tmp_path / "config" / "settings.json"
    assert json.loads(global_file.read_text(encoding="utf-8"))["mcpServers"] == {"context7": {"enabled": True}}
    assert not (tmp_path / ".tynex" / "settings.json").exists()
    assert _run(monkeypatch, "mcp", "list") == 0
    assert "(enabled: global settings)" in capsys.readouterr().out


def test_mcp_add_custom_entry_for_builtin_name_enables_it(tmp_path, monkeypatch, capsys):
    """Overriding a built-in definition still counts as enabling it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYNEX_CONFIG_DIR", str(tmp_path / "config"))

    argv = ["mcp", "add", "context7", "--url", "https://example.test/mcp", "--header", "X-Test=1", "--project"]
    assert _run(monkeypatch, *argv) == 0
    assert "Added MCP server 'context7'" in capsys.readouterr().out
    data = json.loads((tmp_path / ".tynex" / "settings.json").read_text(encoding="utf-8"))
    assert data["mcpServers"]["context7"] == {
        "url": "https://example.test/mcp",
        "headers": {"X-Test": "1"},
        "enabled": True,
    }

    assert _run(monkeypatch, "mcp", "list") == 0
    out = capsys.readouterr().out
    assert "(enabled: project settings)" in out
