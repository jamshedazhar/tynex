"""Tests for the PowerShell flavor of ShellTool."""

import os
import shutil

import pytest

from tynex.coding.tools.shell import ShellTool, _ShellFlavor

_REAL_WHICH = shutil.which
POWERSHELL_PATH = _REAL_WHICH("pwsh") or _REAL_WHICH("powershell")

windows_only = pytest.mark.skipif(os.name != "nt", reason="the powershell flavor is intended for Windows")
requires_powershell = pytest.mark.skipif(POWERSHELL_PATH is None, reason="PowerShell is not installed")


def _stub_powershell(monkeypatch, path: str = "C:/powershell.exe") -> _ShellFlavor:
    """Resolve the PowerShell flavor with bash/sh hidden and ``path`` as the stub executable."""
    monkeypatch.setattr("tynex.coding.tools.shell.os.name", "nt")
    monkeypatch.setattr("tynex.coding.tools.shell.shutil.which", lambda name: path if name == "powershell" else None)
    return _ShellFlavor.get_shell()


def test_prefers_pwsh_over_powershell(monkeypatch):
    monkeypatch.setattr("tynex.coding.tools.shell.os.name", "nt")
    monkeypatch.setattr(
        "tynex.coding.tools.shell.shutil.which",
        lambda name: f"C:/{name}.exe" if name in ("pwsh", "powershell") else None,
    )
    flavour = _ShellFlavor.get_shell()
    assert flavour.name == "pwsh"
    assert flavour.path == "C:/pwsh.exe"


def test_wraps_commands_with_the_utf8_prefix(monkeypatch):
    prefix = "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\n"
    flavour = _stub_powershell(monkeypatch)
    assert flavour.wrap("Get-Date") == prefix + "Get-Date"


def test_join_quotes_argv_for_one_command_line(monkeypatch):
    flavour = _stub_powershell(monkeypatch)
    assert flavour.join(["ruff", "check", "a b"]) == "& 'ruff' 'check' 'a b'; exit $LASTEXITCODE"
    assert flavour.join(["echo", "it's"]) == "& 'echo' 'it''s'; exit $LASTEXITCODE"


@windows_only
@requires_powershell
def test_executes_a_command(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tynex.coding.tools.shell.shutil.which",
        lambda name: _REAL_WHICH(name) if name in ("pwsh", "powershell") else None,
    )
    tool = ShellTool(str(tmp_path))
    tool.flavor = _ShellFlavor.get_shell()
    result = tool.execute("call", {"command": "Write-Output 'hello-powershell'"})
    text = result.content[0].text
    assert not result.is_error
    assert "hello-powershell" in text


@windows_only
@requires_powershell
def test_execute_reports_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tynex.coding.tools.shell.shutil.which",
        lambda name: _REAL_WHICH(name) if name in ("pwsh", "powershell") else None,
    )
    tool = ShellTool(str(tmp_path))
    tool.flavor = _ShellFlavor.get_shell()
    result = tool.execute("call", {"command": "exit 3"})
    assert result.is_error
    assert "Command exited with code 3" in result.content[0].text
