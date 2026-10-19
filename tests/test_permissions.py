"""Permission modes, rules, session grants and the approval gate (all offline)."""

import json
from pathlib import Path

import pytest

from tests.conftest import faux_scenario, text_scenario
from tynex.agent.permissions import (
    DEFAULT_PERMISSION_MODE,
    PermissionGate,
    PermissionRule,
    mode_description,
    normalize_mode,
)
from tynex.agent.types import AgentTool, ToolResult
from tynex.ai.providers import faux
from tynex.coding.agent_session import AgentSession
from tynex.coding.permissions import config_paths, load_permission_config
from tynex.coding.settings import config_dir, project_dir
from tynex.coding.tools.shell import ShellTool, shell_name
from tynex.coding.tools.write import WriteTool


class StubTool(AgentTool):
    """Minimal tool carrying permission metadata."""

    def __init__(self, name: str, kind: str) -> None:
        self.name = name
        self.label = name
        self.permission_kind = kind
        self.mutating = kind in ("edit", "shell")

    def execute(self, tool_call_id, args, signal=None, on_update=None, ctx=None) -> ToolResult:
        return ToolResult.text("ran")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_normalize_mode():
    assert normalize_mode("Accept_Edits") == "accept-edits"
    assert normalize_mode(" bypass ") == "bypass"
    with pytest.raises(ValueError, match="Unknown permission mode"):
        normalize_mode("yolo")
    assert DEFAULT_PERMISSION_MODE == "accept-edits"
    assert "read-only" in mode_description("plan")
    assert mode_description("bogus") == ""


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("plan", {"read": "allow", "web": "allow", "edit": "deny", "shell": "deny", "mcp": "deny"}),
        ("ask", {"read": "allow", "web": "allow", "edit": "ask", "shell": "ask", "mcp": "ask"}),
        ("accept-edits", {"read": "allow", "web": "allow", "edit": "allow", "shell": "ask", "mcp": "ask"}),
        ("bypass", {"read": "allow", "web": "allow", "edit": "allow", "shell": "allow", "mcp": "allow"}),
    ],
)
def test_mode_matrix(mode, expected):
    gate = PermissionGate(mode)
    for kind, action in expected.items():
        assert gate._mode_action(kind) == action, (mode, kind)


def test_ask_without_approver_is_a_denial_with_guidance():
    gate = PermissionGate("accept-edits")
    decision = gate.check(StubTool("custom", "tool"), {})
    assert decision.action == "deny"
    assert "no interactive approval" in decision.reason
    assert "--permission-mode bypass" in decision.reason
    assert decision.origin == "denied"
    assert decision.request.summary() == "custom"
    # read-only kinds still pass
    assert gate.check(StubTool("read", "read"), {}).allowed


def test_rules_last_match_wins():
    rules = [PermissionRule(tool="bash", action="allow"), PermissionRule(tool="bash", match="rm *", action="deny")]
    gate = PermissionGate("ask", rules)
    bash = ShellTool(".")
    assert gate.check(bash, {"command": "ls -la"}).action == "allow"
    assert gate.check(bash, {"command": "rm -rf build"}).action == "deny"
    assert gate.check(bash, {"command": "rmx"}).action == "allow"
    decision = gate.check(bash, {"command": "rm -rf build"})
    assert decision.origin == "rule" and "denied" in decision.reason
    assert decision.request.subject == "rm -rf build"


def test_rule_globs_and_double_star_fallback():
    rules = [PermissionRule(tool="write", match="**/.env", action="deny")]
    gate = PermissionGate("accept-edits", rules)
    write = WriteTool(".")
    assert gate.check(write, {"path": ".env"}).action == "deny"
    assert gate.check(write, {"path": "config/.env"}).action == "deny"
    assert gate.check(write, {"path": "src/app.py"}).action == "allow"


def test_rule_can_target_mcp_tools_by_name():
    rules = [PermissionRule(tool="mcp__github__*", action="deny")]
    gate = PermissionGate("bypass", rules)
    assert gate.check(StubTool("mcp__github__create_issue", "mcp"), {}).action == "deny"
    assert gate.check(StubTool("mcp__gitlab__create_issue", "mcp"), {}).action == "allow"


def test_plan_mode_cannot_be_widened_by_rules():
    rules = [PermissionRule(tool="bash", action="allow"), PermissionRule(tool="write", action="allow")]
    gate = PermissionGate("plan", rules)
    decision = gate.check(ShellTool("."), {"command": "echo hi"})
    assert decision.action == "deny" and decision.origin == "mode"
    assert "read-only" in decision.reason
    assert gate.check(WriteTool("."), {"path": "a.txt"}).action == "deny"
    # Rules can still tighten read-only calls (and read/web stay allowed otherwise).
    strict = PermissionGate("plan", [PermissionRule(tool="read", action="deny")])
    assert strict.check(StubTool("read", "read"), {}).action == "deny"
    assert strict.check(StubTool("grep", "read"), {}).action == "allow"


def test_session_grant_skips_later_asks():
    asked: list[str] = []

    def approve(request):
        asked.append(request.summary())
        return "session"

    gate = PermissionGate("ask", on_ask=approve)
    tool = StubTool("custom", "tool")
    assert gate.check(tool, {}).allowed
    second = gate.check(tool, {})
    assert second.allowed and second.origin == "grant"
    assert len(asked) == 1
    assert gate.session_grants() == ["custom"]
    gate.clear_grants()
    gate.check(tool, {})
    assert len(asked) == 2


def test_allow_once_and_deny_outcomes():
    outcomes = ["allow", "deny"]

    def approve(_request):
        return outcomes.pop(0)

    gate = PermissionGate("ask", on_ask=approve)
    tool = StubTool("custom", "tool")
    first = gate.check(tool, {})
    assert first.allowed and first.origin == "approval"
    second = gate.check(tool, {})
    assert second.action == "deny" and second.origin == "denied"
    assert second.reason == "denied by the user"


def test_approval_failure_or_none_answer_denies():
    def broken(_request):
        raise RuntimeError("ui exploded")

    gate = PermissionGate("ask", on_ask=broken)
    decision = gate.check(StubTool("custom", "tool"), {})
    assert decision.action == "deny" and "ui exploded" in decision.reason

    gate = PermissionGate("ask", on_ask=lambda _request: None)  # e.g. EOF at the prompt
    assert gate.check(StubTool("custom", "tool"), {}).action == "deny"


def test_set_mode_validates_and_returns_applied():
    gate = PermissionGate("accept-edits")
    assert gate.set_mode("Plan") == "plan"
    assert gate.mode == "plan"
    with pytest.raises(ValueError):
        gate.set_mode("nope")


# ---------------------------------------------------------------------------
# permissions.toml loading
# ---------------------------------------------------------------------------


def test_load_permission_config_merges_global_then_project(tmp_path):
    global_file = config_dir() / "permissions.toml"
    global_file.write_text('mode = "ask"\n\n[[rules]]\ntool = "bash"\naction = "deny"\n', encoding="utf-8")
    project_file = project_dir(tmp_path) / "permissions.toml"
    project_file.parent.mkdir(parents=True)
    project_file.write_text(
        '[[rules]]\ntool = "bash"\nmatch = "git *"\naction = "allow"\n\n[[rules]]\naction = "bogus"\n', encoding="utf-8"
    )
    config = load_permission_config(tmp_path)
    assert config.mode == "ask"
    assert [rule.action for rule in config.rules] == ["deny", "allow"]
    assert config.rules[1].source == str(project_file)
    assert any("bogus" in warning for warning in config.warnings)
    assert [Path(entry).name for entry in config.files] == ["permissions.toml", "permissions.toml"]

    gate = PermissionGate(config.mode, config.rules)
    bash = ShellTool(".")
    assert gate.check(bash, {"command": "git status"}).action == "allow"  # project rule wins
    assert gate.check(bash, {"command": "curl http://x"}).action == "deny"


def test_load_permission_config_survives_broken_files(tmp_path):
    rules_file = project_dir(tmp_path) / "permissions.toml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("mode = [not toml", encoding="utf-8")
    config = load_permission_config(tmp_path)
    assert config.mode is None and config.rules == []
    assert len(config.warnings) == 1
    assert "permissions.toml" in config.warnings[0]

    rules_file.write_text('rules = "not a list"\n\n[mode]\nnope = 1\n', encoding="utf-8")
    config = load_permission_config(tmp_path)
    assert config.mode is None
    assert any("must be a list" in warning for warning in config.warnings)


def test_config_paths_are_deduped_and_exist_only(tmp_path):
    (project_dir(tmp_path)).mkdir(parents=True)
    (project_dir(tmp_path) / "permissions.toml").write_text("", encoding="utf-8")
    paths = config_paths(tmp_path)
    assert paths == [project_dir(tmp_path) / "permissions.toml"]


# ---------------------------------------------------------------------------
# Session integration (faux provider)
# ---------------------------------------------------------------------------


def run_session(tmp_path, faux_model, script, tool_names, **kwargs):
    faux.reset_script(script)
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=tool_names, no_session=True, **kwargs)
    events: list[dict] = []
    session.prompt("do it", emit=events.append)
    return session, events


def tool_call(name, arguments, call_id="call_1"):
    return {"type": "toolcall", "id": call_id, "name": name, "arguments": arguments}


def test_shell_is_blocked_by_default_and_the_model_is_told(tmp_path, faux_model):
    script = [
        faux_scenario([tool_call(shell_name, {"command": "echo hi"}), {"type": "stop", "reason": "toolUse"}]),
        text_scenario("ok"),
    ]
    session, events = run_session(tmp_path, faux_model, script, [shell_name])
    decisions = [event for event in events if event["type"] == "permission_decision"]
    assert decisions, events
    assert decisions[0]["action"] == "deny"
    assert decisions[0]["tool"] == shell_name
    endings = [event for event in events if event["type"] == "tool_execution_end"]
    assert endings and endings[0]["isError"] is True
    reason = endings[0]["result"].content[0].text
    assert "approval" in reason and "--permission-mode" in reason
    assert any(entry.type == "permission" for entry in session.session.entries)


def test_allow_rule_lets_the_shell_run(tmp_path, faux_model):
    script = [
        faux_scenario([tool_call(shell_name, {"command": "echo allowed"}), {"type": "stop", "reason": "toolUse"}]),
        text_scenario("done"),
    ]
    _session, events = run_session(
        tmp_path,
        faux_model,
        script,
        [shell_name],
        permission_rules=[PermissionRule(tool=shell_name, match="echo *", action="allow")],
    )
    decisions = [event for event in events if event["type"] == "permission_decision"]
    assert decisions and decisions[0]["action"] == "allow" and decisions[0]["origin"] == "rule"
    endings = [event for event in events if event["type"] == "tool_execution_end"]
    assert endings and endings[0]["isError"] is False


def test_approval_session_grant_applies_to_later_calls(tmp_path, faux_model):
    asked: list[str] = []

    def approve(request):
        asked.append(request.summary())
        return "session"

    script = [
        faux_scenario(
            [
                tool_call(shell_name, {"command": "echo one"}, "call_1"),
                tool_call(shell_name, {"command": "echo two"}, "call_2"),
                {"type": "stop", "reason": "toolUse"},
            ]
        ),
        text_scenario("done"),
    ]
    _, events = run_session(tmp_path, faux_model, script, [shell_name], approval=approve)
    assert len(asked) == 1
    assert asked[0].startswith(f"{shell_name}: echo one")
    decisions = [event for event in events if event["type"] == "permission_decision"]
    assert [d["origin"] for d in decisions] == ["approval", "grant"]
    endings = [event for event in events if event["type"] == "tool_execution_end"]
    assert len(endings) == 2 and all(not ending["isError"] for ending in endings)


def test_plan_mode_denies_writes_and_asks_for_a_plan(tmp_path, faux_model):
    script = [
        faux_scenario(
            [tool_call("write", {"path": "notes.txt", "content": "hi"}), {"type": "stop", "reason": "toolUse"}]
        ),
        text_scenario("here is the plan"),
    ]
    session, events = run_session(tmp_path, faux_model, script, ["write"], permission_mode="plan")
    endings = [event for event in events if event["type"] == "tool_execution_end"]
    assert endings and endings[0]["isError"] is True
    assert "Plan mode" in endings[0]["result"].content[0].text
    assert not (tmp_path / "notes.txt").exists()
    assert "plan mode is active" in session.system_prompt_text().lower()


def test_settings_can_choose_the_mode_and_project_toml_rules_apply(tmp_path, faux_model):
    project = project_dir(tmp_path)
    project.mkdir(parents=True)
    (project / "settings.json").write_text(json.dumps({"permissionMode": "ask"}), encoding="utf-8")
    (project / "permissions.toml").write_text(
        f'[[rules]]\ntool = "{shell_name}"\nmatch = "echo *"\naction = "allow"\n', encoding="utf-8"
    )
    script = [
        faux_scenario(
            [tool_call(shell_name, {"command": "echo from rule file"}), {"type": "stop", "reason": "toolUse"}]
        ),
        text_scenario("ok"),
    ]
    session, events = run_session(tmp_path, faux_model, script, [shell_name])
    assert session.permission_gate.mode == "ask"
    assert session.permission_summary()["rules"][0]["action"] == "allow"
    endings = [event for event in events if event["type"] == "tool_execution_end"]
    assert endings and endings[0]["isError"] is False


def test_set_permission_mode_records_a_session_entry(tmp_path, faux_model):
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
    assert session.permission_gate.mode == "accept-edits"
    assert session.set_permission_mode("plan") == "plan"
    assert any(entry.type == "permission_mode" for entry in session.session.entries)
    summary = session.permission_summary()
    assert summary["mode"] == "plan" and summary["checkpoints"] is True
    with pytest.raises(ValueError):
        session.set_permission_mode("nope")


def test_permission_command_renders(tmp_path, faux_model):
    from io import StringIO

    from rich.console import Console

    from tynex.coding import session_commands

    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
    buffer = StringIO()
    console = Console(file=buffer, width=100)
    session_commands.permissions(session, console, "")
    output = buffer.getvalue()
    assert "accept-edits" in output
    buffer.seek(0)
    buffer.truncate(0)
    session_commands.permissions(session, console, "mode plan")
    assert "plan" in buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)
    session_commands.permissions(session, console, "mode")
    assert "usage" in buffer.getvalue()
