"""The plan/approve gate: plan mode, submit_plan, approval and audit (offline)."""

import pytest

from tests.conftest import faux_scenario, text_scenario
from tynex.ai.providers import faux
from tynex.coding.agent_session import AgentSession

PLAN = """## Goal
Add a retry helper.

## Steps
1. Add `retry()` to `src/app.py`.
2. Cover it with a test.

## Verification
`uv run pytest -q`."""


def tool_call(name, arguments, call_id="call_1"):
    return {"type": "toolcall", "id": call_id, "name": name, "arguments": arguments}


def plan_scenario(plan=PLAN, call_id="call_1"):
    return faux_scenario([tool_call("submit_plan", {"plan": plan}, call_id), {"type": "stop", "reason": "toolUse"}])


def make_session(tmp_path, faux_model, tool_names=None, **kwargs):
    return AgentSession(
        cwd=str(tmp_path),
        model=faux_model,
        tool_names=tool_names or ["read", "write", "submit_plan"],
        no_session=True,
        **kwargs,
    )


def run(session, text, script):
    faux.reset_script(script)
    events: list[dict] = []
    session.prompt(text, emit=events.append)
    return events


def test_submit_plan_ends_the_turn_and_records_the_plan(tmp_path, faux_model):
    session = make_session(tmp_path, faux_model, permission_mode="plan")
    events = run(session, "plan the change", [plan_scenario()])

    ready = [event for event in events if event["type"] == "plan_ready"]
    assert len(ready) == 1 and ready[0]["plan"] == PLAN
    assert session.pending_plan == {"id": ready[0]["planId"], "plan": PLAN}
    assert any(entry.type == "plan" and entry.data["plan"] == PLAN for entry in session.session.entries)

    # The planning turn is over: the tool result terminates the loop instead of
    # letting the model keep working, so nothing was written.
    assert not (tmp_path / "src").exists()
    endings = [event for event in events if event["type"] == "tool_execution_end"]
    assert endings and "decision" in endings[0]["result"].content[0].text


def test_a_plan_ends_the_turn_even_next_to_another_tool_call(tmp_path, faux_model):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    session = make_session(tmp_path, faux_model, permission_mode="plan")
    script = [
        faux_scenario(
            [
                tool_call("read", {"path": "a.txt"}, "call_1"),
                {"type": "toolcall", "id": "call_2", "name": "submit_plan", "arguments": {"plan": PLAN}},
                {"type": "stop", "reason": "toolUse"},
            ]
        ),
        text_scenario("must not be reached"),
    ]
    events = run(session, "plan it", script)

    assert session.pending_plan is not None
    assistant_messages = [
        event["message"] for event in events if event["type"] == "message_end" and event["message"].role == "assistant"
    ]
    assert len(assistant_messages) == 1  # no second model request after the plan


def test_a_second_plan_is_refused_while_one_is_pending(tmp_path, faux_model):
    session = make_session(tmp_path, faux_model, permission_mode="plan")
    script = [
        faux_scenario(
            [
                tool_call("submit_plan", {"plan": "first plan"}, "call_1"),
                tool_call("submit_plan", {"plan": "second plan"}, "call_2"),
                {"type": "stop", "reason": "toolUse"},
            ]
        )
    ]
    events = run(session, "plan it", script)

    ready = [event for event in events if event["type"] == "plan_ready"]
    assert len(ready) == 1 and ready[0]["plan"] == "first plan"
    assert session.pending_plan is not None and session.pending_plan["plan"] == "first plan"
    endings = [event for event in events if event["type"] == "tool_execution_end"]
    assert endings[1]["isError"] is True
    assert "already waiting for approval" in endings[1]["result"].content[0].text


def test_submit_plan_is_refused_outside_plan_mode(tmp_path, faux_model):
    session = make_session(tmp_path, faux_model)  # accept-edits
    events = run(session, "just do it", [plan_scenario()])

    assert session.pending_plan is None
    assert not [event for event in events if event["type"] == "plan_ready"]
    endings = [event for event in events if event["type"] == "tool_execution_end"]
    assert endings and endings[0]["isError"] is True
    assert "only available in plan mode" in endings[0]["result"].content[0].text


def test_approve_plan_switches_mode_and_audits_the_decision(tmp_path, faux_model):
    session = make_session(tmp_path, faux_model, permission_mode="plan")
    run(session, "plan it", [plan_scenario()])
    plan_id = session.pending_plan["id"]

    approved = session.approve_plan()

    assert approved["plan"] == PLAN
    assert session.pending_plan is None
    assert session.permission_gate.mode == "accept-edits"
    assert any(entry.type == "plan_approved" and entry.data["planId"] == plan_id for entry in session.session.entries)
    with pytest.raises(ValueError, match="No plan is waiting"):
        session.approve_plan()


def test_reject_plan_keeps_plan_mode(tmp_path, faux_model):
    session = make_session(tmp_path, faux_model, permission_mode="plan")
    run(session, "plan it", [plan_scenario()])

    session.reject_plan("split step 2")

    assert session.permission_gate.mode == "plan"
    assert session.pending_plan is None
    rejected = [entry for entry in session.session.entries if entry.type == "plan_rejected"]
    assert rejected and rejected[0].data["feedback"] == "split step 2"


def test_a_pending_plan_survives_resume(tmp_path, faux_model):
    first = AgentSession(
        cwd=str(tmp_path), model=faux_model, tool_names=["read", "submit_plan"], permission_mode="plan"
    )
    run(first, "plan it", [plan_scenario()])
    path = first.session.file_path

    resumed = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["submit_plan"], session_file=path)
    assert resumed.pending_plan is not None and resumed.pending_plan["plan"] == PLAN
    resumed.approve_plan()

    decided = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["submit_plan"], session_file=path)
    assert decided.pending_plan is None


def test_an_approved_plan_runs_the_edit_it_was_blocking(tmp_path, faux_model):
    session = make_session(tmp_path, faux_model, permission_mode="plan")
    run(session, "plan it", [plan_scenario()])
    session.approve_plan()

    events = run(
        session,
        "execute the approved plan",
        [
            faux_scenario(
                [
                    tool_call("write", {"path": "app.py", "content": "print('hi')\n"}, "call_2"),
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("done"),
        ],
    )

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert not [event for event in events if event["type"] == "permission_decision"]


def test_plan_mode_prompt_names_the_tool_and_the_gate(tmp_path, faux_model):
    session = make_session(tmp_path, faux_model, permission_mode="plan")
    prompt = session.system_prompt_text()
    assert "submit_plan" in prompt
    assert "plan mode" in prompt.lower()
