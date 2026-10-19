"""Tests for the RPC mode (JSON protocol over stdin/stdout)."""

import io
import json
from types import SimpleNamespace
from typing import Any

import tynex.cli as modes
from tynex.ai.types import UserMessage
from tynex.cli import RpcMode


class FakeSession:
    """Duck-typed AgentSession recording every RPC-driven call."""

    def __init__(self) -> None:
        self.cwd = "C:/work"
        self.running = False
        self.thinking = "low"
        self.tools = [SimpleNamespace(name="read"), SimpleNamespace(name="bash")]
        self.context = SimpleNamespace(messages=[UserMessage(content="hello")])
        self.session = SimpleNamespace(session_id="sess-1", file_path="sess-1.jsonl")
        self.model = SimpleNamespace(provider="faux", id="test")
        self.calls: list[tuple[str, Any]] = []
        self.stats_data: dict[str, Any] = {"messages": 1, "cost": 0.0}
        self.model_result = SimpleNamespace(provider="faux", id="other")
        self.compact_result = True
        self.prompt_error: str | None = None

    def prompt(self, message: str, emit) -> None:
        self.calls.append(("prompt", message))
        if self.prompt_error is not None:
            raise RuntimeError(self.prompt_error)
        emit({"type": "agent_start"})

    def stats(self) -> dict[str, Any]:
        return dict(self.stats_data)

    def set_model_by_pattern(self, pattern: str):
        self.calls.append(("set_model", pattern))
        return self.model_result

    def compact_now(self, instructions: str | None) -> bool:
        self.calls.append(("compact", instructions))
        return self.compact_result

    def new_session(self) -> None:
        self.calls.append(("new_session",))
        self.session.session_id = "sess-2"

    def switch_session(self, path: str) -> None:
        self.calls.append(("switch_session", path))
        self.session.session_id = "sess-3"

    def steer(self, message: str) -> None:
        self.calls.append(("steer", message))

    def follow_up(self, message: str) -> None:
        self.calls.append(("follow_up", message))

    def abort(self) -> None:
        self.calls.append(("abort",))

    def finalize_aborted_turn(self, emit=None) -> None:
        self.calls.append(("finalize_aborted_turn",))


def drive(monkeypatch, session: FakeSession, commands: list[Any]) -> list[dict[str, Any]]:
    """Run the RPC loop against ``commands`` and return the emitted JSON payloads."""
    lines = [command if isinstance(command, str) else json.dumps(command) for command in commands]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()
    monkeypatch.setattr(modes.sys, "stdin", stdin)
    monkeypatch.setattr(modes.sys, "stdout", stdout)
    assert RpcMode(session).run() == 0
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def test_ready_and_immediate_commands(monkeypatch):
    session = FakeSession()
    outputs = drive(
        monkeypatch,
        session,
        [
            {"id": "1", "type": "steer", "message": "left"},
            {"id": "2", "type": "follow_up", "message": "next"},
            {"id": "3", "type": "abort"},
            {"id": "4", "type": "get_state"},
            {"id": "5", "type": "get_messages"},
            {"id": "6", "type": "shutdown"},
        ],
    )

    assert outputs[0] == {
        "type": "ready",
        "session": {"id": "sess-1", "file": "sess-1.jsonl", "model": "faux/test", "cwd": "C:/work"},
    }
    responses = {response.get("id"): response for response in outputs if response["type"] == "response"}
    assert responses["1"]["data"] == {"queued": "steer"}
    assert responses["2"]["data"] == {"queued": "followUp"}
    assert responses["3"]["success"] is True
    assert responses["4"]["data"]["running"] is False
    assert responses["4"]["data"]["thinking"] == "low"
    assert responses["4"]["data"]["tools"] == ["read", "bash"]
    assert responses["4"]["data"]["messages"] == 1
    assert responses["5"]["data"]["messages"][0]["role"] == "user"
    assert responses["6"] == {"type": "response", "id": "6", "command": "shutdown", "success": True, "data": {}}
    assert session.calls == [("steer", "left"), ("follow_up", "next"), ("abort",), ("finalize_aborted_turn",)]


def test_abort_cancels_a_queued_prompt(monkeypatch):
    session = FakeSession()
    mode = RpcMode(session)
    stdout = io.StringIO()
    monkeypatch.setattr(modes.sys, "stdout", stdout)
    mode._commands.put({"id": "1", "type": "prompt", "message": "slow"})
    mode._queued_prompts = 1
    mode._handle_immediate({"id": "2", "type": "abort"})
    mode._commands.put(None)
    mode._worker_loop()

    payloads = [json.loads(line) for line in stdout.getvalue().splitlines()]
    responses = {response.get("id"): response for response in payloads if response["type"] == "response"}
    assert responses["1"]["success"] is True
    assert responses["1"]["data"]["aborted"] is True
    assert ("prompt", "slow") not in session.calls
    assert session.calls == [("abort",), ("finalize_aborted_turn",)]


def test_worker_commands(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(
        modes,
        "list_sessions",
        lambda cwd: [
            {"id": "s1", "messages": 2, "cwd": "C:/work", "file": "s1.jsonl"},
            {"id": "s2", "messages": 0, "cwd": "C:/work", "file": "s2.jsonl"},
        ],
    )
    monkeypatch.setattr(
        modes, "find_session", lambda cwd, identifier: "C:/work/s1.jsonl" if identifier == "s1" else None
    )

    outputs = drive(
        monkeypatch,
        session,
        [
            {"id": "1", "type": "prompt", "message": "do it"},
            {"id": "2", "type": "set_model", "model": "faux/other"},
            {"id": "3", "type": "compact", "instructions": "tight"},
            {"id": "4", "type": "new_session"},
            {"id": "5", "type": "list_sessions"},
            {"id": "6", "type": "resume_session", "session": "s1"},
            {"id": "7", "type": "shutdown"},
        ],
    )

    events = [output for output in outputs if output["type"] == "event"]
    assert events[0]["event"] == {"type": "agent_start"}

    responses = {response.get("id"): response for response in outputs if response["type"] == "response"}
    assert responses["1"]["data"]["state"]["messages"] == 1
    assert responses["2"]["data"] == {"model": "faux/other"}
    assert responses["3"]["data"]["compacted"] is True
    assert responses["4"]["data"] == {"sessionId": "sess-2"}
    assert [entry["id"] for entry in responses["5"]["data"]["sessions"]] == ["s1", "s2"]
    assert all("cwd" not in entry for entry in responses["5"]["data"]["sessions"])
    assert responses["6"]["data"] == {"sessionId": "sess-3"}

    assert session.calls[:4] == [
        ("prompt", "do it"),
        ("set_model", "faux/other"),
        ("compact", "tight"),
        ("new_session",),
    ]
    assert ("switch_session", "C:/work/s1.jsonl") in session.calls


def test_invalid_json_unknown_commands_and_bad_resume(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(modes, "find_session", lambda cwd, identifier: None)
    outputs = drive(
        monkeypatch,
        session,
        [
            "not json",
            "[1, 2, 3]",  # valid JSON but not an object: ignored
            "",  # blank line: ignored
            {"id": "1", "type": "bogus"},
            {"id": "2", "type": "resume_session", "session": "missing"},
            {"id": "3", "type": "shutdown"},
        ],
    )

    responses = [output for output in outputs if output["type"] == "response"]
    invalid = responses[0]
    assert invalid["success"] is False
    assert "Invalid JSON" in invalid["error"]

    by_id = {response.get("id"): response for response in responses}
    assert by_id["1"]["success"] is False
    assert "Unknown command type: 'bogus'" in by_id["1"]["error"]
    assert by_id["2"]["success"] is False
    assert "No session matching 'missing'" in by_id["2"]["error"]
    assert by_id["3"]["success"] is True


def test_worker_reports_errors_and_keeps_processing(monkeypatch):
    session = FakeSession()
    session.prompt_error = "model exploded"
    outputs = drive(
        monkeypatch,
        session,
        [
            {"id": "1", "type": "prompt", "message": "boom"},
            {"id": "2", "type": "compact"},
            {"id": "3", "type": "shutdown"},
        ],
    )

    by_id = {response.get("id"): response for response in outputs if response["type"] == "response"}
    assert by_id["1"]["success"] is False
    assert by_id["1"]["error"] == "model exploded"
    assert by_id["2"]["success"] is True


def test_shutdown_stops_reading_further_commands(monkeypatch):
    session = FakeSession()
    outputs = drive(monkeypatch, session, [{"id": "1", "type": "shutdown"}, {"id": "2", "type": "get_state"}])
    assert [output.get("id") for output in outputs if output["type"] == "response"] == ["1"]


def test_eof_without_shutdown_drains_queued_commands(monkeypatch):
    session = FakeSession()
    outputs = drive(monkeypatch, session, [{"id": "1", "type": "compact"}])
    responses = [output for output in outputs if output["type"] == "response"]
    assert responses[0]["id"] == "1"
    assert responses[0]["data"]["compacted"] is True


def test_immediate_commands_stop_when_shutdown_lands(monkeypatch):
    session = FakeSession()
    mode = RpcMode(session)
    stdin = io.StringIO(
        json.dumps({"id": "1", "type": "get_state"}) + "\n" + json.dumps({"id": "2", "type": "get_state"}) + "\n"
    )
    stdout = io.StringIO()
    monkeypatch.setattr(modes.sys, "stdin", stdin)
    monkeypatch.setattr(modes.sys, "stdout", stdout)

    seen: list[str] = []

    def immediate(command):
        seen.append(command["id"])
        mode._shutdown.set()  # a shutdown processed by the worker lands mid-stream
        mode._respond(command, True, {})
        return True

    monkeypatch.setattr(mode, "_handle_immediate", immediate)
    assert mode.run() == 0
    assert seen == ["1"]
