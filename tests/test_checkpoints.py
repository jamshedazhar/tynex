"""File checkpoints: snapshotting before edits and restoring with /undo (offline)."""

import json

from tests.conftest import faux_scenario, text_scenario
from tynex.ai.providers import faux
from tynex.coding.agent_session import AgentSession
from tynex.coding.checkpoints import MAX_SNAPSHOT_BYTES, CheckpointStore
from tynex.coding.tools.shell import shell_name


def tool_call(name, arguments, call_id="call_1"):
    return {"type": "toolcall", "id": call_id, "name": name, "arguments": arguments}


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_capture_snapshots_modified_created_and_oversized(tmp_path):
    (tmp_path / "app.py").write_text("original", encoding="utf-8")
    big = tmp_path / "big.bin"
    big.write_text("x" * (MAX_SNAPSHOT_BYTES + 1), encoding="utf-8")
    store = CheckpointStore(str(tmp_path), "test-session")
    group = store.capture("edit", ["app.py", "new.txt", "big.bin", ""])
    assert group is not None
    assert group.id == 1 and group.tool == "edit"
    actions = {entry.path: entry.action for entry in group.files}
    assert actions == {"app.py": "modified", "new.txt": "created", "big.bin": "skipped"}
    assert (store.root / "0001" / "files" / "app.py").read_text(encoding="utf-8") == "original"
    assert group.summary().startswith("#1 edit: app.py")
    second = store.capture("edit", ["app.py", "b.txt", "c.txt", "d.txt"])
    assert second.summary() == "#2 edit: app.py, b.txt, c.txt +1 more"

    # a fresh store sees the same groups (resume)
    reloaded = CheckpointStore(str(tmp_path), "test-session")
    assert [entry.id for entry in reloaded.groups()] == [1, 2]


def test_capture_ignores_nothing_and_outside_paths(tmp_path):
    store = CheckpointStore(str(tmp_path), "test-session")
    assert store.capture("edit", ["", "missing-parent/child.txt"]) is not None  # created entry
    assert store.groups()[-1].files[0].path == "missing-parent/child.txt"

    outside = str(tmp_path.parent / "elsewhere.txt")
    (tmp_path.parent / "elsewhere.txt").write_text("secret", encoding="utf-8")
    assert store.capture("edit", [outside]) is None  # outside workspace: nothing captured


def test_undo_refuses_a_tampered_manifest_path(tmp_path):
    """The manifest is plain JSON on disk: a crafted path must not touch anything outside."""
    (tmp_path / "file.txt").write_text("v1", encoding="utf-8")
    store = CheckpointStore(str(tmp_path), "test-session")
    store.capture("edit", ["file.txt"])
    outside = tmp_path.parent / "escape.txt"
    outside.write_text("untouched", encoding="utf-8")
    manifest = store.root / "0001" / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"] = [{"path": "../escape.txt", "action": "created", "size": 0}]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    reloaded = CheckpointStore(str(tmp_path), "test-session")  # read the edited manifest

    assert reloaded.undo(1) == []
    assert outside.read_text(encoding="utf-8") == "untouched"
    assert any("unsafe path" in warning for warning in reloaded.warnings)


def test_undo_restores_and_removes(tmp_path):
    (tmp_path / "keep.txt").write_text("v1", encoding="utf-8")
    store = CheckpointStore(str(tmp_path), "test-session")
    store.capture("edit", ["keep.txt", "fresh.txt"])
    (tmp_path / "keep.txt").write_text("v2", encoding="utf-8")
    (tmp_path / "fresh.txt").write_text("created later", encoding="utf-8")

    restored = store.undo(1)
    assert len(restored) == 1 and restored[0].undone
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "v1"
    assert not (tmp_path / "fresh.txt").exists()
    assert not (store.root / "0001" / "files").exists()  # snapshots cleaned up
    # nothing new to undo
    assert store.undo(1) == []


def test_a_failed_undo_keeps_the_snapshot_and_stays_retryable(tmp_path, monkeypatch):
    from tynex.coding import checkpoints as checkpoint_module

    (tmp_path / "file.txt").write_text("v1", encoding="utf-8")
    store = CheckpointStore(str(tmp_path), "test-session")
    store.capture("edit", ["file.txt"])
    (tmp_path / "file.txt").write_text("v2", encoding="utf-8")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(checkpoint_module.shutil, "copy2", fail)
    assert store.undo(1) == []
    assert any("undo failed" in warning for warning in store.warnings)
    assert store.groups()[0].undone is False  # still pending
    assert (store.root / "0001" / "files" / "file.txt").exists()  # backup kept
    assert (tmp_path / "file.txt").read_text(encoding="utf-8") == "v2"

    monkeypatch.undo()
    restored = store.undo(1)
    assert [group.id for group in restored] == [1]
    assert (tmp_path / "file.txt").read_text(encoding="utf-8") == "v1"


def test_undo_multiple_groups_restores_in_order(tmp_path):
    (tmp_path / "file.txt").write_text("first", encoding="utf-8")
    store = CheckpointStore(str(tmp_path), "test-session")
    store.capture("edit", ["file.txt"])
    (tmp_path / "file.txt").write_text("second", encoding="utf-8")
    store.capture("edit", ["file.txt"])
    (tmp_path / "file.txt").write_text("third", encoding="utf-8")

    restored = store.undo(2)
    assert [group.id for group in restored] == [2, 1]  # newest first
    assert (tmp_path / "file.txt").read_text(encoding="utf-8") == "first"
    assert store.undo(5) == []


def test_disabled_store_does_nothing(tmp_path):
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    store = CheckpointStore(str(tmp_path), "test-session", enabled=False)
    assert store.capture("edit", ["file.txt"]) is None
    assert store.undo(1) == []
    assert not store.root.exists()


def test_oversized_files_are_reported_but_not_restored(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * (MAX_SNAPSHOT_BYTES + 1), encoding="utf-8")
    store = CheckpointStore(str(tmp_path), "test-session")
    store.capture("edit", ["big.txt"])
    big.write_text("overwritten", encoding="utf-8")
    store.undo(1)
    assert big.read_text(encoding="utf-8") == "overwritten"  # not restorable, documented


# ---------------------------------------------------------------------------
# Session integration
# ---------------------------------------------------------------------------


def test_session_captures_edits_and_undo_informs_the_model(tmp_path, faux_model):
    script = [
        faux_scenario(
            [
                tool_call("write", {"path": "notes.md", "content": "first version"}),
                {"type": "stop", "reason": "toolUse"},
            ]
        ),
        text_scenario("written"),
    ]
    faux.reset_script(script)
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["write"], no_session=True)
    session.prompt("write it", emit=lambda event: None)
    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "first version"
    assert [group.tool for group in session.checkpoints.groups()] == ["write"]

    groups = session.undo_last(1)
    assert len(groups) == 1
    assert not (tmp_path / "notes.md").exists()  # write created it
    assert any(entry.type == "undo" for entry in session.session.entries)
    notes = [message for message in session.context.messages if "[undo]" in str(getattr(message, "content", ""))]
    assert notes and "notes.md" in notes[0].content


def test_session_does_not_checkpoint_reads_or_shell(tmp_path, faux_model):
    (tmp_path / "in.txt").write_text("data", encoding="utf-8")
    faux.reset_script(
        [
            faux_scenario(
                [
                    tool_call("read", {"path": "in.txt"}),
                    tool_call(shell_name, {"command": "echo hi"}, "call_2"),
                    {"type": "stop", "reason": "toolUse"},
                ]
            ),
            text_scenario("done"),
        ]
    )
    session = AgentSession(
        cwd=str(tmp_path),
        model=faux_model,
        tool_names=["read", shell_name],
        no_session=True,
        permission_rules=[],  # shell still needs approval -> denied, but nothing captured anyway
    )
    session.prompt("go", emit=lambda event: None)
    assert session.checkpoints.groups() == []


def test_checkpoints_can_be_disabled(tmp_path, faux_model):
    session = AgentSession(
        cwd=str(tmp_path), model=faux_model, tool_names=["write"], no_session=True, checkpoints=False
    )
    assert session.checkpoints is None
    assert session.undo_last(1) == []
    assert session.permission_summary()["checkpoints"] is False


def test_undo_command_renders(tmp_path, faux_model):
    from io import StringIO

    from rich.console import Console

    from tynex.coding import session_commands

    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["write"], no_session=True)
    buffer = StringIO()
    console = Console(file=buffer, width=100)
    session_commands.undo_session(session, console, "")
    assert "Nothing to undo" in buffer.getvalue()

    (tmp_path / "a.txt").write_text("v1", encoding="utf-8")
    session.checkpoints.capture("write", ["a.txt"])
    (tmp_path / "a.txt").write_text("v2", encoding="utf-8")
    buffer.seek(0)
    buffer.truncate(0)
    session_commands.undo_session(session, console, "1")
    assert "Reverted" in buffer.getvalue()
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "v1"

    buffer.seek(0)
    buffer.truncate(0)
    session_commands.undo_session(session, console, "nope")
    assert "Usage" in buffer.getvalue()

    disabled = AgentSession(
        cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True, checkpoints=False
    )
    buffer.seek(0)
    buffer.truncate(0)
    session_commands.undo_session(disabled, console, "")
    assert "disabled" in buffer.getvalue()


def test_checkpoints_follow_new_and_switched_sessions(tmp_path, faux_model):
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["write"], no_session=True)
    first = session.checkpoints.namespace
    session.new_session()
    assert session.checkpoints.namespace != first
    assert session.session.session_id != "ephemeral" or session.checkpoints.namespace == session._ephemeral_id
