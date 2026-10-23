"""Tests for the session manager: persistence, tree branching, compaction."""

import json
import os

from tynex.ai.types import AssistantMessage, TextContent, UserMessage
from tynex.coding.session import SessionManager, find_session, latest_session_file, list_sessions


def append_exchange(session: SessionManager, user_text: str, assistant_text: str):
    user_entry = session.append_message(UserMessage(content=user_text))
    assistant_entry = session.append_message(AssistantMessage(content=[TextContent(text=assistant_text)]))
    return user_entry, assistant_entry


def test_session_persists_and_reloads(tmp_path):
    session = SessionManager(str(tmp_path))
    append_exchange(session, "hello", "hi there")
    file = session.file_path
    assert file is not None

    with open(file, encoding="utf-8") as fh:
        lines = fh.read().strip().splitlines()
    types = [json.loads(line)["type"] for line in lines]
    assert types[0] == "session"
    assert types.count("message") == 2

    reloaded = SessionManager(str(tmp_path), session_file=file)
    messages = reloaded.branch_messages()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].text() == "hello"
    assert messages[1].text() == "hi there"


def test_session_compaction_replaces_older_messages(tmp_path):
    session = SessionManager(str(tmp_path))
    entries = []
    for i in range(1, 6):
        user_entry, _ = append_exchange(session, f"msg {i}", f"answer {i}")
        entries.append(user_entry)
    kept_id = entries[3].id  # keep "msg 4" and later
    session.append_compaction("Summary of messages 1-3", kept_id, 500)

    messages = session.branch_messages()
    assert messages[0].role == "user"
    assert "Context summary" in messages[0].text()
    assert "Summary of messages 1-3" in messages[0].text()
    texts = [m.text() for m in messages[1:]]
    assert "msg 4" in texts
    assert "msg 5" in texts
    assert "msg 1" not in texts


def test_selection_prefers_newest_branch_entry(tmp_path):
    session = SessionManager(str(tmp_path))
    session.append_model_change("anthropic", "claude-sonnet-4-5")
    session.append_message(UserMessage(content="hi"))
    session.append_message(AssistantMessage(content=[TextContent(text="hello")], provider="faux", model="test"))
    # The newest selection-bearing entry wins: the assistant response.
    assert session.selection() == ("faux", "test")

    # The selection is restored from the file on reload.
    session.append_model_change("openai", "gpt-5")
    assert session.selection() == ("openai", "gpt-5")
    reloaded = SessionManager(str(tmp_path), session_file=session.file_path)
    assert reloaded.selection() == ("openai", "gpt-5")


def test_list_sessions_and_labels(tmp_path):
    session = SessionManager(str(tmp_path))
    append_exchange(session, "hi", "hello")
    session.append_label("my session")

    sessions = list_sessions(str(tmp_path))
    assert len(sessions) >= 1
    assert sessions[0]["messages"] == 2
    assert sessions[0]["label"] == "my session"


def test_list_sessions_isolates_cwds_that_share_a_folder(tmp_path):
    from tynex.coding.session import encode_cwd

    first_dir = tmp_path / "my-project"
    second_dir = tmp_path / "my_project"
    first_dir.mkdir()
    second_dir.mkdir()
    assert encode_cwd(str(first_dir)) == encode_cwd(str(second_dir))

    first = SessionManager(str(first_dir))
    second = SessionManager(str(second_dir))
    append_exchange(first, "hello", "hi")
    append_exchange(second, "other", "answer")

    sessions = list_sessions(str(first_dir))
    assert [s["file"] for s in sessions] == [first.file_path]
    assert latest_session_file(str(first_dir)) == first.file_path


def test_find_session_rejects_ambiguous_prefixes(tmp_path):
    first = SessionManager(str(tmp_path))
    second = SessionManager(str(tmp_path))
    prefix = os.path.commonprefix(  # noqa: RUF071 — basenames, not paths
        [os.path.basename(first.file_path), os.path.basename(second.file_path)]
    )
    assert prefix
    assert find_session(str(tmp_path), prefix) is None

    assert find_session(str(tmp_path), first.session_id) == first.file_path
    assert find_session(str(tmp_path), os.path.basename(first.file_path)) == first.file_path


def test_no_session_mode(tmp_path):
    session = SessionManager(str(tmp_path), no_session=True)
    session.append_message(UserMessage(content="x"))
    assert session.file_path is None
    assert len(session.branch_messages()) == 1
    assert session.session_id == "ephemeral" or session.session_id
