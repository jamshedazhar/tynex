"""Session manager: append-only JSONL transcripts with a tree structure."""

import html
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from tynex.ai.types import (
    AssistantMessage,
    BashExecutionMessage,
    ImageContent,
    Message,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    message_from_dict,
    message_text,
    message_to_dict,
)
from tynex.coding.settings import sessions_dir
from tynex.coding.tools.shell import shell_prompt


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class SessionEntry:
    id: str
    parent_id: str | None
    timestamp: int
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "parentId": self.parent_id,
            "timestamp": self.timestamp,
            "type": self.type,
        }
        out.update(self.data)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Self:
        data = {k: v for k, v in d.items() if k not in ("id", "parentId", "timestamp", "type")}
        return SessionEntry(
            id=d.get("id", _new_id()),
            parent_id=d.get("parentId"),
            timestamp=d.get("timestamp", 0),
            type=d.get("type", "message"),
            data=data,
        )

    @property
    def message(self) -> Message | None:
        if self.type != "message":
            return None
        return message_from_dict(self.data.get("message", {}))

    def summary_line(self, max_len: int = 60) -> str:
        if self.type == "message":
            message = self.message
            role = getattr(message, "role", "?") if message else "?"
            text = " ".join(message_text(message).split()) if message else ""
            return f"{role}: {text[:max_len]}"
        if self.type == "compaction":
            return f"compaction: {self.data.get('summary', '')[:max_len]}"
        if self.type == "model_change":
            return f"model: {self.data.get('provider')}/{self.data.get('modelId')}"
        if self.type == "thinking_level_change":
            return f"thinking: {self.data.get('thinkingLevel')}"
        if self.type == "session_label":
            return f"label: {self.data.get('label', '')}"
        return self.type


def encode_cwd(cwd: str) -> str:
    """Directory-safe cwd encoding for the per-project session folder."""
    text = os.path.abspath(cwd)
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return f"--{text}--"


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


class SessionManager:
    """Owns one session file. All persistence goes through ``append``."""

    def __init__(self, cwd: str, session_file: str | None = None, no_session: bool = False) -> None:
        self.cwd = os.path.abspath(cwd)
        self.no_session = no_session
        self.entries: list[SessionEntry] = []
        self.by_id: dict[str, SessionEntry] = {}
        self.current_leaf_id: str | None = None
        self.file_path: str | None = None
        if no_session:
            return
        if session_file:
            path = Path(session_file).expanduser()
            if path.is_file():
                self._load(path)
            else:
                self.file_path = str(path)
                self._init_file()
        else:
            self._create_new()

    # -- lifecycle ----------------------------------------------------------
    def _session_dir(self) -> Path:
        path = sessions_dir() / encode_cwd(self.cwd)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _create_new(self) -> None:
        stamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        name = f"{stamp}_tynex-{_new_id()}.jsonl"
        self.file_path = str(self._session_dir() / name)
        self._init_file()

    def _init_file(self) -> None:
        if self.file_path:
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
        entry = SessionEntry(
            id=_new_id(), parent_id=None, timestamp=_now_ms(), type="session", data={"cwd": self.cwd, "version": 1}
        )
        self.entries.append(entry)
        self.by_id[entry.id] = entry
        self.current_leaf_id = entry.id
        self._write_entry(entry)

    def _load(self, path: Path) -> None:
        self.file_path = str(path)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry = SessionEntry.from_dict(raw)
                self.entries.append(entry)
                self.by_id[entry.id] = entry
        if self.entries:
            self.current_leaf_id = self.entries[-1].id

    def _write_entry(self, entry: SessionEntry) -> None:
        if self.no_session or not self.file_path:
            return
        with open(self.file_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    # -- writing ------------------------------------------------------------
    def append(self, entry_type: str, **data: Any) -> SessionEntry:
        entry = SessionEntry(
            id=_new_id(), parent_id=self.current_leaf_id, timestamp=_now_ms(), type=entry_type, data=data
        )
        self.entries.append(entry)
        self.by_id[entry.id] = entry
        self.current_leaf_id = entry.id
        self._write_entry(entry)
        return entry

    def append_message(self, message: Message) -> SessionEntry:
        return self.append("message", message=message_to_dict(message))

    def append_compaction(self, summary: str, first_kept_entry_id: str, tokens_before: int) -> SessionEntry:
        return self.append(
            "compaction", summary=summary, firstKeptEntryId=first_kept_entry_id, tokensBefore=tokens_before
        )

    def append_model_change(self, provider: str, model_id: str) -> SessionEntry:
        return self.append("model_change", provider=provider, modelId=model_id)

    def append_thinking_level_change(self, level: str) -> SessionEntry:
        return self.append("thinking_level_change", thinkingLevel=level)

    def append_label(self, label: str) -> SessionEntry:
        return self.append("session_label", label=label)

    # -- reading ------------------------------------------------------------
    @property
    def session_id(self) -> str:
        if self.entries:
            return self.entries[0].id
        return "ephemeral"

    @property
    def header_id(self) -> str | None:
        return self.entries[0].id if self.entries else None

    @property
    def label(self) -> str | None:
        for entry in reversed(self.path_entries()):
            if entry.type == "session_label":
                return entry.data.get("label")
        return None

    def path_entries(self) -> list[SessionEntry]:
        """Entries from root to the current leaf."""
        if not self.current_leaf_id:
            return []
        return self.path_to(self.current_leaf_id)

    def path_to(self, leaf_id: str) -> list[SessionEntry]:
        """Entries from root to ``leaf_id`` (the original branch line)."""
        path: list[SessionEntry] = []
        current = self.by_id.get(leaf_id)
        while current is not None:
            path.append(current)
            current = self.by_id.get(current.parent_id) if current.parent_id else None
        path.reverse()
        return path

    def branch_messages(self) -> list[Message]:
        """The transcript the model sees, honoring the latest compaction."""
        path = [e for e in self.path_entries() if e.type in ("message", "compaction")]
        last_compaction_index: int | None = None
        for i, entry in enumerate(path):
            if entry.type == "compaction":
                last_compaction_index = i
        messages: list[Message] = []
        if last_compaction_index is None:
            for entry in path:
                message = entry.message
                if message is not None and not isinstance(message, SystemMessage):
                    messages.append(message)
            return messages
        compaction = path[last_compaction_index]
        first_kept = compaction.data.get("firstKeptEntryId")
        kept_index: int | None = None
        for i, entry in enumerate(path[:last_compaction_index]):
            if entry.id == first_kept:
                kept_index = i
                break
        summary = compaction.data.get("summary", "")
        messages.append(
            UserMessage(
                content=(
                    f"[Context summary from a previous conversation segment]\n\n{summary}\n\n[End of context summary]"
                )
            )
        )
        # firstKeptEntryId is inclusive: it is the first entry kept in context.
        start = kept_index if kept_index is not None else 0
        for entry in path[start:]:
            if entry.type != "message":
                continue
            message = entry.message
            if message is not None and not isinstance(message, SystemMessage):
                messages.append(message)
        return messages

    def message_entries(self) -> list[SessionEntry]:
        return [e for e in self.path_entries() if e.type == "message"]

    def find_entry(self, prefix: str) -> SessionEntry | None:
        if prefix in self.by_id:
            return self.by_id[prefix]
        matches = [e for e in self.entries if e.id.startswith(prefix)]
        return matches[0] if len(matches) == 1 else None

    def thinking_level(self) -> str | None:
        """The newest thinking-level change on the branch (audit record)."""
        for entry in reversed(self.path_entries()):
            if entry.type == "thinking_level_change":
                return entry.data.get("thinkingLevel")
        return None

    def selection(self) -> tuple[str, str] | None:
        """Provider/model implied by the branch."""
        for entry in reversed(self.path_entries()):
            if entry.type == "model_change":
                return entry.data.get("provider", ""), entry.data.get("modelId", "")
            if entry.type == "message":
                message = entry.message
                if getattr(message, "role", None) == "assistant":
                    provider = getattr(message, "provider", "") or ""
                    model_id = getattr(message, "model", "") or ""
                    if provider and model_id:
                        return provider, model_id
        return None


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def list_sessions(cwd: str, limit: int = 50) -> list[dict[str, Any]]:
    directory = sessions_dir() / encode_cwd(cwd)
    if not directory.is_dir():
        return []
    wanted_cwd = _normalized_path(cwd)
    sessions: list[dict[str, Any]] = []
    for file in directory.glob("*.jsonl"):
        try:
            stat = file.stat()
        except OSError:
            continue
        header: dict[str, Any] = {}
        message_count = 0
        label: str | None = None
        try:
            with open(file, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not header:
                        header = raw
                    elif raw.get("type") == "message":
                        message_count += 1
                    elif raw.get("type") == "session_label":
                        label = raw.get("label")
        except OSError:
            continue
        # encode_cwd() collapses punctuation, so distinct directories can share
        # a folder; only sessions recorded for this cwd belong to it. Legacy
        # headers without a cwd are kept.
        stored_cwd = header.get("cwd")
        if isinstance(stored_cwd, str) and stored_cwd and _normalized_path(stored_cwd) != wanted_cwd:
            continue
        sessions.append(
            {
                "file": str(file),
                "id": header.get("id", ""),
                "cwd": header.get("cwd", ""),
                "messages": message_count,
                "modified": stat.st_mtime,
                "label": label,
            }
        )
    sessions.sort(key=lambda s: s["modified"], reverse=True)
    return sessions[:limit]


def latest_session_file(cwd: str) -> str | None:
    sessions = list_sessions(cwd, limit=1)
    return sessions[0]["file"] if sessions else None


def find_session(cwd: str, identifier: str) -> str | None:
    """Match a session by id prefix, file name, or path.

    An identifier that matches several sessions is ambiguous and returns
    ``None`` rather than silently resuming the newest one.
    """
    path = Path(identifier).expanduser()
    if path.is_file():
        return str(path)
    sessions = list_sessions(cwd, limit=500)
    exact = [s for s in sessions if s["id"] == identifier or Path(s["file"]).name == identifier]
    if len(exact) == 1:
        return exact[0]["file"]
    matches = [
        s
        for s in sessions
        if s["id"].startswith(identifier)
        or Path(s["file"]).name.startswith(identifier)
        or identifier in Path(s["file"]).name
    ]
    return matches[0]["file"] if len(matches) == 1 else None


APP_NAME = "tynex"


def _resolve_output(output_path: str | None, default_name: str, cwd: str | None) -> Path:
    path = Path(output_path).expanduser() if output_path else Path(default_name)
    if not path.is_absolute():
        path = Path(cwd or os.getcwd()) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def serialize_session_branch(session: SessionManager) -> str:
    """The active branch as JSONL."""
    header = SessionEntry(
        id=session.session_id,
        parent_id=None,
        timestamp=_now_ms(),
        type="session",
        data={"cwd": session.cwd, "version": 1},
    )
    lines = [json.dumps(header.to_dict(), ensure_ascii=False)]
    parent = header.id
    for entry in session.path_entries():
        if entry.type == "session":
            continue
        record = entry.to_dict()
        record["parentId"] = parent
        parent = entry.id
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def export_session_to_jsonl(session: SessionManager, output_path: str | None = None, cwd: str | None = None) -> str:
    """Write the current branch as JSONL; returns the resolved file path."""
    default_name = f"session-{time.strftime('%Y-%m-%dT%H-%M-%S')}.jsonl"
    path = _resolve_output(output_path, default_name, cwd)
    path.write_text(serialize_session_branch(session), encoding="utf-8", newline="\n")
    return str(path)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 52rem; padding: 1.5rem 1rem 4rem; line-height: 1.5; }
header { border-bottom: 1px solid #8884; padding-bottom: 1rem; margin-bottom: 1.5rem; }
h1 { font-size: 1.3rem; margin: 0 0 .5rem; }
header dl { display: grid; grid-template-columns: max-content 1fr; gap: .1rem .8rem;
            margin: 0; font-size: .85rem; }
header dt { opacity: .6; }
header dd { margin: 0; overflow-wrap: anywhere; }
.entry { border: 1px solid #8884; border-radius: .5rem; padding: .6rem .8rem; margin: .8rem 0; }
.entry .role { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; opacity: .6;
               margin-bottom: .35rem; }
.entry pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: .3rem 0; font-size: .85rem;
             font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; }
.entry p { margin: .3rem 0; }
.user { border-color: #3b82f680; }
.assistant { border-color: #22c55e80; }
.tool { border-color: #eab30880; font-size: .9rem; }
.bash { border-color: #a855f780; }
.summary { border-style: dashed; opacity: .85; }
.meta { font-size: .78rem; opacity: .6; margin: .4rem 0; }
.cmd { font-family: ui-monospace, Consolas, monospace; font-size: .85rem; }
.status { font-size: .78rem; opacity: .7; }
.error { color: #dc2626; }
details summary { cursor: pointer; font-size: .8rem; opacity: .7; }
img { max-width: 100%; border-radius: .35rem; margin: .3rem 0; }
"""


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def _render_user(message: UserMessage) -> str:
    parts: list[str] = []
    if isinstance(message.content, str):
        parts.append(f"<pre>{_esc(message.content)}</pre>")
    else:
        for part in message.content:
            if isinstance(part, TextContent):
                parts.append(f"<pre>{_esc(part.text)}</pre>")
            elif isinstance(part, ImageContent):
                parts.append(f'<img src="data:{_esc(part.mime_type)};base64,{_esc(part.data)}" alt="image">')
    return "".join(parts)


def _render_assistant(message: AssistantMessage) -> str:
    parts: list[str] = []
    for part in message.content:
        if isinstance(part, TextContent) and part.text:
            parts.append(f"<pre>{_esc(part.text)}</pre>")
        elif isinstance(part, ThinkingContent) and part.thinking:
            parts.append(f"<details><summary>thinking</summary><pre>{_esc(part.thinking)}</pre></details>")
        elif isinstance(part, ToolCall):
            args = json.dumps(part.arguments, ensure_ascii=False, indent=2)
            parts.append(f"<p class='cmd'>→ {_esc(part.name)}</p><pre>{_esc(args)}</pre>")
    if message.error_message:
        parts.append(f"<p class='error status'>{_esc(message.error_message)}</p>")
    return "".join(parts)


def _render_bash(message: BashExecutionMessage) -> str:
    prompt = shell_prompt
    parts = [f"<p class='cmd'>{prompt} {_esc(message.command)}</p>"]
    if message.output:
        parts.append(f"<pre>{_esc(message.output)}</pre>")
    status: list[str] = []
    if message.cancelled:
        status.append("command cancelled")
    elif message.exit_code is not None and message.exit_code != 0:
        status.append(f"command exited with code {message.exit_code}")
    if message.truncated and message.full_output_path:
        status.append(f"output truncated, full output: {message.full_output_path}")
    if message.exclude_from_context:
        status.append("not sent to the model")
    if status:
        parts.append(f"<p class='status'>{_esc(' · '.join(status))}</p>")
    return "".join(parts)


def _render_message(message: Message) -> str:
    if isinstance(message, UserMessage):
        return f"<div class='entry user'><div class='role'>user</div>{_render_user(message)}</div>"
    if isinstance(message, AssistantMessage):
        return f"<div class='entry assistant'><div class='role'>assistant</div>{_render_assistant(message)}</div>"
    if isinstance(message, BashExecutionMessage):
        return f"<div class='entry bash'><div class='role'>shell</div>{_render_bash(message)}</div>"
    if isinstance(message, ToolResultMessage):
        css = "error" if message.is_error else ""
        label = f"tool result — {message.tool_name}" if message.tool_name else "tool result"
        return (
            f"<div class='entry tool'><div class='role'>{_esc(label)}</div>"
            f"<pre class='{css}'>{_esc(message.text())}</pre></div>"
        )
    return ""


def _render_entry(entry: SessionEntry) -> str:
    if entry.type == "message":
        message = entry.message
        return _render_message(message) if message is not None else ""
    if entry.type == "compaction":
        summary = entry.data.get("summary", "")
        return f"<div class='entry summary'><div class='role'>compaction summary</div><pre>{_esc(summary)}</pre></div>"
    if entry.type == "model_change":
        model = f"{entry.data.get('provider', '')}/{entry.data.get('modelId', '')}"
        return f"<p class='meta'>model → {_esc(model)}</p>"
    if entry.type == "thinking_level_change":
        return f"<p class='meta'>thinking level → {_esc(entry.data.get('thinkingLevel', ''))}</p>"
    if entry.type == "session_label":
        return f"<p class='meta'>session labeled “{_esc(entry.data.get('label', ''))}”</p>"
    return ""


def _render_html(session: SessionManager) -> str:
    model_changes = [e for e in session.path_entries() if e.type == "model_change"]
    model_label = (
        f"{model_changes[-1].data.get('provider', '')}/{model_changes[-1].data.get('modelId', '')}"
        if model_changes
        else "(unknown)"
    )
    exported = time.strftime("%Y-%m-%d %H:%M:%S")
    entries = "\n".join(_render_entry(entry) for entry in session.path_entries() if entry.type != "session")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{APP_NAME} session {_esc(session.session_id[:8])}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
<h1>{APP_NAME} session transcript</h1>
<dl>
<dt>id</dt><dd>{_esc(session.session_id)}</dd>
<dt>file</dt><dd>{_esc(session.file_path or "(in-memory)")}</dd>
<dt>cwd</dt><dd>{_esc(session.cwd)}</dd>
<dt>model</dt><dd>{_esc(model_label)}</dd>
<dt>exported</dt><dd>{_esc(exported)}</dd>
</dl>
</header>
<main>
{entries}
</main>
</body>
</html>
"""


def export_session_to_html(session: SessionManager, output_path: str | None = None, cwd: str | None = None) -> str:
    """Write the current branch as a static HTML transcript; returns the path."""
    base = Path(session.file_path).stem if session.file_path else session.session_id
    path = _resolve_output(output_path, f"{APP_NAME}-session-{base}.html", cwd)
    path.write_text(_render_html(session), encoding="utf-8")
    return str(path)
