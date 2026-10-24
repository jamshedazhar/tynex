"""prompt_toolkit setup for the terminal UI."""

import os
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, NamedTuple

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from tynex.coding.tools.path_utils import resolve_path

if TYPE_CHECKING:
    from tynex.coding.agent_session import AgentSession

PROMPT_STYLE = Style.from_dict(
    {"prompt": "ansicyan bold", "bottom-toolbar": "fg:ansibrightblack", "placeholder": "fg:ansibrightblack"}
)


def _prompt_key_bindings(
    *, interrupt: Callable[[], None] | None = None, is_busy: Callable[[], bool] | None = None
) -> KeyBindings:
    from prompt_toolkit.filters import Condition, has_focus

    bindings = KeyBindings()

    @bindings.add("enter", filter=has_focus("DEFAULT_BUFFER"))
    def _submit(event: Any) -> None:
        buffer = event.current_buffer
        state = buffer.complete_state
        if state is not None and state.completions:
            completion = state.current_completion or state.completions[0]
            before = buffer.text
            buffer.apply_completion(completion)
            if buffer.text == before:
                buffer.validate_and_handle()
            return
        buffer.validate_and_handle()

    @bindings.add("escape", "enter", filter=has_focus("DEFAULT_BUFFER"))
    def _newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    if interrupt is not None and is_busy is not None:
        busy = Condition(is_busy)

        @bindings.add("escape", filter=busy, eager=True)
        def _escape(event: Any) -> None:
            del event
            interrupt()

        @bindings.add("c-c", filter=busy, eager=True)
        def _ctrl_c(event: Any) -> None:
            del event
            interrupt()

    return bindings


class TynexCompleter(Completer):
    """Complete commands/shortcuts with ``/`` or at line start, and paths with ``@`` or as plain words."""

    def __init__(self, session: "AgentSession") -> None:
        self.session = session

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        del complete_event
        before = document.text_before_cursor
        word = _word_before_cursor(before)
        if not word:
            return
        if word.startswith("/"):
            # Slash words complete anywhere in the message (``/`` lists all).
            yield from self._command_completions(word[1:], consumed=len(word))
        elif word.startswith("@"):
            # @ mentions: complete the path after the "@" (which stays put).
            yield from _path_completions(word[1:], self.session.cwd)
        else:
            # Keystroke completion: command/shortcut names at the start of a
            # line; paths anywhere (two characters or more to avoid noise).
            if before.rsplit("\n", 1)[-1].strip() == word:
                yield from self._command_completions(word, consumed=len(word), loose=True)
            if len(word) >= 2:
                yield from _path_completions(word, self.session.cwd)

    def _command_completions(self, prefix: str, *, consumed: int, loose: bool = False) -> Iterable[Completion]:
        prefix = prefix.lower()
        for name, help_text in _command_table(self.session).items():
            lower = name.lower()
            tail = lower.split(":", 1)[-1]
            if lower.startswith(prefix) or (loose and tail.startswith(prefix)):
                yield Completion("/" + name, start_position=-consumed, display="/" + name, display_meta=help_text)


def _command_table(session: "AgentSession") -> dict[str, str]:
    table = {name: spec.help for name, spec in COMMANDS.items()}
    for alias, target in ALIASES.items():
        table[alias] = f"alias of /{target}"
    for name, extension in session.extensions.commands.items():
        help_text = extension.get("help") or "extension command"
        table.setdefault(name, f"extension · {help_text}")
    for name, template in session.prompt_templates.items():
        description = template.description or "prompt template"
        table.setdefault(name, f"template · {description}")
    for skill in session.skills:
        table.setdefault(f"skill:{skill.name}", f"skill · {skill.description or 'SKILL.md'}")
    return table


def _word_before_cursor(text: str) -> str:
    index = len(text)
    while index > 0 and not text[index - 1].isspace():
        index -= 1
    return text[index:]


def _split_path(word: str) -> tuple[str, str]:
    index = max(word.rfind("/"), word.rfind("\\"))
    if index == -1:
        return "", word
    return word[:index], word[index + 1 :]


def _path_completions(word: str, cwd: str) -> Iterable[Completion]:
    parent, prefix = _split_path(word)
    try:
        base = resolve_path(parent or ".", cwd)
    except ValueError:
        return
    try:
        with os.scandir(base) as scan:
            entries = sorted(scan, key=lambda entry: (not entry.is_dir(), entry.name.lower()))
    except OSError:
        return
    for entry in entries:
        name = entry.name
        if name.startswith(".") and not prefix.startswith("."):
            continue  # hidden files only when asked for (leading ".")
        if not _matches_prefix(name, prefix):
            continue
        is_dir = entry.is_dir()
        text = name + "/" if is_dir else name
        yield Completion(text, start_position=-len(prefix), display=text, display_meta="dir" if is_dir else None)


def _matches_prefix(name: str, prefix: str) -> bool:
    if not prefix:
        return True
    if os.name == "nt":
        return name.lower().startswith(prefix.lower())
    return name.startswith(prefix)


class Command(NamedTuple):
    usage: str
    help: str


COMMANDS: dict[str, Command] = {
    "help": Command("", "show commands and shortcuts"),
    "model": Command("[pattern]", "switch model (no argument picks one from the current provider)"),
    "thinking": Command("[level]", "set the thinking level (no argument picks one)"),
    "compact": Command("[notes]", "summarize older messages now"),
    "new": Command("", "start a fresh session"),
    "resume": Command("[id]", "resume a past session"),
    "export": Command("[path]", "export the transcript (.jsonl branch, else HTML)"),
    "name": Command("<label>", "label the current session"),
    "session": Command("", "session statistics"),
    "permissions": Command("[mode] <name>", "show or switch the tool approval policy"),
    "plan": Command("", "switch to plan mode: investigate read-only, then submit a plan"),
    "approve": Command("", "approve the pending plan and switch to accept-edits"),
    "budget": Command("", "show token/cost budget and usage"),
    "undo": Command("[count]", "revert the files changed by the last tool calls"),
    "mcp": Command("", "list connected MCP servers and their tools"),
    "login": Command("[provider]", "sign in or store an API key"),
    "logout": Command("[provider]", "remove a stored credential"),
    "skills": Command("", "list available skills"),
    "cwd": Command("", "print the working directory"),
    "quit": Command("", "exit"),
}

ALIASES = {"exit": "quit", "q": "quit", "sessions": "resume", "stats": "session"}
