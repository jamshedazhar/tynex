"""Interactive terminal UI: prompt loop, slash commands and session controls."""

import contextlib
import io
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from html import escape as html_escape
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app_session
from prompt_toolkit.application.run_in_terminal import run_in_terminal
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.output import Output
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from tynex.agent.permissions import PermissionRequest
from tynex.ai.models import format_model
from tynex.coding import session_commands, signin
from tynex.coding.agent_session import AgentSession
from tynex.coding.settings import input_history_path
from tynex.coding.skills import expand_template
from tynex.coding.tools.shell import shell_prompt
from tynex.tui.prompt import (
    ALIASES,
    COMMANDS,
    COMPLETION_MENU_ROWS,
    PROMPT_STYLE,
    TynexCompleter,
    _cap_completion_menu,
    _prompt_key_bindings,
)
from tynex.tui.renderer import Renderer, footer_text, running_text
from tynex.tui.select import select_option

try:  # prompt_toolkit's win32 module asserts the platform, so this works on Windows only
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError as _NoConsoleScreenBufferError
except Exception:  # pragma: no cover - non-Windows

    class _NoConsoleScreenBufferError(Exception):
        """Placeholder; only raised on Windows without a real console."""


APPROVAL_SENTINEL = "\x00tynex:approval"  # returned by prompt() when an approval pulled it out


@dataclass
class _ApprovalEntry:
    """One pending approval, answered by the main (prompt) thread."""

    request: PermissionRequest
    done: threading.Event = field(default_factory=threading.Event)
    result: str | None = None


class InteractiveApp:
    def __init__(self, session: AgentSession, console: Console | None = None) -> None:
        self.session = session
        self.console = console or Console()
        # The prompt owns the terminal while a run streams, so the renderer
        # prints completed sections above it instead of using a live region.
        self.renderer = Renderer(session, self.console)
        self._input: PromptSession | None = None
        # The agent runs in a background thread so the prompt stays usable.
        self._run_thread: threading.Thread | None = None
        # Text put back into the editor by /fork.
        self._prefill = ""
        # (used, window) tokens for the footer context meter; refreshed on turns.
        self._context: tuple[int, int] | None = None
        # Tool approvals requested by the agent thread, answered on the prompt thread.
        self._approvals: queue.Queue[_ApprovalEntry] = queue.Queue()
        session.permission_gate.on_ask = self._request_approval

    @property
    def input(self) -> PromptSession:
        """The prompt session, created lazily (needs a real console only when used)."""
        if self._input is None:
            _cap_completion_menu(COMPLETION_MENU_ROWS)
            # NB: ``enable_history_search`` must stay off: prompt_toolkit's
            # PromptSession turns ``complete_while_typing`` off while it is on,
            # which leaves the completion menu empty (Up/Down still walk the
            # history normally without it).
            self._input = PromptSession(
                history=FileHistory(str(input_history_path())),
                style=PROMPT_STYLE,
                completer=TynexCompleter(self.session),
                complete_while_typing=True,
                auto_suggest=AutoSuggestFromHistory(),
                placeholder=[("class:placeholder", "Ask anything · /help · @files · !shell")],
                multiline=True,
                key_bindings=_prompt_key_bindings(interrupt=self._interrupt_run, is_busy=self._busy),
            )
        return self._input

    def _ask(self, text: str, *, password: bool = False) -> str:
        """Ask one line of input.

        ``PromptSession.prompt(is_password=True)`` turns the mask on for the
        whole session and never clears it, so a masked question (an API key)
        would leave every later prompt showing asterisks. Reset the flag once
        the question is answered — also when it is cancelled.
        """
        prompt = self.input
        try:
            return prompt.prompt(text, is_password=password)
        finally:
            prompt.is_password = False

    def _select(self, title: str, options: list[str], *, selected: int = 0) -> int | None:
        if selected:
            return select_option(title, options, selected=selected)
        return select_option(title, options)

    # -- banner / status ------------------------------------------------------
    def _banner(self) -> None:
        stats = self.session.stats()
        session_label = stats["sessionId"][:8] if stats.get("sessionId") else "none"
        model_label = stats["model"] + (" — sign in with /login" if self.session.model is None else "")
        self.console.print(
            f"\n[bold]Tynex[/] [dim]— minimal coding agent harness (Python)[/]\n"
            f"[dim]model [bold]{model_label}[/] · session {session_label} · cwd {self.session.cwd}[/]\n"
            "[dim]Type /help for commands. Esc interrupts a run; messages typed while it runs are queued. "
            "Ctrl+D exits.[/]\n"
        )

    def _toolbar(self) -> HTML:
        text = html_escape(footer_text(self.session, context=self._context), quote=False)
        return HTML(f"<style fg='ansibrightblack'> {text} </style>")

    def _prompt(self) -> HTML:
        if not self._busy():
            return HTML("❯ ")  # noqa: RUF001 — intentional prompt glyph
        status = html_escape(running_text(self.session, self.renderer.activity()), quote=False)
        return HTML(f"<style fg='ansibrightblack'>{status}</style>\n❯ ")  # noqa: RUF001

    def _title_text(self) -> str:
        return f"tynex · {format_model(self.session.model)}" if self.session.model is not None else "tynex"

    def _set_title(self, text: str | None = None) -> None:
        app = getattr(self._input, "app", None) if self._input is not None else None
        output = getattr(app, "output", None)
        if output is None:
            return

        def apply() -> None:
            with contextlib.suppress(Exception):
                output.clear_title() if text is None else output.set_title(text)

        loop = app.loop if getattr(app, "is_running", False) else None
        if loop is None:  # not running: safe to write directly
            apply()
        else:  # called from the agent thread while the prompt is live
            loop.call_soon_threadsafe(apply)

    def _sync_title(self) -> None:
        self._set_title(self._title_text())

    def _update_context(self) -> None:
        model = self.session.model
        self._context = None
        if model is not None:
            with contextlib.suppress(Exception):
                self._context = (self.session.estimate_context_tokens(), model.context_window)

    # -- approvals ------------------------------------------------------------
    def _request_approval(self, request: PermissionRequest) -> str:
        """Called from the agent thread; blocks until the prompt thread answers."""
        entry = _ApprovalEntry(request)
        self._approvals.put(entry)
        deadline = time.monotonic() + 5.0
        while not entry.done.is_set():
            app = getattr(self._input, "app", None) if self._input is not None else None
            if app is not None and app.is_running:
                with contextlib.suppress(Exception):
                    app.loop.call_soon_threadsafe(lambda live=app: live.exit(result=APPROVAL_SENTINEL))
                break
            if time.monotonic() >= deadline:
                # No live prompt (piped input, startup): answer it here.
                try:
                    entry.result = self._prompt_approval(request)
                finally:
                    entry.done.set()
                break
            time.sleep(0.05)
        entry.done.wait()
        return entry.result or "deny"

    def _drain_approvals(self) -> None:
        """Answer every pending approval; runs on the prompt thread only."""
        while True:
            try:
                entry = self._approvals.get_nowait()
            except queue.Empty:
                return
            if entry.done.is_set():
                continue
            try:
                entry.result = self._prompt_approval(entry.request)
            finally:
                entry.done.set()

    def _prompt_approval(self, request: PermissionRequest) -> str:
        kind_label = {
            "shell": "shell command",
            "edit": "file change",
            "mcp": "MCP tool call",
            "web": "web request",
        }.get(request.kind, "tool call")
        lines = [f"[bold]{html_escape(request.summary())}[/]"]
        if request.kind == "mcp" and request.args:
            lines.append(f"[dim]{html_escape(str(request.args)[:400])}[/]")
        lines.append(f"[dim]{kind_label}[/]")
        self.console.print(Panel("\n".join(lines), title="approval needed", border_style="yellow"))
        try:
            answer = (
                self.console.input("[yellow]allow? [bold]y[/]=once · [bold]a[/]=session · [bold]n[/]=no[/] → ")
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            return "deny"
        if answer in ("y", "yes"):
            return "allow"
        if answer in ("a", "always", "s", "session"):
            return "session"
        return "deny"

    # -- main loop --------------------------------------------------------------
    def _busy(self) -> bool:
        return self._run_thread is not None and self._run_thread.is_alive()

    def _refresh(self) -> None:
        app = getattr(self._input, "app", None) if self._input is not None else None
        if app is not None:
            app.invalidate()

    def run(self, initial_message: str | None = None) -> int:
        self._banner()
        message = (initial_message or "").strip()
        try:
            # CLI messages may be slash commands, ``!`` bash lines or prompts.
            if message and self._handle_line(message) is False:
                return 0
            if not self._busy():  # don't open sign-in on top of a background run
                self._maybe_ask_login()
            self._update_context()
            self.input  # noqa: B018 — create the session early so the title can be set
            self._sync_title()
            while True:
                self._drain_approvals()
                try:
                    line = self.input.prompt(self._prompt, bottom_toolbar=self._toolbar, default=self._prefill)
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
                self._prefill = ""
                if line == APPROVAL_SENTINEL:
                    continue
                # The typed line is already on screen; don't render it again.
                if line.strip() and self._handle_line(line.strip(), echo_user=False) is False:
                    break
        finally:
            self._shutdown_run()
            self._drain_approvals()
            self._set_title(None)
            self.console.print("[dim]bye[/]")
        return 0

    def _shutdown_run(self) -> None:
        thread = self._run_thread
        if thread is not None and thread.is_alive():
            self.session.abort()
            thread.join(timeout=5)

    def _handle_line(self, line: str, *, echo_user: bool = True) -> bool | None:
        if self._busy():
            return self._queue_line(line)
        if line.startswith("!") and len(line) > 1:
            # ``!`` runs and adds the output to the conversation; ``!!`` keeps it
            # out of the model context.
            excluded = line.startswith("!!")
            command = line[2:].strip() if excluded else line[1:].strip()
            if command:
                self._run_bash(command, exclude_from_context=excluded)
                return None
        if line.startswith("/"):
            result = self._handle_command(line)
            if result is False:
                return False
            if isinstance(result, str):
                self._submit(result)
            return None
        self._submit(line, echo_user=echo_user)
        return None

    def _queue_line(self, line: str) -> bool | None:
        if line in ("/quit", "/exit", "/q"):
            return False
        if line.startswith(("/", "!")):
            # ``prompt()`` resets the buffer to ``default`` on every call, so a
            # line inserted into the buffer here would be wiped before the next
            # prompt: keep it in the prefill the loop passes as ``default``.
            self._prefill = line
            self.console.print("[dim]busy — press esc; the line is back in the editor[/]")
        else:
            self.session.steer(line)
            self.console.print("[dim]queued[/]")
            self._refresh()
        return None

    def _interrupt_run(self) -> None:
        signal = self.session.signal if self._busy() else None
        if signal is not None and not signal.aborted:
            self.session.abort()
            self.console.print("[yellow]Interrupting…[/]")

    # -- agent run ----------------------------------------------------------------
    def _submit(self, text: str, *, echo_user: bool = True) -> None:
        if self.session.model is None:
            self.console.print("[yellow]No model selected[/] — sign in with [bold]/login[/] or pick one with /model.")
            return
        self._run_thread = threading.Thread(
            target=self._run_prompt, args=(text,), kwargs={"echo_user": echo_user}, daemon=True
        )
        self._run_thread.start()
        self._refresh()  # show the running line above the input right away

    def _run_prompt(self, text: str, *, echo_user: bool = True) -> None:
        self.renderer.echo_user = echo_user
        self._set_title(f"● running · {format_model(self.session.model)}" if self.session.model else "● running")
        last_refresh = 0.0

        def emit(event: dict) -> None:
            nonlocal last_refresh
            self.renderer.handle(event)
            kind = event.get("type")
            if kind in ("agent_end", "compaction"):
                self._update_context()
            now = time.monotonic()
            # Deltas only change the footer activity; throttle those repaints.
            if kind != "message_update" or now - last_refresh >= 0.1:
                last_refresh = now
                self._refresh()

        try:
            self.session.prompt(text, emit=emit)
        except KeyboardInterrupt:
            self.session.abort()
            self.session.finalize_aborted_turn(emit=emit)
            self.console.print("[red]Interrupted.[/]")
        except Exception as exc:
            self.console.print(f"[red]{exc}[/]")
        finally:
            self.renderer.echo_user = True
            self._sync_title()
            self._refresh()

    def _run_bash(self, command: str, *, exclude_from_context: bool = False) -> None:
        self.console.print(f"[bold green]{shell_prompt}[/] [bold]{command}[/]")
        try:
            message = self.session.execute_bash(command, exclude_from_context=exclude_from_context)
        except Exception as exc:
            self.console.print(f"[red]{exc}[/]")
            return
        if message.output:
            self.console.print(Text(message.output))
        status: list[str] = []
        if message.cancelled:
            status.append("command cancelled")
        elif message.exit_code is not None and message.exit_code != 0:
            status.append(f"command exited with code {message.exit_code}")
        if message.truncated and message.full_output_path:
            status.append(f"output truncated — full output: {message.full_output_path}")
        if message.exclude_from_context:
            status.append("not sent to the model")
        if status:
            style = "red" if (message.cancelled or (message.exit_code or 0) != 0) else "dim"
            self.console.print(f"[{style}]{' · '.join(status)}[/]")

    # -- commands ---------------------------------------------------------------------
    def _handle_command(self, line: str) -> bool | str | None:
        name, _, arg = line[1:].partition(" ")
        if not name:
            return None
        name, arg = ALIASES.get(name, name), arg.strip()
        handlers: dict[str, Callable[[str], Any]] = {
            "help": lambda _arg: self._print_help(),
            "quit": lambda _arg: False,
            "model": self._cmd_model,
            "thinking": lambda a: session_commands.thinking(self.session, self.console, a, self._select),
            "tree": lambda a: self._refresh_context(session_commands.tree(self.session, self.console, a)),
            "compact": lambda a: self._refresh_context(session_commands.compact(self.session, self.console, a)),
            "new": lambda _arg: self._refresh_context(session_commands.new_session(self.session, self.console)),
            "resume": lambda a: self._refresh_all(
                session_commands.resume(self.session, self.console, a, self._ask, self._select)
            ),
            "fork": self._fork,
            "clone": lambda _arg: session_commands.clone(self.session, self.console),
            "export": lambda a: session_commands.export_session(self.session, self.console, a),
            "name": lambda a: session_commands.rename_session(self.session, self.console, a),
            "session": lambda _arg: session_commands.session_info(self.session, self.console),
            "permissions": lambda a: session_commands.permissions(self.session, self.console, a),
            "budget": lambda _arg: session_commands.budget(self.session, self.console),
            "undo": lambda a: self._undo(a),
            "mcp": lambda _arg: session_commands.mcp_servers(self.session, self.console),
            "login": self._login,
            "logout": lambda a: signin.logout(self.session, a, self.console, self._select),
            "skills": lambda _arg: session_commands.list_skills(self.session, self.console),
            "cwd": lambda _arg: session_commands.print_cwd(self.session, self.console),
        }
        if (handler := handlers.get(name)) is not None:
            return handler(arg)
        # Skills: /skill:<name> loads SKILL.md into the turn; /skills:<name> is the legacy alias.
        if name.startswith(("skill:", "skills:")):
            return session_commands.run_skill(self.session, self.console, name.split(":", 1)[1], arg)
        ext = self.session.extensions.commands.get(name)
        if ext:
            try:
                ext["handler"](arg)
            except Exception as exc:
                self.console.print(f"[red]Extension command failed: {exc}[/]")
            return None
        template = self.session.prompt_templates.get(name)
        if template:
            return expand_template(template, arg)
        self.console.print(f"[red]Unknown command /{name}. Try /help.[/]")
        return None

    def _refresh_context(self, _result: Any = None) -> None:
        self._update_context()

    def _refresh_all(self, _result: Any = None) -> None:
        self._update_context()
        self._sync_title()

    def _undo(self, arg: str) -> None:
        session_commands.undo_session(self.session, self.console, arg)
        self._update_context()

    def _list_models(self) -> None:
        session_commands.list_models(self.session, self.console)

    def _cmd_model(self, arg: str) -> None:
        if not arg:
            return self._pick_model()
        try:
            model = self.session.set_model_by_pattern(arg)
            self.console.print(f"[green]Switched to {format_model(model)}[/]")
            self._refresh_all()
        except ValueError as exc:
            self.console.print(f"[red]{exc}[/]")

    def _pick_model(self) -> None:
        session_commands.pick_model(self.session, self.console, self._select)
        self._refresh_all()

    def _fork(self, arg: str) -> None:
        selected = session_commands.fork(self.session, self.console, arg, self._select)
        if selected is not None:
            self._prefill = selected
        self._refresh_all()

    def _login(self, arg: str) -> None:
        if not arg and self.session.model is None:
            return self._suggest_login()
        signin.login(self.session, arg, self.console, self._ask, self._select, self._switch_to_provider_if_unusable)
        self._refresh_all()

    def _suggest_login(self) -> None:
        signin.suggest_login(self.session, self.console, self._ask, self._select, self._switch_to_provider_if_unusable)

    def _login_copilot(self) -> None:
        signin.login_copilot(self.session, self.console, self._ask, self._switch_to_provider_if_unusable)
        self._refresh_all()

    def _maybe_ask_login(self) -> None:
        signin.maybe_ask_login(
            self.session, self.console, self._ask, self._select, self._switch_to_provider_if_unusable
        )

    def _switch_to_provider_if_unusable(self, provider: str):
        return signin.switch_to_provider_if_unusable(self.session, provider, self.console)

    _SHORTCUTS = (
        "!command           run a shell command (output joins the conversation)",
        "!!command          run a shell command (output stays out of the model context)",
        "/<template> [args] run a prompt template from .tynex/prompts/",
        "@path              reference a file (paths complete while typing)",
        "Tab                complete /commands, /skill:<name> and @files",
        "Alt+Enter          insert a newline in the prompt",
        "Enter              with a menu open, fill in the row first; Enter again submits",
        "Esc                interrupt a run; messages typed while it runs are queued",
        "y / a / n          answer an approval request (once / allow this session / deny)",
    )

    def _print_help(self) -> None:
        rows = [f"  /{name} {spec.usage}".rstrip().ljust(24) + spec.help for name, spec in COMMANDS.items()]
        rows.append("  /skill:<name>".ljust(24) + "load a skill's instructions into the turn")
        aliases = " · ".join(f"/{alias} = /{target}" for alias, target in ALIASES.items())
        self.console.print(
            Panel.fit(
                "\n".join(
                    [
                        "[bold]Commands[/]",
                        *rows,
                        "",
                        "[bold]Shortcuts[/]",
                        *(f"  {line}" for line in self._SHORTCUTS),
                        "",
                        f"[dim]Aliases: {aliases}[/]",
                    ]
                ),
                title="tynex",
                border_style="dim",
            )
        )
        if self.session.prompt_templates:
            self.console.print(f"[dim]Prompt templates: {', '.join(sorted(self.session.prompt_templates))}[/]")
        if self.session.skills:
            self.console.print(f"[dim]Skills: {', '.join(skill.name for skill in self.session.skills)}[/]")
        if self.session.loaded_extensions:
            self.console.print(f"[dim]Extensions: {', '.join(self.session.loaded_extensions)}[/]")


class _PromptWriter(io.TextIOBase):
    """Write Rich output above the prompt from the UI or the agent thread."""

    def __init__(self) -> None:
        self._session = get_app_session()

    @property
    def encoding(self) -> str:  # Rich probes this when the console is built
        return self._session.output.encoding()

    def fileno(self) -> int:
        return self._session.output.fileno()

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if text:
            app = self._session.app
            loop = app.loop if app is not None and app.is_running else None
            if loop is None:
                self._write(self._session.output, text)
            else:
                try:
                    loop.call_soon_threadsafe(lambda: run_in_terminal(lambda: self._write(app.output, text)))
                except RuntimeError:  # the loop closed between the check and the call
                    self._write(self._session.output, text)
        return len(text)

    @staticmethod
    def _write(output: Output, text: str) -> None:
        output.write_raw(text)
        output.flush()


def run_interactive(session: AgentSession, initial_message: str | None = None) -> int:
    try:
        if not sys.stdout.isatty():
            return InteractiveApp(session).run(initial_message)
        writer = _PromptWriter()
        # Keep stray prints (extension hooks, logging) from corrupting the prompt.
        with contextlib.redirect_stderr(writer):
            return InteractiveApp(session, console=Console(file=writer, highlight=False)).run(initial_message)
    except _NoConsoleScreenBufferError:
        Console().print(
            "[red]Interactive mode needs a real terminal.[/] "
            'Use [bold]tynex -p "<prompt>"[/] for a one-shot run, '
            "or [bold]tynex --json[/] / [bold]tynex --rpc[/] for scripting."
        )
        return 1
