"""Interactive terminal UI: prompt loop, slash commands and session controls."""

import contextlib
import difflib
import io
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from html import escape as html_escape
from typing import Any

from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from tynex.agent.permissions import PermissionRequest, mode_description
from tynex.ai.models import format_model
from tynex.coding import session_commands, signin
from tynex.coding.agent_session import AgentSession
from tynex.coding.settings import input_history_path
from tynex.coding.skills import expand_template
from tynex.coding.tools.shell import shell_prompt
from tynex.tui.docked import DockedPrompt
from tynex.tui.prompt import ALIASES, COMMANDS, PROMPT_STYLE, TynexCompleter, _prompt_key_bindings
from tynex.tui.renderer import Renderer, footer_text, running_text
from tynex.tui.select import select_option
from tynex.tui.transcript import Transcript

try:  # prompt_toolkit's win32 module asserts the platform, so this works on Windows only
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError as _NoConsoleScreenBufferError
except Exception:  # pragma: no cover - non-Windows

    class _NoConsoleScreenBufferError(Exception):
        """Placeholder; only raised on Windows without a real console."""


APPROVAL_SENTINEL = "\x00tynex:approval"  # returned by prompt() when an approval pulled it out
APPROVAL_QUESTION = "allow? y=once · a=session · n=no → "
PLAN_SENTINEL = "\x00tynex:plan"  # returned by prompt() when a submitted plan needs a decision
PLAN_QUESTION = "execute this plan? y=execute · n=keep planning → "


def _approval_outcome(answer: str) -> str:
    """Map a typed approval answer to the permission gate's outcome."""
    answer = answer.strip().lower()
    if answer in ("y", "yes"):
        return "allow"
    if answer in ("a", "always", "s", "session"):
        return "session"
    return "deny"


@dataclass
class _ApprovalEntry:
    """One pending approval, answered by the main (prompt) thread."""

    request: PermissionRequest
    done: threading.Event = field(default_factory=threading.Event)
    result: str | None = None


class InteractiveApp:
    def __init__(self, session: AgentSession, console: Console | None = None, *, terminal: bool | None = None) -> None:
        self.session = session
        # Agent output and command output both become transcript blocks, so the
        # user can scroll them and expand tool blocks (see tui/transcript.py).
        self.transcript = Transcript()
        self.console = console or Console(
            file=_TranscriptWriter(self.transcript),
            highlight=False,
            force_terminal=bool(terminal),
            # Logical lines stay whole: the transcript wraps them at its own
            # width, so a terminal resize reflows instead of leaving hard breaks.
            soft_wrap=True,
        )
        self.renderer = Renderer(session, self.transcript)
        self.transcript.on_change = self._refresh
        self._input: DockedPrompt | None = None
        # The agent runs in a background thread so the prompt stays usable.
        self._run_thread: threading.Thread | None = None
        # One run at a time: prompts submitted while a run is active wait here
        # and are started by the finishing run. ``_run_finished`` closes the
        # window where that thread is still alive but its work is over.
        self._run_lock = threading.Lock()
        self._queued_prompts: list[tuple[str, bool]] = []
        self._run_finished = threading.Event()
        self._run_finished.set()
        self._closing = False
        # A command line typed during a run, restored to the editor.
        self._prefill = ""
        # (used, window) tokens for the footer context meter; refreshed on turns.
        self._context: tuple[int, int] | None = None
        # Tool approvals requested by the agent thread, answered on the prompt thread.
        self._approvals: queue.Queue[_ApprovalEntry] = queue.Queue()
        # Plans submitted for approval, answered the same way.
        self._plans: queue.Queue[dict[str, Any]] = queue.Queue()
        session.permission_gate.on_ask = self._request_approval

    @property
    def input(self) -> DockedPrompt:
        """The docked prompt, created lazily (needs a real console only when used)."""
        if self._input is None:
            # NB: ``enable_history_search`` must stay off: prompt_toolkit turns
            # ``complete_while_typing`` off while it is on, which leaves the
            # completion menu empty (Up/Down still walk the history normally).
            self._input = DockedPrompt(
                self.transcript,
                history=FileHistory(str(input_history_path())),
                style=PROMPT_STYLE,
                completer=TynexCompleter(self.session),
                auto_suggest=AutoSuggestFromHistory(),
                placeholder=[("class:placeholder", "Ask anything · /help · @files · !shell")],
                status=self._status,
                toolbar=self._toolbar,
                on_expand=self._expand_last_tool,
                key_bindings=_prompt_key_bindings(interrupt=self._interrupt_run, is_busy=self._busy),
            )
        return self._input

    def _ask(self, text: str, *, password: bool = False) -> str:
        """Ask one line of input (the docked prompt resets the mask itself)."""
        return self.input.prompt(text, is_password=password)

    def _expand_last_tool(self) -> None:
        block = self.transcript.last_tool()
        if block is not None:
            self.transcript.toggle(block)
            self._refresh()

    def _select(self, title: str, options: list[str], *, selected: int = 0) -> int | None:
        if selected:
            return select_option(title, options, selected=selected)
        return select_option(title, options)

    # -- banner / status ------------------------------------------------------
    def _banner(self) -> None:
        stats = self.session.stats()
        session_label = stats["sessionId"][:8] if stats.get("sessionId") else "none"
        model_label = stats["model"] + (" — sign in with /login" if self.session.model is None else "")
        mode = self.session.permission_gate.mode
        plan_hint = " · /approve executes the plan" if mode == "plan" else ""
        self.console.print(
            f"\n[bold]Tynex[/] [dim]— minimal coding agent harness (Python)[/]\n"
            f"[dim]model [bold]{model_label}[/] · session {session_label} · cwd {self.session.cwd}[/]\n"
            f"[dim]permissions [bold]{mode}[/] — {mode_description(mode)}{plan_hint} · /permissions to switch[/]\n"
            "[dim]Type /help for commands. Esc interrupts a run; messages typed while it runs are queued. "
            "Ctrl+O expands a tool block. Ctrl+D exits.[/]\n"
        )

    def _toolbar(self) -> HTML:
        text = html_escape(footer_text(self.session, context=self._context), quote=False)
        return HTML(f"<style fg='ansibrightblack'> {text} </style>")

    def _prompt(self) -> HTML:
        return HTML("❯ ")  # noqa: RUF001 — intentional prompt glyph

    def _status(self) -> HTML:
        """The line above the input: what the run is doing, blank when idle."""
        if not self._busy():
            return HTML("")
        status = html_escape(running_text(self.session, self.renderer.activity()), quote=False)
        return HTML(f"<style fg='ansibrightblack'>{status}</style>")

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
    def _return_to_prompt(self, sentinel: str) -> None:
        """Hand control back to the prompt thread so it can ask the user something."""
        app = getattr(self._input, "app", None) if self._input is not None else None
        if app is not None and app.is_running:
            with contextlib.suppress(Exception):
                app.loop.call_soon_threadsafe(lambda live=app: live.exit(result=sentinel))

    def _request_approval(self, request: PermissionRequest) -> str:
        """Called from the agent thread; blocks until the prompt thread answers."""
        entry = _ApprovalEntry(request)
        self._approvals.put(entry)
        deadline = time.monotonic() + 5.0
        while not entry.done.is_set():
            app = getattr(self._input, "app", None) if self._input is not None else None
            if app is not None and app.is_running:
                self._return_to_prompt(APPROVAL_SENTINEL)
                break
            if time.monotonic() >= deadline:
                # No live prompt (piped input, startup): answer it here.
                try:
                    entry.result = self._approval_from_stdin(request)
                finally:
                    entry.done.set()
                break
            time.sleep(0.05)
        entry.done.wait()
        return entry.result or "deny"

    def _drain_plans(self) -> None:
        """Print submitted plans and ask for a decision; prompt thread only."""
        while True:
            try:
                event = self._plans.get_nowait()
            except queue.Empty:
                return
            plan = str(event.get("plan") or "").strip()
            pending = self.session.pending_plan
            if not plan or pending is None or pending.get("id") != event.get("planId"):
                continue  # stale: the plan was already approved or rejected
            self.console.print(Panel(Markdown(plan), title="plan ready", border_style="green"))
            if not self.console.is_terminal:
                self.console.print("[dim]Plan recorded. Run /approve to accept it, or keep planning.[/]")
                continue
            try:
                answer = self._ask(PLAN_QUESTION)
            except (EOFError, KeyboardInterrupt):
                answer = ""
            try:
                if answer.strip().lower() in ("y", "yes"):
                    self.session.approve_plan()
                    self.console.print("[green]Plan approved — switching to accept-edits.[/]")
                    # The planning turn may still be finishing; ``_submit``
                    # queues this until that run has released the session.
                    self._submit("Execute the approved plan.")
                else:
                    self.session.reject_plan()
                    self.console.print("[dim]Still in plan mode — keep refining, or send a new prompt.[/]")
            except ValueError as exc:  # the decision was already recorded
                self.console.print(f"[yellow]{exc}[/]")

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

    def _print_approval(self, request: PermissionRequest) -> None:
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

    def _prompt_approval(self, request: PermissionRequest) -> str:
        """Ask on the input line, so y/a/n use the editor the user is already in."""
        self._print_approval(request)
        try:
            return _approval_outcome(self._ask(APPROVAL_QUESTION))
        except (EOFError, KeyboardInterrupt):
            return "deny"

    def _approval_from_stdin(self, request: PermissionRequest) -> str:
        """Ask without a prompt session: piped input, or before the first prompt."""
        self._print_approval(request)
        try:
            return _approval_outcome(self.console.input(f"[yellow]{APPROVAL_QUESTION}[/]"))
        except (EOFError, KeyboardInterrupt):
            return "deny"

    # -- main loop --------------------------------------------------------------
    def _busy(self) -> bool:
        return self._run_thread is not None and self._run_thread.is_alive()

    def _refresh(self) -> None:
        # Called from the agent thread too; ``invalidate`` is thread safe.
        app = getattr(self._input, "app", None) if self._input is not None else None
        if app is not None:
            with contextlib.suppress(Exception):  # the loop may be closing
                app.invalidate()

    def run(self, initial_message: str | None = None) -> int:
        self._closing = False
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
            self._refresh()
            while True:
                self._drain_approvals()
                self._drain_plans()
                try:
                    line = self.input.prompt(self._prompt(), default=self._prefill)
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
                if line is None:  # a programmatic exit without a result ends the loop
                    break
                self._prefill = ""
                if line in (APPROVAL_SENTINEL, PLAN_SENTINEL):
                    continue
                # The typed line is already on screen; don't render it again.
                if line.strip() and self._handle_line(line.strip(), echo_user=False) is False:
                    break
        finally:
            self._closing = True
            self._shutdown_run()
            self._drain_approvals()
            self._drain_plans()
            self._set_title(None)
            self._print_bye()
        return 0

    def _print_bye(self) -> None:
        """Leave the exit message on the real terminal, not only in the transcript.

        The console writes into the transcript, and a transcript change only
        repaints while the prompt application is running. Nothing runs after
        this point, so the transcript copy would never be drawn.
        """
        self.console.print("[dim]bye[/]")
        app = getattr(self._input, "app", None) if self._input is not None else None
        output = getattr(app, "output", None)
        if output is None:
            return
        with contextlib.suppress(Exception):
            output.write("bye\n")
            output.flush()

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
        with self._run_lock:
            if self._closing:
                return
            self._queued_prompts.append((text, echo_user))
            self._start_next_run()

    def _start_next_run(self) -> None:
        """Start the next queued prompt; the caller holds ``_run_lock``."""
        if self._closing:
            self._queued_prompts.clear()
            return
        thread = self._run_thread
        if thread is not None and thread.is_alive() and not self._run_finished.is_set():
            return
        if not self._queued_prompts:
            return
        text, echo_user = self._queued_prompts.pop(0)
        self._run_finished.clear()
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
            if kind == "plan_ready":
                self._plans.put(event)
                self._return_to_prompt(PLAN_SENTINEL)
            if kind in ("agent_end", "compaction", "plan_ready"):
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
            with self._run_lock:
                self._run_finished.set()
                self._start_next_run()

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
            "compact": lambda a: self._refresh_context(session_commands.compact(self.session, self.console, a)),
            "new": lambda _arg: self._refresh_context(session_commands.new_session(self.session, self.console)),
            "resume": lambda a: self._refresh_all(
                session_commands.resume(self.session, self.console, a, self._ask, self._select)
            ),
            "export": lambda a: session_commands.export_session(self.session, self.console, a),
            "name": lambda a: session_commands.rename_session(self.session, self.console, a),
            "session": lambda _arg: session_commands.session_info(self.session, self.console),
            "permissions": lambda a: session_commands.permissions(self.session, self.console, a),
            "plan": lambda _arg: self._enter_plan_mode(),
            "approve": lambda _arg: self._approve_plan(),
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
        known = [*handlers, *(f"skill:{skill.name}" for skill in self.session.skills), *self.session.prompt_templates]
        suggestion = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
        hint = f" Did you mean /{suggestion[0]}?" if suggestion else ""
        self.console.print(f"[red]Unknown command /{name}.[/] Try /help.{hint}")
        return None

    def _refresh_context(self, _result: Any = None) -> None:
        self._update_context()

    def _refresh_all(self, _result: Any = None) -> None:
        self._update_context()
        self._sync_title()

    def _undo(self, arg: str) -> None:
        session_commands.undo_session(self.session, self.console, arg)
        self._update_context()

    def _enter_plan_mode(self) -> None:
        if self.session.permission_gate.mode == "plan":
            self.console.print("[dim]Already in plan mode — the agent submits a plan for /approve.[/]")
            return
        self.session.set_permission_mode("plan")
        self.console.print(
            "[green]Plan mode[/] — read-only. The agent investigates, then calls submit_plan; /approve executes it."
        )
        self._refresh_all()

    def _approve_plan(self) -> None:
        try:
            self.session.approve_plan()
        except ValueError as exc:
            self.console.print(f"[yellow]{exc}[/]")
            return
        self.console.print("[green]Plan approved[/] — permission mode accept-edits. Send a message to execute it.")
        self._refresh_all()

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
        "@path              reference a file (paths complete after @)",
        "Tab                complete /commands, /skill:<name> and @files",
        "Alt+Enter          insert a newline in the prompt",
        "Ctrl+O             expand or collapse the last tool block",
        "PgUp / PgDn        scroll the transcript (mouse wheel and clicks work too)",
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


class _TranscriptWriter(io.TextIOBase):
    """Pipe Rich output (command output, panels, stray logs) into the transcript."""

    def __init__(self, transcript: Transcript) -> None:
        self._transcript = transcript

    @property
    def encoding(self) -> str:  # Rich probes this when the console is built
        return "utf-8"

    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        self._transcript.write(text)
        return len(text)

    def flush(self) -> None:
        self._transcript.flush()


def run_interactive(session: AgentSession, initial_message: str | None = None) -> int:
    try:
        if not sys.stdout.isatty():
            return InteractiveApp(session).run(initial_message)
        app = InteractiveApp(session, terminal=True)
        # Keep stray prints (extension hooks, logging) from corrupting the prompt.
        with contextlib.redirect_stderr(_TranscriptWriter(app.transcript)):
            return app.run(initial_message)
    except _NoConsoleScreenBufferError:
        Console().print(
            "[red]Interactive mode needs a real terminal.[/] "
            'Use [bold]tynex -p "<prompt>"[/] for a one-shot run, '
            "or [bold]tynex --json[/] for scripting."
        )
        return 1
