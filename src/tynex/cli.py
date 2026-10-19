"""Tynex command-line interface."""

import contextlib
import json
import os
import queue
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import click
except ModuleNotFoundError as exc:  # pragma: no cover - only without the CLI extra
    raise SystemExit(
        f"tynex: missing optional dependency '{exc.name or 'click'}' — install it with: pip install 'tynex[cli]'"
    ) from exc

from tynex import __version__
from tynex.agent.budget import parse_budget
from tynex.agent.permissions import PERMISSION_MODES
from tynex.ai.models import (
    default_model_for_provider,
    format_model,
    provider_api_key_env,
    providers,
    register_custom_providers,
)
from tynex.ai.providers.base import THINKING_LEVELS
from tynex.ai.types import AssistantMessage, content_to_dict, message_to_dict
from tynex.coding.agent_session import AgentSession
from tynex.coding.mcp_settings import builtin_catalog, builtin_status, runtime_path
from tynex.coding.model_catalog import ensure_models, resolve_model_retry
from tynex.coding.session import find_session, latest_session_file, list_sessions
from tynex.coding.settings import (
    load_settings,
    project_dir,
    read_settings_file,
    remove_credential,
    settings_path,
    store_api_key,
    write_settings_file,
)
from tynex.coding.signin import (
    NO_KEY_PROVIDERS,
    copilot_login,
    interactive_login,
    register_copilot_provider,
    signed_in_providers,
)
from tynex.tui.select import select_option

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
_EVENT_KEYS = (
    "toolCallId",
    "toolName",
    "isError",
    "durationMs",
    "reason",
    "summary",
    "tokensBefore",
    "messagesSummarized",
    "action",
    "origin",
    "tool",
    "subject",
    "args",
)


def run_click(call: Callable[[], Any]) -> int:
    try:
        result = call()
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        click.echo("Aborted!", err=True)
        return 1
    except KeyboardInterrupt:
        click.echo("Aborted!", err=True)
        return 130
    return result if isinstance(result, int) else 0


class CliError(click.ClickException):
    exit_code = 2

    def show(self, file=None) -> None:
        click.echo(f"tynex: {self.format_message()}", err=True)


def _settings() -> dict[str, Any]:
    settings = load_settings(os.getcwd())
    register_custom_providers(settings.get("providers") or {})
    register_copilot_provider(settings)
    return settings


def print_models(settings: dict[str, Any], provider: str | None = None) -> None:
    names = providers()
    if provider and provider not in names:
        raise CliError(f"Unknown provider {provider!r}. Known providers: {', '.join(names)}")
    for name in [provider] if provider else names:
        models, note = ensure_models(name, settings)
        click.echo(f"{name}:")
        if not models:
            click.echo(f"  ({note or f'no models — run `tynex login {name}`'})")
            continue
        for model in models:
            flags = " (reasoning)" if model.reasoning else ""
            click.echo(f"  {format_model(model)}  — {model.name}, {model.context_window:,} ctx{flags}")


def print_sessions(cwd: str) -> None:
    sessions = list_sessions(cwd)
    if not sessions:
        click.echo("No sessions for this directory.")
        return
    for info in sessions:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(info["modified"]))
        label = f" · {info['label']}" if info.get("label") else ""
        click.echo(f"{info['id'][:8]}  {stamp}  {info['messages']:>4} msgs{label}  {info['file']}")


def _cli_ask(text: str, *, password: bool = False) -> str:
    return click.prompt(
        text, default="", show_default=False, prompt_suffix="", hide_input=password and sys.stdin.isatty()
    )


def _cli_select(title: str, options: list[str]) -> int | None:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise CliError("interactive selection needs a terminal — pass the provider explicitly instead")
    return select_option(title, options)


def _report_default_model(provider: str, settings: dict[str, Any]) -> None:
    _models, note = ensure_models(provider, settings)
    model = default_model_for_provider(provider)
    if model is not None:
        click.echo(f"Tynex will start with {format_model(model)} unless overridden by --model or settings.")
    elif note:
        click.echo(note)


@click.command(
    "login",
    context_settings=CONTEXT_SETTINGS,
    help="Sign in: OAuth device flow, API key, or a custom OpenAI-compatible provider.",
)
@click.argument("provider", required=False)
def login_command(provider: str | None) -> int:
    settings = _settings()
    if provider is None:
        signed = interactive_login(ask=_cli_ask, select=_cli_select, echo=click.echo, settings=settings)
        if signed is None:
            click.echo("Nothing to do.")
            return 0
        _report_default_model(signed, settings)
        return 0
    return _login_provider(provider, settings)


def _login_provider(provider: str, settings: dict[str, Any]) -> int:
    choices = providers()
    if provider not in choices:
        raise CliError(f"Unknown provider {provider!r}. Known providers: {', '.join(choices)}")
    if provider == "github-copilot":
        if not copilot_login(ask=_cli_ask, echo=click.echo, settings=settings):
            return 1
    elif provider in NO_KEY_PROVIDERS:
        click.echo(f"{provider} is a local provider — no credentials needed.")
    else:
        if provider in signed_in_providers():
            click.echo(f"Replacing the stored credentials for {provider}.")
        key = _cli_ask(f"{provider} API key: ", password=True).strip()
        if not key:
            click.echo("Cancelled.")
            return 0
        store_api_key(provider, key)
        click.echo(f"Stored the API key for {provider} in ~/.tynex/agent/auth.json.")
    _report_default_model(provider, settings)
    return 0


@click.command(
    "logout", context_settings=CONTEXT_SETTINGS, help="Remove a provider's credentials from ~/.tynex/agent/auth.json."
)
@click.argument("provider", required=False)
def logout_command(provider: str | None) -> int:
    settings = load_settings(os.getcwd())
    register_custom_providers(settings.get("providers") or {})
    stored = signed_in_providers()
    if provider is None:
        if not stored:
            click.echo("No stored credentials in ~/.tynex/agent/auth.json.")
            return 0
        index = _cli_select("Sign out of which provider?", stored)
        if index is None:
            click.echo("Nothing to do.")
            return 0
        provider = stored[index]
    if provider not in signed_in_providers():
        click.echo(f"No stored credentials for {provider}.")
        return 0
    remove_credential(provider)
    click.echo(f"Signed out of {provider}; removed from ~/.tynex/agent/auth.json.")
    if isinstance((settings.get("apiKeys") or {}).get(provider), str):
        click.echo(f"Note: settings apiKeys still configure {provider}.")
    envs = [name for name in provider_api_key_env(provider) if os.environ.get(name)]
    if envs:
        click.echo(f"Note: {', '.join(envs)} is still set in the environment.")
    return 0


def _mcp_settings_file(project: bool) -> Path:
    return project_dir(os.getcwd()) / "settings.json" if project else settings_path()


def _parse_pairs(pairs: tuple[str, ...], option: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise CliError(f"{option} expects KEY=VALUE, got {pair!r}")
        values[key.strip()] = value
    return values


def _enables_server(settings: dict[str, Any], name: str) -> bool:
    servers = settings.get("mcpServers")
    entry = servers.get(name) if isinstance(servers, dict) else None
    return isinstance(entry, dict) and entry.get("enabled") is not False


@click.group("mcp", context_settings=CONTEXT_SETTINGS, help="List, add and remove MCP servers.")
def mcp_command() -> None:
    pass


@mcp_command.command("list", context_settings=CONTEXT_SETTINGS, help="Show built-in and configured servers.")
def mcp_list_command() -> int:
    cwd = os.getcwd()
    project_settings = read_settings_file(project_dir(cwd) / "settings.json")
    global_settings = read_settings_file(settings_path())
    merged = load_settings(cwd)

    click.echo("Built-in servers (enable with `tynex mcp add <name>`):")
    for status in builtin_status(merged):
        tags: list[str] = []
        if status["enabled"]:
            if _enables_server(project_settings, status["name"]):
                tags.append("enabled: project settings")
            elif _enables_server(global_settings, status["name"]):
                tags.append("enabled: global settings")
            else:
                tags.append("enabled")
        if not status["available"]:
            tags.append(f"install {status['runtime']}")
        mark = "✓" if status["available"] else "✗"
        suffix = f"  ({', '.join(tags)})" if tags else ""
        click.echo(f"  {mark} {status['name']:<20} {status['runtime']:<4} {status['description']}{suffix}")
    for runtime in ("npx", "uvx"):
        click.echo(f"  {runtime}: {runtime_path(runtime) or 'not found'}")

    click.echo("\nConfigured servers (mcpServers):")
    servers = merged.get("mcpServers") if isinstance(merged.get("mcpServers"), dict) else {}
    builtins = set(builtin_catalog())
    custom = {name: entry for name, entry in servers.items() if name not in builtins}
    if not custom:
        click.echo("  (none)")
    project_servers = project_settings.get("mcpServers")
    for name, entry in custom.items():
        where = "project" if isinstance(project_servers, dict) and name in project_servers else "global"
        if isinstance(entry, dict) and entry.get("url"):
            target = f"http {entry['url']}"
        elif isinstance(entry, dict) and entry.get("command"):
            target = " ".join([str(entry.get("command")), *[str(arg) for arg in entry.get("args") or []]])
        else:
            target = "(malformed)"
        disabled = " (disabled)" if isinstance(entry, dict) and entry.get("enabled") is False else ""
        click.echo(f"  {name}: {target}  ({where}){disabled}")
    click.echo("\nSessions pick these up at startup; /mcp shows live state, --no-mcp disables them for a run.")
    return 0


@mcp_command.command("add", context_settings=CONTEXT_SETTINGS)
@click.argument("name")
@click.option("--project", is_flag=True, help="Write to .tynex/settings.json instead of the global settings.json.")
@click.option("--command", "server_command", metavar="CMD", help="Custom stdio server command (npx, uvx, docker, …).")
@click.option("--arg", "server_args", multiple=True, metavar="ARG", help="Argument for --command (repeatable).")
@click.option("--env", "env_pairs", multiple=True, metavar="KEY=VALUE", help="Environment for the server (repeatable).")
@click.option("--url", metavar="URL", help="Custom streamable-HTTP server URL.")
@click.option(
    "--header", "header_pairs", multiple=True, metavar="KEY=VALUE", help="HTTP header for --url (repeatable)."
)
def mcp_add_command(
    name: str,
    project: bool,
    server_command: str | None,
    server_args: tuple[str, ...],
    env_pairs: tuple[str, ...],
    url: str | None,
    header_pairs: tuple[str, ...],
) -> int:
    """Enable a built-in NAME, or add a custom server via --command/--url."""
    if url and server_command:
        raise CliError("pass either --url or --command, not both")
    if header_pairs and not url:
        raise CliError("--header requires --url")
    if (server_args or env_pairs) and not server_command:
        raise CliError("--arg and --env require --command")
    path = _mcp_settings_file(project)
    data = read_settings_file(path)
    catalog = builtin_catalog()
    servers = data.get("mcpServers")
    servers = dict(servers) if isinstance(servers, dict) else {}
    if url or server_command:
        entry: dict[str, Any] = {}
        if url:
            entry["url"] = url
            if header_pairs:
                entry["headers"] = _parse_pairs(header_pairs, "--header")
        else:
            entry["command"] = server_command
            if server_args:
                entry["args"] = list(server_args)
            if env_pairs:
                entry["env"] = _parse_pairs(env_pairs, "--env")
        if name in catalog:
            entry["enabled"] = True
        servers[name] = entry
        data["mcpServers"] = servers
        write_settings_file(path, data)
        click.echo(f"Added MCP server '{name}' to {path}")
        return 0
    spec = catalog.get(name)
    if spec is None:
        known = ", ".join(catalog) or "(none)"
        raise CliError(f"unknown built-in MCP server {name!r} (known: {known}) — or pass --command/--url")
    command = str(spec.get("command") or "")
    if command and shutil.which(command) is None:
        click.echo(f"Warning: '{name}' needs '{command}' on PATH; it will not start until installed.", err=True)
    existing = servers.get(name)
    if isinstance(existing, dict) and existing.get("enabled") is True:
        click.echo(f"Built-in MCP server '{name}' is already enabled in {path}")
        return 0
    servers[name] = {**(existing if isinstance(existing, dict) else {}), "enabled": True}
    data["mcpServers"] = servers
    write_settings_file(path, data)
    click.echo(f"Enabled built-in MCP server '{name}' ({command or 'http'}) in {path}")
    return 0


@mcp_command.command("remove", context_settings=CONTEXT_SETTINGS)
@click.argument("name")
@click.option("--project", is_flag=True, help="Edit .tynex/settings.json instead of the global settings.json.")
def mcp_remove_command(name: str, project: bool) -> int:
    """Remove NAME from the chosen settings file."""
    path = _mcp_settings_file(project)
    data = read_settings_file(path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or name not in servers:
        raise CliError(f"no MCP server '{name}' in {path}")
    del servers[name]
    if not servers:
        data.pop("mcpServers")
    write_settings_file(path, data)
    click.echo(f"Removed MCP server '{name}' from {path}")
    return 0


@click.command("models", context_settings=CONTEXT_SETTINGS, help="List provider models (all providers by default).")
@click.argument("provider", required=False)
def models_command(provider: str | None) -> int:
    print_models(_settings(), provider)
    return 0


@click.command("sessions", context_settings=CONTEXT_SETTINGS, help="List sessions for the current directory.")
def sessions_command() -> int:
    print_sessions(os.getcwd())
    return 0


class TynexGroup(click.Group):
    """Give the default run and the subcommands one Click-generated help screen."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        rest = click.Command.parse_args(self, ctx, args)
        if rest and rest[0] in self.commands:
            ctx._protected_args, ctx.args = rest[:1], rest[1:]
        else:
            ctx.args, ctx._protected_args = rest, []
        return ctx.args


@click.group(cls=TynexGroup, context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.version_option(__version__, "-v", "--version", prog_name="tynex", message="%(prog)s %(version)s")
@click.option("-p", "--print", "print_mode", is_flag=True, help="Run once and print the response.")
@click.option("--json", "json_mode", is_flag=True, help="Run once and emit JSON events.")
@click.option("--rpc", "rpc_mode", is_flag=True, help="Serve the JSON-RPC protocol on stdin/stdout.")
@click.option(
    "-c",
    "-r",
    "--resume",
    is_flag=False,
    flag_value="",
    default=None,
    metavar="[ID]",
    help="Continue or resume a session (id prefix, file name, or path; without ID: the most recent).",
)
@click.option(
    "-m", "--model", metavar="MODEL", help="Model reference: provider/model-id, a unique partial id, or a glob."
)
@click.option("--thinking", type=click.Choice(list(THINKING_LEVELS)), help="Thinking/reasoning level.")
@click.option(
    "--permission-mode",
    type=click.Choice(list(PERMISSION_MODES)),
    help="Approval policy: plan, ask, accept-edits (default), bypass. Per-tool rules: permissions.toml.",
)
@click.option("--budget", metavar="SPEC", help="Hard cap: cost=USD (e.g. cost=2.50) or tokens=N (e.g. tokens=250000).")
@click.option("--no-mcp", is_flag=True, help="Do not start MCP servers configured in settings.json.")
@click.option("--no-session", is_flag=True, help="Do not persist the session to disk.")
@click.option("--append-system-prompt", metavar="TEXT", help="Append text to the system prompt.")
@click.pass_context
def cli(
    ctx: click.Context,
    print_mode: bool,
    json_mode: bool,
    rpc_mode: bool,
    resume: str | None,
    model: str | None,
    thinking: str | None,
    permission_mode: str | None,
    budget: str | None,
    no_mcp: bool,
    no_session: bool,
    append_system_prompt: str | None,
) -> int | None:
    """Tynex — a minimal terminal coding agent harness.

    Pass a prompt to run the agent, or pick a subcommand: ``login``, ``logout``,
    ``mcp``, ``models``, ``sessions``.
    """
    if ctx.invoked_subcommand is not None:
        return None
    modes = [name for name, enabled in (("print", print_mode), ("json", json_mode), ("rpc", rpc_mode)) if enabled]
    if len(modes) > 1:
        raise CliError("choose one of --print, --json or --rpc")
    mode = modes[0] if modes else "interactive"
    text = " ".join(ctx.args).strip()
    if mode in ("print", "json") and not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if mode in ("print", "json") and not text:
        raise CliError(f"{mode} mode requires a prompt (pass it as an argument or on stdin)")

    cwd = os.getcwd()
    settings = _settings()
    chosen = model or settings.get("model")
    try:
        resolved_model = resolve_model_retry(chosen, settings) if chosen else None
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    session_file = None
    if resume is not None:
        session_file = latest_session_file(cwd) if resume == "" else find_session(cwd, resume)
        if not session_file:
            message = "no session to resume for this directory" if resume == "" else f"no session matching {resume!r}"
            raise CliError(message)
    budget_limits = None
    if budget:
        try:
            budget_limits = parse_budget(budget)
        except ValueError as exc:
            raise CliError(str(exc)) from exc
    session = AgentSession(
        cwd=cwd,
        model=resolved_model,
        thinking=thinking,
        session_file=session_file,
        no_session=no_session,
        append_system_prompt=append_system_prompt,
        permission_mode=permission_mode,
        budget=budget_limits,
        mcp=not no_mcp,
    )
    if mode != "interactive" and session.model is None:
        raise CliError(
            "no model available — sign in with `tynex login` or set a provider API key, then retry "
            "(interactive mode offers the sign-in menu)."
        )
    if mode != "interactive":
        _warn_about_unapprovable_tools(session)
    if mode == "print":
        return run_print(session, text)
    if mode == "json":
        return run_json(session, text)
    if mode == "rpc":
        return run_rpc(session)
    try:
        from tynex.tui.app import run_interactive
    except ModuleNotFoundError as exc:  # pragma: no cover - only without the CLI extra
        missing = exc.name or "a UI package"
        raise CliError(f"the interactive UI needs '{missing}'; install it with: pip install 'tynex[cli]'") from exc
    return run_interactive(session, initial_message=text or None)


cli.add_command(login_command)
cli.add_command(logout_command)
cli.add_command(mcp_command)
cli.add_command(models_command)
cli.add_command(sessions_command)


def _warn_about_unapprovable_tools(session: AgentSession) -> None:
    if session.permission_gate.mode not in ("ask", "accept-edits"):
        return
    if not {getattr(tool, "permission_kind", "tool") for tool in session.tools} - {"read", "web"}:
        return
    mode = session.permission_gate.mode
    print(
        f"[tynex] permission mode '{mode}': tool calls that need approval are denied here "
        "(no interactive prompt in this mode).",
        file=sys.stderr,
    )
    print("[tynex] add allow rules to permissions.toml, or use --permission-mode bypass.", file=sys.stderr)


def _ensure_utf8_streams() -> None:
    """Use UTF-8 for redirected stdio (Windows)."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_streams()
    args = list(argv) if argv is not None else sys.argv[1:]
    return run_click(lambda: cli.main(args=args, prog_name="tynex", standalone_mode=False))


def _abort_key_handler(session: AgentSession):
    def handler(_signum, _frame):
        if session.signal is not None and not session.signal.aborted:
            session.abort()
        else:
            raise KeyboardInterrupt

    return handler


def _tool_summary(name: str, args: dict[str, Any]) -> str:
    if name == "bash":
        return f"$ {' '.join(str(args.get('command', '')).split())[:120]}"
    if name in ("read", "write", "edit", "ls"):
        return str(args.get("path", ""))
    if name in ("grep", "find"):
        return f"{args.get('pattern', '')} in {args.get('path', '.')}"
    if name == "websearch":
        return str(args.get("query", ""))[:120]
    return json.dumps(args, ensure_ascii=False)[:120]


def serialize_event(event: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe event for --json / RPC output."""
    out: dict[str, Any] = {"type": event.get("type"), **{key: event[key] for key in _EVENT_KEYS if key in event}}
    if "toolResults" in event:
        out["toolResults"] = [message_to_dict(m) for m in event["toolResults"]]
    if "messages" in event:
        out["messages"] = [message_to_dict(m) for m in event["messages"]]
    if "message" in event and hasattr(event["message"], "role"):
        out["message"] = message_to_dict(event["message"])
    if "delta" in event:
        out["delta"] = {k: v for k, v in event["delta"].items() if k != "partial"}
    if "partialResult" in event:
        out["partialResult"] = {
            "content": [c.text if hasattr(c, "text") else "" for c in event["partialResult"].content]
        }
    if "result" in event:
        result = event["result"]
        out["result"] = {
            "content": [content_to_dict(c) for c in result.content],
            "details": result.details,
            "isError": result.is_error,
        }
    return out


def _prompt(session: AgentSession, text: str, emit: Callable[[dict[str, Any]], None]) -> int:
    old_handler = None
    with contextlib.suppress(ValueError):
        old_handler = signal.signal(signal.SIGINT, _abort_key_handler(session))
    try:
        session.prompt(text, emit=emit)
        return 0
    except KeyboardInterrupt:
        session.abort()
        session.finalize_aborted_turn(emit=emit)
        return 130
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)


def run_print(session: AgentSession, prompt_text: str, quiet: bool = False) -> int:
    """Run one prompt, print the final text to stdout and tool activity to stderr."""
    captured: str | None = None
    last: AssistantMessage | None = None
    limit: str | None = None
    stderr = sys.stderr

    def sink(event: dict[str, Any]) -> None:
        nonlocal captured, last, limit
        etype = event.get("type")
        if etype == "message_end" and isinstance(event.get("message"), AssistantMessage):
            last = event["message"]
            captured = last.text() or captured
        elif etype == "tool_execution_start" and not quiet:
            print(
                f"[tool] {event.get('toolName')}: {_tool_summary(event.get('toolName', ''), event.get('args') or {})}",
                file=stderr,
            )
        elif etype == "tool_execution_end" and event.get("isError") and not quiet:
            result = event.get("result")
            text = (
                result.content[0].text if result and result.content and hasattr(result.content[0], "text") else "error"
            )
            print(f"[tool error] {text.splitlines()[0][:200]}", file=stderr)
        elif etype == "permission_decision" and not quiet:
            print(f"[permission] {event.get('action')}: {event.get('tool')} — {event.get('reason')}", file=stderr)
        elif etype == "limit_reached":
            limit = str(event.get("reason") or "limit reached")
            if not quiet:
                print(f"[limit] {limit}", file=stderr)
        elif etype == "compaction" and not quiet:
            print(f"[compaction] summarized {event.get('messagesSummarized')} messages", file=stderr)

    if (code := _prompt(session, prompt_text, sink)) == 130:
        print("[aborted]", file=stderr)
        return code
    if captured:
        print(captured)
        sys.stdout.flush()
    elif last and last.stop_reason in ("error", "aborted"):
        print(f"tynex: {last.error_message or 'request failed'}", file=stderr)
        return 1
    if limit:
        print(f"tynex: stopped — {limit}", file=stderr)
        return 1
    return 0


def run_json(session: AgentSession, prompt_text: str) -> int:
    """Run one prompt and emit the event stream as JSONL on stdout."""
    lock = threading.Lock()

    def sink(event: dict[str, Any]) -> None:
        with lock:
            print(json.dumps(serialize_event(event), ensure_ascii=False, default=str), flush=True)

    return _prompt(session, prompt_text, sink)


class RpcMode:
    def __init__(self, session: AgentSession) -> None:
        self.session = session
        self._stdout_lock = threading.Lock()
        self._commands: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._queued_prompts = 0
        self._abort_pending = threading.Event()

    def _send(self, payload: dict[str, Any]) -> None:
        with self._stdout_lock:
            print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)

    def _respond(self, command: dict[str, Any], success: bool, data: Any = None, error: str | None = None) -> None:
        payload = {"type": "response", "id": command.get("id"), "command": command.get("type"), "success": success}
        if data is not None:
            payload["data"] = data
        if error is not None:
            payload["error"] = error
        self._send(payload)

    def _emit_event(self, event: dict[str, Any]) -> None:
        self._send({"type": "event", "event": serialize_event(event)})

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            command = self._commands.get()
            if command is None:
                break
            if command.get("type") == "prompt":
                self._queued_prompts = max(0, self._queued_prompts - 1)
                if self._abort_pending.is_set():
                    self._abort_pending.clear()
                    self._respond(command, True, {"state": self.session.stats(), "aborted": True})
                    continue
            try:
                self._execute(command)
            except Exception as exc:
                self._respond(command, False, error=str(exc))

    def _execute(self, command: dict[str, Any]) -> None:
        ctype = command.get("type")
        if ctype == "prompt":
            self.session.prompt(str(command.get("message", "")), emit=self._emit_event)
            data = {"state": self.session.stats()}
        elif ctype == "set_model":
            model = self.session.set_model_by_pattern(str(command.get("model", "")))
            data = {"model": f"{model.provider}/{model.id}"}
        elif ctype == "compact":
            data = {"compacted": self.session.compact_now(command.get("instructions")), "state": self.session.stats()}
        elif ctype == "new_session":
            self.session.new_session()
            data = {"sessionId": self.session.session.session_id}
        elif ctype == "list_sessions":
            sessions = list_sessions(self.session.cwd)
            for info in sessions:
                info.pop("cwd", None)
            data = {"sessions": sessions}
        elif ctype == "resume_session":
            identifier = str(command.get("session", ""))
            path = find_session(self.session.cwd, identifier)
            if not path:
                self._respond(command, False, error=f"No session matching {identifier!r}")
                return
            self.session.switch_session(path)
            data = {"sessionId": self.session.session.session_id}
        elif ctype == "set_permission_mode":
            data = {"mode": self.session.set_permission_mode(str(command.get("mode", "")))}
        elif ctype == "undo":
            data = {"undone": [group.summary() for group in self.session.undo_last(int(command.get("count") or 1))]}
        elif ctype == "shutdown":
            self._respond(command, True, {})
            self._shutdown.set()
            return
        else:
            self._respond(command, False, error=f"Unknown command type: {ctype!r}")
            return
        self._respond(command, True, data)

    def _handle_immediate(self, command: dict[str, Any]) -> bool:
        ctype = command.get("type")
        if ctype == "steer":
            self.session.steer(str(command.get("message", "")))
            self._respond(command, True, {"queued": "steer"})
            return True
        if ctype == "follow_up":
            self.session.follow_up(str(command.get("message", "")))
            self._respond(command, True, {"queued": "followUp"})
            return True
        if ctype == "abort":
            if self._queued_prompts:
                self._abort_pending.set()
            self.session.abort()
            self.session.finalize_aborted_turn(emit=self._emit_event)
            self._respond(command, True, {})
            return True
        if ctype == "get_state":
            stats = self.session.stats()
            stats["running"] = self.session.running
            stats["thinking"] = self.session.thinking
            stats["tools"] = [tool.name for tool in self.session.tools]
            self._respond(command, True, stats)
            return True
        if ctype == "get_messages":
            self._respond(command, True, {"messages": [message_to_dict(m) for m in self.session.context.messages]})
            return True
        return False

    def run(self) -> int:
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        current = self.session.session
        ready = {
            "id": current.session_id,
            "file": current.file_path,
            "model": f"{self.session.model.provider}/{self.session.model.id}",
            "cwd": self.session.cwd,
        }
        self._send({"type": "ready", "session": ready})
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                command = json.loads(raw)
            except json.JSONDecodeError as exc:
                self._send(
                    {"type": "response", "id": None, "command": None, "success": False, "error": f"Invalid JSON: {exc}"}
                )
                continue
            if not isinstance(command, dict):
                continue
            if self._handle_immediate(command):
                if self._shutdown.is_set():
                    break
                continue
            self._commands.put(command)
            if command.get("type") == "prompt":
                self._queued_prompts += 1
            if command.get("type") == "shutdown":
                self._commands.put(None)
                break
        self._commands.put(None)
        if self._worker is not None:
            self._worker.join(timeout=5)
        self._shutdown.set()
        return 0


def run_rpc(session: AgentSession) -> int:
    return RpcMode(session).run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
