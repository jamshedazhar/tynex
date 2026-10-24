"""Slash-command implementations for sessions, models, skills and exports."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from tynex.ai.models import format_model, format_tokens, providers
from tynex.ai.providers.base import THINKING_LEVELS, clamp_thinking
from tynex.coding.model_catalog import ensure_models
from tynex.coding.session import export_session_to_html, export_session_to_jsonl, find_session, list_sessions
from tynex.coding.settings import remember_setting
from tynex.coding.skills import read_skill_instructions

if TYPE_CHECKING:
    from rich.console import Console

    from tynex.coding.agent_session import AgentSession

AskFn = Callable[..., str]
SelectFn = Callable[..., "int | None"]
_INTERRUPTED = -1  # the picker was interrupted: return quietly (no "unchanged" message)


def short(text: str, limit: int) -> str:
    single = " ".join(str(text).split())
    return single if len(single) <= limit else single[: limit - 1] + "…"


def _pick(select: SelectFn, title: str, options: list[str], selected: int | None = None) -> int | None:
    try:
        return select(title, options) if selected is None else select(title, options, selected=selected)
    except (KeyboardInterrupt, EOFError):
        return _INTERRUPTED


def list_models(session: "AgentSession", console: "Console") -> None:
    current = format_model(session.model) if session.model is not None else None
    for provider in providers():
        models, note = ensure_models(provider, session.settings)
        console.print(f"[bold]{provider}[/]")
        if not models:
            console.print(f"  [dim]({note or f'no models — /login {provider}'})[/]")
            continue
        for model in models:
            marker = "[green]●[/]" if current is not None and format_model(model) == current else " "
            reasoning = " ±reasoning" if model.reasoning else ""
            console.print(f"  {marker} {model.id} [dim]({format_tokens(model.context_window)} ctx{reasoning})[/]")
    console.print("[dim]/model <pattern> to switch.\n[/]")


def pick_model(session: "AgentSession", console: "Console", select: SelectFn) -> None:
    """List the current provider's models and switch with the arrow picker."""
    model = session.model
    if model is None:
        return list_models(session, console)
    models, note = ensure_models(model.provider, session.settings)
    if not models:
        console.print(
            f"[yellow]No models available for {model.provider}[/] — {note or f'run /login {model.provider}'}."
        )
        return
    if not console.is_terminal:
        # Piped/one-shot runs: no arrow picker, just list the catalog.
        for entry in models:
            marker = "[green]●[/]" if entry.id == model.id else " "
            console.print(f"  {marker} {entry.id} [dim]({format_tokens(entry.context_window)} ctx)[/]")
        console.print("[dim]/model <id> to switch.\n[/]")
        return
    options = [
        f"{entry.id} ({format_tokens(entry.context_window)} ctx)" + ("  ← current" if entry.id == model.id else "")
        for entry in models
    ]
    start = next((index for index, entry in enumerate(models) if entry.id == model.id), 0)
    index = _pick(select, f"Models — {model.provider}", options, start)
    if index is _INTERRUPTED:
        return
    if index is None:
        console.print("[dim]Model unchanged.[/]")
        return
    chosen = models[index]
    if chosen.id == model.id:
        console.print(f"[dim]{format_model(chosen)} is already active.[/]")
        return
    session.set_model(chosen)
    console.print(f"[green]Switched to {format_model(chosen)}[/]")


def _resumed(session: "AgentSession", console: "Console", path: str) -> None:
    session.switch_session(path)
    console.print(f"[green]Resumed {Path(path).name} ({len(session.context.messages)} messages)[/]")


def resume(session: "AgentSession", console: "Console", arg: str, ask: AskFn, select: SelectFn) -> None:
    sessions = list_sessions(session.cwd)
    if not sessions:
        console.print("[yellow]No sessions for this directory.[/]")
        return
    if arg:
        path = find_session(session.cwd, arg)
        if not path:
            console.print(f"[red]No session matching '{arg}'[/]")
            return
        return _resumed(session, console, path)
    recent = sessions[:20]
    if console.is_terminal:
        options = [
            f"{info['id'][:8]} · {time.strftime('%Y-%m-%d %H:%M', time.localtime(info['modified']))} · "
            f"{info['messages']} msgs {info.get('label') or ''}".rstrip()
            for info in recent
        ]
        index = _pick(select, "Resume session", options)
        # ``_INTERRUPTED`` is -1: without this check an interrupted picker
        # would silently resume ``recent[-1]``.
        if index is _INTERRUPTED or index is None:
            return
        return _resumed(session, console, recent[index]["file"])
    for index, info in enumerate(recent, 1):
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(info["modified"]))
        console.print(
            f"  [bold]{index:>2}[/] {info['id'][:8]} [dim]{stamp} · {info['messages']} msgs {info.get('label') or ''}[/]"
        )
    try:
        choice = ask("Resume session number (empty to cancel): ").strip()
    except (KeyboardInterrupt, EOFError):
        return
    if choice.isdigit() and 1 <= int(choice) <= len(recent):
        _resumed(session, console, recent[int(choice) - 1]["file"])


def session_info(session: "AgentSession", console: "Console") -> None:
    stats = session.stats()
    console.print("[bold]Session[/]")
    if stats.get("label"):
        console.print(f"  name     {stats['label']}")
    console.print(f"  id       {stats['sessionId']}")
    console.print(f"  file     {stats['sessionFile'] or '(in-memory)'}")
    console.print(f"  model    {stats['model']} · thinking {stats['thinking']}")
    console.print("[bold]Messages[/]")
    console.print(
        f"  {stats['totalMessages']} total · {stats['userMessages']} user · "
        f"{stats['assistantMessages']} assistant · {stats['toolCalls']} tool calls · "
        f"{stats['toolResults']} tool results"
    )
    cache = ""
    if stats["cacheRead"] or stats["cacheWrite"]:
        cache = f" · {format_tokens(stats['cacheRead'])} cache read · {format_tokens(stats['cacheWrite'])} cache write"
    console.print("[bold]Tokens[/]")
    console.print(
        f"  {format_tokens(stats['input'])} in · {format_tokens(stats['output'])} out · "
        f"{format_tokens(stats['totalTokens'])} total{cache} · ${stats['cost']:.4f}"
    )


def thinking(session: "AgentSession", console: "Console", arg: str, select: SelectFn) -> None:
    available = session.available_thinking_levels()
    current = session.thinking
    if arg:
        return set_thinking(session, console, arg)
    if not console.is_terminal or len(available) == 1:
        note = " (the current model does not support thinking)" if available == ["off"] else ""
        console.print(f"[dim]thinking {current} · available: {', '.join(available)}{note}[/]")
        return
    options = [level + ("  ← current" if level == current else "") for level in available]
    index = _pick(select, "Thinking level", options, available.index(current) if current in available else 0)
    if index is _INTERRUPTED:
        return
    if index is None:
        console.print("[dim]Thinking level unchanged.[/]")
        return
    set_thinking(session, console, available[index])


def _explain_unavailable_thinking(session: "AgentSession", console: "Console", level: str) -> None:
    """Explain why a level cannot be selected, and what to do about it.

    "choose off" told the user nothing: the level is valid, the model just has no
    reasoning mode, so the fix is a different model (or staying at off).
    """
    model = session.model
    if model is None:
        console.print(
            f"[yellow]Thinking level '{level}' needs a reasoning model, and none is selected — "
            "sign in with /login, or pick one with /model.[/]"
        )
    elif not model.reasoning:
        console.print(
            f"[yellow]Thinking level '{level}' needs a reasoning model — {format_model(model)} does not "
            "support thinking. Use /model to switch, or /thinking off.[/]"
        )
    else:
        console.print(
            f"[yellow]Thinking level '{level}' is not available for {format_model(model)} "
            f"(available: {', '.join(session.available_thinking_levels())}).[/]"
        )


def set_thinking(session: "AgentSession", console: "Console", level: str) -> None:
    requested = level.strip().lower()
    if requested not in THINKING_LEVELS:
        console.print(f'[red]Unknown thinking level "{level}". Available levels: {", ".join(THINKING_LEVELS)}.[/]')
        return
    available = session.available_thinking_levels()
    if requested not in available:
        if len(available) == 1:  # the model cannot think at all
            _explain_unavailable_thinking(session, console, level.strip())
            return
        # The model reasons but the catalog lists other levels: use the nearest.
        nearest = clamp_thinking(requested, available)
        console.print(
            f"[dim]{requested} is not offered by {format_model(session.model)}; "
            f"using {nearest} ({', '.join(available)}).[/]"
        )
    try:
        applied = session.set_thinking(level)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        return
    # Remember the choice in the global settings so new sessions start with it.
    remember_setting("thinking", applied)
    console.print(f"[green]Thinking level: {applied}[/]")


def export_session(session: "AgentSession", console: "Console", arg: str) -> None:
    path_arg = arg.strip() or None
    try:
        if path_arg and path_arg.endswith(".jsonl"):
            path = export_session_to_jsonl(session.session, path_arg, cwd=session.cwd)
        else:
            path = export_session_to_html(session.session, path_arg, cwd=session.cwd)
    except Exception as exc:
        console.print(f"[red]Failed to export session: {exc}[/]")
        return
    console.print(f"[green]Session exported to: {path}[/]")


def compact(session: "AgentSession", console: "Console", arg: str) -> None:
    console.print("[dim]Compacting…[/]")
    ok = session.compact_now(arg or None)
    console.print("[green]Compacted.[/]" if ok else "[yellow]Nothing to compact.[/]")


def new_session(session: "AgentSession", console: "Console") -> None:
    session.new_session()
    console.print(f"[green]New session {session.session.session_id[:8]}[/]")


def rename_session(session: "AgentSession", console: "Console", arg: str) -> None:
    if arg:
        session.session.append_label(arg)
        console.print(f"[green]Session named '{arg}'[/]")
    else:
        console.print(f"[dim]Session label: {session.session_label() or '(none)'}[/]")


def print_cwd(session: "AgentSession", console: "Console") -> None:
    console.print(session.cwd)


def permissions(session: "AgentSession", console: "Console", arg: str) -> None:
    """Show or switch the permission mode and list active rules."""
    from tynex.agent.permissions import PERMISSION_MODES, mode_description

    arg = arg.strip()
    value = ""
    if arg == "mode" or arg.startswith("mode "):
        _, _, value = arg.partition(" ")
        value = value.strip()
        if not value:
            console.print(f"[dim]usage: /permissions mode {'|'.join(PERMISSION_MODES)}[/]")
            return
    elif arg:
        value = arg
    if value:
        try:
            applied = session.set_permission_mode(value)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            console.print(f"[dim]usage: /permissions [mode] {'|'.join(PERMISSION_MODES)}[/]")
            return
        console.print(f"[green]Permission mode: {applied}[/] — {mode_description(applied)}")
        for warning in session.permission_config.warnings:
            console.print(f"[yellow]{warning}[/]")
        return
    summary = session.permission_summary()
    console.print(f"[bold]Permissions[/] mode [green]{summary['mode']}[/] — {mode_description(summary['mode'])}")
    if summary["rules"]:
        console.print("[bold]Rules[/] [dim](last match wins; project rules override global)[/]")
        for rule in summary["rules"]:
            pattern = f" [dim]({rule['match']})[/]" if rule["match"] else ""
            source = Path(rule["source"]).name if rule["source"] not in ("", "cli") else rule["source"]
            console.print(f"  {rule['action']:<5} {rule['tool']}{pattern} [dim]{source}[/]")
    else:
        console.print("[dim]No rules — mode defaults apply. Add .tynex/permissions.toml to codify policy.[/]")
    if summary["grants"]:
        console.print(f"[dim]Session grants: {', '.join(summary['grants'])}[/]")
    if not summary["checkpoints"]:
        console.print("[dim]Checkpoints are disabled (settings 'checkpoints': false).[/]")
    for warning in summary["warnings"]:
        console.print(f"[yellow]{warning}[/]")
    console.print(f"[dim]/permissions {'|'.join(PERMISSION_MODES)} to switch ('mode' is optional).[/]")


def budget(session: "AgentSession", console: "Console") -> None:
    snapshot = session.budget.snapshot(session.budget_limits)
    limits = snapshot["limits"]
    if not limits:
        console.print(
            "[dim]No budget set. Use --budget cost=2.50 or --budget tokens=250000 "
            "(both: --budget cost=2.50,tokens=250000).[/]"
        )
        return
    console.print("[bold]Budget[/]")
    token_limit = limits.get("maxTokens")
    cost_limit = limits.get("maxCostUsd")
    if token_limit is not None:
        console.print(f"  {'tokens':<10} {snapshot['tokens']:,} / {token_limit:,}")
    if cost_limit is not None:
        console.print(f"  {'cost':<10} ${snapshot['cost']:.4f} / ${cost_limit:.4f}")


def undo_session(session: "AgentSession", console: "Console", arg: str) -> None:
    count = 1
    if arg.strip():
        try:
            count = max(1, int(arg.strip()))
        except ValueError:
            console.print("[red]Usage: /undo [count][/]")
            return
    if session.checkpoints is None:
        console.print("[yellow]Checkpoints are disabled[/] (settings 'checkpoints': false).")
        return
    groups = session.undo_last(count)
    if not groups:
        console.print("[yellow]Nothing to undo.[/]")
        return
    for group in groups:
        files = ", ".join(f.path for f in group.files) or "(no files)"
        console.print(f"[green]Reverted[/] {group.tool} — {files}")
    console.print("[dim]The model was told about the revert.[/]")


def mcp_servers(session: "AgentSession", console: "Console") -> None:
    summaries = session.mcp_manager.summaries() if session.mcp_manager is not None else []
    if not summaries:
        console.print(
            "[dim]No MCP servers connected — enable built-ins with[/] [bold]tynex mcp add memory[/][dim], "
            "or add an 'mcpServers' entry to settings.json.[/]"
        )
    for info in summaries:
        if info["connected"]:
            tools = ", ".join(info["tools"]) or "no tools"
            console.print(f"  [green]●[/] [bold]{info['name']}[/] [dim]{info['transport']} · {info['server']}[/]")
            console.print(f"    [dim]{tools}[/]")
        else:
            console.print(f"  [red]✗[/] [bold]{info['name']}[/] [dim]{info['transport']} — not connected[/]")
    for warning in session.mcp_warnings:
        console.print(f"[yellow]{warning}[/]")


def list_skills(session: "AgentSession", console: "Console") -> None:
    if not session.skills:
        console.print("[yellow]No skills found[/] — add .tynex/skills/<name>/SKILL.md.")
        return
    for skill in session.skills:
        console.print(f"  [bold]/skill:{skill.name}[/] [dim]— {skill.description}[/]")
        console.print(f"    [dim]{skill.path}[/]")


def run_skill(session: "AgentSession", console: "Console", name: str, arg: str) -> str | None:
    skill = next((s for s in session.skills if s.name == name), None)
    if skill is None:
        known = ", ".join(s.name for s in session.skills) or "none"
        console.print(f"[red]Unknown skill '{name}'.[/] Known skills: {known}")
        return None
    instructions = read_skill_instructions(skill)
    if instructions is None:
        console.print(f"[red]Could not read skill '{name}' ({skill.path}).[/]")
        return None
    directory = str(Path(skill.path).parent).replace("\\", "/")
    message = (
        f'Skill "{skill.name}" (directory: {directory}). '
        "Follow the instructions below; resolve relative paths against the directory.\n\n"
        f"{instructions}"
    )
    return f"{message}\n\n{arg}" if arg else message
