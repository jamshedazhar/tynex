"""Shell tools: one platform-resolved ``ShellTool`` for bash and PowerShell."""

import contextlib
import os
import queue
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, NamedTuple, Self

from tynex.agent.types import AgentTool, ToolContext, ToolResult, UpdateSink
from tynex.ai.types import AbortSignal
from tynex.coding.process import detached_options, kill_process_tree
from tynex.coding.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_tail,
)

_MAX_OUTPUT_BYTES = 64 * 1024 * 1024  # stop accumulating beyond 64MB
_UPDATE_PREVIEW_BYTES = DEFAULT_MAX_BYTES * 2  # cover the 50KB/2000-line tail without copying the whole buffer
_TYNEX_ENV_KEYS = ("TYNEX_SESSION_ID", "TYNEX_SESSION_FILE", "TYNEX_PROVIDER", "TYNEX_MODEL", "TYNEX_REASONING_LEVEL")


class _ShellFlavor(NamedTuple):
    name: str = "bash"
    snippet: str = "Execute bash commands (ls, grep, find, etc.)"
    prompt: str = "$"
    path: str = "bash"
    args: Sequence[str] = ("-c",)
    wrap: Callable[[str], str] = lambda command: command
    join: Callable[[list[str]], str] = shlex.join

    @classmethod
    def get_shell(cls) -> Self:
        """Resolve this platform's shell: bash/sh on PATH, else PowerShell on Windows, else the default flavor."""
        for candidate in ("bash", "sh"):
            path = shutil.which(candidate.lower())
            if not path or ("system32" in path.lower() and os.name == "nt"):
                continue
            return cls(path=path)
        for candidate in ("pwsh", "powershell"):
            path = shutil.which(candidate.lower())
            if path and os.name == "nt":
                return cls(
                    name=candidate,
                    snippet="Execute PowerShell commands",
                    prompt="PS>",
                    path=path,
                    args=["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"],
                    wrap=lambda command: (
                        "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\n" + command
                    ),
                    join=_join_powershell,
                )
        return cls()


def _join_powershell(command: list[str]) -> str:
    quoted = " ".join("'" + part.replace("'", "''") + "'" for part in command)
    return f"& {quoted}; exit $LASTEXITCODE"


class _ShellResult(NamedTuple):
    """One finished shell run: tail-truncated text plus status and truncation details."""

    text: str
    exit_code: int
    aborted: bool = False
    timed_out: bool = False
    overflowed: bool = False
    truncated: bool = False
    full_output_path: str | None = None
    wall_time: float = 0.0
    timeout: float | None = None
    truncation: TruncationResult | None = None


def _live_preview(buffer: bytearray) -> str:
    """Decode only the tail needed for a throttled UI update."""
    tail = bytes(buffer[-_UPDATE_PREVIEW_BYTES:])
    while tail and (tail[0] & 0xC0) == 0x80:  # do not start mid-codepoint
        tail = tail[1:]
    return tail.decode("utf-8", "replace")


def _run_shell_command(
    command: str,
    *,
    argv: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: float | None = None,
    signal: AbortSignal | None = None,
    on_update: UpdateSink | None = None,
    prefix: str = "tynex",
) -> _ShellResult:
    started = time.monotonic()
    args = [*argv, command]
    proc = subprocess.Popen(
        args,  # detached (own process group) so kill_process_tree() reaches the whole tree
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **detached_options(),
    )
    chunks: queue.Queue[bytes | None] = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        with contextlib.suppress(Exception):
            while data := proc.stdout.read1(65536):
                chunks.put(data)
        chunks.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    buffer, overflowed, aborted, timed_out, last_update = bytearray(), False, False, False, 0.0
    deadline = started + timeout if timeout else None
    while True:
        if signal is not None and signal.aborted:
            aborted = True
        elif deadline is not None and time.monotonic() > deadline:
            timed_out = True
        else:
            try:
                data = chunks.get(timeout=0.1)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if data is None:
                break
            if not overflowed:
                buffer.extend(data)
                overflowed = len(buffer) > _MAX_OUTPUT_BYTES
            if on_update is not None and time.monotonic() - last_update >= 0.15:  # throttle UI updates
                last_update = time.monotonic()
                on_update(ToolResult.text(truncate_tail(_live_preview(buffer)).content))
            continue
        kill_process_tree(proc)
        break
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
    thread.join(timeout=2)
    # Close the pipe ourselves: leaving stdout to the Popen finalizer produces
    # ResourceWarnings (and keeps a reader thread blocked after a kill).
    if proc.stdout is not None:
        with contextlib.suppress(OSError):
            proc.stdout.close()
    full_text = buffer.decode("utf-8", "replace")
    code = proc.returncode if proc.returncode is not None else 1
    exit_code = 128 + abs(code) if code < 0 else code  # signal convention
    truncation = truncate_tail(full_text)
    tmp: str | None = None
    if truncation.truncated:
        fd, tmp = tempfile.mkstemp(prefix=f"{prefix}-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(full_text)
    return _ShellResult(
        text=truncation.content or "",
        exit_code=exit_code,
        aborted=aborted,
        timed_out=timed_out,
        overflowed=overflowed,
        truncated=truncation.truncated,
        full_output_path=tmp,
        wall_time=round(time.monotonic() - started, 1),
        timeout=timeout,
        truncation=truncation,
    )


def _format_shell_result(result: _ShellResult) -> ToolResult:
    text, trunc = result.text, result.truncation
    if trunc is not None and trunc.truncated:
        limit = "" if trunc.truncated_by == "lines" else f" ({format_size(DEFAULT_MAX_BYTES)} limit)"
        shown = (
            f"last {format_size(trunc.output_bytes)} of line {trunc.total_lines}"
            if trunc.last_line_partial
            else f"lines {trunc.total_lines - trunc.output_lines + 1}-{trunc.total_lines} of {trunc.total_lines}{limit}"
        )
        if result.overflowed:
            text += f"\n\n[Showing {shown}. Only the first {format_size(_MAX_OUTPUT_BYTES)} was captured.]"
        else:
            text += f"\n\n[Showing {shown}. Full output: {result.full_output_path}]"
    details: dict[str, Any] = {"fullOutputPath": result.full_output_path} if result.full_output_path else {}
    details.update(truncation=result.truncated, wallTimeSeconds=result.wall_time, exitCode=result.exit_code)
    if result.aborted:
        status = "Command aborted"
    elif result.timed_out:
        status = f"Command timed out after {result.timeout:g} seconds"
    elif result.overflowed:
        captured = format_size(_MAX_OUTPUT_BYTES)
        status = (
            f"Output exceeded {captured}; the first {captured} was saved to {result.full_output_path}"
            if result.full_output_path
            else f"Output exceeded {captured}; only the first {captured} was captured"
        )
    elif result.exit_code != 0:
        status = f"Command exited with code {result.exit_code}"
    else:
        status = ""
    if status:
        text = f"{text}\n\n{status}" if text else status
    elif not text:
        text = "(no output)"
    return ToolResult.text(text, is_error=bool(status), details=details)


SHELL = _ShellFlavor.get_shell()


class ShellTool(AgentTool):
    name = SHELL.name
    label = SHELL.name
    permission_kind = "shell"
    mutating = True
    description = (
        f"Execute a {SHELL.name.title()} command in the current working directory. Returns stdout and stderr. "
        f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
        "(whichever is hit first). If truncated, full output is saved to a temp file."
    )
    prompt_snippet = SHELL.snippet
    prompt_guidelines: ClassVar[list[str]] = [
        "You can inspect TYNEX_* environment variables for current model and session details."
    ]
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"command": {"type": "string", "description": f"{SHELL.name.title()} command to execute"}},
        "required": ["command"],
    }

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd
        self.flavor = SHELL
        self.prefix = f"tynex-{SHELL.name}"

    def _spawn_env(self, ctx: ToolContext | None) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key not in _TYNEX_ENV_KEYS}
        if ctx is not None:
            values = (ctx.session_id, ctx.session_file, ctx.provider, ctx.model, ctx.thinking_level)
            env.update((key, value) for key, value in zip(_TYNEX_ENV_KEYS, values, strict=True) if value)
        return env

    def run_capture(
        self,
        command: str,
        *,
        timeout: float | None = None,
        signal: AbortSignal | None = None,
        on_update: UpdateSink | None = None,
        ctx: ToolContext | None = None,
    ) -> _ShellResult:
        """Run one command; shared by the tool itself and the ``!`` input mode."""
        timeout = min(timeout or 30 * 60, 30 * 60)  # cap every run at 30 minutes
        cwd = ctx.cwd if ctx else self.cwd
        if not os.path.isdir(cwd):
            raise NotADirectoryError(
                f"Working directory does not exist: {cwd}\nCannot execute {self.flavor.name.title()} commands."
            )
        if signal is not None:
            signal.throw_if_aborted()
        cmd, argv, env = self.flavor.wrap(command), [self.flavor.path, *self.flavor.args], self._spawn_env(ctx)
        return _run_shell_command(
            cmd, argv=argv, cwd=cwd, env=env, timeout=timeout, signal=signal, on_update=on_update, prefix=self.prefix
        )

    def execute(
        self,
        tool_call_id: str,
        args: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: UpdateSink | None = None,
        ctx: ToolContext | None = None,
    ) -> ToolResult:
        result = self.run_capture(str(args.get("command", "")), signal=signal, on_update=on_update, ctx=ctx)
        return _format_shell_result(result)


shell_prompt = SHELL.prompt
shell_name = SHELL.name
