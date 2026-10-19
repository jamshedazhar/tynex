"""Embed Tynex in your own program: custom tools, streaming, budgets, approvals.

Run it against any configured provider — ``--model`` takes the same
``provider/model-id`` references as the CLI and resolves them against your
``settings.json``/``auth.json``, so a ``tynex login`` is all the setup needed:

    uv run python examples/sdk_agent.py
    uv run python examples/sdk_agent.py --model openai/gpt-5.2 "count the words in README.md"
    uv run python examples/sdk_agent.py --budget cost=0.25 --permission-mode plan "propose a refactor"

The script shows the four moving parts of the SDK in one place:

* ``@tool``      — a sync and an async function exposed to the model
* ``coding_tools`` — Tynex's built-in read/shell tools alongside them
* ``approval``   — a permission callback, so shell calls must be approved
* ``budget``     — opt-in token/cost caps, so an unattended script cannot overspend
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from tynex.agent.budget import parse_budget
from tynex.agent.permissions import PermissionRequest
from tynex.coding.tools import shell_name
from tynex.sdk import Agent, tool

DEFAULT_PROMPT = (
    "Inspect this repository: list the Python files, count the words in README.md, "
    "and summarize what the entry point does. Then describe the newest commit."
)


@tool(read_only=True)
def word_count(path: str) -> dict:
    """Count the words and lines in a file inside the working directory.

    Args:
        path: Workspace-relative path of the file to measure.
    """
    text = (Path.cwd() / path).read_text(encoding="utf-8")
    return {"path": path, "lines": len(text.splitlines()), "words": len(text.split())}


@tool(read_only=True)
async def latest_commit(repo: str = ".") -> str:
    """Return the one-line summary of the newest git commit in a repository.

    Args:
        repo: Repository path, relative to the working directory.
    """
    process = await asyncio.create_subprocess_exec(
        "git", "-C", repo, "log", "-1", "--pretty=%h %s", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        return f"git failed: {stderr.decode(errors='replace').strip()}"
    return stdout.decode(errors="replace").strip()


def approve(request: PermissionRequest) -> str:
    """Allow read-only shell commands for the session; deny everything else."""
    if request.kind == "shell" and request.subject.startswith(("git ", "ls", "cat ")):
        return "session"
    print(f"[approval] denied: {request.summary()}", file=sys.stderr)
    return "deny"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tynex SDK example")
    parser.add_argument("-m", "--model", default="anthropic/claude-sonnet-4-5", help="provider/model-id to use")
    parser.add_argument("--budget", default="cost=0.50", help="cap: cost=USD or tokens=N")
    parser.add_argument("--permission-mode", default="ask", help="plan, ask, accept-edits or bypass")
    parser.add_argument("prompt", nargs="*", help="prompt to send (defaults to a repository tour)")
    args = parser.parse_args(argv)
    try:
        budget = parse_budget(args.budget)
    except ValueError as exc:
        parser.error(str(exc))
    prompt = " ".join(args.prompt) or DEFAULT_PROMPT

    def log(event: dict) -> None:
        if event.get("type") == "tool_execution_start":
            print(f"  -> {event.get('toolName')}", file=sys.stderr)

    agent = Agent(
        model=args.model,
        tools=[word_count, latest_commit],
        coding_tools=["read", shell_name],
        thinking="low",
        permission_mode=args.permission_mode,
        approval=approve,
        budget=budget,
        on_event=log,
    )
    with agent:
        try:
            for event in agent.stream(prompt):
                if event.get("type") == "message_update":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        sys.stdout.write(delta.get("delta") or "")
                        sys.stdout.flush()
        except ValueError as exc:  # no model configured
            print(f"tynex: {exc}", file=sys.stderr)
            return 2

    usage = agent.usage
    print()
    print(
        f"[{agent.model.key if agent.model else '?'}] "
        f"{agent.steps} model call(s), {agent.tool_calls} tool call(s), "
        f"{usage.input + usage.output} tokens, ${usage.cost:.4f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
