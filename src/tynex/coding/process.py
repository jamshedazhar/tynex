"""Shared child-process management for shell commands and MCP stdio servers.

Both spawn long-running children that fork helpers of their own (bash
pipelines, ``npx`` shims), so both need the same two things: spawn the child
in its own process group, and kill the whole tree on teardown.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from typing import Any


def detached_options() -> dict[str, Any]:
    """Popen options that put a child in its own group so a kill reaches its tree."""
    return {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}


def kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Force-kill ``proc`` and everything it spawned; needs ``detached_options()``; never raises."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        with contextlib.suppress(Exception):
            proc.kill()
