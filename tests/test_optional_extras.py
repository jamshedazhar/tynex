"""The base install must stay lean: the SDK imports without the CLI/TUI extras.

These tests block the optional third-party modules (`click`, `rich`,
`prompt_toolkit`) at the import level, so any regression that re-introduces an
eager import fails loudly. Tools that genuinely need an extra must raise the
actionable ``tynex[cli]`` hint instead of an ImportError.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import tynex

REPO = Path(__file__).resolve().parent.parent
SRC = Path(tynex.__file__).resolve().parent.parent
EXTRAS_HINT = "tynex[cli]"

# Installed before the code under test; raises ModuleNotFoundError for anything
# whose top-level name is in BLOCKED, mimicking a base (SDK-only) install.
_BLOCKER = """
import sys

BLOCKED = {blocked}

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("blocked: " + name, name=name)
        return None

sys.meta_path.insert(0, Blocker())
"""


def run_blocked(code: str, blocked: set[str]) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter with the named modules unavailable."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH", "")]))
    env["PYTHONIOENCODING"] = "utf-8"
    script = _BLOCKER.format(blocked=repr(blocked)) + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env=env,
        cwd=str(REPO),
    )


def test_sdk_imports_without_cli_extras():
    result = run_blocked(
        """
        import tynex.sdk
        from tynex.sdk import Agent, BudgetLimits, tool

        @tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b

        print("sdk ok")
        """,
        {"click", "rich", "prompt_toolkit"},
    )
    assert result.returncode == 0, result.stderr
    assert "sdk ok" in result.stdout


def test_cli_reports_missing_extras():
    result = run_blocked("import tynex.cli", {"click"})
    assert result.returncode == 1
    assert "click" in result.stderr
    assert EXTRAS_HINT in result.stderr
