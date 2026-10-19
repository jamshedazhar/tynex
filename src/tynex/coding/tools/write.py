"""write tool — create or overwrite files."""

import os
from typing import Any, ClassVar

from tynex.agent.types import AgentTool, ToolContext, ToolResult
from tynex.coding.tools.path_utils import resolve_path

SNIPPET = "Create or overwrite files"
GUIDELINES = ["Use write only for new files or complete rewrites."]


class WriteTool(AgentTool):
    name = "write"
    label = "write"
    permission_kind = "edit"
    mutating = True
    description = (
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
        "Automatically creates parent directories."
    )
    prompt_snippet = SNIPPET
    prompt_guidelines: ClassVar[list[str]] = list(GUIDELINES)
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
            "content": {"type": "string", "description": "Content to write to the file"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def execute(
        self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None, ctx: ToolContext | None = None
    ) -> ToolResult:
        cwd = ctx.cwd if ctx else self.cwd
        path_arg = str(args.get("path", ""))
        content = str(args.get("content", ""))
        absolute = resolve_path(path_arg, cwd)
        if os.path.isdir(absolute):
            raise IsADirectoryError(f"Cannot write to {path_arg}: it is a directory.")
        if signal is not None:
            signal.throw_if_aborted()
        parent = os.path.dirname(absolute)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if signal is not None:
            signal.throw_if_aborted()
        absolute = resolve_path(path_arg, cwd)  # re-check after creating parents
        with open(absolute, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        return ToolResult.text(
            f"Successfully wrote to {path_arg}", details={"bytesWritten": len(content.encode("utf-8"))}
        )
