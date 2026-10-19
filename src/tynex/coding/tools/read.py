"""read tool — read file contents, supports text and images."""

import base64
import os
import shlex
from typing import Any, ClassVar

from tynex.agent.types import AgentTool, ToolContext, ToolResult
from tynex.ai.types import ImageContent, TextContent
from tynex.coding.tools.path_utils import resolve_path
from tynex.coding.tools.shell import ShellTool
from tynex.coding.tools.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head

IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
MAX_IMAGE_BYTES = 5 * 1024 * 1024

SNIPPET = "Read file contents"
GUIDELINES = ["Use read to examine files instead of cat or sed."]


def detect_image_mime(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    return IMAGE_MIME_TYPES.get(ext)


class ReadTool(AgentTool):
    name = "read"
    permission_kind = "read"
    label = "read"
    description = (
        "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
        "Images are sent as attachments. For text files, output is truncated to "
        f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
        "Use offset/limit for large files. When you need the full file, continue with offset until complete."
    )
    prompt_snippet = SNIPPET
    prompt_guidelines: ClassVar[list[str]] = list(GUIDELINES)
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
            "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "number", "description": "Maximum number of lines to read"},
        },
        "required": ["path"],
    }

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def execute(
        self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None, ctx: ToolContext | None = None
    ) -> ToolResult:
        cwd = ctx.cwd if ctx else self.cwd
        path_arg = str(args.get("path", ""))
        absolute = resolve_path(path_arg, cwd)
        offset = args.get("offset")
        limit = args.get("limit")
        if signal is not None:
            signal.throw_if_aborted()
        if not os.path.exists(absolute):
            raise FileNotFoundError(f"File not found: {path_arg}")
        if os.path.isdir(absolute):
            raise IsADirectoryError(f"{path_arg} is a directory. Use the ls tool (or bash) to list directories.")
        mime = detect_image_mime(absolute)
        if mime:
            size = os.path.getsize(absolute)
            if size > MAX_IMAGE_BYTES:
                return ToolResult.text(
                    f"Read image file [{mime}]\nImage is {format_size(size)}, exceeding the 5.0MB limit. "
                    "Resize it or use bash to process it."
                )
            with open(absolute, "rb") as fh:
                data = base64.b64encode(fh.read()).decode("ascii")
            return ToolResult(
                content=[TextContent(text=f"Read image file [{mime}]"), ImageContent(data=data, mime_type=mime)]
            )
        start_line = max(0, int(offset) - 1) if offset else 0
        start_line_display = start_line + 1
        limit_lines = max(0, int(limit)) if limit is not None else None
        # Stream the file instead of materializing it: only the head of the
        # requested window is ever shown, so large files stay bounded.
        selected_lines: list[str] = []
        selected_bytes = 0
        total_file_lines = 0
        with open(absolute, encoding="utf-8", errors="replace", newline=None) as fh:
            for raw_line in fh:
                total_file_lines += 1
                if total_file_lines <= start_line:
                    continue
                if limit_lines is not None and total_file_lines > start_line + limit_lines:
                    continue
                # Keep one line past the output limit so truncate_head() can
                # still report that the file continues.
                if len(selected_lines) <= DEFAULT_MAX_LINES and selected_bytes <= DEFAULT_MAX_BYTES:
                    line = raw_line[:-1] if raw_line.endswith("\n") else raw_line
                    selected_lines.append(line)
                    selected_bytes += len(line.encode("utf-8")) + 1
        if total_file_lines and start_line >= total_file_lines:
            raise ValueError(f"Offset {offset} is beyond end of file ({total_file_lines} lines total)")
        if limit_lines is not None:
            user_limited_lines = max(0, min(start_line + limit_lines, total_file_lines) - start_line)
        else:
            user_limited_lines = None
        selected = "\n".join(selected_lines)
        truncation = truncate_head(selected)
        details: dict[str, Any] | None = None
        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(selected_lines[0].encode("utf-8")))
            if ShellTool.name == "bash":
                path_hint = shlex.quote(path_arg)
                hint = f"Use bash: sed -n '{start_line_display}p' {path_hint} | head -c {DEFAULT_MAX_BYTES}"
            else:
                hint = (
                    f"Use powershell: Get-Content '{path_arg}' | Select-Object -Skip {start_line_display - 1} -First 1"
                )
            output = (
                f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                f"{hint}]"
            )
            details = {"truncation": True}
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = end_line_display + 1
            output = truncation.content
            if truncation.truncated_by == "lines":
                output += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                )
            details = {"truncation": True}
        elif user_limited_lines is not None and start_line + user_limited_lines < total_file_lines:
            remaining = total_file_lines - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            notice = f"[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            output = f"{truncation.content}\n\n{notice}" if truncation.content else notice
        else:
            output = truncation.content
        return ToolResult.text(output, details=details)
