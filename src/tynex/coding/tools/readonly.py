"""Read-only tools: grep, find, and ls."""

import fnmatch
import os
import re
from typing import Any, ClassVar

import ripgrep_rs as rg

from tynex.agent.types import AgentTool, ToolContext, ToolResult
from tynex.coding.tools.path_utils import resolve_path
from tynex.coding.tools.truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    format_size,
    truncate_head,
    truncate_line,
)

DEFAULT_GREP_LIMIT = 100
MAX_FIND_RESULTS = 1000
MAX_LS_ENTRIES = 500
# Rows are already capped by the match/result/entry limits, so only the byte
# limit applies (these outputs are tail-truncated, passing
# Number.MAX_SAFE_INTEGER for the line limit).
NO_LINE_LIMIT = 2**53 - 1

# Rust regex metacharacters: escaping them turns a pattern into a literal.
_LITERAL_CHARS = re.compile(r"([\\.^$*+?()\[\]{}|])")


def _literal_pattern(pattern: str) -> str:
    return _LITERAL_CHARS.sub(r"\\\1", pattern)


def _find_glob(pattern: str) -> str:
    if "/" in pattern and not pattern.startswith("/") and not pattern.startswith("**/") and pattern != "**":
        return f"**/{pattern}"
    return pattern


def _matches_find_glob(path: str, pattern: str, root: str) -> bool:
    """Match ``path`` against a find pattern; ``**/`` matches at any depth, including none."""
    relative = _relativize_find_path(path, root)
    if "/" not in pattern:
        return fnmatch.fnmatchcase(os.path.basename(relative), pattern)
    return fnmatch.fnmatchcase(relative, pattern) or fnmatch.fnmatchcase(relative, pattern.removeprefix("**/"))


def _check_abort(signal: Any) -> None:
    if signal is not None:
        signal.throw_if_aborted()


def _display_path(path: str, search_root: str, is_directory: bool) -> str:
    if is_directory:
        try:
            relative = os.path.relpath(path, search_root)
        except ValueError:  # path and root are on different Windows drives
            relative = ""
        if relative and not relative.startswith(".."):
            return relative.replace("\\", "/")
    return os.path.basename(path)


def _relativize_find_path(path: str, search_root: str) -> str:
    if os.path.isabs(path):
        try:
            relative = os.path.relpath(path, search_root)
        except ValueError:  # path and root are on different Windows drives
            relative = path
        path = os.path.basename(path) if relative == "." else relative
    return path.replace("\\", "/")


def _read_file_lines(path: str) -> list[str] | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError:
        return None
    return content.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _resolve_root(args: dict[str, Any], cwd: str, signal: Any) -> str:
    root = resolve_path(str(args.get("path", ".")), cwd)
    if not os.path.exists(root):
        raise FileNotFoundError(f"Path not found: {root}")
    _check_abort(signal)
    return root


def _finish(
    lines: list[str], notices: list[str], details: dict[str, Any], extra: list[str] | None = None
) -> ToolResult:
    truncation = truncate_head("\n".join(lines), max_lines=NO_LINE_LIMIT)
    if truncation.truncated:
        notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        details["truncation"] = True
    notices.extend(extra or [])
    text = truncation.content + (f"\n\n[{'. '.join(notices)}]" if notices else "")
    return ToolResult.text(text, details=details or None)


class GrepTool(AgentTool):
    name = "grep"
    permission_kind = "read"
    label = "grep"
    description = (
        "Search file contents for a pattern. Returns matching lines with file paths and line numbers. "
        f"Respects .gitignore. Output is truncated to {DEFAULT_GREP_LIMIT} matches or "
        f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Long lines are truncated to "
        f"{GREP_MAX_LINE_LENGTH} chars."
    )
    prompt_snippet = "Search file contents for patterns (respects .gitignore)"
    prompt_guidelines: ClassVar[list[str]] = ["Use grep to search file contents instead of running grep via bash."]
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
            "path": {"type": "string", "description": "File or directory to search in (default: cwd)"},
            "glob": {"type": "string", "description": "Filter files by glob, e.g. '*.py' or '**/*.ts'"},
            "ignoreCase": {"type": "boolean", "description": "Case-insensitive search"},
            "literal": {"type": "boolean", "description": "Treat pattern as a literal string instead of a regex"},
            "context": {"type": "integer", "description": "Lines of context to show before and after each match"},
            "limit": {
                "type": "integer",
                "description": f"Maximum number of matches to return (default {DEFAULT_GREP_LIMIT})",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def execute(
        self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None, ctx: ToolContext | None = None
    ) -> ToolResult:
        cwd = ctx.cwd if ctx else self.cwd
        raw_pattern = str(args.get("pattern", ""))
        if not raw_pattern:
            raise ValueError("pattern must not be empty")
        root = _resolve_root(args, cwd, signal)
        pattern = _literal_pattern(raw_pattern) if args.get("literal") else raw_pattern
        glob = args.get("glob")
        context = max(0, int(args.get("context") or 0))
        limit = max(1, int(args.get("limit") or DEFAULT_GREP_LIMIT))
        is_directory = os.path.isdir(root)
        matches = rg.search_structured(
            [pattern],
            paths=[root],
            globs=[str(glob)] if glob else None,
            sort=rg.PySortMode(rg.PySortModeKind.Path),
            max_total=limit + 1,  # one extra tells us whether the limit was hit
            case_sensitive=False if args.get("ignoreCase") else None,
            hidden=True,
        )
        _check_abort(signal)
        limit_reached = len(matches) > limit
        matches = matches[:limit]
        if not matches:
            return ToolResult.text("No matches found")
        output_lines: list[str] = []
        lines_truncated = False
        file_cache: dict[str, list[str] | None] = {}
        match_lines: dict[str, set[int]] = {}
        for match in matches:
            match_lines.setdefault(match.path, set()).add(match.line_number)
        emitted_through: dict[str, int] = {}
        for match in matches:
            relative = _display_path(match.path, root, is_directory)
            if context == 0:
                line_text = match.line_text.replace("\r\n", "\n").replace("\r", "")
                if line_text.endswith("\n"):
                    line_text = line_text[:-1]
                text, was_truncated = truncate_line(line_text)
                lines_truncated = lines_truncated or was_truncated
                output_lines.append(f"{relative}:{match.line_number}: {text}")
                continue
            if match.path not in file_cache:
                file_cache[match.path] = _read_file_lines(match.path)
            lines = file_cache[match.path]
            if not lines:
                output_lines.append(f"{relative}:{match.line_number}: (unable to read file)")
                continue
            start = max(1, match.line_number - context)
            end = min(len(lines), match.line_number + context)
            first = max(start, emitted_through.get(match.path, 0) + 1)
            emitted_through[match.path] = max(end, emitted_through.get(match.path, 0))
            for current in range(first, end + 1):
                line_text = lines[current - 1] if current - 1 < len(lines) else ""
                text, was_truncated = truncate_line(line_text)
                lines_truncated = lines_truncated or was_truncated
                separator = ":" if current in match_lines[match.path] else "-"
                output_lines.append(f"{relative}{separator}{current}{separator} {text}")
        notices: list[str] = []
        details: dict[str, Any] = {}
        extra: list[str] = []
        if limit_reached:
            notices.append(f"{limit} matches limit reached. Use limit={limit * 2} for more, or refine pattern")
            details["matchLimitReached"] = limit
        if lines_truncated:
            extra.append(f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines")
            details["linesTruncated"] = True
        return _finish(output_lines, notices, details, extra)


class FindTool(AgentTool):
    name = "find"
    permission_kind = "read"
    label = "find"
    description = (
        "Search for files by glob pattern. Returns matching file paths relative to the search directory. "
        f"Respects .gitignore. Output is truncated to {MAX_FIND_RESULTS} results or "
        f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
    )
    prompt_snippet = "Find files by glob pattern (respects .gitignore)"
    prompt_guidelines: ClassVar[list[str]] = [
        "Use find to locate files by name pattern instead of running find via bash."
    ]
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern to match file paths against"},
            "path": {"type": "string", "description": "Directory to search in (default: cwd)"},
            "limit": {"type": "integer", "description": f"Maximum number of results (default {MAX_FIND_RESULTS})"},
        },
        "required": ["pattern"],
    }

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def execute(
        self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None, ctx: ToolContext | None = None
    ) -> ToolResult:
        cwd = ctx.cwd if ctx else self.cwd
        pattern = str(args.get("pattern", ""))
        if not pattern:
            raise ValueError("pattern must not be empty")
        root = _resolve_root(args, cwd, signal)
        limit = max(1, int(args.get("limit") or MAX_FIND_RESULTS))
        found = rg.files(paths=[root], sort=rg.PySortMode(rg.PySortModeKind.Path), hidden=True)
        _check_abort(signal)
        glob = _find_glob(pattern)
        matches = [path for path in found if _matches_find_glob(path, glob, root)]
        limit_reached = len(matches) > limit
        results = [_relativize_find_path(path, root) for path in matches[:limit]]
        if not results:
            return ToolResult.text("No files found matching pattern")
        notices: list[str] = []
        details: dict[str, Any] = {}
        if limit_reached:
            notices.append(f"{limit} results limit reached. Use limit={limit * 2} for more, or refine pattern")
            details["resultLimitReached"] = limit
        return _finish(results, notices, details)


class LsTool(AgentTool):
    name = "ls"
    permission_kind = "read"
    label = "ls"
    description = (
        "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. "
        f"Includes dotfiles. Output is truncated to {MAX_LS_ENTRIES} entries or "
        f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
    )
    prompt_snippet = "List directory contents"
    prompt_guidelines: ClassVar[list[str]] = ["Use ls to list directory contents instead of running ls via bash."]
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list (default: cwd)"},
            "limit": {
                "type": "integer",
                "description": f"Maximum number of entries to return (default {MAX_LS_ENTRIES})",
            },
        },
    }

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def execute(
        self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None, ctx: ToolContext | None = None
    ) -> ToolResult:
        cwd = ctx.cwd if ctx else self.cwd
        root = _resolve_root(args, cwd, signal)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"Not a directory: {root}")
        limit = max(1, int(args.get("limit") or MAX_LS_ENTRIES))
        infos = rg.files_with_info(paths=[root], max_depth=1, include_dirs=True, hidden=True, no_ignore=True)
        _check_abort(signal)
        entries = sorted(infos.items(), key=lambda item: os.path.basename(item[0]).lower())
        results: list[str] = []
        entry_limit_reached = False
        for path, info in entries:
            if len(results) >= limit:
                entry_limit_reached = True
                break
            suffix = "/" if info.type == "directory" else ""
            results.append(os.path.basename(path) + suffix)
        if not results:
            return ToolResult.text("(empty directory)")
        notices: list[str] = []
        details: dict[str, Any] = {}
        if entry_limit_reached:
            notices.append(f"{limit} entries limit reached. Use limit={limit * 2} for more")
            details["entryLimitReached"] = limit
        return _finish(results, notices, details)
