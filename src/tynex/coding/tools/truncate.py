"""Tool-output truncation helpers."""

from dataclasses import dataclass

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
GREP_MAX_LINE_LENGTH = 500


@dataclass
class TruncationResult:
    content: str
    truncated: bool = False
    truncated_by: str | None = None  # "lines" | "bytes" | None
    total_lines: int = 0
    total_bytes: int = 0
    output_lines: int = 0
    output_bytes: int = 0
    last_line_partial: bool = False
    first_line_exceeds_limit: bool = False
    max_lines: int = DEFAULT_MAX_LINES
    max_bytes: int = DEFAULT_MAX_BYTES


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def _split_lines_for_counting(content: str) -> list[str]:
    if content == "":
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _untouched(content: str, lines: int, max_lines: int, max_bytes: int) -> TruncationResult:
    total_bytes = len(content.encode("utf-8"))
    return TruncationResult(
        content=content,
        truncated=False,
        total_lines=lines,
        total_bytes=total_bytes,
        output_lines=lines,
        output_bytes=total_bytes,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_head(
    content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> TruncationResult:
    total_bytes = len(content.encode("utf-8"))
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return _untouched(content, total_lines, max_lines, max_bytes)
    first_line_bytes = len(lines[0].encode("utf-8")) if lines else 0
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )
    output: list[str] = []
    output_bytes_count = 0
    truncated_by = "lines"
    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        line_bytes = len(line.encode("utf-8")) + (1 if i > 0 else 0)
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output.append(line)
        output_bytes_count += line_bytes
    if len(output) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"
    output_content = "\n".join(output)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output),
        output_bytes=len(output_content.encode("utf-8")),
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def _truncate_string_to_bytes_from_end(s: str, max_bytes: int) -> str:
    data = s.encode("utf-8")
    if len(data) <= max_bytes:
        return s
    start = len(data) - max_bytes
    while start < len(data) and (data[start] & 0xC0) == 0x80:
        start += 1
    return data[start:].decode("utf-8", errors="replace")


def truncate_tail(
    content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> TruncationResult:
    total_bytes = len(content.encode("utf-8"))
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return _untouched(content, total_lines, max_lines, max_bytes)
    output: list[str] = []
    output_bytes_count = 0
    truncated_by = "lines"
    last_line_partial = False
    for i in range(len(lines) - 1, -1, -1):
        if len(output) >= max_lines:
            break
        line = lines[i]
        line_bytes = len(line.encode("utf-8")) + (1 if output else 0)
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output.insert(0, truncated_line)
                output_bytes_count = len(truncated_line.encode("utf-8"))
                last_line_partial = True
            break
        output.insert(0, line)
        output_bytes_count += line_bytes
    if len(output) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"
    output_content = "\n".join(output)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output),
        output_bytes=len(output_content.encode("utf-8")),
        last_line_partial=last_line_partial,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    if len(line) <= max_chars:
        return line, False
    return f"{line[:max_chars]}... [truncated]", True
