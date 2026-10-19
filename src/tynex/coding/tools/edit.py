"""edit tool — precise multi-edit text replacement and diff helpers."""

import difflib
import json
import os
from dataclasses import dataclass
from typing import Any, ClassVar

from tynex.agent.types import AgentTool, ToolContext, ToolResult
from tynex.coding.tools.path_utils import resolve_path

SNIPPET = "Make precise file edits with exact text replacement, including multiple disjoint edits in one call"
GUIDELINES = [
    "Use edit for precise changes (edits[].oldText must match exactly)",
    "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] "
    "instead of multiple edit calls",
    "Each edits[].oldText is matched against the original file, not after earlier edits are applied. "
    "Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
    "Keep edits[].oldText as small as possible while still being unique in the file. "
    "Do not pad with large unchanged regions.",
]


class EditTool(AgentTool):
    name = "edit"
    label = "edit"
    permission_kind = "edit"
    mutating = True
    description = (
        "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
        "non-overlapping region of the original file. If two changes affect the same block or nearby lines, "
        "merge them into one edit instead of emitting overlapping edits. Do not include large unchanged "
        "regions just to connect distant changes."
    )
    prompt_snippet = SNIPPET
    prompt_guidelines: ClassVar[list[str]] = list(GUIDELINES)
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
            "edits": {
                "type": "array",
                "description": (
                    "One or more targeted replacements. Each edit is matched against the original file, "
                    "not incrementally. Do not include overlapping or nested edits. If two changes touch "
                    "the same block or nearby lines, merge them into one edit instead."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": (
                                "Exact text for one targeted replacement. It must be unique in the original "
                                "file and must not overlap with any other edits[].oldText in the same call."
                            ),
                        },
                        "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
                    },
                    "required": ["oldText", "newText"],
                },
            },
        },
        "required": ["path", "edits"],
    }

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def prepare_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        """Tolerate models that send a JSON string, a single object, or legacy top-level oldText/newText."""
        if not isinstance(args, dict):
            return args
        args = dict(args)
        edits = args.get("edits")
        if isinstance(edits, str):
            try:
                parsed = json.loads(edits)
                if isinstance(parsed, list):
                    args["edits"] = parsed
                elif isinstance(parsed, dict):
                    args["edits"] = [parsed]
            except json.JSONDecodeError:
                pass
        elif isinstance(edits, dict):
            args["edits"] = [edits]
        if isinstance(args.get("oldText"), str) and isinstance(args.get("newText"), str):
            existing = list(args.get("edits") or [])
            existing.append({"oldText": args.pop("oldText"), "newText": args.pop("newText")})
            args["edits"] = existing
        return args

    def execute(
        self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None, ctx: ToolContext | None = None
    ) -> ToolResult:
        cwd = ctx.cwd if ctx else self.cwd
        path_arg = str(args.get("path", ""))
        raw_edits = args.get("edits")
        if not isinstance(raw_edits, list) or not raw_edits:
            raise EditError("Edit tool input is invalid. edits must contain at least one replacement.")
        edits: list[Edit] = []
        for index, item in enumerate(raw_edits):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("oldText"), str)
                or not isinstance(item.get("newText"), str)
            ):
                raise EditError(f"edits[{index}] must be an object with string oldText and newText fields.")
            edits.append(Edit(old_text=item["oldText"], new_text=item["newText"]))
        absolute = resolve_path(path_arg, cwd)
        if signal is not None:
            signal.throw_if_aborted()
        if not os.path.exists(absolute):
            raise EditError(f"Could not edit file: {path_arg}. File does not exist.")
        if os.path.isdir(absolute):
            raise EditError(f"Could not edit file: {path_arg}. It is a directory, not a file.")
        with open(absolute, "rb") as fh:
            raw = fh.read()
        try:
            raw_content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EditError(f"Could not edit file: {path_arg}. It is not valid UTF-8 text.") from exc
        if signal is not None:
            signal.throw_if_aborted()
        bom, content = split_bom(raw_content)
        original_ending = detect_line_ending(content)
        normalized = normalize_to_lf(content)
        base_content, new_content = apply_edits_to_normalized_content(normalized, edits, path_arg)
        if signal is not None:
            signal.throw_if_aborted()
        final_content = bom + restore_line_endings(new_content, original_ending)
        with open(absolute, "w", encoding="utf-8", newline="") as fh:
            fh.write(final_content)
        diff, first_changed_line = generate_diff_string(base_content, new_content)
        patch = generate_unified_patch(path_arg, base_content, new_content)
        return ToolResult.text(
            f"Successfully replaced {len(edits)} block(s) in {path_arg}.",
            details={"diff": diff, "patch": patch, "firstChangedLine": first_changed_line},
        )


class EditError(Exception):
    pass


@dataclass
class Edit:
    old_text: str
    new_text: str


def split_bom(raw: str) -> tuple[str, str]:
    if raw.startswith("\ufeff"):
        return "\ufeff", raw[1:]
    return "", raw


def detect_line_ending(content: str) -> str:
    """Return the first line ending used by ``content`` (``\n``, ``\r\n`` or legacy ``\r``)."""
    lf = content.find("\n")
    cr = content.find("\r")
    if cr == -1:
        return "\n"
    if lf == -1 or cr < lf:
        return "\r\n" if content.startswith("\n", cr + 1) else "\r"
    return "\n"


def normalize_to_lf(content: str) -> str:
    """Normalize CRLF/CR line endings to LF for matching and diffing."""
    return content.replace("\r\n", "\n").replace("\r", "\n")


def normalize_edit_text(text: str) -> str:
    """Normalize model-supplied edit text to the LF form edits match against."""
    return normalize_to_lf(text)


def restore_line_endings(content: str, ending: str) -> str:
    if ending == "\r\n":
        return content.replace("\n", "\r\n")
    if ending == "\r":
        return content.replace("\n", "\r")
    return content


def apply_edits_to_normalized_content(content: str, edits: list[Edit], path: str) -> tuple[str, str]:
    """Returns (base_content, new_content). Raises EditError on any mismatch."""
    base = content
    if not edits:
        raise EditError("Edit tool input is invalid. edits must contain at least one replacement.")
    # Validate: every oldText must be non-empty, unique, non-overlapping.
    replacements: list[tuple[int, int, str]] = []
    consumed: list[tuple[int, int]] = []
    for i, edit in enumerate(edits):
        old_text = normalize_edit_text(edit.old_text)
        new_text = normalize_edit_text(edit.new_text)
        if old_text == "":
            raise EditError(f"edits[{i}].oldText must not be empty.")
        if old_text == new_text:
            raise EditError(f"edits[{i}].oldText and newText are identical; nothing to change.")
        first = content.find(old_text)
        if first == -1:
            preview = old_text if len(old_text) <= 200 else old_text[:200] + "..."
            raise EditError(
                f"Could not find the text for edits[{i}] in {path}. It must match the file exactly "
                f"(including whitespace), and be from the original file:\n---\n{preview}\n---"
            )
        second = content.find(old_text, first + 1)
        if second != -1:
            raise EditError(
                f"edits[{i}].oldText is not unique in {path}: it matches multiple locations. "
                "Include more surrounding context to make it unique."
            )
        span = (first, first + len(old_text))
        for other in consumed:
            if not (span[1] <= other[0] or span[0] >= other[1]):
                raise EditError(
                    f"edits[{i}].oldText overlaps with another edit's region. Merge nearby changes into one edit."
                )
        consumed.append(span)
        replacements.append((span[0], span[1], new_text))
    replacements.sort(key=lambda r: r[0], reverse=True)
    new_content = content
    for start, end, new_text in replacements:
        new_content = new_content[:start] + new_text + new_content[end:]
    return base, new_content


def generate_diff_string(old: str, new: str, context: int = 3) -> tuple[str, int | None]:
    """Produces a compact ``- old / + new`` diff with optional ``@@`` hunks."""
    old_lines = old.split("\n")
    new_lines = new.split("\n")
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=context))
    if not diff:
        return "", None
    out: list[str] = []
    first_changed: int | None = None
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            # Extract the new-file starting line for "go to first change".
            try:
                new_part = line.split("+")[1].split("@@")[0].strip()
                start_line = int(new_part.split(",")[0])
                if first_changed is None:
                    first_changed = start_line
            except (IndexError, ValueError):
                pass
            out.append(line)
        else:
            out.append(line)
    return "\n".join(out), first_changed


def generate_unified_patch(path: str, old: str, new: str) -> str:
    old_lines = old.split("\n")
    new_lines = new.split("\n")
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""))
    return "\n".join(diff)
