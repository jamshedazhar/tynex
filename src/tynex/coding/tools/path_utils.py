"""Shared path helpers for tools."""

import os


class WorkspaceBoundaryError(ValueError):
    """A tool path resolved outside the working directory."""


def resolve_path(path: str, cwd: str) -> str:
    """Resolve a tool path against ``cwd``, require it inside, and return the resolved target."""
    if not path:
        raise ValueError("Path must not be empty")
    expanded = os.path.expanduser(path)
    absolute = os.path.normpath(expanded if os.path.isabs(expanded) else os.path.join(cwd, expanded))
    root = os.path.realpath(os.path.abspath(cwd))
    target = os.path.realpath(absolute)
    try:
        inside = os.path.commonpath([os.path.normcase(root), os.path.normcase(target)]) == os.path.normcase(root)
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        raise WorkspaceBoundaryError(f"Access denied: {path!r} is outside the working directory ({cwd})")
    return target
