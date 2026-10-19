"""File checkpoints: snapshot what a tool is about to touch, restore with /undo.

Every file a mutating tool is about to change is copied (bounded by size) into
``~/.tynex/agent/checkpoints/<session>/<n>/`` before the tool runs. ``/undo``
restores the most recent groups, so a bad edit, a wrong rewrite or a runaway
refactor is one command away — no git required and no repository state touched.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from tynex.coding.settings import config_dir
from tynex.coding.tools.path_utils import WorkspaceBoundaryError, resolve_path

MAX_SNAPSHOT_BYTES = 5 * 1024 * 1024  # larger files are recorded but not restorable


@dataclass
class CheckpointFile:
    path: str  # workspace-relative, forward slashes
    action: str  # modified | created | skipped
    size: int = 0

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "action": self.action, "size": self.size}

    @staticmethod
    def from_dict(data: dict) -> CheckpointFile:
        return CheckpointFile(
            str(data.get("path", "")), str(data.get("action", "modified")), int(data.get("size") or 0)
        )


@dataclass
class CheckpointGroup:
    id: int
    tool: str
    created: float
    files: list[CheckpointFile] = field(default_factory=list)
    undone: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "tool": self.tool,
            "created": self.created,
            "files": [f.to_dict() for f in self.files],
            "undone": self.undone,
        }

    @staticmethod
    def from_dict(data: dict) -> CheckpointGroup:
        return CheckpointGroup(
            id=int(data.get("id") or 0),
            tool=str(data.get("tool", "")),
            created=float(data.get("created") or 0.0),
            files=[CheckpointFile.from_dict(entry) for entry in data.get("files") or []],
            undone=bool(data.get("undone")),
        )

    def summary(self) -> str:
        paths = ", ".join(f.path for f in self.files[:3]) or "(no files)"
        extra = f" +{len(self.files) - 3} more" if len(self.files) > 3 else ""
        return f"#{self.id} {self.tool}: {paths}{extra}"


class CheckpointStore:
    """Per-session snapshot store; safe to call from the tool-execution thread."""

    def __init__(self, cwd: str, namespace: str, enabled: bool = True) -> None:
        self.cwd = os.path.abspath(cwd)
        self.namespace = namespace
        self.enabled = enabled
        self.warnings: list[str] = []
        self.root = config_dir() / "checkpoints" / namespace
        self._groups: list[CheckpointGroup] | None = None

    # -- reading ------------------------------------------------------------
    def groups(self) -> list[CheckpointGroup]:
        if self._groups is None:
            self._groups = self._read_groups()
        return self._groups

    def _read_groups(self) -> list[CheckpointGroup]:
        groups: list[CheckpointGroup] = []
        if not self.root.is_dir():
            return groups
        for manifest in sorted(self.root.glob("*/manifest.json")):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                groups.append(CheckpointGroup.from_dict(data))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
        groups.sort(key=lambda group: group.id)
        return groups

    def _write_manifest(self, group: CheckpointGroup) -> None:
        group_dir = self.root / f"{group.id:04d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "manifest.json").write_text(json.dumps(group.to_dict(), indent=2), encoding="utf-8")

    def _next_id(self) -> int:
        return max((group.id for group in self.groups()), default=0) + 1

    def _snapshot_dir(self, group_id: int) -> Path:
        return self.root / f"{group_id:04d}" / "files"

    def _absolute(self, relative: str) -> Path:
        return Path(self.cwd, *relative.split("/"))

    # -- capturing ----------------------------------------------------------
    def capture(self, tool_name: str, paths: list[str]) -> CheckpointGroup | None:
        """Snapshot ``paths`` before a tool runs. Never raises."""
        if not self.enabled:
            return None
        files: list[CheckpointFile] = []
        group_id = self._next_id()
        try:
            for raw in paths:
                if not isinstance(raw, str) or not raw:
                    continue
                try:
                    absolute = resolve_path(raw, self.cwd)
                except (WorkspaceBoundaryError, ValueError):
                    continue  # outside the workspace: nothing to snapshot
                relative = os.path.relpath(absolute, self.cwd).replace(os.sep, "/")
                if not os.path.exists(absolute):
                    files.append(CheckpointFile(relative, "created"))
                    continue
                if not os.path.isfile(absolute):
                    continue
                size = os.path.getsize(absolute)
                if size > MAX_SNAPSHOT_BYTES:
                    files.append(CheckpointFile(relative, "skipped", size))
                    continue
                target = self._snapshot_dir(group_id) / Path(*relative.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(absolute, target)
                files.append(CheckpointFile(relative, "modified", size))
            if not files:
                return None
            group = CheckpointGroup(id=group_id, tool=tool_name, created=time.time(), files=files)
            self._write_manifest(group)
            self._groups = [*self.groups(), group]
            return group
        except OSError as exc:
            self.warnings.append(f"checkpoint failed: {exc}")
            return None

    # -- restoring ----------------------------------------------------------
    def undo(self, count: int = 1) -> list[CheckpointGroup]:
        """Restore the most recent ``count`` groups (newest first)."""
        if not self.enabled:
            return []
        pending = [group for group in self.groups() if not group.undone]
        if not pending:
            return []
        restored: list[CheckpointGroup] = []
        for group in reversed(pending[-max(1, count) :]):
            try:
                for entry in group.files:
                    absolute = self._absolute(entry.path)
                    snapshot = self._snapshot_dir(group.id) / Path(*entry.path.split("/"))
                    if entry.action == "created":
                        if absolute.exists():
                            absolute.unlink()
                    elif entry.action == "modified" and snapshot.exists():
                        absolute.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(snapshot, absolute)
            except OSError as exc:
                self.warnings.append(f"undo failed for group #{group.id}: {exc}")
            group.undone = True
            self._write_manifest(group)
            shutil.rmtree(self._snapshot_dir(group.id), ignore_errors=True)
            restored.append(group)
        return restored
