"""Loading ``permissions.toml`` (global + project) into policy rules."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from tynex.agent.permissions import PermissionRule, normalize_mode
from tynex.coding.settings import config_dir, find_files

RULES_FILENAME = "permissions.toml"
ACTIONS = ("allow", "ask", "deny")


@dataclass
class PermissionConfig:
    mode: str | None = None
    rules: list[PermissionRule] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


def config_paths(cwd: str | Path) -> list[Path]:
    """Global rules first, then project rules (project files win — evaluated later)."""
    candidates: list[Path] = [config_dir() / RULES_FILENAME]
    found = find_files(RULES_FILENAME, cwd)
    candidates.extend(found if isinstance(found, list) else [found])
    seen: set[str] = set()
    paths: list[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        paths.append(path)
    return paths


def load_permission_config(cwd: str | Path) -> PermissionConfig:
    """Parse every rules file; invalid entries become warnings, never crashes."""
    config = PermissionConfig()
    for path in config_paths(cwd):
        config.files.append(str(path))
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            config.warnings.append(f"{path}: {exc}")
            continue
        configured_mode = data.get("mode")
        if isinstance(configured_mode, str):
            try:
                config.mode = normalize_mode(configured_mode)
            except ValueError as exc:
                config.warnings.append(f"{path}: {exc}")
        entries = data.get("rules")
        if entries is None:
            continue
        if not isinstance(entries, list):
            config.warnings.append(f"{path}: 'rules' must be a list of tables")
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                config.warnings.append(f"{path}: skipping non-table rule {entry!r}")
                continue
            action = str(entry.get("action", "")).strip().lower()
            if action not in ACTIONS:
                config.warnings.append(f"{path}: rule action must be one of {', '.join(ACTIONS)} — got {action!r}")
                continue
            tool = str(entry.get("tool") or "*")
            match = entry.get("match")
            config.rules.append(
                PermissionRule(
                    tool=tool,
                    match=str(match) if match is not None else None,
                    action=action,  # type: ignore[arg-type] — validated above
                    source=str(path),
                )
            )
    return config
