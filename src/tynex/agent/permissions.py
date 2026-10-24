"""Tool-call permissions: modes, rules, session grants and the approval gate.

The gate is deliberately UI-agnostic: the interactive front-ends install an
``on_ask`` callback, non-interactive runs leave it unset (``ask`` then becomes
a denial the model can react to). Tools declare everything the policy needs
through :class:`tynex.agent.types.AgentTool` attributes, so any tool — built-in,
MCP-bridged or SDK-defined — participates without special cases.
"""

from __future__ import annotations

import fnmatch
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

PermissionAction = Literal["allow", "ask", "deny"]
ApprovalOutcome = Literal["allow", "session", "deny"]

PERMISSION_MODES = ("plan", "ask", "accept-edits", "bypass")
DEFAULT_PERMISSION_MODE = "accept-edits"
# Tool kinds that only observe (never mutate) and may run without approval.
READ_KINDS = ("read", "web")
# Past-tense wording for rule-origin permission decisions ("deny" must not become "denyd").
RULE_REASONS = {"allow": "allowed", "deny": "denied", "ask": "approval required"}


def normalize_mode(mode: str) -> str:
    """Validate a permission mode name; raises :class:`ValueError` when unknown."""
    normalized = mode.strip().lower().replace("_", "-")
    if normalized not in PERMISSION_MODES:
        raise ValueError(f'Unknown permission mode "{mode}". Valid modes: {", ".join(PERMISSION_MODES)}.')
    return normalized


def mode_description(mode: str) -> str:
    return {
        "plan": "read-only; file changes and shell commands are denied",
        "ask": "reads run freely; edits, shell and MCP calls need approval",
        "accept-edits": "reads and workspace edits run freely; shell and MCP calls need approval",
        "bypass": "everything runs without approval",
    }.get(mode, "")


def _match(subject: str, pattern: str) -> bool:
    """fnmatch with a globstar-friendly fallback (``**/x`` also matches ``x``)."""
    if fnmatch.fnmatch(subject, pattern):
        return True
    if pattern.startswith("**/"):
        return fnmatch.fnmatch(subject, pattern[3:])
    return False


@dataclass
class PermissionRule:
    """One policy rule. Rules are evaluated in order; the last match wins."""

    tool: str = "*"
    match: str | None = None
    action: PermissionAction = "ask"
    source: str = ""

    def matches(self, tool_name: str, subjects: list[str]) -> bool:
        if not _match(tool_name, self.tool):
            return False
        if self.match is None:
            return True
        return any(_match(subject, self.match) for subject in subjects if subject)

    def describe(self) -> str:
        pattern = f" [dim]({self.match})[/]" if self.match else ""
        return f"{self.action}: {self.tool}{pattern}"


@dataclass
class PermissionRequest:
    """A single tool call that the policy wants the user to approve."""

    tool_name: str
    kind: str
    subject: str
    args: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        subject = " ".join(self.subject.split())
        if len(subject) > 160:
            subject = subject[:157] + "…"
        return f"{self.tool_name}: {subject}" if subject else self.tool_name


@dataclass
class PermissionDecision:
    action: PermissionAction
    reason: str
    origin: str  # "mode" | "rule" | "grant" | "approval" | "denied"
    request: PermissionRequest

    @property
    def allowed(self) -> bool:
        return self.action == "allow"


ApprovalFn = Callable[[PermissionRequest], "ApprovalOutcome | None"]


class PermissionGate:
    """Evaluates tool calls against a mode, rules and session grants."""

    def __init__(
        self,
        mode: str = DEFAULT_PERMISSION_MODE,
        rules: list[PermissionRule] | None = None,
        on_ask: ApprovalFn | None = None,
    ) -> None:
        self.mode = normalize_mode(mode)
        self.rules: list[PermissionRule] = list(rules or [])
        self.on_ask = on_ask
        self._grants: set[str] = set()
        self._lock = threading.Lock()

    # -- policy state -------------------------------------------------------
    def set_mode(self, mode: str) -> str:
        self.mode = normalize_mode(mode)
        return self.mode

    def grant_session(self, tool_name: str) -> None:
        with self._lock:
            self._grants.add(tool_name)

    def session_grants(self) -> list[str]:
        with self._lock:
            return sorted(self._grants)

    def clear_grants(self) -> None:
        with self._lock:
            self._grants.clear()

    # -- evaluation ---------------------------------------------------------
    def _mode_action(self, kind: str) -> PermissionAction:
        if self.mode == "bypass":
            return "allow"
        if self.mode == "plan":
            return "allow" if kind in READ_KINDS else "deny"
        if self.mode == "ask":
            return "allow" if kind in READ_KINDS else "ask"
        return "allow" if kind in READ_KINDS or kind == "edit" else "ask"  # accept-edits

    def check(self, tool: Any, args: dict[str, Any]) -> PermissionDecision:
        """Decide what to do with one tool call.

        ``tool`` is an :class:`~tynex.agent.types.AgentTool`; ``args`` are the
        raw model-provided arguments.
        """
        tool_name = getattr(tool, "name", "") or "?"
        kind = getattr(tool, "permission_kind", "tool")
        subject = ""
        subjects: list[str] = []
        try:
            subject = str(tool.permission_subject(args) or "")
        except Exception:
            subject = ""
        subjects = (
            [subject]
            if subject
            else [
                str(args[key])
                for key in ("path", "command", "query", "url", "glob", "pattern", "name")
                if isinstance(args.get(key), str) and args[key]
            ]
        )
        request = PermissionRequest(tool_name=tool_name, kind=kind, subject=subject, args=dict(args))

        action = self._mode_action(kind)
        reason = f"approval required ({kind} tools need approval in '{self.mode}' mode)"
        origin = "mode"
        for rule in self.rules:
            if rule.matches(tool_name, subjects):
                action = rule.action
                reason = f"{RULE_REASONS[rule.action]} by permission rule ({rule.tool})"
                origin = "rule"
        if self.mode == "plan" and kind not in READ_KINDS:
            # Plan mode is a hard floor: rules may tighten it, never widen it.
            return PermissionDecision("deny", "plan mode is read-only", "mode", request)
        if action == "allow":
            return PermissionDecision("allow", "allowed by policy", origin, request)
        if action == "deny":
            return PermissionDecision("deny", reason, origin, request)
        with self._lock:
            granted = tool_name in self._grants
        if granted:
            return PermissionDecision("allow", "allowed for this session", "grant", request)
        if self.on_ask is None:
            return PermissionDecision(
                "deny",
                f"{reason}; no interactive approval is available here — add an allow rule to "
                f"permissions.toml or rerun with --permission-mode bypass",
                "denied",
                request,
            )
        try:
            outcome = self.on_ask(request)
        except Exception as exc:  # an approval UI failure must never allow the call
            return PermissionDecision("deny", f"approval failed: {exc}", "denied", request)
        if outcome == "session":
            self.grant_session(tool_name)
            return PermissionDecision("allow", "allowed for this session", "approval", request)
        if outcome == "allow":
            return PermissionDecision("allow", "approved for this call", "approval", request)
        return PermissionDecision("deny", "denied by the user", "denied", request)
