"""Budgets: optional hard caps on total tokens or cost for one run."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class BudgetLimits:
    """Optional caps for one run. ``None`` means unlimited."""

    max_tokens: int | None = None  # input + output + cache-read tokens
    max_cost_usd: float | None = None

    @property
    def empty(self) -> bool:
        return self.max_tokens is None and self.max_cost_usd is None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.max_tokens is not None:
            out["maxTokens"] = self.max_tokens
        if self.max_cost_usd is not None:
            out["maxCostUsd"] = self.max_cost_usd
        return out

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> BudgetLimits | None:
        """Read the serialized form; unknown/legacy fields are ignored."""
        if not isinstance(data, dict):
            return None
        kwargs: dict[str, Any] = {}
        for key, field_name, cast in (("maxTokens", "max_tokens", int), ("maxCostUsd", "max_cost_usd", float)):
            value = data.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                kwargs[field_name] = cast(value)
            except (TypeError, ValueError):
                continue
        limits = BudgetLimits(**kwargs)
        return None if limits.empty else limits


def parse_budget(spec: str) -> BudgetLimits:
    """Parse ``--budget``: ``cost=2.50``, ``tokens=250000``, or both comma-separated.

    Raises ``ValueError`` with a user-facing message for anything else.
    """
    limits = BudgetLimits()
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        key, separator, raw_value = item.partition("=")
        if not separator:
            raise ValueError(f'expected cost=AMOUNT or tokens=COUNT, got "{item}"')
        key = key.strip().lower()
        value_text = raw_value.strip().lstrip("$")
        if not value_text:
            raise ValueError(f'missing value for "{key}"')
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(f'invalid {key} value "{raw_value.strip()}"') from exc
        if key == "cost":
            if value <= 0:
                raise ValueError("cost budget must be greater than zero")
            limits.max_cost_usd = value
        elif key == "tokens":
            if value <= 0 or not value.is_integer():
                raise ValueError("tokens budget must be a positive whole number")
            limits.max_tokens = int(value)
        else:
            raise ValueError(f'unknown budget field "{key}" (use cost or tokens)')
    if limits.empty:
        raise ValueError("empty budget — use cost=AMOUNT or tokens=COUNT")
    return limits


@dataclass
class BudgetState:
    """Mutable counters for one session."""

    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    started: float | None = None

    def reset(self) -> None:
        self.steps = 0
        self.tool_calls = 0
        self.tokens = 0
        self.cost_usd = 0.0
        self.started = None

    def add_usage(self, usage: Any) -> None:
        total = int(getattr(usage, "total_tokens", 0) or 0)
        if not total:
            total = int(getattr(usage, "input", 0) or 0) + int(getattr(usage, "output", 0) or 0)
        self.tokens += total + int(getattr(usage, "cache_read", 0) or 0) + int(getattr(usage, "cache_write", 0) or 0)
        self.cost_usd += float(getattr(usage, "cost", 0.0) or 0.0)

    def exceeded(self, limits: BudgetLimits | None) -> str | None:
        """First exceeded limit as a human-readable reason, or ``None``."""
        if limits is None:
            return None
        if limits.max_tokens is not None and self.tokens >= limits.max_tokens:
            return f"token limit reached ({self.tokens:,}/{limits.max_tokens:,} tokens)"
        if limits.max_cost_usd is not None and self.cost_usd >= limits.max_cost_usd:
            return f"cost limit reached (${self.cost_usd:.4f}/${limits.max_cost_usd:.4f})"
        return None

    def begin(self) -> None:
        if self.started is None:
            self.started = time.monotonic()

    def snapshot(self, limits: BudgetLimits | None = None) -> dict[str, Any]:
        elapsed = (time.monotonic() - self.started) if self.started is not None else 0.0
        return {
            "steps": self.steps,
            "toolCalls": self.tool_calls,
            "tokens": self.tokens,
            "cost": round(self.cost_usd, 6),
            "seconds": round(elapsed, 1),
            "limits": limits.to_dict() if limits else {},
        }
