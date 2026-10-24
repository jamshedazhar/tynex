"""Budgets: opt-in token/cost caps, parsing and enforcement in the agent loop (offline)."""

from dataclasses import replace

import pytest

from tests.conftest import faux_scenario, text_scenario
from tynex.agent.budget import BudgetLimits, BudgetState, parse_budget
from tynex.ai.models import Pricing
from tynex.ai.providers import faux
from tynex.ai.types import Usage
from tynex.coding.agent_session import AgentSession


def tool_call(name="bash", arguments=None, call_id="call_1"):
    return {"type": "toolcall", "id": call_id, "name": name, "arguments": arguments or {"command": "echo x"}}


# ---------------------------------------------------------------------------
# Limits and counters
# ---------------------------------------------------------------------------


def test_budget_limits_round_trip():
    limits = BudgetLimits.from_dict({"maxTokens": "1000", "maxCostUsd": 2.5})
    assert limits == BudgetLimits(max_tokens=1000, max_cost_usd=2.5)
    assert not limits.empty
    assert limits.to_dict() == {"maxTokens": 1000, "maxCostUsd": 2.5}
    assert BudgetLimits.from_dict(None) is None
    assert BudgetLimits.from_dict({}) is None
    assert BudgetLimits.from_dict({"maxSteps": 5, "maxToolCalls": 3}) is None  # legacy fields are ignored
    assert BudgetLimits.from_dict({"maxTokens": "nope"}) is None
    assert BudgetLimits().empty


def test_parse_budget_spec():
    assert parse_budget("cost=2.5") == BudgetLimits(max_cost_usd=2.5)
    assert parse_budget("cost=$2.50") == BudgetLimits(max_cost_usd=2.5)
    assert parse_budget("tokens=250000") == BudgetLimits(max_tokens=250_000)
    assert parse_budget("COST=1, tokens=1000") == BudgetLimits(max_cost_usd=1.0, max_tokens=1000)


@pytest.mark.parametrize(
    "spec", ["", "   ", "cost", "cost=", "cost=abc", "cost=0", "cost=-1", "tokens=1.5", "tokens=0", "steps=5"]
)
def test_parse_budget_rejects_invalid_specs(spec):
    with pytest.raises(ValueError):
        parse_budget(spec)


def test_state_counts_usage_and_exceeded_reasons():
    state = BudgetState()
    assert state.exceeded(BudgetLimits()) is None
    assert state.exceeded(None) is None

    usage = Usage(input=100, output=50, cache_read=25, cache_write=5, cost=0.25)
    usage.total_tokens = 150
    state.add_usage(usage)
    assert state.tokens == 180  # input + output + cache read + cache write
    assert state.cost_usd == 0.25

    assert state.exceeded(BudgetLimits(max_tokens=181)) is None
    assert "token limit" in state.exceeded(BudgetLimits(max_tokens=180))
    assert "cost limit" in state.exceeded(BudgetLimits(max_cost_usd=0.2))
    assert state.exceeded(BudgetLimits(max_cost_usd=0.25)) is not None  # >=

    state.reset()
    assert (state.steps, state.tool_calls, state.tokens, state.cost_usd, state.started) == (0, 0, 0, 0.0, None)
    assert state.snapshot(BudgetLimits(max_tokens=1000))["limits"] == {"maxTokens": 1000}


def test_steps_and_tool_calls_are_not_budget_limits():
    state = BudgetState(steps=50, tool_calls=50, tokens=10, cost_usd=0.01)
    assert state.exceeded(BudgetLimits(max_tokens=100, max_cost_usd=1.0)) is None


# ---------------------------------------------------------------------------
# Enforcement through AgentSession
# ---------------------------------------------------------------------------


def make_session(tmp_path, faux_model, limits, tool_names=("read",), **kwargs):
    return AgentSession(
        cwd=str(tmp_path), model=faux_model, tool_names=list(tool_names), no_session=True, budget=limits, **kwargs
    )


def test_token_limit_stops_before_the_next_model_call(tmp_path, faux_model):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    tool_round = faux_scenario(
        [
            {"type": "usage", "usage": {"input": 10, "output": 5}},
            tool_call("read", {"path": "a.txt"}),
            {"type": "stop", "reason": "toolUse"},
        ]
    )
    faux.reset_script([tool_round, tool_round, text_scenario("never reached")])
    session = make_session(tmp_path, faux_model, BudgetLimits(max_tokens=10), tool_names=["read"])
    events: list[dict] = []
    session.prompt("go", emit=events.append)
    limits = [event for event in events if event["type"] == "limit_reached"]
    assert limits and "token limit" in limits[0]["reason"]
    assert session.budget.tokens == 15
    assert events[-1]["type"] == "agent_end"


def test_cost_limit_uses_model_pricing(tmp_path, faux_model):
    pricey = replace(
        faux_model,
        cost=Pricing(input=1_000_000.0, output=1_000_000.0),  # $1 per token
    )
    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "text", "text": "costly"},
                    {"type": "usage", "usage": {"input": 10, "output": 10}},
                    {"type": "stop", "reason": "stop"},
                ]
            ),
            text_scenario("never reached"),
        ]
    )
    session = make_session(tmp_path, pricey, BudgetLimits(max_cost_usd=1.0))
    events: list[dict] = []
    session.prompt("go", emit=events.append)
    assert session.budget.cost_usd >= 20.0
    # The budget only blocks the next model call, so a follow-up prompt stops.
    events.clear()
    session.prompt("again", emit=events.append)
    limits = [event for event in events if event["type"] == "limit_reached"]
    assert limits and "cost limit" in limits[0]["reason"]


def test_no_limits_means_unlimited(tmp_path, faux_model):
    faux.reset_script([text_scenario("plain")])
    session = make_session(tmp_path, faux_model, None)
    events: list[dict] = []
    session.prompt("go", emit=events.append)
    assert not [event for event in events if event["type"] == "limit_reached"]
    assert session.budget.steps == 1
    assert session.stats()["budget"]["steps"] == 1


def test_settings_budget_is_ignored(tmp_path, faux_model):
    from tynex.coding.settings import project_dir

    project_dir(tmp_path).mkdir(parents=True)
    (project_dir(tmp_path) / "settings.json").write_text('{"budget": {"maxTokens": 1}}', encoding="utf-8")
    faux.reset_script([text_scenario("one"), text_scenario("two")])
    session = AgentSession(cwd=str(tmp_path), model=faux_model, tool_names=["read"], no_session=True)
    assert session.budget_limits is None
    events: list[dict] = []
    session.prompt("go", emit=events.append)
    events.clear()
    session.prompt("more", emit=events.append)
    assert not [event for event in events if event["type"] == "limit_reached"]


def test_budget_command_renders(tmp_path, faux_model):
    from io import StringIO

    from rich.console import Console

    from tynex.coding import session_commands

    session = make_session(tmp_path, faux_model, BudgetLimits(max_tokens=1000, max_cost_usd=1.5))
    session.budget.tokens = 250
    session.budget.cost_usd = 0.1234
    buffer = StringIO()
    session_commands.budget(session, Console(file=buffer, width=100))
    output = buffer.getvalue()
    assert "tokens" in output and "250 / 1,000" in output
    assert "cost" in output and "$0.1234 / $1.5000" in output

    token_only = make_session(tmp_path, faux_model, BudgetLimits(max_tokens=5))
    buffer = StringIO()
    session_commands.budget(token_only, Console(file=buffer, width=100))
    output = buffer.getvalue()
    assert "tokens" in output and "0 / 5" in output
    assert "cost" not in output

    plain = make_session(tmp_path, faux_model, None)
    buffer = StringIO()
    session_commands.budget(plain, Console(file=buffer, width=100))
    assert "No budget set" in buffer.getvalue()


def test_usage_totals_and_stats_include_cache(tmp_path, faux_model):
    faux.reset_script(
        [
            faux_scenario(
                [
                    {"type": "text", "text": "hi"},
                    {"type": "usage", "usage": {"input": 10, "output": 5, "cacheRead": 3, "cacheWrite": 2}},
                    {"type": "stop", "reason": "stop"},
                ]
            )
        ]
    )
    session = make_session(tmp_path, faux_model, None)
    session.prompt("go", emit=lambda event: None)
    totals = session.usage_totals()
    assert totals.input == 10 and totals.output == 5
    assert session.budget.tokens == 20  # input + output + cache read + cache write


def test_usage_without_a_total_still_counts_against_the_budget():
    # Anthropic reports input/output/cache tokens and never a total; a token cap
    # that read only ``total_tokens`` would never fire for Claude.
    state = BudgetState()
    state.add_usage(Usage(input=10, output=5, cache_read=3, cache_write=30_000))
    assert state.tokens == 30_018
    assert state.exceeded(BudgetLimits(max_tokens=30_018)) is not None
