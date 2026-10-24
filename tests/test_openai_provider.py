"""Request shape for OpenAI-compatible providers: thinking becomes ``reasoning_effort``."""

from typing import Any

from tynex.ai.models import Model
from tynex.ai.providers.base import LlmContext, StreamOptions, clamp_thinking
from tynex.ai.providers.openai import OpenAICompatProvider


def model(model_id: str = "deepseek-flash", provider: str = "deepseek", **kwargs: Any) -> Model:
    return Model(id=model_id, name=model_id, provider=provider, api="openai-completions", **kwargs)


def body(target: Model, thinking: str) -> dict[str, Any]:
    return OpenAICompatProvider()._chat_body(target, LlmContext(), StreamOptions(api_key="k", thinking=thinking))


def test_clamp_thinking_picks_the_nearest_level_higher_on_a_tie():
    assert clamp_thinking("high", ["low", "high"]) == "high"
    assert clamp_thinking("medium", ["low", "high"]) == "high"  # tie goes up
    assert clamp_thinking("minimal", ["low", "high"]) == "low"
    assert clamp_thinking("max", ["low", "high"]) == "high"
    assert clamp_thinking("high", []) == "off"
    assert clamp_thinking("bogus", ["high"]) == "high"
    assert clamp_thinking("high", ["none"]) == "off"  # unknown levels are ignored


def test_a_catalog_that_lists_levels_gets_the_level_it_asked_for():
    deepseek = model(reasoning=True, thinking_levels=("low", "high", "max"))
    assert body(deepseek, "low")["reasoning_effort"] == "low"
    assert body(deepseek, "high")["reasoning_effort"] == "high"
    assert body(deepseek, "max")["reasoning_effort"] == "max"
    # The model reasons by default, so "off" must be said out loud.
    assert body(deepseek, "off")["reasoning_effort"] == "none"


def test_levels_the_catalog_does_not_list_are_clamped():
    deepseek = model(reasoning=True, thinking_levels=("low", "high", "max"))
    assert body(deepseek, "medium")["reasoning_effort"] == "high"
    assert body(deepseek, "minimal")["reasoning_effort"] == "low"
    assert body(deepseek, "xhigh")["reasoning_effort"] == "max"


def test_a_model_without_thinking_support_sends_no_effort():
    plain = model("deepseek-chat")
    assert "reasoning_effort" not in body(plain, "high")
    assert "reasoning_effort" not in body(plain, "off")


def test_openai_keeps_its_own_effort_mapping():
    gpt = model("gpt-5", provider="openai", reasoning=True)
    assert body(gpt, "minimal")["reasoning_effort"] == "low"
    assert body(gpt, "xhigh")["reasoning_effort"] == "high"
    assert "reasoning_effort" not in body(gpt, "off")


def test_other_openai_compatible_providers_stay_quiet_without_levels():
    # xAI/Groq reason by default; tynex has no level to send them.
    grok = model("grok-5", provider="xai", reasoning=True)
    assert "reasoning_effort" not in body(grok, "high")
