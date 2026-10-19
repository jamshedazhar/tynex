"""Provider registry and streaming entry point."""

from collections.abc import Iterator
from typing import Any

from tynex.ai.models import Model
from tynex.ai.providers.anthropic import AnthropicProvider
from tynex.ai.providers.base import LlmContext, Provider, StreamOptions
from tynex.ai.providers.copilot import CopilotProvider
from tynex.ai.providers.faux import FauxProvider
from tynex.ai.providers.google import GoogleProvider
from tynex.ai.providers.openai import OpenAICompatProvider

_PROVIDERS: dict[str, Provider] = {
    "anthropic-messages": AnthropicProvider(),
    "openai-completions": OpenAICompatProvider(),
    "google-generative-ai": GoogleProvider(),
    "github-copilot": CopilotProvider(),
    "faux": FauxProvider(),
}


def get_provider(api: str) -> Provider:
    provider = _PROVIDERS.get(api)
    if provider is None:
        raise ValueError(f"Unknown provider API: {api!r}. Known: {', '.join(sorted(_PROVIDERS))}")
    return provider


def stream(model: Model, context: LlmContext, options: StreamOptions | None = None) -> Iterator[dict[str, Any]]:
    """Stream an assistant response as ``AssistantMessageEvent`` dicts."""
    options = options or StreamOptions()
    provider = get_provider(model.api)
    if not options.api_key:
        options.api_key = provider.resolve_api_key(model, options)
    if not options.base_url:
        options.base_url = model.base_url
    yield from provider.stream(model, context, options)
