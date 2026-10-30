import copy
import json
import time
from typing import Any

import pytest

from tynex import http
from tynex.ai.model_api import CATALOG_URL
from tynex.ai.models import (
    model_from_spec,
    provider_spec,
    register_custom_providers,
    register_models,
    reset_custom_providers,
    reset_dynamic_models,
    reset_provider_specs,
    resolve_model,
)
from tynex.ai.providers import faux
from tynex.coding import model_catalog
from tynex.coding import settings as paths
from tynex.coding.model_catalog import CACHE_VERSION

# Provider credentials must never leak from the developer's shell into tests
# (they would change model auto-selection and env-var precedence).
PROVIDER_KEY_ENVS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "OPENROUTER_API_KEY",
    "GITHUB_COPILOT_TOKEN",
    "OLLAMA_API_KEY",
    "LMSTUDIO_API_KEY",
    "FAUX_API_KEY",
]

# Production tynex ships no model names: every provider's catalog is fetched from
# its models API (tynex.coding.model_catalog). Tests register a fixed catalog so
# they stay offline and deterministic; the shapes match what the API parsers
# (tynex.ai.model_api) produce. See also TEST_MODE_CACHE below.
TEST_MODELS: dict[str, list[dict[str, Any]]] = {
    "anthropic": [
        {
            "id": "claude-sonnet-4-5",
            "name": "Claude Sonnet 4.5",
            "contextWindow": 200_000,
            "maxTokens": 64_000,
            "reasoning": True,
            "input": ["text", "image"],
        },
        {
            "id": "claude-haiku-4-5",
            "name": "Claude Haiku 4.5",
            "contextWindow": 200_000,
            "maxTokens": 32_000,
            "input": ["text", "image"],
        },
    ],
    "openai": [
        {
            "id": "gpt-5",
            "name": "GPT-5",
            "contextWindow": 400_000,
            "maxTokens": 128_000,
            "reasoning": True,
            "input": ["text", "image"],
        }
    ],
    "google": [
        {
            "id": "gemini-2.5-pro",
            "name": "Gemini 2.5 Pro",
            "contextWindow": 1_048_576,
            "maxTokens": 65_536,
            "reasoning": True,
            "input": ["text", "image"],
        }
    ],
    "groq": [
        {
            "id": "llama-3.3-70b-versatile",
            "name": "Llama 3.3 70B Versatile",
            "contextWindow": 131_072,
            "maxTokens": 32_768,
        }
    ],
    "ollama": [{"id": "qwen3-coder", "name": "Qwen3 Coder", "contextWindow": 262_144, "maxTokens": 32_768}],
}


def register_test_models() -> None:
    """Register :data:`TEST_MODELS` through the real model pipeline."""
    for provider, entries in TEST_MODELS.items():
        spec = provider_spec(provider)
        assert spec is not None
        register_models(
            provider,
            [
                model_from_spec(
                    provider,
                    entry,
                    api=spec.api,
                    base_url=spec.base_url,
                    context_window=spec.context_window,
                    max_tokens=spec.max_tokens,
                )
                for entry in entries
            ],
        )


def models_cache_payload() -> dict[str, Any]:
    """Content for ``TYNEX_CONFIG_DIR/models.json``; subprocess tests write it so
    the shared catalog counts as freshly cached (no network)."""
    now = time.time()
    return {
        "version": CACHE_VERSION,
        "catalog": {"url": CATALOG_URL, "fetchedAt": now},
        "providers": {provider: {"fetchedAt": now, "models": entries} for provider, entries in TEST_MODELS.items()},
    }


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point all config at a temp dir so tests never touch the user's ~/.tynex."""
    config = tmp_path / "tynex-config"
    monkeypatch.setenv("TYNEX_CONFIG_DIR", str(config))
    monkeypatch.delenv("TYNEX_FAUX_SCRIPT", raising=False)
    for name in PROVIDER_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    yield config


@pytest.fixture(autouse=True)
def offline_mcp_defaults(monkeypatch):
    """Keep shipped network MCP servers from connecting during tests.

    ``load_settings`` sees a copy of the shipped defaults with every built-in
    disabled; catalog/status tests still inspect the real shipped data.
    """
    shipped = paths.default_settings()

    def offline_defaults():
        data = copy.deepcopy(shipped)
        servers = data.get("mcpServers")
        if isinstance(servers, dict):
            for entry in servers.values():
                if isinstance(entry, dict):
                    entry["enabled"] = False
        return data

    monkeypatch.setattr(paths, "default_settings", offline_defaults)


@pytest.fixture
def tynex_config(isolated_config):
    """Readable alias for the isolated config directory."""
    return isolated_config


@pytest.fixture
def builtin_root(monkeypatch, tmp_path):
    """Relocate the built-in resources dir (``tynex/agent``, derived from ``paths.__file__``) under ``tmp_path``."""
    monkeypatch.setattr(paths, "__file__", str(tmp_path / "tynex" / "coding" / "paths.py"))
    return tmp_path / "tynex" / "agent"


@pytest.fixture(autouse=True)
def faux_provider_registered():
    register_custom_providers(
        {
            "faux": {
                "api": "faux",
                "models": [{"id": "test", "name": "Faux Test Model", "contextWindow": 200_000, "maxTokens": 8_192}],
            }
        }
    )
    register_test_models()
    faux.reset_script(None)
    yield
    faux.reset_script(None)


@pytest.fixture(autouse=True)
def isolated_model_registry():
    """Provider registrations are global; reset them after each test."""
    yield
    reset_custom_providers()
    reset_dynamic_models()
    reset_provider_specs()
    model_catalog.reset_catalog_state()


@pytest.fixture(autouse=True)
def offline_model_catalog(monkeypatch):
    """Never contact the catalog service or a local endpoint from the default suite."""
    monkeypatch.setattr(model_catalog, "fetch_catalog", lambda **kwargs: {})
    monkeypatch.setattr(model_catalog, "fetch_models", lambda *args, **kwargs: [])
    model_catalog.reset_catalog_state()
    yield
    model_catalog.reset_catalog_state()


@pytest.fixture(autouse=True)
def isolated_http_client():
    """A client installed by one test (mock transport) must never leak into the next."""
    yield
    http.set_client(None)


@pytest.fixture
def faux_model():
    return resolve_model("faux/test")


def faux_scenario(events: list[dict]) -> dict:
    return {"events": events}


def text_scenario(text: str, reason: str = "stop") -> dict:
    return faux_scenario([{"type": "text", "text": text}, {"type": "stop", "reason": reason}])


def json_lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]
