"""Tests for the shared catalog: parsing, the on-disk cache and local catalogs.

Everything runs against a fixture payload and a temp config dir — no network
access needed.
"""

import json
import time
from typing import Any

import httpx2
import pytest

from tynex.ai import model_api
from tynex.ai import models as model_registry
from tynex.ai.model_api import CATALOG_URL, parse_catalog
from tynex.ai.models import Model, format_model_details, provider_spec
from tynex.coding import model_catalog
from tynex.coding.model_catalog import cached_models, ensure_catalog, ensure_models, resolve_model_retry, store_cached


def mock_client(handler) -> httpx2.Client:
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def model(model_id: str, provider: str = "anthropic", **kwargs: Any) -> Model:
    return Model(id=model_id, name=model_id, provider=provider, api="anthropic-messages", **kwargs)


CATALOG: dict[str, Any] = {
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "npm": "@ai-sdk/anthropic",
        "env": ["ANTHROPIC_API_KEY"],
        "doc": "https://docs.anthropic.com",
        "models": {
            "claude-opus": {
                "id": "claude-opus",
                "name": "Claude Opus",
                "limit": {"context": 1_000_000, "output": 128_000},
                "cost": {"input": 4, "output": 20, "cache_read": 0.2, "cache_write": 5},
                "modalities": {"input": ["text", "image"]},
                "reasoning": True,
                "reasoning_options": [{"type": "effort", "values": ["low", "high", "max"]}],
                "tool_call": True,
                "release_date": "2026-01-01",
            },
            "claude-retired": {
                "id": "claude-retired",
                "name": "Claude Retired",
                "limit": {"context": 200_000, "output": 8_192},
                "status": "deprecated",
                "tool_call": True,
            },
        },
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI",
        "npm": "@ai-sdk/openai",
        "env": ["OPENAI_API_KEY"],
        "models": {
            "gpt-new": {
                "id": "gpt-new",
                "name": "GPT New",
                "limit": {"context": 400_000, "output": 128_000},
                "cost": {"input": 2, "output": 8},
                "release_date": "2026-02-01",
                "tool_call": True,
            }
        },
    },
    "local-router": {
        "id": "local-router",
        "name": "Local Router",
        "npm": "@ai-sdk/openai-compatible",
        "api": "https://router.example.com/v1",
        "env": ["LOCAL_ROUTER_API_KEY"],
        "models": {
            "router-model": {
                "id": "router-model",
                "name": "Router Model",
                "limit": {"context": 131_072, "output": 16_384},
                "tool_call": True,
            }
        },
    },
    "bedrock": {"id": "bedrock", "name": "Bedrock", "npm": "@ai-sdk/amazon-bedrock", "models": {}},
    "no-endpoint": {"id": "no-endpoint", "name": "No Endpoint", "npm": "@ai-sdk/openai-compatible", "models": {}},
    "non-chat": {
        "id": "non-chat",
        "name": "Non Chat",
        "npm": "@ai-sdk/openai-compatible",
        "api": "https://nonchat.example.com/v1",
        "models": {
            "embed": {"id": "embed", "type": "embedding"},
            "no-tools": {"id": "no-tools", "tool_call": False},
            "chat": {"id": "chat", "tool_call": True, "limit": {"context": 1_000, "output": 100}},
        },
    },
}


# ---------------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------------


def test_parse_catalog_reads_provider_and_model_details():
    catalog = {item.spec.id: item for item in parse_catalog(CATALOG, environ={})}
    assert set(catalog) == {"anthropic", "openai", "local-router", "non-chat"}

    anthropic = catalog["anthropic"]
    assert anthropic.spec.name == "Anthropic"
    assert anthropic.spec.api == "anthropic-messages"
    assert anthropic.spec.base_url == "https://api.anthropic.com"
    assert anthropic.spec.api_key_env == ("ANTHROPIC_API_KEY",)

    opus = next(model for model in anthropic.models if model.id == "claude-opus")
    assert opus.context_window == 1_000_000
    assert opus.max_tokens == 128_000
    assert opus.cost.input == pytest.approx(4)
    assert opus.cost.output == pytest.approx(20)
    assert opus.cost.cache_read == pytest.approx(0.2)
    assert opus.cost.cache_write == pytest.approx(5)
    assert opus.input == ("text", "image")
    assert opus.reasoning is True
    assert opus.thinking_levels == ("low", "high", "max")
    assert opus.release_date == "2026-01-01"

    retired = next(model for model in anthropic.models if model.id == "claude-retired")
    assert retired.deprecated is True


def test_parse_catalog_skips_unsupported_providers_and_non_chat_models():
    catalog = {item.spec.id: item for item in parse_catalog(CATALOG, environ={})}
    assert "bedrock" not in catalog  # SDK-specific auth is not implemented
    assert "no-endpoint" not in catalog  # no base URL to talk to
    assert [model.id for model in catalog["non-chat"].models] == ["chat"]


def test_parse_catalog_expands_endpoint_placeholders():
    payload = {
        "gateway": {
            "id": "gateway",
            "name": "Gateway",
            "npm": "@ai-sdk/openai-compatible",
            "api": "https://${GATEWAY_HOST}/v1",
            "env": ["GATEWAY_HOST", "GATEWAY_API_KEY"],
            "models": {"m": {"id": "m", "tool_call": True}},
        }
    }
    assert parse_catalog(payload, environ={}) == []

    catalog = parse_catalog(payload, environ={"GATEWAY_HOST": "gw.example.com", "GATEWAY_API_KEY": "sk"})
    assert catalog[0].spec.base_url == "https://gw.example.com/v1"
    assert catalog[0].spec.api_key_env == ("GATEWAY_API_KEY",)  # the host is not a key


def test_parse_catalog_defaults_missing_limits():
    catalog = parse_catalog(
        {
            "mystery": {
                "id": "mystery",
                "npm": "@ai-sdk/openai-compatible",
                "api": "https://mystery.example.com/v1",
                "models": {"m": {"id": "m", "tool_call": True}},
            }
        },
        environ={},
    )
    assert catalog[0].models[0].context_window == model_registry.DEFAULT_CONTEXT_WINDOW
    assert catalog[0].models[0].max_tokens == model_registry.DEFAULT_MAX_TOKENS


# ---------------------------------------------------------------------------
# Catalog cache (models.json)
# ---------------------------------------------------------------------------


def age_cache(path, seconds: float = model_catalog.CACHE_TTL_SECONDS * 2) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    stale = time.time() - seconds
    data["catalog"]["fetchedAt"] = stale
    for entry in data["providers"].values():
        entry["fetchedAt"] = stale
    path.write_text(json.dumps(data), encoding="utf-8")


def test_store_and_load_cached_catalog(tynex_config):
    store_cached(
        "anthropic", [model("claude-x", context_window=123, max_tokens=45, cost=model_registry.Pricing(1, 2, 3, 4))]
    )
    loaded = cached_models("anthropic")
    assert loaded is not None
    models, fetched_at = loaded
    assert [m.id for m in models] == ["claude-x"]
    assert models[0].context_window == 123
    assert models[0].max_tokens == 45
    assert models[0].cost.input == pytest.approx(1)
    assert fetched_at == pytest.approx(time.time(), abs=5)

    data = json.loads((tynex_config / "models.json").read_text(encoding="utf-8"))
    assert data["version"] == model_catalog.CACHE_VERSION
    assert data["providers"]["anthropic"]["models"][0]["id"] == "claude-x"


def test_cached_models_tolerates_corrupt_cache(tynex_config):
    tynex_config.mkdir(exist_ok=True)
    (tynex_config / "models.json").write_text("{not json", encoding="utf-8")
    assert cached_models("anthropic") is None


def test_cached_models_ignores_catalogs_from_older_versions(tynex_config):
    tynex_config.mkdir(parents=True, exist_ok=True)
    (tynex_config / "models.json").write_text(
        json.dumps({"version": 1, "providers": {"anthropic": {"fetchedAt": time.time(), "models": [{"id": "old"}]}}}),
        encoding="utf-8",
    )
    assert cached_models("anthropic") is None


def test_ensure_catalog_fetches_registers_and_caches(tynex_config, monkeypatch):
    model_registry.reset_dynamic_models()
    monkeypatch.setattr(model_catalog, "fetch_catalog", lambda **kwargs: CATALOG)

    assert ensure_catalog({}) is None
    assert "local-router" in model_registry.providers()
    assert [m.id for m in model_registry.models_for_provider("anthropic")] == ["claude-opus", "claude-retired"]
    assert model_registry.models_for_provider("anthropic")[0].context_window == 1_000_000
    loaded = cached_models("anthropic")
    assert loaded is not None and [m.id for m in loaded[0]] == ["claude-opus", "claude-retired"]
    data = json.loads((tynex_config / "models.json").read_text(encoding="utf-8"))
    assert data["catalog"]["url"] == CATALOG_URL


def test_ensure_catalog_uses_a_fresh_cache_without_fetch(monkeypatch):
    monkeypatch.setattr(model_catalog, "fetch_catalog", lambda **kwargs: CATALOG)
    ensure_catalog({})
    model_catalog.reset_catalog_state()
    model_registry.reset_dynamic_models()
    monkeypatch.setattr(model_catalog, "fetch_catalog", lambda **kwargs: pytest.fail("must use the fresh cache"))
    assert ensure_catalog({}) is None
    assert model_registry.models_for_provider("openai")


def test_ensure_catalog_refreshes_a_stale_cache(tynex_config, monkeypatch):
    monkeypatch.setattr(model_catalog, "fetch_catalog", lambda **kwargs: CATALOG)
    ensure_catalog({})
    age_cache(tynex_config / "models.json")
    model_catalog.reset_catalog_state()
    model_registry.reset_dynamic_models()
    changed = {"openai": {**CATALOG["openai"], "models": {"gpt-newer": {"id": "gpt-newer", "tool_call": True}}}}
    monkeypatch.setattr(model_catalog, "fetch_catalog", lambda **kwargs: changed)

    assert ensure_catalog({}) is None
    assert [m.id for m in model_registry.models_for_provider("openai")] == ["gpt-newer"]


def test_ensure_catalog_falls_back_to_a_stale_cache_on_failure(tynex_config, monkeypatch):
    monkeypatch.setattr(model_catalog, "fetch_catalog", lambda **kwargs: CATALOG)
    ensure_catalog({})
    age_cache(tynex_config / "models.json")
    model_catalog.reset_catalog_state()
    model_registry.reset_dynamic_models()

    def failing_fetch(**kwargs):
        raise httpx2.ConnectError("no route to host")

    monkeypatch.setattr(model_catalog, "fetch_catalog", failing_fetch)
    note = ensure_catalog({})
    assert note is not None and "could not fetch the model catalog" in note and "cached catalog" in note
    assert [m.id for m in model_registry.models_for_provider("anthropic")] == ["claude-opus", "claude-retired"]


def test_ensure_catalog_can_skip_the_fetch(monkeypatch):
    monkeypatch.setattr(model_catalog, "fetch_catalog", lambda **kwargs: pytest.fail("fetch is disabled"))
    assert ensure_catalog({}, allow_fetch=False) == "no cached model catalog"


def test_ensure_models_lists_catalog_models_without_credentials(monkeypatch):
    monkeypatch.setattr(model_catalog, "fetch_catalog", lambda **kwargs: CATALOG)
    model_registry.reset_dynamic_models()
    models, note = ensure_models("openai", {})
    assert note is None
    assert [m.id for m in models] == ["gpt-new"]


def test_ensure_models_local_provider_needs_no_key(monkeypatch):
    model_registry.reset_dynamic_models()
    calls: list[str | None] = []
    monkeypatch.setattr(
        model_catalog, "fetch_models", lambda spec, api_key, **kw: calls.append(api_key) or [model("local", "ollama")]
    )
    models, note = ensure_models("ollama", {})
    assert note is None
    assert [m.id for m in models] == ["local"]
    assert calls == [None]
    loaded = cached_models("ollama")
    assert loaded is not None and [m.id for m in loaded[0]] == ["local"]


def test_ensure_models_skips_unknown_providers():
    model_registry.reset_dynamic_models()
    assert ensure_models("no-such-provider", {}) == ([], None)


def test_resolve_model_retry_loads_the_catalog(monkeypatch):
    monkeypatch.setattr(model_catalog, "fetch_catalog", lambda **kwargs: CATALOG)
    model_registry.reset_dynamic_models()
    resolved = resolve_model_retry("anthropic/claude-opus", {})
    assert resolved.id == "claude-opus"
    assert resolved.provider == "anthropic"


def test_resolve_model_retry_reports_an_unknown_model(monkeypatch):
    monkeypatch.setattr(model_catalog, "fetch_catalog", lambda **kwargs: CATALOG)
    model_registry.reset_dynamic_models()
    with pytest.raises(ValueError, match="No model matches"):
        resolve_model_retry("anthropic/does-not-exist", {})
    with pytest.raises(ValueError, match="No models known for 'github-copilot'"):
        resolve_model_retry("github-copilot/nope", {})


def test_models_are_ordered_by_release_date_and_deprecation(tynex_config, monkeypatch):
    monkeypatch.setattr(model_catalog, "fetch_catalog", lambda **kwargs: CATALOG)
    ensure_catalog({})
    ids = [m.id for m in model_registry.models_for_provider("anthropic")]
    assert ids == ["claude-opus", "claude-retired"]
    assert model_registry.default_model_for_provider("anthropic").id == "claude-opus"


def test_latest_aliases_are_listed_first_and_used_as_the_default():
    model_registry.reset_dynamic_models()
    model_registry.register_models(
        "google",
        [
            Model(id="gemini-2.5-flash", name="Gemini 2.5 Flash", provider="google", api="google-generative-ai"),
            Model(id="gemini-flash-latest", name="Gemini Flash Latest", provider="google", api="google-generative-ai"),
            Model(id="gemini-pro-latest", name="Gemini Pro Latest", provider="google", api="google-generative-ai"),
        ],
    )
    assert [m.id for m in model_registry.models_for_provider("google")] == [
        "gemini-flash-latest",
        "gemini-pro-latest",
        "gemini-2.5-flash",
    ]
    assert model_registry.default_model_for_provider("google").id == "gemini-flash-latest"


def test_format_model_details_shows_limits_and_pricing():
    details = format_model_details(
        Model(
            id="m",
            name="M",
            provider="p",
            api="openai-completions",
            context_window=200_000,
            max_tokens=64_000,
            cost=model_registry.Pricing(3, 15),
            input=("text", "image"),
            reasoning=True,
            thinking_levels=("low", "high"),
        )
    )
    assert "200.0k ctx" in details
    assert "64.0k out" in details
    assert "$3/$15 per M" in details
    assert "image" in details
    assert "reasoning (low, high)" in details


# ---------------------------------------------------------------------------
# Live OpenAI-compatible endpoints (custom providers)
# ---------------------------------------------------------------------------


def test_fetch_openai_compatible_models_for_custom_endpoint():
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"data": [{"id": "gpt-oss-120b"}]})

    models = model_api.fetch_openai_compatible_models(
        "my-llm", "https://llm.example.com/v1/", "sk-custom", client=mock_client(handler)
    )
    assert [m.id for m in models] == ["gpt-oss-120b"]
    assert models[0].provider == "my-llm"
    assert models[0].base_url == "https://llm.example.com/v1/"
    assert requests[0].url.path == "/v1/models"
    assert requests[0].headers["authorization"] == "Bearer sk-custom"


def test_ensure_models_uses_the_live_cache(monkeypatch):
    model_registry.reset_dynamic_models()
    store_cached("ollama", [model("cached-local", "ollama")])
    monkeypatch.setattr(model_catalog, "fetch_models", lambda *a, **kw: pytest.fail("must use the cache"))
    models, note = ensure_models("ollama", {})
    assert note is None
    assert [m.id for m in models] == ["cached-local"]


def test_local_catalog_falls_back_to_the_cache_on_failure(tynex_config, monkeypatch):
    store_cached("ollama", [model("old-local", "ollama")])
    path = tynex_config / "models.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["providers"]["ollama"]["fetchedAt"] = 0
    path.write_text(json.dumps(data), encoding="utf-8")
    model_registry.reset_dynamic_models()

    def failing_fetch(*args, **kwargs):
        raise httpx2.ConnectError("connection refused")

    monkeypatch.setattr(model_catalog, "fetch_models", failing_fetch)
    models, note = ensure_models("ollama", {})
    assert [m.id for m in models] == ["old-local"]
    assert note is not None and "could not fetch models" in note and "cached catalog" in note


def test_provider_spec_prefers_catalog_metadata():
    assert provider_spec("anthropic") is not None
    assert provider_spec("anthropic").api == "anthropic-messages"
    assert provider_spec("ollama").api_key_env == ()  # local endpoints need no key
