"""Tests for provider model catalogs: the models-API clients and the on-disk cache.

Everything runs against ``httpx2.MockTransport`` and a temp config dir — no
network access needed.
"""

import json
import time
from typing import Any

import httpx2
import pytest

from tynex.ai import model_api
from tynex.ai import models as model_registry
from tynex.ai.models import Model, provider_spec
from tynex.coding import model_catalog
from tynex.coding.model_catalog import cached_models, ensure_models, resolve_model_retry, store_cached
from tynex.coding.settings import save_auth


def mock_client(handler) -> httpx2.Client:
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def model(model_id: str, provider: str = "anthropic", **kwargs: Any) -> Model:
    return Model(id=model_id, name=model_id, provider=provider, api="anthropic-messages", **kwargs)


# ---------------------------------------------------------------------------
# Provider models-API parsing
# ---------------------------------------------------------------------------


def test_fetch_models_openai_style_catalog():
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "data": [
                    {"id": "gpt-4o", "created": 100},
                    {"id": "text-embedding-3-small", "created": 250},
                    {"id": "gpt-5", "name": "GPT-5", "created": 200},
                ]
            },
        )

    spec = provider_spec("openai")
    assert spec is not None
    models = model_api.fetch_models(spec, "sk-x", client=mock_client(handler))

    assert [m.id for m in models] == ["gpt-5", "gpt-4o"]  # newest first, embeddings filtered
    assert models[0].name == "GPT-5"
    assert models[0].context_window == spec.context_window  # defaults when the API reports none

    request = requests[0]
    assert request.url.path == "/v1/models"
    assert request.headers["authorization"] == "Bearer sk-x"


def test_fetch_models_openrouter_limits_and_pricing():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "data": [
                    {
                        "id": "vendor/model",
                        "name": "Vendor Model",
                        "context_length": 200_000,
                        "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                    }
                ]
            },
        )

    spec = provider_spec("openrouter")
    assert spec is not None
    models = model_api.fetch_models(spec, "sk-or", client=mock_client(handler))
    assert models[0].context_window == 200_000
    assert models[0].cost.input == pytest.approx(3.0)
    assert models[0].cost.output == pytest.approx(15.0)


def test_fetch_models_reads_deepseek_thinking_levels_and_limits():
    # api.deepseek.com/models reports the effort levels thinking accepts, plus
    # the real output limit and the accepted input modalities.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "deepseek-flash",
                        "object": "model",
                        "name": "DeepSeek-V4.1-Flash",
                        "context_window": 1_048_576,
                        "max_output_tokens": 393_216,
                        "input_modalities": ["text", "image"],
                        "effort": {"supported_levels": ["low", "high", "max"], "default_level": "high"},
                    }
                ],
            },
        )

    spec = provider_spec("deepseek")
    assert spec is not None
    models = model_api.fetch_models(spec, "sk-ds", client=mock_client(handler))
    assert len(models) == 1
    model = models[0]
    assert model.context_window == 1_048_576
    assert model.max_tokens == 393_216
    assert model.input == ("text", "image")
    assert model.thinking_levels == ("low", "high", "max")
    assert model.reasoning is True  # levels imply thinking support


def test_fetch_models_reads_openrouter_reasoning_parameter():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "data": [
                    {"id": "vendor/thinker", "supported_parameters": ["tools", "reasoning"]},
                    {"id": "vendor/plain", "supported_parameters": ["tools"]},
                ]
            },
        )

    spec = provider_spec("openrouter")
    assert spec is not None
    models = {m.id: m for m in model_api.fetch_models(spec, "sk-or", client=mock_client(handler))}
    assert models["vendor/thinker"].reasoning is True
    assert models["vendor/thinker"].thinking_levels == ()  # no levels reported
    assert models["vendor/plain"].reasoning is False


def test_fetch_models_anthropic_catalog():
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "data": [
                    {
                        "type": "model",
                        "id": "claude-old",
                        "display_name": "Claude Old",
                        "created_at": "2025-01-01T00:00:00Z",
                    },
                    {
                        "type": "model",
                        "id": "claude-new",
                        "display_name": "Claude New",
                        "created_at": "2025-09-01T00:00:00Z",
                    },
                ]
            },
        )

    spec = provider_spec("anthropic")
    assert spec is not None
    models = model_api.fetch_models(spec, "sk-ant", client=mock_client(handler))

    assert [m.id for m in models] == ["claude-new", "claude-old"]  # newest first
    assert models[0].name == "Claude New"
    assert requests[0].url.path == "/v1/models"
    assert requests[0].headers["x-api-key"] == "sk-ant"
    assert requests[0].headers["anthropic-version"] == model_api.ANTHROPIC_VERSION


def test_fetch_models_google_catalog():
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-2.5-pro",
                        "displayName": "Gemini 2.5 Pro",
                        "inputTokenLimit": 1_048_576,
                        "outputTokenLimit": 65_536,
                        "thinking": True,
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/text-embedding-004",
                        "displayName": "Embedding",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                ]
            },
        )

    spec = provider_spec("google")
    assert spec is not None
    models = model_api.fetch_models(spec, "gm-key", client=mock_client(handler))

    assert [m.id for m in models] == ["gemini-2.5-pro"]  # "models/" prefix stripped, embeddings dropped
    assert models[0].context_window == 1_048_576
    assert models[0].max_tokens == 65_536
    assert models[0].reasoning is True  # from the catalog's "thinking" flag
    assert requests[0].headers["x-goog-api-key"] == "gm-key"


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


# ---------------------------------------------------------------------------
# Catalog cache (models.json)
# ---------------------------------------------------------------------------


def test_store_and_load_cached_catalog(tynex_config):
    store_cached("anthropic", [model("claude-x", context_window=123)])
    loaded = cached_models("anthropic")
    assert loaded is not None
    models, fetched_at = loaded
    assert [m.id for m in models] == ["claude-x"]
    assert models[0].context_window == 123
    assert fetched_at == pytest.approx(time.time(), abs=5)

    data = json.loads((tynex_config / "models.json").read_text(encoding="utf-8"))
    assert data["version"] == model_catalog.CACHE_VERSION
    assert data["providers"]["anthropic"]["models"][0]["id"] == "claude-x"


def test_cached_models_tolerates_corrupt_cache(tynex_config):
    tynex_config.mkdir(exist_ok=True)
    (tynex_config / "models.json").write_text("{not json", encoding="utf-8")
    assert cached_models("anthropic") is None


def test_ensure_models_fetches_registers_and_caches(tynex_config, monkeypatch):
    model_registry.reset_dynamic_models()
    save_auth({"anthropic": {"type": "api_key", "key": "sk-a"}})
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        model_catalog,
        "fetch_models",
        lambda spec, api_key, **kw: calls.append((spec.id, api_key)) or [model("fresh-model")],
    )

    models, note = ensure_models("anthropic", {})
    assert note is None
    assert [m.id for m in models] == ["fresh-model"]
    assert calls == [("anthropic", "sk-a")]
    # Registered in-process and persisted for the next run.
    assert [m.id for m in model_registry.models_for_provider("anthropic")] == ["fresh-model"]
    loaded = cached_models("anthropic")
    assert loaded is not None and [m.id for m in loaded[0]] == ["fresh-model"]


def test_ensure_models_uses_fresh_cache_without_fetch(monkeypatch):
    model_registry.reset_dynamic_models()
    save_auth({"anthropic": {"type": "api_key", "key": "sk-a"}})
    store_cached("anthropic", [model("cached-model")])
    monkeypatch.setattr(model_catalog, "fetch_models", lambda *a, **kw: pytest.fail("must use the fresh cache"))

    models, note = ensure_models("anthropic", {})
    assert note is None
    assert [m.id for m in models] == ["cached-model"]


def test_ensure_models_refreshes_stale_cache(tynex_config, monkeypatch):
    model_registry.reset_dynamic_models()
    save_auth({"anthropic": {"type": "api_key", "key": "sk-a"}})
    # Stale cache: older than the TTL.
    (tynex_config / "models.json").write_text(
        json.dumps(
            {
                "version": model_catalog.CACHE_VERSION,
                "providers": {
                    "anthropic": {
                        "fetchedAt": time.time() - 2 * model_catalog.CACHE_TTL_SECONDS,
                        "models": [{"id": "old-model"}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_catalog, "fetch_models", lambda spec, api_key, **kw: [model("new-model")])

    models, note = ensure_models("anthropic", {})
    assert note is None
    assert [m.id for m in models] == ["new-model"]
    loaded = cached_models("anthropic")
    assert loaded is not None and [m.id for m in loaded[0]] == ["new-model"]
    assert loaded[1] > time.time() - 60  # the cache was refreshed


def test_ensure_models_falls_back_to_stale_cache_on_failure(tynex_config, monkeypatch):
    model_registry.reset_dynamic_models()
    save_auth({"anthropic": {"type": "api_key", "key": "sk-a"}})
    (tynex_config / "models.json").write_text(
        json.dumps(
            {
                "version": model_catalog.CACHE_VERSION,
                "providers": {"anthropic": {"fetchedAt": 0.0, "models": [{"id": "old-model"}]}},
            }
        ),
        encoding="utf-8",
    )

    def failing_fetch(*args, **kwargs):
        raise httpx2.ConnectError("no route to host")

    monkeypatch.setattr(model_catalog, "fetch_models", failing_fetch)
    models, note = ensure_models("anthropic", {})
    assert [m.id for m in models] == ["old-model"]
    assert note is not None and "could not fetch models" in note and "cached catalog" in note


def test_ensure_models_without_credentials_reports_and_uses_cache(tynex_config):
    model_registry.reset_dynamic_models()
    models, note = ensure_models("openai", {})
    assert models == []
    assert note is not None and "no credentials" in note

    store_cached("openai", [model("gpt-x", provider="openai")])
    models, note = ensure_models("openai", {})
    assert [m.id for m in models] == ["gpt-x"]  # usable offline, but told about the missing key
    assert note is not None and "no credentials" in note


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


def test_ensure_models_skips_copilot_and_unknown_providers():
    model_registry.reset_dynamic_models()
    assert ensure_models("github-copilot", {}) == ([], None)
    assert ensure_models("no-such-provider", {}) == ([], None)


# ---------------------------------------------------------------------------
# resolve_model with catalog fetch fallback
# ---------------------------------------------------------------------------


def test_resolve_model_retry_fetches_missing_catalog(monkeypatch):
    model_registry.reset_dynamic_models()
    save_auth({"anthropic": {"type": "api_key", "key": "sk-a"}})
    monkeypatch.setattr(model_catalog, "fetch_models", lambda spec, api_key, **kw: [model("brand-new")])

    resolved = resolve_model_retry("anthropic/brand-new", {})
    assert resolved.id == "brand-new"

    # A bare pattern tries every provider that has credentials.
    model_registry.reset_dynamic_models()
    resolved = resolve_model_retry("brand-new", {})
    assert resolved.provider == "anthropic"


def test_resolve_model_retry_reports_missing_catalog_without_credentials():
    model_registry.reset_dynamic_models()
    with pytest.raises(ValueError, match="No models known for 'anthropic'"):
        resolve_model_retry("anthropic/whatever", {})


def test_resolve_model_requires_credentials_before_matching():
    model_registry.reset_dynamic_models()
    with pytest.raises(ValueError):
        resolve_model_retry("openai/gpt-5", {})


def test_cached_models_ignores_catalogs_from_older_versions(tynex_config):
    tynex_config.mkdir(parents=True, exist_ok=True)
    (tynex_config / "models.json").write_text(
        json.dumps({"version": 1, "providers": {"anthropic": {"fetchedAt": time.time(), "models": [{"id": "old"}]}}}),
        encoding="utf-8",
    )
    assert cached_models("anthropic") is None


def test_cached_models_round_trips_reasoning(tynex_config):
    store_cached(
        "google",
        [
            Model(
                id="gemini-3.8-flash",
                name="Gemini 3.8 Flash",
                provider="google",
                api="google-generative-ai",
                reasoning=True,
            )
        ],
    )
    loaded = cached_models("google")
    assert loaded is not None
    assert loaded[0][0].reasoning is True


def test_latest_aliases_are_listed_first_and_used_as_the_default():
    # Gemini still lists retired ids first; the rolling alias is the safe default.
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


def test_ensure_models_orders_an_old_cached_catalog(tynex_config):
    # The on-disk cache may predate the ordering rule: read it back newest-first anyway.
    model_registry.reset_dynamic_models()
    store_cached(
        "google",
        [
            Model(id="gemini-2.5-flash", name="Gemini 2.5 Flash", provider="google", api="google-generative-ai"),
            Model(id="gemini-flash-latest", name="Gemini Flash Latest", provider="google", api="google-generative-ai"),
        ],
    )

    models, note = ensure_models("google", {})
    assert [m.id for m in models] == ["gemini-flash-latest", "gemini-2.5-flash"]
    assert note == "no credentials for google — run `tynex login google`"
    assert model_registry.default_model_for_provider("google").id == "gemini-flash-latest"
