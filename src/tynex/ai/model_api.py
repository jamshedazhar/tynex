"""Model catalogs: the shared provider catalog and live OpenAI-compatible endpoints."""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from tynex.ai.models import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_TOKENS,
    Model,
    ProviderSpec,
    model_from_spec,
    provider_base_url,
)
from tynex.http import Client, get_json

CATALOG_URL = "https://models.dev/api.json"
CATALOG_TIMEOUT = 30.0
DEFAULT_TIMEOUT = 15.0

_NPM_APIS = {
    "@ai-sdk/anthropic": "anthropic-messages",
    "@ai-sdk/google": "google-generative-ai",
    "@ai-sdk/openai": "openai-completions",
    "@ai-sdk/openai-compatible": "openai-completions",
    "@openrouter/ai-sdk-provider": "openai-completions",
    "@ai-sdk/xai": "openai-completions",
    "@ai-sdk/groq": "openai-completions",
    "@ai-sdk/mistral": "openai-completions",
}

_DEFAULT_BASE_URLS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com",
    "xai": "https://api.x.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
}

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
_THINKING_LEVELS = {"minimal", "low", "medium", "high", "xhigh", "max"}
_NON_CHAT_TYPES = {"embedding", "image", "video", "speech", "transcription", "reranking", "realtime"}
_NON_KEY_ENVS = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AZURE_RESOURCE_NAME",
    "CLOUDFLARE_ACCOUNT_ID",
    "DATABRICKS_HOST",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_VERTEX_LOCATION",
    "GOOGLE_VERTEX_PROJECT",
    "INFOMANIAK_PRODUCT_ID",
    "NEON_AI_GATEWAY_BASE_URL",
    "SNOWFLAKE_ACCOUNT",
}
_KEY_SUFFIXES = ("_API_KEY", "_ACCESS_KEY", "_TOKEN", "_KEY", "_PAT", "_SECRET")

# Endpoint catalogs mix chat models with embeddings/audio/media endpoints.
NON_CHAT_ID_PATTERNS = (
    "embed",
    "whisper",
    "tts",
    "dall-e",
    "sora",
    "transcribe",
    "audio",
    "realtime",
    "moderation",
    "image",
    "computer-use",
)


@dataclass(frozen=True)
class CatalogProvider:
    spec: ProviderSpec
    models: tuple[Model, ...]


def fetch_catalog(*, client: Any = None, timeout: float = CATALOG_TIMEOUT) -> dict[str, Any]:
    payload = get_json(CATALOG_URL, headers={"Accept": "application/json"}, client=client, timeout=timeout)
    if not isinstance(payload, dict):
        raise ValueError("unexpected catalog shape")
    return payload


def parse_catalog(payload: Mapping[str, Any], *, environ: Mapping[str, str] | None = None) -> list[CatalogProvider]:
    """Normalize a catalog payload into providers tynex can talk to."""
    environ = os.environ if environ is None else environ
    providers: list[CatalogProvider] = []
    for key, raw in payload.items():
        if not isinstance(raw, dict):
            continue
        provider_id = str(raw.get("id") or key or "").strip()
        if not provider_id:
            continue
        api = _NPM_APIS.get(str(raw.get("npm") or ""))
        if api is None:
            continue
        base_url = _base_url(raw, provider_id, environ)
        if base_url is None:
            continue
        models_section = raw.get("models")
        if not isinstance(models_section, dict):
            continue
        models = [
            model
            for model_id, entry in models_section.items()
            if isinstance(entry, dict)
            and (model := _catalog_model(provider_id, model_id, entry, api=api, base_url=base_url)) is not None
        ]
        if not models:
            continue
        spec = ProviderSpec(
            id=provider_id,
            name=str(raw.get("name") or provider_id),
            api=api,
            base_url=base_url,
            api_key_env=_api_key_envs(raw.get("env"), raw.get("api")),
            local=_is_local(base_url),
        )
        providers.append(CatalogProvider(spec=spec, models=tuple(models)))
    return providers


def _base_url(raw: Mapping[str, Any], provider_id: str, environ: Mapping[str, str]) -> str | None:
    api = raw.get("api")
    if isinstance(api, str) and api.strip():
        expanded = _expand(api.strip(), environ)
        return expanded.rstrip("/") if expanded else None
    fallback = _DEFAULT_BASE_URLS.get(provider_id)
    return fallback.rstrip("/") if fallback else None


def _expand(value: str, environ: Mapping[str, str]) -> str | None:
    missing = False

    def replace(match: re.Match[str]) -> str:
        nonlocal missing
        replacement = environ.get(match.group(1), "")
        if not replacement:
            missing = True
        return replacement

    expanded = _PLACEHOLDER.sub(replace, value)
    return None if missing else expanded


def _api_key_envs(raw: Any, raw_api: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        return ()
    api = raw_api if isinstance(raw_api, str) else ""
    placeholders = set(_PLACEHOLDER.findall(api))
    candidates = [
        str(name).strip()
        for name in raw
        if isinstance(name, str)
        and name.strip()
        and name.strip() not in placeholders
        and name.strip() not in _NON_KEY_ENVS
    ]
    keyish = [name for name in candidates if name.endswith(_KEY_SUFFIXES)]
    return tuple(keyish or candidates)


def _is_local(base_url: str) -> bool:
    return (urlparse(base_url).hostname or "") in _LOCAL_HOSTS


def _catalog_model(provider_id: str, model_id: Any, raw: Mapping[str, Any], *, api: str, base_url: str) -> Model | None:
    identifier = str(raw.get("id") or model_id or "").strip()
    if not identifier:
        return None
    kind = raw.get("type")
    if isinstance(kind, str) and kind and kind != "chat" and kind in _NON_CHAT_TYPES:
        return None
    if raw.get("tool_call") is False:
        return None
    limits = raw.get("limit") if isinstance(raw.get("limit"), dict) else {}
    modalities = raw.get("modalities") if isinstance(raw.get("modalities"), dict) else {}
    input_parts = modalities.get("input")
    levels = _thinking_levels(raw)
    cost = raw.get("cost") if isinstance(raw.get("cost"), dict) else {}
    spec: dict[str, Any] = {
        "id": identifier,
        "name": str(raw.get("name") or identifier),
        "contextWindow": limits.get("context"),
        "maxTokens": limits.get("output"),
        "reasoning": bool(raw.get("reasoning")) or bool(levels),
        "input": list(input_parts) if isinstance(input_parts, list) and input_parts else ["text"],
        "cost": {
            "input": cost.get("input"),
            "output": cost.get("output"),
            "cacheRead": cost.get("cache_read"),
            "cacheWrite": cost.get("cache_write"),
        },
        "releaseDate": str(raw.get("release_date") or ""),
        "deprecated": raw.get("status") == "deprecated",
    }
    if levels:
        spec["thinkingLevels"] = list(levels)
    return model_from_spec(provider_id, spec, api=api, base_url=base_url)


def _thinking_levels(raw: Mapping[str, Any]) -> tuple[str, ...]:
    options = raw.get("reasoning_options")
    if not isinstance(options, list):
        return ()
    levels: list[str] = []
    for option in options:
        if not isinstance(option, dict) or option.get("type") != "effort":
            continue
        values = option.get("values")
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str):
                level = value.strip().lower()
                if level in _THINKING_LEVELS and level not in levels:
                    levels.append(level)
    return tuple(levels)


def _is_non_chat(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(pattern in lowered for pattern in NON_CHAT_ID_PATTERNS)


def _bearer_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def fetch_models(
    spec: ProviderSpec, api_key: str | None, *, client: Client | None = None, timeout: float = DEFAULT_TIMEOUT
) -> list[Model]:
    """GET ``spec.models_url`` and parse the endpoint's catalog."""
    payload = get_json(spec.models_url, headers=_bearer_headers(api_key), client=client, timeout=timeout)
    return [
        model_from_spec(
            spec.id,
            entry,
            api=spec.api,
            base_url=provider_base_url(spec.id) or spec.base_url,
            context_window=spec.context_window,
            max_tokens=spec.max_tokens,
        )
        for entry in parse_models(payload)
    ]


def fetch_openai_compatible_models(
    provider: str, base_url: str, api_key: str | None, *, client: Client | None = None, timeout: float = DEFAULT_TIMEOUT
) -> list[Model]:
    """List ``{base_url}/models`` for a user-defined OpenAI-compatible endpoint."""
    url = base_url.rstrip("/") + "/models"
    payload = get_json(url, headers=_bearer_headers(api_key), client=client, timeout=timeout)
    return [
        model_from_spec(
            provider,
            entry,
            api="openai-completions",
            base_url=base_url,
            context_window=DEFAULT_CONTEXT_WINDOW,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        for entry in parse_models(payload)
    ]


def parse_models(payload: Any) -> list[dict[str, Any]]:
    """Normalize an OpenAI-style models response into catalog entries (newest first)."""
    entries: list[dict[str, Any]] = []
    items = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        model_id = model_id.strip()
        if _is_non_chat(model_id):
            continue
        entry: dict[str, Any] = {"id": model_id, "name": str(item.get("name") or model_id)}
        context = item.get("context_length") or item.get("context_window")
        if isinstance(context, (int, float)) and context > 0:
            entry["contextWindow"] = int(context)
        output = item.get("max_completion_tokens") or item.get("max_output_tokens") or item.get("max_tokens")
        if isinstance(output, (int, float)) and output > 0:
            entry["maxTokens"] = int(output)
        modalities = item.get("input_modalities")
        if isinstance(modalities, list) and modalities:
            entry["input"] = [str(part) for part in modalities if isinstance(part, str)]
        effort = item.get("effort")
        levels = effort.get("supported_levels") if isinstance(effort, dict) else None
        if isinstance(levels, list):
            supported = [str(level).strip().lower() for level in levels if isinstance(level, str) and level.strip()]
            if supported:
                entry["thinkingLevels"] = supported
        parameters = item.get("supported_parameters")
        if isinstance(parameters, list) and "reasoning" in parameters:
            entry["reasoning"] = True
        created = item.get("created")
        if isinstance(created, (int, float)):
            entry["created"] = float(created)
        pricing = item.get("pricing")
        if isinstance(pricing, dict):
            # OpenRouter prices are USD per token; tynex tracks USD per million.
            p_in = _as_float(pricing.get("prompt")) * 1_000_000
            p_out = _as_float(pricing.get("completion")) * 1_000_000
            if p_in or p_out:
                entry["cost"] = {"input": p_in, "output": p_out}
        entries.append(entry)
    if any("created" in entry for entry in entries):
        entries = sorted(entries, key=lambda entry: entry.get("created", 0.0), reverse=True)
    for entry in entries:
        entry.pop("created", None)
    return entries


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
