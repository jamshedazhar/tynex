"""Fetch provider model catalogs from their own models APIs."""

from datetime import datetime
from typing import Any

from tynex.ai.models import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_TOKENS,
    Model,
    ProviderSpec,
    model_from_spec,
    provider_base_url,
)
from tynex.http import Client, get_json

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_TIMEOUT = 15.0

# OpenAI-style catalogs mix chat models with embeddings/audio/media endpoints
# that this harness cannot chat with; ids are filtered by substring. Define the
# model explicitly in settings.json (providers.<name>.models) to force-include
# a filtered model.
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


def _is_non_chat(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(pattern in lowered for pattern in NON_CHAT_ID_PATTERNS)


def _bearer_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _headers(spec: ProviderSpec, api_key: str | None) -> dict[str, str]:
    if spec.models_style == "anthropic":
        headers = {"anthropic-version": ANTHROPIC_VERSION}
        if api_key:
            headers["x-api-key"] = api_key
        return headers
    if spec.models_style == "google":
        return {"x-goog-api-key": api_key} if api_key else {}
    return _bearer_headers(api_key)


def fetch_models(
    spec: ProviderSpec, api_key: str | None, *, client: Client | None = None, timeout: float = DEFAULT_TIMEOUT
) -> list[Model]:
    """GET ``spec.models_url`` and parse the provider's catalog."""
    payload = get_json(spec.models_url, headers=_headers(spec, api_key), client=client, timeout=timeout)
    return [
        model_from_spec(
            spec.id,
            entry,
            api=spec.api,
            base_url=provider_base_url(spec.id),
            context_window=spec.context_window,
            max_tokens=spec.max_tokens,
        )
        for entry in parse_catalog(spec.models_style, payload)
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
        for entry in parse_catalog("openai", payload)
    ]


def parse_catalog(style: str, payload: Any) -> list[dict[str, Any]]:
    """Normalize a models-API response into catalog entries (newest first)."""
    if style == "anthropic":
        entries = _parse_anthropic(payload)
    elif style == "google":
        entries = _parse_google(payload)
    else:
        entries = _parse_openai(payload)
    return _newest_first(entries)


def _iter_items(payload: Any, key: str) -> list[Any]:
    if isinstance(payload, dict):
        items = payload.get(key)
        return items if isinstance(items, list) else []
    return payload if isinstance(payload, list) else []


def _parse_openai(payload: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in _iter_items(payload, "data"):
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
        output = item.get("max_completion_tokens") or item.get("max_tokens")
        if isinstance(output, (int, float)) and output > 0:
            entry["maxTokens"] = int(output)
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
    return entries


def _parse_anthropic(payload: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in _iter_items(payload, "data"):
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        entry: dict[str, Any] = {"id": model_id.strip(), "name": str(item.get("display_name") or model_id.strip())}
        created = _as_epoch(item.get("created_at"))
        if created:
            entry["created"] = created
        entries.append(entry)
    return entries


def _parse_google(payload: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in _iter_items(payload, "models"):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        model_id = name.strip().removeprefix("models/")
        if not model_id:
            continue
        methods = item.get("supportedGenerationMethods")
        if isinstance(methods, list) and "generateContent" not in methods:
            continue  # embeddings etc. cannot be used for chat
        entry: dict[str, Any] = {"id": model_id, "name": str(item.get("displayName") or model_id)}
        context = item.get("inputTokenLimit")
        if isinstance(context, (int, float)) and context > 0:
            entry["contextWindow"] = int(context)
        output = item.get("outputTokenLimit")
        if isinstance(output, (int, float)) and output > 0:
            entry["maxTokens"] = int(output)
        if item.get("thinking") is True:
            entry["reasoning"] = True
        entries.append(entry)
    return entries


def _newest_first(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _as_epoch(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0
