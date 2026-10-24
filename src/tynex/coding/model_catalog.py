"""Per-provider model catalogs: fetched from the provider, cached on disk."""

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tynex.ai.model_api import fetch_models
from tynex.ai.models import (
    Model,
    model_from_spec,
    model_to_spec,
    models_for_provider,
    provider_base_url,
    provider_spec,
    providers,
    register_models,
    resolve_model,
)
from tynex.coding.settings import models_cache_path, resolve_api_key

CACHE_TTL_SECONDS = 24 * 60 * 60
# v3: catalogs record the model's supported thinking levels (``thinkingLevels``),
CACHE_VERSION = 3
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def cache_path() -> Path:
    return models_cache_path()


def _read_cache() -> dict[str, Any]:
    try:
        with open(cache_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_cache(data: dict[str, Any]) -> None:
    with open(cache_path(), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def cached_models(provider: str) -> tuple[list[Model], float] | None:
    """``provider``'s cached catalog and fetch time, or ``None``."""
    spec = provider_spec(provider)
    if spec is None:
        return None
    data = _read_cache()
    if data.get("version") != CACHE_VERSION:
        return None  # written by an older tynex; refetch instead of guessing fields
    entry = (data.get("providers") or {}).get(provider)
    if not isinstance(entry, dict):
        return None
    raw = entry.get("models")
    if not isinstance(raw, list):
        return None
    models = [
        model_from_spec(
            provider,
            item,
            api=spec.api,
            base_url=provider_base_url(provider),
            context_window=spec.context_window,
            max_tokens=spec.max_tokens,
        )
        for item in raw
        if isinstance(item, dict) and item.get("id")
    ]
    if not models:
        return None
    try:
        fetched_at = float(entry.get("fetchedAt") or 0.0)
    except (TypeError, ValueError):
        fetched_at = 0.0
    return models, fetched_at


def store_cached(provider: str, models: list[Model]) -> None:
    """Persist ``provider``'s catalog for the next process."""
    data = _read_cache()
    if data.get("version") != CACHE_VERSION:
        data = {"version": CACHE_VERSION, "providers": {}}
    section = data.setdefault("providers", {})
    if not isinstance(section, dict):
        section = {}
        data["providers"] = section
    section[provider] = {"fetchedAt": time.time(), "models": [model_to_spec(m) for m in models]}
    _write_cache(data)


def _is_local(url: str) -> bool:
    return (urlparse(url).hostname or "") in _LOCAL_HOSTS


def ensure_models(
    provider: str, settings: dict[str, Any] | None = None, *, now: float | None = None
) -> tuple[list[Model], str | None]:
    """Make ``provider``'s catalog available in the registry."""
    models = models_for_provider(provider)
    if models:
        return models, None
    spec = provider_spec(provider)
    if spec is None or spec.models_style == "copilot" or not spec.models_url:
        # Copilot catalogs are registered by tynex.coding.signin; providers
        # without a spec only ever have settings-defined models.
        return [], None
    cached = cached_models(provider)
    fresh = cached is not None and ((now if now is not None else time.time()) - cached[1]) < CACHE_TTL_SECONDS
    api_key = resolve_api_key(provider, settings)
    if not api_key and not _is_local(spec.models_url):
        note = f"no credentials for {provider} — run `tynex login {provider}`"
        if cached:
            register_models(provider, cached[0])
            return models_for_provider(provider), note
        return [], note
    if fresh and cached is not None:
        register_models(provider, cached[0])
        return models_for_provider(provider), None
    try:
        fetched = fetch_models(spec, api_key)
    except Exception as exc:  # HTTP/JSON errors, malformed payloads
        note = f"could not fetch models from {spec.models_url} ({exc})"
        if cached:
            register_models(provider, cached[0])
            return models_for_provider(provider), f"{note} — using the cached catalog"
        return [], note
    if not fetched:
        note = f"{spec.models_url} listed no chat models"
        if cached:
            register_models(provider, cached[0])
            return models_for_provider(provider), note
        return [], note
    register_models(provider, fetched)
    store_cached(provider, fetched)
    return models_for_provider(provider), None


def fetchable_providers(settings: dict[str, Any] | None = None) -> list[str]:
    """Providers whose catalogs can be fetched now (credentials or local endpoint)."""
    names: list[str] = []
    for provider in providers():
        spec = provider_spec(provider)
        if spec is None or not spec.models_url or spec.models_style == "copilot":
            continue
        if resolve_api_key(provider, settings) or _is_local(spec.models_url):
            names.append(provider)
    return names


def resolve_model_retry(
    pattern: str, settings: dict[str, Any] | None = None, *, preferred_provider: str | None = None
) -> Model:
    """``resolve_model``, fetching any missing catalogs once before giving up."""
    try:
        return resolve_model(pattern, preferred_provider=preferred_provider)
    except ValueError:
        pass
    stripped = (pattern or "").strip()
    head = stripped.split("/", 1)[0]
    attempts = [head] if "/" in stripped and head in providers() else fetchable_providers(settings)
    for provider in attempts:
        ensure_models(provider, settings)
    return resolve_model(pattern, preferred_provider=preferred_provider)
