"""Provider and model catalogs: the shared catalog cache plus live local endpoints."""

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tynex.ai.model_api import CATALOG_URL, fetch_catalog, fetch_models, parse_catalog
from tynex.ai.models import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_TOKENS,
    Model,
    ProviderSpec,
    model_from_spec,
    model_to_spec,
    models_for_provider,
    provider_base_url,
    provider_spec,
    providers,
    register_models,
    register_provider_specs,
    resolve_model,
)
from tynex.coding.settings import detect_configured_provider, models_cache_path, resolve_api_key

CACHE_TTL_SECONDS = 24 * 60 * 60
# v4: per-provider entries with limits, pricing, modalities and thinking levels.
CACHE_VERSION = 4
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
_LIVE_MODEL_PROVIDERS = {"ollama", "lmstudio", "faux"}

_catalog_loaded_at: float | None = None
_catalog_note: str | None = None


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
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a crash cannot truncate the cache.
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    os.replace(temporary, path)


def _versioned(data: dict[str, Any]) -> bool:
    return data.get("version") == CACHE_VERSION


def _providers_section(data: dict[str, Any]) -> dict[str, Any]:
    section = data.get("providers")
    return section if isinstance(section, dict) else {}


def _catalog_fetched_at(data: dict[str, Any]) -> float:
    catalog = data.get("catalog") if isinstance(data.get("catalog"), dict) else {}
    timestamps: list[float] = []
    with contextlib.suppress(TypeError, ValueError):
        timestamps.append(float(catalog.get("fetchedAt") or 0.0))
    for entry in _providers_section(data).values():
        if isinstance(entry, dict):
            with contextlib.suppress(TypeError, ValueError):
                timestamps.append(float(entry.get("fetchedAt") or 0.0))
    return max(timestamps, default=0.0)


def _models_from_entry(provider: str, entry: dict[str, Any], spec: ProviderSpec | None) -> list[Model]:
    raw = entry.get("models")
    if not isinstance(raw, list):
        return []
    api = str(entry.get("api") or (spec.api if spec else "openai-completions"))
    base_url = entry.get("baseUrl") or (spec.base_url if spec else None) or provider_base_url(provider)
    return [
        model_from_spec(
            provider,
            item,
            api=api,
            base_url=base_url,
            context_window=spec.context_window if spec else DEFAULT_CONTEXT_WINDOW,
            max_tokens=spec.max_tokens if spec else DEFAULT_MAX_TOKENS,
        )
        for item in raw
        if isinstance(item, dict) and item.get("id")
    ]


def cached_models(provider: str) -> tuple[list[Model], float] | None:
    """``provider``'s cached catalog and fetch time, or ``None``."""
    data = _read_cache()
    if not _versioned(data):
        return None  # written by an older tynex; refetch instead of guessing fields
    entry = _providers_section(data).get(provider)
    if not isinstance(entry, dict):
        return None
    models = _models_from_entry(provider, entry, provider_spec(provider))
    if not models:
        return None
    try:
        fetched_at = float(entry.get("fetchedAt") or 0.0)
    except (TypeError, ValueError):
        fetched_at = 0.0
    return models, fetched_at


def store_cached(provider: str, models: list[Model]) -> None:
    """Persist one provider's catalog (local endpoints, copilot, tests)."""
    data = _read_cache()
    if not _versioned(data):
        data = {"version": CACHE_VERSION, "providers": {}}
    section = _providers_section(data)
    data["providers"] = section
    entry = section.get(provider)
    entry = dict(entry) if isinstance(entry, dict) else {}
    entry["fetchedAt"] = time.time()
    entry["models"] = [model_to_spec(m) for m in models]
    section[provider] = entry
    _write_cache(data)


def _is_local(url: str) -> bool:
    return (urlparse(url).hostname or "") in _LOCAL_HOSTS


def _cached_provider_specs(data: dict[str, Any]) -> list[ProviderSpec]:
    specs: list[ProviderSpec] = []
    for provider, entry in _providers_section(data).items():
        if not isinstance(entry, dict):
            continue
        known = provider_spec(provider)
        base_url = entry.get("baseUrl") or (known.base_url if known is not None else "")
        api = entry.get("api") or (known.api if known is not None else "")
        if not base_url or not api:
            continue
        env = entry.get("env")
        specs.append(
            ProviderSpec(
                id=provider,
                name=str(entry.get("name") or (known.name if known is not None else provider)),
                api=str(api),
                base_url=str(base_url),
                api_key_env=tuple(str(name) for name in env if isinstance(name, str))
                if isinstance(env, list)
                else (known.api_key_env if known is not None else ()),
                local=bool(entry.get("local")) or _is_local(str(base_url)),
            )
        )
    return specs


def _register_cached(data: dict[str, Any]) -> None:
    register_provider_specs(_cached_provider_specs(data))
    for provider, entry in _providers_section(data).items():
        if provider in _LIVE_MODEL_PROVIDERS or not isinstance(entry, dict):
            continue
        models = _models_from_entry(provider, entry, provider_spec(provider))
        if models:
            register_models(provider, models)


def _store_catalog(catalog: list[Any], now: float) -> dict[str, Any]:
    data = _read_cache()
    if not _versioned(data):
        data = {"version": CACHE_VERSION, "providers": {}}
    section = _providers_section(data)
    data["providers"] = section
    for item in catalog:
        spec = item.spec
        entry = section.get(spec.id)
        entry = dict(entry) if isinstance(entry, dict) else {}
        entry.update(
            {
                "name": spec.name,
                "api": spec.api,
                "baseUrl": spec.base_url,
                "env": list(spec.api_key_env),
                "local": spec.local,
                "fetchedAt": now,
                "models": [model_to_spec(model) for model in item.models],
            }
        )
        section[spec.id] = entry
    data["catalog"] = {"url": CATALOG_URL, "fetchedAt": now}
    _write_cache(data)
    return data


def ensure_catalog(
    settings: dict[str, Any] | None = None, *, now: float | None = None, force: bool = False, allow_fetch: bool = True
) -> str | None:
    """Load the shared catalog, refreshing the cache when it is stale or missing."""
    global _catalog_loaded_at, _catalog_note
    moment = time.time() if now is None else now
    if _catalog_loaded_at is not None and not force:
        return _catalog_note
    data = _read_cache()
    cached = _versioned(data) and bool(_providers_section(data))
    fresh = cached and (moment - _catalog_fetched_at(data)) < CACHE_TTL_SECONDS
    if fresh and not force:
        _register_cached(data)
        _catalog_loaded_at, _catalog_note = moment, None
        return None
    if not allow_fetch:
        note = None if cached else "no cached model catalog"
    else:
        try:
            catalog = parse_catalog(fetch_catalog())
        except Exception as exc:  # HTTP/JSON errors, malformed payloads
            note = f"could not fetch the model catalog ({exc})"
        else:
            if catalog:
                data = _store_catalog(catalog, moment)
                note = None
            else:
                note = "the model catalog listed no usable providers"
        if note is not None and cached:
            note += " — using the cached catalog"
    if cached or note is None:
        _register_cached(data)
    _catalog_loaded_at, _catalog_note = moment, note
    return note


def reset_catalog_state() -> None:
    """Forget the in-process catalog load (used by tests and embedders)."""
    global _catalog_loaded_at, _catalog_note
    _catalog_loaded_at = None
    _catalog_note = None


def _ensure_live_models(
    provider: str, spec: ProviderSpec, settings: dict[str, Any] | None, moment: float
) -> tuple[list[Model], str | None]:
    cached = cached_models(provider)
    if cached is not None and (moment - cached[1]) < CACHE_TTL_SECONDS:
        register_models(provider, cached[0])
        return models_for_provider(provider), None
    api_key = resolve_api_key(provider, settings)
    note: str | None = None
    if not api_key and not spec.local and not _is_local(spec.models_url):
        note = f"no credentials for {provider} — run `tynex login {provider}`"
    else:
        try:
            fetched = fetch_models(spec, api_key)
        except Exception as exc:  # HTTP/JSON errors, malformed payloads
            note = f"could not fetch models from {spec.models_url} ({exc})"
            if cached:
                note += " — using the cached catalog"
        else:
            if fetched:
                register_models(provider, fetched)
                store_cached(provider, fetched)
                return models_for_provider(provider), None
            note = f"{spec.models_url} listed no chat models"
    if cached:
        register_models(provider, cached[0])
        return models_for_provider(provider), note
    return [], note


def ensure_models(
    provider: str, settings: dict[str, Any] | None = None, *, now: float | None = None
) -> tuple[list[Model], str | None]:
    """Make ``provider``'s catalog available in the registry."""
    models = models_for_provider(provider)
    if models:
        return models, None
    spec = provider_spec(provider)
    if spec is None:
        return [], None
    moment = time.time() if now is None else now
    if spec.models_url and (spec.local or provider in _LIVE_MODEL_PROVIDERS):
        return _ensure_live_models(provider, spec, settings, moment)
    note = ensure_catalog(settings, now=moment)
    models = models_for_provider(provider)
    if models:
        return models, note
    if spec.models_url:
        return _ensure_live_models(provider, spec, settings, moment)
    return [], note


def fetchable_providers(settings: dict[str, Any] | None = None) -> list[str]:
    """Providers whose live catalog can be fetched now (credentials or local endpoint)."""
    names: list[str] = []
    for provider in providers():
        spec = provider_spec(provider)
        if spec is None or not spec.models_url:
            continue
        if spec.local or resolve_api_key(provider, settings):
            names.append(provider)
    return names


def resolve_model_retry(
    pattern: str, settings: dict[str, Any] | None = None, *, preferred_provider: str | None = None
) -> Model:
    """``resolve_model``, loading the catalog and any missing live catalogs first."""
    preferred_provider = preferred_provider or detect_configured_provider(settings)
    try:
        return resolve_model(pattern, preferred_provider=preferred_provider)
    except ValueError:
        pass
    ensure_catalog(settings)
    preferred_provider = preferred_provider or detect_configured_provider(settings)
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
