"""Model registry and provider metadata."""

from dataclasses import dataclass, field, replace
from typing import Any

# Defaults when a catalog omits limits; override per model in settings.json.
DEFAULT_CONTEXT_WINDOW = 200_000
DEFAULT_MAX_TOKENS = 16_384


@dataclass
class Pricing:
    """USD per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass
class Model:
    id: str
    name: str
    provider: str
    api: str  # "anthropic-messages" | "openai-completions" | "google-generative-ai" | "github-copilot" | "faux"
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning: bool = False
    thinking_levels: tuple[str, ...] = ()
    input: tuple[str, ...] = ("text",)
    cost: Pricing = field(default_factory=Pricing)
    base_url: str | None = None
    release_date: str = ""
    deprecated: bool = False

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.id}"


@dataclass(frozen=True)
class ProviderSpec:
    """Everything Tynex needs to talk to a provider — except its model list."""

    id: str
    name: str
    api: str
    base_url: str
    api_key_env: tuple[str, ...] = ()
    models_url: str = ""
    local: bool = False
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS


# Providers that accept model ids outside their catalog (ids may contain "/").
OPEN_MODEL_PROVIDERS = {"openrouter", "ollama", "lmstudio"}

# Fallbacks for transport details the catalog omits.
_CORE_PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec("anthropic", "Anthropic", "anthropic-messages", "https://api.anthropic.com", ("ANTHROPIC_API_KEY",)),
    ProviderSpec("openai", "OpenAI", "openai-completions", "https://api.openai.com/v1", ("OPENAI_API_KEY",)),
    ProviderSpec(
        "google",
        "Google",
        "google-generative-ai",
        "https://generativelanguage.googleapis.com",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        context_window=1_048_576,
        max_tokens=65_536,
    ),
    ProviderSpec("xai", "xAI", "openai-completions", "https://api.x.ai/v1", ("XAI_API_KEY",)),
    ProviderSpec("deepseek", "DeepSeek", "openai-completions", "https://api.deepseek.com", ("DEEPSEEK_API_KEY",)),
    ProviderSpec("groq", "Groq", "openai-completions", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
    ProviderSpec("mistral", "Mistral", "openai-completions", "https://api.mistral.ai/v1", ("MISTRAL_API_KEY",)),
    ProviderSpec(
        "openrouter", "OpenRouter", "openai-completions", "https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",)
    ),
    ProviderSpec(
        "ollama",
        "Ollama",
        "openai-completions",
        "http://localhost:11434/v1",
        models_url="http://localhost:11434/v1/models",
        local=True,
        context_window=32_768,
        max_tokens=8_192,
    ),
    ProviderSpec(
        "lmstudio",
        "LM Studio",
        "openai-completions",
        "http://localhost:1234/v1",
        models_url="http://localhost:1234/v1/models",
        local=True,
        context_window=32_768,
        max_tokens=8_192,
    ),
    # OAuth device flow; the account catalog is cached in auth.json at login.
    ProviderSpec(
        "github-copilot", "GitHub Copilot", "github-copilot", "https://api.githubcopilot.com", ("GITHUB_COPILOT_TOKEN",)
    ),
)

_CORE_INDEX: dict[str, ProviderSpec] = {spec.id: spec for spec in _CORE_PROVIDER_SPECS}
_CORE_AUTHORITATIVE = {"ollama", "lmstudio", "github-copilot"}

_CATALOG_SPECS: dict[str, ProviderSpec] = {}
_CATALOG_ORDER: list[str] = []

# Custom model entries merged in at startup (settings.json).
_CUSTOM_MODELS: list[Model] = []
_CUSTOM_API_KEYS_ENV: dict[str, list[str]] = {}
_CUSTOM_BASE_URLS: dict[str, str] = {}
_CUSTOM_APIS: dict[str, str] = {}
_CUSTOM_NAMES: dict[str, str] = {}

# Per-provider dynamic catalogs: shared catalog, local endpoints, copilot.
_DYNAMIC_MODELS: dict[str, list[Model]] = {}


def register_provider_specs(specs: list[ProviderSpec]) -> None:
    """Replace the catalog-derived provider set (idempotent)."""
    global _CATALOG_ORDER
    known: dict[str, ProviderSpec] = {}
    for spec in specs:
        if not spec.id or spec.id in _CORE_AUTHORITATIVE:
            continue
        known[spec.id] = spec
    _CATALOG_SPECS.clear()
    _CATALOG_SPECS.update(known)
    _CATALOG_ORDER = sorted(known, key=lambda provider: (known[provider].name.casefold(), provider))


def reset_provider_specs() -> None:
    """Drop every catalog-derived provider (used by tests and embedders)."""
    global _CATALOG_ORDER
    _CATALOG_SPECS.clear()
    _CATALOG_ORDER = []


def provider_spec(provider: str) -> ProviderSpec | None:
    """Metadata for a provider, or ``None`` for custom-only providers."""
    base = _CATALOG_SPECS.get(provider) or _CORE_INDEX.get(provider)
    if base is None:
        if (
            provider in _CUSTOM_BASE_URLS
            or provider in _CUSTOM_API_KEYS_ENV
            or provider in _CUSTOM_APIS
            or provider in _CUSTOM_NAMES
        ):
            return ProviderSpec(
                id=provider,
                name=_CUSTOM_NAMES.get(provider, provider),
                api=_CUSTOM_APIS.get(provider, "openai-completions"),
                base_url=_CUSTOM_BASE_URLS.get(provider, ""),
                api_key_env=tuple(_CUSTOM_API_KEYS_ENV.get(provider, ())),
            )
        return None
    changes: dict[str, Any] = {}
    if provider in _CUSTOM_NAMES:
        changes["name"] = _CUSTOM_NAMES[provider]
    if provider in _CUSTOM_APIS:
        changes["api"] = _CUSTOM_APIS[provider]
    if provider in _CUSTOM_BASE_URLS:
        changes["base_url"] = _CUSTOM_BASE_URLS[provider]
    if provider in _CUSTOM_API_KEYS_ENV:
        changes["api_key_env"] = tuple(_CUSTOM_API_KEYS_ENV[provider])
    return replace(base, **changes) if changes else base


def provider_base_url(provider: str) -> str | None:
    custom = _CUSTOM_BASE_URLS.get(provider)
    if custom:
        return custom
    spec = provider_spec(provider)
    return spec.base_url if spec else None


def provider_api_key_env(provider: str) -> list[str]:
    if provider in _CUSTOM_API_KEYS_ENV:
        return list(_CUSTOM_API_KEYS_ENV[provider])
    spec = provider_spec(provider)
    if spec is not None:
        return list(spec.api_key_env)
    return [f"{provider.upper()}_API_KEY"]


def provider_label(provider: str) -> str:
    """Human-readable provider name for pickers and messages."""
    spec = provider_spec(provider)
    return spec.name if spec is not None and spec.name else provider


def model_from_spec(
    provider: str,
    spec: dict[str, Any],
    *,
    api: str,
    base_url: str | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Model:
    """Build a :class:`Model` from a catalog entry."""
    model_id = str(spec.get("id") or "").strip()
    if not model_id:
        raise ValueError("Model entry needs an 'id'")
    raw_input = spec.get("input")
    raw_cost = spec.get("cost")
    cost = raw_cost if isinstance(raw_cost, dict) else {}
    context, output = spec.get("contextWindow"), spec.get("maxTokens")
    raw_levels = spec.get("thinkingLevels")
    levels = (
        tuple(str(level).strip().lower() for level in raw_levels if isinstance(level, str) and level.strip())
        if isinstance(raw_levels, (list, tuple))
        else ()
    )
    return Model(
        id=model_id,
        name=str(spec.get("name") or model_id),
        provider=provider,
        api=api,
        context_window=int(context) if isinstance(context, (int, float)) and context > 0 else context_window,
        max_tokens=int(output) if isinstance(output, (int, float)) and output > 0 else max_tokens,
        # A model that advertises thinking levels reasons, even without the flag.
        reasoning=bool(spec.get("reasoning", False)) or bool(levels),
        thinking_levels=levels,
        input=tuple(str(part) for part in raw_input)
        if isinstance(raw_input, (list, tuple)) and raw_input
        else ("text",),
        cost=Pricing(*(float(cost.get(key) or 0.0) for key in ("input", "output", "cacheRead", "cacheWrite"))),
        base_url=spec.get("baseUrl") or base_url,
        release_date=str(spec.get("releaseDate") or ""),
        deprecated=bool(spec.get("deprecated", False)),
    )


def model_to_spec(model: Model) -> dict[str, Any]:
    """Serialize a model for the on-disk catalog cache (``models.json``)."""
    spec: dict[str, Any] = {
        "id": model.id,
        "name": model.name,
        "contextWindow": model.context_window,
        "maxTokens": model.max_tokens,
        "reasoning": model.reasoning,
        "input": list(model.input),
    }
    if model.thinking_levels:
        spec["thinkingLevels"] = list(model.thinking_levels)
    cost = model.cost
    if cost.input or cost.output or cost.cache_read or cost.cache_write:
        spec["cost"] = {
            "input": cost.input,
            "output": cost.output,
            "cacheRead": cost.cache_read,
            "cacheWrite": cost.cache_write,
        }
    if model.base_url:
        spec["baseUrl"] = model.base_url
    if model.release_date:
        spec["releaseDate"] = model.release_date
    if model.deprecated:
        spec["deprecated"] = True
    return spec


def register_models(provider: str, models: list[Model]) -> None:
    """Replace the dynamic model set for ``provider`` (idempotent)."""
    _DYNAMIC_MODELS[provider] = list(models)


def reset_dynamic_models() -> None:
    """Drop all dynamic model sets (used by tests and embedders)."""
    _DYNAMIC_MODELS.clear()


def reset_custom_providers() -> None:
    """Drop every custom provider registration (used by tests and embedders)."""
    _CUSTOM_MODELS.clear()
    _CUSTOM_API_KEYS_ENV.clear()
    _CUSTOM_BASE_URLS.clear()
    _CUSTOM_APIS.clear()
    _CUSTOM_NAMES.clear()


def register_custom_providers(providers: dict[str, Any]) -> None:
    """Merge ``settings["providers"]`` into the registry."""
    for provider, cfg in (providers or {}).items():
        if not isinstance(cfg, dict):
            continue
        base_url = cfg.get("baseUrl")
        if base_url:
            _CUSTOM_BASE_URLS[provider] = base_url
        if cfg.get("apiKeyEnv"):
            _CUSTOM_API_KEYS_ENV[provider] = list(cfg["apiKeyEnv"])
        if cfg.get("api"):
            _CUSTOM_APIS[provider] = str(cfg["api"])
        if cfg.get("name"):
            _CUSTOM_NAMES[provider] = str(cfg["name"])
        api = cfg.get("api", "openai-completions")
        spec = provider_spec(provider)
        for entry in cfg.get("models", []) or []:
            _CUSTOM_MODELS.append(
                model_from_spec(
                    provider,
                    entry,
                    api=api,
                    base_url=base_url or (spec.base_url if spec else None),
                    context_window=spec.context_window if spec else DEFAULT_CONTEXT_WINDOW,
                    max_tokens=spec.max_tokens if spec else DEFAULT_MAX_TOKENS,
                )
            )


def models_for_provider(provider: str) -> list[Model]:
    """The provider's models: fetched catalog overlaid with settings entries that pin metadata for the same id."""
    fetched = _DYNAMIC_MODELS.get(provider) or []
    custom = [m for m in _CUSTOM_MODELS if m.provider == provider]
    if not fetched:
        return _ordered(list(custom))
    overrides = {m.id: m for m in custom}
    fetched_ids = {m.id for m in fetched}
    merged = [overrides.get(m.id, m) for m in fetched] + [m for m in custom if m.id not in fetched_ids]
    return _ordered(merged)


def _release_key(model: Model) -> str:
    return model.release_date or ""


def _ordered(models: list[Model]) -> list[Model]:
    """``-latest`` aliases first, then newest, then deprecated models."""
    aliases = [model for model in models if model.id.endswith("-latest")]
    rest = [model for model in models if not model.id.endswith("-latest")]
    active = sorted((model for model in rest if not model.deprecated), key=_release_key, reverse=True)
    deprecated = sorted((model for model in rest if model.deprecated), key=_release_key, reverse=True)
    return aliases + active + deprecated


def default_model_for_provider(provider: str) -> Model | None:
    """The catalog model auto-selected for ``provider`` when no model is given."""
    for model in models_for_provider(provider):
        if not model.deprecated:
            return model
    return None


def builtin_models() -> list[Model]:
    models = [m for spec in providers() for m in models_for_provider(spec)]
    known = {m.provider for m in models}
    return models + [m for m in _CUSTOM_MODELS if m.provider not in known]


def providers() -> list[str]:
    """Known provider ids, built-ins first."""
    core = [spec.id for spec in _CORE_PROVIDER_SPECS]
    catalog = [provider for provider in _CATALOG_ORDER if provider not in _CORE_INDEX]
    custom = sorted(
        (set(_CUSTOM_BASE_URLS) | set(_CUSTOM_API_KEYS_ENV) | set(_CUSTOM_APIS) | {m.provider for m in _CUSTOM_MODELS})
        - set(core)
        - set(catalog)
    )
    return core + catalog + custom


def resolve_model(pattern: str, preferred_provider: str | None = None) -> Model:
    """Resolve a model reference such as ``provider/model-id``, a unique partial id, or a glob."""
    pattern = (pattern or "").strip()
    if not pattern:
        raise ValueError("Empty model reference")
    catalog = builtin_models()
    provider_filter: str | None = None
    if "/" in pattern:
        head, rest = pattern.split("/", 1)
        if head in OPEN_MODEL_PROVIDERS and head != rest.split("/", 1)[0]:
            # Model ids under open providers may contain "/" (openrouter/<vendor>/<model>).
            provider_filter, pattern = head, rest
        elif head in providers():
            provider_filter, pattern = head, rest
        # else: the whole pattern is a model id containing a slash; match literally.
    if provider_filter and provider_filter not in OPEN_MODEL_PROVIDERS and not models_for_provider(provider_filter):
        raise ValueError(
            f"No models known for {provider_filter!r} yet — run `tynex models {provider_filter}` to load its catalog."
        )
    exact = next((m for m in catalog if m.id == pattern and provider_filter in (None, m.provider)), None)
    if exact is not None:
        return exact
    if provider_filter in OPEN_MODEL_PROVIDERS:
        # Open providers accept any model id; synthesize a catalog entry if unknown.
        spec = provider_spec(provider_filter)
        return Model(
            id=pattern,
            name=pattern,
            provider=provider_filter,
            api=spec.api if spec is not None else "openai-completions",
            context_window=spec.context_window if spec is not None else DEFAULT_CONTEXT_WINDOW,
            max_tokens=spec.max_tokens if spec is not None else DEFAULT_MAX_TOKENS,
            input=("text", "image"),
            base_url=provider_base_url(provider_filter),
        )
    needle = pattern.lower()
    matches = [
        m
        for m in catalog
        if (not provider_filter or m.provider == provider_filter)
        and (needle in f"{m.provider}/{m.id}".lower() or needle in m.name.lower())
    ]
    live = [m for m in matches if not m.deprecated]
    if live:
        matches = live
    preferred = next((m for m in matches if m.provider == preferred_provider), None)
    if preferred is not None:
        return preferred
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No model matches {pattern!r}. Use /model to list available models.")
    ids = ", ".join(m.key for m in matches[:8])
    raise ValueError(f"Model reference {pattern!r} is ambiguous: {ids}")


def format_model(model: Model) -> str:
    return f"{model.provider}/{model.id}"


def format_tokens(count: int) -> str:
    """Compact token/count formatting for footers and model lists."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)


def format_price(value: float) -> str:
    """A per-million-token price without trailing zeroes (``3``, ``0.25``)."""
    if value <= 0:
        return "0"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def format_model_details(model: Model) -> str:
    """One-line catalog details for pickers and ``tynex models``."""
    parts = [f"{format_tokens(model.context_window)} ctx"]
    if model.max_tokens:
        parts.append(f"{format_tokens(model.max_tokens)} out")
    cost = model.cost
    if cost.input or cost.output:
        pricing = f"${format_price(cost.input)}/${format_price(cost.output)} per M"
        if cost.cache_read:
            pricing += f" · ${format_price(cost.cache_read)} cached"
        parts.append(pricing)
    modalities = [part for part in model.input if part != "text"]
    if modalities:
        parts.append("/".join(modalities))
    if model.reasoning:
        levels = ", ".join(model.thinking_levels)
        parts.append(f"reasoning ({levels})" if levels else "reasoning")
    if model.deprecated:
        parts.append("deprecated")
    return " · ".join(parts)
