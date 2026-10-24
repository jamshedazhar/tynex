"""Model registry and provider metadata."""

from dataclasses import dataclass, field
from typing import Any

# Used when a models API does not report limits (and for settings entries that
# omit them). Tune per model via settings.json -> providers.<name>.models.
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

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.id}"


@dataclass(frozen=True)
class ProviderSpec:
    """Everything Tynex needs to talk to a provider — except its model list."""

    id: str
    api: str
    base_url: str
    api_key_env: tuple[str, ...]
    models_url: str
    models_style: str  # "openai" | "anthropic" | "google" | "copilot"
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS


# (id, api, base_url, api_key_env, models_url, models_style, context_window, max_tokens)
_PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        "anthropic",
        "anthropic-messages",
        "https://api.anthropic.com",
        ("ANTHROPIC_API_KEY",),
        "https://api.anthropic.com/v1/models",
        "anthropic",
        200_000,
        32_000,
    ),
    ProviderSpec(
        "openai",
        "openai-completions",
        "https://api.openai.com/v1",
        ("OPENAI_API_KEY",),
        "https://api.openai.com/v1/models",
        "openai",
    ),
    ProviderSpec(
        "google",
        "google-generative-ai",
        "https://generativelanguage.googleapis.com",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "https://generativelanguage.googleapis.com/v1beta/models",
        "google",
        1_048_576,
        65_536,
    ),
    ProviderSpec(
        "xai",
        "openai-completions",
        "https://api.x.ai/v1",
        ("XAI_API_KEY",),
        "https://api.x.ai/v1/models",
        "openai",
        131_072,
        32_000,
    ),
    ProviderSpec(
        "deepseek",
        "openai-completions",
        "https://api.deepseek.com",
        ("DEEPSEEK_API_KEY",),
        "https://api.deepseek.com/models",
        "openai",
        128_000,
        8_192,
    ),
    ProviderSpec(
        "groq",
        "openai-completions",
        "https://api.groq.com/openai/v1",
        ("GROQ_API_KEY",),
        "https://api.groq.com/openai/v1/models",
        "openai",
        131_072,
        32_768,
    ),
    ProviderSpec(
        "mistral",
        "openai-completions",
        "https://api.mistral.ai/v1",
        ("MISTRAL_API_KEY",),
        "https://api.mistral.ai/v1/models",
        "openai",
        131_072,
        32_768,
    ),
    ProviderSpec(
        "openrouter",
        "openai-completions",
        "https://openrouter.ai/api/v1",
        ("OPENROUTER_API_KEY",),
        "https://openrouter.ai/api/v1/models",
        "openai",
    ),
    ProviderSpec(
        "ollama",
        "openai-completions",
        "http://localhost:11434/v1",
        (),
        "http://localhost:11434/v1/models",
        "openai",
        32_768,
        8_192,
    ),
    ProviderSpec(
        "lmstudio",
        "openai-completions",
        "http://localhost:1234/v1",
        (),
        "http://localhost:1234/v1/models",
        "openai",
        32_768,
        8_192,
    ),
    # OAuth device flow; the catalog is fetched at login and cached in auth.json (tynex.coding.signin).
    ProviderSpec(
        "github-copilot", "github-copilot", "https://api.githubcopilot.com", ("GITHUB_COPILOT_TOKEN",), "", "copilot"
    ),
)

_PROVIDER_INDEX: dict[str, ProviderSpec] = {spec.id: spec for spec in _PROVIDER_SPECS}

# Providers that accept arbitrary model ids even when not in their catalog
# (model ids may contain "/"; context window falls back to the defaults).
OPEN_MODEL_PROVIDERS = {"openrouter", "ollama", "lmstudio"}

# Custom model entries merged in at startup (settings.json).
_CUSTOM_MODELS: list[Model] = []
_CUSTOM_API_KEYS_ENV: dict[str, list[str]] = {}
_CUSTOM_BASE_URLS: dict[str, str] = {}

# Per-provider dynamic catalogs: fetched from the provider's models API and/or
# cached on disk (see tynex.coding.model_catalog); copilot registers its own.
_DYNAMIC_MODELS: dict[str, list[Model]] = {}


def provider_spec(provider: str) -> ProviderSpec | None:
    """Metadata for a built-in provider, or ``None`` for custom-only providers."""
    return _PROVIDER_INDEX.get(provider)


def provider_base_url(provider: str) -> str | None:
    custom = _CUSTOM_BASE_URLS.get(provider)
    if custom:
        return custom
    spec = _PROVIDER_INDEX.get(provider)
    return spec.base_url if spec else None


def provider_api_key_env(provider: str) -> list[str]:
    if provider in _CUSTOM_API_KEYS_ENV:
        return list(_CUSTOM_API_KEYS_ENV[provider])
    spec = _PROVIDER_INDEX.get(provider)
    if spec is not None:
        return list(spec.api_key_env)
    return [f"{provider.upper()}_API_KEY"]


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


def register_custom_providers(providers: dict[str, Any]) -> None:
    """Merge ``settings["providers"]`` into the registry."""
    for provider, cfg in (providers or {}).items():
        base_url = cfg.get("baseUrl")
        if base_url:
            _CUSTOM_BASE_URLS[provider] = base_url
        if cfg.get("apiKeyEnv"):
            _CUSTOM_API_KEYS_ENV[provider] = list(cfg["apiKeyEnv"])
        api = cfg.get("api", "openai-completions")
        spec = _PROVIDER_INDEX.get(provider)
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
        return _latest_aliases_first(list(custom))
    overrides = {m.id: m for m in custom}
    fetched_ids = {m.id for m in fetched}
    merged = [overrides.get(m.id, m) for m in fetched] + [m for m in custom if m.id not in fetched_ids]
    return _latest_aliases_first(merged)


def _latest_aliases_first(models: list[Model]) -> list[Model]:
    """Put rolling ``-latest`` aliases first.

    Versioned ids can be retired for new API keys while the catalog still lists
    them (Gemini's ``gemini-2.5-flash`` answers 404: "use models/gemini-3.8-flash").
    The alias always points at the current model, so it is the safe default and
    the first row in the picker.
    """
    aliases = [model for model in models if model.id.endswith("-latest")]
    if not aliases:
        return models
    return aliases + [model for model in models if not model.id.endswith("-latest")]


def default_model_for_provider(provider: str) -> Model | None:
    """The catalog model auto-selected for ``provider`` when no model is given."""
    models = models_for_provider(provider)
    return models[0] if models else None


def builtin_models() -> list[Model]:
    models = [m for spec in _PROVIDER_SPECS for m in models_for_provider(spec.id)]
    known = {m.provider for m in models}
    return models + [m for m in _CUSTOM_MODELS if m.provider not in known]


def providers() -> list[str]:
    return list(dict.fromkeys([spec.id for spec in _PROVIDER_SPECS] + [m.provider for m in _CUSTOM_MODELS]))


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
            f"No models known for {provider_filter!r} yet — run `tynex login {provider_filter}` to fetch its catalog."
        )
    exact = next((m for m in catalog if m.id == pattern and provider_filter in (None, m.provider)), None)
    if exact is not None:
        return exact
    if provider_filter in OPEN_MODEL_PROVIDERS:
        # Open providers accept any model id; synthesize a catalog entry if unknown.
        return Model(
            id=pattern,
            name=pattern,
            provider=provider_filter,
            api="openai-completions",
            context_window=DEFAULT_CONTEXT_WINDOW,
            max_tokens=DEFAULT_MAX_TOKENS,
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
