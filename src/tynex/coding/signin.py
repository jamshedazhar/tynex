"""Provider sign-in and credential commands shared by the interactive TUI and the CLI."""

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tynex.ai import models as model_registry
from tynex.ai.model_api import fetch_openai_compatible_models
from tynex.ai.models import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_TOKENS,
    Model,
    default_model_for_provider,
    format_model,
    provider_api_key_env,
    provider_label,
    providers,
    register_custom_providers,
)
from tynex.ai.providers import copilot as copilot_api
from tynex.coding.model_catalog import ensure_catalog, ensure_models
from tynex.coding.settings import (
    load_auth,
    remove_credential,
    resolve_api_key,
    save_auth,
    store_api_key,
    store_custom_provider,
)
from tynex.http import HTTPError

if TYPE_CHECKING:
    from rich.console import Console

    from tynex.coding.agent_session import AgentSession

# Providers that never need credentials (local/offline): no sign-in required.
NO_KEY_PROVIDERS = ("faux", "ollama", "lmstudio")

# Providers that sign in with a browser device flow (OAuth-style).
OAUTH_PROVIDERS = ("github-copilot",)

_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")

# ask(text, *, password=False) -> one line of input ("" to cancel).
AskFn = Callable[..., str]
# select(title, options, *, selected=0) -> chosen index, or None when cancelled.
SelectFn = Callable[..., "int | None"]
# switch(provider) -> the model selected after sign-in, or None.
SwitchFn = Callable[[str], "Model | None"]
EchoFn = Callable[[str], None]


def oauth_providers() -> list[str]:
    """Registered providers that sign in with a device flow."""
    known = set(providers())
    return [name for name in OAUTH_PROVIDERS if name in known]


def api_key_providers() -> list[str]:
    """Registered providers that take an API key."""
    return [name for name in providers() if name not in OAUTH_PROVIDERS and name not in NO_KEY_PROVIDERS]


def login_hint(provider: str) -> str:
    """Short description of what signing in to ``provider`` needs."""
    if provider in OAUTH_PROVIDERS:
        return "browser device flow"
    if provider in NO_KEY_PROVIDERS:
        return "no sign-in needed (local)"
    envs = provider_api_key_env(provider)
    if not envs:
        return "no key needed (local)"
    return f"API key or {envs[0]}"


def signed_in_providers() -> list[str]:
    """Providers with a key stored in ``~/.tynex/agent/auth.json`` (file order)."""
    found: list[str] = []
    for provider, entry in (load_auth() or {}).items():
        if isinstance(entry, dict):
            entry = entry.get("key")
        if isinstance(entry, str) and entry:
            found.append(provider)
    return found


def interactive_login(
    *, ask: AskFn, select: SelectFn, echo: EchoFn, settings: dict[str, Any] | None = None
) -> str | None:
    """The ``tynex login`` menu: OAuth, API key or a custom OpenAI-compatible provider."""
    ensure_catalog(settings)
    oauth = oauth_providers()
    oauth_label = "OAuth — " + ", ".join(provider_label(name) for name in oauth) if oauth else "OAuth"
    method = select("How do you want to sign in?", [oauth_label, "API key", "Add custom provider (OpenAI-compatible)"])
    if method is None:
        return None
    if method == 0:
        if not oauth:
            echo("No OAuth providers are available.")
            return None
        provider = oauth[0]
        if len(oauth) > 1:
            index = select("OAuth provider:", [provider_label(name) for name in oauth])
            if index is None:
                return None
            provider = oauth[index]
        if provider == "github-copilot":
            if not copilot_login(ask=ask, echo=echo, settings=settings):
                return None
            return provider
        echo(f"No sign-in flow is available for {provider_label(provider)} yet.")
        return None
    if method == 1:
        choices = api_key_providers()
        index = select(
            "API key for which provider?", [f"{provider_label(name)}  —  {login_hint(name)}" for name in choices]
        )
        if index is None:
            return None
        provider = choices[index]
        key = ask(f"{provider} API key: ", password=True).strip()
        if not key:
            return None
        store_api_key(provider, key)
        echo(f"Stored the API key for {provider} in ~/.tynex/agent/auth.json.")
        return provider
    return add_custom_provider(ask=ask, select=select, echo=echo)


def add_custom_provider(*, ask: AskFn, select: SelectFn, echo: EchoFn) -> str | None:
    """Ask for an OpenAI-compatible chat-completions endpoint and store it."""
    while True:
        provider = ask("Provider name (e.g. my-llm): ").strip().lower()
        if not provider:
            return None
        if _NAME_PATTERN.fullmatch(provider):
            break
        echo("Provider names must start with a letter or digit — letters, digits, '-' and '_' only.")
    while True:
        base_url = ask("Base URL of the chat completions endpoint (e.g. https://my-llm.example.com/v1): ").strip()
        if not base_url:
            return None
        if base_url.startswith(("http://", "https://")):
            break
        echo("Base URL must start with http:// or https://.")
    key = ask(f"API key for {provider} (empty when the endpoint needs none): ", password=True).strip()
    models = _fetch_custom_models(provider, base_url, key or None, echo)
    model_spec: dict[str, Any] | None = None
    if models:
        options = [f"{m.id}  —  {m.name}, {m.context_window:,} ctx" if m.name != m.id else m.id for m in models]
        index = select(f"Which {provider} model should tynex use?", options)
        if index is None:
            return None
        chosen = models[index]
        model_spec = {
            "id": chosen.id,
            "name": chosen.name,
            "contextWindow": chosen.context_window,
            "maxTokens": chosen.max_tokens,
        }
    else:
        model_id = ask("Model id (as served by the endpoint): ").strip()
        if not model_id:
            return None
        model_spec = {
            "id": model_id,
            "name": model_id,
            "contextWindow": DEFAULT_CONTEXT_WINDOW,
            "maxTokens": DEFAULT_MAX_TOKENS,
        }
    config: dict[str, Any] = {"api": "openai-completions", "baseUrl": base_url, "models": [model_spec]}
    store_custom_provider(provider, config)
    register_custom_providers({provider: config})
    if key:
        store_api_key(provider, key)
    echo(f"Added {provider} — model {provider}/{model_spec['id']} via {base_url}.")
    echo("Saved to ~/.tynex/agent/settings.json" + (" and auth.json." if key else "."))
    return provider


def _fetch_custom_models(provider: str, base_url: str, api_key: str | None, echo: EchoFn) -> list[Model]:
    try:
        return fetch_openai_compatible_models(provider, base_url, api_key)
    except Exception as exc:  # HTTP errors, malformed payloads
        echo(f"Could not list models from {base_url.rstrip('/')}/models ({exc}).")
    return []


# -- /login and /logout commands ------------------------------------------------
def switch_to_provider_if_unusable(session: "AgentSession", provider: str, console: "Console") -> "Model | None":
    """After sign-in, switch to the provider's default model when the current one is unusable."""
    model = session.model
    if model is not None and (model.provider in NO_KEY_PROVIDERS or session.api_key):
        return None
    _models, note = ensure_models(provider, session.settings)
    target = default_model_for_provider(provider)
    if target is None:
        console.print(f"[yellow]Could not select a {provider} model: {note or 'no models available'}[/]")
        return None
    session.set_model(target)
    return target


def _echo(console: "Console") -> EchoFn:
    return lambda text: console.print(text, highlight=False)


def suggest_login(session: "AgentSession", console: "Console", ask: AskFn, select: SelectFn, switch: SwitchFn) -> None:
    if session.model is None:
        console.print(
            "[yellow]No model is available[/] (no credentials configured, or the provider's model "
            "catalog could not be fetched). Choose a sign-in method, or run [bold]/login <provider>[/] later."
        )
    else:
        console.print(
            "[yellow]No provider credentials configured[/] "
            "(none in auth.json, settings apiKeys or provider env vars). "
            "Choose a sign-in method, or run [bold]/login <provider>[/] later."
        )
    try:
        provider = interactive_login(ask=ask, select=select, echo=_echo(console), settings=session.settings)
    except (KeyboardInterrupt, EOFError):
        return
    if provider is None:
        console.print("[dim]Skipped — set an API key env var or use /login <provider> later.[/]")
        return
    session.reload_api_key()
    switched = switch(provider)
    if switched is not None:
        console.print(f"[green]Signed in to {provider_label(provider)} — switched to {format_model(switched)}.[/]")
    else:
        console.print(f"[green]Signed in to {provider_label(provider)}.[/]")


def maybe_ask_login(
    session: "AgentSession", console: "Console", ask: AskFn, select: SelectFn, switch: SwitchFn
) -> None:
    """Offer sign-in when no credentials are configured anywhere (or no model is available)."""
    model = session.model
    if model is not None and (model.provider in NO_KEY_PROVIDERS or session.api_key):
        return
    suggest_login(session, console, ask, select, switch)


def login(
    session: "AgentSession", arg: str, console: "Console", ask: AskFn, select: SelectFn, switch: SwitchFn
) -> None:
    provider = arg or (session.model.provider if session.model is not None else "")
    if not provider:
        suggest_login(session, console, ask, select, switch)
        return
    if provider == "github-copilot":
        login_copilot(session, console, ask, switch)
        return
    if provider in NO_KEY_PROVIDERS:
        ensure_models(provider, session.settings)
        model = default_model_for_provider(provider)
        if model is None:
            console.print(f"[red]No {provider} models available — start the endpoint or fill settings.json.[/]")
            return
        session.set_model(model)
        console.print(f"[green]{format_model(model)} needs no credentials — switched to it.[/]")
        return
    console.print(f"[dim]Storing an API key for [bold]{provider}[/]. It is saved to ~/.tynex/agent/auth.json.[/]")
    try:
        key = ask(f"{provider} API key: ", password=True).strip()
    except (KeyboardInterrupt, EOFError):
        return
    if not key:
        return
    store_api_key(provider, key)
    session.reload_api_key()
    switched = switch(provider)
    if switched is not None:
        console.print(f"[green]API key stored — switched to {format_model(switched)}.[/]")
    elif session.model is not None and provider == session.model.provider:
        console.print("[green]API key stored; using it now.[/]")
    else:
        console.print("[green]API key stored.[/]")


def login_copilot(session: "AgentSession", console: "Console", ask: AskFn, switch: SwitchFn) -> None:
    """GitHub Copilot sign-in: ask for the company's GHE URL, then run the device flow."""
    if not copilot_login(ask=ask, echo=_echo(console), settings=session.settings):
        return
    session.reload_api_key()
    switched = switch("github-copilot")
    if switched is not None:
        console.print(f"[green]GitHub Copilot is ready — switched to {format_model(switched)}.[/]")
    else:
        console.print("[green]GitHub Copilot is ready. Use /model to pick one of its models.[/]")


def logout(session: "AgentSession", arg: str, console: "Console", select: SelectFn) -> None:
    stored = signed_in_providers()
    if not stored:
        console.print(
            "[yellow]No stored credentials to remove.[/] /logout only removes keys saved to "
            "auth.json by /login; environment variables and settings apiKeys are unchanged."
        )
        return
    provider = arg.strip()
    if provider:
        if provider not in stored:
            console.print(f"[red]No stored credentials for {provider}.[/]")
            return
    elif len(stored) == 1 or not console.is_terminal:
        provider = stored[0]
    else:
        index = select("Sign out of which provider?", [provider_label(p) for p in stored])
        if index is None:
            console.print("[dim]Nothing to do.[/]")
            return
        provider = stored[index]
    if not remove_credential(provider):
        console.print(f"[red]No stored credentials for {provider}.[/]")
        return
    session.reload_api_key()
    if provider in OAUTH_PROVIDERS:
        console.print(f"[green]Logged out of {provider_label(provider)}.[/]")
    else:
        console.print(
            f"[green]Removed stored API key for {provider_label(provider)}.[/] "
            "[dim]Environment variables and settings apiKeys are unchanged.[/]"
        )
    if session.model is not None and session.model.provider == provider and not session.api_key:
        console.print("[yellow]The current model has no credentials now — /login or pick another model.[/]")


PROVIDER = "github-copilot"
ENV_HOST = "GITHUB_COPILOT_ENTERPRISE_URL"
ENV_CLIENT_ID = "GITHUB_COPILOT_CLIENT_ID"


@dataclass
class CopilotConfig:
    """Resolved Copilot configuration for the current settings/auth files."""

    token: str
    host: str  # normalized; "https://github.com" for the public service
    client_id: str
    login: str | None = None  # GitHub account name, when known
    models: list[dict[str, Any]] = field(default_factory=list)  # catalog cached at login


def _provider_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    provider_cfg = ((settings or {}).get("providers") or {}).get(PROVIDER)
    return provider_cfg if isinstance(provider_cfg, dict) else {}


def _configured_host(settings: dict[str, Any] | None) -> str:
    provider_cfg = _provider_settings(settings)
    return copilot_api.normalize_host(
        provider_cfg.get("enterpriseUrl") or provider_cfg.get("host") or os.environ.get(ENV_HOST)
    )


def copilot_config(settings: dict[str, Any] | None = None) -> CopilotConfig | None:
    """Resolve Copilot credentials/config; ``None`` when not authenticated."""
    settings = settings or {}
    provider_cfg = _provider_settings(settings)
    auth_entry = (load_auth() or {}).get(PROVIDER)
    if not isinstance(auth_entry, dict):
        auth_entry = {}
    token = resolve_api_key(PROVIDER, settings)
    if not token:
        return None
    host = copilot_api.normalize_host(
        provider_cfg.get("enterpriseUrl")
        or provider_cfg.get("host")
        or auth_entry.get("host")
        or os.environ.get(ENV_HOST)
    )
    client_id = str(
        provider_cfg.get("clientId")
        or auth_entry.get("clientId")
        or os.environ.get(ENV_CLIENT_ID)
        or copilot_api.DEFAULT_CLIENT_ID
    )
    login_name = auth_entry.get("login")
    cached = auth_entry.get("models")
    return CopilotConfig(
        token=str(token),
        host=host,
        client_id=client_id,
        login=login_name if isinstance(login_name, str) else None,
        models=[m for m in cached if isinstance(m, dict)] if isinstance(cached, list) else [],
    )


def copilot_model_specs_from_api(payload: Any) -> list[dict[str, Any]]:
    """Normalize a ``GET /models`` response into cached model specs."""
    entries = payload.get("data") if isinstance(payload, dict) else None
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id") or "").strip()
        if not model_id or model_id in seen or entry.get("model_picker_enabled") is False:
            continue
        capabilities = entry.get("capabilities") if isinstance(entry.get("capabilities"), dict) else {}
        limits = capabilities.get("limits") if isinstance(capabilities.get("limits"), dict) else {}
        supports = capabilities.get("supports") if isinstance(capabilities.get("supports"), dict) else {}
        endpoints = entry.get("supported_endpoints") or []
        if endpoints and "/chat/completions" not in endpoints:
            continue
        if supports.get("tool_calls") is False:
            continue
        spec: dict[str, Any] = {"id": model_id, "name": str(entry.get("name") or model_id)}
        context = limits.get("max_context_window_tokens") or limits.get("max_prompt_tokens")
        output = limits.get("max_output_tokens")
        if isinstance(context, int) and context > 0:
            spec["contextWindow"] = context
        if isinstance(output, int) and output > 0:
            spec["maxTokens"] = output
        if supports.get("reasoning_effort"):
            spec["reasoning"] = True
        spec["input"] = ["text", "image"] if supports.get("vision") else ["text"]
        seen.add(model_id)
        specs.append(spec)
    return specs


def _model_from_spec(spec: dict[str, Any], host: str | None) -> Model:
    return model_registry.model_from_spec(PROVIDER, spec, api=PROVIDER, base_url=host)


def register_copilot_provider(settings: dict[str, Any] | None = None) -> CopilotConfig | None:
    """(Re-)register the ``github-copilot`` model catalog. Idempotent."""
    cfg = copilot_config(settings)
    if cfg is not None and not copilot_api.is_public_host(cfg.host):
        host: str | None = cfg.host
    else:
        configured = _configured_host(settings)
        host = configured if not copilot_api.is_public_host(configured) else None
    specs = [spec for spec in (cfg.models if cfg is not None else []) if isinstance(spec, dict) and spec.get("id")]
    model_registry.register_models(PROVIDER, [_model_from_spec(spec, host) for spec in specs])
    return cfg


def _host_label(host: str | None) -> str:
    return "github.com" if copilot_api.is_public_host(host) else copilot_api.normalize_host(host)


def copilot_login(
    *,
    ask: Callable[[str], str],
    echo: Callable[[str], None],
    settings: dict[str, Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Interactive device-flow login; returns ``True`` when credentials were stored."""
    settings = settings or {}
    existing = copilot_config(settings)
    default_host = existing.host if existing is not None else _configured_host(settings)
    hint = f" [{_host_label(default_host)}]" if not copilot_api.is_public_host(default_host) else ""
    if existing is not None and existing.login:
        echo(f"Already signed in as @{existing.login} ({_host_label(existing.host)}); the device flow replaces it.")
    try:
        entered = ask(f"GitHub Enterprise URL (empty for github.com){hint}: ").strip()
    except (KeyboardInterrupt, EOFError):
        echo("Login cancelled.")
        return False
    host = copilot_api.normalize_host(entered or default_host)
    client_id = (
        existing.client_id
        if existing is not None
        else str(os.environ.get(ENV_CLIENT_ID) or copilot_api.DEFAULT_CLIENT_ID)
    )
    try:
        device = copilot_api.request_device_code(host, client_id)
    except copilot_api.ProviderError as exc:
        echo(str(exc))
        return False
    echo(f"Open {device.verification_uri} in a browser and enter the code: {device.user_code}")
    last_notice = [0.0]

    def on_wait(elapsed: float) -> None:
        if elapsed - last_notice[0] >= 15:
            last_notice[0] = elapsed
            echo(f"Waiting for authorization… ({int(elapsed)}s)")

    try:
        github_token = copilot_api.poll_for_github_token(host, client_id, device, sleep=sleep, on_wait=on_wait)
    except copilot_api.ProviderError as exc:
        echo(str(exc))
        return False
    except (KeyboardInterrupt, EOFError):
        echo("Login cancelled.")
        return False
    echo("Authorized. Verifying Copilot access…")
    account: dict[str, Any] = {}
    model_specs: list[dict[str, Any]] = []
    verify_error: str | None = None
    try:
        session = copilot_api.default_auth().session(host, github_token, refresh=True)
        account = copilot_api.fetch_account(github_token, host)
        model_specs = copilot_model_specs_from_api(copilot_api.fetch_models(session))
    except (copilot_api.ProviderError, HTTPError) as exc:
        verify_error = str(exc)
    if not account and not model_specs:
        if verify_error:
            echo(f"Could not verify Copilot access: {verify_error}")
        try:
            answer = ask("Store the credential anyway? (y/N): ")
        except (KeyboardInterrupt, EOFError):
            answer = ""
        if not answer.strip().lower().startswith("y"):
            echo("Login cancelled; nothing was stored.")
            return False
    login_name = account.get("login") if isinstance(account.get("login"), str) else None
    entry: dict[str, Any] = {"type": "oauth", "key": github_token, "updatedAt": int(time.time())}
    if not copilot_api.is_public_host(host):
        entry["host"] = host
    if client_id != copilot_api.DEFAULT_CLIENT_ID:
        entry["clientId"] = client_id
    if login_name:
        entry["login"] = login_name
    if model_specs:
        entry["models"] = model_specs
    auth = load_auth()
    auth[PROVIDER] = entry
    save_auth(auth)
    register_copilot_provider(settings)
    who = f" as @{login_name}" if login_name else ""
    echo(
        f"Signed in{who} for {_host_label(host)}. Credentials stored in ~/.tynex/agent/auth.json — "
        "Copilot session tokens refresh automatically, the device flow runs only once."
    )
    if model_specs:
        echo(f"Discovered {len(model_specs)} models for this account (see /model or tynex models).")
    return True
