"""Settings and credentials."""

import contextlib
import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(cwd: str) -> dict[str, Any]:
    """Tynex's shipped defaults overridden by global settings, then by project settings."""
    user = _deep_merge(_read_json(settings_path()), _read_json(project_dir(cwd) / "settings.json"))
    return _deep_merge(default_settings(), user)


def load_auth() -> dict[str, Any]:
    return _read_json(auth_path())


def save_auth(auth: dict[str, Any]) -> None:
    path = auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(auth, fh, indent=2)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def store_api_key(provider: str, api_key: str) -> None:
    auth = load_auth()
    auth[provider] = {"type": "api_key", "key": api_key}
    save_auth(auth)


def remove_credential(provider: str) -> bool:
    """Remove one provider's entry from auth.json; returns whether it existed."""
    auth = load_auth()
    if provider not in auth:
        return False
    del auth[provider]
    save_auth(auth)
    return True


def store_custom_provider(provider: str, config: dict[str, Any]) -> None:
    """Merge an OpenAI-compatible provider definition into the global settings.json."""
    path = settings_path()
    data = _read_json(path)
    providers_section = data.get("providers")
    if not isinstance(providers_section, dict):
        providers_section = {}
        data["providers"] = providers_section
    providers_section[provider] = config
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def resolve_api_key(provider: str, settings: dict[str, Any] | None = None) -> str | None:
    """Resolution order: settings apiKeys > auth.json > provider env vars."""
    settings = settings or {}
    settings_keys = settings.get("apiKeys") or {}
    if isinstance(settings_keys.get(provider), str) and settings_keys[provider]:
        return settings_keys[provider]
    entry = (load_auth() or {}).get(provider)
    if isinstance(entry, str) and entry:
        return entry
    if isinstance(entry, dict) and entry.get("key"):
        return str(entry["key"])
    from tynex.ai.models import provider_api_key_env

    for env_name in provider_api_key_env(provider):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def detect_configured_provider(settings: dict[str, Any] | None = None) -> str | None:
    """Which provider has credentials configured."""
    settings = settings or {}
    entries = load_auth()
    newest: tuple[float, int, str] | None = None
    for index, (provider, entry) in enumerate(entries.items()):
        if isinstance(entry, str):
            key, updated_at = entry, 0.0
        elif isinstance(entry, dict):
            key = entry.get("key")
            try:
                updated_at = float(entry.get("updatedAt") or 0)
            except (TypeError, ValueError):
                updated_at = 0.0
        else:
            continue
        if not key:
            continue
        # Later file positions win ties (most recently added entry).
        candidate = (updated_at, index, provider)
        if newest is None or candidate > newest:
            newest = candidate
    if newest is not None:
        return newest[2]
    for provider, key in (settings.get("apiKeys") or {}).items():
        if isinstance(key, str) and key:
            return provider
    from tynex.ai.models import provider_api_key_env, providers

    for provider in providers():
        if any(os.environ.get(env_name) for env_name in provider_api_key_env(provider)):
            return provider
    return None


def detect_default_model(settings: dict[str, Any] | None = None):
    """The catalog model matching the configured credentials, or ``None``."""
    provider = detect_configured_provider(settings)
    if provider is None:
        return None
    from tynex.ai.models import default_model_for_provider

    return default_model_for_provider(provider)


ENV_CONFIG_DIR = "TYNEX_CONFIG_DIR"

AGENTS_FILENAME = "AGENTS.md"
SYSTEM_FILENAME = "SYSTEM.md"


# -- directories ---------------------------------------------------------------


def config_dir() -> Path:
    """Global config dir (~/.tynex/agent); created on first use."""
    override = os.environ.get(ENV_CONFIG_DIR)
    base = Path(override).expanduser() if override else Path.home() / ".tynex" / "agent"
    base.mkdir(parents=True, exist_ok=True)
    return base


def project_dir(cwd: str | Path) -> Path:
    """Per-project config dir (<cwd>/.tynex)."""
    return Path(cwd) / ".tynex"


def sessions_dir() -> Path:
    """Transcript dir (config_dir()/sessions); created on first use."""
    path = config_dir() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


# -- well-known files ----------------------------------------------------------


def builtin_dir() -> Path:
    """Resources shipped with tynex (skills, prompts, default settings)."""
    return Path(__file__).resolve().parent.parent / "agent"


def settings_path() -> Path:
    """Global settings.json (merged under the project settings.json)."""
    return config_dir() / "settings.json"


def builtin_settings_path() -> Path:
    """Default settings.json shipped with tynex (its ``mcpServers`` are the built-ins)."""
    return builtin_dir() / "settings.json"


def default_settings() -> dict[str, Any]:
    """Tynex's shipped default settings; the base layer of every settings merge."""
    return _read_json(builtin_settings_path())


def read_settings_file(path: Path) -> dict[str, Any]:
    """Read one settings file; a missing or malformed file reads as ``{}``."""
    return _read_json(path)


def write_settings_file(path: Path, data: dict[str, Any]) -> None:
    """Write one settings file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def remember_setting(key: str, value: Any) -> None:
    """Persist one key in the global settings.json (a choice that should outlive the session)."""
    path = settings_path()
    data = read_settings_file(path)
    data[key] = value
    write_settings_file(path, data)


def auth_path() -> Path:
    """auth.json with API keys and provider registrations."""
    return config_dir() / "auth.json"


def models_cache_path() -> Path:
    """Cached provider model catalogs (models.json)."""
    return config_dir() / "models.json"


def input_history_path() -> Path:
    """Interactive prompt history (input_history.txt)."""
    return config_dir() / "input_history.txt"


# -- discovery -----------------------------------------------------------------


def find_files(path_glob: str, cwd: str | Path | None = None, *, one: bool = False) -> list[Path] | Path | None:
    """Discover files matching ``path_glob`` under Tynex's standard directories."""
    base = Path(cwd) if cwd is not None else Path.cwd()
    search = (project_dir(base), base, builtin_dir())
    # dict.fromkeys dedupes while keeping search order (cwd == builtin must not double a file)
    matches = list(dict.fromkeys(path for directory in search for path in sorted(directory.glob(path_glob))))
    if not one:
        return matches
    return matches[0] if matches else None


# -- reading -------------------------------------------------------------------


def read_user_text(path: Path) -> str | None:
    """File text with surrounding whitespace stripped; ``None`` if unreadable or empty."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return text.strip() or None
