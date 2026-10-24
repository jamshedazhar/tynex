"""Tests for the shared sign-in helpers: arrow-key selector and login menu."""

import json
import os
import stat

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from tynex.ai.models import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS, Model, providers
from tynex.coding.signin import add_custom_provider, api_key_providers, interactive_login, oauth_providers
from tynex.tui.select import select_option

# ---------------------------------------------------------------------------
# Arrow-key selector
# ---------------------------------------------------------------------------


def test_select_option_arrow_keys():
    with create_pipe_input() as pipe:
        pipe.send_text("\x1b[B\r")  # down, enter
        assert select_option("pick one", ["a", "b", "c"], input=pipe, output=DummyOutput()) == 1


def test_select_option_honours_preselection():
    with create_pipe_input() as pipe:
        pipe.send_text("\r")  # accept the preselected entry
        assert select_option("pick one", ["a", "b", "c"], input=pipe, output=DummyOutput(), selected=2) == 2


def test_select_option_escape_cancels():
    with create_pipe_input() as pipe:
        pipe.send_text("\x1b")  # escape
        assert select_option("pick one", ["a", "b"], input=pipe, output=DummyOutput()) is None


# ---------------------------------------------------------------------------
# Sign-in menu
# ---------------------------------------------------------------------------


def test_provider_categories():
    assert oauth_providers() == ["github-copilot"]
    assert set(api_key_providers()) == set(providers()) - {"github-copilot", "faux", "ollama", "lmstudio"}


def test_interactive_login_menu_options(tynex_config):
    seen: dict = {}

    def select(title: str, options: list[str]) -> None:
        seen["title"] = title
        seen["options"] = options
        return None

    result = interactive_login(ask=lambda text, **kw: "", select=select, echo=lambda text: None)
    assert result is None
    assert seen["options"][0] == "OAuth — GitHub Copilot"
    assert seen["options"][1] == "API key"
    assert seen["options"][2].startswith("Add custom provider")


def test_interactive_login_cancelled(tynex_config):
    result = interactive_login(ask=lambda text, **kw: None, select=lambda title, options: None, echo=lambda text: None)
    assert result is None
    assert not (tynex_config / "auth.json").exists()


def test_interactive_login_oauth_runs_device_flow(tynex_config, monkeypatch):
    from tynex.coding import signin as signin_module

    calls: list[dict] = []
    monkeypatch.setattr(signin_module, "copilot_login", lambda **kw: calls.append(kw) or True)
    result = interactive_login(
        ask=lambda text, **kw: "",
        select=lambda title, options: 0,
        echo=lambda text: None,
        settings={"model": "anthropic/claude-sonnet-4-5"},
    )
    assert result == "github-copilot"
    assert len(calls) == 1
    assert calls[0]["settings"] == {"model": "anthropic/claude-sonnet-4-5"}


def test_interactive_login_oauth_failure_returns_none(tynex_config, monkeypatch):
    from tynex.coding import signin as signin_module

    monkeypatch.setattr(signin_module, "copilot_login", lambda **kw: False)
    result = interactive_login(ask=lambda text, **kw: "", select=lambda title, options: 0, echo=lambda text: None)
    assert result is None


def test_interactive_login_api_key(tynex_config):
    selections = iter([1, api_key_providers().index("anthropic")])
    result = interactive_login(
        ask=lambda text, **kw: "sk-unit", select=lambda title, options: next(selections), echo=lambda text: None
    )
    assert result == "anthropic"
    auth = json.loads((tynex_config / "auth.json").read_text(encoding="utf-8"))
    assert auth["anthropic"]["key"] == "sk-unit"


def test_interactive_login_api_key_empty_stays_cancelled(tynex_config):
    selections = iter([1, api_key_providers().index("anthropic")])
    result = interactive_login(
        ask=lambda text, **kw: "", select=lambda title, options: next(selections), echo=lambda text: None
    )
    assert result is None
    assert not (tynex_config / "auth.json").exists()


def test_interactive_login_custom_provider(tynex_config, monkeypatch):
    from tynex.coding import signin as signin_module

    monkeypatch.setattr(signin_module, "fetch_openai_compatible_models", _fetch_stub(("gpt-oss-120b", "gpt-oss-120b")))
    answers = iter(["my-llm", "https://llm.example.com/v1", ""])
    selections = iter([2, 0])  # "Add custom provider", then the fetched model
    result = interactive_login(
        ask=lambda text, **kw: next(answers), select=lambda title, options: next(selections), echo=lambda text: None
    )
    assert result == "my-llm"
    settings = json.loads((tynex_config / "settings.json").read_text(encoding="utf-8"))
    entry = settings["providers"]["my-llm"]
    assert entry["api"] == "openai-completions"
    assert entry["baseUrl"] == "https://llm.example.com/v1"
    assert entry["models"] == [
        {
            "id": "gpt-oss-120b",
            "name": "gpt-oss-120b",
            "contextWindow": DEFAULT_CONTEXT_WINDOW,
            "maxTokens": DEFAULT_MAX_TOKENS,
        }
    ]


# ---------------------------------------------------------------------------
# Custom provider wizard
# ---------------------------------------------------------------------------


def _fetch_stub(*models):
    def fetch(provider, base_url, api_key, **kwargs):
        return [Model(id=model_id, name=name, provider=provider, api="openai-completions") for model_id, name in models]

    return fetch


def test_add_custom_provider_picks_from_fetched_models(tynex_config, monkeypatch):
    from tynex.coding import signin as signin_module

    monkeypatch.setattr(
        signin_module,
        "fetch_openai_compatible_models",
        _fetch_stub(("gpt-oss-20b", "gpt-oss-20b"), ("gpt-oss-120b", "gpt-oss-120b")),
    )
    answers = iter(["My-LLM", "https://llm.example.com/v1", "sk-x"])
    echoed: list[str] = []
    provider = add_custom_provider(
        ask=lambda text, **kw: next(answers), select=lambda title, options: 1, echo=echoed.append
    )

    assert provider == "my-llm"  # normalized to lower case
    settings = json.loads((tynex_config / "settings.json").read_text(encoding="utf-8"))
    assert settings["providers"]["my-llm"]["baseUrl"] == "https://llm.example.com/v1"
    assert settings["providers"]["my-llm"]["models"][0]["id"] == "gpt-oss-120b"
    auth = json.loads((tynex_config / "auth.json").read_text(encoding="utf-8"))
    assert auth["my-llm"]["key"] == "sk-x"
    assert any("my-llm" in line for line in echoed)


def test_add_custom_provider_falls_back_to_manual_model_id(tynex_config, monkeypatch):
    from tynex.coding import signin as signin_module

    def failing_fetch(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(signin_module, "fetch_openai_compatible_models", failing_fetch)
    answers = iter(["local-llm", "http://localhost:8000/v1", "", "llama-3"])
    echoed: list[str] = []
    provider = add_custom_provider(
        ask=lambda text, **kw: next(answers), select=lambda title, options: 0, echo=echoed.append
    )

    assert provider == "local-llm"
    settings = json.loads((tynex_config / "settings.json").read_text(encoding="utf-8"))
    assert settings["providers"]["local-llm"]["models"][0]["id"] == "llama-3"
    assert not (tynex_config / "auth.json").exists()
    assert any("Could not list models" in line for line in echoed)


def test_add_custom_provider_cancel_from_model_picker(tynex_config, monkeypatch):
    from tynex.coding import signin as signin_module

    monkeypatch.setattr(signin_module, "fetch_openai_compatible_models", _fetch_stub(("gpt-oss-120b", "gpt-oss-120b")))
    answers = iter(["my-llm", "https://llm.example.com/v1", "sk-x"])
    provider = add_custom_provider(
        ask=lambda text, **kw: next(answers), select=lambda title, options: None, echo=lambda text: None
    )
    assert provider is None
    assert not (tynex_config / "settings.json").exists()
    assert not (tynex_config / "auth.json").exists()


def test_add_custom_provider_validates_input(tynex_config, monkeypatch):
    from tynex.coding import signin as signin_module

    def failing_fetch(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(signin_module, "fetch_openai_compatible_models", failing_fetch)
    echoed: list[str] = []

    # Invalid name, then cancel.
    answers = iter(["Bad Name", ""])
    assert (
        add_custom_provider(ask=lambda text, **kw: next(answers), select=lambda title, options: 0, echo=echoed.append)
        is None
    )
    assert any("Provider names must start" in line for line in echoed)

    # Bad base URL, then cancel.
    echoed.clear()
    answers = iter(["my-llm", "ftp://nope", ""])
    assert (
        add_custom_provider(ask=lambda text, **kw: next(answers), select=lambda title, options: 0, echo=echoed.append)
        is None
    )
    assert any("must start with http" in line for line in echoed)

    # No manual model id cancels when the endpoint cannot be listed.
    answers = iter(["my-llm", "https://llm.example.com/v1", "", ""])
    assert (
        add_custom_provider(
            ask=lambda text, **kw: next(answers), select=lambda title, options: 0, echo=lambda text: None
        )
        is None
    )


@pytest.mark.skipif(os.name == "nt", reason="Windows does not report POSIX mode bits")
def test_auth_file_is_owner_only(tmp_path, monkeypatch):
    """Credentials on disk must not be readable by other local users."""
    from tynex.coding.settings import auth_path, store_api_key

    monkeypatch.setenv("TYNEX_CONFIG_DIR", str(tmp_path))
    store_api_key("anthropic", "sk-test")
    mode = stat.S_IMODE(os.stat(auth_path()).st_mode)
    assert mode == 0o600
