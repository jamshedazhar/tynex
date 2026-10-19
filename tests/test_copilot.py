"""Offline tests for GitHub Copilot support (no network access needed).

Everything runs against ``httpx2.MockTransport``: the device flow, the token
exchange and the chat streaming are all exercised end-to-end against a fake
GitHub/Copilot API.
"""

import json
import time
from typing import Any

import httpx2
import pytest

from tynex import http
from tynex.ai import models as model_registry
from tynex.ai.models import Model
from tynex.ai.providers import copilot as copilot
from tynex.ai.providers.base import LlmContext, StreamOptions
from tynex.ai.types import UserMessage
from tynex.coding import signin as copilot_coding
from tynex.coding.agent_session import AgentSession
from tynex.coding.settings import load_auth, save_auth


@pytest.fixture(autouse=True)
def clean_copilot_state():
    copilot.default_auth().clear()
    model_registry.reset_dynamic_models()
    yield
    copilot.default_auth().clear()
    model_registry.reset_dynamic_models()


def install_transport(handler) -> None:
    """Route every HTTP request (Copilot + OpenAI-compatible streaming) through a mock transport."""
    http.set_client(httpx2.Client(transport=httpx2.MockTransport(handler)))


def sse(*events: dict[str, Any]) -> httpx2.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def copilot_model(host: str | None = None) -> Model:
    return Model(
        id="claude-sonnet-5.5",
        name="Claude Sonnet 5.5",
        provider="github-copilot",
        api="github-copilot",
        context_window=200_000,
        max_tokens=8_192,
        reasoning=True,
        input=("text", "image"),
        base_url=host,
    )


def session_token_response(token: str = "tid-1", api: str = "https://copilot-api.acme.ghe.com") -> httpx2.Response:
    return httpx2.Response(200, json={"token": token, "expires_at": time.time() + 3600, "endpoints": {"api": api}})


# ---------------------------------------------------------------------------
# Endpoint derivation
# ---------------------------------------------------------------------------


def test_endpoint_derivation_per_host_type():
    assert copilot.normalize_host("") == "https://github.com"
    assert copilot.normalize_host("github.com/") == "https://github.com"
    assert copilot.normalize_host("acme.ghe.com") == "https://acme.ghe.com"
    assert copilot.normalize_host("github.corp.example.com/") == "https://github.corp.example.com"

    # github.com
    assert copilot.github_api_base(None) == "https://api.github.com"
    assert copilot.device_code_url("") == "https://github.com/login/device/code"
    assert copilot.session_token_url("") == "https://api.github.com/copilot_internal/v2/token"
    assert copilot.default_chat_base("") == "https://api.githubcopilot.com"

    # GitHub Enterprise Cloud with data residency
    assert copilot.github_api_base("acme.ghe.com") == "https://api.acme.ghe.com"
    assert copilot.device_code_url("acme.ghe.com") == "https://acme.ghe.com/login/device/code"
    assert copilot.access_token_url("acme.ghe.com") == "https://acme.ghe.com/login/oauth/access_token"
    assert copilot.default_chat_base("acme.ghe.com") == "https://copilot-api.acme.ghe.com"

    # GitHub Enterprise Server
    assert copilot.github_api_base("github.corp.example.com") == "https://github.corp.example.com/api/v3"
    assert (
        copilot.session_token_url("github.corp.example.com")
        == "https://github.corp.example.com/api/v3/copilot_internal/v2/token"
    )
    assert copilot.default_chat_base("github.corp.example.com") == "https://api.githubcopilot.com"


# ---------------------------------------------------------------------------
# Device flow (registration happens once)
# ---------------------------------------------------------------------------


def test_device_flow_polls_until_authorized(monkeypatch):
    device_requests: list[dict[str, Any]] = []
    polls: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/login/device/code":
            device_requests.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json={
                    "device_code": "dev-1",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://acme.ghe.com/login/device",
                    "interval": 1,
                    "expires_in": 60,
                },
            )
        if request.url.path == "/login/oauth/access_token":
            polls.append(json.loads(request.content))
            if len(polls) == 1:
                return httpx2.Response(200, json={"error": "authorization_pending"})
            if len(polls) == 2:
                return httpx2.Response(200, json={"error": "slow_down"})
            return httpx2.Response(200, json={"access_token": "gho_test"})
        raise AssertionError(f"unexpected request: {request.url}")

    install_transport(handler)
    codes: list[copilot.DeviceCode] = []
    waits: list[float] = []
    slept: list[float] = []
    token = copilot.device_flow(
        "acme.ghe.com", "client-1", sleep=slept.append, on_code=codes.append, on_wait=waits.append
    )

    assert token == "gho_test"
    assert device_requests == [{"client_id": "client-1", "scope": "read:user"}]
    assert codes and codes[0].user_code == "ABCD-1234"
    assert codes[0].verification_uri == "https://acme.ghe.com/login/device"
    assert [p["grant_type"] for p in polls] == ["urn:ietf:params:oauth:grant-type:device_code"] * 3
    assert [p["device_code"] for p in polls] == ["dev-1"] * 3
    # authorization_pending -> wait `interval`; slow_down -> +5s.
    assert slept == [1.0, 6.0]
    assert len(waits) == 2


def test_device_flow_reports_denied_authorization(monkeypatch):
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/login/device/code":
            return httpx2.Response(
                200, json={"device_code": "dev-1", "user_code": "X", "interval": 1, "expires_in": 60}
            )
        return httpx2.Response(200, json={"error": "access_denied"})

    install_transport(handler)
    with pytest.raises(copilot.ProviderError, match="denied"):
        copilot.device_flow("acme.ghe.com", "client-1", sleep=lambda _seconds: None)


# ---------------------------------------------------------------------------
# Session tokens: cache once, refresh automatically
# ---------------------------------------------------------------------------


def test_session_token_is_cached_and_refreshes_automatically(monkeypatch):
    exchanges: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/copilot_internal/v2/token"
        exchanges.append(request)
        return session_token_response(token=f"tid-{len(exchanges)}")

    install_transport(handler)
    auth = copilot.CopilotAuth()

    first = auth.session("acme.ghe.com", "gho_x")
    second = auth.session("acme.ghe.com", "gho_x")
    assert first is second
    assert len(exchanges) == 1  # cached: no repeat device/token round trip
    assert first.token == "tid-1"
    assert not first.direct
    assert first.api_base == "https://copilot-api.acme.ghe.com"
    assert exchanges[0].headers["authorization"] == "token gho_x"

    # The session token expires -> the next request mints a new one with no
    # user interaction whatsoever.
    first.expires_at = time.time() - 1
    third = auth.session("acme.ghe.com", "gho_x")
    assert len(exchanges) == 2
    assert third.token == "tid-2"


def test_session_falls_back_to_github_token_when_exchange_unavailable(monkeypatch):
    paths: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        if request.url.path == "/copilot_internal/v2/token":
            return httpx2.Response(403, json={"message": "not supported for this credential"})
        if request.url.path == "/copilot_internal/user":
            assert request.headers["x-github-api-version"] == copilot.COPILOT_API_VERSION
            return httpx2.Response(
                200, json={"login": "octocat", "endpoints": {"api": "https://copilot-api.acme.ghe.com"}}
            )
        raise AssertionError(f"unexpected request: {request.url}")

    install_transport(handler)
    session = copilot.CopilotAuth().session("acme.ghe.com", "gho_pat")

    assert session.direct
    assert session.token == "gho_pat"
    assert session.api_base == "https://copilot-api.acme.ghe.com"
    assert paths == ["/copilot_internal/v2/token", "/copilot_internal/user"]


def test_api_base_env_override_skips_network_exchange(monkeypatch):
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no network calls expected")

    install_transport(handler)
    monkeypatch.setenv("GITHUB_COPILOT_API_BASE", "http://localhost:9999/v1")
    session = copilot.CopilotAuth().session("acme.ghe.com", "gho_x")
    assert session.api_base == "http://localhost:9999/v1"
    assert session.expires_at is None


# ---------------------------------------------------------------------------
# Chat streaming
# ---------------------------------------------------------------------------


def test_provider_streams_chat_with_copilot_headers(monkeypatch):
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/copilot_internal/v2/token":
            return session_token_response()
        if request.url.path == "/chat/completions":
            captured["url"] = str(request.url)
            captured["headers"] = request.headers
            captured["body"] = json.loads(request.content)
            return sse(
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " from Copilot"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
            )
        raise AssertionError(f"unexpected request: {request.url}")

    install_transport(handler)
    provider = copilot.CopilotProvider()
    model = copilot_model(host="https://acme.ghe.com")
    context = LlmContext(messages=[UserMessage(content="hi")], tools=[])
    events = list(provider.stream(model, context, StreamOptions(api_key="gho_x", thinking="off")))

    final = events[-1]["partial"]
    assert final.text() == "Hello from Copilot"
    assert final.stop_reason == "stop"
    assert final.usage.input == 10
    assert final.usage.output == 4

    assert captured["url"] == "https://copilot-api.acme.ghe.com/chat/completions"
    headers = captured["headers"]
    assert headers["authorization"] == "Bearer tid-1"
    assert headers["copilot-integration-id"] == "vscode-chat"
    assert headers["editor-version"] == copilot.EDITOR_VERSION
    assert headers["editor-plugin-version"] == copilot.EDITOR_PLUGIN_VERSION
    assert headers["x-initiator"] == "user"
    body = captured["body"]
    assert body["model"] == "claude-sonnet-5.5"
    assert body["stream"] is True
    assert body["max_tokens"] == 8_192


def test_provider_normalizes_anthropic_native_stream_events(monkeypatch):
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/copilot_internal/v2/token":
            return session_token_response()
        if request.url.path == "/chat/completions":
            return sse(
                {"type": "message_start", "message": {"usage": {"input_tokens": 12, "cache_read_input_tokens": 3}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read"},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"path"'},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": ': "a.py"}'},
                },
                {"type": "content_block_stop", "index": 1},
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
                {"type": "message_stop"},
            )
        raise AssertionError(f"unexpected request: {request.url}")

    install_transport(handler)
    provider = copilot.CopilotProvider()
    model = copilot_model(host="https://acme.ghe.com")
    context = LlmContext(messages=[UserMessage(content="read a.py")], tools=[])
    events = list(provider.stream(model, context, StreamOptions(api_key="gho_x")))

    final = events[-1]["partial"]
    assert final.text() == "Hi"
    assert final.stop_reason == "toolUse"
    calls = final.tool_calls()
    assert len(calls) == 1
    assert calls[0].id == "toolu_1"
    assert calls[0].name == "read"
    assert calls[0].arguments == {"path": "a.py"}
    assert final.usage.input == 12
    assert final.usage.cache_read == 3
    assert final.usage.output == 9


def test_provider_retries_once_with_a_fresh_session_token_on_401(monkeypatch):
    exchanges: list[int] = []
    chat_auth: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/copilot_internal/v2/token":
            exchanges.append(1)
            return session_token_response(token=f"tid-{len(exchanges)}", api="https://api.githubcopilot.com")
        if request.url.path == "/chat/completions":
            chat_auth.append(request.headers["authorization"])
            if len(chat_auth) == 1:
                return httpx2.Response(401, json={"error": {"message": "token expired"}})
            return sse(
                {"choices": [{"delta": {"content": "recovered"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            )
        raise AssertionError(f"unexpected request: {request.url}")

    install_transport(handler)
    provider = copilot.CopilotProvider()
    events = list(
        provider.stream(
            copilot_model(), LlmContext(messages=[UserMessage(content="hi")]), StreamOptions(api_key="gho_x")
        )
    )

    assert events[-1]["partial"].text() == "recovered"
    assert chat_auth == ["Bearer tid-1", "Bearer tid-2"]
    assert len(exchanges) == 2  # the second request used a freshly minted token


# ---------------------------------------------------------------------------
# Login, persistence and model registration
# ---------------------------------------------------------------------------


def copilot_handler(models_payload: dict[str, Any] | None = None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if path == "/login/device/code":
            return httpx2.Response(
                200,
                json={
                    "device_code": "dev-1",
                    "user_code": "WXYZ-9876",
                    "verification_uri": "https://acme.ghe.com/login/device",
                    "interval": 1,
                    "expires_in": 60,
                },
            )
        if path == "/login/oauth/access_token":
            return httpx2.Response(200, json={"access_token": "gho_test"})
        if path == "/copilot_internal/v2/token":
            return session_token_response()
        if path == "/copilot_internal/user":
            return httpx2.Response(
                200, json={"login": "octocat", "endpoints": {"api": "https://copilot-api.acme.ghe.com"}}
            )
        if path == "/models":
            return httpx2.Response(200, json=models_payload if models_payload is not None else {"data": []})
        raise AssertionError(f"unexpected request: {request.url}")

    return handler


def test_login_stores_token_host_login_and_cached_models(monkeypatch):
    payload = {
        "data": [
            {
                "id": "gpt-5.4",
                "name": "GPT-5.4",
                "model_picker_enabled": True,
                "supported_endpoints": ["/chat/completions"],
                "capabilities": {
                    "limits": {"max_context_window_tokens": 272_000, "max_output_tokens": 128_000},
                    "supports": {"reasoning_effort": True, "tool_calls": True, "vision": True},
                },
            },
            {
                # Not served on chat completions -> must be filtered out.
                "id": "gpt-5.3-codex",
                "name": "GPT-5.3 Codex",
                "supported_endpoints": ["/responses"],
            },
            {"id": "text-embedding-3-small", "name": "Embed", "model_picker_enabled": False},
        ]
    }
    install_transport(copilot_handler(payload))

    prompts: list[str] = []
    echoed: list[str] = []

    def ask(text: str) -> str:
        prompts.append(text)
        return "acme.ghe.com"

    ok = copilot_coding.copilot_login(ask=ask, echo=echoed.append, sleep=lambda _seconds: None)
    assert ok is True

    # The company's GHE URL is asked for (empty means github.com).
    assert any("GitHub Enterprise URL" in prompt for prompt in prompts)
    assert any("WXYZ-9876" in line for line in echoed)

    auth = load_auth()["github-copilot"]
    assert auth["type"] == "oauth"
    assert auth["key"] == "gho_test"
    assert auth["host"] == "https://acme.ghe.com"
    assert auth["login"] == "octocat"
    assert [spec["id"] for spec in auth["models"]] == ["gpt-5.4"]

    # The fetched catalog is registered with the enterprise host attached.
    registered = model_registry.models_for_provider("github-copilot")
    assert [m.id for m in registered] == ["gpt-5.4"]
    model = model_registry.resolve_model("github-copilot/gpt-5.4")
    assert model.base_url == "https://acme.ghe.com"
    assert model.context_window == 272_000
    assert model.max_tokens == 128_000
    assert model.reasoning is True


def test_register_uses_cached_catalog_with_configured_host():
    save_auth(
        {
            "github-copilot": {
                "type": "oauth",
                "key": "gho_x",
                "host": "https://acme.ghe.com",
                "models": [
                    {"id": "acme-chat-model", "name": "Acme Chat", "contextWindow": 272_000, "maxTokens": 128_000}
                ],
            }
        }
    )
    cfg = copilot_coding.register_copilot_provider({})
    assert cfg is not None
    assert cfg.host == "https://acme.ghe.com"
    models = model_registry.models_for_provider("github-copilot")
    assert [m.id for m in models] == ["acme-chat-model"]
    assert all(m.base_url == "https://acme.ghe.com" for m in models)
    resolved = model_registry.resolve_model("github-copilot/acme-chat-model")
    assert resolved.base_url == "https://acme.ghe.com"
    assert resolved.api == "github-copilot"


def test_config_resolution_precedence(monkeypatch):
    save_auth(
        {
            "github-copilot": {
                "type": "oauth",
                "key": "gho_auth",
                "host": "https://auth.ghe.com",
                "clientId": "client-auth",
            }
        }
    )
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho_env")

    # auth.json wins over the environment.
    cfg = copilot_coding.copilot_config({})
    assert cfg is not None
    assert cfg.token == "gho_auth"
    assert cfg.host == "https://auth.ghe.com"
    assert cfg.client_id == "client-auth"

    # settings.json wins over auth.json.
    settings = {
        "apiKeys": {"github-copilot": "gho_settings"},
        "providers": {"github-copilot": {"enterpriseUrl": "acme.ghe.com", "clientId": "client-settings"}},
    }
    cfg = copilot_coding.copilot_config(settings)
    assert cfg is not None
    assert cfg.token == "gho_settings"
    assert cfg.host == "https://acme.ghe.com"
    assert cfg.client_id == "client-settings"

    # The environment host applies as a fallback.
    save_auth({"github-copilot": {"type": "oauth", "key": "gho_auth"}})
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_URL", "github.corp.example.com")
    cfg = copilot_coding.copilot_config({})
    assert cfg is not None
    assert cfg.host == "https://github.corp.example.com"

    # Without any credential there is no Copilot config at all.
    save_auth({})
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    assert copilot_coding.copilot_config({}) is None


def test_model_specs_parser_filters_unsupported_entries():
    specs = copilot_coding.copilot_model_specs_from_api(
        {
            "data": [
                {
                    "id": "gpt-5.4",
                    "name": "GPT-5.4",
                    "capabilities": {
                        "limits": {"max_context_window_tokens": 272_000, "max_output_tokens": 128_000},
                        "supports": {"reasoning_effort": True, "vision": True},
                    },
                },
                {"id": "no-tools", "name": "No tools", "capabilities": {"supports": {"tool_calls": False}}},
                {"name": "missing id"},
                "not-a-dict",
            ]
        }
    )
    assert [spec["id"] for spec in specs] == ["gpt-5.4"]
    assert specs[0]["contextWindow"] == 272_000
    assert specs[0]["maxTokens"] == 128_000
    assert specs[0]["reasoning"] is True
    assert specs[0]["input"] == ["text", "image"]


# ---------------------------------------------------------------------------
# End-to-end agent session
# ---------------------------------------------------------------------------


def test_agent_session_prompts_with_copilot_model(tmp_path, monkeypatch):
    save_auth(
        {
            "github-copilot": {
                "type": "oauth",
                "key": "gho_x",
                "host": "https://acme.ghe.com",
                "models": [{"id": "gpt-5.4", "name": "GPT-5.4", "contextWindow": 272_000, "maxTokens": 128_000}],
            }
        }
    )
    calls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request.url.path)
        if request.url.path == "/copilot_internal/v2/token":
            return session_token_response()
        if request.url.path == "/chat/completions":
            return sse(
                {"choices": [{"delta": {"content": "copilot says hi"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            )
        raise AssertionError(f"unexpected request: {request.url}")

    install_transport(handler)
    # The CLI registers the cached catalog before resolving --model; mirror that here.
    copilot_coding.register_copilot_provider({})
    model = model_registry.resolve_model("github-copilot/gpt-5.4")

    session = AgentSession(cwd=str(tmp_path), model=model, tool_names=["read"], no_session=True)
    assert session.api_key == "gho_x"  # picked up from auth.json
    session.prompt("hi")

    assert session.context.messages[-1].text() == "copilot says hi"
    assert calls == ["/copilot_internal/v2/token", "/chat/completions"]
