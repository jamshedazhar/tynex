"""GitHub Copilot provider: github.com, GitHub Enterprise Cloud and Server."""

import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from tynex.ai.models import Model
from tynex.ai.providers.base import Chunk, LlmContext, ProviderError, ProviderHttpError, StreamOptions
from tynex.ai.providers.openai import OpenAICompatProvider
from tynex.http import HTTPError, Response, request

DEFAULT_GITHUB_HOST = "https://github.com"
DEFAULT_CHAT_BASE = "https://api.githubcopilot.com"
PUBLIC_API_BASE = "https://api.github.com"
DEFAULT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
EDITOR_VERSION = "vscode/1.95.0"
EDITOR_PLUGIN_VERSION = "copilot-chat/0.26.7"
USER_AGENT = "GitHubCopilotChat/0.26.7"
COPILOT_API_VERSION = "2025-04-01"
INTEGRATION_ID = "vscode-chat"
OPENAI_INTENT = "conversation-panel"
SESSION_REFRESH_BUFFER = 300.0
DIRECT_MODE_TTL = 1800.0
# GitHub instances can be slow to answer; model streams keep their own timeout.
API_TIMEOUT = 60.0


# -- host and endpoint derivation ---------------------------------------------
def normalize_host(host: str | None) -> str:
    """Normalize a GitHub host (empty means github.com) to a base URL."""
    value = (host or "").strip().rstrip("/")
    if not value or value.lower() in ("github.com", "http://github.com", "https://github.com"):
        return DEFAULT_GITHUB_HOST
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    return value


def is_public_host(host: str | None) -> bool:
    return normalize_host(host) == DEFAULT_GITHUB_HOST


def _hostname(host: str | None) -> str:
    return urlparse(normalize_host(host)).hostname or ""


def github_api_base(host: str | None) -> str:
    """REST API base used for Copilot token discovery and exchange."""
    if is_public_host(host):
        return PUBLIC_API_BASE
    hostname = _hostname(host)
    if hostname.endswith(".ghe.com"):
        # GitHub Enterprise Cloud with data residency.
        return f"https://api.{hostname}"
    # GitHub Enterprise Server.
    return f"{normalize_host(host)}/api/v3"


def default_chat_base(host: str | None) -> str:
    """Chat API base used when the API does not advertise ``endpoints.api``."""
    if is_public_host(host):
        return DEFAULT_CHAT_BASE
    hostname = _hostname(host)
    if hostname.endswith(".ghe.com"):
        return f"https://copilot-api.{hostname}"
    return DEFAULT_CHAT_BASE


def device_code_url(host: str | None) -> str:
    return f"{normalize_host(host)}/login/device/code"


def access_token_url(host: str | None) -> str:
    return f"{normalize_host(host)}/login/oauth/access_token"


def session_token_url(host: str | None) -> str:
    return f"{github_api_base(host)}/copilot_internal/v2/token"


def account_url(host: str | None) -> str:
    return f"{github_api_base(host)}/copilot_internal/user"


# -- headers -------------------------------------------------------------------
def github_headers(github_token: str | None = None) -> dict[str, str]:
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "editor-version": EDITOR_VERSION,
        "editor-plugin-version": EDITOR_PLUGIN_VERSION,
        "user-agent": USER_AGENT,
    }
    if github_token:
        headers["authorization"] = f"token {github_token}"
    return headers


def chat_headers(
    session_token: str, *, initiator: str = "user", vision: bool = False, accept: str = "text/event-stream"
) -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        "accept": accept,
        "authorization": f"Bearer {session_token}",
        "copilot-integration-id": INTEGRATION_ID,
        "editor-version": EDITOR_VERSION,
        "editor-plugin-version": EDITOR_PLUGIN_VERSION,
        "user-agent": USER_AGENT,
        "openai-intent": OPENAI_INTENT,
        "x-github-api-version": COPILOT_API_VERSION,
        "x-request-id": uuid.uuid4().hex,
        "x-vscode-user-agent-library-version": "electron-fetch",
        "x-initiator": initiator,
    }
    if vision:
        headers["copilot-vision-request"] = "true"
    return headers


def _get_account(host: str | None, github_token: str) -> dict[str, Any]:
    try:
        response = request(
            "GET",
            account_url(host),
            headers={**github_headers(github_token), "x-github-api-version": COPILOT_API_VERSION},
            timeout=API_TIMEOUT,
        )
        if response.status_code >= 400:
            return {}
        data = response.json()
    except (HTTPError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# -- OAuth device flow ----------------------------------------------------------
@dataclass
class DeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    interval: float = 5.0
    expires_in: float = 900.0


def _json_or_raise(response: Response, url: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderError(f"{url} returned HTTP {response.status_code}: {response.text[:300]}") from exc
    if response.status_code >= 400:
        message = data.get("error_description") or data.get("error") or response.text[:300]
        raise ProviderError(f"{url} returned HTTP {response.status_code}: {message}")
    return data if isinstance(data, dict) else {}


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = request("POST", url, json_body=payload, headers=github_headers(), timeout=API_TIMEOUT)
    return _json_or_raise(response, url)


def request_device_code(host: str | None, client_id: str = DEFAULT_CLIENT_ID) -> DeviceCode:
    """Start the OAuth device flow against ``host``."""
    url = device_code_url(host)
    try:
        data = _post_json(url, {"client_id": client_id, "scope": "read:user"})
    except ProviderError as exc:
        raise ProviderError(
            f"{exc} — for GitHub Enterprise, set the device-flow client id via the provider "
            "config (clientId) or GITHUB_COPILOT_CLIENT_ID, or pre-supply a token with "
            "GITHUB_COPILOT_TOKEN."
        ) from exc
    if not data.get("device_code"):
        raise ProviderError(f"Device-code response from {url} was incomplete: {data}")
    return DeviceCode(
        device_code=str(data["device_code"]),
        user_code=str(data.get("user_code") or ""),
        verification_uri=str(data.get("verification_uri") or f"{normalize_host(host)}/login/device"),
        interval=float(data.get("interval") or 5),
        expires_in=float(data.get("expires_in") or 900),
    )


def poll_for_github_token(
    host: str | None,
    client_id: str,
    device: DeviceCode,
    *,
    sleep: Callable[[float], None] = time.sleep,
    on_wait: Callable[[float], None] | None = None,
) -> str:
    """Poll until the user approves the device code in the browser."""
    url = access_token_url(host)
    interval = max(1.0, float(device.interval))
    started = time.monotonic()
    payload = {
        "client_id": client_id,
        "device_code": device.device_code,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
    }
    while True:
        data = _post_json(url, payload)
        token = data.get("access_token")
        if token:
            return str(token)
        error = str(data.get("error") or "unknown_error")
        if error == "authorization_pending":
            pass
        elif error == "slow_down":
            interval += 5
        elif error == "expired_token":
            raise ProviderError("The device code expired before it was approved. Run the login again.")
        elif error == "access_denied":
            raise ProviderError("The authorization request was denied in the browser.")
        else:
            detail = data.get("error_description")
            suffix = f" ({detail})" if detail else ""
            raise ProviderError(f"Device authorization failed: {error}{suffix}")
        elapsed = time.monotonic() - started
        if elapsed + interval > device.expires_in:
            raise ProviderError("Timed out waiting for device authorization. Run the login again.")
        if on_wait:
            on_wait(elapsed)
        sleep(interval)


def device_flow(
    host: str | None,
    client_id: str = DEFAULT_CLIENT_ID,
    *,
    sleep: Callable[[float], None] = time.sleep,
    on_code: Callable[[DeviceCode], None] | None = None,
    on_wait: Callable[[float], None] | None = None,
) -> str:
    """Run the full device flow: request a code, show it, poll for the token."""
    device = request_device_code(host, client_id)
    if on_code is not None:
        on_code(device)
    return poll_for_github_token(host, client_id, device, sleep=sleep, on_wait=on_wait)


# -- session tokens -------------------------------------------------------------
@dataclass
class CopilotSession:
    """A credential usable against the Copilot chat API."""

    token: str
    api_base: str
    expires_at: float | None = None  # None means valid indefinitely
    direct: bool = False  # True: the GitHub token itself is the credential


class CopilotAuth:
    """Mints and caches Copilot session tokens, refreshing them automatically."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], CopilotSession] = {}
        self._lock = threading.Lock()

    def session(self, host: str | None, github_token: str, *, refresh: bool = False) -> CopilotSession:
        key = (normalize_host(host), github_token)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and not refresh and self._is_fresh(cached):
                return cached
            session = self._acquire(key[0], github_token)
            self._cache[key] = session
            return session

    def invalidate(self, host: str | None, github_token: str) -> None:
        with self._lock:
            self._cache.pop((normalize_host(host), github_token), None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @staticmethod
    def _is_fresh(session: CopilotSession) -> bool:
        return session.expires_at is None or time.time() < session.expires_at - SESSION_REFRESH_BUFFER

    def _acquire(self, host: str, github_token: str) -> CopilotSession:
        override = (os.environ.get("GITHUB_COPILOT_API_BASE") or "").strip().rstrip("/")
        if override:
            return CopilotSession(github_token, override, None, direct=True)
        session = self._exchange_session(host, github_token)
        if session is not None:
            return session
        return self._direct_session(host, github_token)

    @staticmethod
    def _exchange_session(host: str, github_token: str) -> CopilotSession | None:
        try:
            response = request(
                "GET", session_token_url(host), headers=github_headers(github_token), timeout=API_TIMEOUT
            )
            if response.status_code >= 400:
                return None
            data = response.json()
        except (HTTPError, ValueError):
            return None
        token = data.get("token") if isinstance(data, dict) else None
        if not token:
            return None
        endpoints = data.get("endpoints") or {}
        api_base = str(endpoints.get("api") or default_chat_base(host)).rstrip("/")
        try:
            expires_at = float(data.get("expires_at") or 0) or None
        except (TypeError, ValueError):
            expires_at = None
        return CopilotSession(str(token), api_base, expires_at, direct=False)

    @staticmethod
    def _direct_session(host: str, github_token: str) -> CopilotSession:
        api_base = _discover_chat_base(host, github_token) or default_chat_base(host)
        return CopilotSession(github_token, api_base.rstrip("/"), time.time() + DIRECT_MODE_TTL, direct=True)


def _discover_chat_base(host: str, github_token: str) -> str | None:
    endpoints = _get_account(host, github_token).get("endpoints")
    if isinstance(endpoints, dict) and endpoints.get("api"):
        return str(endpoints["api"])
    return None


_default_auth = CopilotAuth()


def default_auth() -> CopilotAuth:
    return _default_auth


def fetch_account(github_token: str, host: str | None = None) -> dict[str, Any]:
    """Best-effort account lookup (login, Copilot plan); returns ``{}`` when unavailable."""
    return _get_account(host, github_token)


def fetch_models(session: CopilotSession) -> dict[str, Any]:
    """GET the model catalog available to the signed-in account."""
    response = request(
        "GET",
        f"{session.api_base}/models",
        headers=chat_headers(session.token, accept="application/json"),
        timeout=API_TIMEOUT,
    )
    if response.status_code >= 400:
        raise ProviderError(f"{session.api_base}/models returned HTTP {response.status_code}")
    data = response.json()
    return data if isinstance(data, dict) else {}


# -- stream dialect normalization ------------------------------------------------
_ANTHROPIC_STOP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _usage_payload(state: dict[str, Any]) -> dict[str, Any]:
    cache_read = int(state["cache_read"])
    return {
        "prompt_tokens": int(state["input_tokens"]) + cache_read,
        "completion_tokens": int(state["output_tokens"]),
        "prompt_tokens_details": {"cached_tokens": cache_read},
    }


def _copilot_event_normalizer() -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
    state: dict[str, Any] = {"tool_count": 0, "block_tools": {}, "input_tokens": 0, "cache_read": 0, "output_tokens": 0}

    def normalize(event: dict[str, Any]) -> list[dict[str, Any]]:
        etype = event.get("type")
        if not isinstance(etype, str):
            return [event]
        if etype == "message_start":
            usage = ((event.get("message") or {}).get("usage")) or {}
            state["input_tokens"] = int(usage.get("input_tokens") or 0)
            state["cache_read"] = int(usage.get("cache_read_input_tokens") or 0)
            state["output_tokens"] = int(usage.get("output_tokens") or 0)
            return []
        if etype == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") != "tool_use":
                return []
            index = int(event.get("index") or 0)
            tool_index = int(state["tool_count"])
            state["tool_count"] = tool_index + 1
            state["block_tools"][index] = tool_index
            return [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tool_index,
                                        "id": block.get("id"),
                                        "function": {"name": block.get("name"), "arguments": ""},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        if etype == "content_block_delta":
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                return [{"choices": [{"delta": {"content": delta.get("text", "")}}]}]
            if dtype == "thinking_delta":
                return [{"choices": [{"delta": {"reasoning_content": delta.get("thinking", "")}}]}]
            if dtype == "input_json_delta":
                index = int(event.get("index") or 0)
                tool_index = int(state["block_tools"].get(index, 0))
                return [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": tool_index, "function": {"arguments": delta.get("partial_json", "")}}
                                    ]
                                }
                            }
                        ]
                    }
                ]
            return []
        if etype == "message_delta":
            usage = event.get("usage") or {}
            if usage.get("output_tokens") is not None:
                state["output_tokens"] = int(usage.get("output_tokens") or 0)
            choice: dict[str, Any] = {"delta": {}}
            stop_reason = (event.get("delta") or {}).get("stop_reason")
            if stop_reason:
                choice["finish_reason"] = _ANTHROPIC_STOP.get(str(stop_reason), "stop")
            return [{"choices": [choice], "usage": _usage_payload(state)}]
        if etype in ("message_stop", "content_block_stop", "ping"):
            return []
        # Unknown events (including Anthropic-style ``error`` payloads) pass
        # through so the shared decoder can surface them.
        return [event]

    return normalize


def _has_image(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url" for part in content
        ):
            return True
    return False


# -- provider ---------------------------------------------------------------------
class CopilotProvider(OpenAICompatProvider):
    """Streams Copilot chat completions; refreshes session tokens as needed."""

    name = "github-copilot"

    def _supports_reasoning_effort(self, model: Model) -> bool:
        # Copilot maps reasoning_effort for every reasoning model it serves.
        return model.reasoning

    def _chunks(self, model: Model, context: LlmContext, options: StreamOptions) -> Iterator[Chunk]:
        github_token = options.api_key or self.resolve_api_key(model, options)
        if not github_token:
            raise ProviderError(
                "No GitHub Copilot credentials. Run Tynex and use /login github-copilot "
                "(saved to ~/.tynex/agent/auth.json) or set GITHUB_COPILOT_TOKEN."
            )
        host = (
            options.base_url or model.base_url or os.environ.get("GITHUB_COPILOT_ENTERPRISE_URL") or DEFAULT_GITHUB_HOST
        )
        body = self._chat_body(model, context, options)
        messages = body.get("messages") or []
        initiator = "agent" if any(m.get("role") in ("assistant", "tool") for m in messages) else "user"
        vision = _has_image(messages)
        for attempt in (0, 1):
            session = default_auth().session(host, github_token, refresh=attempt > 0)
            headers = chat_headers(session.token, initiator=initiator, vision=vision)
            emitted = False
            try:
                for chunk in self._stream_chat(
                    model, f"{session.api_base}/chat/completions", body, headers, normalize=_copilot_event_normalizer()
                ):
                    emitted = True
                    yield chunk
                return
            except ProviderHttpError as exc:
                if exc.status in (401, 403) and not emitted and attempt == 0:
                    continue
                if exc.status in (401, 403):
                    raise ProviderError(
                        f"GitHub Copilot rejected the stored credential (HTTP {exc.status}). "
                        "Run /login github-copilot again or check that your Copilot plan includes this model."
                    ) from exc
                raise
