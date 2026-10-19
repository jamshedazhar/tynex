"""The shared HTTP adapter: one pooled client, per-request options, mock injection."""

import json

import httpx2
import pytest

from tynex import http


class RecordingClient:
    """Stands in for an httpx2 client and captures every keyword argument."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.is_closed = False

    def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        return httpx2.Response(200, text="ok", request=httpx2.Request(method, url))

    def stream(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        return _Context(httpx2.Response(200, text="ok", request=httpx2.Request(method, url)))


class _Context:
    def __init__(self, response: httpx2.Response) -> None:
        self.response = response

    def __enter__(self) -> httpx2.Response:
        return self.response

    def __exit__(self, *exc_info) -> bool:
        self.response.close()
        return False


def json_client(payload: dict) -> httpx2.Client:
    return httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=payload)))


# ---------------------------------------------------------------------------
# Pooling and injection
# ---------------------------------------------------------------------------


def test_get_client_reuses_one_pooled_client():
    assert http.get_client() is http.get_client()


def test_set_client_replaces_the_pool_and_can_be_reset():
    installed = json_client({"ok": True})
    http.set_client(installed)
    assert http.get_client() is installed
    assert http.get_json("https://example.invalid/models") == {"ok": True}
    http.set_client(None)
    assert http.get_client() is not installed


def test_get_json_raises_for_http_errors():
    http.set_client(httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(500, text="boom"))))
    with pytest.raises(httpx2.HTTPStatusError):
        http.get_json("https://example.invalid/broken")


# ---------------------------------------------------------------------------
# Request options
# ---------------------------------------------------------------------------


def test_request_passes_options_and_a_per_request_timeout():
    client = RecordingClient()
    http.set_client(client)
    response = http.request("POST", "https://example.invalid/api", headers={"x": "1"}, json_body={"q": 1}, timeout=8.0)

    assert response.status_code == 200
    method, url, kwargs = client.calls[0]
    assert (method, url) == ("POST", "https://example.invalid/api")
    assert kwargs["headers"] == {"x": "1"}
    assert kwargs["json"] == {"q": 1}
    assert kwargs["follow_redirects"] is True
    assert kwargs["timeout"].read == 8.0
    assert kwargs["timeout"].connect == 8.0


def test_stream_uses_the_long_read_timeout_with_a_short_connect():
    client = RecordingClient()
    http.set_client(client)
    with http.stream("POST", "https://example.invalid/chat", json_body={"m": 1}) as response:
        assert response.status_code == 200

    _method, _url, kwargs = client.calls[0]
    assert kwargs["json"] == {"m": 1}
    assert kwargs["timeout"].read == http.STREAM_TIMEOUT
    assert kwargs["timeout"].connect == http.CONNECT_TIMEOUT


def test_redirects_are_followed_by_default():
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(str(request.url))
        if request.url.path == "/start":
            return httpx2.Response(302, headers={"location": "https://example.invalid/moved"})
        return httpx2.Response(200, text="landed")

    http.set_client(httpx2.Client(transport=httpx2.MockTransport(handler)))
    assert http.request("GET", "https://example.invalid/start", params={"q": "x"}).text == "landed"
    assert seen == ["https://example.invalid/start?q=x", "https://example.invalid/moved"]

    assert http.request("GET", "https://example.invalid/start", follow_redirects=False).status_code == 302
    assert len(seen) == 3  # the last redirect was returned instead of followed


class _PeerStream:
    def get_extra_info(self, name: str):
        return ("93.184.216.34", 443) if name == "server_addr" else None


def test_peer_guard_checks_the_connected_address_before_reading():
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, text="ok", extensions={"network_stream": _PeerStream()})

    http.set_client(httpx2.Client(transport=httpx2.MockTransport(handler)))
    response = http.request("GET", "https://example.invalid/", follow_redirects=False, peer_guard=seen.append)
    assert response.text == "ok"
    assert seen == ["93.184.216.34"]


def test_peer_guard_failure_is_not_swallowed():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, text="secret", extensions={"network_stream": _PeerStream()})

    def reject(peer: str) -> None:
        raise ValueError(f"blocked {peer}")

    http.set_client(httpx2.Client(transport=httpx2.MockTransport(handler)))
    with pytest.raises(ValueError, match="blocked"):
        http.request("GET", "https://example.invalid/", follow_redirects=False, peer_guard=reject)


def test_peer_guard_requires_redirects_to_be_disabled():
    http.set_client(RecordingClient())
    with pytest.raises(ValueError, match="follow_redirects"):
        http.request("GET", "https://example.invalid/", peer_guard=lambda peer: None)


def test_proxied_urls_skip_the_peer_guard():
    client = httpx2.Client(proxy="http://127.0.0.1:9")
    try:
        assert http._is_proxied(client, httpx2.URL("https://example.invalid/"))
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def test_stream_yields_a_live_sse_response():
    def handler(request: httpx2.Request) -> httpx2.Response:
        events = [{"type": "delta", "text": "hi"}, {"type": "done"}]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    http.set_client(httpx2.Client(transport=httpx2.MockTransport(handler)))
    with http.stream("POST", "https://example.invalid/chat", json_body={}) as response:
        payloads = [json.loads(payload) for payload in http.iter_sse(response.iter_lines())]

    assert payloads == [{"type": "delta", "text": "hi"}, {"type": "done"}]


def test_iter_json_events_skips_keepalives_bad_payloads_and_done():
    lines = iter(["", ": keep-alive", 'data: {"a": 1}', "", "data: not-json", "", "data: [DONE]", ""])
    assert list(http.iter_json_events(lines)) == [{"a": 1}]


def test_is_transport_error_classifies_connection_failures():
    assert http.is_transport_error(httpx2.ConnectError("nope"))
    assert http.is_transport_error(httpx2.ReadTimeout("slow"))
    assert not http.is_transport_error(httpx2.InvalidURL("bad"))
    assert not http.is_transport_error(ValueError("nope"))
