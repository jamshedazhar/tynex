"""The single HTTP adapter: every network call in tynex goes through this module."""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from itertools import groupby
from typing import Any

import httpx2
from httpx2 import Client as Client  # re-exported for callers
from httpx2 import HTTPError as HTTPError
from httpx2 import Response as Response

STREAM_TIMEOUT, DEFAULT_TIMEOUT, CONNECT_TIMEOUT = 600.0, 30.0, 30.0
LIMITS = httpx2.Limits(max_connections=32, max_keepalive_connections=16, keepalive_expiry=60.0)
_TRANSPORT_MARKERS = ("ConnectError", "ReadError", "Timeout", "RemoteProtocol")
_pool: list[Client | None] = [None]


def _timeout(seconds: float) -> httpx2.Timeout:
    return httpx2.Timeout(seconds, connect=min(seconds, CONNECT_TIMEOUT))


def get_client() -> Client:
    """The process-wide pooled client, created on first use."""
    if (client := _pool[0]) is None or client.is_closed:
        client = _pool[0] = httpx2.Client(limits=LIMITS, timeout=_timeout(DEFAULT_TIMEOUT))
    return client


def set_client(value: Client | None) -> None:
    """Install the client every call uses; ``None`` restores the pooled one."""
    _pool[0] = value


def new_client(timeout: float = STREAM_TIMEOUT, *, connect: float = CONNECT_TIMEOUT) -> Client:
    """A dedicated client for long-lived sessions (e.g. MCP HTTP transports)."""
    return httpx2.Client(timeout=httpx2.Timeout(timeout, connect=connect), limits=LIMITS)


def is_transport_error(exc: BaseException) -> bool:
    """True for connection/timeout errors raised by the HTTP transport."""
    name = f"{type(exc).__module__}.{type(exc).__name__}"
    return name.startswith(("httpx2.", "httpcore2.")) and any(marker in name for marker in _TRANSPORT_MARKERS)


def _peer_address(response: Response) -> str | None:
    """The address of the socket that answered ``response``, when available."""
    stream = response.extensions.get("network_stream")
    if stream is None:
        return None
    with suppress(Exception):
        peer = stream.get_extra_info("server_addr")
        if peer:
            return str(peer[0])
    return None


def _is_proxied(client: Client, url: Any) -> bool:
    """True when httpx routes this URL through an HTTP or SOCKS proxy."""
    transport_for_url = getattr(client, "_transport_for_url", None)
    if not callable(transport_for_url):
        return False
    pool = getattr(transport_for_url(url), "_pool", None)
    return type(pool).__name__ in {"HTTPProxy", "SOCKSProxy"}


def request(
    method: str,
    url: str,
    *,
    json_body=None,
    client=None,
    timeout=DEFAULT_TIMEOUT,
    peer_guard: Callable[[str], None] | None = None,
    **options,
) -> Response:
    """Send one request on the pooled connection; extra keywords go to httpx.

    ``peer_guard`` inspects the connected peer address before the body is read.
    It requires ``follow_redirects=False``; a proxy resolves the target itself,
    so the guard is limited to direct connections.
    """
    options = {"follow_redirects": True, "json": json_body, **options}
    client = client or get_client()
    if peer_guard is None:
        return client.request(method, url, timeout=_timeout(timeout), **options)
    follow_redirects = options.pop("follow_redirects")
    if follow_redirects:
        raise ValueError("peer_guard requires follow_redirects=False")
    request = client.build_request(method, url, timeout=_timeout(timeout), **options)
    response = client.send(request, stream=True, follow_redirects=False)
    try:
        if not _is_proxied(client, request.url):
            peer = _peer_address(response)
            if peer is not None:
                peer_guard(peer)
        response.read()
        return response
    finally:
        response.close()


def get_json(url: str, **options) -> Any:
    """``GET`` ``url`` and decode the JSON response, raising on HTTP errors."""
    return request("GET", url, **options).raise_for_status().json()


@contextmanager
def stream(method: str, url: str, *, json_body=None, timeout=STREAM_TIMEOUT, **options):
    """Open an SSE/streaming response; leaving the block returns the socket to the pool."""
    options.setdefault("json", json_body)
    with get_client().stream(method, url, timeout=_timeout(timeout), **options) as response:
        yield response


def iter_sse(lines: Iterator[str]) -> Iterator[str]:
    """Yield SSE ``data:`` payloads from an iterator of raw lines."""
    for _, block in groupby((raw.rstrip("\r\n") for raw in lines), key=bool):
        if data := [line[5:].removeprefix(" ") for line in block if line.startswith("data:")]:
            yield "\n".join(data)


def iter_json_events(lines: Iterator[str]) -> Iterator[Any]:
    """Yield JSON values from an SSE stream, skipping keep-alives and bad payloads."""
    for data in iter_sse(lines):
        with suppress(json.JSONDecodeError):
            yield json.loads(data)
