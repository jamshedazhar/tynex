"""Tests for the websearch tool (providers, parsers and page reading; no network)."""

import http.server
import json
import socket
import socketserver
import threading
from contextlib import suppress

import httpx2
import pytest

import tynex.coding.tools.websearch as websearch
from tynex import http as tynex_http
from tynex.ai.providers.anthropic import _map_tool_name, _unmap_tool_name
from tynex.coding.agent_session import AgentSession
from tynex.coding.tools.websearch import WebSearchTool
from tynex.coding.websearch import extract_page, providers, readable, search_web

_DDG_HTML = """<!DOCTYPE html><html><body>
<div class="result results_links">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone&amp;rut=abc">One &quot;quoted&quot;</a>
  </h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone&amp;rut=abc">First <b>snippet</b> &amp; text</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.com/two">Two</a>
  <a class="result__snippet">Second snippet</a>
</div>
</body></html>"""

_BING_RSS = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
  <item>
    <title>Python Packaging User Guide</title>
    <link>https://packaging.python.org/</link>
    <description>The &lt;b&gt;Python&lt;/b&gt; packaging guide</description>
  </item>
  <item>
    <title>Unsafe link</title>
    <link>javascript:alert(1)</link>
    <description>skipped</description>
  </item>
</channel></rss>"""

_BING_JUNK = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
  <item>
    <title>ChatGPT: Chat, Work, Create</title>
    <link>https://chatgpt.com/</link>
    <description>AI assistant</description>
  </item>
</channel></rss>"""

_WIKIPEDIA_JSON = {
    "pages": [
        {
            "id": 23862,
            "key": "Python_(programming_language)",
            "title": "Python (programming language)",
            "description": "General-purpose programming language",
            "excerpt": 'Python is a <span class="searchmatch">high-level</span>, general-purpose language.',
        }
    ]
}

_PAGE_HTML = """<html><head><title>Doc Title</title><style>body{color:red}</style></head><body>
<nav>Home | About</nav>
<main><h1>Heading</h1><p>Main paragraph with <b>bold</b> text.</p>
<ul><li>First item</li><li>Second item</li></ul></main>
<footer>Copyright 2026</footer><script>alert(1)</script>
</body></html>"""


def response(
    status: int = 200, text: str = "", *, content: bytes | None = None, headers: dict[str, str] | None = None
) -> httpx2.Response:
    """Build a real ``httpx2.Response`` without a network round-trip."""
    request = httpx2.Request("GET", "https://test.local/")
    if content is not None:
        return httpx2.Response(status, content=content, headers=headers or {}, request=request)
    return httpx2.Response(status, text=text, headers=headers or {}, request=request)


def make_results(count: int = 5) -> list[dict[str, str]]:
    return [
        {"title": f"Result {i}", "href": f"https://example.com/{i}", "body": f"Snippet {i}"}
        for i in range(1, count + 1)
    ]


def make_tool(monkeypatch, search=None, extract=None) -> WebSearchTool:
    """Patch the module backends and return a tool (no network by default)."""
    if search is not None:
        monkeypatch.setattr(websearch, "search_web", search)
    monkeypatch.setattr(websearch, "extract_page", extract or (lambda url: ""))
    return WebSearchTool()


def patch_request(monkeypatch, handler) -> list[tuple[str, str, dict]]:
    """Route the shared HTTP adapter through ``handler`` and record every call."""
    calls: list[tuple[str, str, dict]] = []

    def fake(method: str, url: str, **kwargs):
        calls.append((method, url, kwargs))
        return handler(method, url, **kwargs)

    monkeypatch.setattr(tynex_http, "request", fake)
    return calls


@pytest.fixture(autouse=True)
def no_keyed_providers(monkeypatch):
    """Keyed providers stay off unless a test opts in."""
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def public_example_dns(monkeypatch):
    """Keep page-read tests offline: example.com resolves to a fixed public address."""
    real = socket.getaddrinfo

    def fake(host, *args, **kwargs):
        if host.lower().rstrip(".") == "example.com":
            return real("93.184.216.34", *args, **kwargs)
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake)


# ---------------------------------------------------------------------------
# Parsers (pure functions)
# ---------------------------------------------------------------------------


def test_parse_duckduckgo_html():
    assert providers._parse_duckduckgo(_DDG_HTML) == [
        {"title": 'One "quoted"', "href": "https://example.com/one", "body": "First snippet & text"},
        {"title": "Two", "href": "https://example.com/two", "body": "Second snippet"},
    ]


def test_parse_duckduckgo_skips_unusable_links():
    markup = '<a class="result__a" href="//example.com/plain">Plain</a><a class="result__snippet">body</a>'
    assert providers._parse_duckduckgo(markup) == [
        {"title": "Plain", "href": "https://example.com/plain", "body": "body"}
    ]


def test_parse_bing_rss():
    assert providers._parse_bing_rss(_BING_RSS) == [
        {
            "title": "Python Packaging User Guide",
            "href": "https://packaging.python.org/",
            "body": "The Python packaging guide",
        }
    ]


def test_parse_bing_rss_rejects_broken_xml():
    assert providers._parse_bing_rss("<not-xml") == []


def test_parse_wikipedia_json():
    assert providers._parse_wikipedia(_WIKIPEDIA_JSON) == [
        {
            "title": "Python (programming language)",
            "href": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "body": "General-purpose programming language — Python is a high-level, general-purpose language.",
            "source": "wikipedia",
        }
    ]


def test_parse_wikipedia_ignores_malformed_payloads():
    assert providers._parse_wikipedia(None) == []
    assert providers._parse_wikipedia({"pages": ["nope"]}) == []


def test_mentions_query():
    results = [{"title": "Python Packaging", "body": "docs"}]
    assert providers._mentions_query("python packaging", results)
    assert not providers._mentions_query("python packaging", [{"title": "ChatGPT", "body": "AI"}])
    assert providers._mentions_query("a b", results)  # no usable terms -> accept


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def test_duckduckgo_provider_returns_results(monkeypatch):
    calls = patch_request(monkeypatch, lambda method, url, **kwargs: response(200, text=_DDG_HTML))
    results = providers._search_duckduckgo("python packaging")
    assert results[0]["href"] == "https://example.com/one"
    assert calls[0][0] == "GET"
    assert calls[0][2]["params"] == {"q": "python packaging"}


def test_duckduckgo_provider_retries_with_post_after_challenge(monkeypatch):
    def handler(method, url, **kwargs):
        return response(202, text="challenge") if method == "GET" else response(200, text=_DDG_HTML)

    calls = patch_request(monkeypatch, handler)
    monkeypatch.setattr(providers.time, "sleep", lambda seconds: None)
    assert providers._search_duckduckgo("q")[0]["title"] == 'One "quoted"'
    assert [method for method, _url, _kwargs in calls] == ["GET", "POST"]


def test_duckduckgo_provider_raises_when_always_challenged(monkeypatch):
    patch_request(monkeypatch, lambda method, url, **kwargs: response(202, text="challenge"))
    monkeypatch.setattr(providers.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="HTTP 202"):
        providers._search_duckduckgo("q")


def test_duckduckgo_provider_returns_empty_for_no_results_page(monkeypatch):
    patch_request(monkeypatch, lambda method, url, **kwargs: response(200, text="<html>no-results</html>"))
    assert providers._search_duckduckgo("q") == []


def test_bing_provider_returns_relevant_results(monkeypatch):
    calls = patch_request(monkeypatch, lambda method, url, **kwargs: response(200, text=_BING_RSS))
    assert providers._search_bing("python packaging")[0]["href"] == "https://packaging.python.org/"
    assert calls[0][2]["params"] == {"q": "python packaging", "format": "rss"}


def test_bing_provider_discards_irrelevant_results(monkeypatch):
    patch_request(monkeypatch, lambda method, url, **kwargs: response(200, text=_BING_JUNK))
    with pytest.raises(RuntimeError, match="irrelevant results"):
        providers._search_bing("python packaging")


def test_bing_provider_raises_on_http_error(monkeypatch):
    patch_request(monkeypatch, lambda method, url, **kwargs: response(503, text=""))
    with pytest.raises(httpx2.HTTPStatusError):
        providers._search_bing("q")


def test_wikipedia_provider_labels_results(monkeypatch):
    calls = patch_request(monkeypatch, lambda method, url, **kwargs: response(200, text=json.dumps(_WIKIPEDIA_JSON)))
    results = providers._search_wikipedia("python packaging")
    assert results[0]["source"] == "wikipedia"
    assert calls[0][2]["headers"]["User-Agent"].startswith("tynex/")


def test_wikipedia_provider_discards_unrelated_pages(monkeypatch):
    payload = {"pages": [{"key": "Dog", "title": "Dog", "description": "Animal", "excerpt": "A dog."}]}
    patch_request(monkeypatch, lambda method, url, **kwargs: response(200, text=json.dumps(payload)))
    assert providers._search_wikipedia("python packaging") == []


def test_firecrawl_provider_accepts_v2_and_v1_shapes(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    item = {"title": "Docs", "url": "https://example.com/docs", "description": "page"}
    v2 = {"success": True, "data": {"web": [item]}}
    v1 = {"success": True, "data": [item]}
    calls = patch_request(monkeypatch, lambda method, url, **kwargs: response(200, text=json.dumps(v2)))
    results = providers._search_firecrawl("python packaging")
    assert results == [{"title": "Docs", "href": "https://example.com/docs", "body": "page"}]
    method, url, kwargs = calls[0]
    assert method == "POST" and url.endswith("/v2/search")
    assert kwargs["headers"]["Authorization"] == "Bearer fc-test"
    assert kwargs["json_body"] == {"query": "python packaging", "limit": providers.MAX_RESULTS}

    patch_request(monkeypatch, lambda method, url, **kwargs: response(200, text=json.dumps(v1)))
    assert providers._search_firecrawl("python packaging") == results


def test_firecrawl_provider_falls_back_to_v1_on_404(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    item = {"title": "Docs", "url": "https://example.com/docs", "description": "page"}

    def handler(method, url, **kwargs):
        if url.endswith("/v2/search"):
            return response(404, text="")
        return response(200, text=json.dumps({"success": True, "data": [item]}))

    calls = patch_request(monkeypatch, handler)
    assert providers._search_firecrawl("q")[0]["href"] == "https://example.com/docs"
    assert [url.rsplit("/", 2)[-2] for _method, url, _kwargs in calls] == ["v2", "v1"]


def test_brave_provider_sends_token(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "tok")
    payload = {"web": {"results": [{"title": "Docs", "url": "https://example.com/docs", "description": "page"}]}}
    calls = patch_request(monkeypatch, lambda method, url, **kwargs: response(200, text=json.dumps(payload)))
    results = providers._search_brave("python packaging")
    assert results == [{"title": "Docs", "href": "https://example.com/docs", "body": "page"}]
    assert calls[0][2]["headers"]["X-Subscription-Token"] == "tok"
    assert calls[0][2]["params"] == {"q": "python packaging", "count": providers.MAX_RESULTS}


# ---------------------------------------------------------------------------
# Provider chain
# ---------------------------------------------------------------------------


def test_provider_chain_defaults_to_keyless_engines():
    assert [name for name, _provider in providers._provider_chain()] == ["duckduckgo", "bing", "wikipedia"]


def test_provider_chain_puts_keyed_providers_first(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "tok")
    assert [name for name, _provider in providers._provider_chain()] == [
        "firecrawl",
        "brave",
        "duckduckgo",
        "bing",
        "wikipedia",
    ]
    monkeypatch.setenv("FIRECRAWL_API_KEY", "   ")
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY")
    assert [name for name, _provider in providers._provider_chain()] == ["duckduckgo", "bing", "wikipedia"]


def test_search_web_falls_back_after_provider_errors(monkeypatch):
    def boom(query: str) -> list[dict[str, str]]:
        raise RuntimeError("HTTP 429")

    monkeypatch.setattr(providers, "_provider_chain", lambda: [("ddg", boom), ("bing", lambda query: make_results(1))])
    assert search_web("q") == make_results(1)


def test_search_web_returns_first_non_empty_answer(monkeypatch):
    def empty(query: str) -> list[dict[str, str]]:
        return []

    monkeypatch.setattr(providers, "_provider_chain", lambda: [("ddg", empty), ("bing", lambda query: make_results(1))])
    assert search_web("q") == make_results(1)


def test_search_web_all_providers_failing_raises_with_hint(monkeypatch):
    def boom(query: str) -> list[dict[str, str]]:
        raise RuntimeError("HTTP 429")

    monkeypatch.setattr(providers, "_provider_chain", lambda: [("duckduckgo", boom), ("bing", boom)])
    with pytest.raises(RuntimeError) as excinfo:
        search_web("q")
    message = str(excinfo.value)
    assert "duckduckgo: HTTP 429" in message and "bing: HTTP 429" in message
    assert "BRAVE_SEARCH_API_KEY" in message and "HTTPS_PROXY" in message


def test_search_web_clean_empty_answers_are_not_errors(monkeypatch):
    def empty(query: str) -> list[dict[str, str]]:
        return []

    def boom(query: str) -> list[dict[str, str]]:
        raise RuntimeError("HTTP 503")

    monkeypatch.setattr(providers, "_provider_chain", lambda: [("ddg", boom), ("bing", empty)])
    assert search_web("q") == []
    monkeypatch.setattr(providers, "_provider_chain", lambda: [("ddg", empty), ("bing", empty)])
    assert search_web("q") == []


def test_search_web_end_to_end_falls_back_to_wikipedia(monkeypatch):
    def handler(method, url, **kwargs):
        if "duckduckgo" in url:
            return response(202, text="challenge")
        if "bing" in url:
            return response(503, text="")
        return response(200, text=json.dumps(_WIKIPEDIA_JSON))

    patch_request(monkeypatch, handler)
    monkeypatch.setattr(providers.time, "sleep", lambda seconds: None)
    results = search_web("python packaging")
    assert [item["source"] for item in results] == ["wikipedia"]


# ---------------------------------------------------------------------------
# Page reading
# ---------------------------------------------------------------------------


def test_html_to_text_prefers_main_content():
    markup = (
        "<html><head><title>T</title></head><body>"
        "<div>" + "junk " * 100 + "</div>"
        "<main><h1>Real</h1><p>Main paragraph with <b>bold</b> text. " + "More detail. " * 20 + "</p></main>"
        "</body></html>"
    )
    text = readable._html_to_text(markup)
    assert "Main paragraph with bold text." in text
    assert "junk" not in text
    assert "# Real" in text


def test_html_to_text_falls_back_when_main_is_tiny():
    text = readable._html_to_text("<body><main>Hi</main><div>Body text here.</div></body>")
    assert "Body text here." in text


def test_html_to_text_separates_adjacent_inline_text():
    text = readable._html_to_text("<div><span>Light mode</span><span>Dark mode</span></div>")
    assert "Light mode Dark mode" in text


def test_html_to_text_drops_chrome_and_formats_lists():
    text = readable._html_to_text(_PAGE_HTML)
    assert text.startswith("# Doc Title")
    assert "Main paragraph with bold text." in text
    assert "- First item" in text
    assert "Home | About" not in text
    assert "Copyright 2026" not in text
    assert "alert(1)" not in text


def test_extract_page_returns_readable_html(monkeypatch):
    calls = patch_request(
        monkeypatch,
        lambda method, url, **kwargs: response(
            200, content=_PAGE_HTML.encode(), headers={"content-type": "text/html; charset=utf-8"}
        ),
    )
    text = extract_page("https://example.com/doc")
    assert "Main paragraph with bold text." in text
    assert calls[0][2]["timeout"] == readable.PAGE_TIMEOUT


def test_extract_page_returns_plain_text_and_json_verbatim(monkeypatch):
    patch_request(
        monkeypatch,
        lambda method, url, **kwargs: response(200, text="plain text", headers={"content-type": "text/plain"}),
    )
    assert extract_page("https://example.com/raw") == "plain text"

    payload = json.dumps({"ok": True})
    patch_request(
        monkeypatch,
        lambda method, url, **kwargs: response(200, text=payload, headers={"content-type": "application/json"}),
    )
    assert extract_page("https://example.com/api") == payload


def test_extract_page_decodes_meta_charset(monkeypatch):
    markup = (
        "<html><head><meta charset='iso-8859-1'><title>Caf\xe9</title></head>"
        "<body><main><p>" + "caf\xe9 " * 60 + "</p></main></body></html>"
    )
    patch_request(
        monkeypatch,
        lambda method, url, **kwargs: response(
            200, content=markup.encode("iso-8859-1"), headers={"content-type": "text/html"}
        ),
    )
    text = extract_page("https://example.com/cafe")
    assert text.startswith("# Café")
    assert "café" in text


def test_extract_page_rejects_binary_content(monkeypatch):
    patch_request(
        monkeypatch,
        lambda method, url, **kwargs: response(200, content=b"\x89PNG", headers={"content-type": "image/png"}),
    )
    with pytest.raises(RuntimeError, match="unsupported content type: image/png"):
        extract_page("https://example.com/image.png")


def test_extract_page_raises_on_http_errors(monkeypatch):
    patch_request(monkeypatch, lambda method, url, **kwargs: response(404, text=""))
    with pytest.raises(httpx2.HTTPStatusError):
        extract_page("https://example.com/missing")


def test_valid_url_rejects_private_hosts():
    assert not providers.valid_url("http://127.0.0.1:8000/admin")
    assert not providers.valid_url("http://10.0.0.1/")
    assert not providers.valid_url("http://169.254.169.254/latest/meta-data/")
    assert not providers.valid_url("http://224.0.0.1/")
    assert not providers.valid_url("http://[::1]/")
    assert not providers.valid_url("http://[ff02::1]/")
    assert not providers.valid_url("http://[64:ff9b::7f00:1]/")
    assert not providers.valid_url("http://localhost:8000/")
    assert not providers.valid_url("http://metadata.google.internal/computeMetadata/v1/")
    assert providers.valid_url("https://example.com/doc")
    assert providers.valid_url("http://[64:ff9b::808:808]/")


def test_ensure_public_address_rejects_non_public_peers():
    for host in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "224.0.0.1", "ff02::1"):
        with pytest.raises(ValueError, match="not public"):
            providers.ensure_public_address(host)
    providers.ensure_public_address("93.184.216.34")


def test_extract_page_rejects_private_hosts(monkeypatch):
    calls = patch_request(monkeypatch, lambda method, url, **kwargs: response(200, text="internal"))
    with pytest.raises(ValueError, match="public"):
        extract_page("http://127.0.0.1:8000/admin")
    assert calls == []


def test_extract_page_rejects_rebinding_to_loopback(monkeypatch):
    """DNS may answer public for the pre-check and loopback for the socket."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"INTERNAL-SECRET"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with suppress(BrokenPipeError):
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    real = socket.getaddrinfo
    calls = {"n": 0}

    def fake(host, *args, **kwargs):
        if host == "rebind.test":
            calls["n"] += 1
            target = "93.184.216.34" if calls["n"] == 1 else "127.0.0.1"
            return real(target, *args, **kwargs)
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    client = httpx2.Client(trust_env=False)
    tynex_http.set_client(client)
    try:
        with pytest.raises(ValueError, match="not public"):
            extract_page(f"http://rebind.test:{port}/")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        client.close()
    assert calls["n"] >= 2


def test_extract_page_rejects_private_redirects(monkeypatch):
    patch_request(
        monkeypatch,
        lambda method, url, **kwargs: response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"}),
    )
    with pytest.raises(ValueError, match="public"):
        extract_page("https://example.com/start")


def test_extract_page_follows_public_redirects(monkeypatch):
    def handler(method: str, url: str, **kwargs):
        if url == "https://example.com/start":
            return response(302, headers={"location": "/moved"})
        return response(200, text="final page", headers={"content-type": "text/plain"})

    calls = patch_request(monkeypatch, handler)
    assert extract_page("https://example.com/start") == "final page"
    assert calls[-1][1] == "https://example.com/moved"


# ---------------------------------------------------------------------------
# Tool behavior
# ---------------------------------------------------------------------------


def test_formats_numbered_results_and_passes_query(monkeypatch):
    calls: list[str] = []

    def fake(query: str) -> list[dict[str, str]]:
        calls.append(query)
        return make_results(2)

    result = make_tool(monkeypatch, fake).execute("c", {"query": "python 3.12 release notes"})
    assert calls == ["python 3.12 release notes"]
    text = result.content[0].text
    assert "1. Result 1\n   https://example.com/1\n   Snippet 1" in text
    assert "2. Result 2" in text
    assert not result.is_error


def test_empty_or_missing_input_raises(monkeypatch):
    tool = make_tool(monkeypatch)
    with pytest.raises(ValueError, match="exactly one"):
        tool.execute("c", {"query": "   "})
    with pytest.raises(ValueError, match="exactly one"):
        tool.execute("c", {})


def test_query_and_url_are_mutually_exclusive(monkeypatch):
    tool = make_tool(monkeypatch)
    with pytest.raises(ValueError, match="exactly one"):
        tool.execute("c", {"query": "x", "url": "https://example.com"})


def test_no_results_message(monkeypatch):
    result = make_tool(monkeypatch, lambda query: []).execute("c", {"query": "asdf"})
    assert not result.is_error
    assert "No results for 'asdf'" in result.content[0].text


def test_backend_error_becomes_error_result(monkeypatch):
    def fake(query: str) -> list[dict[str, str]]:
        raise RuntimeError("all providers failed (duckduckgo: HTTP 429)")

    result = make_tool(monkeypatch, fake).execute("c", {"query": "x"})
    assert result.is_error
    assert "Web search failed: all providers failed (duckduckgo: HTTP 429)" in result.content[0].text


def test_wikipedia_results_are_labelled(monkeypatch):
    results = [
        {"title": "Python", "href": "https://en.wikipedia.org/wiki/Python", "body": "Language", "source": "wikipedia"}
    ]
    text = make_tool(monkeypatch, lambda query: results).execute("c", {"query": "python"}).content[0].text
    assert "[source: Wikipedia — general web search providers were unavailable]" in text


def test_long_output_is_truncated(monkeypatch):
    def fake(query: str) -> list[dict[str, str]]:
        return [{"title": f"T{i}", "href": f"https://example.com/{i}", "body": "x" * 3000} for i in range(20)]

    text = make_tool(monkeypatch, fake).execute("c", {"query": "x"}).content[0].text
    assert len(text.encode("utf-8")) < 51 * 1024
    assert "[50.0KB limit reached]" in text


# ---------------------------------------------------------------------------
# Page extraction included with search results
# ---------------------------------------------------------------------------


def test_extracts_content_for_all_results(monkeypatch):
    extracted: list[str] = []

    def fake_extract(url: str) -> str:
        extracted.append(url)
        return f"Body of {url}"

    result = make_tool(monkeypatch, lambda query: make_results(5), extract=fake_extract).execute(
        "c", {"query": "current weather in singapore"}
    )
    text = result.content[0].text

    assert sorted(extracted) == sorted(f"https://example.com/{i}" for i in range(1, 6))
    assert text.count("Page content:") == 5
    assert "Page content:\nBody of https://example.com/1" in text
    assert not result.is_error


def test_extracts_pages_concurrently(monkeypatch):
    barrier = threading.Barrier(2, timeout=2)

    def fake_extract(url: str) -> str:
        barrier.wait()
        return "page"

    result = make_tool(monkeypatch, lambda query: make_results(2), extract=fake_extract).execute("c", {"query": "x"})
    assert result.content[0].text.count("Page content:") == 2


def test_extraction_is_best_effort(monkeypatch):
    def fake_extract(url: str) -> str:
        if url.endswith("/1"):
            raise RuntimeError("403 blocked")
        return f"Body of {url}" if url.endswith("/2") else "   "

    result = make_tool(monkeypatch, lambda query: make_results(3), extract=fake_extract).execute("c", {"query": "x"})
    text = result.content[0].text
    assert not result.is_error
    assert "Snippet 1" in text and "Snippet 3" in text
    assert text.count("Page content:") == 1
    assert "Body of https://example.com/2" in text


def test_page_content_is_capped(monkeypatch):
    long_page = "x" * 4_100
    result = make_tool(monkeypatch, lambda query: make_results(1), extract=lambda url: long_page).execute(
        "c", {"query": "x"}
    )
    page = result.content[0].text.split("Page content:\n", 1)[1].strip()
    assert len(page) == 4_000


# ---------------------------------------------------------------------------
# URL mode
# ---------------------------------------------------------------------------


def test_url_mode_reads_a_page(monkeypatch):
    tool = make_tool(monkeypatch, extract=lambda url: "# Title\n\nBody text")
    text = tool.execute("c", {"url": "https://example.com/article"}).content[0].text
    assert text == "https://example.com/article\n\n# Title\n\nBody text"


def test_url_mode_rejects_invalid_urls(monkeypatch):
    tool = make_tool(monkeypatch, extract=lambda url: "content")
    with pytest.raises(ValueError, match="public http"):
        tool.execute("c", {"url": "ftp://example.com/file"})


def test_url_mode_fetch_error_becomes_error_result(monkeypatch):
    def fail(url: str) -> str:
        response(404, text="").raise_for_status()
        raise AssertionError("unreachable")

    result = make_tool(monkeypatch, extract=fail).execute("c", {"url": "https://example.com/blocked"})
    assert result.is_error
    assert "Failed to read https://example.com/blocked" in result.content[0].text
    assert "404" in result.content[0].text


def test_url_mode_reports_unsupported_content(monkeypatch):
    def fail(url: str) -> str:
        raise RuntimeError("unsupported content type: image/png")

    result = make_tool(monkeypatch, extract=fail).execute("c", {"url": "https://example.com/image.png"})
    assert result.is_error
    assert "unsupported content type: image/png" in result.content[0].text


def test_url_mode_reports_empty_content(monkeypatch):
    result = make_tool(monkeypatch, extract=lambda url: "   ").execute("c", {"url": "https://example.com/empty"})
    assert not result.is_error
    assert "No readable content at https://example.com/empty" in result.content[0].text


# ---------------------------------------------------------------------------
# Registry / prompt integration
# ---------------------------------------------------------------------------


def test_stealth_mode_maps_websearch_to_claude_code_name():
    assert _map_tool_name("websearch", True) == "WebSearch"
    assert _unmap_tool_name("WebSearch", True) == "websearch"
    assert _map_tool_name("websearch", False) == "websearch"


def test_default_session_offers_websearch(tmp_path):
    session = AgentSession(cwd=str(tmp_path), no_session=True)
    assert "websearch" in [tool.name for tool in session.tools]
    assert "- websearch: Search the web or read a URL (current info, docs)" in session.system_prompt_text()
