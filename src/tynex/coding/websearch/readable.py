"""Page reading: fetch one URL and turn it into readable text."""

import re
from contextlib import suppress
from html.parser import HTMLParser
from urllib.parse import urljoin

from tynex import http
from tynex.coding.websearch.providers import HTML_HEADERS, ensure_public_address, ensure_public_url

PAGE_TIMEOUT = 8.0
MAX_REDIRECTS = 5

_CHARSET_RE = re.compile(rb'<meta[^>]+charset\s*=\s*["\']?\s*([A-Za-z0-9._-]+)', re.I)
_SPACES_RE = re.compile(r"[ \t\f\v]+")

_BLOCK_TAGS = (
    "p div section article main header aside figure figcaption blockquote pre h1 h2 h3 h4 h5 h6 "
    "ul ol li dl dt dd details summary table thead tbody tr td th"
)


class _HtmlText(HTMLParser):
    """Minimal readability pass: drops chrome and prefers ``<main>``/``<article>``."""

    _SKIP = frozenset({"script", "style", "noscript", "template", "svg", "iframe", "head", "nav", "footer", "form"})
    _BLOCK = frozenset(_BLOCK_TAGS.split())
    _INLINE = frozenset({"a", "span", "label", "b", "i", "em", "strong", "u", "small", "code", "time", "cite", "mark"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title, self._in_title, self._boundary = "", False, False
        self.parts: list[str] = []
        self.main_parts: list[str] = []
        self._skip_depth = self._main_depth = 0

    def _append(self, text: str) -> None:
        self.parts.append(text)
        if self._main_depth:
            self.main_parts.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
            return
        if tag in self._SKIP or self._skip_depth:
            self._skip_depth += tag in self._SKIP
            return
        self._main_depth += tag in ("main", "article")
        if prefix := {"br": "\n", "hr": "\n\n", "li": "\n- "}.get(tag):
            self._append(prefix)
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self._BLOCK:
            self._append("\n\n")
        self._boundary = tag in self._INLINE

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag in self._SKIP or self._skip_depth:
            self._skip_depth -= tag in self._SKIP and self._skip_depth > 0
            return
        self._main_depth -= tag in ("main", "article") and self._main_depth > 0
        if tag in self._BLOCK:
            self._append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            last = self.parts[-1] if self.parts else ""
            if self._boundary and data and not data[0].isspace() and last and not last[-1].isspace():
                self._append(" ")
            self._boundary = False
            self._append(data)


def _tidy(text: str) -> str:
    lines = [_SPACES_RE.sub(" ", raw).strip() for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _html_to_text(markup: str) -> str:
    parser = _HtmlText()
    with suppress(Exception):  # html.parser is lenient, but never fail a fetch on it
        parser.feed(markup)
        parser.close()
    main = _tidy("".join(parser.main_parts))
    text = main if len(main) >= 200 else _tidy("".join(parser.parts))
    if (title := _tidy(parser.title)) and not text.startswith(title):
        text = f"# {title}\n\n{text}".rstrip()
    return text


def _decode(response: http.Response) -> str:
    """Decode the body using the header charset, then a ``<meta>`` charset, then UTF-8."""
    content = response.content[:2_000_000]
    content_type = response.headers.get("content-type", "").lower()
    charset = content_type.partition("charset=")[2].split(";")[0].strip().strip("\"'")
    if not charset and (match := _CHARSET_RE.search(content[:4096])):
        charset = match.group(1).decode("ascii", "ignore")
    with suppress(LookupError, UnicodeDecodeError):
        return content.decode(charset)
    return content.decode("utf-8", "replace")


def _request_page(url: str) -> http.Response:
    """Fetch one page without following redirects blindly: every hop must stay public."""
    current = ensure_public_url(url)
    for _ in range(MAX_REDIRECTS + 1):
        response = http.request(
            "GET",
            current,
            headers=HTML_HEADERS,
            timeout=PAGE_TIMEOUT,
            follow_redirects=False,
            peer_guard=ensure_public_address,
        )
        if response.is_redirect and (location := response.headers.get("location")):
            current = ensure_public_url(urljoin(current, location))
            continue
        return response.raise_for_status()
    raise RuntimeError(f"too many redirects (more than {MAX_REDIRECTS})")


def extract_page(url: str) -> str:
    """Fetch one page: HTML becomes readable text, text/JSON/XML is returned as-is."""
    response = _request_page(url)
    kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
    kind = kind or ("text/html" if b"<html" in response.content[:2048].lower() else "text/plain")
    if kind.startswith("text/") or kind.endswith(("json", "xml")):
        text = _decode(response)
        return _html_to_text(text) if kind in ("text/html", "application/xhtml+xml") else text
    raise RuntimeError(f"unsupported content type: {kind}")
