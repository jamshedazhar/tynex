"""Search engines with fallback: keyed APIs first, then the keyless scrapers."""

import html
import ipaddress
import os
import re
import socket
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from contextlib import suppress
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from tynex import __version__, http

Result = dict[str, str]
MAX_RESULTS, SEARCH_TIMEOUT = 5, 10.0

HTML_HEADERS = {  # browser-like: the scraped HTML/RSS endpoints block plain clients
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_UA = f"tynex/{__version__} (+https://pypi.org/project/tynex/)"
_JSON_HEADERS = {"User-Agent": _UA, "Accept": "application/json"}
_HINT = (
    "set FIRECRAWL_API_KEY or BRAVE_SEARCH_API_KEY for an API-backed provider, "
    "or configure HTTP_PROXY/HTTPS_PROXY for this network"
)

_WIKI_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_RESULT_ANCHOR_RE = re.compile(r'<a\b[^>]*?\bclass="result__(a|snippet)"[^>]*>(.*?)</a>', re.S | re.I)


_LOCAL_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa", ".lan")
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def _ip_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    with suppress(ValueError):
        return ipaddress.ip_address(host.strip("[]").split("%", 1)[0])
    return None


def _is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True only for globally routable unicast addresses."""
    if isinstance(address, ipaddress.IPv6Address):
        if (mapped := address.ipv4_mapped) is not None:
            return _is_public_address(mapped)
        if address in _NAT64_PREFIX:
            return _is_public_address(ipaddress.IPv4Address(int(address) & 0xFFFFFFFF))
    return address.is_global and not address.is_multicast


def _is_local_host(host: str) -> bool:
    """True when the host literally names the local machine or a non-public IP."""
    if (address := _ip_address(host)) is not None:
        return not _is_public_address(address)
    lowered = host.lower().rstrip(".")
    return lowered == "localhost" or lowered.endswith(_LOCAL_HOST_SUFFIXES)


def _resolves_to_private(host: str, port: int | None) -> bool:
    """True when DNS maps the host to a loopback, private, link-local or multicast address."""
    try:
        infos = socket.getaddrinfo(host, port if port is not None else 443, type=socket.SOCK_STREAM)
    except OSError:
        return False  # unresolvable hosts fail at connect time; direct connects are re-checked below
    return any((address := _ip_address(info[4][0])) is not None and not _is_public_address(address) for info in infos)


def ensure_public_address(host: str) -> None:
    """Reject a connected peer that is not a globally routable unicast address."""
    address = _ip_address(host)
    if address is None:
        raise ValueError(f"peer {host} is not an IP address")
    if not _is_public_address(address):
        raise ValueError(f"peer {host} is not public")


def valid_url(url: str) -> bool:
    """True for absolute public http(s) URLs; never names the local machine or a private IP."""
    with suppress(ValueError):
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.hostname) and not _is_local_host(parsed.hostname)
    return False


def ensure_public_url(url: str) -> str:
    """Validate a URL that is about to be fetched; raises ``ValueError`` for non-public hosts."""
    if not valid_url(url):
        raise ValueError("url must be a public http(s) URL")
    parsed = urlparse(url)
    assert parsed.hostname is not None
    if _resolves_to_private(parsed.hostname, parsed.port):
        raise ValueError(f"url host {parsed.hostname} resolves to a non-public address")
    return url


def _clean(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", value)).split())


def _request(method: str, url: str, headers: dict[str, str] = HTML_HEADERS, **options) -> http.Response:
    return http.request(method, url, timeout=SEARCH_TIMEOUT, headers=headers, **options)


def _hits(rows: Iterable[Iterable[Any]]) -> list[Result]:
    """Normalize (title, link, body) rows, dropping hits without a usable link."""
    results: list[Result] = []
    for title, link, body in rows:
        if (title := _clean(str(title or ""))) and valid_url(link := str(link or "").strip()):
            results.append({"title": title, "href": link, "body": _clean(str(body or ""))})
    return results


def _mentions_query(query: str, results: list[Result]) -> bool:
    """Guard against engines that answer with unrelated (cached) results."""
    terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) >= 3}
    haystack = " ".join(f"{item.get('title', '')} {item.get('body', '')}" for item in results).lower()
    return not terms or any(term in haystack for term in terms)


def _parse_duckduckgo(markup: str) -> list[Result]:
    anchors = [(m.group(1).lower(), m.group(2), m.group(0)) for m in re.finditer(_RESULT_ANCHOR_RE, markup)]
    rows = []
    for index, (kind, inner, tag) in enumerate(anchors):
        if kind == "a":
            href = re.search(r'href="([^"]+)"', tag)
            target = html.unescape(href.group(1) if href else "").strip()
            target = f"https:{target}" if target.startswith("//") else target
            uddg = parse_qs(urlparse(target).query).get("uddg") if "uddg=" in target else None
            body = anchors[index + 1][1] if index + 1 < len(anchors) and anchors[index + 1][0] == "snippet" else ""
            rows.append((inner, unquote(uddg[0]) if uddg else target, body))
    return _hits(rows)


def _parse_bing_rss(markup: str) -> list[Result]:
    with suppress(ET.ParseError):  # broken XML means "no results", not an error
        root = ET.fromstring(markup)
        return _hits((i.findtext("title"), i.findtext("link"), i.findtext("description")) for i in root.iter("item"))
    return []


def _parse_wikipedia(payload: Any) -> list[Result]:
    results: list[Result] = []
    for page in (payload.get("pages") or []) if isinstance(payload, dict) else []:
        if not isinstance(page, dict):
            continue
        key, title = str(page.get("key") or "").strip(), _clean(str(page.get("title") or ""))
        if not key or not title:
            continue
        parts = (_clean(str(page.get("description") or "")), _clean(str(page.get("excerpt") or "")))
        body = " — ".join(part for part in parts if part)
        link = f"https://en.wikipedia.org/wiki/{quote(key, safe='/()')}"
        results.append({"title": title, "href": link, "body": body, "source": "wikipedia"})
    return results


def _json_results(items: Any) -> list[Result]:
    items = items if isinstance(items, list) else ()
    return _hits((i.get("title"), i.get("url"), i.get("description")) for i in items if isinstance(i, dict))


def _search_duckduckgo(query: str) -> list[Result]:
    """GET first, then a POST retry in case of a rate-limit or challenge page."""
    error = "no usable response"
    for attempt, options in enumerate(({"params": {"q": query}}, {"data": {"q": query}})):
        if attempt:
            time.sleep(0.5)  # let the rate limiter cool off before the retry
        response = _request("GET" if attempt == 0 else "POST", "https://html.duckduckgo.com/html/", **options)
        if response.status_code == 200 and (hits := _parse_duckduckgo(response.text)[:MAX_RESULTS]):
            return hits
        if response.status_code == 200 and ("result__a" in response.text or "no-results" in response.text):
            return []  # an honest empty answer
        error = f"HTTP {response.status_code}" if response.status_code != 200 else "challenge page"
    raise RuntimeError(error)


def _search_bing(query: str) -> list[Result]:
    response = _request("GET", "https://www.bing.com/search", params={"q": query, "format": "rss"})
    if (hits := _parse_bing_rss(response.raise_for_status().text)[:MAX_RESULTS]) and not _mentions_query(query, hits):
        raise RuntimeError("irrelevant results")
    return hits


def _search_wikipedia(query: str) -> list[Result]:
    response = _request("GET", _WIKI_URL, headers=_JSON_HEADERS, params={"q": query, "limit": MAX_RESULTS})
    hits = _parse_wikipedia(response.raise_for_status().json())[:MAX_RESULTS]
    return hits if not hits or _mentions_query(query, hits) else []  # unrelated pages are worse than nothing


def _search_firecrawl(query: str) -> list[Result]:
    auth = _JSON_HEADERS | {"Authorization": f"Bearer {os.environ['FIRECRAWL_API_KEY'].strip()}"}
    body = {"query": query, "limit": MAX_RESULTS}
    response = _request("POST", "https://api.firecrawl.dev/v2/search", headers=auth, json_body=body)
    if response.status_code == 404:  # older deployments only expose the v1 API
        response = _request("POST", "https://api.firecrawl.dev/v1/search", headers=auth, json_body=body)
    payload = response.raise_for_status().json()
    data = payload.get("data") if isinstance(payload, dict) else None
    return _json_results(data.get("web") if isinstance(data, dict) else data)[:MAX_RESULTS]


def _search_brave(query: str) -> list[Result]:
    auth = _JSON_HEADERS | {"X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"].strip()}
    response = _request("GET", _BRAVE_URL, headers=auth, params={"q": query, "count": MAX_RESULTS})
    payload = response.raise_for_status().json()
    web = payload.get("web") if isinstance(payload, dict) else None
    return _json_results(web.get("results") if isinstance(web, dict) else None)[:MAX_RESULTS]


def _provider_chain() -> list[tuple[str, Callable[[str], list[Result]]]]:
    keyed = (("firecrawl", "FIRECRAWL_API_KEY", _search_firecrawl), ("brave", "BRAVE_SEARCH_API_KEY", _search_brave))
    chain = [(name, provider) for name, key, provider in keyed if os.environ.get(key, "").strip()]
    return [*chain, ("duckduckgo", _search_duckduckgo), ("bing", _search_bing), ("wikipedia", _search_wikipedia)]


def search_web(query: str) -> list[Result]:
    """Search the web, moving to the next provider when one fails or is empty."""
    failures: list[str] = []
    for name, provider in _provider_chain():
        try:
            if results := provider(query):
                return results
            failures.append("")  # an honest empty answer: not a failure
        except Exception as exc:  # any provider problem means "try the next one"
            failures.append(f"{name}: {' '.join(str(exc).split())[:160]}")
    if failures and all(failures):
        raise RuntimeError(f"all providers failed ({'; '.join(failures)}) — {_HINT}")
    return []
