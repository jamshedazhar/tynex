"""websearch tool — search the web (keyless, multi-provider) or read one page; engine in :mod:`tynex.coding.websearch`."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

from tynex.agent.types import AgentTool, ToolContext, ToolResult
from tynex.ai.types import AbortSignal
from tynex.coding.tools.truncate import DEFAULT_MAX_BYTES, format_size, truncate_head
from tynex.coding.websearch import extract_page, search_web, valid_url
from tynex.http import HTTPError

WIKIPEDIA_NOTE = "\n\n[source: Wikipedia — general web search providers were unavailable]"


def _fetch_page(url: str) -> str:
    try:
        return extract_page(url)[:4_000].strip()
    except Exception:
        return ""


def _format(results: list[dict[str, Any]], signal: AbortSignal | None = None) -> str:
    if signal is not None:
        signal.throw_if_aborted()
    urls = [str(item.get("href") or item.get("url") or "").strip() for item in results]
    fetchable = list(dict.fromkeys(url for url in urls if url and valid_url(url)))
    with ThreadPoolExecutor(max_workers=min(5, len(fetchable)) or 1) as pool:
        pages = dict(zip(fetchable, pool.map(_fetch_page, fetchable), strict=True))
    if signal is not None:
        signal.throw_if_aborted()
    blocks: list[str] = []
    for index, (item, url) in enumerate(zip(results, urls, strict=True), 1):
        title = " ".join(str(item.get("title") or "").split()) or "(untitled)"
        snippet = " ".join(str(item.get("body") or "").split())
        parts = (f"{index}. {title}", url, snippet, f"Page content:\n{pages[url]}" if pages.get(url) else "")
        blocks.append("\n   ".join(part for part in parts if part))
    text = "\n\n".join(blocks)
    return text + WIKIPEDIA_NOTE if any(item.get("source") == "wikipedia" for item in results) else text


class WebSearchTool(AgentTool):
    name = "websearch"
    label = "websearch"
    permission_kind = "web"
    description = (
        "Search the web for current information (`query`) or read a public http(s) page (`url`); provide "
        "exactly one. Include the library/package and version in keywords when the answer may be newer "
        "than your knowledge."
    )
    prompt_snippet = "Search the web or read a URL (current info, docs)"
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keywords, e.g. python 3.13 release date."},
            "url": {"type": "string", "description": "One public http(s) URL to read."},
        },
    }

    def execute(
        self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None, ctx: ToolContext | None = None
    ) -> ToolResult:
        query, url = (str(args.get(key, "")).strip() for key in ("query", "url"))
        if bool(query) == bool(url):
            raise ValueError("pass exactly one of 'query' or 'url'")
        if signal:
            signal.throw_if_aborted()
        if url:
            if not valid_url(url):
                raise ValueError("url must be a public http(s) URL")
            try:
                text = extract_page(url)[:40_000].strip()
            except (RuntimeError, HTTPError, ValueError) as exc:
                return ToolResult.text(f"Failed to read {url}: {' '.join(str(exc).split())[:160]}", is_error=True)
            return ToolResult.text(f"{url}\n\n{text}" if text else f"No readable content at {url}")
        try:
            results = search_web(query)
        except RuntimeError as exc:
            return ToolResult.text(f"Web search failed: {exc}", is_error=True)
        if not results:
            return ToolResult.text(f"No results for {query!r}")
        truncation = truncate_head(_format(results, signal))
        limit = f"\n\n[{format_size(DEFAULT_MAX_BYTES)} limit reached]" if truncation.truncated else ""
        return ToolResult.text(truncation.content + limit)
