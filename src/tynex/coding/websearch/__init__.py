"""Keyless web search and page reading — the engine behind the ``websearch`` tool.

Both use the shared HTTP adapter: proxy env vars and the OS trust store apply.
"""

from tynex.coding.websearch.providers import search_web, valid_url
from tynex.coding.websearch.readable import extract_page

__all__ = ["extract_page", "search_web", "valid_url"]
