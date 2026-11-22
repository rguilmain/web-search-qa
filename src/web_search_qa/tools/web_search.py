"""Pluggable web search tools.

``ddgs`` is the keyless default. Tavily and Brave are optional hosted providers
selected with ``SEARCH_PROVIDER`` and the corresponding API key.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from bs4 import BeautifulSoup

from web_search_qa.config import Settings
from web_search_qa.tools.base import (
    ToolError,
    ToolResult,
    coerce_int,
    normalize_whitespace,
    require_non_empty_string,
)

_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Web search query. Be specific and include key terms.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results (1-10).",
            "minimum": 1,
            "maximum": 10,
            "default": 5,
        },
    },
    "required": ["query"],
}


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, limit: int) -> list[SearchResult]: ...


class DdgsProvider:
    """Keyless metasearch through the ``ddgs`` package."""

    name = "ddgs"

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    def search(self, query: str, limit: int) -> list[SearchResult]:
        try:
            from ddgs import DDGS

            # ddgs manages its own timeouts/backends; passing a timeout to the
            # constructor changes the backend selection and can return no results.
            raw_results = DDGS().text(query, max_results=limit)
        except Exception as exc:
            raise ToolError(
                f"Web search with ddgs failed: {type(exc).__name__}: {exc}. "
                "Try again, simplify the query, or configure Tavily/Brave."
            ) from exc

        results: list[SearchResult] = []
        for item in raw_results or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("href") or item.get("url") or item.get("link") or "").strip()
            if not url:
                continue
            title = normalize_whitespace(str(item.get("title") or item.get("heading") or url))
            snippet = normalize_whitespace(
                str(item.get("body") or item.get("snippet") or item.get("description") or "")
            )
            results.append(SearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= limit:
                break
        return results


class TavilyProvider:
    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str | None,
        *,
        timeout: float = 30.0,
        client_factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._client_factory = client_factory

    @contextmanager
    def _client(self) -> Any:
        if self._client_factory is not None:
            client = self._client_factory()
            try:
                yield client
            finally:
                client.close()
        else:
            with httpx.Client(timeout=self.timeout) as client:
                yield client

    def search(self, query: str, limit: int) -> list[SearchResult]:
        if not self.api_key:
            raise ToolError("SEARCH_PROVIDER is set to tavily, but TAVILY_API_KEY is not set.")
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": limit,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }
        try:
            with self._client() as client:
                response = client.post(self.endpoint, json=payload)
        except httpx.TimeoutException as exc:
            raise ToolError("The Tavily search request timed out.") from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"The Tavily search request failed: {exc}") from exc
        if response.status_code != 200:
            raise ToolError(f"Tavily returned HTTP {response.status_code}. Check TAVILY_API_KEY.")
        try:
            data = response.json()
        except ValueError as exc:
            raise ToolError("Tavily returned a response that was not valid JSON.") from exc
        raw_results = data.get("results") if isinstance(data, dict) else None
        results: list[SearchResult] = []
        for item in raw_results or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            results.append(
                SearchResult(
                    title=normalize_whitespace(str(item.get("title") or url)),
                    url=url,
                    snippet=normalize_whitespace(str(item.get("content") or "")),
                )
            )
            if len(results) >= limit:
                break
        return results


class BraveProvider:
    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        api_key: str | None,
        *,
        timeout: float = 30.0,
        client_factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._client_factory = client_factory

    @contextmanager
    def _client(self) -> Any:
        if self._client_factory is not None:
            client = self._client_factory()
            try:
                yield client
            finally:
                client.close()
        else:
            with httpx.Client(timeout=self.timeout) as client:
                yield client

    def search(self, query: str, limit: int) -> list[SearchResult]:
        if not self.api_key:
            raise ToolError("SEARCH_PROVIDER is set to brave, but BRAVE_API_KEY is not set.")
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        try:
            with self._client() as client:
                response = client.get(
                    self.endpoint,
                    params={"q": query, "count": limit},
                    headers=headers,
                )
        except httpx.TimeoutException as exc:
            raise ToolError("The Brave search request timed out.") from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"The Brave search request failed: {exc}") from exc
        if response.status_code != 200:
            raise ToolError(f"Brave returned HTTP {response.status_code}. Check BRAVE_API_KEY.")
        try:
            data = response.json()
        except ValueError as exc:
            raise ToolError("Brave returned a response that was not valid JSON.") from exc
        web = data.get("web") if isinstance(data, dict) else None
        raw_results = web.get("results") if isinstance(web, dict) else None
        results: list[SearchResult] = []
        for item in raw_results or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            snippet = BeautifulSoup(str(item.get("description") or ""), "html.parser").get_text(
                " ", strip=True
            )
            results.append(
                SearchResult(
                    title=normalize_whitespace(str(item.get("title") or url)),
                    url=url,
                    snippet=normalize_whitespace(snippet),
                )
            )
            if len(results) >= limit:
                break
        return results


def build_search_provider(settings: Settings) -> SearchProvider:
    if settings.search_provider == "tavily":
        return TavilyProvider(settings.tavily_key(), timeout=settings.request_timeout)
    if settings.search_provider == "brave":
        return BraveProvider(settings.brave_key(), timeout=settings.request_timeout)
    return DdgsProvider(timeout=min(settings.request_timeout, 30.0))


class WebSearchTool:
    name = "web_search"
    description = (
        "Search the web and return titles, URLs, and snippets. Follow useful "
        "results with fetch_url to read the full page and its outgoing links."
    )
    parameters = _SEARCH_PARAMS

    def __init__(self, provider: SearchProvider) -> None:
        self.provider = provider

    def __call__(self, query: str, limit: int | None = None) -> ToolResult:
        query = require_non_empty_string(query, "query")
        limit = coerce_int(limit, field="limit", default=5, minimum=1, maximum=10)
        results = self.provider.search(query, limit)
        if not results:
            return ToolResult(
                f"No web search results found for {query!r} with provider "
                f"{self.provider.name}. Try different terms."
            )
        lines = [f"Web search results for {query!r} (provider: {self.provider.name}):"]
        for index, result in enumerate(results, start=1):
            lines.append(f"{index}. {result.title}\n   URL: {result.url}")
            if result.snippet:
                lines.append(f"   {result.snippet}")
        return ToolResult("\n".join(lines))
