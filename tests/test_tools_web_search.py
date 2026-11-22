from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from web_search_qa.tools.base import ToolError
from web_search_qa.tools.web_search import (
    BraveProvider,
    DdgsProvider,
    SearchResult,
    TavilyProvider,
    WebSearchTool,
    build_search_provider,
)


class FakeProvider:
    name = "fake"

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = results or []
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[SearchResult]:
        self.queries.append((query, limit))
        return self.results


def test_web_search_formats_results() -> None:
    provider = FakeProvider(
        [
            SearchResult(
                title="Example result",
                url="https://example.com/result",
                snippet="A useful snippet.",
            )
        ]
    )
    result = WebSearchTool(provider)("example query", 3)
    assert result.ok is True
    assert "provider: fake" in result.content
    assert "1. Example result" in result.content
    assert "https://example.com/result" in result.content
    assert provider.queries == [("example query", 3)]


def test_web_search_empty_results_is_not_an_error() -> None:
    result = WebSearchTool(FakeProvider())("nothing")
    assert result.ok is True
    assert "No web search results" in result.content


def test_tavily_provider_parses_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.tavily.com/search"
        body = json.loads(request.content)
        assert body["api_key"] == "tavily-key"
        assert body["query"] == "test query"
        assert body["max_results"] == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Tavily result",
                        "url": "https://example.com/tavily",
                        "content": "Tavily snippet.",
                    }
                ]
            },
        )

    provider = TavilyProvider(
        "tavily-key",
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    results = provider.search("test query", 2)
    assert results == [
        SearchResult(
            title="Tavily result",
            url="https://example.com/tavily",
            snippet="Tavily snippet.",
        )
    ]


def test_brave_provider_strips_html_from_descriptions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Subscription-Token"] == "brave-key"
        assert request.url.params["q"] == "test query"
        assert request.url.params["count"] == "2"
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Brave result",
                            "url": "https://example.com/brave",
                            "description": "<strong>Brave</strong> snippet.",
                        }
                    ]
                }
            },
        )

    provider = BraveProvider(
        "brave-key",
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    results = provider.search("test query", 2)
    assert results[0].title == "Brave result"
    assert results[0].snippet == "Brave snippet."


def test_provider_without_key_reports_a_clear_error() -> None:
    provider = TavilyProvider(None)
    with pytest.raises(ToolError, match="TAVILY_API_KEY"):
        provider.search("query", 5)


def test_provider_selection(settings) -> None:
    assert isinstance(build_search_provider(settings), DdgsProvider)

    tavily_settings = settings.model_copy(
        update={"search_provider": "tavily", "tavily_api_key": SecretStr("key")}
    )
    assert isinstance(build_search_provider(tavily_settings), TavilyProvider)

    brave_settings = settings.model_copy(
        update={"search_provider": "brave", "brave_api_key": SecretStr("key")}
    )
    assert isinstance(build_search_provider(brave_settings), BraveProvider)
