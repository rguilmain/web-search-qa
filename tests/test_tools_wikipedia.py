from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr

from web_search_qa.tools.base import ToolError
from web_search_qa.tools.wikipedia import (
    WikipediaClient,
    WikipediaPageTool,
    WikipediaSearchTool,
)

PAGE_HTML = """
<html>
  <head><title>Ada Lovelace</title></head>
  <body>
    <div class="mw-parser-output">
      <p>Ada Lovelace was an English mathematician and writer. She is chiefly
      known for her work on <a href="/wiki/Charles_Babbage">Charles Babbage</a>'s
      proposed mechanical general-purpose computer, the Analytical Engine.</p>
      <p>She described an algorithm for computing Bernoulli numbers with the
      Analytical Engine, which is often regarded as the first computer program.
      This paragraph adds enough prose for the extractor to keep the article.</p>
      <h2>References</h2>
      <ol class="references">
        <li><a href="https://example.com/ada-paper">Lovelace, Ada (1843).
        Sketch of the Analytical Engine.</a> A primary source with notes.</li>
      </ol>
      <p>See also <a href="/wiki/Category:Ada_Lovelace">the category</a> and
      <a href="https://en.wikipedia.org/wiki/Special:Search">search</a>.</p>
    </div>
  </body>
</html>
"""


def make_client(settings, handler) -> WikipediaClient:
    return WikipediaClient(
        settings,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_search_strips_html_and_sends_identity(settings) -> None:
    settings = settings.model_copy(update={"wikipedia_api_key": SecretStr("test-token")})

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["action"] == "query"
        assert request.url.params["list"] == "search"
        assert request.url.params["srsearch"] == "Ada Lovelace"
        assert request.headers["User-Agent"] == settings.wikipedia_user_agent
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(
            200,
            json={
                "query": {
                    "searchinfo": {"totalhits": 2},
                    "search": [
                        {
                            "title": "Ada Lovelace",
                            "snippet": '<span class="searchmatch">Ada</span> Lovelace was a mathematician.',
                            "wordcount": 1234,
                        }
                    ],
                }
            },
        )

    result = WikipediaSearchTool(make_client(settings, handler))("Ada Lovelace", 5)
    assert result.ok is True
    assert "2 total hits" in result.content
    assert "https://en.wikipedia.org/wiki/Ada_Lovelace" in result.content
    assert "<span" not in result.content
    assert "Ada Lovelace was a mathematician." in result.content


def test_page_returns_text_links_and_context(settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["action"] == "parse"
        assert request.url.params["page"] == "Ada Lovelace"
        assert "text" in request.url.params["prop"]
        assert request.url.params["redirects"] == "1"
        assert request.url.params["formatversion"] == "2"
        return httpx.Response(
            200,
            json={
                "parse": {
                    "title": "Ada Lovelace",
                    "pageid": 1,
                    "text": PAGE_HTML,
                }
            },
        )

    result = WikipediaPageTool(make_client(settings, handler))("Ada Lovelace")
    assert result.ok is True
    assert "Wikipedia page: Ada Lovelace" in result.content
    assert "https://en.wikipedia.org/wiki/Ada_Lovelace" in result.content
    assert "English mathematician" in result.content
    assert "Internal Wikipedia links" in result.content
    assert "Charles Babbage" in result.content
    assert "https://en.wikipedia.org/wiki/Charles_Babbage" in result.content
    assert "External links and references" in result.content
    assert "https://example.com/ada-paper" in result.content
    assert "context:" in result.content
    assert "Category:Ada_Lovelace" not in result.content
    assert "Special:Search" not in result.content


def test_page_paginates_and_points_back_to_links(settings) -> None:
    settings = settings.model_copy(update={"wikipedia_page_chars": 80})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"parse": {"title": "Ada Lovelace", "text": PAGE_HTML}},
        )

    result = WikipediaPageTool(make_client(settings, handler))("Ada Lovelace", 80)
    assert result.ok is True
    assert "Characters 80-" in result.content
    assert "Links were included on the first page" in result.content


def test_page_api_error_is_a_tool_error(settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "error": {
                    "code": "missingtitle",
                    "info": "The page you specified doesn't exist.",
                }
            },
        )

    with pytest.raises(ToolError, match="doesn't exist"):
        WikipediaPageTool(make_client(settings, handler))("Missing page")


def test_empty_search_is_not_an_error(settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": {"search": []}})

    result = WikipediaSearchTool(make_client(settings, handler))("nothing at all")
    assert result.ok is True
    assert "No English Wikipedia results" in result.content
