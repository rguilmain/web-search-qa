from __future__ import annotations

from io import BytesIO

import httpx
import pytest
from pypdf import PdfWriter

from web_search_qa.tools.base import ToolError
from web_search_qa.tools.fetch import FetchTool, validate_url

ARTICLE_HTML = """
<html>
  <head><title>Example Research Page</title></head>
  <body>
    <nav>Navigation junk</nav>
    <main>
      <article>
        <h1>Example Research Page</h1>
        <p>This is a substantial article about a historical event. It contains
        enough prose for the extractor to recognize the main content and keep it
        instead of the surrounding navigation and footer.</p>
        <p>Another paragraph includes a
        <a href="/related">related page</a> and more details about the event and
        the people who recorded it.</p>
      </article>
    </main>
    <footer>Footer junk</footer>
  </body>
</html>
"""


def make_fetch_tool(settings, handler) -> FetchTool:
    transport = httpx.MockTransport(handler)
    return FetchTool(
        settings,
        client_factory=lambda: httpx.Client(transport=transport),
    )


def test_validate_url_rejects_local_and_non_http_targets() -> None:
    bad_urls = [
        "file:///etc/passwd",
        "http://127.0.0.1/",
        "http://localhost/",
        "http://10.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://user:password@93.184.216.34/",
    ]
    for url in bad_urls:
        with pytest.raises(ToolError):
            validate_url(url)


def test_validate_url_checks_resolved_addresses() -> None:
    with pytest.raises(ToolError, match="non-public"):
        validate_url("http://example.com/", resolver=lambda host, port: ["10.0.0.5"])

    result = validate_url(
        "example.com/article",
        resolver=lambda host, port: ["93.184.216.34"],
    )
    assert result == "https://example.com/article"


def test_fetch_extracts_main_text_and_links(settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "93.184.216.34"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=ARTICLE_HTML,
        )

    tool = make_fetch_tool(settings, handler)
    result = tool("http://93.184.216.34/article")

    assert result.ok is True
    assert "URL: http://93.184.216.34/article" in result.content
    assert "Title: Example Research Page" in result.content
    assert "substantial article" in result.content
    assert "## Outgoing links" in result.content
    assert "http://93.184.216.34/related" in result.content
    assert "related page" in result.content


def test_fetch_follows_redirects_and_reports_final_url(settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/final"})
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=ARTICLE_HTML,
        )

    tool = make_fetch_tool(settings, handler)
    result = tool("http://93.184.216.34/start")
    assert result.ok is True
    assert "URL: http://93.184.216.34/final" in result.content
    assert "Redirected from: http://93.184.216.34/start" in result.content


def test_fetch_blocks_redirect_to_private_address(settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})

    tool = make_fetch_tool(settings, handler)
    with pytest.raises(ToolError, match="non-public"):
        tool("http://93.184.216.34/start")


def test_fetch_rejects_unsupported_content_type(settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=b"\x89PNG\r\n\x1a\n",
        )

    tool = make_fetch_tool(settings, handler)
    with pytest.raises(ToolError, match="Unsupported content type"):
        tool("http://93.184.216.34/image.png")


def test_fetch_enforces_size_limit(settings) -> None:
    settings = settings.model_copy(update={"fetch_max_bytes": 1_000})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=b"x" * 2_000,
        )

    tool = make_fetch_tool(settings, handler)
    with pytest.raises(ToolError, match="larger than"):
        tool("http://93.184.216.34/big.txt")


def test_fetch_paginates_long_text(settings) -> None:
    settings = settings.model_copy(update={"fetch_page_chars": 80})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="word " * 100,
        )

    tool = make_fetch_tool(settings, handler)
    result = tool("http://93.184.216.34/text", offset=80)
    assert result.ok is True
    assert "Characters 80-" in result.content


def test_fetch_pdf_without_text_is_a_clear_failure(settings) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = BytesIO()
    writer.write(buffer)
    pdf_bytes = buffer.getvalue()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=pdf_bytes,
        )

    tool = make_fetch_tool(settings, handler)
    result = tool("http://93.184.216.34/blank.pdf")
    assert result.ok is False
    assert "no readable text" in result.content.lower()
