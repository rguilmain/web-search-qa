"""Wikipedia tools backed by the MediaWiki Action API."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import ParseResult, quote, unquote, urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

from web_search_qa.config import Settings
from web_search_qa.tools.base import (
    ToolError,
    ToolResult,
    coerce_int,
    link_context,
    normalize_whitespace,
    page_slice,
    require_non_empty_string,
)

_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search terms, for example 'history of the bicycle'.",
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

_PAGE_PARAMS = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Exact Wikipedia article title, for example 'Ada Lovelace'.",
        },
        "offset": {
            "type": "integer",
            "description": "Character offset for reading a long article in chunks.",
            "minimum": 0,
            "default": 0,
        },
    },
    "required": ["title"],
}


class WikipediaClient:
    """Minimal client for the MediaWiki Action API."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": self.settings.wikipedia_user_agent,
            "Accept": "application/json",
        }
        token = self.settings.wikipedia_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def get(self, params: dict[str, Any]) -> dict[str, Any]:
        request_params = {"format": "json", "formatversion": "2", **params}
        try:
            if self._client is not None:
                response = self._client.get(
                    self.settings.wikipedia_api_url,
                    params=request_params,
                    headers=self._headers(),
                )
            else:
                with httpx.Client(
                    timeout=self.settings.request_timeout,
                    headers=self._headers(),
                    follow_redirects=True,
                ) as client:
                    response = client.get(
                        self.settings.wikipedia_api_url,
                        params=request_params,
                    )
        except httpx.TimeoutException as exc:
            raise ToolError(
                "The Wikipedia request timed out. Try again or use a shorter query."
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"The Wikipedia request failed: {exc}") from exc

        if response.status_code != 200:
            raise ToolError(
                f"Wikipedia returned HTTP {response.status_code} for the {params.get('action')} action."
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ToolError("Wikipedia returned a response that was not valid JSON.") from exc
        if not isinstance(data, dict):
            raise ToolError("Wikipedia returned an unexpected response shape.")
        if "error" in data:
            error = data["error"]
            info = error.get("info") if isinstance(error, dict) else str(error)
            raise ToolError(f"Wikipedia API error: {info}")
        return data


class WikipediaSearchTool:
    name = "wikipedia_search"
    description = (
        "Search English Wikipedia. Returns titles, snippets, and URLs. Use "
        "wikipedia_page to read an article and its links."
    )
    parameters = _SEARCH_PARAMS

    def __init__(self, client: WikipediaClient) -> None:
        self.client = client

    def __call__(self, query: str, limit: int | None = None) -> ToolResult:
        query = require_non_empty_string(query, "query")
        limit = coerce_int(limit, field="limit", default=5, minimum=1, maximum=10)
        data = self.client.get(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srprop": "snippet|wordcount|timestamp",
            }
        )
        query_result = data.get("query") or {}
        results = query_result.get("search") or []
        if not results:
            return ToolResult(
                f"No English Wikipedia results found for {query!r}. Try broader or "
                "different search terms."
            )

        total = (query_result.get("searchinfo") or {}).get("totalhits")
        heading = f"Wikipedia search results for {query!r}"
        if isinstance(total, int):
            heading += f" ({total:,} total hits)"
        lines = [heading + ":"]
        for index, result in enumerate(results, start=1):
            title = str(result.get("title") or "").strip()
            snippet = _strip_html(str(result.get("snippet") or ""))
            url = wikipedia_url(self.client.settings, title)
            lines.append(f"{index}. {title}\n   URL: {url}")
            if snippet:
                lines.append(f"   {snippet}")
            wordcount = result.get("wordcount")
            if isinstance(wordcount, int):
                lines.append(f"   Words: {wordcount:,}")
        return ToolResult("\n".join(lines))


@dataclass(frozen=True, slots=True)
class _Link:
    text: str
    url: str
    context: str


class WikipediaPageTool:
    name = "wikipedia_page"
    description = (
        "Read a Wikipedia article as clean text, with numbered internal links and "
        "external/reference links plus context. Use offset to read long articles in "
        "chunks, and follow useful links with wikipedia_page or fetch_url."
    )
    parameters = _PAGE_PARAMS

    def __init__(self, client: WikipediaClient) -> None:
        self.client = client

    def __call__(self, title: str, offset: int | None = None) -> ToolResult:
        title = require_non_empty_string(title, "title")
        offset = coerce_int(offset, field="offset", default=0, minimum=0, maximum=10_000_000)
        data = self.client.get(
            {
                "action": "parse",
                "page": title,
                "prop": "text|displaytitle|revid",
                "redirects": "1",
                "disableeditsection": "1",
            }
        )
        parsed = data.get("parse")
        if not isinstance(parsed, dict):
            raise ToolError(
                f"Wikipedia could not parse the page {title!r}. Use wikipedia_search "
                "to find the exact title."
            )
        html = parsed.get("text") or ""
        if isinstance(html, dict):
            html = html.get("*") or ""
        if not isinstance(html, str) or not html.strip():
            raise ToolError(f"Wikipedia returned no article content for {title!r}.")

        page_title = str(parsed.get("title") or title).strip()
        page_url = wikipedia_url(self.client.settings, page_title)
        soup = BeautifulSoup(html, "html.parser")
        internal: list[_Link] = []
        external: list[_Link] = []
        if offset == 0:
            # Extract links before the fallback text path can mutate the soup.
            internal, external = _extract_links(
                soup, page_url, self.client.settings.wikipedia_max_links
            )
        text = _extract_article_text(page_url, soup)
        if not text:
            raise ToolError(
                f"Wikipedia returned {page_title!r}, but no readable article text "
                "could be extracted."
            )

        chunk, range_note = page_slice(text, offset, self.client.settings.wikipedia_page_chars)
        parts = [
            f"Wikipedia page: {page_title}",
            f"URL: {page_url}",
            range_note,
            chunk,
        ]
        if offset == 0:
            if internal:
                parts.append(
                    "## Internal Wikipedia links\n"
                    "Use wikipedia_page with the link title, or fetch_url with the URL."
                )
                parts.append(_format_links(internal))
            if external:
                parts.append(
                    "## External links and references\n"
                    "Use fetch_url on these URLs to read the underlying sources."
                )
                parts.append(_format_links(external))
        else:
            parts.append(
                "Links were included on the first page (offset=0); call with "
                "offset=0 if you need the link list again."
            )
        return ToolResult("\n\n".join(part for part in parts if part))


def wikipedia_url(settings: Settings, title: str) -> str:
    path_title = quote(title.replace(" ", "_"), safe="/")
    return f"https://{settings.wikipedia_language}.wikipedia.org/wiki/{path_title}"


def _strip_html(value: str) -> str:
    return normalize_whitespace(BeautifulSoup(value, "html.parser").get_text(" ", strip=True))


_ARTICLE_NOISE_SELECTOR = (
    "script, style, noscript, template, sup.reference, .mw-editsection, "
    ".navbox, .vertical-navbox, .sidebar, .metadata, .ambox, .hatnote, "
    ".shortdescription, .mw-jump-link, .mw-empty-elt, .reflist, ol.references, "
    ".mw-references-wrap, #toc"
)


def _extract_article_text(url: str, soup: BeautifulSoup) -> str:
    # Remove navigation boxes, citation markers, and reference lists before
    # extraction. Links have already been collected from the original soup.
    for tag in soup.select(_ARTICLE_NOISE_SELECTOR):
        tag.decompose()
    cleaned_html = str(soup)
    try:
        extracted = trafilatura.extract(
            cleaned_html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_links=False,
            output_format="markdown",
            favor_precision=False,
        )
    except Exception:
        extracted = None
    if extracted and len(normalize_whitespace(extracted)) >= 40:
        return _clean_text(extracted)

    root = soup.select_one(".mw-parser-output") or soup.body or soup
    lines = [
        normalize_whitespace(line)
        for line in root.get_text("\n").splitlines()
        if normalize_whitespace(line)
    ]
    return _clean_text("\n".join(lines))


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_links(
    soup: BeautifulSoup, page_url: str, max_links: int
) -> tuple[list[_Link], list[_Link]]:
    if max_links <= 0:
        return [], []
    root = soup.select_one(".mw-parser-output") or soup
    internal: list[_Link] = []
    external: list[_Link] = []
    seen_internal: set[str] = set()
    seen_external: set[str] = set()
    page_key = _url_key(page_url)

    for anchor in root.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        absolute = urljoin(page_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        absolute = parsed._replace(fragment="").geturl()
        if _url_key(absolute) == page_key:
            continue
        anchor_text = normalize_whitespace(anchor.get_text(" ", strip=True))
        host = (parsed.hostname or "").lower()

        if host == "wikipedia.org" or host.endswith(".wikipedia.org"):
            if not _is_article_link(parsed):
                continue
            key = _url_key(absolute)
            if key in seen_internal or len(internal) >= max_links:
                continue
            seen_internal.add(key)
            internal.append(
                _Link(
                    text=anchor_text or _title_from_wikipedia_url(parsed),
                    url=absolute,
                    context=link_context(anchor),
                )
            )
        else:
            if host.endswith("wikimedia.org") or host == "wikimedia.org":
                continue
            if host in {"donate.wikimedia.org", "wikimediafoundation.org"}:
                continue
            key = _url_key(absolute)
            if key in seen_external or len(external) >= max_links:
                continue
            seen_external.add(key)
            external.append(
                _Link(
                    text=anchor_text or absolute,
                    url=absolute,
                    context=link_context(anchor),
                )
            )
    return internal, external


def _is_article_link(parsed: ParseResult) -> bool:
    path = parsed.path or ""
    if not path.startswith("/wiki/"):
        return False
    title = path.removeprefix("/wiki/")
    if not title or title == "Main_Page":
        return False
    # Namespace pages such as File:, Category:, Special:, and Help: are not
    # useful for research and are usually navigation.
    return ":" not in title


def _title_from_wikipedia_url(parsed: ParseResult) -> str:
    title = (parsed.path or "").removeprefix("/wiki/")
    return unquote(title).replace("_", " ") or parsed.geturl()


def _url_key(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{host}{path}?{parsed.query}"


def _format_links(links: list[_Link]) -> str:
    lines: list[str] = []
    for index, link in enumerate(links, start=1):
        lines.append(f"{index}. {link.text} — {link.url}")
        if link.context:
            lines.append(f"   context: {link.context}")
    return "\n".join(lines)
