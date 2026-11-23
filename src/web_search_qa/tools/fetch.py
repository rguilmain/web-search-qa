"""Fetch and extract readable content from a URL, with SSRF protections."""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from bs4.element import Tag
from pypdf import PdfReader

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

_FETCH_PARAMS = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "HTTP or HTTPS URL to fetch.",
        },
        "offset": {
            "type": "integer",
            "description": "Character offset for reading a long page in chunks.",
            "minimum": 0,
            "default": 0,
        },
    },
    "required": ["url"],
}

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_HTML_TYPES = {"text/html", "application/xhtml+xml"}
_TEXT_TYPES = {
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/json",
    "application/xml",
    "text/xml",
}
_PDF_TYPE = "application/pdf"
_MATH_TOKEN_PREFIX = "WEBSEARCHQAMATH"
_MATH_TOKEN_SUFFIX = "ENDMATH"


@dataclass(frozen=True, slots=True)
class DownloadResult:
    original_url: str
    final_url: str
    body: bytes
    content_type: str
    encoding: str | None
    redirects: list[str]


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    text: str
    title: str
    links: list[tuple[str, str, str]]


def validate_url(
    url: str,
    *,
    resolver: Callable[[str, int], list[str]] | None = None,
) -> str:
    """Validate a URL and reject non-public or non-HTTP(S) targets."""

    candidate = _normalize_url(url)
    try:
        parsed = urlparse(candidate)
    except ValueError as exc:
        raise ToolError(f"Invalid URL: {exc}") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ToolError("Only http:// and https:// URLs are allowed.")
    if parsed.username or parsed.password:
        raise ToolError("URLs containing credentials are not allowed.")
    try:
        host = (parsed.hostname or "").strip().rstrip(".").lower()
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise ToolError(f"Invalid URL host or port: {exc}") from exc
    if not host:
        raise ToolError("The URL must include a hostname.")
    if _is_local_hostname(host):
        raise ToolError(f"Blocked non-public hostname: {host!r}.")

    addresses: list[str]
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        addresses = resolver(host, port) if resolver is not None else _resolve_host(host, port)
    else:
        addresses = [str(literal)]

    if not addresses:
        raise ToolError(f"Could not resolve the hostname {host!r}.")
    for address in addresses:
        if not _is_public_ip(address):
            raise ToolError(
                f"Blocked request to {host!r}: it resolves to the non-public "
                f"address {address}. Only public internet hosts are allowed."
            )
    return candidate


def _normalize_url(url: str) -> str:
    value = url.strip()
    if not value:
        raise ToolError("url must be a non-empty string")
    if value.startswith("//"):
        return f"https:{value}"
    if "://" not in value:
        return f"https://{value}"
    return value


def _is_local_hostname(host: str) -> bool:
    if host in {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}:
        return True
    return any(
        host.endswith(suffix) for suffix in (".localhost", ".local", ".internal", ".home.arpa")
    )


def _resolve_host(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ToolError(f"Could not resolve the hostname {host!r}: {exc}") from exc
    return sorted({str(info[4][0]).split("%", 1)[0] for info in infos})


def _is_public_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(
        ip.is_global
        and not ip.is_private
        and not ip.is_loopback
        and not ip.is_link_local
        and not ip.is_multicast
        and not ip.is_reserved
        and not ip.is_unspecified
        and not getattr(ip, "is_site_local", False)
    )


class FetchTool:
    name = "fetch_url"
    description = (
        "Fetch an HTTP(S) page and extract its readable main content, title, and "
        "outgoing links with context. Math markup is preserved as LaTeX. "
        "Handles redirects, timeouts, size limits, and PDFs. Private/local "
        "network addresses are blocked."
    )
    parameters = _FETCH_PARAMS

    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: Callable[[], httpx.Client] | None = None,
        resolver: Callable[[str, int], list[str]] | None = None,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory
        self._resolver = resolver

    @contextmanager
    def _client(self) -> Iterator[httpx.Client]:
        if self._client_factory is not None:
            client = self._client_factory()
            try:
                yield client
            finally:
                client.close()
        else:
            headers = {
                "User-Agent": self.settings.fetch_user_agent,
                "Accept": (
                    "text/html,application/xhtml+xml,application/pdf,text/plain,"
                    "application/json,application/xml;q=0.9,*/*;q=0.1"
                ),
            }
            with httpx.Client(
                timeout=self.settings.fetch_timeout,
                headers=headers,
                follow_redirects=False,
            ) as client:
                yield client

    def __call__(self, url: str, offset: int | None = None) -> ToolResult:
        url = require_non_empty_string(url, "url")
        offset = coerce_int(offset, field="offset", default=0, minimum=0, maximum=100_000_000)
        download = self._download(url)
        extracted = self._extract(download)
        if not extracted.text.strip():
            return ToolResult.failure(
                f"Fetched {download.final_url}, but no readable text could be "
                "extracted. The page may require JavaScript, block automated access, "
                "or contain only an image or video."
            )

        chunk, range_note = page_slice(extracted.text, offset, self.settings.fetch_page_chars)
        parts = [f"URL: {download.final_url}"]
        if extracted.title:
            parts.append(f"Title: {extracted.title}")
        parts.append(f"Content type: {download.content_type or 'unknown'}")
        if download.final_url != download.original_url:
            parts.append(f"Redirected from: {download.original_url}")
        parts.append(range_note)
        parts.append(chunk)
        if offset == 0 and extracted.links:
            parts.append("## Outgoing links")
            parts.append(_format_links(extracted.links))
        elif offset > 0 and extracted.links:
            parts.append(
                "Outgoing links were included on the first page (offset=0); "
                "call with offset=0 if you need them again."
            )
        return ToolResult("\n\n".join(parts))

    def _download(self, url: str) -> DownloadResult:
        current = validate_url(url, resolver=self._resolver)
        redirects: list[str] = []
        for _ in range(self.settings.fetch_max_redirects + 1):
            with self._client() as client:
                try:
                    with client.stream(
                        "GET",
                        current,
                        headers={"User-Agent": self.settings.fetch_user_agent},
                    ) as response:
                        if response.status_code in _REDIRECT_STATUSES:
                            location = response.headers.get("location")
                            if not location:
                                raise ToolError(
                                    f"HTTP {response.status_code} redirect from {current} "
                                    "did not include a Location header."
                                )
                            redirects.append(current)
                            current = validate_url(
                                urljoin(str(response.url), location),
                                resolver=self._resolver,
                            )
                            continue
                        if response.status_code >= 400:
                            hint = ""
                            if response.status_code in {401, 403}:
                                hint = " The site may block automated requests."
                            elif response.status_code == 429:
                                hint = " The site is rate-limiting requests; wait and retry."
                            raise ToolError(
                                f"HTTP {response.status_code} {response.reason_phrase} "
                                f"while fetching {current}.{hint}"
                            )
                        content_type = (
                            response.headers.get("content-type", "")
                            .split(";", 1)[0]
                            .strip()
                            .lower()
                        )
                        chunks: list[bytes] = []
                        total = 0
                        for chunk in response.iter_bytes():
                            total += len(chunk)
                            if total > self.settings.fetch_max_bytes:
                                raise ToolError(
                                    f"The response from {current} is larger than the "
                                    f"configured {self.settings.fetch_max_bytes:,}-byte "
                                    "limit. Fetch a more specific page instead."
                                )
                            chunks.append(chunk)
                        return DownloadResult(
                            original_url=url,
                            final_url=current,
                            body=b"".join(chunks),
                            content_type=content_type,
                            encoding=response.encoding,
                            redirects=redirects,
                        )
                except httpx.TimeoutException as exc:
                    raise ToolError(
                        f"The request to {current} timed out after "
                        f"{self.settings.fetch_timeout:g} seconds."
                    ) from exc
                except httpx.HTTPError as exc:
                    raise ToolError(f"The request to {current} failed: {exc}") from exc
        raise ToolError(
            f"Too many redirects while fetching {url!r} "
            f"(limit: {self.settings.fetch_max_redirects})."
        )

    def _extract(self, download: DownloadResult) -> ExtractedPage:
        kind = _classify_content(download)
        if kind == "pdf":
            return ExtractedPage(
                text=self._extract_pdf(download.body),
                title="",
                links=[],
            )
        if kind == "html":
            return _extract_html(
                download.body,
                download.encoding,
                download.final_url,
                self.settings.fetch_max_links,
            )
        if kind == "text":
            text = _decode(download.body, download.encoding)
            return ExtractedPage(text=_clean_text(text), title="", links=[])
        raise ToolError(
            f"Unsupported content type {download.content_type or '(missing)'} at "
            f"{download.final_url}. Only HTML, plain text, JSON, XML, and PDF are supported."
        )

    def _extract_pdf(self, body: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(body))
        except Exception as exc:
            raise ToolError(f"Could not read the PDF: {exc}") from exc
        if reader.is_encrypted:
            try:
                if reader.decrypt("") == 0:
                    raise ToolError("The PDF is encrypted and cannot be read.")
            except Exception as exc:
                raise ToolError(f"The PDF is encrypted and cannot be read: {exc}") from exc
        max_pages = 50
        pages = reader.pages
        page_texts: list[str] = []
        for index in range(min(len(pages), max_pages)):
            try:
                page_text = pages[index].extract_text() or ""
            except Exception:
                continue
            page_text = page_text.strip()
            if page_text:
                page_texts.append(f"[Page {index + 1}]\n{page_text}")
        if len(pages) > max_pages:
            page_texts.append(
                f"[Only the first {max_pages} of {len(pages)} PDF pages were extracted.]"
            )
        return _clean_text("\n\n".join(page_texts))


def _classify_content(download: DownloadResult) -> str:
    content_type = download.content_type
    if content_type == _PDF_TYPE:
        return "pdf"
    if content_type in _HTML_TYPES:
        return "html"
    if content_type in _TEXT_TYPES:
        return "text"
    if content_type in {"application/octet-stream", ""}:
        body = download.body
        if body.startswith(b"%PDF") or download.final_url.lower().endswith(".pdf"):
            return "pdf"
        head = body[:512].lstrip().lower()
        if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
            return "html"
        if not content_type:
            return "text"
    return "unsupported"


def _decode(body: bytes, encoding: str | None) -> str:
    for candidate in (encoding, "utf-8", "latin-1"):
        if not candidate:
            continue
        try:
            return body.decode(candidate, errors="replace")
        except LookupError:
            continue
    return body.decode("utf-8", errors="replace")


def _replace_math_with_token(soup: BeautifulSoup, element: Tag, token: str) -> None:
    tag = soup.new_tag("span")
    tag.string = token
    element.replace_with(tag)


def _convert_math_markup(soup: BeautifulSoup) -> dict[str, str]:
    """Replace common web math markup with opaque tokens before extraction.

    Tokens keep LaTeX out of trafilatura's Markdown escaping (for example
    ``_`` becoming ``\\_``) and are restored after extraction.
    """

    mapping: dict[str, str] = {}

    def add(tex: str, display: bool) -> str:
        token = f"{_MATH_TOKEN_PREFIX}{len(mapping)}{_MATH_TOKEN_SUFFIX}"
        mapping[token] = f"$$ {tex} $$" if display else f"\\({tex}\\)"
        return token

    for script in list(soup.find_all("script")):
        script_type = " ".join(str(script.get("type") or "").lower().split(";"))
        if "math/tex" not in script_type:
            continue
        tex = script.get_text().strip()
        if not tex:
            continue
        classes = script.get("class") or []
        if isinstance(classes, str):
            classes = classes.split()
        display = "mode=display" in script_type or any(
            name in {"math-display", "display"} for name in classes
        )
        _replace_math_with_token(soup, script, add(tex, display))

    for math in list(soup.find_all("math")):
        tex = str(math.get("alttext") or "").strip()
        if not tex:
            annotation = math.find("annotation", attrs={"encoding": "application/x-tex"})
            if annotation is not None:
                tex = annotation.get_text().strip()
        if not tex:
            continue
        classes = math.get("class") or []
        if isinstance(classes, str):
            classes = classes.split()
        display = (
            str(math.get("display") or "").lower()
            in {
                "block",
                "true",
            }
            or "mwe-math-element-block" in classes
        )
        _replace_math_with_token(soup, math, add(tex, display))

    return mapping


def _restore_math(text: str, mapping: dict[str, str]) -> str:
    for token, latex in mapping.items():
        text = text.replace(token, latex)
    return text


def _extract_html(body: bytes, encoding: str | None, url: str, max_links: int) -> ExtractedPage:
    html = _decode(body, encoding)
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = normalize_whitespace(soup.title.string)
    if not title:
        heading = soup.find("h1")
        if heading:
            title = normalize_whitespace(heading.get_text(" ", strip=True))
    links = _extract_outgoing_links(soup, url, max_links)
    math_tokens = _convert_math_markup(soup)
    html = str(soup)

    try:
        extracted = trafilatura.extract(
            html,
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
        return ExtractedPage(
            text=_restore_math(_clean_text(extracted), math_tokens),
            title=title,
            links=links,
        )

    for tag in soup.select(
        "script, style, noscript, template, svg, canvas, nav, header, footer, aside, "
        "form, iframe, [role='navigation'], [aria-hidden='true'], .cookie, "
        ".cookies, .newsletter, .advertisement, .ad"
    ):
        tag.decompose()
    root = soup.select_one("main, article") or soup.body or soup
    lines = [
        normalize_whitespace(line)
        for line in root.get_text("\n").splitlines()
        if normalize_whitespace(line)
    ]
    return ExtractedPage(
        text=_restore_math(_clean_text("\n".join(lines)), math_tokens),
        title=title,
        links=links,
    )


def _extract_outgoing_links(
    soup: BeautifulSoup, page_url: str, max_links: int
) -> list[tuple[str, str, str]]:
    if max_links <= 0:
        return []
    root = soup.select_one("main, article") or soup.body or soup
    links: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    page_key = _url_key(page_url)
    for anchor in root.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        if href.lower().startswith(("javascript:", "mailto:", "tel:", "data:")):
            continue
        absolute = urljoin(page_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        absolute = parsed._replace(fragment="").geturl()
        key = _url_key(absolute)
        if key == page_key or key in seen:
            continue
        seen.add(key)
        text = normalize_whitespace(anchor.get_text(" ", strip=True)) or absolute
        links.append((text, absolute, link_context(anchor)))
        if len(links) >= max_links:
            break
    return links


def _format_links(links: list[tuple[str, str, str]]) -> str:
    lines: list[str] = []
    for index, (text, url, context) in enumerate(links, start=1):
        lines.append(f"{index}. {text} — {url}")
        if context:
            lines.append(f"   context: {context}")
    return "\n".join(lines)


def _url_key(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{host}{path}?{parsed.query}"


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
