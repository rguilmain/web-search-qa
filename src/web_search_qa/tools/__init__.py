"""Tool registry construction for the application."""

from __future__ import annotations

from web_search_qa.config import Settings
from web_search_qa.tools.base import ToolRegistry, tool_from
from web_search_qa.tools.fetch import FetchTool
from web_search_qa.tools.web_search import (
    WebSearchTool,
    build_search_provider,
)
from web_search_qa.tools.wikipedia import (
    WikipediaClient,
    WikipediaPageTool,
    WikipediaSearchTool,
)

__all__ = [
    "FetchTool",
    "ToolRegistry",
    "WebSearchTool",
    "WikipediaClient",
    "WikipediaPageTool",
    "WikipediaSearchTool",
    "build_registry",
]


def build_registry(settings: Settings) -> ToolRegistry:
    """Build the default tool registry for a settings object."""

    wikipedia_client = WikipediaClient(settings)
    provider = build_search_provider(settings)
    registry = ToolRegistry()
    registry.register(tool_from(WikipediaSearchTool(wikipedia_client)))
    registry.register(tool_from(WikipediaPageTool(wikipedia_client)))
    registry.register(tool_from(WebSearchTool(provider)))
    registry.register(tool_from(FetchTool(settings)))
    return registry
