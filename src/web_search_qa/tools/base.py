"""Tool primitives and the small tool registry used by the agent."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


class ToolError(Exception):
    """A recoverable tool failure that should be shown to the model."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The result of running a tool.

    ``ok=False`` is used for expected failures (bad URL, HTTP error, empty
    extraction, and so on). The agent always turns both success and failure
    into observations; tools never abort the research loop.
    """

    content: str
    ok: bool = True

    @classmethod
    def failure(cls, content: str) -> ToolResult:
        return cls(content=content, ok=False)


class ToolLike(Protocol):
    """A callable object that exposes a tool's metadata."""

    name: str
    description: str
    parameters: dict[str, Any]

    def __call__(self, **kwargs: Any) -> str | ToolResult: ...


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool definition plus its implementation."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str | ToolResult]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            result = self.func(**arguments)
        except ToolError as exc:
            return ToolResult.failure(str(exc))
        except TypeError as exc:
            return ToolResult.failure(f"Invalid arguments for {self.name}: {exc}")
        except Exception as exc:
            logger.exception("Unexpected error in tool %s", self.name)
            return ToolResult.failure(
                f"The {self.name} tool failed unexpectedly: {type(exc).__name__}: {exc}"
            )
        if isinstance(result, ToolResult):
            return result
        return ToolResult(content=str(result))


def tool_from(obj: ToolLike) -> Tool:
    """Wrap a metadata-bearing callable object as a :class:`Tool`."""

    return Tool(
        name=obj.name,
        description=obj.description,
        parameters=obj.parameters,
        func=obj.__call__,
    )


class ToolRegistry:
    """A tiny registry that converts tool failures into observations."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"A tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def subset(self, names: Sequence[str] | None) -> ToolRegistry:
        if names is None:
            return ToolRegistry(self._tools.values())
        selected = [self._tools[name] for name in names if name in self._tools]
        return ToolRegistry(selected)

    def openai_schemas(self, names: Sequence[str] | None = None) -> list[dict[str, Any]]:
        if names is None:
            selected = self._tools.values()
        else:
            selected = (self._tools[name] for name in names if name in self._tools)
        return [tool.openai_schema() for tool in selected]

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult.failure(
                f"Unknown tool {name!r}. Available tools: {', '.join(self.names) or '(none)'}."
            )
        if not isinstance(arguments, dict):
            return ToolResult.failure(f"Arguments for {name} must be a JSON object.")
        return tool.run(arguments)


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{field} must be a non-empty string")
    return value.strip()


def coerce_int(
    value: Any,
    *,
    field: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Coerce a tool argument to an int and clamp it to a safe range."""

    if value is None:
        return default
    if isinstance(value, bool):
        raise ToolError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{field} must be an integer") from exc
    return max(minimum, min(maximum, parsed))


def clip(text: str, limit: int) -> str:
    """Clip text to ``limit`` characters, adding a short marker."""

    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()} … [truncated]"


def page_slice(text: str, offset: int, limit: int) -> tuple[str, str]:
    """Return a page of ``text`` plus a human-readable range note."""

    if offset < 0:
        raise ToolError("offset must be zero or greater")
    total = len(text)
    if total == 0:
        return "", "The extracted text is empty."
    if offset >= total:
        return "", (
            f"Offset {offset} is beyond the end of the text ({total} characters). "
            "Use offset=0 to start over."
        )
    end = min(offset + limit, total)
    note = f"Characters {offset}-{end} of {total}."
    if end < total:
        note += f" Call the tool again with offset={end} to read the next part."
    else:
        note += " This is the end of the text."
    return text[offset:end], note


def link_context(anchor: Any, limit: int = 260) -> str:
    """Build a short context snippet around an ``<a>`` element."""

    parent = anchor.find_parent(["p", "li", "blockquote", "td", "dd"])
    container = parent if parent is not None else anchor
    text = normalize_whitespace(container.get_text(" ", strip=True))
    anchor_text = normalize_whitespace(anchor.get_text(" ", strip=True))
    if not text:
        return ""
    if anchor_text:
        lowered = text.lower()
        index = lowered.find(anchor_text.lower())
        if index >= 0:
            start = max(0, index - 90)
            end = min(len(text), index + len(anchor_text) + 170)
            snippet = text[start:end]
            if start > 0:
                snippet = f"… {snippet}"
            if end < len(text):
                snippet = f"{snippet} …"
            return clip(snippet, limit)
    return clip(text, limit)


def format_json(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, indent=2, default=str)
