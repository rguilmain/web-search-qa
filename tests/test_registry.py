from __future__ import annotations

from typing import Any, ClassVar

import pytest

from web_search_qa.tools.base import Tool, ToolError, ToolRegistry, ToolResult, tool_from


def test_registry_schemas_and_subset() -> None:
    tool = Tool(
        name="echo",
        description="Echo text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        func=lambda text: text,
    )
    registry = ToolRegistry([tool])
    schema = registry.openai_schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["parameters"]["required"] == ["text"]
    assert registry.subset(["echo"]).names == ["echo"]
    assert registry.subset(["missing"]).names == []
    assert registry.subset(None).names == ["echo"]


def test_registry_converts_tool_errors_and_crashes() -> None:
    def expected_failure() -> str:
        raise ToolError("expected failure")

    def crash() -> str:
        raise RuntimeError("unexpected crash")

    registry = ToolRegistry(
        [
            Tool("expected", "Expected failure.", {"type": "object"}, expected_failure),
            Tool("crash", "Unexpected crash.", {"type": "object"}, crash),
        ]
    )

    expected = registry.run("expected", {})
    assert expected.ok is False
    assert "expected failure" in expected.content

    crashed = registry.run("crash", {})
    assert crashed.ok is False
    assert "RuntimeError" in crashed.content
    assert "unexpected crash" in crashed.content

    unknown = registry.run("nope", {})
    assert unknown.ok is False
    assert "Unknown tool" in unknown.content


def test_registry_rejects_duplicate_names_and_bad_arguments() -> None:
    tool = Tool("echo", "Echo.", {"type": "object"}, lambda text: text)
    registry = ToolRegistry([tool])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)

    bad_args = registry.run("echo", {"missing": "x"})
    assert bad_args.ok is False
    assert "Invalid arguments" in bad_args.content


def test_tool_result_passthrough_and_tool_from() -> None:
    class CallableTool:
        name = "wrapped"
        description = "Wrapped callable."
        parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

        def __call__(self) -> ToolResult:
            return ToolResult("wrapped result")

    tool = tool_from(CallableTool())
    assert tool.run({}) == ToolResult("wrapped result")
