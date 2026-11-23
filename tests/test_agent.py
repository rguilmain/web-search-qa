from __future__ import annotations

import copy
from typing import Any

import pytest

from web_search_qa.agent import (
    Agent,
    FinalAnswerEvent,
    ObservationEvent,
    ThoughtEvent,
    ToolCallEvent,
)
from web_search_qa.llm import LLMResponse, LLMToolCall
from web_search_qa.tools.base import Tool, ToolRegistry, ToolResult


class FakeLLM:
    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append((copy.deepcopy(messages), tools))
        if not self.responses:
            raise AssertionError("FakeLLM ran out of responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_response(
    content: str = "",
    *,
    tool_calls: list[LLMToolCall] | None = None,
    reasoning: str | None = None,
) -> LLMResponse:
    raw_calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.raw_arguments},
        }
        for call in tool_calls or []
    ]
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if raw_calls:
        message["tool_calls"] = raw_calls
    if reasoning:
        message["reasoning"] = reasoning
        message["reasoning_details"] = [{"type": "reasoning.summary", "summary": reasoning}]
    return LLMResponse(
        content=content,
        tool_calls=list(tool_calls or []),
        assistant_message=message,
        reasoning=reasoning,
    )


def make_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> LLMToolCall:
    import json

    raw = json.dumps(arguments)
    return LLMToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments=raw,
    )


def echo_registry(calls: list[str]) -> ToolRegistry:
    def echo(text: str) -> ToolResult:
        calls.append(text)
        return ToolResult(f"echo:{text}")

    return ToolRegistry(
        [
            Tool(
                name="echo",
                description="Echo text.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                func=echo,
            )
        ]
    )


def test_react_loop_preserves_reasoning_and_wraps_observations(settings) -> None:
    calls: list[str] = []
    llm = FakeLLM(
        [
            make_response(
                "I will echo the text.",
                tool_calls=[make_call("echo", {"text": "hello"})],
                reasoning="The user asked for an echo.",
            ),
            make_response("The answer is hello. [source](https://example.com)"),
        ]
    )
    agent = Agent(llm, echo_registry(calls), settings)

    events = list(agent.run("Say hello", max_steps=3))

    assert [type(event) for event in events] == [
        ThoughtEvent,
        ThoughtEvent,
        ToolCallEvent,
        ObservationEvent,
        FinalAnswerEvent,
    ]
    assert calls == ["hello"]
    assert events[-1].text == "The answer is hello. [source](https://example.com)"

    second_messages = llm.calls[1][0]
    assistant = next(m for m in second_messages if m["role"] == "assistant")
    assert assistant["reasoning_details"] == [
        {"type": "reasoning.summary", "summary": "The user asked for an echo."}
    ]
    tool_message = next(m for m in second_messages if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_1"
    assert 'trust="untrusted"' in tool_message["content"]
    assert "echo:hello" in tool_message["content"]


def test_max_steps_forces_final_answer_without_tools(settings) -> None:
    calls: list[str] = []
    llm = FakeLLM(
        [
            make_response(tool_calls=[make_call("echo", {"text": "one"}, "call_1")]),
            make_response(tool_calls=[make_call("echo", {"text": "two"}, "call_2")]),
            make_response("Forced final answer."),
        ]
    )
    agent = Agent(llm, echo_registry(calls), settings)

    events = list(agent.run("Keep going", max_steps=2))

    assert len(llm.calls) == 3
    assert llm.calls[2][1] is None
    forced_messages = llm.calls[2][0]
    assert "Do not call any more" in forced_messages[-1]["content"]
    assert any(
        isinstance(event, ThoughtEvent) and "Step limit reached" in event.text for event in events
    )
    final = events[-1]
    assert isinstance(final, FinalAnswerEvent)
    assert final.forced is True
    assert final.text == "Forced final answer."


def test_tool_errors_become_observations(settings) -> None:
    def explode() -> str:
        raise RuntimeError("boom")

    registry = ToolRegistry(
        [
            Tool(
                name="explode",
                description="Always fails.",
                parameters={"type": "object", "properties": {}},
                func=explode,
            )
        ]
    )
    llm = FakeLLM(
        [
            make_response(tool_calls=[make_call("explode", {})]),
            make_response("I could not use that tool."),
        ]
    )
    events = list(Agent(llm, registry, settings).run("Test failure", max_steps=3))

    observation = next(event for event in events if isinstance(event, ObservationEvent))
    assert observation.ok is False
    assert "boom" in observation.content
    assert isinstance(events[-1], FinalAnswerEvent)


def test_duplicate_tool_calls_are_nudged(settings) -> None:
    calls: list[str] = []
    llm = FakeLLM(
        [
            make_response(tool_calls=[make_call("echo", {"text": "same"}, "call_1")]),
            make_response(tool_calls=[make_call("echo", {"text": "same"}, "call_2")]),
            make_response("Done."),
        ]
    )
    events = list(Agent(llm, echo_registry(calls), settings).run("Repeat", max_steps=4))

    observations = [event for event in events if isinstance(event, ObservationEvent)]
    assert len(observations) == 2
    assert observations[0].ok is True
    assert observations[1].ok is False
    assert "already called" in observations[1].content
    assert calls == ["same"]


def test_repeated_fetch_url_is_blocked_per_run(settings) -> None:
    fetched: list[tuple[str, int]] = []

    def fetch(url: str, offset: int = 0) -> ToolResult:
        fetched.append((url, offset))
        return ToolResult("page body")

    registry = ToolRegistry(
        [
            Tool(
                name="fetch_url",
                description="Fetch.",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "offset": {"type": "integer"},
                    },
                    "required": ["url"],
                },
                func=fetch,
            )
        ]
    )
    llm = FakeLLM(
        [
            make_response(
                tool_calls=[make_call("fetch_url", {"url": "https://example.com/page#a"}, "call_1")]
            ),
            make_response(
                tool_calls=[make_call("fetch_url", {"url": "https://example.com/page#b"}, "call_2")]
            ),
            make_response("Done."),
        ]
    )
    events = list(Agent(llm, registry, settings).run("Fetch twice", max_steps=4))

    observations = [event for event in events if isinstance(event, ObservationEvent)]
    assert len(fetched) == 1
    assert observations[1].ok is False
    assert "already fetched" in observations[1].content


def test_invalid_tool_arguments_become_observations(settings) -> None:
    calls: list[str] = []
    bad_call = LLMToolCall(
        id="call_bad",
        name="echo",
        arguments={},
        raw_arguments="{not json",
        parse_error="Tool arguments were not valid JSON",
    )
    llm = FakeLLM(
        [
            make_response(tool_calls=[bad_call]),
            make_response("I fixed the call."),
        ]
    )
    events = list(Agent(llm, echo_registry(calls), settings).run("Bad call", max_steps=3))

    observation = next(event for event in events if isinstance(event, ObservationEvent))
    assert observation.ok is False
    assert "could not be used" in observation.content
    assert calls == []


def test_history_is_capped(settings) -> None:
    settings = settings.model_copy(update={"max_history_turns": 2})
    llm = FakeLLM([make_response("Answer.")])
    agent = Agent(llm, ToolRegistry(), settings)
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]

    list(agent.run("new question", history=history))

    messages = llm.calls[0][0]
    assert [message["content"] for message in messages] == [
        messages[0]["content"],
        "u2",
        "a2",
        "u3",
        "a3",
        "new question",
    ]


def test_empty_question_is_rejected(settings) -> None:
    agent = Agent(FakeLLM([]), ToolRegistry(), settings)
    with pytest.raises(ValueError, match="question"):
        list(agent.run("   "))


def test_system_prompt_includes_math_rules(settings) -> None:
    agent = Agent(FakeLLM([]), ToolRegistry(), settings)
    assert r"\( ... \)" in agent.system_prompt
    assert "$$ ... $$" in agent.system_prompt
    assert "backticks" in agent.system_prompt
