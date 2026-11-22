from __future__ import annotations

import json

import httpx
import pytest
from openai import OpenAI

from web_search_qa import llm as llm_module
from web_search_qa.llm import LLMError, OpenRouterLLM, get_model_choices


def make_llm(settings, handler) -> OpenRouterLLM:
    client = OpenAI(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return OpenRouterLLM(settings, "test/model", client=client)


def test_complete_preserves_reasoning_details_and_tool_calls(settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "test/model"
        assert body["tools"][0]["function"]["name"] == "web_search"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I will search.",
                            "reasoning": "Need current sources.",
                            "reasoning_details": [
                                {
                                    "type": "reasoning.summary",
                                    "summary": "Need current sources.",
                                }
                            ],
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "current events"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    llm = make_llm(settings, handler)
    response = llm.complete(
        [{"role": "user", "content": "What happened?"}],
        [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert response.content == "I will search."
    assert response.reasoning == "Need current sources."
    assert response.tool_calls[0].arguments == {"query": "current events"}
    assert response.assistant_message["reasoning_details"] == [
        {"type": "reasoning.summary", "summary": "Need current sources."}
    ]
    assert response.assistant_message["tool_calls"][0]["id"] == "call_1"


def test_invalid_tool_arguments_are_reported_not_raised(settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_bad",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": "{not json",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    response = make_llm(settings, handler).complete([{"role": "user", "content": "Test"}])
    assert response.tool_calls[0].parse_error is not None
    assert "not valid JSON" in response.tool_calls[0].parse_error


def test_api_error_body_is_an_llm_error(settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "No credits"}})

    with pytest.raises(LLMError, match="No credits"):
        make_llm(settings, handler).complete([{"role": "user", "content": "Test"}])


def test_model_choices_put_default_first(settings, monkeypatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "_fetch_tool_models",
        lambda base_url, timeout: ("z/model", "a/model"),
    )
    choices = get_model_choices(settings)
    assert choices[0] == settings.default_model
    assert "z/model" in choices
