from __future__ import annotations

import socket
from contextlib import closing

import gradio as gr

from web_search_qa import ui
from web_search_qa.llm import LLMResponse, LLMToolCall
from web_search_qa.tools.base import Tool, ToolRegistry, ToolResult


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def test_trace_parent_and_child_metadata() -> None:
    parent = ui._new_trace_parent("🔬 Research trace", "trace-123")
    assert isinstance(parent, gr.ChatMessage)
    assert parent.metadata == {
        "title": "🔬 Research trace",
        "id": "trace-123",
        "status": "pending",
    }

    child = ui._trace_child("💭 Thought", "I should search.", "trace-123")
    assert child.role == "assistant"
    assert child.content == "I should search."
    assert child.metadata == {
        "title": "💭 Thought",
        "parent_id": "trace-123",
        "status": "done",
    }


def test_trace_events_are_nested_under_one_parent(settings, monkeypatch) -> None:
    class FakeLLM:
        def __init__(self, settings, model=None, **kwargs) -> None:
            self.called = False

        def complete(self, messages, tools=None) -> LLMResponse:
            if not self.called:
                self.called = True
                call = LLMToolCall("call_1", "noop", {}, "{}")
                message = {
                    "role": "assistant",
                    "content": "I will check.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "noop", "arguments": "{}"},
                        }
                    ],
                }
                return LLMResponse("I will check.", [call], message, None)
            return LLMResponse(
                "Final answer.",
                [],
                {"role": "assistant", "content": "Final answer."},
                None,
            )

        def close(self) -> None:
            pass

    registry = ToolRegistry(
        [
            Tool(
                "noop",
                "No-op tool.",
                {"type": "object", "properties": {}},
                lambda: ToolResult("ok"),
            )
        ]
    )
    monkeypatch.setattr(ui, "OpenRouterLLM", FakeLLM)
    monkeypatch.setattr(ui, "build_registry", lambda settings: registry)
    monkeypatch.setattr(ui, "get_model_choices", lambda settings: ["fake/model"])

    demo = ui.create_ui(settings)
    demo.queue(default_concurrency_limit=1)
    _app, url, _share_url = demo.launch(
        server_name="127.0.0.1",
        server_port=_free_port(),
        share=False,
        inbrowser=False,
        prevent_thread_lock=True,
    )
    try:
        from gradio_client import Client

        chat, _textbox = Client(url, verbose=False).predict(
            "Show me the trace", api_name="/respond"
        )
    finally:
        demo.close()

    # user, parent trace, thought, tool call, observation, final answer
    assert len(chat) == 6
    assert chat[0]["role"] == "user"

    parent = chat[1]
    assert parent["role"] == "assistant"
    assert "Research trace" in parent["metadata"]["title"]
    assert parent["metadata"]["status"] == "done"
    assert parent["metadata"]["duration"] >= 0
    trace_id = parent["metadata"]["id"]

    for child in chat[2:5]:
        assert child["role"] == "assistant"
        assert child["metadata"]["parent_id"] == trace_id
        assert child["metadata"]["status"] == "done"

    assert "parent_id" not in (chat[-1].get("metadata") or {})
    assert chat[-1]["role"] == "assistant"
    assert "Final answer." in _message_text(chat[-1])
