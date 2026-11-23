"""Gradio chat interface for the research agent."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

import gradio as gr

from web_search_qa.agent import (
    Agent,
    FinalAnswerEvent,
    ObservationEvent,
    ThoughtEvent,
    ToolCallEvent,
)
from web_search_qa.config import ConfigurationError, Settings, get_settings
from web_search_qa.llm import LLMError, OpenRouterLLM, get_model_choices
from web_search_qa.tools import build_registry

logger = logging.getLogger(__name__)

_UI_OBSERVATION_CHARS = 6_000

# Gradio/KaTeX delimiters. Bare $...$ is intentionally not a delimiter here:
# tool observations contain prices and dollar signs, so only the final answer
# has safe bare-dollar math normalized to \(...\) by _normalize_math().
_LATEX_DELIMITERS: list[dict[str, str | bool]] = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]

# Match a single-dollar inline formula without touching $$...$$, escaped \$,
# whitespace-padded pairs, or currency such as "$5 and $10".
_INLINE_DOLLAR_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(?!\s)([^\n$]+?)(?<!\s)(?<!\\)\$(?!\$)")


def create_ui(settings: Settings | None = None) -> gr.Blocks:
    settings = settings or get_settings()
    registry = build_registry(settings)
    model_choices = get_model_choices(settings)
    tool_names = registry.names
    tool_help = "\n".join(f"- `{tool.name}` — {tool.description}" for tool in registry.tools())

    with gr.Blocks(title="Web Search QA", fill_height=True) as demo:
        gr.Markdown(
            "# Web Search QA\n"
            "A local research agent. It searches, reads pages, follows links, and "
            "answers with citations. Tool output is treated as untrusted data."
        )
        if not settings.has_openrouter_key:
            gr.Markdown(
                "⚠️ **No OpenRouter API key found.** Add `OPENROUTER_API_KEY=...` "
                "to `.env` and restart the app. The interface will start, but "
                "research requests cannot run yet.",
                elem_classes=["warning"],
            )

        chatbot = gr.Chatbot(
            label="Research",
            show_label=False,
            height=560,
            layout="panel",
            group_consecutive_messages=False,
            autoscroll=True,
            latex_delimiters=_LATEX_DELIMITERS,
            placeholder=(
                "Ask a research question. The agent will show its thoughts, tool "
                "calls, and observations here."
            ),
        )

        with gr.Row():
            message = gr.Textbox(
                show_label=False,
                placeholder="e.g. What did the 1969 Apollo 11 mission accomplish, and what did later missions add?",
                lines=2,
                max_lines=6,
                autofocus=True,
                scale=8,
            )
            with gr.Column(scale=1, min_width=120):
                ask = gr.Button("Research", variant="primary")
                stop = gr.Button("Stop", variant="stop")
                clear = gr.Button("Clear")

        conversation = gr.State([])

        with gr.Accordion("Settings", open=False):
            model = gr.Dropdown(
                choices=model_choices,
                value=settings.default_model,
                label="Model",
                info=(
                    "Tool-capable OpenRouter models are listed when the model "
                    "catalogue is reachable. You can also type any model ID."
                ),
                allow_custom_value=True,
                filterable=True,
            )
            tools = gr.CheckboxGroup(
                choices=tool_names,
                value=tool_names,
                label="Enabled tools",
                info="Only the selected tools are offered to the model.",
            )
            gr.Markdown(tool_help)
            max_steps = gr.Slider(
                minimum=1,
                maximum=30,
                value=settings.default_max_steps,
                step=1,
                label="Maximum tool steps",
                info=(
                    "At the limit, the model is forced to answer from the evidence "
                    "already gathered."
                ),
            )

        gr.Markdown(
            f"Search provider: **{settings.search_provider}** · "
            f"Wikipedia: **{settings.wikipedia_language}** · "
            "The server binds to `127.0.0.1` only."
        )

        def respond(
            message_text: str,
            chat_history: list[Any] | None,
            conversation_history: list[dict[str, str]] | None,
            model_name: str,
            enabled_tools: list[str] | None,
            steps: float | int,
        ) -> Iterator[tuple[list[Any], list[dict[str, str]], str]]:
            text = (message_text or "").strip()
            chat = list(chat_history or [])
            convo = list(conversation_history or [])
            if not text:
                yield chat, convo, ""
                return

            chat.append(gr.ChatMessage(role="user", content=text))
            yield chat, convo, ""

            if not enabled_tools:
                chat.append(
                    _error_message("Select at least one tool in Settings before researching.")
                )
                yield chat, convo, ""
                return
            if not settings.has_openrouter_key:
                chat.append(
                    _error_message(
                        "OPENROUTER_API_KEY is not set. Copy `.env.example` to "
                        "`.env`, add your key, and restart the app."
                    )
                )
                yield chat, convo, ""
                return

            llm: OpenRouterLLM | None = None
            answer: str | None = None
            try:
                llm = OpenRouterLLM(settings, model_name)
                agent = Agent(llm, registry, settings)
                for event in agent.run(
                    text,
                    history=convo,
                    tool_names=enabled_tools,
                    max_steps=int(steps),
                ):
                    if isinstance(event, ThoughtEvent):
                        chat.append(_trace_message("💭 Thought", event.text, status="done"))
                    elif isinstance(event, ToolCallEvent):
                        args = json.dumps(
                            event.arguments, indent=2, ensure_ascii=False, default=str
                        )
                        if event.parse_error:
                            args = f"{args}\n\n{event.parse_error}"
                        chat.append(
                            _trace_message(
                                f"🔧 Tool call: {event.name}",
                                f"```json\n{args}\n```",
                                status="done",
                            )
                        )
                    elif isinstance(event, ObservationEvent):
                        icon = "📄" if event.ok else "⚠️"
                        content = _clip_ui(event.content, _UI_OBSERVATION_CHARS)
                        chat.append(
                            _trace_message(
                                f"{icon} {event.name} result",
                                content,
                                status="done",
                            )
                        )
                    elif isinstance(event, FinalAnswerEvent):
                        if event.forced:
                            chat.append(
                                _trace_message(
                                    "⏱️ Step limit reached",
                                    "The model was asked to answer from the evidence "
                                    "already gathered.",
                                    status="done",
                                )
                            )
                        answer = _normalize_math(event.text)
                        chat.append(gr.ChatMessage(role="assistant", content=answer))
                        convo.extend(
                            [
                                {"role": "user", "content": text},
                                {"role": "assistant", "content": answer},
                            ]
                        )
                        convo = _cap_history(convo, settings.max_history_turns)
                    yield chat, convo, ""

                if answer is None:
                    chat.append(
                        _error_message(
                            "The agent stopped without producing a final answer. "
                            "Try again or increase the step limit."
                        )
                    )
                    yield chat, convo, ""
            except (LLMError, ConfigurationError) as exc:
                chat.append(_error_message(str(exc)))
                yield chat, convo, ""
            except Exception as exc:
                logger.exception("Unexpected error while running the agent")
                chat.append(_error_message(f"Unexpected error: {type(exc).__name__}: {exc}"))
                yield chat, convo, ""
            finally:
                if llm is not None:
                    llm.close()

        def clear_chat() -> tuple[list[Any], list[dict[str, str]], str]:
            return [], [], ""

        outputs = [chatbot, conversation, message]
        inputs = [message, chatbot, conversation, model, tools, max_steps]
        ask_event = ask.click(
            respond,
            inputs=inputs,
            outputs=outputs,
            concurrency_limit=1,
            show_progress="minimal",
        )
        submit_event = message.submit(
            respond,
            inputs=inputs,
            outputs=outputs,
            concurrency_limit=1,
            show_progress="minimal",
        )
        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[ask_event, submit_event],
        )
        clear.click(clear_chat, inputs=None, outputs=outputs)

    return demo


def _normalize_math(text: str) -> str:
    """Convert safe bare ``$...$`` formulas to Gradio's ``\\(...\\)`` delimiters.

    ``$$...$$``, ``\\[...\\]``, and ``\\(...\\)`` are already understood by the
    Chatbot and are left untouched. Escaped dollars and currency-like pairs are
    also left untouched so tool observations and prices do not render as math.
    """

    return _INLINE_DOLLAR_MATH_RE.sub(lambda match: f"\\({match.group(1)}\\)", text)


def _trace_message(title: str, content: str, *, status: str = "done") -> gr.ChatMessage:
    return gr.ChatMessage(
        role="assistant",
        content=content.strip() or "(empty)",
        metadata={"title": title, "status": status},
    )


def _error_message(text: str) -> gr.ChatMessage:
    return gr.ChatMessage(role="assistant", content=f"**Error:** {text}")


def _clip_ui(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return (
        f"{text[:limit].rstrip()}\n\n"
        f"… [UI preview truncated at {limit:,} characters; the model saw the "
        "fuller observation.]"
    )


def _cap_history(history: list[dict[str, str]], max_turns: int) -> list[dict[str, str]]:
    if max_turns <= 0:
        return []
    return history[-max_turns * 2 :]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    if not settings.has_openrouter_key:
        logger.warning(
            "OPENROUTER_API_KEY is not set; the UI will start but research requests cannot run."
        )
    demo = create_ui(settings)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
