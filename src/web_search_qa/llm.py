"""OpenRouter chat-completions wrapper.

The wrapper deliberately uses the raw response body so provider-specific fields
such as ``reasoning_details`` can be replayed unchanged on the next tool step,
as required by OpenRouter's reasoning documentation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import httpx
from openai import OpenAI

from web_search_qa.config import Settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the model call fails or returns an unusable response."""


@dataclass(frozen=True, slots=True)
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None


class OpenRouterLLM:
    """A thin synchronous client for OpenRouter's Chat Completions endpoint."""

    def __init__(
        self,
        settings: Settings,
        model: str | None = None,
        *,
        client: OpenAI | None = None,
    ) -> None:
        self.settings = settings
        self.model = (model or settings.default_model).strip()
        if not self.model:
            raise LLMError("A model name is required.")
        if client is not None:
            self._client = client
        else:
            self._client = OpenAI(
                api_key=settings.require_openrouter_key(),
                base_url=settings.openrouter_base_url,
                timeout=settings.request_timeout,
                max_retries=1,
                default_headers={"X-Title": "web-search-qa"},
            )

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools

        try:
            raw = self._client.chat.completions.with_raw_response.create(**payload)
            body = raw.text
        except Exception as exc:
            raise LLMError(f"OpenRouter request failed: {exc}") from exc

        try:
            data = json.loads(body)
        except ValueError as exc:
            raise LLMError("OpenRouter returned a response that was not valid JSON.") from exc
        if not isinstance(data, dict):
            raise LLMError("OpenRouter returned an unexpected response shape.")
        if "error" in data:
            error = data["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise LLMError(f"OpenRouter API error: {message}")

        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise LLMError("OpenRouter returned no completion choices.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise LLMError("OpenRouter returned a completion without a message.")

        content = _content_text(message.get("content"))
        tool_calls = _parse_tool_calls(message.get("tool_calls"))
        reasoning = _extract_reasoning(message)
        assistant_message = dict(message)
        assistant_message["role"] = "assistant"
        assistant_message["content"] = content
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            assistant_message=assistant_message,
            reasoning=reasoning,
        )


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def _parse_tool_calls(raw_calls: Any) -> list[LLMToolCall]:
    if not isinstance(raw_calls, list):
        return []
    calls: list[LLMToolCall] = []
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            function = {}
        name = str(function.get("name") or "").strip()
        call_id = str(raw_call.get("id") or f"call_{index}")
        raw_arguments = function.get("arguments")
        parse_error: str | None = None
        arguments: dict[str, Any] = {}
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
            raw_text = json.dumps(raw_arguments, ensure_ascii=False)
        elif isinstance(raw_arguments, str):
            raw_text = raw_arguments
            if raw_arguments.strip():
                try:
                    parsed = json.loads(raw_arguments)
                except ValueError as exc:
                    parse_error = f"Tool arguments were not valid JSON: {exc}"
                else:
                    if isinstance(parsed, dict):
                        arguments = parsed
                    else:
                        parse_error = "Tool arguments must be a JSON object."
        else:
            raw_text = "{}"
        if not name:
            parse_error = parse_error or "The tool call did not include a function name."
        calls.append(
            LLMToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                raw_arguments=raw_text,
                parse_error=parse_error,
            )
        )
    return calls


def _extract_reasoning(message: dict[str, Any]) -> str | None:
    parts: list[str] = []
    raw_reasoning = message.get("reasoning")
    if isinstance(raw_reasoning, str) and raw_reasoning.strip():
        parts.append(raw_reasoning.strip())
    details = message.get("reasoning_details")
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            for key in ("summary", "text", "content", "reasoning"):
                value = detail.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                    break
    if not parts:
        return None
    return "\n\n".join(dict.fromkeys(parts))


_FALLBACK_MODELS = [
    "deepseek/deepseek-v4.1-flash",
    "deepseek/deepseek-v4.1-flash-1017",
]


@lru_cache(maxsize=4)
def _fetch_tool_models(base_url: str, timeout: float) -> tuple[str, ...]:
    response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        raise ValueError("Unexpected OpenRouter models response")
    ids: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id") or "").strip()
        if not model_id or model_id.startswith("~") or model_id.endswith(":batch"):
            continue
        supported = model.get("supported_parameters")
        if not isinstance(supported, list) or "tools" not in supported:
            continue
        ids.append(model_id)
    return tuple(sorted(dict.fromkeys(ids)))


def get_model_choices(settings: Settings) -> list[str]:
    """Return tool-capable OpenRouter model IDs, with an offline fallback."""

    default = settings.default_model
    try:
        ids = list(
            _fetch_tool_models(settings.openrouter_base_url, min(settings.request_timeout, 8.0))
        )
    except Exception as exc:
        logger.warning("Could not load the OpenRouter model list: %s", exc)
        ids = []
    if not ids:
        ids = list(_FALLBACK_MODELS)
    if default not in ids:
        ids.insert(0, default)
    else:
        ids.remove(default)
        ids.insert(0, default)
    return ids
